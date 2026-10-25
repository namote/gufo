#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <initializer_list>
#include <utility>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/snapshot_transfer.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace gufo::hip {
namespace {

constexpr std::size_t kMaxTargetLayerTaps = 5;
constexpr std::array<std::uint8_t, 8> kCompactSnapshotMagic = {
    'G', 'Q', 'K', 'V', 'S', 'N', 'P', '1'};
constexpr std::uint32_t kCompactSnapshotVersion = 1;
constexpr std::size_t kCompactSnapshotHeaderBytes = 80;

std::size_t CheckedMultiply(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("Qwen GPU allocation size overflows");
  }
  return left * right;
}

std::size_t CheckedSum(std::size_t left, std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error("Qwen GPU allocation total overflows");
  }
  return left + right;
}

void CheckedAdd(std::size_t value, std::size_t* total) {
  *total = CheckedSum(*total, value);
}

void AddAllocation(std::size_t elements, std::size_t element_bytes,
                   std::size_t* total) {
  CheckedAdd(CheckedMultiply(elements, element_bytes), total);
}

void ThrowOnHipError(hipError_t error, std::string_view operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

template<typename T>
  requires(std::is_unsigned_v<T>)
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     T value) {
  if (offset > destination.size() || sizeof(T) > destination.size() - offset) {
    throw std::length_error("Qwen compact snapshot header is truncated");
  }
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    destination[offset + byte] =
        static_cast<std::uint8_t>(value >> (byte * 8U));
  }
}

template<typename T>
  requires(std::is_unsigned_v<T>)
T GetLittleEndian(std::span<const std::uint8_t> source, std::size_t offset) {
  if (offset > source.size() || sizeof(T) > source.size() - offset) {
    throw std::invalid_argument("Qwen compact snapshot header is truncated");
  }
  T value = 0;
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    value |= static_cast<T>(source[offset + byte]) << (byte * 8U);
  }
  return value;
}

std::size_t SizeFromU64(std::uint64_t value) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("Qwen compact snapshot size overflows");
  }
  return static_cast<std::size_t>(value);
}

struct CompactSnapshotLayout {
  std::uint32_t attention_layers{0};
  std::uint32_t kv_width{0};
  std::uint32_t max_context{0};
  std::uint32_t valid_context{0};
  QwenKvCacheStorage kv_storage{QwenKvCacheStorage::kFp32};
  QwenRecurrentStateStorage recurrent_state_storage{
      QwenRecurrentStateStorage::kFp32};
  std::size_t live_kv_elements_per_plane{0};
  std::size_t live_kv_bytes_per_plane{0};
  std::size_t conv_elements{0};
  std::size_t conv_bytes{0};
  std::size_t deltanet_elements{0};
  std::size_t deltanet_bytes{0};
  std::size_t k_offset{kCompactSnapshotHeaderBytes};
  std::size_t v_offset{0};
  std::size_t conv_offset{0};
  std::size_t deltanet_offset{0};
  std::size_t total_bytes{0};
  std::size_t rope_offset{0};
  // Paged form: only the rows past the last full KV block are inline; the
  // full blocks are stored separately, block by block.
  bool paged{false};
  models::qwen::vision::RopeLayout rope;
};

CompactSnapshotLayout MakeCompactSnapshotLayout(
    std::uint32_t attention_layers, std::uint32_t kv_width,
    std::uint32_t max_context, std::uint32_t valid_context,
    QwenKvCacheStorage kv_storage,
    QwenRecurrentStateStorage recurrent_state_storage,
    std::size_t conv_elements, std::size_t deltanet_elements,
    std::size_t image_count = 0, bool paged = false) {
  CompactSnapshotLayout layout{
      .attention_layers = attention_layers,
      .kv_width = kv_width,
      .max_context = max_context,
      .valid_context = valid_context,
      .kv_storage = kv_storage,
      .recurrent_state_storage = recurrent_state_storage,
      .conv_elements = conv_elements,
      .deltanet_elements = deltanet_elements,
      .paged = paged,
  };
  const std::size_t kv_element_bytes = kv_storage == QwenKvCacheStorage::kFp16
                                           ? sizeof(std::uint16_t)
                                           : sizeof(float);
  const std::uint32_t inline_rows =
      paged ? valid_context % kKvBlockTokens : valid_context;
  layout.live_kv_elements_per_plane = CheckedMultiply(
      CheckedMultiply(attention_layers, inline_rows), kv_width);
  layout.live_kv_bytes_per_plane =
      CheckedMultiply(layout.live_kv_elements_per_plane, kv_element_bytes);
  layout.conv_bytes = CheckedMultiply(conv_elements, sizeof(float));
  layout.deltanet_bytes =
      CheckedMultiply(deltanet_elements,
                      QwenRecurrentStateElementBytes(recurrent_state_storage));
  layout.v_offset = CheckedSum(layout.k_offset, layout.live_kv_bytes_per_plane);
  layout.conv_offset =
      CheckedSum(layout.v_offset, layout.live_kv_bytes_per_plane);
  layout.deltanet_offset = CheckedSum(layout.conv_offset, layout.conv_bytes);
  layout.total_bytes =
      CheckedSum(layout.deltanet_offset, layout.deltanet_bytes);
  layout.rope_offset = layout.total_bytes;
  layout.total_bytes =
      CheckedSum(layout.total_bytes, CheckedMultiply(image_count, 12));
  return layout;
}

// Header fields only; `total_bytes` is the compact payload size the caller
// has, whole or streamed. Image grids follow at layout.rope_offset.
CompactSnapshotLayout ParseCompactSnapshotHeader(
    std::span<const std::uint8_t> header, std::size_t total_bytes) {
  if (header.size() < kCompactSnapshotHeaderBytes ||
      !std::equal(kCompactSnapshotMagic.begin(), kCompactSnapshotMagic.end(),
                  header.begin())) {
    throw std::invalid_argument("Qwen compact snapshot header is invalid");
  }
  // 1/2: every row inline (text/images); 3/4: paged, full blocks apart.
  const auto version = GetLittleEndian<std::uint32_t>(header, 8);
  const auto image_count = GetLittleEndian<std::uint32_t>(header, 72);
  if (version < kCompactSnapshotVersion || version > 4 ||
      image_count > GetLittleEndian<std::uint32_t>(header, 28) ||
      ((version == 1 || version == 3) && image_count != 0) ||
      GetLittleEndian<std::uint32_t>(header, 12) !=
          kCompactSnapshotHeaderBytes ||
      GetLittleEndian<std::uint32_t>(header, 76) != 0) {
    throw std::invalid_argument("Qwen compact snapshot version is invalid");
  }

  const std::uint32_t kv_storage_value =
      GetLittleEndian<std::uint32_t>(header, 32);
  const std::uint32_t recurrent_storage_value =
      GetLittleEndian<std::uint32_t>(header, 36);
  if (kv_storage_value >
          static_cast<std::uint32_t>(QwenKvCacheStorage::kFp16) ||
      recurrent_storage_value >
          static_cast<std::uint32_t>(QwenRecurrentStateStorage::kBf16)) {
    throw std::invalid_argument("Qwen compact snapshot precision is invalid");
  }

  auto layout = MakeCompactSnapshotLayout(
      GetLittleEndian<std::uint32_t>(header, 16),
      GetLittleEndian<std::uint32_t>(header, 20),
      GetLittleEndian<std::uint32_t>(header, 24),
      GetLittleEndian<std::uint32_t>(header, 28),
      static_cast<QwenKvCacheStorage>(kv_storage_value),
      static_cast<QwenRecurrentStateStorage>(recurrent_storage_value),
      SizeFromU64(GetLittleEndian<std::uint64_t>(header, 48)),
      SizeFromU64(GetLittleEndian<std::uint64_t>(header, 56)), image_count,
      version >= 3);
  if (layout.live_kv_elements_per_plane !=
          SizeFromU64(GetLittleEndian<std::uint64_t>(header, 40)) ||
      layout.total_bytes !=
          SizeFromU64(GetLittleEndian<std::uint64_t>(header, 64)) ||
      layout.total_bytes != total_bytes) {
    throw std::invalid_argument("Qwen compact snapshot size is invalid");
  }
  return layout;
}

void ReadImageGrids(std::span<const std::uint8_t> grids,
                    CompactSnapshotLayout* layout) {
  for (std::size_t offset = 0; offset + 12 <= grids.size(); offset += 12) {
    layout->rope.images.push_back(
        {GetLittleEndian<std::uint32_t>(grids, offset),
         GetLittleEndian<std::uint32_t>(grids, offset + 4),
         GetLittleEndian<std::uint32_t>(grids, offset + 8)});
  }
  layout->rope.Validate(layout->max_context);
}

CompactSnapshotLayout ParseCompactSnapshotLayout(
    std::span<const std::uint8_t> payload) {
  auto layout = ParseCompactSnapshotHeader(payload, payload.size());
  ReadImageGrids(payload.subspan(layout.rope_offset), &layout);
  return layout;
}

void CheckCompactCompatibility(const core::ModelConfig& config_,
                               std::uint32_t max_context_,
                               const QwenExecutionPolicy& policy_,
                               const CompactSnapshotLayout& layout,
                               std::uint32_t expected_valid_context) {
  const std::uint32_t attention_layers = config_.FullAttentionLayerCount();
  const std::uint32_t kv_width = config_.num_key_value_heads * config_.head_dim;
  const std::size_t conv_elements_per_layer =
      CheckedMultiply(config_.SsmQkvSize(), config_.ssm_conv_kernel);
  const std::size_t deltanet_elements_per_layer = CheckedMultiply(
      CheckedMultiply(config_.ssm_time_step_rank, config_.ssm_state_size),
      config_.SsmValueSize());
  const std::size_t recurrent_layers = config_.num_layers - attention_layers;
  const std::size_t conv_elements =
      CheckedMultiply(recurrent_layers, conv_elements_per_layer);
  const std::size_t deltanet_elements =
      CheckedMultiply(recurrent_layers, deltanet_elements_per_layer);
  if (layout.valid_context != expected_valid_context ||
      layout.valid_context > max_context_ ||
      layout.attention_layers != attention_layers ||
      layout.kv_width != kv_width || layout.max_context != max_context_ ||
      layout.kv_storage != policy_.kv_cache_storage ||
      layout.recurrent_state_storage != policy_.recurrent_state_storage ||
      layout.conv_elements != conv_elements ||
      layout.deltanet_elements != deltanet_elements) {
    throw std::invalid_argument(
        "Qwen compact snapshot is incompatible with the arena");
  }
}

}  // namespace

models::qwen::vision::RopeLayout QwenGpuSnapshot::ReadRopeLayout(
    std::span<const std::uint8_t> payload) {
  return ParseCompactSnapshotLayout(payload).rope;
}

std::size_t QwenGpuSnapshot::SharedKvBytes() const noexcept {
  return kv_pool_ == nullptr ? 0
                             : kv_blocks_.size() * kKvBlockTokens *
                                   kv_pool_->row_bytes() * kv_pool_->layers() *
                                   2;
}

QwenGpuSnapshot::~QwenGpuSnapshot() {
  if (kv_pool_ != nullptr)
    kv_pool_->Release(kv_blocks_);
  if (d_kv_f32_ != nullptr) {
    (void)hipFree(d_kv_f32_);
  }
  if (d_kv_f16_ != nullptr) {
    (void)hipFree(d_kv_f16_);
  }
  if (d_ssm_conv_ != nullptr) {
    (void)hipFree(d_ssm_conv_);
  }
  if (d_ssm_deltanet_ != nullptr) {
    (void)hipFree(d_ssm_deltanet_);
  }
}

std::size_t QwenGpuSnapshot::CompactPayloadBytes(bool paged) const {
  const std::size_t recurrent_layers = num_layers_ - attention_layers_;
  return MakeCompactSnapshotLayout(
             attention_layers_, kv_width_, max_context_, valid_context_,
             kv_storage_, recurrent_state_storage_,
             CheckedMultiply(recurrent_layers, conv_elements_per_layer_),
             CheckedMultiply(recurrent_layers, deltanet_elements_per_layer_),
             vision_layout_.images.size(), paged)
      .total_bytes;
}

std::array<DeviceRegion, 2> QwenGpuSnapshot::KvBlockRegions(
    std::uint32_t block) const {
  if (kv_pool_ == nullptr || block >= kv_blocks_.size())
    throw std::out_of_range("Qwen snapshot has no such shared KV block");
  std::array<DeviceRegion, 2> regions;
  for (std::uint32_t plane = 0; plane < 2; ++plane) {
    const auto region =
        kv_pool_->BlockRegion(plane, kv_blocks_[block], 0, kKvBlockTokens);
    regions[plane] = {region.data, region.pitch, region.width, region.rows};
  }
  return regions;
}

// The compact form of one session, from wherever its state lives: K and V
// regions packed in compact order, then conv and DeltaNet state. A snapshot
// and a live arena give the same bytes for the same state.
static std::vector<QwenGpuSnapshot::CompactSegment> BuildCompactSegments(
    const CompactSnapshotLayout& layout, std::span<const DeviceRegion> k,
    std::span<const DeviceRegion> v, const void* conv, const void* deltanet,
    const models::qwen::vision::RopeLayout& rope) {
  std::vector<QwenGpuSnapshot::CompactSegment> segments;
  // Only the header has reserved bytes; the payload below covers the rest.
  std::vector<std::uint8_t> header(kCompactSnapshotHeaderBytes, 0);
  std::copy(kCompactSnapshotMagic.begin(), kCompactSnapshotMagic.end(),
            header.begin());
  PutLittleEndian<std::uint32_t>(
      header, 8, (layout.paged ? 3U : 1U) + (rope.images.empty() ? 0U : 1U));
  PutLittleEndian<std::uint32_t>(header, 72, rope.images.size());
  PutLittleEndian<std::uint32_t>(
      header, 12, static_cast<std::uint32_t>(kCompactSnapshotHeaderBytes));
  PutLittleEndian<std::uint32_t>(header, 16, layout.attention_layers);
  PutLittleEndian<std::uint32_t>(header, 20, layout.kv_width);
  PutLittleEndian<std::uint32_t>(header, 24, layout.max_context);
  PutLittleEndian<std::uint32_t>(header, 28, layout.valid_context);
  PutLittleEndian<std::uint32_t>(header, 32,
                                 static_cast<std::uint32_t>(layout.kv_storage));
  PutLittleEndian<std::uint32_t>(
      header, 36, static_cast<std::uint32_t>(layout.recurrent_state_storage));
  PutLittleEndian<std::uint64_t>(
      header, 40,
      static_cast<std::uint64_t>(layout.live_kv_elements_per_plane));
  PutLittleEndian<std::uint64_t>(
      header, 48, static_cast<std::uint64_t>(layout.conv_elements));
  PutLittleEndian<std::uint64_t>(
      header, 56, static_cast<std::uint64_t>(layout.deltanet_elements));
  PutLittleEndian<std::uint64_t>(
      header, 64, static_cast<std::uint64_t>(layout.total_bytes));
  segments.push_back({.host = std::move(header)});
  if (layout.live_kv_bytes_per_plane != 0) {
    const auto packed = [](std::span<const DeviceRegion> regions) {
      std::size_t bytes = 0;
      for (const auto& region : regions)
        bytes += region.PackedBytes();
      return bytes;
    };
    if (packed(k) != layout.live_kv_bytes_per_plane ||
        packed(v) != layout.live_kv_bytes_per_plane)
      throw std::logic_error("Qwen compact KV regions do not match the layout");
    for (const auto& region : k)
      segments.push_back({.device = region});
    for (const auto& region : v)
      segments.push_back({.device = region});
  }
  if (layout.conv_bytes != 0)
    segments.push_back(
        {.device = {conv, layout.conv_bytes, layout.conv_bytes, 1}});
  if (layout.deltanet_bytes != 0)
    segments.push_back({.device = {deltanet, layout.deltanet_bytes,
                                   layout.deltanet_bytes, 1}});
  rope.Validate(layout.max_context);
  if (!rope.images.empty()) {
    std::vector<std::uint8_t> grids(rope.images.size() * 12);
    for (std::size_t i = 0; i < rope.images.size(); ++i) {
      const auto& grid = rope.images[i];
      PutLittleEndian<std::uint32_t>(grids, i * 12, grid.offset);
      PutLittleEndian<std::uint32_t>(grids, i * 12 + 4, grid.height);
      PutLittleEndian<std::uint32_t>(grids, i * 12 + 8, grid.width);
    }
    segments.push_back({.host = std::move(grids)});
  }
  return segments;
}

std::vector<QwenGpuSnapshot::CompactSegment>
QwenGpuSnapshot::CompactSegments(bool paged) const {
  if (full_attention_interval_ == 0 || attention_layers_ > num_layers_ ||
      conv_elements_ != CheckedMultiply(num_layers_ - attention_layers_,
                                        conv_elements_per_layer_) ||
      deltanet_elements_ != CheckedMultiply(num_layers_ - attention_layers_,
                                            deltanet_elements_per_layer_)) {
    throw std::logic_error("Qwen compact snapshot metadata is invalid");
  }
  const std::size_t recurrent_layers = num_layers_ - attention_layers_;
  const auto layout = MakeCompactSnapshotLayout(
      attention_layers_, kv_width_, max_context_, valid_context_, kv_storage_,
      recurrent_state_storage_,
      CheckedMultiply(recurrent_layers, conv_elements_per_layer_),
      CheckedMultiply(recurrent_layers, deltanet_elements_per_layer_),
      vision_layout_.images.size(), paged);
  if (paged && kv_pool_ == nullptr)
    throw std::logic_error("only a block-sharing Qwen snapshot is paged");
  if (kv_pool_ != nullptr) {
    std::vector<DeviceRegion> k, v;
    if (layout.live_kv_bytes_per_plane != 0)
      for (std::uint32_t plane = 0; plane < 2; ++plane) {
        if (paged) {
          // Only the partial last block; full blocks go apart.
          const auto region = kv_pool_->BlockRegion(
              plane, kv_blocks_[valid_context_ / kKvBlockTokens], 0,
              valid_context_ % kKvBlockTokens);
          (plane == 0 ? k : v)
              .push_back({region.data, region.pitch, region.width,
                          region.rows});
          continue;
        }
        for (const auto& region :
             kv_pool_->PlaneRegions(plane, kv_blocks_, valid_context_))
          (plane == 0 ? k : v)
              .push_back({region.data, region.pitch, region.width,
                          region.rows});
      }
    return BuildCompactSegments(layout, k, v, d_ssm_conv_, d_ssm_deltanet_,
                                vision_layout_);
  }
  DeviceRegion k{}, v{};
  if (layout.live_kv_bytes_per_plane != 0) {
    const std::size_t kv_element_bytes =
        kv_storage_ == QwenKvCacheStorage::kFp16 ? sizeof(std::uint16_t)
                                                 : sizeof(float);
    // Snapshot planes are already packed: one row per attention layer.
    const std::size_t row_bytes = CheckedMultiply(
        CheckedMultiply(valid_context_, kv_width_), kv_element_bytes);
    const void* source =
        kv_storage_ == QwenKvCacheStorage::kFp16 ? d_kv_f16_ : d_kv_f32_;
    if (source == nullptr) {
      throw std::logic_error("Qwen compact snapshot has no KV storage");
    }
    const std::size_t full_plane_bytes =
        CheckedMultiply(kv_elements_per_plane_, kv_element_bytes);
    k = {source, row_bytes, row_bytes, attention_layers_};
    v = {static_cast<const std::uint8_t*>(source) + full_plane_bytes,
         row_bytes, row_bytes, attention_layers_};
  }
  const std::span<const DeviceRegion> k_regions(&k, k.rows != 0 ? 1 : 0);
  const std::span<const DeviceRegion> v_regions(&v, v.rows != 0 ? 1 : 0);
  return BuildCompactSegments(layout, k_regions, v_regions, d_ssm_conv_,
                              d_ssm_deltanet_, vision_layout_);
}

std::size_t QwenGpuSnapshot::SerializeCompact(
    std::span<std::uint8_t> destination, bool paged) const {
  const auto segments = CompactSegments(paged);
  if (destination.size() != CompactPayloadBytes(paged)) {
    throw std::invalid_argument(
        "Qwen compact snapshot destination size is invalid");
  }
  SnapshotTransfer transfer;
  std::size_t offset = 0;
  for (const auto& segment : segments) {
    if (!segment.host.empty()) {
      std::copy(segment.host.begin(), segment.host.end(),
                destination.begin() + offset);
      offset += segment.host.size();
      continue;
    }
    const auto& region = segment.device;
    transfer.Copy2D(destination.data() + offset, region.width, region.data,
                    region.pitch, region.width, region.rows);
    offset += region.PackedBytes();
  }
  if (offset != destination.size())
    throw std::logic_error("Qwen compact snapshot segments are incomplete");
  return destination.size();
}

void QwenGpuSnapshot::StreamCompact(BounceStream& bounce,
                                    const BounceStream::Sink& sink,
                                    bool paged) const {
  for (const auto& segment : CompactSegments(paged)) {
    if (!segment.host.empty())
      sink(segment.host);
    else
      bounce.Out(segment.device, sink);
  }
}

namespace {

struct PassScratchWidths {
  std::size_t hidden, attention, kv, intermediate, projection, ssm_qkv,
      recurrent, bf16_width, weight_elements;
};

PassScratchWidths ScratchWidths(const core::ModelConfig& config) {
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection = 2 * attention;
  const std::size_t ssm_qkv = config.SsmQkvSize();
  const std::size_t projection = std::max(q_projection, ssm_qkv);
  const std::size_t time_step = config.ssm_time_step_rank;
  return {
      .hidden = config.hidden_size,
      .attention = attention,
      .kv = kv,
      .intermediate = config.intermediate_size,
      .projection = projection,
      .ssm_qkv = ssm_qkv,
      .recurrent = std::max<std::size_t>(attention, config.ssm_inner_size),
      .bf16_width = std::max<std::size_t>(
          {config.intermediate_size, config.hidden_size, projection,
           CheckedSum(ssm_qkv, CheckedSum(config.ssm_inner_size,
                                          CheckedMultiply(2, time_step)))}),
      .weight_elements = CheckedMultiply(
          config.hidden_size,
          std::max<std::size_t>({q_projection, kv, attention,
                                 config.intermediate_size, ssm_qkv,
                                 config.ssm_inner_size, time_step})),
  };
}

// The tiled Q8_1 activation layout groups 16 tokens per tile, so size for a
// batch rounded up to a whole tile. The base layout remains 36 bytes per 32
// elements; reserve another float per block for the K-quant activation-sum
// sidecar without changing the Q8 payload or its stride.
std::size_t Q8ActivationBytes(std::size_t rows, std::size_t row_elements) {
  const std::size_t q8_rows = CheckedMultiply((rows + 15) / 16, 16);
  const std::size_t q8_blocks =
      (CheckedMultiply(q8_rows, row_elements) + 31) / 32;
  return CheckedSum(CheckedMultiply(CheckedMultiply(q8_blocks, sizeof(float)),
                                    10),
                    4096);
}

}  // namespace

std::size_t QwenPassScratchSet::Bytes(const core::ModelConfig& config,
                                      std::size_t rows, bool weights) {
  const auto w = ScratchWidths(config);
  std::size_t bytes = 0;
  for (const std::size_t width :
       {w.hidden, w.hidden, w.attention, w.kv, w.kv, w.hidden, w.intermediate,
        w.intermediate, w.intermediate, w.hidden, w.projection, w.ssm_qkv,
        w.recurrent, w.recurrent})
    AddAllocation(CheckedMultiply(rows, width), sizeof(float), &bytes);
  AddAllocation(CheckedMultiply(rows, w.bf16_width), sizeof(hip_bfloat16),
                &bytes);
  CheckedAdd(Q8ActivationBytes(rows, w.bf16_width), &bytes);
  if (weights)
    AddAllocation(w.weight_elements, sizeof(hip_bfloat16), &bytes);
  return bytes;
}

QwenPassScratchSet QwenPassScratchSet::Allocate(const core::ModelConfig& config,
                                                std::size_t rows,
                                                bool weights) {
  const auto w = ScratchWidths(config);
  QwenPassScratchSet set;
  set.rows = rows;
  try {
    const auto floats = [&](float** buffer, std::size_t width) {
      HIP_CHECK(hipMalloc(buffer, CheckedMultiply(CheckedMultiply(rows, width),
                                                  sizeof(float))));
    };
    floats(&set.hidden, w.hidden);
    floats(&set.normed, w.hidden);
    floats(&set.q, w.attention);
    floats(&set.k, w.kv);
    floats(&set.v, w.kv);
    floats(&set.attn_out, w.hidden);
    floats(&set.ffn_gate, w.intermediate);
    floats(&set.ffn_up, w.intermediate);
    floats(&set.ffn_act, w.intermediate);
    floats(&set.ffn_out, w.hidden);
    floats(&set.ssm_qkv, w.projection);
    floats(&set.conv_out, w.ssm_qkv);
    floats(&set.ssm_gate, w.recurrent);
    floats(&set.ssm_out, w.recurrent);
    HIP_CHECK(hipMalloc(&set.bf16, CheckedMultiply(CheckedMultiply(
                                                       rows, w.bf16_width),
                                                   sizeof(hip_bfloat16))));
    HIP_CHECK(hipMalloc(&set.q8_act, Q8ActivationBytes(rows, w.bf16_width)));
    if (weights)
      HIP_CHECK(hipMalloc(&set.weights_bf16,
                          CheckedMultiply(w.weight_elements,
                                          sizeof(hip_bfloat16))));
  } catch (...) {
    set.Free();
    throw;
  }
  return set;
}

void QwenPassScratchSet::Free() noexcept {
  for (float** buffer :
       {&hidden, &normed, &q, &k, &v, &attn_out, &ffn_gate, &ffn_up,
        &ffn_act, &ffn_out, &ssm_qkv, &conv_out, &ssm_gate, &ssm_out}) {
    if (*buffer != nullptr)
      LogCleanupError(hipFree(*buffer));
    *buffer = nullptr;
  }
  for (void** buffer : {&bf16, &q8_act}) {
    if (*buffer != nullptr)
      LogCleanupError(hipFree(*buffer));
    *buffer = nullptr;
  }
  if (weights_bf16 != nullptr)
    LogCleanupError(hipFree(weights_bf16));
  weights_bf16 = nullptr;
  rows = 0;
}

QwenSharedPassScratch::QwenSharedPassScratch(const core::ModelConfig& config,
                                             std::uint32_t max_context)
    : set_(QwenPassScratchSet::Allocate(
          config,
          std::min<std::size_t>(std::max(max_context, 1U), QwenGpuArena::kMaxPromptBatch),
          true)),
      bytes_(QwenPassScratchSet::Bytes(config, set_.rows, true)) {
  if (const auto error = hipEventCreateWithFlags(&released_,
                                                 hipEventDisableTiming);
      error != hipSuccess) {
    set_.Free();
    throw std::runtime_error(std::string("hipEventCreateWithFlags: ") +
                             hipGetErrorString(error));
  }
  // Touch every page now, at load: otherwise the first prompt pass after
  // startup pays for mapping 2 GiB of fresh device memory.
  try {
    const auto w = ScratchWidths(config);
    const std::size_t rows = set_.rows;
    for (const auto& [buffer, width] :
         std::initializer_list<std::pair<float*, std::size_t>>{
             {set_.hidden, w.hidden},       {set_.normed, w.hidden},
             {set_.q, w.attention},         {set_.k, w.kv},
             {set_.v, w.kv},                {set_.attn_out, w.hidden},
             {set_.ffn_gate, w.intermediate}, {set_.ffn_up, w.intermediate},
             {set_.ffn_act, w.intermediate}, {set_.ffn_out, w.hidden},
             {set_.ssm_qkv, w.projection},  {set_.conv_out, w.ssm_qkv},
             {set_.ssm_gate, w.recurrent},  {set_.ssm_out, w.recurrent}})
      HIP_CHECK(hipMemset(buffer, 0, rows * width * sizeof(float)));
    HIP_CHECK(hipMemset(set_.bf16, 0, rows * w.bf16_width * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMemset(set_.q8_act, 0, Q8ActivationBytes(rows, w.bf16_width)));
    HIP_CHECK(hipMemset(set_.weights_bf16, 0,
                        w.weight_elements * sizeof(hip_bfloat16)));
  } catch (...) {
    LogCleanupError(hipEventDestroy(released_));
    set_.Free();
    throw;
  }
}

QwenSharedPassScratch::~QwenSharedPassScratch() {
  if (released_ != nullptr) {
    LogCleanupError(hipEventSynchronize(released_));
    LogCleanupError(hipEventDestroy(released_));
  }
  set_.Free();
}

void QwenGpuArena::PointPassScratch(const QwenPassScratchSet& set) noexcept {
  d_hidden = set.hidden;
  d_normed = set.normed;
  d_q = set.q;
  d_k = set.k;
  d_v = set.v;
  d_attn_out = set.attn_out;
  d_ffn_gate = set.ffn_gate;
  d_ffn_up = set.ffn_up;
  d_ffn_act = set.ffn_act;
  d_ffn_out = set.ffn_out;
  d_ssm_qkv = set.ssm_qkv;
  d_conv_out = set.conv_out;
  d_ssm_gate = set.ssm_gate;
  d_ssm_out = set.ssm_out;
  d_scratch_bf16 = set.bf16;
  d_scratch_q8_act = set.q8_act;
  d_weights_bf16 = set.weights_bf16;
}

bool QwenGpuArena::BeginPassScratch() {
  if (shared_scratch_ == nullptr || scratch_leased_)
    return false;
  if (shared_scratch_->leased_.exchange(true))
    throw std::logic_error(
        "the shared Qwen pass scratch is leased by another session's pass");
  // The previous lessee's pass may still run on its own stream.
  if (const auto error =
          hipStreamWaitEvent(stream, shared_scratch_->released_, 0);
      error != hipSuccess) {
    shared_scratch_->leased_.store(false);
    throw std::runtime_error(std::string("hipStreamWaitEvent: ") +
                             hipGetErrorString(error));
  }
  scratch_leased_ = true;
  PointPassScratch(shared_scratch_->set_);
  return true;
}

void QwenGpuArena::EndPassScratch() noexcept {
  if (!scratch_leased_)
    return;
  LogCleanupError(hipEventRecord(shared_scratch_->released_, stream));
  PointPassScratch(own_scratch_);
  scratch_leased_ = false;
  shared_scratch_->leased_.store(false);
}

QwenGpuMemoryUsage QwenGpuArena::EstimateMemoryUsage(
    const core::ModelConfig& config, std::uint32_t max_context,
    QwenExecutionPolicy policy, KvGrowth kv_growth, bool shared_scratch) {
  const std::size_t context = std::max(max_context, 1U);
  const std::size_t batch = std::min<std::size_t>(context, QwenGpuArena::kMaxPromptBatch);
  const std::size_t hidden = config.hidden_size;
  const std::size_t intermediate = config.intermediate_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      CheckedMultiply(config.num_key_value_heads, config.head_dim);
  const std::size_t q_projection = CheckedMultiply(2, attention);
  const std::size_t ssm_qkv = config.SsmQkvSize();
  const std::size_t recurrent =
      std::max<std::size_t>(attention, config.ssm_inner_size);
  const std::size_t projection = std::max(q_projection, ssm_qkv);
  const std::size_t time_step = config.ssm_time_step_rank;

  QwenGpuMemoryUsage usage;
  auto& scratch = usage.temporary_scratch_bytes;
  CheckedAdd(QwenPassScratchSet::Bytes(
                 config,
                 shared_scratch
                     ? std::min<std::size_t>(batch, kSessionScratchRows)
                     : batch,
                 !shared_scratch),
             &scratch);
  AddAllocation(CheckedMultiply(batch, time_step), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, time_step), sizeof(float), &scratch);
  AddAllocation(
      CheckedMultiply(CheckedMultiply(batch, config.ssm_group_count), 3),
      sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(CheckedMultiply(batch, time_step), 2),
                sizeof(float), &scratch);
  AddAllocation(config.vocab_size, sizeof(float), &scratch);
  AddAllocation(std::max<std::size_t>(batch, 2), sizeof(std::uint32_t),
                &scratch);
  AddAllocation(CheckedMultiply(kMaxTargetLayerTaps, hidden), sizeof(float),
                &scratch);
  AddAllocation(detail::DecodeAttentionScratchElements(
                    config.num_attention_heads, config.head_dim),
                sizeof(float), &scratch);

  auto& state = usage.request_state_bytes;
  // FP16 rows come from a block pool in whole blocks; an on-demand arena's
  // rows are the shared pool's, charged once for all sessions.
  const std::size_t kv_rows =
      !policy.UsesFp16AttentionKv() ? context
      : kv_growth == KvGrowth::kOnDemand
          ? 0
          : static_cast<std::size_t>(KvBlockPool::BlocksFor(context)) *
                kKvBlockTokens;
  const std::size_t total_kv = CheckedMultiply(
      CheckedMultiply(config.FullAttentionLayerCount(), kv), kv_rows);
  AddAllocation(
      CheckedMultiply(total_kv, 2),
      policy.UsesFp16AttentionKv() ? sizeof(std::uint16_t) : sizeof(float),
      &state);

  const std::size_t total_conv = CheckedMultiply(
      CheckedMultiply(config.SsmLayerCount(), ssm_qkv), config.ssm_conv_kernel);
  AddAllocation(total_conv, sizeof(float), &state);
  const std::size_t total_deltanet = CheckedMultiply(
      CheckedMultiply(CheckedMultiply(config.SsmLayerCount(), time_step),
                      config.ssm_state_size),
      config.SsmValueSize());
  AddAllocation(total_deltanet,
                QwenRecurrentStateElementBytes(policy.recurrent_state_storage),
                &state);
  return usage;
}

QwenGpuMemoryUsage QwenGpuArena::GetMemoryUsage() const {
  return EstimateMemoryUsage(config_, max_context_, policy_, kv_growth_,
                             shared_scratch_ != nullptr);
}

std::size_t QwenGpuArena::SnapshotPayloadBytes(
    std::uint32_t valid_context) const {
  if (valid_context > max_context_)
    throw std::length_error("Qwen snapshot exceeds the context length");
  // Shared blocks: the snapshot owns recurrent state only.
  if (kv_growth_ == KvGrowth::kOnDemand)
    return GetMemoryUsage().request_state_bytes;
  const std::size_t kv_rows =
      policy_.UsesFp16AttentionKv()
          ? static_cast<std::size_t>(KvBlockPool::BlocksFor(max_context_)) *
                kKvBlockTokens
          : max_context_;
  const auto unused = CheckedMultiply(
      CheckedMultiply(
          CheckedMultiply(config_.FullAttentionLayerCount(),
                          kv_rows - valid_context),
          CheckedMultiply(config_.num_key_value_heads, config_.head_dim)),
      policy_.UsesFp16AttentionKv() ? 2 * sizeof(std::uint16_t)
                                    : 2 * sizeof(float));
  return GetMemoryUsage().request_state_bytes - unused;
}

std::vector<QwenGpuSnapshot::CompactSegment> QwenGpuArena::LiveCompactSegments(
    std::uint32_t valid_context, const models::qwen::vision::RopeLayout& rope,
    bool paged) {
  if (valid_context > max_context_)
    throw std::length_error("Qwen snapshot exceeds the context length");
  // Queued rollback/replay work must land before the regions are read.
  assert(d_ssm_conv_state == d_ssm_conv_buffers_[live_recurrent_] &&
         d_ssm_deltanet_state == d_ssm_deltanet_buffers_[live_recurrent_]);
  ThrowOnHipError(hipStreamSynchronize(stream),
                  "failed to synchronize Qwen state before streaming it");
  const std::uint32_t attention_layers = config_.FullAttentionLayerCount();
  const std::uint32_t kv_width = config_.num_key_value_heads * config_.head_dim;
  const std::size_t recurrent_layers = config_.num_layers - attention_layers;
  const auto layout = MakeCompactSnapshotLayout(
      attention_layers, kv_width, max_context_, valid_context,
      policy_.kv_cache_storage, policy_.recurrent_state_storage,
      CheckedMultiply(recurrent_layers, CheckedMultiply(config_.SsmQkvSize(),
                                                        config_.ssm_conv_kernel)),
      CheckedMultiply(recurrent_layers,
                      CheckedMultiply(CheckedMultiply(config_.ssm_time_step_rank,
                                                      config_.ssm_state_size),
                                      config_.SsmValueSize())),
      rope.images.size(), paged);
  if (paged && (kv_pool_ == nullptr || kv_growth_ != KvGrowth::kOnDemand))
    throw std::logic_error("only an on-demand Qwen KV arena is paged");
  std::vector<DeviceRegion> k, v;
  if (layout.live_kv_bytes_per_plane != 0) {
    for (std::uint32_t plane = 0; plane < 2; ++plane) {
      if (paged) {
        const auto region = kv_pool_->BlockRegion(
            plane, kv_blocks_.at(valid_context / kKvBlockTokens), 0,
            valid_context % kKvBlockTokens);
        (plane == 0 ? k : v)
            .push_back({region.data, region.pitch, region.width, region.rows});
        continue;
      }
      for (const auto& region : KvPlaneRegions(plane, valid_context))
        (plane == 0 ? k : v)
            .push_back({region.data, region.pitch, region.width, region.rows});
    }
  }
  return BuildCompactSegments(layout, k, v, d_ssm_conv_state,
                              d_ssm_deltanet_state, rope);
}

std::size_t QwenGpuArena::CompactSnapshotPayloadBytes(
    const core::ModelConfig& config, std::uint32_t max_context,
    QwenExecutionPolicy policy, std::uint32_t valid_context, bool paged) {
  if (valid_context > max_context)
    throw std::length_error("Qwen snapshot exceeds the context length");
  // Same element counts as SaveSnapshot.
  const std::uint32_t attention_layers = config.FullAttentionLayerCount();
  const std::size_t recurrent_layers = config.num_layers - attention_layers;
  const std::size_t conv_elements_per_layer =
      CheckedMultiply(config.SsmQkvSize(), config.ssm_conv_kernel);
  const std::size_t deltanet_elements_per_layer = CheckedMultiply(
      CheckedMultiply(config.ssm_time_step_rank, config.ssm_state_size),
      config.SsmValueSize());
  return MakeCompactSnapshotLayout(
             attention_layers,
             config.num_key_value_heads * config.head_dim, max_context,
             valid_context, policy.kv_cache_storage,
             policy.recurrent_state_storage,
             CheckedMultiply(recurrent_layers, conv_elements_per_layer),
             CheckedMultiply(recurrent_layers, deltanet_elements_per_layer),
             0, paged)
      .total_bytes;
}

std::array<DeviceRegion, 2> QwenGpuArena::KvBlockRegions(
    std::uint32_t block) const {
  if (kv_pool_ == nullptr || block >= kv_blocks_.size())
    throw std::out_of_range("Qwen arena has no such KV block");
  std::array<DeviceRegion, 2> regions;
  for (std::uint32_t plane = 0; plane < 2; ++plane) {
    const auto region =
        kv_pool_->BlockRegion(plane, kv_blocks_[block], 0, kKvBlockTokens);
    regions[plane] = {region.data, region.pitch, region.width, region.rows};
  }
  return regions;
}

void QwenGpuArena::RestoreKvBlock(std::uint32_t block, BounceStream& bounce,
                                  const BounceStream::Source& read) {
  if (kv_pool_ == nullptr || block >= kv_blocks_.size())
    throw std::out_of_range("Qwen arena has no such KV block");
  for (std::uint32_t plane = 0; plane < 2; ++plane)
    bounce.In(kv_pool_->BlockRegion(plane, kv_blocks_[block], 0, kKvBlockTokens),
              read);
}

std::unique_ptr<QwenGpuSnapshot> QwenGpuArena::SaveSnapshot(
    std::uint32_t valid_context) {
  if (valid_context > max_context_) {
    throw std::length_error("Qwen snapshot exceeds the context length");
  }

  auto snapshot = std::unique_ptr<QwenGpuSnapshot>(new QwenGpuSnapshot());
  snapshot->num_layers_ = config_.num_layers;
  snapshot->attention_layers_ = config_.FullAttentionLayerCount();
  snapshot->full_attention_interval_ = config_.full_attention_interval;
  snapshot->kv_width_ = config_.num_key_value_heads * config_.head_dim;
  snapshot->max_context_ = max_context_;
  snapshot->valid_context_ = valid_context;
  snapshot->kv_storage_ = policy_.kv_cache_storage;
  snapshot->recurrent_state_storage_ = policy_.recurrent_state_storage;
  snapshot->kv_elements_per_plane_ = CheckedMultiply(
      CheckedMultiply(snapshot->attention_layers_, valid_context),
      snapshot->kv_width_);
  snapshot->conv_elements_per_layer_ =
      CheckedMultiply(config_.SsmQkvSize(), config_.ssm_conv_kernel);
  snapshot->conv_elements_ =
      CheckedMultiply(config_.num_layers - snapshot->attention_layers_,
                      snapshot->conv_elements_per_layer_);
  snapshot->deltanet_elements_per_layer_ = CheckedMultiply(
      CheckedMultiply(config_.ssm_time_step_rank, config_.ssm_state_size),
      config_.SsmValueSize());
  snapshot->deltanet_elements_ =
      CheckedMultiply(config_.num_layers - snapshot->attention_layers_,
                      snapshot->deltanet_elements_per_layer_);

  const std::size_t kv_f32_bytes = CheckedMultiply(
      CheckedMultiply(snapshot->kv_elements_per_plane_, 2), sizeof(float));
  const std::size_t kv_f16_bytes =
      CheckedMultiply(CheckedMultiply(snapshot->kv_elements_per_plane_, 2),
                      sizeof(std::uint16_t));
  const std::size_t conv_bytes =
      CheckedMultiply(snapshot->conv_elements_, sizeof(float));
  const std::size_t deltanet_bytes = CheckedMultiply(
      snapshot->deltanet_elements_,
      QwenRecurrentStateElementBytes(snapshot->recurrent_state_storage_));
  // An on-demand pooled arena shares its full and partial blocks: the
  // snapshot copies nothing, and this arena copies a shared block before it
  // writes there again.
  const bool share_kv = kv_pool_ != nullptr &&
                        kv_growth_ == KvGrowth::kOnDemand && kv_f16_bytes != 0;
  if (!share_kv)
    CheckedAdd(policy_.UsesFp16AttentionKv() ? kv_f16_bytes : kv_f32_bytes,
               &snapshot->payload_bytes_);
  CheckedAdd(conv_bytes, &snapshot->payload_bytes_);
  CheckedAdd(deltanet_bytes, &snapshot->payload_bytes_);

  if (kv_f32_bytes != 0 && !share_kv) {
    if (policy_.UsesFp16AttentionKv()) {
      ThrowOnHipError(hipMalloc(&snapshot->d_kv_f16_, kv_f16_bytes),
                      "failed to allocate Qwen FP16 snapshot KV");
    } else {
      ThrowOnHipError(hipMalloc(&snapshot->d_kv_f32_, kv_f32_bytes),
                      "failed to allocate Qwen FP32 snapshot KV");
    }
  }
  ThrowOnHipError(hipMalloc(&snapshot->d_ssm_conv_, conv_bytes),
                  "failed to allocate Qwen convolution snapshot");
  ThrowOnHipError(hipMalloc(&snapshot->d_ssm_deltanet_, deltanet_bytes),
                  "failed to allocate Qwen DeltaNet snapshot");

  // The transfer runs on its own stream. Drain state updates queued on the
  // arena stream (rollback, replay) before reading them from the live buffer.
  assert(d_ssm_conv_state == d_ssm_conv_buffers_[live_recurrent_] &&
         d_ssm_deltanet_state == d_ssm_deltanet_buffers_[live_recurrent_]);
  ThrowOnHipError(hipStreamSynchronize(stream),
                  "failed to synchronize Qwen snapshot capture");
  if (share_kv) {
    snapshot->kv_blocks_ = ShareKvBlocks(valid_context);
    snapshot->kv_pool_ = kv_pool_;
  }
  SnapshotTransfer transfer;
  if (kv_f32_bytes != 0 && !share_kv) {
    auto* destination = static_cast<std::uint8_t*>(policy_.UsesFp16AttentionKv()
                                                       ? snapshot->d_kv_f16_
                                                       : snapshot->d_kv_f32_);
    // Regions come in packed order, so each one lands right after the last.
    for (std::uint32_t plane = 0; plane < 2; ++plane)
      for (const auto& region : KvPlaneRegions(plane, valid_context)) {
        transfer.Copy2D(destination, region.width, region.data, region.pitch,
                        region.width, region.rows, hipMemcpyDeviceToDevice);
        destination += region.PackedBytes();
      }
  }
  transfer.Copy(snapshot->d_ssm_conv_, d_ssm_conv_state, conv_bytes,
                hipMemcpyDeviceToDevice);
  transfer.Copy(snapshot->d_ssm_deltanet_, d_ssm_deltanet_state, deltanet_bytes,
                hipMemcpyDeviceToDevice);
  return snapshot;
}

void QwenGpuArena::RestoreSnapshot(const QwenGpuSnapshot& snapshot) {
  const std::uint32_t attention_layers = config_.FullAttentionLayerCount();
  const std::uint32_t kv_width = config_.num_key_value_heads * config_.head_dim;
  const std::size_t conv_elements =
      CheckedMultiply(CheckedMultiply(config_.num_layers - attention_layers,
                                      config_.SsmQkvSize()),
                      config_.ssm_conv_kernel);
  const std::size_t deltanet_elements = CheckedMultiply(
      CheckedMultiply(CheckedMultiply(config_.num_layers - attention_layers,
                                      config_.ssm_time_step_rank),
                      config_.ssm_state_size),
      config_.SsmValueSize());
  if (snapshot.valid_context_ > max_context_ ||
      snapshot.attention_layers_ != attention_layers ||
      snapshot.kv_width_ != kv_width || snapshot.max_context_ != max_context_ ||
      snapshot.kv_storage_ != policy_.kv_cache_storage ||
      snapshot.recurrent_state_storage_ != policy_.recurrent_state_storage ||
      snapshot.conv_elements_ != conv_elements ||
      snapshot.deltanet_elements_ != deltanet_elements) {
    throw std::invalid_argument("Qwen snapshot is incompatible with the arena");
  }

  if (snapshot.kv_pool_ != nullptr && snapshot.kv_pool_ != kv_pool_)
    throw std::invalid_argument(
        "Qwen snapshot shares blocks of another KV pool");
  Reset();
  const std::size_t kv_f32_bytes = CheckedMultiply(
      CheckedMultiply(snapshot.kv_elements_per_plane_, 2), sizeof(float));
  if (snapshot.kv_pool_ != nullptr) {
    // The rows are the snapshot's blocks; the first write past them copies
    // the shared partial block.
    AdoptKvBlocks(snapshot.kv_blocks_);
  } else if (kv_f32_bytes != 0) {
    PrepareKvOverwrite(snapshot.valid_context_);
    const auto* source = static_cast<const std::uint8_t*>(
        policy_.UsesFp16AttentionKv() ? snapshot.d_kv_f16_
                                      : snapshot.d_kv_f32_);
    for (std::uint32_t plane = 0; plane < 2; ++plane)
      for (const auto& region :
           KvPlaneRegions(plane, snapshot.valid_context_)) {
        ThrowOnHipError(
            hipMemcpy2DAsync(region.data, region.pitch, source, region.width,
                             region.width, region.rows, hipMemcpyDeviceToDevice,
                             stream),
            "failed to restore Qwen KV snapshot");
        source += region.PackedBytes();
      }
  }
  ThrowOnHipError(hipMemcpyAsync(d_ssm_conv_state, snapshot.d_ssm_conv_,
                                 snapshot.conv_elements_ * sizeof(float),
                                 hipMemcpyDeviceToDevice, stream),
                  "failed to restore Qwen convolution snapshot");
  ThrowOnHipError(hipMemcpyAsync(d_ssm_deltanet_state, snapshot.d_ssm_deltanet_,
                                 snapshot.deltanet_elements_ *
                                     QwenRecurrentStateElementBytes(
                                         policy_.recurrent_state_storage),
                                 hipMemcpyDeviceToDevice, stream),
                  "failed to restore Qwen DeltaNet snapshot");
  ThrowOnHipError(hipStreamSynchronize(stream),
                  "failed to synchronize Qwen snapshot restore");
}

void QwenGpuArena::RestoreCompactSnapshot(
    std::span<const std::uint8_t> payload,
    std::uint32_t expected_valid_context) {
  const auto layout = ParseCompactSnapshotLayout(payload);
  CheckCompactCompatibility(config_, max_context_, policy_, layout,
                            expected_valid_context);
  Reset();
  PrepareKvOverwrite(layout.valid_context);
  if (layout.live_kv_bytes_per_plane != 0) {
    for (std::uint32_t plane = 0; plane < 2; ++plane) {
      const std::uint8_t* source =
          payload.data() + (plane == 0 ? layout.k_offset : layout.v_offset);
      for (const auto& region : RestoreRegions(plane, layout.valid_context, layout.paged)) {
        ThrowOnHipError(
            hipMemcpy2DAsync(region.data, region.pitch, source, region.width,
                             region.width, region.rows, hipMemcpyHostToDevice,
                             stream),
            "failed to restore compact Qwen KV cache");
        source += region.PackedBytes();
      }
    }
  }
  if (layout.conv_bytes != 0)
    ThrowOnHipError(
        hipMemcpyAsync(d_ssm_conv_state, payload.data() + layout.conv_offset,
                       layout.conv_bytes, hipMemcpyHostToDevice, stream),
        "failed to restore compact Qwen convolution state");
  if (layout.deltanet_bytes != 0)
    ThrowOnHipError(
        hipMemcpyAsync(d_ssm_deltanet_state,
                       payload.data() + layout.deltanet_offset,
                       layout.deltanet_bytes, hipMemcpyHostToDevice, stream),
        "failed to restore compact Qwen DeltaNet state");
  ThrowOnHipError(hipStreamSynchronize(stream),
                  "failed to synchronize compact Qwen snapshot restore");
}

models::qwen::vision::RopeLayout QwenGpuArena::RestoreCompactSnapshotStream(
    BounceStream& bounce, const BounceStream::Source& read,
    std::size_t compact_bytes, std::uint32_t expected_valid_context) {
  std::vector<std::uint8_t> header(kCompactSnapshotHeaderBytes);
  read(header);
  auto layout = ParseCompactSnapshotHeader(header, compact_bytes);
  CheckCompactCompatibility(config_, max_context_, policy_, layout,
                            expected_valid_context);

  Reset();
  // Reset clears recurrent state on the arena stream; the bounce stream's
  // uploads must land after it.
  ThrowOnHipError(hipStreamSynchronize(stream),
                  "failed to synchronize Qwen arena before a streamed restore");
  PrepareKvOverwrite(layout.valid_context);
  // Fresh blocks' table entries upload on the arena stream; the bounce
  // stream writes rows through raw pool addresses, not the table.
  if (layout.live_kv_bytes_per_plane != 0) {
    for (std::uint32_t plane = 0; plane < 2; ++plane)
      for (const auto& region : RestoreRegions(plane, layout.valid_context, layout.paged))
        bounce.In(region, read);
  }
  if (layout.conv_bytes != 0)
    bounce.In({d_ssm_conv_state, layout.conv_bytes, layout.conv_bytes, 1},
              read);
  if (layout.deltanet_bytes != 0)
    bounce.In({d_ssm_deltanet_state, layout.deltanet_bytes,
               layout.deltanet_bytes, 1},
              read);
  std::vector<std::uint8_t> grids(compact_bytes - layout.rope_offset);
  read(grids);
  ReadImageGrids(grids, &layout);
  return layout.rope;
}

QwenGpuArena::QwenGpuArena(const core::ModelConfig& config,
                           std::uint32_t max_context,
                           QwenExecutionPolicy policy,
                           std::shared_ptr<KvBlockPool> kv_pool,
                           KvGrowth kv_growth,
                           std::shared_ptr<QwenSharedPassScratch> shared_scratch)
    : config_(config),
      max_context_(std::max(max_context, 1U)),
      max_batch_(std::min(max_context_, QwenGpuArena::kMaxPromptBatch)),
      policy_(policy),
      shared_scratch_(std::move(shared_scratch)) {
  try {
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIPBLAS_CHECK(hipblasCreate(&hipblas_handle));
    HIPBLAS_CHECK(hipblasSetStream(hipblas_handle, stream));
    hipblaslt_gemm = std::make_unique<HipblasLtGemm>();

    const std::size_t hidden_size = config_.hidden_size;
    const std::size_t vocab_size = config_.vocab_size;
    const std::size_t num_layers = config_.SsmLayerCount();
    const std::size_t batch = max_batch_;
    const std::size_t ssm_qkv_size = config_.SsmQkvSize();
    const std::size_t time_step_rank = config_.ssm_time_step_rank;

    if (shared_scratch_ != nullptr && shared_scratch_->rows() < batch)
      throw std::invalid_argument(
          "shared Qwen pass scratch holds fewer rows than a prompt pass");
    // With a shared set, this session's own holds one decode step's rows.
    own_scratch_ = QwenPassScratchSet::Allocate(
        config_,
        shared_scratch_ != nullptr
            ? std::min<std::size_t>(batch, kSessionScratchRows)
            : batch,
        shared_scratch_ == nullptr);
    PointPassScratch(own_scratch_);
    HIP_CHECK(hipMalloc(&d_alpha_buf, batch * time_step_rank * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_beta_buf, batch * time_step_rank * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ssm_kq_scales,
                        batch * config_.ssm_group_count * 3 * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ssm_alpha_beta,
                        batch * time_step_rank * 2 * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_logits, vocab_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_prompt_tokens, std::max<std::size_t>(batch, 2) *
                                              sizeof(std::uint32_t)));
    HIP_CHECK(
        hipMalloc(&d_target_layer_features, 5 * hidden_size * sizeof(float)));

    const std::size_t split_k_elements = detail::DecodeAttentionScratchElements(
        config_.num_attention_heads, config_.head_dim);
    HIP_CHECK(
        hipMalloc(&d_split_k_attention, split_k_elements * sizeof(float)));

    if (policy_.UsesFp16AttentionKv()) {
      AllocateKvBlocks(std::move(kv_pool), kv_growth);
    } else {
      if (kv_pool != nullptr || kv_growth != KvGrowth::kWholeContext)
        throw std::invalid_argument(
            "Qwen FP32 reference KV does not use a block pool");
      const std::size_t total_kv = GetAttentionKvPlaneElements();
      HIP_CHECK(hipMalloc(&d_kv_cache, total_kv * sizeof(float) * 2));
    }

    const std::size_t total_conv =
        num_layers * ssm_qkv_size * config_.ssm_conv_kernel;
    HIP_CHECK(hipMalloc(&d_ssm_conv_buffers_[0], total_conv * sizeof(float)));

    const std::size_t total_deltanet = num_layers * time_step_rank *
                                       config_.ssm_state_size *
                                       config_.SsmValueSize();
    HIP_CHECK(hipMalloc(&d_ssm_deltanet_buffers_[0],
                        total_deltanet * QwenRecurrentStateElementBytes(
                                             policy_.recurrent_state_storage)));

    Reset();
  } catch (...) {
    FreeAll();
    throw;
  }
}

QwenGpuArena::~QwenGpuArena() {
  FreeAll();
}

void QwenGpuArena::AllocateKvBlocks(std::shared_ptr<KvBlockPool> kv_pool,
                                    KvGrowth kv_growth) {
  const std::uint32_t layers = config_.FullAttentionLayerCount();
  const std::size_t row_bytes = static_cast<std::size_t>(
                                    config_.num_key_value_heads) *
                                config_.head_dim * sizeof(std::uint16_t);
  const std::uint32_t blocks = KvBlockPool::BlocksFor(max_context_);
  if (kv_pool == nullptr) {
    if (kv_growth != KvGrowth::kWholeContext)
      throw std::invalid_argument(
          "on-demand Qwen KV blocks need a shared block pool");
    kv_pool = std::make_shared<KvBlockPool>(layers, row_bytes, blocks);
  }
  if (kv_pool->layers() != layers || kv_pool->row_bytes() != row_bytes)
    throw std::invalid_argument(
        "KV block pool rows do not match the Qwen attention layout");
  kv_growth_ = kv_growth;
  kv_pool_ = std::move(kv_pool);
  d_attention_kv_f16 = kv_pool_->data();
  HIP_CHECK(hipMalloc(&d_kv_block_table_, blocks * sizeof(std::uint32_t)));
  HIP_CHECK(hipHostMalloc(&h_kv_block_table_, blocks * sizeof(std::uint32_t),
                          hipHostMallocDefault));
  if (kv_growth_ == KvGrowth::kWholeContext) {
    kv_blocks_ = kv_pool_->Allocate(blocks);
    std::copy(kv_blocks_.begin(), kv_blocks_.end(), h_kv_block_table_);
    HIP_CHECK(hipMemcpy(d_kv_block_table_, h_kv_block_table_,
                        blocks * sizeof(std::uint32_t),
                        hipMemcpyHostToDevice));
  }
}

void QwenGpuArena::UploadKvTable(std::size_t first, std::size_t last) {
  if (first >= last)
    return;
  HIP_CHECK(hipMemcpyAsync(d_kv_block_table_ + first, h_kv_block_table_ + first,
                           (last - first) * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, stream));
}

void QwenGpuArena::CopyKvBlock(std::uint32_t from, std::uint32_t to) {
  const std::size_t width = kKvBlockTokens * kv_pool_->row_bytes();
  const std::size_t pitch =
      static_cast<std::size_t>(kv_pool_->layer_tokens()) * kv_pool_->row_bytes();
  for (std::uint32_t plane = 0; plane < 2; ++plane) {
    auto* base = static_cast<std::uint8_t*>(kv_pool_->data()) +
                 plane * kv_pool_->plane_bytes();
    HIP_CHECK(hipMemcpy2DAsync(base + to * width, pitch, base + from * width,
                               pitch, width, kv_pool_->layers(),
                               hipMemcpyDeviceToDevice, stream));
  }
}

void QwenGpuArena::EnsureKvRows(std::uint32_t begin, std::uint32_t end) {
  if (kv_pool_ == nullptr || begin >= end)
    return;
  if (end > max_context_)
    throw std::length_error("Qwen KV rows exceed the context length");
  const std::size_t first = begin / kKvBlockTokens;
  const std::size_t need = KvBlockPool::BlocksFor(end);
  const std::size_t mapped = kv_blocks_.size();
  // Shared blocks in [first, min(need, mapped)) must be copied first.
  std::vector<std::size_t> shared;
  for (std::size_t logical = first; logical < std::min(need, mapped);
       ++logical)
    if (kv_pool_->references(kv_blocks_[logical]) > 1)
      shared.push_back(logical);
  const std::size_t grow = need > mapped ? need - mapped : 0;
  if (shared.empty() && grow == 0)
    return;
  // One allocation for both, so a shortage leaves the arena unchanged.
  auto fresh = kv_pool_->Allocate(
      static_cast<std::uint32_t>(shared.size() + grow), kv_reservation_);
  if (!shared.empty()) {
    // Rewrites uploaded entries: drain the pending passes and uploads.
    HIP_CHECK(hipStreamSynchronize(stream));
    for (std::size_t i = 0; i < shared.size(); ++i) {
      const auto logical = shared[i];
      const auto old_block = kv_blocks_[logical];
      CopyKvBlock(old_block, fresh[i]);
      kv_blocks_[logical] = fresh[i];
      h_kv_block_table_[logical] = fresh[i];
      UploadKvTable(logical, logical + 1);
      kv_pool_->Release(std::span(&old_block, 1));
    }
  }
  for (std::size_t i = shared.size(); i < fresh.size(); ++i) {
    h_kv_block_table_[kv_blocks_.size()] = fresh[i];
    kv_blocks_.push_back(fresh[i]);
  }
  UploadKvTable(mapped, kv_blocks_.size());
}

std::vector<MutableDeviceRegion> QwenGpuArena::RestoreRegions(
    std::uint32_t plane, std::uint32_t valid_context, bool paged) const {
  if (!paged)
    return KvPlaneRegions(plane, valid_context);
  if (kv_pool_ == nullptr || kv_growth_ != KvGrowth::kOnDemand)
    throw std::invalid_argument(
        "a paged Qwen snapshot needs an on-demand KV arena");
  // The inline rows are the partial last block; RestoreKvBlock fills the
  // full ones.
  return {kv_pool_->BlockRegion(plane,
                                kv_blocks_.at(valid_context / kKvBlockTokens),
                                0, valid_context % kKvBlockTokens)};
}

void QwenGpuArena::PrepareKvOverwrite(std::uint32_t rows) {
  if (kv_pool_ == nullptr || kv_growth_ != KvGrowth::kOnDemand)
    return;
  // Every row below `rows` is about to be replaced: drop the old blocks
  // rather than copy shared ones first.
  TruncateKvRows(0);
  EnsureKvRows(0, rows);
}

void QwenGpuArena::TruncateKvRows(std::uint32_t rows) {
  if (kv_pool_ == nullptr || kv_growth_ != KvGrowth::kOnDemand)
    return;
  const std::size_t keep = KvBlockPool::BlocksFor(rows);
  if (keep >= kv_blocks_.size())
    return;
  // Passes in flight may still touch these blocks.
  HIP_CHECK(hipStreamSynchronize(stream));
  kv_pool_->Release(std::span(kv_blocks_).subspan(keep));
  kv_blocks_.resize(keep);
}

std::vector<std::uint32_t> QwenGpuArena::ShareKvBlocks(std::uint32_t rows) {
  if (kv_pool_ == nullptr)
    throw std::logic_error("Qwen FP32 reference KV has no blocks to share");
  const std::size_t count = KvBlockPool::BlocksFor(rows);
  if (count > kv_blocks_.size())
    throw std::length_error("Qwen KV rows to share are not mapped");
  std::vector<std::uint32_t> blocks(kv_blocks_.begin(),
                                    kv_blocks_.begin() + count);
  // The sharer reads rows written by passes still in flight.
  HIP_CHECK(hipStreamSynchronize(stream));
  kv_pool_->Retain(blocks);
  return blocks;
}

void QwenGpuArena::AdoptKvBlocks(std::span<const std::uint32_t> blocks) {
  if (kv_pool_ == nullptr || kv_growth_ != KvGrowth::kOnDemand)
    throw std::logic_error("only on-demand Qwen KV arenas adopt blocks");
  if (blocks.size() > KvBlockPool::BlocksFor(max_context_))
    throw std::length_error("adopted Qwen KV blocks exceed the context");
  kv_pool_->Retain(blocks);
  HIP_CHECK(hipStreamSynchronize(stream));
  kv_pool_->Release(kv_blocks_);
  kv_blocks_.assign(blocks.begin(), blocks.end());
  std::copy(kv_blocks_.begin(), kv_blocks_.end(), h_kv_block_table_);
  UploadKvTable(0, kv_blocks_.size());
}

void QwenGpuArena::ReplaceKvBlock(std::uint32_t logical, std::uint32_t block) {
  if (kv_pool_ == nullptr || kv_growth_ != KvGrowth::kOnDemand)
    throw std::logic_error("only on-demand Qwen KV arenas share blocks");
  if (logical >= kv_blocks_.size())
    throw std::out_of_range("Qwen arena has no such KV block");
  const auto old_block = kv_blocks_[logical];
  if (old_block == block)
    return;
  kv_pool_->Retain(std::span(&block, 1));
  // Passes, restores and table uploads in flight may still use the old one.
  HIP_CHECK(hipStreamSynchronize(stream));
  kv_blocks_[logical] = block;
  h_kv_block_table_[logical] = block;
  UploadKvTable(logical, logical + 1);
  kv_pool_->Release(std::span(&old_block, 1));
}

std::vector<MutableDeviceRegion> QwenGpuArena::KvPlaneRegions(
    std::uint32_t plane, std::uint32_t valid_context) const {
  if (valid_context > max_context_)
    throw std::length_error("Qwen KV rows exceed the context length");
  if (kv_pool_ != nullptr)
    return kv_pool_->PlaneRegions(plane, kv_blocks_, valid_context);
  // The FP32 reference is head-major: one row per (layer, KV head).
  const std::size_t rows = static_cast<std::size_t>(
                               config_.FullAttentionLayerCount()) *
                           config_.num_key_value_heads;
  const std::size_t pitch = CheckedMultiply(
      CheckedMultiply(max_context_, config_.head_dim), sizeof(float));
  const std::size_t width = CheckedMultiply(
      CheckedMultiply(valid_context, config_.head_dim), sizeof(float));
  auto* base = reinterpret_cast<std::uint8_t*>(d_kv_cache) +
               plane * CheckedMultiply(pitch, rows);
  return {{base, pitch, width, rows}};
}

QwenGpuScratchView QwenGpuArena::GetScratchView(std::size_t batch_size) {
  const std::size_t batch = std::min<std::size_t>(batch_size, max_batch_);
  if (batch > GetScratchRows())
    throw std::logic_error(
        "Qwen pass needs " + std::to_string(batch) +
        " scratch rows; this session holds " +
        std::to_string(GetScratchRows()) +
        " until it leases the shared pass scratch");
  const std::size_t hidden = config_.hidden_size;
  const std::size_t attention = config_.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config_.num_key_value_heads) * config_.head_dim;
  const std::size_t q_projection = 2 * attention;
  const std::size_t ssm_qkv = config_.SsmQkvSize();
  const std::size_t recurrent =
      std::max<std::size_t>(attention, config_.ssm_inner_size);
  const std::size_t projection = std::max<std::size_t>(q_projection, ssm_qkv);
  const std::size_t bf16_scratch =
      batch *
      std::max<std::size_t>({config_.intermediate_size, hidden, projection,
                             ssm_qkv + config_.ssm_inner_size +
                                 (2 * config_.ssm_time_step_rank)});
  const std::size_t split_k = detail::DecodeAttentionScratchElements(
      config_.num_attention_heads, config_.head_dim);
  const std::size_t weight_bf16 =
      hidden * std::max<std::size_t>({q_projection, kv, attention,
                                      config_.intermediate_size, ssm_qkv,
                                      config_.ssm_inner_size,
                                      config_.ssm_time_step_rank});
  return {
      .decode =
          {
              .hidden = {d_hidden, batch * hidden},
              .normed = {d_normed, batch * hidden},
              .logits = {d_logits, config_.vocab_size},
              .bf16 = {static_cast<hip_bfloat16*>(d_scratch_bf16),
                       bf16_scratch},
              .weight_bf16 = {d_weights_bf16,
                              d_weights_bf16 != nullptr ? weight_bf16 : 0},
              .prompt_tokens = {d_prompt_tokens,
                                std::max<std::size_t>(batch, 2)},
              // d_alpha_buf is reused only after all SSM layers finish.
              // Preserve that exact address while exposing the sampling epoch's
              // uint32 view.
              .sampled_token = {reinterpret_cast<std::uint32_t*>(d_alpha_buf),
                                1},
          },
      .attention =
          {
              .q = {d_q, batch * attention},
              .k = {d_k, batch * kv},
              .v = {d_v, batch * kv},
              .output = {d_attn_out, batch * hidden},
              .split_k = {d_split_k_attention, split_k},
          },
      .ssm =
          {
              .qkv = {d_ssm_qkv, batch * projection},
              .conv_out = {d_conv_out, batch * ssm_qkv},
              .gate = {d_ssm_gate, batch * recurrent},
              .out = {d_ssm_out, batch * recurrent},
              .alpha = {d_alpha_buf, batch * config_.ssm_time_step_rank},
              .beta = {d_beta_buf, batch * config_.ssm_time_step_rank},
          },
      .ffn =
          {
              .gate = {d_ffn_gate, batch * config_.intermediate_size},
              .up = {d_ffn_up, batch * config_.intermediate_size},
              .activation = {d_ffn_act, batch * config_.intermediate_size},
              .out = {d_ffn_out, batch * hidden},
          },
  };
}

void QwenGpuArena::SetTargetLayerCapture(
    std::span<const std::uint32_t> target_layer_ids) {
  if (target_layer_ids.size() > kMaxTargetLayerTaps) {
    throw std::invalid_argument("too many target hidden-layer taps");
  }
  for (const std::uint32_t layer : target_layer_ids) {
    if (layer >= config_.num_layers) {
      throw std::invalid_argument("target hidden-layer tap is out of range");
    }
  }
  target_layer_ids_.assign(target_layer_ids.begin(), target_layer_ids.end());
}

std::optional<std::size_t> QwenGpuArena::GetTargetLayerCaptureIndex(
    std::uint32_t layer) const noexcept {
  const auto iterator = std::ranges::find(target_layer_ids_, layer);
  if (iterator == target_layer_ids_.end()) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(
      std::distance(target_layer_ids_.begin(), iterator));
}

QwenGpuArena::QwenGpuArena(QwenGpuArena&& other) noexcept
    : config_(other.config_),
      max_context_(other.max_context_),
      max_batch_(other.max_batch_),
      policy_(other.policy_),
      target_layer_ids_(std::move(other.target_layer_ids_)),
      kv_pool_(std::move(other.kv_pool_)),
      kv_growth_(other.kv_growth_),
      kv_blocks_(std::move(other.kv_blocks_)),
      d_kv_block_table_(std::exchange(other.d_kv_block_table_, nullptr)),
      h_kv_block_table_(std::exchange(other.h_kv_block_table_, nullptr)),
      kv_reservation_(std::move(other.kv_reservation_)) {
  d_hidden = other.d_hidden;
  d_normed = other.d_normed;
  d_q = other.d_q;
  d_k = other.d_k;
  d_v = other.d_v;
  d_attn_out = other.d_attn_out;
  d_ffn_gate = other.d_ffn_gate;
  d_ffn_up = other.d_ffn_up;
  d_ffn_act = other.d_ffn_act;
  d_ffn_out = other.d_ffn_out;
  d_ssm_qkv = other.d_ssm_qkv;
  d_conv_out = other.d_conv_out;
  d_ssm_gate = other.d_ssm_gate;
  d_ssm_out = other.d_ssm_out;
  d_alpha_buf = other.d_alpha_buf;
  d_beta_buf = other.d_beta_buf;
  d_ssm_kq_scales = other.d_ssm_kq_scales;
  d_ssm_alpha_beta = other.d_ssm_alpha_beta;
  d_logits = other.d_logits;
  d_attention_kv_f16 = other.d_attention_kv_f16;
  d_kv_cache = other.d_kv_cache;
  d_ssm_conv_state = other.d_ssm_conv_state;
  d_ssm_deltanet_state = other.d_ssm_deltanet_state;
  d_prompt_tokens = other.d_prompt_tokens;
  d_target_layer_features = other.d_target_layer_features;
  stream = other.stream;
  hipblas_handle = other.hipblas_handle;
  hipblaslt_gemm = std::move(other.hipblaslt_gemm);
  d_scratch_bf16 = other.d_scratch_bf16;
  d_scratch_q8_act = other.d_scratch_q8_act;
  d_split_k_attention = other.d_split_k_attention;
  d_weights_bf16 = other.d_weights_bf16;
  d_ssm_conv_buffers_ = other.d_ssm_conv_buffers_;
  d_ssm_deltanet_buffers_ = other.d_ssm_deltanet_buffers_;
  live_recurrent_ = other.live_recurrent_;
  saved_recurrent_ = other.saved_recurrent_;
  source_recurrent_ = other.source_recurrent_;
  d_ssm_replay_qkv_ = other.d_ssm_replay_qkv_;
  d_ssm_replay_alpha_ = other.d_ssm_replay_alpha_;
  d_ssm_replay_beta_ = other.d_ssm_replay_beta_;
  d_ssm_replay_enabled_ = other.d_ssm_replay_enabled_;
  saved_context_ = other.saved_context_;
  replay_last_position_ = other.replay_last_position_;
  replay_captured_positions_ = other.replay_captured_positions_;
  has_saved_state_ = other.has_saved_state_;
  replay_capture_active_ = other.replay_capture_active_;
  own_scratch_ = std::exchange(other.own_scratch_, {});
  shared_scratch_ = std::move(other.shared_scratch_);
  scratch_leased_ = std::exchange(other.scratch_leased_, false);

  other.d_hidden = nullptr;
  other.d_normed = nullptr;
  other.d_q = nullptr;
  other.d_k = nullptr;
  other.d_v = nullptr;
  other.d_attn_out = nullptr;
  other.d_ffn_gate = nullptr;
  other.d_ffn_up = nullptr;
  other.d_ffn_act = nullptr;
  other.d_ffn_out = nullptr;
  other.d_ssm_qkv = nullptr;
  other.d_conv_out = nullptr;
  other.d_ssm_gate = nullptr;
  other.d_ssm_out = nullptr;
  other.d_alpha_buf = nullptr;
  other.d_beta_buf = nullptr;
  other.d_ssm_kq_scales = nullptr;
  other.d_ssm_alpha_beta = nullptr;
  other.d_logits = nullptr;
  other.d_attention_kv_f16 = nullptr;
  other.d_kv_cache = nullptr;
  other.d_ssm_conv_state = nullptr;
  other.d_ssm_deltanet_state = nullptr;
  other.d_prompt_tokens = nullptr;
  other.d_target_layer_features = nullptr;
  other.stream = nullptr;
  other.hipblas_handle = nullptr;
  other.d_scratch_bf16 = nullptr;
  other.d_scratch_q8_act = nullptr;
  other.d_split_k_attention = nullptr;
  other.d_weights_bf16 = nullptr;
  other.d_ssm_conv_buffers_ = {};
  other.d_ssm_deltanet_buffers_ = {};
  other.live_recurrent_ = 0;
  other.saved_recurrent_ = 0;
  other.source_recurrent_ = 0;
  other.d_ssm_replay_qkv_ = nullptr;
  other.d_ssm_replay_alpha_ = nullptr;
  other.d_ssm_replay_beta_ = nullptr;
  other.d_ssm_replay_enabled_ = nullptr;
  other.saved_context_ = 0;
  other.replay_last_position_ = 0;
  other.replay_captured_positions_ = 0;
  other.has_saved_state_ = false;
  other.replay_capture_active_ = false;
}

QwenGpuArena& QwenGpuArena::operator=(QwenGpuArena&& other) noexcept {
  if (this != &other) {
    FreeAll();
    config_ = other.config_;
    max_context_ = other.max_context_;
    max_batch_ = other.max_batch_;
    policy_ = other.policy_;
    target_layer_ids_ = std::move(other.target_layer_ids_);
    kv_pool_ = std::move(other.kv_pool_);
    kv_growth_ = other.kv_growth_;
    kv_blocks_ = std::move(other.kv_blocks_);
    d_kv_block_table_ = std::exchange(other.d_kv_block_table_, nullptr);
    h_kv_block_table_ = std::exchange(other.h_kv_block_table_, nullptr);
    kv_reservation_ = std::move(other.kv_reservation_);
    d_hidden = other.d_hidden;
    d_normed = other.d_normed;
    d_q = other.d_q;
    d_k = other.d_k;
    d_v = other.d_v;
    d_attn_out = other.d_attn_out;
    d_ffn_gate = other.d_ffn_gate;
    d_ffn_up = other.d_ffn_up;
    d_ffn_act = other.d_ffn_act;
    d_ffn_out = other.d_ffn_out;
    d_ssm_qkv = other.d_ssm_qkv;
    d_conv_out = other.d_conv_out;
    d_ssm_gate = other.d_ssm_gate;
    d_ssm_out = other.d_ssm_out;
    d_alpha_buf = other.d_alpha_buf;
    d_beta_buf = other.d_beta_buf;
    d_ssm_kq_scales = other.d_ssm_kq_scales;
    d_ssm_alpha_beta = other.d_ssm_alpha_beta;
    d_logits = other.d_logits;
    d_attention_kv_f16 = other.d_attention_kv_f16;
    d_kv_cache = other.d_kv_cache;
    d_ssm_conv_state = other.d_ssm_conv_state;
    d_ssm_deltanet_state = other.d_ssm_deltanet_state;
    d_prompt_tokens = other.d_prompt_tokens;
    d_target_layer_features = other.d_target_layer_features;
    stream = other.stream;
    hipblas_handle = other.hipblas_handle;
    hipblaslt_gemm = std::move(other.hipblaslt_gemm);
    d_scratch_bf16 = other.d_scratch_bf16;
    d_scratch_q8_act = other.d_scratch_q8_act;
    d_split_k_attention = other.d_split_k_attention;
    d_weights_bf16 = other.d_weights_bf16;
    d_ssm_conv_buffers_ = other.d_ssm_conv_buffers_;
    d_ssm_deltanet_buffers_ = other.d_ssm_deltanet_buffers_;
    live_recurrent_ = other.live_recurrent_;
    saved_recurrent_ = other.saved_recurrent_;
    source_recurrent_ = other.source_recurrent_;
    d_ssm_replay_qkv_ = other.d_ssm_replay_qkv_;
    d_ssm_replay_alpha_ = other.d_ssm_replay_alpha_;
    d_ssm_replay_beta_ = other.d_ssm_replay_beta_;
    d_ssm_replay_enabled_ = other.d_ssm_replay_enabled_;
    saved_context_ = other.saved_context_;
    replay_last_position_ = other.replay_last_position_;
    replay_captured_positions_ = other.replay_captured_positions_;
    has_saved_state_ = other.has_saved_state_;
    replay_capture_active_ = other.replay_capture_active_;
    own_scratch_ = std::exchange(other.own_scratch_, {});
    shared_scratch_ = std::move(other.shared_scratch_);
    scratch_leased_ = std::exchange(other.scratch_leased_, false);

    other.d_hidden = nullptr;
    other.d_normed = nullptr;
    other.d_q = nullptr;
    other.d_k = nullptr;
    other.d_v = nullptr;
    other.d_attn_out = nullptr;
    other.d_ffn_gate = nullptr;
    other.d_ffn_up = nullptr;
    other.d_ffn_act = nullptr;
    other.d_ffn_out = nullptr;
    other.d_ssm_qkv = nullptr;
    other.d_conv_out = nullptr;
    other.d_ssm_gate = nullptr;
    other.d_ssm_out = nullptr;
    other.d_alpha_buf = nullptr;
    other.d_beta_buf = nullptr;
    other.d_ssm_kq_scales = nullptr;
    other.d_ssm_alpha_beta = nullptr;
    other.d_logits = nullptr;
    other.d_attention_kv_f16 = nullptr;
    other.d_kv_cache = nullptr;
    other.d_ssm_conv_state = nullptr;
    other.d_ssm_deltanet_state = nullptr;
    other.d_prompt_tokens = nullptr;
    other.d_target_layer_features = nullptr;
    other.stream = nullptr;
    other.hipblas_handle = nullptr;
    other.d_scratch_bf16 = nullptr;
    other.d_scratch_q8_act = nullptr;
    other.d_split_k_attention = nullptr;
    other.d_weights_bf16 = nullptr;
    other.d_ssm_conv_buffers_ = {};
    other.d_ssm_deltanet_buffers_ = {};
    other.live_recurrent_ = 0;
    other.saved_recurrent_ = 0;
    other.source_recurrent_ = 0;
    other.d_ssm_replay_qkv_ = nullptr;
    other.d_ssm_replay_alpha_ = nullptr;
    other.d_ssm_replay_beta_ = nullptr;
    other.d_ssm_replay_enabled_ = nullptr;
    other.saved_context_ = 0;
    other.replay_last_position_ = 0;
    other.replay_captured_positions_ = 0;
    other.has_saved_state_ = false;
    other.replay_capture_active_ = false;
  }
  return *this;
}

void QwenGpuArena::Reset() {
  const std::size_t num_layers = config_.SsmLayerCount();
  const std::size_t total_conv =
      num_layers * config_.SsmQkvSize() * config_.ssm_conv_kernel;
  const std::size_t total_deltanet = num_layers * config_.ssm_time_step_rank *
                                     config_.ssm_state_size *
                                     config_.SsmValueSize();

  // Buffer 0 becomes the live state again; whatever buffer 1 holds is fully
  // overwritten before it is read.
  live_recurrent_ = 0;
  saved_recurrent_ = 0;
  source_recurrent_ = 0;
  SyncLivePointers();

  // Attention only reads positions through the current token. Prefill/decode
  // writes those rows before use; clearing reserved future KV adds O(capacity)
  // work to every short conversation and snapshot restore.
  if (d_ssm_conv_state != nullptr) {
    HIP_CHECK(hipMemsetAsync(d_ssm_conv_state, 0, total_conv * sizeof(float),
                             stream));
  }
  if (d_ssm_deltanet_state != nullptr) {
    HIP_CHECK(hipMemsetAsync(
        d_ssm_deltanet_state, 0,
        total_deltanet *
            QwenRecurrentStateElementBytes(policy_.recurrent_state_storage),
        stream));
  }
  DisableSsmReplayCapture();
  saved_context_ = 0;
  replay_last_position_ = 0;
  replay_captured_positions_ = 0;
  has_saved_state_ = false;
}

void QwenGpuArena::FreeAll() noexcept {
  // A pass that threw mid-lease left the shared set pointed at.
  EndPassScratch();
  own_scratch_.Free();
  shared_scratch_.reset();
  if (d_alpha_buf != nullptr)
    LogCleanupError(hipFree(d_alpha_buf));
  if (d_beta_buf != nullptr)
    LogCleanupError(hipFree(d_beta_buf));
  if (d_ssm_kq_scales != nullptr)
    LogCleanupError(hipFree(d_ssm_kq_scales));
  if (d_ssm_alpha_beta != nullptr)
    LogCleanupError(hipFree(d_ssm_alpha_beta));
  if (d_logits != nullptr)
    LogCleanupError(hipFree(d_logits));
  // FP16 rows belong to the pool; return this arena's blocks to it once its
  // passes are done with them.
  if (kv_pool_ != nullptr) {
    if (stream != nullptr && !kv_blocks_.empty())
      LogCleanupError(hipStreamSynchronize(stream));
    kv_pool_->Release(kv_blocks_);
  }
  kv_reservation_.Reset();
  kv_pool_.reset();
  kv_blocks_.clear();
  if (d_kv_block_table_ != nullptr)
    LogCleanupError(hipFree(d_kv_block_table_));
  d_kv_block_table_ = nullptr;
  if (h_kv_block_table_ != nullptr)
    LogCleanupError(hipHostFree(h_kv_block_table_));
  h_kv_block_table_ = nullptr;
  if (d_target_layer_features != nullptr) {
    LogCleanupError(hipFree(d_target_layer_features));
    d_target_layer_features = nullptr;
  }
  if (d_kv_cache != nullptr)
    LogCleanupError(hipFree(d_kv_cache));
  for (float* buffer : d_ssm_conv_buffers_)
    if (buffer != nullptr)
      LogCleanupError(hipFree(buffer));
  for (void* buffer : d_ssm_deltanet_buffers_)
    if (buffer != nullptr)
      LogCleanupError(hipFree(buffer));
  if (d_prompt_tokens != nullptr)
    LogCleanupError(hipFree(d_prompt_tokens));
  if (d_split_k_attention != nullptr)
    LogCleanupError(hipFree(d_split_k_attention));
  if (d_ssm_replay_qkv_ != nullptr)
    LogCleanupError(hipFree(d_ssm_replay_qkv_));
  if (d_ssm_replay_alpha_ != nullptr)
    LogCleanupError(hipFree(d_ssm_replay_alpha_));
  if (d_ssm_replay_beta_ != nullptr)
    LogCleanupError(hipFree(d_ssm_replay_beta_));
  if (d_ssm_replay_enabled_ != nullptr)
    LogCleanupError(hipFree(d_ssm_replay_enabled_));
  if (hipblas_handle != nullptr)
    LogCleanupError(hipblasDestroy(hipblas_handle));
  hipblaslt_gemm.reset();
  if (stream != nullptr)
    LogCleanupError(hipStreamDestroy(stream));

  d_hidden = nullptr;
  d_normed = nullptr;
  d_q = nullptr;
  d_k = nullptr;
  d_v = nullptr;
  d_attn_out = nullptr;
  d_ffn_gate = nullptr;
  d_ffn_up = nullptr;
  d_ffn_act = nullptr;
  d_ffn_out = nullptr;
  d_ssm_qkv = nullptr;
  d_conv_out = nullptr;
  d_ssm_gate = nullptr;
  d_ssm_out = nullptr;
  d_alpha_buf = nullptr;
  d_beta_buf = nullptr;
  d_ssm_kq_scales = nullptr;
  d_ssm_alpha_beta = nullptr;
  d_logits = nullptr;
  d_attention_kv_f16 = nullptr;
  d_kv_cache = nullptr;
  d_ssm_conv_state = nullptr;
  d_ssm_deltanet_state = nullptr;
  d_prompt_tokens = nullptr;
  d_scratch_bf16 = nullptr;
  d_scratch_q8_act = nullptr;
  d_split_k_attention = nullptr;
  d_weights_bf16 = nullptr;
  d_ssm_conv_buffers_ = {};
  d_ssm_deltanet_buffers_ = {};
  live_recurrent_ = 0;
  saved_recurrent_ = 0;
  source_recurrent_ = 0;
  d_ssm_replay_qkv_ = nullptr;
  d_ssm_replay_alpha_ = nullptr;
  d_ssm_replay_beta_ = nullptr;
  d_ssm_replay_enabled_ = nullptr;
  saved_context_ = 0;
  replay_last_position_ = 0;
  replay_captured_positions_ = 0;
  has_saved_state_ = false;
  replay_capture_active_ = false;
  hipblas_handle = nullptr;
  stream = nullptr;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
