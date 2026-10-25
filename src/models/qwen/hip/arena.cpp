#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/snapshot_transfer.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace gufo::hip {
namespace {

constexpr std::uint32_t kMaxPromptBatch = 4096;
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
  models::qwen::vision::RopeLayout rope;
};

CompactSnapshotLayout MakeCompactSnapshotLayout(
    std::uint32_t attention_layers, std::uint32_t kv_width,
    std::uint32_t max_context, std::uint32_t valid_context,
    QwenKvCacheStorage kv_storage,
    QwenRecurrentStateStorage recurrent_state_storage,
    std::size_t conv_elements, std::size_t deltanet_elements,
    std::size_t image_count = 0) {
  CompactSnapshotLayout layout{
      .attention_layers = attention_layers,
      .kv_width = kv_width,
      .max_context = max_context,
      .valid_context = valid_context,
      .kv_storage = kv_storage,
      .recurrent_state_storage = recurrent_state_storage,
      .conv_elements = conv_elements,
      .deltanet_elements = deltanet_elements,
  };
  const std::size_t kv_element_bytes = kv_storage == QwenKvCacheStorage::kFp16
                                           ? sizeof(std::uint16_t)
                                           : sizeof(float);
  layout.live_kv_elements_per_plane = CheckedMultiply(
      CheckedMultiply(attention_layers, valid_context), kv_width);
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

CompactSnapshotLayout ParseCompactSnapshotLayout(
    std::span<const std::uint8_t> payload) {
  if (payload.size() < kCompactSnapshotHeaderBytes ||
      !std::equal(kCompactSnapshotMagic.begin(), kCompactSnapshotMagic.end(),
                  payload.begin())) {
    throw std::invalid_argument("Qwen compact snapshot header is invalid");
  }
  const auto version = GetLittleEndian<std::uint32_t>(payload, 8);
  const auto image_count = GetLittleEndian<std::uint32_t>(payload, 72);
  if ((version != kCompactSnapshotVersion && version != 2) ||
      image_count > GetLittleEndian<std::uint32_t>(payload, 28) ||
      (version == 1 && image_count != 0) ||
      GetLittleEndian<std::uint32_t>(payload, 12) !=
          kCompactSnapshotHeaderBytes ||
      GetLittleEndian<std::uint32_t>(payload, 76) != 0) {
    throw std::invalid_argument("Qwen compact snapshot version is invalid");
  }

  const std::uint32_t kv_storage_value =
      GetLittleEndian<std::uint32_t>(payload, 32);
  const std::uint32_t recurrent_storage_value =
      GetLittleEndian<std::uint32_t>(payload, 36);
  if (kv_storage_value >
          static_cast<std::uint32_t>(QwenKvCacheStorage::kFp16) ||
      recurrent_storage_value >
          static_cast<std::uint32_t>(QwenRecurrentStateStorage::kBf16)) {
    throw std::invalid_argument("Qwen compact snapshot precision is invalid");
  }

  auto layout = MakeCompactSnapshotLayout(
      GetLittleEndian<std::uint32_t>(payload, 16),
      GetLittleEndian<std::uint32_t>(payload, 20),
      GetLittleEndian<std::uint32_t>(payload, 24),
      GetLittleEndian<std::uint32_t>(payload, 28),
      static_cast<QwenKvCacheStorage>(kv_storage_value),
      static_cast<QwenRecurrentStateStorage>(recurrent_storage_value),
      SizeFromU64(GetLittleEndian<std::uint64_t>(payload, 48)),
      SizeFromU64(GetLittleEndian<std::uint64_t>(payload, 56)), image_count);
  if (layout.live_kv_elements_per_plane !=
          SizeFromU64(GetLittleEndian<std::uint64_t>(payload, 40)) ||
      layout.total_bytes !=
          SizeFromU64(GetLittleEndian<std::uint64_t>(payload, 64)) ||
      layout.total_bytes != payload.size()) {
    throw std::invalid_argument("Qwen compact snapshot size is invalid");
  }
  for (std::size_t i = 0; i < image_count; ++i) {
    const auto offset = layout.rope_offset + i * 12;
    layout.rope.images.push_back(
        {GetLittleEndian<std::uint32_t>(payload, offset),
         GetLittleEndian<std::uint32_t>(payload, offset + 4),
         GetLittleEndian<std::uint32_t>(payload, offset + 8)});
  }
  layout.rope.Validate(layout.max_context);
  return layout;
}

}  // namespace

models::qwen::vision::RopeLayout QwenGpuSnapshot::ReadRopeLayout(
    std::span<const std::uint8_t> payload) {
  return ParseCompactSnapshotLayout(payload).rope;
}

QwenGpuSnapshot::~QwenGpuSnapshot() {
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

std::size_t QwenGpuSnapshot::CompactPayloadBytes() const {
  const std::size_t recurrent_layers = num_layers_ - attention_layers_;
  return MakeCompactSnapshotLayout(
             attention_layers_, kv_width_, max_context_, valid_context_,
             kv_storage_, recurrent_state_storage_,
             CheckedMultiply(recurrent_layers, conv_elements_per_layer_),
             CheckedMultiply(recurrent_layers, deltanet_elements_per_layer_),
             vision_layout_.images.size())
      .total_bytes;
}

std::size_t QwenGpuSnapshot::SerializeCompact(
    std::span<std::uint8_t> destination) const {
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
      vision_layout_.images.size());
  if (destination.size() != layout.total_bytes) {
    throw std::invalid_argument(
        "Qwen compact snapshot destination size is invalid");
  }
  // Only the header has reserved bytes. Every payload byte is overwritten
  // below; clearing it first needlessly competes with inference for bandwidth.
  std::fill_n(destination.begin(), kCompactSnapshotHeaderBytes,
              std::uint8_t{0});
  std::copy(kCompactSnapshotMagic.begin(), kCompactSnapshotMagic.end(),
            destination.begin());
  PutLittleEndian<std::uint32_t>(destination, 8,
                                 vision_layout_.images.empty() ? 1U : 2U);
  PutLittleEndian<std::uint32_t>(destination, 72, vision_layout_.images.size());
  PutLittleEndian<std::uint32_t>(
      destination, 12, static_cast<std::uint32_t>(kCompactSnapshotHeaderBytes));
  PutLittleEndian<std::uint32_t>(destination, 16, attention_layers_);
  PutLittleEndian<std::uint32_t>(destination, 20, kv_width_);
  PutLittleEndian<std::uint32_t>(destination, 24, max_context_);
  PutLittleEndian<std::uint32_t>(destination, 28, valid_context_);
  PutLittleEndian<std::uint32_t>(destination, 32,
                                 static_cast<std::uint32_t>(kv_storage_));
  PutLittleEndian<std::uint32_t>(
      destination, 36, static_cast<std::uint32_t>(recurrent_state_storage_));
  PutLittleEndian<std::uint64_t>(
      destination, 40,
      static_cast<std::uint64_t>(layout.live_kv_elements_per_plane));
  PutLittleEndian<std::uint64_t>(
      destination, 48, static_cast<std::uint64_t>(layout.conv_elements));
  PutLittleEndian<std::uint64_t>(
      destination, 56, static_cast<std::uint64_t>(layout.deltanet_elements));
  PutLittleEndian<std::uint64_t>(
      destination, 64, static_cast<std::uint64_t>(layout.total_bytes));

  SnapshotTransfer transfer;
  if (layout.live_kv_bytes_per_plane != 0) {
    const std::size_t kv_element_bytes =
        kv_storage_ == QwenKvCacheStorage::kFp16 ? sizeof(std::uint16_t)
                                                 : sizeof(float);
    const std::size_t source_pitch = CheckedMultiply(
        CheckedMultiply(valid_context_, kv_width_), kv_element_bytes);
    const void* source =
        kv_storage_ == QwenKvCacheStorage::kFp16 ? d_kv_f16_ : d_kv_f32_;
    if (source == nullptr) {
      throw std::logic_error("Qwen compact snapshot has no KV storage");
    }
    transfer.Copy2D(destination.data() + layout.k_offset,
                    layout.live_kv_bytes_per_plane / attention_layers_, source,
                    source_pitch,
                    layout.live_kv_bytes_per_plane / attention_layers_,
                    attention_layers_);
    const auto* source_bytes = static_cast<const std::uint8_t*>(source);
    const std::size_t full_plane_bytes =
        CheckedMultiply(kv_elements_per_plane_, kv_element_bytes);
    transfer.Copy2D(destination.data() + layout.v_offset,
                    layout.live_kv_bytes_per_plane / attention_layers_,
                    source_bytes + full_plane_bytes, source_pitch,
                    layout.live_kv_bytes_per_plane / attention_layers_,
                    attention_layers_);
  }
  if (layout.conv_bytes != 0) {
    transfer.Copy(destination.data() + layout.conv_offset, d_ssm_conv_,
                  layout.conv_bytes);
  }
  if (layout.deltanet_bytes != 0) {
    transfer.Copy(destination.data() + layout.deltanet_offset, d_ssm_deltanet_,
                  layout.deltanet_bytes);
  }
  vision_layout_.Validate(max_context_);
  for (std::size_t i = 0; i < vision_layout_.images.size(); ++i) {
    const auto& grid = vision_layout_.images[i];
    const auto offset = layout.rope_offset + i * 12;
    PutLittleEndian<std::uint32_t>(destination, offset, grid.offset);
    PutLittleEndian<std::uint32_t>(destination, offset + 4, grid.height);
    PutLittleEndian<std::uint32_t>(destination, offset + 8, grid.width);
  }
  return destination.size();
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
  return CheckedSum(
      CheckedMultiply(CheckedMultiply(q8_blocks, sizeof(float)), 10), 4096);
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
    HIP_CHECK(hipMalloc(&set.bf16,
                        CheckedMultiply(CheckedMultiply(rows, w.bf16_width),
                                        sizeof(hip_bfloat16))));
    HIP_CHECK(hipMalloc(&set.q8_act, Q8ActivationBytes(rows, w.bf16_width)));
    if (weights)
      HIP_CHECK(
          hipMalloc(&set.weights_bf16,
                    CheckedMultiply(w.weight_elements, sizeof(hip_bfloat16))));
  } catch (...) {
    set.Free();
    throw;
  }
  return set;
}

void QwenPassScratchSet::Free() noexcept {
  for (float** buffer :
       {&hidden, &normed, &q, &k, &v, &attn_out, &ffn_gate, &ffn_up, &ffn_act,
        &ffn_out, &ssm_qkv, &conv_out, &ssm_gate, &ssm_out}) {
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
          std::min<std::size_t>(std::max(max_context, 1U), kMaxPromptBatch),
          true)),
      bytes_(QwenPassScratchSet::Bytes(config, set_.rows, true)) {
  if (const auto error =
          hipEventCreateWithFlags(&released_, hipEventDisableTiming);
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
             {set_.hidden, w.hidden},
             {set_.normed, w.hidden},
             {set_.q, w.attention},
             {set_.k, w.kv},
             {set_.v, w.kv},
             {set_.attn_out, w.hidden},
             {set_.ffn_gate, w.intermediate},
             {set_.ffn_up, w.intermediate},
             {set_.ffn_act, w.intermediate},
             {set_.ffn_out, w.hidden},
             {set_.ssm_qkv, w.projection},
             {set_.conv_out, w.ssm_qkv},
             {set_.ssm_gate, w.recurrent},
             {set_.ssm_out, w.recurrent}})
      HIP_CHECK(hipMemset(buffer, 0, rows * width * sizeof(float)));
    HIP_CHECK(
        hipMemset(set_.bf16, 0, rows * w.bf16_width * sizeof(hip_bfloat16)));
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
    QwenExecutionPolicy policy, bool shared_scratch, bool shared_kv) {
  const std::size_t context = std::max(max_context, 1U);
  const std::size_t batch = std::min<std::size_t>(context, kMaxPromptBatch);
  const std::size_t hidden = config.hidden_size;
  const std::size_t kv =
      CheckedMultiply(config.num_key_value_heads, config.head_dim);
  const std::size_t time_step = config.ssm_time_step_rank;

  QwenGpuMemoryUsage usage;
  auto& scratch = usage.temporary_scratch_bytes;
  CheckedAdd(
      QwenPassScratchSet::Bytes(
          config,
          shared_scratch ? std::min<std::size_t>(batch, kSessionScratchRows)
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
  // FP16 rows come from the session's block pool in whole blocks; with a
  // shared pool they are the pool's, charged once for every session.
  const std::size_t kv_rows =
      !policy.UsesFp16AttentionKv() ? context
      : shared_kv
          ? 0
          : CheckedMultiply(KvBlockPool::BlocksFor(context), kKvBlockTokens);
  const std::size_t total_kv = CheckedMultiply(
      CheckedMultiply(config.FullAttentionLayerCount(), kv), kv_rows);
  AddAllocation(
      CheckedMultiply(total_kv, 2),
      policy.UsesFp16AttentionKv() ? sizeof(std::uint16_t) : sizeof(float),
      &state);

  CheckedAdd(RecurrentStateBytes(config, policy), &state);
  return usage;
}

std::size_t QwenGpuArena::RecurrentStateBytes(const core::ModelConfig& config,
                                              QwenExecutionPolicy policy) {
  std::size_t state = 0;
  const std::size_t total_conv = CheckedMultiply(
      CheckedMultiply(config.SsmLayerCount(), config.SsmQkvSize()),
      config.ssm_conv_kernel);
  AddAllocation(total_conv, sizeof(float), &state);
  const std::size_t total_deltanet = CheckedMultiply(
      CheckedMultiply(
          CheckedMultiply(config.SsmLayerCount(), config.ssm_time_step_rank),
          config.ssm_state_size),
      config.SsmValueSize());
  AddAllocation(total_deltanet,
                QwenRecurrentStateElementBytes(policy.recurrent_state_storage),
                &state);
  return state;
}

QwenGpuMemoryUsage QwenGpuArena::GetMemoryUsage() const {
  auto usage = EstimateMemoryUsage(config_, max_context_, policy_,
                                   shared_scratch_ != nullptr, kv_shared_);
  if (d_ssm_conv_buffers_[1] && d_ssm_deltanet_buffers_[1])
    CheckedAdd(RecurrentStateBytes(config_, policy_),
               &usage.request_state_bytes);
  return usage;
}

std::size_t QwenGpuArena::SnapshotPayloadBytes(
    std::uint32_t valid_context) const {
  RequireOwnKvPool();
  if (valid_context > max_context_)
    throw std::length_error("Qwen snapshot exceeds the context length");
  // The state counts whole FP16 blocks: every row past valid_context of the
  // session's KV rows is unused.
  const auto unused = CheckedMultiply(
      CheckedMultiply(
          CheckedMultiply(config_.FullAttentionLayerCount(),
                          KvLayerPitchRows() - valid_context),
          CheckedMultiply(config_.num_key_value_heads, config_.head_dim)),
      policy_.UsesFp16AttentionKv() ? 2 * sizeof(std::uint16_t)
                                    : 2 * sizeof(float));
  return GetMemoryUsage().request_state_bytes - unused;
}

std::unique_ptr<QwenGpuSnapshot> QwenGpuArena::SaveSnapshot(
    std::uint32_t valid_context) {
  RequireOwnKvPool();
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
  CheckedAdd(policy_.UsesFp16AttentionKv() ? kv_f16_bytes : kv_f32_bytes,
             &snapshot->payload_bytes_);
  CheckedAdd(conv_bytes, &snapshot->payload_bytes_);
  CheckedAdd(deltanet_bytes, &snapshot->payload_bytes_);

  if (kv_f32_bytes != 0) {
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
  SnapshotTransfer transfer;
  if (kv_f32_bytes != 0) {
    const auto element_bytes =
        policy_.UsesFp16AttentionKv() ? sizeof(std::uint16_t) : sizeof(float);
    // Canonical FP16 KV is token-major; the FP32 reference plane is head-major.
    const auto width =
        policy_.UsesFp16AttentionKv() ? snapshot->kv_width_ : config_.head_dim;
    const auto rows =
        snapshot->attention_layers_ *
        (policy_.UsesFp16AttentionKv() ? 1U : config_.num_key_value_heads);
    const auto source_pitch = CheckedMultiply(
        CheckedMultiply(KvLayerPitchRows(), width), element_bytes);
    const auto target_pitch =
        CheckedMultiply(CheckedMultiply(valid_context, width), element_bytes);
    const auto target_plane = CheckedMultiply(target_pitch, rows);
    auto* destination = static_cast<std::uint8_t*>(policy_.UsesFp16AttentionKv()
                                                       ? snapshot->d_kv_f16_
                                                       : snapshot->d_kv_f32_);
    for (std::uint32_t plane = 0; plane < 2; ++plane)
      transfer.Copy2D(destination + plane * target_plane, target_pitch,
                      KvPlaneBase(plane), source_pitch, target_pitch, rows,
                      hipMemcpyDeviceToDevice);
  }
  transfer.Copy(snapshot->d_ssm_conv_, d_ssm_conv_state, conv_bytes,
                hipMemcpyDeviceToDevice);
  transfer.Copy(snapshot->d_ssm_deltanet_, d_ssm_deltanet_state, deltanet_bytes,
                hipMemcpyDeviceToDevice);
  return snapshot;
}

void QwenGpuArena::RestoreSnapshot(const QwenGpuSnapshot& snapshot) {
  RequireOwnKvPool();
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

  Reset();
  const std::size_t kv_f32_bytes = CheckedMultiply(
      CheckedMultiply(snapshot.kv_elements_per_plane_, 2), sizeof(float));
  if (kv_f32_bytes != 0) {
    const auto element_bytes =
        policy_.UsesFp16AttentionKv() ? sizeof(std::uint16_t) : sizeof(float);
    const auto width =
        policy_.UsesFp16AttentionKv() ? kv_width : config_.head_dim;
    const auto rows =
        attention_layers *
        (policy_.UsesFp16AttentionKv() ? 1U : config_.num_key_value_heads);
    const auto source_pitch = CheckedMultiply(
        CheckedMultiply(snapshot.valid_context_, width), element_bytes);
    const auto target_pitch = CheckedMultiply(
        CheckedMultiply(KvLayerPitchRows(), width), element_bytes);
    const auto source_plane = CheckedMultiply(source_pitch, rows);
    const auto* source = static_cast<const std::uint8_t*>(
        policy_.UsesFp16AttentionKv() ? snapshot.d_kv_f16_
                                      : snapshot.d_kv_f32_);
    for (std::uint32_t plane = 0; plane < 2; ++plane)
      ThrowOnHipError(
          hipMemcpy2DAsync(KvPlaneBase(plane), target_pitch,
                           source + plane * source_plane, source_pitch,
                           source_pitch, rows, hipMemcpyDeviceToDevice, stream),
          "failed to restore Qwen KV snapshot");
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
  RequireOwnKvPool();
  const auto layout = ParseCompactSnapshotLayout(payload);
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

  Reset();
  if (layout.live_kv_bytes_per_plane != 0) {
    const std::size_t kv_element_bytes =
        policy_.UsesFp16AttentionKv() ? sizeof(std::uint16_t) : sizeof(float);
    const auto width =
        policy_.UsesFp16AttentionKv() ? kv_width : config_.head_dim;
    const auto rows =
        attention_layers *
        (policy_.UsesFp16AttentionKv() ? 1U : config_.num_key_value_heads);
    const std::size_t destination_pitch = CheckedMultiply(
        CheckedMultiply(KvLayerPitchRows(), width), kv_element_bytes);
    const std::size_t source_pitch = layout.live_kv_bytes_per_plane / rows;
    ThrowOnHipError(
        hipMemcpy2DAsync(KvPlaneBase(0), destination_pitch,
                         payload.data() + layout.k_offset, source_pitch,
                         source_pitch, rows, hipMemcpyHostToDevice, stream),
        "failed to restore compact Qwen K cache");
    ThrowOnHipError(
        hipMemcpy2DAsync(KvPlaneBase(1), destination_pitch,
                         payload.data() + layout.v_offset, source_pitch,
                         source_pitch, rows, hipMemcpyHostToDevice, stream),
        "failed to restore compact Qwen V cache");
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

QwenGpuArena::QwenGpuArena(
    const core::ModelConfig& config, std::uint32_t max_context,
    QwenExecutionPolicy policy,
    std::shared_ptr<QwenSharedPassScratch> shared_scratch,
    std::shared_ptr<KvBlockPool> kv_pool)
    : config_(config),
      max_context_(std::max(max_context, 1U)),
      max_batch_(std::min(max_context_, kMaxPromptBatch)),
      policy_(policy),
      kv_shared_(kv_pool != nullptr),
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
      AllocateKvBlocks(std::move(kv_pool));
    } else {
      if (kv_pool != nullptr)
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

void QwenGpuArena::AllocateKvBlocks(std::shared_ptr<KvBlockPool> shared) {
  const KvBlockPlane rows{
      .layers = config_.FullAttentionLayerCount(),
      .row_bytes = static_cast<std::size_t>(config_.num_key_value_heads) *
                   config_.head_dim * sizeof(std::uint16_t)};
  const std::uint32_t blocks = KvBlockPool::BlocksFor(max_context_);
  HIP_CHECK(hipMalloc(&d_kv_block_table_, blocks * sizeof(std::uint32_t)));
  HIP_CHECK(hipHostMalloc(&h_kv_block_table_, blocks * sizeof(std::uint32_t),
                          hipHostMallocDefault));
  if (shared != nullptr) {
    if (shared->geometry().planes().size() != 2 ||
        shared->geometry().planes()[0] != rows ||
        shared->geometry().planes()[1] != rows)
      throw std::invalid_argument(
          "shared KV block pool rows do not match the Qwen attention layout");
    // Starts empty; EnsureKvRows maps blocks as passes write. Unmapped
    // entries name block 0, so a masked read past the frontier stays inside
    // the pool.
    kv_pool_ = std::move(shared);
    std::fill_n(h_kv_block_table_, blocks, 0U);
  } else {
    // K and V planes of the session's own pool.
    kv_pool_ = std::make_shared<KvBlockPool>(std::vector{rows, rows}, blocks);
    kv_blocks_ = kv_pool_->Allocate(blocks);
    for (std::uint32_t i = 0; i < blocks; ++i)
      if (kv_blocks_[i] != i)
        throw std::logic_error(
            "a fresh Qwen KV pool gave nonconsecutive blocks");
    std::ranges::copy(kv_blocks_, h_kv_block_table_);
  }
  d_attention_kv_f16 = kv_pool_->data();
  HIP_CHECK(hipMemcpy(d_kv_block_table_, h_kv_block_table_,
                      blocks * sizeof(std::uint32_t), hipMemcpyHostToDevice));
}

void QwenGpuArena::RequireOwnKvPool() const {
  if (kv_shared_)
    throw std::logic_error(
        "Qwen snapshot copies need a session-owned KV pool; sessions on a "
        "shared pool use the component cache");
}

void QwenGpuArena::EnsureKvRows(std::uint32_t end) {
  if (!kv_shared_)
    return;
  if (end > max_context_)
    throw std::length_error("Qwen KV rows exceed the context length");
  const std::size_t mapped = kv_blocks_.size();
  const std::size_t need = KvBlockPool::BlocksFor(end);
  if (need <= mapped)
    return;
  // All or nothing: a shortage throws before the table changes.
  const auto fresh = kv_pool_->Allocate(
      static_cast<std::uint32_t>(need - mapped), &kv_reservation_);
  // Entries past `mapped` were last uploaded before their blocks were
  // released, after a stream drain, so no pending upload reads them.
  for (const auto block : fresh) {
    h_kv_block_table_[kv_blocks_.size()] = block;
    kv_blocks_.push_back(block);
  }
  HIP_CHECK(hipMemcpyAsync(
      d_kv_block_table_ + mapped, h_kv_block_table_ + mapped,
      (need - mapped) * sizeof(std::uint32_t), hipMemcpyHostToDevice, stream));
}

void QwenGpuArena::TruncateKvRows(std::uint32_t rows) {
  if (!kv_shared_)
    return;
  const std::size_t keep = KvBlockPool::BlocksFor(rows);
  if (keep >= kv_blocks_.size())
    return;
  // Passes and table uploads in flight may still use these blocks.
  HIP_CHECK(hipStreamSynchronize(stream));
  kv_pool_->Release(std::span(kv_blocks_).subspan(keep));
  kv_blocks_.resize(keep);
}

std::uint32_t QwenGpuArena::ReserveKvRows(std::uint32_t rows,
                                          std::uint32_t margin) {
  if (!kv_shared_)
    return 0;
  const std::size_t target =
      std::min<std::size_t>(KvBlockPool::BlocksFor(rows) + std::size_t{margin},
                            KvBlockPool::BlocksFor(max_context_));
  const std::size_t need =
      target > kv_blocks_.size() ? target - kv_blocks_.size() : 0;
  kv_reservation_.Reset();
  auto reservation = kv_pool_->TryReserve(static_cast<std::uint32_t>(need));
  if (!reservation)
    throw std::length_error(
        "the shared KV block pool cannot hold this request: it needs " +
        std::to_string(need) + " more blocks of " +
        std::to_string(kKvBlockTokens) + " tokens, " +
        std::to_string(kv_pool_->free_blocks()) + " of " +
        std::to_string(kv_pool_->blocks()) + " are free");
  kv_reservation_ = std::move(reservation);
  return static_cast<std::uint32_t>(need);
}

std::uint8_t* QwenGpuArena::KvPlaneBase(std::uint32_t plane) const {
  if (plane > 1)
    throw std::out_of_range("Qwen attention KV has two planes");
  if (kv_pool_ != nullptr)
    return reinterpret_cast<std::uint8_t*>(kv_pool_->plane_data(plane));
  return reinterpret_cast<std::uint8_t*>(d_kv_cache +
                                         plane * GetAttentionKvPlaneElements());
}

QwenGpuScratchView QwenGpuArena::GetScratchView(std::size_t batch_size) {
  const std::size_t batch = std::min<std::size_t>(batch_size, max_batch_);
  if (batch > GetScratchRows())
    throw std::logic_error("Qwen pass needs " + std::to_string(batch) +
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
      target_layer_ids_(std::move(other.target_layer_ids_)) {
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
  kv_pool_ = std::move(other.kv_pool_);
  kv_shared_ = other.kv_shared_;
  kv_blocks_ = std::move(other.kv_blocks_);
  d_kv_block_table_ = std::exchange(other.d_kv_block_table_, nullptr);
  h_kv_block_table_ = std::exchange(other.h_kv_block_table_, nullptr);
  kv_reservation_ = std::move(other.kv_reservation_);
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
    kv_pool_ = std::move(other.kv_pool_);
    kv_shared_ = other.kv_shared_;
    kv_blocks_ = std::move(other.kv_blocks_);
    d_kv_block_table_ = std::exchange(other.d_kv_block_table_, nullptr);
    h_kv_block_table_ = std::exchange(other.h_kv_block_table_, nullptr);
    kv_reservation_ = std::move(other.kv_reservation_);
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
  // A shared pool gets this session's rows back; the next pass maps afresh.
  TruncateKvRows(0);
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
  // The FP16 rows are the pool's.
  // Return this arena's blocks once its passes are done with them.
  if (kv_pool_ != nullptr) {
    if (kv_shared_ && stream != nullptr && !kv_blocks_.empty())
      LogCleanupError(hipStreamSynchronize(stream));
    kv_pool_->Release(kv_blocks_);
  }
  kv_reservation_.Reset();
  kv_blocks_.clear();
  kv_pool_.reset();
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
