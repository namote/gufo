#include "src/models/qwen38_flash_next/runtime/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <map>
#include <utility>

#include "src/core/hip/managed_alloc.hpp"
#include "src/core/hip/weight_upload.hpp"
#include "src/models/qwen38_flash_next/exl3_weights.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

/// The quantized GEMM tier reads whole 256-element k-iterations, so a row
/// whose length is only a multiple of 32 over-reads into the next row and,
/// on the last row, past the tensor. Every upload carries this tail so the
/// over-read stays inside the allocation (the extra bytes meet zeroed
/// activation padding and contribute nothing).
constexpr std::size_t kTailMargin = 4096;

struct Conversion {
  void* source;
  void* destination;
  std::size_t count;
  /// Q8_0 conversions only: the F16 source has no other reader and is
  /// released once converted.
  bool release_source{false};
};

/// The trunk and the optional MTP block as one traversal's output.
struct Uploaded {
  DeviceTensor token_embd;
  DeviceTensor output;
  DeviceMixer hc_head;
  std::vector<DeviceLayer> layers;
  DeviceLayer mtp;
};

/// Walks the model's tensors in upload order. With `weights` null the walk
/// only measures: it sums the bytes every allocation would request and does
/// no device work, so the weight placement can be decided from the tensors.
struct Uploader {
  hip::WeightUpload& stager;
  const hip::WeightAllocator* weights;
  std::vector<Conversion>& conversions;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_half_cols;
  std::size_t& max_q8_cols;
  std::string* error;
  std::deque<std::vector<quant::exl3::DeviceLinear>>& descriptors;
  std::map<const Exl3HostLinear*, DeviceTensor> linear_cache;
  std::uint32_t checkpoint_base{0}, mtp_checkpoint_base{0};
  /// EXL3 pack: its F16 HC up weights are replaced by Q8_0 copies.
  bool exl3_model{false};
  bool ok{true};
  std::uint32_t shard_base{0};
  std::uint64_t requested{0};
  /// F16 mixer down weights to quantize into their Q8_0 copies once the
  /// upload has drained (count = elements).
  std::vector<Conversion> q8_conversions;

  [[nodiscard]] bool Measuring() const noexcept { return weights == nullptr; }

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  [[nodiscard]] hipError_t Allocate(void** pointer, std::size_t size) {
    requested += size;
    if (Measuring()) {
      return hipSuccess;
    }
    return weights->Allocate(pointer, size);
  }

  DeviceTensor CopyExl3(const TensorRef& t) {
    const auto found = linear_cache.find(t.exl3.data());
    if (found != linear_cache.end()) {
      auto result = found->second;
      result.rows = static_cast<std::uint32_t>(t.rows);
      result.output_begin = t.output_begin;
      return result;
    }
    const auto align = [](std::size_t bytes) {
      return (bytes + 255) / 256 * 256;
    };
    std::size_t size = 0;
    for (const auto& h : t.exl3)
      size += align(h.view.TrellisBytes()) + align(h.view.k * 2) +
              align(h.view.n * 2);
    const auto descriptor_offset = size;
    size += align(t.exl3.size() * sizeof(quant::exl3::DeviceLinear));
    void* slab = nullptr;
    if (Allocate(&slab, size) != hipSuccess) {
      Fail("EXL3 weight slab allocation failed");
      return {};
    }
    DeviceTensor d;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    d.output_begin = t.output_begin;
    if (Measuring()) {
      linear_cache.emplace(t.exl3.data(), d);
      return d;
    }
    allocations.push_back(slab);
    bytes += size;
    descriptors.emplace_back();
    auto& linears = descriptors.back();
    linears.reserve(t.exl3.size());
    std::size_t offset = 0;
    for (const auto& h : t.exl3) {
      auto l = h.view;
      auto* base = static_cast<std::uint8_t*>(slab);
      const auto copy = [&](std::uint64_t source, std::size_t count,
                            std::size_t dest) {
        if (ok && !stager.Copy(checkpoint_base + h.shard, source, count,
                               base + dest, error))
          Fail("EXL3 slab upload failed");
      };
      l.trellis = reinterpret_cast<const std::uint16_t*>(base + offset);
      if (h.input_blocks.empty() && h.output_blocks.empty()) {
        copy(h.trellis_offset, l.TrellisBytes(), offset);
      } else {
        const auto tile_bytes =
            static_cast<std::size_t>(l.rate.TileWords()) * 2;
        const auto tiles_n = l.n / 16;
        for (std::uint32_t tk = 0; tk < l.k / 16; ++tk) {
          const auto dk =
              h.input_blocks.empty() ? tk : h.input_blocks[tk / 8] * 8 + tk % 8;
          if (h.output_blocks.empty()) {
            copy(h.trellis_offset + std::size_t{tk} * tiles_n * tile_bytes,
                 tiles_n * tile_bytes,
                 offset + std::size_t{dk} * tiles_n * tile_bytes);
          } else {
            for (std::uint32_t bn = 0; bn < l.n / 128; ++bn)
              copy(h.trellis_offset +
                       (std::size_t{tk} * tiles_n + bn * 8) * tile_bytes,
                   8 * tile_bytes,
                   offset +
                       (std::size_t{dk} * tiles_n + h.output_blocks[bn] * 8) *
                           tile_bytes);
          }
        }
      }
      offset += align(l.TrellisBytes());
      const auto scales = [&](std::uint64_t source, std::uint32_t width,
                              const std::vector<std::uint32_t>& blocks) {
        if (blocks.empty())
          copy(source, width * 2, offset);
        else
          for (std::uint32_t b = 0; b < width / 128; ++b)
            copy(source + b * 256, 256, offset + blocks[b] * 256);
        const auto* pointer =
            reinterpret_cast<const std::uint16_t*>(base + offset);
        offset += align(width * 2);
        return pointer;
      };
      l.suh = scales(h.suh_offset, l.k, h.input_blocks);
      l.svh = scales(h.svh_offset, l.n, h.output_blocks);
      linears.push_back(l);
    }
    d.data = const_cast<std::uint16_t*>(linears[0].trellis);
    d.exl3 = linears.data();
    d.device_exl3 = reinterpret_cast<const quant::exl3::DeviceLinear*>(
        static_cast<std::uint8_t*>(slab) + descriptor_offset);
    if (hipMemcpyAsync(const_cast<quant::exl3::DeviceLinear*>(d.device_exl3),
                       linears.data(), linears.size() * sizeof(linears[0]),
                       hipMemcpyHostToDevice, nullptr) != hipSuccess)
      Fail("EXL3 descriptor upload failed");
    linear_cache.emplace(t.exl3.data(), d);
    return d;
  }

  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    if (!t.exl3.empty())
      return CopyExl3(t);
    const std::size_t size = t.SizeBytes();
    void* ptr = nullptr;
    if (Allocate(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("device allocation failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes)");
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    const bool copied =
        t.memory_owned
            ? hipMemcpyAsync(ptr, t.data, size, hipMemcpyHostToDevice,
                             nullptr) == hipSuccess
            : stager.Copy(
                  (t.safetensors ? checkpoint_base : shard_base) + t.shard,
                  t.file_offset, size, ptr, error);
    if (!copied) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr ? ": " + *error : std::string()));
      return d;
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    if (t.type == core::GgmlType::kBF16 || t.type == core::GgmlType::kF16) {
      max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    }
    if (t.type == core::GgmlType::kQ8_0 && t.experts == 1) {
      max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    }
    return d;
  }

  // GGUF packs [fc_embedding | fc_hidden] across each row. Split on a
  // quantization-block boundary without dequantizing or changing any weight.
  void SplitMtpProjection(const TensorRef& t, DeviceTensor& embedding,
                          DeviceTensor& hidden) {
    if (t.empty() || !ok)
      return;
    const auto combined = Copy(t);
    if (!ok || (!Measuring() && !stager.Finish(error))) {
      Fail("MTP projection upload failed");
      return;
    }
    const std::size_t row_bytes = t.SizeBytes() / t.rows / 2;
    const std::size_t part_bytes = row_bytes * t.rows;
    for (std::uint32_t part = 0; part < 2; ++part) {
      auto& dst = part == 0 ? embedding : hidden;
      dst = combined;
      dst.cols /= 2;
      if (Allocate(&dst.data, part_bytes + kTailMargin) != hipSuccess) {
        Fail("MTP split projection allocation failed");
        return;
      }
      if (Measuring()) {
        continue;
      }
      allocations.push_back(dst.data);
      bytes += part_bytes + kTailMargin;
      const auto* src =
          static_cast<const std::uint8_t*>(combined.data) + part * row_bytes;
      if (hipMemcpy2D(dst.data, row_bytes, src, 2 * row_bytes, row_bytes,
                      t.rows, hipMemcpyDeviceToDevice) != hipSuccess ||
          hipMemset(static_cast<std::uint8_t*>(dst.data) + part_bytes, 0,
                    kTailMargin) != hipSuccess) {
        Fail("MTP split projection copy failed");
        return;
      }
    }
    if (Measuring()) {
      return;
    }
    std::erase(allocations, combined.data);
    (void)hipFree(combined.data);
    bytes -= t.SizeBytes() + kTailMargin;
  }

  /// Uploads matrices of one type stacked along rows; every input shares
  /// `cols`. An F32 stack (router logits, GDN alpha/beta: the only
  /// unquantized projections) is narrowed to F16, which the wide-batch GEMM
  /// tier runs at speed. A Q8_0 stack merges projections of one input into
  /// a single decode GEMV.
  DeviceTensor Stack(std::initializer_list<const TensorRef*> parts) {
    DeviceTensor d;
    if (!ok) {
      return d;
    }
    std::size_t rows = 0;
    std::size_t size = 0;
    const core::GgmlType type = (*parts.begin())->type;
    for (const TensorRef* t : parts) {
      if (t->empty() || t->type != type || t->cols != (*parts.begin())->cols ||
          (type != core::GgmlType::kF32 && type != core::GgmlType::kQ8_0 &&
           type != core::GgmlType::kF16)) {
        Fail("stacked upload needs F32, F16 or Q8_0 tensors of one shape");
        return d;
      }
      rows += t->rows;
      size += t->SizeBytes();
    }
    void* ptr = nullptr;
    if (Allocate(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("stacked tensor allocation failed");
      return d;
    }
    if (!Measuring()) {
      allocations.push_back(ptr);
      bytes += size + kTailMargin;
      std::size_t offset = 0;
      for (const TensorRef* t : parts) {
        const bool copied =
            t->memory_owned
                ? hipMemcpyAsync(static_cast<std::uint8_t*>(ptr) + offset,
                                 t->data, t->SizeBytes(), hipMemcpyHostToDevice,
                                 nullptr) == hipSuccess
                : stager.Copy((t->safetensors ? checkpoint_base : shard_base) +
                                  t->shard,
                              t->file_offset, t->SizeBytes(),
                              static_cast<std::uint8_t*>(ptr) + offset, error);
        if (!copied) {
          Fail("upload failed for " + std::string(t->name));
          return d;
        }
        offset += t->SizeBytes();
      }
    }
    if (type == core::GgmlType::kQ8_0 || type == core::GgmlType::kF16) {
      if (Measuring()) {
        return d;
      }
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0,
                           kTailMargin, nullptr);
      d.data = ptr;
      d.type = type;
      d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
      d.rows = static_cast<std::uint32_t>(rows);
      if (type == core::GgmlType::kQ8_0)
        max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
      else
        max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
      return d;
    }
    const std::size_t count = size / sizeof(float);
    void* half = nullptr;
    if (Allocate(&half, count * sizeof(std::uint16_t) + kTailMargin) !=
        hipSuccess) {
      Fail("stacked tensor allocation failed");
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(half);
    bytes += count * sizeof(std::uint16_t) + kTailMargin;
    // Keep the small F32 stacks until the disk pipeline drains. Converting
    // each router immediately would serialize every layer's uploads.
    conversions.push_back({ptr, half, count});
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(half) + count * 2, 0,
                         kTailMargin, nullptr);
    d.data = half;
    d.type = core::GgmlType::kF16;
    d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
    d.rows = static_cast<std::uint32_t>(rows);
    max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
    return d;
  }

  /// A Q8_0 copy of the uploaded F16 matrix `half`, quantized once the
  /// upload has drained; empty on failure or while measuring.
  DeviceTensor Q8Copy(const DeviceTensor& half, const TensorRef& source,
                      bool release_source) {
    const std::size_t count =
        static_cast<std::size_t>(source.cols) * source.rows;
    const std::size_t size = count / 32 * 34;
    void* q8 = nullptr;
    if (Allocate(&q8, size + kTailMargin) != hipSuccess) {
      Fail("device allocation failed for the Q8_0 copy of " +
           std::string(source.name));
      return {};
    }
    if (Measuring())
      return {};
    allocations.push_back(q8);
    bytes += size + kTailMargin;
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(q8) + size, 0, kTailMargin,
                         nullptr);
    q8_conversions.push_back({half.data, q8, count, release_source});
    DeviceTensor d;
    d.data = q8;
    d.type = core::GgmlType::kQ8_0;
    d.cols = half.cols;
    d.rows = half.rows;
    max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
    return d;
  }

  DeviceMixer Mixer(const HcMixer& m) {
    DeviceMixer d{Copy(m.norm), Copy(m.down), Copy(m.up), Copy(m.inject), {}};
    // The EXL3 down projection runs W8A8 (prefill) or the Q8_0 GEMV
    // (decode/verify) on a Q8_0 copy, as the GGUF route's Q8_0 mixer
    // weights do; an unnormed prefill input still reads the F16 rows.
    if (!ok || m.down.empty() || !m.down.exl3.empty() ||
        m.down.type != core::GgmlType::kF16 || m.down.cols % 32 != 0)
      return d;
    d.down_q8 = Q8Copy(d.down, m.down, false);
    // Every up reader (the Q8_0 GEMV of decode, the Q8_0 x F16 GEMM of
    // prefill) takes Q8_0 weights, so the F16 rows are replaced.
    if (ok && exl3_model && !m.up.empty() && m.up.exl3.empty() &&
        m.up.type == core::GgmlType::kF16 && m.up.cols % 32 == 0) {
      const DeviceTensor up = Q8Copy(d.up, m.up, true);
      if (up.data != nullptr)
        d.up = up;
    }
    return d;
  }

  DeviceLayer Layer(const LayerWeights& l) {
    DeviceLayer d;
    d.linear = l.linear;
    d.hc_attn = Mixer(l.hc_attn);
    d.hc_ffn = Mixer(l.hc_ffn);
    // Projections of one input are stacked into one Q8_0 GEMV where the
    // quantization allows; otherwise they stay separate.
    const auto stackable = [](std::initializer_list<const TensorRef*> parts) {
      for (const TensorRef* t : parts) {
        if (t->empty() || t->type != core::GgmlType::kQ8_0 ||
            t->cols != (*parts.begin())->cols) {
          return false;
        }
      }
      return true;
    };
    if (l.linear && stackable({&l.ssm_qkv, &l.ssm_gate})) {
      d.ssm_in = Stack({&l.ssm_qkv, &l.ssm_gate});
    } else {
      d.ssm_qkv = Copy(l.ssm_qkv);
      d.ssm_gate = Copy(l.ssm_gate);
    }
    d.ssm_conv1d = Copy(l.ssm_conv1d);
    if (l.linear) {
      d.ssm_alpha_beta = Stack({&l.ssm_alpha, &l.ssm_beta});
    }
    d.ssm_dt = Copy(l.ssm_dt);
    d.ssm_a = Copy(l.ssm_a);
    d.ssm_norm = Copy(l.ssm_norm);
    d.ssm_out = Copy(l.ssm_out);
    if (!l.linear && stackable({&l.attn_q, &l.attn_k, &l.attn_v})) {
      d.attn_qkv = Stack({&l.attn_q, &l.attn_k, &l.attn_v});
    } else {
      d.attn_q = Copy(l.attn_q);
      d.attn_k = Copy(l.attn_k);
      d.attn_v = Copy(l.attn_v);
    }
    d.attn_out = Copy(l.attn_out);
    d.attn_q_norm = Copy(l.attn_q_norm);
    d.attn_k_norm = Copy(l.attn_k_norm);
    d.indexer_q = Copy(l.indexer_q);
    d.indexer_k = Copy(l.indexer_k);
    d.indexer_q_norm = Copy(l.indexer_q_norm);
    d.indexer_k_norm = Copy(l.indexer_k_norm);
    d.ple_key = Copy(l.ple_key);
    d.ple_value = Copy(l.ple_value);
    d.ple_norm_key = Copy(l.ple_norm_key);
    d.ple_norm_query = Copy(l.ple_norm_query);
    d.ple_norm_conv = Copy(l.ple_norm_conv);
    d.ple_conv1d = Copy(l.ple_conv1d);
    d.router = Stack({&l.router, &l.shexp_gate_inp});
    d.ffn_gate_exps = Copy(l.ffn_gate_exps);
    d.ffn_up_exps = Copy(l.ffn_up_exps);
    d.ffn_down_exps = Copy(l.ffn_down_exps);
    d.shexp_gate = Copy(l.shexp_gate);
    d.shexp_up = Copy(l.shexp_up);
    d.shexp_down = Copy(l.shexp_down);
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    if (!l.nextn_fc_embedding.empty()) {
      d.nextn_fc_embedding = Copy(l.nextn_fc_embedding);
      d.nextn_fc_hidden = Copy(l.nextn_fc_hidden);
    } else {
      SplitMtpProjection(l.nextn_eh_proj, d.nextn_fc_embedding,
                         d.nextn_fc_hidden);
    }
    d.nextn_head = Mixer(l.nextn_head);
    return d;
  }

  Uploaded Model(const ModelWeights& w, const MtpWeights* mtp,
                 std::uint32_t trunk_shards) {
    Uploaded u;
    u.token_embd = Copy(w.token_embd);
    u.output =
        w.output.data == w.token_embd.data ? u.token_embd : Copy(w.output);
    u.hc_head = Mixer(w.hc_head);
    u.layers.reserve(w.layers.size());
    for (const auto& l : w.layers) {
      u.layers.push_back(Layer(l));
      if (!ok) {
        return u;
      }
    }
    if (mtp != nullptr) {
      shard_base = trunk_shards;
      checkpoint_base = mtp_checkpoint_base;
      u.mtp = Layer(mtp->block);
    }
    return u;
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

bool DeviceModel::SetDraftVocab(std::uint32_t prefix_rows,
                                std::vector<std::int32_t> specials,
                                std::string* error_msg) {
  const auto fail = [&](const std::string& message) {
    if (error_msg != nullptr)
      *error_msg = message;
    return false;
  };
  if (output_.empty() || output_.rows == 0 ||
      (output_.exl3 == nullptr && output_bytes_ % output_.rows != 0))
    return fail("draft vocabulary needs a row-addressable output head");
  if (prefix_rows == 0 || prefix_rows >= output_.rows)
    return fail("draft vocabulary prefix must leave rows out");
  if (output_.exl3 != nullptr && prefix_rows % 128 != 0)
    return fail("EXL3 draft prefix must cover complete Hadamard segments");
  for (const auto id : specials) {
    if (id < static_cast<std::int32_t>(prefix_rows) ||
        static_cast<std::uint32_t>(id) >= output_.rows)
      return fail("draft special tokens must lie above the prefix");
  }
  // The prefix is the head's own leading rows: no copy.
  draft_output_ = output_;
  draft_output_.rows = prefix_rows;
  draft_specials_ = std::move(specials);
  return true;
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& w, const core::GgufReader& reader,
    const MtpWeights* mtp, const core::GgufReader* mtp_reader,
    std::string* error_msg) {
  // The CPU reference also reads Q6_K, but the production embedding, dense
  // and routed kernels do not. Reject it before allocating device weights.
  const auto supported = [&](const TensorRef& t) {
    if (t.type != core::GgmlType::kQ6_K)
      return true;
    if (error_msg != nullptr)
      *error_msg = "unsupported HIP tensor format Q6_K: " + std::string(t.name);
    return false;
  };
  const auto layer_supported = [&](const LayerWeights& l) {
    return supported(l.ffn_gate_exps) && supported(l.ffn_up_exps) &&
           supported(l.ffn_down_exps);
  };
  if (!supported(w.token_embd) || !supported(w.output) ||
      !std::all_of(w.layers.begin(), w.layers.end(), layer_supported) ||
      (mtp != nullptr && !layer_supported(mtp->block))) {
    return nullptr;
  }
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = w.config;
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  const auto shard_count = static_cast<std::uint32_t>(shards.size());
  if (mtp != nullptr) {
    if (mtp_reader == nullptr) {
      if (error_msg)
        *error_msg = "MTP weights require their bound reader";
      return nullptr;
    }
    const auto extra = mtp_reader->GetMappedRegions();
    shards.insert(shards.end(), extra.begin(), extra.end());
  }
  const auto checkpoint_base = static_cast<std::uint32_t>(shards.size());
  if (w.exl3_storage) {
    const auto extra = w.exl3_storage->checkpoint->GetMappedRegions();
    shards.insert(shards.end(), extra.begin(), extra.end());
  }
  const auto mtp_checkpoint_base = static_cast<std::uint32_t>(shards.size());
  if (mtp != nullptr && mtp->exl3_storage) {
    const auto extra = mtp->exl3_storage->checkpoint->GetMappedRegions();
    shards.insert(shards.end(), extra.begin(), extra.end());
  }
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  // Measure first: the placement is decided once, from the bytes this walk
  // will request of the device heap.
  std::uint64_t measured = 0;
  {
    std::vector<Conversion> conversions;
    std::vector<void*> allocations;
    std::size_t bytes = 0;
    std::size_t max_half_cols = 1;
    std::size_t max_q8_cols = 32;
    std::deque<std::vector<quant::exl3::DeviceLinear>> descriptors;
    Uploader measure{*stager,     nullptr,   conversions,
                     allocations, bytes,     max_half_cols,
                     max_q8_cols, error_msg, descriptors};
    measure.checkpoint_base = checkpoint_base;
    measure.mtp_checkpoint_base = mtp_checkpoint_base;
    measure.exl3_model = w.exl3_storage != nullptr;
    (void)measure.Model(w, mtp, shard_count);
    if (!measure.ok) {
      return nullptr;
    }
    measured = measure.requested;
  }
  const hip::WeightAllocator weights(measured);
  std::vector<Conversion> conversions;
  Uploader up{*stager,         &weights,  conversions,
              m->allocations_, m->bytes_, m->max_half_cols_,
              m->max_q8_cols_, error_msg, m->exl3_linears_};
  up.checkpoint_base = checkpoint_base;
  up.mtp_checkpoint_base = mtp_checkpoint_base;
  up.exl3_model = w.exl3_storage != nullptr;
  Uploaded uploaded = up.Model(w, mtp, shard_count);
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  assert(up.requested == measured);
  m->token_embd_ = uploaded.token_embd;
  m->output_ = uploaded.output;
  m->output_bytes_ = w.output.SizeBytes();
  m->hc_head_ = uploaded.hc_head;
  m->layers_ = std::move(uploaded.layers);
  m->mtp_ = uploaded.mtp;
  m->has_mtp_ = mtp != nullptr;
  for (const auto& c : conversions) {
    NarrowActivations(static_cast<const float*>(c.source), c.destination, false,
                      c.count, nullptr);
  }
  for (const auto& c : up.q8_conversions)
    QuantizeHalfQ8_0(c.source, c.destination, c.count, nullptr);
  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg =
          "weight conversion failed: " + std::string(hipGetErrorString(status));
    }
    return nullptr;
  }
  for (const auto& c : conversions) {
    std::erase(m->allocations_, c.source);
    (void)hipFree(c.source);
    m->bytes_ -= c.count * sizeof(float) + kTailMargin;
  }
  for (const auto& c : up.q8_conversions) {
    if (!c.release_source)
      continue;
    std::erase(m->allocations_, c.source);
    (void)hipFree(c.source);
    m->bytes_ -= c.count * sizeof(std::uint16_t) + kTailMargin;
  }
  return m;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
