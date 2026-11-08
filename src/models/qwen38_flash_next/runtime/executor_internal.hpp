#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_RUNTIME_EXECUTOR_INTERNAL_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_RUNTIME_EXECUTOR_INTERNAL_HPP_

// Private to the executor*.cpp translation units: the helpers, routing
// thresholds and layouts more than one of them uses. Functions are inline so a
// translation unit that does not use one compiles without a warning.

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "qfn_mmq.h"
#include "src/core/hip/kv_block_layout.hpp"
#include "src/core/hip/managed_alloc.hpp"
#include "src/core/hip/snapshot_transfer.hpp"
#include "src/core/platform/pages.hpp"
#include "src/core/quant/exl3/gemv.hpp"
#include "src/models/qwen38_flash_next/draft_vocab.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/runtime/executor.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

namespace {

using core::GgmlType;

inline void AssignError(std::string* error_msg, const std::string& message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

inline bool Check(hipError_t err, const char* what, std::string* error_msg) {
  if (err == hipSuccess) {
    return true;
  }
  AssignError(error_msg, std::string(what) + ": " + hipGetErrorString(err));
  return false;
}

/// Allocates `count` elements of T and records the allocation.
template<typename T>
T* Alloc(std::vector<void*>& allocations, std::size_t count,
         std::string* error_msg, std::size_t* allocated_bytes = nullptr) {
  void* p = nullptr;
  const std::size_t bytes = std::max<std::size_t>(1, count) * sizeof(T);
  if (hip::AllocateDevice(&p, bytes) != hipSuccess) {
    AssignError(error_msg,
                "allocation of " + std::to_string(bytes) + " bytes failed");
    allocations.push_back(nullptr);
    return nullptr;
  }
  (void)hipMemset(p, 0, bytes);
  allocations.push_back(p);
  if (allocated_bytes)
    *allocated_bytes += bytes;
  return static_cast<T*>(p);
}

inline WeightType SmallType(GgmlType type) {
  switch (type) {
    case GgmlType::kBF16:
      return WeightType::kBF16;
    case GgmlType::kF16:
      return WeightType::kF16;
    case GgmlType::kQ8_0:
      return WeightType::kQ8_0;
    case GgmlType::kF32:
      return WeightType::kF32;
    default:
      throw std::logic_error("unsupported Flash-Next matrix format");
  }
}

// The tier's tiled kernels compute whole column tiles; below this width the
// matrix-vector kernels read each weight once per row and win outright.
constexpr std::uint32_t kVecBatch = 8;
constexpr std::uint32_t kExl3PrefillTileRows = 64;

struct Exl3PrefillLayout {
  std::uint32_t assignments{0}, buckets{0};
  std::size_t bytes{0};
};

inline bool PlanExl3Prefill(const Config& c, std::uint32_t rows,
                            Exl3PrefillLayout* layout, std::string* error) {
  if (rows == 0 || c.num_experts_used == 0 ||
      c.num_experts_used > c.num_experts || c.hidden_size == 0 ||
      c.expert_ff == 0) {
    AssignError(error, "invalid EXL3 prefill scratch geometry");
    return false;
  }
  const std::uint64_t assignments = std::uint64_t{rows} * c.num_experts_used;
  const auto buckets =
      std::min(assignments, std::uint64_t{c.num_experts} +
                                assignments / kExl3PrefillTileRows);
  if (assignments > std::numeric_limits<std::uint32_t>::max() ||
      buckets > 65535) {
    AssignError(error, "EXL3 prefill scratch exceeds device descriptor limits");
    return false;
  }
  std::size_t bytes = 0;
  const auto add = [&](std::uint64_t count, std::size_t element) {
    if (count > (std::numeric_limits<std::size_t>::max() - bytes) / element)
      return false;
    bytes += static_cast<std::size_t>(count) * element;
    return true;
  };
  if (!add(std::uint64_t{c.num_experts} + 1, sizeof(std::uint32_t)) ||
      !add(assignments, sizeof(std::int32_t)) ||
      !add(buckets, sizeof(Exl3ExpertBucket)) ||
      !add(1, sizeof(std::uint32_t)) ||
      !add(assignments * std::max(c.hidden_size, c.expert_ff),
           sizeof(std::uint16_t)) ||
      !add(assignments * c.hidden_size, sizeof(float)) ||
      !add(assignments, sizeof(std::int32_t)) ||
      !add(assignments, sizeof(std::int32_t))) {
    AssignError(error, "EXL3 prefill scratch byte size overflows size_t");
    return false;
  }
  *layout = {static_cast<std::uint32_t>(assignments),
             static_cast<std::uint32_t>(buckets), bytes};
  return true;
}

}  // namespace

/// Keeps prompt arithmetic independent of chunk width. A scoped, per-thread
/// policy also lets the output head retain its existing logits-row
/// arithmetic. Defined in executor.cpp; PrefillPhase sets it.
extern constinit thread_local bool prefill_phase;

namespace {

struct PrefillPhase {
  bool previous;
  explicit PrefillPhase(bool enabled) : previous(prefill_phase) {
    prefill_phase = enabled;
  }
  ~PrefillPhase() { prefill_phase = previous; }
};
inline bool MatrixRows(std::uint32_t rows) {
  return prefill_phase || rows > kVecBatch;
}
inline bool ExpertMatrixRows(std::uint32_t rows) {
  return prefill_phase || rows > 4 * kVecBatch;
}
/// Prefill passes wider than a vector batch take the fused projections and
/// the paired routed expert tiles: they win from 9 rows up to the full chunk.
/// Decode and verification batches keep their single-token arithmetic.
inline bool WidePrefill(std::uint32_t rows) {
  return prefill_phase && rows > kVecBatch;
}

/// Whether mmq's Q8_1 rows of width k carry no padding, so a producer's
/// contiguous element order is the layout.
inline bool UnpaddedQ8(std::uint32_t k) {
  return k % 32 == 0 &&
         qfn_mmq_q8_1_bytes(1, static_cast<int>(k)) == k / 32 * 36;
}

/// Key-tile splits per row of a narrow attention batch (decode at depth).
constexpr std::uint32_t kAttnSplits = 8;
/// Wide dense Q8_0 projections take the F16 WMMA GEMM (F16 activation
/// rows, weights dequantized as they are staged, no per-block scaling) when
/// its 256-row tiles fill the device and the K sweep is short enough that
/// the F16 rows stay cache-resident: measured per shape at 2048 tokens
/// (W8A8 -> F16 ms): 16384x2560 5.7 -> 4.6, 13312x2560 4.3 -> 3.9,
/// 10240x320 1.02 -> 0.79, 2560x2560 0.81 -> 0.74, 2560x640 0.24 -> 0.21;
/// against 2560x6144 1.8 -> 2.1+, 640x2560 0.19 -> 0.23 and 320x10240
/// 0.40 -> 1.3, which keep the int8 route.
constexpr std::uint32_t kDenseF16MinRows = 2048;
constexpr std::uint32_t kDenseF16MaxCols = 2560;

/// Token rows per routed F16 expert GEMM tile: the narrow tile for batches
/// whose buckets pad to one or two 16-row tiles, the wide one when the
/// mean bucket fills most of it (the weight dequantization is per tile).
constexpr std::uint32_t kRoutedTileRowsNarrow = 16;
constexpr std::uint32_t kRoutedTileRowsWide = 48;

inline std::uint32_t RoutedTileRows(std::size_t slots, const Config& c) {
  return slots >= static_cast<std::size_t>(16) * c.num_experts
             ? kRoutedTileRowsWide
             : kRoutedTileRowsNarrow;
}

/// Upper bound on launched routed tiles: every 16-padded bucket contributes
/// at most one partial tile beyond its rows.
inline std::size_t RoutedTileCapacity(std::size_t slots, const Config& c) {
  return (slots + static_cast<std::size_t>(c.num_experts) * 15) /
             kRoutedTileRowsNarrow +
         c.num_experts + 1;
}

inline std::uint32_t IndexerCapacity(const Config& c, std::uint32_t batch,
                                     std::uint32_t context) {
  return static_cast<std::uint32_t>(std::bit_ceil(std::min<std::uint64_t>(
      context, std::uint64_t{c.indexer_top_k} + batch)));
}

/// K, V and pooled indexer keys of every attention layer, then the draft
/// block's, as KV block planes (kernels.hpp kKvPooledRatio). The blocks cover
/// `max_context` tokens and max_context / ratio + 1 pooled rows, the pooled
/// capacity the snapshot and continuation guards address.
/// With KVarN the trunk layers' K/V are one record plane of k4v4 token-major
/// groups (two a block) and only the draft block keeps FP16 K/V planes.
struct SessionKvLayout {
  hip::KvBlockGeometry geometry;
  std::size_t keys{0}, values{0}, pooled{0}, records{0};
  bool fp16_trunk{true};
};
inline SessionKvLayout SessionKvPlanes(const Config& c, bool mtp, bool kvarn,
                                       std::uint32_t max_context, int bits) {
  const std::uint32_t attention = c.num_layers / c.full_attention_interval;
  const std::size_t kv_row = std::size_t{c.AttentionKvDim()} * sizeof(__half);
  SessionKvLayout out;
  out.fp16_trunk = !kvarn;
  std::vector<hip::KvBlockPlane> planes;
  const auto add = [&](hip::KvBlockPlane plane) {
    planes.push_back(plane);
    return planes.size() - 1;
  };
  if (kvarn)
    out.records =
        add(hip::KvarnRecordPlane(attention, c.num_kv_heads, bits,
                                  hip::KvarnValueOrder::kTokenMajor, bits));
  const std::uint32_t fp16_layers = (kvarn ? 0 : attention) + (mtp ? 1 : 0);
  if (fp16_layers != 0) {
    out.keys = add({fp16_layers, kv_row, hip::kKvBlockTokens});
    out.values = add({fp16_layers, kv_row, hip::kKvBlockTokens});
  }
  out.pooled = add({attention + (mtp ? 1 : 0),
                    std::size_t{c.indexer_head_dim} * sizeof(__half),
                    hip::kKvBlockTokens / kKvPooledRatio});
  out.geometry = hip::KvBlockGeometry(
      std::move(planes), hip::KvBlockGeometry::BlocksFor(
                             std::size_t{max_context} + kKvPooledRatio));
  return out;
}

/// Halves of one KVarN session's FP16 sink and tail rows: K and V of every
/// attention layer.
inline std::size_t KvarnFp16Elements(const Config& c) {
  return std::size_t{2} * (c.num_layers / c.full_attention_interval) *
         hip::kKvarnFp16Tokens * c.AttentionKvDim();
}

}  // namespace

/// Column-tile width for the routed expert GEMMs (executor_dense.cpp).
int RoutedTileCols(std::uint32_t n_tokens, std::uint32_t n_used,
                   std::uint32_t n_experts);

namespace {

/// A mixer's F32 inject weights, which its combine can apply (see Combine).
inline const float* InjectWeights(const DeviceMixer& m) {
  return !m.inject.empty() && m.inject.type == GgmlType::kF32 ? m.inject.f32()
                                                              : nullptr;
}

}  // namespace

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_RUNTIME_EXECUTOR_INTERNAL_HPP_
