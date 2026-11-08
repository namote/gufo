#ifndef GUFO_MODELS_QWEN_HIP_KVARN_LAYOUT_HPP_
#define GUFO_MODELS_QWEN_HIP_KVARN_LAYOUT_HPP_

// KVarN attention KV storage for Qwen3.8-27B (plan/archive/gufo-upstream-port/
// KVARN.md) on the shared format of src/core/hip/kvarn_layout.hpp: this
// model's kernel shape (4 KV heads, 24 query heads, channel-major values),
// pool planes and tuned split and prefill policies. Host-only arithmetic, so
// the sizes and the split policy are testable without a GPU.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/core/hip/kv_block_layout.hpp"
#include "src/core/hip/kvarn_layout.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"

namespace gufo::hip {

/// The only shape the Qwen3.8-27B KVarN kernels serve (head width
/// kKvarnHeadDim, src/core/hip/kvarn_layout.hpp).
inline constexpr std::uint32_t kKvarnKvHeads = 4;
inline constexpr std::uint32_t kKvarnQueryHeads = 24;
/// Query rows of one attention launch (decode 1, DFlash2 verification 1-8).
inline constexpr std::uint32_t kKvarnMaxRows = 8;
/// Largest quantized split count of one launch (SelectKvarnDecodeSplits
/// picks at most 20; scratch is sized for this bound).
inline constexpr std::uint32_t kKvarnMaxSplits = 38;

/// Value code bits of a KVarN storage; 0 for FP16/FP32 rows.
[[nodiscard]] constexpr int KvarnValueBits(QwenKvCacheStorage storage) {
  return storage == QwenKvCacheStorage::kKvarnK4V2   ? 2
         : storage == QwenKvCacheStorage::kKvarnK4V4 ? 4
                                                     : 0;
}

/// Planes of a Qwen attention KV block pool for `storage`: FP16 K and V
/// rows, one KVarN record plane, none for the FP32 reference (no pool).
[[nodiscard]] inline std::vector<KvBlockPlane> QwenKvBlockPlanes(
    std::uint32_t layers, std::uint32_t kv_heads, std::uint32_t head_dim,
    QwenKvCacheStorage storage) {
  if (const int bits = KvarnValueBits(storage); bits != 0)
    return {KvarnRecordPlane(layers, kv_heads, bits)};
  if (storage != QwenKvCacheStorage::kFp16)
    return {};
  const KvBlockPlane rows{.layers = layers,
                          .row_bytes = static_cast<std::size_t>(kv_heads) *
                                       head_dim * sizeof(std::uint16_t)};
  return {rows, rows};
}

/// A shared Qwen KV block pool for `planes`: block bytes, the blocks one
/// `max_context` session needs, and the pool's blocks: `pool_bytes` worth,
/// or `sessions` full sessions when it is zero. The same bytes hold about
/// 4.9 (k4v2) or 3.7 (k4v4) times the FP16 tokens.
struct QwenKvPoolSize {
  std::size_t block_bytes{0};
  std::size_t session_blocks{0};
  std::size_t blocks{0};
};
[[nodiscard]] inline QwenKvPoolSize SizeQwenKvPool(
    const std::vector<KvBlockPlane>& planes, std::uint32_t max_context,
    std::size_t sessions, std::size_t pool_bytes) {
  QwenKvPoolSize size;
  size.block_bytes = KvBlockGeometry(planes, 1).bytes_per_block();
  size.session_blocks = KvBlockGeometry::BlocksFor(max_context);
  size.blocks = pool_bytes != 0 ? pool_bytes / size.block_bytes
                                : sessions * size.session_blocks;
  return size;
}

/// Quantized splits of one KVarN attention launch over `groups` quantized
/// groups; the sink and the tail add one split each. From the K4-P2 sweep
/// of the production kernel (KVARN.md "K4-P2 결과", splits 4-38 at 2K-256K,
/// one and eight query rows): 20 is fastest or within noise of it from 12K
/// to 256K in both formats; shallower, the FP16 sink and tail blocks and the
/// combine set the time, and fewer, longer splits win. The count depends on
/// the groups only, never on the query rows, so a verification row adds its
/// partials in the order a decode step at its position does.
[[nodiscard]] constexpr std::uint32_t SelectKvarnDecodeSplits(
    std::uint32_t groups, int value_bits) {
  (void)value_bits;
  if (groups == 0)
    return 0;
  const std::uint32_t splits = groups < 24   ? 6
                               : groups < 48 ? 8
                               : groups < 88 ? 10
                                             : 20;
  return std::min(splits, groups);
}

/// FP16 blocks of one KVarN attention launch: the sink in two halves, the
/// tail after the quantized groups in blocks of 64 positions (fewer than
/// one group plus the 128 rows of a pass).
inline constexpr std::uint32_t kKvarnSinkBlocks = 2;
inline constexpr std::uint32_t kKvarnMaxTailBlocks = 4;

/// Floats of KVarN split partials for `rows` query rows and `splits`
/// quantized splits: [KV head][split][row x GQA][max, sum, head_dim].
[[nodiscard]] constexpr std::size_t KvarnScratchElements(std::uint32_t rows,
                                                         std::uint32_t splits) {
  return std::size_t{kKvarnQueryHeads} * rows *
         (splits + kKvarnSinkBlocks + kKvarnMaxTailBlocks) *
         (kKvarnHeadDim + 2);
}

/// KV heads one prefill pack holds at a time: K [length][256] and V padded to
/// 16 positions, FP16, per head, in workspaces of these bytes.
[[nodiscard]] constexpr std::uint32_t KvarnPrefillHeadsPerGroup(
    std::uint32_t length, std::size_t key_workspace_bytes,
    std::size_t value_workspace_bytes) {
  const std::size_t key_head_bytes =
      std::size_t{length} * kKvarnHeadDim * sizeof(std::uint16_t);
  const std::size_t value_head_bytes =
      ((std::size_t{length} + 15U) & ~std::size_t{15}) * kKvarnHeadDim *
      sizeof(std::uint16_t);
  return static_cast<std::uint32_t>(std::min(
      {std::size_t{kKvarnKvHeads}, key_workspace_bytes / key_head_bytes,
       value_workspace_bytes / value_head_bytes}));
}

/// Whether a prompt chunk of `rows` rows runs as a pass (PlanKvarnPass:
/// split-K attention over the records in launches of eight rows, each row
/// as a decode step) instead of packing the prefix for the prefill WMMA,
/// given the KV heads one pack holds. Below 128 rows the WMMA has too few
/// blocks to fill the GPU, and when the heads do not all fit (about 190K
/// positions and deeper) its launches run one head group after another.
/// From the K5 sweep (KVARN.md "K5 결과", 1-512 rows at 0-256K, both
/// formats): a pass is faster up to 64 rows at every depth and up to 128
/// rows once the heads no longer fit; at 128 rows with every head in one
/// pack the WMMA is as fast or faster (up to 8 % at 128K).
[[nodiscard]] constexpr bool UseKvarnPrefillPass(
    std::uint32_t rows, std::uint32_t heads_per_group) {
  return rows <= 64 ||
         (rows <= kKvarnMaxPassRows && heads_per_group < kKvarnKvHeads);
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KVARN_LAYOUT_HPP_
