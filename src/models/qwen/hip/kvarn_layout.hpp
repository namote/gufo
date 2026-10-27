#ifndef GUFO_MODELS_QWEN_HIP_KVARN_LAYOUT_HPP_
#define GUFO_MODELS_QWEN_HIP_KVARN_LAYOUT_HPP_

// KVarN attention KV storage for Qwen3.8-27B (plan/gufo-upstream-port/
// KVARN.md): Hadamard-rotated keys and values, the first 128 tokens of a
// sequence (the sink) and its unquantized tail in FP16, every other complete
// 128-token group quantized (4-bit keys, 2- or 4-bit values). Host-only
// arithmetic, so the sizes, the split policy and the tail bookkeeping are
// testable without a GPU.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "src/core/hip/kv_block_layout.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"

namespace gufo::hip {

/// Tokens per quantized group; a pool block holds two.
inline constexpr std::uint32_t kKvarnGroupTokens = 128;
inline constexpr std::uint32_t kKvarnGroupShift = 7;
static_assert(kKvarnGroupTokens == 1U << kKvarnGroupShift);
inline constexpr std::uint32_t kKvarnGroupsPerBlock =
    kKvBlockTokens / kKvarnGroupTokens;
static_assert(kKvarnGroupsPerBlock == 2);
/// The sink is group 0: FP16 for the whole sequence.
inline constexpr std::uint32_t kKvarnSinkTokens = kKvarnGroupTokens;
/// FP16 rows past the last quantized group, held in a ring of this many
/// tokens per layer. A pass may write at most RingTokens - 127 rows (every
/// complete group before it is quantized first).
inline constexpr std::uint32_t kKvarnTailRingTokens = 256;
static_assert(kKvarnTailRingTokens % kKvarnGroupTokens == 0);
/// The only shape the KVarN kernels serve.
inline constexpr std::uint32_t kKvarnHeadDim = 256;
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

/// One (group, KV head) record: byte offsets of its parts.
///   K codes  [token 128][128 B]: 256 channels x 4 bit, token-major; a
///            32-bit word holds channels c..c+7, c+2i in nibble i and c+2i+1
///            in nibble i+4.
///   V codes  [channel 256][128 x b bit]: channel-major; a word holds 16
///            tokens (2 bit: token 2i in crumb i, 2i+1 in crumb i+8) or 8
///            tokens (4 bit: the K nibble order).
///   K scale, K zero [256] (per channel), K column [128] (per token),
///   V scale, V zero [128] (per token), V column [256] (per channel), FP16.
/// The value is (code * scale + zero) * column, in the rotated frame.
struct KvarnRecordLayout {
  std::size_t k_codes{0};
  std::size_t v_codes{0};
  std::size_t v_row_bytes{0};
  std::size_t k_scale{0};
  std::size_t k_zero{0};
  std::size_t k_column{0};
  std::size_t v_scale{0};
  std::size_t v_zero{0};
  std::size_t v_column{0};
  std::size_t bytes{0};
};

[[nodiscard]] constexpr KvarnRecordLayout KvarnRecord(int value_bits) {
  KvarnRecordLayout r;
  r.k_codes = 0;
  r.v_codes = r.k_codes + kKvarnGroupTokens * kKvarnHeadDim / 2;
  r.v_row_bytes = static_cast<std::size_t>(kKvarnGroupTokens) * value_bits / 8;
  r.k_scale = r.v_codes + kKvarnHeadDim * r.v_row_bytes;
  r.k_zero = r.k_scale + kKvarnHeadDim * 2;
  r.k_column = r.k_zero + kKvarnHeadDim * 2;
  r.v_scale = r.k_column + kKvarnGroupTokens * 2;
  r.v_zero = r.v_scale + kKvarnGroupTokens * 2;
  r.v_column = r.v_zero + kKvarnGroupTokens * 2;
  r.bytes = r.v_column + kKvarnHeadDim * 2;
  return r;
}
static_assert(KvarnRecord(2).bytes == 26880 && KvarnRecord(4).bytes == 35072);

/// The pool plane of quantized groups: per layer and block,
/// [group in block 2][KV head][record], one plane row per group.
[[nodiscard]] constexpr KvBlockPlane KvarnRecordPlane(std::uint32_t layers,
                                                      std::uint32_t kv_heads,
                                                      int value_bits) {
  return {.layers = layers,
          .row_bytes = static_cast<std::size_t>(kv_heads) *
                       KvarnRecord(value_bits).bytes,
          .rows_per_block = kKvarnGroupsPerBlock};
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

/// Token rows per layer of a sequence's FP16 sink and tail plane.
inline constexpr std::uint32_t kKvarnFp16Tokens =
    kKvarnSinkTokens + kKvarnTailRingTokens;

/// Row of `position` in the FP16 plane: the sink, then the tail ring.
[[nodiscard]] constexpr std::uint32_t KvarnFp16Row(std::uint32_t position) {
  return position < kKvarnSinkTokens
             ? position
             : kKvarnSinkTokens +
                   (position - kKvarnSinkTokens) % kKvarnTailRingTokens;
}

/// Bytes of a sequence's FP16 sink and tail rows (K and V, every layer).
[[nodiscard]] constexpr std::size_t KvarnFp16Bytes(std::uint32_t layers,
                                                   std::uint32_t kv_heads,
                                                   std::uint32_t head_dim) {
  return std::size_t{2} * layers * kKvarnFp16Tokens * kv_heads * head_dim *
         sizeof(std::uint16_t);
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

/// Quantized groups the query at `position` reads: every group that ended
/// at or before it. The rule depends on the position only, so decode,
/// verification and a replay after a rewind read the same values.
[[nodiscard]] constexpr std::uint32_t KvarnReadGroups(std::uint32_t position) {
  const std::uint32_t groups = position >> kKvarnGroupShift;
  return groups > 1 ? groups - 1 : 0;
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

/// FP16 tail of one sequence: positions [kKvarnSinkTokens, flushed_end) are
/// quantized groups, [flushed_end, ...) FP16 rows in the ring. written_end
/// bounds every position written since the last reset.
struct KvarnTail {
  std::uint32_t flushed_end{kKvarnSinkTokens};
  std::uint32_t written_end{0};
};

/// Most rows of one pass: a pass meets at most one group end.
inline constexpr std::uint32_t kKvarnMaxPassRows = kKvarnGroupTokens;

/// What a pass writing rows [start, start + rows) does to the tail. Before
/// the pass, groups [first_group, first_group + groups) are quantized, so
/// that the rows before `boundary` read `read_groups` groups (KvarnReadGroups
/// of `start`). When a group ends inside the pass, at `boundary` (zero when
/// none does), each layer quantizes it once its rows are written, and rows
/// from `boundary` on read read_groups + 1. flushed_end is the tail after
/// the pass. A pass starting inside a quantized group returns it to FP16
/// first, while the ring still holds its rows. `error` names a pass that
/// cannot run.
struct KvarnPassPlan {
  std::uint32_t first_group{0};
  std::uint32_t groups{0};
  std::uint32_t read_groups{0};
  std::uint32_t boundary{0};
  std::uint32_t flushed_end{kKvarnSinkTokens};
  std::string_view error;
};

[[nodiscard]] constexpr KvarnPassPlan PlanKvarnPass(const KvarnTail& tail,
                                                    std::uint32_t start,
                                                    std::uint32_t rows) {
  constexpr std::uint32_t kRing = kKvarnTailRingTokens;
  const auto group_floor = [](std::uint32_t position) {
    return std::max(kKvarnSinkTokens,
                    position >> kKvarnGroupShift << kKvarnGroupShift);
  };
  KvarnPassPlan plan;
  if (rows == 0 || rows > kKvarnMaxPassRows) {
    plan.error = "KVarN passes write 1 to 128 rows";
    return plan;
  }
  std::uint32_t flushed = tail.flushed_end;
  if (start < flushed) {
    const std::uint32_t reopened = group_floor(start);
    if (reopened + kRing < tail.written_end) {
      plan.error =
          "KVarN cannot rewind into a quantized group whose FP16 rows the "
          "tail ring no longer holds";
      return plan;
    }
    flushed = reopened;
  }
  const std::uint32_t end = start + rows;
  const std::uint32_t target = group_floor(start);
  if (end > target && end - target > kRing) {
    plan.error = "KVarN pass writes more rows than the FP16 tail ring holds";
    return plan;
  }
  const std::uint32_t last_end = group_floor(end - 1);
  plan.first_group = flushed >> kKvarnGroupShift;
  plan.groups = (target - flushed) >> kKvarnGroupShift;
  plan.read_groups = KvarnReadGroups(start);
  plan.boundary =
      last_end > start && last_end > kKvarnSinkTokens ? last_end : 0;
  plan.flushed_end = plan.boundary != 0 ? plan.boundary : target;
  return plan;
}

/// What a prompt chunk writing rows [start, start + rows) does to the tail.
/// Before it, groups [first_group, first_group + groups) are quantized from
/// the ring, so the chunk reads read_groups quantized groups (those that
/// ended at or before `start`), FP16 rows [target, start) from the ring and
/// its own rows. Each layer then quantizes the groups that complete inside
/// the chunk from a staging copy and keeps the rest in the ring; flushed_end
/// is the tail after the chunk. Prompt rows are confirmed, so the chunk's
/// own groups are quantized at once. `error` names a chunk that cannot run.
struct KvarnPrefillPlan {
  std::uint32_t first_group{0};
  std::uint32_t groups{0};
  std::uint32_t read_groups{0};
  std::uint32_t target{kKvarnSinkTokens};
  std::uint32_t flushed_end{kKvarnSinkTokens};
  std::string_view error;
};

[[nodiscard]] constexpr KvarnPrefillPlan PlanKvarnPrefill(const KvarnTail& tail,
                                                          std::uint32_t start,
                                                          std::uint32_t rows) {
  const auto group_floor = [](std::uint32_t position) {
    return std::max(kKvarnSinkTokens,
                    position >> kKvarnGroupShift << kKvarnGroupShift);
  };
  KvarnPrefillPlan plan;
  if (rows == 0) {
    plan.error = "KVarN prompt chunks write at least one row";
    return plan;
  }
  std::uint32_t flushed = tail.flushed_end;
  if (start < flushed) {
    const std::uint32_t reopened = group_floor(start);
    if (reopened + kKvarnTailRingTokens < tail.written_end) {
      plan.error =
          "KVarN cannot rewind into a quantized group whose FP16 rows the "
          "tail ring no longer holds";
      return plan;
    }
    flushed = reopened;
  }
  plan.target = group_floor(start);
  plan.first_group = flushed >> kKvarnGroupShift;
  plan.groups = (plan.target - flushed) >> kKvarnGroupShift;
  plan.read_groups = KvarnReadGroups(start);
  plan.flushed_end = group_floor(start + rows);
  return plan;
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

/// Staging rows of one layer a prompt chunk of `rows` rows can need: the
/// ring's unquantized rows before it (fewer than one group) and its own.
[[nodiscard]] constexpr std::size_t KvarnStagingRows(std::uint32_t rows) {
  return std::size_t{kKvarnGroupTokens} + rows;
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KVARN_LAYOUT_HPP_
