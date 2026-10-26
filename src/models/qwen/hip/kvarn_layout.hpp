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
/// Largest quantized split count of SelectKvarnDecodeSplits.
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
/// groups; the sink and the tail add one split each. From the K1b sweep
/// (KVARN.md "K1a·K1b 결과", splits 18/32/38/64/128 at 4K-256K): 18 fills
/// whole rounds of resident blocks up to 64K, 38 beyond, and K4V4 reads
/// enough per group to prefer 38 from 12K. The count depends on the groups
/// only, never on the query rows, so a verification row adds its partials
/// in the order a decode step at its position does.
[[nodiscard]] constexpr std::uint32_t SelectKvarnDecodeSplits(
    std::uint32_t groups, int value_bits) {
  if (groups == 0)
    return 0;
  std::uint32_t splits = groups < 512 ? 18 : kKvarnMaxSplits;
  if (value_bits == 4 && groups >= 96)
    splits = kKvarnMaxSplits;
  return std::min(splits, groups);
}

/// Quantized groups the query at `position` reads: every group that ended
/// at or before it. The rule depends on the position only, so decode,
/// verification and a replay after a rewind read the same values.
[[nodiscard]] constexpr std::uint32_t KvarnReadGroups(std::uint32_t position) {
  const std::uint32_t groups = position >> kKvarnGroupShift;
  return groups > 1 ? groups - 1 : 0;
}

/// Floats of KVarN split partials for `rows` query rows and `splits`
/// quantized splits: [KV head][split + 2][row x GQA][max, sum, head_dim].
[[nodiscard]] constexpr std::size_t KvarnScratchElements(std::uint32_t rows,
                                                         std::uint32_t splits) {
  return std::size_t{kKvarnQueryHeads} * rows * (splits + 2) *
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

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KVARN_LAYOUT_HPP_
