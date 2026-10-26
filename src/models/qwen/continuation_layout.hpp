#ifndef GUFO_MODELS_QWEN_CONTINUATION_LAYOUT_HPP_
#define GUFO_MODELS_QWEN_CONTINUATION_LAYOUT_HPP_

#include <algorithm>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "src/cache/types.hpp"
#include "src/core/model_config.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/kvarn_layout.hpp"

namespace gufo::models::qwen {

enum class ContinuationPart {
  kMetadata,
  kLogits,
  kFrontier,
  kKey,
  kValue,
  kConvolution,
  kRecurrent,
  kDraftMetadata,
  kDraftKey,
  kDraftValue,
  kDraftFeatures,
  // KVarN: one attention layer's quantized group records as token rows (a
  // pool block's record bytes, KvarnRecord(bits).bytes * kv heads per 128
  // rows); the FP16 sink rows [0, min(position, 128)); and the FP16 tail
  // rows [base, position) after the quantized groups (KvarnCheckpointRows).
  // A sink or tail row holds one position of every layer, K then V.
  kKvarnRecords,
  kKvarnSink,
  kKvarnTail
};
struct ContinuationComponent {
  cache::ComponentDescriptor descriptor;
  ContinuationPart part;
  std::uint32_t layer{};
};

inline constexpr std::size_t kContinuationMetadataBytes = 64;
inline constexpr std::size_t kContinuationFrontierMetadataBytes = 96;
inline constexpr std::size_t kContinuationPendingTokens = 9;
// Layout version of the attention KV row components: FP16 K and V rows 1,
// FP32 2, KVarN 5 (k4v2) and 6 (k4v4). KVarN 3 and 4 kept the whole FP16
// sink and ring as one private component; their checkpoints are refused.
[[nodiscard]] constexpr std::uint32_t ContinuationKvLayoutVersion(
    hip::QwenKvCacheStorage storage) noexcept {
  switch (storage) {
    case hip::QwenKvCacheStorage::kFp16:
      return 1;
    case hip::QwenKvCacheStorage::kFp32:
      return 2;
    case hip::QwenKvCacheStorage::kKvarnK4V2:
      return 5;
    case hip::QwenKvCacheStorage::kKvarnK4V4:
      return 6;
  }
  return 0;
}

// The KVarN rows a checkpoint at `position` holds: records of the groups
// before `base`, the sink, and the FP16 tail [base, position). `base` is the
// tail's flushed end, or the start of the group holding `position` when a
// rejected draft left the tail flushed past it (the next pass reopens that
// group from these rows); 128 before the first group.
struct KvarnCheckpointRows {
  cache::Rows records{}, sink{}, tail{};
  bool operator==(const KvarnCheckpointRows&) const = default;
};
[[nodiscard]] constexpr cache::Rows KvarnTailBase(cache::Rows position,
                                                  cache::Rows flushed_end) {
  if (position < hip::kKvarnSinkTokens)
    return hip::kKvarnSinkTokens;
  return std::min(flushed_end,
                  position >> hip::kKvarnGroupShift << hip::kKvarnGroupShift);
}
[[nodiscard]] constexpr KvarnCheckpointRows KvarnRowsAt(
    cache::Rows position, cache::Rows flushed_end) {
  if (position < hip::kKvarnSinkTokens)
    return {.records = position, .sink = position, .tail = 0};
  const auto base = KvarnTailBase(position, flushed_end);
  return {
      .records = base, .sink = hip::kKvarnSinkTokens, .tail = position - base};
}
// The tail of a slot restored from `rows` at `position`; empty when no
// checkpoint at `position` holds those rows.
[[nodiscard]] constexpr std::optional<hip::KvarnTail> KvarnRestoredTail(
    cache::Rows position, const KvarnCheckpointRows& rows) {
  const auto base = position < hip::kKvarnSinkTokens
                        ? cache::Rows{hip::kKvarnSinkTokens}
                        : position - std::min(position, rows.tail);
  if (rows.tail > hip::kKvarnTailRingTokens || base < hip::kKvarnSinkTokens ||
      base % hip::kKvarnGroupTokens != 0 || KvarnRowsAt(position, base) != rows)
    return std::nullopt;
  return hip::KvarnTail{.flushed_end = static_cast<std::uint32_t>(base),
                        .written_end = static_cast<std::uint32_t>(position)};
}
// Tail rows a write of positions [first, end) can change, in the tail
// coordinates of a slot whose tail is flushed to `flushed_end`: only the
// written rows while the write stays inside the open group and moves no
// group end; otherwise every row, as the base moves.
[[nodiscard]] constexpr std::pair<cache::Rows, cache::Rows> KvarnTailWrite(
    cache::Rows flushed_end, cache::Rows first, cache::Rows end) {
  const auto floor = [](cache::Rows position) {
    return std::max<cache::Rows>(
        hip::kKvarnSinkTokens,
        position >> hip::kKvarnGroupShift << hip::kKvarnGroupShift);
  };
  if (first >= end)
    return {0, 0};
  if (flushed_end == floor(first) && floor(first) == floor(end - 1)) {
    if (end <= flushed_end)
      return {0, 0};
    return {first > flushed_end ? first - flushed_end : 0, end - flushed_end};
  }
  return {0, hip::kKvarnTailRingTokens};
}

struct ContinuationBackingClass {
  std::size_t block_bytes{}, blocks{}, metadata_bytes{};
};

// KV rows have a canonical token/head/channel order. FP32's native head-major
// planes are gathered without conversion; their layout version differs from
// FP16. The recurrent storage policy also has its own layout version.
class ContinuationLayout {
public:
  ContinuationLayout(const core::ModelConfig&, hip::QwenExecutionPolicy,
                     std::uint32_t context);
  [[nodiscard]] std::span<const ContinuationComponent> Components() const {
    return components_;
  }
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      cache::Rows) const;
  [[nodiscard]] std::size_t PrivateBytes() const noexcept;
  [[nodiscard]] std::size_t BytesPerToken() const noexcept;
  [[nodiscard]] std::vector<ContinuationBackingClass> BackingPlan(
      std::size_t ram_bytes) const;

private:
  std::uint32_t context_;
  std::vector<ContinuationComponent> components_;
};

}  // namespace gufo::models::qwen
#endif
