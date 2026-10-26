#ifndef GUFO_MODELS_QWEN_HIP_KVARN_CHECKPOINT_HPP_
#define GUFO_MODELS_QWEN_HIP_KVARN_CHECKPOINT_HPP_

// Which KVarN rows a checkpoint or compact snapshot holds (K4-C): records of
// the quantized groups, then only the valid FP16 rows of the sink and the
// tail. Host-only arithmetic shared by the continuation layout and the arena.

#include <algorithm>
#include <cstdint>
#include <optional>

#include "src/models/qwen/hip/kvarn_layout.hpp"

namespace gufo::hip {

// The KVarN rows a checkpoint at `position` holds: records of the groups
// before `base`, the sink, and the FP16 tail [base, position). `base` is the
// tail's flushed end, or the start of the group holding `position` when a
// rejected draft left the tail flushed past it (the next pass reopens that
// group from these rows); 128 before the first group.
struct KvarnCheckpointRows {
  std::uint64_t records{}, sink{}, tail{};
  bool operator==(const KvarnCheckpointRows&) const = default;
};
[[nodiscard]] constexpr std::uint64_t KvarnTailBase(std::uint64_t position,
                                                    std::uint64_t flushed_end) {
  if (position < kKvarnSinkTokens)
    return kKvarnSinkTokens;
  return std::min(flushed_end,
                  position >> kKvarnGroupShift << kKvarnGroupShift);
}
[[nodiscard]] constexpr KvarnCheckpointRows KvarnRowsAt(
    std::uint64_t position, std::uint64_t flushed_end) {
  if (position < kKvarnSinkTokens)
    return {.records = position, .sink = position, .tail = 0};
  const auto base = KvarnTailBase(position, flushed_end);
  return {.records = base, .sink = kKvarnSinkTokens, .tail = position - base};
}
// The tail of a slot restored from `rows` at `position`; empty when no
// checkpoint at `position` holds those rows.
[[nodiscard]] constexpr std::optional<KvarnTail> KvarnRestoredTail(
    std::uint64_t position, const KvarnCheckpointRows& rows) {
  const auto base = position < kKvarnSinkTokens
                        ? std::uint64_t{kKvarnSinkTokens}
                        : position - std::min(position, rows.tail);
  if (rows.tail > kKvarnTailRingTokens || base < kKvarnSinkTokens ||
      base % kKvarnGroupTokens != 0 || KvarnRowsAt(position, base) != rows)
    return std::nullopt;
  return KvarnTail{.flushed_end = static_cast<std::uint32_t>(base),
                   .written_end = static_cast<std::uint32_t>(position)};
}
}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KVARN_CHECKPOINT_HPP_
