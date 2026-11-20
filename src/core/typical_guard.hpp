#ifndef GUFO_CORE_TYPICAL_GUARD_HPP_
#define GUFO_CORE_TYPICAL_GUARD_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace gufo::sampling {

/// Loop guard of typical acceptance (SamplingConfig::typical_acceptance),
/// after Kyojin 9f1b368: a relaxed acceptance is refused when its token
/// would complete a 12-gram the reply already generated at least twice in
/// its last 2048 tokens. Such a refusal is a loop signal: the next 512
/// generated tokens use exact acceptance only, a window that doubles on each
/// later signal up to 16384.
///
/// Per-request host state, carried by SamplerState: Observe follows the
/// committed tokens (SamplerState::Accept), and the exact window is part of
/// the draw state a speculative step publishes (SamplerState::DrawState).
class TypicalGuard {
public:
  static constexpr std::size_t kNgram = 12;
  static constexpr std::size_t kHistory = 2048;
  static constexpr std::size_t kMinSeen = 2;
  static constexpr std::size_t kExactWindow = 512;
  static constexpr std::size_t kMaxExactWindow = 16384;

  /// The part a speculative step publishes: where exact acceptance ends and
  /// the next window's length.
  struct Cooldown {
    std::uint64_t exact_until{0};
    std::uint64_t next_window{kExactWindow};
    std::uint64_t signals{0};

    bool operator==(const Cooldown&) const = default;
  };

  void Reset() noexcept {
    ring_.clear();
    head_ = 0;
    generated_ = 0;
    cooldown_ = {};
  }

  void Observe(std::span<const std::uint32_t> tokens) {
    for (const auto token : tokens) {
      if (ring_.size() < kHistory) {
        ring_.push_back(token);
      } else {
        ring_[head_] = token;
        head_ = (head_ + 1) % kHistory;
      }
      ++generated_;
    }
  }

  [[nodiscard]] bool InExactWindow() const noexcept {
    return generated_ < cooldown_.exact_until;
  }

  /// Whether the next generated token may be accepted by the relaxed rule.
  /// A refusal by repetition starts or extends the exact window.
  [[nodiscard]] bool Allows(std::uint32_t token) {
    if (InExactWindow())
      return false;
    if (Seen(token) < kMinSeen)
      return true;
    cooldown_.exact_until = generated_ + cooldown_.next_window;
    cooldown_.next_window =
        std::min<std::uint64_t>(2 * cooldown_.next_window, kMaxExactWindow);
    ++cooldown_.signals;
    return false;
  }

  /// How often the window's last 11 tokens followed by `token` occur in the
  /// retained history (a 12-gram wholly inside the last kHistory tokens).
  [[nodiscard]] std::size_t Seen(std::uint32_t token) const noexcept {
    const std::size_t size = ring_.size();
    if (size < kNgram)
      return 0;
    // The candidate: the last kNgram - 1 tokens and `token`.
    const auto at = [&](std::size_t index) {
      return ring_[(head_ + index) % size];
    };
    const std::size_t tail = size - (kNgram - 1);
    std::size_t count = 0;
    for (std::size_t start = 0; start + kNgram <= size; ++start) {
      if (at(start + kNgram - 1) != token)
        continue;
      bool same = true;
      for (std::size_t i = 0; i + 1 < kNgram && same; ++i)
        same = at(start + i) == at(tail + i);
      count += same;
    }
    return count;
  }

  [[nodiscard]] std::uint64_t generated() const noexcept { return generated_; }
  [[nodiscard]] const Cooldown& cooldown() const noexcept { return cooldown_; }
  void set_cooldown(const Cooldown& cooldown) noexcept { cooldown_ = cooldown; }

private:
  /// The last kHistory generated tokens; once full, head_ is the oldest.
  std::vector<std::uint32_t> ring_;
  std::size_t head_{0};
  std::uint64_t generated_{0};
  Cooldown cooldown_;
};

}  // namespace gufo::sampling

#endif  // GUFO_CORE_TYPICAL_GUARD_HPP_
