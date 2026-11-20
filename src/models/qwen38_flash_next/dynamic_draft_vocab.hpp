#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_DYNAMIC_DRAFT_VOCAB_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_DYNAMIC_DRAFT_VOCAB_HPP_

// Frequency-ranked dynamic draft vocabulary for the MTP head (the idea of
// exllamav3 47d42fe, sparse_head.py; plan/upstream-sync2 V2-7). Not wired into
// the engine yet: drafting through a ranked subset needs a gathered row
// buffer (the subset is not a prefix), a device remap from subset index to
// token ID and a 4-bit copy of those rows (DeviceModel::SetDraftVocab covers
// only the static prefix). This header holds the host-side policy so it can
// be tested and measured first:
//
//   - MtpDraftFrequency ranks tokens by decayed counts (prompt tokens weigh
//     0.2) and rebuilds its subset after each prompt and every 128
//     generated tokens, replacing a member only for a candidate counted 25%
//     higher;
//   - MtpDraftSubsetSwitch uses the subset only while the attributable miss
//     rate stays below the break-even 2 * saving / round time, and probes the
//     full head every 128 rounds.
//
// Everything is a function of one session's own history, so seeded sampled
// output stays independent of other requests.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace gufo::models::qwen38_flash_next {

class MtpDraftFrequency {
public:
  static constexpr float kPromptWeight = 0.2F;
  static constexpr float kGeneratedWeight = 1.0F;
  /// Observed tokens between rebuilds.
  static constexpr std::size_t kRebuildTokens = 128;
  /// A candidate replaces a member only when counted this much higher.
  static constexpr float kHysteresis = 1.25F;
  /// Per-rebuild decay of every count.
  static constexpr float kDefaultDecay = 0.98F;

  /// `subset_rows`: the subset's size. `vocab`: token IDs are [0, vocab).
  MtpDraftFrequency(std::uint32_t subset_rows, std::uint32_t vocab,
                    float decay = kDefaultDecay)
      : rows_(subset_rows), vocab_(vocab), decay_(decay) {}

  void Reset() {
    counts_.clear();
    epochs_.clear();
    seen_.clear();
    counted_.clear();
    subset_.clear();
    member_.clear();
    epoch_ = 0;
    pending_ = 0;
    version_ = 0;
  }

  /// Counts prompt tokens (weight 0.2) or generated ones (weight 1). Returns
  /// whether the subset was rebuilt. Generated tokens rebuild every
  /// kRebuildTokens; a prompt is one step, rebuilt once after it, so a long
  /// prompt does not decay its own early tokens away.
  bool Observe(std::span<const std::int32_t> tokens, bool prompt) {
    const float weight = prompt ? kPromptWeight : kGeneratedWeight;
    bool rebuilt = false, counted = false;
    for (const auto id : tokens) {
      if (id < 0 || static_cast<std::uint32_t>(id) >= vocab_)
        continue;
      Add(static_cast<std::uint32_t>(id), weight);
      counted = true;
      if (!prompt && ++pending_ >= kRebuildTokens) {
        Rebuild();
        rebuilt = true;
      }
    }
    if (prompt && counted) {
      Rebuild();
      rebuilt = true;
    }
    return rebuilt;
  }

  /// The decayed count of `id` now.
  [[nodiscard]] float Count(std::int32_t id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= counts_.size())
      return 0.0F;
    return Decayed(static_cast<std::uint32_t>(id));
  }
  [[nodiscard]] bool Contains(std::int32_t id) const {
    return id >= 0 && static_cast<std::size_t>(id) < member_.size() &&
           member_[static_cast<std::size_t>(id)];
  }
  /// Subset members, unordered; stable between rebuilds.
  [[nodiscard]] std::span<const std::int32_t> Subset() const noexcept {
    return subset_;
  }
  /// Increments whenever a rebuild changes the members.
  [[nodiscard]] std::uint64_t Version() const noexcept { return version_; }

  /// Ranks now: fills free rows with the most counted tokens, then swaps a
  /// member for a candidate only when the candidate is counted kHysteresis
  /// times higher. Applies one step of decay first.
  void Rebuild() {
    pending_ = 0;
    ++epoch_;
    using Ranked = std::pair<float, std::int32_t>;
    // Highest count first; ties by lower ID (earlier BPE merges).
    const auto higher = [](const Ranked& a, const Ranked& b) {
      return a.first != b.first ? a.first > b.first : a.second < b.second;
    };
    std::vector<Ranked> candidates;
    for (const auto id : seen_)
      if (!Contains(static_cast<std::int32_t>(id)))
        candidates.push_back({Decayed(id), static_cast<std::int32_t>(id)});
    // At most rows_ candidates can enter.
    const auto useful = std::min<std::size_t>(candidates.size(), rows_);
    std::partial_sort(candidates.begin(), candidates.begin() + useful,
                      candidates.end(), higher);
    candidates.resize(useful);
    bool changed = false;
    std::size_t next = 0;
    for (; next < candidates.size() && subset_.size() < rows_; ++next) {
      subset_.push_back(candidates[next].second);
      Mark(candidates[next].second, true);
      changed = true;
    }
    if (next < candidates.size() && !subset_.empty()) {
      std::vector<Ranked> members;
      members.reserve(subset_.size());
      for (const auto id : subset_)
        members.push_back({Decayed(static_cast<std::uint32_t>(id)), id});
      // Weakest member first.
      std::sort(members.begin(), members.end(),
                [&](const Ranked& a, const Ranked& b) { return higher(b, a); });
      std::size_t weakest = 0;
      for (; next < candidates.size() && weakest < members.size();
           ++next, ++weakest) {
        if (!(candidates[next].first > kHysteresis * members[weakest].first))
          break;
        Mark(members[weakest].second, false);
        Mark(candidates[next].second, true);
        members[weakest] = candidates[next];
        changed = true;
      }
      if (weakest != 0) {
        subset_.clear();
        for (const auto& m : members)
          subset_.push_back(m.second);
      }
    }
    if (changed)
      ++version_;
  }

private:
  [[nodiscard]] float Decayed(std::uint32_t id) const {
    const auto steps = epoch_ - epochs_[id];
    return steps == 0
               ? counts_[id]
               : counts_[id] * std::pow(decay_, static_cast<float>(steps));
  }
  void Add(std::uint32_t id, float weight) {
    if (counts_.size() <= id) {
      counts_.resize(std::size_t{id} + 1, 0.0F);
      epochs_.resize(std::size_t{id} + 1, 0);
      counted_.resize(std::size_t{id} + 1, false);
    }
    if (!counted_[id]) {
      counted_[id] = true;
      seen_.push_back(id);
    }
    counts_[id] = Decayed(id) + weight;
    epochs_[id] = epoch_;
  }
  void Mark(std::int32_t id, bool value) {
    if (member_.size() <= static_cast<std::size_t>(id))
      member_.resize(static_cast<std::size_t>(id) + 1, false);
    member_[static_cast<std::size_t>(id)] = value;
  }

  std::uint32_t rows_;
  std::uint32_t vocab_;
  float decay_;
  std::vector<float> counts_;
  std::vector<std::uint32_t> epochs_;  ///< rebuild epoch of each count
  std::vector<std::uint32_t> seen_;    ///< every token ever counted
  std::vector<bool> counted_;
  std::vector<std::int32_t> subset_;
  std::vector<bool> member_;
  std::uint32_t epoch_{0};
  std::size_t pending_{0};
  std::uint64_t version_{0};
};

/// Whether a draft round reads the subset head. A round's miss is
/// attributable when the target's token at the first rejected position (or
/// the token after a fully accepted round) lies outside the subset: the full
/// head could have proposed it. Rounds on the full head report whether their
/// target tokens would have missed, so the rate keeps tracking while the
/// subset is off. The periodic full-head probe exists for what depends on the
/// head (its accepted length and round time, to refresh `saving_ms`), which
/// the caller measures; the miss rate itself does not need it.
class MtpDraftSubsetSwitch {
public:
  /// Every this many rounds one reads the full head.
  static constexpr std::uint64_t kProbeRounds = 128;
  /// Rounds the miss rate averages over.
  static constexpr double kWindow = 128.0;

  /// `saving_ms`: what a round saves on the subset head; `round_ms`: a
  /// round's time on the full head. Break-even miss rate 2 * saving / round.
  MtpDraftSubsetSwitch(double saving_ms, double round_ms)
      : break_even_(round_ms > 0 ? 2.0 * saving_ms / round_ms : 0.0) {}

  void Reset() {
    rounds_ = 0;
    miss_rate_ = 0.0;
  }

  /// The choice for the next round.
  [[nodiscard]] bool UseSubset() const noexcept {
    if (rounds_ % kProbeRounds == kProbeRounds - 1)
      return false;  // probe the full head
    return miss_rate_ < break_even_;
  }
  /// Records the round just verified.
  void Record(bool attributable_miss) noexcept {
    ++rounds_;
    const double sample = attributable_miss ? 1.0 : 0.0;
    miss_rate_ += (sample - miss_rate_) / std::min<double>(rounds_, kWindow);
  }

  [[nodiscard]] double MissRate() const noexcept { return miss_rate_; }
  [[nodiscard]] double BreakEven() const noexcept { return break_even_; }
  [[nodiscard]] std::uint64_t Rounds() const noexcept { return rounds_; }

private:
  double break_even_;
  std::uint64_t rounds_{0};
  double miss_rate_{0.0};
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_DYNAMIC_DRAFT_VOCAB_HPP_
