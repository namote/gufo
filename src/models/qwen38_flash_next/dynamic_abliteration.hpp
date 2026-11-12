#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_DYNAMIC_ABLITERATION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_DYNAMIC_ABLITERATION_HPP_

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/models/qwen/abliteration.hpp"

namespace gufo::models::qwen38_flash_next {

/// Score columns per row: one per loaded detector, at most one per profile.
inline constexpr std::uint32_t kDetectorColumns =
    static_cast<std::uint32_t>(qwen::kAbliterationMaxProfiles);

/// One dynamic profile a request may turn ON (R2): its detector's score
/// column and threshold, and the settings it runs with once ON (the profile
/// defaults).
struct DynamicAbliterationCandidate {
  std::string profile;
  std::uint32_t column{0};
  /// For the last prompt row.
  float threshold{0};
  /// For generated rows; without one they never turn the profile ON.
  std::optional<float> decode_threshold;
  std::shared_ptr<const qwen::AbliterationSettings> settings;
};

/// The detector scores one trunk pass produced for a session. Row r belongs
/// to the token at position `first_position + r`: the last prompt row of a
/// prefill pass, every row of a decode or verification pass.
struct DetectorPass {
  /// Distinct per pass of the session; zero means no pass yet.
  std::uint64_t id{0};
  std::uint32_t first_position{0};
  std::uint32_t rows{0};
  bool prefill{false};
  /// rows x kDetectorColumns.
  std::span<const float> scores;
};

/// What a request reports as `abliteration` (docs/SERVER.md).
struct DynamicAbliterationReport {
  /// The profile detection selected (at most one; see DynamicAbliteration),
  /// also when the request ended before a pass ran with it ON.
  std::vector<std::string> triggered;
  /// Per candidate: the score of the row that crossed, else of the last
  /// prompt row; empty before any scored row.
  std::vector<std::pair<std::string, float>> scores;
  /// Per candidate: the largest score of any committed generated row (the
  /// data a `decode_threshold` is fitted on); empty before one.
  std::vector<std::pair<std::string, float>> max_scores;
  /// The first token position computed (or to be computed) with the
  /// profile ON: the prompt length for a prefill detection; empty without
  /// one.
  std::optional<std::uint32_t> position;
  /// "prefill" or "decode": the pass whose row crossed; empty without one.
  std::string phase;
};

/// Per-request detection state (A1, sticky, one profile at a time).
///
/// Observe() reads a pass's rows that the session committed (positions below
/// `committed`; rows a verification rolled back are ignored). The last
/// prompt row is compared with each candidate's `threshold`; generated rows
/// only with a `decode_threshold` (none by default: they are scored for
/// `max_scores` only). The first row on which a candidate crosses selects
/// the one with the largest margin (score - threshold). Engage() hands its
/// settings, made dynamic at the session's committed length, to the caller
/// before the next trunk pass; from then on the profile stays ON for the rest
/// of the request and later rows are no longer scored. Prompt rows already
/// computed keep their OFF state.
class DynamicAbliteration {
public:
  explicit DynamicAbliteration(std::vector<DynamicAbliterationCandidate> c);

  /// Ignores a pass already observed (same id) and passes after engagement.
  void Observe(const DetectorPass& pass, std::uint32_t committed);
  /// The settings to turn ON from `position` (the committed length), once;
  /// null when no row crossed or after the first call that returned them.
  [[nodiscard]] std::shared_ptr<const qwen::AbliterationSettings> Engage(
      std::uint32_t position);
  /// Prefill rows count only at the last prompt token (position
  /// tokens - 1): a crossing mid-prompt (a request cancelled between prefill
  /// chunks) never selects a profile. Zero (unset) ignores prefill rows.
  void SetPromptLength(std::uint32_t tokens) noexcept {
    prompt_length_ = tokens;
  }
  [[nodiscard]] bool engaged() const noexcept { return engaged_; }
  [[nodiscard]] bool detecting() const noexcept { return !engaged_; }
  [[nodiscard]] const std::vector<DynamicAbliterationCandidate>& candidates()
      const noexcept {
    return candidates_;
  }
  [[nodiscard]] DynamicAbliterationReport Report() const;

private:
  std::vector<DynamicAbliterationCandidate> candidates_;
  std::uint64_t observed_{0};
  std::uint32_t prompt_length_{0};
  /// Scores of the row that crossed, else of the last prompt row.
  std::vector<float> row_scores_;
  bool have_row_{false};
  std::vector<float> max_scores_;
  bool have_max_{false};
  std::optional<std::size_t> pending_;
  bool pending_prefill_{false};
  bool engaged_{false};
  std::uint32_t pending_position_{0};
  std::uint32_t position_{0};
};

}  // namespace gufo::models::qwen38_flash_next
#endif
