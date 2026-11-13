#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ACTIVATION_COLLECTION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ACTIVATION_COLLECTION_HPP_

// The activation collector's record rules, framing and readout, as
// tools/qwen38_flash_next/abliteration/collect.cpp defines them (modes
// collect, answers and segments), for the in-server collection jobs
// (`gufo serve --collect-dir`). CPU only: the caller runs the passes.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::tokenization {
class QwenTokenizer;
}

namespace gufo::models::qwen38_flash_next {

enum class CollectionMode : std::uint8_t { kCollect, kAnswers, kSegments };

/// "collect", "answers" or "segments"; nullopt for anything else.
[[nodiscard]] std::optional<CollectionMode> ParseCollectionMode(
    std::string_view name) noexcept;
[[nodiscard]] std::string_view CollectionModeName(CollectionMode mode) noexcept;
/// collect.cpp's session context per mode: 2048, 8192 or 24576 tokens.
[[nodiscard]] std::uint32_t CollectionContext(CollectionMode mode) noexcept;
/// The manifest's `template` and `readout` names.
[[nodiscard]] std::string_view CollectionTemplate(CollectionMode mode) noexcept;
[[nodiscard]] std::string_view CollectionReadoutName(
    CollectionMode mode) noexcept;
/// The longest framed prompt in every mode.
inline constexpr std::size_t kCollectionMaxPromptTokens = 2048;

/// Checks one input record as collect.cpp does (id characters, group, pair,
/// text, answer, thinking, segments and select), and that every record has
/// the `split` of the first (`*split` empty before the first). Throws
/// std::invalid_argument naming what is wrong.
void ValidateCollectionRecord(CollectionMode mode, const json::Value& record,
                              std::string* split);

/// A record framed and tokenized: the prompt (thinking off for answers,
/// thinking on for segments, the record's `thinking` for collect; medium
/// effort) and, for answers and segments, the teacher-forced tokens.
struct CollectionPlan {
  std::vector<std::int32_t> prompt;
  std::vector<std::int32_t> answer;
  /// Segments: which answer rows (row j predicts answer token j) are
  /// averaged; empty otherwise.
  std::vector<bool> selected;
  std::size_t selected_rows{0};
};

/// Frames and tokenizes a validated record as collect.cpp does. `context`
/// is the session context (the mode's CollectionContext, or less): the
/// prompt is at most kCollectionMaxPromptTokens and the prompt plus the
/// answer at most `context`. Throws std::invalid_argument for a record that
/// cannot run (too long, an empty answer or segment).
[[nodiscard]] CollectionPlan PlanCollectionRecord(
    const tokenization::QwenTokenizer& tokenizer, CollectionMode mode,
    const json::Value& record, std::size_t context);

/// The trunk passes a plan feeds, in order: `observed` false for answers'
/// prompt without its last token, true for every row read.
struct CollectionFeed {
  std::vector<std::int32_t> tokens;
  bool observed{false};
};
[[nodiscard]] std::vector<CollectionFeed> CollectionFeeds(
    CollectionMode mode, const CollectionPlan& plan);

/// collect.cpp's activation observer: each layer's post-layer residual rows
/// ([rows][hc_count * hidden]) reduced to the HC mean in FP64 and read out
/// as FP32: the last prompt row (collect), the mean over every answer row
/// (answers) or over the selected rows (segments). Session observers are
/// called with all rows for answers and segments, the last row for collect.
class CollectionReadout {
public:
  CollectionReadout(CollectionMode mode, std::uint32_t layers,
                    std::uint32_t hidden, std::uint32_t hc_count);
  /// Starts a record (clears what the last one read).
  void Begin(const CollectionPlan& plan);
  /// Whether the following passes are read (CollectionFeed::observed).
  void SetObserving(bool observing) noexcept { observing_ = observing; }
  /// The session's observer. Records a layout error instead of throwing
  /// through the pass; Finish reports it.
  void Observe(std::uint32_t layer, std::span<const float> rows);
  /// The record's [layers][hidden] readout. Throws std::runtime_error when a
  /// layer was not captured or an answer row is missing.
  [[nodiscard]] std::vector<float> Finish(const CollectionPlan& plan);
  [[nodiscard]] bool all_rows() const noexcept {
    return mode_ != CollectionMode::kCollect;
  }

private:
  CollectionMode mode_;
  std::uint32_t layers_, hidden_, hc_count_;
  std::vector<float> activations_;
  std::vector<bool> captured_;
  std::vector<double> sums_;
  std::vector<std::size_t> rows_seen_;
  std::vector<bool> selected_;
  bool observing_{true};
  bool layout_error_{false};
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_ACTIVATION_COLLECTION_HPP_
