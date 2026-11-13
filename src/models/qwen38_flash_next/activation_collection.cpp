#include "src/models/qwen38_flash_next/activation_collection.hpp"

#include <algorithm>
#include <stdexcept>

#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {
void Require(bool ok, const std::string& error) {
  if (!ok)
    throw std::invalid_argument(error);
}
}  // namespace

std::optional<CollectionMode> ParseCollectionMode(
    std::string_view name) noexcept {
  if (name == "collect")
    return CollectionMode::kCollect;
  if (name == "answers")
    return CollectionMode::kAnswers;
  if (name == "segments")
    return CollectionMode::kSegments;
  return std::nullopt;
}

std::string_view CollectionModeName(CollectionMode mode) noexcept {
  switch (mode) {
    case CollectionMode::kCollect:
      return "collect";
    case CollectionMode::kAnswers:
      return "answers";
    case CollectionMode::kSegments:
      return "segments";
  }
  return "collect";
}

std::uint32_t CollectionContext(CollectionMode mode) noexcept {
  return mode == CollectionMode::kSegments  ? 24576
         : mode == CollectionMode::kAnswers ? 8192
                                            : 2048;
}

std::string_view CollectionTemplate(CollectionMode mode) noexcept {
  return mode == CollectionMode::kSegments
             ? "gufo-qwen-native-thinking-medium-v1"
         : mode == CollectionMode::kAnswers ? "gufo-qwen-native-scoring-v1"
                                            : "gufo-qwen-native-medium-v1";
}

std::string_view CollectionReadoutName(CollectionMode mode) noexcept {
  return mode == CollectionMode::kSegments
             ? "selected-segment-end-rows-mean-post-layer-hc-mean-fp32-v1"
         : mode == CollectionMode::kAnswers
             ? "answer-tokens-mean-post-layer-hc-mean-fp32-v1"
             : "last-prompt-token-post-layer-hc-mean-fp32-v1";
}

void ValidateCollectionRecord(CollectionMode mode, const json::Value& record,
                              std::string* split) {
  const bool segments = mode == CollectionMode::kSegments;
  const bool answers = segments || mode == CollectionMode::kAnswers;
  Require(record.is_object(), "a record is a JSON object");
  if (split->empty())
    *split = record.member_str("split");
  Require(record.member_str("split") == *split, "mixed dataset splits");
  const auto id = record.member_str("id");
  Require(!id.empty() && id.find_first_not_of("0123456789abcdef-plainthk") ==
                             std::string::npos,
          "invalid record ID");
  if (segments) {
    const auto* parts = record.find("segments");
    const auto* select = record.find("select");
    Require(parts && parts->is_array() && parts->size() >= 2 && select &&
                select->is_array() && select->size() >= 1,
            "segments need a segments array (>= 2) and a select array");
    for (const auto& part : parts->items())
      Require(part.is_string() && !part.str().empty(),
              "segments must be non-empty strings");
    for (const auto& k : select->items())
      Require(k.is_number() && k.as_size(parts->size()) + 1 < parts->size(),
              "select indexes must name a segment followed by another");
  }
  if (answers)
    Require((record.member_str("group") == "exemplar" ||
             record.member_str("group") == "own") &&
                !record.member_str("pair").empty() &&
                !record.member_str("text").empty() &&
                (segments || !record.member_str("answer").empty()),
            "answers need group exemplar|own, pair, text and answer");
  else
    Require(record.member_str("group") == "refusal" ||
                record.member_str("group") == "ordinary",
            "invalid record class");
  Require(answers ||
              (record.find("thinking") && record.find("thinking")->is_bool()),
          "missing thinking mode");
}

CollectionPlan PlanCollectionRecord(
    const tokenization::QwenTokenizer& tokenizer, CollectionMode mode,
    const json::Value& record, std::size_t context) {
  const bool segments = mode == CollectionMode::kSegments;
  const bool answers = segments || mode == CollectionMode::kAnswers;
  const auto id = record.member_str("id");
  tokenization::ChatMessage message;
  message.role = tokenization::ChatRole::kUser;
  message.content = record.member_str("text");
  tokenization::ChatTemplateOptions chat;
  // Answers are framed as score_assistant scores them: thinking off.
  // Segments continue a thinking-on generation prompt.
  chat.enable_thinking =
      segments || (!answers && record.find("thinking")->as_bool());
  chat.reasoning_effort = tokenization::QwenReasoningEffort::kMedium;
  std::string error;
  auto tokens = tokenization::QwenChatTemplate::RenderAndTokenize(
      tokenizer, std::span(&message, 1), chat, &error);
  Require(bool(tokens) && !tokens->empty() &&
              tokens->size() <= kCollectionMaxPromptTokens &&
              tokens->size() <= context,
          "prompt invalid/too long; no silent truncation: " + id + " " + error);
  CollectionPlan plan;
  plan.prompt.assign(tokens->begin(), tokens->end());
  if (!answers)
    return plan;
  tokenization::TokenizerOptions content;
  content.parse_special_tokens = false;
  if (segments) {
    // Row j predicts answer token j; row ends[k] (the row of segment k's
    // last token) predicts segment k+1's first token.
    std::vector<std::size_t> ends;
    for (const auto& part : record.find("segments")->items()) {
      const auto piece = tokenizer.Encode(part.str(), content);
      Require(!piece.empty(), "segment tokenizes empty: " + id);
      plan.answer.insert(plan.answer.end(), piece.begin(), piece.end());
      ends.push_back(plan.answer.size());
    }
    plan.selected.assign(plan.answer.size(), false);
    for (const auto& k : record.find("select")->items()) {
      const auto row = ends[k.as_size()];
      if (!plan.selected[row])
        ++plan.selected_rows;
      plan.selected[row] = true;
    }
  } else {
    const auto encoded = tokenizer.Encode(record.member_str("answer"), content);
    plan.answer.assign(encoded.begin(), encoded.end());
  }
  Require(!plan.answer.empty() &&
              plan.prompt.size() + plan.answer.size() <= context,
          "answer empty or too long: " + id);
  return plan;
}

std::vector<CollectionFeed> CollectionFeeds(CollectionMode mode,
                                            const CollectionPlan& plan) {
  if (mode == CollectionMode::kCollect)
    return {{plan.prompt, true}};
  // Teacher forcing: the prompt but its last token first, then the rows
  // from the last prompt token (P-1) through the answer's second to last
  // token (P+n-2), n rows that predict the answer's n tokens. The answer's
  // last token is never fed.
  CollectionFeed prompt{{plan.prompt.begin(), plan.prompt.end() - 1}, false};
  CollectionFeed full{plan.prompt, true};
  full.tokens.insert(full.tokens.end(), plan.answer.begin(),
                     plan.answer.end() - 1);
  return {std::move(prompt), std::move(full)};
}

CollectionReadout::CollectionReadout(CollectionMode mode, std::uint32_t layers,
                                     std::uint32_t hidden,
                                     std::uint32_t hc_count)
    : mode_(mode),
      layers_(layers),
      hidden_(hidden),
      hc_count_(hc_count),
      activations_(std::size_t{layers} * hidden),
      captured_(layers),
      sums_(mode == CollectionMode::kCollect ? 0 : activations_.size()),
      rows_seen_(layers) {}

void CollectionReadout::Begin(const CollectionPlan& plan) {
  std::fill(captured_.begin(), captured_.end(), false);
  std::fill(rows_seen_.begin(), rows_seen_.end(), 0);
  std::fill(sums_.begin(), sums_.end(), 0.0);
  selected_ = plan.selected;
  observing_ = mode_ == CollectionMode::kCollect;
  layout_error_ = false;
}

void CollectionReadout::Observe(std::uint32_t layer,
                                std::span<const float> rows) {
  if (!observing_)
    return;
  const bool answers = mode_ != CollectionMode::kCollect;
  const bool segments = mode_ == CollectionMode::kSegments;
  const std::size_t hc_dim = std::size_t{hc_count_} * hidden_;
  if (layer >= layers_ || rows.size() % hc_dim != 0 ||
      (!answers && rows.size() != hc_dim)) {
    layout_error_ = true;
    return;
  }
  for (std::size_t r = 0; r < rows.size() / hc_dim; ++r) {
    const auto row = rows.subspan(r * hc_dim, hc_dim);
    const bool keep = !segments || (rows_seen_[layer] < selected_.size() &&
                                    selected_[rows_seen_[layer]]);
    for (std::size_t h = 0; keep && h < hidden_; ++h) {
      double sum = 0;
      for (std::size_t c = 0; c < hc_count_; ++c)
        sum += row[c * hidden_ + h];
      const auto index = std::size_t{layer} * hidden_ + h;
      if (answers)
        sums_[index] += sum / hc_count_;
      else
        activations_[index] = float(sum / hc_count_);
    }
    ++rows_seen_[layer];
  }
  captured_[layer] = true;
}

std::vector<float> CollectionReadout::Finish(const CollectionPlan& plan) {
  if (layout_error_)
    throw std::runtime_error("collector HC layout mismatch");
  if (mode_ == CollectionMode::kCollect) {
    if (!std::ranges::all_of(captured_, [](bool x) { return x; }))
      throw std::runtime_error("missing target layer capture");
    return activations_;
  }
  for (std::uint32_t layer = 0; layer < layers_; ++layer)
    if (rows_seen_[layer] != plan.answer.size())
      throw std::runtime_error("answer rows missing");
  const double rows = mode_ == CollectionMode::kSegments
                          ? double(plan.selected_rows)
                          : double(plan.answer.size());
  for (std::size_t i = 0; i < activations_.size(); ++i)
    activations_[i] = float(sums_[i] / rows);
  return activations_;
}

}  // namespace gufo::models::qwen38_flash_next
