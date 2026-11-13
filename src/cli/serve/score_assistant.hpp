#ifndef GUFO_SERVER_SCORE_ASSISTANT_HPP_
#define GUFO_SERVER_SCORE_ASSISTANT_HPP_

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace gufo::server {

/// Most assistant-content tokens one `score_assistant` request may score.
inline constexpr std::size_t kMaxScoredTokens = 1024;

/// A `score_assistant` request split at its final assistant message: the
/// conversation before it, rendered with the generation prompt as for a new
/// reply, and the content scored after that prompt. A thinking-turn split
/// (the scored message has reasoning_content, or the request sets
/// score_reasoning) renders the prompt with thinking on, so it ends with the
/// open thinking block, and `reasoning` (message text) then `reasoning_close`
/// (framing) follow it before the scored content.
struct ScoringSplit {
  ChatRequest prompt;
  std::string answer;
  std::string reasoning;
  std::string_view reasoning_close;
};

/// Without reasoning, thinking is off: the prompt ends with the template's
/// answered generation prompt (an empty `<think>\n\n</think>\n\n` block), so
/// the scored content is exactly what a thinking-disabled reply generates
/// after it. With reasoning_content, the prompt ends with `<think>\n`, the
/// reasoning follows trimmed as the template renders a completed turn, then
/// tokenization::kReasoningClose, and the content is scored as that turn's
/// answer. The close's leading newline belongs to `reasoning` (message
/// text): whole-text tokenization, as generation and the template produce it,
/// merges it with the reasoning's last punctuation (".\n"), and `</think>` is
/// a vocabulary token that splits there anyway. With score_reasoning, the
/// reasoning follows verbatim (empty allowed) and the content is scored as its
/// continuation inside the thinking block; nothing closes it. Throws
/// std::invalid_argument for a request that cannot be scored.
[[nodiscard]] inline ScoringSplit SplitScoringRequest(
    const ChatRequest& request) {
  if (request.messages.size() < 2)
    throw std::invalid_argument(
        "score_assistant needs a conversation followed by the assistant "
        "message to score");
  const auto& last = request.messages.back();
  if (last.role != tokenization::ChatRole::kAssistant)
    throw std::invalid_argument(
        "score_assistant requires the last message to be from the assistant");
  if (last.content.empty())
    throw std::invalid_argument(
        "score_assistant requires non-empty assistant content");
  if (!last.tool_calls.empty())
    throw std::invalid_argument(
        "score_assistant scores text content only; the scored message takes "
        "no tool_calls");
  for (const auto& message : request.messages)
    if (!message.images.empty())
      throw std::invalid_argument("score_assistant does not support images");
  if (!request.tools.empty() || request.response_format)
    throw std::invalid_argument(
        "score_assistant does not support tools or response_format");
  const bool thinking = request.score_reasoning || !last.thought.empty();
  if (thinking && !request.reasoning.enabled.value_or(false))
    throw std::invalid_argument(
        "score_assistant with reasoning_content or score_reasoning scores "
        "inside a thinking turn; thinking must be enabled");
  if (!thinking && request.reasoning.enabled.value_or(false))
    throw std::invalid_argument(
        "score_assistant scores the reply after an empty thinking block; "
        "thinking must be disabled");
  ScoringSplit split{request, last.content, {}, {}};
  if (request.score_reasoning) {
    split.reasoning = last.thought;
  } else if (thinking) {
    split.reasoning = std::string(tokenization::TrimReasoning(last.thought));
    split.reasoning += tokenization::kReasoningClose.front();
    split.reasoning_close = tokenization::kReasoningClose.substr(1);
  }
  split.prompt.messages.pop_back();
  split.prompt.reasoning.enabled = thinking;
  split.prompt.reasoning.budget_tokens.reset();
  split.prompt.stop_sequences.clear();
  split.prompt.return_progress = false;
  split.prompt.cache_persist = false;
  split.prompt.score_assistant = false;
  split.prompt.score_max_tokens.reset();
  split.prompt.score_reasoning = false;
  return split;
}

/// Cuts the tokenized scored content to the request's `score_max_tokens`
/// and enforces the scored-token limits. Throws std::invalid_argument for
/// content without tokens or more than kMaxScoredTokens after the cut.
template<class Token>
void LimitScoredTokens(std::vector<Token>* answer, const ChatRequest& request) {
  if (answer->empty())
    throw std::invalid_argument("score_assistant content has no tokens");
  if (request.score_max_tokens && answer->size() > *request.score_max_tokens)
    answer->resize(*request.score_max_tokens);
  if (answer->size() > kMaxScoredTokens)
    throw std::invalid_argument(
        "score_assistant content has " + std::to_string(answer->size()) +
        " tokens; the limit is " + std::to_string(kMaxScoredTokens));
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_SCORE_ASSISTANT_HPP_
