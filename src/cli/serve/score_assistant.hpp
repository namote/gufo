#ifndef GUFO_SERVER_SCORE_ASSISTANT_HPP_
#define GUFO_SERVER_SCORE_ASSISTANT_HPP_

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "src/cli/serve/text_generation_backend.hpp"

namespace gufo::server {

/// Most assistant-content tokens one `score_assistant` request may score.
inline constexpr std::size_t kMaxScoredTokens = 1024;

/// A `score_assistant` request split at its final assistant message: the
/// conversation before it, rendered with the generation prompt as for a new
/// reply, and the content scored after that prompt.
struct ScoringSplit {
  ChatRequest prompt;
  std::string answer;
};

/// Thinking is always off: the prompt ends with the template's answered
/// generation prompt (an empty `<think>\n\n</think>\n\n` block), so the
/// scored content is exactly what a thinking-disabled reply generates after
/// it. Throws std::invalid_argument for a request that cannot be scored.
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
  if (!last.thought.empty() || !last.tool_calls.empty())
    throw std::invalid_argument(
        "score_assistant scores text content only; the scored message takes "
        "no reasoning_content or tool_calls");
  for (const auto& message : request.messages)
    if (!message.images.empty())
      throw std::invalid_argument("score_assistant does not support images");
  if (!request.tools.empty() || request.response_format)
    throw std::invalid_argument(
        "score_assistant does not support tools or response_format");
  if (request.reasoning.enabled.value_or(false))
    throw std::invalid_argument(
        "score_assistant scores the reply after an empty thinking block; "
        "thinking must be disabled");
  ScoringSplit split{request, last.content};
  split.prompt.messages.pop_back();
  split.prompt.reasoning.enabled = false;
  split.prompt.reasoning.budget_tokens.reset();
  split.prompt.stop_sequences.clear();
  split.prompt.return_progress = false;
  split.prompt.cache_persist = false;
  split.prompt.score_assistant = false;
  return split;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_SCORE_ASSISTANT_HPP_
