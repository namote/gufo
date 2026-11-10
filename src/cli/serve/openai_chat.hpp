#ifndef GUFO_SERVER_OPENAI_CHAT_HPP_
#define GUFO_SERVER_OPENAI_CHAT_HPP_

#include "src/cli/serve/http_server.hpp"
#include "src/core/image.hpp"

namespace gufo::server {

/// Handles the supported OpenAI Chat Completions subset. Streaming responses
/// consume scheduler-published token pieces from HttpResponse::streaming_body;
/// socket writes never own or execute model state.
HttpResponse HandleOpenAiChat(const HttpRequest& request,
                              TextGenerationBackend& backend);

/// Count the same prompt as Chat Completions, without scheduling generation.
/// Streaming options are accepted but the result is always one JSON response.
HttpResponse HandleOpenAiChatInputTokens(const HttpRequest& request,
                                         TextGenerationBackend& backend);

/// Shared Chat/Responses result, including unsupported-backend and prompt
/// errors.
HttpResponse CountOpenAiInputTokens(const ChatRequest& chat,
                                    TextGenerationBackend& backend);

/// llama-server's /apply-template: the prompt a Chat Completions body renders
/// to, with the server's reasoning defaults applied, without generating.
/// `model` is optional, as outside llama-server router mode.
HttpResponse HandleApplyTemplate(const HttpRequest& request,
                                 TextGenerationBackend& backend);

/// Responses nests output schema under text.format and effort under reasoning.
/// Shared controls stay validated as in Chat Completions, but the Responses
/// adapter tolerates standard Responses fields with no native effect (hosted
/// tool types, reasoning.summary, text.verbosity) rather than reject
/// the whole request, because conforming Responses clients routinely send them.
std::optional<HttpResponse> ParseOpenAiResponseControls(const json::Value& body,
                                                        ChatRequest* chat);
bool ParseOpenAiResponseMessage(const json::Value& item,
                                tokenization::ChatMessage* message,
                                core::ImageReadBudget& budget,
                                std::string* error);
/// Gufo request-local target transform, shared across chat dialects.
std::optional<HttpResponse> ParseAbliterationControl(
    const json::Value& body, const TextGenerationBackend& backend,
    ChatRequest* chat);

/// Shared effort names; each API applies its own thinking and alias rules.
std::optional<ReasoningEffort> ParseReasoningEffortName(std::string_view value);
/// True when a request body explicitly sets `typical_acceptance: true`.
bool RequestsTypicalAcceptance(const json::Value& body);
/// Logs a warning, once per process, when a generating request explicitly
/// asked for typical acceptance and the loaded backend never applies it.
/// The inherited default never warns; endpoints that only count or render
/// do not call this.
void WarnUnsupportedTypicalAcceptance(bool requested,
                                      const TextGenerationBackend& backend);

/// Messages tools and tool_choice, mapped onto the Chat tool declarations so
/// framing, schema constraints and replayed tool turns match Chat.
std::optional<HttpResponse> ParseAnthropicToolControls(const json::Value& body,
                                                       ChatRequest* chat);
/// One Messages image block (base64 or HTTPS URL source), appended to the
/// message at the current end of its text.
bool ParseAnthropicImage(const json::Value& block,
                         tokenization::ChatMessage* message,
                         core::ImageReadBudget& budget, std::string* error);
/// One Messages turn with tool_use or tool_result blocks. Each tool_result
/// becomes a tool message, with its images; text and images in the same user
/// turn follow as a user message.
bool ParseAnthropicToolMessage(const json::Value& item,
                               std::vector<tokenization::ChatMessage>* messages,
                               core::ImageReadBudget& budget,
                               std::string* error);
/// Responses text output uses the same reasoning/UTF-8 filter and scheduler
/// as Chat Completions, including streaming cancellation and cache retention.
/// `model` is the served name to report; empty reports the primary name.
HttpResponse CreateOpenAiResponse(const HttpRequest& request,
                                  TextGenerationBackend& backend,
                                  const ChatRequest& chat,
                                  std::size_t max_tokens,
                                  const sampling::SamplingConfig& sampling,
                                  bool stream, std::string_view model = {});
/// Messages output through the same path: content blocks when buffered,
/// Anthropic SSE events when streamed.
HttpResponse CreateAnthropicMessage(const HttpRequest& request,
                                    TextGenerationBackend& backend,
                                    const ChatRequest& chat,
                                    std::size_t max_tokens,
                                    const sampling::SamplingConfig& sampling,
                                    bool stream, std::string_view model = {});

}  // namespace gufo::server

#endif  // GUFO_SERVER_OPENAI_CHAT_HPP_
