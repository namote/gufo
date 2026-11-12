#ifndef GUFO_SERVER_TEXT_GENERATION_BACKEND_HPP_
#define GUFO_SERVER_TEXT_GENERATION_BACKEND_HPP_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/component_cache_metrics.hpp"
#include "src/cli/serve/logprobs.hpp"
#include "src/core/reasoning.hpp"
#include "src/core/sampling.hpp"
#include "src/core/text_sampling_defaults.hpp"
#include "src/models/qwen/abliteration.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::server {

enum class TextGenerationErrorCode : std::uint8_t {
  kQueueFull,
  kClientQueueFull,
  kDeadlineExceeded,
  kOutputLimit,
  kOutputBackpressure,
  kSchedulerStopping,
  kToolChoiceUnsatisfied,
  kDeviceLost,
};

/// A loaded runtime abliteration profile as requests and /props see it:
/// geometry and defaults, never the direction vectors.
struct AbliterationInfo {
  /// The profile file name without `.json`; requests select it by this.
  std::string name;
  std::string schema;
  /// SHA-256 of the profile file.
  std::string identity;
  std::uint32_t layers{0};
  /// The default direction layer; empty for a v1 profile, whose one direction
  /// cannot be changed per request.
  std::optional<std::uint32_t> direction_layer;
  std::vector<float> attention;
  std::vector<float> mlp;
  std::vector<float> residual;
  /// Direction kinds a request may name in `direction_kind`; `rank` bounds
  /// the request's `directions`; then the profile defaults.
  std::vector<std::string> direction_kinds{"mean_diff"};
  std::uint32_t rank{1};
  std::string direction_kind{"mean_diff"};
  std::uint32_t direction_count{1};
  bool preserve_norm{false};
  /// "pca", "som" or empty.
  std::string subspace_method;
  /// A dynamic profile's detector layer and threshold; empty for a profile
  /// that turns ON only when a request asks for it.
  std::optional<std::uint32_t> trigger_layer;
  std::optional<float> trigger_threshold;
};

/// A request's dynamic abliteration (`abliteration` in its response): the
/// profiles detection turned ON, the detector scores and where it switched.
struct AbliterationReport {
  std::vector<std::string> triggered;
  /// Per dynamic profile: the score of the row that turned it ON, else of
  /// the last prompt row.
  std::vector<std::pair<std::string, float>> scores;
  /// Per dynamic profile: the largest score of any committed row.
  std::vector<std::pair<std::string, float>> max_scores;
  /// The first token position computed ON (the prompt length when the last
  /// prompt row crossed); empty while OFF.
  std::optional<std::size_t> position;
  /// "prefill" or "decode"; empty while OFF.
  std::string phase;
};

/// Client-facing message for a GPU context that can no longer execute work.
inline constexpr const char* kDeviceLostMessage =
    "GPU context lost; restart required";

class TextGenerationError final : public std::runtime_error {
public:
  TextGenerationError(TextGenerationErrorCode code, const std::string& message)
      : std::runtime_error(message), code_(code) {}

  [[nodiscard]] TextGenerationErrorCode code() const noexcept { return code_; }
  [[nodiscard]] int http_status() const noexcept {
    switch (code_) {
      case TextGenerationErrorCode::kQueueFull:
      case TextGenerationErrorCode::kClientQueueFull:
        return 429;
      case TextGenerationErrorCode::kDeadlineExceeded:
        return 408;
      case TextGenerationErrorCode::kOutputLimit:
      case TextGenerationErrorCode::kOutputBackpressure:
      case TextGenerationErrorCode::kSchedulerStopping:
      case TextGenerationErrorCode::kDeviceLost:
        return 503;
      case TextGenerationErrorCode::kToolChoiceUnsatisfied:
        return 502;
    }
    return 500;
  }
  [[nodiscard]] const char* stable_code() const noexcept {
    switch (code_) {
      case TextGenerationErrorCode::kQueueFull:
        return "queue_full";
      case TextGenerationErrorCode::kClientQueueFull:
        return "client_queue_full";
      case TextGenerationErrorCode::kDeadlineExceeded:
        return "deadline_exceeded";
      case TextGenerationErrorCode::kOutputLimit:
        return "output_limit";
      case TextGenerationErrorCode::kOutputBackpressure:
        return "output_backpressure";
      case TextGenerationErrorCode::kSchedulerStopping:
        return "scheduler_stopping";
      case TextGenerationErrorCode::kToolChoiceUnsatisfied:
        return "tool_choice_unsatisfied";
      case TextGenerationErrorCode::kDeviceLost:
        return "device_lost";
    }
    return "generation_error";
  }
  [[nodiscard]] bool retryable() const noexcept {
    return code_ != TextGenerationErrorCode::kOutputLimit &&
           code_ != TextGenerationErrorCode::kToolChoiceUnsatisfied &&
           code_ != TextGenerationErrorCode::kDeviceLost;
  }

private:
  TextGenerationErrorCode code_;
};

struct ChatRequest {
  enum class ToolChoice : std::uint8_t {
    kAuto,
    kNone,
    kRequired,
  };

  ChatRequest() = default;
  explicit ChatRequest(std::vector<tokenization::ChatMessage> chat_messages)
      : messages(std::move(chat_messages)) {}

  std::vector<tokenization::ChatMessage> messages;
  std::vector<tokenization::ChatTool> tools;
  std::string client_id{"anonymous"};
  ToolChoice tool_choice{ToolChoice::kAuto};
  std::string forced_tool_name;
  /// Responses namespace of each flattened function, echoed on its calls.
  std::map<std::string, std::string> tool_namespaces;
  bool constrained_tools{false};
  bool parallel_tool_calls{true};
  ReasoningOptions reasoning;
  bool add_vision_id{false};
  /// Target-model execution transform, independent of sampling/thinking.
  bool abliteration{false};
  /// Whether the request carried an `abliteration` member: an explicit ON or
  /// OFF overrides dynamic detection.
  bool abliteration_specified{false};
  /// Per-request settings while `abliteration` is ON; empty members keep the
  /// profile defaults.
  models::qwen::AbliterationOverrides abliteration_overrides;
  /// Bypass prompt reuse for this request; its completed state may be retained.
  bool cache_prompt{true};
  /// False keeps this request's checkpoints out of the disk cache; it may
  /// still read existing disk checkpoints and use the RAM cache.
  bool cache_persist{true};
  /// Score the final assistant message's content token by token (teacher
  /// forcing) instead of generating; see score_assistant.hpp.
  bool score_assistant{false};
  /// With score_assistant: score only the first this many content tokens
  /// (1..kMaxScoredTokens); unset scores all of them.
  std::optional<std::size_t> score_max_tokens;
  /// Optional streaming prompt progress; never changes prompt or cache
  /// identity.
  bool return_progress{false};
  std::vector<std::string> stop_sequences;
  std::shared_ptr<const sampling::JsonConstraint> response_format;
  std::string response_format_description;
};

/// Model-agnostic text generation boundary used by the HTTP transport.
///
/// Implementations retain ownership of tokenization, templates, complete
/// continuation state, and execution resources. The transport observes only
/// generated text pieces, optional token probabilities and aggregate metrics.
class TextGenerationBackend {
public:
  using CancellationCheck = std::function<bool()>;
  using TokenCallback = std::function<bool(const TokenPiece&)>;

  /// llama-server `prompt_progress`; `processed` includes cached tokens.
  struct PromptProgress {
    std::size_t total{0};
    std::size_t cache{0};
    std::size_t processed{0};
    std::int64_t time_ms{0};
  };
  using ProgressCallback = std::function<bool(const PromptProgress&)>;
  /// Called at most once, before progress or output, when a streaming request
  /// starts work or has waited in the queue for a bounded time.
  using StartCallback = std::function<bool()>;

  /// One execution session for llama-server `/slots`. Idle sessions report
  /// zero counts rather than the previous request's.
  struct SessionState {
    bool processing{false};
    bool speculative{false};
    std::uint64_t request_id{0};
    std::size_t prompt_tokens{0};
    std::size_t cached_prompt_tokens{0};
    /// Prompt tokens prefilled so far, excluding cached tokens.
    std::size_t processed_prompt_tokens{0};
    std::size_t generated_tokens{0};
    std::size_t remaining_tokens{0};
  };

  enum class FinishReason : std::uint8_t {
    kStop,
    kStopSequence,
    kLength,
    kCancelled,
  };

  enum class InitialOutputState : std::uint8_t {
    kAuto,
    kReasoning,
    kContent,
  };

  struct SamplingDefaults {
    /// Zero means generate until EOS or the remaining context is exhausted.
    std::size_t max_tokens{0};
    sampling::SamplingConfig sampling;
    sampling::TextModelPreset model{sampling::TextModelPreset::kUnspecified};
    sampling::SamplingOverrides supplied{sampling::SamplingOverrides::All()};

    [[nodiscard]] sampling::SamplingConfig Resolve(
        std::optional<bool> thinking) const {
      return sampling::ResolveTextSampling(model, thinking, sampling, supplied);
    }
  };

  struct Result {
    std::optional<ComponentCacheMetrics> component_cache;
    std::string text;
    std::vector<DecodedLogprobs> logprobs;
    std::vector<tokenization::TokenId> tokens;
    std::size_t prompt_tokens{0};
    std::size_t cached_prompt_tokens{0};
    std::size_t cache_restore_bytes{0};
    std::size_t cache_snapshot_bytes{0};
    /// Bytes admitted to background persistence; completion is logged by cache.
    std::size_t cache_disk_queued_bytes{0};
    std::size_t cache_shared_bytes{0};
    /// Shared-prefix snapshots queued while prefilling this request.
    std::size_t cache_shared_prefix_snapshots{0};
    std::size_t cache_shared_prefix_bytes{0};
    std::size_t completion_tokens{0};
    /// Generated reasoning tokens, excluding the closing template delimiter.
    std::size_t reasoning_tokens{0};
    /// Number of speculative verification rounds actually executed.
    std::size_t draft_rounds{0};
    std::size_t draft_tokens{0};
    std::size_t draft_accepted_tokens{0};
    std::size_t lookup_tokens{0};
    std::size_t lookup_accepted_tokens{0};
    /// Hashed n-gram embedding rows gathered for this request, split by how
    /// each was served.
    std::size_t ngram_rows{0};
    std::size_t ngram_dedup_rows{0};
    std::size_t ngram_cache_hits{0};
    std::size_t ngram_disk_reads{0};
    std::size_t prefill_tokens{0};
    std::size_t prefill_chunks{0};
    std::size_t active_decode_prefill_chunks{0};
    std::size_t max_prefill_chunk_tokens{0};
    std::size_t max_consecutive_active_prefill_chunks{0};
    std::size_t configured_active_prefill_tokens{0};
    std::size_t queue_depth_at_submit{0};
    std::size_t client_queue_depth_at_submit{0};
    std::size_t resident_requests_at_admission{0};
    std::size_t requested_logical_concurrency{1};
    std::size_t physical_execution_width{1};
    std::size_t max_buffered_output_bytes{0};
    double queue_ms{0.0};
    /// Part of queue_ms spent waiting for a concurrent request to publish
    /// the prompt prefix both share.
    double shared_prefix_wait_ms{0.0};
    double cache_restore_ms{0.0};
    double cache_snapshot_ms{0.0};
    double cache_disk_enqueue_ms{0.0};
    double cache_shared_prefix_ms{0.0};
    double ttft_ms{0.0};
    double mean_inter_token_ms{0.0};
    double max_inter_token_ms{0.0};
    double prefill_ms{0.0};
    double decode_ms{0.0};
    std::string client_id{"anonymous"};
    std::string execution_plan{"serial-c1"};
    std::string prefill_fallback_reason;
    std::string cache_miss_reason;
    std::size_t cache_common_prefix_tokens{0};
    std::size_t cache_checkpoint_tokens{0};
    FinishReason finish_reason{FinishReason::kStop};
    std::string stop_sequence;
    /// Set when the request ran dynamic abliteration detection.
    std::optional<AbliterationReport> abliteration;
    bool incremental_prefill_supported{false};
    bool cache_hit{false};
    bool cache_disk_hit{false};
    bool cancelled{false};
    /// Internal: the scheduler recorded this request's `/metrics` counters.
    bool token_metrics_recorded{false};
  };

  class GenerationRequest {
  public:
    GenerationRequest() = default;
    virtual ~GenerationRequest() = default;

    GenerationRequest(const GenerationRequest&) = delete;
    GenerationRequest& operator=(const GenerationRequest&) = delete;
    GenerationRequest(GenerationRequest&&) = delete;
    GenerationRequest& operator=(GenerationRequest&&) = delete;

    /// Streaming requests may report their start and prompt progress before
    /// any token. Backends without a queue may never report the start.
    virtual Result Wait(const TokenCallback& on_token = {},
                        const ProgressCallback& on_progress = {},
                        const StartCallback& on_start = {}) = 0;
    virtual void Cancel() noexcept = 0;
    /// Effective constrained tool format, including any schema fallback.
    /// Available before Wait and stable for this admitted request. Backends
    /// without this metadata retain the adapter's legacy format detection.
    [[nodiscard]] virtual std::optional<sampling::JsonConstraint::ToolFormat>
    ToolFormat() const {
      return std::nullopt;
    }
  };

  TextGenerationBackend() = default;
  virtual ~TextGenerationBackend() = default;

  TextGenerationBackend(const TextGenerationBackend&) = delete;
  TextGenerationBackend& operator=(const TextGenerationBackend&) = delete;
  TextGenerationBackend(TextGenerationBackend&&) = delete;
  TextGenerationBackend& operator=(TextGenerationBackend&&) = delete;

  [[nodiscard]] virtual std::string model_id() const = 0;
  /// Every name requests may select this model by, model_id() first.
  [[nodiscard]] virtual std::vector<std::string> model_ids() const {
    return {model_id()};
  }
  [[nodiscard]] bool serves_model(std::string_view name) const {
    const auto names = model_ids();
    return std::ranges::find(names, name) != names.end();
  }
  [[nodiscard]] virtual bool ready() const = 0;
  /// True once a failed generation showed that the execution device can no
  /// longer run work, for example after a GPU reset. The loss is permanent.
  [[nodiscard]] virtual bool device_lost() const { return false; }
  /// Maximum tokens accepted by the loaded model under the configured context.
  /// Zero when no text model is loaded.
  [[nodiscard]] virtual std::uint32_t max_context() const { return 0; }
  /// Live execution sessions; empty when the backend has no session pool.
  [[nodiscard]] virtual std::vector<SessionState> session_states() const {
    return {};
  }
  /// Whether the loaded backend accepts image inputs in chat requests.
  [[nodiscard]] virtual bool supports_images() const { return false; }
  /// The loaded abliteration profiles (`--abliteration-profiles`), in file
  /// name order; empty without one or on a backend that cannot apply them.
  [[nodiscard]] virtual std::vector<AbliterationInfo> abliteration_profiles()
      const {
    return {};
  }
  /// The profile a request's `abliteration.profile` names; an empty name
  /// selects the only loaded profile. Empty when no profile matches.
  [[nodiscard]] std::optional<AbliterationInfo> abliteration_profile(
      std::string_view name) const {
    auto profiles = abliteration_profiles();
    if (name.empty())
      return profiles.size() == 1 ? std::optional(std::move(profiles.front()))
                                  : std::nullopt;
    for (auto& profile : profiles)
      if (profile.name == name)
        return std::move(profile);
    return std::nullopt;
  }
  [[nodiscard]] bool supports_abliteration() const {
    return !abliteration_profiles().empty();
  }
  /// Whether requests apply SamplingConfig::han_scale (Qwen 27B and
  /// Flash-Next, whose runners build a Han token set).
  [[nodiscard]] virtual bool supports_han_scale() const { return false; }
  /// Whether chat requests may set `score_assistant` (teacher-forced
  /// logprobs of the final assistant message instead of generation).
  [[nodiscard]] virtual bool supports_scoring() const { return false; }
  /// Flash-Next's loaded HyperConnection mode (`--hc-mode`); empty for other
  /// models.
  [[nodiscard]] virtual std::string hc_mode() const { return {}; }
  /// The served model file and build revision, as llama-server's /props
  /// reports them. Empty when unknown.
  [[nodiscard]] virtual std::string model_path() const { return {}; }
  [[nodiscard]] virtual std::string build_info() const { return {}; }
  [[nodiscard]] virtual SamplingDefaults sampling_defaults() const {
    return {};
  }
  [[nodiscard]] virtual ReasoningOptions reasoning_defaults() const {
    return {};
  }
  /// The loaded tokenizer's control tokens, or null when it owns none. The
  /// parser treats a pipe-wrapped spelling as call framing only when this trie
  /// knows it; every other lookalike stays literal argument data (#383).
  [[nodiscard]] virtual InitialOutputState initial_output_state(
      const ChatRequest&) const {
    return InitialOutputState::kAuto;
  }

  virtual Result complete(std::string_view prompt, std::size_t max_tokens,
                          const sampling::SamplingConfig& sampling,
                          const CancellationCheck& is_cancelled = {},
                          const TokenCallback& on_token = {},
                          std::string_view client_id = "anonymous",
                          const std::vector<std::string>& stop_sequences = {},
                          bool cache_prompt = true,
                          bool cache_persist = true) = 0;

  virtual Result chat(const ChatRequest& request, std::size_t max_tokens,
                      const sampling::SamplingConfig& sampling,
                      const CancellationCheck& is_cancelled = {},
                      const TokenCallback& on_token = {}) = 0;

  Result complete(std::string_view prompt, std::size_t max_tokens,
                  float temperature, const CancellationCheck& is_cancelled = {},
                  const TokenCallback& on_token = {},
                  const std::vector<std::string>& stop_sequences = {}) {
    sampling::SamplingConfig config;
    config.temperature = temperature;
    return complete(prompt, max_tokens, config, is_cancelled, on_token,
                    "anonymous", stop_sequences);
  }

  Result chat(const ChatRequest& request, std::size_t max_tokens,
              float temperature, const CancellationCheck& is_cancelled = {},
              const TokenCallback& on_token = {}) {
    sampling::SamplingConfig config;
    config.temperature = temperature;
    return chat(request, max_tokens, config, is_cancelled, on_token);
  }

  /// Reserves admission before a streaming response commits successful headers.
  ///
  /// Backends with an asynchronous scheduler override this. The default
  /// request defers the existing blocking chat call until Wait().
  virtual std::shared_ptr<GenerationRequest> start_chat(
      const ChatRequest& request, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled = {}, bool stream_output = false);

  /// Raw prompt generation with admission reserved before SSE headers.
  virtual std::shared_ptr<GenerationRequest> start_complete(
      std::string_view prompt, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled = {}, bool stream_output = false,
      bool ignore_eos = false, std::string_view client_id = "anonymous",
      const std::vector<std::string>& stop_sequences = {},
      bool return_progress = false, bool cache_prompt = true,
      bool cache_persist = true);

  std::shared_ptr<GenerationRequest> start_chat(
      const ChatRequest& request, std::size_t max_tokens, float temperature,
      const CancellationCheck& is_cancelled = {}, bool stream_output = false) {
    sampling::SamplingConfig config;
    config.temperature = temperature;
    return start_chat(request, max_tokens, config, is_cancelled, stream_output);
  }

  [[nodiscard]] virtual std::size_t count_tokens(
      std::string_view text) const = 0;

  /// Prompt tokens a chat request would use after template rendering, without
  /// generating. Empty when the backend cannot tell.
  [[nodiscard]] virtual std::optional<std::size_t> count_chat_tokens(
      const ChatRequest&) const {
    return std::nullopt;
  }

  /// The prompt text a chat request renders to through the model's chat
  /// template, before tokenization. Empty when the backend cannot tell.
  [[nodiscard]] virtual std::optional<std::string> render_chat_prompt(
      const ChatRequest&) const {
    return std::nullopt;
  }

  /// Token ids of raw text as a raw prompt tokenizes: control-token spellings
  /// become their special tokens and nothing is prepended. Empty when the
  /// backend cannot tell.
  [[nodiscard]] virtual std::optional<std::vector<std::uint32_t>> tokenize(
      std::string_view) const {
    return std::nullopt;
  }
};

inline std::shared_ptr<TextGenerationBackend::GenerationRequest>
TextGenerationBackend::start_complete(
    std::string_view prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling,
    const CancellationCheck& is_cancelled, bool stream_output, bool ignore_eos,
    std::string_view client_id, const std::vector<std::string>& stop_sequences,
    bool return_progress, bool cache_prompt, bool cache_persist) {
  (void)stream_output;
  (void)return_progress;
  if (ignore_eos)
    throw std::invalid_argument("backend does not support ignore_eos");
  class DeferredGenerationRequest final : public GenerationRequest {
  public:
    DeferredGenerationRequest(TextGenerationBackend& backend,
                              std::string prompt, std::size_t max_tokens,
                              sampling::SamplingConfig sampling,
                              CancellationCheck cancellation,
                              std::string client_id,
                              std::vector<std::string> stop_sequences,
                              bool cache_prompt, bool cache_persist)
        : backend_(backend),
          prompt_(std::move(prompt)),
          max_tokens_(max_tokens),
          sampling_(sampling),
          cancellation_(std::move(cancellation)),
          client_id_(std::move(client_id)),
          stop_sequences_(std::move(stop_sequences)),
          cache_prompt_(cache_prompt),
          cache_persist_(cache_persist) {}

    Result Wait(const TokenCallback& on_token, const ProgressCallback&,
                const StartCallback&) override {
      if (waited_.exchange(true, std::memory_order_acq_rel))
        throw std::logic_error("generation request was already consumed");
      return backend_.complete(
          prompt_, max_tokens_, sampling_,
          [this] {
            return cancelled_.load(std::memory_order_acquire) ||
                   (cancellation_ && cancellation_());
          },
          on_token, client_id_, stop_sequences_, cache_prompt_, cache_persist_);
    }
    void Cancel() noexcept override {
      cancelled_.store(true, std::memory_order_release);
    }

  private:
    TextGenerationBackend& backend_;
    std::string prompt_;
    std::size_t max_tokens_;
    sampling::SamplingConfig sampling_;
    CancellationCheck cancellation_;
    std::string client_id_;
    std::vector<std::string> stop_sequences_;
    bool cache_prompt_;
    bool cache_persist_;
    std::atomic<bool> waited_{false};
    std::atomic<bool> cancelled_{false};
  };
  return std::make_shared<DeferredGenerationRequest>(
      *this, std::string(prompt), max_tokens, sampling, is_cancelled,
      std::string(client_id), stop_sequences, cache_prompt, cache_persist);
}

inline std::shared_ptr<TextGenerationBackend::GenerationRequest>
TextGenerationBackend::start_chat(const ChatRequest& request,
                                  std::size_t max_tokens,
                                  const sampling::SamplingConfig& sampling,
                                  const CancellationCheck& is_cancelled,
                                  bool stream_output) {
  (void)stream_output;
  class DeferredGenerationRequest final : public GenerationRequest {
  public:
    DeferredGenerationRequest(TextGenerationBackend& backend,
                              ChatRequest chat_request, std::size_t token_limit,
                              sampling::SamplingConfig sampling_config,
                              CancellationCheck external_cancellation)
        : backend_(backend),
          request_(std::move(chat_request)),
          max_tokens_(token_limit),
          sampling_(sampling_config),
          external_cancellation_(std::move(external_cancellation)) {}

    Result Wait(const TokenCallback& on_token, const ProgressCallback&,
                const StartCallback&) override {
      if (waited_.exchange(true, std::memory_order_acq_rel)) {
        throw std::logic_error("generation request was already consumed");
      }
      return backend_.chat(
          request_, max_tokens_, sampling_,
          [this] {
            return cancelled_.load(std::memory_order_acquire) ||
                   (external_cancellation_ && external_cancellation_());
          },
          on_token);
    }

    void Cancel() noexcept override {
      cancelled_.store(true, std::memory_order_release);
    }

  private:
    TextGenerationBackend& backend_;
    ChatRequest request_;
    std::size_t max_tokens_;
    sampling::SamplingConfig sampling_;
    CancellationCheck external_cancellation_;
    std::atomic<bool> waited_{false};
    std::atomic<bool> cancelled_{false};
  };

  return std::make_shared<DeferredGenerationRequest>(*this, request, max_tokens,
                                                     sampling, is_cancelled);
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_TEXT_GENERATION_BACKEND_HPP_
