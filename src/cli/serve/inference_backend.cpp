#include "src/cli/serve/inference_backend.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

#include "src/cli/model_artifact.hpp"
#include "src/cli/serve/component_text_cache.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/persistence_identity.hpp"
#include "src/cli/serve/request_memory_budget.hpp"
#include "src/cli/serve/score_assistant.hpp"
#include "src/cli/serve/text_generation_scheduler.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/device_code_identity.hpp"
#include "src/core/gguf_identity.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/han_script.hpp"
#include "src/core/json.hpp"
#include "src/core/sampling.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/control_tokens.hpp"
#include "src/models/qwen/generator.hpp"
#include "src/models/qwen/vision/image_cache.hpp"
#include "src/models/qwen/vision/memory.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include "src/core/hip/committed_backing.hpp"
#include "src/core/hip/heap_placement.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/host_allocations.hpp"
#include "src/core/quant/exl3/layout.hpp"
#include "src/core/speculative/speculative_verifier.hpp"
#include "src/models/deepseek_v4_flash/dspark_sampler.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/models/qwen/continuation_adapter.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/native_pack.hpp"
#include "src/models/qwen38_flash_next/continuation_adapter.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#endif

namespace gufo::server {

// A selected token ends the turn only when the request admits EOS and the token
// is one of the model's stop tokens. The engine stops there itself and the stop
// is reported by the engine, so this stays per-request state.
template<typename State, typename Tokens>
bool EndsTurn(const State& state, const Tokens& tokens, int token) {
  return state.stop_at_eos() && tokens.IsStopToken(token);
}

namespace {

using Clock = std::chrono::steady_clock;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

#if defined(ENGINE_ENABLE_HIP)

struct QwenImageContext : TextPromptContext {
  std::shared_ptr<const models::qwen::vision::Prompt> prompt;
};

/// A request with runtime abliteration ON (Flash-Next, Qwen3.8-27B): its
/// effective settings.
struct AbliterationPromptContext final : QwenImageContext {
  std::shared_ptr<const models::qwen::AbliterationSettings> settings;
};

/// A Flash-Next request without an `abliteration` member while a profile
/// with a `trigger` is loaded: it starts OFF (the OFF cache identity) and
/// its detectors may turn one profile ON mid-request (R2 A1).
struct DynamicAbliterationContext final : QwenImageContext {};

/// The profiles a model holds, in load order; Qwen3.8-27B holds at most one.
using AbliterationProfileList =
    std::vector<const models::qwen::AbliterationProfile*>;

AbliterationProfileList ListProfiles(
    const models::qwen::AbliterationProfile* profile) {
  return profile ? AbliterationProfileList{profile} : AbliterationProfileList{};
}
AbliterationProfileList ListProfiles(
    std::span<const std::shared_ptr<const models::qwen::AbliterationProfile>>
        profiles) {
  AbliterationProfileList list;
  for (const auto& profile : profiles)
    list.push_back(profile.get());
  return list;
}

std::vector<AbliterationInfo> DescribeAbliteration(
    const AbliterationProfileList& profiles) {
  using models::qwen::AbliterationDirectionKind;
  std::vector<AbliterationInfo> result;
  for (const auto* profile : profiles) {
    std::vector<std::string> kinds;
    for (const auto kind : {AbliterationDirectionKind::kMeanDiff,
                            AbliterationDirectionKind::kProjected})
      if (profile->has_kind(kind))
        kinds.emplace_back(models::qwen::AbliterationDirectionKindName(kind));
    result.push_back(AbliterationInfo{
        .name = profile->name,
        .schema = std::string(profile->schema == 2
                                  ? models::qwen::kAbliterationSchemaV2
                                  : models::qwen::kAbliterationSchemaV1),
        .identity = profile->identity,
        .layers = profile->layers,
        .direction_layer = profile->direction_layer,
        .attention = profile->attention,
        .mlp = profile->mlp,
        .residual = profile->residual,
        .direction_kinds = std::move(kinds),
        .rank = profile->rank,
        .direction_kind =
            std::string(models::qwen::AbliterationDirectionKindName(
                profile->direction_kind)),
        .direction_count = profile->direction_count,
        .preserve_norm = profile->preserve_norm,
        .subspace_method = profile->subspace_method,
        .trigger_layer = profile->trigger
                             ? std::optional(profile->trigger->layer)
                             : std::nullopt,
        .trigger_threshold = profile->trigger
                                 ? std::optional(profile->trigger->threshold)
                                 : std::nullopt,
    });
  }
  return result;
}

tokenization::ChatTemplateOptions QwenChatOptions(const ChatRequest& request,
                                                  std::uint32_t max_context) {
  auto options = tokenization::ResolveQwenChatOptions(request.reasoning,
                                                      request.add_vision_id);
  // gufo #285: the rendered prompt may be as long as the context can hold.
  options.max_output_bytes =
      tokenization::RenderedPromptBoundBytes(max_context);
  return options;
}

// The text before tokenization. An image renders as its one placeholder, not
// the visual tokens Prepare expands it into. Template errors are request
// errors, as in Prepare.
std::optional<std::string> RenderQwenPrompt(const ChatRequest& request,
                                            std::uint32_t max_context) {
  std::string error;
  auto rendered = tokenization::QwenChatTemplate::Render(
      request.messages,
      request.tool_choice == ChatRequest::ToolChoice::kNone
          ? std::span<const tokenization::ChatTool>{}
          : std::span<const tokenization::ChatTool>{request.tools},
      QwenChatOptions(request, max_context), &error);
  if (!rendered)
    throw std::invalid_argument(error);
  return rendered;
}

std::optional<std::size_t> CountQwenPromptTokens(
    const ChatRequest& request, const tokenization::QwenTokenizer& tokenizer,
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder,
    std::uint32_t max_context, models::qwen::vision::ImageCache& image_cache) {
  const auto options = QwenChatOptions(request, max_context);
  const auto tools =
      request.tool_choice == ChatRequest::ToolChoice::kNone
          ? std::span<const tokenization::ChatTool>{}
          : std::span<const tokenization::ChatTool>{request.tools};
  if (std::ranges::none_of(request.messages,
                           [](const auto& m) { return !m.images.empty(); })) {
    const auto tokens = tokenization::QwenChatTemplate::RenderAndTokenize(
        tokenizer, request.messages, tools, options);
    if (!tokens)
      return std::nullopt;
    return tokens->size();
  }
  // Reuse generation's CPU image preparation, including resize, token
  // expansion and projector validation. Counting does not encode images on
  // the GPU or admit the prompt to an execution context. Keep only the shared
  // layout's signed position limit so oversized prompts can still be sized.
  return models::qwen::vision::Prepare(
             tokenizer, request.messages, tools, options,
             encoder ? encoder->identity() : std::string_view{},
             std::numeric_limits<std::int32_t>::max(),
             request.cache_prompt ? &image_cache : nullptr, false)
      .tokens.size();
}

TextPreparedPrompt PrepareQwenPrompt(
    const ChatRequest& request, const tokenization::QwenTokenizer& tokenizer,
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder,
    std::uint32_t max_context, models::qwen::vision::ImageCache& image_cache) {
  const bool has_images = std::ranges::any_of(
      request.messages, [](const auto& m) { return !m.images.empty(); });
  const auto options = QwenChatOptions(request, max_context);
  auto prompt = std::make_shared<models::qwen::vision::Prompt>(
      models::qwen::vision::Prepare(
          tokenizer, request.messages,
          request.tool_choice == ChatRequest::ToolChoice::kNone
              ? std::span<const tokenization::ChatTool>{}
              : std::span<const tokenization::ChatTool>{request.tools},
          options,
          encoder && has_images ? encoder->identity() : std::string_view{},
          max_context, request.cache_prompt ? &image_cache : nullptr,
          request.cache_prompt));
  const auto method = models::qwen::vision::SystemPrefixMethodName(
      prompt->system_prefix_method);
  Logger::LogFormatted(LogLevel::kDebug, "cache",
                       "event=system_prefix method=%.*s tokens=%zu",
                       static_cast<int>(method.size()), method.data(),
                       prompt->system_prefix_tokens);
  if (has_images && Logger::Enabled(LogLevel::kDebug)) {
    const auto stats = image_cache.GetStats();
    Logger::LogFormatted(LogLevel::kDebug, "vision",
                         "event=image_cache enabled=%d entries=%zu "
                         "pixel_bytes=%zu encoded_bytes=%zu "
                         "hits=%zu misses=%zu evictions=%zu",
                         static_cast<int>(request.cache_prompt), stats.entries,
                         stats.pixel_bytes, stats.encoded_bytes, stats.hits,
                         stats.misses, stats.evictions);
  }
  const auto cache_prefix = prompt->stable_prefix_tokens;
  const auto system_prefix = prompt->system_prefix_tokens;
  if (prompt->images.empty())
    return {std::move(prompt->tokens), {}, cache_prefix, system_prefix};
  if (!encoder)
    throw std::invalid_argument(
        "image input requires embedded EXL3 vision weights or a matching "
        "--mmproj");
  auto context = std::make_shared<QwenImageContext>();
  context->cache_identity = prompt->cache_identity;
  for (const auto& image : prompt->images) {
    const auto identity = prompt->IdentityForPrefix(image.grid.offset);
    context->cache_prefixes.push_back(
        {image.grid.offset, {identity.begin(), identity.end()}});
  }
  context->prompt = prompt;
  return {prompt->tokens, std::move(context), cache_prefix, system_prefix};
}

std::shared_ptr<const models::qwen::vision::Prompt> QwenPrompt(
    const std::shared_ptr<const TextPromptContext>& context) {
  if (!context)
    return {};
  const auto* image = dynamic_cast<const QwenImageContext*>(context.get());
  if (!image)
    throw std::invalid_argument("invalid Qwen prompt context");
  return image->prompt;
}

/// Selects the runtime projection for one request. OFF returns `result`
/// unchanged: no new context, identity or bytes. ON resolves the request's
/// effective settings on the profile defaults and salts the text and every
/// image-prefix cache identity with their identity (profile, direction layer
/// and exact strengths), so ON/OFF or different settings never share cached
/// state.
TextPreparedPrompt SelectAbliteration(TextPreparedPrompt result,
                                      const ChatRequest& request,
                                      const AbliterationProfileList& profiles,
                                      bool dynamic = false) {
  if (!request.abliteration) {
    // An explicit OFF, and scoring (dynamic false), never detect.
    if (!dynamic || request.abliteration_specified ||
        std::ranges::none_of(profiles,
                             [](const auto* p) { return bool(p->trigger); }))
      return result;
    auto context = std::make_shared<DynamicAbliterationContext>();
    if (result.context) {
      context->prompt = QwenPrompt(result.context);
      context->cache_identity = result.context->cache_identity;
      context->cache_prefixes = result.context->cache_prefixes;
    }
    result.context = std::move(context);
    return result;
  }
  if (profiles.empty())
    throw std::invalid_argument(
        "abliteration profile is not loaded; prepare it first");
  const models::qwen::AbliterationProfile* profile = nullptr;
  if (request.abliteration_overrides.profile.empty()) {
    if (profiles.size() != 1)
      throw std::invalid_argument(
          "abliteration: name one of the loaded profiles in "
          "'abliteration.profile'");
    profile = profiles.front();
  }
  for (const auto* candidate : profiles)
    if (candidate->name == request.abliteration_overrides.profile)
      profile = candidate;
  if (!profile)
    throw std::invalid_argument("abliteration: unknown profile '" +
                                request.abliteration_overrides.profile + "'");
  auto context = std::make_shared<AbliterationPromptContext>();
  context->settings = models::qwen::ResolveAbliterationSettings(
      *profile, request.abliteration_overrides);
  const auto& identity = context->settings->identity;
  if (result.context) {
    context->prompt = QwenPrompt(result.context);
    context->cache_identity = result.context->cache_identity;
    context->cache_prefixes = result.context->cache_prefixes;
  }
  context->cache_identity = models::qwen::AbliterationInputIdentity(
      context->cache_identity, identity);
  for (auto& prefix : context->cache_prefixes)
    prefix.identity =
        models::qwen::AbliterationInputIdentity(prefix.identity, identity);
  result.context = std::move(context);
  return result;
}

/// A Qwen `score_assistant` request: the prompt before the final assistant
/// content, as generation prepares it (abliteration included), and that
/// content tokenized as message text, never as framing (#383).
TextScoringPrompt PrepareQwenScoring(
    const ChatRequest& request, const tokenization::QwenTokenizer& tokenizer,
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder,
    std::uint32_t max_context, models::qwen::vision::ImageCache& image_cache,
    const AbliterationProfileList& profiles) {
  auto split = SplitScoringRequest(request);
  tokenization::TokenizerOptions content;
  content.parse_special_tokens = false;
  auto answer = tokenizer.Encode(split.answer, content);
  LimitScoredTokens(&answer, request);
  return {SelectAbliteration(PrepareQwenPrompt(split.prompt, tokenizer, encoder,
                                               max_context, image_cache),
                             split.prompt, profiles),
          {answer.begin(), answer.end()}};
}

/// The settings a request's prompt context selects; null for OFF.
std::shared_ptr<const models::qwen::AbliterationSettings> AbliterationSelected(
    const std::shared_ptr<const TextPromptContext>& context) {
  const auto* selection =
      dynamic_cast<const AbliterationPromptContext*>(context.get());
  return selection ? selection->settings : nullptr;
}

// The Han tokens (--han-scale) of a runner whose samplers apply them, built
// once from the vocabulary while the model loads.
std::shared_ptr<const sampling::TokenSet> PrepareHanTokens(
    const TextModelRunner& runner) {
  const auto started = Clock::now();
  auto set = sampling::BuildHanTokens(runner.VocabularySize(),
                                      [&](sampling::TokenId token) {
                                        const TextRunnerToken id = token;
                                        return runner.Decode(std::span(&id, 1));
                                      });
  using Milliseconds = std::chrono::duration<double, std::milli>;
  Logger::Info(
      "loader",
      "event=han_tokens_prepared vocabulary_tokens=" +
          std::to_string(runner.VocabularySize()) +
          " han_tokens=" + std::to_string(set->count) + " prepare_ms=" +
          std::to_string(Milliseconds(Clock::now() - started).count()));
  return set;
}

// The reasoning phase a request that starts reasoning tracks: its token
// budget, or, when only the Han scale's answer scope needs that phase, an
// unlimited budget that marks where reasoning ends and never forces.
std::shared_ptr<const sampling::ReasoningBudget> RequestReasoning(
    const ChatRequest& request, const sampling::SamplingConfig& sampling,
    const std::shared_ptr<const sampling::TokenSet>& han_tokens,
    const TextModelRunner& runner, std::span<const TextRunnerToken> prompt) {
  if (request.reasoning.budget_tokens)
    return runner.BuildReasoningBudget(*request.reasoning.budget_tokens,
                                       prompt);
  if (sampling.han_scale_scope == sampling::HanScaleScope::kAnswer &&
      sampling.han_scale != 1.0F && han_tokens && han_tokens->count != 0)
    return runner.BuildReasoningBudget(sampling::ReasoningBudget::kUnlimited,
                                       prompt);
  return nullptr;
}

// Hash the sidecar and upload its weights while the model loads, before the
// runner claims state capacity and the RAM cache is sized. Otherwise the first
// image request hashes the whole sidecar in its queue time and uploads the
// weights in its prefill, after admission and the cache were sized without
// them.
void PrepareVision(
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder) {
  if (!encoder)
    return;
  const auto started = Clock::now();
  (void)encoder->identity();
  const auto hashed = Clock::now();
  try {
    encoder->Upload();
  } catch (const std::exception& e) {
    throw std::runtime_error(
        std::string("could not upload the Qwen vision encoder: ") + e.what());
  }
  const auto uploaded = Clock::now();
  std::size_t heap_free = 0, heap_total = 0;
  (void)hipMemGetInfo(&heap_free, &heap_total);
  using Milliseconds = std::chrono::duration<double, std::milli>;
  Logger::Info("loader",
               "event=vision_prepared resident_bytes=" +
                   std::to_string(encoder->ResidentBytes()) + " identity_ms=" +
                   std::to_string(Milliseconds(hashed - started).count()) +
                   " upload_ms=" +
                   std::to_string(Milliseconds(uploaded - hashed).count()) +
                   " device_heap_free_bytes=" + std::to_string(heap_free) +
                   " " + Logger::MemoryStatus());
}

constexpr std::string_view kDeepSeekStateAbi =
    "deepseek-v4-flash-gfx1151-state-v4";
constexpr std::uint32_t kQwenComponentPayloadVersion = 2;
constexpr std::size_t kQwenMaxResumeTokens = 9;

// FP16 and FP32 names are the earlier constants; KVarN differs by name.
std::string QwenStateAbi(bool speculative, hip::QwenKvCacheStorage kv_storage,
                         bool bf16_recurrent_state) {
  std::string abi = bf16_recurrent_state
                        ? (speculative ? "qwen-gfx1151-dflash-state-v4-"
                                       : "qwen-gfx1151-state-v3-")
                        : (speculative ? "qwen-gfx1151-dflash-state-v3-"
                                       : "qwen-gfx1151-state-v2-");
  abi += hip::QwenKvCacheStorageName(kv_storage);
  abi += "-kv";
  if (bf16_recurrent_state)
    abi += "-bf16-recurrent";
  return abi;
}

bool DiskCacheEnabled(const TextDiskCacheConfig& config) noexcept {
  return !config.directory.empty();
}

std::vector<std::uint8_t> DeepSeekCompatibilityIdentity(
    std::string_view artifact_fingerprint, std::string_view support_fingerprint,
    std::uint32_t max_context, std::uint32_t max_draft_tokens) {
  if (!IsSha256Hex(artifact_fingerprint)) {
    throw std::invalid_argument(
        "DeepSeek disk cache requires an artifact fingerprint");
  }
  std::ostringstream identity;
  identity << "schema=gufo-text-continuation-v2\n"
           << "model_kind=deepseek4\n"
           << "artifact_id=" << core::kGgufIdentityScheme << ':'
           << artifact_fingerprint << '\n'
           << "tokenizer=joyai-byte-bpe-v1\n"
           << "chat_template=" << models::deepseek_v4_flash::ChatTemplateId()
           << '\n'
           << "chat_template_reference_sha256="
           << models::deepseek_v4_flash::EncoderReferenceSha256() << '\n'
           << "state_abi=" << kDeepSeekStateAbi << '\n'
           << "payload_layout=ds4-rocm-v4-window128-continuation\n"
           // Cached frontiers also depend on the compiled numerical routes.
           << "numerics=ds4-scalar-verifier-v1\n"
           << "context_tokens=" << max_context << '\n'
           << "position_policy=absolute-v1\n"
           << "rope_window_policy=deepseek4-compiled-v1\n"
           << "adapters=none\n";
  identity << "support_id=" << core::kGgufIdentityScheme << ':'
           << support_fingerprint << '\n'
           << "max_draft_tokens=" << max_draft_tokens << '\n'
           << "draft_policy=dspark-cost-v5-window128\n";
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

std::vector<std::uint8_t> QwenCompatibilityIdentity(
    std::string_view artifact_fingerprint,
    std::string_view draft_artifact_fingerprint, std::uint32_t max_context,
    const hip::QwenExecutionPolicy& execution_policy, bool speculative,
    const speculative::SpeculativeOptions& speculative_options) {
  if (!IsSha256Hex(artifact_fingerprint)) {
    throw std::invalid_argument(
        "Qwen disk cache requires an artifact fingerprint");
  }
  if (speculative && !IsSha256Hex(draft_artifact_fingerprint)) {
    throw std::invalid_argument(
        "Qwen DFlash disk cache requires a draft artifact fingerprint");
  }
  const std::string state_abi =
      QwenStateAbi(speculative, execution_policy.kv_cache_storage,
                   execution_policy.UsesBf16RecurrentState());
  std::ostringstream identity;
  identity << "schema=gufo-text-continuation-v2\n"
           << "model_kind=qwen3.8\n"
           << "artifact_id=" << core::kGgufIdentityScheme << ':'
           << artifact_fingerprint << '\n'
           << "tokenizer=embedded-in-artifact\n"
           << "chat_template=qwen38-reasoning-compiled-v3\n"
           << "chat_template_reference_sha256="
           << tokenization::QwenChatTemplate::OfficialTemplateSha256() << '\n'
           << "state_abi=" << state_abi << "-component-v2\n"
           << "payload_layout=qwen-component-v2\n"
           // Older learned boundaries could keep logits from an earlier
           // prefix. Their payload has no readiness tag to distinguish them.
           << "checkpoint_frontier=complete-prefix-v1\n"
           << "numerics=qwen-bf16-fp32-prefill-v1\n"
           << "rmsnorm=fused-square-sum-v1\n"
           << "prefill_attention=visible-causal-tail-v1\n"
           << "attention_split_min_context="
           << hip::detail::kSplitKDecodeAttentionMinContext << '\n'
           << "attention_split_count="
           << hip::detail::kSplitKDecodeAttentionMaxSplits << '\n'
           << "kv_storage="
           << hip::QwenKvCacheStorageName(execution_policy.kv_cache_storage)
           << '\n'
           << "recurrent_storage="
           << (execution_policy.UsesBf16RecurrentState() ? "bf16" : "fp32")
           << '\n'
           << "context_tokens=" << max_context << '\n'
           << "position_policy=absolute-v1\n"
           << "rope_window_policy=qwen-gguf-config-v1\n"
           << "adapters=none\n";
  if (speculative) {
    identity
        << "draft_backend=dflash2-gfx1151-v1\n"
        << "draft_artifact_id=" << core::kGgufIdentityScheme << ':'
        << draft_artifact_fingerprint << '\n'
        << "draft_state_layout=dflash-window-kv-and-frontier-v4\n"
        << "draft_policy="
        << speculative::DFlashDraftPolicyName(speculative_options.dflash_policy)
        << "-v3\n"
        << "draft_max_tokens=" << speculative_options.max_draft_tokens << '\n'
        << "draft_min_tokens=" << speculative_options.min_draft_tokens << '\n'
        << "draft_initial_tokens=" << speculative_options.initial_draft_tokens
        << '\n'
        << "draft_rolling_window=" << speculative_options.rolling_window << '\n'
        << std::setprecision(std::numeric_limits<float>::max_digits10)
        << "draft_target_acceptance="
        << speculative_options.target_acceptance_rate << '\n'
        << "draft_adaptive="
        << (speculative_options.enable_adaptive_draft_length ? "true" : "false")
        << '\n'
        << "draft_batched_verification="
        << (speculative_options.use_batched_verification ? "true" : "false")
        << '\n'
        << "draft_target_verification=decode-equivalent-v1\n";
  }
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

class QwenTextRunnerState final : public TextRunnerState {
public:
  QwenTextRunnerState(std::shared_ptr<const hip::QwenGpuModel> model,
                      models::qwen::ContinuationAdapter& adapter,
                      cache::Slot& slot,
                      speculative::SpeculativeOptions options)
      : model_(std::move(model)),
        executor_(&adapter.GetExecutor(slot)),
        draft_backend_(adapter.GetDraftBackend(slot)),
        adapter_(&adapter),
        slot_(&slot),
        frontier_(adapter.GetFrontier(slot).token),
        frontier_logits_(adapter.GetFrontier(slot).logits),
        frontier_published_(adapter.GetFrontier(slot).published),
        resume_tokens_(adapter.GetFrontier(slot).pending) {
    resume_tokens_.reserve(kQwenMaxResumeTokens);
    rollback_tokens_.reserve(kQwenMaxResumeTokens);
    if (draft_backend_) {
      verifier_ = std::make_unique<speculative::SpeculativeVerifier>(
          *executor_, *draft_backend_, options);
      executor_->SetPromptHiddenCapture(true,
                                        draft_backend_->TargetHiddenLayerIds());
    }
  }

  void SetCancellationCheck(const CancellationCheck& check) override {
    executor_->SetCancellationCheck(check);
  }
  ~QwenTextRunnerState() override { executor_->SetCancellationCheck({}); }
  [[nodiscard]] models::qwen::ContinuationAdapter& adapter() const {
    return *adapter_;
  }
  [[nodiscard]] cache::Slot& slot() const { return *slot_; }

  class PrefillScope {
  public:
    explicit PrefillScope(QwenTextRunnerState& state)
        : state_(state), pending_size_(state.resume_tokens_.size()) {
      std::ranges::copy(state.resume_tokens_, pending_.begin());
      state_.adapter_->BeginPrefill(*state_.slot_);
    }
    ~PrefillScope() { state_.adapter_->FinishPrefill(*state_.slot_); }
    void Rollback() {
      state_.adapter_->RollbackPrefill(*state_.slot_);
      state_.resume_tokens_.assign(pending_.begin(),
                                   pending_.begin() + pending_size_);
    }
    PrefillScope(const PrefillScope&) = delete;
    PrefillScope& operator=(const PrefillScope&) = delete;

  private:
    QwenTextRunnerState& state_;
    std::array<TextRunnerToken, kQwenMaxResumeTokens> pending_{};
    std::size_t pending_size_;
  };
  [[nodiscard]] PrefillScope BeginPrefill() { return PrefillScope(*this); }

  void Invalidate() noexcept override { (void)InvalidateChecked(); }
  /// Invalidate, returning whether the slot (target and draft) was reset.
  [[nodiscard]] bool InvalidateChecked() noexcept {
    const bool reset = adapter_->Invalidate(*slot_);
    sequence_.clear();
    position_ = 0;
    frontier_.reset();
    frontier_logits_.clear();
    frontier_published_ = false;
    resume_tokens_.clear();
    rollback_position_.reset();
    rollback_tokens_.clear();
    rollback_logits_.clear();
    return reset;
  }

  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    try {
      auto usage = executor_->GetMemoryUsage();
      if (draft_backend_ != nullptr) {
        const auto draft = draft_backend_->GetMemoryUsage();
        usage.request_state_bytes += draft.request_state_bytes;
        usage.temporary_scratch_bytes += draft.temporary_scratch_bytes;
      }
      return {
          .per_request_state_bytes = usage.request_state_bytes,
          .temporary_scratch_bytes = usage.temporary_scratch_bytes,
      };
    } catch (...) {
      return {};
    }
  }

  [[nodiscard]] bool HasDecodeFrontier() const {
    return adapter_->HasDecodeFrontier(*slot_);
  }

  [[nodiscard]] bool speculative() const noexcept {
    return verifier_ != nullptr;
  }

  void PrimeSpeculative(std::span<const TextRunnerToken> prompt,
                        bool decode_ready) {
    if (verifier_ == nullptr) {
      throw std::logic_error("Qwen state has no speculative verifier");
    }
    frontier_ = verifier_->Prime(prompt, decode_ready);
    frontier_logits_.clear();
    if (decode_ready) {
      const auto logits = verifier_->CopyLastTargetLogits();
      frontier_logits_.assign(logits.begin(), logits.end());
    }
    sequence_.assign(prompt.begin(), prompt.end());
    position_ = prompt.size();
    frontier_published_ = false;
    resume_tokens_.clear();
    rollback_position_.reset();
  }

  void PreparePrefixReuse(std::span<const TextRunnerToken> prefix) {
    if (position_ != prefix.size() || !frontier_.has_value()) {
      throw std::logic_error(
          "Qwen retained prefix does not match its target state");
    }
    if (verifier_ != nullptr) {
      verifier_->BeginRequest();
      sequence_.assign(prefix.begin(), prefix.end());
      // Verification's frontier may be a sampled draw. A new request must
      // select from its distribution again, including when it switches to
      // greedy sampling or omits the previously published pending token.
      if (!resume_tokens_.empty() && !frontier_logits_.empty()) {
        sampling::SamplerState greedy;
        frontier_ = greedy.Sample(frontier_logits_);
      }
    }
    frontier_published_ = false;
    rollback_position_.reset();
  }

  bool ResumeGeneratedToken(TextRunnerToken token, bool decode_ready) {
    if (resume_tokens_.empty())
      return false;
    if (resume_tokens_.front() != token) {
      resume_tokens_.clear();
      return false;
    }
    resume_tokens_.erase(resume_tokens_.begin());
    // This token was already published by the previous request. Consume it
    // with decode arithmetic, exactly as AR did, before chunking the new
    // prompt suffix. Including it in a prefill chunk changes recurrent state.
    const auto position = static_cast<std::uint32_t>(position_);
    frontier_ = verifier_ != nullptr
                    ? verifier_->AdvanceCommittedToken(token, position)
                    : executor_->ForwardToken(token, position);
    if (verifier_ != nullptr)
      sequence_.push_back(token);
    ++position_;
    frontier_logits_.clear();
    if (decode_ready) {
      const auto logits = executor_->CopyLastLogits();
      frontier_logits_.assign(logits.begin(), logits.end());
    }
    return true;
  }

  void ExtendSpeculative(std::span<const TextRunnerToken> suffix,
                         bool decode_ready) {
    if (verifier_ == nullptr || !frontier_.has_value()) {
      throw std::logic_error("Qwen speculative prefix is not initialized");
    }
    if (sequence_.size() != position_) {
      throw std::logic_error(
          "Qwen speculative sequence does not match retained position");
    }
    frontier_ = verifier_->ExtendPrompt(
        suffix, static_cast<std::uint32_t>(position_), decode_ready);
    sequence_.insert(sequence_.end(), suffix.begin(), suffix.end());
    position_ += suffix.size();
    frontier_logits_.clear();
    if (decode_ready) {
      const auto logits = verifier_->CopyLastTargetLogits();
      frontier_logits_.assign(logits.begin(), logits.end());
    }
    frontier_published_ = false;
  }

  [[nodiscard]] TextRunnerToken SelectFrontier(
      sampling::SamplerState& sampler) const {
    if (!frontier_.has_value()) {
      throw std::logic_error("Qwen state has no next-token frontier");
    }
    if (sampler.config().can_use_unmodified_argmax() ||
        (verifier_ == nullptr && sampler.CanSelectArgmax(*frontier_))) {
      return *frontier_;
    }
    if (frontier_logits_.empty()) {
      return executor_->SampleLastLogits(sampler);
    }
    return executor_->SampleCachedLogits(frontier_logits_, sampler);
  }

  [[nodiscard]] bool PrepareSpeculativeDecode(std::size_t max_tokens,
                                              sampling::SamplerState& sampler,
                                              TextDecodeStep& result) {
    if (verifier_ == nullptr || !frontier_.has_value()) {
      throw std::logic_error("Qwen speculative state has no frontier");
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "Qwen speculative decode budget must be at least one token");
    }
    resume_tokens_.clear();
    if (!frontier_published_) {
      if (!sampler.config().can_use_unmodified_argmax())
        frontier_ = SelectFrontier(sampler);
      const auto logprobs =
          sampler.config().logprobs
              ? sampling::CollectLogprobs(CopyFrontierLogits(), *frontier_,
                                          sampler.config().logprobs)
              : std::nullopt;
      if (!AppendSpeculativeSelection(*frontier_, sampler, result, logprobs))
        return false;
      frontier_published_ = true;
    }
    return result.selections.size() < max_tokens;
  }

  [[nodiscard]] speculative::SpeculativeVerifier::StepRequest
  VerificationRequest(std::size_t remaining, sampling::SamplerState& sampler) {
    rollback_position_ = position_;
    rollback_tokens_.assign(1, *frontier_);
    rollback_logits_ = std::move(frontier_logits_);
    // Verification keeps its own EOS handling: the tokenizer stop set already
    // covers EOS, so a sentinel id here would change nothing. An ignore-EOS
    // request instead publishes the stop token in AppendSpeculativeSelection
    // and continues, at the cost of a window clipped on every EOS crossing.
    return {*verifier_,
            sequence_,
            static_cast<std::uint32_t>(position_),
            *frontier_,
            model_->GetTokenizer().GetEosTokenId(),
            static_cast<std::uint32_t>(std::min<std::size_t>(
                remaining, std::numeric_limits<std::uint32_t>::max())),
            sampler};
  }

  void FinishSpeculativeDecode(
      speculative::SpeculativeVerifier::StepResult verification,
      sampling::SamplerState& sampler, TextDecodeStep& result) {
    result.draft_rounds = verification.draft_count > 0 ? 1 : 0;
    result.draft_tokens = verification.draft_count;
    result.draft_accepted_tokens = verification.accepted_count;
    result.lookup_tokens = verification.lookup_tokens;
    result.lookup_accepted_tokens = verification.lookup_accepted_tokens;
    result.execution_plan = {
        .kind = verification.physical_width > 1
                    ? TextExecutionPlanKind::kBatched
                    : TextExecutionPlanKind::kSerial,
        .physical_width = verification.physical_width,
    };
    std::size_t index = 0;
    for (const TextRunnerToken token : verification.emitted_tokens) {
      auto logprobs = verification.logprobs.empty()
                          ? std::nullopt
                          : std::optional(verification.logprobs.at(index++));
      // Each prediction consumes one input, including a terminal prediction.
      // EOS itself remains the unconsumed frontier and is not published.
      ++position_;
      if (!AppendSpeculativeSelection(token, sampler, result,
                                      std::move(logprobs)))
        break;
      rollback_tokens_.push_back(token);
    }
    frontier_ = verification.next_token;
    frontier_logits_ = std::move(verification.next_token_logits);
    frontier_published_ = !result.stop;
    set_pending_token(frontier_published_ ? frontier_ : std::nullopt);
    if (verification.draft_count == 0)
      rollback_position_.reset();
  }

  void PrepareCancellation() {
    if (!rollback_position_)
      return;
    if (rollback_tokens_.empty() ||
        rollback_tokens_.size() > kQwenMaxResumeTokens ||
        rollback_logits_.empty())
      throw std::logic_error(
          "Qwen cancelled verification has no exact frontier");
    // Verification already saved the target state. Its target features are
    // still pending in the draft, so rewinding needs no new per-step copy.
    draft_backend_->DiscardPendingTargetContext(
        static_cast<std::uint32_t>(*rollback_position_));
    executor_->RestoreState();
    executor_->FinishVerification();
    position_ = *rollback_position_;
    sequence_.resize(position_);
    frontier_logits_ = std::move(rollback_logits_);
    sampling::SamplerState greedy;
    frontier_ = greedy.Sample(frontier_logits_);
    frontier_published_ = false;
    resume_tokens_ = std::move(rollback_tokens_);
    rollback_position_.reset();
  }

  [[nodiscard]] TextDecodeStep DecodeSpeculative(
      std::size_t max_tokens, sampling::SamplerState& sampler) {
    TextDecodeStep result;
    sampling::SamplerState working_sampler = sampler;
    if (PrepareSpeculativeDecode(max_tokens, working_sampler, result)) {
      const auto request = VerificationRequest(
          max_tokens - result.selections.size(), working_sampler);
      auto verification = verifier_->VerifyStep(
          sequence_, request.position, request.current_token, request.eos_id,
          request.max_emitted_tokens, working_sampler);
      FinishSpeculativeDecode(std::move(verification), working_sampler, result);
    }
    sampler.SetRngState(working_sampler.rng_state());
    return result;
  }

  [[nodiscard]] hip::QwenGpuExecutor& executor() const { return *executor_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }
  [[nodiscard]] const std::optional<TextRunnerToken>& frontier()
      const noexcept {
    return frontier_;
  }
  void set_frontier(TextRunnerToken frontier) noexcept {
    frontier_ = frontier;
    frontier_logits_.clear();
    frontier_published_ = false;
    resume_tokens_.clear();
  }
  [[nodiscard]] std::vector<float> CopyFrontierLogits() const {
    if (!frontier_logits_.empty()) {
      return frontier_logits_;
    }
    const auto logits = executor_->CopyLastLogits();
    return {logits.begin(), logits.end()};
  }
  void RestoreFrontierLogits(std::vector<float> logits) {
    frontier_logits_ = std::move(logits);
  }
  [[nodiscard]] const std::vector<TextRunnerToken>& pending_tokens()
      const noexcept {
    return resume_tokens_;
  }
  void set_pending_token(std::optional<TextRunnerToken> token) {
    resume_tokens_.clear();
    if (token)
      resume_tokens_.push_back(*token);
  }
  void RestorePendingTokens(std::span<const TextRunnerToken> tokens) {
    resume_tokens_.assign(tokens.begin(), tokens.end());
    rollback_position_.reset();
  }

private:
  bool AppendSpeculativeSelection(
      TextRunnerToken token, sampling::SamplerState& sampler,
      TextDecodeStep& result,
      std::optional<sampling::TokenLogprobs> logprobs = {}) {
    if (stop_at_eos() && model_->GetTokenizer().IsStopToken(token)) {
      result.terminal_logprobs = std::move(logprobs);
      result.stop = true;
      return false;
    }
    result.selections.push_back({
        .stop = false,
        .token = token,
        .piece = std::string(model_->GetTokenizer().DecodeToken(token)),
        .logprobs = std::move(logprobs),
    });
    sequence_.push_back(token);
    set_pending_token(token);
    sampler.Accept(token);
    return true;
  }

  std::shared_ptr<const hip::QwenGpuModel> model_;
  hip::QwenGpuExecutor* executor_{};
  std::unique_ptr<speculative::SpeculativeVerifier> verifier_;
  // Owned by the verifier for standalone execution, or by the component slot.
  hip::QwenDFlashGpuDraftBackend* draft_backend_{nullptr};
  models::qwen::ContinuationAdapter* adapter_{};
  cache::Slot* slot_{};
  std::vector<TextRunnerToken> sequence_;
  std::size_t position_{0};
  std::optional<TextRunnerToken>& frontier_;
  std::vector<float>& frontier_logits_;
  bool& frontier_published_;
  std::vector<TextRunnerToken>& resume_tokens_;
  std::optional<std::size_t> rollback_position_;
  std::vector<TextRunnerToken> rollback_tokens_;
  std::vector<float> rollback_logits_;
};

QwenTextRunnerState& RequireQwenState(TextRunnerState& state) {
  auto* qwen = dynamic_cast<QwenTextRunnerState*>(&state);
  if (qwen == nullptr) {
    throw std::logic_error("text runner state is not Qwen");
  }
  return *qwen;
}

const QwenTextRunnerState& RequireQwenState(const TextRunnerState& state) {
  const auto* qwen = dynamic_cast<const QwenTextRunnerState*>(&state);
  if (qwen == nullptr) {
    throw std::logic_error("text runner state is not Qwen");
  }
  return *qwen;
}

// Owns a four-byte device buffer and a private stream from load, so probing
// a lost context needs no allocation. HIP context loss is sticky: after a GPU
// reset even this memset fails with a hard error. Only such an error marks the
// device lost; a memset still pending at the deadline may be queued behind
// long kernels on the shared hardware queues, so it counts as usable.
class HipTextModelRunner : public TextModelRunner {
public:
  HipTextModelRunner() {
    HIP_CHECK(hipStreamCreateWithFlags(&probe_stream_, hipStreamNonBlocking));
    if (const auto error = hipMalloc(&probe_buffer_, sizeof(std::uint32_t));
        error != hipSuccess) {
      hip::LogCleanupError(hipStreamDestroy(probe_stream_));
      throw std::runtime_error(std::string("device probe allocation: ") +
                               hipGetErrorString(error));
    }
  }
  ~HipTextModelRunner() override {
    hip::LogCleanupError(hipFree(probe_buffer_));
    hip::LogCleanupError(hipStreamDestroy(probe_stream_));
  }
  HipTextModelRunner(const HipTextModelRunner&) = delete;
  HipTextModelRunner& operator=(const HipTextModelRunner&) = delete;
  HipTextModelRunner(HipTextModelRunner&&) = delete;
  HipTextModelRunner& operator=(HipTextModelRunner&&) = delete;

  [[nodiscard]] DeviceProbeStatus PollDevice() const override {
    if (!probe_pending_) {
      // Clear the failed work unit's thread-local error before probing.
      (void)hipGetLastError();
      if (hipMemsetAsync(probe_buffer_, 0, sizeof(std::uint32_t),
                         probe_stream_) != hipSuccess ||
          hipGetLastError() != hipSuccess)
        return DeviceProbeStatus::kLost;
      probe_pending_ = true;
    }
    const auto status = hipStreamQuery(probe_stream_);
    if (status == hipErrorNotReady)
      return DeviceProbeStatus::kPending;
    probe_pending_ = false;
    return status == hipSuccess ? DeviceProbeStatus::kUsable
                                : DeviceProbeStatus::kLost;
  }

  [[nodiscard]] bool DeviceUsable() const override {
    constexpr auto kTimeout = std::chrono::seconds(5);
    const auto deadline = Clock::now() + kTimeout;
    for (;;) {
      const auto status = PollDevice();
      if (status == DeviceProbeStatus::kUsable)
        return true;
      if (status == DeviceProbeStatus::kLost)
        return false;
      if (Clock::now() >= deadline) {
        Logger::Warn("scheduler", "event=device_probe_timeout timeout_s=" +
                                      std::to_string(kTimeout.count()));
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

private:
  hipStream_t probe_stream_{};
  void* probe_buffer_{};
  mutable bool probe_pending_{false};
};

template<typename AdapterType>
std::unique_ptr<ComponentCacheResources> CreateHipComponentCacheResources(
    std::unique_ptr<AdapterType> adapter, cache::ResourceLedger& ledger,
    std::function<std::size_t()> pending_memory = {},
    std::function<std::shared_ptr<void>(std::size_t)> reserve_memory = {}) {
  struct Backing {
    hip::TransferPool streams{64};
    std::unique_ptr<hip::CommittedBackingPool> pool;
    std::map<std::uint32_t, std::size_t> component_classes;
  };
  auto result = std::make_unique<ComponentCacheResources>();
  result->adapter = std::move(adapter);
  auto backing = std::make_shared<Backing>();
  result->transfer_batch_size = 16;
  const auto rounded = [](std::size_t bytes) {
    return (bytes + 4095) / 4096 * 4096;
  };
  for (const auto& d : result->adapter->Components()) {
    const auto bytes = rounded(d.kind == cache::ComponentKind::kPrivateState
                                   ? d.state_bytes
                                   : d.row_bytes * d.rows_per_chunk);
    backing->component_classes.emplace(d.id.value, bytes);
  }
  // The remainder belongs to index, histories, disk catalog and queue
  // metadata. The pools charge their own per-block allocator headroom.
  result->commit_backing = [backing, &ledger, adapter = result->adapter.get(),
                            pending_memory,
                            reserve_memory](std::size_t budget) {
    const auto plan = static_cast<AdapterType*>(adapter)->BackingPlan(budget);
    if (plan.size() != 1)
      throw std::invalid_argument("insufficient component backing budget");
    const auto& allocation = plan.front();
    // Pages commit as captures need them. A payload cannot span segments,
    // so segments of about 128 MiB bound registration pauses and stranded
    // edge space, fitting at least two of the largest payloads. Prefill
    // chunks prepare backing before GPU work, after any fallback capture.
    // Growth stops when the host keeps less than its reserve.
    std::size_t largest = 0;
    for (const auto& [id, bytes] : backing->component_classes)
      largest = std::max(largest, bytes);
    const auto block = allocation.block_bytes;
    const auto target =
        std::max<std::size_t>(std::size_t{128} << 20, 2 * largest) / block;
    const auto parts =
        std::max<std::size_t>((allocation.blocks + target - 1) / target, 1);
    const auto segment = (allocation.blocks + parts - 1) / parts * block;
    struct Limited {
      std::mutex mutex;
      std::chrono::steady_clock::time_point refused;
      bool active{};
    };
    backing->pool = std::make_unique<hip::CommittedBackingPool>(
        ledger,
        hip::BackingPoolConfig{
            allocation.blocks * allocation.block_bytes +
                allocation.metadata_bytes,
            allocation.block_bytes, allocation.metadata_bytes, segment,
            [limited = std::make_shared<Limited>(),
             pending_memory](std::size_t bytes) {
              const std::lock_guard lock(limited->mutex);
              const auto now = std::chrono::steady_clock::now();
              // Pressure retries ask once per capture; recheck at most
              // once a second while the host is short.
              if (limited->active &&
                  now - limited->refused < std::chrono::seconds(1))
                return false;
              const auto pending = pending_memory ? pending_memory() : 0;
              const auto host_ceiling = HostSnapshotCeilingBytes();
              const auto ceiling =
                  host_ceiling > pending ? host_ceiling - pending : 0;
              // Pinned host pages come out of the same device heap that
              // session and scratch allocations draw on; keep its dispatch
              // reserve so they do not fall back to managed memory.
              std::size_t heap_free = 0, heap_total = 0;
              const bool heap_fits =
                  hipMemGetInfo(&heap_free, &heap_total) == hipSuccess &&
                  heap_free >= pending &&
                  hip::HeapFitsAboveReserve(heap_free - pending, bytes);
              if (ceiling >= bytes && heap_fits) {
                limited->active = false;
                return true;
              }
              if (!limited->active)
                Logger::Warn(
                    "cache",
                    "schema=component-cache-v1 "
                    "event=backing_limited reason=" +
                        std::string(heap_fits ? "host_reserve "
                                              : "device_heap_reserve ") +
                        "segment_bytes=" + std::to_string(bytes) +
                        " host_headroom_bytes=" + std::to_string(ceiling) +
                        " device_heap_free_bytes=" + std::to_string(heap_free));
              limited->active = true;
              limited->refused = now;
              return false;
            },
            reserve_memory});
    // Prepare a bounded reserve before readiness. The first shared-prefix
    // capture needs more than one segment; preparing 2 GiB avoids moving
    // registration onto that request's capture or decode path.
    backing->pool->GrowAhead(2ULL << 30);
    backing->pool->WaitForGrowth();
    Logger::Info(
        "cache",
        "schema=component-cache-v1 event=backing_prepared committed_bytes=" +
            std::to_string(backing->pool->CommittedBytes()) +
            " long_term_pinned_bytes=" +
            std::to_string(backing->pool->LongTermPinnedBytes()));
  };
  result->prepare_prefill = [backing](std::size_t tokens) {
    // Prepare after any fallback capture, alongside actual prefill work.
    // Short turns need a smaller reserve for private states and row tails.
    backing->pool->GrowAhead((tokens >= 512 ? 2ULL : 1ULL) << 30);
  };
  result->can_capture = [backing](auto payloads) {
    return backing->pool->CanFit(payloads);
  };
  result->owner = backing;
  result->stream = [backing]() -> std::unique_ptr<cache::Stream> {
    return backing->streams.TryAcquire();
  };
  result->allocate = [backing](const cache::PayloadRequest& request) {
    // Bound the logical capture, then assign whole shared backing pages.
    const auto capacity =
        backing->component_classes.at(request.component.value);
    if (request.bytes > capacity)
      throw std::logic_error("component capture exceeds its backing class");
    const auto category =
        request.category == cache::ResourceCategory::kBackingAssigned
            ? cache::ResourceCategory::kBackingAssigned
            : request.category;
    const auto segments = backing->pool->PageCommitAllocations();
    const auto started = std::chrono::steady_clock::now();
    auto block = backing->pool->TryAcquire(category, request.bytes);
    if (backing->pool->PageCommitAllocations() != segments)
      Logger::Info(
          "cache",
          "schema=component-cache-v1 event=backing_grown committed_bytes=" +
              std::to_string(backing->pool->CommittedBytes()) + " segments=" +
              std::to_string(backing->pool->PageCommitAllocations()) +
              " long_term_pinned_bytes=" +
              std::to_string(backing->pool->LongTermPinnedBytes()) +
              " acquire_ms=" +
              std::to_string(std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - started)
                                 .count()));
    if (!block)
      throw cache::ResourceExhausted();
    // Do not start registration between a capture's allocations and copies.
    // Each prefill chunk prepares its own reserve before GPU work.
    auto owner = std::make_shared<hip::BackingBlock>(std::move(*block));
    return ComponentCaptureBuffer{
        owner->Bytes(),
        [owner] {
          owner->Convert();
          return std::move(*owner).TakePayload();
        },
        backing->pool.get(),
        [owner, request](cache::SlotLease& lease, auto source) {
          return cache::Payload::Borrowed(
              std::move(*owner).Borrow(lease, request.component, request.first,
                                       request.end, request.bytes, source));
        }};
  };
  result->staging = [](std::size_t bytes) {
    auto allocation = std::make_shared<hip::HostAllocations>();
    void* data = nullptr;
    const auto status =
        allocation->Allocate(&data, bytes, hipHostMallocCoherent);
    if (status != hipSuccess)
      throw std::runtime_error(hipGetErrorString(status));
    auto owner = std::shared_ptr<void>(std::move(allocation), data);
    std::memset(data, 0, bytes);
    return cache::StagingAllocation{std::move(owner),
                                    {static_cast<std::uint8_t*>(data), bytes}};
  };
  result->copy_observed_rows = [](auto source, auto destination,
                                  cache::Stream& stream) {
    return dynamic_cast<hip::TransferStream&>(stream).Copy(
        destination.data(), source.data(), destination.size(),
        hip::CopyDirection::kDeviceToHost, destination.size());
  };
  return result;
}

class QwenTextRunner final : public HipTextModelRunner {
public:
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kQwen;
  }
  QwenTextRunner(
      std::shared_ptr<const hip::QwenGpuModel> model, std::uint32_t max_context,
      std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model = nullptr,
      speculative::SpeculativeOptions speculative_options = {},
      std::string artifact_fingerprint = {},
      std::string draft_artifact_fingerprint = {},
      std::size_t session_count = 1, std::size_t kv_pool_bytes = 0,
      hip::QwenExecutionPolicy execution_policy =
          hip::QwenExecutionPolicy::Production())
      : model_(std::move(model)),
        dflash_model_(std::move(dflash_model)),
        max_context_(max_context),
        speculative_options_(speculative_options),
        execution_policy_(execution_policy),
        // Allocated now, before ResourceClaim reads free memory: sessions
        // then claim only their decode-step scratch.
        pass_scratch_(std::make_shared<hip::QwenSharedPassScratch>(
            model_->GetConfig(), max_context_)),
        kv_pool_(MakeKvPool(session_count, kv_pool_bytes)) {
    Logger::Info("loader",
                 "event=shared_pass_scratch rows=" +
                     std::to_string(pass_scratch_->rows()) +
                     " bytes=" + std::to_string(pass_scratch_->bytes()));
    if (!artifact_fingerprint.empty()) {
      const bool speculative = dflash_model_ != nullptr;
      persistence_ = TextRunnerPersistenceDescriptor{
          .compatibility_identity = QwenCompatibilityIdentity(
              artifact_fingerprint, draft_artifact_fingerprint, max_context_,
              execution_policy_, speculative, speculative_options_),
          .payload_version = kQwenComponentPayloadVersion,
      };
      component_identity_ = persistence_->compatibility_identity;
    } else {
      std::ostringstream identity;
      identity << "schema=gufo-process-qwen-component-v2\n"
               << "model_instance=" << model_.get() << '\n'
               << "draft_instance=" << dflash_model_.get() << '\n'
               << "context_tokens=" << max_context_ << '\n';
      const auto canonical = identity.str();
      component_identity_.assign(canonical.begin(), canonical.end());
    }
  }

  [[nodiscard]] bool UsesComponentCache() const override { return true; }
  [[nodiscard]] std::unique_ptr<ComponentCacheResources>
  CreateComponentCacheResources(cache::ResourceLedger& ledger,
                                std::size_t ram) const override {
    (void)ram;
    auto resources = CreateHipComponentCacheResources(
        std::make_unique<models::qwen::ContinuationAdapter>(
            model_, max_context_, component_identity_, execution_policy_,
            dflash_model_,
            hip::QwenDFlashGpuDraftConfig{
                .max_context = max_context_,
                .max_draft_tokens = speculative_options_.max_draft_tokens,
                .policy = speculative_options_.dflash_policy},
            &ledger, pass_scratch_, kv_pool_),
        ledger);
    // Qwen's 256-row chunks are small. Preserve a bounded group before
    // waiting, instead of synchronizing every half-MiB range individually.
    resources->reassignment_batch_size = 16;
    const auto& adapter =
        static_cast<models::qwen::ContinuationAdapter&>(*resources->adapter);
    resources->pinned_bytes = [qwen = &adapter] {
      return qwen->PinnedCaptureBytes();
    };
    return resources;
  }
  [[nodiscard]] std::unique_ptr<TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    return std::make_unique<QwenTextRunnerState>(
        model_, dynamic_cast<models::qwen::ContinuationAdapter&>(adapter), slot,
        speculative_options_);
  }
  void ReconcileComponentState(TextRunnerState& state,
                               std::size_t position) const override {
    auto& qwen = RequireQwenState(state);
    qwen.set_position(position);
    if (!qwen.frontier())
      qwen.set_frontier(
          0);  // Partial prefill: the suffix computes its frontier.
  }
  [[nodiscard]] bool HasDecodeFrontier(
      const TextRunnerState& state) const override {
    return RequireQwenState(state).HasDecodeFrontier();
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    const bool speculative_enabled = dflash_model_ != nullptr;
    return {
        .model_id = model_->GetConfig().model_name,
        .state_abi = QwenStateAbi(speculative_enabled,
                                  execution_policy_.kv_cache_storage,
                                  execution_policy_.UsesBf16RecurrentState()) +
                     "-component-v2",
        .max_context = max_context_,
        .kv_cache_type = std::string(
            hip::QwenKvCacheStorageName(execution_policy_.kv_cache_storage)),
        .kv_layout_version = models::qwen::ContinuationKvLayoutVersion(
            execution_policy_.kv_cache_storage),
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = true,
                .fork = true,
                .final_token_advance_required = !speculative_enabled,
                .incremental_text_is_exact = true,
                .multi_token_decode = speculative_enabled,
                .batched_multi_token_decode =
                    speculative_enabled && MaximumDecodeBatchWidth() > 1,
                .batched_multi_token_decode_max_width =
                    speculative_enabled ? MaximumDecodeBatchWidth() : 0,
                .prefix_reuse = true,
                .partial_prefill_retention = true,
                .prefill_rollback = true,
            },
        .persistence = persistence_,
    };
  }

  [[nodiscard]] RowBlockPool RowBlocks() const override {
    if (kv_pool_ == nullptr)
      return {};
    return {.total_blocks = kv_pool_->blocks(),
            .free_blocks = kv_pool_->free_blocks(),
            .block_rows = hip::kKvBlockTokens};
  }
  void ReserveRowBlocks(TextRunnerState& state,
                        std::size_t rows) const override {
    auto& qwen = RequireQwenState(state);
    const auto reserved = qwen.adapter().ReserveKvRows(
        qwen.slot(), static_cast<std::uint32_t>(rows), kRowBlockMargin);
    Logger::Info(
        "cache",
        "event=kv_rows_reserved rows=" + std::to_string(rows) +
            " blocks=" + std::to_string(reserved) + " mapped=" +
            std::to_string(qwen.adapter().KvBlocks(qwen.slot()).size()) +
            " free=" + std::to_string(kv_pool_->free_blocks()) +
            " total=" + std::to_string(kv_pool_->blocks()));
  }
  void ReleaseRowBlocks(TextRunnerState& state) const noexcept override {
    auto& qwen = RequireQwenState(state);
    const auto unused = qwen.adapter().ReleaseKvReservation(qwen.slot());
    try {
      Logger::Info(
          "cache",
          "event=kv_rows_released unused=" + std::to_string(unused) +
              " mapped=" +
              std::to_string(qwen.adapter().KvBlocks(qwen.slot()).size()) +
              " free=" + std::to_string(kv_pool_->free_blocks()));
    } catch (...) {
    }
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    // Sessions claim no FP16 KV: the shared pool, allocated at construction,
    // holds it for all of them and is already out of the free figure below.
    auto usage = hip::QwenGpuExecutor::EstimateMemoryUsage(
        model_->GetConfig(), max_context_, execution_policy_, true,
        kv_pool_ != nullptr);
    usage.request_state_bytes += hip::QwenGpuArena::RecurrentStateBytes(
        model_->GetConfig(), execution_policy_);
    std::size_t resident_weights = model_->GetResidentBytes();
    if (dflash_model_ != nullptr) {
      const auto draft = hip::QwenDFlashGpuExecutor::EstimateMemoryUsage(
          *dflash_model_, max_context_, MaximumDecodeBatchWidth());
      usage.request_state_bytes += draft.request_state_bytes;
      usage.temporary_scratch_bytes += draft.temporary_scratch_bytes;
      resident_weights += dflash_model_->GetPackedWeightBytes();
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      capacity = free_bytes;
    }
    // Snapshot buffers are device allocations, but on a unified-memory APU
    // they are carved from the same RAM as every host allocation, and HIP's
    // free figure does not see host pressure. Cap them by the host budget too.
    std::size_t snapshot_capacity = HostSnapshotBudgetBytes();
    std::size_t snapshot_ceiling = HostSnapshotCeilingBytes();
    if (capacity.has_value()) {
      snapshot_capacity = std::min(snapshot_capacity, *capacity);
      snapshot_ceiling = std::min(snapshot_ceiling, *capacity);
    }
    return {
        .resident_weights_bytes = resident_weights,
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = usage.request_state_bytes,
        .temporary_scratch_bytes = usage.temporary_scratch_bytes,
        .retained_snapshot_capacity_bytes = snapshot_capacity,
        .retained_snapshot_ceiling_bytes = snapshot_ceiling,
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {
            .kind = TextExecutionPlanKind::kSerial,
            .physical_width = 1,
        },
    };
    // Report actual session widths, including C6. Speculation reserves up to
    // eight token rows per session in the coordinator's existing scratch.
    for (std::size_t width = 2; width <= MaximumDecodeBatchWidth(); ++width) {
      plans.push_back({
          .kind = TextExecutionPlanKind::kBatched,
          .physical_width = width,
      });
    }
    return plans;
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->GetTokenizer().GetVocabSize(), [this](std::uint32_t id) {
          const auto& tokenizer = model_->GetTokenizer();
          return sampling::ConstraintVocabulary::Piece{
              tokenizer.DecodeTokenCopy(id), tokenizer.IsStopToken(id)};
        });
  }
  [[nodiscard]] std::uint32_t VocabularySize() const override {
    return static_cast<std::uint32_t>(model_->GetTokenizer().GetVocabSize());
  }

  // llama.cpp's Qwen3-Coder format forces "\n</think>" and also ends
  // reasoning at a tool call.
  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override {
    return {"<think>", "\n</think>", {"</think>", "<tool_call>"}};
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return model_->GetTokenizer().Encode(text);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    return tokenization::QwenChatTemplate::RenderAndTokenize(
        model_->GetTokenizer(), request.messages,
        request.tool_choice == ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools},
        QwenChatOptions(request, max_context_));
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    return SelectAbliteration(
        PrepareQwenPrompt(request, model_->GetTokenizer(),
                          model_->VisionEncoder(), max_context_, image_cache_),
        request, ListProfiles(model_->Abliteration()));
  }

  [[nodiscard]] std::optional<std::size_t> CountPromptTokens(
      const ChatRequest& request) const override {
    return CountQwenPromptTokens(request, model_->GetTokenizer(),
                                 model_->VisionEncoder(), max_context_,
                                 image_cache_);
  }

  [[nodiscard]] std::optional<std::string> RenderPrompt(
      const ChatRequest& request) const override {
    return RenderQwenPrompt(request, max_context_);
  }

  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    auto& qwen = RequireQwenState(state);
    auto abliteration = AbliterationSelected(context);
    const auto* active = qwen.executor().ActiveAbliteration();
    if (abliteration ? !active || active->identity != abliteration->identity
                     : active != nullptr) {
      // Neither the target nor the draft keeps rows of other settings. A
      // matched (live or restored) slot already has this request's settings:
      // its cache identity carries them.
      if (!qwen.InvalidateChecked())
        throw std::runtime_error(
            "Qwen slot could not be reset for an abliteration settings change");
      qwen.executor().ConfigureAbliteration(std::move(abliteration),
                                            /*state_reset=*/true);
    }
    qwen.executor().ConfigureVision(QwenPrompt(context),
                                    model_->VisionEncoder());
  }

  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    return QwenChatOptions(request, max_context_).enable_thinking
               ? TextGenerationBackend::InitialOutputState::kReasoning
               : TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->GetTokenizer().Decode(tokens);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    throw std::logic_error(
        "Qwen execution slots are owned by the component cache");
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    auto& qwen = RequireQwenState(state);
    qwen.PreparePrefixReuse(prefix);
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& qwen = RequireQwenState(state);
    // Keep the entry recurrence pinned so cancellation can stop between
    // layers without discarding already completed prompt work.
    auto activity = qwen.BeginPrefill();
    try {
      if (offset != qwen.position()) {
        throw std::logic_error(
            "Qwen prefill offset does not match retained state");
      }
      if (offset >= prompt.size()) {
        throw std::logic_error("Qwen prefill has no remaining input");
      }
      max_input_tokens = std::min<std::size_t>(
          max_input_tokens, qwen.executor().GetMaxPromptBatch());
      if (max_input_tokens != 0 && offset != 0 &&
          qwen.ResumeGeneratedToken(prompt[offset],
                                    offset + 1 == prompt.size())) {
        return {.consumed_tokens = 1,
                .decode_ready = offset + 1 == prompt.size()};
      }
      if (qwen.speculative()) {
        if (offset == 0) {
          const auto consumed = std::min(max_input_tokens, prompt.size());
          if (consumed == 0) {
            throw std::logic_error(
                "Qwen prefill requires a nonzero token budget");
          }
          qwen.PrimeSpeculative(prompt.first(consumed),
                                consumed == prompt.size());
          return {.consumed_tokens = consumed,
                  .decode_ready = consumed == prompt.size()};
        }
        const std::size_t consumed =
            std::min(max_input_tokens, prompt.size() - offset);
        if (consumed == 0) {
          throw std::logic_error(
              "Qwen DFlash retained prefix has no suffix to prefill");
        }
        qwen.ExtendSpeculative(prompt.subspan(offset, consumed),
                               offset + consumed == prompt.size());
        return {
            .consumed_tokens = consumed,
            .decode_ready = offset + consumed == prompt.size(),
        };
      }

      const std::size_t consumed =
          std::min(max_input_tokens, prompt.size() - offset);
      const bool decode_ready = offset + consumed == prompt.size();
      const auto frontier = qwen.executor().ForwardPromptBatch(
          prompt.subspan(offset, consumed), static_cast<std::uint32_t>(offset),
          decode_ready);
      qwen.set_position(offset + consumed);
      if (decode_ready) {
        qwen.set_frontier(frontier);
      }
      return {
          .consumed_tokens = consumed,
          .decode_ready = decode_ready,
      };
    } catch (const hip::QwenPrefillCancelled&) {
      activity.Rollback();
      return {.interrupted = true};
    }
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& qwen = RequireQwenState(state);
    const TextRunnerToken token = qwen.SelectFrontier(sampler);
    auto logprobs =
        sampler.config().logprobs
            ? sampling::CollectLogprobs(qwen.CopyFrontierLogits(), token,
                                        sampler.config().logprobs)
            : std::nullopt;
    if (EndsTurn(qwen, model_->GetTokenizer(), token))
      return {.stop = true,
              .token = token,
              .piece = {},
              .logprobs = std::move(logprobs)};
    // Keep the selected token separate from the argmax frontier: sampled
    // cancellation and prompt snapshot restore must not replace that argmax.
    qwen.set_pending_token(token);
    return {
        .stop = false,
        .token = token,
        .piece = std::string(model_->GetTokenizer().DecodeToken(token)),
        .logprobs = std::move(logprobs),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& qwen = RequireQwenState(state);
    if (qwen.speculative()) {
      throw std::logic_error(
          "Qwen DFlash decoding requires a multi-token decode step");
    }
    if (!qwen.frontier().has_value()) {
      throw std::logic_error("Qwen decode state has no retained frontier");
    }
    const auto frontier = qwen.executor().ForwardToken(
        token, static_cast<std::uint32_t>(qwen.position()));
    qwen.set_position(qwen.position() + 1);
    qwen.set_frontier(frontier);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    auto& qwen = RequireQwenState(state);
    if (!qwen.speculative()) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    return qwen.DecodeSpeculative(max_tokens, sampler);
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    if (dflash_model_ == nullptr || decodes.size() < 2 ||
        decodes.size() > MaximumDecodeBatchWidth()) {
      return TextModelRunner::DecodeBatch(decodes);
    }
    std::vector<TextDecodeStep> steps(decodes.size());
    std::vector<QwenTextRunnerState*> states;
    std::vector<sampling::SamplerState> samplers;
    std::vector<speculative::SpeculativeVerifier::StepRequest> requests;
    std::vector<std::size_t> indices;
    states.reserve(decodes.size());
    samplers.reserve(decodes.size());
    requests.reserve(decodes.size());
    indices.reserve(decodes.size());
    for (std::size_t index = 0; index < decodes.size(); ++index) {
      const auto& decode = decodes[index];
      auto& state = RequireQwenState(decode.state.get());
      states.push_back(&state);
      auto& sampler = samplers.emplace_back(decode.sampler.get());
      if (state.PrepareSpeculativeDecode(decode.max_tokens, sampler,
                                         steps[index])) {
        requests.push_back(state.VerificationRequest(
            decode.max_tokens - steps[index].selections.size(), sampler));
        indices.push_back(index);
      }
    }
    auto verified = speculative::SpeculativeVerifier::VerifyBatch(requests);
    for (std::size_t item = 0; item < indices.size(); ++item) {
      const std::size_t index = indices[item];
      states[index]->FinishSpeculativeDecode(std::move(verified[item]),
                                             samplers[index], steps[index]);
    }
    for (std::size_t index = 0; index < decodes.size(); ++index)
      decodes[index].sampler.get().SetRngState(samplers[index].rng_state());
    return steps;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (dflash_model_ != nullptr) {
      throw std::logic_error(
          "Qwen DFlash sessions do not support batch advance");
    }
    if (advances.size() < 2 || advances.size() > 8) {
      throw std::invalid_argument(
          "Qwen batched decode requires two to eight sessions");
    }

    std::vector<hip::QwenGpuBatchItem> items;
    std::vector<QwenTextRunnerState*> states;
    items.reserve(advances.size());
    states.reserve(advances.size());
    for (const auto& advance : advances) {
      auto& qwen = RequireQwenState(advance.state.get());
      if (!qwen.frontier().has_value()) {
        throw std::logic_error("Qwen batched state has no retained frontier");
      }
      states.push_back(&qwen);
      items.push_back({
          .executor = &qwen.executor(),
          .token_id = advance.token,
          .position = static_cast<std::uint32_t>(qwen.position()),
      });
    }

    const auto frontiers = hip::QwenGpuExecutor::ForwardTokenBatch(items);
    if (frontiers.size() != states.size()) {
      throw std::runtime_error(
          "Qwen batched decode returned an invalid frontier count");
    }
    for (std::size_t index = 0; index < states.size(); ++index) {
      states[index]->set_position(states[index]->position() + 1);
      states[index]->set_frontier(frontiers[index]);
    }
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireQwenState(state).position();
  }
  void PrepareCancellation(TextRunnerState& state) const override {
    RequireQwenState(state).PrepareCancellation();
  }

private:
  [[nodiscard]] std::size_t MaximumDecodeBatchWidth() const noexcept {
    const std::size_t rows_per_session = dflash_model_ != nullptr ? 8 : 1;
    return std::clamp<std::size_t>(max_context_ / rows_per_session, 1, 8);
  }

  mutable models::qwen::vision::ImageCache image_cache_;
  std::shared_ptr<const hip::QwenGpuModel> model_;
  std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model_;
  std::uint32_t max_context_;
  speculative::SpeculativeOptions speculative_options_;
  hip::QwenExecutionPolicy execution_policy_;
  // One prompt-pass scratch for all sessions (passes run one at a time).
  std::shared_ptr<hip::QwenSharedPassScratch> pass_scratch_;
  // FP16 attention rows of every session, taken in blocks as they write;
  // null for the FP32 reference policy.
  std::shared_ptr<hip::KvBlockPool> kv_pool_;

  // `bytes` zero: every session at the full context, the memory each would
  // otherwise hold. An explicit size is used as given.
  [[nodiscard]] std::shared_ptr<hip::KvBlockPool> MakeKvPool(
      std::size_t sessions, std::size_t bytes) const {
    if (!execution_policy_.UsesFp16AttentionKv() &&
        !execution_policy_.UsesKvarnAttentionKv()) {
      if (bytes != 0)
        throw std::invalid_argument(
            "--kv-pool-bytes needs FP16 attention KV; this policy keeps FP32");
      return nullptr;
    }
    const auto& config = model_->GetConfig();
    // FP16 K and V rows, or the KVarN record plane: the arena's planes.
    const auto planes = hip::QwenKvBlockPlanes(
        config.FullAttentionLayerCount(), config.num_key_value_heads,
        config.head_dim, execution_policy_.kv_cache_storage);
    const auto size =
        hip::SizeQwenKvPool(planes, max_context_, sessions, bytes);
    const std::size_t block_bytes = size.block_bytes;
    const std::size_t session_blocks = size.session_blocks;
    const std::size_t blocks = size.blocks;
    // One full-context session must fit.
    if (blocks < session_blocks ||
        blocks > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument(
          "--kv-pool-bytes " + std::to_string(bytes) + " holds " +
          std::to_string(blocks) + " KV blocks of " +
          std::to_string(block_bytes) + " bytes; one " +
          std::to_string(max_context_) + "-token session needs " +
          std::to_string(session_blocks) + " (" +
          std::to_string(session_blocks * block_bytes) + " bytes)");
    auto pool = std::make_shared<hip::KvBlockPool>(
        planes, static_cast<std::uint32_t>(blocks));
    Logger::Info("loader",
                 "event=kv_pool blocks=" + std::to_string(blocks) +
                     " block_tokens=" + std::to_string(hip::kKvBlockTokens) +
                     " block_bytes=" + std::to_string(block_bytes) +
                     " bytes=" + std::to_string(pool->geometry().bytes()) +
                     " sized=" + (bytes != 0 ? "explicit" : "sessions"));
    return pool;
  }
  std::optional<TextRunnerPersistenceDescriptor> persistence_;
  cache::Identity component_identity_;
};

std::vector<TextRunnerToken> DeepSeekRunnerTokens(std::span<const int> tokens) {
  std::vector<TextRunnerToken> converted;
  converted.reserve(tokens.size());
  for (const int token : tokens) {
    if (token < 0) {
      throw std::invalid_argument("DeepSeek token ID must not be negative");
    }
    converted.push_back(static_cast<TextRunnerToken>(token));
  }
  return converted;
}

std::vector<int> DeepSeekEngineTokens(std::span<const TextRunnerToken> tokens) {
  std::vector<int> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("DeepSeek token ID exceeds engine range");
    }
    converted.push_back(static_cast<int>(token));
  }
  return converted;
}

std::string_view ChatRoleName(tokenization::ChatRole role) {
  switch (role) {
    case tokenization::ChatRole::kSystem:
      return "system";
    case tokenization::ChatRole::kDeveloper:
      return "developer";
    case tokenization::ChatRole::kAssistant:
      return "assistant";
    case tokenization::ChatRole::kTool:
      return "tool";
    case tokenization::ChatRole::kUser:
      return "user";
  }
  return "user";
}

class DeepSeekTextRunnerState final : public TextRunnerState {
public:
  DeepSeekTextRunnerState(
      const std::shared_ptr<models::deepseek_v4_flash::Model>& model,
      std::uint32_t max_context, bool use_dspark) {
    std::string error;
    session_ = model->CreateSession(
        use_dspark ? gufo::core::SessionMode::kSpeculative
                   : gufo::core::SessionMode::kAutoregressive,
        max_context, &error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create DeepSeek session: " + error);
    }
  }

  void SetCancellationCheck(const CancellationCheck& is_cancelled) override {
    session_->SetCancellationCheck(is_cancelled);
  }

  void Invalidate() noexcept override {
    session_->SetCancellationCheck({});
    session_->Invalidate();
    position_ = 0;
  }

  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    const std::uint64_t bytes = session_->PayloadBytes();
    if (bytes == 0 || bytes > static_cast<std::uint64_t>(
                                  std::numeric_limits<std::size_t>::max())) {
      return {};
    }
    return {
        .per_request_state_bytes = static_cast<std::size_t>(bytes),
        .temporary_scratch_bytes = std::nullopt,
    };
  }

  [[nodiscard]] models::deepseek_v4_flash::Session& session() const {
    return *session_;
  }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

private:
  std::unique_ptr<models::deepseek_v4_flash::Session> session_;
  std::size_t position_{0};
};

class DeepSeekTextRunnerSnapshot final : public TextRunnerSnapshot {
public:
  DeepSeekTextRunnerSnapshot(
      std::shared_ptr<models::deepseek_v4_flash::Model> model,
      std::unique_ptr<models::deepseek_v4_flash::SessionSnapshot> snapshot,
      std::size_t position)
      : model(std::move(model)),
        snapshot(std::move(snapshot)),
        position(position) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    if (snapshot == nullptr ||
        snapshot->SizeBytes() > static_cast<std::uint64_t>(
                                    std::numeric_limits<std::size_t>::max())) {
      return 0;
    }
    return static_cast<std::size_t>(snapshot->SizeBytes());
  }

  std::shared_ptr<models::deepseek_v4_flash::Model> model;
  std::unique_ptr<models::deepseek_v4_flash::SessionSnapshot> snapshot;
  std::size_t position;
};

DeepSeekTextRunnerState& RequireDeepSeekState(TextRunnerState& state) {
  auto* deepseek = dynamic_cast<DeepSeekTextRunnerState*>(&state);
  if (deepseek == nullptr) {
    throw std::logic_error("text runner state is not DeepSeek");
  }
  return *deepseek;
}

const DeepSeekTextRunnerState& RequireDeepSeekState(
    const TextRunnerState& state) {
  const auto* deepseek = dynamic_cast<const DeepSeekTextRunnerState*>(&state);
  if (deepseek == nullptr) {
    throw std::logic_error("text runner state is not DeepSeek");
  }
  return *deepseek;
}

class DeepSeekTextRunner final : public HipTextModelRunner {
public:
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kDeepSeek;
  }
  DeepSeekTextRunner(std::shared_ptr<models::deepseek_v4_flash::Model> model,
                     std::uint32_t max_context, bool use_dspark,
                     std::uint32_t max_draft_tokens,
                     std::string artifact_fingerprint = {},
                     std::string support_fingerprint = {})
      : model_(std::move(model)),
        max_context_(max_context),
        use_dspark_(use_dspark),
        max_draft_tokens_(std::max(max_draft_tokens, 1u)) {
    if (!artifact_fingerprint.empty()) {
      if (use_dspark_ && !IsSha256Hex(support_fingerprint)) {
        throw std::invalid_argument(
            "DSpark disk cache requires a support SHA-256 fingerprint");
      }
      persistence_ = TextRunnerPersistenceDescriptor{
          .compatibility_identity = DeepSeekCompatibilityIdentity(
              artifact_fingerprint,
              use_dspark_ ? support_fingerprint : std::string{}, max_context_,
              max_draft_tokens_),
          .payload_version = DS4_SESSION_PAYLOAD_VERSION,
      };
    }
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    const bool dspark = use_dspark_;
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kDeepSeekStateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = true,
                .fork = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = dspark,
                .batched_multi_token_decode = dspark,
                .batched_multi_token_decode_max_width = 8u,
                .prefix_reuse = true,
            },
        .persistence = persistence_,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      capacity = free_bytes;
    }
    return {
        .resident_weights_bytes = std::nullopt,
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = std::nullopt,
        .temporary_scratch_bytes = std::nullopt,
        .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
        .retained_snapshot_ceiling_bytes = HostSnapshotCeilingBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {
            .kind = TextExecutionPlanKind::kSerial,
            .physical_width = 1,
        },
    };
    for (std::size_t width = 2; width <= 8; ++width) {
      plans.push_back({
          .kind = TextExecutionPlanKind::kBatched,
          .physical_width = width,
      });
    }
    return plans;
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->VocabSize(), [this](std::uint32_t id) {
          return sampling::ConstraintVocabulary::Piece{
              model_->DecodeToken(static_cast<int>(id)),
              model_->IsStopToken(static_cast<int>(id))};
        });
  }
  [[nodiscard]] std::uint32_t VocabularySize() const override {
    return static_cast<std::uint32_t>(model_->VocabSize());
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return DeepSeekRunnerTokens(model_->Tokenize(text));
  }
  [[nodiscard]] std::vector<TextRunnerToken> TokenizeControl(
      std::string_view text) const override {
    return DeepSeekRunnerTokens(model_->Tokenize(text, true));
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    return PrepareChatTokens(request);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> PrepareChatTokens(
      const ChatRequest& request,
      std::size_t* system_prefix_tokens = nullptr) const {
    const auto input = ConvertChat(request);
    auto tokens = DeepSeekRunnerTokens(model_->EncodeChat(
        input.messages, input.tools, input.options, system_prefix_tokens));
    if (tokens.empty()) {
      return std::nullopt;
    }
    return tokens;
  }

  [[nodiscard]] std::optional<std::string> RenderPrompt(
      const ChatRequest& request) const override {
    const auto input = ConvertChat(request);
    return models::deepseek_v4_flash::RenderChat(input.messages, input.tools,
                                                 input.options);
  }

  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    return models::deepseek_v4_flash::ResolveDeepSeekChatOptions(
               request.reasoning)
                   .enable_thinking
               ? TextGenerationBackend::InitialOutputState::kReasoning
               : TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    std::size_t system_prefix_tokens = 0;
    auto tokens = PrepareChatTokens(request, &system_prefix_tokens);
    if (!tokens)
      return std::nullopt;
    TextPreparedPrompt prepared{
        std::move(*tokens), {}, 0, system_prefix_tokens};
    {
      // DeepSeek joins adjacent user/tool messages into one user block.
      // A new user turn after a discarded assistant can therefore also change
      // the final BPE token before the generation suffix (e.g. ">\n\n").
      // Find the exact stable frontier using the renderer/tokenizer instead
      // of assuming the suffix alone is the only changed part of the prompt.
      auto continuation = request;
      continuation.messages.emplace_back(tokenization::ChatRole::kUser, "");
      const auto continued = RenderAndTokenize(continuation);
      if (continued) {
        prepared.cache_prefix_tokens = static_cast<std::size_t>(
            std::mismatch(prepared.tokens.begin(), prepared.tokens.end(),
                          continued->begin(), continued->end())
                .first -
            prepared.tokens.begin());
      }
    }
    return prepared;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    std::string text;
    for (const TextRunnerToken token : tokens) {
      if (token >
          static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("DeepSeek token ID exceeds engine range");
      }
      text += model_->DecodeToken(static_cast<int>(token));
    }
    return text;
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<DeepSeekTextRunnerState>(model_, max_context_,
                                                     use_dspark_);
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (deepseek.position() != prefix.size()) {
      throw std::logic_error(
          "DeepSeek reused prefix does not match checkpoint");
    }
    deepseek.session().BeginRequest();
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (offset != deepseek.position()) {
      throw std::logic_error(
          "DeepSeek prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error("DeepSeek prefill has no remaining input");
    }

    const std::size_t consumed =
        std::min<std::size_t>({max_input_tokens, prompt.size() - offset,
                               deepseek.session().PrefillCapacity()});
    const std::size_t next_position = offset + consumed;
    const auto prefix = DeepSeekEngineTokens(prompt.first(next_position));
    std::string error;
    if (!deepseek.session().Sync(prefix, &error)) {
      throw std::runtime_error("DeepSeek prefill failed: " + error);
    }
    deepseek.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (deepseek.position() >= max_context_)
      return {.stop = true, .piece = {}};
    // A sampled DSpark cycle may have drawn the next token already.
    int token = deepseek.session().TakePendingDsparkToken();
    std::vector<float> logits;
    if (token < 0 || sampler.config().logprobs) {
      std::string error;
      logits = deepseek.session().CopyLogits(&error);
      if (logits.empty()) {
        throw std::runtime_error("DeepSeek token selection failed: " + error);
      }
      if (token < 0)
        token = static_cast<int>(sampler.Sample(logits));
    }
    const bool stop = deepseek.stop_at_eos() && model_->IsStopToken(token);
    return {.stop = stop,
            .token = static_cast<TextRunnerToken>(token),
            .piece = stop ? std::string{} : model_->DecodeToken(token),
            .logprobs = sampling::CollectLogprobs(logits, token,
                                                  sampler.config().logprobs)};
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    if (token > static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("DeepSeek token ID exceeds engine range");
    }
    auto& deepseek = RequireDeepSeekState(state);
    std::string error;
    if (!deepseek.session().Evaluate(static_cast<int>(token), &error)) {
      throw std::runtime_error("DeepSeek decode failed: " + error);
    }
    deepseek.set_position(deepseek.position() + 1);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (deepseek.position() >= max_context_)
      return TextDecodeSelection{.stop = true, .piece = {}};
    std::string error;
    const auto logits = deepseek.session().CopyLogits(&error);
    if (logits.empty())
      throw std::runtime_error("DeepSeek token preview failed: " + error);
    const auto token = sampler.Sample(logits);
    return TextDecodeSelection{
        .stop = deepseek.stop_at_eos() &&
                model_->IsStopToken(static_cast<int>(token)),
        .token = token,
        .piece = model_->DecodeToken(static_cast<int>(token)),
        .logprobs =
            sampling::CollectLogprobs(logits, token, sampler.config().logprobs),
    };
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    if (!use_dspark_) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "DeepSeek DSpark decode budget must be at least one token");
    }

    auto& deepseek = RequireDeepSeekState(state);
    if (deepseek.position() >= max_context_)
      return {.selections = {}, .stop = true};
    const auto stats_before = deepseek.session().DsparkStatistics();
    std::optional<models::deepseek_v4_flash::DsparkSamplerBridge> bridge;
    if (!sampler.config().can_use_unmodified_argmax() ||
        sampler.config().logprobs) {
      bridge.emplace(sampler);
    }
    std::vector<int> emitted;
    std::string error;
    if (!deepseek.session().DsparkStep(
            max_tokens, max_draft_tokens_, &emitted, &error,
            bridge ? bridge->hook() : nullptr, deepseek.stop_at_eos())) {
      throw std::runtime_error("DeepSeek DSpark decode failed: " + error);
    }
    if (emitted.empty()) {
      throw std::runtime_error("DeepSeek DSpark decode produced no tokens");
    }
    if (bridge) {
      sampler.SetRngState(bridge->rng_state());
    }

    TextDecodeStep step;
    deepseek.set_position(deepseek.session().Position());
    step.selections.reserve(emitted.size());
    for (const int token : emitted) {
      auto logprobs =
          sampler.config().logprobs
              ? std::optional(bridge->logprobs().at(step.selections.size()))
              : std::nullopt;
      if (deepseek.stop_at_eos() && model_->IsStopToken(token)) {
        step.terminal_logprobs = std::move(logprobs);
        step.stop = true;
        break;
      }
      step.selections.push_back({
          .stop = false,
          .token = static_cast<TextRunnerToken>(token),
          .piece = model_->DecodeToken(token),
          .logprobs = std::move(logprobs),
      });
    }
    const auto stats_after = deepseek.session().DsparkStatistics();
    step.draft_rounds = stats_after.steps - stats_before.steps;
    step.draft_tokens =
        stats_after.support_drafted - stats_before.support_drafted;
    step.draft_accepted_tokens =
        stats_after.support_accepted - stats_before.support_accepted;
    return step;
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    if (!use_dspark_ || decodes.size() < 2 || decodes.size() > 8) {
      return TextModelRunner::DecodeBatch(decodes);
    }
    if (std::any_of(decodes.begin(), decodes.end(), [&](const auto& item) {
          return RequireDeepSeekState(item.state.get()).position() >=
                 max_context_;
        })) {
      std::vector<TextRunnerDecode> active;
      std::vector<std::size_t> active_indices;
      std::vector<TextDecodeStep> steps(decodes.size());
      for (std::size_t i = 0; i < decodes.size(); ++i) {
        if (RequireDeepSeekState(decodes[i].state.get()).position() >=
            max_context_) {
          steps[i].stop = true;
        } else {
          active.push_back(decodes[i]);
          active_indices.push_back(i);
        }
      }
      if (!active.empty()) {
        auto active_steps = DecodeBatch(active);
        for (std::size_t i = 0; i < active.size(); ++i)
          steps[active_indices[i]] = std::move(active_steps[i]);
      }
      return steps;
    }
    // Greedy and sampled requests share one DSpark cohort; the runtime keeps
    // the greedy verifier for items without a sampler.
    std::array<models::deepseek_v4_flash::SessionDsparkBatchItem, 8> items{};
    std::array<DeepSeekTextRunnerState*, 8> states{};
    std::array<models::deepseek_v4_flash::Session::DsparkStats, 8>
        stats_before{};
    std::array<std::vector<int>, 8> emitted{};
    std::array<std::optional<models::deepseek_v4_flash::DsparkSamplerBridge>, 8>
        bridges{};
    for (std::size_t index = 0; index < decodes.size(); ++index) {
      auto& deepseek = RequireDeepSeekState(decodes[index].state.get());
      states[index] = &deepseek;
      stats_before[index] = deepseek.session().DsparkStatistics();
      auto& sampler = decodes[index].sampler.get();
      if (!sampler.config().can_use_unmodified_argmax() ||
          sampler.config().logprobs) {
        bridges[index].emplace(sampler);
      }
      items[index] = {
          .session = &deepseek.session(),
          .max_tokens = decodes[index].max_tokens,
          .max_draft_tokens = max_draft_tokens_,
          .emitted = &emitted[index],
          .sampler = bridges[index] ? bridges[index]->hook() : nullptr,
          .stop_at_eos = states[index]->stop_at_eos(),
      };
    }

    std::string error;
    bool interrupted = false;
    if (!model_->DsparkStepBatch(
            std::span<const models::deepseek_v4_flash::SessionDsparkBatchItem>(
                items.data(), decodes.size()),
            &error, &interrupted)) {
      // Cancellation is rejected before sampling or device-state mutation.
      // Isolate the cancelled row rather than failing every client in its
      // batch. Ordinary cycles retain the shared physical batch.
      if (interrupted)
        return TextModelRunner::DecodeBatch(decodes);
      throw std::runtime_error("DeepSeek DSpark batch decode failed: " + error);
    }

    std::vector<TextDecodeStep> steps(decodes.size());
    for (std::size_t index = 0; index < decodes.size(); ++index) {
      if (emitted[index].empty()) {
        throw std::runtime_error("DeepSeek DSpark batch produced no tokens");
      }
      if (bridges[index]) {
        decodes[index].sampler.get().SetRngState(bridges[index]->rng_state());
      }
      auto& step = steps[index];
      step.execution_plan = {
          .kind = TextExecutionPlanKind::kBatched,
          .physical_width = decodes.size(),
      };
      states[index]->set_position(states[index]->session().Position());
      step.selections.reserve(emitted[index].size());
      for (const int token : emitted[index]) {
        auto logprobs = decodes[index].sampler.get().config().logprobs
                            ? std::optional(bridges[index]->logprobs().at(
                                  step.selections.size()))
                            : std::nullopt;
        if (states[index]->stop_at_eos() && model_->IsStopToken(token)) {
          step.terminal_logprobs = std::move(logprobs);
          step.stop = true;
          break;
        }
        step.selections.push_back({
            .stop = false,
            .token = static_cast<TextRunnerToken>(token),
            .piece = model_->DecodeToken(token),
            .logprobs = std::move(logprobs),
        });
      }
      const auto stats_after = states[index]->session().DsparkStatistics();
      step.draft_rounds = stats_after.steps - stats_before[index].steps;
      step.draft_tokens =
          stats_after.support_drafted - stats_before[index].support_drafted;
      step.draft_accepted_tokens =
          stats_after.support_accepted - stats_before[index].support_accepted;
    }
    return steps;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (advances.size() < 2 || advances.size() > 8) {
      throw std::invalid_argument(
          "DeepSeek batched decode requires two to eight sessions");
    }

    std::array<models::deepseek_v4_flash::SessionBatchItem, 8> items{};
    std::array<DeepSeekTextRunnerState*, 8> states{};
    std::size_t item_count = 0;
    for (const auto& advance : advances) {
      if (advance.token >
          static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("DeepSeek token ID exceeds engine range");
      }
      auto& deepseek = RequireDeepSeekState(advance.state.get());
      states[item_count] = &deepseek;
      items[item_count] = {
          .session = &deepseek.session(),
          .token = static_cast<int>(advance.token),
      };
      ++item_count;
    }

    std::string error;
    bool interrupted = false;
    if (!model_->EvaluateBatch(
            std::span<const models::deepseek_v4_flash::SessionBatchItem>(
                items.data(), item_count),
            &error, &interrupted)) {
      if (interrupted) {
        TextModelRunner::AdvanceBatch(advances);
        return;
      }
      throw std::runtime_error("DeepSeek batch decode failed: " + error);
    }
    for (std::size_t index = 0; index < item_count; ++index) {
      states[index]->set_position(states[index]->position() + 1);
    }
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireDeepSeekState(state).position();
  }

  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      const TextRunnerState& state) const override {
    const auto& session = RequireDeepSeekState(state).session();
    if (use_dspark_ && session.DsparkStatistics().context_tokens !=
                           static_cast<uint32_t>(session.Position())) {
      throw std::runtime_error("DSpark prefix lacks complete support state");
    }
    const std::uint64_t bytes = session.PayloadBytes();
    if (bytes == 0 || bytes > static_cast<std::uint64_t>(
                                  std::numeric_limits<std::size_t>::max())) {
      throw std::overflow_error("DeepSeek snapshot size is unavailable");
    }
    return static_cast<std::size_t>(bytes);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    const auto& deepseek = RequireDeepSeekState(state);
    std::string error;
    auto snapshot = deepseek.session().SaveSnapshot(&error);
    if (snapshot == nullptr) {
      throw std::runtime_error("DeepSeek snapshot failed: " + error);
    }
    return std::make_unique<DeepSeekTextRunnerSnapshot>(
        model_, std::move(snapshot), deepseek.position());
  }

  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& snapshot) const override {
    const auto* deepseek_snapshot =
        dynamic_cast<const DeepSeekTextRunnerSnapshot*>(&snapshot);
    if (deepseek_snapshot == nullptr ||
        deepseek_snapshot->model.get() != model_.get() ||
        deepseek_snapshot->snapshot == nullptr) {
      throw std::invalid_argument(
          "DeepSeek snapshot does not belong to this model");
    }
    auto& restored = RequireDeepSeekState(state);
    std::string error;
    if (!restored.session().RestoreSnapshot(*deepseek_snapshot->snapshot,
                                            &error)) {
      throw std::runtime_error("DeepSeek snapshot restore failed: " + error);
    }
    restored.set_position(deepseek_snapshot->position);
  }

  [[nodiscard]] std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& snapshot) const override {
    const auto* deepseek_snapshot =
        dynamic_cast<const DeepSeekTextRunnerSnapshot*>(&snapshot);
    if (deepseek_snapshot == nullptr ||
        deepseek_snapshot->model.get() != model_.get() ||
        deepseek_snapshot->snapshot == nullptr) {
      throw std::invalid_argument(
          "DeepSeek persistent snapshot does not belong to this model");
    }
    return deepseek_snapshot->PayloadBytes();
  }

  [[nodiscard]] std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const override {
    const auto* deepseek_snapshot =
        dynamic_cast<const DeepSeekTextRunnerSnapshot*>(&snapshot);
    if (deepseek_snapshot == nullptr ||
        deepseek_snapshot->model.get() != model_.get() ||
        deepseek_snapshot->snapshot == nullptr ||
        destination.size() != deepseek_snapshot->PayloadBytes() ||
        !deepseek_snapshot->snapshot->CopyTo(destination)) {
      throw std::invalid_argument(
          "DeepSeek persistent snapshot serialization failed");
    }
    return destination.size();
  }

  void StreamPersistentSnapshot(const TextRunnerSnapshot& snapshot,
                                const SnapshotSink& sink) const override {
    (void)PersistentSnapshotPayloadBytes(snapshot);
    sink(dynamic_cast<const DeepSeekTextRunnerSnapshot&>(snapshot)
             .snapshot->bytes());
  }

  void RestorePersistentSnapshot(
      TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    auto& restored = RequireDeepSeekState(state);
    std::string error;
    if (!restored.session().RestoreSnapshot(payload, &error)) {
      throw std::runtime_error("DeepSeek persistent snapshot restore failed: " +
                               error);
    }
    const int position = restored.session().Position();
    if (position <= 0 || static_cast<std::uint64_t>(position) > max_context_) {
      restored.session().Invalidate();
      throw std::runtime_error(
          "DeepSeek persistent snapshot restored an invalid position");
    }
    restored.set_position(static_cast<std::size_t>(position));
  }

private:
  struct DeepSeekChatInput {
    std::vector<models::deepseek_v4_flash::ChatMessage> messages;
    std::vector<models::deepseek_v4_flash::ChatTool> tools;
    models::deepseek_v4_flash::ChatTemplateOptions options;
  };

  [[nodiscard]] static DeepSeekChatInput ConvertChat(
      const ChatRequest& request) {
    std::vector<models::deepseek_v4_flash::ChatMessage> messages;
    messages.reserve(request.messages.size());
    for (const auto& message : request.messages) {
      models::deepseek_v4_flash::ChatMessage converted{
          .role = std::string(ChatRoleName(message.role)),
          .content = message.content + message.framing_suffix,
          .reasoning_content = message.thought,
          .tool_calls = {},
          .tool_call_id = message.tool_call_id,
      };
      converted.tool_calls.reserve(message.tool_calls.size());
      for (const auto& call : message.tool_calls) {
        models::deepseek_v4_flash::ChatMessage::ToolCall converted_call{
            .name = call.name,
            .arguments = {},
            .id = call.id,
        };
        converted_call.arguments.reserve(call.arguments.size());
        for (const auto& argument : call.arguments) {
          converted_call.arguments.push_back({
              .name = argument.name,
              .value = argument.value,
              .is_string = argument.is_string,
          });
        }
        converted.tool_calls.push_back(std::move(converted_call));
      }
      messages.push_back(std::move(converted));
    }

    std::vector<models::deepseek_v4_flash::ChatTool> tools;
    if (request.tool_choice != ChatRequest::ToolChoice::kNone) {
      tools.reserve(request.tools.size());
      for (const auto& tool : request.tools) {
        tools.push_back({
            .name = tool.name,
            .description = tool.description,
            .parameters_json = tool.parameters_json,
            .definition_json =
                tool.definition_json.empty()
                    ? std::string{}
                    : json::parse(tool.definition_json)["function"].dump(),
        });
      }
    }
    auto options = models::deepseek_v4_flash::ResolveDeepSeekChatOptions(
        request.reasoning);
    options.tools_present =
        !request.tools.empty() &&
        request.tool_choice != ChatRequest::ToolChoice::kNone;
    return {std::move(messages), std::move(tools), options};
  }

  std::shared_ptr<models::deepseek_v4_flash::Model> model_;
  std::uint32_t max_context_;
  bool use_dspark_;
  std::uint32_t max_draft_tokens_;
  std::optional<TextRunnerPersistenceDescriptor> persistence_;
};

/// Derives the content identity that keys persistent continuations for one
/// GGUF artifact. Full digests are cached by the open file metadata.
bool FingerprintArtifact(std::string_view label, const core::GgufReader& reader,
                         std::string* fingerprint, std::string* error) {
  const auto start = std::chrono::steady_clock::now();
  Logger::Info("loader", "event=load_phase phase=artifact_identity model=" +
                             std::string(label));
  try {
    *fingerprint =
        reader.GetMetadataBool(quant::exl3::kNativeKey).value_or(false)
            ? models::qwen::NativeArtifactIdentity(reader)
            : core::GgufIdentityHex(reader);
  } catch (const std::exception& exception) {
    SetError(error, std::string("Failed to fingerprint ") + std::string(label) +
                        " GGUF: " + exception.what());
    return false;
  }
  const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start)
                                .count();
  std::ostringstream message;
  message << "Fingerprinted " << label << " artifact ("
          << (reader.GetMetadataBool(quant::exl3::kNativeKey).value_or(false)
                  ? core::kNativeIdentityScheme
                  : core::kGgufIdentityScheme)
          << ", " << reader.GetTensorCount() << " tensors) in " << std::fixed
          << std::setprecision(0) << elapsed_ms << " ms";
  Logger::Info("engine", message.str());
  return true;
}

bool FingerprintArtifactFile(std::string_view label,
                             const std::filesystem::path& path,
                             std::string* fingerprint, std::string* error) {
  std::string open_error;
  const auto reader = cli::OpenModelArtifact(path, &open_error);
  if (reader == nullptr) {
    SetError(error, std::string("Failed to open ") + std::string(label) +
                        " model for fingerprinting: " + open_error);
    return false;
  }
  return FingerprintArtifact(label, *reader, fingerprint, error);
}

#endif

#if defined(ENGINE_ENABLE_HIP)
// Each Flash-Next request owns its recurrent/KV state and snapshots. Target
// projections share batches; sampling and rollback remain request-local.
constexpr std::string_view kQwenFlashNextStateAbi =
    "qwen38-flash-next-rocm-session-v1";

using QwenFlashNextModel = models::qwen38_flash_next::Model;
using QwenFlashNextSession = models::qwen38_flash_next::Session;

std::vector<std::int32_t> QwenFlashNextEngineTokens(
    std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next token ID exceeds engine range");
    }
    converted.push_back(static_cast<std::int32_t>(token));
  }
  return converted;
}

class QwenFlashNextTextRunnerState final : public TextRunnerState {
public:
  QwenFlashNextTextRunnerState(QwenFlashNextSession& session,
                               cache::Adapter& adapter, cache::Slot& slot)
      : session_(&session), adapter_(&adapter), slot_(&slot) {}
  /// A state owning its session, for runners without a component cache
  /// (KVarN K/V): the session pool's slots, invalidated by a session reset.
  explicit QwenFlashNextTextRunnerState(
      std::unique_ptr<QwenFlashNextSession> session)
      : owned_(std::move(session)), session_(owned_.get()) {}

  void Invalidate() noexcept override {
    session_->SetCancellationCheck({});
    if (adapter_ != nullptr)
      (void)adapter_->Invalidate(*slot_);
    else
      session_->ResetForReuse();
    position_ = 0;
  }
  void SetCancellationCheck(const CancellationCheck& check) override {
    session_->SetCancellationCheck(check);
  }
  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    return {.per_request_state_bytes = session_->AllocatedBytes(),
            .temporary_scratch_bytes = 0};
  }

  [[nodiscard]] QwenFlashNextSession& session() const { return *session_; }
  [[nodiscard]] models::qwen38_flash_next::ContinuationAdapter& adapter()
      const {
    if (adapter_ == nullptr)
      throw std::logic_error("Flash-Next state has no continuation adapter");
    return dynamic_cast<models::qwen38_flash_next::ContinuationAdapter&>(
        *adapter_);
  }
  [[nodiscard]] cache::Slot& slot() const { return *slot_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

private:
  std::unique_ptr<QwenFlashNextSession> owned_;
  QwenFlashNextSession* session_{};
  cache::Adapter* adapter_{};
  cache::Slot* slot_{};
  std::size_t position_{0};
};

QwenFlashNextTextRunnerState& RequireQwenFlashNextState(
    TextRunnerState& state) {
  auto* qfn = dynamic_cast<QwenFlashNextTextRunnerState*>(&state);
  if (qfn == nullptr) {
    throw std::logic_error("text runner state is not Qwen3.8-Flash-Next");
  }
  return *qfn;
}

const QwenFlashNextTextRunnerState& RequireQwenFlashNextState(
    const TextRunnerState& state) {
  const auto* qfn = dynamic_cast<const QwenFlashNextTextRunnerState*>(&state);
  if (qfn == nullptr) {
    throw std::logic_error("text runner state is not Qwen3.8-Flash-Next");
  }
  return *qfn;
}

class QwenFlashNextTextRunner final : public HipTextModelRunner {
public:
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kQwen;
  }
  QwenFlashNextTextRunner(std::shared_ptr<QwenFlashNextModel> model,
                          std::uint32_t max_context, bool use_mtp,
                          std::uint32_t max_draft_tokens,
                          std::string artifact_fingerprint = {},
                          std::string mtp_fingerprint = {},
                          std::size_t kv_pool_bytes = 0)
      : model_(std::move(model)),
        max_context_(max_context),
        use_mtp_(use_mtp),
        max_draft_tokens_(max_draft_tokens) {
    if (kv_pool_bytes) {
      const auto geometry =
          model_->KvGeometry(use_mtp_ ? core::SessionMode::kSpeculative
                                      : core::SessionMode::kAutoregressive,
                             max_context_);
      const auto blocks = kv_pool_bytes / geometry.bytes_per_block();
      if (blocks < geometry.blocks() ||
          blocks > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument(
            "Flash-Next KV pool must hold at least one full context");
      kv_pool_ = std::make_shared<hip::KvBlockPool>(
          std::vector<hip::KvBlockPlane>(geometry.planes().begin(),
                                         geometry.planes().end()),
          static_cast<std::uint32_t>(blocks));
      Logger::Info(
          "cache",
          "event=kv_pool_created model=flash-next blocks=" +
              std::to_string(blocks) +
              " block_bytes=" + std::to_string(geometry.bytes_per_block()) +
              " bytes=" + std::to_string(kv_pool_->geometry().bytes()));
    }
    // The typed in-process API owns an immutable loaded model but need not
    // supply file fingerprints when disk caching is disabled. Its model
    // lifetime is an exact process-local identity; durable caches still
    // require the validated full artifact fingerprints below.
    if (!artifact_fingerprint.empty()) {
      // Flash-Next always embeds device code: an empty hash means it could
      // not be read, and disk states of other kernel builds would match.
      const auto& device_code = core::ExecutableDeviceCodeSha256();
      if (device_code.empty())
        Logger::Warn("cache",
                     "event=device_code_identity status=unknown reason="
                     "unreadable_executable: disk checkpoints are not tied to "
                     "this build's kernels");
      else
        Logger::Info("cache",
                     "event=device_code_identity sha256=" + device_code);
      persistence_ = TextRunnerPersistenceDescriptor{
          .compatibility_identity = QwenFlashNextCompatibilityIdentity(
              artifact_fingerprint, use_mtp_ ? mtp_fingerprint : std::string{},
              use_mtp_, max_context_, max_draft_tokens_,
              model_->DecodeConcurrency(),
              models::qwen38_flash_next::RopePositionPolicy(
                  model_->config().rope_scaling),
              model_->IsNativeExl3(),
              models::qwen38_flash_next::Session::kSnapshotPayloadVersion,
              device_code,
              models::qwen38_flash_next::HcModeName(model_->hc_mode())),
          .payload_version =
              models::qwen38_flash_next::Session::kSnapshotPayloadVersion,
      };
      component_identity_ = persistence_->compatibility_identity;
    } else {
      std::ostringstream identity;
      identity << "schema=gufo-process-text-continuation-v1\n"
               << "model_instance=" << model_.get() << '\n'
               << "context_tokens=" << max_context_ << '\n'
               << "mtp=" << use_mtp_ << '\n'
               << "draft_max_tokens=" << max_draft_tokens_ << '\n'
               << "draft_cost_concurrency=" << model_->DecodeConcurrency()
               << '\n'
               << "hc_mode="
               << models::qwen38_flash_next::HcModeName(model_->hc_mode())
               << '\n';
      const auto canonical = identity.str();
      component_identity_.assign(canonical.begin(), canonical.end());
    }
  }

  [[nodiscard]] bool UsesComponentCache() const override { return true; }
  [[nodiscard]] std::unique_ptr<ComponentCacheResources>
  CreateComponentCacheResources(cache::ResourceLedger& ledger,
                                std::size_t ram) const override {
    (void)ram;
    return CreateHipComponentCacheResources(
        std::make_unique<models::qwen38_flash_next::ContinuationAdapter>(
            model_,
            use_mtp_ ? core::SessionMode::kSpeculative
                     : core::SessionMode::kAutoregressive,
            max_context_, component_identity_, kv_pool_),
        ledger,
        [budget = image_memory_, deferred = model_->DeferredScratchBytes()] {
          return budget->ReservedBytes() + deferred;
        },
        [budget = image_memory_,
         deferred = model_->DeferredScratchBytes()](std::size_t bytes) {
          try {
            return budget->Acquire(
                {bytes, 0}, [deferred] { return ImageHeadroomFor(deferred); });
          } catch (const std::length_error&) {
            return std::shared_ptr<void>{};
          }
        });
  }
  [[nodiscard]] std::unique_ptr<TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    auto& flash =
        dynamic_cast<models::qwen38_flash_next::ContinuationAdapter&>(adapter);
    return std::make_unique<QwenFlashNextTextRunnerState>(
        flash.GetSession(slot), adapter, slot);
  }
  void ReconcileComponentState(TextRunnerState& state,
                               std::size_t position) const override {
    RequireQwenFlashNextState(state).set_position(position);
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi =
            std::string(kQwenFlashNextStateAbi) +
            (model_->Kvarn()
                 ? (model_->KvarnBits() == 6 ? "+kvarn-k6v6" : "+kvarn-k4v4")
                 : ""),
        .max_context = max_context_,
        .kv_cache_type =
            model_->Kvarn()
                ? (model_->KvarnBits() == 6 ? "kvarn-k6v6" : "kvarn-k4v4")
                : "fp16",
        .kv_layout_version = 4,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = true,
                .fork = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = use_mtp_,
                .batched_multi_token_decode = use_mtp_,
                .batched_multi_token_decode_max_width = use_mtp_ ? 8u : 0u,
                .prefix_reuse = true,
                .in_pass_checkpoint = !model_->Kvarn() && !kv_pool_,
                .partial_prefill_retention = true,
                // Sparse prefill tiles group four queries from each pass
                // start: one selection block (compress ratio).
                .prefill_alignment =
                    std::max<std::size_t>(1, model_->config().compress_ratio),
            },
        .persistence = persistence_,
    };
  }

  [[nodiscard]] RowBlockPool RowBlocks() const override {
    if (!kv_pool_)
      return {};
    return {.total_blocks = kv_pool_->blocks(),
            .free_blocks = kv_pool_->free_blocks(),
            .block_rows = hip::kKvBlockTokens,
            .row_guard_rows = model_->config().compress_ratio,
            .checkpoint_blocks_retained = true,
            .progressive_row_admission = true};
  }
  void ReserveRowBlocks(TextRunnerState& state,
                        std::size_t rows) const override {
    auto& flash = RequireQwenFlashNextState(state);
    flash.adapter().ReserveKvRows(
        flash.slot(), static_cast<std::uint32_t>(rows), kRowBlockMargin);
  }
  void ReleaseRowBlocks(TextRunnerState& state) const noexcept override {
    auto& flash = RequireQwenFlashNextState(state);
    flash.adapter().ReleaseKvReservation(flash.slot());
  }

  std::string MemoryReport(const TextRunnerState& state) const override {
    const auto& flash =
        dynamic_cast<const QwenFlashNextTextRunnerState&>(state);
    const auto memory = flash.session().Memory();
    std::size_t free = 0, total = 0;
    const bool heap = hipMemGetInfo(&free, &total) == hipSuccess;
    std::ostringstream out;
    out << "session_device_total_bytes=" << memory.device_bytes
        << " session_rollback_bytes=" << memory.rollback_bytes
        << " session_vision_bytes=" << memory.vision_bytes
        << " session_graph_count=" << memory.graph_count
        << " session_host_vector_bytes=" << memory.host_vector_bytes
        << " model_resident_bytes=" << model_->ResidentBytes()
        << " deferred_scratch_budget_bytes=" << model_->DeferredScratchBytes()
        << " pending_image_and_cache_budget_bytes="
        << image_memory_->ReservedBytes() << " graph_bytes=unmeasured";
    if (heap)
      out << " device_free_bytes=" << free;
    if (kv_pool_) {
      const auto reserved = kv_pool_->reserved_blocks();
      const auto available = kv_pool_->free_blocks();
      out << " kv_pool_bytes=" << kv_pool_->geometry().bytes()
          << " kv_total_blocks=" << kv_pool_->blocks()
          << " kv_free_blocks=" << available
          << " kv_reserved_blocks=" << reserved
          << " kv_used_blocks=" << kv_pool_->blocks() - available - reserved;
    }
    return out.str();
  }

  RequestMemoryClaim ImageClaim(const TextPromptContext* context) const {
    const auto* image = dynamic_cast<const QwenImageContext*>(context);
    if (!image || !image->prompt || image->prompt->images.empty())
      return {};
    RequestMemoryClaim claim;
    claim.persistent_bytes =
        std::size_t{image->prompt->rope.PrefixLength()} * 12 + 64;
    for (const auto& input : image->prompt->images) {
      const auto memory = models::qwen::vision::EstimateImageMemory(
          input.pixels->width, input.pixels->height,
          model_->config().hidden_size,
          model_->VisionEncoder() && model_->VisionEncoder()->IsExl3());
      if (memory.embedding_bytes >
          std::numeric_limits<std::size_t>::max() - claim.persistent_bytes)
        throw std::length_error("image request memory size overflows");
      claim.persistent_bytes += memory.embedding_bytes;
      claim.scratch_bytes = std::max(claim.scratch_bytes, memory.scratch_bytes);
    }
    return claim;
  }
  static std::size_t ImageHeadroomFor(std::size_t deferred) {
    std::size_t free = 0, total = 0;
    if (hipMemGetInfo(&free, &total) != hipSuccess)
      throw std::runtime_error("cannot query image memory headroom");
    const auto reserve = kHostSnapshotHeadroomBytes + deferred;
    const auto host = HostSnapshotCeilingBytes();
    return std::min(host > deferred ? host - deferred : 0,
                    free > reserve ? free - reserve : 0);
  }
  bool RequestMemoryReady(const TextPromptContext* context) const override {
    return image_memory_->Ready(ImageClaim(context), [this] {
      return ImageHeadroomFor(model_->DeferredScratchBytes());
    });
  }
  std::shared_ptr<void> ReserveRequestMemory(
      const TextPromptContext* context) const override {
    return image_memory_->Acquire(ImageClaim(context), [this] {
      return ImageHeadroomFor(model_->DeferredScratchBytes());
    });
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      const auto deferred = model_->DeferredScratchBytes();
      capacity = free_bytes > deferred ? free_bytes - deferred : 0;
    }
    // Admission charges complete snapshot payloads. Flash-Next can retain
    // mutable state and protected K/V rows in private device storage.
    return {
        .resident_weights_bytes = model_->ResidentBytes(),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes =
            model_->BaseSessionBytes(
                use_mtp_ ? gufo::core::SessionMode::kSpeculative
                         : gufo::core::SessionMode::kAutoregressive,
                max_context_) -
            (kv_pool_ ? model_
                            ->KvGeometry(
                                use_mtp_ ? core::SessionMode::kSpeculative
                                         : core::SessionMode::kAutoregressive,
                                max_context_)
                            .bytes()
                      : 0),
        // Runtime scratch is shared and already allocated at model load;
        // reserve its remaining lazy buffers once from aggregate capacity.
        .temporary_scratch_bytes = 0,
        .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
        .retained_snapshot_ceiling_bytes = HostSnapshotCeilingBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
    for (std::size_t width = 2; width <= 8; ++width) {
      plans.push_back(
          {.kind = TextExecutionPlanKind::kBatched, .physical_width = width});
    }
    return plans;
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->tokenizer().GetVocabSize(), [this](std::uint32_t id) {
          const auto& tokenizer = model_->tokenizer();
          return sampling::ConstraintVocabulary::Piece{
              tokenizer.DecodeTokenCopy(id), tokenizer.IsStopToken(id)};
        });
  }
  [[nodiscard]] std::uint32_t VocabularySize() const override {
    return static_cast<std::uint32_t>(model_->tokenizer().GetVocabSize());
  }

  // llama.cpp's Qwen3-Coder format forces "\n</think>" and also ends
  // reasoning at a tool call.
  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override {
    return {"<think>", "\n</think>", {"</think>", "<tool_call>"}};
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return model_->tokenizer().Encode(text);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    return tokenization::QwenChatTemplate::RenderAndTokenize(
        model_->tokenizer(), request.messages,
        request.tool_choice == ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools},
        QwenChatOptions(request, max_context_));
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    return SelectAbliteration(
        PrepareQwenPrompt(request, model_->tokenizer(), model_->VisionEncoder(),
                          max_context_, image_cache_),
        request, ListProfiles(model_->AbliterationProfiles()), true);
  }

  [[nodiscard]] bool SupportsScoring() const override { return true; }
  [[nodiscard]] TextScoringPrompt PrepareScoring(
      const ChatRequest& request) const override {
    return PrepareQwenScoring(
        request, model_->tokenizer(), model_->VisionEncoder(), max_context_,
        image_cache_, ListProfiles(model_->AbliterationProfiles()));
  }

  [[nodiscard]] std::vector<sampling::TokenLogprobs> ScoreTokens(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::span<const TextRunnerToken> tokens) const override {
    auto& qfn = RequireQwenFlashNextState(state);
    if (qfn.position() > prompt.size())
      throw std::logic_error(
          "Qwen3.8-Flash-Next scoring state is past its prompt");
    std::vector<sampling::TokenLogprobs> scored;
    std::string error;
    if (!qfn.session().ScorePrompt(QwenFlashNextEngineTokens(prompt),
                                   QwenFlashNextEngineTokens(tokens), &scored,
                                   &error)) {
      qfn.set_position(0);
      throw std::runtime_error("Qwen3.8-Flash-Next scoring failed: " + error);
    }
    qfn.set_position(qfn.session().Position());
    return scored;
  }

  [[nodiscard]] std::size_t ScoringPackRows() const override {
    return QwenFlashNextSession::ScorePackRows(*model_);
  }

  // Requests whose rest fits one pass share prompt passes
  // (QwenFlashNextSession::ScoreBatch).
  void ScoreBatch(std::span<const TextRunnerScoring> requests) const override {
    std::vector<std::vector<std::int32_t>> prompts;
    std::vector<std::vector<std::int32_t>> tokens;
    std::vector<std::string> errors(requests.size());
    const auto ok = std::make_unique<bool[]>(requests.size());
    std::vector<QwenFlashNextSession::ScoreRequest> items;
    prompts.reserve(requests.size());
    tokens.reserve(requests.size());
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto& qfn = RequireQwenFlashNextState(requests[i].state.get());
      prompts.push_back(QwenFlashNextEngineTokens(requests[i].prompt));
      tokens.push_back(QwenFlashNextEngineTokens(requests[i].tokens));
      items.push_back({&qfn.session(), prompts.back(), tokens.back(),
                       requests[i].scored, &errors[i], &ok[i]});
    }
    QwenFlashNextSession::ScoreBatch(items);
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto& qfn = RequireQwenFlashNextState(requests[i].state.get());
      if (ok[i]) {
        qfn.set_position(qfn.session().Position());
        continue;
      }
      qfn.set_position(0);
      auto failure = std::make_exception_ptr(std::runtime_error(
          "Qwen3.8-Flash-Next scoring failed: " + errors[i]));
      if (!requests[i].failure)
        std::rethrow_exception(failure);
      *requests[i].failure = failure;
    }
  }

  [[nodiscard]] std::optional<std::size_t> CountPromptTokens(
      const ChatRequest& request) const override {
    return CountQwenPromptTokens(request, model_->tokenizer(),
                                 model_->VisionEncoder(), max_context_,
                                 image_cache_);
  }

  [[nodiscard]] std::optional<std::string> RenderPrompt(
      const ChatRequest& request) const override {
    return RenderQwenPrompt(request, max_context_);
  }

  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    auto& session = RequireQwenFlashNextState(state).session();
    session.ConfigureAbliteration(AbliterationSelected(context));
    session.ConfigureVision(QwenPrompt(context));
    // Detection only for a request that left `abliteration` out; an
    // explicit ON or OFF and scoring requests run none.
    session.ConfigureDynamicAbliteration(
        dynamic_cast<const DynamicAbliterationContext*>(context.get())
            ? model_->StartDynamicAbliteration()
            : nullptr);
  }

  [[nodiscard]] std::optional<AbliterationReport> DynamicAbliteration(
      TextRunnerState& state) const override {
    const auto report =
        RequireQwenFlashNextState(state).session().DynamicReport();
    if (!report)
      return std::nullopt;
    return AbliterationReport{
        .triggered = report->triggered,
        .scores = report->scores,
        .max_scores = report->max_scores,
        .position = report->position
                        ? std::optional<std::size_t>(*report->position)
                        : std::nullopt,
        .phase = report->phase,
    };
  }

  [[nodiscard]] bool LiveStateReusable(
      const TextRunnerState& state) const override {
    const auto* active =
        RequireQwenFlashNextState(state).session().ActiveAbliteration();
    return !(active && active->dynamic_from);
  }

  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    return QwenChatOptions(request, max_context_).enable_thinking
               ? TextGenerationBackend::InitialOutputState::kReasoning
               : TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->tokenizer().Decode(tokens);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    throw std::logic_error(
        "Flash-Next execution slots are owned by the component cache");
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    const auto& qfn = RequireQwenFlashNextState(state);
    if (qfn.position() != prefix.size()) {
      throw std::logic_error(
          "Qwen3.8-Flash-Next reused prefix does not match checkpoint");
    }
    qfn.session().ResetDraftPolicy();
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& qfn = RequireQwenFlashNextState(state);
    if (offset != qfn.position()) {
      throw std::logic_error(
          "Qwen3.8-Flash-Next prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error(
          "Qwen3.8-Flash-Next prefill has no remaining input");
    }
    const std::size_t remaining = prompt.size() - offset;
    const std::size_t consumed =
        StepTokens(offset, max_input_tokens, remaining,
                   remaining <= model_->PrefillThroughCapacity()
                       ? model_->PrefillThroughCapacity()
                       : model_->PrefillCapacity());
    const std::size_t next_position = offset + consumed;
    const auto prefix = QwenFlashNextEngineTokens(prompt.first(next_position));
    // The next step most likely takes the same budget; its n-gram rows are
    // read during this one and used for the tokens it shares. A step ending
    // at a cache frontier continues into the pool's lookahead.
    const auto rest = next_position == prompt.size()
                          ? state.prefill_lookahead()
                          : prompt.subspan(next_position);
    const std::size_t after = rest.size();
    const auto next = QwenFlashNextEngineTokens(rest.subspan(
        0, std::min<std::size_t>({max_input_tokens, after,
                                  after <= model_->PrefillThroughCapacity()
                                      ? model_->PrefillThroughCapacity()
                                      : model_->PrefillCapacity()})));
    std::string error;
    if (!qfn.session().Sync(prefix, &error, next)) {
      qfn.set_position(0);
      throw std::runtime_error("Qwen3.8-Flash-Next prefill failed: " + error);
    }
    qfn.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] std::optional<std::size_t> PrefillCheckpointBytes(
      const TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget,
      std::size_t boundary) const override {
    const auto& qfn = RequireQwenFlashNextState(state);
    const auto consumed = StepTokens(offset, budget, prompt.size() - offset,
                                     model_->PrefillThroughCapacity());
    if (boundary <= offset + 1 || boundary >= offset + consumed ||
        offset + consumed - boundary > model_->PrefillCheckpointTail())
      return std::nullopt;
    const auto bytes = qfn.session().PrefillCheckpointBytes(
        static_cast<std::uint32_t>(boundary));
    return bytes ? std::optional<std::size_t>(bytes) : std::nullopt;
  }

  [[nodiscard]] TextPrefillStep PrefillThrough(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget, std::size_t boundary,
      std::unique_ptr<TextRunnerSnapshot>* checkpoint,
      const std::function<std::unique_ptr<TextRunnerSnapshot>()>& capture)
      const override {
    auto& qfn = RequireQwenFlashNextState(state);
    if (!checkpoint || !capture || offset != qfn.position())
      throw std::logic_error("invalid component in-pass checkpoint request");
    const auto consumed = StepTokens(offset, budget, prompt.size() - offset,
                                     model_->PrefillThroughCapacity());
    const auto next_position = offset + consumed;
    const auto prefix = QwenFlashNextEngineTokens(prompt.first(next_position));
    double capture_ms = 0;
    try {
      qfn.adapter().PrefillThrough(
          qfn.slot(), prefix, static_cast<std::uint32_t>(boundary), [&] {
            const auto started = std::chrono::steady_clock::now();
            try {
              *checkpoint = capture();
            } catch (...) {
              // Optional retention may refuse; the complete live prefill is
              // valid.
              checkpoint->reset();
            }
            capture_ms += std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - started)
                              .count();
          });
    } catch (...) {
      qfn.set_position(0);
      throw;
    }
    qfn.set_position(next_position);
    return {.consumed_tokens = consumed,
            .decode_ready = next_position == prompt.size(),
            .checkpoint_ms = capture_ms};
  }

  // The engine stops before committing EOS when the request enables it.
  // Every returned token is committed work and is emitted exactly once.
  void ResolveStepTokens(
      QwenFlashNextTextRunnerState&, bool engine_stopped,
      std::span<const std::int32_t> tokens, TextDecodeStep* step,
      const QwenFlashNextSession::DecodeResult& decoded) const {
    step->stop = engine_stopped;
    step->terminal_logprobs = decoded.terminal_logprobs;
    step->selections.reserve(tokens.size());
    for (const std::int32_t token : tokens) {
      step->selections.push_back({
          .stop = false,
          .token = static_cast<TextRunnerToken>(token),
          .piece = model_->TokenText(token),
          .logprobs =
              decoded.logprobs.empty()
                  ? std::nullopt
                  : std::optional(decoded.logprobs.at(step->selections.size())),
      });
    }
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& qfn = RequireQwenFlashNextState(state);
    if (qfn.position() >= max_context_) {
      return {.stop = true, .piece = {}};
    }
    const auto logits = qfn.session().Logits();
    if (logits.empty()) {
      throw std::runtime_error(
          "Qwen3.8-Flash-Next token selection has no logits");
    }
    const auto token = static_cast<std::int32_t>(sampler.Sample(logits));
    const bool stop = EndsTurn(qfn, *model_, token);
    return {.stop = stop,
            .token = static_cast<TextRunnerToken>(token),
            .piece = stop ? std::string{} : model_->TokenText(token),
            .logprobs = sampling::CollectLogprobs(logits, token,
                                                  sampler.config().logprobs)};
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next token ID exceeds engine range");
    }
    auto& qfn = RequireQwenFlashNextState(state);
    std::string error;
    if (!qfn.session().Evaluate(static_cast<std::int32_t>(token), &error)) {
      throw std::runtime_error("Qwen3.8-Flash-Next decode failed: " + error);
    }
    qfn.set_position(qfn.position() + 1);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    if (!use_mtp_ || max_tokens == 1) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next MTP decode budget must be at least one token");
    }
    auto& qfn = RequireQwenFlashNextState(state);
    if (qfn.position() >= max_context_) {
      return {.selections = {}, .stop = true};
    }
    const auto stats_before = qfn.session().Statistics();
    sampling::SamplerState working_sampler = sampler;
    QwenFlashNextSession::DecodeResult decoded;
    std::string error;
    // Use the same request policy as SelectNext and the batched MTP path.
    // The engine bounds the physical block and stops at EOS. Preserve the
    // remaining request budget so its policy can recognize a short tail.
    // A step taken while the scheduler holds another request never verifies
    // wide: the peer would wait for the whole pass.
    const auto residents = static_cast<std::uint32_t>(std::min<std::size_t>(
        qfn.resident_requests(), std::numeric_limits<std::uint32_t>::max()));
    if (!qfn.session().DecodeStep(max_tokens, working_sampler, &decoded, &error,
                                  qfn.stop_at_eos(), 1, residents)) {
      throw std::runtime_error("Qwen3.8-Flash-Next MTP decode failed: " +
                               error);
    }
    // The pool accepts the returned tokens once. Publish the RNG and residual
    // draw so the next batch retains the rejection-conditioned distribution.
    sampler.CopyDrawStateFrom(working_sampler);
    TextDecodeStep step;
    ResolveStepTokens(qfn, decoded.stop, decoded.tokens, &step, decoded);
    qfn.set_position(qfn.session().Position());
    const auto stats_after = qfn.session().Statistics();
    step.draft_rounds = stats_after.cycles - stats_before.cycles;
    step.draft_tokens = stats_after.drafted - stats_before.drafted;
    step.draft_accepted_tokens = stats_after.accepted - stats_before.accepted;
    step.lookup_tokens = stats_after.lookup - stats_before.lookup;
    step.lookup_accepted_tokens =
        stats_after.lookup_accepted - stats_before.lookup_accepted;
    return step;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (advances.size() < 2) {
      return TextModelRunner::AdvanceBatch(advances);
    }
    std::vector<QwenFlashNextSession::BatchOutcome> outcomes(advances.size());
    std::vector<QwenFlashNextSession::AdvanceRequest> requests;
    for (const auto& advance : advances) {
      if (advance.token > static_cast<TextRunnerToken>(
                              std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Flash-Next token exceeds engine range");
      }
      requests.push_back(
          {&RequireQwenFlashNextState(advance.state.get()).session(),
           static_cast<std::int32_t>(advance.token),
           &outcomes[requests.size()]});
    }
    std::string error;
    (void)QwenFlashNextSession::EvaluateBatch(requests, &error);
    for (std::size_t i = 0; i < advances.size(); ++i) {
      const auto& advance = advances[i];
      if (!outcomes[i].completed) {
        auto failure = std::make_exception_ptr(std::runtime_error(
            "Flash-Next advance failed: " + outcomes[i].error));
        if (!advance.failure)
          std::rethrow_exception(failure);
        *advance.failure = failure;
        continue;
      }
      auto& state = RequireQwenFlashNextState(advance.state.get());
      state.set_position(state.session().Position());
    }
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    if (decodes.size() < 2 || !use_mtp_) {
      return TextModelRunner::DecodeBatch(decodes);
    }
    const auto count = decodes.size();
    std::vector<sampling::SamplerState> samplers;
    std::vector<QwenFlashNextSession::DecodeResult> results(count);
    std::vector<QwenFlashNextSession::BatchOutcome> outcomes(count);
    std::vector<QwenFlashNextSession::SpeculativeStats> before;
    std::vector<QwenFlashNextSession::DecodeRequest> requests;
    samplers.reserve(count);
    before.reserve(count);
    requests.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      auto& state = RequireQwenFlashNextState(decodes[i].state.get());
      auto& session = state.session();
      auto& sampler = samplers.emplace_back(decodes[i].sampler.get());
      before.push_back(session.Statistics());
      requests.push_back({&session, decodes[i].max_tokens, &sampler,
                          &results[i], state.stop_at_eos(), &outcomes[i]});
    }
    std::string error;
    (void)QwenFlashNextSession::DecodeBatch(requests, &error);
    const auto active_count = static_cast<std::size_t>(std::count_if(
        results.begin(), results.end(),
        [](const auto& result) { return !result.tokens.empty(); }));
    std::vector<TextDecodeStep> steps(count);
    for (std::size_t i = 0; i < count; ++i) {
      if (!outcomes[i].completed) {
        steps[i].failure = std::make_exception_ptr(
            std::runtime_error("Flash-Next MTP failed: " + outcomes[i].error));
        continue;
      }
      auto& state = RequireQwenFlashNextState(decodes[i].state.get());
      decodes[i].sampler.get().CopyDrawStateFrom(samplers[i]);
      state.set_position(state.session().Position());
      auto& step = steps[i];
      ResolveStepTokens(state, results[i].stop, results[i].tokens, &step,
                        results[i]);
      if (active_count > 1 && !results[i].tokens.empty()) {
        step.execution_plan = {.kind = TextExecutionPlanKind::kBatched,
                               .physical_width = active_count};
      }
      const auto stats = state.session().Statistics();
      step.draft_rounds = stats.cycles - before[i].cycles;
      step.draft_tokens = stats.drafted - before[i].drafted;
      step.draft_accepted_tokens = stats.accepted - before[i].accepted;
      // A request whose peers dropped out runs a single-session step,
      // which can use prompt lookup.
      step.lookup_tokens = stats.lookup - before[i].lookup;
      step.lookup_accepted_tokens =
          stats.lookup_accepted - before[i].lookup_accepted;
    }
    return steps;
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireQwenFlashNextState(state).position();
  }

  [[nodiscard]] TextRunnerNgramRows NgramRows(
      const TextRunnerState& state) const override {
    const auto totals =
        RequireQwenFlashNextState(state).session().NgramRowTotals();
    return {totals.rows, totals.dedup_rows, totals.cache_hits,
            totals.disk_reads};
  }

private:
  /// Tokens one prefill step takes from `offset`. A step the pass capacity
  /// ends before the prompt stops on a selection block (prefill_alignment),
  /// as the serving runner ends budget-limited ones, so the next step
  /// continues the prompt exactly from an unaligned reused frontier too.
  [[nodiscard]] std::size_t StepTokens(std::size_t offset, std::size_t budget,
                                       std::size_t remaining,
                                       std::size_t capacity) const noexcept {
    auto count = std::min({budget, remaining, capacity});
    const std::size_t ratio =
        std::max<std::uint32_t>(1, model_->config().compress_ratio);
    if (const auto over = (offset + count) % ratio;
        count < remaining && count > over)
      count -= over;
    return count;
  }
  std::shared_ptr<RequestMemoryBudget> image_memory_ =
      std::make_shared<RequestMemoryBudget>();
  mutable models::qwen::vision::ImageCache image_cache_;
  std::shared_ptr<QwenFlashNextModel> model_;
  std::shared_ptr<hip::KvBlockPool> kv_pool_;
  std::uint32_t max_context_;
  bool use_mtp_;
  std::uint32_t max_draft_tokens_;
  std::optional<TextRunnerPersistenceDescriptor> persistence_;
  cache::Identity component_identity_;
};
#endif

}  // namespace

#if defined(ENGINE_ENABLE_HIP)
std::shared_ptr<TextModelRunner> CreateQwenFlashNextTextRunner(
    std::shared_ptr<models::qwen38_flash_next::Model> model,
    std::uint32_t max_context) {
  return std::make_shared<QwenFlashNextTextRunner>(std::move(model),
                                                   max_context, false, 0);
}
#endif

struct InferenceBackend::Impl {
#if defined(ENGINE_ENABLE_HIP)
  struct State {
    std::shared_ptr<TextGenerationScheduler> scheduler;
    std::string model_id;
    std::vector<std::string> extra_model_ids;
    SamplingDefaults sampling_defaults;
    std::uint32_t max_context{0};
    bool supports_images{false};
    /// The loaded abliteration profiles; empty without one.
    std::vector<AbliterationInfo> abliteration;
    /// Flash-Next's HyperConnection mode name; empty for other models.
    std::string hc_mode;
    /// The runner's Han tokens; null where its samplers ignore han_scale.
    std::shared_ptr<const sampling::TokenSet> han_tokens;
    ReasoningOptions reasoning_defaults;
    std::string model_path;
    std::string build_info;

    /// The sampling the runner applies, so traces report what it uses.
    [[nodiscard]] sampling::SamplingConfig Applied(
        sampling::SamplingConfig config) const {
      // Without a set the scale stays inactive (SamplingConfig::han_bias).
      config.han_tokens = han_tokens;
      return config;
    }
  };

  class ScheduledGenerationRequest final : public GenerationRequest {
  public:
    ScheduledGenerationRequest(
        std::shared_ptr<const State> model_state,
        TextGenerationScheduler::Request scheduled_request,
        InitialOutputState initial = InitialOutputState::kContent,
        std::optional<sampling::JsonConstraint::ToolFormat> tool_format = {})
        : state_(std::move(model_state)),
          request_(std::move(scheduled_request)),
          tool_format_(tool_format) {
      if (initial == InitialOutputState::kReasoning)
        reasoning_end_ =
            state_->scheduler->runner().TokenizeControl("</think>");
    }

    Result Wait(const TokenCallback& on_token,
                const ProgressCallback& on_progress,
                const StartCallback& on_start) override {
      auto result = request_.Wait(on_token, on_progress, on_start);
      if (!reasoning_end_.empty()) {
        const auto end =
            std::search(result.tokens.begin(), result.tokens.end(),
                        reasoning_end_.begin(), reasoning_end_.end());
        result.reasoning_tokens =
            static_cast<std::size_t>(end - result.tokens.begin());
      }
      return result;
    }

    void Cancel() noexcept override { request_.Cancel(); }

    std::optional<sampling::JsonConstraint::ToolFormat> ToolFormat()
        const override {
      return tool_format_;
    }

  private:
    std::shared_ptr<const State> state_;
    TextGenerationScheduler::Request request_;
    std::vector<tokenization::TokenId> reasoning_end_;
    const std::optional<sampling::JsonConstraint::ToolFormat> tool_format_;
  };

  [[nodiscard]] std::shared_ptr<const State> Snapshot() const {
    const std::lock_guard<std::mutex> lock(state_mutex);
    return state;
  }

  Result GenerateScheduled(
      std::shared_ptr<const State> current,
      std::vector<TextRunnerToken> prompt_tokens,
      Clock::time_point request_start, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled, const TokenCallback& on_token,
      std::string client_id,
      std::shared_ptr<const TextPromptContext> context = {},
      bool cache_prompt = true, std::size_t cache_prefix_tokens = 0,
      const std::vector<std::string>& stop_sequences = {},
      InitialOutputState initial = InitialOutputState::kContent,
      std::size_t system_prefix_tokens = 0, bool cache_persist = true) const {
    Result result;
    result.prompt_tokens = prompt_tokens.size();
    result.client_id = client_id.empty() ? "anonymous" : client_id;
    if (current == nullptr || prompt_tokens.empty()) {
      return result;
    }
    if (is_cancelled && is_cancelled()) {
      result.cancelled = true;
      return result;
    }

    auto request = current->scheduler->Submit(
        std::move(prompt_tokens), max_tokens, current->Applied(sampling),
        is_cancelled, static_cast<bool>(on_token),
        TextRequestMetadata{
            .client_id = std::move(client_id),
            .deadline = std::nullopt,
            .request_start = request_start,
            .prompt_context = std::move(context),
            .cache_prompt = cache_prompt,
            .cache_persist = cache_persist,
            .cache_prefix_tokens = cache_prefix_tokens,
            .system_prefix_tokens = system_prefix_tokens,
            .stop_sequences = stop_sequences,
        });
    ScheduledGenerationRequest generation(std::move(current),
                                          std::move(request), initial);
    return generation.Wait(on_token, {}, {});
  }

  /// A `score_assistant` request: prefill the prompt, then score the final
  /// assistant content (TextRequestMetadata::score_tokens).
  std::shared_ptr<GenerationRequest> StartScoring(
      std::shared_ptr<const State> current, const ChatRequest& request,
      Clock::time_point request_start, const CancellationCheck& is_cancelled,
      bool stream_output) const {
    if (stream_output)
      throw std::invalid_argument("score_assistant does not support streaming");
    if (request.abliteration && current->abliteration.empty())
      throw std::invalid_argument(
          "abliteration profile is not available on this backend");
    const auto& runner = current->scheduler->runner();
    if (!runner.SupportsScoring())
      throw std::invalid_argument("model does not support score_assistant");
    auto scoring = runner.PrepareScoring(request);
    const std::size_t system_prefix_tokens =
        request.cache_prompt ? scoring.prompt.system_prefix_tokens : 0;
    const auto count = scoring.answer.size();
    auto scheduled = current->scheduler->Submit(
        std::move(scoring.prompt.tokens), count, sampling::SamplingConfig{},
        is_cancelled, false,
        TextRequestMetadata{
            .client_id =
                request.client_id.empty() ? "anonymous" : request.client_id,
            .deadline = std::nullopt,
            .request_start = request_start,
            .prompt_context = std::move(scoring.prompt.context),
            .cache_prompt = request.cache_prompt,
            .cache_persist = false,
            .cache_prefix_tokens = scoring.prompt.cache_prefix_tokens,
            .system_prefix_tokens = system_prefix_tokens,
            .score_tokens = std::move(scoring.answer),
        });
    return std::make_shared<ScheduledGenerationRequest>(std::move(current),
                                                        std::move(scheduled));
  }

  mutable std::mutex state_mutex;
  std::shared_ptr<const State> state;
#endif
};

InferenceBackend::InferenceBackend() : impl_(std::make_unique<Impl>()) {}

InferenceBackend::~InferenceBackend() = default;

bool InferenceBackend::load(
    const std::string& model_path, std::string* error,
    std::uint32_t max_context, std::size_t session_count,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy,
    const TextSpeculativeConfig& speculative_config,
    const TextDiskCacheConfig& disk_cache_config,
    const std::string& vision_model_path,
    TextRunnerRamCacheOptions ram_cache_config, bool enable_yarn,
    std::optional<cli::KvCacheType> kv_cache_type,
    const std::string& abliteration_profiles,
    std::optional<models::qwen38_flash_next::HcMode> hc_mode) {
#if defined(ENGINE_ENABLE_HIP)
  TextDiskCacheConfig resolved_disk_cache_config = disk_cache_config;
  std::string load_error;
  std::vector<std::string> profile_files;
  try {
    profile_files =
        models::qwen::ListAbliterationProfiles(abliteration_profiles);
  } catch (const std::exception& e) {
    SetError(error, e.what());
    return false;
  }
  auto reader_owner = cli::OpenModelArtifact(model_path, &load_error);
  if (reader_owner == nullptr) {
    SetError(error, "Failed to open model: " + load_error);
    return false;
  }
  const std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
  const auto architecture =
      reader->GetMetadataString("general.architecture").value_or("");
  if (!profile_files.empty() && architecture != "qwen4exp" &&
      architecture != "qwen35") {
    SetError(error,
             "--abliteration-profiles applies only to Qwen3.8-27B and "
             "Flash-Next");
    return false;
  }
  // Qwen3.8-27B holds one request-selectable profile.
  if (profile_files.size() > 1 && architecture == "qwen35") {
    SetError(error,
             "Qwen3.8-27B loads one profile; --abliteration-profiles holds " +
                 std::to_string(profile_files.size()));
    return false;
  }
  if (hc_mode && architecture != "qwen4exp") {
    SetError(error, "--hc-mode applies only to Qwen3.8-Flash-Next");
    return false;
  }
  if (enable_yarn && architecture != "qwen4exp") {
    SetError(error, "--enable-yarn applies only to Qwen3.8-Flash-Next");
    return false;
  }
  // Every architecture but DeepSeek serves through the component cache.
  // Checked before any weights load.
  if (disk_cache_config.staging_capacity_bytes != 0 &&
      architecture != "deepseek4") {
    SetError(error, std::string(kComponentCacheRejectsDiskStaging));
    return false;
  }
  std::optional<cli::KvCacheType> resolved_kv_cache_type;
  if (!cli::ResolveKvCacheType(architecture, kv_cache_type,
                               &resolved_kv_cache_type, &load_error)) {
    SetError(error, load_error);
    return false;
  }
  // Read once, reused below to derive YaRN from --context for qwen4exp.
  const std::uint32_t qwen4exp_native_context =
      architecture == "qwen4exp"
          ? reader->GetMetadataUint32("qwen4exp.context_length").value_or(0)
          : 0;
  if (max_context == 0) {
    const std::uint64_t native =
        architecture == "qwen4exp"
            ? qwen4exp_native_context
            : reader
                  ->GetMetadataUint64(std::string(architecture) +
                                      ".context_length")
                  .value_or(0);
    if (native < 2 || native > std::numeric_limits<std::uint32_t>::max()) {
      SetError(error,
               "GGUF has no valid native context length; specify --context");
      return false;
    }
    max_context = static_cast<std::uint32_t>(native);
  }
  // Static YaRN changes every request, so a context above Flash-Next's
  // native one needs --enable-yarn. Checked before any weights load.
  std::optional<models::qwen38_flash_next::RopeScaling> rope_scaling;
  if (architecture == "qwen4exp" &&
      !models::qwen38_flash_next::ResolveRopeScaling(
          max_context, qwen4exp_native_context, enable_yarn, &rope_scaling,
          error)) {
    return false;
  }

  if (reader->GetMetadataString("general.architecture") == "deepseek4") {
    if (!vision_model_path.empty()) {
      SetError(error, "DeepSeek does not support --mmproj");
      return false;
    }
    if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
        speculative_config.backend != TextSpeculativeBackend::kDSpark) {
      SetError(error,
               "DeepSeek HTTP models support only DSpark speculative decoding");
      return false;
    }
    if (speculative_config.backend == TextSpeculativeBackend::kDSpark &&
        speculative_config.draft_model_path.empty()) {
      SetError(error, "DeepSeek DSpark HTTP decoding requires --dspark-model");
      return false;
    }
    if (!models::deepseek_v4_flash::ValidateGgufTemplate(*reader,
                                                         &load_error)) {
      SetError(error, "Unsupported DeepSeek chat template: " + load_error);
      return false;
    }
    auto model = models::deepseek_v4_flash::Model::Load(
        model_path,
        models::deepseek_v4_flash::ModelOptions{
            .max_context = max_context,
            .dspark_model_path =
                speculative_config.backend == TextSpeculativeBackend::kDSpark
                    ? speculative_config.draft_model_path
                    : std::string{},
        },
        &load_error);
    if (model == nullptr) {
      SetError(error, "Failed to create DeepSeek model: " + load_error);
      return false;
    }
    if (DiskCacheEnabled(resolved_disk_cache_config) &&
        resolved_disk_cache_config.model_artifact_fingerprint.empty() &&
        !FingerprintArtifact(
            "DeepSeek", *reader,
            &resolved_disk_cache_config.model_artifact_fingerprint, error)) {
      return false;
    }
    if (DiskCacheEnabled(resolved_disk_cache_config) &&
        speculative_config.backend == TextSpeculativeBackend::kDSpark &&
        resolved_disk_cache_config.draft_model_artifact_fingerprint.empty() &&
        !FingerprintArtifactFile(
            "DSpark", speculative_config.draft_model_path,
            &resolved_disk_cache_config.draft_model_artifact_fingerprint,
            error)) {
      return false;
    }
    return load(std::move(model), error, max_context, session_count,
                prefill_policy, scheduler_policy, speculative_config,
                std::move(resolved_disk_cache_config), ram_cache_config);
  }
  if (reader->GetMetadataString("general.architecture") == "qwen4exp") {
    if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
        speculative_config.backend != TextSpeculativeBackend::kMtp) {
      SetError(error,
               "Qwen3.8-Flash-Next HTTP models support only MTP speculative "
               "decoding (--speculative mtp --mtp-model)");
      return false;
    }
    if (speculative_config.backend == TextSpeculativeBackend::kMtp &&
        speculative_config.draft_model_path.empty()) {
      SetError(error,
               "Qwen3.8-Flash-Next MTP HTTP decoding requires --mtp-model");
      return false;
    }
    if (speculative_config.backend == TextSpeculativeBackend::kMtp &&
        (speculative_config.max_draft_tokens == 0 ||
         speculative_config.min_draft_tokens != 1)) {
      SetError(error,
               "Flash-Next MTP requires a positive draft limit and "
               "--min-draft-tokens 1");
      return false;
    }
    // The compiled Qwen3.8 chat template renders through the artifact's
    // own tokenizer (the same vocabulary and pre-tokenizer as Qwen3.8).
    if (!tokenization::QwenChatTemplate::ValidateGgufTemplate(*reader,
                                                              &load_error)) {
      SetError(error,
               "Unsupported Qwen3.8-Flash-Next chat template: " + load_error);
      return false;
    }
    // The model owns prefill geometry for both bulk and scheduled requests.
    auto model = models::qwen38_flash_next::Model::Load(
        model_path,
        models::qwen38_flash_next::ModelOptions{
            .max_context = max_context,
            .mtp_model_path =
                speculative_config.backend == TextSpeculativeBackend::kMtp
                    ? speculative_config.draft_model_path
                    : std::string{},
            .max_draft_tokens = speculative_config.max_draft_tokens,
            .vision_model_path = vision_model_path,
            .decode_concurrency = static_cast<std::uint32_t>(
                std::clamp<std::size_t>(session_count, 1, 8)),
            .rope_scaling = rope_scaling,
            .kvarn = resolved_kv_cache_type == cli::KvCacheType::kKvarnK4V4 ||
                     resolved_kv_cache_type == cli::KvCacheType::kKvarnK6V6,
            .kvarn_bits =
                resolved_kv_cache_type == cli::KvCacheType::kKvarnK6V6 ? 6 : 4,
            .abliteration_profiles = profile_files,
            .hc_mode =
                hc_mode.value_or(models::qwen38_flash_next::kDefaultHcMode),
        },
        &load_error);
    if (model == nullptr) {
      SetError(error,
               "Failed to create Qwen3.8-Flash-Next model: " + load_error);
      return false;
    }
    // Wide copy verification (Executor::ForwardWide) reserves its lazy
    // segment scratch beside the sessions (ResourceClaim). Narrow the width
    // until the configured sessions fit, rather than refusing a session
    // count the ordinary verifier width fits.
    if (std::size_t free_bytes = 0, total_bytes = 0;
        speculative_config.backend == TextSpeculativeBackend::kMtp &&
        hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      const std::size_t sessions = std::max<std::size_t>(session_count, 1);
      for (std::uint32_t rows = models::qwen38_flash_next::Model::kWideCopyRows;
           ; rows -= 8) {
        model->LimitCopyRows(rows);
        if (rows == 8 ||
            sessions * model->SessionBytes(core::SessionMode::kSpeculative,
                                           max_context) +
                    model->DeferredScratchBytes() <=
                free_bytes)
          break;
      }
    }
    // Allocate the decode batch and wide verification scratch before any
    // session, so no decode pass allocates device memory. What it allocates
    // leaves DeferredScratchBytes, so the budgets keep their totals. On
    // failure the passes still allocate on first use.
    {
      const auto deferred = model->DeferredScratchBytes();
      std::string reserve_error;
      if (model->ReserveDecodeScratch(&reserve_error))
        Logger::Info(
            "loader",
            "event=decode_scratch_reserved bytes=" +
                std::to_string(deferred - model->DeferredScratchBytes()) + " " +
                Logger::MemoryStatus());
      else
        Logger::Warn("loader",
                     "event=decode_scratch_reserved status=deferred reason=" +
                         reserve_error);
    }
    Logger::Info("loader",
                 "event=hc_mode mode=" +
                     std::string(models::qwen38_flash_next::HcModeName(
                         model->hc_mode())));
    if (rope_scaling) {
      Logger::Info(
          "loader",
          "event=yarn factor=" + std::to_string(rope_scaling->factor) +
              " original_context=" + std::to_string(qwen4exp_native_context) +
              " context=" + std::to_string(max_context));
    }
    // RAM checkpoints cannot outlive this loaded model. Their process-local
    // identity is sufficient; hashing every shard here adds a complete model
    // read on each fresh container start even when disk caching is disabled.
    if (DiskCacheEnabled(resolved_disk_cache_config) &&
        resolved_disk_cache_config.model_artifact_fingerprint.empty() &&
        !FingerprintArtifact(
            "Qwen3.8-Flash-Next", *reader,
            &resolved_disk_cache_config.model_artifact_fingerprint, error)) {
      return false;
    }
    if (DiskCacheEnabled(resolved_disk_cache_config) &&
        speculative_config.backend == TextSpeculativeBackend::kMtp &&
        resolved_disk_cache_config.draft_model_artifact_fingerprint.empty() &&
        !FingerprintArtifactFile(
            "Qwen3.8-Flash-Next MTP", speculative_config.draft_model_path,
            &resolved_disk_cache_config.draft_model_artifact_fingerprint,
            error)) {
      return false;
    }
    return load(std::move(model), error, max_context, session_count,
                prefill_policy, scheduler_policy, speculative_config,
                std::move(resolved_disk_cache_config), ram_cache_config);
  }
  // Refuse a storage the server's prefix cache cannot hold before mapping
  // 27B weights.
  const auto execution_policy =
      cli::QwenExecutionPolicyFor(resolved_kv_cache_type);
  if (const auto unsupported =
          hip::QwenContinuationUnsupportedReason(execution_policy);
      !unsupported.empty()) {
    SetError(error, std::string(unsupported));
    return false;
  }
  std::shared_ptr<models::qwen::vision::Encoder> vision;
  try {
    if (reader->GetMetadataUint64("qwen35.embedding_length") == 5120) {
      vision = models::qwen::vision::Encoder::Open(model_path,
                                                   vision_model_path, 5120);
    } else if (!vision_model_path.empty()) {
      throw std::invalid_argument(
          "image input supports Qwen3.8-27B and Flash-Next");
    }
  } catch (const std::exception& e) {
    SetError(error, e.what());
    return false;
  }
  Logger::Info("loader", "event=load_phase phase=target_weights " +
                             Logger::MemoryStatus());
  auto model = hip::QwenGpuModel::CreateFromGguf(
      reader, &load_error, std::move(vision),
      profile_files.empty() ? std::string{} : profile_files.front());
  if (model == nullptr) {
    SetError(error, "Failed to create GPU model: " + load_error);
    return false;
  }
  if (DiskCacheEnabled(resolved_disk_cache_config) &&
      resolved_disk_cache_config.model_artifact_fingerprint.empty() &&
      !FingerprintArtifact(
          "Qwen", *reader,
          &resolved_disk_cache_config.model_artifact_fingerprint, error)) {
    return false;
  }
  return load(std::move(model), error, max_context, session_count,
              prefill_policy, scheduler_policy, speculative_config,
              std::move(resolved_disk_cache_config), ram_cache_config,
              execution_policy);
#else
  (void)model_path;
  (void)max_context;
  (void)session_count;
  (void)prefill_policy;
  (void)scheduler_policy;
  (void)speculative_config;
  (void)disk_cache_config;
  (void)vision_model_path;
  (void)ram_cache_config;
  (void)enable_yarn;
  (void)abliteration_profiles;
  (void)kv_cache_type;
  (void)hc_mode;
  SetError(error, "HTTP inference requires the HIP backend");
  return false;
#endif
}

#if defined(ENGINE_ENABLE_HIP)
bool InferenceBackend::load(std::shared_ptr<const hip::QwenGpuModel> model,
                            std::string* error, std::uint32_t max_context,
                            std::size_t session_count,
                            TextPrefillPolicy prefill_policy,
                            TextSchedulerPolicy scheduler_policy,
                            TextSpeculativeConfig speculative_config,
                            TextDiskCacheConfig disk_cache_config,
                            TextRunnerRamCacheOptions ram_cache_config,
                            hip::QwenExecutionPolicy execution_policy) {
  if (model == nullptr) {
    SetError(error, "Qwen GPU model must not be null");
    return false;
  }
  if (const auto unsupported =
          hip::QwenContinuationUnsupportedReason(execution_policy);
      !unsupported.empty()) {
    SetError(error, std::string(unsupported));
    return false;
  }
  if (max_context == 0)
    max_context = model->GetConfig().context_length;
  if (max_context < 2 || max_context > model->GetConfig().context_length) {
    SetError(error, "HTTP context exceeds the loaded Qwen model context");
    return false;
  }

  if (speculative_config.backend == TextSpeculativeBackend::kDSpark) {
    SetError(error, "DSpark HTTP decoding requires a DeepSeek model");
    return false;
  }
  if (speculative_config.backend == TextSpeculativeBackend::kMtp) {
    SetError(error, "MTP HTTP decoding requires a Qwen Flash-Next model");
    return false;
  }
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (DiskCacheEnabled(disk_cache_config) &&
      (!IsSha256Hex(disk_cache_config.model_artifact_fingerprint) ||
       (!disk_cache_config.draft_model_artifact_fingerprint.empty() &&
        !IsSha256Hex(disk_cache_config.draft_model_artifact_fingerprint)) ||
       disk_cache_config.capacity_bytes == 0)) {
    SetError(error, "Qwen persistent disk cache configuration is invalid");
    return false;
  }
  if (speculative_config.max_draft_tokens == 0 ||
      speculative_config.min_draft_tokens == 0 ||
      speculative_config.min_draft_tokens >
          speculative_config.max_draft_tokens) {
    SetError(error, "HTTP speculative draft limits are invalid");
    return false;
  }
  if (speculative_config.backend == TextSpeculativeBackend::kDFlash &&
      speculative_config.min_draft_tokens != 1) {
    SetError(error,
             "DFlash2 requires --min-draft-tokens 1; bound blocks with "
             "--draft-tokens");
    return false;
  }

  try {
    std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model;
    speculative::SpeculativeOptions speculative_options;
    if (speculative_config.backend == TextSpeculativeBackend::kDFlash) {
      if (speculative_config.draft_model_path.empty()) {
        SetError(error, "DFlash HTTP decoding requires --dflash-model");
        return false;
      }
      std::string dflash_error;
      auto dflash_reader_owner = core::GgufReader::OpenFile(
          speculative_config.draft_model_path, &dflash_error);
      if (dflash_reader_owner == nullptr) {
        SetError(error, "Failed to open DFlash GGUF: " + dflash_error);
        return false;
      }
      std::shared_ptr<const core::GgufReader> dflash_reader(
          std::move(dflash_reader_owner));
      if (DiskCacheEnabled(disk_cache_config) &&
          disk_cache_config.draft_model_artifact_fingerprint.empty() &&
          !FingerprintArtifact(
              "DFlash", *dflash_reader,
              &disk_cache_config.draft_model_artifact_fingerprint, error)) {
        return false;
      }
      if (DiskCacheEnabled(disk_cache_config) &&
          !IsSha256Hex(disk_cache_config.draft_model_artifact_fingerprint)) {
        SetError(error,
                 "Qwen DFlash persistent disk cache configuration is "
                 "invalid");
        return false;
      }
      Logger::Info("loader", "event=load_phase phase=draft_weights " +
                                 Logger::MemoryStatus());
      dflash_model = hip::QwenDFlashGpuModel::Create(std::move(dflash_reader),
                                                     model, &dflash_error);
      if (dflash_model == nullptr) {
        SetError(error, "Failed to create DFlash model: " + dflash_error);
        return false;
      }

      speculative_options.dflash_policy = speculative_config.dflash_policy;
      speculative_options.max_draft_tokens =
          speculative_config.max_draft_tokens;
      speculative_options.min_draft_tokens =
          speculative_config.min_draft_tokens;
      speculative_options.initial_draft_tokens =
          speculative_config.max_draft_tokens;
      speculative_options.use_batched_verification = true;
      speculative_options.retain_frontier_logits = true;
      speculative_options.enable_adaptive_draft_length = false;
    }

    PrepareVision(model->VisionEncoder());
    auto new_state = std::make_shared<Impl::State>();
    new_state->sampling_defaults.model =
        sampling::TextPreset(model->GetConfig());
    new_state->sampling_defaults.supplied = {};
    new_state->supports_images = model->VisionEncoder() != nullptr;
    new_state->abliteration =
        DescribeAbliteration(ListProfiles(model->Abliteration()));
    auto runner = std::make_shared<QwenTextRunner>(
        std::move(model), max_context, std::move(dflash_model),
        speculative_options, disk_cache_config.model_artifact_fingerprint,
        disk_cache_config.draft_model_artifact_fingerprint, session_count,
        ram_cache_config.kv_pool_bytes, execution_policy);
    new_state->han_tokens = PrepareHanTokens(*runner);
    new_state->model_id = runner->Descriptor().model_id;
    new_state->max_context = max_context;
    std::optional<TextRunnerDiskCacheOptions> runner_disk_cache;
    if (DiskCacheEnabled(disk_cache_config)) {
      runner_disk_cache = TextRunnerDiskCacheOptions{
          .directory = std::move(disk_cache_config.directory),
          .capacity_bytes = disk_cache_config.capacity_bytes,
          .staging_capacity_bytes = disk_cache_config.staging_capacity_bytes,
      };
    }
    Logger::Info("loader",
                 "event=load_phase phase=sessions " + Logger::MemoryStatus());
    auto runner_pool = std::make_shared<TextRunnerPool>(
        std::move(runner), session_count, std::move(runner_disk_cache),
        ram_cache_config);
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool InferenceBackend::load(
    std::shared_ptr<models::deepseek_v4_flash::Model> model, std::string* error,
    std::uint32_t max_context, std::size_t session_count,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy,
    TextSpeculativeConfig speculative_config,
    TextDiskCacheConfig disk_cache_config,
    TextRunnerRamCacheOptions ram_cache_config) {
  if (ram_cache_config.kv_pool_bytes != 0) {
    SetError(error,
             "--kv-pool-bytes needs a runner with paged KV rows; "
             "this model keeps each session's context");
    return false;
  }
  if (model == nullptr) {
    SetError(error, "DeepSeek model must not be null");
    return false;
  }
  if (max_context == 0)
    max_context = model->MaxContext();

  const bool use_dspark =
      speculative_config.backend == TextSpeculativeBackend::kDSpark;
  if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
      (!use_dspark || !model->HasDspark())) {
    SetError(error, "DeepSeek DSpark requires a loaded support model");
    return false;
  }
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (max_context > model->MaxContext()) {
    SetError(error, "HTTP context exceeds the loaded DeepSeek model context");
    return false;
  }
  if (use_dspark && (speculative_config.max_draft_tokens == 0 ||
                     speculative_config.min_draft_tokens == 0 ||
                     speculative_config.min_draft_tokens >
                         speculative_config.max_draft_tokens)) {
    SetError(error, "DeepSeek DSpark draft limits are invalid");
    return false;
  }
  if (use_dspark && speculative_config.min_draft_tokens != 1) {
    SetError(error,
             "DSpark uses model-owned adaptive drafting; custom draft "
             "floors are unsupported");
    return false;
  }
  if (DiskCacheEnabled(disk_cache_config) &&
      (!IsSha256Hex(disk_cache_config.model_artifact_fingerprint) ||
       (use_dspark &&
        !IsSha256Hex(disk_cache_config.draft_model_artifact_fingerprint)) ||
       disk_cache_config.capacity_bytes == 0)) {
    SetError(error, "DeepSeek persistent disk cache configuration is invalid");
    return false;
  }

  try {
    auto new_state = std::make_shared<Impl::State>();
    new_state->sampling_defaults.model =
        sampling::TextModelPreset::kDeepSeekV4Flash;
    new_state->sampling_defaults.supplied = {};
    auto runner = std::make_shared<DeepSeekTextRunner>(
        std::move(model), max_context, use_dspark,
        speculative_config.max_draft_tokens,
        disk_cache_config.model_artifact_fingerprint,
        disk_cache_config.draft_model_artifact_fingerprint);
    new_state->model_id = runner->Descriptor().model_id;
    new_state->max_context = max_context;
    std::optional<TextRunnerDiskCacheOptions> runner_disk_cache;
    if (DiskCacheEnabled(disk_cache_config)) {
      runner_disk_cache = TextRunnerDiskCacheOptions{
          .directory = std::move(disk_cache_config.directory),
          .capacity_bytes = disk_cache_config.capacity_bytes,
          .staging_capacity_bytes = disk_cache_config.staging_capacity_bytes,
      };
    }
    auto runner_pool = std::make_shared<TextRunnerPool>(
        std::move(runner), session_count, std::move(runner_disk_cache),
        ram_cache_config);
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool InferenceBackend::load(
    std::shared_ptr<models::qwen38_flash_next::Model> model, std::string* error,
    std::uint32_t max_context, std::size_t session_count,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy,
    TextSpeculativeConfig speculative_config,
    TextDiskCacheConfig disk_cache_config,
    TextRunnerRamCacheOptions ram_cache_config) {
  if (model == nullptr) {
    SetError(error, "Qwen3.8-Flash-Next model must not be null");
    return false;
  }
  if (max_context == 0)
    max_context = model->MaxContext();

  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (max_context == 0 || max_context > model->MaxContext()) {
    SetError(
        error,
        "HTTP context exceeds the loaded Qwen3.8-Flash-Next model context");
    return false;
  }
  if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
      (speculative_config.backend != TextSpeculativeBackend::kMtp ||
       !model->HasMtp())) {
    SetError(error,
             "Flash-Next MTP requires a model loaded with its draft sidecar");
    return false;
  }
  if (speculative_config.backend == TextSpeculativeBackend::kMtp &&
      (speculative_config.max_draft_tokens == 0 ||
       speculative_config.min_draft_tokens != 1)) {
    SetError(error,
             "Qwen3.8-Flash-Next MTP drafts a fixed chain; custom draft "
             "floors are unsupported");
    return false;
  }
  if (DiskCacheEnabled(disk_cache_config) &&
      (!IsSha256Hex(disk_cache_config.model_artifact_fingerprint) ||
       (speculative_config.backend == TextSpeculativeBackend::kMtp &&
        !IsSha256Hex(disk_cache_config.draft_model_artifact_fingerprint)) ||
       disk_cache_config.capacity_bytes == 0)) {
    SetError(error,
             "Qwen3.8-Flash-Next persistent disk cache configuration is "
             "invalid");
    return false;
  }
  try {
    PrepareVision(model->VisionEncoder());
    auto new_state = std::make_shared<Impl::State>();
    new_state->sampling_defaults.model = sampling::TextModelPreset::kQwen38;
    new_state->sampling_defaults.supplied = {};
    new_state->supports_images = model->VisionEncoder() != nullptr;
    new_state->abliteration =
        DescribeAbliteration(ListProfiles(model->AbliterationProfiles()));
    new_state->hc_mode =
        models::qwen38_flash_next::HcModeName(model->hc_mode());
    auto runner = std::make_shared<QwenFlashNextTextRunner>(
        std::move(model), max_context,
        speculative_config.backend == TextSpeculativeBackend::kMtp,
        speculative_config.max_draft_tokens,
        disk_cache_config.model_artifact_fingerprint,
        disk_cache_config.draft_model_artifact_fingerprint,
        ram_cache_config.kv_pool_bytes);
    new_state->han_tokens = PrepareHanTokens(*runner);
    new_state->model_id = runner->Descriptor().model_id;
    new_state->max_context = max_context;
    std::optional<TextRunnerDiskCacheOptions> runner_disk_cache;
    if (DiskCacheEnabled(disk_cache_config)) {
      runner_disk_cache = TextRunnerDiskCacheOptions{
          .directory = std::move(disk_cache_config.directory),
          .capacity_bytes = disk_cache_config.capacity_bytes,
          .staging_capacity_bytes = disk_cache_config.staging_capacity_bytes,
      };
    }
    auto runner_pool = std::make_shared<TextRunnerPool>(
        std::move(runner), session_count, std::move(runner_disk_cache),
        ram_cache_config);
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}
#endif

std::string InferenceBackend::model_id() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->model_id : "unknown";
#else
  return "unknown";
#endif
}

std::vector<std::string> InferenceBackend::model_ids() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr)
    return {"unknown"};
  std::vector<std::string> ids{state->model_id};
  ids.insert(ids.end(), state->extra_model_ids.begin(),
             state->extra_model_ids.end());
  return ids;
#else
  return {"unknown"};
#endif
}

bool InferenceBackend::ready() const {
#if defined(ENGINE_ENABLE_HIP)
  return impl_->Snapshot() != nullptr;
#else
  return false;
#endif
}

bool InferenceBackend::device_lost() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr && state->scheduler->device_lost();
#else
  return false;
#endif
}

std::vector<AbliterationInfo> InferenceBackend::abliteration_profiles() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state ? state->abliteration : std::vector<AbliterationInfo>{};
#else
  return {};
#endif
}

bool InferenceBackend::supports_images() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr && state->supports_images;
#else
  return false;
#endif
}

std::string InferenceBackend::hc_mode() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->hc_mode : std::string{};
#else
  return {};
#endif
}

bool InferenceBackend::supports_han_scale() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr && state->han_tokens != nullptr;
#else
  return false;
#endif
}

bool InferenceBackend::supports_scoring() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr && state->scheduler->runner().SupportsScoring();
#else
  return false;
#endif
}

std::uint32_t InferenceBackend::max_context() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->max_context : 0;
#else
  return 0;
#endif
}

std::vector<InferenceBackend::SessionState> InferenceBackend::session_states()
    const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->scheduler->SessionStates()
                          : std::vector<SessionState>{};
#else
  return {};
#endif
}

InferenceBackend::SamplingDefaults InferenceBackend::sampling_defaults() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->sampling_defaults : SamplingDefaults{};
#else
  return {};
#endif
}

ReasoningOptions InferenceBackend::reasoning_defaults() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->reasoning_defaults : ReasoningOptions{};
#else
  return {};
#endif
}

InferenceBackend::InitialOutputState InferenceBackend::initial_output_state(
    const ChatRequest& request) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr
             ? state->scheduler->runner().InitialOutputState(request)
             : InitialOutputState::kAuto;
#else
  (void)request;
  return InitialOutputState::kAuto;
#endif
}

std::string InferenceBackend::model_path() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->model_path : std::string{};
#else
  return {};
#endif
}

std::string InferenceBackend::build_info() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->build_info : std::string{};
#else
  return {};
#endif
}

void InferenceBackend::set_served_artifact(const std::string& model_path,
                                           const std::string& build_info) {
#if defined(ENGINE_ENABLE_HIP)
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->model_path = model_path;
  updated->build_info = build_info;
  impl_->state = std::move(updated);
#else
  (void)model_path;
  (void)build_info;
#endif
}

void InferenceBackend::set_model_ids(std::vector<std::string> model_ids) {
#if defined(ENGINE_ENABLE_HIP)
  if (model_ids.empty()) {
    return;
  }
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->model_id = std::move(model_ids.front());
  updated->extra_model_ids.assign(
      std::make_move_iterator(model_ids.begin() + 1),
      std::make_move_iterator(model_ids.end()));
  impl_->state = std::move(updated);
#else
  (void)model_ids;
#endif
}

void InferenceBackend::set_sampling_defaults(
    std::size_t max_tokens, const sampling::SamplingConfig& sampling_config,
    sampling::SamplingOverrides supplied) {
#if defined(ENGINE_ENABLE_HIP)
  sampling_config.Validate();
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->sampling_defaults.max_tokens = max_tokens;
  updated->sampling_defaults.sampling = sampling_config;
  updated->sampling_defaults.supplied = supplied;
  impl_->state = std::move(updated);
#else
  (void)max_tokens;
  (void)sampling_config;
  (void)supplied;
#endif
}

void InferenceBackend::set_reasoning_defaults(
    const ReasoningOptions& reasoning) {
#if defined(ENGINE_ENABLE_HIP)
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->reasoning_defaults = reasoning;
  impl_->state = std::move(updated);
#else
  (void)reasoning;
#endif
}

InferenceBackend::Result InferenceBackend::complete(
    std::string_view prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token,
    std::string_view client_id, const std::vector<std::string>& stop_sequences,
    bool cache_prompt, bool cache_persist) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return {};
  }
  auto prompt_tokens = state->scheduler->runner().Tokenize(prompt);
  return impl_->GenerateScheduled(
      state, std::move(prompt_tokens), request_start, max_tokens,
      sampling_config, is_cancelled, on_token, std::string(client_id), {},
      cache_prompt, 0, stop_sequences, InitialOutputState::kContent,
      /*system_prefix_tokens=*/0, cache_persist);
#else
  (void)cache_prompt;
  (void)cache_persist;
  (void)client_id;
  (void)stop_sequences;
  (void)prompt;
  (void)max_tokens;
  (void)sampling_config;
  (void)is_cancelled;
  (void)on_token;
  return {};
#endif
}

InferenceBackend::Result InferenceBackend::chat(
    const ChatRequest& request, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return {};
  }
  if (request.score_assistant)
    return impl_
        ->StartScoring(state, request, request_start, is_cancelled, false)
        ->Wait(on_token, {}, {});
  auto effective_sampling = sampling_config;
  auto constrained = ConstrainChatRequest(request, state->scheduler->runner(),
                                          &effective_sampling);
  if (request.abliteration && state->abliteration.empty())
    throw std::invalid_argument(
        "abliteration profile is not available on this backend");
  const auto& effective_request = constrained ? *constrained : request;
  auto prompt = state->scheduler->runner().PreparePrompt(effective_request);
  if (!prompt.has_value() || prompt->tokens.empty()) {
    return {};
  }
  const auto initial =
      state->scheduler->runner().InitialOutputState(effective_request);
  if (initial == TextGenerationBackend::InitialOutputState::kReasoning)
    effective_sampling.reasoning_budget =
        RequestReasoning(request, effective_sampling, state->han_tokens,
                         state->scheduler->runner(), prompt->tokens);
  const std::size_t system_prefix_tokens =
      request.cache_prompt ? prompt->system_prefix_tokens : 0;
  return impl_->GenerateScheduled(
      state, std::move(prompt->tokens), request_start, max_tokens,
      effective_sampling, is_cancelled, on_token, request.client_id,
      std::move(prompt->context), request.cache_prompt,
      prompt->cache_prefix_tokens, request.stop_sequences, initial,
      system_prefix_tokens, request.cache_persist);
#else
  (void)request;
  (void)max_tokens;
  (void)sampling_config;
  (void)is_cancelled;
  (void)on_token;
  return {};
#endif
}

std::shared_ptr<InferenceBackend::GenerationRequest>
InferenceBackend::start_complete(
    std::string_view prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled, bool stream_output, bool ignore_eos,
    std::string_view client_id, const std::vector<std::string>& stop_sequences,
    bool return_progress, bool cache_prompt, bool cache_persist) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return TextGenerationBackend::start_complete(
        prompt, max_tokens, sampling_config, is_cancelled, stream_output,
        ignore_eos, client_id, stop_sequences, return_progress, cache_prompt,
        cache_persist);
  }
  auto prompt_tokens = state->scheduler->runner().Tokenize(prompt);
  auto scheduled_request = state->scheduler->Submit(
      std::move(prompt_tokens), max_tokens, state->Applied(sampling_config),
      is_cancelled, stream_output,
      TextRequestMetadata{
          .client_id = client_id.empty() ? "anonymous" : std::string(client_id),
          .deadline = std::nullopt,
          .request_start = request_start,
          .prompt_context = {},
          .cache_prompt = cache_prompt,
          .cache_persist = cache_persist,
          .cache_prefix_tokens = 0,
          .stop_at_eos = !ignore_eos,
          .stop_sequences = stop_sequences,
          .return_progress = return_progress,
      });
  return std::make_shared<Impl::ScheduledGenerationRequest>(
      state, std::move(scheduled_request));
#else
  return TextGenerationBackend::start_complete(
      prompt, max_tokens, sampling_config, is_cancelled, stream_output,
      ignore_eos, client_id, stop_sequences, return_progress, cache_prompt,
      cache_persist);
#endif
}

std::shared_ptr<InferenceBackend::GenerationRequest>
InferenceBackend::start_chat(const ChatRequest& request, std::size_t max_tokens,
                             const sampling::SamplingConfig& sampling_config,
                             const CancellationCheck& is_cancelled,
                             bool stream_output) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return TextGenerationBackend::start_chat(
        request, max_tokens, sampling_config, is_cancelled, stream_output);
  }

  if (request.score_assistant)
    return impl_->StartScoring(state, request, request_start, is_cancelled,
                               stream_output);

  auto effective_sampling = sampling_config;
  std::optional<sampling::JsonConstraint::ToolFormat> tool_format;
  auto constrained = ConstrainChatRequest(request, state->scheduler->runner(),
                                          &effective_sampling, &tool_format);
  if (request.abliteration && state->abliteration.empty())
    throw std::invalid_argument(
        "abliteration profile is not available on this backend");
  const auto& effective_request = constrained ? *constrained : request;
  auto prompt = state->scheduler->runner().PreparePrompt(effective_request);
  if (!prompt.has_value() || prompt->tokens.empty()) {
    return TextGenerationBackend::start_chat(
        request, max_tokens, sampling_config, is_cancelled, stream_output);
  }

  const auto initial =
      state->scheduler->runner().InitialOutputState(effective_request);
  if (initial == TextGenerationBackend::InitialOutputState::kReasoning)
    effective_sampling.reasoning_budget =
        RequestReasoning(request, effective_sampling, state->han_tokens,
                         state->scheduler->runner(), prompt->tokens);
  const std::string client_id =
      request.client_id.empty() ? "anonymous" : request.client_id;
  // Before Submit moves the tokens: argument evaluation order is unspecified.
  const std::size_t system_prefix_tokens =
      request.cache_prompt ? prompt->system_prefix_tokens : 0;
  auto scheduled_request = state->scheduler->Submit(
      std::move(prompt->tokens), max_tokens, state->Applied(effective_sampling),
      is_cancelled, stream_output,
      TextRequestMetadata{
          .client_id = client_id,
          .deadline = std::nullopt,
          .request_start = request_start,
          .prompt_context = std::move(prompt->context),
          .cache_prompt = request.cache_prompt,
          .cache_persist = request.cache_persist,
          .cache_prefix_tokens = prompt->cache_prefix_tokens,
          .system_prefix_tokens = system_prefix_tokens,
          .stop_sequences = request.stop_sequences,
          .return_progress = request.return_progress,
      });
  return std::make_shared<Impl::ScheduledGenerationRequest>(
      state, std::move(scheduled_request), initial, tool_format);
#else
  return TextGenerationBackend::start_chat(request, max_tokens, sampling_config,
                                           is_cancelled, stream_output);
#endif
}

InferenceBackend::Result InferenceBackend::chat(
    const std::vector<tokenization::ChatMessage>& messages,
    std::size_t max_tokens, const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled) {
  return chat(ChatRequest{messages}, max_tokens, sampling_config, is_cancelled);
}

std::size_t InferenceBackend::count_tokens(std::string_view text) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return 0;
  }
  return state->scheduler->runner().Tokenize(text).size();
#else
  (void)text;
  return 0;
#endif
}

std::optional<std::size_t> InferenceBackend::count_chat_tokens(
    const ChatRequest& request) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return std::nullopt;
  }
  // The same preparation as chat(): constrained tool or format requests add
  // prompt text before the template is rendered.
  auto sampling = sampling::SamplingConfig{};
  auto constrained =
      ConstrainChatRequest(request, state->scheduler->runner(), &sampling);
  if (request.abliteration && state->abliteration.empty())
    throw std::invalid_argument(
        "abliteration profile is not available on this backend");
  const auto& effective_request = constrained ? *constrained : request;
  return state->scheduler->runner().CountPromptTokens(effective_request);
#else
  (void)request;
  return std::nullopt;
#endif
}

std::optional<std::string> InferenceBackend::render_chat_prompt(
    const ChatRequest& request) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return std::nullopt;
  }
  // The same preparation as count_chat_tokens(), so the text is what chat()
  // would tokenize.
  auto sampling = sampling::SamplingConfig{};
  auto constrained =
      ConstrainChatRequest(request, state->scheduler->runner(), &sampling);
  if (request.abliteration && state->abliteration.empty())
    throw std::invalid_argument(
        "abliteration profile is not available on this backend");
  const auto& effective_request = constrained ? *constrained : request;
  return state->scheduler->runner().RenderPrompt(effective_request);
#else
  (void)request;
  return std::nullopt;
#endif
}

std::optional<std::vector<std::uint32_t>> InferenceBackend::tokenize(
    std::string_view text) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return std::nullopt;
  }
  return state->scheduler->runner().Tokenize(text);
#else
  (void)text;
  return std::nullopt;
#endif
}

}  // namespace gufo::server
