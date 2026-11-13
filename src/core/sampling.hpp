#ifndef GUFO_CORE_SAMPLING_HPP_
#define GUFO_CORE_SAMPLING_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/json_constraint.hpp"

namespace gufo::sampling {

using TokenId = std::uint32_t;

/// A set of vocabulary token IDs: bit (id % 32) of words[id / 32]. The
/// layout is the device layout, so GPU samplers upload `words` unchanged.
struct TokenSet {
  std::vector<std::uint32_t> words;
  std::size_t count{0};

  [[nodiscard]] bool Contains(TokenId token) const noexcept {
    const std::size_t word = token / 32U;
    return word < words.size() && ((words[word] >> (token % 32U)) & 1U) != 0;
  }
};

/// A reasoning token budget, enforced like llama.cpp's reasoning-budget
/// sampler (#266). Tokens are counted after the reasoning start, including
/// reasoning text already in the prompt. Once the budget is spent, and the
/// last token completed its UTF-8 character, `forced` is emitted token by
/// token. Any token in `ends` closes reasoning early, as does the request's
/// constraint grammar leaving its reasoning phase. Later tokens are answer
/// content, including literal reasoning tags, as in Gufo's response framing.
struct ReasoningBudget {
  /// A budget of this many tokens never runs out: it only tracks the
  /// reasoning phase (the Han scale's answer scope) and never forces.
  static constexpr std::size_t kUnlimited =
      std::numeric_limits<std::size_t>::max();
  std::size_t tokens{0};
  std::vector<TokenId> forced;
  std::vector<TokenId> ends;
  std::vector<TokenId> prompt;
  /// Indexed by token: its text ends inside a UTF-8 character.
  std::shared_ptr<const std::vector<bool>> partial_utf8;
};

/// Where the Han scale applies (serve --han-scale-scope, request
/// han_scale_scope): every sampled token, or only tokens after the reply's
/// reasoning phase ends.
enum class HanScaleScope : std::uint8_t { kAll, kAnswer };

/// "all" or "answer"; nullopt for any other spelling.
[[nodiscard]] std::optional<HanScaleScope> ParseHanScaleScope(
    std::string_view text) noexcept;
[[nodiscard]] std::string_view HanScaleScopeName(HanScaleScope scope) noexcept;

/// Additive logit offsets of a request's logit-bias sets (serve
/// --logit-bias-profiles, request logit_bias_profiles and logit_bias),
/// dense per reply phase and indexed by vocabulary token. A null phase adds
/// nothing there. Built once per selected combination
/// (src/core/logit_bias.hpp) and shared by every sampler of a request.
struct LogitBias {
  /// Offsets while the reply is in its reasoning phase (sets with scope
  /// all or reasoning).
  std::shared_ptr<const std::vector<float>> reasoning;
  /// Offsets after reasoning ends, and in replies without a reasoning phase
  /// (sets with scope all or answer).
  std::shared_ptr<const std::vector<float>> answer;
  /// The largest offset of each phase, or zero when none is positive: a
  /// raw block maximum plus this bounds every offset logit of the block.
  float reasoning_max{0.0F};
  float answer_max{0.0F};
  /// Canonical description of the combination (cache key, traces).
  std::string key;

  /// Whether the offsets differ between the phases, so samplers must track
  /// where reasoning ends (an unlimited ReasoningBudget).
  [[nodiscard]] bool phase_dependent() const noexcept {
    return reasoning != answer;
  }
};

/// Model-independent controls for selecting a token from a logit row.
///
/// Filters run in this order: penalties, Han scale, logit bias, temperature,
/// top-k, top-p, min-p. The defaults preserve Gufo's existing greedy
/// decoding behavior.
struct SamplingConfig {
  /// Observational raw target log probabilities; null disables collection.
  std::optional<std::uint32_t> logprobs;
  float temperature{0.0F};
  std::int32_t top_k{0};
  float top_p{1.0F};
  float min_p{0.0F};
  std::size_t min_keep{0};
  std::int64_t seed{-1};
  float repeat_penalty{1.0F};
  std::size_t repeat_last_n{64};
  float frequency_penalty{0.0F};
  float presence_penalty{0.0F};
  std::shared_ptr<const TokenConstraint> constraint;
  /// Set only for requests that start in their reasoning phase: a token
  /// budget, or an unlimited one (ReasoningBudget::kUnlimited) that only
  /// tracks the phase for han_scale_scope kAnswer.
  std::shared_ptr<const ReasoningBudget> reasoning_budget;
  /// Han-script probability scale in (0, 1] (serve --han-scale, request
  /// han_scale); 1 disables it. Every token in han_tokens has
  /// han_bias() = ln(han_scale) added to its logit after the penalties and
  /// before temperature and the filters: its temperature-1 probability is
  /// multiplied by han_scale (renormalized), its probability at
  /// temperature T by han_scale^(1/T). Greedy takes the argmax of the
  /// offset logits.
  float han_scale{1.0F};
  /// The model's Han tokens (src/core/han_script.hpp). A null set leaves
  /// han_scale inactive: models whose samplers do not apply it get none.
  std::shared_ptr<const TokenSet> han_tokens{};
  /// kAnswer applies no Han offset while the reply is in its reasoning
  /// phase (SamplerState::han_bias). That phase is tracked only when
  /// reasoning_budget is set; without it the whole reply is the answer.
  HanScaleScope han_scale_scope{HanScaleScope::kAll};
  /// The request's logit-bias offsets, added after the Han offset; null
  /// adds none. Only runners whose samplers apply the Han scale accept them.
  std::shared_ptr<const LogitBias> logit_bias{};

  void Validate() const;

  /// The logit offset of a Han token in the answer: ln(han_scale), or zero
  /// when the scale is inactive. Never positive. SamplerState::han_bias is
  /// the offset at the sampler's position, which han_scale_scope can zero.
  [[nodiscard]] double han_bias() const noexcept;
  /// Whether a logit-bias set is active in either phase.
  [[nodiscard]] bool logit_bias_active() const noexcept {
    return logit_bias && (logit_bias->reasoning || logit_bias->answer);
  }
  /// True when penalties, the Han scale or a logit bias adjust logits
  /// before filtering. Fast paths that take an unmodified argmax or skip
  /// the adjustment must treat all three alike.
  [[nodiscard]] bool penalties_enabled() const noexcept;
  [[nodiscard]] bool uses_random_sampling() const noexcept;
  [[nodiscard]] bool can_use_unmodified_argmax() const noexcept;
};

/// Unique token counts: repetition window and full generated response.
struct TokenPenalty {
  TokenId token{0};
  std::uint32_t generated_count{0};
  std::uint32_t repeated{0};
};

struct Probability {
  TokenId token{0};
  double value{0.0};
};

struct LogitCandidate {
  TokenId token;
  float value;
};

struct TokenLogprob {
  TokenId token{0};
  double logprob{0};
};

struct TokenLogprobs {
  TokenLogprob selected;
  std::vector<TokenLogprob> top;
};

/// Full-vocabulary log-softmax before any sampling transformations.
[[nodiscard]] std::optional<TokenLogprobs> CollectLogprobs(
    std::span<const float> logits, TokenId selected,
    std::optional<std::uint32_t> count);

/// A normalized, post-filter sampling distribution.
class SamplingDistribution {
public:
  SamplingDistribution() = default;
  explicit SamplingDistribution(std::vector<Probability> entries);

  [[nodiscard]] std::span<const Probability> entries() const noexcept;
  [[nodiscard]] TokenId best_token() const;
  [[nodiscard]] double probability(TokenId token) const noexcept;
  [[nodiscard]] TokenId Sample(std::uint64_t* rng_state) const;
  [[nodiscard]] TokenId SampleResidual(
      std::span<const TokenId> candidate_ids,
      std::span<const float> candidate_probabilities,
      std::uint64_t* rng_state) const;

private:
  friend class SamplerState;
  // Entries generated internally have unique IDs; retain their stable order.
  SamplingDistribution(std::vector<Probability> entries, double total);
  std::vector<Probability> entries_;
};

/// Builds the exact distribution from prompt and request-generated tokens.
[[nodiscard]] SamplingDistribution BuildDistribution(
    std::span<const float> logits, const SamplingConfig& config,
    std::span<const TokenId> prompt_tokens = {},
    std::span<const TokenId> generated_tokens = {});

/// Request-owned controls, repetition window, full generated counts and RNG.
///
/// Call Accept only for committed tokens. The type is copyable so speculative
/// decode can work on a tentative state and discard it on rollback.
class SamplerState {
public:
  explicit SamplerState(SamplingConfig config = {},
                        std::span<const TokenId> initial_history = {});
  ~SamplerState() = default;
  SamplerState(const SamplerState& other);
  SamplerState& operator=(const SamplerState& other);
  SamplerState(SamplerState&&) noexcept = default;
  SamplerState& operator=(SamplerState&&) noexcept = default;

  [[nodiscard]] const SamplingConfig& config() const noexcept;
  /// Draft q may remain unconstrained; target p always uses this request's
  /// grammar. While a scoped Han offset or logit bias (phase_scoped)
  /// depends on the reasoning phase, the copy keeps following the grammar's
  /// end of reasoning (a close spelled as text), so each tentative row's
  /// offset matches the target's; a token the grammar rejects ends that
  /// tracking in the copy. Once the copy's reasoning ends it drops the
  /// grammar, so ResetHistory on such a copy tracks only end tokens.
  [[nodiscard]] SamplerState WithoutConstraint() const;
  /// A finite argmax can bypass downloading logits if it satisfies the grammar.
  /// Set penalties_applied only for a winner computed with this state's exact
  /// penalty arithmetic and history; raw model argmaxes leave it false.
  [[nodiscard]] bool CanSelectArgmax(TokenId token,
                                     bool penalties_applied = false) const;
  /// A lazy tool grammar can admit the entire vocabulary before a call.
  /// Retain ordinary CPU/GPU sampling while its mask makes no restriction.
  /// Also true while ForcedToken() is set, so GPU samplers defer to the host.
  [[nodiscard]] bool NeedsConstraintMask() const;
  [[nodiscard]] std::span<const TokenId> history() const noexcept;
  /// The token every sampling path returns next, if any: an exhausted
  /// reasoning budget forces its end sequence.
  [[nodiscard]] std::optional<TokenId> ForcedToken() const noexcept;
  [[nodiscard]] std::span<const TokenPenalty> penalties() const noexcept {
    return penalty_counts_;
  }
  /// Whether the next token is still in the reply's reasoning phase. Always
  /// false without SamplingConfig::reasoning_budget.
  [[nodiscard]] bool InReasoning() const noexcept {
    return reasoning_phase_ != ReasoningPhase::kOff;
  }
  /// The Han offset of the next token: config().han_bias(), or zero while
  /// reasoning when han_scale_scope is kAnswer. Every sampling path and
  /// each GPU row uses the value of the state that samples it.
  [[nodiscard]] double han_bias() const noexcept {
    return reasoning_scoped_ && InReasoning() ? 0 : han_bias_;
  }
  /// The logit-bias offsets of the next token's phase (vocabulary-indexed),
  /// or null when none apply there. Every sampling path and each GPU row
  /// uses the offsets of the state that samples it.
  [[nodiscard]] const std::vector<float>* logit_bias() const noexcept {
    if (!config_.logit_bias)
      return nullptr;
    return InReasoning() ? config_.logit_bias->reasoning.get()
                         : config_.logit_bias->answer.get();
  }
  /// The largest positive offset of logit_bias(), or zero.
  [[nodiscard]] double logit_bias_max() const noexcept {
    if (!config_.logit_bias)
      return 0;
    return InReasoning() ? config_.logit_bias->reasoning_max
                         : config_.logit_bias->answer_max;
  }
  /// Whether an offset of the next token depends on the reasoning phase.
  [[nodiscard]] bool phase_scoped() const noexcept {
    return reasoning_scoped_ ||
           (config_.logit_bias && config_.logit_bias->phase_dependent());
  }
  [[nodiscard]] std::uint64_t rng_state() const noexcept;
  [[nodiscard]] std::uint64_t* mutable_rng_state() noexcept;
  void SetRngState(std::uint64_t state) noexcept;
  /// Publish RNG and any deferred draw without accepting tentative history.
  void CopyDrawStateFrom(const SamplerState& other) noexcept;
  struct DrawState {
    std::uint64_t rng;
    std::optional<TokenId> pending;
  };
  [[nodiscard]] DrawState SaveDrawState() const noexcept {
    return {rng_state_, pending_sample_};
  }
  void RestoreDrawState(DrawState state) noexcept {
    rng_state_ = state.rng;
    pending_sample_ = state.pending;
  }

  /// Starts a request: tokens initialize repetition only; generated counts
  /// reset.
  void ResetHistory(std::span<const TokenId> tokens);
  void Accept(TokenId token);
  void Accept(std::span<const TokenId> tokens);

  [[nodiscard]] SamplingDistribution Distribution(
      std::span<const float> logits) const;
  /// Compact logits use token_ids for penalties; returned IDs index logits.
  [[nodiscard]] SamplingDistribution Distribution(
      std::span<const float> logits, std::span<const TokenId> token_ids) const;
  /// Exact target top-k from a complete candidate union. The caller proves
  /// that omitted tokens cannot enter the post-penalty top-k. Candidates
  /// have unique vocabulary IDs; returned IDs are those vocabulary IDs.
  /// Requires bounded top-k, positive temperature and no active constraint.
  /// Retains the full-row sampler's FP64 filtering and normalization.
  [[nodiscard]] SamplingDistribution DistributionFromCandidates(
      std::span<const LogitCandidate> logits) const;
  /// Retain a speculative residual draw for the next Sample call. Copies
  /// preserve this draw along with the RNG; resetting history discards it.
  void DeferSample(TokenId token);
  [[nodiscard]] TokenId Sample(std::span<const float> logits);
  [[nodiscard]] TokenId SampleResidual(
      std::span<const float> target_logits,
      std::span<const TokenId> candidate_ids,
      std::span<const float> candidate_probabilities);
  [[nodiscard]] double Uniform();

private:
  [[nodiscard]] std::vector<float> ConstrainedLogits(
      std::span<const float> logits, std::span<const TokenId> ids = {}) const;
  void TrimHistory();
  void RebuildPenaltyCounts();
  [[nodiscard]] double AdjustedLogit(TokenId token, float logit) const noexcept;
  [[nodiscard]] TokenId SampleGreedy(std::span<const float> logits) const;
  [[nodiscard]] TokenId SampleConstrainedGreedy(
      std::span<const float> logits) const;
  [[nodiscard]] SamplingDistribution LinearDistribution(
      std::span<const float> logits) const;
  void PrepareSelected(std::span<const float> logits);
  void FilterSelected(double full_softmax_sum = 0,
                      bool has_full_softmax_sum = false);
  [[nodiscard]] SamplingDistribution SelectedDistribution();
  void StartReasoning();
  void AdvanceReasoning(TokenId token);

  SamplingConfig config_;
  double han_bias_{0};  ///< config_.han_bias(), fixed with config_
  /// han_bias_ != 0 and han_scale_scope kAnswer: reasoning zeroes the offset.
  bool reasoning_scoped_{false};
  /// A WithoutConstraint copy's grammar, kept only to end its reasoning
  /// phase as the target does; constraint_state_ is then its state.
  std::shared_ptr<const TokenConstraint> reasoning_grammar_;
  JsonConstraint::State constraint_state_;
  mutable std::shared_ptr<const std::vector<std::uint8_t>> constraint_mask_;
  mutable bool constraint_restricts_{false};
  std::vector<TokenId> history_;
  std::vector<TokenPenalty> penalty_counts_;
  std::vector<Probability> candidate_scratch_;
  std::uint64_t rng_state_{0};
  std::optional<TokenId> pending_sample_;
  enum class ReasoningPhase : std::uint8_t {
    kOff,
    kCounting,
    kWaitingUtf8,
    kForcing,
  };
  ReasoningPhase reasoning_phase_{ReasoningPhase::kOff};
  std::size_t reasoning_remaining_{0};
  std::size_t reasoning_forced_{0};
};

[[nodiscard]] std::uint64_t NextRandom(std::uint64_t* state);
[[nodiscard]] double Uniform(std::uint64_t* state);

// Compatibility helpers for callers that have not yet adopted SamplerState.
[[nodiscard]] TokenId SampleLogits(std::span<const float> logits,
                                   float temperature, std::uint64_t* rng_state);
[[nodiscard]] double TokenProbability(std::span<const float> logits,
                                      float temperature, TokenId token);
[[nodiscard]] TokenId SampleResidual(
    std::span<const float> target_logits, float temperature,
    std::span<const TokenId> candidate_ids,
    std::span<const float> candidate_probabilities, std::uint64_t* rng_state);

}  // namespace gufo::sampling

#endif  // GUFO_CORE_SAMPLING_HPP_
