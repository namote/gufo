#include "src/core/sampling.hpp"

#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/json_constraint.hpp"

namespace {

using gufo::sampling::ConstraintVocabulary;
using gufo::sampling::JsonConstraint;
using gufo::sampling::ReasoningBudget;
using gufo::sampling::SamplerState;
using gufo::sampling::SamplingConfig;
using gufo::sampling::TokenConstraint;
using gufo::sampling::TokenId;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

// Greedy logits whose argmax is `winner`; token 3 stands for `</think>`.
std::vector<float> Logits(TokenId winner) {
  std::vector<float> logits(8, 0.0F);
  logits[winner] = 5.0F;
  return logits;
}

std::shared_ptr<const ReasoningBudget> Budget(
    std::size_t tokens, std::vector<TokenId> forced = {3},
    std::vector<TokenId> prompt = {}) {
  auto budget = std::make_shared<ReasoningBudget>();
  budget->tokens = tokens;
  budget->forced = std::move(forced);
  budget->ends = {3};
  budget->prompt = std::move(prompt);
  return budget;
}

void TestReasoningBudgetForcesTheEnd() {
  SamplingConfig config;
  config.reasoning_budget = Budget(2);
  SamplerState sampler(config);
  Expect(!config.can_use_unmodified_argmax(),
         "a budget keeps requests off the unmodified argmax path");
  Expect(!sampler.ForcedToken() && sampler.CanSelectArgmax(1) &&
             !sampler.NeedsConstraintMask(),
         "the budget does not restrict reasoning before it is spent");
  sampler.Accept(sampler.Sample(Logits(1)));
  sampler.Accept(sampler.Sample(Logits(2)));
  Expect(sampler.ForcedToken() == TokenId{3},
         "two reasoning tokens exhaust a budget of two");
  Expect(sampler.NeedsConstraintMask(),
         "GPU sampling paths defer the forced end to the host sampler");
  Expect(!sampler.CanSelectArgmax(1) && sampler.CanSelectArgmax(3),
         "only the reasoning end is a selectable argmax");
  Expect(sampler.Sample(Logits(1)) == 3 &&
             sampler.Distribution(Logits(1)).best_token() == 3,
         "sampling and distributions return the forced end");
  const std::vector<TokenId> ids{5, 3, 1};
  const std::vector<float> compact{1.0F, 0.0F, 2.0F};
  Expect(sampler.Distribution(compact, ids).best_token() == 1,
         "compact distributions index the forced end among their IDs");
  // An MTP draft shortlist may not contain the forced end. Its proposal then
  // follows the shortlist; target verification rejects it and emits the end.
  const std::vector<TokenId> shortlist{5, 4, 1};
  Expect(sampler.Distribution(compact, shortlist).best_token() == 2,
         "a shortlist without the forced end keeps its own distribution");
  auto copy = sampler;
  copy.Accept(3);
  Expect(!copy.ForcedToken() && !copy.NeedsConstraintMask() &&
             sampler.ForcedToken(),
         "copies advance independently, as speculative verification needs");
  Expect(copy.Sample(Logits(6)) == 6, "the answer is sampled normally");
  copy.ResetHistory({});
  Expect(!copy.ForcedToken(), "a restart resets the reasoning budget");
}

void TestEarlyEndAndNoBudget() {
  SamplingConfig config;
  config.reasoning_budget = Budget(4);
  SamplerState sampler(config);
  sampler.Accept(std::vector<TokenId>{1, 3, 1, 1, 1, 1});
  Expect(!sampler.ForcedToken(),
         "a reasoning end within the budget closes the phase for good");
  SamplerState unlimited(SamplingConfig{});
  unlimited.Accept(std::vector<TokenId>(100, 1));
  Expect(!unlimited.ForcedToken(), "requests without a budget are unchanged");
  config.reasoning_budget = Budget(0);
  Expect(SamplerState(config).ForcedToken() == TokenId{3},
         "a zero budget closes reasoning immediately");
}

// Qwen as llama.cpp runs it: the prompt ends with "<think>\n", the forced end
// is "\n</think>" (tokens 4, 3) and token 7 (a tool call) also ends reasoning.
void TestReasoningBudgetMatchesLlamaCpp() {
  SamplingConfig config;
  for (const std::size_t tokens : {1U, 2U, 4U}) {
    config.reasoning_budget = Budget(tokens, {4, 3}, {4});
    SamplerState sampler(config);
    std::vector<TokenId> emitted;
    for (int step = 0; step < 8; ++step) {
      const auto token = sampler.Sample(Logits(1));
      sampler.Accept(token);
      emitted.push_back(token);
    }
    std::vector<TokenId> expected(tokens - 1, 1);
    expected.insert(expected.end(), {4, 3});
    expected.resize(8, 1);
    Expect(emitted == expected,
           "the prompt newline counts and the newline end is forced");
  }
  config.reasoning_budget = Budget(0, {4, 3}, {4});
  SamplerState zero(config);
  Expect(zero.ForcedToken() == TokenId{3},
         "a zero budget's prompt newline stands for the forced newline");
  zero.Accept(3);
  Expect(!zero.ForcedToken(), "the forced end completes reasoning");

  auto ending = std::make_shared<ReasoningBudget>(*Budget(2, {4, 3}));
  ending->ends.push_back(7);
  config.reasoning_budget = ending;
  SamplerState tool(config);
  tool.Accept(7);
  tool.Accept(std::vector<TokenId>(4, 1));
  Expect(!tool.ForcedToken(), "a tool call ends reasoning before the budget");

  // Token 5 ends inside a UTF-8 character and token 6 completes it.
  auto utf8 = std::make_shared<ReasoningBudget>(*Budget(2));
  utf8->partial_utf8 = std::make_shared<std::vector<bool>>(
      std::vector<bool>{false, false, false, false, false, true, false, false});
  config.reasoning_budget = utf8;
  SamplerState split(config);
  split.Accept(std::vector<TokenId>{1, 5, 5});
  Expect(!split.ForcedToken(), "a spent budget waits for a whole character");
  split.Accept(6);
  Expect(split.ForcedToken() == TokenId{3},
         "the end is forced after the character completes");
}

void TestReasoningBudgetKeepsLiteralAnswer() {
  SamplingConfig config;
  config.reasoning_budget = Budget(2, {4, 3}, {4});
  for (const bool natural_end : {false, true}) {
    SamplerState sampler(config);
    if (natural_end) {
      sampler.Accept(3);
    } else {
      sampler.Accept(std::vector<TokenId>{1, 4, 3});
    }
    // Token 0 stands for a literal <think> inside answer/tool data. Gufo's
    // framing recognizes reasoning markup only in the initial phase.
    sampler.Accept(std::vector<TokenId>{6, 0, 1, 2, 1, 2, 1, 2});
    Expect(!sampler.ForcedToken() && sampler.Sample(Logits(6)) == 6,
           "literal answer tokens must not restart a completed budget");
  }

  const std::vector<TokenId> thinking{1};
  Expect(gufo::sampling::BuildDistribution(Logits(1), config, {}, thinking)
                 .best_token() == 4,
         "distribution construction enforces an exhausted reasoning budget");
  const std::vector<TokenId> answer{1, 4, 3, 6, 0, 1, 2};
  Expect(gufo::sampling::BuildDistribution(Logits(1), config, {}, answer)
                 .best_token() == 1,
         "distribution construction preserves literal answer content");
}

// With a tool or format grammar, the forced end must be a legal grammar step
// and the answer after it stays constrained.
void TestReasoningBudgetWithGrammar() {
  const std::vector<std::string> pieces{
      "thinking", "</think>", "{\"x\":", "true}", "prose", ""};
  auto binding = std::make_shared<TokenConstraint>();
  binding->grammar = JsonConstraint::WithReasoning(JsonConstraint::Object());
  binding->vocabulary = std::make_shared<ConstraintVocabulary>(
      pieces.size(), [&](std::uint32_t i) {
        return ConstraintVocabulary::Piece{pieces[i], i == 5};
      });
  SamplingConfig config;
  config.constraint = binding;
  auto budget = std::make_shared<ReasoningBudget>();
  budget->tokens = 1;
  budget->forced = {1};
  budget->ends = {1};
  config.reasoning_budget = budget;
  SamplerState sampler(config);
  sampler.Accept(sampler.Sample(std::vector<float>{5, 0, 0, 0, 0, 0}));
  const auto forced = sampler.Sample(std::vector<float>{5, 0, 0, 0, 4, 0});
  Expect(forced == 1, "the grammar request also closes at its budget");
  sampler.Accept(forced);
  Expect(!sampler.ForcedToken() &&
             sampler.Sample(std::vector<float>{5, 0, 1, 0, 4, 0}) == 2,
         "after the forced end the answer still follows the grammar");
}

// The grammar decides when reasoning ends: the close spelled as ordinary
// text, or a native call header, ends it before the budget runs out, and the
// budget must not force its close into the answer afterwards.
void TestReasoningBudgetFollowsGrammarEnd() {
  const auto generate = [](std::shared_ptr<const JsonConstraint> grammar,
                           const std::vector<std::string>& pieces,
                           std::size_t tokens,
                           const std::vector<TokenId>& reasoning,
                           const std::vector<TokenId>& preferred) {
    auto binding = std::make_shared<TokenConstraint>();
    binding->grammar = std::move(grammar);
    binding->vocabulary = std::make_shared<ConstraintVocabulary>(
        pieces.size(), [&](std::uint32_t i) {
          return ConstraintVocabulary::Piece{pieces[i], pieces[i].empty()};
        });
    SamplingConfig config;
    config.constraint = binding;
    auto budget = std::make_shared<ReasoningBudget>();
    budget->tokens = tokens;
    budget->forced = {1};
    budget->ends = {1};
    config.reasoning_budget = budget;
    SamplerState sampler(config);
    std::string text;
    try {
      sampler.Accept(reasoning);
      for (const auto token : reasoning)
        text += pieces[token];
      for (const auto token : preferred) {
        std::vector<float> logits(pieces.size(), 0.0F);
        logits[token] = 5.0F;
        const auto sampled = sampler.Sample(logits);
        sampler.Accept(sampled);
        text += pieces[sampled];
      }
    } catch (const std::exception& error) {
      text += std::string("[") + error.what() + "]";
    }
    return text;
  };

  const auto json = JsonConstraint::WithReasoning(JsonConstraint::Object());
  const std::vector<std::string> pieces{"thinking", "</think>", "</", "think>",
                                        "{\"x\":",  "true}",    ""};
  const auto spelled = generate(json, pieces, 4, {0, 2, 3, 4}, {5, 6});
  Expect(spelled == "thinking</think>{\"x\":true}",
         "a close spelled as text ends the budget too, got " + spelled);
  const auto spent = generate(json, pieces, 4, {0, 0, 0, 2}, {4, 4, 5, 6});
  Expect(spent == "thinkingthinkingthinking</</think>{\"x\":true}",
         "a partial close still reasons and the budget forces the end, got " +
             spent);

  const auto arguments = JsonConstraint::ToolParameters(
      gufo::json::parse(R"({"type":"object",
        "properties":{"value":{"type":"string","const":"42"}},
        "required":["value"],"additionalProperties":false})"),
      true, JsonConstraint::ToolFormat::kDeepSeek);
  const auto tools = JsonConstraint::WithReasoning(
      JsonConstraint::WithTools(nullptr, {{"record", arguments}}, true, false,
                                JsonConstraint::ToolFormat::kDeepSeek));
  const std::vector<std::string> native{
      "thinking", "</think>",
      "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"record\">\n",
      "<｜DSML｜parameter name=\"value\" string=\"true\">42"
      "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
      ""};
  const auto called = generate(tools, native, 2, {0, 2}, {3, 4});
  Expect(called == native[0] + native[2] + native[3],
         "a native call header ends the budget too, got " + called);
}

// Bitwise equality of two distributions: same tokens, same FP64 values.
bool SameDistribution(const gufo::sampling::SamplingDistribution& a,
                      const gufo::sampling::SamplingDistribution& b) {
  if (a.entries().size() != b.entries().size())
    return false;
  for (std::size_t i = 0; i < a.entries().size(); ++i)
    if (a.entries()[i].token != b.entries()[i].token ||
        std::bit_cast<std::uint64_t>(a.entries()[i].value) !=
            std::bit_cast<std::uint64_t>(b.entries()[i].value))
      return false;
  return true;
}

// The Han scale's scope. Token 1 is Han and token 3 stands for </think>.
void TestHanScaleScope() {
  using gufo::sampling::HanScaleScope;
  using gufo::sampling::ParseHanScaleScope;
  Expect(ParseHanScaleScope("all") == HanScaleScope::kAll &&
             ParseHanScaleScope("answer") == HanScaleScope::kAnswer &&
             !ParseHanScaleScope("Answer") && !ParseHanScaleScope("") &&
             !ParseHanScaleScope("thinking"),
         "only all and answer name a scope");
  Expect(
      gufo::sampling::HanScaleScopeName(HanScaleScope::kAll) == "all" &&
          gufo::sampling::HanScaleScopeName(HanScaleScope::kAnswer) == "answer",
      "scope names round-trip");
  Expect(SamplingConfig{}.han_scale_scope == HanScaleScope::kAll,
         "the default scope is all");

  auto han = std::make_shared<gufo::sampling::TokenSet>();
  han->words = {1U << 1U};
  han->count = 1;
  // Token 1 (Han) leads token 0 by 0.5 nats; ln 0.5 = -0.69 reverses them.
  const std::vector<float> logits{1.0F, 1.5F, -1.0F, 0.0F, 0.2F, 0, 0, 0};
  const auto config = [&](float temperature, HanScaleScope scope,
                          std::shared_ptr<const ReasoningBudget> budget) {
    SamplingConfig result;
    result.temperature = temperature;
    result.han_scale = 0.5F;
    result.han_tokens = han;
    result.han_scale_scope = scope;
    result.reasoning_budget = std::move(budget);
    return result;
  };
  const auto unlimited = Budget(ReasoningBudget::kUnlimited);
  const double answer = std::log(0.5);

  for (const float temperature : {0.0F, 0.8F}) {
    // scope all: the offset ignores the phase, exactly as without a scope.
    SamplerState plain(config(temperature, HanScaleScope::kAll, nullptr));
    SamplerState all(config(temperature, HanScaleScope::kAll, Budget(100)));
    Expect(all.InReasoning() && all.han_bias() == answer &&
               plain.han_bias() == answer,
           "scope all offsets reasoning tokens");
    Expect(
        SameDistribution(all.Distribution(logits), plain.Distribution(logits)),
        "scope all reasoning matches the unscoped distribution bit for bit");
    Expect(all.WithoutConstraint().han_bias() == answer,
           "scope all tentative copies keep the offset");

    // scope answer: no offset before </think>, ln x after it.
    auto unscaled_config = config(temperature, HanScaleScope::kAll, nullptr);
    unscaled_config.han_scale = 1.0F;
    SamplerState unscaled(unscaled_config);
    SamplerState scoped(config(temperature, HanScaleScope::kAnswer, unlimited));
    Expect(scoped.InReasoning() && scoped.han_bias() == 0,
           "reasoning under the answer scope has no offset");
    Expect(scoped.config().penalties_enabled() &&
               !scoped.config().can_use_unmodified_argmax(),
           "the scoped request keeps the adjusted sampling path");
    Expect(SameDistribution(scoped.Distribution(logits),
                            unscaled.Distribution(logits)),
           "reasoning samples as Han scale 1");
    if (temperature == 0.0F)
      Expect(scoped.Sample(logits) == 1, "the Han argmax survives reasoning");
    scoped.Accept(std::vector<TokenId>{0, 2, 4, 1});
    Expect(scoped.InReasoning() && !scoped.ForcedToken(),
           "an unlimited budget only tracks the phase");
    scoped.Accept(TokenId{3});
    Expect(!scoped.InReasoning() && scoped.han_bias() == answer,
           "the token after </think> gets ln x");
    SamplerState after(config(temperature, HanScaleScope::kAll, nullptr));
    after.Accept(std::vector<TokenId>{0, 2, 4, 1, 3});
    Expect(SameDistribution(scoped.Distribution(logits),
                            after.Distribution(logits)),
           "the answer samples as scope all");
    if (temperature == 0.0F)
      Expect(scoped.Sample(logits) == 0, "the answer argmax is offset");
    // The compact rows MTP drafts and verifies with follow the phase too.
    const std::vector<float> compact{1.5F, 1.0F};
    const std::vector<TokenId> ids{1, 0};
    SamplerState fresh(config(temperature, HanScaleScope::kAnswer, unlimited));
    Expect(SameDistribution(
               fresh.Distribution(compact, ids),
               SamplerState(unscaled_config).Distribution(compact, ids)),
           "compact reasoning rows have no offset");
    Expect(SameDistribution(scoped.Distribution(compact, ids),
                            after.Distribution(compact, ids)),
           "compact answer rows have ln x");
    scoped.ResetHistory({});
    Expect(scoped.InReasoning() && scoped.han_bias() == 0,
           "a restart reasons again");
  }

  {
    // A verification span with </think> in the middle: each tentative row
    // (the state GPU verification builds per row) has its own offset.
    SamplerState sampler(config(0.0F, HanScaleScope::kAnswer, unlimited));
    const std::vector<TokenId> drafts{1, 3, 1, 0};
    auto tentative = sampler.WithoutConstraint();
    std::vector<double> biases;
    for (const auto draft : drafts) {
      biases.push_back(tentative.han_bias());
      tentative.Accept(draft);
    }
    Expect(biases == std::vector<double>{0, 0, answer, answer},
           "rows after a drafted </think> use the answer offset");
    // The target row by row agrees with the tentative offsets.
    for (std::size_t row = 0; row < drafts.size(); ++row) {
      Expect(sampler.han_bias() == biases[row], "target rows match drafts");
      sampler.Accept(drafts[row]);
    }
  }
  {
    // A Qwen budget with the answer scope: the forced end starts the answer.
    auto budget = std::make_shared<ReasoningBudget>(*Budget(1, {4, 3}));
    SamplerState sampler(config(0.0F, HanScaleScope::kAnswer, budget));
    sampler.Accept(sampler.Sample(logits));
    Expect(sampler.ForcedToken() == TokenId{4} && sampler.han_bias() == 0,
           "the forced end is still reasoning");
    sampler.Accept(sampler.Sample(logits));
    sampler.Accept(sampler.Sample(logits));
    Expect(!sampler.ForcedToken() && !sampler.InReasoning() &&
               sampler.han_bias() == answer,
           "the answer after a forced end has ln x");
  }
  {
    // A grammar ends reasoning at a close spelled as text. Tentative copies
    // drop the grammar but still follow that end, as the target does.
    const std::vector<std::string> pieces{
        "thinking", "</", "think>", "{\"x\":", "true}", "", "</think>", "x"};
    auto binding = std::make_shared<TokenConstraint>();
    binding->grammar = JsonConstraint::WithReasoning(JsonConstraint::Object());
    binding->vocabulary = std::make_shared<ConstraintVocabulary>(
        pieces.size(), [&](std::uint32_t i) {
          return ConstraintVocabulary::Piece{pieces[i], pieces[i].empty()};
        });
    auto scoped_config = config(0.0F, HanScaleScope::kAnswer, nullptr);
    scoped_config.constraint = binding;
    auto ends = std::make_shared<ReasoningBudget>(*unlimited);
    ends->ends = {6};
    scoped_config.reasoning_budget = ends;
    SamplerState target(scoped_config);
    auto tentative = target.WithoutConstraint();
    const std::vector<TokenId> drafts{0, 1, 2, 3};
    for (const auto draft : drafts) {
      Expect(tentative.han_bias() == target.han_bias(),
             "a tentative copy follows the grammar's end of reasoning");
      tentative.Accept(draft);
      target.Accept(draft);
    }
    Expect(!target.InReasoning() && !tentative.InReasoning() &&
               tentative.han_bias() == answer,
           "a close spelled as text starts the answer in both");
    Expect(!tentative.NeedsConstraintMask(),
           "the tentative copy stays unconstrained");
    // A draft the grammar rejects ends tracking in the copy: the target can
    // never accept it, so its later rows are discarded.
    auto rejected = SamplerState(scoped_config).WithoutConstraint();
    rejected.Accept(TokenId{0});
    rejected.Accept(TokenId{5});
    Expect(!rejected.InReasoning(), "a rejected draft ends tentative tracking");
    // So does a padded head ID outside the grammar's vocabulary.
    auto padded = SamplerState(scoped_config).WithoutConstraint();
    padded.Accept(TokenId{20});
    Expect(!padded.InReasoning(), "an out-of-vocabulary draft does not throw");
    // Scope all copies keep no grammar: unchanged from before the scope.
    auto all_config = scoped_config;
    all_config.han_scale_scope = HanScaleScope::kAll;
    auto all = SamplerState(all_config).WithoutConstraint();
    all.Accept(std::vector<TokenId>{0, 1, 2});
    Expect(all.InReasoning() && all.han_bias() == answer,
           "scope all copies do not follow the grammar");
  }
}

void TestCompactTargetContract() {
  using gufo::sampling::LogitCandidate;
  const std::vector<float> logits{0.0F, 0.75F, -1.0F, 0.75F, 3.0F, 0.1F};
  // Deliberately unordered IDs. A frequency penalty can promote a token
  // whose raw logit is far below the original top-k.
  const std::vector<LogitCandidate> candidates{
      {4, 3.0F}, {2, -1.0F}, {3, 0.75F}, {1, 0.75F}};
  SamplingConfig config{.temperature = 0.7F,
                        .top_k = 2,
                        .top_p = 0.8F,
                        .frequency_penalty = -5.0F};
  SamplerState state(config);
  state.Accept(2);
  const auto expected = state.Distribution(logits);
  const auto actual = state.DistributionFromCandidates(candidates);
  Expect(expected.entries().size() == actual.entries().size(),
         "compact target retains full top-k support");
  for (std::size_t i = 0; i < expected.entries().size(); ++i)
    Expect(expected.entries()[i].token == actual.entries()[i].token &&
               std::bit_cast<std::uint64_t>(expected.entries()[i].value) ==
                   std::bit_cast<std::uint64_t>(actual.entries()[i].value),
           "compact target retains exact FP64 probabilities");
  for (unsigned mode = 0; mode < 3; ++mode) {
    auto unsupported = config;
    if (mode == 0)
      unsupported.temperature = 0;
    else if (mode == 1)
      unsupported.top_k = 0;
    else
      unsupported.reasoning_budget = Budget(0);
    bool refused = false;
    try {
      (void)SamplerState(unsupported).DistributionFromCandidates(candidates);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    Expect(refused, "compact target refuses unsupported selection semantics");
  }
}

}  // namespace

void TestRawLogprobs() {
  using gufo::sampling::CollectLogprobs;
  const std::vector<float> logits{1000, 999, 998, -1000};
  const auto result = CollectLogprobs(logits, 3, 2);
  const double denominator = std::log(1.0 + std::exp(-1.0) + std::exp(-2.0));
  Expect(std::abs(result->selected.logprob - (-2000.0 - denominator)) < 1e-12,
         "stable log-softmax retains very unlikely selected tokens");
  Expect(result->top.size() == 2 && result->top[0].token == 0 &&
             result->top[1].token == 1,
         "alternatives come from the complete raw vocabulary");
  Expect(!CollectLogprobs({}, 999, std::nullopt),
         "disabled collection does no work");
  Expect(CollectLogprobs(logits, 0, 0)->top.empty(),
         "zero requests selected token only");
  const auto tied = CollectLogprobs(std::vector<float>{0, 0, 0}, 2, 20);
  Expect(tied->top.size() == 3 && tied->top[0].token == 0 &&
             std::abs(tied->selected.logprob + std::log(3.0)) < 1e-12,
         "ties have deterministic token order and exact normalization");
  const auto twenty = CollectLogprobs(std::vector<float>(32, 0), 31, 20);
  Expect(twenty->top.size() == 20 && twenty->top.back().token == 19 &&
             twenty->selected.token == 31,
         "twenty alternatives retain an independently selected token");
  const std::vector<std::string> pieces{"prose", "{", "}", ""};
  auto binding = std::make_shared<TokenConstraint>();
  binding->grammar = JsonConstraint::Object();
  binding->vocabulary = std::make_shared<ConstraintVocabulary>(
      pieces.size(), [&](std::uint32_t i) {
        return ConstraintVocabulary::Piece{pieces[i], i == 3};
      });
  SamplingConfig constrained;
  constrained.constraint = binding;
  constrained.logprobs = 2;
  SamplerState grammar(constrained);
  const auto selected = grammar.Sample(logits);
  const auto raw = CollectLogprobs(logits, selected, constrained.logprobs);
  Expect(selected == 1 && raw->top[0].token == 0 &&
             std::abs(raw->selected.logprob + 1 + denominator) < 1e-12,
         "grammar masks change selection but not raw probabilities");
  for (const float temperature : {0.F, 0.7F}) {
    SamplingConfig config;
    config.temperature = temperature;
    config.top_k = 2;
    config.presence_penalty = 0.5;
    config.seed = 42;
    SamplerState plain(config);
    config.logprobs = 2;
    SamplerState observed(config);
    for (int i = 0; i < 16; ++i) {
      const auto a = plain.Sample(logits), b = observed.Sample(logits);
      const auto probabilities = CollectLogprobs(logits, b, config.logprobs);
      Expect(a == b && plain.rng_state() == observed.rng_state(),
             "observation preserves selection and RNG");
      Expect(std::abs(probabilities->top[0].logprob + denominator) < 1e-12,
             "raw logprobs ignore temperature, truncation and penalties");
      plain.Accept(a);
      observed.Accept(b);
    }
  }
  for (const auto invalid :
       {std::vector<float>{INFINITY}, std::vector<float>{NAN},
        std::vector<float>{-INFINITY}}) {
    bool failed = false;
    try {
      (void)CollectLogprobs(invalid, 0, 0);
    } catch (const std::runtime_error&) {
      failed = true;
    }
    Expect(failed, "invalid raw distributions fail explicitly");
  }
}

int main() {
  TestRawLogprobs();
  TestReasoningBudgetForcesTheEnd();
  TestEarlyEndAndNoBudget();
  TestReasoningBudgetMatchesLlamaCpp();
  TestReasoningBudgetKeepsLiteralAnswer();
  TestReasoningBudgetWithGrammar();
  TestReasoningBudgetFollowsGrammarEnd();
  TestHanScaleScope();
  TestCompactTargetContract();
  std::cout << "sampling_test passed\n";
  return 0;
}
