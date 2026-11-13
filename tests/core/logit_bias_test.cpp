#include "src/core/logit_bias.hpp"

#include <bit>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"

namespace {

using gufo::sampling::BuildLogitBias;
using gufo::sampling::LogitBias;
using gufo::sampling::LogitBiasCatalog;
using gufo::sampling::LogitBiasProfile;
using gufo::sampling::LogitBiasScope;
using gufo::sampling::LogitBiasSelection;
using gufo::sampling::LogitBiasTerm;
using gufo::sampling::ParseLogitBiasProfile;
using gufo::sampling::ReasoningBudget;
using gufo::sampling::SamplerState;
using gufo::sampling::SamplingConfig;
using gufo::sampling::SamplingDistribution;
using gufo::sampling::TokenId;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

template<typename Function>
bool Throws(Function&& function, std::string_view fragment = {}) {
  try {
    function();
  } catch (const std::invalid_argument& error) {
    if (std::string_view(error.what()).find(fragment) ==
        std::string_view::npos) {
      std::cerr << "unexpected message: " << error.what() << '\n';
      return false;
    }
    return true;
  }
  return false;
}

bool SameDistribution(const SamplingDistribution& a,
                      const SamplingDistribution& b) {
  if (a.entries().size() != b.entries().size())
    return false;
  for (std::size_t i = 0; i < a.entries().size(); ++i)
    if (a.entries()[i].token != b.entries()[i].token ||
        std::bit_cast<std::uint64_t>(a.entries()[i].value) !=
            std::bit_cast<std::uint64_t>(b.entries()[i].value))
      return false;
  return true;
}

// Token 3 stands for </think>.
std::shared_ptr<const ReasoningBudget> Unlimited() {
  auto budget = std::make_shared<ReasoningBudget>();
  budget->tokens = ReasoningBudget::kUnlimited;
  budget->ends = {3};
  return budget;
}

// A small vocabulary: decoded token texts by id.
const std::vector<std::string> kVocabulary{
    "<s>", "Wait", " Wait", "wait", "Hmm", " Hmm,", "So", " Alternatively",
    "  ", "\nWait", "Waiting", "Wait "};

std::string Decode(TokenId token) {
  return kVocabulary.at(token);
}

void TestScopes() {
  using gufo::sampling::LogitBiasScopeName;
  using gufo::sampling::ParseLogitBiasScope;
  Expect(ParseLogitBiasScope("all") == LogitBiasScope::kAll &&
             ParseLogitBiasScope("answer") == LogitBiasScope::kAnswer &&
             ParseLogitBiasScope("reasoning") == LogitBiasScope::kReasoning &&
             !ParseLogitBiasScope("thinking") && !ParseLogitBiasScope("All") &&
             !ParseLogitBiasScope(""),
         "three scope spellings");
  for (const auto scope : {LogitBiasScope::kAll, LogitBiasScope::kAnswer,
                           LogitBiasScope::kReasoning})
    Expect(ParseLogitBiasScope(LogitBiasScopeName(scope)) == scope,
           "scope names round-trip");
}

void TestParse() {
  const auto profile = ParseLogitBiasProfile(
      R"({"format": "gufo-logit-bias-v1", "name": "wait", "scope": "reasoning",
          "bias": -4.0, "tokens": [6, 1], "strings": ["Wait", "Hmm"]})",
      "wait.json");
  Expect(profile.name == "wait" &&
             profile.scope == LogitBiasScope::kReasoning &&
             profile.bias == -4.0F &&
             profile.tokens == std::vector<TokenId>{6, 1} &&
             profile.strings == std::vector<std::string>{"Wait", "Hmm"},
         "a profile parses");
  const auto tokens_only = ParseLogitBiasProfile(
      R"({"format":"gufo-logit-bias-v1","name":"a.b-c_1","scope":"all",
          "bias":100,"tokens":[0]})",
      "x");
  Expect(tokens_only.strings.empty() && tokens_only.bias == 100.0F,
         "tokens alone and the bound itself are accepted");
  const auto rejects = [](std::string_view text, std::string_view fragment) {
    Expect(Throws([&] { (void)ParseLogitBiasProfile(text, "bad.json"); },
                  fragment),
           "rejects: " + std::string(text));
  };
  const std::string base =
      R"("format":"gufo-logit-bias-v1","name":"w","scope":"all")";
  rejects("{" + base + R"(,"bias":-1,"tokens":[1],"extra":1})",
          "unknown key 'extra'");
  rejects(R"({"format":"gufo-logit-bias-v2","name":"w","scope":"all",
              "bias":-1,"tokens":[1]})",
          "'format'");
  rejects(R"({"name":"w","scope":"all","bias":-1,"tokens":[1]})",
          "missing 'format'");
  rejects(R"({"format":"gufo-logit-bias-v1","scope":"all","bias":-1,
              "tokens":[1]})",
          "missing 'name'");
  rejects(R"({"format":"gufo-logit-bias-v1","name":"w","bias":-1,
              "tokens":[1]})",
          "missing 'scope'");
  rejects("{" + base + R"(,"tokens":[1]})", "missing 'bias'");
  rejects(R"({"format":"gufo-logit-bias-v1","name":"w","scope":"thinking",
              "bias":-1,"tokens":[1]})",
          "'scope'");
  rejects("{" + base + R"(,"bias":-100.5,"tokens":[1]})", "'bias'");
  rejects("{" + base + R"(,"bias":"-1","tokens":[1]})", "'bias'");
  rejects("{" + base + R"(,"bias":-1,"tokens":[-1]})", "'tokens'");
  rejects("{" + base + R"(,"bias":-1,"tokens":[1.5]})", "'tokens'");
  rejects("{" + base + R"(,"bias":-1,"tokens":"1"})", "'tokens'");
  rejects("{" + base + R"(,"bias":-1,"strings":[""]})", "'strings'");
  rejects("{" + base + R"(,"bias":-1,"strings":[1]})", "'strings'");
  rejects("{" + base + R"(,"bias":-1})", "needs 'tokens' or 'strings'");
  rejects(R"({"format":"gufo-logit-bias-v1","name":"a b","scope":"all",
              "bias":-1,"tokens":[1]})",
          "'name'");
  rejects(R"({"format":"gufo-logit-bias-v1","name":"","scope":"all",
              "bias":-1,"tokens":[1]})",
          "'name'");
  rejects("[1]", "JSON object");
  rejects("{", "invalid JSON");
}

void TestStringExpansion() {
  using gufo::sampling::LogitBiasStringMatches;
  Expect(LogitBiasStringMatches("Wait", "Wait") &&
             LogitBiasStringMatches(" Wait", "Wait") &&
             LogitBiasStringMatches("\n\t Wait", "Wait") &&
             !LogitBiasStringMatches("wait", "Wait") &&
             !LogitBiasStringMatches("Waiting", "Wait") &&
             !LogitBiasStringMatches("Wait ", "Wait") &&
             !LogitBiasStringMatches("  ", "Wait"),
         "only leading whitespace is ignored");

  std::vector<LogitBiasProfile> profiles(2);
  profiles[0].name = "wait";
  profiles[0].strings = {"Wait", "Hmm"};
  profiles[0].tokens = {6, 1};
  profiles[1].name = "alt";
  profiles[1].strings = {"Alternatively", "Nothing"};
  gufo::sampling::ResolveLogitBiasProfiles(profiles, kVocabulary.size(),
                                           Decode);
  Expect(profiles[0].tokens == std::vector<TokenId>{1, 2, 4, 6, 9} &&
             profiles[0].string_matches == 4,
         "strings expand to every exact match, merged with listed ids");
  Expect(profiles[1].tokens == std::vector<TokenId>{7} &&
             profiles[1].string_matches == 1,
         "a string without a match leaves the others");

  const auto resolve_rejects = [](std::vector<LogitBiasProfile> list,
                                  std::string_view fragment) {
    Expect(Throws(
               [&] {
                 gufo::sampling::ResolveLogitBiasProfiles(
                     list, kVocabulary.size(), Decode);
               },
               fragment),
           "resolution rejects: " + std::string(fragment));
  };
  std::vector<LogitBiasProfile> none(1);
  none[0].name = "none";
  none[0].strings = {"Nothing"};
  resolve_rejects(none, "resolves to no token");
  std::vector<LogitBiasProfile> outside(1);
  outside[0].name = "outside";
  outside[0].tokens = {static_cast<TokenId>(kVocabulary.size())};
  resolve_rejects(outside, "outside the vocabulary");
  std::vector<LogitBiasProfile> twice(2);
  twice[0].name = twice[1].name = "same";
  twice[0].tokens = twice[1].tokens = {1};
  resolve_rejects(twice, "two profiles are named 'same'");
}

void TestLoad() {
  namespace fs = std::filesystem;
  const auto directory =
      fs::temp_directory_path() /
      ("gufo-logit-bias-test-" + std::to_string(std::random_device{}()));
  fs::remove_all(directory);
  fs::create_directories(directory);
  const auto write = [&](const std::string& name, const std::string& text) {
    std::ofstream(directory / name) << text;
  };
  Expect(Throws(
             [&] {
               (void)gufo::sampling::LoadLogitBiasProfiles(
                   directory, kVocabulary.size(), Decode);
             },
             "no *.json profile"),
         "an empty directory fails");
  write("b.json", R"({"format":"gufo-logit-bias-v1","name":"hmm",
                      "scope":"answer","bias":-2,"strings":["Hmm"]})");
  write("a.json", R"({"format":"gufo-logit-bias-v1","name":"wait",
                      "scope":"reasoning","bias":-4,"strings":["Wait"]})");
  write("notes.txt", "ignored");
  const auto profiles = gufo::sampling::LoadLogitBiasProfiles(
      directory, kVocabulary.size(), Decode);
  Expect(profiles.size() == 2 && profiles[0].name == "wait" &&
             profiles[1].name == "hmm" &&
             profiles[0].tokens == std::vector<TokenId>{1, 2, 9} &&
             profiles[1].tokens == std::vector<TokenId>{4},
         "every *.json loads in file name order");
  write("c.json", R"({"format":"gufo-logit-bias-v1","name":"x",
                      "scope":"all","bias":-2,"tokens":[1],"typo":true})");
  Expect(Throws(
             [&] {
               (void)gufo::sampling::LoadLogitBiasProfiles(
                   directory, kVocabulary.size(), Decode);
             },
             "c.json: unknown key 'typo'"),
         "a bad file fails the load and is named");
  fs::remove_all(directory);
  Expect(Throws(
             [&] {
               (void)gufo::sampling::LoadLogitBiasProfiles(
                   directory, kVocabulary.size(), Decode);
             },
             "needs a directory"),
         "a missing directory fails");
}

void TestBuild() {
  const std::vector<TokenId> a{1, 2};
  const std::vector<TokenId> b{2, 5};
  const std::vector<LogitBiasTerm> all{
      {LogitBiasScope::kAll, -2.0F, a},
      {LogitBiasScope::kAll, 1.5F, b},
  };
  const auto unscoped = BuildLogitBias(8, all, "k");
  Expect(unscoped->reasoning && unscoped->reasoning == unscoped->answer &&
             !unscoped->phase_dependent(),
         "scope all shares one array");
  Expect(*unscoped->answer ==
             std::vector<float>{0, -2.0F, -0.5F, 0, 0, 1.5F, 0, 0},
         "overlapping sets add");
  Expect(unscoped->answer_max == 1.5F && unscoped->reasoning_max == 1.5F,
         "the largest positive offset bounds the phase");
  const std::vector<LogitBiasTerm> scoped{
      {LogitBiasScope::kReasoning, -2.0F, a},
      {LogitBiasScope::kAll, -1.0F, b},
  };
  const auto phased = BuildLogitBias(8, scoped, "k");
  Expect(phased->phase_dependent() &&
             *phased->reasoning ==
                 std::vector<float>{0, -2.0F, -3.0F, 0, 0, -1.0F, 0, 0} &&
             *phased->answer ==
                 std::vector<float>{0, 0, -1.0F, 0, 0, -1.0F, 0, 0} &&
             phased->reasoning_max == 0 && phased->answer_max == 0,
         "scoped sets apply to their phase only");
  const std::vector<LogitBiasTerm> reasoning_only{
      {LogitBiasScope::kReasoning, -2.0F, a}};
  const auto only = BuildLogitBias(8, reasoning_only, "k");
  Expect(only->reasoning && !only->answer && only->phase_dependent(),
         "a phase no set applies to stays null");
  const std::vector<TokenId> outside{8};
  const std::vector<LogitBiasTerm> bad{{LogitBiasScope::kAll, -1.0F, outside}};
  Expect(Throws([&] { (void)BuildLogitBias(8, bad, "k"); }, "outside"),
         "ids beyond the vocabulary fail");
}

LogitBiasCatalog Catalog() {
  std::vector<LogitBiasProfile> profiles(3);
  profiles[0].name = "wait";
  profiles[0].scope = LogitBiasScope::kReasoning;
  profiles[0].bias = -4.0F;
  profiles[0].tokens = {1};
  profiles[1].name = "answer";
  profiles[1].scope = LogitBiasScope::kAnswer;
  profiles[1].bias = -4.0F;
  profiles[1].tokens = {1};
  profiles[2].name = "boost";
  profiles[2].scope = LogitBiasScope::kAll;
  profiles[2].bias = 2.0F;
  profiles[2].tokens = {2};
  return LogitBiasCatalog(8, std::move(profiles));
}

void TestCatalog() {
  const auto catalog = Catalog();
  Expect(catalog.Find("wait") != nullptr && catalog.Find("none") == nullptr,
         "profiles are found by name");
  Expect(catalog.Resolve({}) == nullptr, "an empty selection adds nothing");
  LogitBiasSelection ab;
  ab.profiles = {{"wait", std::nullopt}, {"boost", std::nullopt}};
  LogitBiasSelection ba;
  ba.profiles = {{"boost", std::nullopt}, {"wait", std::nullopt}};
  const auto first = catalog.Resolve(ab);
  Expect(first && first == catalog.Resolve(ba),
         "equal selections share one cached combination");
  Expect(first->key == "boost=1p+1,wait=-1p+2",
         "the key lists profiles in name order with exact biases");
  Expect((*first->reasoning)[1] == -4.0F && (*first->reasoning)[2] == 2.0F &&
             (*first->answer)[1] == 0.0F && (*first->answer)[2] == 2.0F,
         "each profile keeps its scope");
  LogitBiasSelection overridden = ab;
  overridden.profiles[0].second = -6.0F;
  const auto changed = catalog.Resolve(overridden);
  Expect(changed != first && (*changed->reasoning)[1] == -6.0F,
         "an override replaces the profile's bias");
  LogitBiasSelection openai;
  openai.tokens = {{4, -100.0F}, {5, 3.0F}};
  const auto map = catalog.Resolve(openai);
  Expect(!map->phase_dependent() && (*map->answer)[4] == -100.0F &&
             (*map->answer)[5] == 3.0F && map->answer_max == 3.0F,
         "the OpenAI map is a set with scope all");
  LogitBiasSelection unknown;
  unknown.profiles = {{"none", std::nullopt}};
  Expect(Throws([&] { (void)catalog.Resolve(unknown); }, "unknown"),
         "unknown names fail");
  LogitBiasSelection outside;
  outside.tokens = {{8, 1.0F}};
  Expect(Throws([&] { (void)catalog.Resolve(outside); }, "outside"),
         "tokens beyond the vocabulary fail");
}

// Logits whose greedy winner is token 1 by one nat over token 0.
const std::vector<float> kLogits{1.0F, 2.0F, 0.5F, 0.0F, -1.0F, 0.25F, 0, 0};

SamplingConfig Config(float temperature,
                      std::shared_ptr<const LogitBias> bias,
                      std::shared_ptr<const ReasoningBudget> budget) {
  SamplingConfig config;
  config.temperature = temperature;
  config.logit_bias = std::move(bias);
  config.reasoning_budget = std::move(budget);
  return config;
}

std::vector<float> Offset(std::vector<float> logits,
                          const std::vector<float>& bias) {
  for (std::size_t i = 0; i < logits.size(); ++i)
    logits[i] += bias[i];
  return logits;
}

void TestSamplerScopes() {
  const auto catalog = Catalog();
  const auto select = [&](std::string name) {
    LogitBiasSelection selection;
    selection.profiles = {{std::move(name), std::nullopt}};
    return catalog.Resolve(selection);
  };
  const auto reasoning = select("wait");
  const auto answer = select("answer");
  // -4 on token 1: 2 - 4 = -2 loses to token 0.
  const std::vector<float> suppressed = Offset(kLogits, *reasoning->reasoning);

  {
    const auto config = Config(0.0F, reasoning, Unlimited());
    Expect(config.penalties_enabled() && !config.can_use_unmodified_argmax() &&
               config.logit_bias_active(),
           "an active set keeps the adjusted path");
    SamplerState sampler(config);
    Expect(sampler.InReasoning() && sampler.phase_scoped() &&
               !sampler.CanSelectArgmax(1),
           "a scoped set tracks the phase and blocks the raw argmax");
    Expect(sampler.Sample(kLogits) == 0, "reasoning suppresses token 1");
    sampler.Accept(TokenId{3});
    Expect(!sampler.InReasoning() && sampler.logit_bias() == nullptr &&
               sampler.Sample(kLogits) == 1,
           "the answer is unbiased under the reasoning scope");
  }
  {
    SamplerState sampler(Config(0.0F, answer, Unlimited()));
    Expect(sampler.logit_bias() == nullptr && sampler.Sample(kLogits) == 1,
           "reasoning is unbiased under the answer scope");
    sampler.Accept(TokenId{3});
    Expect(sampler.Sample(kLogits) == 0, "the answer suppresses token 1");
    // Without a tracked phase the whole reply is the answer.
    Expect(SamplerState(Config(0.0F, answer, nullptr)).Sample(kLogits) == 0 &&
               SamplerState(Config(0.0F, reasoning, nullptr))
                       .Sample(kLogits) == 1,
           "untracked replies are answer");
    Expect(gufo::sampling::BuildDistribution(kLogits,
                                             Config(0, answer, nullptr))
                   .best_token() == 0,
           "BuildDistribution applies the answer offsets");
  }

  // The biased distribution equals the unbiased one of offset logits, on
  // every CPU path: greedy, linear, top-k selection, top-p and compact rows.
  // The values are dyadic, so the FP64 sums are exact both ways.
  struct Filter {
    float temperature;
    std::int32_t top_k;
    float top_p;
    float min_p;
  };
  for (const auto filter : {Filter{0.0F, 0, 1.0F, 0.0F},
                            Filter{0.8F, 0, 1.0F, 0.0F},
                            Filter{0.8F, 0, 1.0F, 0.05F},
                            Filter{0.8F, 3, 1.0F, 0.0F},
                            Filter{0.8F, 0, 0.9F, 0.0F}}) {
    auto biased = Config(filter.temperature, reasoning, Unlimited());
    biased.top_k = filter.top_k;
    biased.top_p = filter.top_p;
    biased.min_p = filter.min_p;
    auto plain = biased;
    plain.logit_bias = nullptr;
    const SamplerState with(biased);
    const SamplerState without(plain);
    Expect(SameDistribution(with.Distribution(kLogits),
                            without.Distribution(suppressed)),
           "biased logits sample as offset logits");
    const std::vector<float> compact{kLogits[2], kLogits[1], kLogits[0]};
    const std::vector<float> compact_offset{suppressed[2], suppressed[1],
                                            suppressed[0]};
    const std::vector<TokenId> ids{2, 1, 0};
    Expect(SameDistribution(with.Distribution(compact, ids),
                            without.Distribution(compact_offset, ids)),
           "compact rows look offsets up by vocabulary id");
    if (filter.top_k > 0) {
      std::vector<gufo::sampling::LogitCandidate> candidates;
      for (TokenId token = 0; token < kLogits.size(); ++token)
        candidates.push_back({token, kLogits[token]});
      std::vector<gufo::sampling::LogitCandidate> offset_candidates;
      for (TokenId token = 0; token < kLogits.size(); ++token)
        offset_candidates.push_back({token, suppressed[token]});
      Expect(SameDistribution(
                 with.DistributionFromCandidates(candidates),
                 without.DistributionFromCandidates(offset_candidates)),
             "MTP candidate rows apply the offsets");
    }
  }

  // Tentative copies (per-row verification states) switch phase at a
  // drafted </think>, and each row's offsets match the target's.
  {
    SamplerState target(Config(0.0F, reasoning, Unlimited()));
    auto tentative = target.WithoutConstraint();
    std::vector<const std::vector<float>*> rows;
    for (const TokenId draft : {TokenId{0}, TokenId{3}, TokenId{1}}) {
      rows.push_back(tentative.logit_bias());
      tentative.Accept(draft);
    }
    Expect(rows == std::vector<const std::vector<float>*>{
                       reasoning->reasoning.get(), reasoning->reasoning.get(),
                       nullptr},
           "rows after a drafted </think> use the answer offsets");
    for (const TokenId draft : {TokenId{0}, TokenId{3}}) {
      target.Accept(draft);
    }
    Expect(target.logit_bias() == rows[2], "the target agrees row by row");
  }
}

void TestPositiveBiasAndBlockSkips() {
  // A boosted token in a later block whose raw maximum is below the best:
  // the block skips must still visit it.
  constexpr std::size_t kVocab = 512;
  std::vector<float> logits(kVocab, -4.0F);
  logits[3] = 8.0F;
  logits[300] = 0.0F;
  std::vector<gufo::sampling::LogitBiasProfile> profiles(1);
  profiles[0].name = "boost";
  profiles[0].bias = 16.0F;
  profiles[0].tokens = {300};
  const LogitBiasCatalog catalog(kVocab, std::move(profiles));
  LogitBiasSelection selection;
  selection.profiles = {{"boost", std::nullopt}};
  const auto bias = catalog.Resolve(selection);
  Expect(bias->answer_max == 16.0F, "the bound is the boost");
  SamplerState greedy(Config(0.0F, bias, nullptr));
  Expect(greedy.Sample(logits) == 300, "greedy finds a boosted later token");
  auto sampled = Config(0.7F, bias, nullptr);
  sampled.top_k = 2;
  const auto distribution = SamplerState(sampled).Distribution(logits);
  Expect(distribution.best_token() == 300 &&
             distribution.probability(3) > 0.0,
         "top-k selection keeps a boosted later token");
  sampled.top_k = 0;
  sampled.top_p = 0.5F;
  Expect(SamplerState(sampled).Distribution(logits).best_token() == 300,
         "top-p selection keeps a boosted later token");
}

void TestHanCombines() {
  // Han (ln 0.5 on token 1) adds first, then the logit bias.
  auto han = std::make_shared<gufo::sampling::TokenSet>();
  han->words = {1U << 1U};
  han->count = 1;
  SamplingConfig han_only;
  han_only.temperature = 0.8F;
  han_only.han_scale = 0.5F;
  han_only.han_tokens = han;
  const SamplerState reference(han_only);
  const auto catalog = Catalog();
  LogitBiasSelection selection;
  selection.profiles = {{"boost", std::nullopt}};
  auto combined = han_only;
  combined.logit_bias = catalog.Resolve(selection);
  const auto offset = Offset(kLogits, *combined.logit_bias->answer);
  Expect(SameDistribution(SamplerState(combined).Distribution(kLogits),
                          reference.Distribution(offset)),
         "the logit bias adds after the Han offset");
}

}  // namespace

int main() {
  TestScopes();
  TestParse();
  TestStringExpansion();
  TestLoad();
  TestBuild();
  TestCatalog();
  TestSamplerScopes();
  TestPositiveBiasAndBlockSkips();
  TestHanCombines();
  std::cout << "logit_bias_test passed\n";
  return 0;
}
