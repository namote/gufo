#include "src/core/han_script.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"

namespace {

using gufo::sampling::BuildHanTokens;
using gufo::sampling::IsHanToken;
using gufo::sampling::LogitCandidate;
using gufo::sampling::SamplerState;
using gufo::sampling::SamplingConfig;
using gufo::sampling::TokenId;
using gufo::sampling::TokenSet;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

void Near(double actual, double expected, std::string_view message) {
  Expect(std::abs(actual - expected) <= 1e-12 * std::max(1.0, expected),
         message);
}

void TestClassifier() {
  // Complete ideographs, alone and inside mixed tokens.
  Expect(IsHanToken("\xE4\xB8\xAD"), "U+4E2D");                // 中
  Expect(IsHanToken("\xE9\xBE\xA5"), "U+9FA5 (last common)");  // 龥
  Expect(IsHanToken("\xE3\x90\x80"), "U+3400 Extension A");    // 㐀
  Expect(IsHanToken("\xE4\xB6\xBF"), "U+4DBF Extension A end");
  Expect(IsHanToken("\xEF\xA4\x80"), "U+F900 Compatibility");  // 豈
  Expect(IsHanToken("\xF0\xA0\x80\x80"), "U+20000 Extension B");
  Expect(IsHanToken("\xF0\xB1\x8D\x8A"), "U+3134A Extension G");
  Expect(IsHanToken(" \xE6\xBC\xA2\xE5\xAD\x97"), "space + 漢字");
  Expect(IsHanToken("\xEB\xB2\x95\xE5\xBE\x8B"), "법 + 律");
  Expect(IsHanToken("a\xE5\xAD\x97"
                    "b"),
         "ASCII around 字");
  Expect(IsHanToken("\xB8\xAD\xE5\xAD\x97"), "continuation tail + 字");

  // Partial byte-level and byte-fallback tokens that can only start Han.
  for (const char lead : {'\xE4', '\xE5', '\xE6', '\xE7', '\xE8', '\xE9'})
    Expect(IsHanToken(std::string(1, lead)), "lone Han lead byte");
  Expect(IsHanToken("\xE4\xB8"), "E4 B8 prefix");
  Expect(IsHanToken("\xE3\x90"), "E3 90 prefix (Extension A)");
  Expect(IsHanToken("\xE3\xBF"), "E3 BF prefix (Extension A)");
  Expect(IsHanToken("\xEF\xA4"), "EF A4 prefix (Compatibility)");
  Expect(IsHanToken("\xEF\xAB"), "EF AB prefix (Compatibility)");
  Expect(IsHanToken("\xF0\xA0"), "F0 A0 prefix (Extension B)");
  Expect(IsHanToken("\xF0\xB1\x8D"), "F0 B1 8D prefix (Extension G)");
  Expect(IsHanToken("\xEB\x8A\x94\xE4"), "는 + trailing Han lead");

  // Hangul syllables, jamo and their partial bytes.
  Expect(!IsHanToken("\xEA\xB0\x80"), "U+AC00 가");
  Expect(!IsHanToken("\xED\x9E\xA3"), "U+D7A3 힣");
  Expect(!IsHanToken("\xED\x95\x9C\xEA\xB5\xAD\xEC\x96\xB4"), "한국어");
  Expect(!IsHanToken("\xE1\x84\x80"), "U+1100 Hangul Jamo");
  Expect(!IsHanToken("\xE3\x84\xB1"), "U+3131 compatibility jamo");
  for (const char lead : {'\xEA', '\xEB', '\xEC', '\xED', '\xE1'})
    Expect(!IsHanToken(std::string(1, lead)), "lone Hangul lead byte");
  Expect(!IsHanToken("\xEA\xB0"), "EA B0 prefix");
  Expect(!IsHanToken("\xED\x9E"), "ED 9E prefix");
  Expect(!IsHanToken("\xB0\x80"), "Hangul continuation tail");
  Expect(!IsHanToken("\xB8\xAD"), "Han continuation tail alone");

  // Kana, CJK punctuation, fullwidth forms and neighbours of the ranges.
  Expect(!IsHanToken("\xE3\x81\x82"), "U+3042 hiragana");
  Expect(!IsHanToken("\xE3\x82\xA2"), "U+30A2 katakana");
  Expect(!IsHanToken("\xE3\x80\x82"), "U+3002 ideographic full stop");
  Expect(!IsHanToken("\xE3\x80\x85"), "U+3005 iteration mark");
  Expect(!IsHanToken("\xE3\x80\x8C"), "U+300C corner bracket");
  Expect(!IsHanToken("\xEF\xBC\x8C"), "U+FF0C fullwidth comma");
  Expect(!IsHanToken("\xEF\xBC\xA1"), "U+FF21 fullwidth A");
  Expect(!IsHanToken("\xE3\x8F\xBF"), "U+33FF just below Extension A");
  Expect(!IsHanToken("\xE4\xB7\x80"), "U+4DC0 hexagram");
  Expect(!IsHanToken("\xEF\xAC\x80"), "U+FB00 just above Compatibility");
  Expect(!IsHanToken("\xE2\xBC\x80"), "U+2F00 Kangxi radical");
  Expect(!IsHanToken("\xF0\x9F\x98\x80"), "emoji");
  Expect(!IsHanToken("\xE3"), "lone E3 (kana, punctuation or Ext A)");
  Expect(!IsHanToken("\xE3\x81"), "E3 81 prefix (kana)");
  Expect(!IsHanToken("\xE3\x80"), "E3 80 prefix (punctuation)");
  Expect(!IsHanToken("\xEF"), "lone EF");
  Expect(!IsHanToken("\xEF\xBC"), "EF BC prefix (fullwidth)");
  Expect(!IsHanToken("\xE4\xB7"), "E4 B7 prefix (hexagrams)");
  Expect(!IsHanToken("\xF0"), "lone F0");
  Expect(!IsHanToken("\xF0\x9F"), "F0 9F prefix (emoji)");
  Expect(!IsHanToken(""), "empty token");
  Expect(!IsHanToken("hello"), "ASCII");
  Expect(!IsHanToken("<|im_start|>"), "control text");
  // A malformed lead is skipped; a later Han character still counts.
  Expect(!IsHanToken("\xE4"
                     "a"),
         "Han lead followed by ASCII");
  Expect(IsHanToken("\xE4"
                    "a\xE5\xAD\x97"),
         "malformed lead then 字");
}

std::shared_ptr<const TokenSet> Set(std::size_t size,
                                    std::vector<TokenId> members) {
  std::vector<std::string> text(size, "a");
  for (const auto token : members)
    text[token] = "\xE5\xAD\x97";
  return BuildHanTokens(size, [&](TokenId token) { return text[token]; });
}

void TestBuild() {
  const auto set = Set(70, {0, 31, 32, 69});
  Expect(set->count == 4 && set->words.size() == 3, "set size");
  for (TokenId token = 0; token < 72; ++token)
    Expect(set->Contains(token) ==
               (token == 0 || token == 31 || token == 32 || token == 69),
           "set membership");
}

SamplingConfig Config(float scale, float temperature,
                      std::shared_ptr<const TokenSet> set) {
  SamplingConfig config;
  config.temperature = temperature;
  config.han_scale = scale;
  config.han_tokens = std::move(set);
  return config;
}

void TestSampler() {
  // Token 1 is Han and leads token 0 by 0.5 nats.
  const std::vector<float> logits{1.0F, 1.5F, -1.0F, 0.0F};
  const auto set = Set(logits.size(), {1});
  {
    SamplerState off(Config(1.0F, 0.0F, set));
    Expect(off.Sample(logits) == 1, "scale 1 keeps the Han argmax");
    Expect(!off.config().penalties_enabled(), "scale 1 is inactive");
    Expect(off.config().can_use_unmodified_argmax(), "scale 1 fast path");
    SamplerState no_set(Config(0.2F, 0.0F, nullptr));
    Expect(no_set.Sample(logits) == 1, "a null set leaves the scale inactive");
  }
  {
    SamplerState mild(Config(0.7F, 0.0F, set));  // ln 0.7 = -0.36 > -0.5
    Expect(mild.Sample(logits) == 1, "a small offset keeps the argmax");
    SamplerState strong(Config(0.5F, 0.0F, set));  // ln 0.5 = -0.69
    Expect(strong.config().penalties_enabled(), "scale < 1 adjusts logits");
    Expect(!strong.config().can_use_unmodified_argmax(), "no raw argmax");
    Expect(!strong.CanSelectArgmax(1), "a raw argmax is not the winner");
    Expect(strong.CanSelectArgmax(0, /*penalties_applied=*/true),
           "an adjusted argmax is");
    Expect(strong.Sample(logits) == 0, "greedy uses the scaled logit");
    Near(strong.config().han_bias(), std::log(0.5), "greedy offset is ln x");
  }
  for (const float temperature : {0.7F, 1.0F, 1.3F}) {
    // A fixed ln(x) logit offset: probability times x^(1/T), renormalized.
    const double x = std::pow(static_cast<double>(0.3F),  // the float scale
                              1.0 / static_cast<double>(temperature));
    const auto plain =
        SamplerState(Config(1.0F, temperature, set)).Distribution(logits);
    const auto scaled =
        SamplerState(Config(0.3F, temperature, set)).Distribution(logits);
    Near(Config(0.3F, temperature, set).han_bias(),
         std::log(static_cast<double>(0.3F)),
         "the offset does not depend on temperature");
    const double mass = 1 - plain.probability(1) * (1 - x);
    for (TokenId token = 0; token < logits.size(); ++token)
      Near(scaled.probability(token),
           plain.probability(token) * (token == 1 ? x : 1.0) / mass,
           "full-row probability times x^(1/T)");
    Near(scaled.probability(1) / scaled.probability(0),
         x * plain.probability(1) / plain.probability(0), "odds times x^(1/T)");
  }
  {
    // Top-k sees the scaled ranking; penalties apply first.
    auto config = Config(0.5F, 1.0F, set);
    config.top_k = 2;
    const auto distribution = SamplerState(config).Distribution(logits);
    Expect(distribution.probability(2) == 0, "top-k drops the weakest");
    Expect(distribution.probability(1) > 0 && distribution.probability(3) == 0,
           "ln 0.5 keeps token 1 above token 3");
    config.han_scale = 0.05F;  // ln 0.05 = -3.0 puts token 1 below token 3
    const auto strong = SamplerState(config).Distribution(logits);
    Expect(strong.probability(1) == 0 && strong.probability(3) > 0,
           "top-k ranks the scaled logit");
    // The compact target path agrees with the full row.
    const std::vector<LogitCandidate> candidates{
        {0, 1.0F}, {1, 1.5F}, {2, -1.0F}, {3, 0.0F}};
    const auto compact =
        SamplerState(config).DistributionFromCandidates(candidates);
    for (TokenId token = 0; token < logits.size(); ++token)
      Near(compact.probability(token), strong.probability(token),
           "compact candidates match the full row");
  }
  {
    // Compact draft rows look up Han membership through their token IDs.
    const auto config = Config(0.3F, 1.0F, set);
    const std::vector<float> compact{1.5F, 1.0F};
    const std::vector<TokenId> ids{1, 0};
    const auto distribution = SamplerState(config).Distribution(compact, ids);
    const auto full = SamplerState(config).Distribution(logits);
    Near(distribution.probability(0) / distribution.probability(1),
         full.probability(1) / full.probability(0), "compact IDs are mapped");
  }
  {
    // Penalties come first: presence lowers token 0, then token 1 is Han.
    auto config = Config(0.5F, 0.0F, set);
    config.presence_penalty = 0.4F;
    SamplerState sampler(config);
    sampler.Accept(TokenId{0});
    // 1.0 - 0.4 = 0.6 against 1.5 - 0.69 = 0.81
    Expect(sampler.Sample(logits) == 1, "penalties and Han offset combine");
  }
  for (const float bad : {0.0F, -0.5F, 1.5F, NAN, INFINITY}) {
    bool threw = false;
    try {
      Config(bad, 0.0F, set).Validate();
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    Expect(threw, "Han scale outside (0, 1] is rejected");
  }
}

}  // namespace

int main() {
  TestClassifier();
  TestBuild();
  TestSampler();
  std::cout << "han_script_test passed\n";
  return 0;
}
