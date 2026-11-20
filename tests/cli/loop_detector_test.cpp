#include "src/cli/serve/loop_detector.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {
using gufo::server::GenerationLoopMonitor;
using gufo::server::LineRepeatDetector;
using gufo::server::TokenLoopDetector;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

// Deterministic pseudo-random tokens in [0, vocab).
std::vector<std::uint32_t> Random(std::size_t count, std::uint32_t vocab,
                                  std::uint64_t seed) {
  std::vector<std::uint32_t> tokens;
  for (std::size_t i = 0; i < count; ++i) {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    tokens.push_back(static_cast<std::uint32_t>((seed >> 33) % vocab));
  }
  return tokens;
}

// The first detection over `tokens`, or nullopt.
std::optional<TokenLoopDetector::Detection> FirstDetection(
    TokenLoopDetector& detector, const std::vector<std::uint32_t>& tokens) {
  for (const auto token : tokens)
    if (const auto detection = detector.Feed(token))
      return detection;
  return std::nullopt;
}

// Exact repetition of a varied unit is a loop after 4 periods and 1024
// tokens, with the unit's period; it latches.
void TestExactLoop() {
  const auto unit = Random(300, 5000, 1);
  std::vector<std::uint32_t> tokens;
  for (int copy = 0; copy < 8; ++copy)
    tokens.insert(tokens.end(), unit.begin(), unit.end());
  TokenLoopDetector detector;
  const auto detection = FirstDetection(detector, tokens);
  Expect(detection.has_value(), "an exact loop is detected");
  Expect(detection->period == 300, "the loop's period is the unit");
  Expect(detection->position >= 4 * 300 && detection->position < 6 * 300,
         "an exact loop needs four periods and 1024 tokens");
  Expect(detection->repeats >= 4, "the detection reports its repeats");
  Expect(detector.detected() && !detector.Feed(unit[0]), "a detection latches");
  detector.Reset();
  Expect(!detector.detected(), "reset clears the detection");
}

// A short exact unit still needs 1024 tokens.
void TestShortUnitNeedsLength() {
  const auto unit = Random(20, 5000, 2);
  std::vector<std::uint32_t> tokens;
  for (int copy = 0; copy < 100; ++copy)
    tokens.insert(tokens.end(), unit.begin(), unit.end());
  TokenLoopDetector detector;
  const auto detection = FirstDetection(detector, tokens);
  Expect(detection.has_value() && detection->period == 20 &&
             detection->position >= 1024 && detection->position < 1100,
         "a short exact loop is reported once it spans 1024 tokens");
}

// Text without repetition is never a loop.
void TestVariedText() {
  TokenLoopDetector detector;
  Expect(!FirstDetection(detector, Random(40000, 50000, 3)).has_value(),
         "varied text is not a loop");
}

// Over a tiny vocabulary k-grams recur by structure alone: random text over
// two or four tokens is no loop, while exact repetition still is.
void TestLowDiversity() {
  TokenLoopDetector binary;
  Expect(!FirstDetection(binary, Random(20000, 2, 4)).has_value(),
         "random binary text is not a loop");
  TokenLoopDetector random_detector;
  Expect(!FirstDetection(random_detector, Random(20000, 4, 9)).has_value(),
         "random tiny-vocabulary text is not a loop");
  const auto unit = Random(200, 4, 5);
  std::vector<std::uint32_t> tokens;
  for (int copy = 0; copy < 10; ++copy)
    tokens.insert(tokens.end(), unit.begin(), unit.end());
  TokenLoopDetector exact_detector;
  Expect(FirstDetection(exact_detector, tokens).has_value(),
         "exact tiny-vocabulary repetition is a loop");
}

// Near repetition (one edited token per copy) needs 8 periods and 8192
// tokens: a long unit reaches it, a list of a few edited items does not.
void TestNearRepetition() {
  const auto unit = Random(1200, 5000, 6);
  std::vector<std::uint32_t> tokens;
  for (std::uint32_t copy = 0; copy < 12; ++copy) {
    auto edited = unit;
    for (std::size_t i = 0; i < edited.size(); i += 100)
      edited[i] = 6000 + copy;  // an index that differs between copies
    tokens.insert(tokens.end(), edited.begin(), edited.end());
  }
  TokenLoopDetector detector;
  const auto detection = FirstDetection(detector, tokens);
  Expect(detection.has_value() && detection->position >= 8192 &&
             detection->position >= 8 * 1200 - 1 && detection->period >= 1140 &&
             detection->period <= 1260,
         "near repetition is a loop after 8 periods and 8192 tokens");
  TokenLoopDetector short_list;
  std::vector<std::uint32_t> items(tokens.begin(), tokens.begin() + 6 * 1200);
  Expect(!FirstDetection(short_list, items).has_value(),
         "six near copies are not a loop");
}

// Long thinking lines seen three times are a loop; short lines, lines in a
// code fence and two sightings are not.
void TestLineRule() {
  const std::string line =
      "  Let me reconsider whether the stairs were really the problem.  \n";
  LineRepeatDetector detector;
  Expect(!detector.Feed(line) && !detector.Feed("Something else.\n") &&
             !detector.Feed(line.substr(0, 10)) &&
             !detector.Feed(line.substr(10)),
         "two sightings, split across pieces, are not a loop");
  Expect(detector.Feed("\t" + line), "the third sighting is a loop");
  Expect(detector.detected() && !detector.Feed(line), "detection latches");

  LineRepeatDetector fenced;
  Expect(!fenced.Feed("```cpp\n" + line + line + line + line + "```\n"),
         "fenced lines are not counted");
  Expect(!fenced.Feed(line + line), "counting resumes after the fence");
  Expect(fenced.Feed(line), "lines after the fence count");

  LineRepeatDetector short_lines;
  const std::string short_line = "ok ok ok ok ok ok ok ok ok ok ok ok ok\n";
  Expect(!short_lines.Feed(short_line + short_line + short_line + short_line),
         "lines under 40 non-space characters are not counted");
  // Characters, not bytes: 39 Hangul syllables are 117 bytes.
  std::string hangul;
  for (int i = 0; i < 39; ++i)
    hangul += "가";
  hangul += "\n";
  Expect(!short_lines.Feed(hangul + hangul + hangul),
         "the minimum counts characters");
}

// The monitor applies the line rule only while reasoning, reports the phase,
// reports once per phase and starts fresh after the reasoning end.
void TestMonitor() {
  constexpr std::uint32_t kEnd = 99999;
  const std::string line =
      "I should check the review again for the actual complaint here.\n";
  GenerationLoopMonitor monitor({}, true, {kEnd});
  std::optional<gufo::server::LoopDetection> found;
  for (std::uint32_t i = 0; i < 3 && !found; ++i)
    found = monitor.Observe(i, line);
  Expect(
      found && found->reasoning && found->rule == "line" && found->tokens == 3,
      "a repeated thinking line is reported while reasoning");
  Expect(!monitor.Observe(7, line), "a phase reports once");
  Expect(!monitor.Observe(kEnd, "</think>") && !monitor.reasoning(),
         "the reasoning end starts the answer");
  for (std::uint32_t i = 0; i < 6; ++i)
    Expect(!monitor.Observe(100 + i, line),
           "the line rule does not apply to the answer");
  const auto unit = Random(64, 5000, 7);
  found.reset();
  for (int copy = 0; copy < 40 && !found; ++copy)
    for (const auto token : unit)
      if ((found = monitor.Observe(token, "x")))
        break;
  Expect(found && !found->reasoning && found->rule == "period" &&
             found->period == 64,
         "an answer loop is reported with its period");

  // Without a reasoning phase the reply is all answer.
  GenerationLoopMonitor answer({}, false, {kEnd});
  for (int i = 0; i < 5; ++i)
    Expect(!answer.Observe(static_cast<std::uint32_t>(i), line),
           "the line rule needs reasoning");
}

// Transcribing a prompt's distinct items in a repeated frame is not a loop;
// the same output without that prompt is one (near repetition over 8192
// tokens).
void TestPromptTranscription() {
  const auto frame = Random(40, 5000, 8);
  std::vector<std::uint32_t> output;
  for (int i = 0; i < 200; ++i) {
    const auto item = Random(8, 5000, 100 + i);
    output.insert(output.end(), frame.begin(), frame.end());
    output.insert(output.end(), item.begin(), item.end());
  }
  TokenLoopDetector unprimed;
  Expect(FirstDetection(unprimed, output).has_value(),
         "a frame repeated around generated items is a loop");
  TokenLoopDetector primed;
  primed.Prime(output);  // the prompt lists the same formatted items
  Expect(!FirstDetection(primed, output).has_value(),
         "transcribing distinct prompt items is not a loop");
}
}  // namespace

int main() {
  TestExactLoop();
  TestShortUnitNeedsLength();
  TestVariedText();
  TestLowDiversity();
  TestNearRepetition();
  TestLineRule();
  TestMonitor();
  TestPromptTranscription();
  std::cout << "All loop detector tests passed.\n";
  return 0;
}
