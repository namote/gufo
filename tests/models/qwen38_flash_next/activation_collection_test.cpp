// The in-server collector's record rules and readout
// (activation_collection.hpp) against collect.cpp's definitions.
#include "src/models/qwen38_flash_next/activation_collection.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/json.hpp"

namespace q = gufo::models::qwen38_flash_next;
namespace json = gufo::json;

namespace {
void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "FAILED: " << what << '\n';
    std::exit(1);
  }
}

// The validation message for `line` in `mode`, or "" when it passes.
std::string Refusal(q::CollectionMode mode, const std::string& line,
                    std::string split = {}) {
  try {
    q::ValidateCollectionRecord(mode, json::parse(line), &split);
    return {};
  } catch (const std::invalid_argument& e) {
    return e.what();
  }
}

void TestModes() {
  Expect(
      q::ParseCollectionMode("collect") == q::CollectionMode::kCollect &&
          q::ParseCollectionMode("answers") == q::CollectionMode::kAnswers &&
          q::ParseCollectionMode("segments") == q::CollectionMode::kSegments &&
          !q::ParseCollectionMode("evaluate"),
      "modes: collect, answers and segments only");
  Expect(q::CollectionContext(q::CollectionMode::kCollect) == 2048 &&
             q::CollectionContext(q::CollectionMode::kAnswers) == 8192 &&
             q::CollectionContext(q::CollectionMode::kSegments) == 24576,
         "collect.cpp's contexts");
  Expect(q::CollectionReadoutName(q::CollectionMode::kAnswers) ==
             "answer-tokens-mean-post-layer-hc-mean-fp32-v1",
         "fit_steering.py's READOUT");
  Expect(q::CollectionReadoutName(q::CollectionMode::kSegments) ==
             "selected-segment-end-rows-mean-post-layer-hc-mean-fp32-v1",
         "fit_steering.py's SEGMENT_READOUT");
}

void TestValidation() {
  using M = q::CollectionMode;
  const std::string collect =
      R"({"id":"a1","split":"collection","group":"refusal","thinking":true,)"
      R"("text":"x"})";
  Expect(Refusal(M::kCollect, collect).empty(), "a collect record passes");
  Expect(Refusal(M::kCollect, R"({"id":"a1","group":"x","thinking":true})") ==
             "invalid record class",
         "collect groups are refusal|ordinary");
  Expect(Refusal(M::kCollect, R"({"id":"a1","group":"ordinary"})") ==
             "missing thinking mode",
         "collect records need thinking");
  Expect(Refusal(M::kCollect,
                 R"({"id":"A1","group":"ordinary","thinking":false})") ==
             "invalid record ID",
         "ID characters are collect.cpp's");
  Expect(Refusal(M::kCollect, collect, "train") == "mixed dataset splits",
         "one split per dataset");
  Expect(Refusal(M::kCollect, "[1]") == "a record is a JSON object",
         "records are objects");

  const std::string answer =
      R"({"id":"b2","group":"own","pair":"p","text":"q","answer":"a"})";
  Expect(Refusal(M::kAnswers, answer).empty(), "an answers record passes");
  for (const auto* line :
       {R"({"id":"b2","group":"ordinary","pair":"p","text":"q","answer":"a"})",
        R"({"id":"b2","group":"own","text":"q","answer":"a"})",
        R"({"id":"b2","group":"own","pair":"p","answer":"a"})",
        R"({"id":"b2","group":"own","pair":"p","text":"q"})"})
    Expect(Refusal(M::kAnswers, line) ==
               "answers need group exemplar|own, pair, text and answer",
           std::string("answers refuse ") + line);

  const std::string segments =
      R"({"id":"c3","group":"exemplar","pair":"p","text":"q",)"
      R"("segments":["a","b","c"],"select":[0,1]})";
  Expect(Refusal(M::kSegments, segments).empty(), "a segments record passes");
  const auto shape = "segments need a segments array (>= 2) and a select array";
  Expect(
      Refusal(M::kSegments, R"({"id":"c3","group":"own","pair":"p","text":"q",)"
                            R"("segments":["a"],"select":[0]})") == shape,
      "two segments at least");
  Expect(
      Refusal(M::kSegments, R"({"id":"c3","group":"own","pair":"p","text":"q",)"
                            R"("segments":["a","b"],"select":[]})") == shape,
      "one selection at least");
  Expect(
      Refusal(M::kSegments, R"({"id":"c3","group":"own","pair":"p","text":"q",)"
                            R"("segments":["a",""],"select":[0]})") ==
          "segments must be non-empty strings",
      "segments are non-empty strings");
  Expect(
      Refusal(M::kSegments, R"({"id":"c3","group":"own","pair":"p","text":"q",)"
                            R"("segments":["a","b"],"select":[1]})") ==
          "select indexes must name a segment followed by another",
      "the last segment cannot be selected");
  Expect(
      Refusal(M::kSegments, R"({"id":"c3","group":"own","pair":"p","text":"q",)"
                            R"("segments":["a","b"],"select":[0.5]})") ==
          "select indexes must name a segment followed by another",
      "selections are whole indexes");
}

void TestFeeds() {
  q::CollectionPlan plan;
  plan.prompt = {1, 2, 3};
  plan.answer = {7, 8};
  const auto collect = q::CollectionFeeds(q::CollectionMode::kCollect, plan);
  Expect(collect.size() == 1 && collect[0].observed &&
             collect[0].tokens == plan.prompt,
         "collect feeds the prompt, observed");
  const auto answers = q::CollectionFeeds(q::CollectionMode::kAnswers, plan);
  Expect(answers.size() == 2 && !answers[0].observed &&
             answers[0].tokens == std::vector<std::int32_t>{1, 2} &&
             answers[1].observed &&
             answers[1].tokens == std::vector<std::int32_t>{1, 2, 3, 7},
         "answers feed the prompt but its last token, then the answer rows");
}

// Rows of `hc` streams over hidden 2: stream c of row r holds base + c.
std::vector<float> Rows(std::size_t rows, float base) {
  std::vector<float> values;
  for (std::size_t r = 0; r < rows; ++r)
    for (int c = 0; c < 2; ++c)
      for (int h = 0; h < 2; ++h)
        values.push_back(base + float(r * 10 + c + h * 100));
  return values;
}

void TestReadout() {
  // Collect: the last observed pass's row, the mean of the HC streams.
  q::CollectionPlan plan;
  plan.prompt = {1, 2, 3};
  q::CollectionReadout collect(q::CollectionMode::kCollect, 2, 2, 2);
  collect.Begin(plan);
  for (std::uint32_t layer = 0; layer < 2; ++layer)
    collect.Observe(layer, Rows(1, 0));
  for (std::uint32_t layer = 0; layer < 2; ++layer)
    collect.Observe(layer, Rows(1, float(layer)));
  const auto last = collect.Finish(plan);
  Expect(last == std::vector<float>{0.5F, 100.5F, 1.5F, 101.5F},
         "collect reads the last pass's HC mean");
  collect.Begin(plan);
  collect.Observe(0, Rows(1, 0));
  bool missing = false;
  try {
    (void)collect.Finish(plan);
  } catch (const std::runtime_error&) {
    missing = true;
  }
  Expect(missing, "collect needs every layer");
  collect.Begin(plan);
  collect.Observe(0, Rows(2, 0));
  collect.Observe(1, Rows(1, 0));
  bool layout = false;
  try {
    (void)collect.Finish(plan);
  } catch (const std::runtime_error& e) {
    layout = std::string(e.what()) == "collector HC layout mismatch";
  }
  Expect(layout, "collect observes one row per pass");

  // Answers: the mean over every observed row; unobserved passes are not
  // counted.
  plan.answer = {7, 8, 9};
  q::CollectionReadout answers(q::CollectionMode::kAnswers, 1, 2, 2);
  answers.Begin(plan);
  answers.Observe(0, Rows(4, 1000));  // prompt without its last token: unread
  answers.SetObserving(true);
  answers.Observe(0, Rows(2, 0));
  answers.Observe(0, Rows(1, 20));
  answers.SetObserving(false);
  const auto mean = answers.Finish(plan);
  // Rows 0, 1 and 2 (base 20, r 0): stream means 0.5, 10.5, 20.5.
  Expect(mean == std::vector<float>{10.5F, 110.5F}, "answers average rows");
  answers.Begin(plan);
  answers.SetObserving(true);
  answers.Observe(0, Rows(2, 0));
  bool short_rows = false;
  try {
    (void)answers.Finish(plan);
  } catch (const std::runtime_error&) {
    short_rows = true;
  }
  Expect(short_rows, "answers need one row per answer token");

  // Segments: only the selected rows, divided by their count.
  plan.selected.assign(3, true);
  plan.selected[0] = false;
  plan.selected_rows = 2;
  q::CollectionReadout segments(q::CollectionMode::kSegments, 1, 2, 2);
  segments.Begin(plan);
  segments.SetObserving(true);
  segments.Observe(0, Rows(3, 0));
  Expect(segments.Finish(plan) == std::vector<float>{15.5F, 115.5F},
         "segments average the selected rows");
}
}  // namespace

int main() {
  TestModes();
  TestValidation();
  TestFeeds();
  TestReadout();
  std::cout << "activation collection: ok\n";
  return 0;
}
