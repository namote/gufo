// CPU contracts of dynamic abliteration (R2 A1): which committed rows count,
// the threshold and margin rule, sticky single-profile engagement, the
// report, and the cache identities of a state that switched mid-request.

#include "src/models/qwen38_flash_next/dynamic_abliteration.hpp"

#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace q = gufo::models::qwen38_flash_next;
namespace qwen = gufo::models::qwen;

namespace {
void Require(bool ok, const char* what) {
  if (!ok)
    throw std::runtime_error(std::string("dynamic abliteration: ") + what);
}

std::shared_ptr<const qwen::AbliterationSettings> Settings(
    const std::string& profile, const std::string& identity) {
  auto settings = std::make_shared<qwen::AbliterationSettings>();
  settings->profile = profile;
  settings->attention = {0, 0};
  settings->mlp = {0, 0};
  settings->residual = {1, 1};
  settings->identity = identity;
  return settings;
}

// Two candidates: "pol" in column 1 (threshold 1), "fnx" in column 0
// (threshold 2).
q::DynamicAbliteration Make() {
  return q::DynamicAbliteration({
      {"pol", 1, 1.0F, Settings("pol", std::string(64, 'a'))},
      {"fnx", 0, 2.0F, Settings("fnx", std::string(64, 'b'))},
  });
}

// Rows of (fnx, pol) scores laid out in kDetectorColumns columns.
std::vector<float> Rows(std::initializer_list<std::array<float, 2>> rows) {
  std::vector<float> out;
  for (const auto& row : rows) {
    std::vector<float> wide(q::kDetectorColumns, 0.0F);
    wide[0] = row[0];
    wide[1] = row[1];
    out.insert(out.end(), wide.begin(), wide.end());
  }
  return out;
}

void PrefillTrigger() {
  auto dynamic = Make();
  // An intermediate prefill chunk is replaced by the final one; Observe only
  // sees the pass the session kept.
  const auto scores = Rows({{0.5F, 1.5F}});
  dynamic.Observe({.id = 7,
                   .first_position = 99,
                   .rows = 1,
                   .prefill = true,
                   .scores = scores},
                  100);
  const auto settings = dynamic.Engage(100);
  Require(
      settings && settings->profile == "pol" && settings->dynamic_from == 100u,
      "the last prompt row turns the crossing profile ON from generation");
  Require(settings->identity != std::string(64, 'a'),
          "a dynamic switch has its own identity");
  Require(dynamic.engaged() && !dynamic.Engage(101), "sticky, engaged once");
  // Later rows are no longer scored: even a larger fnx score changes nothing.
  const auto later = Rows({{9.0F, 0.0F}});
  dynamic.Observe({.id = 8,
                   .first_position = 100,
                   .rows = 1,
                   .prefill = false,
                   .scores = later},
                  101);
  const auto report = dynamic.Report();
  Require(report.triggered == std::vector<std::string>{"pol"} &&
              report.position == 100u && report.phase == "prefill",
          "prefill report");
  Require(report.scores.size() == 2 && report.scores[0].first == "pol" &&
              report.scores[0].second == 1.5F &&
              report.scores[1].second == 0.5F,
          "scores of the crossing row");
  Require(report.max_scores[0].second == 1.5F, "max excludes later passes");
}

void DecodeTriggerIgnoresRolledBackRows() {
  auto dynamic = Make();
  const auto prompt = Rows({{0.0F, 0.2F}});
  dynamic.Observe({.id = 1,
                   .first_position = 9,
                   .rows = 1,
                   .prefill = true,
                   .scores = prompt},
                  10);
  Require(!dynamic.Engage(10), "below threshold stays OFF");
  // A verification pass of four rows at 10..13; the session kept two (12
  // committed), so the crossing at row 3 (position 13) does not count.
  const auto verify =
      Rows({{0.0F, 0.1F}, {0.0F, 0.3F}, {0.0F, 0.4F}, {5.0F, 5.0F}});
  dynamic.Observe({.id = 2,
                   .first_position = 10,
                   .rows = 4,
                   .prefill = false,
                   .scores = verify},
                  12);
  Require(!dynamic.Engage(12), "rolled-back rows never trigger");
  // The same pass observed again is ignored.
  dynamic.Observe({.id = 2,
                   .first_position = 10,
                   .rows = 4,
                   .prefill = false,
                   .scores = verify},
                  14);
  Require(!dynamic.Engage(14), "a pass is observed once");
  // Both cross on one row: the larger margin wins (fnx 3.5 - 2 > pol 2 - 1).
  const auto both = Rows({{3.5F, 2.0F}});
  dynamic.Observe({.id = 3,
                   .first_position = 14,
                   .rows = 1,
                   .prefill = false,
                   .scores = both},
                  15);
  const auto settings = dynamic.Engage(15);
  Require(
      settings && settings->profile == "fnx" && settings->dynamic_from == 15u,
      "largest margin, ON from the next token");
  const auto report = dynamic.Report();
  Require(report.phase == "decode" && report.position == 15u &&
              report.triggered == std::vector<std::string>{"fnx"},
          "decode report");
  Require(report.max_scores[0].second == 2.0F &&
              report.max_scores[1].second == 3.5F,
          "max over committed rows only");
}

void ReportWithoutTrigger() {
  auto dynamic = Make();
  Require(dynamic.Report().scores.empty() &&
              dynamic.Report().triggered.empty() && !dynamic.Report().position,
          "empty before any pass");
  const auto prompt = Rows({{1.0F, 0.5F}});
  dynamic.Observe({.id = 4,
                   .first_position = 0,
                   .rows = 1,
                   .prefill = true,
                   .scores = prompt},
                  1);
  const auto decode = Rows({{1.5F, 0.9F}});
  dynamic.Observe({.id = 5,
                   .first_position = 1,
                   .rows = 1,
                   .prefill = false,
                   .scores = decode},
                  2);
  const auto report = dynamic.Report();
  Require(report.triggered.empty() && !report.position && report.phase.empty(),
          "OFF report");
  Require(report.scores[0].second == 0.5F && report.scores[1].second == 1.0F,
          "OFF scores are the last prompt row's");
  Require(report.max_scores[0].second == 0.9F &&
              report.max_scores[1].second == 1.5F,
          "OFF max over every committed row");
  // An equal score does not cross.
  const auto equal = Rows({{2.0F, 1.0F}});
  dynamic.Observe({.id = 6,
                   .first_position = 2,
                   .rows = 1,
                   .prefill = false,
                   .scores = equal},
                  3);
  Require(!dynamic.Engage(3), "the threshold must be exceeded");
}

void Identities() {
  const auto base = Settings("pol", std::string(64, 'a'));
  const auto at10 = qwen::DynamicAbliterationSettings(*base, 10);
  const auto at11 = qwen::DynamicAbliterationSettings(*base, 11);
  Require(at10->identity != at11->identity &&
              at10->identity != base->identity &&
              at10->identity ==
                  qwen::DynamicAbliterationSettings(*base, 10)->identity,
          "dynamic identity binds settings and switch point");
  Require(at10->residual == base->residual &&
              at10->DeviceWords() == base->DeviceWords(),
          "the same projection");
  bool nested = false;
  try {
    (void)qwen::DynamicAbliterationSettings(*at10, 12);
  } catch (const std::logic_error&) {
    nested = true;
  }
  Require(nested, "a dynamic switch is not switched again");
  // States up to the switch point are OFF states.
  Require(qwen::AbliterationForState(at10.get(), 9) == nullptr &&
              qwen::AbliterationForState(at10.get(), 10) == nullptr &&
              qwen::AbliterationForState(at10.get(), 11) == at10.get() &&
              qwen::AbliterationForState(base.get(), 0) == base.get() &&
              qwen::AbliterationForState(nullptr, 5) == nullptr,
          "state settings by length");
  // Component-cache records: the prompt checkpoint of a switched request is
  // an OFF record; a later state matches only the same switch.
  const std::span<const std::uint8_t> none{};
  const auto prompt = qwen::RecordAbliterationContinuation(
      none, qwen::AbliterationForState(at10.get(), 10));
  const auto tail = qwen::RecordAbliterationContinuation(
      none, qwen::AbliterationForState(at10.get(), 12));
  const auto match = [&](const qwen::AbliterationContinuationRecord& r,
                         const qwen::AbliterationSettings* active) {
    return qwen::AbliterationContinuationMatches(r.mode, 0, r.image, none,
                                                 active);
  };
  Require(match(prompt, nullptr), "prompt checkpoint restores into OFF");
  Require(!match(tail, nullptr) && !match(tail, base.get()) &&
              !match(tail, at11.get()) && match(tail, at10.get()),
          "a switched tail matches only its own switch");
}
}  // namespace

int main() {
  try {
    PrefillTrigger();
    DecodeTriggerIgnoresRolledBackRows();
    ReportWithoutTrigger();
    Identities();
    bool rejected = false;
    try {
      q::DynamicAbliteration({});
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    Require(rejected, "no candidates rejected");
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  std::cout << "dynamic abliteration PASS\n";
  return 0;
}
