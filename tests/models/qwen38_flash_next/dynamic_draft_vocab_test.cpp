// Frequency-ranked dynamic draft vocabulary policy (V2-7, exllamav3 47d42fe).
#include "src/models/qwen38_flash_next/dynamic_draft_vocab.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace qfn = gufo::models::qwen38_flash_next;
using Frequency = qfn::MtpDraftFrequency;
using Switch = qfn::MtpDraftSubsetSwitch;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::vector<std::int32_t> Repeat(std::int32_t id, std::size_t count) {
  return std::vector<std::int32_t>(count, id);
}

// Prompt tokens weigh 0.2 and rebuild once per prompt; generated tokens
// rebuild every 128.
void CheckWeightsAndRebuild() {
  Frequency f(4, 1000, 1.0F);
  Require(f.Observe(Repeat(1, 1000), true), "a prompt did not rebuild");
  Require(std::fabs(f.Count(1) - 200.0F) < 1e-2F, "prompt weight is not 0.2");
  Require(f.Contains(1) && f.Version() == 1, "the prompt token is no member");
  Frequency decayed(4, 1000);
  decayed.Observe(Repeat(5, 1000), true);
  Require(std::fabs(decayed.Count(5) - 200.0F * 0.98F) < 1e-2F,
          "a long prompt decayed more than one step");
  Require(!f.Observe(Repeat(2, 3), false), "rebuilt before 128 tokens");
  Require(f.Count(2) == 3.0F, "generated weight is not 1");
  Require(!f.Contains(2), "a member before its rebuild");
  Require(f.Observe(Repeat(3, 125), false), "no rebuild at 128 tokens");
  Require(f.Contains(1) && f.Contains(2) && f.Contains(3),
          "counted tokens are not members");
  Require(!f.Contains(4) && !f.Contains(-1) && !f.Contains(5000),
          "an uncounted token is a member");
  Require(f.Version() == 2, "the rebuild did not change the version");
  // Out-of-range IDs are ignored.
  f.Observe(std::vector<std::int32_t>{-5, 1000, 2000}, false);
  Require(f.Count(-5) == 0.0F && f.Count(1000) == 0.0F, "a bad ID counted");
}

// A full subset swaps a member only for a candidate counted 25% higher, and
// counts decay at each rebuild.
void CheckHysteresis() {
  Frequency f(2, 100, 1.0F);  // no decay: counts are exact
  std::vector<std::int32_t> tokens;
  for (int i = 0; i < 39; ++i)
    tokens.push_back(10);
  for (int i = 0; i < 41; ++i)
    tokens.push_back(11);
  for (int i = 0; i < 48; ++i)
    tokens.push_back(12);  // 12 > 11 > 10 at the first rebuild
  Require(f.Observe(tokens, false), "no rebuild");
  Require(f.Contains(12) && f.Contains(11) && !f.Contains(10),
          "the two most counted tokens are not the subset");
  const auto version = f.Version();
  // Fillers counted twice each stay far below the members.
  const auto fill = [](std::vector<std::int32_t>& v) {
    for (std::int32_t id = 20; v.size() < 128; id = id == 79 ? 20 : id + 1)
      v.push_back(id);
  };
  // 10 reaches 47 > 41 but not 1.25 * 41: no swap.
  std::vector<std::int32_t> next = Repeat(10, 8);
  fill(next);
  Require(f.Observe(next, false), "no rebuild");
  Require(!f.Contains(10) && f.Version() == version,
          "a candidate under the hysteresis replaced a member");
  // 10 then reaches 67 and replaces the weakest member (11).
  next = Repeat(10, 20);
  fill(next);
  Require(f.Observe(next, false), "no rebuild");
  Require(f.Contains(10) && f.Contains(12) && !f.Contains(11),
          "the stronger candidate did not replace the weakest member");
  Require(f.Version() == version + 1, "a swap did not change the version");
  Require(f.Subset().size() == 2, "the subset grew");

  Frequency decayed(2, 100, 0.5F);
  decayed.Observe(Repeat(7, 128), false);  // rebuild: counts then decay
  Require(decayed.Contains(7), "the only token is not a member");
  Require(std::fabs(decayed.Count(7) - 64.0F) < 1e-3F,
          "the count did not decay at the rebuild");
  decayed.Observe(Repeat(8, 128), false);
  Require(std::fabs(decayed.Count(7) - 32.0F) < 1e-3F,
          "decay did not apply to an idle token");
}

// Two sessions with the same history rank identically.
void CheckDeterminism() {
  Frequency a(16, 1000), b(16, 1000);
  std::vector<std::int32_t> tokens;
  for (int i = 0; i < 1000; ++i)
    tokens.push_back((i * 37 + i / 7) % 50);
  a.Observe(std::span(tokens).first(300), true);
  a.Observe(std::span(tokens).subspan(300), false);
  b.Observe(std::span(tokens).first(300), true);
  b.Observe(std::span(tokens).subspan(300), false);
  auto x = std::vector<std::int32_t>(a.Subset().begin(), a.Subset().end());
  auto y = std::vector<std::int32_t>(b.Subset().begin(), b.Subset().end());
  Require(x == y && x.size() == 16, "equal histories ranked differently");
  a.Reset();
  Require(a.Subset().empty() && a.Count(1) == 0.0F, "Reset kept state");
}

// The subset stays on while attributable misses stay below 2 * saving /
// round, turns off above it, and the full head is probed every 128 rounds.
void CheckSwitch() {
  Switch s(1.0, 20.0);  // break-even 10%
  Require(std::fabs(s.BreakEven() - 0.1) < 1e-12, "break-even is wrong");
  int probes = 0, subset = 0;
  for (int round = 0; round < 1280; ++round) {
    const bool use = s.UseSubset();
    probes += !use;
    subset += use;
    s.Record(round % 20 == 19);  // 5% misses
  }
  Require(probes == 10, "the full head was not probed every 128 rounds");
  Require(subset == 1270, "a low miss rate left the subset");
  for (int round = 0; round < 256; ++round)
    s.Record(round % 4 == 0);  // 25% misses
  Require(!s.UseSubset() && s.MissRate() > 0.1,
          "a high miss rate kept the subset");
  for (int round = 0; round < 512; ++round)
    s.Record(false);
  Require(s.UseSubset() ||
              s.Rounds() % Switch::kProbeRounds == Switch::kProbeRounds - 1,
          "the subset did not come back after misses stopped");
  Switch never(1.0, 0.0);
  Require(!never.UseSubset(), "a zero round time allowed the subset");
}

}  // namespace

int main() {
  try {
    CheckWeightsAndRebuild();
    CheckHysteresis();
    CheckDeterminism();
    CheckSwitch();
  } catch (const std::exception& e) {
    std::cerr << "dynamic_draft_vocab_test: " << e.what() << '\n';
    return 1;
  }
  std::cout << "dynamic_draft_vocab_test: ok\n";
  return 0;
}
