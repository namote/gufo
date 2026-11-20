#include "src/cli/serve/loop_detector.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <functional>
#include <utility>

namespace gufo::server {
namespace {

// Rolling hash over 64-bit wrapping arithmetic (exllamav3 uses 2^61 - 1; any
// odd base works for a hash whose collisions only cost a spurious match).
constexpr std::uint64_t kBase = 1'000'003;
constexpr double kExactRate = 0.995;     // recent rate counted as exact
constexpr double kNearExactRate = 0.97;  // ... as near-exact (tiny vocabulary)
constexpr std::size_t kDiverseWindow = 256;
constexpr std::size_t kDiverseMin = 12;
constexpr std::size_t kPromptBloomBits = 16;
constexpr std::int64_t kPromptSourcedMin = 4;

constexpr std::uint64_t BaseToK() {
  std::uint64_t value = 1;
  for (std::size_t i = 0; i < TokenLoopDetector::kK; ++i)
    value *= kBase;
  return value;
}
constexpr std::uint64_t kBaseK = BaseToK();

// The rolling hash's low bits depend only on the tokens' low bits; mix it
// (MurmurHash3 fmix64, a bijection) before its bits pick buckets and Bloom
// bits. Equal k-grams still hash equal.
constexpr std::uint64_t Mix(std::uint64_t h) {
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ULL;
  h ^= h >> 33;
  return h != 0 ? h : 1;
}

// Every k-gram hash of a token list, as Feed computes them.
template<typename Visit>
void KgramHashes(std::span<const std::uint32_t> ids, Visit visit) {
  constexpr std::size_t k = TokenLoopDetector::kK;
  std::uint64_t ring[k]{};
  std::uint64_t hash = 0;
  for (std::size_t t = 0; t < ids.size(); ++t) {
    const std::uint64_t in = std::uint64_t{ids[t]} + 1;
    const auto out = ring[t % k];
    ring[t % k] = in;
    hash = t >= k ? hash * kBase - out * kBaseK + in : hash * kBase + in;
    if (t + 1 >= k)
      visit(Mix(hash));
  }
}

}  // namespace

void TokenLoopDetector::Prime(std::span<const std::uint32_t> prompt) {
  const std::uint64_t bits =
      std::max<std::uint64_t>(prompt.size(), 1) * kPromptBloomBits - 1;
  const std::uint64_t size = std::uint64_t{1}
                             << std::max<int>(10, std::bit_width(bits));
  bloom_mask_ = size - 1;
  bloom_.assign(size >> 3, 0);
  KgramHashes(prompt, [&](std::uint64_t hash) {
    for (const auto x : {hash, hash >> 21, hash >> 42}) {
      const auto bit = x & bloom_mask_;
      bloom_[bit >> 3] |= static_cast<std::uint8_t>(1U << (bit & 7));
    }
  });
}

bool TokenLoopDetector::InPrompt(std::uint64_t hash) const noexcept {
  if (bloom_.empty())
    return false;
  for (const auto x : {hash, hash >> 21, hash >> 42}) {
    const auto bit = x & bloom_mask_;
    if (((bloom_[bit >> 3] >> (bit & 7)) & 1U) == 0)
      return false;
  }
  return true;
}

void TokenLoopDetector::Allocate() {
  std::size_t slots = 1;
  while (slots < 2 * kMaxPeriod)
    slots <<= 1;
  mask_ = slots - 1;
  key_.assign(slots, 0);
  newest_.assign(slots, -1);
  positions_.assign(slots * kWaysPos, -1);
  head_.assign(slots, 0);
  hashes_.assign(kMaxPeriod * kMaxMult + 1, 0);
  tokens_.assign(kDiverseWindow, -1);
}

void TokenLoopDetector::Reset() {
  key_.clear();
  newest_.clear();
  positions_.clear();
  head_.clear();
  hashes_.clear();
  tokens_.clear();
  std::fill(std::begin(ring_), std::end(ring_), 0);
  hash_ = 0;
  t_ = -1;
  std::fill(std::begin(runs_), std::end(runs_), Run{});
  detected_.reset();
}

bool TokenLoopDetector::Diverse(std::int64_t t, std::int64_t period) const {
  const std::int64_t n =
      std::min({std::max<std::int64_t>(period, 64),
                static_cast<std::int64_t>(kDiverseWindow), t + 1});
  if (n < 32)
    return true;
  std::array<std::int64_t, kDiverseWindow> recent;
  const auto size = static_cast<std::size_t>(n);
  for (std::size_t j = 0; j < size; ++j)
    recent[j] =
        tokens_[static_cast<std::size_t>(t - static_cast<std::int64_t>(j)) %
                kDiverseWindow];
  const auto end = recent.begin() + static_cast<std::ptrdiff_t>(size);
  std::sort(recent.begin(), end);
  const auto distinct = static_cast<std::size_t>(
      std::unique(recent.begin(), end) - recent.begin());
  return distinct >= kDiverseMin;
}

bool TokenLoopDetector::IsLoop(Run& run, std::int64_t t) const {
  const auto p = run.period;
  const auto span = t - run.start;
  // Transcription of distinct prompt items: sticky once established, and it
  // lapses two periods after the last new prompt content.
  if (run.sourced >= kPromptSourcedMin &&
      static_cast<double>(run.sourced) >=
          kPromptFraction * static_cast<double>(run.unmatched))
    run.exempt = true;
  if (run.exempt && t - run.sourced_last <= 2 * p)
    return false;
  const bool exact = run.recent >= kExactRate;
  const auto reps = static_cast<std::int64_t>(exact ? kExactReps : kMinReps);
  const auto length = static_cast<std::int64_t>(exact ? kExactLen : kMinLen);
  if (span < reps * p || span < length)
    return false;
  if (static_cast<double>(run.matched) <
          kMinFraction *
              static_cast<double>(std::max<std::int64_t>(span - p, 1)) ||
      run.recent < kMinRecent)
    return false;
  // Tiny vocabulary: structure alone makes k-grams recur, so require
  // (near-)exact repetition.
  if (!Diverse(t, p))
    return exact || (run.recent >= kNearExactRate &&
                     span >= static_cast<std::int64_t>(kLowDiversityLen));
  return true;
}

std::optional<TokenLoopDetector::Detection> TokenLoopDetector::Feed(
    std::uint32_t token) {
  if (key_.empty())
    Allocate();
  const auto t = ++t_;
  constexpr auto k = static_cast<std::int64_t>(kK);
  const std::uint64_t in = std::uint64_t{token} + 1;
  const auto slot = static_cast<std::size_t>(t % k);
  const auto out = ring_[slot];
  ring_[slot] = in;
  hash_ = t >= k ? hash_ * kBase - out * kBaseK + in : hash_ * kBase + in;
  tokens_[static_cast<std::size_t>(t) % kDiverseWindow] = token;
  if (t < k - 1)
    return std::nullopt;
  const std::uint64_t h = Mix(hash_);
  const auto max_period = static_cast<std::int64_t>(kMaxPeriod);

  // Earlier occurrences of this k-gram within kMaxPeriod, as distances.
  const std::size_t bucket = static_cast<std::size_t>(h & mask_ & ~1ULL);
  std::size_t entry = bucket;
  bool found = false;
  for (const auto way : {bucket, bucket + 1}) {
    if (key_[way] == h && newest_[way] >= t - max_period) {
      entry = way;
      found = true;
      break;
    }
  }
  std::int64_t distances[kWaysPos];
  std::size_t count = 0;
  if (found) {
    for (std::size_t j = 0; j < kWaysPos; ++j) {
      const std::int64_t x = positions_[entry * kWaysPos + j];
      if (x >= 0 && t - x <= max_period)
        distances[count++] = t - x;
    }
  } else {
    entry = newest_[bucket] <= newest_[bucket + 1] ? bucket : bucket + 1;
    key_[entry] = h;
    std::fill_n(
        positions_.begin() + static_cast<std::ptrdiff_t>(entry * kWaysPos),
        kWaysPos, -1);
    head_[entry] = 0;
  }
  const auto observed = std::span(distances, count);

  // A k-gram never generated before that occurs in the prompt.
  const bool new_prompt_content = count == 0 && InPrompt(h);

  // Update runs.
  const auto history = static_cast<std::int64_t>(hashes_.size());
  std::optional<Detection> detection;
  for (auto& run : runs_) {
    if (!run.on)
      continue;
    const auto p = run.period;
    const auto tolerance =
        static_cast<std::int64_t>(static_cast<double>(p) * kTolerance);
    // The same k-gram m periods back, within the drift (stored positions).
    std::int64_t hit_m = 0;
    std::int64_t hit_q = 0;
    for (std::int64_t m = 1; m <= static_cast<std::int64_t>(kMaxMult) && !hit_m;
         ++m) {
      const auto lo = m * (p - tolerance);
      const auto hi = m * (p + tolerance);
      for (const auto q : observed) {
        if (lo <= q && q <= hi) {
          hit_m = m;
          hit_q = q;
          break;
        }
      }
    }
    // The few stored positions can all lie inside the current copy when the
    // unit itself repeats phrases; compare the hash exactly m periods back.
    if (!hit_m) {
      for (std::int64_t m = 1; m <= static_cast<std::int64_t>(kMaxMult); ++m) {
        const auto s = t - m * p;
        if (s >= 0 && t - s < history &&
            hashes_[static_cast<std::size_t>(s % history)] == h) {
          hit_m = m;
          hit_q = m * p;
          break;
        }
      }
    }
    const double tau = static_cast<double>(std::max<std::int64_t>(32, p));
    if (hit_m) {
      if (hit_m == 1 && hit_q != p)
        run.period = hit_q;  // follow insertions and deletions
      run.last = t;
      ++run.matched;
      run.recent += (1.0 - run.recent) / tau;
    } else if (t - run.last > static_cast<std::int64_t>(kGap)) {
      run.on = false;
      continue;
    } else {
      run.recent -= run.recent / tau;
      ++run.unmatched;
      if (new_prompt_content) {
        ++run.sourced;
        run.sourced_last = t;
      }
    }
    if (!detected_ && IsLoop(run, t)) {
      detected_ = Detection{
          .period = static_cast<std::size_t>(run.period),
          .repeats = static_cast<std::size_t>((t - run.start) / run.period),
          .position = static_cast<std::size_t>(t),
      };
      detection = detected_;
    }
  }

  // Start runs for distances no live run explains.
  for (const auto q : observed) {
    bool explained = false;
    for (const auto& run : runs_) {
      if (!run.on)
        continue;
      const auto p = run.period;
      const auto tolerance =
          static_cast<std::int64_t>(static_cast<double>(p) * kTolerance);
      for (std::int64_t m = 1; m <= static_cast<std::int64_t>(kMaxMult); ++m) {
        if (std::abs(q - m * p) <= m * tolerance) {
          explained = true;
          break;
        }
      }
      if (explained)
        break;
    }
    if (explained)
      continue;
    Run* free = nullptr;
    for (auto& run : runs_) {
      if (!run.on) {
        free = &run;
        break;
      }
    }
    if (free == nullptr) {
      // Replace a live run that has not matched beyond its first token,
      // otherwise drop the candidate.
      free = std::ranges::min_element(runs_, {}, &Run::matched);
      if (free->matched > 1)
        continue;
    }
    *free =
        Run{.on = true, .start = t - q, .period = q, .last = t, .matched = 1};
  }

  // Record this k-gram.
  hashes_[static_cast<std::size_t>(t % history)] = h;
  const auto way = head_[entry];
  positions_[entry * kWaysPos + way] = static_cast<std::int32_t>(t);
  head_[entry] = static_cast<std::uint8_t>((way + 1) % kWaysPos);
  newest_[entry] = static_cast<std::int32_t>(t);
  return detection;
}

bool LineRepeatDetector::Feed(std::string_view text) {
  bool found = false;
  for (const char c : text) {
    ++offset_;
    if (c != '\n') {
      if (line_.size() < kMaxLineBytes)
        line_.push_back(c);
      else
        line_overflow_ = true;
      continue;
    }
    constexpr std::string_view kSpace = " \t\r\f\v";
    std::string_view line = line_;
    const auto first = line.find_first_not_of(kSpace);
    line = first == std::string_view::npos
               ? std::string_view{}
               : line.substr(first, line.find_last_not_of(kSpace) - first + 1);
    if (line.starts_with("```")) {
      fence_ = !fence_;
    } else if (!fence_ && !line_overflow_) {
      // Characters, not bytes: count UTF-8 lead bytes other than spaces/tabs.
      std::size_t chars = 0;
      for (const char byte : line)
        chars += byte != ' ' && byte != '\t' &&
                 (static_cast<unsigned char>(byte) & 0xC0U) != 0x80U;
      if (chars >= kMinChars) {
        if (seen_.size() >= kMaxLines) {
          // Forget lines no later sighting can count with.
          std::erase_if(seen_, [&](const auto& entry) {
            return offset_ - entry.second.offset > kWindowBytes;
          });
          if (seen_.size() >= kMaxLines)
            seen_.clear();
        }
        auto& sighting = seen_[std::hash<std::string_view>{}(line)];
        sighting.count =
            sighting.count != 0 && offset_ - sighting.offset <= kWindowBytes
                ? sighting.count + 1
                : 1;
        sighting.offset = offset_;
        if (sighting.count >= kRepeats && !detected_) {
          detected_ = true;
          found = true;
        }
      }
    }
    line_.clear();
    line_overflow_ = false;
  }
  return found;
}

void LineRepeatDetector::Reset() {
  line_.clear();
  line_overflow_ = false;
  offset_ = 0;
  fence_ = false;
  detected_ = false;
  seen_.clear();
}

GenerationLoopMonitor::GenerationLoopMonitor(
    std::span<const std::uint32_t> prompt, bool reasoning,
    std::vector<std::uint32_t> reasoning_ends)
    : reasoning_(reasoning), reasoning_ends_(std::move(reasoning_ends)) {
  tokens_.Prime(prompt);
}

std::optional<LoopDetection> GenerationLoopMonitor::Observe(
    std::uint32_t token, std::string_view piece) {
  ++observed_;
  std::optional<LoopDetection> found;
  if (!reported_) {
    if (const auto detection = tokens_.Feed(token)) {
      found = LoopDetection{.reasoning = reasoning_,
                            .rule = "period",
                            .period = detection->period,
                            .repeats = detection->repeats,
                            .tokens = observed_};
    } else if (reasoning_ && lines_.Feed(piece)) {
      found = LoopDetection{.reasoning = true,
                            .rule = "line",
                            .period = 0,
                            .repeats = LineRepeatDetector::kRepeats,
                            .tokens = observed_};
    }
    reported_ = found.has_value();
  }
  if (reasoning_ &&
      std::ranges::find(reasoning_ends_, token) != reasoning_ends_.end()) {
    // The answer starts fresh: it may restate what reasoning repeated.
    reasoning_ = false;
    tokens_.Reset();
    lines_.Reset();
    reported_ = false;
  }
  return found;
}

}  // namespace gufo::server
