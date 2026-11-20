#ifndef GUFO_SERVER_LOOP_DETECTOR_HPP_
#define GUFO_SERVER_LOOP_DETECTOR_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gufo::server {

/// Streaming detection of generation loops: the model repeating the same
/// span of tokens, exactly or nearly, with nothing else in between. A port of
/// exllamav3 805caee (`generator/loop_detect.py`), model independent.
///
/// Each token's k-gram (k consecutive tokens) is hashed and looked up among
/// recent k-gram positions; the distance to an earlier occurrence is a
/// candidate period p. A run follows one period: a token matches it when the
/// same k-gram also ended m periods back (m up to kMaxMult), p drifting by up
/// to 5% (edits in the copies). A run ends after kGap unmatched tokens. It is
/// a loop once it spans 8 periods and 8192 tokens, or 4 periods and 1024
/// tokens when its recent match rate is (near) exact; at least half of its
/// tokens matched and its recent rate is at least 0.75. Copies whose
/// differences are prompt content not generated before (answer options being
/// compared) are exempt until two periods after the last such token. Over a
/// tiny vocabulary only exact repetition, or near-exact repetition over 2400
/// tokens, counts. O(1) per token; about 2.6 MiB of tables per detector,
/// allocated on the first token, plus a prompt Bloom filter of 2 to 4 bytes
/// per prompt token (at least 128 bytes).
class TokenLoopDetector {
public:
  static constexpr std::size_t kK = 6;
  static constexpr std::size_t kMaxPeriod = 32768;
  static constexpr std::size_t kMinReps = 8;
  static constexpr std::size_t kMinLen = 8192;
  static constexpr std::size_t kExactReps = 4;
  static constexpr std::size_t kExactLen = 1024;
  static constexpr double kTolerance = 0.05;
  static constexpr std::size_t kGap = 24;
  static constexpr std::size_t kMaxMult = 3;
  static constexpr double kMinFraction = 0.5;
  static constexpr double kMinRecent = 0.75;
  static constexpr std::size_t kLowDiversityLen = 2400;
  static constexpr double kPromptFraction = 0.25;
  static constexpr std::size_t kMaxRuns = 8;
  static constexpr std::size_t kWaysPos = 4;

  struct Detection {
    std::size_t period{0};
    /// Periods the run spans when it qualified.
    std::size_t repeats{0};
    /// Zero-based index of the token that completed the loop.
    std::size_t position{0};
  };

  /// Indexes the prompt's k-grams (Bloom filter) for the transcription
  /// exemption; without it everything else works as usual.
  void Prime(std::span<const std::uint32_t> prompt);
  /// Observes one generated token. Returns the detection the first time a
  /// run qualifies; detection latches until Reset.
  std::optional<Detection> Feed(std::uint32_t token);
  [[nodiscard]] bool detected() const noexcept { return detected_.has_value(); }
  /// Forgets generated tokens and runs; keeps the primed prompt.
  void Reset();

private:
  struct Run {
    bool on{false};
    std::int64_t start{0};
    std::int64_t period{0};
    std::int64_t last{0};
    std::int64_t matched{0};
    std::int64_t unmatched{0};
    std::int64_t sourced{0};
    std::int64_t sourced_last{0};
    bool exempt{false};
    double recent{1.0};
  };

  void Allocate();
  [[nodiscard]] bool InPrompt(std::uint64_t hash) const noexcept;
  [[nodiscard]] bool Diverse(std::int64_t t, std::int64_t period) const;
  [[nodiscard]] bool IsLoop(Run& run, std::int64_t t) const;

  // k-gram table: 2-way buckets of (hash, newest position, ring of the last
  // kWaysPos positions); a missing k-gram evicts the least recent entry.
  std::vector<std::uint64_t> key_;
  std::vector<std::int32_t> newest_;
  std::vector<std::int32_t> positions_;
  std::vector<std::uint8_t> head_;
  std::uint64_t mask_{0};
  // Rolling hash state, recent k-gram hashes and recent tokens.
  std::uint64_t ring_[kK]{};
  std::uint64_t hash_{0};
  std::vector<std::uint64_t> hashes_;
  std::vector<std::int64_t> tokens_;
  std::int64_t t_{-1};
  Run runs_[kMaxRuns]{};
  std::vector<std::uint8_t> bloom_;
  std::uint64_t bloom_mask_{0};
  std::optional<Detection> detected_;
};

/// Kyojin's long-period thinking rule (`tools/loopguard.py` LineRepeat): a
/// complete line of at least 40 non-space characters (compared after
/// trimming) seen 3 times is a loop. Lines inside a ``` fence are not
/// counted: code repeats its own lines. Unlike Kyojin's rule it is bounded: a
/// sighting counts only within 16 KiB of text after the previous one, lines
/// over 1 KiB are not counted, and at most 1024 lines are remembered.
class LineRepeatDetector {
public:
  static constexpr std::size_t kMinChars = 40;
  static constexpr std::size_t kRepeats = 3;
  static constexpr std::size_t kWindowBytes = 16384;
  static constexpr std::size_t kMaxLineBytes = 1024;
  static constexpr std::size_t kMaxLines = 1024;

  /// Returns true the first time a line reaches kRepeats sightings;
  /// detection latches until Reset.
  bool Feed(std::string_view text);
  [[nodiscard]] bool detected() const noexcept { return detected_; }
  void Reset();

private:
  struct Sighting {
    std::size_t count{0};
    std::uint64_t offset{0};  ///< text offset of the latest sighting
  };

  std::string line_;
  bool line_overflow_{false};
  std::uint64_t offset_{0};
  bool fence_{false};
  bool detected_{false};
  std::unordered_map<std::uint64_t, Sighting> seen_;
};

/// Where a reply is when a loop is found, and which rule found it.
struct LoopDetection {
  bool reasoning{false};
  /// "period" (TokenLoopDetector) or "line" (LineRepeatDetector).
  std::string_view rule;
  /// The repeating unit in tokens; 0 for the line rule.
  std::size_t period{0};
  std::size_t repeats{0};
  /// Generated tokens observed, the loop's last one included.
  std::size_t tokens{0};
};

/// Per-request loop monitor of the shared generation path. Follows the
/// reply's reasoning phase from its start state and the reasoning-end
/// tokens, runs the period rule in both phases and the line rule only while
/// reasoning, and reports at most one detection per phase (a new phase
/// starts fresh detectors).
class GenerationLoopMonitor {
public:
  GenerationLoopMonitor(std::span<const std::uint32_t> prompt, bool reasoning,
                        std::vector<std::uint32_t> reasoning_ends);

  std::optional<LoopDetection> Observe(std::uint32_t token,
                                       std::string_view piece);
  [[nodiscard]] bool reasoning() const noexcept { return reasoning_; }

private:
  bool reasoning_;
  std::vector<std::uint32_t> reasoning_ends_;
  TokenLoopDetector tokens_;
  LineRepeatDetector lines_;
  std::size_t observed_{0};
  bool reported_{false};
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_LOOP_DETECTOR_HPP_
