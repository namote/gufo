#ifndef GUFO_SERVER_TEXT_GENERATION_SCHEDULER_HPP_
#define GUFO_SERVER_TEXT_GENERATION_SCHEDULER_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/cli/serve/text_model_runner.hpp"

namespace gufo::server {

enum class TextRequestPhase : std::uint8_t {
  kQueued,
  kAdmitted,
  kPrefilling,
  kDecodeReady,
  kDecoding,
  kTerminal,
};

inline constexpr std::size_t kDefaultDecodeActivePrefillTokens = 512;
// Admission runs only between work units. While a session is free, a lone
// prefill stops on this absolute interval, the cache grid's spacing, so an
// arrival's wait does not depend on how many grid points the RAM budget keeps.
inline constexpr std::size_t kAdmissionPrefillTokens = 2048;
// R2 A2: a dynamic abliteration detection readmits its request ON only when
// at most this many prompt tokens remain to prefill ON after the longest
// cached ON checkpoint (RAM estimate). At the measured ~900-1000 tok/s of
// deep-context prefill on gfx1151 that bounds the extra latency near 35 s;
// a first detection deep in a long conversation stays OFF and is reported
// with `skipped: "reprefill_cap"`.
inline constexpr std::size_t kMaxAbliterationReprefillTokens = 32768;
inline constexpr std::size_t kDefaultMaxOutputBytes =
    static_cast<std::size_t>(1024) * 1024;
inline constexpr std::size_t kDefaultMaxBufferedOutputBytes =
    static_cast<std::size_t>(64) * 1024;
inline constexpr std::size_t kDefaultMaxBufferedOutputBytesTotal =
    static_cast<std::size_t>(256) * 1024;

struct TextPrefillPolicy {
  std::size_t decode_active_tokens{kDefaultDecodeActivePrefillTokens};
  /// Share device time rather than turns with decoders: an active-decode
  /// prefill chunk that took at least this long lets the decoders keep
  /// running until they have used as much time (capped at
  /// decode_time_share_max). Zero keeps one decode round per chunk.
  std::chrono::milliseconds decode_time_share_min{0};
  std::chrono::milliseconds decode_time_share_max{std::chrono::seconds(1)};
};

struct TextSchedulerPolicy {
  /// Internal cadence; idle monitoring performs no work during inference.
  std::chrono::milliseconds device_probe_interval{std::chrono::seconds(5)};
  // Idle time before idle sessions' live frontiers are written to disk.
  std::chrono::milliseconds idle_write_back_delay{std::chrono::seconds(2)};
  /// Internal bound on waiting for a capture the last step started before
  /// queuing other work. The executor is idle then, so the copy finishes in
  /// milliseconds; behind a peer's prefill pass it waited for the whole pass.
  std::chrono::milliseconds capture_settle_wait{50};
  /// Internal bound after which a still-queued stream reports its start, so
  /// transports can keep the connection alive while it waits.
  std::chrono::milliseconds stream_start_delay{std::chrono::seconds(5)};
  /// Internal: a prefill with at most this many tokens left runs before a
  /// longer peer's next chunk, which is then bounded like a decode-active
  /// chunk. Such out-of-turn picks start only while fewer than this many
  /// tokens have bypassed the long prefill since its last turn, so it waits
  /// at most this bound plus one pass. Zero disables the priority.
  std::size_t short_prefill_priority_tokens{4096};
  std::size_t max_pending_requests{16};
  std::size_t max_pending_requests_per_client{4};
  std::size_t max_output_bytes_per_request{kDefaultMaxOutputBytes};
  std::size_t max_buffered_output_bytes_per_request{
      kDefaultMaxBufferedOutputBytes};
  std::size_t max_buffered_output_bytes_total{
      kDefaultMaxBufferedOutputBytesTotal};
  std::chrono::milliseconds request_timeout{0};
  /// A scheduler that has work but starts no work unit and completes no
  /// pending capture for this long is reported as a lost device. Zero
  /// disables it.
  std::chrono::milliseconds stall_timeout{0};
  bool log_progress{false};
};

/// Single-owner scheduler for opaque text-model runner states.
///
/// Submitters never execute model code. One scheduler thread owns admission,
/// runner leases, prefill/decode work units, cancellation, and reclamation.
class TextGenerationScheduler {
public:
  using Clock = std::chrono::steady_clock;
  using Result = TextGenerationBackend::Result;
  using CancellationCheck = TextGenerationBackend::CancellationCheck;
  using TokenCallback = TextGenerationBackend::TokenCallback;
  using ProgressCallback = TextGenerationBackend::ProgressCallback;
  using StartCallback = TextGenerationBackend::StartCallback;
  using SessionState = TextGenerationBackend::SessionState;

  struct RequestMetadata {
    std::string client_id{"anonymous"};
    std::optional<Clock::time_point> deadline;
    Clock::time_point request_start{Clock::now()};
    std::shared_ptr<const TextPromptContext> prompt_context;
    bool cache_prompt{true};
    /// False keeps this request's checkpoints out of the disk cache.
    bool cache_persist{true};
    std::size_t cache_prefix_tokens{0};
    /// Shared leading-instruction frontier, supplied by prompt preparation.
    std::size_t system_prefix_tokens{0};
    bool stop_at_eos{true};
    std::vector<std::string> stop_sequences;
    bool return_progress{false};
    /// Non-empty: score these tokens after the prompt (`score_assistant`)
    /// instead of generating. Result::tokens and Result::logprobs then hold
    /// them and their raw logprobs; nothing is generated, streamed or
    /// written to the disk cache, and only a prompt checkpoint is retained.
    std::vector<TextRunnerToken> score_tokens;
  };

  class Request {
  public:
    Request();
    ~Request();

    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;
    Request(Request&&) noexcept;
    Request& operator=(Request&&) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] std::uint64_t id() const noexcept;
    [[nodiscard]] TextRequestPhase phase() const noexcept;

    /// Consumes queued output pieces on the calling thread and waits for the
    /// scheduler-owned request to become terminal. Streaming requests report
    /// only the latest prompt progress, always before their first piece.
    /// Their start is reported once at admission, or after
    /// `stream_start_delay` in the queue, before progress and pieces.
    Result Wait(const TokenCallback& on_token = {},
                const ProgressCallback& on_progress = {},
                const StartCallback& on_start = {});
    void Cancel() noexcept;

  private:
    friend class TextGenerationScheduler;
    struct Impl;

    explicit Request(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
  };

  explicit TextGenerationScheduler(std::shared_ptr<TextRunnerPool> runner_pool,
                                   TextPrefillPolicy prefill_policy = {},
                                   TextSchedulerPolicy scheduler_policy = {});
  ~TextGenerationScheduler();

  TextGenerationScheduler(const TextGenerationScheduler&) = delete;
  TextGenerationScheduler& operator=(const TextGenerationScheduler&) = delete;
  TextGenerationScheduler(TextGenerationScheduler&&) = delete;
  TextGenerationScheduler& operator=(TextGenerationScheduler&&) = delete;

  [[nodiscard]] const TextModelRunner& runner() const noexcept;
  [[nodiscard]] std::size_t capacity() const noexcept;
  /// Set once a failed work unit finds the runner's device unusable; later
  /// submissions and failures report TextGenerationErrorCode::kDeviceLost.
  [[nodiscard]] bool device_lost() const noexcept;
  /// One entry per session; admitted requests hold one until terminal.
  [[nodiscard]] std::vector<SessionState> SessionStates() const;
  [[nodiscard]] std::size_t buffered_output_bytes() const noexcept;
  [[nodiscard]] std::size_t max_buffered_output_bytes() const noexcept;

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens,
                               const sampling::SamplingConfig& sampling,
                               const CancellationCheck& is_cancelled = {},
                               bool publish_token_pieces = false);

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens,
                               const sampling::SamplingConfig& sampling,
                               const CancellationCheck& is_cancelled,
                               bool publish_token_pieces,
                               RequestMetadata metadata);

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens, float temperature,
                               const CancellationCheck& is_cancelled = {},
                               bool publish_token_pieces = false);

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens, float temperature,
                               const CancellationCheck& is_cancelled,
                               bool publish_token_pieces,
                               RequestMetadata metadata);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

using TextRequestMetadata = TextGenerationScheduler::RequestMetadata;

}  // namespace gufo::server

#endif  // GUFO_SERVER_TEXT_GENERATION_SCHEDULER_HPP_
