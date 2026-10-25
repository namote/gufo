#ifndef GUFO_SERVER_TEXT_MODEL_RUNNER_HPP_
#define GUFO_SERVER_TEXT_MODEL_RUNNER_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/block_disk_store.hpp"
#include "src/cache/adapter.hpp"
#include "src/cache/ledger.hpp"
#include "src/cli/serve/continuation_cache.hpp"
#include "src/cli/serve/text_generation_backend.hpp"
#include "src/cli/serve/text_publication_wait.hpp"
#include "src/core/sampling.hpp"

namespace gufo::server {

struct ComponentCacheResources;

using TextRunnerToken = ContinuationToken;

/// Immutable model-owned input which travels with one scheduled request.
/// Cache identity supplements token IDs for inputs such as image embeddings.
struct TextPromptContext {
  virtual ~TextPromptContext() = default;
  std::vector<std::uint8_t> cache_identity;
  std::vector<ContinuationInputPrefix> cache_prefixes;
  [[nodiscard]] std::span<const std::uint8_t> CacheIdentity(
      std::size_t token_count) const {
    return PrefixInputIdentity(cache_identity, cache_prefixes, token_count);
  }
  /// The nearest position at or before `position` (`up`: at or after) where
  /// prefill may stop. A model can need an input whole, such as an image
  /// whose rows attend each other; a step starting at such an input then
  /// consumes it entirely, even beyond its token budget.
  [[nodiscard]] virtual std::size_t PrefillStop(std::size_t position,
                                                bool /*up*/) const {
    return position;
  }
};

struct TextPreparedPrompt {
  std::vector<TextRunnerToken> tokens;
  std::shared_ptr<const TextPromptContext> context;
  /// Snapshot before mutable assistant framing. Zero uses the complete prompt.
  std::size_t cache_prefix_tokens{0};
};

struct TextRunnerRamCacheOptions {
  static constexpr std::size_t kAutomaticMaxBytes = std::size_t{32} << 30;
  static constexpr std::size_t kMaxEntries = 128;
  /// Zero selects min(model snapshot budget, 32 GiB), after session allocation.
  /// Explicit limits may exceed that budget up to the model's ceiling, which
  /// keeps kHostSnapshotHeadroomBytes of host RAM free.
  std::size_t capacity_bytes{0};
  /// Paged attention rows: one device pool of this many bytes shared by all
  /// sessions, which take blocks as they write. Zero gives each session its
  /// whole context up front. Runners without row blocks refuse a nonzero
  /// value at load.
  std::size_t kv_pool_bytes{0};
  /// Long prompts also keep a checkpoint every this many tokens, in RAM and
  /// on disk, so a later prompt that shares a long prefix and then differs
  /// (a document with another question) restores the last one before the
  /// difference. Zero: none beyond the defaults.
  std::size_t checkpoint_grid_tokens{0};
  /// Long prompts also keep this many RAM checkpoints, one every
  /// kTailCheckpointTokens over the end of the prompt, so a request that
  /// shares all but the end of a prompt already prefilled past their
  /// divergence restores within kTailCheckpointTokens of it instead of the
  /// nearest evenly spaced checkpoint. Each is a recurrent copy plus block
  /// references and gives way under RAM pressure. Zero: none. Component
  /// cache models and runners whose snapshots copy their rows
  /// (TextRunnerCapabilities::block_sharing_snapshots) do not apply them
  /// (logged at startup).
  std::size_t tail_checkpoints{3};
  static constexpr std::size_t kTailCheckpointTokens = 1024;
  /// One pass scratch for prompt, verification and batched decode passes,
  /// shared by every session (passes run one at a time); sessions keep only
  /// decode-step scratch. Runners without pass scratch ignore it.
  bool shared_pass_scratch{true};
  /// A shared prefix must add at least this many tokens to what a request
  /// can already restore before it waits for a peer prefilling it or a
  /// checkpoint is captured there for later requests. A checkpoint is a
  /// recurrent copy plus block references (a few ms); shorter gaps cost less
  /// to prefill again. The same bound applies to shared prefixes learned
  /// from the disk store. Must be at least one.
  std::size_t shared_prefix_min_tokens{128};
};

struct TextRunnerDiskCacheOptions {
  static constexpr std::size_t kDefaultCapacityBytes =
      std::size_t{8} * 1024U * 1024U * 1024U;
  static constexpr std::size_t kDefaultBlockCapacityBytes =
      std::size_t{32} * 1024U * 1024U * 1024U;
  std::filesystem::path directory;
  std::size_t capacity_bytes{kDefaultCapacityBytes};
  /// Zero selects 1/8 of available host RAM, capped at capacity_bytes.
  std::size_t staging_capacity_bytes{0};
  /// Row blocks of runners that persist rows in blocks, kept apart from the
  /// records (capacity_bytes) under DIR/blocks.
  std::size_t block_capacity_bytes{kDefaultBlockCapacityBytes};
  /// Writes a released frontier's record once it has been idle this many
  /// milliseconds, instead of only when an eviction needs its seat. Zero:
  /// only on eviction and at shutdown.
  std::size_t idle_write_ms{0};
  /// Index resident row blocks by their block keys (needs the row block
  /// store and paged rows).
  bool resident_blocks{true};
  /// Row blocks a restore reads and checks ahead of the one it uploads,
  /// each on its own thread and holding one block in host memory. Zero:
  /// one at a time. At most kMaxReadAheadBlocks.
  std::size_t read_ahead_blocks{16};
  static constexpr std::size_t kMaxReadAheadBlocks = 64;
  /// Bound on shared-prefix snapshots written while prefilling one request.
  std::size_t shared_prefix_max_boundaries{4};
  /// Minimum token advance before a checkpoint is worth persisting.
  ///
  /// Continuations advance by a few hundred tokens per turn, so persisting
  /// every turn rewrites a largely identical snapshot. The tokens between two
  /// checkpoints are cheaper to re-prefill than that write. Zero disables the
  /// gate.
  std::size_t min_checkpoint_step_tokens{2048};
};

/// MemAvailable from /proc/meminfo text minus CmaFree, in bytes. Free CMA
/// pages count as available but only hold movable pages, not GPU
/// allocations; the kernel's KHO scratch area can make that several GiB.
/// Nullopt when MemAvailable is missing.
[[nodiscard]] std::optional<std::uint64_t> MeminfoAvailableBytes(
    std::string_view meminfo);
/// `host_available` bounded by every cgroup v2 memory limit from the process's
/// cgroup (named in `membership_text`, the /proc/self/cgroup contents) up to
/// `root`. Inactive file pages in memory.stat are reclaimable page cache and
/// are not counted as used; without that line, memory.current is used as is.
[[nodiscard]] std::uint64_t CgroupAvailableBytes(
    const std::filesystem::path& root, std::string_view membership_text,
    std::uint64_t host_available);
/// Automatic snapshot budget: half the host RAM available after loading
/// (excluding free CMA pages), after cgroup limits.
[[nodiscard]] std::size_t HostSnapshotBudgetBytes();
/// RAM always left to the OS and other processes by an explicit cache limit.
inline constexpr std::uint64_t kHostSnapshotHeadroomBytes = std::uint64_t{4}
                                                            << 30;
/// Most an explicit --cache-ram-bytes may claim: available host RAM minus
/// kHostSnapshotHeadroomBytes.
[[nodiscard]] std::size_t HostSnapshotCeilingBytes();

enum class TextExecutionPlanKind : std::uint8_t {
  kSerial,
  kBatched,
};

struct TextExecutionPlan {
  TextExecutionPlanKind kind{TextExecutionPlanKind::kSerial};
  std::size_t physical_width{1};

  bool operator==(const TextExecutionPlan&) const = default;
};

struct TextRunnerCapabilities {
  bool incremental_prefill{false};
  bool snapshot{false};
  bool fork{false};
  bool final_token_advance_required{true};
  bool incremental_text_is_exact{false};
  bool multi_token_decode{false};
  bool batched_multi_token_decode{false};
  /// Zero means no physical-width limit.
  std::size_t batched_multi_token_decode_max_width{0};
  bool prefix_reuse{true};
  bool in_pass_checkpoint{false};
  /// A completed prefill step leaves a consistent state, so a cancelled
  /// request can keep it as a live prefix for a superseding request.
  bool partial_prefill_retention{false};
  /// Snapshots reference the state's row blocks instead of copying the
  /// rows, so a checkpoint costs only the state outside them (tail
  /// checkpoints are applied only then).
  bool block_sharing_snapshots{false};
};

/// Model-owned compatibility identity for restart-safe snapshots.
///
/// The bytes are canonical and opaque to serving code. They must cover every
/// model, tokenizer, template, layout, precision, context-policy, and adapter
/// property that can change restored continuation semantics. Executable
/// revisions are intentionally excluded when they retain the same state ABI.
struct TextRunnerPersistenceDescriptor {
  std::vector<std::uint8_t> compatibility_identity;
  std::uint32_t payload_version{0};

  bool operator==(const TextRunnerPersistenceDescriptor&) const = default;
};

struct TextRunnerDescriptor {
  std::string model_id;
  std::string state_abi;
  std::uint32_t max_context{0};
  TextRunnerCapabilities capabilities;
  std::optional<TextRunnerPersistenceDescriptor> persistence;
  /// Longest prompt pass one Prefill call runs; a longer budget is cut to
  /// it. Zero: no fixed limit.
  std::uint32_t max_prefill_pass_tokens{0};
};

/// Bytes of one checkpoint taken at the runner's maximum context.
struct TextRunnerCheckpointBound {
  /// SnapshotPayloadBytes() of a state at max_context (retained GPU snapshot,
  /// budgeted by --cache-ram-bytes: device memory on unified-memory APUs).
  std::size_t ram_snapshot_bytes{0};
  /// PersistentSnapshotPayloadBytes() of that snapshot (disk tier).
  std::size_t persistent_payload_bytes{0};
};

/// Optional byte claims made before state allocation.
///
/// A missing value is an explicit "not yet measurable" claim. When both the
/// aggregate state capacity and per-request state are known, the common pool
/// validates the requested concurrency before creating model-private state.
struct TextRunnerResourceClaim {
  std::optional<std::size_t> resident_weights_bytes;
  std::optional<std::size_t> state_capacity_bytes;
  std::optional<std::size_t> per_request_state_bytes;
  std::optional<std::size_t> temporary_scratch_bytes;
  /// Aggregate bytes currently available for immutable retained snapshots.
  ///
  /// The pool queries this again after creating all mutable request states.
  std::optional<std::size_t> retained_snapshot_capacity_bytes;
  /// Most an explicit RAM-cache limit may claim. Missing means the automatic
  /// retained_snapshot_capacity_bytes is also the ceiling.
  std::optional<std::size_t> retained_snapshot_ceiling_bytes;
  bool requires_device_runtime_lock{false};
};

struct TextRunnerMeasuredResources {
  std::optional<std::size_t> per_request_state_bytes;
  std::optional<std::size_t> temporary_scratch_bytes;
};

struct TextPrefillStep {
  std::size_t consumed_tokens{0};
  bool decode_ready{false};
  double checkpoint_ms{0};
};

struct TextDecodeSelection {
  bool stop{false};
  TextRunnerToken token{0};
  std::string piece;
};

struct TextDecodeStep {
  std::vector<TextDecodeSelection> selections;
  /// Number of speculative verification rounds actually executed.
  std::size_t draft_rounds{0};
  std::size_t draft_tokens{0};
  std::size_t draft_accepted_tokens{0};
  /// Of draft_tokens: proposals copied from the context (prompt lookup).
  std::size_t lookup_tokens{0};
  std::size_t lookup_accepted_tokens{0};
  bool stop{false};
  /// Execution actually used for this request, including model-owned subgroup
  /// dispatch. A runner's advertised maximum is not evidence of batching.
  TextExecutionPlan execution_plan{};
  std::exception_ptr failure{};
};

/// Model-private state driven only through TextModelRunner work units.
class TextRunnerState : public ContinuationState {
public:
  using CancellationCheck = std::function<bool()>;

  void SetStopAtEos(bool value) noexcept { stop_at_eos_ = value; }
  [[nodiscard]] bool stop_at_eos() const noexcept { return stop_at_eos_; }

  /// Installs a request-scoped cancellation check for model calls that can
  /// yield internally. Implementations that only yield between work units may
  /// keep the default no-op behavior.
  virtual void SetCancellationCheck(const CancellationCheck&) {}

  /// Actual request-owned allocations when the provider can measure them.
  [[nodiscard]] virtual TextRunnerMeasuredResources MeasuredResources()
      const noexcept {
    return {};
  }

private:
  bool stop_at_eos_{true};
};

/// Immutable model-owned continuation payload.
///
/// Common serving code may account for the payload but must not inspect or
/// reinterpret its bytes.
class TextRunnerSnapshot : public ContinuationSnapshot {
public:
  TextRunnerSnapshot() = default;
  ~TextRunnerSnapshot() override = default;

  TextRunnerSnapshot(const TextRunnerSnapshot&) = delete;
  TextRunnerSnapshot& operator=(const TextRunnerSnapshot&) = delete;
  TextRunnerSnapshot(TextRunnerSnapshot&&) = delete;
  TextRunnerSnapshot& operator=(TextRunnerSnapshot&&) = delete;
};

struct TextRunnerAdvance {
  std::reference_wrapper<TextRunnerState> state;
  TextRunnerToken token{0};
  std::exception_ptr* failure{nullptr};
};

struct TextRunnerDecode {
  std::reference_wrapper<TextRunnerState> state;
  std::size_t max_tokens{0};
  std::reference_wrapper<sampling::SamplerState> sampler;
};

/// Coarse text-model adapter used by cache, scheduling, and HTTP layers.
///
/// Prefill and Advance are the only model execution work units. SelectNext
/// exposes the already-computed frontier token separately so streaming does
/// not wait for the following decode forward pass.
class TextModelRunner {
public:
  TextModelRunner() = default;
  virtual ~TextModelRunner() = default;

  TextModelRunner(const TextModelRunner&) = delete;
  TextModelRunner& operator=(const TextModelRunner&) = delete;
  TextModelRunner(TextModelRunner&&) = delete;
  TextModelRunner& operator=(TextModelRunner&&) = delete;

  [[nodiscard]] virtual TextRunnerDescriptor Descriptor() const = 0;
  [[nodiscard]] virtual TextRunnerResourceClaim ResourceClaim() const = 0;
  [[nodiscard]] virtual std::vector<TextExecutionPlan> SupportedPlans()
      const = 0;

  /// Prepared before serving and shared by all constrained requests.
  /// A null vocabulary means the runner does not support structured output.
  [[nodiscard]] virtual std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const {
    return nullptr;
  }
  [[nodiscard]] virtual sampling::JsonConstraint::ToolFormat ToolFormat()
      const {
    return sampling::JsonConstraint::ToolFormat::kJson;
  }
  /// Reuse a compiled grammar bound to this runner's vocabulary.
  [[nodiscard]] std::shared_ptr<const sampling::TokenConstraint> BindConstraint(
      std::shared_ptr<const sampling::JsonConstraint> grammar) const;
  /// Builds the vocabulary BindConstraint uses now rather than in the first
  /// constrained request; later calls reuse it. Returns its token count.
  /// Returns zero when the runner has no structured output. Build failures
  /// propagate to the caller.
  std::size_t PrepareConstraintVocabulary() const;

  /// Reasoning tags for a token budget, as llama.cpp's chat format for the
  /// model defines them: `forced_end` is emitted once the budget is spent and
  /// any of `ends` closes reasoning early.
  struct ReasoningTags {
    std::string_view start;
    std::string_view forced_end;
    std::vector<std::string_view> ends;
  };
  [[nodiscard]] virtual ReasoningTags ReasoningBudgetTags() const {
    return {"<think>", "</think>", {"</think>"}};
  }
  /// Zero when the vocabulary cannot be enumerated.
  [[nodiscard]] virtual std::uint32_t VocabularySize() const { return 0; }
  /// A budget for a prompt that opens reasoning, or null when the tags are not
  /// single tokens in this vocabulary.
  [[nodiscard]] std::shared_ptr<const sampling::ReasoningBudget>
  BuildReasoningBudget(std::size_t tokens,
                       std::span<const TextRunnerToken> prompt) const;

  [[nodiscard]] virtual std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const = 0;
  /// Tokenizes control text such as `</think>` as the template renders it.
  [[nodiscard]] virtual std::vector<TextRunnerToken> TokenizeControl(
      std::string_view text) const {
    return Tokenize(text);
  }
  [[nodiscard]] virtual std::optional<std::vector<TextRunnerToken>>
  RenderAndTokenize(const ChatRequest& request) const = 0;
  /// Where conversations under the same leading system messages diverge in
  /// `tokens` (the prepared prompt of `request`): the first token that a
  /// probe with those system messages and an empty user turn renders
  /// differently. Zero without leading system messages, with nothing after
  /// them, or with images in them.
  [[nodiscard]] std::size_t SystemPrefixTokens(
      const ChatRequest& request, std::span<const TextRunnerToken> tokens) const;

  [[nodiscard]] virtual std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const {
    for (const auto& message : request.messages) {
      if (!message.images.empty()) {
        throw std::invalid_argument("model does not support image input");
      }
    }
    auto tokens = RenderAndTokenize(request);
    if (!tokens)
      return std::nullopt;
    return TextPreparedPrompt{std::move(*tokens), {}};
  }
  /// Size a prompt without execution-context admission or inference. Image
  /// runners override this to include their model's visual token expansion.
  [[nodiscard]] virtual std::optional<std::size_t> CountPromptTokens(
      const ChatRequest& request) const {
    // Use the text-only preparation here, including its capability check,
    // without constructing a derived runner's cache metadata.
    const auto prompt = TextModelRunner::PreparePrompt(request);
    if (!prompt)
      return std::nullopt;
    return prompt->tokens.size();
  }
  /// Called before cache restoration, prefill or selection, including
  /// a complete prefix hit. Must replace, never inherit, the previous request.
  virtual void SetPromptContext(
      TextRunnerState&,
      std::shared_ptr<const TextPromptContext> context) const {
    if (context != nullptr) {
      throw std::invalid_argument("model does not support this prompt context");
    }
  }
  [[nodiscard]] virtual TextGenerationBackend::InitialOutputState
  InitialOutputState(const ChatRequest&) const {
    return TextGenerationBackend::InitialOutputState::kAuto;
  }
  /// Reasoning delimiters and tool-call syntax of generated text.
  [[nodiscard]] virtual TextGenerationBackend::OutputMarkup Markup() const {
    return {};
  }
  [[nodiscard]] virtual std::string Decode(
      std::span<const TextRunnerToken> tokens) const = 0;

  [[nodiscard]] virtual std::unique_ptr<TextRunnerState> CreateState()
      const = 0;

  [[nodiscard]] virtual bool UsesComponentCache() const { return false; }
  [[nodiscard]] virtual std::unique_ptr<ComponentCacheResources>
  CreateComponentCacheResources(cache::ResourceLedger&, std::size_t) const;
  [[nodiscard]] virtual std::unique_ptr<TextRunnerState> BindComponentState(
      cache::Adapter&, cache::Slot&) const;
  virtual void ReconcileComponentState(TextRunnerState&, std::size_t) const;

  /// Optional first-token preview without advancing model state. The caller
  /// supplies a copy of the sampler; normal decoding must reproduce the token.
  /// This lets serving publish a token before capturing its prompt snapshot.
  [[nodiscard]] virtual std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState&, sampling::SamplerState&) const {
    return std::nullopt;
  }
  /// Reconciles model-private metadata after an exact mutable or restored
  /// prefix is leased. The default runner state needs no additional work.
  virtual void PreparePrefixReuse(
      TextRunnerState& state, std::span<const TextRunnerToken> prefix) const {
    (void)state;
    (void)prefix;
  }
  /// Tells a request state that its next execution will use a multi-request
  /// batch. Runners with request-local speculative state may discard work that
  /// the batched path cannot consume.
  virtual void PrepareBatchExecution(TextRunnerState& state) const {
    (void)state;
  }
  /// The prompt tokens after the next Prefill's frontier, set before every
  /// prefill step (empty when the frontier ends the prompt), and where later
  /// steps stop inside them (token counts, ascending). A step that stops at a
  /// snapshot boundary must still leave the state exactly at that boundary;
  /// a runner may compute these tokens alongside and adopt the result when a
  /// following step stops at one of `stops` or at their end.
  virtual void SetPromptLookahead(TextRunnerState& state,
                                  std::span<const TextRunnerToken> tokens,
                                  std::span<const std::size_t> stops) const {
    (void)state;
    (void)tokens;
    (void)stops;
  }
  /// Processes at most max_input_tokens, yielding after one model-owned
  /// chunk even when the scheduler grants the whole remaining prompt.
  [[nodiscard]] virtual TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const = 0;
  /// Returns a claim only when this chunk can capture the boundary and keep
  /// prefilling beyond it. Admission happens before allocating checkpoint
  /// state.
  [[nodiscard]] virtual std::optional<std::size_t> PrefillCheckpointBytes(
      const TextRunnerState&, std::span<const TextRunnerToken>, std::size_t,
      std::size_t, std::size_t) const {
    return std::nullopt;
  }
  /// Component runners invoke the consumer synchronously while the coherent
  /// boundary is readable. It must drain captures before execution resumes.
  /// Native snapshot runners leave the consumer empty.
  [[nodiscard]] virtual TextPrefillStep PrefillThrough(
      TextRunnerState&, std::span<const TextRunnerToken>, std::size_t,
      std::size_t, std::size_t, std::unique_ptr<TextRunnerSnapshot>*,
      const std::function<std::unique_ptr<TextRunnerSnapshot>()>& = {}) const {
    throw std::logic_error("runner does not support in-pass checkpoints");
  }
  [[nodiscard]] virtual TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const = 0;
  virtual void Advance(TextRunnerState& state, TextRunnerToken token) const = 0;
  [[nodiscard]] virtual TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const;
  [[nodiscard]] virtual std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const;
  virtual void AdvanceBatch(std::span<const TextRunnerAdvance> advances) const;
  [[nodiscard]] virtual std::size_t CheckpointPosition(
      const TextRunnerState& state) const = 0;
  /// Retain a safe executed frontier when cancellation interrupts publication
  /// of a completed speculative block. Called with cancellation checks cleared.
  virtual void PrepareCancellation(TextRunnerState&) const {}
  enum class DeviceProbeStatus { kUsable, kPending, kLost };
  /// Submit or poll one private device probe without waiting. Called only by
  /// the scheduler between work units. Pending work is reused, never queued
  /// again.
  [[nodiscard]] virtual DeviceProbeStatus PollDevice() const {
    return DeviceProbeStatus::kUsable;
  }
  /// Bounded check, run only after a failed work unit, that the execution
  /// device still accepts work. False means the context is permanently lost;
  /// a probe that is still pending at its bound reports true.
  [[nodiscard]] virtual bool DeviceUsable() const { return true; }

  /// Captures an immutable exact continuation at CheckpointPosition(state).
  ///
  /// The default implementations fail explicitly for runners that do not
  /// advertise the corresponding capabilities.
  [[nodiscard]] virtual std::size_t SnapshotPayloadBytes(
      const TextRunnerState& state) const;
  /// May run on a capture worker while this state is frozen. Must not mutate
  /// shared execution scratch; other sessions may execute concurrently.
  [[nodiscard]] virtual std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const;
  /// Complete state before returning so export cannot introduce device copies
  /// while the session resumes prefill. Defaults to the ordinary snapshot.
  [[nodiscard]] virtual std::unique_ptr<TextRunnerSnapshot>
  SnapshotForPersistence(const TextRunnerState& state) const;

  virtual void RestoreOrFork(TextRunnerState& state,
                             const TextRunnerSnapshot& snapshot) const;

  /// Returns the exact serialized bytes required by a persistent snapshot.
  ///
  /// Persistent serialization is separate from the in-memory snapshot layout:
  /// a provider may retain GPU-private state in RAM while exposing a compact,
  /// versioned disk representation.
  [[nodiscard]] virtual std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& snapshot) const;

  /// Checkpoint sizes at max_context, known before any state exists. The pool
  /// refuses disk budgets that cannot hold one such checkpoint and reports a
  /// smaller retained snapshot budget.
  /// Missing means the runner cannot bound its checkpoints; the pool records
  /// that the floor was not checked.
  [[nodiscard]] virtual std::optional<TextRunnerCheckpointBound>
  MaxContextCheckpointBytes() const;

  /// Serializes one immutable snapshot into exactly-sized common-store staging.
  ///
  /// Returns the number of initialized bytes. It must equal
  /// PersistentSnapshotPayloadBytes(snapshot).
  [[nodiscard]] virtual std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const;

  using SnapshotSink = std::function<void(std::span<const std::uint8_t>)>;
  /// Reads only immutable snapshot storage. May run on the persistence worker.
  /// Host-backed providers stream directly without another payload allocation.
  virtual void StreamPersistentSnapshot(const TextRunnerSnapshot& snapshot,
                                        const SnapshotSink& sink) const;

  /// Restores a version-compatible serialized payload into an existing state.
  virtual void RestorePersistentSnapshot(
      TextRunnerState& state, std::span<const std::uint8_t> payload) const;

  /// Persists a live state in place, without a snapshot copy. The result
  /// holds what is small (logits, draft state) and streams the rest from the
  /// state, which must not change until `stream` returns. The bytes equal
  /// SerializePersistentSnapshot(Snapshot(state)). Missing: the runner keeps
  /// no exact boundary in place, so callers take a snapshot instead.
  struct PersistentStateWriter {
    std::size_t payload_bytes{0};
    std::function<void(const SnapshotSink&)> stream;
    /// Full row blocks the payload leaves out (PersistentRowBlockBytes), and
    /// how to stream block i of them from the state in place.
    std::size_t row_blocks{0};
    std::function<void(std::size_t, const SnapshotSink&)> stream_row_block;
  };
  [[nodiscard]] virtual std::optional<PersistentStateWriter>
  PreparePersistentState(const TextRunnerState&) const {
    return std::nullopt;
  }
  /// True when PreparePersistentState supports every state of this runner.
  [[nodiscard]] virtual bool PersistsStateInPlace() const { return false; }

  /// Paged model rows: states take fixed-size blocks of rows from one pool
  /// shared by all states as they write, instead of a whole context each.
  /// All zero for a runner whose states preallocate their context.
  struct RowBlockPool {
    std::size_t total_blocks{0};
    std::size_t free_blocks{0};
    std::size_t block_rows{0};
  };
  [[nodiscard]] virtual RowBlockPool RowBlocks() const { return {}; }
  /// Holds back the blocks `state` needs to write rows up to `rows` beyond
  /// those it holds, plus copy-on-write room for shared blocks. False, with
  /// nothing held, when the pool lacks them.
  [[nodiscard]] virtual bool ReserveRowBlocks(TextRunnerState&,
                                              std::size_t) const {
    return true;
  }
  /// Returns what ReserveRowBlocks held back and the state has not used.
  virtual void ReleaseRowBlocks(TextRunnerState&) const noexcept {}
  /// Blocks one admission may take beyond its rows: copy-on-write of shared
  /// partial blocks across its captures and restores.
  static constexpr std::size_t kRowBlockCopyMargin = 8;

  /// Fills the whole span with the next payload bytes. Throws on read errors.
  using SnapshotSource = std::function<void(std::span<std::uint8_t>)>;
  /// RestorePersistentSnapshot from a stream of exactly `payload_bytes`.
  /// The caller verifies the stream's checksum afterwards and invalidates the
  /// state on a mismatch. The default reads the whole payload first;
  /// device-backed providers override it to avoid that host copy.
  virtual void RestorePersistentSnapshotStream(
      TextRunnerState& state, std::size_t payload_bytes,
      const SnapshotSource& source) const;

  /// Rows persisted in blocks: a runner whose persistent payload leaves its
  /// full row blocks out has the store keep each block as a separate,
  /// content-addressed file shared by every record with the same prefix.
  /// Bytes of one block's rows and tokens per block; zero when the payload
  /// holds every row.
  [[nodiscard]] virtual std::size_t PersistentRowBlockBytes() const {
    return 0;
  }
  [[nodiscard]] virtual std::size_t PersistentRowBlockRows() const {
    return 0;
  }
  /// Full blocks the persistent form of `snapshot` leaves out.
  [[nodiscard]] virtual std::size_t PersistentRowBlocks(
      const TextRunnerSnapshot&) const {
    return 0;
  }
  virtual void StreamPersistentRowBlock(const TextRunnerSnapshot& snapshot,
                                        std::size_t block,
                                        const SnapshotSink& sink) const;
  /// Full blocks a state restored by RestorePersistentSnapshotStream still
  /// needs; RestorePersistentRowBlock fills them in order.
  [[nodiscard]] virtual std::size_t RestoredRowBlocks(
      const TextRunnerState&) const {
    return 0;
  }
  /// Full blocks of a released state, streamed in place (block i of
  /// StateRowBlocks), so they reach disk before the state's record does.
  [[nodiscard]] virtual std::size_t StateRowBlocks(
      const TextRunnerState&) const {
    return 0;
  }
  virtual void StreamStateRowBlock(const TextRunnerState& state,
                                   std::size_t block,
                                   const SnapshotSink& sink) const;
  virtual void RestorePersistentRowBlock(TextRunnerState& state,
                                         std::size_t block,
                                         const SnapshotSource& source) const;

  /// Paged rows by pool block, for the resident block index: the pool block
  /// holding full block `block` of a state or snapshot. Nullopt when rows
  /// are not paged.
  [[nodiscard]] virtual std::optional<std::uint32_t> StateRowBlockId(
      const TextRunnerState&, std::size_t) const {
    return std::nullopt;
  }
  [[nodiscard]] virtual std::optional<std::uint32_t> SnapshotRowBlockId(
      const TextRunnerSnapshot&, std::size_t) const {
    return std::nullopt;
  }
  /// Points full block `block` of an idle `state` at pool block `id`, which
  /// holds the same rows (equal chain keys), taking one reference to it and
  /// dropping the block the state held.
  virtual void ShareRowBlock(TextRunnerState&, std::size_t,
                             std::uint32_t) const {
    throw std::logic_error("text runner rows are not paged");
  }
  virtual void RetainRowBlock(std::uint32_t) const {
    throw std::logic_error("text runner rows are not paged");
  }
  virtual void ReleaseRowBlock(std::uint32_t) const noexcept {}
  [[nodiscard]] virtual std::size_t RowBlockReferences(std::uint32_t) const {
    return 0;
  }

private:
  mutable std::mutex constraint_mutex_;
  mutable std::shared_ptr<const sampling::ConstraintVocabulary>
      constraint_vocabulary_;
  mutable std::vector<std::shared_ptr<const sampling::TokenConstraint>>
      constraints_;
  mutable std::shared_ptr<const std::vector<bool>> partial_utf8_;
};

/// Bounded pool of opaque runner states with exact-prefix continuation reuse.
class TextRunnerPool {
public:
  using CancellationCheck = std::function<bool()>;

  class Request {
  public:
    struct CommitMetrics {
      std::optional<ComponentCacheMetrics> component_cache;
      std::size_t snapshot_bytes{0};
      /// Capture work through worker completion, excluding later collection.
      double snapshot_ms{0.0};
      std::size_t disk_queued_bytes{0};
      double disk_enqueue_ms{0.0};
      /// Shared-prefix snapshots queued for persistence during prefill.
      std::size_t shared_prefix_snapshots{0};
      std::size_t shared_prefix_bytes{0};
      std::size_t shared_prefix_failures{0};
      double shared_prefix_ms{0.0};
    };

    Request();
    ~Request();

    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;
    Request(Request&&) noexcept;
    Request& operator=(Request&&) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] bool cache_hit() const noexcept;
    [[nodiscard]] ContinuationLookup cache_lookup() const noexcept;
    [[nodiscard]] std::size_t cached_prompt_tokens() const noexcept;
    [[nodiscard]] std::size_t cache_restore_bytes() const noexcept;
    [[nodiscard]] double cache_restore_ms() const noexcept;
    [[nodiscard]] bool cache_disk_hit() const noexcept;
    [[nodiscard]] std::optional<ComponentCacheMetrics> component_cache_metrics()
        const;
    [[nodiscard]] std::size_t prompt_tokens() const noexcept;
    [[nodiscard]] bool prefill_complete() const noexcept;
    [[nodiscard]] std::span<const TextRunnerToken> prompt() const noexcept;
    /// Prompt tokens already in model state, restored or prefilled.
    [[nodiscard]] std::size_t prefill_position() const noexcept;
    [[nodiscard]] std::span<const std::uint8_t> input_identity(
        std::size_t token_count) const;

    /// Lets another request reuse this prompt's first common_tokens.
    ///
    /// Returns the position, at most common_tokens, where this request will
    /// publish a RAM checkpoint before prefilling past it, reusing a planned
    /// checkpoint within kSharedCheckpointSlack tokens. Zero when the runner
    /// cannot snapshot or this request is no longer before that position.
    /// Other requests depend on it, waiting now or arriving later, so it is
    /// retained with continuation priority rather than as an optional copy.
    [[nodiscard]] std::size_t ShareCheckpoint(std::size_t common_tokens);
    /// A planned checkpoint this close to the shared position is cheaper to
    /// use than capturing another one: followers prefill the gap themselves.
    static constexpr std::size_t kSharedCheckpointSlack = 64;

    /// Holds the row blocks this request can write: its prompt and up to
    /// `max_new_tokens` more, capped at the context. Throws
    /// std::length_error when the row pool lacks them.
    void ReserveRows(std::size_t max_new_tokens);

    void PrepareBatchExecution();
    [[nodiscard]] TextPrefillStep Prefill(std::size_t max_input_tokens);
    [[nodiscard]] TextDecodeSelection SelectNext();
    void Advance();
    [[nodiscard]] TextDecodeStep DecodeStep(std::size_t max_tokens);

    /// Publishes a reusable continuation boundary.
    ///
    /// Snapshot-capable runners keep the immutable prompt frontier and the
    /// final live state. Mutable-state runners publish their reported
    /// checkpoint.
    CommitMetrics Commit();
    /// Release a cancelled request, retaining only completed model work and
    /// an already captured prompt snapshot. Does not advance or sample tokens.
    CommitMetrics Cancel() noexcept;
    [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken();
    void CapturePromptSnapshot();
    /// Starts capture and polls completion without blocking other sessions.
    [[nodiscard]] bool PreparePromptSnapshot();
    [[nodiscard]] bool SnapshotPending() const;
    /// Waits up to `timeout` for a pending snapshot; true once none is.
    [[nodiscard]] bool AwaitSnapshot(
        std::chrono::steady_clock::duration timeout) const;
    void Invalidate() noexcept;

  private:
    friend class TextRunnerPool;
    struct Impl;

    explicit Request(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
  };

  TextRunnerPool(
      std::shared_ptr<TextModelRunner> runner, std::size_t state_count,
      std::optional<TextRunnerDiskCacheOptions> disk_cache = std::nullopt,
      TextRunnerRamCacheOptions ram_cache = {});
  ~TextRunnerPool();

  TextRunnerPool(const TextRunnerPool&) = delete;
  TextRunnerPool& operator=(const TextRunnerPool&) = delete;
  TextRunnerPool(TextRunnerPool&&) = delete;
  TextRunnerPool& operator=(TextRunnerPool&&) = delete;

  [[nodiscard]] const TextModelRunner& runner() const noexcept;
  [[nodiscard]] std::size_t capacity() const noexcept;
  /// TextRunnerRamCacheOptions::shared_prefix_min_tokens.
  [[nodiscard]] std::size_t shared_prefix_min_tokens() const noexcept;
  /// False while admitting `prompt` would overwrite another conversation's
  /// frontier that is not on disk yet. That frontier is then being written
  /// in place; the caller keeps the request queued and retries, so active
  /// requests are not blocked by the write.
  /// With paged rows, also false while the row pool cannot hold `prompt`
  /// and `max_new_tokens`, after dropping retained rows not needed for it.
  /// A request that does not reuse cached prompt state (`reuse_prompt`
  /// false) is charged for every block and keeps no retained prefix.
  [[nodiscard]] bool AdmissionReady(std::span<const TextRunnerToken> prompt,
                                    std::size_t max_new_tokens = 0,
                                    bool reuse_prompt = true) const;
  /// The resident row block index; nullopt when there is none (no row
  /// block store or no paged rows).
  [[nodiscard]] std::optional<ResidentBlockIndex::Stats> ResidentBlockStats()
      const;
  /// Blocks the worst admission of `rows` can take; zero without paging.
  [[nodiscard]] std::size_t AdmissionRowBlocks(std::size_t rows) const;
  [[nodiscard]] TextExecutionPlan SelectDecodePlan(
      std::size_t ready_requests) const;
  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<Request*> requests, std::span<const std::size_t> max_tokens,
      const TextExecutionPlan& plan);
  std::vector<std::exception_ptr> AdvanceBatch(std::span<Request*> requests,
                                               const TextExecutionPlan& plan);
  [[nodiscard]] Request Acquire(
      std::vector<TextRunnerToken> prompt,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled = {},
      std::shared_ptr<const TextPromptContext> context = {},
      bool reuse_prompt = true, std::size_t cache_prefix_tokens = 0,
      bool stop_at_eos = true, std::size_t system_prefix_tokens = 0);
  [[nodiscard]] Request Acquire(std::vector<TextRunnerToken> prompt,
                                const CancellationCheck& is_cancelled = {});
  /// Longest RAM-retained prefix of prompt that Acquire could reuse, without
  /// leasing a state. An upper bound: Acquire may still prefer a shorter one.
  [[nodiscard]] std::size_t CachedPrefixTokens(
      std::span<const TextRunnerToken> prompt,
      const TextPromptContext* context) const;
  /// A disk job is still writing a prefix of prompt much deeper than any
  /// restorable checkpoint; admission may wait for it instead of prefilling.
  [[nodiscard]] std::optional<TextPendingPublication> PendingPublication(
      std::span<const TextRunnerToken> prompt, const TextPromptContext* context,
      std::uint64_t checkpoint_id = 0) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Build output constraints and attach server-authored instructions as framing,
/// preserving client message content and the model's native prompt layout.
[[nodiscard]] std::optional<ChatRequest> ConstrainChatRequest(
    const ChatRequest& request, const TextModelRunner& runner,
    sampling::SamplingConfig* sampling,
    std::optional<sampling::JsonConstraint::ToolFormat>* tool_format = nullptr);

}  // namespace gufo::server

#endif  // GUFO_SERVER_TEXT_MODEL_RUNNER_HPP_
