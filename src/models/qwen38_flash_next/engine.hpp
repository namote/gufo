#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ENGINE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ENGINE_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/hip/kv_block_layout.hpp"
#include "src/core/sampling.hpp"
#include "src/core/session_mode.hpp"
#include "src/core/speculative/prompt_lookup.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen/vision/encoder.hpp"
#include "src/models/qwen/vision/prompt.hpp"
#include "src/models/qwen38_flash_next/abliteration.hpp"
#include "src/models/qwen38_flash_next/config.hpp"
#include "src/models/qwen38_flash_next/draft_vocab.hpp"
#include "src/models/qwen38_flash_next/dynamic_abliteration.hpp"
#include "src/models/qwen38_flash_next/hc_mode.hpp"
#include "src/models/qwen38_flash_next/mtp_policy.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"

namespace gufo::hip {
class KvBlockPool;
}

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::qwen38_flash_next {

struct ModelWeights;
struct MtpWeights;
struct MtpCandidateLogits;
class ContinuationAdapter;
class ContinuationHooks;
namespace rocm {
class DeviceModel;
class Executor;
class Session;
class SnapshotState;
struct PrefillCheckpoint;
}  // namespace rocm

struct ModelOptions {
  std::uint32_t max_context = 4096;
  /// Optional MTP draft sidecar (`mtp-...-shared-*.gguf`). Empty leaves
  /// speculative decoding off.
  std::string mtp_model_path;
  /// Maximum proposals per cycle; acceptance history selects the length.
  std::uint32_t max_draft_tokens = kMaxMtpDraftTokens;
  std::string vision_model_path;
  /// Fixed serving capacity used by the calibrated MTP cost model. Keeping
  /// it independent of scheduler timing preserves seeded request replay.
  std::uint32_t decode_concurrency = 1;
  /// Static YaRN scaling, derived by the caller (serve, bench) from the
  /// requested context via `RopeScalingForContext`. Empty leaves the
  /// artifact's own setting, which is always off: no GGUF metadata key sets
  /// it today.
  std::optional<RopeScaling> rope_scaling{};
  /// Trunk attention K/V as KVarN k4v4 records (`--kv-cache-type
  /// kvarn-k4v4`, plan/exl3/archive/F1-KVARN-DESIGN.md); the MTP layer stays
  /// FP16.
  bool kvarn = false;
  int kvarn_bits = 4;
  /// Profile files (`--abliteration-profiles DIR`, listed by
  /// qwen::ListAbliterationProfiles). Empty loads none: nothing is discovered
  /// beside the model. Loading enables none; a profile with a `trigger`
  /// turns ON only in sessions whose detection selects it.
  std::vector<std::string> abliteration_profiles;
  /// Reserve device rows so profiles can be added, replaced and removed
  /// while serving (ReplaceAbliteration; `--allow-profile-updates`): rows for
  /// twice qwen::kAbliterationMaxProfiles profiles of the largest geometry
  /// (2 kinds x layers x qwen::kAbliterationMaxDirections rows each), so every
  /// profile can be replaced while requests still use the old one, plus the
  /// union rows and every detector column (kDetectorColumns).
  bool abliteration_updates = false;
  /// `--hc-mode`: the hyper-connection mixers' weights. q8-calibrated needs a
  /// native EXL3 pack with a valid sidecar; loading fails otherwise.
  HcMode hc_mode = kDefaultHcMode;
};

class Session;
class SessionSnapshot;

/// Gufo-owned API over the ROCm runtime: one resident model, any number of
/// sessions with shared projection batches and independent context state.
class Model final : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      core::SessionMode mode, std::uint32_t max_context,
      std::string* error_msg = nullptr,
      std::shared_ptr<hip::KvBlockPool> pool = {});
  [[nodiscard]] hip::KvBlockGeometry KvGeometry(core::SessionMode mode,
                                                std::uint32_t context) const;

  [[nodiscard]] std::vector<std::int32_t> Tokenize(std::string_view text) const;
  [[nodiscard]] std::string Decode(std::span<const std::int32_t> tokens) const;
  [[nodiscard]] std::string TokenText(std::int32_t token) const;
  [[nodiscard]] std::int32_t EosToken() const noexcept;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept;
  [[nodiscard]] std::uint32_t PrefillCapacity() const noexcept;
  [[nodiscard]] std::uint32_t PrefillThroughCapacity() const noexcept;
  /// Rows a prefill pass may run past an in-pass checkpoint's boundary.
  [[nodiscard]] std::uint32_t PrefillCheckpointTail() const noexcept;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return options_.max_context;
  }
  [[nodiscard]] bool HasMtp() const noexcept;
  /// The loaded profiles now, in load (file name) order.
  [[nodiscard]] AbliterationProfileSet AbliterationProfiles() const;
  /// The loaded profiles now, as a request snapshots them: a later
  /// ReplaceAbliteration never changes a table already handed out. Null when
  /// none was ever loaded.
  [[nodiscard]] std::shared_ptr<const AbliterationTable> Abliteration() const;
  /// A request's detection state over every profile of `table` with a
  /// `trigger` (Session::ConfigureDynamicAbliteration); null when it has
  /// none.
  [[nodiscard]] static std::shared_ptr<DynamicAbliteration>
  StartDynamicAbliteration(std::shared_ptr<const AbliterationTable> table);
  /// The settings of the named loaded profiles with their defaults (load
  /// order; combined when several), as an explicit request naming them.
  [[nodiscard]] std::shared_ptr<const AbliterationSettings>
  ResolveAbliterationDefaults(std::span<const std::string> names) const;
  /// Throws std::invalid_argument when `settings` (several profiles of
  /// `table`, the current table when null) cannot run: PlanAbliteration's
  /// limits, or no free rows for its union bases right now.
  void CheckAbliterationCombination(
      const AbliterationSettings& settings,
      const AbliterationTable* table = nullptr) const;
  /// Whether ReplaceAbliteration may run (ModelOptions::abliteration_updates).
  [[nodiscard]] bool AbliterationUpdatable() const noexcept {
    return options_.abliteration_updates;
  }
  /// Makes `profiles` the loaded set: validates it as Load does
  /// (PlanAbliterationTriggers), uploads the directions of profiles not
  /// already loaded and publishes the new table. Running requests keep the
  /// table they were admitted with. Call on the thread running passes,
  /// between passes (the server's scheduler). Throws std::invalid_argument
  /// for an invalid set, std::length_error when the reserved device rows
  /// are held by older tables and std::runtime_error when an upload fails.
  void ReplaceAbliteration(const AbliterationProfileSet& profiles);
  [[nodiscard]] std::string ArtifactIdentity() const;
  /// Test statistic: batched MTP catch-ups whose cohort mixed two or more
  /// KV-only sessions with proposing ones (counted when received; models
  /// with a KV-only route then run them as their own sub-batch).
  [[nodiscard]] std::uint64_t MixedKvOnlyCohorts() const noexcept;
  [[nodiscard]] bool IsNativeExl3() const noexcept;
  /// Whether sessions keep trunk K/V as KVarN k4v4 (ModelOptions::kvarn).
  [[nodiscard]] bool Kvarn() const noexcept { return options_.kvarn; }
  [[nodiscard]] int KvarnBits() const noexcept { return options_.kvarn_bits; }
  /// The mixers' compute mode (ModelOptions::hc_mode).
  [[nodiscard]] HcMode hc_mode() const noexcept { return options_.hc_mode; }
  [[nodiscard]] std::uint32_t DecodeConcurrency() const noexcept {
    return options_.decode_concurrency;
  }
  [[nodiscard]] std::string ModelName() const;
  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] const tokenization::QwenTokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  /// Worst-case private device state, including the configured rollback cap.
  [[nodiscard]] std::size_t SessionBytes(core::SessionMode mode,
                                         std::uint32_t context) const noexcept;
  /// Text/rollback state only; prepared images receive a request-time claim.
  [[nodiscard]] std::size_t BaseSessionBytes(
      core::SessionMode mode, std::uint32_t context) const noexcept;
  [[nodiscard]] std::size_t DeferredScratchBytes() const;
  /// Widest copy verification (Executor::kWideRows).
  static constexpr std::uint32_t kWideCopyRows = 64;
  /// Caps wide copy verification (Executor::LimitCopyRows); call before any
  /// session is created.
  void LimitCopyRows(std::uint32_t rows) noexcept;
  /// Allocates the decode batch and wide verification scratch now
  /// (Executor::ReserveDecodeScratch); call after LimitCopyRows and before
  /// any session decodes.
  [[nodiscard]] bool ReserveDecodeScratch(std::string* error_msg);
  [[nodiscard]] const std::shared_ptr<qwen::vision::Encoder>& VisionEncoder()
      const noexcept {
    return vision_;
  }

private:
  Model() = default;

  ModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::shared_ptr<core::GgufReader> mtp_reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<MtpWeights> mtp_weights_;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer_;
  std::unique_ptr<NgramTable> ngram_;
  std::unique_ptr<rocm::DeviceModel> device_;
  std::unique_ptr<rocm::Executor> executor_;
  std::shared_ptr<qwen::vision::Encoder> vision_;
  MtpBatchController batch_policy_;
  /// MTP drafts may use the small head (draft_vocab.hpp).
  bool draft_subset_{false};
  /// Special tokens at the start of every session's draft table.
  std::uint32_t draft_specials_{0};
  /// Prefix and special tokens: always in the small head.
  [[nodiscard]] bool InDraftSubset(std::int32_t token) const;

  friend class Session;
  friend class ContinuationAdapter;
};

/// One conversation's context. Sync feeds a prompt (reusing whatever prefix
/// the session already holds), Evaluate appends one token, DecodeStep
/// runs a draft-verify cycle. The logits of the last token are kept.
class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Makes the session state equal to `prompt`: keeps the longest common
  /// prefix with the current tokens, reprocesses the rest. `next` may name
  /// the tokens the following Sync will append, so their n-gram rows can be
  /// read during this one; it never changes the result.
  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr,
                          std::span<const std::int32_t> next = {});
  /// Prefills through a stable text boundary in the same forward, returning
  /// a checkpoint of that boundary while the live session reaches prompt.
  [[nodiscard]] bool SyncThrough(std::span<const std::int32_t> prompt,
                                 std::uint32_t boundary,
                                 std::unique_ptr<SessionSnapshot>* checkpoint,
                                 std::string* error_msg = nullptr,
                                 double* capture_ms = nullptr);
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);
  /// Teacher forcing: the raw log-softmax (sampling::CollectLogprobs) of
  /// each of `tokens` given the session and the tokens before it. The first
  /// is scored on the current frontier logits; the rest are fed in prompt
  /// (kPrefill) passes of up to Executor::max_score_rows() rows that return
  /// every row's logits (Executor::ForwardScoreRows). The last token is
  /// scored but not fed. The session ends at the fed tokens, its draft block
  /// caught up as after a prefill.
  [[nodiscard]] bool ScoreTokens(std::span<const std::int32_t> tokens,
                                 std::vector<sampling::TokenLogprobs>* scored,
                                 std::string* error_msg = nullptr);
  /// Scores `tokens` after `prompt` as ScoreTokens does, first making the
  /// session hold `prompt` as Sync would (keeping a common prefix). The
  /// unfed prompt tail and the fed scored tokens share the same passes, so a
  /// short prompt and its answer take one pass; the first token is scored on
  /// the prompt's last row of that pass.
  [[nodiscard]] bool ScorePrompt(std::span<const std::int32_t> prompt,
                                 std::span<const std::int32_t> tokens,
                                 std::vector<sampling::TokenLogprobs>* scored,
                                 std::string* error_msg = nullptr);
  /// One request of ScoreBatch: ScorePrompt's arguments and its outcome.
  struct ScoreRequest {
    Session* session;
    std::span<const std::int32_t> prompt;
    std::span<const std::int32_t> tokens;
    std::vector<sampling::TokenLogprobs>* scored;
    std::string* error;
    bool* ok;
  };
  /// ScorePrompt for several sessions of one model. Requests whose unfed
  /// rows fit one pass share prompt passes of up to ScorePackRows() rows
  /// and Executor::kScorePackSessions sessions (Executor::ForwardScorePack):
  /// the expert weights stream once per pack. Their sessions skip the draft
  /// block and are left invalid (the next Sync resets them); scoring
  /// requests publish nothing. KVarN and image sessions, and longer
  /// sequences, take ScorePrompt. Each request reports its own outcome.
  static void ScoreBatch(std::span<const ScoreRequest> requests);
  /// Rows one scoring pack may hold.
  [[nodiscard]] static std::size_t ScorePackRows(const Model& model) noexcept;
  struct DecodeResult {
    std::vector<std::int32_t> tokens;
    bool stop{false};
    std::vector<sampling::TokenLogprobs> logprobs;
    std::optional<sampling::TokenLogprobs> terminal_logprobs;
  };
  /// Greedy decoding verifies deterministic drafts. Sampled decoding draws
  /// compact MTP proposals and uses target/draft rejection with a residual
  /// correction. Prompt lookup appends copied tokens to the chain as
  /// point-mass proposals. History contains only committed tokens; stochastic
  /// RNG consumption includes proposal and verification draws. A
  /// one-token budget or a model without MTP uses ordinary decoding.
  /// Benchmarks may continue past EOS by setting stop_at_eos to false.
  /// A greedy step prices its drafts as one of `occupancy` decoding requests,
  /// as DecodeBatch does for an unplanned cohort of that width. `residents`
  /// counts the requests the server holds, this one included (see
  /// TextRunnerState::SetResidentRequests); a copy verifies wide only when it
  /// is 1, so a long wide pass never delays a peer.
  [[nodiscard]] bool DecodeStep(std::size_t max_tokens,
                                sampling::SamplerState& sampler,
                                DecodeResult* result,
                                std::string* error_msg = nullptr,
                                bool stop_at_eos = true,
                                std::uint32_t occupancy = 1,
                                std::uint32_t residents = 1);
  struct BatchOutcome {
    bool completed{false};
    std::string error{};
  };
  struct DecodeRequest {
    Session* session;
    std::size_t max_tokens;
    sampling::SamplerState* sampler;
    DecodeResult* result;
    bool stop_at_eos{true};
    BatchOutcome* outcome{nullptr};
  };
  /// A false return may include completed peers. Inspect each outcome; only
  /// sessions whose persistent state was partly mutated become invalid.
  [[nodiscard]] static bool DecodeBatch(std::span<const DecodeRequest> requests,
                                        std::string* error_msg = nullptr);
  struct AdvanceRequest {
    Session* session;
    std::int32_t token;
    BatchOutcome* outcome{nullptr};
  };
  [[nodiscard]] static bool EvaluateBatch(
      std::span<const AdvanceRequest> requests,
      std::string* error_msg = nullptr);
  [[nodiscard]] std::span<const float> Logits() const noexcept {
    return valid_ ? std::span<const float>(logits_) : std::span<const float>{};
  }
  [[nodiscard]] std::uint32_t Position() const noexcept;
  [[nodiscard]] std::uint32_t ContextSize() const noexcept;
  [[nodiscard]] std::span<const std::int32_t> Tokens() const noexcept {
    return valid_ ? std::span<const std::int32_t>(tokens_)
                  : std::span<const std::int32_t>{};
  }
  void Reset();
  /// Clears the context for the next request on the same slot. A valid
  /// session keeps its MTP rollback rows and verifier graphs; an invalid one
  /// (a failed, stopped or cancelled request) releases all scratch like
  /// Reset().
  void ResetForReuse();
  /// Cancellation is checked at model, layer and snapshot-copy boundaries.
  void SetCancellationCheck(std::function<bool()> check);
  [[nodiscard]] bool IsValid() const noexcept { return valid_; }
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;
  struct MemoryUsage {
    std::size_t device_bytes{}, rollback_bytes{}, vision_bytes{}, graph_count{};
    /// Selected large host vectors; not all C++/HIP runtime allocations.
    std::size_t host_vector_bytes{};
  };
  [[nodiscard]] MemoryUsage Memory() const noexcept;
  void ConfigureVision(std::shared_ptr<const qwen::vision::Prompt> prompt);
  /// Null turns the projection OFF. Different settings never share token
  /// state: a change resets a non-empty session.
  /// `table` is the profile set the settings resolve against (the request's
  /// snapshot, Model::Abliteration); null uses the current one.
  void ConfigureAbliteration(
      std::shared_ptr<const AbliterationSettings> settings,
      std::shared_ptr<const AbliterationTable> table = nullptr);
  /// The session's effective settings while abliteration is ON, else null.
  [[nodiscard]] const AbliterationSettings* ActiveAbliteration() const;
  /// Dynamic abliteration (R2 A1) for the request that follows; null turns
  /// detection off. The session must be OFF. Its trunk passes then score the
  /// loaded detectors; before each decode pass (Evaluate, DecodeStep and the
  /// batched forms) the rows committed so far are checked, and the first
  /// crossing profile turns ON for every later pass without resetting the
  /// computed tokens (ActiveAbliteration then carries `dynamic_from`).
  /// `table` is the snapshot `state` was started from
  /// (Model::StartDynamicAbliteration); null uses the current one.
  void ConfigureDynamicAbliteration(
      std::shared_ptr<DynamicAbliteration> state,
      std::shared_ptr<const AbliterationTable> table = nullptr);
  /// The request's detection report after its last pass; empty without
  /// dynamic detection.
  [[nodiscard]] std::optional<DynamicAbliterationReport> DynamicReport();
  /// The request's prompt length: only a prefill row at its last token
  /// (DynamicAbliteration::SetPromptLength) can turn a profile ON.
  void SetDynamicPromptLength(std::size_t tokens) {
    if (dynamic_)
      dynamic_->SetPromptLength(static_cast<std::uint32_t>(tokens));
  }
  [[nodiscard]] std::vector<std::uint8_t> ContinuationIdentity(
      std::size_t tokens) const;
  using ActivationObserver =
      std::function<void(std::uint32_t, std::span<const float>)>;
  /// Standalone collector, no concurrent requests. The observer receives
  /// each layer's post-layer residual ([rows][hc_dim]) of every trunk pass:
  /// its last row, or with `all_rows` every row.
  void SetActivationObserver(ActivationObserver observer,
                             bool all_rows = false);
  /// The tokens the next pass of a Sync takes when `remaining` tokens follow
  /// the session's own (no checkpoint): Sync of the session's tokens plus
  /// exactly that many runs that one pass, so a caller may split one Sync
  /// into several at its pass boundaries without changing any pass (the
  /// in-server collector runs one pass per scheduler turn). Zero for zero.
  [[nodiscard]] std::size_t PrefillPassTokens(std::size_t remaining) const;
  /// A new request reusing cached context starts its own acceptance history.
  void ResetDraftPolicy();

  /// Diagnostic for draft-policy studies: the greedy MTP chain a speculative
  /// cycle anchored on `anchor` would propose, without verifying or
  /// committing anything. It runs the same catch-up, head choice and
  /// recursive draft steps as the greedy single-session cycle (no prompt
  /// lookup), then rewinds the draft block to the committed position. Each
  /// depth reports the draft token and its top candidate logits.
  struct DraftProbe {
    std::array<std::int32_t, kMaxMtpDraftTokens> tokens{};
    std::array<MtpCandidateLogits, kMaxMtpDraftTokens> candidates{};
    std::uint32_t depth{0};
    bool small_head{false};
  };
  [[nodiscard]] bool ProbeDraftChain(std::int32_t anchor, std::uint32_t depth,
                                     DraftProbe* probe,
                                     std::string* error_msg = nullptr);

  struct SpeculativeStats {
    std::uint64_t cycles{0};
    std::uint64_t drafted{0};
    std::uint64_t accepted{0};
    /// Cycles whose draft scored the small head (draft_vocab.hpp).
    std::uint64_t small_head_cycles{0};
    /// Of drafted/accepted: proposals copied by prompt lookup.
    std::uint64_t lookup{0};
    std::uint64_t lookup_accepted{0};
  };
  [[nodiscard]] const SpeculativeStats& Statistics() const noexcept {
    return stats_;
  }
  /// PLE rows gathered by this execution slot since creation; zero without
  /// a PLE table.
  [[nodiscard]] NgramTotals NgramRowTotals() const noexcept;

  /// Compatibility version; bump on payload or inference arithmetic changes.
  static constexpr std::uint32_t kSnapshotPayloadVersion = 19;
  /// Bytes a snapshot of the current context occupies.
  [[nodiscard]] std::uint64_t SnapshotBytes() const;
  [[nodiscard]] std::uint64_t PrefillCheckpointBytes(
      std::uint32_t position) const;
  /// Captures the whole context and speculative length controller. Mutable
  /// state is copied immediately; append-only cache rows borrow this session
  /// until a rewind, reset, destruction, fork or byte reader needs them.
  enum class SnapshotMode { kBorrowed, kMaterialized };
  [[nodiscard]] std::unique_ptr<SessionSnapshot> SaveSnapshot(
      std::string* error_msg = nullptr,
      SnapshotMode mode = SnapshotMode::kBorrowed) const;
  [[nodiscard]] bool OwnsSnapshot(
      const SessionSnapshot& snapshot) const noexcept;
  /// Replaces this session's context with a snapshot of the same model.
  /// Image snapshots require matching consumed images via ConfigureVision;
  /// the attached prompt may append future images. Pixels are not serialized.
  /// Text snapshots clear images; reattach the request after restoring.
  [[nodiscard]] bool RestoreSnapshot(const SessionSnapshot& snapshot,
                                     std::string* error_msg = nullptr);
  [[nodiscard]] bool RestoreSnapshot(std::span<const std::uint8_t> payload,
                                     std::string* error_msg = nullptr);

private:
  friend class Model;
  friend class ContinuationAdapter;
  Session(std::shared_ptr<Model> model, std::unique_ptr<rocm::Session> session);
  void ResetImpl(bool reuse_mtp);
  /// Whether a reset for slot reuse may keep this session's scratch.
  [[nodiscard]] bool ScratchReusable() const noexcept;

  using PrefillCapture = std::function<void(const rocm::PrefillCheckpoint&)>;
  bool SyncThroughComponents(std::span<const std::int32_t>, std::uint32_t,
                             const PrefillCapture&, std::string*);

  /// The tokens Feed's next pass takes when `remaining` follow tokens_.
  [[nodiscard]] std::size_t PassTokens(std::size_t remaining,
                                       bool prefill) const;
  bool Feed(std::span<const std::int32_t> tokens, std::string* error_msg,
            bool prefill = false, std::uint32_t boundary = 0,
            std::unique_ptr<SessionSnapshot>* checkpoint = nullptr,
            double* capture_ms = nullptr,
            std::span<const std::int32_t> after = {},
            const PrefillCapture& capture = {});
  struct ScorePlan {
    std::vector<std::int32_t> fed;  ///< unfed prompt, then tokens but the last
    std::size_t first{0};           ///< first global row that scores a token
  };
  /// ScorePrompt's checks and prefix sync; scores the first token on the
  /// frontier when the session already holds the whole prompt.
  bool BeginScore(std::span<const std::int32_t> prompt,
                  std::span<const std::int32_t> tokens,
                  std::vector<sampling::TokenLogprobs>* scored, ScorePlan* plan,
                  std::string* error_msg);
  bool ScorePromptImpl(std::span<const std::int32_t> prompt,
                       std::span<const std::int32_t> tokens,
                       std::vector<sampling::TokenLogprobs>* scored,
                       std::string* error_msg);
  bool SyncImpl(std::span<const std::int32_t> prompt, std::string* error_msg,
                std::uint32_t boundary,
                std::unique_ptr<SessionSnapshot>* checkpoint,
                double* capture_ms = nullptr,
                std::span<const std::int32_t> next = {},
                const PrefillCapture& capture = {});
  std::unique_ptr<SessionSnapshot> SaveSnapshotImpl(
      const rocm::Session& state, std::span<const std::int32_t> tokens,
      std::span<const float> logits, std::uint32_t hidden_rows,
      std::string* error_msg, SnapshotMode mode) const;
  bool RestoreSnapshotImpl(std::span<const std::uint8_t> payload,
                           const rocm::SnapshotState* deferred,
                           std::string* error_msg);
  /// Trunk rows the draft block may still read: [hidden_base_, size).
  [[nodiscard]] std::uint32_t KeptHiddenRows() const noexcept;
  /// With `candidates`, a proposing catch-up reports them instead of the
  /// greedy token unless `token_with_candidates` asks for both.
  bool DraftCatchUp(std::int32_t next_token, bool propose,
                    std::string* error_msg,
                    MtpCandidateLogits* candidates = nullptr,
                    bool draft_subset = false,
                    bool token_with_candidates = false);
  bool DraftReplay(std::int32_t next_token, std::vector<std::int32_t>* replay,
                   std::int32_t* hidden_row, std::string* error_msg) const;
  static bool DraftCatchUpBatch(std::span<const AdvanceRequest> requests,
                                std::string* error_msg,
                                std::span<const std::uint8_t> propose = {});
  template<class Request>
  static bool RunIsolatedBatch(std::span<const Request> requests,
                               std::string* error_msg);
  static bool DecodeBatchImpl(std::span<const DecodeRequest> requests,
                              std::string* error_msg, std::uint32_t residents);
  static bool EvaluateBatchImpl(std::span<const AdvanceRequest> requests,
                                std::string* error_msg);
  struct PendingDecode;
  bool PrepareDecode(const DecodeRequest& request, PendingDecode* pending,
                     std::string* error_msg, bool defer_head = false,
                     std::optional<std::uint32_t> batch_drafts = {},
                     std::uint32_t occupancy = 1, std::uint32_t residents = 1);
  static void SelectScaledDraft(PendingDecode& pending);
  static void AppendDraft(PendingDecode& pending);
  /// Appends copies within the prepared budget, stopping at EOG when enabled.
  bool AppendLookup(PendingDecode& pending, bool stop_at_eos,
                    std::size_t minimum = 1, bool require_resume = false);
  bool FinishDecode(const DecodeRequest& request, const PendingDecode& pending,
                    std::string* error_msg);

  std::shared_ptr<Model> model_;
  std::unique_ptr<rocm::Session> session_;
  std::vector<std::int32_t> tokens_;
  std::vector<float> logits_;
  std::int32_t draft_token_{0};
  std::vector<float> verify_logits_;
  std::uint32_t hidden_base_{0};  ///< first position whose hidden row is kept
  MtpLengthController draft_length_;
  /// Index of tokens_ for prompt lookup (single-session MTP drafts).
  speculative::PromptLookup<> lookup_;
  /// Verification rows (anchor included) the next copy-only cycle of a
  /// session decoding alone may use; 0 before the first. Two whole copies
  /// in a row at this width double it up to max_copy_rows(); any rejected
  /// copy returns it to the ordinary verifier width.
  std::uint32_t copy_rows_{0};
  /// Whole copies accepted in a row at copy_rows_ (two widen it).
  std::uint32_t copy_streak_{0};
  SpeculativeStats stats_;
  std::shared_ptr<const qwen::vision::Prompt> image_prompt_;
  [[nodiscard]] std::span<const std::uint8_t> ImageIdentity(
      std::size_t token_count) const;
  bool valid_{true};
  ContinuationHooks* continuation_hooks_{nullptr};
  std::shared_ptr<DynamicAbliteration> dynamic_;
  /// The profile table dynamic_ resolves against.
  std::shared_ptr<const AbliterationTable> dynamic_table_;
  /// Before a trunk decode pass: observes the committed rows and turns the
  /// chosen profile ON.
  void ApplyDynamicAbliteration();
  /// This request's draft vocabulary; the device table follows it.
  MtpDraftVocab draft_vocab_;
  bool draft_vocab_stale_{false};
  std::vector<std::int32_t> draft_added_;
  /// After the token history is replaced: the table restarts from it.
  void ForgetDraftVocab() noexcept { draft_vocab_stale_ = true; }
  /// Brings the draft table up to the committed tokens.
  bool SyncDraftVocab(std::string* error_msg);
  [[nodiscard]] bool MtpEnabled() const noexcept;
};

/// Immutable session checkpoint. Committed cache rows may still reside in
/// the source session. The complete byte form restores in memory and on disk.
class SessionSnapshot final {
public:
  ~SessionSnapshot();
  SessionSnapshot(const SessionSnapshot&) = delete;
  SessionSnapshot& operator=(const SessionSnapshot&) = delete;
  SessionSnapshot(SessionSnapshot&&) = delete;
  SessionSnapshot& operator=(SessionSnapshot&&) = delete;

  /// Complete payload size, including borrowed rows, for cache admission.
  [[nodiscard]] std::uint64_t SizeBytes() const noexcept { return size_; }
  /// Frozen device storage, counting each shared backing block once.
  /// Borrowed live rows and allocator reservations are excluded.
  [[nodiscard]] std::uint64_t DeviceBytes() const;
  /// Materializes borrowed rows; may throw on a failed device transfer.
  [[nodiscard]] std::span<const std::uint8_t> bytes() const;
  [[nodiscard]] bool CopyTo(std::span<std::uint8_t> destination) const noexcept;

private:
  explicit SessionSnapshot(std::uint64_t size);

  std::shared_ptr<std::uint8_t[]> data_;
  std::uint64_t size_{0};
  std::shared_ptr<rocm::SnapshotState> deferred_;

  friend class Session;
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_ENGINE_HPP_
