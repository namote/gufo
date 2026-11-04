#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "src/core/hip/host_allocations.hpp"
#include "src/core/hip/kv_block_pool.hpp"
#include "src/core/hip/kvarn_layout.hpp"
#include "src/core/session_mode.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen/vision/device_input.hpp"
#include "src/models/qwen38_flash_next/continuation_hooks.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/blaslt.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/exl3_moe.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

class Executor;
class SnapshotAllocator;
struct ArgmaxCandidate;
struct SnapshotHeader;
struct SnapshotRegistry;
struct SnapshotLineage;
struct PrefillCheckpoint;

struct SnapshotBlock;
struct SnapshotChunk;

/// Mutable checkpoint state is copied immediately; committed rows borrow
/// append-only device storage until a rewind needs their shared device copy.
/// Byte readers assemble the payload under the same lock used by writers.
class SnapshotState {
public:
  void Materialize() const;
  [[nodiscard]] std::uint64_t DeviceBytes() const;

private:
  friend class Executor;
  friend class Session;
  friend struct SnapshotRegistry;
  struct Saved {
    std::uint64_t offset;
    std::uint64_t bytes;
    std::shared_ptr<const SnapshotChunk> chunk;
  };
  struct Region {
    void* device;
    std::uint8_t* host;
    std::uint64_t bytes;
    std::uint64_t live_bytes;
    std::uint64_t pending_bytes;
    std::uint64_t row_bytes;
    bool draft;
    bool pooled;
    std::vector<Saved> saved;
  };
  void Preserve(std::uint32_t position, std::uint32_t mtp_position,
                std::uint32_t blocks, std::uint32_t mtp_blocks,
                bool detach = true) const;
  /// Materialize with the registry mutex already held.
  void MaterializeLocked() const;
  std::shared_ptr<SnapshotRegistry> registry_;
  std::shared_ptr<const SnapshotLineage> lineage_;
  std::shared_ptr<void> storage_;
  mutable std::vector<Region> regions_;
  std::uint8_t* payload_ = nullptr;
  mutable std::shared_ptr<void> mutable_storage_;
  std::uint8_t* mutable_device_ = nullptr;
  std::uint64_t mutable_offset_ = 0;
  std::uint64_t mutable_bytes_ = 0;
  // The other small non-row regions (indexer ring, draft state, kept rows),
  // also frozen on the device until a byte reader needs them. Copying them
  // into untouched pageable payload pages cost several ms per capture.
  struct Extra {
    std::uint64_t offset;
    std::uint64_t bytes;
    std::uint8_t* device;
  };
  mutable std::vector<Extra> extras_;
  mutable std::shared_ptr<void> extra_storage_;
  std::uint64_t extra_bytes_ = 0;
};

/// Per-sequence state on the device: recurrent SSM state, KV and indexer
/// caches, PLE conv history, plus the host-side n-gram window. A
/// The KVarN plan of one forward over rows [start, start + rows): passes of
/// at most 128 rows quantize a group that ends inside the pass in each layer
/// (hip::PlanKvarnPass), wider prompt chunks write a staging buffer and
/// quantize their completed groups after attention (hip::PlanKvarnPrefill).
struct KvarnPass {
  bool prefill{false};
  std::uint32_t start{0};
  std::uint32_t rows{0};
  /// Prefill: staging row 0's position, and the end of the record rows the
  /// chunk reads.
  std::uint32_t staging_first{0};
  std::uint32_t quant_end{0};
  std::uint32_t flushed_end{hip::kKvarnSinkTokens};
};

/// speculative forward additionally keeps per-token snapshots of every
/// recurrent buffer so the batch can be cut back to its accepted prefix.
class Session {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  [[nodiscard]] bool mtp_enabled() const noexcept { return mtp_enabled_; }
  /// Trunk attention K/V are KVarN k4v4 records (Executor::Options::kvarn).
  [[nodiscard]] bool kvarn() const noexcept { return kvarn_; }
  [[nodiscard]] int kvarn_bits() const noexcept { return kvarn_bits_; }
  [[nodiscard]] std::uint32_t position() const noexcept { return position_; }
  [[nodiscard]] std::uint32_t max_context() const noexcept {
    return max_context_;
  }
  /// Drops every token; the next Forward starts at position 0. Cache slot
  /// reassignment may retain bounded MTP scratch and its recorded graphs.
  void Reset(bool reuse_mtp = false);
  // Stable device table; only newly mapped entries are uploaded before writes.
  void EnsureKvRows(std::uint32_t rows);
  void TrimKvRows(std::uint32_t rows);
  void PrepareKvRestoreRows(std::uint32_t rows);
  void EnsureKvRestoreBlock(std::uint32_t logical);
  void MapKvBlock(std::uint32_t logical, std::uint32_t physical);
  void UploadKvTable();
  void UnshareKvRows(std::uint32_t first, std::uint32_t end);
  void ReleaseMappedKvBlocks(std::size_t first = 0) noexcept;
  static constexpr std::uint32_t kUnmappedKvBlock = ~std::uint32_t{0};
  std::uint32_t ReserveKvRows(std::uint32_t rows, std::uint32_t margin);
  std::uint32_t ReleaseKvReservation() noexcept;
  [[nodiscard]] std::size_t MappedKvBlocks() const { return kv_blocks_.size(); }

  void SetCancellationCheck(std::function<bool()> check);
  [[nodiscard]] bool CheckCancellation(std::string* error) const;
  [[nodiscard]] bool Cancelled() const noexcept { return cancelled_; }
  [[nodiscard]] std::uint64_t MutationEpoch() const noexcept {
    return mutation_epoch_;
  }
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;
  [[nodiscard]] std::size_t RollbackBytes() const noexcept {
    return rollback_bytes_;
  }
  [[nodiscard]] std::size_t VisionBytes() const noexcept {
    return vision_input_.Bytes();
  }
  [[nodiscard]] std::size_t GraphCount() const noexcept {
    return graphs_.size();
  }
  /// PLE rows this session has gathered, including prefetches, since creation.
  [[nodiscard]] NgramTotals NgramRowTotals() const noexcept {
    return ngram_counters_.Load();
  }
  [[nodiscard]] bool OwnsSnapshot(
      const SnapshotState& snapshot) const noexcept {
    return snapshots_ == snapshot.registry_;
  }
  void ConfigureVision(std::shared_ptr<const qwen::vision::Prompt> prompt,
                       std::shared_ptr<qwen::vision::Encoder> encoder,
                       hipStream_t stream);
  void PrepareVision(hipStream_t stream);
  void RestoreVisionLayout(const qwen::vision::RopeLayout& layout,
                           hipStream_t stream);
  [[nodiscard]] const qwen::vision::RopeLayout& VisionLayout() const noexcept {
    return vision_input_.layout();
  }

private:
  friend class Executor;
  friend struct ExecutorTestAccess;
  friend struct PrefillCheckpoint;
  friend class gufo::models::qwen38_flash_next::ContinuationAdapter;
  Session() = default;

  struct LinearState {
    float* conv_state{nullptr};    ///< [kernel-1][channels]
    float* state{nullptr};         ///< [v_heads][d][d]
    RollbackRows conv_snapshots;   ///< [max_spec-1][kernel-1][channels]
    RollbackRows state_snapshots;  ///< [max_spec-1][v_heads][d][d]
  };
  struct AttentionState {
    const qwen::vision::DeviceRope* rope{nullptr};
    __half* k_cache{nullptr};  ///< [max_context][kv_heads*d]
    __half* v_cache{nullptr};  ///< [max_context][kv_heads*d]
    float* index_k{nullptr};   ///< [index_capacity_][indexer_dim] raw ring
    __half* block_k{nullptr};  ///< [max_context/ratio][indexer_dim]
    /// Block table of k_cache, v_cache and block_k (kernels.hpp
    /// kKvPooledRatio); null for contiguous rows.
    const std::uint32_t* kv_blocks{nullptr};
    /// KVarN layers (Options::kvarn): this layer's FP16 sink and tail ring
    /// and record plane, and the session's pass plan. Null for FP16 K/V,
    /// where k_cache/v_cache hold the rows.
    __half* kvarn_k{nullptr};
    __half* kvarn_v{nullptr};
    std::uint8_t* kvarn_records{nullptr};
    const KvarnPass* kvarn_pass{nullptr};
  };
  /// Per-launch values the kernels read from device memory, so a captured
  /// graph replays at any position.
  struct Control {
    std::uint32_t position;      ///< first position of the trunk batch
    std::uint32_t blocks;        ///< indexer blocks pooled so far
    std::uint32_t mtp_position;  ///< first position of the draft batch
    std::int32_t hidden_row;     ///< kept trunk row the draft reads, or -1
    std::uint32_t mtp_blocks;    ///< complete predictor indexer blocks
  };
  struct MtpState {
    __half* k_cache{nullptr};
    __half* v_cache{nullptr};
    float* index_k{nullptr};
    __half* block_k{nullptr};
    const std::uint32_t* kv_blocks{nullptr};  ///< the session's block table
    std::uint32_t blocks{0};
    float* h{nullptr};  ///< [hc_dim] wide residual handed to the next draft
    float* target_hidden{nullptr};  ///< last max_speculative trunk rows
    std::uint32_t position{0};
    bool residual_valid{false};
    /// The request's draft table (draft_vocab.hpp): kMtpDraftExtraRows
    /// output() rows and their token IDs, -1 for an empty slot.
    std::uint8_t* draft_rows{nullptr};
    std::int32_t* draft_ids{nullptr};
  };

  mutable bool cancelled_{false};
  std::uint64_t mutation_epoch_{0};
  const Executor* owner_{nullptr};
  qwen::vision::DeviceInput vision_input_;
  std::uint32_t max_context_{0};
  bool mtp_enabled_{false};
  std::uint32_t index_capacity_{0};  ///< power-of-two raw indexer ring rows
  std::uint32_t position_{0};
  ContinuationHooks* continuation_hooks_{nullptr};
  std::vector<LinearState> linear_;
  std::vector<AttentionState> attention_;
  float* ple_history_{nullptr};  ///< [PleConvHistory()][hc_dim]
  /// KVarN trunk K/V (plan/exl3/F1-KVARN-DESIGN.md): which rows are
  /// quantized, the current forward's plan and the FP16 sink/tail rows
  /// ([K|V][attention layer][kKvarnFp16Tokens][kv row]).
  bool kvarn_{false};
  int kvarn_bits_{4};
  hip::KvarnTail kvarn_tail_{};
  KvarnPass kvarn_pass_{};
  __half* kvarn_fp16_{nullptr};
  std::uint8_t* kvarn_records_{nullptr};  ///< record plane, attention layer 0
  std::size_t kvarn_records_pitch_{0};    ///< bytes per layer of that plane
  std::uint32_t* kv_table_{nullptr};
  std::shared_ptr<hip::KvBlockPool> kv_pool_;
  hip::KvBlockReservation kv_reservation_;
  hip::HostAllocations kv_table_host_owner_;
  std::uint32_t* kv_table_host_{nullptr};
  std::vector<std::uint32_t> kv_blocks_;
  std::size_t kv_unmapped_blocks_{0};

  RollbackRows ple_snapshots_;  ///< [max_spec-1][PleConvHistory()][hc_dim]
  NgramHistory ngram_;
  std::vector<NgramHistory> ngram_snapshots_;
  // Prefetches count against the const session that will consume them.
  mutable NgramCounters ngram_counters_;
  std::uint32_t spec_base_{0};    ///< position before the speculative batch
  std::uint32_t spec_tokens_{0};  ///< tokens of the pending speculative batch
  MtpState mtp_;
  Control* control_{nullptr};
  std::uint32_t blocks_{0};  ///< host shadow of Control::blocks
  /// Captured decode graphs by batch shape, and the shapes that ran eagerly
  /// once (the GEMM tier's arena must be grown before capture).
  std::unordered_map<std::uint64_t, hipGraphExec_t> graphs_;
  std::unordered_set<std::uint64_t> warmed_;
  std::vector<void*> allocations_;
  std::size_t allocated_bytes_{0};
  std::size_t rollback_bytes_{0};
  std::function<bool()> is_cancelled_;
  std::vector<float*> rollback_allocations_;
  std::uint32_t rollback_depth_{0};
  void TrimRollback(std::uint32_t depth) noexcept;
  mutable std::shared_ptr<SnapshotRegistry> snapshots_;
  mutable std::shared_ptr<const SnapshotLineage> snapshot_lineage_;
  mutable std::uint32_t snapshot_position_{0};
  mutable std::uint32_t snapshot_mtp_position_{0};
  void PreserveSnapshots(std::uint32_t position,
                         std::uint32_t mtp_position) const;
};

/// A frozen mutable frontier with append-only caches owned by its live session.
/// It is consumed before another forward can reuse the checkpoint arena.
struct PrefillCheckpoint {
  ~PrefillCheckpoint();
  std::unique_ptr<Session> state;
  Session* source = nullptr;
  std::uint32_t tokens = 0;
  std::uint32_t hidden_rows = 0;
  std::vector<float> logits;
};

/// Runs the trunk graph on the GPU for one session at a time. Buffers are
/// sized once for `max_batch` tokens; longer prompts are fed in chunks.
class Executor {
public:
  struct Options {
    std::uint32_t max_batch{1};
    /// Rows of logits (and hidden states) a Forward call may return.
    std::uint32_t max_logit_rows{1};
    /// Longest speculative batch; bounds the recurrent snapshot storage.
    std::uint32_t max_speculative{1};
    /// Largest session context. Zero selects the native context; values
    /// above it require YaRN (Config::MaxContextLength()).
    std::uint32_t max_context{0};
    /// Trunk attention K/V as KVarN k4v4 records with token-major values
    /// (plan/exl3/F1-KVARN-DESIGN.md); the MTP layer stays FP16.
    bool kvarn{false};
    int kvarn_bits{4};
    /// Diagnostic chunk-policy hook for EXL3. Zero preserves the caller's
    /// existing policy; a nonzero limit is bounded by max_batch. The caller
    /// still supplies checkpoint/cancellation boundaries to PrefillChunkRows.
    std::uint32_t exl3_prefill_chunk_rows{0};
  };

  ~Executor();
  [[nodiscard]] hipStream_t stream() const noexcept { return stream_; }
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  [[nodiscard]] static std::unique_ptr<Executor> Create(
      const DeviceModel& model, NgramTable* ngram, Options options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      core::SessionMode mode, std::uint32_t max_context,
      std::string* error_msg = nullptr,
      std::shared_ptr<hip::KvBlockPool> pool = {}) const;
  [[nodiscard]] hip::KvBlockGeometry KvGeometry(core::SessionMode mode,
                                                std::uint32_t context) const;
  /// KVarN sessions: plans a forward over rows [start, start + n) and
  /// quantizes the groups it needs before running (KvarnPass), then records
  /// the tail after it, quantizing a group that ends exactly at the new
  /// position so a snapshot's records always end at the group floor.
  [[nodiscard]] bool PlanKvarn(Session& session, std::uint32_t start,
                               std::uint32_t n, std::string* error_msg) const;
  [[nodiscard]] bool FinishKvarn(Session& session,
                                 std::string* error_msg) const;
  /// Quantizes groups [first, first + groups) of every attention layer from
  /// the session's FP16 tail ring.
  void FlushKvarnGroups(const Session& session, std::uint32_t first,
                        std::uint32_t groups) const;
  [[nodiscard]] bool EnsureRollback(Session& session, std::uint32_t depth,
                                    std::string* error_msg) const;
  [[nodiscard]] std::size_t SessionBytes(
      core::SessionMode mode, std::uint32_t max_context,
      std::uint32_t rollback_depth) const noexcept;
  [[nodiscard]] std::size_t DeferredScratchBytes() const;

  enum class ForwardMode { kDecode, kVerify, kPrefill };
  [[nodiscard]] std::unique_ptr<PrefillCheckpoint> MakePrefillCheckpoint(
      Session& session, std::span<const std::int32_t> tokens,
      std::uint32_t prefix, std::string* error_msg) const;

  /// Appends `tokens` (at most max_batch) to the session and returns the
  /// logits of the last `n_logits` tokens in `logits` (n_logits * vocab
  /// floats, host memory). A null `logits` keeps the rows on the GPU for
  /// verification. The final wide residual of those tokens stays on
  /// the device for MtpForward. kVerify permits Rollback (at most
  /// max_speculative rows). kPrefill uses consistent prompt arithmetic at
  /// every chunk width. A kPrefill call may name the `next` chunk of the same
  /// prompt: once its own n-gram rows have arrived it starts reading the
  /// next chunk's, so that read overlaps this chunk's layers. The rows are
  /// used only if the next batch has exactly those tokens.
  [[nodiscard]] bool Forward(Session& session,
                             std::span<const std::int32_t> tokens,
                             std::uint32_t n_logits, float* logits,
                             ForwardMode mode, std::string* error_msg,
                             PrefillCheckpoint* checkpoint = nullptr,
                             std::span<const std::int32_t> next = {}) const;

  /// Quality-probe replay of the last Forward's output head, with the same
  /// batch width and requested tail. Call immediately after an AR Forward,
  /// before any other executor operation reuses scratch. Does not advance
  /// session state. `logits` must have n_logits * vocab elements.
  [[nodiscard]] bool ReplayOutputHead(std::uint32_t batch_rows,
                                      std::span<float> logits,
                                      std::string* error_msg) const;

  struct BatchItem {
    Session* session;
    std::span<const std::int32_t> tokens;
    bool speculative;
  };
  /// Packs independent decode chains for shared projections. Every chain
  /// keeps its own recurrent/KV state and the decode arithmetic (<=8 rows).
  /// Logit rows remain in batch order until SelectBatchLogits is called.
  [[nodiscard]] bool ForwardBatch(std::span<const BatchItem> items,
                                  std::string* error_msg) const;
  [[nodiscard]] bool SelectBatchLogits(std::uint32_t offset, std::uint32_t rows,
                                       float* logits,
                                       std::string* error_msg) const;
  [[nodiscard]] bool ReadVerificationRows(std::uint32_t row,
                                          std::span<float> logits,
                                          std::string* error_msg) const;

  /// Keeps the first `keep` (1..n) tokens of the last speculative batch and
  /// discards the rest. If `logits` is supplied, copies the kept frontier
  /// into it using the same synchronization as rollback.
  [[nodiscard]] bool Rollback(Session& session, std::uint32_t keep,
                              std::string* error_msg,
                              float* logits = nullptr) const;

  /// Runs the draft block over `tokens` (at most max_batch) at the session's
  /// MTP position. The hidden input of token i is the trunk residual of row
  /// `hidden_row + i` of the last Forward batch, or, with hidden_row < 0
  /// (single token), the draft block's own residual from the previous call.
  /// Only requested outputs are computed. Production requests a greedy
  /// token or compact candidate logits. Catch-up skips the output head
  /// when only the draft state is needed.
  struct MtpOutput {
    std::int32_t* token{nullptr};
    MtpCandidateLogits* candidates{nullptr};
    MtpTrace* trace{nullptr};  ///< final-row diagnostic; disables graph capture
    /// Prefill and headless steps need only persistent KV. A full forward with
    /// known trunk hidden rows is required before heads/recursive proposals.
    bool kv_only{false};
    /// Draft through the small head (the output() prefix plus the
    /// session's draft table) when the model has one.
    bool draft_subset{false};
  };
  struct MtpHeadItem {
    Session* session;
    MtpOutput output;
  };
  struct MtpBatchItem {
    Session* session;
    std::span<const std::int32_t> tokens;
    std::int32_t hidden_row;
    bool kv_only{false};
  };
  /// Copies output() rows `ids` into the session's draft table from slot
  /// `first` on. Enqueued on stream(); never during graph capture.
  [[nodiscard]] bool AppendDraftRows(Session& session, std::uint32_t first,
                                     std::span<const std::int32_t> ids,
                                     std::string* error_msg) const;
  /// Empties the session's draft table from slot `first` on.
  [[nodiscard]] bool ClearDraftRows(Session& session, std::uint32_t first,
                                    std::string* error_msg) const;
  /// Runs independent short predictor chains with shared projections and
  /// private attention caches, positions and carried hidden states.
  [[nodiscard]] bool MtpForwardBatch(std::span<const MtpBatchItem> items,
                                     std::string* error_msg) const;
  /// Projects each session's carried draft hidden state with shared weights.
  [[nodiscard]] bool MtpHeads(std::span<const MtpHeadItem> items,
                              std::string* error_msg) const;
  [[nodiscard]] bool MtpForward(Session& session,
                                std::span<const std::int32_t> tokens,
                                std::int32_t hidden_row, MtpOutput output,
                                std::string* error_msg,
                                const float* hidden_source = nullptr) const;
  /// Leading kept trunk rows, for independent predictor qualification.
  [[nodiscard]] bool CopyTrunkHidden(const Session& session,
                                     std::span<float> hidden,
                                     std::string* error_msg) const;

  /// Greedy verification keeps full logit rows on the GPU, including penalties.
  [[nodiscard]] bool GreedyMtpPredictions(
      std::span<ArgmaxCandidate> predictions,
      const sampling::SamplerState& sampler,
      std::span<const std::int32_t> drafts, std::string* error_msg) const;
  [[nodiscard]] static bool CanCompactMtpVerification(
      const sampling::SamplerState& sampler);
  /// Candidate spans remain valid until this executor next verifies a
  /// request. One transfer serves all rows; no full CPU rows are reconstructed.
  [[nodiscard]] bool SampledMtpCandidates(
      std::span<std::span<const sampling::LogitCandidate>> rows,
      const sampling::SamplerState& sampler,
      std::span<const std::int32_t> drafts, std::string* error_msg) const;

  /// A session's complete context as one host byte payload: recurrent and
  /// PLE state, KV and indexer caches up to the position, and the draft
  /// block's caches plus the `hidden_rows` most recent kept trunk rows.
  /// The payload restores into any session of this executor whose context
  /// holds the position; the pending speculative batch must be empty.
  struct SnapshotInfo {
    std::uint32_t position{0};
    std::uint32_t hidden_rows{0};
  };
  [[nodiscard]] std::uint64_t SnapshotBytes(const Session& session,
                                            std::uint32_t hidden_rows) const;
  [[nodiscard]] std::uint64_t PrefillCheckpointBytes(
      const Session& session, std::uint32_t position) const;
  [[nodiscard]] bool SaveSnapshot(
      const Session& session, std::uint32_t hidden_rows,
      std::span<std::uint8_t> payload, std::string* error_msg,
      std::shared_ptr<SnapshotState>* deferred = nullptr,
      std::shared_ptr<void> storage = {}) const;
  /// Reuse at most the rollback rows needed by the restored operation;
  /// restoration never grows scratch. Callers derive this bound from the
  /// restored policy (or the concrete verifier width in a diagnostic).
  [[nodiscard]] bool RestoreSnapshot(
      Session& session, std::span<const std::uint8_t> payload,
      SnapshotInfo* info, std::string* error_msg, std::uint32_t next_drafts = 0,
      const SnapshotState* deferred = nullptr) const;

  /// Rewinds the draft block's own context.
  void MtpRewind(Session& session, std::uint32_t position) const {
    if (session.continuation_hooks_ && position < session.mtp_.position)
      session.continuation_hooks_->BeforeWrite(
          true, position, session.mtp_.position,
          std::min(session.mtp_.blocks, position / config().compress_ratio),
          session.mtp_.blocks);
    if (session.mtp_.position != position)
      session.mtp_.residual_valid = false;
    session.mtp_.position = position;
    session.mtp_.blocks =
        std::min(session.mtp_.blocks, position / config().compress_ratio);
  }
  [[nodiscard]] std::uint32_t MtpPosition(
      const Session& session) const noexcept {
    return session.mtp_.position;
  }

  [[nodiscard]] const Config& config() const noexcept {
    return model_->config();
  }
  [[nodiscard]] std::uint32_t max_batch() const noexcept {
    return options_.max_batch;
  }
  [[nodiscard]] std::uint32_t max_speculative() const noexcept {
    return options_.max_speculative;
  }
  [[nodiscard]] bool has_mtp() const noexcept { return model_->has_mtp(); }
  /// Whether trunk K/V are KVarN k4v4 (Options::kvarn).
  [[nodiscard]] bool kvarn() const noexcept { return options_.kvarn; }

  [[nodiscard]] std::uint32_t PrefillChunkRows(
      std::uint32_t requested, std::uint32_t until_checkpoint) const noexcept;
  /// Host-only estimate of the disjoint persistent bucket slabs. Zero means
  /// invalid geometry (error_msg explains); excludes existing stage gate/up,
  /// s_.down_e and the decode path's 64 * used exponents. Uses a 64-row tile.
  [[nodiscard]] static std::size_t EstimateExl3PrefillScratchBytes(
      const Config& config, std::uint32_t rows,
      std::string* error_msg = nullptr);
  [[nodiscard]] std::size_t exl3_prefill_scratch_bytes() const noexcept {
    return exl3_prefill_scratch_bytes_;
  }

  /// Opt-in operator/profiling diagnostics, following MtpTrace's eager path.
  /// No production caller enables this implicitly. Counts cover the whole
  /// executor/cohort; the engine supplies actual committed/proposed tokens
  /// after acceptance, stops and cancellation, including copied proposals.
  struct MtpCycleDiagnostics {
    std::uint64_t committed_tokens{0}, drafts{0}, verify_rows{0};
    /// One routed MLP tile (all three projections), summed over layers.
    std::uint64_t expert_tile_reads{0};
    /// False if a reference/bucket-prefill route ran without eight-row tiles.
    bool expert_tile_reads_complete{true};
    /// 128-column draft-head segments actually projected, including prefix.
    std::uint64_t head_segments{0}, catch_up_rows{0};
    /// Rows that entered the cache-only branch, rather than a full body.
    std::uint64_t cache_only_rows{0};
    /// Routed prefill rows and actual 64-row expert buckets, across layers.
    std::uint64_t prefill_expert_rows{0}, prefill_expert_buckets{0};
    /// Decode/verify assignments sent through the allocated scaled-down path
    /// (which can still select widened fallback for an unsafe row).
    std::uint64_t scaled_down_rows{0};
    /// Dense compressed prefill input rows (summed over projections), and
    /// actual grouped rotation/GEMM calls. Plain prefill rows counted
    /// separately.
    std::uint64_t prefill_dense_rows{0}, prefill_dense_rotation_groups{0},
        prefill_dense_gemm_groups{0}, prefill_plain_rows{0};
    /// Prefill forks actually queued for the shared MLP.
    std::uint64_t prefill_shared_overlaps{0};
    /// Plain prefill rows using FP16 WMMA, including singleton/tail chunks.
    std::uint64_t prefill_plain_half_rows{0};
    /// Exclusive call wall times. draft_ms/catch_up_ms include inline heads;
    /// head_ms covers standalone MtpHeads. Diagnostic count downloads and
    /// synchronization are included; use an uninstrumented run for speed.
    double target_ms{0}, draft_ms{0}, head_ms{0}, catch_up_ms{0}, wall_ms{0};
  };
  [[nodiscard]] bool BeginMtpCycleDiagnostics(std::string* error_msg) const;
  [[nodiscard]] bool EndMtpCycleDiagnostics(std::uint64_t committed_tokens,
                                            std::uint64_t drafts,
                                            MtpCycleDiagnostics* output,
                                            std::string* error_msg) const;

private:
  Executor() = default;
  friend struct ExecutorTestAccess;

  enum class DiagnosticStage { kTarget, kDraft, kHead, kCatchUp };
  struct DiagnosticPhase {
    const Executor& executor;
    DiagnosticStage stage;
    std::chrono::steady_clock::time_point started{};
    DiagnosticPhase* parent{nullptr};
    double children_ms{0};
    bool active{false};
    DiagnosticPhase(const Executor& e, DiagnosticStage s);
    ~DiagnosticPhase();
  };
  mutable bool diagnostics_active_{false};
  mutable std::uint32_t diagnostic_depth_{0};
  mutable DiagnosticPhase* diagnostic_phase_{nullptr};
  mutable std::chrono::steady_clock::time_point diagnostic_started_{};
  mutable MtpCycleDiagnostics cycle_diagnostics_{};
  bool DiagnosticCount(const std::uint32_t* device, std::uint64_t* total,
                       std::string* error_msg) const;

  /// Visits every device region of a snapshot in payload order with
  /// (device pointer or null when sizing, payload offset, bytes, name).
  /// Returns the payload size, or 0 once a visit failed.
  template<typename Visit>
  static std::uint64_t WalkSnapshot(const SnapshotHeader& h,
                                    const Session* session, Visit&& visit);

  /// An activation batch quantized once for the decode GEMVs; `data` is
  /// null when the batch is wide enough for the tiled path.
  struct Q8Input {
    const float* x;
    const void* data;
    std::uint32_t n;
    std::uint32_t k;
  };
  bool Quantize(const float* x, std::uint32_t n_tokens, std::uint32_t k,
                Q8Input* q, std::string* error_msg) const;
  /// Q8_1 rows of a small-batch input that its producer kernel already
  /// wrote (byte-identical to Quantize's), or null.
  const void* ReadyQ8(const float* x, std::uint32_t n_tokens,
                      std::uint32_t k) const;
  /// Forgets the producer-written Q8_1 rows (their sources are rewritten).
  void DropReadyQ8() const;
  bool Dense(const DeviceTensor& w, const Q8Input& q, float* out,
             std::string* error_msg) const;
  bool Dense(const DeviceTensor& w, const float* x, float* out,
             std::uint32_t n_tokens, std::string* error_msg) const;
  /// Cache-only MTP uses the packed GGUF slice or the separate EXL3 K/V
  /// group. Separate outputs retain the full attention path's arithmetic.
  bool MtpKvProjection(const DeviceLayer& layer, const float* x,
                       std::uint32_t rows, std::string* error_msg) const;
  bool StoreMtpKv(const DeviceLayer& layer, const Session& session,
                  std::uint32_t rows, const YarnRope& yarn,
                  std::string* error_msg) const;
  /// out = (up . x) * silu(gate . x); s_.shexp_gate is scratch.
  /// With `down` (the projection that consumes the result), a wide batch
  /// leaves only that projection's staged input form (F16 rows in
  /// s_.x_half or the tiled Q8 layout in s_.x_q8t, registered in the input
  /// cache); `out` then holds the gate projection, not the result.
  bool GatedDense(const DeviceTensor& up, const DeviceTensor& gate,
                  const float* x, float* out, std::uint32_t n_tokens,
                  const DeviceTensor* down, std::string* error_msg) const;
  /// Reuse or populate the F16 activation staging buffer.
  void PrepareHalfInput(const float* x, std::uint32_t rows,
                        std::uint32_t cols) const;
  /// Whether a wide dense Q8_0 projection takes the F16 WMMA GEMM.
  bool DenseF16Route(const DeviceTensor& w, std::uint32_t n_tokens) const;
  void RoutedHints(const DeviceTensor& w, std::uint32_t n_tokens) const;
  /// Reads the routing of the current batch back and derives the tile
  /// hints for its expert GEMMs (tiled batches only).
  bool RouteHints(std::uint32_t n_tokens, std::string* error_msg) const;
  /// Routed gate/up projections followed by SwiGLU.
  bool GatedExperts(const DeviceTensor& a, const DeviceTensor& b,
                    const float* x, const std::int32_t* ids, float* out,
                    std::uint32_t n_tokens, std::uint32_t n_used,
                    std::string* error_msg) const;
  bool Experts(const DeviceTensor& w, const float* x, const std::int32_t* ids,
               float* out, std::uint32_t n_rows, std::uint32_t n_used,
               std::uint32_t n_tokens, std::string* error_msg) const;
  /// `normed` says the previous Combine already produced m's grouped norm
  /// of res (F32 in s_.xn, or F16 plus tiled Q8 on the wide route).
  /// `half_only` leaves only the F16 copy of `mixed`; `no_f32` keeps its
  /// F16 and tiled Q8 copies but skips the F32 rows nothing reads.
  bool HcMix(const DeviceMixer& m, const float* res, bool normed, float* mixed,
             float* inject, std::uint32_t n_tokens, std::string* error_msg,
             bool half_only = false, bool no_f32 = false) const;
  /// Share mixer projections across decode requests without selecting the
  /// prefill arithmetic. Injection rows are compact across the whole batch.
  bool HcMixBatch(const DeviceMixer& m, const float* res, bool normed,
                  float* mixed, float* inject, std::uint32_t rows,
                  std::string* error) const;
  void CombineBatch(float* res, const float* gamma, std::uint32_t rows) const;
  /// Residual update by the block output plus the grouped norm for the next
  /// mixer (`gamma`); wide batches write it as F16 and tiled Q8. With that
  /// mixer's F32 `inject_w`, the fused MoE combine also leaves its inject
  /// partials in s_.inject (inject_fused_).
  void Combine(float* res, const float* gamma, std::uint32_t n_tokens,
               const float* inject_w = nullptr) const;
  /// Hashes the batch's n-gram rows and starts reading them from disk, so
  /// the read overlaps the layers before the PLE one.
  bool PleFetch(Session& s, std::span<const std::int32_t> tokens,
                bool speculative, std::string* error_msg) const;
  bool WaitPle(std::string* error_msg) const;
  /// Starts reading `next`'s rows into the prefetch buffer. Best effort: a
  /// prefetch that cannot start is skipped.
  void PrefetchPle(const Session& s, std::span<const std::int32_t> next) const;
  /// Waits for a prefetch in flight before another read; its rows stay
  /// claimable.
  void FinishPrefetch() const;
  bool Ple(const DeviceLayer& l, Session& s, std::uint32_t n_tokens, float* res,
           bool speculative, std::string* error_msg,
           bool embeddings_ready = false,
           PrefillCheckpoint* checkpoint = nullptr) const;
  bool LinearAttention(const DeviceLayer& l, Session::LinearState& s,
                       const float* x, float* out, std::uint32_t n_tokens,
                       bool speculative, std::string* error_msg,
                       bool projections_ready = false,
                       bool project_output = true,
                       GdnCheckpoint checkpoint = {}) const;
  /// `pos`/`first_block` are device values; `start_pos` and `pool_grid`
  /// are their host-side counterparts for the eager-only decisions.
  bool Attention(const DeviceLayer& l, Session::AttentionState& s,
                 const float* x, float* out, std::uint32_t n_tokens,
                 const std::uint32_t* pos, const std::uint32_t* first_block,
                 std::uint32_t start_pos, std::uint32_t pool_grid,
                 std::uint32_t max_context, bool sparse, std::string* error_msg,
                 bool last_only = false, bool projections_ready = false,
                 bool project_output = true,
                 std::uint32_t checkpoint_tokens = 0) const;
  /// The attention, KVarN chunk bookkeeping and output projection of a
  /// KVarN layer after its rows are written (Attention).
  bool KvarnAttention(const DeviceLayer& l, const Session::AttentionState& s,
                      std::uint32_t n_tokens, const std::uint32_t* pos,
                      std::uint32_t start_pos, const std::uint32_t* mask,
                      bool last_only, bool project_output, float* out,
                      std::string* error_msg) const;
  /// Orders the main stream after the work queued on side_stream_.
  bool JoinSide(std::string* error_msg) const;
  bool Moe(const DeviceLayer& l, const float* x, float* out,
           std::uint32_t n_tokens, std::string* error_msg,
           bool last_only = false) const;
  /// Whether MoeExperts takes the routed F16 WMMA route, whose epilogue the
  /// following combine absorbs (see Moe).
  bool WmmaExperts(const DeviceLayer& l, std::uint32_t n_tokens) const;
  /// Runs routed experts after the router and shared expert are ready.
  bool MoeExperts(const DeviceLayer& l, const float* x, float* out,
                  std::uint32_t n_tokens, std::string* error_msg) const;
  bool MoeBatch(const DeviceLayer& l, const float* x, float* out,
                std::uint32_t rows, std::string* error) const;
  /// Whether a draft of `session` requesting the small head gets it.
  bool UseDraftSubset(const Session& session, bool requested) const noexcept {
    return requested && !model_->draft_output().empty() &&
           session.mtp_.draft_rows != nullptr;
  }
  /// The session's draft table as a weight of kMtpDraftExtraRows rows.
  DeviceTensor DraftTable(const Session& session) const noexcept;
  /// Small-head draft logits: prefix rows, then the table rows with empty
  /// slots at -infinity.
  bool DraftLogits(const Session& session, const float* x, float* logits,
                   std::string* error_msg) const;
  /// Selects the greedy token or compact candidates from MTP draft logits.
  /// With `draft`, the logits come from that session's small head and the
  /// returned IDs are remapped to token IDs.
  bool MtpHead(const DeviceMixer& head, const float* res, bool token,
               bool candidates, const Session* draft,
               std::string* error_msg) const;
  /// Enqueues one trunk batch (control and token upload through logits).
  bool ForwardBody(Session& session, std::uint32_t n, std::uint32_t n_logits,
                   bool download_logits, bool speculative, bool sparse,
                   std::uint32_t start_pos, std::uint32_t pool_grid,
                   std::uint32_t first_layer, std::uint32_t end_layer,
                   std::string* error_msg,
                   PrefillCheckpoint* checkpoint = nullptr,
                   bool embeddings_ready = false) const;
  bool MtpBody(Session& session, std::uint32_t n, std::uint32_t pos, bool token,
               bool candidates, std::string* error_msg, std::uint32_t pool_grid,
               const float* hidden_source, MtpTrace* trace, bool kv_only,
               bool draft_subset) const;
  /// Runs `body` eagerly, or as the session's captured graph for `key`
  /// when `graph` is set. A prefix may leave its work queued so the host
  /// can wait for disk reads while the GPU computes it.
  bool Run(Session& session, std::uint64_t key, bool graph,
           const std::function<bool()>& body, std::string* error_msg,
           bool synchronize = true) const;

  const DeviceModel* model_{nullptr};
  NgramTable* ngram_{nullptr};
  Options options_;
  [[nodiscard]] std::shared_ptr<SnapshotAllocator> GetSnapshotAllocator() const;
  mutable std::once_flag snapshot_allocator_once_;
  mutable std::shared_ptr<SnapshotAllocator> snapshot_allocator_;
  mutable std::shared_ptr<void> checkpoint_storage_;
  /// The stream kernels are queued on. Overlapped work swaps in side_stream_
  /// for its launches and joins back through side_done_ (JoinSide).
  mutable hipStream_t stream_{nullptr};
  hipStream_t side_stream_{nullptr};
  hipEvent_t side_start_{nullptr};
  hipEvent_t side_done_{nullptr};
  mutable bool side_pending_{false};  ///< side_done_ not yet waited on
  hipEvent_t counts_ready_{nullptr};
  hipblasHandle_t blas_{nullptr};
  std::unique_ptr<BlasLt> blaslt_;

  // Scratch, sized for max_batch tokens. Names follow reference.cpp.
  struct Scratch {
    std::int32_t* tokens;
    void* x_half;   ///< activations narrowed to the weight's 16-bit type
    void* x_q8[2];  ///< Q8_1 activations of a decode batch, alternating
    /// Q8_1 rows of a decode batch written by the kernel that produces
    /// them: xn by HcCombine, lo by SiluScale, mixed by the mix epilogue.
    void* xn_q8_1;
    void* lo_q8_1;
    void* mixed_q8_1;
    void* gdn_q8_1;  ///< by the GDN epilogue (ssm_out's input)
    void* ctx_q8_1;  ///< by the attention output gate (attn_out's input)
    void* x_q8t;     ///< tiled Q8 activations of a wide batch (W8A8 route)
    float* res;
    float* xn;
    __half* xn_half;  ///< xn as F16 on the F16 mixer input route
    void* xn_q8t;     ///< xn as tiled Q8 for the W8A8 mixer down projection
    float* lo;
    float* hc_gate;
    float* mixed;
    float* inject;
    float* block_out;
    float* hc_defer_w;  ///< [t][4] stream weights of a deferred combine
    /// [t][4] stream norm scales of a combine that left s_.xn unwritten.
    float* xn_scale{nullptr};
    // linear attention
    float* qkv;
    float* z;
    float* qkvz;  ///< [t][qkv | z] from the stacked projection
    float* alpha_beta;
    float* conv_scratch;
    float* qn;
    float* kn;
    float* gdn_raw;
    float* gdn_out;
    // attention
    float* qg;  ///< [t][q|gate (; k ; v)]
    float* q;
    float* attn_gate;
    float* k;
    float* v;
    float* iq;
    float* ik;
    std::uint32_t* mask;
    float* scores;
    float* ctx;
    float* attn_partials;        ///< split-key partials of a narrow batch
    std::uint32_t* attn_lists;   ///< [row][selection budget] compacted blocks
    std::uint32_t* attn_counts;  ///< selected blocks per narrow-batch row
    // ple
    float* ple_emb;
    float* ple_key;
    float* ple_value;
    float* ple_query;
    float* ple_gated;
    float* ple_norm;
    float* ple_conv;
    float* ple_history_scratch;
    // moe
    float* router;
    std::int32_t* ids;
    std::uint32_t* expert_counts;
    // Routed WMMA route: 16-row padded bucket bounds, scatter cursors, the
    // compact row -> (token, slot) maps and the tiled Q8 gathered rows.
    std::int32_t* routed_bounds;
    std::int32_t* routed_cursors;
    std::int32_t* rows_token;
    std::int32_t* rows_slot;
    std::int32_t* routed_tiles;  ///< (expert | tile << 16) per launched tile
    float* weights;
    float* gate_e;
    float* up_e;
    float* down_e;
    float* shexp_gate;
    float* shexp_up;
    float* shexp_out;
    /// Shared-expert SwiGLU rows narrowed for the F16 down projection, so
    /// s_.x_half keeps the routed experts' token rows.
    __half* shexp_half;
    // head and hidden rows kept for the draft block
    float* logits;
    // mtp
    float* mtp_h;
    float* mtp_embd;
    float* mtp_eproj;
    float* mtp_res;
    ArgmaxCandidate* mtp_argmax;
    std::int32_t* mtp_token;
    std::uint32_t* mtp_ids;
    std::uint32_t* mtp_scratch_ids;
    float* mtp_scores;
  };
  mutable Scratch s_{};
  std::size_t half_input_cols_{0};
  Exl3Scratch exl3_{};
  gufo::hip::Exl3RotateProblem* exl3_dense_rotations_{nullptr};
  gufo::hip::Exl3GemmProblem* exl3_dense_gemms_{nullptr};
  struct Exl3DenseWorkspace {
    std::uint16_t* input{nullptr};
    std::size_t elements{0};
    gufo::hip::Exl3RotateProblem* rotations{nullptr};
    gufo::hip::Exl3GemmProblem* gemms{nullptr};
  };
  Exl3DenseWorkspace exl3_shared_workspace_{};
  bool Exl3DensePrefill(std::initializer_list<const DeviceTensor*> weights,
                        std::initializer_list<float*> outputs, const float* x,
                        std::uint32_t rows, std::string* error,
                        const Exl3DenseWorkspace* workspace = nullptr,
                        hipStream_t launch_stream = nullptr) const;
  Exl3PrefillScratch exl3_prefill_{};
  std::size_t exl3_prefill_scratch_bytes_{0};
  [[nodiscard]] Exl3PrefillScratch Exl3PrefillView() const;
  bool Exl3Group(std::initializer_list<const DeviceTensor*> weights,
                 std::initializer_list<float*> outputs, const float* x,
                 std::uint32_t rows, std::string* error,
                 std::size_t shape_basis_n = 0) const;
  bool PlainDensePrefill(const DeviceTensor& weight, const float* x, float* out,
                         std::uint32_t rows, std::string* error) const;
  bool Exl3Experts(const DeviceLayer& layer, const float* x, float* out,
                   std::uint32_t rows, std::string* error) const;
  [[nodiscard]] Scratch RowScratch(const Scratch& base,
                                   std::uint32_t offset) const;
  void UseScratch(const Scratch& scratch) const;
  bool DenseBatch(const DeviceTensor& w, const float* x, float* out,
                  std::uint32_t rows, std::string* error_msg) const;
  bool GatedDenseBatch(const DeviceTensor& up, const DeviceTensor& gate,
                       const float* x, float* out, std::uint32_t rows,
                       std::string* error) const;
  bool QuantizeBatch(const float* x, std::uint32_t rows, std::uint32_t cols,
                     std::string* error) const;
  bool AllocateBatch(std::string* error_msg) const;
  /// Shared transient storage, grown only for penalty-aware verification.
  mutable sampling::TokenPenalty* verification_penalties_{nullptr};
  mutable std::size_t verification_penalty_capacity_{0};
  static constexpr std::size_t kCompactPenaltyLimit = 4096;
  std::uint32_t* verification_ids_{nullptr};
  std::uint32_t* verification_scratch_ids_{nullptr};
  sampling::LogitCandidate* verification_candidates_{nullptr};
  sampling::LogitCandidate* verification_candidates_host_{nullptr};
  mutable std::vector<sampling::TokenPenalty> verification_penalty_host_;
  /// Allocated only when concurrent decoding is first requested.
  mutable float* batch_logits_{nullptr};
  mutable Session::Control* batch_controls_{nullptr};
  mutable MtpCandidateLogits* batch_candidates_host_{nullptr};
  // Mapped descriptors, one slice per layer: GPU reads cannot race the host
  // preparing the next layer. ForwardBatch drains before reusing this table.
  mutable GdnBatchItem* batch_gdn_host_{nullptr};
  mutable GdnBatchItem* batch_gdn_{nullptr};
  mutable std::uint32_t batch_rows_{0};
  mutable const float* selected_logits_{nullptr};
  [[nodiscard]] const float* VerificationLogits() const {
    return selected_logits_ != nullptr ? selected_logits_ : s_.logits;
  }
  std::uint32_t mask_words_{0};
  /// Session capacity the sparse-mask rows and score scratch are sized for:
  /// max(native context, Options::max_context).
  std::uint32_t context_capacity_{0};
  /// YaRN constants every rope launch receives (disabled: original path).
  YarnRope yarn_{};
  /// Bounded score scratch, shared by sequential selector chunks.
  std::size_t select_score_floats_{0};
  std::vector<void*> allocations_;
  /// Coherent pinned storage, read directly by the GPU through s_.ple_emb.
  /// Reused only after the preceding forward has drained its stream.
  mutable hip::HostAllocations host_allocations_;
  float* host_emb_{nullptr};
  /// KVarN prompt-chunk staging, one layer at a time: K and V rows of
  /// hip::KvarnStagingRows(max_batch) positions.
  __half* kvarn_staging_k_{nullptr};
  __half* kvarn_staging_v_{nullptr};
  mutable std::vector<std::uint32_t> host_rows_;
  mutable std::vector<NgramGatherOwner> ngram_owners_;
  mutable bool ple_pending_{false};
  /// The rows a prefill chunk reads are a pure function of its tokens and
  /// the n-gram history before them, so prefetched rows are claimed only by
  /// a batch with that history whose tokens extend, or are extended by, the
  /// prefetched ones, whichever session it belongs to. A claim copies them
  /// into host_emb_, whose stable address is retained by captured graphs.
  struct PlePrefetch {
    bool pending{false};  ///< read in flight
    bool ready{false};    ///< rows complete in `rows`
    /// Leading batch tokens whose rows WaitPle collects from `rows`.
    std::size_t claimed{0};
    /// The batch extends past `tokens`; the table reads its remaining rows.
    bool tail{false};
    NgramHistory before;
    NgramHistory after;
    std::vector<std::int32_t> tokens;
    std::vector<float> rows;
  };
  mutable PlePrefetch prefetch_;
  mutable std::span<const std::int32_t> prefetch_next_;
  // Pinned host staging the launched (or captured) work reads and writes.
  Session::Control* control_host_{nullptr};
  std::int32_t* tokens_host_{nullptr};
  std::uint32_t* counts_host_{nullptr};
  std::int32_t* tiles_host_{nullptr};         ///< routed tile map staging
  mutable std::uint32_t routed_max_rows_{0};  ///< 0 = no readback yet
  mutable std::uint32_t routed_n_tiles_{0};   ///< routed_tile_rows_ tiles
  mutable std::uint32_t routed_64_tiles_{0};  ///< wide batches' down tiles
  mutable std::uint32_t routed_pair_offset_{0};
  mutable std::uint32_t routed_pair_tiles_{0};
  mutable std::uint32_t routed_tile_rows_{48};  ///< token rows per tile
  mutable int routed_tile_cols_{0};
  float* logits_host_{nullptr};
  std::int32_t* mtp_token_host_{nullptr};
  MtpCandidateLogits* mtp_candidates_host_{nullptr};
  /// The model geometry allows the wide mixer route (see Combine).
  bool wide_mixer_{false};
  /// Set by Moe when its epilogue is left for the combine that follows.
  mutable bool moe_pending_{false};
  /// Set before an attention-side combine whose FFN combine fuses the MoE
  /// epilogue: that combine may then leave its residual update to the next
  /// one (res_deferred_), which reads s_.block_out and s_.hc_defer_w.
  mutable bool defer_residual_{false};
  mutable bool res_deferred_{false};
  /// Set by GatedDense when s_.shexp_half holds the SwiGLU rows its F16
  /// down projection reads.
  mutable bool shexp_half_ready_{false};
  // What s_.x_q8t / s_.x_half currently hold (input pointer, rows, cols,
  // and the half type), so a projection over the same rows skips its
  // activation pass. Cleared whenever the source buffer is rewritten.
  mutable const float* q8t_src_{nullptr};
  mutable std::uint32_t q8t_rows_{0};
  mutable std::size_t q8t_cols_{0};
  mutable const float* half_src_{nullptr};
  mutable std::uint32_t half_rows_{0};
  mutable std::size_t half_cols_{0};
  mutable bool half_bf16_{false};
  /// Set by a combine that wrote s_.xn_half / s_.xn_q8t instead of s_.xn.
  mutable bool xn_half_{false};
  /// An EXL3 prefill combine that wrote the F16 norm and s_.xn_scale but no
  /// F32 s_.xn: its residual, gamma and rows, for the next mixer to
  /// recompute the norm from (or materialize it into s_.xn).
  struct DeferredNorm {
    const float* res{nullptr};
    const float* gamma{nullptr};
    std::uint32_t rows{0};
  };
  mutable DeferredNorm xn_deferred_{};
  /// The norm (residual, gamma, rows) whose tiled Q8 rows an EXL3 prefill
  /// combine left in s_.xn_q8t for the next mixer's W8A8 down projection.
  mutable DeferredNorm xn_q8t_norm_{};
  /// Every trunk mixer has a Q8_0 down copy (DeviceMixer::down_q8): EXL3
  /// prefill combines then emit tiled Q8 norms instead of F16.
  bool hc_down_q8_{false};
  /// Set before an EXL3 prefill MoE whose output only the next combine
  /// reads: the MoE then leaves its expert sum to that combine
  /// (exl3_moe_pending_), which forms it per token in LDS.
  mutable bool exl3_moe_defer_{false};
  struct PendingExl3Moe {
    const DeviceLayer* layer{nullptr};
    std::uint32_t rows{0};
  };
  mutable PendingExl3Moe exl3_moe_pending_{};
  /// Set by Attention for KvarnAttention: the queries are already in the
  /// KVarN frame (AttentionQueryRows), and the output gate is read from the
  /// packed s_.qg rows (attn_gate_stride_) instead of s_.attn_gate.
  mutable bool attn_q_rotated_{false};
  mutable std::uint32_t attn_gate_stride_{0};
  /// Partial sums per inject logit the last HcMix left in s_.inject.
  mutable std::uint32_t inject_parts_{1};
  /// Inject weights whose partials the last combine already wrote to
  /// s_.inject, so the following HcMix skips its inject pass.
  mutable const float* inject_fused_{nullptr};
  mutable unsigned q8_slot_{0};
  /// Producer-written Q8_1 rows (see ReadyQ8): source, rows and width.
  struct ReadyRows {
    const float* src{nullptr};
    const void* data{nullptr};
    std::uint32_t n{0};
    std::uint32_t k{0};
  };
  mutable ReadyRows xn_q8_ready_;
  mutable ReadyRows mixed_q8_ready_;
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXECUTOR_HPP_
