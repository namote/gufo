#ifndef GUFO_MODELS_QWEN_HIP_EXECUTOR_HPP_
#define GUFO_MODELS_QWEN_HIP_EXECUTOR_HPP_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "src/core/model_config.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/generator.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen/tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include "src/core/hip/bounce_stream.hpp"
#include "src/core/hip/kv_block_pool.hpp"
#include "src/core/hip/detail/hip_graph_decode_executor.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/ssm.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen/vision/device_input.hpp"

namespace gufo::hip {

class HipblasLtGemm;
class QwenGpuExecutor;

struct QwenGpuBatchItem {
  QwenGpuExecutor* executor{nullptr};
  tokenization::TokenId token_id{0};
  std::uint32_t position{0};
};

struct QwenGpuVerificationItem {
  QwenGpuExecutor* executor{nullptr};
  std::span<const tokenization::TokenId> tokens;
  std::uint32_t position{0};
  bool capture_logits{false};
};

struct QwenSampledVerificationResult {
  tokenization::TokenId token{0};
  bool accepted{false};
};

struct QwenGpuWeightRegion {
  // Original reader address for tensor offsets; host_copy owns GPU-visible
  // immutable bytes, backed by transparent huge pages when available.
  const void* host_data{nullptr};
  void* device_data{nullptr};
  void* host_copy{nullptr};
  std::size_t size{0};
};

struct QwenGpuMemoryUsage {
  std::size_t request_state_bytes{0};
  std::size_t temporary_scratch_bytes{0};

  [[nodiscard]] std::size_t TotalBytes() const noexcept {
    return request_state_bytes + temporary_scratch_bytes;
  }
};

/// Immutable GPU-visible Qwen model resources shared by executor sessions.
class QwenGpuModel {
public:
  QwenGpuModel(std::shared_ptr<const core::GgufReader> reader,
               models::QwenModelWeights weights,
               std::shared_ptr<const tokenization::QwenTokenizer> tokenizer,
               std::vector<QwenGpuWeightRegion> weight_regions,
               std::shared_ptr<models::qwen::vision::Encoder> vision = {});
  ~QwenGpuModel();

  QwenGpuModel(const QwenGpuModel&) = delete;
  QwenGpuModel& operator=(const QwenGpuModel&) = delete;
  QwenGpuModel(QwenGpuModel&&) = delete;
  QwenGpuModel& operator=(QwenGpuModel&&) = delete;

  [[nodiscard]] static std::shared_ptr<const QwenGpuModel> CreateFromGguf(
      std::shared_ptr<const core::GgufReader> reader,
      std::string* error_msg = nullptr,
      std::shared_ptr<models::qwen::vision::Encoder> vision = {});

  [[nodiscard]] const std::shared_ptr<models::qwen::vision::Encoder>&
  VisionEncoder() const noexcept {
    return vision_;
  }
  [[nodiscard]] const models::QwenModelWeights& GetWeights() const noexcept {
    return weights_;
  }
  [[nodiscard]] const tokenization::QwenTokenizer& GetTokenizer()
      const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept {
    return weights_.config;
  }
  [[nodiscard]] std::size_t GetWeightRegionCount() const noexcept {
    return weight_regions_.size();
  }
  [[nodiscard]] std::size_t GetResidentBytes() const noexcept;

private:
  // Keep the mapped GGUF storage alive until every registered region is
  // released.
  std::shared_ptr<const core::GgufReader> reader_;
  models::QwenModelWeights weights_;
  std::shared_ptr<const tokenization::QwenTokenizer> tokenizer_;
  std::vector<QwenGpuWeightRegion> weight_regions_;
  std::shared_ptr<models::qwen::vision::Encoder> vision_;
};

class QwenGpuSnapshot final {
public:
  /// One piece of the compact form in order: host bytes, or a device
  /// region of this snapshot packed row by row.
  struct CompactSegment {
    std::vector<std::uint8_t> host;
    DeviceRegion device;
  };
  ~QwenGpuSnapshot();

  QwenGpuSnapshot(const QwenGpuSnapshot&) = delete;
  QwenGpuSnapshot& operator=(const QwenGpuSnapshot&) = delete;
  QwenGpuSnapshot(QwenGpuSnapshot&&) = delete;
  QwenGpuSnapshot& operator=(QwenGpuSnapshot&&) = delete;

  [[nodiscard]] std::size_t PayloadBytes() const noexcept {
    return payload_bytes_ + vision_layout_.images.size() *
                                sizeof(models::qwen::vision::ImageGrid);
  }
  [[nodiscard]] std::uint32_t ValidContext() const noexcept {
    return valid_context_;
  }
  /// Pool bytes of the KV blocks this snapshot shares with arenas instead of
  /// copying them; zero for a copied snapshot. Not part of PayloadBytes.
  [[nodiscard]] std::size_t SharedKvBytes() const noexcept;
  [[nodiscard]] std::span<const std::uint32_t> SharedKvBlocks() const noexcept {
    return kv_blocks_;
  }
  [[nodiscard]] QwenKvCacheStorage KvStorage() const noexcept {
    return kv_storage_;
  }
  [[nodiscard]] QwenRecurrentStateStorage RecurrentStateStorage()
      const noexcept {
    return recurrent_state_storage_;
  }
  /// `paged`: the compact form without the full KV blocks, which a
  /// block-sharing snapshot stores apart (KvBlockRegions); the rows past the
  /// last full block stay inline.
  [[nodiscard]] std::vector<CompactSegment> CompactSegments(
      bool paged = false) const;
  [[nodiscard]] std::size_t CompactPayloadBytes(bool paged = false) const;
  [[nodiscard]] std::size_t SerializeCompact(std::span<std::uint8_t> destination,
                                             bool paged = false) const;
  /// K then V rows of shared block `block`, all layers.
  [[nodiscard]] std::array<DeviceRegion, 2> KvBlockRegions(
      std::uint32_t block) const;
  /// The bytes of SerializeCompact through a bounce stream, without a
  /// full-size host buffer. Both use CompactSegments, so they are identical.
  void StreamCompact(BounceStream& bounce, const BounceStream::Sink& sink,
                     bool paged = false) const;
  [[nodiscard]] static models::qwen::vision::RopeLayout ReadRopeLayout(
      std::span<const std::uint8_t> payload);

private:
  QwenGpuSnapshot() = default;

  void* d_kv_f32_{nullptr};
  void* d_kv_f16_{nullptr};
  // Shared rows: one reference to each block holding [0, valid_context_).
  std::shared_ptr<KvBlockPool> kv_pool_;
  std::vector<std::uint32_t> kv_blocks_;
  void* d_ssm_conv_{nullptr};
  void* d_ssm_deltanet_{nullptr};
  std::size_t kv_elements_per_plane_{0};
  std::size_t conv_elements_{0};
  std::size_t deltanet_elements_{0};
  std::size_t conv_elements_per_layer_{0};
  std::size_t deltanet_elements_per_layer_{0};
  std::uint32_t num_layers_{0};
  std::uint32_t attention_layers_{0};
  std::uint32_t full_attention_interval_{0};
  std::uint32_t kv_width_{0};
  std::uint32_t max_context_{0};
  std::uint32_t valid_context_{0};
  std::size_t payload_bytes_{0};
  QwenKvCacheStorage kv_storage_{QwenKvCacheStorage::kFp32};
  QwenRecurrentStateStorage recurrent_state_storage_{
      QwenRecurrentStateStorage::kFp32};

  friend class QwenGpuArena;
  friend class QwenGpuExecutor;
  models::qwen::vision::RopeLayout vision_layout_;
};

/// Shared decode-lifetime workspaces over stable arena allocations.
struct QwenDecodeScratch {
  std::span<float> hidden;
  std::span<float> normed;
  std::span<float> logits;
  std::span<hip_bfloat16> bf16;
  std::span<hip_bfloat16> weight_bf16;
  std::span<std::uint32_t> prompt_tokens;

  /// Aliases the first element of QwenSsmScratch::alpha. It is legal only
  /// after all layer execution has completed, during the sampling/output epoch.
  std::span<std::uint32_t> sampled_token;
};

struct QwenAttentionScratch {
  std::span<float> q;
  std::span<float> k;
  std::span<float> v;
  std::span<float> output;
  std::span<float> split_k;
};

struct QwenSsmScratch {
  std::span<float> qkv;
  std::span<float> conv_out;
  std::span<float> gate;
  std::span<float> out;
  std::span<float> alpha;
  std::span<float> beta;
};

struct QwenFfnScratch {
  std::span<float> gate;
  std::span<float> up;
  std::span<float> activation;
  std::span<float> out;
};

/// Typed non-owning capability views over stable Qwen GPU arena allocations.
/// The nested aggregates carry element counts without owning memory or changing
/// any address used by kernels or graph capture.
struct QwenGpuScratchView {
  QwenDecodeScratch decode;
  QwenAttentionScratch attention;
  QwenSsmScratch ssm;
  QwenFfnScratch ffn;
};

static_assert(std::is_trivially_copyable_v<QwenDecodeScratch>);
static_assert(std::is_trivially_copyable_v<QwenAttentionScratch>);
static_assert(std::is_trivially_copyable_v<QwenSsmScratch>);
static_assert(std::is_trivially_copyable_v<QwenFfnScratch>);
static_assert(std::is_trivially_copyable_v<QwenGpuScratchView>);
static_assert(std::is_same_v<decltype(QwenDecodeScratch::sampled_token),
                             std::span<std::uint32_t>>);

template<typename T>
[[nodiscard]] constexpr T* OffsetIfPresent(T* pointer,
                                           std::size_t offset) noexcept {
  return pointer == nullptr ? nullptr : pointer + offset;
}

/// Preallocated, zero-allocation GPU execution arena on gfx1151.
/// The buffers of one forward pass that grow with its rows: activations,
/// projections and the BF16/Q8 GEMM inputs. Their contents live for one pass
/// only. A session owns a set; with a QwenSharedPassScratch its own set holds
/// only the rows of a single-token decode, and wider passes lease the shared
/// set, which holds the widest pass's rows for every session.
struct QwenPassScratchSet {
  std::size_t rows{0};
  float* hidden{nullptr};
  float* normed{nullptr};
  float* q{nullptr};
  float* k{nullptr};
  float* v{nullptr};
  float* attn_out{nullptr};
  float* ffn_gate{nullptr};
  float* ffn_up{nullptr};
  float* ffn_act{nullptr};
  float* ffn_out{nullptr};
  float* ssm_qkv{nullptr};
  float* conv_out{nullptr};
  float* ssm_gate{nullptr};
  float* ssm_out{nullptr};
  void* bf16{nullptr};
  void* q8_act{nullptr};
  /// The largest layer weight in BF16 (prefill dequantization, packed batch
  /// FFN); rows do not change it. Null in a session set that leases.
  hip_bfloat16* weights_bf16{nullptr};

  [[nodiscard]] static QwenPassScratchSet Allocate(
      const core::ModelConfig& config, std::size_t rows, bool weights);
  [[nodiscard]] static std::size_t Bytes(const core::ModelConfig& config,
                                         std::size_t rows, bool weights);
  void Free() noexcept;
};

/// One pass scratch set for every session of a runner, leased by one pass at
/// a time (passes run one at a time). The next lessee's stream waits for the
/// previous pass's work on it. Leasing it while leased is a scheduling error
/// and throws.
class QwenSharedPassScratch {
public:
  QwenSharedPassScratch(const core::ModelConfig& config,
                        std::uint32_t max_context);
  ~QwenSharedPassScratch();
  QwenSharedPassScratch(const QwenSharedPassScratch&) = delete;
  QwenSharedPassScratch& operator=(const QwenSharedPassScratch&) = delete;

  [[nodiscard]] std::size_t rows() const noexcept { return set_.rows; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
  friend class QwenGpuArena;
  QwenPassScratchSet set_;
  std::size_t bytes_{0};
  hipEvent_t released_{nullptr};
  std::atomic<bool> leased_{false};
};

class QwenGpuArena {
public:
  /// Longest prompt pass (rows) one forward runs; longer prompts take
  /// several.
  static constexpr std::uint32_t kMaxPromptBatch = 4096;
  /// FP16 attention rows live in `kv_pool`, which may be shared with other
  /// arenas. kWholeContext takes the blocks the whole context needs up front;
  /// kOnDemand starts empty and maps blocks as rows are written
  /// (EnsureKvRows). Without a pool the arena makes its own, sized to the
  /// context.
  enum class KvGrowth : std::uint8_t { kWholeContext, kOnDemand };
  explicit QwenGpuArena(
      const core::ModelConfig& config, std::uint32_t max_context = 4096,
      QwenExecutionPolicy policy = QwenExecutionPolicy::Production(),
      std::shared_ptr<KvBlockPool> kv_pool = nullptr,
      KvGrowth kv_growth = KvGrowth::kWholeContext,
      std::shared_ptr<QwenSharedPassScratch> shared_scratch = nullptr);
  ~QwenGpuArena();

  QwenGpuArena(const QwenGpuArena&) = delete;
  QwenGpuArena& operator=(const QwenGpuArena&) = delete;
  QwenGpuArena(QwenGpuArena&&) noexcept;
  QwenGpuArena& operator=(QwenGpuArena&&) noexcept;

  void Reset();
  /// Pins the live recurrent state for rollback. Neither save nor restore
  /// copies anything: the state lives in one of two buffers, and the first
  /// pass after either call reads the pinned buffer and writes the other one.
  void SaveState(std::uint32_t valid_context);
  void RestoreState();
  /// Names the buffers one recurrent-state pass reads and writes. Call exactly
  /// once per forward, replay or prefill pass, before its first launch; every
  /// launch of the pass then writes `*_out`, so the pinned buffer survives
  /// until the next SaveState.
  [[nodiscard]] QwenRecurrentStateIo BeginRecurrentWrite();
  /// Buffer indices of the most recent BeginRecurrentWrite, for graph keys.
  [[nodiscard]] std::uint64_t RecurrentIoIdentity() const noexcept;
  [[nodiscard]] bool BeginSsmReplayCapture();
  void DisableSsmReplayCapture();
  void MarkSsmReplayPosition(std::uint32_t position) noexcept;
  [[nodiscard]] bool CanReplaySsmPosition(
      std::uint32_t position) const noexcept;
  [[nodiscard]] SsmReplayCapture GetSsmReplayCapture() const noexcept;
  [[nodiscard]] bool IsSsmReplayCaptureActive() const noexcept {
    return replay_capture_active_;
  }
  [[nodiscard]] const float* GetReplayQkv(std::uint32_t layer,
                                          std::uint32_t position) const;
  [[nodiscard]] const float* GetReplayAlpha(std::uint32_t layer,
                                            std::uint32_t position) const;
  [[nodiscard]] const float* GetReplayBeta(std::uint32_t layer,
                                           std::uint32_t position) const;

  float* d_hidden{nullptr};
  float* d_normed{nullptr};
  float* d_q{nullptr};
  float* d_k{nullptr};
  float* d_v{nullptr};
  float* d_attn_out{nullptr};
  float* d_ffn_gate{nullptr};
  float* d_ffn_up{nullptr};
  float* d_ffn_act{nullptr};
  float* d_ffn_out{nullptr};
  float* d_ssm_qkv{nullptr};
  float* d_conv_out{nullptr};
  float* d_ssm_gate{nullptr};
  float* d_ssm_out{nullptr};
  float* d_alpha_buf{nullptr};
  float* d_beta_buf{nullptr};
  // opt-c170-deltanet-rowsplit prologue scratch: 3 floats per (token, key head)
  // for the k/q norms and the unnormalized k . q, and 2 floats per (token,
  // value head) for the decay and beta gates.
  float* d_ssm_kq_scales{nullptr};
  float* d_ssm_alpha_beta{nullptr};
  float* d_logits{nullptr};
  void* d_attention_kv_f16{nullptr};
  float* d_kv_cache{nullptr};
  // Mirrors of the live recurrent buffers (see BeginRecurrentWrite); reassigned
  // whenever the live buffer changes, so snapshots and resets read and write
  // the current state without knowing about the rollback pair.
  float* d_ssm_conv_state{nullptr};
  void* d_ssm_deltanet_state{nullptr};
  std::uint32_t* d_prompt_tokens{nullptr};
  float* d_target_layer_features{nullptr};

  hipStream_t stream{nullptr};
  hipblasHandle_t hipblas_handle{nullptr};
  std::unique_ptr<HipblasLtGemm> hipblaslt_gemm;
  void* d_scratch_bf16{nullptr};
  float* d_split_k_attention{nullptr};
  hip_bfloat16* d_weights_bf16{nullptr};
  void* d_scratch_q8_act{nullptr};

  /// Session pass scratch rows with a shared pass scratch: a single-token
  /// decode; batched decode and verification lease the shared set.
  static constexpr std::uint32_t kSessionScratchRows = 64;
  /// Rows the current pass scratch holds: the shared set's while leased.
  [[nodiscard]] std::size_t GetScratchRows() const noexcept {
    return scratch_leased_ ? shared_scratch_->rows() : own_scratch_.rows;
  }
  [[nodiscard]] std::size_t GetOwnScratchRows() const noexcept {
    return own_scratch_.rows;
  }
  [[nodiscard]] float* GetOwnHidden() const noexcept {
    return own_scratch_.hidden;
  }
  [[nodiscard]] bool IsScratchLeased() const noexcept { return scratch_leased_; }
  /// Points this arena's pass buffers at the shared set until
  /// EndPassScratch. False, with nothing changed, without a shared set or
  /// when this arena holds it already (a nested pass). Throws
  /// std::logic_error when another arena holds it.
  [[nodiscard]] bool BeginPassScratch();
  void EndPassScratch() noexcept;

  [[nodiscard]] std::uint32_t GetMaxBatch() const noexcept {
    return max_batch_;
  }
  [[nodiscard]] std::uint32_t GetMaxContext() const noexcept {
    return max_context_;
  }
  /// Elements of one plane (K or V) of the attention cache: the whole pool
  /// for FP16 rows, this arena's context for the FP32 reference. The V plane
  /// starts this many elements after K.
  [[nodiscard]] std::size_t GetAttentionKvPlaneElements() const noexcept {
    // Qwen27B reaches 2^32 elements per plane at context 262144.
    const std::size_t rows =
        kv_pool_ != nullptr ? kv_pool_->layer_tokens() : max_context_;
    return static_cast<std::size_t>(config_.FullAttentionLayerCount()) *
           config_.num_key_value_heads * rows * config_.head_dim;
  }
  /// Where this arena's FP16 rows sit in the pool; null (contiguous) for the
  /// FP32 reference cache.
  [[nodiscard]] KvBlockTable GetKvBlockTable() const noexcept {
    return kv_pool_ == nullptr
               ? KvBlockTable{}
               : KvBlockTable{d_kv_block_table_, kv_pool_->layer_tokens()};
  }
  [[nodiscard]] std::span<const std::uint32_t> GetKvBlocks() const noexcept {
    return kv_blocks_;
  }
  [[nodiscard]] const std::shared_ptr<KvBlockPool>& GetKvPool() const noexcept {
    return kv_pool_;
  }
  /// Rows [0, GetKvMappedRows()) have blocks.
  [[nodiscard]] std::uint32_t GetKvMappedRows() const noexcept {
    return static_cast<std::uint32_t>(kv_blocks_.size()) * kKvBlockTokens;
  }
  /// Makes rows [begin, end) writable before a pass writes them: maps missing
  /// blocks (from the reservation first) and copies any block another owner
  /// shares (copy-on-write). Throws std::length_error when the pool has no
  /// block left; nothing is written then. No-op for the FP32 reference.
  void EnsureKvRows(std::uint32_t begin, std::uint32_t end);
  /// Returns the blocks past `rows` to the pool (on-demand arenas only).
  void TruncateKvRows(std::uint32_t rows);
  /// One new reference to each block holding rows [0, rows): a sharer reads
  /// them and never writes; this arena copies a shared block before writing.
  [[nodiscard]] std::vector<std::uint32_t> ShareKvBlocks(std::uint32_t rows);
  /// Replaces this arena's blocks with `blocks` (taking one new reference to
  /// each) so its rows are theirs without a copy.
  void AdoptKvBlocks(std::span<const std::uint32_t> blocks);
  /// Points mapped block `logical` at pool block `block`, which holds the
  /// same rows, taking one reference to it and dropping the old block.
  void ReplaceKvBlock(std::uint32_t logical, std::uint32_t block);
  /// Blocks EnsureKvRows draws on before the pool's free blocks.
  void SetKvReservation(KvBlockReservation reservation) noexcept {
    kv_reservation_ = std::move(reservation);
  }
  [[nodiscard]] std::uint32_t GetKvReservedBlocks() const noexcept {
    return kv_reservation_.blocks();
  }
  [[nodiscard]] QwenRecurrentStateStorage GetRecurrentStateStorage()
      const noexcept {
    return policy_.recurrent_state_storage;
  }
  /// On-demand arenas leave FP16 KV out of request_state_bytes: those rows
  /// are the shared pool's.
  /// `shared_scratch`: the session's own pass scratch only
  /// (kSessionScratchRows); the shared set is the runner's.
  [[nodiscard]] static QwenGpuMemoryUsage EstimateMemoryUsage(
      const core::ModelConfig& config, std::uint32_t max_context,
      QwenExecutionPolicy policy = QwenExecutionPolicy::Production(),
      KvGrowth kv_growth = KvGrowth::kWholeContext,
      bool shared_scratch = false);
  [[nodiscard]] QwenGpuMemoryUsage GetMemoryUsage() const;
  /// Bytes a snapshot at valid_context owns; shared KV blocks are not its.
  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      std::uint32_t valid_context) const;
  /// QwenGpuSnapshot::CompactPayloadBytes() for a text-only snapshot at
  /// valid_context, from the arena contract alone, before any state exists.
  /// Each image adds 12 bytes.
  /// The compact form of this arena's live state at valid_context, read in
  /// place: no device or host copy. The state must not change until the
  /// segments are consumed.
  [[nodiscard]] std::vector<QwenGpuSnapshot::CompactSegment>
  LiveCompactSegments(std::uint32_t valid_context,
                      const models::qwen::vision::RopeLayout& rope,
                      bool paged = false);
  [[nodiscard]] static std::size_t CompactSnapshotPayloadBytes(
      const core::ModelConfig& config, std::uint32_t max_context,
      QwenExecutionPolicy policy, std::uint32_t valid_context,
      bool paged = false);
  /// K then V rows of mapped block `block`, all layers, read in place.
  [[nodiscard]] std::array<DeviceRegion, 2> KvBlockRegions(
      std::uint32_t block) const;
  /// Fills mapped block `block` (K then V) from a stream: the full blocks a
  /// paged compact restore leaves to the caller.
  void RestoreKvBlock(std::uint32_t block, BounceStream& bounce,
                      const BounceStream::Source& read);

  [[nodiscard]] std::unique_ptr<QwenGpuSnapshot> SaveSnapshot(
      std::uint32_t valid_context);
  void RestoreSnapshot(const QwenGpuSnapshot& snapshot);
  void RestoreCompactSnapshot(std::span<const std::uint8_t> payload,
                              std::uint32_t expected_valid_context);
  [[nodiscard]] models::qwen::vision::RopeLayout RestoreCompactSnapshotStream(
      BounceStream& bounce, const BounceStream::Source& read,
      std::size_t compact_bytes, std::uint32_t expected_valid_context);
  /// Throws std::logic_error past GetScratchRows(): a wide pass must lease
  /// the shared pass scratch first.
  [[nodiscard]] QwenGpuScratchView GetScratchView(std::size_t batch_size = 1);
  void SetTargetLayerCapture(std::span<const std::uint32_t> target_layer_ids);
  [[nodiscard]] std::span<const std::uint32_t> GetTargetLayerCapture()
      const noexcept {
    return target_layer_ids_;
  }
  [[nodiscard]] std::optional<std::size_t> GetTargetLayerCaptureIndex(
      std::uint32_t layer) const noexcept;

private:
  void FreeAll() noexcept;
  void AllocateKvBlocks(std::shared_ptr<KvBlockPool> kv_pool,
                        KvGrowth kv_growth);
  /// Maps fresh blocks for rows [0, rows) that a restore overwrites whole.
  void PrepareKvOverwrite(std::uint32_t rows);
  /// Where a compact restore writes its inline rows of one plane.
  [[nodiscard]] std::vector<MutableDeviceRegion> RestoreRegions(
      std::uint32_t plane, std::uint32_t valid_context, bool paged) const;
  /// Uploads table entries [first, last) from the pinned mirror on `stream`.
  void UploadKvTable(std::size_t first, std::size_t last);
  /// Copies block `from`'s rows into `to`, both planes, on `stream`.
  void CopyKvBlock(std::uint32_t from, std::uint32_t to);
  /// One plane of the rows [0, valid_context) in serialized order.
  [[nodiscard]] std::vector<MutableDeviceRegion> KvPlaneRegions(
      std::uint32_t plane, std::uint32_t valid_context) const;
  void AllocateRecurrentSnapshot();
  void SyncLivePointers() noexcept;
  [[nodiscard]] bool AllocateSsmReplayLog();
  void PointPassScratch(const QwenPassScratchSet& set) noexcept;

  core::ModelConfig config_;
  std::uint32_t max_context_;
  std::uint32_t max_batch_;
  QwenExecutionPolicy policy_;
  std::vector<std::uint32_t> target_layer_ids_;
  std::shared_ptr<KvBlockPool> kv_pool_;
  KvGrowth kv_growth_{KvGrowth::kWholeContext};
  std::vector<std::uint32_t> kv_blocks_;
  std::uint32_t* d_kv_block_table_{nullptr};
  // Pinned copy of the device table. An entry already uploaded is rewritten
  // only after `stream` drains, so a pending upload never reads a newer one.
  std::uint32_t* h_kv_block_table_{nullptr};
  KvBlockReservation kv_reservation_;
  // Buffer 0 comes from the constructor; buffer 1 is allocated by the first
  // SaveState. `live_recurrent_` holds the current state, `saved_recurrent_`
  // the pinned one (meaningful while has_saved_state_), `source_recurrent_`
  // the buffer the most recent pass read.
  std::array<float*, 2> d_ssm_conv_buffers_{};
  std::array<void*, 2> d_ssm_deltanet_buffers_{};
  std::uint8_t live_recurrent_{0};
  std::uint8_t saved_recurrent_{0};
  std::uint8_t source_recurrent_{0};
  float* d_ssm_replay_qkv_{nullptr};
  float* d_ssm_replay_alpha_{nullptr};
  float* d_ssm_replay_beta_{nullptr};
  std::uint32_t* d_ssm_replay_enabled_{nullptr};
  std::uint32_t saved_context_{0};
  std::uint32_t replay_last_position_{0};
  std::size_t replay_captured_positions_{0};
  bool has_saved_state_{false};
  bool replay_capture_active_{false};
  QwenPassScratchSet own_scratch_;
  std::shared_ptr<QwenSharedPassScratch> shared_scratch_;
  bool scratch_leased_{false};
};

/// Where an executor's FP16 attention rows live: its own pool sized to the
/// context by default, or blocks of a pool shared with other executors.
struct QwenKvPlacement {
  std::shared_ptr<KvBlockPool> pool;
  QwenGpuArena::KvGrowth growth{QwenGpuArena::KvGrowth::kWholeContext};
  /// Pass scratch shared with other executors; null: the executor owns a
  /// full set.
  std::shared_ptr<QwenSharedPassScratch> pass_scratch;
};

/// End-to-end GPU model executor running directly on the gfx1151 RDNA 3.5 CUs.
class QwenGpuExecutor {
public:
  explicit QwenGpuExecutor(
      std::shared_ptr<const QwenGpuModel> model,
      std::uint32_t max_context = 4096,
      QwenExecutionPolicy policy = QwenExecutionPolicy::Production(),
      QwenKvPlacement kv_placement = {});
  ~QwenGpuExecutor();

  [[nodiscard]] static std::unique_ptr<QwenGpuExecutor> Create(
      std::shared_ptr<const QwenGpuModel> model,
      std::string* error_msg = nullptr, std::uint32_t max_context = 4096,
      QwenExecutionPolicy policy = QwenExecutionPolicy::Production(),
      QwenKvPlacement kv_placement = {});

  [[nodiscard]] static std::unique_ptr<QwenGpuExecutor> CreateFromGguf(
      std::shared_ptr<const core::GgufReader> reader,
      std::string* error_msg = nullptr, std::uint32_t max_context = 4096,
      QwenExecutionPolicy policy = QwenExecutionPolicy::Production());

  /// Generates tokens auto-regressively on GPU with streaming callback.
  std::vector<tokenization::TokenId> Generate(
      std::span<const tokenization::TokenId> prompt_tokens,
      const models::GenerationOptions& options,
      const std::function<bool(tokenization::TokenId, std::string_view)>&
          on_token = nullptr);

  /// Continues generation from an exact retained prompt prefix.
  ///
  /// The executor state must represent prompt_tokens[0:cached_prefix_tokens].
  /// A zero prefix is a cold prefill over the complete prompt.
  std::vector<tokenization::TokenId> GenerateFromPrefix(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::size_t cached_prefix_tokens,
      const models::GenerationOptions& options,
      const std::function<bool(tokenization::TokenId, std::string_view)>&
          on_token = nullptr);

  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept {
    return weights_.config;
  }

  [[nodiscard]] const tokenization::QwenTokenizer& GetTokenizer()
      const noexcept {
    return *tokenizer_;
  }

  [[nodiscard]] const QwenGpuModel& GetModel() const noexcept {
    return *model_;
  }

  [[nodiscard]] const std::shared_ptr<const QwenGpuModel>& GetSharedModel()
      const noexcept {
    return model_;
  }

  [[nodiscard]] const QwenExecutionPolicy& GetExecutionPolicy() const noexcept {
    return policy_;
  }

  /// Runs one single token forward step on GPU, returning next token ID.
  [[nodiscard]] tokenization::TokenId ForwardToken(
      tokenization::TokenId token_id, std::uint32_t pos,
      bool compute_logits = true);

  /// Advances two to eight independent retained sessions in one
  /// layer-synchronous decode step. Stateless projections share weight reads;
  /// every row still addresses its own KV cache and recurrent state.
  [[nodiscard]] static std::vector<tokenization::TokenId> ForwardTokenBatch(
      std::span<const QwenGpuBatchItem> items);

  /// Runs batched prompt prefill on GPU, returning the first predicted token
  /// ID.
  [[nodiscard]] tokenization::TokenId ForwardPromptBatch(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t start_pos = 0, bool compute_logits = true);

  /// Verifies a chunk of speculative draft candidate tokens in a single
  /// parallel prefill forward pass on the GPU.
  [[nodiscard]] std::vector<tokenization::TokenId> ForwardVerificationChunk(
      std::span<const tokenization::TokenId> candidate_tokens,
      std::uint32_t start_pos, bool capture_logits = false);
  /// Shares exact projections across independent speculative chunks. Each
  /// sequence retains its own causal attention, recurrence and replay state.
  [[nodiscard]] static std::vector<std::vector<tokenization::TokenId>>
  ForwardVerificationBatch(std::span<const QwenGpuVerificationItem> items);
  void CommitVerificationChunk(
      std::span<const tokenization::TokenId> committed_tokens,
      std::uint32_t start_pos);
  [[nodiscard]] std::span<const float> GetVerificationHiddenStates()
      const noexcept {
    return h_verification_hidden_;
  }
  [[nodiscard]] std::span<const float> GetVerificationLogits() const noexcept {
    return h_verification_logits_;
  }
  [[nodiscard]] std::span<const float> CopyVerificationLogits(std::size_t row);

  /// Copies the logits produced by the most recent forward pass to host memory.
  [[nodiscard]] std::span<const float> CopyLastLogits();

  /// Samples the most recent device-resident logit row and transfers one token.
  [[nodiscard]] tokenization::TokenId SampleLastLogits(
      sampling::SamplerState& sampler);

  /// Restores a saved frontier row and samples with the same GPU operations
  /// as a fresh autoregressive request.
  [[nodiscard]] tokenization::TokenId SampleCachedLogits(
      std::span<const float> logits, sampling::SamplerState& sampler);

  /// Samples one row from the most recent verification batch without copying
  /// its vocabulary-sized logits to the host.
  [[nodiscard]] tokenization::TokenId SampleVerificationLogits(
      std::size_t row, sampling::SamplerState& sampler);

  /// Performs one exact sampled-speculation accept/residual decision on the
  /// GPU, returning only the decision and selected token.
  [[nodiscard]] QwenSampledVerificationResult VerifySampledToken(
      std::size_t row, tokenization::TokenId draft_token,
      std::span<const tokenization::TokenId> draft_candidate_ids,
      std::span<const float> draft_candidate_probabilities,
      double draft_token_probability, sampling::SamplerState& sampler);

  /// Enables host capture of every final-layer prompt hidden state. Disabled
  /// by default so ordinary prefill does not incur device-to-host copies.
  void SetPromptHiddenCapture(
      bool enabled, std::span<const std::uint32_t> target_layer_ids = {});

  /// Returns the flattened [prompt_tokens, hidden_size] capture from the most
  /// recent ForwardPromptBatch call.
  [[nodiscard]] std::span<const float> GetPromptHiddenStates() const noexcept {
    return h_prompt_hidden_;
  }

  /// Copies the final-layer hidden state from the most recent forward pass.
  [[nodiscard]] std::span<const float> CopyLastHidden();

  [[nodiscard]] std::uint32_t GetMaxPromptBatch() const noexcept {
    return arena_.GetMaxBatch();
  }
  [[nodiscard]] std::uint32_t GetMaxContext() const noexcept {
    return arena_.GetMaxContext();
  }
  [[nodiscard]] static QwenGpuMemoryUsage EstimateMemoryUsage(
      const core::ModelConfig& config, std::uint32_t max_context,
      QwenExecutionPolicy policy = QwenExecutionPolicy::Production(),
      QwenGpuArena::KvGrowth kv_growth = QwenGpuArena::KvGrowth::kWholeContext,
      bool shared_scratch = false);
  [[nodiscard]] QwenGpuMemoryUsage GetMemoryUsage() const;
  [[nodiscard]] QwenGpuArena& GetArena() noexcept { return arena_; }
  [[nodiscard]] const QwenGpuArena& GetArena() const noexcept {
    return arena_;
  }
  void SetCancellationCheck(std::function<bool()> check) {
    cancellation_check_ = std::move(check);
    vision_input_.SetCancellationCheck(cancellation_check_);
  }
  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      std::uint32_t valid_context) const;
  void ConfigureVision(
      std::shared_ptr<const models::qwen::vision::Prompt> prompt,
      std::shared_ptr<models::qwen::vision::Encoder> encoder);
  [[nodiscard]] models::qwen::vision::DeviceInput& VisionInput() noexcept {
    return vision_input_;
  }
  [[nodiscard]] const models::qwen::vision::RopeLayout& VisionLayout()
      const noexcept {
    return vision_input_.layout();
  }
  [[nodiscard]] std::unique_ptr<QwenGpuSnapshot> SaveSnapshot(
      std::uint32_t valid_context);
  void RestoreSnapshot(const QwenGpuSnapshot& snapshot);
  void RestoreCompactSnapshot(std::span<const std::uint8_t> payload,
                              std::uint32_t expected_valid_context);
  /// RestoreCompactSnapshot from a stream of `compact_bytes` bytes, through
  /// a bounce buffer instead of a full-size host copy.
  /// SaveSnapshot(valid_context) then StreamCompact, without the snapshot:
  /// the bytes stream straight from this session's buffers.
  void StreamLiveCompact(std::uint32_t valid_context, BounceStream& bounce,
                         const BounceStream::Sink& sink, bool paged = false);
  /// Bytes StreamLiveCompact(valid_context, paged) produces.
  [[nodiscard]] std::size_t LiveCompactPayloadBytes(
      std::uint32_t valid_context, bool paged = false) const;
  void RestoreCompactSnapshotStream(BounceStream& bounce,
                                    const BounceStream::Source& read,
                                    std::size_t compact_bytes,
                                    std::uint32_t expected_valid_context);

  /// Resets GPU state. Any reset failure is retained and rethrown by the next
  /// fallible operation; invalidation/cleanup itself remains noexcept.
  void Reset() noexcept;
  void SaveState(std::uint32_t valid_context);
  void RestoreState();
  /// Ends replay capture after the last verification chunk.
  void FinishVerification();

private:
  void CheckPrefillCancellation() const;
  void PrefillLayerCheckpoint(std::uint32_t layer);
  /// Leases the shared pass scratch for one prompt, verification or batched
  /// decode pass when the executor has one. On release the hidden rows later
  /// calls read (the pass's last row, or a batch's first rows) move back to
  /// the session's own set.
  class PassScratchLease {
  public:
    explicit PassScratchLease(QwenGpuExecutor& executor)
        : executor_(executor), leased_(executor.arena_.BeginPassScratch()) {}
    ~PassScratchLease() {
      if (leased_)
        executor_.ReturnPassScratch();
    }
    PassScratchLease(const PassScratchLease&) = delete;
    PassScratchLease& operator=(const PassScratchLease&) = delete;

  private:
    QwenGpuExecutor& executor_;
    bool leased_;
  };
  void ReturnPassScratch() noexcept;
  std::function<bool()> cancellation_check_;
  std::array<hipEvent_t, 2> prefill_events_{};
  void CheckReset() const {
    if (reset_failure_)
      std::rethrow_exception(reset_failure_);
  }
  std::exception_ptr reset_failure_;
  void ReplaySsmState(std::uint32_t position, std::uint32_t count = 1);
  void EnsureVerificationLogits(std::size_t batch_size);
  [[nodiscard]] GpuSamplingParameters PrepareGpuSamplingParameters(
      const sampling::SamplerState& sampler);

  [[nodiscard]] tokenization::TokenId ForwardPromptChunk(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t start_pos, bool compute_logits);
  [[nodiscard]] std::vector<tokenization::TokenId>
  ForwardDecodeEquivalentVerificationChunk(
      std::span<const tokenization::TokenId> candidate_tokens,
      std::uint32_t start_pos, bool capture_logits);

  std::shared_ptr<const QwenGpuModel> model_;
  const models::QwenModelWeights& weights_;
  const tokenization::QwenTokenizer* tokenizer_;
  const QwenExecutionPolicy policy_;
  QwenGpuArena arena_;
  const detail::HipGraphCaptureKey graph_key_;
  detail::HipGraphDecodeExecutor graph_executor_;
  models::qwen::vision::DeviceInput vision_input_;
  std::vector<float> h_logits_;
  std::vector<float> h_prompt_hidden_;
  std::vector<float> h_verification_hidden_;
  std::vector<float> h_verification_logits_;
  std::vector<float> h_last_hidden_;
  float* d_verification_logits_{nullptr};
  std::size_t verification_logits_capacity_{0};
  std::size_t last_verification_rows_{0};
  std::size_t last_hidden_offset_{0};
  float* d_target_layer_features_{nullptr};
  GpuSamplingWorkspace sampling_workspace_;
  std::optional<tokenization::TokenId> next_token_;
  bool capture_prompt_hidden_{false};
  bool replaying_ssm_state_{false};
};

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_EXECUTOR_HPP_
