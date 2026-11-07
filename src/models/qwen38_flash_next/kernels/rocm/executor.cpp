#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "qfn_mmq.h"
#include "src/core/hip/kv_block_layout.hpp"
#include "src/core/hip/managed_alloc.hpp"
#include "src/core/hip/snapshot_transfer.hpp"
#include "src/core/platform/pages.hpp"
#include "src/core/quant/exl3/gemv.hpp"
#include "src/models/qwen38_flash_next/draft_vocab.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

using core::GgmlType;

void AssignError(std::string* error_msg, const std::string& message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

bool Check(hipError_t err, const char* what, std::string* error_msg) {
  if (err == hipSuccess) {
    return true;
  }
  AssignError(error_msg, std::string(what) + ": " + hipGetErrorString(err));
  return false;
}

/// Allocates `count` elements of T and records the allocation.
template<typename T>
T* Alloc(std::vector<void*>& allocations, std::size_t count,
         std::string* error_msg, std::size_t* allocated_bytes = nullptr) {
  void* p = nullptr;
  const std::size_t bytes = std::max<std::size_t>(1, count) * sizeof(T);
  if (hip::AllocateDevice(&p, bytes) != hipSuccess) {
    AssignError(error_msg,
                "allocation of " + std::to_string(bytes) + " bytes failed");
    allocations.push_back(nullptr);
    return nullptr;
  }
  (void)hipMemset(p, 0, bytes);
  allocations.push_back(p);
  if (allocated_bytes)
    *allocated_bytes += bytes;
  return static_cast<T*>(p);
}

WeightType SmallType(GgmlType type) {
  switch (type) {
    case GgmlType::kBF16:
      return WeightType::kBF16;
    case GgmlType::kF16:
      return WeightType::kF16;
    case GgmlType::kQ8_0:
      return WeightType::kQ8_0;
    case GgmlType::kF32:
      return WeightType::kF32;
    default:
      throw std::logic_error("unsupported Flash-Next matrix format");
  }
}

// The tier's tiled kernels compute whole column tiles; below this width the
// matrix-vector kernels read each weight once per row and win outright.
constexpr std::uint32_t kVecBatch = 8;
constexpr std::uint32_t kExl3PrefillTileRows = 64;

struct Exl3PrefillLayout {
  std::uint32_t assignments{0}, buckets{0};
  std::size_t bytes{0};
};

bool PlanExl3Prefill(const Config& c, std::uint32_t rows,
                     Exl3PrefillLayout* layout, std::string* error) {
  if (rows == 0 || c.num_experts_used == 0 ||
      c.num_experts_used > c.num_experts || c.hidden_size == 0 ||
      c.expert_ff == 0) {
    AssignError(error, "invalid EXL3 prefill scratch geometry");
    return false;
  }
  const std::uint64_t assignments = std::uint64_t{rows} * c.num_experts_used;
  const auto buckets =
      std::min(assignments, std::uint64_t{c.num_experts} +
                                assignments / kExl3PrefillTileRows);
  if (assignments > std::numeric_limits<std::uint32_t>::max() ||
      buckets > 65535) {
    AssignError(error, "EXL3 prefill scratch exceeds device descriptor limits");
    return false;
  }
  std::size_t bytes = 0;
  const auto add = [&](std::uint64_t count, std::size_t element) {
    if (count > (std::numeric_limits<std::size_t>::max() - bytes) / element)
      return false;
    bytes += static_cast<std::size_t>(count) * element;
    return true;
  };
  if (!add(std::uint64_t{c.num_experts} + 1, sizeof(std::uint32_t)) ||
      !add(assignments, sizeof(std::int32_t)) ||
      !add(buckets, sizeof(Exl3ExpertBucket)) ||
      !add(1, sizeof(std::uint32_t)) ||
      !add(assignments * std::max(c.hidden_size, c.expert_ff),
           sizeof(std::uint16_t)) ||
      !add(assignments * c.hidden_size, sizeof(float)) ||
      !add(assignments, sizeof(std::int32_t)) ||
      !add(assignments, sizeof(std::int32_t))) {
    AssignError(error, "EXL3 prefill scratch byte size overflows size_t");
    return false;
  }
  *layout = {static_cast<std::uint32_t>(assignments),
             static_cast<std::uint32_t>(buckets), bytes};
  return true;
}
// Keep prompt arithmetic independent of chunk width. A scoped, per-thread
// policy also lets the output head retain its existing logits-row arithmetic.
thread_local bool prefill_phase = false;
struct PrefillPhase {
  bool previous;
  explicit PrefillPhase(bool enabled) : previous(prefill_phase) {
    prefill_phase = enabled;
  }
  ~PrefillPhase() { prefill_phase = previous; }
};
bool MatrixRows(std::uint32_t rows) {
  return prefill_phase || rows > kVecBatch;
}
bool ExpertMatrixRows(std::uint32_t rows) {
  return prefill_phase || rows > 4 * kVecBatch;
}
/// Prefill passes wider than a vector batch take the fused projections and
/// the paired routed expert tiles: they win from 9 rows up to the full chunk.
/// Decode and verification batches keep their single-token arithmetic.
bool WidePrefill(std::uint32_t rows) {
  return prefill_phase && rows > kVecBatch;
}

/// Whether mmq's Q8_1 rows of width k carry no padding, so a producer's
/// contiguous element order is the layout.
bool UnpaddedQ8(std::uint32_t k) {
  return k % 32 == 0 &&
         qfn_mmq_q8_1_bytes(1, static_cast<int>(k)) == k / 32 * 36;
}

/// Key-tile splits per row of a narrow attention batch (decode at depth).
constexpr std::uint32_t kAttnSplits = 8;
/// Wide dense Q8_0 projections take the F16 WMMA GEMM (F16 activation
/// rows, weights dequantized as they are staged, no per-block scaling) when
/// its 256-row tiles fill the device and the K sweep is short enough that
/// the F16 rows stay cache-resident: measured per shape at 2048 tokens
/// (W8A8 -> F16 ms): 16384x2560 5.7 -> 4.6, 13312x2560 4.3 -> 3.9,
/// 10240x320 1.02 -> 0.79, 2560x2560 0.81 -> 0.74, 2560x640 0.24 -> 0.21;
/// against 2560x6144 1.8 -> 2.1+, 640x2560 0.19 -> 0.23 and 320x10240
/// 0.40 -> 1.3, which keep the int8 route.
constexpr std::uint32_t kDenseF16MinRows = 2048;
constexpr std::uint32_t kDenseF16MaxCols = 2560;

/// Token rows per routed F16 expert GEMM tile: the narrow tile for batches
/// whose buckets pad to one or two 16-row tiles, the wide one when the
/// mean bucket fills most of it (the weight dequantization is per tile).
constexpr std::uint32_t kRoutedTileRowsNarrow = 16;
constexpr std::uint32_t kRoutedTileRowsWide = 48;

std::uint32_t RoutedTileRows(std::size_t slots, const Config& c) {
  return slots >= static_cast<std::size_t>(16) * c.num_experts
             ? kRoutedTileRowsWide
             : kRoutedTileRowsNarrow;
}

/// Upper bound on launched routed tiles: every 16-padded bucket contributes
/// at most one partial tile beyond its rows.
std::size_t RoutedTileCapacity(std::size_t slots, const Config& c) {
  return (slots + static_cast<std::size_t>(c.num_experts) * 15) /
             kRoutedTileRowsNarrow +
         c.num_experts + 1;
}

std::uint32_t IndexerCapacity(const Config& c, std::uint32_t batch,
                              std::uint32_t context) {
  return static_cast<std::uint32_t>(std::bit_ceil(std::min<std::uint64_t>(
      context, std::uint64_t{c.indexer_top_k} + batch)));
}

/// K, V and pooled indexer keys of every attention layer, then the draft
/// block's, as KV block planes (kernels.hpp kKvPooledRatio). The blocks cover
/// `max_context` tokens and max_context / ratio + 1 pooled rows, the pooled
/// capacity the snapshot and continuation guards address.
/// With KVarN the trunk layers' K/V are one record plane of k4v4 token-major
/// groups (two a block) and only the draft block keeps FP16 K/V planes.
struct SessionKvLayout {
  hip::KvBlockGeometry geometry;
  std::size_t keys{0}, values{0}, pooled{0}, records{0};
  bool fp16_trunk{true};
};
SessionKvLayout SessionKvPlanes(const Config& c, bool mtp, bool kvarn,
                                std::uint32_t max_context, int bits) {
  const std::uint32_t attention = c.num_layers / c.full_attention_interval;
  const std::size_t kv_row = std::size_t{c.AttentionKvDim()} * sizeof(__half);
  SessionKvLayout out;
  out.fp16_trunk = !kvarn;
  std::vector<hip::KvBlockPlane> planes;
  const auto add = [&](hip::KvBlockPlane plane) {
    planes.push_back(plane);
    return planes.size() - 1;
  };
  if (kvarn)
    out.records =
        add(hip::KvarnRecordPlane(attention, c.num_kv_heads, bits,
                                  hip::KvarnValueOrder::kTokenMajor, bits));
  const std::uint32_t fp16_layers = (kvarn ? 0 : attention) + (mtp ? 1 : 0);
  if (fp16_layers != 0) {
    out.keys = add({fp16_layers, kv_row, hip::kKvBlockTokens});
    out.values = add({fp16_layers, kv_row, hip::kKvBlockTokens});
  }
  out.pooled = add({attention + (mtp ? 1 : 0),
                    std::size_t{c.indexer_head_dim} * sizeof(__half),
                    hip::kKvBlockTokens / kKvPooledRatio});
  out.geometry = hip::KvBlockGeometry(
      std::move(planes), hip::KvBlockGeometry::BlocksFor(
                             std::size_t{max_context} + kKvPooledRatio));
  return out;
}

/// Halves of one KVarN session's FP16 sink and tail rows: K and V of every
/// attention layer.
std::size_t KvarnFp16Elements(const Config& c) {
  return std::size_t{2} * (c.num_layers / c.full_attention_interval) *
         hip::kKvarnFp16Tokens * c.AttentionKvDim();
}

void PopulateSnapshotPrefix(std::uint8_t* data, std::uint64_t bytes,
                            std::uint64_t worker_limit = 4) {
  platform::PopulateWritablePages(data, static_cast<std::size_t>(bytes),
                                  static_cast<std::size_t>(worker_limit));
}

}  // namespace

namespace {
constexpr std::uint64_t kSnapshotReuseBytes = 8ULL << 20;
// Device memory kept free beyond the executor's own lazy scratch, for the
// runtime and driver. Snapshot storage never grows into it.
constexpr std::uint64_t kSnapshotDeviceSlackBytes = 512ULL << 20;

/// Snapshot storage would leave too little device memory. Callers keep the
/// checkpoint in its host payload instead; the request continues.
class SnapshotDeviceExhausted : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};
}  // namespace
// Legacy hipFree synchronizes unrelated inference streams. Pool allocations
// have stream-ordered release, so evicting a checkpoint cannot stall a peer.
// Every transfer finishes before dropping its last storage reference.
class SnapshotAllocator
    : public std::enable_shared_from_this<SnapshotAllocator> {
public:
  // Every capture freezes the same mutable state size. Keep this many released
  // blocks of it committed for the next captures: ROCm commits pool pages on
  // each allocation (about 25 GB/s here), and an allocation also stalls other
  // HIP calls, so a background refill would land in a peer's decode instead.
  static constexpr std::size_t kSpareBlocks = 2;

  /// `reserve` is the device memory every commit leaves free.
  explicit SnapshotAllocator(std::uint64_t reserve) : reserve_(reserve) {
    hipMemPoolProps properties{};
    properties.allocType = hipMemAllocationTypePinned;
    properties.handleTypes = hipMemHandleTypeNone;
    properties.location.type = hipMemLocationTypeDevice;
    CheckStatus(hipGetDevice(&properties.location.id), "snapshot pool device");
    CheckStatus(hipMemPoolCreate(&pool_, &properties), "snapshot pool");
    const auto created =
        hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking);
    if (created != hipSuccess) {
      (void)hipMemPoolDestroy(pool_);
      CheckStatus(created, "snapshot allocator stream");
    }
  }
  ~SnapshotAllocator() {
    for (void* spare : spares_)
      (void)hipFree(spare);
    (void)hipStreamSynchronize(stream_);
    (void)hipStreamDestroy(stream_);
    for (hipStream_t stream : free_streams_)
      (void)hipStreamDestroy(stream);
    (void)hipMemPoolDestroy(pool_);
  }
  std::shared_ptr<void> Allocate(std::uint64_t bytes) {
    std::lock_guard lock(mutex_);
    return Own(Commit(bytes), bytes);
  }
  /// Allocates the recurring mutable-state size, from a released block when
  /// one is kept.
  std::shared_ptr<void> AllocateSpare(std::uint64_t bytes) {
    std::lock_guard lock(mutex_);
    if (spare_bytes_ != bytes) {
      for (void* spare : spares_)
        (void)hipFree(spare);
      spares_.clear();
      spare_bytes_ = bytes;
    }
    if (spares_.empty())
      return Own(Commit(bytes), bytes);
    void* data = spares_.back();
    spares_.pop_back();
    return Own(data, bytes);
  }
  /// Bytes the spare blocks may hold, for scratch accounting.
  [[nodiscard]] static std::uint64_t SpareBytes(std::uint64_t block) {
    return kSpareBlocks * block;
  }
  /// Whether committing `bytes` keeps the reserve free. The snapshot budget
  /// counts host payloads, but on an APU the device limit can be lower than
  /// host RAM. ROCm 7.2's VM pool corrupts its heap when a page commit
  /// fails, so storage is never requested beyond the device headroom.
  [[nodiscard]] bool Fits(std::uint64_t bytes) const {
    std::size_t free = 0;
    std::size_t total = 0;
    if (hipMemGetInfo(&free, &total) != hipSuccess)
      return false;
    return free >= bytes && free - bytes >= reserve_;
  }
  /// A transfer on a stream of its own, reused across transfers: creating a
  /// stream can stall for milliseconds, and a shared stream would make a
  /// capture wait for a peer's bulk materialization.
  [[nodiscard]] std::unique_ptr<gufo::hip::SnapshotTransfer> Transfer() {
    hipStream_t stream = nullptr;
    {
      std::lock_guard lock(streams_mutex_);
      if (!free_streams_.empty()) {
        stream = free_streams_.back();
        free_streams_.pop_back();
      }
    }
    if (stream == nullptr)
      CheckStatus(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
                  "snapshot transfer stream");
    return std::make_unique<gufo::hip::SnapshotTransfer>(
        stream, [owner = shared_from_this()](hipStream_t done) {
          std::lock_guard lock(owner->streams_mutex_);
          owner->free_streams_.push_back(done);
        });
  }

private:
  static void CheckStatus(hipError_t error, const char* operation) {
    if (error != hipSuccess)
      throw std::runtime_error(std::string(operation) + ": " +
                               hipGetErrorString(error));
  }
  void* Commit(std::uint64_t bytes) {
    if (!Fits(bytes))
      throw SnapshotDeviceExhausted("snapshot storage exceeds device headroom");
    void* data = nullptr;
    CheckStatus(hipMallocFromPoolAsync(&data, bytes, pool_, stream_),
                "snapshot allocate");
    const auto ready = hipStreamSynchronize(stream_);
    if (ready != hipSuccess) {
      (void)hipFree(data);
      (void)hipStreamSynchronize(stream_);
      CheckStatus(ready, "snapshot allocation ready");
    }
    return data;
  }
  std::shared_ptr<void> Own(void* data, std::uint64_t bytes) {
    return {data, [owner = shared_from_this(), bytes](void* pointer) {
              owner->Release(pointer, bytes);
            }};
  }
  void Release(void* pointer, std::uint64_t bytes) {
    std::lock_guard lock(mutex_);
    if (bytes == spare_bytes_ && spares_.size() < kSpareBlocks) {
      spares_.push_back(pointer);
      return;
    }
    // In ROCm, hipFree recognizes pool allocations and bypasses SyncAllStreams.
    // Unlike hipFreeAsync, it also permits a peer blocking-stream capture.
    (void)hipFree(pointer);
    // Keep physical memory bounded by retained checkpoints and the spares,
    // including when their source session has already been destroyed.
    (void)hipStreamSynchronize(stream_);
  }
  std::uint64_t reserve_;
  std::mutex mutex_;
  std::vector<void*> spares_;
  std::uint64_t spare_bytes_{0};
  hipStream_t stream_{nullptr};
  std::mutex streams_mutex_;
  std::vector<hipStream_t> free_streams_;
  hipMemPool_t pool_{nullptr};
};

std::shared_ptr<SnapshotAllocator> Executor::GetSnapshotAllocator() const {
  std::call_once(snapshot_allocator_once_, [&] {
    // Admission reserved the lazy scratch; snapshots must not consume it.
    snapshot_allocator_ = std::make_shared<SnapshotAllocator>(
        DeferredScratchBytes() + kSnapshotDeviceSlackBytes);
  });
  return snapshot_allocator_;
}

// Provenance describes rows actually copied by a restore, rather than token
// equality: different prefill shapes can produce different KV for the same
// tokens. A branch shares its parent's rows only below its restored frontier.
// These nodes retain metadata alone, never snapshot payloads or device storage.
struct SnapshotLineage {
  using Rows = std::array<std::uint32_t, 4>;
  std::shared_ptr<const SnapshotLineage> parent;
  Rows prefix{};
  std::size_t depth = 0;
  explicit SnapshotLineage(
      std::shared_ptr<const SnapshotLineage> source = nullptr, Rows rows = {})
      : parent(std::move(source)),
        prefix(rows),
        depth(parent ? parent->depth + 1 : 0) {}

  static Rows Common(std::shared_ptr<const SnapshotLineage> left,
                     std::shared_ptr<const SnapshotLineage> right,
                     Rows bounds) {
    const auto ascend = [&](auto& node) {
      for (std::size_t i = 0; i < bounds.size(); ++i)
        bounds[i] = std::min(bounds[i], node->prefix[i]);
      node = node->parent;
    };
    if (!left || !right)
      return {};
    while (left->depth > right->depth)
      ascend(left);
    while (right->depth > left->depth)
      ascend(right);
    while (left != right) {
      if (!left->parent || !right->parent)
        return {};
      ascend(left);
      ascend(right);
    }
    return bounds;
  }
};

struct SnapshotBlock {
  SnapshotBlock(const std::shared_ptr<SnapshotAllocator>& allocator,
                std::uint64_t bytes)
      : storage(allocator->Allocate(bytes)), capacity(bytes) {}
  std::shared_ptr<void> storage;
  std::uint64_t capacity;
  std::atomic<std::uint32_t> chunks{0};
  std::uint64_t used = 0;
};
struct SnapshotChunk {
  std::shared_ptr<SnapshotBlock> block;
  std::uint8_t* device;
  ~SnapshotChunk() { --block->chunks; }
};
struct SnapshotRegistry {
  explicit SnapshotRegistry(std::shared_ptr<SnapshotAllocator> allocator)
      : allocator(std::move(allocator)) {}
  std::shared_ptr<SnapshotAllocator> allocator;
  std::mutex mutex;
  std::vector<std::weak_ptr<SnapshotState>> pending;
  std::vector<std::weak_ptr<SnapshotBlock>> reserve;
  std::exception_ptr failure;
  void Preserve(std::uint32_t position, std::uint32_t mtp_position,
                std::uint32_t blocks, std::uint32_t mtp_blocks);
};

void SnapshotState::Preserve(std::uint32_t position, std::uint32_t mtp_position,
                             std::uint32_t blocks, std::uint32_t mtp_blocks,
                             bool detach) const {
  std::unique_ptr<gufo::hip::SnapshotTransfer> transfer;
  for (auto& region : regions_) {
    auto rows = region.draft ? mtp_position : position;
    if (region.pooled)
      rows = region.draft ? mtp_blocks : blocks;
    const auto keep = std::uint64_t{rows} * region.row_bytes;
    if (keep < region.pending_bytes) {
      if (!transfer)
        transfer = registry_->allocator->Transfer();
      transfer->Copy(region.host + keep,
                     static_cast<const std::uint8_t*>(region.device) + keep,
                     region.pending_bytes - keep);
      region.pending_bytes = keep;
    }
    if (detach)
      region.live_bytes = std::min(region.live_bytes, keep);
  }
}

void SnapshotState::Materialize() const {
  std::lock_guard lock(registry_->mutex);
  if (registry_->failure)
    std::rethrow_exception(registry_->failure);
  MaterializeLocked();
}

void SnapshotState::MaterializeLocked() const {
  const auto transfer = registry_->allocator->Transfer();
  for (const auto& region : regions_) {
    PopulateSnapshotPrefix(region.host, region.bytes);
    for (const auto& saved : region.saved)
      transfer->Enqueue(region.host + saved.offset, saved.chunk->device,
                        saved.bytes);
  }
  transfer->Finish();
  Preserve(0, 0, 0, 0, false);
  if (mutable_storage_) {
    PopulateSnapshotPrefix(payload_ + mutable_offset_, mutable_bytes_);
    transfer->Copy(payload_ + mutable_offset_, mutable_device_, mutable_bytes_);
    mutable_storage_.reset();
  }
  if (extra_storage_) {
    for (const auto& extra : extras_)
      transfer->Enqueue(payload_ + extra.offset, extra.device, extra.bytes);
    transfer->Finish();
    extras_.clear();
    extra_storage_.reset();
  }
  for (auto& region : regions_)
    region.saved.clear();
}

std::uint64_t SnapshotState::DeviceBytes() const {
  std::lock_guard lock(registry_->mutex);
  std::uint64_t bytes = (mutable_storage_ ? mutable_bytes_ : 0) +
                        (extra_storage_ ? extra_bytes_ : 0);
  std::vector<const SnapshotBlock*> blocks;
  for (const auto& region : regions_)
    for (const auto& saved : region.saved) {
      const auto* block = saved.chunk->block.get();
      if (std::find(blocks.begin(), blocks.end(), block) == blocks.end()) {
        blocks.push_back(block);
        bytes += block->capacity;
      }
    }
  return bytes;
}

void SnapshotRegistry::Preserve(std::uint32_t position,
                                std::uint32_t mtp_position,
                                std::uint32_t blocks,
                                std::uint32_t mtp_blocks) {
  struct Group {
    void* device;
    std::uint64_t keep;
    std::uint64_t end;
    std::vector<std::pair<SnapshotState::Region*, std::size_t>> regions;
  };
  std::vector<std::shared_ptr<SnapshotState>> snapshots;
  std::vector<Group> groups;
  std::vector<std::shared_ptr<SnapshotBlock>> available;
  // Only overwritten rows need backing storage. Appending borrows K/V in
  // place and does not allocate another context's worth of device pages.
  for (auto it = reserve.begin(); it != reserve.end();) {
    if (auto block = it->lock()) {
      if (block->chunks == 0)
        block->used = 0;
      if (block->used < block->capacity &&
          block->capacity - block->used <= kSnapshotReuseBytes)
        available.push_back(std::move(block));
      ++it;
    } else {
      it = reserve.erase(it);
    }
  }
  for (auto it = pending.begin(); it != pending.end();) {
    if (auto snapshot = it->lock()) {
      for (auto& region : snapshot->regions_) {
        auto rows = region.draft ? mtp_position : position;
        if (region.pooled)
          rows = region.draft ? mtp_blocks : blocks;
        const auto keep = std::uint64_t{rows} * region.row_bytes;
        if (keep < region.pending_bytes) {
          auto group = std::find_if(
              groups.begin(), groups.end(),
              [&](const auto& entry) { return entry.device == region.device; });
          if (group == groups.end()) {
            groups.push_back({region.device,
                              keep,
                              region.pending_bytes,
                              {{&region, snapshots.size()}}});
          } else {
            group->end = std::max(group->end, region.pending_bytes);
            group->regions.emplace_back(&region, snapshots.size());
          }
        }
      }
      snapshots.push_back(std::move(snapshot));
      ++it;
    } else {
      it = pending.erase(it);
    }
  }
  struct Copy {
    void* device;
    std::uint64_t begin;
    std::uint64_t end;
    std::vector<SnapshotState::Region*> regions;
    std::vector<std::size_t> owners;
  };
  std::vector<Copy> copies;
  for (const auto& group : groups) {
    std::vector<std::uint64_t> ends{group.keep};
    for (const auto& [region, owner] : group.regions) {
      (void)owner;
      ends.push_back(region->pending_bytes);
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    for (std::size_t index = 1; index < ends.size(); ++index) {
      Copy copy{group.device, ends[index - 1], ends[index], {}, {}};
      for (const auto& [region, owner] : group.regions)
        if (region->pending_bytes >= copy.end) {
          copy.regions.push_back(region);
          copy.owners.push_back(owner);
        }
      copies.push_back(std::move(copy));
    }
  }
  // Pack rows with the same consumers together across layers. Each consumer
  // set gets one allocation, so protecting a deep prefix avoids dozens of
  // small allocations. A short checkpoint retains only its own rows and at
  // most bounded slack from a reused block belonging to other consumers.
  std::stable_sort(copies.begin(), copies.end(),
                   [](const auto& left, const auto& right) {
                     return left.owners < right.owners;
                   });
  std::uint64_t capacity = 0;
  for (const auto& block : available)
    capacity += block->capacity - block->used;
  // Allocate before enqueuing any copy or changing a checkpoint. Allocation
  // failure then leaves the source and every retained checkpoint intact.
  std::vector<std::uint64_t> commits;
  std::uint64_t needed = 0;
  for (std::size_t first = 0; first < copies.size();) {
    auto last = first;
    std::uint64_t bytes = 0;
    while (last < copies.size() &&
           copies[last].owners == copies[first].owners) {
      bytes += copies[last].end - copies[last].begin;
      ++last;
    }
    const auto reused = std::min(capacity, bytes);
    capacity -= reused;
    bytes -= reused;
    if (bytes != 0) {
      commits.push_back(bytes);
      needed += bytes;
    }
    first = last;
  }
  bool device = needed == 0 || allocator->Fits(needed);
  if (device) {
    try {
      for (const auto bytes : commits) {
        auto block = std::make_shared<SnapshotBlock>(allocator, bytes);
        reserve.push_back(block);
        available.push_back(std::move(block));
      }
    } catch (const SnapshotDeviceExhausted&) {
      device = false;
    }
  }
  if (!device) {
    // No device room for the overwritten rows: move each affected checkpoint
    // into its host payload, which the snapshot budget already charges. This
    // also releases the device copies it held from earlier rewinds.
    std::vector<bool> affected(snapshots.size());
    for (const auto& copy : copies)
      for (const auto owner : copy.owners)
        affected[owner] = true;
    for (std::size_t index = 0; index < snapshots.size(); ++index)
      if (affected[index])
        snapshots[index]->MaterializeLocked();
    copies.clear();
    available.clear();
  }
  std::size_t block_index = 0;
  std::unique_ptr<gufo::hip::SnapshotTransfer> transfer;
  for (auto& copy : copies) {
    for (auto begin = copy.begin; begin < copy.end;) {
      if (block_index < available.size() &&
          available[block_index]->used == available[block_index]->capacity)
        ++block_index;
      if (block_index >= available.size())
        throw std::logic_error(
            "snapshot protection exceeded allocated storage");
      auto& used = available[block_index]->used;
      const auto bytes =
          std::min(available[block_index]->capacity - used, copy.end - begin);
      auto chunk = std::make_shared<SnapshotChunk>();
      chunk->block = available[block_index];
      ++chunk->block->chunks;
      chunk->device =
          static_cast<std::uint8_t*>(chunk->block->storage.get()) + used;
      if (!transfer)
        transfer = allocator->Transfer();
      transfer->Enqueue(chunk->device,
                        static_cast<const std::uint8_t*>(copy.device) + begin,
                        bytes, hipMemcpyDeviceToDevice);
      for (auto* region : copy.regions)
        region->saved.push_back({begin, bytes, chunk});
      used += bytes;
      begin += bytes;
    }
  }
  if (transfer)
    transfer->Finish();
  for (const auto& snapshot : snapshots) {
    for (auto& region : snapshot->regions_) {
      auto rows = region.draft ? mtp_position : position;
      if (region.pooled)
        rows = region.draft ? mtp_blocks : blocks;
      const auto keep = std::uint64_t{rows} * region.row_bytes;
      region.pending_bytes = std::min(region.pending_bytes, keep);
      region.live_bytes = std::min(region.live_bytes, keep);
    }
  }
}

void Session::PreserveSnapshots(std::uint32_t position,
                                std::uint32_t mtp_position) const {
  if (!snapshots_ || (position >= snapshot_position_ &&
                      mtp_position >= snapshot_mtp_position_))
    return;
  std::lock_guard lock(snapshots_->mutex);
  if (snapshots_->failure)
    std::rethrow_exception(snapshots_->failure);
  const auto ratio = owner_->config().compress_ratio;
  snapshots_->Preserve(position, mtp_position,
                       ratio ? std::min(blocks_, position / ratio) : 0,
                       ratio ? std::min(mtp_.blocks, mtp_position / ratio) : 0);
  snapshot_position_ = std::min(snapshot_position_, position);
  snapshot_mtp_position_ = std::min(snapshot_mtp_position_, mtp_position);
}

Session::~Session() {
  if (kv_pool_) {
    (void)hipStreamSynchronize(owner_->stream());
    ReleaseMappedKvBlocks();
    kv_reservation_.Reset();
  }
  try {
    PreserveSnapshots(0, 0);
  } catch (...) {
    // The backing device may be lost. Leave retained checkpoints readable as
    // failures instead of allowing a later reader to touch freed allocations.
    std::lock_guard lock(snapshots_->mutex);
    snapshots_->failure = std::current_exception();
  }
  TrimRollback(0);
  for (auto& [key, exec] : graphs_) {
    (void)hipGraphExecDestroy(exec);
  }
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

void Session::RestoreVisionLayout(const qwen::vision::RopeLayout& layout,
                                  hipStream_t stream) {
  layout.Validate(max_context_);
  const auto* previous = vision_input_.rope();
  vision_input_.RestoreLayout(layout, stream);
  for (auto& attention : attention_)
    attention.rope = vision_input_.rope();
  if (previous != vision_input_.rope()) {
    for (const auto& [key, graph] : graphs_)
      (void)hipGraphExecDestroy(graph);
    graphs_.clear();
    warmed_.clear();
  }
}

void Session::PrepareVision(hipStream_t stream) {
  vision_input_.PrepareRope(stream);
  for (auto& attention : attention_)
    attention.rope = vision_input_.rope();
}

void Session::ConfigureVision(
    std::shared_ptr<const qwen::vision::Prompt> prompt,
    std::shared_ptr<qwen::vision::Encoder> encoder, hipStream_t stream) {
  if (prompt)
    prompt->rope.Validate(max_context_);
  const auto* previous = vision_input_.rope();
  vision_input_.Configure(std::move(prompt), std::move(encoder), stream);
  for (auto& attention : attention_)
    attention.rope = vision_input_.rope();
  if (previous != vision_input_.rope()) {
    for (const auto& [key, graph] : graphs_)
      (void)hipGraphExecDestroy(graph);
    graphs_.clear();
    warmed_.clear();
  }
}

void Session::SetCancellationCheck(std::function<bool()> check) {
  cancelled_ = false;
  is_cancelled_ = std::move(check);
  vision_input_.SetCancellationCheck(is_cancelled_);
}

bool Session::CheckCancellation(std::string* error) const {
  try {
    cancelled_ = cancelled_ || (is_cancelled_ && is_cancelled_());
  } catch (...) {
    cancelled_ = true;
  }
  if (cancelled_) {
    AssignError(error, "generation cancelled");
    return false;
  }
  return true;
}

std::size_t Session::AllocatedBytes() const noexcept {
  return allocated_bytes_ + rollback_bytes_ + vision_input_.Bytes();
}

PrefillCheckpoint::~PrefillCheckpoint() {
  if (!state)
    return;
  source->snapshot_position_ =
      std::max(source->snapshot_position_, state->snapshot_position_);
  source->snapshot_mtp_position_ =
      std::max(source->snapshot_mtp_position_, state->snapshot_mtp_position_);
  // The view owns no cache allocations. Its borrowed rows belong to source.
  state->snapshots_.reset();
}

std::unique_ptr<PrefillCheckpoint> Executor::MakePrefillCheckpoint(
    Session& source, std::span<const std::int32_t> tokens, std::uint32_t prefix,
    std::string* error_msg) const {
  if (source.kvarn_ || source.kv_pool_) {
    AssignError(error_msg,
                "KVarN or shared KV sessions take no in-pass checkpoints");
    return nullptr;
  }
  if (source.owner_ != this || prefix == 0 || prefix > tokens.size() ||
      tokens.size() - prefix > kVecBatch ||
      !source.VisionLayout().images.empty()) {
    AssignError(error_msg, "prefill checkpoint must be in the text tail");
    return nullptr;
  }
  const auto& c = config();
  auto checkpoint = std::make_unique<PrefillCheckpoint>();
  checkpoint->source = &source;
  checkpoint->tokens = prefix;
  checkpoint->hidden_rows =
      source.mtp_enabled_ ? std::min(prefix, options_.max_speculative) : 0;
  checkpoint->logits.resize(c.vocab_size);
  checkpoint->state.reset(new Session);
  auto& view = *checkpoint->state;
  view.owner_ = this;
  view.max_context_ = source.max_context_;
  view.index_capacity_ = source.index_capacity_;
  view.position_ = source.position_ + prefix;
  view.mtp_enabled_ = source.mtp_enabled_;
  view.attention_ = source.attention_;
  view.linear_.resize(source.linear_.size());
  view.blocks_ = c.compress_ratio > 0 && view.position_ > c.indexer_top_k
                     ? view.position_ / c.compress_ratio
                     : source.blocks_;
  view.mtp_ = source.mtp_;
  if (source.mtp_enabled_) {
    view.mtp_.position = view.position_ - 1;
    view.mtp_.blocks = view.mtp_.position > c.indexer_top_k
                           ? view.mtp_.position / c.compress_ratio
                           : source.mtp_.blocks;
    if (prefix > 1)
      view.mtp_.residual_valid = false;
  }
  view.ngram_ = source.ngram_;
  std::vector<std::uint32_t> rows(std::size_t{prefix} * c.ple_heads);
  if (c.ple_layer >= 0)
    HashNgramRows(c, view.ngram_, tokens.first(prefix), rows);
  if (!source.snapshots_)
    source.snapshots_ =
        std::make_shared<SnapshotRegistry>(GetSnapshotAllocator());
  if (!source.snapshot_lineage_)
    source.snapshot_lineage_ = std::make_shared<SnapshotLineage>();
  view.snapshots_ = source.snapshots_;
  view.snapshot_lineage_ = source.snapshot_lineage_;
  const auto conv = std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const auto recurrent =
      std::size_t{c.ssm_num_v_heads} * c.ssm_head_dim * c.ssm_head_dim;
  std::size_t floats =
      has_mtp() ? std::size_t{options_.max_speculative} * c.HcDim() : 0;
  for (const auto& linear : source.linear_)
    if (linear.state)
      floats += conv + recurrent;
  if (c.ple_layer >= 0)
    floats += std::size_t{c.PleConvHistory()} * c.HcDim();
  if (!checkpoint_storage_) {
    // Forward scratch is shared by the executor. The checkpoint is frozen
    // into private snapshot storage before another session can reuse it.
    checkpoint_storage_ =
        source.snapshots_->allocator->Allocate(floats * sizeof(float));
  }
  auto* storage = static_cast<float*>(checkpoint_storage_.get());
  for (std::size_t il = 0; il < source.linear_.size(); ++il) {
    if (!source.linear_[il].state)
      continue;
    view.linear_[il].conv_state = storage;
    storage += conv;
    view.linear_[il].state = storage;
    storage += recurrent;
  }
  if (c.ple_layer >= 0) {
    view.ple_history_ = storage;
    storage += std::size_t{c.PleConvHistory()} * c.HcDim();
  }
  if (source.mtp_enabled_)
    view.mtp_.target_hidden = storage;
  return checkpoint;
}

void Session::TrimRollback(std::uint32_t depth) noexcept {
  if (depth >= rollback_depth_)
    return;
  // Keep graphs whose rows still exist. Deeper verifier graphs capture
  // discarded pointers; ordinary decode and MTP graphs do not.
  for (auto it = graphs_.begin(); it != graphs_.end();) {
    if ((it->first & (std::uint64_t{1} << 32)) != 0 &&
        (it->first & (std::uint64_t{1} << 40)) == 0 &&
        (it->first & 0xFFFFU) > depth + 1) {
      (void)hipGraphExecDestroy(it->second);
      it = graphs_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto row = depth; row < rollback_depth_; ++row) {
    (void)hipFree(rollback_allocations_[row]);
    for (auto& layer : linear_) {
      layer.conv_snapshots.rows[row] = nullptr;
      layer.state_snapshots.rows[row] = nullptr;
    }
    ple_snapshots_.rows[row] = nullptr;
  }
  rollback_allocations_.resize(depth);
  rollback_depth_ = depth;
  rollback_bytes_ =
      owner_->SessionBytes(core::SessionMode::kAutoregressive, max_context_,
                           depth) -
      owner_->SessionBytes(core::SessionMode::kAutoregressive, max_context_, 0);
  ngram_snapshots_.resize(depth);
}

void Session::EnsureKvRows(std::uint32_t rows) {
  if (!kv_pool_)
    return;
  if (kv_unmapped_blocks_)
    throw std::logic_error("Flash-Next KV restore has unmapped blocks");
  if (rows > max_context_)
    throw std::length_error("Flash-Next KV rows exceed context");
  const auto need =
      rows ? hip::KvBlockGeometry::BlocksFor(std::size_t{rows} + kKvPooledRatio)
           : 0;
  if (need <= kv_blocks_.size())
    return;
  auto fresh = kv_pool_->Allocate(need - kv_blocks_.size(), &kv_reservation_);
  if (fresh.size() != need - kv_blocks_.size())
    throw std::length_error("Flash-Next shared KV pool exhausted");
  const auto first = kv_blocks_.size();
  kv_blocks_.insert(kv_blocks_.end(), fresh.begin(), fresh.end());
  std::copy(fresh.begin(), fresh.end(), kv_table_host_ + first);
  const auto result = hipMemcpyAsync(kv_table_ + first, kv_table_host_ + first,
                                     fresh.size() * sizeof(std::uint32_t),
                                     hipMemcpyHostToDevice, owner_->stream());
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}

void Session::TrimKvRows(std::uint32_t rows) {
  if (!kv_pool_)
    return;
  const auto keep =
      rows ? hip::KvBlockGeometry::BlocksFor(std::size_t{rows} + kKvPooledRatio)
           : 0;
  if (keep >= kv_blocks_.size())
    return;
  const auto result = hipStreamSynchronize(owner_->stream());
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
  ReleaseMappedKvBlocks(keep);
  kv_blocks_.resize(keep);
}

void Session::ReleaseMappedKvBlocks(std::size_t first) noexcept {
  const auto begin = kv_blocks_.begin() + static_cast<std::ptrdiff_t>(first);
  const auto end = std::remove(begin, kv_blocks_.end(), kUnmappedKvBlock);
  kv_unmapped_blocks_ -= std::distance(end, kv_blocks_.end());
  kv_pool_->Release(std::span(begin, end));
}

void Session::PrepareKvRestoreRows(std::uint32_t rows) {
  if (!kv_pool_)
    return;
  if (rows > max_context_)
    throw std::length_error("Flash-Next KV restore exceeds context");
  const auto need =
      rows ? hip::KvBlockGeometry::BlocksFor(std::size_t{rows} + kKvPooledRatio)
           : 0;
  kv_blocks_.reserve(need);
  TrimKvRows(rows);
  const auto previous = kv_blocks_.size();
  kv_blocks_.resize(need, kUnmappedKvBlock);
  kv_unmapped_blocks_ += need - previous;
}

void Session::EnsureKvRestoreBlock(std::uint32_t logical) {
  auto& physical = kv_blocks_.at(logical);
  if (physical != kUnmappedKvBlock)
    return;
  physical = kv_pool_->Allocate(1, &kv_reservation_).front();
  kv_table_host_[logical] = physical;
  --kv_unmapped_blocks_;
}

void Session::MapKvBlock(std::uint32_t logical, std::uint32_t physical) {
  auto& entry = kv_blocks_.at(logical);
  if (entry == physical)
    return;
  kv_pool_->Retain(std::span(&physical, 1));
  const auto replaced = std::exchange(entry, physical);
  kv_table_host_[logical] = physical;
  if (replaced == kUnmappedKvBlock)
    --kv_unmapped_blocks_;
  else
    kv_pool_->Release(std::span(&replaced, 1), kv_reservation_);
}

void Session::UploadKvTable() {
  if (!kv_pool_ || kv_blocks_.empty())
    return;
  if (kv_unmapped_blocks_)
    throw std::logic_error("Flash-Next KV restore has unmapped blocks");
  const auto result = hipMemcpyAsync(kv_table_, kv_table_host_,
                                     kv_blocks_.size() * sizeof(std::uint32_t),
                                     hipMemcpyHostToDevice, owner_->stream());
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}

void Session::UnshareKvRows(std::uint32_t first, std::uint32_t end) {
  if (!kv_pool_ || first >= end)
    return;
  const auto check = [](hipError_t result) {
    if (result != hipSuccess)
      throw std::runtime_error(hipGetErrorString(result));
  };
  const auto& geometry = kv_pool_->geometry();
  const auto last = std::min<std::size_t>(hip::KvBlockGeometry::BlocksFor(end),
                                          kv_blocks_.size());
  for (std::size_t logical = first >> hip::kKvBlockShift; logical < last;
       ++logical) {
    const auto shared = kv_blocks_[logical];
    if (shared == kUnmappedKvBlock)
      throw std::logic_error("cannot write unmapped Flash-Next KV rows");
    if (kv_pool_->references(shared) < 2)
      continue;
    // Drain graph/table users before changing the pinned table. Copy every
    // plane: target, predictor and pooled indexer share one physical ID.
    check(hipStreamSynchronize(owner_->stream()));
    const auto copy = kv_pool_->Allocate(1, &kv_reservation_).front();
    try {
      for (std::size_t plane = 0; plane < geometry.planes().size(); ++plane) {
        const auto pitch = geometry.layer_pitch(plane);
        const auto width = geometry.block_bytes(plane);
        auto* base = kv_pool_->plane_data(plane);
        check(hipMemcpy2DAsync(base + copy * width, pitch,
                               base + shared * width, pitch, width,
                               geometry.planes()[plane].layers,
                               hipMemcpyDeviceToDevice, owner_->stream()));
      }
      check(hipStreamSynchronize(owner_->stream()));
    } catch (...) {
      if (hipStreamSynchronize(owner_->stream()) == hipSuccess)
        kv_pool_->Release(std::span(&copy, 1), kv_reservation_);
      throw;
    }
    kv_blocks_[logical] = copy;
    kv_table_host_[logical] = copy;
    try {
      check(hipMemcpyAsync(kv_table_ + logical, kv_table_host_ + logical,
                           sizeof(std::uint32_t), hipMemcpyHostToDevice,
                           owner_->stream()));
      check(hipStreamSynchronize(owner_->stream()));
    } catch (...) {
      kv_pool_->Release(std::span(&shared, 1));
      throw;
    }
    kv_pool_->Release(std::span(&shared, 1));
  }
}

std::uint32_t Session::ReserveKvRows(std::uint32_t rows, std::uint32_t margin) {
  if (!kv_pool_)
    return 0;
  if (rows > max_context_)
    throw std::length_error("Flash-Next KV reservation exceeds context");
  const auto target = std::min(hip::KvBlockGeometry::BlocksFor(rows) + margin,
                               hip::KvBlockGeometry::BlocksFor(
                                   std::size_t{max_context_} + kKvPooledRatio));
  if (kv_unmapped_blocks_)
    throw std::logic_error("cannot reserve an incomplete Flash-Next restore");
  auto first = mtp_enabled_ ? std::min(position_, mtp_.position) : position_;
  if (kvarn_)
    first = std::min(first, kvarn_tail_.flushed_end);
  if (rows > owner_->config().indexer_top_k) {
    first = std::min(first, blocks_ * kKvPooledRatio);
    if (mtp_enabled_)
      first = std::min(first, mtp_.blocks * kKvPooledRatio);
  }
  std::uint32_t copies = 0;
  for (std::size_t i = first >> hip::kKvBlockShift;
       i < std::min<std::size_t>(target, kv_blocks_.size()); ++i)
    copies += kv_pool_->references(kv_blocks_[i]) > 1;
  const auto need =
      (target > kv_blocks_.size() ? target - kv_blocks_.size() : 0) + copies;
  kv_reservation_.Reset();
  if (need) {
    kv_reservation_ = kv_pool_->TryReserve(need);
    if (!kv_reservation_)
      throw std::length_error(
          "Flash-Next shared KV pool reservation exhausted");
  }
  return need;
}
std::uint32_t Session::ReleaseKvReservation() noexcept {
  const auto unused = kv_reservation_.blocks();
  kv_reservation_.Reset();
  return unused;
}

void Session::Reset(bool reuse_mtp) {
  if (kv_pool_) {
    const auto result = hipStreamSynchronize(owner_->stream());
    if (result != hipSuccess)
      throw std::runtime_error(hipGetErrorString(result));
    ReleaseMappedKvBlocks();
    kv_blocks_.clear();
    kv_reservation_.Reset();
  }

  // Reset leaves append-only rows intact. Preserve them at the first write
  // that would overwrite them, allowing cancellation to resume by rewind.
  // Scratch is overwritten by the next verification. Retaining its stable
  // addresses also retains the graphs; freeing it at every cache reassignment
  // makes later MTP requests allocate and record those shapes again. Its full
  // bound is already included in speculative SessionBytes and AllocatedBytes.
  if (!reuse_mtp || !mtp_enabled_)
    TrimRollback(0);
  snapshot_lineage_.reset();
  position_ = 0;
  spec_tokens_ = 0;
  kvarn_tail_ = {};
  kvarn_pass_ = {};
  ngram_.Reset();
  mtp_.position = 0;
  mtp_.residual_valid = false;
  mtp_.blocks = 0;
  const Config& c = owner_->config();
  const auto zero = [&](void* pointer, std::size_t bytes) {
    const auto result = hipMemsetAsync(pointer, 0, bytes, owner_->stream());
    // Adapter invalidation may report success only after every zero fill was
    // submitted and completed. Keep legacy cleanup's error policy unchanged.
    if (result != hipSuccess && continuation_hooks_)
      throw std::runtime_error(hipGetErrorString(result));
  };
  for (auto& l : linear_) {
    if (l.state != nullptr) {
      zero(l.conv_state, static_cast<std::size_t>(c.ssm_conv_kernel - 1) *
                             c.SsmConvChannels() * sizeof(float));
      zero(l.state, static_cast<std::size_t>(c.ssm_num_v_heads) *
                        c.ssm_head_dim * c.ssm_head_dim * sizeof(float));
    }
  }
  blocks_ = 0;
  if (ple_history_ != nullptr) {
    zero(ple_history_, static_cast<std::size_t>(c.PleConvHistory()) *
                           c.HcDim() * sizeof(float));
  }
}

Executor::~Executor() {
  // Readers write pinned staging memory; drain them before freeing it,
  // including when a forward failed before reaching PLE.
  if (ple_pending_ || prefetch_.pending) {
    (void)ngram_->WaitRead();
  }
  (void)hipFree(batch_logits_);
  (void)hipFree(verification_penalties_);
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
  host_allocations_.Reset();
  if (blas_ != nullptr) {
    (void)hipblasDestroy(blas_);
  }
  if (counts_ready_ != nullptr) {
    (void)hipEventDestroy(counts_ready_);
  }
  for (hipEvent_t event : {side_start_, side_done_}) {
    if (event != nullptr)
      (void)hipEventDestroy(event);
  }
  if (side_stream_ != nullptr)
    (void)hipStreamDestroy(side_stream_);
  if (stream_ != nullptr) {
    qfn_mmq_release_stream(stream_);
    (void)hipStreamDestroy(stream_);
  }
}

std::unique_ptr<Executor> Executor::Create(const DeviceModel& model,
                                           NgramTable* ngram, Options options,
                                           std::string* error_msg) {
  if (options.kvarn_bits != 4 && options.kvarn_bits != 6) {
    if (error_msg)
      *error_msg = "KVarN bits must be 4 or 6";
    return nullptr;
  }

  std::unique_ptr<Executor> e(new Executor());
  e->model_ = &model;
  e->ngram_ = ngram;
  e->options_ = options;
  e->options_.max_batch = std::max<std::uint32_t>(1, options.max_batch);
  e->options_.max_logit_rows = std::clamp<std::uint32_t>(
      options.max_logit_rows, 1, e->options_.max_batch);
  e->options_.max_speculative = std::clamp<std::uint32_t>(
      options.max_speculative, 1, e->options_.max_logit_rows);
  if (qfn_mmq_init(0) != 0) {
    AssignError(error_msg, "quantized GEMM tier initialization failed");
    return nullptr;
  }
  if (!Check(PrepareW8A8GemmWave64(), "wide Q8 GEMM preparation", error_msg))
    return nullptr;
  // The first greedy verification otherwise resolves the shared argmax
  // module on the decode path. Preparation launches no trial kernels.
  if (e->has_mtp() && !Check(gufo::hip::PrepareBatchedGPUArgmax(),
                             "MTP argmax preparation", error_msg))
    return nullptr;
  if (!Check(hipStreamCreate(&e->stream_), "hipStreamCreate", error_msg)) {
    return nullptr;
  }
  if (!Check(hipEventCreateWithFlags(&e->counts_ready_, hipEventDisableTiming),
             "expert counts event", error_msg) ||
      !Check(hipStreamCreate(&e->side_stream_), "side stream", error_msg) ||
      !Check(hipEventCreateWithFlags(&e->side_start_, hipEventDisableTiming),
             "side stream fork event", error_msg) ||
      !Check(hipEventCreateWithFlags(&e->side_done_, hipEventDisableTiming),
             "side stream join event", error_msg)) {
    return nullptr;
  }
  if (hipblasCreate(&e->blas_) != HIPBLAS_STATUS_SUCCESS ||
      hipblasSetStream(e->blas_, e->stream_) != HIPBLAS_STATUS_SUCCESS) {
    AssignError(error_msg, "hipblasCreate failed");
    return nullptr;
  }
  const Config& c = model.config();
  if (options.max_context > c.MaxContextLength()) {
    AssignError(error_msg, "executor context exceeds the model context");
    return nullptr;
  }
  e->context_capacity_ = std::max(c.context_length, options.max_context);
  e->yarn_ = MakeYarnRope(c.rope_scaling, c.rotary_dim, c.rope_theta);
  const std::size_t T = e->options_.max_batch;
  e->blaslt_ = BlasLt::Create(e->stream_, error_msg);
  if (e->blaslt_ == nullptr) {
    return nullptr;
  }
  if (MatrixRows(T)) {
    // The first indexer heuristic query also loads the library's code object.
    // Do that before readiness instead of leaving a ~100 ms GPU idle gap in
    // the first prefill. Exact geometry still determines every plan; no trial
    // GEMM runs and no model activations or session state are touched.
    for (const auto& layer : model.layers()) {
      for (const auto* tensor : {&layer.indexer_k, &layer.indexer_q}) {
        if (!tensor->empty() &&
            (tensor->type == GgmlType::kBF16 ||
             tensor->type == GgmlType::kF16) &&
            !e->blaslt_->Prepare(
                tensor->type == GgmlType::kBF16 ? HIP_R_16BF : HIP_R_16F,
                tensor->rows, T, tensor->cols, error_msg)) {
          return nullptr;
        }
      }
    }
  }
  const std::size_t hc_dim = c.HcDim();
  const std::size_t hidden = c.hidden_size;
  const std::size_t slots = T * c.num_experts_used;
  auto& a = e->allocations_;
  Scratch& s = e->s_;
  auto f32 = [&](std::size_t n) { return Alloc<float>(a, n, error_msg); };
  if (model.output().exl3 != nullptr) {
    auto& ex = e->exl3_;
    auto& projection = ex.projection;
    const auto width = std::max(
        {c.hidden_size, c.SsmValueDim(), c.AttentionQDim(), c.expert_ff});
    projection.rotated_elements = std::max(T, std::size_t{64}) * width;
    projection.rotated = f32(projection.rotated_elements);
    projection.weight_elements = std::size_t{128} * width;
    projection.weight = f32(projection.weight_elements);
    projection.fast_input_elements = std::size_t{3 * 8} * width;
    projection.fast_input =
        Alloc<std::uint16_t>(a, projection.fast_input_elements, error_msg);
    projection.fast_counter_count =
        std::max(c.vocab_size / 128, 3 * c.AttentionQDim() / 128);
    projection.fast_counters =
        Alloc<std::uint32_t>(a, projection.fast_counter_count, error_msg);
    projection.fast_down_input_elements =
        std::size_t{8} * std::max(c.expert_ff, c.shared_expert_ff);
    projection.fast_down_input =
        Alloc<std::uint16_t>(a, projection.fast_down_input_elements, error_msg);
    ex.rotated_input_elements = std::size_t{3 * 8} * width;
    ex.rotated_input =
        Alloc<std::uint16_t>(a, ex.rotated_input_elements, error_msg);
    ex.slice_elements = std::max(T, std::size_t{64}) * (c.indexer_heads + 1) *
                        c.indexer_head_dim;
    ex.slice = f32(ex.slice_elements);
    const auto assignments = std::size_t{64} * c.num_experts_used;
    ex.routed_input = Alloc<std::uint16_t>(
        a,
        assignments *
            std::max(std::size_t{2} * c.hidden_size, std::size_t{c.expert_ff}),
        error_msg);
    ex.tiles = Alloc<Exl3RouteTile>(a, assignments, error_msg);
    ex.tile_count = Alloc<std::uint32_t>(a, 1, error_msg);
    ex.route_arrivals = Alloc<std::uint32_t>(a, 1, error_msg);
    ex.down_exponents = Alloc<std::int32_t>(a, assignments, error_msg);
    e->exl3_dense_rotations_ =
        Alloc<gufo::hip::Exl3RotateProblem>(a, 3, error_msg);
    e->exl3_dense_gemms_ = Alloc<gufo::hip::Exl3GemmProblem>(a, 3, error_msg);
    if (e->options_.max_batch >= 512) {
      // Shared EXL3 projections may run beside routed prefill. Their rotated
      // operands and descriptors must not borrow either routed or x_half space.
      auto& shared = e->exl3_shared_workspace_;
      shared.elements = std::size_t{e->options_.max_batch} *
                        std::max(std::size_t{2} * c.hidden_size,
                                 std::size_t{c.shared_expert_ff});
      shared.input = Alloc<std::uint16_t>(a, shared.elements, error_msg);
      shared.rotations = Alloc<gufo::hip::Exl3RotateProblem>(a, 3, error_msg);
      shared.gemms = Alloc<gufo::hip::Exl3GemmProblem>(a, 3, error_msg);
    }
    ex.head_segments =
        Alloc<std::uint32_t>(a, kMtpDraftExtraRows + 1, error_msg);
    ex.head_values = f32(c.vocab_size + std::size_t{kMtpDraftExtraRows} * 128);
    Exl3PrefillLayout layout;
    if (!PlanExl3Prefill(c, e->options_.max_batch, &layout, error_msg))
      return nullptr;
    auto& pp = e->exl3_prefill_;
    pp.max_assignments = layout.assignments;
    pp.max_buckets = layout.buckets;
    const std::size_t A = layout.assignments;
    pp.offsets =
        Alloc<std::uint32_t>(a, std::size_t{c.num_experts} + 1, error_msg);
    pp.slots = Alloc<std::int32_t>(a, A, error_msg);
    pp.buckets = Alloc<Exl3ExpertBucket>(a, layout.buckets, error_msg);
    pp.bucket_count = Alloc<std::uint32_t>(a, 1, error_msg);
    pp.input = Alloc<std::uint16_t>(a, A * std::max(c.hidden_size, c.expert_ff),
                                    error_msg);
    // Gate/up borrow the current stage view at dispatch. Packed out must
    // remain separate from stage down_e, which receives the scattered rows.
    pp.out = f32(A * c.hidden_size);
    pp.down_exponents = Alloc<std::int32_t>(a, A, error_msg);
    pp.inverse = Alloc<std::int32_t>(a, A, error_msg);
    e->exl3_prefill_scratch_bytes_ = layout.bytes;
  }
  s.tokens = Alloc<std::int32_t>(a, T, error_msg);
  // F32 plain weights do not contribute to max_half_cols during upload.
  // EXL3 prefill nevertheless narrows their inputs, including F32 HC/PLE.
  e->half_input_cols_ =
      model.output().exl3 != nullptr
          ? std::max(
                {model.max_half_cols(), hc_dim, hidden,
                 std::size_t{c.PleEmbeddingDim()}, std::size_t{c.SsmValueDim()},
                 std::size_t{c.AttentionQDim()}, std::size_t{c.hc_low_rank},
                 std::size_t{c.expert_ff}, std::size_t{c.shared_expert_ff}})
          : model.max_half_cols();
  s.x_half = Alloc<std::uint16_t>(a, T * e->half_input_cols_, error_msg);
  if (model.output().exl3 != nullptr) {
    e->hc_down_q8_ = !model.layers().empty();
    for (const auto& layer : model.layers())
      e->hc_down_q8_ = e->hc_down_q8_ && !layer.hc_attn.down_q8.empty() &&
                       !layer.hc_ffn.down_q8.empty();
  }
  for (void*& slot : s.x_q8) {
    slot = Alloc<std::uint8_t>(
        a,
        qfn_mmq_q8_1_bytes(static_cast<int>(kVecBatch),
                           static_cast<int>(model.max_q8_cols())),
        error_msg);
  }
  const auto q8_1 = [&](std::size_t k) {
    return Alloc<std::uint8_t>(
        a, qfn_mmq_q8_1_bytes(static_cast<int>(kVecBatch), static_cast<int>(k)),
        error_msg);
  };
  s.xn_q8_1 = q8_1(hc_dim);
  s.lo_q8_1 = q8_1(c.hc_low_rank);
  s.mixed_q8_1 = q8_1(hidden);
  s.gdn_q8_1 = q8_1(c.SsmValueDim());
  s.ctx_q8_1 = q8_1(c.AttentionQDim());
  s.x_q8t =
      Alloc<std::uint8_t>(a, Q8TiledBytes(T, model.max_q8_cols()), error_msg);
  s.res = f32(T * hc_dim);
  s.xn = f32(T * hc_dim);
  s.xn_half = Alloc<__half>(a, T * hc_dim, error_msg);
  s.xn_q8t = Alloc<std::uint8_t>(a, Q8TiledBytes(T, hc_dim), error_msg);
  s.lo = f32(T * c.hc_low_rank);
  s.mixed = f32(T * hidden);
  s.inject = f32(T * c.hc_count * HcInjectParts(hidden));
  s.block_out = f32(T * hidden);
  s.hc_defer_w = f32(T * c.hc_count);
  s.xn_scale = f32(T * c.hc_count);
  e->mask_words_ = c.SparseMaskWords(e->context_capacity_);
  const std::size_t linear_projection = std::max<std::size_t>(
      c.SsmConvChannels() + c.SsmValueDim(), c.ple_layer >= 0 ? hc_dim : 0);
  const std::size_t linear_floats =
      T * (linear_projection + 2 * c.ssm_num_v_heads + 2 * c.SsmKeyDim() +
           2 * c.SsmValueDim()) +
      (T + c.ssm_conv_kernel) * c.SsmConvChannels();
  const std::size_t attention_floats =
      T * (5 * c.AttentionQDim() + 4 * c.AttentionKvDim() +
           c.indexer_heads * c.indexer_head_dim + c.indexer_head_dim +
           e->mask_words_);
  const std::size_t expert_floats = slots * (2 * c.expert_ff + hidden);
  // Linear/full attention and routed experts run sequentially on stream_.
  // Their intermediates die before the next stage, including batched rows,
  // MTP catch-up and the deferred MoE epilogue consumed by Combine. Keep
  // stable addresses for graph replay without reserving all three stages.
  float* const stage =
      f32(std::max({linear_floats, attention_floats, expert_floats,
                    T * (linear_projection + hc_dim)}));
  if (stage == nullptr) {
    return nullptr;
  }
  // Mixer gates die before attention/experts start. PLE also uses this
  // buffer as its key, so keep it beyond PLE's gated output at stage.
  s.hc_gate = stage + T * linear_projection;
  float* cursor = stage;
  const auto take = [&](std::size_t count) {
    float* result = cursor;
    cursor += count;
    return result;
  };
  // Stacked and separate SSM projections are mutually exclusive. MTP's
  // projected embedding is consumed before attention and reuses this space.
  s.qkvz = take(T * linear_projection);
  s.qkv = s.qkvz;
  s.z = s.qkvz != nullptr ? s.qkvz + T * c.SsmConvChannels() : nullptr;
  s.alpha_beta = take(T * 2 * c.ssm_num_v_heads);
  s.conv_scratch = take((T + c.ssm_conv_kernel) * c.SsmConvChannels());
  s.qn = take(T * c.SsmKeyDim());
  s.kn = take(T * c.SsmKeyDim());
  s.gdn_raw = take(T * c.SsmValueDim());
  s.gdn_out = take(T * c.SsmValueDim());
  cursor = stage;
  s.qg = take(T * (2 * c.AttentionQDim() + 2 * c.AttentionKvDim()));
  s.q = take(T * c.AttentionQDim());
  s.attn_gate = take(T * c.AttentionQDim());
  s.k = take(T * c.AttentionKvDim());
  s.v = take(T * c.AttentionKvDim());
  s.iq = take(T * c.indexer_heads * c.indexer_head_dim);
  s.ik = take(T * c.indexer_head_dim);
  s.mask = reinterpret_cast<std::uint32_t*>(take(T * e->mask_words_));
  s.ctx = take(T * c.AttentionQDim());
  // Keep score/selection traffic near the device cache size. Captured
  // verification still needs all its rows at the maximum context.
  const std::size_t score_stride = std::size_t{e->mask_words_} * 32;
  e->select_score_floats_ =
      std::max(kVecBatch * score_stride,
               std::min(std::size_t{5 * 1024 * 1024}, 512 * score_stride));
  s.scores = f32(e->select_score_floats_);
  s.attn_partials = f32(static_cast<std::size_t>(kVecBatch) * c.num_heads *
                        kAttnSplits * (c.head_dim + 2));
  s.attn_lists = Alloc<std::uint32_t>(
      a, std::size_t{kVecBatch} * (c.indexer_top_k / c.compress_ratio),
      error_msg);
  s.attn_counts = Alloc<std::uint32_t>(a, kVecBatch, error_msg);
  if (c.ple_layer >= 0) {
    // PLE finishes before this layer's mixer/SSM. Its gate consumes key
    // and query before normalization/convolution reuse those two buffers.
    // The gated input stays separate until PleInject consumes both outputs.
    s.ple_key = s.hc_gate;
    s.ple_value = s.block_out;
    s.ple_query = s.xn;
    s.ple_gated = s.qkvz;
    s.ple_norm = s.hc_gate;
    s.ple_conv = s.xn;
    s.ple_history_scratch =
        f32(static_cast<std::size_t>(c.PleConvHistory()) * hc_dim);
    // The quantizer reads each n-gram row once. On this UMA target it can
    // consume the coherent reader buffer directly, without another device
    // allocation or upload. Its address stays fixed for graph replay; the
    // next-chunk prefetch uses separate storage until this forward finishes.
    void* pinned = nullptr;
    if (!Check(e->host_allocations_.Allocate(
                   &pinned, T * c.PleEmbeddingDim() * sizeof(float),
                   hipHostMallocMapped | hipHostMallocCoherent),
               "pinned n-gram buffer", error_msg)) {
      return nullptr;
    }
    e->host_emb_ = static_cast<float*>(pinned);
    if (!Check(hipHostGetDevicePointer(reinterpret_cast<void**>(&s.ple_emb),
                                       pinned, 0),
               "mapped n-gram buffer", error_msg)) {
      return nullptr;
    }
    e->host_rows_.resize(T * c.ple_heads);
    if (ngram)
      ngram->ReserveGather(T * c.ple_heads);
  }
  if (options.kvarn) {
    if (c.head_dim != hip::kKvarnHeadDim) {
      AssignError(error_msg, "KVarN needs 256-wide attention heads");
      return nullptr;
    }
    const std::size_t staging =
        hip::KvarnStagingRows(T) * std::size_t{c.AttentionKvDim()};
    e->kvarn_staging_k_ = Alloc<__half>(a, staging, error_msg);
    e->kvarn_staging_v_ = Alloc<__half>(a, staging, error_msg);
    if (e->kvarn_staging_k_ == nullptr || e->kvarn_staging_v_ == nullptr)
      return nullptr;
  }
  s.router = f32(T * (c.num_experts + 1));
  s.ids = Alloc<std::int32_t>(a, slots, error_msg);
  s.expert_counts = Alloc<std::uint32_t>(a, c.num_experts, error_msg);
  {
    const std::size_t compact = RoutedCompactRows(slots, c.num_experts);
    s.routed_bounds = Alloc<std::int32_t>(a, c.num_experts + 1, error_msg);
    s.routed_cursors = Alloc<std::int32_t>(a, c.num_experts, error_msg);
    s.rows_token = Alloc<std::int32_t>(a, compact, error_msg);
    s.rows_slot = Alloc<std::int32_t>(a, compact, error_msg);
    s.routed_tiles =
        Alloc<std::int32_t>(a, 3 * RoutedTileCapacity(slots, c), error_msg);
  }
  s.weights = f32(slots);
  cursor = stage;
  s.gate_e = take(slots * c.expert_ff);
  s.up_e = take(slots * c.expert_ff);
  s.down_e = take(slots * hidden);
  s.shexp_gate = f32(T * c.shared_expert_ff);
  s.shexp_up = f32(T * c.shared_expert_ff);
  s.shexp_out = f32(T * hidden);
  s.shexp_half = Alloc<__half>(a, T * c.shared_expert_ff, error_msg);
  s.logits =
      f32(static_cast<std::size_t>(e->options_.max_logit_rows) * c.vocab_size);
  {
    void* control = nullptr;
    void* tokens = nullptr;
    void* logits = nullptr;
    if (!Check(
            e->host_allocations_.Allocate(&control, sizeof(Session::Control)),
            "pinned control buffer", error_msg) ||
        !Check(e->host_allocations_.Allocate(&tokens, T * sizeof(std::int32_t)),
               "pinned token buffer", error_msg) ||
        !Check(
            e->host_allocations_.Allocate(
                &logits, static_cast<std::size_t>(e->options_.max_logit_rows) *
                             c.vocab_size * sizeof(float)),
            "pinned logits buffer", error_msg)) {
      return nullptr;
    }
    void* counts = nullptr;
    if (!Check(e->host_allocations_.Allocate(
                   &counts, c.num_experts * sizeof(std::uint32_t)),
               "pinned expert counts", error_msg)) {
      return nullptr;
    }
    e->counts_host_ = static_cast<std::uint32_t*>(counts);
    void* tiles = nullptr;
    if (!Check(e->host_allocations_.Allocate(
                   &tiles,
                   3 * RoutedTileCapacity(slots, c) * sizeof(std::int32_t)),
               "pinned routed tile map", error_msg)) {
      return nullptr;
    }
    e->tiles_host_ = static_cast<std::int32_t*>(tiles);
    e->control_host_ = static_cast<Session::Control*>(control);
    e->tokens_host_ = static_cast<std::int32_t*>(tokens);
    e->logits_host_ = static_cast<float*>(logits);
  }
  // The wide mixer route (F16 norm for the epilogue, tiled Q8 norm for the
  // W8A8 down projection) needs the four-stream geometry, 32-wide blocks and
  // a Q8_0 down projection.
  e->wide_mixer_ = c.hc_count == 4 && c.hidden_size % 32 == 0 &&
                   !model.layers().empty() &&
                   model.layers()[0].hc_ffn.down.type == GgmlType::kQ8_0;
  if (model.has_mtp()) {
    // The trunk's kept rows are session-owned. Its transient residual and
    // mixer buffers are free while MTP constructs its input. The split
    // projections consume h/embd before the first mixer rewrites xn/mixed.
    s.mtp_h = s.xn;
    s.mtp_embd = s.mixed;
    s.mtp_eproj = s.qkvz;
    s.mtp_res = s.res;
    s.mtp_argmax = Alloc<ArgmaxCandidate>(a, kArgmaxParts, error_msg);
    s.mtp_token = Alloc<std::int32_t>(a, 1, error_msg);
    const auto candidate_ids = std::max<std::size_t>(
        MtpCandidateWorkspaceSize(c.vocab_size),
        e->options_.max_speculative * kArgmaxParts *
            sizeof(PenaltyArgmaxCandidate) / sizeof(std::uint32_t));
    s.mtp_ids = Alloc<std::uint32_t>(a, candidate_ids, error_msg);
    s.mtp_scratch_ids = Alloc<std::uint32_t>(a, candidate_ids, error_msg);
    // Final selection consumes intermediate IDs before writing its scores.
    // Pack those scores behind the 64 returned IDs for one host transfer.
    s.mtp_scores = reinterpret_cast<float*>(s.mtp_ids + kMtpCandidates);
    if (!Check(e->host_allocations_.Allocate(&e->mtp_token_host_,
                                             sizeof(std::int32_t)),
               "pinned draft token", error_msg) ||
        !Check(e->host_allocations_.Allocate(&e->mtp_candidates_host_,
                                             sizeof(MtpCandidateLogits)),
               "pinned draft candidates", error_msg)) {
      return nullptr;
    }
    std::construct_at(e->mtp_candidates_host_);
    e->mtp_candidates_host_->size =
        std::min<std::size_t>(c.vocab_size, kMtpCandidates);
    const auto rows = std::min<std::size_t>(7, e->options_.max_speculative - 1);
    if (rows != 0) {
      e->verification_ids_ = Alloc<std::uint32_t>(
          a, rows * MtpCandidateWorkspaceSize(c.vocab_size), error_msg);
      e->verification_scratch_ids_ = Alloc<std::uint32_t>(
          a, rows * MtpCandidateWorkspaceSize(c.vocab_size), error_msg);
      const auto candidates = rows * (kMtpCandidates + kCompactPenaltyLimit);
      e->verification_candidates_ =
          Alloc<sampling::LogitCandidate>(a, candidates, error_msg);
      if (!Check(e->host_allocations_.Allocate(
                     &e->verification_candidates_host_,
                     candidates * sizeof(sampling::LogitCandidate)),
                 "pinned verification candidates", error_msg) ||
          !Check(hip::AllocateDevice(&e->verification_penalties_,
                                     rows * kCompactPenaltyLimit *
                                         sizeof(sampling::TokenPenalty)),
                 "verification penalties", error_msg))
        return nullptr;
      e->verification_penalty_capacity_ = rows * kCompactPenaltyLimit;
      e->verification_penalty_host_.reserve(e->verification_penalty_capacity_);
    }
  }
  for (void* p : a) {
    if (p == nullptr) {
      return nullptr;
    }
  }
  return e;
}

hip::KvBlockGeometry Executor::KvGeometry(core::SessionMode mode,
                                          std::uint32_t context) const {
  return SessionKvPlanes(config(), mode == core::SessionMode::kSpeculative,
                         options_.kvarn, context, options_.kvarn_bits)
      .geometry;
}

std::unique_ptr<Session> Executor::CreateSession(
    core::SessionMode mode, std::uint32_t max_context, std::string* error_msg,
    std::shared_ptr<hip::KvBlockPool> pool) const {
  std::unique_ptr<Session> s(new Session());
  s->owner_ = this;
  s->mtp_enabled_ = mode == core::SessionMode::kSpeculative;
  if (s->mtp_enabled_ && !has_mtp()) {
    AssignError(error_msg, "MTP session requires a loaded predictor");
    return nullptr;
  }
  const Config& c = config();
  if (max_context == 0 || max_context > context_capacity_) {
    AssignError(error_msg, "session context exceeds the model context");
    return nullptr;
  }
  s->max_context_ = max_context;
  // Before sparse attention starts, pooling can lag by indexer_top_k
  // rows. Afterwards only an incomplete block precedes the current batch.
  // Completed block keys remain in block_k; their raw rows are dead.
  s->index_capacity_ = IndexerCapacity(c, options_.max_batch, max_context);
  s->linear_.resize(c.num_layers);
  s->attention_.resize(c.num_layers);
  auto& a = s->allocations_;
  if (c.compress_ratio != kKvPooledRatio) {
    AssignError(error_msg, "pooled indexer keys need a ratio of 4");
    return nullptr;
  }
  const auto layout = SessionKvPlanes(c, s->mtp_enabled_, options_.kvarn,
                                      max_context, options_.kvarn_bits);
  const auto& logical = layout.geometry;
  if (pool &&
      !std::ranges::equal(pool->geometry().planes(), logical.planes())) {
    AssignError(error_msg, "Flash-Next KV pool has incompatible planes");
    return nullptr;
  }
  s->kv_pool_ = std::move(pool);
  const auto& kv = s->kv_pool_ ? s->kv_pool_->geometry() : logical;
  auto* kv_rows = s->kv_pool_ ? static_cast<std::byte*>(s->kv_pool_->data())
                              : Alloc<std::byte>(a, kv.bytes(), error_msg,
                                                 &s->allocated_bytes_);
  auto* kv_table = Alloc<std::uint32_t>(a, logical.blocks(), error_msg,
                                        &s->allocated_bytes_);
  if (kv_rows == nullptr || kv_table == nullptr)
    return nullptr;
  if (s->kv_pool_) {
    s->kv_blocks_.reserve(logical.blocks());
    if (!Check(
            s->kv_table_host_owner_.Allocate(
                &s->kv_table_host_, logical.blocks() * sizeof(std::uint32_t)),
            "KV host block table", error_msg))
      return nullptr;
  } else {
    std::vector<std::uint32_t> identity(logical.blocks());
    for (std::uint32_t b = 0; b < logical.blocks(); ++b)
      identity[b] = b;
    if (!Check(hipMemcpy(kv_table, identity.data(),
                         identity.size() * sizeof(std::uint32_t),
                         hipMemcpyHostToDevice),
               "KV block table", error_msg))
      return nullptr;
  }
  const auto plane_layer = [&](std::size_t plane, std::uint32_t slot) {
    return kv_rows + kv.plane_offset(plane) + slot * kv.layer_pitch(plane);
  };
  s->kv_table_ = kv_table;
  const std::uint32_t attention_layers =
      c.num_layers / c.full_attention_interval;
  const std::size_t fp16_layer =
      std::size_t{hip::kKvarnFp16Tokens} * c.AttentionKvDim();
  if (options_.kvarn) {
    s->kvarn_ = true;
    s->kvarn_bits_ = options_.kvarn_bits;
    s->kvarn_fp16_ =
        Alloc<__half>(a, KvarnFp16Elements(c), error_msg, &s->allocated_bytes_);
    if (s->kvarn_fp16_ == nullptr)
      return nullptr;
    s->kvarn_records_ =
        reinterpret_cast<std::uint8_t*>(plane_layer(layout.records, 0));
    s->kvarn_records_pitch_ = kv.layer_pitch(layout.records);
  }
  std::uint32_t kv_slot = 0;
  const std::size_t conv_elems =
      static_cast<std::size_t>(c.ssm_conv_kernel - 1) * c.SsmConvChannels();
  const std::size_t state_elems = static_cast<std::size_t>(c.ssm_num_v_heads) *
                                  c.ssm_head_dim * c.ssm_head_dim;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    if (c.IsLinearLayer(il)) {
      auto& l = s->linear_[il];
      l.conv_state =
          Alloc<float>(a, conv_elems, error_msg, &s->allocated_bytes_);
      l.state = Alloc<float>(a, state_elems, error_msg, &s->allocated_bytes_);
    } else {
      auto& at = s->attention_[il];
      if (layout.fp16_trunk) {
        at.k_cache =
            reinterpret_cast<__half*>(plane_layer(layout.keys, kv_slot));
        at.v_cache =
            reinterpret_cast<__half*>(plane_layer(layout.values, kv_slot));
      } else {
        at.kvarn_k = s->kvarn_fp16_ + kv_slot * fp16_layer;
        at.kvarn_v = s->kvarn_fp16_ + (attention_layers + kv_slot) * fp16_layer;
        at.kvarn_records =
            s->kvarn_records_ + std::size_t{kv_slot} * s->kvarn_records_pitch_;
        at.kvarn_pass = &s->kvarn_pass_;
      }
      at.block_k =
          reinterpret_cast<__half*>(plane_layer(layout.pooled, kv_slot));
      at.kv_blocks = kv_table;
      ++kv_slot;
      at.index_k = Alloc<float>(
          a, static_cast<std::size_t>(s->index_capacity_) * c.indexer_head_dim,
          error_msg, &s->allocated_bytes_);
    }
  }
  if (c.ple_layer >= 0) {
    const std::size_t hist =
        static_cast<std::size_t>(c.PleConvHistory()) * c.HcDim();
    s->ple_history_ = Alloc<float>(a, hist, error_msg, &s->allocated_bytes_);
  }
  s->control_ = Alloc<Session::Control>(a, 1, error_msg, &s->allocated_bytes_);
  if (s->mtp_enabled_) {
    s->mtp_.target_hidden = Alloc<float>(
        a, static_cast<std::size_t>(options_.max_speculative) * c.HcDim(),
        error_msg, &s->allocated_bytes_);
    // The draft block's FP16 K/V follow the trunk's FP16 layers, or are the
    // only FP16 layer of a KVarN session.
    const std::uint32_t draft_slot = layout.fp16_trunk ? kv_slot : 0;
    s->mtp_.k_cache =
        reinterpret_cast<__half*>(plane_layer(layout.keys, draft_slot));
    s->mtp_.v_cache =
        reinterpret_cast<__half*>(plane_layer(layout.values, draft_slot));
    s->mtp_.block_k =
        reinterpret_cast<__half*>(plane_layer(layout.pooled, kv_slot));
    s->mtp_.kv_blocks = kv_table;
    s->mtp_.index_k =
        Alloc<float>(a, std::size_t{s->index_capacity_} * c.indexer_head_dim,
                     error_msg, &s->allocated_bytes_);
    s->mtp_.h = Alloc<float>(a, c.HcDim(), error_msg, &s->allocated_bytes_);
    if (!model_->draft_output().empty()) {
      s->mtp_.draft_rows = Alloc<std::uint8_t>(
          a,
          std::size_t{kMtpDraftExtraRows} * model_->output_row_bytes() + 4096,
          error_msg, &s->allocated_bytes_);
      s->mtp_.draft_ids = Alloc<std::int32_t>(a, kMtpDraftExtraRows, error_msg,
                                              &s->allocated_bytes_);
    }
  }
  for (void* p : a) {
    if (p == nullptr) {
      return nullptr;
    }
  }
  // The zero fills above run on the null stream; nothing may read the new
  // buffers until they have landed.
  if (!Check(hipDeviceSynchronize(), "session init", error_msg)) {
    return nullptr;
  }
  // The table starts with the special tokens; every other slot is empty.
  if (s->mtp_.draft_rows != nullptr &&
      (!ClearDraftRows(*s, 0, error_msg) ||
       !AppendDraftRows(*s, 0, model_->draft_specials(), error_msg) ||
       !Check(hipStreamSynchronize(stream_), "draft table init", error_msg))) {
    return nullptr;
  }
  return s;
}

bool Executor::AppendDraftRows(Session& session, std::uint32_t first,
                               std::span<const std::int32_t> ids,
                               std::string* error_msg) const {
  if (ids.empty())
    return true;
  if (session.owner_ != this || session.mtp_.draft_rows == nullptr ||
      first > kMtpDraftExtraRows || ids.size() > kMtpDraftExtraRows - first) {
    AssignError(error_msg, "invalid draft table append");
    return false;
  }
  const auto& output = model_->output();
  const std::size_t row_bytes = model_->output_row_bytes();
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] < 0 || static_cast<std::uint32_t>(ids[i]) >= output.rows) {
      AssignError(error_msg, "draft table token out of range");
      return false;
    }
    // The ID goes by value, so no host buffer has to outlive the call.
    if ((output.exl3 == nullptr &&
         !Check(
             hipMemcpyAsync(session.mtp_.draft_rows + (first + i) * row_bytes,
                            static_cast<const std::uint8_t*>(output.data) +
                                static_cast<std::size_t>(ids[i]) * row_bytes,
                            row_bytes, hipMemcpyDeviceToDevice, stream_),
             "draft table row", error_msg)) ||
        !Check(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(
                                     session.mtp_.draft_ids + first + i),
                                 ids[i], 1, stream_),
               "draft table id", error_msg))
      return false;
  }
  return true;
}

bool Executor::ClearDraftRows(Session& session, std::uint32_t first,
                              std::string* error_msg) const {
  if (session.owner_ != this || session.mtp_.draft_ids == nullptr ||
      first > kMtpDraftExtraRows) {
    AssignError(error_msg, "invalid draft table clear");
    return false;
  }
  // Stale rows stay; an ID of -1 masks the slot.
  return first == kMtpDraftExtraRows ||
         Check(
             hipMemsetAsync(session.mtp_.draft_ids + first, 0xFF,
                            (kMtpDraftExtraRows - first) * sizeof(std::int32_t),
                            stream_),
             "draft table clear", error_msg);
}

DeviceTensor Executor::DraftTable(const Session& session) const noexcept {
  DeviceTensor table = model_->output();
  table.data = session.mtp_.draft_rows;
  table.rows = kMtpDraftExtraRows;
  return table;
}

bool Executor::DraftLogits(const Session& session, const float* x,
                           float* logits, std::string* error_msg) const {
  const auto& prefix = model_->draft_output();
  if (model_->output().exl3 != nullptr) {
    Exl3DraftHead(model_->output(), x, logits, prefix.rows,
                  session.mtp_.draft_ids, kMtpDraftExtraRows, exl3_, stream_);
    if (diagnostics_active_) {
      cycle_diagnostics_.head_segments += (prefix.rows + 127) / 128;
      if (!DiagnosticCount(exl3_.head_segments,
                           &cycle_diagnostics_.head_segments, error_msg))
        return false;
    }
    return true;
  }
  if (!Dense(prefix, x, logits, 1, error_msg) ||
      !Dense(DraftTable(session), x, logits + prefix.rows, 1, error_msg))
    return false;
  MaskDraftLogits(logits + prefix.rows, session.mtp_.draft_ids,
                  kMtpDraftExtraRows, stream_);
  return true;
}

void Executor::FlushKvarnGroups(const Session& session, std::uint32_t first,
                                std::uint32_t groups) const {
  const Config& c = config();
  const std::uint32_t attention = c.num_layers / c.full_attention_interval;
  const std::size_t layer =
      std::size_t{hip::kKvarnFp16Tokens} * c.AttentionKvDim();
  KvarnQuantizeGroups(
      session.kvarn_fp16_, session.kvarn_fp16_ + std::size_t{attention} * layer,
      layer, session.kvarn_records_, session.kvarn_records_pitch_,
      session.kv_table_, c.num_kv_heads, attention, first, groups, stream_,
      kKvarnRingRows, options_.kvarn_bits);
}

bool Executor::PlanKvarn(Session& session, std::uint32_t start, std::uint32_t n,
                         std::string* error_msg) const {
  KvarnPass pass{.start = start, .rows = n};
  if (n <= hip::kKvarnMaxPassRows) {
    const auto plan = hip::PlanKvarnPass(session.kvarn_tail_, start, n);
    if (!plan.error.empty()) {
      AssignError(error_msg, std::string(plan.error));
      return false;
    }
    FlushKvarnGroups(session, plan.first_group, plan.groups);
    pass.flushed_end = plan.flushed_end;
  } else {
    const auto plan = hip::PlanKvarnPrefill(session.kvarn_tail_, start, n);
    if (!plan.error.empty()) {
      AssignError(error_msg, std::string(plan.error));
      return false;
    }
    FlushKvarnGroups(session, plan.first_group, plan.groups);
    pass.prefill = true;
    // A chunk inside the sink stages from position 0 and stores its sink
    // rows afterwards; otherwise the ring rows of its first group lead.
    pass.staging_first = start < hip::kKvarnSinkTokens ? 0 : plan.target;
    pass.quant_end = plan.target;
    pass.flushed_end = plan.flushed_end;
    if (start + n - pass.staging_first >
        hip::KvarnStagingRows(options_.max_batch)) {
      AssignError(error_msg, "KVarN prompt chunk exceeds its staging rows");
      return false;
    }
  }
  session.kvarn_pass_ = pass;
  return true;
}

bool Executor::FinishKvarn(Session& session, std::string* error_msg) const {
  const auto& pass = session.kvarn_pass_;
  const std::uint32_t end = pass.start + pass.rows;
  session.kvarn_tail_.flushed_end = pass.flushed_end;
  session.kvarn_tail_.written_end =
      std::max(session.kvarn_tail_.written_end, end);
  // A pass ending exactly at a group end quantizes that group now, so the
  // records of every committed state reach its group floor (snapshots).
  if (!pass.prefill && end % hip::kKvarnGroupTokens == 0 &&
      end > hip::kKvarnSinkTokens && pass.flushed_end < end) {
    FlushKvarnGroups(session, (end >> hip::kKvarnGroupShift) - 1, 1);
    session.kvarn_tail_.flushed_end = end;
    // The forward already synchronized; a snapshot capture right after it
    // copies these records on its own stream, so they must be complete.
    return Check(hipStreamSynchronize(stream_), "KVarN flush", error_msg);
  }
  return Check(hipGetLastError(), "KVarN flush", error_msg);
}

bool Executor::EnsureRollback(Session& session, std::uint32_t depth,
                              std::string* error_msg) const {
  if (depth <= session.rollback_depth_)
    return true;
  if (depth >= options_.max_speculative ||
      depth > std::size(session.ple_snapshots_.rows) ||
      session.spec_tokens_ != 0) {
    AssignError(error_msg, "invalid rollback allocation depth");
    return false;
  }
  const auto& c = config();
  const std::size_t conv =
      std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const std::size_t linear =
      c.num_layers - c.num_layers / c.full_attention_interval;
  const std::size_t ple =
      c.ple_layer >= 0 ? std::size_t{c.PleConvHistory()} * c.HcDim() : 0;
  session.rollback_allocations_.reserve(depth);
  session.ngram_snapshots_.resize(depth);
  for (auto row = session.rollback_depth_; row < depth; ++row) {
    const auto state = GdnRollbackRowFloats(row, c.ssm_num_k_heads,
                                            c.ssm_num_v_heads, c.ssm_head_dim);
    const auto bytes = (linear * (conv + state) + ple) * sizeof(float);
    float* allocation = nullptr;
    if (!Check(hip::AllocateDevice(&allocation, bytes), "rollback allocation",
               error_msg))
      return false;
    session.rollback_allocations_.push_back(allocation);
    session.rollback_bytes_ += bytes;
    auto* next = allocation;
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      if (!c.IsLinearLayer(il))
        continue;
      session.linear_[il].conv_snapshots.rows[row] = next;
      next += conv;
      session.linear_[il].state_snapshots.rows[row] = next;
      next += state;
    }
    if (ple != 0) {
      session.ple_snapshots_.rows[row] = next;
    }
    session.rollback_depth_ = row + 1;
  }
  return true;
}

std::size_t Executor::SessionBytes(
    core::SessionMode mode, std::uint32_t max_context,
    std::uint32_t rollback_depth) const noexcept {
  const auto& c = config();
  const std::size_t attention = c.num_layers / c.full_attention_interval;
  const std::size_t linear = c.num_layers - attention;
  const std::size_t conv =
      std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const std::size_t state =
      std::size_t{c.ssm_num_v_heads} * c.ssm_head_dim * c.ssm_head_dim;
  const std::size_t ple =
      c.ple_layer >= 0 ? std::size_t{c.PleConvHistory()} * c.HcDim() : 0;
  // K/V/pooled rows and the block table (CreateSession); the raw indexer
  // rings stay per layer.
  const auto kv_layout =
      SessionKvPlanes(c, mode == core::SessionMode::kSpeculative,
                      options_.kvarn, max_context, options_.kvarn_bits);
  const std::size_t kv_blocks =
      kv_layout.geometry.bytes() +
      std::size_t{kv_layout.geometry.blocks()} * sizeof(std::uint32_t) +
      (options_.kvarn ? KvarnFp16Elements(c) * sizeof(__half) : 0);
  const std::size_t index =
      std::size_t{IndexerCapacity(c, options_.max_batch, max_context)} *
      c.indexer_head_dim * sizeof(float);
  const auto rollback_state =
      rollback_depth == 0
          ? 0
          : state + (rollback_depth - 1) *
                        GdnRollbackRowFloats(1, c.ssm_num_k_heads,
                                             c.ssm_num_v_heads, c.ssm_head_dim);
  return ((linear * conv + ple) * (rollback_depth + 1) +
          linear * (state + rollback_state)) *
             sizeof(float) +
         kv_blocks + attention * index + sizeof(Session::Control) +
         (mode == core::SessionMode::kSpeculative
              ? index +
                    std::size_t{options_.max_speculative + 1} * c.HcDim() *
                        sizeof(float) +
                    (model_->draft_output().empty()
                         ? 0
                         : std::size_t{kMtpDraftExtraRows} *
                                   (model_->output_row_bytes() +
                                    sizeof(std::int32_t)) +
                               4096)
              : 0);
}

std::size_t Executor::DeferredScratchBytes() const {
  const auto& c = config();
  std::size_t bytes = batch_logits_ == nullptr
                          ? std::size_t{8} *
                                std::min(8U, options_.max_logit_rows) *
                                c.vocab_size * sizeof(float)
                          : 0;
  const auto linear = c.num_layers - c.num_layers / c.full_attention_interval;
  const auto conv = std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const auto state =
      std::size_t{c.ssm_num_v_heads} * c.ssm_head_dim * c.ssm_head_dim;
  const auto ple =
      c.ple_layer >= 0 ? std::size_t{c.PleConvHistory()} * c.HcDim() : 0;
  const auto mutable_bytes = (linear * (conv + state) + ple) * sizeof(float);
  if (!checkpoint_storage_) {
    bytes +=
        mutable_bytes + (has_mtp() ? std::size_t{options_.max_speculative} *
                                         c.HcDim() * sizeof(float)
                                   : 0);
  }
  // Committed spare blocks for captures, whether or not they exist yet.
  return bytes + SnapshotAllocator::SpareBytes(mutable_bytes);
}

/// Column-tile width for the routed expert GEMMs: the tile at or above twice
/// the mean bucket, so most experts fit one tile with little padding.
int RoutedTileCols(std::uint32_t n_tokens, std::uint32_t n_used,
                   std::uint32_t n_experts) {
  const std::uint32_t mean =
      std::max<std::uint32_t>(1, n_tokens * n_used / std::max(n_experts, 1u));
  for (int cols = 16; cols < 80; cols += 16) {
    if (static_cast<std::uint32_t>(cols) >= 2 * mean) {
      return cols;
    }
  }
  return 80;
}

const void* Executor::ReadyQ8(const float* x, std::uint32_t n_tokens,
                              std::uint32_t k) const {
  for (const ReadyRows* r : {&xn_q8_ready_, &mixed_q8_ready_}) {
    if (r->src != nullptr && r->src == x && r->n == n_tokens && r->k == k) {
      return r->data;
    }
  }
  return nullptr;
}

void Executor::DropReadyQ8() const {
  xn_q8_ready_ = {};
  mixed_q8_ready_ = {};
  exl3_rotated_ = {};  // also producer-rotated rows
}

bool Executor::Quantize(const float* x, std::uint32_t n_tokens, std::uint32_t k,
                        Q8Input* q, std::string* error_msg,
                        bool q8_weight) const {
  q->x = x;
  q->data = nullptr;
  q->n = n_tokens;
  q->k = k;
  if ((model_->output().exl3 != nullptr && !q8_weight) ||
      MatrixRows(n_tokens)) {
    return true;  // EXL3 consumes the original FP32 rows; wide Q8 stages per
                  // call
  }
  if (const void* ready = ReadyQ8(x, n_tokens, k)) {
    q->data = ready;
    return true;
  }
  // Two slots alternate, so an input stays valid across one other
  // quantization; captured graphs replay the same alternation.
  void* slot = s_.x_q8[q8_slot_];
  q8_slot_ ^= 1u;
  if (qfn_mmq_quantize_q8_1(x, slot, static_cast<int>(n_tokens),
                            static_cast<int>(k), stream_) != 0) {
    AssignError(error_msg, "activation quantization failed");
    return false;
  }
  q->data = slot;
  return true;
}

bool Executor::Dense(const DeviceTensor& w, const Q8Input& q, float* out,
                     std::string* error_msg) const {
  if (w.type == GgmlType::kQ8_0 && q.data != nullptr) {
    if (w.cols != q.k) {
      AssignError(error_msg, "quantized input width mismatch");
      return false;
    }
    if (qfn_mmq_q8_0_dense_vec_preq(
            w.data, nullptr, q.data, out, static_cast<int>(w.rows),
            static_cast<int>(q.n), static_cast<int>(w.cols), stream_) != 0) {
      AssignError(error_msg, "Q8_0 GEMV failed");
      return false;
    }
    return true;
  }
  return Dense(w, q.x, out, q.n, error_msg);
}

bool Executor::GatedDense(const DeviceTensor& up, const DeviceTensor& gate,
                          const float* x, float* out, std::uint32_t n_tokens,
                          const DeviceTensor* down,
                          std::string* error_msg) const {
  if (prefill_phase && gate.exl3 != nullptr && up.exl3 != nullptr &&
      gate.cols == up.cols) {
    if (!Exl3DensePrefill({&gate, &up}, {out, s_.shexp_gate}, x, n_tokens,
                          error_msg))
      return false;
    Swiglu(out, s_.shexp_gate, std::size_t{n_tokens} * gate.rows, stream_);
    return true;
  }
  // Decode and verification use the same fused projections and activation.
  // A different SwiGLU rounding can change later activation quantization.
  if (!MatrixRows(n_tokens) && up.type == GgmlType::kQ8_0 &&
      gate.type == GgmlType::kQ8_0 && up.rows == gate.rows &&
      up.cols == gate.cols) {
    Q8Input xq;
    if (!Quantize(x, n_tokens, up.cols, &xq, error_msg)) {
      return false;
    }
    if (qfn_mmq_q8_0_dense_vec_preq(up.data, gate.data, xq.data, out,
                                    static_cast<int>(up.rows),
                                    static_cast<int>(n_tokens),
                                    static_cast<int>(up.cols), stream_) != 0) {
      AssignError(error_msg, "gated Q8_0 GEMV failed");
      return false;
    }
    return true;
  }
  // Swiglu is in place over its first operand.
  if (!Dense(gate, x, out, n_tokens, error_msg) ||
      !Dense(up, x, s_.shexp_gate, n_tokens, error_msg)) {
    return false;
  }
  // A wide batch writes the down projection's staged input directly (the
  // F32 rows are read by nothing else): F16 rows for the F16 route, else
  // the tiled Q8 layout; the cache lets Dense skip its activation pass.
  if (down != nullptr && down->cols == up.rows && MatrixRows(n_tokens) &&
      n_tokens <= options_.max_batch) {
    if (DenseF16Route(*down, n_tokens)) {
      // A private F16 buffer: s_.x_half keeps the narrowed token rows the
      // router left there, which the routed experts read next.
      SwigluHalf(out, s_.shexp_gate, s_.shexp_half,
                 static_cast<std::size_t>(n_tokens) * up.rows, stream_);
      shexp_half_ready_ = true;
      return true;
    }
    if (down->type == GgmlType::kQ8_0 &&
        SwigluQ8Tiled(out, s_.shexp_gate, s_.x_q8t, n_tokens, up.rows,
                      stream_)) {
      q8t_src_ = out;
      q8t_rows_ = n_tokens;
      q8t_cols_ = up.rows;
      return true;
    }
  }
  Swiglu(out, s_.shexp_gate, static_cast<std::size_t>(n_tokens) * up.rows,
         stream_);
  return true;
}

void Executor::PrepareHalfInput(const float* x, std::uint32_t rows,
                                std::uint32_t cols) const {
  if (half_src_ == x && half_rows_ == rows && half_cols_ == cols &&
      !half_bf16_) {
    return;
  }
  NarrowActivations(x, s_.x_half, false, static_cast<std::size_t>(rows) * cols,
                    stream_);
  half_src_ = x;
  half_rows_ = rows;
  half_cols_ = cols;
  half_bf16_ = false;
}

bool Executor::DenseF16Route(const DeviceTensor& w,
                             std::uint32_t n_tokens) const {
  return w.type == GgmlType::kQ8_0 && MatrixRows(n_tokens) &&
         w.rows >= kDenseF16MinRows && w.cols <= kDenseF16MaxCols &&
         w.cols <= model_->max_half_cols();
}

bool Executor::Exl3DensePrefill(
    std::initializer_list<const DeviceTensor*> weights,
    std::initializer_list<float*> outputs, const float* x, std::uint32_t rows,
    std::string* error, const Exl3DenseWorkspace* workspace,
    hipStream_t launch_stream) const {
  const auto execution = launch_stream != nullptr ? launch_stream : stream_;
  auto* rotations = exl3_dense_rotations_;
  auto* gemms = exl3_dense_gemms_;
  try {
    if (weights.size() == 0 || weights.size() != outputs.size() ||
        weights.size() > 3)
      throw std::invalid_argument(
          "EXL3 dense prefill requires 1..3 matching outputs");
    struct Target {
      const DeviceTensor* weight;
      float* output;
    };
    std::array<Target, 3> targets{};
    const auto output_count = weights.size();
    const auto k = (*weights.begin())->cols;
    auto output = outputs.begin();
    bool sliced = false;
    for (std::size_t i = 0; const auto* w : weights) {
      if (w->exl3 == nullptr || w->experts != 1 || w->cols != k ||
          w->exl3->k != k || !gufo::hip::Exl3PrefillSupported(*w->exl3) ||
          std::uint64_t{w->output_begin} + w->rows > w->exl3->n)
        throw std::invalid_argument(
            "unsupported EXL3 dense prefill projection");
      sliced |= w->output_begin != 0 || w->rows != w->exl3->n;
      targets[i++] = {w, *output++};
    }
    if (sliced)
      for (const auto* w : weights)
        if (w->exl3 != targets[0].weight->exl3)
          throw std::invalid_argument(
              "EXL3 dense slices require a shared projection");
    // Output rotation is independent in each 128-channel Hadamard block.
    // An aligned singleton can project only its requested blocks, retaining
    // the parent's input rotation and packed trellis K-step stride.
    const bool direct_slice = sliced && output_count == 1 &&
                              targets[0].weight->output_begin % 128 == 0 &&
                              targets[0].weight->rows % 128 == 0;
    // Multiple views still share one full parent projection and are copied
    // before scratch is reused; an unaligned view also needs the full blocks.
    const auto count = sliced ? std::size_t{1} : output_count;
    if (rows == 0)
      return true;
    // q/k/v have different N; K/V and gate/up can share a GEMM launch.
    // All three still rotate together, each with its own checkpoint suh.
    std::sort(targets.begin(), targets.begin() + count, [](auto a, auto b) {
      return std::pair{a.weight->exl3->n, a.weight->exl3->rate.bits} <
             std::pair{b.weight->exl3->n, b.weight->exl3->rate.bits};
    });
    // Main-stream dense work reuses the larger routed/decode FP16 slab.
    // Concurrent shared experts supply their own operands and descriptors.
    // Tiny batches still retain the decode slab (24 * widest K).
    auto* rotated = exl3_prefill_.input;
    std::size_t capacity = std::size_t{exl3_prefill_.max_assignments} *
                           std::max(config().hidden_size, config().expert_ff);
    if (exl3_.projection.fast_input_elements > capacity) {
      rotated = exl3_.projection.fast_input;
      capacity = exl3_.projection.fast_input_elements;
    }
    if (workspace != nullptr) {
      rotated = workspace->input;
      capacity = workspace->elements;
      rotations = workspace->rotations;
      gemms = workspace->gemms;
    }
    const auto piece = static_cast<std::uint32_t>(
        std::min<std::size_t>(options_.max_batch, capacity / (count * k)));
    if (rotated == nullptr || piece == 0 || rotations == nullptr ||
        gemms == nullptr)
      throw std::runtime_error("EXL3 dense prefill exceeds rotation scratch");
    for (std::uint32_t first = 0; first < rows; first += piece) {
      const auto n = std::min(piece, rows - first);
      std::array<Exl3DensePrefillProblem, 3> problems{};
      for (std::size_t i = 0; i < count; ++i) {
        const auto& t = targets[i];
        const auto& l = *t.weight->exl3;
        if (sliced && !direct_slice &&
            exl3_.slice_elements < std::size_t{n} * l.n)
          throw std::runtime_error("EXL3 dense prefill slice exceeds scratch");
        const auto begin = direct_slice ? t.weight->output_begin : 0;
        problems[i] = {l.trellis + std::size_t{begin} * l.rate.bits,
                       l.suh,
                       l.svh + begin,
                       x + std::size_t{first} * k,
                       k,
                       rotated + i * std::size_t{n} * k,
                       sliced && !direct_slice
                           ? exl3_.slice
                           : t.output + std::size_t{first} * t.weight->rows,
                       n,
                       direct_slice ? l.n : 0};
      }
      PrepareExl3DensePrefill(std::span(problems.data(), count), rotations,
                              gemms, execution);
      gufo::hip::LaunchExl3RotateRowsGrouped(rotations, count, n, k, execution);
      for (std::size_t begin = 0; begin < count;) {
        const auto& l = *targets[begin].weight->exl3;
        auto end = begin + 1;
        while (end < count && targets[end].weight->exl3->n == l.n &&
               targets[end].weight->exl3->rate == l.rate)
          ++end;
        // Always X3, including a singleton/tail; a GEMV row threshold would
        // change the reduction tree when a prompt/checkpoint chunk moves.
        gufo::hip::LaunchExl3PrefillGemmGrouped(
            gemms + begin, end - begin, n, k,
            direct_slice ? targets[begin].weight->rows : l.n, l.rate, false,
            execution, direct_slice);
        if (diagnostics_active_)
          ++cycle_diagnostics_.prefill_dense_gemm_groups;
        begin = end;
      }
      if (sliced && !direct_slice) {
        for (std::size_t i = 0; i < output_count; ++i) {
          const auto& t = targets[i];
          if (!Check(hipMemcpy2DAsync(
                         t.output + std::size_t{first} * t.weight->rows,
                         t.weight->rows * sizeof(float),
                         exl3_.slice + t.weight->output_begin,
                         t.weight->exl3->n * sizeof(float),
                         t.weight->rows * sizeof(float), n,
                         hipMemcpyDeviceToDevice, execution),
                     "EXL3 dense prefill slice", error))
            return false;
        }
      }
      if (diagnostics_active_) {
        cycle_diagnostics_.prefill_dense_rows += std::uint64_t{n} * count;
        ++cycle_diagnostics_.prefill_dense_rotation_groups;
      }
    }
    return true;
  } catch (const std::exception& e) {
    AssignError(error, e.what());
    return false;
  }
}

Exl3InputSlabs Executor::Exl3Slabs(
    std::span<const DeviceTensor* const> consumers, std::uint32_t rows) const {
  if (prefill_phase || exl3_.rotated_input == nullptr || rows == 0 ||
      rows > gufo::hip::kExl3FastRotatedRows || consumers.empty() ||
      consumers.size() > gufo::hip::kExl3FastMaxGroup)
    return {};
  const auto* first = consumers.front()->exl3;
  Exl3InputSlabs slabs{};
  for (const auto* w : consumers) {
    const auto* l = w->exl3;
    if (l == nullptr || first == nullptr || w->experts != 1 ||
        w->output_begin != 0 || w->rows != l->n || w->cols != l->k ||
        !gufo::hip::Exl3FastSupported(*l) || l->k != first->k ||
        !(l->rate == first->rate))
      return {};
    slabs.suh[slabs.count++] = l->suh;
  }
  if (std::size_t{slabs.count} * gufo::hip::kExl3FastRotatedRows * first->k >
      exl3_.rotated_input_elements)
    return {};
  slabs.out = exl3_.rotated_input;
  slabs.pass_rows = gufo::hip::Exl3FastPassRows(rows);
  slabs.k = first->k;
  return slabs;
}

void Executor::NoteExl3Rotated(
    const Exl3InputSlabs& slabs, const float* x, std::uint32_t rows,
    std::span<const DeviceTensor* const> consumers) const {
  exl3_rotated_ = {};
  if (slabs.count == 0 || slabs.count != consumers.size())
    return;
  exl3_rotated_.x = x;
  exl3_rotated_.rows = rows;
  exl3_rotated_.count = slabs.count;
  for (std::size_t c = 0; c < consumers.size(); ++c)
    exl3_rotated_.consumers[c] = consumers[c]->exl3;
}

const std::uint16_t* Executor::TakeExl3Rotated(
    const float* x, std::uint32_t rows,
    std::span<const DeviceTensor* const> consumers) const {
  const auto& r = exl3_rotated_;
  if (r.x == nullptr || r.x != x || r.rows != rows ||
      r.count != consumers.size())
    return nullptr;
  for (std::size_t c = 0; c < consumers.size(); ++c)
    if (r.consumers[c] != consumers[c]->exl3)
      return nullptr;
  exl3_rotated_ = {};
  return exl3_.rotated_input;
}

bool Executor::Exl3Group(std::initializer_list<const DeviceTensor*> weights,
                         std::initializer_list<float*> outputs, const float* x,
                         std::uint32_t rows, std::string* error,
                         std::size_t shape_basis_n) const {
  if (prefill_phase)
    return Exl3DensePrefill(weights, outputs, x, rows, error);
  try {
    if (weights.size() != outputs.size() || weights.size() > 3)
      throw std::logic_error("invalid EXL3 projection group");
    std::array<gufo::hip::Exl3FastTarget, 3> targets{};
    auto output = outputs.begin();
    std::size_t count = 0;
    for (const auto* weight : weights) {
      if (weight->exl3 == nullptr || weight->output_begin != 0 ||
          weight->rows != weight->exl3->n)
        throw std::logic_error("EXL3 group requires complete projections");
      targets[count++] = {weight->exl3, *output++, weight->rows};
    }
    const auto& p = exl3_.projection;
    const gufo::hip::Exl3FastWorkspace ws{
        p.fast_input,      p.fast_input_elements,
        p.fast_counters,   p.fast_counter_count,
        p.fast_down_input, p.fast_down_input_elements};
    const std::span<const gufo::hip::Exl3FastTarget> group(targets.data(),
                                                           count);
    if (!prefill_phase && gufo::hip::Exl3FastGroupSupported(group, ws)) {
      // A producer epilogue may already have rotated x for this group.
      if (const auto* rotated = TakeExl3Rotated(
              x, rows, std::span(weights.begin(), weights.size()))) {
        const gufo::hip::Exl3FastWorkspace pre{
            const_cast<std::uint16_t*>(rotated),
            exl3_.rotated_input_elements,
            p.fast_counters,
            p.fast_counter_count,
            p.fast_down_input,
            p.fast_down_input_elements};
        gufo::hip::LaunchExl3FastProjections(group, x, (*weights.begin())->cols,
                                             rows, pre, stream_, true,
                                             shape_basis_n);
        return true;
      }
      for (std::uint32_t first = 0; first < rows;
           first += gufo::hip::kExl3FastMaxRows) {
        auto pass = targets;
        for (std::size_t i = 0; i < count; ++i)
          pass[i].y += std::size_t{first} * pass[i].y_stride;
        const auto n =
            std::min<std::uint32_t>(gufo::hip::kExl3FastMaxRows, rows - first);
        gufo::hip::LaunchExl3FastProjections(
            std::span(pass.data(), count),
            x + std::size_t{first} * (*weights.begin())->cols,
            (*weights.begin())->cols, n, ws, stream_, false, shape_basis_n);
      }
      return true;
    }
    output = outputs.begin();
    for (const auto* weight : weights)
      if (!Dense(*weight, x, *output++, rows, error))
        return false;
    return true;
  } catch (const std::exception& e) {
    AssignError(error, e.what());
    return false;
  }
}

bool Executor::MtpKvProjection(const DeviceLayer& l, const float* x,
                               std::uint32_t rows,
                               std::string* error_msg) const {
  if (l.attn_qkv.empty() && model_->output().exl3 != nullptr) {
    // Omit Q while retaining the full group's W/TN launch geometry and
    // therefore exactly the K/V reduction used by full MTP execution.
    const auto basis =
        std::size_t{l.attn_q.rows} + l.attn_k.rows + l.attn_v.rows;
    return Exl3Group({&l.attn_k, &l.attn_v}, {s_.k, s_.v}, x, rows, error_msg,
                     basis);
  }
  if (l.attn_qkv.empty())
    return Exl3Group({&l.attn_k, &l.attn_v}, {s_.k, s_.v}, x, rows, error_msg);
  auto kv = l.attn_qkv;
  const TensorRef layout{.type = kv.type, .cols = kv.cols};
  kv.data = static_cast<std::uint8_t*>(kv.data) +
            std::size_t{2} * config().AttentionQDim() * layout.RowBytes();
  kv.rows = 2 * config().AttentionKvDim();
  return DenseBatch(kv, x, s_.qg, rows, error_msg);
}

bool Executor::StoreMtpKv(const DeviceLayer& l, const Session& session,
                          std::uint32_t rows, const YarnRope& yarn,
                          std::string* error_msg) const {
  const auto& c = config();
  if (l.attn_qkv.empty()) {
    PrepareAttentionKv(
        s_.k, s_.v, l.attn_k_norm.f32(), session.mtp_.k_cache,
        session.mtp_.v_cache, rows, c.num_kv_heads, c.head_dim, c.rotary_dim,
        &session.control_->mtp_position, c.rope_theta, c.rms_eps, stream_,
        session.vision_input_.rope(), yarn, session.mtp_.kv_blocks);
    return true;
  }
  if (PrepareAttention(
          s_.qg, 2 * c.AttentionKvDim(), nullptr, l.attn_k_norm.f32(), nullptr,
          nullptr, session.mtp_.k_cache, session.mtp_.v_cache, rows, 0,
          c.num_kv_heads, c.head_dim, c.rotary_dim,
          &session.control_->mtp_position, c.rope_theta, c.rms_eps, stream_,
          session.vision_input_.rope(), false, yarn, session.mtp_.kv_blocks))
    return true;
  AssignError(error_msg, "MTP cache projection failed");
  return false;
}

bool Executor::PlainDensePrefill(const DeviceTensor& w, const float* x,
                                 float* out, std::uint32_t rows,
                                 std::string* error) const {
  if ((w.type != GgmlType::kF16 && w.type != GgmlType::kBF16 &&
       w.type != GgmlType::kF32) ||
      w.cols > half_input_cols_ || w.cols % 32 != 0) {
    AssignError(error, "unsupported plain half prefill geometry/type");
    return false;
  }
  const auto piece = std::max<std::uint32_t>(1, options_.max_batch);
  for (std::uint32_t first = 0; first < rows; first += piece) {
    const auto count = std::min(piece, rows - first);
    PrepareHalfInput(x + std::size_t{first} * w.cols, count, w.cols);
    if (!PlainPrefillGemm(w.data, SmallType(w.type),
                          static_cast<const __half*>(s_.x_half),
                          out + std::size_t{first} * w.rows, count, w.rows,
                          w.cols, stream_)) {
      AssignError(error, "plain half prefill GEMM failed");
      return false;
    }
  }
  if (diagnostics_active_)
    cycle_diagnostics_.prefill_plain_half_rows += rows;
  return true;
}

bool Executor::Dense(const DeviceTensor& w, const float* x, float* out,
                     std::uint32_t n_tokens, std::string* error_msg) const {
  if (prefill_phase && w.exl3 != nullptr)
    return Exl3DensePrefill({&w}, {out}, x, n_tokens, error_msg);
  if (diagnostics_active_ && prefill_phase && model_->output().exl3 != nullptr)
    cycle_diagnostics_.prefill_plain_rows += n_tokens;
  // Prefill plain operands use the accepted FP16 activation boundary
  // at every width. Decode/verify retain the established FP32-input path.
  // An EXL3 model's plain Q8_0 weights (its mixer copies) take the Q8_0
  // routes below.
  if (w.exl3 != nullptr ||
      (model_->output().exl3 != nullptr && w.type != GgmlType::kQ8_0)) {
    try {
      if (prefill_phase && w.exl3 == nullptr &&
          (w.type == GgmlType::kF16 || w.type == GgmlType::kBF16 ||
           w.type == GgmlType::kF32))
        return PlainDensePrefill(w, x, out, n_tokens, error_msg);
      const auto piece = std::max<std::uint32_t>(1, options_.max_batch);
      const DeviceTensor* const consumer[] = {&w};
      const auto* rotated =
          w.exl3 != nullptr && !prefill_phase && n_tokens <= piece
              ? TakeExl3Rotated(x, n_tokens, consumer)
              : nullptr;
      for (std::uint32_t first = 0; first < n_tokens; first += piece) {
        const auto count = std::min(piece, n_tokens - first);
        Exl3Projection(w, x + std::size_t{first} * w.cols,
                       out + std::size_t{first} * w.rows, count, exl3_, stream_,
                       !prefill_phase, rotated);
      }
      return true;
    } catch (const std::exception& error) {
      AssignError(error_msg, error.what());
      return false;
    }
  }
  if (w.type == GgmlType::kQ8_0) {
    if (!MatrixRows(n_tokens)) {
      Q8Input q;
      return Quantize(x, n_tokens, w.cols, &q, error_msg, true) &&
             Dense(w, q, out, error_msg);
    }
    // Wide batches: the F16 route for the shapes it wins (F16 rows in
    // s_.x_half, often left there by a producer), else activations
    // quantized per 32-wide block into the tiled layout and the int8 WMMA
    // GEMM. Both staging buffers hold max_batch rows; a wider call (the
    // draft block folds its streams into rows) runs in pieces.
    const bool f16 = DenseF16Route(w, n_tokens);
    const std::uint32_t piece = static_cast<std::uint32_t>(options_.max_batch);
    for (std::uint32_t r0 = 0; r0 < n_tokens; r0 += piece) {
      const std::uint32_t rows = std::min(piece, n_tokens - r0);
      // The staging buffer may already hold this input (a producer wrote
      // it, or the previous projection read the same rows).
      const float* src = x + static_cast<std::size_t>(r0) * w.cols;
      if (f16) {
        PrepareHalfInput(src, rows, w.cols);
        if (!DenseF16Gemm(w.data, static_cast<const __half*>(s_.x_half),
                          out + static_cast<std::size_t>(r0) * w.rows, rows,
                          w.rows, w.cols, stream_)) {
          AssignError(error_msg, "dense F16 GEMM failed");
          return false;
        }
        continue;
      }
      if (!(q8t_src_ == src && q8t_rows_ == rows && q8t_cols_ == w.cols)) {
        QuantizeQ8Tiled(src, s_.x_q8t, rows, w.cols, stream_);
        q8t_src_ = src;
        q8t_rows_ = rows;
        q8t_cols_ = w.cols;
      }
      if (!W8A8Gemm(w.data, s_.x_q8t,
                    out + static_cast<std::size_t>(r0) * w.rows, rows, w.rows,
                    w.cols, stream_)) {
        AssignError(error_msg, "W8A8 GEMM failed");
        return false;
      }
    }
    return true;
  }
  if (!MatrixRows(n_tokens)) {
    SmallGemm(w.data, SmallType(w.type), x, out, n_tokens, w.rows, w.cols,
              stream_);
    return true;
  }
  // Wide batches of the unquantized projections (router, alpha/beta,
  // indexer): out[t][m] = sum_k w[m][k] x[t][k].
  const int m = static_cast<int>(w.rows);
  const int k = static_cast<int>(w.cols);
  const int n = static_cast<int>(n_tokens);
  if (w.type == GgmlType::kF32) {
    const float alpha = 1.0F;
    const float beta = 0.0F;
    if (hipblasSgemm(blas_, HIPBLAS_OP_T, HIPBLAS_OP_N, m, n, k, &alpha,
                     static_cast<const float*>(w.data), k, x, k, &beta, out,
                     m) != HIPBLAS_STATUS_SUCCESS) {
      AssignError(error_msg, "hipBLAS GEMM failed");
      return false;
    }
    return true;
  }
  const bool bf16 = w.type == GgmlType::kBF16;
  const hipDataType type = bf16 ? HIP_R_16BF : HIP_R_16F;
  if (!(half_src_ == x && half_rows_ == n_tokens && half_cols_ == w.cols &&
        half_bf16_ == bf16)) {
    NarrowActivations(x, s_.x_half, bf16, static_cast<std::size_t>(n) * k,
                      stream_);
    half_src_ = x;
    half_rows_ = n_tokens;
    half_cols_ = w.cols;
    half_bf16_ = bf16;
  }
  return blaslt_->Gemm(w.data, s_.x_half, out, type, m, n, k, error_msg);
}

void Executor::RoutedHints(const DeviceTensor& w,
                           std::uint32_t n_tokens) const {
  // The column grid is bounded by the largest expert bucket and the tile
  // width fitted to the whole distribution (see RouteHints); the fallback
  // bound is the token count with a tile near twice the mean bucket.
  if (routed_max_rows_ > 0) {
    qfn_mmq_set_routed_max_expert_rows(static_cast<int>(routed_max_rows_));
    qfn_mmq_set_routed_tile_cols(routed_tile_cols_);
    return;
  }
  qfn_mmq_set_routed_max_expert_rows(static_cast<int>(n_tokens));
  qfn_mmq_set_routed_tile_cols(
      RoutedTileCols(n_tokens, config().num_experts_used, w.experts));
}

bool Executor::RouteHints(std::uint32_t n_tokens,
                          std::string* error_msg) const {
  // Every column tile past an expert's bucket still costs a dispatch and a
  // full shared-memory reservation, so the grid is cut to the real largest
  // bucket. Counts are downloaded before the shared expert; wait only for
  // that download while the shared expert continues (on the side stream in
  // overlapped passes, joined by JoinSide).
  const Config& c = config();
  routed_max_rows_ = 0;
  routed_n_tiles_ = 0;
  routed_64_tiles_ = 0;
  routed_pair_tiles_ = 0;
  routed_pair_offset_ = 0;
  if (!ExpertMatrixRows(n_tokens)) {
    return true;
  }
  if (!Check(hipEventSynchronize(counts_ready_), "expert counts", error_msg)) {
    return false;
  }
  // The F16 expert GEMM launches one block per (expert, row tile of its
  // 16-padded bucket): the map is built here and uploaded ahead of the
  // launches on the same stream. Tile heights are multiples of 16, so a
  // bucket's tile count follows from its unpadded rows.
  std::uint32_t max_rows = 0;
  std::uint32_t n_tiles = 0;
  routed_tile_rows_ = RoutedTileRows(
      static_cast<std::size_t>(n_tokens) * c.num_experts_used, c);
  for (std::uint32_t e = 0; e < c.num_experts; ++e) {
    max_rows = std::max(max_rows, counts_host_[e]);
  }
  routed_max_rows_ = std::max<std::uint32_t>(1, max_rows);
  const auto append = [&](std::uint32_t tile_rows) {
    for (std::uint32_t e = 0; e < c.num_experts; ++e) {
      for (std::uint32_t j = 0;
           j < (counts_host_[e] + tile_rows - 1) / tile_rows; ++j) {
        tiles_host_[n_tiles++] = static_cast<std::int32_t>(e | (j << 16));
      }
    }
  };
  if (WidePrefill(n_tokens)) {
    // Prefill batches: 64-token tiles for the down projection, then 128-token
    // tiles for the paired gate/up projections.
    append(64);
    routed_64_tiles_ = n_tiles;
    routed_pair_offset_ = n_tiles;
    append(128);
    routed_pair_tiles_ = n_tiles - routed_pair_offset_;
  } else {
    append(routed_tile_rows_);
    routed_n_tiles_ = n_tiles;
  }
  routed_tile_cols_ = qfn_mmq_routed_tile_cols_for_counts(
      counts_host_, static_cast<int>(c.num_experts));
  return n_tiles == 0 || Check(hipMemcpyAsync(s_.routed_tiles, tiles_host_,
                                              n_tiles * sizeof(std::int32_t),
                                              hipMemcpyHostToDevice, stream_),
                               "routed tile map upload", error_msg);
}

bool Executor::Experts(const DeviceTensor& w, const float* x,
                       const std::int32_t* ids, float* out,
                       std::uint32_t n_rows, std::uint32_t n_used,
                       std::uint32_t n_tokens, std::string* error_msg) const {
  const int M = static_cast<int>(w.rows);
  const int K = static_cast<int>(w.cols);
  const int E = static_cast<int>(w.experts);
  const int T = static_cast<int>(n_rows);
  const int U = static_cast<int>(n_used);
  // The vector entries loop over column chunks, so the decode-time down
  // projection (top-k rows, one expert each) stays on them as well.
  // Verification keeps the same quantization and reduction as single-token
  // decoding even when top-k expansion produces more than 32 slot rows.
  const bool tiled =
      (prefill_phase || n_rows > 4 * kVecBatch) && MatrixRows(n_tokens);
  if (tiled) {
    RoutedHints(w, n_tokens);
  }
  int rc = -1;
  if (!tiled) {
    rc = qfn_mmq_moe_vec(static_cast<int>(w.type), w.data, x, ids, out, M, K, T,
                         E, U, stream_);
  } else {
    switch (w.type) {
      case GgmlType::kQ4_K:
        rc = qfn_mmq_q4_K_moe_raw(w.data, x, ids, out, M, K, T, E, U, stream_);
        break;
      case GgmlType::kQ5_K:
        rc = qfn_mmq_q5_K_moe_raw(w.data, x, ids, out, M, K, T, E, U, stream_);
        break;
      case GgmlType::kQ5_1:
        rc = qfn_mmq_q5_1_moe_raw(w.data, x, ids, out, M, K, T, E, U, stream_);
        break;
      case GgmlType::kQ8_0:
        rc = qfn_mmq_q8_0_moe_raw(w.data, x, ids, out, M, K, T, E, U, stream_);
        break;
      default:
        break;
    }
  }
  if (rc != 0) {
    AssignError(error_msg, "expert GEMM failed");
    return false;
  }
  return true;
}

bool Executor::GatedExperts(const DeviceTensor& a, const DeviceTensor& b,
                            const float* x, const std::int32_t* ids, float* out,
                            std::uint32_t n_tokens, std::uint32_t n_used,
                            std::string* error_msg) const {
  const bool same_shape = a.type == b.type && a.rows == b.rows &&
                          a.cols == b.cols && a.experts == b.experts;
  if (!MatrixRows(n_tokens) && n_used <= 32 && same_shape &&
      (a.type == GgmlType::kQ4_K || a.type == GgmlType::kQ5_K ||
       a.type == GgmlType::kQ8_0)) {
    // The shared expert quantized the same rows (or their producer did).
    const void* ready = ReadyQ8(x, n_tokens, a.cols);
    if ((ready != nullptr
             ? qfn_mmq_moe_gated_vec_preq(
                   static_cast<int>(a.type), a.data, b.data, ready, ids, out,
                   static_cast<int>(a.rows), static_cast<int>(a.cols),
                   static_cast<int>(n_tokens), static_cast<int>(a.experts),
                   static_cast<int>(n_used), stream_)
             : qfn_mmq_moe_gated_vec(
                   static_cast<int>(a.type), a.data, b.data, x, ids, out,
                   static_cast<int>(a.rows), static_cast<int>(a.cols),
                   static_cast<int>(n_tokens), static_cast<int>(a.experts),
                   static_cast<int>(n_used), stream_)) != 0) {
      AssignError(error_msg, "gated expert vector projection failed");
      return false;
    }
    return true;
  }
  if (!ExpertMatrixRows(n_tokens) && same_shape) {
    if (qfn_mmq_moe_vec(static_cast<int>(a.type), a.data, x, ids, out,
                        static_cast<int>(a.rows), static_cast<int>(a.cols),
                        static_cast<int>(n_tokens), static_cast<int>(a.experts),
                        static_cast<int>(n_used), stream_, b.data,
                        s_.up_e) != 0) {
      AssignError(error_msg, "expert vector pair GEMM failed");
      return false;
    }
  } else if (same_shape && a.type == GgmlType::kQ4_K) {
    // The wide Q4_K path shares its gather and tiled quantization as well.
    RoutedHints(a, n_tokens);
    if (qfn_mmq_q4_K_moe_pair_unique(
            a.data, b.data, x, ids, out, s_.up_e, static_cast<int>(a.rows),
            static_cast<int>(a.cols), static_cast<int>(n_tokens),
            static_cast<int>(a.experts), static_cast<int>(n_used),
            stream_) != 0) {
      AssignError(error_msg, "expert pair GEMM failed");
      return false;
    }
  } else if (!Experts(a, x, ids, out, n_tokens, n_used, n_tokens, error_msg) ||
             !Experts(b, x, ids, s_.up_e, n_tokens, n_used, n_tokens,
                      error_msg)) {
    return false;
  }
  Swiglu(out, s_.up_e, static_cast<std::size_t>(n_tokens) * n_used * a.rows,
         stream_);
  return true;
}

namespace {

/// A mixer's F32 inject weights, which its combine can apply (see Combine).
const float* InjectWeights(const DeviceMixer& m) {
  return !m.inject.empty() && m.inject.type == GgmlType::kF32 ? m.inject.f32()
                                                              : nullptr;
}

}  // namespace

void Executor::Combine(float* res, const float* gamma, std::uint32_t n_tokens,
                       const float* inject_w) const {
  const Config& c = config();
  inject_fused_ = nullptr;
  xn_deferred_ = {};  // s_.xn is rewritten (or left stale) below
  xn_q8t_norm_ = {};
  // An EXL3 prefill MoE output left to this combine: the four-lane kernel
  // below forms it; any other route gets it in s_.block_out first.
  Exl3PrefillMoeRows exl3_moe{};
  bool exl3_moe_fused = false;
  if (exl3_moe_pending_.layer != nullptr) {
    const PendingExl3Moe pending = exl3_moe_pending_;
    exl3_moe_pending_ = {};
    const auto& d = pending.layer->ffn_down_exps;
    const auto scratch = Exl3PrefillView();
    exl3_moe_fused = prefill_phase && pending.rows == n_tokens &&
                     c.hc_count == 4 && c.hidden_size % 128 == 0 &&
                     c.hidden_size <= 2560 && d.rows == c.hidden_size &&
                     !moe_pending_ && !res_deferred_ &&
                     !(wide_mixer_ && MatrixRows(n_tokens) && gamma != nullptr);
    if (exl3_moe_fused) {
      exl3_moe = {d.device_exl3,
                  s_.ids,
                  scratch.inverse,
                  scratch.out,
                  s_.weights,
                  s_.shexp_out,
                  s_.router + c.num_experts,
                  c.num_experts + 1,
                  c.num_experts_used,
                  d.rows};
    } else {
      Exl3PrefillCombine(*pending.layer, scratch, s_.ids, pending.rows,
                         c.num_experts_used, s_.weights, s_.shexp_out,
                         s_.router + c.num_experts, c.num_experts + 1,
                         s_.block_out, stream_);
    }
  }
  // Wide batches hand the next mixer an F16 norm for its epilogue and the
  // same norm quantized into the tiled Q8 layout for its W8A8 down
  // projection: half the bytes for the combine and the epilogue, and no
  // separate activation pass for the projection.
  xn_half_ = wide_mixer_ && MatrixRows(n_tokens) && gamma != nullptr;
  xn_q8_ready_ = {};  // s_.xn is rewritten (or left stale) below
  // The next mixer's inject pass reads only the F16 norm written here.
  const float* fused_inject = xn_half_ && n_tokens >= 96 ? inject_w : nullptr;
  const bool defer = defer_residual_;
  defer_residual_ = false;
  const bool deferred = res_deferred_;
  res_deferred_ = false;
  if (moe_pending_) {
    // The MoE epilogue was deferred to this combine (see Moe), and with it
    // the previous combine's residual update when that was deferred too.
    moe_pending_ = false;
    const auto* down = reinterpret_cast<const __half*>(s_.down_e);
    const float* fused_w = deferred ? fused_inject : nullptr;
    if (wide_mixer_ && MatrixRows(n_tokens) &&
        HcCombineMoeF16(
            res, down, s_.weights, s_.shexp_out, s_.router + c.num_experts,
            c.num_experts + 1, c.num_experts_used, s_.inject, inject_parts_,
            gamma, s_.xn_half, s_.xn_q8t, n_tokens, c.hidden_size, c.hc_count,
            c.rms_eps, stream_, deferred ? s_.block_out : nullptr,
            deferred ? s_.hc_defer_w : nullptr, s_.inject, fused_w)) {
      inject_fused_ = fused_w;
      return;
    }
    if (deferred) {
      HcApplyDeferred(res, s_.block_out, s_.hc_defer_w, n_tokens, c.hidden_size,
                      stream_);
    }
    MoeEpilogueVec4F16(down, s_.weights, s_.shexp_out,
                       s_.router + c.num_experts, c.num_experts + 1,
                       s_.block_out, n_tokens, c.num_experts_used,
                       c.hidden_size, stream_);
  }
  if (deferred) {
    // Only a fused MoE combine follows a deferring one; settle the update
    // before anything else reads the residual.
    HcApplyDeferred(res, s_.block_out, s_.hc_defer_w, n_tokens, c.hidden_size,
                    stream_);
  }
  if (xn_half_) {
    // The FFN combine that follows rereads this block output (42 MB at
    // 4,096 tokens) instead of this combine writing the residual and the
    // next reading it back (168 MB each).
    const bool defer_now = defer && c.hc_count == 4;
    res_deferred_ =
        HcCombineF16(res, s_.block_out, s_.inject, inject_parts_, gamma,
                     s_.xn_half, s_.xn_q8t, n_tokens, c.hidden_size, c.hc_count,
                     c.rms_eps, stream_, defer_now ? s_.hc_defer_w : nullptr,
                     s_.inject, defer_now ? fused_inject : nullptr);
    if (res_deferred_)
      inject_fused_ = fused_inject;
    return;
  }
  // A decode batch's next mixer quantizes xn for its down projection: the
  // combine writes those rows as well.
  // (Not for EXL3 prefill: every prefill chunk, tails included, must take
  // the same combine below.)
  if (!(prefill_phase && model_->output().exl3 != nullptr) &&
      !MatrixRows(n_tokens) && UnpaddedQ8(c.HcDim()) &&
      HcCombineQ8_1(res, s_.block_out, s_.inject, inject_parts_, gamma, s_.xn,
                    s_.xn_q8_1, n_tokens, c.hidden_size, c.hc_count, c.rms_eps,
                    stream_)) {
    xn_q8_ready_ = {s_.xn, s_.xn_q8_1, n_tokens, c.HcDim()};
    return;
  }
  // EXL3: prefill takes the four-lane combine at every chunk width (28 %
  // less time at 2,048 tokens); decode/verify keep the per-stream kernel.
  const bool exl3 = model_->output().exl3 != nullptr;
  // The next mixer's plain down projection reads this norm as F16: the
  // combine stores that copy too instead of a separate narrowing pass
  // rereading the F32 row.
  const bool half_norm = exl3 && prefill_phase &&
                         n_tokens <= options_.max_batch &&
                         c.HcDim() <= half_input_cols_;
  // Its F32 copy is then left out as well: the mixer recomputes it from the
  // residual and these scales.
  float* const xn_scale =
      half_norm && MatrixRows(n_tokens) ? s_.xn_scale : nullptr;
  // With Q8_0 mixer down weights the down projection is W8A8 and reads the
  // norm as tiled Q8 rows at every prefill width; no F16 copy is needed.
  const bool q8_norm = half_norm && hc_down_q8_;
  if (HcCombine(
          res, s_.block_out, s_.inject, inject_parts_, gamma, s_.xn, n_tokens,
          c.hidden_size, c.hc_count, c.rms_eps, stream_, exl3 && !prefill_phase,
          exl3 && prefill_phase,
          half_norm && !q8_norm ? static_cast<__half*>(s_.x_half) : nullptr,
          xn_scale, exl3_moe_fused ? &exl3_moe : nullptr,
          q8_norm ? s_.xn_q8t : nullptr)) {
    if (q8_norm) {
      xn_q8t_norm_ = {res, gamma, n_tokens};
      if (half_src_ == s_.xn)
        half_src_ = nullptr;
    } else {
      half_src_ = s_.xn;
      half_rows_ = n_tokens;
      half_cols_ = c.HcDim();
      half_bf16_ = false;
    }
    if (xn_scale != nullptr)
      xn_deferred_ = {res, gamma, n_tokens};
  } else if (half_src_ == s_.xn) {
    half_src_ = nullptr;  // the F32 norm was rewritten without its copy
  }
}

bool Executor::HcMix(const DeviceMixer& m, const float* res, bool normed,
                     float* mixed, float* inject, std::uint32_t n_tokens,
                     std::string* error_msg, bool half_only, bool no_f32,
                     std::span<const DeviceTensor* const> rotate_for) const {
  const Config& c = config();
  mixed_q8_ready_ = {};  // `mixed` is rewritten below
  exl3_rotated_ = {};
  // Without `normed` this mixer's grouped norm of res is computed here (F32);
  // otherwise the previous combine produced it, as F16 plus tiled Q8 on the
  // wide route or as F32 in s_.xn (with its Q8_1 rows on a decode batch).
  // A deferred norm feeds only the vectorized epilogue below (and the F16
  // down projection through the half cache); any other reader gets the F32
  // rows the combine left out, in its own expression.
  // The combine's tiled Q8 norm of this mixer's input, for its W8A8 down.
  const bool q8_down = normed && prefill_phase && !m.down_q8.empty() &&
                       xn_q8t_norm_.res == res &&
                       xn_q8t_norm_.gamma == m.norm.f32() &&
                       xn_q8t_norm_.rows == n_tokens;
  xn_q8t_norm_ = {};
  bool norm_from_res = false;
  if (normed && xn_deferred_.res != nullptr) {
    const DeferredNorm d = xn_deferred_;
    const bool other_inject = inject != nullptr && !m.inject.empty() &&
                              m.inject.type != GgmlType::kF32;
    const bool half_down =
        half_src_ == s_.xn && half_rows_ == n_tokens && half_cols_ == c.HcDim();
    norm_from_res = res == d.res && m.norm.f32() == d.gamma &&
                    n_tokens == d.rows && MatrixRows(n_tokens) &&
                    c.hc_count == 4 && c.hidden_size % 4 == 0 &&
                    !other_inject && (half_down || q8_down);
    if (!norm_from_res)
      HcNormFromRes(d.res, d.gamma, s_.xn_scale, s_.xn, d.rows, c.hidden_size,
                    stream_);
  }
  const DeferredNorm deferred_norm = xn_deferred_;
  xn_deferred_ = {};
  if (!normed) {
    xn_half_ = false;
    xn_q8_ready_ = {};
    if (half_src_ == s_.xn)
      half_src_ = nullptr;
    RmsNormRows(res, m.norm.f32(), s_.xn, n_tokens, c.HcDim(), c.hc_count,
                c.rms_eps, stream_);
  }
  const bool extras = n_tokens <= options_.max_batch &&
                      c.hidden_size <= model_->max_half_cols();
  const bool fused_projection =
      xn_half_ && n_tokens >= 96 && extras && c.hc_count == 4 &&
      c.hidden_size == 2560 && c.hc_low_rank == 320 &&
      m.up.type == GgmlType::kQ8_0 && m.up.rows == c.HcDim() &&
      m.up.cols == c.hc_low_rank;
  // EXL3 decode/verify reads the Q8_0 copy the prefill W8A8 route uses:
  // 3.4 instead of 6.6 MB per mixer and token step.
  const bool q8_decode_down = !q8_down && model_->output().exl3 != nullptr &&
                              !xn_half_ && !MatrixRows(n_tokens) &&
                              !m.down_q8.empty();
  const bool fused_down =
      !q8_down && !q8_decode_down && model_->output().exl3 != nullptr &&
      !xn_half_ && !MatrixRows(n_tokens) && m.down.type == GgmlType::kF16 &&
      m.up.type == GgmlType::kF16 && m.down.rows == 320 &&
      m.down.cols == 10240 && c.hc_count == 4 && c.hc_low_rank == 320;
  // EXL3 prefill stores the HC gate as F16 (the Q8_0 up GEMM's F16-store
  // variant), halving its write and the vectorized epilogue's read.
  const bool half_gate =
      !fused_projection && model_->output().exl3 != nullptr && prefill_phase &&
      !xn_half_ && extras && c.hc_count == 4 && c.hidden_size % 4 == 0 &&
      m.up.type == GgmlType::kQ8_0 && m.up.rows % 8 == 0 &&
      DenseF16Route(m.up, n_tokens) && m.up.cols <= half_input_cols_;
  if (fused_projection) {
    // The up projection writes x_half; keep its input in the gate buffer.
    if (!HcDownF16Gemm(m.down.data, s_.xn_q8t,
                       reinterpret_cast<__half*>(s_.hc_gate), n_tokens,
                       stream_)) {
      AssignError(error_msg, "fused HC down projection failed");
      return false;
    }
  } else {
    if (xn_half_ || q8_down) {
      const DeviceTensor& down = q8_down ? m.down_q8 : m.down;
      if (!W8A8Gemm(down.data, s_.xn_q8t, s_.lo, n_tokens, down.rows, down.cols,
                    stream_)) {
        AssignError(error_msg, "W8A8 mixer down projection failed");
        return false;
      }
    } else if (q8_decode_down) {
      if (!Dense(m.down_q8, s_.xn, s_.lo, n_tokens, error_msg))
        return false;
    } else if (fused_down) {
      HcDownSiluF16(m.down.data, s_.xn, s_.lo, n_tokens, stream_);
    } else if (!Dense(m.down, s_.xn, s_.lo, n_tokens, error_msg)) {
      return false;
    }
    const float scale = 1.0F / static_cast<float>(c.hc_count);
    const std::uint32_t lo_q8_cols = static_cast<std::uint32_t>(
        qfn_mmq_q8_1_bytes(1, static_cast<int>(c.hc_low_rank)) / 36 * 32);
    if (!MatrixRows(n_tokens) && m.up.type == GgmlType::kQ8_0 &&
        m.up.cols == c.hc_low_rank &&
        SiluScaleQ8_1(s_.lo, scale, n_tokens, c.hc_low_rank, lo_q8_cols,
                      s_.lo_q8_1, stream_)) {
      const Q8Input lo{s_.lo, s_.lo_q8_1, n_tokens, c.hc_low_rank};
      if (!Dense(m.up, lo, s_.hc_gate, error_msg)) {
        return false;
      }
    } else {
      if (!fused_down)
        SiluScale(s_.lo, scale,
                  static_cast<std::size_t>(n_tokens) * c.hc_low_rank, stream_);
      if (half_gate) {
        if (diagnostics_active_)
          cycle_diagnostics_.prefill_plain_rows += n_tokens;
        PrepareHalfInput(s_.lo, n_tokens, m.up.cols);
        if (!HcUpF16Gemm(m.up.data, static_cast<const __half*>(s_.x_half),
                         reinterpret_cast<__half*>(s_.hc_gate), n_tokens,
                         m.up.rows, m.up.cols, stream_)) {
          AssignError(error_msg, "HC up F16 GEMM failed");
          return false;
        }
      } else if (!Dense(m.up, s_.lo, s_.hc_gate, n_tokens, error_msg)) {
        return false;
      }
    }
  }
  const bool fused_inject =
      inject != nullptr && !m.inject.empty() && m.inject.type == GgmlType::kF32;
  // The previous combine already wrote these partials from the same norm.
  const bool inject_done = fused_inject && inject == s_.inject &&
                           inject_fused_ != nullptr &&
                           inject_fused_ == m.inject.f32();
  inject_fused_ = nullptr;
  // A separate inject projection can overwrite the cached half input. Keep
  // the full mixed row available for the following projection in that case.
  half_only =
      half_only && (inject == nullptr || m.inject.empty() || fused_inject);
  const float* xn = s_.xn;
  const bool vectorized =
      xn_half_ ||
      (MatrixRows(n_tokens) && c.hc_count == 4 && c.hidden_size % 4 == 0);
  // `mixed` is being rewritten: whatever the input caches held of it is
  // stale. The wide F16 route also emits the F16 and tiled Q8 copies the
  // projections that follow read.
  q8t_src_ = nullptr;
  half_src_ = nullptr;
  if (xn_half_) {
    if (fused_projection) {
      if (!HcMixF16Gemm(m.up.data, reinterpret_cast<const __half*>(s_.hc_gate),
                        s_.xn_half,
                        fused_inject && !inject_done ? m.inject.f32() : nullptr,
                        half_only || no_f32 ? nullptr : mixed,
                        static_cast<__half*>(s_.x_half),
                        half_only ? nullptr : s_.x_q8t, inject, n_tokens,
                        c.hidden_size, c.hc_low_rank, stream_)) {
        AssignError(error_msg, "fused HC projection failed");
        return false;
      }
    } else {
      HcMixEpilogueVec4F16(s_.xn_half, s_.hc_gate,
                           fused_inject ? m.inject.f32() : nullptr, mixed,
                           extras ? static_cast<__half*>(s_.x_half) : nullptr,
                           extras ? s_.x_q8t : nullptr, inject, n_tokens,
                           c.hidden_size, stream_);
    }
    if (extras) {
      half_src_ = mixed;
      half_rows_ = n_tokens;
      half_cols_ = c.hidden_size;
      half_bf16_ = false;
      if (!half_only || !fused_projection) {
        q8t_src_ = mixed;
        q8t_rows_ = n_tokens;
        q8t_cols_ = c.hidden_size;
      }
    }
  } else if (vectorized) {
    // EXL3 prefill: the plain F16 projections of `mixed` (router, alpha/beta)
    // read these F16 rows instead of a separate narrowing pass.
    auto* mixed_half = model_->output().exl3 != nullptr && prefill_phase &&
                               extras && c.hidden_size <= half_input_cols_
                           ? static_cast<__half*>(s_.x_half)
                           : nullptr;
    bool half_written = mixed_half != nullptr;
    const auto* gate_half =
        half_gate ? reinterpret_cast<const __half*>(s_.hc_gate) : nullptr;
    if (norm_from_res)
      HcMixEpilogueVec4FromRes(
          deferred_norm.res, deferred_norm.gamma, s_.xn_scale, s_.hc_gate,
          fused_inject ? m.inject.f32() : nullptr, mixed, inject, n_tokens,
          c.hidden_size, stream_, mixed_half, gate_half);
    else
      half_written = HcMixEpilogueVec4(
          xn, s_.hc_gate, fused_inject ? m.inject.f32() : nullptr, mixed,
          inject, n_tokens, c.hidden_size, c.hc_count, stream_, mixed_half,
          gate_half);
    if (half_written) {
      half_src_ = mixed;
      half_rows_ = n_tokens;
      half_cols_ = c.hidden_size;
      half_bf16_ = false;
    }
  } else {
    // A decode batch's projections of `mixed` share its Q8_1 rows, which
    // the epilogue writes as it stores them.
    const bool q8 = !MatrixRows(n_tokens) && UnpaddedQ8(c.hidden_size);
    // EXL3 decode: the epilogue also rotates mixed for the projections that
    // read it next (GDN qkv/z, attention q/k/v or the shared MLP), removing
    // their separate rotation launch; the same rounded FP16 values.
    const auto slabs = Exl3Slabs(rotate_for, n_tokens);
    if (HcMixEpilogue(xn, s_.hc_gate, fused_inject ? m.inject.f32() : nullptr,
                      mixed, inject, n_tokens, c.hidden_size, c.hc_count,
                      stream_, q8 ? s_.mixed_q8_1 : nullptr, slabs))
      NoteExl3Rotated(slabs, mixed, n_tokens, rotate_for);
    if (q8) {
      mixed_q8_ready_ = {mixed, s_.mixed_q8_1, n_tokens, c.hidden_size};
    }
  }
  inject_parts_ = fused_inject ? (vectorized ? HcInjectPartsVec4(c.hidden_size)
                                             : HcInjectParts(c.hidden_size))
                               : 1;
  if (inject != nullptr && !m.inject.empty() && !fused_inject) {
    // A quantized inject projection (the draft block's) reads an F32 norm;
    // on the wide route the combine only produced F16, so norm again.
    if (xn_half_) {
      xn_q8_ready_ = {};
      if (half_src_ == s_.xn)
        half_src_ = nullptr;
      RmsNormRows(res, m.norm.f32(), s_.xn, n_tokens, c.HcDim(), c.hc_count,
                  c.rms_eps, stream_);
    }
    if (!Dense(m.inject, xn, inject, n_tokens, error_msg)) {
      return false;
    }
  }
  return true;
}

bool Executor::PleFetch(Session& session, std::span<const std::int32_t> tokens,
                        bool speculative, std::string* error_msg) const {
  if (ple_pending_ && !WaitPle(error_msg)) {
    return false;
  }
  const Config& c = config();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  if ((prefetch_.pending || prefetch_.ready) && !speculative &&
      session.ngram_ == prefetch_.before) {
    // A cache checkpoint or another step budget can end this chunk before
    // or after the hinted one. Rows are per token, so a common prefix still
    // serves every token both cover.
    const auto shared = static_cast<std::size_t>(
        std::ranges::mismatch(tokens, prefetch_.tokens).in1 - tokens.begin());
    if (shared == tokens.size() || shared == prefetch_.tokens.size()) {
      prefetch_.claimed = shared;
      ple_pending_ = true;
      if (shared == prefetch_.tokens.size()) {
        session.ngram_ = prefetch_.after;
      } else {
        HashNgramRows(
            c, session.ngram_, tokens,
            std::span<std::uint32_t>(host_rows_.data(), n * c.ple_heads));
      }
      if (shared == tokens.size())
        return true;
      // Only one table read runs at a time: finish the prefetch, which had a
      // whole chunk to complete, then read the rows past it.
      FinishPrefetch();
      const auto rest = tokens.subspan(shared);
      const auto rows = rest.size() * c.ple_heads;
      HashNgramRows(c, session.ngram_, rest,
                    std::span<std::uint32_t>(host_rows_.data(), rows));
      const NgramGatherOwner owner{rows, &session.ngram_counters_};
      prefetch_.tail = ngram_->StartRead(
          std::span<const std::uint32_t>(host_rows_.data(), rows),
          std::span<float>(host_emb_ + shared * c.PleEmbeddingDim(),
                           rest.size() * c.PleEmbeddingDim()),
          std::span(&owner, 1));
      if (!prefetch_.tail) {
        prefetch_.claimed = 0;
        prefetch_.ready = false;
        ple_pending_ = false;
        AssignError(error_msg, "n-gram table read could not start");
      }
      return prefetch_.tail;
    }
  }
  FinishPrefetch();
  // Only proper prefixes need snapshots; the full batch keeps its live state.
  if (speculative) {
    for (std::uint32_t i = 0; i < n; ++i) {
      HashNgramRows(c, session.ngram_, tokens.subspan(i, 1),
                    std::span<std::uint32_t>(
                        host_rows_.data() + i * c.ple_heads, c.ple_heads));
      if (i + 1 < n) {
        session.ngram_snapshots_[i] = session.ngram_;
      }
    }
  } else {
    HashNgramRows(c, session.ngram_, tokens,
                  std::span<std::uint32_t>(host_rows_.data(), n * c.ple_heads));
  }
  const NgramGatherOwner owner{n * c.ple_heads, &session.ngram_counters_};
  ple_pending_ = ngram_->StartRead(
      std::span<const std::uint32_t>(host_rows_.data(), n * c.ple_heads),
      std::span<float>(host_emb_,
                       static_cast<std::size_t>(n) * c.PleEmbeddingDim()),
      std::span(&owner, 1));
  if (!ple_pending_) {
    AssignError(error_msg, "n-gram table read could not start");
  }
  return ple_pending_;
}

bool Executor::WaitPle(std::string* error_msg) const {
  bool ok = ple_pending_;
  if (ok && prefetch_.claimed != 0) {
    FinishPrefetch();
    ok = prefetch_.ready;
    if (ok) {
      std::copy_n(prefetch_.rows.data(),
                  prefetch_.claimed * config().PleEmbeddingDim(), host_emb_);
    }
    if (prefetch_.tail)
      ok = ngram_->WaitRead() && ok;
    prefetch_.ready = false;
    prefetch_.claimed = 0;
    prefetch_.tail = false;
  } else if (ok) {
    ok = ngram_->WaitRead();
  }
  ple_pending_ = false;
  if (!ok) {
    AssignError(error_msg, "n-gram table read failed");
  }
  return ok;
}

void Executor::PrefetchPle(const Session& session,
                           std::span<const std::int32_t> next) const {
  const Config& c = config();
  if (ple_pending_ || prefetch_.pending || next.empty() ||
      next.size() > options_.max_batch) {
    return;
  }
  // Hashed from the history after this batch, which PleFetch already
  // advanced; the session keeps it until the next batch claims the rows.
  prefetch_.ready = false;
  prefetch_.before = session.ngram_;
  prefetch_.after = session.ngram_;
  const std::span<std::uint32_t> rows(host_rows_.data(),
                                      next.size() * c.ple_heads);
  HashNgramRows(c, prefetch_.after, next, rows);
  prefetch_.tokens.assign(next.begin(), next.end());
  const std::size_t count = next.size() * c.PleEmbeddingDim();
  if (prefetch_.rows.size() < count) {
    prefetch_.rows.resize(count);
  }
  const NgramGatherOwner owner{rows.size(), &session.ngram_counters_};
  prefetch_.pending =
      ngram_->StartRead(rows, std::span<float>(prefetch_.rows.data(), count),
                        std::span(&owner, 1));
}

void Executor::FinishPrefetch() const {
  if (prefetch_.pending) {
    prefetch_.ready = ngram_->WaitRead();
    prefetch_.pending = false;
  }
}

bool Executor::Ple(const DeviceLayer& l, Session& session, std::uint32_t n,
                   float* res, bool speculative, std::string* error_msg,
                   bool embeddings_ready, PrefillCheckpoint* checkpoint) const {
  const Config& c = config();
  if (!embeddings_ready && !WaitPle(error_msg)) {
    return false;
  }
  if (!embeddings_ready && !prefetch_next_.empty()) {
    PrefetchPle(session, prefetch_next_);
    prefetch_next_ = {};
  }
  const std::uint32_t hc_dim = c.HcDim();
  Q8Input emb;
  if (!Quantize(s_.ple_emb, n, c.PleEmbeddingDim(), &emb, error_msg) ||
      !Dense(l.ple_key, emb, s_.ple_key, error_msg) ||
      !Dense(l.ple_value, emb, s_.ple_value, error_msg)) {
    return false;
  }
  RmsNormRows(s_.ple_key, l.ple_norm_key.f32(), s_.ple_key, n, hc_dim,
              c.hc_count, c.rms_eps, stream_);
  RmsNormRows(res, l.ple_norm_query.f32(), s_.ple_query, n, hc_dim, c.hc_count,
              c.rms_eps, stream_);
  PleGate(s_.ple_key, s_.ple_query, s_.ple_value, s_.ple_gated, n,
          c.hidden_size, c.hc_count, stream_);
  RmsNormRows(s_.ple_gated, l.ple_norm_conv.f32(), s_.ple_norm, n, hc_dim,
              c.hc_count, c.rms_eps, stream_);
  if (checkpoint)
    HistoryPrefix(s_.ple_norm, hc_dim, session.ple_history_,
                  checkpoint->state->ple_history_, checkpoint->tokens, hc_dim,
                  c.PleConvHistory(), stream_);
  PleConv(s_.ple_norm, l.ple_conv1d.f32(), session.ple_history_,
          s_.ple_history_scratch, s_.ple_conv,
          speculative ? session.ple_snapshots_ : RollbackRows{}, n, hc_dim,
          c.ple_conv_kernel, c.ple_ngram_size, stream_);
  PleInject(res, s_.ple_gated, s_.ple_conv,
            static_cast<std::size_t>(n) * hc_dim, stream_);
  return true;
}

bool Executor::LinearAttention(const DeviceLayer& l, Session::LinearState& s,
                               const float* x, float* out,
                               std::uint32_t n_tokens, bool speculative,
                               std::string* error_msg, bool projections_ready,
                               bool project_output,
                               GdnCheckpoint checkpoint) const {
  const Config& c = config();
  const std::uint32_t channels = c.SsmConvChannels();
  const float* qkv = s_.qkv;
  const float* z = s_.z;
  std::uint32_t qkv_stride = channels;
  std::uint32_t z_stride = c.SsmValueDim();
  bool convolved = false;
  hipEvent_t z_ready = nullptr;
  // Until GatedDeltaNet queues its wait for the Z rows, every exit orders
  // the main stream after them: the side stream still writes s_.qkvz.
  struct ZJoin {
    hipStream_t stream;
    hipEvent_t event = nullptr;
    ~ZJoin() {
      if (event != nullptr)
        (void)hipStreamWaitEvent(stream, event, 0);
    }
  } z_join{stream_};
  if (!l.ssm_in.empty()) {
    // Wide prefill convolves QKV in the projection's LDS tile. The raw
    // boundary rows remain available for the rolling history update.
    if (!projections_ready && !speculative && WidePrefill(n_tokens) &&
        n_tokens <= options_.max_batch && DenseF16Route(l.ssm_in, n_tokens)) {
      PrepareHalfInput(x, n_tokens, l.ssm_in.cols);
      // The Z rows feed only the output norm: the side stream computes them
      // beside the convolution boundary, the alpha/beta GEMM and the
      // recurrence; the epilogue waits for them before it reads z. With the
      // shared-expert overlap on ROCm 10.1: pp512 1,287 -> 1,330 tok/s,
      // pp2048 and pp4096 within 1 %.
      convolved = DenseF16SsmGemm(
          l.ssm_in.data, static_cast<const __half*>(s_.x_half),
          l.ssm_conv1d.f32(), s.conv_state, s_.qkvz, s_.conv_scratch, n_tokens,
          l.ssm_in.rows, l.ssm_in.cols, channels, c.ssm_conv_kernel, stream_,
          checkpoint.tokens, side_stream_, side_start_);
      if (convolved) {
        if (!Check(hipEventRecord(side_done_, side_stream_), "side stream join",
                   error_msg)) {
          (void)hipStreamSynchronize(side_stream_);
          return false;
        }
        z_ready = side_done_;
        z_join.event = side_done_;
      }
    }
    if (!projections_ready && !convolved &&
        !Dense(l.ssm_in, x, s_.qkvz, n_tokens, error_msg)) {
      return false;
    }
    qkv = s_.qkvz;
    z = s_.qkvz + channels;
    qkv_stride = z_stride = l.ssm_in.rows;
  } else if (!projections_ready && l.ssm_qkv.exl3 != nullptr) {
    if (!Exl3Group({&l.ssm_qkv, &l.ssm_gate}, {s_.qkv, s_.z}, x, n_tokens,
                   error_msg))
      return false;
  } else if (!projections_ready) {
    Q8Input xq;
    if (!Quantize(x, n_tokens, c.hidden_size, &xq, error_msg) ||
        !Dense(l.ssm_qkv, xq, s_.qkv, error_msg) ||
        !Dense(l.ssm_gate, xq, s_.z, error_msg)) {
      return false;
    }
  }
  if (!projections_ready &&
      !Dense(l.ssm_alpha_beta, x, s_.alpha_beta, n_tokens, error_msg)) {
    return false;
  }
  // Keep the same activation precision across prefill chunk boundaries.
  // The F16 epilogue reuses the F32 output allocation. Switching this
  // projection to Q8 for a short tail changes every subsequent layer.
  const bool tiled = MatrixRows(n_tokens) &&
                     l.ssm_out.type == GgmlType::kQ8_0 &&
                     n_tokens <= options_.max_batch;
  const bool half_output =
      tiled && l.ssm_out.rows == 2560 && l.ssm_out.cols == 6144;
  auto* out_half =
      half_output ? reinterpret_cast<__half*>(s_.gdn_out) : nullptr;
  // Shift successive rows off the 12 KiB pitch while retaining every half.
  // The existing F32 scratch has room for these slightly padded F16 rows.
  const std::uint32_t half_stride =
      half_output && n_tokens >= 1024 ? l.ssm_out.cols + 32 : 0;
  // A decode batch's output projection reads the epilogue's Q8_1 rows.
  const std::uint32_t value_dim = c.SsmValueDim();
  const bool out_q8_1 = project_output && !tiled && !MatrixRows(n_tokens) &&
                        c.ssm_head_dim == 128 &&
                        l.ssm_out.type == GgmlType::kQ8_0 &&
                        l.ssm_out.cols == value_dim && UnpaddedQ8(value_dim);
  // EXL3 decode: the epilogue also rotates its rows for ssm_out.
  const DeviceTensor* const out_consumer[] = {&l.ssm_out};
  auto out_slabs = project_output && !tiled ? Exl3Slabs(out_consumer, n_tokens)
                                            : Exl3InputSlabs{};
  if (!GatedDeltaNet(qkv, qkv_stride, z, z_stride, s_.alpha_beta,
                     l.ssm_conv1d.f32(), l.ssm_a.f32(), l.ssm_dt.f32(),
                     l.ssm_norm.f32(), s.conv_state, s_.conv_scratch, s_.qn,
                     s_.kn, s_.gdn_raw, s.state, s_.gdn_out,
                     tiled && !half_output ? s_.x_q8t : nullptr,
                     speculative ? s.state_snapshots : RollbackRows{},
                     speculative ? s.conv_snapshots : RollbackRows{}, n_tokens,
                     c.ssm_num_k_heads, c.ssm_num_v_heads, c.ssm_head_dim,
                     c.ssm_conv_kernel, MatrixRows(n_tokens) && !speculative,
                     convolved, c.rms_eps, stream_, out_half, checkpoint,
                     half_stride, out_q8_1 ? s_.gdn_q8_1 : nullptr, z_ready,
                     model_->output().exl3 != nullptr,
                     out_slabs.count != 0 ? &out_slabs : nullptr)) {
    AssignError(error_msg, "DeltaNet launch failed");
    return false;
  }
  NoteExl3Rotated(out_slabs, s_.gdn_out, n_tokens, out_consumer);
  z_join.event = nullptr;  // GatedDeltaNet waited for the Z rows
  if (!project_output) {
    return true;
  }
  if (half_output) {
    if (!DenseF16Gemm(l.ssm_out.data, out_half, out, n_tokens, l.ssm_out.rows,
                      l.ssm_out.cols, stream_, half_stride)) {
      AssignError(error_msg, "SSM output F16 GEMM failed");
      return false;
    }
    return true;
  }
  if (tiled) {
    q8t_src_ = nullptr;
    if (!W8A8Gemm(l.ssm_out.data, s_.x_q8t, out, n_tokens, l.ssm_out.rows,
                  l.ssm_out.cols, stream_)) {
      AssignError(error_msg, "W8A8 GEMM failed");
      return false;
    }
    return true;
  }
  if (out_q8_1) {
    return Dense(l.ssm_out,
                 Q8Input{s_.gdn_out, s_.gdn_q8_1, n_tokens, value_dim}, out,
                 error_msg);
  }
  return Dense(l.ssm_out, s_.gdn_out, out, n_tokens, error_msg);
}

bool Executor::Attention(const DeviceLayer& l, Session::AttentionState& s,
                         const float* x, float* out, std::uint32_t n_tokens,
                         const std::uint32_t* pos,
                         const std::uint32_t* first_block,
                         std::uint32_t start_pos, std::uint32_t pool_grid,
                         std::uint32_t max_context, bool sparse,
                         std::string* error_msg, bool last_only,
                         bool projections_ready, bool project_output,
                         std::uint32_t checkpoint_tokens) const {
  // Set below for this call's KvarnAttention only.
  attn_q_rotated_ = false;
  attn_gate_stride_ = 0;
  const Config& c = config();
  // The generic fallback consumes every query's selector mask. Only prune
  // masks when WMMA can honor the same last-query restriction; otherwise
  // uninitialized masks can produce NaNs and invalid downstream routing IDs.
  const auto wmma_limit = WmmaCausalAttentionContextLimit(
      c.num_heads, c.num_kv_heads, c.head_dim, c.compress_ratio);
  const auto live_context = std::uint64_t{start_pos} + n_tokens;
  last_only = last_only && MatrixRows(n_tokens) && wmma_limit != 0 &&
              (s.kvarn_pass == nullptr || s.kvarn_pass->prefill) &&
              (!sparse || (live_context <= wmma_limit &&
                           live_context <= std::uint64_t{mask_words_} *
                                               c.compress_ratio * 32));

  const std::uint32_t kv_row = c.AttentionKvDim();
  const std::uint32_t index_capacity =
      IndexerCapacity(c, options_.max_batch, max_context);
  bool prepared = false;
  bool query_rows = false;
  // KVarN layers write FP16 rows and rotate them into the KVarN frame: a
  // pass into the sink and tail ring, a prompt chunk into the staging rows,
  // whose row 0 holds staging_first (plan/exl3/F1-KVARN-DESIGN.md 3).
  const KvarnPass* kvarn = s.kvarn_pass;
  __half* write_k = s.k_cache;
  __half* write_v = s.v_cache;
  const std::uint32_t* write_rows = s.kv_blocks;
  const std::size_t kvarn_row_bytes = std::size_t{kv_row} * sizeof(__half);
  if (kvarn != nullptr && kvarn->prefill) {
    // The rows before the chunk that it reads in FP16: the sink rows of a
    // chunk starting inside the sink, or the ring rows of its first group.
    const std::uint32_t lead = start_pos - kvarn->staging_first;
    const std::size_t from =
        kvarn->staging_first == 0 ? 0 : hip::KvarnFp16Row(kvarn->staging_first);
    if (lead != 0 &&
        (!Check(hipMemcpyAsync(kvarn_staging_k_, s.kvarn_k + from * kv_row,
                               lead * kvarn_row_bytes, hipMemcpyDeviceToDevice,
                               stream_),
                "KVarN staging lead", error_msg) ||
         !Check(hipMemcpyAsync(kvarn_staging_v_, s.kvarn_v + from * kv_row,
                               lead * kvarn_row_bytes, hipMemcpyDeviceToDevice,
                               stream_),
                "KVarN staging lead", error_msg)))
      return false;
    // Writers address rows by absolute position.
    write_k = kvarn_staging_k_ - std::size_t{kvarn->staging_first} * kv_row;
    write_v = kvarn_staging_v_ - std::size_t{kvarn->staging_first} * kv_row;
    write_rows = nullptr;
  } else if (kvarn != nullptr) {
    write_k = s.kvarn_k;
    write_v = s.kvarn_v;
    write_rows = KvarnFp16Rows();
  }
  if (!l.attn_qkv.empty()) {
    const bool fused_projection =
        !projections_ready && WidePrefill(n_tokens) &&
        n_tokens <= options_.max_batch && DenseF16Route(l.attn_qkv, n_tokens) &&
        l.attn_qkv.rows == 13312 && l.attn_qkv.cols == 2560 &&
        c.num_heads == 24 && c.num_kv_heads == 2 && c.head_dim == 256 &&
        c.rotary_dim == 64;
    if (fused_projection) {
      PrepareHalfInput(x, n_tokens, l.attn_qkv.cols);
      prepared = AttentionF16Gemm(
          l.attn_qkv.data, static_cast<const __half*>(s_.x_half),
          l.attn_q_norm.f32(), l.attn_k_norm.f32(), s_.q, s_.attn_gate, write_k,
          write_v, n_tokens, pos, c.rope_theta, c.rms_eps, stream_, s.rope,
          yarn_, write_rows);
      if (!prepared) {
        AssignError(error_msg, "fused attention projection failed");
        return false;
      }
    } else {
      if (!projections_ready &&
          !Dense(l.attn_qkv, x, s_.qg, n_tokens, error_msg)) {
        return false;
      }
      prepared = PrepareAttention(
          s_.qg, l.attn_qkv.rows, l.attn_q_norm.f32(), l.attn_k_norm.f32(),
          s_.q, s_.attn_gate, write_k, write_v, n_tokens, c.num_heads,
          c.num_kv_heads, c.head_dim, c.rotary_dim, pos, c.rope_theta,
          c.rms_eps, stream_, s.rope, prefill_phase, yarn_, write_rows);
      if (!prepared) {
        UnpackQGate(s_.qg, l.attn_qkv.rows, s_.q, s_.attn_gate, s_.k, s_.v,
                    n_tokens, c.num_heads, c.head_dim, kv_row, stream_);
      }
    }
  } else {
    Q8Input xq;
    if (!projections_ready && l.attn_q.exl3 != nullptr) {
      if (!Exl3Group({&l.attn_q, &l.attn_k, &l.attn_v}, {s_.qg, s_.k, s_.v}, x,
                     n_tokens, error_msg))
        return false;
    } else if (!projections_ready &&
               (!Quantize(x, n_tokens, c.hidden_size, &xq, error_msg) ||
                !Dense(l.attn_q, xq, s_.qg, error_msg) ||
                !Dense(l.attn_k, xq, s_.k, error_msg) ||
                !Dense(l.attn_v, xq, s_.v, error_msg))) {
      return false;
    }
    // EXL3 prefill: the query rows are unpacked, normalized, rotated by
    // RoPE and (KVarN) into the KVarN frame in one pass; a KVarN layer's
    // output gate then reads the packed rows directly.
    const bool kvarn_rows = kvarn != nullptr;
    query_rows = prefill_phase && model_->output().exl3 != nullptr &&
                 AttentionQueryRows(
                     s_.qg, 2 * c.AttentionQDim(), l.attn_q_norm.f32(), s_.q,
                     kvarn_rows ? nullptr : s_.attn_gate, n_tokens, c.num_heads,
                     c.head_dim, c.rotary_dim, pos, c.rope_theta, c.rms_eps,
                     kvarn_rows, stream_, s.rope, yarn_);
    if (query_rows) {
      attn_q_rotated_ = kvarn_rows;
      attn_gate_stride_ = kvarn_rows ? 2 * c.AttentionQDim() : 0;
    } else {
      UnpackQGate(s_.qg, 2 * c.AttentionQDim(), s_.q, s_.attn_gate, nullptr,
                  nullptr, n_tokens, c.num_heads, c.head_dim, 0, stream_);
    }
  }
  if (!prepared) {
    if (!query_rows)
      RmsNormRows(s_.q, l.attn_q_norm.f32(), s_.q, n_tokens * c.num_heads,
                  c.head_dim, 1, c.rms_eps, stream_);
    RmsNormRows(s_.k, l.attn_k_norm.f32(), s_.k, n_tokens * c.num_kv_heads,
                c.head_dim, 1, c.rms_eps, stream_);
    if (!query_rows)
      Rope(s_.q, n_tokens, c.num_heads, c.head_dim, c.rotary_dim, pos,
           c.rope_theta, stream_, s.rope, yarn_);
    Rope(s_.k, n_tokens, c.num_kv_heads, c.head_dim, c.rotary_dim, pos,
         c.rope_theta, stream_, s.rope, yarn_);
    StoreKv(s_.k, write_k, n_tokens, kv_row, pos, stream_, write_rows);
    StoreKv(s_.v, write_v, n_tokens, kv_row, pos, stream_, write_rows);
  }
  if (kvarn != nullptr) {
    if (checkpoint_tokens != 0) {
      AssignError(error_msg, "KVarN sessions take no in-pass checkpoints");
      return false;
    }
    if (kvarn->prefill) {
      RotateKvRows(kvarn_staging_k_, kvarn_staging_v_, n_tokens, c.num_kv_heads,
                   pos, stream_, kvarn->staging_first);
    } else {
      RotateKvRows(s.kvarn_k, s.kvarn_v, n_tokens, c.num_kv_heads, pos,
                   stream_);
      // A group ending inside the pass is quantized before the rows after
      // it read it (hip::PlanKvarnPass boundary).
      KvarnQuantizePassBoundary(s.kvarn_k, s.kvarn_v, s.kvarn_records,
                                s.kv_blocks, c.num_kv_heads, pos, n_tokens,
                                stream_, options_.kvarn_bits);
    }
  }

  // Raw keys survive only until pooling. The ring holds the initial
  // pre-budget backlog and a batch, independently for trunk and predictor.
  const bool shared_indexer = prefill_phase && sparse && s.index_k != nullptr &&
                              l.indexer_q.exl3 != nullptr &&
                              l.indexer_q.exl3 == l.indexer_k.exl3;
  if (shared_indexer &&
      !Exl3DensePrefill({&l.indexer_q, &l.indexer_k}, {s_.iq, s_.ik}, x,
                        n_tokens, error_msg))
    return false;
  if (s.index_k != nullptr) {
    if (!shared_indexer &&
        !(projections_ready
              ? DenseBatch(l.indexer_k, x, s_.ik, n_tokens, error_msg)
              : Dense(l.indexer_k, x, s_.ik, n_tokens, error_msg))) {
      return false;
    }
    StoreRows(s_.ik, s.index_k, n_tokens, c.indexer_head_dim, pos,
              index_capacity, stream_);
  }
  const std::uint32_t* mask = nullptr;
  if (sparse) {
    if (!shared_indexer && !Dense(l.indexer_q, x, s_.iq, n_tokens, error_msg)) {
      return false;
    }
    RmsNormRows(s_.iq, l.indexer_q_norm.f32(), s_.iq,
                n_tokens * c.indexer_heads, c.indexer_head_dim, 1, c.rms_eps,
                stream_);
    Rope(s_.iq, n_tokens, c.indexer_heads, c.indexer_head_dim, c.rotary_dim,
         pos, c.rope_theta, stream_, s.rope, yarn_);
    PoolIndexerBlocks(s.index_k, l.indexer_k_norm.f32(), s.block_k, first_block,
                      pos, n_tokens, pool_grid, c.compress_ratio,
                      c.indexer_head_dim, c.rotary_dim, c.rope_theta, c.rms_eps,
                      index_capacity, stream_, s.rope, yarn_, s.kv_blocks);
    // Align score rows to full cache lines. The selector still considers
    // only complete causal blocks, so padding cannot change the ranking.
    // Wide prefill is never captured: compact score rows to the populated
    // context. Graphs retain the maximum stride for replay at later positions.
    const auto score_context =
        n_tokens > kVecBatch ? start_pos + n_tokens : max_context;
    const std::uint32_t blocks =
        (score_context + c.compress_ratio - 1) / c.compress_ratio;
    const std::uint32_t max_blocks = (blocks + 31) / 32 * 32;
    // Query rows are independent; rounding down to a power of two wasted
    // up to half the bounded scratch and doubled launches above 160K.
    const auto select_chunk = static_cast<std::uint32_t>(
        std::min(std::size_t{512}, select_score_floats_ / max_blocks));
    // Catch-up consumes only the final attention tile. Cover its masks even
    // when the dense/sparse boundary shifts the four-query sparse groups.
    const auto first_query = last_only && n_tokens > 16 ? n_tokens - 16 : 0U;
    for (std::uint32_t t0 = first_query; t0 < n_tokens; t0 += select_chunk) {
      const std::uint32_t n = std::min(select_chunk, n_tokens - t0);
      // Batches wider than kVecBatch are never captured, so the host
      // position bounds the scored range (a replayed graph must cover
      // max_blocks).
      const std::uint32_t live =
          n_tokens > kVecBatch ? (start_pos + t0 + n) / c.compress_ratio : 0;
      SelectBlocks(s_.iq + static_cast<std::size_t>(t0) * c.indexer_heads *
                               c.indexer_head_dim,
                   s.block_k,
                   s_.mask + static_cast<std::size_t>(t0) * mask_words_,
                   s_.scores, n, pos, t0, c.indexer_heads, c.indexer_head_dim,
                   c.compress_ratio, c.indexer_top_k / c.compress_ratio,
                   mask_words_, max_blocks, stream_, live, s.kv_blocks);
    }
    mask = s_.mask;
  }
  // Wide batches run the fused WMMA kernel (never inside a graph: the kv
  // extent is a host value), output gate included, skipping the key tiles
  // no query of a block selected; the per-token kernel covers the rest.
  if (last_only &&
      !Check(hipMemsetAsync(s_.ctx, 0,
                            static_cast<std::size_t>(n_tokens) *
                                c.AttentionQDim() * sizeof(float),
                            stream_),
             "partial attention output initialization", error_msg)) {
    return false;
  }
  if (kvarn != nullptr) {
    return KvarnAttention(l, s, n_tokens, pos, start_pos, mask, last_only,
                          project_output, out, error_msg);
  }
  // Queries before the token budget always take the dense tiles, also in a
  // batch that crosses it, so each position's key sweep (and its rounding)
  // does not depend on where the prefill chunks fall. The sparse part goes
  // first: it is the launch that can be refused.
  const auto attend = [&](std::uint32_t first, std::uint32_t rows, bool last) {
    const std::uint32_t pos = start_pos + first;
    const std::uint32_t dense_rows = mask != nullptr && pos < c.indexer_top_k
                                         ? std::min(rows, c.indexer_top_k - pos)
                                         : 0;
    const auto at = [&](std::uint32_t row) {
      return std::size_t{first + row} * c.AttentionQDim();
    };
    return (dense_rows == rows ||
            WmmaCausalAttention(
                s_.q + at(dense_rows), s_.attn_gate + at(dense_rows), s.k_cache,
                s.v_cache,
                mask ? mask + std::size_t{first + dense_rows} * mask_words_
                     : nullptr,
                mask_words_, s_.ctx + at(dense_rows), rows - dense_rows,
                pos + dense_rows, c.num_heads, c.num_kv_heads, c.head_dim,
                c.compress_ratio, stream_, last, s.kv_blocks)) &&
           (dense_rows == 0 || (last && dense_rows != rows) ||
            WmmaCausalAttention(s_.q + at(0), s_.attn_gate + at(0), s.k_cache,
                                s.v_cache, nullptr, mask_words_, s_.ctx + at(0),
                                dense_rows, pos, c.num_heads, c.num_kv_heads,
                                c.head_dim, c.compress_ratio, stream_, last,
                                s.kv_blocks));
  };
  if (checkpoint_tokens != 0 && checkpoint_tokens < n_tokens) {
    // Sparse tiles compact the union of their queries' selected keys.
    // Keep the checkpoint's query grouping identical to a pass ending there.
    if (!attend(0, checkpoint_tokens, false) ||
        !attend(checkpoint_tokens, n_tokens - checkpoint_tokens, false)) {
      AssignError(error_msg, "checkpoint attention geometry is unsupported");
      return false;
    }
    return !project_output ||
           Dense(l.attn_out, s_.ctx, out, n_tokens, error_msg);
  }
  if (MatrixRows(n_tokens) && attend(0, n_tokens, last_only)) {
    return !project_output ||
           Dense(l.attn_out, s_.ctx, out, n_tokens, error_msg);
  }
  // Narrow batches split each row's key tiles over kAttnSplits blocks so a
  // decode step at depth fills the device.
  const bool split = !MatrixRows(n_tokens);
  rocm::Attention(s_.q, s.k_cache, s.v_cache, mask, mask_words_, s_.ctx,
                  split ? s_.attn_partials : nullptr, kAttnSplits, n_tokens,
                  pos, c.num_heads, c.num_kv_heads, c.head_dim,
                  c.compress_ratio, stream_, split ? s_.attn_lists : nullptr,
                  split ? s_.attn_counts : nullptr,
                  c.compress_ratio > 0 ? c.indexer_top_k / c.compress_ratio : 0,
                  s.kv_blocks, nullptr,
                  model_->output().exl3 != nullptr ? c.indexer_top_k : 0);
  const std::uint32_t q_dim = c.AttentionQDim();
  const std::size_t ctx_count = static_cast<std::size_t>(n_tokens) * q_dim;
  // EXL3 decode: the gate also rotates the context for attn_out.
  const DeviceTensor* const out_consumer[] = {&l.attn_out};
  const auto out_slabs =
      project_output ? Exl3Slabs(out_consumer, n_tokens) : Exl3InputSlabs{};
  // A decode batch's output projection reads the gate's Q8_1 rows.
  if (project_output && split && l.attn_out.type == GgmlType::kQ8_0 &&
      l.attn_out.cols == q_dim && UnpaddedQ8(q_dim) &&
      SigmoidMulQ8_1(s_.ctx, s_.attn_gate, ctx_count, s_.ctx_q8_1, stream_)) {
    return Dense(l.attn_out, Q8Input{s_.ctx, s_.ctx_q8_1, n_tokens, q_dim}, out,
                 error_msg);
  }
  if (out_slabs.count != 0 && SigmoidMulRotate(s_.ctx, s_.attn_gate, n_tokens,
                                               q_dim, out_slabs, stream_))
    NoteExl3Rotated(out_slabs, s_.ctx, n_tokens, out_consumer);
  else
    SigmoidMul(s_.ctx, s_.attn_gate, ctx_count, stream_);
  return !project_output || Dense(l.attn_out, s_.ctx, out, n_tokens, error_msg);
}

bool Executor::KvarnAttention(const DeviceLayer& l,
                              const Session::AttentionState& s,
                              std::uint32_t n_tokens, const std::uint32_t* pos,
                              std::uint32_t start_pos,
                              const std::uint32_t* mask, bool last_only,
                              bool project_output, float* out,
                              std::string* error_msg) const {
  const Config& c = config();
  const KvarnPass& pass = *s.kvarn_pass;
  const std::uint32_t kv_row = c.AttentionKvDim();
  const std::size_t row_bytes = std::size_t{kv_row} * sizeof(__half);
  const KvarnReadView view{
      .k_fp16 = s.kvarn_k,
      .v_fp16 = s.kvarn_v,
      .k_staging = pass.prefill ? kvarn_staging_k_ : nullptr,
      .v_staging = pass.prefill ? kvarn_staging_v_ : nullptr,
      .staging_first = pass.staging_first,
      .records = s.kvarn_records,
      .blocks = s.kv_blocks,
      .quant_end = pass.prefill ? pass.quant_end : 0,
      .kv_heads = c.num_kv_heads,
      .bits = options_.kvarn_bits};
  // Scores and context in the rotated frame: H q . H k = q . k, and the
  // context rotates back before the output gate.
  const bool q_rotated = attn_q_rotated_;
  const std::uint32_t gate_stride = attn_gate_stride_;
  attn_q_rotated_ = false;
  attn_gate_stride_ = 0;
  if (!q_rotated)
    RotateHeadRows(s_.q, n_tokens * c.num_heads, stream_);
  bool attended = false;
  if (pass.prefill && MatrixRows(n_tokens)) {
    // The FP16 path's split: queries before the token budget take the dense
    // tiles. A prompt chunk reads one fixed record range, so the fused
    // kernel's shared key tiles hold for every query.
    const std::uint32_t dense_rows =
        mask != nullptr && start_pos < c.indexer_top_k
            ? std::min(n_tokens, c.indexer_top_k - start_pos)
            : 0;
    const auto at = [&](std::uint32_t row) {
      return std::size_t{row} * c.AttentionQDim();
    };
    attended =
        (dense_rows == n_tokens ||
         WmmaCausalAttention(
             s_.q + at(dense_rows), nullptr, nullptr, nullptr,
             mask ? mask + std::size_t{dense_rows} * mask_words_ : nullptr,
             mask_words_, s_.ctx + at(dense_rows), n_tokens - dense_rows,
             start_pos + dense_rows, c.num_heads, c.num_kv_heads, c.head_dim,
             c.compress_ratio, stream_, last_only, s.kv_blocks, &view)) &&
        (dense_rows == 0 || (last_only && dense_rows != n_tokens) ||
         WmmaCausalAttention(
             s_.q, nullptr, nullptr, nullptr, nullptr, mask_words_, s_.ctx,
             dense_rows, start_pos, c.num_heads, c.num_kv_heads, c.head_dim,
             c.compress_ratio, stream_, last_only, s.kv_blocks, &view));
  }
  // Passes read records by each query's own position (KvarnReadGroups), so
  // their rows run one per query as decode steps do.
  const bool split = !MatrixRows(n_tokens);
  if (!attended) {
    rocm::Attention(
        s_.q, nullptr, nullptr, mask, mask_words_, s_.ctx,
        split ? s_.attn_partials : nullptr, kAttnSplits, n_tokens, pos,
        c.num_heads, c.num_kv_heads, c.head_dim, c.compress_ratio, stream_,
        split ? s_.attn_lists : nullptr, split ? s_.attn_counts : nullptr,
        c.compress_ratio > 0 ? c.indexer_top_k / c.compress_ratio : 0,
        s.kv_blocks, &view,
        model_->output().exl3 != nullptr ? c.indexer_top_k : 0);
  }
  if (gate_stride == 0)
    RotateHeadRows(s_.ctx, n_tokens * c.num_heads, stream_);
  if (pass.prefill) {
    // Quantize the groups the chunk completed from staging, then keep its
    // sink rows and its recent rows.
    const std::uint32_t end = start_pos + n_tokens;
    const std::uint32_t first =
        pass.staging_first == 0 ? 1
                                : pass.staging_first >> hip::kKvarnGroupShift;
    const std::uint32_t complete = end >> hip::kKvarnGroupShift;
    if (complete > first)
      KvarnQuantizeGroups(kvarn_staging_k_, kvarn_staging_v_, 0,
                          s.kvarn_records, 0, s.kv_blocks, c.num_kv_heads, 1,
                          first, complete - first, stream_, pass.staging_first,
                          options_.kvarn_bits);
    const auto copy = [&](std::size_t to, std::uint32_t from_position,
                          std::uint32_t rows) {
      const std::size_t from = from_position - pass.staging_first;
      return rows == 0 ||
             (Check(hipMemcpyAsync(s.kvarn_k + to * kv_row,
                                   kvarn_staging_k_ + from * kv_row,
                                   rows * row_bytes, hipMemcpyDeviceToDevice,
                                   stream_),
                    "KVarN chunk rows", error_msg) &&
              Check(hipMemcpyAsync(s.kvarn_v + to * kv_row,
                                   kvarn_staging_v_ + from * kv_row,
                                   rows * row_bytes, hipMemcpyDeviceToDevice,
                                   stream_),
                    "KVarN chunk rows", error_msg));
    };
    if (pass.staging_first == 0 &&
        !copy(0, 0, std::min(end, hip::kKvarnSinkTokens)))
      return false;
    // The ring keeps every row a later pass may rewind into
    // (hip::PlanKvarnPass): the groups starting within one ring of the end.
    // Pieces split at group starts, where the ring may wrap.
    constexpr std::uint32_t kGroup = hip::kKvarnGroupTokens;
    const std::uint32_t reach =
        end > hip::kKvarnTailRingTokens
            ? (end - hip::kKvarnTailRingTokens + kGroup - 1) / kGroup * kGroup
            : 0;
    for (std::uint32_t p =
             std::max({hip::kKvarnSinkTokens, pass.staging_first, reach});
         p < end;) {
      const std::uint32_t piece = std::min(end, (p / kGroup + 1) * kGroup);
      if (!copy(hip::KvarnFp16Row(p), p, piece - p))
        return false;
      p = piece;
    }
  }
  const std::uint32_t q_dim = c.AttentionQDim();
  const std::size_t ctx_count = static_cast<std::size_t>(n_tokens) * q_dim;
  // EXL3 decode: the gating kernel also rotates the context for attn_out.
  const DeviceTensor* const out_consumer[] = {&l.attn_out};
  const auto out_slabs =
      project_output ? Exl3Slabs(out_consumer, n_tokens) : Exl3InputSlabs{};
  if (gate_stride != 0) {
    if (GateRotatedContext(s_.ctx, s_.qg, gate_stride, n_tokens, c.num_heads,
                           stream_, out_slabs))
      NoteExl3Rotated(out_slabs, s_.ctx, n_tokens, out_consumer);
    return !project_output ||
           Dense(l.attn_out, s_.ctx, out, n_tokens, error_msg);
  }
  if (project_output && split && l.attn_out.type == GgmlType::kQ8_0 &&
      l.attn_out.cols == q_dim && UnpaddedQ8(q_dim) &&
      SigmoidMulQ8_1(s_.ctx, s_.attn_gate, ctx_count, s_.ctx_q8_1, stream_)) {
    return Dense(l.attn_out, Q8Input{s_.ctx, s_.ctx_q8_1, n_tokens, q_dim}, out,
                 error_msg);
  }
  if (out_slabs.count != 0 && SigmoidMulRotate(s_.ctx, s_.attn_gate, n_tokens,
                                               q_dim, out_slabs, stream_))
    NoteExl3Rotated(out_slabs, s_.ctx, n_tokens, out_consumer);
  else
    SigmoidMul(s_.ctx, s_.attn_gate, ctx_count, stream_);
  return !project_output || Dense(l.attn_out, s_.ctx, out, n_tokens, error_msg);
}

bool Executor::Moe(const DeviceLayer& l, const float* x, float* out,
                   std::uint32_t n_tokens, std::string* error_msg,
                   bool last_only) const {
  const Config& c = config();
  const std::uint32_t used = c.num_experts_used;
  // This guard must precede the early shared fork: router failures must
  // still order all side writes before a subsequent pass reuses scratch.
  struct JoinGuard {
    const Executor* executor;
    ~JoinGuard() { (void)executor->JoinSide(nullptr); }
  } join_guard{this};
  const auto full_exl3 = [](const DeviceTensor& w) {
    return w.exl3 != nullptr && w.experts == 1 && w.output_begin == 0 &&
           w.rows == w.exl3->n && w.cols == w.exl3->k;
  };
  const bool exl3_overlap = prefill_phase && !last_only && n_tokens >= 512 &&
                            exl3_shared_workspace_.input != nullptr &&
                            n_tokens <= options_.max_batch &&
                            full_exl3(l.shexp_gate) && full_exl3(l.shexp_up) &&
                            full_exl3(l.shexp_down);
  if (exl3_overlap) {
    if (!Check(hipEventRecord(side_start_, stream_), "shared EXL3 fork",
               error_msg) ||
        !Check(hipStreamWaitEvent(side_stream_, side_start_, 0),
               "shared EXL3 wait", error_msg))
      return false;
    shexp_half_ready_ = false;
    const auto shared = [&] {
      if (!Exl3DensePrefill({&l.shexp_gate, &l.shexp_up},
                            {s_.shexp_up, s_.shexp_gate}, x, n_tokens,
                            error_msg, &exl3_shared_workspace_, side_stream_))
        return false;
      Swiglu(s_.shexp_up, s_.shexp_gate,
             std::size_t{n_tokens} * l.shexp_gate.rows, side_stream_);
      return Exl3DensePrefill({&l.shexp_down}, {s_.shexp_out}, s_.shexp_up,
                              n_tokens, error_msg, &exl3_shared_workspace_,
                              side_stream_);
    };
    const bool shared_ok = shared();
    // Even a partial enqueue must be joined on every exit.
    if (!Check(hipEventRecord(side_done_, side_stream_), "shared EXL3 join",
               error_msg)) {
      (void)hipStreamSynchronize(side_stream_);
      return false;
    }
    side_pending_ = true;
    if (!shared_ok)
      return false;
    if (diagnostics_active_)
      ++cycle_diagnostics_.prefill_shared_overlaps;
  }
  // Router logits and the shared-expert gate come out of one GEMM.
  if (!Dense(l.router, x, s_.router, n_tokens, error_msg)) {
    return false;
  }
  // An EXL3 decode call of one routed chunk also builds its expert tiles in
  // the router launch (Exl3RoutedMlp skips its counter reset and BuildTiles).
  exl3_tiles_ready_ = l.ffn_gate_exps.exl3 != nullptr && !prefill_phase &&
                      n_tokens <= 64 &&
                      n_tokens * used <= kExl3FusedTileAssignments &&
                      exl3_.route_arrivals != nullptr;
  if (exl3_tiles_ready_)
    RouterTopKTiles(s_.router, c.num_experts + 1, s_.ids, s_.weights, n_tokens,
                    c.num_experts, used, exl3_.tiles, exl3_.tile_count,
                    exl3_.route_arrivals, Exl3RouteTileRows(n_tokens), stream_);
  else
    RouterTopK(s_.router, c.num_experts + 1, s_.ids, s_.weights, n_tokens,
               c.num_experts, used, stream_);
  if (l.ffn_gate_exps.exl3 == nullptr && ExpertMatrixRows(n_tokens)) {
    ExpertCounts(s_.ids, s_.expert_counts, n_tokens, c.num_experts, used,
                 stream_);
    if (!Check(hipMemcpyAsync(counts_host_, s_.expert_counts,
                              c.num_experts * sizeof(std::uint32_t),
                              hipMemcpyDeviceToHost, stream_),
               "expert counts download", error_msg) ||
        !Check(hipEventRecord(counts_ready_, stream_), "expert counts event",
               error_msg)) {
      return false;
    }
  }
  // The shared expert (gated by the last router row) does not depend on
  // routing. Queue its GEMMs after the count download, then prepare the
  // routed dispatch on the CPU without waiting for these GEMMs to finish.
  // A wide batch runs them on the side stream beside the routed experts,
  // which leave compute idle while they stream expert weights. The shared
  // expert writes its own scratch and may quantize x into s_.x_q8t, so no
  // main-stream work may use s_.x_q8t before JoinSide.
  // The legacy path below overlaps Q8_0; EXL3 uses the explicit-stream
  // workspace above. Other routes retain their original stream ownership.
  const bool overlap = !exl3_overlap && ExpertMatrixRows(n_tokens) &&
                       !last_only && l.shexp_gate.type == GgmlType::kQ8_0 &&
                       l.shexp_up.type == GgmlType::kQ8_0 &&
                       l.shexp_down.type == GgmlType::kQ8_0;
  const hipStream_t main = stream_;
  if (overlap) {
    if (!Check(hipEventRecord(side_start_, main), "side stream fork",
               error_msg) ||
        !Check(hipStreamWaitEvent(side_stream_, side_start_, 0),
               "side stream wait", error_msg)) {
      return false;
    }
    stream_ = side_stream_;
  }
  const DeviceTensor* const shared_inputs[] = {&l.shexp_gate, &l.shexp_up};
  const auto shared_expert = [&] {
    shexp_half_ready_ = false;
    if (l.shexp_gate.exl3 != nullptr && !prefill_phase &&
        Exl3SharedMlp(l, x, s_.shexp_gate, s_.shexp_up, s_.shexp_out, n_tokens,
                      exl3_, stream_,
                      TakeExl3Rotated(x, n_tokens, shared_inputs)))
      return true;
    if (!GatedDense(l.shexp_up, l.shexp_gate, x, s_.shexp_up, n_tokens,
                    &l.shexp_down, error_msg)) {
      return false;
    }
    if (shexp_half_ready_) {
      shexp_half_ready_ = false;
      if (!DenseF16Gemm(l.shexp_down.data, s_.shexp_half, s_.shexp_out,
                        n_tokens, l.shexp_down.rows, l.shexp_down.cols,
                        stream_)) {
        AssignError(error_msg, "shared expert F16 GEMM failed");
        return false;
      }
      return true;
    }
    return Dense(l.shexp_down, s_.shexp_up, s_.shexp_out, n_tokens, error_msg);
  };
  const bool shared_ok = exl3_overlap || shared_expert();
  stream_ = main;
  if (overlap) {
    if (!Check(hipEventRecord(side_done_, side_stream_), "side stream join",
               error_msg)) {
      // Nothing to join: wait for the whole side stream instead.
      (void)hipStreamSynchronize(side_stream_);
      return false;
    }
    side_pending_ = true;
  }
  if (!shared_ok) {
    return false;
  }
  if (last_only && l.ffn_gate_exps.type == GgmlType::kQ8_0 &&
      l.ffn_up_exps.type == GgmlType::kQ8_0 &&
      l.ffn_down_exps.type == GgmlType::kQ8_0) {
    // Catch-up retains only the final predictor residual. Keep the router
    // and shared-expert shapes, then run its routed experts with the same
    // tiled quantization and accumulation as the complete prompt batch.
    const auto base = s_;
    UseScratch(RowScratch(base, n_tokens - 1));
    PrefillPhase phase(true);
    try {
      const std::size_t offset =
          static_cast<std::size_t>(n_tokens - 1) * c.hidden_size;
      const bool ok = MoeExperts(l, x + offset, out + offset, 1, error_msg);
      UseScratch(base);
      return ok;
    } catch (...) {
      UseScratch(base);
      throw;
    }
  }
  return MoeExperts(l, x, out, n_tokens, error_msg) && JoinSide(error_msg);
}

bool Executor::JoinSide(std::string* error_msg) const {
  if (!side_pending_)
    return true;
  const auto status = hipStreamWaitEvent(stream_, side_done_, 0);
  if (status != hipSuccess)
    (void)hipStreamSynchronize(side_stream_);
  side_pending_ = false;
  return Check(status, "side stream join wait", error_msg);
}

bool Executor::WmmaExperts(const DeviceLayer& l, std::uint32_t n_tokens) const {
  const Config& c = config();
  return ExpertMatrixRows(n_tokens) &&
         (l.ffn_gate_exps.type == GgmlType::kQ4_K ||
          l.ffn_gate_exps.type == GgmlType::kQ5_K) &&
         l.ffn_up_exps.type == l.ffn_gate_exps.type &&
         (l.ffn_down_exps.type == GgmlType::kQ5_1 ||
          l.ffn_down_exps.type == GgmlType::kQ8_0) &&
         c.hidden_size % 256 == 0 && c.expert_ff % 64 == 0;
}

bool Executor::MoeExperts(const DeviceLayer& l, const float* x, float* out,
                          std::uint32_t n_tokens,
                          std::string* error_msg) const {
  const Config& c = config();
  const std::uint32_t used = c.num_experts_used;
  const std::uint32_t slots = n_tokens * used;
  if (l.ffn_gate_exps.exl3 != nullptr)
    return Exl3Experts(l, x, out, n_tokens, error_msg);
  if (!RouteHints(n_tokens, error_msg)) {
    return false;
  }
  // Tiled batches take the WMMA route: assignments compacted by expert
  // into 16-row padded buckets, token rows narrowed to F16 once, then the
  // F16 matrix-core GEMM per (expert, row tile).
  const bool wmma_experts = WmmaExperts(l, n_tokens);
  if (wmma_experts) {
    RoutedCompact(s_.ids, s_.expert_counts, s_.routed_bounds, s_.routed_cursors,
                  s_.rows_token, s_.rows_slot, n_tokens, used, c.num_experts,
                  stream_);
    // The GEMMs read F16 token rows: the router's F16 GEMM (or the mix)
    // usually left them in s_.x_half already.
    if (!(half_src_ == x && half_rows_ == n_tokens &&
          half_cols_ == c.hidden_size && !half_bf16_)) {
      NarrowActivations(x, s_.x_half, false,
                        static_cast<std::size_t>(n_tokens) * c.hidden_size,
                        stream_);
      half_src_ = x;
      half_rows_ = n_tokens;
      half_cols_ = c.hidden_size;
      half_bf16_ = false;
    }
    const auto* x_half = static_cast<const __half*>(s_.x_half);
    // Prefill pairs the gate/up projections and applies SwiGLU without
    // materializing the gate; decode-time batches keep separate projections.
    auto* up_half = reinterpret_cast<__half*>(s_.up_e);
    const WeightType gate_type = l.ffn_gate_exps.type == GgmlType::kQ5_K
                                     ? WeightType::kQ5_K
                                     : WeightType::kQ4_K;
    const bool gated_ok =
        routed_64_tiles_ != 0
            ? RoutedGatedF16Gemm(
                  l.ffn_gate_exps.data, l.ffn_up_exps.data, gate_type, x_half,
                  s_.routed_tiles + routed_pair_offset_, routed_pair_tiles_,
                  s_.routed_bounds, s_.rows_token, s_.rows_slot, up_half,
                  c.expert_ff, c.hidden_size, stream_)
            : (RoutedF16Gemm(l.ffn_gate_exps.data, gate_type, x_half,
                             s_.routed_tiles, routed_n_tiles_,
                             routed_tile_rows_, s_.routed_bounds, s_.rows_token,
                             s_.rows_slot, nullptr, s_.gate_e, nullptr,
                             c.expert_ff, c.hidden_size, stream_) &&
               RoutedF16Gemm(l.ffn_up_exps.data, gate_type, x_half,
                             s_.routed_tiles, routed_n_tiles_,
                             routed_tile_rows_, s_.routed_bounds, s_.rows_token,
                             s_.rows_slot, s_.gate_e, nullptr, up_half,
                             c.expert_ff, c.hidden_size, stream_));
    if (!gated_ok) {
      AssignError(error_msg, "routed F16 gate/up GEMM failed");
      return false;
    }
    const WeightType down_type = l.ffn_down_exps.type == GgmlType::kQ8_0
                                     ? WeightType::kQ8_0
                                     : WeightType::kQ5_1;
    // Prefill takes the 64-token down tiles: they decode each expert's
    // weights over more rows, also where narrower tiles would be fewer.
    const bool wide_down = routed_64_tiles_ != 0;
    // The down projection's rows are F16 too: the epilogue reads half the
    // bytes of the largest routed intermediate.
    if (!RoutedF16Gemm(l.ffn_down_exps.data, down_type, up_half,
                       s_.routed_tiles,
                       wide_down ? routed_64_tiles_ : routed_n_tiles_,
                       wide_down ? 64 : routed_tile_rows_, s_.routed_bounds,
                       s_.rows_slot, s_.rows_slot, nullptr, nullptr,
                       reinterpret_cast<__half*>(s_.down_e), c.hidden_size,
                       c.expert_ff, stream_)) {
      AssignError(error_msg, "routed F16 down GEMM failed");
      return false;
    }
  } else {
    if (!GatedExperts(l.ffn_gate_exps, l.ffn_up_exps, x, s_.ids, s_.gate_e,
                      n_tokens, used, error_msg)) {
      return false;
    }
    // The down projection sees one (token, slot) row per expert id.
    if (!Experts(l.ffn_down_exps, s_.gate_e, s_.ids, s_.down_e, slots, 1,
                 n_tokens, error_msg)) {
      return false;
    }
  }
  // The epilogue reads the shared expert's output.
  if (!JoinSide(error_msg))
    return false;
  if (wmma_experts) {
    // The combine that follows folds this epilogue into its own pass when
    // it takes the F16 route (Combine); otherwise it runs here.
    moe_pending_ = out == s_.block_out;
    if (!moe_pending_) {
      MoeEpilogueVec4F16(reinterpret_cast<const __half*>(s_.down_e), s_.weights,
                         s_.shexp_out, s_.router + c.num_experts,
                         c.num_experts + 1, out, n_tokens, used, c.hidden_size,
                         stream_);
    }
  } else if (MatrixRows(n_tokens)) {
    MoeEpilogueVec4(s_.down_e, s_.weights, s_.shexp_out,
                    s_.router + c.num_experts, c.num_experts + 1, out, n_tokens,
                    used, c.hidden_size, stream_);
  } else {
    MoeEpilogue(s_.down_e, s_.weights, s_.shexp_out, s_.router + c.num_experts,
                c.num_experts + 1, out, n_tokens, used, c.hidden_size, stream_);
  }
  return true;
}

Exl3PrefillScratch Executor::Exl3PrefillView() const {
  auto view = exl3_prefill_;
  // Attention intermediates are dead. Resolve these pointers after any
  // RowScratch narrowing, rather than retaining the arena's base addresses.
  view.gate = s_.gate_e;
  view.up = s_.up_e;
  return view;
}

bool Executor::Exl3Experts(const DeviceLayer& l, const float* x, float* out,
                           std::uint32_t rows, std::string* error) const {
  const auto& c = config();
  const bool tiles_ready = std::exchange(exl3_tiles_ready_, false);
  try {
    if (!prefill_phase) {
      for (std::uint32_t first = 0; first < rows; first += 64) {
        const auto count = std::min(64U, rows - first);
        const auto assignment = std::size_t{first} * c.num_experts_used;
        // Raw down dots: Exl3RoutedCombine restores them in the epilogue.
        Exl3RoutedMlp(l, x + std::size_t{first} * c.hidden_size,
                      s_.ids + assignment, s_.gate_e, s_.up_e,
                      s_.down_e + assignment * c.hidden_size, count,
                      c.num_experts_used, exl3_, stream_, nullptr,
                      tiles_ready && rows <= 64, true);
        if (diagnostics_active_) {
          if (exl3_.down_exponents != nullptr)
            cycle_diagnostics_.scaled_down_rows +=
                std::uint64_t{count} * c.num_experts_used;
          if (!DiagnosticCount(exl3_.tile_count,
                               &cycle_diagnostics_.expert_tile_reads, error))
            return false;
        }
      }
    } else {
      // Keep this route for EVERY kPrefill row, including singleton tails.
      // Selecting decode GEMV by token/bucket width changes the reduction
      // tree when a Sync/checkpoint boundary moves. X3 is a phase contract,
      // not a minimum-row heuristic; its 64-row tile never changes here.
      const auto scratch = Exl3PrefillView();
      // Raw down dots stay packed; the combine below restores and sums them.
      Exl3PrefillExpertsRaw(l, x, s_.ids, rows, c.num_experts_used,
                            kExl3PrefillTileRows, scratch, stream_);
      if (diagnostics_active_) {
        cycle_diagnostics_.prefill_expert_rows += rows;
        cycle_diagnostics_.expert_tile_reads_complete = false;
        if (!DiagnosticCount(exl3_prefill_.bucket_count,
                             &cycle_diagnostics_.prefill_expert_buckets, error))
          return false;
      }
    }
    if (!JoinSide(error))
      return false;
    if (prefill_phase) {
      const bool defer = exl3_moe_defer_ && out == s_.block_out;
      exl3_moe_defer_ = false;
      if (defer) {
        exl3_moe_pending_ = {&l, rows};
        return true;
      }
      Exl3PrefillCombine(l, Exl3PrefillView(), s_.ids, rows, c.num_experts_used,
                         s_.weights, s_.shexp_out, s_.router + c.num_experts,
                         c.num_experts + 1, out, stream_);
      return true;
    }
    Exl3RoutedCombine(l, s_.ids, rows, c.num_experts_used, s_.down_e,
                      s_.weights, s_.shexp_out, s_.router + c.num_experts,
                      c.num_experts + 1, out, stream_);
    return true;
  } catch (const std::exception& e) {
    AssignError(error, e.what());
    return false;
  }
}

bool Executor::CopyTrunkHidden(const Session& session, std::span<float> hidden,
                               std::string* error_msg) const {
  if (!session.mtp_enabled_ || session.owner_ != this ||
      session.position_ == 0 || hidden.empty() ||
      hidden.size() % config().HcDim() != 0 ||
      hidden.size() / config().HcDim() >
          std::min(session.position_, options_.max_speculative)) {
    AssignError(error_msg, "invalid trunk hidden diagnostic input");
    return false;
  }
  return Check(hipMemcpyAsync(hidden.data(), session.mtp_.target_hidden,
                              hidden.size_bytes(), hipMemcpyDeviceToHost,
                              stream_),
               "trunk hidden download", error_msg) &&
         Check(hipStreamSynchronize(stream_), "trunk hidden", error_msg);
}

std::size_t Executor::EstimateExl3PrefillScratchBytes(const Config& config,
                                                      std::uint32_t rows,
                                                      std::string* error_msg) {
  Exl3PrefillLayout layout;
  return PlanExl3Prefill(config, rows, &layout, error_msg) ? layout.bytes : 0;
}

std::uint32_t Executor::PrefillChunkRows(
    std::uint32_t requested, std::uint32_t until_checkpoint) const noexcept {
  auto limit = options_.max_batch;
  if (model_->output().exl3 != nullptr && options_.exl3_prefill_chunk_rows != 0)
    limit = std::min(limit, options_.exl3_prefill_chunk_rows);
  return std::min({requested, until_checkpoint, limit});
}

bool Executor::BeginMtpCycleDiagnostics(std::string* error_msg) const {
  if (diagnostics_active_ || model_->output().exl3 == nullptr) {
    AssignError(error_msg, "MTP cycle diagnostics require idle EXL3 executor");
    return false;
  }
  if (!Check(hipStreamSynchronize(stream_), "diagnostic cycle start",
             error_msg))
    return false;
  cycle_diagnostics_ = {};
  diagnostic_depth_ = 0;
  diagnostic_phase_ = nullptr;
  diagnostic_started_ = std::chrono::steady_clock::now();
  diagnostics_active_ = true;
  return true;
}

bool Executor::EndMtpCycleDiagnostics(std::uint64_t committed_tokens,
                                      std::uint64_t drafts,
                                      MtpCycleDiagnostics* output,
                                      std::string* error_msg) const {
  if (!diagnostics_active_ || diagnostic_depth_ != 0 || output == nullptr) {
    AssignError(error_msg, "invalid MTP diagnostic cycle end");
    return false;
  }
  // Always release the diagnostic scope, including a failed GPU cycle.
  diagnostics_active_ = false;
  if (!Check(hipStreamSynchronize(stream_), "diagnostic cycle end", error_msg))
    return false;
  cycle_diagnostics_.committed_tokens = committed_tokens;
  cycle_diagnostics_.drafts = drafts;
  cycle_diagnostics_.wall_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - diagnostic_started_)
          .count();
  *output = cycle_diagnostics_;
  return true;
}

Executor::DiagnosticPhase::DiagnosticPhase(const Executor& e, DiagnosticStage s)
    : executor(e), stage(s), active(e.diagnostics_active_) {
  if (active) {
    ++executor.diagnostic_depth_;
    parent = executor.diagnostic_phase_;
    executor.diagnostic_phase_ = this;
    started = std::chrono::steady_clock::now();
  }
}

Executor::DiagnosticPhase::~DiagnosticPhase() {
  if (!active)
    return;
  --executor.diagnostic_depth_;
  executor.diagnostic_phase_ = parent;
  const auto elapsed = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - started)
                           .count();
  if (parent != nullptr)
    parent->children_ms += elapsed;
  const auto ms = elapsed - children_ms;
  auto& d = executor.cycle_diagnostics_;
  switch (stage) {
    case DiagnosticStage::kTarget:
      d.target_ms += ms;
      break;
    case DiagnosticStage::kDraft:
      d.draft_ms += ms;
      break;
    case DiagnosticStage::kHead:
      d.head_ms += ms;
      break;
    case DiagnosticStage::kCatchUp:
      d.catch_up_ms += ms;
      break;
  }
}

bool Executor::DiagnosticCount(const std::uint32_t* device,
                               std::uint64_t* total,
                               std::string* error_msg) const {
  std::uint32_t count = 0;
  if (!Check(hipMemcpyAsync(&count, device, sizeof(count),
                            hipMemcpyDeviceToHost, stream_),
             "diagnostic count", error_msg) ||
      !Check(hipStreamSynchronize(stream_), "diagnostic count ready",
             error_msg))
    return false;
  *total += count;
  return true;
}

bool Executor::MtpHead(const DeviceMixer& head, const float* res, bool token,
                       bool candidates, const Session* draft,
                       std::string* error_msg) const {
  const std::uint32_t prefix = model_->draft_output().rows;
  const std::uint32_t rows =
      draft != nullptr ? prefix + kMtpDraftExtraRows : model_->output().rows;
  if (diagnostics_active_ && draft == nullptr &&
      model_->output().exl3 != nullptr)
    cycle_diagnostics_.head_segments += (rows + 127) / 128;
  if (!HcMix(head, res, false, s_.mixed, nullptr, 1, error_msg) ||
      !(draft != nullptr
            ? DraftLogits(*draft, s_.mixed, s_.logits, error_msg)
            : Dense(model_->output(), s_.mixed, s_.logits, 1, error_msg))) {
    return false;
  }
  const auto count = std::min<std::uint32_t>(rows, kMtpCandidates);
  if (candidates) {
    MtpTopCandidates(s_.logits, s_.mtp_ids, s_.mtp_scratch_ids, s_.mtp_scores,
                     rows, stream_);
    if (draft != nullptr)
      RemapDraftIds(s_.mtp_ids, count, prefix, draft->mtp_.draft_ids, stream_);
  }
  if (token) {
    Argmax(s_.logits, s_.mtp_argmax, s_.mtp_token, 1, rows, stream_);
    if (draft != nullptr)
      RemapDraftIds(reinterpret_cast<std::uint32_t*>(s_.mtp_token), 1, prefix,
                    draft->mtp_.draft_ids, stream_);
    if (!Check(
            hipMemcpyAsync(mtp_token_host_, s_.mtp_token, sizeof(std::int32_t),
                           hipMemcpyDeviceToHost, stream_),
            "draft token download", error_msg)) {
      return false;
    }
  }
  if (candidates) {
    static_assert(offsetof(MtpCandidateLogits, logits) ==
                  kMtpCandidates * sizeof(std::uint32_t));
    const auto bytes =
        offsetof(MtpCandidateLogits, logits) + count * sizeof(float);
    if (!Check(hipMemcpyAsync(mtp_candidates_host_, s_.mtp_ids, bytes,
                              hipMemcpyDeviceToHost, stream_),
               "draft candidates download", error_msg)) {
      return false;
    }
  }
  return true;
}

bool Executor::Run(Session& session, std::uint64_t key, bool graph,
                   const std::function<bool()>& body, std::string* error_msg,
                   bool synchronize) const {
  // Host counters must observe each executed operation. Captured/replayed
  // bodies would either duplicate or omit them, as with MtpTrace copies.
  graph = graph && !diagnostics_active_;
  // A batch shape runs eagerly once before it is captured: the first pass
  // grows the GEMM tier's arena, which capture forbids.
  if (graph && session.warmed_.contains(key)) {
    hipGraphExec_t exec = nullptr;
    bool enqueued = false;
    if (const auto it = session.graphs_.find(key);
        it != session.graphs_.end()) {
      exec = it->second;
    } else {
      // A warmed target shape may need its first graph or a replacement
      // after rollback storage was released. Queue this pass first, then
      // record identical commands while the GPU executes it. Short prefix
      // and draft graphs keep their existing route. No second pass is launched.
      if (synchronize && (key & (std::uint64_t{1} << 40)) == 0) {
        auto dispatch = std::tie(
            moe_pending_, defer_residual_, res_deferred_, shexp_half_ready_,
            q8t_src_, q8t_rows_, q8t_cols_, half_src_, half_rows_, half_cols_,
            half_bf16_, xn_half_, inject_parts_, inject_fused_, q8_slot_,
            xn_q8_ready_, mixed_q8_ready_, xn_deferred_.res, xn_deferred_.gamma,
            xn_deferred_.rows, exl3_moe_defer_, exl3_moe_pending_.layer,
            exl3_moe_pending_.rows, xn_q8t_norm_.res, xn_q8t_norm_.gamma,
            xn_q8t_norm_.rows, exl3_rotated_);
        const auto before = std::apply(
            [](const auto&... fields) { return std::make_tuple(fields...); },
            dispatch);
        if (!body())
          return false;
        // Producer readiness and alternating scratch addresses must match
        // the executed pass when the graph records its kernel arguments.
        dispatch = before;
        enqueued = true;
      }
      hipGraph_t captured = nullptr;
      // Frozen peer sessions can copy snapshots on independent nonblocking
      // streams while the scheduler records this session's decode graph.
      if (!Check(
              hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal),
              "graph capture", error_msg)) {
        return false;
      }
      const bool ok = body();
      if (!Check(hipStreamEndCapture(stream_, &captured), "graph capture end",
                 error_msg) ||
          !ok) {
        if (captured != nullptr) {
          (void)hipGraphDestroy(captured);
        }
        return false;
      }
      const bool instantiated =
          Check(hipGraphInstantiate(&exec, captured, nullptr, nullptr, 0),
                "graph instantiate", error_msg);
      (void)hipGraphDestroy(captured);
      if (!instantiated) {
        return false;
      }
      session.graphs_.emplace(key, exec);
    }
    if (!enqueued &&
        !Check(hipGraphLaunch(exec, stream_), "graph launch", error_msg)) {
      return false;
    }
  } else {
    if (!body()) {
      return false;
    }
    session.warmed_.insert(key);
  }
  return !synchronize ||
         Check(hipStreamSynchronize(stream_), "forward", error_msg);
}

bool Executor::Forward(Session& session, std::span<const std::int32_t> tokens,
                       std::uint32_t n_logits, float* logits, ForwardMode mode,
                       std::string* error_msg, PrefillCheckpoint* checkpoint,
                       std::span<const std::int32_t> next) const {
  DiagnosticPhase diagnostic(*this, DiagnosticStage::kTarget);
  // Only this call's PLE layer may start the prefetch of `next`.
  struct NextHint {
    std::span<const std::int32_t>& hint;
    ~NextHint() { hint = {}; }
  } next_hint{prefetch_next_};
  if (mode == ForwardMode::kPrefill) {
    prefetch_next_ = next;
  }
  const bool speculative = mode == ForwardMode::kVerify;
  PrefillPhase phase(mode == ForwardMode::kPrefill);
  selected_logits_ = nullptr;
  DropReadyQ8();
  const Config& c = config();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  if (checkpoint && (mode != ForwardMode::kPrefill ||
                     checkpoint->source != &session || checkpoint->tokens > n ||
                     checkpoint->state->position_ !=
                         session.position_ + checkpoint->tokens)) {
    AssignError(error_msg, "prefill checkpoint does not match this pass");
    return false;
  }
  if (n == 0 || n > options_.max_batch ||
      (speculative && n > options_.max_speculative)) {
    AssignError(error_msg, "token batch is empty or exceeds the batch limit");
    return false;
  }
  if (n_logits > n || n_logits > options_.max_logit_rows) {
    AssignError(error_msg, "requested logit rows exceed the batch or limit");
    return false;
  }
  if (session.owner_ != this || session.position_ + n > session.max_context_) {
    AssignError(error_msg, "session context is full");
    return false;
  }
  for (auto t : tokens) {
    if (t < 0 || static_cast<std::uint32_t>(t) >= c.vocab_size) {
      AssignError(error_msg, "token out of range");
      return false;
    }
  }
  const std::uint32_t start_pos = session.position_;
  if (!session.CheckCancellation(error_msg))
    return false;
  if ((session.kvarn_ || session.kv_pool_) && checkpoint) {
    AssignError(error_msg,
                "KVarN or shared KV sessions take no in-pass checkpoints");
    return false;
  }
  if (session.mtp_enabled_ && prefill_phase &&
      session.mtp_.position != start_pos) {
    AssignError(error_msg,
                "MTP must consume the preceding frontier before prefill");
    return false;
  }
  if (speculative && !EnsureRollback(session, n - 1, error_msg))
    return false;
  // Failed eager/captured work can still be reading pinned inputs. Drain it
  // before another request reuses those buffers; successful forwards already
  // synchronize below and need no additional wait.
  struct PinnedInputGuard {
    hipStream_t stream;
    bool completed{false};
    ~PinnedInputGuard() {
      if (!completed)
        (void)hipStreamSynchronize(stream);
    }
  } pinned_inputs{stream_};
  session.EnsureKvRows(start_pos + n);
  // Start the disk rows before protecting checkpoint rows, which can commit
  // device pages: layer 0 alone cannot hide the whole read.
  if (session.continuation_hooks_) {
    const auto end =
        session.position_ + static_cast<std::uint32_t>(tokens.size());
    session.continuation_hooks_->BeforeWrite(
        false, session.position_, end, session.blocks_,
        end > c.indexer_top_k ? end / c.compress_ratio : session.blocks_);
  }
  if (session.kvarn_ && !PlanKvarn(session, start_pos, n, error_msg))
    return false;
  const auto ngram = session.ngram_;
  if (c.ple_layer >= 0) {
    if (ngram_ == nullptr) {
      AssignError(error_msg, "n-gram table is not open");
      return false;
    }
    if (!PleFetch(session, tokens, speculative, error_msg)) {
      return false;
    }
  }
  try {
    session.PreserveSnapshots(session.position_, session.mtp_.position);
  } catch (...) {
    // The forward never ran; leave the n-gram window where it was.
    (void)WaitPle(nullptr);
    session.ngram_ = ngram;
    throw;
  }
  ++session.mutation_epoch_;
  session.spec_base_ = start_pos;
  session.spec_tokens_ = speculative ? n : 0;
  // Everything the launched work reads from the host sits in pinned
  // buffers the graph nodes point at: tokens, the control block (positions
  // the kernels read on the device) and the n-gram rows.
  std::copy(tokens.begin(), tokens.end(), tokens_host_);
  control_host_->position = start_pos;
  control_host_->blocks = session.blocks_;
  control_host_->mtp_position = session.mtp_.position;
  control_host_->mtp_blocks = session.mtp_.blocks;
  control_host_->hidden_row = -1;
  // Sparse selection only changes the result once a query can see more
  // than the token budget; every layer of this model shares one ratio.
  const bool sparse = c.compress_ratio > 0 && start_pos + n > c.indexer_top_k;
  const std::uint32_t complete =
      c.compress_ratio > 0 ? (start_pos + n) / c.compress_ratio : 0;
  const std::uint32_t pool_grid =
      sparse && complete > session.blocks_ ? complete - session.blocks_ : 0;
  // Decode-sized batches replay as graphs; a pooling backlog (the first
  // batch past the budget) needs the wider eager grid.
  const std::uint32_t graph_pool_grid = n / std::max(c.compress_ratio, 1u) + 1;
  const bool graph = !diagnostics_active_ && !prefill_phase && n <= kVecBatch &&
                     pool_grid <= graph_pool_grid &&
                     session.position_ >= session.VisionLayout().PrefixLength();
  const std::uint64_t key = static_cast<std::uint64_t>(n) |
                            (static_cast<std::uint64_t>(n_logits) << 16) |
                            (static_cast<std::uint64_t>(speculative) << 32) |
                            (static_cast<std::uint64_t>(sparse) << 33) |
                            (std::uint64_t{logits != nullptr} << 34);
  // PLE first consumes the disk rows at its injection layer. Queue the
  // preceding layers before waiting, including on captured graph replay.
  // Both pieces use the same stream and arithmetic as the unsplit graph.
  const std::uint32_t first_layer =
      graph && c.ple_layer > 0 ? static_cast<std::uint32_t>(c.ple_layer) : 0;
  if (first_layer > 0) {
    const auto prefix = [&] {
      return ForwardBody(session, n, 0, false, speculative, sparse, start_pos,
                         graph_pool_grid, 0, first_layer, error_msg);
    };
    constexpr std::uint64_t kPrefixKey = std::uint64_t{1} << 35;
    if (!Run(session, key | kPrefixKey, graph, prefix, error_msg, false)) {
      return false;
    }
  }
  const auto body = [&] {
    return ForwardBody(session, n, n_logits, logits != nullptr, speculative,
                       sparse, start_pos, graph ? graph_pool_grid : pool_grid,
                       first_layer, c.num_layers, error_msg, checkpoint, graph);
  };
  // The prefix is already queued. Wait once before the suffix so recording
  // its graph can reuse these pinned rows while the eager pass executes.
  if (graph && ple_pending_ && !WaitPle(error_msg)) {
    return false;
  }
  if (!Run(session, key, graph, body, error_msg)) {
    return false;
  }
  if (n_logits > 0 && logits != nullptr) {
    std::copy_n(logits_host_, static_cast<std::size_t>(n_logits) * c.vocab_size,
                logits);
  }

  session.position_ += n;
  if (diagnostics_active_ && speculative)
    cycle_diagnostics_.verify_rows += n;
  if (sparse) {
    session.blocks_ = complete;
  }
  if (session.kvarn_ && !FinishKvarn(session, error_msg))
    return false;
  if (session.mtp_enabled_ && prefill_phase && n > 1) {
    // Every successor except the last one is already known. Consume it while
    // the trunk residual is still in shared scratch. MtpBody reads that
    // residual into xn before reusing res for the predictor output.
    PrefillPhase draft_phase(false);
    if (!MtpForward(session, tokens.subspan(1), 0, {.kv_only = true}, error_msg,
                    s_.res))
      return false;
  }
  pinned_inputs.completed = true;
  return true;
}

bool Executor::ForwardBody(Session& session, std::uint32_t n,
                           std::uint32_t n_logits, bool download_logits,
                           bool speculative, bool sparse,
                           std::uint32_t start_pos, std::uint32_t pool_grid,
                           std::uint32_t first_layer, std::uint32_t end_layer,
                           std::string* error_msg,
                           PrefillCheckpoint* checkpoint,
                           bool embeddings_ready) const {
  const Config& c = config();
  if (first_layer == 0) {
    if (!Check(hipMemcpyAsync(session.control_, control_host_,
                              sizeof(Session::Control), hipMemcpyHostToDevice,
                              stream_),
               "control upload", error_msg) ||
        !Check(hipMemcpyAsync(s_.tokens, tokens_host_, n * sizeof(std::int32_t),
                              hipMemcpyHostToDevice, stream_),
               "token upload", error_msg)) {
      return false;
    }
    EmbedTokens(model_->token_embd().data, SmallType(model_->token_embd().type),
                s_.tokens, s_.res, n, c.hidden_size, c.hc_count, stream_);
    session.vision_input_.Inject(s_.res, start_pos, n, c.hidden_size,
                                 c.hc_count, stream_);
  }
  const auto& layers = model_->layers();
  bool normed = false;  ///< xn holds the next mixer's grouped norm of res
  for (std::uint32_t il = first_layer; il < end_layer; ++il) {
    if (!session.CheckCancellation(error_msg))
      return false;
    const DeviceLayer& l = layers[il];
    if (c.IsPleLayer(il) && !Ple(l, session, n, s_.res, speculative, error_msg,
                                 embeddings_ready, checkpoint)) {
      return false;
    }
    // Both stacked SSM projections consume the same F16 input. Avoid writing
    // unused F32/Q8 copies when the wide mixer can supply that input directly.
    const bool half_only = WidePrefill(n) && n <= options_.max_batch &&
                           l.linear && DenseF16Route(l.ssm_in, n) &&
                           l.ssm_alpha_beta.type == GgmlType::kF16 &&
                           l.ssm_alpha_beta.cols == l.ssm_in.cols;
    // The mixer's EXL3 decode projections of `mixed`.
    const std::array<const DeviceTensor*, 3> mixer_rotate =
        l.linear ? std::array<const DeviceTensor*, 3>{&l.ssm_qkv, &l.ssm_gate}
                 : std::array<const DeviceTensor*, 3>{&l.attn_q, &l.attn_k,
                                                      &l.attn_v};
    const std::size_t mixer_count = l.linear ? (l.ssm_in.empty() ? 2 : 0) : 3;
    if (!HcMix(l.hc_attn, s_.res, normed, s_.mixed, s_.inject, n, error_msg,
               half_only, false, std::span(mixer_rotate.data(), mixer_count))) {
      return false;
    }
    // The last target layer's query outputs have no later cache consumer.
    // Preserve all K/V/indexer rows and the final query's original tile.
    const bool last_attention = WidePrefill(n) && !session.mtp_enabled_ &&
                                il + 1 == c.num_layers && n_logits == 1 &&
                                checkpoint == nullptr;
    if (l.linear) {
      const auto capture =
          checkpoint ? GdnCheckpoint{checkpoint->state->linear_[il].state,
                                     checkpoint->state->linear_[il].conv_state,
                                     checkpoint->tokens}
                     : GdnCheckpoint{};
      if (!LinearAttention(l, session.linear_[il], s_.mixed, s_.block_out, n,
                           speculative, error_msg, false, true, capture)) {
        return false;
      }
    } else if (!Attention(l, session.attention_[il], s_.mixed, s_.block_out, n,
                          &session.control_->position,
                          &session.control_->blocks, start_pos, pool_grid,
                          session.max_context_, sparse, error_msg,
                          last_attention, false, true,
                          checkpoint ? checkpoint->tokens : 0)) {
      return false;
    }

    const float* next_norm =
        il + 1 < c.num_layers
            ? (c.IsPleLayer(il + 1) ? nullptr
                                    : layers[il + 1].hc_attn.norm.f32())
            : nullptr;
    const float* next_inject =
        next_norm != nullptr ? InjectWeights(layers[il + 1].hc_attn) : nullptr;
    // Cache updates precede the FFN. Without MTP, only requested head rows
    // and the checkpoint frontier consume the final layer's residual.
    // Keep aligned tiles and at least 96 rows to retain the wide arithmetic.
    const auto needed =
        std::max({96U, n_logits, checkpoint ? n - checkpoint->tokens + 1 : 0U});
    const auto skipped = prefill_phase && !session.mtp_enabled_ &&
                                 il + 1 == c.num_layers && n > needed
                             ? (n - needed) / 128 * 128
                             : 0U;
    const auto finish = [&](std::uint32_t rows) {
      defer_residual_ = WmmaExperts(l, rows);
      // The wide FFN reads the mix only through its F16 rows (router, routed
      // experts) and tiled Q8 rows (shared expert).
      const bool no_f32 = prefill_phase && defer_residual_ &&
                          rows <= options_.max_batch &&
                          l.router.type == GgmlType::kF16 &&
                          l.shexp_gate.type == GgmlType::kQ8_0 &&
                          l.shexp_up.type == GgmlType::kQ8_0;
      Combine(s_.res, l.hc_ffn.norm.f32(), rows, InjectWeights(l.hc_ffn));
      const DeviceTensor* const shared_rotate[] = {&l.shexp_gate, &l.shexp_up};
      if (!HcMix(l.hc_ffn, s_.res, true, s_.mixed, s_.inject, rows, error_msg,
                 false, no_f32, shared_rotate))
        return false;
      // Only the combine below reads this MoE output: EXL3 prefill folds the
      // expert sum into it.
      exl3_moe_defer_ = prefill_phase;
      const bool moe_ok = Moe(l, s_.mixed, s_.block_out, rows, error_msg);
      exl3_moe_defer_ = false;
      if (!moe_ok)
        return false;
      Combine(s_.res, next_norm, rows, next_inject);
      return true;
    };
    if (skipped != 0) {
      struct RestoreScratch {
        const Executor* executor;
        Scratch scratch;
        ~RestoreScratch() { executor->UseScratch(scratch); }
      } restore{this, s_};
      auto tail = RowScratch(s_, skipped);
      tail.inject =
          s_.inject + std::size_t{skipped} * c.hc_count * inject_parts_;
      UseScratch(tail);
      if (!finish(n - skipped))
        return false;
    } else if (!finish(n)) {
      return false;
    }
    normed = next_norm != nullptr;
  }
  if (end_layer < c.num_layers) {
    return true;
  }
  if (checkpoint) {
    if (checkpoint->hidden_rows != 0 &&
        !Check(hipMemcpyAsync(checkpoint->state->mtp_.target_hidden,
                              s_.res + std::size_t{checkpoint->tokens -
                                                   checkpoint->hidden_rows} *
                                           c.HcDim(),
                              std::size_t{checkpoint->hidden_rows} * c.HcDim() *
                                  sizeof(float),
                              hipMemcpyDeviceToDevice, stream_),
               "checkpoint hidden rows", error_msg))
      return false;
    PrefillPhase head_phase(false);
    if (!HcMix(model_->hc_head(),
               s_.res + std::size_t{checkpoint->tokens - 1} * c.HcDim(), false,
               s_.mixed, nullptr, 1, error_msg) ||
        !Dense(model_->output(), s_.mixed, s_.logits, 1, error_msg) ||
        !Check(hipMemcpyAsync(checkpoint->logits.data(), s_.logits,
                              std::size_t{c.vocab_size} * sizeof(float),
                              hipMemcpyDeviceToHost, stream_),
               "checkpoint logits", error_msg))
      return false;
  }
  // Prefill catches up immediately from shared scratch. Only the short tail
  // needed across calls belongs to this session.
  const auto kept = std::min(
      checkpoint && checkpoint->tokens < n ? n - checkpoint->tokens : n,
      options_.max_speculative);
  if (session.mtp_enabled_ &&
      !Check(hipMemcpyAsync(
                 session.mtp_.target_hidden,
                 s_.res + static_cast<std::size_t>(n - kept) * c.HcDim(),
                 static_cast<std::size_t>(kept) * c.HcDim() * sizeof(float),
                 hipMemcpyDeviceToDevice, stream_),
             "hidden keep", error_msg)) {
    return false;
  }
  if (n_logits > 0) {
    PrefillPhase head_phase(false);
    // Only the requested tail needs the head's normalization.
    const std::size_t skip = static_cast<std::size_t>(n - n_logits);
    const DeviceMixer& head = model_->hc_head();
    if (!HcMix(head, s_.res + skip * c.HcDim(), false, s_.mixed, nullptr,
               n_logits, error_msg) ||
        !Dense(model_->output(), s_.mixed, s_.logits, n_logits, error_msg)) {
      return false;
    }
    if (download_logits &&
        !Check(hipMemcpyAsync(logits_host_, s_.logits,
                              static_cast<std::size_t>(n_logits) *
                                  c.vocab_size * sizeof(float),
                              hipMemcpyDeviceToHost, stream_),
               "logits download", error_msg)) {
      return false;
    }
  }
  return true;
}

bool Executor::Rollback(Session& session, std::uint32_t keep,
                        std::string* error_msg, float* logits) const {
  const Config& c = config();
  const std::uint32_t n = session.spec_tokens_;
  if (n == 0 || keep == 0 || keep > n) {
    AssignError(error_msg, "rollback outside the pending speculative batch");
    return false;
  }
  if (session.continuation_hooks_ && keep < n) {
    const auto end = session.spec_base_ + keep;
    session.continuation_hooks_->BeforeWrite(
        false, end, session.position_,
        std::min(session.blocks_, end / c.compress_ratio), session.blocks_);
  }
  session.spec_tokens_ = 0;
  if (logits != nullptr &&
      !Check(hipMemcpyAsync(
                 logits_host_,
                 VerificationLogits() +
                     static_cast<std::size_t>(keep - 1) * c.vocab_size,
                 c.vocab_size * sizeof(float), hipMemcpyDeviceToHost, stream_),
             "frontier download", error_msg)) {
    return false;
  }
  if (keep == n && logits == nullptr) {
    return true;
  }
  if (keep < n) {
    const std::size_t conv_elems =
        static_cast<std::size_t>(c.ssm_conv_kernel - 1) * c.SsmConvChannels();
    const std::size_t slot = keep - 1;
    for (auto& l : session.linear_) {
      if (l.state == nullptr) {
        continue;
      }
      RestoreGdnState(l.state, l.state_snapshots, keep, c.ssm_num_k_heads,
                      c.ssm_num_v_heads, stream_);
      if (!Check(hipGetLastError(), "state rollback", error_msg) ||
          !Check(hipMemcpyAsync(l.conv_state, l.conv_snapshots.rows[slot],
                                conv_elems * sizeof(float),
                                hipMemcpyDeviceToDevice, stream_),
                 "conv rollback", error_msg)) {
        return false;
      }
    }
    if (session.ple_history_ != nullptr) {
      const std::size_t hist =
          static_cast<std::size_t>(c.PleConvHistory()) * c.HcDim();
      if (!Check(hipMemcpyAsync(
                     session.ple_history_, session.ple_snapshots_.rows[slot],
                     hist * sizeof(float), hipMemcpyDeviceToDevice, stream_),
                 "PLE rollback", error_msg)) {
        return false;
      }
      session.ngram_ = session.ngram_snapshots_[slot];
    }
    session.position_ = session.spec_base_ + keep;
    // Pooled block keys past the kept prefix are stale; they are rebuilt
    // from the raw keys when needed.
    session.blocks_ =
        c.compress_ratio == 0
            ? 0
            : std::min(session.blocks_, session.position_ / c.compress_ratio);
  }
  if (!Check(hipStreamSynchronize(stream_), "rollback", error_msg)) {
    return false;
  }
  if (logits != nullptr) {
    std::copy_n(logits_host_, c.vocab_size, logits);
  }
  return true;
}

/// KVarN sessions' payloads (records, FP16 sink and tail rows in place of
/// the FP16 K/V of the trunk layers). FP16 payloads keep kSnapshotMagic.
constexpr std::array<char, 8> kSnapshotKvarnMagic{'Q', 'F', 'N', 'S',
                                                  'N', 'A', 'P', 'K'};
constexpr std::array<char, 8> kSnapshotKvarn6Magic{'Q', 'F', 'N', 'S',
                                                   'N', 'A', 'K', '6'};
constexpr std::array<char, 8> kSnapshotMagic{'Q', 'F', 'N', 'S',
                                             'N', 'A', 'P', '6'};

/// Fixed header ahead of the section bytes. It carries every geometry
/// value the section sizes derive from, so a payload of another artifact
/// or executor configuration is rejected before anything is copied.
struct SnapshotHeader {
  std::array<char, 8> magic;
  std::uint32_t num_layers;
  std::uint32_t full_attention_interval;
  std::uint32_t conv_elems;
  std::uint32_t state_elems;
  std::uint32_t kv_row;
  std::uint32_t indexer_head_dim;
  std::uint32_t compress_ratio;
  std::uint32_t hc_dim;
  std::uint32_t ple_elems;
  std::uint32_t has_mtp;
  std::uint32_t position;
  std::uint32_t blocks;
  std::uint32_t mtp_position;
  std::uint32_t mtp_blocks;
  std::uint32_t mtp_residual_valid;
  std::uint32_t hidden_rows;
  std::uint32_t image_count;
  // Rotated K depends on the rope scaling at every position.
  float rope_factor;
  std::uint32_t rope_original_context;
  float rope_beta_fast;
  float rope_beta_slow;
  std::array<std::int32_t, Config::kMaxPleNgram - 1> ngram_prev;
  std::uint64_t payload_bytes;
};
static_assert(std::is_trivially_copyable_v<SnapshotHeader>);
// The four rope fields must not push payload_bytes off its 8-byte alignment.
static_assert(offsetof(SnapshotHeader, payload_bytes) == 104);

namespace {

SnapshotHeader MakeSnapshotHeader(const Config& c, bool has_mtp,
                                  const Session& session) {
  SnapshotHeader h{};
  h.magic = session.kvarn() ? (session.kvarn_bits() == 6 ? kSnapshotKvarn6Magic
                                                         : kSnapshotKvarnMagic)
                            : kSnapshotMagic;
  h.num_layers = c.num_layers;
  h.full_attention_interval = c.full_attention_interval;
  h.conv_elems = (c.ssm_conv_kernel - 1) * c.SsmConvChannels();
  h.state_elems = c.ssm_num_v_heads * c.ssm_head_dim * c.ssm_head_dim;
  h.kv_row = c.AttentionKvDim();
  h.indexer_head_dim = c.indexer_head_dim;
  h.compress_ratio = c.compress_ratio;
  h.hc_dim = c.HcDim();
  h.ple_elems = c.ple_layer >= 0 ? c.PleConvHistory() * c.HcDim() : 0;
  h.has_mtp = has_mtp ? 1 : 0;
  h.position = session.position();
  h.image_count = std::ranges::count_if(
      session.VisionLayout().images,
      [&](const auto& image) { return image.offset < h.position; });
  // Disabled scaling has one disk-cache identity ("absolute-v1" from
  // RopePositionPolicy) regardless of whatever original_context/beta the
  // config happens to carry, so the header must normalize to the same
  // defaults rather than echo them raw (else two "off" configs that differ
  // only in those unused fields would collide on identity but disagree on
  // SameGeometry, and neither could restore the other's snapshot).
  const RopeScaling rope = NormalizedRopeScaling(c.rope_scaling);
  h.rope_factor = rope.factor;
  h.rope_original_context = rope.original_context;
  h.rope_beta_fast = rope.beta_fast;
  h.rope_beta_slow = rope.beta_slow;
  return h;
}

/// Whether `h` describes this executor's geometry (positions aside).
bool SameGeometry(const SnapshotHeader& h, const SnapshotHeader& mine) {
  return h.magic == mine.magic && h.num_layers == mine.num_layers &&
         h.full_attention_interval == mine.full_attention_interval &&
         h.conv_elems == mine.conv_elems && h.state_elems == mine.state_elems &&
         h.kv_row == mine.kv_row &&
         h.indexer_head_dim == mine.indexer_head_dim &&
         h.compress_ratio == mine.compress_ratio && h.hc_dim == mine.hc_dim &&
         h.ple_elems == mine.ple_elems && h.has_mtp == mine.has_mtp &&
         h.rope_factor == mine.rope_factor &&
         h.rope_original_context == mine.rope_original_context &&
         h.rope_beta_fast == mine.rope_beta_fast &&
         h.rope_beta_slow == mine.rope_beta_slow;
}

}  // namespace

template<typename Visit>
std::uint64_t Executor::WalkSnapshot(const SnapshotHeader& h,
                                     const Session* session, Visit&& visit) {
  std::uint64_t offset = sizeof(SnapshotHeader);
  const auto region = [&](void* device, std::uint64_t bytes, const char* what,
                          std::uint64_t row_bytes = 0, bool draft = false,
                          bool pooled = false) {
    if (bytes != 0 &&
        !visit(device, offset, bytes, what, row_bytes, draft, pooled)) {
      return false;
    }
    offset += bytes;
    return true;
  };
  const auto linear = [&](std::uint32_t il) -> const Session::LinearState* {
    return session != nullptr ? &session->linear_[il] : nullptr;
  };
  const auto attention =
      [&](std::uint32_t il) -> const Session::AttentionState* {
    return session != nullptr ? &session->attention_[il] : nullptr;
  };
  for (std::uint32_t il = 0; il < h.num_layers; ++il) {
    if (((il + 1) % h.full_attention_interval) != 0) {
      const auto* l = linear(il);
      if (!region(l != nullptr ? l->conv_state : nullptr,
                  std::uint64_t{h.conv_elems} * sizeof(float), "conv state") ||
          !region(l != nullptr ? l->state : nullptr,
                  std::uint64_t{h.state_elems} * sizeof(float),
                  "recurrent state")) {
        return 0;
      }
    }
  }
  if (!region(session != nullptr ? session->ple_history_ : nullptr,
              std::uint64_t{h.ple_elems} * sizeof(float), "PLE history")) {
    return 0;
  }
  const std::uint64_t kv_bytes =
      std::uint64_t{h.position} * h.kv_row * sizeof(__half);
  const std::uint64_t index_row_bytes =
      std::uint64_t{h.indexer_head_dim} * sizeof(float);
  const std::uint32_t index_begin = h.blocks * h.compress_ratio;
  const std::uint32_t index_rows = h.position - index_begin;
  const std::uint64_t block_bytes =
      std::uint64_t{h.blocks} * h.indexer_head_dim * sizeof(__half);
  for (std::uint32_t il = 0; il < h.num_layers; ++il) {
    if (((il + 1) % h.full_attention_interval) == 0) {
      const auto* at = attention(il);
      if (h.magic == kSnapshotKvarnMagic || h.magic == kSnapshotKvarn6Magic) {
        const int bits = h.magic == kSnapshotKvarn6Magic ? 6 : 4;
        // The sink, the records of every group before the position's group
        // floor (FinishKvarn keeps them quantized) and the FP16 tail rows
        // after it, all copied when captured: a rewind can requantize a
        // group and the ring is rewritten. Records lie in group order in
        // the session's private span (identity block table).
        const std::uint32_t base = std::max(
            hip::kKvarnSinkTokens,
            h.position >> hip::kKvarnGroupShift << hip::kKvarnGroupShift);
        const std::uint64_t row = std::uint64_t{h.kv_row} * sizeof(__half);
        const std::uint64_t sink =
            std::min(h.position, hip::kKvarnSinkTokens) * row;
        const std::uint64_t record =
            std::uint64_t{h.kv_row / hip::kKvarnHeadDim} *
            hip::KvarnRecord(bits, hip::KvarnValueOrder::kTokenMajor, bits)
                .bytes;
        const std::uint64_t records =
            std::uint64_t{(base >> hip::kKvarnGroupShift) - 1} * record;
        const std::uint64_t tail =
            h.position > base ? std::uint64_t{h.position - base} * row : 0;
        const std::size_t tail_row = hip::KvarnFp16Row(base) * h.kv_row;
        if (!region(at != nullptr ? at->kvarn_k : nullptr, sink,
                    "KVarN K sink") ||
            !region(at != nullptr ? at->kvarn_v : nullptr, sink,
                    "KVarN V sink") ||
            !region(at != nullptr ? at->kvarn_records + record : nullptr,
                    records, "KVarN records") ||
            !region(at != nullptr ? at->kvarn_k + tail_row : nullptr, tail,
                    "KVarN K tail") ||
            !region(at != nullptr ? at->kvarn_v + tail_row : nullptr, tail,
                    "KVarN V tail"))
          return 0;
      } else if (!region(at != nullptr ? at->k_cache : nullptr, kv_bytes,
                         "K cache", std::uint64_t{h.kv_row} * sizeof(__half)) ||
                 !region(at != nullptr ? at->v_cache : nullptr, kv_bytes,
                         "V cache", std::uint64_t{h.kv_row} * sizeof(__half))) {
        return 0;
      }
      // Serialize only unpooled rows, in chronological order. The physical
      // ring may wrap, and a destination session may have another capacity.
      if (session == nullptr) {
        if (!region(nullptr, index_rows * index_row_bytes, "indexer keys"))
          return 0;
      } else {
        const auto first = index_begin & (session->index_capacity_ - 1);
        const auto tail =
            std::min(index_rows, session->index_capacity_ - first);
        if (!region(at->index_k + std::size_t{first} * h.indexer_head_dim,
                    tail * index_row_bytes, "indexer tail") ||
            !region(at->index_k, (index_rows - tail) * index_row_bytes,
                    "indexer wrap"))
          return 0;
      }
      if (!region(at != nullptr ? at->block_k : nullptr, block_bytes,
                  "pooled block keys",
                  std::uint64_t{h.indexer_head_dim} * sizeof(__half), false,
                  true))
        return 0;
    }
  }
  if (h.has_mtp != 0) {
    const auto* mtp = session != nullptr ? &session->mtp_ : nullptr;
    const std::uint32_t begin = h.mtp_blocks * h.compress_ratio;
    const std::uint32_t rows = h.mtp_position - begin;
    if (session == nullptr) {
      if (!region(nullptr, rows * index_row_bytes, "draft indexer keys"))
        return 0;
    } else {
      const auto first = begin & (session->index_capacity_ - 1);
      const auto tail = std::min(rows, session->index_capacity_ - first);
      if (!region(mtp->index_k + std::size_t{first} * h.indexer_head_dim,
                  tail * index_row_bytes, "draft indexer tail") ||
          !region(mtp->index_k, (rows - tail) * index_row_bytes,
                  "draft indexer wrap"))
        return 0;
    }
    if (!region(
            mtp ? mtp->block_k : nullptr,
            std::uint64_t{h.mtp_blocks} * h.indexer_head_dim * sizeof(__half),
            "draft pooled keys",
            std::uint64_t{h.indexer_head_dim} * sizeof(__half), true, true))
      return 0;
    const std::uint64_t mtp_kv_bytes =
        std::uint64_t{h.mtp_position} * h.kv_row * sizeof(__half);
    if (!region(mtp != nullptr ? mtp->k_cache : nullptr, mtp_kv_bytes,
                "draft K cache", std::uint64_t{h.kv_row} * sizeof(__half),
                true) ||
        !region(mtp != nullptr ? mtp->v_cache : nullptr, mtp_kv_bytes,
                "draft V cache", std::uint64_t{h.kv_row} * sizeof(__half),
                true) ||
        !region(
            mtp != nullptr ? mtp->h : nullptr,
            h.mtp_residual_valid ? std::uint64_t{h.hc_dim} * sizeof(float) : 0,
            "draft residual") ||
        !region(mtp != nullptr ? mtp->target_hidden : nullptr,
                std::uint64_t{h.hidden_rows} * h.hc_dim * sizeof(float),
                "kept trunk rows")) {
      return 0;
    }
  }
  return offset +
         std::uint64_t{h.image_count} * sizeof(qwen::vision::ImageGrid);
}

std::uint64_t Executor::SnapshotBytes(const Session& session,
                                      std::uint32_t hidden_rows) const {
  SnapshotHeader h =
      MakeSnapshotHeader(config(), session.mtp_enabled_, session);
  h.blocks = session.blocks_;
  h.mtp_blocks = session.mtp_.blocks;
  h.mtp_position = session.mtp_.position;
  h.mtp_residual_valid = session.mtp_.residual_valid;
  h.hidden_rows = hidden_rows;
  return WalkSnapshot(h, nullptr,
                      [](void*, std::uint64_t, std::uint64_t, const char*,
                         std::uint64_t, bool, bool) { return true; });
}

bool Executor::SaveSnapshot(const Session& session, std::uint32_t hidden_rows,
                            std::span<std::uint8_t> payload,
                            std::string* error_msg,
                            std::shared_ptr<SnapshotState>* deferred,
                            std::shared_ptr<void> storage) const {
  if (session.kv_pool_) {
    AssignError(error_msg, "shared KV sessions use component snapshots");
    return false;
  }
  if (session.owner_ != this) {
    AssignError(error_msg, "session belongs to another executor");
    return false;
  }
  if (session.spec_tokens_ != 0) {
    AssignError(error_msg,
                "snapshot with a pending speculative batch; roll back first");
    return false;
  }
  if (hidden_rows > session.position_ ||
      hidden_rows > options_.max_speculative) {
    AssignError(error_msg, "kept trunk rows exceed the position or batch");
    return false;
  }
  SnapshotHeader h =
      MakeSnapshotHeader(config(), session.mtp_enabled_, session);
  h.blocks = session.blocks_;
  h.mtp_blocks = session.mtp_.blocks;
  h.mtp_position = session.mtp_.position;
  h.mtp_residual_valid = session.mtp_.residual_valid;
  h.hidden_rows = hidden_rows;
  h.ngram_prev = session.ngram_.prev;
  h.payload_bytes = SnapshotBytes(session, hidden_rows);
  if (payload.size() != h.payload_bytes) {
    AssignError(error_msg, "snapshot buffer size does not match the payload");
    return false;
  }
  std::shared_ptr<SnapshotState> pending;
  const auto transfer = GetSnapshotAllocator()->Transfer();
  if (deferred) {
    if (!storage) {
      AssignError(error_msg, "borrowed snapshot needs owned payload storage");
      return false;
    }
    if (!session.snapshots_)
      session.snapshots_ =
          std::make_shared<SnapshotRegistry>(GetSnapshotAllocator());
    pending = std::make_shared<SnapshotState>();
    pending->registry_ = session.snapshots_;
    if (!session.snapshot_lineage_)
      session.snapshot_lineage_ = std::make_shared<SnapshotLineage>();
    pending->lineage_ = session.snapshot_lineage_;
    pending->storage_ = std::move(storage);
    pending->payload_ = payload.data();
    std::uint64_t mutable_prefix = sizeof(SnapshotHeader);
    (void)WalkSnapshot(h, nullptr,
                       [&](void*, std::uint64_t offset, std::uint64_t bytes,
                           const char*, std::uint64_t row_bytes, bool, bool) {
                         if (row_bytes != 0)
                           return false;
                         mutable_prefix = offset + bytes;
                         return true;
                       });
    std::uint64_t extra_bytes = 0;
    (void)WalkSnapshot(h, nullptr,
                       [&](void*, std::uint64_t offset, std::uint64_t bytes,
                           const char*, std::uint64_t row_bytes, bool, bool) {
                         if (row_bytes == 0 && offset >= mutable_prefix)
                           extra_bytes += bytes;
                         return true;
                       });
    pending->mutable_offset_ = sizeof(SnapshotHeader);
    pending->mutable_bytes_ = mutable_prefix - sizeof(SnapshotHeader);
    try {
      pending->mutable_storage_ =
          session.snapshots_->allocator->AllocateSpare(pending->mutable_bytes_);
      pending->mutable_device_ =
          static_cast<std::uint8_t*>(pending->mutable_storage_.get());
      if (extra_bytes != 0) {
        pending->extra_storage_ =
            session.snapshots_->allocator->Allocate(extra_bytes);
        pending->extra_bytes_ = extra_bytes;
      }
    } catch (const SnapshotDeviceExhausted&) {
      // No device room to freeze this capture: copy it to its host payload.
      pending.reset();
    }
  }
  if (!pending)
    PopulateSnapshotPrefix(payload.data(), payload.size());
  std::memcpy(payload.data(), &h, sizeof(h));
  const auto grid_bytes =
      std::size_t{h.image_count} * sizeof(qwen::vision::ImageGrid);
  if (grid_bytes != 0) {
    std::memcpy(payload.data() + payload.size() - grid_bytes,
                session.VisionLayout().images.data(), grid_bytes);
  }
  std::uint64_t extra_used = 0;
  const bool saved =
      WalkSnapshot(
          h, &session,
          [&](void* device, std::uint64_t offset, std::uint64_t bytes,
              const char*, std::uint64_t row_bytes, bool draft, bool pooled) {
            if (!session.CheckCancellation(error_msg))
              return false;
            if (pending && row_bytes != 0) {
              pending->regions_.push_back({device,
                                           payload.data() + offset,
                                           bytes,
                                           bytes,
                                           bytes,
                                           row_bytes,
                                           draft,
                                           pooled,
                                           {}});
            } else {
              auto* destination = payload.data() + offset;
              auto kind = hipMemcpyDeviceToHost;
              if (pending && offset >= pending->mutable_offset_ &&
                  offset + bytes <=
                      pending->mutable_offset_ + pending->mutable_bytes_) {
                destination = pending->mutable_device_ + offset -
                              pending->mutable_offset_;
                kind = hipMemcpyDeviceToDevice;
              } else if (pending && pending->extra_storage_) {
                if (extra_used + bytes > pending->extra_bytes_)
                  throw std::logic_error("snapshot extra regions overflow");
                destination =
                    static_cast<std::uint8_t*>(pending->extra_storage_.get()) +
                    extra_used;
                kind = hipMemcpyDeviceToDevice;
                pending->extras_.push_back({offset, bytes, destination});
                extra_used += bytes;
              }
              if (pending)
                transfer->Enqueue(destination, device, bytes, kind);
              else
                transfer->Copy(destination, device, bytes);
            }
            return true;
          }) != 0;
  transfer->Finish();
  if (saved && pending) {
    std::lock_guard lock(session.snapshots_->mutex);
    std::erase_if(session.snapshots_->pending,
                  [](const auto& entry) { return entry.expired(); });
    session.snapshots_->pending.push_back(pending);
    session.snapshot_position_ =
        std::max(session.snapshot_position_, session.position_);
    session.snapshot_mtp_position_ =
        std::max(session.snapshot_mtp_position_, session.mtp_.position);
    *deferred = std::move(pending);
  }
  return saved;
}

std::uint64_t Executor::PrefillCheckpointBytes(const Session& session,
                                               std::uint32_t position) const {
  if (session.kvarn_ || session.kv_pool_)
    return 0;
  const auto& c = config();
  auto h = MakeSnapshotHeader(c, session.mtp_enabled_, session);
  h.position = position;
  h.blocks = c.compress_ratio > 0 && position > c.indexer_top_k
                 ? position / c.compress_ratio
                 : session.blocks_;
  if (session.mtp_enabled_) {
    h.mtp_position = position - 1;
    h.mtp_blocks = h.mtp_position > c.indexer_top_k
                       ? h.mtp_position / c.compress_ratio
                       : session.mtp_.blocks;
    h.hidden_rows =
        std::min(position - session.position_, options_.max_speculative);
  }
  return WalkSnapshot(h, nullptr,
                      [](void*, std::uint64_t, std::uint64_t, const char*,
                         std::uint64_t, bool, bool) { return true; });
}

bool Executor::RestoreSnapshot(Session& session,
                               std::span<const std::uint8_t> payload,
                               SnapshotInfo* info, std::string* error_msg,
                               std::uint32_t next_drafts,
                               const SnapshotState* deferred) const {
  if (session.kv_pool_) {
    AssignError(error_msg, "shared KV sessions use component snapshots");
    return false;
  }
  if (session.owner_ != this) {
    AssignError(error_msg, "session belongs to another executor");
    return false;
  }
  if (payload.size() < sizeof(SnapshotHeader)) {
    AssignError(error_msg, "snapshot payload is truncated");
    return false;
  }
  SnapshotHeader h{};
  std::memcpy(&h, payload.data(), sizeof(h));
  const SnapshotHeader mine =
      MakeSnapshotHeader(config(), session.mtp_enabled_, session);
  if (!SameGeometry(h, mine)) {
    AssignError(error_msg, "snapshot was taken with another model geometry");
    return false;
  }
  const std::uint32_t max_blocks =
      h.compress_ratio == 0 ? 0 : h.position / h.compress_ratio;
  if (h.image_count > h.position || h.position == 0 ||
      h.position > session.max_context_ || h.blocks > max_blocks ||
      h.mtp_position > h.position ||
      h.mtp_blocks > h.mtp_position / h.compress_ratio ||
      h.mtp_position - h.mtp_blocks * h.compress_ratio >
          session.index_capacity_ ||
      h.hidden_rows > h.position || h.hidden_rows > options_.max_speculative ||
      h.mtp_residual_valid > 1 ||
      (h.mtp_residual_valid != 0 && h.mtp_position == 0) ||
      (h.has_mtp == 0 && (h.mtp_position != 0 || h.hidden_rows != 0 ||
                          h.mtp_residual_valid != 0))) {
    AssignError(error_msg, "snapshot positions do not fit this session");
    return false;
  }
  if (h.position - h.blocks * h.compress_ratio > session.index_capacity_) {
    AssignError(error_msg, "unpooled indexer rows exceed the ring capacity");
    return false;
  }
  if (h.payload_bytes != payload.size() ||
      WalkSnapshot(h, nullptr,
                   [](void*, std::uint64_t, std::uint64_t, const char*,
                      std::uint64_t, bool, bool) { return true; }) !=
          payload.size()) {
    AssignError(error_msg, "snapshot payload size does not match its header");
    return false;
  }
  qwen::vision::RopeLayout layout;
  layout.images.resize(h.image_count);
  const auto grid_bytes =
      layout.images.size() * sizeof(qwen::vision::ImageGrid);
  if (grid_bytes != 0) {
    std::memcpy(layout.images.data(),
                payload.data() + payload.size() - grid_bytes, grid_bytes);
  }
  try {
    layout.Validate(session.max_context_);
  } catch (const std::exception& e) {
    AssignError(error_msg, e.what());
    return false;
  }
  if (!layout.images.empty() &&
      layout.Prefix(h.position) != session.VisionLayout().Prefix(h.position)) {
    AssignError(error_msg,
                "image snapshot layout does not match its prompt attachment");
    return false;
  }
  if (layout.images.empty())
    session.ConfigureVision(nullptr, nullptr, stream_);
  // The session's queued work targets buffers the copies overwrite.
  if (!Check(hipStreamSynchronize(stream_), "restore drain", error_msg)) {
    return false;
  }
  const bool borrowed = deferred && deferred->registry_ == session.snapshots_;
  std::unique_lock<std::mutex> snapshot_lock;
  std::unique_lock<std::mutex> source_lock;
  if (session.snapshots_)
    snapshot_lock =
        std::unique_lock(session.snapshots_->mutex, std::defer_lock);
  if (deferred && !borrowed)
    source_lock = std::unique_lock(deferred->registry_->mutex, std::defer_lock);
  if (snapshot_lock.mutex() && source_lock.mutex())
    std::lock(snapshot_lock, source_lock);
  else if (snapshot_lock.mutex())
    snapshot_lock.lock();
  else if (source_lock.mutex())
    source_lock.lock();
  if (session.snapshots_ && session.snapshots_->failure)
    std::rethrow_exception(session.snapshots_->failure);
  if (deferred && deferred->registry_->failure)
    std::rethrow_exception(deferred->registry_->failure);
  // A same-session restore retains the still-live prefix. Only detached rows
  // are uploaded. Save other checkpoints' overwritten suffixes first.
  std::uint32_t trunk_keep = 0;
  std::uint32_t draft_keep = 0;
  std::uint32_t trunk_blocks = 0;
  std::uint32_t draft_blocks = 0;
  if (borrowed) {
    trunk_keep = h.position;
    draft_keep = h.mtp_position;
    trunk_blocks = h.blocks;
    draft_blocks = h.mtp_blocks;
    for (const auto& region : deferred->regions_) {
      const auto rows =
          static_cast<std::uint32_t>(region.live_bytes / region.row_bytes);
      if (region.pooled) {
        if (region.draft)
          draft_blocks = std::min(draft_blocks, rows);
        else
          trunk_blocks = std::min(trunk_blocks, rows);
      } else {
        if (region.draft)
          draft_keep = std::min(draft_keep, rows);
        else
          trunk_keep = std::min(trunk_keep, rows);
      }
    }
  } else if (deferred) {
    const auto common = SnapshotLineage::Common(
        session.snapshot_lineage_, deferred->lineage_,
        {std::min(session.position_, h.position),
         std::min(session.mtp_.position, h.mtp_position),
         std::min(session.blocks_, h.blocks),
         std::min(session.mtp_.blocks, h.mtp_blocks)});
    trunk_keep = common[0];
    draft_keep = common[1];
    trunk_blocks = common[2];
    draft_blocks = common[3];
  }
  trunk_blocks = h.compress_ratio
                     ? std::min(trunk_blocks, trunk_keep / h.compress_ratio)
                     : 0;
  draft_blocks = h.compress_ratio
                     ? std::min(draft_blocks, draft_keep / h.compress_ratio)
                     : 0;
  auto restored_lineage =
      deferred ? std::make_shared<SnapshotLineage>(
                     deferred->lineage_,
                     SnapshotLineage::Rows{h.position, h.mtp_position, h.blocks,
                                           h.mtp_blocks})
               : nullptr;
  if (session.snapshots_) {
    session.snapshots_->Preserve(trunk_keep, draft_keep, trunk_blocks,
                                 draft_blocks);
  }
  // If restoration fails after a partial write, no provenance may survive.
  session.snapshot_lineage_.reset();
  // Every live state region is overwritten below. Retain only scratch that
  // the restored operation can use; this never allocates new rollback rows.
  // Explicit reset and failed restoration still release all scratch.
  const auto remaining = session.max_context_ - h.position;
  session.TrimRollback(std::min({next_drafts, options_.max_speculative - 1,
                                 remaining ? remaining - 1 : 0}));
  // A borrowed restore may reuse its source slot. Keep recorded graphs, but
  // let unrecorded shapes run eagerly once in the restored request before
  // recording them, just as they would in a fresh execution slot.
  std::erase_if(session.warmed_,
                [&](const auto key) { return !session.graphs_.contains(key); });
  std::unique_ptr<gufo::hip::SnapshotTransfer> fork_transfer;
  const auto upload = [&](void* device, const std::uint8_t* source,
                          std::uint64_t bytes, const char* what,
                          hipMemcpyKind kind = hipMemcpyHostToDevice) {
    if (!session.CheckCancellation(error_msg))
      return false;
    if (deferred) {
      if (!fork_transfer)
        fork_transfer = GetSnapshotAllocator()->Transfer();
      fork_transfer->Enqueue(device, source, bytes, kind);
      return true;
    }
    return Check(hipMemcpy(device, source, bytes, hipMemcpyHostToDevice), what,
                 error_msg);
  };

  if (WalkSnapshot(
          h, &session,
          [&](void* device, std::uint64_t offset, std::uint64_t bytes,
              const char* what, std::uint64_t row_bytes, bool draft,
              bool pooled) {
            const auto original_offset = offset;
            const auto original_bytes = bytes;
            if (deferred && row_bytes != 0) {
              auto rows = draft ? draft_keep : trunk_keep;
              if (pooled)
                rows = draft ? draft_blocks : trunk_blocks;
              const auto skip =
                  std::min(bytes, std::uint64_t{rows} * row_bytes);
              device = static_cast<std::uint8_t*>(device) + skip;
              offset += skip;
              bytes -= skip;
            }
            if (deferred && !borrowed && row_bytes != 0) {
              // Fork directly from the source's still-live prefix.
              // Its detached tail is already present in the payload.
              // Hold both registries until the copy completes so a
              // reset or destruction cannot invalidate either side.
              for (const auto& region : deferred->regions_) {
                if (region.host != payload.data() + original_offset)
                  continue;
                const auto skip = offset - original_offset;
                const auto copied =
                    region.live_bytes > skip
                        ? std::min(bytes, region.live_bytes - skip)
                        : 0;
                if (copied != 0) {
                  if (!session.CheckCancellation(error_msg))
                    return false;
                  if (!fork_transfer)
                    fork_transfer = GetSnapshotAllocator()->Transfer();
                  fork_transfer->Enqueue(
                      device,
                      static_cast<const std::uint8_t*>(region.device) + skip,
                      copied, hipMemcpyDeviceToDevice);
                }
                device = static_cast<std::uint8_t*>(device) + copied;
                offset += copied;
                bytes -= copied;
                break;
              }
            }
            if (bytes == 0)
              return true;
            if (deferred && row_bytes != 0) {
              for (const auto& region : deferred->regions_) {
                if (region.host != payload.data() + original_offset ||
                    region.saved.empty())
                  continue;
                const auto skip = offset - original_offset;
                for (const auto& saved : region.saved) {
                  const auto begin = std::max(skip, saved.offset);
                  const auto end =
                      std::min(original_bytes, saved.offset + saved.bytes);
                  if (begin < end) {
                    if (!session.CheckCancellation(error_msg))
                      return false;
                    if (!fork_transfer)
                      fork_transfer = GetSnapshotAllocator()->Transfer();
                    fork_transfer->Enqueue(
                        static_cast<std::uint8_t*>(device) + begin - skip,
                        saved.chunk->device + begin - saved.offset, end - begin,
                        hipMemcpyDeviceToDevice);
                  }
                }
                return true;
              }
            }
            const auto* source = payload.data() + offset;
            auto kind = hipMemcpyHostToDevice;
            if (deferred && deferred->mutable_storage_ &&
                offset >= deferred->mutable_offset_ &&
                offset + bytes <=
                    deferred->mutable_offset_ + deferred->mutable_bytes_) {
              source = deferred->mutable_device_ + offset -
                       deferred->mutable_offset_;
              kind = hipMemcpyDeviceToDevice;
            } else if (deferred && deferred->extra_storage_ && row_bytes == 0) {
              // The destination may split a ring region differently from the
              // source, so copy every overlapping frozen piece.
              auto* target = static_cast<std::uint8_t*>(device);
              for (const auto& extra : deferred->extras_) {
                const auto begin = std::max(offset, extra.offset);
                const auto end =
                    std::min(offset + bytes, extra.offset + extra.bytes);
                if (begin < end &&
                    !upload(target + (begin - offset),
                            extra.device + (begin - extra.offset), end - begin,
                            what, hipMemcpyDeviceToDevice))
                  return false;
              }
              return true;
            }
            return upload(device, source, bytes, what, kind);
          }) == 0) {
    if (fork_transfer)
      fork_transfer->Finish();
    if (snapshot_lock.owns_lock())
      snapshot_lock.unlock();
    session.Reset();
    return false;
  }
  if (fork_transfer)
    fork_transfer->Finish();
  session.RestoreVisionLayout(layout, stream_);
  session.position_ = h.position;
  session.blocks_ = h.blocks;
  if (session.kvarn_) {
    // The payload's records end at the group floor and its ring rows start
    // there, so a rewind before it must be refused: written_end is raised
    // past the ring's reach of the group before (hip::PlanKvarnPass).
    const std::uint32_t base =
        std::max(hip::kKvarnSinkTokens,
                 h.position >> hip::kKvarnGroupShift << hip::kKvarnGroupShift);
    session.kvarn_tail_ = {
        .flushed_end = base,
        .written_end =
            base == hip::kKvarnSinkTokens
                ? h.position
                : std::max(h.position, base + hip::kKvarnTailRingTokens -
                                           hip::kKvarnGroupTokens + 1)};
    session.kvarn_pass_ = {};
  }
  session.mtp_.position = h.mtp_position;
  session.mtp_.blocks = h.mtp_blocks;
  session.mtp_.residual_valid = h.mtp_residual_valid != 0;
  session.snapshot_lineage_ = std::move(restored_lineage);
  if (borrowed)
    for (auto& region : deferred->regions_)
      region.live_bytes = region.bytes;
  session.snapshot_position_ = borrowed ? h.position : trunk_keep;
  session.snapshot_mtp_position_ = borrowed ? h.mtp_position : draft_keep;
  session.spec_base_ = h.position;
  session.spec_tokens_ = 0;
  session.ngram_.prev = h.ngram_prev;
  if (info != nullptr) {
    info->position = h.position;
    info->hidden_rows = h.hidden_rows;
  }
  return true;
}

bool Executor::ReplayOutputHead(std::uint32_t batch_rows,
                                std::span<float> logits,
                                std::string* error_msg) const {
  const auto& c = config();
  const auto rows = logits.size() / c.vocab_size;
  if (batch_rows == 0 || batch_rows > options_.max_batch || rows == 0 ||
      rows > batch_rows || rows > options_.max_logit_rows ||
      logits.size() % c.vocab_size != 0) {
    AssignError(error_msg, "invalid output head replay rows");
    return false;
  }
  DropReadyQ8();
  q8t_src_ = nullptr;
  half_src_ = nullptr;
  PrefillPhase head_phase(false);
  return HcMix(model_->hc_head(), s_.res + (batch_rows - rows) * c.HcDim(),
               false, s_.mixed, nullptr, static_cast<std::uint32_t>(rows),
               error_msg) &&
         Dense(model_->output(), s_.mixed, s_.logits,
               static_cast<std::uint32_t>(rows), error_msg) &&
         Check(hipMemcpyAsync(logits.data(), s_.logits, logits.size_bytes(),
                              hipMemcpyDeviceToHost, stream_),
               "output head replay download", error_msg) &&
         Check(hipStreamSynchronize(stream_), "output head replay", error_msg);
}

bool Executor::ReadVerificationRows(std::uint32_t row, std::span<float> logits,
                                    std::string* error_msg) const {
  const auto vocab = config().vocab_size;
  if (row >= options_.max_speculative || logits.empty() ||
      logits.size() % vocab != 0 ||
      logits.size() / vocab > options_.max_speculative - row) {
    AssignError(error_msg, "invalid verification row");
    return false;
  }
  return Check(hipMemcpyAsync(logits.data(), VerificationLogits() + row * vocab,
                              logits.size_bytes(), hipMemcpyDeviceToHost,
                              stream_),
               "constrained verification row download", error_msg) &&
         Check(hipStreamSynchronize(stream_), "constrained verification",
               error_msg);
}

bool Executor::GreedyMtpPredictions(std::span<ArgmaxCandidate> predictions,
                                    const sampling::SamplerState& sampler,
                                    std::span<const std::int32_t> drafts,
                                    std::string* error_msg) const {
  const auto rows = predictions.size();
  if (rows == 0 || rows > options_.max_logit_rows || rows > 7 ||
      drafts.size() != rows || s_.mtp_ids == nullptr) {
    AssignError(error_msg, "invalid greedy MTP verification request");
    return false;
  }
  // Proposal selection has finished. Its two ID buffers and argmax scratch
  // can be reused until the next draft head overwrites them.
  const auto vocab = config().vocab_size;
  // Keep the host upload alive until the final stream synchronization. This
  // workspace is shared, never part of a session or its persistent snapshot.
  std::vector<sampling::TokenPenalty> penalties;
  if (sampler.config().penalties_enabled()) {
    penalties.reserve(rows * (sampler.penalties().size() + rows));
    // Drafts may violate the target grammar. Only their conditional penalty
    // history is needed here; FinishDecode checks each winner's grammar.
    auto tentative = sampler.WithoutConstraint();
    GreedyPenaltyRows batch{};
    for (std::size_t row = 0; row < rows; ++row) {
      const auto counts = tentative.penalties();
      penalties.insert(penalties.end(), counts.begin(), counts.end());
      batch.offsets[row + 1] = static_cast<std::uint32_t>(penalties.size());
      if (row + 1 < rows)
        tentative.Accept(static_cast<sampling::TokenId>(drafts[row]));
    }
    if (penalties.size() > verification_penalty_capacity_) {
      const auto capacity =
          std::max(penalties.size(), 2 * verification_penalty_capacity_);
      sampling::TokenPenalty* allocated = nullptr;
      if (!Check(hip::AllocateDevice(&allocated, capacity * sizeof(*allocated)),
                 "verification penalty allocation", error_msg))
        return false;
      (void)hipFree(verification_penalties_);
      verification_penalties_ = allocated;
      verification_penalty_capacity_ = capacity;
    }
    batch.penalties = verification_penalties_;
    if (!penalties.empty() &&
        !Check(hipMemcpyAsync(verification_penalties_, penalties.data(),
                              penalties.size() * sizeof(penalties.front()),
                              hipMemcpyHostToDevice, stream_),
               "verification penalty upload", error_msg))
      return false;
    const auto& config = sampler.config();
    PenalizedArgmax(
        VerificationLogits(), batch, config.repeat_penalty,
        config.frequency_penalty, config.presence_penalty,
        reinterpret_cast<PenaltyArgmaxCandidate*>(s_.mtp_scratch_ids),
        s_.mtp_argmax, rows, vocab, stream_);
  } else {
    gufo::hip::LaunchBatchedGPUArgmax(
        VerificationLogits(), s_.mtp_ids, rows, vocab,
        {reinterpret_cast<float*>(s_.mtp_scratch_ids),
         MtpCandidateWorkspaceSize(vocab)},
        stream_);
    GatherArgmaxCandidates(VerificationLogits(), s_.mtp_ids, s_.mtp_argmax,
                           static_cast<std::uint32_t>(rows), vocab, stream_);
  }
  const bool copied = Check(
      hipMemcpyAsync(predictions.data(), s_.mtp_argmax,
                     predictions.size_bytes(), hipMemcpyDeviceToHost, stream_),
      "greedy MTP predictions download", error_msg);
  // Drain even after a failed download before releasing the upload source.
  const bool completed = Check(hipStreamSynchronize(stream_),
                               "greedy MTP verification", error_msg);
  return copied && completed;
}

bool Executor::CanCompactMtpVerification(
    const sampling::SamplerState& sampler) {
  const auto& config = sampler.config();
  return config.uses_random_sampling() && config.top_k > 0 &&
         std::size_t(config.top_k) <= kMtpCandidates &&
         config.min_keep <= kMtpCandidates &&
         sampler.penalties().size() + 8 <= kCompactPenaltyLimit &&
         !sampler.NeedsConstraintMask();
}

bool Executor::SampledMtpCandidates(
    std::span<std::span<const sampling::LogitCandidate>> output,
    const sampling::SamplerState& sampler, std::span<const std::int32_t> drafts,
    std::string* error_msg) const {
  const auto rows = output.size();
  if (rows == 0 || rows > 7 || rows >= options_.max_speculative ||
      rows != drafts.size() || verification_ids_ == nullptr ||
      !CanCompactMtpVerification(sampler)) {
    AssignError(error_msg, "invalid compact MTP verification request");
    return false;
  }
  auto& penalties = verification_penalty_host_;
  penalties.clear();
  GreedyPenaltyRows batch{};
  auto tentative = sampler.WithoutConstraint();
  for (std::size_t row = 0; row < rows; ++row) {
    const auto counts = tentative.penalties();
    penalties.insert(penalties.end(), counts.begin(), counts.end());
    batch.offsets[row + 1] = penalties.size();
    if (row + 1 < rows)
      tentative.Accept(static_cast<sampling::TokenId>(drafts[row]));
  }
  batch.penalties = verification_penalties_;
  if (!penalties.empty() &&
      !Check(hipMemcpyAsync(verification_penalties_, penalties.data(),
                            penalties.size() * sizeof(penalties[0]),
                            hipMemcpyHostToDevice, stream_),
             "compact verification penalties", error_msg))
    return false;
  VerificationCandidates(VerificationLogits(), batch, verification_ids_,
                         verification_scratch_ids_, verification_candidates_,
                         rows, config().vocab_size, stream_);
  const auto count = rows * kMtpCandidates + penalties.size();
  const bool copied = Check(
      hipMemcpyAsync(verification_candidates_host_, verification_candidates_,
                     count * sizeof(sampling::LogitCandidate),
                     hipMemcpyDeviceToHost, stream_),
      "compact verification candidates", error_msg);
  const bool completed =
      Check(hipStreamSynchronize(stream_), "compact verification", error_msg);
  if (!copied || !completed)
    return false;
  for (std::size_t row = 0; row < rows; ++row)
    output[row] = {
        verification_candidates_host_ + row * kMtpCandidates +
            batch.offsets[row],
        kMtpCandidates + batch.offsets[row + 1] - batch.offsets[row]};
  return true;
}

bool Executor::MtpForward(Session& session,
                          std::span<const std::int32_t> tokens,
                          std::int32_t hidden_row, MtpOutput output,
                          std::string* error_msg,
                          const float* hidden_source) const {
  DiagnosticPhase diagnostic(*this, hidden_row >= 0 ? DiagnosticStage::kCatchUp
                                                    : DiagnosticStage::kDraft);
  DropReadyQ8();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  if (session.owner_ != this ||
      std::any_of(tokens.begin(), tokens.end(), [&](auto t) {
        return t < 0 || static_cast<std::uint32_t>(t) >= config().vocab_size;
      })) {
    AssignError(error_msg, "invalid MTP session or token");
    return false;
  }
  if (!session.CheckCancellation(error_msg))
    return false;
  if (!session.mtp_enabled_) {
    AssignError(error_msg, "no MTP block loaded");
    return false;
  }

  if (n == 0 || n > options_.max_batch || (hidden_row < 0 && n != 1) ||
      (hidden_row >= 0 &&
       static_cast<std::uint32_t>(hidden_row) + n >
           (hidden_source ? options_.max_batch : options_.max_speculative))) {
    AssignError(error_msg, "MTP batch outside the kept hidden rows");
    return false;
  }
  if (output.trace != nullptr &&
      !output.trace->Valid(config().hidden_size, config().HcDim())) {
    AssignError(error_msg, "invalid MTP trace destinations");
    return false;
  }
  if ((hidden_row < 0 && !session.mtp_.residual_valid) ||
      (output.kv_only &&
       (hidden_row < 0 || output.token != nullptr ||
        output.candidates != nullptr || output.trace != nullptr))) {
    AssignError(error_msg, "MTP residual requires a full known-hidden forward");
    return false;
  }
  const std::uint32_t pos = session.mtp_.position;
  if (pos + n > session.max_context_) {
    AssignError(error_msg, "MTP context is full");
    return false;
  }
  session.EnsureKvRows(pos + n);
  if (session.continuation_hooks_)
    session.continuation_hooks_->BeforeWrite(
        true, pos, pos + n, session.mtp_.blocks,
        pos + n > config().indexer_top_k ? (pos + n) / config().compress_ratio
                                         : session.mtp_.blocks);
  session.PreserveSnapshots(session.position_, session.mtp_.position);
  ++session.mutation_epoch_;
  session.mtp_.residual_valid = false;
  std::copy(tokens.begin(), tokens.end(), tokens_host_);
  control_host_->position = session.position_;
  control_host_->blocks = session.blocks_;
  control_host_->mtp_position = pos;
  control_host_->hidden_row = hidden_row;
  control_host_->mtp_blocks = session.mtp_.blocks;
  const auto& c = config();
  const bool sparse = pos + n > c.indexer_top_k;
  const auto complete = (pos + n) / c.compress_ratio;
  const auto pool = sparse ? complete - session.mtp_.blocks : 0;
  const auto graph_pool = n / c.compress_ratio + 1;
  const bool graph = !diagnostics_active_ && output.trace == nullptr &&
                     hidden_source == nullptr && n <= kVecBatch &&
                     pool <= graph_pool &&
                     pos + 1 >= session.VisionLayout().PrefixLength();
  // Captured graphs bake in the head; key on the one this draft reads.
  const bool subset = UseDraftSubset(session, output.draft_subset) &&
                      (output.token != nullptr || output.candidates != nullptr);
  const std::uint64_t key =
      static_cast<std::uint64_t>(n) |
      (static_cast<std::uint64_t>(hidden_row < 0) << 32) |
      (std::uint64_t{1} << 40) |
      (std::uint64_t{output.token != nullptr} << 41) |
      (std::uint64_t{output.candidates != nullptr} << 43) |
      (std::uint64_t{sparse} << 44) | (std::uint64_t{output.kv_only} << 45) |
      (std::uint64_t{subset} << 46);
  const auto body = [&] {
    return MtpBody(session, n, pos, output.token != nullptr,
                   output.candidates != nullptr, error_msg,
                   graph ? graph_pool : pool, hidden_source, output.trace,
                   output.kv_only, subset);
  };
  if (!Run(session, key, graph, body, error_msg)) {
    return false;
  }
  if (output.token != nullptr) {
    *output.token = *mtp_token_host_;
  }
  if (output.candidates != nullptr) {
    *output.candidates = *mtp_candidates_host_;
  }
  session.mtp_.position = pos + n;
  if (diagnostics_active_ && hidden_row >= 0)
    cycle_diagnostics_.catch_up_rows += n;
  session.mtp_.residual_valid = !output.kv_only;
  if (sparse)
    session.mtp_.blocks = complete;
  return true;
}

bool Executor::MtpBody(Session& session, std::uint32_t n, std::uint32_t pos,
                       bool token, bool candidates, std::string* error_msg,
                       std::uint32_t pool_grid, const float* hidden_source,
                       MtpTrace* trace, bool kv_only, bool draft_subset) const {
  const Config& c = config();
  const DeviceLayer& l = model_->mtp();
  const std::uint32_t hc_dim = c.HcDim();
  const std::size_t final_row = static_cast<std::size_t>(n - 1) * hc_dim;
  const auto copy_trace = [&](const float* source,
                              std::span<float> destination) {
    return destination.empty() ||
           Check(hipMemcpyAsync(destination.data(), source,
                                destination.size_bytes(), hipMemcpyDeviceToHost,
                                stream_),
                 "MTP trace download", error_msg);
  };
  if (!Check(hipMemcpyAsync(session.control_, control_host_,
                            sizeof(Session::Control), hipMemcpyHostToDevice,
                            stream_),
             "control upload", error_msg) ||
      !Check(hipMemcpyAsync(s_.tokens, tokens_host_, n * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream_),
             "token upload", error_msg)) {
    return false;
  }
  EmbedTokens(model_->token_embd().data, SmallType(model_->token_embd().type),
              s_.tokens, s_.mtp_embd, n, c.hidden_size, 1, stream_);
  // MTP embeds shifted token IDs. Visual information arrives in the trunk
  // hidden stream; image embeddings belong only to the target input.
  RmsNormRows(s_.mtp_embd, l.nextn_enorm.f32(), s_.mtp_embd, n, c.hidden_size,
              1, c.rms_eps, stream_);
  // The hidden input: kept trunk rows from `hidden_row`, or the block's
  // own carried residual.
  const float* hidden = s_.mtp_h;
  if (hidden_source && control_host_->hidden_row >= 0) {
    // Eager catch-up can normalize the trunk rows directly. Graphs use the
    // device-selected source below because their hidden-row index can change
    // on replay. Neither path modifies the trunk residual.
    hidden = hidden_source +
             static_cast<std::size_t>(control_host_->hidden_row) * hc_dim;
  } else {
    MtpHidden(hidden_source ? hidden_source : session.mtp_.target_hidden,
              session.mtp_.h, &session.control_->hidden_row, s_.mtp_h, n,
              hc_dim, stream_);
  }
  // Unlike the HC mixers, this normalization spans the complete HC * H row.
  RmsNormRows(hidden, l.nextn_hnorm.f32(), s_.mtp_h, n, hc_dim, 1, c.rms_eps,
              stream_);
  if (trace && !copy_trace(s_.mtp_h + final_row, trace->normalized_hidden))
    return false;
  // These projections determine persistent draft KV. Use the same row-wise
  // activation quantization and reduction for prompt chunks and decode;
  // changing between F16 GEMM and Q8 GEMV changes sampled replay after reuse.
  if (!DenseBatch(l.nextn_fc_embedding, s_.mtp_embd, s_.mtp_eproj, n,
                  error_msg) ||
      !DenseBatch(l.nextn_fc_hidden, s_.mtp_h, s_.mtp_res, n * c.hc_count,
                  error_msg)) {
    return false;
  }
  MtpAddEmbedding(s_.mtp_eproj, s_.mtp_res, n, c.hidden_size, c.hc_count,
                  stream_);
  if (trace && !copy_trace(s_.mtp_res + final_row, trace->fused))
    return false;
  Session::AttentionState attn;
  attn.rope = session.vision_input_.rope();
  attn.k_cache = session.mtp_.k_cache;
  attn.v_cache = session.mtp_.v_cache;
  attn.index_k = session.mtp_.index_k;
  attn.block_k = session.mtp_.block_k;
  attn.kv_blocks = session.mtp_.kv_blocks;
  // The draft block's attention runs at its own position.
  // Catch-up exports KV for every row but only carries its final residual.
  // Preserve that row's original query tile; earlier attention results
  // never contribute to the carried state.
  const bool last_only =
      !token && !candidates && n > 32 && control_host_->hidden_row >= 0;
  // Keep the final 128-column tile (and its predecessor for short tails).
  // At least 96 rows retain the wide mixer/projection arithmetic. Attention
  // still writes every KV/indexer row before the scratch view is narrowed.
  // Short catch-up also carries only its final row. Keep two rows so the
  // router and projections retain their multi-row arithmetic; the single
  // row route has a different floating-point reduction.
  const auto skipped =
      control_host_->hidden_row >= 0 && n > 2 && n <= kVecBatch
          ? n - 2
          : (last_only && n >= 224 ? (n - 96) / 128 * 128 : 0U);
  const bool cache_only =
      kv_only && (!l.attn_qkv.empty() ||
                  (l.attn_k.exl3 != nullptr && l.attn_v.exl3 != nullptr));
  if (!HcMixBatch(l.hc_attn, s_.mtp_res, false, s_.mixed,
                  cache_only ? nullptr : s_.inject, n, error_msg)) {
    return false;
  }
  if (cache_only) {
    if (diagnostics_active_)
      cycle_diagnostics_.cache_only_rows += n;
    // Known trunk rows replace the predictor residual on the next forward.
    // Only K/V and pooled indexer keys survive this catch-up.
    if (!MtpKvProjection(l, s_.mixed, n, error_msg) ||
        !StoreMtpKv(l, session, n, yarn_, error_msg)) {
      return false;
    }
    const auto capacity =
        IndexerCapacity(c, options_.max_batch, session.max_context_);
    if (!DenseBatch(l.indexer_k, s_.mixed, s_.ik, n, error_msg))
      return false;
    StoreRows(s_.ik, attn.index_k, n, c.indexer_head_dim,
              &session.control_->mtp_position, capacity, stream_);
    if (pos + n > c.indexer_top_k) {
      PoolIndexerBlocks(attn.index_k, l.indexer_k_norm.f32(), attn.block_k,
                        &session.control_->mtp_blocks,
                        &session.control_->mtp_position, n, pool_grid,
                        c.compress_ratio, c.indexer_head_dim, c.rotary_dim,
                        c.rope_theta, c.rms_eps, capacity, stream_, attn.rope,
                        yarn_, attn.kv_blocks);
    }
    return true;
  }
  if (!l.attn_qkv.empty()) {
    if (!DenseBatch(l.attn_qkv, s_.mixed, s_.qg, n, error_msg))
      return false;
  } else if (l.attn_q.exl3 != nullptr) {
    if (!Exl3Group({&l.attn_q, &l.attn_k, &l.attn_v}, {s_.qg, s_.k, s_.v},
                   s_.mixed, n, error_msg))
      return false;
  } else if (!DenseBatch(l.attn_q, s_.mixed, s_.qg, n, error_msg) ||
             !DenseBatch(l.attn_k, s_.mixed, s_.k, n, error_msg) ||
             !DenseBatch(l.attn_v, s_.mixed, s_.v, n, error_msg)) {
    return false;
  }
  if (!Attention(l, attn, s_.mixed, s_.block_out, n,
                 &session.control_->mtp_position, &session.control_->mtp_blocks,
                 pos, pool_grid, session.max_context_,
                 pos + n > c.indexer_top_k, error_msg, last_only, true,
                 skipped == 0)) {
    return false;
  }
  struct RestoreScratch {
    const Executor* executor;
    Scratch scratch;
    bool changed;
    ~RestoreScratch() {
      if (changed)
        executor->UseScratch(scratch);
    }
  } restore{this, s_, skipped != 0};
  const auto tail_rows = n - skipped;
  const auto tail_last = static_cast<std::size_t>(tail_rows - 1) * hc_dim;
  if (skipped != 0) {
    auto tail = RowScratch(s_, skipped);
    // Keep the producer's inject stride: it may use vectorized partials or
    // a single quantized projection rather than RowScratch's scalar layout.
    tail.inject = s_.inject + static_cast<std::size_t>(skipped) * c.hc_count *
                                  inject_parts_;
    UseScratch(tail);
    if (!Dense(l.attn_out, s_.ctx, s_.block_out, tail_rows, error_msg))
      return false;
  }
  Combine(s_.mtp_res, l.hc_ffn.norm.f32(), tail_rows);
  if (trace && !copy_trace(s_.mtp_res + tail_last, trace->attention))
    return false;
  if (!session.CheckCancellation(error_msg))
    return false;
  const DeviceTensor* const shared_rotate[] = {&l.shexp_gate, &l.shexp_up};
  if (!HcMix(l.hc_ffn, s_.mtp_res, true, s_.mixed, s_.inject, tail_rows,
             error_msg, false, false, shared_rotate) ||
      (trace && !copy_trace(s_.mixed + static_cast<std::size_t>(tail_rows - 1) *
                                           c.hidden_size,
                            trace->ffn_input))) {
    return false;
  }
  if (skipped != 0 && tail_rows == 2)
    exl3_rotated_ = {};  // row 0 is rewritten below; its slab row is stale
  if (skipped != 0 && tail_rows == 2 &&
      !Check(hipMemcpyAsync(s_.mixed, s_.mixed + c.hidden_size,
                            c.hidden_size * sizeof(float),
                            hipMemcpyDeviceToDevice, stream_),
             "MTP tail padding", error_msg)) {
    return false;
  }
  // The discarded padding row shares the final row's routed experts, so
  // preserving multi-row arithmetic does not reread another expert set.
  if (!Moe(l, s_.mixed, s_.block_out, tail_rows, error_msg, last_only)) {
    return false;
  }
  Combine(s_.mtp_res, nullptr, tail_rows);
  if (trace &&
      !copy_trace(s_.block_out +
                      static_cast<std::size_t>(tail_rows - 1) * c.hidden_size,
                  trace->ffn_output))
    return false;
  if (trace && !copy_trace(s_.mtp_res + tail_last, trace->hidden))
    return false;
  if (trace && !trace->head.empty()) {
    if (!HcMix(l.nextn_head, s_.mtp_res + tail_last, false, s_.mixed, nullptr,
               1, error_msg) ||
        !copy_trace(s_.mixed, trace->head))
      return false;
  }
  const float* last = s_.mtp_res + tail_last;
  if (!Check(hipMemcpyAsync(session.mtp_.h, last, hc_dim * sizeof(float),
                            hipMemcpyDeviceToDevice, stream_),
             "MTP hidden carry", error_msg)) {
    return false;
  }
  return (!token && !candidates) ||
         MtpHead(l.nextn_head, last, token, candidates,
                 draft_subset ? &session : nullptr, error_msg);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
