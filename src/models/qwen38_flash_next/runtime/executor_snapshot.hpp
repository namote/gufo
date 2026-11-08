#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_RUNTIME_EXECUTOR_SNAPSHOT_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_RUNTIME_EXECUTOR_SNAPSHOT_HPP_

// Private snapshot storage of the executor: the pinned allocator, lineage and
// registry that executor_snapshot.cpp implements and executor.cpp releases
// and sizes with each session.

#include "src/models/qwen38_flash_next/runtime/executor_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

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

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_RUNTIME_EXECUTOR_SNAPSHOT_HPP_
