#ifndef GUFO_CACHE_STREAMING_HPP_
#define GUFO_CACHE_STREAMING_HPP_

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <span>
#include <stop_token>
#include <string>
#include <vector>

#include "src/cache/adapter.hpp"
#include "src/cache/disk.hpp"

namespace gufo::cache {
class BorrowedRows;
inline constexpr std::size_t kDefaultDiskStagingBytes = 1ULL << 20;
// Pinned room for device-read pieces queued ahead of a disk write. A piece
// copied from a busy device waits behind the model work queued before it (a
// peer's decode round); queuing many lets one such wait drain them all.
inline constexpr std::size_t kPersistenceReadAheadBytes = 48ULL << 20;
// Transfer streams a disk write leaves free in the pool. Restores and
// captures fail when they find no stream, and two slots' idle spills may hold
// a 16-wide batch each, so keep both batches plus one more free. The
// server's 96-stream pool, less two slot preservation leases and this
// reserve, leaves a write more streams than the 48 pieces (48 MiB) of
// read-ahead it queues: about 64 decode rounds, ~3.4 s at ~54 ms per round,
// for a 3.2 GB state.
inline constexpr std::size_t kReservedTransferStreams = 2 * 16 + 1;
// Longest a disk write waits for foreground transfers to leave a stream it
// may lease before it fails.
inline constexpr std::chrono::seconds kWriteStreamWaitLimit{10};
inline constexpr std::chrono::milliseconds kPersistenceYieldLimit{500};
// Restores and observed-row captures move whole rows per staging piece; a
// smaller staging budget can never restore a checkpoint of these components.
[[nodiscard]] inline std::size_t MinimumStagingBytes(
    std::span<const ComponentDescriptor> components) {
  std::size_t bytes = 1;
  for (const auto& d : components)
    if (d.kind == ComponentKind::kAppendRows)
      bytes = std::max(bytes, d.row_bytes);
  return bytes;
}
struct StagingAllocation {
  std::shared_ptr<void> owner;
  std::span<std::uint8_t> bytes;
};
using StagingAllocator = std::function<StagingAllocation(std::size_t)>;
using StreamLease = std::function<std::unique_ptr<Stream>()>;
// Leases a stream only while at least `keep_free` others stay free in the same
// pool, atomically; null otherwise.
using ReserveStreamLease =
    std::function<std::unique_ptr<Stream>(std::size_t keep_free)>;

struct TransferTiming {
  std::uint64_t allocator_ns{}, stream_acquire_ns{}, stream_wait_ns{};
  std::uint64_t metadata_lock_ns{}, filesystem_ns{}, fsync_ns{}, checksum_ns{};
  std::uint64_t yield_ns{}, bytes{}, pieces{};
  std::uint64_t staging_wait_ns{}, io_lock_ns{};
  // Why a queued job did not publish (PersistenceQueue completion only):
  // "superseded", "stopped", "disk_capacity" or the write error.
  std::string failure;
};
// Sources retain immutable storage, assigned spill backing or discardable
// observations. Backed borrowers wait for preservation before disk work
// unless they read live pieces; live reads and observations pin one native
// piece until its leased transfer completes.
// Capture supplies private state/tails before submitting a job to the queue.
class PersistenceSource {
public:
  using Copy =
      std::function<Completion(std::uint64_t, std::span<std::byte>, Stream&)>;
  PersistenceSource(bool private_file, DiskFileId, std::uint64_t bytes,
                    std::size_t alignment, ResourceCharge,
                    std::shared_ptr<const void> owner, Copy);
  using CopyRows = std::function<Completion(std::span<const std::byte>,
                                            std::span<std::byte>, Stream&)>;
  // Backed borrowed rows. Without read_live, the writer waits until slot
  // preservation has copied them to their backing. With it, a piece still in
  // the slot is read from there under a reader pin held through completion,
  // so a busy or reused slot cannot hold the job (and its pins) back.
  PersistenceSource(bool private_file, DiskFileId, std::uint64_t bytes,
                    std::size_t alignment, std::shared_ptr<BorrowedRows>,
                    CopyRows read_live = {});
  // Discardable live rows, for histories exceeding RAM. Reads pin exactly
  // one staging piece through completion; mutation retires unread sources.
  // Private state/tails remain immutable committed copies. No publication
  // completes with unread retired rows. No full-history backing is assigned.
  PersistenceSource(bool private_file, DiskFileId, std::uint64_t bytes,
                    std::size_t alignment, std::shared_ptr<BorrowedRows>,
                    std::size_t first_byte, CopyRows);
  // Rows retained in device storage (Payload::Device). The owner keeps them
  // unchanged through the job; copy reads each piece into staging. No RAM
  // backing is charged or pinned.
  PersistenceSource(bool private_file, DiskFileId, std::uint64_t bytes,
                    std::size_t alignment, std::shared_ptr<const void> owner,
                    Copy);
  PersistenceSource(PersistenceSource&&) noexcept = default;
  PersistenceSource& operator=(PersistenceSource&&) = delete;
  PersistenceSource(const PersistenceSource&) = delete;

private:
  friend class StreamedStore;
  friend class PersistenceJob;
  friend class PersistenceQueue;
  [[nodiscard]] PersistencePin Pin() const;
  [[nodiscard]] std::size_t Capacity() const;
  ResourceCharge charge_;
  PersistencePin pin_;
  std::shared_ptr<const void> owner_;
  std::shared_ptr<BorrowedRows> rows_;
  Copy copy_;
  bool live_reads_{};
  bool private_file_;
  DiskFileId file_;
  std::uint64_t bytes_;
  std::size_t alignment_;
};
// One initialization-time allocation, leased by piece to writers and held by
// foreground restores. Optional yielding never holds staging or a stream. A HIP
// allocator supplies committed pinned storage outside this CPU-only package.
// The disk store, ledger and stream provider must outlive this object.
class StreamedStore {
public:
  // read_ahead_bytes, when nonzero, is a second allocation for device-read
  // pieces a write queues ahead of the one it is writing. Writes queue ahead
  // only with ahead_lease, and lease every stream through it, so that
  // kReservedTransferStreams stay free for foreground transfers.
  StreamedStore(ResourceLedger&, DiskStore&, std::size_t staging_bytes,
                StagingAllocator, StreamLease, std::size_t read_ahead_bytes = 0,
                ReserveStreamLease ahead_lease = {});
  StreamedStore(ResourceLedger&, DiskStore&, StagingAllocator, StreamLease);
  ~StreamedStore();
  StreamedStore(const StreamedStore&) = delete;
  StreamedStore& operator=(const StreamedStore&) = delete;
  void Write(DiskFileId, const DiskManifest&, std::span<PersistenceSource>,
             TransferTiming&, std::stop_token = {},
             const std::function<bool()>& model_idle = {});
  // Invalidation happens before fallback. Cancellation also invalidates, but
  // does not run fallback. All loads and disk pins settle before either path.
  // dependency_failed runs after invalidation and before fallback, only for
  // filesystem/checksum failures; it can quarantine reverse dependencies.
  [[nodiscard]] bool Restore(
      CheckpointId, CompatibilityDigest expected_compatibility, Adapter&, Slot&,
      TransferTiming&, std::stop_token = {},
      const std::function<void()>& fallback = {},
      const std::function<void(bool, DiskFileId)>& dependency_failed = {});
  [[nodiscard]] std::size_t StagingBytes() const;
  [[nodiscard]] std::size_t ReadAheadBytes() const;
  [[nodiscard]] std::uint64_t StartupAllocatorNanoseconds() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
class PersistenceJob {
public:
  PersistenceJob(ResourceLedger&, DiskFileId, const DiskManifest&,
                 std::vector<PersistenceSource>);
  PersistenceJob(PersistenceJob&&) noexcept = default;
  PersistenceJob& operator=(PersistenceJob&&) = delete;
  PersistenceJob(const PersistenceJob&) = delete;
  // A job whose sources hold an exclusive slot (a slot-held capture) is
  // never replaced by a newer unrelated job under queue pressure: dropping
  // it would release the slot unwritten, and the capture would be retried.
  // It also starts before older pending jobs, and its device sources (the
  // slot) are released as soon as its payloads are read, before their syncs.
  void KeepUnderPressure() noexcept { keep_ = true; }

private:
  friend class PersistenceQueue;
  bool keep_{};
  bool jumped_{};  // A slot-holding job already started before it.
  ResourceCharge metadata_;
  DiskFileId file_;
  DiskManifest manifest_;
  std::vector<PersistenceSource> sources_;
  std::size_t pinned_bytes_{};
  std::uint64_t submitted_{};
};
struct PersistenceQueueStats {
  std::uint64_t accepted{}, skipped{}, coalesced{}, superseded{}, completed{},
      failed{};
  std::size_t pending{}, pinned_bytes{}, peak_pending{}, peak_pinned_bytes{};
  TransferTiming timing;
};
// One optional worker. Capacity counts the active job as well as pending jobs;
// TrySubmit never waits for capacity. Stop cancels active/pending work and
// joins. Callbacks must be thread-safe, nonblocking and outlive the queue.
// Model idle is checked between pieces, without holding metadata or
// device-source locks. A job yields for at most yield_limit after its write
// starts, so continuous load cannot hold its pins indefinitely.
// Keep the first pending boundary and coalesce later pending ancestors into
// their newest descendant. Under pressure, recent jobs
// may replace the oldest pending non-learned job; active writes are untouched.
// Completion reports false for superseded work, outside the queue lock, with
// the job's write costs (zero for work that never started). The worker is
// the only disk writer of its store, so publication never races its own
// reclamation. A finished or dropped job (and whatever its sources own) is
// destroyed outside the queue lock, before Drain can return.
class PersistenceQueue {
public:
  PersistenceQueue(
      ResourceLedger&, StreamedStore&, std::size_t max_depth,
      std::size_t max_pinned_bytes, std::function<bool()> model_idle,
      std::function<void(CheckpointId, bool, const TransferTiming&)> completed =
          {},
      std::function<bool(const DiskManifest&)> reclaim = {},
      std::chrono::milliseconds yield_limit = kPersistenceYieldLimit);
  ~PersistenceQueue();
  PersistenceQueue(const PersistenceQueue&) = delete;
  PersistenceQueue& operator=(const PersistenceQueue&) = delete;
  bool TrySubmit(PersistenceJob);
  void Drain();
  void Stop();
  [[nodiscard]] PersistenceQueueStats Stats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gufo::cache
#endif
