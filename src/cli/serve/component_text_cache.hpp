#ifndef GUFO_SERVER_COMPONENT_TEXT_CACHE_HPP_
#define GUFO_SERVER_COMPONENT_TEXT_CACHE_HPP_

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string_view>

#include "src/cache/checkpoint.hpp"
#include "src/cache/streaming.hpp"
#include "src/cli/serve/component_cache_metrics.hpp"
#include "src/cli/serve/continuation_cache.hpp"
#include "src/cli/serve/text_publication_wait.hpp"

namespace gufo::server {
class TextModelRunner;
class TextRunnerState;
class TextRunnerSnapshot;
struct TextRunnerDiskCacheOptions;
struct TextPromptContext;

// Component-cache runners (Qwen27B, Flash-Next) stream every disk transfer
// through a fixed internal buffer, so they refuse the legacy staging option.
inline constexpr std::string_view kComponentCacheRejectsDiskStaging =
    "--cache-disk-staging-bytes applies only to DeepSeek's disk cache; "
    "Qwen27B and Flash-Next stream checkpoints of any size through an "
    "internal 1 MiB buffer, so omit the option";

// Default room for reassignment snapshots (ComponentCacheResources): one
// 325K-token Flash-Next state (3.2 GB) with margin. A second deep state
// reassigned before the first is written keeps its slot instead.
inline constexpr std::size_t kReassignmentSnapshotBytes = std::size_t{4} << 30;
// Pooled snapshot arenas are freed after this long unused: deep conversations
// rotating through the slots reuse them (a turn and its reassignment come
// seconds apart), while an idle server returns the pinned memory to the host
// and the RAM cache within a minute.
inline constexpr std::chrono::milliseconds kReassignmentSnapshotIdle{60000};

// Model-independent serving orchestration; the model supplies physical storage
// and binds its ordinary execution facade to adapter-owned slots. There is no
// RAM checkpoint tier: a request continues its slot's live frontier, or
// restores a disk checkpoint. Captures go to disk through the persistence
// queue; private states, tails and rows the slot cannot lend are copied first
// into the bounded host arenas of the snapshot pool.
struct ComponentCacheResources {
  std::shared_ptr<void> owner;
  std::unique_ptr<cache::Adapter> adapter;
  cache::StreamLease stream;
  // Optional lease from the same pool that keeps a number of streams free;
  // disk writes queue device reads ahead only through it.
  cache::ReserveStreamLease reserve_stream;
  cache::StagingAllocator staging;
  // Optional host bytes the slots pinned and charged to the ledger as they
  // allocated. Serving charges them on top of the cache's budget, and adds
  // them back to the runner's claim, which measures the host after them.
  std::function<std::size_t()> pinned_bytes;
  cache::PersistenceSource::CopyRows copy_observed_rows;
  // At most this many transfers of one capture/restore are in flight before
  // waiting (16 at most). When the shared pool is momentarily exhausted, the
  // batch settles its own leases and retries. One keeps generic providers
  // usable without a pool-capacity contract.
  std::size_t transfer_batch_size{1};
  std::size_t reassignment_batch_size{1};
  // Host bytes (snapshot arenas from `snapshot_allocate`, else `staging`)
  // that captures handed to the disk writer may occupy at once: private
  // states and tails of every capture, and on reassignment the whole
  // idle-slot state, which admission copies there in the foreground before
  // releasing the slot instead of keeping it leased until the writer has
  // read it. A state over the remaining room keeps the slot (SlotHold) where
  // it may, and is otherwise not captured. Zero disables the arenas.
  std::size_t reassignment_snapshot_bytes{kReassignmentSnapshotBytes};
  // Snapshots take arenas of this size (a larger piece takes its own), which
  // return to a pool once written and are reused. Pooled arenas unused for
  // reassignment_snapshot_idle are freed.
  std::size_t reassignment_snapshot_arena_bytes{std::size_t{64} << 20};
  std::chrono::milliseconds reassignment_snapshot_idle{
      kReassignmentSnapshotIdle};
  // Optional allocator of snapshot arenas, `staging` otherwise. The copy
  // overwrites every byte the writer reads, so it need not touch the pages.
  cache::StagingAllocator snapshot_allocate;
  // Optional: false when the host cannot pin that many more snapshot bytes
  // now (beyond the host ceiling every snapshot respects); the slot is kept.
  std::function<bool(std::size_t)> snapshot_room;
};
class ComponentTextCache {
public:
  struct Impl;
  class Lease {
  public:
    struct Impl;
    explicit Lease(std::unique_ptr<Impl>);
    ~Lease();
    Lease(const Lease&) = delete;
    [[nodiscard]] ContinuationState& state() const;
    [[nodiscard]] bool cache_hit() const noexcept;
    [[nodiscard]] std::size_t cached_tokens() const noexcept;
    [[nodiscard]] std::size_t restored_snapshot_bytes() const noexcept;
    [[nodiscard]] double restore_ms() const noexcept;
    [[nodiscard]] bool restored_from_disk() const noexcept;
    [[nodiscard]] ContinuationLookup lookup() const noexcept;
    [[nodiscard]] bool HasSnapshotFor(std::span<const ContinuationToken>) const;
    [[nodiscard]] bool TryReserveSnapshot(std::size_t, std::size_t, bool,
                                          SnapshotPurpose,
                                          std::span<const ContinuationToken>);
    void SkipSnapshot(SnapshotEventReason, std::size_t, std::size_t) noexcept;
    std::size_t PublishSnapshot(std::vector<ContinuationToken>,
                                std::shared_ptr<const ContinuationSnapshot>,
                                bool);
    std::size_t Commit(std::vector<ContinuationToken>,
                       std::shared_ptr<const ContinuationSnapshot>,
                       std::vector<ContinuationToken>);
    void Invalidate() noexcept;
    [[nodiscard]] std::size_t CaptureBytes() const;
    [[nodiscard]] std::size_t CaptureBytes(std::size_t boundary) const;
    // True when the disk would not take a capture at `boundary` (no disk
    // cache, a prompt kept off the disk, or within the disk step).
    [[nodiscard]] bool CaptureRefused(std::size_t boundary) const;
    [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Capture();
    [[nodiscard]] ComponentCacheMetrics Metrics() const;

  private:
    friend class ComponentTextCache;
    std::unique_ptr<Impl> impl_;
  };
  // Slots allocate under ceiling_bytes (host memory they may pin). The
  // ledger then holds what they pinned, bounded metadata (index, histories,
  // disk catalog) and the disk staging buffers.
  ComponentTextCache(std::shared_ptr<TextModelRunner>, std::size_t,
                     const std::optional<TextRunnerDiskCacheOptions>&,
                     std::size_t ceiling_bytes = 0);
  ~ComponentTextCache();
  [[nodiscard]] std::unique_ptr<Lease> Acquire(
      std::span<const ContinuationToken>,
      const ContinuationCache::CancellationCheck&,
      std::shared_ptr<const TextPromptContext>, bool, std::size_t, bool,
      bool persist = true);
  // A disk job is writing a prefix of the prompt much deeper than any
  // restorable boundary; admission may wait for it instead of prefilling.
  // A nonzero checkpoint ID follows that one job, even if another deeper job
  // is submitted later. Its current gain may shrink as usable cache improves.
  [[nodiscard]] std::optional<TextPendingPublication> PendingPublication(
      std::span<const ContinuationToken> prompt,
      const TextPromptContext* context, std::uint64_t checkpoint_id = 0) const;
  // Zero when the estimate itself fails (no metadata room).
  [[nodiscard]] std::size_t CachedPrefixTokens(
      std::span<const ContinuationToken>, const TextPromptContext*,
      bool common = false) const;
  // Re-tokenizing a conversation can split a previous answer differently from
  // the tokens the model sampled, and a recurrent frontier cannot rewind to
  // the first difference (upstream #1154). When the prompt repeats the whole
  // text of an idle text-only live frontier, replace that part with the
  // frontier's own tokens. With `trimmed` (a chat template rendered the
  // prompt), whitespace it trimmed from the edges of a reply stays as the
  // model generated it; whitespace missing anywhere else is an edit. A
  // frontier ending inside a prompt token continues with the rest of that
  // token. The first `keep` prompt tokens never change; `stable` follows the
  // rewrite, and the prompt stays below `max_tokens`. Returns whether the
  // prompt changed.
  bool AlignToLiveFrontier(std::vector<ContinuationToken>& prompt,
                           std::size_t& stable, std::size_t keep,
                           std::size_t max_tokens, bool trimmed) const;
  [[nodiscard]] std::size_t capacity() const noexcept;
  // Outcome of ReclaimRows.
  enum class RowReclaim : std::uint8_t {
    kReady,      // enough() holds
    kWaiting,    // a busy slot may still free rows
    kExhausted,  // no slot is busy and nothing is left to return
  };
  // Returns the paged rows idle slots hold until `enough()` holds. Slots
  // without a live frontier go first. Then live frontiers, a frontier some
  // disk checkpoint covers before the only copy of its work, oldest first;
  // an uncovered one is written to disk first when there is a disk cache.
  // With `reuse`, only the actually selected live frontier is kept, using the
  // same input identity/stable boundary as Acquire. Other token-equal
  // histories may own duplicate physical blocks. A slot is reset at most once
  // per call, and not again until it is leased. `enough` runs without the
  // cache lock.
  [[nodiscard]] RowReclaim ReclaimRows(
      std::span<const ContinuationToken>, bool reuse,
      const std::function<bool()>& enough,
      const TextPromptContext* context = nullptr, std::size_t stable = 0);
  // Rows Acquire with the same arguments would not need new paged blocks
  // for: those of the idle live frontier it would continue, whose slot
  // already maps them (a partly written last block included). Zero for a
  // cold slot or a disk restore.
  struct HeldRows {
    std::size_t tokens{};
    bool live{};
  };
  [[nodiscard]] HeldRows ContinuedTokens(
      std::span<const ContinuationToken> prompt,
      const TextPromptContext* context, bool reuse, std::size_t stable) const;
  // With a disk cache, writes the live frontier of every slot idle for at
  // least `idle_for` that no disk checkpoint covers within
  // kUncoveredCaptureTokens to disk, so a restart resumes from the last turn
  // instead of its prompt boundary. The write shares the frontier's lineage
  // chunks and is not limited by the disk step; a frontier whose own
  // execution's checkpoint is still queued waits for it, and
  // WriteBackDeferred() then reports true until a later call writes it.
  // Other slots may be running; call from the scheduler thread. Returns the
  // slots queued. The destructor drains the queue and writes back every
  // uncovered frontier once more.
  std::size_t WriteBackIdle(std::chrono::milliseconds idle_for) noexcept;
  // Before admitting `prompt` (same arguments as Acquire). False while the
  // slot it would use, its live frontier's or the one cold assignment takes,
  // is held by a queued disk write: that request waits, the other slots keep
  // running. When cold assignment would reset an idle frontier no checkpoint
  // covers, captures it first, to disk by the queue worker: it copies the
  // state to host memory here, with no other
  // model work running, and releases the slot (reassignment_snapshot_bytes);
  // without room for that copy, the worker holds the slot until it has
  // written it (false meanwhile). Call from the scheduler thread; never
  // writes the disk itself.
  [[nodiscard]] bool PrepareAdmission(std::span<const ContinuationToken> prompt,
                                      const TextPromptContext* context,
                                      bool reuse, std::size_t stable);
  [[nodiscard]] bool WriteBackDeferred() const noexcept;
  // The disk writer has pending or active jobs, whose pinned host memory
  // (reassignment snapshots, staging) it releases when they finish.
  [[nodiscard]] bool HostWritesPending() const;

private:
  // Captures the idle live frontier of slot `index` (leased exactly at
  // `location`) to disk, before the caller resets the slot. Optional: a
  // refusal or failure is logged (with `reason`, the caller's cause) and the
  // caller proceeds. Without `hold`, the whole state is copied into host
  // arenas first. With `hold`, it goes to the queue worker as a host
  // snapshot, or else keeps the slot for it, and the result is true while
  // that held write is queued: the slot is not the caller's to reset then.
  // `held` reports whether a held write took the slot (queued or not): the
  // caller must not reset it then.
  bool CaptureBeforeReset(std::size_t index, cache::BorrowedLocation location,
                          std::vector<cache::Token> tokens,
                          cache::InputIdentity input, std::string_view reason,
                          bool hold, bool* held = nullptr) noexcept;
  // Before a cold reset of slot `index`: rows of its live frontier that
  // queued disk jobs still read get host copies, so the reset does not fail
  // those jobs. Optional: a refusal is logged.
  void HostObservedRows(std::size_t index) noexcept;
  [[nodiscard]] std::size_t CachedPrefixTokensOrThrow(
      std::span<const ContinuationToken>, const TextPromptContext*,
      bool common) const;
  std::unique_ptr<Impl> impl_;
};

// Only one implementation exists for a pool. Legacy models keep their old
// lease; converted models use the component store and never construct it.
class ServingCacheLease {
public:
  ServingCacheLease(ContinuationCache::Lease lease)
      : legacy_(std::move(lease)) {}
  ServingCacheLease(std::unique_ptr<ComponentTextCache::Lease> lease)
      : component_(std::move(lease)) {}
  ServingCacheLease(ServingCacheLease&&) noexcept = default;
  ServingCacheLease& operator=(ServingCacheLease&&) noexcept = default;
  [[nodiscard]] explicit operator bool() const noexcept {
    return component_ || bool(legacy_);
  }
  [[nodiscard]] bool component() const noexcept { return bool(component_); }
  [[nodiscard]] ContinuationState& state() const;
  [[nodiscard]] bool cache_hit() const noexcept;
  [[nodiscard]] std::size_t cached_tokens() const noexcept;
  [[nodiscard]] std::size_t restored_snapshot_bytes() const noexcept;
  [[nodiscard]] double restore_ms() const noexcept;
  [[nodiscard]] bool restored_from_disk() const noexcept;
  [[nodiscard]] ContinuationLookup lookup() const noexcept;
  [[nodiscard]] bool HasSnapshotFor(std::span<const ContinuationToken>) const;
  void AdoptRestoredPrefix(std::size_t, std::size_t, double);
  [[nodiscard]] bool TryReserveSnapshot(
      std::size_t, std::size_t, bool = false,
      SnapshotPurpose = SnapshotPurpose::kContinuation,
      std::span<const ContinuationToken> = {});
  void SkipSnapshot(SnapshotEventReason, std::size_t, std::size_t) noexcept;
  std::size_t PublishSnapshot(std::vector<ContinuationToken>,
                              std::shared_ptr<const ContinuationSnapshot>,
                              bool = false);
  std::size_t Commit(std::vector<ContinuationToken>,
                     std::shared_ptr<const ContinuationSnapshot> = {},
                     std::vector<ContinuationToken> = {});
  void Invalidate() noexcept;
  [[nodiscard]] std::size_t CaptureBytes() const;
  [[nodiscard]] std::size_t CaptureBytes(std::size_t boundary) const;
  [[nodiscard]] bool CaptureRefused(std::size_t boundary) const;
  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Capture();
  [[nodiscard]] std::optional<ComponentCacheMetrics> Metrics() const;

private:
  ContinuationCache::Lease legacy_;
  std::unique_ptr<ComponentTextCache::Lease> component_;
};
}  // namespace gufo::server
#endif
