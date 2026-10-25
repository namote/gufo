#ifndef GUFO_SERVER_CONTINUATION_CACHE_HPP_
#define GUFO_SERVER_CONTINUATION_CACHE_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace gufo::server {

using ContinuationToken = std::uint32_t;

/// Supplemental input identity for prefixes ending at or before token_count.
/// Ordered boundaries allow a checkpoint before a new image to retain its
/// original identity. The complete request identity applies after the last one.
struct ContinuationInputPrefix {
  std::size_t token_count{0};
  std::vector<std::uint8_t> identity;
};

[[nodiscard]] inline std::span<const std::uint8_t> PrefixInputIdentity(
    std::span<const std::uint8_t> complete,
    std::span<const ContinuationInputPrefix> prefixes, std::size_t count) {
  for (const auto& prefix : prefixes) {
    if (count <= prefix.token_count)
      return prefix.identity;
  }
  return complete;
}

/// Model-private continuation state retained by the common serving cache.
///
/// Implementations own all attention, recurrent, position, and graph-bound
/// state. The cache deliberately cannot inspect or copy those bytes.
class ContinuationState {
public:
  ContinuationState() = default;
  virtual ~ContinuationState() = default;

  ContinuationState(const ContinuationState&) = delete;
  ContinuationState& operator=(const ContinuationState&) = delete;
  ContinuationState(ContinuationState&&) = delete;
  ContinuationState& operator=(ContinuationState&&) = delete;

  /// Discards any partially or fully computed continuation.
  virtual void Invalidate() noexcept = 0;
};

/// Immutable model-private continuation payload.
class ContinuationSnapshot {
public:
  ContinuationSnapshot() = default;
  virtual ~ContinuationSnapshot() = default;

  ContinuationSnapshot(const ContinuationSnapshot&) = delete;
  ContinuationSnapshot& operator=(const ContinuationSnapshot&) = delete;
  ContinuationSnapshot(ContinuationSnapshot&&) = delete;
  ContinuationSnapshot& operator=(ContinuationSnapshot&&) = delete;

  [[nodiscard]] virtual std::size_t PayloadBytes() const noexcept = 0;
  /// Prefer an available state that still holds this snapshot's borrowed rows.
  /// This is an allocation hint; every state must remain able to restore it.
  [[nodiscard]] virtual bool PrefersState(
      const ContinuationState&) const noexcept {
    return false;
  }
};

enum class SnapshotEventAction : std::uint8_t {
  kRemoved,
  kSkipped,
};

enum class SnapshotEventReason : std::uint8_t {
  kByteCapacity,
  kEntryCapacity,
  kExactReplacement,
  kCaptureFailure,
  kReservationMismatch,
  /// Dropped to free model rows (paged attention blocks) for an admission.
  kRowCapacity,
};

/// Continuation boundaries take precedence over optional history/retry copies.
/// A branch point is a continuation boundary learned where two prompts
/// diverged; it stays a branch point after the older prompt's checkpoints go.
enum class SnapshotPurpose : std::uint8_t {
  kContinuation,
  kHistory,
  kRetry,
  kBranchPoint,
};

/// Sanitized snapshot-retention event.
///
/// Token values and prompt contents are deliberately absent.
struct SnapshotEvent {
  SnapshotEventAction action{SnapshotEventAction::kSkipped};
  SnapshotEventReason reason{SnapshotEventReason::kByteCapacity};
  std::size_t snapshot_bytes{0};
  std::size_t token_count{0};
  std::size_t retained_snapshot_bytes{0};
  std::size_t reserved_snapshot_bytes{0};
  std::size_t capacity_bytes{0};
};

/// Explains a miss without exposing prompt or token contents.
struct ContinuationLookup {
  std::string_view miss_reason;
  std::size_t common_prefix_tokens{0};
  std::size_t checkpoint_tokens{0};
};

/// Bounded exact-prefix cache over opaque model continuation states.
///
/// The first deployment uses one entry. Supporting a bounded entry count here
/// keeps cache policy independent from Qwen, DeepSeek, and future providers.
class ContinuationCache {
public:
  using StateFactory = std::function<std::unique_ptr<ContinuationState>()>;
  using CancellationCheck = std::function<bool()>;
  using SnapshotRestore =
      std::function<void(ContinuationState&, const ContinuationSnapshot&)>;
  using SnapshotCapacity = std::function<std::size_t()>;
  using SnapshotEventSink = std::function<void(const SnapshotEvent&)>;

  struct SnapshotSupport {
    SnapshotRestore restore;
    SnapshotCapacity capacity_bytes;
    SnapshotEventSink on_event;
  };

  class Lease {
  public:
    Lease() = default;
    ~Lease();

    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept;
    Lease& operator=(Lease&& other) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept {
      return cache_ != nullptr;
    }
    [[nodiscard]] ContinuationState& state() const;
    [[nodiscard]] bool cache_hit() const noexcept { return cache_hit_; }
    [[nodiscard]] std::size_t cached_tokens() const noexcept {
      return cached_tokens_;
    }
    [[nodiscard]] std::size_t restored_snapshot_bytes() const noexcept {
      return restored_snapshot_bytes_;
    }
    [[nodiscard]] double restore_ms() const noexcept { return restore_ms_; }
    [[nodiscard]] bool restored_from_disk() const noexcept {
      return restored_from_disk_;
    }
    [[nodiscard]] ContinuationLookup lookup() const noexcept { return lookup_; }
    [[nodiscard]] bool HasSnapshotFor(
        std::span<const ContinuationToken> tokens) const;

    /// Records a successful restore performed by an optional lower cache tier.
    void AdoptRestoredPrefix(std::size_t cached_tokens,
                             std::size_t restored_bytes, double restore_ms);

    /// Reserves aggregate retained-snapshot capacity before model allocation.
    ///
    /// Byte-pressure evictions happen synchronously before this returns true.
    /// A verified replacement prefix can retire this family's old boundary
    /// before evicting another family's last copy, unless preserve_source.
    [[nodiscard]] bool TryReserveSnapshot(
        std::size_t snapshot_bytes, std::size_t token_count,
        bool preserve_source = false,
        SnapshotPurpose purpose = SnapshotPurpose::kContinuation,
        std::span<const ContinuationToken> replacement_prefix = {});

    /// Releases an admitted reservation and records a sanitized skip reason.
    void SkipSnapshot(SnapshotEventReason reason, std::size_t snapshot_bytes,
                      std::size_t token_count) noexcept;

    /// Publishes a reserved immutable checkpoint without releasing this state.
    /// Peers can fork it before this lease advances the live continuation.
    /// Intermediate captures preserve the original branching source so later
    /// byte/entry admission cannot replace the required fallback with them.
    std::size_t PublishSnapshot(
        std::vector<ContinuationToken> tokens,
        std::shared_ptr<const ContinuationSnapshot> snapshot,
        bool preserve_source = false);

    /// Atomically publishes the state and the exact tokens it represents.
    ///
    /// Returns the payload bytes actually retained. Snapshot-mode commits may
    /// publish no snapshot when admission or capture failed.
    std::size_t Commit(
        std::vector<ContinuationToken> tokens,
        std::shared_ptr<const ContinuationSnapshot> snapshot = nullptr,
        std::vector<ContinuationToken> live_tokens = {});

    /// Explicitly discards partial state. The destructor does the same if a
    /// lease is not committed.
    void Invalidate() noexcept;

  private:
    friend class ContinuationCache;

    Lease(ContinuationCache* cache, std::size_t index, bool cache_hit,
          std::size_t cached_tokens, std::size_t source_index,
          std::size_t restored_snapshot_bytes, double restore_ms) noexcept;

    ContinuationCache* cache_{nullptr};
    std::size_t index_{0};
    bool cache_hit_{false};
    std::size_t cached_tokens_{0};
    std::size_t source_index_{0};
    std::size_t restored_snapshot_bytes_{0};
    double restore_ms_{0.0};
    bool restored_from_disk_{false};
    std::size_t reserved_snapshot_bytes_{0};
    SnapshotPurpose snapshot_purpose_{SnapshotPurpose::kContinuation};
    std::size_t prompt_tokens_{0};
    std::size_t stable_prefix_tokens_{0};
    std::vector<std::uint8_t> input_identity_;
    std::vector<ContinuationInputPrefix> input_prefixes_;
    [[nodiscard]] std::span<const std::uint8_t> InputIdentity(
        std::size_t count) const {
      return PrefixInputIdentity(input_identity_, input_prefixes_, count);
    }
    ContinuationLookup lookup_;
  };

  /// Snapshot records are separate from execution slots. A zero record count
  /// defaults to capacity; the byte budget bounds retained/in-flight payloads.
  ContinuationCache(std::size_t capacity, const StateFactory& factory,
                    SnapshotSupport snapshot_support = {},
                    std::size_t snapshot_capacity = 0);
  ~ContinuationCache();

  ContinuationCache(const ContinuationCache&) = delete;
  ContinuationCache& operator=(const ContinuationCache&) = delete;
  ContinuationCache(ContinuationCache&&) = delete;
  ContinuationCache& operator=(ContinuationCache&&) = delete;

  /// A nonzero stable prefix requires a fallback at or before that boundary
  /// before a later checkpoint can be reused (including exact retries).
  [[nodiscard]] Lease Acquire(
      std::span<const ContinuationToken> prompt,
      const CancellationCheck& is_cancelled = {},
      std::span<const std::uint8_t> input_identity = {},
      const std::function<void(ContinuationState&)>& prepare_state = {},
      bool reuse_prompt = true, std::size_t stable_prefix_tokens = 0,
      std::span<const ContinuationInputPrefix> input_prefixes = {});

  /// Longest retained prefix of prompt with a matching input identity,
  /// ignoring stable-prefix fallback rules. Leases nothing.
  [[nodiscard]] std::size_t CachedPrefixTokens(
      std::span<const ContinuationToken> prompt,
      std::span<const std::uint8_t> input_identity = {},
      std::span<const ContinuationInputPrefix> input_prefixes = {}) const;

  /// Longest prefix prompt shares with any retained checkpoint or live
  /// frontier of the same input identity, whether or not that record ends
  /// there. This is where prompts diverge, so a checkpoint captured there
  /// serves later prompts that share it. Leases nothing.
  [[nodiscard]] std::size_t CommonPrefixTokens(
      std::span<const ContinuationToken> prompt,
      std::span<const std::uint8_t> input_identity = {},
      std::span<const ContinuationInputPrefix> input_prefixes = {}) const;

  /// In-place persistence of released live frontiers, on eviction only: a
  /// frontier is written when an admission needs its seat, so conversations
  /// that keep their seats cost no disk writes.
  void EnablePersistence();
  /// False while every free seat holds a frontier not yet on disk; the
  /// oldest is then marked for writing and the caller retries the admission
  /// later instead of blocking. True for a resuming conversation, a seat that
  /// is free to reuse, or when persistence is off.
  [[nodiscard]] bool SeatReadyFor(std::span<const ContinuationToken> prompt);
  /// A released seat held for an in-place write: not reassigned, nothing
  /// executes on it; a resuming conversation sets the cancel flag and waits
  /// for EndPersist. `urgent_only` selects seats SeatReadyFor marked;
  /// otherwise (shutdown) every unwritten frontier, oldest first.
  struct PersistingSeat {
    std::size_t index{0};
    ContinuationState* state{nullptr};
    std::vector<ContinuationToken> tokens;
    std::vector<std::uint8_t> identity;
  };
  /// `min_idle` (not urgent_only): only frontiers released at least that
  /// long ago, written while idle so a later eviction finds them on disk.
  [[nodiscard]] std::optional<PersistingSeat> BeginPersist(
      bool urgent_only,
      std::chrono::milliseconds min_idle = std::chrono::milliseconds::zero());
  [[nodiscard]] bool PersistCancelled(std::size_t index) const;
  /// Holds the oldest released seat whose frontier's row blocks are not on
  /// disk yet, as BeginPersist does, for a write of those blocks alone; its
  /// record stays for eviction. EndBlockWrite releases it; `complete` marks
  /// the blocks written.
  [[nodiscard]] std::optional<PersistingSeat> BeginBlockWrite();
  void EndBlockWrite(std::size_t index, bool complete) noexcept;
  /// An admission waits for a frontier write (SeatReadyFor marked one).
  [[nodiscard]] bool PersistUrgent() const;
  /// Releases the hold; `persisted` marks the frontier as stored on disk.
  void EndPersist(std::size_t index, bool persisted) noexcept;

  /// Outcome of ReclaimForRows.
  enum class RowReclaim : std::uint8_t {
    kReady,      // enough() holds
    kWaiting,    // a frontier is being written, or leased seats may free rows
    kExhausted,  // nothing left to drop and no seat is busy
  };
  /// Frees model rows that retained snapshots and idle seats hold until
  /// `enough()` holds, sparing what `prompt` would reuse. Snapshots go first,
  /// by removal priority and age; then idle seats, oldest first. With
  /// persistence on, a frontier not yet on disk is written before it goes:
  /// it is marked for writing and kWaiting returned. `enough` is called
  /// without the cache lock.
  [[nodiscard]] RowReclaim ReclaimForRows(
      std::span<const ContinuationToken> prompt,
      const std::function<bool()>& enough);

  [[nodiscard]] std::size_t capacity() const noexcept;
  /// Immutable checkpoint records, independent of execution capacity().
  [[nodiscard]] std::size_t entry_capacity() const noexcept;
  [[nodiscard]] std::size_t snapshot_capacity_bytes() const noexcept;
  [[nodiscard]] std::size_t retained_snapshot_bytes() const noexcept;
  [[nodiscard]] std::size_t reserved_snapshot_bytes() const noexcept;

private:
  struct Entry;

  [[nodiscard]] ContinuationState& StateAt(std::size_t index);
  /// Reclaiming a retained copy of the same prefix keeps its learned branch
  /// point: purpose becomes kBranchPoint for the publication.
  [[nodiscard]] bool ReserveSnapshot(
      std::size_t source_index, std::size_t snapshot_bytes,
      std::size_t token_count, bool preserve_source, SnapshotPurpose& purpose,
      std::span<const ContinuationToken> replacement_prefix,
      std::span<const std::uint8_t> input_identity);
  void SkipSnapshot(std::size_t reservation_bytes, SnapshotEventReason reason,
                    std::size_t snapshot_bytes,
                    std::size_t token_count) noexcept;
  [[nodiscard]] std::size_t Commit(
      std::size_t index, std::size_t source_index,
      std::size_t reservation_bytes, std::vector<ContinuationToken> tokens,
      std::shared_ptr<const ContinuationSnapshot> snapshot,
      std::vector<std::uint8_t> input_identity,
      std::vector<ContinuationToken> live_tokens,
      std::vector<std::uint8_t> live_identity, bool release_state = true,
      std::size_t* published_index = nullptr,
      std::size_t stable_prefix_tokens = 0,
      SnapshotPurpose purpose = SnapshotPurpose::kContinuation);
  void Invalidate(std::size_t index, std::size_t reservation_bytes) noexcept;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_CONTINUATION_CACHE_HPP_
