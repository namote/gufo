#include "src/cli/serve/component_text_cache.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <thread>
#include <tuple>
#include <utility>

#include "src/cache/disk_catalog.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/snapshot_pool.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/platform/os_info.hpp"

namespace gufo::server {
namespace {
// Room for device-read pieces a disk write queues ahead. It is pinned RAM the
// ledger adds on top of the slot ceiling, so take at most 1/64 of that
// ceiling, in whole staging pieces; caches too small for two pieces read
// each piece in place.
std::size_t ReadAheadBytes(std::size_t ceiling) {
  const auto bytes = std::min(cache::kPersistenceReadAheadBytes,
                              ceiling / 64 / cache::kDefaultDiskStagingBytes *
                                  cache::kDefaultDiskStagingBytes);
  return bytes >= 2 * cache::kDefaultDiskStagingBytes ? bytes : 0;
}
// Host bookkeeping the cache charges to its ledger (index, histories,
// leases, the disk store and catalog, publication scratch): a fixed base plus
// a share of the disk budget, since the disk catalog grows with the disk
// cache (about 2 MiB per GiB of stored checkpoints in production). There is
// no RAM checkpoint tier: nothing else competes for this room, so it holds
// metadata only and never refuses it for cached payload.
constexpr std::size_t kMetadataBaseBytes = std::size_t{256} << 20;
constexpr std::size_t kMetadataDiskShare = 64;
std::size_t SaturatingAdd(std::size_t a, std::size_t b) {
  return a > std::numeric_limits<std::size_t>::max() - b
             ? std::numeric_limits<std::size_t>::max()
             : a + b;
}
using Clock = std::chrono::steady_clock;
class SettledSignal final : public cache::CompletionSignal {
public:
  bool Ready() const noexcept override { return true; }
  cache::TransferResult Wait() noexcept override {
    return cache::TransferResult::kSucceeded;
  }
};
cache::InputIdentity Inputs(std::size_t count,
                            const TextPromptContext* context) {
  std::vector<cache::InputPrefix> prefixes;
  if (context)
    for (const auto& p : context->cache_prefixes)
      prefixes.push_back({p.token_count, p.identity});
  return cache::InputIdentity(
      count, context ? context->cache_identity : cache::Identity{},
      std::move(prefixes));
}
cache::Identity At(const cache::InputIdentity& input, std::size_t boundary) {
  const auto bytes = input.At(boundary);
  return {bytes.begin(), bytes.end()};
}
std::size_t PayloadBytes(const cache::DiskManifest& manifest) {
  std::size_t bytes = 0;
  for (const auto& component : manifest.components) {
    for (const auto& chunk : component.chunks)
      bytes += chunk.bytes;
    if (component.tail)
      bytes += component.tail->bytes;
    if (component.private_state)
      bytes += component.private_state->bytes;
  }
  return bytes;
}
struct DurableCapture {
  cache::ResourceCharge metadata;
  std::shared_ptr<const void> branch;
  cache::DiskReadPin ancestor_pin;
  cache::DiskManifest manifest;
  std::vector<cache::PersistenceSource> sources;
  // An idle write-back of a live frontier: not limited by the disk step.
  bool write_back{};
  // Its sources read a slot the capture keeps leased (SlotHold) until the
  // queue worker has written them.
  bool holds_slot{};
  // Its sources are a host snapshot of a slot reset right after (the only
  // copy of that state).
  bool host_snapshot{};
  // Some sources are device rows the adapter retains for this write
  // (RetainRows); they return to a shared pool when it ends.
  bool device_rows{};
  // A held write that may start before older jobs (KeepUnderPressure).
  // False when older jobs still read rows of the slot it holds, which its
  // release would let a reset retire.
  bool jump{true};
  // A rewind point written to disk (PersistRewind): idle, reassignment and
  // reset writes of it are not limited by the disk step.
  bool step_exempt{};
  // Its sources are a rewind point's own pieces, which the point drops on a
  // reset of its slot: like a host snapshot, the job may be that boundary's
  // only copy, so it is kept under queue pressure.
  bool rewind{};
};
// A stable boundary kept in memory on its slot, for a conversation whose
// client sends its history back without the generated reply as the model
// produced it (dropped or discarded reasoning). The next prompt then diverges
// inside that reply, where the recurrent state cannot rewind, and the
// boundary is usually within the disk step of an earlier checkpoint. The
// point holds what a disk checkpoint of that boundary would: completed rows
// the adapter retains in device storage (the slot's own KV blocks, no copy),
// and host copies of the rest (private states, partial tails, rows it cannot
// retain). Nothing is lent from the slot, so later execution, restores and
// resets leave it intact. Restoring it into its slot re-maps the retained
// rows and copies only the host pieces back.
// Bytes charged against a counter until destroyed (or released).
struct ArenaCharge {
  std::atomic<std::size_t>* charged{};
  std::size_t bytes{};
  ArenaCharge() = default;
  ArenaCharge(const ArenaCharge&) = delete;
  ArenaCharge& operator=(const ArenaCharge&) = delete;
  ~ArenaCharge() {
    if (charged)
      charged->fetch_sub(bytes);
  }
};
// The host arenas of a rewind point, charged for whole arenas until the last
// holder (the point's pieces, or a disk job reading them) drops them.
struct RewindArenas {
  std::vector<std::shared_ptr<const void>> arenas;
  ArenaCharge charge;
};
struct RewindPoint {
  cache::ResourceCharge metadata;
  std::vector<cache::Token> tokens;
  // Input identity at the boundary, and with the image boundaries within.
  cache::Identity input;
  cache::InputIdentity inputs;
  std::vector<cache::ComponentPosition> positions;
  cache::LineageId lineage;
  struct Piece {
    cache::ComponentId component;
    bool private_state{};
    cache::Rows first{}, end{};
    // Retained device rows (device) or a host arena.
    std::shared_ptr<const void> owner;
    std::span<const std::byte> bytes;
    bool device{};
  };
  // In component order, rows ascending.
  std::vector<Piece> pieces;
  std::size_t host_bytes{}, device_bytes{};
  // Snapshot-arena bytes its host pieces occupy (whole arenas), and their
  // holder, which each host piece's owner shares.
  std::size_t arena_bytes{};
  std::weak_ptr<RewindArenas> arenas;
  std::size_t host_pieces{};
  // Under the mutex: the arena bytes dropping this point returns, which is
  // none while a disk job still reads its pieces.
  [[nodiscard]] std::size_t ReleasableArenaBytes() const {
    const auto holders = arenas.use_count();
    return holders && static_cast<std::size_t>(holders) == host_pieces
               ? arena_bytes
               : 0;
  }
  // False for a request that keeps its prompt off the disk.
  bool persist{true};
  // The disk job last queued for it, while it is pending (under the cache
  // mutex). Coverage is rechecked otherwise: a failed, superseded or
  // evicted write leaves the point to be written again.
  std::uint64_t pending_checkpoint{};
  // One submitted idle write per committed turn, shared by write-back
  // and release maintenance. Even a failed/untracked/evicted publication
  // consumes it; another turn enables another attempt.
  std::optional<Clock::time_point> idle_attempt;
};
// One piece a capture writes from a slot (inherited chunks excepted).
struct SlotPiece {
  cache::ComponentId component;
  bool private_state;
  cache::Rows first, end;
  std::size_t bytes, alignment;
  cache::DiskFileId file;
  bool private_file;
};
class ComponentSnapshot final : public TextRunnerSnapshot {
public:
  std::shared_ptr<DurableCapture> durable;
  std::size_t bytes{};
  std::size_t PayloadBytes() const noexcept override { return bytes; }
};
void Check(cache::Completion completion) {
  if (completion.Wait() != cache::TransferResult::kSucceeded)
    throw std::runtime_error("component cache transfer failed");
}
// Every checkpoint candidate is a disk publication; a live frontier is not.
bool Published(const cache::PrefixCandidate& candidate) {
  return bool(candidate.durable);
}
// A learned (shared) boundary is written whatever its spacing. Like a grid
// capture, it does not consume the spacing allowance of the conversation's
// own next checkpoint.
// A copy of a live frontier (CaptureBeforeReset, WriteBackIdle) restores at
// any position: its state is the one live reuse would continue. A prompt
// checkpoint restores exactly only on the prefill alignment.
bool AnyPosition(const cache::PrefixCandidate& candidate) {
  return candidate.durable && candidate.durable->Manifest().purpose ==
                                  cache::CheckpointPurpose::kGenerated;
}
std::string_view PurposeName(cache::CheckpointPurpose purpose) {
  switch (purpose) {
    case cache::CheckpointPurpose::kPrompt:
      return "prompt";
    case cache::CheckpointPurpose::kGenerated:
      return "generated";
    case cache::CheckpointPurpose::kGrid:
      return "grid";
    case cache::CheckpointPurpose::kLearned:
      return "learned";
  }
  return "unknown";
}
bool SpacingAnchor(cache::CheckpointPurpose purpose) {
  return purpose != cache::CheckpointPurpose::kLearned;
}
bool SpacingAnchor(const cache::PrefixCandidate& candidate) {
  return candidate.durable &&
         SpacingAnchor(candidate.durable->Manifest().purpose);
}
// What the current exception says, for a log line in its handler.
std::string CurrentExceptionReason() {
  try {
    throw;
  } catch (const std::exception& error) {
    return error.what();
  } catch (...) {
    return "non-standard exception";
  }
}
class TransferBatch {
public:
  explicit TransferBatch(std::size_t capacity) : capacity_(capacity) {
    if (!capacity || capacity > jobs_.size())
      throw std::invalid_argument("invalid cache transfer batch size");
  }
  // Acquire returns null when the shared pool is momentarily exhausted. Settle
  // this batch's own leases before retrying, so a wide batch never fails a
  // transfer that a narrower one would have completed.
  template<class Acquire, class Submit>
  void Add(Acquire acquire, Submit submit) {
    if (size_ == capacity_)
      Finish();
    auto stream = acquire();
    if (!stream && size_) {
      Finish();
      stream = acquire();
    }
    if (!stream)
      throw std::runtime_error("component cache transfer pool exhausted");
    auto completion = submit(*stream);
    jobs_[size_++].emplace(Job{std::move(stream), std::move(completion)});
  }
  void Finish() {
    // Jobs keep their streams and completions together. On any exception,
    // destruction drains every remaining transfer before buffers unwind.
    for (std::size_t i = 0; i < size_; ++i) {
      Check(std::move(jobs_[i]->completion));
      jobs_[i].reset();
    }
    size_ = 0;
  }

private:
  struct Job {
    std::unique_ptr<cache::Stream> stream;
    cache::Completion completion;
  };
  std::array<std::optional<Job>, 16> jobs_;
  std::size_t capacity_, size_{};
};
}  // namespace

struct ComponentTextCache::Impl {
  struct Entry {
    // Guard fallbacks outlive their slots. Idle pieces and capture/load work
    // obtain fresh one-completion leases from the shared transfer pool.
    std::unique_ptr<cache::Stream> preservation;
    std::unique_ptr<cache::LeasedSlot> slot;
    std::optional<cache::IndexEntryId> live;
    std::unique_ptr<cache::ExecutionHistory> history;
    // Identity of uninterrupted native execution; token equality alone does
    // not establish that a durable chunk belongs to this execution.
    std::shared_ptr<const void> branch;
    std::shared_ptr<const cache::DiskDescription> durable;
    // The live frontier's prompt, to tell whether any checkpoint covers it.
    std::vector<cache::Token> live_tokens;
    // Its input identity, with the request's image-prefix boundaries: a
    // checkpoint inside the prompt carries the identity of the images before
    // it, so rebuilding this from the complete identity alone fails every
    // capture whose history ends before an image.
    cache::InputIdentity live_input;
    // Edges of replies within live_tokens, where a chat template trims
    // whitespace from message content: where a reply begins, and where it
    // ends or its final token begins. Kept across live continuations
    // (AlignToLiveFrontier, upstream #1158).
    std::vector<std::size_t> reply_starts, reply_ends;
    // Acquisition order; cold assignment replaces the oldest live frontier.
    std::uint64_t used{};
    bool busy{};
    // Paged rows: false once ReclaimRows emptied the slot, until it is leased.
    bool holds_rows{};
    // Busy because a queued disk write keeps the slot leased (SlotHold); it
    // returns to idle, with its live frontier, once the write finishes.
    bool writing{};
    // The execution (branch) and length of the live frontier admission last
    // handed to the disk writer (PrepareAdmission). A write that failed or
    // was dropped is not handed over again: cold assignment then captures it
    // without the slot. Another execution of the same length is a new
    // frontier.
    std::shared_ptr<const void> held_branch;
    std::size_t held_tried{};
    // Where the live frontier is, to lease it exactly for a capture.
    cache::BorrowedLocation live_location{};
    // When the live frontier was last committed.
    Clock::time_point idle_since{};
    // Length of the live frontier last queued by WriteBackIdle; reset with
    // the lineage (durable) when the slot starts another execution, and when
    // that write fails.
    std::size_t written_back{};
    // Length of the live frontier WriteBackIdle last tried, queued or not; it
    // retries once the frontier grows kUncoveredCaptureTokens past it.
    std::size_t write_back_tried{};
    // False when the request that left the live frontier asked to keep its
    // prompt off the disk (cache_persist=false): it is never captured.
    bool live_persist{true};
    // The rewind point of this slot's execution, a prefix of live_tokens;
    // dropped (after a disk write) whenever the slot is reset or restored.
    std::shared_ptr<RewindPoint> rewind;
    // The stable boundary of the request that left the live frontier.
    std::size_t live_stable{};
    // Its conversation sent a generated reply back rewritten: its requests
    // keep rewind points, and its idle and reassignment writes take the
    // rewind point instead of the frontier no later prompt continues.
    bool drops_reply{};
    // The current rewind boundary was published and released. Its generated
    // reply frontier must not become a replacement disk write on idle/reset.
    bool frontier_dead{};
  };
  // Under the mutex: the idle slot whose rewind point serves `prompt`
  // deepest, beyond `live_floor` and at least at `floor`, with its boundary.
  struct RewindChoice {
    std::size_t index, boundary;
  };
  [[nodiscard]] std::optional<RewindChoice> FindRewind(
      std::span<const cache::Token> prompt, const cache::InputIdentity& input,
      std::size_t live_floor = 0, std::size_t floor = 0) const {
    const std::size_t alignment = std::max<std::size_t>(
        1, runner->Descriptor().capabilities.prefill_alignment);
    std::optional<RewindChoice> best;
    for (std::size_t i = 0; i < entries.size(); ++i) {
      const auto& e = *entries[i];
      const auto* point = e.rewind.get();
      if (!point || e.busy || !e.live || !e.holds_rows)
        continue;
      const auto n = point->tokens.size();
      // Below the prompt end (the next token is decoded from it), aligned
      // (prefill continues it exactly), and still the slot's own history.
      if (!n || n >= prompt.size() || n % alignment != 0 || n <= live_floor ||
          n < floor || (best && n <= best->boundary) ||
          !Prefixes(point->tokens, prompt) ||
          !Prefixes(point->tokens, e.live_tokens) ||
          !std::ranges::equal(input.At(n), point->input))
        continue;
      best = RewindChoice{i, n};
    }
    return best;
  }
  // Under the mutex: FindRewind past what Acquire would otherwise reuse from
  // `found`: deeper than a live frontier, and at least as deep as a
  // checkpoint it could restore (no disk read).
  [[nodiscard]] std::optional<RewindChoice> FindRewind(
      const cache::PrefixLookup& found, std::span<const cache::Token> prompt,
      const cache::InputIdentity& input,
      const TextPromptContext* context) const {
    const bool whole = runner->ReusesWholePrompt(context);
    const std::size_t alignment = std::max<std::size_t>(
        1, runner->Descriptor().capabilities.prefill_alignment);
    std::size_t live_floor = 0, floor = 0;
    for (const auto& candidate : found.candidates) {
      if (candidate.boundary >= prompt.size() && !whole)
        continue;
      // Only an idle live frontier is one Acquire continues.
      if (candidate.live) {
        if (!entries.at(candidate.live->slot.value - 1)->busy)
          live_floor = std::max<std::size_t>(live_floor, candidate.boundary);
      } else if (candidate.boundary % alignment == 0 ||
                 candidate.boundary >= prompt.size() || AnyPosition(candidate))
        floor = std::max<std::size_t>(floor, candidate.boundary);
    }
    return FindRewind(prompt, input, live_floor, floor);
  }
  // Under the mutex: whether `point` is the stable boundary of the request
  // that left the live frontier, so it, not the frontier, is what the
  // conversation's next prompt continues. An older point (the newest capture
  // was refused) leaves the frontier its own writes.
  static bool RewindCurrent(const Entry& e, const RewindPoint& point) {
    return e.live &&
           point.tokens.size() + kUncoveredCaptureTokens >= e.live_stable &&
           Prefixes(point.tokens, e.live_tokens);
  }
  // Under the mutex: whether `prompt` rewrites the generated reply of an
  // idle live frontier: it repeats that frontier's prompt through its stable
  // boundary but not the whole frontier.
  [[nodiscard]] bool RewritesReply(std::span<const cache::Token> prompt,
                                   const cache::InputIdentity& input) const {
    return std::ranges::any_of(entries, [&](const auto& e) {
      const auto n = e->live_stable;
      return !e->busy && e->live && n && n < e->live_tokens.size() &&
             n <= prompt.size() &&
             std::equal(e->live_tokens.begin(), e->live_tokens.begin() + n,
                        prompt.begin()) &&
             !Prefixes(e->live_tokens, prompt) &&
             std::ranges::equal(input.At(n), e->live_input.At(n));
    });
  }
  // A live frontier this many tokens or more past every checkpoint of its
  // prompt is captured before its slot is reset for another request.
  static constexpr std::size_t kUncoveredCaptureTokens = 64;
  static bool Prefixes(std::span<const cache::Token> prefix,
                       std::span<const cache::Token> tokens) {
    return prefix.size() <= tokens.size() &&
           std::equal(prefix.begin(), prefix.end(), tokens.begin());
  }
  // Tokens of the live frontier a durable checkpoint holds.
  static std::size_t Durable(const Entry& e) {
    return e.durable && Prefixes(e.durable->Manifest().tokens, e.live_tokens)
               ? e.durable->Manifest().tokens.size()
               : 0;
  }
  // Also counting a write-back still queued, as its completion will.
  static std::size_t Covered(const Entry& e) {
    return std::max(e.written_back, Durable(e));
  }
  struct Uncovered {
    std::size_t index;
    cache::BorrowedLocation location;
    std::vector<cache::Token> tokens;
    cache::InputIdentity input;
  };
  // Keeps an idle slot leased for a queued disk write that reads it, so its
  // rows stay unchanged; releasing it (when the worker drops the job) returns
  // the slot, live frontier intact, to admission.
  struct SlotHold {
    SlotHold(Impl* cache, Entry* entry, cache::SlotLease slot)
        : cache(cache), entry(entry), slot(std::move(slot)) {
      const std::lock_guard lock(cache->mutex);
      entry->writing = true;
    }
    SlotHold(const SlotHold&) = delete;
    ~SlotHold() {
      try {
        // The write only read the slot: its generation stays valid.
        slot.Commit();
      } catch (...) {
      }
      const std::lock_guard lock(cache->mutex);
      entry->writing = false;
      entry->busy = false;
    }
    Impl* cache;
    Entry* entry;
    cache::SlotLease slot;
  };
  // Under the mutex: the entry's live frontier when no checkpoint comes within
  // kUncoveredCaptureTokens of it.
  [[nodiscard]] std::optional<Uncovered> UncoveredLive(std::size_t i) const {
    const auto& e = *entries[i];
    if (e.frontier_dead || !e.live ||
        e.live_tokens.size() < kUncoveredCaptureTokens ||
        index.HasCheckpoint({identity, e.live_tokens, e.live_input},
                            e.live_tokens.size() - kUncoveredCaptureTokens))
      return {};
    // A host snapshot of this frontier queued for the disk (admission's
    // reassignment capture) reads nothing from the slot: it covers the
    // frontier through a reset, until its write fails.
    if (std::ranges::any_of(pending_durable, [&](const auto& capture) {
          return capture && capture->host_snapshot &&
                 capture->branch == e.branch &&
                 std::ranges::equal(capture->manifest.tokens, e.live_tokens) &&
                 capture->manifest.input ==
                     At(e.live_input, e.live_tokens.size());
        }))
      return {};
    return Uncovered{i, e.live_location, e.live_tokens, e.live_input};
  }
  std::shared_ptr<TextModelRunner> runner;
  std::string kv_cache_type = runner->Descriptor().kv_cache_type;
  std::uint32_t kv_layout_version = runner->Descriptor().kv_layout_version;
  cache::ResourceLedger ledger;
  std::unique_ptr<ComponentCacheResources> resources;
  cache::Identity identity;
  cache::CompatibilityDigest digest;
  cache::DiskFileId nonce{};
  // Durable descriptions in the index retain a Store facade until index
  // teardown has finished using its publication pins.
  std::unique_ptr<cache::DiskStore> disk;
  cache::PrefixIndex index;
  mutable std::mutex mutex;
  // Snapshot-arena bytes the host pieces of rewind points occupy, whole
  // arenas, while a point or a disk job of one holds them (RewindArenas).
  std::atomic<std::size_t> rewind_arena_bytes{};
  [[nodiscard]] std::size_t RewindArenaBudget() const {
    return resources->reassignment_snapshot_bytes / 4;
  }
  std::vector<std::unique_ptr<Entry>> entries;
  std::uint64_t clock{};
  std::unique_ptr<cache::DiskCatalog> catalog;
  std::unique_ptr<cache::StreamedStore> streaming;
  std::unique_ptr<cache::PersistenceQueue> queue;
  static constexpr std::size_t kPersistenceDepth = 8;
  // The last WriteBackIdle skipped a frontier for its own pending write.
  std::atomic<bool> write_back_deferred{false};
  std::array<std::shared_ptr<DurableCapture>, 2 * kPersistenceDepth>
      pending_durable;
  std::size_t disk_capacity{}, disk_step{};
  // Under the mutex: a queued or active disk job reads device rows kept for
  // it (a capture's retained rows), which return to a shared pool when it
  // ends.
  [[nodiscard]] bool WritesHoldDeviceRows() const {
    return std::ranges::any_of(pending_durable, [](const auto& capture) {
      return capture && capture->device_rows;
    });
  }
  // Components whose completed chunks the adapter retains in device storage.
  std::vector<cache::ComponentId> device_components;
  // Host arenas of reassignment snapshots, null when disabled; the arenas a
  // queued job holds keep it alive.
  std::shared_ptr<SnapshotPool> snapshot_pool;
  cache::ResourceCharge metadata;
  // A lookup only reads the index, but its result charges metadata: without
  // room it finds nothing, and the request runs cold instead of failing.
  [[nodiscard]] cache::PrefixLookup Lookup(
      const cache::PrefixQuery& query) const {
    try {
      return index.Lookup(query);
    } catch (const std::exception& error) {
      try {
        Logger::Info("cache", std::string("schema=component-cache-v1 "
                                          "event=lookup_failed reason=") +
                                  error.what());
      } catch (...) {
      }
      return {};
    }
  }
  // Bytes a capture at `positions` copies out of the slot (private states,
  // tails and rows the device does not retain), beyond its lineage.
  [[nodiscard]] std::size_t CaptureCopyBytes(
      const cache::ExecutionHistory& history,
      std::span<const cache::ComponentPosition> positions) const {
    return history.NewPayloadBytes(positions, device_components);
  }

  Impl(std::shared_ptr<TextModelRunner> runner, std::size_t count,
       const std::optional<TextRunnerDiskCacheOptions>& options,
       std::size_t host_ceiling)
      : runner(std::move(runner)),
        // Disk transfers lease one fixed pinned staging buffer. Its size does
        // not change restore or write speed (#562), and a larger one only
        // takes pinned RAM and device heap from the sessions. Writes may also
        // queue device-read pieces ahead (ReadAheadBytes). Slots allocate
        // under the host ceiling; what they pin stays charged afterwards.
        ledger(
            {SaturatingAdd(SaturatingAdd(host_ceiling, MetadataBytes(options)),
                           options ? cache::kDefaultDiskStagingBytes +
                                         ReadAheadBytes(host_ceiling)
                                   : 0),
             SaturatingAdd(host_ceiling, MetadataBytes(options)),
             options ? cache::kDefaultDiskStagingBytes +
                           ReadAheadBytes(host_ceiling)
                     : 0,
             SaturatingAdd(host_ceiling, MetadataBytes(options))}),
        resources(this->runner->CreateComponentCacheResources(ledger)),
        identity(resources->adapter->CompatibilityIdentity()),
        index(ledger) {
    metadata = ledger.Reserve(cache::ResourceCategory::kMetadata, sizeof(Impl))
                   .Convert();
    if (resources->reassignment_snapshot_bytes &&
        (resources->snapshot_allocate || resources->staging))
      snapshot_pool = std::make_shared<SnapshotPool>(
          resources->reassignment_snapshot_bytes,
          resources->reassignment_snapshot_idle,
          resources->snapshot_allocate ? resources->snapshot_allocate
                                       : resources->staging);
    for (const auto& d : resources->adapter->Components())
      if (resources->adapter->RetainsRows(d.id))
        device_components.push_back(d.id);
    crypto::Sha256Hasher hash;
    hash.Update(identity);
    digest = hash.Finish();
    if (!platform::FillRandomBytes(nonce))
      throw std::runtime_error("cannot obtain cache publication nonce");
    index.Register(identity, resources->adapter->Components(), digest);
    if (options) {
      const auto row_bytes =
          cache::MinimumStagingBytes(resources->adapter->Components());
      if (cache::kDefaultDiskStagingBytes < row_bytes)
        throw std::logic_error("a " + std::to_string(row_bytes) +
                               "-byte cache row exceeds the disk staging "
                               "buffer");
      if (options->staging_capacity_bytes)
        throw std::invalid_argument(
            std::string(kComponentCacheRejectsDiskStaging));
      // Take directory ownership before slots allocate, so a second server
      // fails before claiming their memory.
      disk = std::make_unique<cache::DiskStore>(ledger, options->directory,
                                                options->capacity_bytes);
    }
    for (std::size_t i = 0; i < count; ++i) {
      auto entry = std::make_unique<Entry>();
      entry->preservation = Stream();
      entry->slot = std::make_unique<cache::LeasedSlot>(
          ledger, *resources->adapter, *entry->preservation,
          cache::SlotId{i + 1},
          cache::IdleSpillConfig{1ULL << 20, resources->stream,
                                 [this] {
                                   const std::lock_guard lock(mutex);
                                   return std::ranges::none_of(
                                       entries,
                                       [](const auto& e) { return e->busy; });
                                 },
                                 resources->reassignment_batch_size});
      entries.push_back(std::move(entry));
    }
    if (options) {
      catalog = std::make_unique<cache::DiskCatalog>(ledger, *disk, index);
      for (const auto& durable : disk->Entries()) {
        cache::ExecutionHistory::ObserveDurableId(
            durable.Manifest().checkpoint);
        if (durable.Manifest().compatibility == digest)
          (void)index.Insert(identity,
                             catalog->Find(durable.Manifest().checkpoint));
      }
    }
    // The slots allocated under the host ceiling; what they pinned stays
    // charged. Above that the ledger holds only metadata and disk staging:
    // there is no RAM checkpoint tier.
    const auto pinned = resources->pinned_bytes ? resources->pinned_bytes() : 0;
    // A catalog already past its share keeps the base room above it, so
    // admissions are not refused from the start.
    const auto limit = std::min(
        ledger.Limits().ram_bytes,
        std::max(
            SaturatingAdd(pinned, MetadataBytes(options)),
            SaturatingAdd(ledger.Snapshot().ram_bytes, kMetadataBaseBytes)));
    const auto staging = ledger.Limits().staging_bytes;
    ledger.ConstrainLimits(
        {SaturatingAdd(limit, staging), limit, staging, limit});
    if (options) {
      streaming = std::make_unique<cache::StreamedStore>(
          ledger, *disk, cache::kDefaultDiskStagingBytes, resources->staging,
          resources->stream,
          resources->reserve_stream ? ReadAheadBytes(host_ceiling) : 0,
          resources->reserve_stream);
      disk_capacity = options->capacity_bytes;
      disk_step = options->min_checkpoint_step_tokens;
      disk->ReclaimOrphans();
      if (disk->Stats().managed_bytes > disk_capacity) {
        // A lowered --cache-disk-bytes keeps its meaning: reclaim orphans and
        // temporaries first, then evict the oldest checkpoints by creation
        // (startup has no use history) to fit.
        disk->WaitForReclamation();
        std::size_t evicted{};
        while (disk->Stats().managed_bytes > disk_capacity &&
               catalog->EvictOne())
          ++evicted;
        const auto managed = disk->Stats().managed_bytes;
        const auto fields = " managed_bytes=" + std::to_string(managed) +
                            " capacity_bytes=" + std::to_string(disk_capacity);
        if (evicted)
          Logger::Info("cache",
                       "schema=component-cache-v1 event=disk_evicted "
                       "reason=budget entries=" +
                           std::to_string(evicted) + fields);
        if (managed > disk_capacity)
          Logger::Warn(
              "cache",
              "schema=component-cache-v1 event=disk_over_budget" + fields);
      }
      queue = std::make_unique<cache::PersistenceQueue>(
          // Sources pin no ledger-charged payload: they read the slot,
          // retained device rows or host snapshot arenas. The queue needs a
          // nonzero bound; one byte refuses any job that would.
          ledger, *streaming, kPersistenceDepth, 1,
          [this] {
            // A slot held for this queue's own write runs no model work.
            const std::lock_guard lock(mutex);
            return std::ranges::none_of(entries, [](const auto& entry) {
              return entry->busy && !entry->writing;
            });
          },
          [this](cache::CheckpointId id, bool success,
                 const cache::TransferTiming& timing) {
            Completed(id, success, timing);
          },
          [this](const cache::DiskManifest& manifest) {
            return ReclaimDisk(manifest);
          });
    }
    Logger::Info(
        "cache",
        "schema=component-cache-v1 event=configured sessions=" +
            std::to_string(count) + " metadata_limit_bytes=" +
            std::to_string(limit) + " staging_capacity_bytes=" +
            std::to_string(ledger.Limits().staging_bytes) +
            " snapshot_capacity_bytes=" +
            std::to_string(
                snapshot_pool ? resources->reassignment_snapshot_bytes : 0) +
            " disk_capacity_bytes=" + std::to_string(disk_capacity) +
            " disk_read_ahead_bytes=" +
            std::to_string(streaming ? streaming->ReadAheadBytes() : 0) +
            " slot_pinned_bytes=" + std::to_string(pinned));
  }
  // Metadata room the ledger keeps (kMetadataBaseBytes, kMetadataDiskShare).
  static std::size_t MetadataBytes(
      const std::optional<TextRunnerDiskCacheOptions>& options) {
    return SaturatingAdd(
        kMetadataBaseBytes,
        options ? options->capacity_bytes / kMetadataDiskShare : 0);
  }
  ~Impl() {
    // Settle immutable-source readers before slots/resources/index unwind.
    if (queue)
      queue->Drain();
    for (auto& entry : entries) {
      // Retained rows return to the backend before the slots unwind.
      entry->rewind.reset();
      entry->slot->StopIdleSpill();
    }
  }
  // Persistence completion of checkpoint `id` (queue worker).
  void Completed(cache::CheckpointId id, bool success,
                 const cache::TransferTiming& timing) {
    const std::lock_guard lock(mutex);
    std::shared_ptr<const void> branch;
    std::shared_ptr<DurableCapture> completed;
    for (auto& capture : pending_durable)
      if (capture && capture->manifest.checkpoint == id) {
        completed = capture;
        branch = capture->branch;
        capture.reset();
      }
    if (!success) {
      // A failed write-back no longer covers its frontier.
      if (completed && completed->write_back)
        for (auto& entry : entries)
          if (entry->branch == completed->branch &&
              entry->written_back == completed->manifest.tokens.size())
            entry->written_back = 0;
      Logger::Info("cache",
                   "schema=component-cache-v1 event=persistence_incomplete "
                   "checkpoint_id=" +
                       std::to_string(id.value) + " reason=\"" +
                       timing.failure + "\"");
      return;
    }
    // Indexing the publication is optional: on failure (metadata room) the
    // file stays on disk, unused until a restart finds it, and requests run
    // as without it.
    try {
      Track(id, branch, timing);
    } catch (const std::exception& error) {
      Logger::Info("cache",
                   "schema=component-cache-v1 event=publication_untracked "
                   "checkpoint_id=" +
                       std::to_string(id.value) + " reason=\"" + error.what() +
                       "\"");
    }
  }
  // Under the mutex: adopts the published checkpoint `id` in the catalog,
  // the index and the lineage of the execution it came from.
  void Track(cache::CheckpointId id, const std::shared_ptr<const void>& branch,
             const cache::TransferTiming& timing) {
    catalog->Track(id);
    if (const auto description = catalog->Find(id)) {
      // A published checkpoint is the durable ancestor of the execution it
      // came from, so a later write-back of that slot reuses its chunk
      // files instead of writing them again.
      // Only that execution: conversations forked from one shared
      // checkpoint share its lineage and may share its tokens, but
      // computed their own rows. A busy entry's frontier is not known
      // yet (see Covered); an idle one must still extend it.
      const auto& tokens = description->Manifest().tokens;
      for (auto& entry : entries)
        if (branch && entry->branch == branch &&
            (entry->busy || Prefixes(tokens, entry->live_tokens)) &&
            (!entry->durable ||
             entry->durable->Manifest().tokens.size() < tokens.size()))
          entry->durable = description;
      (void)index.Insert(identity, description);
      Logger::Info(
          "cache",
          "schema=component-cache-v1 event=published tokens=" +
              std::to_string(description->Manifest().tokens.size()) +
              " checkpoint_id=" + std::to_string(id.value) + " payload_bytes=" +
              std::to_string(PayloadBytes(description->Manifest())) +
              " purpose=" +
              std::string(PurposeName(description->Manifest().purpose)) +
              " yield_ms=" + std::to_string(timing.yield_ns / 1e6) +
              " stream_wait_ms=" + std::to_string(timing.stream_wait_ns / 1e6) +
              " filesystem_ms=" + std::to_string(timing.filesystem_ns / 1e6) +
              " io_lock_ms=" + std::to_string(timing.io_lock_ns / 1e6));
    }
  }
  // Frees disk room for `manifest` by evicting the least recently used
  // checkpoint. Only the queue worker writes the store, and it alone calls
  // this, so no other publication holds the store lock meanwhile.
  bool ReclaimDisk(const cache::DiskManifest& manifest) {
    // A checkpoint larger than the entire disk budget cannot be admitted by
    // deleting other conversations.
    const auto bytes = PayloadBytes(manifest);
    if (bytes > disk_capacity)
      return false;
    // Startup orphan reclamation holds the store lock asynchronously and
    // eviction only tries that lock, so a write racing it would be dropped.
    try {
      disk->WaitForReclamation();
    } catch (const std::exception& error) {
      Logger::Warn("cache", std::string("schema=component-cache-v1 "
                                        "event=disk_reclamation_failed "
                                        "reason=") +
                                error.what());
    }
    const std::lock_guard lock(mutex);
    std::shared_ptr<const cache::DiskDescription> evicted;
    if (!catalog->EvictOne(&evicted))
      return false;
    if (evicted) {
      const auto& victim = evicted->Manifest();
      Logger::Info(
          "cache",
          "schema=component-cache-v1 event=disk_evicted reason=lru "
          "checkpoint_id=" +
              std::to_string(victim.checkpoint.value) +
              " tokens=" + std::to_string(victim.tokens.size()) +
              " payload_bytes=" + std::to_string(PayloadBytes(victim)) +
              " managed_bytes=" + std::to_string(disk->Stats().managed_bytes) +
              " capacity_bytes=" + std::to_string(disk_capacity) +
              " for_tokens=" + std::to_string(manifest.tokens.size()));
    }
    return true;
  }
  // Under the mutex: the idle slot a cold acquisition takes, or null.
  Entry* SelectCold() {
    Entry* entry = nullptr;
    // A frontier far beyond every checkpoint of its prompt is the only
    // copy of that work, as for a conversation whose captures are refused:
    // take it last. Then completed
    // preservation first: it needs no foreground copy. Among those, a live
    // frontier holds generated tokens no checkpoint covers, such as the
    // tool call the next turn replays. Take a slot without one, then a
    // shallow frontier before a deep one, then the least recently used.
    // Faster idle preservation must not make short side requests evict a
    // deep conversation merely because its backing copy finished earlier.
    constexpr std::size_t kUncoveredTokens = 4096;
    const auto uncovered = [&](const Entry& e) {
      if (!e.live || e.live_tokens.size() <= kUncoveredTokens)
        return false;
      return !index.HasCheckpoint({identity, e.live_tokens, e.live_input},
                                  e.live_tokens.size() - kUncoveredTokens);
    };
    using Rank = std::tuple<bool, bool, bool, bool, std::uint64_t>;
    const auto rank = [&](const Entry& e) {
      return Rank{uncovered(e), !e.slot->PreservationComplete(), bool(e.live),
                  e.live_tokens.size() >= kUncoveredTokens, e.used};
    };
    std::optional<Rank> selected_rank;
    for (auto& candidate : entries) {
      if (candidate->busy)
        continue;
      if (!entry) {
        entry = candidate.get();
        continue;
      }
      if (!selected_rank)
        selected_rank = rank(*entry);
      const auto candidate_rank = rank(*candidate);
      if (candidate_rank < *selected_rank) {
        entry = candidate.get();
        selected_rank = candidate_rank;
      }
    }
    // Keep ordinary cold selection unchanged. Only a cancelled executor
    // with outstanding work needs an alternative; do not query every
    // stream or look up exact checkpoint coverage on normal admissions.
    if (entry && !entry->slot->ExecutionComplete()) {
      if (!selected_rank)
        selected_rank = rank(*entry);
      Entry* ready = nullptr;
      std::optional<Rank> ready_rank;
      for (auto& candidate : entries) {
        if (candidate.get() == entry || candidate->busy ||
            !candidate->slot->ExecutionComplete())
          continue;
        const auto candidate_rank = rank(*candidate);
        // A ready slot must not sacrifice uncovered work or introduce
        // foreground preservation. Keep a deep generated tail unless an
        // exact checkpoint or another idle live copy still covers it.
        const auto live_copy = [&] {
          return candidate->history &&
                 std::ranges::any_of(entries, [&](const auto& other) {
                   return other != candidate && !other->busy && other->live &&
                          other->history &&
                          other->history->Lineage() ==
                              candidate->history->Lineage() &&
                          other->live_tokens == candidate->live_tokens &&
                          other->live_input.Matches(
                              candidate->live_input,
                              candidate->live_tokens.size());
                 });
        };
        if (std::tie(std::get<0>(candidate_rank), std::get<1>(candidate_rank)) >
                std::tie(std::get<0>(*selected_rank),
                         std::get<1>(*selected_rank)) ||
            (candidate->live &&
             candidate->live_tokens.size() >= kUncoveredTokens &&
             !index.HasCheckpoint(
                 {identity, candidate->live_tokens, candidate->live_input},
                 candidate->live_tokens.size()) &&
             !live_copy()))
          continue;
        if (!ready || candidate_rank < *ready_rank) {
          ready = candidate.get();
          ready_rank = candidate_rank;
        }
      }
      if (ready)
        entry = ready;
    }
    return entry;
  }
  std::unique_ptr<cache::Stream> Stream() {
    auto stream = resources->stream();
    if (!stream)
      throw std::runtime_error("component cache transfer pool exhausted");
    return stream;
  }
  cache::DiskFileId File(std::uint64_t kind, std::uint64_t id) const {
    crypto::Sha256Hasher hash;
    hash.Update(nonce);
    std::array<std::uint8_t, 16> suffix{};
    for (std::size_t i = 0; i < 8; ++i) {
      suffix[i] = static_cast<std::uint8_t>(kind >> (8 * i));
      suffix[8 + i] = static_cast<std::uint8_t>(id >> (8 * i));
    }
    hash.Update(suffix);
    const auto bytes = hash.Finish();
    cache::DiskFileId file;
    std::copy_n(bytes.begin(), file.size(), file.begin());
    return file;
  }
  // Whether Persist would queue `capture` now: a snapshot is not copied for
  // a job it would refuse.
  bool CanQueue(const DurableCapture& capture) {
    const auto& manifest = capture.manifest;
    if (!capture.write_back && !capture.step_exempt &&
        manifest.purpose != cache::CheckpointPurpose::kLearned &&
        WithinDiskStep(manifest.tokens, manifest.input))
      return false;
    {
      const std::lock_guard lock(mutex);
      if (std::ranges::find(pending_durable, nullptr) == pending_durable.end())
        return false;
    }
    return queue->CanAccept(manifest, true);
  }
  bool WithinDiskStep(std::span<const cache::Token> tokens,
                      const cache::Identity& input) {
    const std::lock_guard lock(mutex);
    auto found =
        Lookup({identity, tokens, cache::InputIdentity(tokens.size(), input)});
    std::size_t prior = 0;
    for (const auto& candidate : found.candidates)
      if (Published(candidate) && SpacingAnchor(candidate))
        prior = std::max(prior, static_cast<std::size_t>(candidate.boundary));
    const auto compatible = [&](auto prefix, const auto& prefix_input) {
      return prefix_input == input && prefix.size() <= tokens.size() &&
             std::equal(prefix.begin(), prefix.end(), tokens.begin());
    };
    for (const auto& capture : pending_durable)
      if (capture && SpacingAnchor(capture->manifest.purpose) &&
          compatible(capture->manifest.tokens, capture->manifest.input))
        prior = std::max(prior, capture->manifest.tokens.size());
    return prior && tokens.size() - prior < disk_step;
  }
  std::size_t Persist(std::shared_ptr<DurableCapture> capture) {
    if (capture->sources.empty())
      return 0;
    if (!capture->write_back && !capture->step_exempt &&
        WithinDiskStep(capture->manifest.tokens, capture->manifest.input) &&
        capture->manifest.purpose != cache::CheckpointPurpose::kLearned)
      return 0;
    auto job = cache::PersistenceJob(
        ledger, File(1, capture->manifest.checkpoint.value), capture->manifest,
        std::move(capture->sources));
    if (capture->holds_slot || capture->host_snapshot || capture->rewind)
      job.KeepUnderPressure(capture->jump);
    std::size_t pending_slot;
    // A write-back covers its frontier from before submission: the job may
    // fail, and its callback clear the mark, before TrySubmit returns.
    Entry* covered = nullptr;
    std::size_t uncovered = 0;
    {
      const std::lock_guard lock(mutex);
      const auto position =
          std::find(pending_durable.begin(), pending_durable.end(), nullptr);
      if (position == pending_durable.end())
        return 0;
      pending_slot = position - pending_durable.begin();
      *position = capture;
      if (capture->write_back)
        for (auto& entry : entries)
          if (capture->branch && entry->branch == capture->branch) {
            covered = entry.get();
            uncovered = std::exchange(covered->written_back,
                                      capture->manifest.tokens.size());
          }
    }
    if (!queue->TrySubmit(std::move(job))) {
      const std::lock_guard lock(mutex);
      if (pending_durable[pending_slot] == capture)
        pending_durable[pending_slot].reset();
      if (covered && covered->branch == capture->branch &&
          covered->written_back == capture->manifest.tokens.size())
        covered->written_back = uncovered;
      return 0;
    }
    return PayloadBytes(capture->manifest);
  }
  // Queues the rewind point `point` of `entry` for the disk once, unless a
  // checkpoint already covers it within kUncoveredCaptureTokens. Its sources
  // are the point's own pieces (host arenas, retained device rows), so the
  // slot may be reset meanwhile; completed chunks of the entry's durable
  // ancestor are shared. With `step_exempt` (idle, reassignment and reset
  // writes) the disk step does not apply. Optional: returns the bytes
  // queued, zero when refused or failed (logged).
  std::size_t PersistRewind(Entry& entry,
                            const std::shared_ptr<RewindPoint>& point,
                            bool step_exempt, std::string_view reason,
                            bool exact_boundary = false) noexcept {
    if (!point || !point->persist || !queue || !resources->copy_observed_rows)
      return 0;
    const auto n = point->tokens.size();
    try {
      std::shared_ptr<const cache::DiskDescription> ancestor;
      std::shared_ptr<const void> branch;
      {
        const std::lock_guard lock(mutex);
        if ((point->pending_checkpoint &&
             std::ranges::any_of(pending_durable,
                                 [&](const auto& capture) {
                                   return capture &&
                                          capture->manifest.checkpoint.value ==
                                              point->pending_checkpoint;
                                 })) ||
            index.HasCheckpoint(
                {identity, point->tokens, point->inputs},
                exact_boundary
                    ? n
                    : (n > kUncoveredCaptureTokens ? n - kUncoveredCaptureTokens
                                                   : 0)))
          return 0;
        if (reason == "idle" && point->idle_attempt == entry.idle_since)
          return 0;
        ancestor = entry.durable;
        branch = entry.branch;
      }
      if (!step_exempt && WithinDiskStep(point->tokens, point->input))
        return 0;
      const auto descriptors = resources->adapter->Components();
      auto durable = std::make_shared<DurableCapture>();
      durable->metadata =
          ledger
              .Reserve(cache::ResourceCategory::kMetadata,
                       sizeof(DurableCapture) + n * sizeof(cache::Token) +
                           point->input.size() +
                           descriptors.size() * sizeof(cache::DiskComponent) +
                           point->pieces.size() *
                               (sizeof(cache::DiskPayload) +
                                sizeof(cache::PersistenceSource) + 256))
              .Convert();
      durable->branch = branch;
      durable->step_exempt = step_exempt;
      durable->rewind = true;
      if (ancestor) {
        const auto& previous = ancestor->Manifest();
        if (previous.tokens.size() > n ||
            !std::equal(previous.tokens.begin(), previous.tokens.end(),
                        point->tokens.begin()) ||
            !std::ranges::equal(previous.input,
                                point->inputs.At(previous.tokens.size())))
          ancestor.reset();
        else {
          durable->ancestor_pin = ancestor->Pin();
          if (!durable->ancestor_pin)
            ancestor.reset();
        }
      }
      auto& manifest = durable->manifest;
      manifest.checkpoint = cache::ExecutionHistory::AllocateCheckpointId();
      manifest.lineage =
          ancestor ? ancestor->Manifest().lineage : point->lineage;
      manifest.compatibility = digest;
      manifest.input = point->input;
      manifest.tokens = point->tokens;
      manifest.purpose = cache::CheckpointPurpose::kPrompt;
      durable->sources.reserve(point->pieces.size());
      auto piece = point->pieces.begin();
      for (const auto& d : descriptors) {
        const auto p = std::ranges::find(point->positions, d.id,
                                         &cache::ComponentPosition::id);
        if (p == point->positions.end())
          throw std::logic_error("incomplete rewind point positions");
        cache::DiskComponent component{.descriptor = d, .position = *p};
        const auto completed =
            d.kind == cache::ComponentKind::kAppendRows
                ? p->valid_rows / d.rows_per_chunk * d.rows_per_chunk
                : 0;
        cache::Rows inherited = 0;
        if (completed && ancestor) {
          const auto& components = ancestor->Manifest().components;
          const auto previous = std::ranges::find_if(
              components,
              [&](const auto& c) { return c.descriptor.id == d.id; });
          if (previous != components.end()) {
            const auto chunks = std::min<std::size_t>(
                previous->chunks.size(), completed / d.rows_per_chunk);
            component.chunks.assign(previous->chunks.begin(),
                                    previous->chunks.begin() + chunks);
            inherited = chunks * d.rows_per_chunk;
          }
        }
        for (; piece != point->pieces.end() && piece->component == d.id;
             ++piece) {
          if (!piece->private_state && piece->end <= inherited)
            continue;
          const bool private_file =
              piece->private_state || piece->first >= completed;
          const auto bytes = piece->bytes.size();
          const auto file =
              File(0, cache::ExecutionHistory::AllocateCheckpointId().value);
          const cache::DiskPayload payload{file, bytes, 0};
          if (piece->private_state)
            component.private_state = payload;
          else if (private_file)
            component.tail = payload;
          else
            component.chunks.push_back(payload);
          const auto alignment =
              piece->private_state ? std::size_t{1} : d.row_bytes;
          if (piece->device) {
            durable->device_rows = true;
            durable->sources.emplace_back(
                false, file, bytes, alignment, piece->owner,
                [source = piece->bytes, copy = resources->copy_observed_rows](
                    std::uint64_t offset, std::span<std::byte> destination,
                    cache::Stream& stream) {
                  return copy(source.subspan(offset, destination.size()),
                              destination, stream);
                });
          } else {
            durable->sources.emplace_back(private_file, file, bytes, alignment,
                                          piece->owner, piece->bytes);
          }
        }
        manifest.components.push_back(std::move(component));
      }
      if (piece != point->pieces.end())
        throw std::logic_error("rewind point pieces out of component order");
      if (!CanQueue(*durable))
        return 0;
      const auto checkpoint = manifest.checkpoint.value;
      const auto queued = Persist(durable);
      // A job that already completed is no longer pending: the next attempt
      // rechecks the index instead.
      if (queued) {
        const std::lock_guard lock(mutex);
        point->pending_checkpoint = checkpoint;
        // Refusal before submission leaves the idle allowance available.
        // A job that already completed (even failed) still consumes it.
        if (reason == "idle")
          point->idle_attempt = entry.idle_since;
      }
      Logger::Info(
          "cache",
          "schema=component-cache-v1 event=rewind_persisted "
          "reason=" +
              std::string(reason) + " tokens=" + std::to_string(n) +
              " checkpoint_id=" + std::to_string(manifest.checkpoint.value) +
              " queued_bytes=" + std::to_string(queued));
      return queued;
    } catch (const std::exception& error) {
      try {
        Logger::Info("cache",
                     "schema=component-cache-v1 event=rewind_persist_failed "
                     "tokens=" +
                         std::to_string(n) + " reason=\"" + error.what() +
                         "\"");
      } catch (...) {
      }
      return 0;
    }
  }
  // Takes the rewind point of `entry` before its slot is reset or restored:
  // written to disk first (unless covered), then dropped, so the retained
  // rows return to the backend with the slot's.
  void DropRewind(Entry& entry, std::string_view reason) noexcept {
    std::shared_ptr<RewindPoint> point;
    {
      const std::lock_guard lock(mutex);
      point = std::exchange(entry.rewind, {});
    }
    if (point)
      (void)PersistRewind(entry, point, true, reason);
  }
};

struct ComponentTextCache::Lease::Impl {
  ComponentTextCache::Impl* cache;
  ComponentTextCache::Impl::Entry* entry;
  cache::SlotLease slot;
  std::unique_ptr<TextRunnerState> state;
  cache::InputIdentity input;
  std::vector<cache::Token> prompt;
  std::size_t cached{}, restored{}, stable{};
  double restore_time{};
  bool disk_hit{}, active{true};
  // False keeps this lease's prompt off the disk: it captures nothing.
  bool persist{true};
  SnapshotPurpose purpose{SnapshotPurpose::kContinuation};
  // CaptureDurable for WriteBackIdle: the disk step does not apply.
  bool write_back{};
  // CaptureBeforeReset: the slot is reset right after the capture unless the
  // capture keeps it (hold_slot), so rows it lends would be retired unread.
  bool before_reset{};
  // An idle slot (WriteBackIdle, admission's reassignment capture) that a disk
  // capture may keep leased until the writer has read it (SlotHold).
  bool hold_slot{};
  // CaptureDurable moved `slot` into a SlotHold: the caller neither commits
  // it nor returns the entry to idle.
  bool held{};
  // Admission's reassignment capture: before keeping the slot for the
  // writer, CaptureDurable copies its state to host memory in the
  // foreground (SnapshotSlot), and keeps the slot only if that fails.
  bool snapshot_slot{};
  // A copy of a live frontier (CaptureBeforeReset, WriteBackIdle), published
  // as CheckpointPurpose::kGenerated.
  bool live_frontier{};
  std::vector<cache::Token> capture_tokens;
  // Reply edges of the live frontier this lease continues.
  std::vector<std::size_t> reply_starts, reply_ends;
  // The conversation rewrites generated replies (RewritesReply, or it was
  // served from a rewind point, or its slot is marked): this request keeps a
  // rewind point at its stable boundary.
  bool rewind_wanted{};
  // Served from its slot's rewind point.
  bool rewound{};
  // The rewind point this request captured, its slot's at Commit.
  std::shared_ptr<RewindPoint> rewind_point;
  ContinuationLookup lookup_result;
  ComponentCacheMetrics metrics;
  Impl(ComponentTextCache::Impl* cache, ComponentTextCache::Impl::Entry* entry,
       cache::SlotLease slot, std::span<const cache::Token> prompt,
       cache::InputIdentity input, std::size_t stable)
      : cache(cache),
        entry(entry),
        slot(std::move(slot)),
        state(cache->runner->BindComponentState(*cache->resources->adapter,
                                                this->slot.Execution())),
        input(std::move(input)),
        prompt(prompt.begin(), prompt.end()),
        stable(stable) {}
  std::unique_ptr<TextRunnerSnapshot> CaptureDurable();
  // Whether a capture at `boundary` becomes this request's rewind point: a
  // boundary past the reused one and before the prompt end.
  [[nodiscard]] bool RewindEligible(std::size_t boundary) const {
    return rewind_wanted && cache->snapshot_pool && !write_back &&
           !live_frontier && !before_reset && boundary > cached &&
           boundary < prompt.size();
  }
  // Keeps the slot's state at capture_tokens as this request's rewind point,
  // and queues it for the disk too when the disk step allows. Returns null
  // when refused (logged).
  std::unique_ptr<TextRunnerSnapshot> CaptureRewind();
  // Restores `point` into this lease's slot. Throws on failure; the caller
  // resets the slot.
  void Rewind(const RewindPoint& point);
  // Copies every piece from the slot into host arenas and adds them to
  // `durable` as host sources. Returns why it did not (the capture then
  // keeps the slot), or empty.
  struct SnapshotStats {
    double alloc_ms{}, copy_ms{};
    std::size_t reused_bytes{}, allocated_bytes{}, arenas{};
    // Observed rows of queued disk jobs given host copies: shared with the
    // frontier's pieces, or read again (extra).
    std::size_t observed{}, shared_bytes{}, extra_bytes{};
    // The queue check, planning, and attaching host copies to observed rows.
    double queue_ms{}, plan_ms{}, attach_ms{}, sources_ms{};
    std::size_t pieces{};
    // Bytes of the arenas taken, page and arena padding included.
    std::size_t taken_bytes{};
  };
  std::string SnapshotSlot(std::span<const SlotPiece> pieces,
                           DurableCapture& durable, SnapshotStats& stats);
  // Copies `pieces` from the slot to host arenas (as host sources of
  // `durable`, when given). With `observed`, before a reset, also gives the
  // slot's observed rows host copies, shared with those pieces where they
  // hold the same rows. Returns why it did not, or empty (also when there is
  // nothing to copy).
  // `hosted`, when given, receives each piece's host copy (in `pieces`
  // order), owning its arena.
  std::string HostCopy(
      std::span<const SlotPiece> pieces, DurableCapture* durable,
      SnapshotStats& stats, bool observed = true,
      std::vector<std::shared_ptr<const void>>* hosted = nullptr,
      const std::function<bool(std::size_t)>& admit = {});
};

std::unique_ptr<TextRunnerSnapshot>
ComponentTextCache::Lease::Impl::CaptureDurable() {
  if (purpose != SnapshotPurpose::kBranchPoint &&
      purpose != SnapshotPurpose::kHistory &&
      RewindEligible(capture_tokens.size()) &&
      std::equal(capture_tokens.begin(), capture_tokens.end(), prompt.begin()))
    // An optional rewind point the arenas refused leaves the ordinary disk
    // capture, as without one.
    if (auto snapshot = CaptureRewind())
      return snapshot;
  if (!persist || !cache->queue || !cache->resources->copy_observed_rows ||
      purpose == SnapshotPurpose::kHistory)
    throw cache::ResourceExhausted();
  const auto capture_input = At(input, capture_tokens.size());
  if (purpose != SnapshotPurpose::kBranchPoint && !write_back &&
      cache->WithinDiskStep(capture_tokens, capture_input)) {
    Logger::Info("cache",
                 "schema=component-cache-v1 event=capture_refused "
                 "reason=min_step tokens=" +
                     std::to_string(capture_tokens.size()));
    return {};
  }
  const auto positions = cache->resources->adapter->Positions(slot.Execution());
  const auto descriptors = cache->resources->adapter->Components();
  struct Copy {
    cache::PayloadRequest request;
    cache::DiskFileId file;
    bool private_file;
  };
  // Every piece this capture writes, for a direct write that reads each one
  // from the slot (or its host snapshot).
  using Piece = SlotPiece;
  std::size_t pieces = 0;
  for (const auto& d : descriptors) {
    const auto p =
        std::ranges::find(positions, d.id, &cache::ComponentPosition::id);
    if (p == positions.end())
      throw std::logic_error("incomplete durable capture positions");
    pieces += d.kind == cache::ComponentKind::kPrivateState
                  ? 1
                  : p->valid_rows / d.rows_per_chunk +
                        bool(p->valid_rows % d.rows_per_chunk);
  }
  // Admit metadata before vectors/owners. No append payload allocation is
  // admitted for the native rows: observations may be retired at any guard.
  auto admission = cache->ledger.Reserve(
      cache::ResourceCategory::kMetadata,
      sizeof(DurableCapture) + sizeof(ComponentSnapshot) +
          capture_tokens.size() * sizeof(cache::Token) + capture_input.size() +
          descriptors.size() * sizeof(cache::DiskComponent) +
          pieces *
              (sizeof(cache::DiskPayload) + sizeof(cache::PersistenceSource) +
               sizeof(Copy) + sizeof(Piece) + 256));
  auto durable = std::make_shared<DurableCapture>();
  durable->metadata = admission.Convert();
  durable->write_back = write_back;
  std::shared_ptr<const cache::DiskDescription> ancestor;
  {
    const std::lock_guard lock(cache->mutex);
    durable->branch = entry->branch;
    ancestor = entry->durable;
  }
  if (ancestor) {
    const auto& previous = ancestor->Manifest();
    if (previous.tokens.size() > capture_tokens.size() ||
        !std::equal(previous.tokens.begin(), previous.tokens.end(),
                    capture_tokens.begin()) ||
        previous.input != At(input, previous.tokens.size()))
      ancestor.reset();
    else {
      durable->ancestor_pin = ancestor->Pin();
      if (!durable->ancestor_pin)
        ancestor.reset();
    }
  }
  auto& manifest = durable->manifest;
  manifest.checkpoint = cache::ExecutionHistory::AllocateCheckpointId();
  manifest.lineage =
      ancestor ? ancestor->Manifest().lineage : entry->history->Lineage();
  manifest.compatibility = cache->digest;
  manifest.input = capture_input;
  manifest.tokens = capture_tokens;
  manifest.purpose = purpose == SnapshotPurpose::kBranchPoint
                         ? cache::CheckpointPurpose::kLearned
                     : live_frontier ? cache::CheckpointPurpose::kGenerated
                                     : cache::CheckpointPurpose::kPrompt;
  manifest.components.reserve(descriptors.size());
  durable->sources.reserve(pieces);
  std::vector<Copy> copies;
  copies.reserve(pieces);
  std::vector<Piece> all_pieces;
  all_pieces.reserve(pieces);
  // A capture before a reset that may keep its slot does: the writer then
  // reads every piece from it (SlotHold). Without the slot, the reset right
  // after would retire every row it lends, so it lends none and stages them.
  bool direct = before_reset && hold_slot;
  const bool lend = !before_reset;
  const auto device_rows = [&](cache::ComponentId id) {
    return std::ranges::find(cache->device_components, id) !=
           cache->device_components.end();
  };
  for (const auto& d : descriptors) {
    const auto p =
        std::ranges::find(positions, d.id, &cache::ComponentPosition::id);
    cache::DiskComponent component{.descriptor = d, .position = *p};
    const auto completed =
        d.kind == cache::ComponentKind::kAppendRows
            ? p->valid_rows / d.rows_per_chunk * d.rows_per_chunk
            : 0;
    cache::Rows inherited = 0;
    if (completed && ancestor) {
      const auto& components = ancestor->Manifest().components;
      const auto previous = std::ranges::find_if(
          components, [&](const auto& c) { return c.descriptor.id == d.id; });
      if (previous != components.end()) {
        const auto chunks = std::min<std::size_t>(previous->chunks.size(),
                                                  completed / d.rows_per_chunk);
        component.chunks.assign(previous->chunks.begin(),
                                previous->chunks.begin() + chunks);
        inherited = chunks * d.rows_per_chunk;
      }
    }
    // Completed rows the adapter retains in device storage are written from
    // those references, which no later mutation changes. The others are lent
    // as one view, or chunk by chunk when they lie on several runs of device
    // blocks; only rows the slot can neither retain nor lend are staged.
    const bool retains = !direct && device_rows(d.id);
    std::shared_ptr<cache::BorrowedRows> rows;
    std::vector<std::shared_ptr<cache::BorrowedRows>> chunk_rows;
    if (completed) {
      if (inherited != completed && !direct && !retains && lend) {
        auto view = cache->resources->adapter->BorrowRowsView(
            slot.Execution(), d.id, inherited, completed);
        if (!view.empty())
          rows = slot.Observe(d.id, inherited, completed, view);
        else if (completed - inherited > d.rows_per_chunk) {
          chunk_rows.resize((completed - inherited) / d.rows_per_chunk);
          for (std::size_t i = 0; i < chunk_rows.size(); ++i) {
            const auto first = inherited + i * d.rows_per_chunk;
            const auto end = first + d.rows_per_chunk;
            auto chunk = cache->resources->adapter->BorrowRowsView(
                slot.Execution(), d.id, first, end);
            if (!chunk.empty())
              chunk_rows[i] = slot.Observe(d.id, first, end, chunk);
          }
        }
      }
      component.chunks.reserve(completed / d.rows_per_chunk);
    }
    const auto capture = [&](cache::Rows first, cache::Rows end,
                             bool private_file) {
      const bool private_state = d.kind == cache::ComponentKind::kPrivateState;
      const auto bytes =
          private_state ? d.state_bytes : (end - first) * d.row_bytes;
      const auto file =
          cache->File(0, cache::ExecutionHistory::AllocateCheckpointId().value);
      all_pieces.push_back({d.id, private_state, first, end, bytes,
                            private_state ? std::size_t{1} : d.row_bytes, file,
                            private_file});
      if (direct)
        return cache::DiskPayload{file, bytes, 0};
      const auto chunk =
          private_file || chunk_rows.empty()
              ? std::shared_ptr<cache::BorrowedRows>{}
              : chunk_rows[(first - inherited) / d.rows_per_chunk];
      if (retains && !private_file) {
        auto retained = cache->resources->adapter->RetainRows(slot.Execution(),
                                                              d.id, first, end);
        if (retained.owner && retained.bytes >= bytes) {
          durable->device_rows = true;
          metrics.device_payload_bytes += bytes;
          auto owner = std::move(retained.owner);
          durable->sources.emplace_back(
              false, file, bytes, d.row_bytes, owner,
              [owner, copy = cache->resources->copy_observed_rows](
                  std::uint64_t offset, std::span<std::byte> destination,
                  cache::Stream& stream) {
                return copy(
                    {static_cast<const std::byte*>(owner.get()) + offset,
                     destination.size()},
                    destination, stream);
              });
          return cache::DiskPayload{file, bytes, 0};
        }
      }
      if (rows && !private_file) {
        durable->sources.emplace_back(false, file, bytes, d.row_bytes, rows,
                                      (first - inherited) * d.row_bytes,
                                      cache->resources->copy_observed_rows);
      } else if (chunk) {
        durable->sources.emplace_back(false, file, bytes, d.row_bytes, chunk, 0,
                                      cache->resources->copy_observed_rows);
      } else {
        const cache::PayloadRequest request{
            d.id,
            private_state  ? cache::ResourceCategory::kPrivateState
            : private_file ? cache::ResourceCategory::kPrivateTail
                           : cache::ResourceCategory::kBackingAssigned,
            first, end, bytes};
        copies.push_back({request, file, private_file});
      }
      return cache::DiskPayload{file, bytes, 0};
    };
    if (d.kind == cache::ComponentKind::kPrivateState)
      component.private_state = capture(0, 0, true);
    else {
      for (cache::Rows first = inherited; first < completed;
           first += d.rows_per_chunk)
        component.chunks.push_back(
            capture(first, first + d.rows_per_chunk, false));
      if (completed != p->valid_rows)
        component.tail = capture(completed, p->valid_rows, true);
    }
    manifest.components.push_back(std::move(component));
  }
  // Private states, tails, and rows the slot can neither retain nor lend are
  // copied now into host arenas (the snapshot pool), which the writer reads
  // and returns. A capture that may keep its slot does so instead when the
  // arenas have no room; any other capture is refused.
  if (!direct && !copies.empty()) {
    std::vector<SlotPiece> staged;
    staged.reserve(copies.size());
    std::size_t staged_bytes = 0;
    for (const auto& copy : copies) {
      const auto& r = copy.request;
      const bool private_state =
          r.category == cache::ResourceCategory::kPrivateState;
      staged.push_back({r.component, private_state, r.first, r.end, r.bytes,
                        private_state
                            ? std::size_t{1}
                            : std::ranges::find(descriptors, r.component,
                                                &cache::ComponentDescriptor::id)
                                  ->row_bytes,
                        copy.file, copy.private_file});
      staged_bytes += r.bytes;
    }
    SnapshotStats stats;
    // Before a reset, rows older queued jobs observe from this slot get host
    // copies too (shared with these pieces where they hold the same rows).
    const auto reason = HostCopy(staged, durable.get(), stats, before_reset);
    if (reason.empty()) {
      metrics.captured_bytes += staged_bytes;
      metrics.copied_payload_bytes += staged_bytes;
    } else {
      Logger::Info("cache",
                   "schema=component-cache-v1 event=capture_staging_refused "
                   "reason=" +
                       reason +
                       " tokens=" + std::to_string(capture_tokens.size()) +
                       " bytes=" + std::to_string(staged_bytes) +
                       " held_write=" + (hold_slot ? "true" : "false"));
      // A failed device copy fails the capture; only a lack of arena room
      // falls back to keeping the slot for the writer.
      if (reason == "copy_failed")
        throw std::runtime_error("capture staging copy failed");
      if (!hold_slot)
        throw cache::ResourceExhausted();
      direct = true;
    }
  }
  if (direct && snapshot_slot) {
    // Lent views, retained rows and staged copies taken above are released.
    copies.clear();
    durable->sources.clear();
    durable->device_rows = false;
    const auto started = Clock::now();
    std::size_t bytes = 0;
    for (const auto& piece : all_pieces)
      bytes += piece.bytes;
    SnapshotStats stats;
    const auto reason = SnapshotSlot(all_pieces, *durable, stats);
    const auto fields =
        " slot=" + std::to_string(slot.Location().slot.value) +
        " tokens=" + std::to_string(capture_tokens.size()) +
        " bytes=" + std::to_string(bytes) + " ms=" +
        std::to_string(
            std::chrono::duration<double, std::milli>(Clock::now() - started)
                .count());
    if (reason.empty()) {
      Logger::Info(
          "cache",
          "event=slot_snapshot" + fields +
              " queue_ms=" + std::to_string(stats.queue_ms) +
              " plan_ms=" + std::to_string(stats.plan_ms) +
              " alloc_ms=" + std::to_string(stats.alloc_ms) +
              " copy_ms=" + std::to_string(stats.copy_ms) +
              " attach_ms=" + std::to_string(stats.attach_ms) +
              " sources_ms=" + std::to_string(stats.sources_ms) +
              " pieces=" + std::to_string(stats.pieces) +
              " arenas=" + std::to_string(stats.arenas) +
              " reused_bytes=" + std::to_string(stats.reused_bytes) +
              " allocated_bytes=" + std::to_string(stats.allocated_bytes) +
              " observed_rows=" + std::to_string(stats.observed) +
              " shared_bytes=" + std::to_string(stats.shared_bytes) +
              " extra_bytes=" + std::to_string(stats.extra_bytes));
      durable->host_snapshot = true;
      ++metrics.captures;
      metrics.captured_checkpoint_ids.push_back(
          durable->manifest.checkpoint.value);
      auto snapshot = std::make_unique<ComponentSnapshot>();
      snapshot->bytes = cache->CaptureCopyBytes(*entry->history, positions);
      snapshot->durable = std::move(durable);
      return snapshot;
    }
    Logger::Info("cache", "event=slot_snapshot_fallback reason=" + reason +
                              fields + " held_write=true");
  }
  if (direct) {
    // The queue worker reads every piece from the slot through the bounded
    // disk staging; the sources keep it leased until the job is dropped. Lent
    // views, retained rows and staged copies taken above are released.
    copies.clear();
    durable->sources.clear();
    durable->device_rows = false;
    if (before_reset) {
      // The slot is reset once this write has read it. Rows older jobs
      // still read from it (a prompt-end capture) get host copies first; if
      // they cannot, this write waits its turn so those jobs read them
      // before the hold returns the slot.
      SnapshotStats stats;
      const auto reason = HostCopy({}, nullptr, stats);
      if (!reason.empty())
        durable->jump = false;
      if (!reason.empty() || stats.observed)
        Logger::Info(
            "cache",
            "event=held_write_observed_rows slot=" +
                std::to_string(slot.Location().slot.value) +
                " tokens=" + std::to_string(capture_tokens.size()) +
                (reason.empty()
                     ? " observed_rows=" + std::to_string(stats.observed) +
                           " extra_bytes=" + std::to_string(stats.extra_bytes)
                     : " in_order=true reason=" + reason));
    }
    auto* adapter = cache->resources->adapter.get();
    auto hold = std::make_shared<ComponentTextCache::Impl::SlotHold>(
        cache, entry, std::move(slot));
    held = true;
    durable->holds_slot = true;
    for (const auto& piece : all_pieces) {
      durable->sources.emplace_back(
          piece.private_file, piece.file, piece.bytes, piece.alignment, hold,
          [adapter, hold, piece](std::uint64_t offset,
                                 std::span<std::byte> destination,
                                 cache::Stream& stream) {
            auto& execution = hold->slot.Execution();
            if (piece.private_state)
              return adapter->CapturePrivatePiece(execution, piece.component,
                                                  offset, destination, stream);
            const auto row = piece.bytes / (piece.end - piece.first);
            const auto first = piece.first + offset / row;
            return adapter->CopyRowsOut(execution, piece.component, first,
                                        first + destination.size() / row,
                                        destination, stream);
          });
    }
    ++metrics.captures;
    metrics.captured_checkpoint_ids.push_back(
        durable->manifest.checkpoint.value);
    auto snapshot = std::make_unique<ComponentSnapshot>();
    snapshot->bytes = cache->CaptureCopyBytes(*entry->history, positions);
    snapshot->durable = std::move(durable);
    return snapshot;
  }
  ++metrics.captures;
  metrics.captured_checkpoint_ids.push_back(durable->manifest.checkpoint.value);
  auto snapshot = std::make_unique<ComponentSnapshot>();
  snapshot->bytes = cache->CaptureCopyBytes(*entry->history, positions);
  snapshot->durable = std::move(durable);
  return snapshot;
}

std::unique_ptr<TextRunnerSnapshot>
ComponentTextCache::Lease::Impl::CaptureRewind() {
  const auto started = Clock::now();
  const auto n = capture_tokens.size();
  auto& adapter = *cache->resources->adapter;
  const auto positions = adapter.Positions(slot.Execution());
  const auto descriptors = adapter.Components();
  const auto refused = [&](std::string_view why, std::size_t bytes) {
    Logger::Info("cache",
                 "schema=component-cache-v1 event=rewind_refused reason=" +
                     std::string(why) + " tokens=" + std::to_string(n) +
                     " host_bytes=" + std::to_string(bytes));
    return std::unique_ptr<TextRunnerSnapshot>{};
  };
  // Retained rows of an earlier point of this execution are the same device
  // rows: share them instead of retaining each chunk again.
  auto previous = rewind_point;
  if (!previous) {
    const std::lock_guard lock(cache->mutex);
    previous = entry->rewind;
  }
  std::map<std::pair<std::uint32_t, cache::Rows>, const RewindPoint::Piece*>
      shared;
  if (previous && previous->tokens.size() <= n &&
      std::equal(previous->tokens.begin(), previous->tokens.end(),
                 capture_tokens.begin()))
    for (const auto& piece : previous->pieces)
      if (piece.device)
        shared.emplace(std::pair(piece.component.value, piece.first), &piece);
  auto point = std::make_shared<RewindPoint>();
  point->tokens = capture_tokens;
  point->input = At(input, n);
  point->inputs = input.Through(n);
  point->positions = positions;
  point->lineage = entry->history->Lineage();
  point->persist = persist;
  // Host pieces, and the point's piece each fills.
  std::vector<SlotPiece> staged;
  std::vector<std::size_t> staged_at;
  std::size_t reused = 0;
  const auto host = [&](SlotPiece piece) {
    staged_at.push_back(point->pieces.size());
    point->pieces.push_back({piece.component,
                             piece.private_state,
                             piece.first,
                             piece.end,
                             {},
                             {},
                             false});
    point->host_bytes += piece.bytes;
    staged.push_back(piece);
  };
  for (const auto& d : descriptors) {
    const auto p =
        std::ranges::find(positions, d.id, &cache::ComponentPosition::id);
    if (p == positions.end())
      throw std::logic_error("incomplete rewind capture positions");
    if (d.kind == cache::ComponentKind::kPrivateState) {
      host({d.id, true, 0, 0, d.state_bytes, 1, {}, true});
      continue;
    }
    const auto completed = p->valid_rows / d.rows_per_chunk * d.rows_per_chunk;
    const bool retains = std::ranges::find(cache->device_components, d.id) !=
                         cache->device_components.end();
    for (cache::Rows first = 0; first < completed; first += d.rows_per_chunk) {
      const auto end = first + d.rows_per_chunk;
      const auto bytes = (end - first) * d.row_bytes;
      if (const auto it = shared.find(std::pair(d.id.value, first));
          it != shared.end() && it->second->end == end) {
        point->pieces.push_back(*it->second);
        point->device_bytes += bytes;
        ++reused;
        continue;
      }
      if (retains) {
        auto retained = adapter.RetainRows(slot.Execution(), d.id, first, end);
        if (retained.owner && retained.bytes >= bytes) {
          const auto* data =
              static_cast<const std::byte*>(retained.owner.get());
          point->pieces.push_back(
              {d.id, false, first, end, std::move(retained.owner),
               std::span<const std::byte>(data, bytes), true});
          point->device_bytes += bytes;
          continue;
        }
      }
      host({d.id, false, first, end, bytes, d.row_bytes, {}, false});
    }
    if (completed != p->valid_rows)
      host({d.id,
            false,
            completed,
            p->valid_rows,
            (p->valid_rows - completed) * d.row_bytes,
            d.row_bytes,
            {},
            true});
  }
  // Rewind points of all slots, and their queued disk writes, together
  // occupy at most a quarter of the snapshot arenas, which reassignment
  // snapshots and capture staging share.
  const auto budget = cache->RewindArenaBudget();
  if (point->host_bytes > budget)
    return refused("host_cap", point->host_bytes);
  point->metadata =
      cache->ledger
          .Reserve(cache::ResourceCategory::kMetadata,
                   sizeof(RewindPoint) + n * sizeof(cache::Token) +
                       point->input.size() +
                       positions.size() * sizeof(cache::ComponentPosition) +
                       point->pieces.size() * sizeof(RewindPoint::Piece))
          .Convert();
  // This point supersedes the slot's older one (and an earlier one of this
  // request), whose device rows it already shares. Their arenas are credited
  // to its reservation, taken before any arena (reassignment snapshots share
  // the pool), but they stay until it is admitted and copied: a refused
  // replacement keeps the older point usable.
  std::size_t credit = 0;
  {
    const std::lock_guard lock(cache->mutex);
    if (rewind_point)
      credit += rewind_point->ReleasableArenaBytes();
    if (entry->rewind && entry->rewind != rewind_point &&
        ComponentTextCache::Impl::Prefixes(entry->rewind->tokens,
                                           capture_tokens))
      credit += entry->rewind->ReleasableArenaBytes();
  }
  shared.clear();
  previous.reset();
  // The reservation is held from the exchange on: any later failure or
  // exception returns it, success hands it to the arenas' holder.
  auto arenas = std::make_shared<RewindArenas>();
  const auto admit = [&](std::size_t need) {
    auto charged = cache->rewind_arena_bytes.load();
    do {
      if (charged + need > budget + std::min(credit, charged))
        return false;
    } while (!cache->rewind_arena_bytes.compare_exchange_weak(charged,
                                                              charged + need));
    arenas->charge.charged = &cache->rewind_arena_bytes;
    arenas->charge.bytes = need;
    return true;
  };
  SnapshotStats stats;
  std::vector<std::shared_ptr<const void>> hosted;
  if (const auto reason =
          HostCopy(staged, nullptr, stats, false, &hosted, admit);
      !reason.empty())
    return refused(reason, point->host_bytes);
  point->arena_bytes = arenas->charge.bytes;
  point->arenas = arenas;
  point->host_pieces = staged.size();
  for (std::size_t i = 0; i < staged.size(); ++i) {
    auto& piece = point->pieces[staged_at[i]];
    const auto* data = static_cast<const std::byte*>(hosted[i].get());
    piece.bytes = {data, staged[i].bytes};
    piece.owner = std::shared_ptr<const void>(arenas, data);
  }
  arenas->arenas = std::move(hosted);
  // Admitted: the superseded points (and their arenas) go now.
  {
    std::shared_ptr<RewindPoint> superseded;
    {
      const std::lock_guard lock(cache->mutex);
      if (entry->rewind && ComponentTextCache::Impl::Prefixes(
                               entry->rewind->tokens, capture_tokens))
        superseded = std::exchange(entry->rewind, {});
    }
    rewind_point = point;
  }
  metrics.captured_bytes += point->host_bytes;
  metrics.copied_payload_bytes += point->host_bytes;
  // Also on the disk when its step allows, from the same pieces.
  const auto queued = persist && cache->queue
                          ? cache->PersistRewind(*entry, point, false, "prompt")
                          : 0;
  metrics.disk_queued_bytes += queued;
  Logger::Info("cache",
               "schema=component-cache-v1 event=rewind_captured slot=" +
                   std::to_string(slot.Location().slot.value) +
                   " tokens=" + std::to_string(n) +
                   " host_bytes=" + std::to_string(point->host_bytes) +
                   " device_bytes=" + std::to_string(point->device_bytes) +
                   " pieces=" + std::to_string(point->pieces.size()) +
                   " shared_pieces=" + std::to_string(reused) +
                   " copy_ms=" + std::to_string(stats.copy_ms) + " ms=" +
                   std::to_string(std::chrono::duration<double, std::milli>(
                                      Clock::now() - started)
                                      .count()) +
                   " disk_queued_bytes=" + std::to_string(queued));
  auto snapshot = std::make_unique<ComponentSnapshot>();
  snapshot->bytes = cache->CaptureCopyBytes(*entry->history, positions);
  return snapshot;
}

void ComponentTextCache::Lease::Impl::Rewind(const RewindPoint& point) {
  auto& resources = *cache->resources;
  auto& adapter = *resources.adapter;
  auto& execution = slot.Execution();
  adapter.BeginRestore(execution, point.positions);
  {
    TransferBatch transfers(
        std::clamp<std::size_t>(resources.transfer_batch_size, 1, 16));
    for (const auto& piece : point.pieces)
      transfers.Add([&] { return resources.stream(); },
                    [&](cache::Stream& stream) {
                      return piece.private_state
                                 ? adapter.LoadPrivate(execution,
                                                       piece.component,
                                                       piece.bytes, stream)
                                 : adapter.CopyRowsIn(
                                       execution, piece.component, piece.first,
                                       piece.end, piece.bytes, stream);
                    });
    transfers.Finish();
  }
  if (!adapter.Validate(execution, point.positions))
    throw std::runtime_error("rewind validation failed");
}

std::string ComponentTextCache::Lease::Impl::SnapshotSlot(
    std::span<const SlotPiece> pieces, DurableCapture& durable,
    SnapshotStats& stats) {
  if (pieces.empty())
    return "empty";
  const auto started = Clock::now();
  const bool queued = !cache->snapshot_pool || cache->CanQueue(durable);
  stats.queue_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  if (!queued)
    return "queue_full";
  return HostCopy(pieces, &durable, stats);
}
std::string ComponentTextCache::Lease::Impl::HostCopy(
    std::span<const SlotPiece> pieces, DurableCapture* durable,
    SnapshotStats& stats, bool with_observed,
    std::vector<std::shared_ptr<const void>>* hosted,
    const std::function<bool(std::size_t)>& admit) {
  auto& resources = *cache->resources;
  // Pack the pieces in the order the writer reads them (shared files before
  // private, by file ID), so each arena returns as early as it can. Each
  // piece starts on a page, keeping device reads on the fast path; a piece
  // larger than an arena takes one of its own.
  constexpr std::size_t kPage = 4096;
  const auto page = [](std::size_t bytes) {
    return (bytes + kPage - 1) / kPage * kPage;
  };
  const auto arena_bytes = page(
      std::max<std::size_t>(resources.reassignment_snapshot_arena_bytes, 1));
  // Rows of this slot that disk jobs still queued or writing observe (a
  // prompt-end capture): the reset would retire them unread, so they get host
  // copies too. Rows the frontier snapshot copies are shared from it; only
  // the others (below its first row, from an older ancestor) are read again,
  // chunk by chunk, as extra pieces after the frontier's.
  const auto plan_started = Clock::now();
  const auto observed =
      with_observed ? slot.ObservedRows() : decltype(slot.ObservedRows()){};
  if (pieces.empty() && observed.empty())
    return {};
  if (!cache->snapshot_pool)
    return resources.reassignment_snapshot_bytes ? "no_host_allocator"
                                                 : "disabled";
  const auto descriptors = resources.adapter->Components();
  // The frontier's row pieces by component and first row, for lookups.
  std::vector<std::size_t> by_row;
  by_row.reserve(pieces.size());
  for (std::size_t i = 0; i < pieces.size(); ++i)
    if (!pieces[i].private_state)
      by_row.push_back(i);
  const auto key = [&](std::size_t i) {
    return std::pair(pieces[i].component.value, pieces[i].first);
  };
  std::ranges::sort(by_row, {}, key);
  // The first piece of `component` ending after `row`, or none.
  const auto next_piece = [&](cache::ComponentId component,
                              cache::Rows row) -> const SlotPiece* {
    auto it = std::ranges::upper_bound(by_row, std::pair(component.value, row),
                                       {}, key);
    if (it != by_row.begin() && pieces[*std::prev(it)].component == component &&
        pieces[*std::prev(it)].end > row)
      --it;
    return it != by_row.end() && pieces[*it].component == component
               ? &pieces[*it]
               : nullptr;
  };
  struct Alias {
    std::size_t rows, offset, piece, piece_offset, bytes;
  };
  std::vector<Alias> aliases;
  std::vector<SlotPiece> extras;
  std::vector<std::pair<std::size_t, std::size_t>> extra_targets;
  for (std::size_t r = 0; r < observed.size(); ++r) {
    const auto& rows = *observed[r];
    const auto d = std::ranges::find(descriptors, rows.Component(),
                                     &cache::ComponentDescriptor::id);
    if (d == descriptors.end() || !d->row_bytes || !d->rows_per_chunk)
      return "observed_rows";
    for (auto row = rows.First(); row < rows.End();) {
      const auto* next = next_piece(rows.Component(), row);
      if (next && next->first <= row) {
        const auto end = std::min(rows.End(), next->end);
        aliases.push_back({r, (row - rows.First()) * d->row_bytes,
                           static_cast<std::size_t>(next - pieces.data()),
                           (row - next->first) * d->row_bytes,
                           (end - row) * d->row_bytes});
        row = end;
        continue;
      }
      auto end = next ? std::min(rows.End(), next->first) : rows.End();
      // Chunk-sized device reads, at chunk boundaries.
      end = std::min(end, (row / d->rows_per_chunk + 1) * d->rows_per_chunk);
      extras.push_back({rows.Component(),
                        false,
                        row,
                        end,
                        (end - row) * d->row_bytes,
                        d->row_bytes,
                        {},
                        false});
      extra_targets.push_back({r, (row - rows.First()) * d->row_bytes});
      row = end;
    }
  }
  // The frontier's pieces, then the extras, each on its own page.
  std::vector<std::size_t> order(pieces.size());
  std::iota(order.begin(), order.end(), 0);
  std::ranges::sort(order, [&](auto a, auto b) {
    return std::tie(pieces[a].private_file, pieces[a].file) <
           std::tie(pieces[b].private_file, pieces[b].file);
  });
  for (std::size_t i = 0; i < extras.size(); ++i)
    order.push_back(pieces.size() + i);
  const auto item = [&](std::size_t i) -> const SlotPiece& {
    return i < pieces.size() ? pieces[i] : extras[i - pieces.size()];
  };
  struct Placement {
    std::size_t arena, offset;
  };
  std::vector<Placement> placement(order.size());
  std::vector<std::size_t> sizes;
  // End of the pieces in the last standard arena; none open at first.
  std::optional<std::size_t> used;
  for (const auto i : order) {
    const auto bytes = item(i).bytes;
    if (bytes > arena_bytes) {
      sizes.push_back(page(bytes));
      placement[i] = {sizes.size() - 1, 0};
      used.reset();
      continue;
    }
    if (!used || page(*used) > arena_bytes - bytes) {
      sizes.push_back(arena_bytes);
      used = 0;
    }
    placement[i] = {sizes.size() - 1, page(*used)};
    used = page(*used) + bytes;
  }
  const auto started = Clock::now();
  stats.plan_ms =
      std::chrono::duration<double, std::milli>(started - plan_started).count();
  stats.pieces = order.size();
  // A caller's own budget, for the whole arenas, before any is taken.
  if (admit &&
      !admit(std::accumulate(sizes.begin(), sizes.end(), std::size_t{0})))
    return "arena_budget";
  auto taken = cache->snapshot_pool->Take(sizes, [&](std::size_t bytes) {
    return bytes <= HostSnapshotCeilingBytes() &&
           (!resources.snapshot_room || resources.snapshot_room(bytes));
  });
  stats.alloc_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  if (!taken.refused.empty()) {
    if (!taken.error.empty())
      Logger::Info(
          "cache",
          std::string("event=slot_snapshot_allocation_failed error=\"") +
              taken.error + "\"");
    return taken.refused;
  }
  stats.reused_bytes = taken.reused;
  stats.taken_bytes =
      std::accumulate(sizes.begin(), sizes.end(), std::size_t{0});
  stats.allocated_bytes = taken.allocated;
  stats.arenas = sizes.size();
  const auto at = [&](std::size_t i) {
    return taken.arenas[placement[i].arena]->Data() + placement[i].offset;
  };
  auto& adapter = *resources.adapter;
  const auto copy_started = Clock::now();
  try {
    // The scheduler runs no other model work meanwhile, so the device reads
    // run at full speed; batch them as the reassignment copy does.
    // An interrupted decode can invalidate the execution state without
    // overwriting its observed prefix rows. Read those through their pinned
    // views, like the disk writer, rather than the adapter's state capture.
    // Pins outlive transfers even when a submission or completion throws.
    std::vector<cache::RowPin> pins;
    pins.reserve(extras.size());
    TransferBatch transfers(
        std::clamp<std::size_t>(std::max(resources.transfer_batch_size,
                                         resources.reassignment_batch_size),
                                1, 16));
    for (std::size_t i = 0; i < order.size(); ++i) {
      const auto& piece = item(i);
      const std::span<std::byte> destination(at(i), piece.bytes);
      if (i >= pieces.size()) {
        const auto [rows, offset] = extra_targets[i - pieces.size()];
        auto pin = observed[rows]->PinOrCopyHosted(offset, destination);
        if (!pin)
          continue;
        const auto source = pin->Bytes().subspan(offset, piece.bytes);
        pins.push_back(std::move(*pin));
        transfers.Add([&] { return resources.stream(); },
                      [&](cache::Stream& stream) {
                        return resources.copy_observed_rows(source, destination,
                                                            stream);
                      });
        continue;
      }
      transfers.Add([&] { return resources.stream(); },
                    [&](cache::Stream& stream) {
                      return piece.private_state
                                 ? adapter.CapturePrivate(slot.Execution(),
                                                          piece.component,
                                                          destination, stream)
                                 : adapter.CopyRowsOut(slot.Execution(),
                                                       piece.component,
                                                       piece.first, piece.end,
                                                       destination, stream);
                    });
    }
    transfers.Finish();
    stats.copy_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - copy_started)
            .count();
    const auto owner = [&](std::size_t i) {
      return std::shared_ptr<const void>(taken.arenas[placement[i].arena],
                                         at(i));
    };
    // Attach each observed range's copies in one call: one lock of its rows,
    // never waiting for the writer's device reads (they hold no lock).
    const auto attach_started = Clock::now();
    std::vector<std::vector<cache::BorrowedRows::HostCopy>> copies(
        observed.size());
    for (const auto& alias : aliases) {
      copies[alias.rows].push_back(
          {alias.offset,
           std::span<const std::byte>(at(alias.piece) + alias.piece_offset,
                                      alias.bytes),
           owner(alias.piece)});
      stats.shared_bytes += alias.bytes;
    }
    for (std::size_t e = 0; e < extras.size(); ++e) {
      const auto i = pieces.size() + e;
      const auto [rows, offset] = extra_targets[e];
      copies[rows].push_back(
          {offset, std::span<const std::byte>(at(i), extras[e].bytes),
           owner(i)});
      stats.extra_bytes += extras[e].bytes;
    }
    for (std::size_t r = 0; r < observed.size(); ++r)
      observed[r]->Host(std::move(copies[r]));
    stats.observed = observed.size();
    const auto sources_started = Clock::now();
    stats.attach_ms = std::chrono::duration<double, std::milli>(
                          sources_started - attach_started)
                          .count();
    if (hosted) {
      hosted->clear();
      for (std::size_t i = 0; i < pieces.size(); ++i)
        hosted->push_back(owner(i));
    }
    for (std::size_t i = 0; durable && i < pieces.size(); ++i) {
      const auto& piece = pieces[i];
      const auto* data = at(i);
      durable->sources.emplace_back(
          piece.private_file, piece.file, piece.bytes, piece.alignment,
          std::shared_ptr<const void>(taken.arenas[placement[i].arena], data),
          std::span<const std::byte>(data, piece.bytes));
    }
    stats.sources_ms = std::chrono::duration<double, std::milli>(
                           Clock::now() - sources_started)
                           .count();
  } catch (const std::exception& error) {
    if (durable)
      durable->sources.clear();
    Logger::Info("cache",
                 std::string("event=slot_snapshot_copy_failed error=\"") +
                     error.what() + "\"");
    return "copy_failed";
  }
  return {};
}

ComponentTextCache::ComponentTextCache(
    std::shared_ptr<TextModelRunner> runner, std::size_t count,
    const std::optional<TextRunnerDiskCacheOptions>& disk, std::size_t ceiling)
    : impl_(std::make_unique<Impl>(std::move(runner), count, disk, ceiling)) {}
ComponentTextCache::~ComponentTextCache() {
  // Graceful shutdown: once queued checkpoints are durable, write the live
  // frontiers no checkpoint covers, sharing those checkpoints' chunks. A
  // conversation stopped right after its turn otherwise resumes at its prompt
  // boundary. ~Impl drains these writes.
  if (!impl_->queue)
    return;
  impl_->queue->Drain();
  (void)WriteBackIdle(std::chrono::milliseconds::zero());
}
std::size_t ComponentTextCache::capacity() const noexcept {
  return impl_->entries.size();
}

bool ComponentTextCache::CaptureBeforeReset(std::size_t index,
                                            cache::BorrowedLocation location,
                                            std::vector<cache::Token> tokens,
                                            cache::InputIdentity input,
                                            std::string_view reason, bool hold,
                                            bool* held_out) noexcept {
  auto* entry = impl_->entries[index].get();
  const auto started = Clock::now();
  // A write-back still queued from this slot fails once the reset retires the
  // rows it observes: without a durable checkpoint covering the frontier, the
  // capture goes to disk itself, as the write-back would have.
  bool write_back, persist;
  {
    std::shared_ptr<RewindPoint> point;
    {
      const std::lock_guard lock(impl_->mutex);
      if (entry->frontier_dead) {
        if (held_out)
          *held_out = false;
        return false;
      }
      if (entry->drops_reply && entry->rewind &&
          Impl::RewindCurrent(*entry, *entry->rewind))
        point = entry->rewind;
    }
    // No later prompt of this conversation continues the frontier: its
    // rewind point is what goes to disk.
    if (point) {
      (void)impl_->PersistRewind(*entry, point, true, reason);
      if (held_out)
        *held_out = false;
      const std::lock_guard lock(impl_->mutex);
      entry->busy = true;
      return false;
    }
  }
  {
    const std::lock_guard lock(impl_->mutex);
    persist = entry->live_persist;
    write_back =
        persist && impl_->queue &&
        Impl::Durable(*entry) + Impl::kUncoveredCaptureTokens <= tokens.size();
  }
  bool held = false, writing = false;
  const auto log = [&](std::string_view event, std::string_view detail) {
    Logger::Info("cache", std::string("event=") + std::string(event) +
                              " reason=" + std::string(reason) +
                              " slot=" + std::to_string(index + 1) +
                              " tokens=" + std::to_string(tokens.size()) +
                              std::string(detail));
  };
  try {
    Lease lease(std::make_unique<Lease::Impl>(
        impl_.get(), entry, entry->slot->Acquire(location), tokens,
        std::move(input), tokens.size()));
    lease.impl_->write_back = write_back;
    lease.impl_->live_frontier = true;
    lease.impl_->before_reset = true;
    lease.impl_->hold_slot = hold;
    // Only admission holds here: copy the state out instead where it can.
    lease.impl_->snapshot_slot = hold;
    lease.impl_->persist = persist;
    try {
      if (!lease.TryReserveSnapshot(lease.CaptureBytes(), tokens.size(), false,
                                    SnapshotPurpose::kRetry, tokens))
        log("live_frontier_capture_refused", " cause=no_disk_cache");
      else {
        (void)lease.PublishSnapshot(tokens, lease.Capture(), false);
        if (lease.impl_->metrics.disk_queued_bytes == 0)
          // Within the disk step of a checkpoint, or the queue refused it.
          log("live_frontier_capture_refused", " cause=not_queued");
        else
          log("live_frontier_captured",
              " disk_queued_bytes=" +
                  std::to_string(lease.impl_->metrics.disk_queued_bytes) +
                  " ms=" +
                  std::to_string(std::chrono::duration<double, std::milli>(
                                     Clock::now() - started)
                                     .count()));
      }
    } catch (const cache::ResourceExhausted&) {
      log("live_frontier_capture_refused", " cause=byte_capacity");
    } catch (const std::bad_alloc&) {
      log("live_frontier_capture_refused", " cause=allocation_failed");
    } catch (const std::exception& error) {
      // Optional: the request proceeds cold-reassigned, as without a capture.
      log("live_frontier_capture_failed",
          std::string(" error=") + error.what());
    }
    // The capture only reads the slot, which stays valid and busy; the
    // caller's cold lease preserves the rows a new checkpoint borrows and
    // resets it. A held slot belongs to the queued write instead, and
    // returns to idle when it is written (or dropped unwritten).
    held = lease.impl_->held;
    writing = held && lease.impl_->metrics.disk_queued_bytes != 0;
    if (!held)
      lease.impl_->slot.Commit();
    lease.impl_->active = false;
  } catch (const std::exception& error) {
    log("live_frontier_capture_failed", std::string(" error=") + error.what());
  }
  if (held_out)
    *held_out = held;
  // A lease that could not be taken leaves the entry to the caller.
  if (!held) {
    const std::lock_guard lock(impl_->mutex);
    entry->busy = true;
  }
  return writing;
}

void ComponentTextCache::HostObservedRows(std::size_t index) noexcept {
  try {
    auto* entry = impl_->entries[index].get();
    cache::BorrowedLocation location;
    std::vector<cache::Token> tokens;
    cache::InputIdentity input;
    {
      const std::lock_guard lock(impl_->mutex);
      if (!entry->live)
        return;
      location = entry->live_location;
      tokens = entry->live_tokens;
      input = entry->live_input;
    }
    Lease lease(std::make_unique<Lease::Impl>(
        impl_.get(), entry, entry->slot->Acquire(location), tokens,
        std::move(input), tokens.size()));
    Lease::Impl::SnapshotStats stats;
    const auto reason = lease.impl_->HostCopy({}, nullptr, stats);
    if (!reason.empty() || stats.observed)
      Logger::Info("cache", "event=observed_rows_hosted slot=" +
                                std::to_string(index + 1) +
                                " rows=" + std::to_string(stats.observed) +
                                " bytes=" + std::to_string(stats.extra_bytes) +
                                (reason.empty() ? "" : " refused=" + reason));
    // Only read: the slot keeps its generation until the caller resets it.
    lease.impl_->slot.Commit();
    lease.impl_->active = false;
  } catch (const std::exception& error) {
    try {
      Logger::Info("cache", std::string("event=observed_rows_hosted slot=") +
                                std::to_string(index + 1) +
                                " error=" + error.what());
    } catch (...) {
    }
  } catch (...) {
  }
}

bool ComponentTextCache::PrepareAdmission(
    std::span<const ContinuationToken> prompt, const TextPromptContext* context,
    bool reuse, std::size_t stable) {
  std::optional<Impl::Uncovered> uncovered;
  Impl::Entry* entry = nullptr;
  {
    const std::lock_guard lock(impl_->mutex);
    const auto found =
        impl_->Lookup({impl_->identity, prompt, Inputs(prompt.size(), context),
                       stable, reuse});
    // Acquire continues an idle slot from its rewind point.
    if (reuse && impl_->FindRewind(found, prompt,
                                   Inputs(prompt.size(), context), context))
      return true;
    if (found.selected && found.selected->live) {
      const auto& live =
          *impl_->entries.at(found.selected->live->slot.value - 1);
      // Its own frontier is being written: continue it once that is done.
      if (live.writing)
        return false;
      if (!live.busy)
        return true;
    }
    entry = impl_->SelectCold();
    if (!entry)
      // A finishing write returns a slot.
      return std::ranges::none_of(impl_->entries,
                                  [](const auto& e) { return e->writing; });
    if (!impl_->queue)
      return true;
    for (std::size_t i = 0; i < impl_->entries.size(); ++i)
      if (impl_->entries[i].get() == entry)
        uncovered = impl_->UncoveredLive(i);
    if (!uncovered || (entry->held_tried == uncovered->tokens.size() &&
                       entry->held_branch == entry->branch))
      return true;
    entry->held_tried = uncovered->tokens.size();
    entry->held_branch = entry->branch;
    entry->busy = true;
  }
  const bool writing = CaptureBeforeReset(
      uncovered->index, uncovered->location, std::move(uncovered->tokens),
      std::move(uncovered->input), "reassignment", true);
  if (!writing) {
    // Captured from a host snapshot now, or refused: Acquire proceeds as
    // before.
    const std::lock_guard lock(impl_->mutex);
    if (!entry->writing)
      entry->busy = false;
  }
  return !writing;
}

bool ComponentTextCache::HostWritesPending() const {
  if (!impl_->queue)
    return false;
  const auto stats = impl_->queue->Stats();
  return stats.pending != 0 || stats.pinned_bytes != 0;
}

bool ComponentTextCache::WriteBackDeferred() const noexcept {
  return impl_->write_back_deferred.load(std::memory_order_relaxed);
}

std::size_t ComponentTextCache::ReleaseIdleRewinds(
    std::chrono::milliseconds idle_for) {
  if (!impl_->queue)
    return 0;
  const auto now = Clock::now();
  std::size_t released = 0;
  for (auto& entry : impl_->entries) {
    std::shared_ptr<RewindPoint> uncovered, retired;
    {
      const std::lock_guard lock(impl_->mutex);
      // The scheduler observes no queued requests before maintenance.
      // Be conservative for callers outside it: no slot may be in use.
      if (std::ranges::any_of(impl_->entries,
                              [](const auto& e) { return e->busy; }))
        return released;
      auto& point = entry->rewind;
      if (!point || !point->persist || now - entry->idle_since < idle_for ||
          !entry->live || !entry->holds_rows ||
          !Impl::Prefixes(point->tokens, entry->live_tokens))
        continue;
      // Publication can precede the writer dropping its sources. Neither a
      // pending job nor shared arenas count as memory we can release yet.
      if (std::ranges::any_of(impl_->pending_durable,
                              [&](const auto& capture) {
                                return capture &&
                                       capture->manifest.checkpoint.value ==
                                           point->pending_checkpoint;
                              }) ||
          point->ReleasableArenaBytes() != point->arena_bytes)
        continue;
      if (!impl_->index.HasCheckpoint(
              {impl_->identity, point->tokens, point->inputs},
              point->tokens.size())) {
        uncovered = point;
      } else {
        const auto idle_seconds =
            std::chrono::duration_cast<std::chrono::seconds>(now -
                                                             entry->idle_since)
                .count();
        Logger::Info("cache",
                     "schema=component-cache-v1 event=rewind_released "
                     "reason=idle host_bytes=" +
                         std::to_string(point->host_bytes) +
                         " idle_s=" + std::to_string(idle_seconds));
        // Only the retained references go away; the slot still owns its KV
        // rows and live frontier. ArenaCharge follows the last arena owner.
        if (Impl::RewindCurrent(*entry, *point))
          entry->frontier_dead = true;
        retired = std::exchange(point, {});
        ++released;
      }
    }
    // Unlike ordinary write-back's nearby coverage, release requires this
    // exact boundary. Failures retain the point; PersistRewind bounds idle
    // attempts until another turn commits.
    if (uncovered)
      (void)impl_->PersistRewind(*entry, uncovered, true, "idle", true);
  }
  return released;
}

std::size_t ComponentTextCache::WriteBackIdle(
    std::chrono::milliseconds idle_for) noexcept {
  impl_->write_back_deferred.store(false, std::memory_order_relaxed);
  if (!impl_->queue)
    return 0;
  const auto now = Clock::now();
  std::size_t queued = 0;
  bool deferred = false;
  for (std::size_t i = 0; i < impl_->entries.size(); ++i) {
    auto* entry = impl_->entries[i].get();
    cache::BorrowedLocation location;
    std::vector<cache::Token> tokens;
    cache::InputIdentity input;
    {
      // A conversation that rewrites its replies writes its rewind point
      // instead of the frontier no later prompt continues.
      std::shared_ptr<RewindPoint> point;
      {
        const std::lock_guard lock(impl_->mutex);
        if (entry->drops_reply && entry->rewind && !entry->busy &&
            now - entry->idle_since >= idle_for &&
            Impl::RewindCurrent(*entry, *entry->rewind))
          point = entry->rewind;
      }
      if (point) {
        const auto reason =
            idle_for == std::chrono::milliseconds::zero() ? "shutdown" : "idle";
        queued += impl_->PersistRewind(*entry, point, true, reason) != 0;
        continue;
      }
    }
    {
      const std::lock_guard lock(impl_->mutex);
      const auto size = entry->live_tokens.size();
      // Shutdown (idle_for zero) retries frontiers an earlier attempt could
      // not queue.
      const auto covered =
          idle_for == std::chrono::milliseconds::zero()
              ? Impl::Covered(*entry)
              : std::max(Impl::Covered(*entry), entry->write_back_tried);
      if (entry->frontier_dead || entry->busy || !entry->live ||
          !entry->holds_rows || !entry->live_persist ||
          now - entry->idle_since < idle_for ||
          size < Impl::kUncoveredCaptureTokens ||
          (covered && covered + Impl::kUncoveredCaptureTokens > size))
        continue;
      // A frontier extending a checkpoint of its own lineage still queued
      // would write that checkpoint's chunks again: wait until it is durable
      // (or dropped). The cache destructor writes what is still waiting.
      const auto& live = entry->live_input;
      const auto extends = [&](const auto& prefix, const auto& prefix_input) {
        return prefix.size() < size &&
               At(live, prefix.size()) == prefix_input &&
               std::equal(prefix.begin(), prefix.end(),
                          entry->live_tokens.begin());
      };
      // Only this execution's own pending checkpoints may cover it: the
      // persistence callback adopts no other (DurableCapture::branch).
      if (std::ranges::any_of(impl_->pending_durable, [&](const auto& capture) {
            return capture && capture->branch == entry->branch &&
                   extends(capture->manifest.tokens, capture->manifest.input);
          })) {
        // Written once that write settles: the caller tries again.
        deferred = true;
        continue;
      }
      entry->busy = true;
      location = entry->live_location;
      tokens = entry->live_tokens;
      input = entry->live_input;
    }
    const auto started = Clock::now();
    bool submitted = false, failed = false, held = false;
    try {
      Lease lease(std::make_unique<Lease::Impl>(
          impl_.get(), entry, entry->slot->Acquire(location), tokens,
          std::move(input), tokens.size()));
      lease.impl_->write_back = true;
      lease.impl_->live_frontier = true;
      lease.impl_->hold_slot = true;
      lease.impl_->capture_tokens = tokens;
      try {
        if (auto snapshot = lease.impl_->CaptureDurable()) {
          (void)lease.PublishSnapshot(tokens, std::move(snapshot), false);
          submitted = lease.impl_->metrics.disk_queued_bytes != 0;
        }
      } catch (const std::exception& error) {
        failed = true;
        Logger::Info(
            "cache",
            std::string("event=live_frontier_write_back_failed slot=") +
                std::to_string(i + 1) + " tokens=" +
                std::to_string(tokens.size()) + " error=" + error.what());
      }
      // The capture only reads the slot. Committing keeps its generation, so
      // the live frontier stays valid for the conversation's next turn
      // whether or not it was written. A held slot is committed and returned
      // to idle by its queued write (SlotHold).
      held = lease.impl_->held;
      if (!held)
        lease.impl_->slot.Commit();
      lease.impl_->active = false;
      if (!failed)
        Logger::Info(
            "cache",
            std::string("event=live_frontier_written_back slot=") +
                std::to_string(i + 1) +
                " tokens=" + std::to_string(tokens.size()) +
                " queued=" + (submitted ? "true" : "false") + " ms=" +
                std::to_string(std::chrono::duration<double, std::milli>(
                                   Clock::now() - started)
                                   .count()));
    } catch (const std::exception& error) {
      // Leasing the slot failed: nothing was read and the frontier is as it
      // was.
      Logger::Info("cache",
                   std::string("event=live_frontier_write_back_failed slot=") +
                       std::to_string(i + 1) +
                       " tokens=" + std::to_string(tokens.size()) +
                       " error=" + error.what());
    }
    const std::lock_guard lock(impl_->mutex);
    // Only a queued write covers the frontier (Persist marks it); shutdown
    // retries the rest.
    entry->write_back_tried = tokens.size();
    if (!held)
      entry->busy = false;
    queued += submitted;
  }
  impl_->write_back_deferred.store(deferred, std::memory_order_relaxed);
  return queued;
}

ComponentTextCache::RowReclaim ComponentTextCache::ReclaimRows(
    std::span<const ContinuationToken> prompt, bool reuse,
    const std::function<bool()>& enough, const TextPromptContext* context,
    std::size_t stable) {
  std::vector<const Impl::Entry*> visited;
  bool observed = false;
  std::optional<cache::BorrowedLocation> protected_live;
  {
    const std::lock_guard lock(impl_->mutex);
    auto found = impl_->Lookup({impl_->identity, prompt,
                                Inputs(prompt.size(), context), stable, reuse});
    // Token-equal cold executions can own distinct physical blocks; only the
    // selected live frontier serves this request, including its image
    // identity and stable boundary.
    if (found.selected)
      protected_live = found.selected->live;
    if (reuse)
      if (const auto choice = impl_->FindRewind(
              found, prompt, Inputs(prompt.size(), context), context))
        protected_live = impl_->entries[choice->index]->live_location;
  }
  while (!enough()) {
    Impl::Entry* entry = nullptr;
    std::size_t slot = 0, live_tokens = 0;
    std::optional<Impl::Uncovered> uncovered;
    bool live = false;
    cache::BorrowedLocation location{};
    {
      const std::lock_guard lock(impl_->mutex);
      const auto covered = [&](const Impl::Entry& e) {
        return impl_->index.HasCheckpoint(
            {impl_->identity, e.live_tokens, e.live_input},
            e.live_tokens.size());
      };
      using Rank = std::tuple<bool, bool, std::uint64_t>;
      std::optional<Rank> selected;
      bool busy = false;
      for (std::size_t i = 0; i < impl_->entries.size(); ++i) {
        auto& candidate = impl_->entries[i];
        if (candidate->busy) {
          busy = true;
          continue;
        }
        if (!candidate->holds_rows ||
            std::ranges::find(visited, candidate.get()) != visited.end() ||
            (candidate->live && protected_live &&
             protected_live->slot.value == i + 1))
          continue;
        const Rank rank{bool(candidate->live),
                        candidate->live && !covered(*candidate),
                        candidate->used};
        if (!selected || rank < *selected) {
          entry = candidate.get();
          slot = i + 1;
          selected = rank;
        }
      }
      if (!entry)
        // Rows return later only from a busy slot, a slot a queued disk job
        // still reads, or device rows a disk job keeps until it ends.
        return busy || observed || impl_->WritesHoldDeviceRows()
                   ? RowReclaim::kWaiting
                   : RowReclaim::kExhausted;
      visited.push_back(entry);
      entry->busy = true;
      location = entry->live_location;
      live = bool(entry->live);
    }
    // A queued disk job may still read rows this slot lent it: resetting
    // would wait on the scheduler thread for an active write, or fail a
    // queued one. Leave the slot until the job ends.
    if (live) {
      bool lent = false;
      try {
        auto lease = entry->slot->Acquire(location);
        lent = !lease.ObservedRows().empty();
        lease.Commit();
      } catch (...) {
        lent = true;
      }
      if (lent) {
        const std::lock_guard lock(impl_->mutex);
        entry->busy = false;
        observed = true;
        continue;
      }
    }
    {
      // With a disk cache, the uncovered frontier goes to disk, copied to
      // host memory first (or written from the slot it then keeps until the
      // writer has read it); without one it is dropped.
      const std::lock_guard lock(impl_->mutex);
      uncovered = impl_->UncoveredLive(slot - 1);
      if (uncovered && !impl_->device_components.empty() &&
          (!impl_->queue || !entry->live_persist)) {
        Logger::Info("cache",
                     "event=live_frontier_dropped reason=row_capacity "
                     "cause=no_disk_cache slot=" +
                         std::to_string(slot) +
                         " tokens=" + std::to_string(uncovered->tokens.size()));
        uncovered.reset();
      }
    }
    if (uncovered) {
      const bool device_rows = !impl_->device_components.empty();
      bool held = false;
      (void)CaptureBeforeReset(
          uncovered->index, uncovered->location, std::move(uncovered->tokens),
          std::move(uncovered->input), "row_capacity", device_rows, &held);
      // A held write keeps the slot until the writer, which yields to the
      // model for at most kPersistenceYieldLimit, has read it; the slot then
      // returns, its frontier covered, for a later reclaim.
      if (held)
        continue;
    }
    // Its rewind point retains the slot's rows: written to disk unless
    // covered, then dropped with them.
    impl_->DropRewind(*entry, "row_capacity");
    {
      const std::lock_guard lock(impl_->mutex);
      entry->branch = std::make_shared<const std::uint8_t>(0);
      entry->durable.reset();
      entry->written_back = 0;
      entry->write_back_tried = 0;
      entry->frontier_dead = false;
      live_tokens = entry->live_tokens.size();
      if (entry->live) {
        impl_->index.Erase(*entry->live);
        entry->live.reset();
        entry->live_tokens.clear();
      }
    }
    try {
      // A cold lease resets the slot, which returns its rows; dropping the
      // lease uncommitted leaves it empty.
      (void)entry->slot->Acquire();
    } catch (...) {
      const std::lock_guard lock(impl_->mutex);
      entry->busy = false;
      throw;
    }
    {
      const std::lock_guard lock(impl_->mutex);
      entry->busy = false;
      entry->holds_rows = false;
    }
    Logger::Info("cache",
                 "event=live_frontier_dropped reason=row_capacity "
                 "slot=" +
                     std::to_string(slot) +
                     " tokens=" + std::to_string(live_tokens));
  }
  return RowReclaim::kReady;
}

ComponentTextCache::HeldRows ComponentTextCache::ContinuedTokens(
    std::span<const ContinuationToken> prompt, const TextPromptContext* context,
    bool reuse, std::size_t stable) const {
  const auto input = Inputs(prompt.size(), context);
  const std::lock_guard lock(impl_->mutex);
  const auto found =
      impl_->Lookup({impl_->identity, prompt, input, stable, reuse});
  // A rewind keeps the slot's blocks up to its point.
  if (reuse)
    if (const auto choice = impl_->FindRewind(found, prompt, input, context))
      return {.tokens = choice->boundary, .live = true};
  if (!found.selected || found.selected->boundary >= prompt.size())
    return {};
  if (found.selected->live) {
    const auto& entry =
        *impl_->entries.at(found.selected->live->slot.value - 1);
    if (entry.busy || !entry.holds_rows)
      return {};
    return {.tokens = found.selected->boundary, .live = true};
  }
  // A disk restore copies every row into new blocks.
  return {};
}

std::unique_ptr<ComponentTextCache::Lease> ComponentTextCache::Acquire(
    std::span<const ContinuationToken> prompt,
    const ContinuationCache::CancellationCheck& cancelled,
    std::shared_ptr<const TextPromptContext> context, bool reuse,
    std::size_t stable, bool stop_at_eos, bool persist) {
  if (cancelled && cancelled())
    return {};
  auto input = Inputs(prompt.size(), context.get());
  cache::PrefixLookup found;
  Impl::Entry* entry = nullptr;
  std::size_t slot_index = 0;
  std::optional<cache::BorrowedLocation> live;
  std::optional<Impl::Uncovered> evicted;
  std::vector<std::size_t> reply_starts, reply_ends;
  std::size_t common = 0;
  // The slot's rewind point this request is served from, and whether the
  // request keeps one of its own.
  std::shared_ptr<RewindPoint> rewind;
  bool rewind_wanted = false;
  // Prefill continues a checkpoint exactly only from an aligned position.
  // The complete prompt stays off the alignment for an unchanged retry,
  // which continues it without prefill; a longer prompt takes an aligned
  // boundary instead (TextRunnerCapabilities::prefill_alignment).
  const std::size_t alignment = std::max<std::size_t>(
      1, impl_->runner->Descriptor().capabilities.prefill_alignment);
  {
    const std::lock_guard lock(impl_->mutex);
    if (impl_->catalog)
      impl_->catalog->Reconcile();
    found = impl_->Lookup({impl_->identity, prompt, input, stable, reuse});
    // A miss reports how far a stored prompt agreed, so an empty cache can
    // be told apart from one holding only incompatible boundaries.
    if (reuse && !found.selected) {
      try {
        common =
            impl_->index.CommonPrefixTokens({impl_->identity, prompt, input});
      } catch (const std::exception&) {
        // Only reported (lookup_result): unknown agreement reads as none.
      }
    }
    if (reuse) {
      if (const auto choice =
              impl_->FindRewind(found, prompt, input, context.get())) {
        entry = impl_->entries[choice->index].get();
        live = entry->live_location;
        rewind = entry->rewind;
      }
      rewind_wanted = rewind || impl_->RewritesReply(prompt, input);
    }
    if (!entry && found.selected && found.selected->live) {
      const auto location = *found.selected->live;
      auto& selected_entry = *impl_->entries.at(location.slot.value - 1);
      if (!selected_entry.busy) {
        entry = &selected_entry;
        live = location;
        rewind_wanted = rewind_wanted || entry->drops_reply;
      }
    }
    if (!entry)
      entry = impl_->SelectCold();
    if (!entry)
      throw cache::SlotBusy();
    while (impl_->entries[slot_index].get() != entry)
      ++slot_index;
    if (!live)
      evicted = impl_->UncoveredLive(slot_index);
    entry->busy = true;
    entry->holds_rows = true;
    entry->used = ++impl_->clock;
  }
  // The live frontier this cold assignment replaces would otherwise be lost;
  // the next turn of that conversation then restores an older checkpoint.
  if (evicted)
    CaptureBeforeReset(evicted->index, evicted->location,
                       std::move(evicted->tokens), std::move(evicted->input),
                       "reassignment", false);
  // Its rewind point goes too: written to disk unless covered.
  if (!live)
    impl_->DropRewind(*entry, "reassignment");
  // Disk jobs still queued may read rows this slot lent them (a prompt-end
  // capture): the cold reset would retire those rows and fail the jobs.
  // Give them host copies first; without room the jobs fail, uncached.
  if (!live)
    HostObservedRows(slot_index);
  {
    const std::lock_guard lock(impl_->mutex);
    if (!live) {
      entry->branch = std::make_shared<const std::uint8_t>(0);
      entry->durable.reset();
      entry->written_back = 0;
      entry->write_back_tried = 0;
      entry->frontier_dead = false;
      entry->drops_reply = false;
    }
    if (entry->live) {
      impl_->index.Erase(*entry->live);
      entry->live.reset();
      entry->live_tokens.clear();
    }
    if (live) {
      reply_starts = std::move(entry->reply_starts);
      reply_ends = std::move(entry->reply_ends);
    }
    entry->reply_starts.clear();
    entry->reply_ends.clear();
  }
  std::unique_ptr<Lease> lease;
  try {
    auto slot = entry->slot->Acquire(live);
    lease = std::make_unique<Lease>(std::make_unique<Lease::Impl>(
        impl_.get(), entry, std::move(slot), prompt, input, stable));
    auto& request = *lease->impl_;
    request.persist = persist;
    request.rewind_wanted = rewind_wanted;
    request.state->SetCancellationCheck(cancelled);
    request.state->SetStopAtEos(stop_at_eos);
    impl_->runner->SetPromptContext(*request.state, context);
    const auto started = Clock::now();
    // Cold Acquire has already reset the slot. Only an attempted restore (or
    // an unusable live frontier) can require another reset before cold use.
    bool clean = !live;
    // A frontier captured between prefill passes cannot select the next token
    // of a prompt that ends there; serve that prompt from a shallower boundary.
    // A request whose last prompt token must run (a detecting dynamic
    // abliteration request) reuses at most all but that token.
    const bool whole_prompt = impl_->runner->ReusesWholePrompt(context.get());
    const auto decodable = [&](std::size_t boundary) {
      return boundary < prompt.size() ||
             (whole_prompt && impl_->runner->HasDecodeFrontier(*request.state));
    };
    if (live && !rewind && !decodable(found.selected->boundary)) {
      // The restore or reset below overwrites the live rows: disk jobs still
      // reading rows it lent them get host copies first.
      try {
        Lease::Impl::SnapshotStats stats;
        (void)request.HostCopy({}, nullptr, stats);
      } catch (...) {
      }
      live.reset();
      impl_->DropRewind(*entry, "restore");
      const std::lock_guard lock(impl_->mutex);
      entry->branch = std::make_shared<const std::uint8_t>(0);
      entry->durable.reset();
      entry->written_back = 0;
      entry->write_back_tried = 0;
      entry->frontier_dead = false;
    }
    if (rewind) {
      // Rows of the frontier past the point are overwritten: disk jobs
      // still reading rows it lent them get host copies first.
      try {
        Lease::Impl::SnapshotStats stats;
        (void)request.HostCopy({}, nullptr, stats);
      } catch (...) {
      }
      const auto n = rewind->tokens.size();
      try {
        request.Rewind(*rewind);
        Logger::Info(
            "cache",
            "schema=component-cache-v1 event=rewind_restored slot=" +
                std::to_string(slot_index + 1) +
                " tokens=" + std::to_string(n) +
                " host_bytes=" + std::to_string(rewind->host_bytes) +
                " device_bytes=" + std::to_string(rewind->device_bytes) +
                " ms=" +
                std::to_string(std::chrono::duration<double, std::milli>(
                                   Clock::now() - started)
                                   .count()));
        // A new execution from the point: the abandoned frontier's queued
        // writes must not become its lineage. Its durable ancestor stays
        // only when it lies within the point.
        entry->history = std::make_unique<cache::ExecutionHistory>(
            cache::ExecutionHistory::RestoredUnshared(
                impl_->ledger, impl_->resources->adapter->Components(),
                impl_->identity));
        request.rewound = true;
        request.restored = rewind->host_bytes;
        const std::lock_guard lock(impl_->mutex);
        entry->branch = std::make_shared<const std::uint8_t>(0);
        if (entry->durable &&
            (entry->durable->Manifest().tokens.size() > n ||
             !Impl::Prefixes(entry->durable->Manifest().tokens, prompt)))
          entry->durable.reset();
        entry->written_back = 0;
        entry->write_back_tried = 0;
        entry->frontier_dead = false;
        // Reply edges past the point belong to the abandoned frontier.
        std::erase_if(reply_starts, [&](std::size_t at) { return at >= n; });
        std::erase_if(reply_ends, [&](std::size_t at) { return at > n; });
      } catch (...) {
        try {
          Logger::Info("cache",
                       std::string("schema=component-cache-v1 "
                                   "event=restore_failed source=rewind") +
                           " tokens=" + std::to_string(n) + " reason=\"" +
                           CurrentExceptionReason() + "\"");
        } catch (...) {
        }
        request.rewound = false;
        request.restored = 0;
        if (!impl_->resources->adapter->Invalidate(request.slot.Execution()))
          throw;
        clean = true;
        live.reset();
        rewind.reset();
        impl_->DropRewind(*entry, "restore_failed");
        const std::lock_guard lock(impl_->mutex);
        entry->branch = std::make_shared<const std::uint8_t>(0);
        entry->durable.reset();
        entry->written_back = 0;
        entry->write_back_tried = 0;
        entry->frontier_dead = false;
      }
    }
    bool restored = bool(live);
    if (live) {
      request.cached =
          rewind ? rewind->tokens.size() : found.selected->boundary;
      request.reply_starts = std::move(reply_starts);
      request.reply_ends = std::move(reply_ends);
      // The disk catalog evicts least recently used first. A conversation
      // served from its live frontier never restores its durable lineage,
      // so mark that lineage used too.
      const std::lock_guard lock(impl_->mutex);
      if (impl_->catalog && entry->durable)
        impl_->catalog->Touch(entry->durable->Manifest().checkpoint);
    }
    for (const auto& candidate : found.candidates) {
      if (restored || (cancelled && cancelled()))
        break;
      // Never load a whole-prompt candidate the request cannot use.
      if (candidate.live ||
          (!whole_prompt && candidate.boundary >= prompt.size()) ||
          (candidate.boundary % alignment != 0 &&
           candidate.boundary < prompt.size() && !AnyPosition(candidate)))
        continue;
      try {
        if (candidate.durable && impl_->streaming) {
          clean = false;
          cache::TransferTiming timing;
          const auto id = candidate.durable->Manifest().checkpoint;
          restored = impl_->streaming->Restore(
              id, impl_->digest, *impl_->resources->adapter,
              request.slot.Execution(), timing, {}, {},
              [this](bool private_file, cache::DiskFileId file) {
                const std::lock_guard lock(impl_->mutex);
                impl_->catalog->Invalidate(private_file, file);
              });
          if (restored) {
            request.disk_hit = true;
            request.restored = timing.bytes;
            Logger::Info(
                "cache",
                "schema=component-cache-v1 event=disk_restore tokens=" +
                    std::to_string(candidate.boundary) +
                    " bytes=" + std::to_string(timing.bytes) + " pieces=" +
                    std::to_string(timing.pieces) + " stream_acquire_ms=" +
                    std::to_string(timing.stream_acquire_ns / 1e6) +
                    " stream_wait_ms=" +
                    std::to_string(timing.stream_wait_ns / 1e6) +
                    " filesystem_ms=" +
                    std::to_string(timing.filesystem_ns / 1e6) +
                    " checksum_ms=" + std::to_string(timing.checksum_ns / 1e6) +
                    " staging_wait_ms=" +
                    std::to_string(timing.staging_wait_ns / 1e6));
            entry->history = std::make_unique<cache::ExecutionHistory>(
                cache::ExecutionHistory::RestoredUnshared(
                    impl_->ledger, impl_->resources->adapter->Components(),
                    impl_->identity));
            const std::lock_guard lock(impl_->mutex);
            impl_->catalog->Touch(id);
          }
        }
        if (restored && !decodable(candidate.boundary)) {
          if (!impl_->resources->adapter->Invalidate(request.slot.Execution()))
            throw std::runtime_error("component restore reset failed");
          entry->history.reset();
          request.restored = 0;
          request.disk_hit = false;
          restored = false;
          continue;
        }
        if (restored) {
          request.cached = candidate.boundary;
          const std::lock_guard lock(impl_->mutex);
          // This execution's durable lineage, for chunk reuse and recency of
          // later live turns.
          entry->durable = candidate.durable;
        }
      } catch (...) {
        // The next candidate is tried, so a failed restore only shortens the
        // reuse; name it so that shorter reuse is never silent.
        try {
          Logger::Info("cache",
                       std::string("schema=component-cache-v1 "
                                   "event=restore_failed source=disk") +
                           " tokens=" + std::to_string(candidate.boundary) +
                           " reason=\"" + CurrentExceptionReason() + "\"");
        } catch (...) {
          // Logging must not keep the slot from being reset below.
        }
        if (!impl_->resources->adapter->Invalidate(request.slot.Execution()))
          throw;
        clean = true;
        request.restored = 0;
      }
    }
    if (!restored) {
      if (!clean &&
          !impl_->resources->adapter->Invalidate(request.slot.Execution()))
        throw std::runtime_error("component cold initialization failed");
      entry->history = std::make_unique<cache::ExecutionHistory>(
          cache::ExecutionHistory::Cold(impl_->ledger,
                                        impl_->resources->adapter->Components(),
                                        impl_->identity));
    }
    // Restore/reset may clear the attachment; install this request again.
    impl_->runner->SetPromptContext(*request.state, context);
    impl_->runner->ReconcileComponentState(*request.state, request.cached);
    request.restore_time =
        request.restored
            ? std::chrono::duration<double, std::milli>(Clock::now() - started)
                  .count()
            : 0;
    request.metrics.reuse = request.rewound    ? "rewind"
                            : live             ? "live"
                            : request.disk_hit ? "disk"
                                               : "none";
    request.metrics.selection_reason = static_cast<std::uint32_t>(found.reason);
    request.metrics.selected = request.cached;
    request.metrics.restored_bytes = request.restored;
    for (const auto& candidate : found.candidates) {
      auto& deepest = candidate.live ? request.metrics.deepest_live
                                     : request.metrics.deepest_disk;
      deepest = std::max(deepest, static_cast<std::size_t>(candidate.boundary));
    }
    std::size_t nearest = 0;
    for (const auto& candidate : found.candidates)
      nearest = std::max(nearest, static_cast<std::size_t>(candidate.boundary));
    request.lookup_result = {request.cached      ? ""
                             : !reuse            ? "disabled"
                             : common || nearest ? "no_compatible_boundary"
                                                 : "no_checkpoint",
                             request.cached ? 0 : common,
                             request.cached ? request.cached : nearest};
    Logger::Info("cache",
                 "schema=component-cache-v1 event=acquire reuse=" +
                     request.metrics.reuse +
                     " selected_tokens=" + std::to_string(request.cached) +
                     " restored_bytes=" + std::to_string(request.restored));
    return lease;
  } catch (...) {
    if (lease)
      lease->Invalidate();
    else {
      const std::lock_guard lock(impl_->mutex);
      entry->busy = false;
    }
    throw;
  }
}

std::optional<TextPendingPublication> ComponentTextCache::PendingPublication(
    std::span<const ContinuationToken> prompt, const TextPromptContext* context,
    std::uint64_t checkpoint_id) const {
  if (!impl_->queue)
    return std::nullopt;
  const auto input = Inputs(prompt.size(), context);
  const std::size_t alignment = std::max<std::size_t>(
      1, impl_->runner->Descriptor().capabilities.prefill_alignment);
  const std::lock_guard lock(impl_->mutex);
  const auto found = impl_->Lookup({impl_->identity, prompt, input, 0, true});
  std::size_t best = found.selected ? found.selected->boundary : 0;
  if (const auto choice = impl_->FindRewind(prompt, input))
    best = std::max(best, choice->boundary);
  std::optional<TextPendingPublication> selected;
  for (const auto& capture : impl_->pending_durable) {
    if (!capture)
      continue;
    const auto id = capture->manifest.checkpoint.value;
    if (checkpoint_id && id != checkpoint_id)
      continue;
    const auto& tokens = capture->manifest.tokens;
    // Acquire skips an unaligned prompt checkpoint below the prompt end.
    if (tokens.size() > prompt.size() ||
        (tokens.size() % alignment != 0 && tokens.size() < prompt.size() &&
         capture->manifest.purpose != cache::CheckpointPurpose::kGenerated) ||
        capture->manifest.input != At(input, tokens.size()) ||
        !std::equal(tokens.begin(), tokens.end(), prompt.begin()))
      continue;
    // The queued capture is already known, independently of the live slot.
    // A retry can arrive before the source's cancellation settles or while
    // it still decodes. Waiting is bounded by the scheduler in either case.
    const bool short_replay =
        best == 0 &&
        prompt.size() <= TextPendingPublication::kMinimumGainTokens;
    TextPendingPublication candidate{id, best, tokens.size(), short_replay};
    if (!checkpoint_id && !candidate.WorthWaiting())
      continue;
    if (!selected || candidate.pending_tokens > selected->pending_tokens ||
        (candidate.pending_tokens == selected->pending_tokens &&
         id < selected->checkpoint_id))
      selected = candidate;
  }
  return selected;
}

std::size_t ComponentTextCache::CachedPrefixTokens(
    std::span<const ContinuationToken> prompt, const TextPromptContext* context,
    bool common) const {
  // An optional estimate: without metadata room it is zero, not an error.
  try {
    return CachedPrefixTokensOrThrow(prompt, context, common);
  } catch (const std::exception&) {
    return 0;
  }
}
std::size_t ComponentTextCache::CachedPrefixTokensOrThrow(
    std::span<const ContinuationToken> prompt, const TextPromptContext* context,
    bool common) const {
  const auto input = Inputs(prompt.size(), context);
  const std::lock_guard lock(impl_->mutex);
  const cache::PrefixQuery query{impl_->identity, prompt, input};
  if (common) {
    // Captures still queued for the disk count too: with no RAM checkpoints,
    // a conversation that just ended may only be on its way to disk, and a
    // peer diverging from it should still learn the shared boundary. Their
    // complete input identity gates it, as for index records.
    auto longest = impl_->index.CommonPrefixTokens(query);
    for (const auto& capture : impl_->pending_durable) {
      if (!capture)
        continue;
      const auto& tokens = capture->manifest.tokens;
      const auto boundary = tokens.size();
      const auto expected =
          boundary <= prompt.size() ? input.At(boundary) : input.Complete();
      if (!std::ranges::equal(expected, capture->manifest.input))
        continue;
      const auto agreed = static_cast<std::size_t>(
          std::ranges::mismatch(tokens, prompt).in1 - tokens.begin());
      longest = std::max<cache::Rows>(longest, agreed);
    }
    return longest;
  }
  auto cached = impl_->index.CachedPrefixTokens(query);
  if (const auto choice = impl_->FindRewind(prompt, input))
    cached = std::max<cache::Rows>(cached, choice->boundary);
  return cached >= prompt.size() && !impl_->runner->ReusesWholePrompt(context)
             ? prompt.size() - 1
             : cached;
}
namespace {
// Unicode White_Space and the separators Python's str.strip also removes:
// what chat templates trim from the ends of message content.
bool TrimmedWhitespace(std::string_view text) {
  if (text.empty())
    return false;
  for (std::size_t i = 0; i < text.size();) {
    const auto lead = static_cast<unsigned char>(text[i]);
    char32_t cp = lead;
    std::size_t length = 1;
    if (lead >= 0x80) {
      // No whitespace needs four bytes.
      if ((lead & 0xE0) == 0xC0) {
        cp = lead & 0x1F;
        length = 2;
      } else if ((lead & 0xF0) == 0xE0) {
        cp = lead & 0x0F;
        length = 3;
      } else {
        return false;
      }
      if (i + length > text.size())
        return false;
      for (std::size_t k = 1; k < length; ++k) {
        const auto next = static_cast<unsigned char>(text[i + k]);
        if ((next & 0xC0) != 0x80)
          return false;
        cp = (cp << 6) | (next & 0x3F);
      }
    }
    if (!((cp >= 0x09 && cp <= 0x0D) || cp == 0x20 ||
          (cp >= 0x1C && cp <= 0x1F) || cp == 0x85 || cp == 0xA0 ||
          cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 ||
          cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000))
      return false;
    i += length;
  }
  return true;
}

struct FrontierEnd {
  // Prompt tokens the frontier's text covers.
  std::size_t prompt{};
  // Bytes of the last of them past the frontier's end, when it ends inside
  // one, as a reply cut off by the token limit can.
  std::string partial;
};

// Where a chat template may have trimmed whitespace from a reply in
// `frontier`: at the start of a reply, or at its end or before its final
// (end-of-turn) token. Without a template nothing was trimmed.
struct TrimEdges {
  std::span<const std::size_t> starts, ends;
};

// Where the text of all of `frontier` ends in `prompt`, or nothing when the
// texts differ first. Only tokens past an ID mismatch are decoded; equal IDs
// at a shared boundary are skipped as they are. Control tokens (template
// framing such as <|im_end|>) match only by identical ID: a client's content
// may spell one in ordinary tokens, and bytes never match across them. The
// template trims whitespace around message content, so frontier whitespace
// the prompt lacks matches only at the edges of a reply, where the frontier
// ends or both continue with the same token. Whitespace a client removed
// elsewhere is an edit.
std::optional<FrontierEnd> FrontierTextEnd(
    std::span<const ContinuationToken> frontier,
    std::span<const ContinuationToken> prompt, std::size_t common,
    const TextModelRunner& runner, TrimEdges edges) {
  // The bytes of an ordinary token; nothing for a control token.
  const auto piece =
      [&](ContinuationToken token) -> std::optional<std::string> {
    if (runner.IsControlToken(token))
      return std::nullopt;
    return runner.Decode(std::span<const ContinuationToken>(&token, 1));
  };
  std::size_t f = common, p = common;
  // Bytes one side has consumed past the other's last token boundary.
  std::string ahead;
  bool frontier_ahead = false;
  for (;;) {
    if (ahead.empty()) {
      if (f == frontier.size())
        return FrontierEnd{p, {}};
      if (p == prompt.size())
        return std::nullopt;
      if (frontier[f] == prompt[p]) {
        ++f;
        ++p;
        continue;
      }
      auto first_piece = piece(frontier[f++]);
      if (!first_piece || first_piece->empty())
        return std::nullopt;
      ahead = std::move(*first_piece);
      frontier_ahead = true;
    }
    const auto lagging = frontier_ahead ? prompt : frontier;
    auto& at = frontier_ahead ? p : f;
    if (at == lagging.size()) {
      if (frontier_ahead)
        return std::nullopt;
      return FrontierEnd{p, std::move(ahead)};
    }
    // A control token never matches bytes; only trimmed whitespace before
    // it (below) or an identical ID resumes the match.
    const auto next = piece(lagging[at++]);
    const auto n = next ? std::min(ahead.size(), next->size()) : 0;
    if (!next || next->empty() || ahead.compare(0, n, *next, 0, n) != 0) {
      --at;
      if (!frontier_ahead || !TrimmedWhitespace(ahead))
        return std::nullopt;
      // `ahead` is the rest of frontier token f - 1; all of it when the
      // whitespace starts at that token.
      const auto first = f - 1;
      const bool whole = ahead.size() == piece(frontier[first])->size();
      while (f < frontier.size() && frontier[f] != prompt[p]) {
        const auto skipped = piece(frontier[f]);
        if (!skipped || !TrimmedWhitespace(*skipped))
          break;
        ++f;
      }
      if (f < frontier.size() && frontier[f] != prompt[p])
        return std::nullopt;
      if (std::ranges::find(edges.ends, f) == edges.ends.end() &&
          (!whole ||
           std::ranges::find(edges.starts, first) == edges.starts.end()))
        return std::nullopt;
      ahead.clear();
      continue;
    }
    if (next->size() <= ahead.size()) {
      ahead.erase(0, n);
    } else {
      ahead = next->substr(n);
      frontier_ahead = !frontier_ahead;
    }
  }
}
}  // namespace

bool ComponentTextCache::AlignToLiveFrontier(
    std::vector<ContinuationToken>& prompt, std::size_t& stable,
    std::size_t keep, std::size_t max_tokens, bool trimmed) const {
  // Idle text-only frontiers the prompt leaves token for token, copied under
  // the lock; their text is compared outside it, so decoding long replies
  // never blocks Acquire, Commit or publication lookups.
  struct Candidate {
    const Impl::Entry* entry;
    cache::IndexEntryId live;
    std::vector<ContinuationToken> tokens;
    std::vector<std::size_t> starts, ends;
    std::size_t common;
  };
  std::vector<Candidate> candidates;
  {
    const std::lock_guard lock(impl_->mutex);
    for (const auto& entry : impl_->entries) {
      const auto& frontier = entry->live_tokens;
      // Positions of image inputs would move with the rewrite.
      if (!entry->live || entry->busy || frontier.empty() ||
          !entry->live_input.Matches(Inputs(frontier.size(), nullptr),
                                     frontier.size()))
        continue;
      const auto common = static_cast<std::size_t>(
          std::ranges::mismatch(frontier, prompt).in1 - frontier.begin());
      // The prompt already continues this frontier token for token.
      if (common == frontier.size())
        return false;
      // `keep` leading tokens (a shared system prefix) must stay as they are.
      if (common == 0 || common < keep)
        continue;
      candidates.push_back(
          {entry.get(), *entry->live, frontier,
           trimmed ? entry->reply_starts : std::vector<std::size_t>{},
           trimmed ? entry->reply_ends : std::vector<std::size_t>{}, common});
    }
  }
  // Longest frontier first: the first that matches is used.
  std::ranges::stable_sort(candidates, std::ranges::greater{},
                           [](const Candidate& c) { return c.tokens.size(); });
  for (const auto& candidate : candidates) {
    const auto& frontier = candidate.tokens;
    auto end =
        FrontierTextEnd(frontier, prompt, candidate.common, *impl_->runner,
                        TrimEdges{candidate.starts, candidate.ends});
    if (!end || (stable != 0 && stable < end->prompt))
      continue;
    std::vector<ContinuationToken> rest;
    if (!end->partial.empty()) {
      // A fragment that does not tokenize back to its own bytes (a split
      // UTF-8 sequence, say) cannot follow the frontier, nor one that
      // tokenizes to a control token.
      try {
        rest = impl_->runner->Tokenize(end->partial);
      } catch (const std::exception&) {
        continue;
      }
      if (rest.empty() || impl_->runner->Decode(rest) != end->partial ||
          std::ranges::any_of(rest, [&](ContinuationToken token) {
            return impl_->runner->IsControlToken(token);
          }))
        continue;
    }
    if (frontier.size() + rest.size() + (prompt.size() - end->prompt) >=
        max_tokens)
      continue;
    {
      // The frontier must still be the same idle one.
      const std::lock_guard lock(impl_->mutex);
      const auto* entry = candidate.entry;
      if (!entry->live || *entry->live != candidate.live || entry->busy ||
          entry->live_tokens.size() != frontier.size())
        return false;
    }
    const auto seam = end->prompt;
    if (Logger::Enabled(LogLevel::kDebug))
      Logger::Debug("cache",
                    "event=frontier_aligned frontier_tokens=" +
                        std::to_string(frontier.size()) +
                        " prompt_seam=" + std::to_string(seam) +
                        " split_tokens=" + std::to_string(rest.size()) +
                        " prompt_tokens=" + std::to_string(prompt.size()));
    if (stable != 0)
      stable = stable - seam + frontier.size() + rest.size();
    std::vector<ContinuationToken> aligned;
    aligned.reserve(frontier.size() + rest.size() + prompt.size() - seam);
    aligned.assign(frontier.begin(), frontier.end());
    aligned.insert(aligned.end(), rest.begin(), rest.end());
    aligned.insert(aligned.end(), prompt.begin() + seam, prompt.end());
    prompt = std::move(aligned);
    return true;
  }
  return false;
}
ComponentTextCache::Lease::Lease(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ComponentTextCache::Lease::~Lease() {
  Invalidate();
}
ContinuationState& ComponentTextCache::Lease::state() const {
  return *impl_->state;
}
bool ComponentTextCache::Lease::cache_hit() const noexcept {
  return impl_->cached != 0;
}
std::size_t ComponentTextCache::Lease::cached_tokens() const noexcept {
  return impl_->cached;
}
std::size_t ComponentTextCache::Lease::restored_snapshot_bytes()
    const noexcept {
  return impl_->restored;
}
double ComponentTextCache::Lease::restore_ms() const noexcept {
  return impl_->restore_time;
}
bool ComponentTextCache::Lease::restored_from_disk() const noexcept {
  return impl_->disk_hit;
}
ContinuationLookup ComponentTextCache::Lease::lookup() const noexcept {
  return impl_->lookup_result;
}
bool ComponentTextCache::Lease::HasSnapshotFor(
    std::span<const ContinuationToken> tokens) const {
  // The rewind point this request was served from is that state, immutable.
  if (impl_->rewound && tokens.size() == impl_->cached)
    return true;
  const std::lock_guard lock(impl_->cache->mutex);
  return impl_->cache->index.HasCheckpoint(
      {impl_->cache->identity, tokens,
       cache::InputIdentity(tokens.size(), At(impl_->input, tokens.size()))},
      tokens.size());
}
bool ComponentTextCache::Lease::TryReserveSnapshot(
    std::size_t bytes, std::size_t boundary, bool preserve,
    SnapshotPurpose purpose, std::span<const ContinuationToken> tokens) {
  (void)bytes;
  (void)preserve;
  // Every capture goes to disk, or becomes a rewind point; grid (history)
  // points were RAM-only edits.
  if (!boundary || boundary > impl_->prompt.size() ||
      purpose == SnapshotPurpose::kHistory ||
      ((!impl_->persist || !impl_->cache->queue) &&
       (purpose == SnapshotPurpose::kBranchPoint ||
        !impl_->RewindEligible(boundary))))
    return false;
  if (tokens.empty())
    tokens = std::span<const cache::Token>(impl_->prompt).first(boundary);
  impl_->capture_tokens.assign(tokens.begin(), tokens.end());
  impl_->purpose = purpose;
  return true;
}
void ComponentTextCache::Lease::SkipSnapshot(SnapshotEventReason reason,
                                             std::size_t bytes,
                                             std::size_t boundary) noexcept {
  try {
    Logger::Info("cache",
                 "schema=component-cache-v1 event=snapshot_skipped reason=" +
                     std::to_string(static_cast<unsigned>(reason)) +
                     " tokens=" + std::to_string(boundary) +
                     " bytes=" + std::to_string(bytes));
  } catch (...) {
  }
}

bool ComponentTextCache::Lease::CaptureRefused(std::size_t boundary) const {
  auto& cache = *impl_->cache;
  if (!boundary || boundary > impl_->prompt.size())
    return false;
  // A rewind point is kept in memory whatever the disk would take.
  if (impl_->RewindEligible(boundary))
    return false;
  // Every capture goes to disk (CaptureDurable), which refuses one without a
  // disk cache, for a request that keeps its prompt off the disk, or within
  // the disk step of an earlier checkpoint.
  if (!impl_->persist || !cache.queue || !cache.resources->copy_observed_rows)
    return true;
  return cache.WithinDiskStep(
      std::span<const cache::Token>(impl_->prompt).first(boundary),
      At(impl_->input, boundary));
}
std::size_t ComponentTextCache::Lease::CaptureBytes() const {
  return impl_->cache->CaptureCopyBytes(
      *impl_->entry->history,
      impl_->cache->resources->adapter->Positions(impl_->slot.Execution()));
}
std::size_t ComponentTextCache::Lease::CaptureBytes(
    std::size_t boundary) const {
  return impl_->cache->CaptureCopyBytes(
      *impl_->entry->history,
      impl_->cache->resources->adapter->PrefillCheckpointPositions(
          impl_->slot.Execution(), boundary));
}
std::unique_ptr<TextRunnerSnapshot> ComponentTextCache::Lease::Capture() {
  auto& request = *impl_;
  try {
    return request.CaptureDurable();
  } catch (const cache::ResourceExhausted&) {
    Logger::Info("cache",
                 "schema=component-cache-v1 event=capture_refused "
                 "reason=byte_capacity tokens=" +
                     std::to_string(request.capture_tokens.size()));
    throw;
  } catch (const std::exception& error) {
    Logger::Info(
        "cache",
        std::string("schema=component-cache-v1 event=capture_failed reason=") +
            error.what());
    throw;
  }
}
std::size_t ComponentTextCache::Lease::PublishSnapshot(
    std::vector<ContinuationToken>,
    std::shared_ptr<const ContinuationSnapshot> snapshot, bool) {
  const auto* component =
      dynamic_cast<const ComponentSnapshot*>(snapshot.get());
  if (!component || !component->durable)
    return 0;
  // Queuing is optional: a refusal or failure leaves the prompt uncached.
  try {
    const auto started = Clock::now();
    const auto queued = impl_->cache->Persist(component->durable);
    impl_->metrics.disk_queued_bytes += queued;
    impl_->metrics.disk_enqueue_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - started)
            .count();
    (void)queued;
    // No RAM copy is retained: the snapshot's bytes are on their way to disk.
    return 0;
  } catch (const std::exception& error) {
    Logger::Info("cache", std::string("schema=component-cache-v1 "
                                      "event=persistence_refused reason=") +
                              error.what());
    return 0;
  }
}
std::size_t ComponentTextCache::Lease::Commit(
    std::vector<ContinuationToken> tokens,
    std::shared_ptr<const ContinuationSnapshot> snapshot,
    std::vector<ContinuationToken> live) {
  const auto bytes =
      snapshot ? PublishSnapshot(std::move(tokens), std::move(snapshot), true)
               : 0;
  if (live.empty()) {
    Invalidate();
    return bytes;
  }
  auto positions =
      impl_->cache->resources->adapter->Positions(impl_->slot.Execution());
  const auto location = impl_->slot.Location();
  impl_->slot.Commit();
  std::shared_ptr<RewindPoint> stale;
  {
    const std::lock_guard lock(impl_->cache->mutex);
    auto& starts = impl_->entry->reply_starts;
    auto& ends = impl_->entry->reply_ends;
    starts = std::move(impl_->reply_starts);
    ends = std::move(impl_->reply_ends);
    if (live.size() > impl_->prompt.size()) {
      starts.push_back(impl_->prompt.size());
      // Before the reply's final token only when that token is framing (an
      // end of turn): a reply cut off inside its text ends at its last token.
      if (impl_->cache->runner->IsControlToken(live.back()))
        ends.push_back(live.size() - 1);
      ends.push_back(live.size());
    }
    impl_->entry->live_tokens = live;
    impl_->entry->live_persist = impl_->persist;
    impl_->entry->live_stable = impl_->stable;
    impl_->entry->drops_reply = impl_->rewind_wanted;
    // A completed turn establishes a new frontier, even on a live continue.
    impl_->entry->frontier_dead = false;
    // An earlier point of this execution stays valid: the slot only appended
    // after it.
    if (impl_->rewind_point)
      impl_->entry->rewind = std::move(impl_->rewind_point);
    // A point the new frontier does not extend serves no prompt: its rows
    // return now (released below, outside the lock).
    if (impl_->entry->rewind &&
        !ComponentTextCache::Impl::Prefixes(impl_->entry->rewind->tokens, live))
      stale = std::exchange(impl_->entry->rewind, {});
    impl_->entry->live_input = impl_->input.Through(live.size());
    impl_->entry->live_location = location;
    impl_->entry->idle_since = Clock::now();
    // The slot keeps its state either way; without an index entry the next
    // turn cannot continue it and restores from disk instead.
    try {
      impl_->entry->live = impl_->cache->index.Insert(cache::LiveFrontier{
          location, std::move(live), impl_->cache->identity,
          At(impl_->input, impl_->prompt.size()), std::move(positions)});
    } catch (const std::exception& error) {
      impl_->entry->live.reset();
      impl_->entry->live_tokens.clear();
      try {
        Logger::Info("cache",
                     std::string("schema=component-cache-v1 "
                                 "event=live_frontier_unindexed reason=") +
                         error.what());
      } catch (...) {
      }
    }
    impl_->entry->busy = false;
  }
  impl_->active = false;
  return bytes;
}
void ComponentTextCache::Lease::Invalidate() noexcept {
  if (!impl_ || !impl_->active)
    return;
  try {
    impl_->state->SetCancellationCheck({});
    // Disk jobs this lease queued may still read rows it lent them: the
    // reset below would retire those rows and fail the jobs. Host copies
    // keep them; without room the jobs fail and the prompt stays uncached.
    try {
      Impl::SnapshotStats stats;
      const auto reason = impl_->HostCopy({}, nullptr, stats);
      if (!reason.empty())
        Logger::Info("cache",
                     "event=observed_rows_hosted slot=" +
                         std::to_string(impl_->slot.Location().slot.value) +
                         " refused=" + reason);
    } catch (...) {
    }
    // The deepest rewind point goes to disk before the reset drops it.
    {
      std::shared_ptr<RewindPoint> point;
      {
        const std::lock_guard lock(impl_->cache->mutex);
        point = std::exchange(impl_->entry->rewind, {});
      }
      if (impl_->rewind_point)
        point = std::exchange(impl_->rewind_point, {});
      if (point)
        (void)impl_->cache->PersistRewind(*impl_->entry, point, true,
                                          "invalidated");
    }
    impl_->slot = {};
    impl_->entry->history.reset();
    const std::lock_guard lock(impl_->cache->mutex);
    impl_->entry->branch.reset();
    impl_->entry->durable.reset();
    impl_->entry->written_back = 0;
    impl_->entry->frontier_dead = false;
    impl_->entry->write_back_tried = 0;
    impl_->entry->busy = false;
  } catch (...) {
  }
  impl_->active = false;
}

ComponentCacheMetrics ComponentTextCache::Lease::Metrics() const {
  auto report = impl_->metrics;
  report.kv_cache_type = impl_->cache->kv_cache_type;
  report.kv_layout_version = impl_->cache->kv_layout_version;
  report.ledger = impl_->cache->ledger.Snapshot();
  if (impl_->cache->queue)
    report.persistence = impl_->cache->queue->Stats();
  for (const auto& entry : impl_->cache->entries) {
    const auto spill = entry->slot->Metrics();
    report.spill.idle_bytes += spill.idle_bytes;
    report.spill.foreground_bytes += spill.foreground_bytes;
    report.spill.idle_copy_ns += spill.idle_copy_ns;
    report.spill.foreground_copy_ns += spill.foreground_copy_ns;
    report.spill.stream_wait_ns += spill.stream_wait_ns;
    report.spill.residual_wait_ns += spill.residual_wait_ns;
    report.spill.reassignments += spill.reassignments;
    report.spill.idle_failures += spill.idle_failures;
  }
  return report;
}
std::optional<ComponentCacheMetrics> ServingCacheLease::Metrics() const {
  return component_ ? std::optional(component_->Metrics()) : std::nullopt;
}

#define FORWARD_RESULT(type, name, qualifiers, args, call)         \
  type ServingCacheLease::name args qualifiers {                   \
    return component_ ? component_->name call : legacy_.name call; \
  }
FORWARD_RESULT(ContinuationState&, state, const, (), ())
FORWARD_RESULT(bool, cache_hit, const noexcept, (), ())
FORWARD_RESULT(std::size_t, cached_tokens, const noexcept, (), ())
FORWARD_RESULT(std::size_t, restored_snapshot_bytes, const noexcept, (), ())
FORWARD_RESULT(double, restore_ms, const noexcept, (), ())
FORWARD_RESULT(bool, restored_from_disk, const noexcept, (), ())
FORWARD_RESULT(ContinuationLookup, lookup, const noexcept, (), ())
FORWARD_RESULT(bool, HasSnapshotFor, const,
               (std::span<const ContinuationToken> tokens), (tokens))
FORWARD_RESULT(bool, TryReserveSnapshot, ,
               (std::size_t bytes, std::size_t boundary, bool preserve,
                SnapshotPurpose purpose,
                std::span<const ContinuationToken> tokens),
               (bytes, boundary, preserve, purpose, tokens))
FORWARD_RESULT(void, SkipSnapshot, noexcept,
               (SnapshotEventReason reason, std::size_t bytes,
                std::size_t boundary),
               (reason, bytes, boundary))
FORWARD_RESULT(std::size_t, PublishSnapshot, ,
               (std::vector<ContinuationToken> tokens,
                std::shared_ptr<const ContinuationSnapshot> snapshot,
                bool preserve),
               (std::move(tokens), std::move(snapshot), preserve))
FORWARD_RESULT(std::size_t, Commit, ,
               (std::vector<ContinuationToken> tokens,
                std::shared_ptr<const ContinuationSnapshot> snapshot,
                std::vector<ContinuationToken> live),
               (std::move(tokens), std::move(snapshot), std::move(live)))
FORWARD_RESULT(void, Invalidate, noexcept, (), ())
#undef FORWARD_RESULT
void ServingCacheLease::AdoptRestoredPrefix(std::size_t tokens,
                                            std::size_t bytes, double ms) {
  if (component_)
    throw std::logic_error("component restore is owned by the component cache");
  legacy_.AdoptRestoredPrefix(tokens, bytes, ms);
}

std::size_t ServingCacheLease::CaptureBytes() const {
  return component_->CaptureBytes();
}
std::size_t ServingCacheLease::CaptureBytes(std::size_t boundary) const {
  return component_->CaptureBytes(boundary);
}
bool ServingCacheLease::CaptureRefused(std::size_t boundary) const {
  return component_ && component_->CaptureRefused(boundary);
}
std::unique_ptr<TextRunnerSnapshot> ServingCacheLease::Capture() {
  return component_->Capture();
}
}  // namespace gufo::server
