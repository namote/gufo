#include "src/models/qwen38_flash_next/continuation_adapter.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/core/hip/snapshot_copy.hpp"
#include "src/models/qwen38_flash_next/continuation_hooks.hpp"
#include "src/models/qwen38_flash_next/runtime/executor.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {
using cache::Rows;
using cache::TransferResult;
void Check(hipError_t result) {
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}
bool Draft(ContinuationPart part) {
  return part == ContinuationPart::kDraftKey ||
         part == ContinuationPart::kDraftValue ||
         part == ContinuationPart::kDraftPooledKeys ||
         part == ContinuationPart::kDraftRawKeys ||
         part == ContinuationPart::kDraftResidual;
}
bool Pool(ContinuationPart part) {
  return part == ContinuationPart::kPooledKeys ||
         part == ContinuationPart::kDraftPooledKeys;
}
struct Metadata {
  std::uint32_t version{2};
  std::uint32_t target{}, draft{}, blocks{}, draft_blocks{}, hidden_base{};
  std::uint32_t residual_valid{}, image_bytes{};
  NgramHistory ngram;
  MtpLengthState policy;
  speculative::PromptLookup<>::State lookup;
  /// Raw image identity OFF; salted with the profile identity ON.
  std::array<std::uint8_t, 32> image{};
  /// AbliterationContinuationRecord::mode. Appended in the four bytes that
  /// every earlier capture zero-filled, so OFF (0) entries keep their bytes.
  std::uint32_t abliteration{};
};
static_assert(offsetof(Metadata, abliteration) == 156 &&
              offsetof(Metadata, abliteration) ==
                  offsetof(Metadata, image) + sizeof(Metadata::image));
static_assert(sizeof(Metadata) <= kContinuationMetadataBytes);
using Coverage = std::vector<std::pair<std::size_t, std::size_t>>;
bool Covered(Coverage ranges, std::size_t end) {
  std::ranges::sort(ranges);
  std::size_t frontier = 0;
  for (const auto& [first, last] : ranges) {
    if (first > frontier)
      return false;
    frontier = std::max(frontier, last);
  }
  return frontier == end;
}
}  // namespace

class ContinuationAdapter::State final : public cache::Slot,
                                         public ContinuationHooks {
public:
  State(ContinuationAdapter& adapter, cache::MutationGuard& guard,
        std::unique_ptr<Session> session)
      : adapter(adapter), guard(guard), session(std::move(session)) {
    this->session->continuation_hooks_ = this;
    this->session->session_->continuation_hooks_ = this;
  }
  struct Job {
    State* owner;
    bool load;
    std::optional<cache::Completion> completion;
    std::optional<TransferResult> result;
    mutable std::mutex mutex;
    // Never blocks: a job locked by its submit counts as unsettled.
    bool Settled() const noexcept {
      const std::unique_lock lock(mutex, std::try_to_lock);
      return lock.owns_lock() && result.has_value();
    }
    bool Ready() const noexcept {
      const std::lock_guard lock(mutex);
      return result || completion->Ready();
    }
    TransferResult Wait() noexcept {
      const std::lock_guard lock(mutex);
      if (!result) {
        result = completion->Wait();
        completion.reset();
        if (load && *result == TransferResult::kFailed)
          owner->failed = true;
        --owner->pending;
      }
      return *result;
    }
  };
  class Signal final : public cache::CompletionSignal {
  public:
    explicit Signal(std::shared_ptr<Job> job) : job_(std::move(job)) {}
    bool Ready() const noexcept override { return job_->Ready(); }
    TransferResult Wait() noexcept override { return job_->Wait(); }

  private:
    std::shared_ptr<Job> job_;
  };
  ~State() override {
    for (const auto& job : jobs)
      (void)job->Wait();
    BeforeRelease();
    session->continuation_hooks_ = nullptr;
    session->session_->continuation_hooks_ = nullptr;
    if (written != nullptr)
      (void)hipEventDestroy(written);
  }
  bool IsValid() const noexcept override {
    return !restoring && !failed && (guarding || session->IsValid());
  }
  void BeforeExecution() override {
    if (!IsValid() || pending || checkpoint)
      throw std::logic_error("Flash-Next slot is not ready for execution");
  }
  void AfterExecution() noexcept override {
    // Mark the end of this session's device work on the shared executor
    // stream, so RetainRows waits for it and not for a peer's later passes.
    if (!adapter.kv_pool_)
      return;
    const std::lock_guard lock(written_mutex);
    if (written == nullptr &&
        hipEventCreateWithFlags(&written, hipEventDisableTiming) != hipSuccess)
      written = nullptr;
    if (written != nullptr &&
        hipEventRecord(written, adapter.model_->executor_->stream()) ==
            hipSuccess)
      unrecorded = false;
    else
      unrecorded = true;
  }
  // Blocks until every write this session submitted to the executor stream
  // has finished. Without a recorded event covering all of them (a write
  // outside BeforeExecution/AfterExecution, a restore, a reset or a failed
  // record), it drains the whole stream as before.
  void WaitForWrites() const {
    const auto stream = adapter.model_->executor_->stream();
    hipEvent_t event = nullptr;
    {
      const std::lock_guard lock(written_mutex);
      if (!unrecorded)
        event = written;
    }
    if (event != nullptr) {
      Check(hipEventSynchronize(event));
      return;
    }
    if (hipStreamQuery(stream) != hipSuccess)
      Check(hipStreamSynchronize(stream));
  }
  void MarkUnrecorded() noexcept {
    const std::lock_guard lock(written_mutex);
    unrecorded = true;
  }
  void BeforeWrite(bool draft, std::uint32_t first, std::uint32_t end,
                   std::uint32_t first_block,
                   std::uint32_t end_block) override {
    // Model execution temporarily disables its public validity flag. The
    // adapter's restore/failure status and outstanding transfers still gate
    // every native write, including graph replay and speculative rollback.
    if (restoring || failed || pending || checkpoint)
      throw std::logic_error("Flash-Next slot is not ready for mutation");
    MarkUnrecorded();
    CaptureDuringGuard readable(*this);
    for (const auto& component : adapter.layout_.Components()) {
      if (component.descriptor.kind != cache::ComponentKind::kAppendRows ||
          Draft(component.part) != draft)
        continue;
      const bool records = component.part == ContinuationPart::kKvarnRecords;
      const auto a = records                ? hip::KvarnReadGroups(first)
                     : Pool(component.part) ? first_block
                                            : first;
      const auto b =
          records ? hip::KvarnReadGroups(end + hip::kKvarnGroupTokens - 1)
          : Pool(component.part) ? end_block
                                 : end;
      if (a < b)
        guard.BeforeOverwrite(component.descriptor.id, a, b);
    }
    auto& native = *session->session_;
    auto write_first = first;
    if (!draft && native.kvarn_)
      write_first = std::min(write_first, native.kvarn_tail_.flushed_end);
    native.UnshareKvRows(write_first, end);
    if (first_block < end_block)
      native.UnshareKvRows(
          first_block * adapter.model_->config().compress_ratio,
          (end_block + 1) * adapter.model_->config().compress_ratio);
  }
  void BeforeReset() override {
    MarkUnrecorded();
    if (invalidating)
      return;
    if (restoring || pending || checkpoint)
      throw std::logic_error("cannot reset a busy Flash-Next slot");
    GuardAll();
  }
  void AfterReset() override {
    // Native Reset queues zero fills. Settle them before exposing an empty
    // slot to captures on independent streams.
    Check(hipStreamSynchronize(adapter.model_->executor_->stream()));
  }
  void BeforeRestore() override {
    MarkUnrecorded();
    BeforeExecution();
    GuardAll();
  }
  void BeforeRelease() noexcept override {
    CaptureDuringGuard readable(*this);
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeRelease(
            component.descriptor.id, 0,
            component.part == ContinuationPart::kKvarnRecords
                ? hip::KvarnReadGroups(adapter.context_)
            : Pool(component.part)
                ? adapter.context_ / adapter.model_->config().compress_ratio + 1
                : adapter.context_);
  }
  void GuardAll() {
    CaptureDuringGuard readable(*this);
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeOverwrite(
            component.descriptor.id, 0,
            component.part == ContinuationPart::kKvarnRecords
                ? hip::KvarnReadGroups(adapter.context_)
            : Pool(component.part)
                ? adapter.context_ / adapter.model_->config().compress_ratio + 1
                : adapter.context_);
  }
  ContinuationAdapter& adapter;
  cache::MutationGuard& guard;
  std::unique_ptr<Session> session;
  const rocm::PrefillCheckpoint* checkpoint{};
  rocm::Session& CaptureState() const {
    return checkpoint ? *checkpoint->state : *session->session_;
  }
  struct CaptureDuringGuard {
    explicit CaptureDuringGuard(State& state)
        : state(state), previous(std::exchange(state.guarding, true)) {}
    ~CaptureDuringGuard() { state.guarding = previous; }
    State& state;
    bool previous;
  };
  // Event recorded after this session's last execution, and whether a write
  // may have been submitted after it (guarded by written_mutex).
  mutable std::mutex written_mutex;
  hipEvent_t written{nullptr};
  bool unrecorded{true};
  bool restoring{false}, guarding{false}, invalidating{false};
  std::atomic<bool> failed{false};
  std::atomic<std::size_t> pending{0};
  std::mutex jobs_mutex;
  std::vector<std::shared_ptr<Job>> jobs;
  std::vector<cache::ComponentPosition> expected;
  std::vector<Coverage> coverage;
  enum class Restored { kNone, kMapped, kCopied };
  std::vector<Restored> restored_blocks;
  std::array<std::byte, kContinuationMetadataBytes> metadata{};
};

struct ContinuationAdapter::Region {
  struct Piece {
    void* pointer;
    std::size_t offset, bytes;
    bool host{false};
  };
  std::vector<Piece> pieces;
};

ContinuationAdapter::ContinuationAdapter(std::shared_ptr<Model> model,
                                         core::SessionMode mode,
                                         std::uint32_t context,
                                         cache::Identity identity,
                                         std::shared_ptr<hip::KvBlockPool> pool)
    : model_(std::move(model)),
      kv_pool_(std::move(pool)),
      mode_(mode),
      context_(context),
      layout_(model_ ? model_->config() : Config{},
              mode == core::SessionMode::kSpeculative, context,
              model_ ? model_->executor_->max_speculative() : 0,
              model_ && model_->Kvarn(), model_ ? model_->KvarnBits() : 4),
      identity_(std::move(identity)) {
  if ((mode != core::SessionMode::kAutoregressive &&
       mode != core::SessionMode::kSpeculative) ||
      identity_.empty() || context > model_->MaxContext() ||
      (mode == core::SessionMode::kSpeculative && !model_->HasMtp()))
    throw std::invalid_argument("invalid Flash-Next adapter configuration");
  const auto put = [&](std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i)
      identity_.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  };
  put(4);  // Block-aligned retained chunks, including offset KVarN records.
  put(model_->Kvarn());
  put(model_->Kvarn() ? model_->KvarnBits() : 16);
  put(model_->Kvarn() ? hip::kKvarnGroupTokens : 0);
  put(static_cast<std::uint32_t>(hip::KvarnValueOrder::kTokenMajor));
  put(Session::kSnapshotPayloadVersion);
  put(context);
  put(mode == core::SessionMode::kSpeculative);
  put(model_->executor_->max_speculative());
  put(model_->PrefillCapacity());
  put(model_->DecodeConcurrency());
  for (const auto& component : layout_.Components())
    descriptors_.push_back(component.descriptor);
}
ContinuationAdapter::~ContinuationAdapter() = default;
ContinuationAdapter::State& ContinuationAdapter::As(cache::Slot& slot) const {
  auto& state = dynamic_cast<State&>(slot);
  if (&state.adapter != this)
    throw std::invalid_argument("slot belongs to another Flash-Next adapter");
  return state;
}
const ContinuationAdapter::State& ContinuationAdapter::As(
    const cache::Slot& slot) const {
  return As(const_cast<cache::Slot&>(slot));
}
Session& ContinuationAdapter::GetSession(cache::Slot& slot) {
  auto& state = As(slot);
  state.BeforeExecution();
  return *state.session;
}
std::uint32_t ContinuationAdapter::ReserveKvRows(cache::Slot& slot,
                                                 std::uint32_t rows,
                                                 std::uint32_t margin) {
  return As(slot).session->session_->ReserveKvRows(rows, margin);
}
std::uint32_t ContinuationAdapter::ReleaseKvReservation(
    cache::Slot& slot) noexcept {
  return As(slot).session->session_->ReleaseKvReservation();
}
std::size_t ContinuationAdapter::MappedKvBlocks(const cache::Slot& slot) const {
  return As(slot).session->session_->MappedKvBlocks();
}
void ContinuationAdapter::PrefillThrough(cache::Slot& slot,
                                         std::span<const std::int32_t> prompt,
                                         std::uint32_t boundary,
                                         const std::function<void()>& capture) {
  auto& state = As(slot);
  state.BeforeExecution();
  if (!capture)
    throw std::invalid_argument("missing component checkpoint consumer");
  std::string error;
  const auto consume = [&](const rocm::PrefillCheckpoint& checkpoint) {
    struct BoundaryScope {
      State& state;
      bool guarding;
      BoundaryScope(State& state, const rocm::PrefillCheckpoint& checkpoint)
          : state(state), guarding(std::exchange(state.guarding, true)) {
        state.checkpoint = &checkpoint;
      }
      ~BoundaryScope() {
        // The checkpoint arena may be reused by the next native pass. Even a
        // rejected consumer must finish reading it before leaving this scope.
        {
          const std::lock_guard lock(state.jobs_mutex);
          for (const auto& job : state.jobs)
            (void)job->Wait();
        }
        state.checkpoint = nullptr;
        state.guarding = guarding;
      }
    } scope(state, checkpoint);
    // Every write up to the checkpoint is submitted: mark it so retained
    // rows wait for this pass, not a peer's. The next write marks it stale.
    state.AfterExecution();
    capture();
    if (state.pending)
      throw std::logic_error("in-pass capture returned with pending transfers");
  };
  if (!state.session->SyncThroughComponents(prompt, boundary, consume, &error))
    throw std::runtime_error("component in-pass prefill failed: " + error);
}
cache::Capabilities ContinuationAdapter::GetCapabilities() const {
  return {.continuation = true, .persistent_encoding = true};
}
std::span<const cache::ComponentDescriptor> ContinuationAdapter::Components()
    const {
  return descriptors_;
}
cache::Identity ContinuationAdapter::CompatibilityIdentity() const {
  return identity_;
}
std::unique_ptr<cache::Slot> ContinuationAdapter::CreateSlot(
    cache::MutationGuard& guard) {
  std::string error;
  auto session = model_->CreateSession(mode_, context_, &error, kv_pool_);
  if (!session)
    throw std::runtime_error(error);
  return std::make_unique<State>(*this, guard, std::move(session));
}
std::vector<cache::ComponentPosition> ContinuationAdapter::Positions(
    const cache::Slot& slot) const {
  const auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("cannot query an invalid Flash-Next slot");
  const auto& native = state.CaptureState();
  return layout_.Positions(native.position_, native.mtp_.position,
                           native.blocks_, native.mtp_.blocks);
}
std::vector<Rows> ContinuationAdapter::PlanPrefill(const cache::Slot& slot,
                                                   Rows first, Rows end) const {
  const auto& state = As(slot);
  if (!state.IsValid() || first > end || end > context_)
    throw std::invalid_argument("invalid Flash-Next prefill plan");
  std::vector<Rows> result;
  while (first < end) {
    const auto remaining = end - first;
    const auto capacity = remaining <= model_->PrefillThroughCapacity()
                              ? model_->PrefillThroughCapacity()
                              : model_->PrefillCapacity();
    first += std::min<Rows>(remaining, capacity);
    result.push_back(first);
  }
  return result;
}
std::vector<cache::ComponentPosition>
ContinuationAdapter::PrefillCheckpointPositions(const cache::Slot& slot,
                                                Rows boundary) const {
  const auto& state = As(slot);
  if (!state.IsValid() || state.pending || state.checkpoint)
    throw std::logic_error(
        "Flash-Next slot is not ready for checkpoint planning");
  if (boundary <= state.session->Position() || boundary > context_ ||
      !state.session->PrefillCheckpointBytes(
          static_cast<std::uint32_t>(boundary)))
    throw std::invalid_argument("unsupported Flash-Next in-pass boundary");
  const auto& native = *state.session->session_;
  const auto& config = model_->config();
  const auto draft =
      mode_ == core::SessionMode::kSpeculative ? boundary - 1 : 0;
  return layout_.Positions(
      boundary, draft,
      boundary > config.indexer_top_k ? boundary / config.compress_ratio
                                      : native.blocks_,
      draft > config.indexer_top_k ? draft / config.compress_ratio
                                   : native.mtp_.blocks);
}
const ContinuationComponent& ContinuationAdapter::Component(
    cache::ComponentId id) const {
  if (!id.value || id.value > layout_.Components().size())
    throw std::invalid_argument("invalid Flash-Next component ID");
  return layout_.Components()[id.value - 1];
}
void* ContinuationAdapter::RowData(
    State& state, const ContinuationComponent& component) const {
  auto& native = state.CaptureState();
  switch (component.part) {
    case ContinuationPart::kTokens:
      return state.session->tokens_.data();
    case ContinuationPart::kKvarnRecords:
      return native.attention_[component.layer].kvarn_records;
    case ContinuationPart::kKey:
      return native.attention_[component.layer].k_cache;
    case ContinuationPart::kValue:
      return native.attention_[component.layer].v_cache;
    case ContinuationPart::kPooledKeys:
      return native.attention_[component.layer].block_k;
    case ContinuationPart::kDraftKey:
      return native.mtp_.k_cache;
    case ContinuationPart::kDraftValue:
      return native.mtp_.v_cache;
    case ContinuationPart::kDraftPooledKeys:
      return native.mtp_.block_k;
    default:
      throw std::invalid_argument("component has no append rows");
  }
}

Rows ContinuationAdapter::BlockRows(
    const ContinuationComponent& component) const {
  return component.part == ContinuationPart::kKvarnRecords
             ? hip::kKvarnGroupsPerBlock
         : Pool(component.part)
             ? hip::kKvBlockTokens / model_->config().compress_ratio
             : hip::kKvBlockTokens;
}

ContinuationAdapter::Region ContinuationAdapter::RowRegion(
    State& state, const ContinuationComponent& component, Rows first,
    Rows end) const {
  Region result;
  if (first == end)
    return result;
  auto& native = state.CaptureState();
  auto* base = static_cast<std::byte*>(RowData(state, component));
  const auto width = component.descriptor.row_bytes;
  if (component.part == ContinuationPart::kTokens) {
    result.pieces.push_back(
        {base + first * width, 0, (end - first) * width, true});
    return result;
  }
  const bool records = component.part == ContinuationPart::kKvarnRecords;
  const Rows block_rows = BlockRows(component);
  const Rows offset = records ? 1 : 0;  // Group zero is the private FP16 sink.
  for (Rows row = first; row < end;) {
    const Rows logical = row + offset;
    const Rows count = std::min(end - row, block_rows - logical % block_rows);
    const Rows block = logical / block_rows;
    const Rows physical = native.kv_pool_ ? native.kv_blocks_.at(block) : block;
    auto* pointer =
        base + (physical * block_rows + logical % block_rows) * width;
    const auto payload_offset = (row - first) * width;
    if (!result.pieces.empty() && !result.pieces.back().host &&
        static_cast<std::byte*>(result.pieces.back().pointer) +
                result.pieces.back().bytes ==
            pointer &&
        result.pieces.back().offset + result.pieces.back().bytes ==
            payload_offset) {
      result.pieces.back().bytes += count * width;
    } else {
      result.pieces.push_back({pointer, payload_offset, count * width, false});
    }
    row += count;
  }
  return result;
}

void ContinuationAdapter::BeginRestore(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  state.BeforeExecution();
  if (positions.size() != descriptors_.size())
    throw std::invalid_argument("incomplete Flash-Next restore positions");
  std::vector<cache::ComponentPosition> ordered;
  ordered.reserve(descriptors_.size());
  for (const auto& descriptor : descriptors_) {
    if (std::ranges::count(positions, descriptor.id,
                           &cache::ComponentPosition::id) != 1)
      throw std::invalid_argument("missing or duplicate Flash-Next frontier");
    ordered.push_back(*std::ranges::find(positions, descriptor.id,
                                         &cache::ComponentPosition::id));
  }
  const auto frontier = [&](ContinuationPart part) -> std::uint32_t {
    for (std::size_t i = 0; i < ordered.size(); ++i)
      if (layout_.Components()[i].part == part) {
        if (ordered[i].valid_rows > context_)
          throw std::invalid_argument("Flash-Next frontier exceeds context");
        return static_cast<std::uint32_t>(ordered[i].valid_rows);
      }
    return 0;
  };
  const auto target = frontier(ContinuationPart::kTokens);
  const auto draft = frontier(ContinuationPart::kDraftKey);
  const auto expected =
      layout_.Positions(target, draft, frontier(ContinuationPart::kPooledKeys),
                        frontier(ContinuationPart::kDraftPooledKeys));
  if (ordered != expected)
    throw std::invalid_argument("inconsistent Flash-Next component frontiers");
  // Allocate host capacity and coverage first; any refusal leaves execution
  // valid. Guard all ranges while the original state is still readable.
  state.session->tokens_.reserve(target);
  state.session->logits_.reserve(model_->VocabSize());
  std::vector<Coverage> coverage(descriptors_.size());
  std::vector<State::Restored> restored(
      state.session->session_->kv_pool_ && target
          ? hip::KvBlockGeometry::BlocksFor(std::size_t{target} +
                                            model_->config().compress_ratio)
          : 0,
      State::Restored::kNone);
  state.GuardAll();
  state.MarkUnrecorded();
  Check(hipStreamSynchronize(model_->executor_->stream()));
  state.session->session_->PrepareKvRestoreRows(target);
  state.restored_blocks = std::move(restored);
  state.session->session_->PreserveSnapshots(0, 0);
  state.expected = std::move(ordered);
  state.coverage = std::move(coverage);
  state.metadata.fill(std::byte{});
  state.restoring = true;
  state.session->valid_ = false;
  // The restore rewrites the tokens in place, not through RestoreSnapshot.
  state.session->lookup_.Clear();
  state.session->tokens_.resize(target);
  state.session->ForgetDraftVocab();
  state.session->logits_.resize(model_->VocabSize());
}

cache::Completion ContinuationAdapter::Transfer(
    State& state, bool load, cache::Stream& stream,
    const std::function<void(gufo::hip::TransferStream&)>& submit) {
  auto& hip_stream = dynamic_cast<gufo::hip::TransferStream&>(stream);
  auto job = std::make_shared<State::Job>();
  job->owner = &state;
  job->load = load;
  auto signal = std::make_unique<State::Signal>(job);
  // Hold the job's lock from before publication until it has a completion or
  // a result: a boundary scope on another thread may Wait on it as soon as it
  // is listed. Readers lock jobs_mutex, then a job's mutex; taking jobs_mutex
  // here while holding this job's mutex cannot deadlock, since no reader can
  // reach this job before it is listed, and submit takes neither lock.
  const std::lock_guard job_lock(job->mutex);
  {
    const std::lock_guard lock(state.jobs_mutex);
    // A job still being submitted elsewhere is not settled; skip it rather
    // than wait for its submit while holding jobs_mutex.
    std::erase_if(state.jobs,
                  [](const auto& previous) { return previous->Settled(); });
    state.jobs.push_back(job);
  }
  ++state.pending;
  try {
    submit(hip_stream);
    job->completion.emplace(hip_stream.Complete());
  } catch (...) {
    (void)hip_stream.Synchronize();
    job->result = TransferResult::kFailed;
    --state.pending;
    if (load)
      state.failed = true;
    throw;
  }
  return cache::Completion(std::move(signal));
}

ContinuationAdapter::Region ContinuationAdapter::PrivateRegion(
    State& state, const ContinuationComponent& component, bool loading) const {
  auto& session = *state.session;
  auto& native = state.CaptureState();
  Region result;
  const auto add = [&](void* pointer, std::size_t bytes, bool host = false,
                       std::size_t offset = 0) {
    if (bytes)
      result.pieces.push_back({pointer, offset, bytes, host});
  };
  const auto bytes = component.descriptor.state_bytes;
  switch (component.part) {
    case ContinuationPart::kLogits:
      add(state.checkpoint ? const_cast<float*>(state.checkpoint->logits.data())
                           : session.logits_.data(),
          bytes, true);
      break;
    case ContinuationPart::kMetadata:
      add(state.metadata.data(), bytes, true);
      break;
    case ContinuationPart::kConvolution:
      add(native.linear_[component.layer].conv_state, bytes);
      break;
    case ContinuationPart::kRecurrent:
      add(native.linear_[component.layer].state, bytes);
      break;
    case ContinuationPart::kPleHistory:
      add(native.ple_history_, bytes);
      break;
    case ContinuationPart::kDraftResidual:
      if (loading || native.mtp_.residual_valid)
        add(native.mtp_.h, bytes);
      break;
    case ContinuationPart::kKeptHidden:
      add(native.mtp_.target_hidden,
          loading ? bytes
                  : std::size_t{state.checkpoint ? state.checkpoint->hidden_rows
                                                 : session.KeptHiddenRows()} *
                        model_->config().HcDim() * sizeof(float));
      break;
    case ContinuationPart::kKvarnKeyTail:
    case ContinuationPart::kKvarnValueTail: {
      const auto target =
          loading ? state.expected.front().valid_rows : native.position_;
      const auto width = std::size_t{model_->config().AttentionKvDim()} * 2;
      auto& attention = native.attention_[component.layer];
      auto* base = reinterpret_cast<std::byte*>(
          component.part == ContinuationPart::kKvarnKeyTail
              ? attention.kvarn_k
              : attention.kvarn_v);
      const auto sink = std::min<Rows>(target, hip::kKvarnSinkTokens);
      add(base, sink * width);
      const auto tail = std::max<Rows>(
          hip::kKvarnSinkTokens,
          target / hip::kKvarnGroupTokens * hip::kKvarnGroupTokens);
      if (target > tail) {
        const auto physical = hip::KvarnFp16Row(tail);
        add(base + physical * width, (target - tail) * width, false,
            physical * width);
      }
      break;
    }
    case ContinuationPart::kRawKeys:
    case ContinuationPart::kDraftRawKeys: {
      const auto positions = loading ? state.expected : Positions(state);
      const bool draft = Draft(component.part);
      std::uint32_t position = 0, blocks = 0;
      for (std::size_t i = 0; i < positions.size(); ++i) {
        const auto part = layout_.Components()[i].part;
        if (part ==
            (draft ? ContinuationPart::kDraftKey : ContinuationPart::kTokens))
          position = static_cast<std::uint32_t>(positions[i].valid_rows);
        if (part == (draft ? ContinuationPart::kDraftPooledKeys
                           : ContinuationPart::kPooledKeys))
          blocks = static_cast<std::uint32_t>(positions[i].valid_rows);
      }
      const auto& c = model_->config();
      const auto begin = blocks * c.compress_ratio;
      const auto rows = position - begin;
      const auto physical = begin & (native.index_capacity_ - 1);
      const auto tail = std::min(rows, native.index_capacity_ - physical);
      const auto width = std::size_t{c.indexer_head_dim} * sizeof(float);
      auto* pointer = reinterpret_cast<std::byte*>(
          draft ? native.mtp_.index_k
                : native.attention_[component.layer].index_k);
      add(pointer + physical * width, tail * width);
      add(pointer, (rows - tail) * width, false, tail * width);
      break;
    }
    default:
      throw std::invalid_argument("component is not private state");
  }
  return result;
}

cache::Completion ContinuationAdapter::CapturePrivate(
    const cache::Slot& slot, cache::ComponentId id, std::span<std::byte> bytes,
    cache::Stream& stream) {
  if (bytes.size() != Component(id).descriptor.state_bytes)
    throw std::invalid_argument("incorrect Flash-Next private buffer size");
  return CapturePrivatePiece(slot, id, 0, bytes, stream);
}
cache::Completion ContinuationAdapter::CapturePrivatePiece(
    const cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<std::byte> bytes, cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  if (!state.IsValid() || state.session->session_->spec_tokens_)
    throw std::logic_error(
        "cannot capture an invalid or speculative Flash-Next frontier");
  const auto& component = Component(id);
  const auto total = component.descriptor.state_bytes;
  if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
      offset > total || bytes.size() > total - offset)
    throw std::invalid_argument("invalid Flash-Next private piece");
  std::array<std::byte, kContinuationMetadataBytes> host_metadata{};
  if (component.part == ContinuationPart::kMetadata) {
    auto& session = *state.session;
    auto& native = state.CaptureState();
    Metadata metadata{};
    metadata.target = native.position_;
    metadata.draft = native.mtp_.position;
    metadata.blocks = native.blocks_;
    metadata.draft_blocks = native.mtp_.blocks;
    metadata.hidden_base =
        state.checkpoint ? native.position_ - state.checkpoint->hidden_rows
                         : session.hidden_base_;
    metadata.residual_valid = native.mtp_.residual_valid;
    metadata.ngram = native.ngram_;
    metadata.policy = session.draft_length_.State();
    metadata.lookup = session.lookup_.Save();
    const auto identity = session.ImageIdentity(native.position_);
    metadata.image_bytes = static_cast<std::uint32_t>(identity.size());
    // The captured state's own mode: a prefill checkpoint view copies it.
    const auto record = RecordAbliterationContinuation(
        identity,
        native.AbliterationEnabled() ? model_->Abliteration() : nullptr);
    metadata.image = record.image;
    metadata.abliteration = record.mode;
    std::memcpy(host_metadata.data(), &metadata, sizeof(metadata));
  }
  auto region = PrivateRegion(state, component, false);
  if (component.part == ContinuationPart::kMetadata)
    region.pieces = {{host_metadata.data(), 0, host_metadata.size(), true}};
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        // Pieces overwrite their ranges. Clear only uncovered bytes, which
        // serialize as zeros: clearing the whole recycled copy on the host
        // dominated capture time.
        std::vector<std::pair<std::size_t, std::size_t>> covered;
        covered.reserve(region.pieces.size());
        for (const auto& piece : region.pieces) {
          const auto first = std::max(offset, piece.offset);
          const auto end =
              std::min(offset + bytes.size(), piece.offset + piece.bytes);
          if (first < end)
            covered.emplace_back(first - offset, end - offset);
        }
        std::ranges::sort(covered);
        std::size_t cleared = 0;
        for (const auto& [first, end] : covered) {
          if (first > cleared)
            std::fill(bytes.begin() + cleared, bytes.begin() + first,
                      std::byte{});
          cleared = std::max(cleared, end);
        }
        std::fill(bytes.begin() + std::min(cleared, bytes.size()), bytes.end(),
                  std::byte{});
        for (const auto& piece : region.pieces) {
          const auto first = std::max(offset, piece.offset);
          const auto end =
              std::min(offset + bytes.size(), piece.offset + piece.bytes);
          if (first >= end)
            continue;
          auto* source =
              static_cast<std::byte*>(piece.pointer) + first - piece.offset;
          auto* destination = bytes.data() + first - offset;
          if (piece.host)
            std::memcpy(destination, source, end - first);
          else
            Check(hip::CopySnapshotToHost(destination, source, end - first,
                                          transfer.Native()));
        }
      });
}
cache::Completion ContinuationAdapter::LoadPrivate(
    cache::Slot& slot, cache::ComponentId id, std::span<const std::byte> bytes,
    cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Flash-Next load requires an unfailed restore");
  try {
    if (bytes.size() != Component(id).descriptor.state_bytes)
      throw std::invalid_argument("incorrect Flash-Next private buffer size");
    return LoadPrivatePiece(slot, id, 0, bytes, stream);
  } catch (...) {
    state.failed = true;
    throw;
  }
}
cache::Completion ContinuationAdapter::LoadPrivatePiece(
    cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Flash-Next load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto total = component.descriptor.state_bytes;
    if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
        offset > total || bytes.size() > total - offset)
      throw std::invalid_argument("invalid Flash-Next private piece");
    const auto region = PrivateRegion(state, component, true);
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(offset, offset + bytes.size());
    }
    return Transfer(
        state, true, stream, [&](gufo::hip::TransferStream& transfer) {
          for (const auto& piece : region.pieces) {
            const auto first = std::max(offset, piece.offset);
            const auto end =
                std::min(offset + bytes.size(), piece.offset + piece.bytes);
            if (first >= end)
              continue;
            auto* destination =
                static_cast<std::byte*>(piece.pointer) + first - piece.offset;
            const auto* source = bytes.data() + first - offset;
            if (piece.host)
              std::memcpy(destination, source, end - first);
            else
              Check(hip::CopySnapshotToDevice(destination, source, end - first,
                                              transfer.Native()));
          }
        });
  } catch (...) {
    state.failed = true;
    throw;
  }
}
cache::Completion ContinuationAdapter::CopyRowsOut(const cache::Slot& slot,
                                                   cache::ComponentId id,
                                                   Rows first, Rows end,
                                                   std::span<std::byte> bytes,
                                                   cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  const auto& component = Component(id);
  if (!state.IsValid())
    throw std::logic_error("cannot read an invalid Flash-Next slot");
  const auto& native = state.CaptureState();
  const auto rows =
      component.part == ContinuationPart::kKvarnRecords
          ? hip::KvarnReadGroups(native.position_)
      : component.part == ContinuationPart::kTokens ? native.position_
      : Pool(component.part)
          ? (Draft(component.part) ? native.mtp_.blocks : native.blocks_)
          : (Draft(component.part) ? native.mtp_.position : native.position_);
  const auto row = component.descriptor.row_bytes;
  if (!row || first > end || end > rows || bytes.size() != (end - first) * row)
    throw std::invalid_argument("invalid Flash-Next row capture");
  const auto region = RowRegion(state, component, first, end);
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        for (const auto& piece : region.pieces) {
          if (piece.host)
            std::memcpy(bytes.data() + piece.offset, piece.pointer,
                        piece.bytes);
          else
            Check(hip::CopySnapshotToHost(bytes.data() + piece.offset,
                                          piece.pointer, piece.bytes,
                                          transfer.Native()));
        }
      });
}

std::span<const std::byte> ContinuationAdapter::BorrowRowsView(
    const cache::Slot& slot, cache::ComponentId id, Rows first,
    Rows end) const {
  const auto& state = As(slot);
  const auto& component = Component(id);
  // The host token vector can move when execution appends. Device append
  // allocations stay fixed for the session and are guarded before mutation.
  if (component.part == ContinuationPart::kTokens || kv_pool_ ||
      component.part == ContinuationPart::kKvarnRecords)
    return {};
  if (!state.IsValid() ||
      component.descriptor.kind != cache::ComponentKind::kAppendRows)
    throw std::logic_error("cannot borrow invalid Flash-Next rows");
  const auto positions = Positions(slot);
  const auto position =
      std::ranges::find(positions, id, &cache::ComponentPosition::id);
  const auto row = component.descriptor.row_bytes;
  if (position == positions.end() || first >= end || end > position->valid_rows)
    throw std::invalid_argument("invalid Flash-Next borrowed range");
  return {static_cast<const std::byte*>(
              RowData(const_cast<State&>(state), component)) +
              first * row,
          (end - first) * row};
}
struct ContinuationAdapter::RetainedBlock {
  RetainedBlock(std::shared_ptr<hip::KvBlockPool> pool, std::uint32_t block)
      : pool(std::move(pool)), block(block) {}
  ~RetainedBlock() { pool->Release(std::span(&block, 1)); }
  std::shared_ptr<hip::KvBlockPool> pool;
  std::uint32_t block;
};

bool ContinuationAdapter::RetainsRows(cache::ComponentId id) const {
  const auto& component = Component(id);
  return kv_pool_ && component.descriptor.row_bytes &&
         component.part != ContinuationPart::kTokens;
}

cache::Adapter::RetainedRows ContinuationAdapter::RetainRows(
    const cache::Slot& slot, cache::ComponentId id, Rows first,
    Rows end) const {
  const auto& state = As(slot);
  const auto& component = Component(id);
  if (!RetainsRows(id) || !state.IsValid() || first >= end)
    return {};
  const auto& native = state.CaptureState();
  const auto frontier =
      component.part == ContinuationPart::kKvarnRecords
          ? hip::KvarnReadGroups(native.position_)
      : Pool(component.part)
          ? (Draft(component.part) ? native.mtp_.blocks : native.blocks_)
          : (Draft(component.part) ? native.mtp_.position : native.position_);
  if (end > frontier)
    throw std::invalid_argument("retained Flash-Next rows exceed frontier");
  const auto rows = BlockRows(component);
  const auto offset = component.part == ContinuationPart::kKvarnRecords ? 1 : 0;
  if ((first + offset) / rows != (end - 1 + offset) / rows)
    return {};
  // A later capture must not create COW debt after admission. Only retain
  // blocks already complete in every shared plane, including the lagging
  // predictor and the indexer's dense-to-sparse backlog. Tails stay private.
  auto stable = std::min(native.position_,
                         native.blocks_ * model_->config().compress_ratio);
  if (native.mtp_enabled_)
    stable = std::min({stable, native.mtp_.position,
                       native.mtp_.blocks * model_->config().compress_ratio});
  const auto logical = (first + offset) / rows;
  if ((logical + 1) * hip::kKvBlockTokens > stable)
    return {};
  const auto physical = native.kv_blocks_.at(logical);
  std::shared_ptr<RetainedBlock> retained;
  {
    const std::lock_guard lock(retained_mutex_);
    if (retained_.size() != kv_pool_->blocks())
      retained_.resize(kv_pool_->blocks());
    retained = retained_[physical].lock();
  }
  if (!retained) {
    // A new retention exposes the block to independent streams. Wait for
    // this session's own writes, outside the lock: draining the shared
    // executor stream here made the capture wait for a peer's prefill.
    state.WaitForWrites();
    const std::lock_guard lock(retained_mutex_);
    retained = retained_[physical].lock();
    if (!retained) {
      kv_pool_->Retain(std::span(&physical, 1));
      try {
        retained = std::make_shared<RetainedBlock>(kv_pool_, physical);
      } catch (...) {
        kv_pool_->Release(std::span(&physical, 1));
        throw;
      }
      retained_[physical] = retained;
    }
  }
  const auto region =
      RowRegion(const_cast<State&>(state), component, first, end);
  if (region.pieces.size() != 1 || region.pieces.front().host)
    throw std::logic_error("retained Flash-Next rows are not one device span");
  const auto& piece = region.pieces.front();
  return {std::shared_ptr<const void>(std::move(retained), piece.pointer),
          piece.bytes};
}

Rows ContinuationAdapter::RetainedTokenRows(cache::ComponentId id,
                                            Rows rows) const {
  if (!rows)
    return 0;
  const auto part = Component(id).part;
  return part == ContinuationPart::kKvarnRecords
             ? (rows + 1) * hip::kKvarnGroupTokens
         : Pool(part) ? rows * model_->config().compress_ratio
                      : rows;
}

bool ContinuationAdapter::OnlyRetained(const void* pointer) const {
  if (!kv_pool_)
    return false;
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  const auto base = reinterpret_cast<std::uintptr_t>(kv_pool_->data());
  const auto& geometry = kv_pool_->geometry();
  if (address < base || address - base >= geometry.bytes())
    return false;
  auto offset = address - base;
  std::size_t plane = geometry.planes().size() - 1;
  while (plane && offset < geometry.plane_offset(plane))
    --plane;
  offset =
      (offset - geometry.plane_offset(plane)) % geometry.layer_pitch(plane);
  return kv_pool_->references(offset / geometry.block_bytes(plane)) == 1;
}

cache::Completion ContinuationAdapter::CopyRowsIn(
    cache::Slot& slot, cache::ComponentId id, Rows first, Rows end,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Flash-Next load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto row = component.descriptor.row_bytes;
    if (!row || first > end || end > state.expected[id.value - 1].valid_rows ||
        bytes.size() != (end - first) * row)
      throw std::invalid_argument("invalid Flash-Next row restore");
    Region region;
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(first * row, end * row);
      if (!RetainsRows(id)) {
        region = RowRegion(state, component, first, end);
      } else {
        auto& native = *state.session->session_;
        const auto rows = BlockRows(component);
        const Rows offset =
            component.part == ContinuationPart::kKvarnRecords ? 1 : 0;
        const auto base =
            reinterpret_cast<std::uintptr_t>(RowData(state, component));
        const auto block_bytes = rows * row;
        const auto layer_bytes = kv_pool_->blocks() * block_bytes;
        for (auto r = first; r < end;) {
          const auto logical = (r + offset) / rows;
          const auto within = (r + offset) % rows;
          const auto stop = std::min(end, r + rows - within);
          const auto source = reinterpret_cast<std::uintptr_t>(
              bytes.data() + (r - first) * row);
          std::optional<std::uint32_t> physical;
          if (source >= base && source - base < layer_bytes &&
              (source - base) % block_bytes == within * row)
            physical = (source - base) / block_bytes;
          auto& restored = state.restored_blocks.at(logical);
          if (physical && (native.kv_blocks_.at(logical) == *physical ||
                           restored == State::Restored::kNone)) {
            native.MapKvBlock(logical, *physical);
            restored = State::Restored::kMapped;
          } else {
            native.EnsureKvRestoreBlock(logical);
            native.UnshareKvRows(logical * hip::kKvBlockTokens,
                                 (logical + 1) * hip::kKvBlockTokens);
            restored = State::Restored::kCopied;
            auto part = RowRegion(state, component, r, stop);
            for (auto piece : part.pieces) {
              piece.offset += (r - first) * row;
              region.pieces.push_back(piece);
            }
          }
          r = stop;
        }
      }
    }
    return Transfer(
        state, true, stream, [&](gufo::hip::TransferStream& transfer) {
          for (const auto& piece : region.pieces) {
            if (piece.host)
              std::memcpy(piece.pointer, bytes.data() + piece.offset,
                          piece.bytes);
            else
              Check(hip::CopySnapshotToDevice(piece.pointer,
                                              bytes.data() + piece.offset,
                                              piece.bytes, transfer.Native()));
          }
        });
  } catch (...) {
    state.failed = true;
    throw;
  }
}

bool ContinuationAdapter::Validate(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (!state.restoring)
    throw std::logic_error("Flash-Next validation requires an active restore");
  const auto refuse = [&] {
    state.failed = true;
    return false;
  };
  if (state.failed || state.pending ||
      positions.size() != state.expected.size())
    return refuse();
  for (std::size_t i = 0; i < state.expected.size(); ++i) {
    const auto& expected = state.expected[i];
    if (std::ranges::count(positions, expected) != 1)
      return refuse();
    const auto& descriptor = descriptors_[i];
    const auto total = descriptor.row_bytes
                           ? expected.valid_rows * descriptor.row_bytes
                           : descriptor.state_bytes;
    if (!Covered(state.coverage[i], total))
      return refuse();
  }
  Metadata metadata{};
  std::memcpy(&metadata, state.metadata.data(), sizeof(metadata));
  auto& session = *state.session;
  auto& native = *session.session_;
  auto policy = session.draft_length_;
  if (metadata.version != 2 || metadata.hidden_base > metadata.target ||
      metadata.target - metadata.hidden_base >
          model_->executor_->max_speculative() ||
      metadata.residual_valid > 1 ||
      (metadata.residual_valid && !metadata.draft) ||
      (metadata.image_bytes != 0 && metadata.image_bytes != 32) ||
      !policy.Restore(metadata.policy) ||
      !speculative::PromptLookup<>::Valid(metadata.lookup, metadata.target))
    return refuse();
  try {
    if (layout_.Positions(metadata.target, metadata.draft, metadata.blocks,
                          metadata.draft_blocks) != state.expected)
      return refuse();
    const auto identity = session.ImageIdentity(metadata.target);
    // A mode or profile other than the session's is a cache miss.
    if (!AbliterationContinuationMatches(
            metadata.abliteration, metadata.image_bytes, metadata.image,
            identity, session.ActiveAbliteration()))
      return refuse();
    if (metadata.image_bytes || (session.image_prompt_ && identity.empty())) {
      // Preserve the attached request's future images as well as its matching
      // computed prefix, including checkpoints before the first image.
      native.RestoreVisionLayout(session.image_prompt_->rope,
                                 model_->executor_->stream());
    } else {
      native.ConfigureVision(nullptr, nullptr, model_->executor_->stream());
      session.image_prompt_.reset();
    }
    Check(hipStreamSynchronize(model_->executor_->stream()));
    native.TrimKvRows(metadata.target);
    if (kv_pool_) {
      for (std::size_t i = 0; i < native.kv_blocks_.size(); ++i)
        native.EnsureKvRestoreBlock(i);
      native.UploadKvTable();
      Check(hipStreamSynchronize(model_->executor_->stream()));
    }
  } catch (...) {
    return refuse();
  }
  if (native.kvarn_) {
    const auto base = std::max(
        hip::kKvarnSinkTokens,
        metadata.target / hip::kKvarnGroupTokens * hip::kKvarnGroupTokens);
    native.kvarn_tail_ = {
        base, base == hip::kKvarnSinkTokens
                  ? metadata.target
                  : std::max(metadata.target, base + hip::kKvarnTailRingTokens -
                                                  hip::kKvarnGroupTokens + 1)};
    native.kvarn_pass_ = {};
  }
  native.position_ = metadata.target;
  native.blocks_ = metadata.blocks;
  native.mtp_.position = metadata.draft;
  native.mtp_.blocks = metadata.draft_blocks;
  native.mtp_.residual_valid = metadata.residual_valid;
  native.ngram_ = metadata.ngram;
  native.spec_base_ = metadata.target;
  native.spec_tokens_ = 0;
  native.snapshot_lineage_.reset();
  session.hidden_base_ = metadata.hidden_base;
  session.draft_length_ = policy;
  session.lookup_.Restore(metadata.lookup);
  session.draft_token_ = 0;
  session.stats_ = {};
  session.valid_ = true;
  state.restoring = false;
  state.coverage.clear();
  state.expected.clear();
  state.restored_blocks.clear();
  return true;
}
bool ContinuationAdapter::Invalidate(cache::Slot& slot) noexcept {
  State* resetting = nullptr;
  try {
    auto& state = As(slot);
    if (state.pending || state.checkpoint)
      return false;
    resetting = &state;
    state.BeforeRelease();
    struct ResetScope {
      bool& active;
      ~ResetScope() { active = false; }
    } reset_scope{state.invalidating};
    state.invalidating = true;
    // Reassign a healthy resident slot without rebuilding its MTP graphs
    // (Session::ScratchReusable). An invalid session, adapter failure
    // (`failed`) and explicit Session::Reset release all scratch.
    state.session->ResetImpl(!state.failed && state.session->ScratchReusable());
    state.failed = false;
    state.restoring = false;
    state.expected.clear();
    state.coverage.clear();
    state.guard.AfterReset();
    return true;
  } catch (...) {
    if (resetting)
      resetting->failed = true;
    return false;
  }
}
}  // namespace gufo::models::qwen38_flash_next
