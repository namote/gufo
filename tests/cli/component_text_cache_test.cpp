#include "src/cli/serve/component_text_cache.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <numeric>
#include <semaphore>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

#include "src/cache/retention.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/snapshot_pool.hpp"
#include "src/cli/serve/text_generation_scheduler.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "tests/cache/fake_adapter.hpp"
#include "tests/support/temporary_directory.hpp"

using namespace gufo;
using namespace gufo::server;
using namespace gufo::cache::testing;
namespace {

struct State final : TextRunnerState {
  FakeAdapter& adapter;
  cache::Slot& slot;
  State(FakeAdapter& adapter, cache::Slot& slot)
      : adapter(adapter), slot(slot) {}
  void Invalidate() noexcept override { (void)adapter.Invalidate(slot); }
};
class Runner : public TextModelRunner {
public:
  mutable FakeAdapter* adapter{};
  bool defer_idle{};
  // Throws once from SetPromptContext, as an Acquire failure after restore
  // candidates are pinned.
  mutable bool fail_context{};
  // Runs once on the next transfer stream request (cleared before it runs).
  mutable std::function<void()> on_stream;
  // Runs on every transfer stream request from another thread (the disk
  // writer, idle preservation) before it is served.
  mutable std::function<void()> background_stream;
  // ComponentCacheResources::reassignment_snapshot_bytes.
  std::size_t snapshot_bytes{kReassignmentSnapshotBytes};
  std::size_t snapshot_arena_bytes{64 << 10};
  std::chrono::milliseconds snapshot_idle{kReassignmentSnapshotIdle};
  std::function<bool(std::size_t)> snapshot_room;
  // FakeAdapter::SetStableAppends.
  // Device storage leaves earlier rows in place on append, so rows a disk
  // capture lends stay readable while the request decodes on.
  bool stable_appends{true};
  // Staging allocations fail (std::bad_alloc) while set.
  mutable std::atomic<bool> fail_staging{};
  // Staging bytes currently allocated (startup staging included).
  std::shared_ptr<std::atomic<std::size_t>> staging_live =
      std::make_shared<std::atomic<std::size_t>>();
  bool in_pass{};
  bool partial{};
  std::size_t alignment{1};
  /// Every prefill step's end and in-pass checkpoint, in order.
  mutable std::vector<std::size_t> pass_ends;
  mutable std::vector<std::size_t> in_pass_points;
  cache::Rows rows_per_chunk{4};
  bool retain_rows{};
  cache::Rows retained_row_limit{};
  cache::Rows retained_row_first{};
  cache::Rows retained_draft_first{};
  cache::Rows retained_draft_end{};
  std::size_t row_block_rows{};
  RowBlockPool RowBlocks() const override {
    return {.block_rows = row_block_rows};
  }
  cache::Rows retained_tokens_per_row{1};
  mutable std::size_t through_calls{};
  bool UsesComponentCache() const override { return true; }
  TextRunnerDescriptor Descriptor() const override {
    return {.model_id = "component-fake",
            .state_abi = "component-fake-v1",
            .max_context = 4096,
            .capabilities = {.incremental_prefill = true,
                             .snapshot = true,
                             .fork = true,
                             .final_token_advance_required = false,
                             .prefix_reuse = true,
                             .in_pass_checkpoint = in_pass,
                             .partial_prefill_retention = partial,
                             .prefill_alignment = alignment}};
  }
  TextRunnerResourceClaim ResourceClaim() const override {
    return {.state_capacity_bytes = 1ULL << 30,
            .per_request_state_bytes = 256,
            .retained_snapshot_capacity_bytes = 1ULL << 22};
  }
  std::vector<TextExecutionPlan> SupportedPlans() const override {
    return {{TextExecutionPlanKind::kSerial, 1},
            {TextExecutionPlanKind::kBatched, 2}};
  }
  std::vector<TextRunnerToken> Tokenize(std::string_view) const override {
    return {1};
  }
  std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest&) const override {
    return std::vector<TextRunnerToken>{1};
  }
  std::string Decode(std::span<const TextRunnerToken>) const override {
    return "";
  }
  std::unique_ptr<TextRunnerState> CreateState() const override {
    throw std::logic_error("legacy state must never be constructed");
  }
  // Image prompts carry their cache identity in the context.
  void SetPromptContext(
      TextRunnerState&,
      std::shared_ptr<const TextPromptContext>) const override {
    if (std::exchange(fail_context, false))
      throw std::runtime_error("prompt context failed");
  }
  // The cache's ledger, for injected bookkeeping failures.
  mutable cache::ResourceLedger* ledger{};
  std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
      cache::ResourceLedger& cache_ledger) const override {
    ledger = &cache_ledger;
    auto resources = std::make_unique<ComponentCacheResources>();
    resources->adapter = std::make_unique<FakeAdapter>(rows_per_chunk);
    adapter = dynamic_cast<FakeAdapter*>(resources->adapter.get());
    adapter->SetRetainRows(retain_rows);
    adapter->SetStableAppends(stable_appends);
    adapter->SetRetainedRowUnits(retained_row_limit, retained_tokens_per_row,
                                 retained_row_first);
    adapter->SetRetainedDraftRange(retained_draft_first, retained_draft_end);
    resources->stream = [this, foreground = std::this_thread::get_id()]()
        -> std::unique_ptr<cache::Stream> {
      if (defer_idle && std::this_thread::get_id() != foreground)
        return {};
      if (on_stream)
        std::exchange(on_stream, {})();
      if (background_stream && std::this_thread::get_id() != foreground)
        background_stream();
      return std::make_unique<FakeStream>(true);
    };
    resources->transfer_batch_size = 4;
    resources->staging = [this, live = staging_live](std::size_t size) {
      if (fail_staging)
        throw std::bad_alloc();
      auto* bytes = new std::vector<std::uint8_t>(size);
      *live += size;
      auto data = std::shared_ptr<std::vector<std::uint8_t>>(
          bytes, [live, size](auto* p) {
            *live -= size;
            delete p;
          });
      return cache::StagingAllocation{data, *data};
    };
    resources->reassignment_snapshot_bytes = snapshot_bytes;
    resources->reassignment_snapshot_arena_bytes = snapshot_arena_bytes;
    resources->reassignment_snapshot_idle = snapshot_idle;
    resources->snapshot_room = snapshot_room;
    resources->copy_observed_rows = [](auto source, auto destination,
                                       cache::Stream& stream) {
      return dynamic_cast<FakeStream&>(stream).Submit([source, destination] {
        std::memcpy(destination.data(), source.data(), destination.size());
        return cache::TransferResult::kSucceeded;
      });
    };
    return resources;
  }
  std::unique_ptr<TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    return std::make_unique<State>(dynamic_cast<FakeAdapter&>(adapter), slot);
  }
  void ReconcileComponentState(TextRunnerState& state,
                               std::size_t position) const override {
    const auto& s = dynamic_cast<State&>(state);
    assert(s.adapter.Positions(s.slot).front().valid_rows == position);
  }
  TextPrefillStep Prefill(TextRunnerState& state,
                          std::span<const TextRunnerToken> prompt,
                          std::size_t offset,
                          std::size_t budget) const override {
    auto& s = dynamic_cast<State&>(state);
    const auto count = std::min(budget, prompt.size() - offset);
    s.adapter.Append(s.slot, prompt.subspan(offset, count),
                     prompt.subspan(offset, count));
    pass_ends.push_back(offset + count);
    return {count, offset + count == prompt.size()};
  }
  std::optional<std::size_t> PrefillCheckpointBytes(
      const TextRunnerState&, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget,
      std::size_t boundary) const override {
    const auto end = offset + std::min(budget, prompt.size() - offset);
    return boundary > offset && boundary < end && end - boundary <= 8
               ? std::optional<std::size_t>(1)
               : std::nullopt;
  }
  TextPrefillStep PrefillThrough(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget, std::size_t boundary,
      std::unique_ptr<TextRunnerSnapshot>* checkpoint,
      const std::function<std::unique_ptr<TextRunnerSnapshot>()>& capture)
      const override {
    ++through_calls;
    auto& s = dynamic_cast<State&>(state);
    const auto count = std::min(budget, prompt.size() - offset);
    const auto head = prompt.subspan(offset, boundary - offset);
    s.adapter.Append(s.slot, head, head);
    try {
      *checkpoint = capture();
    } catch (...) {
      checkpoint->reset();
    }
    const auto tail = prompt.subspan(boundary, offset + count - boundary);
    s.adapter.Append(s.slot, tail, tail);
    in_pass_points.push_back(boundary);
    pass_ends.push_back(offset + count);
    return {count, offset + count == prompt.size()};
  }
  TextDecodeSelection SelectNext(TextRunnerState& state,
                                 sampling::SamplerState&) const override {
    auto& s = dynamic_cast<State&>(state);
    return {.token = static_cast<TextRunnerToken>(
                s.adapter.RecurrentHash(s.slot) % 100000),
            .piece = "x"};
  }
  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& s = dynamic_cast<State&>(state);
    s.adapter.Append(s.slot, {&token, 1}, {&token, 1});
  }
  std::size_t CheckpointPosition(const TextRunnerState& state) const override {
    const auto& s = dynamic_cast<const State&>(state);
    return s.adapter.Positions(s.slot).front().valid_rows;
  }
};
// Waits until a prefix of `prompt` at least `tokens` long is cached: disk
// publications complete on the writer thread.
template<class Cache>
bool WaitCached(const Cache& cache, std::span<const TextRunnerToken> prompt,
                std::size_t tokens) {
  for (int i = 0; i < 1000; ++i) {
    if (cache.CachedPrefixTokens(prompt, nullptr) >= tokens)
      return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return false;
}
// Waits until the disk writer has no pending or active job.
template<class Cache>
bool WaitWritten(const Cache& cache) {
  for (int i = 0; i < 1000; ++i) {
    if (!cache.HostWritesPending())
      return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return false;
}
void Prefill(TextRunnerPool::Request& request) {
  while (!request.prefill_complete())
    (void)request.Prefill(8);
}
TextRunnerToken Generate(TextRunnerPool::Request& request) {
  const auto token = request.SelectNext().token;
  request.Advance();
  return token;
}
void InPassCapturePreservesBoundaryAndLiveTail() {
  std::vector<TextRunnerToken> prompt(40);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto cold_runner = std::make_shared<Runner>();
  TextRunnerPool cold_pool(cold_runner, 1);
  auto cold =
      cold_pool.Acquire(prompt, sampling::SamplingConfig{}, {}, {}, false);
  Prefill(cold);
  const auto expected = Generate(cold);
  cold.Invalidate();
  for (const bool fail : {false, true}) {
    // The in-pass checkpoint goes to disk: its private state is copied to a
    // host arena inside the pass, before the pass continues to the end.
    const TemporaryDirectory temp("gufo-component-in-pass");
    auto runner = std::make_shared<Runner>();
    runner->in_pass = true;
    TextRunnerPool pool(
        runner, 2,
        TextRunnerDiskCacheOptions{.directory = temp.path().string(),
                                   .capacity_bytes = 1 << 22,
                                   .min_checkpoint_step_tokens = 1});
    auto seed =
        pool.Acquire(prompt, sampling::SamplingConfig{}, {}, {}, true, 35);
    if (fail)
      runner->adapter->FailNextTransfer();
    const auto step = seed.Prefill(64);
    assert(step.consumed_tokens == 40 && step.decode_ready);
    assert(runner->through_calls == 1);
    assert(Generate(seed) == expected);
    if (fail) {
      seed.Invalidate();
      continue;
    }
    (void)seed.Commit();
    assert(WaitWritten(pool));
    auto branch = prompt;
    branch.back() = 999;
    auto restored =
        pool.Acquire(branch, sampling::SamplingConfig{}, {}, {}, true, 35);
    assert(restored.cached_prompt_tokens() == 35 && restored.cache_disk_hit());
    Prefill(restored);
    const auto resumed = Generate(restored);
    restored.Invalidate();
    auto independent =
        cold_pool.Acquire(branch, sampling::SamplingConfig{}, {}, {}, false);
    Prefill(independent);
    assert(Generate(independent) == resumed);
    independent.Invalidate();
  }
}
void ColdAcquireResetsOnce() {
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 1, {});
  const std::vector<TextRunnerToken> prompt{1, 2, 3};
  auto first = cache.Acquire(prompt, {}, {}, true, 0, true);
  assert(runner->adapter->Invalidations() == 1);
  auto& state = dynamic_cast<State&>(first->state());
  runner->adapter->Append(state.slot, prompt, prompt);
  (void)first->Commit(prompt, {}, prompt);
  auto second = cache.Acquire(prompt, {}, {}, false, 0, true);
  assert(!second->cache_hit());
  assert(runner->adapter->Invalidations() == 2);
}
void ColdAssignmentAvoidsDrainingCancelledSlot(bool deep = false) {
  auto runner = std::make_shared<Runner>();
  runner->defer_idle = true;
  ComponentTextCache cache(runner, 2, {});
  std::vector<TextRunnerToken> prompt(deep ? 4096 : 3);
  std::iota(prompt.begin(), prompt.end(), 1);
  auto completed = cache.Acquire(prompt, {}, {}, true, 0, true);
  auto& state = dynamic_cast<State&>(completed->state());
  auto* ready_slot = &state.slot;
  runner->adapter->Append(state.slot, prompt, prompt);
  const auto hash = runner->adapter->RecurrentHash(state.slot);
  if (deep) {
    assert(completed->TryReserveSnapshot(
        completed->CaptureBytes(), prompt.size(), false,
        SnapshotPurpose::kContinuation, prompt));
    (void)completed->PublishSnapshot(prompt, completed->Capture(), false);
    runner->adapter->GuardRows(state.slot, cache::testing::kTarget, 0,
                               prompt.size());
    runner->adapter->GuardRows(state.slot, cache::testing::kDraft, 0,
                               prompt.size());
  }
  (void)completed->Commit({}, {}, prompt);

  const std::vector<TextRunnerToken> long_prompt{11, 12, 13};
  auto cancelled = cache.Acquire(long_prompt, {}, {}, true, 0, true);
  auto* draining = &dynamic_cast<State&>(cancelled->state()).slot;
  assert(draining != ready_slot);
  runner->adapter->SetExecutionComplete(*draining, false);
  cancelled->Invalidate();

  const std::vector<TextRunnerToken> unrelated{21, 22, 23};
  auto peer = cache.Acquire(unrelated, {}, {}, true, 0, true);
  assert(&dynamic_cast<State&>(peer->state()).slot == ready_slot &&
         "an unrelated peer waited for a draining cancelled slot");
  // Readiness is a preference, not an admission condition. If the ready slot
  // is occupied, the remaining slot is still available for ordered reuse.
  auto fallback = cache.Acquire(unrelated, {}, {}, false, 0, true);
  assert(&dynamic_cast<State&>(fallback->state()).slot == draining);
  runner->adapter->SetExecutionComplete(*draining, true);
  peer->Invalidate();
  fallback->Invalidate();
  if (deep) {
    auto restored = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(restored->cached_tokens() == prompt.size());
    auto& restored_state = dynamic_cast<State&>(restored->state());
    assert(runner->adapter->RecurrentHash(restored_state.slot) == hash);
  }
}
void CancelledPrefillKeepsCompletedSteps() {
  // A prefill cancelled between steps keeps the frontier it reached as a live
  // prefix (#486, #491), including on the component path Flash-Next uses.
  for (const bool partial : {false, true}) {
    auto runner = std::make_shared<Runner>();
    runner->partial = partial;
    TextRunnerPool pool(runner, 1);
    std::vector<TextRunnerToken> prompt(40);
    std::iota(prompt.begin(), prompt.end(), 10);
    auto request = pool.Acquire(prompt, {}, [] { return false; }, {}, true, 0);
    assert(request.Prefill(16).consumed_tokens == 16);
    (void)request.Cancel();
    prompt.push_back(99);
    auto resend = pool.Acquire(prompt, {}, {}, {}, true, 0);
    assert(resend.cached_prompt_tokens() == (partial ? 16 : 0));
    resend.Invalidate();
  }
}

void PartialFrontierDoesNotServeAnEqualPrompt() {
  // A prefill cancelled between steps keeps a frontier with no next token. A
  // prompt ending exactly there must recompute its tail instead of decoding
  // from a placeholder, whether the frontier is live or restored.
  class FrontierRunner final : public Runner {
  public:
    mutable std::vector<std::size_t> ready;
    TextPrefillStep Prefill(TextRunnerState& state,
                            std::span<const TextRunnerToken> prompt,
                            std::size_t offset,
                            std::size_t budget) const override {
      const auto step = Runner::Prefill(state, prompt, offset, budget);
      if (step.decode_ready)
        ready.push_back(CheckpointPosition(state));
      return step;
    }
    bool HasDecodeFrontier(const TextRunnerState& state) const override {
      return std::ranges::find(ready, CheckpointPosition(state)) != ready.end();
    }
  };
  for (const bool replace_slot : {false, true}) {
    auto runner = std::make_shared<FrontierRunner>();
    runner->partial = true;
    TextRunnerPool pool(runner, 1);
    std::vector<TextRunnerToken> prompt(40);
    std::iota(prompt.begin(), prompt.end(), 10);
    auto request = pool.Acquire(prompt, {}, [] { return false; }, {}, true, 0);
    assert(request.Prefill(16).consumed_tokens == 16);
    (void)request.Cancel();
    if (replace_slot) {
      auto unrelated = prompt;
      unrelated.front() = 999;
      auto other = pool.Acquire(unrelated, {}, {}, {}, true, 0);
      Prefill(other);
      (void)other.Commit();
    }
    const std::vector<TextRunnerToken> equal(prompt.begin(),
                                             prompt.begin() + 16);
    auto resend = pool.Acquire(equal, {}, {}, {}, true, 0);
    assert(resend.cached_prompt_tokens() < equal.size());
    Prefill(resend);
    (void)Generate(resend);
    resend.Invalidate();
  }
}
void PartialReuseRequestsRunTheirLastPromptToken() {
  // A runner that must run every request's last prompt token (Flash-Next's
  // detecting dynamic abliteration requests): neither a live frontier nor a
  // checkpoint of the whole prompt is reused, and the estimate agrees.
  class LastTokenRunner final : public Runner {
  public:
    bool ReusesWholePrompt(const TextPromptContext*) const override {
      return false;
    }
  };
  auto runner = std::make_shared<LastTokenRunner>();
  TextRunnerPool pool(runner, 1);
  std::vector<TextRunnerToken> prompt(40);
  std::iota(prompt.begin(), prompt.end(), 10);
  {
    auto first = pool.Acquire(prompt, {}, {}, {}, true, 0);
    Prefill(first);
    (void)first.Commit();
  }
  assert(pool.CachedPrefixTokens(prompt, nullptr) < prompt.size());
  {
    auto live = pool.Acquire(prompt, {}, {}, {}, true, 0);
    assert(live.cached_prompt_tokens() < prompt.size());
    Prefill(live);
    (void)live.Commit();
  }
  // Replace the slot so only the whole-prompt checkpoint and shorter ones
  // remain.
  auto unrelated = prompt;
  unrelated.front() = 999;
  {
    auto other = pool.Acquire(unrelated, {}, {}, {}, true, 0);
    Prefill(other);
    (void)other.Commit();
  }
  auto replay = pool.Acquire(prompt, {}, {}, {}, true, 0);
  assert(replay.cached_prompt_tokens() < prompt.size());
  Prefill(replay);
  replay.Invalidate();
}

void ColdAssignmentKeepsLiveFrontiers() {
  // No checkpoints are borrowed, so every slot has completed preservation.
  // Requests that cannot continue a frontier take the empty slot, then the
  // least recently used one, so the generated reply stays live (#562).
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 2, {});
  const auto turn = [&](const std::vector<TextRunnerToken>& tokens) {
    auto lease = cache.Acquire(tokens, {}, {}, true, 0, true);
    const auto live = lease->cache_hit() && !lease->restored_snapshot_bytes();
    const auto cached = lease->cached_tokens();
    auto& state = dynamic_cast<State&>(lease->state());
    const auto tail = std::span(tokens).subspan(cached);
    runner->adapter->Append(state.slot, tail, tail);
    (void)lease->Commit({}, {}, tokens);
    return live ? cached : 0;
  };
  std::vector<TextRunnerToken> conversation(12, 7);
  (void)turn(conversation);
  for (const TextRunnerToken other : {100, 101}) {
    (void)turn(std::vector<TextRunnerToken>(12, other));
    const auto frontier = conversation.size();
    conversation.push_back(8);
    assert(turn(conversation) == frontier);
  }
}
void RestartAndCorruption() {
  const TemporaryDirectory temp("gufo-component-serving-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20};
  const std::vector<TextRunnerToken> prompt{10, 11, 12, 13, 14, 15, 16, 17, 18};
  TextRunnerToken token;
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto first = pool.Acquire(prompt);
    Prefill(first);
    token = Generate(first);
    (void)first.Commit();
  }
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto retry = pool.Acquire(prompt);
    assert(retry.cache_disk_hit());
    assert(retry.cached_prompt_tokens() == prompt.size());
    assert(Generate(retry) == token);
    (void)retry.Commit();
  }
  for (const auto& kind : {"chunks", "private"}) {
    for (const auto& file :
         std::filesystem::directory_iterator(directory / "v2" / kind)) {
      std::fstream bytes(file.path(),
                         std::ios::in | std::ios::out | std::ios::binary);
      char byte;
      bytes.read(&byte, 1);
      byte ^= 0x5a;
      bytes.seekp(0);
      bytes.write(&byte, 1);
    }
  }
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto corrupted = pool.Acquire(prompt);
    assert(!corrupted.cache_hit());
    Prefill(corrupted);
    assert(Generate(corrupted) == token);
    corrupted.Invalidate();
  }
  std::filesystem::remove_all(directory);
}
void DiskBudgetEvictsAnUnpinnedConversation() {
  const TemporaryDirectory temp("gufo-component-budget-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1024};
  const std::vector<TextRunnerToken> first{10, 11, 12, 13, 14, 15, 16, 17, 18};
  auto second = first;
  second.front() = 999;
  for (const auto& prompt : {first, second}) {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto request = pool.Acquire(prompt);
    Prefill(request);
    (void)Generate(request);
    assert(request.Commit().disk_queued_bytes > 0);
  }
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto retained = pool.Acquire(second);
    assert(retained.cache_disk_hit());
    retained.Invalidate();
    auto evicted = pool.Acquire(first);
    assert(!evicted.cache_hit());
    evicted.Invalidate();
  }
  std::filesystem::remove_all(directory);
}
std::uintmax_t ManagedBytes(const std::filesystem::path& directory) {
  std::uintmax_t bytes{};
  for (const auto& file :
       std::filesystem::recursive_directory_iterator(directory / "v2"))
    if (file.is_regular_file())
      bytes += file.file_size();
  return bytes;
}
// The RAM checkpoint tier is gone: a prompt captured at its end goes to disk
// (private state and tail through a host arena, completed rows lent by the
// slot) and restores byte for byte after a restart. The ledger holds no
// payload category at any point.
void DiskOnlyRoundTripIsByteIdentical() {
  const TemporaryDirectory temp("gufo-component-disk-only-round-trip");
  const TextRunnerDiskCacheOptions disk{.directory = temp.path().string(),
                                        .capacity_bytes = 1 << 24};
  std::vector<TextRunnerToken> prompt(1000);
  std::iota(prompt.begin(), prompt.end(), 10);
  // Cold references: the prompt's next token, and the token after it.
  TextRunnerToken expected{}, second{};
  auto next = prompt;
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1);
    auto cold = pool.Acquire(prompt, sampling::SamplingConfig{}, {}, {}, false);
    Prefill(cold);
    expected = Generate(cold);
    cold.Invalidate();
    next.push_back(expected);
    auto cold_next =
        pool.Acquire(next, sampling::SamplingConfig{}, {}, {}, false);
    Prefill(cold_next);
    second = Generate(cold_next);
    cold_next.Invalidate();
  }
  const auto payload = [](const cache::ResourceSnapshot& ledger) {
    std::size_t bytes = 0;
    for (const auto category : {cache::ResourceCategory::kBackingFree,
                                cache::ResourceCategory::kBackingAssigned,
                                cache::ResourceCategory::kBackingMaterialized,
                                cache::ResourceCategory::kPrivateState,
                                cache::ResourceCategory::kPrivateTail})
      bytes += ledger.peak_bytes[static_cast<std::size_t>(category)];
    return bytes;
  };
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto request = pool.Acquire(prompt);
    Prefill(request);
    assert(Generate(request) == expected);
    const auto committed = request.Commit();
    assert(committed.disk_queued_bytes > 0);
    assert(committed.snapshot_bytes == 0);
    assert(committed.component_cache &&
           payload(committed.component_cache->ledger) == 0);
    assert(WaitWritten(pool));
    // Shutdown also writes the live frontier (prompt and generated token).
  }
  {
    // After a restart the prompt's own checkpoint and the frontier written at
    // shutdown restore from disk, and continue as a cold run does.
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto longer = prompt;
    longer.push_back(12345);
    auto branch = pool.Acquire(longer);
    assert(branch.cache_disk_hit() &&
           branch.cached_prompt_tokens() == prompt.size());
    branch.Invalidate();
    auto restored = pool.Acquire(next);
    assert(restored.cache_disk_hit());
    assert(restored.cached_prompt_tokens() >= prompt.size());
    Prefill(restored);
    assert(Generate(restored) == second);
    const auto metrics = restored.Commit();
    assert(metrics.component_cache &&
           metrics.component_cache->reuse == "disk" &&
           payload(metrics.component_cache->ledger) == 0);
  }
}
// A long run of conversations, each published at its prompt end, with a
// disk budget far smaller than their total: the writer evicts the oldest,
// every publication succeeds, the ledger stays within its limit (metadata
// does not accumulate with the disk catalog's churn) and every admission
// succeeds.
void LongPublicationRunKeepsTheLedgerBounded() {
  const TemporaryDirectory temp("gufo-component-long-publication-run");
  const TextRunnerDiskCacheOptions disk{.directory = temp.path().string(),
                                        .capacity_bytes = 256 << 10,
                                        .min_checkpoint_step_tokens = 1};
  const auto previous_level = Logger::Level();
  Logger::SetLevel(LogLevel::kInfo);
  std::ostringstream output;
  auto* previous_sink = std::clog.rdbuf(output.rdbuf());
  std::size_t settled_metadata = 0, last_metadata = 0;
  {
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 2, disk);
    const auto metadata = [&] {
      return runner->ledger->Snapshot()
          .bytes[static_cast<std::size_t>(cache::ResourceCategory::kMetadata)];
    };
    constexpr int kConversations = 300;
    for (int i = 0; i < kConversations; ++i) {
      std::vector<TextRunnerToken> prompt(400);
      std::iota(prompt.begin(), prompt.end(), 100000 * (i + 1));
      auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
      assert(lease);
      auto& state = dynamic_cast<State&>(lease->state());
      runner->adapter->Append(state.slot, prompt, prompt);
      assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(),
                                       false, SnapshotPurpose::kContinuation,
                                       {}));
      (void)lease->Commit(prompt, lease->Capture(), prompt);
      assert(lease->Metrics().disk_queued_bytes > 0);
      lease.reset();
      assert(WaitWritten(cache));
      const auto snapshot = runner->ledger->Snapshot();
      const auto limits = runner->ledger->Limits();
      assert(snapshot.ram_bytes <= limits.ram_bytes);
      assert(snapshot.total_bytes <= limits.total_bytes);
      if (i == 49)
        settled_metadata = metadata();
      last_metadata = metadata();
    }
    // The catalog holds about as many entries after 300 conversations as
    // after 50: eviction returns their metadata.
    assert(last_metadata <= settled_metadata + (64 << 10));
  }
  std::clog.rdbuf(previous_sink);
  Logger::SetLevel(previous_level);
  const auto log = output.str();
  std::size_t published = 0;
  for (auto at = log.find("event=published"); at != std::string::npos;
       at = log.find("event=published", at + 1))
    ++published;
  assert(published == 300);
  // 300 publications cannot fit 256 KiB: each eviction is logged with the
  // victim and the store's size afterwards.
  assert(log.find("event=disk_evicted reason=lru checkpoint_id=") !=
         std::string::npos);
  assert(log.find(" managed_bytes=") != std::string::npos);
  assert(ManagedBytes(temp.path()) <= (256 << 10));
  assert(log.find("persistence_incomplete") == std::string::npos);
  assert(log.find("persistence_refused") == std::string::npos);
  assert(log.find("publication_untracked") == std::string::npos);
}
// A bookkeeping failure anywhere while a finished request is cached (the
// publication, the live frontier's index entry, the slot commit) leaves it
// uncached: Commit returns, and the next request runs normally.
void CommitFailureLeavesTheRequestUncached() {
  std::vector<TextRunnerToken> prompt(200);
  std::iota(prompt.begin(), prompt.end(), 10);
  TextRunnerToken expected{}, second{};
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1);
    auto cold = pool.Acquire(prompt, sampling::SamplingConfig{}, {}, {}, false);
    Prefill(cold);
    expected = Generate(cold);
    auto next = prompt;
    next.push_back(expected);
    cold.Invalidate();
    auto cold_next =
        pool.Acquire(next, sampling::SamplingConfig{}, {}, {}, false);
    Prefill(cold_next);
    second = Generate(cold_next);
    cold_next.Invalidate();
  }
  for (const auto step :
       {cache::LedgerStep::kReserve, cache::LedgerStep::kConvert})
    for (std::size_t n = 0; n < 24; ++n) {
      const TemporaryDirectory temp("gufo-component-commit-failure");
      auto runner = std::make_shared<Runner>();
      TextRunnerPool pool(
          runner, 1,
          TextRunnerDiskCacheOptions{.directory = temp.path().string(),
                                     .capacity_bytes = 1 << 22,
                                     .min_checkpoint_step_tokens = 1});
      auto request = pool.Acquire(prompt);
      Prefill(request);
      assert(Generate(request) == expected);
      runner->ledger->FailAfter(step, n);
      (void)request.Commit();
      runner->ledger->ClearFaults();
      assert(WaitWritten(pool));
      auto next = prompt;
      next.push_back(expected);
      auto follow = pool.Acquire(next);
      Prefill(follow);
      assert(Generate(follow) == second);
      follow.Invalidate();
    }
}
void UnpersistedRequestStaysOffDisk() {
  const TemporaryDirectory temp("gufo-component-unpersisted-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .min_checkpoint_step_tokens = 1};
  const std::vector<TextRunnerToken> prompt{10, 11, 12, 13, 14, 15, 16, 17, 18};
  const auto acquire = [&](TextRunnerPool& pool, bool persist) {
    return pool.Acquire(prompt, {}, {}, {}, true, 0, true, 0, persist);
  };
  TextRunnerToken token;
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto first = acquire(pool, false);
    Prefill(first);
    token = Generate(first);
    assert(first.Commit().disk_queued_bytes == 0);
    // Without RAM checkpoints only the live frontier (prompt and generated
    // token) remains; the bare prompt is not reused.
    auto again = acquire(pool, false);
    assert(!again.cache_disk_hit());
    again.Invalidate();
  }
  assert(ManagedBytes(directory) == 0);
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto cold = acquire(pool, true);
    assert(!cold.cache_hit());
    Prefill(cold);
    assert(Generate(cold) == token);
    assert(cold.Commit().disk_queued_bytes > 0);
  }
  assert(ManagedBytes(directory) > 0);
  {
    // Reading what other requests stored is allowed.
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto reader = acquire(pool, false);
    assert(reader.cache_disk_hit());
    assert(reader.cached_prompt_tokens() == prompt.size());
    reader.Invalidate();
  }
  std::filesystem::remove_all(directory);
}
// The live frontier of a cache_persist=false request is not written back:
// neither by the idle write-back nor by the capture before its slot is reset.
void UnpersistedLiveFrontierStaysOffDisk() {
  const TemporaryDirectory temp("gufo-component-unpersisted-live-test");
  const TextRunnerDiskCacheOptions disk{.directory = temp.path().string(),
                                        .capacity_bytes = 1 << 22};
  std::vector<TextRunnerToken> conversation(100);
  std::iota(conversation.begin(), conversation.end(), 10);
  {
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 1, disk);
    const auto turn = [&](const std::vector<TextRunnerToken>& tokens,
                          bool persist) {
      auto lease = cache.Acquire(tokens, {}, {}, true, 0, true, persist);
      auto& state = dynamic_cast<State&>(lease->state());
      const auto tail = std::span(tokens).subspan(lease->cached_tokens());
      runner->adapter->Append(state.slot, tail, tail);
      (void)lease->Commit({}, {}, tokens);
    };
    turn(conversation, false);
    assert(cache.WriteBackIdle(std::chrono::milliseconds(0)) == 0);
    // The only slot goes to another prompt: the frontier stays in RAM only.
    turn(std::vector<TextRunnerToken>(12, 5), true);
  }
  assert(ManagedBytes(temp.path()) == 0);
}
void UnpersistedCaptureIsRefused() {
  // Every capture goes to disk: a request that stays off disk captures none.
  const TemporaryDirectory temp("gufo-component-unpersisted-capture-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .min_checkpoint_step_tokens = 1};
  auto runner = std::make_shared<Runner>();
  {
    ComponentTextCache cache(runner, 1, disk);
    std::vector<TextRunnerToken> prompt(64);
    std::iota(prompt.begin(), prompt.end(), 10);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true, false);
    auto& state = dynamic_cast<State&>(lease->state());
    runner->adapter->Append(state.slot, prompt, prompt);
    assert(lease->CaptureRefused(prompt.size()));
    assert(!lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(),
                                      false, SnapshotPurpose::kContinuation,
                                      prompt));
    lease->Invalidate();
  }
  assert(ManagedBytes(directory) == 0);
  std::filesystem::remove_all(directory);
}
void LoweredDiskBudgetEvictsOldestAtStartup() {
  const TemporaryDirectory temp("gufo-component-lowered-budget-test");
  const std::filesystem::path directory = temp.path();
  TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                  .capacity_bytes = 1 << 20};
  const std::vector<TextRunnerToken> first{10, 11, 12, 13, 14, 15, 16, 17, 18};
  auto second = first;
  second.front() = 999;
  for (const auto& prompt : {first, second}) {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto request = pool.Acquire(prompt);
    Prefill(request);
    (void)Generate(request);
    assert(request.Commit().disk_queued_bytes > 0);
  }
  // Restarting below the retained bytes keeps the budget's meaning: the
  // oldest conversation leaves at startup instead of failing to start.
  disk.capacity_bytes = ManagedBytes(directory) - 1;
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    assert(ManagedBytes(directory) <= disk.capacity_bytes);
    auto retained = pool.Acquire(second);
    assert(retained.cache_disk_hit());
    retained.Invalidate();
    auto evicted = pool.Acquire(first);
    assert(!evicted.cache_hit());
    evicted.Invalidate();
  }
  std::filesystem::remove_all(directory);
}
void OrphansLeaveBeforeCheckpoints() {
  const TemporaryDirectory temp("gufo-component-orphan-test");
  const std::filesystem::path directory = temp.path();
  TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                  .capacity_bytes = 1 << 20};
  const std::vector<TextRunnerToken> first{10, 11, 12, 13, 14, 15, 16, 17, 18};
  auto second = first;
  second.front() = 999;
  for (const auto& prompt : {first, second}) {
    TextRunnerPool pool(std::make_shared<Runner>(), 1, disk);
    auto request = pool.Acquire(prompt);
    Prefill(request);
    (void)Generate(request);
    assert(request.Commit().disk_queued_bytes > 0);
  }
  // A crash left an unreferenced chunk that pushes the directory over a budget
  // the referenced checkpoints still fit; reclaiming it must suffice.
  disk.capacity_bytes = ManagedBytes(directory);
  const auto orphan = directory / "v2/chunks" / (std::string(32, 'f') + ".bin");
  std::ofstream(orphan, std::ios::binary) << std::string(4096, 'x');
  {
    TextRunnerPool pool(std::make_shared<Runner>(), 1, disk);
    assert(!std::filesystem::exists(orphan));
    assert(ManagedBytes(directory) <= disk.capacity_bytes);
    for (const auto& prompt : {first, second}) {
      auto restored = pool.Acquire(prompt);
      assert(restored.cache_disk_hit());
      restored.Invalidate();
    }
  }
  std::filesystem::remove_all(directory);
}
void StagingIsInternal() {
  const TemporaryDirectory temp("gufo-component-staging-test");
  const std::filesystem::path directory = temp.path();
  // Without the legacy option (rejected here), disk writes and restores
  // stream through the internal staging buffer.
  TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                  .capacity_bytes = 1 << 20};
  const std::vector<TextRunnerToken> prompt{10, 11, 12, 13, 14, 15, 16, 17, 18};
  {
    TextRunnerPool pool(std::make_shared<Runner>(), 1, disk);
    auto request = pool.Acquire(prompt);
    Prefill(request);
    (void)Generate(request);
    assert(request.Commit().disk_queued_bytes > 0);
  }
  {
    TextRunnerPool pool(std::make_shared<Runner>(), 1, disk);
    auto restored = pool.Acquire(prompt);
    assert(restored.cache_disk_hit());
    restored.Invalidate();
  }
  std::filesystem::remove_all(directory);
}
void DiskDirectoryHasOneOwner() {
  class CommitRunner final : public Runner {
  public:
    mutable std::size_t streams{};
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger);
      // Every execution slot takes a preservation stream when it allocates.
      resources->stream = [this, stream = std::move(resources->stream)] {
        ++streams;
        return stream();
      };
      return resources;
    }
  };
  const TemporaryDirectory temp("gufo-component-owner-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20};
  {
    TextRunnerPool owner(std::make_shared<CommitRunner>(), 1, disk);
    auto second_runner = std::make_shared<CommitRunner>();
    std::string error;
    try {
      TextRunnerPool second(second_runner, 1, disk);
    } catch (const std::exception& e) {
      error = e.what();
    }
    assert(error.find("ownership") != std::string::npos &&
           error.find(directory.string()) != std::string::npos);
    // The losing server fails before allocating slots.
    assert(!second_runner->streams);
  }
  // Ownership ends with the owning server.
  {
    auto runner = std::make_shared<CommitRunner>();
    TextRunnerPool reopened(runner, 1, disk);
    assert(runner->streams);
  }
  std::filesystem::remove_all(directory);
}
void PublishedCheckpointKeepsDiskStep() {
  // A capture within the disk step of a published checkpoint it continues is
  // refused.
  const TemporaryDirectory temp("gufo-component-step-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .min_checkpoint_step_tokens = 16};
  std::vector<TextRunnerToken> prompt(48);
  std::iota(prompt.begin(), prompt.end(), 10);
  const std::vector<TextRunnerToken> first(prompt.begin(), prompt.begin() + 40);
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 1, disk);
  {
    auto lease = cache.Acquire(first, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(lease->state());
    runner->adapter->Append(state.slot, first, first);
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), first.size(), false,
                                     SnapshotPurpose::kContinuation, {}));
    (void)lease->PublishSnapshot(first, lease->Capture(), false);
    assert(lease->Metrics().disk_queued_bytes > 0);
    lease->Invalidate();
    // Invalidation releases the execution slot so the idle queue can publish.
    // Wait for completion rather than guessing when a created file is attached.
    for (int i = 0; i < 500 && lease->Metrics().persistence->completed == 0;
         ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    assert(lease->Metrics().persistence->completed == 1);
  }
  auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
  assert(lease->cached_tokens() == first.size() && lease->restored_from_disk());
  auto& state = dynamic_cast<State&>(lease->state());
  const auto tail = std::span(prompt).subspan(first.size());
  runner->adapter->Append(state.slot, tail, tail);
  assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(), false,
                                   SnapshotPurpose::kContinuation, {}));
  (void)lease->PublishSnapshot(prompt, lease->Capture(), false);
  assert(lease->Metrics().disk_queued_bytes == 0);
  lease->Invalidate();
  std::filesystem::remove_all(directory);
}
void LearnedBoundaryLeavesTheNextStep() {
  // A shared (learned) boundary written to disk, such as a system prompt
  // boundary, does not consume the spacing allowance: the conversation's own
  // checkpoint within the step after it is still written (upstream
  // disk-spacing: the first turn reaches disk).
  const TemporaryDirectory temp("gufo-component-learned-step-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .min_checkpoint_step_tokens = 16};
  std::vector<TextRunnerToken> prompt(48);
  std::iota(prompt.begin(), prompt.end(), 10);
  const std::vector<TextRunnerToken> shared(prompt.begin(),
                                            prompt.begin() + 40);
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 1, disk);
  {
    auto lease = cache.Acquire(shared, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(lease->state());
    runner->adapter->Append(state.slot, shared, shared);
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), shared.size(),
                                     false, SnapshotPurpose::kBranchPoint, {}));
    (void)lease->PublishSnapshot(shared, lease->Capture(), false);
    assert(lease->Metrics().disk_queued_bytes > 0);
    lease->Invalidate();
    for (int i = 0; i < 500 && lease->Metrics().persistence->completed == 0;
         ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    assert(lease->Metrics().persistence->completed == 1);
  }
  auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
  assert(lease->cached_tokens() == shared.size());
  auto& state = dynamic_cast<State&>(lease->state());
  const auto tail = std::span(prompt).subspan(shared.size());
  runner->adapter->Append(state.slot, tail, tail);
  assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(), false,
                                   SnapshotPurpose::kContinuation, {}));
  (void)lease->PublishSnapshot(prompt, lease->Capture(), false);
  assert(lease->Metrics().disk_queued_bytes > 0);
  lease->Invalidate();
  std::filesystem::remove_all(directory);
}
void NearTailDiskBranchSurvivesRestart() {
  const TemporaryDirectory temp("gufo-component-branch-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20};
  std::vector<TextRunnerToken> seed(1100);
  std::iota(seed.begin(), seed.end(), 10);
  auto branch = seed;
  branch.resize(800);
  branch.back() = 9999;
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto first = pool.Acquire(seed);
    Prefill(first);
    (void)Generate(first);
    (void)first.Commit();
    auto fork = pool.Acquire(branch);
    Prefill(fork);
    (void)Generate(fork);
    (void)fork.Commit();
  }
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    branch.back() = 8888;
    auto restored = pool.Acquire(branch);
    assert(restored.cache_disk_hit());
    assert(restored.cached_prompt_tokens() == 799);
    Prefill(restored);
    const auto token = Generate(restored);
    restored.Invalidate();
    auto cold = pool.Acquire(branch, sampling::SamplingConfig{}, {}, {}, false);
    Prefill(cold);
    assert(Generate(cold) == token);
  }
  std::filesystem::remove_all(directory);
}
void UnbackedDeepRowsSurviveDiskRestart() {
  struct Gate {
    std::atomic<bool> entered{false};
    std::binary_semaphore reading{0}, release{0};
  };
  struct HeldRunner final : Runner {
    std::shared_ptr<Gate> gate = std::make_shared<Gate>();
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger);
      resources->copy_observed_rows =
          [gate = gate, copy = resources->copy_observed_rows](
              auto source, auto destination, cache::Stream& stream) {
            if (!gate->entered.exchange(true)) {
              gate->reading.release();
              gate->release.acquire();
            }
            return copy(source, destination, stream);
          };
      return resources;
    }
  };
  const TemporaryDirectory temp("gufo-component-deep-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .min_checkpoint_step_tokens = 1};
  std::vector<TextRunnerToken> prompt(5001);
  std::iota(prompt.begin(), prompt.end(), 10);
  std::uint64_t hash;
  {
    auto runner = std::make_shared<HeldRunner>();
    ComponentTextCache cache(runner, 1, disk);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(lease->state());
    runner->adapter->Append(state.slot, prompt, prompt);
    hash = runner->adapter->RecurrentHash(state.slot);
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(),
                                     false, SnapshotPurpose::kContinuation,
                                     prompt));
    // Over 40,000 logical row bytes exceed the 32,768-byte row pool.
    auto snapshot = lease->Capture();
    assert(snapshot);
    assert(lease->PublishSnapshot(prompt, std::move(snapshot), false) == 0);
    assert(lease->Metrics().disk_queued_bytes > 40000);
    assert(runner->gate->reading.try_acquire_for(std::chrono::seconds(5)));
    const auto pending = cache.PendingPublication(prompt, {});
    assert(pending && pending->checkpoint_id &&
           pending->pending_tokens == prompt.size() &&
           pending->available_tokens == 0 && pending->Gain() == prompt.size());
    assert(!cache.PendingPublication(prompt, {}, pending->checkpoint_id + 1));
    assert(!cache.PendingPublication(std::span(prompt).first(4096), {}));
    auto different = prompt;
    ++different.front();
    assert(!cache.PendingPublication(different, {}));
    TextPromptContext image;
    image.cache_identity = {99};
    assert(!cache.PendingPublication(prompt, &image));
    (void)lease->Commit(prompt, {}, prompt);
    // A newly usable live frontier makes the pending write unhelpful. The
    // tracked job remains identifiable so its existing deadline can shrink;
    // a fresh admission must not begin waiting for it.
    const auto tracked =
        cache.PendingPublication(prompt, {}, pending->checkpoint_id);
    assert(tracked && tracked->Gain() == 0);
    assert(!cache.PendingPublication(prompt, {}));
    runner->gate->release.release();
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (cache.PendingPublication(prompt, {}, pending->checkpoint_id) &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    assert(!cache.PendingPublication(prompt, {}, pending->checkpoint_id));
  }
  {
    const auto seed_size = prompt.size();
    std::vector<TextRunnerToken> appended(63);
    std::iota(appended.begin(), appended.end(), 8000);
    prompt.insert(prompt.end(), appended.begin(), appended.end());
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 1, disk);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(lease->restored_from_disk() && lease->cached_tokens() == seed_size);
    auto& state = dynamic_cast<State&>(lease->state());
    assert(runner->adapter->RecurrentHash(state.slot) == hash);
    runner->adapter->Append(state.slot, appended, appended);
    hash = runner->adapter->RecurrentHash(state.slot);
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(),
                                     false, SnapshotPurpose::kContinuation,
                                     prompt));
    auto snapshot = lease->Capture();
    assert(snapshot);
    assert(lease->PublishSnapshot(prompt, std::move(snapshot), false) == 0);
    assert(lease->Metrics().disk_queued_bytes > 40000);
    (void)lease->Commit(prompt, {}, prompt);
  }
  {
    cache::ResourceLedger ledger({1 << 24, 1 << 24, 0, 0});
    cache::DiskStore store(ledger, directory, disk.capacity_bytes);
    assert(store.Entries().size() == 2);
    const cache::DiskManifest* parent = nullptr;
    const cache::DiskManifest* child = nullptr;
    for (const auto& entry : store.Entries()) {
      if (entry.Manifest().tokens.size() == 5001)
        parent = &entry.Manifest();
      else if (entry.Manifest().tokens.size() == prompt.size())
        child = &entry.Manifest();
    }
    assert(parent && child);
    for (std::size_t i = 0; i < parent->components.size(); ++i) {
      const auto& before = parent->components[i];
      const auto& after = child->components[i];
      for (std::size_t j = 0; j < before.chunks.size(); ++j)
        assert(before.chunks[j].file == after.chunks[j].file);
      if (before.private_state)
        assert(before.private_state->file != after.private_state->file);
      if (before.tail) {
        assert(after.chunks.size() > before.chunks.size());
        assert(before.tail->file != after.chunks[before.chunks.size()].file);
      }
    }
  }
  {
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 1, disk);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(lease->restored_from_disk() &&
           lease->cached_tokens() == prompt.size());
    auto& state = dynamic_cast<State&>(lease->state());
    assert(runner->adapter->RecurrentHash(state.slot) == hash);
  }
  std::filesystem::remove_all(directory);
}
// Commits `tokens` as the live frontier of a fresh lease; returns the reuse.
std::pair<std::size_t, bool> LiveTurn(
    ComponentTextCache& cache, Runner& runner,
    const std::vector<TextRunnerToken>& tokens,
    std::shared_ptr<const TextPromptContext> context = {}) {
  auto lease = cache.Acquire(tokens, {}, std::move(context), true, 0, true);
  const auto cached = lease->cached_tokens();
  const bool disk_hit = lease->restored_from_disk();
  auto& state = dynamic_cast<State&>(lease->state());
  const auto tail = std::span(tokens).subspan(cached);
  runner.adapter->Append(state.slot, tail, tail);
  (void)lease->Commit({}, {}, tokens);
  return {cached, disk_hit};
}
// Polls admission of `prompt` until the slot it needs is free (a queued
// disk write holding it has finished).
bool WaitForAdmission(ComponentTextCache& cache,
                      const std::vector<TextRunnerToken>& prompt,
                      std::shared_ptr<const TextPromptContext> context = {}) {
  for (int i = 0; i < 1000; ++i) {
    if (cache.PrepareAdmission(prompt, context.get(), true, 0))
      return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}
// Blocks the first transfer stream another thread (the disk writer) asks
// for, until released.
struct WriterGate {
  std::atomic<bool> first{false};
  std::binary_semaphore entered{0}, release{0};
  std::function<void()> Hook() {
    return [this] {
      if (!first.exchange(true)) {
        entered.release();
        release.acquire();
      }
    };
  }
};
// Polls until the runner's live staging bytes are `bytes`.
bool WaitForStaging(const Runner& runner, std::size_t bytes) {
  for (int i = 0; i < 1000; ++i) {
    if (runner.staging_live->load() == bytes)
      return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}
void HeldReassignmentWritesOnTheWriter() {
  // A frontier RAM cannot hold (8192 rows per append component, 2048 backing
  // blocks) sits in the only idle slot when another prompt is admitted, and
  // cannot be snapshot to host memory: the allocation fails, snapshots are
  // disabled, a transfer stream fails mid-batch, or the host has no room. The
  // admission check hands the slot to the disk writer instead of writing on
  // the scheduler thread, and keeps the request that needs the slot waiting;
  // the other slot keeps running. Once written, the slot is admitted and the
  // frontier restores after a restart. Snapshot memory is returned.
  for (const std::string_view reason :
       {"allocation_failed", "disabled", "copy_failed", "host_memory"}) {
    const bool lend = reason == "allocation_failed" || reason == "copy_failed";
    const auto previous_level = Logger::Level();
    Logger::SetLevel(LogLevel::kInfo);
    std::ostringstream output;
    auto* previous_sink = std::clog.rdbuf(output.rdbuf());
    const TemporaryDirectory temp("gufo-component-held-reassignment");
    const std::filesystem::path directory = temp.path();
    const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                          .capacity_bytes = 1 << 22};
    std::vector<TextRunnerToken> conversation(8192);
    std::iota(conversation.begin(), conversation.end(), 10);
    const std::vector<TextRunnerToken> side(12, 5), other(12, 6);
    {
      auto runner = std::make_shared<Runner>();
      if (reason == "disabled")
        runner->snapshot_bytes = 0;
      if (reason == "host_memory")
        runner->snapshot_room = [](std::size_t) { return false; };
      runner->snapshot_idle = std::chrono::milliseconds(20);
      ComponentTextCache cache(runner, 2, disk);
      const auto staged = runner->staging_live->load();
      runner->fail_staging = reason == "allocation_failed";
      if (!lend)
        runner->adapter->LimitBorrowedRows(1);
      (void)LiveTurn(cache, *runner, conversation);
      // The other slot stays busy with a running request.
      auto running = cache.Acquire(side, {}, {}, true, 0, true);
      WriterGate gate;
      runner->background_stream = gate.Hook();
      // The sixth stream of the snapshot copy (in its second batch of four)
      // cannot be leased.
      std::size_t streams = 0;
      std::function<void()> lose_stream = [&] {
        if (++streams == 6)
          throw std::runtime_error("stream lost");
        runner->on_stream = lose_stream;
      };
      if (reason == "copy_failed")
        runner->on_stream = lose_stream;
      assert(!cache.PrepareAdmission(other, nullptr, true, 0));
      assert(reason != "copy_failed" || streams == 6);
      assert(gate.entered.try_acquire_for(std::chrono::seconds(10)));
      // The scheduler thread returned while the write is in flight; the
      // request needing that slot still waits, the running one progresses.
      assert(!cache.PrepareAdmission(other, nullptr, true, 0));
      auto& state = dynamic_cast<State&>(running->state());
      runner->adapter->Append(state.slot, side, side);
      (void)running->Commit({}, {}, side);
      gate.release.release();
      assert(WaitForAdmission(cache, other));
      (void)LiveTurn(cache, *runner, other);
      runner->background_stream = {};
      // Arenas the failed copy used return to the pool and are freed idle.
      assert(WaitForStaging(*runner, staged));
      runner->fail_staging = false;
    }
    std::clog.rdbuf(previous_sink);
    Logger::SetLevel(previous_level);
    const auto log = output.str();
    assert(
        log.find("event=slot_snapshot_fallback reason=" + std::string(reason) +
                 " slot=1 tokens=8192") != std::string::npos);
    assert(log.find("held_write=true") != std::string::npos);
    assert(log.find("event=slot_snapshot ") == std::string::npos);
    assert((reason == "copy_failed") ==
           (log.find("event=slot_snapshot_copy_failed") != std::string::npos));
    assert((reason == "copy_failed") ==
           (log.find("event=slot_snapshot_released") != std::string::npos));
    auto runner = std::make_shared<Runner>();
    {
      ComponentTextCache cache(runner, 1, disk);
      auto next = conversation;
      next.push_back(7);
      auto lease = cache.Acquire(next, {}, {}, true, 0, true);
      assert(lease->cached_tokens() == conversation.size() &&
             lease->restored_from_disk());
      lease->Invalidate();
    }
    std::filesystem::remove_all(directory);
  }
}
void FailedHeldWriteIsNotRetriedForever() {
  // A reassignment write the disk can never take (the frontier exceeds the
  // whole disk budget) fails once, from a held slot or from a host snapshot
  // (whose memory it then frees); admission then proceeds without handing
  // the same frontier to the writer again.
  for (const bool snapshot : {false, true}) {
    const TemporaryDirectory temp("gufo-component-held-failure");
    const TextRunnerDiskCacheOptions disk{.directory = temp.path().string(),
                                          .capacity_bytes = 1024};
    std::vector<TextRunnerToken> conversation(8192);
    std::iota(conversation.begin(), conversation.end(), 10);
    const std::vector<TextRunnerToken> side(12, 5), other(12, 6);
    auto runner = std::make_shared<Runner>();
    if (!snapshot)
      runner->snapshot_bytes = 0;
    runner->snapshot_idle = std::chrono::milliseconds(20);
    ComponentTextCache cache(runner, 2, disk);
    const auto staged = runner->staging_live->load();
    (void)LiveTurn(cache, *runner, conversation);
    auto running = cache.Acquire(side, {}, {}, true, 0, true);
    assert(WaitForAdmission(cache, other));
    assert(WaitForStaging(*runner, staged));
    running->Invalidate();
    (void)LiveTurn(cache, *runner, other);
  }
}
void ReassignmentSnapshotReleasesTheSlotBeforeTheWrite() {
  // Two deep frontiers RAM cannot hold sit in idle slots while a third slot
  // runs a request. Room for one reassignment snapshot fits only the
  // smaller: the deeper one, reassigned first, keeps its slot for the writer
  // (cap reached), and that write stalls on the writer. The smaller is
  // reassigned next: admission copies it to host memory (four page-aligned
  // pieces per 16 KiB arena, four arenas), releases its slot at once, and its
  // write queues behind the stalled one. The new request runs in that slot
  // meanwhile, with no second capture of the frontier. Both frontiers
  // restore byte for byte after a restart, and the snapshot memory is freed
  // once written and idle.
  const TemporaryDirectory temp("gufo-component-reassignment-snapshot");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 22};
  std::vector<TextRunnerToken> deep(8192), shallow(6000);
  std::iota(deep.begin(), deep.end(), 10);
  std::iota(shallow.begin(), shallow.end(), 100000);
  const std::vector<TextRunnerToken> side(12, 5), first(12, 6), second(12, 8);
  std::uint64_t deep_hash{}, shallow_hash{};
  const auto previous_level = Logger::Level();
  Logger::SetLevel(LogLevel::kInfo);
  std::ostringstream output;
  auto* previous_sink = std::clog.rdbuf(output.rdbuf());
  {
    auto runner = std::make_shared<Runner>();
    // 4 KiB chunks. 6000 rows take 13 pieces (5 chunks and a tail per append
    // component, and the recurrent state), 48032 bytes in four arenas; 8192
    // rows take 17 pieces in five.
    runner->rows_per_chunk = 1024;
    runner->snapshot_arena_bytes = 16 << 10;
    runner->snapshot_bytes = 72 << 10;
    runner->snapshot_idle = std::chrono::milliseconds(20);
    ComponentTextCache cache(runner, 3, disk);
    const auto staged = runner->staging_live->load();
    const auto hash = [&](const std::vector<TextRunnerToken>& tokens) {
      auto next = tokens;
      next.push_back(7);
      auto lease = cache.Acquire(next, {}, {}, true, 0, true);
      assert(lease->cached_tokens() == tokens.size());
      const auto value = runner->adapter->RecurrentHash(
          dynamic_cast<State&>(lease->state()).slot);
      (void)lease->Commit({}, {}, tokens);
      return value;
    };
    (void)LiveTurn(cache, *runner, deep);
    (void)LiveTurn(cache, *runner, shallow);
    deep_hash = hash(deep);
    shallow_hash = hash(shallow);
    auto running = cache.Acquire(side, {}, {}, true, 0, true);
    WriterGate gate;
    runner->background_stream = gate.Hook();
    assert(!cache.PrepareAdmission(first, nullptr, true, 0));
    assert(gate.entered.try_acquire_for(std::chrono::seconds(10)));
    // The writer is stalled on the held write; the snapshot does not wait.
    assert(cache.PrepareAdmission(second, nullptr, true, 0));
    assert(runner->staging_live->load() == staged + (64 << 10));
    const auto captures = output.str();
    {
      auto lease = cache.Acquire(second, {}, {}, true, 0, true);
      assert(lease->cached_tokens() == 0);
      auto& state = dynamic_cast<State&>(lease->state());
      runner->adapter->Append(state.slot, second, second);
      (void)lease->Commit({}, {}, second);
    }
    // Acquire captured nothing more: the queued snapshot covers the frontier.
    assert(output.str().find("live_frontier_capture", captures.size()) ==
           std::string::npos);
    gate.release.release();
    assert(WaitForAdmission(cache, first));
    assert(WaitForStaging(*runner, staged));
    running->Invalidate();
    runner->background_stream = {};
  }
  std::clog.rdbuf(previous_sink);
  Logger::SetLevel(previous_level);
  const auto log = output.str();
  assert(log.find("event=slot_snapshot_fallback reason=cap slot=") !=
         std::string::npos);
  assert(log.find("event=slot_snapshot slot=") != std::string::npos);
  assert(log.find(" tokens=6000 bytes=48032 ms=") != std::string::npos);
  assert(log.find(" arenas=4 reused_bytes=0 allocated_bytes=65536") !=
         std::string::npos);
  assert(log.find("event=slot_snapshot_released") != std::string::npos);
  assert(log.find("charged_bytes=0") != std::string::npos);
  auto runner = std::make_shared<Runner>();
  runner->rows_per_chunk = 1024;
  {
    ComponentTextCache cache(runner, 1, disk);
    for (const auto* tokens : {&deep, &shallow}) {
      auto next = *tokens;
      next.push_back(7);
      auto lease = cache.Acquire(next, {}, {}, true, 0, true);
      assert(lease->cached_tokens() == tokens->size() &&
             lease->restored_from_disk());
      const auto& slot = dynamic_cast<State&>(lease->state()).slot;
      assert(std::ranges::equal(runner->adapter->TargetRows(slot), *tokens));
      assert(runner->adapter->RecurrentHash(slot) ==
             (tokens == &deep ? deep_hash : shallow_hash));
      lease->Invalidate();
    }
  }
  std::filesystem::remove_all(directory);
}
void SnapshotHostsRowsQueuedJobsObserve() {
  // A conversation's two prompt checkpoints go to the disk writer reading
  // their rows from the slot (observed rows): the first is
  // writing, stopped after reading two of its pieces, and the second is
  // queued behind it. The conversation's generated frontier is then
  // reassigned: its snapshot also gives both jobs host copies of the rows
  // they observe, shared with its own pieces (all of them here), before the
  // slot is reset for the new request. When the snapshot is refused, the
  // held write gives those rows host copies of their own (read again), or,
  // without room for them either, waits its turn behind the jobs reading
  // them. All three checkpoints publish, none incomplete, and each restores
  // byte for byte after a restart.
  for (const std::string_view mode : {"snapshot", "held_hosted", "in_order"}) {
    const TemporaryDirectory temp("gufo-component-snapshot-observed");
    const TextRunnerDiskCacheOptions disk{.directory = temp.path().string(),
                                          .capacity_bytes = 1 << 22,
                                          .min_checkpoint_step_tokens = 1};
    std::vector<TextRunnerToken> prompt(6144), generated(100);
    std::iota(prompt.begin(), prompt.end(), 10);
    std::iota(generated.begin(), generated.end(), 50000);
    const std::vector<TextRunnerToken> half(prompt.begin(),
                                            prompt.begin() + 3072);
    auto live = prompt;
    live.insert(live.end(), generated.begin(), generated.end());
    const std::vector<TextRunnerToken> side(12, 5), other(12, 6);
    std::uint64_t half_hash{}, prompt_hash{}, live_hash{};
    const auto previous_level = Logger::Level();
    Logger::SetLevel(LogLevel::kInfo);
    std::ostringstream output;
    auto* previous_sink = std::clog.rdbuf(output.rdbuf());
    {
      auto runner = std::make_shared<Runner>();
      runner->rows_per_chunk = 1024;
      runner->snapshot_arena_bytes = 16 << 10;
      // The two prompt captures stage their private states and tails in host
      // arenas first (two allocations); the third asks for the frontier.
      if (mode == "held_hosted")
        // The host has no room for the frontier, then room again.
        runner->snapshot_room = [calls = 0](std::size_t) mutable {
          return calls++ != 2;
        };
      if (mode == "in_order")
        // No room for the frontier nor for host copies of observed rows.
        runner->snapshot_room = [calls = 0](std::size_t) mutable {
          return calls++ < 2;
        };
      ComponentTextCache cache(runner, 2, disk);
      // The writer stops at its third device read.
      std::atomic<int> reads{};
      std::binary_semaphore entered{0}, release{0};
      runner->background_stream = [&] {
        if (++reads == 3) {
          entered.release();
          release.acquire();
        }
      };
      auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
      auto& state = dynamic_cast<State&>(lease->state());
      const auto checkpoint = [&](const std::vector<TextRunnerToken>& tokens) {
        assert(lease->TryReserveSnapshot(lease->CaptureBytes(), tokens.size(),
                                         false, SnapshotPurpose::kContinuation,
                                         tokens));
        const auto queued = lease->Metrics().disk_queued_bytes;
        (void)lease->PublishSnapshot(tokens, lease->Capture(), false);
        assert(lease->Metrics().disk_queued_bytes > queued);
      };
      runner->adapter->Append(state.slot, half, half);
      half_hash = runner->adapter->RecurrentHash(state.slot);
      checkpoint(half);
      assert(entered.try_acquire_for(std::chrono::seconds(10)));
      const auto rest = std::span(prompt).subspan(half.size());
      runner->adapter->Append(state.slot, rest, rest);
      prompt_hash = runner->adapter->RecurrentHash(state.slot);
      checkpoint(prompt);
      runner->adapter->Append(state.slot, generated, generated);
      live_hash = runner->adapter->RecurrentHash(state.slot);
      (void)lease->Commit({}, {}, live);
      lease.reset();
      auto running = cache.Acquire(side, {}, {}, true, 0, true);
      // A held write keeps the slot until it has read it.
      assert(cache.PrepareAdmission(other, nullptr, true, 0) ==
             (mode == "snapshot"));
      if (mode != "snapshot") {
        release.release();
        assert(WaitForAdmission(cache, other));
      }
      (void)LiveTurn(cache, *runner, other);
      if (mode == "snapshot")
        release.release();
      running->Invalidate();
    }
    std::clog.rdbuf(previous_sink);
    Logger::SetLevel(previous_level);
    const auto log = output.str();
    assert(log.find("event=persistence_incomplete") == std::string::npos);
    // Each foreground step is timed.
    if (mode == "snapshot")
      for (const auto* field :
           {" queue_ms=", " plan_ms=", " alloc_ms=", " copy_ms=", " attach_ms=",
            " sources_ms=", " pieces=15 "})
        assert(log.find(field) != std::string::npos);
    // Two components observed by each job, every row shared, or read again.
    assert(log.find(mode == "snapshot"
                        ? " observed_rows=4 shared_bytes=73728 extra_bytes=0"
                    : mode == "held_hosted"
                        ? "event=held_write_observed_rows slot=1 tokens=6244 "
                          "observed_rows=4 extra_bytes=73728"
                        : "event=held_write_observed_rows slot=1 tokens=6244 "
                          "in_order=true reason=host_memory") !=
           std::string::npos);
    auto runner = std::make_shared<Runner>();
    runner->rows_per_chunk = 1024;
    ComponentTextCache cache(runner, 1, disk);
    using Expected =
        std::pair<const std::vector<TextRunnerToken>*, std::uint64_t>;
    for (const auto& [tokens, hash] :
         {Expected{&half, half_hash}, Expected{&prompt, prompt_hash},
          Expected{&live, live_hash}}) {
      auto next = *tokens;
      next.push_back(7);
      auto lease = cache.Acquire(next, {}, {}, true, 0, true);
      assert(lease->cached_tokens() == tokens->size() &&
             lease->restored_from_disk());
      const auto& slot = dynamic_cast<State&>(lease->state()).slot;
      assert(std::ranges::equal(runner->adapter->TargetRows(slot), *tokens));
      assert(runner->adapter->RecurrentHash(slot) == hash);
      lease->Invalidate();
    }
  }
}
void SnapshotArenasArePooledAndReused() {
  // A snapshot's arenas return to the pool once written: the next snapshot
  // of the same size allocates nothing, and they stay pinned until they have
  // been idle for the configured time. Destroying the cache frees them.
  const TemporaryDirectory temp("gufo-component-snapshot-pool");
  const TextRunnerDiskCacheOptions disk{.directory = temp.path().string(),
                                        .capacity_bytes = 1 << 22};
  std::vector<TextRunnerToken> a(6000), b(6000);
  std::iota(a.begin(), a.end(), 10);
  std::iota(b.begin(), b.end(), 100000);
  const std::vector<TextRunnerToken> side(12, 5), first(12, 6), second(12, 8);
  const auto previous_level = Logger::Level();
  Logger::SetLevel(LogLevel::kInfo);
  std::ostringstream output;
  auto* previous_sink = std::clog.rdbuf(output.rdbuf());
  auto runner = std::make_shared<Runner>();
  runner->rows_per_chunk = 1024;
  runner->snapshot_arena_bytes = 16 << 10;
  {
    ComponentTextCache cache(runner, 2, disk);
    const auto staged = runner->staging_live->load();
    const auto written = [&](const std::vector<TextRunnerToken>& tokens) {
      for (int i = 0; i < 1000; ++i) {
        if (cache.CachedPrefixTokens(tokens, nullptr) != 0)
          return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      return false;
    };
    (void)LiveTurn(cache, *runner, a);
    auto running = cache.Acquire(side, {}, {}, true, 0, true);
    assert(cache.PrepareAdmission(first, nullptr, true, 0));
    (void)LiveTurn(cache, *runner, first);
    assert(written(a));
    (void)LiveTurn(cache, *runner, b);
    assert(cache.PrepareAdmission(second, nullptr, true, 0));
    (void)LiveTurn(cache, *runner, second);
    assert(written(b));
    // Pooled, not freed: the default idle time is a minute.
    assert(runner->staging_live->load() == staged + (64 << 10));
    running->Invalidate();
  }
  assert(runner->staging_live->load() == 0);
  std::clog.rdbuf(previous_sink);
  Logger::SetLevel(previous_level);
  const auto log = output.str();
  assert(log.find(" arenas=4 reused_bytes=0 allocated_bytes=65536") !=
         std::string::npos);
  assert(log.find(" arenas=4 reused_bytes=65536 allocated_bytes=0") !=
         std::string::npos);
  assert(log.find("event=slot_snapshot_released") == std::string::npos);
  auto restarted = std::make_shared<Runner>();
  restarted->rows_per_chunk = 1024;
  ComponentTextCache cache(restarted, 1, disk);
  for (const auto* tokens : {&a, &b}) {
    auto next = *tokens;
    next.push_back(7);
    auto lease = cache.Acquire(next, {}, {}, true, 0, true);
    assert(lease->cached_tokens() == tokens->size() &&
           lease->restored_from_disk());
    assert(std::ranges::equal(restarted->adapter->TargetRows(
                                  dynamic_cast<State&>(lease->state()).slot),
                              *tokens));
    lease->Invalidate();
  }
}
void SnapshotPoolCountsEvictsAndTrims() {
  // The pool's charge counts every arena it allocated, pooled or in use.
  // A partly failed allocation keeps what it made pooled; a cap refusal may
  // first give up pooled arenas of other sizes, which the trim thread frees;
  // idle arenas are freed and the charge returns to zero.
  constexpr std::size_t kKiB = 1024;
  auto live = std::make_shared<std::atomic<std::size_t>>();
  std::atomic<int> fail_after{-1};
  std::atomic<std::size_t> allocations{};
  const auto allocate = [&, live](std::size_t bytes) {
    if (fail_after.load() == 0)
      throw std::bad_alloc();
    if (fail_after.load() > 0)
      --fail_after;
    ++allocations;
    *live += bytes;
    auto data = std::shared_ptr<std::vector<std::uint8_t>>(
        new std::vector<std::uint8_t>(bytes), [live, bytes](auto* p) {
          *live -= bytes;
          delete p;
        });
    return cache::StagingAllocation{data, *data};
  };
  const auto wait_live = [&](std::size_t bytes) {
    for (int i = 0; i < 1000 && live->load() != bytes; ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return live->load() == bytes;
  };
  {
    // Partial allocation failure: the first new arena stays pooled, is
    // reused, and is trimmed once idle.
    auto pool = std::make_shared<SnapshotPool>(
        64 * kKiB, std::chrono::milliseconds(50), allocate);
    const std::array<std::size_t, 3> three{16 * kKiB, 16 * kKiB, 16 * kKiB};
    fail_after = 1;
    auto failed = pool->Take(three, {});
    assert(failed.refused == "allocation_failed" && failed.arenas.empty());
    assert(pool->Charged() == 16 * kKiB && live->load() == 16 * kKiB);
    fail_after = -1;
    const std::array<std::size_t, 1> one{16 * kKiB};
    auto reused = pool->Take(one, {});
    assert(reused.refused.empty() && reused.reused == 16 * kKiB &&
           reused.allocated == 0 && allocations == 1);
    reused = {};
    assert(wait_live(0));
    assert(pool->Charged() == 0);
  }
  {
    // A room refusal gives up nothing.
    auto pool = std::make_shared<SnapshotPool>(64 * kKiB, std::chrono::hours(1),
                                               allocate);
    const std::array<std::size_t, 1> small{16 * kKiB}, large{48 * kKiB},
        middle{32 * kKiB};
    (void)pool->Take(small, {});
    assert(pool->Charged() == 16 * kKiB);
    auto refused = pool->Take(large, [](std::size_t) { return false; });
    assert(refused.refused == "host_memory");
    assert(pool->Charged() == 16 * kKiB && live->load() == 16 * kKiB);
    // Room for 48 KiB evicts the pooled 16 KiB arena only if needed: it
    // fits beside it.
    auto held = pool->Take(large, {});
    assert(held.refused.empty() && held.allocated == 48 * kKiB);
    assert(pool->Charged() == 64 * kKiB);
    // 32 KiB does not fit even without the pooled arena: refused, after
    // that arena was given up and freed by the trim thread.
    auto capped = pool->Take(middle, {});
    assert(capped.refused == "cap");
    assert(pool->Charged() == 48 * kKiB);
    assert(wait_live(48 * kKiB));
    // Over the whole cap: refused without giving anything up.
    const std::array<std::size_t, 1> huge{80 * kKiB};
    assert(pool->Take(huge, {}).refused == "cap");
    assert(pool->Charged() == 48 * kKiB);
    held = {};
    // Freed with the pool.
  }
  assert(live->load() == 0);
}
void HeldTryIsPerExecution() {
  // A failed held write marks only that execution's frontier. Another
  // conversation of the same length later in the same slot is handed to the
  // writer again instead of being reset unwritten.
  const TemporaryDirectory temp("gufo-component-held-per-execution");
  const TextRunnerDiskCacheOptions disk{.directory = temp.path().string(),
                                        .capacity_bytes = 1024};
  std::vector<TextRunnerToken> first(8192), second(8192);
  std::iota(first.begin(), first.end(), 10);
  std::iota(second.begin(), second.end(), 20000);
  const std::vector<TextRunnerToken> side(12, 5), other(12, 6);
  auto runner = std::make_shared<Runner>();
  runner->snapshot_bytes = 0;
  ComponentTextCache cache(runner, 2, disk);
  (void)LiveTurn(cache, *runner, first);
  auto running = cache.Acquire(side, {}, {}, true, 0, true);
  // The first frontier's held write fails (larger than the whole disk).
  assert(WaitForAdmission(cache, other));
  (void)LiveTurn(cache, *runner, second);
  assert(!cache.PrepareAdmission(other, nullptr, true, 0));
  assert(WaitForAdmission(cache, other));
  running->Invalidate();
}
void SchedulerAdmitsOthersWhileASlotIsHeld() {
  // While the disk writer holds a conversation's slot, its next turn waits,
  // and an unrelated request queued behind it runs in the other slot.
  const TemporaryDirectory temp("gufo-component-scheduler-held");
  const TextRunnerDiskCacheOptions disk{.directory = temp.path().string(),
                                        .capacity_bytes = 1 << 22};
  // No host arenas: the write-back cannot stage the recurrent state, so it
  // keeps the slot for the writer (and the prompt end is not captured).
  std::vector<TextRunnerToken> conversation(4000);
  std::iota(conversation.begin(), conversation.end(), 10);
  auto runner = std::make_shared<Runner>();
  runner->snapshot_bytes = 0;
  auto pool = std::make_shared<TextRunnerPool>(runner, 2, disk);
  std::vector<TextRunnerToken> live = conversation;
  {
    auto request = pool->Acquire(conversation);
    Prefill(request);
    live.push_back(Generate(request));
    (void)request.Commit();
  }
  WriterGate gate;
  runner->background_stream = gate.Hook();
  assert(pool->WriteBackIdle(std::chrono::milliseconds(0)) == 1);
  assert(gate.entered.try_acquire_for(std::chrono::seconds(10)));
  auto next = live;
  next.push_back(7);
  {
    TextGenerationScheduler scheduler(pool);
    auto held = scheduler.Submit(next, 2, 0.0F);
    auto other = scheduler.Submit(std::vector<TextRunnerToken>(12, 5), 2, 0.0F);
    auto done = std::async(std::launch::async, [&] { return other.Wait(); });
    assert(done.wait_for(std::chrono::seconds(10)) ==
           std::future_status::ready);
    (void)done.get();
    gate.release.release();
    const auto result = held.Wait();
    assert(result.cached_prompt_tokens == live.size());
  }
  runner->background_stream = {};
}
void UnlendableWriteBackStreamsFromTheSlot() {
  // Rows the adapter neither lends nor retains, and no host arenas to copy
  // them to: the idle write-back hands the slot to the disk writer, which reads
  // them from it through the disk staging. WriteBackIdle returns before the
  // write; the next turn of that conversation waits for its slot and then
  // continues the live frontier, which also restores after a restart.
  const TemporaryDirectory temp("gufo-component-unlendable-write-back");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 22};
  std::vector<TextRunnerToken> conversation(8192);
  std::iota(conversation.begin(), conversation.end(), 10);
  auto next = conversation;
  next.push_back(7);
  {
    auto runner = std::make_shared<Runner>();
    // No host arenas to stage the unlent rows in.
    runner->snapshot_bytes = 0;
    ComponentTextCache cache(runner, 2, disk);
    runner->adapter->LimitBorrowedRows(1);
    (void)LiveTurn(cache, *runner, conversation);
    const auto copied = runner->adapter->CopiedOutRows();
    WriterGate gate;
    runner->background_stream = gate.Hook();
    assert(cache.WriteBackIdle(std::chrono::milliseconds(0)) == 1);
    assert(gate.entered.try_acquire_for(std::chrono::seconds(10)));
    assert(!cache.PrepareAdmission(next, nullptr, true, 0));
    // Another conversation runs in the other slot meanwhile.
    const auto [side_cached, side_disk] =
        LiveTurn(cache, *runner, std::vector<TextRunnerToken>(12, 5));
    assert(side_cached == 0 && !side_disk);
    gate.release.release();
    assert(WaitForAdmission(cache, next));
    runner->background_stream = {};
    // Every row was read once, straight into the disk staging.
    assert(runner->adapter->CopiedOutRows() - copied ==
           2 * conversation.size());
    const auto [cached, disk_hit] = LiveTurn(cache, *runner, next);
    assert(cached == conversation.size() && !disk_hit);
  }
  auto runner = std::make_shared<Runner>();
  {
    ComponentTextCache cache(runner, 1, disk);
    auto lease = cache.Acquire(next, {}, {}, true, 0, true);
    assert(lease->cached_tokens() == conversation.size() &&
           lease->restored_from_disk());
    lease->Invalidate();
  }
  std::filesystem::remove_all(directory);
}
void SingleWriterEvictsBetweenQueuedWrites() {
  // Two held write-backs queue together; the disk holds one. The writer
  // publishes the first, then evicts it for the second, with no other writer
  // contending for the store: the second survives a restart.
  const TemporaryDirectory temp("gufo-component-single-writer");
  const std::filesystem::path directory = temp.path();
  std::vector<TextRunnerToken> first(8192), second(8192);
  std::iota(first.begin(), first.end(), 10);
  std::iota(second.begin(), second.end(), 20000);
  std::size_t one{};
  {
    // Measure one frontier on an unbounded disk.
    const TemporaryDirectory probe("gufo-component-single-writer-probe");
    auto runner = std::make_shared<Runner>();
    {
      ComponentTextCache cache(
          runner, 1,
          TextRunnerDiskCacheOptions{.directory = probe.path().string(),
                                     .capacity_bytes = 1 << 24});
      runner->adapter->LimitBorrowedRows(1);
      (void)LiveTurn(cache, *runner, first);
      assert(cache.WriteBackIdle(std::chrono::milliseconds(0)) == 1);
    }
    one = ManagedBytes(probe.path());
  }
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = one * 3 / 2};
  {
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 2, disk);
    runner->adapter->LimitBorrowedRows(1);
    (void)LiveTurn(cache, *runner, first);
    (void)LiveTurn(cache, *runner, second);
    assert(cache.WriteBackIdle(std::chrono::milliseconds(0)) == 2);
  }
  auto runner = std::make_shared<Runner>();
  {
    ComponentTextCache cache(runner, 1, disk);
    auto next = second;
    next.push_back(7);
    auto lease = cache.Acquire(next, {}, {}, true, 0, true);
    assert(lease->cached_tokens() == second.size() &&
           lease->restored_from_disk());
    lease->Invalidate();
  }
  std::filesystem::remove_all(directory);
}
void RetainedRowsWriteBackWithoutStaging() {
  // Rows the adapter retains as device references (a shared KV pool) but
  // does not lend are written from those references: only the draft rows,
  // which it does not retain, are copied out.
  const TemporaryDirectory temp("gufo-component-retained-write-back");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 22};
  std::vector<TextRunnerToken> conversation(100);
  std::iota(conversation.begin(), conversation.end(), 10);
  {
    auto runner = std::make_shared<Runner>();
    runner->retain_rows = true;
    ComponentTextCache cache(runner, 2, disk);
    runner->adapter->LimitBorrowedRows(1);
    (void)LiveTurn(cache, *runner, conversation);
    const auto copied = runner->adapter->CopiedOutRows();
    assert(cache.WriteBackIdle(std::chrono::milliseconds(0)) == 1);
    assert(runner->adapter->CopiedOutRows() - copied == conversation.size());
  }
  auto runner = std::make_shared<Runner>();
  {
    ComponentTextCache cache(runner, 1, disk);
    auto next = conversation;
    next.push_back(7);
    auto lease = cache.Acquire(next, {}, {}, true, 0, true);
    assert(lease->cached_tokens() == conversation.size() &&
           lease->restored_from_disk());
    lease->Invalidate();
  }
  std::filesystem::remove_all(directory);
}
void ImageFrontierCapturesAfterPrefixCheckpoint() {
  // A conversation with an image at token 20 checkpoints at 8, before it,
  // under the identity of the images before 8. Its live frontier keeps the
  // image boundary, so the capture before its slot is reassigned continues
  // that history instead of failing ("checkpoint input changed within
  // inherited prefix"), and the next turn restores it from disk.
  auto context = std::make_shared<TextPromptContext>();
  context->cache_identity = {9, 9};
  context->cache_prefixes.push_back({20, {1}});
  std::vector<TextRunnerToken> conversation(100);
  std::iota(conversation.begin(), conversation.end(), 10);
  const TemporaryDirectory temp("gufo-component-image-frontier");
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(
      runner, 1,
      TextRunnerDiskCacheOptions{.directory = temp.path().string(),
                                 .capacity_bytes = 1 << 22,
                                 .min_checkpoint_step_tokens = 1});
  {
    auto lease = cache.Acquire(conversation, {}, context, true, 0, true);
    auto& state = dynamic_cast<State&>(lease->state());
    const auto head = std::span(conversation).first(8);
    runner->adapter->Append(state.slot, head, head);
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), 8, false,
                                     SnapshotPurpose::kBranchPoint, {}));
    (void)lease->PublishSnapshot({head.begin(), head.end()}, lease->Capture(),
                                 false);
    const auto rest = std::span(conversation).subspan(8);
    runner->adapter->Append(state.slot, rest, rest);
    (void)lease->Commit({}, {}, conversation);
  }
  assert(WaitWritten(cache));
  const auto previous_level = Logger::Level();
  Logger::SetLevel(LogLevel::kInfo);
  std::ostringstream output;
  auto* previous_sink = std::clog.rdbuf(output.rdbuf());
  (void)LiveTurn(cache, *runner, std::vector<TextRunnerToken>(12, 5));
  std::clog.rdbuf(previous_sink);
  Logger::SetLevel(previous_level);
  const auto log = output.str();
  assert(log.find("event=live_frontier_captured reason=reassignment") !=
         std::string::npos);
  assert(log.find("capture_failed") == std::string::npos);
  assert(WaitWritten(cache));
  auto next = conversation;
  next.push_back(7);
  const auto [cached, disk_hit] = LiveTurn(cache, *runner, next, context);
  assert(cached == conversation.size() && disk_hit);
}
}  // namespace
// Paged rows: admission waits while a shared row pool is short, after
// returning the rows idle slots hold; a reservation holds blocks until the
// request ends; a request no pool can hold is refused by name.
constexpr std::size_t kPagedTokens = 4;  // tokens per block
constexpr std::size_t kPagedBlocks = 7;
void PagedRowsAdmitWaitAndReclaim() {
  class PagedRunner final : public Runner {
  public:
    mutable std::vector<cache::Slot*> slots;
    mutable std::map<cache::Slot*, std::size_t> reserved;
    // Blocks a reservation needs beyond AdmissionRowBlocks' count (copies
    // of shared blocks).
    std::size_t hidden_copies = 0;
    std::unique_ptr<TextRunnerState> BindComponentState(
        cache::Adapter& adapter, cache::Slot& slot) const override {
      if (std::ranges::find(slots, &slot) == slots.end())
        slots.push_back(&slot);
      return Runner::BindComponentState(adapter, slot);
    }
    std::size_t Mapped(cache::Slot* slot) const {
      return (adapter->Positions(*slot).front().valid_rows + kPagedTokens - 1) /
             kPagedTokens;
    }
    RowBlockPool RowBlocks() const override {
      std::size_t used = 0;
      for (auto* slot : slots) {
        const auto mapped = Mapped(slot);
        const auto it = reserved.find(slot);
        used += it == reserved.end() ? mapped : std::max(mapped, it->second);
      }
      return {.total_blocks = kPagedBlocks,
              .free_blocks = kPagedBlocks - used,
              .block_rows = kPagedTokens};
    }
    void ReserveRowBlocks(TextRunnerState& state,
                          std::size_t rows) const override {
      auto* slot = &dynamic_cast<State&>(state).slot;
      const auto target =
          (rows + kPagedTokens - 1) / kPagedTokens + kRowBlockMargin;
      const auto mapped = Mapped(slot);
      const auto old = reserved.contains(slot) ? reserved.at(slot) : 0;
      const auto credit = old > mapped ? old - mapped : 0;
      if (target > mapped &&
          target - mapped + hidden_copies > RowBlocks().free_blocks + credit)
        throw std::length_error("row pool cannot hold this request");
      reserved[slot] = target;
    }
    void ReleaseRowBlocks(TextRunnerState& state) const noexcept override {
      reserved.erase(&dynamic_cast<State&>(state).slot);
    }
  };
  auto runner = std::make_shared<PagedRunner>();
  TextRunnerPool pool(runner, 2);
  const std::vector<TextRunnerToken> first(8, 3), second(12, 4), third(4, 5);
  // 8 tokens + 4 new: 3 blocks, one more for speculative rows and one for a
  // copy of a block a checkpoint holds.
  assert(pool.AdmissionRowBlocks(12) == 5);
  assert(pool.AdmissionReady(first, 4));
  auto a = pool.Acquire(first);
  a.ReserveRows(4);
  assert(runner->RowBlocks().free_blocks == 2);
  Prefill(a);
  (void)Generate(a);
  (void)a.Commit();
  // The live frontier keeps its 3 rows' blocks; the reservation is back.
  assert(runner->reserved.empty() && runner->RowBlocks().free_blocks == 4);
  const auto resets = runner->adapter->Invalidations();
  // 12 + 4 tokens need 6 blocks: the idle frontier, which this prompt does
  // not continue, gives its rows back.
  assert(pool.AdmissionReady(second, 4));
  assert(runner->adapter->Invalidations() > resets &&
         runner->RowBlocks().free_blocks == kPagedBlocks);
  auto b = pool.Acquire(second);
  b.ReserveRows(4);
  assert(runner->RowBlocks().free_blocks == 1);
  // A resident holds the rest: the next admission waits rather than fail,
  // and waiting passes do not reset the emptied slot again.
  const auto waiting_resets = runner->adapter->Invalidations();
  assert(!pool.AdmissionReady(third, 4));
  assert(!pool.AdmissionReady(third, 4));
  assert(runner->adapter->Invalidations() == waiting_resets);
  Prefill(b);
  (void)b.Commit();
  // A continuation of the idle frontier keeps it; the other slot is empty.
  auto longer = second;
  longer.push_back(6);
  assert(pool.AdmissionReady(longer, 1));
  auto c = pool.Acquire(longer);
  assert(c.cached_prompt_tokens() == second.size());
  c.Invalidate();
  // No slot is busy and nothing else frees: admit, and refuse by name.
  const std::vector<TextRunnerToken> huge(40, 9);
  assert(pool.AdmissionReady(huge, 1));
  auto d = pool.Acquire(huge);
  bool refused = false;
  try {
    d.ReserveRows(1);
  } catch (const std::length_error&) {
    refused = true;
  }
  assert(refused && runner->reserved.empty());
  d.Invalidate();
  // A continuation needs only the blocks past those its slot already maps:
  // another conversation's idle frontier keeps its rows.
  auto paged = std::make_shared<PagedRunner>();
  TextRunnerPool rows(paged, 2);
  const std::vector<TextRunnerToken> other(4, 7), talk(12, 8);
  for (const auto* tokens : {&other, &talk}) {
    auto request = rows.Acquire(*tokens);
    request.ReserveRows(1);
    Prefill(request);
    (void)request.Commit();
  }
  assert(paged->RowBlocks().free_blocks == kPagedBlocks - 4);
  auto more = talk;
  more.push_back(9);
  const auto kept = paged->adapter->Invalidations();
  // 13 + 1 tokens: 4 blocks plus the margin, 3 of them mapped already.
  assert(rows.AdmissionReady(more, 1));
  assert(paged->adapter->Invalidations() == kept);
  auto continued = rows.Acquire(more);
  assert(continued.cached_prompt_tokens() == talk.size());
  continued.ReserveRows(1);
  continued.Invalidate();

  // A live borrower's growth may reclaim another idle copy of its prefix.
  // Protecting that copy with reuse=true would prevent guarantor promotion.
  auto growing = std::make_shared<PagedRunner>();
  TextRunnerPool growth(growing, 2);
  const std::vector<TextRunnerToken> prefix{20, 21, 22, 23};
  auto leader = growth.Acquire(prefix);
  leader.ReserveRows(0);
  Prefill(leader);
  auto extended = prefix;
  extended.push_back(24);
  auto borrower = growth.Acquire(extended);
  borrower.ReserveRows(0);
  Prefill(borrower);
  leader.Commit();
  const auto invalidated = growing->adapter->Invalidations();
  {
    // Growth the pool's estimate allows but the runner cannot reserve waits,
    // keeping the old reservation, instead of failing the request.
    growing->hidden_copies = kPagedBlocks;
    const auto held = growing->reserved;
    assert(!growth.TryGrowRows(borrower, 12));
    assert(growing->reserved == held);
    growing->hidden_copies = 0;
  }
  assert(growth.TryGrowRows(borrower, 12));
  assert(growing->adapter->Invalidations() > invalidated);
  const auto held_before = growing->reserved;
  assert(!growth.TryGrowRows(borrower, 16));
  assert(growing->reserved == held_before);
  borrower.Invalidate();
}
// A copy of a live frontier restores at its own position even off the
// prefill alignment: it is the state live reuse would have continued (the
// idle disk write-back).
void UnalignedLiveFrontierCopiesRestore() {
  std::vector<TextRunnerToken> conversation(101);
  std::iota(conversation.begin(), conversation.end(), 10);
  auto next = conversation;
  next.push_back(7);
  const auto turn = [](ComponentTextCache& cache, Runner& runner,
                       const std::vector<TextRunnerToken>& tokens) {
    auto lease = cache.Acquire(tokens, {}, {}, true, 0, true);
    const auto cached = lease->cached_tokens();
    const bool disk_hit = lease->restored_from_disk();
    auto& state = dynamic_cast<State&>(lease->state());
    const auto tail = std::span(tokens).subspan(cached);
    runner.adapter->Append(state.slot, tail, tail);
    (void)lease->Commit({}, {}, tokens);
    return std::pair{cached, disk_hit};
  };
  const TemporaryDirectory temp("gufo-component-unaligned-frontier-test");
  const TextRunnerDiskCacheOptions disk{.directory = temp.path().string(),
                                        .capacity_bytes = 1 << 22};
  {
    auto runner = std::make_shared<Runner>();
    runner->alignment = 4;
    ComponentTextCache cache(runner, 2, disk);
    (void)turn(cache, *runner, conversation);
    assert(cache.WriteBackIdle(std::chrono::milliseconds(0)) == 1);
  }
  auto runner = std::make_shared<Runner>();
  runner->alignment = 4;
  ComponentTextCache cache(runner, 2, disk);
  const auto [cached, disk_hit] = turn(cache, *runner, next);
  assert(cached == conversation.size() && disk_hit);
}
void IdleWriteBackResumesFromLastTurn() {
  // A finished turn leaves only a live frontier (no checkpoint at its end).
  // The idle write-back, or else the shutdown pass of the destructor, puts
  // it on disk: the next turn after a restart restores the whole frontier,
  // although it lies within the disk step of nothing published, and the live
  // frontier stays usable meanwhile.
  std::vector<TextRunnerToken> conversation(100);
  std::iota(conversation.begin(), conversation.end(), 10);
  auto next = conversation;
  next.push_back(7);
  const auto turn = [](ComponentTextCache& cache, Runner& runner,
                       const std::vector<TextRunnerToken>& tokens) {
    auto lease = cache.Acquire(tokens, {}, {}, true, 0, true);
    const auto cached = lease->cached_tokens();
    const bool disk_hit = lease->restored_from_disk();
    auto& state = dynamic_cast<State&>(lease->state());
    const auto tail = std::span(tokens).subspan(cached);
    runner.adapter->Append(state.slot, tail, tail);
    (void)lease->Commit({}, {}, tokens);
    return std::pair{cached, disk_hit};
  };
  for (const bool write_back : {false, true}) {
    const TemporaryDirectory temp("gufo-component-write-back-test");
    const std::filesystem::path directory = temp.path();
    const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                          .capacity_bytes = 1 << 22};
    {
      auto runner = std::make_shared<Runner>();
      ComponentTextCache cache(runner, 2, disk);
      (void)turn(cache, *runner, conversation);
      if (write_back) {
        // Not idle long enough yet.
        assert(cache.WriteBackIdle(std::chrono::hours(1)) == 0);
        assert(cache.WriteBackIdle(std::chrono::milliseconds(0)) == 1);
        assert(cache.WriteBackIdle(std::chrono::milliseconds(0)) == 0);
        // Persistence runs only while no lease is held: look in from the
        // other slot between waits (a short side prompt, never written back).
        const std::vector<TextRunnerToken> side(12, 5);
        std::size_t completed = 0;
        for (int i = 0; i < 500 && completed == 0; ++i) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          auto peek = cache.Acquire(side, {}, {}, true, 0, true);
          completed = peek->Metrics().persistence->completed;
          (void)peek->Commit({}, {}, {});
        }
        assert(completed == 1);
        // The live frontier still serves the next turn without a restore.
        const auto [cached, disk_hit] = turn(cache, *runner, next);
        assert(cached == conversation.size() && !disk_hit);
        assert(cache.WriteBackIdle(std::chrono::milliseconds(0)) == 0);
      }
    }
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 2, disk);
    const auto [cached, disk_hit] = turn(cache, *runner, next);
    assert(cached == conversation.size() && disk_hit);
    std::filesystem::remove_all(directory);
  }
}
void RefusedCaptureIsKnownBeforePrefill() {
  // Every capture goes to disk: without a disk cache every boundary is
  // refused (asked before prefill reaches it, as the runner does). With one,
  // only a boundary within the disk step of a published or queued checkpoint
  // is.
  std::vector<TextRunnerToken> prompt(200);
  std::iota(prompt.begin(), prompt.end(), 10);
  {
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 1, {});
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(lease->CaptureRefused(100) && lease->CaptureRefused(prompt.size()));
    lease->Invalidate();
  }
  const TemporaryDirectory temp("gufo-component-refused-before-prefill");
  const TextRunnerDiskCacheOptions disk{.directory = temp.path().string(),
                                        .capacity_bytes = 1 << 22,
                                        .min_checkpoint_step_tokens = 64};
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 1, disk);
  auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
  auto& state = dynamic_cast<State&>(lease->state());
  const auto head = std::span(prompt).first(100);
  runner->adapter->Append(state.slot, head, head);
  assert(!lease->CaptureRefused(100) && !lease->CaptureRefused(prompt.size()));
  assert(lease->TryReserveSnapshot(lease->CaptureBytes(), head.size(), false,
                                   SnapshotPurpose::kContinuation, {}));
  (void)lease->PublishSnapshot({head.begin(), head.end()}, lease->Capture(),
                               false);
  assert(lease->Metrics().disk_queued_bytes > 0);
  // Within the step of the queued checkpoint, then past it.
  assert(lease->CaptureRefused(150));
  assert(!lease->CaptureRefused(prompt.size()));
  lease->Invalidate();
}
void WriteBackReusesPersistedChunks() {
  // The prompt boundary goes to disk from a RAM checkpoint; the write-back of
  // the longer frontier reuses its chunk files and writes only new ones.
  const TemporaryDirectory temp("gufo-component-write-back-reuse");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 22};
  const auto files = [&] {
    std::size_t n = 0;
    for (const auto& f :
         std::filesystem::recursive_directory_iterator(directory))
      n += f.is_regular_file();
    return n;
  };
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 2, disk);
  const std::vector<TextRunnerToken> side(12, 5);
  const auto wait = [&](std::uint64_t target) {
    std::uint64_t completed = 0;
    for (int i = 0; i < 500 && completed < target; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      auto peek = cache.Acquire(side, {}, {}, true, 0, true);
      completed = peek->Metrics().persistence->completed;
      (void)peek->Commit({}, {}, {});
    }
    assert(completed == target);
  };
  std::vector<TextRunnerToken> prompt(200);
  std::iota(prompt.begin(), prompt.end(), 10);
  const std::vector<TextRunnerToken> head(prompt.begin(), prompt.begin() + 100);
  {
    auto lease = cache.Acquire(head, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(lease->state());
    runner->adapter->Append(state.slot, head, head);
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), head.size(), false,
                                     SnapshotPurpose::kContinuation, {}));
    (void)lease->Commit(head, lease->Capture(), head);
  }
  wait(1);
  const auto first = files();
  {
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(lease->cached_tokens() == head.size());
    auto& state = dynamic_cast<State&>(lease->state());
    const auto tail = std::span(prompt).subspan(head.size());
    runner->adapter->Append(state.slot, tail, tail);
    (void)lease->Commit({}, {}, prompt);
  }
  assert(cache.WriteBackIdle(std::chrono::milliseconds(0)) == 1);
  wait(2);
  // Without reuse the write-back would add at least as many files as the
  // first checkpoint (twice the rows); with it, about as many again.
  assert(files() - first < first + first / 2);
  std::filesystem::remove_all(directory);
}
void ShutdownWriteBackSharesQueuedCheckpoint() {
  // A turn ends while its prompt checkpoint may still be queued; the server
  // stops at once. The destructor drains the queue first, so the frontier's
  // write-back reuses that checkpoint's chunk files, and a restart resumes at
  // the frontier.
  std::vector<TextRunnerToken> prompt(200);
  std::iota(prompt.begin(), prompt.end(), 10);
  const std::vector<TextRunnerToken> head(prompt.begin(), prompt.begin() + 100);
  const auto run = [&](bool extend) {
    const TemporaryDirectory temp("gufo-component-shutdown-write-back");
    const std::filesystem::path directory = temp.path();
    const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                          .capacity_bytes = 1 << 22};
    {
      auto runner = std::make_shared<Runner>();
      ComponentTextCache cache(runner, 2, disk);
      {
        auto lease = cache.Acquire(head, {}, {}, true, 0, true);
        auto& state = dynamic_cast<State&>(lease->state());
        runner->adapter->Append(state.slot, head, head);
        assert(lease->TryReserveSnapshot(lease->CaptureBytes(), head.size(),
                                         false, SnapshotPurpose::kContinuation,
                                         {}));
        (void)lease->Commit(head, lease->Capture(), head);
      }
      if (extend) {
        auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
        assert(lease->cached_tokens() == head.size());
        auto& state = dynamic_cast<State&>(lease->state());
        const auto tail = std::span(prompt).subspan(head.size());
        runner->adapter->Append(state.slot, tail, tail);
        (void)lease->Commit({}, {}, prompt);
      }
    }
    std::size_t files = 0;
    for (const auto& f :
         std::filesystem::recursive_directory_iterator(directory))
      files += f.is_regular_file();
    auto runner = std::make_shared<Runner>();
    std::size_t cached;
    bool disk_hit;
    {
      ComponentTextCache cache(runner, 2, disk);
      auto next = extend ? prompt : head;
      next.push_back(7);
      auto lease = cache.Acquire(next, {}, {}, true, 0, true);
      cached = lease->cached_tokens();
      disk_hit = lease->restored_from_disk();
      lease->Invalidate();
    }
    std::filesystem::remove_all(directory);
    return std::tuple{files, cached, disk_hit};
  };
  const auto [first, head_cached, head_hit] = run(false);
  assert(head_cached == head.size() && head_hit);
  const auto [files, cached, disk_hit] = run(true);
  assert(cached == prompt.size() && disk_hit);
  // Twice the rows: about twice the files with reuse, three times without.
  assert(files < first * 2 + first / 2);
}
void ReassignmentWritesUncoveredFrontier() {
  // The prompt boundary (100) is on disk; the next turn's 200-token frontier
  // lies within the disk step of it. When the only slot goes to another
  // prompt, the frontier captured there goes to disk too, so a restart
  // resumes at 200 rather than at the prompt boundary.
  const TemporaryDirectory temp("gufo-component-reassignment-write");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 22};
  std::vector<TextRunnerToken> conversation(200);
  std::iota(conversation.begin(), conversation.end(), 10);
  const std::vector<TextRunnerToken> head(conversation.begin(),
                                          conversation.begin() + 100);
  const std::vector<TextRunnerToken> side(12, 5);
  {
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 1, disk);
    const auto run = [&](const std::vector<TextRunnerToken>& tokens) {
      auto lease = cache.Acquire(tokens, {}, {}, true, 0, true);
      const auto cached = lease->cached_tokens();
      const auto completed = lease->Metrics().persistence->completed;
      auto& state = dynamic_cast<State&>(lease->state());
      const auto tail = std::span(tokens).subspan(cached);
      runner->adapter->Append(state.slot, tail, tail);
      (void)lease->Commit({}, {}, tokens);
      return std::pair{cached, completed};
    };
    {
      auto lease = cache.Acquire(head, {}, {}, true, 0, true);
      auto& state = dynamic_cast<State&>(lease->state());
      runner->adapter->Append(state.slot, head, head);
      assert(lease->TryReserveSnapshot(lease->CaptureBytes(), head.size(),
                                       false, SnapshotPurpose::kContinuation,
                                       {}));
      (void)lease->Commit(head, lease->Capture(), head);
    }
    std::uint64_t completed = 0;
    for (int i = 0; i < 500 && completed == 0; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      completed = run(side).second;
    }
    assert(completed == 1);
    assert(run(conversation).first == head.size());
    (void)run(side);
  }
  auto runner = std::make_shared<Runner>();
  {
    ComponentTextCache cache(runner, 1, disk);
    auto next = conversation;
    next.push_back(7);
    auto lease = cache.Acquire(next, {}, {}, true, 0, true);
    assert(lease->cached_tokens() == conversation.size() &&
           lease->restored_from_disk());
    lease->Invalidate();
  }
  std::filesystem::remove_all(directory);
}
void SharedLineageCoversOnlyItsOwnFrontier() {
  // Conversations A and B fork from one prompt checkpoint and share its
  // lineage. A's 200-token frontier goes to disk when its slot is reassigned;
  // that checkpoint must not count as covering B's equally long frontier, so
  // shutdown still writes B back and a restart resumes B at 200.
  const TemporaryDirectory temp("gufo-component-shared-lineage");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 22};
  std::vector<TextRunnerToken> a(200), b(200);
  std::iota(a.begin(), a.end(), 10);
  std::iota(b.begin(), b.end(), 10);
  std::iota(b.begin() + 100, b.end(), 5000);
  const std::vector<TextRunnerToken> head(a.begin(), a.begin() + 100);
  {
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 2, disk);
    const auto run = [&](const std::vector<TextRunnerToken>& tokens) {
      auto lease = cache.Acquire(tokens, {}, {}, true, 0, true);
      const auto cached = lease->cached_tokens();
      auto& state = dynamic_cast<State&>(lease->state());
      const auto tail = std::span(tokens).subspan(cached);
      runner->adapter->Append(state.slot, tail, tail);
      (void)lease->Commit({}, {}, tokens);
      return cached;
    };
    {
      auto lease = cache.Acquire(head, {}, {}, true, 0, true);
      auto& state = dynamic_cast<State&>(lease->state());
      runner->adapter->Append(state.slot, head, head);
      assert(lease->TryReserveSnapshot(lease->CaptureBytes(), head.size(),
                                       false, SnapshotPurpose::kContinuation,
                                       {}));
      (void)lease->Commit(head, lease->Capture(), head);
    }
    assert(WaitWritten(cache));
    assert(run(a) == head.size());
    assert(run(b) == head.size());
    // A third prompt takes A's slot (the older one).
    (void)run(std::vector<TextRunnerToken>(12, 5));
  }
  auto runner = std::make_shared<Runner>();
  {
    ComponentTextCache cache(runner, 2, disk);
    auto next = b;
    next.push_back(7);
    auto lease = cache.Acquire(next, {}, {}, true, 0, true);
    assert(lease->cached_tokens() == b.size() && lease->restored_from_disk());
    lease->Invalidate();
  }
  std::filesystem::remove_all(directory);
}
// The prompt's checkpoint, captured and committed; returns its metrics.
ComponentCacheMetrics CapturePrompt(
    ComponentTextCache& cache, Runner& runner,
    const std::vector<TextRunnerToken>& prompt) {
  auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
  auto& state = dynamic_cast<State&>(lease->state());
  runner.adapter->Append(state.slot, prompt, prompt);
  assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(), false,
                                   SnapshotPurpose::kContinuation, {}));
  (void)lease->Commit(prompt, lease->Capture(), prompt);
  return lease->Metrics();
}
void RetainedChunksReachDisk() {
  // A disk job reads retained chunks from their storage; a restart restores
  // the prompt from disk.
  const TemporaryDirectory temp("gufo-component-retained-disk");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 22};
  std::vector<TextRunnerToken> prompt(120);
  std::iota(prompt.begin(), prompt.end(), 10);
  const std::vector<TextRunnerToken> side(12, 5);
  {
    auto runner = std::make_shared<Runner>();
    runner->retain_rows = true;
    ComponentTextCache cache(runner, 1, disk);
    assert(CapturePrompt(cache, *runner, prompt).device_payload_bytes ==
           120 * sizeof(cache::Token));
    std::uint64_t completed = 0;
    for (int i = 0; i < 500 && completed == 0; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      auto lease = cache.Acquire(side, {}, {}, true, 0, true);
      completed = lease->Metrics().persistence->completed;
      lease->Invalidate();
    }
    assert(completed >= 1);
  }
  auto runner = std::make_shared<Runner>();
  {
    ComponentTextCache cache(runner, 1, disk);
    auto next = prompt;
    next.push_back(7);
    auto lease = cache.Acquire(next, {}, {}, true, 0, true);
    assert(lease->cached_tokens() == prompt.size() &&
           lease->restored_from_disk());
    auto& state = dynamic_cast<State&>(lease->state());
    assert(std::ranges::equal(runner->adapter->TargetRows(state.slot), prompt));
    lease->Invalidate();
  }
  std::filesystem::remove_all(directory);
}
void RowReclaimWaitsForPendingDiskWrites() {
  // With nothing left to reset (the only slot holds the live frontier the
  // request continues), a pending disk write returns rows only when it reads
  // device rows kept for it (retained chunks of the checkpoint it writes):
  // then admission waits for it instead of admitting a request the rows
  // cannot hold. A write of RAM copies returns no rows: no wait.
  for (const bool retain : {true, false}) {
    const TemporaryDirectory temp("gufo-component-reclaim-writes");
    const TextRunnerDiskCacheOptions disk{.directory = temp.path().string(),
                                          .capacity_bytes = 1 << 22,
                                          .min_checkpoint_step_tokens = 0};
    std::vector<TextRunnerToken> prompt(40);
    std::iota(prompt.begin(), prompt.end(), 10);
    auto runner = std::make_shared<Runner>();
    runner->retain_rows = retain;
    ComponentTextCache cache(runner, 1, disk);
    WriterGate gate;
    runner->background_stream = gate.Hook();
    (void)CapturePrompt(cache, *runner, prompt);
    assert(gate.entered.try_acquire_for(std::chrono::seconds(10)));
    assert(cache.HostWritesPending());
    auto next = prompt;
    next.push_back(7);
    assert(cache.ReclaimRows(next, true, [] { return false; }) ==
           (retain ? ComponentTextCache::RowReclaim::kWaiting
                   : ComponentTextCache::RowReclaim::kExhausted));
    gate.release.release();
    for (int i = 0; i < 1000 && cache.HostWritesPending(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    assert(!cache.HostWritesPending());
    assert(cache.ReclaimRows(next, true, [] { return false; }) ==
           ComponentTextCache::RowReclaim::kExhausted);
  }
}
void RowReclaimHeldWriteProgressesUnderLoad() {
  // Retained device rows, a frontier no checkpoint covers and no room to
  // copy it to host memory: reclaim hands the slot to a held disk write. The
  // other slot stays busy throughout, so the model is never idle (as with
  // decoders parked waiting for these rows). The writer yields to the model
  // only for a bounded time: the write completes, the slot returns with its
  // frontier covered, and a later reclaim resets it.
  const TemporaryDirectory temp("gufo-component-reclaim-held");
  const TextRunnerDiskCacheOptions disk{.directory = temp.path().string(),
                                        .capacity_bytes = 1 << 22};
  std::vector<TextRunnerToken> other(100);
  std::iota(other.begin(), other.end(), 3000);
  const std::vector<TextRunnerToken> side(12, 5), next(12, 6);
  const auto previous_level = Logger::Level();
  Logger::SetLevel(LogLevel::kInfo);
  std::ostringstream output;
  auto* previous_sink = std::clog.rdbuf(output.rdbuf());
  {
    auto runner = std::make_shared<Runner>();
    runner->retain_rows = true;
    runner->snapshot_bytes = 0;
    ComponentTextCache cache(runner, 2, disk);
    (void)LiveTurn(cache, *runner, other);
    auto running = cache.Acquire(side, {}, {}, true, 0, true);
    const auto resets = runner->adapter->Invalidations();
    const auto enough = [&] {
      return runner->adapter->Invalidations() > resets;
    };
    assert(cache.ReclaimRows(next, true, enough) ==
           ComponentTextCache::RowReclaim::kWaiting);
    auto result = ComponentTextCache::RowReclaim::kWaiting;
    for (int i = 0;
         i < 1000 && result != ComponentTextCache::RowReclaim::kReady; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      result = cache.ReclaimRows(next, true, enough);
    }
    assert(result == ComponentTextCache::RowReclaim::kReady);
    running->Invalidate();
  }
  std::clog.rdbuf(previous_sink);
  Logger::SetLevel(previous_level);
  const auto log = output.str();
  assert(log.find("event=live_frontier_captured reason=row_capacity slot=1 "
                  "tokens=100") != std::string::npos);
  assert(log.find("persistence_incomplete") == std::string::npos);
}
void RowReclaimSkipsSlotsQueuedWritesRead() {
  // A disk job reads rows a live slot lent it (observed rows) and is stopped
  // mid-write. Reclaim must not reset that slot, which would wait for the
  // write on the scheduler thread or fail it, but report waiting; once the
  // job has ended the slot is reset and the checkpoint has published.
  const TemporaryDirectory temp("gufo-component-reclaim-observed");
  const TextRunnerDiskCacheOptions disk{.directory = temp.path().string(),
                                        .capacity_bytes = 1 << 22,
                                        .min_checkpoint_step_tokens = 1};
  std::vector<TextRunnerToken> prompt(6144);
  std::iota(prompt.begin(), prompt.end(), 10);
  const std::vector<TextRunnerToken> next(12, 5);
  const auto previous_level = Logger::Level();
  Logger::SetLevel(LogLevel::kInfo);
  std::ostringstream output;
  auto* previous_sink = std::clog.rdbuf(output.rdbuf());
  {
    auto runner = std::make_shared<Runner>();
    runner->rows_per_chunk = 1024;
    runner->stable_appends = true;
    ComponentTextCache cache(runner, 1, disk);
    std::atomic<int> reads{};
    std::binary_semaphore entered{0}, release{0};
    runner->background_stream = [&] {
      if (++reads == 3) {
        entered.release();
        release.acquire();
      }
    };
    {
      auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
      auto& state = dynamic_cast<State&>(lease->state());
      runner->adapter->Append(state.slot, prompt, prompt);
      assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(),
                                       false, SnapshotPurpose::kContinuation,
                                       prompt));
      (void)lease->PublishSnapshot(prompt, lease->Capture(), false);
      assert(lease->Metrics().disk_queued_bytes > 0);
      assert(entered.try_acquire_for(std::chrono::seconds(10)));
      (void)lease->Commit({}, {}, prompt);
    }
    const auto resets = runner->adapter->Invalidations();
    const auto enough = [&] {
      return runner->adapter->Invalidations() > resets;
    };
    assert(cache.ReclaimRows(next, true, enough) ==
           ComponentTextCache::RowReclaim::kWaiting);
    assert(runner->adapter->Invalidations() == resets);
    release.release();
    auto result = ComponentTextCache::RowReclaim::kWaiting;
    for (int i = 0;
         i < 1000 && result != ComponentTextCache::RowReclaim::kReady; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      result = cache.ReclaimRows(next, true, enough);
    }
    assert(result == ComponentTextCache::RowReclaim::kReady);
  }
  std::clog.rdbuf(previous_sink);
  Logger::SetLevel(previous_level);
  const auto log = output.str();
  assert(log.find("event=published tokens=6144") != std::string::npos);
  assert(log.find("persistence_incomplete") == std::string::npos);
}
void FirstFollowerRestoresTheSystemPromptCheckpoint() {
  // The leader keeps a checkpoint where its leading system messages end, so
  // the first follower under the same system prompt restores exactly that
  // point, also when checkpoints copy their rows. Without it the follower
  // starts cold (no tail checkpoints here).
  std::vector<TextRunnerToken> leader_prompt(2100);
  std::iota(leader_prompt.begin(), leader_prompt.end(), 10);
  auto follower_prompt = leader_prompt;
  std::iota(follower_prompt.begin() + 1500, follower_prompt.end(), 20000);
  TextRunnerToken expected{};
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1);
    auto cold = pool.Acquire(follower_prompt, {}, {}, {}, false);
    Prefill(cold);
    expected = Generate(cold);
    cold.Invalidate();
  }
  // A system prompt under the shared-prefix minimum (128) is not worth a
  // checkpoint. The follower restores it from disk.
  for (const std::size_t system : {0, 100, 1500})
    for (const bool retain : {false, true}) {
      const TemporaryDirectory temp("gufo-component-system-prompt");
      auto runner = std::make_shared<Runner>();
      runner->retain_rows = retain;
      TextRunnerPool pool(
          runner, 1,
          TextRunnerDiskCacheOptions{.directory = temp.path().string(),
                                     .capacity_bytes = 1 << 24});
      auto leader =
          pool.Acquire(leader_prompt, {}, {}, {}, true, 0, true, system);
      Prefill(leader);
      (void)Generate(leader);
      (void)leader.Commit();
      if (system >= 128)
        assert(WaitCached(pool, follower_prompt, system));
      auto follower =
          pool.Acquire(follower_prompt, {}, {}, {}, true, 0, true, system);
      assert(follower.cached_prompt_tokens() == (system >= 128 ? system : 0));
      Prefill(follower);
      assert(Generate(follower) == expected);
      follower.Invalidate();
    }
}
void FailedWriteBackKeepsTheLiveFrontier() {
  // An idle write-back that fails leaves the live frontier in its slot for
  // the next turn, and does not count it as covered: a later attempt waits
  // until it grows (idle passes) but shutdown tries it again.
  const TemporaryDirectory temp("gufo-component-write-back-failure");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 22};
  std::vector<TextRunnerToken> conversation(100);
  std::iota(conversation.begin(), conversation.end(), 10);
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 2, disk);
  {
    auto lease = cache.Acquire(conversation, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(lease->state());
    runner->adapter->Append(state.slot, conversation, conversation);
    (void)lease->Commit({}, {}, conversation);
  }
  const auto invalidations = runner->adapter->Invalidations();
  runner->adapter->FailNextTransfer();
  assert(cache.WriteBackIdle(std::chrono::milliseconds(0)) == 0);
  assert(runner->adapter->Invalidations() == invalidations);
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  assert(cache.WriteBackIdle(std::chrono::milliseconds(1)) == 0);
  // Shutdown ignores the idle backoff.
  assert(cache.WriteBackIdle(std::chrono::milliseconds(0)) == 1);
  auto next = conversation;
  next.push_back(7);
  {
    auto lease = cache.Acquire(next, {}, {}, true, 0, true);
    assert(lease->cached_tokens() == conversation.size() &&
           lease->Metrics().reuse == "live");
    lease->Invalidate();
  }
  std::filesystem::remove_all(directory);
}
void ScatteredRowsWriteBackChunkByChunk() {
  // Completed rows the adapter cannot lend as one view (rows spread over
  // device blocks) are lent chunk by chunk, not staged in RAM, and restore
  // after a restart.
  const TemporaryDirectory temp("gufo-component-scattered-write-back");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 22};
  std::vector<TextRunnerToken> conversation(100);
  std::iota(conversation.begin(), conversation.end(), 10);
  {
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 1, disk);
    runner->adapter->LimitBorrowedRows(runner->rows_per_chunk);
    {
      auto lease = cache.Acquire(conversation, {}, {}, true, 0, true);
      auto& state = dynamic_cast<State&>(lease->state());
      runner->adapter->Append(state.slot, conversation, conversation);
      (void)lease->Commit({}, {}, conversation);
    }
    const auto copied = runner->adapter->CopiedOutRows();
    assert(cache.WriteBackIdle(std::chrono::milliseconds(0)) == 1);
    assert(runner->adapter->CopiedOutRows() == copied);
  }
  auto runner = std::make_shared<Runner>();
  {
    ComponentTextCache cache(runner, 1, disk);
    auto next = conversation;
    next.push_back(7);
    auto lease = cache.Acquire(next, {}, {}, true, 0, true);
    assert(lease->cached_tokens() == conversation.size() &&
           lease->restored_from_disk());
    lease->Invalidate();
  }
  std::filesystem::remove_all(directory);
}
void QueuedWriteBackFailureUncoversTheFrontier() {
  // A write-back that is queued and then fails no longer covers the
  // frontier, however soon it fails: shutdown writes it again.
  const TemporaryDirectory temp("gufo-component-write-back-late-failure");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1};
  std::vector<TextRunnerToken> conversation(100);
  std::iota(conversation.begin(), conversation.end(), 10);
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 1, disk);
  {
    auto lease = cache.Acquire(conversation, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(lease->state());
    runner->adapter->Append(state.slot, conversation, conversation);
    (void)lease->Commit({}, {}, conversation);
  }
  assert(cache.WriteBackIdle(std::chrono::milliseconds(0)) == 1);
  bool retried = false;
  for (int i = 0; i < 200 && !retried; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    retried = cache.WriteBackIdle(std::chrono::milliseconds(0)) == 1;
  }
  assert(retried);
  std::filesystem::remove_all(directory);
}
void RowReclaimWritesRetainedFrontiersToDisk() {
  // Retained device rows (a shared KV pool): an uncovered live frontier reset
  // for rows is not kept as a RAM checkpoint, which would hold the rows the
  // reset returns. With a disk cache it is copied to host memory and written
  // to disk alone, and restores byte for byte after a restart.
  const TemporaryDirectory temp("gufo-component-row-reclaim-disk");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 22};
  std::vector<TextRunnerToken> other(100);
  std::iota(other.begin(), other.end(), 3000);
  std::uint64_t hash{};
  const auto previous_level = Logger::Level();
  Logger::SetLevel(LogLevel::kInfo);
  std::ostringstream output;
  auto* previous_sink = std::clog.rdbuf(output.rdbuf());
  {
    auto runner = std::make_shared<Runner>();
    runner->retain_rows = true;
    ComponentTextCache cache(runner, 1, disk);
    {
      auto lease = cache.Acquire(other, {}, {}, true, 0, true);
      auto& state = dynamic_cast<State&>(lease->state());
      runner->adapter->Append(state.slot, other, other);
      hash = runner->adapter->RecurrentHash(state.slot);
      (void)lease->Commit({}, {}, other);
    }
    const auto chunks = runner->adapter->RetainedChunks();
    const auto resets = runner->adapter->Invalidations();
    const std::vector<TextRunnerToken> next(12, 5);
    assert(cache.ReclaimRows(next, true, [&] {
      return runner->adapter->Invalidations() > resets;
    }) == ComponentTextCache::RowReclaim::kReady);
    assert(runner->adapter->RetainedChunks() == chunks);
  }
  std::clog.rdbuf(previous_sink);
  Logger::SetLevel(previous_level);
  const auto log = output.str();
  assert(log.find("event=live_frontier_captured reason=row_capacity") !=
         std::string::npos);
  assert(log.find("event=slot_snapshot slot=1 tokens=100") !=
         std::string::npos);
  auto runner = std::make_shared<Runner>();
  runner->retain_rows = true;
  ComponentTextCache cache(runner, 1, disk);
  auto next = other;
  next.push_back(7);
  auto lease = cache.Acquire(next, {}, {}, true, 0, true);
  assert(lease->cached_tokens() == other.size() && lease->restored_from_disk());
  const auto& slot = dynamic_cast<State&>(lease->state()).slot;
  assert(std::ranges::equal(runner->adapter->TargetRows(slot), other));
  assert(runner->adapter->RecurrentHash(slot) == hash);
  lease->Invalidate();
}
void SiblingDoesNotAdoptAnotherExecutionsCheckpoint() {
  // A and B fork from one prompt checkpoint (one lineage) and prefill the
  // same 60 tokens past it, each in its own slot. A's checkpoint at 160
  // reaches disk; B computed its own rows, so B's write-back writes them
  // rather than reusing A's chunk files.
  const TemporaryDirectory temp("gufo-component-sibling-chunks");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 22,
                                        .min_checkpoint_step_tokens = 0};
  const auto files = [&] {
    std::size_t n = 0;
    for (const auto& f :
         std::filesystem::recursive_directory_iterator(directory))
      n += f.is_regular_file();
    return n;
  };
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 3, disk);
  const std::vector<TextRunnerToken> side(12, 5);
  const auto wait = [&](std::uint64_t target) {
    std::uint64_t completed = 0;
    for (int i = 0; i < 500 && completed < target; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      auto peek = cache.Acquire(side, {}, {}, true, 0, true);
      completed = peek->Metrics().persistence->completed;
      (void)peek->Commit({}, {}, {});
    }
    assert(completed == target);
  };
  std::vector<TextRunnerToken> b(200);
  std::iota(b.begin(), b.end(), 10);
  const std::vector<TextRunnerToken> head(b.begin(), b.begin() + 100),
      a(b.begin(), b.begin() + 160);
  (void)CapturePrompt(cache, *runner, head);
  wait(1);
  const auto before = files();
  auto first = cache.Acquire(a, {}, {}, true, 0, true);
  assert(first->cached_tokens() == head.size());
  auto second = cache.Acquire(b, {}, {}, true, 0, true);
  assert(second->cached_tokens() == head.size());
  {
    auto& state = dynamic_cast<State&>(first->state());
    const auto tail = std::span(a).subspan(head.size());
    runner->adapter->Append(state.slot, tail, tail);
    assert(first->TryReserveSnapshot(first->CaptureBytes(), a.size(), false,
                                     SnapshotPurpose::kContinuation, {}));
    (void)first->Commit(a, first->Capture(), a);
  }
  {
    auto& state = dynamic_cast<State&>(second->state());
    const auto tail = std::span(b).subspan(head.size());
    runner->adapter->Append(state.slot, tail, tail);
    (void)second->Commit({}, {}, b);
  }
  wait(2);
  const auto sibling = files() - before;
  assert(cache.WriteBackIdle(std::chrono::milliseconds(0)) >= 1);
  wait(3);
  // B's rows past the shared prompt are its own: more files than A's 60.
  assert(files() - before - sibling > sibling);
  std::filesystem::remove_all(directory);
}
// A test case named by its call.
#define CASE(call) \
  std::pair<std::string, std::function<void()>>(#call, [] { call; })
int main(int argc, char** argv) {
  // Optional arguments select tests by name prefix.
  const std::vector<std::pair<std::string, std::function<void()>>> tests{
      CASE(DiskOnlyRoundTripIsByteIdentical()),
      CASE(LongPublicationRunKeepsTheLedgerBounded()),
      CASE(CommitFailureLeavesTheRequestUncached()),
      CASE(FailedWriteBackKeepsTheLiveFrontier()),
      CASE(QueuedWriteBackFailureUncoversTheFrontier()),
      CASE(ScatteredRowsWriteBackChunkByChunk()),
      CASE(RowReclaimWritesRetainedFrontiersToDisk()),
      CASE(RowReclaimWaitsForPendingDiskWrites()),
      CASE(RowReclaimHeldWriteProgressesUnderLoad()),
      CASE(RowReclaimSkipsSlotsQueuedWritesRead()),
      CASE(SiblingDoesNotAdoptAnotherExecutionsCheckpoint()),
      CASE(FirstFollowerRestoresTheSystemPromptCheckpoint()),
      CASE(RetainedChunksReachDisk()),
      CASE(SharedLineageCoversOnlyItsOwnFrontier()),
      CASE(ReassignmentWritesUncoveredFrontier()),
      CASE(ShutdownWriteBackSharesQueuedCheckpoint()),
      CASE(IdleWriteBackResumesFromLastTurn()),
      CASE(UnalignedLiveFrontierCopiesRestore()),
      CASE(RefusedCaptureIsKnownBeforePrefill()),
      CASE(WriteBackReusesPersistedChunks()),
      CASE(PagedRowsAdmitWaitAndReclaim()),
      CASE(PublishedCheckpointKeepsDiskStep()),
      CASE(LearnedBoundaryLeavesTheNextStep()),
      CASE(CancelledPrefillKeepsCompletedSteps()),
      CASE(PartialFrontierDoesNotServeAnEqualPrompt()),
      CASE(PartialReuseRequestsRunTheirLastPromptToken()),
      CASE(InPassCapturePreservesBoundaryAndLiveTail()),
      CASE(ColdAcquireResetsOnce()),
      CASE(ColdAssignmentAvoidsDrainingCancelledSlot()),
      CASE(ColdAssignmentKeepsLiveFrontiers()),
      CASE(RestartAndCorruption()),
      CASE(DiskBudgetEvictsAnUnpinnedConversation()),
      CASE(LoweredDiskBudgetEvictsOldestAtStartup()),
      CASE(UnpersistedRequestStaysOffDisk()),
      CASE(UnpersistedLiveFrontierStaysOffDisk()),
      CASE(UnpersistedCaptureIsRefused()),
      CASE(OrphansLeaveBeforeCheckpoints()),
      CASE(StagingIsInternal()),
      CASE(DiskDirectoryHasOneOwner()),
      CASE(NearTailDiskBranchSurvivesRestart()),
      CASE(UnbackedDeepRowsSurviveDiskRestart()),
      CASE(HeldReassignmentWritesOnTheWriter()),
      CASE(ReassignmentSnapshotReleasesTheSlotBeforeTheWrite()),
      CASE(SnapshotHostsRowsQueuedJobsObserve()),
      CASE(SnapshotArenasArePooledAndReused()),
      CASE(SnapshotPoolCountsEvictsAndTrims()),
      CASE(FailedHeldWriteIsNotRetriedForever()),
      CASE(HeldTryIsPerExecution()),
      CASE(SchedulerAdmitsOthersWhileASlotIsHeld()),
      CASE(UnlendableWriteBackStreamsFromTheSlot()),
      CASE(SingleWriterEvictsBetweenQueuedWrites()),
      CASE(RetainedRowsWriteBackWithoutStaging()),
      CASE(ImageFrontierCapturesAfterPrefixCheckpoint()),
  };
  for (const auto& [name, test] : tests) {
    if (argc > 1 && std::none_of(argv + 1, argv + argc, [&](const char* arg) {
          return name.starts_with(arg);
        }))
      continue;
    std::cerr << "test " << name << '\n';
    test();
  }
}
