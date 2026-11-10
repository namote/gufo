#include "src/models/qwen38_flash_next/continuation_adapter.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cache/slot.hpp"
#include "src/core/hip/kv_block_pool.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "tests/models/qwen38_flash_next/mtp_replay.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace cache = gufo::cache;
namespace sampling = gufo::sampling;
void FailNextResetSubmission();
void FailNextTransferWait();
namespace {
void Require(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
}
template<class F>
void Reject(F&& call) {
  bool rejected = false;
  try {
    call();
  } catch (const std::exception&) {
    rejected = true;
  }
  Require(rejected, "invalid adapter operation was accepted");
}
struct IncompatibleStream final : cache::Stream {
  cache::TransferResult Synchronize() noexcept override {
    return cache::TransferResult::kSucceeded;
  }
};
struct Finished final : cache::CompletionSignal {
  bool Ready() const noexcept override { return true; }
  cache::TransferResult Wait() noexcept override {
    return cache::TransferResult::kSucceeded;
  }
};
struct Checkpoint {
  std::vector<cache::ComponentPosition> positions;
  std::vector<std::vector<std::byte>> bytes;
};
void Wait(cache::Completion done) {
  Require(done.Wait() == cache::TransferResult::kSucceeded, "transfer failed");
}
Checkpoint Capture(qfn::ContinuationAdapter& adapter, cache::Slot& slot,
                   gufo::hip::TransferPool& streams, bool registered = false) {
  Checkpoint result{adapter.Positions(slot), {}};
  constexpr std::size_t piece = (1 << 20) + 13;
  for (const auto& component : adapter.Components()) {
    const auto rows = result.positions[component.id.value - 1].valid_rows;
    auto& bytes = result.bytes.emplace_back(component.row_bytes
                                                ? rows * component.row_bytes
                                                : component.state_bytes);
    void* mapped{};
    if (registered && !bytes.empty())
      Require(hipHostMalloc(&mapped, bytes.size(), hipHostMallocDefault) ==
                  hipSuccess,
              "registered capture allocation failed");
    const std::unique_ptr<void, decltype(&hipHostFree)> mapping(mapped,
                                                                hipHostFree);
    const auto buffer =
        mapped ? std::span(static_cast<std::byte*>(mapped), bytes.size())
               : std::span(bytes);
    if (component.row_bytes) {
      for (cache::Rows first = 0; first < rows;) {
        const auto end = std::min(rows, first + 97);
        auto stream = streams.TryAcquire();
        Require(bool(stream), "stream exhausted");
        Wait(adapter.CopyRowsOut(
            slot, component.id, first, end,
            buffer.subspan(first * component.row_bytes,
                           (end - first) * component.row_bytes),
            *stream));
        first = end;
      }
    } else {
      for (std::size_t offset = 0; offset < bytes.size();) {
        const auto count = std::min(piece, bytes.size() - offset);
        auto stream = streams.TryAcquire();
        Require(bool(stream), "stream exhausted");
        Wait(adapter.CapturePrivatePiece(slot, component.id, offset,
                                         buffer.subspan(offset, count),
                                         *stream));
        offset += count;
      }
    }
    if (mapped)
      std::memcpy(bytes.data(), mapped, bytes.size());
  }
  return result;
}
void Load(qfn::ContinuationAdapter& adapter, cache::Slot& slot,
          const Checkpoint& checkpoint, gufo::hip::TransferPool& streams,
          bool omit_last = false,
          const std::vector<cache::RowPin>* native_rows = nullptr) {
  adapter.BeginRestore(slot, checkpoint.positions);
  Require(!slot.IsValid(), "BeginRestore exposed executable state");
  Reject([&] { (void)adapter.GetSession(slot); });
  // Reverse component and piece order: metadata arrives after the device
  // regions, and the first piece of each tensor arrives last.
  constexpr std::size_t piece = (1 << 20) + 13;
  std::vector<cache::Completion> pending;
  auto settle = [&] {
    for (auto& completion : pending)
      Wait(std::move(completion));
    pending.clear();
  };
  auto acquire = [&] {
    auto stream = streams.TryAcquire();
    if (!stream) {
      settle();
      stream = streams.TryAcquire();
    }
    Require(bool(stream), "stream exhausted");
    return stream;
  };
  for (std::size_t i = adapter.Components().size(); i-- > 0;) {
    const auto& component = adapter.Components()[i];
    const auto bytes = native_rows && !(*native_rows)[i].Bytes().empty()
                           ? (*native_rows)[i].Bytes()
                           : std::span<const std::byte>(checkpoint.bytes[i]);
    if (component.row_bytes) {
      const auto rows = checkpoint.positions[i].valid_rows;
      for (cache::Rows end = rows; end;) {
        const auto first = end > 97 ? end - 97 : 0;
        auto stream = acquire();
        pending.push_back(adapter.CopyRowsIn(
            slot, component.id, first, end,
            std::span(bytes).subspan(first * component.row_bytes,
                                     (end - first) * component.row_bytes),
            *stream));
        end = first;
      }
    } else {
      for (std::size_t end = bytes.size(); end;) {
        const auto first = end > piece ? end - piece : 0;
        auto stream = acquire();
        auto count = end - first;
        if (omit_last && component.id.value == 2 && !first)
          --count;
        pending.push_back(adapter.LoadPrivatePiece(
            slot, component.id, first, std::span(bytes).subspan(first, count),
            *stream));
        end = first;
      }
    }
  }
  settle();
}
struct Guard final : cache::MutationGuard {
  qfn::ContinuationAdapter* adapter{};
  cache::Slot* slot{};
  gufo::hip::TransferPool* streams{};
  const Checkpoint* borrowed{};
  std::vector<std::vector<std::byte>> preserved;
  std::vector<bool> detached;
  std::size_t overwrites{}, releases{}, resets{};
  bool release_failed{false};
  void Borrow(const Checkpoint& checkpoint) {
    borrowed = &checkpoint;
    preserved.resize(checkpoint.bytes.size());
    detached.assign(checkpoint.bytes.size(), false);
  }
  void BeforeOverwrite(cache::ComponentId id, cache::Rows first,
                       cache::Rows end) override {
    ++overwrites;
    if (!borrowed || detached[id.value - 1] || !end ||
        first >= borrowed->positions[id.value - 1].valid_rows)
      return;
    const auto& component = adapter->Components()[id.value - 1];
    if (!component.row_bytes)
      return;
    const auto rows = borrowed->positions[id.value - 1].valid_rows;
    auto& bytes = preserved[id.value - 1];
    bytes.resize(rows * component.row_bytes);
    auto stream = streams->TryAcquire();
    Require(bool(stream), "guard stream exhausted");
    Wait(adapter->CopyRowsOut(*slot, id, 0, rows, bytes, *stream));
    Require(bytes == borrowed->bytes[id.value - 1],
            "borrowed rows overwritten before preservation");
    detached[id.value - 1] = true;
  }
  void BeforeRelease(cache::ComponentId id, cache::Rows first,
                     cache::Rows end) noexcept override {
    ++releases;
    try {
      BeforeOverwrite(id, first, end);
    } catch (...) {
      release_failed = true;
    }
  }
  void AfterReset() noexcept override { ++resets; }
};
std::vector<std::int32_t> Decode(qfn::Session& session, std::size_t count,
                                 bool sampled = false) {
  const auto history = session.Tokens();
  std::vector<sampling::TokenId> initial(history.begin(), history.end());
  sampling::SamplerState sampler({.temperature = sampled ? 0.8F : 0.0F,
                                  .top_k = 20,
                                  .top_p = 0.95F,
                                  .seed = 73},
                                 initial);
  std::vector<std::int32_t> tokens;
  std::string error;
  while (tokens.size() < count) {
    qfn::Session::DecodeResult result;
    Require(session.DecodeStep(count - tokens.size(), sampler, &result, &error,
                               false),
            error);
    tokens.insert(tokens.end(), result.tokens.begin(), result.tokens.end());
  }
  return tokens;
}
void SameLogits(const qfn::Session& a, const qfn::Session& b) {
  Require(a.Logits().size() == b.Logits().size() &&
              std::memcmp(a.Logits().data(), b.Logits().data(),
                          a.Logits().size_bytes()) == 0,
          "restored next-step logits differ");
}
void CheckSlotReuse(const std::shared_ptr<qfn::Model>& model, bool mtp) {
  const auto mode = mtp ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive;
  qfn::ContinuationAdapter adapter(model, mode, 4096,
                                   {'r', 'e', 'u', 's', 'e'});
  Guard guard;
  auto slot = adapter.CreateSlot(guard);
  auto& reused = adapter.GetSession(*slot);
  std::string error;
  auto fresh = model->CreateSession(mode, 4096, &error);
  Require(bool(fresh), error);
  const auto empty_bytes = reused.AllocatedBytes();
  const auto fresh_empty_bytes = fresh->AllocatedBytes();
  const auto first = model->Tokenize("Continue: red, blue, red, blue,");
  const auto next = model->Tokenize(
      "Write a Python function that retries a failed request three times.");
  Require(reused.Sync(first, &error), error);
  (void)Decode(reused, 48);
  for (const bool sampled : {false, true}) {
    const auto warm_bytes = reused.AllocatedBytes();
    Require(warm_bytes <= model->SessionBytes(mode, 4096),
            "slot scratch exceeded its admission bound");
    if (mtp)
      Require(warm_bytes > empty_bytes && reused.Statistics().drafted > 0,
              "slot reuse did not exercise MTP rollback");
    Require(adapter.Invalidate(*slot), "slot reassignment failed");
    Require(reused.Position() == 0 && reused.Tokens().empty(),
            "slot reassignment retained logical state");
    Require(reused.AllocatedBytes() == warm_bytes,
            "slot reassignment discarded reusable rollback");
    fresh->Reset();
    Require(reused.Sync(next, &error) && fresh->Sync(next, &error), error);
    SameLogits(reused, *fresh);
    const auto reused_before = reused.Statistics();
    const auto fresh_before = fresh->Statistics();
    Require(Decode(reused, 48, sampled) == Decode(*fresh, 48, sampled),
            "retained slot scratch changed continuation tokens");
    SameLogits(reused, *fresh);
    Require(reused.Statistics().drafted - reused_before.drafted ==
                    fresh->Statistics().drafted - fresh_before.drafted &&
                reused.Statistics().accepted - reused_before.accepted ==
                    fresh->Statistics().accepted - fresh_before.accepted,
            "retained slot scratch changed draft acceptance");
  }
  // A request stopped by cancellation leaves the session invalid, and an
  // invalid session releases its rollback scratch when the slot is reused:
  // keeping it slowed later short prefills on gfx1151 (S3).
  if (mtp)
    Require(reused.AllocatedBytes() > empty_bytes,
            "cancelled slot had no rollback scratch to release");
  reused.SetCancellationCheck([] { return true; });
  Require(!reused.Evaluate(first.front(), &error) && !reused.IsValid(),
          "cancelled slot remained valid");
  reused.SetCancellationCheck({});
  Require(adapter.Invalidate(*slot), "cancelled slot reassignment failed");
  Require(reused.Position() == 0 && reused.Tokens().empty(),
          "cancelled slot reassignment retained logical state");
  Require(reused.AllocatedBytes() == empty_bytes,
          "cancelled slot reassignment retained rollback scratch");
  fresh->Reset();
  Require(reused.Sync(next, &error) && fresh->Sync(next, &error), error);
  SameLogits(reused, *fresh);
  Require(Decode(reused, 16) == Decode(*fresh, 16),
          "cancelled-slot reassignment changed continuation tokens");
  // Owned sessions without a component cache (KVarN) reassign through
  // ResetForReuse: a valid one keeps its scratch, a cancelled one releases
  // it.
  const auto owned_bytes = fresh->AllocatedBytes();
  if (mtp)
    Require(owned_bytes > fresh_empty_bytes,
            "owned session had no rollback scratch to keep");
  fresh->ResetForReuse();
  Require(fresh->Position() == 0 && fresh->Tokens().empty() &&
              fresh->AllocatedBytes() == owned_bytes,
          "valid owned-session reset discarded reusable rollback");
  reused.Reset();
  Require(reused.Sync(next, &error) && fresh->Sync(next, &error), error);
  SameLogits(reused, *fresh);
  Require(Decode(reused, 16) == Decode(*fresh, 16),
          "retained owned-session scratch changed continuation tokens");
  fresh->SetCancellationCheck([] { return true; });
  Require(!fresh->Evaluate(next.front(), &error) && !fresh->IsValid(),
          "cancelled owned session remained valid");
  fresh->SetCancellationCheck({});
  fresh->ResetForReuse();
  Require(fresh->Position() == 0 && fresh->Tokens().empty() &&
              fresh->AllocatedBytes() == fresh_empty_bytes,
          "cancelled owned-session reset retained rollback scratch");
  reused.Reset();
  Require(reused.Sync(next, &error) && fresh->Sync(next, &error), error);
  SameLogits(reused, *fresh);
  Require(Decode(reused, 16) == Decode(*fresh, 16),
          "owned-session reassignment changed continuation tokens");
  reused.Reset();
  Require(reused.AllocatedBytes() == empty_bytes,
          "explicit reset retained rollback scratch");
  std::cout << (mtp ? "MTP" : "AR")
            << ",slot_reuse,sampled/greedy/cancellation/reset,PASS\n";
}
void CheckInPass(const std::shared_ptr<qfn::Model>& model, bool mtp,
                 std::span<const std::int32_t> prompt) {
  gufo::hip::TransferPool streams(2);
  qfn::ContinuationAdapter adapter(
      model,
      mtp ? gufo::core::SessionMode::kSpeculative
          : gufo::core::SessionMode::kAutoregressive,
      16384, {'i', 'n', 'p', 'a', 's', 's'});
  Guard source_guard, reference_guard;
  auto source = adapter.CreateSlot(source_guard);
  auto reference = adapter.CreateSlot(reference_guard);
  auto& a = adapter.GetSession(*source);
  auto& b = adapter.GetSession(*reference);
  std::string error;
  for (const auto boundary : {2047U, 8192U, 8203U}) {
    const auto predicted =
        adapter.PrefillCheckpointPositions(*source, boundary);
    Checkpoint frozen;
    bool called = false;
    adapter.PrefillThrough(*source, prompt.first(boundary + 5), boundary, [&] {
      called = true;
      Require(adapter.Positions(*source) == predicted,
              "in-pass positions differ from admission prediction");
      Reject([&] { (void)adapter.GetSession(*source); });
      Reject([&] { (void)a.Sync(prompt.first(boundary + 5), &error); });
      Reject([&] { a.Reset(); });
      Require(!adapter.Invalidate(*source),
              "in-pass consumer invalidated its capture frontier");
      frozen = Capture(adapter, *source, streams);
    });
    Require(called && a.Position() == boundary + 5,
            "in-pass capture changed the completed live frontier");
    std::unique_ptr<qfn::SessionSnapshot> native;
    Require(
        b.SyncThrough(prompt.first(boundary + 5), boundary, &native, &error),
        error);
    SameLogits(a, b);
    Require(b.RestoreSnapshot(*native, &error), error);
    const auto independent = Capture(adapter, *reference, streams);
    Require(frozen.positions == independent.positions &&
                frozen.bytes == independent.bytes,
            "component in-pass bytes differ from native boundary restore");
    Load(adapter, *source, frozen, streams);
    Require(adapter.Validate(*source, frozen.positions),
            "component in-pass restore rejected");
    SameLogits(a, b);
    Require(Decode(a, 24, mtp) == Decode(b, 24, mtp),
            "in-pass sampled continuation differs");
    SameLogits(a, b);
    if (mtp)
      Require(a.Statistics().drafted > 0, "in-pass MTP ran no drafts");
    Require(adapter.Invalidate(*source) && adapter.Invalidate(*reference),
            "in-pass reset failed");
  }
  const auto component =
      std::ranges::find_if(adapter.Components(), [](const auto& c) {
        return c.kind == cache::ComponentKind::kPrivateState;
      });
  Require(component != adapter.Components().end(), "missing private component");
  for (const bool throw_consumer : {false, true}) {
    auto stream = streams.TryAcquire();
    Require(bool(stream), "in-pass failure stream exhausted");
    std::vector<std::byte> bytes(
        std::min<std::size_t>(64, component->state_bytes));
    std::optional<cache::Completion> pending;
    Reject([&] {
      adapter.PrefillThrough(*source, prompt.first(2052), 2047, [&] {
        pending.emplace(adapter.CapturePrivatePiece(*source, component->id, 0,
                                                    bytes, *stream));
        if (throw_consumer)
          throw std::runtime_error("rejected in-pass consumer");
      });
    });
    Require(pending && pending->Ready() &&
                pending->Wait() == cache::TransferResult::kSucceeded,
            "failed in-pass consumer left a transfer reading retired scratch");
    Require(adapter.Invalidate(*source),
            "failed in-pass transfer did not drain");
  }
  std::cout << (mtp ? "MTP" : "AR") << ",in_pass,PASS\n";
}
void CheckTransferAndGuardLifetime(const std::shared_ptr<qfn::Model>& model,
                                   bool mtp,
                                   std::span<const std::int32_t> prompt) {
  std::cout << (mtp ? "MTP" : "AR") << ",transfer_and_guard_lifetime"
            << std::endl;
  qfn::ContinuationAdapter adapter(
      model,
      mtp ? gufo::core::SessionMode::kSpeculative
          : gufo::core::SessionMode::kAutoregressive,
      16384, {'l', 'i', 'f', 'e'});
  gufo::hip::TransferPool streams(2);
  Guard source_guard, destination_guard;
  auto source = adapter.CreateSlot(source_guard);
  auto destination = adapter.CreateSlot(destination_guard);
  const auto checkpoint = Capture(adapter, *source, streams);
  for (const bool overwrite : {false, true}) {
    Load(adapter, *destination, checkpoint, streams);
    gufo::hip::TransferPool failed_streams(1);
    auto stream = failed_streams.TryAcquire();
    {
      auto copy =
          adapter.LoadPrivate(*destination, {4}, checkpoint.bytes[3], *stream);
      FailNextTransferWait();
      if (overwrite)
        copy = cache::Completion(std::make_unique<Finished>());
    }
    Require(!adapter.Validate(*destination, checkpoint.positions),
            "discarded completion error was not latched");
    Require(adapter.Invalidate(*destination),
            "discarded completion error could not recover");
  }
  auto bytes = checkpoint.bytes[3];
  auto stream = streams.TryAcquire();
  auto copy = adapter.CapturePrivate(*source, {4}, bytes, *stream);
  source.reset();
  Wait(std::move(copy));
  Require(bytes == checkpoint.bytes[3],
          "slot destruction failed to drain an outstanding capture");
  stream.reset();
  destination.reset();

  // The real slot guard leases a new independent transfer for every piece.
  // The idle worker is kept dormant so each foreground mutation is the oracle.
  for (const int path : {0, 1, 2, 3, 4}) {
    cache::ResourceLedger ledger({1 << 30, 1 << 30, 1 << 30, 1 << 30});
    IncompatibleStream unused;
    cache::LeasedSlot leased(
        ledger, adapter, unused, {1},
        cache::IdleSpillConfig{
            .piece_bytes = 4096,
            .acquire_stream = [&]() -> std::unique_ptr<cache::Stream> {
              return streams.TryAcquire();
            },
            .device_idle = [] { return false; }});
    auto lease = leased.Acquire();
    auto& session = adapter.GetSession(lease.Execution());
    std::string error;
    const auto depth = path == 0 ? 2047u : path == 1 ? 8203u : 17u;
    Require(session.Sync(prompt.first(depth), &error), error);
    const auto positions = adapter.Positions(lease.Execution());
    auto legacy = session.SaveSnapshot(&error);
    Require(bool(legacy), error);
    std::vector<cache::ResourceCharge> charges;
    std::vector<std::shared_ptr<cache::BorrowedRows>> borrowed;
    std::vector<std::vector<std::byte>> expected;
    for (const auto& component : adapter.Components()) {
      const auto rows = positions[component.id.value - 1].valid_rows;
      if (!component.row_bytes || !rows)
        continue;
      auto& oracle = expected.emplace_back(rows * component.row_bytes);
      auto transfer = streams.TryAcquire();
      Wait(adapter.CopyRowsOut(lease.Execution(), component.id, 0, rows, oracle,
                               *transfer));
      transfer.reset();
      auto reservation =
          ledger.Reserve(cache::ResourceCategory::kBackingFree, oracle.size());
      auto backing = std::make_shared<std::vector<std::byte>>(oracle.size());
      auto charge = reservation.Convert();
      auto assigned =
          charge.ReserveBacking(cache::ResourceCategory::kBackingAssigned);
      charges.push_back(charge);
      borrowed.push_back(lease.Borrow(
          component.id, 0, rows, std::move(assigned), backing, *backing,
          adapter.BorrowRowsView(lease.Execution(), component.id, 0, rows)));
    }
    {
      const auto saved = Capture(adapter, lease.Execution(), streams);
      std::vector<cache::RowPin> pins(adapter.Components().size());
      std::size_t next = 0;
      for (const auto& component : adapter.Components()) {
        const auto rows = positions[component.id.value - 1].valid_rows;
        if (component.row_bytes && rows)
          pins[component.id.value - 1] = borrowed[next++]->Pin();
      }
      Guard destination_guard;
      auto destination = adapter.CreateSlot(destination_guard);
      Load(adapter, *destination, saved, streams, false, &pins);
      Require(adapter.Validate(*destination, saved.positions),
              "pinned native rows did not validate");
      Require(Capture(adapter, *destination, streams).bytes == saved.bytes,
              "pinned native rows changed the restored checkpoint");
      Require(std::ranges::any_of(pins,
                                  [](const auto& pin) {
                                    return pin.Location() &&
                                           !pin.Bytes().empty();
                                  }),
              "native borrowed restore did not read a live device range");
    }
    if (path == 0)
      session.Reset();
    else if (path == 1)
      Require(session.RestoreSnapshot(*legacy, &error), error);
    else if (path == 2)
      adapter.BeginRestore(lease.Execution(), positions);
    else if (path == 3) {
      auto edited =
          std::vector<std::int32_t>(prompt.begin(), prompt.begin() + 17);
      edited[3] = edited[3] == 1 ? 2 : 1;
      Require(session.Sync(edited, &error), error);
    } else {
      Guard peer_guard;
      auto peer = adapter.CreateSlot(peer_guard);
      auto& other = adapter.GetSession(*peer);
      Require(other.Sync(prompt.first(17), &error), error);
      std::array<qfn::Session::BatchOutcome, 2> outcomes;
      const std::array<qfn::Session::AdvanceRequest, 2> advances{
          {{&session, prompt[17], &outcomes[0]},
           {&other, prompt[17], &outcomes[1]}}};
      Require(qfn::Session::EvaluateBatch(advances, &error), error);
      Require(outcomes[0].completed && outcomes[1].completed,
              "guarded advance batch did not execute both peers");
      std::array<sampling::SamplerState, 2> samplers;
      std::array<qfn::Session::DecodeResult, 2> results;
      const std::array<qfn::Session::DecodeRequest, 2> decodes{
          {{&session, 4, &samplers[0], &results[0], false, &outcomes[0]},
           {&other, 4, &samplers[1], &results[1], false, &outcomes[1]}}};
      Require(qfn::Session::DecodeBatch(decodes, &error), error);
      Require(outcomes[0].completed && outcomes[1].completed,
              "guarded decode batch did not execute both peers");
      session.Reset();
    }
    for (std::size_t i = 0; i < borrowed.size(); ++i) {
      Require(borrowed[i]->IsValid() && !borrowed[i]->Location(),
              "real guard did not preserve a borrowed component");
      const auto owner = std::static_pointer_cast<const std::vector<std::byte>>(
          borrowed[i]->Owner());
      Require(owner && *owner == expected[i],
              "real guard preserved changed rows");
    }
  }
  std::cout << (mtp ? "MTP" : "AR") << ",transfer_and_guard_lifetime,PASS\n";
}
void BatchIsolation(qfn::Session& rejected, qfn::Session& healthy,
                    std::int32_t token) {
  const auto rejected_position = rejected.Position();
  auto healthy_position = healthy.Position();
  std::array<qfn::Session::BatchOutcome, 2> outcomes;
  std::string error;
  const std::array<qfn::Session::AdvanceRequest, 2> advances{
      {{&rejected, token, &outcomes[0]}, {&healthy, token, &outcomes[1]}}};
  Require(!qfn::Session::EvaluateBatch(advances, &error),
          "unready batch peer was admitted");
  Require(!outcomes[0].completed && !outcomes[0].error.empty() &&
              outcomes[1].completed && outcomes[1].error.empty() &&
              healthy.Position() == healthy_position + 1 &&
              rejected.Position() == rejected_position,
          "advance readiness failure prevented healthy peer execution");
  std::array<sampling::SamplerState, 2> samplers{
      sampling::SamplerState({.temperature = 0.0F}),
      sampling::SamplerState({.temperature = 0.0F})};
  std::array<qfn::Session::DecodeResult, 2> results;
  outcomes = {};
  healthy_position = healthy.Position();
  const std::array<qfn::Session::DecodeRequest, 2> decodes{
      {{&rejected, 4, &samplers[0], &results[0], false, &outcomes[0]},
       {&healthy, 4, &samplers[1], &results[1], false, &outcomes[1]}}};
  Require(!qfn::Session::DecodeBatch(decodes, &error),
          "unready decode peer was admitted");
  Require(
      !outcomes[0].completed && !outcomes[0].error.empty() &&
          outcomes[1].completed && outcomes[1].error.empty() &&
          !results[1].tokens.empty() &&
          healthy.Position() == healthy_position + results[1].tokens.size() &&
          rejected.Position() == rejected_position,
      "decode readiness failure prevented healthy peer execution");
}
void Check(const std::shared_ptr<qfn::Model>& model, bool mtp,
           std::span<const std::int32_t> prompt) {
  gufo::hip::TransferPool streams(2);
  qfn::ContinuationAdapter adapter(
      model,
      mtp ? gufo::core::SessionMode::kSpeculative
          : gufo::core::SessionMode::kAutoregressive,
      16384, {'t', 'e', 's', 't'});
  Guard source_guard, destination_guard;
  auto source = adapter.CreateSlot(source_guard);
  auto destination = adapter.CreateSlot(destination_guard);
  source_guard.adapter = destination_guard.adapter = &adapter;
  source_guard.streams = destination_guard.streams = &streams;
  source_guard.slot = source.get();
  destination_guard.slot = destination.get();
  auto& a = adapter.GetSession(*source);
  auto& b = adapter.GetSession(*destination);
  std::string error;
  const auto capacity = model->PrefillCapacity();
  const auto through = model->PrefillThroughCapacity();
  Require(adapter.PlanPrefill(*source, 0, through) ==
              std::vector<cache::Rows>{through},
          "prefill plan split a supported tail");
  Require(adapter.PlanPrefill(*source, 0, through + 1) ==
              std::vector<cache::Rows>{capacity, through + 1},
          "prefill plan exceeds the executor pass capacity");
  Require(adapter.PlanPrefill(*source, 17, 17 + through) ==
              std::vector<cache::Rows>{17 + through},
          "prefill plan lost its continuation offset");
  Reject([&] { (void)adapter.PlanPrefill(*source, 2, 1); });
  Reject([&] { (void)adapter.PlanPrefill(*source, 0, 16385); });
  const auto empty = Capture(adapter, *source, streams);
  Load(adapter, *destination, empty, streams);
  Require(adapter.Validate(*destination, empty.positions),
          "empty restore rejected");
  Require(Capture(adapter, *destination, streams).bytes == empty.bytes,
          "empty state differs");
  for (const auto boundary : {2047U, 8203U}) {
    std::cout << (mtp ? "MTP" : "AR") << ",boundary," << boundary << std::endl;
    Require(a.Sync(prompt.first(boundary), &error), error);
    const auto checkpoint = Capture(adapter, *source, streams);
    const auto registered = Capture(adapter, *source, streams, true);
    Require(registered.bytes == checkpoint.bytes,
            "registered capture differs from pageable HIP capture");
    source_guard.Borrow(checkpoint);
    Load(adapter, *destination, registered, streams);
    Require(adapter.Validate(*destination, checkpoint.positions),
            "complete restore rejected");
    Require(Capture(adapter, *destination, streams).bytes == checkpoint.bytes,
            "component round trip differs, including chronological raw ring");
    SameLogits(a, b);
    Reject([&] { (void)adapter.Validate(*destination, checkpoint.positions); });
    const auto before = source_guard.overwrites;
    const auto continued = Decode(a, 24, boundary == 8203);
    Require(continued == Decode(b, 24, boundary == 8203),
            "restored draft/continuation tokens differ");
    SameLogits(a, b);
    Require(a.Statistics().drafted == b.Statistics().drafted &&
                a.Statistics().accepted == b.Statistics().accepted,
            "restored draft sequence or acceptance differs");
    if (mtp)
      Require(a.Statistics().drafted > 0, "MTP qualification ran no drafts");
    Require(source_guard.overwrites > before,
            "decode did not invoke mutation guards");
    // Capture the carried draft residual and trained acceptance controller,
    // rather than only the initial prefill frontier.
    const auto mid_decode = Capture(adapter, *source, streams);
    Load(adapter, *destination, mid_decode, streams);
    Require(adapter.Validate(*destination, mid_decode.positions),
            "mid-decode restore rejected");
    Require(Capture(adapter, *destination, streams).bytes == mid_decode.bytes,
            "mid-decode private state changed during restore");
    const auto next = Decode(a, 16, boundary == 8203);
    Require(next == Decode(b, 16, boundary == 8203),
            "carried draft state replay differs");
    SameLogits(a, b);
    if (mtp && boundary == 8203)
      Require(a.Statistics().accepted < a.Statistics().drafted,
              "sampled qualification exercised no rejected draft");

    // A shorter checkpoint replaces every recurrent component; subsequent
    // edits start from that exact boundary rather than truncating recurrence.
    Load(adapter, *source, checkpoint, streams);
    Require(adapter.Validate(*source, checkpoint.positions),
            "shorter restore rejected");
    Require(Capture(adapter, *source, streams).bytes == checkpoint.bytes,
            "earlier recurrent boundary was truncated");
    auto edited =
        std::vector<std::int32_t>(prompt.begin(), prompt.begin() + boundary);
    edited[boundary / 2] = edited[boundary / 2] == 1 ? 2 : 1;
    Require(a.Sync(edited, &error), error);
    Require(source_guard.overwrites > before, "edited prefill skipped guards");
    source_guard.borrowed = nullptr;
    Require(adapter.Invalidate(*source) && source->IsValid(), "reset failed");
    Require(adapter.Invalidate(*destination), "destination reset failed");
  }
  std::cout << (mtp ? "MTP" : "AR") << ",earlier_edit" << std::endl;
  Require(a.Sync(prompt.first(1023), &error), error);
  const auto earlier = Capture(adapter, *source, streams);
  Require(a.Sync(prompt.first(8203), &error), error);
  auto edited =
      std::vector<std::int32_t>(prompt.begin(), prompt.begin() + 8203);
  edited[2048] = edited[2048] == 1 ? 2 : 1;
  Load(adapter, *destination, earlier, streams);
  Require(adapter.Validate(*destination, earlier.positions),
          "earlier edit checkpoint rejected");
  Require(Capture(adapter, *destination, streams).bytes == earlier.bytes,
          "earlier edit checkpoint state changed");
  // Compare the edited branch with an uninterrupted run using the same
  // prefill pass boundaries. Different GEMM batch geometries are a separate
  // numerical contract, rather than a snapshot restore oracle.
  a.Reset();
  Require(a.Sync(std::span(edited).first(1023), &error), error);
  Require(a.Sync(edited, &error) && b.Sync(edited, &error), error);
  SameLogits(a, b);
  Require(Decode(a, 8) == Decode(b, 8),
          "edit after an earlier checkpoint changed drafts");
  Require(adapter.Invalidate(*source), "edit source reset failed");
  Require(a.Sync(prompt.first(8203), &error), error);
  const auto checkpoint = Capture(adapter, *source, streams);
  Load(adapter, *destination, checkpoint, streams);
  Require(adapter.Validate(*destination, checkpoint.positions),
          "peer overlap restore failed");
  {
    auto first_stream = streams.TryAcquire();
    auto second_stream = streams.TryAcquire();
    Require(first_stream && second_stream,
            "settled transfers retained stream leases");
    auto first_bytes = checkpoint.bytes[3];
    auto second_bytes = checkpoint.bytes[4];
    auto first_copy =
        adapter.CapturePrivate(*source, {4}, first_bytes, *first_stream);
    auto second_copy =
        adapter.CapturePrivate(*source, {5}, second_bytes, *second_stream);
    Require(b.Evaluate(prompt[0], &error), error);
    Wait(std::move(first_copy));
    Wait(std::move(second_copy));
    Require(first_bytes == checkpoint.bytes[3] &&
                second_bytes == checkpoint.bytes[4],
            "peer graph execution changed a frozen capture");
  }
  std::cout << (mtp ? "MTP" : "AR") << ",pending_and_failure" << std::endl;
  auto stream = streams.TryAcquire();
  auto bytes = checkpoint.bytes[3];
  auto capture = adapter.CapturePrivate(*source, {4}, bytes, *stream);
  Reject([&] { (void)a.Evaluate(prompt[0], &error); });
  Require(!adapter.Invalidate(*source), "reset ignored a pending read");
  BatchIsolation(a, b, prompt[0]);
  Wait(std::move(capture));
  stream.reset();
  Load(adapter, *destination, checkpoint, streams, true);
  Require(!adapter.Validate(*destination, checkpoint.positions),
          "missing private byte accepted");
  Require(!adapter.Validate(*destination, checkpoint.positions),
          "validation failure was not latched");
  BatchIsolation(b, a, prompt[0]);
  Require(adapter.Invalidate(*destination),
          "failed restore could not be invalidated");
  adapter.BeginRestore(*destination, checkpoint.positions);
  BatchIsolation(b, a, prompt[0]);
  stream = streams.TryAcquire();
  IncompatibleStream incompatible;
  Reject([&] {
    (void)adapter.LoadPrivatePiece(
        *destination, {2}, 0,
        std::span<const std::byte>(checkpoint.bytes[1]).first(1), incompatible);
  });
  // Submission/type errors latch even when the caller discards the error.
  Require(!adapter.Validate(*destination, checkpoint.positions),
          "failed load was not latched");
  Require(adapter.Invalidate(*destination), "submission failure reset failed");
  stream.reset();
  FailNextResetSubmission();
  Require(!adapter.Invalidate(*destination) && !destination->IsValid(),
          "failed zero-fill submission exposed an empty slot");
  Reject([&] { (void)b.Sync(prompt.first(17), &error); });
  Reject([&] { (void)adapter.GetSession(*destination); });
  Require(adapter.Invalidate(*destination),
          "failed zero-fill submission could not recover");
  const auto reset_checkpoint = Capture(adapter, *source, streams);
  source_guard.Borrow(reset_checkpoint);
  a.Reset();
  Require(!source_guard.release_failed, "reset preservation failed");
  for (std::size_t i = 0; i < adapter.Components().size(); ++i)
    if (adapter.Components()[i].row_bytes &&
        reset_checkpoint.positions[i].valid_rows)
      Require(source_guard.detached[i],
              "reset failed to preserve borrowed component");
  source_guard.borrowed = nullptr;
  const auto reset_state = Capture(adapter, *source, streams);
  Load(adapter, *destination, reset_state, streams);
  Require(adapter.Validate(*destination, reset_state.positions),
          "reset empty restore failed");
  Require(Capture(adapter, *destination, streams).bytes == reset_state.bytes,
          "reset state was captured before zero fills completed");
  Require(a.Sync(prompt.first(8203), &error), error);
  const auto destroyed = Capture(adapter, *source, streams);
  source_guard.Borrow(destroyed);
  source.reset();
  Require(!source_guard.release_failed, "destruction preservation failed");
  source_guard.borrowed = nullptr;
  std::cout << (mtp ? "MTP" : "AR") << ",private_bytes,";
  std::size_t private_bytes = 0;
  for (const auto& component : adapter.Components())
    private_bytes += component.state_bytes;
  std::cout << private_bytes << ",PASS\n";
}
void CheckImages(const std::shared_ptr<qfn::Model>& model, bool mtp) {
  namespace vision = gufo::models::qwen::vision;
  std::cout << (mtp ? "MTP" : "AR") << ",images" << std::endl;
  Require(bool(model->VisionEncoder()),
          "image adapter qualification requires the adjacent projector");
  auto image = [](bool blue) {
    auto prompt = std::make_shared<vision::Prompt>();
    const vision::ImageGrid grid{0, 2, 2};
    prompt->tokens = {vision::kImageToken, vision::kImageToken,
                      vision::kImageToken, vision::kImageToken, 42};
    prompt->rope.images.push_back(grid);
    gufo::core::Image pixels;
    pixels.width = pixels.height = 64;
    pixels.pixels.resize(64 * 64 * 3);
    for (std::size_t i = blue ? 2 : 0; i < pixels.pixels.size(); i += 3)
      pixels.pixels[i] = 255;
    prompt->images.push_back(
        {std::make_shared<const gufo::core::Image>(std::move(pixels)), grid});
    prompt->images.back().prefix_identity.fill(blue ? 2 : 1);
    prompt->cache_identity.assign(32, blue ? 2 : 1);
    return prompt;
  };
  const auto red = image(false), blue = image(true);
  const std::vector<std::int32_t> tokens(red->tokens.begin(),
                                         red->tokens.end());
  gufo::hip::TransferPool streams(2);
  qfn::ContinuationAdapter adapter(
      model,
      mtp ? gufo::core::SessionMode::kSpeculative
          : gufo::core::SessionMode::kAutoregressive,
      64, {'i', 'm', 'a', 'g', 'e'});
  Guard first_guard, second_guard;
  auto first = adapter.CreateSlot(first_guard);
  auto second = adapter.CreateSlot(second_guard);
  auto& a = adapter.GetSession(*first);
  auto& b = adapter.GetSession(*second);
  std::string error;
  a.ConfigureVision(red);
  Require(a.Sync(std::span(tokens).first(2), &error), error);
  const auto checkpoint = Capture(adapter, *first, streams);
  Load(adapter, *second, checkpoint, streams);
  Require(!adapter.Validate(*second, checkpoint.positions),
          "image restored without its attachment");
  Require(adapter.Invalidate(*second), "missing attachment reset failed");
  b.ConfigureVision(blue);
  Load(adapter, *second, checkpoint, streams);
  Require(!adapter.Validate(*second, checkpoint.positions),
          "different pixels accepted at the boundary");
  Require(adapter.Invalidate(*second), "wrong attachment reset failed");
  b.ConfigureVision(red);
  Load(adapter, *second, checkpoint, streams);
  Require(adapter.Validate(*second, checkpoint.positions),
          "matching image restore rejected");
  Require(a.Sync(tokens, &error) && b.Sync(tokens, &error), error);
  SameLogits(a, b);
  a.ConfigureVision(nullptr);
  const std::array<std::int32_t, 2> plain{42, 43};
  Require(a.Sync(plain, &error), error);
  const auto text = Capture(adapter, *first, streams);
  Load(adapter, *second, text, streams);
  Require(adapter.Validate(*second, text.positions),
          "text restore retained stale image input");
  Require(a.Evaluate(44, &error) && b.Evaluate(44, &error), error);
  SameLogits(a, b);
  auto suffix = std::make_shared<vision::Prompt>();
  suffix->tokens.push_back(42);
  for (const auto& input : {red, blue}) {
    auto prepared = input->images.front();
    prepared.grid.offset = static_cast<std::uint32_t>(suffix->tokens.size());
    suffix->rope.images.push_back(prepared.grid);
    suffix->images.push_back(std::move(prepared));
    suffix->tokens.insert(suffix->tokens.end(), input->tokens.begin(),
                          input->tokens.end());
  }
  suffix->cache_identity.assign(32, 2);
  const std::vector<std::int32_t> suffix_tokens(suffix->tokens.begin(),
                                                suffix->tokens.end());
  for (const std::size_t boundary : {1, 6}) {
    qfn::ContinuationAdapter suffix_adapter(
        model,
        mtp ? gufo::core::SessionMode::kSpeculative
            : gufo::core::SessionMode::kAutoregressive,
        64, {'s', 'u', 'f', 'f', 'i', 'x'});
    Guard ga, gb;
    auto source = suffix_adapter.CreateSlot(ga);
    auto destination = suffix_adapter.CreateSlot(gb);
    auto& sa = suffix_adapter.GetSession(*source);
    auto& sb = suffix_adapter.GetSession(*destination);
    sa.ConfigureVision(suffix);
    sb.ConfigureVision(suffix);
    Require(sa.Sync(std::span(suffix_tokens).first(boundary), &error), error);
    const auto saved = Capture(suffix_adapter, *source, streams);
    Load(suffix_adapter, *destination, saved, streams);
    Require(suffix_adapter.Validate(*destination, saved.positions),
            "future image checkpoint rejected");
    Require(sa.Sync(suffix_tokens, &error) && sb.Sync(suffix_tokens, &error),
            error);
    SameLogits(sa, sb);
    Require(Capture(suffix_adapter, *source, streams).bytes ==
                Capture(suffix_adapter, *destination, streams).bytes,
            "restoration changed future image component bytes");
  }
  std::cout << (mtp ? "MTP" : "AR") << ",image_restore,PASS\n";
}
void CheckSharedKvarn(const std::shared_ptr<qfn::Model>& model, bool mtp,
                      std::span<const std::int32_t> prompt) {
  const auto mode = mtp ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive;
  constexpr std::uint32_t context = 4096;
  const auto geometry = model->KvGeometry(mode, context);
  auto pool = std::make_shared<gufo::hip::KvBlockPool>(
      std::vector<gufo::hip::KvBlockPlane>(geometry.planes().begin(),
                                           geometry.planes().end()),
      geometry.blocks() * 2);
  qfn::ContinuationAdapter adapter(model, mode, context,
                                   {'s', 'h', 'a', 'r', 'e'}, pool);
  const auto identity = adapter.CompatibilityIdentity();
  // Five caller bytes, schema, enabled, then little-endian codec bits.
  Require(identity.size() >= 17 && identity[13] == model->KvarnBits() &&
              identity[14] == 0 && identity[15] == 0 && identity[16] == 0,
          "component compatibility identity lost KV precision");
  gufo::hip::TransferPool streams(2);
  std::array<Guard, 4> guards;
  std::array<std::unique_ptr<cache::Slot>, 4> slots;
  for (std::size_t i = 0; i < slots.size(); ++i)
    slots[i] = adapter.CreateSlot(guards[i]);
  Require(pool->free_blocks() == pool->blocks(),
          "empty slots preallocated whole contexts");
  adapter.ReserveKvRows(*slots[0], context, 2);
  adapter.ReserveKvRows(*slots[1], context, 2);
  Reject([&] { adapter.ReserveKvRows(*slots[2], 1, 2); });
  Require(pool->free_blocks() == 0, "failed reservation leaked capacity");
  adapter.ReleaseKvReservation(*slots[0]);
  adapter.ReleaseKvReservation(*slots[1]);
  Require(pool->free_blocks() == pool->blocks(),
          "reservations were not released");
  std::string error;
  auto oracle = model->CreateSession(mode, context, &error);
  Require(bool(oracle), error);
  auto& a = adapter.GetSession(*slots[0]);
  auto& b = adapter.GetSession(*slots[1]);
  for (const auto boundary :
       {127U, 128U, 129U, 255U, 256U, 257U, 2047U, 2048U, 2049U}) {
    for (auto& slot : slots)
      Require(adapter.Invalidate(*slot), "shared slot reset failed");
    oracle->Reset();
    // Interleave another slot between the first and later logical blocks.
    Require(a.Sync(prompt.first(17), &error), error);
    Require(adapter.GetSession(*slots[2]).Sync(prompt.first(17), &error),
            error);
    Require(adapter.GetSession(*slots[3]).Sync(prompt.first(17), &error),
            error);
    Require(oracle->Sync(prompt.first(17), &error), error);
    Require(a.Sync(prompt.first(boundary), &error) &&
                oracle->Sync(prompt.first(boundary), &error),
            error);
    SameLogits(a, *oracle);
    const auto saved = Capture(adapter, *slots[0], streams);
    Require(!a.SaveSnapshot(&error),
            "legacy snapshot accepted shared device rows");
    Load(adapter, *slots[1], saved, streams);
    Require(adapter.Validate(*slots[1], saved.positions),
            "shared restore rejected");
    Require(Capture(adapter, *slots[1], streams).bytes == saved.bytes,
            "fragmented restore changed component bytes");
    SameLogits(a, b);
    std::unique_ptr<qfn::SessionSnapshot> legacy;
    if (boundary == 257) {
      legacy = oracle->SaveSnapshot(&error,
                                    qfn::Session::SnapshotMode::kMaterialized);
      Require(bool(legacy), error);
      auto bytes = legacy->bytes();
      std::vector<std::uint8_t> wrong(bytes.begin(), bytes.end());
      // The other codec's native header must reject before copying payload.
      std::uint64_t executor_bytes = 0;
      Require(wrong.size() >= 32, "native snapshot outer header truncated");
      std::memcpy(&executor_bytes, wrong.data() + 24, sizeof(executor_bytes));
      Require(executor_bytes >= 8 && executor_bytes <= wrong.size(),
              "native snapshot executor length invalid");
      const auto base = wrong.size() - executor_bytes;
      const std::string_view expected_magic =
          model->KvarnBits() == 6 ? "QFNSNAK6" : "QFNSNAPK";
      Require(std::memcmp(wrong.data() + base, expected_magic.data(), 8) == 0,
              "native snapshot did not record selected KV codec");
      wrong[base + 6] = model->KvarnBits() == 6 ? 'P' : 'K';
      wrong[base + 7] = model->KvarnBits() == 6 ? 'K' : '6';
      Require(!oracle->RestoreSnapshot(wrong, &error),
              "native snapshot accepted the other KV codec");
      Require(oracle->RestoreSnapshot(bytes, &error), error);
    }
    const auto expected = Decode(*oracle, 12, boundary == 2049);
    if (legacy) {
      Require(oracle->RestoreSnapshot(legacy->bytes(), &error), error);
      Require(Decode(*oracle, 12, false) == expected,
              "native KV snapshot changed replay");
    }
    Require(Decode(a, 12, boundary == 2049) == expected &&
                Decode(b, 12, boundary == 2049) == expected,
            "shared mapping or restored KVarN boundary changed decoding");
    SameLogits(a, b);
    // Restoring an earlier frontier must return now-unused physical blocks.
    const auto short_saved = [&] {
      auto& short_session = adapter.GetSession(*slots[3]);
      (void)short_session;
      return Capture(adapter, *slots[3], streams);
    }();
    const auto before = pool->free_blocks();
    const auto mapped = adapter.MappedKvBlocks(*slots[1]);
    Load(adapter, *slots[1], short_saved, streams);
    Require(adapter.Validate(*slots[1], short_saved.positions),
            "short restore rejected");
    Require(adapter.MappedKvBlocks(*slots[1]) == 1 &&
                pool->free_blocks() == before + mapped - 1,
            "short restore retained unreachable blocks");
    std::cout << (mtp ? "MTP" : "AR") << ",shared_kvarn," << boundary << ",PASS"
              << std::endl;
  }
  for (auto& slot : slots)
    Require(adapter.Invalidate(*slot), "saturated fixture reset failed");
  Require(a.Sync(prompt.first(context - 1), &error), error);
  Require(b.Sync(prompt.first(context - 1), &error), error);
  Require(pool->free_blocks() == 0,
          "two full contexts did not saturate the pool");
  Require(adapter.ReserveKvRows(*slots[0], context, 2) == 0 &&
              adapter.ReserveKvRows(*slots[1], context, 2) == 0,
          "live continuation reserved its already-mapped guard twice");
  Require(a.Evaluate(prompt[context - 1], &error) &&
              b.Evaluate(prompt[context - 1], &error),
          error);
  SameLogits(a, b);
  Require(pool->free_blocks() == 0, "saturated continuation changed capacity");
  Require(adapter.Invalidate(*slots[0]), "full source reset failed");
  a.SetCancellationCheck([] { return true; });
  Require(!a.Evaluate(prompt[0], &error), "cancelled shared session executed");
  a.SetCancellationCheck({});
  for (auto& slot : slots)
    Require(adapter.Invalidate(*slot), "shared cancellation/reset failed");
  Require(pool->free_blocks() == pool->blocks(), "shared reset leaked blocks");
}

void CheckSharedReferences(const std::shared_ptr<qfn::Model>& model, bool mtp,
                           std::span<const std::int32_t> prompt,
                           bool reservation_only = false) {
  const auto mode = mtp ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive;
  constexpr std::uint32_t context = 8192;
  const auto geometry = model->KvGeometry(mode, context);
  auto pool = std::make_shared<gufo::hip::KvBlockPool>(
      std::vector<gufo::hip::KvBlockPlane>(geometry.planes().begin(),
                                           geometry.planes().end()),
      geometry.blocks() * 2);
  qfn::ContinuationAdapter adapter(model, mode, context, {'r', 'e', 'f'}, pool);
  gufo::hip::TransferPool streams(2);
  std::array<Guard, 4> guards;
  std::array<std::unique_ptr<cache::Slot>, 4> slots;
  for (std::size_t i = 0; i < slots.size(); ++i)
    slots[i] = adapter.CreateSlot(guards[i]);
  struct Piece {
    cache::Rows first, end;
    cache::Adapter::RetainedRows rows;
  };
  std::string error;
  for (const auto boundary :
       {251U, 255U, 256U, 383U, 384U, 2048U, 2049U, 2304U, 4096U}) {
    if (reservation_only && boundary != 251 && boundary != 4096)
      continue;
    for (auto& slot : slots)
      Require(adapter.Invalidate(*slot), "reference fixture reset failed");
    auto& parent = adapter.GetSession(*slots[0]);
    Require(parent.Sync(prompt.first(17), &error), error);
    Require(adapter.GetSession(*slots[3]).Sync(prompt.first(17), &error),
            error);
    Require(parent.Sync(prompt.first(boundary), &error), error);
    // Production reserves before checkpoint capture. Retaining this prefix
    // must not add an unreserved copy-on-write obligation.
    adapter.ReserveKvRows(*slots[0], boundary + 32, 2);
    const auto saved = Capture(adapter, *slots[0], streams);
    std::vector<std::vector<Piece>> pieces(adapter.Components().size());
    std::size_t retained = 0;
    for (const auto& d : adapter.Components()) {
      if (!d.row_bytes)
        continue;
      const auto end = saved.positions[d.id.value - 1].valid_rows;
      for (cache::Rows first = 0; first < end; first += d.rows_per_chunk) {
        const auto last = std::min(end, first + d.rows_per_chunk);
        auto rows = last - first == d.rows_per_chunk
                        ? adapter.RetainRows(*slots[0], d.id, first, last)
                        : cache::Adapter::RetainedRows{};
        retained += bool(rows.owner);
        pieces[d.id.value - 1].push_back({first, last, std::move(rows)});
      }
    }
    if (boundary <= 2048)
      Require(retained == 0,
              "dense indexer backlog was retained before completion");
    if (boundary >= 2304)
      Require(retained > 0, "completed shared prefix has no retained GPU rows");
    const auto restore = [&](cache::Slot& slot) {
      adapter.BeginRestore(slot, saved.positions);
      for (std::size_t i = adapter.Components().size(); i-- > 0;) {
        const auto& d = adapter.Components()[i];
        if (!d.row_bytes) {
          auto stream = streams.TryAcquire();
          Wait(adapter.LoadPrivate(slot, d.id, saved.bytes[i], *stream));
          continue;
        }
        for (const auto& piece : pieces[i]) {
          auto stream = streams.TryAcquire();
          const auto bytes =
              piece.rows.owner
                  ? std::span(
                        static_cast<const std::byte*>(piece.rows.owner.get()),
                        piece.rows.bytes)
                  : std::span<const std::byte>(saved.bytes[i])
                        .subspan(piece.first * d.row_bytes,
                                 (piece.end - piece.first) * d.row_bytes);
          Wait(adapter.CopyRowsIn(slot, d.id, piece.first, piece.end, bytes,
                                  *stream));
        }
      }
      Require(adapter.Validate(slot, saved.positions),
              "retained restore rejected");
      Require(Capture(adapter, slot, streams).bytes == saved.bytes,
              "retained restore changed component bytes");
    };
    // Only two free blocks: eager allocation of a duplicate prefix cannot fit.
    const auto before = pool->free_blocks();
    auto filler =
        retained ? pool->Allocate(before - 2) : std::vector<std::uint32_t>{};
    restore(*slots[1]);
    pool->Release(filler);
    if (retained)
      Require(before - pool->free_blocks() <= 2,
              "restored branch allocated duplicate shared prefix blocks");
    restore(*slots[2]);
    auto& child = adapter.GetSession(*slots[1]);
    auto& sibling = adapter.GetSession(*slots[2]);
    adapter.ReserveKvRows(*slots[1], boundary + 32, 2);
    auto growth_filler = pool->Allocate(pool->free_blocks());
    const auto decode_reserved = [&](std::size_t index, qfn::Session& session) {
      const auto reserved = pool->reserved_blocks();
      const auto mapped = adapter.MappedKvBlocks(*slots[index]);
      auto result = Decode(session, 12, boundary == 2049);
      Require(reserved - pool->reserved_blocks() ==
                  adapter.MappedKvBlocks(*slots[index]) - mapped,
              "checkpoint capture introduced reserved COW debt");
      return result;
    };
    const auto expected = decode_reserved(0, parent);
    Require(decode_reserved(1, child) == expected,
            "shared-reference AR/MTP replay changed output");
    pool->Release(growth_filler);
    adapter.ReleaseKvReservation(*slots[0]);
    adapter.ReleaseKvReservation(*slots[1]);
    SameLogits(parent, child);
    Require(Capture(adapter, *slots[2], streams).bytes == saved.bytes,
            "parent or child append mutated the untouched sibling");
    Require(sibling.Evaluate(prompt[boundary + 1], &error), error);
    (void)Decode(sibling, 3, false);
    restore(*slots[2]);
    // Retained checkpoint bytes survive divergent writes and source release.
    Require(adapter.Invalidate(*slots[0]), "source release failed");
    restore(*slots[1]);
    pieces.clear();
    for (auto& slot : slots)
      Require(adapter.Invalidate(*slot), "reference final reset failed");
    Require(pool->free_blocks() == pool->blocks(),
            "retained GPU references leaked blocks");
    std::cout << (mtp ? "MTP" : "AR") << ",shared_references," << boundary
              << ",retained=" << retained << ",PASS" << std::endl;
  }
}

void CheckLearnedDraftPolicy(const std::shared_ptr<qfn::Model>& model) {
  namespace qt = gufo::tokenization;
  gufo::hip::TransferPool streams(2);
  qfn::ContinuationAdapter adapter(model, gufo::core::SessionMode::kSpeculative,
                                   4096, {'p', 'o', 'l', 'i', 'c', 'y'});
  Guard source_guard, destination_guard;
  auto source = adapter.CreateSlot(source_guard);
  auto destination = adapter.CreateSlot(destination_guard);
  auto& live = adapter.GetSession(*source);
  auto& restored = adapter.GetSession(*destination);
  const std::array messages{
      qt::ChatMessage{qt::ChatRole::kUser,
                      "Write a Python LRU cache implementation with get, "
                      "put, eviction and tests. Explain the edge cases."}};
  std::string error;
  const auto encoded = qt::QwenChatTemplate::RenderAndTokenize(
      model->tokenizer(), messages, {.enable_thinking = false}, &error);
  Require(encoded.has_value(), error);
  const std::vector<std::int32_t> prompt(encoded->begin(), encoded->end());
  Require(live.Sync(prompt, &error) && restored.Sync(prompt, &error), error);
  sampling::SamplerState sampler(
      {.temperature = 0.8F, .top_k = 20, .top_p = 0.95F, .seed = 73},
      std::vector<sampling::TokenId>(prompt.begin(), prompt.end()));
  auto stale_sampler = sampler;
  (void)gufo::test::DecodeMtpReplay(restored, stale_sampler, 32);
  (void)gufo::test::DecodeMtpReplay(live, sampler, 128);
  const auto checkpoint = Capture(adapter, *source, streams);
  Load(adapter, *destination, checkpoint, streams);
  Require(adapter.Validate(*destination, checkpoint.positions),
          "learned MTP policy checkpoint rejected");
  SameLogits(live, restored);
  auto replay_sampler = sampler;
  Require(gufo::test::DecodeMtpReplay(live, sampler, 64) ==
              gufo::test::DecodeMtpReplay(restored, replay_sampler, 64),
          "component restore lost learned MTP acceptance history");
  SameLogits(live, restored);
  std::cout << "MTP,learned_draft_policy,PASS\n";
}
}  // namespace
int main(int argc, char** argv) {
  const bool reservation_only =
      argc == 6 && std::string_view(argv[5]) == "--shared-reservation-only";
  const bool shared_kvarn_only =
      argc == 6 &&
      (std::string_view(argv[5]) == "--shared-kvarn-only" ||
       std::string_view(argv[5]) == "--shared-k6v6-only" ||
       std::string_view(argv[5]) == "--shared-fp16-only" || reservation_only);
  const int kvarn_bits =
      argc == 6 && std::string_view(argv[5]) == "--shared-k6v6-only" ? 6 : 4;
  const bool guard_only =
      argc == 6 && std::string_view(argv[5]) == "--guard-only";
  const bool image_only =
      argc == 6 && std::string_view(argv[5]) == "--image-only";
  const bool in_pass_only =
      argc == 6 && std::string_view(argv[5]) == "--in-pass-only";
  const bool policy_only =
      argc == 6 && std::string_view(argv[5]) == "--policy-only";
  const bool reuse_only =
      argc == 6 && std::string_view(argv[5]) == "--reuse-only";
  if ((argc != 5 && !guard_only && !image_only && !in_pass_only &&
       !policy_only && !reuse_only && !shared_kvarn_only) ||
      std::string_view(argv[1]) != "--model" ||
      std::string_view(argv[3]) != "--mtp-model") {
    std::cerr << "Usage: continuation_adapter_test --model FIRST.gguf "
                 "--mtp-model MTP.gguf "
                 "[--guard-only|--image-only|--in-pass-only|--policy-only|"
                 "--reuse-only|--shared-kvarn-only]\n";
    return 77;
  }
  try {
    std::string error;
    auto model = qfn::Model::Load(
        argv[2],
        {.max_context = 16384,
         .mtp_model_path = argv[4],
         .kvarn = shared_kvarn_only &&
                  std::string_view(argv[5]) != "--shared-fp16-only",
         .kvarn_bits = kvarn_bits},
        &error);
    Require(bool(model), error);
    if (policy_only) {
      CheckLearnedDraftPolicy(model);
      return 0;
    }
    if (reuse_only) {
      CheckSlotReuse(model, false);
      CheckSlotReuse(model, true);
      return 0;
    }
    const auto pattern = model->Tokenize(
        "The quick brown fox jumps over the lazy dog. Strix Halo executes this "
        "deterministic benchmark sequence. ");
    Require(!pattern.empty(), "empty prompt pattern");
    std::vector<std::int32_t> prompt(10000);
    for (std::size_t i = 0; i < prompt.size(); ++i)
      prompt[i] = pattern[i % pattern.size()];
    if (shared_kvarn_only) {
      if (model->Kvarn() && !reservation_only) {
        CheckSharedKvarn(model, false, prompt);
        CheckSharedKvarn(model, true, prompt);
      }
      CheckSharedReferences(model, false, prompt, reservation_only);
      CheckSharedReferences(model, true, prompt, reservation_only);
      return 0;
    }
    // The repeated prefix can accept every draft. End the
    // sampled boundary with real code generation to exercise rejection and
    // residual replay as well as the long recurrent/KV state.
    namespace qt = gufo::tokenization;
    const std::array messages{
        qt::ChatMessage{qt::ChatRole::kUser,
                        "Write a Python LRU cache implementation with get, "
                        "put, eviction and tests. Explain the edge cases."}};
    const auto sampled_tail = qt::QwenChatTemplate::RenderAndTokenize(
        model->tokenizer(), messages, {.enable_thinking = false}, &error);
    Require(sampled_tail && sampled_tail->size() < 8203, error);
    std::copy(sampled_tail->begin(), sampled_tail->end(),
              prompt.begin() + 8203 - sampled_tail->size());
    if (!image_only && !in_pass_only) {
      CheckTransferAndGuardLifetime(model, false, prompt);
      CheckTransferAndGuardLifetime(model, true, prompt);
    }
    if (guard_only)
      return 0;
    if (!image_only) {
      CheckInPass(model, false, prompt);
      CheckInPass(model, true, prompt);
    }
    if (in_pass_only)
      return 0;
    if (!image_only) {
      Check(model, false, prompt);
      Check(model, true, prompt);
      CheckLearnedDraftPolicy(model);
      CheckSlotReuse(model, false);
      CheckSlotReuse(model, true);
    }
    CheckImages(model, false);
    CheckImages(model, true);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
