#include "src/models/qwen/continuation_adapter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/cache/slot.hpp"
#include "src/core/gguf_reader.hpp"

namespace cache = gufo::cache;
namespace hip = gufo::hip;
namespace qwen = gufo::models::qwen;
namespace vision = qwen::vision;
using Token = gufo::tokenization::TokenId;
void FailNextResetSubmission();
void FailNextTransferWait();
namespace {
void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
template<class F>
void Reject(F&& fn) {
  bool rejected = false;
  try {
    fn();
  } catch (const std::exception&) {
    rejected = true;
  }
  Require(rejected, "invalid/busy operation was accepted");
}
void Wait(cache::Completion completion) {
  Require(completion.Wait() == cache::TransferResult::kSucceeded,
          "transfer failed");
}
struct Checkpoint {
  std::vector<cache::ComponentPosition> positions;
  std::vector<std::vector<std::byte>> bytes;
};
Checkpoint Capture(qwen::ContinuationAdapter& adapter, cache::Slot& slot,
                   hip::TransferPool& pool, std::byte fill = std::byte{}) {
  Checkpoint checkpoint{adapter.Positions(slot), {}};
  checkpoint.bytes.resize(adapter.Components().size());
  for (std::size_t i = 0; i < checkpoint.bytes.size(); ++i) {
    const auto& descriptor = adapter.Components()[i];
    auto& bytes = checkpoint.bytes[i];
    bytes.assign(descriptor.row_bytes
                     ? checkpoint.positions[i].valid_rows * descriptor.row_bytes
                     : descriptor.state_bytes,
                 fill);
    auto stream = pool.TryAcquire();
    Require(bool(stream), "transfer pool did not recycle");
    if (descriptor.row_bytes)
      Wait(adapter.CopyRowsOut(slot, descriptor.id, 0,
                               checkpoint.positions[i].valid_rows, bytes,
                               *stream));
    else
      Wait(adapter.CapturePrivate(slot, descriptor.id, bytes, *stream));
  }
  return checkpoint;
}
void Load(qwen::ContinuationAdapter& adapter, cache::Slot& slot,
          const Checkpoint& checkpoint, hip::TransferPool& pool,
          bool missing = false, cache::Slot* borrowed_source = nullptr) {
  adapter.BeginRestore(slot, checkpoint.positions);
  for (std::size_t i = 0; i < checkpoint.bytes.size(); ++i) {
    const auto& descriptor = adapter.Components()[i];
    const auto bytes = std::span(checkpoint.bytes[i]);
    if (descriptor.row_bytes) {
      const auto rows = checkpoint.positions[i].valid_rows;
      const auto cut = std::min<cache::Rows>(17, rows);
      auto first = pool.TryAcquire();
      auto second = pool.TryAcquire();
      Require(first && second, "independent row streams unavailable");
      const auto borrowed =
          borrowed_source
              ? adapter.BorrowRowsView(*borrowed_source, descriptor.id, 0, rows)
              : std::span<const std::byte>{};
      if (borrowed_source)
        Require(borrowed.size() == bytes.size(),
                "borrowed KV view is incomplete");
      const auto source = borrowed_source ? borrowed : bytes;
      auto tail = adapter.CopyRowsIn(slot, descriptor.id, cut, rows,
                                     source.subspan(cut * descriptor.row_bytes),
                                     *first);
      auto head =
          adapter.CopyRowsIn(slot, descriptor.id, 0, cut,
                             source.first(cut * descriptor.row_bytes), *second);
      Wait(std::move(head));
      Wait(std::move(tail));
    } else {
      const auto end = bytes.size() - (missing && i == 1 ? 1 : 0);
      const auto cut = std::min<std::size_t>(37, end);
      auto first = pool.TryAcquire();
      auto second = pool.TryAcquire();
      Require(first && second, "independent private streams unavailable");
      auto tail = adapter.LoadPrivatePiece(
          slot, descriptor.id, cut, bytes.subspan(cut, end - cut), *first);
      auto head = adapter.LoadPrivatePiece(slot, descriptor.id, 0,
                                           bytes.first(cut), *second);
      Wait(std::move(head));
      Wait(std::move(tail));
    }
  }
}
struct Guard final : cache::MutationGuard {
  qwen::ContinuationAdapter* adapter{};
  cache::Slot* slot{};
  hip::TransferPool* pool{};
  const Checkpoint* borrowed{};
  std::vector<bool> detached;
  void Borrow(const Checkpoint& checkpoint) {
    borrowed = &checkpoint;
    detached.assign(checkpoint.bytes.size(), false);
  }
  void BeforeOverwrite(cache::ComponentId id, cache::Rows first,
                       cache::Rows end) override {
    if (refuse)
      throw std::runtime_error("guard refusal fixture");
    if (!borrowed || detached[id.value - 1] ||
        first >= borrowed->positions[id.value - 1].valid_rows || !end)
      return;
    const auto& descriptor = adapter->Components()[id.value - 1];
    if (!descriptor.row_bytes)
      return;
    auto stream = pool->TryAcquire();
    Require(bool(stream), "preservation stream unavailable");
    auto bytes = borrowed->bytes[id.value - 1];
    Wait(adapter->CopyRowsOut(*slot, id, 0,
                              borrowed->positions[id.value - 1].valid_rows,
                              bytes, *stream));
    Require(bytes == borrowed->bytes[id.value - 1],
            "borrowed bytes changed before preservation");
    detached[id.value - 1] = true;
  }
  void BeforeRelease(cache::ComponentId id, cache::Rows first,
                     cache::Rows end) noexcept override {
    try {
      BeforeOverwrite(id, first, end);
    } catch (...) {
      release_failed = true;
    }
  }
  void AfterReset() noexcept override { borrowed = nullptr; }
  void Preserved() {
    Require(!release_failed, "borrower release failed");
    for (std::size_t i = 0; i < detached.size(); ++i)
      if (adapter->Components()[i].row_bytes &&
          borrowed->positions[i].valid_rows)
        Require(detached[i], "mutation missed a borrowed component");
    borrowed = nullptr;
  }
  void Untouched() {
    Require(!release_failed, "borrower release failed");
    for (std::size_t i = 0; i < detached.size(); ++i)
      Require(!detached[i], "rollback preserved unchanged rows");
    borrowed = nullptr;
  }
  bool release_failed{};
  bool refuse{};
};
std::vector<float> Logits(hip::QwenGpuExecutor& executor) {
  auto values = executor.CopyLastLogits();
  Require(std::ranges::all_of(values, [](float x) { return std::isfinite(x); }),
          "nonfinite logits");
  return {values.begin(), values.end()};
}
void SameLogits(hip::QwenGpuExecutor& a, hip::QwenGpuExecutor& b) {
  const auto first = Logits(a), second = Logits(b);
  Require(first.size() == second.size() &&
              std::memcmp(first.data(), second.data(),
                          first.size() * sizeof(float)) == 0,
          "restored logits differ bytewise");
}
void CheckLeasedGuard(const std::shared_ptr<const hip::QwenGpuModel>& model) {
  qwen::ContinuationAdapter adapter(model, 64, {'l', 'e', 'a', 's', 'e'});
  hip::TransferPool pool(4);
  constexpr std::size_t budget = 8ULL << 20;
  cache::ResourceLedger ledger({budget, budget, 0, budget});
  const std::array<Token, 17> prompt{42, 43, 44, 45, 46, 47, 48, 49, 50,
                                     51, 52, 53, 54, 55, 56, 57, 58};
  for (int path = 0; path < 4; ++path) {
    auto facade_stream = pool.TryAcquire();
    cache::LeasedSlot facade(
        ledger, adapter, *facade_stream, {static_cast<std::uint64_t>(path + 1)},
        cache::IdleSpillConfig{4096,
                               [&]() -> std::unique_ptr<cache::Stream> {
                                 return pool.TryAcquire();
                               },
                               [] { return false; }});
    auto lease = facade.Acquire();
    auto& executor = adapter.GetExecutor(lease.Execution());
    (void)executor.ForwardPromptBatch(prompt);
    const auto positions = adapter.Positions(lease.Execution());
    auto legacy = executor.SaveSnapshot(prompt.size());
    std::vector<cache::ResourceCharge> charges;
    std::vector<std::shared_ptr<cache::BorrowedRows>> borrowed;
    std::vector<std::vector<std::byte>> expected;
    for (const auto& descriptor : adapter.Components()) {
      if (!descriptor.row_bytes)
        continue;
      const auto rows = positions[descriptor.id.value - 1].valid_rows;
      auto& oracle = expected.emplace_back(rows * descriptor.row_bytes);
      auto stream = pool.TryAcquire();
      Wait(adapter.CopyRowsOut(lease.Execution(), descriptor.id, 0, rows,
                               oracle, *stream));
      auto reservation =
          ledger.Reserve(cache::ResourceCategory::kBackingFree, oracle.size());
      auto backing = std::make_shared<std::vector<std::byte>>(oracle.size());
      auto charge = reservation.Convert();
      auto assigned =
          charge.ReserveBacking(cache::ResourceCategory::kBackingAssigned);
      charges.push_back(charge);
      borrowed.push_back(lease.Borrow(descriptor.id, 0, rows,
                                      std::move(assigned), backing, *backing));
    }
    if (path == 0)
      executor.Reset();
    if (path == 1)
      executor.RestoreSnapshot(*legacy);
    if (path == 2)
      adapter.BeginRestore(lease.Execution(), positions);
    if (path == 3) {
      executor.Reset();
      (void)executor.ForwardPromptBatch(std::span(prompt).first(8));
    }
    for (std::size_t i = 0; i < borrowed.size(); ++i) {
      Require(borrowed[i]->IsValid() && !borrowed[i]->Location(),
              "real guard left rows borrowed");
      auto owner = std::static_pointer_cast<const std::vector<std::byte>>(
          borrowed[i]->Owner());
      Require(owner && *owner == expected[i],
              "real guard preserved changed bytes");
    }
  }
  std::cout << "real_leased_guard,PASS\n";
}
void Check(const std::shared_ptr<const hip::QwenGpuModel>& model, bool fp16,
           bool shared_scratch = false) {
  const std::string label = std::string(fp16 ? "FP16" : "FP32") +
                            (shared_scratch ? "-shared-scratch" : "");
  std::cout << label << ",boundaries" << std::endl;
  hip::QwenExecutionPolicy policy;
  policy.kv_cache_storage =
      fp16 ? hip::QwenKvCacheStorage::kFp16 : hip::QwenKvCacheStorage::kFp32;
  constexpr std::uint32_t kContext = 12288;
  qwen::ContinuationAdapter adapter(
      model, kContext, {'q', 'w', 'e', 'n'}, policy, nullptr, {}, nullptr,
      shared_scratch ? std::make_shared<hip::QwenSharedPassScratch>(
                           model->GetConfig(), kContext)
                     : nullptr);
  hip::TransferPool pool(2);
  Guard ga, gb;
  auto source = adapter.CreateSlot(ga), destination = adapter.CreateSlot(gb);
  ga.adapter = gb.adapter = &adapter;
  ga.slot = source.get();
  gb.slot = destination.get();
  ga.pool = gb.pool = &pool;
  auto& a = adapter.GetExecutor(*source);
  auto& b = adapter.GetExecutor(*destination);
  const auto text = model->GetTokenizer().Encode(
      "Memory pages map virtual addresses to physical memory. A page fault "
      "asks the operating system to provide a missing page.\n");
  std::vector<Token> tokens(9000);
  for (std::size_t i = 0; i < tokens.size(); ++i)
    tokens[i] = text[i % text.size()];
  gufo::models::GenerationOptions generation;
  generation.max_new_tokens = 4;
  generation.sampling.temperature = 0;
  auto history = std::vector<Token>(tokens.begin(), tokens.begin() + 64);
  history.reserve(history.size() + generation.max_new_tokens);
  const auto generated = a.Generate(history, generation);
  history.insert(history.end(), generated.begin(), generated.end());
  const auto generated_frontier = Capture(adapter, *source, pool);
  Load(adapter, *destination, generated_frontier, pool);
  Require(adapter.Validate(*destination, generated_frontier.positions),
          "generated frontier restore rejected");
  Require(a.GenerateFromPrefix(history, history.size(), generation) ==
              b.GenerateFromPrefix(history, history.size(), generation),
          "cached generation frontier changed after restore");
  for (auto boundary : {257U, 8203U}) {
    a.Reset();
    b.Reset();
    (void)a.ForwardPromptBatch(std::span(tokens).first(boundary));
    const auto checkpoint = Capture(adapter, *source, pool);
    Load(adapter, *destination, checkpoint, pool);
    Reject([&] { (void)b.ForwardToken(tokens[boundary], boundary); });
    Require(adapter.Validate(*destination, checkpoint.positions),
            "complete restore rejected");
    Require(Capture(adapter, *destination, pool).bytes == checkpoint.bytes,
            "restored component bytes differ");
    SameLogits(a, b);
    for (std::uint32_t i = 0; i < 12; ++i) {
      const auto first = a.ForwardToken(tokens[boundary + i], boundary + i);
      const auto second = b.ForwardToken(tokens[boundary + i], boundary + i);
      Require(first == second, "next token differs");
      SameLogits(a, b);
    }
  }
  a.Reset();
  b.Reset();
  (void)a.ForwardPromptBatch(std::span(tokens).first(257));
  const auto earlier = Capture(adapter, *source, pool);
  (void)a.ForwardPromptBatch(std::span(tokens).subspan(257, 1025), 257);
  auto edited = tokens;
  edited[258] = edited[258] == 1 ? 2 : 1;
  Load(adapter, *destination, earlier, pool);
  Require(adapter.Validate(*destination, earlier.positions),
          "earlier frontier rejected");
  a.Reset();
  (void)a.ForwardPromptBatch(std::span(edited).first(257));
  (void)a.ForwardPromptBatch(std::span(edited).subspan(257, 1025), 257);
  (void)b.ForwardPromptBatch(std::span(edited).subspan(257, 1025), 257);
  SameLogits(a, b);
  const auto stable = Capture(adapter, *source, pool);
  for (int mutation = 0; mutation < 5; ++mutation) {
    Load(adapter, *source, stable, pool);
    Require(adapter.Validate(*source, stable.positions),
            "guard source restore failed");
    const auto legacy = a.SaveSnapshot(1282);
    ga.Borrow(stable);
    if (mutation == 0)
      a.Reset();
    if (mutation == 1)
      a.RestoreSnapshot(*legacy);
    if (mutation == 2) {
      std::vector<std::uint8_t> payload(legacy->CompactPayloadBytes());
      (void)legacy->SerializeCompact(payload);
      a.RestoreCompactSnapshot(payload, 1282);
    }
    if (mutation == 3) {
      // Rollback swaps recurrent state only: rows below the saved position
      // stay borrowed, which keeps disk-only checkpoints publishable.
      a.SaveState(1282);
      Reject([&] { (void)Capture(adapter, *source, pool); });
      (void)a.ForwardToken(tokens[1282], 1282);
      a.RestoreState();
      a.FinishVerification();
      ga.Untouched();
      continue;
    }
    if (mutation == 4)
      (void)a.ForwardPromptBatch(std::span(edited).subspan(5, 7), 5);
    ga.Preserved();
  }
  Load(adapter, *source, stable, pool);
  Require(adapter.Validate(*source, stable.positions),
          "pending source restore failed");
  const auto& descriptor = adapter.Components()[2];
  auto stream = pool.TryAcquire();
  auto bytes = stable.bytes[2];
  auto read = adapter.CapturePrivate(*source, descriptor.id, bytes, *stream);
  Reject([&] { (void)a.ForwardToken(tokens[1282], 1282); });
  Require(!adapter.Invalidate(*source), "invalidation ignored a pending read");
  Wait(std::move(read));
  stream.reset();
  Load(adapter, *destination, stable, pool, true);
  Require(!adapter.Validate(*destination, stable.positions),
          "incomplete private load accepted");
  Require(!adapter.Validate(*destination, stable.positions),
          "restore failure was not latched");
  Require(adapter.Invalidate(*destination), "failed restore could not recover");
  Reject([&] { (void)adapter.Validate(*destination, stable.positions); });
  Require(destination->IsValid(), "out-of-restore validation changed validity");
  for (bool move_assignment : {false, true}) {
    Load(adapter, *destination, stable, pool);
    hip::TransferPool failure_pool(2);
    auto stream = failure_pool.TryAcquire();
    auto load =
        adapter.LoadPrivatePiece(*destination, adapter.Components()[2].id, 0,
                                 std::span(stable.bytes[2]).first(1), *stream);
    FailNextTransferWait();
    if (move_assignment) {
      auto replacement_stream = failure_pool.TryAcquire();
      auto replacement_bytes = stable.bytes[3];
      auto replacement =
          adapter.CapturePrivate(*source, adapter.Components()[3].id,
                                 replacement_bytes, *replacement_stream);
      load = std::move(replacement);
      Wait(std::move(load));
    }
    // Destruction also settles the load, retaining its failure in the slot.
    else {
      auto discarded = std::move(load);
    }
    Require(!adapter.Validate(*destination, stable.positions),
            "discarded load error was not latched");
    Require(adapter.Invalidate(*destination),
            "discarded load error did not recover");
  }
  FailNextResetSubmission();
  Require(!adapter.Invalidate(*destination) && !destination->IsValid(),
          "reset submission failure exposed valid state");
  Reject([&] { (void)b.ForwardToken(tokens[0], 0); });
  Require(adapter.Invalidate(*destination), "reset failure did not recover");
  // Batched native writes preserve both sessions before touching the
  // coordinator stream.
  Load(adapter, *source, earlier, pool);
  Require(adapter.Validate(*source, earlier.positions),
          "batch source restore failed");
  Load(adapter, *destination, earlier, pool);
  Require(adapter.Validate(*destination, earlier.positions),
          "batch destination restore failed");
  ga.Borrow(earlier);
  gb.Borrow(earlier);
  const std::array<hip::QwenGpuBatchItem, 2> batch{
      {{&a, tokens[0], 0}, {&b, tokens[0], 0}}};
  gb.refuse = true;
  Reject([&] { (void)hip::QwenGpuExecutor::ForwardTokenBatch(batch); });
  gb.refuse = false;
  Require(source->IsValid() && destination->IsValid(),
          "guard refusal invalidated an untouched batch peer");
  Require(Capture(adapter, *source, pool).bytes == earlier.bytes &&
              Capture(adapter, *destination, pool).bytes == earlier.bytes,
          "guard refusal changed batch peer bytes");
  (void)hip::QwenGpuExecutor::ForwardTokenBatch(batch);
  ga.Preserved();
  gb.Preserved();
  Guard lifetime_guard;
  auto temporary = adapter.CreateSlot(lifetime_guard);
  Load(adapter, *temporary, earlier, pool);
  Require(adapter.Validate(*temporary, earlier.positions),
          "lifetime restore failed");
  auto lifetime_stream = pool.TryAcquire();
  auto lifetime_bytes = earlier.bytes[2];
  auto lifetime_copy = adapter.CapturePrivate(
      *temporary, adapter.Components()[2].id, lifetime_bytes, *lifetime_stream);
  temporary.reset();
  Wait(std::move(lifetime_copy));
  Require(lifetime_bytes == earlier.bytes[2],
          "slot destruction changed pending capture");
  std::cout << label << ",restore_edit_guard,PASS\n";
}
// Slots of one adapter share a prompt-pass scratch: another slot's prefill
// between a prefill and its decode must not change any logits, and each
// session keeps only decode-step scratch rows of its own.
void CheckSharedPassScratch(
    const std::shared_ptr<const hip::QwenGpuModel>& model) {
  std::cout << "shared_pass_scratch" << std::endl;
  constexpr std::uint32_t kContext = 8192;
  auto scratch = std::make_shared<hip::QwenSharedPassScratch>(
      model->GetConfig(), kContext);
  qwen::ContinuationAdapter own(model, kContext, {'o', 'w', 'n'});
  qwen::ContinuationAdapter leased(model, kContext, {'s', 'h', 'r'}, {},
                                   nullptr, {}, nullptr, scratch);
  Guard gr, ga, gb;
  auto reference = own.CreateSlot(gr);
  auto first = leased.CreateSlot(ga), second = leased.CreateSlot(gb);
  auto& r = own.GetExecutor(*reference);
  auto& a = leased.GetExecutor(*first);
  auto& b = leased.GetExecutor(*second);
  Require(scratch->rows() == r.GetMaxPromptBatch() &&
              a.GetMemoryUsage().temporary_scratch_bytes <
                  r.GetMemoryUsage().temporary_scratch_bytes,
          "sessions sharing a pass scratch still hold a whole one");
  const auto text = model->GetTokenizer().Encode(
      "A cache line holds bytes fetched together from memory. Eviction picks "
      "a line to replace when the set is full.\n");
  std::vector<Token> tokens(4600);
  for (std::size_t i = 0; i < tokens.size(); ++i)
    tokens[i] = text[(i * 7) % text.size()];
  // Two prompt passes (4096 + 404 rows), then a peer's pass in between.
  (void)r.ForwardPromptBatch(std::span(tokens).first(4500));
  (void)a.ForwardPromptBatch(std::span(tokens).first(4500));
  (void)b.ForwardPromptBatch(std::span(tokens).subspan(100, 333));
  SameLogits(r, a);
  for (std::uint32_t i = 0; i < 8; ++i) {
    const auto expected = r.ForwardToken(tokens[4500 + i], 4500 + i);
    Require(a.ForwardToken(tokens[4500 + i], 4500 + i) == expected,
            "shared pass scratch changed a decode token");
    SameLogits(r, a);
  }
  const auto hidden = r.CopyLastHidden();
  const std::vector<float> expected(hidden.begin(), hidden.end());
  Require(std::ranges::equal(expected, a.CopyLastHidden()),
          "shared pass scratch changed the last hidden row");
  std::cout << "shared_pass_scratch,PASS\n";
}

// Slots of one adapter take FP16 rows from one shared KV block pool as they
// write. Interleaved sessions get scattered blocks; logits, decode tokens,
// batched decode across block boundaries, captured bytes and a restore into
// another slot must equal sessions holding their whole context. A pass the
// pool cannot map is refused before it writes; reservations hold blocks back
// and a reset returns a session's blocks.
void CheckSharedKvPool(const std::shared_ptr<const hip::QwenGpuModel>& model) {
  std::cout << "shared_kv_pool" << std::endl;
  constexpr std::uint32_t kContext = 8192;
  const auto& config = model->GetConfig();
  const hip::KvBlockPlane rows{
      .layers = config.FullAttentionLayerCount(),
      .row_bytes = static_cast<std::size_t>(config.num_key_value_heads) *
                   config.head_dim * sizeof(std::uint16_t)};
  // Fewer blocks than two whole contexts (2 x 32).
  constexpr std::uint32_t kPoolBlocks = 40;
  auto kv_pool =
      std::make_shared<hip::KvBlockPool>(std::vector{rows, rows}, kPoolBlocks);
  qwen::ContinuationAdapter own(model, kContext, {'o', 'w', 'n'});
  qwen::ContinuationAdapter paged(model, kContext, {'p', 'k', 'v'}, {}, nullptr,
                                  {}, nullptr, nullptr, kv_pool);
  hip::TransferPool pool(2);
  Guard g1, g2, g3, g4, g5;
  auto r1 = own.CreateSlot(g1), r2 = own.CreateSlot(g2);
  auto p1 = paged.CreateSlot(g3), p2 = paged.CreateSlot(g4);
  auto& ra = own.GetExecutor(*r1);
  auto& rb = own.GetExecutor(*r2);
  auto& pa = paged.GetExecutor(*p1);
  auto& pb = paged.GetExecutor(*p2);
  Require(paged.KvPool() == kv_pool && paged.KvBlocks(*p1).empty() &&
              kv_pool->free_blocks() == kPoolBlocks,
          "a shared-pool session starts with rows mapped");
  Require(pa.GetMemoryUsage().request_state_bytes +
                  hip::KvBlockGeometry({rows, rows},
                                       hip::KvBlockPool::BlocksFor(kContext))
                      .bytes() ==
              ra.GetMemoryUsage().request_state_bytes,
          "a shared-pool session still claims its whole KV");
  const auto text = model->GetTokenizer().Encode(
      "A page table maps virtual pages to frames. The translation buffer "
      "caches recent mappings so most loads skip the walk.\n");
  std::vector<Token> tokens(6000);
  for (std::size_t i = 0; i < tokens.size(); ++i)
    tokens[i] = text[(i * 5) % text.size()];
  const auto other = std::span(tokens).subspan(100, 500);
  // The second session writes first, so the first one's rows start at pool
  // block 2 and its blocks are not where a whole-context arena puts them.
  (void)rb.ForwardPromptBatch(other);
  (void)pb.ForwardPromptBatch(other);
  constexpr std::uint32_t kPrompt = 4605;
  (void)ra.ForwardPromptBatch(std::span(tokens).first(kPrompt));
  (void)pa.ForwardPromptBatch(std::span(tokens).first(kPrompt));
  const auto blocks = paged.KvBlocks(*p1);
  Require(blocks.size() == hip::KvBlockPool::BlocksFor(kPrompt) &&
              blocks.front() == 2,
          "interleaved sessions did not get scattered pool blocks");
  Require(kv_pool->free_blocks() == kPoolBlocks - blocks.size() - 2,
          "mapped blocks do not match the rows written");
  SameLogits(ra, pa);
  SameLogits(rb, pb);
  // Single decode across the block boundary at 4608.
  std::uint32_t pos = kPrompt;
  for (; pos < kPrompt + 8; ++pos) {
    Require(
        pa.ForwardToken(tokens[pos], pos) == ra.ForwardToken(tokens[pos], pos),
        "shared KV pool changed a decode token");
    SameLogits(ra, pa);
  }
  // Batched decode; the second session crosses its block boundary at 512.
  std::uint32_t other_pos = other.size();
  for (int step = 0; step < 16; ++step, ++pos, ++other_pos) {
    const std::array<hip::QwenGpuBatchItem, 2> expected_batch{
        {{&ra, tokens[pos], pos}, {&rb, tokens[other_pos], other_pos}}};
    const std::array<hip::QwenGpuBatchItem, 2> paged_batch{
        {{&pa, tokens[pos], pos}, {&pb, tokens[other_pos], other_pos}}};
    Require(hip::QwenGpuExecutor::ForwardTokenBatch(paged_batch) ==
                hip::QwenGpuExecutor::ForwardTokenBatch(expected_batch),
            "shared KV pool changed a batched decode token");
    SameLogits(ra, pa);
    SameLogits(rb, pb);
  }
  // Captured rows read through the table; a restore into the other slot maps
  // its own blocks and continues exactly.
  const auto expected = Capture(own, *r1, pool);
  const auto checkpoint = Capture(paged, *p1, pool);
  Require(checkpoint.bytes == expected.bytes,
          "shared-pool rows read back differently");
  Load(paged, *p2, checkpoint, pool);
  Require(paged.Validate(*p2, checkpoint.positions),
          "restore into a shared-pool slot rejected");
  Require(Capture(paged, *p2, pool).bytes == expected.bytes,
          "restored shared-pool bytes differ");
  Require(paged.KvBlocks(*p2).size() == hip::KvBlockPool::BlocksFor(pos),
          "restore mapped other than its rows");
  Require(
      pb.ForwardToken(tokens[pos], pos) == ra.ForwardToken(tokens[pos], pos),
      "restored shared-pool session changed a decode token");
  SameLogits(ra, pb);
  // Two sessions of 19 blocks leave 2: a pass needing 4 is refused before
  // it writes, and the session stays usable.
  auto p3 = paged.CreateSlot(g5);
  auto& pc = paged.GetExecutor(*p3);
  const auto free = kv_pool->free_blocks();
  Require(free == kPoolBlocks - 2 * hip::KvBlockPool::BlocksFor(pos + 1),
          "pool accounting after restore");
  Reject([&] { (void)pc.ForwardPromptBatch(std::span(tokens).first(1000)); });
  Require(p3->IsValid() && kv_pool->free_blocks() == free &&
              paged.KvBlocks(*p3).empty(),
          "a refused pass took blocks or invalidated its session");
  // A reservation holds blocks back from other sessions' passes.
  Reject([&] { (void)paged.ReserveKvRows(*p3, 300, 1); });
  Require(kv_pool->reserved_blocks() == 0 && kv_pool->free_blocks() == free,
          "a refused reservation held blocks");
  Require(paged.ReserveKvRows(*p3, 256, 0) == 1 &&
              kv_pool->free_blocks() == free - 1,
          "reservation did not hold its block");
  Reject([&] {
    (void)pa.ForwardPromptBatch(std::span(tokens).subspan(pos, 800), pos);
  });
  (void)pc.ForwardPromptBatch(std::span(tokens).first(256));
  Require(kv_pool->reserved_blocks() == 0 && paged.KvBlocks(*p3).size() == 1 &&
              kv_pool->free_blocks() == free - 1,
          "a reserved session did not map from its reservation");
  Require(paged.ReleaseKvReservation(*p3) == 0,
          "a used reservation still held blocks");
  // Reset returns a session's blocks.
  Require(paged.Invalidate(*p1) && paged.KvBlocks(*p1).empty() &&
              kv_pool->free_blocks() ==
                  free - 1 + hip::KvBlockPool::BlocksFor(pos + 1),
          "reset kept shared-pool blocks");
  std::cout << "shared_kv_pool,PASS\n";
}

void CheckRetainedKvBlocks(
    const std::shared_ptr<const hip::QwenGpuModel>& model) {
  std::cout << "retained_kv_blocks" << std::endl;
  constexpr std::uint32_t kContext = 8192;
  const auto& config = model->GetConfig();
  const hip::KvBlockPlane rows{
      .layers = config.FullAttentionLayerCount(),
      .row_bytes = static_cast<std::size_t>(config.num_key_value_heads) *
                   config.head_dim * sizeof(std::uint16_t)};
  constexpr std::uint32_t kPoolBlocks = 40;
  auto kv_pool =
      std::make_shared<hip::KvBlockPool>(std::vector{rows, rows}, kPoolBlocks);
  qwen::ContinuationAdapter own(model, kContext, {'o', 'w', 'n'});
  qwen::ContinuationAdapter paged(model, kContext, {'p', 'k', 'v'}, {}, nullptr,
                                  {}, nullptr, nullptr, kv_pool);
  hip::TransferPool pool(2);
  Guard g1, g2, g3;
  auto r1 = own.CreateSlot(g1);
  auto p1 = paged.CreateSlot(g2), p2 = paged.CreateSlot(g3);
  auto& ra = own.GetExecutor(*r1);
  auto& pa = paged.GetExecutor(*p1);
  auto& pb = paged.GetExecutor(*p2);
  const auto text = model->GetTokenizer().Encode(
      "A page table maps virtual pages to frames. The translation buffer "
      "caches recent mappings so most loads skip the walk.\n");
  std::vector<Token> tokens(4000);
  for (std::size_t i = 0; i < tokens.size(); ++i)
    tokens[i] = text[(i * 5) % text.size()];
  // Five complete blocks and a 20-row tail.
  constexpr std::uint32_t kPrompt = 1300;
  (void)ra.ForwardPromptBatch(std::span(tokens).first(kPrompt));
  (void)pa.ForwardPromptBatch(std::span(tokens).first(kPrompt));
  const auto expected = Capture(own, *r1, pool);
  // A checkpoint retains every complete K/V chunk of the first session.
  const auto components = paged.Components();
  std::vector<std::vector<cache::Adapter::RetainedRows>> retained(
      components.size());
  std::size_t key = components.size();
  for (std::size_t i = 0; i < components.size(); ++i) {
    const auto& d = components[i];
    Require(!own.RetainsRows(d.id), "an own-pool session retained rows");
    if (!paged.RetainsRows(d.id))
      continue;
    key = std::min(key, i);
    for (cache::Rows first = 0;
         first + hip::kKvBlockTokens <= expected.positions[i].valid_rows;
         first += hip::kKvBlockTokens) {
      auto chunk =
          paged.RetainRows(*p1, d.id, first, first + hip::kKvBlockTokens);
      Require(chunk.owner && chunk.bytes == hip::kKvBlockTokens * d.row_bytes,
              "a complete K/V chunk was not retained");
      retained[i].push_back(std::move(chunk));
    }
    Require(retained[i].size() == 5, "retained other than complete chunks");
  }
  Require(key < components.size(), "no component retains rows");
  Require(
      paged.RetainRows(*p1, components[key].id, 1280, 1300).owner == nullptr,
      "a partial chunk was retained");
  const std::vector<std::uint32_t> first_blocks(paged.KvBlocks(*p1).begin(),
                                                paged.KvBlocks(*p1).end());
  for (std::size_t k = 0; k < 5; ++k)
    Require(kv_pool->references(first_blocks[k]) == 2,
            "a retained block holds other than one reference");
  Require(!paged.OnlyRetained(retained[key][0].owner.get()),
          "a block its session maps counts as only retained");
  // Another session restores complete chunks from the retained blocks and
  // the tails from host bytes: it maps those blocks rather than copying.
  const auto free = kv_pool->free_blocks();
  paged.BeginRestore(*p2, expected.positions);
  // Each transfer leases its own stream, as the cache does.
  const auto stream = [&] {
    auto leased = pool.TryAcquire();
    Require(bool(leased), "restore stream unavailable");
    return leased;
  };
  for (std::size_t i = 0; i < components.size(); ++i) {
    const auto& d = components[i];
    const auto bytes = std::span(expected.bytes[i]);
    if (!d.row_bytes) {
      Wait(paged.LoadPrivate(*p2, d.id, bytes, *stream()));
      continue;
    }
    const auto valid = expected.positions[i].valid_rows;
    cache::Rows first = 0;
    for (const auto& chunk : retained[i]) {
      Wait(paged.CopyRowsIn(
          *p2, d.id, first, first + hip::kKvBlockTokens,
          {static_cast<const std::byte*>(chunk.owner.get()), chunk.bytes},
          *stream()));
      first += hip::kKvBlockTokens;
    }
    if (first < valid)
      Wait(paged.CopyRowsIn(*p2, d.id, first, valid,
                            bytes.subspan(first * d.row_bytes), *stream()));
  }
  Require(paged.Validate(*p2, expected.positions),
          "a mapped restore was rejected");
  for (std::size_t k = 0; k < 5; ++k)
    Require(paged.KvBlocks(*p2)[k] == first_blocks[k] &&
                kv_pool->references(first_blocks[k]) == 3,
            "a retained block was copied instead of mapped");
  Require(kv_pool->free_blocks() == free - 1,
          "a mapped restore took more than its tail block");
  Require(Capture(paged, *p2, pool).bytes == expected.bytes,
          "mapped restore bytes differ");
  std::uint32_t pos = kPrompt;
  Require(
      pb.ForwardToken(tokens[pos], pos) == ra.ForwardToken(tokens[pos], pos),
      "a mapped restore changed a decode token");
  SameLogits(ra, pb);
  ++pos;
  // Rewriting rows inside a retained block copies that block first; the
  // first session and the checkpoint keep their rows.
  const auto rewrite = std::span(tokens).subspan(3000, 100);
  (void)ra.ForwardPromptBatch(rewrite, 300);
  (void)pb.ForwardPromptBatch(rewrite, 300);
  SameLogits(ra, pb);
  Require(paged.KvBlocks(*p2)[1] != first_blocks[1] &&
              kv_pool->references(first_blocks[1]) == 2,
          "a write into a retained block did not copy it");
  Require(paged.KvBlocks(*p2)[0] == first_blocks[0] &&
              paged.KvBlocks(*p2)[2] == first_blocks[2],
          "a write copied blocks it does not touch");
  Require(Capture(paged, *p1, pool).bytes == expected.bytes,
          "a write changed rows another session holds");
  const auto row = components[key].row_bytes;
  std::vector<std::byte> held(retained[key][1].bytes);
  Require(hipMemcpy(held.data(), retained[key][1].owner.get(), held.size(),
                    hipMemcpyDeviceToHost) == hipSuccess,
          "retained rows unreadable");
  Require(std::ranges::equal(held, std::span(expected.bytes[key])
                                       .subspan(hip::kKvBlockTokens * row,
                                                hip::kKvBlockTokens * row)),
          "a write changed retained rows");
  // With both sessions reset only the checkpoint holds the blocks; dropping
  // it returns them.
  Require(paged.Invalidate(*p1) && paged.Invalidate(*p2),
          "reset of retaining sessions failed");
  Require(paged.OnlyRetained(retained[key][0].owner.get()),
          "a block no session maps is not only retained");
  retained.clear();
  Require(kv_pool->free_blocks() == kPoolBlocks,
          "dropped checkpoints kept their blocks");
  std::cout << "retained_kv_blocks,PASS\n";
}

void CheckBatchAdmission(
    const std::shared_ptr<const hip::QwenGpuModel>& model) {
  qwen::ContinuationAdapter adapter(model, 64, {'b', 'a', 't', 'c', 'h'});
  hip::TransferPool pool(2);
  Guard ga, gb;
  auto source = adapter.CreateSlot(ga), destination = adapter.CreateSlot(gb);
  auto& a = adapter.GetExecutor(*source);
  auto& b = adapter.GetExecutor(*destination);
  const std::array<Token, 3> prefix{42, 43, 44};
  (void)a.ForwardPromptBatch(prefix);
  (void)b.ForwardPromptBatch(prefix);
  const auto before_a = Capture(adapter, *source, pool);
  const auto before_b = Capture(adapter, *destination, pool);
  const std::array<hip::QwenGpuBatchItem, 2> batch{{{&a, 45, 3}, {&b, 45, 3}}};
  const std::array<Token, 2> tokens{45, 46};
  const std::array<hip::QwenGpuVerificationItem, 2> verification{
      {{&a, tokens, 3}, {&b, tokens, 3}}};
  gb.refuse = true;
  Reject([&] { (void)hip::QwenGpuExecutor::ForwardTokenBatch(batch); });
  Reject([&] {
    (void)hip::QwenGpuExecutor::ForwardVerificationBatch(verification);
  });
  gb.refuse = false;
  Require(source->IsValid() && destination->IsValid(),
          "batch admission invalidated untouched sessions");
  Require(Capture(adapter, *source, pool).bytes == before_a.bytes &&
              Capture(adapter, *destination, pool).bytes == before_b.bytes,
          "batch admission changed session bytes");
  (void)hip::QwenGpuExecutor::ForwardTokenBatch(batch);
  SameLogits(a, b);
  ga.adapter = gb.adapter = &adapter;
  ga.slot = source.get();
  gb.slot = destination.get();
  ga.pool = gb.pool = &pool;
  const auto borrowed_a = Capture(adapter, *source, pool);
  const auto borrowed_b = Capture(adapter, *destination, pool);
  ga.Borrow(borrowed_a);
  gb.Borrow(borrowed_b);
  const std::array<hip::QwenGpuVerificationItem, 2> overwrite{
      {{&a, tokens, 0}, {&b, tokens, 0}}};
  const auto predictions =
      hip::QwenGpuExecutor::ForwardVerificationBatch(overwrite);
  Require(predictions[0] == predictions[1], "verification batch peers differ");
  ga.Preserved();
  gb.Preserved();
  std::cout << "batch_admission,PASS\n";
}

void CheckMutationPaths(const std::shared_ptr<const hip::QwenGpuModel>& model) {
  const std::array<Token, 4> prefix{42, 43, 44, 45};
  const std::array<Token, 2> verified{46, 47};
  for (bool fp16 : {true, false}) {
    auto policy = hip::QwenExecutionPolicy::Production();
    if (!fp16)
      policy.kv_cache_storage = hip::QwenKvCacheStorage::kFp32;
    qwen::ContinuationAdapter adapter(model, 64, {'m', 'u', 't'}, policy);
    hip::TransferPool pool(2);
    Guard guard, other_guard;
    auto slot = adapter.CreateSlot(guard),
         other = adapter.CreateSlot(other_guard);
    auto& a = adapter.GetExecutor(*slot);
    auto& b = adapter.GetExecutor(*other);
    guard.adapter = &adapter;
    guard.slot = slot.get();
    guard.pool = &pool;
    (void)a.ForwardPromptBatch(prefix);
    const auto checkpoint = Capture(adapter, *slot, pool);
    // These calls intentionally overwrite existing KV rows; recurrence quality
    // after an edit is covered separately by restoring the exact boundary.
    for (bool commit : {false, true}) {
      Load(adapter, *slot, checkpoint, pool);
      Require(adapter.Validate(*slot, checkpoint.positions),
              "mutation restore rejected");
      guard.Borrow(checkpoint);
      if (commit)
        a.CommitVerificationChunk(verified, 0);
      else
        (void)a.ForwardToken(46, 0);
      guard.Preserved();
    }
    Load(adapter, *slot, checkpoint, pool);
    Require(adapter.Validate(*slot, checkpoint.positions),
            "replay source restore rejected");
    Load(adapter, *other, checkpoint, pool);
    Require(adapter.Validate(*other, checkpoint.positions),
            "replay control restore rejected");
    a.SaveState(4);
    (void)a.ForwardVerificationChunk(verified, 4);
    a.RestoreState();
    a.CommitVerificationChunk(std::span(verified).first(1), 4);
    a.FinishVerification();
    (void)b.ForwardToken(verified[0], 4, false);
    Require(Capture(adapter, *slot, pool).bytes ==
                Capture(adapter, *other, pool).bytes,
            "committed recurrent replay differs from scalar continuation");
  }
  std::cout << "scalar_commit_replay_guard,PASS\n";
}

std::shared_ptr<vision::Prompt> Image(bool blue) {
  auto prompt = std::make_shared<vision::Prompt>();
  vision::ImageGrid grid{0, 2, 2};
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
}
void CheckImages(const std::shared_ptr<const hip::QwenGpuModel>& model) {
  Require(bool(model->VisionEncoder()),
          "adjacent projector required for image qualification");
  qwen::ContinuationAdapter adapter(model, 64, {'i', 'm', 'g'});
  hip::TransferPool pool(2);
  Guard ga, gb;
  auto source = adapter.CreateSlot(ga), destination = adapter.CreateSlot(gb);
  auto& a = adapter.GetExecutor(*source);
  auto& b = adapter.GetExecutor(*destination);
  auto red = Image(false), blue = Image(true);
  a.ConfigureVision(red, model->VisionEncoder());
  (void)a.ForwardPromptBatch(std::span(red->tokens).first(2));
  const auto checkpoint = Capture(adapter, *source, pool);
  Load(adapter, *destination, checkpoint, pool);
  Require(!adapter.Validate(*destination, checkpoint.positions),
          "missing image attachment accepted");
  Require(adapter.Invalidate(*destination), "image reset failed");
  b.ConfigureVision(blue, model->VisionEncoder());
  Load(adapter, *destination, checkpoint, pool);
  Require(!adapter.Validate(*destination, checkpoint.positions),
          "changed image accepted");
  Require(adapter.Invalidate(*destination), "changed image reset failed");
  b.ConfigureVision(red, model->VisionEncoder());
  Load(adapter, *destination, checkpoint, pool);
  Require(adapter.Validate(*destination, checkpoint.positions),
          "matching image rejected");
  (void)a.ForwardPromptBatch(std::span(red->tokens).subspan(2), 2);
  (void)b.ForwardPromptBatch(std::span(red->tokens).subspan(2), 2);
  SameLogits(a, b);
  Require(a.VisionLayout().Position(4)[0] != 4,
          "image fixture did not change positions");
  Reject([&] { a.ConfigureVision(blue, model->VisionEncoder()); });
  a.Reset();
  a.ConfigureVision(nullptr, nullptr);
  const std::array<Token, 2> plain{42, 43};
  (void)a.ForwardPromptBatch(plain);
  const auto text = Capture(adapter, *source, pool);
  Load(adapter, *destination, text, pool);
  Require(adapter.Validate(*destination, text.positions),
          "text restore rejected");
  Require(b.VisionLayout().images.empty(),
          "text restore retained image layout");
  (void)a.ForwardToken(44, 2);
  (void)b.ForwardToken(44, 2);
  SameLogits(a, b);
  std::cout << "images,PASS\n";
}
void CheckImageSuffix(const std::shared_ptr<const hip::QwenGpuModel>& model) {
  auto prompt = std::make_shared<vision::Prompt>();
  prompt->tokens.push_back(42);
  for (bool blue : {false, true}) {
    const auto image = Image(blue);
    auto prepared = image->images.front();
    prepared.grid.offset = static_cast<std::uint32_t>(prompt->tokens.size());
    prompt->rope.images.push_back(prepared.grid);
    prompt->images.push_back(std::move(prepared));
    prompt->tokens.insert(prompt->tokens.end(), image->tokens.begin(),
                          image->tokens.end());
  }
  prompt->cache_identity.assign(32, 2);
  for (const std::size_t boundary : {1, 6}) {
    qwen::ContinuationAdapter adapter(model, 64,
                                      {'s', 'u', 'f', 'f', 'i', 'x'});
    hip::TransferPool pool(2);
    Guard ga, gb;
    auto source = adapter.CreateSlot(ga), destination = adapter.CreateSlot(gb);
    auto& a = adapter.GetExecutor(*source);
    auto& b = adapter.GetExecutor(*destination);
    a.ConfigureVision(prompt, model->VisionEncoder());
    b.ConfigureVision(prompt, model->VisionEncoder());
    (void)a.ForwardPromptBatch(std::span(prompt->tokens).first(boundary));
    const auto checkpoint = Capture(adapter, *source, pool);
    Load(adapter, *destination, checkpoint, pool);
    Require(adapter.Validate(*destination, checkpoint.positions),
            "image suffix checkpoint rejected");
    Require(b.VisionLayout() == prompt->rope,
            "restoration discarded future image positions");
    (void)a.ForwardPromptBatch(std::span(prompt->tokens).subspan(boundary),
                               static_cast<std::uint32_t>(boundary));
    (void)b.ForwardPromptBatch(std::span(prompt->tokens).subspan(boundary),
                               static_cast<std::uint32_t>(boundary));
    SameLogits(a, b);
    Require(Capture(adapter, *source, pool).bytes ==
                Capture(adapter, *destination, pool).bytes,
            "restoration changed image suffix component bytes");
  }
  std::cout << "image_suffix,PASS\n";
}
void CheckPrefillCancellation(
    const std::shared_ptr<const hip::QwenGpuModel>& model, bool fp16) {
  auto policy = hip::QwenExecutionPolicy::Production();
  policy.kv_cache_storage =
      fp16 ? hip::QwenKvCacheStorage::kFp16 : hip::QwenKvCacheStorage::kFp32;
  qwen::ContinuationAdapter adapter(model, 512, {'c', 'a', 'n', 'c', 'e', 'l'},
                                    policy);
  Guard ga, gb;
  hip::TransferPool pool(2);
  auto source = adapter.CreateSlot(ga), control = adapter.CreateSlot(gb);
  auto& a = adapter.GetExecutor(*source);
  auto& b = adapter.GetExecutor(*control);
  const auto fixture = model->GetTokenizer().Encode(
      "Prompt cancellation must preserve the completed recurrent state. ");
  Require(!fixture.empty(), "empty cancellation fixture");
  std::vector<Token> tokens(256);
  for (std::size_t i = 0; i < tokens.size(); ++i)
    tokens[i] = fixture[i % fixture.size()];
  for (const auto prefix : {0U, 37U}) {
    for (const auto stop : {4U, 12U}) {
      Require(adapter.Invalidate(*source) && adapter.Invalidate(*control),
              "cancellation fixture reset failed");
      if (prefix) {
        (void)a.ForwardPromptBatch(std::span(tokens).first(prefix));
        (void)b.ForwardPromptBatch(std::span(tokens).first(prefix));
      }
      const auto before = Capture(adapter, *source, pool);
      unsigned polls = 0;
      a.SetCancellationCheck([&] { return ++polls >= stop; });
      adapter.BeginPrefill(*source);
      bool interrupted = false;
      try {
        (void)a.ForwardPromptBatch(std::span(tokens).subspan(prefix, 129),
                                   prefix);
      } catch (const hip::QwenPrefillCancelled&) {
        adapter.RollbackPrefill(*source);
        interrupted = true;
      }
      a.SetCancellationCheck({});
      Require(interrupted && polls >= stop,
              "prefill ignored cancellation between layers");
      Require(Capture(adapter, *source, pool, std::byte{0x5a}).bytes ==
                  before.bytes,
              "cancelled prefill changed its completed frontier");
      adapter.BeginPrefill(*source);
      (void)a.ForwardPromptBatch(std::span(tokens).subspan(prefix, 129),
                                 prefix);
      adapter.FinishPrefill(*source);
      (void)b.ForwardPromptBatch(std::span(tokens).subspan(prefix, 129),
                                 prefix);
      SameLogits(a, b);
      Require(Capture(adapter, *source, pool).bytes ==
                  Capture(adapter, *control, pool).bytes,
              "prefill rollback changed resumed state");
    }
  }
  std::cout << (fp16 ? "FP16" : "FP32") << ",prefill_cancellation,PASS\n";
}
void CheckDraftComponents(std::shared_ptr<const hip::QwenGpuModel> model,
                          const char* draft_path) {
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(draft_path, &error);
  Require(bool(reader), error.c_str());
  auto draft_model = hip::QwenDFlashGpuModel::Create(
      std::shared_ptr<const gufo::core::GgufReader>(std::move(reader)), model,
      &error);
  Require(bool(draft_model), error.c_str());
  qwen::ContinuationAdapter adapter(
      model, 256, {'d', 'r', 'a', 'f', 't'},
      hip::QwenExecutionPolicy::Production(), draft_model,
      {.max_context = 256, .max_draft_tokens = 4});
  Guard ga, gb;
  hip::TransferPool pool(2);
  auto source = adapter.CreateSlot(ga), destination = adapter.CreateSlot(gb);
  auto& a = adapter.GetExecutor(*source);
  auto& b = adapter.GetExecutor(*destination);
  auto& da = *adapter.GetDraftBackend(*source);
  auto& db = *adapter.GetDraftBackend(*destination);
  a.SetPromptHiddenCapture(true, da.TargetHiddenLayerIds());
  b.SetPromptHiddenCapture(true, db.TargetHiddenLayerIds());
  const auto fixture = model->GetTokenizer().Encode(
      "The quick brown fox jumps over the lazy dog. ");
  Require(!fixture.empty(), "empty draft fixture");
  std::vector<Token> tokens(64);
  for (std::size_t i = 0; i < tokens.size(); ++i)
    tokens[i] = fixture[i % fixture.size()];
  auto next = a.ForwardPromptBatch(tokens);
  const auto hidden = a.GetPromptHiddenStates();
  Require(da.PrimeTargetContext({.prompt_tokens = tokens,
                                 .prompt_hidden_states = hidden,
                                 .hidden_size = hidden.size() / tokens.size(),
                                 .first_token = next}),
          "draft prime failed");
  for (std::uint32_t i = 0; i < 3; ++i) {
    tokens.push_back(next);
    next = a.ForwardToken(next, 64 + i);
    da.UpdateTargetHidden(a.CopyLastHidden());
  }
  auto& frontier = adapter.GetFrontier(*source);
  frontier.token = next;
  frontier.pending = {next, fixture.front()};
  frontier.published = true;
  const auto logits = a.CopyLastLogits();
  frontier.logits.assign(logits.begin(), logits.end());
  const auto checkpoint = Capture(adapter, *source, pool);
  Require(Capture(adapter, *source, pool, std::byte{0xa5}).bytes ==
              checkpoint.bytes,
          "draft checkpoint depends on recycled destination bytes");
  for (std::size_t i = 0; i < adapter.Components().size(); ++i) {
    const auto& descriptor = adapter.Components()[i];
    if (descriptor.kind != cache::ComponentKind::kPrivateState)
      continue;
    std::vector<std::byte> pieces(descriptor.state_bytes, std::byte{0x5a});
    const auto cut = std::min<std::size_t>(37, pieces.size());
    for (const auto [first, count] : {std::pair{cut, pieces.size() - cut},
                                      std::pair{std::size_t{0}, cut}}) {
      auto stream = pool.TryAcquire();
      Require(bool(stream), "private-piece stream unavailable");
      Wait(adapter.CapturePrivatePiece(*source, descriptor.id, first,
                                       std::span(pieces).subspan(first, count),
                                       *stream));
    }
    Require(pieces == checkpoint.bytes[i],
            "partial draft capture differs from the canonical checkpoint");
  }
  Require(da.GetInjectedContextLength() == 64 &&
              checkpoint.positions.front().valid_rows == 67,
          "draft frontier was replaced by target frontier");
  Load(adapter, *destination, checkpoint, pool, false, source.get());
  Require(adapter.Validate(*destination, checkpoint.positions),
          "draft restore failed");
  Require(db.GetInjectedContextLength() == 64,
          "independent draft frontier was not restored");
  Require(Capture(adapter, *destination, pool).bytes == checkpoint.bytes,
          "restored draft components differ");
  SameLogits(a, b);
  std::uint64_t rng_a = 12345, rng_b = rng_a;
  const auto proposal_a = da.ProposeSampled(tokens, 67, 4, 0.8F, &rng_a);
  const auto proposal_b = db.ProposeSampled(tokens, 67, 4, 0.8F, &rng_b);
  Require(!proposal_a.tokens.empty(), "draft fixture produced no proposal");
  Require(proposal_a.tokens == proposal_b.tokens &&
              proposal_a.candidate_ids == proposal_b.candidate_ids &&
              proposal_a.candidate_probabilities ==
                  proposal_b.candidate_probabilities &&
              rng_a == rng_b,
          "draft proposals changed after component restore");
  Reject([&] { (void)Capture(adapter, *source, pool); });
  da.AcceptFeedback(proposal_a.tokens, next);
  db.AcceptFeedback(proposal_b.tokens, next);
  Require(Capture(adapter, *source, pool).bytes ==
              Capture(adapter, *destination, pool).bytes,
          "draft controller or pending features diverged after feedback");
  // Establish an actual sampled copy before checkpointing its private resume
  // state. Teacher-forcing a repeated suffix isolates cache behavior from the
  // model's preference for any particular continuation.
  Load(adapter, *source, checkpoint, pool);
  Require(adapter.Validate(*source, checkpoint.positions),
          "reload copy source");
  adapter.GetFrontier(*source) = {};
  bool copied = false;
  const auto advance = [&](Token token) {
    (void)a.ForwardToken(token, tokens.size());
    tokens.push_back(token);
    da.UpdateTargetHidden(a.CopyLastHidden());
  };
  for (std::size_t round = 0; round < 32 && !copied; ++round) {
    const auto anchor = fixture[tokens.size() % fixture.size()];
    auto sequence = tokens;
    sequence.push_back(anchor);
    const auto proposal =
        da.ProposeSampled(sequence, tokens.size(), 4, 0.8F, &rng_a);
    copied = proposal.from_lookup;
    if (copied)
      for (std::size_t i = 0; i < proposal.tokens.size(); ++i)
        Require(proposal.tokens[i] ==
                    fixture[(tokens.size() + 1 + i) % fixture.size()],
                "copy differs from the independent repeating fixture");
    da.AcceptFeedback(
        copied ? std::span(proposal.tokens) : std::span<const Token>{},
        fixture.front());
    advance(anchor);
    if (copied)
      for (auto token : proposal.tokens)
        advance(token);
  }
  Require(copied, "component fixture never exercised lookup");
  const auto copy_checkpoint = Capture(adapter, *source, pool);
  Load(adapter, *destination, copy_checkpoint, pool);
  Require(adapter.Validate(*destination, copy_checkpoint.positions),
          "lookup component restore failed");
  Require(Capture(adapter, *destination, pool).bytes == copy_checkpoint.bytes,
          "lookup private state changed after restore");
  auto sequence = tokens;
  sequence.push_back(fixture[tokens.size() % fixture.size()]);
  rng_b = rng_a;
  const auto initial_rng = rng_a;
  const auto original =
      da.ProposeSampled(sequence, tokens.size(), 4, 0.8F, &rng_a);
  const auto restored =
      db.ProposeSampled(sequence, tokens.size(), 4, 0.8F, &rng_b);
  Require(original.from_lookup && restored.from_lookup &&
              original.tokens == restored.tokens &&
              original.candidate_probabilities ==
                  restored.candidate_probabilities &&
              rng_a == initial_rng && rng_b == initial_rng,
          "component restore lost a sampled copy or consumed RNG");
  // Draft metadata is the first of its metadata, layer K/V and feature parts.
  const auto metadata_index = adapter.Components().size() -
                              2 * draft_model->GetDFlashConfig().num_layers - 2;
  auto legacy = copy_checkpoint;
  Require(legacy.bytes[metadata_index].size() == 64,
          "draft metadata fixture layout changed");
  legacy.bytes[metadata_index][0] = std::byte{1};
  std::fill(legacy.bytes[metadata_index].begin() + 24,
            legacy.bytes[metadata_index].begin() + 32, std::byte{0});
  Load(adapter, *destination, legacy, pool);
  Require(adapter.Validate(*destination, legacy.positions),
          "old draft component metadata must remain readable");
  const auto old = db.ProposeSampled(sequence, tokens.size(), 4, 0.8F, &rng_b);
  Require(!old.from_lookup && rng_b != initial_rng,
          "old metadata must start without a previous request's copy state");
  auto invalid = copy_checkpoint;
  std::fill(invalid.bytes[metadata_index].begin() + 28,
            invalid.bytes[metadata_index].begin() + 32, std::byte{0xff});
  invalid.bytes[metadata_index][28] = std::byte{0xfe};
  Load(adapter, *destination, invalid, pool);
  Require(!adapter.Validate(*destination, invalid.positions),
          "component restore accepted a lookup boundary beyond the history");
  std::cout << "draft_components_lookup,PASS\n";
  std::cout << "draft_components,PASS\n";
}
}  // namespace
int main(int argc, const char* const* argv) {
  try {
    const bool guard_only = argc == 3 && std::string(argv[2]) == "--guard-only";
    const bool image_only = argc == 3 && std::string(argv[2]) == "--image-only";
    const bool draft_only = argc == 4 && std::string(argv[2]) == "--draft";
    const bool cancel_only =
        argc == 3 && std::string(argv[2]) == "--prefill-cancel";
    if (argc != 2 && !guard_only && !image_only && !draft_only && !cancel_only)
      throw std::invalid_argument(
          "Usage: continuation_adapter_test MODEL.gguf "
          "[--guard-only|--image-only|--prefill-cancel|--draft DRAFT.gguf]");
    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
    Require(bool(reader), error.c_str());
    auto encoder = vision::Encoder::Open(argv[1], {}, 5120);
    auto model = hip::QwenGpuModel::CreateFromGguf(
        std::shared_ptr<const gufo::core::GgufReader>(std::move(reader)),
        &error, encoder);
    Require(bool(model), error.c_str());
    if (cancel_only) {
      CheckPrefillCancellation(model, true);
      CheckPrefillCancellation(model, false);
      return 0;
    }
    if (draft_only) {
      CheckDraftComponents(model, argv[3]);
      std::cout << "Qwen continuation draft adapter: PASS\n";
      return 0;
    }
    if (!guard_only && !image_only) {
      Check(model, true);
      Check(model, false);
      Check(model, true, true);
      CheckSharedPassScratch(model);
      CheckSharedKvPool(model);
      CheckRetainedKvBlocks(model);
      CheckPrefillCancellation(model, true);
      CheckPrefillCancellation(model, false);
    }
    if (!image_only) {
      CheckLeasedGuard(model);
      CheckBatchAdmission(model);
      CheckMutationPaths(model);
    }
    if (!guard_only) {
      CheckImages(model);
      CheckImageSuffix(model);
    }
    std::cout << "Qwen continuation adapter: PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
