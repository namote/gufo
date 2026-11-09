// Shared KVarN record layout and planners (src/core/hip/kvarn_layout.hpp,
// kvarn_checkpoint.hpp), host-only. The Qwen3.8-27B policies built on them
// are covered by tests/models/qwen/hip/kvarn_layout_test.cpp.
#include "src/core/hip/kvarn_layout.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string_view>

#include "src/core/hip/kvarn_checkpoint.hpp"

namespace {

using namespace gufo::hip;

int failures = 0;
void Expect(bool ok, std::string_view what) {
  if (!ok) {
    std::cerr << "FAIL " << what << '\n';
    ++failures;
  }
}

void TestRecordOrders() {
  for (const int bits : {2, 4}) {
    const auto channel = KvarnRecord(bits, KvarnValueOrder::kChannelMajor);
    const auto token = KvarnRecord(bits, KvarnValueOrder::kTokenMajor);
    // Both orders keep every part at the same offset; only the meaning of a
    // value-code row changes.
    Expect(
        channel.v_codes == token.v_codes && channel.k_scale == token.k_scale &&
            channel.k_zero == token.k_zero &&
            channel.k_column == token.k_column &&
            channel.v_scale == token.v_scale &&
            channel.v_zero == token.v_zero &&
            channel.v_column == token.v_column && channel.bytes == token.bytes,
        "value orders share offsets");
    Expect(channel.v_codes == kKvarnGroupTokens * kKvarnHeadDim / 2,
           "value codes follow the key codes");
    Expect(channel.v_row_bytes ==
               static_cast<std::size_t>(kKvarnGroupTokens) * bits / 8,
           "channel-major row is one channel's group of tokens");
    Expect(
        token.v_row_bytes == static_cast<std::size_t>(kKvarnHeadDim) * bits / 8,
        "token-major row is one token's channels");
    Expect(channel.k_scale - channel.v_codes ==
               std::size_t{kKvarnGroupTokens} * kKvarnHeadDim * bits / 8,
           "value code bytes");
    // The default stays the Qwen3.8-27B channel-major order.
    Expect(KvarnRecord(bits).v_row_bytes == channel.v_row_bytes,
           "default order");
  }
  Expect(KvarnRecord(4, KvarnValueOrder::kTokenMajor).bytes == 35072,
         "k4v4 record bytes");
}

void TestPlanes() {
  // Qwen3.8 Flash-Next k4v4: 12 attention layers of 2 KV heads, token-major.
  const auto plane = KvarnRecordPlane(12, 2, 4, KvarnValueOrder::kTokenMajor);
  Expect(plane.layers == 12 && plane.rows_per_block == kKvarnGroupsPerBlock &&
             plane.row_bytes == 2 * 35072,
         "Flash-Next record plane");
  // Per token and layer: two groups per 256-token block.
  const double per_token = static_cast<double>(plane.row_bytes) *
                           plane.rows_per_block / kKvBlockTokens;
  Expect(per_token == 548.0, "Flash-Next k4v4 bytes per token and layer");
  Expect(KvarnRecordPlane(16, 4, 2).row_bytes == 4 * 26880,
         "27B k4v2 record plane unchanged");
}

void TestSharedPlanners() {
  // The FP16 rows: the sink, then a ring.
  Expect(KvarnFp16Row(5) == 5 &&
             KvarnFp16Row(kKvarnSinkTokens) == kKvarnSinkTokens &&
             KvarnFp16Row(kKvarnSinkTokens + kKvarnTailRingTokens) ==
                 kKvarnSinkTokens,
         "FP16 sink and ring rows");
  Expect(KvarnReadGroups(127) == 0 && KvarnReadGroups(256) == 1 &&
             KvarnReadGroups(384) == 2,
         "read groups");
  const auto pass = PlanKvarnPass(KvarnTail{}, 250, 8);
  Expect(pass.error.empty() && pass.boundary == 256 && pass.read_groups == 0,
         "a pass ending a group");
  const auto rows = KvarnRowsAt(300, 256);
  Expect(
      rows.records == 256 && rows.sink == kKvarnSinkTokens && rows.tail == 44,
      "checkpoint rows");
  const auto restored = KvarnRestoredTail(300, rows);
  Expect(restored.has_value() && restored->flushed_end == 256 &&
             restored->written_end ==
                 256 + kKvarnTailRingTokens - kKvarnGroupTokens + 1,
         "restored tail");
}

// C6-AUDIT F1: the ring holds FP16 rows only from where a 27B prompt chunk
// stopped quantizing from staging, or from a restored base. A rewind into a
// group before that is refused instead of requantizing rows never written.
void TestRingReach() {
  // A prompt chunk over [0, 1000) quantizes groups 1..6 from staging; the
  // ring holds rows 896..999 (QwenGpuArena::PrepareKvarnPrefill).
  KvarnTail tail{.flushed_end = kKvarnSinkTokens, .written_end = 0};
  const auto chunk = PlanKvarnPrefill(tail, 0, 1000);
  Expect(chunk.error.empty() && chunk.flushed_end == 896, "prompt chunk");
  tail.flushed_end = chunk.flushed_end;
  tail.written_end = KvarnRingReach(chunk.flushed_end, 1000);
  Expect(!PlanKvarnPass(tail, 800, 8).error.empty(),
         "a rewind before the chunk's ring rows is refused");
  Expect(!PlanKvarnPrefill(tail, 800, 8).error.empty(),
         "a prompt rewind before the chunk's ring rows is refused");
  const auto apply = [](KvarnTail& t, std::uint32_t start, std::uint32_t n) {
    const auto plan = PlanKvarnPass(t, start, n);
    Expect(plan.error.empty(), "pass");
    t.flushed_end = plan.flushed_end;
    t.written_end = std::max(t.written_end, start + n);
  };
  // Verification crosses 1024, then drafts are rejected: the rewinds into
  // the ring's group 896 still reopen it; one into group 768 does not.
  apply(tail, 1020, 8);
  Expect(tail.flushed_end == 1024, "verification crosses a group end");
  Expect(PlanKvarnPass(tail, 1000, 8).error.empty() &&
             PlanKvarnPass(tail, 896, 8).error.empty(),
         "the ring's group still reopens");
  Expect(!PlanKvarnPass(tail, 895, 1).error.empty(),
         "the group before it does not");
  // A restore at 1024 holds no ring rows before its base.
  const auto rows = KvarnRowsAt(1024, 1024);
  const auto restored = KvarnRestoredTail(1024, rows);
  Expect(
      restored.has_value() && !PlanKvarnPass(*restored, 1023, 1).error.empty(),
      "a rewind before a restored base is refused");
  Expect(PlanKvarnPass(*restored, 1024, 8).error.empty(),
         "a restored slot continues");
  auto mid = KvarnRestoredTail(1100, KvarnRowsAt(1100, 1024));
  Expect(mid.has_value() && mid->written_end == 1024 + 129,
         "a restored tail reaches its base");
  // Verification from the restored position crosses 1152 and is rejected:
  // the rewind into the restored group reopens it, one before it does not.
  apply(*mid, 1100, 60);
  Expect(mid->flushed_end == 1152, "verification crosses a group end");
  Expect(PlanKvarnPass(*mid, 1101, 8).error.empty(),
         "a rewind into the restored group");
  Expect(!PlanKvarnPass(*mid, 1023, 1).error.empty(),
         "a rewind before the restored base is refused");
  // Before the first group the sink is the whole ring.
  Expect(KvarnRingReach(kKvarnSinkTokens, 70) == 70, "sink only");
}

// C6-AUDIT F2: a write first quantizes the group a pass ending at a group end
// left in the ring, so it changes records from that group on.
void TestWriteFirst() {
  KvarnTail tail;
  const auto pass =
      PlanKvarnPass(KvarnTail{.flushed_end = 384, .written_end = 500}, 500, 12);
  Expect(pass.error.empty() && pass.flushed_end == 384 && pass.boundary == 0,
         "a pass ending at a group end leaves its group in the ring");
  tail = {.flushed_end = pass.flushed_end, .written_end = 512};
  const auto next = PlanKvarnPass(tail, 512, 1);
  Expect(next.first_group == 3 && next.groups == 1, "the next pass flushes it");
  Expect(KvarnWriteFirst(512, tail.flushed_end) == 384,
         "the write guard starts at the flushed group");
  Expect(KvarnWriteFirst(520, 512) == 512 && KvarnWriteFirst(500, 512) == 384,
         "otherwise at the group holding the write");
  Expect(KvarnWriteFirst(70, kKvarnSinkTokens) == 0, "sink rows");
}

}  // namespace

int main() {
  TestRingReach();
  TestWriteFirst();
  static_assert(
      gufo::hip::KvarnRecord(6, gufo::hip::KvarnValueOrder::kTokenMajor, 6)
          .bytes == 51456);

  TestRecordOrders();
  TestPlanes();
  TestSharedPlanners();
  if (failures == 0)
    std::cout << "kvarn layout: PASS\n";
  return failures == 0 ? 0 : 1;
}
