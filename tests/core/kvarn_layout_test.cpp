// Shared KVarN record layout and planners (src/core/hip/kvarn_layout.hpp,
// kvarn_checkpoint.hpp), host-only. The Qwen3.8-27B policies built on them
// are covered by tests/models/qwen/hip/kvarn_layout_test.cpp.
#include "src/core/hip/kvarn_layout.hpp"

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
             restored->written_end == 300,
         "restored tail");
}

}  // namespace

int main() {
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
