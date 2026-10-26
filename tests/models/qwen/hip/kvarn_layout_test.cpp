// KVarN storage sizes, split policy and tail bookkeeping
// (src/models/qwen/hip/kvarn_layout.hpp), host-only.
#include "src/models/qwen/hip/kvarn_layout.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

#include "src/core/hip/kv_block_layout.hpp"

namespace {

using namespace gufo::hip;

int failures = 0;
void Expect(bool ok, std::string_view what) {
  if (!ok) {
    std::cerr << "FAIL " << what << '\n';
    ++failures;
  }
}

void TestSizes() {
  // Qwen3.8-27B: 16 attention layers, 4 KV heads of 256.
  constexpr std::uint32_t kLayers = 16, kKv = 4, kDim = 256;
  Expect(KvarnValueBits(QwenKvCacheStorage::kKvarnK4V2) == 2 &&
             KvarnValueBits(QwenKvCacheStorage::kKvarnK4V4) == 4 &&
             KvarnValueBits(QwenKvCacheStorage::kFp16) == 0 &&
             KvarnValueBits(QwenKvCacheStorage::kFp32) == 0,
         "value bits per storage");
  const auto fp16 =
      QwenKvBlockPlanes(kLayers, kKv, kDim, QwenKvCacheStorage::kFp16);
  const KvBlockPlane rows{.layers = kLayers, .row_bytes = kKv * kDim * 2};
  Expect(fp16.size() == 2 && fp16[0] == rows && fp16[1] == rows,
         "FP16 pool keeps its K and V row planes");
  Expect(
      QwenKvBlockPlanes(kLayers, kKv, kDim, QwenKvCacheStorage::kFp32).empty(),
      "FP32 reference has no pool");
  // FP16 block: 256 tokens x 64 KiB.
  Expect(KvBlockGeometry(fp16, 1).bytes_per_block() == 256ULL * 65536,
         "FP16 block bytes");
  for (const auto [storage, record] :
       {std::pair{QwenKvCacheStorage::kKvarnK4V2, 26880ULL},
        std::pair{QwenKvCacheStorage::kKvarnK4V4, 35072ULL}}) {
    const auto planes = QwenKvBlockPlanes(kLayers, kKv, kDim, storage);
    Expect(planes.size() == 1 && planes[0].rows_per_block == 2 &&
               planes[0].row_bytes == kKv * record &&
               planes[0].layers == kLayers,
           "KVarN record plane");
    // Two groups of four records per layer.
    Expect(KvBlockGeometry(planes, 1).bytes_per_block() ==
               kLayers * 2ULL * kKv * record,
           "KVarN block bytes");
    // 256K context: 1024 blocks, 3.3 GiB for K4V2 (16 GiB FP16).
    const KvBlockGeometry pool(planes, KvBlockGeometry::BlocksFor(262144));
    Expect(pool.bytes() == 1024ULL * kLayers * 2 * kKv * record,
           "KVarN 256K pool bytes");
  }
  Expect(KvarnFp16Bytes(kLayers, kKv, kDim) ==
             2ULL * kLayers * (128 + 256) * kKv * kDim * 2,
         "FP16 sink and tail bytes");
  Expect(KvarnFp16Row(0) == 0 && KvarnFp16Row(127) == 127 &&
             KvarnFp16Row(128) == 128 && KvarnFp16Row(383) == 383 &&
             KvarnFp16Row(384) == 128 && KvarnFp16Row(640) == 128 &&
             KvarnFp16Row(641) == 129,
         "sink then ring rows");
  Expect(KvarnScratchElements(8, 38) == 24ULL * 8 * 40 * 258,
         "scratch elements");
}

void TestSplits() {
  Expect(SelectKvarnDecodeSplits(0, 2, 1) == 0, "no groups, no splits");
  Expect(SelectKvarnDecodeSplits(5, 2, 1) == 5, "splits capped by groups");
  // 4K, 32K, 64K, 128K, 256K depths (groups after the sink).
  for (const int bits : {2, 4})
    for (const std::uint32_t rows : {1U, 8U}) {
      const bool v4_decode = bits == 4 && rows == 1;
      Expect(SelectKvarnDecodeSplits(31, bits, rows) == 18, "4K splits");
      Expect(SelectKvarnDecodeSplits(255, bits, rows) == (v4_decode ? 38 : 18),
             "32K splits");
      Expect(SelectKvarnDecodeSplits(511, bits, rows) == (v4_decode ? 38 : 18),
             "64K splits");
      Expect(SelectKvarnDecodeSplits(1023, bits, rows) == 38, "128K splits");
      Expect(SelectKvarnDecodeSplits(2047, bits, rows) == kKvarnMaxSplits,
             "256K splits");
    }
}

void TestTail() {
  KvarnTail tail;
  const auto apply = [&](std::uint32_t start, std::uint32_t rows,
                         std::uint32_t confirmed) {
    const auto plan = PlanKvarnPass(tail, start, rows, confirmed);
    if (plan.error.empty()) {
      tail.flushed_end = plan.flushed_end;
      tail.written_end = std::max(tail.written_end, start + rows);
    }
    return plan;
  };
  // Decode through the sink: nothing to quantize.
  for (std::uint32_t pos = 0; pos < 256; ++pos) {
    const auto plan = apply(pos, 1, pos);
    Expect(plan.error.empty() && plan.groups == 0 && plan.flushed_end == 128,
           "no flush before the first group completes");
  }
  // Position 256 sees group 1 complete and confirmed.
  auto plan = apply(256, 1, 256);
  Expect(plan.error.empty() && plan.first_group == 1 && plan.groups == 1 &&
             plan.flushed_end == 256,
         "group 1 flushes at 256");
  // A pinned rollback point at 300 keeps rows past it FP16 ...
  for (std::uint32_t pos = 257; pos < 400; ++pos)
    Expect(apply(pos, 1, 300).groups == 0, "speculative rows stay FP16");
  // Confirming 400 quantizes group 2.
  plan = apply(400, 8, 400);
  Expect(plan.error.empty() && plan.first_group == 2 && plan.groups == 1 &&
             plan.flushed_end == 384,
         "group 2 flushes once confirmed");
  // Rewinding inside the tail keeps the groups; into group 2 reopens it
  // from the ring.
  plan = apply(390, 8, 390);
  Expect(plan.error.empty() && plan.groups == 0 && plan.flushed_end == 384,
         "a rewind inside the tail keeps the groups");
  plan = apply(300, 8, 300);
  Expect(plan.error.empty() && plan.groups == 0 && plan.flushed_end == 256,
         "a rewind into a quantized group reopens it");
  // A pass that would overrun the ring quantizes an unconfirmed group.
  plan = PlanKvarnPass({.flushed_end = 256, .written_end = 400}, 400, 120, 300);
  Expect(plan.error.empty() && plan.first_group == 2 && plan.groups == 1 &&
             plan.flushed_end == 384,
         "a full ring forces the flush");
  // Its FP16 rows are overwritten once the ring wraps past it.
  tail.written_end = 600;
  plan = PlanKvarnPass(tail, 200, 1, 200);
  Expect(!plan.error.empty(), "a rewind past the ring is refused");
  // Verification widths up to the ring.
  KvarnTail fresh{.flushed_end = 1024, .written_end = 1100};
  plan = PlanKvarnPass(fresh, 1100, 8, 1100);
  Expect(plan.error.empty() && plan.first_group == 8 && plan.groups == 0 &&
             plan.flushed_end == 1024,
         "verification after the last group");
  plan = PlanKvarnPass(fresh, 1151, 129, 1151);
  Expect(plan.error.empty() && plan.groups == 0, "the widest pass fits");
  plan = PlanKvarnPass(fresh, 1151, 130, 1151);
  Expect(!plan.error.empty(), "a pass wider than the ring is refused");
  // Long prompts flush every complete group before the pass.
  KvarnTail prompt;
  plan = PlanKvarnPass(prompt, 4096, 100, 4096);
  Expect(plan.error.empty() && plan.first_group == 1 && plan.groups == 31 &&
             plan.flushed_end == 4096,
         "a pass flushes all complete confirmed groups");
}

}  // namespace

int main() {
  TestSizes();
  TestSplits();
  TestTail();
  if (failures != 0) {
    std::cerr << failures << " KVarN layout checks failed\n";
    return 1;
  }
  std::cout << "KVarN layout tests passed.\n";
  return 0;
}
