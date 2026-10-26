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
  Expect(SelectKvarnDecodeSplits(0, 2) == 0, "no groups, no splits");
  Expect(SelectKvarnDecodeSplits(5, 2) == 5, "splits capped by groups");
  // 4K, 32K, 64K, 128K, 256K depths (groups after the sink).
  for (const int bits : {2, 4}) {
    Expect(SelectKvarnDecodeSplits(31, bits) == 18, "4K splits");
    Expect(SelectKvarnDecodeSplits(255, bits) == (bits == 4 ? 38 : 18),
           "32K splits");
    Expect(SelectKvarnDecodeSplits(511, bits) == (bits == 4 ? 38 : 18),
           "64K splits");
    Expect(SelectKvarnDecodeSplits(1023, bits) == 38, "128K splits");
    Expect(SelectKvarnDecodeSplits(2047, bits) == kKvarnMaxSplits,
           "256K splits");
  }
  Expect(KvarnReadGroups(0) == 0 && KvarnReadGroups(255) == 0 &&
             KvarnReadGroups(256) == 1 && KvarnReadGroups(383) == 1 &&
             KvarnReadGroups(384) == 2,
         "a query reads every group that ended at or before it");
}

void TestTail() {
  KvarnTail tail;
  const auto apply = [&](std::uint32_t start, std::uint32_t rows) {
    const auto plan = PlanKvarnPass(tail, start, rows);
    if (plan.error.empty()) {
      tail.flushed_end = plan.flushed_end;
      tail.written_end = std::max(tail.written_end, start + rows);
    }
    return plan;
  };
  // Decode through the sink: nothing to quantize.
  for (std::uint32_t pos = 0; pos < 256; ++pos) {
    const auto plan = apply(pos, 1);
    Expect(plan.error.empty() && plan.groups == 0 && plan.boundary == 0 &&
               plan.read_groups == 0 && plan.flushed_end == 128,
           "no flush before the first group completes");
  }
  // Position 256 reads group 1, quantized first.
  auto plan = apply(256, 1);
  Expect(plan.error.empty() && plan.first_group == 1 && plan.groups == 1 &&
             plan.read_groups == 1 && plan.flushed_end == 256,
         "group 1 flushes at 256");
  for (std::uint32_t pos = 257; pos < 380; ++pos)
    Expect(apply(pos, 1).groups == 0, "decode inside group 2");
  // Verification rows 380-387 cross the end of group 2 at 384: rows from
  // 384 read it quantized, as decode steps there do.
  plan = apply(380, 8);
  Expect(plan.error.empty() && plan.groups == 0 && plan.read_groups == 1 &&
             plan.boundary == 384 && plan.flushed_end == 384,
         "a group ending inside a pass is quantized in the pass");
  // Rejected drafts: the next pass starts at 382, before the boundary, and
  // reopens group 2 from the ring.
  plan = apply(382, 8);
  Expect(plan.error.empty() && plan.groups == 0 && plan.read_groups == 1 &&
             plan.boundary == 384 && plan.flushed_end == 384,
         "a rewind before a group end reopens and requantizes it");
  // A rewind inside the tail keeps the groups.
  plan = apply(386, 1);
  Expect(plan.error.empty() && plan.groups == 0 && plan.read_groups == 2 &&
             plan.boundary == 0 && plan.flushed_end == 384,
         "a rewind inside the tail keeps the groups");
  plan = apply(300, 8);
  Expect(plan.error.empty() && plan.groups == 0 && plan.read_groups == 1 &&
             plan.flushed_end == 256,
         "a rewind into a quantized group reopens it");
  // Its FP16 rows are overwritten once the ring wraps past it.
  tail.written_end = 600;
  plan = PlanKvarnPass(tail, 200, 1);
  Expect(!plan.error.empty(), "a rewind past the ring is refused");
  // Pass widths up to one group.
  KvarnTail fresh{.flushed_end = 1024, .written_end = 1100};
  plan = PlanKvarnPass(fresh, 1100, 8);
  Expect(plan.error.empty() && plan.first_group == 8 && plan.groups == 0 &&
             plan.read_groups == 7 && plan.flushed_end == 1024,
         "verification after the last group");
  plan = PlanKvarnPass(fresh, 1151, 128);
  Expect(plan.error.empty() && plan.groups == 0 && plan.boundary == 1152 &&
             plan.flushed_end == 1152,
         "the widest pass fits");
  plan = PlanKvarnPass(fresh, 1151, 129);
  Expect(!plan.error.empty(), "a pass wider than a group is refused");
  // Every group ending before a pass is quantized first.
  KvarnTail prompt{.flushed_end = 128, .written_end = 4096};
  plan = PlanKvarnPass(prompt, 4096, 100);
  Expect(plan.error.empty() && plan.first_group == 1 && plan.groups == 31 &&
             plan.flushed_end == 4096,
         "a pass flushes all complete groups before it");
  // The sink end is not a group end.
  plan = PlanKvarnPass(KvarnTail{}, 120, 8);
  Expect(plan.error.empty() && plan.boundary == 0, "the sink end");
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
