#include "src/cli/serve/request_memory_budget.hpp"

#include <cassert>
#include <future>
#include <latch>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "src/models/qwen/vision/memory.hpp"

int main() {
  using gufo::server::MemoryFit;
  using gufo::server::RequestMemoryBudget;
  using gufo::server::RequestMemoryRefused;
  RequestMemoryBudget budget;
  auto available = [] { return std::size_t{1000}; };
  auto first = budget.Acquire({100, 300}, available);
  assert(budget.ReservedBytes() == 400);
  auto second = budget.Acquire({200, 200}, available);
  assert(budget.ReservedBytes() == 600);  // shared scratch max, state sum
  assert(budget.Check({500, 400}, available).fit == MemoryFit::kWait);
  try {
    (void)budget.Acquire({500, 400}, available);
    assert(false);
  } catch (const std::length_error&) {
  }
  assert(budget.ReservedBytes() == 600);
  first.reset();
  assert(budget.ReservedBytes() == 400);
  second.reset();
  assert(budget.ReservedBytes() == 0);
  // A claim that cannot fit waits only while releasable memory is held
  // elsewhere (pending writes); with nothing to release it is refused, never
  // left waiting for no owner.
  assert(budget.Check({1001, 0}, available, true).fit == MemoryFit::kWait);
  assert(budget.Check({1001, 0}, available).fit == MemoryFit::kRefused);
  assert(budget.Check({}, [] { return std::size_t{0}; }).fit ==
         MemoryFit::kFits);
  try {
    (void)budget.Acquire({std::numeric_limits<std::size_t>::max(), 1},
                         available);
    assert(false);
  } catch (const std::length_error&) {
  }
  assert(budget.ReservedBytes() == 0);
  // An in-flight cache commit promises the SAME memory as image admission.
  {
    auto commit = budget.Acquire({700, 0}, available);
    assert(budget.Check({200, 200}, available).fit == MemoryFit::kWait);
    commit.reset();  // physical commit failure/cancellation releases promise
    assert(budget.Check({200, 200}, available).fit == MemoryFit::kFits);
  }
  std::latch start(1);
  const auto concurrent = [&] {
    start.wait();
    try {
      return budget.Acquire({600, 0}, available);
    } catch (const std::length_error&) {
      return std::shared_ptr<void>{};
    }
  };
  auto one = std::async(std::launch::async, concurrent);
  auto two = std::async(std::launch::async, concurrent);
  start.count_down();
  auto owner_one = one.get();
  auto owner_two = two.get();
  assert(bool(owner_one) != bool(owner_two));
  assert(budget.ReservedBytes() == 600);
  owner_one.reset();
  owner_two.reset();
  assert(budget.ReservedBytes() == 0);
  {
    // Production refusal, 4 sessions (runtime-20261117a): host 8454 MiB
    // available, device heap 6311145472 B free, 354443264 B deferred.
    using gufo::server::ComputeMemoryHeadroom;
    constexpr std::uint64_t kMiB = 1 << 20, kGiB = kMiB << 10;
    // Host and device each keep 4 GiB (kHostSnapshotHeadroomBytes,
    // hip::kDispatchReserve); the device term binds.
    const auto headroom = ComputeMemoryHeadroom(
        8454 * kMiB, 4 * kGiB, 6311145472, 4 * kGiB, 354443264);
    assert(headroom.bytes == 6311145472 - 4 * kGiB - 354443264);
    assert(ComputeMemoryHeadroom(3 * kGiB, 4 * kGiB, 9 * kGiB, kGiB, 0).bytes ==
           0);
    assert(ComputeMemoryHeadroom(9 * kGiB, 4 * kGiB, kGiB + 5, kGiB, 6).bytes ==
           0);
    assert(ComputeMemoryHeadroom(9 * kGiB, 4 * kGiB, kGiB + 7, kGiB, 6).bytes ==
           1);
    // A check carries the claim, other holders, the headroom and its inputs
    // for the server log.
    RequestMemoryBudget images;
    const auto check = images.Check({kGiB, kGiB}, [&] { return headroom; });
    assert(check.fit == MemoryFit::kRefused);
    const auto detail = gufo::server::DescribeMemoryCheck(check);
    for (const auto* key :
         {"claim_persistent_bytes=1073741824", "claim_scratch_bytes=1073741824",
          "reserved_by_others_bytes=0", "headroom_bytes=1661734912",
          "host_available_bytes=8864661504", "host_reserve_bytes=4294967296",
          "device_free_bytes=6311145472", "device_reserve_bytes=4294967296",
          "deferred_scratch_bytes=354443264"})
      assert(detail.find(key) != std::string::npos);
    auto held = images.Acquire({kGiB, 0}, [&] { return headroom; });
    // Another image claim will be released: wait, not refuse.
    assert(images.Check({kGiB, kGiB}, [&] { return headroom; }).fit ==
           MemoryFit::kWait);
    try {
      (void)images.Acquire({kGiB, kGiB}, [&] { return headroom; });
      assert(false);
    } catch (const RequestMemoryRefused& error) {
      // The client sees only a short message; numbers stay in detail().
      assert(std::string(error.what()) == gufo::server::kRequestMemoryRefused);
      assert(!error.logged());
      assert(error.detail().find("stage=acquire") != std::string::npos);
      assert(error.detail().find("reserved_by_others_bytes=1073741824") !=
             std::string::npos);
    }
    held.reset();
    // 4K native EXL3 vision now fits this headroom (prefix state aside);
    // before chunking it needed 3,537,174,528 B.
    const auto image4k =
        gufo::models::qwen::vision::EstimateImageMemory(3840, 2176, 2560, true);
    assert(images
               .Check({image4k.embedding_bytes, image4k.scratch_bytes},
                      [&] { return headroom; })
               .fit == MemoryFit::kFits);
  }
  using gufo::models::qwen::vision::EstimateImageMemory;
  using gufo::models::qwen::vision::PlanImageEncoding;
  {
    // Up to 4096 patches the graph is unchanged: one chunk, existing tiles.
    const auto small = PlanImageEncoding(320, 224);
    assert(small.count == 280 && small.chunk_rows == 280 && small.tile == 128);
    const auto square = PlanImageEncoding(1024, 1024);
    assert(square.count == 4096 && square.chunk_rows == 4096 &&
           square.tile == 64);
    const auto wide = PlanImageEncoding(1600, 896);
    assert(wide.count == 5600 && wide.chunk_rows == 4096 && wide.tile == 128);
    // Tiles stay 128 rows while 16 x 128 x count x 6 B fits 128 MiB.
    assert(PlanImageEncoding(2560, 1088).tile == 128);  // 10,880 patches
    assert(PlanImageEncoding(2560, 1440).tile == 96);
    assert(PlanImageEncoding(3840, 2176).tile == 32);
    assert(PlanImageEncoding(4096, 4096).tile == 16);
    for (const auto& [width, height] :
         {std::pair{1600U, 896U}, std::pair{3840U, 2176U},
          std::pair{4096U, 4096U}, std::pair{4096U, 32U},
          std::pair{32U, 32U}}) {
      const auto plan = PlanImageEncoding(width, height);
      // Chunks cover every patch once and hold whole 2x2 merge groups, so
      // merged rows begin at begin / 4.
      assert(plan.chunk_rows % 4 == 0 || plan.chunk_rows == plan.count);
      std::size_t covered = 0, merged = 0;
      for (std::uint32_t begin = 0; begin < plan.count;
           begin += plan.chunk_rows) {
        const auto rows = std::min(plan.chunk_rows, plan.count - begin);
        assert(begin % 4 == 0 && rows % 4 == 0);
        assert(merged == begin / 4);
        covered += rows;
        merged += rows / 4;
      }
      assert(covered == plan.count && merged == plan.count / 4);
      assert(plan.tile >= 1 && plan.tile <= plan.count);
      assert(std::size_t{16} * plan.tile * plan.count * 6 <=
                 gufo::models::qwen::vision::kAttentionTileBytes ||
             plan.tile == std::min<std::uint32_t>(plan.count, 128));
    }
  }
  const auto image = EstimateImageMemory(1024, 1024, 2560);
  assert(image.embedding_bytes == 10485760);
  // Allocation census: 4096 patches, 64-row attention tiles, RGB GPU and
  // pinned upload buffers. Runtime/BLAS workspace is outside this census.
  assert(image.scratch_bytes == 194150400);
  const auto native = EstimateImageMemory(1024, 1024, 2560, true);
  assert(native.embedding_bytes == image.embedding_bytes);
  assert(native.scratch_bytes > image.scratch_bytes + 128ULL * 1024 * 1024);
  const auto large = EstimateImageMemory(4096, 4096, 2560);
  assert(large.embedding_bytes == 167772160);
  // 65,536 patches: whole-image Q/K/V, residual and attention output stay;
  // row-local stages are chunked and attention tiles shrink to 16 rows.
  assert(large.scratch_bytes == 1179590656);
  assert(EstimateImageMemory(4096, 4096, 2560, true).scratch_bytes ==
         1505304576);
  // Typical screenshots, native EXL3 (production K8 vision): 1600x900 and
  // 4K resize to 1600x896 and 3840x2176.
  assert(EstimateImageMemory(1600, 896, 2560, true).scratch_bytes == 586907648);
  assert(EstimateImageMemory(3840, 2176, 2560, true).scratch_bytes ==
         1024966656);
  assert(EstimateImageMemory(2560, 1440, 2560, true).scratch_bytes ==
         785825792);
  for (const auto width : {0U, 31U, 8192U}) {
    try {
      (void)EstimateImageMemory(width, 4096, 2560);
      assert(false);
    } catch (const std::invalid_argument&) {
    }
  }
}
