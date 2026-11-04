#include "src/cli/serve/request_memory_budget.hpp"

#include <cassert>
#include <future>
#include <latch>
#include <limits>
#include <stdexcept>

#include "src/models/qwen/vision/memory.hpp"

int main() {
  using gufo::server::RequestMemoryBudget;
  RequestMemoryBudget budget;
  auto available = [] { return std::size_t{1000}; };
  auto first = budget.Acquire({100, 300}, available);
  assert(budget.ReservedBytes() == 400);
  auto second = budget.Acquire({200, 200}, available);
  assert(budget.ReservedBytes() == 600);  // shared scratch max, state sum
  assert(!budget.Ready({500, 400}, available));
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
  // Fresh requests that cannot fit must fail, not wait forever for no owner.
  try {
    (void)budget.Ready({1001, 0}, available);
    assert(false);
  } catch (const std::length_error&) {
  }
  assert(budget.Ready({}, [] { return std::size_t{0}; }));
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
    assert(!budget.Ready({200, 200}, available));
    commit.reset();  // physical commit failure/cancellation releases promise
    assert(budget.Ready({200, 200}, available));
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
  using gufo::models::qwen::vision::EstimateImageMemory;
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
  assert(large.scratch_bytes > 3ULL * 1024 * 1024 * 1024);
  for (const auto width : {0U, 31U, 8192U}) {
    try {
      (void)EstimateImageMemory(width, 4096, 2560);
      assert(false);
    } catch (const std::invalid_argument&) {
    }
  }
}
