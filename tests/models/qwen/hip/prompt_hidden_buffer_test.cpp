#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/cache/ledger.hpp"
#include "src/models/qwen/hip/prompt_hidden_buffer.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

namespace {

void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "Prompt hidden buffer test failure: " << message << '\n';
    std::abort();
  }
}

bool IsPinned(const float* data) {
  hipPointerAttribute_t attributes{};
  if (hipPointerGetAttributes(&attributes, data) != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  return attributes.type == hipMemoryTypeHost;
}

void Fill(gufo::hip::PromptHiddenBuffer& buffer, std::size_t from) {
  float* data = buffer.Data();
  for (std::size_t i = from; i < buffer.Size(); ++i) {
    data[i] = static_cast<float>(i);
  }
}

bool Holds(const gufo::hip::PromptHiddenBuffer& buffer) {
  const auto view = buffer.View();
  for (std::size_t i = 0; i < view.size(); ++i) {
    if (view[i] != static_cast<float>(i)) {
      return false;
    }
  }
  return true;
}

// A prompt within the bound captures into pinned memory and keeps its
// capacity across prompts.
void TestPinnedWithinBound() {
  gufo::hip::PromptHiddenBuffer buffer;
  buffer.Reserve(4096, 8192);
  buffer.Resize(1024);
  Fill(buffer, 0);
  buffer.Resize(4096);
  Fill(buffer, 1024);
  Check(IsPinned(buffer.Data()), "bounded capture must be pinned");
  Check(Holds(buffer), "bounded capture must keep its rows");
  const float* first = buffer.Data();
  buffer.Clear();
  Check(buffer.Size() == 0, "clear must empty the capture");
  buffer.Reserve(2048, 8192);
  Check(buffer.Data() == first, "a smaller prompt must reuse the capacity");
}

// Growth inside the bound keeps the captured rows.
void TestPinnedGrowth() {
  gufo::hip::PromptHiddenBuffer buffer;
  buffer.Reserve(1024, 8192);
  buffer.Resize(1024);
  Fill(buffer, 0);
  buffer.Resize(6000);
  Fill(buffer, 1024);
  Check(IsPinned(buffer.Data()), "growth inside the bound must stay pinned");
  Check(Holds(buffer), "growth must keep the captured rows");
}

// A larger request uses pageable memory, and growth past the bound moves the
// captured rows there.
void TestPageableBeyondBound() {
  gufo::hip::PromptHiddenBuffer reserved;
  reserved.Reserve(16384, 8192);
  reserved.Resize(16384);
  Fill(reserved, 0);
  Check(!IsPinned(reserved.Data()),
        "a request past the bound must be pageable");
  Check(Holds(reserved), "pageable capture must keep its rows");

  gufo::hip::PromptHiddenBuffer grown;
  grown.Reserve(4096, 8192);
  grown.Resize(4096);
  Fill(grown, 0);
  grown.Resize(12288);
  Fill(grown, 4096);
  Check(!IsPinned(grown.Data()), "growth past the bound must be pageable");
  Check(Holds(grown), "moving to pageable memory must keep the rows");

  grown.Clear();
  grown.Reserve(4096, 8192);
  grown.Resize(4096);
  Check(IsPinned(grown.Data()), "the next bounded prompt must be pinned again");
}

std::size_t ChargedBytes(const gufo::cache::ResourceLedger& ledger) {
  return ledger.Snapshot().bytes[static_cast<std::size_t>(
      gufo::cache::ResourceCategory::kMetadata)];
}

// A ledger is charged for exactly the pinned capacity until the buffer frees
// it. A refused charge captures into pageable memory and holds nothing.
void TestLedgerCharge() {
  constexpr std::size_t kLimit = std::size_t{1} << 20;
  gufo::cache::ResourceLedger ledger({kLimit, kLimit, 0, kLimit});
  {
    gufo::hip::PromptHiddenBuffer buffer;
    buffer.Reserve(1024, 8192, &ledger);
    Check(IsPinned(buffer.Data()), "a charged capture must be pinned");
    Check(ChargedBytes(ledger) == 1024 * sizeof(float),
          "the pinned bytes must be charged");
    buffer.Resize(1024);
    Fill(buffer, 0);
    buffer.Resize(6000);
    Fill(buffer, 1024);
    Check(Holds(buffer), "charged growth must keep the captured rows");
    Check(ChargedBytes(ledger) == 6000 * sizeof(float),
          "growth must charge only the new capacity");
    // The ledger stays attached: a later prompt that grows is charged too.
    buffer.Clear();
    buffer.Reserve(8192, 8192);
    Check(IsPinned(buffer.Data()), "a later growth must stay pinned");
    Check(ChargedBytes(ledger) == 8192 * sizeof(float),
          "a later growth must be charged");
  }
  Check(ChargedBytes(ledger) == 0, "destruction must release the charge");

  gufo::cache::ResourceLedger small({4096, 4096, 0, 4096});
  gufo::hip::PromptHiddenBuffer refused;
  refused.Reserve(4096, 8192, &small);
  refused.Resize(4096);
  Fill(refused, 0);
  Check(!IsPinned(refused.Data()), "a refused charge must capture pageable");
  Check(Holds(refused), "a refused charge must keep its rows");
  Check(ChargedBytes(small) == 0, "a refused charge must not be held");

  // The fallback is sticky: a later prompt does not retry the pinned
  // allocation, even one the ledger could now admit.
  refused.Clear();
  refused.Reserve(512, 8192);
  refused.Resize(512);
  Check(!IsPinned(refused.Data()), "a failed pin must not be retried");
  Check(ChargedBytes(small) == 0, "a failed pin must stay uncharged");
}

// Slots share one capture's pinned pages: they are allocated and charged once,
// a later capture takes them over, and the earlier rows are then reported
// missing instead of returned with another slot's contents.
void TestSharedPages() {
  constexpr std::size_t kLimit = std::size_t{1} << 20;
  gufo::cache::ResourceLedger ledger({kLimit, kLimit, 0, kLimit});
  auto pages = std::make_shared<gufo::hip::PromptHiddenPinned>();
  {
    gufo::hip::PromptHiddenBuffer first;
    gufo::hip::PromptHiddenBuffer second;
    first.Share(pages, nullptr);
    second.Share(pages, nullptr);
    first.Reserve(4096, 8192, &ledger);
    second.Reserve(4096, 8192, &ledger);
    Check(pages->ChargedBytes() == 4096 * sizeof(float),
          "shared pages must be charged once");
    Check(ChargedBytes(ledger) == 4096 * sizeof(float),
          "the ledger must hold one capture's pages");

    first.Clear();
    first.Reserve(4096, 8192);
    first.Resize(4096);
    Fill(first, 0);
    Check(IsPinned(first.Data()), "a shared capture must be pinned");
    Check(Holds(first), "the holder must keep its rows");
    const float* data = first.Data();

    second.Clear();
    second.Reserve(2048, 8192);
    second.Resize(2048);
    Fill(second, 0);
    Check(second.Data() == data, "a later capture must reuse the pages");
    Check(Holds(second), "the new holder must keep its rows");
    Check(first.View().empty(), "taken-over rows must be reported missing");
    bool refused = false;
    try {
      first.Resize(8192);
    } catch (const std::logic_error&) {
      refused = true;
    }
    Check(refused, "growing taken-over rows must fail");

    first.Clear();
    first.Reserve(1024, 8192);
    first.Resize(1024);
    Fill(first, 0);
    Check(first.Data() == data && Holds(first),
          "a cleared buffer must capture into the pages again");
    second.Clear();
    Check(second.View().empty(), "a cleared buffer holds no rows");
  }
  Check(pages->ChargedBytes() == 4096 * sizeof(float),
        "the pages outlive the buffers that shared them");
  pages.reset();
  Check(ChargedBytes(ledger) == 0, "freeing the pages must release the charge");
}

}  // namespace
#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  if (const int gate = gufo::test::GateHipDevice(
          gufo::test::HipDeviceRequirement::kOptional, "prompt hidden buffer");
      gate != gufo::test::kHipTestSuccess) {
    return gate;
  }
  TestPinnedWithinBound();
  TestPinnedGrowth();
  TestPageableBeyondBound();
  TestLedgerCharge();
  TestSharedPages();
  std::cout << "Prompt hidden buffer tests passed.\n";
  return EXIT_SUCCESS;
#else
  std::cerr << "HIP disabled; prompt hidden buffer test unavailable.\n";
  return EXIT_FAILURE;
#endif
}
