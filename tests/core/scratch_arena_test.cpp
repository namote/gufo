// CPU checks of ScratchArena (src/core/hip/scratch_arena.*) with a fake page
// backend: carving stays inside the reservation and on 2 MiB boundaries,
// MapTransient backs every carved page in bounded runs and is idempotent, a
// failed map leaves nothing mapped, ReleaseTransient unmaps everything, and
// the reservation is freed on destruction.
#include "src/core/hip/scratch_arena.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>

namespace {

using gufo::hip::ScratchArena;

constexpr std::size_t kPage = ScratchArena::kPage;

int failures = 0;

void Expect(bool condition, const char* what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

struct Ledger {
  std::size_t reserved{0};
  std::size_t freed{0};
  /// Mapped runs by start offset: bytes.
  std::map<std::uintptr_t, std::size_t> mapped;
  std::size_t largest_map{0};
  std::size_t maps{0};
  /// Map call (1-based) that fails; 0 never.
  std::size_t fail_at{0};
  bool fail_reserve{false};
};

class FakePages final : public ScratchArena::Pages {
public:
  explicit FakePages(Ledger* ledger) : ledger_(ledger) {}
  void* Reserve(std::size_t bytes, std::string* error) override {
    if (ledger_->fail_reserve) {
      *error = "fake reserve failure";
      return nullptr;
    }
    ledger_->reserved = bytes;
    // Address space only: a fixed, page-aligned fake base never touched.
    return reinterpret_cast<void*>(std::uintptr_t{64} * kPage);
  }
  void Free(void*, std::size_t bytes) noexcept override {
    ledger_->freed = bytes;
  }
  bool Map(void* at, std::size_t bytes, std::uint64_t* handle,
           std::string* error) override {
    ++ledger_->maps;
    if (ledger_->fail_at == ledger_->maps) {
      *error = "fake map failure";
      return false;
    }
    const auto start = reinterpret_cast<std::uintptr_t>(at);
    if (start % kPage != 0 || bytes % kPage != 0 ||
        ledger_->mapped.contains(start))
      std::abort();
    ledger_->mapped[start] = bytes;
    ledger_->largest_map = std::max(ledger_->largest_map, bytes);
    *handle = start ^ 0x5a5a;
    return true;
  }
  void Unmap(void* at, std::size_t bytes,
             std::uint64_t handle) noexcept override {
    const auto start = reinterpret_cast<std::uintptr_t>(at);
    const auto it = ledger_->mapped.find(start);
    if (it == ledger_->mapped.end() || it->second != bytes ||
        handle != (start ^ 0x5a5a))
      std::abort();
    ledger_->mapped.erase(it);
  }

private:
  Ledger* ledger_;
};

std::size_t MappedBytes(const Ledger& ledger) {
  std::size_t bytes = 0;
  for (const auto& [start, size] : ledger.mapped)
    bytes += size;
  return bytes;
}

void TestCarveMapRelease() {
  Ledger ledger;
  std::string error;
  {
    auto arena = ScratchArena::Reserve(std::make_unique<FakePages>(&ledger),
                                       600 * kPage + 1, &error);
    Expect(arena != nullptr, "reservation succeeds");
    Expect(ledger.reserved == 601 * kPage, "capacity rounds up to pages");
    auto* a = static_cast<std::uint8_t*>(arena->Take(3 * kPage - 5, &error));
    auto* b = static_cast<std::uint8_t*>(arena->Take(1, &error));
    auto* c = static_cast<std::uint8_t*>(arena->Take(400 * kPage, &error));
    Expect(a != nullptr && b == a + 3 * kPage && c == b + kPage,
           "carves consecutive page-aligned ranges");
    Expect(arena->transient_bytes() == 404 * kPage,
           "transient bytes are the carved pages");
    Expect(ledger.mapped.empty() && !arena->transient_mapped(),
           "carving maps nothing");
    for (int round = 0; round < 3; ++round) {
      Expect(arena->MapTransient(&error), "map succeeds");
      Expect(arena->transient_mapped(), "mapped after MapTransient");
      Expect(MappedBytes(ledger) == 404 * kPage, "every carved page mapped");
      Expect(ledger.largest_map <= ScratchArena::kMaxRunBytes,
             "runs are bounded");
      const auto maps = ledger.maps;
      Expect(arena->MapTransient(&error) && ledger.maps == maps,
             "mapping again is a no-op");
      Expect(arena->Take(kPage, &error) == nullptr, "no carving while mapped");
      arena->ReleaseTransient();
      Expect(ledger.mapped.empty() && !arena->transient_mapped(),
             "release unmaps everything");
    }
    arena->ReleaseTransient();  // no-op when released
    Expect(arena->Take(197 * kPage, &error) != nullptr,
           "carving resumes after release up to the reservation");
    error.clear();
    Expect(arena->Take(1, &error) == nullptr && !error.empty(),
           "carving stops at the reservation");
    Expect(arena->MapTransient(&error), "map the full reservation");
  }
  {
    Ledger dense;
    auto arena = ScratchArena::Reserve(std::make_unique<FakePages>(&dense),
                                       4 * kPage, &error);
    auto* a = static_cast<std::uint8_t*>(arena->Take(100, &error, 256));
    auto* b = static_cast<std::uint8_t*>(arena->Take(10, &error, 256));
    auto* c = static_cast<std::uint8_t*>(arena->Take(kPage, &error, 256));
    Expect(a != nullptr && b == a + 256 && c == b + 256,
           "aligned carves pack densely");
    Expect(arena->carved_bytes() == 512 + kPage &&
               arena->transient_bytes() == 2 * kPage,
           "transient bytes cover the touched pages only");
    Expect(arena->Take(1, &error, 3) == nullptr, "alignment must be 2^n");
    auto* d = static_cast<std::uint8_t*>(arena->Take(1, &error));
    Expect(d == a + 2 * kPage, "default alignment is a page");
    Expect(arena->Take(2 * kPage, &error, 256) == nullptr,
           "dense carves stop at the reservation");
    Expect(arena->MapTransient(&error) && MappedBytes(dense) == 3 * kPage,
           "maps the touched pages");
  }
  Expect(ledger.mapped.empty(), "destruction unmaps mapped pages");
  Expect(ledger.freed == 601 * kPage, "destruction frees the reservation");
}

void TestFailedMapLeavesNothingMapped() {
  Ledger ledger;
  std::string error;
  auto arena = ScratchArena::Reserve(std::make_unique<FakePages>(&ledger),
                                     1000 * kPage, &error);
  Expect(arena->Take(900 * kPage, &error) != nullptr, "carve");
  ledger.fail_at = 3;
  error.clear();
  Expect(!arena->MapTransient(&error), "a failed run fails the map");
  Expect(error == "fake map failure", "the backend's error is reported");
  Expect(ledger.mapped.empty() && !arena->transient_mapped(),
         "a failed map leaves nothing mapped");
  ledger.fail_at = 0;
  Expect(arena->MapTransient(&error) && MappedBytes(ledger) == 900 * kPage,
         "a later map succeeds");
}

void TestReservationFailures() {
  std::string error;
  Expect(ScratchArena::Reserve(nullptr, kPage, &error) == nullptr &&
             !error.empty(),
         "no backend, no arena");
  Ledger ledger;
  error.clear();
  Expect(ScratchArena::Reserve(std::make_unique<FakePages>(&ledger), 0,
                               &error) == nullptr,
         "an empty reservation is refused");
  ledger.fail_reserve = true;
  error.clear();
  Expect(ScratchArena::Reserve(std::make_unique<FakePages>(&ledger), kPage,
                               &error) == nullptr &&
             error == "fake reserve failure",
         "a failed reservation reports the backend's error");
}

}  // namespace

int main() {
  TestCarveMapRelease();
  TestFailedMapLeavesNothingMapped();
  TestReservationFailures();
  if (failures != 0) {
    std::fprintf(stderr, "scratch_arena_test: %d failures\n", failures);
    return 1;
  }
  std::puts("scratch_arena_test: PASS");
  return 0;
}
