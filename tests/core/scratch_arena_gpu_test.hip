// GPU checks of transient pass scratch on HIP virtual memory
// (src/core/hip/scratch_arena*): carved ranges are usable by kernels only
// while mapped, keep their contents within a mapping, map and release
// repeatedly, and the reservation bounds Take. Ported from upstream #1265's
// tests/mem/scratch_arena_test.cpp (no resident ranges here).
#include <hip/hip_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "src/core/hip/scratch_arena.hpp"

namespace {

using gufo::hip::ScratchArena;

constexpr std::size_t kPage = ScratchArena::kPage;

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
    std::exit(1);
  }
}

__global__ void Fill(std::uint32_t* data, std::size_t words,
                     std::uint32_t salt) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
       i += gridDim.x * blockDim.x)
    data[i] = salt ^ static_cast<std::uint32_t>(i * 2654435761u);
}

void FillWords(void* data, std::size_t bytes, std::uint32_t salt) {
  Fill<<<256, 256>>>(static_cast<std::uint32_t*>(data), bytes / 4, salt);
  Check(hipGetLastError(), "fill launch");
  Check(hipDeviceSynchronize(), "fill");
}

bool Matches(const void* data, std::size_t bytes, std::uint32_t salt) {
  std::vector<std::uint32_t> host(bytes / 4);
  Check(hipMemcpy(host.data(), data, bytes, hipMemcpyDeviceToHost), "read");
  for (std::size_t i = 0; i < host.size(); ++i)
    if (host[i] != (salt ^ static_cast<std::uint32_t>(i * 2654435761u)))
      return false;
  return true;
}

void Require(bool condition, const char* what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    std::exit(1);
  }
}

}  // namespace

int main() {
  Check(hipSetDevice(0), "device");
  std::string reason;
  auto pages = gufo::hip::VmmScratchPages(0, &reason);
  if (!pages) {
    // Callers then never run wide passes; nothing else to check.
    std::printf("scratch_arena_gpu_test: no virtual memory (%s)\n",
                reason.c_str());
    return 0;
  }
  std::string error;
  auto arena = ScratchArena::Reserve(std::move(pages), 300 * kPage, &error);
  Require(arena != nullptr, "reservation");

  auto* a = static_cast<std::uint8_t*>(arena->Take(3 * kPage, &error));
  auto* b = static_cast<std::uint8_t*>(arena->Take(200 * kPage, &error));
  Require(a != nullptr && b != nullptr, "carve");
  Require(reinterpret_cast<std::uintptr_t>(b) % kPage == 0, "alignment");
  Require(arena->transient_bytes() == 203 * kPage, "transient bytes");

  for (std::uint32_t round = 0; round < 3; ++round) {
    Require(arena->MapTransient(&error), "map");
    Require(arena->MapTransient(&error), "map again");
    FillWords(a, 3 * kPage, 10 + round);
    FillWords(b, 200 * kPage, 20 + round);
    Require(Matches(a, 3 * kPage, 10 + round), "first range contents");
    Require(Matches(b, 200 * kPage, 20 + round),
            "second range contents across mapping runs");
    arena->ReleaseTransient();
    Require(!arena->transient_mapped(), "released");
  }
  Require(arena->Take(98 * kPage, &error) == nullptr, "reservation bound");
  arena.reset();
  std::puts("scratch_arena_gpu_test: PASS");
  return 0;
}
