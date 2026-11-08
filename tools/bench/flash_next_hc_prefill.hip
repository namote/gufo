// Flash-Next EXL3 prefill HC combine (B6 P5): the four-lane combine as the
// EXL3 prefill route runs it (residual update written back, F32 norm left
// out, per-stream scales and the tiled Q8 norm for the next mixer), then the
// W8A8 HC down projection (10240 -> 320) that reads that Q8 norm, alone and
// together. c03 shapes, 2,048 and 4,096 rows (and 256, a short chunk that
// keeps plain stores). The residual store policy is
// the B6 change: build this bench at the parent commit and at the change and
// compare µs/call (best of kReps averages over kCalls calls) and the printed
// FNV-1a hashes of the residual, Q8 norm and projection, which must match.
//
// Build (CMake, links the production library):
//   cmake --build --preset gpu-test --target qwen38_flash_next_hc_prefill_bench
// Run:
//   build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_hc_prefill_bench
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace fn = gufo::models::qwen38_flash_next::rocm;

namespace {

constexpr unsigned kHidden = 2560, kStreams = 4, kRank = 320;
constexpr unsigned kHc = kHidden * kStreams;
constexpr int kCalls = 8;
constexpr int kReps = 5;

void Ck(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}

std::uint32_t Next(std::uint32_t* s) {
  *s ^= *s << 13;
  *s ^= *s >> 17;
  *s ^= *s << 5;
  return *s;
}

template<class T>
T* Upload(const std::vector<T>& h) {
  T* d = nullptr;
  Ck(hipMalloc(&d, h.size() * sizeof(T)), "malloc");
  Ck(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice),
     "upload");
  return d;
}

std::vector<float> Values(std::size_t n, std::uint32_t seed, float scale) {
  std::vector<float> v(n);
  for (auto& x : v)
    x = scale * (static_cast<float>(Next(&seed) % 2001) / 1000.0F - 1.0F);
  return v;
}

std::uint64_t Hash(const void* d, std::size_t bytes) {
  std::vector<std::uint8_t> h(bytes);
  Ck(hipMemcpy(h.data(), d, bytes, hipMemcpyDeviceToHost), "download");
  std::uint64_t x = 1469598103934665603ULL;
  for (auto c : h)
    x = (x ^ c) * 1099511628211ULL;
  return x;
}

template<class F>
double TimeUs(hipStream_t stream, F&& body) {
  hipEvent_t a, b;
  Ck(hipEventCreate(&a), "event");
  Ck(hipEventCreate(&b), "event");
  body();
  Ck(hipStreamSynchronize(stream), "warm");
  float best = 1e30F;
  for (int r = 0; r < kReps; ++r) {
    Ck(hipEventRecord(a, stream), "record");
    for (int c = 0; c < kCalls; ++c)
      body();
    Ck(hipEventRecord(b, stream), "record");
    Ck(hipEventSynchronize(b), "sync");
    float ms = 0;
    Ck(hipEventElapsedTime(&ms, a, b), "elapsed");
    best = std::min(best, ms);
  }
  Ck(hipEventDestroy(a), "destroy");
  Ck(hipEventDestroy(b), "destroy");
  return 1e3 * best / kCalls;
}

}  // namespace

int main() {
  hipStream_t stream;
  Ck(hipStreamCreate(&stream), "stream");
  // Q8_0 HC down weights: 320 rows of 10240 (34-byte blocks).
  std::vector<std::uint8_t> w(std::size_t{kRank} * (kHc / 32) * 34);
  {
    std::uint32_t s = 0xC0DE0001U;
    for (std::size_t b = 0; b < w.size(); b += 34) {
      const _Float16 d = static_cast<_Float16>(0.002F);
      std::memcpy(&w[b], &d, 2);
      for (int i = 2; i < 34; ++i)
        w[b + i] = static_cast<std::uint8_t>(Next(&s));
    }
  }
  const auto* d_w = Upload(w);
  const auto* d_gamma = Upload(Values(kHc, 0x6A33A001U, 1.0F));
  for (const unsigned rows : {256U, 2048U, 4096U}) {
    const auto res0 = Values(std::size_t{rows} * kHc, 0x5E5E0000U + rows, 4.0F);
    auto* d_res = Upload(res0);
    const auto* d_block =
        Upload(Values(std::size_t{rows} * kHidden, 0xB10C0000U + rows, 1.0F));
    const auto* d_inject =
        Upload(Values(std::size_t{rows} * kStreams, 0x1A1A0000U + rows, 2.0F));
    auto* d_xn = Upload(std::vector<float>(kStreams));  // F32 norm left out
    auto* d_scale = Upload(std::vector<float>(std::size_t{rows} * kStreams));
    const std::size_t q8_bytes = fn::Q8TiledBytes(rows, kHc);
    auto* d_q8 = Upload(std::vector<std::uint8_t>(q8_bytes));
    auto* d_lo = Upload(std::vector<float>(std::size_t{rows} * kRank));
    const auto combine = [&] {
      fn::HcCombine(d_res, d_block, d_inject, 1, d_gamma, d_xn, rows, kHidden,
                    kStreams, 1.0e-6F, stream, false, true, nullptr, d_scale,
                    nullptr, d_q8, nullptr, {});
      Ck(hipGetLastError(), "combine");
    };
    const auto down = [&] {
      if (!fn::W8A8Gemm(d_w, d_q8, d_lo, rows, kRank, kHc, stream))
        std::exit(1);
    };
    // Hashes from one call on the initial residual.
    combine();
    down();
    Ck(hipStreamSynchronize(stream), "hash");
    std::printf("rows %u: hash res %016llx q8 %016llx down %016llx\n", rows,
                static_cast<unsigned long long>(
                    Hash(d_res, std::size_t{rows} * kHc * sizeof(float))),
                static_cast<unsigned long long>(Hash(d_q8, q8_bytes)),
                static_cast<unsigned long long>(
                    Hash(d_lo, std::size_t{rows} * kRank * sizeof(float))));
    // Timed calls keep updating res; restore it between sections.
    const auto reset = [&] {
      Ck(hipMemcpy(d_res, res0.data(), res0.size() * sizeof(float),
                   hipMemcpyHostToDevice),
         "reset");
    };
    reset();
    const double c_us = TimeUs(stream, combine);
    reset();
    const double d_us = TimeUs(stream, down);
    reset();
    const double cd_us = TimeUs(stream, [&] {
      combine();
      down();
    });
    std::printf("  combine %8.1f  HC down %8.1f  combine + down %8.1f µs\n",
                c_us, d_us, cd_us);
    for (const void* p :
         {static_cast<const void*>(d_res), static_cast<const void*>(d_block),
          static_cast<const void*>(d_inject), static_cast<const void*>(d_xn),
          static_cast<const void*>(d_scale), static_cast<const void*>(d_q8),
          static_cast<const void*>(d_lo)})
      Ck(hipFree(const_cast<void*>(p)), "free");
  }
  return 0;
}
