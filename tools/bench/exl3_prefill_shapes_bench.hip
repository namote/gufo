// Flash-Next EXL3 prefill microbenchmarks at the production shapes (W3,
// plan/exl3/F3-PLAN.md): plain HC operand (PlainPrefillGemm FP16 WMMA vs
// LaunchPlainPrefillStable), X3 trellis WMMA (LaunchExl3PrefillGemm) vs
// reconstruct-once to FP16 + the Flash-Next FP16 WMMA GEMM. Linked against
// the build's libraries (tools/bench/exl3_prefill_shapes_bench.sh).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "src/core/hip/detail/exl3_gemv.hpp"
#include "src/models/qwen/hip/ops/plain_prefill_stable.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace fn = gufo::models::qwen38_flash_next::rocm;
using gufo::quant::exl3::BitRate;
using gufo::quant::exl3::Codebook;
using gufo::quant::exl3::DeviceLinear;

namespace {

void Check(hipError_t e) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(e));
    std::exit(1);
  }
}

template<class T>
T* Alloc(std::size_t n) {
  T* p = nullptr;
  Check(hipMalloc(&p, n * sizeof(T)));
  return p;
}

__global__ void Fill(std::uint32_t* p, std::size_t n, std::uint32_t seed) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       i < n; i += std::size_t(gridDim.x) * blockDim.x) {
    std::uint32_t x = static_cast<std::uint32_t>(i) * 0x9E3779B9U + seed;
    x ^= x >> 16;
    x *= 0x85EBCA6BU;
    x ^= x >> 13;
    p[i] = x;
  }
}

// FP16 values in [-1, 1) and FP16 scales near 1.
__global__ void FillHalf(std::uint16_t* p, std::size_t n, std::uint32_t seed,
                         bool scale) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       i < n; i += std::size_t(gridDim.x) * blockDim.x) {
    std::uint32_t x = static_cast<std::uint32_t>(i) * 0x9E3779B9U + seed;
    x ^= x >> 15;
    x *= 0x2C1B3C6DU;
    const float f = scale ? 0.75F + float(x % 512) / 1024.0F
                          : (float(x % 65536) / 32768.0F - 1.0F);
    p[i] = __half_as_ushort(__float2half(f));
  }
}

__global__ void FillFloat(float* p, std::size_t n, std::uint32_t seed) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       i < n; i += std::size_t(gridDim.x) * blockDim.x) {
    std::uint32_t x = static_cast<std::uint32_t>(i) * 0x9E3779B9U + seed;
    x ^= x >> 15;
    x *= 0x2C1B3C6DU;
    p[i] = float(x % 65536) / 32768.0F - 1.0F;
  }
}

// Reconstruct-once: decode every trellis tile into FP16 W'[n][k] (row = output
// channel, the Flash-Next FP16 GEMM's weight layout). One wave per tile.
template<int K>
__global__ void Reconstruct(const std::uint32_t* trellis, unsigned k,
                            unsigned n, std::uint16_t* out) {
  const unsigned lane = threadIdx.x % 32;
  const unsigned tile = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  const unsigned n_tiles = n / 16;
  if (tile >= (k / 16) * n_tiles)
    return;
  const unsigned kt = tile / n_tiles, nt = tile % n_tiles;
  const auto lw = gufo::hip::exl3::MakeLaneWindow<K>(lane);
  gufo::hip::exl3::Half2 w[4];
  gufo::hip::exl3::DecodeLaneMul1<K>(trellis + std::size_t(tile) * 8 * K, lw,
                                     w);
  const unsigned r0 = (lane % 4) * 2, c0 = lane / 4;
  for (int q = 0; q < 4; ++q) {
    const unsigned col = nt * 16 + c0 + 8 * (q / 2);
    const unsigned row = kt * 16 + r0 + 8 * (q % 2);
    // [col][row]: two consecutive k of one channel, one dword store.
    *reinterpret_cast<std::uint32_t*>(out + std::size_t(col) * k + row) =
        __builtin_bit_cast(std::uint32_t, w[q]);
  }
}

float TimeIt(const auto& fn, int reps) {
  hipEvent_t a, b;
  Check(hipEventCreate(&a));
  Check(hipEventCreate(&b));
  for (int i = 0; i < 2; ++i)
    fn();
  Check(hipEventRecord(a));
  for (int i = 0; i < reps; ++i)
    fn();
  Check(hipEventRecord(b));
  Check(hipEventSynchronize(b));
  float ms = 0;
  Check(hipEventElapsedTime(&ms, a, b));
  Check(hipEventDestroy(a));
  Check(hipEventDestroy(b));
  return ms / static_cast<float>(reps);
}

struct Shape {
  const char* name;
  unsigned k, n, bits;
};

}  // namespace

int main(int argc, char** argv) {
  const unsigned rows = argc > 1 ? static_cast<unsigned>(std::atoi(argv[1])) : 2048;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
  std::printf("rows %u reps %d\n", rows, reps);
  // Plain HC operands: down 10240 -> 320 and up 320 -> 10240, F16.
  for (auto [k, n] : {std::pair{10240U, 320U}, std::pair{320U, 10240U},
                      std::pair{2560U, 544U}}) {
    auto* w = Alloc<std::uint16_t>(std::size_t(n) * k);
    auto* x = Alloc<float>(std::size_t(rows) * k);
    auto* xh = Alloc<std::uint16_t>(std::size_t(rows) * k);
    auto* y = Alloc<float>(std::size_t(rows) * n);
    hipLaunchKernelGGL(FillHalf, dim3(1024), dim3(256), 0, 0, w,
                       std::size_t(n) * k, 3U, false);
    hipLaunchKernelGGL(FillFloat, dim3(1024), dim3(256), 0, 0, x,
                       std::size_t(rows) * k, 5U);
    Check(hipDeviceSynchronize());
    const double flop = 2.0 * rows * k * n;
    const float stable = TimeIt(
        [&] {
          gufo::hip::LaunchPlainPrefillStable(gufo::core::GgmlType::kF16, w, x,
                                              y, rows, n, k, nullptr);
        },
        reps);
    const float narrow = TimeIt(
        [&] { fn::NarrowActivations(x, xh, false, std::size_t(rows) * k, nullptr); },
        reps);
    const float wmma = TimeIt(
        [&] {
          fn::PlainPrefillGemm(w, fn::WeightType::kF16,
                               reinterpret_cast<const __half*>(xh), y, rows, n,
                               k, nullptr);
        },
        reps);
    std::printf(
        "plain %5u->%-5u: PlainStable %.3f ms (%.2f TFLOPS) | narrow %.3f "
        "ms + FP16 WMMA %.3f ms (%.2f TFLOPS)\n",
        k, n, stable, flop / stable / 1e9, narrow, wmma, flop / wmma / 1e9);
    Check(hipFree(w));
    Check(hipFree(x));
    Check(hipFree(xh));
    Check(hipFree(y));
  }
  const Shape shapes[] = {
      {"gdn qkv", 2560, 10240, 7}, {"gdn z", 2560, 6144, 7},
      {"gdn out", 6144, 2560, 7},  {"attn q", 2560, 12288, 7},
      {"attn k", 2560, 512, 7},    {"attn o", 6144, 2560, 7},
      {"shared gate", 2560, 640, 7}, {"shared down", 640, 2560, 7},
      {"indexer", 2560, 640, 5},
  };
  for (const Shape& s : shapes) {
    const std::size_t words = std::size_t(s.k) * s.n * s.bits / 32;
    auto* t = Alloc<std::uint32_t>(words);
    auto* svh = Alloc<std::uint16_t>(s.n);
    auto* xh = Alloc<std::uint16_t>(std::size_t(rows) * s.k);
    auto* y = Alloc<float>(std::size_t(rows) * s.n);
    auto* wf = Alloc<std::uint16_t>(std::size_t(s.k) * s.n);
    hipLaunchKernelGGL(Fill, dim3(1024), dim3(256), 0, 0, t, words, s.k + s.n);
    hipLaunchKernelGGL(FillHalf, dim3(64), dim3(256), 0, 0, svh, std::size_t(s.n),
                       7U, true);
    hipLaunchKernelGGL(FillHalf, dim3(1024), dim3(256), 0, 0, xh,
                       std::size_t(rows) * s.k, 9U, false);
    Check(hipDeviceSynchronize());
    DeviceLinear l{};
    l.trellis = reinterpret_cast<const std::uint16_t*>(t);
    l.svh = svh;
    l.suh = svh;
    l.k = s.k;
    l.n = s.n;
    l.rate = BitRate{static_cast<int>(s.bits), false};
    l.codebook = Codebook::kMul1;
    const double flop = 2.0 * rows * s.k * s.n;
    const float x3 = TimeIt(
        [&] { gufo::hip::LaunchExl3PrefillGemm(l, xh, y, rows, false, nullptr); },
        reps);
    const unsigned tiles = (s.k / 16) * (s.n / 16);
    const float recon = TimeIt(
        [&] {
          if (s.bits == 7)
            hipLaunchKernelGGL(Reconstruct<7>, dim3((tiles + 7) / 8), dim3(256),
                               0, 0, t, s.k, s.n, wf);
          else
            hipLaunchKernelGGL(Reconstruct<5>, dim3((tiles + 7) / 8), dim3(256),
                               0, 0, t, s.k, s.n, wf);
        },
        reps);
    const float f16 = TimeIt(
        [&] {
          fn::UnquantizedF16Gemm(wf, reinterpret_cast<const __half*>(xh), y,
                                 rows, s.n, s.k, nullptr);
        },
        reps);
    std::printf(
        "%-12s K%u %5u->%-5u X3 %.3f ms %.2f TFLOPS | reconstruct %.3f ms + "
        "FP16 WMMA %.3f ms %.2f TFLOPS = %.3f ms\n",
        s.name, s.bits, s.k, s.n, x3, flop / x3 / 1e9, recon, f16,
        flop / f16 / 1e9, recon + f16);
    for (void* p : {static_cast<void*>(t), static_cast<void*>(svh),
                    static_cast<void*>(xh), static_cast<void*>(y),
                    static_cast<void*>(wf)})
      Check(hipFree(p));
  }
  // Routed experts: 512 K5 experts, rows * 10 assignments routed uniformly
  // at random, packed per expert into tile_rows-row buckets.
  for (auto [k, n] : {std::pair{2560U, 640U}, std::pair{640U, 2560U}}) {
    constexpr unsigned kExperts = 512, kUsed = 10;
    const unsigned assignments = rows * kUsed;
    const std::size_t words = std::size_t(k) * n * 5 / 32;
    auto* t = Alloc<std::uint32_t>(words * kExperts);
    auto* svh = Alloc<std::uint16_t>(n);
    hipLaunchKernelGGL(Fill, dim3(4096), dim3(256), 0, 0, t, words * kExperts,
                       11U);
    std::vector<DeviceLinear> bank(kExperts);
    for (unsigned e = 0; e < kExperts; ++e) {
      bank[e].trellis = reinterpret_cast<const std::uint16_t*>(t + words * e);
      bank[e].svh = svh;
      bank[e].suh = svh;
      bank[e].k = k;
      bank[e].n = n;
      bank[e].rate = BitRate{5, false};
      bank[e].codebook = Codebook::kMul1;
    }
    std::vector<unsigned> per(kExperts, 0);
    std::uint32_t lcg = 12345;
    for (unsigned a = 0; a < assignments; ++a) {
      lcg = lcg * 1664525U + 1013904223U;
      ++per[(lcg >> 8) % kExperts];
    }
    auto* d_bank = Alloc<DeviceLinear>(kExperts);
    Check(hipMemcpy(d_bank, bank.data(), kExperts * sizeof(DeviceLinear),
                    hipMemcpyHostToDevice));
    auto* xh = Alloc<std::uint16_t>(std::size_t(assignments) * k);
    auto* y = Alloc<float>(std::size_t(assignments) * n);
    hipLaunchKernelGGL(FillHalf, dim3(1024), dim3(256), 0, 0, xh,
                       std::size_t(assignments) * k, 13U, false);
    const double unique = double(words) * 4 * kExperts;
    for (unsigned tile_rows : {16U, 32U, 64U}) {
      std::vector<gufo::hip::Exl3PrefillBucket> buckets;
      unsigned first = 0;
      for (unsigned e = 0; e < kExperts; ++e) {
        for (unsigned r = 0; r < per[e]; r += tile_rows)
          buckets.push_back({e, std::min(tile_rows, per[e] - r), first + r});
        first += per[e];
      }
      const unsigned count = static_cast<unsigned>(buckets.size());
      const unsigned max_buckets = kExperts + assignments / tile_rows;
      buckets.resize(max_buckets, gufo::hip::Exl3PrefillBucket{0, 0, 0});
      auto* d_buckets = Alloc<gufo::hip::Exl3PrefillBucket>(max_buckets);
      auto* d_count = Alloc<std::uint32_t>(1);
      Check(hipMemcpy(d_buckets, buckets.data(),
                      max_buckets * sizeof(gufo::hip::Exl3PrefillBucket),
                      hipMemcpyHostToDevice));
      Check(hipMemcpy(d_count, &count, 4, hipMemcpyHostToDevice));
      const float ms = TimeIt(
          [&] {
            gufo::hip::LaunchExl3PrefillBucketInner(d_bank, d_buckets, d_count,
                                                    max_buckets, tile_rows, xh,
                                                    y, k, n,
                                                    BitRate{5, false}, nullptr);
          },
          reps);
      const double flop = 2.0 * assignments * k * n;
      std::printf(
          "routed %4u->%-4u tile_rows %2u: %u buckets (max %u) %.3f ms %.2f "
          "TFLOPS, unique weights %.0f MB -> %.0f GB/s\n",
          k, n, tile_rows, count, max_buckets, ms, flop / ms / 1e9,
          unique / 1e6, unique / ms / 1e6);
      Check(hipFree(d_buckets));
      Check(hipFree(d_count));
    }
    for (void* p : {static_cast<void*>(t), static_cast<void*>(svh),
                    static_cast<void*>(d_bank), static_cast<void*>(xh),
                    static_cast<void*>(y)})
      Check(hipFree(p));
  }
  return 0;
}
