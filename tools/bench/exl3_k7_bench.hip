// Flash-Next K7 dense decode/verify timing (plan/exl3/F3-PLAN.md item 2):
// the production X2b GEMV (LaunchExl3FastProjections, including its input
// rotation) against the X3 narrow WMMA tile (LaunchExl3RotateRows +
// LaunchExl3PrefillGemm) at the Flash-Next K7 trunk shapes for 1..8 rows,
// with cold weights (copies rotated past the last-level cache). Reports
// microseconds per call and trellis GB/s. Single translation unit: the
// production kernels are compiled in.
//
//   hipcc -O3 -std=c++20 --offload-arch=gfx1151 -DENGINE_ENABLE_HIP=1 -I. \
//     tools/bench/exl3_k7_bench.hip -o exl3_k7_bench
//   exl3_k7_bench [reps] [bits]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#ifndef ENGINE_ENABLE_HIP
#define ENGINE_ENABLE_HIP 1
#endif
#include "src/models/qwen/hip/kernels/exl3_gemv.hip"
#include "src/models/qwen/hip/kernels/prefill_fp16.hip"

namespace {

using gufo::quant::exl3::BitRate;
using gufo::quant::exl3::Codebook;
using gufo::quant::exl3::DeviceLinear;

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

__global__ void FillKernel(std::uint32_t* p, std::size_t n,
                           std::uint32_t seed) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       i < n; i += std::size_t(gridDim.x) * blockDim.x) {
    std::uint32_t x = static_cast<std::uint32_t>(i) * 0x9E3779B9U + seed;
    x ^= x >> 16;
    x *= 0x85EBCA6BU;
    x ^= x >> 13;
    p[i] = x;
  }
}

// FP16 scales near 1 (0x3800..0x3c00 range), so outputs stay finite.
__global__ void ScaleKernel(std::uint16_t* p, std::size_t n) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       i < n; i += std::size_t(gridDim.x) * blockDim.x)
    p[i] = static_cast<std::uint16_t>(0x3800 + (i * 37) % 0x400);
}

struct Shape {
  const char* name;
  std::vector<std::pair<unsigned, unsigned>> mats;  // (k, n), one group
};

constexpr std::size_t kColdBytes = 512ull << 20;

float TimeIt(const auto& fn, int reps) {
  hipEvent_t a, b;
  Check(hipEventCreate(&a));
  Check(hipEventCreate(&b));
  // Warm up on copies 0..2, then measure from copy 3 on (the copy cursor
  // continues, so the measured calls never reuse a just-warmed copy).
  for (int i = 0; i < 3; ++i)
    fn(i);
  Check(hipEventRecord(a));
  for (int i = 0; i < reps; ++i)
    fn(3 + i);
  Check(hipEventRecord(b));
  Check(hipEventSynchronize(b));
  float ms = 0;
  Check(hipEventElapsedTime(&ms, a, b));
  Check(hipEventDestroy(a));
  Check(hipEventDestroy(b));
  return ms * 1000.0F / static_cast<float>(reps);
}

// The production pass (exl3_gemv.hip LaunchPass) without the rotation and
// with a forced launch shape: GEMV-only timing of the shape choices.
template<unsigned M>
void LaunchGemvShape(const std::vector<DeviceLinear>& g, float* const (&y)[3],
                     unsigned rows, const gufo::hip::Exl3FastWorkspace& ws,
                     gufo::hip::LaunchShape shape) {
  using namespace gufo::hip;
  auto* slab = reinterpret_cast<_Float16*>(ws.input);
  GemvArgs gemv{};
  gemv.slots = static_cast<unsigned>(g.size());
  gemv.k = g[0].k;
  gemv.rows = rows;
  std::size_t counter = 0;
  for (std::size_t s = 0; s < g.size(); ++s) {
    GemvSlot& slot = gemv.slot[s];
    slot.trellis = g[s].trellis;
    slot.a = slab + s * exl3::kMaxRows * g[0].k;
    slot.svh = reinterpret_cast<const _Float16*>(g[s].svh);
    slot.y_stride = g[s].n;
    slot.y = y[s];
    slot.counters = ws.counters + counter;
    slot.n = g[s].n;
    counter += g[s].n / exl3::kSegment;
  }
  LaunchGemvBits<M>(g[0].rate.bits, gemv, shape, nullptr);
}

void LaunchGemvShapeRows(const std::vector<DeviceLinear>& g,
                         float* const (&y)[3], unsigned rows,
                         const gufo::hip::Exl3FastWorkspace& ws,
                         gufo::hip::LaunchShape shape) {
  switch (rows) {
    case 1:
      LaunchGemvShape<1>(g, y, rows, ws, shape);
      break;
    case 2:
      LaunchGemvShape<2>(g, y, rows, ws, shape);
      break;
    case 3:
    case 4:
      LaunchGemvShape<4>(g, y, rows, ws, shape);
      break;
    default:
      LaunchGemvShape<8>(g, y, rows, ws, shape);
      break;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 200;
  const int bits = argc > 2 ? std::atoi(argv[2]) : 7;
  const BitRate rate{bits, false};
  const Shape shapes[] = {
      {"ssm qkv+z 2560->10240+6144", {{2560, 10240}, {2560, 6144}}},
      {"ssm out 6144->2560", {{6144, 2560}}},
      {"attn q+k+v 2560->12288+512+512",
       {{2560, 12288}, {2560, 512}, {2560, 512}}},
      {"attn o 6144->2560", {{6144, 2560}}},
      {"shared gate+up 2560->640x2", {{2560, 640}, {2560, 640}}},
      {"shared down 640->2560", {{640, 2560}}},
  };
  constexpr unsigned kMaxK = 6144, kMaxN = 12288, kRows = 8;
  float* x = Alloc<float>(kRows * kMaxK);
  {
    std::vector<float> h(kRows * kMaxK);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    for (auto& v : h)
      v = nd(rng);
    Check(hipMemcpy(x, h.data(), h.size() * 4, hipMemcpyHostToDevice));
  }
  std::uint16_t* scales = Alloc<std::uint16_t>(2 * kMaxN + 3 * kMaxK);
  hipLaunchKernelGGL(ScaleKernel, dim3(64), dim3(256), 0, 0, scales,
                     std::size_t(2 * kMaxN + 3 * kMaxK));
  float* y[3] = {Alloc<float>(kRows * kMaxN), Alloc<float>(kRows * kMaxN),
                 Alloc<float>(kRows * kMaxN)};
  std::uint16_t* slab_rows = Alloc<std::uint16_t>(3 * kRows * kMaxK);
  gufo::hip::Exl3FastWorkspace ws{};
  ws.input_elements = 3 * 8 * kMaxK;
  ws.input = Alloc<std::uint16_t>(ws.input_elements);
  ws.counter_count = 3 * kMaxN / 128;
  ws.counters = Alloc<std::uint32_t>(ws.counter_count);
  Check(hipMemset(ws.counters, 0, ws.counter_count * 4));
  std::printf("K%d reps %d; us per call, trellis GB/s (cold)\n", bits, reps);
  for (const Shape& s : shapes) {
    std::size_t group_bytes = 0;
    for (auto [k, n] : s.mats)
      group_bytes += std::size_t(k) * n * bits / 8;
    const std::size_t copies =
        std::max<std::size_t>(2, kColdBytes / group_bytes + 1);
    std::vector<std::vector<DeviceLinear>> lin(copies);
    std::vector<void*> owned;
    for (std::size_t c = 0; c < copies; ++c) {
      for (auto [k, n] : s.mats) {
        DeviceLinear l{};
        const std::size_t words = std::size_t(k) * n * bits / 32;
        auto* t = Alloc<std::uint32_t>(words);
        owned.push_back(t);
        hipLaunchKernelGGL(FillKernel, dim3(256), dim3(256), 0, 0, t, words,
                           static_cast<std::uint32_t>(c * 977 + k + n));
        l.trellis = reinterpret_cast<const std::uint16_t*>(t);
        // Each projection its own suh (production rotates per slot).
        l.suh = scales + 2 * kMaxN + lin[c].size() * kMaxK;
        l.svh = scales;
        l.k = k;
        l.n = n;
        l.rate = rate;
        l.codebook = Codebook::kMul1;
        lin[c].push_back(l);
      }
    }
    Check(hipDeviceSynchronize());
    std::printf("%s (%.2f MB/group, %zu copies)\n", s.name, group_bytes / 1e6,
                copies);
    for (unsigned rows : {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U}) {
      const float gemv = TimeIt(
          [&](int i) {
            const auto& g = lin[std::size_t(i) % copies];
            gufo::hip::Exl3FastTarget t[3];
            for (std::size_t m = 0; m < g.size(); ++m)
              t[m] = {&g[m], y[m], g[m].n};
            gufo::hip::LaunchExl3FastProjections({t, g.size()}, x, g[0].k, rows,
                                                 ws, nullptr);
          },
          reps);
      const float wmma = TimeIt(
          [&](int i) {
            const auto& g = lin[std::size_t(i) % copies];
            for (std::size_t m = 0; m < g.size(); ++m) {
              std::uint16_t* slab = slab_rows + m * kRows * g[0].k;
              gufo::hip::LaunchExl3RotateRows(x, g[0].k, rows, g[0].k, g[m].suh,
                                              slab, nullptr);
              gufo::hip::LaunchExl3PrefillGemm(g[m], slab, y[m], rows, false,
                                               nullptr);
            }
          },
          reps);
      std::printf(
          "  R=%u  gemv %8.1f us %6.1f GB/s | wmma %8.1f us %6.1f GB/s\n", rows,
          gemv, group_bytes / (gemv * 1e3), wmma, group_bytes / (wmma * 1e3));
    }
    // GEMV-only shape sweep. Before each row count, one production call
    // (untimed) packs the lane-major slabs for that M and gives the
    // reference output each forced shape is checked against.
    unsigned blocks_two_tiles = 0;
    for (const auto& l : lin[0])
      blocks_two_tiles += l.n / 32;
    const auto pick = gufo::hip::PickShape(lin[0][0].k / 16, blocks_two_tiles);
    std::printf("  shapes (tiles/wave x waves; production %ux%u):\n",
                pick.tiles_per_wave, pick.waves);
    for (unsigned rows : {1U, 2U, 4U, 8U}) {
      std::vector<std::vector<float>> want(lin[0].size());
      {
        gufo::hip::Exl3FastTarget t[3];
        for (std::size_t m = 0; m < lin[0].size(); ++m)
          t[m] = {&lin[0][m], y[m], lin[0][m].n};
        gufo::hip::LaunchExl3FastProjections({t, lin[0].size()}, x, lin[0][0].k,
                                             rows, ws, nullptr);
        Check(hipDeviceSynchronize());
        for (std::size_t m = 0; m < lin[0].size(); ++m) {
          want[m].resize(std::size_t(rows) * lin[0][m].n);
          Check(hipMemcpy(want[m].data(), y[m], want[m].size() * 4,
                          hipMemcpyDeviceToHost));
        }
      }
      std::printf("   R=%u", rows);
      for (gufo::hip::LaunchShape sh :
           {gufo::hip::LaunchShape{2, 4}, gufo::hip::LaunchShape{2, 8},
            gufo::hip::LaunchShape{2, 16}, gufo::hip::LaunchShape{4, 4}}) {
        LaunchGemvShapeRows(lin[0], y, rows, ws, sh);
        Check(hipDeviceSynchronize());
        double num = 0, den = 0;
        for (std::size_t m = 0; m < lin[0].size(); ++m) {
          std::vector<float> got(want[m].size());
          Check(hipMemcpy(got.data(), y[m], got.size() * 4,
                          hipMemcpyDeviceToHost));
          for (std::size_t i = 0; i < got.size(); ++i) {
            num +=
                (double(got[i]) - want[m][i]) * (double(got[i]) - want[m][i]);
            den += double(want[m][i]) * want[m][i];
          }
        }
        const double rel = std::sqrt(num / den);
        if (!(rel < 1e-5)) {
          std::printf("\nFAIL: shape %ux%u R=%u rel %.3g vs production\n",
                      sh.tiles_per_wave, sh.waves, rows, rel);
          return 1;
        }
        const float us = TimeIt(
            [&](int i) {
              LaunchGemvShapeRows(lin[std::size_t(i) % copies], y, rows, ws,
                                  sh);
            },
            reps);
        std::printf("  %ux%u %6.1f us %6.1f GB/s", sh.tiles_per_wave, sh.waves,
                    us, group_bytes / (us * 1e3));
      }
      std::printf("\n");
    }
    for (void* p : owned)
      Check(hipFree(p));
  }
  return 0;
}
