// Flash-Next routed decode GEMV (W3): RoutedGemv with tiles fetched four
// ahead (exl3_moe.hip) against the former one-tile-per-iteration kernel
// (copied from git HEAD as RoutedGemvOld): bitwise equality and time, cold
// expert banks (512 experts), 10 routed experts, 1..8 tokens.
#include <cstdio>
#include <cstring>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/exl3_moe.hip"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {
template<int Bits, bool FullPrecision, unsigned MaxRows = 8,
         bool Scaled = false>
__global__ void RoutedGemvOld(const ex::DeviceLinear* gate,
                              const ex::DeviceLinear* up,
                              const Exl3RouteTile* tiles, const unsigned* count,
                              const std::uint16_t* input, float* gate_y,
                              float* up_y, unsigned k, unsigned n,
                              unsigned assignments,
                              const std::int32_t* exponents) {
  if (blockIdx.y >= *count)
    return;
  const auto tile_rows = tiles[blockIdx.y];
  const unsigned projection = blockIdx.z;
  const auto& l = (projection == 0 ? gate : up)[tile_rows.expert];
  const unsigned lane = threadIdx.x;
  const unsigned r0 = (lane % 4) * 2;
  const unsigned nt = blockIdx.x;
  float a[MaxRows]{}, b[MaxRows]{};
  const auto lw = arithmetic::MakeLaneWindow<Bits>(lane);
  const auto* trellis = reinterpret_cast<const std::uint32_t*>(l.trellis);
  const auto* x = reinterpret_cast<const _Float16*>(input) +
                  std::size_t{projection} * assignments * k;
  for (unsigned kt = 0; kt < k / 16; ++kt) {
    Half2 w[4];
    arithmetic::DecodeLaneMul1<Bits>(
        trellis + (std::size_t{kt} * (n / 16) + nt) * (8 * Bits), lw, w);
    for (unsigned r = 0; r < MaxRows; ++r) {
      if (r >= tile_rows.count)
        break;
      const auto* row = x + std::size_t{tile_rows.slots[r]} * k + kt * 16 + r0;
      const Half2 v01{row[0], row[1]};
      const Half2 v89{row[8], row[9]};
      if (FullPrecision || (Scaled && exponents[tile_rows.slots[r]] < 0)) {
        a[r] = DotPairF32(w[0], v01, a[r]);
        a[r] = DotPairF32(w[1], v89, a[r]);
        b[r] = DotPairF32(w[2], v01, b[r]);
        b[r] = DotPairF32(w[3], v89, b[r]);
      } else {
        a[r] = __builtin_amdgcn_fdot2(w[0], v01, a[r], false);
        a[r] = __builtin_amdgcn_fdot2(w[1], v89, a[r], false);
        b[r] = __builtin_amdgcn_fdot2(w[2], v01, b[r], false);
        b[r] = __builtin_amdgcn_fdot2(w[3], v89, b[r], false);
      }
    }
  }
  for (unsigned r = 0; r < MaxRows; ++r) {
    if (r >= tile_rows.count)
      break;
    a[r] += __shfl_xor(a[r], 1, 32);
    b[r] += __shfl_xor(b[r], 1, 32);
    a[r] += __shfl_xor(a[r], 2, 32);
    b[r] += __shfl_xor(b[r], 2, 32);
    if (lane % 4 == 0) {
      auto* y = (projection == 0 ? gate_y : up_y) +
                std::size_t{tile_rows.slots[r]} * n + nt * 16;
      float scale = 1.0F;
      if constexpr (Scaled) {
        const int exponent = exponents[tile_rows.slots[r]];
        if (exponent > 0)
          scale = __builtin_bit_cast(float, unsigned(127 - exponent) << 23);
      }
      y[lane / 4] = a[r] * scale;
      y[lane / 4 + 8] = b[r] * scale;
    }
  }
}

}  // namespace
}  // namespace gufo::models::qwen38_flash_next::rocm

namespace fnr = gufo::models::qwen38_flash_next::rocm;
namespace ex = gufo::quant::exl3;

namespace {
void Ck(hipError_t e) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(e));
    std::exit(1);
  }
}
__global__ void FillU32(std::uint32_t* p, std::size_t n, std::uint32_t seed) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       i < n; i += std::size_t(gridDim.x) * blockDim.x) {
    std::uint32_t x = static_cast<std::uint32_t>(i) * 0x9E3779B9U + seed;
    x ^= x >> 16;
    x *= 0x85EBCA6BU;
    p[i] = x ^ (x >> 13);
  }
}
__global__ void FillH(std::uint16_t* p, std::size_t n) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       i < n; i += std::size_t(gridDim.x) * blockDim.x)
    p[i] =
        __half_as_ushort(__float2half(float(i * 37 % 1000) / 1000.0F - 0.5F));
}
float TimeIt(const auto& fn, int reps) {
  hipEvent_t a, b;
  Ck(hipEventCreate(&a));
  Ck(hipEventCreate(&b));
  for (int i = 0; i < 3; ++i)
    fn(i);
  Ck(hipEventRecord(a));
  for (int i = 0; i < reps; ++i)
    fn(3 + i);
  Ck(hipEventRecord(b));
  Ck(hipEventSynchronize(b));
  float ms = 0;
  Ck(hipEventElapsedTime(&ms, a, b));
  return ms * 1000.0F / float(reps);
}
template<unsigned R>
void Run(unsigned k, unsigned n, unsigned tokens) {
  constexpr unsigned kExperts = 512, kUsed = 10;
  const std::size_t words = std::size_t(k) * n * 5 / 32;
  std::uint32_t* t;
  Ck(hipMalloc(&t, words * 4 * kExperts));
  hipLaunchKernelGGL(FillU32, dim3(4096), dim3(256), 0, 0, t, words * kExperts,
                     3U);
  std::vector<ex::DeviceLinear> bank(kExperts);
  for (unsigned e = 0; e < kExperts; ++e) {
    bank[e].trellis = reinterpret_cast<const std::uint16_t*>(t + words * e);
    bank[e].k = k;
    bank[e].n = n;
    bank[e].rate = ex::BitRate{5, false};
    bank[e].codebook = ex::Codebook::kMul1;
  }
  ex::DeviceLinear* d_bank;
  Ck(hipMalloc(&d_bank, kExperts * sizeof(ex::DeviceLinear)));
  Ck(hipMemcpy(d_bank, bank.data(), kExperts * sizeof(ex::DeviceLinear),
               hipMemcpyHostToDevice));
  const unsigned assignments = tokens * kUsed;
  std::uint16_t* x;
  Ck(hipMalloc(&x, std::size_t(2) * assignments * k * 2));
  hipLaunchKernelGGL(FillH, dim3(256), dim3(256), 0, 0, x,
                     std::size_t(2) * assignments * k);
  float *g0, *u0, *g1, *u1;
  for (float** p : {&g0, &u0, &g1, &u1})
    Ck(hipMalloc(p, std::size_t(assignments) * n * 4));
  // Tile sets per call: experts drawn per token slot, a different set per
  // repetition so weights stay cold; each tile one expert, its rows.
  constexpr unsigned kSets = 64;
  std::vector<fnr::Exl3RouteTile> tiles(kSets * assignments);
  std::uint32_t lcg = 7;
  for (unsigned set = 0; set < kSets; ++set)
    for (unsigned a = 0; a < assignments; ++a) {
      lcg = lcg * 1664525U + 1013904223U;
      auto& tile = tiles[set * assignments + a];
      tile = {};
      tile.expert = (lcg >> 8) % kExperts;
      tile.count = 1;
      tile.slots[0] = a;
    }
  fnr::Exl3RouteTile* d_tiles;
  Ck(hipMalloc(&d_tiles, tiles.size() * sizeof(fnr::Exl3RouteTile)));
  Ck(hipMemcpy(d_tiles, tiles.data(), tiles.size() * sizeof(fnr::Exl3RouteTile),
               hipMemcpyHostToDevice));
  unsigned* d_count;
  Ck(hipMalloc(&d_count, 4));
  Ck(hipMemcpy(d_count, &assignments, 4, hipMemcpyHostToDevice));
  const dim3 grid(n / 16, assignments, 2);
  const auto old = [&](int i) {
    hipLaunchKernelGGL((fnr::RoutedGemvOld<5, false, R, false>), grid, dim3(32),
                       0, nullptr, d_bank, d_bank,
                       d_tiles + (i % kSets) * assignments, d_count, x, g0, u0,
                       k, n, assignments, nullptr);
  };
  const auto cur = [&](int i) {
    // The production kernel packs kRoutedWaves output tiles per block.
    const dim3 packed((n / 16 + fnr::kRoutedWaves - 1) / fnr::kRoutedWaves,
                      assignments, 2);
    hipLaunchKernelGGL((fnr::RoutedGemv<5, false, R, false>), packed,
                       dim3(32 * fnr::kRoutedWaves), 0, nullptr, d_bank, d_bank,
                       d_tiles + (i % kSets) * assignments, d_count, x, g1, u1,
                       k, n, assignments, nullptr);
  };
  old(0);
  cur(0);
  Ck(hipDeviceSynchronize());
  std::vector<float> a(std::size_t(assignments) * n), b(a.size());
  Ck(hipMemcpy(a.data(), g0, a.size() * 4, hipMemcpyDeviceToHost));
  Ck(hipMemcpy(b.data(), g1, b.size() * 4, hipMemcpyDeviceToHost));
  bool same = std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
  Ck(hipMemcpy(a.data(), u0, a.size() * 4, hipMemcpyDeviceToHost));
  Ck(hipMemcpy(b.data(), u1, b.size() * 4, hipMemcpyDeviceToHost));
  same = same && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
  const float t_old = TimeIt(old, 300), t_new = TimeIt(cur, 300);
  const double bytes = 2.0 * assignments * words * 4;
  std::printf(
      "routed %4u->%-4u tokens %u: old %6.1f us %5.1f GB/s | ahead4 "
      "%6.1f us %5.1f GB/s %s\n",
      k, n, tokens, t_old, bytes / t_old / 1e3, t_new, bytes / t_new / 1e3,
      same ? "bit-identical" : "DIFFERS");
  for (void* p :
       {static_cast<void*>(t), static_cast<void*>(d_bank),
        static_cast<void*>(x), static_cast<void*>(g0), static_cast<void*>(u0),
        static_cast<void*>(g1), static_cast<void*>(u1),
        static_cast<void*>(d_tiles), static_cast<void*>(d_count)})
    Ck(hipFree(p));
}
}  // namespace

int main() {
  Run<1>(2560, 640, 1);
  Run<1>(640, 2560, 1);
  Run<8>(2560, 640, 4);
  Run<8>(640, 2560, 4);
  return 0;
}
