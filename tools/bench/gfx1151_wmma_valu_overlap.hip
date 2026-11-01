// Does a VALU wave overlap a WMMA wave on the same gfx1151 SIMD?
//
// One block of eight waves; each wave reads its SIMD from HW_ID1 and the
// first two waves to arrive on a SIMD take roles A (WMMA f16 16x16x16 only,
// eight independent accumulators) and B (a trellis-decode-like VALU mix:
// alignbit, mad_u32_u16, sad_u8, pk_fma_f16, eight independent chains).
// Modes: 1 = A alone, 2 = B alone, 3 = A and B together on each SIMD,
// 4 = A's wave runs both loops interleaved. B's trip count is calibrated so
// that A and B alone take about the same time; with no overlap mode 3 then
// takes about twice as long as either alone, with full overlap the same.
//
// Measured (2026-11-01, two runs): A alone 3.07-3.20 ms (0.70 TFLOP/s
// per SIMD, the device peak with one wave), B alone 3.25 ms, together A
// 3.23 ms and B 6.44 ms (= the sum; B issues nothing while A runs), one
// interleaved wave 6.35 ms. No cross-wave WMMA/VALU overlap on a SIMD.
//
// Build: hipcc -O3 --offload-arch=gfx1151 (see tools/bench/build.sh).
// Run: ./gfx1151_wmma_valu_overlap [wmma_iters]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_OK(expr)                                                    \
  do {                                                                  \
    const hipError_t err_ = (expr);                                     \
    if (err_ != hipSuccess) {                                           \
      std::fprintf(stderr, "%s:%d hip error: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(err_));                            \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

using f16x16 = __attribute__((__vector_size__(32))) _Float16;
using f32x8 = __attribute__((__vector_size__(32))) float;
using f16x2 = __attribute__((__vector_size__(4))) _Float16;

namespace {

constexpr int kWaves = 8;

struct Record {
  unsigned long long cycles;
  unsigned hw_id, role, ran;
};

__device__ __forceinline__ void WmmaBody(f32x8 (&c)[8], f16x16 a, f16x16 b) {
#pragma unroll
  for (int i = 0; i < 8; ++i)
    c[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c[i]);
}

__device__ __forceinline__ void ValuBody(std::uint32_t (&s)[8], f16x2 (&h)[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    std::uint32_t v = __builtin_amdgcn_alignbit(s[i], s[(i + 1) & 7], 13U);
    v = (v & 0xffffU) * (s[i] >> 16) + s[(i + 3) & 7];  // v_mad_u32_u16
    v = __builtin_amdgcn_sad_u8(v, s[(i + 5) & 7], v >> 7);
    s[i] = v;
    h[i] = __builtin_elementwise_fma(h[i], h[(i + 2) & 7], h[(i + 4) & 7]);
  }
}

template<int Mode>
__global__ __launch_bounds__(kWaves * 32) void Overlap(unsigned wmma_iters,
                                                       unsigned valu_iters,
                                                       Record* records,
                                                       float* sink) {
  __shared__ unsigned arrivals[8];
  if (threadIdx.x < 8)
    arrivals[threadIdx.x] = 0;
  __syncthreads();
  const unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32;
  // gfx11 HW_REG_HW_ID1 (23): SIMD_ID in bits 9:8.
  const unsigned hw = __builtin_amdgcn_s_getreg(23 | (31 << 11));
  const unsigned simd = (hw >> 8) & 3;
  unsigned role = 0;
  if (lane == 0)
    role = atomicAdd(&arrivals[simd], 1U);
  role = __shfl(role, 0, 32);
  __syncthreads();
  const bool run_a = role == 0 && (Mode == 1 || Mode == 3 || Mode == 4);
  const bool run_b = role == 1 && (Mode == 2 || Mode == 3);

  f16x16 a, b;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    a[i] = static_cast<_Float16>((lane + i) % 7) * _Float16(0.125);
    b[i] = static_cast<_Float16>((lane * 3 + i) % 5) * _Float16(0.0625);
  }
  f32x8 c[8] = {};
  std::uint32_t s[8];
  f16x2 h[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    s[i] = 0x9e3779b9U * (lane + 1) + i * 0x85ebca6bU;
    h[i] = f16x2{_Float16(0.5F + i * 0.01F), _Float16(0.25F)};
  }
  const unsigned long long t0 = wall_clock64();
  if (run_a && Mode == 4) {
    for (unsigned it = 0; it < wmma_iters; ++it) {
      WmmaBody(c, a, b);
      // The same VALU work per WMMA as the calibrated B wave.
      for (unsigned v = 0; v < valu_iters / wmma_iters; ++v)
        ValuBody(s, h);
    }
  } else if (run_a) {
    for (unsigned it = 0; it < wmma_iters; ++it)
      WmmaBody(c, a, b);
  } else if (run_b) {
    for (unsigned it = 0; it < valu_iters; ++it)
      ValuBody(s, h);
  }
  const unsigned long long t1 = wall_clock64();
  float acc = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc += c[i][0] + c[i][7] + static_cast<float>(s[i] & 0xff) +
           static_cast<float>(h[i][0]);
  }
  sink[threadIdx.x] = acc;
  if (lane == 0)
    records[wave] = {t1 - t0, hw, role,
                     unsigned(run_a) | (unsigned(run_b) << 1)};
}

template<int Mode>
std::vector<Record> Run(unsigned wi, unsigned vi, Record* d_rec, float* sink,
                        float* ms) {
  hipEvent_t e0, e1;
  HIP_OK(hipEventCreate(&e0));
  HIP_OK(hipEventCreate(&e1));
  Overlap<Mode><<<1, kWaves * 32>>>(wi / 16, vi / 16, d_rec, sink);
  HIP_OK(hipEventRecord(e0));
  Overlap<Mode><<<1, kWaves * 32>>>(wi, vi, d_rec, sink);
  HIP_OK(hipEventRecord(e1));
  HIP_OK(hipEventSynchronize(e1));
  HIP_OK(hipEventElapsedTime(ms, e0, e1));
  std::vector<Record> r(kWaves);
  HIP_OK(hipMemcpy(r.data(), d_rec, sizeof(Record) * kWaves,
                   hipMemcpyDeviceToHost));
  return r;
}

// Longest running wave of the role(s) in `mask` (1 = A, 2 = B), in us
// (wall_clock64 runs at 100 MHz).
double Longest(const std::vector<Record>& r, unsigned mask) {
  unsigned long long m = 0;
  for (const auto& x : r)
    if (x.ran & mask)
      m = std::max(m, x.cycles);
  return double(m) / 100.0;
}

}  // namespace

int main(int argc, char** argv) {
  const unsigned wi = argc > 1 ? unsigned(std::atoi(argv[1])) : 1U << 15;
  Record* d_rec;
  float* sink;
  HIP_OK(hipMalloc(&d_rec, sizeof(Record) * kWaves));
  HIP_OK(hipMalloc(&sink, sizeof(float) * kWaves * 32));
  float ms = 0;
  // Calibrate B to A's time alone.
  const auto a1 = Run<1>(wi, wi, d_rec, sink, &ms);
  const double ta = Longest(a1, 1);
  const auto b0 = Run<2>(wi, wi, d_rec, sink, &ms);
  const double tb0 = Longest(b0, 2);
  // An integer number of VALU bodies per WMMA body, so mode 4 interleaves
  // exactly B's work.
  const unsigned ratio = std::max(1U, unsigned(ta / tb0 + 0.5));
  const unsigned vi = wi * ratio;
  std::printf("placement (wave: SIMD, role):");
  for (unsigned w = 0; w < kWaves; ++w)
    std::printf(" %u:%u,%u", w, (a1[w].hw_id >> 8) & 3, a1[w].role);
  std::printf("\n");
  const auto b2 = Run<2>(wi, vi, d_rec, sink, &ms);
  const double tb = Longest(b2, 2);
  const auto r3 = Run<3>(wi, vi, d_rec, sink, &ms);
  const double t3a = Longest(r3, 1), t3b = Longest(r3, 2);
  const auto r4 = Run<4>(wi, vi, d_rec, sink, &ms);
  const double t4 = Longest(r4, 1);
  const double wmma_tf = 8.0 * 4096 * 2;  // flops per WmmaBody per wave
  std::printf(
      "WMMA iters %u, VALU iters %u (%.2f VALU bodies per WMMA "
      "body)\n",
      wi, vi, double(vi) / wi);
  std::printf("(1) A alone         %9.1f us  (%.2f TFLOP/s per SIMD)\n", ta,
              wmma_tf * wi / (ta * 1e-6) / 1e12);
  std::printf("(2) B alone         %9.1f us\n", tb);
  std::printf("(3) A+B, 2 waves    A %9.1f us  B %9.1f us\n", t3a, t3b);
  std::printf("(4) A+B, one wave   %9.1f us\n", t4);
  const double sum = ta + tb, both = std::max(t3a, t3b);
  std::printf(
      "(3) vs (1)+(2): %.1f vs %.1f us -> overlap hides %.0f %% of "
      "the shorter part\n",
      both, sum, 100.0 * (sum - both) / std::min(ta, tb));
  std::printf("(4) vs (1)+(2): %.1f vs %.1f us\n", t4, sum);
  HIP_OK(hipFree(d_rec));
  HIP_OK(hipFree(sink));
  return 0;
}
