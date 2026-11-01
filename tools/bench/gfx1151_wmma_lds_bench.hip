// Achievable WMMA f16 rate on gfx1151 when the operands come from LDS.
//
// Each wave holds RA A and RB B fragments (16 x 16 halves; a lane reads the
// 32 bytes of its row, lanes 16-31 the same rows as 0-15, as the X3 GEMM's
// LoadSwizzled does) and per k step reads RA + RB fragments from LDS and
// issues RA * RB WMMAs: (RA + RB) / (RA * RB) fragment reads per WMMA. The
// X3 bucket 8x1 tile is RA = 1, RB = 4 (1.25); 2x2 is 4 x 2 (0.75). Mode
// "reg" keeps the fragments in registers (the issue ceiling). The whole GPU
// runs: `blocks` blocks of `waves` waves each.
//
// Measured (2026-11-01): 2.0 reads/WMMA 28-30 TFLOP/s, 1.5 36-38, 1.25
// 39-44, 1.0 46-48, 0.75 46-49, 0.5 47-50, registers 48-54: about 128 B/clk
// of LDS reads per WGP bounds a WMMA core fed at low fragment reuse.
//
// Run: ./gfx1151_wmma_lds_bench [iters]
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define HIP_OK(expr)                                                    \
  do {                                                                  \
    const hipError_t err_ = (expr);                                     \
    if (err_ != hipSuccess) {                                           \
      std::fprintf(stderr, "%s:%d hip error: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(err_));                            \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

namespace {

using v16h = __attribute__((__vector_size__(32))) _Float16;
using v8f = __attribute__((__vector_size__(32))) float;

// 4 k steps x 16 fragments x 16 rows x 16 halves = 32 KB.
constexpr int kSteps = 4, kFrags = 16;

__device__ __forceinline__ v16h Load(const short* p) {
  union {
    v16h f;
    uint4 q[2];
  } bits;
  const unsigned row = threadIdx.x & 15, shift = (row ^ (row >> 2)) & 1;
  bits.q[0] = reinterpret_cast<const uint4*>(p)[shift];
  bits.q[1] = reinterpret_cast<const uint4*>(p)[shift ^ 1];
  return bits.f;
}

template<int RA, int RB, bool Lds>
__global__ __launch_bounds__(256) void Kernel(int iters, float* sink) {
  __shared__ short s[kSteps][kFrags][16][16];
  for (int i = threadIdx.x; i < kSteps * kFrags * 256; i += blockDim.x)
    (&s[0][0][0][0])[i] = static_cast<short>((i * 2654435761U) >> 20 & 0x3bff);
  __syncthreads();
  const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
  const int sl = lane & 15;
  v8f acc[RA][RB] = {};
  v16h a[RA], b[RB];
#pragma unroll
  for (int i = 0; i < RA; ++i)
    a[i] = Load(&s[0][(wave + i) % kFrags][sl][0]);
#pragma unroll
  for (int j = 0; j < RB; ++j)
    b[j] = Load(&s[0][(wave + 8 + j) % kFrags][sl][0]);
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int ks = 0; ks < kSteps; ++ks) {
      if constexpr (Lds) {
        // Re-read every k step, as the GEMM does (no hoisting).
        asm volatile("" ::: "memory");
#pragma unroll
        for (int i = 0; i < RA; ++i)
          a[i] = Load(&s[ks][(wave + i) % kFrags][sl ^ (ks & 3)][0]);
#pragma unroll
        for (int j = 0; j < RB; ++j)
          b[j] = Load(&s[ks][(wave + 8 + j) % kFrags][sl ^ (ks & 3)][0]);
      }
#pragma unroll
      for (int i = 0; i < RA; ++i)
#pragma unroll
        for (int j = 0; j < RB; ++j)
          acc[i][j] =
              __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b[j], a[i], acc[i][j]);
    }
  }
  float t = 0;
#pragma unroll
  for (int i = 0; i < RA; ++i)
#pragma unroll
    for (int j = 0; j < RB; ++j)
      t += acc[i][j][0] + acc[i][j][7];
  sink[blockIdx.x * blockDim.x + threadIdx.x] = t;
}

template<int RA, int RB, bool Lds>
void Run(int iters, int blocks, int waves, float* sink) {
  hipEvent_t e0, e1;
  HIP_OK(hipEventCreate(&e0));
  HIP_OK(hipEventCreate(&e1));
  Kernel<RA, RB, Lds><<<blocks, waves * 32>>>(iters / 8, sink);
  HIP_OK(hipEventRecord(e0));
  Kernel<RA, RB, Lds><<<blocks, waves * 32>>>(iters, sink);
  HIP_OK(hipEventRecord(e1));
  HIP_OK(hipEventSynchronize(e1));
  float ms = 0;
  HIP_OK(hipEventElapsedTime(&ms, e0, e1));
  const double flops =
      double(blocks) * waves * iters * kSteps * RA * RB * 16.0 * 16 * 16 * 2;
  std::printf(
      "%-4s RA=%d RB=%d (%.2f reads/WMMA) %2d waves x %4d blocks: "
      "%5.1f TFLOP/s\n",
      Lds ? "lds" : "reg", RA, RB, Lds ? double(RA + RB) / (RA * RB) : 0.0,
      waves, blocks, flops / (ms * 1e-3) / 1e12);
}

}  // namespace

int main(int argc, char** argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 2048;
  float* sink;
  HIP_OK(hipMalloc(&sink, sizeof(float) * 4096 * 256));
  for (int waves : {4, 8}) {
    const int blocks = 40 * 2 * 8 / waves * 4;  // several blocks per CU
    Run<1, 4, false>(iters, blocks, waves, sink);
    Run<1, 1, true>(iters, blocks, waves, sink);
    Run<1, 2, true>(iters, blocks, waves, sink);
    Run<1, 4, true>(iters, blocks, waves, sink);
    Run<2, 2, true>(iters, blocks, waves, sink);
    Run<2, 4, true>(iters, blocks, waves, sink);
    Run<4, 2, true>(iters, blocks, waves, sink);
    Run<4, 4, true>(iters, blocks, waves, sink);
  }
  HIP_OK(hipFree(sink));
  return 0;
}
