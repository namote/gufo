// X3 step 1 (plan/exl3/archive/X3-DESIGN.md): standalone prefill GEMM
// experiment on gfx1151. The trellis-decode weight producer in the frame of the
// production FP16 prefill GEMM (src/models/qwen/hip/kernels/prefill_fp16.hip,
// HalfPrefillGemmKernel<256, 256, 8, 4>: BM = 256 weight rows (output
// channels) x BN = 256 tokens, 32 waves, WMMA f32_16x16x16_f16, FP32
// accumulation in K16 order, swizzled LDS tiles s_a (weights) and s_b
// (tokens)) against the real GGUF prefill route linked from its sources.
//
// One 16 x 16 trellis tile (k x n) is one weight sub-tile of the frame (16
// channels x 16 k). Each lane decodes its eight positions with the X2b lane
// window (src/core/hip/detail/exl3_gemv.hpp) and stores four FP16 pairs into
// the swizzled weight tile; the WMMA loop is the frame's.
//
// Producers (all BM 256 x BN 256, 32 waves):
//   fp16   predecoded FP16 weights [n][k] (the frame without any decode)
//   v1     trellis, hash x * C as one v_mul_lo_u32 (quarter rate), K64 stage
//   v2     trellis, hash as v_pk_mul_lo_u16 + v_mad_u32_u16, K64 stage
//   v3     as v2 with a K32 stage (BK 2)
//   v2+out v2 with the output rotation y = svh * H(acc) in the dead GEMM LDS
// and the full EXL3 pipeline: FP32 -> FP16 rotated input (suh, row-major),
// v2+out. The GGUF baseline is the production route of prefill_chunk.cpp:
// tokens < 1024 quantize the FP32 activation to Q8_1 and run the W-quant x A8
// WMMA GEMM; tokens >= 1024 convert to FP16 and run HalfPrefillGemmKernel.
//
// The pk hash follows upstream exllamav3 v1.6.5
// exllamav3_ext/rocm/quant/exl3_gemv_tiles_rdna.cuh (MIT License, Copyright
// (c) 2025 Turboderp); the rest is written anew. Weights are random (trellis
// words, GGUF blocks with finite scales) and rotated over copies well beyond
// the 32 MiB MALL.
//
// Modes: check | time [reps]
//   check  the pk hash over all 65,536 states (with high-half garbage), every
//          producer against an FP64 reference, producer bit identity, the
//          output rotation and the input rotation against FP64.
//   time   projection microbenchmarks (one projection each; production fuses
//          RMSNorm into the producer, gate/up/SwiGLU and down/residual):
//          gate/up, down, k/v at 512 and 2048 tokens, against the GGUF
//          production route for the dominant UD-Q4_K_XL types of the
//          projection; then one FFN chain (RMSNorm -> gate/up -> SwiGLU -> down
//          + residual) per method. Producer and GEMM run inside one event pair;
//          weights are rotated over copies beyond the MALL in whole cycles, and
//          the bytes traversed are reported.
// Build: plan/exl3/archive/X3-RESULTS.md.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/detail/exl3_gemv.hpp"
#include "src/core/quant/exl3/format.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"

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

namespace exl3q = gufo::quant::exl3;
namespace dexl3 = gufo::hip::exl3;
using gufo::core::GgmlType;

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

// --- frame helpers, as in prefill_fp16.hip -------------------------------

__device__ __forceinline__ v8f WmmaHalf(v16h a, v16h b, v8f c) {
#if __clang_major__ >= 24
  asm("v_wmma_f32_16x16x16_f16 %0, %1, %2, %0" : "+v"(c) : "v"(a), "v"(b));
  return c;
#else
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#endif
}

__device__ __forceinline__ v16h LoadSwizzled(const short* p) {
  union {
    v16h f;
    uint4 q[2];
  } bits;
  const unsigned row = threadIdx.x & 15, shift = (row ^ (row >> 2)) & 1;
  bits.q[0] = reinterpret_cast<const uint4*>(p)[shift];
  bits.q[1] = reinterpret_cast<const uint4*>(p)[shift ^ 1];
  return bits.f;
}

__device__ __forceinline__ void StoreSwizzled(short* p, uint4 lo, uint4 hi,
                                              unsigned row) {
  const unsigned shift = (row ^ (row >> 2)) & 1;
  reinterpret_cast<uint4*>(p)[shift] = lo;
  reinterpret_cast<uint4*>(p)[shift ^ 1] = hi;
}

// --- trellis decode ---------------------------------------------------------

constexpr std::uint32_t kMul1 = 0x83DCD12DU;

// x * kMul1 mod 2^32 for the 16-bit x in the low half of src, two full-rate
// ops; the upper half of src is ignored by both.
__device__ __forceinline__ std::uint32_t HashPk(std::uint32_t src) {
  constexpr std::uint32_t k_hi = kMul1 & 0xffff0000U;
  constexpr std::uint32_t c_lo = kMul1 & 0xffffU;
  std::uint32_t r;
  std::uint32_t p;
  asm("v_pk_mul_lo_u16 %0, %1, %2 op_sel_hi:[0,1]"
      : "=v"(r)
      : "v"(src), "s"(k_hi));
  asm("v_mad_u32_u16 %0, %1, %2, %3" : "=v"(p) : "v"(src), "s"(c_lo), "v"(r));
  return p;
}

__global__ void HashCheckKernel(std::uint32_t* out_pk, std::uint32_t* out_mul) {
  const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= 65536)
    return;
  // Arbitrary upper half, as the window shift leaves it.
  const std::uint32_t src = x | ((x * 2654435761U) & 0xffff0000U);
  out_pk[x] = HashPk(src);
  out_mul[x] = (src & 0xffffU) * kMul1;
}

struct RawTile {
  std::uint32_t d0, d1, d2;
};

template<int K, bool kPk>
__device__ __forceinline__ void DecodeRaw(const RawTile& raw,
                                          const dexl3::LaneWindow& lw,
                                          dexl3::Half2 (&w)[4]) {
  const std::uint32_t hi = __builtin_amdgcn_alignbit(raw.d0, raw.d1, lw.shift);
  const std::uint32_t lo = __builtin_amdgcn_alignbit(raw.d1, raw.d2, lw.shift);
  const std::uint64_t window = (static_cast<std::uint64_t>(hi) << 32) | lo;
  std::uint32_t p[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const auto state = static_cast<std::uint32_t>(window >> (48 - K * j));
    p[j] = kPk ? HashPk(state) : (state & 0xffffU) * kMul1;
  }
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    w[q] = dexl3::DecodePairMul1(p[2 * q], p[2 * q + 1]);
  }
}

// --- GEMM -----------------------------------------------------------------

enum class Producer { kFp16, kTrellisMul, kTrellisPk };

constexpr int kBM = 256, kBN = 256, kWM = 8, kWN = 4;
constexpr int kThreads = kWM * kWN * 32;

// y[t][n] = sum_k x[t][k] W[k][n] (Rotate: then svh * H per 128 channels).
// Weight rows are output channels n. Complete tiles only.
template<int BK, int K, Producer P, bool Rotate>
__launch_bounds__(kThreads) __global__
    void PrefillGemmKernel(const void* __restrict__ weights,
                           const short* __restrict__ x,
                           const _Float16* __restrict__ svh,
                           float* __restrict__ y, int batch, int m, int k) {
  constexpr int WaveSize = 32;
  constexpr int RS = kBM / 16;
  constexpr int TS = kBN / 16;
  constexpr int WRS = RS / kWM;
  constexpr int WTS = TS / kWN;
  constexpr int KS = 16;
  constexpr int kTiles = BK * RS;
  constexpr int kWaves = kWM * kWN;
  constexpr int TPW = kTiles / kWaves;  // trellis tiles per wave and stage
  constexpr int PFA = (kBM * BK + kThreads - 1) / kThreads;  // fp16 rows
  constexpr int PFB = (kBN * BK + kThreads - 1) / kThreads;
  constexpr unsigned kTileDwords = 8 * K;
  static_assert(kTiles % kWaves == 0);

  // The GEMM tiles, reused as the output-rotation staging buffer.
  struct GemmTiles {
    short a[BK][RS][16][KS];
    short b[BK][TS][16][KS];
  };
  constexpr int kStageTokens =
      static_cast<int>(sizeof(GemmTiles) / sizeof(float)) / kBM;
  static_assert(!Rotate || kStageTokens == kBN / kWN);
  __shared__ union {
    GemmTiles g;
    float c[kStageTokens][kBM];
  } smem;
  auto& s_a = smem.g.a;
  auto& s_b = smem.g.b;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / WaveSize;
  const int lane = tid % WaveSize;
  const int sl = lane & 15;
  const int wr = wave / kWN;
  const int wt = wave % kWN;
  // Production block order (prefill_fp16.hip HalfPrefillGemmKernel): groups
  // of eight weight-row tiles walk the token tiles together.
  unsigned row_tile = blockIdx.y, token_tile = blockIdx.x;
  {
    constexpr unsigned group_shift = 3;
    constexpr unsigned group = 1U << group_shift;
    const unsigned first = (blockIdx.y >> group_shift) << group_shift,
                   within = (blockIdx.y & (group - 1)) * gridDim.x + blockIdx.x,
                   rows = min(group, gridDim.y - first);
    row_tile = first + within % rows;
    token_tile = within / rows;
  }
  const int r_block = row_tile * kBM, t_block = token_tile * kBN;
  const int ksteps = k / 16;
  const int n_tiles = m / 16;

  v8f acc[WRS][WTS] = {};

  const dexl3::LaneWindow lw = dexl3::MakeLaneWindow<K>(lane);
  const unsigned r0 = (lane % 4) * 2;
  const unsigned c0 = lane / 4;
  const auto* base = static_cast<const std::uint32_t*>(weights);

  const short* x_row[PFB];
#pragma unroll
  for (int p = 0; p < PFB; ++p) {
    const int t = t_block + (((p * kThreads) + tid) / BK);
    x_row[p] = x + (static_cast<std::size_t>(t) * k);
  }
  const short* w_row[PFA];
#pragma unroll
  for (int p = 0; p < PFA; ++p) {
    const int r = r_block + (((p * kThreads) + tid) / BK);
    w_row[p] =
        static_cast<const short*>(weights) + (static_cast<std::size_t>(r) * k);
  }

  RawTile raw[TPW];
  uint4 ra[PFA][2];
  uint4 rb[PFB][2];
  const auto fetch = [&](int ks0) {
    if constexpr (P == Producer::kFp16) {
#pragma unroll
      for (int p = 0; p < PFA; ++p) {
        const int idx = p * kThreads + tid, kk = ks0 + idx % BK;
        if (idx >= kBM * BK)
          continue;
        const short* src = w_row[p] + static_cast<std::size_t>(kk) * 16;
        ra[p][0] = *reinterpret_cast<const uint4*>(src);
        ra[p][1] = *reinterpret_cast<const uint4*>(src + 8);
      }
    } else {
#pragma unroll
      for (int i = 0; i < TPW; ++i) {
        const int t = wave * TPW + i;
        const int ks = t / RS, rt = t % RS;
        const std::uint32_t* tile =
            base + (static_cast<std::size_t>(ks0 + ks) * n_tiles +
                    (r_block / 16) + rt) *
                       kTileDwords;
        raw[i] = {tile[lw.o0], tile[lw.o1], tile[lw.o2]};
      }
    }
#pragma unroll
    for (int p = 0; p < PFB; ++p) {
      const int idx = p * kThreads + tid, kk = ks0 + idx % BK;
      if (idx < kBN * BK) {
        const short* src = x_row[p] + static_cast<std::size_t>(kk) * 16;
        rb[p][0] = *reinterpret_cast<const uint4*>(src);
        rb[p][1] = *reinterpret_cast<const uint4*>(src + 8);
      }
    }
  };
  const auto commit = [&]() {
    if constexpr (P == Producer::kFp16) {
#pragma unroll
      for (int p = 0; p < PFA; ++p) {
        const int idx = p * kThreads + tid, row = idx / BK, ks = idx % BK;
        if (idx >= kBM * BK)
          continue;
        const int placed = (row ^ (ks & 3));
        StoreSwizzled(&s_a[ks][placed / 16][placed % 16][0], ra[p][0], ra[p][1],
                      row);
      }
    } else {
#pragma unroll
      for (int i = 0; i < TPW; ++i) {
        const int t = wave * TPW + i;
        const int ks = t / RS, rt = t % RS;
        dexl3::Half2 w[4];
        DecodeRaw<K, P == Producer::kTrellisPk>(raw[i], lw, w);
        // Column c of the trellis tile is channel rt * 16 + c; rows r0,
        // r0 + 1 (16-byte group 0) and r0 + 8, r0 + 9 (group 1) are k.
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const unsigned row = rt * 16 + c0 + 8 * half;
          const unsigned placed = row ^ (ks & 3);
          const unsigned shift = (row ^ (row >> 2)) & 1;
          auto* pairs = reinterpret_cast<dexl3::Half2*>(
              &s_a[ks][placed / 16][placed % 16][0]);
          pairs[((0 ^ shift) * 8 + r0) / 2] = w[2 * half];
          pairs[((1 ^ shift) * 8 + r0) / 2] = w[2 * half + 1];
        }
      }
    }
#pragma unroll
    for (int p = 0; p < PFB; ++p) {
      const int idx = p * kThreads + tid, row = idx / BK, ks = idx % BK;
      if (idx < kBN * BK) {
        const int placed = (row ^ (ks & 3));
        StoreSwizzled(&s_b[ks][placed / 16][placed % 16][0], rb[p][0], rb[p][1],
                      row);
      }
    }
  };
  fetch(0);
  for (int ks0 = 0; ks0 < ksteps; ks0 += BK) {
    commit();
    __syncthreads();
    if (ks0 + BK < ksteps)
      fetch(ks0 + BK);
#pragma unroll
    for (int ks = 0; ks < BK; ++ks) {
      v16h a[WRS];
#pragma unroll
      for (int i = 0; i < WRS; ++i) {
        a[i] = LoadSwizzled(&s_a[ks][(wr * WRS) + i][sl ^ (ks & 3)][0]);
      }
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const v16h b = LoadSwizzled(&s_b[ks][(wt * WTS) + j][sl ^ (ks & 3)][0]);
#pragma unroll
        for (int i = 0; i < WRS; ++i) {
          acc[i][j] = WmmaHalf(b, a[i], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  const int hi = lane / 16;
  if constexpr (!Rotate) {
#pragma unroll
    for (int i = 0; i < WRS; ++i) {
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const int row = r_block + ((wr * WRS + i) * 16) + sl;
        const int t0 = t_block + ((wt * WTS + j) * 16);
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          y[std::size_t(t0 + 2 * l + hi) * m + row] = acc[i][j][l];
        }
      }
    }
  } else {
    // The block owns two whole 128-channel segments for its 256 tokens.
    // Stage one wave column of tokens (64) at a time in the dead GEMM LDS,
    // then one wave per (token, segment): butterflies, scale, svh.
    for (int pass = 0; pass < kWN; ++pass) {
      if (wt == pass) {
#pragma unroll
        for (int i = 0; i < WRS; ++i) {
#pragma unroll
          for (int j = 0; j < WTS; ++j) {
            const int row = (wr * WRS + i) * 16 + sl;
#pragma unroll
            for (int l = 0; l < 8; ++l) {
              smem.c[j * 16 + 2 * l + hi][row] = acc[i][j][l];
            }
          }
        }
      }
      __syncthreads();
      for (int item = wave; item < kStageTokens * 2; item += kWaves) {
        const int t = item / 2, seg = item % 2;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
          v[j] = smem.c[t][seg * 128 + lane * 4 + j];
        dexl3::Butterfly128(v, static_cast<unsigned>(lane));
        const int row = r_block + seg * 128 + lane * 4;
        float* dst =
            y + std::size_t(t_block + pass * kStageTokens + t) * m + row;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          dst[j] =
              (v[j] * dexl3::kHadamardScale) * static_cast<float>(svh[row + j]);
        }
      }
      __syncthreads();
    }
  }
}

// out[t][i] = FP16(H(suh * x[t])) per 128-element segment, row-major (the
// X2b input-rotation arithmetic). One wave per (token, segment).
__global__ __launch_bounds__(256) void RotateInputRowsKernel(
    const float* __restrict__ x, const _Float16* __restrict__ suh,
    _Float16* __restrict__ out, int tokens, int k) {
  const int lane = threadIdx.x % 32;
  const int item = blockIdx.x * 8 + threadIdx.x / 32;
  const int segments = k / 128;
  if (item >= tokens * segments)
    return;
  const int t = item / segments, seg = item % segments;
  const std::size_t first = std::size_t(t) * k + seg * 128 + lane * 4;
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] = x[first + j] * static_cast<float>(suh[seg * 128 + lane * 4 + j]);
    asm volatile("" : "+v"(v[j]));
  }
  dexl3::Butterfly128(v, static_cast<unsigned>(lane));
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] *= dexl3::kHadamardScale;
    asm volatile("" : "+v"(v[j]));
    out[first + j] = static_cast<_Float16>(v[j]);
  }
}

// W' FP16 [n][k] from the trellis (one wave per tile, the producer's decode):
// the FP16 control's weights are exactly the trellis weights.
template<int K>
__global__ void DecodeToFp16Kernel(const std::uint16_t* __restrict__ trellis,
                                   int k, int m, _Float16* __restrict__ out) {
  const unsigned lane = threadIdx.x % 32;
  const unsigned tile = blockIdx.x;
  const unsigned n_tiles = m / 16;
  const unsigned kt = tile / n_tiles, nt = tile % n_tiles;
  const dexl3::LaneWindow lw = dexl3::MakeLaneWindow<K>(lane);
  const auto* t = reinterpret_cast<const std::uint32_t*>(trellis) +
                  std::size_t(tile) * 8 * K;
  dexl3::Half2 w[4];
  DecodeRaw<K, true>({t[lw.o0], t[lw.o1], t[lw.o2]}, lw, w);
  const unsigned r0 = (lane % 4) * 2, c0 = lane / 4;
  for (int q = 0; q < 4; ++q) {
    const unsigned n = nt * 16 + c0 + 8 * (q / 2);
    const unsigned kk = kt * 16 + r0 + 8 * (q % 2);
    out[std::size_t(n) * k + kk] = w[q].x;
    out[std::size_t(n) * k + kk + 1] = w[q].y;
  }
}

// Every 16-bit state through the pk hash and the pair decode: lo = state s,
// hi = a permutation of s, so each state is decoded in both halves.
__global__ void DecodeStatesKernel(std::uint32_t* out) {
  const unsigned s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= 65536)
    return;
  const unsigned s2 = (s * 40503U + 12345U) & 0xffffU;
  const dexl3::Half2 v = dexl3::DecodePairMul1(HashPk(s), HashPk(s2));
  out[s] = __builtin_bit_cast(std::uint32_t, v);
}

// The fused FFN producer between gate/up and down (unfused upper bound):
// out[t][i] = FP16(H(suh_d * silu(g) * u)) per 128 elements, g and u the
// rotated FP32 GEMM outputs; SwiGLU as BatchedSwiGLUActivationKernel.
__global__ __launch_bounds__(256) void SwiGLURotateKernel(
    const float* __restrict__ gate, const float* __restrict__ up,
    const _Float16* __restrict__ suh, _Float16* __restrict__ out, int tokens,
    int k) {
  const int lane = threadIdx.x % 32;
  const int item = blockIdx.x * 8 + threadIdx.x / 32;
  const int segments = k / 128;
  if (item >= tokens * segments)
    return;
  const int t = item / segments, seg = item % segments;
  const std::size_t first = std::size_t(t) * k + seg * 128 + lane * 4;
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float g = gate[first + j], u = up[first + j];
    const float sig = 1.0F / (1.0F + expf(-g));
    float val = (g * sig) * u;
    asm volatile("" : "+v"(val));
    v[j] = val * static_cast<float>(suh[seg * 128 + lane * 4 + j]);
    asm volatile("" : "+v"(v[j]));
  }
  dexl3::Butterfly128(v, static_cast<unsigned>(lane));
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] *= dexl3::kHadamardScale;
    asm volatile("" : "+v"(v[j]));
    out[first + j] = static_cast<_Float16>(v[j]);
  }
}

enum Variant { kFp16Ctl, kV1, kV2, kV3, kV2Out, kVariants };
const char* const kVariantName[kVariants] = {"fp16-ctl", "exl3-v1", "exl3-v2",
                                             "exl3-v3", "exl3-v2+out"};

template<int K>
void LaunchVariant(int v, const void* w, const short* x, const _Float16* svh,
                   float* y, int batch, int m, int k, hipStream_t stream) {
  const dim3 grid(batch / kBN, m / kBM), block(kThreads);
  switch (v) {
    case kFp16Ctl:
      hipLaunchKernelGGL((PrefillGemmKernel<4, K, Producer::kFp16, false>),
                         grid, block, 0, stream, w, x, svh, y, batch, m, k);
      break;
    case kV1:
      hipLaunchKernelGGL(
          (PrefillGemmKernel<4, K, Producer::kTrellisMul, false>), grid, block,
          0, stream, w, x, svh, y, batch, m, k);
      break;
    case kV2:
      hipLaunchKernelGGL((PrefillGemmKernel<4, K, Producer::kTrellisPk, false>),
                         grid, block, 0, stream, w, x, svh, y, batch, m, k);
      break;
    case kV3:
      hipLaunchKernelGGL((PrefillGemmKernel<2, K, Producer::kTrellisPk, false>),
                         grid, block, 0, stream, w, x, svh, y, batch, m, k);
      break;
    default:
      hipLaunchKernelGGL((PrefillGemmKernel<4, K, Producer::kTrellisPk, true>),
                         grid, block, 0, stream, w, x, svh, y, batch, m, k);
      break;
  }
}

void Launch(int bits, int v, const void* w, const short* x, const _Float16* svh,
            float* y, int batch, int m, int k, hipStream_t stream) {
  switch (bits) {
    case 4:
      LaunchVariant<4>(v, w, x, svh, y, batch, m, k, stream);
      break;
    case 6:
      LaunchVariant<6>(v, w, x, svh, y, batch, m, k, stream);
      break;
    default:
      LaunchVariant<5>(v, w, x, svh, y, batch, m, k, stream);
      break;
  }
}

// --- data -----------------------------------------------------------------

__device__ unsigned Hash(unsigned value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

__global__ void InitBlocks(unsigned char* data, std::size_t blocks,
                           int block_bytes, int scale, int scale_min,
                           unsigned seed) {
  const std::size_t b =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (b >= blocks)
    return;
  unsigned char* block = data + b * block_bytes;
  const unsigned h = Hash(static_cast<unsigned>(b) ^ seed);
  for (int i = 0; i < block_bytes; ++i)
    block[i] = static_cast<unsigned char>(Hash(h + i));
  if (scale >= 0)
    *reinterpret_cast<__half*>(block + scale) =
        __float2half(0.002F + (h % 1000) * 0.000002F);
  if (scale_min >= 0)
    *reinterpret_cast<__half*>(block + scale_min) =
        __float2half(0.001F + (Hash(h) % 1000) * 0.000001F);
}

__global__ void RandomWords(std::uint16_t* x, std::size_t n, unsigned seed) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < n)
    x[i] = static_cast<std::uint16_t>(Hash(static_cast<unsigned>(i) ^ seed));
}

__global__ void RandomHalves(_Float16* x, std::size_t n, float lo, float hi,
                             unsigned seed) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < n)
    x[i] = static_cast<_Float16>(
        lo +
        (hi - lo) * (Hash(static_cast<unsigned>(i) ^ seed) % 10001) * 1e-4F);
}

__global__ void RandomFloats(float* x, std::size_t n, float lo, float hi,
                             unsigned seed) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < n)
    x[i] = lo +
           (hi - lo) * (Hash(static_cast<unsigned>(i) ^ seed) % 10001) * 1e-4F;
}

unsigned Blocks(std::size_t n) {
  return static_cast<unsigned>((n + 255) / 256);
}

struct GgufType {
  const char* name;
  GgmlType type;
  int qk, block_bytes, scale, scale_min;
};
constexpr GgufType kQ4K{"Q4_K", GgmlType::kQ4_K, 256, 144, 0, 2};
constexpr GgufType kQ5K{"Q5_K", GgmlType::kQ5_K, 256, 176, 0, 2};
constexpr GgufType kQ6K{"Q6_K", GgmlType::kQ6_K, 256, 210, 208, -1};
constexpr GgufType kIQ4XS{"IQ4_XS", GgmlType::kIQ4_XS, 256, 136, 0, -1};
constexpr GgufType kQ80{"Q8_0", GgmlType::kQ8_0, 32, 34, 0, -1};

// --- check --------------------------------------------------------------

double RelL2(const std::vector<float>& got, const std::vector<double>& want,
             bool* finite) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    *finite = *finite && std::isfinite(got[i]);
    num += (got[i] - want[i]) * (got[i] - want[i]);
    den += want[i] * want[i];
  }
  return std::sqrt(num / den);
}

// The kernel's FP32 input-rotation contract on the CPU, rounded to FP16.
std::vector<std::uint16_t> RotateContract(const float* x,
                                          const std::uint16_t* suh, int k) {
  std::vector<std::uint16_t> out(k);
  for (int s = 0; s < k; s += 128) {
    float seg[128];
    for (int i = 0; i < 128; ++i)
      seg[i] = x[s + i] * static_cast<float>(exl3q::HalfToDouble(suh[s + i]));
    for (int len = 1; len < 128; len <<= 1)
      for (int i = 0; i < 128; i += len << 1)
        for (int j = i; j < i + len; ++j) {
          const float a = seg[j], b = seg[j + len];
          seg[j] = a + b;
          seg[j + len] = a - b;
        }
    for (int i = 0; i < 128; ++i)
      out[s + i] = exl3q::HalfFromDouble(seg[i] * 0.0883883461356163F);
  }
  return out;
}

template<typename T>
T* Upload(const std::vector<T>& h) {
  T* d = nullptr;
  HIP_OK(hipMalloc(&d, h.size() * sizeof(T)));
  HIP_OK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
  return d;
}

template<typename T>
std::vector<T> Download(const T* d, std::size_t n) {
  std::vector<T> h(n);
  HIP_OK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
  return h;
}

void DecodeToFp16(int bits, const std::uint16_t* trellis, int k, int m,
                  _Float16* out) {
  const dim3 grid((k / 16) * (m / 16));
  switch (bits) {
    case 4:
      hipLaunchKernelGGL(DecodeToFp16Kernel<4>, grid, dim3(32), 0, 0, trellis,
                         k, m, out);
      break;
    case 6:
      hipLaunchKernelGGL(DecodeToFp16Kernel<6>, grid, dim3(32), 0, 0, trellis,
                         k, m, out);
      break;
    default:
      hipLaunchKernelGGL(DecodeToFp16Kernel<5>, grid, dim3(32), 0, 0, trellis,
                         k, m, out);
      break;
  }
}

int Check() {
  int failures = 0;
  {
    std::uint32_t *d_pk = nullptr, *d_mul = nullptr;
    HIP_OK(hipMalloc(&d_pk, 65536 * 4));
    HIP_OK(hipMalloc(&d_mul, 65536 * 4));
    hipLaunchKernelGGL(HashCheckKernel, dim3(256), dim3(256), 0, 0, d_pk,
                       d_mul);
    const auto pk = Download(d_pk, 65536), mul = Download(d_mul, 65536);
    std::size_t diff = 0;
    for (std::uint32_t x = 0; x < 65536; ++x)
      diff += pk[x] != mul[x] || mul[x] != x * kMul1;
    hipLaunchKernelGGL(DecodeStatesKernel, dim3(256), dim3(256), 0, 0, d_pk);
    const auto dec = Download(d_pk, 65536);
    std::size_t ddiff = 0;
    for (std::uint32_t s = 0; s < 65536; ++s) {
      const std::uint32_t s2 = (s * 40503U + 12345U) & 0xffffU;
      ddiff +=
          (dec[s] & 0xffffU) != exl3q::DecodeState(s, exl3q::Codebook::kMul1) ||
          (dec[s] >> 16) != exl3q::DecodeState(s2, exl3q::Codebook::kMul1);
    }
    std::printf(
        "check pk hash: %zu / 65536 states differ; pair decode vs "
        "DecodeState: %zu / 65536 differ\n",
        diff, ddiff);
    failures += diff != 0 || ddiff != 0;
    HIP_OK(hipFree(d_pk));
    HIP_OK(hipFree(d_mul));
  }
  std::mt19937 rng(20261031);
  std::uniform_real_distribution<float> dist(-2.0F, 2.0F);
  struct Case {
    int bits, m, k, batch;
  };
  // k up to the production 17408 (down).
  const Case cases[] = {{4, 512, 256, 512},
                        {5, 512, 256, 512},
                        {6, 512, 256, 512},
                        {5, 256, 17408, 256}};
  for (const Case& c : cases) {
    const exl3q::BitRate rate{c.bits, false};
    std::vector<std::uint16_t> trellis((c.k / 16) * (c.m / 16) *
                                       rate.TileWords());
    for (auto& w : trellis)
      w = static_cast<std::uint16_t>(rng());
    std::vector<std::uint16_t> xh(static_cast<std::size_t>(c.batch) * c.k);
    for (auto& v : xh)
      v = exl3q::HalfFromDouble(dist(rng));
    std::vector<std::uint16_t> svh(c.m);
    for (auto& v : svh)
      v = exl3q::HalfFromDouble(0.01 + 0.02 * dist(rng));
    std::vector<std::uint16_t> inner(static_cast<std::size_t>(c.k) * c.m);
    exl3q::DequantInner(trellis.data(), c.k, c.m, rate, exl3q::Codebook::kMul1,
                        inner.data());
    std::vector<double> wd(inner.size());
    for (std::size_t i = 0; i < inner.size(); ++i)
      wd[i] = exl3q::HalfToDouble(inner[i]);
    std::vector<double> ref(static_cast<std::size_t>(c.batch) * c.m, 0.0);
    for (int t = 0; t < c.batch; ++t)
      for (int i = 0; i < c.k; ++i) {
        const double xv = exl3q::HalfToDouble(xh[std::size_t(t) * c.k + i]);
        const double* wr = &wd[std::size_t(i) * c.m];
        double* yr = &ref[std::size_t(t) * c.m];
        for (int n = 0; n < c.m; ++n)
          yr[n] += xv * wr[n];
      }
    std::vector<double> rotated = ref;
    for (int t = 0; t < c.batch; ++t) {
      exl3q::HadamardBlocks(
          std::span<double>(&rotated[std::size_t(t) * c.m], c.m));
      for (int n = 0; n < c.m; ++n)
        rotated[std::size_t(t) * c.m + n] *= exl3q::HalfToDouble(svh[n]);
    }
    auto* dw = Upload(trellis);
    auto* dx = Upload(xh);
    auto* dsvh = Upload(svh);
    _Float16* dwt = nullptr;
    HIP_OK(hipMalloc(&dwt, std::size_t(c.m) * c.k * 2));
    DecodeToFp16(c.bits, dw, c.k, c.m, dwt);
    float* dy = nullptr;
    HIP_OK(hipMalloc(&dy, ref.size() * 4));
    std::vector<float> first;
    for (int v = 0; v < kVariants; ++v) {
      HIP_OK(hipMemset(dy, 0, ref.size() * 4));
      Launch(c.bits, v, v == kFp16Ctl ? static_cast<const void*>(dwt) : dw,
             reinterpret_cast<const short*>(dx),
             reinterpret_cast<const _Float16*>(dsvh), dy, c.batch, c.m, c.k,
             nullptr);
      HIP_OK(hipDeviceSynchronize());
      const auto y = Download(dy, ref.size());
      bool finite = true;
      const double rel = RelL2(y, v == kV2Out ? rotated : ref, &finite);
      failures += !(rel < 1e-5) || !finite;
      bool same = true;
      if (v == kFp16Ctl)
        first = y;
      else if (v != kV2Out)
        same = std::memcmp(first.data(), y.data(), y.size() * 4) == 0;
      failures += !same;
      std::printf("check K=%d k=%d %s: rel L2 vs FP64 %.3e%s%s\n", c.bits, c.k,
                  kVariantName[v], rel, finite ? "" : ", NOT FINITE",
                  v == kV2Out ? " (with output rotation)"
                  : same      ? ", bit-identical to fp16-ctl"
                              : ", DIFFERS from fp16-ctl");
    }
    for (void* p : {static_cast<void*>(dw), static_cast<void*>(dwt),
                    static_cast<void*>(dx), static_cast<void*>(dsvh),
                    static_cast<void*>(dy)})
      HIP_OK(hipFree(p));
  }
  {
    // Input rotation against the FP32 contract (FP16 bytes), then the
    // composed rotation -> GEMM + output rotation against FP64.
    const int tokens = 256, k = 1024, m = 512, bits = 5;
    const exl3q::BitRate rate{bits, false};
    std::vector<float> x(static_cast<std::size_t>(tokens) * k);
    for (auto& v : x)
      v = 2.0F * dist(rng);
    std::vector<std::uint16_t> suh(k), svh(m);
    for (auto& v : suh)
      v = exl3q::HalfFromDouble(0.5 + 0.25 * dist(rng));
    for (auto& v : svh)
      v = exl3q::HalfFromDouble(0.01 + 0.02 * dist(rng));
    std::vector<std::uint16_t> trellis((k / 16) * (m / 16) * rate.TileWords());
    for (auto& w : trellis)
      w = static_cast<std::uint16_t>(rng());
    auto* dx = Upload(x);
    auto* dsuh = Upload(suh);
    auto* dsvh = Upload(svh);
    auto* dw = Upload(trellis);
    _Float16* drot = nullptr;
    float* dy = nullptr;
    HIP_OK(hipMalloc(&drot, x.size() * 2));
    HIP_OK(hipMalloc(&dy, std::size_t(tokens) * m * 4));
    hipLaunchKernelGGL(
        RotateInputRowsKernel, dim3(tokens * k / 128 / 8), dim3(256), 0, 0, dx,
        reinterpret_cast<const _Float16*>(dsuh), drot, tokens, k);
    Launch(bits, kV2Out, dw, reinterpret_cast<const short*>(drot),
           reinterpret_cast<const _Float16*>(dsvh), dy, tokens, m, k, nullptr);
    HIP_OK(hipDeviceSynchronize());
    const auto got_rot =
        Download(reinterpret_cast<std::uint16_t*>(drot), x.size());
    std::size_t rot_diff = 0;
    std::vector<std::uint16_t> want_rot(x.size());
    for (int t = 0; t < tokens; ++t) {
      const auto r = RotateContract(&x[std::size_t(t) * k], suh.data(), k);
      std::copy(r.begin(), r.end(), want_rot.begin() + std::size_t(t) * k);
    }
    for (std::size_t i = 0; i < x.size(); ++i)
      rot_diff += got_rot[i] != want_rot[i];
    std::vector<std::uint16_t> inner(static_cast<std::size_t>(k) * m);
    exl3q::DequantInner(trellis.data(), k, m, rate, exl3q::Codebook::kMul1,
                        inner.data());
    std::vector<double> ref(std::size_t(tokens) * m, 0.0);
    for (int t = 0; t < tokens; ++t) {
      double* yr = &ref[std::size_t(t) * m];
      for (int i = 0; i < k; ++i) {
        const double xv = exl3q::HalfToDouble(want_rot[std::size_t(t) * k + i]);
        for (int n = 0; n < m; ++n)
          yr[n] += xv * exl3q::HalfToDouble(inner[std::size_t(i) * m + n]);
      }
      exl3q::HadamardBlocks(std::span<double>(yr, m));
      for (int n = 0; n < m; ++n)
        yr[n] *= exl3q::HalfToDouble(svh[n]);
    }
    const auto y = Download(dy, ref.size());
    bool finite = true;
    const double rel = RelL2(y, ref, &finite);
    failures += rot_diff != 0 || !(rel < 1e-5) || !finite;
    std::printf(
        "check input rotation: %zu / %zu FP16 values differ from the "
        "FP32 contract; composed rotation -> GEMM -> output rotation: "
        "rel L2 vs FP64 %.3e%s\n",
        rot_diff, x.size(), rel, finite ? "" : ", NOT FINITE");
    for (void* p : {static_cast<void*>(dx), static_cast<void*>(dsuh),
                    static_cast<void*>(dsvh), static_cast<void*>(dw),
                    static_cast<void*>(drot), static_cast<void*>(dy)})
      HIP_OK(hipFree(p));
  }
  std::printf("check: %s\n", failures == 0 ? "all passed" : "FAILED");
  return failures == 0 ? 0 : 1;
}

// --- time ---------------------------------------------------------------

// Weights as copies whose total exceeds the 32 MiB MALL; every measurement
// visits whole cycles of copies, and reports the bytes it traversed.
struct Copies {
  unsigned char* data = nullptr;
  std::size_t bytes = 0;  // one copy
  int count = 0;
  const void* At(int i) const { return data + std::size_t(i % count) * bytes; }
};

Copies Alloc(std::size_t bytes) {
  Copies c;
  c.bytes = (bytes + 255) / 256 * 256;
  c.count =
      static_cast<int>(std::max<std::size_t>(2, (256U << 20) / c.bytes + 1));
  HIP_OK(hipMalloc(&c.data, c.bytes * c.count));
  return c;
}

Copies TrellisCopies(int k, int m, int bits, unsigned seed) {
  const exl3q::BitRate rate{bits, false};
  Copies c = Alloc(std::size_t(k / 16) * (m / 16) * rate.TileWords() * 2);
  const std::size_t words = c.bytes * c.count / 2;
  hipLaunchKernelGGL(RandomWords, dim3(Blocks(words)), dim3(256), 0, 0,
                     reinterpret_cast<std::uint16_t*>(c.data), words, seed);
  return c;
}

Copies GgufCopies(const GgufType& t, int k, int m, unsigned seed) {
  Copies c = Alloc(std::size_t(m) * (k / t.qk) * t.block_bytes);
  const std::size_t per_copy = c.bytes / t.block_bytes;  // whole blocks
  for (int i = 0; i < c.count; ++i)
    hipLaunchKernelGGL(InitBlocks, dim3(Blocks(per_copy)), dim3(256), 0, 0,
                       c.data + std::size_t(i) * c.bytes, per_copy,
                       t.block_bytes, t.scale, t.scale_min, seed + i);
  return c;
}

void Free(Copies& c) {
  HIP_OK(hipFree(c.data));
}

class Timer {
public:
  Timer() {
    HIP_OK(hipEventCreate(&e0_));
    HIP_OK(hipEventCreate(&e1_));
  }
  ~Timer() {
    (void)hipEventDestroy(e0_);
    (void)hipEventDestroy(e1_);
  }
  // Median ms of launch(i) over whole cycles of `cycle` copies (at least
  // `reps` launches), after one warm-up cycle.
  template<typename F>
  float Median(F&& launch, int cycle, int reps) {
    const int n = ((std::max(reps, cycle) + cycle - 1) / cycle) * cycle;
    for (int i = 0; i < cycle; ++i)
      launch(i);
    HIP_OK(hipDeviceSynchronize());
    std::vector<float> t;
    for (int i = 0; i < n; ++i) {
      HIP_OK(hipEventRecord(e0_, nullptr));
      launch(i);
      HIP_OK(hipEventRecord(e1_, nullptr));
      HIP_OK(hipEventSynchronize(e1_));
      float ms = 0.0F;
      HIP_OK(hipEventElapsedTime(&ms, e0_, e1_));
      t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    launches_ = n;
    return t[t.size() / 2];
  }
  int launches() const { return launches_; }

private:
  hipEvent_t e0_, e1_;
  int launches_ = 0;
};

constexpr int kTokens[] = {512, 2048};

void Row(const char* scope, const char* shape, int k, int m, int tokens,
         const std::string& method, float ms, double flop,
         double traversed_mb) {
  std::printf("%s\t%s\t%d\t%d\t%d\t%s\t%.3f\t%.2f\t%.0f\n", scope, shape, k, m,
              tokens, method.c_str(), ms, flop / (ms * 1e-3) / 1e12,
              traversed_mb);
}

// Projection microbenchmarks: one projection, EXL3 producers and the GGUF
// production route for the dominant UD-Q4_K_XL types of that projection.
void TimeProjections(Timer& timer, int reps) {
  struct Shape {
    const char* name;
    int k, m;
    std::vector<GgufType> gguf;  // dominant UD-Q4_K_XL types
  };
  const Shape shapes[] = {{"gate_up", 5120, 17408, {kIQ4XS, kQ5K, kQ4K}},
                          {"down", 17408, 5120, {kQ5K, kIQ4XS}},
                          {"k_v", 5120, 1024, {kQ6K, kQ80}}};
  for (const Shape& s : shapes) {
    Copies trellis = TrellisCopies(s.k, s.m, 5, 5U);
    Copies fp16 = Alloc(std::size_t(s.m) * s.k * 2);
    for (int i = 0; i < fp16.count; ++i)
      DecodeToFp16(5, static_cast<const std::uint16_t*>(trellis.At(i)), s.k,
                   s.m,
                   reinterpret_cast<_Float16*>(fp16.data + i * fp16.bytes));
    std::vector<Copies> gguf;
    for (std::size_t g = 0; g < s.gguf.size(); ++g)
      gguf.push_back(GgufCopies(s.gguf[g], s.k, s.m, 11U + 7 * g));
    const int max_tokens = kTokens[std::size(kTokens) - 1];
    const std::size_t xn = std::size_t(max_tokens) * s.k;
    float *x32 = nullptr, *y = nullptr;
    _Float16 *x16 = nullptr, *suh = nullptr, *svh = nullptr;
    void* xq8 = nullptr;
    HIP_OK(hipMalloc(&x32, xn * 4));
    hipLaunchKernelGGL(RandomFloats, dim3(Blocks(xn)), dim3(256), 0, 0, x32, xn,
                       -2.0F, 2.0F, 3U);
    HIP_OK(hipMalloc(&x16, xn * 2));
    hipLaunchKernelGGL(RandomHalves, dim3(Blocks(xn)), dim3(256), 0, 0, x16, xn,
                       -2.0F, 2.0F, 4U);
    HIP_OK(
        hipMalloc(&xq8, gufo::hip::QuantizedActivationBytes(max_tokens, s.k)));
    HIP_OK(hipMalloc(&suh, s.k * 2));
    hipLaunchKernelGGL(RandomHalves, dim3(Blocks(s.k)), dim3(256), 0, 0, suh,
                       std::size_t(s.k), 0.5F, 1.0F, 8U);
    HIP_OK(hipMalloc(&svh, s.m * 2));
    hipLaunchKernelGGL(RandomHalves, dim3(Blocks(s.m)), dim3(256), 0, 0, svh,
                       std::size_t(s.m), 0.005F, 0.02F, 9U);
    HIP_OK(hipMalloc(&y, std::size_t(max_tokens) * s.m * 4));
    HIP_OK(hipDeviceSynchronize());
    for (const int tokens : kTokens) {
      const double flop = 2.0 * tokens * s.m * double(s.k);
      const auto mb = [&](const Copies& c) {
        return double(c.bytes) * timer.launches() / 1e6;
      };
      for (int v = 0; v < kVariants; ++v) {
        const Copies& w = v == kFp16Ctl ? fp16 : trellis;
        const float ms = timer.Median(
            [&](int i) {
              Launch(5, v, w.At(i), reinterpret_cast<const short*>(x16), svh, y,
                     tokens, s.m, s.k, nullptr);
            },
            w.count, reps);
        Row("projection", s.name, s.k, s.m, tokens, kVariantName[v], ms, flop,
            mb(w));
      }
      const float rot = timer.Median(
          [&](int) {
            hipLaunchKernelGGL(RotateInputRowsKernel,
                               dim3(tokens * (s.k / 128) / 8), dim3(256), 0, 0,
                               x32, suh, x16, tokens, s.k);
          },
          1, reps);
      Row("projection", s.name, s.k, s.m, tokens, "exl3-input-rotation", rot,
          flop, 0.0);
      // Producer and GEMM inside one event pair.
      const float full = timer.Median(
          [&](int i) {
            hipLaunchKernelGGL(RotateInputRowsKernel,
                               dim3(tokens * (s.k / 128) / 8), dim3(256), 0, 0,
                               x32, suh, x16, tokens, s.k);
            Launch(5, kV2Out, trellis.At(i),
                   reinterpret_cast<const short*>(x16), svh, y, tokens, s.m,
                   s.k, nullptr);
          },
          trellis.count, reps);
      Row("projection", s.name, s.k, s.m, tokens, "exl3-full", full, flop,
          mb(trellis));
      // GGUF production route (prefill_chunk.cpp): Q8_1 below 1024 tokens,
      // FP16 at 1024 and above, each with its activation producer.
      const bool half = tokens >= 1024;
      for (std::size_t g = 0; g < s.gguf.size(); ++g) {
        const std::string name =
            std::string(half ? "gguf-fp16-" : "gguf-q8_1-") + s.gguf[g].name;
        const auto gemm = [&](int i) {
          if (half)
            gufo::hip::LaunchBatchedQuantGEMMFp16(s.gguf[g].type, gguf[g].At(i),
                                                  x16, y, tokens, s.m, s.k,
                                                  nullptr);
          else
            gufo::hip::LaunchBatchedQuantGEMMPreQuantized(
                s.gguf[g].type, gguf[g].At(i), xq8, y, tokens, s.m, s.k,
                nullptr);
        };
        const auto produce = [&] {
          if (half)
            gufo::hip::LaunchFloatToFp16(x32, x16, std::size_t(tokens) * s.k,
                                         nullptr);
          else
            gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(x32, xq8, tokens,
                                                            s.k, nullptr);
        };
        const float gm = timer.Median(gemm, gguf[g].count, reps);
        Row("projection", s.name, s.k, s.m, tokens, name, gm, flop,
            mb(gguf[g]));
        const float fl = timer.Median(
            [&](int i) {
              produce();
              gemm(i);
            },
            gguf[g].count, reps);
        Row("projection", s.name, s.k, s.m, tokens, name + "-full", fl, flop,
            mb(gguf[g]));
      }
    }
    Free(trellis);
    Free(fp16);
    for (Copies& c : gguf)
      Free(c);
    for (void* p : {static_cast<void*>(x32), static_cast<void*>(x16), xq8,
                    static_cast<void*>(suh), static_cast<void*>(svh),
                    static_cast<void*>(y)})
      HIP_OK(hipFree(p));
  }
}

// One FFN chain (RMSNorm -> gate/up -> SwiGLU -> down + residual) per
// method. GGUF: the production route of prefill_chunk.cpp for the dominant
// UD-Q4_K_XL types (gate IQ4_XS, up Q5_K, down Q5_K). EXL3: unfused
// producers (production norm, two input rotations, gate and up GEMMs with
// output rotation, the SwiGLU/down-input rotation, down GEMM with output
// rotation, residual add), an upper bound until the producers are fused.
void TimeFfnChain(Timer& timer, int reps) {
  constexpr int kHidden = 5120, kInter = 17408;
  const int max_tokens = kTokens[std::size(kTokens) - 1];
  Copies tg = TrellisCopies(kHidden, kInter, 5, 21U);
  Copies tu = TrellisCopies(kHidden, kInter, 5, 22U);
  Copies td = TrellisCopies(kInter, kHidden, 5, 23U);
  Copies gg = GgufCopies(kIQ4XS, kHidden, kInter, 31U);
  Copies gu = GgufCopies(kQ5K, kHidden, kInter, 32U);
  Copies gd = GgufCopies(kQ5K, kInter, kHidden, 33U);
  const int cycle =
      std::max({tg.count, tu.count, td.count, gg.count, gu.count, gd.count});
  float *hidden = nullptr, *normed = nullptr, *norm_w = nullptr;
  float *gate = nullptr, *up = nullptr, *down = nullptr;
  _Float16 *xin_g = nullptr, *xin_u = nullptr, *xdown = nullptr;
  _Float16 *suh_g = nullptr, *suh_u = nullptr, *suh_d = nullptr;
  _Float16 *svh_g = nullptr, *svh_u = nullptr, *svh_d = nullptr;
  void *xq8 = nullptr, *xq8_down = nullptr;
  const std::size_t hn = std::size_t(max_tokens) * kHidden;
  const std::size_t in = std::size_t(max_tokens) * kInter;
  for (float** p : {&hidden, &normed, &down})
    HIP_OK(hipMalloc(p, hn * 4));
  for (float** p : {&gate, &up})
    HIP_OK(hipMalloc(p, in * 4));
  HIP_OK(hipMalloc(&norm_w, kHidden * 4));
  for (_Float16** p : {&xin_g, &xin_u})
    HIP_OK(hipMalloc(p, hn * 2));
  HIP_OK(hipMalloc(&xdown, in * 2));
  for (_Float16** p : {&suh_g, &suh_u})
    HIP_OK(hipMalloc(p, kHidden * 2));
  HIP_OK(hipMalloc(&suh_d, kInter * 2));
  for (_Float16** p : {&svh_g, &svh_u})
    HIP_OK(hipMalloc(p, kInter * 2));
  HIP_OK(hipMalloc(&svh_d, kHidden * 2));
  HIP_OK(hipMalloc(&xq8,
                   gufo::hip::QuantizedActivationBytes(max_tokens, kHidden)));
  HIP_OK(hipMalloc(&xq8_down,
                   gufo::hip::QuantizedActivationBytes(max_tokens, kInter)));
  hipLaunchKernelGGL(RandomFloats, dim3(Blocks(hn)), dim3(256), 0, 0, hidden,
                     hn, -2.0F, 2.0F, 41U);
  hipLaunchKernelGGL(RandomFloats, dim3(Blocks(kHidden)), dim3(256), 0, 0,
                     norm_w, std::size_t(kHidden), 0.9F, 1.1F, 42U);
  for (_Float16* p : {suh_g, suh_u})
    hipLaunchKernelGGL(RandomHalves, dim3(Blocks(kHidden)), dim3(256), 0, 0, p,
                       std::size_t(kHidden), 0.5F, 1.0F, 43U);
  hipLaunchKernelGGL(RandomHalves, dim3(Blocks(kInter)), dim3(256), 0, 0, suh_d,
                     std::size_t(kInter), 0.5F, 1.0F, 44U);
  for (_Float16* p : {svh_g, svh_u})
    hipLaunchKernelGGL(RandomHalves, dim3(Blocks(kInter)), dim3(256), 0, 0, p,
                       std::size_t(kInter), 0.005F, 0.02F, 45U);
  hipLaunchKernelGGL(RandomHalves, dim3(Blocks(kHidden)), dim3(256), 0, 0,
                     svh_d, std::size_t(kHidden), 0.005F, 0.02F, 46U);
  HIP_OK(hipDeviceSynchronize());
  const double flop_per_token = 2.0 * 3 * double(kHidden) * kInter;
  for (const int tokens : kTokens) {
    const double flop = flop_per_token * tokens;
    const auto rot_grid = [&](int k) { return dim3(tokens * (k / 128) / 8); };
    const float exl3 = timer.Median(
        [&](int i) {
          gufo::hip::LaunchBatchedRMSNorm(hidden, norm_w, normed, nullptr,
                                          tokens, kHidden, 1e-6F, nullptr);
          hipLaunchKernelGGL(RotateInputRowsKernel, rot_grid(kHidden),
                             dim3(256), 0, 0, normed, suh_g, xin_g, tokens,
                             kHidden);
          hipLaunchKernelGGL(RotateInputRowsKernel, rot_grid(kHidden),
                             dim3(256), 0, 0, normed, suh_u, xin_u, tokens,
                             kHidden);
          Launch(5, kV2Out, tg.At(i), reinterpret_cast<const short*>(xin_g),
                 svh_g, gate, tokens, kInter, kHidden, nullptr);
          Launch(5, kV2Out, tu.At(i), reinterpret_cast<const short*>(xin_u),
                 svh_u, up, tokens, kInter, kHidden, nullptr);
          hipLaunchKernelGGL(SwiGLURotateKernel, rot_grid(kInter), dim3(256), 0,
                             0, gate, up, suh_d, xdown, tokens, kInter);
          Launch(5, kV2Out, td.At(i), reinterpret_cast<const short*>(xdown),
                 svh_d, down, tokens, kHidden, kInter, nullptr);
          gufo::hip::LaunchBatchedResidualAdd(hidden, down, hidden, tokens,
                                              kHidden, nullptr);
        },
        cycle, reps);
    Row("ffn-chain", "ffn", kHidden, kInter, tokens, "exl3-unfused-producers",
        exl3, flop, 0.0);
    const bool half = tokens >= 1024;
    const float gguf = timer.Median(
        [&](int i) {
          if (half) {
            gufo::hip::LaunchBatchedRMSNormFp16(hidden, nullptr, norm_w,
                                                nullptr, xin_g, tokens, kHidden,
                                                1e-6F, nullptr);
            const bool paired =
                gufo::hip::TryLaunchBatchedDualQuantGEMMSwiGLUFp16(
                    kIQ4XS.type, kQ5K.type, gg.At(i), gu.At(i), xin_g, xdown,
                    tokens, kInter, kHidden, nullptr);
            if (!paired) {
              gufo::hip::LaunchBatchedQuantGEMMFp16(kIQ4XS.type, gg.At(i),
                                                    xin_g, gate, tokens, kInter,
                                                    kHidden, nullptr);
              gufo::hip::LaunchBatchedQuantGEMMSwiGLUFp16(
                  kQ5K.type, gu.At(i), xin_g, gate, xdown, tokens, kInter,
                  kHidden, nullptr);
            }
            gufo::hip::LaunchBatchedQuantGEMMResidualFp16(
                kQ5K.type, gd.At(i), xdown, hidden, tokens, kHidden, kInter,
                nullptr);
          } else {
            gufo::hip::LaunchBatchedFusedRMSNormQuantizeQ8_1(
                hidden, nullptr, norm_w, nullptr, xq8, tokens, kHidden, 1e-6F,
                nullptr);
            gufo::hip::LaunchBatchedQuantGEMMPreQuantized(
                kIQ4XS.type, gg.At(i), xq8, gate, tokens, kInter, kHidden,
                nullptr);
            gufo::hip::LaunchBatchedQuantGEMMPreQuantized(
                kQ5K.type, gu.At(i), xq8, up, tokens, kInter, kHidden, nullptr);
            gufo::hip::LaunchBatchedFusedSwiGLUQuantizeQ8_1(
                gate, up, xq8_down, tokens, kInter, nullptr);
            gufo::hip::LaunchBatchedQuantGEMMPreQuantized(
                kQ5K.type, gd.At(i), xq8_down, down, tokens, kHidden, kInter,
                nullptr);
            gufo::hip::LaunchBatchedResidualAdd(hidden, down, hidden, tokens,
                                                kHidden, nullptr);
          }
        },
        cycle, reps);
    Row("ffn-chain", "ffn", kHidden, kInter, tokens,
        half ? "gguf-fp16-route" : "gguf-q8_1-route", gguf, flop, 0.0);
  }
  for (Copies* c : {&tg, &tu, &td, &gg, &gu, &gd})
    Free(*c);
  for (void* p : {static_cast<void*>(hidden), static_cast<void*>(normed),
                  static_cast<void*>(norm_w), static_cast<void*>(gate),
                  static_cast<void*>(up), static_cast<void*>(down),
                  static_cast<void*>(xin_g), static_cast<void*>(xin_u),
                  static_cast<void*>(xdown), static_cast<void*>(suh_g),
                  static_cast<void*>(suh_u), static_cast<void*>(suh_d),
                  static_cast<void*>(svh_g), static_cast<void*>(svh_u),
                  static_cast<void*>(svh_d), xq8, xq8_down})
    HIP_OK(hipFree(p));
}

int Time(int reps) {
  std::printf(
      "scope\tshape\tk\tm\ttokens\tmethod\tmedian_ms\tTFLOPS\t"
      "weight_MB_traversed\n");
  Timer timer;
  TimeProjections(timer, reps);
  TimeFfnChain(timer, reps);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "check";
  if (mode == "check")
    return Check();
  if (mode == "time")
    return Time(argc > 2 ? std::atoi(argv[2]) : 10);
  std::fprintf(stderr, "usage: %s check | time [reps]\n", argv[0]);
  return 2;
}
