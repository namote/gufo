// X3 step 1 (plan/exl3/X3-DESIGN.md): standalone prefill GEMM experiment on
// gfx1151. The trellis-decode weight producer in the frame of the production
// FP16 prefill GEMM (src/models/qwen/hip/kernels/prefill_fp16.hip,
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
//   time   gate/up, down, k/v at 512 and 2048 tokens.
// Build: plan/exl3/X3-RESULTS.md.
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
  const int r_block = blockIdx.y * kBM, t_block = blockIdx.x * kBN;
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

__global__ void RandomHalves(short* x, std::size_t n, unsigned seed) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= n)
    return;
  const unsigned h = Hash(static_cast<unsigned>(i) ^ seed);
  const __half hv = __float2half((static_cast<int>(h % 2001) - 1000) * 0.002F);
  x[i] = *reinterpret_cast<const short*>(&hv);
}

__global__ void RandomFloats(float* x, std::size_t n, unsigned seed) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= n)
    return;
  x[i] =
      (static_cast<int>(Hash(static_cast<unsigned>(i) ^ seed) % 2001) - 1000) *
      0.002F;
}

struct GgufType {
  const char* name;
  GgmlType type;
  int block_bytes, scale, scale_min;
};
constexpr GgufType kGguf[] = {{"Q4_K", GgmlType::kQ4_K, 144, 0, 2},
                              {"Q6_K", GgmlType::kQ6_K, 210, 208, -1},
                              {"IQ4_XS", GgmlType::kIQ4_XS, 136, 0, -1}};
constexpr int kGgufTypes = 3;

// --- check --------------------------------------------------------------

double RelL2(const std::vector<float>& got, const std::vector<double>& want) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    num += (got[i] - want[i]) * (got[i] - want[i]);
    den += want[i] * want[i];
  }
  return std::sqrt(num / den);
}

int Check() {
  int failures = 0;
  {
    std::uint32_t *d_pk = nullptr, *d_mul = nullptr;
    HIP_OK(hipMalloc(&d_pk, 65536 * 4));
    HIP_OK(hipMalloc(&d_mul, 65536 * 4));
    hipLaunchKernelGGL(HashCheckKernel, dim3(256), dim3(256), 0, 0, d_pk,
                       d_mul);
    std::vector<std::uint32_t> pk(65536), mul(65536);
    HIP_OK(hipMemcpy(pk.data(), d_pk, 65536 * 4, hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(mul.data(), d_mul, 65536 * 4, hipMemcpyDeviceToHost));
    std::size_t diff = 0;
    for (std::uint32_t x = 0; x < 65536; ++x)
      diff += pk[x] != mul[x] || mul[x] != x * kMul1;
    std::printf("check pk hash: %zu / 65536 states differ\n", diff);
    failures += diff != 0;
    HIP_OK(hipFree(d_pk));
    HIP_OK(hipFree(d_mul));
  }
  std::mt19937 rng(20261031);
  for (int bits : {4, 5, 6}) {
    const int m = 512, k = 256, batch = 512;
    const exl3q::BitRate rate{bits, false};
    std::vector<std::uint16_t> trellis((k / 16) * (m / 16) * rate.TileWords());
    for (auto& w : trellis)
      w = static_cast<std::uint16_t>(rng());
    std::vector<std::uint16_t> xh(static_cast<std::size_t>(batch) * k);
    std::uniform_real_distribution<float> dist(-2.0F, 2.0F);
    for (auto& v : xh)
      v = exl3q::HalfFromDouble(dist(rng));
    std::vector<std::uint16_t> svh(m);
    for (auto& v : svh)
      v = exl3q::HalfFromDouble(0.01 + 0.02 * dist(rng));
    std::vector<std::uint16_t> inner(static_cast<std::size_t>(k) * m);
    exl3q::DequantInner(trellis.data(), k, m, rate, exl3q::Codebook::kMul1,
                        inner.data());
    // FP16 predecoded control weights, [n][k].
    std::vector<std::uint16_t> wt(static_cast<std::size_t>(m) * k);
    for (int i = 0; i < k; ++i)
      for (int n = 0; n < m; ++n)
        wt[std::size_t(n) * k + i] = inner[std::size_t(i) * m + n];
    std::vector<double> ref(static_cast<std::size_t>(batch) * m, 0.0);
    for (int t = 0; t < batch; ++t)
      for (int i = 0; i < k; ++i) {
        const double xv = exl3q::HalfToDouble(xh[std::size_t(t) * k + i]);
        for (int n = 0; n < m; ++n)
          ref[std::size_t(t) * m + n] +=
              xv * exl3q::HalfToDouble(inner[std::size_t(i) * m + n]);
      }
    std::vector<double> rotated = ref;
    for (int t = 0; t < batch; ++t) {
      exl3q::HadamardBlocks(std::span<double>(&rotated[std::size_t(t) * m], m));
      for (int n = 0; n < m; ++n)
        rotated[std::size_t(t) * m + n] *= exl3q::HalfToDouble(svh[n]);
    }
    std::uint16_t *dw = nullptr, *dwt = nullptr;
    short* dx = nullptr;
    _Float16* dsvh = nullptr;
    float* dy = nullptr;
    HIP_OK(hipMalloc(&dw, trellis.size() * 2));
    HIP_OK(hipMemcpy(dw, trellis.data(), trellis.size() * 2,
                     hipMemcpyHostToDevice));
    HIP_OK(hipMalloc(&dwt, wt.size() * 2));
    HIP_OK(hipMemcpy(dwt, wt.data(), wt.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMalloc(&dx, xh.size() * 2));
    HIP_OK(hipMemcpy(dx, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMalloc(&dsvh, m * 2));
    HIP_OK(hipMemcpy(dsvh, svh.data(), m * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMalloc(&dy, ref.size() * 4));
    std::vector<float> first;
    for (int v = 0; v < kVariants; ++v) {
      HIP_OK(hipMemset(dy, 0, ref.size() * 4));
      Launch(bits, v, v == kFp16Ctl ? static_cast<const void*>(dwt) : dw, dx,
             dsvh, dy, batch, m, k, nullptr);
      HIP_OK(hipDeviceSynchronize());
      std::vector<float> y(ref.size());
      HIP_OK(hipMemcpy(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost));
      const double rel = RelL2(y, v == kV2Out ? rotated : ref);
      failures += !(rel < 1e-5);
      bool same = true;
      if (v == kFp16Ctl)
        first = y;
      else if (v != kV2Out)
        same = std::memcmp(first.data(), y.data(), y.size() * 4) == 0;
      failures += !same;
      std::printf("check K=%d %s: rel L2 vs FP64 %.3e%s\n", bits,
                  kVariantName[v], rel,
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
    // Input rotation against FP64 (rounded to FP16 the same way).
    const int tokens = 64, k = 512;
    std::vector<float> x(static_cast<std::size_t>(tokens) * k);
    std::uniform_real_distribution<float> dist(-2.0F, 2.0F);
    for (auto& v : x)
      v = dist(rng);
    std::vector<std::uint16_t> suh(k);
    for (auto& v : suh)
      v = exl3q::HalfFromDouble(0.5 + 0.5 * dist(rng));
    float* dx = nullptr;
    _Float16 *dsuh = nullptr, *dout = nullptr;
    HIP_OK(hipMalloc(&dx, x.size() * 4));
    HIP_OK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMalloc(&dsuh, k * 2));
    HIP_OK(hipMemcpy(dsuh, suh.data(), k * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMalloc(&dout, x.size() * 2));
    hipLaunchKernelGGL(RotateInputRowsKernel, dim3(tokens * k / 128 / 8),
                       dim3(256), 0, 0, dx, dsuh, dout, tokens, k);
    std::vector<std::uint16_t> out(x.size());
    HIP_OK(hipMemcpy(out.data(), dout, out.size() * 2, hipMemcpyDeviceToHost));
    double worst = 0.0;
    for (int t = 0; t < tokens; ++t) {
      std::vector<double> r(k);
      for (int i = 0; i < k; ++i)
        r[i] = x[std::size_t(t) * k + i] * exl3q::HalfToDouble(suh[i]);
      exl3q::HadamardBlocks(std::span<double>(r));
      for (int i = 0; i < k; ++i)
        worst = std::max(
            worst,
            std::fabs(exl3q::HalfToDouble(out[std::size_t(t) * k + i]) - r[i]) /
                (std::fabs(r[i]) + 1e-2));
    }
    // FP16 rounding of the result bounds the error (2^-11 relative).
    failures += !(worst < 1e-3);
    std::printf("check input rotation: worst relative %.3e (FP16 output)\n",
                worst);
    for (void* p : {static_cast<void*>(dx), static_cast<void*>(dsuh),
                    static_cast<void*>(dout)})
      HIP_OK(hipFree(p));
  }
  std::printf("check: %s\n", failures == 0 ? "all passed" : "FAILED");
  return failures == 0 ? 0 : 1;
}

// --- time ---------------------------------------------------------------

struct Shape {
  const char* name;
  int k, m;
};
// The review's order: gate/up, down, k/v.
constexpr Shape kShapes[] = {
    {"gate_up", 5120, 17408}, {"down", 17408, 5120}, {"k_v", 5120, 1024}};
constexpr int kTokens[] = {512, 2048};

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
  // Median ms of `launch(copy)` over reps, rotating the weight copy.
  template<typename F>
  float Median(F&& launch, int copies, int reps) {
    for (int i = 0; i < 2; ++i)
      launch(i % copies);
    HIP_OK(hipDeviceSynchronize());
    std::vector<float> t;
    for (int i = 0; i < reps; ++i) {
      HIP_OK(hipEventRecord(e0_, nullptr));
      launch(i % copies);
      HIP_OK(hipEventRecord(e1_, nullptr));
      HIP_OK(hipEventSynchronize(e1_));
      float ms = 0.0F;
      HIP_OK(hipEventElapsedTime(&ms, e0_, e1_));
      t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
  }

private:
  hipEvent_t e0_, e1_;
};

int CopiesFor(std::size_t bytes) {
  return static_cast<int>(std::max<std::size_t>(2, (256U << 20) / bytes + 1));
}

int Time(int reps) {
  std::printf("shape\tk\tm\ttokens\tmethod\tmedian_ms\tTFLOPS\n");
  Timer timer;
  for (const Shape& s : kShapes) {
    const exl3q::BitRate rate{5, false};
    const std::size_t trellis_bytes =
        static_cast<std::size_t>(s.k / 16) * (s.m / 16) * rate.TileWords() * 2;
    const std::size_t fp16_bytes = static_cast<std::size_t>(s.m) * s.k * 2;
    const int tc = CopiesFor(trellis_bytes), fc = CopiesFor(fp16_bytes);
    std::uint16_t *trellis = nullptr, *fp16w = nullptr;
    HIP_OK(hipMalloc(&trellis, trellis_bytes * tc));
    hipLaunchKernelGGL(RandomHalves, dim3((trellis_bytes * tc / 2 + 255) / 256),
                       dim3(256), 0, 0, reinterpret_cast<short*>(trellis),
                       trellis_bytes * tc / 2, 5U);
    HIP_OK(hipMalloc(&fp16w, fp16_bytes * fc));
    hipLaunchKernelGGL(RandomHalves, dim3((fp16_bytes * fc / 2 + 255) / 256),
                       dim3(256), 0, 0, reinterpret_cast<short*>(fp16w),
                       fp16_bytes * fc / 2, 6U);
    void* gguf[kGgufTypes] = {};
    std::size_t gguf_bytes[kGgufTypes] = {};
    int gc[kGgufTypes] = {};
    for (int g = 0; g < kGgufTypes; ++g) {
      const std::size_t blocks = static_cast<std::size_t>(s.m) * (s.k / 256);
      gguf_bytes[g] = blocks * kGguf[g].block_bytes;
      gc[g] = CopiesFor(gguf_bytes[g]);
      HIP_OK(hipMalloc(&gguf[g], gguf_bytes[g] * gc[g]));
      hipLaunchKernelGGL(InitBlocks, dim3((blocks * gc[g] + 255) / 256),
                         dim3(256), 0, 0, static_cast<unsigned char*>(gguf[g]),
                         blocks * gc[g], kGguf[g].block_bytes, kGguf[g].scale,
                         kGguf[g].scale_min, 11U + g);
    }
    const int max_tokens = kTokens[std::size(kTokens) - 1];
    const std::size_t xn = static_cast<std::size_t>(max_tokens) * s.k;
    float* x32 = nullptr;
    short* x16 = nullptr;
    void* xq8 = nullptr;
    _Float16 *suh = nullptr, *svh = nullptr;
    float* y = nullptr;
    HIP_OK(hipMalloc(&x32, xn * 4));
    hipLaunchKernelGGL(RandomFloats, dim3((xn + 255) / 256), dim3(256), 0, 0,
                       x32, xn, 3U);
    HIP_OK(hipMalloc(&x16, xn * 2));
    HIP_OK(
        hipMalloc(&xq8, gufo::hip::QuantizedActivationBytes(max_tokens, s.k)));
    HIP_OK(hipMalloc(&suh, s.k * 2));
    hipLaunchKernelGGL(RandomHalves, dim3((s.k + 255) / 256), dim3(256), 0, 0,
                       reinterpret_cast<short*>(suh), s.k, 8U);
    HIP_OK(hipMalloc(&svh, s.m * 2));
    hipLaunchKernelGGL(RandomHalves, dim3((s.m + 255) / 256), dim3(256), 0, 0,
                       reinterpret_cast<short*>(svh), s.m, 9U);
    HIP_OK(hipMalloc(&y, static_cast<std::size_t>(max_tokens) * s.m * 4));
    hipLaunchKernelGGL(RandomHalves, dim3((xn + 255) / 256), dim3(256), 0, 0,
                       x16, xn, 4U);
    HIP_OK(hipDeviceSynchronize());
    for (const int tokens : kTokens) {
      const double flop = 2.0 * tokens * s.m * double(s.k);
      const auto row = [&](const char* method, float ms) {
        std::printf("%s\t%d\t%d\t%d\t%s\t%.3f\t%.2f\n", s.name, s.k, s.m,
                    tokens, method, ms, flop / (ms * 1e-3) / 1e12);
      };
      for (int v = 0; v < kVariants; ++v) {
        const bool fp16 = v == kFp16Ctl;
        const std::size_t stride = fp16 ? fp16_bytes : trellis_bytes;
        const auto* base = reinterpret_cast<const unsigned char*>(
            fp16 ? static_cast<void*>(fp16w) : trellis);
        row(kVariantName[v], timer.Median(
                                 [&](int c) {
                                   Launch(5, v, base + c * stride, x16, svh, y,
                                          tokens, s.m, s.k, nullptr);
                                 },
                                 fp16 ? fc : tc, reps));
      }
      const float rot = timer.Median(
          [&](int) {
            hipLaunchKernelGGL(RotateInputRowsKernel,
                               dim3(tokens * (s.k / 128) / 8), dim3(256), 0, 0,
                               x32, suh, reinterpret_cast<_Float16*>(x16),
                               tokens, s.k);
          },
          1, reps);
      row("exl3-input-rotation", rot);
      row("exl3-full",
          rot + timer.Median(
                    [&](int c) {
                      Launch(5, kV2Out,
                             reinterpret_cast<const unsigned char*>(trellis) +
                                 c * trellis_bytes,
                             x16, svh, y, tokens, s.m, s.k, nullptr);
                    },
                    tc, reps));
      // GGUF production route (prefill_chunk.cpp): Q8_1 below 1024 tokens,
      // FP16 at 1024 and above, each with its activation producer.
      const bool half = tokens >= 1024;
      const float produce = timer.Median(
          [&](int) {
            if (half)
              gufo::hip::LaunchFloatToFp16(x32, x16, xn / max_tokens * tokens,
                                           nullptr);
            else
              gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(x32, xq8, tokens,
                                                              s.k, nullptr);
          },
          1, reps);
      row(half ? "gguf-fp16-activation" : "gguf-q8_1-activation", produce);
      for (int g = 0; g < kGgufTypes; ++g) {
        const auto* gb = static_cast<const unsigned char*>(gguf[g]);
        const float gemm = timer.Median(
            [&](int c) {
              if (half)
                gufo::hip::LaunchBatchedQuantGEMMFp16(
                    kGguf[g].type, gb + c * gguf_bytes[g], x16, y, tokens, s.m,
                    s.k, nullptr);
              else
                gufo::hip::LaunchBatchedQuantGEMMPreQuantized(
                    kGguf[g].type, gb + c * gguf_bytes[g], xq8, y, tokens, s.m,
                    s.k, nullptr);
            },
            gc[g], reps);
        const std::string name =
            std::string(half ? "gguf-fp16-" : "gguf-q8_1-") + kGguf[g].name;
        row(name.c_str(), gemm);
        row((name + "-full").c_str(), produce + gemm);
      }
    }
    for (void* p : {static_cast<void*>(trellis), static_cast<void*>(fp16w),
                    gguf[0], gguf[1], gguf[2], static_cast<void*>(x32),
                    static_cast<void*>(x16), xq8, static_cast<void*>(suh),
                    static_cast<void*>(svh), static_cast<void*>(y)})
      HIP_OK(hipFree(p));
  }
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
