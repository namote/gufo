// EXL3 prefill GEMM ceiling screen on gfx1151
// (plan/exl3/archive/B7-CEILING-BRIEF.md Q1). Real Flash-Next c03 shapes:
// routed gate/up K2560 -> N640 (two banks) and down K640 -> N2560 at 40..640
// rows per expert (10-of-512 routing, balanced and skewed), and dense K2560 ->
// N10240, K6144 -> N2560, K2560 -> N6144 at 2048/4096/8192 rows. Variants per
// shape:
//   a   production: the real launchers (LaunchExl3PrefillBucketInnerHalf with
//       device buckets from Exl3BuildExpertBuckets;
//       LaunchExl3PrefillGemmGrouped)
//   a*  this file's copy of the production trellis kernel (same tiles, block
//       mapping, staging, swizzle, WMMA order and epilogue); must equal a,
//       which validates b..d as derived from today's kernel
//   b   no-decode at FP16 fetch cost: a* with the A operand read as
//       pre-decoded FP16 tiles from global (512 B per tile versus 128/160/192
//       B of K4/K5/K6 trellis, so 2.7-4x the weight bytes); equal to a. NOT
//       an upper bound: where weight bytes matter (routed, few rows per
//       expert) b can lose to a. c is the no-decode bound.
//   c   WMMA-only: A decoded into LDS once and kept resident; the activation
//       tiles are still staged every K step (tiling + B-fetch ceiling)
//   c0  LDS-only: A and B staged once; the K loop keeps a*'s two block
//       barriers per K64 stage plus LDS reads and WMMA
//   d   decode-only: a*'s trellis fetch, decode, LDS stores and barriers; no
//       B operand, no WMMA, no epilogue (in-kernel decode cost)
//   dg  standalone decode of each bank/matrix once to global FP16 (decode +
//       store throughput; also produces b's operand). The GEMMs decode a
//       tile once per bucket (routed, > 128 rows per expert) or per token
//       tile row (dense), so dg undercounts in-GEMM decode work.
//   e   warp-specialized: producer waves fetch and decode A (and stage B)
//       into an LDS ring, consumer waves only read LDS and issue WMMA.
//       e-bar: ring of 2, one block barrier per stage; e-flag: ring of 3,
//       LDS counters (no block barrier in the K loop). Every output keeps
//       the same operands and ascending K16 WMMA chain as a: bit-equal.
// Exactness: a*, b, e-bar and e-flag outputs are compared bit for bit with a
// (unwritten outputs are NaN sentinels and are counted). c, c0, d compute
// other values by design. µs per call is the best of kReps averages; TF
// counts useful FLOPs (2 * rows * K * N per projection). Resources (VGPR,
// LDS, scratch, blocks per CU as the runtime reports them, which may be per
// CU or per WGP) come from the runtime for this file's kernels. Production
// resources from its ISA (device-only -S of prefill_fp16.hip with the
// build's flags, .amdhsa_next_free_vgpr / group_segment_fixed_size): routed
// 128x64/128x128 121/169 VGPR, 24/32 KiB LDS (a* identical); dense 256x256
// 187 VGPR (K5/K6), 64 KiB (a*: 192). Only the a vs a* time verifies the
// copy. e is not occupancy-matched: routed e-bar 128x128 uses 64 KiB LDS
// (production 32 KiB) and dense e stages K32 (bar) or K16 (flag) per
// barrier/flag versus production's K64. Dense trellis (10-20 MB) stays in
// the 32 MB MALL across repetitions; b's FP16 copy is 31 MB (N2560, N6144:
// about MALL size) or 52 MB (N10240), so the b-versus-a cache bias depends
// on the shape. A real GEMM reuses its weights across token tiles anyway.
// Balanced 80 rows/expert ("quick") uses only the 128-token routed tile.
//
// Build (CMake, links the production library, its default HIP flags):
//   cmake --build --preset gpu-test --target
//   qwen38_flash_next_exl3_wmma_ceiling_bench
// Run (all, about 3-5 minutes; peak about 10 GB device memory):
//   .../qwen38_flash_next_exl3_wmma_ceiling_bench [routed|dense] [quick]
// "quick" keeps 80 rows per expert / 4096 dense rows only.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "src/core/hip/detail/exl3_gemv.hpp"
#include "src/core/quant/exl3/device.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/exl3_moe.hpp"

namespace ex = gufo::quant::exl3;
namespace ar = gufo::hip::exl3;
namespace fn = gufo::models::qwen38_flash_next::rocm;
using Bucket = gufo::hip::Exl3PrefillBucket;

namespace {

constexpr unsigned kExperts = 512, kUsed = 10, kHidden = 2560, kFf = 640;
constexpr unsigned kTileRows = 64;
constexpr int kReps = 5;
constexpr double kTargetRepMs = 15.0;

void Ck(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}

template<class T>
T* Alloc(std::size_t count) {
  T* d = nullptr;
  Ck(hipMalloc(&d, std::max<std::size_t>(count, 1) * sizeof(T)), "malloc");
  return d;
}

template<class T>
T* Upload(const std::vector<T>& h) {
  T* d = Alloc<T>(h.size());
  Ck(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice),
     "upload");
  return d;
}

std::uint32_t Next(std::uint32_t* s) {
  *s ^= *s << 13;
  *s ^= *s >> 17;
  *s ^= *s << 5;
  return *s;
}

// ---------------------------------------------------------------- device ----

__device__ __forceinline__ std::uint32_t Mix(std::uint64_t i,
                                             std::uint32_t seed) {
  std::uint32_t h = static_cast<std::uint32_t>(i) * 0x9E3779B1U ^
                    static_cast<std::uint32_t>(i >> 32) * 0x85EBCA77U ^ seed;
  h ^= h >> 16;
  h *= 0x7FEB352DU;
  h ^= h >> 15;
  h *= 0x846CA68BU;
  h ^= h >> 16;
  return h;
}

__global__ void FillWords(std::uint32_t* p, std::size_t n, std::uint32_t seed) {
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
       i < n; i += std::size_t{gridDim.x} * blockDim.x)
    p[i] = Mix(i, seed);
}

// FP16 values (hash % 4096 - 2048) / 1024 * scale.
__global__ void FillHalf(_Float16* p, std::size_t n, std::uint32_t seed,
                         float scale) {
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
       i < n; i += std::size_t{gridDim.x} * blockDim.x)
    p[i] = static_cast<_Float16>(
        (static_cast<float>(Mix(i, seed) % 4096) - 2048.0F) / 1024.0F * scale);
}

__global__ void FillExponents(std::int32_t* p, std::size_t n,
                              std::uint32_t seed) {
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
       i < n; i += std::size_t{gridDim.x} * blockDim.x) {
    const unsigned h = Mix(i, seed) % 8;
    p[i] = h < 5 ? 0 : static_cast<std::int32_t>(h - 4);  // 0 or 1..3
  }
}

// counts[0]: words differing from ref; counts[1]: NaN sentinels left in ref.
template<class Word>
__global__ void CountDiff(const Word* ref, const Word* cand, std::size_t n,
                          unsigned long long* counts) {
  unsigned long long diff = 0, unwritten = 0;
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
       i < n; i += std::size_t{gridDim.x} * blockDim.x) {
    diff += ref[i] != cand[i];
    unwritten += ref[i] == static_cast<Word>(~Word{0});
  }
  if (diff)
    atomicAdd(&counts[0], diff);
  if (unwritten)
    atomicAdd(&counts[1], unwritten);
}

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

// Production WMMA, LDS swizzle and stores (src/models/qwen/hip/kernels/
// prefill_fp16.hip), verbatim.
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

struct TrellisRaw {
  std::uint32_t d0, d1, d2;
};

// The lane's four decoded pairs of one 16 x 16 tile, in production order:
// p[2h], p[2h + 1] are rows (r0, r0+1) and (r0+8, r0+9) of column c0 + 8h.
template<int Code>
__device__ __forceinline__ void DecodeTile(const TrellisRaw& t,
                                           const ar::LaneWindow& lw,
                                           ar::Half2 (&p)[4]) {
  std::uint32_t hashed[8];
  ar::LaneStates<Code>(t.d0, t.d1, t.d2, lw.shift, hashed);
#pragma unroll
  for (int j = 0; j < 8; ++j)
    hashed[j] = ar::HashMul1Packed(hashed[j]);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    p[2 * h] = ar::DecodePairMul1(hashed[4 * h], hashed[4 * h + 1]);
    p[2 * h + 1] = ar::DecodePairMul1(hashed[4 * h + 2], hashed[4 * h + 3]);
  }
}

// Production placement of a decoded tile: `slab` is s_a[ks] ([RS][16][16]),
// rt the 16-row weight tile, swz the K step's XOR (ks & 3).
__device__ __forceinline__ void StoreTile(short* slab, int rt, unsigned swz,
                                          unsigned lane,
                                          const ar::Half2 (&p)[4]) {
  const unsigned r0 = (lane % 4) * 2, c0 = lane / 4;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const unsigned row = rt * 16 + c0 + 8 * h;
    const unsigned placed = row ^ swz;
    const unsigned shift = (row ^ (row >> 2)) & 1;
    auto* pairs = reinterpret_cast<ar::Half2*>(slab + placed * 16);
    pairs[((0 ^ shift) * 8 + r0) / 2] = p[2 * h];
    pairs[((1 ^ shift) * 8 + r0) / 2] = p[2 * h + 1];
  }
}

__device__ __forceinline__ uint4 PackPairs(const ar::Half2 (&p)[4]) {
  return {__builtin_bit_cast(std::uint32_t, p[0]),
          __builtin_bit_cast(std::uint32_t, p[1]),
          __builtin_bit_cast(std::uint32_t, p[2]),
          __builtin_bit_cast(std::uint32_t, p[3])};
}
__device__ __forceinline__ void UnpackPairs(uint4 q, ar::Half2 (&p)[4]) {
  p[0] = __builtin_bit_cast(ar::Half2, q.x);
  p[1] = __builtin_bit_cast(ar::Half2, q.y);
  p[2] = __builtin_bit_cast(ar::Half2, q.z);
  p[3] = __builtin_bit_cast(ar::Half2, q.w);
}

// Decode tiles [first, first + count) of a trellis of `tiles` tiles into
// lane-major FP16 pairs: one wave per tile, one uint4 per lane.
template<int Code>
__global__ void PredecodeKernel(const std::uint32_t* __restrict__ trellis,
                                uint4* __restrict__ out, std::size_t tiles) {
  const unsigned lane = threadIdx.x % 32;
  const ar::LaneWindow lw = ar::MakeLaneWindow<Code>(lane);
  for (std::size_t t =
           blockIdx.x * std::size_t{blockDim.x / 32} + threadIdx.x / 32;
       t < tiles; t += std::size_t{gridDim.x} * (blockDim.x / 32)) {
    const std::uint32_t* tile = trellis + t * ar::TileDwords(Code);
    ar::Half2 p[4];
    DecodeTile<Code>({tile[lw.o0], tile[lw.o1], tile[lw.o2]}, lw, p);
    out[t * 32 + lane] = PackPairs(p);
  }
}

struct Args {
  const void* w;      // dense: trellis dwords; routed: DeviceLinear bank
  const uint4* half;  // pre-decoded tiles (b): dense base / routed bank base
  std::size_t expert_half;  // routed: uint4 per expert in `half`
  const short* x;           // FP16 rows [batch][k] (routed: packed slab)
  void* y;                  // dense FP32 [batch][m]; routed FP16 [rows][m]
  const _Float16* svh;      // dense epilogue
  const Bucket* buckets;
  const std::uint32_t* bucket_count;
  const std::int32_t* row_exponents;
  int batch, m, k;
  unsigned* error;  // [0] error bits (1 timeout, 2 bank contract), [1] sink
};

enum class Src : int { kTrellis, kHalf, kResident, kLdsOnly, kDecodeOnly };

// Block setup shared by both kernels: tile origin, operands and row count.
// Returns false when the block has no work (exactly as production).
template<int BM, int BN, bool Routed>
__device__ __forceinline__ bool Setup(const Args& args, int& r_block,
                                      int& t_block, int& batch,
                                      const std::uint32_t*& trellis,
                                      const uint4*& half, const short*& x,
                                      void*& y, const std::int32_t*& exps,
                                      int code) {
  const int m = args.m, k = args.k;
  batch = args.batch;
  unsigned row_tile = blockIdx.y, token_tile = blockIdx.x;
  if constexpr (!Routed) {
    // Production block mapping (Batching 1, 256 x 256, kStore).
    unsigned group_shift = 3;
    if (k == 6144 && m == 2560 && batch == 2048 && gridDim.z == 1)
      group_shift = 0;
    const unsigned group = 1U << group_shift;
    const unsigned first = (blockIdx.y >> group_shift) << group_shift,
                   within = (blockIdx.y & (group - 1)) * gridDim.x + blockIdx.x,
                   rows = min(group, gridDim.y - first);
    row_tile = first + within % rows;
    token_tile = within / rows;
  }
  r_block = row_tile * BM;
  t_block = token_tile * BN;
  x = args.x;
  y = args.y;
  exps = args.row_exponents;
  if constexpr (Routed) {
    if (blockIdx.z >= *args.bucket_count)
      return false;
    const Bucket bucket = args.buckets[blockIdx.z];
    batch = static_cast<int>(bucket.count);
    if (t_block >= batch || batch > BN || (BN > 64 && batch <= BN / 2))
      return false;
    const auto& linear =
        static_cast<const ex::DeviceLinear*>(args.w)[bucket.expert];
    if (linear.codebook != ex::Codebook::kMul1 ||
        ar::RateCode(linear.rate.bits, linear.rate.half) != code ||
        linear.k != unsigned(k) || linear.n != unsigned(m)) {
      if (threadIdx.x == 0)
        atomicOr(args.error, 2U);
      return false;
    }
    x += static_cast<std::size_t>(bucket.first) * k;
    y = static_cast<_Float16*>(y) + static_cast<std::size_t>(bucket.first) * m;
    if (exps != nullptr)
      exps += bucket.first;
    trellis = reinterpret_cast<const std::uint32_t*>(linear.trellis);
    half = args.half == nullptr ? nullptr
                                : args.half + bucket.expert * args.expert_half;
  } else {
    if (t_block >= batch)
      return false;
    trellis = static_cast<const std::uint32_t*>(args.w);
    half = args.half;
  }
  return true;
}

// Routed epilogue (production Batching 3): FP16 rows, divided by
// 2^row_exponents[row] when that is positive.
template<int WRS, int WTS>
__device__ __forceinline__ void StoreRouted(const v8f (&acc)[WRS][WTS],
                                            _Float16* y,
                                            const std::int32_t* exps, int m,
                                            int r_block, int t_block, int batch,
                                            int wr, int wt, int sl, int hi) {
#pragma unroll
  for (int j = 0; j < WTS; ++j) {
#pragma unroll
    for (int l = 0; l < 8; ++l) {
      const int token = t_block + (wt * WTS + j) * 16 + 2 * l + hi;
      if (token >= batch)
        continue;
      float scale = 1.0F;
      if (exps != nullptr) {
        const int e = exps[token];
        if (e > 0)
          scale = __builtin_bit_cast(float, unsigned(127 - e) << 23);
      }
#pragma unroll
      for (int i = 0; i < WRS; ++i) {
        const int row = r_block + (wr * WRS + i) * 16 + sl;
        reinterpret_cast<__half*>(
            y)[static_cast<std::size_t>(token) * m + row] =
            __float2half_rn(acc[i][j][l] * scale);
      }
    }
  }
}

// Dense epilogue (production trellis kStore): y = svh * H(acc) per 128-row
// segment, staged through `c` (Stage tokens of BM floats), owner column
// waves write, every wave of the block rotates. Must be called by all waves.
// Unrolled: compile-time passes, so every accumulator index is constant
// (the warp-specialized kernel otherwise indexes acc through scratch).
template<int BM, int BN, int WRS, int WTS, int Stage, bool Unrolled = false>
__device__ __forceinline__ void StoreDense(const v8f (&acc)[WRS][WTS], float* c,
                                           float* y, const _Float16* svh, int m,
                                           int r_block, int t_block, int batch,
                                           int wr, int wt, bool writer,
                                           int wave, int waves, int lane) {
  static_assert(BM % 128 == 0 && Stage % 16 == 0 && (WTS * 16) % Stage == 0);
  constexpr int TilesPerStage = Stage / 16;
  const int sl = lane & 15, hi = lane / 16;
  const auto one_pass = [&](int pass) {
    const int owner = pass / (WTS / TilesPerStage);
    const int j0 = (pass % (WTS / TilesPerStage)) * TilesPerStage;
    if (writer && wt == owner) {
#pragma unroll
      for (int i = 0; i < WRS; ++i) {
#pragma unroll
        for (int j = 0; j < WTS; ++j) {
          if (j < j0 || j >= j0 + TilesPerStage)
            continue;
          const int row = (wr * WRS + i) * 16 + sl;
#pragma unroll
          for (int l = 0; l < 8; ++l)
            c[((j - j0) * 16 + 2 * l + hi) * BM + row] = acc[i][j][l];
        }
      }
    }
    __syncthreads();
    for (int item = wave; item < Stage * (BM / 128); item += waves) {
      const int t = item / (BM / 128), seg = item % (BM / 128);
      const int token = t_block + pass * Stage + t;
      float v[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        v[j] = c[t * BM + seg * 128 + lane * 4 + j];
      ar::Butterfly128(v, static_cast<unsigned>(lane));
      if (token < batch) {
        const int row = r_block + seg * 128 + lane * 4;
        float* dst = y + static_cast<std::size_t>(token) * m + row;
#pragma unroll
        for (int j = 0; j < 4; ++j)
          dst[j] =
              (v[j] * ar::kHadamardScale) * static_cast<float>(svh[row + j]);
      }
    }
    __syncthreads();
  };
  if constexpr (Unrolled) {
    [&]<int... P>(std::integer_sequence<int, P...>) {
      (one_pass(P), ...);
    }(std::make_integer_sequence<int, BN / Stage>{});
  } else {
    for (int pass = 0; pass < BN / Stage; ++pass)
      one_pass(pass);
  }
}

// ---- a*, b, c, c0, d: the production kernel structure ----
template<int BM, int BN, int WM, int WN, int Code, bool Routed, Src S>
__launch_bounds__(WM * WN * 32) __global__ void StagedGemm(Args args) {
  constexpr int BK = 4, kThreads = WM * WN * 32, RS = BM / 16, TS = BN / 16;
  constexpr int WRS = RS / WM, WTS = TS / WN;
  constexpr int PFB = (BN * BK + kThreads - 1) / kThreads;
  constexpr int TPW = (BK * RS) / (WM * WN);
  static_assert((BK * RS) % (WM * WN) == 0 && BM % (16 * WM) == 0 &&
                BN % (16 * WN) == 0);
  constexpr bool kStageB =
      S == Src::kTrellis || S == Src::kHalf || S == Src::kResident;
  __shared__ short s_a[BK][RS][16][16];
  __shared__ short s_b[BK][TS][16][16];

  const int tid = static_cast<int>(threadIdx.x), wave = tid / 32,
            lane = tid % 32, sl = lane & 15, hi = lane / 16;
  const int wr = wave / WN, wt = wave % WN;
  int r_block, t_block, batch;
  const std::uint32_t* trellis;
  const uint4* half;
  const short* x;
  void* y;
  const std::int32_t* exps;
  if (!Setup<BM, BN, Routed>(args, r_block, t_block, batch, trellis, half, x, y,
                             exps, Code))
    return;
  const int m = args.m, k = args.k, ksteps = k / 16;
  v8f acc[WRS][WTS] = {};
  const ar::LaneWindow lw = ar::MakeLaneWindow<Code>(lane);
  TrellisRaw traw[TPW];
  uint4 hraw[S == Src::kHalf ? TPW : 1];
  const short* x_row[PFB];
#pragma unroll
  for (int p = 0; p < PFB; ++p) {
    const int t = t_block + (((p * kThreads) + tid) / BK);
    x_row[p] = x + static_cast<std::size_t>(t < batch ? t : batch - 1) * k;
  }
  uint4 rb[PFB][2];
  const auto fetch = [&](int ks0, bool first) {
    if (S == Src::kTrellis || S == Src::kDecodeOnly || first) {
#pragma unroll
      for (int i = 0; i < TPW; ++i) {
        const int t = wave * TPW + i, ks = t / RS, rt = t % RS;
        const std::size_t index =
            static_cast<std::size_t>(ks0 + ks) * (m / 16) + (r_block / 16) + rt;
        if constexpr (S == Src::kHalf) {
          hraw[i] = half[index * 32 + lane];
        } else {
          const std::uint32_t* tile = trellis + index * ar::TileDwords(Code);
          traw[i] = {tile[lw.o0], tile[lw.o1], tile[lw.o2]};
        }
      }
    } else if constexpr (S == Src::kHalf) {
#pragma unroll
      for (int i = 0; i < TPW; ++i) {
        const int t = wave * TPW + i, ks = t / RS, rt = t % RS;
        const std::size_t index =
            static_cast<std::size_t>(ks0 + ks) * (m / 16) + (r_block / 16) + rt;
        hraw[i] = half[index * 32 + lane];
      }
    }
    if (kStageB || (S == Src::kLdsOnly && first)) {
#pragma unroll
      for (int p = 0; p < PFB; ++p) {
        const int idx = p * kThreads + tid, kk = ks0 + idx % BK;
        if constexpr (Routed) {
          if (t_block + ((idx / BK) / 16) * 16 >= batch)
            continue;
        }
        // Production's guard: always true for EXL3 k (a multiple of 128),
        // but the dense kernel evaluates it (Batching 1).
        if (idx < BN * BK && (Routed || kk < ksteps)) {
          const short* src = x_row[p] + static_cast<std::size_t>(kk) * 16;
          rb[p][0] = *reinterpret_cast<const uint4*>(src);
          rb[p][1] = *reinterpret_cast<const uint4*>(src + 8);
        } else {
          rb[p][0] = make_uint4(0, 0, 0, 0);
          rb[p][1] = make_uint4(0, 0, 0, 0);
        }
      }
    }
  };
  const auto commit = [&](bool first) {
    if (S == Src::kTrellis || S == Src::kHalf || S == Src::kDecodeOnly ||
        first) {
#pragma unroll
      for (int i = 0; i < TPW; ++i) {
        const int t = wave * TPW + i, ks = t / RS, rt = t % RS;
        ar::Half2 p[4];
        if constexpr (S == Src::kHalf)
          UnpackPairs(hraw[i], p);
        else
          DecodeTile<Code>(traw[i], lw, p);
        StoreTile(&s_a[ks][0][0][0], rt, ks & 3, lane, p);
      }
    }
    if (kStageB || (S == Src::kLdsOnly && first)) {
#pragma unroll
      for (int p = 0; p < PFB; ++p) {
        const int idx = p * kThreads + tid, row = idx / BK, ks = idx % BK;
        if constexpr (Routed) {
          if (t_block + (row / 16) * 16 >= batch)
            continue;
        }
        if (idx < BN * BK) {
          const int placed = row ^ (ks & 3);
          StoreSwizzled(&s_b[ks][placed / 16][placed % 16][0], rb[p][0],
                        rb[p][1], row);
        }
      }
    }
  };
  unsigned sink = 0;
  fetch(0, true);
  for (int ks0 = 0; ks0 < ksteps; ks0 += BK) {
    commit(ks0 == 0);
    __syncthreads();
    if (ks0 + BK < ksteps)
      fetch(ks0 + BK, false);
    if constexpr (S == Src::kDecodeOnly) {
      // Keep the LDS stores live: one read per lane and step.
      sink ^= static_cast<unsigned short>(s_a[ks0 & 3][wave % RS][sl][hi]);
    } else {
#pragma unroll
      for (int ks = 0; ks < BK; ++ks) {
        v16h a[WRS];
#pragma unroll
        for (int i = 0; i < WRS; ++i)
          a[i] = LoadSwizzled(&s_a[ks][(wr * WRS) + i][sl ^ (ks & 3)][0]);
#pragma unroll
        for (int j = 0; j < WTS; ++j) {
          if constexpr (Routed) {
            if (t_block + (wt * WTS + j) * 16 >= batch)
              continue;
          }
          const v16h b =
              LoadSwizzled(&s_b[ks][(wt * WTS) + j][sl ^ (ks & 3)][0]);
#pragma unroll
          for (int i = 0; i < WRS; ++i)
            acc[i][j] = WmmaHalf(b, a[i], acc[i][j]);
        }
      }
      if constexpr (!Routed && BM == 256 && BN == 256)
        __builtin_amdgcn_iglp_opt(0);
    }
    __syncthreads();
  }
  if constexpr (S == Src::kDecodeOnly) {
    if (sink == 0xBEEFU)              // reachable, so the LDS stores stay live
      atomicAdd(args.error + 1, 1U);  // a sink word, not an error bit
    return;
  } else if constexpr (Routed) {
    StoreRouted<WRS, WTS>(acc, static_cast<_Float16*>(y), exps, m, r_block,
                          t_block, batch, wr, wt, sl, hi);
  } else {
    constexpr int Stage = static_cast<int>(sizeof(s_a) / sizeof(float)) / BM;
    StoreDense<BM, BN, WRS, WTS, Stage>(
        acc, reinterpret_cast<float*>(&s_a[0][0][0][0]), static_cast<float*>(y),
        args.svh, m, r_block, t_block, batch, wr, wt, true, wave, WM * WN,
        lane);
  }
}

// ---- e: warp-specialized prototype ----
// Waves [0, CW) consume (CWM x CWN grid of the production output tiling, LDS
// reads + WMMA only); waves [CW, CW + PW) produce stage s (BK K16 steps of
// decoded A tiles and FP16 B tiles) into ring slot s % Ring. Flags = false:
// Ring 2 and one block barrier per stage (producers fill s + 1 while
// consumers compute s). Flags = true: monotonic LDS counters full[slot]
// (producer waves done) and empty[slot] (consumer waves done); waits are
// bounded (error bit 1 on timeout) so a protocol bug cannot hang the GPU.
__device__ __forceinline__ bool WaitAtLeast(const unsigned* p, unsigned target,
                                            unsigned* error) {
  for (unsigned spin = 0;; ++spin) {
    if (__hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP) >=
        target)
      break;
    if (spin > (1U << 20)) {
      atomicOr(error, 1U);
      return false;
    }
    __builtin_amdgcn_s_sleep(1);
  }
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
  return true;
}

template<int BM, int BN, int CWM, int CWN, int PW, int Code, bool Routed,
         int BK, int Ring, bool Flags>
__launch_bounds__((CWM * CWN + PW) * 32) __global__
    void WarpSpecGemm(Args args) {
  constexpr int CW = CWM * CWN, RS = BM / 16, TS = BN / 16;
  constexpr int WRS = RS / CWM, WTS = TS / CWN;
  constexpr int PT = PW * 32;
  constexpr int TPW = (BK * RS) / PW;
  constexpr int PFB = (BN * BK + PT - 1) / PT;
  constexpr int SlotA = BK * RS * 256, SlotB = BK * TS * 256;
  constexpr int SlotShorts = SlotA + SlotB;
  constexpr int RingBytes = Ring * SlotShorts * 2;
  static_assert((BK * RS) % PW == 0 && BM % (16 * CWM) == 0 &&
                BN % (16 * CWN) == 0 && 8 % BK == 0);
  static_assert(Flags || Ring == 2, "barrier mode is double buffered");
  static_assert(RingBytes + (Flags ? 8 * Ring : 0) <= 65536, "LDS");
  __shared__ __attribute__((aligned(16))) short ring[Ring * SlotShorts];

  const int tid = static_cast<int>(threadIdx.x), wave = tid / 32,
            lane = tid % 32, sl = lane & 15, hi = lane / 16;
  const bool producer = wave >= CW;
  const int wr = wave / CWN, wt = wave % CWN;
  const int pwave = wave - CW, ptid = tid - CW * 32;
  int r_block, t_block, batch;
  const std::uint32_t* trellis;
  const uint4* half;
  const short* x;
  void* y;
  const std::int32_t* exps;
  if (!Setup<BM, BN, Routed>(args, r_block, t_block, batch, trellis, half, x, y,
                             exps, Code))
    return;
  const int m = args.m, k = args.k, stages = k / (16 * BK);
  const auto slot_a = [&](int slot, int ks) {
    return ring + slot * SlotShorts + ks * RS * 256;
  };
  const auto slot_b = [&](int slot, int ks) {
    return ring + slot * SlotShorts + SlotA + ks * TS * 256;
  };

  // Producer state.
  const ar::LaneWindow lw = ar::MakeLaneWindow<Code>(lane);
  TrellisRaw traw[TPW];
  uint4 rb[PFB][2];
  const short* x_row[PFB];
#pragma unroll
  for (int p = 0; p < PFB; ++p) {
    // Consumers (ptid < 0) never dereference their rows.
    const int t = t_block + max(0, (p * PT) + ptid) / BK;
    x_row[p] = x + static_cast<std::size_t>(t < batch ? t : batch - 1) * k;
  }
  const auto fetch = [&](int s) {
    const int ks0 = s * BK;
#pragma unroll
    for (int i = 0; i < TPW; ++i) {
      const int t = pwave * TPW + i, ks = t / RS, rt = t % RS;
      const std::uint32_t* tile =
          trellis + (static_cast<std::size_t>(ks0 + ks) * (m / 16) +
                     (r_block / 16) + rt) *
                        ar::TileDwords(Code);
      traw[i] = {tile[lw.o0], tile[lw.o1], tile[lw.o2]};
    }
#pragma unroll
    for (int p = 0; p < PFB; ++p) {
      const int idx = p * PT + ptid, kk = ks0 + idx % BK;
      if constexpr (Routed) {
        if (t_block + ((idx / BK) / 16) * 16 >= batch)
          continue;
      }
      if (idx < BN * BK) {
        const short* src = x_row[p] + static_cast<std::size_t>(kk) * 16;
        rb[p][0] = *reinterpret_cast<const uint4*>(src);
        rb[p][1] = *reinterpret_cast<const uint4*>(src + 8);
      }
    }
  };
  const auto commit = [&](int s, int slot) {
    const int ks0 = s * BK;
#pragma unroll
    for (int i = 0; i < TPW; ++i) {
      const int t = pwave * TPW + i, ks = t / RS, rt = t % RS;
      ar::Half2 p[4];
      DecodeTile<Code>(traw[i], lw, p);
      StoreTile(slot_a(slot, ks), rt, (ks0 + ks) & 3, lane, p);
    }
#pragma unroll
    for (int p = 0; p < PFB; ++p) {
      const int idx = p * PT + ptid, row = idx / BK, ks = idx % BK;
      if constexpr (Routed) {
        if (t_block + (row / 16) * 16 >= batch)
          continue;
      }
      if (idx < BN * BK) {
        const int placed = row ^ ((ks0 + ks) & 3);
        StoreSwizzled(slot_b(slot, ks) + placed * 16, rb[p][0], rb[p][1], row);
      }
    }
  };

  // Consumer state lives only on the consumer path, so the producer path
  // does not carry the accumulators (and vice versa).
  const auto compute = [&](v8f(&acc)[WRS][WTS], int s, int slot) {
#pragma unroll
    for (int ks = 0; ks < BK; ++ks) {
      const unsigned swz = (s * BK + ks) & 3;
      v16h a[WRS];
#pragma unroll
      for (int i = 0; i < WRS; ++i)
        a[i] = LoadSwizzled(slot_a(slot, ks) +
                            ((wr * WRS + i) * 16 + (sl ^ swz)) * 16);
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        if constexpr (Routed) {
          if (t_block + (wt * WTS + j) * 16 >= batch)
            continue;
        }
        const v16h b = LoadSwizzled(slot_b(slot, ks) +
                                    ((wt * WTS + j) * 16 + (sl ^ swz)) * 16);
#pragma unroll
        for (int i = 0; i < WRS; ++i)
          acc[i][j] = WmmaHalf(b, a[i], acc[i][j]);
      }
    }
  };
  // Largest 16-token multiple of tiles per dense epilogue pass in the ring.
  constexpr int TilesPerStage = RingBytes / (16 * BM * 4) >= WTS       ? WTS
                                : RingBytes / (16 * BM * 4) >= WTS / 2 ? WTS / 2
                                                                       : 1;
  static_assert(16 * TilesPerStage * BM * 4 <= RingBytes);
  // Every wave calls this once (the dense epilogue has block barriers).
  const auto epilogue = [&](const v8f(&acc)[WRS][WTS], bool writer) {
    if constexpr (Routed) {
      if (writer)
        StoreRouted<WRS, WTS>(acc, static_cast<_Float16*>(y), exps, m, r_block,
                              t_block, batch, wr, wt, sl, hi);
    } else {
      StoreDense<BM, BN, WRS, WTS, 16 * TilesPerStage, true>(
          acc, reinterpret_cast<float*>(ring), static_cast<float*>(y), args.svh,
          m, r_block, t_block, batch, wr, wt, writer, wave, CW + PW, lane);
    }
  };

  // Barrier counts match on both paths: s_barrier counts waves, not sites.
  [[maybe_unused]] unsigned* full = nullptr;
  [[maybe_unused]] unsigned* empty = nullptr;
  if constexpr (Flags) {
    __shared__ unsigned counters[2 * Ring];
    full = counters;
    empty = counters + Ring;
    if (tid < 2 * Ring)
      counters[tid] = 0;
    __syncthreads();
  }
  if (producer) {
    if constexpr (!Flags) {
      fetch(0);
      commit(0, 0);
      if (stages > 1)
        fetch(1);
      __syncthreads();
      for (int s = 0; s < stages; ++s) {
        if (s + 1 < stages) {
          commit(s + 1, (s + 1) & 1);
          if (s + 2 < stages)
            fetch(s + 2);
        }
        __syncthreads();
      }
    } else {
      // After a timeout (error bit 1) the waits are skipped, not the work:
      // the loops keep a single exit (an early break made the consumer
      // spill its accumulators).
      bool live = true;
      fetch(0);
      for (int s = 0; s < stages; ++s) {
        const int slot = s % Ring;
        if (s >= Ring)
          live = live && WaitAtLeast(&empty[slot], unsigned(CW * (s / Ring)),
                                     args.error);
        commit(s, slot);
        // Lane 0 releases for the wave: the release fence's s_waitcnt
        // lgkmcnt(0) is wave-wide, so every lane's LDS stores are complete
        // (AMDGPU semantics, not the formal C++ model).
        if (lane == 0)
          __hip_atomic_fetch_add(&full[slot], 1U, __ATOMIC_RELEASE,
                                 __HIP_MEMORY_SCOPE_WORKGROUP);
        if (s + 1 < stages)
          fetch(s + 1);
      }
      __syncthreads();
    }
    const v8f none[WRS][WTS] = {};
    epilogue(none, false);
  } else {
    v8f acc[WRS][WTS] = {};
    if constexpr (!Flags) {
      __syncthreads();
      for (int s = 0; s < stages; ++s) {
        compute(acc, s, s & 1);
        __syncthreads();
      }
    } else {
      bool live = true;
      for (int s = 0; s < stages; ++s) {
        const int slot = s % Ring;
        live = live && WaitAtLeast(&full[slot], unsigned(PW * (s / Ring + 1)),
                                   args.error);
        compute(acc, s, slot);
        if (lane == 0)
          __hip_atomic_fetch_add(&empty[slot], 1U, __ATOMIC_RELEASE,
                                 __HIP_MEMORY_SCOPE_WORKGROUP);
      }
      __syncthreads();
    }
    epilogue(acc, true);
  }
}

// ------------------------------------------------------------------ host ----

struct Resources {
  int vgpr = -1, lds = -1, scratch = -1, threads = 0, blocks_per_cu = -1;
};

template<class Kernel>
Resources Query(Kernel kernel, int threads) {
  Resources r;
  hipFuncAttributes attr{};
  if (hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel)) ==
      hipSuccess) {
    r.vgpr = attr.numRegs;
    r.lds = static_cast<int>(attr.sharedSizeBytes);
    r.scratch = static_cast<int>(attr.localSizeBytes);
  }
  r.threads = threads;
  int blocks = 0;
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads,
                                                   0) == hipSuccess)
    r.blocks_per_cu = blocks;
  return r;
}

std::string Describe(const Resources& r) {
  if (r.threads == 0)
    return "(library kernel; equal to a* code)";
  char text[96];
  std::snprintf(text, sizeof(text),
                "%4d thr %3d vgpr %6d lds %3d scr %d blk/CU", r.threads, r.vgpr,
                r.lds, r.scratch, r.blocks_per_cu);
  return text;
}

// Routed: two kernels per launch (64-token and hot 65..128-token buckets),
// as production LaunchBucketTiles; grid (1, n / 128, max_buckets).
struct RoutedVariant {
  const char* name;
  bool exact;  // expected bit-equal to production
  bool needs_half;
  void (*run)(const Args&, unsigned buckets, hipStream_t);
  std::vector<Resources> resources;
};

template<auto K64, int T64, auto K128, int T128>
void RunRouted(const Args& args, unsigned buckets, hipStream_t stream) {
  const unsigned gy = static_cast<unsigned>(args.m) / 128;
  K64<<<dim3(1, gy, buckets), T64, 0, stream>>>(args);
  K128<<<dim3(1, gy, buckets), T128, 0, stream>>>(args);
}

template<auto K64, int T64, auto K128, int T128>
RoutedVariant MakeRouted(const char* name, bool exact, bool needs_half) {
  return {name,
          exact,
          needs_half,
          &RunRouted<K64, T64, K128, T128>,
          {Query(K64, T64), Query(K128, T128)}};
}

template<int Code>
std::vector<RoutedVariant> RoutedVariants() {
  constexpr bool R = true;
  return {
      MakeRouted<StagedGemm<128, 64, 8, 1, Code, R, Src::kTrellis>, 256,
                 StagedGemm<128, 128, 8, 1, Code, R, Src::kTrellis>, 256>(
          "a* copy", true, false),
      MakeRouted<StagedGemm<128, 64, 8, 1, Code, R, Src::kHalf>, 256,
                 StagedGemm<128, 128, 8, 1, Code, R, Src::kHalf>, 256>(
          "b no-decode", true, true),
      MakeRouted<StagedGemm<128, 64, 8, 1, Code, R, Src::kResident>, 256,
                 StagedGemm<128, 128, 8, 1, Code, R, Src::kResident>, 256>(
          "c WMMA-only", false, false),
      MakeRouted<StagedGemm<128, 64, 8, 1, Code, R, Src::kLdsOnly>, 256,
                 StagedGemm<128, 128, 8, 1, Code, R, Src::kLdsOnly>, 256>(
          "c0 LDS-only", false, false),
      MakeRouted<StagedGemm<128, 64, 8, 1, Code, R, Src::kDecodeOnly>, 256,
                 StagedGemm<128, 128, 8, 1, Code, R, Src::kDecodeOnly>, 256>(
          "d decode-only", false, false),
      MakeRouted<WarpSpecGemm<128, 64, 8, 1, 4, Code, R, 4, 2, false>, 384,
                 WarpSpecGemm<128, 128, 8, 1, 4, Code, R, 4, 2, false>, 384>(
          "e-bar 8C+4P r2", true, false),
      MakeRouted<WarpSpecGemm<128, 64, 8, 1, 8, Code, R, 4, 2, false>, 512,
                 WarpSpecGemm<128, 128, 8, 1, 8, Code, R, 4, 2, false>, 512>(
          "e-bar 8C+8P r2", true, false),
      MakeRouted<WarpSpecGemm<128, 64, 8, 1, 4, Code, R, 2, 3, true>, 384,
                 WarpSpecGemm<128, 128, 8, 1, 4, Code, R, 2, 3, true>, 384>(
          "e-flag 8C+4P r3", true, false),
  };
}

struct DenseVariant {
  const char* name;
  bool exact;
  bool needs_half;
  void (*run)(const Args&, hipStream_t);
  Resources resources;
};

template<auto Kernel, int Threads>
void RunDense(const Args& args, hipStream_t stream) {
  Kernel<<<dim3((args.batch + 255) / 256, args.m / 256, 1), Threads, 0,
           stream>>>(args);
}

template<auto Kernel, int Threads>
DenseVariant MakeDense(const char* name, bool exact, bool needs_half) {
  return {name, exact, needs_half, &RunDense<Kernel, Threads>,
          Query(Kernel, Threads)};
}

template<int Code>
std::vector<DenseVariant> DenseVariants() {
  constexpr bool D = false;
  return {
      MakeDense<StagedGemm<256, 256, 8, 4, Code, D, Src::kTrellis>, 1024>(
          "a* copy", true, false),
      MakeDense<StagedGemm<256, 256, 8, 4, Code, D, Src::kHalf>, 1024>(
          "b no-decode", true, true),
      MakeDense<StagedGemm<256, 256, 8, 4, Code, D, Src::kResident>, 1024>(
          "c WMMA-only", false, false),
      MakeDense<StagedGemm<256, 256, 8, 4, Code, D, Src::kLdsOnly>, 1024>(
          "c0 LDS-only", false, false),
      MakeDense<StagedGemm<256, 256, 8, 4, Code, D, Src::kDecodeOnly>, 1024>(
          "d decode-only", false, false),
      MakeDense<WarpSpecGemm<256, 256, 4, 4, 8, Code, D, 2, 2, false>, 768>(
          "e-bar 16C+8P r2", true, false),
      MakeDense<WarpSpecGemm<256, 256, 4, 4, 8, Code, D, 1, 3, true>, 768>(
          "e-flag 16C+8P r3", true, false),
  };
}

void Predecode(int code, const std::uint16_t* trellis, uint4* out,
               std::size_t tiles, hipStream_t stream) {
  const auto* words = reinterpret_cast<const std::uint32_t*>(trellis);
  const dim3 grid(static_cast<unsigned>(
      std::min<std::size_t>((tiles + 7) / 8, 65535 * 16))),
      block(256);
  switch (code) {
    case 4:
      PredecodeKernel<4><<<grid, block, 0, stream>>>(words, out, tiles);
      break;
    case 5:
      PredecodeKernel<5><<<grid, block, 0, stream>>>(words, out, tiles);
      break;
    case 6:
      PredecodeKernel<6><<<grid, block, 0, stream>>>(words, out, tiles);
      break;
    default:
      std::fprintf(stderr, "predecode: unsupported K%d\n", code);
      std::exit(1);
  }
  Ck(hipGetLastError(), "predecode");
}

template<class F>
double TimeUs(hipStream_t stream, F&& body) {
  hipEvent_t a, b;
  Ck(hipEventCreate(&a), "event");
  Ck(hipEventCreate(&b), "event");
  body();
  Ck(hipStreamSynchronize(stream), "warm");
  Ck(hipEventRecord(a, stream), "record");
  body();
  Ck(hipEventRecord(b, stream), "record");
  Ck(hipEventSynchronize(b), "sync");
  float once = 0;
  Ck(hipEventElapsedTime(&once, a, b), "elapsed");
  const int calls = std::clamp(
      static_cast<int>(kTargetRepMs / std::max(once, 0.001F)), 1, 64);
  float best = 1e30F;
  for (int r = 0; r < kReps; ++r) {
    Ck(hipEventRecord(a, stream), "record");
    for (int c = 0; c < calls; ++c)
      body();
    Ck(hipEventRecord(b, stream), "record");
    Ck(hipEventSynchronize(b), "sync");
    float ms = 0;
    Ck(hipEventElapsedTime(&ms, a, b), "elapsed");
    best = std::min(best, ms / calls);
  }
  Ck(hipGetLastError(), "timed launches");
  Ck(hipEventDestroy(a), "destroy");
  Ck(hipEventDestroy(b), "destroy");
  return 1e3 * best;
}

unsigned* g_error = nullptr;
unsigned long long* g_counts = nullptr;
bool g_ok = true;

unsigned TakeError() {
  unsigned e = 0;
  Ck(hipMemcpy(&e, g_error, sizeof(e), hipMemcpyDeviceToHost), "error");
  Ck(hipMemset(g_error, 0, sizeof(e)), "error reset");
  return e;
}

// Bitwise comparison of n words; returns a status string.
template<class Word>
std::string Compare(const Word* ref, const Word* cand, std::size_t n,
                    hipStream_t stream) {
  Ck(hipMemsetAsync(g_counts, 0, 2 * sizeof(unsigned long long), stream),
     "counts");
  CountDiff<Word><<<1024, 256, 0, stream>>>(ref, cand, n, g_counts);
  Ck(hipGetLastError(), "compare");
  unsigned long long counts[2];
  Ck(hipMemcpyAsync(counts, g_counts, sizeof(counts), hipMemcpyDeviceToHost,
                    stream),
     "counts");
  Ck(hipStreamSynchronize(stream), "compare");
  char text[96];
  if (counts[0] == 0 && counts[1] == 0)
    std::snprintf(text, sizeof(text), "EQUAL");
  else
    std::snprintf(text, sizeof(text), "DIFF %llu/%zu unwritten %llu", counts[0],
                  n, counts[1]);
  return text;
}

void PrintRow(const char* name, double us, double flops, double base_us,
              const std::string& status, const std::string& res) {
  std::printf("  %-18s %10.1f us %6.2f TF %5.2fx  %-28s %s\n", name, us,
              flops / (us * 1e6), base_us / us, status.c_str(), res.c_str());
}

void Summary(const char* shape, const char* routing, unsigned rows, int code,
             const char* variant, double us, double flops) {
  std::printf("SUMMARY %s %s rows=%u K%d %s us=%.1f tf=%.3f\n", shape, routing,
              rows, code, variant, us, flops / (us * 1e6));
}

void FillWordsAsync(void* p, std::size_t bytes, std::uint32_t seed,
                    hipStream_t s) {
  FillWords<<<2048, 256, 0, s>>>(static_cast<std::uint32_t*>(p), bytes / 4,
                                 seed);
  Ck(hipGetLastError(), "fill");
}
void FillHalfAsync(void* p, std::size_t n, std::uint32_t seed, float scale,
                   hipStream_t s) {
  FillHalf<<<2048, 256, 0, s>>>(static_cast<_Float16*>(p), n, seed, scale);
  Ck(hipGetLastError(), "fill");
}

// ---------------------------------------------------------------- routed ----

struct Bank {
  std::uint16_t* trellis = nullptr;
  std::uint16_t* scales = nullptr;
  ex::DeviceLinear* device = nullptr;
  uint4* half = nullptr;
  std::size_t expert_tiles = 0;
};

Bank MakeBank(unsigned k, unsigned n, int bits, std::uint32_t seed,
              hipStream_t stream) {
  Bank bank;
  const ex::BitRate rate{bits, false};
  bank.expert_tiles = std::size_t{k / 16} * (n / 16);
  const std::size_t words = bank.expert_tiles * rate.TileWords();
  bank.trellis = Alloc<std::uint16_t>(words * kExperts);
  FillWordsAsync(bank.trellis, words * kExperts * 2, seed, stream);
  bank.scales = Alloc<std::uint16_t>(std::max(k, n));
  FillHalfAsync(bank.scales, std::max(k, n), seed + 1, 0.05F, stream);
  std::vector<ex::DeviceLinear> linears(kExperts);
  for (unsigned e = 0; e < kExperts; ++e)
    linears[e] = {
        bank.trellis + e * words, bank.scales, bank.scales, k, n, rate,
        ex::Codebook::kMul1};
  bank.device = Upload(linears);
  return bank;
}

void FreeBank(Bank& bank) {
  for (void* p :
       {static_cast<void*>(bank.trellis), static_cast<void*>(bank.scales),
        static_cast<void*>(bank.device), static_cast<void*>(bank.half)})
    if (p != nullptr)
      Ck(hipFree(p), "free");
  bank = {};
}

std::vector<std::int32_t> Routing(unsigned tokens, bool skewed,
                                  std::uint32_t seed) {
  std::vector<std::int32_t> ids(std::size_t{tokens} * kUsed);
  for (unsigned t = 0; t < tokens; ++t) {
    for (unsigned s = 0; s < kUsed; ++s) {
      if (!skewed) {
        // Exactly tokens * 10 / 512 rows per expert, distinct per token.
        ids[t * kUsed + s] =
            static_cast<std::int32_t>((std::size_t{t} * kUsed + s) % kExperts);
        continue;
      }
      // Skewed (as flash_next_moe_glue): low ids favoured, u^2.
      for (;;) {
        const float u = static_cast<float>(Next(&seed) % 65536) / 65536.0F;
        const auto e = static_cast<std::int32_t>(u * u * kExperts);
        bool fresh = true;
        for (unsigned p = 0; p < s; ++p)
          fresh = fresh && ids[t * kUsed + p] != e;
        if (fresh) {
          ids[t * kUsed + s] = e;
          break;
        }
      }
    }
  }
  return ids;
}

template<int Code>
void RoutedShape(const char* shape, unsigned k, unsigned n, int projections,
                 const std::vector<unsigned>& per_expert, hipStream_t stream) {
  const ex::BitRate rate{Code, false};
  std::vector<Bank> banks;
  for (int p = 0; p < projections; ++p)
    banks.push_back(MakeBank(
        k, n, Code, 0xB7000000U + 0x1000U * p + Code * 17 + k, stream));
  auto variants = RoutedVariants<Code>();
  // dg: decode every bank to FP16 (b's operand), timed per bank.
  const std::size_t tiles = banks[0].expert_tiles * kExperts;
  for (auto& bank : banks)
    bank.half = Alloc<uint4>(tiles * 32);
  const double dg_us = TimeUs(stream, [&] {
    for (auto& bank : banks)
      Predecode(Code, bank.trellis, bank.half, tiles, stream);
  });
  Ck(hipStreamSynchronize(stream), "predecode");

  for (const unsigned rpe : per_expert) {
    const unsigned tokens = rpe * kExperts / kUsed;
    const unsigned assignments = tokens * kUsed;
    const unsigned max_buckets =
        std::min(assignments, kExperts + assignments / kTileRows);
    const std::size_t in_count = std::size_t{assignments} * k;
    const std::size_t out_count = std::size_t{assignments} * n;
    auto* input = Alloc<_Float16>(in_count);
    FillHalfAsync(input, in_count, 0x1A2B0000U + rpe, 1.0F, stream);
    std::int32_t* exps = nullptr;
    if (projections == 1) {  // down: power-of-two row scaling as production
      exps = Alloc<std::int32_t>(assignments);
      FillExponents<<<256, 256, 0, stream>>>(exps, assignments, rpe);
      Ck(hipGetLastError(), "exponents");
    }
    std::vector<_Float16*> ref(projections), out(projections);
    for (int p = 0; p < projections; ++p) {
      ref[p] = Alloc<_Float16>(out_count);
      out[p] = Alloc<_Float16>(out_count);
    }
    auto* offsets = Alloc<std::uint32_t>(kExperts + 1);
    auto* slots = Alloc<std::int32_t>(assignments);
    auto* inverse = Alloc<std::int32_t>(assignments);
    auto* buckets = Alloc<Bucket>(max_buckets);
    auto* bucket_count = Alloc<std::uint32_t>(1);
    for (const bool skewed : {false, true}) {
      const auto ids = Routing(tokens, skewed, 0x5EED0000U + rpe);
      auto* d_ids = Upload(ids);
      fn::Exl3BuildExpertBuckets(d_ids, assignments, kExperts, kTileRows,
                                 offsets, slots, buckets, bucket_count,
                                 max_buckets, stream, inverse);
      std::uint32_t live = 0;
      Ck(hipMemcpyAsync(&live, bucket_count, sizeof(live),
                        hipMemcpyDeviceToHost, stream),
         "count");
      Ck(hipStreamSynchronize(stream), "buckets");
      std::vector<unsigned> counts(kExperts, 0);
      for (const auto e : ids)
        ++counts[e];
      const unsigned hottest = *std::max_element(counts.begin(), counts.end());
      const double flops = 2.0 * assignments * k * n * projections;
      const char* routing = skewed ? "skewed" : "balanced";
      std::printf(
          "\n%s K%u->N%u x%d K%d, %u rows/expert (%u tokens, %u assignments, "
          "%s: %u buckets, hottest expert %u rows)\n",
          shape, k, n, projections, Code, rpe, tokens, assignments, routing,
          live, hottest);
      const auto production = [&](std::vector<_Float16*>& dst) {
        for (int p = 0; p < projections; ++p)
          gufo::hip::LaunchExl3PrefillBucketInnerHalf(
              banks[p].device, buckets, bucket_count, max_buckets, kTileRows,
              reinterpret_cast<const std::uint16_t*>(input),
              reinterpret_cast<std::uint16_t*>(dst[p]), k, n, rate,
              ex::Codebook::kMul1, exps, stream);
      };
      const auto args_for = [&](int p, _Float16* dst, bool half) {
        Args a{};
        a.w = banks[p].device;
        a.half = half ? banks[p].half : nullptr;
        a.expert_half = banks[p].expert_tiles * 32;
        a.x = reinterpret_cast<const short*>(input);
        a.y = dst;
        a.buckets = buckets;
        a.bucket_count = bucket_count;
        a.row_exponents = exps;
        a.batch = 0;
        a.m = static_cast<int>(n);
        a.k = static_cast<int>(k);
        a.error = g_error;
        return a;
      };
      for (int p = 0; p < projections; ++p)
        Ck(hipMemsetAsync(ref[p], 0xFF, out_count * 2, stream), "sentinel");
      production(ref);
      const double base = TimeUs(stream, [&] { production(ref); });
      PrintRow("a production", base, flops, base, "reference",
               Describe(Resources{}));
      Summary(shape, routing, rpe, Code, "a", base, flops);
      for (const auto& v : variants) {
        const auto launch = [&] {
          for (int p = 0; p < projections; ++p)
            v.run(args_for(p, out[p], v.needs_half), max_buckets, stream);
        };
        std::string status = "n/a (other values)";
        TakeError();
        if (v.exact) {
          for (int p = 0; p < projections; ++p)
            Ck(hipMemsetAsync(out[p], 0xFF, out_count * 2, stream), "sentinel");
          launch();
          Ck(hipGetLastError(), v.name);
          status.clear();
          for (int p = 0; p < projections; ++p) {
            const auto one =
                Compare(reinterpret_cast<const std::uint16_t*>(ref[p]),
                        reinterpret_cast<const std::uint16_t*>(out[p]),
                        out_count, stream);
            status += (p ? " / " : "") + one;
            g_ok = g_ok && one == "EQUAL";
          }
        }
        const double us = TimeUs(stream, launch);
        if (const unsigned e = TakeError(); e != 0) {
          status += " ERROR " + std::to_string(e);
          g_ok = false;
        }
        std::string res = Describe(v.resources[0]) + " | 128-tile " +
                          Describe(v.resources[1]);
        PrintRow(v.name, us, flops, base, status, res);
        Summary(shape, routing, rpe, Code, v.name, us, flops);
      }
      std::printf("  %-18s %10.1f us (whole bank%s to FP16 in global)\n",
                  "dg bank decode", dg_us, projections > 1 ? "s" : "");
      Ck(hipFree(d_ids), "free");
    }
    for (void* p : {static_cast<void*>(input), static_cast<void*>(exps),
                    static_cast<void*>(offsets), static_cast<void*>(slots),
                    static_cast<void*>(inverse), static_cast<void*>(buckets),
                    static_cast<void*>(bucket_count)})
      if (p != nullptr)
        Ck(hipFree(p), "free");
    for (int p = 0; p < projections; ++p) {
      Ck(hipFree(ref[p]), "free");
      Ck(hipFree(out[p]), "free");
    }
  }
  for (auto& bank : banks)
    FreeBank(bank);
}

// ----------------------------------------------------------------- dense ----

template<int Code>
void DenseShape(const char* shape, unsigned k, unsigned n,
                const std::vector<unsigned>& row_counts, hipStream_t stream) {
  const ex::BitRate rate{Code, false};
  const std::size_t tiles = std::size_t{k / 16} * (n / 16);
  auto* trellis = Alloc<std::uint16_t>(tiles * rate.TileWords());
  FillWordsAsync(trellis, tiles * rate.TileWords() * 2, 0xD0000000U + Code + n,
                 stream);
  auto* svh = Alloc<_Float16>(n);
  FillHalfAsync(svh, n, 0xD5000000U + n, 0.05F, stream);
  auto* half = Alloc<uint4>(tiles * 32);
  const double dg_us =
      TimeUs(stream, [&] { Predecode(Code, trellis, half, tiles, stream); });
  auto variants = DenseVariants<Code>();
  for (const unsigned rows : row_counts) {
    const std::size_t in_count = std::size_t{rows} * k;
    const std::size_t out_count = std::size_t{rows} * n;
    auto* x = Alloc<_Float16>(in_count);
    FillHalfAsync(x, in_count, 0xA0000000U + rows, 1.0F, stream);
    auto* ref = Alloc<float>(out_count);
    auto* out = Alloc<float>(out_count);
    const gufo::hip::Exl3GemmProblem problem{
        trellis, reinterpret_cast<const std::uint16_t*>(svh), x, ref, rows, 0};
    auto* problems = Upload(std::vector{problem});
    const double flops = 2.0 * rows * k * n;
    std::printf("\n%s K%u->N%u K%d, %u rows\n", shape, k, n, Code, rows);
    const auto production = [&] {
      gufo::hip::LaunchExl3PrefillGemmGrouped(problems, 1, rows, k, n, rate,
                                              ex::Codebook::kMul1, false,
                                              stream);
    };
    Ck(hipMemsetAsync(ref, 0xFF, out_count * 4, stream), "sentinel");
    production();
    const double base = TimeUs(stream, production);
    PrintRow("a production", base, flops, base, "reference",
             Describe(Resources{}));
    Summary(shape, "dense", rows, Code, "a", base, flops);
    for (const auto& v : variants) {
      Args a{};
      a.w = trellis;
      a.half = v.needs_half ? half : nullptr;
      a.x = reinterpret_cast<const short*>(x);
      a.y = out;
      a.svh = svh;
      a.batch = static_cast<int>(rows);
      a.m = static_cast<int>(n);
      a.k = static_cast<int>(k);
      a.error = g_error;
      std::string status = "n/a (other values)";
      TakeError();
      if (v.exact) {
        Ck(hipMemsetAsync(out, 0xFF, out_count * 4, stream), "sentinel");
        v.run(a, stream);
        Ck(hipGetLastError(), v.name);
        status = Compare(reinterpret_cast<const std::uint32_t*>(ref),
                         reinterpret_cast<const std::uint32_t*>(out), out_count,
                         stream);
        g_ok = g_ok && status == "EQUAL";
      }
      const double us = TimeUs(stream, [&] { v.run(a, stream); });
      if (const unsigned e = TakeError(); e != 0) {
        status += " ERROR " + std::to_string(e);
        g_ok = false;
      }
      PrintRow(v.name, us, flops, base, status, Describe(v.resources));
      Summary(shape, "dense", rows, Code, v.name, us, flops);
    }
    std::printf("  %-18s %10.1f us (whole matrix to FP16 in global)\n",
                "dg decode", dg_us);
    for (void* p : {static_cast<void*>(x), static_cast<void*>(ref),
                    static_cast<void*>(out), static_cast<void*>(problems)})
      Ck(hipFree(p), "free");
  }
  for (void* p : {static_cast<void*>(trellis), static_cast<void*>(svh),
                  static_cast<void*>(half)})
    Ck(hipFree(p), "free");
}

}  // namespace

int main(int argc, char** argv) {
  bool routed = true, dense = true, quick = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "routed")
      dense = false;
    else if (arg == "dense")
      routed = false;
    else if (arg == "quick")
      quick = true;
    else {
      std::fprintf(stderr, "usage: %s [routed|dense] [quick]\n", argv[0]);
      return 2;
    }
  }
  hipStream_t stream;
  Ck(hipStreamCreate(&stream), "stream");
  g_error = Alloc<unsigned>(2);  // [0] error bits, [1] decode-only sink
  Ck(hipMemset(g_error, 0, 2 * sizeof(unsigned)), "error");
  g_counts = Alloc<unsigned long long>(2);
  const std::vector<unsigned> per_expert =
      quick ? std::vector<unsigned>{80}
            : std::vector<unsigned>{40, 80, 160, 320, 640};
  const std::vector<unsigned> rows =
      quick ? std::vector<unsigned>{4096}
            : std::vector<unsigned>{2048, 4096, 8192};
  if (routed) {
    RoutedShape<4>("routed gate/up", kHidden, kFf, 2, per_expert, stream);
    RoutedShape<5>("routed gate/up", kHidden, kFf, 2, per_expert, stream);
    RoutedShape<4>("routed down", kFf, kHidden, 1, per_expert, stream);
    RoutedShape<5>("routed down", kFf, kHidden, 1, per_expert, stream);
  }
  if (dense) {
    DenseShape<5>("dense GDN qkv", 2560, 10240, rows, stream);
    DenseShape<6>("dense GDN qkv", 2560, 10240, rows, stream);
    DenseShape<6>("dense GDN out", 6144, 2560, rows, stream);
    DenseShape<5>("dense GDN out", 6144, 2560, rows, stream);
    DenseShape<5>("dense z/q", 2560, 6144, rows, stream);
    DenseShape<6>("dense z/q", 2560, 6144, rows, stream);
  }
  std::printf("\n%s\n", g_ok ? "PASS" : "FAIL");
  return g_ok ? 0 : 1;
}
