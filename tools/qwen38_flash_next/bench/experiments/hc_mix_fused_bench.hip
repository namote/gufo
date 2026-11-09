// U4-5 screen (plan/upstream-sync/PLAN.md): upstream gufo eb1d52e6 (#894)
// HcMixQ8Fused, a decode batch's HC combine, Q8_0 down projection, SiLU
// scale, Q8_0 up projection and mix epilogue in one work-list launch,
// copied here as a candidate (not production) against this tree's
// production decode sequence: HcCombineQ8_1, HcDownSiluQ8_1 (down + SiLU +
// lo Q8_1 in one launch, B4 H3), the Q8_0 up GEMV and HcMixEpilogue.
// Production's EXL3 epilogue also rotates `mixed` into the next
// projections' input slabs (Exl3InputSlabs, 2-3 consumers); neither variant
// does here. An integrated candidate would bring back a separate rotation
// launch, so the fused speedup printed here is an upper bound. Both are
// replayed from hipGraphs of `mixers` consecutive mixers whose weights
// rotate through `copies` sets (about 7 MB each) past the 32 MiB MALL.
// Prints, per token count 1/2/4/8, the median us per mixer of each and
// whether the candidate's res, xn, lo, mixed, mixed Q8_1 and inject bytes
// equal production's (they need not: the candidate repeats upstream's
// kernels of November 1, not this tree's, except the down tile's
// rounding, aligned to this tree).
//
//   qwen38_flash_next_hc_mix_fused_bench [mixers] [copies] [replays]
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "qfn_mmq.h"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm::candidate {

/// Operands of HcMixQ8Fused (upstream kernels.hpp of eb1d52e6).
struct HcFusedMix {
  const void* down;
  const void* up;
  float* res;
  const float* block_out;
  const float* inject;
  std::uint32_t inject_parts;
  const float* gamma;
  float* xn;
  void* xn_q8;
  float* lo;
  void* lo_q8;
  std::uint32_t lo_q8_cols;
  const float* inject_w;
  float* mixed;
  void* mixed_q8;
  float* inject_out;
  std::uint32_t hidden;
  std::uint32_t streams;
  std::uint32_t rank;
  float eps;
  float scale;
};

// This tree's HcCombine inject sum (hc.hip.cpp), which the candidate's
// combine phase calls by name.

/// Up to kMaxParts inject partials held in registers: every load issues
/// before the first add (indices clamped, so none waits behind a branch),
/// and Sum adds parts[0] + ... + parts[n - 1] in order from zero.
struct InjectParts {
  static constexpr std::uint32_t kMaxParts = 16;
  const float* parts;
  std::uint32_t n;
  float v[kMaxParts];

  __device__ __forceinline__ InjectParts(const float* p, std::uint32_t count)
      : parts(p), n(count) {
    if (n - 1 < kMaxParts) {
#pragma unroll
      for (std::uint32_t i = 0; i < kMaxParts; ++i) {
        v[i] = parts[i < n ? i : n - 1];
      }
    }
  }

  __device__ __forceinline__ float Sum() const {
    float total = Materialize(0.0f);
    if (n - 1 >= kMaxParts) {
      for (std::uint32_t i = 0; i < n; ++i) {
        total += parts[i];
      }
      return total;
    }
#pragma unroll
    for (std::uint32_t i = 0; i < kMaxParts; ++i) {
      if (i < n) {
        total = Materialize(total + v[i]);
      }
    }
    return total;
  }
};

// ---- upstream eb1d52e6 kernels.hip.cpp additions; only the down tile's
// single-token rounding is aligned to this tree (see below) ----
// A decode batch's mixer in one launch (HcMixQ8Fused). Its five kernels,
// HcCombineKernel with Q8_1 rows, qfn_q8_hc_down_kernel, SiluScaleQ8Kernel,
// mul_mat_vec_q8 and HcMixEpilogueKernel<4>, each cost a replayed graph
// ~1.9 us of idle gap. Here every phase is a list of items the running
// blocks take from a counter, starting a phase once the previous one's
// done count is complete. Only a running block ever holds an item, so the
// launch neither needs co-resident blocks nor can deadlock behind other
// work, and the last block to exit zeroes every counter for the next
// launch or graph replay. Each item repeats its kernel's lane mapping,
// rounded products, FMA operands and reduction order.
constexpr std::uint32_t kFusedHidden = 2560;
constexpr std::uint32_t kFusedStreams = 4;
constexpr std::uint32_t kFusedRank = 320;
constexpr std::uint32_t kFusedDim = kFusedStreams * kFusedHidden;
constexpr std::uint32_t kFusedRankBlocks = kFusedRank / 32;
constexpr std::uint32_t kFusedDimBlocks = kFusedDim / 32;
constexpr std::uint32_t kFusedWaves = kThreads / 32;
/// Down tiles of one row per wave, and mix units of one 32-element block of
/// all four streams (an epilogue wave of one inject part).
constexpr std::uint32_t kFusedDownTiles = kFusedRank / kFusedWaves;
constexpr std::uint32_t kFusedUnits = kFusedHidden / 32;
constexpr std::uint32_t kFusedParts = kFusedHidden / kThreads;
constexpr std::uint32_t kFusedMaxTokens = 8;

struct HcFusedState {
  unsigned taken[3];  ///< items taken: combine rows, down tiles, mix units
  unsigned done[2];   ///< combine rows and down tiles finished
  unsigned exited;    ///< blocks finished
  unsigned parts[kFusedParts];  ///< mix units finished per inject part
  float down[kFusedMaxTokens * kFusedRank];  ///< down projection outputs
  /// Epilogue wave sums [t][o][unit], totalled per part of eight units.
  float dots[kFusedMaxTokens * kFusedStreams * kFusedUnits];
};

/// The next item of a phase, for the whole block.
__device__ __forceinline__ unsigned FusedTake(unsigned* counter,
                                              unsigned* item) {
  __syncthreads();  // readers of the previous item
  if (threadIdx.x == 0) {
    *item = atomicAdd(counter, 1u);
  }
  __syncthreads();
  return *item;
}

/// Publishes the block's writes, then counts its item done.
__device__ __forceinline__ void FusedDone(unsigned* counter) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    __hip_atomic_fetch_add(counter, 1u, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_AGENT);
  }
}

/// Waits for `count` done items, then reads their writes from L2.
__device__ __forceinline__ void FusedWait(unsigned* counter, unsigned count) {
  if (threadIdx.x == 0) {
    while (__hip_atomic_load(counter, __ATOMIC_ACQUIRE,
                             __HIP_MEMORY_SCOPE_AGENT) < count) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

/// Zeroes a counter no other block of this launch uses any more.
__device__ __forceinline__ void FusedClear(unsigned* counter) {
  __hip_atomic_store(counter, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

/// mmq's scatter_reduce_sum (the warp_reduce_sum<32> pairs and operands;
/// lane l finishes sum l >> (5 - levels)).
template<int kOffset, int kCount>
__device__ __forceinline__ void FusedScatterLevel(float* s, int lane) {
  const bool upper = (lane & kOffset) != 0;
#pragma unroll
  for (int i = 0; i < kCount / 2; ++i) {
    const float keep = upper ? s[kCount / 2 + i] : s[i];
    const float send = upper ? s[i] : s[kCount / 2 + i];
    s[i] = keep + XorLane(send, kOffset);
  }
}

constexpr int FusedScatterLevels(int n) {
  return n <= 1 ? 0 : n <= 2 ? 1 : n <= 4 ? 2 : 3;
}

template<int kN>
__device__ __forceinline__ float FusedScatterSum(const float (&v)[kN],
                                                 int lane) {
  constexpr int kLevels = FusedScatterLevels(kN);
  float s[1 << kLevels];
#pragma unroll
  for (int j = 0; j < (1 << kLevels); ++j) {
    s[j] = j < kN ? v[j] : 0.0f;
  }
  if constexpr (kLevels == 3) {
    FusedScatterLevel<16, 8>(s, lane);
    FusedScatterLevel<8, 4>(s, lane);
    FusedScatterLevel<4, 2>(s, lane);
  } else if constexpr (kLevels == 2) {
    FusedScatterLevel<16, 4>(s, lane);
    FusedScatterLevel<8, 2>(s, lane);
    s[0] += XorLane(s[0], 4);
  } else {
    FusedScatterLevel<16, 2>(s, lane);
    s[0] += XorLane(s[0], 8);
    s[0] += XorLane(s[0], 4);
  }
  s[0] += XorLane(s[0], 2);
  s[0] += XorLane(s[0], 1);
  return s[0];
}

/// Two consecutive ints of a Q8 block's codes (get_int_b2/b4).
__device__ __forceinline__ void FusedCodes(const std::int8_t* qs, int i,
                                           int (&v)[2]) {
  __builtin_memcpy(v, qs + 4 * i, sizeof(v));
}

template<int kTokens>
__launch_bounds__(kThreads) __global__
    void HcMixQ8FusedKernel(HcFusedMix a, HcFusedState* st) {
  __shared__ float shared[32];
  __shared__ unsigned item;
  __shared__ unsigned last;
  __shared__ Q8_1Block lo_s[kTokens * kFusedRankBlocks];
  __shared__ float gate_s[kTokens][kFusedStreams][32];
  const auto* down = static_cast<const Q8_0Block*>(a.down);
  const auto* up = static_cast<const Q8_0Block*>(a.up);
  const int lane = static_cast<int>(threadIdx.x % 32);
  const unsigned wave = threadIdx.x / 32;
  constexpr unsigned kRows = kTokens * kFusedStreams;

  // HcCombineKernel, grid (t, s), for rows of up to kHeld * kThreads.
  for (unsigned u; (u = FusedTake(&st->taken[0], &item)) < kRows;) {
    const std::uint32_t hidden = a.hidden;
    const std::uint32_t streams = a.streams;
    const std::uint32_t t = u / kFusedStreams;
    const std::uint32_t s = u % kFusedStreams;
    const float* parts =
        a.inject + (static_cast<std::size_t>(t) * streams + s) * a.inject_parts;
    const std::size_t base =
        (static_cast<std::size_t>(t) * streams + s) * hidden;
    float* dst = a.res + base;
    const float* src = a.block_out + static_cast<std::size_t>(t) * hidden;
    const float* g = a.gamma + static_cast<std::size_t>(s) * hidden;
    constexpr std::uint32_t kHeld = 10;
    const InjectParts inject_sum(parts, a.inject_parts);
    float r[kHeld];
    float b[kHeld];
    float gv[kHeld];
#pragma unroll
    for (std::uint32_t c = 0; c < kHeld; ++c) {
      const std::uint32_t i = threadIdx.x + c * blockDim.x;
      const std::uint32_t k = i < hidden ? i : hidden - 1;
      r[c] = dst[k];
      b[c] = src[k];
      gv[c] = g[k];
    }
    const float w =
        2.0f * SigmoidF(inject_sum.Sum() / static_cast<float>(streams));
    float ss = 0.0f;
#pragma unroll
    for (std::uint32_t c = 0; c < kHeld; ++c) {
      const std::uint32_t i = threadIdx.x + c * blockDim.x;
      if (i < hidden) {
        r[c] = r[c] + b[c] * w;
        dst[i] = r[c];
        ss = Materialize(ss + r[c] * r[c]);
      }
    }
    ss = BlockSum(ss, shared);
    const float scale = rsqrtf(ss / static_cast<float>(hidden) + a.eps);
#pragma unroll
    for (std::uint32_t c = 0; c < kHeld; ++c) {
      const std::uint32_t i = threadIdx.x + c * blockDim.x;
      if (i < hidden) {
        const float v = r[c] * scale * gv[c];
        a.xn[base + i] = v;
        QuantizeQ8_1Lane(v, a.xn_q8, base + i);
      }
    }
    FusedDone(&st->done[0]);
  }

  // qfn_q8_hc_down_kernel<kTokens>: one row per wave, two iterations of
  // loads ahead, the MMVQ products and FMA chain.
  for (unsigned tile;
       (tile = FusedTake(&st->taken[1], &item)) < kFusedDownTiles;) {
    FusedWait(&st->done[0], kRows);
    constexpr int kPrefetch = 2;
    const unsigned row = tile * kFusedWaves + wave;
    const auto* input = static_cast<const Q8_1Block*>(a.xn_q8);
    const int part = (lane & 3) * 2;
    float sum[kTokens] = {};
    for (int first = lane / 4; first < static_cast<int>(kFusedDimBlocks);
         first += 8 * kPrefetch) {
      int v[kPrefetch][2], u[kPrefetch][kTokens][2];
      __half dw[kPrefetch], dx[kPrefetch][kTokens];
#pragma unroll
      for (int j = 0; j < kPrefetch; ++j) {
        const int kb = first + 8 * j;
        const Q8_0Block& wb = down[row * kFusedDimBlocks + kb];
        dw[j] = wb.d;
        FusedCodes(wb.qs, part, v[j]);
#pragma unroll
        for (int t = 0; t < kTokens; ++t) {
          const Q8_1Block& x = input[t * kFusedDimBlocks + kb];
          dx[j][t] = __low2half(x.ds);
          FusedCodes(x.qs, part, u[j][t]);
        }
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int j = 0; j < kPrefetch; ++j) {
#pragma unroll
        for (int t = 0; t < kTokens; ++t) {
          int dot = __builtin_amdgcn_sudot4(true, v[j][0], true, u[j][t][0], 0,
                                            false);
          dot = __builtin_amdgcn_sudot4(true, v[j][1], true, u[j][t][1], dot,
                                        false);
          // Aligned to this tree: one rounding sequence for every width
          // (upstream's kTokens == 1 branch was removed in w8a8.hip.cpp).
          const float product =
              __fmul_rn(__half2float(dw[j]), static_cast<float>(dot));
          sum[t] = __fmaf_rn(__half2float(dx[j][t]), product, sum[t]);
        }
      }
    }
#pragma unroll
    for (int t = 0; t < kTokens; ++t) {
      sum[t] = WaveSum(sum[t]);
      if (lane == 0) {
        st->down[t * kFusedRank + row] = sum[t];
      }
    }
    FusedDone(&st->done[1]);
  }

  // SiluScaleQ8Kernel into LDS (and, for the first unit, the F32 and Q8_1
  // rows), then mul_mat_vec_q8 over the unit's 4 x 32 up rows and the
  // unit's HcMixEpilogueKernel<4> wave. Once the combine is done, a unit
  // loads its first weights, inject weights and norm while the down tiles
  // run (not before: the combine's few reads would queue behind them).
  constexpr int kAhead = kTokens <= 4 ? 16 : 8;
  bool lo_ready = false;
  for (unsigned unit; (unit = FusedTake(&st->taken[2], &item)) < kFusedUnits;) {
    // Four lanes per Q8 block; lanes 0-7 also take the row's last two.
    const int kb0 = lane / 4;
    const int kqs = (lane % 4) * 2;
    const bool second = lane < 8;
    const int kb1 = second ? kb0 + 8 : kb0;
    const unsigned s = wave / 2;
    const unsigned e0 = (wave % 2) * 16;
    const Q8_0Block* rows =
        up +
        (std::size_t{s} * kFusedHidden + unit * 32 + e0) * kFusedRankBlocks;
    int v0[kAhead][2], v1[kAhead][2];
    __half w0[kAhead], w1[kAhead];
    const auto load = [&](int r0) {
#pragma unroll
      for (int r = 0; r < kAhead; ++r) {
        const Q8_0Block* wr = rows + (r0 + r) * kFusedRankBlocks;
        FusedCodes(wr[kb0].qs, kqs, v0[r]);
        FusedCodes(wr[kb1].qs, kqs, v1[r]);
        w0[r] = wr[kb0].d;
        w1[r] = wr[kb1].d;
      }
    };
    FusedWait(&st->done[0], kRows);
    load(0);
    // The epilogue's operands: wave t's norm and inject weights.
    const std::uint32_t hidden = a.hidden;
    const std::uint32_t i = unit * 32 + lane;
    float v[kFusedStreams];
    float wv[kFusedStreams][kFusedStreams];
    if (wave < kTokens) {
      const float* x = a.xn + static_cast<std::size_t>(wave) * kFusedDim;
#pragma unroll
      for (std::uint32_t q = 0; q < kFusedStreams; ++q) {
        v[q] = x[static_cast<std::size_t>(q) * hidden + i];
      }
#pragma unroll
      for (std::uint32_t o = 0; o < kFusedStreams; ++o) {
#pragma unroll
        for (std::uint32_t q = 0; q < kFusedStreams; ++q) {
          wv[o][q] = a.inject_w != nullptr
                         ? a.inject_w[o * kFusedDim +
                                      static_cast<std::size_t>(q) * hidden + i]
                         : 0.0f;
        }
      }
    }
    if (!lo_ready) {
      FusedWait(&st->done[1], kFusedDownTiles);
      // Whole 32-lane groups, which stay within a 320-wide row.
      for (std::uint32_t k = threadIdx.x; k < kTokens * kFusedRank;
           k += kThreads) {
        const float lo = SiluF(st->down[k] * a.scale);
        QuantizeQ8_1Lane(lo, lo_s, k);
        if (unit == 0) {
          a.lo[k] = lo;
          QuantizeQ8_1Lane(lo, a.lo_q8,
                           (k / kFusedRank) * a.lo_q8_cols + k % kFusedRank);
        }
      }
      __syncthreads();
      lo_ready = true;
    }
    int u0[kTokens][2], u1[kTokens][2];
    float d10[kTokens], d11[kTokens];
#pragma unroll
    for (int t = 0; t < kTokens; ++t) {
      const Q8_1Block& x0 = lo_s[t * kFusedRankBlocks + kb0];
      const Q8_1Block& x1 = lo_s[t * kFusedRankBlocks + kb1];
      FusedCodes(x0.qs, kqs, u0[t]);
      FusedCodes(x1.qs, kqs, u1[t]);
      d10[t] = __low2float(x0.ds);
      d11[t] = __low2float(x1.ds);
    }
    const float zero = Materialize(0.0f);
#pragma unroll
    for (int r0 = 0; r0 < 16; r0 += kAhead) {
      if (r0 != 0) {
        load(r0);
      }
#pragma unroll
      for (int r = 0; r < kAhead; ++r) {
        float sum[kTokens];
#pragma unroll
        for (int t = 0; t < kTokens; ++t) {
          int dot =
              __builtin_amdgcn_sudot4(true, v0[r][0], true, u0[t][0], 0, false);
          dot = __builtin_amdgcn_sudot4(true, v0[r][1], true, u0[t][1], dot,
                                        false);
          sum[t] =
              __fmaf_rn(__fmul_rn(__half2float(w0[r]), static_cast<float>(dot)),
                        d10[t], zero);
          if (second) {
            dot = __builtin_amdgcn_sudot4(true, v1[r][0], true, u1[t][0], 0,
                                          false);
            dot = __builtin_amdgcn_sudot4(true, v1[r][1], true, u1[t][1], dot,
                                          false);
            sum[t] = __fmaf_rn(
                __fmul_rn(__half2float(w1[r]), static_cast<float>(dot)), d11[t],
                sum[t]);
          }
        }
        if constexpr (kTokens == 1) {
          const float value = WaveSum(sum[0]);
          if (lane == 0) {
            gate_s[0][s][e0 + r0 + r] = value;
          }
        } else {
          constexpr int kShift = 5 - FusedScatterLevels(kTokens);
          const float value = FusedScatterSum(sum, lane);
          const int t = lane >> kShift;
          if ((lane & ((1 << kShift) - 1)) == 0 && t < kTokens) {
            gate_s[t][s][e0 + r0 + r] = value;
          }
        }
      }
    }
    __syncthreads();
    if (wave < kTokens) {
      const std::uint32_t t = wave;
      float gv[kFusedStreams];
#pragma unroll
      for (std::uint32_t q = 0; q < kFusedStreams; ++q) {
        gv[q] = gate_s[t][q][lane];
      }
      float acc = 0.0f;
#pragma unroll
      for (std::uint32_t q = 0; q < kFusedStreams; ++q) {
        acc = Materialize(acc + v[q] * SigmoidF(gv[q]));
      }
      const float out = acc / static_cast<float>(kFusedStreams);
      a.mixed[static_cast<std::size_t>(t) * hidden + i] = out;
      if (a.mixed_q8 != nullptr) {
        QuantizeQ8_1Lane(out, a.mixed_q8,
                         static_cast<std::size_t>(t) * hidden + i);
      }
      if (a.inject_w != nullptr) {
#pragma unroll
        for (std::uint32_t o = 0; o < kFusedStreams; ++o) {
          float dot = 0.0f;
#pragma unroll
          for (std::uint32_t q = 0; q < kFusedStreams; ++q) {
            dot = Materialize(dot + wv[o][q] * v[q]);
          }
          dot = WaveSum(dot);
          if (lane == 0) {
            st->dots[(t * kFusedStreams + o) * kFusedUnits + unit] = dot;
          }
        }
      }
    }
    if (a.inject_w == nullptr) {
      continue;
    }
    // The part's last unit totals its eight waves in order, as the
    // epilogue block's thread per logit does.
    const unsigned p = unit / kFusedWaves;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      last =
          __hip_atomic_fetch_add(&st->parts[p], 1u, __ATOMIC_ACQ_REL,
                                 __HIP_MEMORY_SCOPE_AGENT) == kFusedWaves - 1;
    }
    __syncthreads();
    if (last) {
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
      if (threadIdx.x < kRows) {
        const float* dots =
            st->dots + threadIdx.x * kFusedUnits + p * kFusedWaves;
        float total = Materialize(0.0f);
        for (std::uint32_t q = 0; q < kFusedWaves; ++q) {
          total = Materialize(total + dots[q]);
        }
        a.inject_out[threadIdx.x * kFusedParts + p] = total;
      }
      if (threadIdx.x == 0) {
        FusedClear(&st->parts[p]);
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0 &&
      __hip_atomic_fetch_add(&st->exited, 1u, __ATOMIC_ACQ_REL,
                             __HIP_MEMORY_SCOPE_AGENT) == gridDim.x - 1) {
    for (unsigned& counter : st->taken) {
      FusedClear(&counter);
    }
    for (unsigned& counter : st->done) {
      FusedClear(&counter);
    }
    FusedClear(&st->exited);
  }
}

std::size_t HcMixQ8FusedStateBytes() {
  return sizeof(HcFusedState);
}

template<int kTokens>
static void LaunchHcMixQ8Fused(const HcFusedMix& mix, HcFusedState* state,
                               std::uint32_t n_tokens, hipStream_t stream) {
  if (n_tokens == kTokens) {
    // A block per down tile and mix unit, so every unit can load its
    // weights early; the first blocks take the combine rows before units.
    hipLaunchKernelGGL(HcMixQ8FusedKernel<kTokens>,
                       dim3(kFusedDownTiles + kFusedUnits), dim3(kThreads), 0,
                       stream, mix, state);
  } else if constexpr (kTokens < static_cast<int>(kFusedMaxTokens)) {
    LaunchHcMixQ8Fused<kTokens + 1>(mix, state, n_tokens, stream);
  }
}

bool HcMixQ8Fused(void* state, const HcFusedMix& mix, std::uint32_t n_tokens,
                  hipStream_t stream) {
  if (state == nullptr || n_tokens == 0 || n_tokens > kFusedMaxTokens ||
      mix.hidden != kFusedHidden || mix.streams != kFusedStreams ||
      mix.rank != kFusedRank || mix.gamma == nullptr ||
      mix.lo_q8_cols < kFusedRank || mix.lo_q8_cols % 32 != 0 ||
      (mix.inject_w != nullptr && mix.inject_out == nullptr)) {
    return false;
  }
  LaunchHcMixQ8Fused<1>(mix, static_cast<HcFusedState*>(state), n_tokens,
                        stream);
  return true;
}


}  // namespace gufo::models::qwen38_flash_next::rocm::candidate

namespace {

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace cand = gufo::models::qwen38_flash_next::rocm::candidate;

constexpr std::uint32_t kHidden = 2560;
constexpr std::uint32_t kStreams = 4;
constexpr std::uint32_t kRank = 320;
constexpr std::uint32_t kDim = kHidden * kStreams;
constexpr std::uint32_t kMaxTokens = 8;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

std::uint32_t Next(std::uint32_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

void* Device(std::size_t bytes) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, bytes), "hipMalloc");
  CheckHip(hipMemset(p, 0, bytes), "hipMemset");
  return p;
}

void* Q8Weights(std::size_t blocks, std::uint32_t seed) {
  std::vector<std::uint8_t> host(blocks * 34);
  for (std::size_t b = 0; b < blocks; ++b) {
    const __half d =
        __float2half_rn(static_cast<float>(1 + Next(&seed) % 13) / 2048.0F);
    std::memcpy(host.data() + b * 34, &d, sizeof(d));
    for (std::size_t j = 0; j < 32; ++j)
      host[b * 34 + 2 + j] =
          static_cast<std::uint8_t>(static_cast<int>(Next(&seed) % 255) - 127);
  }
  void* p = Device(host.size());
  CheckHip(hipMemcpy(p, host.data(), host.size(), hipMemcpyHostToDevice),
           "weights");
  return p;
}

std::vector<float> Host(std::size_t count, std::uint32_t seed, float scale,
                        float offset = 0.0F) {
  std::vector<float> host(count);
  for (float& v : host)
    v = offset +
        scale *
            static_cast<float>(static_cast<int>(Next(&seed) & 0xFFFF) - 32768) /
            32768.0F;
  return host;
}

float* Upload(const std::vector<float>& host) {
  auto* p = static_cast<float*>(Device(host.size() * sizeof(float)));
  CheckHip(hipMemcpy(p, host.data(), host.size() * sizeof(float),
                     hipMemcpyHostToDevice),
           "values");
  return p;
}

std::vector<std::uint8_t> Bytes(const void* p, std::size_t bytes) {
  std::vector<std::uint8_t> host(bytes);
  CheckHip(hipMemcpy(host.data(), p, bytes, hipMemcpyDeviceToHost), "read");
  return host;
}

double MedianMicros(hipGraphExec_t exec, hipStream_t stream, int replays,
                    int mixers) {
  hipEvent_t start, stop;
  CheckHip(hipEventCreate(&start), "event");
  CheckHip(hipEventCreate(&stop), "event");
  for (int i = 0; i < 3; ++i)
    CheckHip(hipGraphLaunch(exec, stream), "warm replay");
  std::vector<double> times;
  for (int i = 0; i < replays; ++i) {
    CheckHip(hipEventRecord(start, stream), "record");
    CheckHip(hipGraphLaunch(exec, stream), "replay");
    CheckHip(hipEventRecord(stop, stream), "record");
    CheckHip(hipEventSynchronize(stop), "sync");
    float ms = 0.0F;
    CheckHip(hipEventElapsedTime(&ms, start, stop), "elapsed");
    times.push_back(1000.0 * ms / mixers);
  }
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

hipGraphExec_t Capture(hipStream_t stream, const std::function<void()>& body) {
  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  CheckHip(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
           "capture");
  body();
  CheckHip(hipStreamEndCapture(stream, &graph), "capture end");
  CheckHip(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
           "instantiate");
  (void)hipGraphDestroy(graph);
  return exec;
}

/// One variant's buffers, so both start from the same inputs.
struct Buffers {
  float* res;
  float* block_out;
  float* inject;
  float* xn;
  void* xn_q8;
  float* lo;
  void* lo_q8;
  float* gate;
  float* mixed;
  void* mixed_q8;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const int mixers = std::max(1, argc > 1 ? std::atoi(argv[1]) : 48);
    const int copies = std::max(1, argc > 2 ? std::atoi(argv[2]) : 16);
    const int replays = std::max(1, argc > 3 ? std::atoi(argv[3]) : 21);
    hipStream_t stream = nullptr;
    CheckHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    CheckHip(q::PrepareHcDecodeKernels(), "prepare HC decode kernels");
    std::vector<void*> down(copies), up(copies);
    for (int i = 0; i < copies; ++i) {
      down[i] = Q8Weights(std::size_t{kRank} * kDim / 32, 0x1000U + i);
      up[i] = Q8Weights(std::size_t{kDim} * kRank / 32, 0x2000U + i);
    }
    const std::uint32_t parts = q::HcInjectParts(kHidden);
    const auto lo_cols =
        static_cast<std::uint32_t>(qfn_mmq_q8_1_bytes(1, kRank) / 36 * 32);
    const auto res0 = Host(std::size_t{kMaxTokens} * kDim, 1, 1.0F);
    const auto block0 = Host(std::size_t{kMaxTokens} * kHidden, 2, 0.01F);
    const auto inject0 = Host(std::size_t{kMaxTokens} * kStreams * parts, 3,
                              0.05F);
    float* gamma = Upload(Host(kDim, 4, 0.25F, 1.0F));
    float* inject_w = Upload(Host(std::size_t{kStreams} * kDim, 5, 0.02F));
    const auto make = [&] {
      return Buffers{
          Upload(res0),
          Upload(block0),
          Upload(inject0),
          static_cast<float*>(Device(sizeof(float) * kMaxTokens * kDim)),
          Device(qfn_mmq_q8_1_bytes(kMaxTokens, kDim)),
          static_cast<float*>(Device(sizeof(float) * kMaxTokens * kRank)),
          Device(qfn_mmq_q8_1_bytes(kMaxTokens, kRank)),
          static_cast<float*>(Device(sizeof(float) * kMaxTokens * kDim)),
          static_cast<float*>(Device(sizeof(float) * kMaxTokens * kHidden)),
          Device(qfn_mmq_q8_1_bytes(kMaxTokens, kHidden))};
    };
    const auto reset = [&](const Buffers& b) {
      CheckHip(hipMemcpy(b.res, res0.data(), res0.size() * sizeof(float),
                         hipMemcpyHostToDevice),
               "reset res");
      CheckHip(hipMemcpy(b.inject, inject0.data(),
                         inject0.size() * sizeof(float), hipMemcpyHostToDevice),
               "reset inject");
    };
    Buffers prod = make();
    Buffers fused = make();
    void* state = Device(cand::HcMixQ8FusedStateBytes());
    // Production's decode mixer (executor_hc.cpp, Q8_0 HC sidecar route).
    const auto production = [&](std::uint32_t n, int c) {
      if (!q::HcCombineQ8_1(prod.res, prod.block_out, prod.inject, parts,
                            gamma, prod.xn, prod.xn_q8, n, kHidden, kStreams,
                            1e-6F, stream) ||
          !q::HcDownSiluQ8_1(down[c], prod.xn_q8, prod.lo, 1.0F / kStreams, n,
                             lo_cols, prod.lo_q8, stream) ||
          qfn_mmq_q8_0_dense_vec_preq(up[c], nullptr, prod.lo_q8, prod.gate,
                                      kDim, static_cast<int>(n), kRank,
                                      stream) != 0)
        throw std::runtime_error("production mixer rejected");
      // HcMixEpilogue always launches; its result only reports whether it
      // also rotated `mixed` into EXL3 input slabs, which this bench omits.
      q::HcMixEpilogue(prod.xn, prod.gate, inject_w, prod.mixed, prod.inject,
                       n, kHidden, kStreams, stream, prod.mixed_q8);
      CheckHip(hipGetLastError(), "production epilogue");
    };
    const auto candidate = [&](std::uint32_t n, int c) {
      const cand::HcFusedMix mix{
          down[c],     up[c],         fused.res, fused.block_out,
          fused.inject, parts,        gamma,     fused.xn,
          fused.xn_q8, fused.lo,      fused.lo_q8, lo_cols,
          inject_w,    fused.mixed,   fused.mixed_q8, fused.inject,
          kHidden,     kStreams,      kRank,     1e-6F,
          1.0F / kStreams};
      if (!cand::HcMixQ8Fused(state, mix, n, stream))
        throw std::runtime_error("fused mixer rejected");
      CheckHip(hipGetLastError(), "fused mixer");
    };
    std::printf("tokens  production_us  fused_us  speedup  bytes\n");
    for (const std::uint32_t n : {1U, 2U, 4U, 8U}) {
      // One eager mixer of each from identical inputs, compared byte for byte.
      reset(prod);
      reset(fused);
      production(n, 0);
      candidate(n, 0);
      CheckHip(hipStreamSynchronize(stream), "eager mixers");
      const auto same = [&](const void* a, const void* b, std::size_t bytes) {
        return Bytes(a, bytes) == Bytes(b, bytes);
      };
      // Largest absolute difference of an F32 output, to tell rounding from
      // a layout or ordering bug.
      const auto max_diff = [&](const float* a, const float* b,
                                std::size_t count) {
        const auto x = Bytes(a, count * sizeof(float));
        const auto y = Bytes(b, count * sizeof(float));
        double worst = 0;
        for (std::size_t i = 0; i < count; ++i) {
          float u, v;
          std::memcpy(&u, x.data() + 4 * i, 4);
          std::memcpy(&v, y.data() + 4 * i, 4);
          worst = std::max(worst, static_cast<double>(std::abs(u - v)));
        }
        return worst;
      };
      const bool equal =
          same(prod.res, fused.res, sizeof(float) * n * kDim) &&
          same(prod.xn, fused.xn, sizeof(float) * n * kDim) &&
          same(prod.xn_q8, fused.xn_q8, qfn_mmq_q8_1_bytes(n, kDim)) &&
          same(prod.lo_q8, fused.lo_q8, std::size_t{n} * lo_cols / 32 * 36) &&
          same(prod.lo, fused.lo, sizeof(float) * n * kRank) &&
          same(prod.mixed, fused.mixed, sizeof(float) * n * kHidden) &&
          same(prod.mixed_q8, fused.mixed_q8, qfn_mmq_q8_1_bytes(n, kHidden)) &&
          same(prod.inject, fused.inject,
               sizeof(float) * n * kStreams * parts);
      // The eager pair's differences, before the graph chains move the
      // buffers on.
      const double diff_lo =
          max_diff(prod.lo, fused.lo, std::size_t{n} * kRank);
      const double diff_mixed =
          max_diff(prod.mixed, fused.mixed, std::size_t{n} * kHidden);
      const double diff_inject = max_diff(prod.inject, fused.inject,
                                          std::size_t{n} * kStreams * parts);
      const auto prod_exec = Capture(stream, [&] {
        for (int m = 0; m < mixers; ++m)
          production(n, m % copies);
      });
      const auto fused_exec = Capture(stream, [&] {
        for (int m = 0; m < mixers; ++m)
          candidate(n, m % copies);
      });
      std::vector<double> prod_rounds, fused_rounds;
      for (int round = 0; round < 7; ++round) {
        prod_rounds.push_back(MedianMicros(prod_exec, stream, replays, mixers));
        fused_rounds.push_back(
            MedianMicros(fused_exec, stream, replays, mixers));
      }
      std::sort(prod_rounds.begin(), prod_rounds.end());
      std::sort(fused_rounds.begin(), fused_rounds.end());
      const double prod_us = prod_rounds[prod_rounds.size() / 2];
      const double fused_us = fused_rounds[fused_rounds.size() / 2];
      std::printf(
          "%6u  %13.2f  %8.2f  %7.3f  %s (max |diff| lo %.3g mixed %.3g "
          "inject %.3g)\n",
          n, prod_us, fused_us, prod_us / fused_us,
          equal ? "identical" : "DIFFER", diff_lo, diff_mixed, diff_inject);
      (void)hipGraphExecDestroy(prod_exec);
      (void)hipGraphExecDestroy(fused_exec);
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
