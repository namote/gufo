// Row-tiled byte-stable FP32 GEMM (ops/plain_prefill_stable.hpp).
//
// SmallGemmStable gives each output a fixed reduction: 32 lane-class
// partial FMA chains over the elements 4L .. 4L + 3 (mod 128) in increasing
// order, then the xor butterfly to lane 0. That reduction does not depend on
// which tokens or weight rows a wave handles, so this kernel keeps it per
// output while tiling: a block stages a 128-element step of BT token rows and
// BR weight rows in LDS (reused by every wave), and each lane, as lane class
// L, extends the chains of a TT x TR register tile of outputs.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <stdexcept>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops/plain_prefill_stable.hpp"

namespace gufo::hip {
namespace {

constexpr int kStep = 128;  // elements per step: 32 lane classes x 4
constexpr int kTT = 8;      // tokens per wave tile
constexpr int kTR = 8;      // weight rows per wave tile
constexpr int kWavesT = 2, kWavesR = 4;
constexpr int kBT = kTT * kWavesT;  // 16 token rows per block
constexpr int kBR = kTR * kWavesR;  // 32 weight rows per block
constexpr int kThreads = kWavesT * kWavesR * 32;

// Exact widening, as SmallGemmKernel's RowElement / vector loads.
template<core::GgmlType Type>
__device__ __forceinline__ float Widen(const void* w, std::size_t i) {
  if constexpr (Type == core::GgmlType::kF32) {
    return static_cast<const float*>(w)[i];
  } else if constexpr (Type == core::GgmlType::kBF16) {
    return __uint_as_float(
        static_cast<std::uint32_t>(static_cast<const std::uint16_t*>(w)[i])
        << 16);
  } else {
    return static_cast<float>(static_cast<const _Float16*>(w)[i]);
  }
}

// Lane (i ^ offset) of a full wave32, as the reference's XorLane / shfl_xor
// (only the transport may differ; the adds are the same).
__device__ __forceinline__ float XorLane(float v, int offset) {
  return __shfl_xor(v, offset, 32);
}

template<core::GgmlType Type>
__global__ __launch_bounds__(kThreads) void PlainStableKernel(
    const void* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, std::uint32_t rows, std::uint32_t n,
    std::uint32_t k) {
  // Double-buffered steps: x[BT][128] and the widened w[BR][128].
  __shared__ float4 s_x[2][kBT][kStep / 4];
  __shared__ float4 s_w[2][kBR][kStep / 4];
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % 32, wave = tid / 32;
  const int wt = wave % kWavesT, wr = wave / kWavesT;
  const std::uint32_t r_block = blockIdx.x * kBR;
  const std::uint32_t t_block = blockIdx.y * kBT;
  const std::uint32_t steps = k / kStep + (k % kStep != 0 ? 1U : 0U);

  // Stage step s: each thread fills float4 slots; elements past k, rows past
  // `rows` and weight rows past n are zero.
  const auto stage = [&](std::uint32_t s, int buf) {
    const std::size_t i0 = std::size_t(s) * kStep;
    for (int slot = tid; slot < (kBT + kBR) * (kStep / 4); slot += kThreads) {
      const int row = slot / (kStep / 4), q = slot % (kStep / 4);
      float v[4];
      if (row < kBT) {
        const std::uint32_t t = t_block + static_cast<std::uint32_t>(row);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const std::size_t i = i0 + static_cast<std::size_t>(q * 4 + e);
          v[e] = t < rows && i < k ? x[std::size_t(t) * k + i] : 0.0F;
        }
        s_x[buf][row][q] = make_float4(v[0], v[1], v[2], v[3]);
      } else {
        const std::uint32_t j = r_block + static_cast<std::uint32_t>(row - kBT);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const std::size_t i = i0 + static_cast<std::size_t>(q * 4 + e);
          v[e] = j < n && i < k ? Widen<Type>(w, std::size_t(j) * k + i) : 0.0F;
        }
        s_w[buf][row - kBT][q] = make_float4(v[0], v[1], v[2], v[3]);
      }
    }
  };

  float acc[kTT][kTR];
#pragma unroll
  for (int a = 0; a < kTT; ++a)
#pragma unroll
    for (int b = 0; b < kTR; ++b)
      acc[a][b] = 0.0F;

  stage(0, 0);
  __syncthreads();
  for (std::uint32_t s = 0; s < steps; ++s) {
    const int buf = static_cast<int>(s & 1U);
    if (s + 1 < steps)
      stage(s + 1, buf ^ 1);
    float4 xv[kTT], wv[kTR];
#pragma unroll
    for (int a = 0; a < kTT; ++a)
      xv[a] = s_x[buf][wt * kTT + a][lane];
#pragma unroll
    for (int b = 0; b < kTR; ++b)
      wv[b] = s_w[buf][wr * kTR + b][lane];
    // The reference chain: four FMAs per step in element order. A lane class
    // whose step starts at or past k runs none (the reference breaks there),
    // since fmaf(0, 0, -0) would turn an underflowed -0 chain into +0; a
    // partially valid step runs all four with zeros, as the reference does.
    if (std::size_t(s) * kStep + static_cast<std::size_t>(lane) * 4 < k) {
#pragma unroll
      for (int a = 0; a < kTT; ++a) {
#pragma unroll
        for (int b = 0; b < kTR; ++b) {
          float c = acc[a][b];
          c = __builtin_fmaf(wv[b].x, xv[a].x, c);
          c = __builtin_fmaf(wv[b].y, xv[a].y, c);
          c = __builtin_fmaf(wv[b].z, xv[a].z, c);
          c = __builtin_fmaf(wv[b].w, xv[a].w, c);
          acc[a][b] = c;
        }
      }
    }
    __syncthreads();
  }

  // The reference's WaveSum per output; lane 0's total is stored.
#pragma unroll
  for (int a = 0; a < kTT; ++a) {
#pragma unroll
    for (int b = 0; b < kTR; ++b) {
      float v = acc[a][b];
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        v += XorLane(v, offset);
      acc[a][b] = v;
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int a = 0; a < kTT; ++a) {
      const std::uint32_t t =
          t_block + static_cast<std::uint32_t>(wt * kTT + a);
#pragma unroll
      for (int b = 0; b < kTR; ++b) {
        const std::uint32_t j =
            r_block + static_cast<std::uint32_t>(wr * kTR + b);
        if (t < rows && j < n)
          y[std::size_t(t) * n + j] = acc[a][b];
      }
    }
  }
}

}  // namespace

void LaunchPlainPrefillStable(core::GgmlType type, const void* weights,
                              const float* input, float* output,
                              std::uint32_t rows, std::uint32_t n,
                              std::uint32_t k, hipStream_t stream) {
  if (n == 0 || k == 0 || weights == nullptr || input == nullptr ||
      output == nullptr) {
    throw std::invalid_argument("plain stable GEMM: bad arguments");
  }
  if (type != core::GgmlType::kF32 && type != core::GgmlType::kBF16 &&
      type != core::GgmlType::kF16) {
    throw std::invalid_argument("plain stable GEMM: F16, BF16 or F32 only");
  }
  // grid.y <= 65535 token blocks per launch; values do not depend on the
  // chunking (each output's reduction is fixed).
  constexpr std::uint32_t kChunkRows = 65535U * kBT;
  const std::uint32_t row_blocks = n / kBR + (n % kBR != 0 ? 1U : 0U);
  for (std::uint32_t first = 0, count = 0; first < rows; first += count) {
    count = std::min(kChunkRows, rows - first);
    const dim3 grid(row_blocks, count / kBT + (count % kBT != 0 ? 1U : 0U));
    const float* x = input + std::size_t(first) * k;
    float* y = output + std::size_t(first) * n;
    switch (type) {
      case core::GgmlType::kF32:
        PlainStableKernel<core::GgmlType::kF32>
            <<<grid, kThreads, 0, stream>>>(weights, x, y, count, n, k);
        break;
      case core::GgmlType::kBF16:
        PlainStableKernel<core::GgmlType::kBF16>
            <<<grid, kThreads, 0, stream>>>(weights, x, y, count, n, k);
        break;
      default:
        PlainStableKernel<core::GgmlType::kF16>
            <<<grid, kThreads, 0, stream>>>(weights, x, y, count, n, k);
        break;
    }
    HIP_CHECK(hipGetLastError());
  }
}

}  // namespace gufo::hip
