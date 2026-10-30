// X2b EXL3 decode/verify projection
// (src/models/qwen/hip/kernels/exl3_gemv.hpp). Device helpers and the
// arithmetic contract: src/core/hip/detail/exl3_gemv.hpp.

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <stdexcept>

#include "src/core/hip/detail/exl3_gemv.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/kernels/exl3_gemv.hpp"

namespace gufo::hip {
namespace {

using exl3::Half2;
using exl3::Half4;
using exl3::kHadamardScale;

constexpr unsigned kRotateWaves = 8;

// DPP quad_perm move: 0xB1 = [1,0,3,2] (lane ^ 1), 0x4E = [2,3,0,1]
// (lane ^ 2). Full-rate VALU instead of ds_bpermute, the same additions as a
// __shfl_xor reduction (bit-identical, xb3).
template<int kCtrl>
__device__ __forceinline__ float QuadMove(float v) {
  return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), kCtrl,
                                                    0xF, 0xF, false));
}
constexpr int kTilesPerWave = 2;

// out (lane-major, M rows) = FP16(H(suh * x)) per 128-element segment. Rows
// m .. M - 1 of the pass are written as zeros.
template<unsigned M>
__global__ __launch_bounds__(kRotateWaves * 32) void Exl3RotateInputKernel(
    const float* __restrict__ x, std::size_t x_stride, unsigned rows,
    unsigned k, const _Float16* __restrict__ suh, _Float16* __restrict__ out) {
  const unsigned lane = threadIdx.x % 32;
  const unsigned segment = blockIdx.x * kRotateWaves + threadIdx.x / 32;
  const unsigned row = blockIdx.y;
  if (segment >= k / exl3::kSegment) {
    return;
  }
  const unsigned first = segment * exl3::kSegment + lane * 4;
  float v[4];
  if (row < rows) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[j] = x[row * x_stride + first + j] * static_cast<float>(suh[first + j]);
      // Pin the FP32 rounding of the product: without it LLVM fuses it into
      // the first butterfly add (v_fma_mix_f32 with the FP16 suh operand).
      asm volatile("" : "+v"(v[j]));
    }
    exl3::Butterfly128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[j] *= kHadamardScale;
      // Pin the FP32 rounding before the FP16 conversion (no v_fma_mix).
      asm volatile("" : "+v"(v[j]));
    }
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[j] = 0.0F;
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    out[exl3::LaneMajorOffset(first + j, row, M)] = static_cast<_Float16>(v[j]);
  }
}

// y[r][n] = svh * H(sum_k a[r][k] W'[k][n]) for r < rows, M >= rows padded
// rows. Blocks of W waves own kTilesPerWave adjacent N tiles and split K;
// the block that completes a 128-column segment rotates it in place.
template<int K, unsigned M, unsigned W>
__global__ __launch_bounds__(W * 32) void Exl3GemvKernel(
    const std::uint16_t* __restrict__ trellis, const _Float16* __restrict__ a,
    const _Float16* __restrict__ svh, float* __restrict__ y,
    std::size_t y_stride, std::uint32_t* __restrict__ counters, unsigned k,
    unsigned n, unsigned rows) {
  constexpr unsigned kTileDwords = 8 * K;
  constexpr int TN = kTilesPerWave;
  __shared__ float red[W][TN][M][16];
  __shared__ unsigned last_block;

  const unsigned lane = threadIdx.x % 32;
  const unsigned warp = threadIdx.x / 32;
  const unsigned n_tiles = n / 16;
  const unsigned k_tiles = k / 16;
  const unsigned nt0 = blockIdx.x * TN;
  const unsigned kb0 = k_tiles * warp / W;
  const unsigned kb1 = k_tiles * (warp + 1) / W;
  const exl3::LaneWindow lw = exl3::MakeLaneWindow<K>(lane);
  const unsigned r0 = (lane % 4) * 2;

  float acc_a[TN][M];
  float acc_b[TN][M];
#pragma unroll
  for (int t = 0; t < TN; ++t) {
#pragma unroll
    for (unsigned r = 0; r < M; ++r) {
      acc_a[t][r] = 0.0F;
      acc_b[t][r] = 0.0F;
    }
  }

  const std::uint32_t* base = reinterpret_cast<const std::uint32_t*>(trellis);
  for (unsigned kt = kb0; kt < kb1; ++kt) {
    Half2 a01[M];
    Half2 a89[M];
    const Half4* group = reinterpret_cast<const Half4*>(
        a + (static_cast<std::size_t>(kt) * 16 + r0 * 2) * M);
#pragma unroll
    for (unsigned r = 0; r < M; ++r) {
      const Half4 v = group[r];
      a01[r] = Half2{v.x, v.y};
      a89[r] = Half2{v.z, v.w};
    }
#pragma unroll
    for (int t = 0; t < TN; ++t) {
      const std::uint32_t* tile =
          base +
          (static_cast<std::size_t>(kt) * n_tiles + nt0 + t) * kTileDwords;
      Half2 w[4];
      exl3::DecodeLaneMul1<K>(tile, lw, w);
#pragma unroll
      for (unsigned r = 0; r < M; ++r) {
        acc_a[t][r] = __builtin_amdgcn_fdot2(w[0], a01[r], acc_a[t][r], false);
        acc_a[t][r] = __builtin_amdgcn_fdot2(w[1], a89[r], acc_a[t][r], false);
        acc_b[t][r] = __builtin_amdgcn_fdot2(w[2], a01[r], acc_b[t][r], false);
        acc_b[t][r] = __builtin_amdgcn_fdot2(w[3], a89[r], acc_b[t][r], false);
      }
    }
  }

  // Quad reduction: lanes 4c .. 4c + 3 hold columns c and c + 8.
#pragma unroll
  for (int t = 0; t < TN; ++t) {
#pragma unroll
    for (unsigned r = 0; r < M; ++r) {
      float va = acc_a[t][r];
      float vb = acc_b[t][r];
      va += QuadMove<0xB1>(va);
      vb += QuadMove<0xB1>(vb);
      va += QuadMove<0x4E>(va);
      vb += QuadMove<0x4E>(vb);
      if (lane % 4 == 0) {
        red[warp][t][r][lane / 4] = va;
        red[warp][t][r][lane / 4 + 8] = vb;
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
    if (lane < 16) {
#pragma unroll
      for (int t = 0; t < TN; ++t) {
#pragma unroll
        for (unsigned r = 0; r < M; ++r) {
          if (r < rows) {
            float sum = red[0][t][r][lane];
#pragma unroll
            for (unsigned v = 1; v < W; ++v) {
              sum += red[v][t][r][lane];
            }
            y[r * y_stride + (nt0 + t) * 16 + lane] = sum;
          }
        }
      }
    }
    __threadfence();
    unsigned last = 0;
    const unsigned segment = nt0 / 8;
    if (lane == 0) {
      constexpr unsigned kArrivals = 8 / TN;
      const unsigned old = atomicAdd(&counters[segment], 1U);
      last = old == kArrivals - 1 ? 1U : 0U;
      if (last != 0) {
        counters[segment] = 0;
      }
    }
    last_block = __shfl(last, 0, 32);
  }
  __syncthreads();
  if (last_block == 0) {
    return;
  }
  __threadfence();

  // The segment is complete: rotate it in place, one wave per row.
  const unsigned seg_base = (nt0 / 8) * exl3::kSegment;
  for (unsigned r = warp; r < rows; r += W) {
    float* row = y + r * y_stride + seg_base + lane * 4;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[j] = __hip_atomic_load(row + j, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
    }
    exl3::Butterfly128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      row[j] = (v[j] * kHadamardScale) *
               static_cast<float>(svh[seg_base + lane * 4 + j]);
    }
  }
}

// Split-K waves per block for kTilesPerWave = 2, from the xb2 sweep
// (plan/exl3/XB-RESULTS.md): narrow outputs need more waves, long K and the
// 384-512-block shapes run best at 8, the rest at 4.
unsigned SplitWaves(unsigned k_tiles, unsigned blocks) {
  if (blocks < 64) {
    return 16;
  }
  if (k_tiles >= 384 || (blocks >= 352 && blocks < 512)) {
    return 8;
  }
  return 4;
}

template<int K, unsigned M>
void LaunchGemvRows(const quant::exl3::DeviceLinear& linear,
                    const _Float16* input, float* y, std::size_t y_stride,
                    std::uint32_t* counters, unsigned rows,
                    hipStream_t stream) {
  const unsigned blocks = linear.n / 16 / kTilesPerWave;
  const unsigned waves = SplitWaves(linear.k / 16, blocks);
  const auto* svh = reinterpret_cast<const _Float16*>(linear.svh);
  switch (waves) {
    case 4:
      hipLaunchKernelGGL((Exl3GemvKernel<K, M, 4>), dim3(blocks), dim3(128), 0,
                         stream, linear.trellis, input, svh, y, y_stride,
                         counters, linear.k, linear.n, rows);
      break;
    case 8:
      hipLaunchKernelGGL((Exl3GemvKernel<K, M, 8>), dim3(blocks), dim3(256), 0,
                         stream, linear.trellis, input, svh, y, y_stride,
                         counters, linear.k, linear.n, rows);
      break;
    default:
      hipLaunchKernelGGL((Exl3GemvKernel<K, M, 16>), dim3(blocks), dim3(512), 0,
                         stream, linear.trellis, input, svh, y, y_stride,
                         counters, linear.k, linear.n, rows);
      break;
  }
}

template<unsigned M>
void LaunchPass(const quant::exl3::DeviceLinear& linear, const float* x,
                std::size_t x_stride, unsigned rows, float* y,
                std::size_t y_stride, const Exl3FastWorkspace& ws,
                hipStream_t stream) {
  auto* input = reinterpret_cast<_Float16*>(ws.input);
  const unsigned segments = linear.k / exl3::kSegment;
  hipLaunchKernelGGL(Exl3RotateInputKernel<M>,
                     dim3((segments + kRotateWaves - 1) / kRotateWaves, M),
                     dim3(kRotateWaves * 32), 0, stream, x, x_stride, rows,
                     linear.k, reinterpret_cast<const _Float16*>(linear.suh),
                     input);
  switch (linear.rate.bits) {
    case 1:
      LaunchGemvRows<1, M>(linear, input, y, y_stride, ws.counters, rows,
                           stream);
      break;
    case 2:
      LaunchGemvRows<2, M>(linear, input, y, y_stride, ws.counters, rows,
                           stream);
      break;
    case 3:
      LaunchGemvRows<3, M>(linear, input, y, y_stride, ws.counters, rows,
                           stream);
      break;
    case 4:
      LaunchGemvRows<4, M>(linear, input, y, y_stride, ws.counters, rows,
                           stream);
      break;
    case 5:
      LaunchGemvRows<5, M>(linear, input, y, y_stride, ws.counters, rows,
                           stream);
      break;
    default:
      LaunchGemvRows<6, M>(linear, input, y, y_stride, ws.counters, rows,
                           stream);
      break;
  }
  HIP_CHECK(hipGetLastError());
}

}  // namespace

bool Exl3FastSupported(const quant::exl3::DeviceLinear& linear) noexcept {
  return linear.codebook == quant::exl3::Codebook::kMul1 && !linear.rate.half &&
         linear.rate.bits >= 1 && linear.rate.bits <= 6 &&
         linear.k % exl3::kSegment == 0 && linear.n % exl3::kSegment == 0;
}

void LaunchExl3FastProjection(const quant::exl3::DeviceLinear& linear,
                              const float* x, std::size_t x_stride,
                              std::size_t rows, float* y, std::size_t y_stride,
                              const Exl3FastWorkspace& workspace,
                              hipStream_t stream) {
  if (!Exl3FastSupported(linear) || rows == 0 || rows > kExl3FastMaxRows) {
    throw std::invalid_argument("EXL3 fast projection: unsupported call");
  }
  if (workspace.input_elements <
          static_cast<std::size_t>(exl3::kMaxRows) * linear.k ||
      workspace.counter_count < linear.n / exl3::kSegment) {
    throw std::invalid_argument("EXL3 fast projection: workspace too small");
  }
  for (std::size_t first = 0; first < rows; first += exl3::kMaxRows) {
    const auto pass = static_cast<unsigned>(
        std::min<std::size_t>(exl3::kMaxRows, rows - first));
    const float* xp = x + first * x_stride;
    float* yp = y + first * y_stride;
    if (pass == 1) {
      LaunchPass<1>(linear, xp, x_stride, pass, yp, y_stride, workspace,
                    stream);
    } else if (pass == 2) {
      LaunchPass<2>(linear, xp, x_stride, pass, yp, y_stride, workspace,
                    stream);
    } else if (pass <= 4) {
      LaunchPass<4>(linear, xp, x_stride, pass, yp, y_stride, workspace,
                    stream);
    } else {
      LaunchPass<8>(linear, xp, x_stride, pass, yp, y_stride, workspace,
                    stream);
    }
  }
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
