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
constexpr unsigned kMaxSlots = kExl3FastMaxGroup;

// DPP quad_perm move: 0xB1 = [1,0,3,2] (lane ^ 1), 0x4E = [2,3,0,1]
// (lane ^ 2). Full-rate VALU instead of ds_bpermute, the same additions as a
// __shfl_xor reduction (bit-identical, xb3).
template<int kCtrl>
__device__ __forceinline__ float QuadMove(float v) {
  return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), kCtrl,
                                                    0xF, 0xF, false));
}

struct RotateArgs {
  const _Float16* suh[kMaxSlots];
  _Float16* out[kMaxSlots];
};

// out[slot] (lane-major, M rows) = FP16(H(suh[slot] * x)) per 128-element
// segment; blockIdx.z is the slot. Rows m .. M - 1 of the pass are zeros.
template<unsigned M>
__global__ __launch_bounds__(kRotateWaves * 32) void Exl3RotateInputKernel(
    const float* __restrict__ x, std::size_t x_stride, unsigned rows,
    unsigned k, RotateArgs args) {
  const unsigned lane = threadIdx.x % 32;
  const unsigned segment = blockIdx.x * kRotateWaves + threadIdx.x / 32;
  const unsigned row = blockIdx.y;
  if (segment >= k / exl3::kSegment) {
    return;
  }
  const _Float16* suh = args.suh[0];
  _Float16* out = args.out[0];
  if (blockIdx.z == 1) {
    suh = args.suh[1];
    out = args.out[1];
  } else if (blockIdx.z == 2) {
    suh = args.suh[2];
    out = args.out[2];
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

// One matrix of a launch: its trellis, rotated input, output and counters.
// Blocks [block_begin, next slot's block_begin) work it.
struct GemvSlot {
  const std::uint16_t* trellis;
  const _Float16* a;
  const _Float16* svh;
  float* y;
  std::size_t y_stride;
  std::uint32_t* counters;
  unsigned n;
  unsigned block_begin;
};

struct GemvArgs {
  GemvSlot slot[kMaxSlots];
  unsigned slots;
  unsigned k;
  unsigned rows;
};

// y[r][n] = svh * H(sum_k a[r][k] W'[k][n]) for r < rows, M >= rows padded
// rows, for every slot. Blocks of W waves own TN adjacent N tiles of one
// slot and split K; the block that completes a 128-column segment rotates it
// in place.
template<int K, unsigned M, unsigned W, unsigned TN>
__global__ __launch_bounds__(W * 32) void Exl3GemvKernel(GemvArgs args) {
  constexpr unsigned kTileDwords = 8 * K;
  __shared__ float red[W][TN][M][16];
  __shared__ unsigned last_block;

  // Wave-uniform slot selection (no dynamic indexing of the argument block).
  GemvSlot sl = args.slot[0];
  if (args.slots > 1 && blockIdx.x >= args.slot[1].block_begin) {
    sl = args.slot[1];
  }
  if (args.slots > 2 && blockIdx.x >= args.slot[2].block_begin) {
    sl = args.slot[2];
  }

  const unsigned lane = threadIdx.x % 32;
  const unsigned warp = threadIdx.x / 32;
  const unsigned n_tiles = sl.n / 16;
  const unsigned k_tiles = args.k / 16;
  const unsigned nt0 = (blockIdx.x - sl.block_begin) * TN;
  const unsigned kb0 = k_tiles * warp / W;
  const unsigned kb1 = k_tiles * (warp + 1) / W;
  const exl3::LaneWindow lw = exl3::MakeLaneWindow<K>(lane);
  const unsigned r0 = (lane % 4) * 2;

  float acc_a[TN][M];
  float acc_b[TN][M];
#pragma unroll
  for (unsigned t = 0; t < TN; ++t) {
#pragma unroll
    for (unsigned r = 0; r < M; ++r) {
      acc_a[t][r] = 0.0F;
      acc_b[t][r] = 0.0F;
    }
  }

  const std::uint32_t* base =
      reinterpret_cast<const std::uint32_t*>(sl.trellis);
  for (unsigned kt = kb0; kt < kb1; ++kt) {
    Half2 a01[M];
    Half2 a89[M];
    const Half4* group = reinterpret_cast<const Half4*>(
        sl.a + (static_cast<std::size_t>(kt) * 16 + r0 * 2) * M);
#pragma unroll
    for (unsigned r = 0; r < M; ++r) {
      const Half4 v = group[r];
      a01[r] = Half2{v.x, v.y};
      a89[r] = Half2{v.z, v.w};
    }
#pragma unroll
    for (unsigned t = 0; t < TN; ++t) {
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
  for (unsigned t = 0; t < TN; ++t) {
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
      for (unsigned t = 0; t < TN; ++t) {
#pragma unroll
        for (unsigned r = 0; r < M; ++r) {
          if (r < args.rows) {
            float sum = red[0][t][r][lane];
#pragma unroll
            for (unsigned v = 1; v < W; ++v) {
              sum += red[v][t][r][lane];
            }
            sl.y[r * sl.y_stride + (nt0 + t) * 16 + lane] = sum;
          }
        }
      }
    }
    __threadfence();
    unsigned last = 0;
    const unsigned segment = nt0 / 8;
    if (lane == 0) {
      constexpr unsigned kArrivals = 8 / TN;
      const unsigned old = atomicAdd(&sl.counters[segment], 1U);
      last = old == kArrivals - 1 ? 1U : 0U;
      if (last != 0) {
        sl.counters[segment] = 0;
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
  for (unsigned r = warp; r < args.rows; r += W) {
    float* row = sl.y + r * sl.y_stride + seg_base + lane * 4;
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
               static_cast<float>(sl.svh[seg_base + lane * 4 + j]);
    }
  }
}

struct LaunchShape {
  unsigned tiles_per_wave;
  unsigned waves;
};

// From the xb2/xb3 sweeps (plan/exl3/XB-RESULTS.md): launches of 352 to 511
// two-tile blocks (q, q + k + v) run best with four tiles per wave and four
// waves; otherwise two tiles per wave, with more waves for narrow outputs
// and long K.
LaunchShape PickShape(unsigned k_tiles, unsigned blocks_two_tiles) {
  if (blocks_two_tiles >= 352 && blocks_two_tiles < 512) {
    return {4, 4};
  }
  if (blocks_two_tiles < 64) {
    return {2, 16};
  }
  if (k_tiles >= 384) {
    return {2, 8};
  }
  return {2, 4};
}

template<int K, unsigned M, unsigned W, unsigned TN>
void LaunchGemvKernel(GemvArgs args, hipStream_t stream) {
  unsigned blocks = 0;
  for (unsigned i = 0; i < args.slots; ++i) {
    args.slot[i].block_begin = blocks;
    blocks += args.slot[i].n / 16 / TN;
  }
  hipLaunchKernelGGL((Exl3GemvKernel<K, M, W, TN>), dim3(blocks), dim3(W * 32),
                     0, stream, args);
}

template<int K, unsigned M>
void LaunchGemvRows(const GemvArgs& args, const LaunchShape& shape,
                    hipStream_t stream) {
  if (shape.tiles_per_wave == 4) {
    LaunchGemvKernel<K, M, 4, 4>(args, stream);
    return;
  }
  switch (shape.waves) {
    case 4:
      LaunchGemvKernel<K, M, 4, 2>(args, stream);
      break;
    case 8:
      LaunchGemvKernel<K, M, 8, 2>(args, stream);
      break;
    default:
      LaunchGemvKernel<K, M, 16, 2>(args, stream);
      break;
  }
}

template<unsigned M>
void LaunchGemvBits(int bits, const GemvArgs& args, const LaunchShape& shape,
                    hipStream_t stream) {
  switch (bits) {
    case 1:
      LaunchGemvRows<1, M>(args, shape, stream);
      break;
    case 2:
      LaunchGemvRows<2, M>(args, shape, stream);
      break;
    case 3:
      LaunchGemvRows<3, M>(args, shape, stream);
      break;
    case 4:
      LaunchGemvRows<4, M>(args, shape, stream);
      break;
    case 5:
      LaunchGemvRows<5, M>(args, shape, stream);
      break;
    default:
      LaunchGemvRows<6, M>(args, shape, stream);
      break;
  }
}

template<unsigned M>
void LaunchPass(std::span<const Exl3FastTarget> targets, const float* x,
                std::size_t x_stride, unsigned rows, std::size_t first_row,
                const Exl3FastWorkspace& ws, hipStream_t stream) {
  const auto& first = *targets[0].linear;
  auto* slab = reinterpret_cast<_Float16*>(ws.input);
  RotateArgs rotate{};
  GemvArgs gemv{};
  gemv.slots = static_cast<unsigned>(targets.size());
  gemv.k = first.k;
  gemv.rows = rows;
  std::size_t counter = 0;
  unsigned blocks_two_tiles = 0;
  for (std::size_t s = 0; s < targets.size(); ++s) {
    const auto& linear = *targets[s].linear;
    _Float16* input = slab + s * exl3::kMaxRows * first.k;
    rotate.suh[s] = reinterpret_cast<const _Float16*>(linear.suh);
    rotate.out[s] = input;
    GemvSlot& slot = gemv.slot[s];
    slot.trellis = linear.trellis;
    slot.a = input;
    slot.svh = reinterpret_cast<const _Float16*>(linear.svh);
    slot.y_stride = targets[s].y_stride;
    slot.y = targets[s].y + first_row * slot.y_stride;
    slot.counters = ws.counters + counter;
    slot.n = linear.n;
    counter += linear.n / exl3::kSegment;
    blocks_two_tiles += linear.n / 32;
  }
  const unsigned segments = first.k / exl3::kSegment;
  hipLaunchKernelGGL(
      Exl3RotateInputKernel<M>,
      dim3((segments + kRotateWaves - 1) / kRotateWaves, M, gemv.slots),
      dim3(kRotateWaves * 32), 0, stream, x, x_stride, rows, first.k, rotate);
  LaunchGemvBits<M>(first.rate.bits, gemv,
                    PickShape(first.k / 16, blocks_two_tiles), stream);
  HIP_CHECK(hipGetLastError());
}

}  // namespace

bool Exl3FastSupported(const quant::exl3::DeviceLinear& linear) noexcept {
  return linear.codebook == quant::exl3::Codebook::kMul1 && !linear.rate.half &&
         linear.rate.bits >= 1 && linear.rate.bits <= 6 &&
         linear.k % exl3::kSegment == 0 && linear.n % exl3::kSegment == 0;
}

bool Exl3FastGroupSupported(std::span<const Exl3FastTarget> targets,
                            const Exl3FastWorkspace& workspace) noexcept {
  if (targets.empty() || targets.size() > kExl3FastMaxGroup) {
    return false;
  }
  const auto& first = *targets[0].linear;
  std::size_t counters = 0;
  for (const Exl3FastTarget& t : targets) {
    if (!Exl3FastSupported(*t.linear) || t.linear->k != first.k ||
        !(t.linear->rate == first.rate)) {
      return false;
    }
    counters += t.linear->n / exl3::kSegment;
  }
  return workspace.input_elements >= targets.size() * exl3::kMaxRows *
                                         static_cast<std::size_t>(first.k) &&
         workspace.counter_count >= counters;
}

void LaunchExl3FastProjections(std::span<const Exl3FastTarget> targets,
                               const float* x, std::size_t x_stride,
                               std::size_t rows,
                               const Exl3FastWorkspace& workspace,
                               hipStream_t stream) {
  if (!Exl3FastGroupSupported(targets, workspace) || rows == 0 ||
      rows > kExl3FastMaxRows) {
    throw std::invalid_argument("EXL3 fast projection: unsupported call");
  }
  for (std::size_t first = 0; first < rows; first += exl3::kMaxRows) {
    const auto pass = static_cast<unsigned>(
        std::min<std::size_t>(exl3::kMaxRows, rows - first));
    const float* xp = x + first * x_stride;
    if (pass == 1) {
      LaunchPass<1>(targets, xp, x_stride, pass, first, workspace, stream);
    } else if (pass == 2) {
      LaunchPass<2>(targets, xp, x_stride, pass, first, workspace, stream);
    } else if (pass <= 4) {
      LaunchPass<4>(targets, xp, x_stride, pass, first, workspace, stream);
    } else {
      LaunchPass<8>(targets, xp, x_stride, pass, first, workspace, stream);
    }
  }
}

void LaunchExl3FastProjection(const quant::exl3::DeviceLinear& linear,
                              const float* x, std::size_t x_stride,
                              std::size_t rows, float* y, std::size_t y_stride,
                              const Exl3FastWorkspace& workspace,
                              hipStream_t stream) {
  const Exl3FastTarget target{&linear, y, y_stride};
  LaunchExl3FastProjections(std::span(&target, 1), x, x_stride, rows, workspace,
                            stream);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
