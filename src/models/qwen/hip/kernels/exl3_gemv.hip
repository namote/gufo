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

// RMSNorm of one 5120-wide row, the arithmetic of norm.hip RMSNormKernel
// <5120> (cached row, FMA sum, the same tree), stored as FP32 and, for every
// consumer, rotated with its suh into its FP16 slab (M = 1 lane-major), with
// the input-rotation arithmetic of Exl3RotateInputKernel.
constexpr std::size_t kNormDim = 5120;

__global__ __launch_bounds__(256) void Exl3RmsNormRotateKernel(
    const float* __restrict__ x, const float* __restrict__ weight,
    float* __restrict__ out, float eps, RotateArgs rotate, unsigned consumers) {
  __shared__ float s_sum[256];
  __shared__ float s_row[kNormDim];
  const std::size_t tid = threadIdx.x;
  constexpr std::size_t kItems = kNormDim / 256;
  float values[kItems];
  float weights[kItems];
  float sum_sq = 0.0F;
#pragma unroll
  for (std::size_t item = 0; item < kItems; ++item) {
    const std::size_t i = tid + item * 256;
    const float val = x[i];
    values[item] = val;
    weights[item] = weight != nullptr ? weight[i] : 1.0F;
    sum_sq = __fmaf_rn(val, val, sum_sq);
  }
  s_sum[tid] = sum_sq;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    __syncthreads();
  }
  const float mean_sq = s_sum[0] / static_cast<float>(kNormDim);
  const float inv_rms = rsqrtf(mean_sq + eps);
#pragma unroll
  for (std::size_t item = 0; item < kItems; ++item) {
    const float normed = (values[item] * inv_rms) * weights[item];
    out[tid + item * 256] = normed;
    s_row[tid + item * 256] = normed;
  }
  __syncthreads();
  const unsigned lane = tid % 32;
  const unsigned wave = tid / 32;
  constexpr unsigned kSegments = kNormDim / exl3::kSegment;
  for (unsigned c = 0; c < consumers; ++c) {
    const _Float16* suh = rotate.suh[0];
    _Float16* slab = rotate.out[0];
    if (c == 1) {
      suh = rotate.suh[1];
      slab = rotate.out[1];
    } else if (c == 2) {
      suh = rotate.suh[2];
      slab = rotate.out[2];
    }
    for (unsigned segment = wave; segment < kSegments; segment += 8) {
      const unsigned first = segment * exl3::kSegment + lane * 4;
      float v[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        v[j] = s_row[first + j] * static_cast<float>(suh[first + j]);
        asm volatile("" : "+v"(v[j]));
      }
      exl3::Butterfly128(v, lane);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        v[j] *= kHadamardScale;
        asm volatile("" : "+v"(v[j]));
        slab[exl3::LaneMajorOffset(first + j, 0, 1)] =
            static_cast<_Float16>(v[j]);
      }
    }
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
  // kSwiGLUDown only: the down projection's suh and its lane-major FP16
  // input slab (M rows of the gate/up width).
  const _Float16* down_suh;
  _Float16* down_input;
};

// kStore: rotate each completed segment of y in place (y = svh * H(acc)).
// kSwiGLUDown: slots 0 and 1 are gate and up of the same width sharing one
// counter per segment; the last block of a segment rotates both, applies
// SwiGLU exactly as LaunchBatchedSwiGLUActivation, and writes the down
// projection's rotated FP16 input (the input-rotation arithmetic of
// Exl3RotateInputKernel) instead of y.
constexpr int kStore = 0;
constexpr int kSwiGLUDown = 1;

// y[r][n] = svh * H(sum_k a[r][k] W'[k][n]) for r < rows, M >= rows padded
// rows, for every slot. Blocks of W waves own TN adjacent N tiles of one
// slot and split K; the block that completes a 128-column segment rotates it
// in place.
template<int K, unsigned M, unsigned W, unsigned TN, int kEpilogue>
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
      constexpr unsigned kArrivals =
          (kEpilogue == kSwiGLUDown ? 2 : 1) * 8 / TN;
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

  const unsigned seg_base = (nt0 / 8) * exl3::kSegment;
  if constexpr (kEpilogue == kSwiGLUDown) {
    // Gate and up of the segment are complete: one wave per row.
    const GemvSlot& gate = args.slot[0];
    const GemvSlot& up = args.slot[1];
    for (unsigned r = warp; r < args.rows; r += W) {
      const unsigned first = seg_base + lane * 4;
      float g[4];
      float u[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        g[j] = __hip_atomic_load(gate.y + r * gate.y_stride + first + j,
                                 __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        u[j] = __hip_atomic_load(up.y + r * up.y_stride + first + j,
                                 __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      exl3::Butterfly128(g, lane);
      exl3::Butterfly128(u, lane);
      float v[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        // The stored kStore values, then BatchedSwiGLUActivationKernel.
        float gv =
            (g[j] * kHadamardScale) * static_cast<float>(gate.svh[first + j]);
        float uv =
            (u[j] * kHadamardScale) * static_cast<float>(up.svh[first + j]);
        asm volatile("" : "+v"(gv));
        asm volatile("" : "+v"(uv));
        const float sig_g = 1.0F / (1.0F + expf(-gv));
        float val = (gv * sig_g) * uv;
        asm volatile("" : "+v"(val));
        v[j] = val * static_cast<float>(args.down_suh[first + j]);
        asm volatile("" : "+v"(v[j]));
      }
      exl3::Butterfly128(v, lane);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        v[j] *= kHadamardScale;
        asm volatile("" : "+v"(v[j]));
        args.down_input[exl3::LaneMajorOffset(first + j, r, M)] =
            static_cast<_Float16>(v[j]);
      }
    }
    return;
  }
  // The segment is complete: rotate it in place, one wave per row.
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

template<int K, unsigned M, unsigned W, unsigned TN, int kEpilogue>
void LaunchGemvKernel(GemvArgs args, hipStream_t stream) {
  unsigned blocks = 0;
  for (unsigned i = 0; i < args.slots; ++i) {
    args.slot[i].block_begin = blocks;
    blocks += args.slot[i].n / 16 / TN;
  }
  hipLaunchKernelGGL((Exl3GemvKernel<K, M, W, TN, kEpilogue>), dim3(blocks),
                     dim3(W * 32), 0, stream, args);
}

template<int K, unsigned M, int kEpilogue>
void LaunchGemvRows(const GemvArgs& args, const LaunchShape& shape,
                    hipStream_t stream) {
  if (shape.tiles_per_wave == 4) {
    LaunchGemvKernel<K, M, 4, 4, kEpilogue>(args, stream);
    return;
  }
  switch (shape.waves) {
    case 4:
      LaunchGemvKernel<K, M, 4, 2, kEpilogue>(args, stream);
      break;
    case 8:
      LaunchGemvKernel<K, M, 8, 2, kEpilogue>(args, stream);
      break;
    default:
      LaunchGemvKernel<K, M, 16, 2, kEpilogue>(args, stream);
      break;
  }
}

template<unsigned M, int kEpilogue = kStore>
void LaunchGemvBits(int bits, const GemvArgs& args, const LaunchShape& shape,
                    hipStream_t stream) {
  switch (bits) {
    case 1:
      LaunchGemvRows<1, M, kEpilogue>(args, shape, stream);
      break;
    case 2:
      LaunchGemvRows<2, M, kEpilogue>(args, shape, stream);
      break;
    case 3:
      LaunchGemvRows<3, M, kEpilogue>(args, shape, stream);
      break;
    case 4:
      LaunchGemvRows<4, M, kEpilogue>(args, shape, stream);
      break;
    case 5:
      LaunchGemvRows<5, M, kEpilogue>(args, shape, stream);
      break;
    case 7:
      LaunchGemvRows<7, M, kEpilogue>(args, shape, stream);
      break;
    default:
      LaunchGemvRows<6, M, kEpilogue>(args, shape, stream);
      break;
  }
}

template<unsigned M>
void LaunchPass(std::span<const Exl3FastTarget> targets, const float* x,
                std::size_t x_stride, unsigned rows, std::size_t first_row,
                const Exl3FastWorkspace& ws, hipStream_t stream,
                bool input_rotated) {
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
  if (!input_rotated) {
    hipLaunchKernelGGL(
        Exl3RotateInputKernel<M>,
        dim3((segments + kRotateWaves - 1) / kRotateWaves, M, gemv.slots),
        dim3(kRotateWaves * 32), 0, stream, x, x_stride, rows, first.k, rotate);
  }
  LaunchGemvBits<M>(first.rate.bits, gemv,
                    PickShape(first.k / 16, blocks_two_tiles), stream);
  HIP_CHECK(hipGetLastError());
}

// One pass of the fused MLP: gate and up of the rotated input with the
// kSwiGLUDown epilogue, then down from its rotated slab into out.
template<unsigned M>
void LaunchMlpPass(const Exl3FastMlp& mlp, const float* x, std::size_t x_stride,
                   unsigned rows, std::size_t first_row,
                   const Exl3FastWorkspace& ws, hipStream_t stream,
                   bool input_rotated) {
  const auto& gate = *mlp.gate;
  const auto& up = *mlp.up;
  const auto& down = *mlp.down;
  auto* slab = reinterpret_cast<_Float16*>(ws.input);
  auto* down_slab = reinterpret_cast<_Float16*>(ws.down_input);
  RotateArgs rotate{};
  GemvArgs gemv{};
  gemv.slots = 2;
  gemv.k = gate.k;
  gemv.rows = rows;
  gemv.down_suh = reinterpret_cast<const _Float16*>(down.suh);
  gemv.down_input = down_slab;
  const quant::exl3::DeviceLinear* pair[2] = {&gate, &up};
  float* scratch[2] = {mlp.gate_y, mlp.up_y};
  for (unsigned s = 0; s < 2; ++s) {
    _Float16* input = slab + s * exl3::kMaxRows * gate.k;
    rotate.suh[s] = reinterpret_cast<const _Float16*>(pair[s]->suh);
    rotate.out[s] = input;
    GemvSlot& slot = gemv.slot[s];
    slot.trellis = pair[s]->trellis;
    slot.a = input;
    slot.svh = reinterpret_cast<const _Float16*>(pair[s]->svh);
    slot.y_stride = gate.n;
    slot.y = scratch[s];
    slot.counters = ws.counters;  // shared: both slots arrive per segment
    slot.n = gate.n;
  }
  const unsigned segments = gate.k / exl3::kSegment;
  if (!input_rotated) {
    hipLaunchKernelGGL(Exl3RotateInputKernel<M>,
                       dim3((segments + kRotateWaves - 1) / kRotateWaves, M, 2),
                       dim3(kRotateWaves * 32), 0, stream, x, x_stride, rows,
                       gate.k, rotate);
  }
  LaunchGemvBits<M, kSwiGLUDown>(
      gate.rate.bits, gemv, PickShape(gate.k / 16, 2 * (gate.n / 32)), stream);
  GemvArgs down_args{};
  down_args.slots = 1;
  down_args.k = down.k;
  down_args.rows = rows;
  GemvSlot& slot = down_args.slot[0];
  slot.trellis = down.trellis;
  slot.a = down_slab;
  slot.svh = reinterpret_cast<const _Float16*>(down.svh);
  slot.y_stride = mlp.out_stride;
  slot.y = mlp.out + first_row * slot.y_stride;
  slot.counters = ws.counters;
  slot.n = down.n;
  LaunchGemvBits<M>(down.rate.bits, down_args,
                    PickShape(down.k / 16, down.n / 32), stream);
  HIP_CHECK(hipGetLastError());
}

}  // namespace

bool Exl3FastSupported(const quant::exl3::DeviceLinear& linear) noexcept {
  return linear.codebook == quant::exl3::Codebook::kMul1 && !linear.rate.half &&
         linear.rate.bits >= 1 && linear.rate.bits <= 7 &&
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
                               hipStream_t stream, bool input_rotated) {
  if (!Exl3FastGroupSupported(targets, workspace) || rows == 0 ||
      rows > kExl3FastMaxRows || (input_rotated && rows != 1)) {
    throw std::invalid_argument("EXL3 fast projection: unsupported call");
  }
  if (input_rotated) {
    LaunchPass<1>(targets, x, x_stride, 1, 0, workspace, stream, true);
    return;
  }
  for (std::size_t first = 0; first < rows; first += exl3::kMaxRows) {
    const auto pass = static_cast<unsigned>(
        std::min<std::size_t>(exl3::kMaxRows, rows - first));
    const float* xp = x + first * x_stride;
    if (pass == 1) {
      LaunchPass<1>(targets, xp, x_stride, pass, first, workspace, stream,
                    false);
    } else if (pass == 2) {
      LaunchPass<2>(targets, xp, x_stride, pass, first, workspace, stream,
                    false);
    } else if (pass <= 4) {
      LaunchPass<4>(targets, xp, x_stride, pass, first, workspace, stream,
                    false);
    } else {
      LaunchPass<8>(targets, xp, x_stride, pass, first, workspace, stream,
                    false);
    }
  }
}

bool Exl3FastMlpSupported(const Exl3FastMlp& mlp,
                          const Exl3FastWorkspace& workspace) noexcept {
  const auto& gate = *mlp.gate;
  const auto& up = *mlp.up;
  const auto& down = *mlp.down;
  return Exl3FastSupported(gate) && Exl3FastSupported(up) &&
         Exl3FastSupported(down) && up.k == gate.k && up.n == gate.n &&
         up.rate == gate.rate && down.k == gate.n &&
         workspace.down_input != nullptr &&
         workspace.input_elements >=
             2 * exl3::kMaxRows * static_cast<std::size_t>(gate.k) &&
         workspace.down_input_elements >=
             exl3::kMaxRows * static_cast<std::size_t>(down.k) &&
         workspace.counter_count >= gate.n / exl3::kSegment &&
         workspace.counter_count >= down.n / exl3::kSegment;
}

void LaunchExl3FastMlp(const Exl3FastMlp& mlp, const float* x,
                       std::size_t x_stride, std::size_t rows,
                       const Exl3FastWorkspace& workspace, hipStream_t stream,
                       bool input_rotated) {
  if (!Exl3FastMlpSupported(mlp, workspace) || rows == 0 ||
      rows > kExl3FastMaxRows || (input_rotated && rows != 1)) {
    throw std::invalid_argument("EXL3 fast MLP: unsupported call");
  }
  if (input_rotated) {
    LaunchMlpPass<1>(mlp, x, x_stride, 1, 0, workspace, stream, true);
    return;
  }
  for (std::size_t first = 0; first < rows; first += exl3::kMaxRows) {
    const auto pass = static_cast<unsigned>(
        std::min<std::size_t>(exl3::kMaxRows, rows - first));
    const float* xp = x + first * x_stride;
    if (pass == 1) {
      LaunchMlpPass<1>(mlp, xp, x_stride, pass, first, workspace, stream,
                       false);
    } else if (pass == 2) {
      LaunchMlpPass<2>(mlp, xp, x_stride, pass, first, workspace, stream,
                       false);
    } else if (pass <= 4) {
      LaunchMlpPass<4>(mlp, xp, x_stride, pass, first, workspace, stream,
                       false);
    } else {
      LaunchMlpPass<8>(mlp, xp, x_stride, pass, first, workspace, stream,
                       false);
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
                            stream, false);
}

bool LaunchExl3FastRmsNorm(
    const float* x, const float* weight, float* normed, std::size_t dim,
    float eps, std::span<const quant::exl3::DeviceLinear* const> consumers,
    const Exl3FastWorkspace& workspace, hipStream_t stream) {
  if (dim != kNormDim || consumers.empty() ||
      consumers.size() > kExl3FastMaxGroup ||
      workspace.input_elements < consumers.size() * exl3::kMaxRows * dim) {
    return false;
  }
  RotateArgs rotate{};
  auto* slab = reinterpret_cast<_Float16*>(workspace.input);
  for (std::size_t c = 0; c < consumers.size(); ++c) {
    if (!Exl3FastSupported(*consumers[c]) || consumers[c]->k != dim) {
      return false;
    }
    rotate.suh[c] = reinterpret_cast<const _Float16*>(consumers[c]->suh);
    rotate.out[c] = slab + c * exl3::kMaxRows * dim;
  }
  hipLaunchKernelGGL(Exl3RmsNormRotateKernel, dim3(1), dim3(256), 0, stream, x,
                     weight, normed, eps, rotate,
                     static_cast<unsigned>(consumers.size()));
  HIP_CHECK(hipGetLastError());
  return true;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
