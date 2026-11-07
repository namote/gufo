#ifndef GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_EXL3_GEMV_REFERENCE_HPP_
#define GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_EXL3_GEMV_REFERENCE_HPP_

// The EXL3 decode GEMV (src/core/quant/exl3/gemv.hip) as it was before B5 J1
// (commit e8e938ed): one K tile (TN output tiles) per step, every load
// waited at its use.
// Kept verbatim so tests and benches can compare the production kernel's
// output bytes against it. Compile with gemv.hip's device flags (-O2, no
// fast-math), as exl3_ops_test.cpp is.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "src/core/hip/detail/exl3_gemv.hpp"
#include "src/core/quant/exl3/device.hpp"
#include "src/core/quant/exl3/gemv.hpp"

namespace gufo::tests::exl3_gemv_reference {

using hip::exl3::Half2;
using hip::exl3::Half4;
using hip::exl3::kHadamardScale;
namespace exl3 = hip::exl3;

constexpr unsigned kMaxSlots = hip::kExl3FastMaxGroup;

// DPP quad_perm move: 0xB1 = [1,0,3,2] (lane ^ 1), 0x4E = [2,3,0,1]
// (lane ^ 2). Full-rate VALU instead of ds_bpermute, the same additions as a
// __shfl_xor reduction (bit-identical, xb3).
template<int kCtrl>
__device__ __forceinline__ float QuadMove(float v) {
  return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), kCtrl,
                                                    0xF, 0xF, false));
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
__global__ __launch_bounds__(W * 32) void ReferenceGemvKernel(GemvArgs args) {
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
inline LaunchShape PickShape(unsigned k_tiles, unsigned blocks_two_tiles) {
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

template<int K, unsigned M, int kEpilogue>
void LaunchRows(GemvArgs args, const LaunchShape& shape, hipStream_t stream) {
  const auto launch = [&]<unsigned W, unsigned TN>() {
    unsigned blocks = 0;
    for (unsigned i = 0; i < args.slots; ++i) {
      args.slot[i].block_begin = blocks;
      blocks += args.slot[i].n / 16 / TN;
    }
    hipLaunchKernelGGL((ReferenceGemvKernel<K, M, W, TN, kEpilogue>),
                       dim3(blocks), dim3(W * 32), 0, stream, args);
  };
  if (shape.tiles_per_wave == 4) {
    launch.template operator()<4, 4>();
  } else if (shape.waves == 4) {
    launch.template operator()<4, 2>();
  } else if (shape.waves == 8) {
    launch.template operator()<8, 2>();
  } else {
    launch.template operator()<16, 2>();
  }
}

// The rates the tests and benches use.
template<unsigned M, int kEpilogue>
void LaunchBits(int bits, const GemvArgs& args, const LaunchShape& shape,
                hipStream_t stream) {
  switch (bits) {
    case 3:
      LaunchRows<3, M, kEpilogue>(args, shape, stream);
      break;
    case 4:
      LaunchRows<4, M, kEpilogue>(args, shape, stream);
      break;
    case 5:
      LaunchRows<5, M, kEpilogue>(args, shape, stream);
      break;
    case 6:
      LaunchRows<6, M, kEpilogue>(args, shape, stream);
      break;
    case 7:
      LaunchRows<7, M, kEpilogue>(args, shape, stream);
      break;
    case 8:
      LaunchRows<8, M, kEpilogue>(args, shape, stream);
      break;
    default:
      throw std::invalid_argument("reference GEMV: untested rate");
  }
}

template<int kEpilogue = kStore>
void LaunchPassRows(unsigned m, int bits, const GemvArgs& args,
                    const LaunchShape& shape, hipStream_t stream) {
  switch (m) {
    case 1:
      LaunchBits<1, kEpilogue>(bits, args, shape, stream);
      break;
    case 2:
      LaunchBits<2, kEpilogue>(bits, args, shape, stream);
      break;
    case 4:
      LaunchBits<4, kEpilogue>(bits, args, shape, stream);
      break;
    default:
      LaunchBits<8, kEpilogue>(bits, args, shape, stream);
      break;
  }
}

// LaunchExl3FastProjections with input_rotated: `slabs` holds each target's
// lane-major FP16 input (Exl3FastPassRows(rows) rows, kMaxRows * k halves
// apart); `counters` are zero and hold n / 128 per target.
inline void Projections(std::span<const hip::Exl3FastTarget> targets,
                        const std::uint16_t* slabs, unsigned rows,
                        std::uint32_t* counters, hipStream_t stream,
                        std::size_t shape_basis_n = 0) {
  const auto& first = *targets[0].linear;
  GemvArgs gemv{};
  gemv.slots = static_cast<unsigned>(targets.size());
  gemv.k = first.k;
  gemv.rows = rows;
  std::size_t counter = 0;
  unsigned blocks_two_tiles = 0;
  for (std::size_t s = 0; s < targets.size(); ++s) {
    const auto& linear = *targets[s].linear;
    GemvSlot& slot = gemv.slot[s];
    slot.trellis = linear.trellis;
    slot.a =
        reinterpret_cast<const _Float16*>(slabs) + s * exl3::kMaxRows * first.k;
    slot.svh = reinterpret_cast<const _Float16*>(linear.svh);
    slot.y_stride = targets[s].y_stride;
    slot.y = targets[s].y;
    slot.counters = counters + counter;
    slot.n = linear.n;
    counter += linear.n / exl3::kSegment;
    blocks_two_tiles += linear.n / 32;
  }
  if (shape_basis_n != 0) {
    blocks_two_tiles = static_cast<unsigned>(shape_basis_n / 32);
  }
  LaunchPassRows(hip::Exl3FastPassRows(rows), first.rate.bits, gemv,
                 PickShape(first.k / 16, blocks_two_tiles), stream);
}

// LaunchExl3FastMlp with input_rotated (slab 0 gate, slab 1 up).
inline void Mlp(const hip::Exl3FastMlp& mlp, const std::uint16_t* slabs,
                unsigned rows, std::uint16_t* down_slab,
                std::uint32_t* counters, hipStream_t stream) {
  const auto& gate = *mlp.gate;
  const auto& up = *mlp.up;
  const auto& down = *mlp.down;
  const unsigned m = hip::Exl3FastPassRows(rows);
  GemvArgs gemv{};
  gemv.slots = 2;
  gemv.k = gate.k;
  gemv.rows = rows;
  gemv.down_suh = reinterpret_cast<const _Float16*>(down.suh);
  gemv.down_input = reinterpret_cast<_Float16*>(down_slab);
  const quant::exl3::DeviceLinear* pair[2] = {&gate, &up};
  float* scratch[2] = {mlp.gate_y, mlp.up_y};
  for (unsigned s = 0; s < 2; ++s) {
    GemvSlot& slot = gemv.slot[s];
    slot.trellis = pair[s]->trellis;
    slot.a =
        reinterpret_cast<const _Float16*>(slabs) + s * exl3::kMaxRows * gate.k;
    slot.svh = reinterpret_cast<const _Float16*>(pair[s]->svh);
    slot.y_stride = gate.n;
    slot.y = scratch[s];
    slot.counters = counters;
    slot.n = gate.n;
  }
  LaunchPassRows<kSwiGLUDown>(m, gate.rate.bits, gemv,
                              PickShape(gate.k / 16, 2 * (gate.n / 32)),
                              stream);
  GemvArgs down_args{};
  down_args.slots = 1;
  down_args.k = down.k;
  down_args.rows = rows;
  GemvSlot& slot = down_args.slot[0];
  slot.trellis = down.trellis;
  slot.a = reinterpret_cast<const _Float16*>(down_slab);
  slot.svh = reinterpret_cast<const _Float16*>(down.svh);
  slot.y_stride = mlp.out_stride;
  slot.y = mlp.out;
  slot.counters = counters;
  slot.n = down.n;
  LaunchPassRows(m, down.rate.bits, down_args,
                 PickShape(down.k / 16, down.n / 32), stream);
}

}  // namespace gufo::tests::exl3_gemv_reference

#endif  // GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_EXL3_GEMV_REFERENCE_HPP_
