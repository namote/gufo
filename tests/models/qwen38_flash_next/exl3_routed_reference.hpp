#ifndef GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_EXL3_ROUTED_REFERENCE_HPP_
#define GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_EXL3_ROUTED_REFERENCE_HPP_

// Frozen copy of the routed expert decode GEMV (src/models/qwen38_flash_next/
// kernels/rocm/exl3_moe.hip.cpp RoutedGemv) at commit e8e938ed: one output tile
// per wave, MaxRows 3/4/8 kernels with the route tile promoted to LDS.
// Production now keeps the route tile in registers and gives each down wave
// two output tiles at R <= 4, without changing any lane's arithmetic; this
// copy pins that contract, so tests compare the production kernel's output
// bytes against it. Kept verbatim; compile with exl3_moe.hip.cpp's device flags
// (-O3, -ffast-math, -fno-finite-math-only), as exl3_moe_kernel_test.cpp is.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "src/core/hip/detail/exl3_gemv.hpp"
#include "src/core/quant/exl3/device.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/exl3_moe.hpp"

namespace gufo::tests::exl3_routed_reference {

namespace ex = quant::exl3;
namespace arithmetic = gufo::hip::exl3;
using arithmetic::Half2;
using models::qwen38_flash_next::rocm::Exl3RouteTile;

// The down slab can be entirely FP16 subnormal after SwiGLU. Widen its
// operands before arithmetic and pin the boundary so LLVM cannot fold it
// back into half/mixed instructions under this TU's fast-math flags.
__device__ __forceinline__ float DotPairF32(Half2 w, Half2 x, float acc) {
  float w0 = static_cast<float>(w.x);
  float w1 = static_cast<float>(w.y);
  float x0 = static_cast<float>(x.x);
  float x1 = static_cast<float>(x.y);
  asm volatile("" : "+v"(w0), "+v"(w1), "+v"(x0), "+v"(x1));
  acc = __builtin_fmaf(w0, x0, acc);
  return __builtin_fmaf(w1, x1, acc);
}

// Waves per routed GEMV block: adjacent output tiles of the same expert
// tile. Each wave's arithmetic is the former one-wave block's; packing only
// cuts the block count (40 x 10 x 2 gate/up and 160 x 10 down blocks at
// R=1 became a quarter as many).
constexpr unsigned kRoutedWaves = 4;

// FP16 activation halves a routed GEMV block stages in LDS per K chunk:
// a whole 2560-wide row per tile row, capped so a four-wave block keeps 16
// blocks (64 waves) resident per WGP.
constexpr unsigned kRoutedStageRowHalves = 2560;
constexpr unsigned kRoutedStageHalves = 4096;
constexpr unsigned RoutedStageHalves(unsigned rows) {
  return rows * kRoutedStageRowHalves < kRoutedStageHalves
             ? rows * kRoutedStageRowHalves
             : kRoutedStageHalves;
}

// One wave owns one output tile and up to eight actual rows of one expert.
// No floating-point atomics or masked reads from absent assignment rows.
// The block stages its rows' activations in LDS (`stage`,
// RoutedStageHalves(MaxRows) halves) once per K chunk, so the K loop issues
// only trellis global loads: an activation global load at its use made the
// wave wait for the next tile group's loads too (vmcnt counts in order).
// Every wave of the block reaches the staging barriers; a wave past the
// last output tile computes a duplicate of it and stores nothing.
template<int Bits, bool FullPrecision, unsigned MaxRows = 8,
         bool Scaled = false, bool ExactRows = false>
__device__ __forceinline__ void RoutedGemvBody(
    const ex::DeviceLinear* gate, const ex::DeviceLinear* up,
    const Exl3RouteTile* tiles, const unsigned* count,
    const std::uint16_t* input, float* gate_y, float* up_y, unsigned k,
    unsigned n, unsigned assignments, const std::int32_t* exponents,
    _Float16* stage) {
  if (blockIdx.y >= *count)
    return;
  const auto tile_rows = tiles[blockIdx.y];
  const unsigned projection = blockIdx.z;
  const auto& l = (projection == 0 ? gate : up)[tile_rows.expert];
  const unsigned lane = threadIdx.x % 32;
  const unsigned r0 = (lane % 4) * 2;
  const unsigned n_tiles = n / 16;
  const unsigned wave_tile = blockIdx.x * kRoutedWaves + threadIdx.x / 32;
  const bool live = wave_tile < n_tiles;
  const unsigned nt = live ? wave_tile : n_tiles - 1;
  if (n_tiles == 0)
    return;
  float a[MaxRows]{}, b[MaxRows]{};
  const auto lw = arithmetic::MakeLaneWindow<Bits>(lane);
  const auto* trellis = reinterpret_cast<const std::uint32_t*>(l.trellis);
  const auto* x = reinterpret_cast<const _Float16*>(input) +
                  std::size_t{projection} * assignments * k;
  // Tiles are fetched kAhead at a time so a wave keeps several loads in
  // flight (one tile per iteration left decode latency-bound, ~150 GB/s);
  // they are decoded and accumulated in K order exactly as before. The
  // next group's loads are issued before this group is decoded, so the
  // wave never drains to zero loads in flight at a group boundary (the
  // short K = 640 down projection has ten such boundaries per row). Indices
  // past the end clamp to the last tile, already cached by this group.
  constexpr unsigned kAhead = 4;
  const unsigned k_tiles = k / 16;
  // Global, not flat, loads: a flat load also counts against lgkmcnt, so a
  // scalar wait in the loop would drain the group still in flight.
  using GlobalDword = const __attribute__((address_space(1))) std::uint32_t;
  const auto fetch = [&](unsigned kt0, std::uint32_t (&raw)[kAhead][3]) {
#pragma unroll
    for (unsigned u = 0; u < kAhead; ++u) {
      const unsigned kt = min(kt0 + u, k_tiles - 1);
      // An address-space conversion is spelled as a C cast.
      const auto* tile =
          (GlobalDword*)(trellis + (std::size_t{kt} * (n / 16) + nt) *
                                       arithmetic::TileDwords(Bits));
      raw[u][0] = tile[lw.o0];
      raw[u][1] = tile[lw.o1];
      raw[u][2] = tile[lw.o2];
    }
  };
  if (k_tiles == 0)
    return;
  // K tiles per staged chunk (a whole number of fetch groups) and the
  // block's staging work: 16-byte vectors, two per tile row, spread over
  // the block's threads (at most four per thread). The compiler sinks each
  // load into its guarded store, a round trip per vector; issuing them all
  // first kept 30-50 more VGPRs live through the K loop at R3-R8.
  constexpr unsigned kChunk =
      RoutedStageHalves(MaxRows) / (MaxRows * 16) / kAhead * kAhead;
  constexpr unsigned kRowHalves = kChunk * 16;
  constexpr unsigned kThreads = 32 * kRoutedWaves;
  constexpr unsigned kPerThread =
      (MaxRows * kChunk * 2 + kThreads - 1) / kThreads;
  static_assert(kChunk >= kAhead && MaxRows * kRowHalves <= kRoutedStageHalves);
  const unsigned rows = ExactRows ? MaxRows : tile_rows.count;
  const auto stage_chunk = [&](unsigned base) {
    const unsigned vectors = min(kChunk, k_tiles - base) * 2;
    const unsigned total = rows * vectors;
    uint4 data[kPerThread];
    __syncthreads();  // every wave is done with the previous chunk
#pragma unroll
    for (unsigned j = 0; j < kPerThread; ++j) {
      const unsigned i = min(threadIdx.x + j * kThreads, total - 1);
      const unsigned r = i / vectors;
      unsigned slot = tile_rows.slots[0];
      // Fixed indices: a dynamically indexed slot array promotes to LDS.
#pragma unroll
      for (unsigned q = 1; q < MaxRows; ++q)
        slot = r == q ? tile_rows.slots[q] : slot;
      data[j] = reinterpret_cast<const uint4*>(x + std::size_t{slot} * k +
                                               base * 16)[i % vectors];
    }
#pragma unroll
    for (unsigned j = 0; j < kPerThread; ++j) {
      const unsigned i = threadIdx.x + j * kThreads;
      if (i < total)
        reinterpret_cast<uint4*>(stage + (i / vectors) *
                                             kRowHalves)[i % vectors] = data[j];
    }
    __syncthreads();
  };
  // A row's widened-FMA choice is fixed for the whole K loop.
  bool widened[MaxRows];
#pragma unroll
  for (unsigned r = 0; r < MaxRows; ++r)
    widened[r] =
        FullPrecision || (Scaled && (ExactRows || r < tile_rows.count) &&
                          exponents[tile_rows.slots[r]] < 0);
  // Two register sets alternate, so the group in flight is never copied
  // (a copy at the loop edge waited for it before the next group began).
  std::uint32_t even[kAhead][3], odd[kAhead][3];
  // The K-end test stays in every group: unguarded groups let the compiler
  // decode a whole group at once (30-40 more VGPRs at R3/R4). The loop edge
  // then makes each pair's first group wait for its own loads before it
  // issues the next group's; no tile waits for loads issued after it.
  const auto group = [&](unsigned kt0, const std::uint32_t (&raw)[kAhead][3],
                         std::uint32_t (
                             &next)[kAhead][3]) __attribute__((always_inline)) {
    if (kt0 % kChunk == 0)
      stage_chunk(kt0);
    const unsigned chunk_base = kt0 - kt0 % kChunk;
    fetch(kt0 + kAhead, next);
#pragma unroll
    for (unsigned u = 0; u < kAhead; ++u) {
      const unsigned kt = kt0 + u;
      if (kt >= k_tiles)
        break;
      Half2 w[4];
      {
        std::uint32_t state[8];
        arithmetic::LaneStates<Bits>(raw[u][0], raw[u][1], raw[u][2], lw.shift,
                                     state);
        std::uint32_t p[8];
#pragma unroll
        for (int j = 0; j < 8; ++j)
          p[j] = (state[j] & 0xffffU) * arithmetic::kMul1Multiplier;
#pragma unroll
        for (int q = 0; q < 4; ++q)
          w[q] = arithmetic::DecodePairMul1(p[2 * q], p[2 * q + 1]);
      }
      // Fixed row indices keep the accumulators and route slots in VGPRs.
      // A dynamically indexed local array otherwise promotes to 40 KiB LDS
      // per one-wave block on the pinned compiler, starving verification.
#pragma unroll
      for (unsigned r = 0; r < MaxRows; ++r) {
        if (!ExactRows && r >= tile_rows.count)
          continue;
        const auto* row = stage + r * kRowHalves + (kt - chunk_base) * 16 + r0;
        const Half2 v01 = *reinterpret_cast<const Half2*>(row);
        const Half2 v89 = *reinterpret_cast<const Half2*>(row + 8);
        if (widened[r]) {
          a[r] = DotPairF32(w[0], v01, a[r]);
          a[r] = DotPairF32(w[1], v89, a[r]);
          b[r] = DotPairF32(w[2], v01, b[r]);
          b[r] = DotPairF32(w[3], v89, b[r]);
        } else {
          a[r] = __builtin_amdgcn_fdot2(w[0], v01, a[r], false);
          a[r] = __builtin_amdgcn_fdot2(w[1], v89, a[r], false);
          b[r] = __builtin_amdgcn_fdot2(w[2], v01, b[r], false);
          b[r] = __builtin_amdgcn_fdot2(w[3], v89, b[r], false);
        }
      }
    }
  };
  fetch(0, even);
  for (unsigned kt0 = 0; kt0 < k_tiles; kt0 += 2 * kAhead) {
    group(kt0, even, odd);
    if (kt0 + kAhead >= k_tiles)
      break;
    group(kt0 + kAhead, odd, even);
  }
#pragma unroll
  for (unsigned r = 0; r < MaxRows; ++r) {
    if (!ExactRows && r >= tile_rows.count)
      continue;
    a[r] += __shfl_xor(a[r], 1, 32);
    b[r] += __shfl_xor(b[r], 1, 32);
    a[r] += __shfl_xor(a[r], 2, 32);
    b[r] += __shfl_xor(b[r], 2, 32);
    if (live && lane % 4 == 0) {
      auto* y = (projection == 0 ? gate_y : up_y) +
                std::size_t{tile_rows.slots[r]} * n + nt * 16;
      float scale = 1.0F;
      if constexpr (Scaled) {
        const int exponent = exponents[tile_rows.slots[r]];
        if (exponent > 0)
          scale = __builtin_bit_cast(float, unsigned(127 - exponent) << 23);
      }
      y[lane / 4] = a[r] * scale;
      y[lane / 4 + 8] = b[r] * scale;
    }
  }
}

// Select the live-row loop once per expert tile, avoiding absent-row checks
// in every K step. Rate-specific decoding stays in the body; only the row
// loop is specialized, independently of the bit rate.
// The generic body remains for MaxRows 1/2 and unscaled full-precision down.
template<int Bits, bool FullPrecision, unsigned MaxRows = 8,
         bool Scaled = false>
__global__ void RoutedGemv(const ex::DeviceLinear* gate,
                           const ex::DeviceLinear* up,
                           const Exl3RouteTile* tiles, const unsigned* count,
                           const std::uint16_t* input, float* gate_y,
                           float* up_y, unsigned k, unsigned n,
                           unsigned assignments,
                           const std::int32_t* exponents) {
  if (blockIdx.y >= *count)
    return;
  // One staging buffer for whichever row body the tile selects.
  __shared__
      __attribute__((aligned(16))) _Float16 stage[RoutedStageHalves(MaxRows)];
  if constexpr (!FullPrecision &&
                (MaxRows == 3 || MaxRows == 4 || MaxRows == 8)) {
    // Each live-row body runs exactly the generic body's arithmetic for the
    // tile's rows; R5-R8 verify tiles (MaxRows 8) mostly hold 1-2 rows.
    const auto exact = [&]<unsigned Rows>() __attribute__((always_inline)) {
      RoutedGemvBody<Bits, FullPrecision, Rows, Scaled, true>(
          gate, up, tiles, count, input, gate_y, up_y, k, n, assignments,
          exponents, stage);
    };
    switch (tiles[blockIdx.y].count) {
      case 1:
        exact.template operator()<1>();
        return;
      case 2:
        exact.template operator()<2>();
        return;
      case 3:
        exact.template operator()<3>();
        return;
      case 4:
        if constexpr (MaxRows >= 4) {
          exact.template operator()<4>();
          return;
        }
        break;
      case 5:
        if constexpr (MaxRows == 8) {
          exact.template operator()<5>();
          return;
        }
        break;
      case 6:
        if constexpr (MaxRows == 8) {
          exact.template operator()<6>();
          return;
        }
        break;
      case 7:
        if constexpr (MaxRows == 8) {
          exact.template operator()<7>();
          return;
        }
        break;
      case 8:
        if constexpr (MaxRows == 8) {
          exact.template operator()<8>();
          return;
        }
        break;
      default:
        break;
    }
  }
  RoutedGemvBody<Bits, FullPrecision, MaxRows, Scaled>(
      gate, up, tiles, count, input, gate_y, up_y, k, n, assignments, exponents,
      stage);
}

// LaunchRouted's grid for the old kernel: `assignments` route tiles of
// at most R rows, `projections` (1, or 2 for gate and up).
template<int Bits, bool FullPrecision, unsigned R, bool Scaled>
void Launch(const ex::DeviceLinear* gate, const ex::DeviceLinear* up,
            const Exl3RouteTile* tiles, const unsigned* count,
            const std::uint16_t* input, float* gate_y, float* up_y, unsigned k,
            unsigned n, unsigned assignments, unsigned projections,
            const std::int32_t* exponents, hipStream_t stream) {
  hipLaunchKernelGGL((RoutedGemv<Bits, FullPrecision, R, Scaled>),
                     dim3((n / 16 + kRoutedWaves - 1) / kRoutedWaves,
                          assignments, projections),
                     dim3(32 * kRoutedWaves), 0, stream, gate, up, tiles, count,
                     input, gate_y, up_y, k, n, assignments, exponents);
}

// Launch for a tile-row cap (1..8, as Exl3RouteTileRows) and the rates the
// tests use (K3.5, K4, K5).
template<bool FullPrecision, bool Scaled>
void LaunchRate(ex::BitRate rate, unsigned tile_rows,
                const ex::DeviceLinear* gate, const ex::DeviceLinear* up,
                const Exl3RouteTile* tiles, const unsigned* count,
                const std::uint16_t* input, float* gate_y, float* up_y,
                unsigned k, unsigned n, unsigned assignments,
                unsigned projections, const std::int32_t* exponents,
                hipStream_t stream) {
  const auto rows = [&]<int Bits>() {
    const auto go = [&]<unsigned R>() {
      Launch<Bits, FullPrecision, R, Scaled>(gate, up, tiles, count, input,
                                             gate_y, up_y, k, n, assignments,
                                             projections, exponents, stream);
    };
    switch (tile_rows) {
      case 1:
        go.template operator()<1>();
        break;
      case 2:
        go.template operator()<2>();
        break;
      case 3:
        go.template operator()<3>();
        break;
      case 4:
        go.template operator()<4>();
        break;
      default:
        go.template operator()<8>();
        break;
    }
  };
  if (rate.half && rate.bits == 3)
    rows.template operator()<arithmetic::RateCode(3, true)>();
  else if (!rate.half && rate.bits == 4)
    rows.template operator()<4>();
  else if (!rate.half && rate.bits == 5)
    rows.template operator()<5>();
  else
    throw std::invalid_argument("routed reference: untested rate");
}

}  // namespace gufo::tests::exl3_routed_reference

#endif  // GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_EXL3_ROUTED_REFERENCE_HPP_
