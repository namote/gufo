#ifndef GUFO_CORE_HIP_DETAIL_EXL3_GEMV_HPP_
#define GUFO_CORE_HIP_DETAIL_EXL3_GEMV_HPP_

// Device helpers of the EXL3 trellis decode GEMV (plan/exl3/XB-DESIGN.md
// section 3.1, measured in plan/exl3/XB-RESULTS.md windows xb1 and xb2).
// Header-only, model independent.
//
// Tile decode. Lane L of a wave decodes the positions 8L .. 8L + 7 of a 16 x
// 16 tile. The tensor-core permutation of the format (docs/EXL3-FORMAT.md
// "Tile layout") puts them at rows (r0, r0+1, r0+8, r0+9), r0 = (L % 4) * 2,
// of columns L / 4 (positions 0-3) and L / 4 + 8 (positions 4-7). For
// K <= 6 the lane's eight 16-bit states lie in one 64-bit window of the tile
// bit string, cut out of three dwords with two v_alignbit_b32; each state is
// then a constant shift of the window (the last state of K7 and K8 also
// takes low bits from the third dword). The decode is exact integer arithmetic
// plus one fused FP16 FMA, bit for bit the CPU reference
// (src/core/quant/exl3/format.hpp DequantTile).
//
// The lane layout and the mul1 pair decode (v_sad_u8 / v_sad_hi_u8 byte sums
// packed straight into an FP16 pair) follow upstream exllamav3 v1.6.5
// exllamav3_ext/rocm/quant/exl3_gemv_kernel_rdna.cuh and
// exl3_gemv_tiles_rdna.cuh (MIT License, Copyright (c) 2025 Turboderp); the
// code is written anew.
//
// Input rotation arithmetic (the producer side, FP32): x * suh rounded once,
// the unnormalized butterflies of Butterfly128 (lengths 1, 2, ..., 64, one
// rounded add or subtract per pair), one multiply by kHadamardScale, then FP16
// round to nearest even. Each product is pinned by a register barrier so it
// cannot fuse with a neighbouring add or conversion.
//
// Rotated input layout ("lane-major"). For an M-row input of width k, each
// 16-element k tile kt occupies 16 * M halves at kt * 16 * M: four groups
// q = 0..3 (the lanes with r0 = 2q), each [M rows][x[2q], x[2q+1], x[2q+8],
// x[2q+9]], so a lane reads its M rows of a tile as one contiguous 8M bytes.

#include <cstddef>
#include <cstdint>

namespace gufo::hip::exl3 {

/// Rows of one GEMV pass (the verify width of DFlash2: anchor + 7 drafts).
inline constexpr unsigned kMaxRows = 8;
/// Output columns per Hadamard segment and per counter.
inline constexpr unsigned kSegment = 128;
/// float(1 / sqrt(128)), correctly rounded.
inline constexpr float kHadamardScale = 0.0883883461356163F;

/// Compile-time rate codes of the templated trellis kernels: K for an integer
/// rate K (1..8), kHalfRate + K for the half-integer rate K + 0.5 (mul1,
/// K = 1..3; the format's mask 0xAAAA: even positions take K bits, odd
/// positions K + 1, docs/EXL3-FORMAT.md "Positions").
inline constexpr int kHalfRate = 16;
constexpr int RateCode(int bits, bool half) noexcept {
  return half ? kHalfRate + bits : bits;
}
constexpr int RateBits(int code) noexcept {
  return code % kHalfRate;
}
constexpr bool RateHalf(int code) noexcept {
  return code > kHalfRate;
}
/// Dwords of one 256-weight tile (8 K, or 8 K + 4 for K + 0.5).
constexpr unsigned TileDwords(int code) noexcept {
  return 8U * static_cast<unsigned>(RateBits(code)) +
         (RateHalf(code) ? 4U : 0U);
}
/// Exclusive end bit of tile position `p` (src/core/quant/exl3 StateEndBit).
constexpr int StateEnd(int code, int p) noexcept {
  const int k = RateBits(code);
  if (!RateHalf(code))
    return (p + 1) * k;
  return (p / 2) * (2 * k + 1) + (p % 2 == 0 ? k : 2 * k + 1);
}

/// Offset of element `i` of row `r` in an M-row lane-major input.
constexpr std::size_t LaneMajorOffset(std::size_t i, unsigned r, unsigned m) {
  const std::size_t kt = i / 16;
  const unsigned w = static_cast<unsigned>(i % 16);
  const unsigned q = (w & 7U) >> 1;
  const unsigned pos = (w & 1U) + ((w >> 3) << 1);
  return kt * 16 * m + q * 4 * m + r * 4 + pos;
}

}  // namespace gufo::hip::exl3

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>

namespace gufo::hip::exl3 {

using Half2 = _Float16 __attribute__((ext_vector_type(2)));
using Half4 = _Float16 __attribute__((ext_vector_type(4)));

inline constexpr std::uint32_t kMul1Multiplier = 0x83DCD12DU;

/// Lane-constant placement of the lane's state window in a tile of
/// TileDwords(Code) dwords: the three source dwords and 32 minus the bit
/// offset of the first state's window. That offset is never 0: for an integer
/// K it is (8L + 1) K - 16 (always 24 for K8); for K + 0.5 it is
/// 4L (2K + 1) + K - 16, odd for K1, K3 and 2 mod 4 for K2. The lane's eight
/// states span 7K + 16 bits (integer, at most 72 for K8, ending exactly at the
/// third dword's end) or 7K + 20 bits (K + 0.5, at most 41): always inside
/// the three dwords.
struct LaneWindow {
  unsigned o0, o1, o2;
  unsigned shift;
};

template<int Code>
__device__ __forceinline__ LaneWindow MakeLaneWindow(unsigned lane) {
  static_assert(
      (!RateHalf(Code) && Code >= 1 && Code <= 8) ||
          (RateHalf(Code) && RateBits(Code) >= 1 && RateBits(Code) <= 3),
      "supported rates are K1 through K8 and K1.5 through K3.5");
  constexpr int kTileBits = 32 * static_cast<int>(TileDwords(Code));
  constexpr unsigned kWords = TileDwords(Code);
  const int start =
      (StateEnd(Code, 8 * static_cast<int>(lane)) - 16 + kTileBits) % kTileBits;
  const unsigned i0 = static_cast<unsigned>(start) / 32;
  LaneWindow w;
  w.o0 = i0;
  w.o1 = (i0 + 1) % kWords;
  w.o2 = (i0 + 2) % kWords;
  w.shift = 32U - static_cast<unsigned>(start) % 32;
  return w;
}

/// x * kMul1Multiplier mod 2^32 for the 16-bit state in the low half of
/// src, as two full-rate ops (v_pk_mul_lo_u16 + v_mad_u32_u16, after upstream
/// exllamav3 exl3_gemv_tiles_rdna.cuh); the upper half of src is ignored.
/// Equal to (src & 0xffff) * kMul1Multiplier for every state (checked over
/// all 65,536 states by tools/bench/exl3_prefill_bench and the kernel test).
__device__ __forceinline__ std::uint32_t HashMul1Packed(std::uint32_t src) {
  constexpr std::uint32_t k_hi = kMul1Multiplier & 0xffff0000U;
  constexpr std::uint32_t c_lo = kMul1Multiplier & 0xffffU;
  std::uint32_t r;
  std::uint32_t p;
  asm("v_pk_mul_lo_u16 %0, %1, %2 op_sel_hi:[0,1]"
      : "=v"(r)
      : "v"(src), "s"(k_hi));
  asm("v_mad_u32_u16 %0, %1, %2, %3" : "=v"(p) : "v"(src), "s"(c_lo), "v"(r));
  return p;
}

/// Two mul1-hashed states -> the decoded FP16 pair (lo = first state):
/// fma(1024 + bytesum, 0x1eee, 0xc931) rounded once.
__device__ __forceinline__ Half2 DecodePairMul1(std::uint32_t p0,
                                                std::uint32_t p1) {
  std::uint32_t s = __builtin_amdgcn_sad_u8(p0, 0U, 0x64006400U);
  s = __builtin_amdgcn_sad_hi_u8(p1, 0U, s);
  const Half2 u = __builtin_bit_cast(Half2, s);
  const Half2 inv = __builtin_bit_cast(Half2, 0x1eee1eeeU);
  const Half2 bias = __builtin_bit_cast(Half2, 0xc931c931U);
  return __builtin_elementwise_fma(u, inv, bias);
}

/// The lane's eight 16-bit states (in the low halves; the upper halves are
/// arbitrary) from its three dwords: state j is the 16 bits ending at
/// StateEnd(Code, 8L + j) - StateEnd(Code, 8L) + 16 bits past the window
/// offset in d0:d1:d2, most significant bit first (bits [off + Kj, off + Kj +
/// 16) for an integer K). Every state lies in the 64-bit window cut out by two
/// v_alignbit_b32, except the last state of K7 (K8), which also needs window
/// bit 64 (bits 64 .. 71): the d2 bits just below the window offset.
template<int Code>
__device__ __forceinline__ void LaneStates(std::uint32_t d0, std::uint32_t d1,
                                           std::uint32_t d2, unsigned shift,
                                           std::uint32_t (&state)[8]) {
  const std::uint32_t hi = __builtin_amdgcn_alignbit(d0, d1, shift);
  const std::uint32_t lo = __builtin_amdgcn_alignbit(d1, d2, shift);
  const std::uint64_t window = (static_cast<std::uint64_t>(hi) << 32) | lo;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int end = StateEnd(Code, j) - StateEnd(Code, 0) + 16;
    if (64 - end >= 0) {
      state[j] = static_cast<std::uint32_t>(window >> (64 - end));
    } else {
      // K7 (K8), j = 7: window bits 49 (56) .. 63, then `extra` bits from
      // 64: d2's bits just below the offset (shift = 32 - offset >= extra;
      // K8 has offset 24, shift 8).
      const int extra = end - 64;
      state[j] =
          (lo << extra) | ((d2 >> (shift - extra)) & ((1U << extra) - 1U));
    }
  }
}

/// The lane's eight decoded weights as four pairs: w[0] rows (r0, r0+1) and
/// w[1] rows (r0+8, r0+9) of column L / 4; w[2], w[3] the same rows of
/// column L / 4 + 8.
template<int Code>
__device__ __forceinline__ void DecodeLaneMul1(
    const std::uint32_t* __restrict__ tile, const LaneWindow& lw,
    Half2 (&w)[4]) {
  std::uint32_t state[8];
  LaneStates<Code>(tile[lw.o0], tile[lw.o1], tile[lw.o2], lw.shift, state);
  std::uint32_t p[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    p[j] = (state[j] & 0xffffU) * kMul1Multiplier;
  }
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    w[q] = DecodePairMul1(p[2 * q], p[2 * q + 1]);
  }
}

/// In-place unnormalized 128-point Walsh-Hadamard butterflies of one segment
/// held four consecutive elements per lane (lane l owns 4l .. 4l + 3), in the
/// CPU reference order: lengths 1, 2 in the lane, then 4 .. 64 across lanes.
__device__ __forceinline__ void Butterfly128(float (&v)[4], unsigned lane) {
#pragma unroll
  for (unsigned length = 1; length < 4; length <<= 1) {
#pragma unroll
    for (unsigned i = 0; i < 4; i += length << 1) {
#pragma unroll
      for (unsigned j = i; j < i + length; ++j) {
        const float a = v[j];
        const float b = v[j + length];
        v[j] = a + b;
        v[j + length] = a - b;
      }
    }
  }
#pragma unroll
  for (unsigned mask = 1; mask < 32; mask <<= 1) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float other = __shfl_xor(v[j], static_cast<int>(mask), 32);
      v[j] = (lane & mask) != 0 ? other - v[j] : v[j] + other;
    }
  }
}

/// The input rotation of one segment held four consecutive elements per lane
/// (v[j] is element first + j, first = segment * 128 + 4 * lane): stores
/// FP16(H(suh * v)) at row `row` of an M-row lane-major slab with the
/// producer arithmetic above, bit for bit the standalone rotation kernels'.
/// Producers that already hold the FP32 row call this instead of a separate
/// rotation launch rereading it.
__device__ __forceinline__ void RotateInputSegment(float (&v)[4], unsigned lane,
                                                   unsigned first,
                                                   const _Float16* suh,
                                                   _Float16* slab, unsigned row,
                                                   unsigned m) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    // The producer's value as stored: its own products must not fuse with
    // the suh product (fast-math producer TUs).
    asm volatile("" : "+v"(v[j]));
    v[j] = v[j] * static_cast<float>(suh[first + j]);
    asm volatile("" : "+v"(v[j]));
  }
  Butterfly128(v, lane);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] *= kHadamardScale;
    asm volatile("" : "+v"(v[j]));
    slab[LaneMajorOffset(first + j, row, m)] = static_cast<_Float16>(v[j]);
  }
}

}  // namespace gufo::hip::exl3
#endif  // defined(__HIPCC__)

#endif  // GUFO_CORE_HIP_DETAIL_EXL3_GEMV_HPP_
