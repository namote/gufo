#ifndef GUFO_CORE_HIP_DETAIL_HADAMARD_HPP_
#define GUFO_CORE_HIP_DETAIL_HADAMARD_HPP_

// 128-point input rotation x' = H·D·x for rotated GGUF weights
// (src/core/quant/rotation.hpp). Header-only device helpers with no Gufo
// dependencies so model kernel trees can include it directly.
//
// Arithmetic contract. Every helper produces the same bits for the same
// 128 inputs, independent of how the values are spread over lanes or LDS:
//   1. the sign multiply is an exact sign-bit flip (bit set -> negate);
//   2. butterflies run in the CPU reference order, lengths 1, 2, 4, ..., 64,
//      each pair (a, b) at (j, j + length) becoming (a + b, a - b) with one
//      rounded add or subtract;
//   3. every output is then multiplied once by kHadamard128Scale and rounded
//      to FP32 (a compiler barrier keeps a later FP16 conversion from fusing
//      with that multiply).
// The CPU reference in tests/core/quant/rotation_reference.hpp follows the
// same steps, so device results are compared bit for bit.
//
// Sign vectors are packed: bit j of word w set means element 32·w + j has
// sign -1. A rotation segment s of a row covers elements [128·s, 128·s + 128)
// and its signs are words [4·s, 4·s + 4).

#include <cstddef>
#include <cstdint>

namespace gufo::hip::hadamard {

inline constexpr unsigned kSegment = 128;
inline constexpr unsigned kSignWordsPerSegment = kSegment / 32;
// float(1 / sqrt(128)), correctly rounded.
inline constexpr float kHadamard128Scale = 0.0883883461356163F;

}  // namespace gufo::hip::hadamard

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>

namespace gufo::hip::hadamard {

__device__ __forceinline__ float ApplySign(float value, std::uint32_t word,
                                           unsigned bit) {
  const std::uint32_t flip = ((word >> bit) & 1U) << 31;
  return __uint_as_float(__float_as_uint(value) ^ flip);
}

/// In-place rotation of one 128-element segment held by kLanes = 128 / E
/// consecutive lanes (lane l owns elements E·l .. E·l + E - 1, l being the
/// lane index within the segment group). `signs` points at the segment's
/// four sign words. Works for E = 2 (wave64), 4 (wave32) and larger.
template<unsigned E>
__device__ __forceinline__ void RotateSegmentLanes(
    float (&v)[E], const std::uint32_t* __restrict__ signs, unsigned lane) {
  static_assert(E >= 1 && E <= kSegment && (kSegment % E) == 0 &&
                (E & (E - 1)) == 0);
  constexpr unsigned kLanes = kSegment / E;
  const unsigned first = lane * E;
#pragma unroll
  for (unsigned j = 0; j < E; ++j) {
    const unsigned element = first + j;
    v[j] = ApplySign(v[j], signs[element / 32], element % 32);
  }
#pragma unroll
  for (unsigned length = 1; length < E; length <<= 1) {
#pragma unroll
    for (unsigned i = 0; i < E; i += length << 1) {
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
  for (unsigned mask = 1; mask < kLanes; mask <<= 1) {
#pragma unroll
    for (unsigned j = 0; j < E; ++j) {
      const float other =
          __shfl_xor(v[j], static_cast<int>(mask), static_cast<int>(kLanes));
      // The lower element of the pair keeps a + b; the upper takes a - b
      // with a from the lower lane.
      v[j] = (lane & mask) != 0 ? other - v[j] : v[j] + other;
    }
  }
#pragma unroll
  for (unsigned j = 0; j < E; ++j) {
    v[j] *= kHadamard128Scale;
    // Pin the FP32 rounding of the scaled value. Without this register
    // dependency LLVM may fold the multiply into a following FP16 conversion
    // (v_fma_mix*), rounding the exact product once and moving FP16 ties.
    asm volatile("" : "+v"(v[j]));
  }
}

/// Same rotation for the strided layout: lane l of kLanes = 128 / E lanes
/// owns elements l + kLanes·j, j = 0..E-1. Lengths below kLanes pair lanes,
/// the rest pair registers; the stage order and arithmetic are unchanged.
template<unsigned E>
__device__ __forceinline__ void RotateSegmentStrided(
    float (&v)[E], const std::uint32_t* __restrict__ signs, unsigned lane) {
  static_assert(E >= 1 && E <= kSegment && (kSegment % E) == 0 &&
                (E & (E - 1)) == 0);
  constexpr unsigned kLanes = kSegment / E;
#pragma unroll
  for (unsigned j = 0; j < E; ++j) {
    const unsigned element = lane + kLanes * j;
    v[j] = ApplySign(v[j], signs[element / 32], element % 32);
  }
#pragma unroll
  for (unsigned mask = 1; mask < kLanes; mask <<= 1) {
#pragma unroll
    for (unsigned j = 0; j < E; ++j) {
      const float other =
          __shfl_xor(v[j], static_cast<int>(mask), static_cast<int>(kLanes));
      v[j] = (lane & mask) != 0 ? other - v[j] : v[j] + other;
    }
  }
#pragma unroll
  for (unsigned length = 1; length < E; length <<= 1) {
#pragma unroll
    for (unsigned i = 0; i < E; i += length << 1) {
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
  for (unsigned j = 0; j < E; ++j) {
    v[j] *= kHadamard128Scale;
    // Pin the FP32 rounding of the scaled value. Without this register
    // dependency LLVM may fold the multiply into a following FP16 conversion
    // (v_fma_mix*), rounding the exact product once and moving FP16 ties.
    asm volatile("" : "+v"(v[j]));
  }
}

/// Rotates whole segments of `row` in place with one wave per segment.
/// `row` is usually LDS; `signs` covers the same row. All threads of the
/// block must call this; the caller synchronizes before and after.
template<unsigned WaveSize = 32>
__device__ __forceinline__ void RotateRowSegments(
    float* row, std::size_t dim, const std::uint32_t* __restrict__ signs,
    unsigned thread, unsigned threads) {
  constexpr unsigned E = kSegment / WaveSize;
  const unsigned wave = thread / WaveSize;
  const unsigned lane = thread % WaveSize;
  const unsigned waves = threads / WaveSize;
  const std::size_t segments = dim / kSegment;
  for (std::size_t s = wave; s < segments; s += waves) {
    float* segment = row + s * kSegment;
    float v[E];
#pragma unroll
    for (unsigned j = 0; j < E; ++j) {
      v[j] = segment[lane * E + j];
    }
    RotateSegmentLanes<E>(v, signs + s * kSignWordsPerSegment, lane);
#pragma unroll
    for (unsigned j = 0; j < E; ++j) {
      segment[lane * E + j] = v[j];
    }
  }
}

}  // namespace gufo::hip::hadamard
#endif  // defined(__HIPCC__)

#endif  // GUFO_CORE_HIP_DETAIL_HADAMARD_HPP_
