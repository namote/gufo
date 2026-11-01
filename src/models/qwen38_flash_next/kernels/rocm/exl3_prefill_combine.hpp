#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXL3_PREFILL_COMBINE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXL3_PREFILL_COMBINE_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/hip/detail/exl3_gemv.hpp"
#include "src/core/quant/exl3/device.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

/// What the EXL3 prefill MoE combine reads: the packed FP16 raw down dots
/// (Exl3PrefillExpertsRaw), their bucket bank and inverse slot map, the
/// routing weights and the gated shared expert.
struct Exl3PrefillMoeRows {
  const quant::exl3::DeviceLinear* bank{nullptr};
  const unsigned* offsets{nullptr};
  unsigned experts{0};
  const std::int32_t* inverse{nullptr};
  const void* raw{nullptr};  ///< _Float16 [packed][width]
  const float* weights{nullptr};
  const float* shared{nullptr};
  const float* gate{nullptr};
  unsigned gate_stride{0};
  unsigned used{0};
  unsigned width{0};
};

#if defined(__HIPCC__)
namespace detail {
__device__ __forceinline__ unsigned PrefillBucketExpert(const unsigned* offsets,
                                                        unsigned experts,
                                                        unsigned row) {
  unsigned lo = 0, hi = experts;
  while (lo + 1 < hi) {
    const unsigned mid = (lo + hi) / 2;
    if (offsets[mid] <= row)
      lo = mid;
    else
      hi = mid;
  }
  return lo;
}
}  // namespace detail

/// One wave: token t's MoE output for the 128 columns from segment * 128,
/// four per lane. Each slot's raw row is restored (inverse Hadamard and
/// sign/scale vector) and weighted in slot order, then the gated shared
/// expert is added: PrefillCombineKernel's expressions with the same
/// pinned values, so the result equals its output bit for bit.
__device__ __forceinline__ void Exl3PrefillMoeSegment(
    const Exl3PrefillMoeRows& m, unsigned t, unsigned segment, unsigned lane,
    float (&out)[4]) {
  namespace arithmetic = gufo::hip::exl3;
  const unsigned first = segment * 128 + lane * 4;
  constexpr unsigned kMaxSlots = 16;
  float acc[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    acc[j] = 0.0F;
    asm volatile("" : "+v"(acc[j]));
  }
  for (unsigned slot = 0; slot < kMaxSlots; ++slot) {
    if (slot >= m.used)
      break;
    const unsigned assignment = t * m.used + slot;
    const unsigned packed = static_cast<unsigned>(m.inverse[assignment]);
    const unsigned lo =
        detail::PrefillBucketExpert(m.offsets, m.experts, packed);
    const auto* svh = reinterpret_cast<const _Float16*>(m.bank[lo].svh);
    const auto offset = std::size_t{packed} * m.width + first;
    float v[4];
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      v[j] =
          static_cast<float>(static_cast<const _Float16*>(m.raw)[offset + j]);
      asm volatile("" : "+v"(v[j]));
    }
    arithmetic::Butterfly128(v, lane);
    const float w = m.weights[assignment];
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      float restored = (v[j] * arithmetic::kHadamardScale) *
                       static_cast<float>(svh[first + j]);
      asm volatile("" : "+v"(restored));
      float a = acc[j] + w * restored;
      asm volatile("" : "+v"(a));
      acc[j] = a;
    }
  }
  const float g = m.gate[std::size_t{t} * m.gate_stride];
  const float sig = 1.0f / (1.0f + __expf(-g));
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const std::size_t idx = std::size_t{t} * m.width + first + j;
    float o = acc[j] + sig * m.shared[idx];
    // Stored by PrefillCombineKernel: no later product may fold into it.
    asm volatile("" : "+v"(o));
    out[j] = o;
  }
}
#endif

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXL3_PREFILL_COMBINE_HPP_
