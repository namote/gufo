#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_ABLITERATION_ROW_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_ABLITERATION_ROW_HPP_

// The runtime abliteration row projection shared by the Qwen3.8-27B and
// Qwen3.8-Flash-Next kernels. Device code only; include from HIP sources.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/abliteration.hpp"

namespace gufo::models::qwen::hip_detail {

/// Threads per row block; the reductions below assume exactly this many.
inline constexpr unsigned kAbliterationRowThreads = 256;

/// One block projects one `hidden`-float row in place with a session's
/// device settings (AbliterationSettings::DeviceWords):
///
///   y -= w * sum_j d_j * dot(d_j, y),   j < count
///
/// over `count` consecutive orthonormal rows d_j of `directions` starting at
/// the packed row, then, with norm preservation, y *= |y_before| / |y_after|.
/// Every dot product is computed from the unprojected row in one pass, with
/// FP32 per-thread sums and a fixed tree reduction, so results are
/// deterministic. One direction without norm preservation (word 0 is the
/// bare row) runs the single-direction code verbatim, block-uniformly, so
/// its output stays bit-identical to the single-direction kernel even where
/// the library compiles with -ffast-math. Strength 0 returns before touching
/// the row.
/// The block-shared scratch of AbliterateRowIn. Slots 0..kMax-1 of
/// `partial`: dot products; kMax: |y|^2 before; kMax + 1: after. A kernel
/// with other row paths declares it once and shares it between them.
using AbliterationRowPartial =
    float[kAbliterationMaxDirections + 2][kAbliterationRowThreads];
using AbliterationRowSingle = float[kAbliterationRowThreads];

__device__ inline void AbliterateRowIn(AbliterationRowPartial& partial,
                                       AbliterationRowSingle& single,
                                       float* __restrict__ row,
                                       const float* __restrict__ directions,
                                       const float* __restrict__ settings,
                                       std::uint32_t weight_index,
                                       std::uint32_t hidden) {
  constexpr unsigned kMax = kAbliterationMaxDirections;
  const float weight = settings[weight_index];
  if (weight == 0)
    return;
  const std::uint32_t word = __float_as_uint(settings[0]);
  if (word <= kAbliterationRowMask) {
    // The single-direction projection, unchanged.
    const float* direction = directions + std::size_t{word} * hidden;
    float sum = 0;
    for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x)
      sum += row[i] * direction[i];
    single[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned step = kAbliterationRowThreads / 2; step; step >>= 1) {
      if (threadIdx.x < step)
        single[threadIdx.x] += single[threadIdx.x + step];
      __syncthreads();
    }
    const float amount = weight * single[0];
    for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x)
      row[i] -= amount * direction[i];
    return;
  }
  const unsigned count = ((word >> kAbliterationCountShift) & (kMax - 1)) + 1;
  const bool preserve = (word & kAbliterationPreserveNormBit) != 0;
  const float* direction =
      directions + std::size_t{word & kAbliterationRowMask} * hidden;
  const unsigned t = threadIdx.x;
  float sums[kMax];
#pragma unroll
  for (unsigned j = 0; j < kMax; ++j)
    sums[j] = 0;
  float before = 0;
  for (std::uint32_t i = t; i < hidden; i += blockDim.x) {
    const float y = row[i];
#pragma unroll
    for (unsigned j = 0; j < kMax; ++j)
      if (j < count)
        sums[j] += y * direction[std::size_t{j} * hidden + i];
    if (preserve)
      before += y * y;
  }
#pragma unroll
  for (unsigned j = 0; j < kMax; ++j)
    if (j < count)
      partial[j][t] = sums[j];
  partial[kMax][t] = before;
  __syncthreads();
  for (unsigned step = kAbliterationRowThreads / 2; step; step >>= 1) {
    if (t < step) {
#pragma unroll
      for (unsigned j = 0; j < kMax; ++j)
        if (j < count)
          partial[j][t] += partial[j][t + step];
      if (preserve)
        partial[kMax][t] += partial[kMax][t + step];
    }
    __syncthreads();
  }
  float amounts[kMax];
#pragma unroll
  for (unsigned j = 0; j < kMax; ++j)
    amounts[j] = j < count ? weight * partial[j][0] : 0.0F;
  float after = 0;
  for (std::uint32_t i = t; i < hidden; i += blockDim.x) {
    float y = row[i];
#pragma unroll
    for (unsigned j = 0; j < kMax; ++j)
      if (j < count)
        y -= amounts[j] * direction[std::size_t{j} * hidden + i];
    row[i] = y;
    if (preserve)
      after += y * y;
  }
  if (!preserve)
    return;
  // Each thread rescales only the elements it wrote, so the row needs no
  // barrier; the norm reduction does.
  partial[kMax + 1][t] = after;
  __syncthreads();
  for (unsigned step = kAbliterationRowThreads / 2; step; step >>= 1) {
    if (t < step)
      partial[kMax + 1][t] += partial[kMax + 1][t + step];
    __syncthreads();
  }
  const float squared_before = partial[kMax][0];
  const float squared_after = partial[kMax + 1][0];
  // A row projected to exactly zero has no direction to rescale.
  if (!(squared_after > 0))
    return;
  const float scale = sqrtf(squared_before / squared_after);
  for (std::uint32_t i = t; i < hidden; i += blockDim.x)
    row[i] *= scale;
}

/// AbliterateRowIn with its own block-shared scratch.
__device__ inline void AbliterateRow(float* __restrict__ row,
                                     const float* __restrict__ directions,
                                     const float* __restrict__ settings,
                                     std::uint32_t weight_index,
                                     std::uint32_t hidden) {
  __shared__ AbliterationRowPartial partial;
  __shared__ AbliterationRowSingle single;
  AbliterateRowIn(partial, single, row, directions, settings, weight_index,
                  hidden);
}

}  // namespace gufo::models::qwen::hip_detail

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_ABLITERATION_ROW_HPP_
