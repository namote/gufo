#ifndef GUFO_TESTS_MODELS_QWEN_SUPPORT_ABLITERATION_SUBSPACE_HPP_
#define GUFO_TESTS_MODELS_QWEN_SUPPORT_ABLITERATION_SUBSPACE_HPP_

// Shared GPU-test support for the Qwen runtime abliteration projection
// (Qwen3.8-27B and Qwen3.8-Flash-Next): orthonormal direction blocks, the
// device word 0 layout written independently, an FP64 reference of the
// subspace projection with norm preservation, and a verbatim copy of the
// single-direction (D16) kernel the new kernel must match bit for bit.
// Include from HIP sources only.

#include <hip/hip_runtime.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace gufo::tests::qwen_abliteration {

/// `count` orthonormal FP32 rows of `hidden` floats (Gram-Schmidt in FP64,
/// twice, then rounded).
inline std::vector<float> OrthonormalRows(std::uint32_t count,
                                          std::uint32_t hidden,
                                          std::mt19937& rng) {
  std::normal_distribution<double> normal;
  std::vector<std::vector<double>> basis;
  while (basis.size() < count) {
    std::vector<double> v(hidden);
    for (auto& x : v)
      x = normal(rng);
    for (int pass = 0; pass < 2; ++pass)
      for (const auto& b : basis) {
        double dot = 0;
        for (std::uint32_t i = 0; i < hidden; ++i)
          dot += b[i] * v[i];
        for (std::uint32_t i = 0; i < hidden; ++i)
          v[i] -= dot * b[i];
      }
    double norm = 0;
    for (double x : v)
      norm += x * x;
    norm = std::sqrt(norm);
    for (auto& x : v)
      x /= norm;
    basis.push_back(std::move(v));
  }
  std::vector<float> result;
  for (const auto& b : basis)
    for (double x : b)
      result.push_back(static_cast<float>(x));
  return result;
}

/// Device word 0: first row in bits 0-23, count - 1 in bits 24-26, norm
/// preservation in bit 27 (models::qwen::AbliterationSettings::DeviceWords).
inline float Word0(std::uint32_t row, std::uint32_t count, bool preserve) {
  const std::uint32_t bits =
      row | (count - 1) << 24 | (preserve ? 1U << 27 : 0U);
  return std::bit_cast<float>(bits);
}

/// One row's projection in FP64: y - w * sum_j d_j * dot(d_j, y), every dot
/// from the unprojected y, then (preserve) y * |y| / |y'|. `directions` holds
/// `count` rows of `hidden` floats.
inline std::vector<double> ReferenceRow(const float* y, const float* directions,
                                        std::uint32_t count, double weight,
                                        bool preserve, std::uint32_t hidden) {
  std::vector<double> result(y, y + hidden);
  if (weight == 0)
    return result;
  for (std::uint32_t j = 0; j < count; ++j) {
    const float* d = directions + std::size_t{j} * hidden;
    double dot = 0;
    for (std::uint32_t i = 0; i < hidden; ++i)
      dot += double{d[i]} * y[i];
    for (std::uint32_t i = 0; i < hidden; ++i)
      result[i] -= weight * dot * d[i];
  }
  if (preserve) {
    double before = 0, after = 0;
    for (std::uint32_t i = 0; i < hidden; ++i) {
      before += double{y[i]} * y[i];
      after += result[i] * result[i];
    }
    if (after > 0)
      for (auto& x : result)
        x *= std::sqrt(before / after);
  }
  return result;
}

/// The single-direction kernel body before subspaces existed, verbatim: one
/// 256-thread block per row, `settings` shared by every row.
__device__ inline void D16AbliterateRow(float* __restrict__ rows,
                                        const float* __restrict__ directions,
                                        const float* __restrict__ settings,
                                        std::uint32_t weight_index,
                                        std::uint32_t hidden) {
  const float weight = settings[weight_index];
  if (weight == 0)
    return;
  const float* direction =
      directions + std::size_t{__float_as_uint(settings[0])} * hidden;
  __shared__ float partial[256];
  float* row = rows + std::size_t{blockIdx.x} * hidden;
  float sum = 0;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x)
    sum += row[i] * direction[i];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned step = 128; step; step >>= 1) {
    if (threadIdx.x < step)
      partial[threadIdx.x] += partial[threadIdx.x + step];
    __syncthreads();
  }
  const float amount = weight * partial[0];
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x)
    row[i] -= amount * direction[i];
}

/// Qwen3.8-27B declared 256-thread launch bounds.
__launch_bounds__(256) __global__
    void D16AbliterateRows(float* __restrict__ rows,
                           const float* __restrict__ directions,
                           const float* __restrict__ settings,
                           std::uint32_t weight_index, std::uint32_t hidden) {
  D16AbliterateRow(rows, directions, settings, weight_index, hidden);
}

/// Qwen3.8-Flash-Next declared none (and compiles with -ffast-math; build
/// the including test like its library).
__global__ void D16AbliterateRowsUnbounded(float* __restrict__ rows,
                                           const float* __restrict__ directions,
                                           const float* __restrict__ settings,
                                           std::uint32_t weight_index,
                                           std::uint32_t hidden) {
  D16AbliterateRow(rows, directions, settings, weight_index, hidden);
}

}  // namespace gufo::tests::qwen_abliteration

#endif  // GUFO_TESTS_MODELS_QWEN_SUPPORT_ABLITERATION_SUBSPACE_HPP_
