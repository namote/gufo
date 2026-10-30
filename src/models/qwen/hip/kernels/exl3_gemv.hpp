#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_EXL3_GEMV_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_EXL3_GEMV_HPP_

// X2b fast path of an EXL3 projection for decode and verify rows
// (plan/exl3/XB-DESIGN.md): the input rows are scaled by suh, rotated per 128
// elements and rounded to FP16 into a lane-major slab, then a trellis decode
// GEMV multiplies them (FP32 accumulation) and its epilogue rotates each
// completed 128-column segment of y in place and scales it by svh.

#include <cstddef>
#include <cstdint>

#include "src/core/quant/exl3/device.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Session workspace of the fast path (QwenGpuArena::AllocateExl3).
struct Exl3FastWorkspace {
  /// FP16 lane-major rotated input, >= 8 * widest input halves.
  std::uint16_t* input = nullptr;
  std::size_t input_elements = 0;
  /// Segment arrival counters, >= widest output / 128; zero when allocated,
  /// and each launch leaves them zero.
  std::uint32_t* counters = nullptr;
  std::size_t counter_count = 0;
};

/// Rows above this run the reference path (prefill).
inline constexpr std::size_t kExl3FastMaxRows = 64;

/// Whether the fast path takes `linear` (mul1 codebook, integer K 1..6).
[[nodiscard]] bool Exl3FastSupported(
    const quant::exl3::DeviceLinear& linear) noexcept;

/// y[r][0..n) = x[r][0..k) W for 1 <= rows <= kExl3FastMaxRows, in passes of
/// up to 8 rows. Strides in floats.
void LaunchExl3FastProjection(const quant::exl3::DeviceLinear& linear,
                              const float* x, std::size_t x_stride,
                              std::size_t rows, float* y, std::size_t y_stride,
                              const Exl3FastWorkspace& workspace,
                              hipStream_t stream);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)
#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_EXL3_GEMV_HPP_
