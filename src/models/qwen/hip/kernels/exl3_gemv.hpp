#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_EXL3_GEMV_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_EXL3_GEMV_HPP_

// X2b fast path of an EXL3 projection for decode and verify rows
// (plan/exl3/XB-DESIGN.md): the input rows are scaled by suh, rotated per 128
// elements and rounded to FP16 into a lane-major slab, then a trellis decode
// GEMV multiplies them (FP32 accumulation) and its epilogue rotates each
// completed 128-column segment of y in place and scales it by svh.

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/core/quant/exl3/device.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Session workspace of the fast path (QwenGpuArena::AllocateExl3).
struct Exl3FastWorkspace {
  /// FP16 lane-major rotated input: 8 rows of the input per matrix of a
  /// group, >= 3 * 8 * hidden and >= 8 * widest input halves.
  std::uint16_t* input = nullptr;
  std::size_t input_elements = 0;
  /// Segment arrival counters, >= the outputs of one launch / 128; zero
  /// when allocated, and each launch leaves them zero.
  std::uint32_t* counters = nullptr;
  std::size_t counter_count = 0;
};

/// Rows above this run the reference path (prefill).
inline constexpr std::size_t kExl3FastMaxRows = 64;

/// Matrices one launch can carry (q + k + v).
inline constexpr std::size_t kExl3FastMaxGroup = 3;

/// One projection of a group: y[r][0..n) = x[r] W (y stride in floats).
struct Exl3FastTarget {
  const quant::exl3::DeviceLinear* linear;
  float* y;
  std::size_t y_stride;
};

/// Whether the fast path takes `linear` (mul1 codebook, integer K 1..6).
[[nodiscard]] bool Exl3FastSupported(
    const quant::exl3::DeviceLinear& linear) noexcept;

/// Whether one launch can carry `targets`: 1..kExl3FastMaxGroup supported
/// matrices of the same k and bit rate (each still reads its own suh), with
/// a rotated-input slab per matrix and counters for all of them in
/// `workspace`.
[[nodiscard]] bool Exl3FastGroupSupported(
    std::span<const Exl3FastTarget> targets,
    const Exl3FastWorkspace& workspace) noexcept;

/// Every target's projection of the shared input x (rows of x_stride floats),
/// one rotation launch and one GEMV launch per pass of up to 8 rows. The
/// group must pass Exl3FastGroupSupported; 1 <= rows <= kExl3FastMaxRows.
void LaunchExl3FastProjections(std::span<const Exl3FastTarget> targets,
                               const float* x, std::size_t x_stride,
                               std::size_t rows,
                               const Exl3FastWorkspace& workspace,
                               hipStream_t stream);

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
