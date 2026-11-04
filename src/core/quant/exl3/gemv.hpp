#ifndef GUFO_CORE_QUANT_EXL3_GEMV_HPP_
#define GUFO_CORE_QUANT_EXL3_GEMV_HPP_

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

/// Session workspace of the fast path, owned by the model arena (Qwen:
/// QwenGpuArena::AllocateExl3).
struct Exl3FastWorkspace {
  /// FP16 lane-major rotated input: 8 rows of the input per matrix of a
  /// group, >= 3 * 8 * hidden and >= 8 * widest input halves.
  std::uint16_t* input = nullptr;
  std::size_t input_elements = 0;
  /// Segment arrival counters, >= the outputs of one launch / 128; zero
  /// when allocated, and each launch leaves them zero.
  std::uint32_t* counters = nullptr;
  std::size_t counter_count = 0;
  /// The down projection's FP16 lane-major input written by the fused
  /// gate/up epilogue, >= 8 * intermediate halves; null disables the fused
  /// MLP.
  std::uint16_t* down_input = nullptr;
  std::size_t down_input_elements = 0;
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

/// One problem of a grouped prefill rotation (an expert bucket). Device
/// memory.
struct Exl3RotateProblem {
  const float* x;                  ///< FP32 source rows
  std::size_t x_stride;            ///< floats between source rows
  const std::uint32_t* row_index;  ///< gathered source row per output row, or
                                   ///< null for rows 0 .. rows-1
  const std::uint16_t* suh;        ///< FP16 [k]
  std::uint16_t* out;              ///< FP16 [rows][k]
  std::uint32_t rows;              ///< 0 skips the problem
  std::uint32_t reserved;
};

/// Every problem's rows rotated as LaunchExl3RotateRows, in one launch
/// (count <= 65535; max_rows bounds every problem's rows).
void LaunchExl3RotateRowsGrouped(const Exl3RotateProblem* problems,
                                 std::size_t count, std::size_t max_rows,
                                 std::size_t k, hipStream_t stream);

/// Prefill input rotation: out[r][i] = FP16(H(suh * x[r])) per 128
/// elements, row-major [rows][k] (the decode path's input-rotation
/// arithmetic).
void LaunchExl3RotateRows(const float* x, std::size_t x_stride,
                          std::size_t rows, std::size_t k,
                          const std::uint16_t* suh, std::uint16_t* out,
                          hipStream_t stream);

/// Whether the fast path takes `linear` (mul1 codebook, integer K 1..7).
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
/// With input_rotated (rows == 1 only) the slabs already hold the rotated
/// row (LaunchExl3FastRmsNorm) and x is not read.
///
/// shape_basis_n (a multiple of 32; 0 = these targets) is the total output
/// width of the group whose launch geometry (waves splitting K, tiles per
/// wave) to use. The geometry decides each output's reduction order, so a
/// subset of a group (e.g. k and v without q) launched with the full group's
/// total n gives every target the bits of the full-group launch.
void LaunchExl3FastProjections(std::span<const Exl3FastTarget> targets,
                               const float* x, std::size_t x_stride,
                               std::size_t rows,
                               const Exl3FastWorkspace& workspace,
                               hipStream_t stream, bool input_rotated = false,
                               std::size_t shape_basis_n = 0);

/// A SwiGLU MLP: out = down(silu(x gate) * (x up)). gate_y and up_y are
/// [8][gate.n] FP32 scratch rows (the pre-activation sums).
struct Exl3FastMlp {
  const quant::exl3::DeviceLinear* gate;
  const quant::exl3::DeviceLinear* up;
  const quant::exl3::DeviceLinear* down;
  float* gate_y;
  float* up_y;
  float* out;
  std::size_t out_stride;
};

/// Whether LaunchExl3FastMlp takes `mlp` with `workspace`.
[[nodiscard]] bool Exl3FastMlpSupported(
    const Exl3FastMlp& mlp, const Exl3FastWorkspace& workspace) noexcept;

/// The whole MLP of 1..kExl3FastMaxRows rows, two GEMV launches per pass of 8
/// rows: gate and up together, whose epilogue applies SwiGLU and writes the
/// down projection's rotated input, then down. The activation never reaches
/// memory in FP32; the values are those of the unfused launches.
void LaunchExl3FastMlp(const Exl3FastMlp& mlp, const float* x,
                       std::size_t x_stride, std::size_t rows,
                       const Exl3FastWorkspace& workspace, hipStream_t stream,
                       bool input_rotated = false);

/// RMSNorm of one row of width 5120 (the arithmetic of LaunchRMSNorm) into
/// `normed`, fused with the rotation of the normed row into the FP16 slab of
/// every consumer (slab region c for consumer c: the order of a following
/// LaunchExl3FastProjections / LaunchExl3FastMlp with input_rotated).
/// Returns false and launches nothing when it does not apply (other widths,
/// consumers the fast path does not take).
[[nodiscard]] bool LaunchExl3FastRmsNorm(
    const float* x, const float* weight, float* normed, std::size_t dim,
    float eps, std::span<const quant::exl3::DeviceLinear* const> consumers,
    const Exl3FastWorkspace& workspace, hipStream_t stream);

/// y[r][0..n) = x[r][0..k) W for 1 <= rows <= kExl3FastMaxRows, in passes of
/// up to 8 rows. Strides in floats.
void LaunchExl3FastProjection(const quant::exl3::DeviceLinear& linear,
                              const float* x, std::size_t x_stride,
                              std::size_t rows, float* y, std::size_t y_stride,
                              const Exl3FastWorkspace& workspace,
                              hipStream_t stream);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)
#endif  // GUFO_CORE_QUANT_EXL3_GEMV_HPP_
