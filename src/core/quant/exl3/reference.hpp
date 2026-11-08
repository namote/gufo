#ifndef GUFO_CORE_QUANT_EXL3_REFERENCE_HPP_
#define GUFO_CORE_QUANT_EXL3_REFERENCE_HPP_

// X2a reference GPU path for EXL3 projections (plan/exl3/archive/XA-RESULTS.md
// "X2a 설계 메모" section 4): simple kernels, correctness only. For a
// projection y = x W (docs/EXL3-FORMAT.md):
//
//   x' = H_k(suh * x)              FP32 per row
//   W' chunk -> FP32 [n_chunk][k]  bit-exact decode of the trellis (X0)
//   t  = x' W'                     FP32 GEMM
//   y  = svh * H_n(t)              FP32 per row, in place
//
// No weight is rounded below its FP16 trellis value, so this path also
// measures how much of the X1 BF16-GGUF difference came from re-rounding.

#include <cstddef>
#include <cstdint>

#include "src/core/quant/exl3/device.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::quant::exl3 {

/// Device workspace of LaunchReferenceProjection.
struct ReferenceScratch {
  float* rotated = nullptr;  ///< >= rows * k floats
  std::size_t rotated_elements = 0;
  float* weight = nullptr;  ///< >= 128 * k floats; larger chunks run faster
  std::size_t weight_elements = 0;
  /// X2b fast decode/verify path (src/core/quant/exl3/gemv.hpp):
  /// FP16 rotated-input slab and segment counters. Null keeps every call on
  /// the reference path.
  std::uint16_t* fast_input = nullptr;
  std::size_t fast_input_elements = 0;
  std::uint32_t* fast_counters = nullptr;
  std::size_t fast_counter_count = 0;
  /// The fused MLP's down-projection input slab (FP16); null keeps gate/up,
  /// SwiGLU and down as separate launches.
  std::uint16_t* fast_down_input = nullptr;
  std::size_t fast_down_input_elements = 0;
  /// X3 prefill: FP16 rotated input rows [max batch][widest input]; null keeps
  /// prefill on the reference path.
  std::uint16_t* fast_prefill_input = nullptr;
  std::size_t fast_prefill_input_elements = 0;
};

/// Decodes columns [n_begin, n_begin + n_count) of W' into out[n_count][k]
/// (FP32 holding the exact FP16 values). n_begin, n_count multiples of 16.
void LaunchReferenceDequantF32(const DeviceLinear& linear, std::size_t n_begin,
                               std::size_t n_count, float* out,
                               hipStream_t stream);

/// out[r] = post * H(pre * x[r]) per 128-element block of `width` columns
/// (`pre` / `post` FP16 vectors of length width, or null). `out` may equal
/// `x` when the strides match.
void LaunchReferenceRotateRows(const float* x, std::size_t x_stride,
                               std::size_t rows, std::size_t width,
                               const std::uint16_t* pre,
                               const std::uint16_t* post, float* out,
                               std::size_t out_stride, hipStream_t stream);

/// y[r][0..n) = x[r][0..k) W for `rows` rows (strides in floats).
void LaunchReferenceProjection(const DeviceLinear& linear, const float* x,
                               std::size_t x_stride, std::size_t rows, float* y,
                               std::size_t y_stride,
                               const ReferenceScratch& scratch,
                               hipStream_t stream);

}  // namespace gufo::quant::exl3

#endif  // defined(ENGINE_ENABLE_HIP)
#endif  // GUFO_CORE_QUANT_EXL3_REFERENCE_HPP_
