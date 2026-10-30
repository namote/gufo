#ifndef GUFO_MODELS_QWEN_HIP_EXL3_DISPATCH_HPP_
#define GUFO_MODELS_QWEN_HIP_EXL3_DISPATCH_HPP_

// The single dispatch point of an EXL3 projection (QwenTensorRef::exl3) in
// the Qwen HIP forward paths (plan/exl3/XA-RESULTS.md "X2a 설계 메모",
// plan/exl3/XB-DESIGN.md). Every call site that multiplies by a projection
// branches here when the tensor is EXL3. X2a runs the reference path
// (src/core/quant/exl3/reference.hpp); X2b adds its fast kernels behind the
// same function.

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "src/models/qwen/state.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/quant/exl3/reference.hpp"

namespace gufo::hip {

/// Whether `weight` is an EXL3 projection.
[[nodiscard]] inline bool IsExl3(const models::QwenTensorRef& weight) noexcept {
  return weight.exl3 != nullptr;
}

/// y[r][0..n) = x[r][0..k) W for `rows` rows of an EXL3 projection; strides
/// in floats (0 means dense: k and n). `scratch` is the session's workspace
/// (QwenGpuArena::Exl3Scratch).
inline void LaunchExl3Projection(const models::QwenTensorRef& weight,
                                 const float* x, std::size_t rows, float* y,
                                 const quant::exl3::ReferenceScratch* scratch,
                                 hipStream_t stream, std::size_t x_stride = 0,
                                 std::size_t y_stride = 0) {
  if (weight.exl3 == nullptr || scratch == nullptr) {
    throw std::logic_error(
        "EXL3 projection without a descriptor or session workspace");
  }
  const auto& linear = *weight.exl3;
  quant::exl3::LaunchReferenceProjection(
      linear, x, x_stride != 0 ? x_stride : linear.k, rows, y,
      y_stride != 0 ? y_stride : linear.n, *scratch, stream);
}

/// The largest input width of the model's projections (workspace sizing).
[[nodiscard]] inline std::size_t MaxExl3InputWidth(
    const core::ModelConfig& config) noexcept {
  std::size_t width = config.hidden_size;
  width = std::max<std::size_t>(width, config.intermediate_size);
  width = std::max<std::size_t>(width, config.ssm_inner_size);
  width = std::max<std::size_t>(width, config.AttentionSize());
  return width;
}

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)
#endif  // GUFO_MODELS_QWEN_HIP_EXL3_DISPATCH_HPP_
