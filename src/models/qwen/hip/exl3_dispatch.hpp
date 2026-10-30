#ifndef GUFO_MODELS_QWEN_HIP_EXL3_DISPATCH_HPP_
#define GUFO_MODELS_QWEN_HIP_EXL3_DISPATCH_HPP_

// The single dispatch point of an EXL3 projection (QwenTensorRef::exl3) in
// the Qwen HIP forward paths (plan/exl3/XA-RESULTS.md "X2a 설계 메모",
// plan/exl3/XB-DESIGN.md). Every call site that multiplies by a projection
// branches here when the tensor is EXL3. X2a runs the reference path
// (src/core/quant/exl3/reference.hpp); X2b adds its fast kernels behind the
// same function.

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>

#include "src/models/qwen/state.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/quant/exl3/reference.hpp"
#include "src/models/qwen/hip/kernels/exl3_gemv.hpp"

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
  // Decode and verify rows take the X2b trellis GEMV; prefill batches and
  // formats it does not cover stay on the reference path.
  if (scratch->fast_input != nullptr && rows <= kExl3FastMaxRows &&
      Exl3FastSupported(linear)) {
    const Exl3FastWorkspace workspace{
        scratch->fast_input, scratch->fast_input_elements,
        scratch->fast_counters, scratch->fast_counter_count};
    LaunchExl3FastProjection(linear, x, x_stride != 0 ? x_stride : linear.k,
                             rows, y, y_stride != 0 ? y_stride : linear.n,
                             workspace, stream);
    return;
  }
  quant::exl3::LaunchReferenceProjection(
      linear, x, x_stride != 0 ? x_stride : linear.k, rows, y,
      y_stride != 0 ? y_stride : linear.n, *scratch, stream);
}

/// One projection of a shared-input group (q/k/v, qkv/z, gate/up): its
/// weight, output rows and output stride in floats (0: dense, n).
struct Exl3GroupOutput {
  const models::QwenTensorRef* weight;
  float* y;
  std::size_t y_stride = 0;
};

/// Every output's projection of the same input rows. Decode and verify rows
/// of a group the fast path takes run as one launch per pass of 8 rows; any
/// other group runs projection by projection through LaunchExl3Projection.
inline void LaunchExl3Projections(std::span<const Exl3GroupOutput> outputs,
                                  const float* x, std::size_t rows,
                                  const quant::exl3::ReferenceScratch* scratch,
                                  hipStream_t stream,
                                  std::size_t x_stride = 0) {
  if (scratch == nullptr || outputs.empty()) {
    throw std::logic_error(
        "EXL3 projection group without outputs or workspace");
  }
  if (scratch->fast_input != nullptr && rows <= kExl3FastMaxRows &&
      outputs.size() <= kExl3FastMaxGroup) {
    std::array<Exl3FastTarget, kExl3FastMaxGroup> targets{};
    bool all_exl3 = true;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      const auto* linear = outputs[i].weight->exl3;
      all_exl3 = all_exl3 && linear != nullptr;
      if (!all_exl3)
        break;
      targets[i] = {linear, outputs[i].y,
                    outputs[i].y_stride != 0 ? outputs[i].y_stride : linear->n};
    }
    const Exl3FastWorkspace workspace{
        scratch->fast_input, scratch->fast_input_elements,
        scratch->fast_counters, scratch->fast_counter_count};
    const std::span<const Exl3FastTarget> group(targets.data(), outputs.size());
    if (all_exl3 && Exl3FastGroupSupported(group, workspace)) {
      LaunchExl3FastProjections(group, x,
                                x_stride != 0 ? x_stride : targets[0].linear->k,
                                rows, workspace, stream);
      return;
    }
  }
  for (const Exl3GroupOutput& output : outputs) {
    LaunchExl3Projection(*output.weight, x, rows, output.y, scratch, stream,
                         x_stride, output.y_stride);
  }
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
