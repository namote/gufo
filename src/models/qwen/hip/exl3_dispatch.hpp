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
#include <initializer_list>
#include <span>
#include <stdexcept>

#include "src/models/qwen/state.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/quant/exl3/reference.hpp"
#include "src/models/qwen/hip/kernels/exl3_gemv.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"

namespace gufo::hip {

/// From this many rows a session with the prefill slab uses the trellis WMMA
/// GEMM (X3b narrow tile, X3 prefill tile) instead of 8-row GEMV passes;
/// provisional until the final window compares the two at 9-64 rows.
inline constexpr std::size_t kExl3WmmaMinRows = 17;

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
  // Batches from kExl3WmmaMinRows rows take the X3 trellis WMMA GEMM (the
  // X3b narrow tile up to 64 rows): rotate the rows into the FP16 slab, then
  // GEMM with the svh epilogue. Decode and verify rows take the X2b trellis
  // GEMV. Anything else stays on the reference path.
  const bool wmma = scratch->fast_prefill_input != nullptr &&
                    rows >= kExl3WmmaMinRows && Exl3PrefillSupported(linear) &&
                    scratch->fast_prefill_input_elements >= rows * linear.k &&
                    (y_stride == 0 || y_stride == linear.n);
  if (wmma) {
    LaunchExl3RotateRows(x, x_stride != 0 ? x_stride : linear.k, rows, linear.k,
                         linear.suh, scratch->fast_prefill_input, stream);
    LaunchExl3PrefillGemm(linear, scratch->fast_prefill_input, y, rows, false,
                          stream);
    return;
  }
  if (scratch->fast_input != nullptr && rows <= kExl3FastMaxRows &&
      Exl3FastSupported(linear)) {
    const Exl3FastWorkspace workspace{
        scratch->fast_input,      scratch->fast_input_elements,
        scratch->fast_counters,   scratch->fast_counter_count,
        scratch->fast_down_input, scratch->fast_down_input_elements};
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
/// `input_rotated`: LaunchExl3NormForConsumers wrote this group's rotated
/// input (rows == 1); x still holds the unrotated rows for any fallback.
inline void LaunchExl3Projections(std::span<const Exl3GroupOutput> outputs,
                                  const float* x, std::size_t rows,
                                  const quant::exl3::ReferenceScratch* scratch,
                                  hipStream_t stream, std::size_t x_stride = 0,
                                  bool input_rotated = false) {
  if (scratch == nullptr || outputs.empty()) {
    throw std::logic_error(
        "EXL3 projection group without outputs or workspace");
  }
  if (scratch->fast_input != nullptr && rows <= kExl3FastMaxRows &&
      (rows < kExl3WmmaMinRows || scratch->fast_prefill_input == nullptr) &&
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
        scratch->fast_input,      scratch->fast_input_elements,
        scratch->fast_counters,   scratch->fast_counter_count,
        scratch->fast_down_input, scratch->fast_down_input_elements};
    const std::span<const Exl3FastTarget> group(targets.data(), outputs.size());
    if (all_exl3 && Exl3FastGroupSupported(group, workspace)) {
      LaunchExl3FastProjections(
          group, x, x_stride != 0 ? x_stride : targets[0].linear->k, rows,
          workspace, stream, input_rotated && rows == 1);
      return;
    }
  }
  for (const Exl3GroupOutput& output : outputs) {
    LaunchExl3Projection(*output.weight, x, rows, output.y, scratch, stream,
                         x_stride, output.y_stride);
  }
}

/// The FFN of `rows` rows: out = down(silu(x gate) * (x up)). Decode and
/// verify rows run the fused MLP (two GEMV launches per 8 rows, no FP32
/// activation); otherwise gate/up, SwiGLU into `activation`, then down.
/// gate_y, up_y and activation are rows x intermediate FP32 scratch; out has
/// rows of hidden floats.
inline void LaunchExl3SwiGLUMlp(const models::QwenTensorRef& gate,
                                const models::QwenTensorRef& up,
                                const models::QwenTensorRef& down,
                                const float* x, std::size_t rows, float* gate_y,
                                float* up_y, float* activation, float* out,
                                const quant::exl3::ReferenceScratch* scratch,
                                hipStream_t stream,
                                bool input_rotated = false) {
  if (scratch == nullptr || gate.exl3 == nullptr || up.exl3 == nullptr ||
      down.exl3 == nullptr) {
    throw std::logic_error("EXL3 MLP without descriptors or workspace");
  }
  if (scratch->fast_input != nullptr && rows <= kExl3FastMaxRows &&
      (rows < kExl3WmmaMinRows || scratch->fast_prefill_input == nullptr)) {
    const Exl3FastWorkspace workspace{
        scratch->fast_input,      scratch->fast_input_elements,
        scratch->fast_counters,   scratch->fast_counter_count,
        scratch->fast_down_input, scratch->fast_down_input_elements};
    const Exl3FastMlp mlp{gate.exl3, up.exl3, down.exl3,   gate_y,
                          up_y,      out,     down.exl3->n};
    if (Exl3FastMlpSupported(mlp, workspace)) {
      LaunchExl3FastMlp(mlp, x, gate.exl3->k, rows, workspace, stream,
                        input_rotated && rows == 1);
      return;
    }
  }
  const Exl3GroupOutput gate_up[] = {{&gate, gate_y}, {&up, up_y}};
  LaunchExl3Projections(gate_up, x, rows, scratch, stream);
  LaunchBatchedSwiGLUActivation(gate_y, up_y, activation, nullptr,
                                rows * gate.exl3->n, stream);
  LaunchExl3Projection(down, activation, rows, out, scratch, stream);
}

/// Decode RMSNorm (one row) of `x` into `normed`, fused with the rotation of
/// the normed row into the fast-path slabs of `consumers` (in the order of
/// the following group or MLP call, which then passes input_rotated). Returns
/// false, launching nothing, when the fast path does not apply; the caller
/// then runs its usual norm.
inline bool LaunchExl3NormForConsumers(
    const float* x, const float* weight, float* normed, std::size_t dim,
    float eps, std::initializer_list<const models::QwenTensorRef*> consumers,
    const quant::exl3::ReferenceScratch* scratch, hipStream_t stream) {
  if (scratch == nullptr || scratch->fast_input == nullptr ||
      consumers.size() > kExl3FastMaxGroup) {
    return false;
  }
  std::array<const quant::exl3::DeviceLinear*, kExl3FastMaxGroup> linears{};
  std::size_t count = 0;
  for (const auto* weight_ref : consumers) {
    if (weight_ref->exl3 == nullptr) {
      return false;
    }
    linears[count++] = weight_ref->exl3;
  }
  const Exl3FastWorkspace workspace{
      scratch->fast_input,      scratch->fast_input_elements,
      scratch->fast_counters,   scratch->fast_counter_count,
      scratch->fast_down_input, scratch->fast_down_input_elements};
  return LaunchExl3FastRmsNorm(x, weight, normed, dim, eps,
                               std::span(linears.data(), count), workspace,
                               stream);
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
