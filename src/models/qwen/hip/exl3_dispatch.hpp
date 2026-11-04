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
#include <cstdint>
#include <initializer_list>
#include <span>
#include <stdexcept>

#include "src/models/qwen/state.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/quant/exl3/gemv.hpp"
#include "src/core/quant/exl3/reference.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"

namespace gufo::hip {

/// From this many rows a prefill chunk uses the trellis WMMA GEMM (X3b narrow
/// tile up to 64 rows, X3 prefill tile above) instead of 8-row GEMV passes.
/// Measured per projection on the Qwen3.8 27B 5.00bpw pack at 1-64 rows
/// (tools/bench/exl3_wmma_threshold_bench.hip, docs/GFX1151-KNOWHOW.md): one
/// WMMA launch costs about 1.7x a GEMV pass for the 5120 x 17408 FFN
/// matrices, so they switch at the second pass (9 rows); the smaller
/// projections (up to 5120 x 12288) cost 2.1-4.2x and switch at the third
/// (17 rows). Decode and verify (batched sessions included) always stay on the
/// GEMV, whose per-row results do not depend on the batch.
inline constexpr std::size_t kExl3WmmaMinRows = 17;
inline constexpr std::size_t kExl3WmmaMinRowsLarge = 9;
/// Weight elements (k x n) from which a projection counts as large.
inline constexpr std::uint64_t kExl3WmmaLargeElements = std::uint64_t{1} << 26;

/// The prefill row count from which `linear` takes the WMMA GEMM.
[[nodiscard]] inline std::size_t Exl3WmmaMinRows(
    const quant::exl3::DeviceLinear& linear) noexcept {
  return std::uint64_t{linear.k} * linear.n >= kExl3WmmaLargeElements
             ? kExl3WmmaMinRowsLarge
             : kExl3WmmaMinRows;
}

/// Which forward pass calls a projection.
enum class Exl3Pass { kDecodeVerify, kPrefill };

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
                                 std::size_t y_stride = 0,
                                 Exl3Pass pass = Exl3Pass::kDecodeVerify) {
  if (weight.exl3 == nullptr || scratch == nullptr) {
    throw std::logic_error(
        "EXL3 projection without a descriptor or session workspace");
  }
  const auto& linear = *weight.exl3;
  // Prefill chunks from Exl3WmmaMinRows rows take the X3 trellis WMMA GEMM
  // (the X3b narrow tile up to 64 rows): rotate the rows into the FP16 slab,
  // then GEMM with the svh epilogue. Decode and verify rows (up to 64) take
  // the X2b trellis GEMV. Anything else stays on the reference path.
  const bool wmma =
      pass == Exl3Pass::kPrefill && scratch->fast_prefill_input != nullptr &&
      rows >= Exl3WmmaMinRows(linear) && Exl3PrefillSupported(linear) &&
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
  if (scratch->fast_input != nullptr && rows <= kExl3FastMaxRows) {
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

/// Which shared input of a layer a group of projections reads.
enum class Exl3LayerInput { kAttention, kFfn };

/// LaunchExl3NormForConsumers for the projections of `layer` that read
/// `input`, in the order of LaunchExl3AttentionInputs (q, k, v or qkv, z) and
/// LaunchExl3SwiGLUMlp (gate, up). False, launching nothing, for non-EXL3
/// layers or when the fast path does not apply.
inline bool LaunchExl3LayerNorm(const models::QwenLayerWeights& layer,
                                Exl3LayerInput input, const float* x,
                                float* normed, std::size_t dim, float eps,
                                const quant::exl3::ReferenceScratch* scratch,
                                hipStream_t stream) {
  if (input == Exl3LayerInput::kFfn) {
    return LaunchExl3NormForConsumers(
        x, static_cast<const float*>(layer.ffn_norm.data), normed, dim, eps,
        {&layer.ffn_gate, &layer.ffn_up}, scratch, stream);
  }
  const auto* weight = static_cast<const float*>(layer.attn_norm.data);
  return layer.is_full_attention
             ? LaunchExl3NormForConsumers(
                   x, weight, normed, dim, eps,
                   {&layer.attn_q, &layer.attn_k, &layer.attn_v}, scratch,
                   stream)
             : LaunchExl3NormForConsumers(x, weight, normed, dim, eps,
                                          {&layer.attn_qkv, &layer.attn_gate},
                                          scratch, stream);
}

/// Destinations of a layer's attention input projections (dense rows).
/// Full attention writes q (gate interleaved) to `qkv`, k and v; linear
/// attention writes qkv and z (`gate`). Unused pointers may be null.
struct Exl3AttentionInputs {
  float* qkv;
  float* k;
  float* v;
  float* gate;
};

/// The EXL3 attention input projections of `layer` as one group: q, k, v for
/// full attention, qkv and z for linear attention (whose alpha and beta stay
/// unquantized). `input_rotated` as in LaunchExl3Projections.
inline void LaunchExl3AttentionInputs(
    const models::QwenLayerWeights& layer, const float* x, std::size_t rows,
    const Exl3AttentionInputs& out,
    const quant::exl3::ReferenceScratch* scratch, hipStream_t stream,
    bool input_rotated = false) {
  if (layer.is_full_attention) {
    const Exl3GroupOutput qkv[] = {{&layer.attn_q, out.qkv},
                                   {&layer.attn_k, out.k},
                                   {&layer.attn_v, out.v}};
    LaunchExl3Projections(qkv, x, rows, scratch, stream, 0, input_rotated);
    return;
  }
  const Exl3GroupOutput qkvz[] = {{&layer.attn_qkv, out.qkv},
                                  {&layer.attn_gate, out.gate}};
  LaunchExl3Projections(qkvz, x, rows, scratch, stream, 0, input_rotated);
}

/// Whether the attention input projections of `layer` are EXL3.
[[nodiscard]] inline bool IsExl3AttentionInput(
    const models::QwenLayerWeights& layer) noexcept {
  return IsExl3(layer.is_full_attention ? layer.attn_q : layer.attn_qkv);
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
