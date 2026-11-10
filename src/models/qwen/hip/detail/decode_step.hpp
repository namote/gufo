#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_DECODE_STEP_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_DECODE_STEP_HPP_

#include <cstdint>

#include "src/models/qwen/hip/ops/ssm.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen/vision/device_input.hpp"

namespace gufo::models {
struct QwenModelWeights;
}  // namespace gufo::models

namespace gufo::hip {

class QwenGpuArena;
struct QwenExecutionPolicy;

/// Emits pure decode route decisions before entering graph capture. This must
/// never be called from the capture callback.
void EmitDecodeRouteTelemetry(const models::QwenModelWeights& weights,
                              const QwenExecutionPolicy& policy);

/// Runs one graph-capture-safe decode layer stack using stable arena storage.
/// `recurrent_state` is the pass's QwenGpuArena::BeginRecurrentWrite result;
/// a captured graph bakes its pointers, so the graph key must carry
/// QwenGpuArena::RecurrentIoIdentity. `abliteration_settings` (the session's
/// device settings, null OFF) projects its row of `abliteration_directions`
/// out of each attention/GDN and MLP writer output before its residual add;
/// the graph key must distinguish ON from OFF. The kernels read the settings
/// at run time, so their values need no graph key.
void ExecuteDecodeStep(QwenGpuArena& arena,
                       const models::QwenModelWeights& weights,
                       const QwenExecutionPolicy& policy,
                       tokenization::TokenId token_id, std::uint32_t pos,
                       bool compute_logits,
                       QwenRecurrentStateIo recurrent_state,
                       models::qwen::vision::DeviceInput* image_input = nullptr,
                       std::uint32_t kvarn_groups = 0,
                       const float* abliteration_settings = nullptr,
                       const float* abliteration_directions = nullptr);

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_DECODE_STEP_HPP_
