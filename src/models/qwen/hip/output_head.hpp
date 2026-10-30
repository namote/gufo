#ifndef GUFO_MODELS_QWEN_HIP_OUTPUT_HEAD_HPP_
#define GUFO_MODELS_QWEN_HIP_OUTPUT_HEAD_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/quant/exl3/reference.hpp"
#include "src/models/qwen/state.hpp"

namespace gufo::hip {

/// Applies the Qwen output head to one hidden row on `stream`:
/// normed = RMSNorm(hidden) · output_norm, rotated when the head input is
/// rotated (QwenModelWeights::head_in_rotation), then
/// logits = output · normed. Device pointers; `normed` holds hidden_size
/// floats and `logits` vocab_size floats. This is the decode and prefill
/// head; callers that apply the head to captured hidden rows use it so a
/// rotated GGUF needs no extra handling. An EXL3 head needs the session's
/// workspace (`exl3`, QwenGpuArena::Exl3Scratch).
void LaunchQwenOutputHead(const models::QwenModelWeights& weights,
                          const float* hidden, float* normed, float* logits,
                          hipStream_t stream,
                          const quant::exl3::ReferenceScratch* exl3 = nullptr);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OUTPUT_HEAD_HPP_
