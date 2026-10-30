#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/detail/decode_step.hpp"

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/exl3_dispatch.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/hip/ops/hadamard.hpp"
#include "src/models/qwen/hip/output_head.hpp"
#include "src/models/qwen/modules/modules.hpp"

namespace gufo::hip {

void LaunchQwenOutputHead(const models::QwenModelWeights& weights,
                          const float* hidden, float* normed, float* logits,
                          hipStream_t stream,
                          const quant::exl3::ReferenceScratch* exl3) {
  const std::size_t hidden_size = weights.config.hidden_size;
  const auto* norm = static_cast<const float*>(weights.output_norm.data);
  if (weights.head_in_rotation != nullptr) {
    LaunchRMSNormRotated(hidden, norm, normed, hidden_size, 1e-6F,
                         weights.head_in_rotation, stream);
  } else {
    LaunchRMSNorm(hidden, norm, normed, hidden_size, 1e-6F, stream);
  }
  if (IsExl3(weights.output)) {
    LaunchExl3Projection(weights.output, normed, 1, logits, exl3, stream);
    return;
  }
  LaunchGEMV(weights.output.data, weights.output.type, normed, logits,
             weights.config.vocab_size, hidden_size, stream);
}

void EmitDecodeRouteTelemetry(const models::QwenModelWeights& weights,
                              const QwenExecutionPolicy& policy) {
  if (!detail::DispatchTelemetryEnabled()) {
    return;
  }
  for (std::uint32_t layer_index = 0; layer_index < weights.config.num_layers;
       ++layer_index) {
    const auto& layer = weights.layers[layer_index];
    const auto resolution = ResolveQwenLayerRouteWithReasons(
        policy, QwenExecutionMode::kDecode, layer.is_full_attention);
    detail::EmitQwenRouteResolution(
        "decode", layer_index, layer.is_full_attention ? "attention" : "ssm",
        resolution.plan.Fingerprint(),
        static_cast<std::uint32_t>(resolution.rejected));
  }
}

void ExecuteDecodeStep(QwenGpuArena& arena,
                       const models::QwenModelWeights& weights,
                       const QwenExecutionPolicy& policy,
                       tokenization::TokenId token_id, std::uint32_t pos,
                       bool compute_logits,
                       QwenRecurrentStateIo recurrent_state,
                       models::qwen::vision::DeviceInput* image_input,
                       std::uint32_t kvarn_groups) {
  const auto* rope = image_input != nullptr ? image_input->rope() : nullptr;
  (void)token_id;
  const auto& config = weights.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config.SsmQkvSize();
  const std::size_t ssm_inner_size = config.ssm_inner_size;
  const std::size_t time_step_rank = config.ssm_time_step_rank;
  auto scratch = arena.GetScratchView();
  const auto* exl3 = arena.Exl3Scratch();
  auto& decode_scratch = scratch.decode;
  auto& attention_scratch = scratch.attention;
  auto& ssm_scratch = scratch.ssm;
  auto& ffn_scratch = scratch.ffn;
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  const bool use_split_k_decode = detail::IsSplitKDecodeAttentionSupported(
      sequence_length, config.num_attention_heads, config.num_key_value_heads,
      config.head_dim);

  // GPU parameter buffers
  const std::uint32_t* d_in_token = decode_scratch.prompt_tokens.data();
  const std::uint32_t* d_in_pos = decode_scratch.prompt_tokens.data() + 1;
  auto* d_out_token = decode_scratch.sampled_token.data();

  // 1. Embedding lookup
  LaunchEmbeddingLookup(weights.token_embd.data, weights.token_embd.type,
                        d_in_token, decode_scratch.hidden.data(), hidden_size,
                        arena.stream);

  if (image_input != nullptr)
    image_input->Inject(decode_scratch.hidden.data(), pos, 1, hidden_size, 1,
                        arena.stream);

  // 2. Layer stack
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    const auto& layer = weights.layers[l];
    const auto route_plan = ResolveQwenLayerRoute(
        policy, QwenExecutionMode::kDecode, layer.is_full_attention);
    const gufo::models::qwen::HipModuleContext module_ctx(
        static_cast<void*>(arena.stream), l, pos);

    const auto& rotation = layer.rotation;
    if (rotation.attn_in != nullptr) {
      LaunchRMSNormRotated(decode_scratch.hidden.data(),
                           static_cast<const float*>(layer.attn_norm.data),
                           decode_scratch.normed.data(), hidden_size, 1e-6F,
                           rotation.attn_in, arena.stream);
    } else {
      // Pre-RMSNorm, routed through the norm module (HIP backend). Same
      // kernel, same args, same arena slices (d_hidden in, d_normed out);
      // behavior identical to the former inline `LaunchRMSNorm` call.
      const auto norm_view =
          gufo::models::qwen::MakeAttnNormView(layer, config);
      gufo::models::qwen::NormForward(
          module_ctx, norm_view, decode_scratch.hidden, decode_scratch.normed);
    }

    if (layer.is_full_attention) {
      // Full attention path
      const std::size_t total_k = arena.GetAttentionKvPlaneElements();
      const std::uint32_t attn_layer_idx = l / config.full_attention_interval;

      if (IsExl3(layer.attn_q)) {
        const Exl3GroupOutput qkv[] = {
            {&layer.attn_q, ssm_scratch.qkv.data()},
            {&layer.attn_k, attention_scratch.k.data()},
            {&layer.attn_v, attention_scratch.v.data()}};
        LaunchExl3Projections(qkv, decode_scratch.normed.data(), 1, exl3,
                              arena.stream);
      } else {
        LaunchFusedQKVProjections(
            layer.attn_q.data, layer.attn_q.type, layer.attn_k.data,
            layer.attn_k.type, layer.attn_v.data, layer.attn_v.type,
            decode_scratch.normed.data(), ssm_scratch.qkv.data(),
            attention_scratch.k.data(), attention_scratch.v.data(),
            q_projection_size, kv_size, hidden_size, arena.stream);
      }
      // De-interleave Q and Gate from attn_q projection
      LaunchUnpackQG(ssm_scratch.qkv.data(), attention_scratch.q.data(),
                     ssm_scratch.gate.data(), config.num_attention_heads,
                     config.head_dim, arena.stream);

      // QK-Norm + RoPE + KV-cache write fused into one kernel
      // (opt-c010-qk-rope-kv). The unfused chain stays wired behind the
      // policy toggle as the independent reference.
      const bool fused_qknorm_rope_kv =
          route_plan.fuse_qk_norm_rope_kv &&
          detail::IsFusedQkNormSupported(config.head_dim);
      if (fused_qknorm_rope_kv) {
        LaunchFusedQKNormRoPEKvWrite(
            attention_scratch.q.data(), attention_scratch.k.data(),
            attention_scratch.v.data(),
            static_cast<const float*>(layer.attn_q_norm.data),
            static_cast<const float*>(layer.attn_k_norm.data),
            attention_scratch.q.data(), attention_scratch.k.data(),
            arena.d_kv_cache, OffsetIfPresent(arena.d_kv_cache, total_k),
            arena.d_attention_kv_f16,
            OffsetIfPresent(
                static_cast<std::uint16_t*>(arena.d_attention_kv_f16), total_k),
            arena.GetKvBlockTable(), attn_layer_idx, d_in_pos,
            arena.GetMaxContext(), config.num_attention_heads,
            config.num_key_value_heads, config.head_dim, config.rotary_dim,
            config.rope_theta, 1e-6F, arena.stream, rope);
      } else {
        if (!layer.attn_q_norm.empty()) {
          LaunchPerHeadRMSNorm(
              attention_scratch.q.data(),
              static_cast<const float*>(layer.attn_q_norm.data),
              attention_scratch.q.data(), config.num_attention_heads,
              config.head_dim, 1e-6F, arena.stream);
        }
        if (!layer.attn_k_norm.empty()) {
          LaunchPerHeadRMSNorm(
              attention_scratch.k.data(),
              static_cast<const float*>(layer.attn_k_norm.data),
              attention_scratch.k.data(), config.num_key_value_heads,
              config.head_dim, 1e-6F, arena.stream);
        }

        // RoPE (using device pos pointer for graph capture invariance)
        LaunchRoPE(attention_scratch.q.data(), attention_scratch.k.data(),
                   config.num_attention_heads, config.num_key_value_heads,
                   config.head_dim, config.rotary_dim, d_in_pos,
                   config.rope_theta, arena.stream, rope);
      }

      // Softmax Attention + Gating
      if (arena.UsesKvarn()) {
        // KVarN: the norm/RoPE step wrote no cache; rotate the rows into the
        // FP16 tail and attend over sink, quantized groups and tail. The
        // position comes from the device, so graph capture stays valid.
        const auto kv = arena.GetKvarnView();
        LaunchKvarnWriteRows(attention_scratch.k.data(),
                             attention_scratch.v.data(), kv, attn_layer_idx,
                             d_in_pos, 0, 1, arena.stream);
        LaunchKvarnDecodeAttention(
            attention_scratch.q.data(), ssm_scratch.gate.data(), kv,
            ssm_scratch.out.data(), attn_layer_idx, d_in_pos, pos, 1,
            kvarn_groups, 0, arena.GetKvarnScratch(), arena.stream);
      } else if (use_split_k_decode) {
        LaunchAttention(
            attention_scratch.q.data(), attention_scratch.k.data(),
            attention_scratch.v.data(), ssm_scratch.gate.data(),
            arena.d_kv_cache, OffsetIfPresent(arena.d_kv_cache, total_k),
            arena.d_attention_kv_f16,
            OffsetIfPresent(
                static_cast<std::uint16_t*>(arena.d_attention_kv_f16), total_k),
            arena.GetKvBlockTable(), ssm_scratch.out.data(), attn_layer_idx,
            pos, arena.GetMaxContext(), config.num_attention_heads,
            config.num_key_value_heads, config.head_dim, arena.stream,
            attention_scratch.split_k.data(), fused_qknorm_rope_kv);
      } else {
        LaunchAttention(
            attention_scratch.q.data(), attention_scratch.k.data(),
            attention_scratch.v.data(), ssm_scratch.gate.data(),
            arena.d_kv_cache, OffsetIfPresent(arena.d_kv_cache, total_k),
            arena.d_attention_kv_f16,
            OffsetIfPresent(
                static_cast<std::uint16_t*>(arena.d_attention_kv_f16), total_k),
            arena.GetKvBlockTable(), ssm_scratch.out.data(), attn_layer_idx,
            d_in_pos, arena.GetMaxContext(), config.num_attention_heads,
            config.num_key_value_heads, config.head_dim, arena.stream,
            fused_qknorm_rope_kv);
      }

      if (rotation.attn_out_in != nullptr) {
        LaunchRotateRowsFp32(ssm_scratch.out.data(), rotation.attn_out_in, 1,
                             attention_size, arena.stream);
      }
      // Output projection, routed through the quant_gemm module (HIP
      // backend). Same kernel, same args, same arena slices (d_ssm_out in,
      // d_attn_out out); behavior-identical to the former inline `LaunchGEMV`.
      if (IsExl3(layer.attn_output)) {
        LaunchExl3Projection(layer.attn_output, ssm_scratch.out.data(), 1,
                             attention_scratch.output.data(), exl3,
                             arena.stream);
      } else {
        gufo::models::qwen::QuantGemm(module_ctx, layer.attn_output,
                                      ssm_scratch.out.first(attention_size),
                                      hidden_size, attention_size,
                                      attention_scratch.output);
      }
    } else {
      // SSM path

      if (IsExl3(layer.attn_qkv)) {
        // EXL3 qkv and z; alpha and beta stay unquantized GGUF tensors.
        const Exl3GroupOutput qkvz[] = {
            {&layer.attn_qkv, ssm_scratch.qkv.data()},
            {&layer.attn_gate, ssm_scratch.gate.data()}};
        LaunchExl3Projections(qkvz, decode_scratch.normed.data(), 1, exl3,
                              arena.stream);
        LaunchGEMV(layer.ssm_alpha.data, layer.ssm_alpha.type,
                   decode_scratch.normed.data(), ssm_scratch.alpha.data(),
                   time_step_rank, hidden_size, arena.stream);
        LaunchGEMV(layer.ssm_beta.data, layer.ssm_beta.type,
                   decode_scratch.normed.data(), ssm_scratch.beta.data(),
                   time_step_rank, hidden_size, arena.stream);
      } else {
        LaunchFusedSSMInputProjections(
            layer.attn_qkv.data, layer.attn_qkv.type, layer.attn_gate.data,
            layer.attn_gate.type, layer.ssm_alpha.data, layer.ssm_alpha.type,
            layer.ssm_beta.data, layer.ssm_beta.type,
            decode_scratch.normed.data(), ssm_scratch.qkv.data(),
            ssm_scratch.gate.data(), ssm_scratch.alpha.data(),
            ssm_scratch.beta.data(), hidden_size, ssm_qkv_size, ssm_inner_size,
            time_step_rank, arena.stream);
      }
      LaunchSSMConvRecurrence(
          ssm_scratch.qkv.data(),
          static_cast<const float*>(layer.ssm_conv1d.data), recurrent_state,
          ssm_scratch.conv_out.data(), ssm_scratch.alpha.data(),
          ssm_scratch.beta.data(), static_cast<const float*>(layer.ssm_a.data),
          static_cast<const float*>(layer.ssm_dt.data),
          static_cast<const float*>(layer.ssm_norm.data),
          ssm_scratch.gate.data(), ssm_scratch.out.data(),
          config.SsmLayerIndex(l), ssm_qkv_size, config.ssm_group_count,
          config.ssm_time_step_rank, config.ssm_state_size,
          config.SsmValueSize(), arena.stream, arena.GetSsmReplayCapture(),
          arena.GetRecurrentStateStorage());

      if (rotation.ssm_out_in != nullptr) {
        LaunchRotateRowsFp32(ssm_scratch.out.data(), rotation.ssm_out_in, 1,
                             ssm_inner_size, arena.stream);
      }
      if (IsExl3(layer.ssm_out)) {
        LaunchExl3Projection(layer.ssm_out, ssm_scratch.out.data(), 1,
                             attention_scratch.output.data(), exl3,
                             arena.stream);
      } else {
        LaunchGEMV(layer.ssm_out.data, layer.ssm_out.type,
                   ssm_scratch.out.data(), attention_scratch.output.data(),
                   hidden_size, ssm_inner_size, arena.stream);
      }
    }

    {
      gufo::models::qwen::ResidualAdd(module_ctx, decode_scratch.hidden,
                                      attention_scratch.output);

      {
        // FFN Pre-RMSNorm
        if (rotation.ffn_in != nullptr) {
          LaunchRMSNormRotated(decode_scratch.hidden.data(),
                               static_cast<const float*>(layer.ffn_norm.data),
                               decode_scratch.normed.data(), hidden_size, 1e-6F,
                               rotation.ffn_in, arena.stream);
        } else {
          LaunchRMSNorm(decode_scratch.hidden.data(),
                        static_cast<const float*>(layer.ffn_norm.data),
                        decode_scratch.normed.data(), hidden_size, 1e-6F,
                        arena.stream);
        }
      }
    }

    // Fused SwiGLU FFN
    if (IsExl3(layer.ffn_gate)) {
      LaunchExl3SwiGLUMlp(layer.ffn_gate, layer.ffn_up, layer.ffn_down,
                          decode_scratch.normed.data(), 1,
                          ffn_scratch.gate.data(), ffn_scratch.up.data(),
                          ffn_scratch.activation.data(), ffn_scratch.out.data(),
                          exl3, arena.stream);
    } else if (rotation.ffn_down_in != nullptr) {
      // The module's two launches with the SwiGLU row rotated in between.
      const std::size_t intermediate_size = config.intermediate_size;
      LaunchFusedSwiGLUGEMV(layer.ffn_gate.data, layer.ffn_gate.type,
                            layer.ffn_up.data, layer.ffn_up.type,
                            decode_scratch.normed.data(),
                            ffn_scratch.activation.data(), intermediate_size,
                            hidden_size, arena.stream);
      LaunchRotateRowsFp32(ffn_scratch.activation.data(), rotation.ffn_down_in,
                           1, intermediate_size, arena.stream);
      LaunchGEMV(layer.ffn_down.data, layer.ffn_down.type,
                 ffn_scratch.activation.data(), ffn_scratch.out.data(),
                 hidden_size, intermediate_size, arena.stream);
    } else {
      // Non-fused FFN, routed through the module. Same fused SwiGLU kernel +
      // same down GEMV as the former inline calls; behavior identical. The
      // module reads the arena device spans (x = d_normed, act_scratch =
      // d_ffn_act, out = d_ffn_out) directly.
      const auto ffn_view = gufo::models::qwen::MakeFfnView(layer, config);
      gufo::models::qwen::FfnForward(
          module_ctx, ffn_view, decode_scratch.normed, ffn_scratch.activation,
          ffn_scratch.activation, ffn_scratch.activation, ffn_scratch.out);
    }

    // Residual Add
    gufo::models::qwen::ResidualAdd(module_ctx, decode_scratch.hidden,
                                    ffn_scratch.out);

    if (const auto tap_index = arena.GetTargetLayerCaptureIndex(l);
        tap_index.has_value()) {
      HIP_CHECK(hipMemcpyAsync(
          arena.d_target_layer_features + (*tap_index * hidden_size),
          decode_scratch.hidden.data(), hidden_size * sizeof(float),
          hipMemcpyDeviceToDevice, arena.stream));
    }
  }

  if (compute_logits) {
    // 3-4. Final output norm and LM head GEMV
    LaunchQwenOutputHead(weights, decode_scratch.hidden.data(),
                         decode_scratch.normed.data(),
                         decode_scratch.logits.data(), arena.stream, exl3);

    // 5. Parallel GPU Argmax
    LaunchBatchedGPUArgmax(decode_scratch.logits.data(), d_out_token, 1,
                           vocab_size, ffn_scratch.out, arena.stream);
  }
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
