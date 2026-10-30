#ifndef GUFO_MODELS_QWEN_EXL3_WEIGHTS_HPP_
#define GUFO_MODELS_QWEN_EXL3_WEIGHTS_HPP_

// Qwen3.5 / Qwen3.8 (GGUF architecture qwen35) tensors from an EXL3
// checkpoint, in the layout llama.cpp's convert_hf_to_gguf.py writes for the
// same Hugging Face model (upstream 8ab29fae, conversion/qwen.py
// Qwen3NextModel / _LinearAttentionVReorderBase / _QwenMtpMixin):
//
// - quantized projections become their effective weight
//   diag(suh) H W' H diag(svh) (docs/EXL3-FORMAT.md), rounded to BF16;
// - linear-attention V heads move from grouped to tiled order (qkv V rows,
//   z rows, alpha / beta rows, out_proj input columns, conv1d V channels,
//   A_log, dt_bias);
// - norm weights (except linear_attn.norm) get + 1 and A_log becomes
//   -exp(A_log), in FP32 (the BF16 GGUF holds them as F32);
// - mtp.* becomes block num_hidden_layers (nextn).
//
// This is the X1 correctness path (plan/exl3/XA-BRIEF.md): the produced
// tensors feed the existing BF16 GGUF code paths unchanged.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/exl3/checkpoint.hpp"

namespace gufo::models::qwen_exl3 {

/// Linear-attention head geometry (GGUF qwen35.ssm.* keys).
struct Geometry {
  int num_layers = 0;   // num_hidden_layers (the MTP block follows)
  int num_k_heads = 0;  // ssm.group_count
  int num_v_heads = 0;  // ssm.time_step_rank
  int head_k_dim = 0;   // ssm.state_size
  int head_v_dim = 0;   // ssm.inner_size / num_v_heads
};

/// Geometry from a qwen35 GGUF's metadata; nullopt with `error` set when a
/// key is missing.
[[nodiscard]] std::optional<Geometry> GeometryFromGguf(
    const core::GgufReader& reader, std::string* error);

/// What a GGUF tensor of the qwen35 layout is built from.
struct TensorPlan {
  std::string gguf_name;
  core::GgmlType type = core::GgmlType::kF32;
  std::vector<std::uint64_t> ne;  // GGUF dims, ne[0] contiguous
  std::size_t bytes = 0;
};

class Source {
public:
  Source(const quant::exl3::Checkpoint& checkpoint, Geometry geometry)
      : checkpoint_(checkpoint), geometry_(geometry) {}

  /// The plan for `gguf_name`, or nullopt when the checkpoint has no source
  /// for it (the reason in `error`).
  [[nodiscard]] std::optional<TensorPlan> Plan(std::string_view gguf_name,
                                               std::string* error) const;

  /// Writes the tensor bytes (plan.bytes) into `out`. Quantized projections
  /// use `threads` worker threads.
  void Produce(const TensorPlan& plan, std::span<std::uint8_t> out,
               int threads) const;

private:
  const quant::exl3::Checkpoint& checkpoint_;
  Geometry geometry_;
};

/// BF16 of `value`, round to nearest even (NaN stays NaN).
[[nodiscard]] std::uint16_t FloatToBf16(float value) noexcept;
[[nodiscard]] float Bf16ToFloat(std::uint16_t value) noexcept;

}  // namespace gufo::models::qwen_exl3

#endif  // GUFO_MODELS_QWEN_EXL3_WEIGHTS_HPP_
