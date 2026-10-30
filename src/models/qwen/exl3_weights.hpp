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
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/exl3/checkpoint.hpp"
#include "src/core/quant/exl3/device.hpp"
#include "src/models/qwen/state.hpp"

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

/// One raw GGUF key/value entry (key, type and value bytes as stored).
struct RawKv {
  std::string key;
  std::vector<std::uint8_t> bytes;
};

/// The key/value entries of a GGUF file, in order.
[[nodiscard]] std::vector<RawKv> ReadRawMetadata(
    const std::filesystem::path& path);

/// A GGUF string key/value entry.
[[nodiscard]] RawKv StringKv(std::string_view key, std::string_view value);

/// GGUF key naming the EXL3 checkpoint directory of a sidecar (relative to
/// the sidecar's directory).
inline constexpr std::string_view kCheckpointKey = "gufo.exl3.checkpoint";
/// Optional sidecar key: "bf16" keeps the token embedding in BF16 (the
/// X2a embedding comparison); otherwise it is quantized to Q8_0.
inline constexpr std::string_view kEmbeddingKey = "gufo.exl3.embedding";

/// The projection host copy of one EXL3 linear in GGUF orientation: the
/// linear-attention V-head reorders become permutations of whole 128-wide
/// tile blocks (exact: H acts within those blocks).
struct HostLinear {
  std::string gguf_name;
  std::vector<std::uint16_t> trellis;
  std::vector<std::uint16_t> suh;
  std::vector<std::uint16_t> svh;
  quant::exl3::DeviceLinear view;  // pointers into the vectors above
};

/// Embedding storage of a directly loaded checkpoint.
enum class EmbeddingFormat : std::uint8_t { kBf16, kQ8_0 };

/// The tensor names the 27B target forward binds (no MTP block), in GGUF
/// naming.
[[nodiscard]] std::vector<std::string> TargetTensorNames(
    const core::ModelConfig& config);

/// Q8_0 blocks of `values` (llama.cpp quantize_row_q8_0_ref); values.size()
/// must be a multiple of 32.
void QuantizeQ8_0(std::span<const float> values, std::uint8_t* out);

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

  /// Whether `gguf_name` is an EXL3 projection.
  [[nodiscard]] bool IsProjection(std::string_view gguf_name) const;

  /// Host copy of the EXL3 projection `gguf_name` (block-permuted).
  [[nodiscard]] HostLinear PrepareLinear(std::string_view gguf_name) const;

  /// A GGUF image holding `metadata` and every tensor of `names` that is not
  /// an EXL3 projection, with the embedding stored as `embedding`.
  [[nodiscard]] std::vector<std::uint8_t> BuildImage(
      const std::vector<RawKv>& metadata, const std::vector<std::string>& names,
      EmbeddingFormat embedding, int threads) const;

private:
  const quant::exl3::Checkpoint& checkpoint_;
  Geometry geometry_;
};

/// Checks a metadata sidecar against the checkpoint it names: config.json
/// hyperparameters and tokenizer.json vocabulary against the GGUF keys and
/// `tokenizer.ggml.tokens`. Returns an empty string when consistent, or the
/// first mismatch by name.
[[nodiscard]] std::string CheckSidecar(const std::filesystem::path& checkpoint,
                                       const core::GgufReader& sidecar);

/// Direct EXL3 loading (X2a): the sidecar's metadata plus an in-memory GGUF
/// of the unquantized tensors (the returned reader owns its bytes), and the
/// target weights with EXL3 projections (QwenTensorRef::exl3, host
/// pointers owned by weights.exl3_storage).
struct LoadedExl3 {
  std::shared_ptr<const core::GgufReader> reader;
  QwenModelWeights weights;
  std::size_t image_bytes = 0;
  std::size_t projection_bytes = 0;  // trellis + suh + svh
};

/// Loads the checkpoint named by `sidecar_path`'s kCheckpointKey. Returns
/// nullopt with `error` set on any problem.
[[nodiscard]] std::optional<LoadedExl3> LoadFromSidecar(
    const std::filesystem::path& sidecar_path, const core::GgufReader& sidecar,
    EmbeddingFormat embedding, std::string* error, int threads = 8);

/// The effective weight of `linear` as BF16 [n][k] (no reorders), the X1
/// computation; used to check host copies against Source::Produce.
void EffectiveBf16(const quant::exl3::DeviceLinear& linear, std::uint16_t* out,
                   int threads);

/// BF16 of `value`, round to nearest even (NaN stays NaN).
[[nodiscard]] std::uint16_t FloatToBf16(float value) noexcept;
[[nodiscard]] float Bf16ToFloat(std::uint16_t value) noexcept;

}  // namespace gufo::models::qwen_exl3

#endif  // GUFO_MODELS_QWEN_EXL3_WEIGHTS_HPP_
