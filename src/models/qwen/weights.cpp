#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "src/core/quant/rotation.hpp"
#include "src/models/qwen/state.hpp"

namespace gufo::models {
namespace {

// EXL3 projections (QwenModelWeights::LoadFromGguf `projections`), consulted
// when the reader does not hold a tensor. Set only while LoadFromGguf runs.
thread_local const std::vector<std::pair<std::string, QwenTensorRef>>*
    g_projections = nullptr;

QwenTensorRef ExtractTensorRef(const core::GgufReader& reader,
                               std::string_view name) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr && g_projections != nullptr) {
    for (const auto& [projection_name, ref] : *g_projections) {
      if (projection_name == name) {
        return ref;
      }
    }
  }
  if (tensor == nullptr || tensor->data == nullptr) {
    return {};
  }

  const auto tensor_address = reinterpret_cast<std::uintptr_t>(tensor->data);
  std::size_t available_bytes = 0;
  for (const auto& region : reader.GetMappedRegions()) {
    const auto region_address = reinterpret_cast<std::uintptr_t>(region.data);
    if (tensor_address >= region_address) {
      const auto offset = tensor_address - region_address;
      if (offset < region.size) {
        available_bytes = region.size - offset;
        break;
      }
    }
  }
  return {.data = tensor->data,
          .type = tensor->type,
          .num_elements = tensor->ElementCount(),
          .available_bytes = available_bytes};
}

enum class TensorRole : std::uint8_t {
  kEmbedding,
  kNorm,
  kProjection,
  kSsmParameter,
};

[[nodiscard]] constexpr std::string_view TensorRoleName(
    TensorRole role) noexcept {
  switch (role) {
    case TensorRole::kEmbedding:
      return "embedding";
    case TensorRole::kNorm:
      return "normalization";
    case TensorRole::kProjection:
      return "projection";
    case TensorRole::kSsmParameter:
      return "SSM parameter";
  }
  return "unknown";
}

[[nodiscard]] constexpr bool SupportsTensorType(core::GgmlType type,
                                                TensorRole role) noexcept {
  switch (role) {
    case TensorRole::kEmbedding:
      // The production HIP embedding kernel has exact F32, BF16, and Q8_0
      // implementations. Treating every other type as Q8_0 would decode the
      // wrong block layout.
      // opt-q4kxl: token_embd.weight is Q4_K in the UD-Q4_K_XL shard; the
      // lookup kernels decode any DecodeQuantSub16-backed format directly.
      return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16 ||
             type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ4_K ||
             type == core::GgmlType::kQ3_K || type == core::GgmlType::kQ5_K ||
             type == core::GgmlType::kQ6_K || type == core::GgmlType::kIQ4_NL ||
             type == core::GgmlType::kIQ4_XS || type == core::GgmlType::kIQ3_S;
    case TensorRole::kNorm:
    case TensorRole::kSsmParameter:
      // HIP norm, convolution, and recurrence kernels consume these tensors as
      // float pointers. Accepting BF16 or packed data would reinterpret bytes.
      return type == core::GgmlType::kF32;
    case TensorRole::kProjection:
      // opt-q4kxl: the Unsloth UD-Q4_K_XL shard mixes Q5_K, IQ4_XS, Q4_K, Q6_K,
      // IQ4_NL, Q3_K and IQ3_S across projections, and every one of them has an
      // in-kernel decoder, so all are accepted here.
      return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16 ||
             type == core::GgmlType::kQ8_K || type == core::GgmlType::kQ8_0 ||
             type == core::GgmlType::kQ5_K || type == core::GgmlType::kQ6_K ||
             type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ3_K ||
             type == core::GgmlType::kIQ4_NL ||
             type == core::GgmlType::kIQ4_XS || type == core::GgmlType::kIQ3_S;
  }
  return false;
}

bool ValidateTensor(const QwenTensorRef& tensor, std::size_t expected_elements,
                    TensorRole role, std::string_view name,
                    std::string* error_msg, std::size_t row_elements = 0) {
  if (tensor.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "Missing required Qwen tensor: " + std::string(name);
    }
    return false;
  }
  if (tensor.exl3 != nullptr) {
    // EXL3 projection: W [in][out] with in = row_elements.
    const auto& linear = *tensor.exl3;
    const std::size_t elements = static_cast<std::size_t>(linear.k) * linear.n;
    if (role != TensorRole::kProjection || elements != expected_elements ||
        (row_elements != 0 && linear.k != row_elements)) {
      if (error_msg != nullptr) {
        *error_msg = "EXL3 tensor shape or role mismatch for " +
                     std::string(name) + ": " + std::to_string(linear.k) +
                     " x " + std::to_string(linear.n);
      }
      return false;
    }
    return true;
  }
  if (!SupportsTensorType(tensor.type, role)) {
    if (error_msg != nullptr) {
      *error_msg = "Unsupported " + std::string(TensorRoleName(role)) +
                   " tensor type for " + std::string(name) + ": " +
                   std::string(core::ToString(tensor.type));
    }
    return false;
  }
  if (tensor.num_elements != expected_elements) {
    if (error_msg != nullptr) {
      *error_msg = "Tensor shape mismatch for " + std::string(name) +
                   ": expected " + std::to_string(expected_elements) +
                   " elements, found " + std::to_string(tensor.num_elements);
    }
    return false;
  }
  if (!tensor.FitsAvailableStorage()) {
    if (error_msg != nullptr) {
      *error_msg = "Tensor payload is truncated or has invalid encoded size: " +
                   std::string(name);
    }
    return false;
  }
  const bool quantized = tensor.type != core::GgmlType::kF32 &&
                         tensor.type != core::GgmlType::kF16 &&
                         tensor.type != core::GgmlType::kBF16;
  if (quantized && row_elements != 0 &&
      gufo::quant::QuantizedRowBytes(tensor.type, row_elements) == 0) {
    if (error_msg != nullptr) {
      *error_msg =
          "Quantized tensor row is not block aligned: " + std::string(name) +
          " (row elements " + std::to_string(row_elements) + ")";
    }
    return false;
  }
  return true;
}

// First of `names` present in the file, or empty. Mirrors the alias order of
// the tensor binding below.
std::string PresentName(const core::GgufReader& reader,
                        std::initializer_list<std::string_view> names) {
  for (const auto name : names) {
    if (reader.FindTensor(name) != nullptr) {
      return std::string(name);
    }
  }
  return {};
}

// Qwen 27B input groups (plan/archive/rotated-quant/PLAN.md, "27B 입력 그룹");
// the converter's tools/qwen27b/rotquant_groups.py defines the same groups.
// Only layers of the main stack are listed, so a folded weight of an embedded
// MTP (nextn) layer is refused and never counts toward "rotated in some layers
// only". ssm_alpha and ssm_beta read the attn_in activation but are not
// foldable upstream; they stay unrotated in the file and are folded at load
// (FoldAlphaBeta) so every DeltaNet input projection reads one rotated row.
gufo::quant::rotation::GroupTable BuildRotationGroups(
    const core::GgufReader& reader, const QwenModelWeights& weights) {
  gufo::quant::rotation::GroupTable table;
  const auto add = [&](std::string group, std::string key,
                       std::vector<std::string> names) {
    std::erase(names, std::string{});
    table.instances.push_back(
        {std::move(group), std::move(key), std::move(names)});
  };
  for (std::uint32_t i = 0; i < weights.config.num_layers; ++i) {
    const std::string p = "blk." + std::to_string(i) + ".";
    const std::string l = ":" + std::to_string(i);
    const auto name = [&](std::initializer_list<std::string_view> suffixes) {
      std::vector<std::string> full;
      for (const auto suffix : suffixes) {
        full.push_back(p + std::string(suffix));
      }
      for (const auto& candidate : full) {
        if (reader.FindTensor(candidate) != nullptr) {
          return candidate;
        }
      }
      return std::string{};
    };
    if (weights.layers[i].is_full_attention) {
      add("attn_in", "attn_in" + l,
          {name({"attn_q.weight", "wq.weight"}),
           name({"attn_k.weight", "wk.weight"}),
           name({"attn_v.weight", "wv.weight"})});
      add("attn_out_in", "attn_out_in" + l,
          {name({"attn_output.weight", "attn_out.weight", "wo.weight"})});
    } else {
      add("attn_in", "attn_in" + l,
          {name({"attn_qkv.weight", "wqkv.weight"}),
           name({"attn_gate.weight", "wqkv_gate.weight"})});
      add("ssm_out_in", "ssm_out_in" + l, {name({"ssm_out.weight"})});
    }
    add("ffn_in", "ffn_in" + l,
        {name({"ffn_gate.weight"}), name({"ffn_up.weight"})});
    add("ffn_down_in", "ffn_down_in" + l, {name({"ffn_down.weight"})});
  }
  const std::string head = PresentName(reader, {"output.weight"});
  if (!head.empty()) {
    add("head_in", "head_in", {head});
  }
  return table;
}

// float -> IEEE binary16 bits, round to nearest even.
std::uint16_t FloatToHalf(float value) noexcept {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16) & 0x8000U;
  const std::uint32_t magnitude = bits & 0x7FFFFFFFU;
  if (magnitude >= 0x7F800000U) {
    return static_cast<std::uint16_t>(sign | 0x7C00U |
                                      (magnitude > 0x7F800000U ? 0x200U : 0U));
  }
  if (magnitude >= 0x477FF000U) {  // rounds to beyond the largest half
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  if (magnitude < 0x38800000U) {  // subnormal half (or zero)
    const float scaled = std::bit_cast<float>(magnitude) * 16777216.0F;
    return static_cast<std::uint16_t>(
        sign | static_cast<std::uint32_t>(std::nearbyint(scaled)));
  }
  const std::uint32_t rounded =
      magnitude + 0xFFFU + ((magnitude >> 13) & 1U) - (112U << 23);
  return static_cast<std::uint16_t>(sign | (rounded >> 13));
}

// float -> BF16 bits, round to nearest even (finite input).
std::uint16_t FloatToBf16(float value) noexcept {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >>
                                    16);
}

// One row of `tensor` (K elements) as FP32.
void DecodeRow(const QwenTensorRef& tensor, std::size_t row, std::size_t k,
               std::span<float> out) {
  const auto* base = static_cast<const std::uint8_t*>(tensor.data);
  switch (tensor.type) {
    case core::GgmlType::kF32:
      std::memcpy(out.data(), base + row * k * sizeof(float),
                  k * sizeof(float));
      return;
    case core::GgmlType::kBF16:
      for (std::size_t i = 0; i < k; ++i) {
        std::uint16_t half = 0;
        std::memcpy(&half, base + (row * k + i) * sizeof(half), sizeof(half));
        out[i] = std::bit_cast<float>(static_cast<std::uint32_t>(half) << 16);
      }
      return;
    case core::GgmlType::kQ8_0:
      gufo::quant::DequantizeQ8_0(
          base + row * gufo::quant::QuantizedRowBytes(tensor.type, k),
          out.data(), k);
      return;
    default:
      return;
  }
}

// One FP32 row encoded back to `type`, with the reference rounding of each
// format (ggml quantize_row_q8_0_ref for Q8_0).
void EncodeRow(core::GgmlType type, std::span<const float> row,
               std::uint8_t* out) {
  switch (type) {
    case core::GgmlType::kF32:
      std::memcpy(out, row.data(), row.size() * sizeof(float));
      return;
    case core::GgmlType::kBF16:
      for (std::size_t i = 0; i < row.size(); ++i) {
        const std::uint16_t half = FloatToBf16(row[i]);
        std::memcpy(out + i * sizeof(half), &half, sizeof(half));
      }
      return;
    case core::GgmlType::kQ8_0: {
      constexpr std::size_t kBlock = 32;
      for (std::size_t b = 0; b < row.size() / kBlock; ++b) {
        float amax = 0.0F;
        for (std::size_t j = 0; j < kBlock; ++j) {
          amax = std::max(amax, std::abs(row[b * kBlock + j]));
        }
        const float d = amax / 127.0F;
        const float id = d != 0.0F ? 1.0F / d : 0.0F;
        gufo::quant::block_q8_0 block{};
        block.d = FloatToHalf(d);
        for (std::size_t j = 0; j < kBlock; ++j) {
          block.qs[j] =
              static_cast<std::int8_t>(std::round(row[b * kBlock + j] * id));
        }
        std::memcpy(out + b * sizeof(block), &block, sizeof(block));
      }
      return;
    }
    default:
      return;
  }
}

// Folds ssm_alpha and ssm_beta of every DeltaNet layer into the rotated
// basis, W' = W·D·H with the attn_in (hidden-width) signs, re-encoded in
// their own type so every prefill and decode route selection is unchanged.
// Each layer's alpha and beta stay adjacent for the packed verification
// path. The file keeps them unrotated (upstream llama.cpp cannot fold them).
bool FoldAlphaBeta(QwenModelWeights& weights, const std::uint32_t* bits,
                   std::string* error_msg) {
  const auto start = std::chrono::steady_clock::now();
  const std::size_t k = weights.config.hidden_size;
  std::size_t total = 0;
  for (std::size_t i = 0; i < weights.layers.size(); ++i) {
    auto& layer = weights.layers[i];
    if (layer.is_full_attention) {
      continue;
    }
    for (const auto* tensor : {&layer.ssm_alpha, &layer.ssm_beta}) {
      if (tensor->type != core::GgmlType::kF32 &&
          tensor->type != core::GgmlType::kBF16 &&
          tensor->type != core::GgmlType::kQ8_0) {
        if (error_msg != nullptr) {
          *error_msg = "Rotated GGUF refused: blk." + std::to_string(i) +
                       " ssm_alpha/ssm_beta type " +
                       std::string(core::ToString(tensor->type)) +
                       " cannot be folded into the rotated attn_in basis "
                       "(F32, BF16 and Q8_0 are supported)";
        }
        return false;
      }
      total += tensor->EncodedSizeBytes();
    }
  }
  auto storage = std::make_shared<std::vector<std::uint8_t>>(total);
  const std::span<const std::uint32_t> signs(
      bits, gufo::quant::rotation::PackedSignWords(k));
  std::vector<float> row(k);
  std::size_t offset = 0;
  for (auto& layer : weights.layers) {
    if (layer.is_full_attention) {
      continue;
    }
    for (auto* tensor : {&layer.ssm_alpha, &layer.ssm_beta}) {
      const std::size_t rows = tensor->num_elements / k;
      const std::size_t row_bytes = tensor->EncodedSizeBytes() / rows;
      std::uint8_t* destination = storage->data() + offset;
      for (std::size_t r = 0; r < rows; ++r) {
        DecodeRow(*tensor, r, k, row);
        gufo::quant::rotation::RotateRow(row, signs);
        EncodeRow(tensor->type, row, destination + r * row_bytes);
      }
      tensor->data = destination;
      tensor->available_bytes = total - offset;
      offset += rows * row_bytes;
    }
  }
  weights.folded_weights = std::move(storage);
  weights.folded_bytes = total;
  weights.fold_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  return true;
}

// Reads the prism.hadamard metadata, binds one sign vector per input width
// to every rotated group (5120: attn_in, ffn_in, head_in; 6144: ssm_out_in,
// attn_out_in; 17408: ffn_down_in) and folds alpha/beta.
bool BindRotation(const core::GgufReader& reader, QwenModelWeights& weights,
                  std::string* error_msg) {
  const auto info = gufo::quant::rotation::ReadRotation(
      reader, BuildRotationGroups(reader, weights), error_msg);
  if (!info.has_value()) {
    return false;
  }
  if (!info->Any()) {
    return true;
  }
  auto storage = std::make_shared<std::vector<std::uint32_t>>();
  std::size_t words = 0;
  for (const auto& width : info->widths) {
    words += width.bits.size();
  }
  storage->reserve(words);
  std::vector<std::pair<std::size_t, std::size_t>> offsets;
  for (const auto& width : info->widths) {
    offsets.emplace_back(width.k, storage->size());
    storage->insert(storage->end(), width.bits.begin(), width.bits.end());
  }
  const auto signs = [&](std::string_view group,
                         std::size_t k) -> const std::uint32_t* {
    if (!info->IsGroupRotated(group)) {
      return nullptr;
    }
    for (const auto& [width, offset] : offsets) {
      if (width == k) {
        return storage->data() + offset;
      }
    }
    return nullptr;
  };
  const std::size_t hidden = weights.config.hidden_size;
  const std::size_t attention = weights.config.AttentionSize();
  const std::size_t inner = weights.config.ssm_inner_size;
  const std::size_t intermediate = weights.config.intermediate_size;
  for (auto& layer : weights.layers) {
    auto& rotation = layer.rotation;
    rotation.attn_in = signs("attn_in", hidden);
    rotation.ffn_in = signs("ffn_in", hidden);
    rotation.ffn_down_in = signs("ffn_down_in", intermediate);
    if (layer.is_full_attention) {
      rotation.attn_out_in = signs("attn_out_in", attention);
    } else {
      rotation.ssm_out_in = signs("ssm_out_in", inner);
    }
  }
  weights.head_in_rotation = signs("head_in", hidden);
  const auto* attn_in = signs("attn_in", hidden);
  weights.rotation_signs = std::move(storage);
  weights.rotated_groups = info->GroupList();
  if (attn_in != nullptr) {
    return FoldAlphaBeta(weights, attn_in, error_msg);
  }
  return true;
}

}  // namespace

std::optional<QwenModelWeights> QwenModelWeights::LoadFromGguf(
    const core::GgufReader& reader, std::string* error_msg,
    const std::vector<std::pair<std::string, QwenTensorRef>>* projections) {
  g_projections = projections;
  struct Reset {
    ~Reset() { g_projections = nullptr; }
  } reset;
  const auto config_opt = reader.ExtractModelConfig(error_msg);
  if (!config_opt.has_value()) {
    return std::nullopt;
  }

  QwenModelWeights weights;
  weights.config = *config_opt;

  weights.token_embd = ExtractTensorRef(reader, "token_embd.weight");
  weights.output_norm = ExtractTensorRef(reader, "output_norm.weight");
  weights.output = ExtractTensorRef(reader, "output.weight");
  if (weights.output.empty()) {
    // Tied LM head shares token embeddings
    weights.output = weights.token_embd;
  }
  const std::size_t hidden_size = weights.config.hidden_size;
  const std::size_t vocab_size = weights.config.vocab_size;
  if (!ValidateTensor(weights.token_embd, vocab_size * hidden_size,
                      TensorRole::kEmbedding, "token_embd.weight", error_msg,
                      hidden_size) ||
      !ValidateTensor(weights.output_norm, hidden_size, TensorRole::kNorm,
                      "output_norm.weight", error_msg) ||
      !ValidateTensor(weights.output, vocab_size * hidden_size,
                      TensorRole::kProjection, "output.weight", error_msg,
                      hidden_size)) {
    return std::nullopt;
  }

  weights.layers.resize(weights.config.num_layers);
  for (std::uint32_t i = 0; i < weights.config.num_layers; ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    auto& l = weights.layers[i];

    l.attn_norm = ExtractTensorRef(reader, prefix + "attn_norm.weight");
    if (l.attn_norm.empty()) {
      l.attn_norm = ExtractTensorRef(reader, prefix + "input_norm.weight");
    }

    l.ffn_norm = ExtractTensorRef(reader, prefix + "ffn_norm.weight");
    if (l.ffn_norm.empty()) {
      l.ffn_norm =
          ExtractTensorRef(reader, prefix + "post_attention_norm.weight");
    }
    if (l.ffn_norm.empty()) {
      l.ffn_norm = ExtractTensorRef(reader, prefix + "attn_post_norm.weight");
    }

    // Check if full attention layer
    l.attn_q = ExtractTensorRef(reader, prefix + "attn_q.weight");
    if (l.attn_q.empty()) {
      l.attn_q = ExtractTensorRef(reader, prefix + "wq.weight");
    }

    if (!l.attn_q.empty()) {
      l.is_full_attention = true;
      l.attn_k = ExtractTensorRef(reader, prefix + "attn_k.weight");
      if (l.attn_k.empty()) {
        l.attn_k = ExtractTensorRef(reader, prefix + "wk.weight");
      }
      l.attn_v = ExtractTensorRef(reader, prefix + "attn_v.weight");
      if (l.attn_v.empty()) {
        l.attn_v = ExtractTensorRef(reader, prefix + "wv.weight");
      }
      l.attn_output = ExtractTensorRef(reader, prefix + "attn_output.weight");
      if (l.attn_output.empty()) {
        l.attn_output = ExtractTensorRef(reader, prefix + "attn_out.weight");
      }
      if (l.attn_output.empty()) {
        l.attn_output = ExtractTensorRef(reader, prefix + "wo.weight");
      }
      l.attn_q_norm = ExtractTensorRef(reader, prefix + "attn_q_norm.weight");
      l.attn_k_norm = ExtractTensorRef(reader, prefix + "attn_k_norm.weight");
    } else {
      l.is_full_attention = false;
      l.attn_qkv = ExtractTensorRef(reader, prefix + "attn_qkv.weight");
      if (l.attn_qkv.empty()) {
        l.attn_qkv = ExtractTensorRef(reader, prefix + "wqkv.weight");
      }
      l.attn_gate = ExtractTensorRef(reader, prefix + "attn_gate.weight");
      if (l.attn_gate.empty()) {
        l.attn_gate = ExtractTensorRef(reader, prefix + "wqkv_gate.weight");
      }
      l.ssm_a = ExtractTensorRef(reader, prefix + "ssm_a");
      if (l.ssm_a.empty()) {
        l.ssm_a = ExtractTensorRef(reader, prefix + "ssm_a.weight");
      }
      l.ssm_conv1d = ExtractTensorRef(reader, prefix + "ssm_conv1d.weight");
      if (l.ssm_conv1d.empty()) {
        l.ssm_conv1d = ExtractTensorRef(reader, prefix + "ssm_conv1d");
      }
      l.ssm_dt = ExtractTensorRef(reader, prefix + "ssm_dt.bias");
      if (l.ssm_dt.empty()) {
        l.ssm_dt = ExtractTensorRef(reader, prefix + "ssm_dt.weight");
      }
      if (l.ssm_dt.empty()) {
        l.ssm_dt = ExtractTensorRef(reader, prefix + "ssm_dt");
      }
      l.ssm_alpha = ExtractTensorRef(reader, prefix + "ssm_alpha.weight");
      l.ssm_beta = ExtractTensorRef(reader, prefix + "ssm_beta.weight");
      l.ssm_norm = ExtractTensorRef(reader, prefix + "ssm_norm.weight");
      l.ssm_out = ExtractTensorRef(reader, prefix + "ssm_out.weight");
    }

    l.ffn_gate = ExtractTensorRef(reader, prefix + "ffn_gate.weight");
    l.ffn_up = ExtractTensorRef(reader, prefix + "ffn_up.weight");
    l.ffn_down = ExtractTensorRef(reader, prefix + "ffn_down.weight");

    const bool expected_full_attention =
        ((i + 1) % weights.config.full_attention_interval) == 0;
    if (l.is_full_attention != expected_full_attention) {
      if (error_msg != nullptr) {
        *error_msg =
            "Unexpected Qwen layer kind at blk." + std::to_string(i) +
            ": expected " +
            (expected_full_attention ? "full attention" : "Gated DeltaNet");
      }
      return std::nullopt;
    }

    const std::size_t intermediate_size = weights.config.intermediate_size;
    if (!ValidateTensor(l.attn_norm, hidden_size, TensorRole::kNorm,
                        prefix + "attn_norm.weight", error_msg) ||
        !ValidateTensor(l.ffn_norm, hidden_size, TensorRole::kNorm,
                        prefix + "post_attention_norm.weight", error_msg) ||
        !ValidateTensor(l.ffn_gate, intermediate_size * hidden_size,
                        TensorRole::kProjection, prefix + "ffn_gate.weight",
                        error_msg, hidden_size) ||
        !ValidateTensor(l.ffn_up, intermediate_size * hidden_size,
                        TensorRole::kProjection, prefix + "ffn_up.weight",
                        error_msg, hidden_size) ||
        !ValidateTensor(l.ffn_down, hidden_size * intermediate_size,
                        TensorRole::kProjection, prefix + "ffn_down.weight",
                        error_msg, intermediate_size)) {
      return std::nullopt;
    }

    if (l.is_full_attention) {
      const std::size_t attention_size = weights.config.AttentionSize();
      const std::size_t kv_size =
          static_cast<std::size_t>(weights.config.num_key_value_heads) *
          weights.config.head_dim;
      if (!ValidateTensor(l.attn_q, 2 * attention_size * hidden_size,
                          TensorRole::kProjection, prefix + "attn_q.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.attn_k, kv_size * hidden_size,
                          TensorRole::kProjection, prefix + "attn_k.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.attn_v, kv_size * hidden_size,
                          TensorRole::kProjection, prefix + "attn_v.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.attn_output, hidden_size * attention_size,
                          TensorRole::kProjection,
                          prefix + "attn_output.weight", error_msg,
                          attention_size) ||
          !ValidateTensor(l.attn_q_norm, weights.config.head_dim,
                          TensorRole::kNorm, prefix + "attn_q_norm.weight",
                          error_msg) ||
          !ValidateTensor(l.attn_k_norm, weights.config.head_dim,
                          TensorRole::kNorm, prefix + "attn_k_norm.weight",
                          error_msg)) {
        return std::nullopt;
      }
    } else {
      const std::size_t qkv_size = weights.config.SsmQkvSize();
      const std::size_t inner_size = weights.config.ssm_inner_size;
      const std::size_t rank = weights.config.ssm_time_step_rank;
      if (!ValidateTensor(l.attn_qkv, qkv_size * hidden_size,
                          TensorRole::kProjection, prefix + "attn_qkv.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.attn_gate, inner_size * hidden_size,
                          TensorRole::kProjection, prefix + "attn_gate.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.ssm_a, rank, TensorRole::kSsmParameter,
                          prefix + "ssm_a", error_msg) ||
          !ValidateTensor(l.ssm_conv1d,
                          qkv_size * weights.config.ssm_conv_kernel,
                          TensorRole::kSsmParameter,
                          prefix + "ssm_conv1d.weight", error_msg) ||
          !ValidateTensor(l.ssm_dt, rank, TensorRole::kSsmParameter,
                          prefix + "ssm_dt.bias", error_msg) ||
          !ValidateTensor(l.ssm_alpha, rank * hidden_size,
                          TensorRole::kProjection, prefix + "ssm_alpha.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.ssm_beta, rank * hidden_size,
                          TensorRole::kProjection, prefix + "ssm_beta.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.ssm_norm, weights.config.SsmValueSize(),
                          TensorRole::kSsmParameter, prefix + "ssm_norm.weight",
                          error_msg) ||
          !ValidateTensor(l.ssm_out, hidden_size * inner_size,
                          TensorRole::kProjection, prefix + "ssm_out.weight",
                          error_msg, inner_size)) {
        return std::nullopt;
      }
    }
  }

  if (!BindRotation(reader, weights, error_msg)) {
    return std::nullopt;
  }
  return weights;
}

}  // namespace gufo::models
