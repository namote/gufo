// Qwen 27B rotated-GGUF binding (prism.hadamard) on a tiny synthetic qwen35
// model: the group table handed to src/core/quant/rotation, one sign vector
// per input width shared by every layer and group, the embedded MTP (nextn)
// layer, the load-time fold of ssm_alpha/ssm_beta into the rotated basis
// (F32, BF16, Q8_0), and the CPU, MTP and v1 refusals. Also measures the
// fold error of alpha/beta-sized matrices for R2 (printed, not judged).

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/rotation.hpp"
#include "src/models/qwen/generator.hpp"
#include "src/models/qwen/mtp_reference.hpp"
#include "src/models/qwen/state.hpp"
#include "tests/core/quant/rotation_reference.hpp"
#include "tests/core/quant/rotation_test_gguf.hpp"

namespace {

using gufo::core::GgmlType;
using gufo::models::QwenModelWeights;
using gufo::test::rotation::GgufImage;

int failures = 0;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

// hidden 256, 4 main layers (blk.3 full attention) plus one nextn layer,
// intermediate 384, attention 2 x 128 = 256, SSM inner 128.
constexpr std::uint64_t kHidden = 256;
constexpr std::uint64_t kFfn = 384;
constexpr std::uint64_t kAttention = 256;
constexpr std::uint64_t kInner = 128;
constexpr std::uint64_t kQkv = 384;
constexpr std::uint64_t kVocab = 16;

float Value(std::size_t i) {
  return static_cast<float>(static_cast<int>(i * 7919U % 61U) - 30) / 32.0F;
}

std::vector<float> Ramp(std::size_t n, std::size_t seed) {
  std::vector<float> values(n);
  for (std::size_t i = 0; i < n; ++i) {
    values[i] = Value(i + seed * 131U);
  }
  return values;
}

std::vector<std::uint8_t> EncodeBf16(const std::vector<float>& values) {
  std::vector<std::uint8_t> out(values.size() * 2);
  for (std::size_t i = 0; i < values.size(); ++i) {
    const auto bits = std::bit_cast<std::uint32_t>(values[i]);
    const auto half = static_cast<std::uint16_t>(
        (bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
    std::memcpy(out.data() + i * 2, &half, 2);
  }
  return out;
}

// ggml quantize_row_q8_0_ref.
std::vector<std::uint8_t> EncodeQ8_0(const std::vector<float>& values) {
  std::vector<std::uint8_t> out(values.size() / 32 * 34);
  for (std::size_t b = 0; b < values.size() / 32; ++b) {
    float amax = 0.0F;
    for (std::size_t j = 0; j < 32; ++j) {
      amax = std::max(amax, std::abs(values[b * 32 + j]));
    }
    const float d = amax / 127.0F;
    const float id = d != 0.0F ? 1.0F / d : 0.0F;
    const std::uint16_t half = gufo::test::rotation::FloatToHalfBits(d);
    std::memcpy(out.data() + b * 34, &half, 2);
    for (std::size_t j = 0; j < 32; ++j) {
      out[b * 34 + 2 + j] = static_cast<std::uint8_t>(
          static_cast<std::int8_t>(std::round(values[b * 32 + j] * id)));
    }
  }
  return out;
}

std::vector<float> Decode(GgmlType type, const void* data, std::size_t n) {
  std::vector<float> out(n);
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  if (type == GgmlType::kF32) {
    std::memcpy(out.data(), bytes, n * 4);
  } else if (type == GgmlType::kBF16) {
    for (std::size_t i = 0; i < n; ++i) {
      std::uint16_t half = 0;
      std::memcpy(&half, bytes + i * 2, 2);
      out[i] = std::bit_cast<float>(static_cast<std::uint32_t>(half) << 16);
    }
  } else {
    gufo::quant::DequantizeQ8_0(bytes, out.data(), n);
  }
  return out;
}

std::vector<std::int32_t> Signs(std::size_t k, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<std::int32_t> signs(k);
  for (auto& sign : signs) {
    sign = (rng() & 1U) != 0 ? -1 : 1;
  }
  return signs;
}

std::vector<std::uint32_t> Pack(const std::vector<std::int32_t>& signs) {
  std::vector<std::uint32_t> bits(signs.size() / 32, 0U);
  for (std::size_t i = 0; i < signs.size(); ++i) {
    if (signs[i] < 0) {
      bits[i / 32] |= 1U << (i % 32);
    }
  }
  return bits;
}

struct Model {
  GgmlType alpha_type = GgmlType::kF32;
  bool with_output = true;
  bool rotate = true;
  std::vector<std::string> folded;  // prism.hadamard.weight_names
  std::map<std::uint64_t, std::vector<std::int32_t>> signs = {
      {kHidden, Signs(kHidden, 1)},
      {kInner, Signs(kInner, 2)},
      {kFfn, Signs(kFfn, 3)}};
  std::vector<std::pair<std::string, std::vector<std::uint64_t>>> projections;
  std::map<std::string, std::vector<float>> alpha_beta;  // original values

  Model() {
    for (int layer = 0; layer < 4; ++layer) {
      const std::string p = "blk." + std::to_string(layer) + ".";
      if (layer == 3) {
        projections.push_back({p + "attn_q.weight", {kHidden, 2 * kAttention}});
        projections.push_back({p + "attn_k.weight", {kHidden, 128}});
        projections.push_back({p + "attn_v.weight", {kHidden, 128}});
        projections.push_back(
            {p + "attn_output.weight", {kAttention, kHidden}});
      } else {
        projections.push_back({p + "attn_qkv.weight", {kHidden, kQkv}});
        projections.push_back({p + "attn_gate.weight", {kHidden, kInner}});
        projections.push_back({p + "ssm_out.weight", {kInner, kHidden}});
        alpha_beta[p + "ssm_alpha.weight"] = Ramp(kHidden, 10 + layer);
        alpha_beta[p + "ssm_beta.weight"] = Ramp(kHidden, 20 + layer);
      }
      projections.push_back({p + "ffn_gate.weight", {kHidden, kFfn}});
      projections.push_back({p + "ffn_up.weight", {kHidden, kFfn}});
      projections.push_back({p + "ffn_down.weight", {kFfn, kHidden}});
    }
    for (const auto& [name, dims] : projections) {
      folded.push_back(name);
    }
    folded.push_back("output.weight");
  }

  [[nodiscard]] std::vector<std::uint8_t> Build() const {
    GgufImage image;
    image.AddString("general.architecture", "qwen35");
    image.AddUint32("qwen35.block_count", 5);
    image.AddUint32("qwen35.nextn_predict_layers", 1);
    image.AddUint32("qwen35.embedding_length", kHidden);
    image.AddUint32("qwen35.feed_forward_length", kFfn);
    image.AddUint32("qwen35.attention.head_count", 2);
    image.AddUint32("qwen35.attention.head_count_kv", 1);
    image.AddUint32("qwen35.attention.key_length", 128);
    image.AddUint32("qwen35.context_length", 4096);
    image.AddUint32("qwen35.full_attention_interval", 4);
    image.AddUint32("qwen35.ssm.conv_kernel", 4);
    image.AddUint32("qwen35.ssm.state_size", 128);
    image.AddUint32("qwen35.ssm.group_count", 1);
    image.AddUint32("qwen35.ssm.time_step_rank", 1);
    image.AddUint32("qwen35.ssm.inner_size", kInner);
    image.AddTensorF32("token_embd.weight", {kHidden, kVocab});
    image.AddTensorF32("output_norm.weight", {kHidden});
    if (with_output) {
      image.AddTensorF32("output.weight", {kHidden, kVocab});
    }
    for (int layer = 0; layer < 4; ++layer) {
      const std::string p = "blk." + std::to_string(layer) + ".";
      image.AddTensorF32(p + "attn_norm.weight", {kHidden});
      image.AddTensorF32(p + "post_attention_norm.weight", {kHidden});
      if (layer == 3) {
        image.AddTensorF32(p + "attn_q_norm.weight", {128});
        image.AddTensorF32(p + "attn_k_norm.weight", {128});
      } else {
        image.AddTensorF32(p + "ssm_a", {1});
        image.AddTensorF32(p + "ssm_conv1d.weight", {4, kQkv});
        image.AddTensorF32(p + "ssm_dt.bias", {1});
        image.AddTensorF32(p + "ssm_norm.weight", {128});
      }
    }
    for (const auto& [name, dims] : projections) {
      image.AddTensorF32(name, dims);
    }
    for (const auto& [name, values] : alpha_beta) {
      if (alpha_type == GgmlType::kF32) {
        image.AddTensorF32Values(name, {kHidden, 1}, values);
      } else if (alpha_type == GgmlType::kBF16) {
        image.AddTensorRaw(name, {kHidden, 1}, GgmlType::kBF16,
                           EncodeBf16(values));
      } else {
        image.AddTensorRaw(name, {kHidden, 1}, GgmlType::kQ8_0,
                           EncodeQ8_0(values));
      }
    }
    // Embedded MTP layer, outside the main stack.
    image.AddTensorF32("blk.4.nextn.eh_proj.weight", {2 * kHidden, kHidden});
    image.AddTensorF32("blk.4.ffn_gate.weight", {kHidden, kFfn});
    if (rotate) {
      image.AddUint32("prism.hadamard.version", 1);
      image.AddUint32("prism.hadamard.block_size", 128);
      image.AddString("prism.hadamard.transform",
                      "normalized-sylvester-walsh-hadamard");
      image.AddString("prism.hadamard.axis", "input-last-dimension");
      image.AddString("prism.hadamard.sign_mode", "explicit");
      std::vector<std::int32_t> widths;
      std::vector<std::int32_t> values;
      for (const auto& [width, vector] : signs) {
        widths.push_back(static_cast<std::int32_t>(width));
        values.insert(values.end(), vector.begin(), vector.end());
      }
      image.AddInt32Array("prism.hadamard.sign_widths", widths);
      image.AddInt32Array("prism.hadamard.sign_values", values);
      image.AddStringArray("prism.hadamard.weight_names", folded);
    }
    return image.Build();
  }
};

struct Loaded {
  std::unique_ptr<gufo::core::GgufReader> reader;
  std::optional<QwenModelWeights> weights;
  std::string error;
};

Loaded Load(const std::vector<std::uint8_t>& bytes) {
  Loaded loaded;
  loaded.reader = gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(),
                                                     &loaded.error);
  if (loaded.reader != nullptr) {
    loaded.weights =
        QwenModelWeights::LoadFromGguf(*loaded.reader, &loaded.error);
  }
  return loaded;
}

bool SameBits(const std::uint32_t* bits,
              const std::vector<std::int32_t>& signs) {
  if (bits == nullptr) {
    return false;
  }
  const auto expected = Pack(signs);
  return std::memcmp(bits, expected.data(), expected.size() * 4) == 0;
}

void TestUnrotated() {
  Model model;
  model.rotate = false;
  const auto bytes = model.Build();
  const auto loaded = Load(bytes);
  Expect(loaded.weights.has_value(), "unrotated model loads: " + loaded.error);
  if (!loaded.weights) {
    return;
  }
  Expect(!loaded.weights->IsRotated(), "unrotated model has no rotation");
  Expect(loaded.weights->folded_weights == nullptr, "nothing folded");
  for (const auto& layer : loaded.weights->layers) {
    Expect(layer.rotation.attn_in == nullptr &&
               layer.rotation.ssm_out_in == nullptr &&
               layer.rotation.attn_out_in == nullptr &&
               layer.rotation.ffn_in == nullptr &&
               layer.rotation.ffn_down_in == nullptr,
           "no layer rotation");
  }
}

void TestBinding(GgmlType alpha_type, std::string_view label) {
  Model model;
  model.alpha_type = alpha_type;
  const auto bytes = model.Build();
  const auto loaded = Load(bytes);
  Expect(loaded.weights.has_value(),
         std::string(label) + ": rotated model loads: " + loaded.error);
  if (!loaded.weights) {
    return;
  }
  const auto& weights = *loaded.weights;
  Expect(weights.layers.size() == 4, "nextn layer is not a main layer");
  Expect(weights.rotated_groups ==
             "attn_in, ssm_out_in, ffn_in, ffn_down_in, attn_out_in, head_in",
         "rotated groups: " + weights.rotated_groups);
  // One vector per width, shared by every group and layer of that width.
  const auto* hidden = weights.head_in_rotation;
  Expect(SameBits(hidden, model.signs.at(kHidden)), "hidden-width signs");
  for (int i = 0; i < 4; ++i) {
    const auto& r = weights.layers[i].rotation;
    Expect(r.attn_in == hidden && r.ffn_in == hidden,
           "attn_in and ffn_in share the hidden-width vector");
    Expect(SameBits(r.ffn_down_in, model.signs.at(kFfn)),
           "ffn_down_in uses the intermediate-width vector");
    if (i == 3) {
      // Attention 256 equals hidden 256 in this toy model.
      Expect(r.attn_out_in == hidden, "attn_out_in uses its width's vector");
      Expect(r.ssm_out_in == nullptr, "no ssm_out_in on attention layer");
    } else {
      Expect(SameBits(r.ssm_out_in, model.signs.at(kInner)),
             "ssm_out_in uses the inner-width vector");
      Expect(r.attn_out_in == nullptr, "no attn_out_in on SSM layer");
    }
  }
  // alpha/beta folded into the rotated basis, same type, alpha then beta
  // adjacent per layer.
  Expect(weights.folded_weights != nullptr &&
             weights.folded_bytes == weights.folded_weights->size(),
         std::string(label) + ": folded storage");
  const auto bits = Pack(model.signs.at(kHidden));
  for (int i = 0; i < 3; ++i) {
    const std::string p = "blk." + std::to_string(i) + ".";
    const auto& layer = weights.layers[i];
    Expect(
        layer.ssm_alpha.type == alpha_type && layer.ssm_beta.type == alpha_type,
        std::string(label) + ": folded type unchanged");
    Expect(static_cast<const std::uint8_t*>(layer.ssm_alpha.data) +
                   layer.ssm_alpha.EncodedSizeBytes() ==
               layer.ssm_beta.data,
           std::string(label) + ": alpha and beta adjacent");
    for (const auto* name : {"ssm_alpha.weight", "ssm_beta.weight"}) {
      const auto& tensor = std::string_view(name) == "ssm_alpha.weight"
                               ? layer.ssm_alpha
                               : layer.ssm_beta;
      // Expected: decode the stored original, rotate, re-encode, decode.
      std::vector<float> row;
      std::vector<std::uint8_t> original;
      if (alpha_type == GgmlType::kF32) {
        row = model.alpha_beta.at(p + name);
      } else {
        original = alpha_type == GgmlType::kBF16
                       ? EncodeBf16(model.alpha_beta.at(p + name))
                       : EncodeQ8_0(model.alpha_beta.at(p + name));
        row = Decode(alpha_type, original.data(), kHidden);
      }
      gufo::test::rotation::RotateRow(row, bits);
      std::vector<std::uint8_t> expected;
      if (alpha_type == GgmlType::kF32) {
        expected.resize(kHidden * 4);
        std::memcpy(expected.data(), row.data(), expected.size());
      } else {
        expected =
            alpha_type == GgmlType::kBF16 ? EncodeBf16(row) : EncodeQ8_0(row);
      }
      Expect(
          std::memcmp(tensor.data, expected.data(), expected.size()) == 0,
          std::string(label) + ": " + p + name + " equals the reference fold");
    }
  }
}

void ExpectRefused(const Model& model, std::string_view fragment,
                   std::string_view label) {
  const auto bytes = model.Build();
  const auto loaded = Load(bytes);
  Expect(!loaded.weights.has_value(), std::string(label) + ": accepted");
  Expect(loaded.error.find(fragment) != std::string::npos,
         std::string(label) + ": '" + loaded.error + "' lacks '" +
             std::string(fragment) + "'");
}

void TestRefusals() {
  {
    Model model;
    model.folded.push_back("blk.4.ffn_gate.weight");
    ExpectRefused(model,
                  "blk.4.ffn_gate.weight is rotated but its role has no "
                  "rotation path",
                  "nextn layer rotated");
  }
  {
    Model model;
    std::erase(model.folded, std::string("blk.2.ffn_down.weight"));
    ExpectRefused(model,
                  "group ffn_down_in is rotated in some layers only (3 of 4; "
                  "ffn_down_in:2",
                  "partial ffn_down_in");
  }
  {
    Model model;
    std::erase(model.folded, std::string("blk.0.attn_gate.weight"));
    ExpectRefused(model, "group instance attn_in:0 mixes rotated",
                  "mixed attn_in");
  }
  {
    Model model;
    model.folded.push_back("blk.0.ssm_alpha.weight");
    ExpectRefused(model, "src/llama-model.cpp:2199",
                  "ssm_alpha folded in the file");
  }
  {
    Model model;
    model.with_output = false;
    ExpectRefused(model, "src/llama-model.cpp:2271", "missing output.weight");
  }
}

void TestCpuMtpAndV1Refusals() {
  Model model;
  const auto bytes = model.Build();
  std::string error;
  auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
  Expect(reader != nullptr, "reader: " + error);
  if (reader == nullptr) {
    return;
  }
  error.clear();
  auto generator = gufo::models::QwenGenerator::CreateFromGguf(*reader, &error);
  Expect(generator == nullptr, "CPU generator refuses a rotated model");
  Expect(error.find("the Qwen CPU backend (--cpu) does not support rotated "
                    "input groups (attn_in,") != std::string::npos,
         "CPU refusal message: " + error);

  error.clear();
  const auto mtp =
      gufo::speculative::QwenMtpWeights::LoadFromGguf(*reader, &error);
  Expect(!mtp.has_value(), "MTP refuses a rotated head");
  Expect(error == gufo::speculative::kMtpRotatedHeadError,
         "MTP refusal message: " + error);

  // v1 files are refused by name.
  {
    GgufImage image;
    image.AddString("general.architecture", "qwen35");
    image.AddUint32("gufo.rotation.version", 1);
    image.AddTensorF32("token_embd.weight", {kHidden, kVocab});
    const auto old = image.Build();
    auto old_reader =
        gufo::core::GgufReader::OpenMemory(old.data(), old.size(), &error);
    error.clear();
    gufo::quant::rotation::GroupTable table;
    Expect(!gufo::quant::rotation::ReadRotation(*old_reader, table, &error) &&
               error.find("v1 rotation format") != std::string::npos,
           "v1 refused: " + error);
  }
}

// Fold error of alpha/beta-sized matrices (48 x 5120, Gaussian weights with
// a few large columns) for R2: relative error of W'·(H·D·x) against W·x,
// where W' is the fold re-encoded in the original type.
void MeasureFoldError() {
  constexpr std::size_t kRows = 48;
  constexpr std::size_t kK = 5120;
  std::mt19937 rng(20261029);
  std::normal_distribution<float> normal(0.0F, 0.02F);
  std::vector<float> w(kRows * kK);
  for (auto& value : w) {
    value = normal(rng);
  }
  for (std::size_t r = 0; r < kRows; ++r) {
    w[r * kK + 17] *= 20.0F;
  }
  const auto bits = Pack(Signs(kK, 5));
  std::vector<float> x(kK);
  std::normal_distribution<float> unit(0.0F, 1.0F);
  for (auto& value : x) {
    value = unit(rng);
  }
  x[17] = 80.0F;  // a massive activation component
  auto x_rot = x;
  gufo::test::rotation::RotateRow(x_rot, bits);
  for (const auto type : {GgmlType::kBF16, GgmlType::kQ8_0}) {
    double err = 0;
    double ref = 0;
    double base_err = 0;
    for (std::size_t r = 0; r < kRows; ++r) {
      std::vector<float> row(w.begin() + r * kK, w.begin() + (r + 1) * kK);
      const auto stored =
          type == GgmlType::kBF16 ? EncodeBf16(row) : EncodeQ8_0(row);
      auto original = Decode(type, stored.data(), kK);
      auto folded = original;
      gufo::test::rotation::RotateRow(folded, bits);
      const auto refolded =
          type == GgmlType::kBF16 ? EncodeBf16(folded) : EncodeQ8_0(folded);
      const auto runtime = Decode(type, refolded.data(), kK);
      double exact = 0;
      double stored_dot = 0;
      double rotated_dot = 0;
      for (std::size_t i = 0; i < kK; ++i) {
        exact += static_cast<double>(row[i]) * x[i];
        stored_dot += static_cast<double>(original[i]) * x[i];
        rotated_dot += static_cast<double>(runtime[i]) * x_rot[i];
      }
      err += (rotated_dot - stored_dot) * (rotated_dot - stored_dot);
      base_err += (stored_dot - exact) * (stored_dot - exact);
      ref += stored_dot * stored_dot;
    }
    std::cout << "fold error " << (type == GgmlType::kBF16 ? "BF16" : "Q8_0")
              << ": relL2(folded vs stored) = " << std::sqrt(err / ref)
              << ", relL2(stored vs FP32 original) = "
              << std::sqrt(base_err / ref) << "\n";
  }
}

}  // namespace

int main() {
  TestUnrotated();
  TestBinding(GgmlType::kF32, "F32");
  TestBinding(GgmlType::kBF16, "BF16");
  TestBinding(GgmlType::kQ8_0, "Q8_0");
  TestRefusals();
  TestCpuMtpAndV1Refusals();
  MeasureFoldError();
  if (failures != 0) {
    std::cerr << failures << " rotation binding check(s) failed\n";
    return 1;
  }
  std::cout << "qwen27b_rotation_binding_test: all checks passed\n";
  return 0;
}
