// Qwen 27B rotated-GGUF binding on a tiny synthetic qwen35 model: the group
// table handed to src/core/quant/rotation, the per-layer sign pointers, the
// embedded MTP (nextn) layer, a tied head, and the CPU and MTP refusals.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/generator.hpp"
#include "src/models/qwen/mtp_reference.hpp"
#include "src/models/qwen/state.hpp"
#include "tests/core/quant/rotation_reference.hpp"
#include "tests/core/quant/rotation_test_gguf.hpp"

namespace {

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
// intermediate 384, attention 2 x 128, SSM inner 128 (one 128-wide head).
constexpr std::uint64_t kHidden = 256;
constexpr std::uint64_t kFfn = 384;
constexpr std::uint64_t kAttention = 256;
constexpr std::uint64_t kInner = 128;
constexpr std::uint64_t kQkv = 384;
constexpr std::uint64_t kVocab = 16;

struct Model {
  GgufImage image;
  // tensor name -> K (ne0) for every projection, so tests can rotate them.
  std::map<std::string, std::uint64_t> projections;
  std::map<std::string, std::vector<float>> signs;
  bool with_output = true;
  bool markers = true;

  void Projection(const std::string& name, std::uint64_t k, std::uint64_t n) {
    image.AddTensorF32(name, {k, n});
    projections[name] = k;
  }

  Model() {
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
    for (int layer = 0; layer < 4; ++layer) {
      const std::string p = "blk." + std::to_string(layer) + ".";
      image.AddTensorF32(p + "attn_norm.weight", {kHidden});
      image.AddTensorF32(p + "post_attention_norm.weight", {kHidden});
      if (layer == 3) {
        Projection(p + "attn_q.weight", kHidden, 2 * kAttention);
        Projection(p + "attn_k.weight", kHidden, 128);
        Projection(p + "attn_v.weight", kHidden, 128);
        Projection(p + "attn_output.weight", kAttention, kHidden);
        image.AddTensorF32(p + "attn_q_norm.weight", {128});
        image.AddTensorF32(p + "attn_k_norm.weight", {128});
      } else {
        Projection(p + "attn_qkv.weight", kHidden, kQkv);
        Projection(p + "attn_gate.weight", kHidden, kInner);
        Projection(p + "ssm_alpha.weight", kHidden, 1);
        Projection(p + "ssm_beta.weight", kHidden, 1);
        Projection(p + "ssm_out.weight", kInner, kHidden);
        image.AddTensorF32(p + "ssm_a", {1});
        image.AddTensorF32(p + "ssm_conv1d.weight", {4, kQkv});
        image.AddTensorF32(p + "ssm_dt.bias", {1});
        image.AddTensorF32(p + "ssm_norm.weight", {128});
      }
      Projection(p + "ffn_gate.weight", kHidden, kFfn);
      Projection(p + "ffn_up.weight", kHidden, kFfn);
      Projection(p + "ffn_down.weight", kFfn, kHidden);
    }
    // Embedded MTP layer, outside the main stack.
    image.AddTensorF32("blk.4.nextn.eh_proj.weight", {2 * kHidden, kHidden});
    Projection("blk.4.ffn_gate.weight", kHidden, kFfn);
  }

  [[nodiscard]] static std::string Instance(const std::string& name) {
    static const std::vector<std::pair<std::string_view, std::string_view>>
        groups = {{"attn_qkv", "attn_in"},        {"attn_gate", "attn_in"},
                  {"ssm_alpha", "attn_in"},       {"ssm_beta", "attn_in"},
                  {"attn_q.", "attn_in"},         {"attn_k.", "attn_in"},
                  {"attn_v.", "attn_in"},         {"ssm_out", "ssm_out_in"},
                  {"attn_output", "attn_out_in"}, {"ffn_gate", "ffn_in"},
                  {"ffn_up", "ffn_in"},           {"ffn_down", "ffn_down_in"}};
    if (name == "output.weight") {
      return "head_in";
    }
    const std::string layer = name.substr(4, name.find('.', 4) - 4);
    for (const auto& [part, group] : groups) {
      if (name.find(part) != std::string::npos) {
        return std::string(group) + ":" + layer;
      }
    }
    return name;
  }

  // Rotates every main-stack projection and the head; one sign vector per
  // group instance, as the converter writes them.
  void RotateAll() {
    std::map<std::string, std::vector<float>> by_instance;
    std::mt19937 rng(11);
    auto all = projections;
    if (with_output) {
      all["output.weight"] = kHidden;
    }
    for (const auto& [name, k] : all) {
      if (name.starts_with("blk.4.")) {
        continue;
      }
      auto& sign = by_instance[Instance(name)];
      if (sign.empty()) {
        sign.resize(k);
        for (auto& value : sign) {
          value = (rng() & 1U) != 0 ? -1.0F : 1.0F;
        }
      }
      signs[name] = sign;
    }
  }

  [[nodiscard]] std::vector<std::uint8_t> Build() const {
    GgufImage out = image;
    if (with_output) {
      out.AddTensorF32("output.weight", {kHidden, kVocab});
    }
    if (markers) {
      out.AddUint32("gufo.rotation.version", 1);
      out.AddString("gufo.rotation.kind", "hadamard");
      out.AddUint32("gufo.rotation.block_size", 128);
      out.AddString("gufo.rotation.side", "input");
    }
    for (const auto& [name, values] : signs) {
      out.AddTensorF16(name + ".rot", values);
    }
    return out.Build();
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

bool SameSigns(const std::uint32_t* bits, const std::vector<float>& signs) {
  if (bits == nullptr) {
    return false;
  }
  for (std::size_t i = 0; i < signs.size(); ++i) {
    if (((bits[i / 32] >> (i % 32)) & 1U) != (signs[i] < 0.0F ? 1U : 0U)) {
      return false;
    }
  }
  return true;
}

void TestUnrotated() {
  Model model;
  model.markers = false;
  // The fixture must stay large: the bytes back the reader.
  const auto bytes = model.Build();
  const auto loaded = Load(bytes);
  Expect(loaded.weights.has_value(), "unrotated model loads: " + loaded.error);
  if (!loaded.weights) {
    return;
  }
  Expect(!loaded.weights->IsRotated(), "unrotated model has no rotation");
  Expect(loaded.weights->head_in_rotation == nullptr, "no head rotation");
  for (const auto& layer : loaded.weights->layers) {
    Expect(layer.rotation.attn_in == nullptr &&
               layer.rotation.ssm_out_in == nullptr &&
               layer.rotation.attn_out_in == nullptr &&
               layer.rotation.ffn_in == nullptr &&
               layer.rotation.ffn_down_in == nullptr,
           "no layer rotation");
  }
}

void TestBinding() {
  Model model;
  model.RotateAll();
  const auto bytes = model.Build();
  const auto loaded = Load(bytes);
  Expect(loaded.weights.has_value(), "rotated model loads: " + loaded.error);
  if (!loaded.weights) {
    return;
  }
  const auto& weights = *loaded.weights;
  Expect(weights.layers.size() == 4, "nextn layer is not a main layer");
  Expect(weights.rotated_groups ==
             "attn_in, ssm_out_in, ffn_in, ffn_down_in, attn_out_in, head_in",
         "rotated groups: " + weights.rotated_groups);
  Expect(SameSigns(weights.head_in_rotation, model.signs.at("output.weight")),
         "head_in signs");
  for (int i = 0; i < 4; ++i) {
    const std::string p = "blk." + std::to_string(i) + ".";
    const auto& r = weights.layers[i].rotation;
    const auto& in =
        model.signs.at(p + (i == 3 ? "attn_q.weight" : "attn_qkv.weight"));
    Expect(SameSigns(r.attn_in, in), "attn_in signs, layer " + p);
    Expect(SameSigns(r.ffn_in, model.signs.at(p + "ffn_up.weight")),
           "ffn_in signs");
    Expect(SameSigns(r.ffn_down_in, model.signs.at(p + "ffn_down.weight")),
           "ffn_down_in signs");
    if (i == 3) {
      Expect(SameSigns(r.attn_out_in, model.signs.at(p + "attn_output.weight")),
             "attn_out_in signs");
      Expect(r.ssm_out_in == nullptr, "no ssm_out_in on attention layer");
    } else {
      Expect(SameSigns(r.ssm_out_in, model.signs.at(p + "ssm_out.weight")),
             "ssm_out_in signs");
      Expect(r.attn_out_in == nullptr, "no attn_out_in on SSM layer");
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
    model.RotateAll();
    model.signs["blk.4.ffn_gate.weight"] = std::vector<float>(kHidden, 1.0F);
    ExpectRefused(model,
                  "blk.4.ffn_gate.weight is rotated but its role has no "
                  "rotation path",
                  "nextn layer rotated");
  }
  {
    Model model;
    model.RotateAll();
    model.signs.erase("blk.2.ffn_down.weight");
    ExpectRefused(model,
                  "group ffn_down_in is rotated in some layers only (3 of 4; "
                  "ffn_down_in:2",
                  "partial ffn_down_in");
  }
  {
    Model model;
    model.RotateAll();
    model.signs.erase("blk.0.ssm_beta.weight");
    ExpectRefused(model, "group instance attn_in:0 mixes rotated",
                  "mixed attn_in");
  }
  {
    Model model;
    model.with_output = false;
    model.RotateAll();
    model.signs["token_embd.weight"] = std::vector<float>(kHidden, 1.0F);
    ExpectRefused(model, "token_embd.weight is rotated, but it is tied",
                  "tied head");
  }
}

void TestCpuAndMtpRefusals() {
  Model model;
  model.RotateAll();
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

  // Without a rotated head, the CPU backend still refuses the body rotation.
  Model body;
  body.RotateAll();
  body.signs.erase("output.weight");
  const auto body_bytes = body.Build();
  auto body_reader = gufo::core::GgufReader::OpenMemory(
      body_bytes.data(), body_bytes.size(), &error);
  error.clear();
  Expect(gufo::models::QwenGenerator::CreateFromGguf(*body_reader, &error) ==
                 nullptr &&
             error.find("--cpu") != std::string::npos,
         "CPU refuses body-only rotation: " + error);
}

}  // namespace

int main() {
  TestUnrotated();
  TestBinding();
  TestRefusals();
  TestCpuAndMtpRefusals();
  if (failures != 0) {
    std::cerr << failures << " rotation binding check(s) failed\n";
    return 1;
  }
  std::cout << "qwen27b_rotation_binding_test: all checks passed\n";
  return 0;
}
