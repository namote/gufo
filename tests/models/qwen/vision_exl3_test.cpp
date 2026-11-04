#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "src/core/json.hpp"
#include "src/models/qwen/vision/exl3_weights.hpp"

namespace {
using gufo::json::Value;
using gufo::models::qwen::vision::Exl3Weights;
namespace exl3 = gufo::quant::exl3;

struct Fixture {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("gufo-vision-exl3-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  Value header = Value::object();
  std::uint64_t payload_size = 0;

  Fixture(bool quantized, bool partial = false, bool mixed_ffn = false) {
    std::filesystem::create_directory(path);
    Config();
    std::ofstream(path / "preprocessor_config.json") << R"({
      "patch_size":16,"temporal_patch_size":2,"merge_size":2,
      "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],
      "size":{"shortest_edge":65536,"longest_edge":16777216}})";
    const std::string root = "model.visual.";
    Tensor(root + "patch_embed.proj.weight", {1152, 3, 2, 16, 16});
    Tensor(root + "patch_embed.proj.bias", {1152});
    Tensor(root + "pos_embed.weight", {2304, 1152}, "BF16");
    for (unsigned i = 0; i < 27; ++i) {
      const auto p = root + "blocks." + std::to_string(i) + ".";
      for (auto norm : {"norm1", "norm2"}) {
        Tensor(p + norm + ".weight", {1152});
        Tensor(p + norm + ".bias", {1152});
      }
      // Retained original fused weights coexist with EXL3 split Q/K/V.
      Matrix(p + "attn.qkv", 1152, 3456, false);
      if (quantized) {
        Matrix(p + "attn.q_proj", 1152, 1152, true);
        if (!partial || i != 0) {
          Matrix(p + "attn.k_proj", 1152, 1152, true);
          Matrix(p + "attn.v_proj", 1152, 1152, true);
        }
      }
      Matrix(p + "attn.proj", 1152, 1152, quantized);
      Matrix(p + "mlp.linear_fc1", 1152, 4304,
             quantized && !(mixed_ffn && i == 0));
      Matrix(p + "mlp.linear_fc2", 4304, 1152,
             quantized && !(mixed_ffn && i == 1));
    }
    Tensor(root + "merger.norm.weight", {1152});
    Tensor(root + "merger.norm.bias", {1152});
    Matrix(root + "merger.linear_fc1", 4608, 4608, quantized);
    Matrix(root + "merger.linear_fc2", 4608, 2560, quantized);
    Tensor("model.language_model.unused", {1});
    Save();
  }
  ~Fixture() { std::filesystem::remove_all(path); }
  void Config(unsigned width = 2560, bool deepstack = false) {
    std::ofstream(path / "config.json")
        << R"({"quantization_config":{"quant_method":"exl3"},
      "vision_config":{"depth":27,"hidden_size":1152,
      "hidden_act":"gelu_pytorch_tanh","intermediate_size":4304,
      "num_heads":16,"num_position_embeddings":2304,
      "patch_size":16,"temporal_patch_size":2,"spatial_merge_size":2,
      "out_hidden_size":)"
        << width << R"(,"deepstack_visual_indexes":)"
        << (deepstack ? "[8]" : "[]") << "}}";
  }
  void Tensor(const std::string& name, std::initializer_list<unsigned> shape,
              const char* dtype = "F16") {
    auto entry = Value::object();
    entry["dtype"] = dtype;
    auto dimensions = Value::array();
    std::uint64_t count = 1;
    for (auto d : shape) {
      dimensions.push_back(static_cast<int>(d));
      count *= d;
    }
    entry["shape"] = std::move(dimensions);
    const std::uint64_t bytes = count * (std::string(dtype) == "I32" ? 4 : 2);
    auto offsets = Value::array();
    offsets.push_back(payload_size);
    offsets.push_back(payload_size + bytes);
    entry["data_offsets"] = std::move(offsets);
    header[name] = std::move(entry);
    payload_size += bytes;
  }
  void Matrix(const std::string& key, unsigned in, unsigned out, bool quant) {
    if (quant) {
      in = (in + 127) / 128 * 128;
      out = (out + 127) / 128 * 128;
      // K8 for the merger, K6 elsewhere: readers must not hard-code K6.
      const unsigned words = key.find("merger") == std::string::npos ? 96 : 128;
      Tensor(key + ".trellis", {in / 16, out / 16, words}, "I16");
      Tensor(key + ".suh", {in});
      Tensor(key + ".svh", {out});
      Tensor(key + ".mul1", {}, "I32");
    } else {
      Tensor(key + ".weight", {out, in}, "BF16");
    }
    Tensor(key + ".bias", {out});
  }
  void Save() const {
    std::string text = header.dump();
    text.resize((text.size() + 7) / 8 * 8, ' ');
    const std::uint64_t size = text.size();
    std::ofstream out(path / "model.safetensors", std::ios::binary);
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(text.data(), text.size());
    // Sparse fixture: hundreds of tensor shapes, no model-sized disk writes.
    out.seekp(static_cast<std::streamoff>(8 + size + payload_size - 1));
    out.put(0);
  }
  void ChangeByte(const std::string& name, unsigned char byte) const {
    const auto* tensor = header.find(name);
    const auto offset = tensor->find("data_offsets")->items()[0].as_size();
    const auto size = (header.dump().size() + 7) / 8 * 8;
    std::fstream out(path / "model.safetensors",
                     std::ios::binary | std::ios::in | std::ios::out);
    out.seekp(static_cast<std::streamoff>(8 + size + offset));
    out.put(static_cast<char>(byte));
  }
};

template<class F>
void Rejects(F&& f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::exception&) {
    rejected = true;
  }
  assert(rejected);
}

void Conversion() {
  const std::array<std::uint16_t, 3> fp16{0x3c00, 0xc000, 0x0001};
  exl3::Tensor tensor;
  tensor.dtype = exl3::DType::kF16;
  tensor.shape = {3};
  tensor.data = reinterpret_cast<const std::uint8_t*>(fp16.data());
  tensor.bytes = sizeof(fp16);
  const auto values = Exl3Weights::FloatValues(tensor);
  assert(values[0] == 1 && values[1] == -2 && values[2] == 0x1p-24F);
  const std::array<std::uint16_t, 3> bf16{0x3f80, 0xc000, 0x0001};
  tensor.dtype = exl3::DType::kBF16;
  tensor.data = reinterpret_cast<const std::uint8_t*>(bf16.data());
  const auto bvalues = Exl3Weights::FloatValues(tensor);
  assert(bvalues[0] == 1 && bvalues[1] == -2 && bvalues[2] == 0x1p-133F);
  const std::array<float, 3> fp32{1, -2, 0.5F};
  tensor.dtype = exl3::DType::kF32;
  tensor.data = reinterpret_cast<const std::uint8_t*>(fp32.data());
  tensor.bytes = sizeof(fp32);
  assert(Exl3Weights::FloatValues(tensor) ==
         std::vector<float>(fp32.begin(), fp32.end()));
  tensor.bytes -= 1;
  Rejects([&] { (void)Exl3Weights::FloatValues(tensor); });
  const float infinity = std::numeric_limits<float>::infinity();
  tensor.data = reinterpret_cast<const std::uint8_t*>(&infinity);
  tensor.bytes = sizeof(float);
  tensor.shape = {1};
  Rejects([&] { (void)Exl3Weights::FloatValues(tensor); });
}

void NativePack() {
  Fixture fixture(true);
  assert(Exl3Weights::HasVision(fixture.path));
  Exl3Weights weights(fixture.path, 2560);
  assert(weights.matrices().contains("v.blk.0.attn_q.weight"));
  assert(!weights.matrices().contains("v.blk.0.attn_qkv.weight"));
  const auto& up = weights.matrices().at("v.blk.0.ffn_up.weight");
  assert(up.in_features == 1152 && up.out_features == 4352);
  assert(weights.tensors().at("v.blk.0.ffn_up.bias")->shape[0] == 4352);
  assert(weights.matrices().at("v.blk.0.ffn_down.weight").in_features == 4352);
  assert(weights.matrices().at("mm.0.weight").quantized->rate.bits == 8);
  assert(up.quantized->rate.bits == 6);
  assert(weights.matrices().at("patch").raw->shape ==
         std::vector<std::int64_t>({1152, 3, 2, 16, 16}));
  const auto identity = weights.identity();
  // Text payload and unused original fused QKV are outside native vision's
  // identity. A selected projection bias must invalidate it.
  fixture.ChangeByte("model.language_model.unused", 1);
  fixture.ChangeByte("model.visual.blocks.0.attn.qkv.weight", 1);
  assert(Exl3Weights(fixture.path, 2560).identity() == identity);
  fixture.ChangeByte("model.visual.blocks.0.attn.q_proj.bias", 1);
  assert(Exl3Weights(fixture.path, 2560).identity() != identity);
  Rejects([&] { Exl3Weights mismatch(fixture.path, 5120); });
  fixture.Config(2560, true);
  Rejects([&] { Exl3Weights deepstack(fixture.path, 2560); });
  fixture.Config();
  std::ofstream(fixture.path / "preprocessor_config.json") << "{}";
  Rejects([&] { Exl3Weights preprocessing(fixture.path, 2560); });
  std::ofstream(fixture.path / "config.json") << "{}";
  assert(!Exl3Weights::HasVision(fixture.path));
}

void RawAndCorruptPacks() {
  {
    Fixture fixture(false);
    Exl3Weights weights(fixture.path, 2560);
    assert(weights.matrices().at("v.blk.0.attn_qkv.weight").raw != nullptr);
    assert(weights.matrices().at("v.blk.0.ffn_up.weight").out_features == 4304);
    assert(!weights.matrices().at("mm.0.weight").quantized);
    // A supported architecture declaration does not hide missing tensors.
    std::filesystem::resize_file(fixture.path / "model.safetensors", 16);
    Rejects([&] { Exl3Weights truncated(fixture.path, 2560); });
  }
  {
    Fixture fixture(true, true);
    Rejects([&] { Exl3Weights partial(fixture.path, 2560); });
  }
  {
    Fixture fixture(true, false, true);
    Exl3Weights weights(fixture.path, 2560);
    // Runtime consumers must pad or crop between these independently stored
    // projections; the loader must not pretend all FFNs have one width.
    assert(weights.matrices().at("v.blk.0.ffn_up.weight").out_features == 4304);
    assert(weights.matrices().at("v.blk.0.ffn_down.weight").in_features ==
           4352);
    assert(weights.matrices().at("v.blk.1.ffn_up.weight").out_features == 4352);
    assert(weights.matrices().at("v.blk.1.ffn_down.weight").in_features ==
           4304);
  }
}
}  // namespace

int main() {
  Conversion();
  NativePack();
  RawAndCorruptPacks();
  std::cout << "EXL3 vision metadata, weights and identity: passed\n";
}
