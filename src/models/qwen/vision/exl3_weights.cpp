#include "src/models/qwen/vision/exl3_weights.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"

namespace gufo::models::qwen::vision {
namespace {
namespace exl3 = quant::exl3;

[[noreturn]] void Fail(const std::string& message) {
  throw std::invalid_argument("EXL3 vision: " + message);
}

json::Value ReadJson(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file)
    Fail("cannot read " + path.string());
  std::ostringstream text;
  text << file.rdbuf();
  return json::parse(text.str());
}

void Number(const json::Value& object, const char* key, double expected,
            bool optional = false) {
  const auto* value = object.find(key);
  if (!value && optional)
    return;
  if (!value || !value->is_number() || value->as_double() != expected)
    Fail(std::string("incompatible metadata: ") + key);
}

void Boolean(const json::Value& object, const char* key, bool expected) {
  const auto* value = object.find(key);
  if (value && (!value->is_bool() || value->as_bool() != expected))
    Fail(std::string("incompatible preprocessing: ") + key);
}

void ValidateMetadata(const std::filesystem::path& path, unsigned width) {
  const auto config = ReadJson(path / "config.json");
  const auto* vision = config.find("vision_config");
  if (!vision || !vision->is_object() || (width != 2560 && width != 5120))
    Fail("missing or incompatible vision_config/output width");
  for (const auto& [name, value] :
       std::initializer_list<std::pair<const char*, double>>{
           {"depth", 27},
           {"hidden_size", 1152},
           {"intermediate_size", 4304},
           {"num_heads", 16},
           {"num_position_embeddings", 2304},
           {"out_hidden_size", width},
           {"patch_size", 16},
           {"spatial_merge_size", 2},
           {"temporal_patch_size", 2}})
    Number(*vision, name, value);
  Number(*vision, "in_channels", 3, true);
  Number(*vision, "num_channels", 3, true);
  Number(*vision, "layer_norm_eps", 1e-6, true);
  Number(*vision, "layernorm_eps", 1e-6, true);
  Number(*vision, "rope_theta", 10000, true);
  const auto* deepstack = vision->find("deepstack_visual_indexes");
  if (!deepstack || !deepstack->is_array() || deepstack->size() != 0)
    Fail("only the final-layer vision merger is supported");
  if (vision->member_str("hidden_act") != "gelu_pytorch_tanh")
    Fail("unsupported vision activation");

  const auto processor = ReadJson(path / "preprocessor_config.json");
  Number(processor, "patch_size", 16);
  Number(processor, "temporal_patch_size", 2);
  Number(processor, "merge_size", 2);
  Number(processor, "rescale_factor", 1.0 / 255.0, true);
  Number(processor, "resample", 3, true);
  for (const auto* key : {"do_resize", "do_rescale", "do_normalize"})
    Boolean(processor, key, true);
  for (const auto* key : {"image_mean", "image_std"}) {
    const auto* array = processor.find(key);
    if (!array || !array->is_array() || array->size() != 3)
      Fail(std::string("invalid normalization: ") + key);
    for (const auto& value : array->items()) {
      if (!value.is_number() || value.as_double() != 0.5)
        Fail(std::string("unsupported normalization: ") + key);
    }
  }
  const auto* size = processor.find("size");
  if (!size || !size->is_object())
    Fail("missing image size limits");
  Number(*size, "shortest_edge", 65536);
  Number(*size, "longest_edge", 16777216);
}
}  // namespace

bool Exl3Weights::HasVision(const std::filesystem::path& directory) {
  return ReadJson(directory / "config.json").find("vision_config") != nullptr;
}

Exl3Weights::Exl3Weights(const std::filesystem::path& directory,
                         std::uint32_t output_width)
    : checkpoint_([&] {
        ValidateMetadata(directory, output_width);
        return exl3::Checkpoint::Open(directory);
      }()),
      output_width_(output_width) {
  const std::string root = "model.visual.";
  const auto* patch =
      Require(root + "patch_embed.proj.weight", {1152, 3, 2, 16, 16});
  matrices_.emplace("patch", Matrix{1536, 1152, {}, patch});
  AddTensor("v.patch_embd.bias", root + "patch_embed.proj.bias", {1152});
  AddTensor("v.position_embd.weight", root + "pos_embed.weight", {2304, 1152});
  for (unsigned layer = 0; layer < 27; ++layer) {
    const auto source = root + "blocks." + std::to_string(layer) + ".";
    const auto dest = "v.blk." + std::to_string(layer) + ".";
    for (unsigned norm = 1; norm <= 2; ++norm) {
      for (const auto* suffix : {".weight", ".bias"})
        AddTensor(dest + "ln" + std::to_string(norm) + suffix,
                  source + "norm" + std::to_string(norm) + suffix, {1152});
    }
    // Native EXL3 loaders prioritize quantized split projections over the
    // original fused BF16 qkv that may also be retained in the checkpoint.
    unsigned split = 0;
    for (const auto* projection : {"q", "k", "v"}) {
      const auto module = source + "attn." + projection + "_proj";
      if (checkpoint_.IsQuantized(module) ||
          checkpoint_.Find(module + ".weight") != nullptr)
        ++split;
    }
    if (split != 0 && split != 3)
      Fail("partial split QKV group: " + source);
    if (split == 3) {
      for (const auto* projection : {"q", "k", "v"})
        AddMatrix(dest + "attn_" + projection,
                  source + "attn." + projection + "_proj", 1152, 1152);
    } else {
      AddMatrix(dest + "attn_qkv", source + "attn.qkv", 1152, 3456);
    }
    AddMatrix(dest + "attn_out", source + "attn.proj", 1152, 1152);
    AddMatrix(dest + "ffn_up", source + "mlp.linear_fc1", 1152, 4304);
    AddMatrix(dest + "ffn_down", source + "mlp.linear_fc2", 4304, 1152);
  }
  for (const auto* suffix : {".weight", ".bias"})
    AddTensor(std::string("v.post_ln") + suffix, root + "merger.norm" + suffix,
              {1152});
  AddMatrix("mm.0", root + "merger.linear_fc1", 4608, 4608);
  AddMatrix("mm.2", root + "merger.linear_fc2", 4608, output_width);
}

const exl3::Tensor* Exl3Weights::Require(const std::string& name,
                                         std::vector<std::int64_t> shape) {
  const auto* tensor = checkpoint_.Find(name);
  if (!tensor || tensor->shape != shape ||
      (tensor->dtype != exl3::DType::kF16 &&
       tensor->dtype != exl3::DType::kBF16 &&
       tensor->dtype != exl3::DType::kF32))
    Fail("missing or invalid floating tensor: " + name);
  selected_[name] = tensor;
  return tensor;
}

void Exl3Weights::AddTensor(const std::string& key, const std::string& name,
                            std::vector<std::int64_t> shape) {
  tensors_[key] = Require(name, std::move(shape));
}

void Exl3Weights::AddMatrix(const std::string& key, const std::string& module,
                            std::size_t in, std::size_t out) {
  Matrix matrix{in, out, {}, nullptr};
  std::size_t bias_width = out;
  if (checkpoint_.IsQuantized(module)) {
    matrix.quantized = checkpoint_.GetLinear(module);
    const auto& linear = *matrix.quantized;
    if (linear.in_features != ((in + 127) / 128) * 128 ||
        linear.out_features != ((out + 127) / 128) * 128)
      Fail("incompatible quantized dimensions: " + module);
    if (!linear.bias)
      Fail("missing projection bias: " + module);
    matrix.in_features = linear.in_features;
    matrix.out_features = linear.out_features;
    bias_width = linear.out_features;
    // Do not upload corrupt scales into an otherwise well-formed trellis.
    for (const auto* suffix : {".suh", ".svh"})
      (void)FloatValues(*checkpoint_.Find(module + suffix));
    for (const auto* suffix :
         {".trellis", ".suh", ".svh", ".bias", ".mul1", ".mcg"}) {
      if (const auto* tensor = checkpoint_.Find(module + suffix))
        selected_[module + suffix] = tensor;
    }
  } else {
    matrix.raw = Require(module + ".weight", {static_cast<std::int64_t>(out),
                                              static_cast<std::int64_t>(in)});
  }
  AddTensor(key + ".bias", module + ".bias",
            {static_cast<std::int64_t>(bias_width)});
  matrices_.emplace(key + ".weight", std::move(matrix));
}

std::vector<float> Exl3Weights::FloatValues(const exl3::Tensor& tensor) {
  if (tensor.dtype != exl3::DType::kF16 && tensor.dtype != exl3::DType::kBF16 &&
      tensor.dtype != exl3::DType::kF32)
    Fail("expected floating point tensor");
  const auto size = static_cast<std::size_t>(tensor.Elements());
  const std::size_t element = tensor.dtype == exl3::DType::kF32 ? 4 : 2;
  if (size == 0 || tensor.bytes / element != size ||
      tensor.bytes % element != 0 || tensor.data == nullptr)
    Fail("invalid floating tensor storage");
  std::vector<float> result(size);
  for (std::size_t i = 0; i < size; ++i) {
    if (tensor.dtype == exl3::DType::kF32) {
      std::memcpy(&result[i], tensor.data + i * 4, 4);
    } else {
      std::uint16_t bits;
      std::memcpy(&bits, tensor.data + i * 2, 2);
      result[i] = tensor.dtype == exl3::DType::kBF16
                      ? std::bit_cast<float>(std::uint32_t{bits} << 16)
                      : static_cast<float>(exl3::HalfToDouble(bits));
    }
    if (!std::isfinite(result[i]))
      Fail("non-finite floating tensor value");
  }
  return result;
}

const std::string& Exl3Weights::identity() const {
  std::call_once(identity_once_, [&] {
    crypto::Sha256Hasher hash;
    const auto text = [&](const std::string& value) {
      hash.Update(
          {reinterpret_cast<const std::uint8_t*>(value.data()), value.size()});
      const std::uint8_t zero = 0;
      hash.Update({&zero, 1});
    };
    // Domain/version separates the FP16 native encoder from GGUF and future
    // numerical/preprocessing contracts. Geometry is fully validated above.
    text("gufo-native-exl3-vision-v1");
    text(std::to_string(output_width_));
    for (const auto& [name, tensor] : selected_) {
      text(name);
      text(tensor->dtype_name);
      for (auto dim : tensor->shape)
        text(std::to_string(dim));
      text("payload");
      hash.Update({tensor->data, tensor->bytes});
    }
    identity_ = hash.FinishHex();
  });
  return identity_;
}
}  // namespace gufo::models::qwen::vision
