#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen38_flash_next/exl3_weights.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {
[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error("Flash-Next HC Q8: " + message);
}
std::string ReadManifest(const std::filesystem::path& path) {
  if (!std::filesystem::is_regular_file(path) ||
      std::filesystem::file_size(path) > 1024 * 1024)
    Fail("missing or oversized manifest");
  std::ifstream in(path, std::ios::binary);
  if (!in)
    Fail("cannot read manifest");
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
}  // namespace

void PrepareExl3HcQ8(Exl3Storage& storage, const Config& config,
                     const std::filesystem::path& model) {
  const auto directory = model / "hc_q8";
  const auto receipt = model / "hc_q8.json";
  if (!std::filesystem::exists(directory) &&
      !std::filesystem::is_symlink(directory) &&
      !std::filesystem::exists(receipt) &&
      !std::filesystem::is_symlink(receipt))
    return;
  if (!std::filesystem::is_directory(directory) || !storage.checkpoint)
    Fail("invalid sidecar directory/checkpoint");
  const auto manifest = json::parse(ReadManifest(receipt));
  if (manifest.member_str("schema") != "gufo.flash-next.hc-q8.v1" ||
      manifest.member_str("model_index_sha256") !=
          crypto::Sha256FileHex(model / "model.safetensors.index.json"))
    Fail("manifest schema/model identity mismatch");
  const auto* entries = manifest.find("tensors");
  const std::size_t count = 2 * (2 * config.num_layers + 4);
  if (!entries || !entries->is_object() || entries->members().size() != count)
    Fail("incomplete tensor manifest");
  auto packed = std::make_shared<quant::exl3::Checkpoint>(
      quant::exl3::Checkpoint::Open(directory));
  if (packed->tensors().size() != count)
    Fail("sidecar tensor count mismatch");
  const auto check = [&](const std::string& prefix) {
    for (const bool up : {false, true}) {
      const auto name =
          prefix + ".input_mix_weight_" + (up ? "up" : "down") + ".weight";
      const auto cols = up ? config.hc_low_rank : config.HcDim();
      const auto rows = up ? config.HcDim() : config.hc_low_rank;
      const auto* original = storage.checkpoint->Find(name);
      const auto* tensor = packed->Find(name);
      const auto* entry = entries->find(name);
      if (!original || original->dtype != quant::exl3::DType::kF16 ||
          original->shape != std::vector<std::int64_t>{rows, cols} || !tensor ||
          !entry || !entry->is_object() || cols % 32 ||
          tensor->dtype != quant::exl3::DType::kU8 ||
          tensor->shape != std::vector<std::int64_t>{rows, cols / 32, 34} ||
          tensor->bytes != std::size_t{rows} * (cols / 32) * 34)
        Fail("invalid tensor shape/dtype " + name);
      if (entry->member_str("target_weight_sha256") !=
              crypto::Sha256Hex({original->data, original->bytes}) ||
          entry->member_str("packed_weight_sha256") !=
              crypto::Sha256Hex({tensor->data, tensor->bytes}))
        Fail("tensor identity mismatch " + name);
      // Validate the stored format before the GPU sees it. No fallback to RTN
      // when an explicitly supplied calibrated artifact is malformed.
      for (std::size_t offset = 0; offset < tensor->bytes; offset += 34) {
        std::uint16_t bits;
        std::memcpy(&bits, tensor->data + offset, 2);
        const auto scale = quant::exl3::HalfToDouble(bits);
        if (!std::isfinite(scale) || scale < 0)
          Fail("invalid Q8 scale " + name);
        for (std::size_t i = 2; i < 34; ++i)
          if (tensor->data[offset + i] == 128)
            Fail("invalid Q8 code " + name);
      }
    }
  };
  for (std::uint32_t i = 0; i < config.num_layers; ++i) {
    const auto prefix = "model.language_model.layers." + std::to_string(i);
    check(prefix + ".attn_hyper_connection");
    check(prefix + ".mlp_hyper_connection");
  }
  check("model.language_model.hyper_connection_mixer");
  check("mtp.layers.0.attn_hyper_connection");
  check("mtp.layers.0.mlp_hyper_connection");
  check("mtp.hyper_connection_mixer");
  storage.hc_q8 = std::move(packed);
}
}  // namespace gufo::models::qwen38_flash_next
