#include <bit>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen38_flash_next/exl3_weights.hpp"

namespace q = gufo::models::qwen38_flash_next;
namespace ex = gufo::quant::exl3;
namespace {
void Require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
template<class F>
void Reject(F&& f, std::string_view expected) {
  bool rejected = false;
  try {
    f();
  } catch (const std::exception& error) {
    rejected = true;
    Require(
        std::string_view(error.what()).find(expected) != std::string_view::npos,
        "fixture failed for an unexpected reason");
  }
  Require(rejected, "invalid loader fixture was accepted");
}
struct Fixture {
  std::ostringstream entries;
  std::vector<std::uint8_t> payload;
  bool first = true;
  void Tensor(const std::string& name, const char* dtype,
              std::vector<std::int64_t> shape, std::size_t bytes,
              unsigned seed = 0) {
    if (!first)
      entries << ',';
    first = false;
    entries << '"' << name << "\":{\"dtype\":\"" << dtype << "\",\"shape\":[";
    for (std::size_t i = 0; i < shape.size(); ++i)
      entries << (i ? "," : "") << shape[i];
    entries << "],\"data_offsets\":[" << payload.size() << ','
            << payload.size() + bytes << "]}";
    payload.insert(payload.end(), bytes, static_cast<std::uint8_t>(seed));
  }
  void Linear(const std::string& name, int bits, bool marker = true,
              bool bias = false, bool half = false) {
    const int words = 16 * bits + (half ? 8 : 0);
    Tensor(name + ".trellis", "I16", {8, 8, words}, 8 * 8 * words * 2);
    Tensor(name + ".suh", "F16", {128}, 256);
    Tensor(name + ".svh", "F16", {128}, 256);
    if (marker)
      Tensor(name + ".mul1", "I32", {}, 4);
    if (bias)
      Tensor(name + ".bias", "F16", {128}, 256);
  }
  void Mixer(const q::Config& c, unsigned mask = 7, unsigned seed = 0x31) {
    const auto width = static_cast<std::int64_t>(c.HcDim());
    const std::string p = "mtp.hyper_connection_mixer.";
    if (mask & 1)
      Tensor(p + "hc_norm.weight", "F16", {width}, width * 2, seed);
    if (mask & 2)
      Tensor(p + "input_mix_weight_down.weight", "F16", {c.hc_low_rank, width},
             c.hc_low_rank * width * 2, seed);
    if (mask & 4)
      Tensor(p + "input_mix_weight_up.weight", "F16", {width, c.hc_low_rank},
             c.hc_low_rank * width * 2, seed);
  }
  static void Raw(const std::filesystem::path& path, std::string header,
                  const std::vector<std::uint8_t>& payload = {}) {
    while (header.size() % 8)
      header += ' ';
    std::ofstream out(path, std::ios::binary);
    const std::uint64_t length = header.size();
    out.write(reinterpret_cast<const char*>(&length), 8);
    out.write(header.data(), header.size());
    out.write(reinterpret_cast<const char*>(payload.data()), payload.size());
    Require(bool(out), "cannot write loader fixture");
  }
  void Write(const std::filesystem::path& path) const {
    Raw(path, "{" + entries.str() + "}", payload);
  }
};
struct Directory {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("qfn-exl3-rates-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  Directory() {
    std::filesystem::create_directories(path);
    std::ofstream(path / "config.json")
        << R"({"quantization_config":{"quant_method":"exl3","version":"1.5.0","bits":5.05,"head_bits":6,"out_scales":"always","codebook":"mul1"}})";
  }
  ~Directory() { std::filesystem::remove_all(path); }
  q::Exl3Storage Open(const Fixture& fixture) {
    fixture.Write(path / "model.safetensors");
    q::Exl3Storage storage;
    storage.checkpoint =
        std::make_shared<ex::Checkpoint>(ex::Checkpoint::Open(path));
    return storage;
  }
};
void Binding() {
  Directory directory;
  Fixture f;
  for (int bits : {3, 4, 5, 6, 7, 8}) {
    f.Linear("projection" + std::to_string(bits), bits);
    for (unsigned expert = 0; expert < 3; ++expert)
      f.Linear("bank" + std::to_string(bits) + "." + std::to_string(expert),
               bits);
  }
  f.Linear("mixed.0", 4);
  f.Linear("mixed.1", 6);
  f.Linear("integer1", 1);
  f.Linear("integer2", 2);
  f.Linear("fractional", 3, true, false, true);
  f.Linear("fractional1", 1, true, false, true);
  f.Linear("fractional2", 2, true, false, true);
  f.Linear("codebook", 4, false);
  f.Linear("mixedcb.0", 4);
  f.Linear("mixedcb.1", 4, false);
  f.Linear("biased", 4, true, true);
  auto storage = directory.Open(f);
  // A global bits=5.05 setting must not override any actual tensor rate.
  const auto first = q::BindExl3LinearBank(storage, "projection3", 128, 128);
  for (int bits : {3, 4, 5, 6, 7, 8}) {
    const auto single = q::BindExl3LinearBank(
        storage, "projection" + std::to_string(bits), 128, 128);
    Require(single.exl3[0].view.rate == ex::BitRate{bits, false},
            "per-projection tensor rate was not retained");
    const auto bank = q::BindExl3LinearBank(
        storage, "bank" + std::to_string(bits) + ".{}", 128, 128, 3);
    for (const auto& member : bank.exl3)
      Require(member.view.rate == ex::BitRate{bits, false},
              "uniform bank rate was not retained");
  }
  // Every format of the checkpoint has fast kernels: K1, K2 and every
  // half-integer rate of mul1, and the 3inst codebook (no marker).
  for (const auto& [name, rate] :
       {std::pair<const char*, ex::BitRate>{"integer1", {1, false}},
        {"integer2", {2, false}},
        {"fractional", {3, true}},
        {"fractional1", {1, true}},
        {"fractional2", {2, true}}})
    Require(q::BindExl3LinearBank(storage, name, 128, 128).exl3[0].view.rate ==
                rate,
            "projection rate was not retained");
  Require(q::BindExl3LinearBank(storage, "codebook", 128, 128)
                  .exl3[0]
                  .view.codebook == ex::Codebook::k3Inst,
          "3inst projection codebook was not retained");
  const auto retained = storage.linears.size();
  Reject([&] { (void)q::BindExl3LinearBank(storage, "mixed.{}", 128, 128, 2); },
         "mixed expert formats");
  Reject(
      [&] { (void)q::BindExl3LinearBank(storage, "mixedcb.{}", 128, 128, 2); },
      "mixed expert formats");
  Require(
      storage.linears.size() == retained && first.exl3[0].view.rate.bits == 3,
      "failed bank binding invalidated retained descriptors");
  Reject([&] { (void)q::BindExl3LinearBank(storage, "biased", 128, 128); },
         "invalid projection");
  Reject([&] { (void)q::BindExl3LinearBank(storage, "projection4", 256, 128); },
         "invalid projection");
  Reject([&] { (void)q::BindExl3LinearBank(storage, "mixed", 128, 128, 2); },
         "invalid expert bank");
  const auto mtp = q::BindExl3LinearBank(storage, "bank6.{}", 128, 128, 3);
  Require(mtp.exl3[0].view.rate.bits == 6, "MTP bank must retain its own rate");
}
void MixerSources() {
  q::Config c;
  c.hidden_size = 128;
  c.hc_count = 4;
  c.hc_low_rank = 2;
  Directory directory;
  Fixture full;
  full.Mixer(c);
  const auto patch = directory.path / "patch.safetensors";
  full.Write(patch);
  {
    auto storage = directory.Open(full);
    q::PrepareExl3MtpMixer(storage, c);
    q::PrepareExl3MtpMixer(storage, c, patch);
    Require(storage.supplementary.empty(),
            "in-shard mixer was replaced by patch");
    Fixture changed;
    changed.Mixer(c, 7, 0x32);
    changed.Write(directory.path / "changed.safetensors");
    Reject(
        [&] {
          q::PrepareExl3MtpMixer(storage, c,
                                 directory.path / "changed.safetensors");
        },
        "differs from patch");
    for (const std::string bad : {"[]", "7"}) {
      Fixture::Raw(directory.path / "bad.safetensors", bad);
      Reject(
          [&] {
            q::PrepareExl3MtpMixer(storage, c,
                                   directory.path / "bad.safetensors");
          },
          "must be an object");
    }
    const std::string name = "mtp.hyper_connection_mixer.hc_norm.weight";
    for (const std::string offset : {"-1", "0.5", "\"bad\""}) {
      const std::string header =
          "{\"" + name +
          "\":{\"dtype\":\"F16\",\"shape\":[512],\"data_offsets\":[" + offset +
          ",1024]}}";
      Fixture::Raw(directory.path / "bad.safetensors", header,
                   std::vector<std::uint8_t>(1024, 0x31));
      Reject(
          [&] {
            q::PrepareExl3MtpMixer(storage, c,
                                   directory.path / "bad.safetensors");
          },
          "invalid MTP mixer patch offset");
    }
  }
  for (unsigned mask : {0U, 1U, 3U}) {
    Fixture partial;
    partial.Mixer(c, mask);
    auto storage = directory.Open(partial);
    Reject([&] { q::PrepareExl3MtpMixer(storage, c); },
           "missing/invalid MTP mixer");
    q::PrepareExl3MtpMixer(storage, c, patch);
    Require(storage.supplementary.size() == 3 - std::popcount(mask),
            "patch did not supply exactly missing mixer tensors");
    for (const auto& [name, tensor] : storage.supplementary)
      Require(tensor.bytes != 0 && tensor.data[0] == 0x31,
              "supplementary mixer lost its owned payload");
  }
}
void CalibratedHc() {
  namespace json = gufo::json;
  namespace crypto = gufo::crypto;
  Directory directory;
  q::Config c;
  c.num_layers = 1;
  c.hidden_size = 8;
  c.hc_count = 4;
  c.hc_low_rank = 32;
  std::vector<std::string> names{
      "model.language_model.layers.0.attn_hyper_connection",
      "model.language_model.layers.0.mlp_hyper_connection",
      "model.language_model.hyper_connection_mixer",
      "mtp.layers.0.attn_hyper_connection",
      "mtp.layers.0.mlp_hyper_connection",
      "mtp.hyper_connection_mixer"};
  Fixture original, packed;
  for (const auto& prefix : names)
    for (const auto* projection : {"down", "up"}) {
      const auto name = prefix + ".input_mix_weight_" + projection + ".weight";
      original.Tensor(name, "F16", {32, 32}, 32 * 32 * 2, 0x31);
      packed.Tensor(name, "U8", {32, 1, 34}, 32 * 34);
    }
  auto storage = directory.Open(original);
  // The single-shard fixture is already mapped; only this file's identity
  // participates in the artifact association tested below.
  std::ofstream(directory.path / "model.safetensors.index.json") << "{}";
  q::PrepareExl3HcQ8(storage, c, directory.path);
  Require(!storage.hc_q8, "absent sidecar changed the legacy path");
  const auto sidecar = directory.path / "hc_q8";
  std::filesystem::create_directory(sidecar);
  Reject([&] { q::PrepareExl3HcQ8(storage, c, directory.path); }, "manifest");
  std::filesystem::copy_file(directory.path / "config.json",
                             sidecar / "config.json");
  packed.Write(sidecar / "model.safetensors");
  auto manifest = json::Value::object();
  manifest["schema"] = "gufo.flash-next.hc-q8.v1";
  manifest["model_index_sha256"] =
      crypto::Sha256FileHex(directory.path / "model.safetensors.index.json");
  auto entries = json::Value::object();
  const auto refresh = [&] {
    auto checkpoint = ex::Checkpoint::Open(sidecar);
    for (const auto& [name, tensor] : checkpoint.tensors()) {
      auto entry = json::Value::object();
      const auto* source = storage.checkpoint->Find(name);
      entry["target_weight_sha256"] =
          crypto::Sha256Hex({source->data, source->bytes});
      entry["packed_weight_sha256"] =
          crypto::Sha256Hex({tensor.data, tensor.bytes});
      entries[name] = std::move(entry);
    }
    manifest["tensors"] = entries;
    std::ofstream(directory.path / "hc_q8.json") << manifest.dump();
  };
  refresh();
  q::PrepareExl3HcQ8(storage, c, directory.path);
  Require(storage.hc_q8 && storage.hc_q8->tensors().size() == 12,
          "complete sidecar not retained");
  storage.hc_q8.reset();
  // --hc-mode: only q8-calibrated reads the sidecar; fp16 and q8 ignore a
  // valid one.
  for (const auto mode : {q::HcMode::kFp16, q::HcMode::kQ8}) {
    q::ApplyExl3HcMode(storage, c, directory.path, mode);
    Require(!storage.hc_q8, "fp16/q8 read the calibrated sidecar");
  }
  q::ApplyExl3HcMode(storage, c, directory.path, q::HcMode::kQ8Calibrated);
  Require(storage.hc_q8 && storage.hc_q8->tensors().size() == 12,
          "q8-calibrated did not load the sidecar");
  q::ApplyExl3HcMode(storage, c, directory.path, q::HcMode::kFp16);
  Require(!storage.hc_q8, "fp16 kept a previously loaded sidecar");
  const auto key = names.front() + ".input_mix_weight_down.weight";
  entries[key]["target_weight_sha256"] = std::string(64, '0');
  manifest["tensors"] = entries;
  std::ofstream(directory.path / "hc_q8.json") << manifest.dump();
  Reject([&] { q::PrepareExl3HcQ8(storage, c, directory.path); },
         "identity mismatch");
  refresh();
  packed.payload[2] = 128;
  packed.Write(sidecar / "model.safetensors");
  Reject([&] { q::PrepareExl3HcQ8(storage, c, directory.path); },
         "identity mismatch");
  refresh();
  Reject([&] { q::PrepareExl3HcQ8(storage, c, directory.path); },
         "invalid Q8 code");
  packed.payload[2] = 0;
  packed.payload[0] = 0;
  packed.payload[1] = 0x7c;
  packed.Write(sidecar / "model.safetensors");
  refresh();
  Reject([&] { q::PrepareExl3HcQ8(storage, c, directory.path); },
         "invalid Q8 scale");
  packed.payload[1] = 0;
  packed.Write(sidecar / "model.safetensors");
  refresh();
  manifest["model_index_sha256"] = std::string(64, '0');
  std::ofstream(directory.path / "hc_q8.json") << manifest.dump();
  Reject([&] { q::PrepareExl3HcQ8(storage, c, directory.path); },
         "model identity");
  // An invalid sidecar fails q8-calibrated, naming the mode; the other modes
  // never open it.
  Reject(
      [&] {
        q::ApplyExl3HcMode(storage, c, directory.path,
                           q::HcMode::kQ8Calibrated);
      },
      "--hc-mode q8-calibrated: Flash-Next HC Q8: manifest schema/model "
      "identity");
  // The message carries one prefix.
  try {
    q::ApplyExl3HcMode(storage, c, directory.path, q::HcMode::kQ8Calibrated);
  } catch (const std::exception& error) {
    Require(std::string_view(error.what()).starts_with("--hc-mode "),
            "q8-calibrated error lost its mode prefix");
  }
  Require(!storage.hc_q8, "a rejected sidecar stayed bound");
  for (const auto mode : {q::HcMode::kFp16, q::HcMode::kQ8})
    q::ApplyExl3HcMode(storage, c, directory.path, mode);
  // Without a sidecar q8-calibrated fails instead of switching modes.
  std::filesystem::remove_all(sidecar);
  std::filesystem::remove(directory.path / "hc_q8.json");
  Reject(
      [&] {
        q::ApplyExl3HcMode(storage, c, directory.path,
                           q::HcMode::kQ8Calibrated);
      },
      "--hc-mode q8-calibrated requires the calibrated sidecar");
  Require(!storage.hc_q8, "absent sidecar bound calibrated weights");
  for (const auto mode : {q::HcMode::kFp16, q::HcMode::kQ8}) {
    q::ApplyExl3HcMode(storage, c, directory.path, mode);
    Require(!storage.hc_q8, "fp16/q8 bound calibrated weights");
  }
}
}  // namespace
int main() {
  try {
    Binding();
    MixerSources();
    CalibratedHc();
    std::cout << "Flash-Next EXL3 rate binding and mixer sources passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
