// Frozen diagnostic only: all HC sites, scalar teacher-forced decode.
#pragma once
#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/runtime/device_model.hpp"

namespace gufo_hc_all {
inline bool active = false;
inline std::size_t window = 0, position = 0, first = 0;
inline std::map<const void*, std::size_t> sites;
inline std::vector<std::string> names;
struct Output {
  std::ofstream file;
  std::size_t rows = 0;
};
inline std::array<Output, 200> outputs;
inline const char* Root() {
  return std::getenv("GUFO_HC_ALL_DIR");
}
inline bool Enabled() {
  return Root() != nullptr && Root()[0] != '\0';
}
inline void Require(bool ok, const char* why) {
  if (!ok)
    throw std::runtime_error(why);
}
inline std::string Id(std::size_t i) {
  return "site-" + std::string(i < 10 ? "00" : (i < 100 ? "0" : "")) +
         std::to_string(i);
}
inline void Register(
    const gufo::models::qwen38_flash_next::rocm::DeviceModel& model) {
  Require(sites.empty() && model.layers().size() == 48,
          "HC registry already populated or wrong model");
  const auto add = [&](const auto& m, const std::string& name) {
    Require(m.down.data && m.down_q8.data && m.up.data,
            "HC registry missing native tensors");
    Require(m.down_q8.rows == 320 && m.down_q8.cols == 10240 &&
                m.up.rows == 10240 && m.up.cols == 320,
            "HC registry shape differs");
    Require(sites.emplace(m.down.data, names.size()).second,
            "duplicate HC pointer");
    names.push_back(name);
    const auto transfer = [&](const char* env, bool import) {
      const char* root = std::getenv(env);
      if (!root || !root[0])
        return;
      auto dir = std::filesystem::path(root) / Id(names.size() - 1);
      if (!import) {
        Require(!std::filesystem::exists(dir), "native Q8 export exists");
        std::filesystem::create_directories(dir);
      }
      for (unsigned p = 0; p < 2; ++p) {
        const auto& tensor = p ? m.up : m.down_q8;
        const auto bytes = std::size_t{tensor.rows} * (tensor.cols / 32) * 34;
        std::vector<char> data(bytes);
        auto path = dir / (p ? "up.q8" : "down.q8");
        if (import) {
          Require(std::filesystem::file_size(path) == bytes,
                  "candidate Q8 size differs");
          std::ifstream in(path, std::ios::binary);
          in.read(data.data(), data.size());
          Require(bool(in), "candidate Q8 read failed");
          Require(hipMemcpy(tensor.data, data.data(), bytes,
                            hipMemcpyHostToDevice) == hipSuccess,
                  "candidate Q8 upload failed");
        } else {
          Require(hipMemcpy(data.data(), tensor.data, bytes,
                            hipMemcpyDeviceToHost) == hipSuccess,
                  "native Q8 download failed");
          std::ofstream out(path, std::ios::binary);
          out.write(data.data(), data.size());
          Require(bool(out), "native Q8 export failed");
        }
      }
    };
    transfer("GUFO_HC_Q8_EXPORT", false);
    transfer("GUFO_HC_Q8_IMPORT", true);
  };
  for (std::size_t i = 0; i < 48; ++i) {
    const auto prefix = "model.language_model.layers." + std::to_string(i);
    add(model.layers()[i].hc_attn, prefix + ".attn_hyper_connection");
    add(model.layers()[i].hc_ffn, prefix + ".mlp_hyper_connection");
  }
  add(model.hc_head(), "model.language_model.hyper_connection_mixer");
  add(model.mtp().hc_attn, "mtp.layers.0.attn_hyper_connection");
  add(model.mtp().hc_ffn, "mtp.layers.0.mlp_hyper_connection");
  add(model.mtp().nextn_head, "mtp.hyper_connection_mixer");
  Require(names.size() == 100, "HC registry coverage incomplete");
}
inline void Begin(std::size_t w, std::size_t p) {
  Require(!active && w < 16 && p == 32, "unsupported HC window");
  window = w;
  position = first = p;
  for (auto& out : outputs) {
    out.file.close();
    out.file.clear();
    out.rows = 0;
  }
  if (Enabled())
    Require(!std::filesystem::exists(std::filesystem::path(Root()) /
                                     std::to_string(w)),
            "capture window exists");
  active = Enabled();
}
inline void Save(const void* mixer, bool up, const float* input,
                 std::size_t rows, std::size_t cols, hipStream_t stream) {
  if (!active)
    return;
  Require(rows == 1 && cols == (up ? 320U : 10240U),
          "scalar HC capture shape changed");
  const auto it = sites.find(mixer);
  Require(it != sites.end(), "unregistered HC site");
  auto& out = outputs[it->second * 2 + (up ? 1 : 0)];
  Require(out.rows == position - first, "duplicate or skipped HC row");
  if (!out.file.is_open()) {
    auto dir =
        std::filesystem::path(Root()) / std::to_string(window) / Id(it->second);
    std::filesystem::create_directories(dir);
    auto file = dir / (up ? "up.f32" : "down.f32");
    Require(!std::filesystem::exists(file), "capture payload exists");
    out.file.open(file, std::ios::binary);
  }
  std::vector<float> host(cols);
  Require(hipMemcpyAsync(host.data(), input, cols * sizeof(float),
                         hipMemcpyDeviceToHost, stream) == hipSuccess &&
              hipStreamSynchronize(stream) == hipSuccess,
          "HC input copy failed");
  for (float x : host)
    Require(std::isfinite(x), "nonfinite HC input");
  out.file.write(reinterpret_cast<const char*>(host.data()),
                 cols * sizeof(float));
  Require(bool(out.file), "HC input write failed");
  ++out.rows;
}
inline void End(std::size_t end) {
  if (active) {
    for (auto& out : outputs) {
      Require(out.rows == end - first, "HC coverage incomplete");
      out.file.close();
      Require(bool(out.file), "HC capture close failed");
    }
    auto file = std::filesystem::path(Root()) / std::to_string(window) /
                "COVERAGE.json";
    std::ofstream meta(file);
    meta << "{\"phase\":\"scalar_decode_teacher_forced\",\"first\":" << first
         << ",\"end\":" << end << ",\"rows_per_tensor\":" << end - first
         << ",\"sites\":[";
    for (std::size_t i = 0; i < names.size(); ++i)
      meta << (i ? "," : "") << "\"" << names[i] << "\"";
    meta << "]}\n";
    meta.close();
    Require(bool(meta), "HC coverage receipt failed");
  }
  active = false;
}
}  // namespace gufo_hc_all
