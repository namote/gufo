// Experiment-only host capture. Included exclusively in the frozen collector.
#pragma once
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gufo_hc_capture {
inline std::size_t window = 0, position = 0;
inline std::string phase;
inline const char* Root() { return std::getenv("GUFO_HC_CAPTURE_DIR"); }
inline bool Enabled() { return Root() != nullptr && Root()[0] != '\0'; }
inline void Context(std::size_t w, std::size_t p, bool decode) {
  window = w;
  position = p;
  phase = decode ? "decode" : "prefill";
}
inline void Save(const char* projection, const float* input,
                 std::size_t rows, std::size_t cols, hipStream_t stream) {
  if (!Enabled()) return;
  if (rows == 0 || rows > 512 || (cols != 10240 && cols != 320) ||
      window >= 64 || position + rows > 4096 || phase.empty())
    throw std::runtime_error("HC capture exceeds explicit pilot bounds");
  std::vector<float> host(rows * cols);
  auto status = hipMemcpyAsync(host.data(), input, host.size() * sizeof(float),
                               hipMemcpyDeviceToHost, stream);
  if (status != hipSuccess || hipStreamSynchronize(stream) != hipSuccess)
    throw std::runtime_error("HC capture copy failed");
  for (float v : host)
    if (!std::isfinite(v)) throw std::runtime_error("HC capture nonfinite input");
  const auto dir = std::filesystem::path(Root()) / std::to_string(window);
  std::filesystem::create_directories(dir);
  const auto file = dir / (std::string(projection) + "-" + std::to_string(position) + ".f32");
  if (std::filesystem::exists(file))
    throw std::runtime_error("HC capture would overwrite a completed chunk");
  std::ofstream out(file, std::ios::binary);
  out.write(reinterpret_cast<const char*>(host.data()), host.size() * sizeof(float));
  out.close();
  if (!out) throw std::runtime_error("HC capture write failed");
  std::ofstream meta(dir / "chunks.jsonl", std::ios::app);
  meta << "{\"projection\":\"" << projection << "\",\"position\":" << position
       << ",\"rows\":" << rows << ",\"cols\":" << cols
       << ",\"phase\":\"" << phase << "\",\"file\":\"" << file.filename().string()
       << "\"}\n";
  meta.close();
  if (!meta) throw std::runtime_error("HC capture metadata write failed");
}
}  // namespace gufo_hc_capture
