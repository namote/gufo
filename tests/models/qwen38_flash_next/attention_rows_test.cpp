// AttentionQueryRows and GateRotatedContext against the separate kernels
// they replace, byte for byte, at the Flash-Next attention geometry (24 query
// heads of 256, RoPE over 64 dims), with and without the KVarN rotation and
// with YaRN off and on.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

void CheckHip(hipError_t e, const char* what) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

template<typename T>
struct Buffer {
  explicit Buffer(std::size_t n) : count(n) {
    CheckHip(hipMalloc(&data, n * sizeof(T) + 256), "malloc");
  }
  ~Buffer() { (void)hipFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void Upload(const std::vector<T>& v) {
    CheckHip(
        hipMemcpy(data, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice),
        "upload");
  }
  std::vector<T> Download() const {
    std::vector<T> v(count);
    CheckHip(
        hipMemcpy(v.data(), data, count * sizeof(T), hipMemcpyDeviceToHost),
        "download");
    return v;
  }
  T* data{nullptr};
  std::size_t count;
};

std::vector<float> Normal(std::size_t n, std::uint32_t seed, float scale) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d;
  std::vector<float> v(n);
  for (auto& x : v)
    x = d(rng) * scale;
  return v;
}

bool Same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace

int main() {
  try {
    constexpr std::uint32_t kHeads = 24, kDim = 256, kRotary = 64;
    constexpr std::uint32_t kTokens = 37;
    constexpr std::uint32_t kStride = 2 * kHeads * kDim;
    constexpr float kTheta = 10000000.0F, kEps = 1e-6F;
    Buffer<float> qg(std::size_t{kTokens} * kStride);
    Buffer<float> gamma(kDim);
    Buffer<std::uint32_t> pos(1);
    qg.Upload(Normal(qg.count, 1, 1.0F));
    gamma.Upload(Normal(kDim, 2, 0.3F));
    pos.Upload({1234});
    const std::size_t rows = std::size_t{kTokens} * kHeads * kDim;
    bool ok = true;
    for (const bool yarn_on : {false, true}) {
      gufo::models::qwen38_flash_next::YarnRope yarn{};
      if (yarn_on) {
        yarn.enabled = 1;
        yarn.inv_factor = 0.25F;
        yarn.low = 4.0F;
        yarn.high = 24.0F;
        yarn.mscale = 1.1F;
      }
      for (const bool rotate : {false, true}) {
        Buffer<float> q_ref(rows), gate_ref(rows), q_new(rows), gate_new(rows);
        q::UnpackQGate(qg.data, kStride, q_ref.data, gate_ref.data, nullptr,
                       nullptr, kTokens, kHeads, kDim, 0, nullptr);
        q::RmsNormRows(q_ref.data, gamma.data, q_ref.data, kTokens * kHeads,
                       kDim, 1, kEps, nullptr);
        q::Rope(q_ref.data, kTokens, kHeads, kDim, kRotary, pos.data, kTheta,
                nullptr, nullptr, yarn);
        if (rotate)
          q::RotateHeadRows(q_ref.data, kTokens * kHeads, nullptr);
        if (!q::AttentionQueryRows(qg.data, kStride, gamma.data, q_new.data,
                                   gate_new.data, kTokens, kHeads, kDim,
                                   kRotary, pos.data, kTheta, kEps, rotate,
                                   nullptr, nullptr, yarn))
          throw std::runtime_error("AttentionQueryRows refused the shape");
        CheckHip(hipDeviceSynchronize(), "sync");
        const bool q_ok = Same(q_ref.Download(), q_new.Download());
        const bool g_ok = Same(gate_ref.Download(), gate_new.Download());
        std::cout << "query rows yarn=" << yarn_on << " rotate=" << rotate
                  << ": q " << (q_ok ? "equal" : "DIFFERS") << ", gate "
                  << (g_ok ? "equal" : "DIFFERS") << '\n';
        ok = ok && q_ok && g_ok;
      }
    }
    // Context: rotate out of the KVarN frame, then the output gate.
    Buffer<float> ctx_ref(rows), ctx_new(rows), gate(rows);
    const auto ctx = Normal(rows, 3, 1.0F);
    ctx_ref.Upload(ctx);
    ctx_new.Upload(ctx);
    Buffer<float> q_scratch(rows);
    q::UnpackQGate(qg.data, kStride, q_scratch.data, gate.data, nullptr,
                   nullptr, kTokens, kHeads, kDim, 0, nullptr);
    q::RotateHeadRows(ctx_ref.data, kTokens * kHeads, nullptr);
    q::SigmoidMul(ctx_ref.data, gate.data, rows, nullptr);
    q::GateRotatedContext(ctx_new.data, qg.data, kStride, kTokens, kHeads,
                          nullptr);
    CheckHip(hipDeviceSynchronize(), "sync");
    const bool c_ok = Same(ctx_ref.Download(), ctx_new.Download());
    std::cout << "gated rotated context: " << (c_ok ? "equal" : "DIFFERS")
              << '\n';
    return ok && c_ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
