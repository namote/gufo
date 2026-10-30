// X2a EXL3 reference GPU path (src/core/quant/exl3/reference.hpp) against the
// X0 CPU reference: trellis decode bit for bit (every K, codebook and
// half-integer K), the 128-point rotation, and whole projections.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "src/core/quant/exl3/format.hpp"
#include "src/core/quant/exl3/reference.hpp"

namespace {

using namespace gufo::quant::exl3;

int failures = 0;

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << "\n";
    ++failures;
  }
}

template<typename T>
T* Upload(const std::vector<T>& host) {
  T* device = nullptr;
  if (hipMalloc(&device, host.size() * sizeof(T)) != hipSuccess)
    std::abort();
  if (hipMemcpy(device, host.data(), host.size() * sizeof(T),
                hipMemcpyHostToDevice) != hipSuccess)
    std::abort();
  return device;
}

template<typename T>
std::vector<T> Download(const T* device, std::size_t count) {
  std::vector<T> host(count);
  if (hipMemcpy(host.data(), device, count * sizeof(T),
                hipMemcpyDeviceToHost) != hipSuccess)
    std::abort();
  return host;
}

std::vector<std::uint16_t> RandomScales(std::mt19937& rng, std::size_t n,
                                        std::uint16_t base) {
  std::vector<std::uint16_t> v(n);
  for (auto& x : v) {
    const std::uint32_t r = rng();
    x = static_cast<std::uint16_t>(((r >> 10) & 1) << 15 | base | (r & 0x3ff));
  }
  return v;
}

void TestDequant() {
  std::mt19937 rng(20261030);
  const std::size_t k = 256, n = 256;
  std::vector<std::pair<Codebook, BitRate>> cases;
  for (auto cb : {Codebook::k3Inst, Codebook::kMcg, Codebook::kMul1})
    for (int b = 1; b <= 8; ++b)
      cases.push_back({cb, {b, false}});
  for (int b = 1; b <= 3; ++b)
    cases.push_back({Codebook::kMul1, {b, true}});
  for (const auto& [cb, rate] : cases) {
    std::vector<std::uint16_t> trellis((k / 16) * (n / 16) * rate.TileWords());
    for (auto& w : trellis)
      w = static_cast<std::uint16_t>(rng());
    std::vector<std::uint16_t> inner(k * n);
    DequantInner(trellis.data(), k, n, rate, cb, inner.data());
    auto* d_trellis = Upload(trellis);
    float* d_out = nullptr;
    (void)hipMalloc(&d_out, 64 * k * sizeof(float));
    DeviceLinear lin{d_trellis,
                     nullptr,
                     nullptr,
                     static_cast<std::uint32_t>(k),
                     static_cast<std::uint32_t>(n),
                     rate,
                     cb};
    bool same = true;
    for (std::size_t n0 = 0; n0 < n; n0 += 64) {
      LaunchReferenceDequantF32(lin, n0, 64, d_out, nullptr);
      const auto out = Download(d_out, 64 * k);
      for (std::size_t j = 0; j < 64 && same; ++j)
        for (std::size_t i = 0; i < k && same; ++i)
          same = HalfFromDouble(out[j * k + i]) == inner[i * n + n0 + j] &&
                 static_cast<double>(out[j * k + i]) ==
                     HalfToDouble(inner[i * n + n0 + j]);
    }
    Expect(same, "dequant K=" + std::to_string(rate.bits) +
                     (rate.half ? ".5" : "") + " codebook " +
                     std::to_string(static_cast<int>(cb)));
    (void)hipFree(d_trellis);
    (void)hipFree(d_out);
  }
}

void TestRotate() {
  std::mt19937 rng(7);
  const std::size_t rows = 3, width = 384;
  std::normal_distribution<float> normal;
  std::vector<float> x(rows * width);
  for (auto& v : x)
    v = normal(rng);
  const auto pre = RandomScales(rng, width, 0x3800);
  const auto post = RandomScales(rng, width, 0x3c00);
  auto* dx = Upload(x);
  auto* dpre = Upload(pre);
  auto* dpost = Upload(post);
  float* dout = nullptr;
  (void)hipMalloc(&dout, x.size() * sizeof(float));
  LaunchReferenceRotateRows(dx, width, rows, width, dpre, dpost, dout, width,
                            nullptr);
  const auto out = Download(dout, x.size());
  double worst = 0.0;
  for (std::size_t r = 0; r < rows; ++r) {
    std::vector<double> v(width);
    for (std::size_t i = 0; i < width; ++i)
      v[i] = x[r * width + i] * HalfToDouble(pre[i]);
    HadamardBlocks(std::span<double>(v));
    // Relative to the row's largest output, as for projections: an FP32
    // butterfly's error is absolute (~1e-7 of the row scale), so a
    // per-element ratio would grade near-zero outputs on noise.
    double scale = 0.0;
    for (std::size_t i = 0; i < width; ++i)
      scale = std::max(scale, std::fabs(v[i] * HalfToDouble(post[i])));
    for (std::size_t i = 0; i < width; ++i) {
      const double expect = v[i] * HalfToDouble(post[i]);
      worst = std::max(worst, std::fabs(out[r * width + i] - expect) / scale);
    }
  }
  Expect(worst < 1e-5, "rotation relative error " + std::to_string(worst));
  std::cout << "rotation max relative error " << worst << "\n";
  for (auto* p : {static_cast<void*>(dx), static_cast<void*>(dpre),
                  static_cast<void*>(dpost), static_cast<void*>(dout)})
    (void)hipFree(p);
}

void TestProjection() {
  std::mt19937 rng(11);
  const std::size_t k = 512, n = 384, rows = 5;
  const BitRate rate{5, false};
  std::vector<std::uint16_t> trellis((k / 16) * (n / 16) * rate.TileWords());
  for (auto& w : trellis)
    w = static_cast<std::uint16_t>(rng());
  const auto suh = RandomScales(rng, k, 0x3800);
  const auto svh = RandomScales(rng, n, 0x2000);
  std::normal_distribution<float> normal;
  std::vector<float> x(rows * k);
  for (auto& v : x)
    v = normal(rng);
  auto* dt = Upload(trellis);
  auto* dsuh = Upload(suh);
  auto* dsvh = Upload(svh);
  auto* dx = Upload(x);
  float *dy = nullptr, *drot = nullptr, *dw = nullptr;
  (void)hipMalloc(&dy, rows * n * sizeof(float));
  (void)hipMalloc(&drot, rows * k * sizeof(float));
  (void)hipMalloc(&dw, 128 * k * sizeof(float));  // forces three column chunks
  DeviceLinear lin{dt,
                   dsuh,
                   dsvh,
                   static_cast<std::uint32_t>(k),
                   static_cast<std::uint32_t>(n),
                   rate,
                   Codebook::kMul1};
  LaunchReferenceProjection(lin, dx, k, rows, dy, n,
                            {drot, rows * k, dw, 128 * k}, nullptr);
  const auto y = Download(dy, rows * n);
  std::vector<std::uint16_t> inner(k * n);
  DequantInner(trellis.data(), k, n, rate, Codebook::kMul1, inner.data());
  double worst = 0.0;
  for (std::size_t r = 0; r < rows; ++r) {
    std::vector<double> xr(k), yr(n);
    for (std::size_t i = 0; i < k; ++i)
      xr[i] = x[r * k + i];
    GemvReference(xr, inner.data(), suh.data(), svh.data(), k, n, yr);
    double scale = 0.0;
    for (const double v : yr)
      scale = std::max(scale, std::fabs(v));
    for (std::size_t j = 0; j < n; ++j)
      worst = std::max(worst, std::fabs(y[r * n + j] - yr[j]) / scale);
  }
  Expect(worst < 1e-5, "projection relative error " + std::to_string(worst));
  std::cout << "projection max relative error " << worst << "\n";
  for (auto* p : {static_cast<void*>(dt), static_cast<void*>(dsuh),
                  static_cast<void*>(dsvh), static_cast<void*>(dx),
                  static_cast<void*>(dy), static_cast<void*>(drot),
                  static_cast<void*>(dw)})
    (void)hipFree(p);
}

}  // namespace

int main() {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
    std::cerr << "no HIP device\n";
    return 77;
  }
  TestDequant();
  TestRotate();
  TestProjection();
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "exl3_reference_test: OK\n";
  return 0;
}
