#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/exl3/format.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/exl3_moe.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace ex = gufo::quant::exl3;
namespace {
void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}
void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}
struct Buffer {
  void* data{nullptr};
  explicit Buffer(std::size_t bytes) { Check(hipMalloc(&data, bytes)); }
  ~Buffer() { (void)hipFree(data); }
};

void PlainProjection(unsigned k, unsigned n, bool half) {
  const std::string name = std::string(half ? "F16" : "F32") +
                           " k=" + std::to_string(k) +
                           " n=" + std::to_string(n);
  constexpr unsigned rows = 136;
  std::mt19937 rng(33771 + k + n);
  std::vector<float> x(rows * k), decoded(std::size_t{k} * n);
  for (auto& value : x)
    value = (static_cast<int>(rng() % 2000) - 1000) * 0.00003125F;
  std::vector<std::uint8_t> weight(decoded.size() * (half ? 2 : 4));
  for (std::size_t i = 0; i < decoded.size(); ++i) {
    const float value = (static_cast<int>(rng() % 2000) - 1000) * 0.000001F;
    if (half) {
      const auto bits = ex::HalfFromDouble(value);
      std::memcpy(weight.data() + i * 2, &bits, 2);
      decoded[i] = static_cast<float>(ex::HalfToDouble(bits));
    } else {
      std::memcpy(weight.data() + i * 4, &value, 4);
      decoded[i] = value;
    }
  }
  Buffer dw(weight.size()), dx(x.size() * 4), dy(std::size_t{rows} * n * 4);
  Check(hipMemset(dy.data, 0, std::size_t{rows} * n * 4));
  Check(
      hipMemcpy(dw.data, weight.data(), weight.size(), hipMemcpyHostToDevice));
  Check(hipMemcpy(dx.data, x.data(), x.size() * 4, hipMemcpyHostToDevice));
  q::DeviceTensor w;
  w.data = dw.data;
  w.type = half ? gufo::core::GgmlType::kF16 : gufo::core::GgmlType::kF32;
  w.cols = k;
  w.rows = n;
  const q::Exl3Scratch scratch;
  const auto project = [&](unsigned first, unsigned count, bool fast) {
    q::Exl3Projection(w, static_cast<const float*>(dx.data) + first * k,
                      static_cast<float*>(dy.data) + first * n, count, scratch,
                      nullptr, fast);
  };
  const auto download = [&] {
    std::vector<float> result(std::size_t{rows} * n);
    Check(hipMemcpy(result.data(), dy.data, result.size() * 4,
                    hipMemcpyDeviceToHost));
    return result;
  };
  for (unsigned r = 0; r < 8; ++r)
    project(r, 1, true);
  const auto serial = download();
  for (unsigned count : {2U, 4U, 6U, 8U}) {
    project(0, count, true);
    const auto actual = download();
    Require(std::memcmp(serial.data(), actual.data(), count * n * 4) == 0,
            name + " scalar/cohort mismatch at C" + std::to_string(count));
    std::cout << "PASS " << name << " C" << count << " exact\n";
  }
  // Real session regression: the EXL3 pack's plain HC/router/GDN operands
  // must keep the same precision and reduction in all prefill chunks.
  project(0, rows, false);
  const auto bulk = download();
  for (unsigned boundary : {94U, 103U, 104U, 127U, 128U, 135U}) {
    project(0, boundary, false);
    project(boundary, rows - boundary, false);
    const auto split = download();
    Require(
        std::memcmp(bulk.data(), split.data(), bulk.size() * 4) == 0,
        name + " prefill 136 mismatch at boundary " + std::to_string(boundary));
    std::cout << "PASS " << name << " prefill 136 boundary=" << boundary
              << " exact\n";
  }
  Require(std::memcmp(bulk.data(), serial.data(), 8 * n * 4) == 0,
          name + " prefill/decode arithmetic differs");
  double error = 0, energy = 0;
  for (unsigned j = 0; j < std::min(n, 16U); ++j) {
    double expected = 0;
    for (unsigned i = 0; i < k; ++i)
      expected += static_cast<double>(x[i]) * decoded[std::size_t{j} * k + i];
    error += std::pow(bulk[j] - expected, 2);
    energy += expected * expected;
  }
  const double relative = std::sqrt(error / std::max(energy, 1e-30));
  Require(relative <= 1e-5,
          name + " differs from FP64: " + std::to_string(relative));
}
}  // namespace

int main() {
  try {
    PlainProjection(10240, 320, true);  // Plain HC down.
    PlainProjection(320, 10240, true);  // Ragged HC up, not a whole K pass.
    PlainProjection(2560, 513, true);   // Router + shared gate.
    PlainProjection(2560, 96, false);   // Permuted GDN alpha/beta.
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
