#include <hip/hip_fp16.h>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/runtime/executor.hpp"
namespace q = gufo::models::qwen38_flash_next::rocm;
void Check(hipError_t x) {
  if (x != hipSuccess)
    throw std::runtime_error(hipGetErrorString(x));
}
// Seed the production deferred-MoE path without loading a model. This catches
// projection before aggregation and double application of the prior residual.
namespace gufo::models::qwen38_flash_next::rocm {
struct ExecutorTestAccess {
  static void DeferredMoe() {
    DeviceModel model;
    model.config_.hidden_size = 4;
    model.config_.hc_count = 4;
    model.config_.num_layers = 1;
    model.config_.num_experts = 2;
    model.config_.num_experts_used = 2;
    Executor executor;
    executor.model_ = &model;
    auto alloc = [&](std::vector<float> values) {
      float* device = nullptr;
      Check(hipMalloc(&device, values.size() * sizeof(float)));
      executor.allocations_.push_back(device);
      Check(hipMemcpy(device, values.data(), values.size() * sizeof(float),
                      hipMemcpyHostToDevice));
      return device;
    };
    const std::vector<float> residual(16, .125f);
    const std::vector<float> attention{.25f, -.5f, .75f, 1.f};
    const std::vector<float> old_weights{.25f, .5f, .75f, 1.f};
    const std::vector<float> shared{.5f, -.25f, 1.f, -.5f};
    executor.s_.res = alloc(residual);
    executor.s_.block_out = alloc(attention);
    executor.s_.hc_defer_w = alloc(old_weights);
    executor.s_.weights = alloc({.25f, .75f});
    executor.s_.shexp_out = alloc(shared);
    executor.s_.router = alloc({0.f, 0.f, 0.f});
    executor.s_.inject = alloc({0.f, 0.f, 0.f, 0.f});
    executor.s_.xn = alloc(std::vector<float>(16));
    executor.s_.down_e = alloc(std::vector<float>(4));
    std::array<__half, 8> down;
    const std::array<float, 8> down_float{1.f,   .5f, -.5f, -1.f,
                                          -.25f, .5f, .75f, 1.f};
    for (unsigned i = 0; i < down.size(); ++i)
      down[i] = __float2half(down_float[i]);
    Check(hipMemcpy(executor.s_.down_e, down.data(), sizeof(down),
                    hipMemcpyHostToDevice));
    executor.abliteration_direction_ = alloc({.5f, .5f, .5f, .5f});
    executor.moe_pending_ = true;
    executor.res_deferred_ = true;
    executor.Combine(executor.s_.res, nullptr, 1, nullptr, false, .75f);
    std::array<float, 16> result;
    Check(hipMemcpy(result.data(), executor.s_.res, sizeof(result),
                    hipMemcpyDeviceToHost));
    std::array<double, 4> moe;
    double dot = 0;
    for (unsigned h = 0; h < 4; ++h) {
      moe[h] = .25 * down_float[h] + .75 * down_float[4 + h] + .5 * shared[h];
      dot += .5 * moe[h];
    }
    for (unsigned c = 0; c < 4; ++c)
      for (unsigned h = 0; h < 4; ++h) {
        const double expected = residual[c * 4 + h] +
                                attention[h] * old_weights[c] + moe[h] -
                                .75 * .5 * dot;
        if (!std::isfinite(result[c * 4 + h]) ||
            std::abs(result[c * 4 + h] - expected) > 2e-6)
          throw std::runtime_error(
              "deferred attention/MoE projection order mismatch");
      }
  }
};
}  // namespace gufo::models::qwen38_flash_next::rocm
int main() {
  for (unsigned hidden : {3u, 255u, 256u, 257u, 2560u}) {
    std::vector<float> direction(hidden);
    double norm = 0;
    for (unsigned j = 0; j < hidden; ++j) {
      direction[j] = std::sin(float(j + 1));
      norm += double(direction[j]) * direction[j];
    }
    for (auto& x : direction)
      x /= std::sqrt(norm);
    float* d = nullptr;
    Check(hipMalloc(&d, hidden * sizeof(float)));
    Check(hipMemcpy(d, direction.data(), hidden * sizeof(float),
                    hipMemcpyHostToDevice));
    for (unsigned rows : {1u, 8u, 9u, 33u}) {
      std::vector<float> input(rows * hidden), result(input.size()),
          expected(input.size());
      for (unsigned i = 0; i < input.size(); ++i)
        input[i] = std::cos(float(i) * .13f) * .01f;
      float* y = nullptr;
      Check(hipMalloc(&y, input.size() * sizeof(float)));
      for (float weight : {0.f, .5f, 1.f}) {
        Check(hipMemcpy(y, input.data(), input.size() * sizeof(float),
                        hipMemcpyHostToDevice));
        for (unsigned r = 0; r < rows; ++r) {
          double dot = 0;
          for (unsigned h = 0; h < hidden; ++h)
            dot += double(input[r * hidden + h]) * direction[h];
          for (unsigned h = 0; h < hidden; ++h)
            expected[r * hidden + h] = float(
                input[r * hidden + h] - double(weight) * dot * direction[h]);
        }
        q::AbliterateRows(y, d, weight, rows, hidden, nullptr);
        Check(hipMemcpy(result.data(), y, result.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
        for (unsigned i = 0; i < result.size(); ++i)
          if (!std::isfinite(result[i]) ||
              std::abs(result[i] - expected[i]) > 2e-7f ||
              (weight == 0 && result[i] != input[i]))
            throw std::runtime_error("projection mismatch");
      }
      Check(hipFree(y));
    }
    Check(hipFree(d));
  }
  q::ExecutorTestAccess::DeferredMoe();
  // Mixed sessions: OFF rows must retain their original bytes.
  float* y = nullptr;
  float* d = nullptr;
  Check(hipMalloc(&y, 12 * sizeof(float)));
  Check(hipMalloc(&d, 4 * sizeof(float)));
  std::array<float, 12> mixed;
  mixed.fill(1.f);
  const std::array<float, 4> direction{.5f, .5f, .5f, .5f};
  Check(hipMemcpy(y, mixed.data(), sizeof(mixed), hipMemcpyHostToDevice));
  Check(
      hipMemcpy(d, direction.data(), sizeof(direction), hipMemcpyHostToDevice));
  const std::array<q::AbliterationSpan, 3> spans{
      {{1, 0.f}, {2, 1.f}, {3, .5f}}};
  q::AbliterateBatchRows(y, d, spans, 3, 4, nullptr);
  Check(hipMemcpy(mixed.data(), y, sizeof(mixed), hipMemcpyDeviceToHost));
  for (unsigned i = 0; i < 12; ++i)
    if (mixed[i] != (i < 4 ? 1.f : i < 8 ? 0.f : .5f))
      throw std::runtime_error("mixed row mask mismatch");
  Check(hipFree(y));
  Check(hipFree(d));
  std::cout << "FP64 formula, deferred HC, mixed rows, tails, OFF preservation "
               "PASS\n";
}
