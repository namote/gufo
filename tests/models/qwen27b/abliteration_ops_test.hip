// GPU checks of the Qwen3.8-27B runtime projection (writer-before-residual):
// an independent FP64 formula, unaligned widths, mixed ON/OFF session rows,
// rows past the last span, a zero weight launching nothing, and the
// production order (project the writer output, then the residual add), which
// must leave the accumulated residual itself unprojected.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/models/qwen/hip/ops/abliteration.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"

namespace {
namespace hip = gufo::hip;

void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}
void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct Device {
  float* data{nullptr};
  explicit Device(const std::vector<float>& values) {
    Check(hipMalloc(&data, values.size() * sizeof(float)));
    Check(hipMemcpy(data, values.data(), values.size() * sizeof(float),
                    hipMemcpyHostToDevice));
  }
  ~Device() { (void)hipFree(data); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  std::vector<float> Read(std::size_t count) const {
    std::vector<float> values(count);
    Check(hipMemcpy(values.data(), data, count * sizeof(float),
                    hipMemcpyDeviceToHost));
    return values;
  }
};

std::vector<float> UnitDirection(std::uint32_t hidden, std::mt19937& rng) {
  std::normal_distribution<double> normal;
  std::vector<double> v(hidden);
  double norm = 0;
  for (auto& x : v) {
    x = normal(rng);
    norm += x * x;
  }
  std::vector<float> result(hidden);
  for (std::uint32_t i = 0; i < hidden; ++i)
    result[i] = static_cast<float>(v[i] / std::sqrt(norm));
  return result;
}

// y - w * d * dot(d, y) in FP64.
std::vector<double> Reference(const std::vector<float>& rows,
                              const std::vector<float>& direction,
                              const std::vector<float>& row_weights,
                              std::uint32_t hidden) {
  std::vector<double> result(rows.begin(), rows.end());
  for (std::size_t r = 0; r < row_weights.size(); ++r) {
    double dot = 0;
    for (std::uint32_t i = 0; i < hidden; ++i)
      dot += double{direction[i]} * rows[r * hidden + i];
    for (std::uint32_t i = 0; i < hidden; ++i)
      result[r * hidden + i] -= row_weights[r] * direction[i] * dot;
  }
  return result;
}

void CheckClose(const std::vector<float>& actual,
                const std::vector<double>& expected, const char* message) {
  for (std::size_t i = 0; i < actual.size(); ++i)
    Require(std::abs(actual[i] - expected[i]) <=
                2e-5 * (1 + std::abs(expected[i])),
            message);
}

void Formula(std::uint32_t hidden, std::mt19937& rng) {
  const auto direction = UnitDirection(hidden, rng);
  // Three sessions: two rows ON, one row OFF, three rows at another weight,
  // then two rows past every span (left unchanged).
  const std::vector<hip::QwenAbliterationSpan> spans{
      {2, 1.0F}, {3, 0.0F}, {6, 0.5F}};
  const std::vector<float> row_weights{1, 1, 0, .5F, .5F, .5F, 0, 0};
  const auto rows = static_cast<std::uint32_t>(row_weights.size());
  std::normal_distribution<float> normal(0.F, 3.F);
  std::vector<float> values(std::size_t{rows} * hidden);
  for (auto& x : values)
    x = normal(rng);
  const Device device_rows(values), device_direction(direction);
  hip::LaunchQwenAbliterateRows(device_rows.data, device_direction.data, spans,
                                rows, hidden, nullptr);
  Check(hipDeviceSynchronize());
  const auto actual = device_rows.Read(values.size());
  CheckClose(actual, Reference(values, direction, row_weights, hidden),
             "projection differs from the FP64 formula");
  for (std::size_t r = 0; r < rows; ++r)
    if (row_weights[r] == 0)
      for (std::uint32_t i = 0; i < hidden; ++i)
        Require(actual[r * hidden + i] == values[r * hidden + i],
                "an OFF row changed");
  // Full strength removes the direction from an ON row.
  double residual_dot = 0;
  for (std::uint32_t i = 0; i < hidden; ++i)
    residual_dot += double{direction[i]} * actual[i];
  Require(std::abs(residual_dot) < 1e-3, "full strength left the direction");
}

void ZeroWeightLaunchesNothing(std::mt19937& rng) {
  const std::uint32_t hidden = 64;
  const auto direction = UnitDirection(hidden, rng);
  std::vector<float> values(2 * hidden, 1.25F);
  const Device device_rows(values), device_direction(direction);
  // A null direction would fault if any launch read it.
  hip::LaunchQwenAbliterateRows(device_rows.data, nullptr, 0.0F, 2, hidden,
                                nullptr);
  const std::vector<hip::QwenAbliterationSpan> spans{{1, 0.F}, {2, 0.F}};
  hip::LaunchQwenAbliterateRows(device_rows.data, nullptr, spans, 2, hidden,
                                nullptr);
  Check(hipDeviceSynchronize());
  Require(device_rows.Read(values.size()) == values,
          "a zero weight changed rows");
  bool rejected = false;
  try {
    const std::vector<hip::QwenAbliterationSpan> nine(9, {1, 1.F});
    hip::LaunchQwenAbliterateRows(device_rows.data, device_direction.data,
                                  nine, 1, hidden, nullptr);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "more than eight spans accepted");
}

void ProductionOrder(std::mt19937& rng) {
  const std::uint32_t hidden = 5120, rows = 3;
  const auto direction = UnitDirection(hidden, rng);
  std::normal_distribution<float> normal(0.F, 1.F);
  std::vector<float> residual(std::size_t{rows} * hidden),
      writer(residual.size());
  for (auto& x : residual)
    x = normal(rng);
  for (auto& x : writer)
    x = normal(rng);
  const Device device_residual(residual), device_writer(writer),
      device_direction(direction);
  hip::LaunchQwenAbliterateRows(device_writer.data, device_direction.data,
                                0.75F, rows, hidden, nullptr);
  hip::LaunchBatchedResidualAdd(device_residual.data, device_writer.data,
                                device_residual.data, rows, hidden, nullptr);
  Check(hipDeviceSynchronize());
  const auto projected =
      Reference(writer, direction, std::vector<float>(rows, .75F), hidden);
  std::vector<double> expected(residual.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
    expected[i] = residual[i] + projected[i];
  CheckClose(device_residual.Read(residual.size()), expected,
             "writer projection then residual add differs");
}
}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cout << "SKIP: no HIP device\n";
    return 77;
  }
  try {
    std::mt19937 rng(27);
    for (const std::uint32_t hidden : {5120U, 13U, 257U})
      Formula(hidden, rng);
    ZeroWeightLaunchesNothing(rng);
    ProductionOrder(rng);
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "qwen27b abliteration ops PASS\n";
  return 0;
}
