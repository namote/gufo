// GPU checks of the Qwen3.8-27B runtime projection (writer-before-residual):
// an independent FP64 formula, unaligned widths, mixed ON/OFF session rows
// with their own directions and strengths, rows past the last span, OFF
// launching nothing, in-place settings updates under a captured graph, and
// the production order (project the writer output, then the residual add),
// which must leave the accumulated residual itself unprojected. The
// post-layer residual projection (D16) gets the same checks with its own
// strengths, and the full writer, add, residual order. Subspace projection
// (k in {1, 2, 5, 8} orthonormal directions) and norm preservation are checked
// against FP64 with mixed sessions, one direction against the D16 kernel bit
// for bit, and in-place count/norm updates under a captured graph.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/models/qwen/hip/ops/abliteration.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "tests/models/qwen/support/abliteration_subspace.hpp"

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

// One layer's device settings (AbliterationSettings::DeviceWords), written
// independently: direction row bits, attention strength, mlp strength.
std::vector<float> Settings(std::uint32_t row, float attention, float mlp) {
  return {std::bit_cast<float>(row), attention, mlp};
}
// Two layers: direction row bits, attention[2], mlp[2].
std::vector<float> Settings2(std::uint32_t row, std::array<float, 2> attention,
                             std::array<float, 2> mlp) {
  return {std::bit_cast<float>(row), attention[0], attention[1], mlp[0],
          mlp[1]};
}

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
  // Two direction rows; sessions pick their own.
  const auto direction = UnitDirection(hidden, rng);
  const auto other = UnitDirection(hidden, rng);
  std::vector<float> directions(other);
  directions.insert(directions.end(), direction.begin(), direction.end());
  // Four sessions over two layers, read at layer 1's MLP writer: two rows ON
  // with direction 1 at 1.0, one row OFF, three rows ON with direction 1 at
  // 0.5, one row ON with direction 0 at strength 0, then one row past every
  // span (left unchanged).
  const Device full(Settings2(1, {9, 9}, {7, 1.0F})),
      half(Settings2(1, {9, 9}, {7, .5F})), zero(Settings2(0, {9, 9}, {7, 0}));
  const std::vector<hip::QwenAbliterationSpan> spans{
      {2, full.data}, {3, nullptr}, {6, half.data}, {7, zero.data}};
  const std::vector<float> row_weights{1, 1, 0, .5F, .5F, .5F, 0, 0};
  const auto rows = static_cast<std::uint32_t>(row_weights.size());
  std::normal_distribution<float> normal(0.F, 3.F);
  std::vector<float> values(std::size_t{rows} * hidden);
  for (auto& x : values)
    x = normal(rng);
  const Device device_rows(values), device_directions(directions);
  hip::LaunchQwenAbliterateRows(device_rows.data, device_directions.data,
                                spans, 1, true, 2, rows, hidden, nullptr);
  Check(hipDeviceSynchronize());
  const auto actual = device_rows.Read(values.size());
  CheckClose(actual, Reference(values, direction, row_weights, hidden),
             "projection differs from the FP64 formula");
  for (std::size_t r = 0; r < rows; ++r)
    if (row_weights[r] == 0)
      for (std::uint32_t i = 0; i < hidden; ++i)
        Require(actual[r * hidden + i] == values[r * hidden + i],
                "an OFF or zero-strength row changed");
  // Full strength removes the direction from an ON row.
  double residual_dot = 0;
  for (std::uint32_t i = 0; i < hidden; ++i)
    residual_dot += double{direction[i]} * actual[i];
  Require(std::abs(residual_dot) < 1e-3, "full strength left the direction");
}

void OffLaunchesNothing(std::mt19937& rng) {
  const std::uint32_t hidden = 64;
  const auto direction = UnitDirection(hidden, rng);
  std::vector<float> values(2 * hidden, 1.25F);
  const Device device_rows(values), device_direction(direction);
  // A null direction would fault if any launch read it.
  hip::LaunchQwenAbliterateRows(device_rows.data, nullptr, nullptr, 0, false, 1,
                                2, hidden, nullptr);
  const std::vector<hip::QwenAbliterationSpan> spans{{1, nullptr},
                                                     {2, nullptr}};
  hip::LaunchQwenAbliterateRows(device_rows.data, nullptr, spans, 0, false, 1,
                                2, hidden, nullptr);
  Check(hipDeviceSynchronize());
  Require(device_rows.Read(values.size()) == values, "OFF changed rows");
  const Device settings(Settings(0, 1, 1));
  bool rejected = false;
  try {
    const std::vector<hip::QwenAbliterationSpan> nine(9, {1, settings.data});
    hip::LaunchQwenAbliterateRows(device_rows.data, device_direction.data,
                                  nine, 0, false, 1, 1, hidden, nullptr);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "more than eight spans accepted");
}

// A captured graph reads the settings at launch: in-place strength and
// direction updates change its result without recapture.
void GraphFollowsSettings(std::mt19937& rng) {
  const std::uint32_t hidden = 257;
  const auto first = UnitDirection(hidden, rng);
  const auto second = UnitDirection(hidden, rng);
  std::vector<float> directions(first);
  directions.insert(directions.end(), second.begin(), second.end());
  std::normal_distribution<float> normal(0.F, 1.F);
  std::vector<float> values(hidden);
  for (auto& x : values)
    x = normal(rng);
  const Device rows(values), device_directions(directions),
      settings(Settings(0, 0, 0));
  hipStream_t stream = nullptr;
  Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  Check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
  hip::LaunchQwenAbliterateRows(rows.data, device_directions.data,
                                settings.data, 0, false, 1, 1, hidden, stream);
  Check(hipStreamEndCapture(stream, &graph));
  Check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  const auto run = [&](const std::vector<float>& words) {
    Check(hipMemcpy(rows.data, values.data(), hidden * sizeof(float),
                    hipMemcpyHostToDevice));
    Check(hipMemcpy(settings.data, words.data(), words.size() * sizeof(float),
                    hipMemcpyHostToDevice));
    Check(hipGraphLaunch(exec, stream));
    Check(hipStreamSynchronize(stream));
    return rows.Read(hidden);
  };
  Require(run(Settings(0, 0, 0)) == values, "zero strength changed a row");
  CheckClose(run(Settings(0, .75F, 0)),
             Reference(values, first, {.75F}, hidden),
             "graph ignored an in-place strength update");
  CheckClose(run(Settings(1, .75F, 0)),
             Reference(values, second, {.75F}, hidden),
             "graph ignored an in-place direction update");
  Check(hipGraphExecDestroy(exec));
  Check(hipGraphDestroy(graph));
  Check(hipStreamDestroy(stream));
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
      device_direction(direction), settings(Settings(0, 0.75F, 0));
  hip::LaunchQwenAbliterateRows(device_writer.data, device_direction.data,
                                settings.data, 0, false, 1, rows, hidden,
                                nullptr);
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
// One layer with a residual strength: direction row bits, attention, mlp,
// residual (AbliterationSettings::DeviceWords, 1 + 3 x layers words).
std::vector<float> Settings3(std::uint32_t row, float attention, float mlp,
                             float residual) {
  return {std::bit_cast<float>(row), attention, mlp, residual};
}
// Two layers: direction row bits, attention[2], mlp[2], residual[2].
std::vector<float> Settings6(std::uint32_t row, std::array<float, 2> residual) {
  return {std::bit_cast<float>(row), 9, 9, 7, 7, residual[0], residual[1]};
}

// The post-layer residual projection reads only the residual strengths:
// mixed sessions, rows past the last span, zero strength an exact no-op.
void ResidualFormula(std::uint32_t hidden, std::mt19937& rng) {
  const auto direction = UnitDirection(hidden, rng);
  const auto other = UnitDirection(hidden, rng);
  std::vector<float> directions(other);
  directions.insert(directions.end(), direction.begin(), direction.end());
  // Read at layer 1: two rows at 1.0, one OFF row, two rows at 2.5, one row
  // whose layer-1 residual strength is 0 (layer 0's is not), one row past
  // every span. Writer strengths (9, 7) must not apply.
  const Device full(Settings6(1, {3, 1.0F})), over(Settings6(1, {0, 2.5F})),
      zero(Settings6(0, {4, 0}));
  const std::vector<hip::QwenAbliterationSpan> spans{
      {2, full.data}, {3, nullptr}, {5, over.data}, {6, zero.data}};
  const std::vector<float> row_weights{1, 1, 0, 2.5F, 2.5F, 0, 0};
  const auto rows = static_cast<std::uint32_t>(row_weights.size());
  std::normal_distribution<float> normal(0.F, 3.F);
  std::vector<float> values(std::size_t{rows} * hidden);
  for (auto& x : values)
    x = normal(rng);
  const Device device_rows(values), device_directions(directions);
  hip::LaunchQwenAbliterateResidualRows(device_rows.data,
                                        device_directions.data, spans, 1, 2,
                                        rows, hidden, nullptr);
  Check(hipDeviceSynchronize());
  const auto actual = device_rows.Read(values.size());
  CheckClose(actual, Reference(values, direction, row_weights, hidden),
             "residual projection differs from the FP64 formula");
  for (std::size_t r = 0; r < rows; ++r)
    if (row_weights[r] == 0)
      for (std::uint32_t i = 0; i < hidden; ++i)
        Require(actual[r * hidden + i] == values[r * hidden + i],
                "an OFF or zero-strength residual row changed");
  for (std::size_t r = 0; r < 2; ++r) {
    double dot = 0;
    for (std::uint32_t i = 0; i < hidden; ++i)
      dot += double{direction[i]} * actual[r * hidden + i];
    Require(std::abs(dot) < 1e-3, "residual strength 1 left the direction");
  }
  // No residual session: nothing launches (a null direction would fault).
  const std::vector<hip::QwenAbliterationSpan> off{{rows, nullptr}};
  hip::LaunchQwenAbliterateResidualRows(device_rows.data, nullptr, off, 1, 2,
                                        rows, hidden, nullptr);
  Check(hipDeviceSynchronize());
  Require(device_rows.Read(values.size()) == actual,
          "OFF residual launch changed rows");
}

// A captured residual projection follows in-place residual updates.
void ResidualGraphFollowsSettings(std::mt19937& rng) {
  const std::uint32_t hidden = 257;
  const auto first = UnitDirection(hidden, rng);
  const auto second = UnitDirection(hidden, rng);
  std::vector<float> directions(first);
  directions.insert(directions.end(), second.begin(), second.end());
  std::normal_distribution<float> normal(0.F, 1.F);
  std::vector<float> values(hidden);
  for (auto& x : values)
    x = normal(rng);
  const Device rows(values), device_directions(directions),
      settings(Settings3(0, 0, 0, 0));
  hipStream_t stream = nullptr;
  Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  Check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
  hip::LaunchQwenAbliterateResidualRows(rows.data, device_directions.data,
                                        settings.data, 0, 1, 1, hidden, stream);
  Check(hipStreamEndCapture(stream, &graph));
  Check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  const auto run = [&](const std::vector<float>& words) {
    Check(hipMemcpy(rows.data, values.data(), hidden * sizeof(float),
                    hipMemcpyHostToDevice));
    Check(hipMemcpy(settings.data, words.data(), words.size() * sizeof(float),
                    hipMemcpyHostToDevice));
    Check(hipGraphLaunch(exec, stream));
    Check(hipStreamSynchronize(stream));
    return rows.Read(hidden);
  };
  Require(run(Settings3(0, 2, 2, 0)) == values,
          "zero residual strength changed a row");
  CheckClose(run(Settings3(0, 0, 0, 1)), Reference(values, first, {1}, hidden),
             "graph ignored an in-place residual update");
  CheckClose(run(Settings3(1, 0, 0, .5F)),
             Reference(values, second, {.5F}, hidden),
             "graph ignored an in-place direction update");
  Check(hipGraphExecDestroy(exec));
  Check(hipGraphDestroy(graph));
  Check(hipStreamDestroy(stream));
}

// The D16 order: writer projections, the residual add, then the residual
// projection. Residual strength 1 removes the component however it entered
// (here mostly through the incoming residual, which writer projection can
// never reach).
void ResidualProductionOrder(std::mt19937& rng) {
  const std::uint32_t hidden = 5120, rows = 3;
  const auto direction = UnitDirection(hidden, rng);
  std::normal_distribution<float> normal(0.F, 1.F);
  std::vector<float> residual(std::size_t{rows} * hidden),
      writer(residual.size());
  for (std::size_t i = 0; i < residual.size(); ++i)
    residual[i] = normal(rng) + 4 * direction[i % hidden];
  for (auto& x : writer)
    x = normal(rng);
  const Device device_residual(residual), device_writer(writer),
      device_direction(direction), settings(Settings3(0, 0, 0.5F, 1));
  hip::LaunchQwenAbliterateRows(device_writer.data, device_direction.data,
                                settings.data, 0, true, 1, rows, hidden,
                                nullptr);
  hip::LaunchBatchedResidualAdd(device_residual.data, device_writer.data,
                                device_residual.data, rows, hidden, nullptr);
  hip::LaunchQwenAbliterateResidualRows(device_residual.data,
                                        device_direction.data, settings.data,
                                        0, 1, rows, hidden, nullptr);
  Check(hipDeviceSynchronize());
  const auto projected =
      Reference(writer, direction, std::vector<float>(rows, .5F), hidden);
  std::vector<float> added(residual.size());
  for (std::size_t i = 0; i < added.size(); ++i)
    added[i] = static_cast<float>(residual[i] + projected[i]);
  const auto actual = device_residual.Read(residual.size());
  CheckClose(actual,
             Reference(added, direction, std::vector<float>(rows, 1), hidden),
             "writer, residual add, residual projection differs");
  for (std::uint32_t r = 0; r < rows; ++r) {
    double dot = 0;
    for (std::uint32_t i = 0; i < hidden; ++i)
      dot += double{direction[i]} * actual[std::size_t{r} * hidden + i];
    Require(std::abs(dot) < 1e-3, "post-layer residual kept the direction");
  }
}
namespace sub = gufo::tests::qwen_abliteration;

// Two layers' settings with a packed word 0 and the writer/residual
// strengths of layer 1 (AbliterationSettings::DeviceWords: 1 + 3 x layers).
std::vector<float> PackedSettings(std::uint32_t row, std::uint32_t count,
                                  bool preserve, float mlp, float residual) {
  return {sub::Word0(row, count, preserve), 0, 0, 0, mlp, 0, residual};
}

void CheckRows(const std::vector<float>& actual,
               const std::vector<double>& expected, double tolerance,
               const char* message) {
  for (std::size_t i = 0; i < actual.size(); ++i)
    Require(std::abs(actual[i] - expected[i]) <=
                tolerance * (1 + std::abs(expected[i])),
            message);
}

// Subspace projection (k orthonormal rows of a rank-8 layout) against FP64
// for the MLP writer and the post-layer residual: mixed sessions with their
// own first row, count, norm preservation and strength, an OFF row, a
// zero-strength row (bit-identical even with norm preservation) and a row
// past every span.
void SubspaceFormula(std::uint32_t hidden, std::uint32_t k, bool residual,
                     std::mt19937& rng) {
  constexpr std::uint32_t kRank = 8;
  // Three blocks of eight rows: sessions start at row 8 or 16.
  std::vector<float> directions;
  for (int block = 0; block < 3; ++block) {
    const auto rows = sub::OrthonormalRows(kRank, hidden, rng);
    directions.insert(directions.end(), rows.begin(), rows.end());
  }
  const std::uint32_t other_k = k == kRank ? 3 : k + 1;
  const Device full(PackedSettings(8, k, false, 1, 1)),
      half_norm(PackedSettings(8, k, true, .5F, .5F)),
      strong(PackedSettings(16, other_k, false, 2, 2)),
      zero(PackedSettings(16, k, true, 0, 0));
  const std::vector<hip::QwenAbliterationSpan> spans{{2, full.data},
                                                     {3, nullptr},
                                                     {5, half_norm.data},
                                                     {6, strong.data},
                                                     {7, zero.data}};
  struct Row {
    std::uint32_t first, count;
    double weight;
    bool preserve;
  };
  const std::vector<Row> expected_rows{
      {8, k, 1, false},  {8, k, 1, false},        {0, 1, 0, false},
      {8, k, .5, true},  {8, k, .5, true},        {16, other_k, 2, false},
      {16, k, 0, true},  {0, 1, 0, false}};
  const auto rows = static_cast<std::uint32_t>(expected_rows.size());
  std::normal_distribution<float> normal(0.F, 3.F);
  std::vector<float> values(std::size_t{rows} * hidden);
  for (auto& x : values)
    x = normal(rng);
  // Give the ON rows a large component in their subspace.
  for (std::uint32_t r = 0; r < rows; ++r)
    for (std::uint32_t j = 0; j < expected_rows[r].count; ++j)
      for (std::uint32_t i = 0; i < hidden; ++i)
        values[std::size_t{r} * hidden + i] +=
            4 * directions[(std::size_t{expected_rows[r].first} + j) * hidden +
                           i];
  const Device device_rows(values), device_directions(directions);
  if (residual)
    hip::LaunchQwenAbliterateResidualRows(device_rows.data,
                                          device_directions.data, spans, 1, 2,
                                          rows, hidden, nullptr);
  else
    hip::LaunchQwenAbliterateRows(device_rows.data, device_directions.data,
                                  spans, 1, true, 2, rows, hidden, nullptr);
  Check(hipDeviceSynchronize());
  const auto actual = device_rows.Read(values.size());
  for (std::uint32_t r = 0; r < rows; ++r) {
    const auto& row = expected_rows[r];
    const auto* y = values.data() + std::size_t{r} * hidden;
    const std::vector<float> got(actual.begin() + std::size_t{r} * hidden,
                                 actual.begin() + std::size_t{r + 1} * hidden);
    if (row.weight == 0) {
      Require(std::equal(got.begin(), got.end(), y),
              "an OFF or zero-strength row changed");
      continue;
    }
    CheckRows(got,
              sub::ReferenceRow(y,
                                directions.data() +
                                    std::size_t{row.first} * hidden,
                                row.count, row.weight, row.preserve, hidden),
              5e-5, "subspace projection differs from the FP64 formula");
    double before = 0, after = 0;
    for (std::uint32_t i = 0; i < hidden; ++i) {
      before += double{y[i]} * y[i];
      after += double{got[i]} * got[i];
    }
    if (row.preserve)
      Require(std::abs(std::sqrt(after) - std::sqrt(before)) <=
                  3e-5 * std::sqrt(before),
              "norm preservation changed the row norm");
    // Strength 1 leaves no component in the subspace.
    if (row.weight == 1)
      for (std::uint32_t j = 0; j < row.count; ++j) {
        double dot = 0;
        for (std::uint32_t i = 0; i < hidden; ++i)
          dot += double{got[i]} *
                 directions[(std::size_t{row.first} + j) * hidden + i];
        Require(std::abs(dot) < 1e-3, "strength 1 left a subspace component");
      }
  }
}

// One direction without norm preservation is the D16 kernel, bit for bit,
// for the writer and residual entry points.
void SingleDirectionMatchesD16(std::uint32_t hidden, std::mt19937& rng) {
  const std::uint32_t rows = 5;
  auto directions = sub::OrthonormalRows(3, hidden, rng);
  std::normal_distribution<float> normal(0.F, 2.F);
  std::vector<float> values(std::size_t{rows} * hidden);
  for (auto& x : values)
    x = normal(rng);
  // Row 2 as the session's direction; strengths 0.75 (mlp), 1.25 (residual).
  const auto words = PackedSettings(2, 1, false, .75F, 1.25F);
  Require(std::bit_cast<std::uint32_t>(words[0]) == 2,
          "single-direction word 0 is not the bare row");
  const Device settings(words), device_directions(directions);
  for (const bool residual : {false, true}) {
    const Device current(values), reference(values);
    if (residual)
      hip::LaunchQwenAbliterateResidualRows(current.data,
                                            device_directions.data,
                                            settings.data, 1, 2, rows, hidden,
                                            nullptr);
    else
      hip::LaunchQwenAbliterateRows(current.data, device_directions.data,
                                    settings.data, 1, true, 2, rows, hidden,
                                    nullptr);
    sub::D16AbliterateRows<<<rows, 256>>>(reference.data,
                                          device_directions.data,
                                          settings.data, residual ? 6 : 4,
                                          hidden);
    Check(hipGetLastError());
    Check(hipDeviceSynchronize());
    const auto a = current.Read(values.size());
    const auto b = reference.Read(values.size());
    for (std::size_t i = 0; i < a.size(); ++i)
      Require(std::bit_cast<std::uint32_t>(a[i]) ==
                  std::bit_cast<std::uint32_t>(b[i]),
              "single direction differs from the D16 kernel");
  }
}

// A captured graph follows in-place changes of the direction count and norm
// preservation (word 0) as well as the strength.
void SubspaceGraphFollowsSettings(std::mt19937& rng) {
  const std::uint32_t hidden = 257;
  const auto directions = sub::OrthonormalRows(8, hidden, rng);
  std::normal_distribution<float> normal(0.F, 1.F);
  std::vector<float> values(hidden);
  for (auto& x : values)
    x = normal(rng);
  const Device rows(values), device_directions(directions),
      settings(PackedSettings(0, 1, false, 0, 0));
  hipStream_t stream = nullptr;
  Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  Check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
  hip::LaunchQwenAbliterateResidualRows(rows.data, device_directions.data,
                                        settings.data, 1, 2, 1, hidden,
                                        stream);
  Check(hipStreamEndCapture(stream, &graph));
  Check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  const auto run = [&](const std::vector<float>& words) {
    Check(hipMemcpy(rows.data, values.data(), hidden * sizeof(float),
                    hipMemcpyHostToDevice));
    Check(hipMemcpy(settings.data, words.data(), words.size() * sizeof(float),
                    hipMemcpyHostToDevice));
    Check(hipGraphLaunch(exec, stream));
    Check(hipStreamSynchronize(stream));
    return rows.Read(hidden);
  };
  Require(run(PackedSettings(0, 4, true, 0, 0)) == values,
          "zero strength changed a row");
  CheckRows(run(PackedSettings(0, 4, false, 0, 1)),
            sub::ReferenceRow(values.data(), directions.data(), 4, 1, false,
                              hidden),
            5e-5, "graph ignored an in-place direction count");
  CheckRows(run(PackedSettings(2, 6, true, 0, .5F)),
            sub::ReferenceRow(values.data(), directions.data() + 2 * hidden,
                              6, .5, true, hidden),
            5e-5, "graph ignored in-place norm preservation");
  Check(hipGraphExecDestroy(exec));
  Check(hipGraphDestroy(graph));
  Check(hipStreamDestroy(stream));
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
    OffLaunchesNothing(rng);
    GraphFollowsSettings(rng);
    ProductionOrder(rng);
    for (const std::uint32_t hidden : {5120U, 13U, 257U})
      ResidualFormula(hidden, rng);
    ResidualGraphFollowsSettings(rng);
    ResidualProductionOrder(rng);
    for (const std::uint32_t hidden : {5120U, 13U, 257U})
      for (const std::uint32_t k : {1U, 2U, 5U, 8U})
        for (const bool residual : {false, true})
          SubspaceFormula(hidden, k, residual, rng);
    for (const std::uint32_t hidden : {5120U, 13U, 257U})
      SingleDirectionMatchesD16(hidden, rng);
    SubspaceGraphFollowsSettings(rng);
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "qwen27b abliteration ops PASS\n";
  return 0;
}
