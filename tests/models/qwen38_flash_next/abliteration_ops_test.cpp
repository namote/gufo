#include <hip/hip_fp16.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/models/qwen/abliteration.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/runtime/executor.hpp"
#include "tests/models/qwen/support/abliteration_subspace.hpp"
namespace q = gufo::models::qwen38_flash_next::rocm;
void Check(hipError_t x) {
  if (x != hipSuccess)
    throw std::runtime_error(hipGetErrorString(x));
}
// One layer's device settings (AbliterationSettings::DeviceWords), written
// independently: direction row bits, attention strength, mlp strength.
std::vector<float> Settings(std::uint32_t row, float attention, float mlp) {
  return {std::bit_cast<float>(row), attention, mlp};
}
float* Upload(const std::vector<float>& values) {
  float* device = nullptr;
  Check(hipMalloc(&device, values.size() * sizeof(float)));
  Check(hipMemcpy(device, values.data(), values.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  return device;
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
    executor.abliteration_directions_ = alloc({.5f, .5f, .5f, .5f});
    // One layer: direction row 0, attention strength .75.
    const float* settings = alloc(Settings(0, .75f, 0.f));
    executor.moe_pending_ = true;
    executor.res_deferred_ = true;
    executor.Combine(executor.s_.res, nullptr, 1, nullptr, false,
                     {settings, 0, false});
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

// Two layers with residual strengths (1 + 3 x layers words, written
// independently): direction row bits, attention[2], mlp[2], residual[2].
std::vector<float> ResidualSettings(std::uint32_t row,
                                    std::array<float, 2> residual) {
  return {std::bit_cast<float>(row), 5, 5, 6, 6, residual[0], residual[1]};
}

// The post-layer HC residual ([tokens][4 streams][hidden]): every stream of
// an ON token is projected with its session's residual strength (FP64
// reference), the stream mean loses its component at strength 1, OFF and
// zero-strength tokens keep their bytes, and a captured graph follows
// in-place updates.
void ResidualStreams() {
  constexpr unsigned kStreams = 4;
  for (unsigned hidden : {3u, 257u, 2560u}) {
    // Rows 0 and 1 are two sessions' directions.
    std::vector<float> directions(2 * hidden);
    for (unsigned r = 0; r < 2; ++r) {
      double norm = 0;
      for (unsigned j = 0; j < hidden; ++j) {
        directions[r * hidden + j] = std::sin(float(j + 1) * (r + 1) * .7f);
        norm += double(directions[r * hidden + j]) * directions[r * hidden + j];
      }
      for (unsigned j = 0; j < hidden; ++j)
        directions[r * hidden + j] /= std::sqrt(norm);
    }
    // Tokens: 2 of session A (row 1, strength 1), 1 OFF, 3 of session B
    // (row 0, strength 2.5), 1 of session C (layer-1 strength 0), 1 past
    // every span. Layer 1 is read; writer strengths must not apply.
    const std::vector<double> token_weight{1, 1, 0, 2.5, 2.5, 2.5, 0, 0};
    const std::vector<unsigned> token_row{1, 1, 0, 0, 0, 0, 0, 0};
    const unsigned tokens = token_weight.size();
    std::vector<float> input(std::size_t{tokens} * kStreams * hidden);
    for (unsigned i = 0; i < input.size(); ++i)
      input[i] = std::cos(float(i) * .37f) +
                 .5f * directions[hidden + i % hidden];  // a large component
    float* d = Upload(directions);
    float* a = Upload(ResidualSettings(1, {3.f, 1.f}));
    float* b = Upload(ResidualSettings(0, {0.f, 2.5f}));
    float* c = Upload(ResidualSettings(1, {4.f, 0.f}));
    float* res = Upload(input);
    const std::array<q::AbliterationSpan, 4> spans{
        {{2, a}, {3, nullptr}, {6, b}, {7, c}}};
    q::AbliterateResidualBatchRows(res, d, spans, 1, 2, tokens, hidden,
                                   kStreams, nullptr);
    std::vector<float> result(input.size());
    Check(hipMemcpy(result.data(), res, result.size() * sizeof(float),
                    hipMemcpyDeviceToHost));
    for (unsigned t = 0; t < tokens; ++t) {
      const float* dir = directions.data() + token_row[t] * hidden;
      std::vector<double> mean(hidden);
      for (unsigned s = 0; s < kStreams; ++s) {
        const std::size_t row = (std::size_t{t} * kStreams + s) * hidden;
        double dot = 0;
        for (unsigned h = 0; h < hidden; ++h)
          dot += double(input[row + h]) * dir[h];
        for (unsigned h = 0; h < hidden; ++h) {
          const double expected =
              input[row + h] - token_weight[t] * dot * dir[h];
          if (!std::isfinite(result[row + h]) ||
              std::abs(result[row + h] - expected) >
                  2e-5 * (1 + std::abs(expected)) ||
              (token_weight[t] == 0 && result[row + h] != input[row + h]))
            throw std::runtime_error("HC residual projection mismatch");
          mean[h] += result[row + h] / kStreams;
        }
      }
      if (token_weight[t] == 1) {
        double dot = 0;
        for (unsigned h = 0; h < hidden; ++h)
          dot += mean[h] * dir[h];
        if (std::abs(dot) > 1e-4)
          throw std::runtime_error(
              "strength 1 left the stream mean's component");
      }
    }
    // No residual session launches nothing (a null direction would fault).
    const std::array<q::AbliterationSpan, 1> off{{{tokens, nullptr}}};
    q::AbliterateResidualBatchRows(res, nullptr, off, 1, 2, tokens, hidden,
                                   kStreams, nullptr);
    Check(hipDeviceSynchronize());
    std::vector<float> after(result.size());
    Check(hipMemcpy(after.data(), res, after.size() * sizeof(float),
                    hipMemcpyDeviceToHost));
    if (after != result)
      throw std::runtime_error("OFF residual launch changed rows");
    // Graph replay: one session's single token, strength updated in place.
    hipStream_t stream = nullptr;
    Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    Check(hipMemcpy(c, ResidualSettings(1, {0.f, 0.f}).data(),
                    7 * sizeof(float), hipMemcpyHostToDevice));
    const q::AbliterationSpan single{1, c};
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    Check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    q::AbliterateResidualBatchRows(res, d, std::span(&single, 1), 1, 2, 1,
                                   hidden, kStreams, stream);
    Check(hipStreamEndCapture(stream, &graph));
    Check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    const auto run = [&](float strength) {
      Check(hipMemcpy(res, input.data(), kStreams * hidden * sizeof(float),
                      hipMemcpyHostToDevice));
      Check(hipMemcpy(c, ResidualSettings(1, {0.f, strength}).data(),
                      7 * sizeof(float), hipMemcpyHostToDevice));
      Check(hipGraphLaunch(exec, stream));
      Check(hipStreamSynchronize(stream));
      std::vector<float> rows(kStreams * hidden);
      Check(hipMemcpy(rows.data(), res, rows.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
      return rows;
    };
    const auto unchanged = run(0.f);
    if (!std::equal(unchanged.begin(), unchanged.end(), input.begin()))
      throw std::runtime_error("zero residual strength changed a row");
    const auto projected = run(1.f);
    for (unsigned s = 0; s < kStreams; ++s) {
      double dot = 0;
      for (unsigned h = 0; h < hidden; ++h)
        dot += double(projected[s * hidden + h]) * directions[hidden + h];
      if (std::abs(dot) > 1e-4)
        throw std::runtime_error("graph ignored an in-place residual update");
    }
    Check(hipGraphExecDestroy(exec));
    Check(hipGraphDestroy(graph));
    Check(hipStreamDestroy(stream));
    for (float* p : {d, a, b, c, res})
      Check(hipFree(p));
  }
}

namespace sub = gufo::tests::qwen_abliteration;

// Subspace projection of the HC residual and the writer rows: each of the
// four streams of a token is its own row (projected and, with norm
// preservation, rescaled to its own norm), sessions mix their first row,
// count, norm preservation and strength, and one direction without norm
// preservation is the D16 kernel bit for bit.
void SubspaceStreams() {
  constexpr unsigned kStreams = 4, kRank = 8;
  std::mt19937 rng(38);
  for (unsigned hidden : {13u, 257u, 2560u})
    for (unsigned k : {1u, 2u, 5u, 8u}) {
      // Two rank-8 blocks; sessions start at row 0 or 8.
      auto directions = sub::OrthonormalRows(kRank, hidden, rng);
      const auto second = sub::OrthonormalRows(kRank, hidden, rng);
      directions.insert(directions.end(), second.begin(), second.end());
      struct Token {
        unsigned first, count;
        double weight;
        bool preserve;
      };
      const unsigned other_k = k == kRank ? 2 : k + 1;
      // Two tokens of session A, one OFF, two of session B, one of a
      // zero-strength session with norm preservation, one past every span.
      const std::vector<Token> tokens{
          {8, k, 1, false},       {8, k, 1, false},       {0, 1, 0, false},
          {0, other_k, .5, true}, {0, other_k, .5, true}, {8, k, 0, true},
          {0, 1, 0, false}};
      // Two layers; layer 1's residual strength (word 6), writer strengths
      // that must not apply.
      const auto words = [](unsigned row, unsigned count, bool preserve,
                            float residual) {
        return std::vector<float>{
            sub::Word0(row, count, preserve), 5, 5, 6, 6, 9, residual};
      };
      float* a = Upload(words(8, k, false, 1));
      float* b = Upload(words(0, other_k, true, .5f));
      float* c = Upload(words(8, k, true, 0));
      const std::array<q::AbliterationSpan, 4> spans{
          {{2, a}, {3, nullptr}, {5, b}, {6, c}}};
      std::normal_distribution<float> normal(0.f, 2.f);
      std::vector<float> input(tokens.size() * kStreams * hidden);
      for (auto& x : input)
        x = normal(rng);
      float* res = Upload(input);
      float* d = Upload(directions);
      q::AbliterateResidualBatchRows(res, d, spans, 1, 2, tokens.size(), hidden,
                                     kStreams, nullptr);
      Check(hipDeviceSynchronize());
      std::vector<float> output(input.size());
      Check(hipMemcpy(output.data(), res, output.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
      for (unsigned t = 0; t < tokens.size(); ++t)
        for (unsigned s = 0; s < kStreams; ++s) {
          const std::size_t offset = (std::size_t{t} * kStreams + s) * hidden;
          const float* y = input.data() + offset;
          const float* got = output.data() + offset;
          if (tokens[t].weight == 0) {
            if (!std::equal(got, got + hidden, y))
              throw std::runtime_error("OFF or zero-strength stream changed");
            continue;
          }
          const auto expected = sub::ReferenceRow(
              y, directions.data() + std::size_t{tokens[t].first} * hidden,
              tokens[t].count, tokens[t].weight, tokens[t].preserve, hidden);
          double before = 0, after = 0;
          for (unsigned i = 0; i < hidden; ++i) {
            if (std::abs(got[i] - expected[i]) >
                5e-5 * (1 + std::abs(expected[i])))
              throw std::runtime_error("subspace stream differs from FP64");
            before += double{y[i]} * y[i];
            after += double{got[i]} * got[i];
          }
          if (tokens[t].preserve &&
              std::abs(std::sqrt(after) - std::sqrt(before)) >
                  3e-5 * std::sqrt(before))
            throw std::runtime_error("stream norm not preserved");
        }
      // One direction, no norm preservation: bit-identical to D16 for the
      // residual streams and the writer rows.
      float* single = Upload(words(8, 1, false, 1.25f));
      if (std::bit_cast<std::uint32_t>(words(8, 1, false, 1)[0]) != 8)
        throw std::runtime_error("single-direction word 0 is not the row");
      const unsigned rows = 3 * kStreams;
      const std::vector<float> slice(
          input.begin(), input.begin() + std::size_t{rows} * hidden);
      for (const bool residual : {true, false}) {
        float* current = Upload(slice);
        float* reference = Upload(slice);
        if (residual) {
          const std::array<q::AbliterationSpan, 1> all{{{3, single}}};
          q::AbliterateResidualBatchRows(current, d, all, 1, 2, 3, hidden,
                                         kStreams, nullptr);
        } else {
          q::AbliterateRows(current, d, single, 1, false, 2, rows, hidden,
                            nullptr);
        }
        // Word 6 is layer 1's residual strength, word 2 its attention one.
        sub::D16AbliterateRowsUnbounded<<<rows, 256>>>(
            reference, d, single, residual ? 6 : 2, hidden);
        Check(hipGetLastError());
        Check(hipDeviceSynchronize());
        std::vector<float> x(slice.size()), y(slice.size());
        Check(hipMemcpy(x.data(), current, x.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
        Check(hipMemcpy(y.data(), reference, y.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
        for (std::size_t i = 0; i < x.size(); ++i)
          if (std::bit_cast<std::uint32_t>(x[i]) !=
              std::bit_cast<std::uint32_t>(y[i]))
            throw std::runtime_error("single direction differs from D16");
        Check(hipFree(current));
        Check(hipFree(reference));
      }
      for (float* p : {a, b, c, res, d, single})
        Check(hipFree(p));
    }
}

// Dynamic-abliteration detector scores (R2): the HC-stream mean's dot with
// each listed detector direction, into its own column of each row, against
// an FP64 reference; unlisted columns stay untouched.
void DetectorScores() {
  for (unsigned hidden : {3u, 257u, 2560u}) {
    constexpr unsigned kStreams = 4, kRows = 5, kStride = 8;
    std::vector<float> directions(3 * hidden);
    for (unsigned d = 0; d < 3; ++d) {
      double norm = 0;
      for (unsigned j = 0; j < hidden; ++j) {
        directions[d * hidden + j] = std::sin(float((d + 1) * (j + 1)));
        norm += double(directions[d * hidden + j]) * directions[d * hidden + j];
      }
      for (unsigned j = 0; j < hidden; ++j)
        directions[d * hidden + j] /= std::sqrt(norm);
    }
    std::vector<float> res(kRows * kStreams * hidden);
    for (unsigned i = 0; i < res.size(); ++i)
      res[i] = std::cos(float(i) * .37f) * (1.f + float(i % 7));
    float* dev_d = Upload(directions);
    float* dev_res = Upload(res);
    float* dev_scores = Upload(std::vector<float>(kRows * kStride, -7.f));
    // Detectors 2 and 0 into columns 2 and 0; column 1 stays -7.
    const std::array<std::uint32_t, 2> list{2, 0};
    q::AbliterationDetectorScores(dev_res, dev_d, list, list, dev_scores,
                                  kStride, kRows, hidden, kStreams, nullptr);
    std::vector<float> scores(kRows * kStride);
    Check(hipMemcpy(scores.data(), dev_scores, scores.size() * sizeof(float),
                    hipMemcpyDeviceToHost));
    for (unsigned r = 0; r < kRows; ++r)
      for (unsigned d = 0; d < 3; ++d) {
        // FP32 accumulation: bound the error by the terms' magnitude.
        double mean_dot = 0, magnitude = 0;
        for (unsigned s = 0; s < kStreams; ++s)
          for (unsigned j = 0; j < hidden; ++j) {
            const double term = double(res[(r * kStreams + s) * hidden + j]) *
                                directions[d * hidden + j];
            mean_dot += term;
            magnitude += std::abs(term);
          }
        mean_dot /= kStreams;
        const float got = scores[r * kStride + d];
        if (d == 1 ? got != -7.f
                   : !(std::abs(got - mean_dot) <= 1e-6 * (1 + magnitude)))
          throw std::runtime_error("detector score mismatch");
      }
    Check(hipFree(dev_d));
    Check(hipFree(dev_res));
    Check(hipFree(dev_scores));
  }
}

// A combination's weighted block (kAbliterationWeightedBit): two
// orthonormal rows with their own strengths at the site, against FP64.
void WeightedRows() {
  constexpr unsigned kHidden = 257, kRows = 3;
  std::vector<float> directions(3 * kHidden, 0.f);
  directions[1 * kHidden + 0] = 1.f;  // row 1: e0
  directions[2 * kHidden + 1] = 1.f;  // row 2: e1
  // One layer: word 0 then 3 sites x 8 weights; attention site weights
  // 2 (row 1) and 0.5 (row 2).
  std::vector<float> settings(1 + 3 * 8, 0.f);
  settings[0] = std::bit_cast<float>(1U | 1U << 24 | (1U << 29));
  settings[1] = 2.f;
  settings[2] = .5f;
  std::vector<float> input(kRows * kHidden);
  for (unsigned i = 0; i < input.size(); ++i)
    input[i] = std::cos(float(i) * .29f);
  float* d = Upload(directions);
  float* w = Upload(settings);
  float* y = Upload(input);
  q::AbliterateRows(y, d, w, 0, false, 1, kRows, kHidden, nullptr);
  std::vector<float> result(input.size());
  Check(hipMemcpy(result.data(), y, result.size() * sizeof(float),
                  hipMemcpyDeviceToHost));
  for (unsigned r = 0; r < kRows; ++r)
    for (unsigned h = 0; h < kHidden; ++h) {
      const double x = input[r * kHidden + h];
      const double want = h == 0 ? x - 2.0 * x : h == 1 ? x - .5 * x : x;
      if (std::abs(result[r * kHidden + h] - want) > 1e-5)
        throw std::runtime_error("weighted block mismatch");
    }
  Check(hipFree(d));
  Check(hipFree(w));
  Check(hipFree(y));
}

int main() {
  DetectorScores();
  WeightedRows();
  for (unsigned hidden : {3u, 255u, 256u, 257u, 2560u}) {
    // Row 1 is the session's direction; row 0 must never be read.
    std::vector<float> directions(2 * hidden, 1e30f);
    double norm = 0;
    for (unsigned j = 0; j < hidden; ++j) {
      directions[hidden + j] = std::sin(float(j + 1));
      norm += double(directions[hidden + j]) * directions[hidden + j];
    }
    for (unsigned j = 0; j < hidden; ++j)
      directions[hidden + j] /= std::sqrt(norm);
    const float* direction = directions.data() + hidden;
    float* d = Upload(directions);
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
        // The attention strength of layer 0; the mlp strength differs.
        float* settings = Upload(Settings(1, weight, 3.f));
        q::AbliterateRows(y, d, settings, 0, false, 1, rows, hidden, nullptr);
        Check(hipMemcpy(result.data(), y, result.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
        Check(hipFree(settings));
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
  // Mixed sessions with their own directions and strengths: OFF rows must
  // retain their original bytes.
  float* y = nullptr;
  Check(hipMalloc(&y, 12 * sizeof(float)));
  std::array<float, 12> mixed;
  mixed.fill(1.f);
  float* d = Upload({.5f, .5f, .5f, .5f, 1.f, 0.f, 0.f, 0.f});
  float* a = Upload(Settings(0, 1.f, 0.f));
  float* b = Upload(Settings(1, .5f, 0.f));
  Check(hipMemcpy(y, mixed.data(), sizeof(mixed), hipMemcpyHostToDevice));
  const std::array<q::AbliterationSpan, 3> spans{
      {{1, nullptr}, {2, a}, {3, b}}};
  q::AbliterateBatchRows(y, d, spans, 0, false, 1, 3, 4, nullptr);
  // Their mlp strengths are zero: nothing changes.
  q::AbliterateBatchRows(y, d, spans, 0, true, 1, 3, 4, nullptr);
  Check(hipMemcpy(mixed.data(), y, sizeof(mixed), hipMemcpyDeviceToHost));
  const std::array<float, 12> want{1, 1, 1, 1, 0, 0, 0, 0, .5f, 1, 1, 1};
  if (mixed != want)
    throw std::runtime_error("mixed row settings mismatch");
  // A captured graph reads the settings at launch: an in-place strength
  // update changes its result without recapture.
  hipStream_t stream = nullptr;
  Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  mixed.fill(1.f);
  Check(hipMemcpy(y, mixed.data(), sizeof(mixed), hipMemcpyHostToDevice));
  Check(hipMemcpy(a, Settings(0, 0.f, 0.f).data(), 3 * sizeof(float),
                  hipMemcpyHostToDevice));
  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  Check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
  q::AbliterateRows(y, d, a, 0, false, 1, 1, 4, stream);
  Check(hipStreamEndCapture(stream, &graph));
  Check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  Check(hipGraphLaunch(exec, stream));
  Check(hipStreamSynchronize(stream));
  Check(hipMemcpy(mixed.data(), y, sizeof(mixed), hipMemcpyDeviceToHost));
  if (mixed[0] != 1.f)
    throw std::runtime_error("zero strength changed a row");
  Check(hipMemcpy(a, Settings(1, 1.f, 0.f).data(), 3 * sizeof(float),
                  hipMemcpyHostToDevice));
  Check(hipGraphLaunch(exec, stream));
  Check(hipStreamSynchronize(stream));
  Check(hipMemcpy(mixed.data(), y, sizeof(mixed), hipMemcpyDeviceToHost));
  if (mixed[0] != 0.f || mixed[1] != 1.f)
    throw std::runtime_error("graph ignored an in-place settings update");
  Check(hipGraphExecDestroy(exec));
  Check(hipGraphDestroy(graph));
  Check(hipStreamDestroy(stream));
  for (float* p : {y, d, a, b})
    Check(hipFree(p));
  ResidualStreams();
  SubspaceStreams();
  std::cout << "FP64 formula, deferred HC, mixed rows, tails, OFF "
               "preservation, in-place graph settings, HC residual streams, "
               "subspace and norm preservation, D16 bit identity PASS\n";
}
