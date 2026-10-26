// KVarN fake quantization of FP16 attention KV rows for the K0 accuracy study
// (plan/gufo-upstream-port/KVARN.md). The algorithm follows the reference
// store path of huawei-csl/KVarN (Apache-2.0,
// vllm/model_executor/layers/quantization/kvarn/sinkhorn.py and
// vllm/v1/attention/ops/kvarn_store.py): a normalized Sylvester Hadamard
// rotation along head_dim, log-domain alternating column/row standard
// deviation balancing with the best-so-far state, then asymmetric per-row
// round-to-nearest-even. A key tile is [head_dim, group] (per-channel RTN), a
// value tile is [group, head_dim] (per-token RTN); the absorbed row scale and
// zero point and the other-axis scale are FP16, as the store writes them.
//
// The rotation is orthogonal and symmetric, so attention over rows written
// back as dequant(quant(x H)) H equals attention in the rotated frame. The
// study therefore needs no attention kernel change.
#ifndef GUFO_TESTS_MODELS_QWEN27B_KVARN_FAKE_QUANT_HPP_
#define GUFO_TESTS_MODELS_QWEN27B_KVARN_FAKE_QUANT_HPP_

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace gufo::testing::kvarn {

enum class Method : std::uint8_t {
  kFp16,         // rows unchanged: checks the capture and restore round trip
  kRtn,          // KIVI-style asymmetric RTN, no rotation or balancing
  kHadamardRtn,  // rotation, no balancing
  kKvarn,        // rotation and balancing
};

struct Config {
  std::string name;
  Method method{Method::kFp16};
  int key_bits{16};
  int value_bits{16};
  std::size_t group{128};
  int iterations{8};
};

// "fp16", or <method>_k<bits>v<bits>_g<group> with method rtn, hrtn or kvarn
// and 2, 4 or 8 bits, e.g. kvarn_k4v2_g128.
[[nodiscard]] inline std::optional<Config> ParseConfig(std::string_view name) {
  Config config{std::string(name)};
  if (name == "fp16")
    return config;
  const auto underscore = name.find('_');
  if (underscore == std::string_view::npos)
    return std::nullopt;
  const auto method = name.substr(0, underscore);
  if (method == "rtn")
    config.method = Method::kRtn;
  else if (method == "hrtn")
    config.method = Method::kHadamardRtn;
  else if (method == "kvarn")
    config.method = Method::kKvarn;
  else
    return std::nullopt;
  int group = 0;
  if (std::sscanf(std::string(name.substr(underscore + 1)).c_str(),
                  "k%dv%d_g%d", &config.key_bits, &config.value_bits,
                  &group) != 3)
    return std::nullopt;
  const auto bits_ok = [](int bits) {
    return bits == 2 || bits == 4 || bits == 8;
  };
  if (!bits_ok(config.key_bits) || !bits_ok(config.value_bits) ||
      (group != 64 && group != 128))
    return std::nullopt;
  config.group = static_cast<std::size_t>(group);
  return config;
}

[[nodiscard]] inline float HalfToFloat(std::uint16_t bits) noexcept {
  _Float16 value;
  std::memcpy(&value, &bits, sizeof(bits));
  return static_cast<float>(value);
}

[[nodiscard]] inline std::uint16_t FloatToHalf(float value) noexcept {
  const auto half = static_cast<_Float16>(value);
  std::uint16_t bits;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

[[nodiscard]] inline float RoundHalf(float value) noexcept {
  return static_cast<float>(static_cast<_Float16>(value));
}

// x <- H x with H the Sylvester Hadamard matrix normalized by 1/sqrt(n).
inline void Hadamard(std::span<float> x) {
  const auto n = x.size();
  for (std::size_t length = 1; length < n; length <<= 1)
    for (std::size_t i = 0; i < n; i += length << 1)
      for (std::size_t j = i; j < i + length; ++j) {
        const float a = x[j], b = x[j + length];
        x[j] = a + b;
        x[j + length] = a - b;
      }
  const float scale = 1.0F / std::sqrt(static_cast<float>(n));
  for (auto& value : x)
    value *= scale;
}

namespace detail {
// Correctly rounded float log and exp: computed in double, rounded once. The
// GPU quantizer rounds the same way; glibc's logf/expf miss the correctly
// rounded result for about 0.1 % of inputs, which moved a stored scale.
[[nodiscard]] inline float Log(float x) noexcept {
  return static_cast<float>(std::log(static_cast<double>(x)));
}
[[nodiscard]] inline float Exp(float x) noexcept {
  return static_cast<float>(std::exp(static_cast<double>(x)));
}

// Unbiased standard deviation of `count` values `stride` apart.
inline float Std(const float* values, std::size_t count, std::size_t stride) {
  double mean = 0;
  for (std::size_t i = 0; i < count; ++i)
    mean += values[i * stride];
  mean /= static_cast<double>(count);
  double sum = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const double d = values[i * stride] - mean;
    sum += d * d;
  }
  return static_cast<float>(std::sqrt(sum / static_cast<double>(count - 1)));
}

inline float Imbalance(const std::vector<float>& tile, std::size_t rows,
                       std::size_t cols) {
  float column_max = 0, column_min = INFINITY, row_max = 0, row_min = INFINITY;
  for (std::size_t c = 0; c < cols; ++c) {
    const float s = Std(tile.data() + c, rows, cols);
    column_max = std::max(column_max, s);
    column_min = std::min(column_min, s);
  }
  for (std::size_t r = 0; r < rows; ++r) {
    const float s = Std(tile.data() + r * cols, cols, 1);
    row_max = std::max(row_max, s);
    row_min = std::min(row_min, s);
  }
  return column_max / std::max(column_min, 1e-8F) +
         row_max / std::max(row_min, 1e-8F);
}
}  // namespace detail

// Log-domain balancing of a row-major [rows, cols] tile: tile becomes
// tile / column_scale / row_scale for the lowest-imbalance iteration.
inline void Balance(std::vector<float>& tile, std::size_t rows,
                    std::size_t cols, int iterations,
                    std::vector<float>& column_scale,
                    std::vector<float>& row_scale) {
  constexpr float kStdMin = 1e-3F, kStdMax = 1e3F, kLogMin = -0.3F,
                  kLogMax = 10.0F;
  const std::vector<float> source = tile;
  std::vector<float> log_column(cols, 0.0F), log_row(rows, 0.0F);
  column_scale.assign(cols, 1.0F);
  row_scale.assign(rows, 1.0F);
  float best = detail::Imbalance(tile, rows, cols);
  std::vector<float> current = source;
  std::vector<float> column(cols), row(rows);
  const auto apply = [&] {
    for (std::size_t c = 0; c < cols; ++c)
      column[c] = detail::Exp(log_column[c]);
    for (std::size_t r = 0; r < rows; ++r)
      row[r] = detail::Exp(log_row[r]);
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t c = 0; c < cols; ++c)
        current[r * cols + c] = source[r * cols + c] / column[c] / row[r];
  };
  for (int iteration = 0; iteration < iterations; ++iteration) {
    for (std::size_t c = 0; c < cols; ++c) {
      const float s = std::clamp(detail::Std(current.data() + c, rows, cols),
                                 kStdMin, kStdMax);
      log_column[c] =
          std::clamp(log_column[c] + detail::Log(s), kLogMin, kLogMax);
    }
    apply();
    for (std::size_t r = 0; r < rows; ++r) {
      const float s = std::clamp(
          detail::Std(current.data() + r * cols, cols, 1), kStdMin, kStdMax);
      log_row[r] = std::clamp(log_row[r] + detail::Log(s), kLogMin, kLogMax);
    }
    apply();
    const float imbalance = detail::Imbalance(current, rows, cols);
    if (imbalance <= best) {
      best = imbalance;
      for (std::size_t c = 0; c < cols; ++c)
        column_scale[c] = detail::Exp(log_column[c]);
      for (std::size_t r = 0; r < rows; ++r)
        row_scale[r] = detail::Exp(log_row[r]);
    }
  }
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t c = 0; c < cols; ++c)
      tile[r * cols + c] =
          source[r * cols + c] / column_scale[c] / row_scale[r];
}

// Stored form of one quantized row-major [rows, cols] tile: codes, the FP16
// row scale and zero with the row's balancing scale absorbed, and the FP16
// column scale. The value is (code * scale + zero) * column.
struct QuantizedTile {
  std::vector<std::uint8_t> codes;         // [rows][cols]
  std::vector<std::uint16_t> scale, zero;  // [rows]
  std::vector<std::uint16_t> column;       // [cols]
};

// Per-row asymmetric RTN after optional balancing, in the frame the tile is
// given; `tile` is left balanced.
inline QuantizedTile QuantizeTileCodes(std::vector<float>& tile,
                                       std::size_t rows, std::size_t cols,
                                       int bits, bool balance, int iterations) {
  std::vector<float> column_scale(cols, 1.0F), row_scale(rows, 1.0F);
  if (balance)
    Balance(tile, rows, cols, iterations, column_scale, row_scale);
  const float qmax = static_cast<float>((1 << bits) - 1);
  QuantizedTile out;
  out.codes.resize(rows * cols);
  for (std::size_t r = 0; r < rows; ++r) {
    const float* row = tile.data() + r * cols;
    const auto [lo, hi] = std::minmax_element(row, row + cols);
    const float low = *lo;
    const float scale = std::max((*hi - low) / qmax, 1e-10F);
    out.scale.push_back(FloatToHalf(row_scale[r] * scale));
    out.zero.push_back(FloatToHalf(row_scale[r] * low));
    for (std::size_t c = 0; c < cols; ++c)
      out.codes[r * cols + c] = static_cast<std::uint8_t>(
          std::clamp(std::nearbyint((row[c] - low) / scale), 0.0F, qmax));
  }
  for (std::size_t c = 0; c < cols; ++c)
    out.column.push_back(FloatToHalf(column_scale[c]));
  return out;
}

// Quantizes and dequantizes a row-major [rows, cols] tile in place, in the
// frame it is given: per-row asymmetric RTN after optional balancing, with
// the stored FP16 scales.
inline void QuantizeTile(std::vector<float>& tile, std::size_t rows,
                         std::size_t cols, int bits, bool balance,
                         int iterations) {
  const auto q = QuantizeTileCodes(tile, rows, cols, bits, balance, iterations);
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t c = 0; c < cols; ++c)
      tile[r * cols + c] =
          (static_cast<float>(q.codes[r * cols + c]) * HalfToFloat(q.scale[r]) +
           HalfToFloat(q.zero[r])) *
          HalfToFloat(q.column[c]);
}

struct Error {
  double squared{0};    // sum of squared differences, original frame
  double reference{0};  // sum of squared reference values
};

// Fake-quantizes tokens [group, end_group * group) of FP16 rows laid out
// [token][head][dim]: every complete group after the first (the FP16 sink)
// before `end_group`, starting at `first_group` (at least 1). Rows past the
// last complete group are the FP16 tail and stay unchanged.
inline Error FakeQuantize(std::span<std::uint16_t> rows, std::size_t heads,
                          std::size_t dim, bool key, const Config& config,
                          std::size_t first_group, std::size_t end_group) {
  if (config.method == Method::kFp16 || end_group <= first_group)
    return {};
  if (first_group == 0 || (dim & (dim - 1)) != 0 ||
      rows.size() < end_group * config.group * heads * dim)
    throw std::invalid_argument("invalid KVarN fake-quantization range");
  const bool rotate = config.method != Method::kRtn;
  const bool balance = config.method == Method::kKvarn;
  const int bits = key ? config.key_bits : config.value_bits;
  const std::size_t group = config.group;
  const std::size_t tiles = (end_group - first_group) * heads;
  std::atomic<std::size_t> next{0};
  std::vector<Error> errors(std::max(1U, std::thread::hardware_concurrency()));
  const auto work = [&](Error& error) {
    std::vector<float> tile(group * dim), original(group * dim);
    for (std::size_t index; (index = next++) < tiles;) {
      const std::size_t g = first_group + index / heads, head = index % heads;
      const auto at = [&](std::size_t t) {
        return rows.data() + ((g * group + t) * heads + head) * dim;
      };
      // Rotated rows as the store's FP16 GEMM writes them.
      for (std::size_t t = 0; t < group; ++t) {
        std::span<float> x(original.data() + t * dim, dim);
        for (std::size_t d = 0; d < dim; ++d)
          x[d] = HalfToFloat(at(t)[d]);
        std::vector<float> y(x.begin(), x.end());
        if (rotate) {
          Hadamard(y);
          for (auto& v : y)
            v = RoundHalf(v);
        }
        for (std::size_t d = 0; d < dim; ++d)
          tile[key ? d * group + t : t * dim + d] = y[d];
      }
      if (key)
        QuantizeTile(tile, dim, group, bits, balance, config.iterations);
      else
        QuantizeTile(tile, group, dim, bits, balance, config.iterations);
      std::vector<float> y(dim);
      for (std::size_t t = 0; t < group; ++t) {
        for (std::size_t d = 0; d < dim; ++d)
          y[d] = RoundHalf(tile[key ? d * group + t : t * dim + d]);
        if (rotate)
          Hadamard(y);
        for (std::size_t d = 0; d < dim; ++d) {
          const auto value = FloatToHalf(y[d]);
          const double reference = original[t * dim + d];
          const double difference = HalfToFloat(value) - reference;
          error.squared += difference * difference;
          error.reference += reference * reference;
          at(t)[d] = value;
        }
      }
    }
  };
  std::vector<std::thread> threads;
  for (std::size_t i = 1; i < errors.size(); ++i)
    threads.emplace_back(work, std::ref(errors[i]));
  work(errors[0]);
  for (auto& thread : threads)
    thread.join();
  Error total;
  for (const auto& error : errors) {
    total.squared += error.squared;
    total.reference += error.reference;
  }
  return total;
}

}  // namespace gufo::testing::kvarn

#endif  // GUFO_TESTS_MODELS_QWEN27B_KVARN_FAKE_QUANT_HPP_
