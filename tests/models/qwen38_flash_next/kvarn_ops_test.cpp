// KVarN store operators of Qwen3.8 Flash-Next
// (plan/exl3/archive/F1-KVARN-DESIGN.md 3, stage K1b) against the CPU reference
// tests/models/qwen27b/kvarn_fake_quant.hpp:
//   * K/V writers given kKvarnFp16RowTag store position p at the FP16 sink
//     and tail ring row hip::KvarnFp16Row(p), the same bytes the contiguous
//     cache gets at row p;
//   * RotateKvRows rotates those rows by the normalized Hadamard bit for bit;
//   * KvarnQuantizeGroups writes k4v4 token-major records, found through a
//     scrambled block table, equal to the reference codes and FP16 scales
//     or within the tolerance below. Balancing reaches float-precision ties
//     and rounding boundaries where the GPU's float-float sums and the
//     reference's double sums may choose differently (plan/exl3/
//     F1-RESULTS.md, K1T diagnosis): uniform rows, and with k4v4
//     token-major values also Qwen3.8-27B-style rows (Gaussian keys with
//     per-channel gains, offsets and outlier channels, Gaussian values; the
//     second K1T window). There a tile must match a tied reference step.
// Positions run past the ring, so later groups overwrite earlier ring rows.
// The read path (stage K2):
//   * RotateHeadRows applies the normalized Hadamard to float rows;
//   * one layer runs the executor's KVarN protocol (prompt chunks through
//     staging, passes with in-pass boundary flushes, a rewind that
//     requantizes a group, the end-of-group flush), and every attention
//     launch reading a hip::KvarnReadView matches the same kernel reading a
//     contiguous FP16 cache of the rows the view rule names: the sink,
//     dequantized records below its quant_end (or the query's
//     hip::KvarnReadGroups), rotated FP16 rows above.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/hip/kvarn_layout.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "tests/models/qwen27b/kvarn_fake_quant.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace hip = gufo::hip;
namespace kvarn = gufo::testing::kvarn;
namespace {

constexpr std::uint32_t kKvHeads = 2, kDim = hip::kKvarnHeadDim;
constexpr std::uint32_t kRow = kKvHeads * kDim, kGroup = hip::kKvarnGroupTokens;
constexpr std::uint32_t kLayers = 2;
constexpr std::size_t kFp16Rows = hip::kKvarnFp16Tokens;
constexpr std::size_t kLayerElements = kFp16Rows * kRow;
constexpr auto kOrder = hip::KvarnValueOrder::kTokenMajor;
int bits = 4;
auto kRecord = hip::KvarnRecord(4, kOrder);

void CheckHip(hipError_t error, const char* what) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " +
                             hipGetErrorString(error));
}

template<typename T>
class Buffer {
public:
  explicit Buffer(std::size_t count) : count_(count) {
    CheckHip(hipMalloc(&data_, bytes()), "hipMalloc");
    CheckHip(hipMemset(data_, 0, bytes()), "hipMemset");
  }
  ~Buffer() { (void)hipFree(data_); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return count_ * sizeof(T);
  }
  void Upload(const std::vector<T>& v) {
    CheckHip(
        hipMemcpy(data_, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice),
        "upload");
  }
  [[nodiscard]] std::vector<T> Download() const {
    std::vector<T> v(count_);
    CheckHip(hipMemcpy(v.data(), data_, bytes(), hipMemcpyDeviceToHost),
             "download");
    return v;
  }

private:
  T* data_{nullptr};
  std::size_t count_;
};

std::uint32_t NextRandom(std::uint32_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

/// Rows of the record check.
enum class Data { kGaussian, kUniform, kStretched };
const char* Name(Data data) {
  return data == Data::kGaussian  ? "27B-style Gaussian"
         : data == Data::kUniform ? "uniform"
                                  : "stretched uniform";
}

/// The Qwen3.8-27B KVarN test's rows (tests/models/qwen/hip/attention/
/// kvarn_ops_test.cpp RandomRows) at this row width: keys normal with a
/// per-channel gain and offset, every 37th hashed channel an outlier, values
/// plain normal.
void GaussianRows(std::size_t tokens, std::uint32_t seed, std::vector<float>* k,
                  std::vector<float>* v) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal;
  std::vector<float> gain(kRow), mean(kRow);
  for (std::uint32_t i = 0; i < kRow; ++i) {
    const bool outlier = (i * 2654435761U >> 7) % 37 == 0;
    gain[i] = outlier ? 6.0F : 0.5F + 0.5F * std::abs(normal(rng));
    mean[i] = outlier ? 3.0F * normal(rng) : 0.3F * normal(rng);
  }
  k->resize(tokens * kRow);
  v->resize(k->size());
  for (std::size_t i = 0; i < k->size(); ++i) {
    (*k)[i] = normal(rng) * gain[i % kRow] + mean[i % kRow];
    (*v)[i] = normal(rng);
  }
}

/// Rows [token][kv head][256], uniform in [-1, 1). With `stretch`, channel c
/// is scaled by 1 + (c % 37) / 8, so balancing has more work to do; without
/// it the rows look like the Qwen3.8-27B KVarN test's.
std::vector<float> MakeRows(std::size_t tokens, std::uint32_t seed,
                            bool stretch = true) {
  std::vector<float> rows(tokens * kRow);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const float unit =
        static_cast<float>(static_cast<int>(NextRandom(&seed) & 0xFFFFU) -
                           32768) /
        32768.0F;
    rows[i] = stretch ? unit * (1.0F + static_cast<float>(i % kDim % 37) / 8.0F)
                      : unit;
  }
  return rows;
}

std::uint16_t Half(float x) {
  return kvarn::FloatToHalf(kvarn::HalfToFloat(kvarn::FloatToHalf(x)));
}

/// The KVarN store of one FP16 head row: rotated, then rounded to FP16.
void Rotate(const std::uint16_t* in, std::uint16_t* out) {
  std::vector<float> x(kDim);
  for (std::uint32_t d = 0; d < kDim; ++d)
    x[d] = kvarn::HalfToFloat(in[d]);
  kvarn::Hadamard(x);
  for (std::uint32_t d = 0; d < kDim; ++d)
    out[d] = kvarn::FloatToHalf(x[d]);
}

kvarn::QuantizedTile Unpack(const std::uint8_t* record, bool key) {
  kvarn::QuantizedTile out;
  const auto halves = [&](std::size_t offset, std::size_t n) {
    std::vector<std::uint16_t> v(n);
    std::memcpy(v.data(), record + offset, 2 * n);
    return v;
  };
  // Both K and token-major V codes: word t * 32 + c / 8 holds channels
  // c..c+7 of token t, c + 2i in nibble i and c + 2i + 1 in nibble i + 4.
  const auto* words = reinterpret_cast<const std::uint32_t*>(
      record + (key ? kRecord.k_codes : kRecord.v_codes));
  out.codes.resize(std::size_t{kDim} * kGroup);
  for (std::uint32_t t = 0; t < kGroup; ++t)
    for (std::uint32_t c = 0; c < kDim; ++c) {
      const std::uint32_t j = c % 8;
      const std::uint32_t nibble = (j % 2 == 0) ? j / 2 : 4 + j / 2;
      auto code = static_cast<std::uint8_t>(
          (words[t * (kDim * bits / 32) + c / 8] >> (4 * nibble)) & 15);
      if (bits == 6) {
        std::uint16_t high;
        std::memcpy(&high,
                    record + (key ? kRecord.k_codes : kRecord.v_codes) +
                        t * (kDim * bits / 8) + kDim / 2 + (c / 8) * 2,
                    2);
        code |= ((high >> (2 * j)) & 3) << 4;
      }
      // The reference tile: K rows are channels, V rows are tokens.
      out.codes[key ? std::size_t{c} * kGroup + t : std::size_t{t} * kDim + c] =
          code;
    }
  if (key) {
    out.scale = halves(kRecord.k_scale, kDim);
    out.zero = halves(kRecord.k_zero, kDim);
    out.column = halves(kRecord.k_column, kGroup);
  } else {
    out.scale = halves(kRecord.v_scale, kGroup);
    out.zero = halves(kRecord.v_zero, kGroup);
    out.column = halves(kRecord.v_column, kDim);
  }
  return out;
}

/// The reference balancing (kvarn_fake_quant.hpp Balance), recording every
/// candidate: the initial state, then each iteration's imbalance and FP16
/// column and row scales. Balance keeps the last candidate whose imbalance
/// is <= the best so far.
struct BalanceStep {
  float imbalance{0};
  std::vector<float> column_scale, row_scale;
  std::vector<std::uint16_t> column, row;
};
std::vector<BalanceStep> TraceBalance(const std::vector<float>& source,
                                      std::size_t rows, std::size_t cols) {
  namespace d = kvarn::detail;
  constexpr float kStdMin = 1e-3F, kStdMax = 1e3F, kLogMin = -0.3F,
                  kLogMax = 10.0F;
  std::vector<float> log_column(cols, 0.0F), log_row(rows, 0.0F);
  std::vector<float> current = source, column(cols), row(rows);
  std::vector<BalanceStep> steps;
  const auto record = [&] {
    BalanceStep step;
    step.imbalance = d::Imbalance(current, rows, cols);
    for (std::size_t c = 0; c < cols; ++c) {
      step.column_scale.push_back(d::Exp(log_column[c]));
      step.column.push_back(kvarn::FloatToHalf(step.column_scale.back()));
    }
    for (std::size_t r = 0; r < rows; ++r) {
      step.row_scale.push_back(d::Exp(log_row[r]));
      step.row.push_back(kvarn::FloatToHalf(step.row_scale.back()));
    }
    steps.push_back(std::move(step));
  };
  const auto apply = [&] {
    for (std::size_t c = 0; c < cols; ++c)
      column[c] = d::Exp(log_column[c]);
    for (std::size_t r = 0; r < rows; ++r)
      row[r] = d::Exp(log_row[r]);
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t c = 0; c < cols; ++c)
        current[r * cols + c] = source[r * cols + c] / column[c] / row[r];
  };
  record();
  for (int iteration = 0; iteration < 8; ++iteration) {
    for (std::size_t c = 0; c < cols; ++c) {
      const float s =
          std::clamp(d::Std(current.data() + c, rows, cols), kStdMin, kStdMax);
      log_column[c] = std::clamp(log_column[c] + d::Log(s), kLogMin, kLogMax);
    }
    apply();
    for (std::size_t r = 0; r < rows; ++r) {
      const float s = std::clamp(d::Std(current.data() + r * cols, cols, 1),
                                 kStdMin, kStdMax);
      log_row[r] = std::clamp(log_row[r] + d::Log(s), kLogMin, kLogMax);
    }
    apply();
    record();
  }
  return steps;
}

/// The candidate Balance keeps.
std::size_t ChosenStep(const std::vector<BalanceStep>& steps) {
  std::size_t chosen = 0;
  float best = steps[0].imbalance;
  for (std::size_t i = 1; i < steps.size(); ++i)
    if (steps[i].imbalance <= best) {
      best = steps[i].imbalance;
      chosen = i;
    }
  return chosen;
}

/// The reference record of `source` balanced with `step`'s scales: the end of
/// kvarn::QuantizeTileCodes (4-bit RTN per row, stored FP16 scales).
kvarn::QuantizedTile QuantizeAtStep(const std::vector<float>& source,
                                    std::size_t rows, std::size_t cols,
                                    const BalanceStep& step) {
  std::vector<float> tile(source.size());
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t c = 0; c < cols; ++c)
      tile[r * cols + c] =
          source[r * cols + c] / step.column_scale[c] / step.row_scale[r];
  const float kMax = static_cast<float>((1 << bits) - 1);
  kvarn::QuantizedTile out;
  out.codes.resize(rows * cols);
  for (std::size_t r = 0; r < rows; ++r) {
    const float* row = tile.data() + r * cols;
    const auto [lo, hi] = std::minmax_element(row, row + cols);
    const float low = *lo;
    const float scale = std::max((*hi - low) / kMax, 1e-10F);
    out.scale.push_back(kvarn::FloatToHalf(step.row_scale[r] * scale));
    out.zero.push_back(kvarn::FloatToHalf(step.row_scale[r] * low));
    for (std::size_t c = 0; c < cols; ++c)
      out.codes[r * cols + c] = static_cast<std::uint8_t>(
          std::clamp(std::nearbyint((row[c] - low) / scale), 0.0F, kMax));
  }
  for (std::size_t c = 0; c < cols; ++c)
    out.column.push_back(kvarn::FloatToHalf(step.column_scale[c]));
  return out;
}

/// Tolerance for uniform rows: at most kNearCodes codes of a tile (about
/// 1e-4 of 32,768) and every stored FP16 value within one ulp.
constexpr std::size_t kNearCodes = 3;
bool Near(const kvarn::QuantizedTile& a, const kvarn::QuantizedTile& b) {
  std::size_t codes = 0;
  for (std::size_t i = 0; i < a.codes.size(); ++i)
    codes += a.codes[i] != b.codes[i];
  const auto ulp = [](const std::vector<std::uint16_t>& x,
                      const std::vector<std::uint16_t>& y) {
    for (std::size_t i = 0; i < x.size(); ++i) {
      // Same sign, adjacent FP16 encodings (or equal).
      if ((x[i] ^ y[i]) & 0x8000U) {
        if (((x[i] | y[i]) & 0x7fffU) != 0)
          return false;
        continue;
      }
      if ((x[i] > y[i] ? x[i] - y[i] : y[i] - x[i]) > 1)
        return false;
    }
    return true;
  };
  return codes <= kNearCodes && ulp(a.scale, b.scale) && ulp(a.zero, b.zero) &&
         ulp(a.column, b.column);
}

bool Same(const kvarn::QuantizedTile& a, const kvarn::QuantizedTile& b) {
  return a.codes == b.codes && a.scale == b.scale && a.zero == b.zero &&
         a.column == b.column;
}

int failures = 0;
void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << '\n';
  failures += ok ? 0 : 1;
}

/// The writers' tagged rows equal their contiguous rows: positions below the
/// ring's end land on the same row; later ones wrap.
void CheckWriterRows() {
  constexpr std::uint32_t kStride = 2 * (24 * kDim + kRow);
  for (const auto [start, n] : {std::pair{0U, 300U}, std::pair{500U, 8U}}) {
    const auto packed = MakeRows(std::size_t{n} * kStride / kRow, 7 + start);
    std::vector<float> gamma(2 * kDim, 1.0F);
    Buffer<float> d_packed(packed.size()), d_gamma(gamma.size()),
        d_q(std::size_t{n} * 24 * kDim), d_gate(std::size_t{n} * 24 * kDim),
        d_src(std::size_t{n} * kRow);
    d_packed.Upload(packed);
    d_gamma.Upload(gamma);
    d_src.Upload(MakeRows(n, 11 + start));
    Buffer<std::uint32_t> d_pos(1);
    d_pos.Upload({start});
    const std::size_t contiguous = std::size_t{start + n} * kRow;
    Buffer<__half> ck(contiguous), cv(contiguous), cs(contiguous),
        rk(kLayerElements), rv(kLayerElements), rs(kLayerElements);
    if (!q::PrepareAttention(d_packed.get(), kStride, d_gamma.get(),
                             d_gamma.get() + kDim, d_q.get(), d_gate.get(),
                             ck.get(), cv.get(), n, 24, kKvHeads, kDim, 64,
                             d_pos.get(), 1e7F, 1e-6F, nullptr) ||
        !q::PrepareAttention(d_packed.get(), kStride, d_gamma.get(),
                             d_gamma.get() + kDim, d_q.get(), d_gate.get(),
                             rk.get(), rv.get(), n, 24, kKvHeads, kDim, 64,
                             d_pos.get(), 1e7F, 1e-6F, nullptr, nullptr, false,
                             {}, q::KvarnFp16Rows()))
      throw std::runtime_error("preparation rejected its geometry");
    q::StoreKv(d_src.get(), cs.get(), n, kRow, d_pos.get(), nullptr);
    q::StoreKv(d_src.get(), rs.get(), n, kRow, d_pos.get(), nullptr,
               q::KvarnFp16Rows());
    CheckHip(hipDeviceSynchronize(), "writers");
    std::size_t differ = 0;
    for (auto [c, r] :
         {std::pair{&ck, &rk}, std::pair{&cv, &rv}, std::pair{&cs, &rs}}) {
      const auto a = c->Download(), b = r->Download();
      for (std::uint32_t p = start; p < start + n; ++p)
        differ += std::memcmp(a.data() + std::size_t{p} * kRow,
                              b.data() + hip::KvarnFp16Row(p) * kRow,
                              kRow * sizeof(__half)) != 0;
    }
    Check(differ == 0, "writer ring rows start=" + std::to_string(start) +
                           " n=" + std::to_string(n) +
                           " differing=" + std::to_string(differ));
  }
}

/// Sink plus groups 1..4 written in two phases (the ring holds two groups),
/// rotated, quantized into scrambled-table records, compared with the
/// reference tile by tile.
void CheckRotateAndQuantize(Data data) {
  constexpr std::uint32_t kTokens = 5 * kGroup;  // sink + groups 1..4
  constexpr std::uint32_t kBlocks = (kTokens + 255) / 256;  // 3
  constexpr std::uint32_t kPool = 2 * kBlocks + 4;          // 10
  std::vector<std::uint32_t> table(kBlocks);
  for (std::uint32_t b = 0; b < kBlocks; ++b)
    table[b] = (kPool - 1) - (b * 3) % kPool;  // 9, 6, 3
  const std::size_t layer_pitch =
      std::size_t{kPool} * hip::kKvarnGroupsPerBlock * kKvHeads * kRecord.bytes;
  Buffer<std::uint8_t> records(kLayers * layer_pitch);
  Buffer<std::uint32_t> d_table(kBlocks), d_pos(1);
  d_table.Upload(table);
  Buffer<__half> k_rows(kLayers * kLayerElements),
      v_rows(kLayers * kLayerElements);
  std::vector<std::vector<float>> k_src(kLayers), v_src(kLayers);
  for (std::uint32_t l = 0; l < kLayers; ++l) {
    if (data == Data::kGaussian) {
      GaussianRows(kTokens, 300 + l, &k_src[l], &v_src[l]);
    } else {
      k_src[l] = MakeRows(kTokens, 100 + l, data == Data::kStretched);
      v_src[l] = MakeRows(kTokens, 200 + l, data == Data::kStretched);
    }
  }
  std::size_t sink_differ = 0, ring_differ = 0;
  // Phase 1 writes the sink and groups 1-2, phase 2 groups 3-4 over them.
  for (const auto [begin, end] :
       {std::pair{0U, 3 * kGroup}, std::pair{3 * kGroup, 5 * kGroup}}) {
    const std::uint32_t n = end - begin;
    d_pos.Upload({begin});
    for (std::uint32_t l = 0; l < kLayers; ++l) {
      Buffer<float> k(std::size_t{n} * kRow), v(std::size_t{n} * kRow);
      k.Upload(std::vector<float>(k_src[l].begin() + std::size_t{begin} * kRow,
                                  k_src[l].begin() + std::size_t{end} * kRow));
      v.Upload(std::vector<float>(v_src[l].begin() + std::size_t{begin} * kRow,
                                  v_src[l].begin() + std::size_t{end} * kRow));
      q::StoreKv(k.get(), k_rows.get() + l * kLayerElements, n, kRow,
                 d_pos.get(), nullptr, q::KvarnFp16Rows());
      q::StoreKv(v.get(), v_rows.get() + l * kLayerElements, n, kRow,
                 d_pos.get(), nullptr, q::KvarnFp16Rows());
      q::RotateKvRows(k_rows.get() + l * kLayerElements,
                      v_rows.get() + l * kLayerElements, n, kKvHeads,
                      d_pos.get(), nullptr);
    }
    const std::uint32_t first = std::max(1U, begin / kGroup);
    q::KvarnQuantizeGroups(k_rows.get(), v_rows.get(), kLayerElements,
                           records.get(), layer_pitch, d_table.get(), kKvHeads,
                           kLayers, first, end / kGroup - first, nullptr,
                           q::kKvarnRingRows, bits);
    CheckHip(hipDeviceSynchronize(), "rotate and quantize");
    // The rotated rows of this phase against the reference rotation.
    const auto got_k = k_rows.Download(), got_v = v_rows.Download();
    for (std::uint32_t l = 0; l < kLayers; ++l)
      for (std::uint32_t p = begin; p < end; ++p)
        for (std::uint32_t h = 0; h < kKvHeads; ++h)
          for (const bool key : {true, false}) {
            const auto& src = key ? k_src[l] : v_src[l];
            std::uint16_t in[kDim], want[kDim];
            for (std::uint32_t d = 0; d < kDim; ++d)
              in[d] = Half(src[(std::size_t{p} * kKvHeads + h) * kDim + d]);
            Rotate(in, want);
            const auto& got = key ? got_k : got_v;
            const bool same =
                std::memcmp(got.data() + l * kLayerElements +
                                (hip::KvarnFp16Row(p) * kKvHeads + h) * kDim,
                            want, sizeof(want)) == 0;
            (p < hip::kKvarnSinkTokens ? sink_differ : ring_differ) += !same;
          }
  }
  Check(sink_differ == 0 && ring_differ == 0,
        "rotated rows sink_differing=" + std::to_string(sink_differ) +
            " ring_differing=" + std::to_string(ring_differ));
  // Records of groups 1..4 against the reference quantizer.
  const auto bytes = records.Download();
  std::size_t tiles = 0, exact = 0, tied = 0, boundary = 0, mismatched = 0,
              mirror_mismatch = 0;
  for (std::uint32_t l = 0; l < kLayers; ++l)
    for (std::uint32_t g = 1; g < kTokens / kGroup; ++g)
      for (std::uint32_t h = 0; h < kKvHeads; ++h)
        for (const bool key : {true, false}) {
          const auto& src = key ? k_src[l] : v_src[l];
          std::vector<float> tile(std::size_t{kDim} * kGroup);
          for (std::uint32_t t = 0; t < kGroup; ++t) {
            std::uint16_t in[kDim], rotated[kDim];
            for (std::uint32_t d = 0; d < kDim; ++d)
              in[d] = Half(
                  src[(std::size_t{g * kGroup + t} * kKvHeads + h) * kDim + d]);
            Rotate(in, rotated);
            for (std::uint32_t d = 0; d < kDim; ++d)
              tile[key ? std::size_t{d} * kGroup + t
                       : std::size_t{t} * kDim + d] =
                  kvarn::HalfToFloat(rotated[d]);
          }
          const std::vector<float> source_tile = tile;
          const auto ref = kvarn::QuantizeTileCodes(
              tile, key ? kDim : kGroup, key ? kGroup : kDim, bits, true, 8);
          const std::size_t plane_row =
              std::size_t{table[g / hip::kKvarnGroupsPerBlock]} *
                  hip::kKvarnGroupsPerBlock +
              g % hip::kKvarnGroupsPerBlock;
          const auto got =
              Unpack(bytes.data() + l * layer_pitch +
                         (plane_row * kKvHeads + h) * kRecord.bytes,
                     key);
          ++tiles;
          if (Same(ref, got)) {
            ++exact;
            continue;
          }
          const auto steps = TraceBalance(source_tile, key ? kDim : kGroup,
                                          key ? kGroup : kDim);
          const std::size_t chosen = ChosenStep(steps);
          // The step-wise mirror must reproduce the reference at its step.
          mirror_mismatch +=
              !Same(QuantizeAtStep(source_tile, key ? kDim : kGroup,
                                   key ? kGroup : kDim, steps[chosen]),
                    ref);
          // Candidates: the reference's step and every step whose imbalance
          // ties it within one float ulp.
          const float best = steps[chosen].imbalance;
          const float tie = std::nextafter(best, INFINITY) - best;
          std::string kind;
          for (std::size_t i = 0; i < steps.size() && kind.empty(); ++i) {
            if (std::abs(steps[i].imbalance - best) > tie)
              continue;
            if (Near(QuantizeAtStep(source_tile, key ? kDim : kGroup,
                                    key ? kGroup : kDim, steps[i]),
                     got))
              kind = i == chosen ? "boundary" : "tie";
          }
          if (!kind.empty()) {
            std::size_t codes = 0;
            for (std::size_t i = 0; i < ref.codes.size(); ++i)
              codes += ref.codes[i] != got.codes[i];
            std::cout << "  " << kind << " tile layer=" << l << " group=" << g
                      << " head=" << h << (key ? " K" : " V")
                      << " codes_differ=" << codes << '\n';
            ++(kind == "tie" ? tied : boundary);
            continue;
          }
          ++mismatched;
          std::size_t codes = 0;
          for (std::size_t i = 0; i < ref.codes.size(); ++i)
            codes += ref.codes[i] != got.codes[i];
          std::string matches;
          for (std::size_t i = 0; i < steps.size(); ++i)
            if (steps[i].column == got.column)
              matches += (matches.empty() ? "" : ",") + std::to_string(i);
          std::cout << "  mismatched tile layer=" << l << " group=" << g
                    << " head=" << h << (key ? " K" : " V")
                    << " codes_differ=" << codes << " reference_step=" << chosen
                    << " gpu_column_matches_step="
                    << (matches.empty() ? "none" : matches)
                    << " imbalance:" << std::setprecision(9);
          for (const auto& step : steps)
            std::cout << ' ' << step.imbalance;
          std::cout << std::setprecision(6) << '\n';
        }
  const std::string counts =
      std::string(Name(data)) +
      " token-major records tiles=" + std::to_string(tiles) +
      " exact=" + std::to_string(exact) + " tied_step=" + std::to_string(tied) +
      " boundary=" + std::to_string(boundary) +
      " mismatched=" + std::to_string(mismatched) +
      " mirror_mismatch=" + std::to_string(mirror_mismatch);
  Check(mismatched == 0 && mirror_mismatch == 0, counts + " (tolerance)");
}

/// RotateHeadRows against the reference rotation; twice is the identity.
void CheckRotateHeadRows() {
  constexpr std::uint32_t kRows = 24 * 3;
  std::mt19937 rng(5);
  std::normal_distribution<float> normal;
  std::vector<float> x(std::size_t{kRows} * kDim);
  for (auto& value : x)
    value = normal(rng);
  Buffer<float> d(x.size());
  d.Upload(x);
  q::RotateHeadRows(d.get(), kRows, nullptr);
  CheckHip(hipDeviceSynchronize(), "rotate head rows");
  const auto once = d.Download();
  q::RotateHeadRows(d.get(), kRows, nullptr);
  CheckHip(hipDeviceSynchronize(), "rotate head rows twice");
  const auto twice = d.Download();
  float rotate_error = 0, identity_error = 0;
  for (std::uint32_t r = 0; r < kRows; ++r) {
    std::vector<float> want(x.begin() + std::size_t{r} * kDim,
                            x.begin() + std::size_t{r + 1} * kDim);
    kvarn::Hadamard(want);
    for (std::uint32_t i = 0; i < kDim; ++i) {
      const std::size_t at = std::size_t{r} * kDim + i;
      rotate_error = std::max(rotate_error, std::abs(once[at] - want[i]));
      identity_error = std::max(identity_error, std::abs(twice[at] - x[at]));
    }
  }
  Check(rotate_error < 1e-5F && identity_error < 1e-5F,
        "RotateHeadRows max_error=" + std::to_string(rotate_error) +
            " twice_identity_error=" + std::to_string(identity_error));
}

/// One layer's KVarN state driven the way the executor drives it
/// (Executor::PlanKvarn, Attention, KvarnAttention and FinishKvarn), each
/// attention checked against the contiguous cache its view rule names.
class ReadPathCheck {
public:
  static constexpr std::uint32_t kHeads = 24, kRatio = 4;
  static constexpr std::uint32_t kPositions = 1152;
  static constexpr std::uint32_t kTableBlocks = (kPositions + 255) / 256;
  static constexpr std::uint32_t kPool = 7;
  static constexpr std::uint32_t kMaxChunk = 400;
  static constexpr std::uint32_t kMaskWords = 2048 / kRatio / 32;
  static constexpr std::size_t kQRow = std::size_t{kHeads} * kDim;

  ReadPathCheck()
      : k_(kLayerElements),
        v_(kLayerElements),
        staging_k_(hip::KvarnStagingRows(kMaxChunk) * kRow),
        staging_v_(hip::KvarnStagingRows(kMaxChunk) * kRow),
        records_(std::size_t{kPool} * hip::kKvarnGroupsPerBlock * kKvHeads *
                 kRecord.bytes),
        table_(kTableBlocks),
        pos_(1) {
    for (std::uint32_t b = 0; b < kTableBlocks; ++b)
      table_host_.push_back(b * 3 % kPool);  // 0, 3, 6, 2, 5
    table_.Upload(table_host_);
    GaussianRows(kPositions, 900, &k_src_, &v_src_);
    for (const auto* src : {&k_src_, &v_src_}) {
      auto& rotated = src == &k_src_ ? k_rot_ : v_rot_;
      rotated.resize(src->size());
      for (std::uint32_t p = 0; p < kPositions; ++p)
        for (std::uint32_t h = 0; h < kKvHeads; ++h) {
          const std::size_t at = (std::size_t{p} * kKvHeads + h) * kDim;
          std::uint16_t in[kDim];
          for (std::uint32_t d = 0; d < kDim; ++d)
            in[d] = Half((*src)[at + d]);
          Rotate(in, rotated.data() + at);
        }
    }
  }

  /// A pass of 1..128 rows (decode, verification, a short prompt).
  void Pass(std::uint32_t start, std::uint32_t n) {
    const auto plan = hip::PlanKvarnPass(tail_, start, n);
    if (!plan.error.empty())
      throw std::runtime_error(std::string(plan.error));
    Flush(plan.first_group, plan.groups);
    Write(start, n, k_.get(), v_.get(), q::KvarnFp16Rows());
    q::RotateKvRows(k_.get(), v_.get(), n, kKvHeads, pos_.get(), nullptr);
    q::KvarnQuantizePassBoundary(k_.get(), v_.get(), records_.get(),
                                 table_.get(), kKvHeads, pos_.get(), n, nullptr,
                                 bits);
    Attend(start, n, false, 0, 0);
    const std::uint32_t end = start + n;
    tail_.flushed_end = plan.flushed_end;
    tail_.written_end = std::max(tail_.written_end, end);
    if (end % kGroup == 0 && end > hip::kKvarnSinkTokens &&
        plan.flushed_end < end) {
      Flush(end / kGroup - 1, 1);
      tail_.flushed_end = end;
    }
  }

  /// A prompt chunk of more than 128 rows through the staging rows.
  void Chunk(std::uint32_t start, std::uint32_t n) {
    const auto plan = hip::PlanKvarnPrefill(tail_, start, n);
    if (!plan.error.empty())
      throw std::runtime_error(std::string(plan.error));
    Flush(plan.first_group, plan.groups);
    const std::uint32_t first = start < hip::kKvarnSinkTokens ? 0 : plan.target;
    const std::size_t row_bytes = std::size_t{kRow} * sizeof(__half);
    const std::size_t from = first == 0 ? 0 : hip::KvarnFp16Row(first);
    if (start > first) {
      CheckHip(hipMemcpy(staging_k_.get(), k_.get() + from * kRow,
                         (start - first) * row_bytes, hipMemcpyDeviceToDevice),
               "staging lead");
      CheckHip(hipMemcpy(staging_v_.get(), v_.get() + from * kRow,
                         (start - first) * row_bytes, hipMemcpyDeviceToDevice),
               "staging lead");
    }
    Write(start, n, staging_k_.get() - std::size_t{first} * kRow,
          staging_v_.get() - std::size_t{first} * kRow, nullptr);
    q::RotateKvRows(staging_k_.get(), staging_v_.get(), n, kKvHeads, pos_.get(),
                    nullptr, first);
    Attend(start, n, true, first, plan.target);
    const std::uint32_t end = start + n;
    const std::uint32_t first_group = first == 0 ? 1 : first / kGroup;
    const std::uint32_t complete = end / kGroup;
    if (complete > first_group)
      q::KvarnQuantizeGroups(staging_k_.get(), staging_v_.get(), 0,
                             records_.get(), 0, table_.get(), kKvHeads, 1,
                             first_group, complete - first_group, nullptr,
                             first, bits);
    const auto copy = [&](std::size_t to, std::uint32_t position,
                          std::uint32_t rows) {
      const std::size_t at = std::size_t{position - first} * kRow;
      CheckHip(hipMemcpy(k_.get() + to * kRow, staging_k_.get() + at,
                         rows * row_bytes, hipMemcpyDeviceToDevice),
               "chunk rows");
      CheckHip(hipMemcpy(v_.get() + to * kRow, staging_v_.get() + at,
                         rows * row_bytes, hipMemcpyDeviceToDevice),
               "chunk rows");
    };
    if (first == 0)
      copy(0, 0, std::min(end, hip::kKvarnSinkTokens));
    const std::uint32_t reach =
        end > hip::kKvarnTailRingTokens
            ? (end - hip::kKvarnTailRingTokens + kGroup - 1) / kGroup * kGroup
            : 0;
    for (std::uint32_t p = std::max({hip::kKvarnSinkTokens, first, reach});
         p < end;) {
      const std::uint32_t piece = std::min(end, (p / kGroup + 1) * kGroup);
      copy(hip::KvarnFp16Row(p), p, piece - p);
      p = piece;
    }
    tail_.flushed_end = plan.flushed_end;
    tail_.written_end = std::max(tail_.written_end, end);
  }

  void Report() const {
    Check(launches_ > 0 && failed_.empty(),
          "KVarN read path launches=" + std::to_string(launches_) +
              " max_abs_difference=" + std::to_string(max_difference_) +
              " bitwise_launches=" + std::to_string(bitwise_) +
              (failed_.empty() ? "" : " failed:" + failed_));
  }

private:
  void Flush(std::uint32_t first, std::uint32_t groups) {
    if (groups != 0)
      q::KvarnQuantizeGroups(k_.get(), v_.get(), kLayerElements, records_.get(),
                             0, table_.get(), kKvHeads, 1, first, groups,
                             nullptr, q::kKvarnRingRows, bits);
  }

  void Write(std::uint32_t start, std::uint32_t n, __half* k, __half* v,
             const std::uint32_t* rows) {
    pos_.Upload({start});
    for (const bool key : {true, false}) {
      const auto& src = key ? k_src_ : v_src_;
      Buffer<float> d(std::size_t{n} * kRow);
      d.Upload(std::vector<float>(src.begin() + std::size_t{start} * kRow,
                                  src.begin() + std::size_t{start + n} * kRow));
      q::StoreKv(d.get(), key ? k : v, n, kRow, pos_.get(), nullptr, rows);
      CheckHip(hipDeviceSynchronize(), "rows");
    }
  }

  /// Rotated FP16 row of position j as the view rule reads it.
  void ReadRow(const std::vector<std::uint8_t>& records, std::uint32_t j,
               std::uint32_t records_end, std::uint16_t* k,
               std::uint16_t* v) const {
    for (std::uint32_t h = 0; h < kKvHeads; ++h)
      for (const bool key : {true, false}) {
        auto* out = (key ? k : v) + h * kDim;
        const std::size_t at = (std::size_t{j} * kKvHeads + h) * kDim;
        if (j < hip::kKvarnSinkTokens || j >= records_end) {
          std::memcpy(out, (key ? k_rot_ : v_rot_).data() + at, 2 * kDim);
          continue;
        }
        const std::uint32_t g = j / kGroup, t = j % kGroup;
        const std::size_t plane_row =
            std::size_t{table_host_[g / hip::kKvarnGroupsPerBlock]} *
                hip::kKvarnGroupsPerBlock +
            g % hip::kKvarnGroupsPerBlock;
        const auto tile = Unpack(
            records.data() + (plane_row * kKvHeads + h) * kRecord.bytes, key);
        for (std::uint32_t c = 0; c < kDim; ++c) {
          const float code = tile.codes[key ? std::size_t{c} * kGroup + t
                                            : std::size_t{t} * kDim + c];
          const float scale = kvarn::HalfToFloat(tile.scale[key ? c : t]);
          const float zero = kvarn::HalfToFloat(tile.zero[key ? c : t]);
          const float column = kvarn::HalfToFloat(tile.column[key ? t : c]);
          out[c] = kvarn::FloatToHalf(std::fma(code, scale, zero) * column);
        }
      }
  }

  void Attend(std::uint32_t start, std::uint32_t n, bool prefill,
              std::uint32_t staging_first, std::uint32_t quant_end) {
    CheckHip(hipDeviceSynchronize(), "KVarN writes");
    std::mt19937 rng(start * 131 + n);
    std::normal_distribution<float> normal;
    std::vector<float> query(n * kQRow);
    for (auto& value : query)
      value = normal(rng);
    Buffer<float> d_q(query.size()), out(query.size()), ref(query.size()),
        partials(std::size_t{n} * kHeads * 8 * (kDim + 2));
    d_q.Upload(query);
    const q::KvarnReadView view{
        .k_fp16 = k_.get(),
        .v_fp16 = v_.get(),
        .k_staging = prefill ? staging_k_.get() : nullptr,
        .v_staging = prefill ? staging_v_.get() : nullptr,
        .staging_first = staging_first,
        .records = records_.get(),
        .blocks = table_.get(),
        .quant_end = prefill ? quant_end : 0,
        .kv_heads = kKvHeads,
        .bits = bits};
    const bool split = n == 1;
    const auto attend = [&](float* o, const __half* k, const __half* v,
                            std::uint32_t first, std::uint32_t rows,
                            const std::uint32_t* blocks,
                            const q::KvarnReadView* kvarn) {
      Buffer<std::uint32_t> d_pos(1);
      d_pos.Upload({start + first});
      if (prefill) {
        if (!q::WmmaCausalAttention(d_q.get() + first * kQRow, nullptr, k, v,
                                    nullptr, kMaskWords, o + first * kQRow,
                                    rows, start + first, kHeads, kKvHeads, kDim,
                                    kRatio, nullptr, false, blocks, kvarn))
          throw std::runtime_error("WMMA attention rejected the geometry");
      } else {
        q::Attention(d_q.get() + first * kQRow, k, v, nullptr, kMaskWords,
                     o + first * kQRow, split ? partials.get() : nullptr, 8,
                     rows, d_pos.get(), kHeads, kKvHeads, kDim, kRatio, nullptr,
                     nullptr, nullptr, 0, blocks, kvarn);
      }
      CheckHip(hipDeviceSynchronize(), "attention");
    };
    attend(out.get(), nullptr, nullptr, 0, n, table_.get(), &view);
    // The reference: per run of queries reading the same records.
    const auto records = records_.Download();
    for (std::uint32_t first = 0; first < n;) {
      const auto records_end = [&](std::uint32_t p) {
        return prefill ? quant_end : kGroup * (1 + hip::KvarnReadGroups(p));
      };
      std::uint32_t last = first + 1;
      while (last < n &&
             records_end(start + last) == records_end(start + first))
        ++last;
      const std::uint32_t context = start + last;
      std::vector<std::uint16_t> k_rows(std::size_t{context} * kRow),
          v_rows(k_rows.size());
      for (std::uint32_t j = 0; j < context; ++j)
        ReadRow(records, j, records_end(start + first),
                k_rows.data() + std::size_t{j} * kRow,
                v_rows.data() + std::size_t{j} * kRow);
      Buffer<__half> d_k(k_rows.size()), d_v(v_rows.size());
      CheckHip(hipMemcpy(d_k.get(), k_rows.data(), 2 * k_rows.size(),
                         hipMemcpyHostToDevice),
               "reference K");
      CheckHip(hipMemcpy(d_v.get(), v_rows.data(), 2 * v_rows.size(),
                         hipMemcpyHostToDevice),
               "reference V");
      attend(ref.get(), d_k.get(), d_v.get(), first, last - first, nullptr,
             nullptr);
      first = last;
    }
    const auto got = out.Download(), want = ref.Download();
    float difference = 0;
    for (std::size_t i = 0; i < got.size(); ++i)
      difference = std::max(difference, std::abs(got[i] - want[i]));
    ++launches_;
    bitwise_ += std::memcmp(got.data(), want.data(), 4 * got.size()) == 0;
    max_difference_ = std::max(max_difference_, difference);
    if (!(difference <= 2e-3F))
      failed_ += " " + std::string(prefill ? "chunk" : "pass") + "@" +
                 std::to_string(start) + "+" + std::to_string(n) + "=" +
                 std::to_string(difference);
  }

  Buffer<__half> k_, v_, staging_k_, staging_v_;
  Buffer<std::uint8_t> records_;
  Buffer<std::uint32_t> table_, pos_;
  std::vector<std::uint32_t> table_host_;
  std::vector<float> k_src_, v_src_;
  std::vector<std::uint16_t> k_rot_, v_rot_;
  hip::KvarnTail tail_;
  std::size_t launches_ = 0, bitwise_ = 0;
  float max_difference_ = 0;
  std::string failed_;
};

void CheckReadPath() {
  ReadPathCheck check;
  check.Chunk(0, 300);   // starts in the sink, completes group 1
  check.Pass(300, 1);    // decode in group 2
  check.Pass(301, 100);  // crosses 384: in-pass flush of group 2
  for (std::uint32_t p = 401; p < 512; p += 37)
    check.Pass(p, std::min(37U, 512 - p));  // the last ends at 512
  check.Pass(512, 1);
  check.Chunk(513, 400);  // stages the ring rows of group 4
  check.Pass(913, 8);     // verification width
  check.Pass(921, 99);    // up to 1020
  check.Pass(1020, 10);   // crosses 1024
  check.Pass(1022, 1);    // rewinds into group 7, requantized at 1024
  check.Pass(1023, 1);    // ends at 1024: end-of-group flush
  check.Pass(1024, 8);
  check.Report();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--k6v6")
      bits = 6;
    else if (argc != 1)
      throw std::invalid_argument("usage: kvarn_ops_test [--k6v6]");
    kRecord = hip::KvarnRecord(bits, kOrder, bits);
    std::cout << "KV bits=" << bits << ", record bytes=" << kRecord.bytes
              << "\n";
    CheckWriterRows();
    CheckRotateAndQuantize(Data::kGaussian);
    CheckRotateAndQuantize(Data::kUniform);
    CheckRotateAndQuantize(Data::kStretched);
    CheckRotateHeadRows();
    CheckReadPath();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return failures == 0 ? 0 : 1;
}
