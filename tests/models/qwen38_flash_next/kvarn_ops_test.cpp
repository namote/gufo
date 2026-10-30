// KVarN store operators of Qwen3.8 Flash-Next (plan/exl3/F1-KVARN-DESIGN.md
// 3, stage K1b) against the CPU reference
// tests/models/qwen27b/kvarn_fake_quant.hpp:
//   * K/V writers given kKvarnFp16RowTag store position p at the FP16 sink
//     and tail ring row hip::KvarnFp16Row(p), the same bytes the contiguous
//     cache gets at row p;
//   * RotateKvRows rotates those rows by the normalized Hadamard bit for bit;
//   * KvarnQuantizeGroups writes k4v4 token-major records, found through a
//     scrambled block table, equal to the reference codes and FP16 scales.
// Positions run past the ring, so later groups overwrite earlier ring rows.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
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
constexpr auto kRecord = hip::KvarnRecord(4, kOrder);

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
      const auto code = static_cast<std::uint8_t>(
          (words[t * 32 + c / 8] >> (4 * nibble)) & 15);
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
    for (std::size_t c = 0; c < cols; ++c)
      step.column.push_back(kvarn::FloatToHalf(d::Exp(log_column[c])));
    for (std::size_t r = 0; r < rows; ++r)
      step.row.push_back(kvarn::FloatToHalf(d::Exp(log_row[r])));
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
void CheckRotateAndQuantize(bool stretch) {
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
    k_src[l] = MakeRows(kTokens, 100 + l, stretch);
    v_src[l] = MakeRows(kTokens, 200 + l, stretch);
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
                           kLayers, first, end / kGroup - first, nullptr);
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
  std::size_t tiles = 0, exact = 0, code_diff = 0, scale_diff = 0;
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
              tile, key ? kDim : kGroup, key ? kGroup : kDim, 4, true, 8);
          const std::size_t plane_row =
              std::size_t{table[g / hip::kKvarnGroupsPerBlock]} *
                  hip::kKvarnGroupsPerBlock +
              g % hip::kKvarnGroupsPerBlock;
          const auto got =
              Unpack(bytes.data() + l * layer_pitch +
                         (plane_row * kKvHeads + h) * kRecord.bytes,
                     key);
          std::size_t codes = 0, scales = 0;
          for (std::size_t i = 0; i < ref.codes.size(); ++i)
            codes += ref.codes[i] != got.codes[i];
          const auto count = [&](const std::vector<std::uint16_t>& a,
                                 const std::vector<std::uint16_t>& b) {
            for (std::size_t i = 0; i < a.size(); ++i)
              scales += a[i] != b[i];
          };
          count(ref.scale, got.scale);
          count(ref.zero, got.zero);
          count(ref.column, got.column);
          ++tiles;
          exact += codes == 0 && scales == 0;
          if (codes != 0 || scales != 0) {
            // Which reference balancing candidate the stored columns match.
            const auto steps = TraceBalance(source_tile, key ? kDim : kGroup,
                                            key ? kGroup : kDim);
            const std::size_t chosen = ChosenStep(steps);
            std::string matches;
            for (std::size_t i = 0; i < steps.size(); ++i)
              if (steps[i].column == got.column)
                matches += (matches.empty() ? "" : ",") + std::to_string(i);
            std::size_t first = 0;
            while (first < ref.scale.size() &&
                   ref.scale[first] == got.scale[first])
              ++first;
            std::cout << "  tile layer=" << l << " group=" << g << " head=" << h
                      << (key ? " K" : " V") << " codes_differ=" << codes
                      << " scales_differ=" << scales
                      << " reference_step=" << chosen
                      << " gpu_column_matches_step="
                      << (matches.empty() ? "none" : matches)
                      << " first_scale=" << first;
            if (first < ref.scale.size())
              std::cout << " ref=" << kvarn::HalfToFloat(ref.scale[first])
                        << " gpu=" << kvarn::HalfToFloat(got.scale[first]);
            std::cout << " imbalance:" << std::setprecision(9);
            for (const auto& step : steps)
              std::cout << ' ' << step.imbalance;
            std::cout << std::setprecision(6) << '\n';
          }
          code_diff += codes;
          scale_diff += scales;
        }
  Check(exact == tiles,
        std::string(stretch ? "stretched" : "27B-like") +
            " k4v4 token-major records tiles=" + std::to_string(tiles) +
            " exact=" + std::to_string(exact) +
            " code_diff=" + std::to_string(code_diff) +
            " scale_diff=" + std::to_string(scale_diff));
}

}  // namespace

int main() {
  try {
    CheckWriterRows();
    CheckRotateAndQuantize(false);
    CheckRotateAndQuantize(true);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return failures == 0 ? 0 : 1;
}
