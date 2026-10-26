// KVarN attention KV kernels (src/models/qwen/hip/kernels/attention_kvarn.hip)
// against host references: the rotating row write bit for bit, the
// group-flush quantizer bit for bit against the K0 reference rules
// (tests/models/qwen27b/kvarn_fake_quant.hpp), the fused-dequant decode and
// verification attention against double attention over the same stored
// codes, and the whole write -> flush -> attend path against attention over
// the K0 fake-quantized rows in the original frame.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/models/qwen/hip/kvarn_layout.hpp"
#include "tests/models/qwen27b/kvarn_fake_quant.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops/kvarn.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"

namespace {

namespace kvarn = gufo::testing::kvarn;
using gufo::hip::kKvarnFp16Tokens;
using gufo::hip::kKvarnGroupTokens;
using gufo::hip::KvarnFp16Row;
using gufo::hip::KvarnKvView;
using gufo::hip::KvarnRecord;
using gufo::test::DeviceBuffer;

constexpr unsigned kHeads = 24, kKvHeads = 4, kGqa = 6, kDim = 256;
constexpr unsigned kGroup = kKvarnGroupTokens, kRowWidth = kKvHeads * kDim;
constexpr unsigned kLayers = 2, kLayer = 1;
constexpr std::size_t kFp16LayerElements =
    static_cast<std::size_t>(kKvarnFp16Tokens) * kRowWidth;

int failures = 0;
void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << '\n';
  if (!ok)
    ++failures;
}

void ParallelFor(std::size_t n, const std::function<void(std::size_t)>& f) {
  const unsigned threads = std::max(1U, std::thread::hardware_concurrency());
  std::atomic<std::size_t> next{0};
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < threads; ++t)
    pool.emplace_back([&] {
      for (std::size_t i; (i = next++) < n;)
        f(i);
    });
  for (auto& thread : pool)
    thread.join();
}

// FP16 -> rotated FP16, the K0 reference's rotation of a stored row.
void RotateRow(const std::uint16_t* in, std::uint16_t* out) {
  std::vector<float> x(kDim);
  for (unsigned d = 0; d < kDim; ++d)
    x[d] = kvarn::HalfToFloat(in[d]);
  kvarn::Hadamard(x);
  for (unsigned d = 0; d < kDim; ++d)
    out[d] = kvarn::FloatToHalf(x[d]);
}

void HadamardDouble(double* x) {
  for (unsigned length = 1; length < kDim; length <<= 1)
    for (unsigned i = 0; i < kDim; i += length << 1)
      for (unsigned j = i; j < i + length; ++j) {
        const double a = x[j], b = x[j + length];
        x[j] = a + b;
        x[j + length] = a - b;
      }
  for (unsigned i = 0; i < kDim; ++i)
    x[i] /= 16.0;
}

// K rows with per-channel offsets and outlier channels, V rows plain.
struct Rows {
  std::vector<float> k, v;  // [token][kv][256]
};
Rows RandomRows(unsigned tokens, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal;
  std::vector<float> gain(kRowWidth), mean(kRowWidth);
  for (unsigned i = 0; i < kRowWidth; ++i) {
    const bool outlier = (i * 2654435761U >> 7) % 37 == 0;
    gain[i] = outlier ? 6.0F : 0.5F + 0.5F * std::abs(normal(rng));
    mean[i] = outlier ? 3.0F * normal(rng) : 0.3F * normal(rng);
  }
  Rows rows;
  rows.k.resize(static_cast<std::size_t>(tokens) * kRowWidth);
  rows.v.resize(rows.k.size());
  for (std::size_t i = 0; i < rows.k.size(); ++i) {
    rows.k[i] = normal(rng) * gain[i % kRowWidth] + mean[i % kRowWidth];
    rows.v[i] = normal(rng);
  }
  return rows;
}

std::vector<std::uint16_t> ToHalf(std::span<const float> x) {
  std::vector<std::uint16_t> out(x.size());
  for (std::size_t i = 0; i < x.size(); ++i)
    out[i] = kvarn::FloatToHalf(x[i]);
  return out;
}

// One sequence's KVarN storage on the device: the record plane of a pool
// whose blocks the table scatters, and the FP16 sink and tail plane.
struct Storage {
  int bits;
  unsigned pool_blocks;
  std::size_t record_row;
  std::vector<std::uint32_t> table;
  DeviceBuffer<std::uint8_t> records;
  DeviceBuffer<std::uint32_t> blocks;
  DeviceBuffer<std::uint16_t> k_rows, v_rows;

  Storage(int value_bits, unsigned logical_blocks)
      : bits(value_bits),
        pool_blocks(logical_blocks + 3),
        record_row(kKvHeads * KvarnRecord(value_bits).bytes),
        records(std::size_t{kLayers} * pool_blocks * 2 * record_row),
        k_rows(kLayers * kFp16LayerElements),
        v_rows(kLayers * kFp16LayerElements) {
    // Reversed, above the spare blocks, so a kernel that ignores the table
    // reads the wrong records.
    for (unsigned b = 0; b < logical_blocks; ++b)
      table.push_back(pool_blocks - 1 - b);
    std::vector<std::uint32_t> sorted = table;
    std::ranges::sort(sorted);
    if (std::ranges::adjacent_find(sorted) != sorted.end())
      throw std::logic_error("test block table repeats a block");
    blocks = DeviceBuffer<std::uint32_t>(table);
    HIP_CHECK(hipMemset(records.data(), 0x7b, records.size()));
    HIP_CHECK(hipMemset(k_rows.data(), 0, k_rows.size() * 2));
    HIP_CHECK(hipMemset(v_rows.data(), 0, v_rows.size() * 2));
  }
  [[nodiscard]] KvarnKvView View() {
    return {.value_bits = bits,
            .records = records.data(),
            .records_layer_pitch = std::size_t{pool_blocks} * 2 * record_row,
            .blocks = blocks.data(),
            .k_rows = k_rows.data(),
            .v_rows = v_rows.data()};
  }
  // Host offset of a group's record in one layer.
  [[nodiscard]] std::size_t RecordOffset(unsigned layer, unsigned group,
                                         unsigned kv) const {
    return layer * (std::size_t{pool_blocks} * 2 * record_row) +
           (std::size_t{table[group / 2]} * 2 + group % 2) * record_row +
           kv * KvarnRecord(bits).bytes;
  }
};

// A record -> logical tiles: K [256 channels][128 tokens], V [128][256].
kvarn::QuantizedTile Unpack(const std::uint8_t* record, int bits, bool key) {
  const auto r = KvarnRecord(bits);
  kvarn::QuantizedTile out;
  const auto halves = [&](std::size_t offset, std::size_t n) {
    std::vector<std::uint16_t> v(n);
    std::memcpy(v.data(), record + offset, 2 * n);
    return v;
  };
  if (key) {
    out.codes.resize(kDim * kGroup);
    const auto* words = reinterpret_cast<const std::uint32_t*>(record);
    for (unsigned t = 0; t < kGroup; ++t)
      for (unsigned c = 0; c < kDim; ++c) {
        const unsigned j = c % 8;
        const unsigned nibble = (j % 2 == 0) ? j / 2 : 4 + j / 2;
        out.codes[c * kGroup + t] =
            (words[t * 32 + c / 8] >> (4 * nibble)) & 15;
      }
    out.scale = halves(r.k_scale, kDim);
    out.zero = halves(r.k_zero, kDim);
    out.column = halves(r.k_column, kGroup);
  } else {
    out.codes.resize(kGroup * kDim);
    const auto* words =
        reinterpret_cast<const std::uint32_t*>(record + r.v_codes);
    for (unsigned c = 0; c < kDim; ++c)
      for (unsigned t = 0; t < kGroup; ++t) {
        unsigned code;
        if (bits == 2) {
          const unsigned u = t % 16;
          const unsigned crumb = (u % 2 == 0) ? u / 2 : 8 + u / 2;
          code = (words[c * 8 + t / 16] >> (2 * crumb)) & 3;
        } else {
          const unsigned u = t % 8;
          const unsigned nibble = (u % 2 == 0) ? u / 2 : 4 + u / 2;
          code = (words[c * 16 + t / 8] >> (4 * nibble)) & 15;
        }
        out.codes[t * kDim + c] = static_cast<std::uint8_t>(code);
      }
    out.scale = halves(r.v_scale, kGroup);
    out.zero = halves(r.v_zero, kGroup);
    out.column = halves(r.v_column, kDim);
  }
  return out;
}

// Dequantized [token 128][channel 256] rows (rotated frame) of a record.
std::vector<float> DequantRows(const std::uint8_t* record, int bits, bool key) {
  const auto t = Unpack(record, bits, key);
  std::vector<float> out(kGroup * kDim);
  const std::size_t cols = key ? kGroup : kDim;
  const std::size_t rows = key ? kDim : kGroup;
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t c = 0; c < cols; ++c) {
      const float value = (static_cast<float>(t.codes[r * cols + c]) *
                               kvarn::HalfToFloat(t.scale[r]) +
                           kvarn::HalfToFloat(t.zero[r])) *
                          kvarn::HalfToFloat(t.column[c]);
      const std::size_t token = key ? c : r, channel = key ? r : c;
      out[token * kDim + channel] = value;
    }
  return out;
}

// Causal gated attention in double; key/value(t, kv) give 256 floats.
std::vector<double> Attention(
    const std::vector<float>& q, const std::vector<float>& gate, unsigned rows,
    unsigned length, bool rotate,
    const std::function<const float*(unsigned, unsigned)>& key,
    const std::function<const float*(unsigned, unsigned)>& value,
    bool round_q = false) {
  std::vector<double> out(static_cast<std::size_t>(rows) * kHeads * kDim);
  ParallelFor(rows * kHeads, [&](std::size_t index) {
    const unsigned r = index / kHeads, head = index % kHeads, kv = head / kGqa;
    double qv[kDim];
    for (unsigned d = 0; d < kDim; ++d)
      qv[d] = q[index * kDim + d];
    if (rotate || round_q)
      HadamardDouble(qv);
    if (round_q) {
      // The kernel's FP16 query in the rotated frame.
      for (unsigned d = 0; d < kDim; ++d)
        qv[d] = kvarn::RoundHalf(static_cast<float>(qv[d]));
      if (!rotate)
        HadamardDouble(qv);
    }
    const unsigned last = length - rows + r;
    std::vector<double> s(last + 1);
    double mx = -INFINITY;
    for (unsigned t = 0; t <= last; ++t) {
      const float* k = key(t, kv);
      double dot = 0;
      for (unsigned d = 0; d < kDim; ++d)
        dot += qv[d] * k[d];
      s[t] = dot / 16.0;
      mx = std::max(mx, s[t]);
    }
    double sum = 0, o[kDim] = {};
    for (unsigned t = 0; t <= last; ++t) {
      const double p = std::exp(s[t] - mx);
      sum += p;
      const float* v = value(t, kv);
      for (unsigned d = 0; d < kDim; ++d)
        o[d] += p * v[d];
    }
    for (unsigned d = 0; d < kDim; ++d)
      o[d] /= sum;
    if (rotate)
      HadamardDouble(o);
    for (unsigned d = 0; d < kDim; ++d)
      out[index * kDim + d] =
          o[d] / (1.0 + std::exp(-static_cast<double>(gate[index * kDim + d])));
  });
  return out;
}

double RelativeError(std::span<const float> got,
                     const std::vector<double>& ref) {
  double num = 0, den = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - ref[i];
    if (!std::isfinite(d))
      return INFINITY;
    num += d * d;
    den += ref[i] * ref[i];
  }
  return std::sqrt(num / den);
}

std::vector<float> RandomFloats(std::size_t n, unsigned seed, float sigma) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0F, sigma);
  std::vector<float> x(n);
  for (auto& v : x)
    v = normal(rng);
  return x;
}

// Write: rows across the sink/ring boundary and the ring wrap, with host
// and device positions, are the K0 rotation of the FP16 rows bit for bit.
void TestWrite() {
  Storage storage(2, 4);
  struct Case {
    unsigned start, rows;
    bool device_position;
  };
  for (const Case c :
       {Case{0, 1, false}, Case{124, 8, false}, Case{380, 8, true},
        Case{511, 1, true}, Case{256, 129, false}}) {
    const auto rows = RandomRows(c.rows, 17 + c.start);
    DeviceBuffer<float> k(rows.k), v(rows.v);
    DeviceBuffer<std::uint32_t> pos(std::vector<std::uint32_t>{c.start});
    gufo::hip::LaunchKvarnWriteRows(k.data(), v.data(), storage.View(), kLayer,
                                    c.device_position ? pos.data() : nullptr,
                                    c.device_position ? 0 : c.start, c.rows,
                                    nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    const auto kh = storage.k_rows.CopyToHost();
    const auto vh = storage.v_rows.CopyToHost();
    std::size_t mismatches = 0;
    for (unsigned r = 0; r < c.rows; ++r)
      for (unsigned kv = 0; kv < kKvHeads; ++kv)
        for (const bool key : {true, false}) {
          const auto& src = key ? rows.k : rows.v;
          const auto half = ToHalf(std::span(src).subspan(
              (static_cast<std::size_t>(r) * kKvHeads + kv) * kDim, kDim));
          std::uint16_t expected[kDim];
          RotateRow(half.data(), expected);
          const auto& got = key ? kh : vh;
          const std::size_t base =
              kLayer * kFp16LayerElements +
              static_cast<std::size_t>(KvarnFp16Row(c.start + r)) * kRowWidth +
              kv * kDim;
          for (unsigned d = 0; d < kDim; ++d)
            mismatches += got[base + d] != expected[d];
        }
    Check(mismatches == 0, "write start=" + std::to_string(c.start) +
                               " rows=" + std::to_string(c.rows) +
                               (c.device_position ? " device" : " host") +
                               " mismatches=" + std::to_string(mismatches));
  }
}

// Flush: rounds of two groups through the ring (groups 1-2 ... 7-8), every
// layer, against the K0 reference's codes and FP16 scales bit for bit.
void TestFlush(int bits) {
  constexpr unsigned kRounds = 4;
  Storage storage(bits, kRounds + 1);
  std::size_t tiles = 0, exact = 0, code_diff = 0, scale_diff = 0;
  for (unsigned round = 0; round < kRounds; ++round) {
    const unsigned first_group = 1 + 2 * round;
    const unsigned first = first_group * kGroup;
    // Rotated FP16 rows of both groups in every layer, through the kernel.
    std::vector<Rows> layer_rows;
    for (unsigned layer = 0; layer < kLayers; ++layer) {
      layer_rows.push_back(
          RandomRows(2 * kGroup, 1000 * bits + 10 * round + layer));
      DeviceBuffer<float> k(layer_rows.back().k), v(layer_rows.back().v);
      for (unsigned chunk = 0; chunk < 2; ++chunk)
        gufo::hip::LaunchKvarnWriteRows(
            k.data() + chunk * kGroup * kRowWidth,
            v.data() + chunk * kGroup * kRowWidth, storage.View(), layer,
            nullptr, first + chunk * kGroup, kGroup, nullptr);
      HIP_CHECK(hipDeviceSynchronize());
    }
    gufo::hip::LaunchKvarnFlush(storage.View(), kLayers, first_group, 2,
                                nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    const auto records = storage.records.CopyToHost();
    std::vector<std::size_t> tile_exact(kLayers * 2 * kKvHeads * 2);
    std::vector<std::size_t> tile_codes(tile_exact.size()),
        tile_scales(tile_exact.size());
    ParallelFor(tile_exact.size(), [&](std::size_t index) {
      const bool key = index % 2 == 0;
      const unsigned kv = (index / 2) % kKvHeads;
      const unsigned g = (index / 2 / kKvHeads) % 2;
      const unsigned layer = index / 2 / kKvHeads / 2;
      const auto& src = key ? layer_rows[layer].k : layer_rows[layer].v;
      std::vector<float> tile(kDim * kGroup);
      for (unsigned t = 0; t < kGroup; ++t) {
        const auto half = ToHalf(std::span(src).subspan(
            (static_cast<std::size_t>(g * kGroup + t) * kKvHeads + kv) * kDim,
            kDim));
        std::uint16_t rotated[kDim];
        RotateRow(half.data(), rotated);
        for (unsigned d = 0; d < kDim; ++d)
          tile[key ? d * kGroup + t : t * kDim + d] =
              kvarn::HalfToFloat(rotated[d]);
      }
      const auto ref = kvarn::QuantizeTileCodes(tile, key ? kDim : kGroup,
                                                key ? kGroup : kDim,
                                                key ? 4 : bits, true, 8);
      const auto got = Unpack(
          records.data() + storage.RecordOffset(layer, first_group + g, kv),
          bits, key);
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
      tile_codes[index] = codes;
      tile_scales[index] = scales;
      tile_exact[index] = codes == 0 && scales == 0;
    });
    for (std::size_t i = 0; i < tile_exact.size(); ++i) {
      ++tiles;
      exact += tile_exact[i];
      code_diff += tile_codes[i];
      scale_diff += tile_scales[i];
    }
  }
  Check(exact == tiles, "flush k4v" + std::to_string(bits) +
                            " tiles=" + std::to_string(tiles) +
                            " exact=" + std::to_string(exact) +
                            " code_diff=" + std::to_string(code_diff) +
                            " scale_diff=" + std::to_string(scale_diff));
}

// Attention over random records and FP16 rows: kernel arithmetic against
// double attention over the same stored values, rotated frame.
void TestAttention(int bits, unsigned groups, unsigned tail, unsigned rows,
                   unsigned splits, bool device_position) {
  // Without quantized groups `tail` is the whole length.
  const unsigned length = groups == 0 ? tail : kGroup * (1 + groups) + tail;
  if (length < rows)
    throw std::logic_error("attention case shorter than its rows");
  const unsigned logical_blocks = (1 + groups) / 2 + 1;
  Storage storage(bits, logical_blocks);
  const auto layout = KvarnRecord(bits);
  // Random records: codes anywhere, scales so values stay near unit size.
  {
    std::vector<std::uint8_t> records(storage.records.size());
    std::mt19937 rng(groups * 31 + tail + bits);
    for (std::size_t i = 0; i + 4 <= records.size(); i += 4) {
      const std::uint32_t word = rng();
      std::memcpy(records.data() + i, &word, 4);
    }
    std::uniform_real_distribution<float> uniform(0.0F, 1.0F);
    for (unsigned g = 1; g <= groups; ++g)
      for (unsigned kv = 0; kv < kKvHeads; ++kv) {
        std::uint8_t* rec =
            records.data() + storage.RecordOffset(kLayer, g, kv);
        auto put = [&](std::size_t offset, unsigned n, float lo, float hi) {
          for (unsigned i = 0; i < n; ++i) {
            const auto h = kvarn::FloatToHalf(lo + (hi - lo) * uniform(rng));
            std::memcpy(rec + offset + 2 * i, &h, 2);
          }
        };
        put(layout.k_scale, kDim, 0.05F, 0.3F);
        put(layout.k_zero, kDim, -2.0F, -0.5F);
        put(layout.k_column, kGroup, 0.5F, 1.5F);
        const float qmax = static_cast<float>((1 << bits) - 1);
        put(layout.v_scale, kGroup, 0.5F / qmax, 2.0F / qmax);
        put(layout.v_zero, kGroup, -1.2F, -0.3F);
        put(layout.v_column, kDim, 0.5F, 1.5F);
      }
    storage.records.CopyFrom(records);
  }
  // FP16 sink and tail rows, already in the rotated frame.
  std::vector<std::uint16_t> kh(storage.k_rows.size()),
      vh(storage.v_rows.size());
  {
    const auto k = RandomFloats(kh.size(), 5 + tail, 1.0F);
    const auto v = RandomFloats(vh.size(), 6 + tail, 1.0F);
    kh = ToHalf(k);
    vh = ToHalf(v);
    storage.k_rows.CopyFrom(kh);
    storage.v_rows.CopyFrom(vh);
  }
  const auto records = storage.records.CopyToHost();
  // Dequantized rows per group in [token][kv][256].
  std::vector<std::vector<float>> kq(groups), vq(groups);
  for (unsigned g = 0; g < groups; ++g) {
    kq[g].resize(kGroup * kRowWidth);
    vq[g].resize(kGroup * kRowWidth);
  }
  ParallelFor(static_cast<std::size_t>(groups) * kKvHeads * 2,
              [&](std::size_t index) {
                const bool key = index % 2 == 0;
                const unsigned kv = (index / 2) % kKvHeads;
                const unsigned g = index / 2 / kKvHeads;
                const auto rows_kv = DequantRows(
                    records.data() + storage.RecordOffset(kLayer, g + 1, kv),
                    bits, key);
                auto& dst = key ? kq[g] : vq[g];
                for (unsigned t = 0; t < kGroup; ++t)
                  std::copy_n(rows_kv.data() + t * kDim, kDim,
                              dst.data() + (t * kKvHeads + kv) * kDim);
              });
  std::vector<float> kf(kh.size()), vf(vh.size());
  for (std::size_t i = 0; i < kh.size(); ++i) {
    kf[i] = kvarn::HalfToFloat(kh[i]);
    vf[i] = kvarn::HalfToFloat(vh[i]);
  }
  const unsigned flushed_end = kGroup * (1 + groups);
  const auto row_of = [&](const std::vector<float>& fp16,
                          const std::vector<std::vector<float>>& quantized,
                          unsigned t, unsigned kv) -> const float* {
    if (t >= kGroup && t < flushed_end) {
      const unsigned g = t / kGroup - 1;
      return quantized[g].data() + ((t % kGroup) * kKvHeads + kv) * kDim;
    }
    return fp16.data() + kLayer * kFp16LayerElements +
           static_cast<std::size_t>(KvarnFp16Row(t)) * kRowWidth + kv * kDim;
  };
  const auto q =
      RandomFloats(std::size_t{rows} * kHeads * kDim, 9 + rows, 1.0F);
  const auto gate = RandomFloats(q.size(), 10 + rows, 1.0F);
  const auto ref = Attention(
      q, gate, rows, length, true,
      [&](unsigned t, unsigned kv) { return row_of(kf, kq, t, kv); },
      [&](unsigned t, unsigned kv) { return row_of(vf, vq, t, kv); });
  // The same with the query rounded to FP16 as the kernel holds it: how
  // much of the kernel's error that rounding explains.
  const auto ref_q16 = Attention(
      q, gate, rows, length, true,
      [&](unsigned t, unsigned kv) { return row_of(kf, kq, t, kv); },
      [&](unsigned t, unsigned kv) { return row_of(vf, vq, t, kv); }, true);

  const unsigned used_splits =
      splits != 0 ? splits : gufo::hip::SelectKvarnDecodeSplits(groups, bits);
  DeviceBuffer<float> dq(q), dgate(gate), out(q.size());
  DeviceBuffer<float> scratch(
      gufo::hip::KvarnScratchElements(rows, used_splits));
  // Stale scratch must not leak into the result.
  HIP_CHECK(hipMemset(scratch.data(), 0xff, scratch.size() * sizeof(float)));
  DeviceBuffer<std::uint32_t> pos(std::vector<std::uint32_t>{length - rows});
  gufo::hip::LaunchKvarnDecodeAttention(
      dq.data(), dgate.data(), storage.View(), out.data(), kLayer,
      device_position ? pos.data() : nullptr,
      device_position ? 0 : length - rows, rows, groups, splits,
      {scratch.data(), scratch.size()}, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  const auto got = out.CopyToHost();
  const double error = RelativeError(got, ref);
  char line[256];
  std::snprintf(line, sizeof(line),
                "attention k4v%d groups=%u tail=%u length=%u rows=%u "
                "splits=%u%s rel=%.3e rel_vs_q16_ref=%.3e",
                bits, groups, tail, length, rows, used_splits,
                device_position ? " device" : "", error,
                RelativeError(got, ref_q16));
  Check(error < 2e-3, line);
}

// One pass as the arena drives it: plan, quantize the groups before it,
// write every layer's rows, quantize a group that ends inside it.
gufo::hip::KvarnPassPlan Pass(Storage& storage, gufo::hip::KvarnTail& tail,
                              const DeviceBuffer<float>& k,
                              const DeviceBuffer<float>& v, unsigned start,
                              unsigned n) {
  const auto plan = gufo::hip::PlanKvarnPass(tail, start, n);
  if (!plan.error.empty())
    throw std::logic_error(std::string(plan.error));
  gufo::hip::LaunchKvarnFlush(storage.View(), kLayers, plan.first_group,
                              plan.groups, nullptr);
  tail.flushed_end = plan.flushed_end;
  tail.written_end = std::max(tail.written_end, start + n);
  for (unsigned layer = 0; layer < kLayers; ++layer)
    gufo::hip::LaunchKvarnWriteRows(k.data() + std::size_t{start} * kRowWidth,
                                    v.data() + std::size_t{start} * kRowWidth,
                                    storage.View(), layer, nullptr, start, n,
                                    nullptr);
  if (plan.boundary != 0)
    gufo::hip::LaunchKvarnFlush(storage.View(), kLayers,
                                plan.boundary / kGroup - 1, 1, nullptr);
  return plan;
}

// Positions [from, to) in passes of up to 100 rows.
void Feed(Storage& storage, gufo::hip::KvarnTail& tail,
          const DeviceBuffer<float>& k, const DeviceBuffer<float>& v,
          unsigned from, unsigned to) {
  for (unsigned start = from; start < to;) {
    const unsigned n = std::min(100U, to - start);
    (void)Pass(storage, tail, k, v, start, n);
    start += n;
  }
}

// Write -> flush -> attend as the arena drives it, against attention over
// the K0 fake-quantized FP16 rows in the original frame.
void TestEndToEnd(int bits, unsigned length, unsigned rows) {
  const unsigned logical_blocks = length / 256 + 1;
  Storage storage(bits, logical_blocks);
  const auto all = RandomRows(length, 77 + length + bits);
  gufo::hip::KvarnTail tail;
  DeviceBuffer<float> k(all.k), v(all.v);
  Feed(storage, tail, k, v, 0, length - rows);
  const auto plan = Pass(storage, tail, k, v, length - rows, rows);
  if (plan.boundary != 0)
    throw std::logic_error("end-to-end case crosses a group end");
  const std::uint32_t groups = plan.read_groups;
  const auto config =
      *kvarn::ParseConfig(bits == 2 ? "kvarn_k4v2_g128" : "kvarn_k4v4_g128");
  auto kh = ToHalf(all.k), vh = ToHalf(all.v);
  kvarn::FakeQuantize(kh, kKvHeads, kDim, true, config, 1, groups + 1);
  kvarn::FakeQuantize(vh, kKvHeads, kDim, false, config, 1, groups + 1);
  std::vector<float> kf(kh.size()), vf(vh.size());
  for (std::size_t i = 0; i < kh.size(); ++i) {
    kf[i] = kvarn::HalfToFloat(kh[i]);
    vf[i] = kvarn::HalfToFloat(vh[i]);
  }
  const auto q = RandomFloats(std::size_t{rows} * kHeads * kDim, 3, 1.0F);
  const auto gate = RandomFloats(q.size(), 4, 1.0F);
  const auto ref = Attention(
      q, gate, rows, length, false,
      [&](unsigned t, unsigned kv) {
        return kf.data() + (std::size_t{t} * kKvHeads + kv) * kDim;
      },
      [&](unsigned t, unsigned kv) {
        return vf.data() + (std::size_t{t} * kKvHeads + kv) * kDim;
      });
  const auto ref_q16 = Attention(
      q, gate, rows, length, false,
      [&](unsigned t, unsigned kv) {
        return kf.data() + (std::size_t{t} * kKvHeads + kv) * kDim;
      },
      [&](unsigned t, unsigned kv) {
        return vf.data() + (std::size_t{t} * kKvHeads + kv) * kDim;
      },
      true);
  DeviceBuffer<float> dq(q), dgate(gate), out(q.size());
  DeviceBuffer<float> scratch(gufo::hip::KvarnScratchElements(
      rows, gufo::hip::SelectKvarnDecodeSplits(groups, bits)));
  gufo::hip::LaunchKvarnDecodeAttention(
      dq.data(), dgate.data(), storage.View(), out.data(), kLayer, nullptr,
      length - rows, rows, groups, 0, {scratch.data(), scratch.size()},
      nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  const auto got = out.CopyToHost();
  const double error = RelativeError(got, ref);
  char line[256];
  std::snprintf(line, sizeof(line),
                "end-to-end k4v%d length=%u groups=%u rows=%u "
                "vs_k0_fake_quant_rel=%.3e vs_q16_k0_fake_quant_rel=%.3e",
                bits, length, groups, rows, error, RelativeError(got, ref_q16));
  Check(error < 3e-3, line);
}

// Verification rows equal decode steps bit for bit: one pass of `rows`
// rows after `start` history rows (LaunchKvarnPassAttention, which
// quantizes a group ending inside the pass), against one pass per row.
void TestVerifyMatchesDecode(int bits, unsigned start, unsigned rows) {
  const unsigned length = start + rows;
  const auto all = RandomRows(length, 91 + length + bits);
  DeviceBuffer<float> k(all.k), v(all.v);
  const std::size_t width = std::size_t{kHeads} * kDim;
  const auto q = RandomFloats(rows * width, 13 + start, 1.0F);
  const auto gate = RandomFloats(q.size(), 14 + start, 1.0F);
  DeviceBuffer<float> dq(q), dgate(gate);
  DeviceBuffer<float> scratch(
      gufo::hip::KvarnScratchElements(rows, gufo::hip::kKvarnMaxSplits));
  const std::span<float> scratch_span{scratch.data(), scratch.size()};

  Storage verify_storage(bits, length / 256 + 1);
  gufo::hip::KvarnTail verify_tail;
  Feed(verify_storage, verify_tail, k, v, 0, start);
  const auto plan = gufo::hip::PlanKvarnPass(verify_tail, start, rows);
  if (!plan.error.empty())
    throw std::logic_error(std::string(plan.error));
  gufo::hip::LaunchKvarnFlush(verify_storage.View(), kLayers, plan.first_group,
                              plan.groups, nullptr);
  DeviceBuffer<float> verify_out(q.size());
  for (unsigned layer = 0; layer < kLayers; ++layer)
    gufo::hip::LaunchKvarnPassAttention(
        dq.data(), k.data() + std::size_t{start} * kRowWidth,
        v.data() + std::size_t{start} * kRowWidth, dgate.data(),
        verify_storage.View(), verify_out.data(), layer, start, rows, plan,
        scratch_span, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  const auto verified = verify_out.CopyToHost();
  const auto verify_partials = scratch.CopyToHost();
  const unsigned verify_rows =
      plan.boundary != 0 ? start + rows - plan.boundary : rows;

  Storage decode_storage(bits, length / 256 + 1);
  gufo::hip::KvarnTail decode_tail;
  Feed(decode_storage, decode_tail, k, v, 0, start);
  DeviceBuffer<float> decode_out(q.size());
  std::vector<float> decode_partials;
  for (unsigned r = 0; r < rows; ++r) {
    const auto step = Pass(decode_storage, decode_tail, k, v, start + r, 1);
    gufo::hip::LaunchKvarnDecodeAttention(
        dq.data() + r * width, dgate.data() + r * width, decode_storage.View(),
        decode_out.data() + r * width, kLayer, nullptr, start + r, 1,
        step.read_groups, 0, scratch_span, nullptr);
    if (r == 0) {
      HIP_CHECK(hipDeviceSynchronize());
      decode_partials = scratch.CopyToHost();
    }
  }
  HIP_CHECK(hipDeviceSynchronize());
  const auto decoded = decode_out.CopyToHost();
  if (plan.boundary == 0) {
    // Where the first row's decode partials differ from its verification
    // row's (diagnostic): [kv][split][row m][max, sum, 256 outputs].
    const unsigned r = 0, vr = 0;
    const unsigned groups = gufo::hip::KvarnReadGroups(start + r);
    const unsigned nsplit =
        gufo::hip::SelectKvarnDecodeSplits(groups, bits) + 2;
    std::string where;
    for (unsigned kv = 0; kv < kKvHeads && where.size() < 400; ++kv)
      for (unsigned s = 0; s < nsplit; ++s)
        for (unsigned h = 0; h < kGqa; ++h) {
          const float* d =
              decode_partials.data() + ((kv * nsplit + s) * kGqa + h) * 258;
          const float* w =
              verify_partials.data() +
              ((kv * nsplit + s) * kGqa * verify_rows + vr * kGqa + h) * 258;
          unsigned first = 258;
          for (unsigned i = 0; i < 258; ++i)
            if (std::memcmp(d + i, w + i, 4) != 0) {
              first = i;
              break;
            }
          if (first != 258)
            where += " kv" + std::to_string(kv) + "/s" + std::to_string(s) +
                     "/h" + std::to_string(h) + "@" + std::to_string(first);
        }
    std::cout << "partials last row k4v" << bits << " start=" << start
              << " splits=" << nsplit - 2 << " differ:" << where << '\n';
  }
  unsigned differing_rows = 0;
  float max_abs = 0;
  std::string which;
  for (unsigned r = 0; r < rows; ++r) {
    bool same = true;
    for (std::size_t i = r * width; i < (r + 1) * width; ++i) {
      if (std::memcmp(&verified[i], &decoded[i], sizeof(float)) != 0)
        same = false;
      max_abs = std::max(max_abs, std::abs(verified[i] - decoded[i]));
    }
    differing_rows += !same;
    if (!same)
      which += " " + std::to_string(r);
  }
  char line[320];
  std::snprintf(line, sizeof(line),
                "verify==decode k4v%d start=%u rows=%u boundary=%u "
                "differing_rows=%u [%s ] max_abs=%.3e",
                bits, start, rows, plan.boundary, differing_rows, which.c_str(),
                max_abs);
  Check(differing_rows == 0, line);
}

// `time`: microseconds per attention launch at the production split policy,
// for comparison with the K1b prototype sweep (KVARN.md). Inputs cycle
// through copies of at least 256 MiB, past the 32 MiB MALL, and the timed
// loop alternates formats. Values are random; only the time is reported.
void TimeAttention() {
  struct Shape {
    unsigned depth, rows;
  };
  constexpr Shape kShapes[] = {{4096, 1},   {4096, 8},  {32768, 1},
                               {32768, 8},  {65536, 1}, {65536, 8},
                               {262144, 1}, {262144, 8}};
  for (const auto shape : kShapes) {
    for (const int bits : {2, 4}) {
      const unsigned groups = shape.depth / kGroup - 2;
      const unsigned blocks = shape.depth / 256 + 1;
      const std::size_t record_row = kKvHeads * KvarnRecord(bits).bytes;
      const std::size_t layer_pitch = std::size_t{blocks} * 2 * record_row;
      const std::size_t bytes = groups * record_row;
      const unsigned copies = static_cast<unsigned>(
          std::max<std::size_t>(1, (256ULL << 20) / bytes + 1));
      DeviceBuffer<std::uint8_t> records(copies * layer_pitch);
      HIP_CHECK(hipMemset(records.data(), 0x11, records.size()));
      std::vector<std::uint32_t> table(blocks);
      for (unsigned b = 0; b < blocks; ++b)
        table[b] = b;
      DeviceBuffer<std::uint32_t> dtable(table);
      DeviceBuffer<std::uint16_t> rows_k(copies * kFp16LayerElements),
          rows_v(copies * kFp16LayerElements);
      HIP_CHECK(hipMemset(rows_k.data(), 0, rows_k.size() * 2));
      HIP_CHECK(hipMemset(rows_v.data(), 0, rows_v.size() * 2));
      const KvarnKvView view{.value_bits = bits,
                             .records = records.data(),
                             .records_layer_pitch = layer_pitch,
                             .blocks = dtable.data(),
                             .k_rows = rows_k.data(),
                             .v_rows = rows_v.data()};
      const auto q =
          RandomFloats(std::size_t{shape.rows} * kHeads * kDim, 1, 1.0F);
      DeviceBuffer<float> dq(q), out(q.size());
      const unsigned splits = gufo::hip::SelectKvarnDecodeSplits(groups, bits);
      DeviceBuffer<float> scratch(
          gufo::hip::KvarnScratchElements(shape.rows, splits));
      const unsigned start = shape.depth - shape.rows;
      const auto launch = [&](unsigned copy) {
        gufo::hip::LaunchKvarnDecodeAttention(
            dq.data(), nullptr, view, out.data(), copy, nullptr, start,
            shape.rows, groups, 0, {scratch.data(), scratch.size()}, nullptr);
      };
      for (unsigned i = 0; i < copies; ++i)
        launch(i);
      HIP_CHECK(hipDeviceSynchronize());
      const unsigned iterations = std::max(copies, 20U);
      hipEvent_t begin, end;
      HIP_CHECK(hipEventCreate(&begin));
      HIP_CHECK(hipEventCreate(&end));
      HIP_CHECK(hipEventRecord(begin, nullptr));
      for (unsigned i = 0; i < iterations; ++i)
        launch(i % copies);
      HIP_CHECK(hipEventRecord(end, nullptr));
      HIP_CHECK(hipEventSynchronize(end));
      float ms = 0;
      HIP_CHECK(hipEventElapsedTime(&ms, begin, end));
      HIP_CHECK(hipEventDestroy(begin));
      HIP_CHECK(hipEventDestroy(end));
      std::printf(
          "time depth=%u rows=%u fmt=k4v%d splits=%u us=%.1f MiB=%.2f "
          "copies=%u\n",
          shape.depth, shape.rows, bits, splits, 1000.0 * ms / iterations,
          static_cast<double>(bytes) / (1 << 20), copies);
    }
  }
}

}  // namespace
#endif

int main(int argc, char** argv) {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kOptional, "Qwen KVarN ops test");
  if (device_status != 0)
    return device_status;
  if (argc > 1 && std::string_view(argv[1]) == "time") {
    TimeAttention();
    return 0;
  }
  if (argc > 1 && std::string_view(argv[1]) == "verify") {
    for (const int bits : {2, 4})
      for (const auto [start, rows] :
           {std::pair{100U, 8U}, std::pair{700U, 8U}, std::pair{636U, 8U},
            std::pair{254U, 3U}})
        TestVerifyMatchesDecode(bits, start, rows);
    return failures == 0 ? 0 : 1;
  }
  try {
    TestWrite();
    for (const int bits : {2, 4}) {
      TestFlush(bits);
      struct Case {
        unsigned groups, tail, rows, splits;
        bool device_position;
      };
      for (const Case c : {
               Case{0, 50, 1, 0, true},    // inside the sink
               Case{0, 178, 8, 0, false},  // sink and tail
               Case{0, 128, 3, 0, false},  // the sink exactly
               Case{1, 44, 1, 0, false},   // one group
               Case{1, 44, 5, 1, false},
               Case{30, 100, 1, 0, false},  // 4K
               Case{30, 100, 2, 7, false},
               Case{30, 100, 8, 30, false},
               Case{254, 133, 1, 0, false},  // 32K, a two-chunk tail
               Case{254, 133, 8, 0, false},
               Case{254, 133, 4, 38, false},
               Case{254, 7, 7, 0, true},      // queries start at the tail
               Case{1023, 120, 1, 0, false},  // 128K, the large split
               Case{1023, 120, 8, 0, false},
               Case{2047, 6, 6, 0, false},  // 256K
           })
        TestAttention(bits, c.groups, c.tail, c.rows, c.splits,
                      c.device_position);
      TestEndToEnd(bits, 1000, 1);
      TestEndToEnd(bits, 1000, 8);
      TestVerifyMatchesDecode(bits, 100, 8);    // the sink
      TestVerifyMatchesDecode(bits, 700, 8);    // inside a group
      TestVerifyMatchesDecode(bits, 636, 8);    // across the end of group 4
      TestVerifyMatchesDecode(bits, 254, 3);    // across the end of group 1
      TestVerifyMatchesDecode(bits, 16380, 8);  // 38 splits for K4V4
    }
  } catch (const std::exception& error) {
    std::cerr << "Qwen KVarN ops test: " << error.what() << '\n';
    return 1;
  }
  if (failures != 0) {
    std::cerr << "Qwen KVarN ops test: " << failures << " checks failed\n";
    return 1;
  }
  std::cout << "Qwen KVarN ops tests passed.\n";
#else
  (void)argc;
  (void)argv;
#endif
  return 0;
}
