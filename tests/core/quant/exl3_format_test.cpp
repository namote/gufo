// EXL3 CPU reference (src/core/quant/exl3/format.hpp) self-checks:
// FP16 rounding, the trellis window against a naive bit reader, the tile
// permutation, codebook values against an independent Python emulation
// (struct 'e' rounding), the Hadamard rotation and the reference GEMV against
// an explicit rotated weight. Bit-exactness against the upstream runtime is
// tests/core/quant/exl3_upstream_test.cpp.

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "src/core/quant/exl3/checkpoint.hpp"
#include "src/core/quant/exl3/format.hpp"
#include "src/core/quant/exl3/layout.hpp"

namespace {

using namespace gufo::quant::exl3;

int failures = 0;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

void TestHalfRoundTrip() {
  for (std::uint32_t bits = 0; bits < 0x10000; ++bits) {
    const auto h = static_cast<std::uint16_t>(bits);
    const double value = HalfToDouble(h);
    if (std::isnan(value)) {
      continue;
    }
    if (HalfFromDouble(value) != h) {
      Expect(false, "half round trip " + std::to_string(bits));
      return;
    }
  }
  // Ties to even: 1 + 2^-11 lies halfway between 1 and 1 + 2^-10.
  Expect(HalfFromDouble(1.0 + std::ldexp(1.0, -11)) == 0x3c00,
         "tie to even down");
  Expect(HalfFromDouble(1.0 + 3 * std::ldexp(1.0, -11)) == 0x3c02,
         "tie to even up");
  Expect(HalfFromDouble(std::ldexp(1.0, -25)) == 0x0000,
         "smallest tie to zero");
  Expect(HalfFromDouble(3 * std::ldexp(1.0, -25)) == 0x0002,
         "subnormal tie up");
  Expect(HalfFromDouble(65519.0) == 0x7bff, "max finite");
  Expect(HalfFromDouble(65520.0) == 0x7c00, "overflow tie to infinity");
}

// Bit j of the tile string: MSB-first within little-endian uint32 words.
int NaiveBit(const std::vector<std::uint16_t>& tile, int j) {
  const int word = j / 32;
  const std::uint32_t w =
      static_cast<std::uint32_t>(tile[2 * word]) |
      (static_cast<std::uint32_t>(tile[2 * word + 1]) << 16);
  return static_cast<int>((w >> (31 - (j % 32))) & 1u);
}

std::uint16_t NaiveState(const std::vector<std::uint16_t>& tile, BitRate rate,
                         int position) {
  const int total = rate.TileBits();
  // End bit from summing per-position widths (independent of StateEndBit).
  int end = 0;
  for (int p = 0; p <= position; ++p) {
    end += rate.half ? (p % 2 == 0 ? rate.bits : rate.bits + 1) : rate.bits;
  }
  std::uint16_t v = 0;
  for (int q = 0; q < 16; ++q) {
    const int j = ((end - 16 + q) % total + total) % total;
    v = static_cast<std::uint16_t>((v << 1) | NaiveBit(tile, j));
  }
  return v;
}

void TestTileState() {
  std::mt19937 rng(20261030);
  std::vector<BitRate> rates;
  for (int b = 1; b <= 8; ++b)
    rates.push_back({b, false});
  for (int b = 1; b <= 3; ++b)
    rates.push_back({b, true});
  for (const auto rate : rates) {
    Expect(BitRateFromTileWords(rate.TileWords()) == rate, "rate from words");
    Expect(StateEndBit(rate, 255) == rate.TileBits(),
           "last position ends the tile");
    for (int trial = 0; trial < 20; ++trial) {
      std::vector<std::uint16_t> tile(rate.TileWords());
      for (auto& w : tile)
        w = static_cast<std::uint16_t>(rng());
      for (int p = 0; p < kTileSize; ++p) {
        if (TileState(tile.data(), rate, p) != NaiveState(tile, rate, p)) {
          Expect(false, "tile state K=" + std::to_string(rate.bits) +
                            (rate.half ? ".5" : "") +
                            " p=" + std::to_string(p));
          return;
        }
      }
    }
  }
  Expect(!BitRateFromTileWords(0), "rate 0 rejected");
  Expect(!BitRateFromTileWords(16 * 9), "rate 9 rejected");
  Expect(!BitRateFromTileWords(16 * 4 + 8), "rate 4.5 rejected");
  Expect(!Supports(Codebook::kMcg, {2, true}), "half rate needs mul1");
}

void TestPermutation() {
  const auto& perm = TilePositionToElement();
  std::set<int> seen(perm.begin(), perm.end());
  Expect(seen.size() == kTileSize, "permutation is a bijection");
  // Spot values from the upstream formula: p = 0 -> (0, 0), 1 -> (1, 0),
  // 2 -> (8, 0), 4 -> (0, 8), 8 -> (2, 0), 32 -> (0, 1), 255 -> (15, 15).
  Expect(perm[0] == 0 && perm[1] == 16 && perm[2] == 128 && perm[4] == 8,
         "permutation first lane");
  Expect(perm[8] == 32 && perm[32] == 1 && perm[255] == 255,
         "permutation later lanes");
}

void TestCodebooks() {
  // Expected FP16 bits from an independent Python emulation (struct 'e').
  struct Sample {
    Codebook cb;
    std::uint16_t state, bits;
  };
  const Sample samples[] = {
      {Codebook::k3Inst, 0, 0x3a25},      {Codebook::k3Inst, 1, 0xbb5b},
      {Codebook::k3Inst, 255, 0x3cd1},    {Codebook::k3Inst, 0x1234, 0xc066},
      {Codebook::k3Inst, 0xffff, 0xb110}, {Codebook::kMcg, 0, 0x3f60},
      {Codebook::kMcg, 1, 0x304e},        {Codebook::kMcg, 0x1234, 0x3e0f},
      {Codebook::kMcg, 0xabcd, 0x35e7},   {Codebook::kMcg, 0xffff, 0x3acd},
      {Codebook::kMul1, 0, 0xc2e8},       {Codebook::kMul1, 1, 0x3921},
      {Codebook::kMul1, 255, 0x3ab3},     {Codebook::kMul1, 0x8000, 0xbaf2},
      {Codebook::kMul1, 0xffff, 0xb936},
  };
  for (const auto& s : samples) {
    Expect(DecodeState(s.state, s.cb) == s.bits,
           "codebook sample state " + std::to_string(s.state));
  }
  // FNV-1a 64 of all 3 x 65536 outputs (little-endian u16), same emulation.
  std::uint64_t h = 0xcbf29ce484222325ull;
  for (const auto cb : {Codebook::k3Inst, Codebook::kMcg, Codebook::kMul1}) {
    for (std::uint32_t s = 0; s < 0x10000; ++s) {
      const std::uint16_t v = DecodeState(static_cast<std::uint16_t>(s), cb);
      for (const std::uint8_t byte : {static_cast<std::uint8_t>(v & 0xff),
                                      static_cast<std::uint8_t>(v >> 8)}) {
        h ^= byte;
        h *= 0x100000001b3ull;
      }
    }
  }
  Expect(h == 0x93d38569a10f53ccull, "codebook table checksum");
}

void TestHadamard() {
  std::vector<double> v(2 * kHadamardSize, 0.0);
  v[0] = 1.0;
  v[kHadamardSize + 5] = 2.0;
  HadamardBlocks(std::span<double>(v));
  const double s = 1.0 / std::sqrt(128.0);
  bool ok = true;
  for (int i = 0; i < kHadamardSize; ++i) {
    ok = ok && std::fabs(v[i] - s) < 1e-15;
    const double expect =
        2.0 * s *
        ((__builtin_popcount(static_cast<unsigned>(i & 5)) % 2) ? -1 : 1);
    ok = ok && std::fabs(v[kHadamardSize + i] - expect) < 1e-15;
  }
  Expect(ok, "Hadamard of unit vectors (natural Sylvester order)");
  HadamardBlocks(std::span<double>(v));
  Expect(std::fabs(v[0] - 1.0) < 1e-14 &&
             std::fabs(v[kHadamardSize + 5] - 2.0) < 1e-14,
         "Hadamard is an involution");
}

void TestGemv() {
  const std::size_t k = 256, n = 128;
  const BitRate rate{4, false};
  std::mt19937 rng(7);
  std::vector<std::uint16_t> trellis((k / 16) * (n / 16) * rate.TileWords());
  for (auto& w : trellis)
    w = static_cast<std::uint16_t>(rng());
  std::vector<std::uint16_t> inner(k * n);
  DequantInner(trellis.data(), k, n, rate, Codebook::kMul1, inner.data());
  std::vector<std::uint16_t> suh(k), svh(n);
  std::uniform_real_distribution<double> scale(0.5, 1.5);
  for (auto& v : suh)
    v = HalfFromDouble((rng() & 1 ? -1 : 1) * scale(rng));
  for (auto& v : svh)
    v = HalfFromDouble((rng() & 1 ? -1 : 1) * scale(rng) * 0.01);
  std::vector<double> x(k), y(n);
  std::normal_distribution<double> normal;
  for (auto& v : x)
    v = normal(rng);
  GemvReference(x, inner.data(), suh.data(), svh.data(), k, n, y);

  // Explicit W = diag(suh) H_k W' H_n diag(svh), all in double.
  std::vector<double> w(k * n);
  for (std::size_t j = 0; j < n; ++j) {
    std::vector<double> column(k);
    for (std::size_t i = 0; i < k; ++i)
      column[i] = HalfToDouble(inner[i * n + j]);
    HadamardBlocks(std::span<double>(column));  // H symmetric: H_k W' column
    for (std::size_t i = 0; i < k; ++i)
      w[i * n + j] = column[i];
  }
  for (std::size_t i = 0; i < k; ++i) {
    std::span<double> row(w.data() + i * n, n);
    HadamardBlocks(row);
    for (std::size_t j = 0; j < n; ++j) {
      row[j] *= HalfToDouble(suh[i]) * HalfToDouble(svh[j]);
    }
  }
  double max_rel = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    double acc = 0.0, mag = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
      acc += x[i] * w[i * n + j];
      mag += std::fabs(x[i] * w[i * n + j]);
    }
    max_rel = std::max(max_rel, std::fabs(acc - y[j]) / mag);
  }
  Expect(max_rel < 1e-12,
         "GEMV reference equals x W with the explicit rotated weight");

  // Column slices agree with the full dequant.
  std::vector<std::uint16_t> slice(k * 32);
  DequantInnerColumns(trellis.data(), k, n, rate, Codebook::kMul1, 64, 32,
                      slice.data());
  bool same = true;
  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = 0; j < 32; ++j)
      same = same && slice[i * 32 + j] == inner[i * n + 64 + j];
  Expect(same, "column slice dequant");
}

// Tensor::Value widens F16, BF16 and F32 exactly and rejects other dtypes.
void TestTensorValue() {
  const std::uint16_t f16[] = {0x3c00, 0xc000};   // 1, -2
  const std::uint16_t bf16[] = {0x3f80, 0x4040};  // 1, 3
  const float f32[] = {0.5F, -7.25F};
  Tensor t;
  t.dtype = DType::kF16;
  t.data = reinterpret_cast<const std::uint8_t*>(f16);
  Expect(t.Value(0) == 1.0F && t.Value(1) == -2.0F, "F16 tensor value");
  t.dtype = DType::kBF16;
  t.data = reinterpret_cast<const std::uint8_t*>(bf16);
  Expect(t.Value(0) == 1.0F && t.Value(1) == 3.0F, "BF16 tensor value");
  t.dtype = DType::kF32;
  t.data = reinterpret_cast<const std::uint8_t*>(f32);
  Expect(t.Value(0) == 0.5F && t.Value(1) == -7.25F, "F32 tensor value");
  t.dtype = DType::kU8;
  bool threw = false;
  try {
    (void)t.Value(0);
  } catch (const std::logic_error&) {
    threw = true;
  }
  Expect(threw, "non-float tensor value must throw");
}

// TiledVHeadIndex: grouped (k head a, v head b, h) -> tiled (b, a, h), a
// bijection, for head vectors and per-head scalars.
void TestTiledVHeadIndex() {
  constexpr std::size_t kK = 16, kV = 48;
  for (const std::size_t dim : {std::size_t{1}, std::size_t{128}}) {
    std::vector<bool> seen(kV * dim);
    for (std::size_t head = 0; head < kV; ++head) {
      for (std::size_t h = 0; h < dim; ++h) {
        const auto actual = TiledVHeadIndex(head * dim + h, dim, kK, kV);
        const auto expected = ((head % 3) * kK + head / 3) * dim + h;
        Expect(actual == expected && !seen[actual], "tiled V head index");
        seen[actual] = true;
      }
    }
  }
}

}  // namespace

int main() {
  TestHalfRoundTrip();
  TestTileState();
  TestPermutation();
  TestCodebooks();
  TestHadamard();
  TestGemv();
  TestTensorValue();
  TestTiledVHeadIndex();
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "exl3_format_test: OK\n";
  return 0;
}
