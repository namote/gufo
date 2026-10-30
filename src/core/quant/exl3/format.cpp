// CPU reference of the EXL3 weight format. The codebook constants, the
// trellis window arithmetic, the tile permutation and the rotation order
// follow upstream exllamav3 (https://github.com/turboderp-org/exllamav3,
// v1.6.5, commit cdc785db89e263e65e5849cb53e1ada156411414):
//   exllamav3/exllamav3_ext/quant/codebook.cuh (decode_3inst),
//   exllamav3/exllamav3_ext/quant/exl3_dq.cuh (dq, dq8_half),
//   exllamav3/modules/quant/exl3_lib/quantize.py (tensor_core_perm, frac_k),
//   exllamav3/modules/quant/exl3.py (get_weight_tensor).
//
// exllamav3 is MIT licensed:
//   Copyright (c) 2025 Turboderp
//   Permission is hereby granted, free of charge, to any person obtaining a
//   copy of this software and associated documentation files (the
//   "Software"), to deal in the Software without restriction, including
//   without limitation the rights to use, copy, modify, merge, publish,
//   distribute, sublicense, and/or sell copies of the Software, and to permit
//   persons to whom the Software is furnished to do so, subject to the
//   following conditions: The above copyright notice and this permission
//   notice shall be included in all copies or substantial portions of the
//   Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

#include "src/core/quant/exl3/format.hpp"

#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gufo::quant::exl3 {
namespace {

constexpr std::uint32_t k3InstMul = 89226354u;
constexpr std::uint32_t k3InstAdd = 64248484u;
constexpr std::uint32_t kMcgMul = 0xCBAC1FEDu;
constexpr std::uint32_t kMul1Mul = 0x83DCD12Du;
constexpr std::uint16_t kMul1Inv = 0x1eee;   // 0.00677 = 1 / 147.7
constexpr std::uint16_t kMul1Bias = 0xc931;  // -10.39

// Round half to even of a non-negative double holding an exactly
// representable dyadic value.
double RoundHalfEven(double x) noexcept {
  const double floor = std::floor(x);
  const double diff = x - floor;
  if (diff > 0.5) {
    return floor + 1.0;
  }
  if (diff < 0.5) {
    return floor;
  }
  return std::fmod(floor, 2.0) == 0.0 ? floor : floor + 1.0;
}

// (x & 0x8fff8fff) ^ 0x3b603b60, then the FP16 sum of the two halves.
std::uint16_t DecodeLop3Sum(std::uint32_t x) noexcept {
  x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
  const double lo = HalfToDouble(static_cast<std::uint16_t>(x & 0xffffu));
  const double hi = HalfToDouble(static_cast<std::uint16_t>(x >> 16));
  return HalfFromDouble(lo + hi);  // the double sum of two halves is exact
}

std::array<std::uint8_t, kTileSize> BuildPermutation() noexcept {
  std::array<std::uint8_t, kTileSize> perm{};
  for (int t = 0; t < 32; ++t) {
    const int r0 = (t % 4) * 2;
    const int rows[4] = {r0, r0 + 1, r0 + 8, r0 + 9};
    const int c0 = t / 4;
    for (int j = 0; j < 8; ++j) {
      const int c = j < 4 ? c0 : c0 + 8;
      perm[t * 8 + j] = static_cast<std::uint8_t>(rows[j % 4] * 16 + c);
    }
  }
  return perm;
}

template<typename T>
void HadamardBlocksImpl(std::span<T> values) noexcept {
  const double scale = 1.0 / std::sqrt(static_cast<double>(kHadamardSize));
  for (std::size_t base = 0; base + kHadamardSize <= values.size();
       base += kHadamardSize) {
    T* v = values.data() + base;
    // Iterative fast Walsh-Hadamard transform, natural (Sylvester) order.
    for (int h = 1; h < kHadamardSize; h *= 2) {
      for (int i = 0; i < kHadamardSize; i += 2 * h) {
        for (int j = i; j < i + h; ++j) {
          const T a = v[j];
          const T b = v[j + h];
          v[j] = a + b;
          v[j + h] = a - b;
        }
      }
    }
    for (int i = 0; i < kHadamardSize; ++i) {
      v[i] = static_cast<T>(v[i] * scale);
    }
  }
}

}  // namespace

double HalfToDouble(std::uint16_t h) noexcept {
  const int sign = (h >> 15) & 1;
  const int exponent = (h >> 10) & 0x1f;
  const int mantissa = h & 0x3ff;
  double value;
  if (exponent == 0) {
    value = std::ldexp(static_cast<double>(mantissa), -24);
  } else if (exponent == 0x1f) {
    value = mantissa == 0 ? std::numeric_limits<double>::infinity()
                          : std::numeric_limits<double>::quiet_NaN();
  } else {
    value = std::ldexp(static_cast<double>(1024 + mantissa), exponent - 25);
  }
  return sign ? -value : value;
}

std::uint16_t HalfFromDouble(double value) noexcept {
  if (std::isnan(value)) {
    return 0x7e00;
  }
  const std::uint16_t sign = std::signbit(value) ? 0x8000 : 0x0000;
  const double a = std::fabs(value);
  if (a >= 65520.0) {  // rounds to infinity
    return static_cast<std::uint16_t>(sign | 0x7c00);
  }
  if (a < std::ldexp(1.0, -14)) {  // subnormal range, unit 2^-24
    const double r = RoundHalfEven(std::ldexp(a, 24));
    return static_cast<std::uint16_t>(sign | static_cast<std::uint16_t>(r));
  }
  int e = 0;
  std::frexp(a, &e);     // a = m * 2^e, m in [0.5, 1)
  int exponent = e - 1;  // a = 1.f * 2^exponent
  double r = RoundHalfEven(std::ldexp(a, 10 - exponent));  // in [1024, 2048]
  if (r >= 2048.0) {
    r = 1024.0;
    ++exponent;
  }
  if (exponent > 15) {
    return static_cast<std::uint16_t>(sign | 0x7c00);
  }
  return static_cast<std::uint16_t>(
      sign | static_cast<std::uint16_t>((exponent + 15) << 10) |
      static_cast<std::uint16_t>(static_cast<int>(r) - 1024));
}

std::optional<BitRate> BitRateFromTileWords(std::size_t tile_words) noexcept {
  if (tile_words % 16 == 0) {
    const auto bits = static_cast<int>(tile_words / 16);
    if (bits >= 1 && bits <= 8) {
      return BitRate{bits, false};
    }
    return std::nullopt;
  }
  if (tile_words % 16 == 8) {
    const auto bits = static_cast<int>(tile_words / 16);
    if (bits >= 1 && bits <= 3) {
      return BitRate{bits, true};
    }
  }
  return std::nullopt;
}

bool Supports(Codebook codebook, BitRate rate) noexcept {
  if (rate.bits < 1 || rate.bits > 8) {
    return false;
  }
  if (rate.half) {
    return codebook == Codebook::kMul1 && rate.bits <= 3;
  }
  return true;
}

int StateEndBit(BitRate rate, int position) noexcept {
  if (!rate.half) {
    return (position + 1) * rate.bits;
  }
  // Even positions take bits, odd positions bits + 1 (mask 0xAAAA).
  const int pair = 2 * rate.bits + 1;
  return (position / 2) * pair + ((position % 2) == 0 ? rate.bits : pair);
}

std::uint16_t TileState(const std::uint16_t* tile, BitRate rate,
                        int position) noexcept {
  const int words32 = rate.TileWords() / 2;
  const int total = rate.TileBits();
  // Offset by one whole tile so the window start never goes negative.
  const int end = StateEndBit(rate, position) + total;
  const int i0 = (end - 16) / 32;
  const int i1 = (end - 1) / 32;
  const int shift = (i1 + 1) * 32 - end;
  auto word = [&](int i) {
    const int w = i % words32;
    return static_cast<std::uint32_t>(tile[2 * w]) |
           (static_cast<std::uint32_t>(tile[2 * w + 1]) << 16);
  };
  const std::uint64_t merged =
      (static_cast<std::uint64_t>(word(i0)) << 32) | word(i1);
  return static_cast<std::uint16_t>((merged >> shift) & 0xffffu);
}

std::uint16_t DecodeState(std::uint16_t state, Codebook codebook) noexcept {
  std::uint32_t x = state;
  switch (codebook) {
    case Codebook::k3Inst:
      return DecodeLop3Sum(x * k3InstMul + k3InstAdd);
    case Codebook::kMcg:
      return DecodeLop3Sum(x * kMcgMul);
    case Codebook::kMul1: {
      x *= kMul1Mul;
      const std::uint32_t byte_sum =
          (x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) + (x >> 24);
      // 0x6400 + byte_sum as FP16 bits is exactly 1024 + byte_sum.
      const double h = 1024.0 + static_cast<double>(byte_sum);
      // A fused FP16 fma: the double product and sum are exact here.
      return HalfFromDouble(h * HalfToDouble(kMul1Inv) +
                            HalfToDouble(kMul1Bias));
    }
  }
  return 0;
}

const std::array<std::uint8_t, kTileSize>& TilePositionToElement() noexcept {
  static const std::array<std::uint8_t, kTileSize> perm = BuildPermutation();
  return perm;
}

void DequantTile(const std::uint16_t* tile, BitRate rate, Codebook codebook,
                 std::span<std::uint16_t, kTileSize> out) noexcept {
  const auto& perm = TilePositionToElement();
  for (int p = 0; p < kTileSize; ++p) {
    out[perm[p]] = DecodeState(TileState(tile, rate, p), codebook);
  }
}

void DequantInnerColumns(const std::uint16_t* trellis, std::size_t k,
                         std::size_t n, BitRate rate, Codebook codebook,
                         std::size_t n_begin, std::size_t n_count,
                         std::uint16_t* out) {
  if (k % 16 != 0 || n % 16 != 0 || n_begin % 16 != 0 || n_count % 16 != 0 ||
      n_begin + n_count > n) {
    throw std::invalid_argument("EXL3 dequant: shape is not a tile multiple");
  }
  const std::size_t words = static_cast<std::size_t>(rate.TileWords());
  const std::size_t tiles_n = n / 16;
  std::array<std::uint16_t, kTileSize> tile{};
  for (std::size_t tk = 0; tk < k / 16; ++tk) {
    for (std::size_t tn = n_begin / 16; tn < (n_begin + n_count) / 16; ++tn) {
      DequantTile(trellis + (tk * tiles_n + tn) * words, rate, codebook, tile);
      for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 16; ++c) {
          out[(tk * 16 + r) * n_count + (tn * 16 - n_begin) + c] =
              tile[r * 16 + c];
        }
      }
    }
  }
}

void DequantInner(const std::uint16_t* trellis, std::size_t k, std::size_t n,
                  BitRate rate, Codebook codebook, std::uint16_t* out) {
  DequantInnerColumns(trellis, k, n, rate, codebook, 0, n, out);
}

void HadamardBlocks(std::span<double> values) noexcept {
  HadamardBlocksImpl(values);
}

void HadamardBlocks(std::span<float> values) noexcept {
  HadamardBlocksImpl(values);
}

void GemvReference(std::span<const double> x, const std::uint16_t* inner,
                   const std::uint16_t* suh, const std::uint16_t* svh,
                   std::size_t k, std::size_t n, std::span<double> y) {
  if (x.size() != k || y.size() != n || k % kHadamardSize != 0 ||
      n % kHadamardSize != 0) {
    throw std::invalid_argument("EXL3 GEMV reference: bad shape");
  }
  std::vector<double> xh(k);
  for (std::size_t i = 0; i < k; ++i) {
    xh[i] = x[i] * HalfToDouble(suh[i]);
  }
  HadamardBlocks(std::span<double>(xh));
  std::vector<double> t(n, 0.0);
  for (std::size_t i = 0; i < k; ++i) {
    const std::uint16_t* row = inner + i * n;
    for (std::size_t j = 0; j < n; ++j) {
      t[j] += xh[i] * HalfToDouble(row[j]);
    }
  }
  HadamardBlocks(std::span<double>(t));
  for (std::size_t j = 0; j < n; ++j) {
    y[j] = t[j] * HalfToDouble(svh[j]);
  }
}

}  // namespace gufo::quant::exl3
