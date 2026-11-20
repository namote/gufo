// The 4-bit draft head copy (upstream #1209 on an EXL3 head): the Q4_K
// superblock quantizer the load-time kernel runs, and the dense form of an
// EXL3 projection that BuildDraftHeadQ4K quantizes.
#include "src/models/qwen38_flash_next/draft_head_q4k.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

#include "src/core/quant/exl3/format.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace q4k = gufo::models::qwen38_flash_next::q4k;
namespace exl3 = gufo::quant::exl3;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

// Deterministic values in [-1, 1).
struct Random {
  std::uint64_t state;
  float Next() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>(state >> 40) / static_cast<float>(1 << 23) - 1.0F;
  }
};

// The quantizer's FP16 conversions agree with the EXL3 format helpers,
// including subnormals, overflow and ties.
void CheckHalf() {
  std::vector<float> values{
      0.0F,           -0.0F,          1.0F,     -2.5F,
      65504.0F,       65520.0F,       1e9F,     6.1e-5F,
      5.96e-8F,       2.98e-8F,       2.99e-8F, 1.0009765625F,
      1.00048828125F, 1.00146484375F, 3.0e-6F,  -7.7e-6F};
  Random random{1};
  for (int i = 0; i < 100000; ++i) {
    const float scale = std::ldexp(1.0F, static_cast<int>(i % 48) - 30);
    values.push_back(random.Next() * scale);
  }
  for (const float value : values) {
    const auto bits = q4k::HalfBits(value);
    Require(bits == exl3::HalfFromDouble(value), "FP16 rounding differs");
    Require(
        static_cast<double>(q4k::HalfFloat(bits)) == exl3::HalfToDouble(bits),
        "FP16 decoding differs");
  }
}

// Dequantized copies stay close to their inputs, read with a stride, and
// zero rows stay zero.
void CheckQuantizer() {
  Random random{7};
  constexpr std::size_t kStride = 3;
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<float> strided(256 * kStride);
    std::vector<float> x(256);
    const float scale = std::ldexp(1.0F, trial % 12 - 8);
    for (int i = 0; i < 256; ++i) {
      // Mostly small values with a few outliers, like a head row.
      float v = random.Next() * scale;
      if (i % 37 == trial % 37)
        v *= 6;
      if (trial % 5 == 0)
        v = std::fabs(v);  // all positive: a nonzero minimum
      x[i] = v;
      strided[i * kStride] = v;
      strided[i * kStride + 1] = 1e6F;  // never read
    }
    q4k::Block block;
    q4k::QuantizeSuperblock(strided.data(), kStride, &block);
    std::vector<float> y(256);
    gufo::quant::DequantizeQ4_K(&block, y.data(), 256);
    double error = 0, energy = 0, max_x = 0;
    for (int i = 0; i < 256; ++i) {
      error += (y[i] - x[i]) * double(y[i] - x[i]);
      energy += double(x[i]) * x[i];
      max_x = std::max(max_x, std::fabs(double(x[i])));
    }
    Require(std::sqrt(error / energy) < 0.15, "Q4_K error is too large");
    for (int i = 0; i < 256; ++i)
      Require(std::fabs(y[i] - x[i]) <= 0.25 * max_x,
              "a Q4_K value is far from its input");
    // The same values read contiguously give the same block.
    q4k::Block contiguous;
    q4k::QuantizeSuperblock(x.data(), 1, &contiguous);
    Require(std::memcmp(&block, &contiguous, sizeof(block)) == 0,
            "the stride changed the block");
  }
  std::vector<float> zero(256, 0.0F);
  q4k::Block block;
  q4k::QuantizeSuperblock(zero.data(), 1, &block);
  std::vector<float> y(256, 1.0F);
  gufo::quant::DequantizeQ4_K(&block, y.data(), 256);
  for (const float v : y)
    Require(v == 0.0F, "a zero row did not stay zero");
}

// The dense rows BuildDraftHeadQ4K quantizes reproduce the EXL3 projection:
// rows of W' rotated with suh, then columns rotated with svh.
void CheckDenseForm() {
  constexpr std::size_t k = 256, n = 256;
  Random random{11};
  std::vector<std::uint16_t> inner(k * n), suh(k), svh(n);  // W' is [k][n]
  for (auto& w : inner)
    w = exl3::HalfFromDouble(random.Next() * 0.1);
  for (auto& s : suh)
    s = exl3::HalfFromDouble(random.Next() < 0 ? -1.0 : 1.0);
  for (auto& s : svh)
    s = exl3::HalfFromDouble(0.5 + 0.5 * (random.Next() + 1.0));
  // Rows of W' (one per output), rotated along the input with suh after.
  std::vector<float> rows(n * k);
  for (std::size_t o = 0; o < n; ++o) {
    std::span<float> row(rows.data() + o * k, k);
    for (std::size_t i = 0; i < k; ++i)
      row[i] = static_cast<float>(exl3::HalfToDouble(inner[i * n + o]));
    exl3::HadamardBlocks(row);
    for (std::size_t i = 0; i < k; ++i)
      row[i] *= static_cast<float>(exl3::HalfToDouble(suh[i]));
  }
  // Each input column rotated along the outputs with svh after.
  std::vector<float> dense(n * k);
  std::vector<float> column(n);
  for (std::size_t i = 0; i < k; ++i) {
    for (std::size_t o = 0; o < n; ++o)
      column[o] = rows[o * k + i];
    exl3::HadamardBlocks(std::span<float>(column));
    for (std::size_t o = 0; o < n; ++o)
      dense[o * k + i] =
          column[o] * static_cast<float>(exl3::HalfToDouble(svh[o]));
  }
  for (int trial = 0; trial < 4; ++trial) {
    std::vector<double> x(k), expected(n);
    for (auto& v : x)
      v = random.Next();
    exl3::GemvReference(x, inner.data(), suh.data(), svh.data(), k, n,
                        expected);
    for (std::size_t o = 0; o < n; ++o) {
      double y = 0;
      for (std::size_t i = 0; i < k; ++i)
        y += dense[o * k + i] * x[i];
      Require(std::fabs(y - expected[o]) <= 1e-4 * (1 + std::fabs(expected[o])),
              "the dense form differs from the EXL3 projection");
    }
  }
}

}  // namespace

int main() {
  try {
    CheckHalf();
    CheckQuantizer();
    CheckDenseForm();
  } catch (const std::exception& e) {
    std::cerr << "draft_head_q4k_test: " << e.what() << '\n';
    return 1;
  }
  std::cout << "draft_head_q4k_test: ok\n";
  return 0;
}
