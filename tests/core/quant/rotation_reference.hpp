#ifndef GUFO_TESTS_CORE_QUANT_ROTATION_REFERENCE_HPP_
#define GUFO_TESTS_CORE_QUANT_ROTATION_REFERENCE_HPP_

// CPU reference for the 128-point input rotation x' = H·D·x. It follows the
// arithmetic contract of src/core/hip/detail/hadamard.hpp step by step (exact
// sign flip, butterflies of length 1..64 in order, one final scale), so GPU
// helpers are compared against it bit for bit. RotateRowFp64 is the
// independent exact formula used to check the reference itself.

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "src/core/hip/detail/hadamard.hpp"

namespace gufo::test::rotation {

inline constexpr std::size_t kSegment = hip::hadamard::kSegment;

[[nodiscard]] inline bool SignBit(std::span<const std::uint32_t> bits,
                                  std::size_t i) noexcept {
  return ((bits[i / 32] >> (i % 32)) & 1U) != 0;
}

/// In-place FP32 rotation of a row whose length is a multiple of 128.
inline void RotateRow(std::span<float> row,
                      std::span<const std::uint32_t> bits) {
  for (std::size_t base = 0; base < row.size(); base += kSegment) {
    float* x = row.data() + base;
    for (std::size_t j = 0; j < kSegment; ++j) {
      if (SignBit(bits, base + j)) {
        x[j] = std::bit_cast<float>(std::bit_cast<std::uint32_t>(x[j]) ^
                                    0x80000000U);
      }
    }
    for (std::size_t length = 1; length < kSegment; length <<= 1) {
      for (std::size_t i = 0; i < kSegment; i += length << 1) {
        for (std::size_t j = i; j < i + length; ++j) {
          const float a = x[j];
          const float b = x[j + length];
          x[j] = a + b;
          x[j + length] = a - b;
        }
      }
    }
    for (std::size_t j = 0; j < kSegment; ++j) {
      x[j] *= hip::hadamard::kHadamard128Scale;
    }
  }
}

/// x' = H·D·x evaluated as an explicit FP64 matrix product.
[[nodiscard]] inline std::vector<double> RotateRowFp64(
    std::span<const float> row, std::span<const std::uint32_t> bits) {
  std::vector<double> out(row.size(), 0.0);
  const double scale = 1.0 / std::sqrt(static_cast<double>(kSegment));
  for (std::size_t base = 0; base < row.size(); base += kSegment) {
    for (std::size_t i = 0; i < kSegment; ++i) {
      double sum = 0.0;
      for (std::size_t j = 0; j < kSegment; ++j) {
        // Sylvester H[i][j] = (-1)^popcount(i & j).
        const double h = (std::popcount(i & j) & 1U) != 0 ? -1.0 : 1.0;
        const double d = SignBit(bits, base + j) ? -1.0 : 1.0;
        sum += h * d * static_cast<double>(row[base + j]);
      }
      out[base + i] = sum * scale;
    }
  }
  return out;
}

}  // namespace gufo::test::rotation

#endif  // GUFO_TESTS_CORE_QUANT_ROTATION_REFERENCE_HPP_
