#ifndef GUFO_TESTS_CORE_QUANT_ROTATION_TEST_GGUF_HPP_
#define GUFO_TESTS_CORE_QUANT_ROTATION_TEST_GGUF_HPP_

// Minimal in-memory GGUF v3 writer for rotation tests: string and uint32
// metadata, F32 and F16 tensors with real payloads.

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::test::rotation {

/// float -> IEEE half bits, round to nearest even (finite, normal range).
[[nodiscard]] inline std::uint16_t FloatToHalfBits(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16) & 0x8000U;
  const int exponent = static_cast<int>((bits >> 23) & 0xFFU) - 127 + 15;
  std::uint32_t mantissa = bits & 0x7FFFFFU;
  if (value == 0.0F) {
    return static_cast<std::uint16_t>(sign);
  }
  // Callers only store small normal values (signs and test values).
  std::uint32_t half =
      (static_cast<std::uint32_t>(exponent) << 10) | (mantissa >> 13);
  const std::uint32_t rest = mantissa & 0x1FFFU;
  if (rest > 0x1000U || (rest == 0x1000U && (half & 1U) != 0)) {
    ++half;
  }
  return static_cast<std::uint16_t>(sign | half);
}

class GgufImage {
public:
  void AddString(std::string_view key, std::string_view value) {
    Key(key, core::GgufValueType::kString);
    String(meta_, value);
    ++metadata_count_;
  }

  void AddUint32(std::string_view key, std::uint32_t value) {
    Key(key, core::GgufValueType::kUint32);
    Pod(meta_, value);
    ++metadata_count_;
  }

  /// `dims` are GGUF ne order (ne0 = K first). Values are a deterministic
  /// ramp; tests here only need the tensor geometry.
  void AddTensorF32(std::string_view name, std::vector<std::uint64_t> dims) {
    std::size_t count = 1;
    for (const auto dim : dims) {
      count *= dim;
    }
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
      values[i] = static_cast<float>(i % 17) * 0.125F - 1.0F;
    }
    AddTensorF32Values(name, std::move(dims), values);
  }

  void AddTensorF32Values(std::string_view name,
                          std::vector<std::uint64_t> dims,
                          const std::vector<float>& values) {
    std::vector<std::uint8_t> payload(values.size() * sizeof(float));
    std::memcpy(payload.data(), values.data(), payload.size());
    tensors_.push_back({std::string(name), std::move(dims),
                        core::GgmlType::kF32, std::move(payload)});
  }

  void AddTensorF16(std::string_view name, const std::vector<float>& values) {
    std::vector<std::uint8_t> payload(values.size() * sizeof(std::uint16_t));
    for (std::size_t i = 0; i < values.size(); ++i) {
      const std::uint16_t half = FloatToHalfBits(values[i]);
      std::memcpy(payload.data() + i * sizeof(half), &half, sizeof(half));
    }
    tensors_.push_back({std::string(name),
                        {values.size()},
                        core::GgmlType::kF16,
                        std::move(payload)});
  }

  [[nodiscard]] std::vector<std::uint8_t> Build() const {
    std::vector<std::uint8_t> out;
    out.insert(out.end(), {'G', 'G', 'U', 'F'});
    Pod(out, std::uint32_t{3});
    Pod(out, static_cast<std::uint64_t>(tensors_.size()));
    Pod(out, metadata_count_);
    out.insert(out.end(), meta_.begin(), meta_.end());
    std::uint64_t offset = 0;
    std::vector<std::uint64_t> offsets;
    for (const auto& tensor : tensors_) {
      String(out, tensor.name);
      Pod(out, static_cast<std::uint32_t>(tensor.dims.size()));
      for (const auto dim : tensor.dims) {
        Pod(out, dim);
      }
      Pod(out, static_cast<std::uint32_t>(tensor.type));
      Pod(out, offset);
      offsets.push_back(offset);
      offset += Align(tensor.payload.size());
    }
    out.resize(Align(out.size()), 0);
    const std::size_t data_start = out.size();
    out.resize(data_start + offset, 0);
    for (std::size_t i = 0; i < tensors_.size(); ++i) {
      std::memcpy(out.data() + data_start + offsets[i],
                  tensors_[i].payload.data(), tensors_[i].payload.size());
    }
    return out;
  }

private:
  struct Tensor {
    std::string name;
    std::vector<std::uint64_t> dims;
    core::GgmlType type;
    std::vector<std::uint8_t> payload;
  };

  static std::size_t Align(std::size_t size) { return (size + 31) & ~31ULL; }

  template<typename T>
  static void Pod(std::vector<std::uint8_t>& out, T value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
  }

  static void String(std::vector<std::uint8_t>& out, std::string_view text) {
    Pod(out, static_cast<std::uint64_t>(text.size()));
    out.insert(out.end(), text.begin(), text.end());
  }

  void Key(std::string_view key, core::GgufValueType type) {
    String(meta_, key);
    Pod(meta_, static_cast<std::uint32_t>(type));
  }

  std::vector<std::uint8_t> meta_;
  std::uint64_t metadata_count_{0};
  std::vector<Tensor> tensors_;
};

}  // namespace gufo::test::rotation

#endif  // GUFO_TESTS_CORE_QUANT_ROTATION_TEST_GGUF_HPP_
