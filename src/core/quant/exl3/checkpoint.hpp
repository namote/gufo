#ifndef GUFO_CORE_QUANT_EXL3_CHECKPOINT_HPP_
#define GUFO_CORE_QUANT_EXL3_CHECKPOINT_HPP_

// Reader of EXL3 checkpoints (Hugging Face safetensors directories with an
// exl3 quantization_config; docs/EXL3-FORMAT.md). Model independent: it maps
// the shards, exposes every tensor by name and assembles the per-layer view
// of a quantized linear (trellis, suh, svh, codebook marker, bias).

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/quant/exl3/format.hpp"

namespace gufo::quant::exl3 {

/// Safetensors element type of a stored tensor.
enum class DType : std::uint8_t { kF16, kBF16, kF32, kI16, kI32, kU8, kOther };

/// One stored tensor: a view into a mapped shard.
struct Tensor {
  DType dtype = DType::kOther;
  std::string dtype_name;
  std::vector<std::int64_t> shape;
  const std::uint8_t* data = nullptr;
  std::size_t bytes = 0;

  [[nodiscard]] std::int64_t Elements() const noexcept;
};

/// The checkpoint-level quantization_config of config.json.
struct QuantizationConfig {
  std::string quant_method;
  std::string version;
  double bits = 0.0;
  int head_bits = 0;
  int mtp_bits = 0;  // 0 when absent
  std::string codebook;
  std::string out_scales;
};

/// A quantized linear layer computing y = x W, W [in_features, out_features].
struct Linear {
  std::string name;
  std::size_t in_features = 0;   // k
  std::size_t out_features = 0;  // n
  BitRate rate;
  Codebook codebook = Codebook::k3Inst;
  const std::uint16_t* trellis = nullptr;  // [k/16][n/16][rate.TileWords()]
  const std::uint16_t* suh = nullptr;      // F16 [k]
  const std::uint16_t* svh = nullptr;      // F16 [n]
  const std::uint16_t* bias = nullptr;     // F16 [n] or nullptr
};

class Checkpoint {
public:
  Checkpoint(const Checkpoint&) = delete;
  Checkpoint& operator=(const Checkpoint&) = delete;
  Checkpoint(Checkpoint&&) noexcept;
  Checkpoint& operator=(Checkpoint&&) noexcept;
  ~Checkpoint();

  /// Opens `directory`: config.json, then model.safetensors.index.json (or a
  /// single model.safetensors). Throws std::runtime_error naming the problem.
  static Checkpoint Open(const std::filesystem::path& directory);

  /// Whether `config.json` at `directory` declares quant_method "exl3".
  static bool IsExl3(const std::filesystem::path& directory);

  [[nodiscard]] const QuantizationConfig& config() const noexcept {
    return config_;
  }
  [[nodiscard]] const std::map<std::string, Tensor, std::less<>>& tensors()
      const noexcept {
    return tensors_;
  }
  [[nodiscard]] const Tensor* Find(std::string_view name) const noexcept;

  /// Whether `module` (for example "lm_head") is stored as an EXL3 linear.
  [[nodiscard]] bool IsQuantized(std::string_view module) const noexcept;

  /// The EXL3 linear stored under `module`. Throws std::runtime_error when a
  /// tensor is missing or inconsistent (shapes, dtypes, bit rate, codebook).
  [[nodiscard]] Linear GetLinear(std::string_view module) const;

private:
  Checkpoint();
  struct Shard;
  QuantizationConfig config_;
  std::vector<std::unique_ptr<Shard>> shards_;
  std::map<std::string, Tensor, std::less<>> tensors_;
};

}  // namespace gufo::quant::exl3

#endif  // GUFO_CORE_QUANT_EXL3_CHECKPOINT_HPP_
