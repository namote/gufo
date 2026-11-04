#ifndef GUFO_MODELS_QWEN_VISION_EXL3_WEIGHTS_HPP_
#define GUFO_MODELS_QWEN_VISION_EXL3_WEIGHTS_HPP_

#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "src/core/quant/exl3/checkpoint.hpp"

namespace gufo::models::qwen::vision {

/// The native pack's vision weights, without reading language-model payloads.
/// Views remain valid for this object's lifetime. No GGUF conversion is used.
class Exl3Weights {
public:
  struct Matrix {
    // Actual runtime dimensions. Quantized projections retain their padded
    // dimensions: in particular, do not crop an EXL3 FFN's up output
    // before its activation/down projection (upstream retains those channels).
    std::size_t in_features;
    std::size_t out_features;
    std::optional<quant::exl3::Linear> quantized;
    const quant::exl3::Tensor* raw{nullptr};
  };

  /// A declared vision_config selects native vision. Malformed or incomplete
  /// declared vision is rejected by the constructor, never treated as absent.
  [[nodiscard]] static bool HasVision(const std::filesystem::path& directory);
  Exl3Weights(const std::filesystem::path& directory,
              std::uint32_t output_width);
  [[nodiscard]] const std::map<std::string, Matrix>& matrices() const {
    return matrices_;
  }
  /// F32-convertible vectors/positions in the encoder's canonical names.
  /// Quantized biases include padded output channels; retain them until the
  /// projection's actual output is deliberately cropped.
  [[nodiscard]] const std::map<std::string, const quant::exl3::Tensor*>&
  tensors() const {
    return tensors_;
  }
  [[nodiscard]] const std::string& identity() const;
  [[nodiscard]] static std::vector<float> FloatValues(
      const quant::exl3::Tensor& tensor);

private:
  const quant::exl3::Tensor* Require(const std::string& name,
                                     std::vector<std::int64_t> shape);
  void AddTensor(const std::string& key, const std::string& name,
                 std::vector<std::int64_t> shape);
  void AddMatrix(const std::string& key, const std::string& module,
                 std::size_t in, std::size_t out);
  quant::exl3::Checkpoint checkpoint_;
  std::map<std::string, Matrix> matrices_;
  std::map<std::string, const quant::exl3::Tensor*> tensors_;
  std::map<std::string, const quant::exl3::Tensor*> selected_;
  std::uint32_t output_width_;
  mutable std::once_flag identity_once_;
  mutable std::string identity_;
};

}  // namespace gufo::models::qwen::vision
#endif
