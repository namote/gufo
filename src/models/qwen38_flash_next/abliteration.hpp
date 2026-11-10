#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ABLITERATION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ABLITERATION_HPP_

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::qwen38_flash_next {

/// Immutable, model-bound runtime projection. No model weights are edited.
/// JSON keeps this small sidecar independent of the weight shard loader.
struct AbliterationProfile {
  std::vector<float> direction;
  std::vector<float> attention;
  std::vector<float> mlp;
  std::string identity;
  std::string model_identity;

  static AbliterationProfile Load(const std::filesystem::path& path,
                                  std::uint32_t hidden, std::uint32_t layers,
                                  std::string_view model_identity);
};

/// Domain-separated input identity; OFF callers keep their original bytes.
std::vector<std::uint8_t> AbliterationInputIdentity(
    std::span<const std::uint8_t> input, std::string_view profile_identity);

}  // namespace gufo::models::qwen38_flash_next
#endif
