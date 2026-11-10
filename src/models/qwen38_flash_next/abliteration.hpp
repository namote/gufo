#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ABLITERATION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ABLITERATION_HPP_

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
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

/// Loads the explicit `--abliteration-profile` file. An empty option loads
/// none; no file beside the model is ever picked up implicitly.
std::shared_ptr<const AbliterationProfile> LoadAbliterationOption(
    const std::string& option, std::uint32_t hidden, std::uint32_t layers,
    std::string_view model_identity);

/// Mode and profile recorded in a component-cache continuation: a guard
/// independent of the salted context identity. OFF (`active` null) records
/// mode 0 and the raw image identity, exactly the bytes Flash-Next
/// continuations carried before abliteration existed. ON records mode 1 and
/// the image identity salted with the full profile identity.
struct AbliterationContinuationRecord {
  static constexpr std::uint32_t kOff = 0;
  static constexpr std::uint32_t kOn = 1;
  std::uint32_t mode{kOff};
  std::array<std::uint8_t, 32> image{};
};
/// `image` is the session's image identity at the captured position (empty
/// or 32 bytes).
AbliterationContinuationRecord RecordAbliterationContinuation(
    std::span<const std::uint8_t> image, const AbliterationProfile* active);
/// Whether a recorded continuation may restore into a session whose image
/// identity is `image` and whose active profile is `active` (null for OFF).
/// `image_bytes` is the recorded raw image identity length.
bool AbliterationContinuationMatches(std::uint32_t mode,
                                     std::uint32_t image_bytes,
                                     std::span<const std::uint8_t, 32> recorded,
                                     std::span<const std::uint8_t> image,
                                     const AbliterationProfile* active);

}  // namespace gufo::models::qwen38_flash_next
#endif
