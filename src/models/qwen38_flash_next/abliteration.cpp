#include "src/models/qwen38_flash_next/abliteration.hpp"

#include <stdexcept>

namespace gufo::models::qwen38_flash_next {

AbliterationProfile AbliterationProfile::Load(const std::filesystem::path& path,
                                              std::uint32_t hidden,
                                              std::uint32_t layers,
                                              std::string_view model_identity) {
  AbliterationProfile result;
  static_cast<qwen::AbliterationProfile&>(result) =
      qwen::AbliterationProfile::Load(path, hidden, layers, model_identity,
                                      qwen::kAbliterationWriterBeforeHc);
  return result;
}

AbliterationProfileSet LoadAbliterationProfiles(
    std::span<const std::string> files, std::uint32_t hidden,
    std::uint32_t layers, std::string_view model_identity) {
  if (files.size() > qwen::kAbliterationMaxProfiles)
    throw std::invalid_argument("abliteration: too many profiles");
  AbliterationProfileSet result;
  for (const auto& file : files) {
    auto profile = std::make_shared<const AbliterationProfile>(
        AbliterationProfile::Load(file, hidden, layers, model_identity));
    for (const auto& loaded : result)
      if (loaded->name == profile->name)
        throw std::invalid_argument("abliteration: two profiles are named '" +
                                    profile->name + "'");
    result.push_back(std::move(profile));
  }
  return result;
}

}  // namespace gufo::models::qwen38_flash_next
