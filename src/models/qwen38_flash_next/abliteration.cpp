#include "src/models/qwen38_flash_next/abliteration.hpp"

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

std::shared_ptr<const AbliterationProfile> LoadAbliterationOption(
    const std::string& option, std::uint32_t hidden, std::uint32_t layers,
    std::string_view model_identity) {
  if (option.empty())
    return nullptr;
  return std::make_shared<const AbliterationProfile>(
      AbliterationProfile::Load(option, hidden, layers, model_identity));
}

}  // namespace gufo::models::qwen38_flash_next
