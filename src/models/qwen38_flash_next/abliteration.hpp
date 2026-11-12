#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ABLITERATION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ABLITERATION_HPP_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen/abliteration.hpp"

namespace gufo::models::qwen38_flash_next {

/// The shared Qwen runtime projection profile, bound to Flash-Next's
/// `writer-before-hc` placement. No model weights are edited.
struct AbliterationProfile : qwen::AbliterationProfile {
  static AbliterationProfile Load(const std::filesystem::path& path,
                                  std::uint32_t hidden, std::uint32_t layers,
                                  std::string_view model_identity);
};

using qwen::AbliterationContinuationMatches;
using qwen::AbliterationContinuationRecord;
using qwen::AbliterationInputIdentity;
using qwen::AbliterationOverrides;
using qwen::AbliterationSettings;
using qwen::RecordAbliterationContinuation;
using qwen::ResolveAbliterationSettings;

using qwen::AbliterationTrigger;
using qwen::SelectAbliterationProfile;
using AbliterationProfileSet =
    std::vector<std::shared_ptr<const qwen::AbliterationProfile>>;

/// Loads the profile files (qwen::ListAbliterationProfiles of
/// `--abliteration-profiles`), each bound to this model. Names must be
/// unique. No file beside the model is ever picked up implicitly.
AbliterationProfileSet LoadAbliterationProfiles(
    std::span<const std::string> files, std::uint32_t hidden,
    std::uint32_t layers, std::string_view model_identity);

}  // namespace gufo::models::qwen38_flash_next
#endif
