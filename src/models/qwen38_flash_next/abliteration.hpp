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

/// Most device blocks of one combination, and most profiles active at one
/// site whose rows a block combines.
inline constexpr std::size_t kAbliterationMaxBlocks = 16;

/// One device block of a combination (several profiles ON): weighted
/// directions (qwen::kAbliterationWeightedBit) projected at its sites.
struct AbliterationBlockPlan {
  /// One profile's own rows (`profile`, from `row`), or a union basis
  /// (`basis`, `key` naming its ordered sources) uploaded for it.
  std::string profile;
  std::uint32_t row{0};
  std::vector<float> basis;
  std::string key;
  std::uint32_t count{1};
  bool preserve_norm{false};
  /// [3 * layers][kAbliterationMaxDirections]: each direction's strength at
  /// each site (zero where the block does not run).
  std::vector<float> weights;
};
struct AbliterationPlan {
  std::vector<AbliterationBlockPlan> blocks;
  /// Each site's block (AbliterationWeightIndex - 1), -1 for none.
  std::vector<std::int16_t> sites;
};

/// The blocks of several profiles ON together (`parts` in load order, all
/// of `loaded`'s geometry). Additive profiles (R5) take no part in a site's
/// projection: they add afterwards, one block each, counted in the limit. At
/// each site the profiles with a strength there are ordered by descending
/// strength (ties: load order) and their direction rows orthonormalized in that
/// order (qwen::OrthonormalizeDirections); each basis row keeps the strength of
/// the profile that introduced it, so a component two profiles share is
/// removed once, at the stronger one's strength, and each profile's own
/// remaining directions at its own. Sites with the same ordered profiles
/// share one block. Throws std::invalid_argument when a site's union spans
/// more than kAbliterationMaxDirections rows, the sites need more than
/// kAbliterationMaxBlocks blocks, or profiles acting at one site disagree
/// on norm preservation. Pure host work: the server runs it to validate a
/// set before the request reaches a session.
AbliterationPlan PlanAbliteration(
    std::span<const qwen::AbliterationSettings* const> parts,
    std::span<const std::shared_ptr<const qwen::AbliterationProfile>> loaded);

}  // namespace gufo::models::qwen38_flash_next
#endif
