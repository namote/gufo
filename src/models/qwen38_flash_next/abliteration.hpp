#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ABLITERATION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ABLITERATION_HPP_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen/abliteration.hpp"
#include "src/models/qwen38_flash_next/dynamic_abliteration.hpp"

namespace gufo::models::qwen38_flash_next {

/// The shared Qwen runtime projection profile, bound to Flash-Next's
/// `writer-before-hc` placement. No model weights are edited.
struct AbliterationProfile : qwen::AbliterationProfile {
  static AbliterationProfile Load(const std::filesystem::path& path,
                                  std::uint32_t hidden, std::uint32_t layers,
                                  std::string_view model_identity);
  /// A runtime upload of the same file bytes, named `name`.
  static AbliterationProfile Parse(std::string_view bytes, std::string name,
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

/// `profiles` with `profile` added, or replacing the one of its name, in
/// load order (the file name order of `--abliteration-profiles`: the name
/// followed by `.json`). Throws std::invalid_argument when that makes more
/// than qwen::kAbliterationMaxProfiles profiles.
AbliterationProfileSet WithAbliterationProfile(
    const AbliterationProfileSet& profiles,
    std::shared_ptr<const qwen::AbliterationProfile> profile);
/// `profiles` without the profile `name`; throws std::out_of_range when none
/// has it.
AbliterationProfileSet WithoutAbliterationProfile(
    const AbliterationProfileSet& profiles, std::string_view name);

/// The settings of the named profiles of `profiles` with their defaults (load
/// order; combined when several), as an explicit request naming them.
std::shared_ptr<const qwen::AbliterationSettings> ResolveAbliterationDefaults(
    std::span<const std::shared_ptr<const qwen::AbliterationProfile>> profiles,
    std::span<const std::string> names);

/// Validates a whole profile set as the model loader does: at most
/// qwen::kAbliterationMaxProfiles distinct names, every `trigger.with` names
/// a loaded profile other than its own, and every set a detection can turn
/// ON, and the union of all of them, forms device blocks (PlanAbliteration).
/// Returns one candidate per profile with a `trigger` (load order), its
/// `column` left 0 for the caller to assign. Throws std::invalid_argument.
std::vector<DynamicAbliterationCandidate> PlanAbliterationTriggers(
    const AbliterationProfileSet& profiles);

/// One version of a model's loaded profiles, as requests snapshot it: a
/// request keeps the table it was admitted with, so replacing or removing a
/// profile never changes a running request. `leases` keep each profile's
/// device direction rows and detector column allocated while any table (or
/// session holding one) still uses them.
struct AbliterationTable {
  /// Load order.
  AbliterationProfileSet profiles;
  /// Per profile: its first row in the model's direction table, and its
  /// detector's score column (profiles with a `trigger`).
  std::vector<std::uint32_t> rows;
  std::vector<std::optional<std::uint32_t>> columns;
  std::vector<std::shared_ptr<const void>> leases;
  /// The dynamic profiles (PlanAbliterationTriggers) with their columns.
  std::vector<DynamicAbliterationCandidate> candidates;

  [[nodiscard]] const qwen::AbliterationProfile* Find(
      std::string_view name) const noexcept;
  /// Throws std::invalid_argument for a profile the table does not hold.
  [[nodiscard]] std::uint32_t Row(std::string_view name) const;
  [[nodiscard]] std::vector<const qwen::AbliterationProfile*> List() const;
};

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
