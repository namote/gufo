#include "src/models/qwen38_flash_next/abliteration.hpp"

#include <algorithm>
#include <numeric>
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

AbliterationPlan PlanAbliteration(
    std::span<const qwen::AbliterationSettings* const> parts,
    std::span<const std::shared_ptr<const qwen::AbliterationProfile>> loaded) {
  if (parts.empty() || parts.size() > qwen::kAbliterationMaxProfiles)
    throw std::invalid_argument("abliteration: combine 1..8 profiles");
  std::vector<const qwen::AbliterationProfile*> profiles;
  for (const auto* part : parts) {
    const qwen::AbliterationProfile* found = nullptr;
    for (const auto& profile : loaded)
      if (profile->name == part->profile)
        found = profile.get();
    if (!found || !qwen::AbliterationSettingsFit(*found, *part))
      throw std::invalid_argument("abliteration: profile '" + part->profile +
                                  "' does not fit");
    profiles.push_back(found);
  }
  const auto layers = profiles.front()->layers;
  const auto hidden = std::size_t{profiles.front()->hidden};
  const std::uint32_t site_count = 3 * layers;
  constexpr std::size_t kRows = qwen::kAbliterationMaxDirections;
  const auto strength = [&](std::size_t p, std::uint32_t site) {
    const auto layer = site % layers;
    switch (site / layers) {
      case 0:
        return parts[p]->attention.at(layer);
      case 1:
        return parts[p]->mlp.at(layer);
      default:
        return parts[p]->residual.at(layer);
    }
  };
  AbliterationPlan plan;
  plan.sites.assign(site_count, -1);
  // Blocks by their ordered profiles; each basis row's profile.
  std::vector<std::vector<std::size_t>> orders;
  std::vector<std::vector<std::size_t>> row_owner;
  for (std::uint32_t site = 0; site < site_count; ++site) {
    // Suppression profiles only: additive ones (R5) add after the site's
    // projection, one block each (the executor appends them).
    std::vector<std::size_t> order;
    for (std::size_t p = 0; p < parts.size(); ++p)
      if (!parts[p]->additive && strength(p, site) > 0)
        order.push_back(p);
    if (order.empty())
      continue;
    std::ranges::stable_sort(order, [&](std::size_t a, std::size_t b) {
      return strength(a, site) > strength(b, site);
    });
    auto found = std::ranges::find(orders, order);
    if (found == orders.end()) {
      if (orders.size() == kAbliterationMaxBlocks)
        throw std::invalid_argument(
            "abliteration: the profiles' strengths need more than 16 "
            "combinations of sites");
      AbliterationBlockPlan block;
      block.preserve_norm = parts[order.front()]->preserve_norm;
      for (const auto p : order)
        if (parts[p]->preserve_norm != block.preserve_norm)
          throw std::invalid_argument(
              "abliteration: profiles acting at one site must agree on "
              "'preserve_norm' ('" +
              parts[p]->profile + "', '" + parts[order.front()]->profile +
              "')");
      std::vector<std::size_t> owners;
      if (order.size() == 1) {
        // One profile here: its own rows, exactly its own projection.
        const auto* part = parts[order.front()];
        block.profile = part->profile;
        block.row = part->direction_row;
        block.count = part->direction_count;
        owners.assign(block.count, order.front());
      } else {
        std::vector<float> gathered;
        std::vector<std::size_t> source_owner;
        for (const auto p : order) {
          const auto* part = parts[p];
          const auto first = std::size_t{part->direction_row} * hidden;
          gathered.insert(gathered.end(),
                          profiles[p]->directions.begin() + first,
                          profiles[p]->directions.begin() + first +
                              std::size_t{part->direction_count} * hidden);
          source_owner.insert(source_owner.end(), part->direction_count, p);
          block.key += part->profile + ":" +
                       std::to_string(part->direction_row) + ":" +
                       std::to_string(part->direction_count) + ";";
        }
        auto rows = qwen::OrthonormalizeDirections(gathered, hidden);
        if (rows.origins.empty() || rows.origins.size() > kRows)
          throw std::invalid_argument(
              "abliteration: profiles acting at one site span more than 8 "
              "directions");
        block.basis = std::move(rows.basis);
        block.count = static_cast<std::uint32_t>(rows.origins.size());
        for (const auto origin : rows.origins)
          owners.push_back(source_owner[origin]);
      }
      block.weights.assign(std::size_t{site_count} * kRows, 0.0F);
      plan.blocks.push_back(std::move(block));
      orders.push_back(order);
      row_owner.push_back(std::move(owners));
      found = std::prev(orders.end());
    }
    const auto index = static_cast<std::size_t>(found - orders.begin());
    auto& block = plan.blocks[index];
    for (std::size_t j = 0; j < row_owner[index].size(); ++j)
      block.weights[std::size_t{site} * kRows + j] =
          strength(row_owner[index][j], site);
    plan.sites[site] = static_cast<std::int16_t>(index);
  }
  const auto adds = static_cast<std::size_t>(std::ranges::count_if(
      parts, [](const auto* part) { return part->additive; }));
  if (adds > 8 || plan.blocks.size() + adds > kAbliterationMaxBlocks)
    throw std::invalid_argument(
        "abliteration: the profiles need more than 16 blocks (or more than "
        "8 additive profiles)");
  return plan;
}

}  // namespace gufo::models::qwen38_flash_next
