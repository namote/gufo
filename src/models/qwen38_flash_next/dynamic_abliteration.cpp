#include "src/models/qwen38_flash_next/dynamic_abliteration.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace gufo::models::qwen38_flash_next {

DynamicAbliteration::DynamicAbliteration(
    std::vector<DynamicAbliterationCandidate> c, AbliterationResolver resolve)
    : candidates_(std::move(c)),
      resolve_(std::move(resolve)),
      row_scores_(candidates_.size()),
      max_scores_(candidates_.size()) {
  if (candidates_.empty() || candidates_.size() > kDetectorColumns || !resolve_)
    throw std::invalid_argument("dynamic abliteration needs 1..8 profiles");
  for (const auto& candidate : candidates_)
    if (candidate.column >= kDetectorColumns || candidate.profiles.empty() ||
        candidate.profiles.front() != candidate.profile)
      throw std::invalid_argument("invalid dynamic abliteration candidate");
}

void DynamicAbliteration::Observe(const DetectorPass& pass,
                                  std::uint32_t committed) {
  if (engaged_ || pass.id == 0 || pass.id == observed_)
    return;
  observed_ = pass.id;
  if (pass.scores.size() < std::size_t{pass.rows} * kDetectorColumns)
    throw std::logic_error("detector pass is shorter than its rows");
  for (std::uint32_t r = 0; r < pass.rows; ++r) {
    const std::uint32_t position = pass.first_position + r;
    if (position >= committed)
      break;
    // A prefill row counts only at the request's last prompt token.
    if (pass.prefill && position + 1 != prompt_length_)
      continue;
    const auto row = pass.scores.subspan(std::size_t{r} * kDetectorColumns,
                                         kDetectorColumns);
    if (!pass.prefill) {
      for (std::size_t i = 0; i < candidates_.size(); ++i) {
        const float score = row[candidates_[i].column];
        max_scores_[i] = have_max_ ? std::max(max_scores_[i], score) : score;
      }
      have_max_ = true;
    }
    if (!pending_.empty())
      continue;
    // Every candidate crossing on this row: the prompt threshold on the last
    // prompt row, the decode threshold (when the trigger has one) on
    // generated rows.
    std::vector<std::size_t> crossed;
    for (std::size_t i = 0; i < candidates_.size(); ++i) {
      const auto threshold = pass.prefill
                                 ? std::optional(candidates_[i].threshold)
                                 : candidates_[i].decode_threshold;
      if (threshold && row[candidates_[i].column] > *threshold)
        crossed.push_back(i);
    }
    if (!crossed.empty() || pass.prefill) {
      for (std::size_t i = 0; i < candidates_.size(); ++i)
        row_scores_[i] = row[candidates_[i].column];
      have_row_ = true;
    }
    if (!crossed.empty()) {
      // All of them and their companions, without duplicates.
      std::vector<std::string> names;
      for (const auto i : crossed)
        for (const auto& name : candidates_[i].profiles)
          if (std::ranges::find(names, name) == names.end())
            names.push_back(name);
      union_ = resolve_(names);
      pending_ = std::move(crossed);
      pending_prefill_ = pass.prefill;
      pending_position_ = position + 1;
    }
  }
}

std::shared_ptr<const qwen::AbliterationSettings> DynamicAbliteration::Engage(
    std::uint32_t position) {
  // A prefill detection is served by readmitting the request ON
  // (TextModelRunner::PrefillReadmission), never by switching mid-request.
  if (engaged_ || pending_.empty() || pending_prefill_)
    return nullptr;
  engaged_ = true;
  position_ = position;
  return qwen::DynamicAbliterationSettings(*union_, position);
}

DynamicAbliterationReport DynamicAbliteration::Report() const {
  DynamicAbliterationReport report;
  for (std::size_t i = 0; i < candidates_.size(); ++i) {
    if (have_row_)
      report.scores.emplace_back(candidates_[i].profile, row_scores_[i]);
    if (have_max_)
      report.max_scores.emplace_back(candidates_[i].profile, max_scores_[i]);
  }
  // A crossing reports its profile even when the request ended before a
  // pass ran ON; the position is where ON starts (or would have started).
  if (!pending_.empty()) {
    // Every profile the detection turns ON: the crossing ones and their
    // companions, in load order.
    for (const auto* part : qwen::AbliterationParts(*union_))
      report.triggered.push_back(part->profile);
    for (const auto i : pending_)
      report.crossed.push_back(candidates_[i].profile);
    report.position = engaged_ ? position_ : pending_position_;
    report.phase = pending_prefill_ ? "prefill" : "decode";
  }
  return report;
}

}  // namespace gufo::models::qwen38_flash_next
