#include "src/models/qwen38_flash_next/dynamic_abliteration.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace gufo::models::qwen38_flash_next {

DynamicAbliteration::DynamicAbliteration(
    std::vector<DynamicAbliterationCandidate> c)
    : candidates_(std::move(c)),
      row_scores_(candidates_.size()),
      max_scores_(candidates_.size()) {
  if (candidates_.empty() || candidates_.size() > kDetectorColumns)
    throw std::invalid_argument("dynamic abliteration needs 1..8 profiles");
  for (const auto& candidate : candidates_)
    if (!candidate.settings || candidate.column >= kDetectorColumns ||
        candidate.settings->dynamic_from)
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
    if (pass.first_position + r >= committed)
      break;
    const auto row = pass.scores.subspan(std::size_t{r} * kDetectorColumns,
                                         kDetectorColumns);
    for (std::size_t i = 0; i < candidates_.size(); ++i) {
      const float score = row[candidates_[i].column];
      max_scores_[i] = have_max_ ? std::max(max_scores_[i], score) : score;
    }
    have_max_ = true;
    if (pending_)
      continue;
    // The best margin among the candidates that crossed on this row.
    std::optional<std::size_t> best;
    for (std::size_t i = 0; i < candidates_.size(); ++i) {
      const float margin =
          row[candidates_[i].column] - candidates_[i].threshold;
      if (margin > 0 && (!best || margin > row[candidates_[*best].column] -
                                               candidates_[*best].threshold))
        best = i;
    }
    if (best || pass.prefill) {
      for (std::size_t i = 0; i < candidates_.size(); ++i)
        row_scores_[i] = row[candidates_[i].column];
      have_row_ = true;
    }
    if (best) {
      pending_ = best;
      pending_prefill_ = pass.prefill;
    }
  }
}

std::shared_ptr<const qwen::AbliterationSettings> DynamicAbliteration::Engage(
    std::uint32_t position) {
  if (engaged_ || !pending_)
    return nullptr;
  engaged_ = true;
  chosen_ = *pending_;
  position_ = position;
  return qwen::DynamicAbliterationSettings(*candidates_[chosen_].settings,
                                           position);
}

DynamicAbliterationReport DynamicAbliteration::Report() const {
  DynamicAbliterationReport report;
  for (std::size_t i = 0; i < candidates_.size(); ++i) {
    if (have_row_)
      report.scores.emplace_back(candidates_[i].profile, row_scores_[i]);
    if (have_max_)
      report.max_scores.emplace_back(candidates_[i].profile, max_scores_[i]);
  }
  if (engaged_) {
    report.triggered.push_back(candidates_[chosen_].profile);
    report.position = position_;
    report.phase = pending_prefill_ ? "prefill" : "decode";
  }
  return report;
}

}  // namespace gufo::models::qwen38_flash_next
