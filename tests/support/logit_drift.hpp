// Full-vocabulary drift of a candidate route's logits from a reference
// route's on the same positions: RMSE of the raw logits, KL(reference ||
// candidate) over the full softmax and top-1 agreement. Used by the
// recurrent-state comparisons (qwen27b_state_compare, Flash-Next gpu_probe
// --state-compare).
#ifndef GUFO_TESTS_SUPPORT_LOGIT_DRIFT_HPP_
#define GUFO_TESTS_SUPPORT_LOGIT_DRIFT_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string_view>

namespace gufo::test {

struct LogitDrift {
  double rmse{};
  double kl{};
  std::uint32_t reference_top{};
  std::uint32_t candidate_top{};
};

[[nodiscard]] inline std::uint32_t ArgMax(std::span<const float> logits) {
  return static_cast<std::uint32_t>(
      std::distance(logits.begin(), std::ranges::max_element(logits)));
}

[[nodiscard]] inline LogitDrift CompareLogits(
    std::span<const float> reference, std::span<const float> candidate) {
  if (reference.size() != candidate.size() || reference.empty())
    throw std::invalid_argument("logit rows differ in size");
  const double ref_max = *std::ranges::max_element(reference);
  const double cand_max = *std::ranges::max_element(candidate);
  double ref_sum = 0.0;
  double cand_sum = 0.0;
  double squared = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    ref_sum += std::exp(reference[i] - ref_max);
    cand_sum += std::exp(candidate[i] - cand_max);
    const double d = static_cast<double>(reference[i]) - candidate[i];
    squared += d * d;
  }
  const double ref_lse = ref_max + std::log(ref_sum);
  const double cand_lse = cand_max + std::log(cand_sum);
  double kl = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double log_p = reference[i] - ref_lse;
    const double log_q = candidate[i] - cand_lse;
    kl += std::exp(log_p) * (log_p - log_q);
  }
  return {std::sqrt(squared / static_cast<double>(reference.size())),
          std::max(kl, 0.0), ArgMax(reference), ArgMax(candidate)};
}

/// Mean and maximum drift over a group of rows (prefill pass ends, decode
/// steps).
class DriftSummary {
public:
  void Add(const LogitDrift& drift) {
    ++rows_;
    rmse_sum_ += drift.rmse;
    kl_sum_ += drift.kl;
    rmse_max_ = std::max(rmse_max_, drift.rmse);
    kl_max_ = std::max(kl_max_, drift.kl);
    agree_ += drift.reference_top == drift.candidate_top ? 1 : 0;
  }
  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }

  /// One JSON object; `name` is the group.
  void Write(std::ostream& out, std::string_view name) const {
    const double n = rows_ == 0 ? 1.0 : static_cast<double>(rows_);
    out << '"' << name << "\": {\"rows\": " << rows_
        << ", \"rmse_mean\": " << rmse_sum_ / n
        << ", \"rmse_max\": " << rmse_max_ << ", \"kl_mean\": " << kl_sum_ / n
        << ", \"kl_max\": " << kl_max_
        << ", \"top1_agreement\": " << static_cast<double>(agree_) / n << '}';
  }

private:
  std::size_t rows_{};
  std::size_t agree_{};
  double rmse_sum_{};
  double kl_sum_{};
  double rmse_max_{};
  double kl_max_{};
};

/// The comparison report both state-comparison tools print.
/// `greedy_divergence`: the first decode step whose candidate top-1 differs
/// from the reference's greedy token (empty: none in `steps`). Feeding the
/// reference's greedy tokens to both routes makes this exactly where two
/// independent greedy completions part.
inline void WriteStateComparison(std::ostream& out, std::string_view reference,
                                 std::string_view candidate,
                                 std::size_t prompt_tokens, std::size_t steps,
                                 const DriftSummary& prefill,
                                 const DriftSummary& decode,
                                 std::optional<std::size_t> greedy_divergence) {
  out << "{\"reference\": \"" << reference << "\", \"candidate\": \""
      << candidate << "\", \"prompt_tokens\": " << prompt_tokens
      << ", \"decode_steps\": " << steps << ", ";
  prefill.Write(out, "prefill");
  out << ", ";
  decode.Write(out, "decode");
  out << ", \"greedy_divergence_step\": ";
  if (greedy_divergence)
    out << *greedy_divergence;
  else
    out << "null";
  out << "}\n";
}

}  // namespace gufo::test

#endif  // GUFO_TESTS_SUPPORT_LOGIT_DRIFT_HPP_
