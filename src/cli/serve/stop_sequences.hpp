#ifndef GUFO_SERVER_STOP_SEQUENCES_HPP_
#define GUFO_SERVER_STOP_SEQUENCES_HPP_

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::server {

inline constexpr std::size_t kMaxStopSequences = 64;
inline constexpr std::size_t kMaxStopSequenceBytes = 4096;
inline constexpr std::size_t kMaxStopTotalBytes = 16384;

inline void ValidateStopSequences(const std::vector<std::string>& sequences,
                                  std::size_t max_count = kMaxStopSequences) {
  if (sequences.size() > max_count)
    throw std::invalid_argument("too many stop sequences (maximum " +
                                std::to_string(max_count) + ")");
  std::size_t bytes = 0;
  for (const auto& sequence : sequences) {
    if (sequence.empty() || sequence.size() > kMaxStopSequenceBytes)
      throw std::invalid_argument("stop sequences must contain 1 to " +
                                  std::to_string(kMaxStopSequenceBytes) +
                                  " UTF-8 bytes");
    bytes += sequence.size();
    if (bytes > kMaxStopTotalBytes)
      throw std::invalid_argument("stop sequences exceed the total byte limit");
  }
}

enum class StopSequenceFormat { kOpenAi, kAnthropic };

inline std::optional<std::string> ParseStopSequences(
    const json::Value* value, StopSequenceFormat format,
    std::vector<std::string>* output) {
  output->clear();
  if (value == nullptr ||
      (format == StopSequenceFormat::kOpenAi && value->is_null()))
    return std::nullopt;
  if (format == StopSequenceFormat::kOpenAi && value->is_string()) {
    output->push_back(value->str());
  } else if (value->is_array()) {
    for (const auto& item : value->items()) {
      if (!item.is_string())
        return "stop sequences must be strings";
      output->push_back(item.str());
    }
  } else {
    return format == StopSequenceFormat::kOpenAi
               ? "'stop' must be null, a string, or an array of strings"
               : "'stop_sequences' must be an array of strings";
  }
  try {
    ValidateStopSequences(
        *output, format == StopSequenceFormat::kOpenAi ? 4 : kMaxStopSequences);
  } catch (const std::invalid_argument& error) {
    return error.what();
  }
  return std::nullopt;
}

/// Only this many old visible bytes can participate in a new stop match.
inline std::string StopSequenceSuffix(
    std::string_view prior, const std::vector<std::string>& sequences) {
  std::size_t keep = 0;
  for (const auto& sequence : sequences)
    if (!sequence.empty())
      keep = std::max(keep, sequence.size() - 1);
  return std::string(prior.substr(prior.size() - std::min(keep, prior.size())));
}

/// Matches raw accepted output, retaining only a possible stop prefix.
/// The first completed match wins; ties at the same end prefer the longer
/// sequence. This is independent of how token pieces are split into chunks.
class StopSequenceFilter {
public:
  explicit StopSequenceFilter(std::vector<std::string> sequences = {},
                              std::string_view prior = {})
      : sequences_(std::move(sequences)) {
    ValidateStopSequences(sequences_);
    pending_ = StopSequenceSuffix(prior, sequences_);
    prior_bytes_ = pending_.size();
  }

  [[nodiscard]] bool enabled() const noexcept { return !sequences_.empty(); }
  [[nodiscard]] bool stopped() const noexcept { return match_.has_value(); }
  [[nodiscard]] const std::string& matched_sequence() const {
    return sequences_.at(match_.value());
  }

  std::string Push(std::string_view piece) {
    if (stopped())
      return {};
    pending_.append(piece);
    std::size_t end = std::string::npos;
    std::size_t start = std::string::npos;
    for (std::size_t i = 0; i < sequences_.size(); ++i) {
      auto position = pending_.find(sequences_[i]);
      // A short stop wholly inside the old suffix already belongs to the
      // previous request. Only matches consuming a new byte can stop this one.
      while (position != std::string::npos &&
             position + sequences_[i].size() <= prior_bytes_)
        position = pending_.find(sequences_[i], position + 1);
      if (position == std::string::npos)
        continue;
      const auto candidate_end = position + sequences_[i].size();
      if (candidate_end < end || (candidate_end == end && position < start)) {
        end = candidate_end;
        start = position;
        match_ = i;
      }
    }
    if (stopped()) {
      const auto skip = std::min(prior_bytes_, start);
      auto safe = pending_.substr(skip, start - skip);
      pending_.clear();
      prior_bytes_ = 0;
      return safe;
    }
    std::size_t keep = 0;
    for (const auto& sequence : sequences_) {
      for (auto length = std::min(pending_.size(), sequence.size() - 1);
           length > keep; --length) {
        if (std::string_view(pending_).ends_with(
                std::string_view(sequence).substr(0, length))) {
          keep = length;
          break;
        }
      }
    }
    const auto ready = pending_.size() - keep;
    const auto skip = std::min(prior_bytes_, ready);
    auto safe = pending_.substr(skip, ready - skip);
    pending_.erase(0, ready);
    prior_bytes_ -= skip;
    return safe;
  }

  std::string Finish() {
    auto safe = pending_.substr(prior_bytes_);
    pending_.clear();
    prior_bytes_ = 0;
    return safe;
  }

private:
  std::vector<std::string> sequences_;
  std::string pending_;
  std::size_t prior_bytes_{0};
  std::optional<std::size_t> match_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_STOP_SEQUENCES_HPP_
