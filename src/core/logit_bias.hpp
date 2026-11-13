#ifndef GUFO_CORE_LOGIT_BIAS_HPP_
#define GUFO_CORE_LOGIT_BIAS_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/sampling.hpp"

namespace gufo::sampling {

/// Where a logit-bias set applies: every sampled token, only after the
/// reply's reasoning phase ends (as the Han scale's answer scope), or only
/// inside that phase. Replies without a reasoning phase are all answer.
enum class LogitBiasScope : std::uint8_t { kAll, kAnswer, kReasoning };

/// "all", "answer" or "reasoning"; nullopt for any other spelling.
[[nodiscard]] std::optional<LogitBiasScope> ParseLogitBiasScope(
    std::string_view text) noexcept;
[[nodiscard]] std::string_view LogitBiasScopeName(
    LogitBiasScope scope) noexcept;

/// The profile file format tag.
inline constexpr std::string_view kLogitBiasFormat = "gufo-logit-bias-v1";
/// Every bias (profile, override and OpenAI logit_bias value) lies in
/// [-kMaxLogitBias, kMaxLogitBias], as in the OpenAI API.
inline constexpr float kMaxLogitBias = 100.0F;
/// Profiles one request may select.
inline constexpr std::size_t kMaxRequestLogitBiasProfiles = 8;
/// Token entries of one request's OpenAI logit_bias map.
inline constexpr std::size_t kMaxRequestLogitBiasTokens = 1024;
/// Profile files one --logit-bias-profiles directory may hold.
inline constexpr std::size_t kMaxLogitBiasProfiles = 256;

/// One named token-set bias, loaded from one --logit-bias-profiles file.
struct LogitBiasProfile {
  std::string name;
  LogitBiasScope scope{LogitBiasScope::kAll};
  float bias{0.0F};
  /// Listed ids; after ResolveLogitBiasProfiles, the sorted unique union of
  /// those ids and every token the strings matched.
  std::vector<TokenId> tokens;
  /// Literal token texts, matched by LogitBiasStringMatches.
  std::vector<std::string> strings;
  /// Tokens the strings matched (after resolution), for the load log.
  std::size_t string_matches{0};
};

/// The string expansion rule: a token's decoded text, after removing its
/// leading ASCII whitespace (space, tab, CR, LF), equals `text` exactly.
/// Case, trailing whitespace and punctuation must match; list each variant
/// ("Wait", "wait", "Wait,") in the profile.
[[nodiscard]] bool LogitBiasStringMatches(std::string_view decoded,
                                          std::string_view text) noexcept;

/// Parses one profile file. `source` names it in errors. Throws
/// std::invalid_argument on an unknown key, a missing or ill-typed field, a
/// bias outside the bound, a bad scope, a negative or non-integer id, or a
/// profile with neither tokens nor strings.
[[nodiscard]] LogitBiasProfile ParseLogitBiasProfile(std::string_view text,
                                                     std::string_view source);

/// Expands strings against the vocabulary (decoding each token once, only
/// when some profile has strings) and checks ids. Throws
/// std::invalid_argument for an id outside the vocabulary, a profile that
/// resolves to no token, or two profiles of the same name.
void ResolveLogitBiasProfiles(
    std::span<LogitBiasProfile> profiles, std::size_t vocabulary_size,
    const std::function<std::string(TokenId)>& decode);

/// Reads every *.json in `directory` (file name order), then resolves them.
/// Throws std::invalid_argument with the offending file named.
[[nodiscard]] std::vector<LogitBiasProfile> LoadLogitBiasProfiles(
    const std::filesystem::path& directory, std::size_t vocabulary_size,
    const std::function<std::string(TokenId)>& decode);

/// A request's selection: profile names with optional bias overrides, and
/// the OpenAI logit_bias map (an ad hoc set with scope all).
struct LogitBiasSelection {
  std::vector<std::pair<std::string, std::optional<float>>> profiles;
  /// Sorted by token, unique.
  std::vector<std::pair<TokenId, float>> tokens;

  [[nodiscard]] bool empty() const noexcept {
    return profiles.empty() && tokens.empty();
  }
};

/// One term of a combination: `bias` added to every token of `tokens`.
struct LogitBiasTerm {
  LogitBiasScope scope{LogitBiasScope::kAll};
  float bias{0.0F};
  std::span<const TokenId> tokens;
};

/// Sums the terms into dense per-phase offsets (overlapping sets add, in
/// term order). A phase no term applies to stays null; when every term has
/// scope all both phases share one array. Ids at or beyond
/// `vocabulary_size` are an error (std::invalid_argument).
[[nodiscard]] std::shared_ptr<const LogitBias> BuildLogitBias(
    std::size_t vocabulary_size, std::span<const LogitBiasTerm> terms,
    std::string key);

/// The loaded profiles of one model and a cache of built combinations.
/// Thread-safe.
class LogitBiasCatalog {
public:
  LogitBiasCatalog(std::size_t vocabulary_size,
                   std::vector<LogitBiasProfile> profiles);

  [[nodiscard]] std::size_t vocabulary_size() const noexcept {
    return vocabulary_size_;
  }
  [[nodiscard]] std::span<const LogitBiasProfile> profiles() const noexcept {
    return profiles_;
  }
  [[nodiscard]] const LogitBiasProfile* Find(std::string_view name) const;

  /// The offsets of a selection; null for an empty one. Profiles combine in
  /// name order, then the OpenAI map, so equal selections give equal
  /// offsets and share one cached array. Throws std::invalid_argument for
  /// an unknown profile or a token outside the vocabulary.
  [[nodiscard]] std::shared_ptr<const LogitBias> Resolve(
      const LogitBiasSelection& selection) const;

private:
  static constexpr std::size_t kCacheEntries = 16;
  std::size_t vocabulary_size_;
  std::vector<LogitBiasProfile> profiles_;
  mutable std::mutex mutex_;
  mutable std::map<std::string, std::shared_ptr<const LogitBias>, std::less<>>
      cache_;
};

}  // namespace gufo::sampling

#endif  // GUFO_CORE_LOGIT_BIAS_HPP_
