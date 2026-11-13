#include "src/core/logit_bias.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "src/core/json.hpp"

namespace gufo::sampling {
namespace {

[[noreturn]] void Fail(std::string_view source, const std::string& what) {
  throw std::invalid_argument("logit bias: " + std::string(source) + ": " +
                              what);
}

[[nodiscard]] bool ValidName(std::string_view name) noexcept {
  if (name.empty() || name.size() > 64)
    return false;
  return std::ranges::all_of(name, [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
  });
}

[[nodiscard]] float ReadBias(const json::Value& value, std::string_view source,
                             std::string_view field) {
  if (!value.is_number() || !std::isfinite(value.as_double()) ||
      value.as_double() < -static_cast<double>(kMaxLogitBias) ||
      value.as_double() > static_cast<double>(kMaxLogitBias))
    Fail(source,
         "'" + std::string(field) + "' must be a number in [-100, 100]");
  return static_cast<float>(value.as_double());
}

/// A float's exact spelling, so cache keys never merge distinct biases.
/// std::to_chars spells hex floats alike on every platform.
[[nodiscard]] std::string ExactFloat(float value) {
  char text[32];
  const auto end =
      std::to_chars(text, text + sizeof(text), value, std::chars_format::hex)
          .ptr;
  return std::string(text, end);
}

/// The text the string rule compares: `decoded` without its leading ASCII
/// whitespace; empty when nothing else remains.
[[nodiscard]] std::string_view StripLeading(std::string_view decoded) noexcept {
  const auto first = decoded.find_first_not_of(" \t\r\n");
  return first == std::string_view::npos ? std::string_view{}
                                         : decoded.substr(first);
}

}  // namespace

std::optional<LogitBiasScope> ParseLogitBiasScope(
    std::string_view text) noexcept {
  if (text == "all")
    return LogitBiasScope::kAll;
  if (text == "answer")
    return LogitBiasScope::kAnswer;
  if (text == "reasoning")
    return LogitBiasScope::kReasoning;
  return std::nullopt;
}

std::string_view LogitBiasScopeName(LogitBiasScope scope) noexcept {
  switch (scope) {
    case LogitBiasScope::kAnswer:
      return "answer";
    case LogitBiasScope::kReasoning:
      return "reasoning";
    case LogitBiasScope::kAll:
      break;
  }
  return "all";
}

bool LogitBiasStringMatches(std::string_view decoded,
                            std::string_view text) noexcept {
  return StripLeading(decoded) == text;
}

LogitBiasProfile ParseLogitBiasProfile(std::string_view text,
                                       std::string_view source) {
  json::Value root;
  try {
    root = json::parse(text);
  } catch (const std::exception& error) {
    Fail(source, std::string("invalid JSON: ") + error.what());
  }
  if (!root.is_object())
    Fail(source, "a profile must be a JSON object");
  LogitBiasProfile profile;
  bool format = false, name = false, scope = false, bias = false;
  for (const auto& [key, value] : root.members()) {
    if (key == "format") {
      if (!value.is_string() || value.str() != kLogitBiasFormat)
        Fail(source,
             "'format' must be \"" + std::string(kLogitBiasFormat) + "\"");
      format = true;
    } else if (key == "name") {
      if (!value.is_string() || !ValidName(value.str()))
        Fail(source,
             "'name' must be 1-64 characters of letters, digits, '_', '-' "
             "or '.'");
      profile.name = value.str();
      name = true;
    } else if (key == "scope") {
      const auto parsed =
          value.is_string() ? ParseLogitBiasScope(value.str()) : std::nullopt;
      if (!parsed)
        Fail(source, "'scope' must be \"all\", \"answer\" or \"reasoning\"");
      profile.scope = *parsed;
      scope = true;
    } else if (key == "bias") {
      profile.bias = ReadBias(value, source, "bias");
      bias = true;
    } else if (key == "tokens") {
      if (!value.is_array())
        Fail(source, "'tokens' must be an array of token ids");
      for (const auto& id : value.items()) {
        const double number = id.is_number() ? id.as_double() : -1.0;
        if (!id.is_number() || !std::isfinite(number) || number < 0 ||
            std::floor(number) != number ||
            number > static_cast<double>(std::numeric_limits<TokenId>::max()))
          Fail(source, "'tokens' must hold nonnegative integer token ids");
        profile.tokens.push_back(static_cast<TokenId>(number));
      }
    } else if (key == "strings") {
      if (!value.is_array())
        Fail(source, "'strings' must be an array of strings");
      for (const auto& entry : value.items()) {
        if (!entry.is_string() || entry.str().empty())
          Fail(source, "'strings' must hold nonempty strings");
        profile.strings.push_back(entry.str());
      }
    } else {
      Fail(source, "unknown key '" + key + "'");
    }
  }
  if (!format)
    Fail(source, "missing 'format'");
  if (!name)
    Fail(source, "missing 'name'");
  if (!scope)
    Fail(source, "missing 'scope'");
  if (!bias)
    Fail(source, "missing 'bias'");
  if (profile.tokens.empty() && profile.strings.empty())
    Fail(source, "a profile needs 'tokens' or 'strings'");
  return profile;
}

void ResolveLogitBiasProfiles(
    std::span<LogitBiasProfile> profiles, std::size_t vocabulary_size,
    const std::function<std::string(TokenId)>& decode) {
  for (std::size_t i = 0; i < profiles.size(); ++i)
    for (std::size_t j = 0; j < i; ++j)
      if (profiles[i].name == profiles[j].name)
        throw std::invalid_argument("logit bias: two profiles are named '" +
                                    profiles[i].name + "'");
  // Each wanted string, mapped to the profiles that list it.
  std::unordered_map<std::string, std::vector<std::size_t>> wanted;
  for (std::size_t i = 0; i < profiles.size(); ++i)
    for (const auto& text : profiles[i].strings)
      wanted[text].push_back(i);
  std::vector<std::vector<TokenId>> matched(profiles.size());
  if (!wanted.empty()) {
    if (!decode)
      throw std::invalid_argument("logit bias: strings need a vocabulary");
    for (std::size_t token = 0; token < vocabulary_size; ++token) {
      const std::string text = decode(static_cast<TokenId>(token));
      const auto stripped = StripLeading(text);
      if (stripped.empty())
        continue;
      const auto found = wanted.find(std::string(stripped));
      if (found == wanted.end())
        continue;
      for (const auto index : found->second)
        matched[index].push_back(static_cast<TokenId>(token));
    }
  }
  for (std::size_t i = 0; i < profiles.size(); ++i) {
    auto& profile = profiles[i];
    for (const auto id : profile.tokens)
      if (id >= vocabulary_size)
        throw std::invalid_argument("logit bias: profile '" + profile.name +
                                    "': token id " + std::to_string(id) +
                                    " is outside the vocabulary of " +
                                    std::to_string(vocabulary_size));
    auto& strings = matched[i];
    std::ranges::sort(strings);
    strings.erase(std::unique(strings.begin(), strings.end()), strings.end());
    profile.string_matches = strings.size();
    profile.tokens.insert(profile.tokens.end(), strings.begin(), strings.end());
    std::ranges::sort(profile.tokens);
    profile.tokens.erase(
        std::unique(profile.tokens.begin(), profile.tokens.end()),
        profile.tokens.end());
    if (profile.tokens.empty())
      throw std::invalid_argument("logit bias: profile '" + profile.name +
                                  "' resolves to no token");
  }
}

std::vector<LogitBiasProfile> LoadLogitBiasProfiles(
    const std::filesystem::path& directory, std::size_t vocabulary_size,
    const std::function<std::string(TokenId)>& decode) {
  namespace fs = std::filesystem;
  std::error_code error;
  if (!fs::is_directory(directory, error))
    throw std::invalid_argument(
        "logit bias: --logit-bias-profiles needs a directory");
  std::vector<fs::path> files;
  for (const auto& entry : fs::directory_iterator(directory)) {
    if (entry.path().extension() != ".json")
      continue;
    if (!entry.is_regular_file())
      throw std::invalid_argument("logit bias: profile " +
                                  entry.path().filename().string() +
                                  " is not a regular file");
    files.push_back(entry.path());
  }
  if (files.empty())
    throw std::invalid_argument(
        "logit bias: no *.json profile in --logit-bias-profiles");
  if (files.size() > kMaxLogitBiasProfiles)
    throw std::invalid_argument("logit bias: more than " +
                                std::to_string(kMaxLogitBiasProfiles) +
                                " profiles");
  std::ranges::sort(files);
  std::vector<LogitBiasProfile> profiles;
  profiles.reserve(files.size());
  for (const auto& file : files) {
    std::ifstream input(file, std::ios::binary);
    if (!input)
      throw std::invalid_argument("logit bias: cannot read " + file.string());
    std::ostringstream text;
    text << input.rdbuf();
    profiles.push_back(
        ParseLogitBiasProfile(text.str(), file.filename().string()));
  }
  ResolveLogitBiasProfiles(profiles, vocabulary_size, decode);
  return profiles;
}

std::shared_ptr<const LogitBias> BuildLogitBias(
    std::size_t vocabulary_size, std::span<const LogitBiasTerm> terms,
    std::string key) {
  const auto build = [&](bool reasoning) {
    std::shared_ptr<std::vector<float>> offsets;
    for (const auto& term : terms) {
      if (term.scope ==
          (reasoning ? LogitBiasScope::kAnswer : LogitBiasScope::kReasoning))
        continue;
      if (!offsets)
        offsets = std::make_shared<std::vector<float>>(vocabulary_size, 0.0F);
      for (const auto token : term.tokens) {
        if (token >= vocabulary_size)
          throw std::invalid_argument("logit bias token " +
                                      std::to_string(token) +
                                      " is outside the vocabulary");
        (*offsets)[token] += term.bias;
      }
    }
    return std::shared_ptr<const std::vector<float>>(std::move(offsets));
  };
  const auto maximum = [](const std::shared_ptr<const std::vector<float>>& v) {
    return v && !v->empty() ? std::max(0.0F, *std::ranges::max_element(*v))
                            : 0.0F;
  };
  auto result = std::make_shared<LogitBias>();
  result->key = std::move(key);
  const bool scoped = std::ranges::any_of(terms, [](const auto& term) {
    return term.scope != LogitBiasScope::kAll;
  });
  result->reasoning = build(true);
  result->answer = scoped ? build(false) : result->reasoning;
  result->reasoning_max = maximum(result->reasoning);
  result->answer_max = maximum(result->answer);
  return result;
}

LogitBiasCatalog::LogitBiasCatalog(std::size_t vocabulary_size,
                                   std::vector<LogitBiasProfile> profiles)
    : vocabulary_size_(vocabulary_size), profiles_(std::move(profiles)) {}

const LogitBiasProfile* LogitBiasCatalog::Find(std::string_view name) const {
  for (const auto& profile : profiles_)
    if (profile.name == name)
      return &profile;
  return nullptr;
}

std::shared_ptr<const LogitBias> LogitBiasCatalog::Resolve(
    const LogitBiasSelection& selection) const {
  if (selection.empty())
    return nullptr;
  std::vector<std::pair<const LogitBiasProfile*, float>> chosen;
  chosen.reserve(selection.profiles.size());
  for (const auto& [name, override_bias] : selection.profiles) {
    const auto* profile = Find(name);
    if (profile == nullptr)
      throw std::invalid_argument("unknown logit bias profile '" + name + "'");
    chosen.emplace_back(profile, override_bias.value_or(profile->bias));
  }
  std::ranges::sort(chosen, {}, [](const auto& entry) {
    return std::string_view(entry.first->name);
  });
  for (const auto& [token, bias] : selection.tokens)
    if (token >= vocabulary_size_)
      throw std::invalid_argument("logit_bias token " + std::to_string(token) +
                                  " is outside the vocabulary of " +
                                  std::to_string(vocabulary_size_));
  std::string key;
  for (const auto& [profile, bias] : chosen) {
    if (!key.empty())
      key += ',';
    key += profile->name + '=' + ExactFloat(bias);
  }
  if (!selection.tokens.empty()) {
    if (!key.empty())
      key += ',';
    key += "logit_bias{";
    for (std::size_t i = 0; i < selection.tokens.size(); ++i)
      key += (i ? "," : "") + std::to_string(selection.tokens[i].first) + '=' +
             ExactFloat(selection.tokens[i].second);
    key += '}';
  }
  {
    std::lock_guard lock(mutex_);
    if (const auto found = cache_.find(key); found != cache_.end())
      return found->second;
  }
  std::vector<LogitBiasTerm> terms;
  terms.reserve(chosen.size() + selection.tokens.size());
  for (const auto& [profile, bias] : chosen)
    terms.push_back({profile->scope, bias, profile->tokens});
  for (const auto& entry : selection.tokens)
    terms.push_back({LogitBiasScope::kAll, entry.second,
                     std::span<const TokenId>(&entry.first, 1)});
  auto built = BuildLogitBias(vocabulary_size_, terms, key);
  std::lock_guard lock(mutex_);
  if (const auto found = cache_.find(key); found != cache_.end())
    return found->second;
  // Simple bound: a full cache starts over; requests keep their arrays.
  if (cache_.size() >= kCacheEntries)
    cache_.clear();
  cache_.emplace(key, built);
  return built;
}

}  // namespace gufo::sampling
