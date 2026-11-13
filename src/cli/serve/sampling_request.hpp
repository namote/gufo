#ifndef GUFO_SERVER_SAMPLING_REQUEST_HPP_
#define GUFO_SERVER_SAMPLING_REQUEST_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "src/core/json.hpp"
#include "src/core/logit_bias.hpp"
#include "src/core/sampling.hpp"

namespace gufo::server {

struct SamplingRequestError {
  std::string message;
  std::string code;
};

namespace detail {

inline constexpr std::int64_t kMaxExactJsonInteger = INT64_C(9007199254740991);
inline constexpr std::size_t kMaxSamplingWindow =
    std::numeric_limits<std::uint32_t>::max();

inline std::string SamplingErrorCode(std::string_view field) {
  std::string code = "invalid_";
  code.append(field);
  for (char& character : code) {
    if (character == '-') {
      character = '_';
    }
  }
  return code;
}

inline std::optional<SamplingRequestError> ReadSamplingFloat(
    const json::Value& body, std::string_view field, float* output,
    bool nullable = false) {
  const json::Value* value = body.find(std::string(field));
  if (value == nullptr || (nullable && value->is_null())) {
    return std::nullopt;
  }
  if (!value->is_number() || !std::isfinite(value->as_double()) ||
      value->as_double() <
          -static_cast<double>(std::numeric_limits<float>::max()) ||
      value->as_double() >
          static_cast<double>(std::numeric_limits<float>::max())) {
    return SamplingRequestError{
        .message = "'" + std::string(field) + "' must be a finite number",
        .code = SamplingErrorCode(field),
    };
  }
  *output = static_cast<float>(value->as_double());
  return std::nullopt;
}

template<typename Integer>
inline std::optional<SamplingRequestError> ReadSamplingInteger(
    const json::Value& body, std::string_view field, Integer minimum,
    Integer maximum, Integer* output, bool nullable = false) {
  const json::Value* value = body.find(std::string(field));
  if (value == nullptr || (nullable && value->is_null())) {
    return std::nullopt;
  }
  const double number = value->is_number() ? value->as_double() : 0.0;
  if (!value->is_number() || !std::isfinite(number) ||
      std::floor(number) != number || number < static_cast<double>(minimum) ||
      number > static_cast<double>(maximum)) {
    return SamplingRequestError{
        .message = "'" + std::string(field) + "' must be an integer between " +
                   std::to_string(minimum) + " and " + std::to_string(maximum),
        .code = SamplingErrorCode(field),
    };
  }
  *output = static_cast<Integer>(number);
  return std::nullopt;
}

inline constexpr std::string_view kLlamaSamplerFields[] = {
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    "typical_p",
    "tfs_z",
    "dynatemp_range",
    "dynatemp_exponent",
    "top_n_sigma",
    "xtc_probability",
    "xtc_threshold",
    "dry_multiplier",
    "dry_base",
    "dry_allowed_length",
    "dry_penalty_last_n",
    "dry_sequence_breakers",
};

inline bool IsLlamaSamplerField(std::string_view field) {
  for (const auto known : kLlamaSamplerFields) {
    if (field == known) {
      return true;
    }
  }
  return false;
}

/// llama.cpp sampler fields that gufo does not implement. A field that is
/// absent, null, or set to a value that leaves llama.cpp's sampler chain
/// unchanged is ignored, so clients that always send their sampler settings
/// (Open WebUI 0.12 sends mirostat 0) still work. A value that would activate
/// one of these samplers is rejected rather than sampling another
/// distribution. Neutral values follow llama.cpp: the server's clamps and
/// each sampler's early return.
inline std::optional<SamplingRequestError> CheckNeutralLlamaSamplers(
    const json::Value& body) {
  const auto invalid = [](std::string_view field, std::string_view what) {
    return SamplingRequestError{
        .message = "'" + std::string(field) + "' must be " + std::string(what),
        .code = "invalid_sampling",
    };
  };
  const auto unsupported = [](std::string_view field) {
    return SamplingRequestError{
        .message = "request field '" + std::string(field) +
                   "' is not supported with a value that enables it",
        .code = "unsupported_sampling",
    };
  };
  std::optional<SamplingRequestError> error;
  // Reads a present, non-null number; null counts as absent.
  const auto number = [&](std::string_view field) -> std::optional<double> {
    const json::Value* value = body.find(std::string(field));
    if (value == nullptr || value->is_null()) {
      return std::nullopt;
    }
    if (!value->is_number() || !std::isfinite(value->as_double())) {
      if (!error) {
        error = invalid(field, "a finite number");
      }
      return std::nullopt;
    }
    return value->as_double();
  };
  const auto window = [&](std::string_view field) -> std::optional<double> {
    const auto value = number(field);
    if (value && (std::floor(*value) != *value || *value < 0.0 ||
                  *value > std::numeric_limits<std::int32_t>::max())) {
      if (!error) {
        error = invalid(field, "an integer between 0 and 2147483647");
      }
    }
    return value;
  };
  const auto mirostat = number("mirostat");
  (void)number("mirostat_tau");
  (void)number("mirostat_eta");
  const auto typical_p = number("typical_p");
  (void)number("tfs_z");
  const auto dynatemp_range = number("dynatemp_range");
  (void)number("dynatemp_exponent");
  const auto top_n_sigma = number("top_n_sigma");
  const auto xtc_probability = number("xtc_probability");
  const auto xtc_threshold = number("xtc_threshold");
  const auto dry_multiplier = number("dry_multiplier");
  (void)number("dry_base");
  (void)window("dry_allowed_length");
  const auto dry_penalty_last_n = window("dry_penalty_last_n");
  if (const json::Value* breakers = body.find("dry_sequence_breakers");
      breakers != nullptr && !breakers->is_null() && !error) {
    bool strings = breakers->is_array() && !breakers->empty();
    if (strings) {
      for (const auto& breaker : breakers->items()) {
        strings = strings && breaker.is_string();
      }
    }
    if (!strings) {
      error = invalid("dry_sequence_breakers", "a nonempty array of strings");
    }
  }
  if (error) {
    return error;
  }
  // mirostat_tau/eta only matter with mirostat 1 or 2, dynatemp_exponent
  // only with a positive dynatemp_range, and llama.cpp no longer has TFS.
  if (mirostat && *mirostat > 0.0) {
    return unsupported("mirostat");
  }
  if (typical_p && *typical_p < 1.0) {
    return unsupported("typical_p");
  }
  if (dynatemp_range && *dynatemp_range > 0.0) {
    return unsupported("dynatemp_range");
  }
  if (top_n_sigma && *top_n_sigma > 0.0) {
    return unsupported("top_n_sigma");
  }
  // XTC acts only when the clamped probability is positive and the clamped
  // threshold is at most 0.5 (llama.cpp defaults: 0.0 and 0.1).
  if (std::clamp(xtc_probability.value_or(0.0), 0.0, 1.0) > 0.0 &&
      std::clamp(xtc_threshold.value_or(0.1), 0.0, 1.0) <= 0.5) {
    return unsupported(xtc_probability ? "xtc_probability" : "xtc_threshold");
  }
  // DRY acts only with a nonzero multiplier and a nonzero window
  // (common.h defaults 0.0 and 64; the server limits the window to
  // [0, INT32_MAX]). The sampler skips dry_base < 1, but the server replaces
  // such a base with its default first, so dry_base never disables DRY.
  if (dry_multiplier.value_or(0.0) != 0.0 &&
      dry_penalty_last_n.value_or(64.0) != 0.0) {
    return unsupported("dry_multiplier");
  }
  return std::nullopt;
}

/// Reads the Gufo fields logit_bias_profiles (an array of loaded profile
/// names) and logit_bias_overrides (an object mapping a selected name to its
/// bias), and the OpenAI logit_bias map (decimal token id string to bias).
/// Null or absent fields select nothing; biases lie in [-100, 100].
inline std::optional<SamplingRequestError> ParseLogitBiasSelection(
    const json::Value& body, const sampling::LogitBiasCatalog& catalog,
    sampling::LogitBiasSelection* output) {
  const auto error = [](std::string message, std::string code) {
    return SamplingRequestError{.message = std::move(message),
                                .code = std::move(code)};
  };
  const auto bias = [](const json::Value& value) -> std::optional<float> {
    if (!value.is_number() || !std::isfinite(value.as_double()) ||
        std::abs(value.as_double()) >
            static_cast<double>(sampling::kMaxLogitBias))
      return std::nullopt;
    return static_cast<float>(value.as_double());
  };
  if (const auto* names = body.find("logit_bias_profiles");
      names != nullptr && !names->is_null()) {
    if (!names->is_array())
      return error("'logit_bias_profiles' must be an array of profile names",
                   "invalid_logit_bias_profiles");
    if (names->size() > sampling::kMaxRequestLogitBiasProfiles)
      return error("'logit_bias_profiles' selects at most " +
                       std::to_string(sampling::kMaxRequestLogitBiasProfiles) +
                       " profiles",
                   "invalid_logit_bias_profiles");
    for (const auto& name : names->items()) {
      if (!name.is_string())
        return error("'logit_bias_profiles' must be an array of profile names",
                     "invalid_logit_bias_profiles");
      if (catalog.Find(name.str()) == nullptr)
        return error("unknown logit bias profile '" + name.str() + "'",
                     "unknown_logit_bias_profile");
      if (std::ranges::any_of(output->profiles, [&](const auto& entry) {
            return entry.first == name.str();
          }))
        return error("'logit_bias_profiles' names '" + name.str() + "' twice",
                     "invalid_logit_bias_profiles");
      output->profiles.emplace_back(name.str(), std::nullopt);
    }
  }
  if (const auto* overrides = body.find("logit_bias_overrides");
      overrides != nullptr && !overrides->is_null()) {
    if (!overrides->is_object())
      return error("'logit_bias_overrides' must map profile names to biases",
                   "invalid_logit_bias_overrides");
    for (const auto& [name, value] : overrides->members()) {
      const auto found =
          std::ranges::find_if(output->profiles, [&](const auto& entry) {
            return entry.first == name;
          });
      if (found == output->profiles.end())
        return error("'logit_bias_overrides' names '" + name +
                         "', which 'logit_bias_profiles' does not select",
                     "invalid_logit_bias_overrides");
      const auto parsed = bias(value);
      if (!parsed)
        return error("'logit_bias_overrides." + name +
                         "' must be a number between -100 and 100",
                     "invalid_logit_bias_overrides");
      found->second = *parsed;
    }
  }
  if (const auto* map = body.find("logit_bias");
      map != nullptr && !map->is_null()) {
    if (!map->is_object())
      return error("'logit_bias' must map token ids to biases",
                   "invalid_logit_bias");
    if (map->size() > sampling::kMaxRequestLogitBiasTokens)
      return error("'logit_bias' holds at most " +
                       std::to_string(sampling::kMaxRequestLogitBiasTokens) +
                       " tokens",
                   "invalid_logit_bias");
    for (const auto& [key, value] : map->members()) {
      // A decimal id without sign, leading zeros or spaces.
      const bool digits =
          !key.empty() && key.size() <= 10 &&
          std::ranges::all_of(key,
                              [](char c) { return c >= '0' && c <= '9'; }) &&
          (key.size() == 1 || key.front() != '0');
      const auto id = digits ? std::stoull(key) : 0ULL;
      if (!digits || id >= catalog.vocabulary_size())
        return error("'logit_bias' key '" + key +
                         "' must be a token id below " +
                         std::to_string(catalog.vocabulary_size()),
                     "invalid_logit_bias");
      const auto parsed = bias(value);
      if (!parsed)
        return error("'logit_bias." + key +
                         "' must be a number between -100 and 100",
                     "invalid_logit_bias");
      output->tokens.emplace_back(static_cast<sampling::TokenId>(id), *parsed);
    }
    std::ranges::sort(output->tokens);
  }
  return std::nullopt;
}

}  // namespace detail

/// Reads all shared sampling controls, using defaults for omitted fields.
/// A null `logit_bias` catalog (a backend whose samplers cannot apply logit
/// biases) rejects logit_bias_profiles, logit_bias_overrides and a nonempty
/// logit_bias as unsupported.
inline std::optional<SamplingRequestError> ParseSamplingConfig(
    const json::Value& body, const sampling::SamplingConfig& defaults,
    sampling::SamplingConfig* output,
    const sampling::LogitBiasCatalog* logit_bias = nullptr) {
  if (output == nullptr) {
    return SamplingRequestError{
        .message = "sampling output must not be null",
        .code = "invalid_sampling",
    };
  }
  *output = defaults;

  // These controls alter proposal/target probabilities. Reject unsupported
  // spellings instead of accepting a request with a different distribution.
  for (const auto& [field, value] : body.members()) {
    if ((field == "logit_bias" || field == "logit_bias_profiles" ||
         field == "logit_bias_overrides") &&
        (logit_bias != nullptr || value.is_null() ||
         (field == "logit_bias" && value.is_object() && value.empty())))
      continue;
    const bool draft_control = field.starts_with("draft_") ||
                               field.ends_with("_draft") || field == "draft" ||
                               field == "speculative";
    if (draft_control || field == "samplers" ||
        field.starts_with("logit_bias") ||
        ((field.starts_with("xtc_") || field.starts_with("dry_")) &&
         !detail::IsLlamaSamplerField(field))) {
      return SamplingRequestError{
          .message = "request field '" + field + "' is not supported",
          .code = "unsupported_sampling",
      };
    }
  }
  if (auto error = detail::CheckNeutralLlamaSamplers(body)) {
    return error;
  }

  if (auto error = detail::ReadSamplingFloat(body, "temperature",
                                             &output->temperature, true)) {
    return error;
  }
  if (auto error = detail::ReadSamplingInteger(
          body, "top_k", std::int32_t{0},
          std::numeric_limits<std::int32_t>::max(), &output->top_k)) {
    return error;
  }
  if (auto error =
          detail::ReadSamplingFloat(body, "top_p", &output->top_p, true)) {
    return error;
  }
  if (auto error = detail::ReadSamplingFloat(body, "min_p", &output->min_p)) {
    return error;
  }
  if (auto error = detail::ReadSamplingInteger(body, "min_keep", std::size_t{0},
                                               detail::kMaxSamplingWindow,
                                               &output->min_keep)) {
    return error;
  }
  if (auto error = detail::ReadSamplingInteger(body, "seed", std::int64_t{-1},
                                               detail::kMaxExactJsonInteger,
                                               &output->seed, true)) {
    return error;
  }
  if (auto error = detail::ReadSamplingFloat(body, "repeat_penalty",
                                             &output->repeat_penalty)) {
    return error;
  }
  if (auto error = detail::ReadSamplingInteger(
          body, "repeat_last_n", std::size_t{0}, detail::kMaxSamplingWindow,
          &output->repeat_last_n)) {
    return error;
  }
  if (auto error = detail::ReadSamplingFloat(
          body, "frequency_penalty", &output->frequency_penalty, true)) {
    return error;
  }
  if (auto error = detail::ReadSamplingFloat(body, "presence_penalty",
                                             &output->presence_penalty, true)) {
    return error;
  }
  if (auto error = detail::ReadSamplingFloat(body, "han_scale",
                                             &output->han_scale, true))
    return error;
  // Gufo extension: (0, 1], checked on the JSON number like the bounds below
  // and on the float, which a tiny positive number rounds to zero.
  if (const auto* scale = body.find("han_scale");
      scale && scale->is_number() &&
      !(scale->as_double() > 0.0 && scale->as_double() <= 1.0 &&
        output->han_scale > 0.0F)) {
    return SamplingRequestError{
        .message = "'han_scale' must be greater than 0 and at most 1",
        .code = "invalid_han_scale",
    };
  }
  // Gufo extension: where the Han scale applies; null inherits.
  if (const auto* scope = body.find("han_scale_scope");
      scope != nullptr && !scope->is_null()) {
    const auto parsed = scope->is_string()
                            ? sampling::ParseHanScaleScope(scope->str())
                            : std::nullopt;
    if (!parsed) {
      return SamplingRequestError{
          .message = "'han_scale_scope' must be \"all\" or \"answer\"",
          .code = "invalid_han_scale_scope",
      };
    }
    output->han_scale_scope = *parsed;
  }
  if (logit_bias != nullptr) {
    sampling::LogitBiasSelection selection;
    if (auto error =
            detail::ParseLogitBiasSelection(body, *logit_bias, &selection))
      return error;
    try {
      output->logit_bias = logit_bias->Resolve(selection);
    } catch (const std::invalid_argument& exception) {
      return SamplingRequestError{
          .message = exception.what(),
          .code = "invalid_logit_bias",
      };
    }
  }

  // Check the original JSON number, not the narrowed float: 1.00000001 must
  // not round into the permitted top-p range. Native sampler controls may
  // be broader; these bounds implement the public API contract.
  for (const auto& [field, minimum, maximum] :
       {std::tuple{"temperature", 0.0, 2.0},
        {"top_p", 0.0, 1.0},
        {"min_p", 0.0, 1.0},
        {"frequency_penalty", -2.0, 2.0},
        {"presence_penalty", -2.0, 2.0}}) {
    const auto* value = body.find(field);
    if (value && value->is_number() &&
        (value->as_double() < minimum || value->as_double() > maximum)) {
      return SamplingRequestError{
          .message = "'" + std::string(field) + "' must be between " +
                     std::to_string(minimum) + " and " +
                     std::to_string(maximum),
          .code = detail::SamplingErrorCode(field),
      };
    }
  }

  try {
    output->Validate();
  } catch (const std::invalid_argument& exception) {
    return SamplingRequestError{
        .message = exception.what(),
        .code = "invalid_sampling",
    };
  }
  if (output->temperature > 2.0F) {
    return SamplingRequestError{
        .message = "'temperature' must be between 0 and 2",
        .code = "invalid_temperature",
    };
  }
  for (const auto& [field, value] :
       {std::pair{"frequency_penalty", output->frequency_penalty},
        std::pair{"presence_penalty", output->presence_penalty}}) {
    if (value < -2.0F || value > 2.0F) {
      return SamplingRequestError{
          .message = "'" + std::string(field) + "' must be between -2 and 2",
          .code = detail::SamplingErrorCode(field),
      };
    }
  }
  return std::nullopt;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_SAMPLING_REQUEST_HPP_
