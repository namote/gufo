#include "src/models/qwen/abliteration.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"

namespace gufo::models::qwen {
namespace {
[[noreturn]] void Invalid(const std::string& what) {
  throw std::invalid_argument("abliteration: invalid " + what);
}

// Appends `size` finite numbers with |x| <= bound (or 0 <= x <= bound when
// `nonnegative`).
void AppendVector(const json::Value* value, const std::string& name,
                  std::size_t size, double bound, bool nonnegative,
                  std::vector<float>* out) {
  if (!value || !value->is_array() || value->size() != size)
    Invalid(name);
  for (const auto& item : value->items()) {
    const double x = item.as_double();
    if (!item.is_number() || !std::isfinite(x) || std::abs(x) > bound ||
        (nonnegative && x < 0))
      Invalid(name);
    out->push_back(static_cast<float>(x));
  }
}

void RequireUnit(std::span<const float> direction, const char* name) {
  double norm = 0;
  for (float x : direction)
    norm += double{x} * x;
  if (std::abs(norm - 1.0) > 1e-4)
    throw std::invalid_argument(std::string("abliteration: ") + name +
                                " must have unit norm");
}

void AppendU32(crypto::Sha256Hasher& hash, std::uint32_t value) {
  std::array<std::uint8_t, 4> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i)
    bytes[i] = static_cast<std::uint8_t>(value >> (8 * i));
  hash.Update(bytes);
}
}  // namespace

AbliterationProfile AbliterationProfile::Load(const std::filesystem::path& path,
                                              std::uint32_t hidden,
                                              std::uint32_t layers,
                                              std::string_view expected,
                                              std::string_view placement) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::invalid_argument("abliteration: cannot open profile");
  std::string bytes(kAbliterationMaxProfileBytes + 1, '\0');
  file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  bytes.resize(static_cast<std::size_t>(file.gcount()));
  if (bytes.size() > kAbliterationMaxProfileBytes)
    throw std::invalid_argument("abliteration: profile exceeds 16 MiB");
  const auto root = json::parse(bytes);
  const auto schema = root.member_str("schema");
  if ((schema != kAbliterationSchemaV1 && schema != kAbliterationSchemaV2) ||
      root.member_str("model_identity") != expected || expected.empty() ||
      placement.empty() || root.member_str("placement") != placement ||
      !root.find("hidden") || !root.find("hidden")->is_number() ||
      root.find("hidden")->as_double() != hidden || !root.find("layers") ||
      !root.find("layers")->is_number() ||
      root.find("layers")->as_double() != layers || hidden == 0 || layers == 0)
    throw std::invalid_argument("abliteration: incompatible profile/model");
  AbliterationProfile result;
  result.hidden = hidden;
  result.layers = layers;
  result.model_identity = std::string(expected);
  result.placement = std::string(placement);
  if (schema == kAbliterationSchemaV1) {
    // Unchanged v1 contract: strengths bounded in magnitude.
    result.schema = 1;
    AppendVector(root.find("direction"), "direction", hidden, 1.0, false,
                 &result.directions);
    RequireUnit(result.directions, "direction");
    AppendVector(root.find("attention"), "attention", layers,
                 kAbliterationMaxWeight, false, &result.attention);
    AppendVector(root.find("mlp"), "mlp", layers, kAbliterationMaxWeight, false,
                 &result.mlp);
  } else {
    result.schema = 2;
    const auto* directions = root.find("directions");
    if (!directions || !directions->is_array() || directions->size() != layers)
      Invalid("directions");
    result.directions.reserve(std::size_t{layers} * hidden);
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
      const std::string name = "directions[" + std::to_string(layer) + "]";
      AppendVector(&directions->items()[layer], name, hidden, 1.0, false,
                   &result.directions);
      RequireUnit(std::span(result.directions)
                      .subspan(std::size_t{layer} * hidden, hidden),
                  name.c_str());
    }
    const auto* layer = root.find("direction_layer");
    if (!layer || !layer->is_number() || !(layer->as_double() >= 0) ||
        layer->as_double() >= layers ||
        layer->as_double() != std::floor(layer->as_double()))
      Invalid("direction_layer");
    result.direction_layer = static_cast<std::uint32_t>(layer->as_double());
    AppendVector(root.find("attention"), "attention", layers,
                 kAbliterationMaxWeight, true, &result.attention);
    AppendVector(root.find("mlp"), "mlp", layers, kAbliterationMaxWeight, true,
                 &result.mlp);
  }
  result.identity = crypto::Sha256Hex(std::span(
      reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
  return result;
}

std::vector<float> AbliterationSettings::DeviceWords() const {
  if (attention.size() != mlp.size())
    throw std::logic_error("abliteration settings are inconsistent");
  const auto layers = static_cast<std::uint32_t>(attention.size());
  std::vector<float> words(AbliterationDeviceWords(layers));
  words[0] = std::bit_cast<float>(direction_row);
  for (std::uint32_t layer = 0; layer < layers; ++layer) {
    words[AbliterationWeightIndex(layers, layer, false)] = attention[layer];
    words[AbliterationWeightIndex(layers, layer, true)] = mlp[layer];
  }
  return words;
}

std::shared_ptr<const AbliterationSettings> ResolveAbliterationSettings(
    const AbliterationProfile& profile,
    const AbliterationOverrides& overrides) {
  const auto layers = profile.layers;
  auto result = std::make_shared<AbliterationSettings>();
  if (overrides.direction_layer) {
    if (!profile.selectable_direction())
      throw std::invalid_argument(
          "abliteration: this v1 profile has one fixed direction; "
          "'direction_layer' needs a gufo-abliteration-v2 profile");
    if (*overrides.direction_layer >= layers)
      throw std::invalid_argument(
          "abliteration: 'direction_layer' must be in [0, " +
          std::to_string(layers) + ")");
    result->direction_layer = overrides.direction_layer;
  } else {
    result->direction_layer = profile.direction_layer;
  }
  // v2 rows are indexed by layer; v1 has its one row.
  result->direction_row = result->direction_layer.value_or(0);
  if (result->direction_row >= profile.direction_rows())
    throw std::logic_error("abliteration direction row out of range");
  const auto strengths = [&](const std::vector<float>& requested,
                             const std::vector<float>& defaults,
                             const char* name) {
    if (requested.empty())
      return defaults;
    if (requested.size() != 1 && requested.size() != layers)
      throw std::invalid_argument(std::string("abliteration: '") + name +
                                  "' must be one number or " +
                                  std::to_string(layers) + " numbers");
    for (float x : requested)
      if (!std::isfinite(x) || x < 0 || x > kAbliterationMaxWeight)
        throw std::invalid_argument(std::string("abliteration: '") + name +
                                    "' strengths must be in [0, 4]");
    // -0 and +0 are one strength, so they share one identity.
    auto result = requested.size() == 1
                      ? std::vector<float>(layers, requested[0])
                      : requested;
    for (float& x : result)
      x = x == 0 ? 0.0F : x;
    return result;
  };
  result->attention =
      strengths(overrides.attention, profile.attention, "attention");
  result->mlp = strengths(overrides.mlp, profile.mlp, "mlp");
  crypto::Sha256Hasher hash;
  const std::string prefix =
      "gufo-abliteration-settings-v1:" + profile.identity + ":";
  hash.Update(std::span(reinterpret_cast<const std::uint8_t*>(prefix.data()),
                        prefix.size()));
  AppendU32(hash, result->direction_layer.value_or(0xFFFFFFFFU));
  AppendU32(hash, layers);
  for (const auto* values : {&result->attention, &result->mlp})
    for (float x : *values)
      AppendU32(hash, std::bit_cast<std::uint32_t>(x));
  result->identity = hash.FinishHex();
  return result;
}

std::vector<std::uint8_t> AbliterationInputIdentity(
    std::span<const std::uint8_t> input, std::string_view settings_identity) {
  const std::string prefix =
      "gufo-abliteration-input-v1:" + std::string(settings_identity) + ":";
  crypto::Sha256Hasher hash;
  hash.Update(std::span(reinterpret_cast<const std::uint8_t*>(prefix.data()),
                        prefix.size()));
  hash.Update(input);
  const auto digest = hash.Finish();
  return {digest.begin(), digest.end()};
}

std::shared_ptr<const AbliterationProfile> LoadAbliterationOption(
    const std::string& option, std::uint32_t hidden, std::uint32_t layers,
    std::string_view model_identity, std::string_view placement) {
  if (option.empty())
    return nullptr;
  return std::make_shared<const AbliterationProfile>(AbliterationProfile::Load(
      option, hidden, layers, model_identity, placement));
}

AbliterationContinuationRecord RecordAbliterationContinuation(
    std::span<const std::uint8_t> image, const AbliterationSettings* active) {
  AbliterationContinuationRecord record;
  if (!image.empty() && image.size() != record.image.size())
    throw std::logic_error("invalid Qwen image identity");
  if (!active) {
    std::ranges::copy(image, record.image.begin());
    return record;
  }
  record.mode = AbliterationContinuationRecord::kOn;
  const auto salted = AbliterationInputIdentity(image, active->identity);
  std::ranges::copy(salted, record.image.begin());
  return record;
}

bool AbliterationContinuationMatches(std::uint32_t mode,
                                     std::uint32_t image_bytes,
                                     std::span<const std::uint8_t, 32> recorded,
                                     std::span<const std::uint8_t> image,
                                     const AbliterationSettings* active) {
  if (!active) {
    // The unchanged OFF contract: an imageless entry checks no identity.
    return mode == AbliterationContinuationRecord::kOff &&
           (image_bytes == 0 || (image.size() == recorded.size() &&
                                 std::ranges::equal(image, recorded)));
  }
  return mode == AbliterationContinuationRecord::kOn &&
         image_bytes == image.size() &&
         std::ranges::equal(AbliterationInputIdentity(image, active->identity),
                            recorded);
}
}  // namespace gufo::models::qwen
