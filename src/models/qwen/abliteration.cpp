#include "src/models/qwen/abliteration.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
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

// Standard base64 (exact length, padding only at the end) of little-endian
// float32 values; unused low bits of a padded group are ignored.
std::vector<float> DecodeFloats(const json::Value& value,
                                const std::string& name, std::size_t count) {
  constexpr std::string_view kAlphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  if (!value.is_string())
    Invalid(name);
  const std::string_view text = value.str();
  const std::size_t bytes = count * sizeof(float);
  if (text.size() != (bytes + 2) / 3 * 4)
    Invalid(name);
  std::vector<std::uint8_t> raw;
  raw.reserve(text.size() / 4 * 3);
  for (std::size_t i = 0; i < text.size(); i += 4) {
    std::uint32_t group = 0;
    std::size_t padding = 0;
    for (std::size_t j = 0; j < 4; ++j) {
      const char c = text[i + j];
      if (c == '=' && i + 4 == text.size() && j >= 2) {
        ++padding;
        group <<= 6;
        continue;
      }
      const auto digit = kAlphabet.find(c);
      if (digit == std::string_view::npos || padding)
        Invalid(name);
      group = group << 6 | static_cast<std::uint32_t>(digit);
    }
    for (std::size_t j = 0; j < 3 - padding; ++j)
      raw.push_back(static_cast<std::uint8_t>(group >> (16 - 8 * j)));
  }
  if (raw.size() != bytes)
    Invalid(name);
  std::vector<float> result(count);
  for (std::size_t i = 0; i < count; ++i) {
    std::uint32_t bits = 0;
    for (std::size_t b = 0; b < 4; ++b)
      bits |= std::uint32_t{raw[i * 4 + b]} << (8 * b);
    result[i] = std::bit_cast<float>(bits);
    if (!std::isfinite(result[i]) || std::abs(result[i]) > 1)
      Invalid(name);
  }
  return result;
}

// Consecutive rows of `hidden` floats must be orthonormal: the subspace
// projection removes sum_j d_j dot(d_j, y) in one pass.
void RequireOrthonormal(std::span<const float> block, std::size_t hidden,
                        const std::string& name) {
  const std::size_t rows = block.size() / hidden;
  for (std::size_t a = 0; a < rows; ++a) {
    RequireUnit(block.subspan(a * hidden, hidden), name.c_str());
    for (std::size_t b = 0; b < a; ++b) {
      double dot = 0;
      for (std::size_t i = 0; i < hidden; ++i)
        dot += double{block[a * hidden + i]} * block[b * hidden + i];
      if (std::abs(dot) > 1e-4)
        throw std::invalid_argument("abliteration: " + name +
                                    " must be orthonormal");
    }
  }
}

// A JSON integer in [low, high].
std::uint32_t ReadCount(const json::Value* value, const char* name,
                        std::uint32_t low, std::uint32_t high) {
  if (!value || !value->is_number() || !(value->as_double() >= low) ||
      value->as_double() > high ||
      value->as_double() != std::floor(value->as_double()))
    Invalid(name);
  return static_cast<std::uint32_t>(value->as_double());
}

void AppendU32(crypto::Sha256Hasher& hash, std::uint32_t value) {
  std::array<std::uint8_t, 4> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i)
    bytes[i] = static_cast<std::uint8_t>(value >> (8 * i));
  hash.Update(bytes);
}
// The optional subspace members of a v2 profile; `result` holds the rank-1
// mean-difference rows on entry and the complete row layout on return.
void LoadSubspace(const json::Value& root, AbliterationProfile* result) {
  const auto hidden = std::size_t{result->hidden};
  const auto layers = result->layers;
  if (const auto* rank = root.find("rank"))
    result->rank = ReadCount(rank, "rank", 1, kAbliterationMaxDirections);
  const auto rank = result->rank;
  const std::vector<float> first = std::move(result->directions);
  result->directions.assign(std::size_t{layers} * rank * hidden, 0.0F);
  const auto* subspace = root.find("subspace_directions");
  if ((rank > 1) != (subspace != nullptr) ||
      (subspace && (!subspace->is_array() || subspace->size() != layers)))
    Invalid("subspace_directions (one string per layer, only when rank > 1)");
  for (std::uint32_t layer = 0; layer < layers; ++layer) {
    float* block =
        result->directions.data() + std::size_t{layer} * rank * hidden;
    std::copy_n(first.begin() + static_cast<std::ptrdiff_t>(layer * hidden),
                hidden, block);
    const std::string name =
        "subspace_directions[" + std::to_string(layer) + "]";
    if (rank > 1) {
      const auto rest =
          DecodeFloats(subspace->items()[layer], name, (rank - 1) * hidden);
      std::ranges::copy(rest, block + hidden);
      RequireOrthonormal(std::span<const float>(block, rank * hidden), hidden,
                         name);
    }
  }
  if (const auto* projected = root.find("projected_directions")) {
    if (!projected->is_array() || projected->size() != layers)
      Invalid("projected_directions");
    result->kinds = 2;
    result->directions.reserve(result->directions.size() * 2);
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
      const std::string name =
          "projected_directions[" + std::to_string(layer) + "]";
      const auto block =
          DecodeFloats(projected->items()[layer], name, rank * hidden);
      RequireOrthonormal(block, hidden, name);
      result->directions.insert(result->directions.end(), block.begin(),
                                block.end());
    }
  }
  if (const auto* kind = root.find("direction_kind")) {
    const auto parsed = kind->is_string()
                            ? ParseAbliterationDirectionKind(kind->str())
                            : std::nullopt;
    if (!parsed || !result->has_kind(*parsed))
      Invalid(
          "direction_kind (mean_diff, or projected with "
          "projected_directions)");
    result->direction_kind = *parsed;
  }
  if (const auto* count = root.find("direction_count"))
    result->direction_count = ReadCount(count, "direction_count", 1, rank);
  if (const auto* preserve = root.find("preserve_norm")) {
    if (!preserve->is_bool())
      Invalid("preserve_norm");
    result->preserve_norm = preserve->as_bool();
  }
  if (const auto* method = root.find("subspace_method")) {
    if (!method->is_string() ||
        (method->str() != "pca" && method->str() != "som"))
      Invalid("subspace_method");
    result->subspace_method = method->str();
  }
}

// The optional v2 `trigger` member (fit_trigger.py).
AbliterationTrigger LoadTrigger(const json::Value& value,
                                const AbliterationProfile& profile) {
  if (!value.is_object() ||
      value.member_str("schema") != kAbliterationTriggerSchema)
    Invalid("trigger (schema gufo-abliteration-trigger-v1)");
  if (value.member_str("readout") != AbliterationReadout(profile.placement))
    Invalid("trigger.readout (not this placement's collector readout)");
  AbliterationTrigger trigger;
  trigger.layer =
      ReadCount(value.find("layer"), "trigger.layer", 0, profile.layers - 1);
  const auto* direction = value.find("direction");
  if (!direction)
    Invalid("trigger.direction");
  trigger.direction =
      DecodeFloats(*direction, "trigger.direction", profile.hidden);
  RequireUnit(trigger.direction, "trigger.direction");
  const auto* threshold = value.find("threshold");
  if (!threshold || !threshold->is_number() ||
      !std::isfinite(threshold->as_double()) ||
      std::abs(threshold->as_double()) > 1e6)
    Invalid("trigger.threshold");
  trigger.threshold = static_cast<float>(threshold->as_double());
  if (const auto* decode = value.find("decode_threshold")) {
    if (!decode->is_number() || !std::isfinite(decode->as_double()) ||
        std::abs(decode->as_double()) > 1e6)
      Invalid("trigger.decode_threshold");
    trigger.decode_threshold = static_cast<float>(decode->as_double());
  }
  return trigger;
}

constexpr std::array<std::string_view, 7> kSubspaceMembers{
    "rank",           "subspace_directions", "projected_directions",
    "direction_kind", "direction_count",     "preserve_norm",
    "subspace_method"};
}  // namespace

std::string_view AbliterationReadout(std::string_view placement) noexcept {
  if (placement == kAbliterationWriterBeforeHc)
    return "last-prompt-token-post-layer-hc-mean-fp32-v1";
  if (placement == kAbliterationWriterBeforeResidual)
    return "last-prompt-token-post-layer-fp32-v1";
  return {};
}

std::string_view AbliterationDirectionKindName(
    AbliterationDirectionKind kind) noexcept {
  return kind == AbliterationDirectionKind::kProjected ? "projected"
                                                       : "mean_diff";
}

std::optional<AbliterationDirectionKind> ParseAbliterationDirectionKind(
    std::string_view name) noexcept {
  if (name == "mean_diff")
    return AbliterationDirectionKind::kMeanDiff;
  if (name == "projected")
    return AbliterationDirectionKind::kProjected;
  return std::nullopt;
}

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
    throw std::invalid_argument("abliteration: profile exceeds 48 MiB");
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
    // Residual strengths and the subspace members are v2 only.
    if (root.find("residual"))
      Invalid("residual (needs gufo-abliteration-v2)");
    for (const auto member : kSubspaceMembers)
      if (root.find(std::string(member)))
        Invalid(std::string(member) + " (needs gufo-abliteration-v2)");
    if (root.find("trigger"))
      Invalid("trigger (needs gufo-abliteration-v2)");
    result.residual.assign(layers, 0.0F);
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
    // Optional: profiles written before residual strengths existed load with
    // all zeros, so their behavior is unchanged.
    if (const auto* residual = root.find("residual"))
      AppendVector(residual, "residual", layers, kAbliterationMaxWeight, true,
                   &result.residual);
    else
      result.residual.assign(layers, 0.0F);
    // Optional: without them, rank 1, mean difference, no norm preservation,
    // and `directions` keeps exactly its rows.
    LoadSubspace(root, &result);
    if (const auto* trigger = root.find("trigger"))
      result.trigger = LoadTrigger(*trigger, result);
  }
  result.name = path.stem().string();
  result.identity = crypto::Sha256Hex(std::span(
      reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
  return result;
}

std::vector<float> AbliterationSettings::DeviceWords(
    std::uint32_t row_offset) const {
  if (attention.size() != mlp.size() || attention.size() != residual.size())
    throw std::logic_error("abliteration settings are inconsistent");
  const auto layers = static_cast<std::uint32_t>(attention.size());
  if (row_offset > kAbliterationRowMask ||
      direction_row > kAbliterationRowMask - row_offset ||
      direction_count == 0 || direction_count > kAbliterationMaxDirections)
    throw std::logic_error("abliteration settings are inconsistent");
  std::vector<float> words(AbliterationDeviceWords(layers));
  words[0] =
      std::bit_cast<float>((direction_row + row_offset) |
                           (direction_count - 1) << kAbliterationCountShift |
                           (preserve_norm ? kAbliterationPreserveNormBit : 0U));
  using enum AbliterationPoint;
  for (std::uint32_t layer = 0; layer < layers; ++layer) {
    words[AbliterationWeightIndex(layers, layer, kAttention)] =
        attention[layer];
    words[AbliterationWeightIndex(layers, layer, kMlp)] = mlp[layer];
    words[AbliterationWeightIndex(layers, layer, kResidual)] = residual[layer];
  }
  return words;
}

bool AbliterationSettings::ResidualActive() const noexcept {
  return std::ranges::any_of(residual, [](float x) { return x != 0; });
}

std::shared_ptr<const AbliterationSettings> ResolveAbliterationSettings(
    const AbliterationProfile& profile,
    const AbliterationOverrides& overrides) {
  const auto layers = profile.layers;
  auto result = std::make_shared<AbliterationSettings>();
  result->profile = profile.name;
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
  result->direction_kind =
      overrides.direction_kind.value_or(profile.direction_kind);
  if (!profile.has_kind(result->direction_kind))
    throw std::invalid_argument(
        "abliteration: the loaded profile has no '" +
        std::string(AbliterationDirectionKindName(result->direction_kind)) +
        "' directions");
  result->direction_count =
      overrides.direction_count.value_or(profile.direction_count);
  if (result->direction_count < 1 || result->direction_count > profile.rank)
    throw std::invalid_argument(
        "abliteration: 'directions' must be an integer in [1, " +
        std::to_string(profile.rank) + "]");
  result->preserve_norm =
      overrides.preserve_norm.value_or(profile.preserve_norm);
  // v2 rows are indexed by kind, layer and rank; v1 has its one row.
  result->direction_row = profile.DirectionRow(
      result->direction_kind, result->direction_layer.value_or(0));
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
  result->residual =
      strengths(overrides.residual, profile.residual, "residual");
  if (!AbliterationSettingsFit(profile, *result))
    throw std::logic_error("abliteration direction row out of range");
  crypto::Sha256Hasher hash;
  // v2: the residual strengths follow the writer strengths. v3: then the
  // direction kind, the direction count and norm preservation.
  const std::string prefix =
      "gufo-abliteration-settings-v3:" + profile.identity + ":";
  hash.Update(std::span(reinterpret_cast<const std::uint8_t*>(prefix.data()),
                        prefix.size()));
  AppendU32(hash, result->direction_layer.value_or(0xFFFFFFFFU));
  AppendU32(hash, layers);
  for (const auto* values :
       {&result->attention, &result->mlp, &result->residual})
    for (float x : *values)
      AppendU32(hash, std::bit_cast<std::uint32_t>(x));
  AppendU32(hash, static_cast<std::uint32_t>(result->direction_kind));
  AppendU32(hash, result->direction_count);
  AppendU32(hash, result->preserve_norm ? 1U : 0U);
  result->identity = hash.FinishHex();
  return result;
}

std::shared_ptr<const AbliterationSettings> DynamicAbliterationSettings(
    const AbliterationSettings& base, std::uint32_t position) {
  if (base.dynamic_from)
    throw std::logic_error("abliteration settings are already dynamic");
  auto result = std::make_shared<AbliterationSettings>(base);
  result->dynamic_from = position;
  crypto::Sha256Hasher hash;
  const std::string prefix =
      "gufo-abliteration-dynamic-v1:" + base.identity + ":";
  hash.Update(std::span(reinterpret_cast<const std::uint8_t*>(prefix.data()),
                        prefix.size()));
  AppendU32(hash, position);
  result->identity = hash.FinishHex();
  return result;
}

const AbliterationSettings* AbliterationForState(
    const AbliterationSettings* active, std::size_t tokens) noexcept {
  if (active && active->dynamic_from && tokens <= *active->dynamic_from)
    return nullptr;
  return active;
}

bool AbliterationSettingsFit(const AbliterationProfile& profile,
                             const AbliterationSettings& settings) noexcept {
  return settings.attention.size() == profile.layers &&
         settings.mlp.size() == profile.layers &&
         settings.residual.size() == profile.layers &&
         settings.direction_count >= 1 &&
         settings.direction_count <= profile.rank &&
         settings.direction_row <= kAbliterationRowMask &&
         std::size_t{settings.direction_row} + settings.direction_count <=
             profile.direction_rows();
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

std::vector<std::string> ListAbliterationProfiles(const std::string& option) {
  if (option.empty())
    return {};
  namespace fs = std::filesystem;
  std::error_code error;
  if (!fs::is_directory(option, error))
    throw std::invalid_argument(
        "abliteration: --abliteration-profiles needs a directory");
  std::vector<std::string> files;
  for (const auto& entry : fs::directory_iterator(option)) {
    if (entry.path().extension() != ".json")
      continue;
    if (!entry.is_regular_file())
      throw std::invalid_argument("abliteration: profile " +
                                  entry.path().filename().string() +
                                  " is not a regular file");
    files.push_back(entry.path().string());
  }
  if (files.empty())
    throw std::invalid_argument(
        "abliteration: no *.json profile in --abliteration-profiles");
  if (files.size() > kAbliterationMaxProfiles)
    throw std::invalid_argument("abliteration: more than " +
                                std::to_string(kAbliterationMaxProfiles) +
                                " profiles");
  std::ranges::sort(files);
  return files;
}

const AbliterationProfile& SelectAbliterationProfile(
    std::span<const std::shared_ptr<const AbliterationProfile>> profiles,
    std::string_view name) {
  const auto names = [&] {
    std::string list;
    for (const auto& profile : profiles)
      list += (list.empty() ? "'" : ", '") + profile->name + "'";
    return list.empty() ? std::string("none") : list;
  };
  if (name.empty()) {
    if (profiles.size() == 1)
      return *profiles.front();
    throw std::invalid_argument(
        "abliteration: name one of the loaded profiles in "
        "'abliteration.profile': " +
        names());
  }
  for (const auto& profile : profiles)
    if (profile->name == name)
      return *profile;
  throw std::invalid_argument("abliteration: unknown profile '" +
                              std::string(name) + "'; loaded: " + names());
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
