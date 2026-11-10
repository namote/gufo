// Qwen3.8-27B runtime abliteration: profile binding (placement, identity,
// geometry) and the component-cache metadata guard, whose OFF bytes must stay
// those of every pre-abliteration checkpoint.
#include "src/models/qwen/abliteration.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen/continuation_layout.hpp"

namespace qwen = gufo::models::qwen;
namespace json = gufo::json;

namespace {
void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
template<class F>
bool Rejects(F&& fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}
// The metadata component bytes a capture writes: the record zero-filled to
// kContinuationMetadataBytes.
std::array<std::byte, qwen::kContinuationMetadataBytes> Bytes(
    const qwen::ContinuationMetadata& metadata) {
  std::array<std::byte, qwen::kContinuationMetadataBytes> bytes{};
  std::memcpy(bytes.data(), &metadata, sizeof(metadata));
  return bytes;
}
// Pre-abliteration metadata, written independently: six little-endian u32
// (version 1, position, logits_valid, has_next, next, image_bytes), the raw
// 32-byte image identity, then zeros.
std::array<std::byte, qwen::kContinuationMetadataBytes> Legacy(
    std::uint32_t position, std::span<const std::uint8_t> image) {
  std::array<std::byte, qwen::kContinuationMetadataBytes> bytes{};
  const std::uint32_t fields[] = {
      1, position, 1, 1, 7, static_cast<std::uint32_t>(image.size())};
  for (std::size_t i = 0; i < std::size(fields); ++i)
    for (std::size_t b = 0; b < 4; ++b)
      bytes[i * 4 + b] = static_cast<std::byte>(fields[i] >> (8 * b));
  for (std::size_t i = 0; i < image.size(); ++i)
    bytes[24 + i] = static_cast<std::byte>(image[i]);
  return bytes;
}
qwen::ContinuationMetadata Current(std::uint32_t position) {
  qwen::ContinuationMetadata metadata;
  metadata.position = position;
  metadata.logits_valid = 1;
  metadata.has_next = 1;
  metadata.next = 7;
  return metadata;
}

std::uint32_t Bits(float x) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &x, sizeof(bits));
  return bits;
}

// A v1 profile: its defaults, strength overrides, and no direction choice.
void TestV1Settings(const qwen::AbliterationProfile& profile) {
  const auto defaults = qwen::ResolveAbliterationSettings(profile);
  Require(!defaults->direction_layer && defaults->direction_row == 0 &&
              defaults->attention == profile.attention &&
              defaults->mlp == profile.mlp && defaults->identity.size() == 64,
          "v1 default settings");
  Require(Rejects([&] {
            (void)qwen::ResolveAbliterationSettings(profile,
                                                    {.direction_layer = 0});
          }),
          "v1 profile accepted a direction layer");
  const auto scaled =
      qwen::ResolveAbliterationSettings(profile, {.attention = {2}});
  Require(scaled->attention == std::vector<float>{2, 2} &&
              scaled->mlp == profile.mlp &&
              scaled->identity != defaults->identity,
          "v1 strength override");
}

void HashU32(gufo::crypto::Sha256Hasher& hash, std::uint32_t value) {
  const std::array<std::uint8_t, 4> bytes{
      static_cast<std::uint8_t>(value), static_cast<std::uint8_t>(value >> 8),
      static_cast<std::uint8_t>(value >> 16),
      static_cast<std::uint8_t>(value >> 24)};
  hash.Update(bytes);
}

// The settings identity, recomputed independently from its documented
// layout: SHA-256 of "gufo-abliteration-settings-v3:<profile identity>:", the
// direction layer (0xFFFFFFFF for v1) and the layer count as little-endian
// u32, the float bits of attention, mlp and residual, then the direction
// kind (0 mean_diff, 1 projected), the direction count and norm
// preservation (0/1) as u32.
std::string ExpectedIdentity(const qwen::AbliterationProfile& profile,
                             const qwen::AbliterationSettings& settings) {
  gufo::crypto::Sha256Hasher hash;
  const std::string prefix =
      "gufo-abliteration-settings-v3:" + profile.identity + ":";
  hash.Update(std::span(reinterpret_cast<const std::uint8_t*>(prefix.data()),
                        prefix.size()));
  HashU32(hash, settings.direction_layer.value_or(0xFFFFFFFFU));
  HashU32(hash, profile.layers);
  for (const auto* values :
       {&settings.attention, &settings.mlp, &settings.residual})
    for (float x : *values)
      HashU32(hash, Bits(x));
  HashU32(hash, static_cast<std::uint32_t>(settings.direction_kind));
  HashU32(hash, settings.direction_count);
  HashU32(hash, settings.preserve_norm ? 1 : 0);
  return hash.FinishHex();
}

// Standard padded base64 of little-endian float32 values, written
// independently of the loader's decoder.
std::string Base64(const std::vector<float>& values) {
  constexpr char kAlphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::vector<std::uint8_t> bytes;
  for (float x : values)
    for (int b = 0; b < 4; ++b)
      bytes.push_back(static_cast<std::uint8_t>(Bits(x) >> (8 * b)));
  std::string text;
  for (std::size_t i = 0; i < bytes.size(); i += 3) {
    const std::size_t n = std::min<std::size_t>(3, bytes.size() - i);
    std::uint32_t group = 0;
    for (std::size_t j = 0; j < 3; ++j)
      group = group << 8 | (j < n ? bytes[i + j] : 0U);
    for (std::size_t j = 0; j < 4; ++j)
      text += j <= n ? kAlphabet[(group >> (18 - 6 * j)) & 63] : '=';
  }
  return text;
}

json::Value Strings(const std::vector<std::vector<float>>& rows) {
  json::Value array = json::Value::array();
  for (const auto& row : rows)
    array.push_back(Base64(row));
  return array;
}

// Projected, multi-direction and norm-preserving settings: the optional v2
// members, their validation, request overrides, device word 0, identities,
// and the unchanged behavior of a profile without them.
void TestSubspace(const std::filesystem::path& path, std::string_view site,
                  const json::Value& v2) {
  const auto write = [&](const json::Value& value) {
    std::ofstream(path) << value.dump();
  };
  const auto load = [&] {
    return qwen::AbliterationProfile::Load(path, 3, 2, "fixture", site);
  };
  // Backward compatibility: rank 1, mean difference, no norm preservation;
  // the rows and device word 0 are exactly the single-direction ones.
  write(v2);
  const auto legacy = load();
  const auto legacy_defaults = qwen::ResolveAbliterationSettings(legacy);
  Require(
      legacy.rank == 1 && legacy.kinds == 1 && legacy.directions.size() == 6 &&
          legacy.directions[5] == 1 &&
          legacy.direction_kind == qwen::AbliterationDirectionKind::kMeanDiff &&
          legacy.direction_count == 1 && !legacy.preserve_norm &&
          legacy.subspace_method.empty() &&
          legacy_defaults->direction_row == 1 &&
          legacy_defaults->direction_count == 1 &&
          !legacy_defaults->preserve_norm &&
          Bits(legacy_defaults->DeviceWords()[0]) == 1 &&
          legacy_defaults->identity ==
              ExpectedIdentity(legacy, *legacy_defaults),
      "profile without subspace members");
  Require(Rejects([&] {
            (void)qwen::ResolveAbliterationSettings(
                legacy, {.direction_kind =
                             qwen::AbliterationDirectionKind::kProjected});
          }) &&
              Rejects([&] {
                (void)qwen::ResolveAbliterationSettings(legacy,
                                                        {.direction_count = 2});
              }),
          "legacy profile accepted a missing kind or rank");
  const auto legacy_norm =
      qwen::ResolveAbliterationSettings(legacy, {.preserve_norm = true});
  Require(legacy_norm->identity != legacy_defaults->identity &&
              Bits(legacy_norm->DeviceWords()[0]) ==
                  (1U | qwen::kAbliterationPreserveNormBit),
          "norm preservation on a rank-1 profile");

  // Rank 2 with both kinds. Mean difference: directions[l] then
  // subspace_directions[l]; projected: projected_directions[l] (2 rows).
  auto full = v2;
  full["rank"] = 2;
  full["subspace_directions"] = Strings({{-0.8F, 0.6F, 0}, {1, 0, 0}});
  full["projected_directions"] =
      Strings({{0, 0, 1, 1, 0, 0}, {0, 1, 0, 1, 0, 0}});
  full["subspace_method"] = "som";
  write(full);
  const auto profile = load();
  Require(profile.rank == 2 && profile.kinds == 2 &&
              profile.direction_rows() == 8 &&
              profile.subspace_method == "som" &&
              profile.has_kind(qwen::AbliterationDirectionKind::kProjected),
          "rank-2 profile load");
  // Rows: [kind][layer][rank]: kind 0 layer 1 starts at row 2, kind 1
  // layer 0 at row 4, kind 1 layer 1 at row 6.
  const auto row = [&](std::size_t r) {
    return std::vector<float>(profile.directions.begin() + r * 3,
                              profile.directions.begin() + r * 3 + 3);
  };
  Require(row(0) == std::vector<float>{0.6F, 0.8F, 0} &&
              row(1) == std::vector<float>{-0.8F, 0.6F, 0} &&
              row(2) == std::vector<float>{0, 0, 1} &&
              row(3) == std::vector<float>{1, 0, 0} &&
              row(4) == std::vector<float>{0, 0, 1} &&
              row(6) == std::vector<float>{0, 1, 0} &&
              row(7) == std::vector<float>{1, 0, 0},
          "rank-2 row layout");
  using enum qwen::AbliterationDirectionKind;
  const auto defaults = qwen::ResolveAbliterationSettings(profile);
  Require(defaults->direction_row == 2 && defaults->direction_count == 1 &&
              defaults->direction_kind == kMeanDiff &&
              Bits(defaults->DeviceWords()[0]) == 2 &&
              defaults->identity == ExpectedIdentity(profile, *defaults),
          "rank-2 defaults");
  const auto all =
      qwen::ResolveAbliterationSettings(profile, {.direction_layer = 0,
                                                  .direction_kind = kProjected,
                                                  .direction_count = 2,
                                                  .preserve_norm = true});
  Require(all->direction_row == 4 && all->direction_count == 2 &&
              all->preserve_norm &&
              Bits(all->DeviceWords()[0]) ==
                  (4U | 1U << qwen::kAbliterationCountShift |
                   qwen::kAbliterationPreserveNormBit) &&
              all->identity == ExpectedIdentity(profile, *all) &&
              qwen::AbliterationSettingsFit(profile, *all),
          "projected rank-2 settings");
  // Each option is its own identity; explicit defaults are the defaults.
  const auto projected = qwen::ResolveAbliterationSettings(
      profile, {.direction_kind = kProjected});
  const auto two =
      qwen::ResolveAbliterationSettings(profile, {.direction_count = 2});
  const auto norm =
      qwen::ResolveAbliterationSettings(profile, {.preserve_norm = true});
  const auto explicit_defaults =
      qwen::ResolveAbliterationSettings(profile, {.direction_kind = kMeanDiff,
                                                  .direction_count = 1,
                                                  .preserve_norm = false});
  Require(projected->identity != defaults->identity &&
              two->identity != defaults->identity &&
              norm->identity != defaults->identity &&
              projected->identity != two->identity &&
              two->identity != norm->identity &&
              explicit_defaults->identity == defaults->identity &&
              projected->direction_row == 6 && two->direction_row == 2,
          "option identities");
  for (const auto& overrides : std::vector<qwen::AbliterationOverrides>{
           {.direction_count = 0}, {.direction_count = 3}}) {
    Require(Rejects([&] {
              (void)qwen::ResolveAbliterationSettings(profile, overrides);
            }),
            "direction count outside [1, rank] resolved");
  }
  auto misfit = *two;
  misfit.direction_count = 3;
  Require(!qwen::AbliterationSettingsFit(profile, misfit) &&
              !qwen::AbliterationSettingsFit(legacy, *two),
          "settings fit");
  // The continuation guard binds the options too.
  auto metadata = Current(4);
  metadata.RecordImage({}, norm.get());
  Require(metadata.ImageMatches({}, norm.get()) &&
              !metadata.ImageMatches({}, defaults.get()),
          "option-bound continuation guard");

  // Profile defaults of the options.
  auto defaulted = full;
  defaulted["direction_kind"] = "projected";
  defaulted["direction_count"] = 2;
  defaulted["preserve_norm"] = true;
  write(defaulted);
  const auto with_defaults = load();
  const auto resolved = qwen::ResolveAbliterationSettings(with_defaults);
  Require(with_defaults.direction_kind == kProjected &&
              with_defaults.direction_count == 2 &&
              with_defaults.preserve_norm && resolved->direction_row == 6 &&
              resolved->direction_count == 2 && resolved->preserve_norm &&
              qwen::ResolveAbliterationSettings(with_defaults,
                                                {.direction_kind = kMeanDiff,
                                                 .direction_count = 1,
                                                 .preserve_norm = false})
                      ->direction_row == 2,
          "profile option defaults");

  // Invalid members.
  const auto rejects = [&](json::Value value) {
    write(value);
    return Rejects(load);
  };
  const auto with = [&](const char* key, json::Value value) {
    auto result = full;
    result[key] = std::move(value);
    return result;
  };
  auto rank_only = v2;
  rank_only["rank"] = 2;
  auto subspace_only = v2;
  subspace_only["subspace_directions"] = full["subspace_directions"];
  auto projected_rank1 = v2;
  projected_rank1["projected_directions"] =
      Strings({{0, 0, 1}, {0, 1, 0, 1, 0, 0}});
  auto kind_only = v2;
  kind_only["direction_kind"] = "projected";
  auto count_only = v2;
  count_only["direction_count"] = 2;
  std::string bad_char = Base64({-0.8F, 0.6F, 0});
  bad_char[0] = '*';
  std::string bad_padding = Base64({-0.8F, 0.6F, 0});
  bad_padding[4] = '=';
  // Layer 0's string as given, layer 1's valid.
  const auto texts = [](const std::string& first) {
    json::Value array = json::Value::array();
    array.push_back(first);
    array.push_back(Base64({1, 0, 0}));
    return array;
  };
  Require(!Rejects([&] {
    auto valid = full;
    valid["subspace_directions"] = texts(Base64({-0.8F, 0.6F, 0}));
    write(valid);
    (void)load();
  }),
          "valid subspace strings rejected");
  Require(
      rejects(with("rank", 9)) && rejects(with("rank", 0)) &&
          rejects(with("rank", 1.5)) && rejects(with("rank", "2")) &&
          rejects(rank_only) && rejects(subspace_only) &&
          rejects(projected_rank1) && rejects(kind_only) &&
          rejects(count_only) &&
          rejects(with("subspace_directions",
                       Strings({{0.6F, 0.8F, 0}, {1, 0, 0}}))) &&
          rejects(with("subspace_directions",
                       Strings({{-0.8F, 0.6F}, {1, 0, 0}}))) &&
          rejects(with("subspace_directions", Strings({{-0.8F, 0.6F, 0}}))) &&
          rejects(with("subspace_directions",
                       Strings({{-0.8F, 0.6F, 0}, {1, 1, 0}}))) &&
          rejects(with("subspace_directions",
                       json::parse(R"([[-0.8,0.6,0],[1,0,0]])"))) &&
          rejects(with("subspace_directions", texts(bad_char))) &&
          rejects(with("subspace_directions", texts(bad_padding))) &&
          rejects(with("projected_directions",
                       Strings({{0, 0, 1, 0, 0, 1}, {0, 1, 0, 1, 0, 0}}))) &&
          rejects(
              with("projected_directions", Strings({{0, 0, 1, 1, 0, 0}}))) &&
          rejects(with("direction_kind", "pca")) &&
          rejects(with("direction_kind", 1)) &&
          rejects(with("direction_count", 3)) &&
          rejects(with("direction_count", 0)) &&
          rejects(with("preserve_norm", "yes")) &&
          rejects(with("subspace_method", "kmeans")),
      "invalid subspace members loaded");
  // NaN bits in a base64 row.
  Require(rejects(with("subspace_directions",
                       Strings({{std::numeric_limits<float>::quiet_NaN(), 0, 0},
                                {1, 0, 0}}))),
          "nonfinite subspace loaded");
  // v1 never carries them.
  auto v1 = json::parse(
      R"({"schema":"gufo-abliteration-v1","placement":"","model_identity":
      "fixture","hidden":3,"layers":2,"direction":[0.6,0.8,0],
      "attention":[1,0],"mlp":[0,0.5]})");
  v1["placement"] = std::string(site);
  for (const char* key : {"rank", "preserve_norm", "direction_kind"}) {
    auto invalid = v1;
    invalid[key] = key == std::string_view("preserve_norm") ? json::Value(true)
                   : key == std::string_view("rank")        ? json::Value(1)
                                                     : json::Value("mean_diff");
    Require(rejects(invalid), "v1 profile with a subspace member loaded");
  }
}

// Post-layer residual strengths (D16): optional v2 member, request
// overrides, identity and device words.
void TestResidual(const std::filesystem::path& path, std::string_view site,
                  const json::Value& v2,
                  const qwen::AbliterationSettings& writer_defaults) {
  const auto write = [&](const json::Value& value) {
    std::ofstream(path) << value.dump();
  };
  // Absent: all zero, so a profile written before D16 behaves as before.
  write(v2);
  const auto plain =
      qwen::AbliterationProfile::Load(path, 3, 2, "fixture", site);
  Require(
      plain.residual == std::vector<float>{0, 0} &&
          !writer_defaults.ResidualActive() &&
          writer_defaults.residual == std::vector<float>{0, 0} &&
          writer_defaults.identity == ExpectedIdentity(plain, writer_defaults),
      "v2 profile without residual");
  auto with = v2;
  with["residual"] = json::parse("[0,1]");
  write(with);
  const auto profile =
      qwen::AbliterationProfile::Load(path, 3, 2, "fixture", site);
  Require(profile.residual == std::vector<float>{0, 1}, "v2 residual load");
  const auto defaults = qwen::ResolveAbliterationSettings(profile);
  Require(defaults->residual == profile.residual &&
              defaults->ResidualActive() &&
              defaults->identity == ExpectedIdentity(profile, *defaults),
          "v2 residual defaults");
  for (const char* raw : {"[1,-0.5]", "[1,4.5]", "[1]", "[1,1,1]", "1",
                          "[1,\"1\"]", "[1,null]"}) {
    auto invalid = v2;
    invalid["residual"] = json::parse(raw);
    write(invalid);
    Require(Rejects([&] {
              (void)qwen::AbliterationProfile::Load(path, 3, 2, "fixture",
                                                    site);
            }),
            "invalid v2 residual loaded");
  }
  // v1 never carries residual strengths.
  auto v1 = json::parse(
      R"({"schema":"gufo-abliteration-v1","placement":"","model_identity":
      "fixture","hidden":3,"layers":2,"direction":[0.6,0.8,0],
      "attention":[1,0],"mlp":[0,0.5]})");
  v1["placement"] = std::string(site);
  write(v1);
  const auto v1_profile =
      qwen::AbliterationProfile::Load(path, 3, 2, "fixture", site);
  Require(v1_profile.residual == std::vector<float>{0, 0}, "v1 residual zero");
  v1["residual"] = json::parse("[0,0]");
  write(v1);
  Require(Rejects([&] {
            (void)qwen::AbliterationProfile::Load(path, 3, 2, "fixture", site);
          }),
          "v1 profile with residual loaded");
  // Requests: a scalar, an array, -0 folding and bounds.
  const auto scalar =
      qwen::ResolveAbliterationSettings(profile, {.residual = {1}});
  const auto array =
      qwen::ResolveAbliterationSettings(profile, {.residual = {1, 1}});
  const auto off =
      qwen::ResolveAbliterationSettings(profile, {.residual = {0}});
  const auto negative_zero =
      qwen::ResolveAbliterationSettings(profile, {.residual = {-0.0F, 0}});
  Require(scalar->residual == std::vector<float>{1, 1} &&
              scalar->identity == array->identity &&
              scalar->identity != defaults->identity &&
              scalar->attention == profile.attention &&
              scalar->mlp == profile.mlp,
          "residual overrides");
  Require(!off->ResidualActive() && !negative_zero->ResidualActive() &&
              off->identity == negative_zero->identity &&
              Bits(negative_zero->residual[0]) == 0 &&
              off->identity == ExpectedIdentity(profile, *off),
          "residual zero folding");
  // Residual and writer strengths are separate settings: moving a strength
  // from one to another is another identity.
  const auto writer_one = qwen::ResolveAbliterationSettings(
      profile, {.attention = {1}, .mlp = {0}, .residual = {0}});
  const auto residual_one = qwen::ResolveAbliterationSettings(
      profile, {.attention = {0}, .mlp = {0}, .residual = {1}});
  Require(writer_one->identity != residual_one->identity,
          "residual and writer identities");
  for (const auto& overrides : std::vector<qwen::AbliterationOverrides>{
           {.residual = {-1}},
           {.residual = {4.5F}},
           {.residual = {1, 1, 1}},
           {.residual = {std::numeric_limits<float>::quiet_NaN()}}}) {
    Require(Rejects([&] {
              (void)qwen::ResolveAbliterationSettings(profile, overrides);
            }),
            "invalid residual settings resolved");
  }
  // Device words: 1 + 3 x layers; the residual block follows mlp.
  const auto words = residual_one->DeviceWords();
  using enum qwen::AbliterationPoint;
  Require(qwen::AbliterationDeviceWords(2) == 7 && words.size() == 7 &&
              qwen::AbliterationWeightIndex(2, 0, kAttention) == 1 &&
              qwen::AbliterationWeightIndex(2, 1, kMlp) == 4 &&
              qwen::AbliterationWeightIndex(2, 0, kResidual) == 5 &&
              qwen::AbliterationWeightIndex(2, 1, kResidual) == 6 &&
              qwen::AbliterationWeightIndex(2, 1, true) ==
                  qwen::AbliterationWeightIndex(2, 1, kMlp) &&
              qwen::AbliterationWeightIndex(2, 1, false) ==
                  qwen::AbliterationWeightIndex(2, 1, kAttention) &&
              words[5] == 1 && words[6] == 1 && words[1] == 0 &&
              words[2] == 0 && words[3] == 0 && words[4] == 0,
          "residual device words");
  const auto mixed = qwen::ResolveAbliterationSettings(
      profile, {.attention = {2, 3}, .mlp = {0.5F, 0.25F}, .residual = {0, 4}});
  const auto mixed_words = mixed->DeviceWords();
  Require(mixed_words[1] == 2 && mixed_words[2] == 3 &&
              mixed_words[3] == 0.5F && mixed_words[4] == 0.25F &&
              mixed_words[5] == 0 && mixed_words[6] == 4 &&
              Bits(mixed_words[0]) == 1,
          "mixed device words");
  // ON metadata is bound to the residual strengths too.
  auto metadata = Current(4);
  metadata.RecordImage({}, residual_one.get());
  Require(metadata.ImageMatches({}, residual_one.get()) &&
              !metadata.ImageMatches({}, writer_one.get()) &&
              !metadata.ImageMatches({}, nullptr),
          "residual-bound continuation guard");
}

// Schema v2: every layer's direction plus request defaults, validated like
// v1; per-request settings and their identities.
void TestV2Profile(const std::filesystem::path& path, std::string_view site) {
  const auto v2 = json::parse(
      R"({"schema":"gufo-abliteration-v2","placement":"writer-before-residual",
      "model_identity":"fixture","hidden":3,"layers":2,
      "directions":[[0.6,0.8,0],[0,0,1]],"direction_layer":1,
      "attention":[1,0],"mlp":[0,0.5]})");
  const auto write = [&](const json::Value& value) {
    std::ofstream(path) << value.dump();
  };
  write(v2);
  const auto profile =
      qwen::AbliterationProfile::Load(path, 3, 2, "fixture", site);
  Require(profile.schema == 2 && profile.direction_rows() == 2 &&
              profile.directions.size() == 6 && profile.directions[5] == 1 &&
              profile.direction_layer == 1u && profile.selectable_direction(),
          "v2 profile load");
  const auto rejects = [&](const char* key, const char* raw) {
    auto invalid = v2;
    invalid[key] = json::parse(raw);
    write(invalid);
    return Rejects([&] {
      (void)qwen::AbliterationProfile::Load(path, 3, 2, "fixture", site);
    });
  };
  Require(rejects("directions", "[[0.6,0.8,0]]") &&
              rejects("directions", "[[0.6,0.8,0],[1,1,0]]") &&
              rejects("directions", "[[0.6,0.8,0],[0,1]]") &&
              rejects("directions", "[[0.6,0.8,0],[0,0,\"1\"]]") &&
              rejects("direction_layer", "2") &&
              rejects("direction_layer", "-1") &&
              rejects("direction_layer", "0.5") &&
              rejects("direction_layer", "\"1\"") &&
              rejects("attention", "[1,-0.5]") &&
              rejects("attention", "[1,4.5]") && rejects("mlp", "[1]") &&
              rejects("schema", "\"gufo-abliteration-v3\""),
          "invalid v2 profile loaded");
  // v2 without `directions` (a v1 body) and v1 with them are rejected.
  auto mixed = v2;
  mixed["schema"] = "gufo-abliteration-v1";
  write(mixed);
  Require(Rejects([&] {
            (void)qwen::AbliterationProfile::Load(path, 3, 2, "fixture", site);
          }),
          "v1 schema with v2 members loaded");

  const auto defaults = qwen::ResolveAbliterationSettings(profile);
  Require(defaults->direction_layer == 1u && defaults->direction_row == 1 &&
              defaults->attention == profile.attention &&
              defaults->mlp == profile.mlp,
          "v2 default settings");
  // Identity covers only the effective values: explicit defaults match.
  const auto explicit_defaults = qwen::ResolveAbliterationSettings(
      profile, {.direction_layer = 1, .attention = {1, 0}, .mlp = {0, 0.5F}});
  Require(explicit_defaults->identity == defaults->identity,
          "explicit defaults changed the identity");
  const auto layer0 =
      qwen::ResolveAbliterationSettings(profile, {.direction_layer = 0});
  const auto scalar = qwen::ResolveAbliterationSettings(profile, {.mlp = {2}});
  const auto array =
      qwen::ResolveAbliterationSettings(profile, {.mlp = {2, 2}});
  const auto nudged =
      qwen::ResolveAbliterationSettings(profile, {.mlp = {0, 0.50001F}});
  const auto negative_zero =
      qwen::ResolveAbliterationSettings(profile, {.attention = {1, -0.0F}});
  Require(layer0->direction_row == 0 && scalar->mlp == std::vector<float>{2, 2},
          "v2 overrides");
  Require(layer0->identity != defaults->identity &&
              scalar->identity != defaults->identity &&
              scalar->identity == array->identity &&
              nudged->identity != defaults->identity &&
              negative_zero->identity == defaults->identity,
          "settings identities");
  // Another profile file with the same values is other settings.
  auto renamed = v2;
  renamed["note"] = "other file bytes";
  write(renamed);
  const auto other =
      qwen::AbliterationProfile::Load(path, 3, 2, "fixture", site);
  Require(
      qwen::ResolveAbliterationSettings(other)->identity != defaults->identity,
      "settings identity ignores the profile identity");
  for (const auto& overrides : std::vector<qwen::AbliterationOverrides>{
           {.direction_layer = 2},
           {.attention = {1, 1, 1}},
           {.mlp = {-1}},
           {.mlp = {4.5F}},
           {.mlp = {std::numeric_limits<float>::quiet_NaN()}},
           {.mlp = {std::numeric_limits<float>::infinity(), 0}}}) {
    Require(Rejects([&] {
              (void)qwen::ResolveAbliterationSettings(profile, overrides);
            }),
            "invalid settings resolved");
  }
  // Device words: direction row bits, then attention, then mlp per layer.
  const auto words = layer0->DeviceWords();
  Require(words.size() == qwen::AbliterationDeviceWords(2) &&
              Bits(words[0]) == 0 && Bits(defaults->DeviceWords()[0]) == 1 &&
              words[qwen::AbliterationWeightIndex(2, 0, false)] == 1 &&
              words[qwen::AbliterationWeightIndex(2, 1, false)] == 0 &&
              words[qwen::AbliterationWeightIndex(2, 1, true)] == 0.5F &&
              qwen::AbliterationWeightIndex(2, 0, true) == 3,
          "device settings layout");
  // ON metadata is bound to the settings: other strengths are a miss.
  auto metadata = Current(4);
  metadata.RecordImage({}, defaults.get());
  Require(metadata.ImageMatches({}, defaults.get()) &&
              metadata.ImageMatches({}, explicit_defaults.get()) &&
              !metadata.ImageMatches({}, nudged.get()) &&
              !metadata.ImageMatches({}, layer0.get()) &&
              !metadata.ImageMatches({}, nullptr),
          "settings-bound continuation guard");
  TestResidual(path, site, v2, *defaults);
  TestSubspace(path, site, v2);
  // A 27B-sized v2 profile written by prepare.py fits the size cap.
  Require(qwen::kAbliterationMaxProfileBytes >= 64ull * 5120 * 14,
          "size cap below a 27B v2 profile");
}
}  // namespace

int main() {
  const auto path =
      std::filesystem::temp_directory_path() /
      ("gufo-qwen27b-abliteration-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()) +
       ".json");
  struct Cleanup {
    std::filesystem::path p;
    ~Cleanup() { std::filesystem::remove(p); }
  } cleanup{path};
  const auto original = json::parse(
      R"({"schema":"gufo-abliteration-v1","placement":"writer-before-residual",
      "model_identity":"fixture","hidden":3,"layers":2,"direction":[0.6,0.8,0],
      "attention":[1,0],"mlp":[0,0.5],"status":"candidate-not-qualified"})");
  const auto write = [&](const json::Value& value) {
    std::ofstream(path) << value.dump();
  };
  write(original);
  constexpr auto kResidual = qwen::kAbliterationWriterBeforeResidual;
  const auto profile =
      qwen::AbliterationProfile::Load(path, 3, 2, "fixture", kResidual);
  Require(profile.directions.size() == 3 && profile.schema == 1 &&
              !profile.selectable_direction() &&
              profile.identity.size() == 64 && profile.placement == kResidual &&
              profile.mlp[1] == 0.5F,
          "27B profile load");
  // A Flash-Next placement never loads into 27B and vice versa, before the
  // model identity check.
  Require(Rejects([&] {
            (void)qwen::AbliterationProfile::Load(
                path, 3, 2, "fixture", qwen::kAbliterationWriterBeforeHc);
          }),
          "27B profile loaded with the Flash-Next placement");
  auto flash_next = original;
  flash_next["placement"] = "writer-before-hc";
  write(flash_next);
  Require(Rejects([&] {
            (void)qwen::AbliterationProfile::Load(path, 3, 2, "fixture",
                                                  kResidual);
          }),
          "Flash-Next profile loaded into 27B");
  write(original);
  Require(Rejects([&] {
            (void)qwen::AbliterationProfile::Load(path, 3, 2, "other",
                                                  kResidual);
          }),
          "profile of another artifact loaded");
  Require(Rejects([&] {
            (void)qwen::AbliterationProfile::Load(path, 4, 2, "fixture",
                                                  kResidual);
          }) &&
              Rejects([&] {
                (void)qwen::AbliterationProfile::Load(path, 3, 3, "fixture",
                                                      kResidual);
              }),
          "profile of other geometry loaded");
  Require(Rejects([&] {
            (void)qwen::AbliterationProfile::Load(path, 3, 2, "fixture", "");
          }),
          "profile loaded without a placement");
  // Only the explicit option loads a profile.
  Require(
      qwen::LoadAbliterationOption("", 3, 2, "fixture", kResidual) == nullptr,
      "empty option loaded a profile");
  const auto loaded =
      qwen::LoadAbliterationOption(path.string(), 3, 2, "fixture", kResidual);
  Require(loaded && loaded->identity == profile.identity, "option load");
  TestV1Settings(profile);
  TestV2Profile(path, kResidual);
  // The ON settings the metadata guard binds: the v1 profile defaults.
  const auto settings = qwen::ResolveAbliterationSettings(profile);

  // Metadata layout: OFF bytes equal pre-abliteration captures exactly.
  std::array<std::uint8_t, 32> raw{};
  for (std::size_t i = 0; i < raw.size(); ++i)
    raw[i] = static_cast<std::uint8_t>(i + 1);
  const std::span<const std::uint8_t> none{};
  auto off_text = Current(5);
  off_text.RecordImage(none, nullptr);
  Require(Bytes(off_text) == Legacy(5, none), "OFF text metadata bytes");
  auto off_image = Current(9);
  off_image.RecordImage(raw, nullptr);
  Require(Bytes(off_image) == Legacy(9, raw), "OFF image metadata bytes");
  Require(off_text.ImageMatches(none, nullptr) &&
              off_image.ImageMatches(raw, nullptr),
          "OFF round trip");

  auto on_text = Current(5);
  on_text.RecordImage(none, settings.get());
  auto on_image = Current(9);
  on_image.RecordImage(raw, settings.get());
  Require(on_text.abliteration == 1 && on_image.abliteration == 1 &&
              on_text.image_bytes == 0 && on_image.image_bytes == 32,
          "ON mode recorded");
  Require(std::ranges::equal(on_image.image, qwen::AbliterationInputIdentity(
                                                 raw, settings->identity)) &&
              on_image.image != raw && on_text.image != on_image.image,
          "ON image identity salted");
  Require(on_text.ImageMatches(none, settings.get()) &&
              on_image.ImageMatches(raw, settings.get()),
          "ON round trip");
  auto other = *settings;
  other.identity = std::string(64, '0');
  // Mode mismatch either way, another profile or another image are misses.
  Require(!off_text.ImageMatches(none, settings.get()) &&
              !off_image.ImageMatches(raw, settings.get()) &&
              !on_text.ImageMatches(none, nullptr) &&
              !on_image.ImageMatches(raw, nullptr) &&
              !on_text.ImageMatches(none, &other) &&
              !on_image.ImageMatches(raw, &other) &&
              !on_image.ImageMatches(none, settings.get()),
          "mismatched mode, profile or image restored");
  auto changed = raw;
  changed[0] ^= 1;
  Require(!off_image.ImageMatches(changed, nullptr) &&
              !on_image.ImageMatches(changed, settings.get()),
          "changed image restored");
  // A pre-abliteration capture (mode bytes zero) restores OFF only.
  qwen::ContinuationMetadata legacy;
  const auto legacy_bytes = Legacy(9, raw);
  std::memcpy(&legacy, legacy_bytes.data(), sizeof(legacy));
  Require(legacy.ImageMatches(raw, nullptr) &&
              !legacy.ImageMatches(raw, settings.get()),
          "legacy capture guard");
  Require(Rejects([&] {
            const std::array<std::uint8_t, 3> short_image{1, 2, 3};
            auto metadata = Current(1);
            metadata.RecordImage(short_image, nullptr);
          }),
          "invalid image identity recorded");
  std::cout << "qwen27b abliteration profile and metadata PASS\n";
}
