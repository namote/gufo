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
// layout: SHA-256 of "gufo-abliteration-settings-v2:<profile identity>:", the
// direction layer (0xFFFFFFFF for v1) and the layer count as little-endian
// u32, then the float bits of attention, mlp and residual.
std::string ExpectedIdentity(const qwen::AbliterationProfile& profile,
                             const qwen::AbliterationSettings& settings) {
  gufo::crypto::Sha256Hasher hash;
  const std::string prefix =
      "gufo-abliteration-settings-v2:" + profile.identity + ":";
  hash.Update(std::span(reinterpret_cast<const std::uint8_t*>(prefix.data()),
                        prefix.size()));
  HashU32(hash, settings.direction_layer.value_or(0xFFFFFFFFU));
  HashU32(hash, profile.layers);
  for (const auto* values :
       {&settings.attention, &settings.mlp, &settings.residual})
    for (float x : *values)
      HashU32(hash, Bits(x));
  return hash.FinishHex();
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
