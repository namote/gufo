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
#include <stdexcept>
#include <string>

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
  Require(profile.direction.size() == 3 && profile.identity.size() == 64 &&
              profile.placement == kResidual && profile.mlp[1] == 0.5F,
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
  on_text.RecordImage(none, &profile);
  auto on_image = Current(9);
  on_image.RecordImage(raw, &profile);
  Require(on_text.abliteration == 1 && on_image.abliteration == 1 &&
              on_text.image_bytes == 0 && on_image.image_bytes == 32,
          "ON mode recorded");
  Require(std::ranges::equal(on_image.image, qwen::AbliterationInputIdentity(
                                                 raw, profile.identity)) &&
              on_image.image != raw && on_text.image != on_image.image,
          "ON image identity salted");
  Require(on_text.ImageMatches(none, &profile) &&
              on_image.ImageMatches(raw, &profile),
          "ON round trip");
  auto other = profile;
  other.identity = std::string(64, '0');
  // Mode mismatch either way, another profile or another image are misses.
  Require(!off_text.ImageMatches(none, &profile) &&
              !off_image.ImageMatches(raw, &profile) &&
              !on_text.ImageMatches(none, nullptr) &&
              !on_image.ImageMatches(raw, nullptr) &&
              !on_text.ImageMatches(none, &other) &&
              !on_image.ImageMatches(raw, &other) &&
              !on_image.ImageMatches(none, &profile),
          "mismatched mode, profile or image restored");
  auto changed = raw;
  changed[0] ^= 1;
  Require(!off_image.ImageMatches(changed, nullptr) &&
              !on_image.ImageMatches(changed, &profile),
          "changed image restored");
  // A pre-abliteration capture (mode bytes zero) restores OFF only.
  qwen::ContinuationMetadata legacy;
  const auto legacy_bytes = Legacy(9, raw);
  std::memcpy(&legacy, legacy_bytes.data(), sizeof(legacy));
  Require(
      legacy.ImageMatches(raw, nullptr) && !legacy.ImageMatches(raw, &profile),
      "legacy capture guard");
  Require(Rejects([&] {
            const std::array<std::uint8_t, 3> short_image{1, 2, 3};
            auto metadata = Current(1);
            metadata.RecordImage(short_image, nullptr);
          }),
          "invalid image identity recorded");
  std::cout << "qwen27b abliteration profile and metadata PASS\n";
}
