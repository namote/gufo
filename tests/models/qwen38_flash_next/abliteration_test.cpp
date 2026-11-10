#include "src/models/qwen38_flash_next/abliteration.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "src/core/json.hpp"

namespace q = gufo::models::qwen38_flash_next;
namespace json = gufo::json;
void Require(bool b) {
  if (!b)
    throw std::runtime_error("abliteration contract failed");
}
int main() {
  const auto path =
      std::filesystem::temp_directory_path() /
      ("gufo-abliteration-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()) +
       ".json");
  struct Cleanup {
    std::filesystem::path p;
    ~Cleanup() { std::filesystem::remove(p); }
  } cleanup{path};
  const auto original = json::parse(
      R"({"schema":"gufo-abliteration-v1","placement":"writer-before-hc",
      "model_identity":"fixture","hidden":3,"layers":2,"direction":[0.6,0.8,0],"attention":[1,0],"mlp":[0,0.5]})");
  const auto write = [&](const json::Value& value) {
    std::ofstream(path) << value.dump();
  };
  write(original);
  const auto profile = q::AbliterationProfile::Load(path, 3, 2, "fixture");
  Require(profile.directions.size() == 3 && profile.identity.size() == 64 &&
          profile.placement == "writer-before-hc");
  const auto settings = q::ResolveAbliterationSettings(profile);
  Require(q::AbliterationProfile::Load(path, 3, 2, "fixture").identity ==
          profile.identity);
  const std::vector<std::uint8_t> image{1, 2, 3};
  Require(q::AbliterationInputIdentity({}, profile.identity) !=
          q::AbliterationInputIdentity(image, profile.identity));
  Require(q::AbliterationInputIdentity(image, "other") !=
          q::AbliterationInputIdentity(image, profile.identity));
  Require(q::AbliterationInputIdentity({}, profile.identity).size() == 32);

  // Model::Load resolves profiles only through this option, which takes no
  // model path: an empty option loads none even with the default.json that
  // earlier builds discovered beside the model.
  const auto model_dir = path.parent_path() / (path.stem().string() + "-model");
  struct DirCleanup {
    std::filesystem::path p;
    ~DirCleanup() { std::filesystem::remove_all(p); }
  } dir_cleanup{model_dir};
  std::filesystem::create_directories(model_dir / "abliteration");
  std::filesystem::copy_file(path, model_dir / "abliteration/default.json");
  Require(q::LoadAbliterationOption("", 3, 2, "fixture") == nullptr);
  const auto loaded = q::LoadAbliterationOption(path.string(), 3, 2, "fixture");
  Require(loaded && loaded->identity == profile.identity);

  // Component-cache record: OFF keeps the pre-abliteration bytes (mode 0,
  // raw or zero image identity); ON is bound to the full profile identity.
  using Record = q::AbliterationContinuationRecord;
  std::array<std::uint8_t, 32> raw{};
  for (std::size_t i = 0; i < raw.size(); ++i)
    raw[i] = static_cast<std::uint8_t>(i + 1);
  const std::span<const std::uint8_t> none{};
  const auto off_text = q::RecordAbliterationContinuation(none, nullptr);
  Require(off_text.mode == 0 &&
          std::ranges::all_of(off_text.image, [](auto b) { return b == 0; }));
  const auto off_image = q::RecordAbliterationContinuation(raw, nullptr);
  Require(off_image.mode == 0 && off_image.image == raw);
  auto other = *settings;
  other.identity = std::string(64, '0');
  const auto on_text = q::RecordAbliterationContinuation(none, settings.get());
  const auto on_image = q::RecordAbliterationContinuation(raw, settings.get());
  Require(on_text.mode == Record::kOn && on_image.mode == Record::kOn);
  Require(std::ranges::equal(
      on_image.image, q::AbliterationInputIdentity(raw, settings->identity)));
  Require(on_text.image != on_image.image && on_image.image != raw);
  const auto matches = [](const Record& r, std::uint32_t image_bytes,
                          std::span<const std::uint8_t> image,
                          const q::AbliterationSettings* active) {
    return q::AbliterationContinuationMatches(r.mode, image_bytes, r.image,
                                              image, active);
  };
  // Round trips.
  Require(matches(off_text, 0, none, nullptr));
  Require(matches(off_image, 32, raw, nullptr));
  Require(matches(on_text, 0, none, settings.get()));
  Require(matches(on_image, 32, raw, settings.get()));
  // Mode mismatch either way, other profile and other image are misses.
  Require(!matches(off_text, 0, none, settings.get()));
  Require(!matches(off_image, 32, raw, settings.get()));
  Require(!matches(on_text, 0, none, nullptr));
  Require(!matches(on_image, 32, raw, nullptr));
  Require(!matches(on_text, 0, none, &other));
  Require(!matches(on_image, 32, raw, &other));
  Require(!matches(on_text, 0, raw, settings.get()));
  Require(!matches(on_image, 32, none, settings.get()));
  auto changed = raw;
  changed[0] ^= 1;
  Require(!matches(off_image, 32, changed, nullptr));
  Require(!matches(on_image, 32, changed, settings.get()));
  bool invalid_image = false;
  try {
    const std::array<std::uint8_t, 3> short_image{1, 2, 3};
    (void)q::RecordAbliterationContinuation(short_image, nullptr);
  } catch (const std::logic_error&) {
    invalid_image = true;
  }
  Require(invalid_image);
  for (const auto* key : {"direction", "attention", "mlp", "hidden", "layers",
                          "schema", "placement", "model_identity"}) {
    auto invalid = original;
    invalid[key] = "wrong";
    write(invalid);
    bool rejected = false;
    try {
      (void)q::AbliterationProfile::Load(path, 3, 2, "fixture");
    } catch (const std::exception&) {
      rejected = true;
    }
    Require(rejected);
  }
  for (const auto* raw : {"[0,0,0]", "[1,1,0]", "[true,0,0]", "[1e300,0,0]"}) {
    auto invalid = original;
    invalid["direction"] = json::parse(raw);
    write(invalid);
    bool rejected = false;
    try {
      (void)q::AbliterationProfile::Load(path, 3, 2, "fixture");
    } catch (const std::exception&) {
      rejected = true;
    }
    Require(rejected);
  }
  std::ofstream(path) << std::string(
      gufo::models::qwen::kAbliterationMaxProfileBytes + 1, ' ');
  bool rejected = false;
  try {
    (void)q::AbliterationProfile::Load(path, 3, 2, "fixture");
  } catch (const std::exception&) {
    rejected = true;
  }
  Require(rejected);

  // Schema v2 through the Flash-Next placement: every layer's direction.
  std::ofstream(path)
      << R"({"schema":"gufo-abliteration-v2","placement":"writer-before-hc",
      "model_identity":"fixture","hidden":3,"layers":2,
      "directions":[[0,1,0],[0.6,0.8,0]],"direction_layer":0,
      "attention":[2,2],"mlp":[0.5,0.5]})";
  const auto v2 = q::AbliterationProfile::Load(path, 3, 2, "fixture");
  Require(v2.schema == 2 && v2.direction_rows() == 2 &&
          v2.direction_layer == 0u);
  const auto chosen = q::ResolveAbliterationSettings(
      v2, {.direction_layer = 1, .attention = {1}});
  Require(chosen->direction_row == 1 &&
          chosen->attention == std::vector<float>{1, 1} &&
          chosen->mlp == v2.mlp &&
          chosen->identity != q::ResolveAbliterationSettings(v2)->identity);
  std::cout << "profile and cache identity PASS\n";
}
