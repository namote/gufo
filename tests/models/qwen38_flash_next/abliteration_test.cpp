#include "src/models/qwen38_flash_next/abliteration.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
  namespace qwen = gufo::models::qwen;
  Require(qwen::ListAbliterationProfiles("").empty());
  Require(q::LoadAbliterationProfiles({}, 3, 2, "fixture").empty());
  const std::vector<std::string> files{path.string()};
  const auto loaded = q::LoadAbliterationProfiles(files, 3, 2, "fixture");
  Require(loaded.size() == 1 && loaded[0]->identity == profile.identity &&
          loaded[0]->name == path.stem().string() && !loaded[0]->trigger);
  // `--abliteration-profiles DIR`: its *.json files in name order, nothing
  // else; two files of one name cannot load together.
  const auto listed =
      qwen::ListAbliterationProfiles((model_dir / "abliteration").string());
  Require(listed.size() == 1 && listed[0].ends_with("default.json"));
  std::ofstream(model_dir / "abliteration/notes.txt") << "x";
  std::filesystem::copy_file(path, model_dir / "abliteration/a.json");
  const auto two =
      qwen::ListAbliterationProfiles((model_dir / "abliteration").string());
  Require(two.size() == 2 && two[0].ends_with("a.json") &&
          two[1].ends_with("default.json"));
  const auto pair = q::LoadAbliterationProfiles(two, 3, 2, "fixture");
  Require(pair.size() == 2 && pair[0]->name == "a" &&
          pair[1]->name == "default");
  Require(&qwen::SelectAbliterationProfile(pair, "default") == pair[1].get());
  const auto throws = [](const auto& f) {
    try {
      f();
    } catch (const std::exception&) {
      return true;
    }
    return false;
  };
  Require(throws([&] { (void)qwen::SelectAbliterationProfile(pair, ""); }));
  Require(throws([&] { (void)qwen::SelectAbliterationProfile(pair, "b"); }));
  Require(&qwen::SelectAbliterationProfile(loaded, "") == loaded[0].get());
  const std::vector<std::string> twice{path.string(), path.string()};
  Require(throws(
      [&] { (void)q::LoadAbliterationProfiles(twice, 3, 2, "fixture"); }));
  Require(throws([&] {
    (void)qwen::ListAbliterationProfiles((model_dir / "none").string());
  }));
  Require(throws([&] { (void)qwen::ListAbliterationProfiles(path.string()); }));

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
  // Without `residual` the profile keeps writer-only behavior.
  Require(v2.residual == std::vector<float>{0, 0} &&
          !q::ResolveAbliterationSettings(v2)->ResidualActive());
  // Post-layer residual strengths (each HC stream) through this placement.
  std::ofstream(path)
      << R"({"schema":"gufo-abliteration-v2","placement":"writer-before-hc",
      "model_identity":"fixture","hidden":3,"layers":2,
      "directions":[[0,1,0],[0.6,0.8,0]],"direction_layer":1,
      "attention":[0,0],"mlp":[0,0],"residual":[1,0.5]})";
  const auto residual = q::AbliterationProfile::Load(path, 3, 2, "fixture");
  const auto residual_defaults = q::ResolveAbliterationSettings(residual);
  Require(residual.residual == std::vector<float>{1, 0.5F} &&
          residual_defaults->ResidualActive());
  const auto words = residual_defaults->DeviceWords();
  Require(words.size() == gufo::models::qwen::AbliterationDeviceWords(2) &&
          words.size() == 7 && words[5] == 1 && words[6] == 0.5F);
  const auto writer_only =
      q::ResolveAbliterationSettings(residual, {.residual = {0}});
  Require(!writer_only->ResidualActive() &&
          writer_only->identity != residual_defaults->identity);
  // Word 0 of a profile without subspace members is the bare direction row.
  Require(residual.rank == 1 && residual.kinds == 1 &&
          std::bit_cast<std::uint32_t>(words[0]) == 1);
  // Rank 2 with projected directions through this placement (base64
  // little-endian float32: [1,0,0], [0,0,1]; projected [0,0,1],[1,0,0] and
  // [1,0,0],[0,1,0]).
  std::ofstream(path)
      << R"({"schema":"gufo-abliteration-v2","placement":"writer-before-hc",
      "model_identity":"fixture","hidden":3,"layers":2,
      "directions":[[0,1,0],[0.6,0.8,0]],"direction_layer":1,
      "attention":[0,0],"mlp":[0,0],"residual":[1,1],"rank":2,
      "subspace_directions":["AACAPwAAAAAAAAAA","AAAAAAAAAAAAAIA/"],
      "projected_directions":["AAAAAAAAAAAAAIA/AACAPwAAAAAAAAAA",
      "AACAPwAAAAAAAAAAAAAAAAAAgD8AAAAA"],"subspace_method":"pca",
      "direction_kind":"projected","direction_count":2,"preserve_norm":true})";
  const auto subspace = q::AbliterationProfile::Load(path, 3, 2, "fixture");
  const auto subspace_defaults = q::ResolveAbliterationSettings(subspace);
  Require(subspace.rank == 2 && subspace.kinds == 2 &&
          subspace.direction_rows() == 8 &&
          subspace.directions[3 * 3 + 2] == 1 &&
          subspace.directions[7 * 3 + 1] == 1 &&
          subspace_defaults->direction_row == 6 &&
          std::bit_cast<std::uint32_t>(subspace_defaults->DeviceWords()[0]) ==
              (6U | 1U << qwen::kAbliterationCountShift |
               qwen::kAbliterationPreserveNormBit));
  const auto single = q::ResolveAbliterationSettings(
      subspace, {.direction_kind = qwen::AbliterationDirectionKind::kMeanDiff,
                 .direction_count = 1,
                 .preserve_norm = false});
  Require(single->direction_row == 2 &&
          std::bit_cast<std::uint32_t>(single->DeviceWords()[0]) == 2 &&
          single->identity != subspace_defaults->identity);
  // A profile's rows start at its own offset in a shared direction table;
  // the identity does not depend on where it is loaded.
  Require(std::bit_cast<std::uint32_t>(single->DeviceWords(5)[0]) == 7 &&
          single->profile == path.stem().string());
  Require(
      throws([&] { (void)single->DeviceWords(qwen::kAbliterationRowMask); }));

  // A dynamic profile's detector (`trigger`): layer, unit direction (base64
  // float32 [0,0,1]), threshold and the placement's collector readout.
  const auto with_trigger = [&](const std::string& trigger) {
    std::ofstream(path)
        << R"({"schema":"gufo-abliteration-v2","placement":"writer-before-hc",
        "model_identity":"fixture","hidden":3,"layers":2,
        "directions":[[0,1,0],[0.6,0.8,0]],"direction_layer":1,
        "attention":[0,0],"mlp":[0,0],"residual":[1,1],"trigger":)"
        << trigger << "}";
    return q::AbliterationProfile::Load(path, 3, 2, "fixture");
  };
  const std::string trigger =
      R"({"schema":"gufo-abliteration-trigger-v1","layer":1,
      "readout":"last-prompt-token-post-layer-hc-mean-fp32-v1",
      "direction":"AAAAAAAAAAAAAIA/","threshold":1.25,"fit":{"auc":0.99}})";
  const auto dynamic = with_trigger(trigger);
  Require(dynamic.trigger && dynamic.trigger->layer == 1 &&
          dynamic.trigger->threshold == 1.25F &&
          dynamic.trigger->direction == std::vector<float>{0, 0, 1});
  Require(!dynamic.trigger->decode_threshold);
  {
    auto text = trigger;
    text.insert(text.find(R"("threshold")"), R"("decode_threshold":2.5,)");
    Require(with_trigger(text).trigger->decode_threshold == 2.5F);
    text.replace(text.find("2.5"), 3, "\"y\"");
    Require(throws([&] { (void)with_trigger(text); }));
  }
  for (const auto& [from, to] :
       std::initializer_list<std::pair<std::string_view, std::string_view>>{
           {R"("layer":1)", R"("layer":2)"},
           {R"("layer":1)", R"("layer":-1)"},
           {"hc-mean-fp32", "fp32"},
           {"trigger-v1", "trigger-v2"},
           {"AAAAAAAAAAAAAIA/", "AAAAAAAAAAAAAAA/"},
           {"AAAAAAAAAAAAAIA/", "AAAAAAAAAAAAAIA"},
           {"1.25", "\"x\""},
       }) {
    auto text = trigger;
    text.replace(text.find(from), from.size(), to);
    Require(throws([&] { (void)with_trigger(text); }));
  }
  Require(throws([&] { (void)with_trigger("1"); }));
  auto v1_trigger = original;
  v1_trigger["trigger"] = json::parse(trigger);
  write(v1_trigger);
  Require(throws(
      [&] { (void)q::AbliterationProfile::Load(path, 3, 2, "fixture"); }));
  // Companions (`trigger.with`): distinct names.
  {
    auto text = trigger;
    text.insert(text.find(R"("threshold")"), R"("with":["fnx"],)");
    Require(with_trigger(text).trigger->with ==
            std::vector<std::string>{"fnx"});
    for (const auto* bad : {R"("with":"fnx",)", R"("with":["fnx","fnx"],)",
                            R"("with":[""],)", R"("with":[1],)"}) {
      auto invalid = trigger;
      invalid.insert(invalid.find(R"("threshold")"), bad);
      Require(throws([&] { (void)with_trigger(invalid); }));
    }
  }

  // Several profiles ON together: load order, one identity per set.
  {
    auto a = residual;
    a.name = "a";
    a.identity = std::string(64, 'a');
    auto b = residual;
    b.name = "b";
    b.identity = std::string(64, 'b');
    const std::array<const qwen::AbliterationProfile*, 2> loaded{&a, &b};
    const std::vector<qwen::AbliterationOverrides> ab{{.profile = "a"},
                                                      {.profile = "b"}};
    const std::vector<qwen::AbliterationOverrides> ba{{.profile = "b"},
                                                      {.profile = "a"}};
    const auto set = qwen::ResolveAbliterationSet(loaded, ab);
    const auto reversed = qwen::ResolveAbliterationSet(loaded, ba);
    Require(set->identity == reversed->identity && set->parts.size() == 2 &&
            set->parts[0]->profile == "a" && set->profile == "a+b" &&
            qwen::AbliterationParts(*set).size() == 2 && set->ResidualActive());
    const std::vector<qwen::AbliterationOverrides> one{{.profile = "b"}};
    const auto single_b = qwen::ResolveAbliterationSet(loaded, one);
    Require(single_b->parts.empty() &&
            single_b->identity ==
                qwen::ResolveAbliterationSettings(b)->identity &&
            qwen::AbliterationParts(*single_b).front() == single_b.get());
    const std::vector<qwen::AbliterationOverrides> stronger{
        {.profile = "a"}, {.profile = "b", .residual = {0.5F}}};
    Require(qwen::ResolveAbliterationSet(loaded, stronger)->identity !=
            set->identity);
    for (const auto& bad :
         std::vector<std::vector<qwen::AbliterationOverrides>>{
             {{.profile = "a"}, {.profile = "a"}},
             {{.profile = "c"}},
             {{.profile = ""}},
             {}}) {
      Require(throws([&] { (void)qwen::ResolveAbliterationSet(loaded, bad); }));
    }
    // Orthonormal union: a shared direction counts once, and the span
    // does not depend on the order.
    const std::vector<float> rows{1, 0, 0, 0.6F, 0.8F, 0, 0, 1, 0};
    const auto basis = qwen::OrthonormalizeDirections(rows, 3);
    Require(basis.size() == 6 && std::abs(basis[0] - 1) < 1e-6F &&
            std::abs(basis[4] - 1) < 1e-6F && std::abs(basis[2]) < 1e-6F);
    const std::vector<float> reordered{0, 1, 0, 1, 0, 0, 0.6F, 0.8F, 0};
    const auto other = qwen::OrthonormalizeDirections(reordered, 3);
    // Same projector: both bases span the xy plane.
    Require(other.size() == 6 && std::abs(other[2]) < 1e-6F &&
            std::abs(other[5]) < 1e-6F);
    // A dynamic switch keeps the parts and salts the combined identity.
    const auto dynamic_set = qwen::DynamicAbliterationSettings(*set, 4);
    Require(dynamic_set->parts.size() == 2 &&
            dynamic_set->identity != set->identity);
  }
  std::cout << "profile and cache identity PASS\n";
}
