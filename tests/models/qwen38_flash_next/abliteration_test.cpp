#include "src/models/qwen38_flash_next/abliteration.hpp"

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
  Require(profile.direction.size() == 3 && profile.identity.size() == 64);
  Require(q::AbliterationProfile::Load(path, 3, 2, "fixture").identity ==
          profile.identity);
  const std::vector<std::uint8_t> image{1, 2, 3};
  Require(q::AbliterationInputIdentity({}, profile.identity) !=
          q::AbliterationInputIdentity(image, profile.identity));
  Require(q::AbliterationInputIdentity(image, "other") !=
          q::AbliterationInputIdentity(image, profile.identity));
  Require(q::AbliterationInputIdentity({}, profile.identity).size() == 32);
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
  std::ofstream(path) << std::string(1024 * 1024 + 1, ' ');
  bool rejected = false;
  try {
    (void)q::AbliterationProfile::Load(path, 3, 2, "fixture");
  } catch (const std::exception&) {
    rejected = true;
  }
  Require(rejected);
  std::cout << "profile and cache identity PASS\n";
}
