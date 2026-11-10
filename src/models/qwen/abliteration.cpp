#include "src/models/qwen/abliteration.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"

namespace gufo::models::qwen {
namespace {
std::vector<float> Vector(const json::Value& root, const char* name,
                          std::size_t size, double bound) {
  const auto* value = root.find(name);
  if (!value || !value->is_array() || value->size() != size)
    throw std::invalid_argument(std::string("abliteration: invalid ") + name);
  std::vector<float> result;
  for (const auto& item : value->items()) {
    const double x = item.as_double();
    if (!item.is_number() || !std::isfinite(x) || std::abs(x) > bound)
      throw std::invalid_argument(std::string("abliteration: invalid ") + name);
    result.push_back(static_cast<float>(x));
  }
  return result;
}
}  // namespace

AbliterationProfile AbliterationProfile::Load(const std::filesystem::path& path,
                                              std::uint32_t hidden,
                                              std::uint32_t layers,
                                              std::string_view expected,
                                              std::string_view placement) {
  constexpr std::size_t kMaxBytes = 1024 * 1024;
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::invalid_argument("abliteration: cannot open profile");
  std::string bytes(kMaxBytes + 1, '\0');
  file.read(bytes.data(), bytes.size());
  bytes.resize(static_cast<std::size_t>(file.gcount()));
  if (bytes.size() > kMaxBytes)
    throw std::invalid_argument("abliteration: profile exceeds 1 MiB");
  const auto root = json::parse(bytes);
  if (root.member_str("schema") != kAbliterationSchema ||
      root.member_str("model_identity") != expected || expected.empty() ||
      placement.empty() || root.member_str("placement") != placement ||
      !root.find("hidden") || !root.find("hidden")->is_number() ||
      root.find("hidden")->as_double() != hidden || !root.find("layers") ||
      !root.find("layers")->is_number() ||
      root.find("layers")->as_double() != layers)
    throw std::invalid_argument("abliteration: incompatible profile/model");
  AbliterationProfile result;
  result.model_identity = std::string(expected);
  result.placement = std::string(placement);
  result.direction = Vector(root, "direction", hidden, 1.0);
  result.attention = Vector(root, "attention", layers, 4.0);
  result.mlp = Vector(root, "mlp", layers, 4.0);
  double norm = 0;
  for (float x : result.direction)
    norm += double{x} * x;
  if (std::abs(norm - 1.0) > 1e-4)
    throw std::invalid_argument("abliteration: direction must have unit norm");
  result.identity = crypto::Sha256Hex(std::span(
      reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
  return result;
}

std::vector<std::uint8_t> AbliterationInputIdentity(
    std::span<const std::uint8_t> input, std::string_view profile_identity) {
  const std::string prefix =
      "gufo-abliteration-input-v1:" + std::string(profile_identity) + ":";
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
    std::span<const std::uint8_t> image, const AbliterationProfile* active) {
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
                                     const AbliterationProfile* active) {
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
