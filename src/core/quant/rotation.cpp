#include "src/core/quant/rotation.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <unordered_map>
#include <utility>

namespace gufo::quant::rotation {
namespace {

bool Fail(std::string* error_msg, std::string message) {
  if (error_msg != nullptr) {
    *error_msg = "Rotated GGUF refused: " + std::move(message);
  }
  return false;
}

bool EndsWith(std::string_view text, std::string_view suffix) noexcept {
  return text.size() > suffix.size() &&
         text.substr(text.size() - suffix.size()) == suffix;
}

bool Contains(const std::vector<std::string>& values,
              std::string_view value) noexcept {
  return std::find(values.begin(), values.end(), value) != values.end();
}

std::string MetadataText(const core::GgufMetadataValue& value) {
  if (const auto* text = std::get_if<std::string_view>(&value.value)) {
    return "\"" + std::string(*text) + "\"";
  }
  if (const auto* number = std::get_if<std::uint64_t>(&value.value)) {
    return std::to_string(*number);
  }
  if (const auto* number = std::get_if<std::int64_t>(&value.value)) {
    return std::to_string(*number);
  }
  return "a value of GGUF type " + std::to_string(static_cast<int>(value.type));
}

bool CheckMarkers(const core::GgufReader& reader, std::string* error_msg) {
  for (const auto key : {kVersionKey, kKindKey, kBlockSizeKey, kSideKey}) {
    const auto* value = reader.FindMetadata(key);
    if (value == nullptr) {
      return Fail(error_msg, "missing " + std::string(key));
    }
    bool ok = false;
    std::string expected;
    if (key == kVersionKey || key == kBlockSizeKey) {
      const std::uint32_t want = key == kVersionKey ? kVersion : kBlockSize;
      ok = reader.GetMetadataUint32(key) == want;
      expected = std::to_string(want);
    } else {
      const std::string_view want = key == kKindKey ? kKind : kSide;
      ok = reader.GetMetadataString(key) == want;
      expected = "\"" + std::string(want) + "\"";
    }
    if (!ok) {
      return Fail(error_msg, "unsupported " + std::string(key) + " = " +
                                 MetadataText(*value) + " (expected " +
                                 expected + ")");
    }
  }
  return true;
}

// Reads one `.rot` tensor into packed sign bits after checking its storage.
bool ReadSigns(const core::GgufTensorInfo& sign,
               const core::GgufTensorInfo& base,
               std::vector<std::uint32_t>& bits, std::string* error_msg) {
  const std::string name(sign.name);
  if (sign.type != core::GgmlType::kF16 || sign.dimensions.size() != 1) {
    return Fail(error_msg, name + " must be a one-dimensional F16 tensor");
  }
  if (base.dimensions.empty()) {
    return Fail(error_msg, std::string(base.name) + " has no dimensions");
  }
  const std::uint64_t k = base.dimensions.front();
  if (sign.dimensions.front() != k) {
    return Fail(error_msg, name + " has " +
                               std::to_string(sign.dimensions.front()) +
                               " values, but " + std::string(base.name) +
                               " has K = " + std::to_string(k));
  }
  if (k == 0 || k % kBlockSize != 0) {
    return Fail(error_msg,
                std::string(base.name) + " has K = " + std::to_string(k) +
                    ", not a multiple of " + std::to_string(kBlockSize));
  }
  if (sign.data == nullptr || sign.size_bytes < k * sizeof(std::uint16_t)) {
    return Fail(error_msg, name + " payload is truncated");
  }
  bits.assign(PackedSignWords(static_cast<std::size_t>(k)), 0U);
  const auto* bytes = static_cast<const std::uint8_t*>(sign.data);
  for (std::size_t i = 0; i < k; ++i) {
    std::uint16_t value = 0;
    std::memcpy(&value, bytes + i * sizeof(value), sizeof(value));
    if (value == kF16MinusOne) {
      bits[i / 32] |= 1U << (i % 32);
    } else if (value != kF16PlusOne) {
      return Fail(error_msg, name + " holds values other than +1 and -1");
    }
  }
  return true;
}

}  // namespace

bool RotationInfo::IsGroupRotated(std::string_view group) const noexcept {
  return std::find(rotated_groups.begin(), rotated_groups.end(), group) !=
         rotated_groups.end();
}

const InstanceSigns* RotationInfo::Find(std::string_view key) const noexcept {
  for (const auto& instance : instances) {
    if (instance.key == key) {
      return &instance;
    }
  }
  return nullptr;
}

std::string RotationInfo::GroupList() const {
  std::string text;
  for (const auto& group : rotated_groups) {
    if (!text.empty()) {
      text += ", ";
    }
    text += group;
  }
  return text;
}

std::optional<RotationInfo> ReadRotation(const core::GgufReader& reader,
                                         const GroupTable& table,
                                         std::string* error_msg) {
  // `<base>` -> its sign tensor. Provenance keys (gufo.rotation.source_*) are
  // not markers: the converter writes them for unrotated outputs too.
  std::map<std::string, const core::GgufTensorInfo*, std::less<>> signs;
  for (const auto& tensor : reader.GetTensors()) {
    if (EndsWith(tensor.name, kSignSuffix)) {
      signs.emplace(std::string(tensor.name.substr(
                        0, tensor.name.size() - kSignSuffix.size())),
                    &tensor);
    }
  }
  bool has_markers = false;
  for (const auto key : {kVersionKey, kKindKey, kBlockSizeKey, kSideKey}) {
    has_markers = has_markers || reader.FindMetadata(key) != nullptr;
  }
  if (signs.empty() && !has_markers) {
    return RotationInfo{};
  }
  if (!has_markers) {
    Fail(error_msg, "file has .rot tensors (" + signs.begin()->first +
                        ".rot) but no gufo.rotation metadata");
    return std::nullopt;
  }
  if (signs.empty()) {
    Fail(error_msg, "file has gufo.rotation metadata but no .rot tensors");
    return std::nullopt;
  }
  if (!CheckMarkers(reader, error_msg)) {
    return std::nullopt;
  }

  std::unordered_map<std::string_view, const GroupInstance*> owner;
  for (const auto& instance : table.instances) {
    for (const auto& tensor : instance.tensors) {
      owner.emplace(tensor, &instance);
    }
  }
  for (const auto& [base, sign] : signs) {
    if (Contains(table.tied_tensors, base)) {
      Fail(error_msg, base +
                          " is rotated, but it is tied to the token "
                          "embedding; a rotated tied head is not supported");
      return std::nullopt;
    }
    if (reader.FindTensor(base) == nullptr) {
      Fail(error_msg, base + ".rot has no matching tensor " + base);
      return std::nullopt;
    }
    const auto found = owner.find(base);
    if (found == owner.end()) {
      Fail(error_msg, base + " is rotated but its role has no rotation path");
      return std::nullopt;
    }
    if (Contains(table.unsupported_groups, found->second->group)) {
      Fail(error_msg, base + " is rotated, but group " + found->second->group +
                          " has no rotation path in this runtime");
      return std::nullopt;
    }
  }

  RotationInfo info;
  // group -> (rotated instance count, first unrotated instance key, total)
  struct GroupCount {
    std::size_t rotated{0};
    std::size_t total{0};
    std::string first_unrotated;
  };
  std::vector<std::pair<std::string, GroupCount>> counts;
  const auto count_for = [&](const std::string& group) -> GroupCount& {
    for (auto& [name, count] : counts) {
      if (name == group) {
        return count;
      }
    }
    return counts.emplace_back(group, GroupCount{}).second;
  };

  for (const auto& instance : table.instances) {
    if (instance.tensors.empty()) {
      continue;
    }
    std::vector<std::string_view> rotated;
    std::vector<std::string_view> plain;
    for (const auto& tensor : instance.tensors) {
      (signs.contains(tensor) ? rotated : plain).push_back(tensor);
    }
    auto& count = count_for(instance.group);
    ++count.total;
    if (rotated.empty()) {
      if (count.first_unrotated.empty()) {
        count.first_unrotated = instance.key;
      }
      continue;
    }
    if (!plain.empty()) {
      Fail(error_msg, "group instance " + instance.key +
                          " mixes rotated and unrotated tensors (" +
                          std::string(rotated.front()) + " is rotated, " +
                          std::string(plain.front()) + " is not)");
      return std::nullopt;
    }
    InstanceSigns result{
        .group = instance.group, .key = instance.key, .k = 0, .bits = {}};
    for (const auto tensor : rotated) {
      const auto* base = reader.FindTensor(tensor);
      const auto* sign = signs.find(tensor)->second;
      std::vector<std::uint32_t> bits;
      if (!ReadSigns(*sign, *base, bits, error_msg)) {
        return std::nullopt;
      }
      if (result.bits.empty()) {
        result.k = static_cast<std::size_t>(base->dimensions.front());
        result.bits = std::move(bits);
      } else if (base->dimensions.front() != result.k || bits != result.bits) {
        Fail(error_msg, "group instance " + instance.key + ": " +
                            std::string(tensor) +
                            " uses a different sign vector than " +
                            std::string(rotated.front()));
        return std::nullopt;
      }
    }
    ++count.rotated;
    info.instances.push_back(std::move(result));
  }
  for (const auto& [group, count] : counts) {
    if (count.rotated == 0) {
      continue;
    }
    if (count.rotated != count.total) {
      Fail(error_msg, "group " + group + " is rotated in some layers only (" +
                          std::to_string(count.rotated) + " of " +
                          std::to_string(count.total) + "; " +
                          count.first_unrotated + " is not)");
      return std::nullopt;
    }
    info.rotated_groups.push_back(group);
  }
  return info;
}

}  // namespace gufo::quant::rotation
