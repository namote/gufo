#include "src/core/quant/rotation.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <map>
#include <regex>
#include <set>
#include <unordered_map>
#include <utility>
#include <variant>

#include "src/core/hip/detail/hadamard.hpp"

namespace gufo::quant::rotation {
namespace {

bool Fail(std::string* error_msg, std::string message) {
  if (error_msg != nullptr) {
    *error_msg = "Rotated GGUF refused: " + std::move(message);
  }
  return false;
}

// A refusal of the upstream loader, tagged with its source line.
bool Upstream(std::string* error_msg, int line, std::string message) {
  return Fail(error_msg, std::move(message) + " (" + std::string(kUpstream) +
                             ":" + std::to_string(line) + ")");
}

bool EndsWith(std::string_view text, std::string_view suffix) noexcept {
  return text.size() > suffix.size() &&
         text.substr(text.size() - suffix.size()) == suffix;
}

bool Contains(const std::vector<std::string>& values,
              std::string_view value) noexcept {
  return std::find(values.begin(), values.end(), value) != values.end();
}

// llama-model.cpp:2151-2163: architectures whose matmuls all pass through
// build_lora_mm.
bool IsSupportedArchitecture(std::string_view arch) noexcept {
  for (const std::string_view name :
       {"llama", "qwen3", "qwen3moe", "qwen35", "qwen35moe", "qwen3next"}) {
    if (arch == name) {
      return true;
    }
  }
  return false;
}

// llama-model.cpp:2166-2199: weights with a verified Hadamard-aware matmul.
bool IsFoldable(const std::string& name) {
  static const std::regex pattern(
      R"(output\.weight|blk\.\d+\.()"
      R"(attn_q|attn_k|attn_v|attn_qkv|attn_gate|attn_output)"
      R"(|ffn_gate|ffn_up|ffn_down)"
      R"(|ffn_gate_exps|ffn_up_exps|ffn_down_exps|ffn_gate_up_exps)"
      R"(|ffn_gate_shexp|ffn_up_shexp|ffn_down_shexp)"
      R"(|ssm_out)\.weight)");
  return std::regex_match(name, pattern);
}

// Typed metadata access with the GGUF types the upstream getters accept.
// Array element types are not kept by the reader: any integer array stands
// for upstream's INT32/UINT32 arrays.
struct Metadata {
  const core::GgufReader& reader;
  std::string* error_msg;
  bool failed{false};

  const core::GgufMetadataValue* Find(std::string_view key) const {
    return reader.FindMetadata(key);
  }

  bool Missing(std::string_view key) {
    failed = true;
    return Fail(error_msg, "key not found in model: " + std::string(key) +
                               " (llama-model-loader get_key/get_arr)");
  }

  bool WrongType(std::string_view key) {
    failed = true;
    return Fail(error_msg, std::string(key) +
                               " has the wrong GGUF type "
                               "(llama-model-loader get_key/get_arr)");
  }

  std::optional<std::uint32_t> U32(std::string_view key, bool required) {
    const auto* value = Find(key);
    if (value == nullptr) {
      if (required) {
        Missing(key);
      }
      return std::nullopt;
    }
    if (value->type != core::GgufValueType::kUint32) {
      WrongType(key);
      return std::nullopt;
    }
    return static_cast<std::uint32_t>(std::get<std::uint64_t>(value->value));
  }

  std::optional<bool> Bool(std::string_view key) {
    const auto* value = Find(key);
    if (value == nullptr) {
      return std::nullopt;
    }
    if (value->type != core::GgufValueType::kBool) {
      WrongType(key);
      return std::nullopt;
    }
    return std::get<bool>(value->value);
  }

  std::optional<std::string> String(std::string_view key) {
    const auto* value = Find(key);
    if (value == nullptr) {
      Missing(key);
      return std::nullopt;
    }
    if (value->type != core::GgufValueType::kString) {
      WrongType(key);
      return std::nullopt;
    }
    return std::string(std::get<std::string_view>(value->value));
  }

  std::optional<std::vector<std::string>> Strings(std::string_view key,
                                                  bool required) {
    const auto* value = Find(key);
    if (value == nullptr) {
      if (required) {
        Missing(key);
      }
      return std::nullopt;
    }
    const auto* items =
        std::get_if<std::vector<std::string_view>>(&value->value);
    if (value->type != core::GgufValueType::kArray || items == nullptr) {
      WrongType(key);
      return std::nullopt;
    }
    return std::vector<std::string>(items->begin(), items->end());
  }

  std::optional<std::vector<std::int64_t>> Integers(std::string_view key) {
    const auto* value = Find(key);
    if (value == nullptr) {
      Missing(key);
      return std::nullopt;
    }
    if (value->type == core::GgufValueType::kArray) {
      if (const auto* items =
              std::get_if<std::vector<std::int64_t>>(&value->value)) {
        return *items;
      }
      if (const auto* items =
              std::get_if<std::vector<std::uint64_t>>(&value->value)) {
        std::vector<std::int64_t> out;
        out.reserve(items->size());
        for (const auto item : *items) {
          out.push_back(static_cast<std::int64_t>(item));
        }
        return out;
      }
    }
    WrongType(key);
    return std::nullopt;
  }
};

constexpr std::string_view kV1Markers[] = {
    "gufo.rotation.version", "gufo.rotation.kind", "gufo.rotation.block_size",
    "gufo.rotation.side"};

bool RefuseV1(const core::GgufReader& reader, std::string* error_msg) {
  bool v1 = false;
  for (const auto key : kV1Markers) {
    v1 = v1 || reader.FindMetadata(key) != nullptr;
  }
  for (const auto& tensor : reader.GetTensors()) {
    v1 = v1 || EndsWith(tensor.name, ".rot");
  }
  if (v1) {
    return Fail(error_msg,
                "the gufo v1 rotation format (gufo.rotation.* markers and "
                ".rot tensors) is no longer supported; reconvert the model "
                "with tools/rotquant to prism.hadamard metadata");
  }
  return true;
}

struct Upstreamed {
  std::uint32_t version{0};
  bool tied{false};
  std::uint32_t block{0};
  std::string sign_mode;
  bool gdn{false};
  std::vector<std::string> weight_names;
  std::vector<std::string> inverse_names;
  // width -> signs in file order
  std::vector<std::pair<std::size_t, std::vector<std::int8_t>>> signs;
};

// Stage 1: the upstream loader's refusals, in its order. Returns false with
// a message, or true with `out` empty (no folding declared) or filled.
bool ReadUpstream(const core::GgufReader& reader,
                  std::optional<Upstreamed>& out, std::string* error_msg) {
  Metadata meta{reader, error_msg};
  out.reset();
  const auto tied_value = meta.Bool(kTiedOutputKey);
  if (meta.failed) {
    return false;
  }
  const bool tied = tied_value.value_or(false);
  const auto version = meta.U32(kVersionKey, /*required=*/false);
  if (meta.failed) {
    return false;
  }
  if (!version.has_value()) {
    if (tied) {
      return Upstream(error_msg, 2224,
                      "prism.hadamard.tied_output requires version 2");
    }
    return true;
  }
  if (*version != 1 && *version != 2) {
    return Upstream(
        error_msg, 2078,
        "unsupported prism.hadamard.version: " + std::to_string(*version));
  }
  if ((*version == 2) != tied) {
    return Upstream(error_msg, 2082,
                    "prism.hadamard version 2 requires tied_output=true; "
                    "version 1 forbids it");
  }
  if (tied && reader.FindTensor("output.weight") != nullptr) {
    return Upstream(error_msg, 2085,
                    "prism.hadamard.tied_output requires output.weight to be "
                    "absent");
  }
  Upstreamed result{.version = *version,
                    .tied = tied,
                    .block = 0,
                    .sign_mode = {},
                    .gdn = false,
                    .weight_names = {},
                    .inverse_names = {},
                    .signs = {}};
  const auto block = meta.U32(kBlockSizeKey, /*required=*/true);
  if (meta.failed) {
    return false;
  }
  const auto transform = meta.String(kTransformKey);
  if (meta.failed) {
    return false;
  }
  const auto axis = meta.String(kAxisKey);
  if (meta.failed) {
    return false;
  }
  const auto sign_mode = meta.String(kSignModeKey);
  if (meta.failed) {
    return false;
  }
  const auto weight_names = meta.Strings(kWeightNamesKey, /*required=*/true);
  if (meta.failed) {
    return false;
  }
  if (*block == 0 || !std::has_single_bit(*block)) {
    return Upstream(
        error_msg, 2101,
        "invalid prism.hadamard.block_size: " + std::to_string(*block));
  }
  if (*transform != kTransform) {
    return Upstream(error_msg, 2104,
                    "unsupported prism.hadamard.transform: " + *transform);
  }
  if (*axis != kAxis) {
    return Upstream(error_msg, 2107,
                    "unsupported prism.hadamard.axis: " + *axis);
  }
  if (*sign_mode != "identity" && *sign_mode != "explicit") {
    return Upstream(error_msg, 2110,
                    "unsupported prism.hadamard.sign_mode: " + *sign_mode);
  }
  if (weight_names->empty()) {
    return Upstream(error_msg, 2113, "prism.hadamard.weight_names is empty");
  }
  result.block = *block;
  result.sign_mode = *sign_mode;
  result.weight_names = *weight_names;
  if (*sign_mode == "explicit") {
    const auto widths = meta.Integers(kSignWidthsKey);
    if (meta.failed) {
      return false;
    }
    const auto values = meta.Integers(kSignValuesKey);
    if (meta.failed) {
      return false;
    }
    if (widths->empty()) {
      return Upstream(error_msg, 2123,
                      "prism.hadamard.sign_mode is explicit but sign_widths "
                      "is empty");
    }
    std::size_t offset = 0;
    for (const auto width : *widths) {
      if (width <= 0 || width % *block != 0 ||
          offset + static_cast<std::size_t>(width) > values->size()) {
        return Upstream(
            error_msg, 2128,
            "invalid prism.hadamard sign width: " + std::to_string(width));
      }
      const auto k = static_cast<std::size_t>(width);
      for (const auto& [seen, unused] : result.signs) {
        if (seen == k) {
          return Upstream(
              error_msg, 2131,
              "duplicate prism.hadamard sign width: " + std::to_string(width));
        }
      }
      std::vector<std::int8_t> vector(k);
      for (std::size_t i = 0; i < k; ++i) {
        const auto value = (*values)[offset + i];
        if (value != 1 && value != -1) {
          return Upstream(error_msg, 2137,
                          "prism.hadamard sign values must be +/-1");
        }
        vector[i] = static_cast<std::int8_t>(value);
      }
      result.signs.emplace_back(k, std::move(vector));
      offset += k;
    }
    if (offset != values->size()) {
      return Upstream(error_msg, 2143,
                      "prism.hadamard.sign_values length mismatch");
    }
  }
  const auto gdn = meta.Bool(kGdnVGroupedKey);
  if (meta.failed) {
    return false;
  }
  result.gdn = gdn.value_or(false);
  const auto arch = reader.GetMetadataString("general.architecture");
  if (!arch.has_value() || !IsSupportedArchitecture(*arch)) {
    return Upstream(error_msg, 2159,
                    "prism.hadamard: arch '" + std::string(arch.value_or("")) +
                        "' is not verified to apply the activation "
                        "transform");
  }
  std::set<std::string> seen;
  for (const auto& name : result.weight_names) {
    if (!IsFoldable(name)) {
      return Upstream(error_msg, 2199,
                      "prism.hadamard: weight '" + name +
                          "' is not on a verified Hadamard-aware matmul path");
    }
    if (!seen.insert(name).second) {
      return Upstream(error_msg, 2203,
                      "duplicate prism.hadamard weight: " + name);
    }
  }
  const auto inverse = meta.Strings(kInverseWeightNamesKey, /*required=*/false);
  if (meta.failed) {
    return false;
  }
  std::set<std::string> inverse_seen;
  for (const auto& name : inverse.value_or(std::vector<std::string>{})) {
    if (name != "token_embd.weight") {
      return Upstream(error_msg, 2213,
                      "prism.hadamard: weight '" + name +
                          "' is not a verified inverse-after-lookup table");
    }
    if (seen.contains(name) || !inverse_seen.insert(name).second) {
      return Upstream(error_msg, 2217,
                      "duplicate prism.hadamard inverse weight: " + name);
    }
    result.inverse_names.push_back(name);
  }
  const bool has_output = reader.FindTensor("output.weight") != nullptr;
  if (tied && !inverse_seen.contains("token_embd.weight")) {
    return Upstream(error_msg, 2228,
                    "prism.hadamard.tied_output requires a latent token "
                    "embedding");
  }
  if (!tied && inverse_seen.contains("token_embd.weight") && !has_output) {
    return Upstream(error_msg, 2232,
                    "a tied Hadamard output requires version 2 and "
                    "tied_output=true");
  }
  std::vector<std::string> all = result.weight_names;
  all.insert(all.end(), result.inverse_names.begin(),
             result.inverse_names.end());
  for (const auto& name : all) {
    const auto* tensor = reader.FindTensor(name);
    if (tensor == nullptr || tensor->dimensions.empty()) {
      return Upstream(error_msg, 2271,
                      "prism.hadamard weight not found: " + name);
    }
    const auto k = tensor->dimensions.front();
    if (k % *block != 0) {
      return Upstream(error_msg, 2274,
                      "prism.hadamard block size " + std::to_string(*block) +
                          " does not divide input dimension " +
                          std::to_string(k) + " for " + name);
    }
    if (!result.signs.empty() &&
        std::none_of(result.signs.begin(), result.signs.end(),
                     [&](const auto& entry) { return entry.first == k; })) {
      return Upstream(error_msg, 2348,
                      "prism.hadamard has no sign vector for width " +
                          std::to_string(k) + " (" + name + ")");
    }
  }
  out = std::move(result);
  return true;
}

}  // namespace

bool RotationInfo::IsGroupRotated(std::string_view group) const noexcept {
  return std::find(rotated_groups.begin(), rotated_groups.end(), group) !=
         rotated_groups.end();
}

const WidthSigns* RotationInfo::ForWidth(std::size_t k) const noexcept {
  for (const auto& width : widths) {
    if (width.k == k) {
      return &width;
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
  if (!RefuseV1(reader, error_msg)) {
    return std::nullopt;
  }
  std::optional<Upstreamed> upstream;
  if (!ReadUpstream(reader, upstream, error_msg)) {
    return std::nullopt;
  }

  // Stage 2: the subset gufo executes.
  if (!upstream.has_value()) {
    std::string stray;
    for (const auto key :
         {kTiedOutputKey, kBlockSizeKey, kTransformKey, kAxisKey, kSignModeKey,
          kSignWidthsKey, kSignValuesKey, kWeightNamesKey,
          kInverseWeightNamesKey, kGdnVGroupedKey}) {
      if (reader.FindMetadata(key) != nullptr) {
        stray += (stray.empty() ? "" : ", ") + std::string(key);
      }
    }
    if (!stray.empty()) {
      Fail(error_msg, stray + " without " + std::string(kVersionKey));
      return std::nullopt;
    }
    return RotationInfo{};
  }
  const auto& prism = *upstream;
  if (prism.version != 1 || prism.tied) {
    Fail(error_msg, "only untied prism.hadamard version 1 files are supported");
    return std::nullopt;
  }
  if (prism.block != kBlockSize) {
    Fail(error_msg, "unsupported " + std::string(kBlockSizeKey) + " = " +
                        std::to_string(prism.block) + " (expected " +
                        std::to_string(kBlockSize) + ")");
    return std::nullopt;
  }
  if (prism.sign_mode != "explicit") {
    Fail(error_msg, "unsupported " + std::string(kSignModeKey) + " = '" +
                        prism.sign_mode + "' (expected 'explicit')");
    return std::nullopt;
  }
  if (!prism.inverse_names.empty()) {
    Fail(error_msg, std::string(kInverseWeightNamesKey) + " is not supported");
    return std::nullopt;
  }
  if (prism.gdn) {
    Fail(error_msg, std::string(kGdnVGroupedKey) +
                        " is not supported: rotation is defined in the GGUF "
                        "(tiled) value-head order");
    return std::nullopt;
  }

  const std::set<std::string> folded(prism.weight_names.begin(),
                                     prism.weight_names.end());
  std::unordered_map<std::string_view, const GroupInstance*> owner;
  for (const auto& instance : table.instances) {
    for (const auto& tensor : instance.tensors) {
      owner.emplace(tensor, &instance);
    }
  }
  for (const auto& name : prism.weight_names) {
    const auto found = owner.find(name);
    if (found == owner.end()) {
      Fail(error_msg, name + " is rotated but its role has no rotation path");
      return std::nullopt;
    }
    if (Contains(table.unsupported_groups, found->second->group)) {
      Fail(error_msg, name + " is rotated, but group " + found->second->group +
                          " has no rotation path in this runtime");
      return std::nullopt;
    }
  }

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
      (folded.contains(tensor) ? rotated : plain).push_back(tensor);
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
    ++count.rotated;
  }
  RotationInfo info;
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

  std::set<std::uint64_t> used;
  for (const auto& name : prism.weight_names) {
    used.insert(reader.FindTensor(name)->dimensions.front());
  }
  for (const auto& [k, values] : prism.signs) {
    if (!used.contains(k)) {
      Fail(error_msg,
           std::string(kSignWidthsKey) +
               " lists a width no folded weight uses: " + std::to_string(k));
      return std::nullopt;
    }
    WidthSigns width{
        .k = k, .bits = std::vector<std::uint32_t>(PackedSignWords(k), 0U)};
    for (std::size_t i = 0; i < k; ++i) {
      if (values[i] < 0) {
        width.bits[i / 32] |= 1U << (i % 32);
      }
    }
    info.widths.push_back(std::move(width));
  }
  info.weight_names = prism.weight_names;
  return info;
}

void RotateRow(std::span<float> row, std::span<const std::uint32_t> bits) {
  constexpr std::size_t kSegment = hip::hadamard::kSegment;
  for (std::size_t base = 0; base + kSegment <= row.size(); base += kSegment) {
    float* x = row.data() + base;
    for (std::size_t j = 0; j < kSegment; ++j) {
      const std::size_t i = base + j;
      if (((bits[i / 32] >> (i % 32)) & 1U) != 0) {
        x[j] = std::bit_cast<float>(std::bit_cast<std::uint32_t>(x[j]) ^
                                    0x80000000U);
      }
    }
    for (std::size_t length = 1; length < kSegment; length <<= 1) {
      for (std::size_t i = 0; i < kSegment; i += length << 1) {
        for (std::size_t j = i; j < i + length; ++j) {
          const float a = x[j];
          const float b = x[j + length];
          x[j] = a + b;
          x[j + length] = a - b;
        }
      }
    }
    for (std::size_t j = 0; j < kSegment; ++j) {
      x[j] *= hip::hadamard::kHadamard128Scale;
    }
  }
}

}  // namespace gufo::quant::rotation
