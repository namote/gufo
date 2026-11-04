#include "src/models/qwen/native_pack.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>

#include "src/core/gguf_identity.hpp"
#include "src/core/platform/fs.hpp"
#include "src/core/platform/mapped_file.hpp"
#include "src/core/quant/exl3/checkpoint.hpp"
#include "src/core/quant/exl3/layout.hpp"

namespace gufo::models::qwen {
namespace {

constexpr std::string_view kSourceFilesKey = "gufo.exl3.source_files";
constexpr std::string_view kSourceStampsKey = "gufo.exl3.source_stamps";

std::string Read(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot read " + p.string());
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

// Qwen3.5 tokenizer contract: NFC, the qwen35 split regex, byte-level BPE.
void TokenizerContract(const json::Value& tokenizer) {
  const auto fail = [] {
    throw std::runtime_error(
        "unsupported native tokenizer contract (requires Qwen35 NFC/byte BPE)");
  };
  if (NativeMember(tokenizer, "normalizer").member_str("type") != "NFC")
    fail();
  const auto& pre = NativeMember(tokenizer, "pre_tokenizer");
  const auto& parts = NativeMember(pre, "pretokenizers");
  if (pre.member_str("type") != "Sequence" || !parts.is_array() ||
      parts.size() != 2)
    fail();
  const auto& split = parts.items()[0];
  constexpr auto regex =
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
  if (split.member_str("type") != "Split" ||
      NativeMember(split, "pattern").member_str("Regex") != regex ||
      split.member_str("behavior") != "Isolated" ||
      NativeBoolean(split, "invert"))
    fail();
  for (const auto* v :
       {&parts.items()[1], &NativeMember(tokenizer, "post_processor"),
        &NativeMember(tokenizer, "decoder")})
    if (v->member_str("type") != "ByteLevel" ||
        NativeBoolean(*v, "add_prefix_space") ||
        NativeBoolean(*v, "trim_offsets") || NativeBoolean(*v, "use_regex"))
      fail();
  const auto& model = NativeMember(tokenizer, "model");
  if (model.member_str("type") != "BPE" ||
      !NativeMember(model, "dropout").is_null() ||
      !NativeMember(model, "unk_token").is_null() ||
      !NativeMember(model, "continuing_subword_prefix").str().empty() ||
      !NativeMember(model, "end_of_word_suffix").str().empty() ||
      NativeBoolean(model, "fuse_unk") ||
      NativeBoolean(model, "byte_fallback") ||
      NativeBoolean(model, "ignore_merges"))
    fail();
  if (!NativeMember(tokenizer, "added_tokens").is_array())
    fail();
  for (const auto& token : NativeMember(tokenizer, "added_tokens").items())
    if (!NativeMember(token, "special").is_bool() ||
        NativeBoolean(token, "single_word") || NativeBoolean(token, "lstrip") ||
        NativeBoolean(token, "rstrip") || NativeBoolean(token, "normalized"))
      fail();
}

std::string Stamp(const std::filesystem::path& p) {
  auto file = platform::ReadOnlyMappedFile::Open(p);
  platform::FileInfo info{};
  std::error_code ec;
  if (!file.valid() || !platform::Stat(file.PlatformHandle(), &info, ec) ||
      !info.regular)
    throw std::runtime_error("cannot identify native source " + p.string());
  return std::to_string(info.device) + ":" + std::to_string(info.inode) + ":" +
         std::to_string(info.size) + ":" + std::to_string(info.modified_ns) +
         ":" + std::to_string(info.changed_ns);
}

}  // namespace

const json::Value& NativeMember(const json::Value& value, const char* key) {
  const auto* result = value.find(key);
  if (!result)
    throw std::runtime_error(std::string("missing native configuration ") +
                             key);
  return *result;
}

bool NativeBoolean(const json::Value& value, const char* key) {
  const auto& member = NativeMember(value, key);
  if (!member.is_bool())
    throw std::runtime_error(std::string("invalid native boolean ") + key);
  return member.as_bool();
}

std::uint64_t NativeUInt(const json::Value& value) {
  const double d = value.as_double(-1);
  if (!value.is_number() || !std::isfinite(d) || d < 0 || d > UINT32_MAX ||
      std::floor(d) != d)
    throw std::runtime_error("invalid native integer");
  return static_cast<std::uint64_t>(d);
}

NativePackMetadata::NativePackMetadata(
    const std::filesystem::path& directory, std::string_view model_type,
    std::span<const std::string_view> required_files,
    std::span<const std::string_view> optional_files)
    : directory_(directory) {
  if (!quant::exl3::Checkpoint::IsExl3(directory))
    throw std::runtime_error("directory is not an EXL3 pack");
  std::set<std::string> names{"config.json", "tokenizer.json",
                              "tokenizer_config.json", "chat_template.jinja",
                              "model.safetensors.index.json"};
  for (const auto name : required_files)
    names.emplace(name);
  const auto index_path = directory / "model.safetensors.index.json";
  const auto index_stamp = Stamp(index_path);
  const auto index = json::parse(Read(index_path));
  const auto& weight_map = NativeMember(index, "weight_map");
  if (!weight_map.is_object() || weight_map.size() == 0)
    throw std::runtime_error("empty native weight index");
  for (const auto& [key, value] : weight_map.members()) {
    const auto& name = value.str();
    if (name.empty() || name.find('/') != std::string::npos ||
        name.find("..") != std::string::npos ||
        std::filesystem::path(name).is_absolute())
      throw std::runtime_error("invalid native shard name");
    names.insert(name);
  }
  for (const auto name : optional_files)
    if (std::filesystem::is_regular_file(directory / name))
      names.emplace(name);
  for (const auto& name : names) {
    file_names_.push_back(name);
    stamps_.push_back(Stamp(directory / name));
  }
  if (Stamp(index_path) != index_stamp)
    throw std::runtime_error("native index changed while loading");
  config_ = json::parse(Read(directory / "config.json"));
  if (text_config().member_str("model_type") != model_type)
    throw std::runtime_error("unsupported native EXL3 architecture");
  tokenizer_ = json::parse(Read(directory / "tokenizer.json"));
  tokenizer_config_ = json::parse(Read(directory / "tokenizer_config.json"));
  TokenizerContract(tokenizer_);
}

const json::Value& NativePackMetadata::text_config() const {
  return NativeMember(config_, "text_config");
}

std::uint64_t NativePackMetadata::Number(const char* key) const {
  return NativeUInt(NativeMember(text_config(), key));
}

std::string_view NativePackMetadata::Own(std::string value) {
  strings_.push_back(std::move(value));
  return strings_.back();
}

void NativePackMetadata::Text(std::string key, std::string value) {
  metadata_.emplace_back(std::move(key),
                         core::GgufMetadataValue{core::GgufValueType::kString,
                                                 Own(std::move(value))});
}

void NativePackMetadata::U32(std::string key, std::uint64_t value) {
  metadata_.emplace_back(
      std::move(key),
      core::GgufMetadataValue{core::GgufValueType::kUint32, value});
}

void NativePackMetadata::F32(std::string key, double value) {
  if (!std::isfinite(value))
    throw std::runtime_error("invalid native float");
  metadata_.emplace_back(
      std::move(key),
      core::GgufMetadataValue{core::GgufValueType::kFloat32, value});
}

void NativePackMetadata::Bool(std::string key, bool value) {
  metadata_.emplace_back(std::move(key),
                         core::GgufMetadataValue{core::GgufValueType::kBool,
                                                 static_cast<bool>(value)});
}

void NativePackMetadata::Array(std::string key,
                               std::vector<std::uint64_t> values) {
  metadata_.emplace_back(
      std::move(key),
      core::GgufMetadataValue{core::GgufValueType::kArray, std::move(values)});
}

std::unique_ptr<core::GgufReader> NativePackMetadata::Finish(
    std::optional<std::uint64_t> bos_token_id) {
  using V = core::GgufMetadataValue;
  using T = core::GgufValueType;
  const auto& model = NativeMember(tokenizer_, "model");
  const auto& added = NativeMember(tokenizer_, "added_tokens");
  const auto vocab_size = Number("vocab_size");
  if (vocab_size == 0 || vocab_size > 1000000)
    throw std::runtime_error("unsupported vocabulary size");
  std::vector<std::string> tokens(vocab_size);
  std::vector<bool> seen(vocab_size);
  std::vector<std::uint64_t> token_types(vocab_size, 1);
  const auto add = [&](const std::string& value, std::uint64_t id) {
    if (id >= vocab_size || (seen[id] && tokens[id] != value))
      throw std::runtime_error("invalid native token ID");
    seen[id] = true;
    tokens[id] = value;
  };
  for (const auto& [value, id] : NativeMember(model, "vocab").members())
    add(value, NativeUInt(id));
  for (const auto& v : added.items()) {
    const auto id = NativeUInt(NativeMember(v, "id"));
    const auto content = v.member_str("content");
    add(content, id);
    // convert_hf_to_gguf.py: special or <|...|>-looking tokens are CONTROL
    // (3), other added tokens USER_DEFINED (4).
    const bool looks_special =
        content.starts_with("<|") && content.ends_with("|>");
    token_types[id] = NativeBoolean(v, "special") || looks_special ? 3 : 4;
  }
  const auto padding = std::find(seen.begin(), seen.end(), false);
  if (padding == seen.begin() ||
      std::find(padding, seen.end(), true) != seen.end())
    throw std::runtime_error("native vocabulary has holes");
  for (auto i = static_cast<std::size_t>(padding - seen.begin());
       i < tokens.size(); ++i) {
    tokens[i] = "[PAD" + std::to_string(i) + "]";
    token_types[i] = 5;
  }
  std::vector<std::string_view> token_views, merges;
  for (auto& value : tokens)
    token_views.push_back(Own(std::move(value)));
  for (const auto& v : NativeMember(model, "merges").items()) {
    if (v.is_string())
      merges.push_back(Own(v.str()));
    else if (v.is_array() && v.size() == 2)
      merges.push_back(Own(v.items()[0].str() + " " + v.items()[1].str()));
    else
      throw std::runtime_error("invalid native merge");
  }
  metadata_.emplace_back("tokenizer.ggml.tokens",
                         V{T::kArray, std::move(token_views)});
  metadata_.emplace_back("tokenizer.ggml.merges",
                         V{T::kArray, std::move(merges)});
  Array("tokenizer.ggml.token_type", std::move(token_types));
  Text("tokenizer.ggml.model", "gpt2");
  Text("tokenizer.ggml.pre", "qwen35");
  const auto special_id = [&](const char* field) {
    const auto& value = NativeMember(tokenizer_config_, field);
    const auto name =
        value.is_string() ? value.str() : value.member_str("content");
    for (const auto& v : added.items())
      if (v.member_str("content") == name &&
          NativeMember(v, "special").as_bool())
        return NativeUInt(NativeMember(v, "id"));
    throw std::runtime_error(std::string("missing special token ") + field);
  };
  U32("tokenizer.ggml.eos_token_id", special_id("eos_token"));
  U32("tokenizer.ggml.padding_token_id", special_id("pad_token"));
  if (bos_token_id) {
    U32("tokenizer.ggml.bos_token_id", *bos_token_id);
    Bool("tokenizer.ggml.add_bos_token", false);
  }
  const auto template_text = Read(directory_ / "chat_template.jinja");
  if (const auto* embedded = tokenizer_config_.find("chat_template");
      embedded && embedded->is_string() && embedded->str() != template_text)
    throw std::runtime_error("native chat templates disagree");
  Text("tokenizer.chat_template", template_text);

  Text(std::string(quant::exl3::kCheckpointKey),
       std::filesystem::absolute(directory_).string());
  Bool(std::string(quant::exl3::kNativeKey), true);
  std::vector<std::string_view> source_names, source_stamps;
  for (std::size_t i = 0; i < file_names_.size(); ++i) {
    if (Stamp(directory_ / file_names_[i]) != stamps_[i])
      throw std::runtime_error("native source changed while loading metadata");
    source_names.push_back(Own(file_names_[i]));
    source_stamps.push_back(Own(stamps_[i]));
  }
  metadata_.emplace_back(std::string(kSourceFilesKey),
                         V{T::kArray, std::move(source_names)});
  metadata_.emplace_back(std::string(kSourceStampsKey),
                         V{T::kArray, std::move(source_stamps)});
  return core::GgufReader::FromMetadata(std::move(metadata_));
}

void ValidateNativeSources(const core::GgufReader& metadata) {
  if (!metadata.GetMetadataBool(quant::exl3::kNativeKey).value_or(false))
    return;
  const auto directory =
      metadata.GetMetadataString(quant::exl3::kCheckpointKey);
  const auto files = metadata.GetMetadataStringArray(kSourceFilesKey);
  const auto stamps = metadata.GetMetadataStringArray(kSourceStampsKey);
  if (!directory || files.empty() || files.size() != stamps.size())
    throw std::runtime_error("missing native source identity");
  for (std::size_t i = 0; i < files.size(); ++i)
    if (Stamp(std::filesystem::path(*directory) / files[i]) != stamps[i])
      throw std::runtime_error("native source changed since metadata load");
}

std::string NativeArtifactIdentity(const core::GgufReader& metadata) {
  ValidateNativeSources(metadata);
  const auto directory =
      metadata.GetMetadataString(quant::exl3::kCheckpointKey);
  const auto files = metadata.GetMetadataStringArray(kSourceFilesKey);
  std::vector<std::string> names;
  for (const auto file : files)
    names.emplace_back(file);
  const auto result =
      core::FileSetIdentityHex(std::filesystem::path(*directory), names);
  ValidateNativeSources(metadata);
  return result;
}

}  // namespace gufo::models::qwen
