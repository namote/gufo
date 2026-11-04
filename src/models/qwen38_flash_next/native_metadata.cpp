#include "src/models/qwen38_flash_next/native_metadata.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "src/core/gguf_identity.hpp"
#include "src/core/json.hpp"
#include "src/core/platform/fs.hpp"
#include "src/core/quant/exl3/checkpoint.hpp"
#include "src/core/quant/exl3/layout.hpp"
#include "src/models/qwen38_flash_next/exl3_ngram.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {
std::string Read(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot read " + p.string());
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}
const json::Value& Member(const json::Value& v, const char* key) {
  const auto* result = v.find(key);
  if (!result)
    throw std::runtime_error(std::string("missing native configuration ") +
                             key);
  return *result;
}
bool Boolean(const json::Value& v, const char* key) {
  const auto& value = Member(v, key);
  if (!value.is_bool())
    throw std::runtime_error(std::string("invalid native boolean ") + key);
  return value.as_bool();
}
void TokenizerContract(const json::Value& tokenizer) {
  const auto fail = [] {
    throw std::runtime_error(
        "unsupported native tokenizer contract (requires Qwen35 NFC/byte BPE)");
  };
  if (Member(tokenizer, "normalizer").member_str("type") != "NFC")
    fail();
  const auto& pre = Member(tokenizer, "pre_tokenizer");
  const auto& parts = Member(pre, "pretokenizers");
  if (pre.member_str("type") != "Sequence" || !parts.is_array() ||
      parts.size() != 2)
    fail();
  const auto& split = parts.items()[0];
  constexpr auto regex =
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
  if (split.member_str("type") != "Split" ||
      Member(split, "pattern").member_str("Regex") != regex ||
      split.member_str("behavior") != "Isolated" || Boolean(split, "invert"))
    fail();
  for (const auto* v : {&parts.items()[1], &Member(tokenizer, "post_processor"),
                        &Member(tokenizer, "decoder")})
    if (v->member_str("type") != "ByteLevel" ||
        Boolean(*v, "add_prefix_space") || Boolean(*v, "trim_offsets") ||
        Boolean(*v, "use_regex"))
      fail();
  const auto& model = Member(tokenizer, "model");
  if (model.member_str("type") != "BPE" ||
      !Member(model, "dropout").is_null() ||
      !Member(model, "unk_token").is_null() ||
      !Member(model, "continuing_subword_prefix").str().empty() ||
      !Member(model, "end_of_word_suffix").str().empty() ||
      Boolean(model, "fuse_unk") || Boolean(model, "byte_fallback") ||
      Boolean(model, "ignore_merges"))
    fail();
  if (!Member(tokenizer, "added_tokens").is_array())
    fail();
  for (const auto& token : Member(tokenizer, "added_tokens").items())
    if (!Member(token, "special").is_bool() || Boolean(token, "single_word") ||
        Boolean(token, "lstrip") || Boolean(token, "rstrip") ||
        Boolean(token, "normalized"))
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
std::uint64_t UInt(const json::Value& v) {
  const double d = v.as_double(-1);
  if (!v.is_number() || !std::isfinite(d) || d < 0 || d > UINT32_MAX ||
      std::floor(d) != d)
    throw std::runtime_error("invalid native integer");
  return static_cast<std::uint64_t>(d);
}
}  // namespace
std::unique_ptr<core::GgufReader> OpenArtifactMetadata(
    const std::filesystem::path& path, std::string* error) {
  std::error_code ec;
  if (!std::filesystem::is_directory(path, ec)) {
    auto reader = core::GgufReader::OpenFile(path, error);
    if (reader && reader->FindMetadata(quant::exl3::kCheckpointKey) &&
        reader->GetMetadataString("general.architecture") ==
            std::optional<std::string_view>("qwen4exp")) {
      if (error)
        *error =
            "EXL3 metadata sidecar files are not supported; pass the EXL3 "
            "pack directory";
      return nullptr;
    }
    return reader;
  }
  try {
    if (!quant::exl3::Checkpoint::IsExl3(path))
      throw std::runtime_error("directory is not an EXL3 pack");
    std::set<std::string> file_names{"config.json",
                                     "tokenizer.json",
                                     "tokenizer_config.json",
                                     "chat_template.jinja",
                                     "ngram_embedding.safetensors",
                                     "model.safetensors.index.json"};
    const auto index_stamp = Stamp(path / "model.safetensors.index.json");
    const auto index = json::parse(Read(path / "model.safetensors.index.json"));
    if (!Member(index, "weight_map").is_object() ||
        Member(index, "weight_map").size() == 0)
      throw std::runtime_error("empty native weight index");
    for (const auto& [key, value] : Member(index, "weight_map").members()) {
      const auto& name = value.str();
      if (name.empty() || name.find('/') != std::string::npos ||
          name.find("..") != std::string::npos ||
          std::filesystem::path(name).is_absolute())
        throw std::runtime_error("invalid native shard name");
      file_names.insert(name);
    }
    if (std::filesystem::is_regular_file(
            path / "mtp_hyper_connection_mixer_patch.safetensors"))
      file_names.insert("mtp_hyper_connection_mixer_patch.safetensors");
    std::vector<std::string> stamps;
    for (const auto& name : file_names)
      stamps.push_back(Stamp(path / name));
    if (Stamp(path / "model.safetensors.index.json") != index_stamp)
      throw std::runtime_error("native index changed while loading");
    const auto config = json::parse(Read(path / "config.json"));
    const auto& c = Member(config, "text_config");
    if (c.member_str("model_type") != "qwen4_exp_text")
      throw std::runtime_error("unsupported native EXL3 architecture");
    const auto tokenizer = json::parse(Read(path / "tokenizer.json"));
    const auto tc = json::parse(Read(path / "tokenizer_config.json"));
    TokenizerContract(tokenizer);
    const auto& tm = Member(tokenizer, "model");
    if (tm.member_str("type") != "BPE")
      throw std::runtime_error("native tokenizer must use BPE");
    auto ngram = Exl3Ngram::Open(path / "ngram_embedding.safetensors");
    using V = core::GgufMetadataValue;
    using T = core::GgufValueType;
    std::vector<std::pair<std::string, V>> metadata;
    std::deque<std::string> strings;
    const auto own = [&](std::string s) -> std::string_view {
      strings.push_back(std::move(s));
      return strings.back();
    };
    const auto text = [&](std::string k, std::string s) {
      metadata.emplace_back(std::move(k), V{T::kString, own(std::move(s))});
    };
    const auto u32 = [&](std::string k, std::uint64_t n) {
      metadata.emplace_back(std::move(k), V{T::kUint32, n});
    };
    const auto f32 = [&](std::string k, double n) {
      if (!std::isfinite(n))
        throw std::runtime_error("invalid native float");
      metadata.emplace_back(std::move(k), V{T::kFloat32, n});
    };
    const auto array = [&](std::string k, std::vector<std::uint64_t> values) {
      metadata.emplace_back(std::move(k), V{T::kArray, std::move(values)});
    };
    text("general.architecture", "qwen4exp");
    text("general.name", "Qwen3.8 Flash Next");
    text(std::string(quant::exl3::kCheckpointKey),
         std::filesystem::absolute(path).string());
    metadata.emplace_back(std::string(quant::exl3::kNativeKey),
                          V{T::kBool, true});
    const auto number = [&](const char* key) { return UInt(Member(c, key)); };
    if (c.member_str("hidden_act") != "silu" ||
        c.member_str("output_gate_type") != "sigmoid" ||
        Boolean(c, "attention_bias") || Boolean(c, "tie_word_embeddings") ||
        Boolean(c, "mtp_use_dedicated_embeddings") ||
        number("indexer_kv_heads") != 1)
      throw std::runtime_error("unsupported native layer contract");
    const auto layers = number("num_hidden_layers");
    const auto mtp = number("mtp_num_hidden_layers");
    u32("qwen4exp.block_count", layers + mtp);
    u32("qwen4exp.nextn_predict_layers", mtp);
    const std::pair<const char*, const char*> keys[] = {
        {"embedding_length", "hidden_size"},
        {"context_length", "max_position_embeddings"},
        {"hyper_connection.count", "hc_count"},
        {"hyper_connection.low_rank", "hc_lowrank"},
        {"full_attention_interval", "full_attention_interval"},
        {"attention.head_count", "num_attention_heads"},
        {"attention.head_count_kv", "num_key_value_heads"},
        {"attention.key_length", "head_dim"},
        {"attention.value_length", "head_dim"},
        {"attention.indexer.head_count", "indexer_n_heads"},
        {"attention.indexer.key_length", "indexer_head_dim"},
        {"attention.indexer.top_k", "indexer_budget"},
        {"ssm.conv_kernel", "linear_conv_kernel_dim"},
        {"ssm.state_size", "linear_key_head_dim"},
        {"ssm.group_count", "linear_num_key_heads"},
        {"ssm.time_step_rank", "linear_num_value_heads"},
        {"expert_count", "num_experts"},
        {"expert_used_count", "num_experts_per_tok"},
        {"expert_feed_forward_length", "moe_intermediate_size"},
        {"expert_shared_feed_forward_length",
         "shared_expert_intermediate_size"},
        {"ple.ngram_size", "ngram_size"},
        {"ple.heads_per_ngram", "heads_per_ngram"},
        {"ple.conv_kernel", "ple_conv_kernel_size"},
        {"ple.eos_token_id", "eos_token_id"}};
    for (const auto& [dst, src] : keys)
      u32(std::string("qwen4exp.") + dst, number(src));
    if (number("linear_key_head_dim") != number("linear_value_head_dim"))
      throw std::runtime_error("unsupported SSM value dimension");
    u32("qwen4exp.ssm.inner_size",
        number("linear_value_head_dim") * number("linear_num_value_heads"));
    u32("qwen4exp.embedding_length_per_layer_input", ngram.dim);
    if (number("ple_embed_dim") != ngram.dim * ngram.offsets.size())
      throw std::runtime_error("PLE embedding width mismatch");
    f32("qwen4exp.attention.layer_norm_rms_epsilon",
        Member(c, "rms_norm_eps").as_double());
    const auto& rope = Member(c, "rope_parameters");
    if (!Boolean(rope, "mrope_interleaved") ||
        rope.member_str("rope_type") != "default")
      throw std::runtime_error("unsupported native rope type");
    f32("qwen4exp.rope.freq_base", Member(rope, "rope_theta").as_double());
    const auto rotary =
        Member(rope, "partial_rotary_factor").as_double() * number("head_dim");
    u32("qwen4exp.rope.dimension_count", UInt(json::Value(rotary)));
    std::vector<std::uint64_t> sections;
    for (const auto& v : Member(rope, "mrope_section").items())
      sections.push_back(UInt(v));
    sections.push_back(0);
    array("qwen4exp.rope.dimension_sections", std::move(sections));
    std::vector<std::uint64_t> ratios;
    const auto& types = Member(c, "layer_types");
    if (types.size() != layers || layers > 1024 || mtp > 1)
      throw std::runtime_error("invalid native layer types");
    for (const auto& type : types.items()) {
      if (type.str() != "linear_attention" && type.str() != "full_attention")
        throw std::runtime_error("unknown native layer type");
      ratios.push_back(type.str() == "full_attention"
                           ? number("indexer_compress_ratio")
                           : 0);
    }
    if (mtp) {
      const auto& next = Member(c, "mtp");
      if (!Boolean(next, "hybrid") ||
          UInt(Member(next, "num_hidden_layers")) != mtp ||
          Member(next, "layer_types").size() != 1 ||
          Member(next, "layer_types").items()[0].str() != "full_attention" ||
          !Member(next, "mtp_use_hidden_state_from_layer").is_null() ||
          Member(next, "rope_theta").as_double() !=
              Member(rope, "rope_theta").as_double())
        throw std::runtime_error("unsupported native MTP contract");
    }
    ratios.insert(ratios.end(), mtp, number("indexer_compress_ratio"));
    array("qwen4exp.attention.compress_ratios", std::move(ratios));
    std::vector<std::uint64_t> ple;
    for (const auto& v : Member(c, "ple_layer_ids").items()) {
      const auto id = UInt(v);
      if (id == 0)
        throw std::runtime_error("PLE layer IDs are one-based");
      ple.push_back(id - 1);
    }
    array("qwen4exp.ple.layers", std::move(ple));
    array("qwen4exp.ple.head_offsets", ngram.offsets);
    array("qwen4exp.ple.head_vocab_sizes", ngram.vocab);
    array("qwen4exp.ple.layer_multipliers", ngram.multipliers);

    const auto vocab_size = number("vocab_size");
    if (vocab_size == 0 || vocab_size > 1000000)
      throw std::runtime_error("unsupported vocabulary size");
    std::vector<std::string> tokens(vocab_size);
    std::vector<bool> seen(vocab_size);
    std::vector<std::uint64_t> token_types(vocab_size, 1);
    const auto add = [&](const std::string& value, std::uint64_t id,
                         bool special) {
      if (id >= vocab_size || (seen[id] && tokens[id] != value))
        throw std::runtime_error("invalid native token ID");
      seen[id] = true;
      tokens[id] = value;
      if (special)
        token_types[id] = 3;
    };
    for (const auto& [value, id] : Member(tm, "vocab").members())
      add(value, UInt(id), false);
    for (const auto& v : Member(tokenizer, "added_tokens").items()) {
      const auto id = UInt(Member(v, "id"));
      add(v.member_str("content"), id, Boolean(v, "special"));
      token_types[id] = Boolean(v, "special") ? 3 : 4;
    }
    const auto padding = std::find(seen.begin(), seen.end(), false);
    if (padding == seen.begin() ||
        std::find(padding, seen.end(), true) != seen.end())
      throw std::runtime_error("native vocabulary has holes");
    for (std::size_t i = padding - seen.begin(); i < tokens.size(); ++i) {
      tokens[i] = "[PAD" + std::to_string(i) + "]";
      token_types[i] = 5;
    }
    std::vector<std::string_view> token_views, merges;
    for (auto& value : tokens)
      token_views.push_back(own(std::move(value)));
    for (const auto& v : Member(tm, "merges").items()) {
      if (v.is_string())
        merges.push_back(own(v.str()));
      else if (v.is_array() && v.size() == 2)
        merges.push_back(own(v.items()[0].str() + " " + v.items()[1].str()));
      else
        throw std::runtime_error("invalid native merge");
    }
    metadata.emplace_back("tokenizer.ggml.tokens",
                          V{T::kArray, std::move(token_views)});
    metadata.emplace_back("tokenizer.ggml.merges",
                          V{T::kArray, std::move(merges)});
    array("tokenizer.ggml.token_type", std::move(token_types));
    text("tokenizer.ggml.model", "gpt2");
    text("tokenizer.ggml.pre", "qwen35");
    const auto special_id = [&](const char* field) {
      const auto& value = Member(tc, field);
      const auto name =
          value.is_string() ? value.str() : value.member_str("content");
      for (const auto& v : Member(tokenizer, "added_tokens").items())
        if (v.member_str("content") == name && Member(v, "special").as_bool())
          return UInt(Member(v, "id"));
      throw std::runtime_error(std::string("missing special token ") + field);
    };
    u32("tokenizer.ggml.eos_token_id", special_id("eos_token"));
    u32("tokenizer.ggml.padding_token_id", special_id("pad_token"));
    // HF has no BOS insertion for this chat tokenizer. The model's EOS is
    // used for PLE hashing above; generation stops on im_end/endoftext.
    const auto template_text = Read(path / "chat_template.jinja");
    if (const auto* embedded = tc.find("chat_template");
        embedded && embedded->is_string() && embedded->str() != template_text)
      throw std::runtime_error("native chat templates disagree");
    text("tokenizer.chat_template", template_text);
    std::vector<std::string_view> source_names, source_stamps;
    std::size_t file_index = 0;
    for (const auto& name : file_names) {
      if (Stamp(path / name) != stamps[file_index])
        throw std::runtime_error(
            "native source changed while loading metadata");
      source_names.push_back(own(name));
      source_stamps.push_back(own(stamps[file_index++]));
    }
    metadata.emplace_back("gufo.exl3.source_files",
                          V{T::kArray, std::move(source_names)});
    metadata.emplace_back("gufo.exl3.source_stamps",
                          V{T::kArray, std::move(source_stamps)});
    auto result = core::GgufReader::FromMetadata(std::move(metadata));
    auto parsed = Config::FromGguf(*result, true, error);
    if (!parsed)
      return nullptr;
    ngram.Validate(*parsed);
    return result;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return nullptr;
  }
}
}  // namespace gufo::models::qwen38_flash_next

namespace gufo::models::qwen38_flash_next {
void ValidateNativeSources(const core::GgufReader& metadata) {
  if (!metadata.GetMetadataBool(quant::exl3::kNativeKey).value_or(false))
    return;
  const auto directory =
      metadata.GetMetadataString(quant::exl3::kCheckpointKey);
  const auto files = metadata.GetMetadataStringArray("gufo.exl3.source_files");
  const auto stamps =
      metadata.GetMetadataStringArray("gufo.exl3.source_stamps");
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
  const auto files = metadata.GetMetadataStringArray("gufo.exl3.source_files");
  std::vector<std::string> names;
  for (const auto file : files)
    names.emplace_back(file);
  const auto result =
      core::FileSetIdentityHex(std::filesystem::path(*directory), names);
  ValidateNativeSources(metadata);
  return result;
}
}  // namespace gufo::models::qwen38_flash_next
