#include "src/models/qwen38_flash_next/native_metadata.hpp"

#include <stdexcept>
#include <string_view>
#include <vector>

#include "src/core/quant/exl3/layout.hpp"
#include "src/models/qwen/native_pack.hpp"
#include "src/models/qwen38_flash_next/exl3_ngram.hpp"

namespace gufo::models::qwen38_flash_next {
std::unique_ptr<core::GgufReader> OpenArtifactMetadata(
    const std::filesystem::path& path, std::string* error) {
  std::error_code ec;
  if (!std::filesystem::is_directory(path, ec)) {
    auto reader = core::GgufReader::OpenFile(path, error);
    if (reader && reader->FindMetadata(quant::exl3::kCheckpointKey)) {
      if (error)
        *error =
            "EXL3 metadata sidecar files are not supported; pass the EXL3 "
            "pack directory";
      return nullptr;
    }
    return reader;
  }
  try {
    const auto ngram_files = Exl3Ngram::PackFiles(path);
    std::vector<std::string> ngram_names;
    for (const auto& file : ngram_files)
      ngram_names.push_back(file.filename().string());
    const std::vector<std::string_view> required(ngram_names.begin(),
                                                 ngram_names.end());
    constexpr std::string_view kOptional[] = {
        "mtp_hyper_connection_mixer_patch.safetensors", "hc_q8.json"};
    qwen::NativePackMetadata pack(path, "qwen4_exp_text", required, kOptional);
    const auto& c = pack.text_config();
    auto ngram = Exl3Ngram::Open(ngram_files);
    pack.Text("general.architecture", "qwen4exp");
    pack.Text("general.name", "Qwen3.8 Flash Next");
    const auto number = [&](const char* key) { return pack.Number(key); };
    if (c.member_str("hidden_act") != "silu" ||
        c.member_str("output_gate_type") != "sigmoid" ||
        qwen::NativeBoolean(c, "attention_bias") ||
        qwen::NativeBoolean(c, "tie_word_embeddings") ||
        qwen::NativeBoolean(c, "mtp_use_dedicated_embeddings") ||
        number("indexer_kv_heads") != 1)
      throw std::runtime_error("unsupported native layer contract");
    const auto layers = number("num_hidden_layers");
    const auto mtp = number("mtp_num_hidden_layers");
    pack.U32("qwen4exp.block_count", layers + mtp);
    pack.U32("qwen4exp.nextn_predict_layers", mtp);
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
      pack.U32(std::string("qwen4exp.") + dst, number(src));
    if (number("linear_key_head_dim") != number("linear_value_head_dim"))
      throw std::runtime_error("unsupported SSM value dimension");
    pack.U32("qwen4exp.ssm.inner_size", number("linear_value_head_dim") *
                                            number("linear_num_value_heads"));
    pack.U32("qwen4exp.embedding_length_per_layer_input", ngram.dim);
    if (number("ple_embed_dim") != ngram.dim * ngram.offsets.size())
      throw std::runtime_error("PLE embedding width mismatch");
    pack.F32("qwen4exp.attention.layer_norm_rms_epsilon",
             qwen::NativeMember(c, "rms_norm_eps").as_double());
    const auto& rope = qwen::NativeMember(c, "rope_parameters");
    if (!qwen::NativeBoolean(rope, "mrope_interleaved") ||
        rope.member_str("rope_type") != "default")
      throw std::runtime_error("unsupported native rope type");
    pack.F32("qwen4exp.rope.freq_base",
             qwen::NativeMember(rope, "rope_theta").as_double());
    const auto rotary =
        qwen::NativeMember(rope, "partial_rotary_factor").as_double() *
        number("head_dim");
    pack.U32("qwen4exp.rope.dimension_count",
             qwen::NativeUInt(json::Value(rotary)));
    std::vector<std::uint64_t> sections;
    for (const auto& v : qwen::NativeMember(rope, "mrope_section").items())
      sections.push_back(qwen::NativeUInt(v));
    sections.push_back(0);
    pack.Array("qwen4exp.rope.dimension_sections", std::move(sections));
    std::vector<std::uint64_t> ratios;
    const auto& types = qwen::NativeMember(c, "layer_types");
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
      const auto& next = qwen::NativeMember(c, "mtp");
      if (!qwen::NativeBoolean(next, "hybrid") ||
          qwen::NativeUInt(qwen::NativeMember(next, "num_hidden_layers")) !=
              mtp ||
          qwen::NativeMember(next, "layer_types").size() != 1 ||
          qwen::NativeMember(next, "layer_types").items()[0].str() !=
              "full_attention" ||
          !qwen::NativeMember(next, "mtp_use_hidden_state_from_layer")
               .is_null() ||
          qwen::NativeMember(next, "rope_theta").as_double() !=
              qwen::NativeMember(rope, "rope_theta").as_double())
        throw std::runtime_error("unsupported native MTP contract");
    }
    ratios.insert(ratios.end(), mtp, number("indexer_compress_ratio"));
    pack.Array("qwen4exp.attention.compress_ratios", std::move(ratios));
    std::vector<std::uint64_t> ple;
    for (const auto& v : qwen::NativeMember(c, "ple_layer_ids").items()) {
      const auto id = qwen::NativeUInt(v);
      if (id == 0)
        throw std::runtime_error("PLE layer IDs are one-based");
      ple.push_back(id - 1);
    }
    pack.Array("qwen4exp.ple.layers", std::move(ple));
    pack.Array("qwen4exp.ple.head_offsets", ngram.offsets);
    pack.Array("qwen4exp.ple.head_vocab_sizes", ngram.vocab);
    pack.Array("qwen4exp.ple.layer_multipliers", ngram.multipliers);

    auto result = pack.Finish();
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
