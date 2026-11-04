#include "src/models/qwen/native_metadata.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include "src/models/qwen/native_pack.hpp"

namespace gufo::models::qwen {

std::unique_ptr<core::GgufReader> OpenNativeMetadata(
    const std::filesystem::path& directory, std::string* error) {
  try {
    NativePackMetadata pack(directory, "qwen3_5_text");
    const auto& c = pack.text_config();
    const auto number = [&](const char* key) { return pack.Number(key); };
    if (c.member_str("hidden_act") != "silu" ||
        NativeBoolean(c, "attention_bias") ||
        NativeBoolean(c, "tie_word_embeddings") ||
        !NativeBoolean(c, "attn_output_gate"))
      throw std::runtime_error("unsupported native layer contract");
    const auto name = directory.filename().empty()
                          ? directory.parent_path().filename()
                          : directory.filename();
    pack.Text("general.architecture", "qwen35");
    pack.Text("general.name", name.string());
    const auto layers = number("num_hidden_layers");
    pack.U32("qwen35.block_count", layers);
    const std::pair<const char*, const char*> keys[] = {
        {"context_length", "max_position_embeddings"},
        {"embedding_length", "hidden_size"},
        {"feed_forward_length", "intermediate_size"},
        {"attention.head_count", "num_attention_heads"},
        {"attention.head_count_kv", "num_key_value_heads"},
        {"attention.key_length", "head_dim"},
        {"attention.value_length", "head_dim"},
        {"ssm.conv_kernel", "linear_conv_kernel_dim"},
        {"ssm.state_size", "linear_key_head_dim"},
        {"ssm.group_count", "linear_num_key_heads"},
        {"ssm.time_step_rank", "linear_num_value_heads"},
        {"full_attention_interval", "full_attention_interval"}};
    for (const auto& [dst, src] : keys)
      pack.U32(std::string("qwen35.") + dst, number(src));
    pack.U32("qwen35.ssm.inner_size", number("linear_value_head_dim") *
                                          number("linear_num_value_heads"));
    pack.F32("qwen35.attention.layer_norm_rms_epsilon",
             NativeMember(c, "rms_norm_eps").as_double());
    const auto& rope = NativeMember(c, "rope_parameters");
    if (!NativeBoolean(rope, "mrope_interleaved") ||
        rope.member_str("rope_type") != "default")
      throw std::runtime_error("unsupported native rope type");
    pack.F32("qwen35.rope.freq_base",
             NativeMember(rope, "rope_theta").as_double());
    const auto rotary =
        NativeMember(rope, "partial_rotary_factor").as_double() *
        static_cast<double>(number("head_dim"));
    pack.U32("qwen35.rope.dimension_count", NativeUInt(json::Value(rotary)));
    std::vector<std::uint64_t> sections;
    for (const auto& v : NativeMember(rope, "mrope_section").items())
      sections.push_back(NativeUInt(v));
    sections.push_back(0);
    pack.Array("qwen35.rope.dimension_sections", std::move(sections));
    const auto& types = NativeMember(c, "layer_types");
    if (types.size() != layers || layers > 1024)
      throw std::runtime_error("invalid native layer types");
    const auto interval = number("full_attention_interval");
    std::vector<std::uint64_t> recurrent;
    for (std::size_t i = 0; i < types.size(); ++i) {
      const auto& type = types.items()[i].str();
      const bool full = type == "full_attention";
      if (!full && type != "linear_attention")
        throw std::runtime_error("unknown native layer type");
      if (interval == 0 || full != ((i + 1) % interval == 0))
        throw std::runtime_error(
            "native layer types disagree with full_attention_interval");
      recurrent.push_back(full ? 0 : 1);
    }
    pack.Array("qwen35.attention.recurrent_layers", std::move(recurrent));
    return pack.Finish(number("bos_token_id"));
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return nullptr;
  }
}

}  // namespace gufo::models::qwen
