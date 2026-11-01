#include "src/models/qwen38_flash_next/exl3_weights.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {
namespace ex = quant::exl3;

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error("Flash-Next EXL3: " + message);
}

std::string Read(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    Fail("cannot read " + path.string());
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

std::string Digest(std::string_view text) {
  return crypto::Sha256Hex(
      {reinterpret_cast<const std::uint8_t*>(text.data()), text.size()});
}

std::filesystem::path Reference(const std::filesystem::path& path,
                                const core::GgufReader& reader,
                                std::string_view key) {
  const auto value = reader.GetMetadataString(key);
  if (!value || value->empty() || std::filesystem::path(*value).is_absolute())
    Fail("missing relative reference " + std::string(key));
  return (path.parent_path() / *value).lexically_normal();
}

void CheckDigest(const core::GgufReader& sidecar, std::string_view key,
                 std::string_view digest) {
  if (sidecar.GetMetadataString(key) != digest)
    Fail("source identity mismatch: " + std::string(key));
}

void CheckMetadata(const std::filesystem::path& directory,
                   const core::GgufReader& sidecar, const Config& c) {
  const auto config_bytes = Read(directory / "config.json");
  CheckDigest(sidecar, "gufo.exl3.config_sha256", Digest(config_bytes));
  const auto config = json::parse(config_bytes);
  const auto* text = config.find("text_config");
  if (text == nullptr)
    Fail("missing text_config");
  const std::pair<const char*, std::uint32_t> geometry[] = {
      {"hidden_size", c.hidden_size},
      {"num_hidden_layers", c.num_layers},
      {"num_attention_heads", c.num_heads},
      {"num_key_value_heads", c.num_kv_heads},
      {"head_dim", c.head_dim},
      {"hc_count", c.hc_count},
      {"hc_lowrank", c.hc_low_rank},
      {"full_attention_interval", c.full_attention_interval},
      {"linear_num_key_heads", c.ssm_num_k_heads},
      {"linear_num_value_heads", c.ssm_num_v_heads},
      {"linear_key_head_dim", c.ssm_head_dim},
      {"linear_value_head_dim", c.ssm_head_dim},
      {"linear_conv_kernel_dim", c.ssm_conv_kernel},
      {"num_experts", c.num_experts},
      {"num_experts_per_tok", c.num_experts_used},
      {"moe_intermediate_size", c.expert_ff},
      {"shared_expert_intermediate_size", c.shared_expert_ff},
      {"indexer_n_heads", c.indexer_heads},
      {"indexer_head_dim", c.indexer_head_dim},
      {"indexer_budget", c.indexer_top_k},
      {"indexer_compress_ratio", c.compress_ratio},
      {"max_position_embeddings", c.context_length},
      {"vocab_size", c.vocab_size}};
  for (const auto& [key, expected] : geometry)
    if (text->member_double(key, -1) != expected)
      Fail("config.json geometry mismatch: " + std::string(key));
  if (static_cast<float>(text->member_double("rms_norm_eps", -1)) != c.rms_eps)
    Fail("config.json rms_norm_eps mismatch");
  const auto* rope = text->find("rope_parameters");
  if (rope == nullptr ||
      rope->member_double("rope_theta", -1) != c.rope_theta ||
      rope->member_double("partial_rotary_factor", -1) * c.head_dim !=
          c.rotary_dim)
    Fail("config.json rotary geometry mismatch");
  const auto* sections = rope->find("mrope_section");
  if (sections == nullptr || sections->size() != 3)
    Fail("config.json mrope_section missing");
  for (std::size_t i = 0; i < 3; ++i)
    if (sections->items()[i].as_double() != c.rope_sections[i])
      Fail("config.json mrope_section mismatch");
  const auto tokenizer_bytes = Read(directory / "tokenizer.json");
  CheckDigest(sidecar, "gufo.exl3.tokenizer_sha256", Digest(tokenizer_bytes));
  const auto tokenizer = json::parse(tokenizer_bytes);
  const auto tokens = sidecar.GetMetadataStringArray("tokenizer.ggml.tokens");
  std::vector<bool> seen(tokens.size());
  const auto token = [&](std::size_t id, std::string_view value) {
    if (id >= tokens.size() || tokens[id] != value)
      Fail("tokenizer vocabulary mismatch at " + std::to_string(id));
    seen[id] = true;
  };
  const auto* model = tokenizer.find("model");
  if (model == nullptr || model->find("vocab") == nullptr)
    Fail("tokenizer vocabulary missing");
  for (const auto& [value, id] : model->find("vocab")->members())
    token(id.as_size(), value);
  if (const auto* added = tokenizer.find("added_tokens"))
    for (const auto& entry : added->items())
      token(entry.member_size("id"), entry.member_str("content"));
  const auto padding = std::find(seen.begin(), seen.end(), false);
  if (padding == seen.begin() ||
      std::find(padding, seen.end(), true) != seen.end())
    Fail("tokenizer vocabulary has holes before its padded tail");
  const auto merges = sidecar.GetMetadataStringArray("tokenizer.ggml.merges");
  const auto* hf_merges = model->find("merges");
  if (hf_merges == nullptr || hf_merges->size() != merges.size())
    Fail("tokenizer merges count mismatch");
  for (std::size_t i = 0; i < merges.size(); ++i) {
    const auto& value = hf_merges->items()[i];
    const auto merge = value.is_string() ? value.str()
                                         : value.items()[0].str() + " " +
                                               value.items()[1].str();
    if (merges[i] != merge)
      Fail("tokenizer merge mismatch at " + std::to_string(i));
  }
}

std::string HeadersDigest(
    const ex::Checkpoint& checkpoint,
    const core::GgufMappedRegion* supplementary = nullptr) {
  crypto::Sha256Hasher hash;
  auto regions = checkpoint.GetMappedRegions();
  if (supplementary != nullptr)
    regions.push_back(*supplementary);
  std::sort(regions.begin(), regions.end(), [](const auto& a, const auto& b) {
    return a.source_path.filename() < b.source_path.filename();
  });
  for (const auto& region : regions) {
    const auto label = region.source_path.filename().string() + '\0' +
                       std::to_string(region.size) + '\0';
    hash.Update(
        {reinterpret_cast<const std::uint8_t*>(label.data()), label.size()});
    std::uint64_t count;
    std::memcpy(&count, region.data, 8);
    hash.Update({static_cast<const std::uint8_t*>(region.data),
                 static_cast<std::size_t>(8 + count)});
  }
  return hash.FinishHex();
}

struct Binder {
  Exl3Storage& storage;
  const Config& c;

  std::string_view Name(const std::string& name) {
    storage.names.push_back(name);
    return storage.names.back();
  }

  TensorRef Plain(const std::string& name, std::uint32_t cols,
                  std::uint32_t rows = 1, bool f32 = false,
                  bool plus_one = false, bool neg_exp = false,
                  bool v_order = false, bool qkv_order = false) {
    const auto* t = storage.checkpoint->Find(name);
    const bool owned = t == nullptr && storage.supplementary.contains(name);
    if (owned)
      t = &storage.supplementary.at(name);
    if (t == nullptr || t->Elements() != std::int64_t{cols} * rows ||
        t->shape.empty() ||
        (rows > 1 &&
         (t->shape[0] != rows || (cols > 1 && t->shape.back() != cols))) ||
        (t->dtype != ex::DType::kF16 && t->dtype != ex::DType::kBF16 &&
         t->dtype != ex::DType::kF32))
      Fail("invalid unquantized operand " + name);
    TensorRef ref;
    ref.data = t->data;
    ref.cols = cols;
    ref.rows = rows;
    ref.type = t->dtype == ex::DType::kF16    ? core::GgmlType::kF16
               : t->dtype == ex::DType::kBF16 ? core::GgmlType::kBF16
                                              : core::GgmlType::kF32;
    ref.shard = t->shard;
    ref.file_offset = t->file_offset;
    ref.name = Name(name);
    ref.safetensors = !owned;
    ref.memory_owned = owned;
    if (!f32 && !v_order && !qkv_order)
      return ref;
    storage.converted.emplace_back(std::size_t{cols} * rows * 4);
    auto* out = storage.converted.back().data();
    const auto qk = 2 * c.SsmKeyDim();
    for (std::uint32_t r = 0; r < rows; ++r) {
      const auto dest =
          qkv_order && r >= qk
              ? qk + Exl3VHeadDestination(r - qk, c.ssm_head_dim, c)
          : v_order ? Exl3VHeadDestination(r, 1, c)
                    : r;
      for (std::uint32_t col = 0; col < cols; ++col) {
        const auto i = std::size_t{r} * cols + col;
        float value;
        if (t->dtype == ex::DType::kF32) {
          std::memcpy(&value, t->data + i * 4, 4);
        } else {
          std::uint16_t bits;
          std::memcpy(&bits, t->data + i * 2, 2);
          value = t->dtype == ex::DType::kBF16
                      ? std::bit_cast<float>(std::uint32_t{bits} << 16)
                      : static_cast<float>(ex::HalfToDouble(bits));
        }
        if (plus_one)
          value += 1.0F;
        if (neg_exp)
          value = -std::exp(value);
        std::memcpy(out + (std::size_t{dest} * cols + col) * 4, &value, 4);
      }
    }
    ref.data = out;
    ref.type = core::GgmlType::kF32;
    ref.memory_owned = true;
    return ref;
  }

  TensorRef Linear(const std::string& module, std::uint32_t k, std::uint32_t n,
                   std::uint32_t experts = 1, bool input_v = false,
                   bool output_v = false, bool qkv_v = false) {
    auto ref = BindExl3LinearBank(storage, module, k, n, experts);
    for (auto& host : storage.linears.back()) {
      const auto blocks = [&](std::uint32_t width, bool v, bool qkv) {
        std::vector<std::uint32_t> permutation;
        if (!v && !qkv)
          return permutation;
        permutation.resize(width / 128);
        const auto qk = 2 * c.SsmKeyDim();
        for (std::uint32_t i = 0; i < width; i += 128) {
          const auto dest = qkv && i >= qk ? qk + Exl3VHeadDestination(
                                                      i - qk, c.ssm_head_dim, c)
                            : v ? Exl3VHeadDestination(i, c.ssm_head_dim, c)
                                : i;
          permutation[i / 128] = dest / 128;
        }
        return permutation;
      };
      host.input_blocks = blocks(k, input_v, false);
      host.output_blocks = blocks(n, output_v, qkv_v);
    }
    return ref;
  }

  HcMixer Mixer(const std::string& prefix, bool inject) {
    HcMixer m;
    m.norm = Plain(prefix + "hc_norm.weight", c.HcDim(), 1, true, true);
    m.down = Plain(prefix + "input_mix_weight_down.weight", c.HcDim(),
                   c.hc_low_rank);
    m.up =
        Plain(prefix + "input_mix_weight_up.weight", c.hc_low_rank, c.HcDim());
    if (inject)
      m.inject = Plain(prefix + "block_inject_weight.weight", c.HcDim(),
                       c.hc_count, true);
    return m;
  }

  LayerWeights Layer(std::uint32_t index, bool mtp) {
    const auto p =
        mtp ? std::string("mtp.layers.0.")
            : "model.language_model.layers." + std::to_string(index) + ".";
    LayerWeights l;
    l.linear = !mtp && c.IsLinearLayer(index);
    l.hc_attn = Mixer(p + "attn_hyper_connection.", true);
    l.hc_ffn = Mixer(p + "mlp_hyper_connection.", true);
    if (l.linear) {
      const auto a = p + "linear_attn.";
      l.ssm_qkv = Linear(a + "in_proj_qkv", c.hidden_size, c.SsmConvChannels(),
                         1, false, false, true);
      l.ssm_gate = Linear(a + "in_proj_z", c.hidden_size, c.SsmValueDim(), 1,
                          false, true);
      l.ssm_out =
          Linear(a + "out_proj", c.SsmValueDim(), c.hidden_size, 1, true);
      l.ssm_conv1d =
          Plain(a + "conv1d.weight", c.ssm_conv_kernel, c.SsmConvChannels(),
                true, false, false, false, true);
      l.ssm_alpha = Plain(a + "in_proj_a.weight", c.hidden_size,
                          c.ssm_num_v_heads, true, false, false, true);
      l.ssm_beta = Plain(a + "in_proj_b.weight", c.hidden_size,
                         c.ssm_num_v_heads, true, false, false, true);
      // Vectors are treated as one scalar per output head for the reorder.
      l.ssm_a =
          Plain(a + "A_log", 1, c.ssm_num_v_heads, true, false, true, true);
      l.ssm_dt =
          Plain(a + "dt_bias", 1, c.ssm_num_v_heads, true, false, false, true);
      l.ssm_norm = Plain(a + "norm.weight", c.ssm_head_dim, 1, true);
    } else {
      const auto a = p + "self_attn.";
      l.attn_q = Linear(a + "q_proj", c.hidden_size, 2 * c.AttentionQDim());
      l.attn_k = Linear(a + "k_proj", c.hidden_size, c.AttentionKvDim());
      l.attn_v = Linear(a + "v_proj", c.hidden_size, c.AttentionKvDim());
      l.attn_out = Linear(a + "o_proj", c.AttentionQDim(), c.hidden_size);
      l.attn_q_norm = Plain(a + "q_norm.weight", c.head_dim, 1, true, true);
      l.attn_k_norm = Plain(a + "k_norm.weight", c.head_dim, 1, true, true);
      l.indexer_q = Linear(a + "indexer.index_qk_proj", c.hidden_size,
                           (c.indexer_heads + 1) * c.indexer_head_dim);
      l.indexer_k = l.indexer_q;
      l.indexer_q.rows = c.indexer_heads * c.indexer_head_dim;
      l.indexer_k.rows = c.indexer_head_dim;
      l.indexer_k.output_begin = l.indexer_q.rows;
      l.indexer_q_norm = Plain(a + "indexer.q_layernorm.weight",
                               c.indexer_head_dim, 1, true, true);
      l.indexer_k_norm = Plain(a + "indexer.k_layernorm.weight",
                               c.indexer_head_dim, 1, true, true);
    }
    if (!mtp && c.IsPleLayer(index)) {
      const auto a = p + "ple.";
      l.ple_key = Plain(a + "key_proj.weight", c.PleEmbeddingDim(), c.HcDim());
      l.ple_value =
          Plain(a + "value_proj.weight", c.PleEmbeddingDim(), c.hidden_size);
      l.ple_norm_key = Plain(a + "norm_key.weight", c.HcDim(), 1, true, true);
      l.ple_norm_query =
          Plain(a + "norm_query.weight", c.HcDim(), 1, true, true);
      l.ple_norm_conv = Plain(a + "norm_conv.weight", c.HcDim(), 1, true, true);
      l.ple_conv1d =
          Plain(a + "conv1d.weight", c.ple_conv_kernel, c.HcDim(), true);
    }
    l.router = Plain(p + "mlp.gate.weight", c.hidden_size, c.num_experts);
    l.shexp_gate_inp =
        Plain(p + "mlp.shared_expert_gate.weight", c.hidden_size);
    l.ffn_gate_exps = Linear(p + "mlp.experts.{}.gate_proj", c.hidden_size,
                             c.expert_ff, c.num_experts);
    l.ffn_up_exps = Linear(p + "mlp.experts.{}.up_proj", c.hidden_size,
                           c.expert_ff, c.num_experts);
    l.ffn_down_exps = Linear(p + "mlp.experts.{}.down_proj", c.expert_ff,
                             c.hidden_size, c.num_experts);
    l.shexp_gate = Linear(p + "mlp.shared_expert.gate_proj", c.hidden_size,
                          c.shared_expert_ff);
    l.shexp_up = Linear(p + "mlp.shared_expert.up_proj", c.hidden_size,
                        c.shared_expert_ff);
    l.shexp_down = Linear(p + "mlp.shared_expert.down_proj", c.shared_expert_ff,
                          c.hidden_size);
    if (mtp) {
      l.nextn_enorm = Plain("mtp.pre_fc_norm_embedding.weight", c.hidden_size,
                            1, true, true);
      l.nextn_hnorm =
          Plain("mtp.pre_fc_norm_hidden.weight", c.HcDim(), 1, true, true);
      l.nextn_fc_embedding =
          Linear("mtp.fc_embedding", c.hidden_size, c.hidden_size);
      l.nextn_fc_hidden = Linear("mtp.fc_hidden", c.hidden_size, c.hidden_size);
      l.nextn_head = Mixer("mtp.hyper_connection_mixer.", false);
    }
    return l;
  }
};

std::shared_ptr<Exl3Storage> Open(const std::filesystem::path& path,
                                  const core::GgufReader& sidecar, Config& c,
                                  bool mtp) {
  if (sidecar.GetTensorCount() != 0)
    Fail("EXL3 metadata sidecar must be tensorless");
  c.vocab_size = static_cast<std::uint32_t>(
      sidecar.GetMetadataStringArray("tokenizer.ggml.tokens").size());
  const auto directory = Reference(path, sidecar, kExl3Checkpoint);
  CheckMetadata(directory, sidecar, c);
  auto storage = std::make_shared<Exl3Storage>();
  storage->checkpoint =
      std::make_shared<ex::Checkpoint>(ex::Checkpoint::Open(directory));
  std::filesystem::path patch;
  if (mtp && sidecar.GetMetadataString(kExl3MtpMixer))
    patch = Reference(path, sidecar, kExl3MtpMixer);
  const auto digest = mtp ? PrepareExl3MtpMixer(*storage, c, patch)
                          : HeadersDigest(*storage->checkpoint);
  CheckDigest(sidecar,
              mtp ? "gufo.exl3.mtp_headers_sha256" : "gufo.exl3.headers_sha256",
              digest);
  return storage;
}

}  // namespace

TensorRef BindExl3LinearBank(Exl3Storage& storage, const std::string& module,
                             std::uint32_t k, std::uint32_t n,
                             std::uint32_t experts) {
  if (experts == 0 || (experts > 1 && module.find("{}") == std::string::npos))
    Fail("invalid expert bank " + module);
  // Commit storage only after the entire bank has validated.
  std::vector<Exl3HostLinear> bank;
  bank.reserve(experts);
  for (std::uint32_t e = 0; e < experts; ++e) {
    auto name = module;
    if (name.find("{}") != std::string::npos)
      name.replace(name.find("{}"), 2, std::to_string(e));
    const auto lin = storage.checkpoint->GetLinear(name);
    if (lin.in_features != k || lin.out_features != n || lin.rate.half ||
        lin.rate.bits < 3 || lin.rate.bits > 7 ||
        lin.codebook != ex::Codebook::kMul1 || lin.bias != nullptr)
      Fail("invalid projection geometry/rate/codebook " + name);
    if (!bank.empty() && lin.rate != bank.front().view.rate)
      Fail("mixed expert rates in bank " + module + " at " + name);
    const auto* trellis = storage.checkpoint->Find(name + ".trellis");
    const auto* suh = storage.checkpoint->Find(name + ".suh");
    const auto* svh = storage.checkpoint->Find(name + ".svh");
    if (suh->shard != trellis->shard || svh->shard != trellis->shard)
      Fail("projection quartet spans shards: " + name);
    Exl3HostLinear host;
    host.view = {lin.trellis, lin.suh, lin.svh, k, n, lin.rate, lin.codebook};
    host.shard = trellis->shard;
    host.trellis_offset = trellis->file_offset;
    host.suh_offset = suh->file_offset;
    host.svh_offset = svh->file_offset;
    bank.push_back(std::move(host));
  }
  storage.linears.push_back(std::move(bank));
  storage.names.push_back(module);
  TensorRef ref;
  ref.data = storage.linears.back().front().view.trellis;
  ref.cols = k;
  ref.rows = n;
  ref.experts = experts;
  ref.name = storage.names.back();
  ref.exl3 = storage.linears.back();
  return ref;
}

std::string PrepareExl3MtpMixer(Exl3Storage& storage, const Config& c,
                                const std::filesystem::path& patch) {
  const std::string prefix = "mtp.hyper_connection_mixer.";
  const auto width = static_cast<std::int64_t>(c.HcDim());
  const std::map<std::string, std::vector<std::int64_t>> shapes{
      {prefix + "hc_norm.weight", {width}},
      {prefix + "input_mix_weight_down.weight", {c.hc_low_rank, width}},
      {prefix + "input_mix_weight_up.weight", {width, c.hc_low_rank}}};
  std::vector<std::uint8_t> header;
  std::uint64_t file_size = 0;
  if (!patch.empty()) {
    std::ifstream in(patch, std::ios::binary | std::ios::ate);
    if (!in || in.tellg() < 8)
      Fail("cannot read MTP mixer patch " + patch.string());
    file_size = static_cast<std::uint64_t>(in.tellg());
    in.seekg(0);
    std::uint64_t length = 0;
    in.read(reinterpret_cast<char*>(&length), 8);
    if (length == 0 || length > (64U << 20) || length > file_size - 8)
      Fail("invalid MTP mixer patch header");
    header.resize(8 + length);
    std::memcpy(header.data(), &length, 8);
    in.read(reinterpret_cast<char*>(header.data() + 8), length);
    const auto entries = json::parse(std::string_view(
        reinterpret_cast<const char*>(header.data() + 8), length));
    if (!entries.is_object())
      Fail("MTP mixer patch header must be an object");
    for (const auto& [name, entry] : entries.members()) {
      if (name == "__metadata__")
        continue;
      const auto expected = shapes.find(name);
      if (expected == shapes.end())
        Fail("unexpected tensor in MTP mixer patch " + name);
      ex::Tensor tensor;
      tensor.dtype_name = entry.member_str("dtype");
      const unsigned element = tensor.dtype_name == "F32" ? 4 : 2;
      tensor.dtype = tensor.dtype_name == "F16"    ? ex::DType::kF16
                     : tensor.dtype_name == "BF16" ? ex::DType::kBF16
                     : tensor.dtype_name == "F32"  ? ex::DType::kF32
                                                   : ex::DType::kOther;
      const auto* dims = entry.find("shape");
      const auto* offsets = entry.find("data_offsets");
      if (tensor.dtype == ex::DType::kOther || dims == nullptr ||
          !dims->is_array() || dims->size() != expected->second.size() ||
          offsets == nullptr || !offsets->is_array() || offsets->size() != 2)
        Fail("invalid MTP mixer patch tensor " + name);
      std::uint64_t elements = 1;
      for (std::size_t i = 0; i < dims->size(); ++i) {
        if (dims->items()[i].as_double() != expected->second[i])
          Fail("MTP mixer patch geometry mismatch " + name);
        tensor.shape.push_back(expected->second[i]);
        elements *= expected->second[i];
      }
      const auto offset = [&](const json::Value& value) {
        const auto position =
            value.as_size(std::numeric_limits<std::size_t>::max());
        if (position == std::numeric_limits<std::size_t>::max())
          Fail("invalid MTP mixer patch offset " + name);
        return position;
      };
      const auto begin = offset(offsets->items()[0]);
      const auto end = offset(offsets->items()[1]);
      if (begin > end || end > file_size - 8 - length ||
          end - begin != elements * element)
        Fail("MTP mixer patch payload mismatch " + name);
      tensor.bytes = end - begin;
      std::vector<std::uint8_t> bytes(tensor.bytes);
      in.seekg(8 + length + begin);
      in.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
      if (!in)
        Fail("truncated MTP mixer patch tensor " + name);
      if (const auto* existing = storage.checkpoint->Find(name)) {
        if (existing->dtype != tensor.dtype ||
            existing->shape != tensor.shape ||
            existing->bytes != tensor.bytes ||
            std::memcmp(existing->data, bytes.data(), tensor.bytes) != 0)
          Fail("in-shard MTP mixer differs from patch " + name);
      } else {
        storage.converted.push_back(std::move(bytes));
        tensor.data = storage.converted.back().data();
        storage.supplementary.insert_or_assign(name, std::move(tensor));
      }
    }
  }
  for (const auto& [name, shape] : shapes) {
    const auto* tensor = storage.checkpoint->Find(name);
    if (tensor == nullptr && storage.supplementary.contains(name))
      tensor = &storage.supplementary.at(name);
    if (tensor == nullptr || tensor->shape != shape ||
        (tensor->dtype != ex::DType::kF16 &&
         tensor->dtype != ex::DType::kBF16 && tensor->dtype != ex::DType::kF32))
      Fail("missing/invalid MTP mixer operand " + name);
  }
  if (patch.empty())
    return HeadersDigest(*storage.checkpoint);
  const core::GgufMappedRegion region{header.data(), file_size, {}, patch};
  return HeadersDigest(*storage.checkpoint, &region);
}

std::string Exl3PleHeadersDigest(const core::GgufReader& reader) {
  crypto::Sha256Hasher hash;
  for (const auto& region : reader.GetMappedRegions()) {
    const auto* base = static_cast<const std::uint8_t*>(region.data);
    const auto label = region.source_path.filename().string() + '\0' +
                       std::to_string(region.size) + '\0';
    hash.Update(
        {reinterpret_cast<const std::uint8_t*>(label.data()), label.size()});
    hash.Update({base, region.data_offset});
  }
  return hash.FinishHex();
}

std::uint32_t Exl3VHeadDestination(std::uint32_t index, std::uint32_t dim,
                                   const Config& c) {
  const auto head = index / dim;
  const auto group = c.ssm_num_v_heads / c.ssm_num_k_heads;
  return ((head % group) * c.ssm_num_k_heads + head / group) * dim +
         index % dim;
}

std::optional<ModelWeights> LoadExl3Model(const std::filesystem::path& path,
                                          const core::GgufReader& sidecar,
                                          std::string* error) {
  try {
    ModelWeights w;
    auto config = Config::FromGguf(sidecar, true, error);
    if (!config)
      return std::nullopt;
    w.config = *config;
    w.exl3_storage = Open(path, sidecar, w.config, false);
    auto& storage = *w.exl3_storage;
    Binder b{storage, w.config};
    w.token_embd = b.Plain("model.language_model.embed_tokens.weight",
                           w.config.hidden_size, w.config.vocab_size);
    if (w.token_embd.type != core::GgmlType::kBF16)
      Fail("embedding must remain BF16");
    w.output = b.Linear("lm_head", w.config.hidden_size, w.config.vocab_size);
    w.hc_head = b.Mixer("model.language_model.hyper_connection_mixer.", false);
    if (w.config.ple_layer >= 0) {
      storage.ple_reader = core::GgufReader::OpenFile(
          Reference(path, sidecar, kExl3PleSource), error);
      if (!storage.ple_reader)
        return std::nullopt;
      CheckDigest(sidecar, "gufo.exl3.ple_headers_sha256",
                  Exl3PleHeadersDigest(*storage.ple_reader));
      auto source = ModelWeights::Bind(*storage.ple_reader, error);
      if (!source)
        return std::nullopt;
      const auto& a = source->config;
      const auto& c = w.config;
      auto compatible = a;
      compatible.nextn_layers = 1;
      if (!compatible.MtpMatches(c) || a.ple_layer != c.ple_layer ||
          a.ple_rows != c.ple_rows || a.ple_multipliers != c.ple_multipliers ||
          a.ple_head_offsets != c.ple_head_offsets ||
          a.ple_head_vocab != c.ple_head_vocab ||
          a.ple_eos_token != c.ple_eos_token ||
          a.ple_head_dim != c.ple_head_dim ||
          source->ple_table.type != core::GgmlType::kIQ4_NL)
        Fail("retained IQ4_NL PLE source metadata mismatch");
      for (const auto key :
           {"tokenizer.ggml.tokens", "tokenizer.ggml.merges",
            "tokenizer.chat_template", "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id", "tokenizer.ggml.pre",
            "tokenizer.ggml.token_type"}) {
        const auto* expected = sidecar.FindMetadata(key);
        const auto* actual = storage.ple_reader->FindMetadata(key);
        if ((expected == nullptr) != (actual == nullptr) ||
            (expected && (expected->type != actual->type ||
                          expected->value != actual->value)))
          Fail("retained source tokenizer/template mismatch: " +
               std::string(key));
      }
      w.ple_table = source->ple_table;
    }
    for (std::uint32_t i = 0; i < w.config.num_layers; ++i)
      w.layers.push_back(b.Layer(i, false));
    return w;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return std::nullopt;
  }
}

std::optional<MtpWeights> LoadExl3Mtp(const std::filesystem::path& path,
                                      const core::GgufReader& sidecar,
                                      const ModelWeights& trunk,
                                      std::string* error) {
  try {
    MtpWeights w;
    auto config = Config::FromGguf(sidecar, false, error);
    if (!config)
      return std::nullopt;
    if (!config->MtpMatches(trunk.config) || !trunk.exl3_storage)
      Fail("MTP EXL3 metadata does not match trunk");
    w.config = *config;
    w.exl3_storage = Open(path, sidecar, w.config, true);
    CheckDigest(sidecar, "gufo.exl3.headers_sha256",
                HeadersDigest(*trunk.exl3_storage->checkpoint));
    Binder b{*w.exl3_storage, w.config};
    w.block = b.Layer(0, true);
    return w;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return std::nullopt;
  }
}

void Exl3MatVec(const Exl3HostLinear& host, std::span<const float> x,
                std::span<float> y, bool decode_storage) {
  const auto& l = host.view;
  if (x.size() != l.k || y.size() != l.n)
    Fail("CPU projection shape mismatch");
  const auto destination = [](std::size_t i, const auto& blocks) {
    return blocks.empty() ? i : blocks[i / 128] * 128 + i % 128;
  };
  std::vector<double> rotated(l.k), inner_y(l.n);
  if (decode_storage) {
    for (std::size_t base = 0; base < l.k; base += 128) {
      float segment[128];
      for (std::size_t i = 0; i < 128; ++i) {
        // The input producer rounds each product and butterfly to FP32
        // before storing its normalized Hadamard result as FP16.
        volatile float product =
            x[destination(base + i, host.input_blocks)] *
            static_cast<float>(ex::HalfToDouble(l.suh[base + i]));
        segment[i] = product;
      }
      for (unsigned length = 1; length < 128; length *= 2)
        for (unsigned first = 0; first < 128; first += 2 * length)
          for (unsigned i = first; i < first + length; ++i) {
            const float a = segment[i], b = segment[i + length];
            segment[i] = a + b;
            segment[i + length] = a - b;
          }
      for (unsigned i = 0; i < 128; ++i) {
        volatile float scaled = segment[i] * 0.0883883461356163F;
        rotated[base + i] = ex::HalfToDouble(ex::HalfFromDouble(scaled));
      }
    }
  } else {
    for (std::size_t i = 0; i < l.k; ++i)
      rotated[i] =
          x[destination(i, host.input_blocks)] * ex::HalfToDouble(l.suh[i]);
    ex::HadamardBlocks(rotated);
  }
  std::vector<std::uint16_t> tile(l.k * 128);
  for (std::size_t n0 = 0; n0 < l.n; n0 += 128) {
    ex::DequantInnerColumns(l.trellis, l.k, l.n, l.rate, l.codebook, n0, 128,
                            tile.data());
    for (std::size_t j = 0; j < 128; ++j) {
      double sum = 0;
      for (std::size_t i = 0; i < l.k; ++i)
        sum += rotated[i] * ex::HalfToDouble(tile[i * 128 + j]);
      inner_y[n0 + j] = sum;
    }
  }
  ex::HadamardBlocks(inner_y);
  for (std::size_t j = 0; j < l.n; ++j)
    y[destination(j, host.output_blocks)] =
        static_cast<float>(inner_y[j] * ex::HalfToDouble(l.svh[j]));
}

}  // namespace gufo::models::qwen38_flash_next
