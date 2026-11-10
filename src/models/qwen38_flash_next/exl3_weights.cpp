#include "src/models/qwen38_flash_next/exl3_weights.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>

#include "src/core/json.hpp"
#include "src/core/quant/exl3/coverage.hpp"
#include "src/core/quant/exl3/layout.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {
namespace ex = quant::exl3;

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error("Flash-Next EXL3: " + message);
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
        float value = t->Value(i);
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
    if (storage.hc_q8) {
      const auto calibrated = [&](const TensorRef& original) {
        const auto* tensor = storage.hc_q8->Find(original.name);
        if (!tensor)
          Fail("missing calibrated HC " + std::string(original.name));
        auto ref = original;
        ref.data = tensor->data;
        ref.type = core::GgmlType::kQ8_0;
        ref.safetensors = false;
        // A separate immutable mapping owns this buffer; it is not a shard
        // offset inside the original model's upload stager.
        ref.memory_owned = true;
        return ref;
      };
      m.down_q8 = calibrated(m.down);
      m.up_q8 = calibrated(m.up);
    }
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
                                  const core::GgufReader& metadata, Config& c,
                                  bool mtp) {
  if (!metadata.GetMetadataBool(quant::exl3::kNativeKey).value_or(false))
    Fail("EXL3 metadata sidecars are not supported; pass the pack directory");
  c.vocab_size = static_cast<std::uint32_t>(
      metadata.GetMetadataStringArray("tokenizer.ggml.tokens").size());
  auto storage = std::make_shared<Exl3Storage>();
  storage->checkpoint =
      std::make_shared<ex::Checkpoint>(ex::Checkpoint::Open(path));
  if (mtp) {
    std::filesystem::path patch =
        path / "mtp_hyper_connection_mixer_patch.safetensors";
    if (!std::filesystem::is_regular_file(patch))
      patch.clear();
    PrepareExl3MtpMixer(*storage, c, patch);
  }
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
    // Every format of the checkpoint (ex::Supports) has fast kernels.
    if (lin.in_features != k || lin.out_features != n ||
        !ex::Supports(lin.codebook, lin.rate) || lin.bias != nullptr)
      Fail("invalid projection geometry/rate/codebook " + name);
    if (!bank.empty() && (lin.rate != bank.front().view.rate ||
                          lin.codebook != bank.front().view.codebook))
      Fail("mixed expert formats in bank " + module + " at " + name);
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

void PrepareExl3MtpMixer(Exl3Storage& storage, const Config& c,
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
}

std::uint32_t Exl3VHeadDestination(std::uint32_t index, std::uint32_t dim,
                                   const Config& c) {
  return static_cast<std::uint32_t>(
      ex::TiledVHeadIndex(index, dim, c.ssm_num_k_heads, c.ssm_num_v_heads));
}

void ApplyExl3HcMode(Exl3Storage& storage, const Config& config,
                     const std::filesystem::path& model, HcMode mode) {
  storage.hc_q8.reset();
  if (mode != HcMode::kQ8Calibrated)
    return;
  // Never fall back to another mode: an absent or invalid sidecar fails.
  try {
    PrepareExl3HcQ8(storage, config, model);
  } catch (const std::exception& error) {
    storage.hc_q8.reset();
    throw std::runtime_error(std::string("--hc-mode q8-calibrated: ") +
                             error.what());
  }
  if (!storage.hc_q8)
    Fail(
        "--hc-mode q8-calibrated requires the calibrated sidecar hc_q8/ and "
        "hc_q8.json in " +
        model.string());
}

namespace {
// One startup line per load naming the projections no fast kernel takes.
void LogKernelCoverage(std::string_view model, const Exl3Storage& storage) {
  std::vector<const ex::DeviceLinear*> views;
  for (const auto& bank : storage.linears)
    for (const auto& host : bank)
      views.push_back(&host.view);
  ex::LogReferencePath(model, views);
}
}  // namespace

std::optional<ModelWeights> LoadExl3Model(const std::filesystem::path& path,
                                          const core::GgufReader& metadata,
                                          HcMode hc_mode, std::string* error) {
  try {
    ModelWeights w;
    auto config = Config::FromGguf(metadata, true, error);
    if (!config)
      return std::nullopt;
    w.config = *config;
    w.exl3_storage = Open(path, metadata, w.config, false);
    auto& storage = *w.exl3_storage;
    ApplyExl3HcMode(storage, w.config, path, hc_mode);
    w.hc_mode = hc_mode;
    Binder b{storage, w.config};
    w.token_embd = b.Plain("model.language_model.embed_tokens.weight",
                           w.config.hidden_size, w.config.vocab_size);
    if (w.token_embd.type != core::GgmlType::kBF16)
      Fail("embedding must remain BF16");
    w.output = b.Linear("lm_head", w.config.hidden_size, w.config.vocab_size);
    w.hc_head = b.Mixer("model.language_model.hyper_connection_mixer.", false);
    for (std::uint32_t i = 0; i < w.config.num_layers; ++i)
      w.layers.push_back(b.Layer(i, false));
    LogKernelCoverage("qwen38_flash_next", storage);
    return w;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return std::nullopt;
  }
}

std::optional<MtpWeights> LoadExl3Mtp(const std::filesystem::path& path,
                                      const core::GgufReader& metadata,
                                      const ModelWeights& trunk,
                                      std::string* error) {
  try {
    MtpWeights w;
    auto config = Config::FromGguf(metadata, false, error);
    if (!config)
      return std::nullopt;
    if (!config->MtpMatches(trunk.config) || !trunk.exl3_storage)
      Fail("MTP EXL3 metadata does not match trunk");
    w.config = *config;
    w.exl3_storage = Open(path, metadata, w.config, true);
    const auto source = trunk.exl3_storage->checkpoint->GetMappedRegions();
    const auto draft = w.exl3_storage->checkpoint->GetMappedRegions();
    if (source.size() != draft.size())
      Fail("native MTP must use the trunk pack");
    for (std::size_t i = 0; i < source.size(); ++i)
      if (!std::filesystem::equivalent(source[i].source_path,
                                       draft[i].source_path))
        Fail("native MTP must use the trunk pack");
    w.exl3_storage->hc_q8 = trunk.exl3_storage->hc_q8;
    Binder b{*w.exl3_storage, w.config};
    w.block = b.Layer(0, true);
    LogKernelCoverage("qwen38_flash_next_mtp", *w.exl3_storage);
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
