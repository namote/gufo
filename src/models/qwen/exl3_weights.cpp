#include "src/models/qwen/exl3_weights.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "src/core/json.hpp"
#include "src/core/quant/exl3/format.hpp"
#include "src/core/quant/exl3/layout.hpp"

namespace gufo::models::qwen_exl3 {
namespace {

namespace exl3 = quant::exl3;

// How a dimension is permuted from Hugging Face order to GGUF order.
enum class Reorder : std::uint8_t {
  kNone,
  kVHeads,   // whole dimension: V heads of head_v_dim, grouped -> tiled
  kVScalar,  // whole dimension: one value per V head
  kQkvV,     // only the V part after q and k (2 * nk * hk entries)
  kConvV,    // conv channels: V part after q and k channels
};

enum class Kind : std::uint8_t {
  kLinear,       // EXL3 projection -> BF16 [out][in]
  kBf16Copy,     // BF16 tensor copied as is
  kF16ToBf16,    // unquantized F16 matrix [out][in] -> BF16
  kNormPlusOne,  // BF16 norm -> F32 w + 1
  kToF32,        // BF16 vector -> F32
  kNegExp,       // A_log -> F32 -exp(w)
  kConv,         // conv1d [ch, 1, kernel] -> F32 [ch][kernel]
};

struct Recipe {
  Kind kind = Kind::kLinear;
  std::string source;                  // module (kLinear) or tensor name
  Reorder out_order = Reorder::kNone;  // rows (output features)
  Reorder in_order = Reorder::kNone;   // columns (input features)
};

std::uint32_t F32Bits(float f) {
  return std::bit_cast<std::uint32_t>(f);
}

// Destination index of Hugging Face index `index` under `order` for a
// dimension of `size` entries.
std::size_t Destination(Reorder order, std::size_t index, std::size_t size,
                        const Geometry& g) {
  if (order == Reorder::kNone) {
    return index;
  }
  const auto nk = static_cast<std::size_t>(g.num_k_heads);
  auto tiled = [&](std::size_t i, std::size_t head_dim) {
    return exl3::TiledVHeadIndex(i, head_dim, nk,
                                 static_cast<std::size_t>(g.num_v_heads));
  };
  const auto qk = 2 * nk * static_cast<std::size_t>(g.head_k_dim);
  switch (order) {
    case Reorder::kNone:
      return index;
    case Reorder::kVHeads:
      return tiled(index, static_cast<std::size_t>(g.head_v_dim));
    case Reorder::kVScalar:
      return tiled(index, 1);
    case Reorder::kQkvV:
    case Reorder::kConvV:
      if (index < qk || size <= qk) {
        return index;
      }
      return qk + tiled(index - qk, static_cast<std::size_t>(g.head_v_dim));
  }
  return index;
}

std::optional<Recipe> RecipeFor(std::string_view name, const Geometry& g,
                                const exl3::Checkpoint& checkpoint) {
  const std::string lm = "model.language_model.";
  if (name == "token_embd.weight") {
    return Recipe{Kind::kBf16Copy, lm + "embed_tokens.weight"};
  }
  if (name == "output_norm.weight") {
    return Recipe{Kind::kNormPlusOne, lm + "norm.weight"};
  }
  if (name == "output.weight") {
    return Recipe{Kind::kLinear, "lm_head"};
  }
  if (!name.starts_with("blk.")) {
    return std::nullopt;
  }
  const auto dot = name.find('.', 4);
  if (dot == std::string_view::npos) {
    return std::nullopt;
  }
  const int layer = std::stoi(std::string(name.substr(4, dot - 4)));
  const std::string rest(name.substr(dot + 1));
  const bool mtp = layer == g.num_layers;
  if (layer < 0 || layer > g.num_layers) {
    return std::nullopt;
  }
  const std::string p = mtp ? std::string("mtp.layers.0.")
                            : lm + "layers." + std::to_string(layer) + ".";
  if (mtp) {
    if (rest == "nextn.eh_proj.weight")
      return Recipe{Kind::kLinear, "mtp.fc"};
    if (rest == "nextn.enorm.weight")
      return Recipe{Kind::kNormPlusOne, "mtp.pre_fc_norm_embedding.weight"};
    if (rest == "nextn.hnorm.weight")
      return Recipe{Kind::kNormPlusOne, "mtp.pre_fc_norm_hidden.weight"};
    if (rest == "nextn.shared_head_norm.weight")
      return Recipe{Kind::kNormPlusOne, "mtp.norm.weight"};
  }
  if (rest == "attn_norm.weight")
    return Recipe{Kind::kNormPlusOne, p + "input_layernorm.weight"};
  if (rest == "post_attention_norm.weight")
    return Recipe{Kind::kNormPlusOne, p + "post_attention_layernorm.weight"};
  if (rest == "ffn_gate.weight")
    return Recipe{Kind::kLinear, p + "mlp.gate_proj"};
  if (rest == "ffn_up.weight")
    return Recipe{Kind::kLinear, p + "mlp.up_proj"};
  if (rest == "ffn_down.weight")
    return Recipe{Kind::kLinear, p + "mlp.down_proj"};
  // Full attention.
  if (rest == "attn_q.weight")
    return Recipe{Kind::kLinear, p + "self_attn.q_proj"};
  if (rest == "attn_k.weight")
    return Recipe{Kind::kLinear, p + "self_attn.k_proj"};
  if (rest == "attn_v.weight")
    return Recipe{Kind::kLinear, p + "self_attn.v_proj"};
  if (rest == "attn_output.weight")
    return Recipe{Kind::kLinear, p + "self_attn.o_proj"};
  if (rest == "attn_q_norm.weight")
    return Recipe{Kind::kNormPlusOne, p + "self_attn.q_norm.weight"};
  if (rest == "attn_k_norm.weight")
    return Recipe{Kind::kNormPlusOne, p + "self_attn.k_norm.weight"};
  // Linear attention (gated delta net).
  const std::string la = p + "linear_attn.";
  if (rest == "attn_qkv.weight")
    return Recipe{Kind::kLinear, la + "in_proj_qkv", Reorder::kQkvV};
  if (rest == "attn_gate.weight")
    return Recipe{Kind::kLinear, la + "in_proj_z", Reorder::kVHeads};
  if (rest == "ssm_out.weight")
    return Recipe{Kind::kLinear, la + "out_proj", Reorder::kNone,
                  Reorder::kVHeads};
  if (rest == "ssm_alpha.weight")
    return Recipe{Kind::kF16ToBf16, la + "in_proj_a.weight", Reorder::kVScalar};
  if (rest == "ssm_beta.weight")
    return Recipe{Kind::kF16ToBf16, la + "in_proj_b.weight", Reorder::kVScalar};
  if (rest == "ssm_a")
    return Recipe{Kind::kNegExp, la + "A_log", Reorder::kVScalar};
  if (rest == "ssm_dt.bias")
    return Recipe{Kind::kToF32, la + "dt_bias", Reorder::kVScalar};
  if (rest == "ssm_conv1d.weight")
    return Recipe{Kind::kConv, la + "conv1d.weight", Reorder::kConvV};
  if (rest == "ssm_norm.weight")
    return Recipe{Kind::kToF32, la + "norm.weight"};
  (void)checkpoint;
  return std::nullopt;
}

// Effective BF16 weight of an EXL3 linear, written [out][in] with reorders.
void ProduceLinear(const exl3::Linear& lin, const Geometry& g,
                   Reorder out_order, Reorder in_order, std::uint16_t* out,
                   int threads) {
  const std::size_t k = lin.in_features;
  const std::size_t n = lin.out_features;
  constexpr std::size_t kChunk = 512;  // columns per work item (x 128)
  const std::size_t chunks = (n + kChunk - 1) / kChunk;
  std::vector<std::size_t> in_dest(k), out_dest(n);
  for (std::size_t i = 0; i < k; ++i)
    in_dest[i] = Destination(in_order, i, k, g);
  for (std::size_t j = 0; j < n; ++j)
    out_dest[j] = Destination(out_order, j, n, g);
  std::vector<double> suh(k), svh(n);
  for (std::size_t i = 0; i < k; ++i)
    suh[i] = exl3::HalfToDouble(lin.suh[i]);
  for (std::size_t j = 0; j < n; ++j)
    svh[j] = exl3::HalfToDouble(lin.svh[j]);

  std::atomic<std::size_t> next{0};
  auto work = [&] {
    std::vector<std::uint16_t> inner;
    std::vector<double> block;
    std::vector<double> column(k);
    for (std::size_t c = next++; c < chunks; c = next++) {
      const std::size_t n0 = c * kChunk;
      const std::size_t w = std::min(kChunk, n - n0);
      inner.resize(k * w);
      exl3::DequantInnerColumns(lin.trellis, k, n, lin.rate, lin.codebook, n0,
                                w, inner.data());
      block.assign(k * w, 0.0);
      for (std::size_t i = 0; i < k * w; ++i)
        block[i] = exl3::HalfToDouble(inner[i]);
      // H_k along each column, then scale rows by suh.
      for (std::size_t j = 0; j < w; ++j) {
        for (std::size_t i = 0; i < k; ++i)
          column[i] = block[i * w + j];
        exl3::HadamardBlocks(std::span<double>(column));
        for (std::size_t i = 0; i < k; ++i)
          block[i * w + j] = column[i] * suh[i];
      }
      // H_n within each 128-column block of every row, then scale by svh.
      for (std::size_t i = 0; i < k; ++i) {
        std::span<double> row(block.data() + i * w, w);
        exl3::HadamardBlocks(row);
        for (std::size_t j = 0; j < w; ++j) {
          out[out_dest[n0 + j] * k + in_dest[i]] =
              FloatToBf16(static_cast<float>(row[j] * svh[n0 + j]));
        }
      }
    }
  };
  const int count = std::max(1, threads);
  std::vector<std::thread> pool;
  for (int t = 1; t < count; ++t)
    pool.emplace_back(work);
  work();
  for (auto& t : pool)
    t.join();
}

}  // namespace

void EffectiveBf16(const quant::exl3::DeviceLinear& linear, std::uint16_t* out,
                   int threads) {
  exl3::Linear lin;
  lin.in_features = linear.k;
  lin.out_features = linear.n;
  lin.rate = linear.rate;
  lin.codebook = linear.codebook;
  lin.trellis = linear.trellis;
  lin.suh = linear.suh;
  lin.svh = linear.svh;
  ProduceLinear(lin, Geometry{}, Reorder::kNone, Reorder::kNone, out, threads);
}

namespace {

std::size_t ElementBytes(core::GgmlType type) {
  return type == core::GgmlType::kF32 ? 4 : 2;
}

}  // namespace

std::uint16_t FloatToBf16(float value) noexcept {
  const std::uint32_t bits = F32Bits(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return static_cast<std::uint16_t>((bits >> 16) | 0x40u);
  }
  const std::uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<std::uint16_t>(rounded >> 16);
}

float Bf16ToFloat(std::uint16_t value) noexcept {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16);
}

std::optional<Geometry> GeometryFromGguf(const core::GgufReader& reader,
                                         std::string* error) {
  const auto get = [&](const char* key) -> std::optional<std::uint32_t> {
    auto v = reader.GetMetadataUint32(key);
    if (!v && error != nullptr)
      *error = std::string("missing ") + key;
    return v;
  };
  const auto blocks = get("qwen35.block_count");
  const auto nextn = reader.GetMetadataUint32("qwen35.nextn_predict_layers");
  const auto groups = get("qwen35.ssm.group_count");
  const auto rank = get("qwen35.ssm.time_step_rank");
  const auto state = get("qwen35.ssm.state_size");
  const auto inner = get("qwen35.ssm.inner_size");
  if (!blocks || !groups || !rank || !state || !inner || *rank == 0 ||
      *groups == 0 || *rank % *groups != 0) {
    if (error != nullptr && error->empty())
      *error = "invalid ssm geometry";
    return std::nullopt;
  }
  Geometry g;
  g.num_layers = static_cast<int>(*blocks - nextn.value_or(0));
  g.num_k_heads = static_cast<int>(*groups);
  g.num_v_heads = static_cast<int>(*rank);
  g.head_k_dim = static_cast<int>(*state);
  g.head_v_dim = static_cast<int>(*inner / *rank);
  return g;
}

std::optional<TensorPlan> Source::Plan(std::string_view gguf_name,
                                       std::string* error) const {
  const auto recipe = RecipeFor(gguf_name, geometry_, checkpoint_);
  auto fail = [&](const std::string& why) -> std::optional<TensorPlan> {
    if (error != nullptr)
      *error = std::string(gguf_name) + ": " + why;
    return std::nullopt;
  };
  if (!recipe)
    return fail("no EXL3 source mapping");
  TensorPlan plan;
  plan.gguf_name = std::string(gguf_name);
  if (recipe->kind == Kind::kLinear) {
    if (!checkpoint_.IsQuantized(recipe->source))
      return fail("no EXL3 linear " + recipe->source);
    const auto lin = checkpoint_.GetLinear(recipe->source);
    plan.type = core::GgmlType::kBF16;
    plan.ne = {lin.in_features, lin.out_features};
  } else {
    const auto* t = checkpoint_.Find(recipe->source);
    if (t == nullptr)
      return fail("missing tensor " + recipe->source);
    switch (recipe->kind) {
      case Kind::kBf16Copy:
      case Kind::kF16ToBf16:
        if (t->shape.size() != 2)
          return fail("expected a matrix");
        plan.type = core::GgmlType::kBF16;
        plan.ne = {static_cast<std::uint64_t>(t->shape[1]),
                   static_cast<std::uint64_t>(t->shape[0])};
        break;
      case Kind::kConv:
        if (t->shape.size() != 3 || t->shape[1] != 1)
          return fail("conv shape");
        plan.type = core::GgmlType::kF32;
        plan.ne = {static_cast<std::uint64_t>(t->shape[2]),
                   static_cast<std::uint64_t>(t->shape[0])};
        break;
      default:
        if (t->shape.size() != 1)
          return fail("expected a vector");
        plan.type = core::GgmlType::kF32;
        plan.ne = {static_cast<std::uint64_t>(t->shape[0])};
        break;
    }
    if (recipe->kind == Kind::kBf16Copy &&
        t->dtype != quant::exl3::DType::kBF16)
      return fail("expected BF16");
  }
  std::size_t elements = 1;
  for (const auto d : plan.ne)
    elements *= d;
  plan.bytes = elements * ElementBytes(plan.type);
  return plan;
}

void Source::Produce(const TensorPlan& plan, std::span<std::uint8_t> out,
                     int threads) const {
  if (out.size() != plan.bytes) {
    throw std::invalid_argument(plan.gguf_name + ": output size");
  }
  const auto recipe = RecipeFor(plan.gguf_name, geometry_, checkpoint_);
  const auto& g = geometry_;
  if (recipe->kind == Kind::kLinear) {
    const auto lin = checkpoint_.GetLinear(recipe->source);
    ProduceLinear(lin, g, recipe->out_order, recipe->in_order,
                  reinterpret_cast<std::uint16_t*>(out.data()), threads);
    return;
  }
  const auto& t = *checkpoint_.Find(recipe->source);
  auto* f32 = reinterpret_cast<float*>(out.data());
  auto* u16 = reinterpret_cast<std::uint16_t*>(out.data());
  switch (recipe->kind) {
    case Kind::kBf16Copy:
      std::memcpy(out.data(), t.data, plan.bytes);
      break;
    case Kind::kF16ToBf16: {
      const auto rows = static_cast<std::size_t>(t.shape[0]);
      const auto cols = static_cast<std::size_t>(t.shape[1]);
      for (std::size_t r = 0; r < rows; ++r) {
        const auto d = Destination(recipe->out_order, r, rows, g);
        for (std::size_t c = 0; c < cols; ++c) {
          u16[d * cols + c] = FloatToBf16(t.Value(r * cols + c));
        }
      }
      break;
    }
    case Kind::kNormPlusOne:
      for (std::size_t i = 0; i < plan.ne[0]; ++i) {
        f32[i] = t.Value(i) + 1.0f;
      }
      break;
    case Kind::kToF32: {
      const auto size = static_cast<std::size_t>(plan.ne[0]);
      for (std::size_t i = 0; i < size; ++i) {
        f32[Destination(recipe->out_order, i, size, g)] = t.Value(i);
      }
      break;
    }
    case Kind::kNegExp: {
      const auto size = static_cast<std::size_t>(plan.ne[0]);
      for (std::size_t i = 0; i < size; ++i) {
        f32[Destination(recipe->out_order, i, size, g)] = -std::exp(t.Value(i));
      }
      break;
    }
    case Kind::kConv: {
      const auto channels = static_cast<std::size_t>(t.shape[0]);
      const auto kernel = static_cast<std::size_t>(t.shape[2]);
      for (std::size_t c = 0; c < channels; ++c) {
        const auto d = Destination(recipe->out_order, c, channels, g);
        for (std::size_t j = 0; j < kernel; ++j) {
          f32[d * kernel + j] = t.Value(c * kernel + j);
        }
      }
      break;
    }
    case Kind::kLinear:
      break;
  }
}

// ---------------------------------------------------------------- X2a

std::vector<RawKv> ReadRawMetadata(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot open " + path.string());
  auto read = [&](void* p, std::size_t n) {
    if (!in.read(static_cast<char*>(p), static_cast<std::streamsize>(n)))
      throw std::runtime_error("truncated GGUF " + path.string());
  };
  char magic[4];
  read(magic, 4);
  if (std::memcmp(magic, "GGUF", 4) != 0)
    throw std::runtime_error(path.string() + " is not GGUF");
  std::uint32_t version;
  std::uint64_t tensors, kvs;
  read(&version, 4);
  read(&tensors, 8);
  read(&kvs, 8);
  static const int kSizes[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
  std::function<void(std::uint32_t)> skip = [&](std::uint32_t type) {
    if (type == 8) {
      std::uint64_t n;
      read(&n, 8);
      in.seekg(static_cast<std::streamoff>(n), std::ios::cur);
    } else if (type == 9) {
      std::uint32_t element;
      std::uint64_t n;
      read(&element, 4);
      read(&n, 8);
      if (element == 8 || element == 9) {
        for (std::uint64_t i = 0; i < n; ++i)
          skip(element);
      } else if (element < 13) {
        in.seekg(static_cast<std::streamoff>(n * kSizes[element]),
                 std::ios::cur);
      } else {
        throw std::runtime_error("unknown GGUF array type");
      }
    } else if (type < 13) {
      in.seekg(kSizes[type], std::ios::cur);
    } else {
      throw std::runtime_error("unknown GGUF value type");
    }
  };
  std::vector<RawKv> out;
  for (std::uint64_t i = 0; i < kvs; ++i) {
    const auto start = in.tellg();
    std::uint64_t length;
    read(&length, 8);
    std::string key(length, '\0');
    read(key.data(), length);
    std::uint32_t type;
    read(&type, 4);
    skip(type);
    const auto end = in.tellg();
    RawKv kv;
    kv.key = key;
    kv.bytes.resize(static_cast<std::size_t>(end - start));
    in.seekg(start);
    read(kv.bytes.data(), kv.bytes.size());
    out.push_back(std::move(kv));
  }
  return out;
}

RawKv StringKv(std::string_view key, std::string_view value) {
  RawKv kv;
  kv.key = std::string(key);
  auto put = [&](const void* p, std::size_t n) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    kv.bytes.insert(kv.bytes.end(), b, b + n);
  };
  const std::uint64_t key_length = key.size();
  const std::uint32_t type = 8;
  const std::uint64_t value_length = value.size();
  put(&key_length, 8);
  put(key.data(), key.size());
  put(&type, 4);
  put(&value_length, 8);
  put(value.data(), value.size());
  return kv;
}

std::vector<std::string> TargetTensorNames(const core::ModelConfig& config) {
  std::vector<std::string> names = {"token_embd.weight", "output_norm.weight",
                                    "output.weight"};
  for (std::uint32_t i = 0; i < config.num_layers; ++i) {
    const std::string p = "blk." + std::to_string(i) + ".";
    for (const char* n :
         {"attn_norm.weight", "post_attention_norm.weight", "ffn_gate.weight",
          "ffn_up.weight", "ffn_down.weight"})
      names.push_back(p + n);
    const bool full = ((i + 1) % config.full_attention_interval) == 0;
    if (full) {
      for (const char* n :
           {"attn_q.weight", "attn_k.weight", "attn_v.weight",
            "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight"})
        names.push_back(p + n);
    } else {
      for (const char* n :
           {"attn_qkv.weight", "attn_gate.weight", "ssm_a", "ssm_conv1d.weight",
            "ssm_dt.bias", "ssm_alpha.weight", "ssm_beta.weight",
            "ssm_norm.weight", "ssm_out.weight"})
        names.push_back(p + n);
    }
  }
  return names;
}

void QuantizeQ8_0(std::span<const float> values, std::uint8_t* out) {
  constexpr std::size_t kBlock = 32;
  if (values.size() % kBlock != 0) {
    throw std::invalid_argument("Q8_0: length is not a multiple of 32");
  }
  for (std::size_t b = 0; b < values.size() / kBlock; ++b) {
    const float* x = values.data() + b * kBlock;
    float amax = 0.0f;
    for (std::size_t j = 0; j < kBlock; ++j)
      amax = std::max(amax, std::fabs(x[j]));
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const std::uint16_t d16 = exl3::HalfFromDouble(static_cast<double>(d));
    std::uint8_t* block = out + b * 34;
    std::memcpy(block, &d16, 2);
    for (std::size_t j = 0; j < kBlock; ++j) {
      const auto q = static_cast<std::int8_t>(std::roundf(x[j] * id));
      std::memcpy(block + 2 + j, &q, 1);
    }
  }
}

bool Source::IsProjection(std::string_view gguf_name) const {
  const auto recipe = RecipeFor(gguf_name, geometry_, checkpoint_);
  return recipe && recipe->kind == Kind::kLinear;
}

HostLinear Source::PrepareLinear(std::string_view gguf_name) const {
  const auto recipe = RecipeFor(gguf_name, geometry_, checkpoint_);
  if (!recipe || recipe->kind != Kind::kLinear) {
    throw std::runtime_error(std::string(gguf_name) +
                             " is not an EXL3 projection");
  }
  const auto lin = checkpoint_.GetLinear(recipe->source);
  const std::size_t k = lin.in_features, n = lin.out_features;
  const std::size_t words = static_cast<std::size_t>(lin.rate.TileWords());
  const std::size_t tiles_k = k / 16, tiles_n = n / 16;
  HostLinear h;
  h.gguf_name = std::string(gguf_name);
  h.trellis.resize(tiles_k * tiles_n * words);
  h.suh.resize(k);
  h.svh.resize(n);
  std::vector<std::size_t> row_tile(tiles_k), col_tile(tiles_n);
  for (std::size_t t = 0; t < tiles_k; ++t)
    row_tile[t] = Destination(recipe->in_order, t * 16, k, geometry_) / 16;
  for (std::size_t t = 0; t < tiles_n; ++t)
    col_tile[t] = Destination(recipe->out_order, t * 16, n, geometry_) / 16;
  for (std::size_t tk = 0; tk < tiles_k; ++tk) {
    for (std::size_t tn = 0; tn < tiles_n; ++tn) {
      std::memcpy(
          h.trellis.data() + (row_tile[tk] * tiles_n + col_tile[tn]) * words,
          lin.trellis + (tk * tiles_n + tn) * words, words * 2);
    }
  }
  for (std::size_t i = 0; i < k; ++i)
    h.suh[Destination(recipe->in_order, i, k, geometry_)] = lin.suh[i];
  for (std::size_t j = 0; j < n; ++j)
    h.svh[Destination(recipe->out_order, j, n, geometry_)] = lin.svh[j];
  h.view.trellis = h.trellis.data();
  h.view.suh = h.suh.data();
  h.view.svh = h.svh.data();
  h.view.k = static_cast<std::uint32_t>(k);
  h.view.n = static_cast<std::uint32_t>(n);
  h.view.rate = lin.rate;
  h.view.codebook = lin.codebook;
  return h;
}

std::vector<std::uint8_t> Source::BuildImage(
    const std::vector<RawKv>& metadata, const std::vector<std::string>& names,
    EmbeddingFormat embedding, int threads) const {
  struct Entry {
    TensorPlan plan;      // as produced
    core::GgmlType type;  // as stored
    std::size_t bytes;    // as stored
  };
  std::vector<Entry> entries;
  for (const auto& name : names) {
    if (IsProjection(name))
      continue;
    std::string error;
    auto plan = Plan(name, &error);
    if (!plan)
      throw std::runtime_error(error);
    Entry e{*plan, plan->type, plan->bytes};
    if (name == "token_embd.weight" && embedding == EmbeddingFormat::kQ8_0) {
      e.type = core::GgmlType::kQ8_0;
      e.bytes = plan->bytes / 2 / 32 * 34;
    }
    entries.push_back(std::move(e));
  }
  constexpr std::size_t kAlign = 32;
  auto align = [](std::size_t x) { return (x + kAlign - 1) / kAlign * kAlign; };
  std::vector<std::uint8_t> image;
  auto put = [&](const void* p, std::size_t n) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    image.insert(image.end(), b, b + n);
  };
  auto u32 = [&](std::uint32_t v) { put(&v, 4); };
  auto u64 = [&](std::uint64_t v) { put(&v, 8); };
  put("GGUF", 4);
  u32(3);
  u64(entries.size());
  u64(metadata.size());
  for (const auto& kv : metadata)
    put(kv.bytes.data(), kv.bytes.size());
  std::size_t offset = 0;
  for (const auto& e : entries) {
    u64(e.plan.gguf_name.size());
    put(e.plan.gguf_name.data(), e.plan.gguf_name.size());
    u32(static_cast<std::uint32_t>(e.plan.ne.size()));
    for (const auto d : e.plan.ne)
      u64(d);
    u32(static_cast<std::uint32_t>(e.type));
    u64(offset);
    offset = align(offset + e.bytes);
  }
  image.resize(align(image.size()), 0);
  const std::size_t data_start = image.size();
  image.resize(data_start + offset, 0);
  std::size_t position = data_start;
  std::vector<std::uint8_t> scratch;
  for (const auto& e : entries) {
    std::span<std::uint8_t> out(image.data() + position, e.bytes);
    if (e.type == e.plan.type) {
      Produce(e.plan, out, threads);
    } else {
      scratch.resize(e.plan.bytes);
      Produce(e.plan, scratch, threads);
      const std::size_t count = e.plan.bytes / 2;
      std::vector<float> row(static_cast<std::size_t>(e.plan.ne[0]));
      const auto* bf16 = reinterpret_cast<const std::uint16_t*>(scratch.data());
      const std::size_t rows = count / row.size();
      const std::size_t row_bytes = row.size() / 32 * 34;
      for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t j = 0; j < row.size(); ++j)
          row[j] = Bf16ToFloat(bf16[r * row.size() + j]);
        QuantizeQ8_0(row, out.data() + r * row_bytes);
      }
    }
    position += align(e.bytes);
  }
  return image;
}

std::string CheckSidecar(const std::filesystem::path& checkpoint,
                         const core::GgufReader& sidecar) {
  auto read_json = [](const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
      throw std::runtime_error("cannot read " + path.string());
    std::ostringstream text;
    text << in.rdbuf();
    return json::parse(text.str());
  };
  try {
    const auto config = read_json(checkpoint / "config.json");
    const json::Value* text = config.find("text_config");
    const json::Value& c = text != nullptr ? *text : config;
    const auto u32 = [&](const char* key) {
      return sidecar.GetMetadataUint32(std::string("qwen35.") + key);
    };
    const auto nextn = u32("nextn_predict_layers").value_or(0);
    const struct {
      const char* hf;
      std::optional<std::uint32_t> gguf;
      const char* gguf_name;
    } checks[] = {
        {"hidden_size", u32("embedding_length"), "embedding_length"},
        {"intermediate_size", u32("feed_forward_length"),
         "feed_forward_length"},
        {"num_attention_heads", u32("attention.head_count"),
         "attention.head_count"},
        {"num_key_value_heads", u32("attention.head_count_kv"),
         "attention.head_count_kv"},
        {"head_dim", u32("attention.key_length"), "attention.key_length"},
        {"linear_num_key_heads", u32("ssm.group_count"), "ssm.group_count"},
        {"linear_num_value_heads", u32("ssm.time_step_rank"),
         "ssm.time_step_rank"},
        {"linear_key_head_dim", u32("ssm.state_size"), "ssm.state_size"},
        {"linear_conv_kernel_dim", u32("ssm.conv_kernel"), "ssm.conv_kernel"},
    };
    for (const auto& check : checks) {
      const auto hf = static_cast<std::uint64_t>(c.member_double(check.hf, -1));
      if (!check.gguf || hf != *check.gguf) {
        return std::string("config.json ") + check.hf +
               " differs from GGUF qwen35." + check.gguf_name;
      }
    }
    const auto layers =
        static_cast<std::uint64_t>(c.member_double("num_hidden_layers", -1));
    const auto blocks = u32("block_count");
    if (!blocks || layers + nextn != *blocks) {
      return "config.json num_hidden_layers differs from GGUF "
             "qwen35.block_count";
    }
    const auto inner = u32("ssm.inner_size");
    if (!inner || static_cast<std::uint64_t>(
                      c.member_double("linear_value_head_dim", -1)) *
                          static_cast<std::uint64_t>(
                              c.member_double("linear_num_value_heads", -1)) !=
                      *inner) {
      return "config.json linear value size differs from GGUF "
             "qwen35.ssm.inner_size";
    }
    const auto tokens = sidecar.GetMetadataStringArray("tokenizer.ggml.tokens");
    if (static_cast<double>(tokens.size()) !=
        c.member_double("vocab_size", -1)) {
      return "config.json vocab_size differs from tokenizer.ggml.tokens";
    }
    const auto tokenizer = read_json(checkpoint / "tokenizer.json");
    std::size_t checked = 0;
    auto check_token = [&](std::size_t id, const std::string& token) {
      if (id >= tokens.size() || tokens[id] != token) {
        throw std::runtime_error("tokenizer.json token " + std::to_string(id) +
                                 " differs from tokenizer.ggml.tokens");
      }
      ++checked;
    };
    if (const auto* model = tokenizer.find("model")) {
      if (const auto* vocab = model->find("vocab")) {
        for (const auto& [token, id] : vocab->members())
          check_token(id.as_size(), token);
      }
    }
    if (const auto* added = tokenizer.find("added_tokens")) {
      for (const auto& t : added->items())
        check_token(t.member_size("id"), t.member_str("content"));
    }
    if (checked == 0)
      return "tokenizer.json has no vocabulary";
  } catch (const std::exception& e) {
    return e.what();
  }
  return {};
}

namespace {

struct Exl3Storage {
  std::vector<HostLinear> linears;
};

}  // namespace

std::optional<LoadedExl3> LoadFromSidecar(
    const std::filesystem::path& sidecar_path, const core::GgufReader& sidecar,
    EmbeddingFormat embedding, std::string* error, int threads) {
  auto fail = [&](const std::string& why) -> std::optional<LoadedExl3> {
    if (error != nullptr)
      *error = "EXL3 sidecar " + sidecar_path.string() + ": " + why;
    return std::nullopt;
  };
  try {
    const auto relative = sidecar.GetMetadataString(exl3::kCheckpointKey);
    if (!relative)
      return fail(std::string("no ") + std::string(exl3::kCheckpointKey) +
                  " key");
    const auto directory = sidecar_path.parent_path() / std::string(*relative);
    if (auto mismatch = CheckSidecar(directory, sidecar); !mismatch.empty())
      return fail(mismatch);
    std::string why;
    const auto geometry = GeometryFromGguf(sidecar, &why);
    if (!geometry)
      return fail(why);
    const auto config = sidecar.ExtractModelConfig(&why);
    if (!config)
      return fail(why);
    const auto checkpoint = quant::exl3::Checkpoint::Open(directory);
    const Source source(checkpoint, *geometry);
    const auto names = TargetTensorNames(*config);

    auto image = std::make_shared<std::vector<std::uint8_t>>(source.BuildImage(
        ReadRawMetadata(sidecar_path), names, embedding, threads));
    auto* raw = core::GgufReader::OpenMemory(image->data(), image->size(), &why)
                    .release();
    if (raw == nullptr)
      return fail("in-memory GGUF: " + why);
    std::shared_ptr<const core::GgufReader> reader(
        raw, [image](const core::GgufReader* r) { delete r; });

    auto storage = std::make_shared<Exl3Storage>();
    std::size_t projections = 0;
    for (const auto& name : names)
      projections += source.IsProjection(name) ? 1 : 0;
    storage->linears.reserve(projections);  // views must not move
    std::vector<std::pair<std::string, QwenTensorRef>> refs;
    LoadedExl3 loaded;
    for (const auto& name : names) {
      if (!source.IsProjection(name))
        continue;
      storage->linears.push_back(source.PrepareLinear(name));
      auto& h = storage->linears.back();
      h.view.trellis = h.trellis.data();
      h.view.suh = h.suh.data();
      h.view.svh = h.svh.data();
      QwenTensorRef ref;
      ref.data = h.trellis.data();
      ref.num_elements = static_cast<std::size_t>(h.view.k) * h.view.n;
      ref.exl3 = &h.view;
      refs.emplace_back(name, ref);
      loaded.projection_bytes +=
          h.trellis.size() * 2 + h.suh.size() * 2 + h.svh.size() * 2;
    }
    auto weights = QwenModelWeights::LoadFromGguf(*reader, &why, &refs);
    if (!weights)
      return fail(why);
    weights->exl3_storage = storage;
    loaded.image_bytes = image->size();
    loaded.reader = std::move(reader);
    loaded.weights = std::move(*weights);
    return loaded;
  } catch (const std::exception& e) {
    return fail(e.what());
  }
}

}  // namespace gufo::models::qwen_exl3
