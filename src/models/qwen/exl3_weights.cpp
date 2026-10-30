#include "src/models/qwen/exl3_weights.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "src/core/quant/exl3/format.hpp"

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
  const auto nk = static_cast<std::size_t>(g.num_k_heads);
  const auto per_k = static_cast<std::size_t>(g.num_v_heads / g.num_k_heads);
  auto tiled = [&](std::size_t i, std::size_t head_dim) {
    // grouped (a = k head, b = v within group, h) -> tiled (b, a, h)
    const std::size_t h = i % head_dim;
    const std::size_t head = i / head_dim;
    const std::size_t a = head / per_k;
    const std::size_t b = head % per_k;
    return (b * nk + a) * head_dim + h;
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

float SourceValue(const exl3::Tensor& t, std::size_t i) {
  std::uint16_t bits;
  std::memcpy(&bits, t.data + 2 * i, 2);
  if (t.dtype == exl3::DType::kBF16) {
    return Bf16ToFloat(bits);
  }
  return static_cast<float>(exl3::HalfToDouble(bits));
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
          u16[d * cols + c] = FloatToBf16(SourceValue(t, r * cols + c));
        }
      }
      break;
    }
    case Kind::kNormPlusOne:
      for (std::size_t i = 0; i < plan.ne[0]; ++i) {
        f32[i] = SourceValue(t, i) + 1.0f;
      }
      break;
    case Kind::kToF32: {
      const auto size = static_cast<std::size_t>(plan.ne[0]);
      for (std::size_t i = 0; i < size; ++i) {
        f32[Destination(recipe->out_order, i, size, g)] = SourceValue(t, i);
      }
      break;
    }
    case Kind::kNegExp: {
      const auto size = static_cast<std::size_t>(plan.ne[0]);
      for (std::size_t i = 0; i < size; ++i) {
        f32[Destination(recipe->out_order, i, size, g)] =
            -std::exp(SourceValue(t, i));
      }
      break;
    }
    case Kind::kConv: {
      const auto channels = static_cast<std::size_t>(t.shape[0]);
      const auto kernel = static_cast<std::size_t>(t.shape[2]);
      for (std::size_t c = 0; c < channels; ++c) {
        const auto d = Destination(recipe->out_order, c, channels, g);
        for (std::size_t j = 0; j < kernel; ++j) {
          f32[d * kernel + j] = SourceValue(t, c * kernel + j);
        }
      }
      break;
    }
    case Kind::kLinear:
      break;
  }
}

}  // namespace gufo::models::qwen_exl3
