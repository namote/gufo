// KVarN K0 accuracy study for Qwen3.8 Flash-Next
// (plan/exl3/archive/F1-DESIGN.md 3.2, following
// plan/archive/gufo-upstream-port/KVARN.md K0): prefills a chat prompt with
// FP16 KV in an autoregressive session, records the FP16 greedy continuation
// and its logits, then for each configuration restores the prefilled state with
// fake-quantized full-attention K/V rows and measures, per decode step, the KL
// divergence from the FP16 distribution and top-1 agreement under teacher
// forcing, followed by a free greedy run.
//
// Only the public engine API is used. The state moves through a
// materialized snapshot payload; its attention K/V sections are rewritten in
// place with tests/support/kvarn_fake_quant.hpp (heads x head_dim rows
// are generic) and restored. Pooled indexer keys, raw indexer rows, DeltaNet
// state and the PLE history stay FP16/F32: KVarN covers attention K/V only,
// so block selection sees unchanged indexer keys.
//
// `--prefill-chunk N` prefills N tokens at a time and quantizes each
// completed group before the next chunk reads it; `--quantize-decode`
// quantizes a group completed during decoding before the next position reads
// it. One JSON line per configuration goes to --out.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/session_mode.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "tests/support/kvarn_fake_quant.hpp"

namespace json = gufo::json;
namespace kvarn = gufo::testing::kvarn;
namespace qfn = gufo::models::qwen38_flash_next;
namespace tok = gufo::tokenization;
using Token = std::int32_t;

namespace {
using Clock = std::chrono::steady_clock;

void Require(bool ok, const std::string& error) {
  if (!ok)
    throw std::runtime_error(error);
}

double Ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

std::string ReadFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  Require(bool(file), "cannot read " + path);
  return {std::istreambuf_iterator<char>(file), {}};
}

std::vector<float> LogSoftmax(std::span<const float> logits) {
  const float top = *std::ranges::max_element(logits);
  double sum = 0;
  for (const float v : logits)
    sum += std::exp(static_cast<double>(v) - top);
  const double log_sum = std::log(sum) + top;
  std::vector<float> result(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i)
    result[i] = static_cast<float>(logits[i] - log_sum);
  return result;
}

Token Argmax(std::span<const float> values) {
  return static_cast<Token>(std::ranges::max_element(values) - values.begin());
}

// KL(reference || candidate) of two log-probability rows.
double Kl(std::span<const float> reference, std::span<const float> candidate) {
  double kl = 0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double p = std::exp(static_cast<double>(reference[i]));
    if (p > 0)
      kl += p * (reference[i] - static_cast<double>(candidate[i]));
  }
  return std::max(kl, 0.0);
}

json::Value Distribution(std::vector<double> values) {
  auto result = json::Value::object();
  if (values.empty())
    return result;
  std::ranges::sort(values);
  const auto at = [&](double q) {
    return values[std::min(values.size() - 1,
                           static_cast<std::size_t>(q * values.size()))];
  };
  result["median"] = at(0.5);
  result["p95"] = at(0.95);
  result["max"] = values.back();
  return result;
}

template<typename T>
T Field(std::span<const std::uint8_t> bytes, std::size_t offset) {
  Require(offset + sizeof(T) <= bytes.size(), "snapshot field out of range");
  T value;
  std::memcpy(&value, bytes.data() + offset, sizeof(T));
  return value;
}

/// The full-attention K and V sections of a text snapshot payload. Mirrors
/// the session framing (engine.cpp SessionSnapshotHeader: executor_bytes at
/// byte 24, executor payload last) and the executor section order
/// (runtime/executor_snapshot.cpp WalkSnapshot, "QFNSNAP6": DeltaNet conv
/// and state per linear layer, PLE history, then per attention layer K, V,
/// unpooled indexer rows and pooled keys). The walk must end exactly at the
/// executor payload's recorded size.
struct KvSections {
  std::uint32_t position{0}, kv_row{0};
  std::vector<std::size_t> keys, values;  ///< byte offsets in the payload
};

KvSections FindKvSections(std::span<const std::uint8_t> payload) {
  constexpr std::size_t kExecutorHeader = 112;
  const auto executor_bytes = Field<std::uint64_t>(payload, 24);
  Require(executor_bytes <= payload.size(), "snapshot executor size");
  const std::size_t base = payload.size() - executor_bytes;
  const auto exec = payload.subspan(base);
  Require(std::memcmp(exec.data(), "QFNSNAP6", 8) == 0,
          "unexpected executor snapshot magic");
  const auto layers = Field<std::uint32_t>(exec, 8);
  const auto interval = Field<std::uint32_t>(exec, 12);
  const auto conv = Field<std::uint32_t>(exec, 16);
  const auto state = Field<std::uint32_t>(exec, 20);
  const auto kv_row = Field<std::uint32_t>(exec, 24);
  const auto index_dim = Field<std::uint32_t>(exec, 28);
  const auto ratio = Field<std::uint32_t>(exec, 32);
  const auto ple = Field<std::uint32_t>(exec, 40);
  const auto has_mtp = Field<std::uint32_t>(exec, 44);
  const auto position = Field<std::uint32_t>(exec, 48);
  const auto blocks = Field<std::uint32_t>(exec, 52);
  const auto images = Field<std::uint32_t>(exec, 72);
  const auto recorded = Field<std::uint64_t>(exec, 104);
  Require(has_mtp == 0 && images == 0 && recorded == executor_bytes,
          "K0 needs a text snapshot of an autoregressive session");
  KvSections s{position, kv_row, {}, {}};
  std::uint64_t offset = kExecutorHeader;
  for (std::uint32_t il = 0; il < layers; ++il)
    if ((il + 1) % interval != 0)
      offset += (std::uint64_t{conv} + state) * sizeof(float);
  offset += std::uint64_t{ple} * sizeof(float);
  const std::uint64_t kv_bytes = std::uint64_t{position} * kv_row * 2;
  const std::uint64_t index_rows = position - std::uint64_t{blocks} * ratio;
  for (std::uint32_t il = 0; il < layers; ++il) {
    if ((il + 1) % interval != 0)
      continue;
    s.keys.push_back(base + offset);
    offset += kv_bytes;
    s.values.push_back(base + offset);
    offset += kv_bytes;
    offset += index_rows * index_dim * sizeof(float);
    offset += std::uint64_t{blocks} * index_dim * 2;
  }
  Require(offset == executor_bytes, "snapshot sections do not add up");
  return s;
}

struct QuantError {
  kvarn::Error key, value;
};

double Relative(const kvarn::Error& error) {
  return error.reference > 0 ? std::sqrt(error.squared / error.reference) : 0;
}

/// Fake-quantizes groups [first_group, end_group) of every attention layer's
/// K and V rows in `payload`.
QuantError QuantizePayload(std::vector<std::uint8_t>& payload,
                           const kvarn::Config& config, std::uint32_t heads,
                           std::uint32_t dim, std::size_t first_group,
                           std::size_t end_group) {
  QuantError total;
  if (config.method == kvarn::Method::kFp16 || end_group <= first_group)
    return total;
  const auto sections = FindKvSections(payload);
  Require(sections.kv_row == heads * dim, "K0 KV row width");
  const std::size_t count = std::size_t{sections.position} * sections.kv_row;
  for (const bool key : {true, false}) {
    for (const std::size_t offset : key ? sections.keys : sections.values) {
      std::span rows(reinterpret_cast<std::uint16_t*>(payload.data() + offset),
                     count);
      const auto e = kvarn::FakeQuantize(rows, heads, dim, key, config,
                                         first_group, end_group);
      auto& sum = key ? total.key : total.value;
      sum.squared += e.squared;
      sum.reference += e.reference;
    }
  }
  return total;
}

}  // namespace

// FP16 reference of --dump-reference: u32 steps, u32 vocabulary, the
// greedy tokens (i32) and every step's logits (f32), host byte order.
void WriteReference(const std::string& path, std::span<const Token> tokens,
                    const std::vector<std::vector<float>>& logits) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  Require(bool(out), "cannot write " + path);
  const std::uint32_t head[2] = {
      static_cast<std::uint32_t>(tokens.size()),
      static_cast<std::uint32_t>(logits.empty() ? 0 : logits[0].size())};
  out.write(reinterpret_cast<const char*>(head), sizeof(head));
  for (const Token token : tokens) {
    const auto id = static_cast<std::int32_t>(token);
    out.write(reinterpret_cast<const char*>(&id), sizeof(id));
  }
  for (const auto& step : logits)
    out.write(reinterpret_cast<const char*>(step.data()),
              static_cast<std::streamsize>(step.size() * sizeof(float)));
  Require(bool(out), "cannot write " + path);
}

// Teacher-forced KL and top-1 of the loaded (KVarN) model against the dumped
// FP16 reference, then a free greedy run from a fresh prefill.
int RunReal(qfn::Model& model, std::span<const Token> prompt,
            std::uint32_t session_context, const std::string& reference_path,
            const std::string& out_path, std::uint32_t steps) {
  std::ifstream in(reference_path, std::ios::binary);
  Require(bool(in), "cannot read " + reference_path);
  std::uint32_t head[2]{};
  in.read(reinterpret_cast<char*>(head), sizeof(head));
  std::vector<Token> reference_tokens(head[0]);
  for (auto& token : reference_tokens) {
    std::int32_t id = 0;
    in.read(reinterpret_cast<char*>(&id), sizeof(id));
    token = static_cast<Token>(id);
  }
  std::vector<std::vector<float>> reference_logits(head[0],
                                                   std::vector<float>(head[1]));
  for (auto& step : reference_logits)
    in.read(reinterpret_cast<char*>(step.data()),
            static_cast<std::streamsize>(step.size() * sizeof(float)));
  Require(bool(in) && head[0] != 0, "truncated reference " + reference_path);

  std::string error;
  const auto prefilled = [&]() {
    auto session = model.CreateSession(gufo::core::SessionMode::kAutoregressive,
                                       session_context, &error);
    Require(bool(session), "session failed: " + error);
    Require(session->Sync(prompt, &error), "prefill failed: " + error);
    return session;
  };
  auto start = Clock::now();
  auto session = prefilled();
  const double prefill_ms = Ms(start);
  std::vector<double> kls;
  std::size_t agree = 0, first_disagreement = reference_tokens.size();
  float max_logit_difference = 0;
  for (std::size_t i = 0; i < reference_tokens.size(); ++i) {
    const auto logits = session->Logits();
    Require(logits.size() == head[1], "vocabulary differs from the reference");
    for (std::size_t v = 0; v < logits.size(); ++v)
      max_logit_difference = std::max(
          max_logit_difference, std::abs(logits[v] - reference_logits[i][v]));
    kls.push_back(Kl(LogSoftmax(reference_logits[i]), LogSoftmax(logits)));
    if (Argmax(logits) == reference_tokens[i])
      ++agree;
    else if (first_disagreement == reference_tokens.size())
      first_disagreement = i;
    if (i + 1 < reference_tokens.size())
      Require(session->Evaluate(reference_tokens[i], &error),
              "evaluate failed: " + error);
  }
  session.reset();
  session = prefilled();
  std::vector<Token> tokens;
  start = Clock::now();
  for (std::uint32_t i = 0; i < steps; ++i) {
    tokens.push_back(Argmax(session->Logits()));
    if (model.IsStopToken(tokens.back()))
      break;
    Require(session->Evaluate(tokens.back(), &error),
            "evaluate failed: " + error);
  }
  const double decode_ms = Ms(start);
  const auto divergence = static_cast<std::size_t>(
      std::ranges::mismatch(tokens, reference_tokens).in1 - tokens.begin());
  auto row = json::Value::object();
  row["model"] = "qwen3.8-flash-next";
  row["config"] = "kvarn_k4v4_real";
  row["prompt_tokens"] = prompt.size();
  row["context"] = std::size_t{session_context};
  row["steps"] = reference_tokens.size();
  row["kl"] = Distribution(kls);
  auto kl_steps = json::Value::array();
  for (const double kl : kls)
    kl_steps.push_back(kl);
  row["kl_steps"] = std::move(kl_steps);
  row["top1_agreement"] = static_cast<double>(agree) / reference_tokens.size();
  row["first_top1_disagreement"] = first_disagreement;
  row["max_abs_logit_difference"] = max_logit_difference;
  row["free_tokens"] = tokens.size();
  row["free_divergence"] = divergence;
  row["free_identical"] = tokens == reference_tokens;
  row["prefill_ms"] = prefill_ms;
  row["free_decode_ms"] = decode_ms;
  std::ofstream out(out_path, std::ios::app);
  Require(bool(out), "cannot write " + out_path);
  out << row.dump() << '\n' << std::flush;
  std::cerr << "kvarn_k4v4_real kl_median=" << row["kl"].member_double("median")
            << " top1=" << row["top1_agreement"].as_double()
            << " free_identical=" << (tokens == reference_tokens) << std::endl;
  return 0;
}

int main(int argc, char** argv) {
  try {
    std::string model_path, system_path, user_path, out_path,
        configs = "fp16,kvarn_k4v2_g128", dump_path, real_path;
    std::uint32_t steps = 384, prefill_chunk = 0, context = 0;
    bool quantize_decode = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto value = [&]() -> std::string {
        Require(i + 1 < argc, arg + " needs a value");
        return argv[++i];
      };
      if (arg == "--system")
        system_path = value();
      else if (arg == "--user")
        user_path = value();
      else if (arg == "--out")
        out_path = value();
      else if (arg == "--configs")
        configs = value();
      else if (arg == "--steps")
        steps = static_cast<std::uint32_t>(std::stoul(value()));
      else if (arg == "--prefill-chunk")
        prefill_chunk = static_cast<std::uint32_t>(std::stoul(value()));
      else if (arg == "--context")
        context = static_cast<std::uint32_t>(std::stoul(value()));
      else if (arg == "--quantize-decode")
        quantize_decode = true;
      else if (arg == "--dump-reference")
        dump_path = value();
      else if (arg == "--real")
        real_path = value();
      else if (model_path.empty() && !arg.starts_with("--"))
        model_path = arg;
      else
        throw std::invalid_argument("unknown argument " + arg);
    }
    Require(!model_path.empty() && !user_path.empty() && !out_path.empty() &&
                steps > 0,
            "Usage: qwen38_flash_next_kvarn_k0 MODEL.gguf --user FILE "
            "[--system FILE] --out FILE.jsonl "
            "[--configs fp16,kvarn_k4v2_g128,...] [--steps N] [--context N] "
            "[--prefill-chunk N] [--quantize-decode] "
            "[--dump-reference FILE | --real FILE]");
    std::vector<kvarn::Config> parsed;
    for (std::stringstream list(configs); list.good();) {
      std::string name;
      std::getline(list, name, ',');
      const auto config = kvarn::ParseConfig(name);
      Require(config.has_value(), "unknown configuration " + name);
      parsed.push_back(*config);
    }

    // The model allows `--context` (default: the native 262,144); the
    // session takes the prompt plus the steps, rounded to a 256-token block.
    std::string error;
    std::vector<tok::ChatMessage> messages;
    if (!system_path.empty())
      messages.emplace_back(tok::ChatRole::kSystem, ReadFile(system_path));
    messages.emplace_back(tok::ChatRole::kUser, ReadFile(user_path));
    qfn::ModelOptions options;
    options.max_context = context != 0 ? context : 262144;
    options.decode_concurrency = 1;
    // --real runs the production KVarN k4v4 executor against the FP16
    // reference a --dump-reference run wrote (one model per process).
    options.kvarn = !real_path.empty();
    auto model = qfn::Model::Load(model_path, options, &error);
    Require(bool(model), "model load failed: " + error);
    tok::ChatTemplateOptions chat;
    chat.enable_thinking = false;
    chat.max_output_bytes = tok::RenderedPromptBoundBytes(options.max_context);
    const auto rendered = tok::QwenChatTemplate::RenderAndTokenize(
        model->tokenizer(), messages, chat, &error);
    Require(rendered.has_value(), "render failed: " + error);
    const std::vector<Token> prompt(rendered->begin(), rendered->end());
    const auto n = static_cast<std::uint32_t>(prompt.size());
    const std::uint32_t session_context = std::min<std::uint32_t>(
        options.max_context, (n + steps + 1 + 255) / 256 * 256);
    Require(n + steps + 1 <= session_context,
            "prompt and steps exceed --context " +
                std::to_string(options.max_context));
    const auto& c = model->config();
    const std::uint32_t heads = c.num_kv_heads, dim = c.head_dim;
    auto session = model->CreateSession(
        gufo::core::SessionMode::kAutoregressive, session_context, &error);
    Require(bool(session), "session failed: " + error);
    std::cerr << "prompt_tokens=" << n << " context=" << session_context
              << std::endl;

    const auto snapshot = [&]() {
      auto s = session->SaveSnapshot(&error,
                                     qfn::Session::SnapshotMode::kMaterialized);
      Require(bool(s), "snapshot failed: " + error);
      const auto bytes = s->bytes();
      return std::vector<std::uint8_t>(bytes.begin(), bytes.end());
    };
    const auto restore = [&](const std::vector<std::uint8_t>& payload) {
      Require(session->RestoreSnapshot(payload, &error),
              "restore failed: " + error);
    };
    const auto evaluate = [&](Token token) {
      Require(session->Evaluate(token, &error), "evaluate failed: " + error);
    };

    if (!real_path.empty())
      return RunReal(*model, prompt, session_context, real_path, out_path,
                     steps);

    // FP16 reference: the prefilled state, the greedy continuation and the
    // log-probabilities at every step.
    auto start = Clock::now();
    Require(session->Sync(prompt, &error), "prefill failed: " + error);
    const double prefill_ms = Ms(start);
    const auto base = snapshot();
    std::vector<Token> reference_tokens;
    std::vector<std::vector<float>> reference_logprobs;
    std::vector<std::vector<float>> reference_logits;
    start = Clock::now();
    for (std::uint32_t i = 0; i < steps; ++i) {
      const auto logits = session->Logits();
      reference_logits.emplace_back(logits.begin(), logits.end());
      reference_logprobs.push_back(LogSoftmax(logits));
      reference_tokens.push_back(Argmax(logits));
      if (model->IsStopToken(reference_tokens.back()))
        break;
      evaluate(reference_tokens.back());
    }
    const double decode_ms = Ms(start);
    std::cerr << "reference prefill_ms=" << prefill_ms
              << " decode_tokens=" << reference_tokens.size()
              << " decode_ms=" << decode_ms << std::endl;

    if (!dump_path.empty())
      WriteReference(dump_path, reference_tokens, reference_logits);

    std::ofstream out(out_path, std::ios::app);
    Require(bool(out), "cannot write " + out_path);
    for (const auto& config : parsed) {
      const std::size_t group = config.group;
      std::vector<std::uint8_t> initial;
      QuantError quant_error;
      double quantize_ms = 0;
      const auto add = [&](const QuantError& e) {
        quant_error.key.squared += e.key.squared;
        quant_error.key.reference += e.key.reference;
        quant_error.value.squared += e.value.squared;
        quant_error.value.reference += e.value.reference;
      };
      start = Clock::now();
      if (prefill_chunk == 0 || config.method == kvarn::Method::kFp16) {
        initial = base;
        const auto t = Clock::now();
        add(QuantizePayload(initial, config, heads, dim, 1, n / group));
        quantize_ms = Ms(t);
      } else {
        session->Reset();
        std::size_t done = 1;
        for (std::uint32_t end = std::min(prefill_chunk, n);;
             end = std::min(end + prefill_chunk, n)) {
          Require(session->Sync(std::span(prompt).first(end), &error),
                  "chunked prefill failed: " + error);
          const std::size_t full = end / group;
          if (full > done) {
            auto state = snapshot();
            const auto t = Clock::now();
            add(QuantizePayload(state, config, heads, dim, done, full));
            quantize_ms += Ms(t);
            restore(state);
            done = full;
          }
          if (end == n)
            break;
        }
        initial = snapshot();
      }
      const double prepare_ms = Ms(start);

      // --quantize-decode: a group that ends during decoding is quantized
      // before the position after it reads it.
      const auto forward = [&](Token token, std::uint32_t position) {
        if (quantize_decode && config.method != kvarn::Method::kFp16 &&
            position % group == 0 && position / group >= 2) {
          auto state = snapshot();
          const auto t = Clock::now();
          (void)QuantizePayload(state, config, heads, dim, position / group - 1,
                                position / group);
          quantize_ms += Ms(t);
          restore(state);
        }
        evaluate(token);
      };

      // Teacher forcing on the FP16 continuation.
      restore(initial);
      std::vector<double> kls;
      std::size_t agree = 0, first_disagreement = reference_tokens.size();
      float max_logit_difference = 0;
      for (std::size_t i = 0; i < reference_tokens.size(); ++i) {
        const auto logits = session->Logits();
        for (std::size_t v = 0; v < logits.size(); ++v)
          max_logit_difference =
              std::max(max_logit_difference,
                       std::abs(logits[v] - reference_logits[i][v]));
        kls.push_back(Kl(reference_logprobs[i], LogSoftmax(logits)));
        if (Argmax(logits) == reference_tokens[i])
          ++agree;
        else if (first_disagreement == reference_tokens.size())
          first_disagreement = i;
        if (i + 1 < reference_tokens.size())
          forward(reference_tokens[i], n + static_cast<std::uint32_t>(i));
      }

      // Free greedy run from the same state.
      restore(initial);
      std::vector<Token> tokens;
      for (std::uint32_t i = 0; i < steps; ++i) {
        tokens.push_back(Argmax(session->Logits()));
        if (model->IsStopToken(tokens.back()))
          break;
        forward(tokens.back(), n + i);
      }
      const auto divergence = static_cast<std::size_t>(
          std::ranges::mismatch(tokens, reference_tokens).in1 - tokens.begin());
      const auto text = [&](std::span<const Token> ids) {
        std::vector<Token> visible(ids.begin(), ids.end());
        if (!visible.empty() && model->IsStopToken(visible.back()))
          visible.pop_back();
        return model->Decode(visible);
      };

      auto row = json::Value::object();
      row["model"] = "qwen3.8-flash-next";
      row["config"] = config.name;
      row["prompt_tokens"] = std::size_t{n};
      row["context"] = std::size_t{session_context};
      row["prefill_chunk"] = std::size_t{
          config.method == kvarn::Method::kFp16 ? 0U : prefill_chunk};
      row["quantize_decode"] =
          quantize_decode && config.method != kvarn::Method::kFp16;
      row["quantized_tokens"] =
          config.method == kvarn::Method::kFp16
              ? 0
              : (n / group > 1 ? (n / group - 1) * group : 0);
      row["key_relative_error"] = Relative(quant_error.key);
      row["value_relative_error"] = Relative(quant_error.value);
      row["steps"] = reference_tokens.size();
      row["kl"] = Distribution(kls);
      auto kl_steps = json::Value::array();
      for (const double kl : kls)
        kl_steps.push_back(kl);
      row["kl_steps"] = std::move(kl_steps);
      row["top1_agreement"] =
          static_cast<double>(agree) / reference_tokens.size();
      row["first_top1_disagreement"] = first_disagreement;
      row["max_abs_logit_difference"] = max_logit_difference;
      row["free_tokens"] = tokens.size();
      row["free_divergence"] = divergence;
      row["free_identical"] = tokens == reference_tokens;
      row["free_text"] = text(tokens);
      row["reference_text"] = text(reference_tokens);
      row["reference_prefill_ms"] = prefill_ms;
      row["reference_decode_ms"] = decode_ms;
      row["prepare_ms"] = prepare_ms;
      row["quantize_ms"] = quantize_ms;
      out << row.dump() << '\n' << std::flush;
      std::cerr << config.name
                << " kl_median=" << row["kl"].member_double("median")
                << " top1=" << row["top1_agreement"].as_double()
                << " free_identical=" << (tokens == reference_tokens)
                << std::endl;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
