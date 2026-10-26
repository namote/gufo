// KVarN K0 accuracy study (plan/gufo-upstream-port/KVARN.md): prefills a chat
// prompt with FP16 KV, records the FP16 greedy continuation and its logits,
// then for each configuration restores the prefilled state with fake-quantized
// attention K/V rows and measures, per decode step, the KL divergence from the
// FP16 distribution and top-1 agreement under teacher forcing, followed by a
// free greedy run. `--prefill-chunk N` instead prefills N tokens at a time and
// quantizes each completed group before the next chunk reads it, as a
// quantized cache would during a long prefill. One JSON line per
// configuration goes to --out.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "tests/models/qwen27b/kvarn_fake_quant.hpp"
#include "tests/models/qwen27b/kvarn_session.hpp"

namespace json = gufo::json;
namespace kvarn = gufo::testing::kvarn;
using Token = gufo::tokenization::TokenId;
using gufo::testing::kvarn::Quantize;
using gufo::testing::kvarn::Relative;
using gufo::testing::kvarn::Session;
using gufo::testing::kvarn::State;
using Error = gufo::testing::kvarn::SessionError;

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

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string model_path, system_path, user_path, out_path,
        configs = "fp16,kvarn_k4v2_g128";
    std::uint32_t steps = 384, prefill_chunk = 0;
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
      else if (model_path.empty() && !arg.starts_with("--"))
        model_path = arg;
      else
        throw std::invalid_argument("unknown argument " + arg);
    }
    Require(!model_path.empty() && !user_path.empty() && !out_path.empty() &&
                steps > 0,
            "Usage: qwen27b_kvarn_k0 MODEL.gguf --user FILE [--system FILE] "
            "--out FILE.jsonl [--configs fp16,kvarn_k4v2_g128,...] "
            "[--steps N] [--prefill-chunk N]");
    std::vector<kvarn::Config> parsed;
    for (std::stringstream list(configs); list.good();) {
      std::string name;
      std::getline(list, name, ',');
      const auto config = kvarn::ParseConfig(name);
      Require(config.has_value(), "unknown configuration " + name);
      parsed.push_back(*config);
    }

    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
    Require(bool(reader), error);
    auto model = gufo::hip::QwenGpuModel::CreateFromGguf(
        std::shared_ptr<const gufo::core::GgufReader>(std::move(reader)),
        &error);
    Require(bool(model), error);
    const auto& tokenizer = model->GetTokenizer();
    std::vector<gufo::tokenization::ChatMessage> messages;
    if (!system_path.empty())
      messages.emplace_back(gufo::tokenization::ChatRole::kSystem,
                            ReadFile(system_path));
    messages.emplace_back(gufo::tokenization::ChatRole::kUser,
                          ReadFile(user_path));
    gufo::tokenization::ChatTemplateOptions options;
    options.enable_thinking = false;
    const auto rendered =
        gufo::tokenization::QwenChatTemplate::RenderAndTokenize(
            tokenizer, messages, options, &error);
    Require(rendered.has_value(), error);
    const std::vector<Token>& prompt = *rendered;
    const auto n = static_cast<std::uint32_t>(prompt.size());
    const auto& c = model->GetConfig();
    const std::uint32_t context = (n + steps + 1 + 255) / 256 * 256;
    Session session(model, context);
    auto& executor = session.Executor();
    std::cerr << "prompt_tokens=" << n << " context=" << context << std::endl;

    // FP16 reference: the prefilled state, the greedy continuation and the
    // log-probabilities at every step.
    auto start = Clock::now();
    (void)executor.ForwardPromptBatch(prompt);
    const double prefill_ms = Ms(start);
    const State base = session.Capture();
    std::vector<Token> reference_tokens;
    std::vector<std::vector<float>> reference_logprobs;
    std::vector<std::vector<float>> reference_logits;
    start = Clock::now();
    for (std::uint32_t i = 0; i < steps; ++i) {
      const auto logits = executor.CopyLastLogits();
      reference_logits.emplace_back(logits.begin(), logits.end());
      reference_logprobs.push_back(LogSoftmax(logits));
      reference_tokens.push_back(Argmax(logits));
      if (tokenizer.IsStopToken(reference_tokens.back()))
        break;
      (void)executor.ForwardToken(reference_tokens.back(), n + i);
    }
    const double decode_ms = Ms(start);
    std::cerr << "reference prefill_ms=" << prefill_ms
              << " decode_tokens=" << reference_tokens.size()
              << " decode_ms=" << decode_ms << std::endl;

    std::ofstream out(out_path, std::ios::app);
    Require(bool(out), "cannot write " + out_path);
    for (const auto& config : parsed) {
      const std::size_t group = config.group;
      State initial;
      Error quant_error;
      double quantize_ms = 0;
      start = Clock::now();
      if (prefill_chunk == 0 || config.method == kvarn::Method::kFp16) {
        initial = base;
        const auto t = Clock::now();
        quant_error = Quantize(session, initial, config, c.num_key_value_heads,
                               c.head_dim, 1, n / group);
        quantize_ms = Ms(t);
      } else {
        executor.Reset();
        std::size_t done = 1;
        for (std::uint32_t offset = 0; offset < n; offset += prefill_chunk) {
          const auto length = std::min(prefill_chunk, n - offset);
          (void)executor.ForwardPromptBatch(
              std::span(prompt).subspan(offset, length), offset);
          const std::size_t full = (offset + length) / group;
          if (full <= done)
            continue;
          State state = session.Capture();
          const auto t = Clock::now();
          const auto e = Quantize(session, state, config, c.num_key_value_heads,
                                  c.head_dim, done, full);
          quantize_ms += Ms(t);
          quant_error.key.squared += e.key.squared;
          quant_error.key.reference += e.key.reference;
          quant_error.value.squared += e.value.squared;
          quant_error.value.reference += e.value.reference;
          session.Restore(state);
          done = full;
        }
        initial = session.Capture();
      }
      const double prepare_ms = Ms(start);

      // Teacher forcing on the FP16 continuation.
      session.Restore(initial);
      std::vector<double> kls;
      std::size_t agree = 0, first_disagreement = reference_tokens.size();
      float max_logit_difference = 0;
      for (std::size_t i = 0; i < reference_tokens.size(); ++i) {
        const auto logits = executor.CopyLastLogits();
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
          (void)executor.ForwardToken(reference_tokens[i],
                                      n + static_cast<std::uint32_t>(i));
      }

      // Free greedy run from the same state.
      session.Restore(initial);
      std::vector<Token> tokens;
      for (std::uint32_t i = 0; i < steps; ++i) {
        tokens.push_back(Argmax(executor.CopyLastLogits()));
        if (tokenizer.IsStopToken(tokens.back()))
          break;
        (void)executor.ForwardToken(tokens.back(), n + i);
      }
      const auto divergence = static_cast<std::size_t>(
          std::ranges::mismatch(tokens, reference_tokens).in1 - tokens.begin());
      const auto text = [&](std::span<const Token> ids) {
        std::vector<Token> visible(ids.begin(), ids.end());
        if (!visible.empty() && tokenizer.IsStopToken(visible.back()))
          visible.pop_back();
        return tokenizer.Decode(visible);
      };

      auto row = json::Value::object();
      row["config"] = config.name;
      row["prompt_tokens"] = std::size_t{n};
      row["prefill_chunk"] = std::size_t{
          config.method == kvarn::Method::kFp16 ? 0U : prefill_chunk};
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
