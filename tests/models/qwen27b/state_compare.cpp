// Recurrent-state storage quality check (plan/upstream-sync2/V3-4-BRIEF.md):
// runs one long prompt and a greedy continuation through the FP32-state
// executor, then teacher-forces the same tokens through a compact-state
// executor (fp16 by default, or the rejected bf16 route) and reports their
// full-vocabulary drift: logit RMSE, KL(fp32 || candidate) and top-1
// agreement at the end of every prefill pass and at every decode step, plus
// the first decode step whose greedy token differs (the position where two
// independent greedy completions part).
//
// qwen27b_state_compare --model MODEL --prompt-file TEXT --tokens N
//     [--generate G] (default 256) [--batch B] (default 2048, the HTTP
//     scheduler's admission pass) [--candidate fp16|bf16]
//     [--kv-cache-type fp16|kvarn-k4v2|kvarn-k4v4] (default kvarn-k4v2, the
//     serving default) [--json OUT]
// TEXT must tokenize to at least N tokens; its first N are the prompt.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cli/model_artifact.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "tests/support/logit_drift.hpp"

namespace {

using Token = gufo::tokenization::TokenId;
using Storage = gufo::hip::QwenRecurrentStateStorage;

struct Options {
  std::string model;
  std::string prompt_file;
  std::size_t tokens = 0;
  std::size_t generate = 256;
  std::size_t batch = 2048;
  Storage candidate = Storage::kFp16;
  gufo::hip::QwenKvCacheStorage kv = gufo::hip::QwenKvCacheStorage::kKvarnK4V2;
  std::string json;
};

std::size_t ParseCount(const std::string& value, const char* name) {
  if (value.empty() || value.front() < '0' || value.front() > '9')
    throw std::invalid_argument(std::string(name) + " must be positive");
  std::size_t used = 0;
  const auto parsed = std::stoull(value, &used);
  if (used != value.size() || parsed == 0)
    throw std::invalid_argument(std::string(name) + " must be positive");
  return static_cast<std::size_t>(parsed);
}

Options Parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (i + 1 >= argc)
      throw std::invalid_argument(arg + " needs a value");
    const std::string value = argv[++i];
    if (arg == "--model") {
      o.model = value;
    } else if (arg == "--prompt-file") {
      o.prompt_file = value;
    } else if (arg == "--tokens") {
      o.tokens = ParseCount(value, "--tokens");
    } else if (arg == "--generate") {
      o.generate = ParseCount(value, "--generate");
    } else if (arg == "--batch") {
      o.batch = ParseCount(value, "--batch");
    } else if (arg == "--candidate") {
      if (value != "fp16" && value != "bf16")
        throw std::invalid_argument("--candidate must be fp16 or bf16");
      o.candidate = value == "fp16" ? Storage::kFp16 : Storage::kBf16;
    } else if (arg == "--kv-cache-type") {
      using Kv = gufo::hip::QwenKvCacheStorage;
      bool found = false;
      for (const auto kv : {Kv::kFp16, Kv::kKvarnK4V2, Kv::kKvarnK4V4}) {
        if (value == gufo::hip::QwenKvCacheStorageName(kv)) {
          o.kv = kv;
          found = true;
        }
      }
      if (!found)
        throw std::invalid_argument(
            "--kv-cache-type must be fp16, kvarn-k4v2 or kvarn-k4v4");
    } else if (arg == "--json") {
      o.json = value;
    } else {
      throw std::invalid_argument("unknown argument " + arg);
    }
  }
  if (o.model.empty() || o.prompt_file.empty() || o.tokens == 0)
    throw std::invalid_argument(
        "--model, --prompt-file and --tokens are required");
  return o;
}

std::vector<float> Copy(std::span<const float> logits) {
  return {logits.begin(), logits.end()};
}

/// One route: prefill in `batch`-token passes, keeping each pass's last-row
/// logits, then decode `steps` tokens. With empty `forced`, the decoded
/// tokens are the greedy ones and are returned in `*greedy`; otherwise
/// `forced` is fed.
std::vector<std::vector<float>> Run(
    const std::shared_ptr<const gufo::hip::QwenGpuModel>& model,
    const Options& o, Storage storage, std::span<const Token> prompt,
    std::span<const Token> forced, std::vector<Token>* greedy) {
  auto policy = gufo::hip::QwenExecutionPolicy::Production();
  policy.kv_cache_storage = o.kv;
  policy.recurrent_state_storage = storage;
  const auto context = static_cast<std::uint32_t>(
      (prompt.size() + o.generate + 256) / 256 * 256);
  gufo::hip::QwenGpuExecutor executor(model, context, policy);
  std::vector<std::vector<float>> rows;
  for (std::size_t start = 0; start < prompt.size(); start += o.batch) {
    const auto chunk =
        prompt.subspan(start, std::min(o.batch, prompt.size() - start));
    (void)executor.ForwardPromptBatch(chunk, static_cast<std::uint32_t>(start));
    rows.push_back(Copy(executor.CopyLastLogits()));
  }
  for (std::size_t step = 0; step < o.generate; ++step) {
    const Token token =
        forced.empty() ? static_cast<Token>(gufo::test::ArgMax(rows.back()))
                       : forced[step];
    if (greedy != nullptr)
      greedy->push_back(token);
    (void)executor.ForwardToken(
        token, static_cast<std::uint32_t>(prompt.size() + step));
    rows.push_back(Copy(executor.CopyLastLogits()));
  }
  return rows;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto o = Parse(argc, argv);
    std::string error;
    auto reader = gufo::cli::OpenModelArtifact(o.model, &error);
    if (!reader)
      throw std::runtime_error("open failed: " + error);
    auto model = gufo::hip::QwenGpuModel::CreateFromGguf(
        std::shared_ptr<const gufo::core::GgufReader>(std::move(reader)),
        &error);
    if (!model)
      throw std::runtime_error("model load failed: " + error);
    std::ifstream file(o.prompt_file, std::ios::binary);
    if (!file)
      throw std::runtime_error("cannot read " + o.prompt_file);
    const std::string text{std::istreambuf_iterator<char>(file), {}};
    const auto encoded = model->GetTokenizer().Encode(text);
    if (encoded.size() < o.tokens)
      throw std::runtime_error("the prompt file holds only " +
                               std::to_string(encoded.size()) + " tokens");
    const std::vector<Token> prompt(
        encoded.begin(),
        encoded.begin() + static_cast<std::ptrdiff_t>(o.tokens));

    std::vector<Token> greedy;
    const auto reference = Run(model, o, Storage::kFp32, prompt, {}, &greedy);
    const auto candidate = Run(model, o, o.candidate, prompt, greedy, nullptr);
    const std::size_t passes = reference.size() - o.generate;
    gufo::test::DriftSummary prefill;
    gufo::test::DriftSummary decode;
    // Generated token s is the top-1 of row passes - 1 + s.
    std::optional<std::size_t> divergence;
    for (std::size_t row = 0; row < reference.size(); ++row) {
      const auto drift =
          gufo::test::CompareLogits(reference[row], candidate[row]);
      if (row < passes) {
        prefill.Add(drift);
        continue;
      }
      decode.Add(drift);
    }
    for (std::size_t row = passes - 1;
         row < passes - 1 + o.generate && !divergence; ++row) {
      if (gufo::test::ArgMax(reference[row]) !=
          gufo::test::ArgMax(candidate[row]))
        divergence = row + 1 - passes;
    }
    std::ostringstream report;
    gufo::test::WriteStateComparison(
        report, "fp32", gufo::hip::QwenRecurrentStateStorageName(o.candidate),
        prompt.size(), o.generate, prefill, decode, divergence);
    std::cout << report.str();
    if (!o.json.empty()) {
      std::ofstream out(o.json);
      out << report.str();
      if (!out)
        throw std::runtime_error("cannot write " + o.json);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "qwen27b_state_compare: " << e.what() << '\n';
    return 1;
  }
}
