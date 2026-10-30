// RQ R2 loader check (RQ2I): writes a llama.cpp `--kl-divergence-base` file
// from gufo logits, so upstream `llama-perplexity --kl-divergence` can compare
// its own logits on the same GGUF and tokens with gufo's. Format and row
// encoding follow upstream llama.cpp 8ab29fae tools/perplexity/perplexity.cpp
// (log_softmax with uint16 output, process_logits, the "_logits_" header);
// tools/qwen27b/rq_quality/logits_io.py mirrors it in Python.
//
// Chunks: chunk c takes --n-ctx tokens of corpus window (c % windows) starting
// at n_ctx * (c / windows); each chunk starts from a reset executor, as
// llama.cpp clears its KV cache per chunk. Logits come from the production
// prefill head on captured hidden rows (LaunchQwenOutputHead, as
// qwen27b_rq_logits --path prefill), for positions n_ctx/2 .. n_ctx-2. Qwen3.8
// adds no BOS, so tokens are written unchanged.
//
// `--encode-rows IN OUT` encodes raw float32 rows (int32 n_vocab, then rows)
// without a GPU, for the CPU test against the Python mirror.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/exl3_dispatch.hpp"
#include "src/models/qwen/hip/output_head.hpp"

namespace json = gufo::json;
using Token = gufo::tokenization::TokenId;

namespace {

void Require(bool ok, const std::string& error) {
  if (!ok)
    throw std::runtime_error(error);
}

void HipCheck(hipError_t status, const char* what) {
  Require(status == hipSuccess,
          std::string(what) + ": " + hipGetErrorString(status));
}

std::string ReadFile(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  Require(bool(file), "cannot read " + path.string());
  return {std::istreambuf_iterator<char>(file), {}};
}

// perplexity.cpp nearest_int.
int NearestInt(float fval) {
  float val = fval + 12582912.F;
  int i = 0;
  std::memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

std::size_t RowWidth(std::size_t n_vocab) {
  return 2 * ((n_vocab + 1) / 2) + 4;
}

// perplexity.cpp log_softmax(int, const float*, uint16_t*, int), the row part.
void EncodeRow(std::span<const float> logits, std::span<std::uint16_t> row) {
  const int n_vocab = static_cast<int>(logits.size());
  float max_logit = logits[0];
  float min_logit = logits[0];
  for (int i = 1; i < n_vocab; ++i) {
    max_logit = std::max(max_logit, logits[i]);
    min_logit = std::min(min_logit, logits[i]);
  }
  min_logit = std::max(min_logit, max_logit - 16);
  double sum_exp = 0.0;
  for (int i = 0; i < n_vocab; ++i)
    sum_exp += expf(logits[i] - max_logit);
  const float log_sum_exp = log(sum_exp);
  const float min_log_prob = min_logit - max_logit - log_sum_exp;
  const float scale = (max_logit - min_logit) / 65535.F;
  std::fill(row.begin(), row.end(), std::uint16_t{0});
  std::memcpy(row.data(), &scale, sizeof(float));
  std::memcpy(row.data() + 2, &min_log_prob, sizeof(float));
  std::uint16_t* log_prob = row.data() + 4;
  if (scale != 0.F) {
    const float inv_scale = 1 / scale;
    for (int i = 0; i < n_vocab; ++i)
      log_prob[i] = logits[i] > min_logit
                        ? static_cast<std::uint16_t>(
                              NearestInt(inv_scale * (logits[i] - min_logit)))
                        : 0;
  }
}

int EncodeRowsCommand(const std::string& in_path, const std::string& out_path) {
  const std::string raw = ReadFile(in_path);
  Require(raw.size() >= 4, "input too short");
  std::int32_t n_vocab = 0;
  std::memcpy(&n_vocab, raw.data(), 4);
  Require(n_vocab > 0 && (raw.size() - 4) % (4 * std::size_t(n_vocab)) == 0,
          "input is not int32 n_vocab followed by float32 rows");
  const std::size_t rows = (raw.size() - 4) / (4 * std::size_t(n_vocab));
  std::vector<float> logits(n_vocab);
  std::vector<std::uint16_t> row(RowWidth(n_vocab));
  std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
  Require(bool(out), "cannot write " + out_path);
  for (std::size_t r = 0; r < rows; ++r) {
    std::memcpy(logits.data(), raw.data() + 4 + (r * 4 * n_vocab),
                4 * std::size_t(n_vocab));
    EncodeRow(logits, row);
    out.write(reinterpret_cast<const char*>(row.data()),
              static_cast<std::streamsize>(row.size() * 2));
  }
  Require(bool(out), "write failed");
  return 0;
}

std::vector<std::vector<Token>> ReadWindows(const std::filesystem::path& path) {
  const auto index = json::parse(ReadFile(path));
  Require(index.member_str("schema") == "gufo.rq-corpus-windows.v1",
          "unexpected windows schema");
  const std::string bytes =
      ReadFile(path.parent_path() / index.member_str("tokens_file"));
  std::vector<std::vector<Token>> windows;
  for (const auto& entry : index.find("windows")->items()) {
    const std::size_t offset = entry.member_size("offset");
    const std::size_t length = entry.member_size("length");
    Require((offset + length) * 4 <= bytes.size(), "window out of range");
    std::vector<Token> tokens(length);
    for (std::size_t i = 0; i < length; ++i) {
      std::int32_t v = 0;
      std::memcpy(&v, bytes.data() + ((offset + i) * 4), 4);
      tokens[i] = static_cast<Token>(v);
    }
    windows.push_back(std::move(tokens));
  }
  Require(!windows.empty(), "no windows");
  return windows;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 4 && std::string(argv[1]) == "--encode-rows")
      return EncodeRowsCommand(argv[2], argv[3]);
    std::string model_path, windows_path, out_path;
    std::size_t n_ctx = 512, chunks = 16;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto value = [&]() -> std::string {
        Require(i + 1 < argc, arg + " needs a value");
        return argv[++i];
      };
      if (arg == "--windows")
        windows_path = value();
      else if (arg == "--out")
        out_path = value();
      else if (arg == "--n-ctx")
        n_ctx = std::stoul(value());
      else if (arg == "--chunks")
        chunks = std::stoul(value());
      else if (model_path.empty() && !arg.starts_with("--"))
        model_path = arg;
      else
        throw std::invalid_argument("unknown argument " + arg);
    }
    Require(!model_path.empty() && !windows_path.empty() && !out_path.empty(),
            "Usage: qwen27b_rq_kld_base MODEL.gguf --windows windows.json "
            "--out FILE [--n-ctx 512] [--chunks 16] | --encode-rows IN OUT");
    Require(n_ctx >= 4 && chunks >= 1, "--n-ctx >= 4 and --chunks >= 1");

    const auto windows = ReadWindows(windows_path);
    std::vector<Token> tokens;
    tokens.reserve(n_ctx * chunks);
    for (std::size_t c = 0; c < chunks; ++c) {
      const auto& window = windows[c % windows.size()];
      const std::size_t start = n_ctx * (c / windows.size());
      Require(start + n_ctx <= window.size(),
              "a window is too short for chunk " + std::to_string(c));
      tokens.insert(tokens.end(), window.begin() + start,
                    window.begin() + start + n_ctx);
    }

    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
    Require(bool(reader), error);
    auto model = gufo::hip::QwenGpuModel::CreateFromGguf(
        std::shared_ptr<const gufo::core::GgufReader>(std::move(reader)),
        &error);
    Require(bool(model), error);
    const auto& config = model->GetConfig();
    const std::size_t n_vocab = config.vocab_size;
    const std::size_t hidden = config.hidden_size;
    gufo::hip::QwenGpuExecutor executor(
        model, static_cast<std::uint32_t>((n_ctx + 255) / 256 * 256));

    hipStream_t stream = nullptr;
    HipCheck(hipStreamCreate(&stream), "hipStreamCreate");
    float *d_hidden = nullptr, *d_normed = nullptr, *d_logits = nullptr;
    HipCheck(hipMalloc(&d_hidden, hidden * sizeof(float)), "hipMalloc");
    HipCheck(hipMalloc(&d_normed, hidden * sizeof(float)), "hipMalloc");
    HipCheck(hipMalloc(&d_logits, n_vocab * sizeof(float)), "hipMalloc");
    // An EXL3 head (gufo.gguf sidecar) needs a projection workspace, sized
    // like the DFlash2 executor's; GGUF heads keep the null workspace.
    gufo::quant::exl3::ReferenceScratch exl3{};
    if (gufo::hip::IsExl3(model->GetWeights().output)) {
      exl3.rotated_elements = hidden;
      exl3.weight_elements = std::size_t{64} << 20U;
      HipCheck(hipMalloc(&exl3.rotated, exl3.rotated_elements * sizeof(float)),
               "hipMalloc");
      HipCheck(hipMalloc(&exl3.weight, exl3.weight_elements * sizeof(float)),
               "hipMalloc");
    }
    std::vector<float> logits(n_vocab);
    std::vector<std::uint16_t> row(RowWidth(n_vocab));

    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    Require(bool(out), "cannot write " + out_path);
    const std::uint32_t header_ctx = static_cast<std::uint32_t>(n_ctx);
    const std::int32_t header_vocab = static_cast<std::int32_t>(n_vocab);
    const std::int32_t header_chunks = static_cast<std::int32_t>(chunks);
    out.write("_logits_", 8);
    out.write(reinterpret_cast<const char*>(&header_ctx), 4);
    out.write(reinterpret_cast<const char*>(&header_vocab), 4);
    out.write(reinterpret_cast<const char*>(&header_chunks), 4);
    for (const Token t : tokens) {
      const auto v = static_cast<std::int32_t>(t);
      out.write(reinterpret_cast<const char*>(&v), 4);
    }
    const std::size_t first = n_ctx / 2;
    std::size_t checks = 0;
    for (std::size_t c = 0; c < chunks; ++c) {
      const std::span<const Token> chunk(tokens.data() + (c * n_ctx), n_ctx);
      executor.Reset();
      executor.SetPromptHiddenCapture(true);
      (void)executor.ForwardPromptBatch(chunk.first(n_ctx - 1));
      const auto last = executor.CopyLastLogits();
      const std::vector<float> production(last.begin(), last.end());
      const auto states = executor.GetPromptHiddenStates();
      Require(states.size() == (n_ctx - 1) * hidden, "hidden capture size");
      for (std::size_t p = first; p + 1 < n_ctx; ++p) {
        HipCheck(hipMemcpyAsync(d_hidden, states.data() + (p * hidden),
                                hidden * sizeof(float), hipMemcpyHostToDevice,
                                stream),
                 "upload hidden");
        gufo::hip::LaunchQwenOutputHead(
            model->GetWeights(), d_hidden, d_normed, d_logits, stream,
            exl3.weight != nullptr ? &exl3 : nullptr);
        HipCheck(
            hipMemcpyAsync(logits.data(), d_logits, n_vocab * sizeof(float),
                           hipMemcpyDeviceToHost, stream),
            "download logits");
        HipCheck(hipStreamSynchronize(stream), "head sync");
        if (p + 2 == n_ctx) {
          Require(std::ranges::equal(logits, production),
                  "head replay differs from the prefill last-row logits");
          ++checks;
        }
        EncodeRow(logits, row);
        out.write(reinterpret_cast<const char*>(row.data()),
                  static_cast<std::streamsize>(row.size() * 2));
      }
      executor.SetPromptHiddenCapture(false);
      std::cerr << "chunk " << c << " done" << std::endl;
    }
    out.close();
    Require(bool(out), "write failed");
    (void)hipFree(d_hidden);
    (void)hipFree(d_normed);
    (void)hipFree(d_logits);
    (void)hipFree(exl3.rotated);
    (void)hipFree(exl3.weight);
    (void)hipStreamDestroy(stream);
    std::cout << "RESULT rq_kld_base chunks=" << chunks << " n_ctx=" << n_ctx
              << " head_replay_checks=" << checks
              << " bytes=" << std::filesystem::file_size(out_path) << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
}
