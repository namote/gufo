// RQ R0 quality harness collector (plan/rotated-quant/W2-BRIEF.md): teacher
// forces the token windows of an evaluation corpus through one Qwen27B
// execution path and stores, per scored position, the top-k token ids and
// logits, the full-vocabulary logsumexp, the logit of the next corpus token
// and, against a reference run, the logits at the reference's top-k ids.
// tools/qwen27b/rq_quality/metrics.py turns two such runs into paired NLL,
// truncated KL and top-1 agreement; tools/qwen27b/rq_quality/logits_io.py
// owns the record layout.
//
// Paths (every one starts each window from a reset executor):
//   prefill  ForwardPromptBatch over the window with prompt hidden capture;
//            each row's logits use the production head
//            (LaunchQwenOutputHead: RMSNorm, the head-input rotation of a
//            rotated GGUF, the output GEMV), as prefill does for its last
//            row. The last row is checked bit for bit against
//            CopyLastLogits.
//   decode   ForwardToken per position (the production AR step).
//   verify   SaveState, ForwardVerificationChunk of --verify-rows tokens with
//            captured logits, FinishVerification: the DFlash2 verify pass
//            with every draft accepted.
// --prefix N first prefills N tokens of each window with ForwardPromptBatch
// (unscored) for the decode and verify paths.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/exl3_dispatch.hpp"
#include "src/models/qwen/hip/output_head.hpp"

namespace json = gufo::json;
using Token = gufo::tokenization::TokenId;

namespace {
using Clock = std::chrono::steady_clock;

constexpr const char* kSchema = "gufo.rq-logits.v1";

void Require(bool ok, const std::string& error) {
  if (!ok)
    throw std::runtime_error(error);
}

void HipCheck(hipError_t status, const char* what) {
  Require(status == hipSuccess,
          std::string(what) + ": " + hipGetErrorString(status));
}

double Ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

std::string ReadFile(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  Require(bool(file), "cannot read " + path.string());
  return {std::istreambuf_iterator<char>(file), {}};
}

std::string Sha256(const std::string& bytes) {
  return gufo::crypto::Sha256Hex(std::span(
      reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
}

struct Window {
  std::string id, category;
  std::vector<Token> tokens;
};

struct Corpus {
  std::vector<Window> windows;
  std::string index_sha256, tokens_sha256;
};

// windows.json (tools/qwen27b/rq_quality/corpus.py) indexes a little-endian
// int32 token file.
Corpus ReadCorpus(const std::filesystem::path& index_path) {
  const std::string index_text = ReadFile(index_path);
  const auto index = json::parse(index_text);
  Require(index.member_str("schema") == "gufo.rq-corpus-windows.v1",
          "unexpected windows schema in " + index_path.string());
  const auto tokens_path =
      index_path.parent_path() / index.member_str("tokens_file");
  const std::string token_bytes = ReadFile(tokens_path);
  Corpus corpus;
  corpus.index_sha256 = Sha256(index_text);
  corpus.tokens_sha256 = Sha256(token_bytes);
  Require(corpus.tokens_sha256 == index.member_str("tokens_sha256"),
          "token file hash differs from windows.json");
  Require(token_bytes.size() % 4 == 0, "token file is not int32");
  const auto* windows = index.find("windows");
  Require(windows != nullptr && windows->is_array(), "windows.json: windows");
  for (const auto& entry : windows->items()) {
    Window window{entry.member_str("id"), entry.member_str("category"), {}};
    const std::size_t offset = entry.member_size("offset");
    const std::size_t length = entry.member_size("length");
    Require(length >= 2 && (offset + length) * 4 <= token_bytes.size(),
            "window " + window.id + " is out of range");
    window.tokens.resize(length);
    for (std::size_t i = 0; i < length; ++i) {
      std::int32_t value = 0;
      std::memcpy(&value, token_bytes.data() + ((offset + i) * 4), 4);
      Require(value >= 0, "negative token id");
      window.tokens[i] = static_cast<Token>(value);
    }
    corpus.windows.push_back(std::move(window));
  }
  Require(!corpus.windows.empty(), "no windows");
  return corpus;
}

std::uint64_t Key(std::uint32_t window, std::uint32_t position) {
  return (std::uint64_t{window} << 32U) | position;
}

// The reference run's top-k ids per scored position.
struct Reference {
  std::string meta_path, bin_sha256;
  std::size_t top_k{0};
  std::unordered_map<std::uint64_t, std::vector<std::int32_t>> ids;
};

// Record layout (little-endian, packed); logits_io.py RecordDtype mirrors it.
//   int32 window, int32 position, int32 target, float32 target_logit,
//   float64 logsumexp, int32 ids[k], float32 logits[k], float32 ref_logits[k]
std::size_t RecordBytes(std::size_t k) {
  return 24 + (12 * k);
}

Reference ReadReference(const std::string& prefix, const Corpus& corpus) {
  Reference reference;
  reference.meta_path = prefix + ".json";
  const auto meta = json::parse(ReadFile(reference.meta_path));
  Require(meta.member_str("schema") == kSchema, "reference schema differs");
  Require(meta.member_str("windows_tokens_sha256") == corpus.tokens_sha256,
          "reference was collected on different corpus windows");
  reference.top_k = meta.member_size("top_k");
  const std::string bin = ReadFile(prefix + ".bin");
  reference.bin_sha256 = Sha256(bin);
  Require(reference.bin_sha256 == meta.member_str("bin_sha256"),
          "reference .bin hash differs from its .json");
  const std::size_t record = RecordBytes(reference.top_k);
  Require(reference.top_k > 0 && bin.size() % record == 0,
          "reference .bin size is not a whole number of records");
  for (std::size_t offset = 0; offset < bin.size(); offset += record) {
    std::int32_t window = 0, position = 0;
    std::memcpy(&window, bin.data() + offset, 4);
    std::memcpy(&position, bin.data() + offset + 4, 4);
    std::vector<std::int32_t> ids(reference.top_k);
    std::memcpy(ids.data(), bin.data() + offset + 24, 4 * reference.top_k);
    reference.ids.emplace(Key(static_cast<std::uint32_t>(window),
                              static_cast<std::uint32_t>(position)),
                          std::move(ids));
  }
  return reference;
}

class Writer {
public:
  Writer(const std::string& path, std::size_t k)
      : out_(path, std::ios::binary | std::ios::trunc), k_(k) {
    Require(bool(out_), "cannot write " + path);
  }

  // `logits` predicts corpus token `target` after `position`.
  void Add(std::uint32_t window, std::uint32_t position, Token target,
           std::span<const float> logits, const Reference* reference) {
    double top = -std::numeric_limits<double>::infinity();
    for (const float v : logits) {
      Require(std::isfinite(v), "non-finite logit");
      top = std::max(top, static_cast<double>(v));
    }
    double sum = 0;
    for (const float v : logits)
      sum += std::exp(static_cast<double>(v) - top);
    const double logsumexp = top + std::log(sum);

    order_.resize(logits.size());
    std::iota(order_.begin(), order_.end(), 0);
    // Descending logit, ties to the lower id, as a full-vocabulary argmax.
    std::partial_sort(order_.begin(), order_.begin() + k_, order_.end(),
                      [&](std::int32_t a, std::int32_t b) {
                        return logits[a] > logits[b] ||
                               (logits[a] == logits[b] && a < b);
                      });
    record_.assign(RecordBytes(k_), std::byte{0});
    auto put = [&](std::size_t offset, const void* value, std::size_t size) {
      std::memcpy(record_.data() + offset, value, size);
    };
    const auto w = static_cast<std::int32_t>(window);
    const auto p = static_cast<std::int32_t>(position);
    const auto t = static_cast<std::int32_t>(target);
    Require(t >= 0 && static_cast<std::size_t>(t) < logits.size(),
            "target outside the vocabulary");
    const float target_logit = logits[t];
    put(0, &w, 4);
    put(4, &p, 4);
    put(8, &t, 4);
    put(12, &target_logit, 4);
    put(16, &logsumexp, 8);
    const std::size_t ids_at = 24, logits_at = 24 + (4 * k_),
                      ref_at = 24 + (8 * k_);
    std::span<const std::int32_t> ref_ids(order_.data(), k_);
    if (reference != nullptr) {
      const auto found = reference->ids.find(Key(window, position));
      Require(found != reference->ids.end(),
              "position " + std::to_string(position) + " of window " +
                  std::to_string(window) + " is not in the reference");
      ref_ids = found->second;
    }
    for (std::size_t i = 0; i < k_; ++i) {
      const std::int32_t id = order_[i];
      const float logit = logits[id];
      const float ref_logit = logits[ref_ids[i]];
      put(ids_at + (4 * i), &id, 4);
      put(logits_at + (4 * i), &logit, 4);
      put(ref_at + (4 * i), &ref_logit, 4);
    }
    out_.write(reinterpret_cast<const char*>(record_.data()),
               static_cast<std::streamsize>(record_.size()));
    hasher_.Update(std::span(
        reinterpret_cast<const std::uint8_t*>(record_.data()), record_.size()));
    ++records_;
  }

  std::string Finish() {
    out_.close();
    Require(bool(out_), "write failed");
    return hasher_.FinishHex();
  }
  [[nodiscard]] std::size_t Records() const noexcept { return records_; }

private:
  std::ofstream out_;
  std::size_t k_;
  std::vector<std::int32_t> order_;
  std::vector<std::byte> record_;
  gufo::crypto::Sha256Hasher hasher_;
  std::size_t records_{0};
};

// Production head over captured prefill hidden rows.
class Head {
public:
  Head(const gufo::models::QwenModelWeights& weights, std::size_t hidden,
       std::size_t vocab)
      : weights_(weights), hidden_(hidden), vocab_(vocab), logits_(vocab) {
    HipCheck(hipStreamCreate(&stream_), "hipStreamCreate");
    HipCheck(hipMalloc(&d_hidden_, hidden * sizeof(float)), "hipMalloc");
    HipCheck(hipMalloc(&d_normed_, hidden * sizeof(float)), "hipMalloc");
    HipCheck(hipMalloc(&d_logits_, vocab * sizeof(float)), "hipMalloc");
    // An EXL3 head (gufo.gguf sidecar) needs a projection workspace, sized
    // like the DFlash2 executor's; GGUF heads keep the null workspace.
    if (gufo::hip::IsExl3(weights.output)) {
      exl3_.rotated_elements = hidden;
      exl3_.weight_elements = std::size_t{64} << 20U;
      HipCheck(
          hipMalloc(&exl3_.rotated, exl3_.rotated_elements * sizeof(float)),
          "hipMalloc");
      HipCheck(hipMalloc(&exl3_.weight, exl3_.weight_elements * sizeof(float)),
               "hipMalloc");
    }
  }
  ~Head() {
    (void)hipFree(d_hidden_);
    (void)hipFree(d_normed_);
    (void)hipFree(d_logits_);
    (void)hipFree(exl3_.rotated);
    (void)hipFree(exl3_.weight);
    (void)hipStreamDestroy(stream_);
  }
  Head(const Head&) = delete;
  Head& operator=(const Head&) = delete;

  std::span<const float> Logits(std::span<const float> hidden_row) {
    Require(hidden_row.size() == hidden_, "hidden row width");
    HipCheck(
        hipMemcpyAsync(d_hidden_, hidden_row.data(), hidden_ * sizeof(float),
                       hipMemcpyHostToDevice, stream_),
        "upload hidden");
    gufo::hip::LaunchQwenOutputHead(weights_, d_hidden_, d_normed_, d_logits_,
                                    stream_,
                                    exl3_.weight != nullptr ? &exl3_ : nullptr);
    HipCheck(hipMemcpyAsync(logits_.data(), d_logits_, vocab_ * sizeof(float),
                            hipMemcpyDeviceToHost, stream_),
             "download logits");
    HipCheck(hipStreamSynchronize(stream_), "head sync");
    return logits_;
  }

private:
  const gufo::models::QwenModelWeights& weights_;
  std::size_t hidden_, vocab_;
  std::vector<float> logits_;
  hipStream_t stream_{nullptr};
  float* d_hidden_{nullptr};
  float* d_normed_{nullptr};
  float* d_logits_{nullptr};
  gufo::quant::exl3::ReferenceScratch exl3_{};
};

struct DeviceMemory {
  std::size_t peak_used{0}, total{0};
  void Sample() {
    std::size_t free = 0;
    if (hipMemGetInfo(&free, &total) == hipSuccess)
      peak_used = std::max(peak_used, total - free);
  }
};

std::uint64_t FileBytes(const std::string& path) {
  std::error_code error;
  const auto size = std::filesystem::file_size(path, error);
  return error ? 0 : size;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string model_path, windows_path, out_prefix, reference_prefix,
        path_name, label, toolchain;
    std::size_t top_k = 64, prefix = 0, verify_rows = 8,
                window_limit = std::numeric_limits<std::size_t>::max(),
                position_limit = std::numeric_limits<std::size_t>::max();
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto value = [&]() -> std::string {
        Require(i + 1 < argc, arg + " needs a value");
        return argv[++i];
      };
      if (arg == "--windows")
        windows_path = value();
      else if (arg == "--out")
        out_prefix = value();
      else if (arg == "--reference")
        reference_prefix = value();
      else if (arg == "--path")
        path_name = value();
      else if (arg == "--top-k")
        top_k = std::stoul(value());
      else if (arg == "--prefix")
        prefix = std::stoul(value());
      else if (arg == "--verify-rows")
        verify_rows = std::stoul(value());
      else if (arg == "--windows-limit")
        window_limit = std::stoul(value());
      else if (arg == "--positions-limit")
        position_limit = std::stoul(value());
      else if (arg == "--label")
        label = value();
      else if (arg == "--toolchain")
        toolchain = value();
      else if (model_path.empty() && !arg.starts_with("--"))
        model_path = arg;
      else
        throw std::invalid_argument("unknown argument " + arg);
    }
    Require(!model_path.empty() && !windows_path.empty() &&
                !out_prefix.empty() &&
                (path_name == "prefill" || path_name == "decode" ||
                 path_name == "verify"),
            "Usage: qwen27b_rq_logits MODEL.gguf --windows windows.json "
            "--path prefill|decode|verify --out PREFIX [--reference PREFIX] "
            "[--top-k 64] [--prefix N] [--verify-rows 8] [--windows-limit N] "
            "[--positions-limit N] [--label TEXT] [--toolchain TEXT]");
    Require(top_k >= 1 && top_k <= 1024, "--top-k must be 1..1024");
    Require(verify_rows >= 1 && verify_rows <= 8, "--verify-rows must be 1..8");
    Require(path_name != "prefill" || prefix == 0,
            "--prefix applies to the decode and verify paths");

    const auto corpus = ReadCorpus(windows_path);
    std::unique_ptr<Reference> reference;
    if (!reference_prefix.empty()) {
      reference =
          std::make_unique<Reference>(ReadReference(reference_prefix, corpus));
      Require(reference->top_k == top_k, "reference --top-k differs");
    }

    DeviceMemory memory;
    memory.Sample();
    const auto load_start = Clock::now();
    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
    Require(bool(reader), error);
    auto model = gufo::hip::QwenGpuModel::CreateFromGguf(
        std::shared_ptr<const gufo::core::GgufReader>(std::move(reader)),
        &error);
    Require(bool(model), error);
    const double load_ms = Ms(load_start);
    const auto& config = model->GetConfig();
    Require(top_k <= config.vocab_size, "--top-k exceeds the vocabulary");
    std::size_t longest = 0;
    for (const auto& window : corpus.windows)
      longest = std::max(longest, window.tokens.size());
    Require(prefix + 1 < longest || path_name == "prefill",
            "--prefix leaves nothing to score");
    const auto context =
        static_cast<std::uint32_t>((longest + 1 + 255) / 256 * 256);
    gufo::hip::QwenGpuExecutor executor(model, context);
    memory.Sample();
    Head head(model->GetWeights(), config.hidden_size, config.vocab_size);

    Writer writer(out_prefix + ".bin", top_k);
    auto windows_meta = json::Value::array();
    std::size_t head_checks = 0;
    const auto total_start = Clock::now();
    for (std::size_t w = 0; w < corpus.windows.size() && w < window_limit;
         ++w) {
      const auto& tokens = corpus.windows[w].tokens;
      const auto wi = static_cast<std::uint32_t>(w);
      // Position p predicts tokens[p + 1]; the last token has no target.
      const std::size_t first = path_name == "prefill" ? 0 : prefix;
      const std::size_t end =
          std::min(tokens.size() - 1,
                   first + std::min(position_limit, tokens.size() - 1));
      Require(first < end,
              "window " + corpus.windows[w].id + " has no scored position");
      executor.Reset();
      const auto start = Clock::now();
      double forward_ms = 0;
      if (path_name == "prefill") {
        executor.SetPromptHiddenCapture(true);
        auto t = Clock::now();
        (void)executor.ForwardPromptBatch(std::span(tokens).first(end));
        forward_ms += Ms(t);
        const auto last = executor.CopyLastLogits();
        const std::vector<float> production(last.begin(), last.end());
        const auto hidden = executor.GetPromptHiddenStates();
        Require(hidden.size() == end * config.hidden_size,
                "prompt hidden capture has the wrong size");
        for (std::size_t p = first; p < end; ++p) {
          const auto logits = head.Logits(
              hidden.subspan(p * config.hidden_size, config.hidden_size));
          if (p + 1 == end) {
            // The production prefill head computed this row itself.
            Require(std::ranges::equal(logits, production),
                    "head replay differs from the prefill last-row logits");
            ++head_checks;
          }
          writer.Add(wi, static_cast<std::uint32_t>(p), tokens[p + 1], logits,
                     reference.get());
        }
        executor.SetPromptHiddenCapture(false);
      } else {
        if (prefix > 0) {
          auto t = Clock::now();
          (void)executor.ForwardPromptBatch(std::span(tokens).first(prefix));
          forward_ms += Ms(t);
        }
        if (path_name == "decode") {
          for (std::size_t p = first; p < end; ++p) {
            auto t = Clock::now();
            (void)executor.ForwardToken(tokens[p],
                                        static_cast<std::uint32_t>(p));
            const auto logits = executor.CopyLastLogits();
            forward_ms += Ms(t);
            writer.Add(wi, static_cast<std::uint32_t>(p), tokens[p + 1], logits,
                       reference.get());
          }
        } else {
          for (std::size_t p = first; p < end; p += verify_rows) {
            const std::size_t rows = std::min(verify_rows, end - p);
            const auto position = static_cast<std::uint32_t>(p);
            auto t = Clock::now();
            executor.SaveState(position);
            (void)executor.ForwardVerificationChunk(
                std::span(tokens).subspan(p, rows), position, true);
            forward_ms += Ms(t);
            for (std::size_t r = 0; r < rows; ++r)
              writer.Add(wi, static_cast<std::uint32_t>(p + r),
                         tokens[p + r + 1], executor.CopyVerificationLogits(r),
                         reference.get());
            executor.FinishVerification();
          }
        }
      }
      memory.Sample();
      auto entry = json::Value::object();
      entry["index"] = w;
      entry["id"] = corpus.windows[w].id;
      entry["category"] = corpus.windows[w].category;
      entry["tokens"] = tokens.size();
      entry["first_position"] = first;
      entry["scored"] = end - first;
      entry["forward_ms"] = forward_ms;
      entry["wall_ms"] = Ms(start);
      windows_meta.push_back(std::move(entry));
      std::cerr << "window " << corpus.windows[w].id << " scored "
                << (end - first) << " in " << Ms(start) << " ms" << std::endl;
    }

    auto meta = json::Value::object();
    meta["schema"] = kSchema;
    meta["label"] = label;
    meta["toolchain"] = toolchain;
    meta["compiler"] = std::string(__VERSION__);
    meta["hip_version"] = std::to_string(HIP_VERSION_MAJOR) + "." +
                          std::to_string(HIP_VERSION_MINOR) + "." +
                          std::to_string(HIP_VERSION_PATCH);
    meta["model"] = model_path;
    meta["model_file_bytes"] = FileBytes(model_path);
    meta["path"] = path_name;
    meta["top_k"] = top_k;
    meta["prefix"] = prefix;
    meta["verify_rows"] = path_name == "verify" ? verify_rows : 0;
    meta["context"] = std::size_t{context};
    meta["max_prompt_batch"] = std::size_t{executor.GetMaxPromptBatch()};
    meta["windows_file"] = windows_path;
    meta["windows_index_sha256"] = corpus.index_sha256;
    meta["windows_tokens_sha256"] = corpus.tokens_sha256;
    meta["record_bytes"] = RecordBytes(top_k);
    meta["records"] = writer.Records();
    meta["bin_sha256"] = writer.Finish();
    if (reference) {
      auto ref = json::Value::object();
      ref["meta"] = reference->meta_path;
      ref["bin_sha256"] = reference->bin_sha256;
      meta["reference"] = std::move(ref);
    } else {
      meta["reference"] = nullptr;
    }
    meta["head_replay_checks"] = head_checks;
    meta["load_ms"] = load_ms;
    meta["total_ms"] = Ms(total_start);
    meta["model_resident_bytes"] = model->GetResidentBytes();
    meta["hip_peak_used_bytes"] = memory.peak_used;
    meta["hip_total_bytes"] = memory.total;
    meta["windows"] = std::move(windows_meta);
    std::ofstream out(out_prefix + ".json", std::ios::trunc);
    Require(bool(out), "cannot write " + out_prefix + ".json");
    out << meta.dump() << "\n";
    Require(bool(out), "write failed");
    std::cout << "RESULT rq_logits path=" << path_name
              << " records=" << writer.Records()
              << " total_ms=" << Ms(total_start) << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
}
