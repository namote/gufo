// Flash-Next R0 quality collector. Shares the gufo.rq-logits.v1 layout
// with qwen27b/rq_logits.cpp and tools/qwen27b/rq_quality/logits_io.py.
// Teacher-forced positions use the production trunk and full output head.
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
#include "src/core/quant/exl3/layout.hpp"
#include "src/core/session_mode.hpp"
#include "src/models/qwen/native_pack.hpp"
#include "src/models/qwen38_flash_next/exl3_weights.hpp"
#include "src/models/qwen38_flash_next/native_metadata.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/runtime/device_model.hpp"
#include "src/models/qwen38_flash_next/runtime/executor.hpp"

namespace json = gufo::json;
namespace q = gufo::models::qwen38_flash_next;
using Token = std::int32_t;

namespace {
using Clock = std::chrono::steady_clock;

constexpr const char* kSchema = "gufo.rq-logits.v1";

void Require(bool ok, const std::string& error) {
  if (!ok)
    throw std::runtime_error(error);
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
    Require(length >= 2 && length <= 4096 && offset <= token_bytes.size() / 4 &&
                length <= token_bytes.size() / 4 - offset,
            "window " + window.id + " is out of range");
    window.tokens.resize(length);
    for (std::size_t i = 0; i < length; ++i) {
      std::int32_t value = 0;
      std::memcpy(&value, token_bytes.data() + ((offset + i) * 4), 4);
      Require(value >= 0 && value < 248320, "token id outside vocabulary");
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
  Require(reference.top_k > 0 && reference.top_k <= 1024,
          "invalid reference top-k");
  const std::string bin = ReadFile(prefix + ".bin");
  reference.bin_sha256 = Sha256(bin);
  Require(reference.bin_sha256 == meta.member_str("bin_sha256"),
          "reference .bin hash differs from its .json");
  const std::size_t record = RecordBytes(reference.top_k);
  Require(!bin.empty() && bin.size() % record == 0 &&
              meta.member_size("record_bytes") == record &&
              meta.member_size("records") == bin.size() / record,
          "reference .bin size differs from metadata");
  for (std::size_t offset = 0; offset < bin.size(); offset += record) {
    std::int32_t window = 0, position = 0;
    std::memcpy(&window, bin.data() + offset, 4);
    std::memcpy(&position, bin.data() + offset + 4, 4);
    std::vector<std::int32_t> ids(reference.top_k);
    std::memcpy(ids.data(), bin.data() + offset + 24, 4 * reference.top_k);
    Require(window >= 0 &&
                static_cast<std::size_t>(window) < corpus.windows.size() &&
                position >= 0 &&
                static_cast<std::size_t>(position) + 1 <
                    corpus.windows[window].tokens.size(),
            "reference position outside corpus");
    for (const auto id : ids)
      Require(id >= 0 && id < 248320, "reference id outside vocabulary");
    const auto inserted =
        reference.ids.emplace(Key(static_cast<std::uint32_t>(window),
                                  static_cast<std::uint32_t>(position)),
                              std::move(ids));
    Require(inserted.second, "duplicate reference position");
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

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string model_path, windows_path, out_prefix, reference_prefix,
        path_name, label, toolchain;
    std::size_t top_k = 256, prefix = 2048, batch = 512, window_limit = 8,
                position_limit = std::numeric_limits<std::size_t>::max();
    bool kvarn = false;
    int kvarn_bits = 4;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto value = [&]() -> std::string {
        Require(i + 1 < argc, arg + " needs a value");
        return argv[++i];
      };
      const auto number = [&]() -> std::size_t {
        const auto text = value();
        Require(!text.empty() &&
                    text.find_first_not_of("0123456789") == std::string::npos,
                arg + " requires a nonnegative integer");
        return std::stoull(text);
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
        top_k = number();
      else if (arg == "--prefix")
        prefix = number();
      else if (arg == "--batch")
        batch = number();
      else if (arg == "--windows-limit")
        window_limit = number();
      else if (arg == "--positions-limit")
        position_limit = number();
      else if (arg == "--kvarn-k6v6") {
        kvarn = true;
        kvarn_bits = 6;
      } else if (arg == "--kvarn")
        kvarn = true;
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
                (path_name == "prefill" || path_name == "decode"),
            "Usage: qwen38_flash_next_rq_logits MODEL.gguf "
            "--windows windows.json --path prefill|decode --out PREFIX "
            "[--reference PREFIX] [--top-k 256] [--prefix 2048] "
            "[--positions-limit 512] [--windows-limit 8] [--batch 512] "
            "[--kvarn|--kvarn-k6v6] [--label TEXT] [--toolchain TEXT]");
    Require(top_k >= 1 && top_k <= 1024, "--top-k must be 1..1024");
    Require(batch >= 1 && batch <= 4096, "--batch must be 1..4096");
    Require(window_limit > 0 && position_limit > 0, "empty collection");
    if (path_name == "decode" &&
        position_limit == std::numeric_limits<std::size_t>::max())
      position_limit = 512;
    const auto corpus = ReadCorpus(windows_path);
    std::size_t longest = 0;
    for (std::size_t w = 0; w < corpus.windows.size() && w < window_limit;
         ++w) {
      const auto length = corpus.windows[w].tokens.size();
      Require(path_name == "prefill" || prefix < length - 1,
              "decode prefix leaves nothing to score");
      longest = std::max(longest, length);
    }
    Require(longest <= std::numeric_limits<std::uint32_t>::max() - 255,
            "context too long");
    const auto context =
        static_cast<std::uint32_t>((longest + 255) / 256 * 256);
    std::unique_ptr<Reference> reference;
    if (!reference_prefix.empty()) {
      Require(reference_prefix != out_prefix,
              "output would overwrite reference");
      reference =
          std::make_unique<Reference>(ReadReference(reference_prefix, corpus));
      Require(reference->top_k == top_k, "reference --top-k differs");
      const auto meta = json::parse(ReadFile(reference->meta_path));
      Require(meta.member_str("path") == path_name, "reference path differs");
    }

    const auto load_start = Clock::now();
    std::string error;
    auto reader = q::OpenArtifactMetadata(model_path, &error);
    Require(bool(reader), error);
    const bool exl3 =
        reader->FindMetadata(gufo::quant::exl3::kCheckpointKey) != nullptr;
    auto weights = exl3 ? q::LoadExl3Model(model_path, *reader, &error)
                        : q::ModelWeights::Bind(*reader, &error);
    Require(bool(weights), error);
    const auto& c = weights->config;
    Require(c.vocab_size == 248320, "unexpected Flash-Next vocabulary");
    std::unique_ptr<q::NgramTable> ngram;
    if (c.ple_layer >= 0) {
      ngram = q::NgramTable::ForModel(model_path, *reader, *weights, &error);
      Require(bool(ngram), error);
    }
    auto device = q::rocm::DeviceModel::Upload(*weights, *reader, nullptr,
                                               nullptr, &error);
    Require(bool(device), error);
    q::rocm::Executor::Options options;
    options.max_batch = static_cast<std::uint32_t>(batch);
    options.max_logit_rows = static_cast<std::uint32_t>(batch);
    options.max_context = context;
    options.kvarn = kvarn;
    options.kvarn_bits = kvarn_bits;
    auto executor =
        q::rocm::Executor::Create(*device, ngram.get(), options, &error);
    Require(bool(executor), error);
    auto session = executor->CreateSession(
        gufo::core::SessionMode::kAutoregressive, context, &error);
    Require(bool(session), error);
    gufo::models::qwen::ValidateNativeSources(*reader);
    const double load_ms = Ms(load_start);
    std::filesystem::remove(out_prefix + ".json");
    Writer writer(out_prefix + ".bin", top_k);
    auto windows_meta = json::Value::array();
    std::vector<float> logits(batch * c.vocab_size),
        replay(batch * c.vocab_size);
    std::size_t head_checks = 0;
    const auto total_start = Clock::now();
    for (std::size_t w = 0; w < corpus.windows.size() && w < window_limit;
         ++w) {
      const auto& tokens = corpus.windows[w].tokens;
      const auto first = path_name == "prefill" ? 0 : prefix;
      const auto end =
          first + std::min(position_limit, tokens.size() - 1 - first);
      session->Reset();
      const auto start = Clock::now();
      double forward_ms = 0;
      for (std::size_t p = 0; p < end;) {
        const bool scored = p >= first;
        const bool decode = scored && path_name == "decode";
        const auto n = std::min(decode ? std::size_t{1} : batch,
                                (scored ? end : first) - p);
        const auto rows = static_cast<std::uint32_t>(scored ? n : 0);
        const auto t = Clock::now();
        const auto mode = decode ? q::rocm::Executor::ForwardMode::kDecode
                                 : q::rocm::Executor::ForwardMode::kPrefill;
        Require(
            executor->Forward(*session, std::span(tokens).subspan(p, n), rows,
                              scored ? logits.data() : nullptr, mode, &error),
            error);
        forward_ms += Ms(t);
        if (scored) {
          // Same row width/dispatch as production, including EXL3 and the HC
          // mixer. Check every returned row before any scratch is reused.
          auto check = std::span(replay).first(n * c.vocab_size);
          Require(executor->ReplayOutputHead(static_cast<std::uint32_t>(n),
                                             check, &error),
                  error);
          Require(
              std::memcmp(logits.data(), check.data(), check.size_bytes()) == 0,
              "head replay differs from production logits");
          ++head_checks;
          for (std::size_t r = 0; r < n; ++r)
            writer.Add(
                static_cast<std::uint32_t>(w),
                static_cast<std::uint32_t>(p + r), tokens[p + r + 1],
                std::span(logits).subspan(r * c.vocab_size, c.vocab_size),
                reference.get());
        }
        p += n;
      }
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
                << end - first << " in " << Ms(start) << " ms\n";
    }
    auto meta = json::Value::object();
    meta["schema"] = kSchema;
    meta["model"] = model_path;
    meta["backend"] = exl3 ? "exl3" : "gguf";
    meta["label"] = label;
    meta["toolchain"] = toolchain;
    meta["compiler"] = std::string(__VERSION__);
    meta["hip_version"] = std::to_string(HIP_VERSION_MAJOR) + "." +
                          std::to_string(HIP_VERSION_MINOR) + "." +
                          std::to_string(HIP_VERSION_PATCH);
    meta["path"] = path_name;
    meta["kv_cache"] =
        kvarn ? (kvarn_bits == 6 ? "kvarn-k6v6" : "kvarn-k4v4") : "fp16";
    meta["vocab_size"] = std::size_t{c.vocab_size};
    meta["top_k"] = top_k;
    meta["prefix"] = path_name == "prefill" ? 0 : prefix;
    meta["context"] = std::size_t{context};
    meta["max_prompt_batch"] = batch;
    meta["windows_file"] = windows_path;
    meta["windows_index_sha256"] = corpus.index_sha256;
    meta["windows_tokens_sha256"] = corpus.tokens_sha256;
    meta["record_bytes"] = RecordBytes(top_k);
    meta["records"] = writer.Records();
    meta["bin_sha256"] = writer.Finish();
    meta["reference"] = nullptr;
    if (reference) {
      auto ref = json::Value::object();
      ref["meta"] = reference->meta_path;
      ref["bin_sha256"] = reference->bin_sha256;
      meta["reference"] = std::move(ref);
    }
    meta["head_replay_checks"] = head_checks;
    meta["load_ms"] = load_ms;
    meta["total_ms"] = Ms(total_start);
    meta["windows"] = std::move(windows_meta);
    std::ofstream out(out_prefix + ".json", std::ios::trunc);
    Require(bool(out), "cannot write metadata");
    out << meta.dump() << '\n';
    Require(bool(out), "metadata write failed");
    std::cout << "RESULT fn_rq_logits records=" << writer.Records() << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
