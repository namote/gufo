// Dumps trunk prefill logits and optionally compares them with the F32 oracle.
// Use the production CLI for generation and the session test for replay.
//
// gpu_probe --model FIRST_SHARD.gguf --prompt TEXT [--reference]
//           [--batch T] [--context N] [--yarn-factor F] [--dump logits.bin]
//           [--logit-rows N] [--nll nll.bin]
// The dump holds the last 64 rows of the final prefill pass, or with
// --logit-rows the last N rows of every pass in position order. --nll
// writes each position's negative log likelihood of the next prompt token
// (float32, tokens - 1 values) and prints the mean.
// Independent predictor oracle: --mtp-model MTP.gguf --mtp-audit
// MTP cost calibration: --mtp-model MTP.gguf --cost-audit C (0 = all)
//                      [--depth N] (default: 0, 4096, 32768)
//                      [--kv-cache-type fp16|kvarn-k4v4] (default fp16;
//                      kvarn-k4v4 prices the serving K4V4 trunk cache)
//                      [--batch T] (default 2048; serving uses 4224)
// Greedy MTP chain trace: --mtp-model MTP.gguf --mtp-trace SUITE.json
//                         --out TRACE.jsonl [--context N] [--max-tokens N]
//                         [--prefill-step N] (default 2048, the HTTP
//                         scheduler's single-session admission passes)
//                         (mtp_trace.cpp; uses the production Session)
// Recurrent-state quality (plan/upstream-sync2/V3-4-BRIEF.md):
//   --prompt-file P --tokens N --state-compare G [--batch B] [--context C]
//   [--kv-cache-type fp16|kvarn-k4v4] [--json OUT]
//   runs the first N prompt tokens in B-token prefill passes and G greedy
//   decode steps on FP32 recurrent state, then teacher-forces the same
//   tokens on FP16 state, and prints full-vocabulary logit RMSE, KL and top-1
//   agreement per pass end and decode step, plus the first diverging greedy
//   step (tests/support/logit_drift.hpp). --recurrent-state fp16 runs every
//   other probe on FP16 state.
// Matched-token decode: --prompt-file P --decode-file C --dump rows.bin
//                       [--decode-ids ids.bin] (prefill P, then feed C's
//                       tokens one at a time; row i predicts the token
//                       after ids[i], so score row i against ids[i + 1])
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/exl3/layout.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/draft_vocab.hpp"
#include "src/models/qwen38_flash_next/exl3_weights.hpp"
#include "src/models/qwen38_flash_next/native_metadata.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/reference.hpp"
#include "src/models/qwen38_flash_next/runtime/device_model.hpp"
#include "src/models/qwen38_flash_next/runtime/executor.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"
#include "tests/models/qwen38_flash_next/mtp_trace.hpp"
#include "tests/support/logit_drift.hpp"

namespace q = gufo::models::qwen38_flash_next;

void AuditMtp(q::rocm::Executor& executor, const q::rocm::DeviceModel& device,
              const q::ModelWeights& weights, const q::MtpWeights& mtp,
              const gufo::tokenization::QwenTokenizer& tokenizer,
              const std::filesystem::path& path);
void AuditMtpCosts(q::rocm::Executor& executor,
                   const gufo::tokenization::QwenTokenizer& tokenizer,
                   std::span<const std::uint32_t> depths,
                   std::span<const std::uint32_t> concurrencies,
                   bool sampled_head);

namespace {

struct Stats {
  std::uint32_t argmax_a;
  std::uint32_t argmax_b;
  double kl;
  double max_abs;
};

Stats Compare(const float* a, const float* b, std::size_t n) {
  std::vector<double> pa(n);
  std::vector<double> pb(n);
  const float ma = *std::max_element(a, a + n);
  const float mb = *std::max_element(b, b + n);
  double sa = 0.0;
  double sb = 0.0;
  double max_abs = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    pa[i] = std::exp(static_cast<double>(a[i] - ma));
    pb[i] = std::exp(static_cast<double>(b[i] - mb));
    sa += pa[i];
    sb += pb[i];
    max_abs = std::max(max_abs, std::fabs(static_cast<double>(a[i] - b[i])));
  }
  double kl = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    pa[i] /= sa;
    pb[i] /= sb;
    kl += pa[i] * (std::log(pa[i] + 1e-30) - std::log(pb[i] + 1e-30));
  }
  return {static_cast<std::uint32_t>(std::max_element(a, a + n) - a),
          static_cast<std::uint32_t>(std::max_element(b, b + n) - b), kl,
          max_abs};
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::string prompt = "The capital of France is";
  std::string dump_path;
  std::uint32_t every_pass_rows = 0;  ///< --logit-rows: rows kept per pass
  std::string nll_path;
  std::string mtp_path;
  std::string decode_text;
  std::string decode_ids_path;
  bool decode = false;
  bool mtp_audit = false;
  bool reference = false;
  bool cost_audit = false;
  bool sampled_head = false;
  std::vector<std::uint32_t> cost_concurrencies;
  std::optional<std::uint32_t> cost_depth;
  std::uint32_t batch = 512;
  std::uint32_t context = 4096;
  bool context_given = false;
  bool batch_given = false;
  std::string trace_suite;
  std::string trace_out;
  std::uint32_t trace_max_tokens = 256;
  std::uint32_t trace_prefill_step = 2048;
  float yarn_factor = 1.0F;
  bool kvarn = false;
  auto recurrent_state = q::rocm::GdnStateFormat::kFp32;
  std::uint32_t state_compare = 0;  ///< --state-compare: greedy steps
  std::uint32_t prompt_tokens = 0;  ///< --tokens: prompt prefix (0: all)
  std::string json_path;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : std::string();
    };
    if (arg == "--model") {
      model_path = next();
    } else if (arg == "--mtp-model") {
      mtp_path = next();
    } else if (arg == "--mtp-trace") {
      trace_suite = next();
    } else if (arg == "--out") {
      trace_out = next();
    } else if (arg == "--prefill-step") {
      const auto value = next();
      const auto [end, ec] = std::from_chars(
          value.data(), value.data() + value.size(), trace_prefill_step);
      if (ec != std::errc{} || end != value.data() + value.size()) {
        std::fprintf(stderr, "--prefill-step requires a nonnegative integer\n");
        return 2;
      }
    } else if (arg == "--max-tokens") {
      const auto value = next();
      const auto [end, ec] = std::from_chars(
          value.data(), value.data() + value.size(), trace_max_tokens);
      if (ec != std::errc{} || end != value.data() + value.size() ||
          trace_max_tokens == 0) {
        std::fprintf(stderr, "--max-tokens requires a positive integer\n");
        return 2;
      }
    } else if (arg == "--mtp-audit") {
      mtp_audit = true;
    } else if (arg == "--sampled-head") {
      sampled_head = true;
    } else if (arg == "--cost-audit") {
      cost_audit = true;
      const auto value = next();
      if (value != "0") {
        const char* begin = value.data();
        const char* finish = begin + value.size();
        do {
          std::uint32_t concurrency = 0;
          const auto [end, ec] = std::from_chars(begin, finish, concurrency);
          if (ec != std::errc{} || (end != finish && *end != ',') ||
              (concurrency != 1 && concurrency != 2 && concurrency != 4 &&
               concurrency != 6 && concurrency != 8) ||
              (end != finish && end + 1 == finish)) {
            std::fprintf(stderr,
                         "--cost-audit requires 0 (all) or a list of "
                         "1,2,4,6,8\n");
            return 2;
          }
          cost_concurrencies.push_back(concurrency);
          begin = end == finish ? finish : end + 1;
        } while (begin != finish);
      }
    } else if (arg == "--depth") {
      const auto value = next();
      std::uint32_t depth = 0;
      const auto [end, ec] =
          std::from_chars(value.data(), value.data() + value.size(), depth);
      if (ec != std::errc{} || end != value.data() + value.size()) {
        std::fprintf(stderr, "--depth requires a nonnegative integer\n");
        return 2;
      }
      cost_depth = depth;
    } else if (arg == "--prompt") {
      prompt = next();
    } else if (arg == "--prompt-file" || arg == "--decode-file") {
      const auto path = next();
      std::ifstream in(path, std::ios::binary);
      if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return 2;
      }
      std::string text((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
      if (in.bad()) {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        return 2;
      }
      (arg == "--prompt-file" ? prompt : decode_text) = std::move(text);
      decode |= arg == "--decode-file";
    } else if (arg == "--decode-ids") {
      decode_ids_path = next();
    } else if (arg == "--reference") {
      reference = true;
    } else if (arg == "--batch" || arg == "--context") {
      const auto value = next();
      auto& count = arg == "--batch" ? batch : context;
      context_given |= arg == "--context";
      batch_given |= arg == "--batch";
      const auto [end, ec] =
          std::from_chars(value.data(), value.data() + value.size(), count);
      if (ec != std::errc{} || end != value.data() + value.size() ||
          count == 0) {
        std::fprintf(stderr, "%s requires a positive integer\n", arg.c_str());
        return 2;
      }
    } else if (arg == "--yarn-factor") {
      const auto value = next();
      const auto [end, ec] = std::from_chars(
          value.data(), value.data() + value.size(), yarn_factor);
      if (ec != std::errc{} || end != value.data() + value.size() ||
          !(yarn_factor >= 1.0F)) {
        std::fprintf(stderr, "--yarn-factor requires a number >= 1\n");
        return 2;
      }
    } else if (arg == "--kv-cache-type") {
      const auto value = next();
      if (value != "fp16" && value != "kvarn-k4v4") {
        std::fprintf(stderr, "--kv-cache-type requires fp16 or kvarn-k4v4\n");
        return 2;
      }
      kvarn = value == "kvarn-k4v4";
    } else if (arg == "--dump") {
      dump_path = next();
    } else if (arg == "--logit-rows") {
      const auto value = next();
      std::uint32_t rows = 0;
      const auto [end, ec] =
          std::from_chars(value.data(), value.data() + value.size(), rows);
      if (ec != std::errc{} || end != value.data() + value.size() ||
          rows == 0) {
        std::fprintf(stderr, "--logit-rows requires a positive integer\n");
        return 2;
      }
      every_pass_rows = rows;
    } else if (arg == "--nll") {
      nll_path = next();
    } else if (arg == "--recurrent-state") {
      const auto value = next();
      if (value != "fp32" && value != "fp16") {
        std::fprintf(stderr, "--recurrent-state requires fp32 or fp16\n");
        return 2;
      }
      recurrent_state = value == "fp16" ? q::rocm::GdnStateFormat::kFp16
                                        : q::rocm::GdnStateFormat::kFp32;
    } else if (arg == "--state-compare" || arg == "--tokens") {
      const auto value = next();
      std::uint32_t count = 0;
      const auto [end, ec] =
          std::from_chars(value.data(), value.data() + value.size(), count);
      if (ec != std::errc{} || end != value.data() + value.size() ||
          count == 0) {
        std::fprintf(stderr, "%s requires a positive integer\n", arg.c_str());
        return 2;
      }
      (arg == "--tokens" ? prompt_tokens : state_compare) = count;
    } else if (arg == "--json") {
      json_path = next();
    } else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }
  if (model_path.empty()) {
    std::fprintf(stderr, "--model is required\n");
    return 2;
  }
  if (!trace_suite.empty() || !trace_out.empty()) {
    if (trace_suite.empty() || trace_out.empty() || mtp_path.empty() ||
        mtp_audit || cost_audit || reference || decode || !dump_path.empty() ||
        !nll_path.empty()) {
      std::fprintf(stderr,
                   "--mtp-trace requires --mtp-model and --out and cannot be "
                   "combined with other probes\n");
      return 2;
    }
    try {
      MtpTraceOptions trace{.model_path = model_path,
                            .mtp_path = mtp_path,
                            .suite_path = trace_suite,
                            .out_path = trace_out,
                            .max_tokens = trace_max_tokens,
                            .prefill_step = trace_prefill_step};
      if (context_given)
        trace.context = context;
      return RunMtpTrace(trace);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "MTP trace failed: %s\n", e.what());
      return 1;
    }
  }
  if ((decode &&
       (decode_text.empty() || reference || mtp_audit || cost_audit)) ||
      (!decode && !decode_ids_path.empty())) {
    std::fprintf(stderr,
                 "--decode-file requires a nonempty continuation and cannot "
                 "be combined with other probes; --decode-ids requires it\n");
    return 2;
  }
  if (state_compare != 0 &&
      (decode || reference || mtp_audit || cost_audit || !dump_path.empty() ||
       !nll_path.empty() || every_pass_rows != 0 ||
       recurrent_state != q::rocm::GdnStateFormat::kFp32)) {
    std::fprintf(stderr,
                 "--state-compare compares fp32 with fp16 state on its own\n");
    return 2;
  }
  if ((prompt_tokens != 0 || !json_path.empty()) && state_compare == 0) {
    std::fprintf(stderr, "--tokens and --json require --state-compare\n");
    return 2;
  }
  if ((mtp_audit || cost_audit) && mtp_path.empty()) {
    std::fprintf(stderr, "MTP audits require --mtp-model\n");
    return 2;
  }
  if (!nll_path.empty() && (every_pass_rows != 0 || !dump_path.empty())) {
    std::fprintf(stderr, "--nll cannot be combined with logit dumps\n");
    return 2;
  }
  if ((every_pass_rows != 0 || !nll_path.empty()) &&
      (reference || decode || cost_audit || mtp_audit)) {
    std::fprintf(stderr,
                 "--logit-rows and --nll apply to plain prefill runs only\n");
    return 2;
  }
  if (cost_audit && (mtp_audit || reference || !dump_path.empty())) {
    std::fprintf(stderr, "--cost-audit cannot be combined with other probes\n");
    return 2;
  }
  if (cost_audit && batch_given) {
    // The 16-row suffix and every width-8 verification row must fit.
    std::uint32_t widest = 8;
    if (!cost_concurrencies.empty())
      widest = *std::ranges::max_element(cost_concurrencies);
    if (batch < std::max<std::uint32_t>(16, 8 * widest)) {
      std::fprintf(stderr, "--batch is too small for --cost-audit\n");
      return 2;
    }
  }
  if ((cost_depth || sampled_head) && !cost_audit) {
    std::fprintf(stderr, "--depth and --sampled-head require --cost-audit\n");
    return 2;
  }
  std::string error;
  auto reader = q::OpenArtifactMetadata(model_path, &error);
  if (!reader) {
    std::fprintf(stderr, "open failed: %s\n", error.c_str());
    return 1;
  }
  const bool exl3 =
      reader->FindMetadata(gufo::quant::exl3::kCheckpointKey) != nullptr;
  auto weights =
      exl3 ? q::LoadExl3Model(model_path, *reader, q::kDefaultHcMode, &error)
           : q::ModelWeights::Bind(*reader, &error);
  if (!weights) {
    std::fprintf(stderr, "bind failed: %s\n", error.c_str());
    return 1;
  }
  auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (!tokenizer) {
    std::fprintf(stderr, "tokenizer failed: %s\n", error.c_str());
    return 1;
  }
  if (yarn_factor != 1.0F) {
    const q::RopeScaling scaling{
        .factor = yarn_factor,
        .original_context = weights->config.context_length};
    if (!q::ValidateRopeScaling(scaling, weights->config.context_length,
                                &error)) {
      std::fprintf(stderr, "invalid YaRN factor: %s\n", error.c_str());
      return 2;
    }
    weights->config.rope_scaling = scaling;
  }
  const auto& c = weights->config;
  // Sessions may reach the YaRN-extended executor context.
  const auto audit_context = std::max(c.context_length, context);
  if (cost_depth && (audit_context < 96 || *cost_depth > audit_context - 96)) {
    std::fprintf(stderr, "--depth leaves no room for the cost-audit suffix\n");
    return 2;
  }
  std::unique_ptr<q::NgramTable> ngram;
  if (c.ple_layer >= 0) {
    ngram = q::NgramTable::ForModel(model_path, *reader, *weights, &error);
    if (!ngram) {
      std::fprintf(stderr, "n-gram table failed: %s\n", error.c_str());
      return 1;
    }
  }

  std::shared_ptr<gufo::core::GgufReader> mtp_reader;
  std::optional<q::MtpWeights> mtp_weights;
  if (!mtp_path.empty()) {
    mtp_reader = q::OpenArtifactMetadata(mtp_path, &error);
    if (mtp_reader)
      mtp_weights =
          exl3 ? q::LoadExl3Mtp(mtp_path, *mtp_reader, *weights, &error)
               : q::MtpWeights::Bind(*mtp_reader, c, &error);
    if (!mtp_weights) {
      std::fprintf(stderr, "MTP bind failed: %s\n", error.c_str());
      return 1;
    }
  }
  auto device = q::rocm::DeviceModel::Upload(
      *weights, *reader, mtp_weights ? &*mtp_weights : nullptr,
      mtp_reader.get(), &error);
  if (!device) {
    std::fprintf(stderr, "upload failed: %s\n", error.c_str());
    return 1;
  }
  if (sampled_head) {
    std::vector<std::int32_t> specials;
    for (std::uint32_t id = q::kMtpDraftVocab; id < c.vocab_size; ++id)
      if (tokenizer->IsSpecialToken(id))
        specials.push_back(id);
    if (!device->SetDraftVocab(q::kMtpDraftVocab, std::move(specials),
                               &error)) {
      std::fprintf(stderr, "draft vocabulary failed: %s\n", error.c_str());
      return 1;
    }
  }
  q::rocm::Executor::Options options;
  options.max_batch = batch;
  options.max_context = context;
  options.kvarn = kvarn;
  options.recurrent_state = recurrent_state;
  options.max_logit_rows =
      !nll_path.empty()
          ? batch
          : std::min<std::uint32_t>(
                batch, every_pass_rows != 0 ? every_pass_rows : 64);
  if (mtp_audit || cost_audit)
    options.max_speculative = 8;

  if (cost_audit) {
    // --batch matches serving's prefill capacity (4224 for Flash-Next).
    if (!batch_given)
      options.max_batch = 2048;
    options.max_logit_rows = 8;
  }
  if (state_compare != 0) {
    std::vector<std::int32_t> ids;
    for (auto id : tokenizer->Encode(prompt))
      ids.push_back(static_cast<std::int32_t>(id));
    if (prompt_tokens != 0) {
      if (ids.size() < prompt_tokens) {
        std::fprintf(stderr, "the prompt holds only %zu tokens\n", ids.size());
        return 2;
      }
      ids.resize(prompt_tokens);
    }
    if (ids.empty() || ids.size() + state_compare > context) {
      std::fprintf(stderr, "prompt and steps must fit in --context\n");
      return 2;
    }
    options.max_logit_rows = 1;
    // One route: each prefill pass's last row, then each decode step's row.
    // Empty `forced` decodes greedily and records the tokens in `*greedy`.
    const auto run = [&](q::rocm::GdnStateFormat format,
                         std::span<const std::int32_t> forced,
                         std::vector<std::int32_t>* greedy)
        -> std::optional<std::vector<std::vector<float>>> {
      auto route = options;
      route.recurrent_state = format;
      auto exec =
          q::rocm::Executor::Create(*device, ngram.get(), route, &error);
      auto session =
          exec ? exec->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                     context, &error)
               : nullptr;
      if (!session) {
        std::fprintf(stderr, "state compare setup failed: %s\n", error.c_str());
        return std::nullopt;
      }
      std::vector<std::vector<float>> rows;
      for (std::size_t off = 0; off < ids.size(); off += batch) {
        const std::size_t n = std::min<std::size_t>(batch, ids.size() - off);
        const std::size_t next =
            std::min<std::size_t>(batch, ids.size() - off - n);
        std::vector<float> row(c.vocab_size);
        if (!exec->Forward(
                *session, std::span<const std::int32_t>(ids.data() + off, n), 1,
                row.data(), q::rocm::Executor::ForwardMode::kPrefill, &error,
                nullptr,
                std::span<const std::int32_t>(ids.data() + off + n, next))) {
          std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
          return std::nullopt;
        }
        rows.push_back(std::move(row));
      }
      for (std::uint32_t step = 0; step < state_compare; ++step) {
        const std::int32_t token =
            forced.empty()
                ? static_cast<std::int32_t>(gufo::test::ArgMax(rows.back()))
                : forced[step];
        if (greedy != nullptr)
          greedy->push_back(token);
        std::vector<float> row(c.vocab_size);
        if (!exec->Forward(*session, std::span<const std::int32_t>(&token, 1),
                           1, row.data(),
                           q::rocm::Executor::ForwardMode::kDecode, &error)) {
          std::fprintf(stderr, "decode failed: %s\n", error.c_str());
          return std::nullopt;
        }
        rows.push_back(std::move(row));
      }
      return rows;
    };
    std::vector<std::int32_t> greedy;
    const auto reference_rows =
        run(q::rocm::GdnStateFormat::kFp32, {}, &greedy);
    if (!reference_rows)
      return 1;
    const auto candidate_rows =
        run(q::rocm::GdnStateFormat::kFp16, greedy, nullptr);
    if (!candidate_rows)
      return 1;
    const std::size_t passes = reference_rows->size() - state_compare;
    gufo::test::DriftSummary prefill_drift;
    gufo::test::DriftSummary decode_drift;
    for (std::size_t row = 0; row < reference_rows->size(); ++row) {
      const auto drift = gufo::test::CompareLogits((*reference_rows)[row],
                                                   (*candidate_rows)[row]);
      (row < passes ? prefill_drift : decode_drift).Add(drift);
    }
    // Generated token s is the top-1 of row passes - 1 + s.
    std::optional<std::size_t> divergence;
    for (std::size_t row = passes - 1;
         row < passes - 1 + state_compare && !divergence; ++row) {
      if (gufo::test::ArgMax((*reference_rows)[row]) !=
          gufo::test::ArgMax((*candidate_rows)[row]))
        divergence = row + 1 - passes;
    }
    std::ostringstream report;
    gufo::test::WriteStateComparison(report, "fp32", "fp16", ids.size(),
                                     state_compare, prefill_drift, decode_drift,
                                     divergence);
    std::fputs(report.str().c_str(), stdout);
    if (!json_path.empty()) {
      std::ofstream out(json_path);
      out << report.str();
      if (!out) {
        std::fprintf(stderr, "cannot write %s\n", json_path.c_str());
        return 1;
      }
    }
    return 0;
  }
  auto executor =
      q::rocm::Executor::Create(*device, ngram.get(), options, &error);
  if (!executor) {
    std::fprintf(stderr, "executor failed: %s\n", error.c_str());
    return 1;
  }
  if (cost_audit) {
    try {
      const std::array<std::uint32_t, 3> depths{0, 4096, 32768};
      std::span<const std::uint32_t> selected_depths = depths;
      if (cost_depth)
        selected_depths = std::span(&*cost_depth, 1);
      AuditMtpCosts(*executor, *tokenizer, selected_depths, cost_concurrencies,
                    sampled_head);
      return 0;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "cost audit failed: %s\n", e.what());
      return 1;
    }
  }
  if (mtp_audit) {
    try {
      AuditMtp(*executor, *device, *weights, *mtp_weights, *tokenizer,
               model_path);
      return 0;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "MTP oracle failed: %s\n", e.what());
      return 1;
    }
  }
  auto session = executor->CreateSession(
      gufo::core::SessionMode::kAutoregressive, context, &error);
  if (!session) {
    std::fprintf(stderr, "session failed: %s\n", error.c_str());
    return 1;
  }

  std::vector<std::int32_t> tokens;
  for (auto id : tokenizer->Encode(prompt)) {
    tokens.push_back(static_cast<std::int32_t>(id));
  }
  if (tokens.empty() || tokens.size() > context) {
    std::fprintf(stderr, "prompt must contain 1..%u tokens\n", context);
    return 2;
  }
  std::vector<std::int32_t> continuation;
  if (decode) {
    const auto ids = tokenizer->Encode(decode_text);
    continuation.assign(ids.begin(), ids.end());
    if (continuation.empty() || continuation.size() > context - tokens.size()) {
      std::fprintf(stderr, "complete continuation must fit in the context\n");
      return 2;
    }
  }
  std::printf("prompt tokens (%zu)\n", tokens.size());
  std::ofstream dump;
  if (!dump_path.empty()) {
    dump.open(dump_path, std::ios::binary);
    if (!dump) {
      std::fprintf(stderr, "cannot open logit dump %s\n", dump_path.c_str());
      return 1;
    }
  }

  std::vector<float> gpu_logits;
  std::vector<float> nll;  ///< --nll: per-position next-token NLL
  std::size_t logit_rows = 0;
  for (std::size_t off = 0; off < tokens.size(); off += batch) {
    const std::size_t n = std::min<std::size_t>(batch, tokens.size() - off);
    const auto rows = static_cast<std::uint32_t>(
        std::min<std::size_t>(n, options.max_logit_rows));
    std::vector<float> out(static_cast<std::size_t>(rows) * c.vocab_size);
    // Name the following chunk as Session::Feed does, so its n-gram read
    // overlaps this one.
    const std::size_t next =
        std::min<std::size_t>(batch, tokens.size() - off - n);
    if (!executor->Forward(
            *session, std::span<const std::int32_t>(tokens.data() + off, n),
            rows, out.data(), q::rocm::Executor::ForwardMode::kPrefill, &error,
            nullptr,
            std::span<const std::int32_t>(tokens.data() + off + n, next))) {
      std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
      return 1;
    }
    const bool last = off + n == tokens.size();
    if (!nll_path.empty()) {
      // Row i predicts token off + i + 1; rows split across threads.
      const std::size_t scored = last ? n - 1 : n;
      nll.resize(off + scored);
      const unsigned workers =
          std::max(1U, std::min(16U, std::thread::hardware_concurrency()));
      std::vector<std::thread> pool;
      for (unsigned w = 0; w < workers; ++w) {
        pool.emplace_back([&, w] {
          for (std::size_t i = w; i < scored; i += workers) {
            const float* row = out.data() + i * c.vocab_size;
            const float peak = *std::max_element(row, row + c.vocab_size);
            double sum = 0.0;
            for (std::size_t v = 0; v < c.vocab_size; ++v)
              sum += std::exp(static_cast<double>(row[v]) - peak);
            const auto target = static_cast<std::size_t>(tokens[off + i + 1]);
            nll[off + i] = static_cast<float>(
                std::log(sum) - (static_cast<double>(row[target]) - peak));
          }
        });
      }
      for (auto& t : pool)
        t.join();
    } else if (every_pass_rows != 0) {
      gpu_logits.insert(gpu_logits.end(), out.begin(), out.end());
      logit_rows += rows;
    } else if (last) {
      gpu_logits = std::move(out);
      logit_rows = rows;
    }
  }
  if (!nll_path.empty()) {
    std::ofstream file(nll_path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(nll.data()),
               static_cast<std::streamsize>(nll.size() * sizeof(float)));
    if (!file) {
      std::fprintf(stderr, "cannot write %s\n", nll_path.c_str());
      return 1;
    }
    const double total = std::accumulate(nll.begin(), nll.end(), 0.0);
    std::printf("nll %zu positions mean %.6f ppl %.4f\n", nll.size(),
                total / static_cast<double>(nll.size()),
                std::exp(total / static_cast<double>(nll.size())));
    return 0;
  }
  if (decode) {
    // Matched-token decode: feed a fixed continuation one token at a time
    // through the decode path and dump each step's logits.
    std::vector<float> row(c.vocab_size);
    std::size_t steps = 0;
    for (const auto t : continuation) {
      if (!executor->Forward(*session, std::span<const std::int32_t>(&t, 1), 1,
                             row.data(),
                             q::rocm::Executor::ForwardMode::kDecode, &error)) {
        std::fprintf(stderr, "decode failed: %s\n", error.c_str());
        return 1;
      }
      if (dump.is_open())
        dump.write(reinterpret_cast<const char*>(row.data()),
                   static_cast<std::streamsize>(row.size() * sizeof(float)));
      ++steps;
    }
    if (dump.is_open()) {
      dump.close();
      if (!dump) {
        std::fprintf(stderr, "cannot write logit dump %s\n", dump_path.c_str());
        return 1;
      }
    }
    if (!decode_ids_path.empty()) {
      std::ofstream ids(decode_ids_path, std::ios::binary);
      ids.write(reinterpret_cast<const char*>(continuation.data()),
                static_cast<std::streamsize>(continuation.size() *
                                             sizeof(continuation[0])));
      ids.close();
      if (!ids) {
        std::fprintf(stderr, "cannot write token IDs %s\n",
                     decode_ids_path.c_str());
        return 1;
      }
    }
    std::printf("decode steps %zu after %zu prompt tokens\n", steps,
                tokens.size());
    return 0;
  }
  if (dump.is_open()) {
    dump.write(reinterpret_cast<const char*>(gpu_logits.data()),
               static_cast<std::streamsize>(gpu_logits.size() * sizeof(float)));
    dump.flush();
    if (!dump) {
      std::fprintf(stderr, "cannot write logit dump %s\n", dump_path.c_str());
      return 1;
    }
    if (every_pass_rows != 0) {
      // Which rows the dump holds: f32_reference.py checks it against the
      // fixture, since another batch can give the same row count.
      // FNV-1a over the token ids' little-endian bytes: the fixture's
      // tokens.json must hold the same ids, not only the same count.
      std::uint64_t ids_hash = 0xCBF29CE484222325ULL;
      for (const std::int32_t id : tokens) {
        const auto bits = static_cast<std::uint32_t>(id);
        for (int byte = 0; byte < 4; ++byte) {
          ids_hash ^= (bits >> (8 * byte)) & 0xFFU;
          ids_hash *= 0x100000001B3ULL;
        }
      }
      std::ofstream meta(dump_path + ".json");
      meta << "{\"tokens\": " << tokens.size() << ", \"ids_fnv1a\": \""
           << std::hex << ids_hash << std::dec << "\", \"batch\": " << batch
           << ", \"rows_per_pass\": " << every_pass_rows
           << ", \"rows\": " << logit_rows << "}\n";
      if (!meta) {
        std::fprintf(stderr, "cannot write %s.json\n", dump_path.c_str());
        return 1;
      }
    }
  }

  if (reference) {
    q::ReferenceModel ref(*weights, ngram.get(),
                          static_cast<std::uint32_t>(tokens.size()));
    std::vector<float> ref_logits(c.vocab_size);
    const std::size_t first = tokens.size() - logit_rows;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      if (!ref.Step(tokens[i], ref_logits, {}, &error)) {
        std::fprintf(stderr, "reference failed: %s\n", error.c_str());
        return 1;
      }
      if (i < first) {
        continue;
      }
      const Stats st =
          Compare(ref_logits.data(),
                  gpu_logits.data() + (i - first) * c.vocab_size, c.vocab_size);
      std::printf("pos %3zu ref %6u gpu %6u %s  KL %.5f  max|d| %.3f\n", i,
                  st.argmax_a, st.argmax_b,
                  st.argmax_a == st.argmax_b ? "OK  " : "DIFF", st.kl,
                  st.max_abs);
    }
  }

  return 0;
}
