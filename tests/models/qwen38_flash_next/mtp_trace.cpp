// Greedy MTP chain trace for draft-policy studies (plan W9 M1).
//
// For every prompt of a serving suite (gufo.speculative-corpus.v1 JSON), the
// trace renders the chat prompt as the HTTP server does (thinking off), decodes
// greedily through the production Session, and before committing each token
// asks Session::ProbeDraftChain for the 1..depth greedy MTP drafts a
// speculative cycle anchored on that token would propose. A cycle's accepted
// length is the number of leading drafts equal to the following committed
// tokens, so any length or prompt-lookup policy can be replayed offline.
//
// Output (JSON lines): one "case" line per prompt, then one line per decoded
// position with the anchor, whether the small draft head was used, and per
// depth [draft token, top-candidate softmax probability, top-2 logit gap].
#include "tests/models/qwen38_flash_next/mtp_trace.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/sampling.hpp"
#include "src/core/session_mode.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace tok = gufo::tokenization;

namespace {

std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot open " + path);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::string JsonString(const std::string& s) {
  return gufo::json::Value(s).dump();
}

/// Softmax probability of the drafted token over the reported candidates
/// and its logit gap to the best other candidate. A token missing from the
/// candidates (not expected: the draft is their argmax) reports p = -1.
std::pair<double, double> Confidence(const qfn::MtpCandidateLogits& c,
                                     std::int32_t token) {
  std::size_t self = c.size;
  float peak = -INFINITY;
  for (std::size_t i = 0; i < c.size; ++i) {
    peak = std::max(peak, c.logits[i]);
    if (static_cast<std::int32_t>(c.ids[i]) == token && self == c.size)
      self = i;
  }
  if (self == c.size)
    return {-1.0, 0.0};
  double sum = 0.0, other = -INFINITY;
  for (std::size_t i = 0; i < c.size; ++i) {
    sum += std::exp(static_cast<double>(c.logits[i]) - peak);
    if (i != self)
      other = std::max(other, static_cast<double>(c.logits[i]));
  }
  const double p = std::exp(static_cast<double>(c.logits[self]) - peak) / sum;
  return {p, c.size > 1 ? c.logits[self] - other : 0.0};
}

}  // namespace

int RunMtpTrace(const MtpTraceOptions& o) {
  std::string error;
  qfn::ModelOptions options;
  options.max_context = o.context;
  options.mtp_model_path = o.mtp_path;
  options.max_draft_tokens = qfn::kMaxMtpDraftTokens;
  options.decode_concurrency = 1;
  auto model = qfn::Model::Load(o.model_path, options, &error);
  if (!model || !model->HasMtp()) {
    std::fprintf(stderr, "model load failed: %s\n", error.c_str());
    return 1;
  }
  auto session = model->CreateSession(gufo::core::SessionMode::kSpeculative,
                                      o.context, &error);
  if (!session) {
    std::fprintf(stderr, "session failed: %s\n", error.c_str());
    return 1;
  }
  const auto suite = gufo::json::parse(ReadFile(o.suite_path));
  const auto* prompts = suite.find("prompts");
  if (prompts == nullptr || !prompts->is_array())
    throw std::runtime_error("suite has no prompts array");
  std::ofstream out(o.out_path, std::ios::binary);
  if (!out)
    throw std::runtime_error("cannot write " + o.out_path);

  // The replay stops a chain at a drafted stop token, as PrepareDecode does.
  std::string stops;
  for (std::uint32_t id = 0; id < model->VocabSize(); ++id)
    if (model->IsStopToken(static_cast<std::int32_t>(id)))
      stops += (stops.empty() ? "" : ",") + std::to_string(id);
  out << "{\"stop_tokens\":[" << stops << "],\"depth\":" << o.depth
      << ",\"max_tokens\":" << o.max_tokens << ",\"context\":" << o.context
      << "}\n";

  tok::ChatTemplateOptions chat;
  chat.enable_thinking = false;
  chat.max_output_bytes = tok::RenderedPromptBoundBytes(o.context);
  std::size_t index = 0;
  for (const auto& entry : prompts->items()) {
    const auto id = entry.member_str("id");
    const auto category = entry.member_str("category");
    const std::vector<tok::ChatMessage> messages{
        tok::ChatMessage(tok::ChatRole::kUser, entry.member_str("text"))};
    const auto rendered = tok::QwenChatTemplate::RenderAndTokenize(
        model->tokenizer(), messages, chat, &error);
    if (!rendered) {
      std::fprintf(stderr, "%s: render failed: %s\n", id.c_str(),
                   error.c_str());
      return 1;
    }
    const std::vector<std::int32_t> prompt(rendered->begin(), rendered->end());
    session->ResetDraftPolicy();
    if (!session->Sync(prompt, &error)) {
      std::fprintf(stderr, "%s: prefill failed: %s\n", id.c_str(),
                   error.c_str());
      return 1;
    }
    gufo::sampling::SamplerState sampler;  // greedy, no penalties
    std::vector<std::int32_t> completion;
    std::vector<std::string> lines;
    bool stop = false;
    for (std::uint32_t pos = 0; pos < o.max_tokens; ++pos) {
      const auto anchor =
          static_cast<std::int32_t>(sampler.Sample(session->Logits()));
      if (model->IsStopToken(anchor)) {
        stop = true;
        break;
      }
      qfn::Session::DraftProbe probe;
      if (!session->ProbeDraftChain(anchor, o.depth, &probe, &error)) {
        std::fprintf(stderr, "%s: probe failed at %u: %s\n", id.c_str(), pos,
                     error.c_str());
        return 1;
      }
      std::string line =
          "{\"c\":" + std::to_string(index) + ",\"t\":" + std::to_string(pos) +
          ",\"a\":" + std::to_string(anchor) +
          ",\"s\":" + (probe.small_head ? "1" : "0") + ",\"d\":[";
      for (std::uint32_t d = 0; d < probe.depth; ++d) {
        const auto [p, gap] = Confidence(probe.candidates[d], probe.tokens[d]);
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%s[%d,%.6g,%.6g]", d ? "," : "",
                      probe.tokens[d], p, gap);
        line += buf;
      }
      line += "]}";
      lines.push_back(std::move(line));
      completion.push_back(anchor);
      if (pos + 1 < o.max_tokens && !session->Evaluate(anchor, &error)) {
        std::fprintf(stderr, "%s: decode failed at %u: %s\n", id.c_str(), pos,
                     error.c_str());
        return 1;
      }
    }
    const auto join = [](std::span<const std::int32_t> v) {
      std::string text;
      for (std::size_t i = 0; i < v.size(); ++i)
        text += (i ? "," : "") + std::to_string(v[i]);
      return text;
    };
    const auto ids = join(completion);
    out << "{\"case\":" << JsonString(id) << ",\"index\":" << index
        << ",\"category\":" << JsonString(category)
        << ",\"prompt_tokens\":" << prompt.size() << ",\"prompt\":["
        << join(prompt) << "],\"stop\":" << (stop ? "true" : "false")
        << ",\"completion\":[" << ids
        << "],\"text\":" << JsonString(model->Decode(completion)) << "}\n";
    for (const auto& line : lines)
      out << line << '\n';
    out.flush();
    if (!out)
      throw std::runtime_error("cannot write " + o.out_path);
    std::printf("trace %s prompt %zu completion %zu stop %d\n", id.c_str(),
                prompt.size(), completion.size(), stop ? 1 : 0);
    std::fflush(stdout);
    ++index;
  }
  return 0;
}
