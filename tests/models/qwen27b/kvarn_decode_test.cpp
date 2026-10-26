// KVarN decode and verification through the Qwen3.8-27B executor (K1c-1,
// plan/gufo-upstream-port/K1-BRIEF.md): a prompt fed token by token (the
// first rows through the captured decode graph, then split-K with group
// flushes) against the FP16 executor under teacher forcing, DFlash2-style
// verification rows against scalar decoding of the same tokens, in-memory
// snapshots, prompt prefill, and the compact snapshot refusal that stays
// until a KVarN checkpoint layout (K2) exists.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "tests/models/qwen27b/kvarn_session.hpp"

namespace {

using Token = gufo::tokenization::TokenId;
using Executor = gufo::hip::QwenGpuExecutor;
using gufo::hip::QwenKvCacheStorage;

void Expect(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::vector<float> Logits(Executor& executor) {
  const auto view = executor.CopyLastLogits();
  Expect(!view.empty() && std::ranges::all_of(
                              view, [](float v) { return std::isfinite(v); }),
         "logits must be finite and nonempty");
  return {view.begin(), view.end()};
}

// KL(p || q) of two logit rows, in double.
double Kl(std::span<const float> p_logits, std::span<const float> q_logits) {
  const auto log_softmax = [](std::span<const float> x) {
    const double mx = *std::ranges::max_element(x);
    double sum = 0;
    for (const float v : x)
      sum += std::exp(v - mx);
    const double lse = mx + std::log(sum);
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
      out[i] = x[i] - lse;
    return out;
  };
  const auto lp = log_softmax(p_logits), lq = log_softmax(q_logits);
  double kl = 0;
  for (std::size_t i = 0; i < lp.size(); ++i)
    kl += std::exp(lp[i]) * (lp[i] - lq[i]);
  return std::max(kl, 0.0);
}

std::size_t Argmax(std::span<const float> x) {
  return static_cast<std::size_t>(std::ranges::max_element(x) - x.begin());
}

std::vector<Token> Fixture(const Executor& executor, std::size_t count) {
  std::string text;
  for (int i = 0; text.size() < count * 8; ++i)
    text += "Entry " + std::to_string(i) + ": the station at kilometer " +
            std::to_string(17 * i % 101) + " reports " +
            std::to_string(3 * i % 7) +
            " delayed trains, and the operating system pages " +
            std::to_string(i * i % 53) + " frames back to memory.\n";
  auto tokens = executor.GetTokenizer().Encode(text);
  Expect(tokens.size() >= count, "fixture has too few tokens");
  tokens.resize(count);
  return tokens;
}

void CheckStorage(const std::shared_ptr<const gufo::hip::QwenGpuModel>& model,
                  QwenKvCacheStorage storage) {
  std::vector<std::string> deferred;
  constexpr std::uint32_t kContext = 1024;
  // Ends inside group 4; the verification rows cross into group 5.
  constexpr std::uint32_t kPrompt = 636, kRows = 8, kTail = 24;
  auto policy = gufo::hip::QwenExecutionPolicy::Production();
  policy.kv_cache_storage = storage;
  const std::string name(gufo::hip::QwenKvCacheStorageName(storage));

  std::string error;
  auto created = Executor::Create(model, &error, kContext, policy);
  Expect(created != nullptr, name + ": executor creation: " + error);
  Executor fp16(model, kContext);
  Executor& kvarn = *created;
  const auto tokens = Fixture(fp16, kPrompt + kRows + kTail);

  // Teacher forcing: three executors read the same tokens. `fake` is the
  // FP16 executor of the K0 study whose K/V rows of every group that ended
  // at or before the next position are fake-quantized in place
  // (kvarn_fake_quant.hpp), the rule KvarnReadGroups gives the real path.
  gufo::testing::kvarn::Session fake_session(model, kContext);
  auto& fake = fake_session.Executor();
  // The control: the same fake quantization of rows each moved by about one
  // FP16 rounding first (a rotation round trip), measuring how far
  // last-bit differences carry through the discontinuous RTN codes of later
  // groups. Real KVarN reaches its groups' rows through the same kind of
  // differences (rotated FP16 rows, FP16 query and P in the kernel).
  gufo::testing::kvarn::Session control_session(model, kContext);
  auto& control = control_session.Executor();
  // control_k (K1e, approved 2026-10-26): the fake run with every K/V row
  // moved by one rotation round trip as it enters the cache (sink and tail
  // included), the rounding the real path applies to every rotated FP16
  // row; groups are fake-quantized from the moved rows.
  gufo::testing::kvarn::Session control_k_session(model, kContext);
  auto& control_k = control_k_session.Executor();
  const auto config = *gufo::testing::kvarn::ParseConfig(
      storage == QwenKvCacheStorage::kKvarnK4V2 ? "kvarn_k4v2_g128"
                                                : "kvarn_k4v4_g128");
  const auto& c = model->GetConfig();
  struct Step {
    double fp16_real, fp16_fake, fake_real, fake_control, fake_control_k;
    bool top1_real, top1_fake, top1_real_fake, top1_fake_control;
  };
  // Rows [first, end) of every K/V component through H, FP16, H, FP16.
  const auto round_trip_rows = [&](gufo::testing::kvarn::Session& session,
                                   gufo::testing::kvarn::State& state,
                                   std::size_t first, std::size_t end) {
    const std::size_t width = std::size_t{c.num_key_value_heads} * c.head_dim;
    std::vector<float> x(c.head_dim);
    for (std::size_t i = 0; i < state.components.size(); ++i) {
      if (!session.IsRows(i))
        continue;
      auto* rows = reinterpret_cast<std::uint16_t*>(state.components[i].data());
      for (std::size_t t = first; t < end; ++t)
        for (std::size_t h = 0; h < c.num_key_value_heads; ++h) {
          std::uint16_t* row = rows + t * width + h * c.head_dim;
          for (std::size_t d = 0; d < c.head_dim; ++d)
            x[d] = gufo::testing::kvarn::HalfToFloat(row[d]);
          gufo::testing::kvarn::Hadamard(x);
          for (auto& value : x)
            value = gufo::testing::kvarn::RoundHalf(value);
          gufo::testing::kvarn::Hadamard(x);
          for (std::size_t d = 0; d < c.head_dim; ++d)
            row[d] = gufo::testing::kvarn::FloatToHalf(x[d]);
        }
    }
  };
  std::vector<Step> steps;
  for (std::uint32_t pos = 0; pos < kPrompt; ++pos) {
    if (pos >= 256 && pos % 128 == 0) {
      auto state = fake_session.Capture();
      (void)gufo::testing::kvarn::Quantize(fake_session, state, config,
                                           c.num_key_value_heads, c.head_dim,
                                           pos / 128 - 1, pos / 128);
      fake_session.Restore(state);
      auto moved = control_session.Capture();
      round_trip_rows(control_session, moved, (pos / 128 - 1) * 128,
                      pos / 128 * 128);
      (void)gufo::testing::kvarn::Quantize(control_session, moved, config,
                                           c.num_key_value_heads, c.head_dim,
                                           pos / 128 - 1, pos / 128);
      control_session.Restore(moved);
    }
    (void)fp16.ForwardToken(tokens[pos], pos);
    (void)kvarn.ForwardToken(tokens[pos], pos);
    (void)fake.ForwardToken(tokens[pos], pos);
    (void)control.ForwardToken(tokens[pos], pos);
    if (pos > 0) {
      // The row the previous step wrote, then (at a group end) the group.
      auto moved = control_k_session.Capture();
      round_trip_rows(control_k_session, moved, pos - 1, pos);
      if (pos >= 256 && pos % 128 == 0)
        (void)gufo::testing::kvarn::Quantize(control_k_session, moved, config,
                                             c.num_key_value_heads, c.head_dim,
                                             pos / 128 - 1, pos / 128);
      control_k_session.Restore(moved);
    }
    (void)control_k.ForwardToken(tokens[pos], pos);
    const auto reference = Logits(fp16);
    const auto real = Logits(kvarn);
    const auto faked = Logits(fake);
    const auto controlled = Logits(control);
    const auto controlled_k = Logits(control_k);
    if (pos < 128) {
      // The sink alone: rotation and FP16 rounding of rotated rows only.
      const double kl = Kl(reference, real);
      Expect(kl < 1e-3, name + ": sink-only KL too large at " +
                            std::to_string(pos) + ": " + std::to_string(kl));
      continue;
    }
    steps.push_back(
        {Kl(reference, real), Kl(reference, faked), Kl(faked, real),
         Kl(faked, controlled), Kl(faked, controlled_k),
         Argmax(reference) == Argmax(real), Argmax(reference) == Argmax(faked),
         Argmax(real) == Argmax(faked), Argmax(controlled) == Argmax(faked)});
    const auto& step = steps.back();
    std::cout << name << " step " << pos << " kl_fp16_real " << step.fp16_real
              << " kl_fp16_fake " << step.fp16_fake << " kl_fake_real "
              << step.fake_real << " top1_fp16_real " << step.top1_real
              << " top1_fp16_fake " << step.top1_fake << " top1_real_fake "
              << step.top1_real_fake << " kl_fake_control " << step.fake_control
              << " top1_fake_control " << step.top1_fake_control
              << " kl_fake_control_k " << step.fake_control_k << '\n';
  }
  const auto summary = [&](auto field, std::uint32_t from, std::uint32_t to) {
    std::vector<double> values;
    for (std::size_t i = 0; i < steps.size(); ++i)
      if (128 + i >= from && 128 + i < to)
        values.push_back(steps[i].*field);
    std::ranges::sort(values);
    return std::array<double, 3>{values[values.size() / 2],
                                 values[values.size() * 95 / 100],
                                 values.back()};
  };
  std::size_t real_agree = 0, fake_agree = 0, real_fake_agree = 0,
              control_agree = 0;
  for (const auto& step : steps) {
    real_agree += step.top1_real;
    fake_agree += step.top1_fake;
    real_fake_agree += step.top1_real_fake;
    control_agree += step.top1_fake_control;
  }
  for (const auto [from, to] :
       {std::pair{128U, 256U}, std::pair{256U, 384U}, std::pair{384U, 512U},
        std::pair{512U, kPrompt}, std::pair{128U, kPrompt}}) {
    const auto a = summary(&Step::fp16_real, from, to);
    const auto b = summary(&Step::fp16_fake, from, to);
    const auto d = summary(&Step::fake_real, from, to);
    const auto e = summary(&Step::fake_control, from, to);
    const auto f = summary(&Step::fake_control_k, from, to);
    std::cout << name << " positions " << from << "-" << to - 1
              << " KL median/p95/max fp16||real " << a[0] << "/" << a[1] << "/"
              << a[2] << " fp16||fake " << b[0] << "/" << b[1] << "/" << b[2]
              << " fake||real " << d[0] << "/" << d[1] << "/" << d[2]
              << " fake||control " << e[0] << "/" << e[1] << "/" << e[2]
              << " fake||control_k " << f[0] << "/" << f[1] << "/" << f[2]
              << " ratio_p95/max real:control " << d[1] / std::max(e[1], 1e-6)
              << "/" << d[2] / std::max(e[2], 1e-6) << " real:max(control,"
              << "control_k) " << d[1] / std::max({e[1], f[1], 1e-6}) << "/"
              << d[2] / std::max({e[2], f[2], 1e-6}) << '\n';
  }
  std::cout << name << " top-1 of " << steps.size() << ": fp16=real "
            << real_agree << " fp16=fake " << fake_agree << " real=fake "
            << real_fake_agree << " fake=control " << control_agree << '\n';
  // Without a quantized group the rows differ from FP16 only by the
  // rotation's rounding.
  Expect(summary(&Step::fp16_real, 128, 256)[2] < 1e-2,
         name + ": FP16-only tail KL too large");
  Expect(real_agree * 100 >= steps.size() * 95,
         name + ": top-1 agreement below 95 %");
  // Real and fake quantization store the same codes for the same rows
  // (bit-exact in the kernel test); their rows differ by last bits, which
  // later groups' RTN codes amplify. The control measures that
  // amplification, control_k the real path's rounding of every rotated row
  // (K1e): per window, the real path may depart from the fake one by at
  // most ten times the larger control's departure (p95 and max), and agree
  // on top-1 no worse than the control by more than one step in 100.
  for (const auto [from, to] : {std::pair{256U, 384U}, std::pair{384U, 512U},
                                std::pair{512U, kPrompt}}) {
    const auto real_fake = summary(&Step::fake_real, from, to);
    const auto fake_control = summary(&Step::fake_control, from, to);
    const auto fake_control_k = summary(&Step::fake_control_k, from, to);
    for (const std::size_t q : {1U, 2U})
      if (real_fake[q] >
          10 * std::max({fake_control[q], fake_control_k[q], 1e-6}))
        deferred.push_back(name +
                           ": real KVarN departs from K0 fake quant beyond the "
                           "rounding control at positions " +
                           std::to_string(from) + "-" + std::to_string(to - 1) +
                           (q == 1 ? " (p95)" : " (max)"));
  }
  if (real_fake_agree * 100 + steps.size() < control_agree * 100)
    deferred.push_back(name +
                       ": real and fake KVarN disagree on top-1 beyond the "
                       "control");
  // The bound is checked after the remaining checks have run.

  // An in-memory snapshot of this state, restored below.
  auto snapshot = kvarn.SaveSnapshot(kPrompt);
  // Verification rows against scalar decoding of the same tokens.
  kvarn.SaveState(kPrompt);
  const auto verify = std::span(tokens).subspan(kPrompt, kRows);
  const auto predictions =
      kvarn.ForwardVerificationChunk(verify, kPrompt, true);
  const auto rows = kvarn.GetVerificationLogits();
  const std::size_t vocab = kvarn.GetConfig().vocab_size;
  Expect(predictions.size() == kRows && rows.size() == kRows * vocab,
         name + ": verification rows");
  const std::vector<float> verified(rows.begin(), rows.end());
  kvarn.RestoreState();
  float max_abs = 0;
  bool exact = true;
  std::vector<float> first_scalar;
  for (std::uint32_t row = 0; row < kRows; ++row) {
    (void)kvarn.ForwardToken(verify[row], kPrompt + row);
    const auto scalar = Logits(kvarn);
    if (row == 0)
      first_scalar = scalar;
    const auto batch = std::span(verified).subspan(
        static_cast<std::size_t>(row) * vocab, vocab);
    exact = exact && std::memcmp(scalar.data(), batch.data(),
                                 vocab * sizeof(float)) == 0;
    for (std::size_t i = 0; i < vocab; ++i)
      max_abs = std::max(max_abs, std::abs(scalar[i] - batch[i]));
    Expect(Argmax(scalar) == Argmax(batch) &&
               predictions[row] == static_cast<Token>(Argmax(scalar)),
           name + ": verification row " + std::to_string(row) +
               " predicts differently from decoding");
  }
  std::cout << name << " verification rows " << kPrompt << "-"
            << kPrompt + kRows - 1 << " vs scalar decode: exact=" << exact
            << " max_abs=" << max_abs << '\n';
  // Verification rows read what decode steps at their positions read
  // (KvarnReadGroups) with the same split order: equal logits, as FP16.
  Expect(exact, name + ": verification differs from decoding");

  // Decoding on past group 5's end (quantized at 768).
  for (std::uint32_t pos = kPrompt + kRows; pos < kPrompt + kRows + kTail;
       ++pos) {
    (void)kvarn.ForwardToken(tokens[pos], pos);
    (void)Logits(kvarn);
  }

  // The snapshot restores the state exactly (CheckCompact covers its
  // compact form).
  kvarn.RestoreSnapshot(*snapshot);
  (void)kvarn.ForwardToken(verify[0], kPrompt);
  Expect(std::memcmp(Logits(kvarn).data(), first_scalar.data(),
                     vocab * sizeof(float)) == 0,
         name + ": a restored snapshot decodes differently");

  // Prompt prefill. One chunk reads only its own FP16 rows, so it matches
  // the FP16 prefill but for the order of the attention's sums; two chunks
  // read the groups that ended before the second one quantized.
  for (const std::uint32_t split : {kPrompt, 300U}) {
    fp16.Reset();
    kvarn.Reset();
    for (auto* executor : {&fp16, &kvarn}) {
      (void)executor->ForwardPromptBatch(std::span(tokens).first(split));
      if (split < kPrompt)
        (void)executor->ForwardPromptBatch(
            std::span(tokens).subspan(split, kPrompt - split), split);
    }
    const auto reference = Logits(fp16);
    const auto real = Logits(kvarn);
    const double kl = Kl(reference, real);
    std::cout << name << " prefill of " << kPrompt << " tokens in "
              << (split < kPrompt ? 2 : 1) << " chunks: KL fp16||real " << kl
              << " top-1 " << (Argmax(reference) == Argmax(real)) << '\n';
    if (split == kPrompt)
      Expect(kl < 1e-4, name + ": a one-chunk prefill departs from FP16");
    // Decoding continues from the prefilled tail.
    for (std::uint32_t pos = kPrompt; pos < kPrompt + 4; ++pos) {
      (void)fp16.ForwardToken(tokens[pos], pos);
      (void)kvarn.ForwardToken(tokens[pos], pos);
      std::cout << name << "   decode " << pos << " after prefill: KL "
                << Kl(Logits(fp16), Logits(kvarn)) << '\n';
    }
  }
  // A reset starts over from the sink.
  kvarn.Reset();
  fp16.Reset();
  for (std::uint32_t pos = 0; pos < 4; ++pos) {
    (void)fp16.ForwardToken(tokens[pos], pos);
    (void)kvarn.ForwardToken(tokens[pos], pos);
  }
  Expect(Kl(Logits(fp16), Logits(kvarn)) < 1e-3, name + ": reset");
  for (const auto& failure : deferred)
    std::cerr << "FAIL " << failure << '\n';
  Expect(deferred.empty(), name + ": real vs fake bound failed");
}

// Continuation (K2-1): a KVarN slot captured into the cache's components
// and restored into another slot is the same state byte for byte (records,
// FP16 sink and tail ring, tail state) and decodes the same logits. With
// `pooled`, both slots take blocks from one shared pool, as serving does.
// `committed` > 0: after decoding to kEnd, verify 8 draft rows and keep
// `committed` of them, so the tail can be flushed past the position (a group
// ends among the rejected rows). The checkpoint holds `records` record rows,
// the valid sink rows and a tail state whose tail starts `tail` rows before
// the position (K4-C1).
void CheckContinuation(
    const std::shared_ptr<const gufo::hip::QwenGpuModel>& model,
    QwenKvCacheStorage storage, std::uint32_t kPrompt, std::uint32_t kEnd,
    bool pooled, std::uint32_t committed, std::uint32_t expected_records,
    std::uint32_t expected_tail) {
  constexpr std::uint32_t kContext = 1024;
  auto policy = gufo::hip::QwenExecutionPolicy::Production();
  policy.kv_cache_storage = storage;
  const auto& config = model->GetConfig();
  const auto pool =
      pooled ? std::make_shared<gufo::hip::KvBlockPool>(
                   gufo::hip::QwenKvBlockPlanes(
                       config.FullAttentionLayerCount(),
                       config.num_key_value_heads, config.head_dim, storage),
                   16)
             : nullptr;
  const std::string name =
      std::string(gufo::hip::QwenKvCacheStorageName(storage)) +
      (pooled ? " pooled" : "") + " at " + std::to_string(kEnd) +
      (committed ? " + " + std::to_string(committed) + " of 8 verified" : "");
  gufo::testing::kvarn::Session source(model, kContext, policy, pool);
  auto& executor = source.Executor();
  const auto tokens = Fixture(executor, kEnd + 9);
  (void)executor.ForwardPromptBatch(std::span(tokens).first(kPrompt));
  for (std::uint32_t pos = kPrompt; pos < kEnd; ++pos)
    (void)executor.ForwardToken(tokens[pos], pos);
  std::uint32_t position = kEnd;
  if (committed != 0) {
    (void)executor.ForwardVerificationChunk(std::span(tokens).subspan(kEnd, 8),
                                            kEnd);
    executor.CommitVerificationChunk(std::span(tokens).subspan(kEnd, committed),
                                     kEnd);
    position += committed;
  }
  const auto captured = source.Capture();
  // Records, the FP16 sink rows (one position of every layer) and the tail
  // state: version, flushed end (the tail's base), written end.
  constexpr std::size_t kFp16Row = 2ULL * 16 * 4 * 256 * 2;
  constexpr std::size_t kTailState = 64 + 256 * kFp16Row;
  std::size_t records = 0, sink = 0, tails = 0, fp16_bytes = 0;
  for (std::size_t i = 0; i < captured.positions.size(); ++i) {
    const auto rows = captured.positions[i].valid_rows;
    if (!source.IsRows(i)) {
      if (captured.components[i].size() != kTailState)
        continue;
      ++tails;
      std::array<std::uint32_t, 3> header{};
      std::memcpy(header.data(), captured.components[i].data(), sizeof(header));
      const auto tail = position < 128 ? 0 : position - header[1];
      Expect(header[0] == 2 && header[2] == position && tail == expected_tail,
             name + ": KVarN tail state " + std::to_string(header[1]));
      fp16_bytes += kTailState;
      continue;
    }
    if (source.RowBytes(i) == kFp16Row) {
      ++sink;
      Expect(rows == std::min(position, 128U),
             name + ": KVarN sink rows " + std::to_string(rows));
      fp16_bytes += rows * kFp16Row;
      continue;
    }
    ++records;
    Expect(rows == expected_records,
           name + ": KVarN record rows " + std::to_string(rows));
  }
  Expect(records == 16 && sink == 1 && tails == 1, name + ": KVarN components");
  std::cout << name << ": FP16 sink rows and tail state " << fp16_bytes
            << " bytes\n";

  gufo::testing::kvarn::Session restored(model, kContext, policy, pool);
  restored.Restore(captured);
  const auto again = restored.Capture();
  Expect(again.positions == captured.positions,
         name + ": restored positions differ");
  std::size_t differing = 0;
  for (std::size_t i = 0; i < captured.components.size(); ++i)
    differing += captured.components[i] != again.components[i];
  std::cout << name << " continuation: " << captured.components.size()
            << " components, " << differing << " differ after restore\n";
  Expect(differing == 0, name + ": a restored KVarN state differs");

  (void)executor.ForwardToken(tokens[position], position);
  const auto expected = Logits(executor);
  auto& copy = restored.Executor();
  (void)copy.ForwardToken(tokens[position], position);
  const auto actual = Logits(copy);
  Expect(std::memcmp(expected.data(), actual.data(),
                     expected.size() * sizeof(float)) == 0,
         name + ": a restored KVarN slot decodes differently");
  std::cout << name << " continuation decode after restore: exact\n";
}

// Compact snapshot (K4-C2): a KVarN slot serialized, restored into another
// executor and serialized again gives the same bytes; the payload holds the
// records before the tail base and only the valid FP16 rows (the
// continuation rule); and the next decode matches exactly.
void CheckCompact(const std::shared_ptr<const gufo::hip::QwenGpuModel>& model,
                  QwenKvCacheStorage storage, std::uint32_t kPrompt,
                  std::uint32_t kEnd, std::uint32_t committed,
                  std::uint32_t expected_records, std::uint32_t expected_tail) {
  constexpr std::uint32_t kContext = 1024;
  auto policy = gufo::hip::QwenExecutionPolicy::Production();
  policy.kv_cache_storage = storage;
  std::string error;
  auto source = Executor::Create(model, &error, kContext, policy);
  auto target = Executor::Create(model, &error, kContext, policy);
  Expect(source && target, error);
  const std::string name =
      std::string(gufo::hip::QwenKvCacheStorageName(storage)) + " compact at " +
      std::to_string(kEnd) +
      (committed ? " + " + std::to_string(committed) + " of 8 verified" : "");
  const auto tokens = Fixture(*source, kEnd + 9);
  (void)source->ForwardPromptBatch(std::span(tokens).first(kPrompt));
  for (std::uint32_t pos = kPrompt; pos < kEnd; ++pos)
    (void)source->ForwardToken(tokens[pos], pos);
  std::uint32_t position = kEnd;
  if (committed != 0) {
    (void)source->ForwardVerificationChunk(std::span(tokens).subspan(kEnd, 8),
                                           kEnd);
    source->CommitVerificationChunk(std::span(tokens).subspan(kEnd, committed),
                                    kEnd);
    position += committed;
  }
  const auto saved = source->SaveSnapshot(position);
  std::vector<std::uint8_t> payload(saved->CompactPayloadBytes());
  Expect(saved->SerializeCompact(payload) == payload.size(),
         name + ": serialize");
  // Header, records of 16 layers, K and V of 16 layers for the sink and
  // tail rows, then the recurrent state (as for FP16 KV).
  const std::size_t record_row =
      4ULL * gufo::hip::KvarnRecord(gufo::hip::KvarnValueBits(storage)).bytes /
      128;
  const std::size_t kv =
      16 * expected_records * record_row +
      2ULL * 16 * (std::min(position, 128U) + expected_tail) * 2048;
  const auto fp16_policy = gufo::hip::QwenExecutionPolicy::Production();
  auto fp16 = Executor::Create(model, &error, kContext, fp16_policy);
  Expect(fp16 != nullptr, error);
  const auto empty = fp16->SaveSnapshot(0);
  std::vector<std::uint8_t> state(empty->CompactPayloadBytes());
  Expect(payload.size() == state.size() + kv,
         name + ": payload of " + std::to_string(payload.size()) + " bytes");
  target->RestoreCompactSnapshot(payload, position);
  const auto again = target->SaveSnapshot(position);
  std::vector<std::uint8_t> round(again->CompactPayloadBytes());
  Expect(again->SerializeCompact(round) == round.size() && round == payload,
         name + ": a restored compact snapshot differs");
  (void)source->ForwardToken(tokens[position], position);
  const auto expected = Logits(*source);
  (void)target->ForwardToken(tokens[position], position);
  const auto actual = Logits(*target);
  Expect(std::memcmp(expected.data(), actual.data(),
                     expected.size() * sizeof(float)) == 0,
         name + ": a restored compact snapshot decodes differently");
  std::cout << name << ": " << payload.size() << " bytes (" << kv
            << " KV), round trip and decode exact\n";
}

}  // namespace

int main(int argc, const char* const* argv) {
  try {
    const char* path = argc > 1 ? argv[1] : std::getenv("GUFO_QWEN27B_MODEL");
    if (path == nullptr) {
      std::cout << "Set GUFO_QWEN27B_MODEL for the Qwen27B KVarN check.\n";
      return 77;
    }
    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(path, &error);
    Expect(reader != nullptr, error);
    const auto model = gufo::hip::QwenGpuModel::CreateFromGguf(
        std::shared_ptr<const gufo::core::GgufReader>(std::move(reader)),
        &error);
    Expect(model != nullptr, error);
    Expect(model->GetConfig().hidden_size == 5120,
           "the KVarN check requires Qwen3.8 27B");
    const bool continuation_only =
        argc > 2 && std::string_view(argv[2]) == "--continuation-only";
    bool failed = false;
    for (const auto storage :
         {QwenKvCacheStorage::kKvarnK4V2, QwenKvCacheStorage::kKvarnK4V4}) {
      try {
        // Inside the sink; at its end; inside a group; at the block boundary
        // 512 with rows reaching into a block without records (the K2 bug);
        // at a group end the next pass flushes (128 tail rows); and with a
        // rejected draft whose group end the tail was flushed past.
        CheckContinuation(model, storage, 60, 70, false, 0, 70, 0);
        CheckContinuation(model, storage, 100, 128, true, 0, 128, 0);
        CheckContinuation(model, storage, 300, 646, false, 0, 640, 6);
        CheckContinuation(model, storage, 520, 530, true, 0, 512, 18);
        CheckContinuation(model, storage, 600, 640, true, 0, 512, 128);
        CheckContinuation(model, storage, 600, 636, true, 2, 512, 126);
        CheckCompact(model, storage, 60, 70, 0, 70, 0);
        CheckCompact(model, storage, 300, 646, 0, 640, 6);
        CheckCompact(model, storage, 600, 640, 0, 512, 128);
        CheckCompact(model, storage, 600, 636, 2, 512, 126);
        if (!continuation_only)
          CheckStorage(model, storage);
      } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        failed = true;
      }
    }
    if (failed)
      return 1;
    std::cout << "Qwen27B KVarN decode checks passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
