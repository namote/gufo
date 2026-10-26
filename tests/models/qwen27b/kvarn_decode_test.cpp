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
  const auto config = *gufo::testing::kvarn::ParseConfig(
      storage == QwenKvCacheStorage::kKvarnK4V2 ? "kvarn_k4v2_g128"
                                                : "kvarn_k4v4_g128");
  const auto& c = model->GetConfig();
  struct Step {
    double fp16_real, fp16_fake, fake_real, fake_control;
    bool top1_real, top1_fake, top1_real_fake, top1_fake_control;
  };
  const auto round_trip = [&](gufo::testing::kvarn::Session& session,
                              gufo::testing::kvarn::State& state,
                              std::size_t group) {
    const std::size_t width = std::size_t{c.num_key_value_heads} * c.head_dim;
    std::vector<float> x(c.head_dim);
    for (std::size_t i = 0; i < state.components.size(); ++i) {
      if (!session.IsRows(i))
        continue;
      auto* rows = reinterpret_cast<std::uint16_t*>(state.components[i].data());
      for (std::size_t t = group * 128; t < (group + 1) * 128; ++t)
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
      round_trip(control_session, moved, pos / 128 - 1);
      (void)gufo::testing::kvarn::Quantize(control_session, moved, config,
                                           c.num_key_value_heads, c.head_dim,
                                           pos / 128 - 1, pos / 128);
      control_session.Restore(moved);
    }
    (void)fp16.ForwardToken(tokens[pos], pos);
    (void)kvarn.ForwardToken(tokens[pos], pos);
    (void)fake.ForwardToken(tokens[pos], pos);
    (void)control.ForwardToken(tokens[pos], pos);
    const auto reference = Logits(fp16);
    const auto real = Logits(kvarn);
    const auto faked = Logits(fake);
    const auto controlled = Logits(control);
    if (pos < 128) {
      // The sink alone: rotation and FP16 rounding of rotated rows only.
      const double kl = Kl(reference, real);
      Expect(kl < 1e-3, name + ": sink-only KL too large at " +
                            std::to_string(pos) + ": " + std::to_string(kl));
      continue;
    }
    steps.push_back({Kl(reference, real), Kl(reference, faked), Kl(faked, real),
                     Kl(faked, controlled), Argmax(reference) == Argmax(real),
                     Argmax(reference) == Argmax(faked),
                     Argmax(real) == Argmax(faked),
                     Argmax(controlled) == Argmax(faked)});
    const auto& step = steps.back();
    std::cout << name << " step " << pos << " kl_fp16_real " << step.fp16_real
              << " kl_fp16_fake " << step.fp16_fake << " kl_fake_real "
              << step.fake_real << " top1_fp16_real " << step.top1_real
              << " top1_fp16_fake " << step.top1_fake << " top1_real_fake "
              << step.top1_real_fake << " kl_fake_control " << step.fake_control
              << " top1_fake_control " << step.top1_fake_control << '\n';
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
    std::cout << name << " positions " << from << "-" << to - 1
              << " KL median/p95/max fp16||real " << a[0] << "/" << a[1] << "/"
              << a[2] << " fp16||fake " << b[0] << "/" << b[1] << "/" << b[2]
              << " fake||real " << d[0] << "/" << d[1] << "/" << d[2]
              << " fake||control " << e[0] << "/" << e[1] << "/" << e[2]
              << '\n';
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
  // amplification: per window, the real path may depart from the fake one
  // by at most ten times the control's departure (p95 and max), and agree
  // on top-1 no worse than the control by more than one step in 100.
  for (const auto [from, to] : {std::pair{256U, 384U}, std::pair{384U, 512U},
                                std::pair{512U, kPrompt}}) {
    const auto real_fake = summary(&Step::fake_real, from, to);
    const auto fake_control = summary(&Step::fake_control, from, to);
    for (const std::size_t q : {1U, 2U})
      Expect(real_fake[q] <= 10 * std::max(fake_control[q], 1e-6),
             name +
                 ": real KVarN departs from K0 fake quant beyond the "
                 "rounding control at positions " +
                 std::to_string(from) + "-" + std::to_string(to - 1));
  }
  Expect(real_fake_agree * 100 + steps.size() >= control_agree * 100,
         name + ": real and fake KVarN disagree on top-1 beyond the control");

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

  // The snapshot restores the state exactly; its compact form waits for
  // a KVarN checkpoint layout.
  kvarn.RestoreSnapshot(*snapshot);
  (void)kvarn.ForwardToken(verify[0], kPrompt);
  Expect(std::memcmp(Logits(kvarn).data(), first_scalar.data(),
                     vocab * sizeof(float)) == 0,
         name + ": a restored snapshot decodes differently");
  try {
    std::vector<std::uint8_t> payload(snapshot->CompactPayloadBytes());
    Expect(false, name + ": compact snapshots must refuse");
  } catch (const std::logic_error&) {
  }

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
    for (const auto storage :
         {QwenKvCacheStorage::kKvarnK4V2, QwenKvCacheStorage::kKvarnK4V4})
      CheckStorage(model, storage);
    std::cout << "Qwen27B KVarN decode checks passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
