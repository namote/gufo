// KVarN decode and verification through the Qwen3.8-27B executor (K1c-1,
// plan/gufo-upstream-port/K1-BRIEF.md): a prompt fed token by token (the
// first rows through the captured decode graph, then split-K with group
// flushes) against the FP16 executor under teacher forcing, DFlash2-style
// verification rows against scalar decoding of the same tokens, and the
// refusals that stay until prefill (K1c-2) and snapshots (K2) exist.
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

  // Release entry points still refuse KVarN.
  std::string error;
  Expect(Executor::Create(model, &error, kContext, policy) == nullptr &&
             error.find("not implemented yet") != std::string::npos,
         name + ": release executor creation must refuse");

  Executor fp16(model, kContext);
  Executor kvarn(model, kContext, policy,
                 gufo::hip::QwenKvarnDecodeOnlyAccess{});
  const auto tokens = Fixture(fp16, kPrompt + kRows + kTail);

  try {
    (void)kvarn.ForwardPromptBatch(std::span(tokens).first(16));
    Expect(false, name + ": prompt prefill must refuse");
  } catch (const std::logic_error& refusal) {
    Expect(std::string_view(refusal.what()).find("not implemented yet") !=
               std::string_view::npos,
           name + ": prefill refusal names the gap");
  }
  kvarn.Reset();

  // Teacher forcing: three executors read the same tokens. `fake` is the
  // FP16 executor of the K0 study whose K/V rows of every group that ended
  // at or before the next position are fake-quantized in place
  // (kvarn_fake_quant.hpp), the rule KvarnReadGroups gives the real path.
  gufo::testing::kvarn::Session fake_session(model, kContext);
  auto& fake = fake_session.Executor();
  const auto config = *gufo::testing::kvarn::ParseConfig(
      storage == QwenKvCacheStorage::kKvarnK4V2 ? "kvarn_k4v2_g128"
                                                : "kvarn_k4v4_g128");
  const auto& c = model->GetConfig();
  struct Step {
    double fp16_real, fp16_fake, fake_real;
    bool top1_real, top1_fake, top1_real_fake;
  };
  std::vector<Step> steps;
  for (std::uint32_t pos = 0; pos < kPrompt; ++pos) {
    if (pos >= 256 && pos % 128 == 0) {
      auto state = fake_session.Capture();
      (void)gufo::testing::kvarn::Quantize(fake_session, state, config,
                                           c.num_key_value_heads, c.head_dim,
                                           pos / 128 - 1, pos / 128);
      fake_session.Restore(state);
    }
    (void)fp16.ForwardToken(tokens[pos], pos);
    (void)kvarn.ForwardToken(tokens[pos], pos);
    (void)fake.ForwardToken(tokens[pos], pos);
    const auto reference = Logits(fp16);
    const auto real = Logits(kvarn);
    const auto faked = Logits(fake);
    if (pos < 128) {
      // The sink alone: rotation and FP16 rounding of rotated rows only.
      const double kl = Kl(reference, real);
      Expect(kl < 1e-3, name + ": sink-only KL too large at " +
                            std::to_string(pos) + ": " + std::to_string(kl));
      continue;
    }
    steps.push_back({Kl(reference, real), Kl(reference, faked), Kl(faked, real),
                     Argmax(reference) == Argmax(real),
                     Argmax(reference) == Argmax(faked),
                     Argmax(real) == Argmax(faked)});
    const auto& step = steps.back();
    std::cout << name << " step " << pos << " kl_fp16_real " << step.fp16_real
              << " kl_fp16_fake " << step.fp16_fake << " kl_fake_real "
              << step.fake_real << " top1_fp16_real " << step.top1_real
              << " top1_fp16_fake " << step.top1_fake << " top1_real_fake "
              << step.top1_real_fake << '\n';
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
  std::size_t real_agree = 0, fake_agree = 0, real_fake_agree = 0;
  for (const auto& step : steps) {
    real_agree += step.top1_real;
    fake_agree += step.top1_fake;
    real_fake_agree += step.top1_real_fake;
  }
  for (const auto [from, to] :
       {std::pair{128U, 256U}, std::pair{256U, 384U}, std::pair{384U, 512U},
        std::pair{512U, kPrompt}, std::pair{128U, kPrompt}}) {
    const auto a = summary(&Step::fp16_real, from, to);
    const auto b = summary(&Step::fp16_fake, from, to);
    const auto d = summary(&Step::fake_real, from, to);
    std::cout << name << " positions " << from << "-" << to - 1
              << " KL median/p95/max fp16||real " << a[0] << "/" << a[1] << "/"
              << a[2] << " fp16||fake " << b[0] << "/" << b[1] << "/" << b[2]
              << " fake||real " << d[0] << "/" << d[1] << "/" << d[2] << '\n';
  }
  std::cout << name << " top-1 of " << steps.size() << ": fp16=real "
            << real_agree << " fp16=fake " << fake_agree << " real=fake "
            << real_fake_agree << '\n';
  // Without a quantized group the rows differ from FP16 only by the
  // rotation's rounding.
  Expect(summary(&Step::fp16_real, 128, 256)[2] < 1e-2,
         name + ": FP16-only tail KL too large");
  Expect(real_agree * 100 >= steps.size() * 95,
         name + ": top-1 agreement below 95 %");
  // Real and fake quantization store the same codes (bit-exact in the
  // kernel test); they differ by attention arithmetic (kernel vs K0 fake
  // quant relative error 7.5e-4) and the extra FP16 rounding of rotated
  // rows, the same order as the rotation rounding whose KL stays below
  // 1.5e-6 in the FP16-only window. The bound allows about 600 times that.
  const auto real_fake = summary(&Step::fake_real, 128, kPrompt);
  Expect(real_fake[2] < 1e-3, name + ": real KVarN departs from K0 fake quant");
  Expect(real_fake_agree * 100 >= steps.size() * 99,
         name + ": real and fake KVarN disagree on top-1");

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
  for (std::uint32_t row = 0; row < kRows; ++row) {
    (void)kvarn.ForwardToken(verify[row], kPrompt + row);
    const auto scalar = Logits(kvarn);
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

  // Decoding on past the pinned point (group 4 stays FP16 until the ring
  // needs it), then a fresh pinned point confirms and quantizes it.
  for (std::uint32_t pos = kPrompt + kRows; pos < kPrompt + kRows + kTail;
       ++pos) {
    if (pos == kPrompt + kRows + kTail / 2)
      kvarn.SaveState(pos);
    (void)kvarn.ForwardToken(tokens[pos], pos);
    (void)Logits(kvarn);
  }

  try {
    (void)kvarn.SaveSnapshot(kPrompt);
    Expect(false, name + ": snapshots must refuse");
  } catch (const std::logic_error&) {
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
