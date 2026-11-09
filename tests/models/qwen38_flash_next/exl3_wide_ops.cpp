// Flash-Next wide copy verification case of qwen38_flash_next_exl3_ops_test:
// Executor::ForwardWide against consecutive ordinary verifications.

#include "tests/models/qwen38_flash_next/exl3_ops_fixture.hpp"

/// A wide copy verification (ForwardWide) must give every row the logits and
/// trunk hidden rows of consecutive verification passes of at most eight
/// rows, and a rollback to any kept prefix must leave the recurrent state
/// and continue exactly as those passes cut back at the same token. The EXL3
/// fixture has a linear (GDN) layer, a sparse attention layer and the MTP
/// block; chains cross segment boundaries, end mid-segment, keep a one-row
/// last segment and, from the longer prompt, start past the sparse budget.
void WideVerification() {
  BeginCase("executor EXL3 wide copy verification vs eight-row passes");
  MtpExecutorFixture f(64, 128, 6, true, 0, 512, 8);
  auto& exec = *f.exec;
  const auto& c = f.config;
  using Access = gpu::ExecutorTestAccess;
  Require(exec.max_copy_rows() == 64, "fixture must allow 64-row copies");
  const auto vocab = c.vocab_size;
  std::string error;
  const auto prefill = [&](std::uint32_t length) {
    auto session =
        exec.CreateSession(gufo::core::SessionMode::kSpeculative, 512, &error);
    Require(session != nullptr, error);
    std::vector<std::int32_t> prompt(length);
    for (std::uint32_t t = 0; t < length; ++t)
      prompt[t] =
          static_cast<std::int32_t>(1 + (t * 2246822519U + 29U) % (vocab - 2));
    std::vector<float> logits(vocab);
    for (std::uint32_t first = 0; first < length; first += 64) {
      const auto count = std::min(64U, length - first);
      Require(exec.Forward(*session, std::span(prompt).subspan(first, count),
                           first + count == length ? 1 : 0,
                           first + count == length ? logits.data() : nullptr,
                           gpu::Executor::ForwardMode::kPrefill, &error),
              error);
      // The draft block consumes an intermediate pass before the next one
      // (Forward requires it), at the row and position the engine uses.
      if (first + count < length) {
        const auto next = prompt[first + count];
        Require(
            exec.MtpForward(
                *session, std::span(&next, 1),
                static_cast<std::int32_t>(
                    std::min<std::size_t>(count, exec.max_speculative()) - 1),
                {}, &error),
            error);
      }
    }
    return session;
  };
  const auto argmax = [&](std::span<const float> row) {
    return static_cast<std::int32_t>(std::max_element(row.begin(), row.end()) -
                                     row.begin());
  };
  // Three greedy decode steps after the kept prefix.
  const auto continue_from = [&](gpu::Session& session, std::int32_t token) {
    std::vector<float> out, logits(vocab);
    for (int step = 0; step < 3; ++step) {
      Require(exec.Forward(session, std::span<const std::int32_t>(&token, 1), 1,
                           logits.data(), gpu::Executor::ForwardMode::kDecode,
                           &error),
              error);
      out.insert(out.end(), logits.begin(), logits.end());
      token = argmax(logits);
    }
    return out;
  };
  bool equal = true;
  for (const std::uint32_t prompt : {40U, 200U}) {
    for (const std::uint32_t n : {64U, 20U, 17U, 9U}) {
      std::vector<std::int32_t> chain(n);
      for (std::uint32_t t = 0; t < n; ++t)
        chain[t] =
            static_cast<std::int32_t>(1 + (t * 40503U + prompt) % (vocab - 2));
      for (const std::uint32_t keep :
           {n, 1U, 7U, 8U, 9U, 16U, 17U, 40U, 56U, 57U, n - 1}) {
        if (keep == 0 || keep > n)
          continue;
        const auto name = "prompt=" + std::to_string(prompt) +
                          " rows=" + std::to_string(n) +
                          " keep=" + std::to_string(keep);
        auto wide = prefill(prompt);
        std::vector<float> wide_rows(std::size_t{n} * vocab);
        std::vector<float> wide_hidden(std::size_t{n} * c.HcDim());
        // From the longer prompt a fully kept chain is committed by the
        // next forward (CommitPendingSpeculation) instead of Rollback.
        const bool commit_by_forward = keep == n && prompt == 200;
        Require(exec.ForwardWide(*wide, chain, &error) &&
                    exec.SelectBatchLogits(0, n, wide_rows.data(), &error) &&
                    exec.CopyTrunkHidden(*wide, wide_hidden, &error) &&
                    (commit_by_forward || exec.Rollback(*wide, keep, &error)),
                error);
        // Consecutive passes of at most eight rows, the one holding the last
        // kept token cut back to it.
        auto reference = prefill(prompt);
        std::vector<float> rows, hidden;
        for (std::uint32_t first = 0; first < keep; first += 8) {
          const auto count = std::min(8U, n - first);
          std::vector<float> logits(std::size_t{count} * vocab);
          std::vector<float> part(std::size_t{count} * c.HcDim());
          Require(exec.Forward(*reference,
                               std::span<const std::int32_t>(chain).subspan(
                                   first, count),
                               count, logits.data(),
                               gpu::Executor::ForwardMode::kVerify, &error) &&
                      exec.CopyTrunkHidden(*reference, part, &error) &&
                      exec.Rollback(*reference, std::min(count, keep - first),
                                    &error),
                  error);
          rows.insert(rows.end(), logits.begin(), logits.end());
          hidden.insert(hidden.end(), part.begin(), part.end());
        }
        wide_rows.resize(rows.size());
        wide_hidden.resize(hidden.size());
        equal = CompareBits(rows, wide_rows, name + " logits", vocab) && equal;
        equal = CompareBits(hidden, wide_hidden, name + " trunk hidden rows",
                            c.HcDim()) &&
                equal;
        if (!commit_by_forward) {
          const auto a = Access::GdnState(*reference, c);
          const auto b = Access::GdnState(*wide, c);
          equal = CompareBits(a.first, b.first, name + " convolution history",
                              c.SsmConvChannels()) &&
                  equal;
          equal = CompareBits(a.second, b.second, name + " recurrent state",
                              c.ssm_head_dim) &&
                  equal;
        }
        const bool same_position = wide->position() == reference->position();
        std::cout << (same_position ? "PASS " : "FAIL ") << name
                  << " position\n";
        equal = same_position && equal;
        const auto next = argmax(std::span<const float>(wide_rows).subspan(
            std::size_t{keep - 1} * vocab, vocab));
        equal = CompareBits(continue_from(*reference, next),
                            continue_from(*wide, next), name + " continuation",
                            vocab) &&
                equal;
      }
    }
  }
  Require(equal, "wide copy verification differs from eight-row passes");
}
