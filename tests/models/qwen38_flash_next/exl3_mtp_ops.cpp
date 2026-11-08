// Flash-Next EXL3 MTP cases of qwen38_flash_next_exl3_ops_test: executor
// catch-up and grouped MTP K/V.

#include "tests/models/qwen38_flash_next/exl3_ops_fixture.hpp"

void MtpExecutorCatchUp(bool batched) {
  BeginCase(batched ? "executor EXL3 ragged batch cache-only YaRN"
                    : "executor EXL3 scalar cache-only YaRN");
  MtpExecutorFixture f;
  auto& exec = *f.exec;
  using Access = gpu::ExecutorTestAccess;
  const unsigned sessions = batched ? 3 : 1;
  std::vector<std::unique_ptr<gpu::Session>> reference(sessions),
      cached(sessions);
  const std::array<unsigned, 3> prefix_rows{6, 7, 5}, hidden_rows{2, 0, 4};
  const std::array<std::vector<std::int32_t>, 3> tokens{
      {{11, 12, 13}, {21, 22}, {31}}};
  std::string error;
  const auto state_equal = [&](unsigned i, const std::string& stage) {
    const auto a = Access::ReadState(*reference[i], f.config);
    const auto b = Access::ReadState(*cached[i], f.config);
    bool equal = a.position == b.position && a.blocks == b.blocks;
    std::cout << (equal ? "PASS " : "FAIL ") << stage
              << " position/block shadows\n";
    equal =
        CompareBits(a.k, b.k, stage + " K cache", f.config.AttentionKvDim()) &&
        equal;
    equal =
        CompareBits(a.v, b.v, stage + " V cache", f.config.AttentionKvDim()) &&
        equal;
    equal = CompareBits(a.raw, b.raw,
                        stage + " raw indexer ring including untouched rows",
                        f.config.indexer_head_dim) &&
            equal;
    equal = CompareBits(a.pooled, b.pooled, stage + " pooled indexer keys",
                        f.config.indexer_head_dim) &&
            equal;
    return equal;
  };
  std::vector<gpu::Executor::MtpBatchItem> items;
  unsigned expected_rows = 0;
  for (unsigned i = 0; i < sessions; ++i) {
    reference[i] =
        exec.CreateSession(gufo::core::SessionMode::kSpeculative, 128, &error);
    cached[i] =
        exec.CreateSession(gufo::core::SessionMode::kSpeculative, 128, &error);
    Require(reference[i] && cached[i], error);
    auto hidden = Input(8 * f.config.HcDim());
    for (std::size_t j = 0; j < hidden.size(); ++j)
      hidden[j] += 0.001F * static_cast<float>((j + 7 * i) % 17);
    for (auto* session : {reference[i].get(), cached[i].get()}) {
      Access::Hidden(*session, hidden);
      std::vector<std::int32_t> prefix(prefix_rows[i]);
      for (unsigned r = 0; r < prefix.size(); ++r)
        prefix[r] = 40 + i * 10 + r;
      Require(exec.MtpForward(*session, prefix, 0, {}, &error), error);
    }
    Require(
        exec.MtpForward(*reference[i], tokens[i], hidden_rows[i], {}, &error),
        error);
    items.push_back({cached[i].get(), tokens[i],
                     static_cast<std::int32_t>(hidden_rows[i]), true});
    expected_rows += tokens[i].size();
  }
  Require(exec.BeginMtpCycleDiagnostics(&error), error);
  if (batched)
    Require(exec.MtpForwardBatch(items, &error), error);
  else
    Require(exec.MtpForward(*cached[0], tokens[0], hidden_rows[0],
                            {.kv_only = true}, &error),
            error);
  gpu::Executor::MtpCycleDiagnostics diagnostic;
  Require(exec.EndMtpCycleDiagnostics(0, 0, &diagnostic, &error), error);
  Require(diagnostic.cache_only_rows == expected_rows,
          "EXL3 cache-only dispatch did not execute every requested row");
  Require(diagnostic.expert_tile_reads == 0,
          "cache-only catch-up unexpectedly executed experts");
  bool equal = true;
  for (unsigned i = 0; i < sessions; ++i) {
    equal =
        state_equal(i, "ragged catch-up session=" + std::to_string(i)) && equal;
    Require(!Access::ReadState(*cached[i], f.config).residual_valid,
            "cache-only catch-up exposed a carried residual");
  }
  // The first two ragged sessions cross the sparse threshold and finish
  // pooling blocks. Prove the fixture distinguishes the formerly buggy
  // unscaled pooling kernel, rather than merely comparing two equal zeros.
  Require(Access::ReadState(*cached[0], f.config).blocks >= 2,
          "fixture did not pool blocks across sparse threshold");
  Require(Access::YarnWitness(exec, *reference[0]),
          "fixture cannot distinguish scaled and unscaled pooled keys");
  // Eager -> capture -> replay of the same scalar cache-only shape, at new
  // positions with changed tokens. The ragged batch above remains a real
  // MtpForwardBatch call; none of these checks substitutes projection helpers.
  std::size_t captured_graphs = 0;
  for (unsigned pass = 0; pass < 3; ++pass) {
    const std::array<std::int32_t, 3> more{
        static_cast<std::int32_t>(70 + pass),
        static_cast<std::int32_t>(80 + pass),
        static_cast<std::int32_t>(90 + pass)};
    Require(exec.MtpForward(*reference[0], more, 2, {}, &error), error);
    Require(exec.MtpForward(*cached[0], more, 2, {.kv_only = true}, &error),
            error);
    equal =
        state_equal(0, "graph catch-up pass=" + std::to_string(pass)) && equal;
    if (pass == 1) {
      captured_graphs = Access::Graphs(*cached[0]);
      Require(captured_graphs != 0, "cache-only graph was not captured");
    }
    if (pass == 2)
      Require(Access::Graphs(*cached[0]) == captured_graphs,
              "cache-only replay created another graph instead of reusing it");
  }
  for (unsigned i = 0; i < sessions; ++i) {
    const std::int32_t next = 101 + i;
    for (auto* session : {reference[i].get(), cached[i].get()}) {
      if (i == 0)
        Access::PoisonSelection(exec);
      Require(exec.MtpForward(*session, std::span(&next, 1), 1, {}, &error),
              error);
      if (i == 0)
        Require(Access::HasExpectedSparseSelection(exec, *session),
                "sparse selector must replace poison with exactly the nonempty "
                "causal budget");
    }
    equal = state_equal(i, "known-hidden continuation") && equal;
    const auto a = Access::FullLogits(exec, *reference[i]);
    const auto b = Access::FullLogits(exec, *cached[i]);
    Require(std::ranges::all_of(a, [](float v) { return std::isfinite(v); }),
            "continuation logits are nonfinite");
    equal = CompareBits(a, b, "subsequent full-vocabulary logits",
                        f.config.vocab_size) &&
            equal;
    for (auto* session : {reference[i].get(), cached[i].get()})
      Require(exec.MtpForward(*session, std::span(&next, 1), -1, {}, &error),
              error);
    equal = state_equal(i, "recursive continuation") && equal;
    equal =
        CompareBits(Access::FullLogits(exec, *reference[i]),
                    Access::FullLogits(exec, *cached[i]),
                    "recursive full-vocabulary logits", f.config.vocab_size) &&
        equal;
  }
  Require(equal, "all MTP states and continuation logits match bitwise");
}

void MtpKvGroup() {
  namespace fast = gufo::hip;
  constexpr unsigned k = 256, q_width = 1024, kv_width = 512;
  Buffers buffers;
  Bank q(buffers, k, q_width, 7, 1, 17);
  Bank key(buffers, k, kv_width, 7, 1, 71);
  Bank value(buffers, k, kv_width, 7, 1, 137);
  auto scratch = Scratch(buffers, 64, k, q_width);
  const auto& p = scratch.projection;
  const fast::Exl3FastWorkspace ws{
      p.fast_input,         p.fast_input_elements, p.fast_counters,
      p.fast_counter_count, p.fast_down_input,     p.fast_down_input_elements};
  bool equal = true;
  for (const unsigned rows : {1U, 2U, 3U, 7U, 8U, 9U, 32U, 33U, 64U}) {
    BeginCase("MTP K7 QKV vs KV group rows=" + std::to_string(rows));
    const auto input = Input(rows * k);
    auto* x = buffers.Copy(input);  // ends at the last row's allocation
    auto* query = buffers.Make<float>(rows * q_width);
    auto* k_full = buffers.Make<float>(rows * kv_width);
    auto* v_full = buffers.Make<float>(rows * kv_width);
    auto* k_only = buffers.Make<float>(rows * kv_width + 8);
    auto* v_only = buffers.Make<float>(rows * kv_width + 8);
    const std::array<fast::Exl3FastTarget, 3> full{
        {{q.weight.exl3, query, q_width},
         {key.weight.exl3, k_full, kv_width},
         {value.weight.exl3, v_full, kv_width}}};
    const std::array<fast::Exl3FastTarget, 2> kv{
        {{key.weight.exl3, k_only, kv_width},
         {value.weight.exl3, v_only, kv_width}}};
    Require(fast::Exl3FastGroupSupported(full, ws) &&
                fast::Exl3FastGroupSupported(kv, ws),
            "K7 full and K/V groups supported");
    Check(hipMemset(k_only, 0x7F, (rows * kv_width + 8) * sizeof(float)));
    Check(hipMemset(v_only, 0x7F, (rows * kv_width + 8) * sizeof(float)));
    fast::LaunchExl3FastProjections(full, x, k, rows, ws, nullptr);
    fast::LaunchExl3FastProjections(kv, x, k, rows, ws, nullptr, false,
                                    std::size_t{q_width} + 2 * kv_width);
    for (const auto& [bank, a, b] : {std::tuple{&key, k_full, k_only},
                                     std::tuple{&value, v_full, v_only}}) {
      std::vector<float> expected(rows * kv_width), actual(expected.size());
      Check(hipMemcpy(expected.data(), a, expected.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
      Check(hipMemcpy(actual.data(), b, actual.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
      equal = CompareBits(expected, actual,
                          std::string(bank == &key ? "K" : "V") +
                              " removing Q, rows=" + std::to_string(rows),
                          kv_width) &&
              equal;
      for (const unsigned row : {0U, rows - 1})
        Near(std::span(actual).subspan(row * kv_width, kv_width),
             bank->Reference(0, input.data() + row * k, true), 1e-5,
             "K/V group vs independent FP64 projection");
      std::uint32_t guard;
      Check(hipMemcpy(&guard, b + rows * kv_width, sizeof(guard),
                      hipMemcpyDeviceToHost));
      Require(guard == 0x7F7F7F7FU, "K/V group preserves output guard");
    }
  }
  Require(equal, "all K/V group projections preserve full-QKV bits");
}
