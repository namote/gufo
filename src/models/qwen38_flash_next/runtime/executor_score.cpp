// Packed scoring: several sessions' prompt (kPrefill) rows in one pass.

#include "src/models/qwen38_flash_next/runtime/executor_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

bool Executor::ForwardScorePack(std::span<const ScorePackItem> items,
                                std::span<const ScoredLogprob>* results,
                                std::string* error_msg) const {
  const Config& c = config();
  *results = {};
  PrefillPhase phase(true);
  selected_logits_ = nullptr;
  wide_rows_ = 0;
  DropReadyQ8();
  if (items.empty() || items.size() > kScorePackSessions) {
    AssignError(error_msg, "a scoring pack holds 1..32 sessions");
    return false;
  }
  std::array<std::uint32_t, kScorePackSessions> offsets{};
  std::array<std::uint32_t, kScorePackSessions> logit_offsets{};
  std::array<bool, kScorePackSessions> sparse{};
  std::array<std::uint32_t, kScorePackSessions> complete{};
  std::uint32_t rows = 0;
  std::uint32_t logit_rows = 0;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    const auto n = static_cast<std::uint32_t>(item.tokens.size());
    if (item.session == nullptr || item.session->owner_ != this || n == 0 ||
        item.targets.size() > n || item.session->kvarn_ ||
        item.session->VisionLayout().PrefixLength() != 0 ||
        item.session->position_ + n > item.session->max_context_) {
      AssignError(error_msg, "invalid session or rows in a scoring pack");
      return false;
    }
    for (std::size_t j = 0; j < i; ++j)
      if (items[j].session == item.session) {
        AssignError(error_msg, "a scoring pack names a session twice");
        return false;
      }
    for (const auto token : item.tokens)
      if (token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) {
        AssignError(error_msg, "token out of range");
        return false;
      }
    offsets[i] = rows;
    logit_offsets[i] = logit_rows;
    rows += n;
    logit_rows += static_cast<std::uint32_t>(item.targets.size());
    const auto end = item.session->position_ + n;
    sparse[i] = c.compress_ratio > 0 && end > c.indexer_top_k;
    complete[i] = c.compress_ratio > 0 ? end / c.compress_ratio : 0;
  }
  if (rows > options_.max_batch) {
    AssignError(error_msg, "a scoring pack exceeds the executor's rows");
    return false;
  }
  // Pinned per-session controls and the shared MoE output, allocated on first
  // use, beside the scoring buffers (EnsureScoreBuffers).
  const auto allocate = [&] {
    if (score_controls_ == nullptr) {
      void* controls = nullptr;
      if (!Check(host_allocations_.Allocate(
                     &controls, kScorePackSessions * sizeof(Session::Control)),
                 "pinned scoring controls", error_msg))
        return false;
      score_controls_ = static_cast<Session::Control*>(controls);
    }
    if (score_moe_out_ == nullptr &&
        !Check(hip::AllocateDevice(&score_moe_out_,
                                   std::size_t{options_.max_batch} *
                                       c.hidden_size * sizeof(float)),
               "scoring MoE output", error_msg))
      return false;
    return true;
  };
  if (!allocate()) {
    (void)hipGetLastError();
    return false;
  }
  if (!EnsureScoreBuffers(error_msg))
    return false;
  for (std::size_t i = 0; i < items.size(); ++i)
    std::copy(items[i].targets.begin(), items[i].targets.end(),
              score_targets_host_ + logit_offsets[i]);
  for (const auto& item : items)
    if (!item.session->CheckCancellation(error_msg))
      return false;
  if (ple_pending_ && !WaitPle(error_msg))
    return false;
  FinishPrefetch();
  // Per-session preparation, as Forward does for one session. Scoring
  // sessions skip the draft block: the caller invalidates them afterwards.
  for (std::size_t i = 0; i < items.size(); ++i) {
    auto& session = *items[i].session;
    const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
    const auto end = session.position_ + n;
    if (!CommitPendingSpeculation(session, error_msg))
      return false;
    session.EnsureKvRows(end);
    if (session.continuation_hooks_)
      session.continuation_hooks_->BeforeWrite(
          false, session.position_, end, session.blocks_,
          end > c.indexer_top_k ? end / c.compress_ratio : session.blocks_);
    session.PreserveSnapshots(session.position_, session.mtp_.position);
    ++session.mutation_epoch_;
    session.spec_base_ = session.position_;
    session.spec_tokens_ = 0;
    score_controls_[i] = {session.position_, session.blocks_,
                          session.mtp_.position, -1, session.mtp_.blocks};
    std::copy(items[i].tokens.begin(), items[i].tokens.end(),
              tokens_host_ + offsets[i]);
  }
  if (c.ple_layer >= 0) {
    if (ngram_ == nullptr) {
      AssignError(error_msg, "n-gram table is not open");
      return false;
    }
    ngram_owners_.clear();
    for (std::size_t i = 0; i < items.size(); ++i) {
      auto& session = *items[i].session;
      const auto n = items[i].tokens.size();
      ngram_owners_.push_back({n * c.ple_heads, &session.ngram_counters_});
      HashNgramRows(c, session.ngram_, items[i].tokens,
                    std::span(host_rows_)
                        .subspan(offsets[i] * c.ple_heads, n * c.ple_heads));
    }
    if (!ngram_->StartRead(
            std::span(host_rows_).first(std::size_t{rows} * c.ple_heads),
            std::span(host_emb_, std::size_t{rows} * c.PleEmbeddingDim()),
            ngram_owners_)) {
      AssignError(error_msg, "n-gram table read could not start");
      return false;
    }
    ple_pending_ = true;
  }

  // Row-wise work (embedding, MoE, head) runs once over every row, so the
  // routed experts stream once for the pack. Everything that reads a
  // session's own state or a row's neighbours runs per segment, in the
  // order a single-session prefill pass of that segment's rows takes, on
  // the segment's view of the scratch rows: PLE, the mixers, attention or
  // GatedDeltaNet with their projections, and the combines with the
  // session's abliteration. Each segment's block runs from its FFN combine
  // of the previous layer through its FFN mixer of this one, so the
  // executor-level staging those steps hand on (F16/Q8 norms, deferred
  // norms, inject partials) never crosses a segment. Nothing is deferred
  // across the shared MoE: the attention combine settles its residual
  // update and the MoE writes its own output buffer.
  const Scratch base = s_;
  const auto view = [&](std::size_t i) {
    auto v = RowScratch(base, offsets[i]);
    v.block_out = score_moe_out_ + std::size_t{offsets[i]} * c.hidden_size;
    return v;
  };
  std::array<bool, kScorePackSessions> normed{};
  const auto& layers = model_->layers();
  // The FFN combine of layer `il` for segment i: the MoE output rows the
  // segment's view names, the next mixer's norm (unless the session
  // projects its residual), and that projection.
  const auto ffn_combine = [&](std::size_t i, std::uint32_t il) {
    auto& session = *items[i].session;
    const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
    const float* const residual_settings = ResidualAbliteration(session);
    const float* next_norm =
        il + 1 < c.num_layers && residual_settings == nullptr
            ? (c.IsPleLayer(il + 1) ? nullptr
                                    : layers[il + 1].hc_attn.norm.f32())
            : nullptr;
    const float* next_inject =
        next_norm != nullptr ? InjectWeights(layers[il + 1].hc_attn) : nullptr;
    Combine(s_.res, next_norm, n, next_inject, false,
            AbliterationAt(session, il, true));
    if (residual_settings != nullptr) {
      ApplyPendingResidual();
      const AbliterationSpan span{n, residual_settings};
      AbliterateResidualBatchRows(s_.res, abliteration_directions_,
                                  std::span(&span, 1), il, c.num_layers, n,
                                  c.hidden_size, c.hc_count, stream_);
    }
    normed[i] = next_norm != nullptr;
  };
  // Layer `il` of segment i up to its FFN mixer.
  const auto mixer_block = [&](std::size_t i, std::uint32_t il) {
    auto& session = *items[i].session;
    const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
    const DeviceLayer& l = layers[il];
    if (c.IsPleLayer(il) && !Ple(l, session, n, s_.res, false, error_msg, true))
      return false;
    const bool half_only = WidePrefill(n) && l.linear &&
                           DenseF16Route(l.ssm_in, n) &&
                           l.ssm_alpha_beta.type == GgmlType::kF16 &&
                           l.ssm_alpha_beta.cols == l.ssm_in.cols;
    const std::array<const DeviceTensor*, 3> mixer_rotate =
        l.linear ? std::array<const DeviceTensor*, 3>{&l.ssm_qkv, &l.ssm_gate}
                 : std::array<const DeviceTensor*, 3>{&l.attn_q, &l.attn_k,
                                                      &l.attn_v};
    const std::size_t mixer_count = l.linear ? (l.ssm_in.empty() ? 2 : 0) : 3;
    if (!HcMix(l.hc_attn, s_.res, normed[i], s_.mixed, s_.inject, n, error_msg,
               half_only, false, std::span(mixer_rotate.data(), mixer_count)))
      return false;
    if (l.linear) {
      if (!LinearAttention(l, session.linear_[il], s_.mixed, s_.block_out, n,
                           false, error_msg, false, true))
        return false;
    } else {
      const std::uint32_t pool_grid = sparse[i] && complete[i] > session.blocks_
                                          ? complete[i] - session.blocks_
                                          : 0;
      if (!Attention(l, session.attention_[il], s_.mixed, s_.block_out, n,
                     &session.control_->position, &session.control_->blocks,
                     session.position_, pool_grid, session.max_context_,
                     sparse[i], error_msg, false, false, true))
        return false;
    }
    defer_residual_ = false;
    Combine(s_.res, l.hc_ffn.norm.f32(), n, InjectWeights(l.hc_ffn), false,
            AbliterationAt(session, il, false));
    ApplyPendingResidual();
    const DeviceTensor* const shared_rotate[] = {&l.shexp_gate, &l.shexp_up};
    return HcMix(l.hc_ffn, s_.res, true, s_.mixed, s_.inject, n, error_msg,
                 false, false, shared_rotate);
  };
  const auto body = [&]() -> bool {
    for (std::size_t i = 0; i < items.size(); ++i)
      if (!Check(hipMemcpyAsync(items[i].session->control_, score_controls_ + i,
                                sizeof(Session::Control), hipMemcpyHostToDevice,
                                stream_),
                 "scoring control upload", error_msg))
        return false;
    if (!Check(hipMemcpyAsync(base.tokens, tokens_host_,
                              std::size_t{rows} * sizeof(std::int32_t),
                              hipMemcpyHostToDevice, stream_),
               "scoring token upload", error_msg))
      return false;
    exl3_res_pending_ = {};
    EmbedTokens(model_->token_embd().data, SmallType(model_->token_embd().type),
                base.tokens, base.res, rows, c.hidden_size, c.hc_count,
                stream_);
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      // As ForwardBody checks each layer: a cancelled (or timed-out) member
      // stops the whole pack, whose caller then scores the others alone.
      for (const auto& item : items)
        if (!item.session->CheckCancellation(error_msg))
          return false;
      if (c.IsPleLayer(il) && !WaitPle(error_msg))
        return false;
      for (std::size_t i = 0; i < items.size(); ++i) {
        UseScratch(view(i));
        if (il > 0)
          ffn_combine(i, il - 1);
        if (!mixer_block(i, il))
          return false;
        ApplyPendingResidual();
      }
      UseScratch(base);
      exl3_moe_defer_ = false;
      if (!Moe(layers[il], base.mixed, score_moe_out_, rows, error_msg))
        return false;
    }
    for (std::size_t i = 0; i < items.size(); ++i) {
      UseScratch(view(i));
      ffn_combine(i, c.num_layers - 1);
      ApplyPendingResidual();
    }
    // Each segment's head rows, packed into the mixed rows in item order.
    PrefillPhase head_phase(false);
    for (std::size_t i = 0; i < items.size(); ++i) {
      const auto k = static_cast<std::uint32_t>(items[i].targets.size());
      if (k == 0)
        continue;
      const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
      UseScratch(base);
      if (!HcMix(model_->hc_head(),
                 base.res + std::size_t{offsets[i] + n - k} * c.HcDim(), false,
                 base.mixed + std::size_t{logit_offsets[i]} * c.hidden_size,
                 nullptr, k, error_msg))
        return false;
    }
    // The head in groups of max_score_rows() rows (the head output buffer),
    // each reduced on the device to its rows' target log-softmax before the
    // next group overwrites it; only the reduced rows come to the host.
    if (logit_rows != 0 &&
        !Check(hipMemcpyAsync(score_targets_, score_targets_host_,
                              std::size_t{logit_rows} * sizeof(std::int32_t),
                              hipMemcpyHostToDevice, stream_),
               "scoring target upload", error_msg))
      return false;
    for (std::uint32_t first = 0; first < logit_rows;) {
      const auto count = std::min(max_score_rows(), logit_rows - first);
      UseScratch(base);
      if (!Dense(model_->output(),
                 base.mixed + std::size_t{first} * c.hidden_size, score_logits_,
                 count, error_msg) ||
          !ScoreHeadRows(first, count, error_msg))
        return false;
      first += count;
    }
    UseScratch(base);
    return logit_rows == 0 ||
           Check(hipMemcpyAsync(score_results_host_, score_results_,
                                std::size_t{logit_rows} * sizeof(ScoredLogprob),
                                hipMemcpyDeviceToHost, stream_),
                 "scoring results download", error_msg);
  };
  bool ok = false;
  try {
    ok = body();
  } catch (...) {
    (void)hipStreamSynchronize(stream_);
    UseScratch(base);
    throw;
  }
  const auto status = hipStreamSynchronize(stream_);
  UseScratch(base);
  if (!ok || !Check(status, "scoring pack", error_msg))
    return false;
  for (std::size_t i = 0; i < items.size(); ++i) {
    auto& session = *items[i].session;
    session.position_ += static_cast<std::uint32_t>(items[i].tokens.size());
    if (sparse[i])
      session.blocks_ = complete[i];
  }
  *results = std::span<const ScoredLogprob>(score_results_host_, logit_rows);
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
