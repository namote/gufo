// The forward pass: Run (graph capture and replay), Forward and its
// per-layer body, speculative commit and rollback, and output-head replay.

#include "src/models/qwen38_flash_next/runtime/executor_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

bool Executor::CopyTrunkHidden(const Session& session, std::span<float> hidden,
                               std::string* error_msg) const {
  if (!session.mtp_enabled_ || session.owner_ != this ||
      session.position_ == 0 || hidden.empty() ||
      hidden.size() % config().HcDim() != 0 ||
      hidden.size() / config().HcDim() >
          std::min(session.position_, options_.max_speculative)) {
    AssignError(error_msg, "invalid trunk hidden diagnostic input");
    return false;
  }
  return Check(hipMemcpyAsync(hidden.data(), session.mtp_.target_hidden,
                              hidden.size_bytes(), hipMemcpyDeviceToHost,
                              stream_),
               "trunk hidden download", error_msg) &&
         Check(hipStreamSynchronize(stream_), "trunk hidden", error_msg);
}

bool Executor::Run(Session& session, std::uint64_t key, bool graph,
                   const std::function<bool()>& body, std::string* error_msg,
                   bool synchronize) const {
  // Host counters must observe each executed operation. Captured/replayed
  // bodies would either duplicate or omit them, as with MtpTrace copies.
  graph = graph && !diagnostics_active_;
  // A batch shape runs eagerly once before it is captured: the first pass
  // grows the GEMM tier's arena, which capture forbids.
  if (graph && session.warmed_.contains(key)) {
    hipGraphExec_t exec = nullptr;
    bool enqueued = false;
    if (const auto it = session.graphs_.find(key);
        it != session.graphs_.end()) {
      exec = it->second;
    } else {
      // A warmed target shape may need its first graph or a replacement
      // after rollback storage was released. Queue this pass first, then
      // record identical commands while the GPU executes it. Short prefix
      // and draft graphs keep their existing route. No second pass is launched.
      if (synchronize && (key & (std::uint64_t{1} << 40)) == 0) {
        auto dispatch = std::tie(
            moe_pending_, defer_residual_, res_deferred_, shexp_half_ready_,
            q8t_src_, q8t_rows_, q8t_cols_, half_src_, half_rows_, half_cols_,
            half_bf16_, xn_half_, inject_parts_, inject_fused_, q8_slot_,
            xn_q8_ready_, mixed_q8_ready_, xn_deferred_.res, xn_deferred_.gamma,
            xn_deferred_.rows, exl3_moe_defer_, exl3_moe_pending_.layer,
            exl3_moe_pending_.rows, xn_q8t_norm_.res, xn_q8t_norm_.gamma,
            xn_q8t_norm_.rows, exl3_rotated_, exl3_res_pending_);
        const auto before = std::apply(
            [](const auto&... fields) { return std::make_tuple(fields...); },
            dispatch);
        if (!body())
          return false;
        // Producer readiness and alternating scratch addresses must match
        // the executed pass when the graph records its kernel arguments.
        dispatch = before;
        enqueued = true;
      }
      hipGraph_t captured = nullptr;
      // Frozen peer sessions can copy snapshots on independent nonblocking
      // streams while the scheduler records this session's decode graph.
      if (!Check(
              hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal),
              "graph capture", error_msg)) {
        return false;
      }
      const bool ok = body();
      if (!Check(hipStreamEndCapture(stream_, &captured), "graph capture end",
                 error_msg) ||
          !ok) {
        if (captured != nullptr) {
          (void)hipGraphDestroy(captured);
        }
        return false;
      }
      const bool instantiated =
          Check(hipGraphInstantiate(&exec, captured, nullptr, nullptr, 0),
                "graph instantiate", error_msg);
      (void)hipGraphDestroy(captured);
      if (!instantiated) {
        return false;
      }
      session.graphs_.emplace(key, exec);
    }
    if (!enqueued &&
        !Check(hipGraphLaunch(exec, stream_), "graph launch", error_msg)) {
      return false;
    }
  } else {
    if (!body()) {
      return false;
    }
    session.warmed_.insert(key);
  }
  return !synchronize ||
         Check(hipStreamSynchronize(stream_), "forward", error_msg);
}

bool Executor::Forward(Session& session, std::span<const std::int32_t> tokens,
                       std::uint32_t n_logits, float* logits, ForwardMode mode,
                       std::string* error_msg, PrefillCheckpoint* checkpoint,
                       std::span<const std::int32_t> next) const {
  DiagnosticPhase diagnostic(*this, DiagnosticStage::kTarget);
  // Only this call's PLE layer may start the prefetch of `next`.
  struct NextHint {
    std::span<const std::int32_t>& hint;
    ~NextHint() { hint = {}; }
  } next_hint{prefetch_next_};
  if (mode == ForwardMode::kPrefill) {
    prefetch_next_ = next;
  }
  const bool speculative = mode == ForwardMode::kVerify;
  PrefillPhase phase(mode == ForwardMode::kPrefill);
  selected_logits_ = nullptr;
  DropReadyQ8();
  const Config& c = config();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  if (checkpoint && (mode != ForwardMode::kPrefill ||
                     checkpoint->source != &session || checkpoint->tokens > n ||
                     checkpoint->state->position_ !=
                         session.position_ + checkpoint->tokens)) {
    AssignError(error_msg, "prefill checkpoint does not match this pass");
    return false;
  }
  if (n == 0 || n > options_.max_batch ||
      (speculative && n > options_.max_speculative)) {
    AssignError(error_msg, "token batch is empty or exceeds the batch limit");
    return false;
  }
  if (n_logits > n || n_logits > options_.max_logit_rows) {
    AssignError(error_msg, "requested logit rows exceed the batch or limit");
    return false;
  }
  if (session.owner_ != this || session.position_ + n > session.max_context_) {
    AssignError(error_msg, "session context is full");
    return false;
  }
  for (auto t : tokens) {
    if (t < 0 || static_cast<std::uint32_t>(t) >= c.vocab_size) {
      AssignError(error_msg, "token out of range");
      return false;
    }
  }
  const std::uint32_t start_pos = session.position_;
  if (!session.CheckCancellation(error_msg))
    return false;
  if ((session.kvarn_ || session.kv_pool_) && checkpoint) {
    AssignError(error_msg,
                "KVarN or shared KV sessions take no in-pass checkpoints");
    return false;
  }
  if (session.mtp_enabled_ && prefill_phase &&
      session.mtp_.position != start_pos) {
    AssignError(error_msg,
                "MTP must consume the preceding frontier before prefill");
    return false;
  }
  // A pending batch nobody rolled back is kept whole before anything
  // reuses its records or rollback rows.
  if (!CommitPendingSpeculation(session, error_msg))
    return false;
  if (speculative && !EnsureRollback(session, n - 1, error_msg))
    return false;
  // Failed eager/captured work can still be reading pinned inputs. Drain it
  // before another request reuses those buffers; successful forwards already
  // synchronize below and need no additional wait.
  struct PinnedInputGuard {
    hipStream_t stream;
    bool completed{false};
    ~PinnedInputGuard() {
      if (!completed)
        (void)hipStreamSynchronize(stream);
    }
  } pinned_inputs{stream_};
  session.EnsureKvRows(start_pos + n);
  // Start the disk rows before protecting checkpoint rows, which can commit
  // device pages: layer 0 alone cannot hide the whole read.
  if (session.continuation_hooks_) {
    const auto end =
        session.position_ + static_cast<std::uint32_t>(tokens.size());
    session.continuation_hooks_->BeforeWrite(
        false, session.position_, end, session.blocks_,
        end > c.indexer_top_k ? end / c.compress_ratio : session.blocks_);
  }
  if (session.kvarn_ && !PlanKvarn(session, start_pos, n, error_msg))
    return false;
  const auto ngram = session.ngram_;
  if (c.ple_layer >= 0) {
    if (ngram_ == nullptr) {
      AssignError(error_msg, "n-gram table is not open");
      return false;
    }
    if (!PleFetch(session, tokens, speculative, error_msg)) {
      return false;
    }
  }
  try {
    session.PreserveSnapshots(session.position_, session.mtp_.position);
  } catch (...) {
    // The forward never ran; leave the n-gram window where it was.
    (void)WaitPle(nullptr);
    session.ngram_ = ngram;
    throw;
  }
  ++session.mutation_epoch_;
  session.spec_base_ = start_pos;
  session.spec_tokens_ = speculative ? n : 0;
  // Everything the launched work reads from the host sits in pinned
  // buffers the graph nodes point at: tokens, the control block (positions
  // the kernels read on the device) and the n-gram rows.
  std::copy(tokens.begin(), tokens.end(), tokens_host_);
  control_host_->position = start_pos;
  control_host_->blocks = session.blocks_;
  control_host_->mtp_position = session.mtp_.position;
  control_host_->mtp_blocks = session.mtp_.blocks;
  control_host_->hidden_row = -1;
  // Sparse selection only changes the result once a query can see more
  // than the token budget; every layer of this model shares one ratio.
  const bool sparse = c.compress_ratio > 0 && start_pos + n > c.indexer_top_k;
  const std::uint32_t complete =
      c.compress_ratio > 0 ? (start_pos + n) / c.compress_ratio : 0;
  const std::uint32_t pool_grid =
      sparse && complete > session.blocks_ ? complete - session.blocks_ : 0;
  // Decode-sized batches replay as graphs; a pooling backlog (the first
  // batch past the budget) needs the wider eager grid.
  const std::uint32_t graph_pool_grid = n / std::max(c.compress_ratio, 1u) + 1;
  const bool graph = !diagnostics_active_ && !prefill_phase && n <= kVecBatch &&
                     pool_grid <= graph_pool_grid &&
                     session.position_ >= session.VisionLayout().PrefixLength();
  const std::uint64_t key = static_cast<std::uint64_t>(n) |
                            (static_cast<std::uint64_t>(n_logits) << 16) |
                            (static_cast<std::uint64_t>(speculative) << 32) |
                            (static_cast<std::uint64_t>(sparse) << 33) |
                            (std::uint64_t{logits != nullptr} << 34);
  // PLE first consumes the disk rows at its injection layer. Queue the
  // preceding layers before waiting, including on captured graph replay.
  // Both pieces use the same stream and arithmetic as the unsplit graph.
  const std::uint32_t first_layer =
      graph && c.ple_layer > 0 ? static_cast<std::uint32_t>(c.ple_layer) : 0;
  if (first_layer > 0) {
    const auto prefix = [&] {
      return ForwardBody(session, n, 0, false, speculative, sparse, start_pos,
                         graph_pool_grid, 0, first_layer, error_msg);
    };
    constexpr std::uint64_t kPrefixKey = std::uint64_t{1} << 35;
    if (!Run(session, key | kPrefixKey, graph, prefix, error_msg, false)) {
      return false;
    }
  }
  const auto body = [&] {
    return ForwardBody(session, n, n_logits, logits != nullptr, speculative,
                       sparse, start_pos, graph ? graph_pool_grid : pool_grid,
                       first_layer, c.num_layers, error_msg, checkpoint, graph);
  };
  // The prefix is already queued. Wait once before the suffix so recording
  // its graph can reuse these pinned rows while the eager pass executes.
  if (graph && ple_pending_ && !WaitPle(error_msg)) {
    return false;
  }
  if (!Run(session, key, graph, body, error_msg)) {
    return false;
  }
  if (n_logits > 0 && logits != nullptr) {
    std::copy_n(logits_host_, static_cast<std::size_t>(n_logits) * c.vocab_size,
                logits);
  }

  session.position_ += n;
  if (diagnostics_active_ && speculative)
    cycle_diagnostics_.verify_rows += n;
  if (sparse) {
    session.blocks_ = complete;
  }
  if (session.kvarn_ && !FinishKvarn(session, error_msg))
    return false;
  if (session.mtp_enabled_ && prefill_phase && n > 1) {
    // Every successor except the last one is already known. Consume it while
    // the trunk residual is still in shared scratch. MtpBody reads that
    // residual into xn before reusing res for the predictor output.
    PrefillPhase draft_phase(false);
    if (!MtpForward(session, tokens.subspan(1), 0, {.kv_only = true}, error_msg,
                    s_.res))
      return false;
  }
  pinned_inputs.completed = true;
  return true;
}

bool Executor::ForwardBody(Session& session, std::uint32_t n,
                           std::uint32_t n_logits, bool download_logits,
                           bool speculative, bool sparse,
                           std::uint32_t start_pos, std::uint32_t pool_grid,
                           std::uint32_t first_layer, std::uint32_t end_layer,
                           std::string* error_msg,
                           PrefillCheckpoint* checkpoint,
                           bool embeddings_ready) const {
  const Config& c = config();
  exl3_res_pending_ = {};  // only a failed earlier pass can leave one
  if (first_layer == 0) {
    if (!Check(hipMemcpyAsync(session.control_, control_host_,
                              sizeof(Session::Control), hipMemcpyHostToDevice,
                              stream_),
               "control upload", error_msg) ||
        !Check(hipMemcpyAsync(s_.tokens, tokens_host_, n * sizeof(std::int32_t),
                              hipMemcpyHostToDevice, stream_),
               "token upload", error_msg)) {
      return false;
    }
    EmbedTokens(model_->token_embd().data, SmallType(model_->token_embd().type),
                s_.tokens, s_.res, n, c.hidden_size, c.hc_count, stream_);
    session.vision_input_.Inject(s_.res, start_pos, n, c.hidden_size,
                                 c.hc_count, stream_);
  }
  const auto& layers = model_->layers();
  bool normed = false;  ///< xn holds the next mixer's grouped norm of res
  for (std::uint32_t il = first_layer; il < end_layer; ++il) {
    if (!session.CheckCancellation(error_msg))
      return false;
    const DeviceLayer& l = layers[il];
    if (c.IsPleLayer(il) && !Ple(l, session, n, s_.res, speculative, error_msg,
                                 embeddings_ready, checkpoint)) {
      return false;
    }
    // Both stacked SSM projections consume the same F16 input. Avoid writing
    // unused F32/Q8 copies when the wide mixer can supply that input directly.
    const bool half_only = WidePrefill(n) && n <= options_.max_batch &&
                           l.linear && DenseF16Route(l.ssm_in, n) &&
                           l.ssm_alpha_beta.type == GgmlType::kF16 &&
                           l.ssm_alpha_beta.cols == l.ssm_in.cols;
    // The mixer's EXL3 decode projections of `mixed`.
    const std::array<const DeviceTensor*, 3> mixer_rotate =
        l.linear ? std::array<const DeviceTensor*, 3>{&l.ssm_qkv, &l.ssm_gate}
                 : std::array<const DeviceTensor*, 3>{&l.attn_q, &l.attn_k,
                                                      &l.attn_v};
    const std::size_t mixer_count = l.linear ? (l.ssm_in.empty() ? 2 : 0) : 3;
    if (!HcMix(l.hc_attn, s_.res, normed, s_.mixed, s_.inject, n, error_msg,
               half_only, false, std::span(mixer_rotate.data(), mixer_count))) {
      return false;
    }
    // The last target layer's query outputs have no later cache consumer.
    // Preserve all K/V/indexer rows and the final query's original tile.
    const bool last_attention = WidePrefill(n) && !session.mtp_enabled_ &&
                                il + 1 == c.num_layers && n_logits == 1 &&
                                checkpoint == nullptr;
    if (l.linear) {
      const auto capture =
          checkpoint ? GdnCheckpoint{checkpoint->state->linear_[il].state,
                                     checkpoint->state->linear_[il].conv_state,
                                     checkpoint->tokens}
                     : GdnCheckpoint{};
      if (!LinearAttention(l, session.linear_[il], s_.mixed, s_.block_out, n,
                           speculative, error_msg, false, true, capture)) {
        return false;
      }
    } else if (!Attention(l, session.attention_[il], s_.mixed, s_.block_out, n,
                          &session.control_->position,
                          &session.control_->blocks, start_pos, pool_grid,
                          session.max_context_, sparse, error_msg,
                          last_attention, false, true,
                          checkpoint ? checkpoint->tokens : 0)) {
      return false;
    }

    const float* next_norm =
        il + 1 < c.num_layers
            ? (c.IsPleLayer(il + 1) ? nullptr
                                    : layers[il + 1].hc_attn.norm.f32())
            : nullptr;
    const float* next_inject =
        next_norm != nullptr ? InjectWeights(layers[il + 1].hc_attn) : nullptr;
    // Cache updates precede the FFN. Without MTP, only requested head rows
    // and the checkpoint frontier consume the final layer's residual.
    // Keep aligned tiles and at least 96 rows to retain the wide arithmetic.
    const auto needed =
        std::max({96U, n_logits, checkpoint ? n - checkpoint->tokens + 1 : 0U});
    const auto skipped = prefill_phase && !session.mtp_enabled_ &&
                                 il + 1 == c.num_layers && n > needed
                             ? (n - needed) / 128 * 128
                             : 0U;
    const auto finish = [&](std::uint32_t rows) {
      defer_residual_ = WmmaExperts(l, rows);
      // The wide FFN reads the mix only through its F16 rows (router, routed
      // experts) and tiled Q8 rows (shared expert).
      const bool no_f32 = prefill_phase && defer_residual_ &&
                          rows <= options_.max_batch &&
                          l.router.type == GgmlType::kF16 &&
                          l.shexp_gate.type == GgmlType::kQ8_0 &&
                          l.shexp_up.type == GgmlType::kQ8_0;
      // EXL3 experts settle or consume the deferred update before writing
      // block_out.
      Combine(s_.res, l.hc_ffn.norm.f32(), rows, InjectWeights(l.hc_ffn),
              l.ffn_gate_exps.exl3 != nullptr);
      const DeviceTensor* const shared_rotate[] = {&l.shexp_gate, &l.shexp_up};
      if (!HcMix(l.hc_ffn, s_.res, true, s_.mixed, s_.inject, rows, error_msg,
                 false, no_f32, shared_rotate))
        return false;
      // Only the combine below reads this MoE output: EXL3 prefill folds the
      // expert sum into it.
      exl3_moe_defer_ = prefill_phase;
      const bool moe_ok = Moe(l, s_.mixed, s_.block_out, rows, error_msg);
      exl3_moe_defer_ = false;
      if (!moe_ok)
        return false;
      Combine(s_.res, next_norm, rows, next_inject);
      return true;
    };
    if (skipped != 0) {
      struct RestoreScratch {
        const Executor* executor;
        Scratch scratch;
        ~RestoreScratch() { executor->UseScratch(scratch); }
      } restore{this, s_};
      auto tail = RowScratch(s_, skipped);
      tail.inject =
          s_.inject + std::size_t{skipped} * c.hc_count * inject_parts_;
      UseScratch(tail);
      if (!finish(n - skipped))
        return false;
    } else if (!finish(n)) {
      return false;
    }
    normed = next_norm != nullptr;
  }
  // Every layer's FFN combine consumes its deferred update; never leave one
  // across a split forward or to the head.
  ApplyPendingResidual();
  if (end_layer < c.num_layers) {
    return true;
  }
  if (checkpoint) {
    if (checkpoint->hidden_rows != 0 &&
        !Check(hipMemcpyAsync(checkpoint->state->mtp_.target_hidden,
                              s_.res + std::size_t{checkpoint->tokens -
                                                   checkpoint->hidden_rows} *
                                           c.HcDim(),
                              std::size_t{checkpoint->hidden_rows} * c.HcDim() *
                                  sizeof(float),
                              hipMemcpyDeviceToDevice, stream_),
               "checkpoint hidden rows", error_msg))
      return false;
    PrefillPhase head_phase(false);
    if (!HcMix(model_->hc_head(),
               s_.res + std::size_t{checkpoint->tokens - 1} * c.HcDim(), false,
               s_.mixed, nullptr, 1, error_msg) ||
        !Dense(model_->output(), s_.mixed, s_.logits, 1, error_msg) ||
        !Check(hipMemcpyAsync(checkpoint->logits.data(), s_.logits,
                              std::size_t{c.vocab_size} * sizeof(float),
                              hipMemcpyDeviceToHost, stream_),
               "checkpoint logits", error_msg))
      return false;
  }
  // Prefill catches up immediately from shared scratch. Only the short tail
  // needed across calls belongs to this session.
  const auto kept = std::min(
      checkpoint && checkpoint->tokens < n ? n - checkpoint->tokens : n,
      options_.max_speculative);
  if (session.mtp_enabled_ &&
      !Check(hipMemcpyAsync(
                 session.mtp_.target_hidden,
                 s_.res + static_cast<std::size_t>(n - kept) * c.HcDim(),
                 static_cast<std::size_t>(kept) * c.HcDim() * sizeof(float),
                 hipMemcpyDeviceToDevice, stream_),
             "hidden keep", error_msg)) {
    return false;
  }
  if (n_logits > 0) {
    PrefillPhase head_phase(false);
    // Only the requested tail needs the head's normalization.
    const std::size_t skip = static_cast<std::size_t>(n - n_logits);
    const DeviceMixer& head = model_->hc_head();
    if (!HcMix(head, s_.res + skip * c.HcDim(), false, s_.mixed, nullptr,
               n_logits, error_msg) ||
        !Dense(model_->output(), s_.mixed, s_.logits, n_logits, error_msg)) {
      return false;
    }
    if (download_logits &&
        !Check(hipMemcpyAsync(logits_host_, s_.logits,
                              static_cast<std::size_t>(n_logits) *
                                  c.vocab_size * sizeof(float),
                              hipMemcpyDeviceToHost, stream_),
               "logits download", error_msg)) {
      return false;
    }
  }
  return true;
}

bool Executor::CommitPendingSpeculation(Session& session,
                                        std::string* error_msg) const {
  // A pending batch nobody rolled back keeps every token, as the in-place
  // recurrence did before deferral.
  const auto n = std::exchange(session.spec_tokens_, 0U);
  return CommitGdnStates(session, n, n, error_msg);
}

bool Executor::Rollback(Session& session, std::uint32_t keep,
                        std::string* error_msg, float* logits) const {
  const Config& c = config();
  const std::uint32_t n = session.spec_tokens_;
  if (n == 0 || keep == 0 || keep > n) {
    AssignError(error_msg, "rollback outside the pending speculative batch");
    return false;
  }
  if (session.continuation_hooks_ && keep < n) {
    const auto end = session.spec_base_ + keep;
    session.continuation_hooks_->BeforeWrite(
        false, end, session.position_,
        std::min(session.blocks_, end / c.compress_ratio), session.blocks_);
  }
  session.spec_tokens_ = 0;
  // The batch left each recurrent state at its value before the batch:
  // commit the kept prefix, accepted in full or not.
  if (!CommitGdnStates(session, keep, n, error_msg))
    return false;
  if (logits != nullptr &&
      !Check(hipMemcpyAsync(
                 logits_host_,
                 VerificationLogits() +
                     static_cast<std::size_t>(keep - 1) * c.vocab_size,
                 c.vocab_size * sizeof(float), hipMemcpyDeviceToHost, stream_),
             "frontier download", error_msg)) {
    return false;
  }
  // Snapshot and continuation captures copy the states on other streams:
  // settle the commit first.
  if (keep == n && logits == nullptr) {
    return n <= 1 ||
           Check(hipStreamSynchronize(stream_), "rollback commit", error_msg);
  }
  if (keep < n) {
    const std::size_t conv_elems =
        static_cast<std::size_t>(c.ssm_conv_kernel - 1) * c.SsmConvChannels();
    const std::size_t slot = keep - 1;
    for (auto& l : session.linear_) {
      if (l.state == nullptr) {
        continue;
      }
      if (!Check(hipMemcpyAsync(l.conv_state, l.conv_snapshots.rows[slot],
                                conv_elems * sizeof(float),
                                hipMemcpyDeviceToDevice, stream_),
                 "conv rollback", error_msg)) {
        return false;
      }
    }
    if (session.ple_history_ != nullptr) {
      const std::size_t hist =
          static_cast<std::size_t>(c.PleConvHistory()) * c.HcDim();
      if (!Check(hipMemcpyAsync(
                     session.ple_history_, session.ple_snapshots_.rows[slot],
                     hist * sizeof(float), hipMemcpyDeviceToDevice, stream_),
                 "PLE rollback", error_msg)) {
        return false;
      }
      session.ngram_ = session.ngram_snapshots_[slot];
    }
    session.position_ = session.spec_base_ + keep;
    // Pooled block keys past the kept prefix are stale; they are rebuilt
    // from the raw keys when needed.
    session.blocks_ =
        c.compress_ratio == 0
            ? 0
            : std::min(session.blocks_, session.position_ / c.compress_ratio);
  }
  if (!Check(hipStreamSynchronize(stream_), "rollback", error_msg)) {
    return false;
  }
  if (logits != nullptr) {
    std::copy_n(logits_host_, c.vocab_size, logits);
  }
  return true;
}

bool Executor::ReplayOutputHead(std::uint32_t batch_rows,
                                std::span<float> logits,
                                std::string* error_msg) const {
  const auto& c = config();
  const auto rows = logits.size() / c.vocab_size;
  if (batch_rows == 0 || batch_rows > options_.max_batch || rows == 0 ||
      rows > batch_rows || rows > options_.max_logit_rows ||
      logits.size() % c.vocab_size != 0) {
    AssignError(error_msg, "invalid output head replay rows");
    return false;
  }
  DropReadyQ8();
  q8t_src_ = nullptr;
  half_src_ = nullptr;
  PrefillPhase head_phase(false);
  return HcMix(model_->hc_head(), s_.res + (batch_rows - rows) * c.HcDim(),
               false, s_.mixed, nullptr, static_cast<std::uint32_t>(rows),
               error_msg) &&
         Dense(model_->output(), s_.mixed, s_.logits,
               static_cast<std::uint32_t>(rows), error_msg) &&
         Check(hipMemcpyAsync(logits.data(), s_.logits, logits.size_bytes(),
                              hipMemcpyDeviceToHost, stream_),
               "output head replay download", error_msg) &&
         Check(hipStreamSynchronize(stream_), "output head replay", error_msg);
}

bool Executor::ReadVerificationRows(std::uint32_t row, std::span<float> logits,
                                    std::string* error_msg) const {
  const auto vocab = config().vocab_size;
  if (row >= options_.max_speculative || logits.empty() ||
      logits.size() % vocab != 0 ||
      logits.size() / vocab > options_.max_speculative - row) {
    AssignError(error_msg, "invalid verification row");
    return false;
  }
  return Check(hipMemcpyAsync(logits.data(), VerificationLogits() + row * vocab,
                              logits.size_bytes(), hipMemcpyDeviceToHost,
                              stream_),
               "constrained verification row download", error_msg) &&
         Check(hipStreamSynchronize(stream_), "constrained verification",
               error_msg);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
