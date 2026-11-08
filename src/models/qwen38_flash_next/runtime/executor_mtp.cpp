// MTP: draft rows and logits, the MTP K/V projection, the MTP head, and
// greedy/sampled draft candidates and the MTP forward pass.

#include "src/models/qwen38_flash_next/runtime/executor_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

bool Executor::AppendDraftRows(Session& session, std::uint32_t first,
                               std::span<const std::int32_t> ids,
                               std::string* error_msg) const {
  if (ids.empty())
    return true;
  if (session.owner_ != this || session.mtp_.draft_rows == nullptr ||
      first > kMtpDraftExtraRows || ids.size() > kMtpDraftExtraRows - first) {
    AssignError(error_msg, "invalid draft table append");
    return false;
  }
  const auto& output = model_->output();
  const std::size_t row_bytes = model_->output_row_bytes();
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] < 0 || static_cast<std::uint32_t>(ids[i]) >= output.rows) {
      AssignError(error_msg, "draft table token out of range");
      return false;
    }
    // The ID goes by value, so no host buffer has to outlive the call.
    if ((output.exl3 == nullptr &&
         !Check(
             hipMemcpyAsync(session.mtp_.draft_rows + (first + i) * row_bytes,
                            static_cast<const std::uint8_t*>(output.data) +
                                static_cast<std::size_t>(ids[i]) * row_bytes,
                            row_bytes, hipMemcpyDeviceToDevice, stream_),
             "draft table row", error_msg)) ||
        !Check(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(
                                     session.mtp_.draft_ids + first + i),
                                 ids[i], 1, stream_),
               "draft table id", error_msg))
      return false;
  }
  return true;
}

bool Executor::ClearDraftRows(Session& session, std::uint32_t first,
                              std::string* error_msg) const {
  if (session.owner_ != this || session.mtp_.draft_ids == nullptr ||
      first > kMtpDraftExtraRows) {
    AssignError(error_msg, "invalid draft table clear");
    return false;
  }
  // Stale rows stay; an ID of -1 masks the slot.
  return first == kMtpDraftExtraRows ||
         Check(
             hipMemsetAsync(session.mtp_.draft_ids + first, 0xFF,
                            (kMtpDraftExtraRows - first) * sizeof(std::int32_t),
                            stream_),
             "draft table clear", error_msg);
}

DeviceTensor Executor::DraftTable(const Session& session) const noexcept {
  DeviceTensor table = model_->output();
  table.data = session.mtp_.draft_rows;
  table.rows = kMtpDraftExtraRows;
  return table;
}

bool Executor::DraftLogits(const Session& session, const float* x,
                           float* logits, std::string* error_msg) const {
  const auto& prefix = model_->draft_output();
  if (model_->output().exl3 != nullptr) {
    Exl3DraftHead(model_->output(), x, logits, prefix.rows,
                  session.mtp_.draft_ids, kMtpDraftExtraRows, exl3_, stream_);
    if (diagnostics_active_) {
      cycle_diagnostics_.head_segments += (prefix.rows + 127) / 128;
      if (!DiagnosticCount(exl3_.head_segments,
                           &cycle_diagnostics_.head_segments, error_msg))
        return false;
    }
    return true;
  }
  if (!Dense(prefix, x, logits, 1, error_msg) ||
      !Dense(DraftTable(session), x, logits + prefix.rows, 1, error_msg))
    return false;
  MaskDraftLogits(logits + prefix.rows, session.mtp_.draft_ids,
                  kMtpDraftExtraRows, stream_);
  return true;
}

bool Executor::MtpKvProjection(const DeviceLayer& l, const float* x,
                               std::uint32_t rows,
                               std::string* error_msg) const {
  if (l.attn_qkv.empty() && model_->output().exl3 != nullptr) {
    // Omit Q while retaining the full group's W/TN launch geometry and
    // therefore exactly the K/V reduction used by full MTP execution.
    const auto basis =
        std::size_t{l.attn_q.rows} + l.attn_k.rows + l.attn_v.rows;
    return Exl3Group({&l.attn_k, &l.attn_v}, {s_.k, s_.v}, x, rows, error_msg,
                     basis);
  }
  if (l.attn_qkv.empty())
    return Exl3Group({&l.attn_k, &l.attn_v}, {s_.k, s_.v}, x, rows, error_msg);
  auto kv = l.attn_qkv;
  const TensorRef layout{.type = kv.type, .cols = kv.cols};
  kv.data = static_cast<std::uint8_t*>(kv.data) +
            std::size_t{2} * config().AttentionQDim() * layout.RowBytes();
  kv.rows = 2 * config().AttentionKvDim();
  return DenseBatch(kv, x, s_.qg, rows, error_msg);
}

bool Executor::StoreMtpKv(const DeviceLayer& l, const Session& session,
                          std::uint32_t rows, const YarnRope& yarn,
                          std::string* error_msg) const {
  const auto& c = config();
  if (l.attn_qkv.empty()) {
    PrepareAttentionKv(
        s_.k, s_.v, l.attn_k_norm.f32(), session.mtp_.k_cache,
        session.mtp_.v_cache, rows, c.num_kv_heads, c.head_dim, c.rotary_dim,
        &session.control_->mtp_position, c.rope_theta, c.rms_eps, stream_,
        session.vision_input_.rope(), yarn, session.mtp_.kv_blocks);
    return true;
  }
  if (PrepareAttention(
          s_.qg, 2 * c.AttentionKvDim(), nullptr, l.attn_k_norm.f32(), nullptr,
          nullptr, session.mtp_.k_cache, session.mtp_.v_cache, rows, 0,
          c.num_kv_heads, c.head_dim, c.rotary_dim,
          &session.control_->mtp_position, c.rope_theta, c.rms_eps, stream_,
          session.vision_input_.rope(), false, yarn, session.mtp_.kv_blocks))
    return true;
  AssignError(error_msg, "MTP cache projection failed");
  return false;
}

bool Executor::MtpHead(const DeviceMixer& head, const float* res, bool token,
                       bool candidates, const Session* draft,
                       std::string* error_msg) const {
  const std::uint32_t prefix = model_->draft_output().rows;
  const std::uint32_t rows =
      draft != nullptr ? prefix + kMtpDraftExtraRows : model_->output().rows;
  if (diagnostics_active_ && draft == nullptr &&
      model_->output().exl3 != nullptr)
    cycle_diagnostics_.head_segments += (rows + 127) / 128;
  if (!HcMix(head, res, false, s_.mixed, nullptr, 1, error_msg) ||
      !(draft != nullptr
            ? DraftLogits(*draft, s_.mixed, s_.logits, error_msg)
            : Dense(model_->output(), s_.mixed, s_.logits, 1, error_msg))) {
    return false;
  }
  const auto count = std::min<std::uint32_t>(rows, kMtpCandidates);
  if (candidates) {
    MtpTopCandidates(s_.logits, s_.mtp_ids, s_.mtp_scratch_ids, s_.mtp_scores,
                     rows, stream_);
    if (draft != nullptr)
      RemapDraftIds(s_.mtp_ids, count, prefix, draft->mtp_.draft_ids,
                    kMtpDraftExtraRows, stream_);
  }
  if (token) {
    Argmax(s_.logits, s_.mtp_argmax, s_.mtp_token, 1, rows, stream_);
    if (draft != nullptr)
      RemapDraftIds(reinterpret_cast<std::uint32_t*>(s_.mtp_token), 1, prefix,
                    draft->mtp_.draft_ids, kMtpDraftExtraRows, stream_);
    if (!Check(
            hipMemcpyAsync(mtp_token_host_, s_.mtp_token, sizeof(std::int32_t),
                           hipMemcpyDeviceToHost, stream_),
            "draft token download", error_msg)) {
      return false;
    }
  }
  if (candidates) {
    static_assert(offsetof(MtpCandidateLogits, logits) ==
                  kMtpCandidates * sizeof(std::uint32_t));
    const auto bytes =
        offsetof(MtpCandidateLogits, logits) + count * sizeof(float);
    if (!Check(hipMemcpyAsync(mtp_candidates_host_, s_.mtp_ids, bytes,
                              hipMemcpyDeviceToHost, stream_),
               "draft candidates download", error_msg)) {
      return false;
    }
  }
  return true;
}

bool Executor::GreedyMtpPredictions(std::span<ArgmaxCandidate> predictions,
                                    const sampling::SamplerState& sampler,
                                    std::span<const std::int32_t> drafts,
                                    std::string* error_msg) const {
  const auto rows = predictions.size();
  if (rows == 0 || rows > options_.max_logit_rows || rows > 7 ||
      drafts.size() != rows || s_.mtp_ids == nullptr) {
    AssignError(error_msg, "invalid greedy MTP verification request");
    return false;
  }
  // Proposal selection has finished. Its two ID buffers and argmax scratch
  // can be reused until the next draft head overwrites them.
  const auto vocab = config().vocab_size;
  // Keep the host upload alive until the final stream synchronization. This
  // workspace is shared, never part of a session or its persistent snapshot.
  std::vector<sampling::TokenPenalty> penalties;
  if (sampler.config().penalties_enabled()) {
    penalties.reserve(rows * (sampler.penalties().size() + rows));
    // Drafts may violate the target grammar. Only their conditional penalty
    // history is needed here; FinishDecode checks each winner's grammar.
    auto tentative = sampler.WithoutConstraint();
    GreedyPenaltyRows batch{};
    for (std::size_t row = 0; row < rows; ++row) {
      const auto counts = tentative.penalties();
      penalties.insert(penalties.end(), counts.begin(), counts.end());
      batch.offsets[row + 1] = static_cast<std::uint32_t>(penalties.size());
      if (row + 1 < rows)
        tentative.Accept(static_cast<sampling::TokenId>(drafts[row]));
    }
    if (penalties.size() > verification_penalty_capacity_) {
      const auto capacity =
          std::max(penalties.size(), 2 * verification_penalty_capacity_);
      sampling::TokenPenalty* allocated = nullptr;
      if (!Check(hip::AllocateDevice(&allocated, capacity * sizeof(*allocated)),
                 "verification penalty allocation", error_msg))
        return false;
      (void)hipFree(verification_penalties_);
      verification_penalties_ = allocated;
      verification_penalty_capacity_ = capacity;
    }
    batch.penalties = verification_penalties_;
    if (!penalties.empty() &&
        !Check(hipMemcpyAsync(verification_penalties_, penalties.data(),
                              penalties.size() * sizeof(penalties.front()),
                              hipMemcpyHostToDevice, stream_),
               "verification penalty upload", error_msg))
      return false;
    const auto& config = sampler.config();
    PenalizedArgmax(
        VerificationLogits(), batch, config.repeat_penalty,
        config.frequency_penalty, config.presence_penalty,
        reinterpret_cast<PenaltyArgmaxCandidate*>(s_.mtp_scratch_ids),
        s_.mtp_argmax, rows, vocab, stream_);
  } else {
    gufo::hip::LaunchBatchedGPUArgmax(
        VerificationLogits(), s_.mtp_ids, rows, vocab,
        {reinterpret_cast<float*>(s_.mtp_scratch_ids),
         MtpCandidateWorkspaceSize(vocab)},
        stream_);
    GatherArgmaxCandidates(VerificationLogits(), s_.mtp_ids, s_.mtp_argmax,
                           static_cast<std::uint32_t>(rows), vocab, stream_);
  }
  const bool copied = Check(
      hipMemcpyAsync(predictions.data(), s_.mtp_argmax,
                     predictions.size_bytes(), hipMemcpyDeviceToHost, stream_),
      "greedy MTP predictions download", error_msg);
  // Drain even after a failed download before releasing the upload source.
  const bool completed = Check(hipStreamSynchronize(stream_),
                               "greedy MTP verification", error_msg);
  return copied && completed;
}

bool Executor::CanCompactMtpVerification(
    const sampling::SamplerState& sampler) {
  const auto& config = sampler.config();
  return config.uses_random_sampling() && config.top_k > 0 &&
         std::size_t(config.top_k) <= kMtpCandidates &&
         config.min_keep <= kMtpCandidates &&
         sampler.penalties().size() + 8 <= kCompactPenaltyLimit &&
         !sampler.NeedsConstraintMask();
}

bool Executor::SampledMtpCandidates(
    std::span<std::span<const sampling::LogitCandidate>> output,
    const sampling::SamplerState& sampler, std::span<const std::int32_t> drafts,
    std::string* error_msg) const {
  const auto rows = output.size();
  if (rows == 0 || rows > 7 || rows >= options_.max_speculative ||
      rows != drafts.size() || verification_ids_ == nullptr ||
      !CanCompactMtpVerification(sampler)) {
    AssignError(error_msg, "invalid compact MTP verification request");
    return false;
  }
  auto& penalties = verification_penalty_host_;
  penalties.clear();
  GreedyPenaltyRows batch{};
  auto tentative = sampler.WithoutConstraint();
  for (std::size_t row = 0; row < rows; ++row) {
    const auto counts = tentative.penalties();
    penalties.insert(penalties.end(), counts.begin(), counts.end());
    batch.offsets[row + 1] = penalties.size();
    if (row + 1 < rows)
      tentative.Accept(static_cast<sampling::TokenId>(drafts[row]));
  }
  batch.penalties = verification_penalties_;
  if (!penalties.empty() &&
      !Check(hipMemcpyAsync(verification_penalties_, penalties.data(),
                            penalties.size() * sizeof(penalties[0]),
                            hipMemcpyHostToDevice, stream_),
             "compact verification penalties", error_msg))
    return false;
  VerificationCandidates(VerificationLogits(), batch, verification_ids_,
                         verification_scratch_ids_, verification_candidates_,
                         rows, config().vocab_size, stream_);
  const auto count = rows * kMtpCandidates + penalties.size();
  const bool copied = Check(
      hipMemcpyAsync(verification_candidates_host_, verification_candidates_,
                     count * sizeof(sampling::LogitCandidate),
                     hipMemcpyDeviceToHost, stream_),
      "compact verification candidates", error_msg);
  const bool completed =
      Check(hipStreamSynchronize(stream_), "compact verification", error_msg);
  if (!copied || !completed)
    return false;
  for (std::size_t row = 0; row < rows; ++row)
    output[row] = {
        verification_candidates_host_ + row * kMtpCandidates +
            batch.offsets[row],
        kMtpCandidates + batch.offsets[row + 1] - batch.offsets[row]};
  return true;
}

bool Executor::MtpForward(Session& session,
                          std::span<const std::int32_t> tokens,
                          std::int32_t hidden_row, MtpOutput output,
                          std::string* error_msg,
                          const float* hidden_source) const {
  DiagnosticPhase diagnostic(*this, hidden_row >= 0 ? DiagnosticStage::kCatchUp
                                                    : DiagnosticStage::kDraft);
  DropReadyQ8();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  if (session.owner_ != this ||
      std::any_of(tokens.begin(), tokens.end(), [&](auto t) {
        return t < 0 || static_cast<std::uint32_t>(t) >= config().vocab_size;
      })) {
    AssignError(error_msg, "invalid MTP session or token");
    return false;
  }
  if (!session.CheckCancellation(error_msg))
    return false;
  if (!session.mtp_enabled_) {
    AssignError(error_msg, "no MTP block loaded");
    return false;
  }

  if (n == 0 || n > options_.max_batch || (hidden_row < 0 && n != 1) ||
      (hidden_row >= 0 &&
       static_cast<std::uint32_t>(hidden_row) + n >
           (hidden_source ? options_.max_batch : options_.max_speculative))) {
    AssignError(error_msg, "MTP batch outside the kept hidden rows");
    return false;
  }
  if (output.trace != nullptr &&
      !output.trace->Valid(config().hidden_size, config().HcDim())) {
    AssignError(error_msg, "invalid MTP trace destinations");
    return false;
  }
  if ((hidden_row < 0 && !session.mtp_.residual_valid) ||
      (output.kv_only &&
       (hidden_row < 0 || output.token != nullptr ||
        output.candidates != nullptr || output.trace != nullptr))) {
    AssignError(error_msg, "MTP residual requires a full known-hidden forward");
    return false;
  }
  const std::uint32_t pos = session.mtp_.position;
  if (pos + n > session.max_context_) {
    AssignError(error_msg, "MTP context is full");
    return false;
  }
  session.EnsureKvRows(pos + n);
  if (session.continuation_hooks_)
    session.continuation_hooks_->BeforeWrite(
        true, pos, pos + n, session.mtp_.blocks,
        pos + n > config().indexer_top_k ? (pos + n) / config().compress_ratio
                                         : session.mtp_.blocks);
  session.PreserveSnapshots(session.position_, session.mtp_.position);
  ++session.mutation_epoch_;
  session.mtp_.residual_valid = false;
  std::copy(tokens.begin(), tokens.end(), tokens_host_);
  control_host_->position = session.position_;
  control_host_->blocks = session.blocks_;
  control_host_->mtp_position = pos;
  control_host_->hidden_row = hidden_row;
  control_host_->mtp_blocks = session.mtp_.blocks;
  const auto& c = config();
  const bool sparse = pos + n > c.indexer_top_k;
  const auto complete = (pos + n) / c.compress_ratio;
  const auto pool = sparse ? complete - session.mtp_.blocks : 0;
  const auto graph_pool = n / c.compress_ratio + 1;
  const bool graph = !diagnostics_active_ && output.trace == nullptr &&
                     hidden_source == nullptr && n <= kVecBatch &&
                     pool <= graph_pool &&
                     pos + 1 >= session.VisionLayout().PrefixLength();
  // Captured graphs bake in the head; key on the one this draft reads.
  const bool subset = UseDraftSubset(session, output.draft_subset) &&
                      (output.token != nullptr || output.candidates != nullptr);
  const std::uint64_t key =
      static_cast<std::uint64_t>(n) |
      (static_cast<std::uint64_t>(hidden_row < 0) << 32) |
      (std::uint64_t{1} << 40) |
      (std::uint64_t{output.token != nullptr} << 41) |
      (std::uint64_t{output.candidates != nullptr} << 43) |
      (std::uint64_t{sparse} << 44) | (std::uint64_t{output.kv_only} << 45) |
      (std::uint64_t{subset} << 46);
  const auto body = [&] {
    return MtpBody(session, n, pos, output.token != nullptr,
                   output.candidates != nullptr, error_msg,
                   graph ? graph_pool : pool, hidden_source, output.trace,
                   output.kv_only, subset);
  };
  if (!Run(session, key, graph, body, error_msg)) {
    return false;
  }
  if (output.token != nullptr) {
    *output.token = *mtp_token_host_;
  }
  if (output.candidates != nullptr) {
    *output.candidates = *mtp_candidates_host_;
  }
  session.mtp_.position = pos + n;
  if (diagnostics_active_ && hidden_row >= 0)
    cycle_diagnostics_.catch_up_rows += n;
  session.mtp_.residual_valid = !output.kv_only;
  if (sparse)
    session.mtp_.blocks = complete;
  return true;
}

bool Executor::MtpBody(Session& session, std::uint32_t n, std::uint32_t pos,
                       bool token, bool candidates, std::string* error_msg,
                       std::uint32_t pool_grid, const float* hidden_source,
                       MtpTrace* trace, bool kv_only, bool draft_subset) const {
  const Config& c = config();
  // s_.mtp_res aliases s_.res: a failed trunk pass must not leave its
  // deferred attention update for this pass's first combine.
  exl3_res_pending_ = {};
  const DeviceLayer& l = model_->mtp();
  const std::uint32_t hc_dim = c.HcDim();
  const std::size_t final_row = static_cast<std::size_t>(n - 1) * hc_dim;
  const auto copy_trace = [&](const float* source,
                              std::span<float> destination) {
    return destination.empty() ||
           Check(hipMemcpyAsync(destination.data(), source,
                                destination.size_bytes(), hipMemcpyDeviceToHost,
                                stream_),
                 "MTP trace download", error_msg);
  };
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
              s_.tokens, s_.mtp_embd, n, c.hidden_size, 1, stream_);
  // MTP embeds shifted token IDs. Visual information arrives in the trunk
  // hidden stream; image embeddings belong only to the target input.
  RmsNormRows(s_.mtp_embd, l.nextn_enorm.f32(), s_.mtp_embd, n, c.hidden_size,
              1, c.rms_eps, stream_);
  // The hidden input: kept trunk rows from `hidden_row`, or the block's
  // own carried residual.
  const float* hidden = s_.mtp_h;
  if (hidden_source && control_host_->hidden_row >= 0) {
    // Eager catch-up can normalize the trunk rows directly. Graphs use the
    // device-selected source below because their hidden-row index can change
    // on replay. Neither path modifies the trunk residual.
    hidden = hidden_source +
             static_cast<std::size_t>(control_host_->hidden_row) * hc_dim;
  } else {
    MtpHidden(hidden_source ? hidden_source : session.mtp_.target_hidden,
              session.mtp_.h, &session.control_->hidden_row, s_.mtp_h, n,
              hc_dim, stream_);
  }
  // Unlike the HC mixers, this normalization spans the complete HC * H row.
  RmsNormRows(hidden, l.nextn_hnorm.f32(), s_.mtp_h, n, hc_dim, 1, c.rms_eps,
              stream_);
  if (trace && !copy_trace(s_.mtp_h + final_row, trace->normalized_hidden))
    return false;
  // These projections determine persistent draft KV. Use the same row-wise
  // activation quantization and reduction for prompt chunks and decode;
  // changing between F16 GEMM and Q8 GEMV changes sampled replay after reuse.
  if (!DenseBatch(l.nextn_fc_embedding, s_.mtp_embd, s_.mtp_eproj, n,
                  error_msg) ||
      !DenseBatch(l.nextn_fc_hidden, s_.mtp_h, s_.mtp_res, n * c.hc_count,
                  error_msg)) {
    return false;
  }
  MtpAddEmbedding(s_.mtp_eproj, s_.mtp_res, n, c.hidden_size, c.hc_count,
                  stream_);
  if (trace && !copy_trace(s_.mtp_res + final_row, trace->fused))
    return false;
  Session::AttentionState attn;
  attn.rope = session.vision_input_.rope();
  attn.k_cache = session.mtp_.k_cache;
  attn.v_cache = session.mtp_.v_cache;
  attn.index_k = session.mtp_.index_k;
  attn.block_k = session.mtp_.block_k;
  attn.kv_blocks = session.mtp_.kv_blocks;
  // The draft block's attention runs at its own position.
  // Catch-up exports KV for every row but only carries its final residual.
  // Preserve that row's original query tile; earlier attention results
  // never contribute to the carried state.
  const bool last_only =
      !token && !candidates && n > 32 && control_host_->hidden_row >= 0;
  // Keep the final 128-column tile (and its predecessor for short tails).
  // At least 96 rows retain the wide mixer/projection arithmetic. Attention
  // still writes every KV/indexer row before the scratch view is narrowed.
  // Short catch-up also carries only its final row. Keep two rows so the
  // router and projections retain their multi-row arithmetic; the single
  // row route has a different floating-point reduction.
  const auto skipped =
      control_host_->hidden_row >= 0 && n > 2 && n <= kVecBatch
          ? n - 2
          : (last_only && n >= 224 ? (n - 96) / 128 * 128 : 0U);
  const bool cache_only =
      kv_only && (!l.attn_qkv.empty() ||
                  (l.attn_k.exl3 != nullptr && l.attn_v.exl3 != nullptr));
  if (!HcMixBatch(l.hc_attn, s_.mtp_res, false, s_.mixed,
                  cache_only ? nullptr : s_.inject, n, error_msg)) {
    return false;
  }
  if (cache_only) {
    if (diagnostics_active_)
      cycle_diagnostics_.cache_only_rows += n;
    // Known trunk rows replace the predictor residual on the next forward.
    // Only K/V and pooled indexer keys survive this catch-up.
    if (!MtpKvProjection(l, s_.mixed, n, error_msg) ||
        !StoreMtpKv(l, session, n, yarn_, error_msg)) {
      return false;
    }
    const auto capacity =
        IndexerCapacity(c, options_.max_batch, session.max_context_);
    if (!DenseBatch(l.indexer_k, s_.mixed, s_.ik, n, error_msg))
      return false;
    StoreRows(s_.ik, attn.index_k, n, c.indexer_head_dim,
              &session.control_->mtp_position, capacity, stream_);
    if (pos + n > c.indexer_top_k) {
      PoolIndexerBlocks(attn.index_k, l.indexer_k_norm.f32(), attn.block_k,
                        &session.control_->mtp_blocks,
                        &session.control_->mtp_position, n, pool_grid,
                        c.compress_ratio, c.indexer_head_dim, c.rotary_dim,
                        c.rope_theta, c.rms_eps, capacity, stream_, attn.rope,
                        yarn_, attn.kv_blocks);
    }
    return true;
  }
  if (!l.attn_qkv.empty()) {
    if (!DenseBatch(l.attn_qkv, s_.mixed, s_.qg, n, error_msg))
      return false;
  } else if (l.attn_q.exl3 != nullptr) {
    if (!Exl3Group({&l.attn_q, &l.attn_k, &l.attn_v}, {s_.qg, s_.k, s_.v},
                   s_.mixed, n, error_msg))
      return false;
  } else if (!DenseBatch(l.attn_q, s_.mixed, s_.qg, n, error_msg) ||
             !DenseBatch(l.attn_k, s_.mixed, s_.k, n, error_msg) ||
             !DenseBatch(l.attn_v, s_.mixed, s_.v, n, error_msg)) {
    return false;
  }
  if (!Attention(l, attn, s_.mixed, s_.block_out, n,
                 &session.control_->mtp_position, &session.control_->mtp_blocks,
                 pos, pool_grid, session.max_context_,
                 pos + n > c.indexer_top_k, error_msg, last_only, true,
                 skipped == 0)) {
    return false;
  }
  struct RestoreScratch {
    const Executor* executor;
    Scratch scratch;
    bool changed;
    ~RestoreScratch() {
      if (changed)
        executor->UseScratch(scratch);
    }
  } restore{this, s_, skipped != 0};
  const auto tail_rows = n - skipped;
  const auto tail_last = static_cast<std::size_t>(tail_rows - 1) * hc_dim;
  if (skipped != 0) {
    auto tail = RowScratch(s_, skipped);
    // Keep the producer's inject stride: it may use vectorized partials or
    // a single quantized projection rather than RowScratch's scalar layout.
    tail.inject = s_.inject + static_cast<std::size_t>(skipped) * c.hc_count *
                                  inject_parts_;
    UseScratch(tail);
    if (!Dense(l.attn_out, s_.ctx, s_.block_out, tail_rows, error_msg))
      return false;
  }
  Combine(s_.mtp_res, l.hc_ffn.norm.f32(), tail_rows);
  if (trace && !copy_trace(s_.mtp_res + tail_last, trace->attention))
    return false;
  if (!session.CheckCancellation(error_msg))
    return false;
  const DeviceTensor* const shared_rotate[] = {&l.shexp_gate, &l.shexp_up};
  if (!HcMix(l.hc_ffn, s_.mtp_res, true, s_.mixed, s_.inject, tail_rows,
             error_msg, false, false, shared_rotate) ||
      (trace && !copy_trace(s_.mixed + static_cast<std::size_t>(tail_rows - 1) *
                                           c.hidden_size,
                            trace->ffn_input))) {
    return false;
  }
  if (skipped != 0 && tail_rows == 2)
    exl3_rotated_ = {};  // row 0 is rewritten below; its slab row is stale
  if (skipped != 0 && tail_rows == 2 &&
      !Check(hipMemcpyAsync(s_.mixed, s_.mixed + c.hidden_size,
                            c.hidden_size * sizeof(float),
                            hipMemcpyDeviceToDevice, stream_),
             "MTP tail padding", error_msg)) {
    return false;
  }
  // The discarded padding row shares the final row's routed experts, so
  // preserving multi-row arithmetic does not reread another expert set.
  if (!Moe(l, s_.mixed, s_.block_out, tail_rows, error_msg, last_only)) {
    return false;
  }
  Combine(s_.mtp_res, nullptr, tail_rows);
  if (trace &&
      !copy_trace(s_.block_out +
                      static_cast<std::size_t>(tail_rows - 1) * c.hidden_size,
                  trace->ffn_output))
    return false;
  if (trace && !copy_trace(s_.mtp_res + tail_last, trace->hidden))
    return false;
  if (trace && !trace->head.empty()) {
    if (!HcMix(l.nextn_head, s_.mtp_res + tail_last, false, s_.mixed, nullptr,
               1, error_msg) ||
        !copy_trace(s_.mixed, trace->head))
      return false;
  }
  const float* last = s_.mtp_res + tail_last;
  if (!Check(hipMemcpyAsync(session.mtp_.h, last, hc_dim * sizeof(float),
                            hipMemcpyDeviceToDevice, stream_),
             "MTP hidden carry", error_msg)) {
    return false;
  }
  return (!token && !candidates) ||
         MtpHead(l.nextn_head, last, token, candidates,
                 draft_subset ? &session : nullptr, error_msg);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
