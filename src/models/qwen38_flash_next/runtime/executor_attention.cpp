// Full attention layers: FP16 K/V and KVarN.

#include "src/models/qwen38_flash_next/runtime/executor_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

bool Executor::Attention(const DeviceLayer& l, Session::AttentionState& s,
                         const float* x, float* out, std::uint32_t n_tokens,
                         const std::uint32_t* pos,
                         const std::uint32_t* first_block,
                         std::uint32_t start_pos, std::uint32_t pool_grid,
                         std::uint32_t max_context, bool sparse,
                         std::string* error_msg, bool last_only,
                         bool projections_ready, bool project_output,
                         std::uint32_t checkpoint_tokens) const {
  // Set below for this call's KvarnAttention only.
  attn_q_rotated_ = false;
  attn_gate_stride_ = 0;
  const Config& c = config();
  // The generic fallback consumes every query's selector mask. Only prune
  // masks when WMMA can honor the same last-query restriction; otherwise
  // uninitialized masks can produce NaNs and invalid downstream routing IDs.
  const auto wmma_limit = WmmaCausalAttentionContextLimit(
      c.num_heads, c.num_kv_heads, c.head_dim, c.compress_ratio);
  const auto live_context = std::uint64_t{start_pos} + n_tokens;
  last_only = last_only && MatrixRows(n_tokens) && wmma_limit != 0 &&
              (s.kvarn_pass == nullptr || s.kvarn_pass->prefill) &&
              (!sparse || (live_context <= wmma_limit &&
                           live_context <= std::uint64_t{mask_words_} *
                                               c.compress_ratio * 32));

  const std::uint32_t kv_row = c.AttentionKvDim();
  const std::uint32_t index_capacity =
      IndexerCapacity(c, options_.max_batch, max_context);
  bool prepared = false;
  bool query_rows = false;
  // The WMMA prefill kernel reads its output gate from the packed [q | gate]
  // projection rows: AttentionQueryRows skips the unpacked copy.
  bool packed_gate = false;
  // KVarN layers write FP16 rows and rotate them into the KVarN frame: a
  // pass into the sink and tail ring, a prompt chunk into the staging rows,
  // whose row 0 holds staging_first (plan/exl3/archive/F1-KVARN-DESIGN.md 3).
  const KvarnPass* kvarn = s.kvarn_pass;
  __half* write_k = s.k_cache;
  __half* write_v = s.v_cache;
  const std::uint32_t* write_rows = s.kv_blocks;
  const std::size_t kvarn_row_bytes = std::size_t{kv_row} * sizeof(__half);
  if (kvarn != nullptr && kvarn->prefill) {
    // The rows before the chunk that it reads in FP16: the sink rows of a
    // chunk starting inside the sink, or the ring rows of its first group.
    const std::uint32_t lead = start_pos - kvarn->staging_first;
    const std::size_t from =
        kvarn->staging_first == 0 ? 0 : hip::KvarnFp16Row(kvarn->staging_first);
    if (lead != 0 &&
        (!Check(hipMemcpyAsync(kvarn_staging_k_, s.kvarn_k + from * kv_row,
                               lead * kvarn_row_bytes, hipMemcpyDeviceToDevice,
                               stream_),
                "KVarN staging lead", error_msg) ||
         !Check(hipMemcpyAsync(kvarn_staging_v_, s.kvarn_v + from * kv_row,
                               lead * kvarn_row_bytes, hipMemcpyDeviceToDevice,
                               stream_),
                "KVarN staging lead", error_msg)))
      return false;
    // Writers address rows by absolute position.
    write_k = kvarn_staging_k_ - std::size_t{kvarn->staging_first} * kv_row;
    write_v = kvarn_staging_v_ - std::size_t{kvarn->staging_first} * kv_row;
    write_rows = nullptr;
  } else if (kvarn != nullptr) {
    write_k = s.kvarn_k;
    write_v = s.kvarn_v;
    write_rows = KvarnFp16Rows();
  }
  const bool shared_indexer = prefill_phase && sparse && s.index_k != nullptr &&
                              l.indexer_q.exl3 != nullptr &&
                              l.indexer_q.exl3 == l.indexer_k.exl3;
  // The indexer keys and block selection read only x, the indexer caches and
  // their own scratch (s_.ik, s_.iq, s_.scores, s_.mask).
  const std::uint32_t* mask = nullptr;
  const auto indexer = [&]() -> bool {
    if (s.index_k != nullptr) {
      if (!shared_indexer &&
          !(projections_ready
                ? DenseBatch(l.indexer_k, x, s_.ik, n_tokens, error_msg)
                : Dense(l.indexer_k, x, s_.ik, n_tokens, error_msg))) {
        return false;
      }
      StoreRows(s_.ik, s.index_k, n_tokens, c.indexer_head_dim, pos,
                index_capacity, stream_);
    }
    if (sparse) {
      if (!shared_indexer &&
          !Dense(l.indexer_q, x, s_.iq, n_tokens, error_msg)) {
        return false;
      }
      RmsNormRows(s_.iq, l.indexer_q_norm.f32(), s_.iq,
                  n_tokens * c.indexer_heads, c.indexer_head_dim, 1, c.rms_eps,
                  stream_);
      Rope(s_.iq, n_tokens, c.indexer_heads, c.indexer_head_dim, c.rotary_dim,
           pos, c.rope_theta, stream_, s.rope, yarn_);
      PoolIndexerBlocks(s.index_k, l.indexer_k_norm.f32(), s.block_k,
                        first_block, pos, n_tokens, pool_grid, c.compress_ratio,
                        c.indexer_head_dim, c.rotary_dim, c.rope_theta,
                        c.rms_eps, index_capacity, stream_, s.rope, yarn_,
                        s.kv_blocks);
      // Align score rows to full cache lines. The selector still considers
      // only complete causal blocks, so padding cannot change the ranking.
      // Wide prefill is never captured: compact score rows to the populated
      // context. Graphs retain the maximum stride for replay at later
      // positions.
      const auto score_context =
          n_tokens > kVecBatch ? start_pos + n_tokens : max_context;
      const std::uint32_t blocks =
          (score_context + c.compress_ratio - 1) / c.compress_ratio;
      const std::uint32_t max_blocks = (blocks + 31) / 32 * 32;
      // Query rows are independent; rounding down to a power of two wasted
      // up to half the bounded scratch and doubled launches above 160K.
      const auto select_chunk = static_cast<std::uint32_t>(
          std::min(std::size_t{512}, select_score_floats_ / max_blocks));
      // Catch-up consumes only the final attention tile. Cover its masks even
      // when the dense/sparse boundary shifts the four-query sparse groups.
      const auto first_query = last_only && n_tokens > 16 ? n_tokens - 16 : 0U;
      for (std::uint32_t t0 = first_query; t0 < n_tokens; t0 += select_chunk) {
        const std::uint32_t n = std::min(select_chunk, n_tokens - t0);
        // Batches wider than kVecBatch are never captured, so the host
        // position bounds the scored range (a replayed graph must cover
        // max_blocks).
        const std::uint32_t live =
            n_tokens > kVecBatch ? (start_pos + t0 + n) / c.compress_ratio : 0;
        SelectBlocks(s_.iq + static_cast<std::size_t>(t0) * c.indexer_heads *
                                 c.indexer_head_dim,
                     s.block_k,
                     s_.mask + static_cast<std::size_t>(t0) * mask_words_,
                     s_.scores, n, pos, t0, c.indexer_heads, c.indexer_head_dim,
                     c.compress_ratio, c.indexer_top_k / c.compress_ratio,
                     mask_words_, max_blocks, stream_, live, s.kv_blocks);
      }
      mask = s_.mask;
    }
    return true;
  };
  // EXL3 decode/verify: a side branch runs them beside the q/k/v GEMV and
  // the query/cache preparation (exl3_side_ scratch for its projections);
  // attention joins it.
  struct JoinGuard {
    const Executor* executor;
    ~JoinGuard() { (void)executor->JoinSide(nullptr); }
  } join_guard{this};
  // Only fast-path projections: the side scratch has no reference-path
  // buffers.
  const auto exl3_view = [&](const DeviceTensor& w) {
    return w.exl3 != nullptr && w.cols == c.hidden_size &&
           gufo::hip::Exl3FastSupported(*w.exl3) &&
           w.exl3->n / 128 <= exl3_side_.projection.fast_counter_count;
  };
  const bool side_indexer =
      !projections_ready && l.attn_qkv.empty() && l.attn_q.exl3 != nullptr &&
      (s.index_k != nullptr || sparse) && n_tokens <= kVecBatch &&
      (s.index_k == nullptr || exl3_view(l.indexer_k)) &&
      (!sparse || exl3_view(l.indexer_q)) && SideBranchAvailable();
  if (side_indexer && !ForkSide(indexer, error_msg))
    return false;
  if (!l.attn_qkv.empty()) {
    const bool fused_projection =
        !projections_ready && WidePrefill(n_tokens) &&
        n_tokens <= options_.max_batch && DenseF16Route(l.attn_qkv, n_tokens) &&
        l.attn_qkv.rows == 13312 && l.attn_qkv.cols == 2560 &&
        c.num_heads == 24 && c.num_kv_heads == 2 && c.head_dim == 256 &&
        c.rotary_dim == 64;
    if (fused_projection) {
      PrepareHalfInput(x, n_tokens, l.attn_qkv.cols);
      prepared = AttentionF16Gemm(
          l.attn_qkv.data, static_cast<const __half*>(s_.x_half),
          l.attn_q_norm.f32(), l.attn_k_norm.f32(), s_.q, s_.attn_gate, write_k,
          write_v, n_tokens, pos, c.rope_theta, c.rms_eps, stream_, s.rope,
          yarn_, write_rows);
      if (!prepared) {
        AssignError(error_msg, "fused attention projection failed");
        return false;
      }
    } else {
      if (!projections_ready &&
          !Dense(l.attn_qkv, x, s_.qg, n_tokens, error_msg)) {
        return false;
      }
      prepared = PrepareAttention(
          s_.qg, l.attn_qkv.rows, l.attn_q_norm.f32(), l.attn_k_norm.f32(),
          s_.q, s_.attn_gate, write_k, write_v, n_tokens, c.num_heads,
          c.num_kv_heads, c.head_dim, c.rotary_dim, pos, c.rope_theta,
          c.rms_eps, stream_, s.rope, prefill_phase, yarn_, write_rows);
      if (!prepared) {
        UnpackQGate(s_.qg, l.attn_qkv.rows, s_.q, s_.attn_gate, s_.k, s_.v,
                    n_tokens, c.num_heads, c.head_dim, kv_row, stream_);
      }
    }
  } else {
    Q8Input xq;
    if (!projections_ready && l.attn_q.exl3 != nullptr) {
      if (!Exl3Group({&l.attn_q, &l.attn_k, &l.attn_v}, {s_.qg, s_.k, s_.v}, x,
                     n_tokens, error_msg))
        return false;
    } else if (!projections_ready &&
               (!Quantize(x, n_tokens, c.hidden_size, &xq, error_msg) ||
                !Dense(l.attn_q, xq, s_.qg, error_msg) ||
                !Dense(l.attn_k, xq, s_.k, error_msg) ||
                !Dense(l.attn_v, xq, s_.v, error_msg))) {
      return false;
    }
    // EXL3 prefill: the query rows are unpacked, normalized, rotated by
    // RoPE and (KVarN) into the KVarN frame in one pass; a KVarN layer's
    // output gate then reads the packed rows directly.
    const bool kvarn_rows = kvarn != nullptr;
    // Only where the WMMA launch accepts the extent (its fallback reruns the
    // query pass for the unpacked gate).
    packed_gate = !kvarn_rows && MatrixRows(n_tokens) && wmma_limit != 0 &&
                  (!sparse || (live_context <= wmma_limit &&
                               live_context <= std::uint64_t{mask_words_} *
                                                   c.compress_ratio * 32));
    query_rows =
        prefill_phase && model_->output().exl3 != nullptr &&
        AttentionQueryRows(
            s_.qg, 2 * c.AttentionQDim(), l.attn_q_norm.f32(), s_.q,
            kvarn_rows || packed_gate ? nullptr : s_.attn_gate, n_tokens,
            c.num_heads, c.head_dim, c.rotary_dim, pos, c.rope_theta, c.rms_eps,
            kvarn_rows, stream_, s.rope, yarn_);
    packed_gate = packed_gate && query_rows;
    // Decode/verify: one launch unpacks the query, normalizes and rotates
    // q/k and writes both caches from the split K/V rows (the stacked
    // route's PrepareAttention; RoPE with its explicit FMAs).
    if (query_rows) {
      attn_q_rotated_ = kvarn_rows;
      attn_gate_stride_ = kvarn_rows ? 2 * c.AttentionQDim() : 0;
    } else if (!prefill_phase && model_->output().exl3 != nullptr &&
               PrepareAttention(
                   s_.qg, 2 * c.AttentionQDim(), l.attn_q_norm.f32(),
                   l.attn_k_norm.f32(), s_.q, s_.attn_gate, write_k, write_v,
                   n_tokens, c.num_heads, c.num_kv_heads, c.head_dim,
                   c.rotary_dim, pos, c.rope_theta, c.rms_eps, stream_, s.rope,
                   false, yarn_, write_rows, s_.k, s_.v)) {
      prepared = true;
    } else {
      UnpackQGate(s_.qg, 2 * c.AttentionQDim(), s_.q, s_.attn_gate, nullptr,
                  nullptr, n_tokens, c.num_heads, c.head_dim, 0, stream_);
    }
  }
  if (!prepared) {
    if (!query_rows)
      RmsNormRows(s_.q, l.attn_q_norm.f32(), s_.q, n_tokens * c.num_heads,
                  c.head_dim, 1, c.rms_eps, stream_);
    RmsNormRows(s_.k, l.attn_k_norm.f32(), s_.k, n_tokens * c.num_kv_heads,
                c.head_dim, 1, c.rms_eps, stream_);
    if (!query_rows)
      Rope(s_.q, n_tokens, c.num_heads, c.head_dim, c.rotary_dim, pos,
           c.rope_theta, stream_, s.rope, yarn_);
    Rope(s_.k, n_tokens, c.num_kv_heads, c.head_dim, c.rotary_dim, pos,
         c.rope_theta, stream_, s.rope, yarn_);
    StoreKv(s_.k, write_k, n_tokens, kv_row, pos, stream_, write_rows);
    StoreKv(s_.v, write_v, n_tokens, kv_row, pos, stream_, write_rows);
  }
  if (kvarn != nullptr) {
    if (checkpoint_tokens != 0) {
      AssignError(error_msg, "KVarN sessions take no in-pass checkpoints");
      return false;
    }
    if (kvarn->prefill) {
      RotateKvRows(kvarn_staging_k_, kvarn_staging_v_, n_tokens, c.num_kv_heads,
                   pos, stream_, kvarn->staging_first);
    } else {
      RotateKvRows(s.kvarn_k, s.kvarn_v, n_tokens, c.num_kv_heads, pos,
                   stream_);
      // A group ending inside the pass is quantized before the rows after
      // it read it (hip::PlanKvarnPass boundary).
      KvarnQuantizePassBoundary(s.kvarn_k, s.kvarn_v, s.kvarn_records,
                                s.kv_blocks, c.num_kv_heads, pos, n_tokens,
                                stream_, options_.kvarn_bits);
    }
  }

  // Raw keys survive only until pooling. The ring holds the initial
  // pre-budget backlog and a batch, independently for trunk and predictor.
  if (shared_indexer &&
      !Exl3DensePrefill({&l.indexer_q, &l.indexer_k}, {s_.iq, s_.ik}, x,
                        n_tokens, error_msg))
    return false;
  if (side_indexer ? !JoinSide(error_msg) : !indexer())
    return false;
  // Wide batches run the fused WMMA kernel (never inside a graph: the kv
  // extent is a host value), output gate included, skipping the key tiles
  // no query of a block selected; the per-token kernel covers the rest.
  if (last_only &&
      !Check(hipMemsetAsync(s_.ctx, 0,
                            static_cast<std::size_t>(n_tokens) *
                                c.AttentionQDim() * sizeof(float),
                            stream_),
             "partial attention output initialization", error_msg)) {
    return false;
  }
  if (kvarn != nullptr) {
    return KvarnAttention(l, s, n_tokens, pos, start_pos, mask, last_only,
                          project_output, out, error_msg);
  }
  // Queries before the token budget always take the dense tiles, also in a
  // batch that crosses it, so each position's key sweep (and its rounding)
  // does not depend on where the prefill chunks fall. The sparse part goes
  // first: it is the launch that can be refused.
  const auto attend = [&](std::uint32_t first, std::uint32_t rows, bool last) {
    const std::uint32_t pos = start_pos + first;
    const std::uint32_t dense_rows = mask != nullptr && pos < c.indexer_top_k
                                         ? std::min(rows, c.indexer_top_k - pos)
                                         : 0;
    const auto at = [&](std::uint32_t row) {
      return std::size_t{first + row} * c.AttentionQDim();
    };
    const std::uint32_t gate_stride = packed_gate ? 2 * c.AttentionQDim() : 0;
    const auto gate = [&](std::uint32_t row) {
      return packed_gate
                 ? s_.qg + std::size_t{first + row} * gate_stride + c.head_dim
                 : s_.attn_gate + at(row);
    };
    return (dense_rows == rows ||
            WmmaCausalAttention(
                s_.q + at(dense_rows), gate(dense_rows), s.k_cache, s.v_cache,
                mask ? mask + std::size_t{first + dense_rows} * mask_words_
                     : nullptr,
                mask_words_, s_.ctx + at(dense_rows), rows - dense_rows,
                pos + dense_rows, c.num_heads, c.num_kv_heads, c.head_dim,
                c.compress_ratio, stream_, last, s.kv_blocks, nullptr,
                gate_stride)) &&
           (dense_rows == 0 || (last && dense_rows != rows) ||
            WmmaCausalAttention(s_.q + at(0), gate(0), s.k_cache, s.v_cache,
                                nullptr, mask_words_, s_.ctx + at(0),
                                dense_rows, pos, c.num_heads, c.num_kv_heads,
                                c.head_dim, c.compress_ratio, stream_, last,
                                s.kv_blocks, nullptr, gate_stride));
  };
  if (checkpoint_tokens != 0 && checkpoint_tokens < n_tokens) {
    // Sparse tiles compact the union of their queries' selected keys.
    // Keep the checkpoint's query grouping identical to a pass ending there.
    if (!attend(0, checkpoint_tokens, false) ||
        !attend(checkpoint_tokens, n_tokens - checkpoint_tokens, false)) {
      AssignError(error_msg, "checkpoint attention geometry is unsupported");
      return false;
    }
    return !project_output ||
           Dense(l.attn_out, s_.ctx, out, n_tokens, error_msg);
  }
  if (MatrixRows(n_tokens) && attend(0, n_tokens, last_only)) {
    return !project_output ||
           Dense(l.attn_out, s_.ctx, out, n_tokens, error_msg);
  }
  // A refused WMMA launch: the per-token kernel and SigmoidMul read the
  // unpacked gate. Rerun the query pass for it (it rewrites q unchanged).
  if (packed_gate &&
      !AttentionQueryRows(s_.qg, 2 * c.AttentionQDim(), l.attn_q_norm.f32(),
                          s_.q, s_.attn_gate, n_tokens, c.num_heads, c.head_dim,
                          c.rotary_dim, pos, c.rope_theta, c.rms_eps, false,
                          stream_, s.rope, yarn_)) {
    AssignError(error_msg, "attention gate unpack failed");
    return false;
  }
  // Narrow batches split each row's key tiles over kAttnSplits blocks so a
  // decode step at depth fills the device.
  const bool split = !MatrixRows(n_tokens);
  rocm::Attention(s_.q, s.k_cache, s.v_cache, mask, mask_words_, s_.ctx,
                  split ? s_.attn_partials : nullptr, kAttnSplits, n_tokens,
                  pos, c.num_heads, c.num_kv_heads, c.head_dim,
                  c.compress_ratio, stream_, split ? s_.attn_lists : nullptr,
                  split ? s_.attn_counts : nullptr,
                  c.compress_ratio > 0 ? c.indexer_top_k / c.compress_ratio : 0,
                  s.kv_blocks, nullptr,
                  model_->output().exl3 != nullptr ? c.indexer_top_k : 0);
  const std::uint32_t q_dim = c.AttentionQDim();
  const std::size_t ctx_count = static_cast<std::size_t>(n_tokens) * q_dim;
  // EXL3 decode: the gate also rotates the context for attn_out.
  const DeviceTensor* const out_consumer[] = {&l.attn_out};
  const auto out_slabs =
      project_output ? Exl3Slabs(out_consumer, n_tokens) : Exl3InputSlabs{};
  // A decode batch's output projection reads the gate's Q8_1 rows.
  if (project_output && split && l.attn_out.type == GgmlType::kQ8_0 &&
      l.attn_out.cols == q_dim && UnpaddedQ8(q_dim) &&
      SigmoidMulQ8_1(s_.ctx, s_.attn_gate, ctx_count, s_.ctx_q8_1, stream_)) {
    return Dense(l.attn_out, Q8Input{s_.ctx, s_.ctx_q8_1, n_tokens, q_dim}, out,
                 error_msg);
  }
  if (out_slabs.count != 0 && SigmoidMulRotate(s_.ctx, s_.attn_gate, n_tokens,
                                               q_dim, out_slabs, stream_))
    NoteExl3Rotated(out_slabs, s_.ctx, n_tokens, out_consumer);
  else
    SigmoidMul(s_.ctx, s_.attn_gate, ctx_count, stream_);
  return !project_output || Dense(l.attn_out, s_.ctx, out, n_tokens, error_msg);
}

bool Executor::KvarnAttention(const DeviceLayer& l,
                              const Session::AttentionState& s,
                              std::uint32_t n_tokens, const std::uint32_t* pos,
                              std::uint32_t start_pos,
                              const std::uint32_t* mask, bool last_only,
                              bool project_output, float* out,
                              std::string* error_msg) const {
  const Config& c = config();
  const KvarnPass& pass = *s.kvarn_pass;
  const std::uint32_t kv_row = c.AttentionKvDim();
  const std::size_t row_bytes = std::size_t{kv_row} * sizeof(__half);
  const KvarnReadView view{
      .k_fp16 = s.kvarn_k,
      .v_fp16 = s.kvarn_v,
      .k_staging = pass.prefill ? kvarn_staging_k_ : nullptr,
      .v_staging = pass.prefill ? kvarn_staging_v_ : nullptr,
      .staging_first = pass.staging_first,
      .records = s.kvarn_records,
      .blocks = s.kv_blocks,
      .quant_end = pass.prefill ? pass.quant_end : 0,
      .kv_heads = c.num_kv_heads,
      .bits = options_.kvarn_bits};
  // Scores and context in the rotated frame: H q . H k = q . k, and the
  // context rotates back before the output gate.
  const bool q_rotated = attn_q_rotated_;
  const std::uint32_t gate_stride = attn_gate_stride_;
  attn_q_rotated_ = false;
  attn_gate_stride_ = 0;
  if (!q_rotated)
    RotateHeadRows(s_.q, n_tokens * c.num_heads, stream_);
  bool attended = false;
  if (pass.prefill && MatrixRows(n_tokens)) {
    // The FP16 path's split: queries before the token budget take the dense
    // tiles. A prompt chunk reads one fixed record range, so the fused
    // kernel's shared key tiles hold for every query.
    const std::uint32_t dense_rows =
        mask != nullptr && start_pos < c.indexer_top_k
            ? std::min(n_tokens, c.indexer_top_k - start_pos)
            : 0;
    const auto at = [&](std::uint32_t row) {
      return std::size_t{row} * c.AttentionQDim();
    };
    attended =
        (dense_rows == n_tokens ||
         WmmaCausalAttention(
             s_.q + at(dense_rows), nullptr, nullptr, nullptr,
             mask ? mask + std::size_t{dense_rows} * mask_words_ : nullptr,
             mask_words_, s_.ctx + at(dense_rows), n_tokens - dense_rows,
             start_pos + dense_rows, c.num_heads, c.num_kv_heads, c.head_dim,
             c.compress_ratio, stream_, last_only, s.kv_blocks, &view)) &&
        (dense_rows == 0 || (last_only && dense_rows != n_tokens) ||
         WmmaCausalAttention(
             s_.q, nullptr, nullptr, nullptr, nullptr, mask_words_, s_.ctx,
             dense_rows, start_pos, c.num_heads, c.num_kv_heads, c.head_dim,
             c.compress_ratio, stream_, last_only, s.kv_blocks, &view));
  }
  // Passes read records by each query's own position (KvarnReadGroups), so
  // their rows run one per query as decode steps do.
  const bool split = !MatrixRows(n_tokens);
  if (!attended) {
    rocm::Attention(
        s_.q, nullptr, nullptr, mask, mask_words_, s_.ctx,
        split ? s_.attn_partials : nullptr, kAttnSplits, n_tokens, pos,
        c.num_heads, c.num_kv_heads, c.head_dim, c.compress_ratio, stream_,
        split ? s_.attn_lists : nullptr, split ? s_.attn_counts : nullptr,
        c.compress_ratio > 0 ? c.indexer_top_k / c.compress_ratio : 0,
        s.kv_blocks, &view,
        model_->output().exl3 != nullptr ? c.indexer_top_k : 0);
  }
  if (gate_stride == 0)
    RotateHeadRows(s_.ctx, n_tokens * c.num_heads, stream_);
  if (pass.prefill) {
    // Quantize the groups the chunk completed from staging, then keep its
    // sink rows and its recent rows.
    const std::uint32_t end = start_pos + n_tokens;
    const std::uint32_t first =
        pass.staging_first == 0 ? 1
                                : pass.staging_first >> hip::kKvarnGroupShift;
    const std::uint32_t complete = end >> hip::kKvarnGroupShift;
    if (complete > first)
      KvarnQuantizeGroups(kvarn_staging_k_, kvarn_staging_v_, 0,
                          s.kvarn_records, 0, s.kv_blocks, c.num_kv_heads, 1,
                          first, complete - first, stream_, pass.staging_first,
                          options_.kvarn_bits);
    const auto copy = [&](std::size_t to, std::uint32_t from_position,
                          std::uint32_t rows) {
      const std::size_t from = from_position - pass.staging_first;
      return rows == 0 ||
             (Check(hipMemcpyAsync(s.kvarn_k + to * kv_row,
                                   kvarn_staging_k_ + from * kv_row,
                                   rows * row_bytes, hipMemcpyDeviceToDevice,
                                   stream_),
                    "KVarN chunk rows", error_msg) &&
              Check(hipMemcpyAsync(s.kvarn_v + to * kv_row,
                                   kvarn_staging_v_ + from * kv_row,
                                   rows * row_bytes, hipMemcpyDeviceToDevice,
                                   stream_),
                    "KVarN chunk rows", error_msg));
    };
    if (pass.staging_first == 0 &&
        !copy(0, 0, std::min(end, hip::kKvarnSinkTokens)))
      return false;
    // The ring keeps every row a later pass may rewind into
    // (hip::PlanKvarnPass): the groups starting within one ring of the end.
    // Pieces split at group starts, where the ring may wrap.
    constexpr std::uint32_t kGroup = hip::kKvarnGroupTokens;
    const std::uint32_t reach =
        end > hip::kKvarnTailRingTokens
            ? (end - hip::kKvarnTailRingTokens + kGroup - 1) / kGroup * kGroup
            : 0;
    for (std::uint32_t p =
             std::max({hip::kKvarnSinkTokens, pass.staging_first, reach});
         p < end;) {
      const std::uint32_t piece = std::min(end, (p / kGroup + 1) * kGroup);
      if (!copy(hip::KvarnFp16Row(p), p, piece - p))
        return false;
      p = piece;
    }
  }
  const std::uint32_t q_dim = c.AttentionQDim();
  const std::size_t ctx_count = static_cast<std::size_t>(n_tokens) * q_dim;
  // EXL3 decode: the gating kernel also rotates the context for attn_out.
  const DeviceTensor* const out_consumer[] = {&l.attn_out};
  const auto out_slabs =
      project_output ? Exl3Slabs(out_consumer, n_tokens) : Exl3InputSlabs{};
  if (gate_stride != 0) {
    if (GateRotatedContext(s_.ctx, s_.qg, gate_stride, n_tokens, c.num_heads,
                           stream_, out_slabs))
      NoteExl3Rotated(out_slabs, s_.ctx, n_tokens, out_consumer);
    return !project_output ||
           Dense(l.attn_out, s_.ctx, out, n_tokens, error_msg);
  }
  if (project_output && split && l.attn_out.type == GgmlType::kQ8_0 &&
      l.attn_out.cols == q_dim && UnpaddedQ8(q_dim) &&
      SigmoidMulQ8_1(s_.ctx, s_.attn_gate, ctx_count, s_.ctx_q8_1, stream_)) {
    return Dense(l.attn_out, Q8Input{s_.ctx, s_.ctx_q8_1, n_tokens, q_dim}, out,
                 error_msg);
  }
  if (out_slabs.count != 0 && SigmoidMulRotate(s_.ctx, s_.attn_gate, n_tokens,
                                               q_dim, out_slabs, stream_))
    NoteExl3Rotated(out_slabs, s_.ctx, n_tokens, out_consumer);
  else
    SigmoidMul(s_.ctx, s_.attn_gate, ctx_count, stream_);
  return !project_output || Dense(l.attn_out, s_.ctx, out, n_tokens, error_msg);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
