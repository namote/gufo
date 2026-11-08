// MoE layers: routing hints, routed expert GEMMs (GGUF, WMMA and EXL3)
// and the MoE block with its shared expert.

#include "src/models/qwen38_flash_next/kernels/rocm/executor_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

void Executor::RoutedHints(const DeviceTensor& w,
                           std::uint32_t n_tokens) const {
  // The column grid is bounded by the largest expert bucket and the tile
  // width fitted to the whole distribution (see RouteHints); the fallback
  // bound is the token count with a tile near twice the mean bucket.
  if (routed_max_rows_ > 0) {
    qfn_mmq_set_routed_max_expert_rows(static_cast<int>(routed_max_rows_));
    qfn_mmq_set_routed_tile_cols(routed_tile_cols_);
    return;
  }
  qfn_mmq_set_routed_max_expert_rows(static_cast<int>(n_tokens));
  qfn_mmq_set_routed_tile_cols(
      RoutedTileCols(n_tokens, config().num_experts_used, w.experts));
}

bool Executor::RouteHints(std::uint32_t n_tokens,
                          std::string* error_msg) const {
  // Every column tile past an expert's bucket still costs a dispatch and a
  // full shared-memory reservation, so the grid is cut to the real largest
  // bucket. Counts are downloaded before the shared expert; wait only for
  // that download while the shared expert continues (on the side stream in
  // overlapped passes, joined by JoinSide).
  const Config& c = config();
  routed_max_rows_ = 0;
  routed_n_tiles_ = 0;
  routed_64_tiles_ = 0;
  routed_pair_tiles_ = 0;
  routed_pair_offset_ = 0;
  if (!ExpertMatrixRows(n_tokens)) {
    return true;
  }
  if (!Check(hipEventSynchronize(counts_ready_), "expert counts", error_msg)) {
    return false;
  }
  // The F16 expert GEMM launches one block per (expert, row tile of its
  // 16-padded bucket): the map is built here and uploaded ahead of the
  // launches on the same stream. Tile heights are multiples of 16, so a
  // bucket's tile count follows from its unpadded rows.
  std::uint32_t max_rows = 0;
  std::uint32_t n_tiles = 0;
  routed_tile_rows_ = RoutedTileRows(
      static_cast<std::size_t>(n_tokens) * c.num_experts_used, c);
  for (std::uint32_t e = 0; e < c.num_experts; ++e) {
    max_rows = std::max(max_rows, counts_host_[e]);
  }
  routed_max_rows_ = std::max<std::uint32_t>(1, max_rows);
  const auto append = [&](std::uint32_t tile_rows) {
    for (std::uint32_t e = 0; e < c.num_experts; ++e) {
      for (std::uint32_t j = 0;
           j < (counts_host_[e] + tile_rows - 1) / tile_rows; ++j) {
        tiles_host_[n_tiles++] = static_cast<std::int32_t>(e | (j << 16));
      }
    }
  };
  if (WidePrefill(n_tokens)) {
    // Prefill batches: 64-token tiles for the down projection, then 128-token
    // tiles for the paired gate/up projections.
    append(64);
    routed_64_tiles_ = n_tiles;
    routed_pair_offset_ = n_tiles;
    append(128);
    routed_pair_tiles_ = n_tiles - routed_pair_offset_;
  } else {
    append(routed_tile_rows_);
    routed_n_tiles_ = n_tiles;
  }
  routed_tile_cols_ = qfn_mmq_routed_tile_cols_for_counts(
      counts_host_, static_cast<int>(c.num_experts));
  return n_tiles == 0 || Check(hipMemcpyAsync(s_.routed_tiles, tiles_host_,
                                              n_tiles * sizeof(std::int32_t),
                                              hipMemcpyHostToDevice, stream_),
                               "routed tile map upload", error_msg);
}

bool Executor::Experts(const DeviceTensor& w, const float* x,
                       const std::int32_t* ids, float* out,
                       std::uint32_t n_rows, std::uint32_t n_used,
                       std::uint32_t n_tokens, std::string* error_msg) const {
  const int M = static_cast<int>(w.rows);
  const int K = static_cast<int>(w.cols);
  const int E = static_cast<int>(w.experts);
  const int T = static_cast<int>(n_rows);
  const int U = static_cast<int>(n_used);
  // The vector entries loop over column chunks, so the decode-time down
  // projection (top-k rows, one expert each) stays on them as well.
  // Verification keeps the same quantization and reduction as single-token
  // decoding even when top-k expansion produces more than 32 slot rows.
  const bool tiled =
      (prefill_phase || n_rows > 4 * kVecBatch) && MatrixRows(n_tokens);
  if (tiled) {
    RoutedHints(w, n_tokens);
  }
  int rc = -1;
  if (!tiled) {
    rc = qfn_mmq_moe_vec(static_cast<int>(w.type), w.data, x, ids, out, M, K, T,
                         E, U, stream_);
  } else {
    switch (w.type) {
      case GgmlType::kQ4_K:
        rc = qfn_mmq_q4_K_moe_raw(w.data, x, ids, out, M, K, T, E, U, stream_);
        break;
      case GgmlType::kQ5_K:
        rc = qfn_mmq_q5_K_moe_raw(w.data, x, ids, out, M, K, T, E, U, stream_);
        break;
      case GgmlType::kQ5_1:
        rc = qfn_mmq_q5_1_moe_raw(w.data, x, ids, out, M, K, T, E, U, stream_);
        break;
      case GgmlType::kQ8_0:
        rc = qfn_mmq_q8_0_moe_raw(w.data, x, ids, out, M, K, T, E, U, stream_);
        break;
      default:
        break;
    }
  }
  if (rc != 0) {
    AssignError(error_msg, "expert GEMM failed");
    return false;
  }
  return true;
}

bool Executor::GatedExperts(const DeviceTensor& a, const DeviceTensor& b,
                            const float* x, const std::int32_t* ids, float* out,
                            std::uint32_t n_tokens, std::uint32_t n_used,
                            std::string* error_msg) const {
  const bool same_shape = a.type == b.type && a.rows == b.rows &&
                          a.cols == b.cols && a.experts == b.experts;
  if (!MatrixRows(n_tokens) && n_used <= 32 && same_shape &&
      (a.type == GgmlType::kQ4_K || a.type == GgmlType::kQ5_K ||
       a.type == GgmlType::kQ8_0)) {
    // The shared expert quantized the same rows (or their producer did).
    const void* ready = ReadyQ8(x, n_tokens, a.cols);
    if ((ready != nullptr
             ? qfn_mmq_moe_gated_vec_preq(
                   static_cast<int>(a.type), a.data, b.data, ready, ids, out,
                   static_cast<int>(a.rows), static_cast<int>(a.cols),
                   static_cast<int>(n_tokens), static_cast<int>(a.experts),
                   static_cast<int>(n_used), stream_)
             : qfn_mmq_moe_gated_vec(
                   static_cast<int>(a.type), a.data, b.data, x, ids, out,
                   static_cast<int>(a.rows), static_cast<int>(a.cols),
                   static_cast<int>(n_tokens), static_cast<int>(a.experts),
                   static_cast<int>(n_used), stream_)) != 0) {
      AssignError(error_msg, "gated expert vector projection failed");
      return false;
    }
    return true;
  }
  if (!ExpertMatrixRows(n_tokens) && same_shape) {
    if (qfn_mmq_moe_vec(static_cast<int>(a.type), a.data, x, ids, out,
                        static_cast<int>(a.rows), static_cast<int>(a.cols),
                        static_cast<int>(n_tokens), static_cast<int>(a.experts),
                        static_cast<int>(n_used), stream_, b.data,
                        s_.up_e) != 0) {
      AssignError(error_msg, "expert vector pair GEMM failed");
      return false;
    }
  } else if (same_shape && a.type == GgmlType::kQ4_K) {
    // The wide Q4_K path shares its gather and tiled quantization as well.
    RoutedHints(a, n_tokens);
    if (qfn_mmq_q4_K_moe_pair_unique(
            a.data, b.data, x, ids, out, s_.up_e, static_cast<int>(a.rows),
            static_cast<int>(a.cols), static_cast<int>(n_tokens),
            static_cast<int>(a.experts), static_cast<int>(n_used),
            stream_) != 0) {
      AssignError(error_msg, "expert pair GEMM failed");
      return false;
    }
  } else if (!Experts(a, x, ids, out, n_tokens, n_used, n_tokens, error_msg) ||
             !Experts(b, x, ids, s_.up_e, n_tokens, n_used, n_tokens,
                      error_msg)) {
    return false;
  }
  Swiglu(out, s_.up_e, static_cast<std::size_t>(n_tokens) * n_used * a.rows,
         stream_);
  return true;
}

bool Executor::Moe(const DeviceLayer& l, const float* x, float* out,
                   std::uint32_t n_tokens, std::string* error_msg,
                   bool last_only) const {
  const Config& c = config();
  const std::uint32_t used = c.num_experts_used;
  // This guard must precede the early shared fork: router failures must
  // still order all side writes before a subsequent pass reuses scratch.
  struct JoinGuard {
    const Executor* executor;
    ~JoinGuard() { (void)executor->JoinSide(nullptr); }
  } join_guard{this};
  const auto full_exl3 = [](const DeviceTensor& w) {
    return w.exl3 != nullptr && w.experts == 1 && w.output_begin == 0 &&
           w.rows == w.exl3->n && w.cols == w.exl3->k;
  };
  const bool exl3_overlap = prefill_phase && !last_only && n_tokens >= 512 &&
                            exl3_shared_workspace_.input != nullptr &&
                            n_tokens <= options_.max_batch &&
                            full_exl3(l.shexp_gate) && full_exl3(l.shexp_up) &&
                            full_exl3(l.shexp_down);
  if (exl3_overlap) {
    if (!Check(hipEventRecord(side_start_, stream_), "shared EXL3 fork",
               error_msg) ||
        !Check(hipStreamWaitEvent(side_stream_, side_start_, 0),
               "shared EXL3 wait", error_msg))
      return false;
    shexp_half_ready_ = false;
    const auto shared = [&] {
      if (!Exl3DensePrefill({&l.shexp_gate, &l.shexp_up},
                            {s_.shexp_up, s_.shexp_gate}, x, n_tokens,
                            error_msg, &exl3_shared_workspace_, side_stream_))
        return false;
      Swiglu(s_.shexp_up, s_.shexp_gate,
             std::size_t{n_tokens} * l.shexp_gate.rows, side_stream_);
      return Exl3DensePrefill({&l.shexp_down}, {s_.shexp_out}, s_.shexp_up,
                              n_tokens, error_msg, &exl3_shared_workspace_,
                              side_stream_);
    };
    const bool shared_ok = shared();
    // Even a partial enqueue must be joined on every exit.
    if (!Check(hipEventRecord(side_done_, side_stream_), "shared EXL3 join",
               error_msg)) {
      (void)hipStreamSynchronize(side_stream_);
      return false;
    }
    side_pending_ = true;
    if (!shared_ok)
      return false;
    if (diagnostics_active_)
      ++cycle_diagnostics_.prefill_shared_overlaps;
  }
  // Router logits and the shared-expert gate come out of one GEMM.
  if (!Dense(l.router, x, s_.router, n_tokens, error_msg)) {
    return false;
  }
  // An EXL3 decode call of one routed chunk also builds its expert tiles in
  // the router launch (Exl3RoutedMlp skips its counter reset and BuildTiles).
  exl3_tiles_ready_ = l.ffn_gate_exps.exl3 != nullptr && !prefill_phase &&
                      n_tokens <= 64 &&
                      n_tokens * used <= kExl3FusedTileAssignments &&
                      exl3_.route_arrivals != nullptr;
  if (exl3_tiles_ready_)
    RouterTopKTiles(s_.router, c.num_experts + 1, s_.ids, s_.weights, n_tokens,
                    c.num_experts, used, exl3_.tiles, exl3_.tile_count,
                    exl3_.route_arrivals, Exl3RouteTileRows(n_tokens), stream_);
  else
    RouterTopK(s_.router, c.num_experts + 1, s_.ids, s_.weights, n_tokens,
               c.num_experts, used, stream_);
  if (l.ffn_gate_exps.exl3 == nullptr && ExpertMatrixRows(n_tokens)) {
    ExpertCounts(s_.ids, s_.expert_counts, n_tokens, c.num_experts, used,
                 stream_);
    if (!Check(hipMemcpyAsync(counts_host_, s_.expert_counts,
                              c.num_experts * sizeof(std::uint32_t),
                              hipMemcpyDeviceToHost, stream_),
               "expert counts download", error_msg) ||
        !Check(hipEventRecord(counts_ready_, stream_), "expert counts event",
               error_msg)) {
      return false;
    }
  }
  // The shared expert (gated by the last router row) does not depend on
  // routing. Queue its GEMMs after the count download, then prepare the
  // routed dispatch on the CPU without waiting for these GEMMs to finish.
  // A wide batch runs them on the side stream beside the routed experts,
  // which leave compute idle while they stream expert weights. The shared
  // expert writes its own scratch and may quantize x into s_.x_q8t, so no
  // main-stream work may use s_.x_q8t before JoinSide.
  // The fork below overlaps wide Q8_0 batches and EXL3 decode/verify; EXL3
  // prefill uses the explicit-stream workspace above. Other routes retain
  // their original stream ownership.
  // EXL3 decode/verify: the fused shared MLP (its own fast-path slabs,
  // counters and shexp rows) runs beside the routed GEMVs (routed_input,
  // tiles, gate_e/up_e/down_e); Exl3Experts joins before the combine reads
  // shexp_out.
  const bool exl3_decode_overlap =
      !exl3_overlap && !prefill_phase && !last_only &&
      full_exl3(l.shexp_gate) && full_exl3(l.shexp_up) &&
      full_exl3(l.shexp_down) && l.ffn_gate_exps.exl3 != nullptr;
  const bool overlap = exl3_decode_overlap ||
                       (!exl3_overlap && ExpertMatrixRows(n_tokens) &&
                        !last_only && l.shexp_gate.type == GgmlType::kQ8_0 &&
                        l.shexp_up.type == GgmlType::kQ8_0 &&
                        l.shexp_down.type == GgmlType::kQ8_0);
  const hipStream_t main = stream_;
  if (overlap) {
    if (!Check(hipEventRecord(side_start_, main), "side stream fork",
               error_msg) ||
        !Check(hipStreamWaitEvent(side_stream_, side_start_, 0),
               "side stream wait", error_msg)) {
      return false;
    }
    stream_ = side_stream_;
  }
  // However the shared expert exits, later work returns to the main stream.
  struct MainStream {
    const Executor* executor;
    hipStream_t main;
    ~MainStream() { executor->stream_ = main; }
  } main_stream{this, main};
  const DeviceTensor* const shared_inputs[] = {&l.shexp_gate, &l.shexp_up};
  const auto shared_expert = [&] {
    shexp_half_ready_ = false;
    if (l.shexp_gate.exl3 != nullptr && !prefill_phase &&
        Exl3SharedMlp(l, x, s_.shexp_gate, s_.shexp_up, s_.shexp_out, n_tokens,
                      exl3_, stream_,
                      TakeExl3Rotated(x, n_tokens, shared_inputs)))
      return true;
    if (!GatedDense(l.shexp_up, l.shexp_gate, x, s_.shexp_up, n_tokens,
                    &l.shexp_down, error_msg)) {
      return false;
    }
    if (shexp_half_ready_) {
      shexp_half_ready_ = false;
      if (!DenseF16Gemm(l.shexp_down.data, s_.shexp_half, s_.shexp_out,
                        n_tokens, l.shexp_down.rows, l.shexp_down.cols,
                        stream_)) {
        AssignError(error_msg, "shared expert F16 GEMM failed");
        return false;
      }
      return true;
    }
    return Dense(l.shexp_down, s_.shexp_up, s_.shexp_out, n_tokens, error_msg);
  };
  const bool shared_ok = exl3_overlap || shared_expert();
  stream_ = main;
  if (overlap) {
    if (!Check(hipEventRecord(side_done_, side_stream_), "side stream join",
               error_msg)) {
      // Nothing to join: wait for the whole side stream instead (not while
      // a graph captures it).
      if (!SideCapturing())
        (void)hipStreamSynchronize(side_stream_);
      return false;
    }
    side_pending_ = true;
  }
  if (!shared_ok) {
    return false;
  }
  if (last_only && l.ffn_gate_exps.type == GgmlType::kQ8_0 &&
      l.ffn_up_exps.type == GgmlType::kQ8_0 &&
      l.ffn_down_exps.type == GgmlType::kQ8_0) {
    // Catch-up retains only the final predictor residual. Keep the router
    // and shared-expert shapes, then run its routed experts with the same
    // tiled quantization and accumulation as the complete prompt batch.
    const auto base = s_;
    UseScratch(RowScratch(base, n_tokens - 1));
    PrefillPhase phase(true);
    try {
      const std::size_t offset =
          static_cast<std::size_t>(n_tokens - 1) * c.hidden_size;
      const bool ok = MoeExperts(l, x + offset, out + offset, 1, error_msg);
      UseScratch(base);
      return ok;
    } catch (...) {
      UseScratch(base);
      throw;
    }
  }
  return MoeExperts(l, x, out, n_tokens, error_msg) && JoinSide(error_msg);
}

bool Executor::WmmaExperts(const DeviceLayer& l, std::uint32_t n_tokens) const {
  const Config& c = config();
  return ExpertMatrixRows(n_tokens) &&
         (l.ffn_gate_exps.type == GgmlType::kQ4_K ||
          l.ffn_gate_exps.type == GgmlType::kQ5_K) &&
         l.ffn_up_exps.type == l.ffn_gate_exps.type &&
         (l.ffn_down_exps.type == GgmlType::kQ5_1 ||
          l.ffn_down_exps.type == GgmlType::kQ8_0) &&
         c.hidden_size % 256 == 0 && c.expert_ff % 64 == 0;
}

bool Executor::MoeExperts(const DeviceLayer& l, const float* x, float* out,
                          std::uint32_t n_tokens,
                          std::string* error_msg) const {
  const Config& c = config();
  const std::uint32_t used = c.num_experts_used;
  const std::uint32_t slots = n_tokens * used;
  if (l.ffn_gate_exps.exl3 != nullptr)
    return Exl3Experts(l, x, out, n_tokens, error_msg);
  if (!RouteHints(n_tokens, error_msg)) {
    return false;
  }
  // Tiled batches take the WMMA route: assignments compacted by expert
  // into 16-row padded buckets, token rows narrowed to F16 once, then the
  // F16 matrix-core GEMM per (expert, row tile).
  const bool wmma_experts = WmmaExperts(l, n_tokens);
  if (wmma_experts) {
    RoutedCompact(s_.ids, s_.expert_counts, s_.routed_bounds, s_.routed_cursors,
                  s_.rows_token, s_.rows_slot, n_tokens, used, c.num_experts,
                  stream_);
    // The GEMMs read F16 token rows: the router's F16 GEMM (or the mix)
    // usually left them in s_.x_half already.
    if (!(half_src_ == x && half_rows_ == n_tokens &&
          half_cols_ == c.hidden_size && !half_bf16_)) {
      NarrowActivations(x, s_.x_half, false,
                        static_cast<std::size_t>(n_tokens) * c.hidden_size,
                        stream_);
      half_src_ = x;
      half_rows_ = n_tokens;
      half_cols_ = c.hidden_size;
      half_bf16_ = false;
    }
    const auto* x_half = static_cast<const __half*>(s_.x_half);
    // Prefill pairs the gate/up projections and applies SwiGLU without
    // materializing the gate; decode-time batches keep separate projections.
    auto* up_half = reinterpret_cast<__half*>(s_.up_e);
    const WeightType gate_type = l.ffn_gate_exps.type == GgmlType::kQ5_K
                                     ? WeightType::kQ5_K
                                     : WeightType::kQ4_K;
    const bool gated_ok =
        routed_64_tiles_ != 0
            ? RoutedGatedF16Gemm(
                  l.ffn_gate_exps.data, l.ffn_up_exps.data, gate_type, x_half,
                  s_.routed_tiles + routed_pair_offset_, routed_pair_tiles_,
                  s_.routed_bounds, s_.rows_token, s_.rows_slot, up_half,
                  c.expert_ff, c.hidden_size, stream_)
            : (RoutedF16Gemm(l.ffn_gate_exps.data, gate_type, x_half,
                             s_.routed_tiles, routed_n_tiles_,
                             routed_tile_rows_, s_.routed_bounds, s_.rows_token,
                             s_.rows_slot, nullptr, s_.gate_e, nullptr,
                             c.expert_ff, c.hidden_size, stream_) &&
               RoutedF16Gemm(l.ffn_up_exps.data, gate_type, x_half,
                             s_.routed_tiles, routed_n_tiles_,
                             routed_tile_rows_, s_.routed_bounds, s_.rows_token,
                             s_.rows_slot, s_.gate_e, nullptr, up_half,
                             c.expert_ff, c.hidden_size, stream_));
    if (!gated_ok) {
      AssignError(error_msg, "routed F16 gate/up GEMM failed");
      return false;
    }
    const WeightType down_type = l.ffn_down_exps.type == GgmlType::kQ8_0
                                     ? WeightType::kQ8_0
                                     : WeightType::kQ5_1;
    // Prefill takes the 64-token down tiles: they decode each expert's
    // weights over more rows, also where narrower tiles would be fewer.
    const bool wide_down = routed_64_tiles_ != 0;
    // The down projection's rows are F16 too: the epilogue reads half the
    // bytes of the largest routed intermediate.
    if (!RoutedF16Gemm(l.ffn_down_exps.data, down_type, up_half,
                       s_.routed_tiles,
                       wide_down ? routed_64_tiles_ : routed_n_tiles_,
                       wide_down ? 64 : routed_tile_rows_, s_.routed_bounds,
                       s_.rows_slot, s_.rows_slot, nullptr, nullptr,
                       reinterpret_cast<__half*>(s_.down_e), c.hidden_size,
                       c.expert_ff, stream_)) {
      AssignError(error_msg, "routed F16 down GEMM failed");
      return false;
    }
  } else {
    if (!GatedExperts(l.ffn_gate_exps, l.ffn_up_exps, x, s_.ids, s_.gate_e,
                      n_tokens, used, error_msg)) {
      return false;
    }
    // The down projection sees one (token, slot) row per expert id.
    if (!Experts(l.ffn_down_exps, s_.gate_e, s_.ids, s_.down_e, slots, 1,
                 n_tokens, error_msg)) {
      return false;
    }
  }
  // The epilogue reads the shared expert's output.
  if (!JoinSide(error_msg))
    return false;
  if (wmma_experts) {
    // The combine that follows folds this epilogue into its own pass when
    // it takes the F16 route (Combine); otherwise it runs here.
    moe_pending_ = out == s_.block_out;
    if (!moe_pending_) {
      MoeEpilogueVec4F16(reinterpret_cast<const __half*>(s_.down_e), s_.weights,
                         s_.shexp_out, s_.router + c.num_experts,
                         c.num_experts + 1, out, n_tokens, used, c.hidden_size,
                         stream_);
    }
  } else if (MatrixRows(n_tokens)) {
    MoeEpilogueVec4(s_.down_e, s_.weights, s_.shexp_out,
                    s_.router + c.num_experts, c.num_experts + 1, out, n_tokens,
                    used, c.hidden_size, stream_);
  } else {
    MoeEpilogue(s_.down_e, s_.weights, s_.shexp_out, s_.router + c.num_experts,
                c.num_experts + 1, out, n_tokens, used, c.hidden_size, stream_);
  }
  return true;
}

bool Executor::Exl3Experts(const DeviceLayer& l, const float* x, float* out,
                           std::uint32_t rows, std::string* error) const {
  const auto& c = config();
  const bool tiles_ready = std::exchange(exl3_tiles_ready_, false);
  try {
    if (!prefill_phase) {
      for (std::uint32_t first = 0; first < rows; first += 64) {
        const auto count = std::min(64U, rows - first);
        const auto assignment = std::size_t{first} * c.num_experts_used;
        // Raw down dots: Exl3RoutedCombine restores them in the epilogue.
        Exl3RoutedMlp(l, x + std::size_t{first} * c.hidden_size,
                      s_.ids + assignment, s_.gate_e, s_.up_e,
                      s_.down_e + assignment * c.hidden_size, count,
                      c.num_experts_used, exl3_, stream_, nullptr,
                      tiles_ready && rows <= 64, true);
        if (diagnostics_active_) {
          if (exl3_.down_exponents != nullptr)
            cycle_diagnostics_.scaled_down_rows +=
                std::uint64_t{count} * c.num_experts_used;
          if (!DiagnosticCount(exl3_.tile_count,
                               &cycle_diagnostics_.expert_tile_reads, error))
            return false;
        }
      }
    } else {
      // Keep this route for EVERY kPrefill row, including singleton tails.
      // Selecting decode GEMV by token/bucket width changes the reduction
      // tree when a Sync/checkpoint boundary moves. X3 is a phase contract,
      // not a minimum-row heuristic; its 64-row tile never changes here.
      const auto scratch = Exl3PrefillView();
      // Raw down dots stay packed; the combine below restores and sums them.
      Exl3PrefillExpertsRaw(l, x, s_.ids, rows, c.num_experts_used,
                            kExl3PrefillTileRows, scratch, stream_);
      if (diagnostics_active_) {
        cycle_diagnostics_.prefill_expert_rows += rows;
        cycle_diagnostics_.expert_tile_reads_complete = false;
        if (!DiagnosticCount(exl3_prefill_.bucket_count,
                             &cycle_diagnostics_.prefill_expert_buckets, error))
          return false;
      }
    }
    if (!JoinSide(error))
      return false;
    if (prefill_phase) {
      const bool defer = exl3_moe_defer_ && out == s_.block_out;
      exl3_moe_defer_ = false;
      if (defer) {
        exl3_moe_pending_ = {&l, rows};
        return true;
      }
      if (out == s_.block_out)
        ApplyPendingResidual();  // its update reads the attention output
      Exl3PrefillCombine(l, Exl3PrefillView(), s_.ids, rows, c.num_experts_used,
                         s_.weights, s_.shexp_out, s_.router + c.num_experts,
                         c.num_experts + 1, out, stream_);
      return true;
    }
    Exl3RoutedCombine(l, s_.ids, rows, c.num_experts_used, s_.down_e,
                      s_.weights, s_.shexp_out, s_.router + c.num_experts,
                      c.num_experts + 1, out, stream_);
    return true;
  } catch (const std::exception& e) {
    AssignError(error, e.what());
    return false;
  }
}

}  // namespace gufo::models::qwen38_flash_next::rocm
