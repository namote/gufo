// Hyper-connection streams: the residual combine and the HC mix.

#include "src/models/qwen38_flash_next/runtime/executor_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

void Executor::ApplyPendingResidual() const {
  const PendingResidual p = std::exchange(exl3_res_pending_, {});
  if (p.res != nullptr)
    HcApplyDeferred(p.res, p.update.block_out, p.update.w, p.rows,
                    config().hidden_size, stream_);
}

void Executor::Combine(float* res, const float* gamma, std::uint32_t n_tokens,
                       const float* inject_w, bool defer_res,
                       AbliterationSite ablation) const {
  const Config& c = config();
  inject_fused_ = nullptr;
  xn_deferred_ = {};  // s_.xn is rewritten (or left stale) below
  xn_q8t_norm_ = {};
  // An EXL3 prefill MoE output left to this combine: the four-lane kernel
  // below forms it; any other route gets it in s_.block_out first.
  Exl3PrefillMoeRows exl3_moe{};
  bool exl3_moe_fused = false;
  if (exl3_moe_pending_.layer != nullptr) {
    const PendingExl3Moe pending = exl3_moe_pending_;
    exl3_moe_pending_ = {};
    const auto& d = pending.layer->ffn_down_exps;
    const auto scratch = Exl3PrefillView();
    exl3_moe_fused = ablation.settings == nullptr && prefill_phase &&
                     pending.rows == n_tokens && c.hc_count == 4 &&
                     c.hidden_size % 128 == 0 && c.hidden_size <= 2560 &&
                     d.rows == c.hidden_size && !moe_pending_ &&
                     !res_deferred_ &&
                     !(wide_mixer_ && MatrixRows(n_tokens) && gamma != nullptr);
    if (!exl3_moe_fused || exl3_res_pending_.res != res ||
        exl3_res_pending_.rows != n_tokens)
      ApplyPendingResidual();  // the combine below rewrites block_out
    if (exl3_moe_fused) {
      exl3_moe = {d.device_exl3,
                  s_.ids,
                  scratch.inverse,
                  scratch.out,
                  s_.weights,
                  s_.shexp_out,
                  s_.router + c.num_experts,
                  c.num_experts + 1,
                  c.num_experts_used,
                  d.rows};
    } else {
      Exl3PrefillCombine(*pending.layer, scratch, s_.ids, pending.rows,
                         c.num_experts_used, s_.weights, s_.shexp_out,
                         s_.router + c.num_experts, c.num_experts + 1,
                         s_.block_out, stream_);
    }
  }
  // The fused EXL3 MoE combine adds a deferred attention update itself;
  // any other route gets it in res first.
  const DeferredResidual prior =
      exl3_moe_fused ? std::exchange(exl3_res_pending_, {}).update
                     : DeferredResidual{};
  ApplyPendingResidual();
  // Wide batches hand the next mixer an F16 norm for its epilogue and the
  // same norm quantized into the tiled Q8 layout for its W8A8 down
  // projection: half the bytes for the combine and the epilogue, and no
  // separate activation pass for the projection.
  xn_half_ = wide_mixer_ && MatrixRows(n_tokens) && gamma != nullptr;
  xn_q8_ready_ = {};  // s_.xn is rewritten (or left stale) below
  // The next mixer's inject pass reads only the F16 norm written here.
  const float* fused_inject = xn_half_ && n_tokens >= 96 ? inject_w : nullptr;
  const bool defer = defer_residual_;
  defer_residual_ = false;
  bool deferred = res_deferred_;
  res_deferred_ = false;
  if (moe_pending_) {
    // The MoE epilogue was deferred to this combine (see Moe), and with it
    // the previous combine's residual update when that was deferred too.
    moe_pending_ = false;
    const auto* down = reinterpret_cast<const __half*>(s_.down_e);
    const float* fused_w = deferred ? fused_inject : nullptr;
    if (ablation.settings == nullptr && wide_mixer_ && MatrixRows(n_tokens) &&
        HcCombineMoeF16(
            res, down, s_.weights, s_.shexp_out, s_.router + c.num_experts,
            c.num_experts + 1, c.num_experts_used, s_.inject, inject_parts_,
            gamma, s_.xn_half, s_.xn_q8t, n_tokens, c.hidden_size, c.hc_count,
            c.rms_eps, stream_, deferred ? s_.block_out : nullptr,
            deferred ? s_.hc_defer_w : nullptr, s_.inject, fused_w)) {
      inject_fused_ = fused_w;
      return;
    }
    if (deferred) {
      HcApplyDeferred(res, s_.block_out, s_.hc_defer_w, n_tokens, c.hidden_size,
                      stream_);
      deferred = false;  // block_out below becomes MoE output, not attention
    }
    MoeEpilogueVec4F16(down, s_.weights, s_.shexp_out,
                       s_.router + c.num_experts, c.num_experts + 1,
                       s_.block_out, n_tokens, c.num_experts_used,
                       c.hidden_size, stream_);
  }
  if (deferred) {
    // Only a fused MoE combine follows a deferring one; settle the update
    // before anything else reads the residual.
    HcApplyDeferred(res, s_.block_out, s_.hc_defer_w, n_tokens, c.hidden_size,
                    stream_);
  }
  ProjectAbliteration(s_.block_out, n_tokens, ablation);
  if (xn_half_) {
    // The FFN combine that follows rereads this block output (42 MB at
    // 4,096 tokens) instead of this combine writing the residual and the
    // next reading it back (168 MB each).
    const bool defer_now = defer && c.hc_count == 4;
    res_deferred_ =
        HcCombineF16(res, s_.block_out, s_.inject, inject_parts_, gamma,
                     s_.xn_half, s_.xn_q8t, n_tokens, c.hidden_size, c.hc_count,
                     c.rms_eps, stream_, defer_now ? s_.hc_defer_w : nullptr,
                     s_.inject, defer_now ? fused_inject : nullptr);
    if (res_deferred_)
      inject_fused_ = fused_inject;
    return;
  }
  // A decode batch's next mixer quantizes xn for its down projection: the
  // combine writes those rows as well.
  // (Not for EXL3 prefill: every prefill chunk, tails included, must take
  // the same combine below.)
  if (!(prefill_phase && model_->output().exl3 != nullptr) &&
      !MatrixRows(n_tokens) && UnpaddedQ8(c.HcDim()) &&
      HcCombineQ8_1(res, s_.block_out, s_.inject, inject_parts_, gamma, s_.xn,
                    s_.xn_q8_1, n_tokens, c.hidden_size, c.hc_count, c.rms_eps,
                    stream_)) {
    xn_q8_ready_ = {s_.xn, s_.xn_q8_1, n_tokens, c.HcDim()};
    return;
  }
  // EXL3: prefill takes the four-lane combine at every chunk width (28 %
  // less time at 2,048 tokens); decode/verify keep the per-stream kernel.
  const bool exl3 = model_->output().exl3 != nullptr;
  // The next mixer's plain down projection reads this norm as F16: the
  // combine stores that copy too instead of a separate narrowing pass
  // rereading the F32 row.
  const bool half_norm = exl3 && prefill_phase &&
                         n_tokens <= options_.max_batch &&
                         c.HcDim() <= half_input_cols_;
  // Its F32 copy is then left out as well: the mixer recomputes it from the
  // residual and these scales.
  float* const xn_scale =
      half_norm && MatrixRows(n_tokens) ? s_.xn_scale : nullptr;
  // With Q8_0 mixer down weights the down projection is W8A8 and reads the
  // norm as tiled Q8 rows at every prefill width; no F16 copy is needed.
  const bool q8_norm = half_norm && hc_down_q8_;
  // The FFN mixer then recomputes the norm from res (HcMixEpilogueVec4FromRes
  // / HcNormFromRes) and the next combine rereads res: both add this
  // combine's update themselves, so res is not written here (one FP32
  // residual write and reread fewer per layer).
  const bool defer_now = defer_res && exl3 && prefill_phase &&
                         xn_scale != nullptr && !exl3_moe_fused &&
                         c.hc_count == 4 && c.hidden_size % 128 == 0 &&
                         c.hidden_size <= 2560;
  if (HcCombine(
          res, s_.block_out, s_.inject, inject_parts_, gamma, s_.xn, n_tokens,
          c.hidden_size, c.hc_count, c.rms_eps, stream_, exl3 && !prefill_phase,
          exl3 && prefill_phase,
          half_norm && !q8_norm ? static_cast<__half*>(s_.x_half) : nullptr,
          xn_scale, exl3_moe_fused ? &exl3_moe : nullptr,
          q8_norm ? s_.xn_q8t : nullptr, defer_now ? s_.hc_defer_w : nullptr,
          prior)) {
    if (defer_now)
      exl3_res_pending_ = {res, {s_.block_out, s_.hc_defer_w}, n_tokens};
    if (q8_norm) {
      xn_q8t_norm_ = {res, gamma, n_tokens};
      if (half_src_ == s_.xn)
        half_src_ = nullptr;
    } else {
      half_src_ = s_.xn;
      half_rows_ = n_tokens;
      half_cols_ = c.HcDim();
      half_bf16_ = false;
    }
    if (xn_scale != nullptr)
      xn_deferred_ = {res, gamma, n_tokens};
  } else if (half_src_ == s_.xn) {
    half_src_ = nullptr;  // the F32 norm was rewritten without its copy
  }
}

bool Executor::HcMix(const DeviceMixer& m, const float* res, bool normed,
                     float* mixed, float* inject, std::uint32_t n_tokens,
                     std::string* error_msg, bool half_only, bool no_f32,
                     std::span<const DeviceTensor* const> rotate_for) const {
  const Config& c = config();
  mixed_q8_ready_ = {};  // `mixed` is rewritten below
  exl3_rotated_ = {};
  // Without `normed` this mixer's grouped norm of res is computed here (F32);
  // otherwise the previous combine produced it, as F16 plus tiled Q8 on the
  // wide route or as F32 in s_.xn (with its Q8_1 rows on a decode batch).
  // A deferred norm feeds only the vectorized epilogue below (and the F16
  // down projection through the half cache); any other reader gets the F32
  // rows the combine left out, in its own expression.
  // The combine's tiled Q8 norm of this mixer's input, for its W8A8 down.
  const bool q8_down = normed && prefill_phase && !m.down_q8.empty() &&
                       xn_q8t_norm_.res == res &&
                       xn_q8t_norm_.gamma == m.norm.f32() &&
                       xn_q8t_norm_.rows == n_tokens;
  xn_q8t_norm_ = {};
  // A deferred update of exactly this residual is added by the norm readers
  // below; any other pending one is written out first.
  if (exl3_res_pending_.res != nullptr &&
      (!normed || exl3_res_pending_.res != res ||
       exl3_res_pending_.rows != n_tokens))
    ApplyPendingResidual();
  const DeferredResidual prior = exl3_res_pending_.update;
  bool norm_from_res = false;
  if (normed && xn_deferred_.res != nullptr) {
    const DeferredNorm d = xn_deferred_;
    const bool other_inject = inject != nullptr && !m.inject.empty() &&
                              m.inject.type != GgmlType::kF32;
    const bool half_down =
        half_src_ == s_.xn && half_rows_ == n_tokens && half_cols_ == c.HcDim();
    norm_from_res = res == d.res && m.norm.f32() == d.gamma &&
                    n_tokens == d.rows && MatrixRows(n_tokens) &&
                    c.hc_count == 4 && c.hidden_size % 4 == 0 &&
                    !other_inject && (half_down || q8_down);
    if (!norm_from_res)
      HcNormFromRes(d.res, d.gamma, s_.xn_scale, s_.xn, d.rows, c.hidden_size,
                    stream_, d.res == res ? prior : DeferredResidual{});
  }
  const DeferredNorm deferred_norm = xn_deferred_;
  xn_deferred_ = {};
  if (!normed) {
    xn_half_ = false;
    xn_q8_ready_ = {};
    if (half_src_ == s_.xn)
      half_src_ = nullptr;
    RmsNormRows(res, m.norm.f32(), s_.xn, n_tokens, c.HcDim(), c.hc_count,
                c.rms_eps, stream_);
  }
  const bool extras = n_tokens <= options_.max_batch &&
                      c.hidden_size <= model_->max_half_cols();
  const bool fused_projection =
      xn_half_ && n_tokens >= 96 && extras && c.hc_count == 4 &&
      c.hidden_size == 2560 && c.hc_low_rank == 320 &&
      m.up.type == GgmlType::kQ8_0 && m.up.rows == c.HcDim() &&
      m.up.cols == c.hc_low_rank;
  // EXL3 decode/verify reads the Q8_0 copy the prefill W8A8 route uses:
  // 3.4 instead of 6.6 MB per mixer and token step.
  const bool q8_decode_down = !q8_down && model_->output().exl3 != nullptr &&
                              !xn_half_ && !MatrixRows(n_tokens) &&
                              !m.down_q8.empty();
  // EXL3 decode/verify: each down block finishes its own Q8_1 group of lo
  // (SiLU, scale, quantize) for the Q8_0 up projection: one launch fewer,
  // the same arithmetic, no block waiting for another.
  const bool q8_decode_silu =
      q8_decode_down && n_tokens <= kHcDecodeMaxRows && c.hc_count == 4 &&
      c.hc_low_rank == 320 && m.down_q8.type == GgmlType::kQ8_0 &&
      m.down_q8.rows == 320 && m.down_q8.cols == 10240 &&
      m.up.type == GgmlType::kQ8_0 && m.up.cols == c.hc_low_rank;
  const bool fused_down =
      !q8_down && !q8_decode_down && model_->output().exl3 != nullptr &&
      !xn_half_ && !MatrixRows(n_tokens) && m.down.type == GgmlType::kF16 &&
      m.up.type == GgmlType::kF16 && m.down.rows == 320 &&
      m.down.cols == 10240 && c.hc_count == 4 && c.hc_low_rank == 320;
  // EXL3 prefill stores the HC gate as F16 (the Q8_0 up GEMM's F16-store
  // variant), halving its write and the vectorized epilogue's read.
  const bool half_gate =
      !fused_projection && model_->output().exl3 != nullptr && prefill_phase &&
      !xn_half_ && extras && c.hc_count == 4 && c.hidden_size % 4 == 0 &&
      m.up.type == GgmlType::kQ8_0 && m.up.rows % 8 == 0 &&
      DenseF16Route(m.up, n_tokens) && m.up.cols <= half_input_cols_;
  if (fused_projection) {
    // The up projection writes x_half; keep its input in the gate buffer.
    if (!HcDownF16Gemm(m.down.data, s_.xn_q8t,
                       reinterpret_cast<__half*>(s_.hc_gate), n_tokens,
                       stream_)) {
      AssignError(error_msg, "fused HC down projection failed");
      return false;
    }
  } else {
    if (xn_half_ || q8_down) {
      const DeviceTensor& down = q8_down ? m.down_q8 : m.down;
      if (!W8A8Gemm(down.data, s_.xn_q8t, s_.lo, n_tokens, down.rows, down.cols,
                    stream_)) {
        AssignError(error_msg, "W8A8 mixer down projection failed");
        return false;
      }
    } else if (q8_decode_silu) {
      Q8Input q;
      if (!Quantize(s_.xn, n_tokens, m.down_q8.cols, &q, error_msg, true))
        return false;
      if (!HcDownSiluQ8_1(
              m.down_q8.data, q.data, s_.lo,
              1.0F / static_cast<float>(c.hc_count), n_tokens,
              static_cast<std::uint32_t>(
                  qfn_mmq_q8_1_bytes(1, static_cast<int>(c.hc_low_rank)) / 36 *
                  32),
              s_.lo_q8_1, stream_)) {
        AssignError(error_msg, "fused HC down projection failed");
        return false;
      }
    } else if (q8_decode_down) {
      if (!Dense(m.down_q8, s_.xn, s_.lo, n_tokens, error_msg))
        return false;
    } else if (fused_down) {
      HcDownSiluF16(m.down.data, s_.xn, s_.lo, n_tokens, stream_);
    } else if (!Dense(m.down, s_.xn, s_.lo, n_tokens, error_msg)) {
      return false;
    }
    const float scale = 1.0F / static_cast<float>(c.hc_count);
    const std::uint32_t lo_q8_cols = static_cast<std::uint32_t>(
        qfn_mmq_q8_1_bytes(1, static_cast<int>(c.hc_low_rank)) / 36 * 32);
    if (q8_decode_silu ||
        (!MatrixRows(n_tokens) && m.up.type == GgmlType::kQ8_0 &&
         m.up.cols == c.hc_low_rank &&
         SiluScaleQ8_1(s_.lo, scale, n_tokens, c.hc_low_rank, lo_q8_cols,
                       s_.lo_q8_1, stream_))) {
      const Q8Input lo{s_.lo, s_.lo_q8_1, n_tokens, c.hc_low_rank};
      if (!Dense(m.up, lo, s_.hc_gate, error_msg)) {
        return false;
      }
    } else {
      if (!fused_down)
        SiluScale(s_.lo, scale,
                  static_cast<std::size_t>(n_tokens) * c.hc_low_rank, stream_);
      if (half_gate) {
        if (diagnostics_active_)
          cycle_diagnostics_.prefill_plain_rows += n_tokens;
        PrepareHalfInput(s_.lo, n_tokens, m.up.cols);
        if (!HcUpF16Gemm(m.up.data, static_cast<const __half*>(s_.x_half),
                         reinterpret_cast<__half*>(s_.hc_gate), n_tokens,
                         m.up.rows, m.up.cols, stream_)) {
          AssignError(error_msg, "HC up F16 GEMM failed");
          return false;
        }
      } else if (!Dense(m.up, s_.lo, s_.hc_gate, n_tokens, error_msg)) {
        return false;
      }
    }
  }
  const bool fused_inject =
      inject != nullptr && !m.inject.empty() && m.inject.type == GgmlType::kF32;
  // The previous combine already wrote these partials from the same norm.
  const bool inject_done = fused_inject && inject == s_.inject &&
                           inject_fused_ != nullptr &&
                           inject_fused_ == m.inject.f32();
  inject_fused_ = nullptr;
  // A separate inject projection can overwrite the cached half input. Keep
  // the full mixed row available for the following projection in that case.
  half_only =
      half_only && (inject == nullptr || m.inject.empty() || fused_inject);
  const float* xn = s_.xn;
  const bool vectorized =
      xn_half_ ||
      (MatrixRows(n_tokens) && c.hc_count == 4 && c.hidden_size % 4 == 0);
  // `mixed` is being rewritten: whatever the input caches held of it is
  // stale. The wide F16 route also emits the F16 and tiled Q8 copies the
  // projections that follow read.
  q8t_src_ = nullptr;
  half_src_ = nullptr;
  if (xn_half_) {
    if (fused_projection) {
      if (!HcMixF16Gemm(m.up.data, reinterpret_cast<const __half*>(s_.hc_gate),
                        s_.xn_half,
                        fused_inject && !inject_done ? m.inject.f32() : nullptr,
                        half_only || no_f32 ? nullptr : mixed,
                        static_cast<__half*>(s_.x_half),
                        half_only ? nullptr : s_.x_q8t, inject, n_tokens,
                        c.hidden_size, c.hc_low_rank, stream_)) {
        AssignError(error_msg, "fused HC projection failed");
        return false;
      }
    } else {
      HcMixEpilogueVec4F16(s_.xn_half, s_.hc_gate,
                           fused_inject ? m.inject.f32() : nullptr, mixed,
                           extras ? static_cast<__half*>(s_.x_half) : nullptr,
                           extras ? s_.x_q8t : nullptr, inject, n_tokens,
                           c.hidden_size, stream_);
    }
    if (extras) {
      half_src_ = mixed;
      half_rows_ = n_tokens;
      half_cols_ = c.hidden_size;
      half_bf16_ = false;
      if (!half_only || !fused_projection) {
        q8t_src_ = mixed;
        q8t_rows_ = n_tokens;
        q8t_cols_ = c.hidden_size;
      }
    }
  } else if (vectorized) {
    // EXL3 prefill: the plain F16 projections of `mixed` (router, alpha/beta)
    // read these F16 rows instead of a separate narrowing pass.
    auto* mixed_half = model_->output().exl3 != nullptr && prefill_phase &&
                               extras && c.hidden_size <= half_input_cols_
                           ? static_cast<__half*>(s_.x_half)
                           : nullptr;
    bool half_written = mixed_half != nullptr;
    const auto* gate_half =
        half_gate ? reinterpret_cast<const __half*>(s_.hc_gate) : nullptr;
    if (norm_from_res)
      HcMixEpilogueVec4FromRes(
          deferred_norm.res, deferred_norm.gamma, s_.xn_scale, s_.hc_gate,
          fused_inject ? m.inject.f32() : nullptr, mixed, inject, n_tokens,
          c.hidden_size, stream_, mixed_half, gate_half, prior);
    else
      half_written = HcMixEpilogueVec4(
          xn, s_.hc_gate, fused_inject ? m.inject.f32() : nullptr, mixed,
          inject, n_tokens, c.hidden_size, c.hc_count, stream_, mixed_half,
          gate_half);
    if (half_written) {
      half_src_ = mixed;
      half_rows_ = n_tokens;
      half_cols_ = c.hidden_size;
      half_bf16_ = false;
    }
  } else {
    // A decode batch's projections of `mixed` share its Q8_1 rows, which
    // the epilogue writes as it stores them.
    const bool q8 = !MatrixRows(n_tokens) && UnpaddedQ8(c.hidden_size);
    // EXL3 decode: the epilogue also rotates mixed for the projections that
    // read it next (GDN qkv/z, attention q/k/v or the shared MLP), removing
    // their separate rotation launch; the same rounded FP16 values.
    const auto slabs = Exl3Slabs(rotate_for, n_tokens);
    if (HcMixEpilogue(xn, s_.hc_gate, fused_inject ? m.inject.f32() : nullptr,
                      mixed, inject, n_tokens, c.hidden_size, c.hc_count,
                      stream_, q8 ? s_.mixed_q8_1 : nullptr, slabs))
      NoteExl3Rotated(slabs, mixed, n_tokens, rotate_for);
    if (q8) {
      mixed_q8_ready_ = {mixed, s_.mixed_q8_1, n_tokens, c.hidden_size};
    }
  }
  inject_parts_ = fused_inject ? (vectorized ? HcInjectPartsVec4(c.hidden_size)
                                             : HcInjectParts(c.hidden_size))
                               : 1;
  if (inject != nullptr && !m.inject.empty() && !fused_inject) {
    // A quantized inject projection (the draft block's) reads an F32 norm;
    // on the wide route the combine only produced F16, so norm again.
    if (xn_half_) {
      xn_q8_ready_ = {};
      if (half_src_ == s_.xn)
        half_src_ = nullptr;
      ApplyPendingResidual();
      RmsNormRows(res, m.norm.f32(), s_.xn, n_tokens, c.HcDim(), c.hc_count,
                  c.rms_eps, stream_);
    }
    if (!Dense(m.inject, xn, inject, n_tokens, error_msg)) {
      return false;
    }
  }
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
