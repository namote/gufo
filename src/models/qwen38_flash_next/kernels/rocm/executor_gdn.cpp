// Gated DeltaNet (linear attention) layers and their state commit.

#include "src/models/qwen38_flash_next/kernels/rocm/executor_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

bool Executor::LinearAttention(const DeviceLayer& l, Session::LinearState& s,
                               const float* x, float* out,
                               std::uint32_t n_tokens, bool speculative,
                               std::string* error_msg, bool projections_ready,
                               bool project_output,
                               GdnCheckpoint checkpoint) const {
  const Config& c = config();
  const std::uint32_t channels = c.SsmConvChannels();
  const float* qkv = s_.qkv;
  const float* z = s_.z;
  std::uint32_t qkv_stride = channels;
  std::uint32_t z_stride = c.SsmValueDim();
  bool convolved = false;
  hipEvent_t z_ready = nullptr;
  // Until GatedDeltaNet queues its wait for the Z rows, every exit orders
  // the main stream after them: the side stream still writes s_.qkvz.
  struct ZJoin {
    hipStream_t stream;
    hipEvent_t event = nullptr;
    ~ZJoin() {
      if (event != nullptr)
        (void)hipStreamWaitEvent(stream, event, 0);
    }
  } z_join{stream_};
  // EXL3 decode/verify: the alpha/beta projection reads only x, so a side
  // branch runs it beside the qkv+z GEMV; GatedDeltaNet joins it first.
  struct JoinGuard {
    const Executor* executor;
    ~JoinGuard() { (void)executor->JoinSide(nullptr); }
  } join_guard{this};
  const bool side_alpha_beta = !projections_ready && l.ssm_in.empty() &&
                               l.ssm_qkv.exl3 != nullptr &&
                               l.ssm_alpha_beta.exl3 == nullptr &&
                               n_tokens <= kVecBatch && SideBranchAvailable();
  if (side_alpha_beta && !ForkSide(
                             [&] {
                               return Dense(l.ssm_alpha_beta, x, s_.alpha_beta,
                                            n_tokens, error_msg);
                             },
                             error_msg))
    return false;
  if (!l.ssm_in.empty()) {
    // Wide prefill convolves QKV in the projection's LDS tile. The raw
    // boundary rows remain available for the rolling history update.
    if (!projections_ready && !speculative && WidePrefill(n_tokens) &&
        n_tokens <= options_.max_batch && DenseF16Route(l.ssm_in, n_tokens)) {
      PrepareHalfInput(x, n_tokens, l.ssm_in.cols);
      // The Z rows feed only the output norm: the side stream computes them
      // beside the convolution boundary, the alpha/beta GEMM and the
      // recurrence; the epilogue waits for them before it reads z. With the
      // shared-expert overlap on ROCm 10.1: pp512 1,287 -> 1,330 tok/s,
      // pp2048 and pp4096 within 1 %.
      convolved = DenseF16SsmGemm(
          l.ssm_in.data, static_cast<const __half*>(s_.x_half),
          l.ssm_conv1d.f32(), s.conv_state, s_.qkvz, s_.conv_scratch, n_tokens,
          l.ssm_in.rows, l.ssm_in.cols, channels, c.ssm_conv_kernel, stream_,
          checkpoint.tokens, side_stream_, side_start_);
      if (convolved) {
        if (!Check(hipEventRecord(side_done_, side_stream_), "side stream join",
                   error_msg)) {
          (void)hipStreamSynchronize(side_stream_);
          return false;
        }
        z_ready = side_done_;
        z_join.event = side_done_;
      }
    }
    if (!projections_ready && !convolved &&
        !Dense(l.ssm_in, x, s_.qkvz, n_tokens, error_msg)) {
      return false;
    }
    qkv = s_.qkvz;
    z = s_.qkvz + channels;
    qkv_stride = z_stride = l.ssm_in.rows;
  } else if (!projections_ready && l.ssm_qkv.exl3 != nullptr) {
    if (!Exl3Group({&l.ssm_qkv, &l.ssm_gate}, {s_.qkv, s_.z}, x, n_tokens,
                   error_msg))
      return false;
  } else if (!projections_ready) {
    Q8Input xq;
    if (!Quantize(x, n_tokens, c.hidden_size, &xq, error_msg) ||
        !Dense(l.ssm_qkv, xq, s_.qkv, error_msg) ||
        !Dense(l.ssm_gate, xq, s_.z, error_msg)) {
      return false;
    }
  }
  if (!projections_ready && !side_alpha_beta &&
      !Dense(l.ssm_alpha_beta, x, s_.alpha_beta, n_tokens, error_msg)) {
    return false;
  }
  if (side_alpha_beta && !JoinSide(error_msg))
    return false;
  // Keep the same activation precision across prefill chunk boundaries.
  // The F16 epilogue reuses the F32 output allocation. Switching this
  // projection to Q8 for a short tail changes every subsequent layer.
  const bool tiled = MatrixRows(n_tokens) &&
                     l.ssm_out.type == GgmlType::kQ8_0 &&
                     n_tokens <= options_.max_batch;
  const bool half_output =
      tiled && l.ssm_out.rows == 2560 && l.ssm_out.cols == 6144;
  auto* out_half =
      half_output ? reinterpret_cast<__half*>(s_.gdn_out) : nullptr;
  // Shift successive rows off the 12 KiB pitch while retaining every half.
  // The existing F32 scratch has room for these slightly padded F16 rows.
  const std::uint32_t half_stride =
      half_output && n_tokens >= 1024 ? l.ssm_out.cols + 32 : 0;
  // A decode batch's output projection reads the epilogue's Q8_1 rows.
  const std::uint32_t value_dim = c.SsmValueDim();
  const bool out_q8_1 = project_output && !tiled && !MatrixRows(n_tokens) &&
                        c.ssm_head_dim == 128 &&
                        l.ssm_out.type == GgmlType::kQ8_0 &&
                        l.ssm_out.cols == value_dim && UnpaddedQ8(value_dim);
  // EXL3 decode: the epilogue also rotates its rows for ssm_out.
  const DeviceTensor* const out_consumer[] = {&l.ssm_out};
  auto out_slabs = project_output && !tiled ? Exl3Slabs(out_consumer, n_tokens)
                                            : Exl3InputSlabs{};
  if (!GatedDeltaNet(qkv, qkv_stride, z, z_stride, s_.alpha_beta,
                     l.ssm_conv1d.f32(), l.ssm_a.f32(), l.ssm_dt.f32(),
                     l.ssm_norm.f32(), s.conv_state, s_.conv_scratch, s_.qn,
                     s_.kn, s_.gdn_raw, s.state, s_.gdn_out,
                     tiled && !half_output ? s_.x_q8t : nullptr,
                     speculative ? s.state_snapshots : RollbackRows{},
                     speculative ? s.conv_snapshots : RollbackRows{}, n_tokens,
                     c.ssm_num_k_heads, c.ssm_num_v_heads, c.ssm_head_dim,
                     c.ssm_conv_kernel, MatrixRows(n_tokens) && !speculative,
                     convolved, c.rms_eps, stream_, out_half, checkpoint,
                     half_stride, out_q8_1 ? s_.gdn_q8_1 : nullptr, z_ready,
                     model_->output().exl3 != nullptr,
                     out_slabs.count != 0 ? &out_slabs : nullptr)) {
    AssignError(error_msg, "DeltaNet launch failed");
    return false;
  }
  NoteExl3Rotated(out_slabs, s_.gdn_out, n_tokens, out_consumer);
  z_join.event = nullptr;  // GatedDeltaNet waited for the Z rows
  if (!project_output) {
    return true;
  }
  if (half_output) {
    if (!DenseF16Gemm(l.ssm_out.data, out_half, out, n_tokens, l.ssm_out.rows,
                      l.ssm_out.cols, stream_, half_stride)) {
      AssignError(error_msg, "SSM output F16 GEMM failed");
      return false;
    }
    return true;
  }
  if (tiled) {
    q8t_src_ = nullptr;
    if (!W8A8Gemm(l.ssm_out.data, s_.x_q8t, out, n_tokens, l.ssm_out.rows,
                  l.ssm_out.cols, stream_)) {
      AssignError(error_msg, "W8A8 GEMM failed");
      return false;
    }
    return true;
  }
  if (out_q8_1) {
    return Dense(l.ssm_out,
                 Q8Input{s_.gdn_out, s_.gdn_q8_1, n_tokens, value_dim}, out,
                 error_msg);
  }
  return Dense(l.ssm_out, s_.gdn_out, out, n_tokens, error_msg);
}

bool Executor::CommitGdnStates(Session& session, std::uint32_t keep,
                               std::uint32_t n, std::string* error_msg) const {
  if (n <= 1)
    return true;  // a one-token batch updates its state in place
  const Config& c = config();
  for (auto& l : session.linear_) {
    if (l.state == nullptr)
      continue;
    RestoreGdnState(l.state, l.state_snapshots, keep, n, c.ssm_num_k_heads,
                    c.ssm_num_v_heads, stream_);
    if (!Check(hipGetLastError(), "state commit", error_msg))
      return false;
  }
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
