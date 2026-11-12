#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KERNELS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KERNELS_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/core/quant/exl3/gemv.hpp"
#include "src/core/sampling.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#include "src/models/qwen/vision/rope.hpp"
#include "src/models/qwen38_flash_next/rope_scaling.hpp"

/// Model-private HIP launchers for everything outside the quantized GEMM
/// tier. Activations are row-major float32 [tokens][dim] unless noted; every
/// launch is asynchronous on `stream`. Weights referenced here are device
/// pointers in their GGUF encoding.
#include "src/models/qwen38_flash_next/kernels/rocm/exl3_prefill_combine.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
/// Rows [previous span's end_row, end_row) use one session's device
/// settings (qwen::AbliterationSettings::DeviceWords); null leaves them
/// unchanged.
struct AbliterationSpan {
  std::uint32_t end_row;
  const float* settings;
};
/// In place, per row of an ON span: y -= w * sum_j d_j * dot(d_j, y), where
/// d_j are the session's direction_count orthonormal rows of `directions`
/// ([rows][hidden]) and w its strength for `layer`'s attention (`mlp` false)
/// or MLP writer; with preserve_norm the row is then rescaled to its former
/// L2 norm. All of it is read on the device at run time. One direction
/// without norm preservation is the single-direction projection, bit for
/// bit. Launches nothing when every span is OFF; an ON span launches even at
/// strength 0, so settings updates never change a captured graph.
void AbliterateBatchRows(float* output, const float* directions,
                         std::span<const AbliterationSpan> spans,
                         std::uint32_t layer, bool mlp, std::uint32_t layers,
                         std::uint32_t rows, std::uint32_t hidden,
                         hipStream_t stream);
void AbliterateRows(float* output, const float* directions,
                    const float* settings, std::uint32_t layer, bool mlp,
                    std::uint32_t layers, std::uint32_t rows,
                    std::uint32_t hidden, hipStream_t stream);
/// The post-layer residual `res` ([tokens][streams][hidden]): each stream
/// row of an ON span's tokens loses w * sum_j d_j * dot(d_j, row) (and keeps
/// its own norm with preserve_norm), w being the session's `residual`
/// strength for `layer`. Projecting every stream also removes the
/// stream mean's component (what the collector measured): with w = 1 none is
/// left. Span end rows count tokens. Callers pass only sessions whose
/// residual strengths are not all zero (null otherwise).
void AbliterateResidualBatchRows(float* res, const float* directions,
                                 std::span<const AbliterationSpan> spans,
                                 std::uint32_t layer, std::uint32_t layers,
                                 std::uint32_t tokens, std::uint32_t hidden,
                                 std::uint32_t streams, hipStream_t stream);

/// Abliteration fused into the one-block-per-stream HC combine (HcCombine's
/// per-stream route, HcCombineQ8_1): token t uses the settings of its span
/// (spans[0, count), end rows counting tokens; null settings: no projection).
/// The block output row first loses the `writer` site's projection (weight
/// index qwen::AbliterationWeightIndex, AbliterateBatchRows' formula), each
/// updated stream row then the `residual` one (AbliterateResidualBatchRows'
/// formula, per stream) before it is stored and normalized for the next
/// mixer. kNone skips a site; zero strengths skip their projection on the
/// device. Rows at most kHcAbliterationMaxHidden wide.
struct HcAbliteration {
  static constexpr std::uint32_t kNone = ~0U;
  static constexpr std::uint32_t kMaxSpans = 8;
  const float* directions{nullptr};
  AbliterationSpan spans[kMaxSpans]{};
  std::uint32_t count{0};
  std::uint32_t writer{kNone};
  std::uint32_t residual{kNone};
};
inline constexpr std::uint32_t kHcAbliterationMaxHidden = 2560;
/// Whether HcCombine with these arguments takes its one-block-per-stream
/// kernel (the route that can fuse HcAbliteration).
bool HcCombinePerStream(std::uint32_t n_tokens, std::uint32_t hidden,
                        std::uint32_t streams, bool decode, bool any_width);

/// Dynamic-abliteration detector scores (R2): for each of `rows` token rows
/// of the post-layer residual `res` ([tokens][streams][hidden]) and each
/// listed detector d, scores[row * stride + column] = mean over the HC
/// streams of dot(stream row, direction d), which equals dot(stream mean,
/// d): the collector's readout. `columns[i]` is detector `detectors[i]`'s
/// output column; at most kAbliterationMaxProfiles detectors. FP32 sums with
/// a fixed tree reduction (deterministic).
void AbliterationDetectorScores(const float* res, const float* directions,
                                std::span<const std::uint32_t> detectors,
                                std::span<const std::uint32_t> columns,
                                float* scores, std::uint32_t stride,
                                std::uint32_t rows, std::uint32_t hidden,
                                std::uint32_t streams, hipStream_t stream);

struct Exl3RouteTile;  // exl3_moe.hpp

/// A four-lane combine's residual update left out of res (HcCombine
/// defer_w): res[t][s][i] + block_out[t][i] * w[t * 4 + s].
struct DeferredResidual {
  const float* block_out{nullptr};
  const float* w{nullptr};
};

/// FP16 lane-major input slabs of the EXL3 decode projections that follow a
/// producer epilogue (gufo::hip::LaunchExl3FastProjections / LaunchExl3FastMlp
/// with input_rotated): consumer c's slab starts at out + c * 8 * k and holds
/// the producer's rows rotated with suh[c], M = pass_rows
/// (Exl3FastPassRows of the row count). count == 0: no slabs.
struct Exl3InputSlabs {
  const std::uint16_t* suh[gufo::hip::kExl3FastMaxGroup]{};
  std::uint16_t* out{nullptr};
  std::uint32_t count{0};
  std::uint32_t pass_rows{0};
  std::uint32_t k{0};
};

/// Host launch values, copied into kernel arguments before returning. The
/// device descriptor tables may be reused on the same stream after GEMM;
/// no mapped host descriptor is overwritten while the GPU still reads it.
struct Exl3DensePrefillProblem {
  const std::uint16_t* trellis;
  const std::uint16_t* suh;
  const std::uint16_t* svh;
  const float* x;
  std::size_t x_stride;
  std::uint16_t* rotated;
  float* y;
  std::uint32_t rows;
  std::uint32_t trellis_rows{0};
};
void PrepareExl3DensePrefill(std::span<const Exl3DensePrefillProblem> problems,
                             gufo::hip::Exl3RotateProblem* rotations,
                             gufo::hip::Exl3GemmProblem* gemms,
                             hipStream_t stream);

/// Tokens per pooled indexer key row. K/V rows (one per token) and pooled rows
/// share a session's block table (src/core/hip/kv_block_table.hpp): logical
/// block b holds tokens [256 b, 256 b + 256) and pooled rows [64 b, 64 b + 64).
/// Every `kv_blocks` argument below is that table, or null for contiguous
/// rows indexed by position.
inline constexpr std::uint32_t kKvPooledRatio = 4;

/// A `kv_blocks` value for the K/V writers (PrepareAttention,
/// AttentionF16Gemm, StoreKv) of a KVarN session: position p goes to row
/// hip::KvarnFp16Row(p) of the session's FP16 sink and tail ring
/// ([kKvarnFp16Tokens][kv row] per layer) instead of a block-table row.
/// Readers never take it.
inline constexpr std::uintptr_t kKvarnFp16RowTag = 1;
inline const std::uint32_t* KvarnFp16Rows() noexcept {
  return reinterpret_cast<const std::uint32_t*>(kKvarnFp16RowTag);
}

/// Rotates the K and V rows of positions [*start_pos, *start_pos + n_tokens)
/// in a KVarN FP16 sink/tail ring (hip::KvarnFp16Row, rows of kv_heads x 256
/// halves) in place by the normalized 256-point Hadamard, as the KVarN
/// store writes them (src/core/hip/detail/kvarn_device.hpp Fwht256).
/// With `staging_first` other than kKvarnRingRows the rows are a contiguous
/// staging buffer whose row 0 holds that position.
inline constexpr std::uint32_t kKvarnRingRows = ~0U;
void RotateKvRows(__half* k_rows, __half* v_rows, std::uint32_t n_tokens,
                  std::uint32_t kv_heads, const std::uint32_t* start_pos,
                  hipStream_t stream,
                  std::uint32_t staging_first = kKvarnRingRows);

/// Rotates `rows` FP32 rows of 256 in place by the normalized Hadamard: the
/// queries into, and the attention context out of, the KVarN frame.
void RotateHeadRows(float* x, std::uint32_t rows, hipStream_t stream);

/// UnpackQGate's query half, RmsNormRows (gamma), Rope and, with `rotate`,
/// RotateHeadRows of every (token, head) row of 256 in one pass, each in
/// the separate kernel's arithmetic: q[t][h] from the packed [q | gate] head
/// pairs of qg. Non-null `gate` also receives the gate half. Returns false
/// (launching nothing) unless d == 256 and rotary_dim <= 256.
bool AttentionQueryRows(const float* qg, std::uint32_t qg_stride,
                        const float* gamma, float* q, float* gate,
                        std::uint32_t n_tokens, std::uint32_t heads,
                        std::uint32_t d, std::uint32_t rotary_dim,
                        const std::uint32_t* start_pos, float theta, float eps,
                        bool rotate, hipStream_t stream,
                        const qwen::vision::DeviceRope* rope,
                        const YarnRope& yarn);
/// RotateHeadRows of the context rows, then SigmoidMul by the gate half of
/// the packed qg rows (UnpackQGate's layout), in one pass.
/// Non-empty `slabs` (one consumer, k == heads * 256) also receive the
/// gated context rotated for the EXL3 output projection; returns whether
/// they were written.
bool GateRotatedContext(float* ctx, const float* qg, std::uint32_t qg_stride,
                        std::uint32_t n_tokens, std::uint32_t heads,
                        hipStream_t stream, const Exl3InputSlabs& slabs = {});

/// One layer's KVarN K/V as the attention kernels read it
/// (plan/exl3/archive/F1-KVARN-DESIGN.md 4). Position j of a query at position
/// p is
///   * a staging row when `k_staging` is set and j >= staging_first (the rows
///     a prompt chunk writes, and the ring rows before it);
///   * a sink row for j < 128;
///   * a k4v4 token-major record row for j < quant_end, or with quant_end
///     zero for j < 128 * (1 + hip::KvarnReadGroups(p)) (decode and passes);
///   * otherwise a tail ring row (hip::KvarnFp16Row).
/// Rows are in the rotated frame: rotate queries in and the context out.
struct KvarnReadView {
  const __half* k_fp16{nullptr};  ///< [kKvarnFp16Tokens][kv_heads * 256]
  const __half* v_fp16{nullptr};
  const __half* k_staging{nullptr};
  const __half* v_staging{nullptr};
  std::uint32_t staging_first{0};
  const std::uint8_t* records{nullptr};  ///< this layer's record plane
  const std::uint32_t* blocks{nullptr};  ///< block table; null: plane row g
  std::uint32_t quant_end{0};
  std::uint32_t kv_heads{0};
  int bits{4};
};

/// The in-pass group flush of one layer: when the pass of `n_tokens` rows at
/// *start_pos ends a group (hip::PlanKvarnPass boundary), quantizes it from
/// the ring into its record; otherwise does nothing. Reads the position on
/// the device, so a captured decode graph replays it at any position.
void KvarnQuantizePassBoundary(const __half* k_fp16, const __half* v_fp16,
                               std::uint8_t* records,
                               const std::uint32_t* kv_blocks,
                               std::uint32_t kv_heads,
                               const std::uint32_t* start_pos,
                               std::uint32_t n_tokens, hipStream_t stream,
                               int bits = 4);

/// Quantizes groups [first_group, first_group + groups) of `layers` layers
/// from the rotated FP16 tail ring (`fp16_layer_elements` halves per layer
/// and K or V) into KVarN k4v4 records with token-major values
/// (hip::KvarnRecord(4, kTokenMajor)), plane row kv_blocks[g / 2] * 2 + g % 2
/// of `records` (`records_layer_pitch` bytes per layer; group g itself when
/// kv_blocks is null). first_group >= 1; the ring must hold every row.
void KvarnQuantizeGroups(const __half* k_rows, const __half* v_rows,
                         std::size_t fp16_layer_elements, std::uint8_t* records,
                         std::size_t records_layer_pitch,
                         const std::uint32_t* kv_blocks, std::uint32_t kv_heads,
                         std::uint32_t layers, std::uint32_t first_group,
                         std::uint32_t groups, hipStream_t stream,
                         std::uint32_t staging_first = kKvarnRingRows,
                         int bits = 4);

/// Stable per-prefix rollback addresses. Growing the depth does not move
/// existing buffers or invalidate graphs captured for a smaller batch.
struct RollbackRows {
  float* rows[7]{};
};

// Recurrent rollback of an n-token speculative batch (n > 1): the batch
// leaves the live state at its value before the batch and records each
// token's exact FP32 keys/decays/betas/errors (record t in row t, the last
// token's record after row 0's); RestoreGdnState then replays the kept
// prefix in place. No inverse, quantization or full-state snapshot.
constexpr std::size_t GdnRollbackRecordFloats(std::uint32_t k_heads,
                                              std::uint32_t v_heads,
                                              std::uint32_t dim) {
  return std::size_t{k_heads} * dim + v_heads * (dim + 2);
}
constexpr std::size_t GdnRollbackRowFloats(std::uint32_t row,
                                           std::uint32_t k_heads,
                                           std::uint32_t v_heads,
                                           std::uint32_t dim) {
  return (row == 0 ? 2 : 1) * GdnRollbackRecordFloats(k_heads, v_heads, dim);
}
/// Commits `keep` (1..n) tokens of a deferred n-token speculative batch to
/// `state`, the arithmetic of the recurrence (bit for bit its final state
/// for keep == n).
void RestoreGdnState(float* state, RollbackRows snapshots, std::uint32_t keep,
                     std::uint32_t n, std::uint32_t k_heads,
                     std::uint32_t v_heads, hipStream_t stream);

/// GGUF type ids the runtime accepts for the small-matrix and lookup paths.
enum class WeightType : std::uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ5_1 = 7,
  kQ8_0 = 8,
  kQ4_K = 12,
  kQ5_K = 13,
  kBF16 = 30,
};

/// res[t][s][hidden] = table[tokens[t]] for every stream s (Q8_0 rows).
void EmbedTokens(const void* table, WeightType type, const std::int32_t* tokens,
                 float* res, std::uint32_t n_tokens, std::uint32_t hidden,
                 std::uint32_t streams, hipStream_t stream);

/// out[t][d] = rmsnorm(x[t][d]) * gamma[d]; `groups` independent norms of
/// `dim/groups` elements each share the gamma row. gamma may be null.
void RmsNormRows(const float* x, const float* gamma, float* out,
                 std::uint32_t n_rows, std::uint32_t dim, std::uint32_t groups,
                 float eps, hipStream_t stream);

/// Hyper-connection mix epilogue: mixed[t][i] = mean_s xn[t][s][i] *
/// sigmoid(gate[t][s][i]); with `inject_w` ([streams][hc_dim] f32) also
/// inject[t][s] = dot(inject_w[s], xn[t]).
/// Partial sums per inject logit the fused epilogue emits; `inject` holds
/// n_tokens * streams * HcInjectParts(hidden) floats.
std::uint32_t HcInjectParts(std::uint32_t hidden);
std::uint32_t HcInjectPartsVec4(std::uint32_t hidden);
/// A non-null `mixed_q8` (hidden % 32 == 0) also receives mixed as the
/// small-batch projections' Q8_1 rows, byte-identical to
/// qfn_mmq_quantize_q8_1.
/// Non-empty `slabs` (four streams, hidden % 256 == 0, k == hidden) also
/// receive mixed rotated for the next EXL3 decode projections; returns
/// whether they were written (false launches the plain epilogue).
bool HcMixEpilogue(const float* xn, const float* gate, const float* inject_w,
                   float* mixed, float* inject, std::uint32_t n_tokens,
                   std::uint32_t hidden, std::uint32_t streams,
                   hipStream_t stream, void* mixed_q8 = nullptr,
                   const Exl3InputSlabs& slabs = {});
/// Four adjacent hidden lanes per thread. The model's four-stream geometry
/// emits HcInjectPartsVec4(hidden) partial sums per inject logit.
/// A non-null `mixed_half` also receives NarrowActivations' F16 rows of
/// `mixed` (vectorized route only; returns whether they were written). A
/// non-null `gate_half` (HcUpF16Gemm's F16 gate; four streams, hidden % 4
/// == 0 required) replaces `gate`.
bool HcMixEpilogueVec4(const float* xn, const float* gate,
                       const float* inject_w, float* mixed, float* inject,
                       std::uint32_t n_tokens, std::uint32_t hidden,
                       std::uint32_t streams, hipStream_t stream,
                       __half* mixed_half = nullptr,
                       const __half* gate_half = nullptr);
/// The vec4 epilogue over an F16 `xn` (the F16 mixer input route); the
/// four-stream geometry is the caller's contract. Non-null `mixed_half` /
/// `mixed_q8` also receive the mixed rows as F16 and in the W8A8 tiled Q8
/// layout (K = hidden).
void HcMixEpilogueVec4F16(const __half* xn, const float* gate,
                          const float* inject_w, float* mixed,
                          __half* mixed_half, void* mixed_q8, float* inject,
                          std::uint32_t n_tokens, std::uint32_t hidden,
                          hipStream_t stream);

/// res[t][s][i] += block_out[t][i] * 2*sigmoid(inject[t][s] / streams), and
/// when `gamma` is non-null also the next mixer's grouped RMSNorm of the
/// updated residual into `xn`.
/// `inject` is [t][s][inject_parts] partial sums (1 part for a plain GEMM).
/// Concatenated decode requests retain scalar reduction with `decode=true`.
/// xn_half, if given, also receives the norm rounded to F16 (the separate
/// narrowing pass's conversion) where the four-lane kernel runs with a
/// gamma; returns whether it was written. With xn_scale as well, the F32
/// norm is not stored: xn_scale receives each row's four stream scales,
/// from which HcNormFromRes / HcMixEpilogueVec4FromRes recompute it.
/// Four-lane route only: a non-null `defer_w` (with xn_scale, no `moe`)
/// leaves res unchanged and stores each row's four stream weights there
/// instead; the readers (HcMixEpilogueVec4FromRes, HcNormFromRes, the next
/// HcCombine, HcApplyDeferred) reapply it given `prior`, in the combine's
/// own FMA. `prior` adds such a deferred update first.
bool HcCombine(float* res, const float* block_out, const float* inject,
               std::uint32_t inject_parts, const float* gamma, float* xn,
               std::uint32_t n_tokens, std::uint32_t hidden,
               std::uint32_t streams, float eps, hipStream_t stream,
               bool decode = false, bool any_width = false,
               __half* xn_half = nullptr, float* xn_scale = nullptr,
               const Exl3PrefillMoeRows* moe = nullptr, void* xn_q8t = nullptr,
               float* defer_w = nullptr, DeferredResidual prior = {},
               const HcAbliteration* abliteration = nullptr);
/// The F32 norm HcCombine left out (four streams): res * scale * gamma in
/// the combine's own expression.
void HcNormFromRes(const float* res, const float* gamma, const float* xn_scale,
                   float* xn, std::uint32_t n_tokens, std::uint32_t hidden,
                   hipStream_t stream, DeferredResidual prior = {});
/// HcMixEpilogueVec4 reading that norm from res, xn_scale and gamma instead
/// of an F32 xn (four streams, hidden % 4 == 0).
void HcMixEpilogueVec4FromRes(const float* res, const float* gamma,
                              const float* xn_scale, const float* gate,
                              const float* inject_w, float* mixed,
                              float* inject, std::uint32_t n_tokens,
                              std::uint32_t hidden, hipStream_t stream,
                              __half* mixed_half = nullptr,
                              const __half* gate_half = nullptr,
                              DeferredResidual prior = {});
// any_width: the four-lane kernel at every token count (no 16-row
// threshold), so a route fixed by phase never depends on the chunk width.
/// The one-block-per-stream HcCombine that also writes xn's Q8_1 rows (as
/// qfn_mmq_quantize_q8_1 would) to `xn_q8` for a small-batch mixer down
/// projection. False, with nothing launched, without `gamma` or unless
/// hidden % 32 == 0.
/// A non-null `abliteration` with an ON span projects as HcAbliteration
/// describes, here and on HcCombine's per-stream route (HcCombine throws on
/// another route); without one the plain kernel runs.
bool HcCombineQ8_1(float* res, const float* block_out, const float* inject,
                   std::uint32_t inject_parts, const float* gamma, float* xn,
                   void* xn_q8, std::uint32_t n_tokens, std::uint32_t hidden,
                   std::uint32_t streams, float eps, hipStream_t stream,
                   const HcAbliteration* abliteration = nullptr);
/// HcCombine writing the normalized output as F16, the width the next
/// mixer's projection and epilogue consume on the F16 mixer input route.
/// A non-null `xn_q8` also receives the norm quantized into the tiled Q8
/// layout (`Q8TiledBytes(n_tokens, streams * hidden)`) for the W8A8 down
/// projection; hidden must be a multiple of 32. A non-null `defer_w`
/// ([tokens][streams]) defers the residual update: res is left unchanged,
/// the stream weights land in defer_w, and true is returned. The update is
/// then owed by HcCombineMoeF16 (prior_out, prior_w) or HcApplyDeferred.
/// A deferring combine given `inject_w` also writes the next mixer's
/// inject partials to `inject_out` (see HcCombineMoeF16).
bool HcCombineF16(float* res, const float* block_out, const float* inject,
                  std::uint32_t inject_parts, const float* gamma, __half* xn,
                  void* xn_q8, std::uint32_t n_tokens, std::uint32_t hidden,
                  std::uint32_t streams, float eps, hipStream_t stream,
                  float* defer_w = nullptr, float* inject_out = nullptr,
                  const float* inject_w = nullptr);
/// Applies the residual update a deferring HcCombineF16 left out (four
/// streams), with the same arithmetic.
void HcApplyDeferred(float* res, const float* block_out, const float* w,
                     std::uint32_t n_tokens, std::uint32_t hidden,
                     hipStream_t stream);
/// HcCombineF16 with the MoE epilogue fused in: the block output is
/// MoeEpilogueVec4F16's result over `expert_out` ([tokens][used][hidden]
/// F16 rows), `weights`, the gated shared expert; it is formed in
/// registers and never written. Returns false (launching nothing) for a
/// geometry the fused kernel does not cover (four 2,560-wide streams).
/// A null gamma updates only the residual; otherwise xn_q8 is required.
/// A non-null `prior_out` first applies a deferred HcCombineF16 update
/// (its block output and `prior_w` stream weights). A non-null `inject_w`
/// (with gamma and `prior_out`) also writes the next mixer's inject partials
/// to `inject_out`, as HcMixF16Gemm would from the F16 norm; they may
/// replace this combine's own `inject` logits.
bool HcCombineMoeF16(
    float* res, const __half* expert_out, const float* weights,
    const float* shared_out, const float* gate, std::uint32_t gate_stride,
    std::uint32_t used, const float* inject, std::uint32_t inject_parts,
    const float* gamma, __half* xn, void* xn_q8, std::uint32_t n_tokens,
    std::uint32_t hidden, std::uint32_t streams, float eps, hipStream_t stream,
    const float* prior_out = nullptr, const float* prior_w = nullptr,
    float* inject_out = nullptr, const float* inject_w = nullptr);

/// x[i] = silu(x[i] * scale), in place over `count` floats.
void SiluScale(float* x, float scale, std::size_t count, hipStream_t stream);
/// SiluScale over `rows` rows of `k` that also writes x's Q8_1 rows, each
/// `k_padded` wide, to `x_q8` as qfn_mmq_quantize_q8_1 would (the padding
/// blocks are left as they are). False, with nothing launched, unless k and
/// k_padded >= k are multiples of 32.
bool SiluScaleQ8_1(float* x, float scale, std::uint32_t rows, std::uint32_t k,
                   std::uint32_t k_padded, void* x_q8, hipStream_t stream);
/// Rows (decode/verify tokens) HcDownSiluQ8_1 takes.
constexpr std::uint32_t kHcDecodeMaxRows = 8;
/// The decode HC down projection (Q8_0 weights, 320 x 10240) of `rows`
/// Q8_1 input rows into lo, then SiluScaleQ8_1(lo, scale, rows, 320,
/// k_padded, lo_q8) in the same launch: each block owns one 32-row Q8_1
/// group and finishes it in-block. Same arithmetic as
/// qfn_mmq_q8_0_dense_vec_preq plus SiluScaleQ8_1. False on an unsupported
/// call or a launch error.
bool HcDownSiluQ8_1(const void* weights, const void* input, float* lo,
                    float scale, std::uint32_t rows, std::uint32_t k_padded,
                    void* lo_q8, hipStream_t stream);
/// Resolves HcDownSiluQ8_1's kernels before serving, without launching.
hipError_t PrepareHcDecodeKernels();
/// gate[i] = silu(gate[i]) * up[i], in place in `gate`.
void Swiglu(float* gate, const float* up, std::size_t count,
            hipStream_t stream);
/// out[i] = silu(gate[i]) * up[i] as F16 (`count` elements).
void SwigluHalf(const float* gate, const float* up, __half* out,
                std::size_t count, hipStream_t stream);
/// silu(gate) * up over n_rows rows of k elements, written only into the
/// W8A8 tiled Q8 layout `out_q8`; false (nothing launched) unless k % 32 == 0.
bool SwigluQ8Tiled(const float* gate, const float* up, void* out_q8,
                   std::size_t n_rows, std::size_t k, hipStream_t stream);
/// x[i] *= sigmoid(g[i]).
void SigmoidMul(float* x, const float* g, std::size_t count,
                hipStream_t stream);
/// SigmoidMul over rows of `width` (count = rows * width) that also writes
/// x rotated into `slabs` (one consumer, k == width, width % 128 == 0). False,
/// with nothing launched, otherwise.
bool SigmoidMulRotate(float* x, const float* g, std::uint32_t rows,
                      std::uint32_t width, const Exl3InputSlabs& slabs,
                      hipStream_t stream);
/// SigmoidMul that also writes x's Q8_1 rows to `x_q8` as
/// qfn_mmq_quantize_q8_1 would for unpadded rows. False, with nothing
/// launched, unless count is a multiple of 32.
bool SigmoidMulQ8_1(float* x, const float* g, std::size_t count, void* x_q8,
                    hipStream_t stream);

/// out[t][m] = sum_k W[m][k] * x[t][k] for F32/BF16/F16 weights; meant for
/// the narrow projections (routers, alpha/beta, indexer, inject) that the
/// quantized tier does not cover.
/// Converts `count` floats to BF16 (or F16) for a 16-bit hipBLAS GEMM.
void NarrowActivations(const float* x, void* out, bool bf16, std::size_t count,
                       hipStream_t stream);

/// W8A8 route for wide batches over Q8_0 weights: activations quantized per
/// 32-wide block into the tiled fragment layout (`Q8TiledBytes(batch, k)`
/// bytes, padded to the GEMM's 128-token macro tile), then an int8 WMMA
/// GEMM. out is [batch][m]. W8A8Gemm returns false, launching nothing, when
/// k is not a multiple of 32.
std::size_t Q8TiledBytes(std::size_t batch, std::size_t k);
void QuantizeQ8Tiled(const float* x, void* out, std::size_t batch,
                     std::size_t k, hipStream_t stream);
/// F16 values (count % 32 == 0) to ggml Q8_0 blocks: d = amax / 127 as
/// F16, codes round(x * (1 / d)), each step IEEE-rounded like the host's.
void QuantizeHalfQ8_0(const void* half, void* q8, std::size_t count,
                      hipStream_t stream);
bool W8A8Gemm(const void* w, const void* x_tiled, float* out, std::size_t batch,
              std::size_t m, std::size_t k, hipStream_t stream);
/// Native wave64 launch, selected by W8A8Gemm for wide output projections.
/// Resolve the kernel before serving, without executing a projection.
hipError_t PrepareW8A8GemmWave64();
void W8A8GemmWave64(const void* w, const void* x_tiled, float* out,
                    std::size_t batch, std::size_t m, std::size_t k,
                    hipStream_t stream);

/// Q8_0 HC down projection [320,10240], SiLU(x / 4), then F16 output.
/// Matches the separate operators' rounding. Supports at least 96 tokens.
bool HcDownF16Gemm(const void* w, const void* x_tiled, __half* out,
                   std::uint32_t n_tokens, hipStream_t stream);

/// Stacked Q8_0 QKV projection [13312,2560], head normalization and RoPE.
/// Writes Q/gates in F32 and K/V caches in F16, preserving separate rounding.
/// Fixed geometry: 24 query heads, two KV heads, 256 dimensions, 64 rotary.
/// Requires at least 1024 tokens; cache capacity must include position +
/// tokens.
/// With yarn enabled, applies static YaRN (see rope_scaling.hpp).
bool AttentionF16Gemm(const void* weights, const __half* input,
                      const float* q_gamma, const float* k_gamma, float* query,
                      float* gate, __half* keys, __half* values,
                      std::uint32_t n_tokens, const std::uint32_t* position,
                      float theta, float eps, hipStream_t stream,
                      const qwen::vision::DeviceRope* rope = nullptr,
                      const YarnRope& yarn = {},
                      const std::uint32_t* kv_blocks = nullptr);

/// Exact Q8_0 HC up projection and F16-input mixer for the Flash Next
/// 2560-hidden, rank-320 geometry. Returns false below 96 tokens or for other
/// shapes. low_rank must not overlap mixed_half; side outputs are optional.
bool HcMixF16Gemm(const void* up, const __half* low_rank, const __half* xn,
                  const float* inject_w, float* mixed, __half* mixed_half,
                  void* mixed_q8, float* inject, std::uint32_t n_tokens,
                  std::uint32_t hidden, std::uint32_t rank, hipStream_t stream);

/// F16 activation rows [batch][k] and F16 weights [m][k], F32 accumulation.
/// out is [batch][m]. Unsupported shapes launch nothing.
bool UnquantizedF16Gemm(const void* w, const __half* x, float* out,
                        std::size_t batch, std::size_t m, std::size_t k,
                        hipStream_t stream);
/// EXL3 plain prefill: fixed tile and two K16 chains for every row count.
/// FP16 activation rows; F16/BF16/F32 weights rounded to FP16 in LDS. FP32
/// output. K is a multiple of 32. Other model dispatch remains unchanged.
bool PlainPrefillGemm(const void* w, WeightType type, const __half* x,
                      float* out, std::size_t batch, std::size_t m,
                      std::size_t k, hipStream_t stream);
/// DenseF16Gemm over Q8_0 weights storing each FP32 sum as F16 (RNE): the
/// EXL3 prefill HC up projection's gate rows. Requires m % 8 == 0.
bool HcUpF16Gemm(const void* w, const __half* x, __half* gate,
                 std::size_t batch, std::size_t m, std::size_t k,
                 hipStream_t stream);
/// Q8_0 weights dequantized to F16 in LDS and F16 activation rows.
/// input_stride is in half elements; zero means k. Padded rows are supported
/// by the wide SSM output projection (batch >= 1024, m=2560, k=6144).
bool DenseF16Gemm(const void* w, const __half* x, float* out, std::size_t batch,
                  std::size_t m, std::size_t k, hipStream_t stream,
                  std::size_t input_stride = 0);
/// BF16 weight rows [m][k] times BF16 activation rows [batch][k] with one
/// F32 K16 chain per output, in K order for every token: a row's result does
/// not depend on its position in the batch (BlasLt::Gemm). out is [batch][m].
bool DenseBf16Gemm(const void* w, const void* x, float* out, std::size_t batch,
                   std::size_t m, std::size_t k, hipStream_t stream);

/// SSM Q8_0 projection fused with its four-tap convolution. Supports
/// [m=16384,k=2560,channels=10240] and at least 1024 tokens. qkvz retains
/// every Z row plus the QKV rows required for tile boundaries and rolling
/// history; other QKV rows are not written. convolved is [tokens][channels].
/// history is read only; pass convolved=true to GatedDeltaNet to consume
/// this result and update history. Speculative snapshots need all raw QKV
/// rows and must use DenseF16Gemm instead. Unsupported shapes launch nothing.
/// A non-null `z_stream` receives the Z rows' tiles (the same products)
/// after the QKV tiles (ordered through `z_start`), so they run beside the
/// convolution boundary and the work that follows it until a reader of the
/// Z rows waits for them.
bool DenseF16SsmGemm(const void* w, const __half* x, const float* conv_w,
                     const float* history, float* qkvz, float* convolved,
                     std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
                     std::uint32_t channels, std::uint32_t kernel,
                     hipStream_t stream, std::uint32_t checkpoint_tokens = 0,
                     hipStream_t z_stream = nullptr,
                     hipEvent_t z_start = nullptr);

/// Routed expert GEMMs. RoutedCompact sorts the (token, slot) assignments
/// by expert into `rows_token`/`rows_slot` (RoutedCompactRows(slots,
/// experts) entries, -1 for padding) with every bucket padded to 16 rows
/// (`pad_bounds`, experts + 1 entries) from the per-expert `counts`.
std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts);
void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_token, std::int32_t* rows_slot,
                   std::uint32_t n_tokens, std::uint32_t k,
                   std::uint32_t n_experts, hipStream_t stream);

/// Routed expert GEMM on the F16 WMMA cores over F16 activation rows
/// `x` ([rows][k]): the weights are dequantized to F16 after the LDS read
/// and the whole K extent accumulates in F32. `tiles` (n_tiles entries)
/// packs each launched (expert, 48-row token tile) as expert | tile << 16,
/// so empty tiles cost nothing. Exactly one of `out` (F32) and `out_half`
/// (F16) receives the result; a non-null `swiglu_gate` (the gate
/// projection's F32 output over the same rows) turns it into
/// silu(gate) * result.
/// `tile_rows` is the token rows per tile the map was built with: 16 or 48,
/// or 64 for Q5_1/Q8_0.
/// Q4_K and Q5_K need k % 256 == 0, Q5_1 and Q8_0 k % 64 == 0; other types
/// and tile widths return false.
bool RoutedF16Gemm(const void* w, WeightType type, const __half* x,
                   const std::int32_t* tiles, std::uint32_t n_tiles,
                   std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                   const std::int32_t* rows_in, const std::int32_t* rows_out,
                   const float* swiglu_gate, float* out, __half* out_half,
                   std::size_t m, std::size_t k, hipStream_t stream);

/// Paired Q4_K/Q5_K gate/up GEMM with SwiGLU and F16 output. Uses the same
/// routing layout as RoutedF16Gemm, with 128 token rows per tile. The two
/// projections share one launch and never materialize the gate output.
bool RoutedGatedF16Gemm(const void* gate, const void* up, WeightType type,
                        const __half* x, const std::int32_t* tiles,
                        std::uint32_t n_tiles, const std::int32_t* pad_bounds,
                        const std::int32_t* rows_in,
                        const std::int32_t* rows_out, __half* out,
                        std::size_t m, std::size_t k, hipStream_t stream);

/// Exact F16 320 x 10240 HC down projection followed by SiLU(x / 4).
/// Uses SmallGemmStable's explicit FMA chain and F32 boundary at every width.
void HcDownSiluF16(const void* w, const float* x, float* out,
                   std::uint32_t n_tokens, hipStream_t stream);

/// F32/BF16/F16 projection in groups of up to eight token rows. Each token
/// keeps the same accumulation order at every batch width.
void SmallGemm(const void* w, WeightType type, const float* x, float* out,
               std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
               hipStream_t stream);

/// EXL3 plain operands: pinned widened FP32 FMAs, also during prefill.
void SmallGemmStable(const void* w, WeightType type, const float* x, float* out,
                     std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
                     hipStream_t stream);

/// PLE gate: for every stream s, sc = dot(key_n[t][s], query_n[t][s]) /
/// sqrt(hidden); gated[t][s][i] = value[t][i] * sigmoid(sgn(sc) *
/// sqrt(max(|sc|, 1e-6))).
void PleGate(const float* key_n, const float* query_n, const float* value,
             float* gated, std::uint32_t n_tokens, std::uint32_t hidden,
             std::uint32_t streams, hipStream_t stream);

/// Dilated depthwise causal conv over tokens with SiLU:
/// out[t][c] = silu(sum_k w[c*kernel+k] * in[t - (kernel-1-k)*dilation][c]),
/// where tokens before the chunk come from `history` ([hist][channels],
/// oldest first). `history` is then advanced past the chunk.
/// `snapshots`, when populated, receives the history as it stands after
/// each proper prefix: [n_tokens-1][hist][channels]. The full batch's
/// history remains in `history`.
void PleConv(const float* in, const float* w, float* history,
             float* history_scratch, float* out, RollbackRows snapshots,
             std::uint32_t n_tokens, std::uint32_t channels,
             std::uint32_t kernel, std::uint32_t dilation, hipStream_t stream);

/// res[t][c] += gated[t][c] + conv[t][c] over hc_dim channels.
void PleInject(float* res, const float* gated, const float* conv,
               std::size_t count, hipStream_t stream);

/// Copies rolling history at a prefix without advancing the live history.
void HistoryPrefix(const float* in, std::uint32_t stride, const float* history,
                   float* destination, std::uint32_t tokens,
                   std::uint32_t channels, std::uint32_t history_rows,
                   hipStream_t stream);

struct GdnCheckpoint {
  float* state = nullptr;
  float* history = nullptr;
  std::uint32_t tokens = 0;
};

/// Gated DeltaNet over a chunk of tokens for one layer. Runs the causal
/// conv (with rolling `conv_state`, [kernel-1][channels]) and the recurrence
/// on `state` ([v_heads][d][d]) sequentially over tokens, parallel over heads
/// and value dims. `alpha_beta` holds both projections per token:
/// [t][alpha(v_heads) | beta(v_heads)]. Outputs the normalized,
/// sigmoid-gated attention rows [tokens][v_heads*d] ready for the output
/// projection.
/// `conv_snapshots` ([n_tokens-1][(kernel-1)*channels]), when populated,
/// receive the convolution history after each proper prefix; the live
/// history holds the full batch. Populated `state_snapshots` (rows of
/// GdnRollbackRowFloats, n_tokens > 1) receive every token's update record
/// instead, and the live recurrent state keeps its value before the batch
/// until RestoreGdnState commits the kept prefix.
/// Scratch: `conv_scratch` ((n_tokens + kernel) * channels), `qn`/`kn`
/// (n_tokens * k_heads * d), `raw` (n_tokens * v_heads * d).
/// `qkv` rows are `qkv_stride` floats apart and `z` rows `z_stride`, so a
/// stacked [qkv|z] projection feeds both without unpacking. A non-null
/// `out_q8` receives the rows quantized into the W8A8 tiled layout
/// (K = v_heads * d) instead of `out`. With null `out_q8`, a non-null
/// `out_half` receives F16 rows instead of `out`. With both null, a non-null
/// `out_q8_1` also receives the rows' unpadded Q8_1 rows (as
/// qfn_mmq_quantize_q8_1 writes them; d == 128). The epilogue, the reader
/// of `z`, waits for `z_ready` when it is set. Returns false when that wait
/// cannot be queued. `convolved`
/// consumes convolution rows already in conv_scratch; it requires null
/// speculative snapshots. A non-null `out_slabs` with one consumer (F32
/// `out` rows, d == 128, k == v_heads * d) also receives the rows rotated
/// for the EXL3 output projection; its count is zeroed when it does not.
bool GatedDeltaNet(const float* qkv, std::uint32_t qkv_stride, const float* z,
                   std::uint32_t z_stride, const float* alpha_beta,
                   const float* conv_w, const float* a, const float* dt,
                   const float* norm_w, float* conv_state, float* conv_scratch,
                   float* qn, float* kn, float* raw, float* state, float* out,
                   void* out_q8, RollbackRows state_snapshots,
                   RollbackRows conv_snapshots, std::uint32_t n_tokens,
                   std::uint32_t k_heads, std::uint32_t v_heads,
                   std::uint32_t d, std::uint32_t kernel, bool row_split,
                   bool convolved, float eps, hipStream_t stream,
                   __half* out_half = nullptr, GdnCheckpoint checkpoint = {},
                   std::uint32_t out_half_stride = 0, void* out_q8_1 = nullptr,
                   hipEvent_t z_ready = nullptr, bool stable_conv = false,
                   Exl3InputSlabs* out_slabs = nullptr);

/// Private rows for one request in a decode batch. Scratch regions and all
/// recurrent/history/rollback buffers must be disjoint between requests.
struct GdnBatchItem {
  const float* qkv;
  const float* z;
  const float* alpha_beta;
  float* conv_state;
  float* conv_scratch;
  float* qn;
  float* kn;
  float* raw;
  float* state;
  float* out;
  RollbackRows state_snapshots;
  RollbackRows conv_snapshots;
  std::uint32_t n_tokens;
};

/// Runs the decode arithmetic for 1–8 independent requests of 1–8 rows,
/// 128-wide heads and four convolution taps. `items` is device-accessible;
/// inactive requests are untouched. The caller zeros their output rows.
bool GatedDeltaNetBatch(const GdnBatchItem* items, std::uint32_t count,
                        std::uint32_t max_tokens, std::uint32_t active,
                        std::uint32_t qkv_stride, std::uint32_t z_stride,
                        const float* conv_w, const float* a, const float* dt,
                        const float* norm_w, std::uint32_t k_heads,
                        std::uint32_t v_heads, float eps, hipStream_t stream,
                        bool stable_conv = false);

/// Splits the interleaved [q|gate] projection (rows `qg_stride` apart) into
/// q [t][heads][d] and gate [t][heads*d]. With non-null `k`, the row
/// continues with k and v (`kv_width` each), copied out contiguously.
void UnpackQGate(const float* qg, std::uint32_t qg_stride, float* q,
                 float* gate, float* k, float* v, std::uint32_t n_tokens,
                 std::uint32_t heads, std::uint32_t d, std::uint32_t kv_width,
                 hipStream_t stream);

/// Unpacks a stacked Q/gate/K/V projection, normalizes and rotates Q/K,
/// and writes the F16 caches. Returns false for heads wider than 256.
/// With zero query heads, packed contains only K/V and q/gate may be null.
/// With yarn enabled, applies static YaRN (see rope_scaling.hpp).
/// Non-null `k_src` and `v_src` ([t][kv_heads * d] rows, both or neither)
/// supply K/V from split projections; packed then holds only [q|gate].
bool PrepareAttention(
    const float* packed, std::uint32_t stride, const float* q_gamma,
    const float* k_gamma, float* q, float* gate, __half* k_cache,
    __half* v_cache, std::uint32_t n_tokens, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t d, std::uint32_t rotary_dim,
    const std::uint32_t* start_pos, float theta, float eps, hipStream_t stream,
    const qwen::vision::DeviceRope* rope = nullptr, bool prefill = false,
    const YarnRope& yarn = {}, const std::uint32_t* kv_blocks = nullptr,
    const float* k_src = nullptr, const float* v_src = nullptr);

/// Separate K/V catch-up, with exactly the unfused attention path's F32
/// normalization and RoPE boundaries. K is normalized/rotated in place;
/// V is read-only. No query, gate or attention output is computed.
void PrepareAttentionKv(float* k, const float* v, const float* k_gamma,
                        __half* k_cache, __half* v_cache,
                        std::uint32_t n_tokens, std::uint32_t kv_heads,
                        std::uint32_t d, std::uint32_t rotary_dim,
                        const std::uint32_t* start_pos, float theta, float eps,
                        hipStream_t stream,
                        const qwen::vision::DeviceRope* rope = nullptr,
                        const YarnRope& yarn = {},
                        const std::uint32_t* kv_blocks = nullptr);

/// NEOX partial rotary on x [t][heads][d] at positions start_pos + t.
/// Positions are read from device memory (`start_pos` points at the
/// session's control block) so a captured decode graph replays at any
/// position.
/// With yarn enabled, applies static YaRN (see rope_scaling.hpp).
void Rope(float* x, std::uint32_t n_tokens, std::uint32_t heads,
          std::uint32_t d, std::uint32_t rotary_dim,
          const std::uint32_t* start_pos, float theta, hipStream_t stream,
          const qwen::vision::DeviceRope* rope = nullptr,
          const YarnRope& yarn = {});

/// Stores f32 rows into the f16 cache at positions start_pos + t:
/// cache[(start_pos + t)][row_dim].
void StoreKv(const float* src, __half* cache, std::uint32_t n_tokens,
             std::uint32_t row_dim, const std::uint32_t* start_pos,
             hipStream_t stream, const std::uint32_t* kv_blocks = nullptr);
/// Stores raw indexer rows in a power-of-two ring of `capacity` rows.
void StoreRows(const float* src, float* dst, std::uint32_t n_tokens,
               std::uint32_t row_dim, const std::uint32_t* start_pos,
               std::uint32_t capacity, hipStream_t stream);

/// Pools raw indexer keys ([pos % capacity][dim] f32) of the blocks the batch
/// completes, [*first_block, (*start_pos + n_tokens) / ratio), into block
/// keys: mean over `ratio` positions, RMSNorm with gamma, rotary at the
/// block start position, then round once to F16 for scoring.
/// `grid_blocks` bounds how many blocks one launch
/// can pool (a device-side bound is not available at launch time).
/// With yarn enabled, applies static YaRN (see rope_scaling.hpp).
void PoolIndexerBlocks(const float* raw_keys, const float* gamma,
                       __half* blocks, const std::uint32_t* first_block,
                       const std::uint32_t* start_pos, std::uint32_t n_tokens,
                       std::uint32_t grid_blocks, std::uint32_t ratio,
                       std::uint32_t dim, std::uint32_t rotary_dim, float theta,
                       float eps, std::uint32_t capacity, hipStream_t stream,
                       const qwen::vision::DeviceRope* rope = nullptr,
                       const YarnRope& yarn = {},
                       const std::uint32_t* kv_blocks = nullptr);

/// Per query t (position *start_pos + first_token + t): scores every
/// complete block below its own tail, keeps the `budget` highest, and
/// writes a visibility bitmap (`mask_words` uint32 per query, bit b = block
/// b visible). Every block is visible when the count fits the budget.
/// Queries retain F32 precision; pooled cache keys are F16. Scores
/// still accumulate in FP32. `scores` holds n_tokens * max_blocks floats;
/// blocks past max_blocks are neither scored nor selected, whatever the
/// device position (`mask_words` * 32 must cover max_blocks).
/// `live_blocks`, when nonzero, bounds the complete blocks of the last
/// query (eager launches only: a captured graph must cover max_blocks).
void SelectBlocks(const float* q, const __half* blocks, std::uint32_t* mask,
                  float* scores, std::uint32_t n_tokens,
                  const std::uint32_t* start_pos, std::uint32_t first_token,
                  std::uint32_t heads, std::uint32_t dim, std::uint32_t ratio,
                  std::uint32_t budget, std::uint32_t mask_words,
                  std::uint32_t max_blocks, hipStream_t stream,
                  std::uint32_t live_blocks = 0,
                  const std::uint32_t* kv_blocks = nullptr);

/// Per-token attention (decode and narrow batches). With `partials`
/// (n_tokens * heads * splits * (d + 2) floats) the key tiles are split
/// across `splits` blocks per row and merged in a second launch.
/// With `lists` (n_tokens * list_capacity entries), `counts` (n_tokens) and a
/// mask at ratio 4, each row's selected blocks are first compacted once into
/// `lists`; `list_capacity` is the selection budget (blocks per row). The
/// output is bitwise identical to the mask walk. Null lists use the mask walk.
/// Complete blocks past a row's `mask_words` * 32 bits are not selected,
/// whatever the device position; keys up to the position must exist.
/// Nonzero dense_prefix forces queries at earlier absolute positions through
/// the dense sweep even if this chunk also contains sparse queries (EXL3).
void Attention(const float* q, const __half* k_cache, const __half* v_cache,
               const std::uint32_t* mask, std::uint32_t mask_words, float* out,
               float* partials, std::uint32_t splits, std::uint32_t n_tokens,
               const std::uint32_t* start_pos, std::uint32_t heads,
               std::uint32_t kv_heads, std::uint32_t d, std::uint32_t ratio,
               hipStream_t stream, std::uint32_t* lists = nullptr,
               std::uint32_t* counts = nullptr, std::uint32_t list_capacity = 0,
               const std::uint32_t* kv_blocks = nullptr,
               const KvarnReadView* kvarn = nullptr,
               std::uint32_t dense_prefix = 0);

/// Maximum visible sparse context for this WMMA geometry, or zero when the
/// geometry is unsupported. Checkpoint planning uses the same dispatch limit.
[[nodiscard]] std::uint32_t WmmaCausalAttentionContextLimit(
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t d,
    std::uint32_t ratio);

/// Fused causal attention on the WMMA cores for wide batches: scores, online
/// softmax, PV and the sigmoid output gate in one launch. `mask` follows
/// Attention's block-selection contract (null for the dense window). Returns
/// false, launching nothing, when the geometry is not the model's 24 x 256
/// heads over two KV heads. `last_only` computes only the final dense query
/// tile, retaining its key sweep and leaving earlier output rows untouched.
/// Returns false when the visible extent exceeds 262,144 tokens; the caller
/// falls back to Attention. `gate_stride` 0: gate is [token][heads * d];
/// otherwise gate points at the first gate half of packed [q | gate] head
/// pairs with that token stride (the projection rows, no unpacked copy).
/// With a mask, each tile packs four consecutive queries from the first row
/// and gathers the union of their selections, so a row's rounding depends on
/// which queries share its tile. A prompt split therefore matches one pass
/// only when every pass after the first starts on a multiple of four.
/// Session::Feed keeps its own passes there, and the serving runner rounds
/// pass ends and checkpoints to TextRunnerCapabilities::prefill_alignment
/// (asserted in TextRunnerPool::Request::Prefill).
bool WmmaCausalAttention(
    const float* q, const float* gate, const __half* k_cache,
    const __half* v_cache, const std::uint32_t* mask, std::uint32_t mask_words,
    float* out, std::uint32_t n_tokens, std::uint32_t start_pos,
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t d,
    std::uint32_t ratio, hipStream_t stream, bool last_only = false,
    const std::uint32_t* kv_blocks = nullptr,
    const KvarnReadView* kvarn = nullptr, std::uint32_t gate_stride = 0);

/// counts[e] = number of (token, slot) pairs routed to expert e.
void ExpertCounts(const std::int32_t* ids, std::uint32_t* counts,
                  std::uint32_t n_tokens, std::uint32_t n_experts,
                  std::uint32_t k, hipStream_t stream);

/// Softmax over `n_experts` logits per token (rows `stride` apart), top-k
/// selection, renormalized weights. ids [t][k] int32, weights [t][k] f32.
/// A row whose softmax is NaN (any NaN or +Inf logit, or all -Inf) takes
/// the lowest unused experts at weight 0, so every ID is below n_experts
/// when k <= n_experts.
void RouterTopK(const float* logits, std::uint32_t stride, std::int32_t* ids,
                float* weights, std::uint32_t n_tokens, std::uint32_t n_experts,
                std::uint32_t k, hipStream_t stream);
/// RouterTopK, then the routed decode's expert tiles for all n_tokens * k
/// IDs (Exl3RoutedMlp's BuildTiles grouping: assignment order per expert,
/// tile_rows per tile) by the last block to finish, with the tile count
/// stored rather than accumulated. `arrivals` is zero-initialized and
/// returns to zero. Requires n_tokens * k <= kExl3FusedTileAssignments.
void RouterTopKTiles(const float* logits, std::uint32_t stride,
                     std::int32_t* ids, float* weights, std::uint32_t n_tokens,
                     std::uint32_t n_experts, std::uint32_t k,
                     Exl3RouteTile* tiles, std::uint32_t* tile_count,
                     std::uint32_t* arrivals, std::uint32_t tile_rows,
                     hipStream_t stream);

/// out[t][i] = sum_s weights[t][s] * expert_out[t*k + s][i]
///           + sigmoid(gate[t * gate_stride]) * shared[t][i].
void MoeEpilogue(const float* expert_out, const float* weights,
                 const float* shared, const float* gate,
                 std::uint32_t gate_stride, float* out, std::uint32_t n_tokens,
                 std::uint32_t k, std::uint32_t dim, hipStream_t stream);
/// Same contract, four adjacent lanes per thread (`dim % 4 == 0`, else the
/// scalar kernel runs).
void MoeEpilogueVec4(const float* expert_out, const float* weights,
                     const float* shared, const float* gate,
                     std::uint32_t gate_stride, float* out,
                     std::uint32_t n_tokens, std::uint32_t k, std::uint32_t dim,
                     hipStream_t stream);
/// MoeEpilogueVec4 over F16 expert rows.
void MoeEpilogueVec4F16(const __half* expert_out, const float* weights,
                        const float* shared, const float* gate,
                        std::uint32_t gate_stride, float* out,
                        std::uint32_t n_tokens, std::uint32_t k,
                        std::uint32_t dim, hipStream_t stream);

/// dst[t] = *row < 0 ? alt[t] : base[*row + t] (rows of `width` floats).
void MtpHidden(const float* base, const float* alt, const std::int32_t* row,
               float* dst, std::uint32_t n_tokens, std::uint32_t width,
               hipStream_t stream);

/// Add the projected embedding once to each projected hidden branch.
void MtpAddEmbedding(const float* embedding, float* residual,
                     std::uint32_t n_tokens, std::uint32_t hidden,
                     std::uint32_t streams, hipStream_t stream);

/// One teacher-forced row's raw log-softmax at its target token.
struct ScoredLogprob {
  double logprob;
  /// The row holds NaN or +inf, has no finite value, or the target is out of
  /// range (sampling::CollectLogprobs refuses such rows); logprob is then 0.
  std::uint32_t nonfinite;
};
/// out[r] = logits[r][targets[r]] - max_r - log(sum exp(logits[r] - max_r)):
/// sampling::CollectLogprobs' selected value with the sum of FP32
/// exponentials accumulated in FP64. Expected difference from that FP64
/// reference: below 1e-5 (exponential rounding, about 1e-7 relative per
/// term, plus FP32 rounding of x - max for terms that contribute).
void ScoreLogprobs(const float* logits, const std::int32_t* targets,
                   ScoredLogprob* out, std::uint32_t rows, std::uint32_t vocab,
                   hipStream_t stream);

inline constexpr std::uint32_t kArgmaxParts = 64;
struct ArgmaxCandidate {
  float value;
  std::int32_t index;
};
/// Full-vocabulary argmax into out[t], with the lowest index winning ties.
/// Matches std::max_element, including NaNs. Scratch holds
/// n_tokens * kArgmaxParts candidates.
void Argmax(const float* logits, ArgmaxCandidate* scratch, std::int32_t* out,
            std::uint32_t n_tokens, std::uint32_t vocab, hipStream_t stream);

/// Gather selected IDs and their original logits from independent rows.
/// The caller checks finite values only for the verification prefix it visits.
// Sorted sparse histories, one per verification row. Penalty arithmetic is
// FP64, matching SamplerState; output values remain the original logits.
struct GreedyPenaltyRows {
  const sampling::TokenPenalty* penalties;
  std::uint32_t offsets[8]{};
  /// The Han offset of each row (SamplerState::han_bias of that row's
  /// tentative state, so a row after a drafted reasoning end gets the
  /// answer offset), added in FP64 after the penalties to every token of
  /// this device set (sampling::TokenSet layout); a null set skips it.
  const std::uint32_t* han_tokens{nullptr};
  double han_bias[7]{};
};
struct PenaltyArgmaxCandidate {
  double value;
  std::int32_t index;
};
void PenalizedArgmax(const float* logits, GreedyPenaltyRows penalties,
                     float repeat, float frequency, float presence,
                     PenaltyArgmaxCandidate* partial, ArgmaxCandidate* out,
                     std::uint32_t rows, std::uint32_t vocab,
                     hipStream_t stream);

void GatherArgmaxCandidates(const float* logits, const std::uint32_t* ids,
                            ArgmaxCandidate* out, std::uint32_t rows,
                            std::uint32_t vocab, hipStream_t stream);

/// Exact top-64 unpenalized tokens plus every penalized token, retaining
/// original F32 logits. This union contains the target's complete top-k for
/// k <= 64, even when penalties promote tokens. The top 64 rank the Han
/// offset, so lowered Han tokens cannot crowd out the target's top-k either.
/// That key is F32 and can reorder near-equal keys, so an active Han scale
/// compacts only k <= 63 (CanCompactMtpVerification). The CPU keeps its
/// existing FP64 penalties, filtering, normalization and rejection sampler.
/// Each ID workspace holds rows * MtpCandidateWorkspaceSize(vocab) entries.
/// Output row r starts at r * 64 + penalties.offsets[r]; invalid IDs have
/// -infinity logits.
void VerificationCandidates(const float* logits, GreedyPenaltyRows penalties,
                            std::uint32_t* ids, std::uint32_t* scratch_ids,
                            sampling::LogitCandidate* out, std::uint32_t rows,
                            std::uint32_t vocab, hipStream_t stream);

/// Draft logits of empty table slots (table[i] < 0) become -infinity.
void MaskDraftLogits(float* logits, const std::int32_t* table, std::uint32_t n,
                     hipStream_t stream);

/// Full-vocabulary logits to the small-head layout, in place: row
/// prefix + i becomes logits[table[i]], or -infinity for an empty slot
/// (table[i] < 0). n is at most 1024 (one block).
void GatherDraftLogits(float* logits, const std::int32_t* table,
                       std::uint32_t prefix, std::uint32_t n,
                       hipStream_t stream);

/// Draft head rows back to token IDs: ids[i] >= prefix (a row of the
/// table_rows-row table) becomes table[ids[i] - prefix]; prefix rows are
/// already token IDs. A row past the table becomes UINT32_MAX, as an empty
/// slot (-1) does.
void RemapDraftIds(std::uint32_t* ids, std::uint32_t n, std::uint32_t prefix,
                   const std::int32_t* table, std::uint32_t table_rows,
                   hipStream_t stream);

/// Number of IDs in each of the two selection buffers.
std::uint32_t MtpCandidateWorkspaceSize(std::uint32_t vocab);

/// Exact MTP top-64 selection, descending score with lowest-token-ID ties.
/// Nonfinite scores become -infinity. Both ID buffers have room for
/// MtpCandidateWorkspaceSize(vocab); scores receives min(vocab, 64) values.
/// Scores may reuse the ID workspace starting at offset 64.
/// All candidate operations are graph-safe.
void MtpTopCandidates(const float* logits, std::uint32_t* ids,
                      std::uint32_t* scratch_ids, float* scores,
                      std::uint32_t vocab, hipStream_t stream);

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KERNELS_HPP_
