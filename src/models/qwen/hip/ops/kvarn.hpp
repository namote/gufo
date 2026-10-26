#ifndef GUFO_MODELS_QWEN_HIP_OPS_KVARN_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_KVARN_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/models/qwen/hip/kvarn_layout.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Device view of one sequence's KVarN attention KV (kvarn_layout.hpp). The
/// FP16 plane holds K and V rows, already rotated, as
/// [layer][kKvarnFp16Tokens][kv head][head_dim] (KvarnFp16Row); the record
/// plane is the pool's, its groups found through the block table.
struct KvarnKvView {
  int value_bits{0};
  std::uint8_t* records{nullptr};
  std::size_t records_layer_pitch{0};
  const std::uint32_t* blocks{nullptr};
  void* k_rows{nullptr};
  void* v_rows{nullptr};
  /// One zeroed arrival counter per KV head: a decode step's last block
  /// combines its splits (and leaves the counter zeroed).
  unsigned* counters{nullptr};
};

/// Rotates the K and V rows of `rows` consecutive positions ([row][kv
/// head][256] floats; K already normed and roped) by the normalized
/// Hadamard and stores them as FP16 in the sink or tail ring of `layer`.
/// Positions start at `*d_pos` when it is given (graph capture), else at
/// `start_pos`. Each row is rounded to FP16 before the rotation, as the
/// FP16 cache stores it, and the rotated row once more.
void LaunchKvarnWriteRows(const float* k, const float* v, const KvarnKvView& kv,
                          std::uint32_t layer, const std::uint32_t* d_pos,
                          std::uint32_t start_pos, std::uint32_t rows,
                          hipStream_t stream);

/// Quantizes groups [first_group, first_group + groups) of layers
/// [first_layer, first_layer + layers) from the FP16 tail ring into the
/// record plane. first_group >= 1 (group 0 is the sink) and the ring must
/// still hold every row of those groups.
void LaunchKvarnFlush(const KvarnKvView& kv, std::uint32_t layers,
                      std::uint32_t first_group, std::uint32_t groups,
                      hipStream_t stream, std::uint32_t first_layer = 0);

/// Causal attention of `rows` (1 to kKvarnMaxRows) consecutive query rows
/// ([row][24][256] floats, roped) over the sink, `groups` quantized groups
/// after it and the FP16 tail to the last row's position, gated like the
/// FP16 decode path (out *= sigmoid(gate) when gate is non-null). Positions
/// start at `*d_pos` when it is given, else at `start_pos`; the first query
/// position must not precede the end of the quantized groups. `start_pos`
/// sets the FP16 tail blocks also with `d_pos` (a captured graph runs
/// before the first group's end, where there are none). `splits` zero
/// selects SelectKvarnDecodeSplits. The scratch must hold
/// KvarnScratchElements(rows, splits).
void LaunchKvarnDecodeAttention(const float* q, const float* gate,
                                const KvarnKvView& kv, float* out_context,
                                std::uint32_t layer, const std::uint32_t* d_pos,
                                std::uint32_t start_pos, std::uint32_t rows,
                                std::uint32_t groups, std::uint32_t splits,
                                std::span<float> scratch, hipStream_t stream);

/// One layer of a pass over rows [start, start + rows) planned by
/// PlanKvarnPass: writes the rows, quantizes the group ending inside the
/// pass once its rows are written (plan.boundary), and attends rows before
/// the boundary over plan.read_groups groups and rows from it over one
/// more, each row as a decode step at its position does.
void LaunchKvarnPassAttention(const float* q, const float* k, const float* v,
                              const float* gate, const KvarnKvView& kv,
                              float* out_context, std::uint32_t layer,
                              std::uint32_t start, std::uint32_t rows,
                              const KvarnPassPlan& plan,
                              std::span<float> scratch, hipStream_t stream);

/// Where one layer of a prompt chunk stages its rows (KvarnStagingRows(rows)
/// rows of K then V) and quantizes the groups it completes. With a
/// quantize stream the quantizer runs there, after `ready` (recorded on the
/// pass stream once the rows are staged), and records `done`: it overlaps
/// the next layers instead of stalling the pass for its latency. Without
/// one it runs on the pass stream.
struct KvarnStaging {
  std::span<std::uint16_t> rows;
  hipStream_t quantize_stream{nullptr};
  hipEvent_t ready{nullptr};
  hipEvent_t done{nullptr};
};

/// One layer of a prompt chunk over rows [start, start + rows) planned by
/// PlanKvarnPrefill. Packs the K/V of positions [0, start + rows) in the
/// original frame into the head-major WMMA workspaces (quantized groups and
/// rotated FP16 rows dequantized and rotated back, the chunk's own rows as
/// the FP16 cache would hold them), in head groups that fit, runs the
/// prefill WMMA attention on each (output rows from `first_query`), then
/// stores the chunk: sink rows, groups completing inside the chunk
/// quantized from `staging` (KvarnStagingRows(rows) rows of K then V), the
/// rest into the ring. Throws std::length_error when one head does not fit
/// the workspaces.
void LaunchKvarnPrefillAttention(
    const float* q, const float* k, const float* v, const float* gate,
    const KvarnKvView& kv, float* out_context, std::uint32_t layer,
    std::uint32_t start, std::uint32_t rows, const KvarnPrefillPlan& plan,
    std::span<float> packed_k_workspace, std::span<float> packed_v_workspace,
    const KvarnStaging& staging, std::size_t first_query, hipStream_t stream);

/// The prefill WMMA attention (kernels/attention_wmma.hip) over KV heads
/// [first_head, first_head + heads) already packed head-major: K
/// [head][length][256], V [head][length / 16][256][16] (length padded to
/// 16), FP16, length = start_pos + batch_size.
void LaunchQwenWmmaPackedHeadsAttention(
    const float* q, const float* gate, const void* packed_k,
    const void* packed_v, float* out_context, std::uint32_t start_pos,
    std::size_t batch_size, std::uint32_t first_head, std::uint32_t heads,
    std::size_t first_query, hipStream_t stream);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_KVARN_HPP_
