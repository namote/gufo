#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXL3_MOE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXL3_MOE_HPP_

#include <hip/hip_runtime.h>

#include "src/core/quant/exl3/reference.hpp"
#include "src/models/qwen/hip/ops/exl3_prefill_bucket.hpp"
#include "src/models/qwen38_flash_next/runtime/device_model.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

struct Exl3RouteTile {
  std::uint32_t expert, count;
  std::uint32_t slots[8];
};

struct Exl3Scratch {
  quant::exl3::ReferenceScratch projection;
  float* slice{nullptr};
  std::size_t slice_elements{0};
  float* gather{nullptr};
  float* gate{nullptr};
  float* up{nullptr};
  float* out{nullptr};
  /// [rows * used][max(2 * hidden, expert_ff)] FP16 elements.
  std::uint16_t* routed_input{nullptr};
  Exl3RouteTile* tiles{nullptr};
  std::uint32_t* tile_count{nullptr};
  /// [rows * used], negative means widened-FMA fallback. Optional: null
  /// retains widened down arithmetic; otherwise routed_input is scaled in
  /// place.
  std::int32_t* down_exponents{nullptr};
  std::uint32_t* head_segments{nullptr};
  float* head_values{nullptr};
  /// Zero-initialized arrival counter of RouterTopKTiles; it returns to zero
  /// at the end of every launch.
  std::uint32_t* route_arrivals{nullptr};
  /// FP16 lane-major input slabs written by decode producer epilogues
  /// (Exl3InputSlabs, 3 * 8 * widest input halves); only producers write it,
  /// so other fast launches between producer and consumer cannot clobber it.
  std::uint16_t* rotated_input{nullptr};
  std::size_t rotated_input_elements{0};
};

/// Rows per routed decode tile for a call of `rows` tokens. R=3 does not
/// carry eight independent accumulator pairs through the complete K loop.
constexpr std::uint32_t Exl3RouteTileRows(std::uint32_t rows) {
  return rows <= 3 ? rows : rows <= 4 ? 4 : 8;
}
/// Largest assignment count whose tiles RouterTopKTiles builds.
constexpr std::uint32_t kExl3FusedTileAssignments = 256;

/// Compressed projections, or the pack's plain operands with fixed FP32 math.
/// A non-null `rotated` (scratch.rotated_input) holds the rows already
/// rotated (slab 0, rows <= 8) for the fast path, which then does not read x.
void Exl3Projection(const DeviceTensor& weight, const float* x, float* out,
                    std::uint32_t rows, const Exl3Scratch& scratch,
                    hipStream_t stream, bool fast = true,
                    const std::uint16_t* rotated = nullptr);
void Exl3Gather(const float* x, const std::int32_t* rows, float* out,
                std::uint32_t count, std::uint32_t width, hipStream_t stream);
void Exl3Scatter(const float* x, const std::int32_t* slots, float* out,
                 std::uint32_t count, std::uint32_t width, hipStream_t stream);
/// Optional device destinations for operator qualification. Copies occur
/// between launches; kernels and their arithmetic do not depend on tracing.
struct Exl3RoutedTrace {
  std::uint16_t* gate_up_input{nullptr};  ///< [2][rows * used][hidden]
  float* gate_inner{nullptr};             ///< [rows * used][expert_ff]
  float* up_inner{nullptr};               ///< [rows * used][expert_ff]
  float* activation{nullptr};             ///< [rows * used][expert_ff]
  std::uint16_t* down_input{nullptr};     ///< [rows * used][expert_ff]
  float* down_inner{nullptr};             ///< [rows * used][hidden]
  float* down_packed_inner{nullptr};      ///< Same input, original packed dot.
  float* down_widened_inner{nullptr};     ///< Same input, widened-FMA control.
};

/// Shared expert's decode/verify contract, common to serial and cohort calls.
/// Returns false without launches when the fused fast MLP cannot apply.
/// A non-null `rotated` (scratch.rotated_input) holds the gate and up slabs
/// of the rows (rows <= 8), so x is not read.
bool Exl3SharedMlp(const DeviceLayer& layer, const float* x, float* gate,
                   float* up, float* out, std::uint32_t rows,
                   const Exl3Scratch& scratch, hipStream_t stream,
                   const std::uint16_t* rotated = nullptr);

/// Groups actual assignments by expert into 1..8-row tiles on device. Every
/// slot keeps its own rotation and output; the caller performs ordered MoE sum.
/// tiles_ready: RouterTopKTiles already built scratch.tiles/tile_count for
/// these IDs with Exl3RouteTileRows(rows). raw_down: out keeps the down
/// projection's FP32 dots, without H/svh, for Exl3RoutedCombine. A trace
/// runs the unfused restore/scale launches (the fused ones store the same
/// bytes) so that it can copy their intermediates.
void Exl3RoutedMlp(const DeviceLayer& layer, const float* x,
                   const std::int32_t* ids, float* gate, float* up, float* out,
                   std::uint32_t rows, std::uint32_t used,
                   const Exl3Scratch& scratch, hipStream_t stream,
                   const Exl3RoutedTrace* trace = nullptr,
                   bool tiles_ready = false, bool raw_down = false);
/// MoeEpilogue over Exl3RoutedMlp's raw_down rows: out[t] = sum_s
/// weights[t*used+s] * restored(t, s) + sigmoid(gate[t*stride]) * shared[t],
/// restoring each [t*used+s] row as Exl3RoutedMlp does, bit for bit, without
/// the restored rows' round trip through memory. Requires used <= 16 and a
/// down width divisible by 128.
void Exl3RoutedCombine(const DeviceLayer& layer, const std::int32_t* ids,
                       std::uint32_t rows, std::uint32_t used, const float* raw,
                       const float* weights, const float* shared,
                       const float* gate, std::uint32_t gate_stride, float* out,
                       hipStream_t stream);
/// Inner routed-down stage for a pre-rounded FP16 slab. Uses the existing
/// scratch tiles/count (each count <= max_tile_rows, 1..8). With exponents
/// allocated, scales input in place by an exact power of two where safe,
/// restores the FP32 dot scale before returning, and falls back per row.
/// No output Hadamard/svh is applied here. Zero assignments is a no-op.
void Exl3RoutedDown(const DeviceTensor& down, std::uint16_t* input, float* out,
                    std::uint32_t assignments, std::uint32_t max_tile_rows,
                    const Exl3Scratch& scratch, hipStream_t stream);

/// One compact tile of an expert bucket, with contiguous packed input rows.
/// slots map the bucket rows back to [token * used + expert_slot]. A final
/// partial tile has count < tile_rows; inactive descriptors have count == 0.
using Exl3ExpertBucket = gufo::hip::Exl3PrefillBucket;

/// Valid IDs in [0, experts) are required.
/// Device-only bucket construction; capacities: offsets[experts + 1],
/// slots[assignments], buckets[max_buckets], bucket_count[1]. max_buckets
/// must be >= min(assignments, experts + assignments / tile_rows). Slots
/// remain in assignment order within each expert. No host count download.
/// With tile_rows 64, an expert with more than 64 rows gets an even split
/// into buckets of at most 128 rows (the X3 128-token tile).
void Exl3BuildExpertBuckets(const std::int32_t* ids, std::uint32_t assignments,
                            std::uint32_t experts, std::uint32_t tile_rows,
                            std::uint32_t* offsets, std::int32_t* slots,
                            Exl3ExpertBucket* buckets,
                            std::uint32_t* bucket_count,
                            std::uint32_t max_buckets, hipStream_t stream,
                            std::int32_t* inverse = nullptr);
/// Gather + FP32 rotation + FP16 RNE for bucketed gate/up or down. For
/// gate/up, input_row = slot / used; for down it is slot. Offsets and slots
/// are produced by Exl3BuildExpertBuckets. Packed output is [assignments][k].
void Exl3RotateExpertBuckets(const quant::exl3::DeviceLinear* bank,
                             const float* input, std::uint16_t* rotated,
                             const std::uint32_t* offsets,
                             const std::int32_t* slots, std::uint32_t experts,
                             std::uint32_t assignments, std::uint32_t width,
                             std::uint32_t used, bool down, hipStream_t stream);
/// Unpadded packed output -> original slots; ordered MoE sum stays in caller.
void Exl3ScatterExpertBuckets(const float* packed, const std::int32_t* slots,
                              float* out, std::uint32_t assignments,
                              std::uint32_t width, hipStream_t stream);

/// Raw-inner adapter signature for qualification. One call launches all device
/// buckets, bounded by max_buckets, with no host count/descriptor reads.
/// Input/output are packed [assignments][k/n]. Output MUST be the raw
/// FP32 inner dot, without H/svh. This model owns restoration, down-row
/// unscaling and widened fallback. Reads/stores must mask ragged row and
/// output tails (notably n=640); no slab padding is required.
using Exl3BucketGemm = void (*)(const quant::exl3::DeviceLinear* bank,
                                const Exl3ExpertBucket* buckets,
                                const std::uint32_t* bucket_count,
                                std::uint32_t max_buckets,
                                std::uint32_t tile_rows,
                                const std::uint16_t* input, float* output,
                                std::uint32_t k, std::uint32_t n,
                                quant::exl3::BitRate rate, hipStream_t stream);

/// Production adapter: the shared X3 bucket WMMA at the bank's rate (K3..K8).
void Exl3PrefillBucketInner(const quant::exl3::DeviceLinear* bank,
                            const Exl3ExpertBucket* buckets,
                            const std::uint32_t* bucket_count,
                            std::uint32_t max_buckets, std::uint32_t tile_rows,
                            const std::uint16_t* input, float* inner,
                            std::uint32_t k, std::uint32_t n,
                            quant::exl3::BitRate rate, hipStream_t stream);

struct Exl3PrefillScratch {
  std::uint32_t* offsets{nullptr};        ///< [experts + 1]
  std::int32_t* slots{nullptr};           ///< [max_assignments]
  Exl3ExpertBucket* buckets{nullptr};     ///< [max_buckets]
  std::uint32_t* bucket_count{nullptr};   ///< [1]
  std::uint16_t* input{nullptr};          ///< [max_assignments][max_width]
  float* gate{nullptr};                   ///< [max_assignments][expert_ff]
  float* up{nullptr};                     ///< [max_assignments][expert_ff]
  float* out{nullptr};                    ///< [max_assignments][hidden]
  std::int32_t* down_exponents{nullptr};  ///< [max_assignments]
  /// Packed position of each assignment (slots inverted), for the fused
  /// combine; optional elsewhere.
  std::int32_t* inverse{nullptr};  ///< [max_assignments]
  std::uint32_t max_assignments{0}, max_buckets{0};
};

/// Optional qualification snapshots in packed bucket order. Counters report
/// actual launched fallback blocks and unsafe rows processed, respectively.
struct Exl3PrefillTrace {
  std::uint16_t* gate_up_input{nullptr};
  float* gate_inner{nullptr};
  float* up_inner{nullptr};
  float* activation{nullptr};
  std::uint16_t* down_input{nullptr};
  std::uint32_t* fallback_counts{nullptr};  ///< [2], reset on each invocation
};

/// Consume scratch.input as already-rounded, packed FP16 down rows. This is
/// also the down stage of Exl3PrefillExperts, including scaling, widened
/// fallback and restoration. scratch.out receives the packed raw dots; the
/// restored rows go back to scratch.out (packed) or, with scatter_out, to
/// scatter_out row scratch.slots[row] (Exl3ScatterExpertBuckets fused).
void Exl3PrefillDown(const DeviceTensor& down, std::uint32_t assignments,
                     std::uint32_t tile_rows, const Exl3PrefillScratch& scratch,
                     Exl3BucketGemm gemm, hipStream_t stream,
                     const Exl3PrefillTrace* trace = nullptr,
                     float* scatter_out = nullptr, bool raw_only = false);

/// Complete bucketed prefill dispatch using the shared X3 production adapter.
/// Valid, distinct expert IDs per token are required. The caller retains
/// scratch until all launches finish and performs the ordered MoE sum.
/// Input/output and all scratch slabs must be disjoint.
void Exl3PrefillExperts(const DeviceLayer& layer, const float* x,
                        const std::int32_t* ids, float* out, std::uint32_t rows,
                        std::uint32_t used, std::uint32_t tile_rows,
                        const Exl3PrefillScratch& scratch, hipStream_t stream,
                        const Exl3PrefillTrace* trace = nullptr);
/// Exl3PrefillExperts up to the down projection's raw dots (scratch.out,
/// packed, with scratch.down_exponents and scratch.inverse); then
/// Exl3PrefillCombine restores them and forms the MoE output out[t] =
/// sum_s weights[t*used+s] * restored(t, s) + sigmoid(gate[t*stride]) *
/// shared[t]: the values of Exl3PrefillExperts + MoeEpilogue, bit for bit,
/// without the restored rows' round trip through memory.
void Exl3PrefillExpertsRaw(const DeviceLayer& layer, const float* x,
                           const std::int32_t* ids, std::uint32_t rows,
                           std::uint32_t used, std::uint32_t tile_rows,
                           const Exl3PrefillScratch& scratch,
                           hipStream_t stream);
/// The raw path's stages (also its operator bench entry points). Gate/up
/// input: each of a token's `used` assignments gets x[row] rotated with its
/// expert's gate and up suh, stored as FP16 at the assignment's packed row
/// inverse[row * used + s] (1 <= used <= 32, width % 128 == 0).
void Exl3RotateGateUpTokens(const quant::exl3::DeviceLinear* gate,
                            const quant::exl3::DeviceLinear* up, const float* x,
                            std::uint16_t* gate_out, std::uint16_t* up_out,
                            const std::int32_t* ids,
                            const std::int32_t* inverse, std::uint32_t rows,
                            std::uint32_t width, std::uint32_t used,
                            hipStream_t stream);
/// Down input: each packed row's FP16 gate/up dots restored, SwiGLU, rotated
/// with the down suh and power-of-two scaled into FP16 `output`, with the
/// row's exponent (width = expert FF, a multiple of 128 up to 4,096).
void Exl3RestoreActivateScale(
    const quant::exl3::DeviceLinear* gate, const quant::exl3::DeviceLinear* up,
    const quant::exl3::DeviceLinear* down, const std::uint32_t* offsets,
    std::uint32_t experts, const std::uint16_t* gate_y,
    const std::uint16_t* up_y, std::uint16_t* output, std::int32_t* exponents,
    std::uint32_t assignments, std::uint32_t width, hipStream_t stream);
/// Retain the original IDs passed to Exl3PrefillExpertsRaw until combine
/// completes; inverse rows and weights use that same token/slot order.
void Exl3PrefillCombine(const DeviceLayer& layer,
                        const Exl3PrefillScratch& scratch,
                        const std::int32_t* ids, std::uint32_t rows,
                        std::uint32_t used, const float* weights,
                        const float* shared, const float* gate,
                        std::uint32_t gate_stride, float* out,
                        hipStream_t stream);
/// Qualification overload: substitute an independent inner-product control.
void Exl3PrefillExperts(const DeviceLayer& layer, const float* x,
                        const std::int32_t* ids, float* out, std::uint32_t rows,
                        std::uint32_t used, std::uint32_t tile_rows,
                        const Exl3PrefillScratch& scratch, Exl3BucketGemm gemm,
                        hipStream_t stream, const Exl3PrefillTrace* trace);

/// Prefix plus unique complete 128-column segments for the dynamic IDs.
/// Original trellis N stride is retained, and only final logits are gathered.
void Exl3DraftHead(const DeviceTensor& head, const float* x, float* logits,
                   std::uint32_t prefix, const std::int32_t* ids,
                   std::uint32_t extras, const Exl3Scratch& scratch,
                   hipStream_t stream);

}  // namespace gufo::models::qwen38_flash_next::rocm
#endif
