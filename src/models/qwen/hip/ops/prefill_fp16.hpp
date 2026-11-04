#ifndef GUFO_MODELS_QWEN_HIP_OPS_PREFILL_FP16_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_PREFILL_FP16_HPP_
#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/exl3/device.hpp"
#include "src/models/qwen/hip/ops/exl3_prefill_bucket.hpp"
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
namespace gufo::hip {
// The half buffers contain IEEE FP16, not BF16. Callers reuse the existing
// scratch allocations after their BF16 contents are no longer live.
void LaunchFloatToFp16(const float* input, void* output, std::size_t elements,
                       hipStream_t stream);
// EXL3 trellis-decode WMMA GEMM (plan/exl3/X3-DESIGN.md; model-independent
// API: plan/exl3/F3-PLAN.md "F3-API"). x is the FP16 rotated input [rows][k]
// (H(suh * x) per 128 elements, row-major, LaunchExl3RotateRows); y [rows][n]
// FP32 receives svh * H(x W') (residual: added to y). The FP16 WMMA
// accumulation bound is (k / 16) * 2^-23 relative L2 (X3-RESULTS.md).
// Supported: mul1, integer K 1..8, n and k positive multiples of 128.
[[nodiscard]] bool Exl3GemmShapeSupported(
    std::uint32_t k, std::uint32_t n, quant::exl3::BitRate rate,
    quant::exl3::Codebook codebook) noexcept;
[[nodiscard]] bool Exl3PrefillSupported(
    const quant::exl3::DeviceLinear& linear) noexcept;
void LaunchExl3PrefillGemm(const quant::exl3::DeviceLinear& linear,
                           const void* x, float* y, std::size_t batch,
                           bool residual, hipStream_t stream);
// One problem of a grouped launch (an expert bucket). Device memory.
struct Exl3GemmProblem {
  const std::uint16_t* trellis;  // [k/16][n/16][16K]
  const std::uint16_t* svh;      // FP16 [n]
  const void* x;                 // FP16 rotated rows [rows][k]
  float* y;                      // [rows][n] FP32
  std::uint32_t rows;            // 0 skips the problem
  // Parent output width for an aligned slice; zero means n.
  std::uint32_t trellis_rows;
};
// Every problem of `problems` (device array, count <= 65535) with the same k,
// n and rate, in one launch; blocks past a problem's rows exit. max_rows
// bounds every problem's rows and picks the tile (<= 64: narrow). When
// strided_trellis is true, trellis_rows supplies the parent K-step stride;
// trellis/svh already point at a 128-channel-aligned output slice of width n.
void LaunchExl3PrefillGemmGrouped(const Exl3GemmProblem* problems,
                                  std::size_t count, std::size_t max_rows,
                                  std::uint32_t k, std::uint32_t n,
                                  quant::exl3::BitRate rate, bool residual,
                                  hipStream_t stream,
                                  bool strided_trellis = false);
// Raw inner products of expert buckets (Flash-Next routed prefill): for
// every active descriptor b < min(*device_bucket_count, max_buckets) and
// row r < count, inner[(first + r) * n + j] = sum_i input[(first + r) * k +
// i] * W'[i][j] in FP32 for the bank's expert (its trellis only; no output
// Hadamard, svh or suh). input is the FP16 rotated slab [rows][k], inner
// [rows][n], both packed; rows past count are neither read nor written. The
// count is read on the device (graph-replay safe). Each value depends only
// on its own input row and expert, never on the partition into buckets,
// tile_rows or its position in a tile. Bank entries must be mul1 at `rate`
// (integer K1..K8) with this k and n (checked on the device: a mismatching
// bucket writes NaN); k and n multiples of 128, tile_rows 1..64 (16/32/64
// are the tuned tiles). Every count <= tile_rows, except that with
// tile_rows > 32 a bucket may also hold 65..128 rows (a hot expert's).
void LaunchExl3PrefillBucketInner(const quant::exl3::DeviceLinear* device_bank,
                                  const Exl3PrefillBucket* device_buckets,
                                  const std::uint32_t* device_bucket_count,
                                  std::uint32_t max_buckets,
                                  std::uint32_t tile_rows,
                                  const std::uint16_t* input, float* inner,
                                  std::uint32_t k, std::uint32_t n,
                                  quant::exl3::BitRate rate,
                                  hipStream_t stream);
// y[r][j] = sum_i w[j][i] * x[r][i] for unquantized FP16 weights w
// [m][k] and FP16 rows x [rows][k]; FP32 accumulation in one WMMA chain per
// output in K order (independent of rows or tiling). k % 16 == 0.
void LaunchPlainF16PrefillGemm(const void* w, const void* x, float* y,
                               std::size_t rows, std::uint32_t m,
                               std::uint32_t k, hipStream_t stream);
// LaunchExl3PrefillBucketInner with FP16 outputs: each value (divided first
// by 2^row_exponents[row] when that entry is positive; row_exponents is
// indexed by the packed input row and may be null) rounded once to FP16.
void LaunchExl3PrefillBucketInnerHalf(
    const quant::exl3::DeviceLinear* device_bank,
    const Exl3PrefillBucket* device_buckets,
    const std::uint32_t* device_bucket_count, std::uint32_t max_buckets,
    std::uint32_t tile_rows, const std::uint16_t* input, std::uint16_t* inner,
    std::uint32_t k, std::uint32_t n, quant::exl3::BitRate rate,
    const std::int32_t* row_exponents, hipStream_t stream);
// Optional residual and sum_out preserve the FP32 residual link; sum_out may
// alias input. The normalization reduction matches LaunchBatchedRMSNorm.
// The FP16 output must be disjoint from the FP32 inputs and weights.
void LaunchBatchedRMSNormFp16(const float* input, const float* residual,
                              const float* weight, float* sum_out, void* output,
                              std::size_t batch, std::size_t dim, float eps,
                              hipStream_t stream);
// Packed GGUF weights are scaled in FP32, rounded to FP16 inside the kernel,
// then multiplied by FP16 activations with FP32 accumulation in K16 order.
// Supports the Qwen27B Q4 shard's native quant formats and K divisible by 256.
void LaunchBatchedQuantGEMMFp16(core::GgmlType type, const void* weights,
                                const void* input, float* output,
                                std::size_t batch, std::size_t m, std::size_t k,
                                hipStream_t stream);
// Fuses the up projection and SwiGLU. gate is FP32; output is FP16 and must
// not alias input. Its storage may reuse the otherwise dead FP32 up buffer.
void LaunchBatchedQuantGEMMSwiGLUFp16(core::GgmlType type, const void* weights,
                                      const void* input, const float* gate,
                                      void* output, std::size_t batch,
                                      std::size_t m, std::size_t k,
                                      hipStream_t stream);
// Adds the completed FP32 dot product to residual in place. Input and residual
// must not alias; accumulation starts at zero, as in the standalone GEMM.
void LaunchBatchedQuantGEMMResidualFp16(core::GgmlType type,
                                        const void* weights, const void* input,
                                        float* residual, std::size_t batch,
                                        std::size_t m, std::size_t k,
                                        hipStream_t stream);
// Qualified gate/up weight pairs share a kernel that emits FP16 SwiGLU
// directly. Returns false without launching for other pairs.
// All buffers are disjoint; the output may reuse the dead FP32 up allocation.
bool TryLaunchBatchedDualQuantGEMMSwiGLUFp16(
    core::GgmlType gate_type, core::GgmlType up_type, const void* gate_weights,
    const void* up_weights, const void* input, void* output, std::size_t batch,
    std::size_t m, std::size_t k, hipStream_t stream);
}  // namespace gufo::hip
#endif
#endif
