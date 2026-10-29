#ifndef GUFO_MODELS_QWEN_HIP_OPS_HADAMARD_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_HADAMARD_HPP_

// Input rotation x' = H·D·x for rotated Qwen GGUF weights
// (src/core/quant/rotation.hpp, src/core/hip/detail/hadamard.hpp).
//
// `signs` is the packed sign vector of one rotated input group (bit set ->
// negate) covering the row length. Rows are rotated per 128-element segment.
// Each fused producer below computes its unrotated value exactly as the
// unrotated launcher does, rotates the FP32 row with the shared device
// helper, and only then writes FP32, BF16, FP16 or tiled Q8_1. Rotation is
// therefore applied before any rounding to a narrower format.

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// In-place x' = H·D·x over `rows` FP32 rows of `k` elements (k % 128 == 0,
/// row stride k). The standalone form for activations whose producer writes
/// FP32 (attention output, SSM epilogue, SwiGLU on decode and verification).
void LaunchRotateRowsFp32(float* x, const std::uint32_t* signs,
                          std::size_t rows, std::size_t k,
                          hipStream_t stream = nullptr);

/// LaunchRMSNorm followed by the rotation of its FP32 output row.
void LaunchRMSNormRotated(const float* x, const float* weight, float* out,
                          std::size_t dim, float eps,
                          const std::uint32_t* signs,
                          hipStream_t stream = nullptr);

/// LaunchBatchedRMSNorm with the FP32 (and optional BF16) output rotated.
void LaunchBatchedRMSNormRotated(const float* x, const float* weight,
                                 float* out, void* out_bf16,
                                 std::size_t batch_size, std::size_t dim,
                                 float eps, const std::uint32_t* signs,
                                 hipStream_t stream = nullptr);

/// LaunchBatchedFusedRMSNormQuantizeQ8_1 with the normed row rotated in LDS
/// before the tiled Q8_1 quantization.
void LaunchBatchedFusedRMSNormRotateQuantizeQ8_1(
    const float* x, const float* residual, const float* weight, float* sum_out,
    void* q8_1_out, std::size_t batch_size, std::size_t dim, float eps,
    const std::uint32_t* signs, hipStream_t stream = nullptr);

/// LaunchBatchedRMSNormFp16 (no residual) with the row rotated before the
/// FP16 conversion.
void LaunchBatchedRMSNormRotatedFp16(const float* x, const float* weight,
                                     void* out, std::size_t batch,
                                     std::size_t dim, float eps,
                                     const std::uint32_t* signs,
                                     hipStream_t stream = nullptr);

/// SiLU(gate) * up, rotated, then the tiled Q8_1 encoding
/// LaunchBatchedFusedSwiGLUQuantizeQ8_1 writes.
void LaunchBatchedFusedSwiGLURotateQuantizeQ8_1(
    const float* gate, const float* up, void* q8_1_out, std::size_t batch,
    std::size_t k, const std::uint32_t* signs, hipStream_t stream = nullptr);

/// SiLU(gate) * up, rotated, then rounded to FP16 [batch, k] (the half
/// prefill ffn_down input).
void LaunchBatchedSwiGLURotateFp16(const float* gate, const float* up,
                                   void* out_fp16, std::size_t batch,
                                   std::size_t k, const std::uint32_t* signs,
                                   hipStream_t stream = nullptr);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_HADAMARD_HPP_
