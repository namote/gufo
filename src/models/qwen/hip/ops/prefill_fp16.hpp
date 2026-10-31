#ifndef GUFO_MODELS_QWEN_HIP_OPS_PREFILL_FP16_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_PREFILL_FP16_HPP_
#include <cstddef>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/exl3/device.hpp"
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
namespace gufo::hip {
// The half buffers contain IEEE FP16, not BF16. Callers reuse the existing
// scratch allocations after their BF16 contents are no longer live.
void LaunchFloatToFp16(const float* input, void* output, std::size_t elements,
                       hipStream_t stream);
// EXL3 prefill projection (plan/exl3/X3-DESIGN.md): the FP16 GEMM frame with
// the trellis decode producer. x is the FP16 rotated input [batch][k]
// (H(suh * x) per 128 elements, row-major); y [batch][n] FP32 receives
// svh * H(x W') (residual: added to y). Supported: mul1, integer K 1..6,
// n and k positive, n % 256 == 0, k % 128 == 0.
[[nodiscard]] bool Exl3PrefillSupported(
    const quant::exl3::DeviceLinear& linear) noexcept;
void LaunchExl3PrefillGemm(const quant::exl3::DeviceLinear& linear,
                           const void* x, float* y, std::size_t batch,
                           bool residual, hipStream_t stream);
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
