#ifndef GUFO_MODELS_QWEN_HIP_OPS_PLAIN_PREFILL_STABLE_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_PLAIN_PREFILL_STABLE_HPP_

// Row-tiled FP32 GEMM of unquantized F16/BF16/F32 weights and FP32
// activations whose every output is byte-equal to Flash-Next's
// SmallGemmStable (src/models/qwen38_flash_next/kernels/rocm/
// kernels.hip.cpp, SmallGemmKernel<..., stable = true>) for any row count,
// tiling or position (plan/exl3/F3-PLAN.md "F3-API").

#include <cstdint>

#include "src/core/gguf_reader.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// output[r][j] = sum_i weights[j][i] * input[r][i] for r < rows, j < n:
/// weights row-major [n][k] of `type` (kF16, kBF16 or kF32), input [rows][k]
/// and output [rows][n] FP32, packed. The arithmetic is SmallGemmStable's:
/// lane class L (0..31) owns elements 4L .. 4L + 3 of every 128-element step
/// and accumulates fmaf(w, x, acc) over them in increasing element order
/// from +0 (weights widened exactly); the 32 partials are then summed by the
/// wave32 xor butterfly (offsets 16, 8, 4, 2, 1) and lane 0's total is
/// stored. Any k, n, rows (ragged tails masked; rows == 0 is a no-op).
void LaunchPlainPrefillStable(core::GgmlType type, const void* weights,
                              const float* input, float* output,
                              std::uint32_t rows, std::uint32_t n,
                              std::uint32_t k, hipStream_t stream);

}  // namespace gufo::hip

#endif  // ENGINE_ENABLE_HIP
#endif  // GUFO_MODELS_QWEN_HIP_OPS_PLAIN_PREFILL_STABLE_HPP_
