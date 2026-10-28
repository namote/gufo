// Two-bit weight kernels for tools/bench/rq_type_bench.hip (Q2_K, IQ2_XXS,
// IQ2_XS, IQ2_S). Gufo's Qwen 27B path has no kernels for these formats, so
// the bench measures upstream ggml's HIP matrix-vector kernel instead.
//
// Kernels: llama.cpp ggml/src/ggml-cuda/{mmvq.cu,vecdotq.cuh,quantize.cu},
// MIT License, Copyright (c) 2023-2026 The ggml authors, llama.cpp commit
// 5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a. They are linked unchanged from the
// in-tree HIP copy in src/models/deepseek_v4_flash/kernels/rocm/mmq (see its
// VENDOR.md): mul_mat_vec_q_switch_type (mmvq.hip.cpp) and
// ds4_quantize_row_q8_1_hip (quantize.hip.cpp). On gfx1151 (RDNA3.5) upstream
// selects its RDNA2 parameter table: one wave32 per output row and block.
//
// This header adds the launch wrapper and an independent scalar dequantizer
// for the correctness check, written from ggml-quants.c dequantize_row_* of
// the same commit. Compile the implementation as its own unit with the
// vendored kernels' flags (see plan/rotated-quant/R0B-RESULTS.md):
//   hipcc -x hip -DRQ_MMVQ_IMPLEMENTATION -std=c++17 -DGGML_USE_HIP ...
#ifndef GUFO_TOOLS_BENCH_RQ_TYPE_BENCH_MMVQ_HPP_
#define GUFO_TOOLS_BENCH_RQ_TYPE_BENCH_MMVQ_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>

namespace rq_mmvq {

// ggml_type values.
enum class Type : int { kQ2_K = 10, kIQ2_XXS = 16, kIQ2_XS = 17, kIQ2_S = 22 };

// Bytes of the upstream row-major block_q8_1 activation for rows x k.
std::size_t Q8_1Bytes(std::size_t rows, std::size_t k);

// Upstream quantize_row_q8_1: x[rows, k] FP32 to block_q8_1 rows.
void QuantizeQ8_1(const float* x, void* q8_1, std::size_t rows, std::size_t k,
                  hipStream_t stream);

// Upstream mmvq: y[rows, m] = q8_1[rows, k] * w[m, k]^T, rows <= 8.
void Mmvq(Type type, const void* w, const void* q8_1, float* y,
          std::size_t rows, std::size_t m, std::size_t k, hipStream_t stream);

// Scalar reference: dequantizes the first m rows of w to out[m, k].
void Dequantize(Type type, const void* w, float* out, std::size_t m,
                std::size_t k, hipStream_t stream);

}  // namespace rq_mmvq

#if defined(RQ_MMVQ_IMPLEMENTATION)

#include "mmvq.hip.hpp"
#include "quantize.hip.hpp"

namespace rq_mmvq {

std::size_t Q8_1Bytes(std::size_t rows, std::size_t k) {
  return rows * (k / QK8_1) * sizeof(block_q8_1);
}

void QuantizeQ8_1(const float* x, void* q8_1, std::size_t rows, std::size_t k,
                  hipStream_t stream) {
  // Upstream pads rows to MATRIX_ROW_PADDING; every Qwen 27B K is a multiple.
  GGML_ASSERT(k % MATRIX_ROW_PADDING == 0);
  const auto n = static_cast<std::int64_t>(k);
  const auto r = static_cast<std::int64_t>(rows);
  ds4_quantize_row_q8_1_hip(x, nullptr, q8_1, GGML_TYPE_Q2_K, n, n, n * r,
                            n * r, n, r, 1, 1, stream);
}

void Mmvq(Type type, const void* w, const void* q8_1, float* y,
          std::size_t rows, std::size_t m, std::size_t k, hipStream_t stream) {
  const auto t = static_cast<ggml_type>(type);
  const int blocks_per_row = static_cast<int>(k / ggml_blck_size(t));
  const int y_blocks = static_cast<int>(k / QK8_1);
  const ds4_ggml_hip_mm_fusion_args_device fusion{};
  // Same strides as upstream ggml_cuda_mul_mat_vec_q for contiguous 2-D
  // operands: one channel and one sample.
  mul_mat_vec_q_switch_type(
      w, t, q8_1, nullptr, fusion, y, static_cast<int>(k), static_cast<int>(m),
      static_cast<int>(rows), blocks_per_row, y_blocks, static_cast<int>(m), 1,
      1, 1, static_cast<int>(m) * blocks_per_row,
      static_cast<int>(rows) * y_blocks, static_cast<int>(rows * m), 1, 1,
      static_cast<int>(m) * blocks_per_row, static_cast<int>(rows) * y_blocks,
      static_cast<int>(rows * m), 0, stream);
}

namespace {

__device__ float Half(const unsigned char* p) {
  return __half2float(*reinterpret_cast<const __half*>(p));
}

__device__ float Sign(unsigned signs, int j) {
  return (signs & kmask_iq2xs[j]) ? -1.0F : 1.0F;
}

// One thread per 256-value block, following ggml-quants.c dequantize_row_*.
template<int Type>
__global__ void DequantizeKernel(const unsigned char* w, float* out,
                                 std::size_t blocks) {
  const std::size_t b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= blocks)
    return;
  float* y = out + b * 256;
  if constexpr (Type == GGML_TYPE_Q2_K) {
    const unsigned char* x = w + b * 84;
    const unsigned char* scales = x;
    const unsigned char* q = x + 16;
    const float d = Half(x + 80);
    const float min = Half(x + 82);
    int is = 0;
    for (int n = 0; n < 256; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        unsigned sc = scales[is++];
        float dl = d * (sc & 0xF);
        float ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l)
          *y++ = dl * ((q[l] >> shift) & 3) - ml;
        sc = scales[is++];
        dl = d * (sc & 0xF);
        ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l)
          *y++ = dl * ((q[l + 16] >> shift) & 3) - ml;
        shift += 2;
      }
      q += 32;
    }
  } else if constexpr (Type == GGML_TYPE_IQ2_XXS) {
    const unsigned char* x = w + b * 66;
    const float d = Half(x);
    for (int ib32 = 0; ib32 < 8; ++ib32) {
      const unsigned char* aux8 = x + 2 + 8 * ib32;
      unsigned aux1 = 0;
      for (int i = 0; i < 4; ++i)
        aux1 |= static_cast<unsigned>(aux8[4 + i]) << (8 * i);
      const float db = d * (0.5F + (aux1 >> 28)) * 0.25F;
      for (int l = 0; l < 4; ++l) {
        const auto* grid =
            reinterpret_cast<const unsigned char*>(iq2xxs_grid + aux8[l]);
        const unsigned signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
        for (int j = 0; j < 8; ++j)
          *y++ = db * grid[j] * Sign(signs, j);
      }
    }
  } else if constexpr (Type == GGML_TYPE_IQ2_XS) {
    const unsigned char* x = w + b * 74;
    const float d = Half(x);
    const unsigned char* scales = x + 66;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
      const float db[2] = {d * (0.5F + (scales[ib32] & 0xF)) * 0.25F,
                           d * (0.5F + (scales[ib32] >> 4)) * 0.25F};
      for (int l = 0; l < 4; ++l) {
        const unsigned char* p = x + 2 + 2 * (4 * ib32 + l);
        const unsigned qs = p[0] | (static_cast<unsigned>(p[1]) << 8);
        const auto* grid =
            reinterpret_cast<const unsigned char*>(iq2xs_grid + (qs & 511));
        const unsigned signs = ksigns_iq2xs[qs >> 9];
        for (int j = 0; j < 8; ++j)
          *y++ = db[l / 2] * grid[j] * Sign(signs, j);
      }
    }
  } else {
    static_assert(Type == GGML_TYPE_IQ2_S);
    const unsigned char* x = w + b * 82;
    const float d = Half(x);
    const unsigned char* qs = x + 2;
    const unsigned char* signs = qs + 32;
    const unsigned char* qh = x + 66;
    const unsigned char* scales = x + 74;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
      const float db[2] = {d * (0.5F + (scales[ib32] & 0xF)) * 0.25F,
                           d * (0.5F + (scales[ib32] >> 4)) * 0.25F};
      for (int l = 0; l < 4; ++l) {
        const unsigned index = qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300);
        const auto* grid =
            reinterpret_cast<const unsigned char*>(iq2s_grid + index);
        for (int j = 0; j < 8; ++j)
          *y++ = db[l / 2] * grid[j] * Sign(signs[l], j);
      }
      qs += 4;
      signs += 4;
    }
  }
}

}  // namespace

void Dequantize(Type type, const void* w, float* out, std::size_t m,
                std::size_t k, hipStream_t stream) {
  const std::size_t blocks = m * k / 256;
  const auto* bytes = static_cast<const unsigned char*>(w);
  const dim3 grid(static_cast<unsigned>((blocks + 63) / 64));
  switch (type) {
    case Type::kQ2_K:
      DequantizeKernel<GGML_TYPE_Q2_K>
          <<<grid, 64, 0, stream>>>(bytes, out, blocks);
      break;
    case Type::kIQ2_XXS:
      DequantizeKernel<GGML_TYPE_IQ2_XXS>
          <<<grid, 64, 0, stream>>>(bytes, out, blocks);
      break;
    case Type::kIQ2_XS:
      DequantizeKernel<GGML_TYPE_IQ2_XS>
          <<<grid, 64, 0, stream>>>(bytes, out, blocks);
      break;
    case Type::kIQ2_S:
      DequantizeKernel<GGML_TYPE_IQ2_S>
          <<<grid, 64, 0, stream>>>(bytes, out, blocks);
      break;
  }
}

}  // namespace rq_mmvq

#endif  // defined(RQ_MMVQ_IMPLEMENTATION)

#endif  // GUFO_TOOLS_BENCH_RQ_TYPE_BENCH_MMVQ_HPP_
