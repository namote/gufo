#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/hip/detail/hadamard.hpp"
#include "src/models/qwen/hip/ops/hadamard.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip {

// With Rotate the normed row is staged in dynamic LDS (dim floats) and
// rotated before both stores; the normed values are unchanged.
template<std::size_t CachedDim = 0, bool Rotate = false>
__global__ void BatchedRMSNormKernel(const float* __restrict__ x,
                                     const float* __restrict__ weight,
                                     float* __restrict__ out,
                                     hip_bfloat16* __restrict__ out_bf16,
                                     std::size_t batch_size, std::size_t dim,
                                     float eps,
                                     const std::uint32_t* __restrict__ signs) {
  const std::size_t b = blockIdx.x;
  if (b >= batch_size) {
    return;
  }

  const std::size_t tid = threadIdx.x;
  const std::size_t block_size = blockDim.x;
  const float* in_row = x + (b * dim);
  float* out_row = (out != nullptr) ? out + (b * dim) : nullptr;
  hip_bfloat16* out_bf16_row =
      (out_bf16 != nullptr) ? out_bf16 + (b * dim) : nullptr;

  static_assert(CachedDim % 256 == 0);
  constexpr std::size_t cached_items = CachedDim == 0 ? 1 : CachedDim / 256;
  float values[cached_items];
  float weights[cached_items];
  float sum_sq = 0.0F;
  if constexpr (CachedDim != 0) {
#pragma unroll
    for (std::size_t item = 0; item < cached_items; ++item) {
      const std::size_t i = tid + item * 256;
      const float val = in_row[i];
      values[item] = val;
      weights[item] = weight != nullptr ? weight[i] : 1.0F;
      // Unrolling must retain the generic loop's fused accumulation.
      sum_sq = __fmaf_rn(val, val, sum_sq);
    }
  } else {
    for (std::size_t i = tid; i < dim; i += block_size) {
      const float val = in_row[i];
      sum_sq += val * val;
    }
  }

  __shared__ float s_sum[256];
  s_sum[tid] = sum_sq;
  __syncthreads();

  for (std::size_t stride = block_size / 2;
       stride > (CachedDim != 0 ? 16U : 0U); stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    __syncthreads();
  }
  if constexpr (CachedDim != 0) {
    // Finish the same descending tree in its first wave.
    if (tid < 32) {
      float sum = s_sum[tid];
#pragma unroll
      for (unsigned offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset);
      }
      if (tid == 0)
        s_sum[0] = sum;
    }
    __syncthreads();
  }

  const float mean_sq = s_sum[0] / static_cast<float>(dim);
  const float inv_rms = rsqrtf(mean_sq + eps);

  if constexpr (Rotate) {
    extern __shared__ float s_row[];
    if constexpr (CachedDim != 0) {
#pragma unroll
      for (std::size_t item = 0; item < cached_items; ++item) {
        s_row[tid + item * 256] = (values[item] * inv_rms) * weights[item];
      }
    } else {
      for (std::size_t i = tid; i < dim; i += block_size) {
        const float w = (weight != nullptr) ? weight[i] : 1.0F;
        s_row[i] = (in_row[i] * inv_rms) * w;
      }
    }
    __syncthreads();
    hadamard::RotateRowSegments<32>(s_row, dim, signs,
                                    static_cast<unsigned>(tid),
                                    static_cast<unsigned>(block_size));
    __syncthreads();
    for (std::size_t i = tid; i < dim; i += block_size) {
      const float rotated = s_row[i];
      if (out_row != nullptr)
        out_row[i] = rotated;
      if (out_bf16_row != nullptr)
        out_bf16_row[i] = hip_bfloat16(rotated);
    }
  } else if constexpr (CachedDim != 0) {
#pragma unroll
    for (std::size_t item = 0; item < cached_items; ++item) {
      const std::size_t i = tid + item * 256;
      const float normed = (values[item] * inv_rms) * weights[item];
      if (out_row != nullptr)
        out_row[i] = normed;
      if (out_bf16_row != nullptr)
        out_bf16_row[i] = hip_bfloat16(normed);
    }
  } else {
    for (std::size_t i = tid; i < dim; i += block_size) {
      const float w = (weight != nullptr) ? weight[i] : 1.0F;
      const float normed = (in_row[i] * inv_rms) * w;
      if (out_row != nullptr)
        out_row[i] = normed;
      if (out_bf16_row != nullptr)
        out_bf16_row[i] = hip_bfloat16(normed);
    }
  }
}

void LaunchBatchedRMSNorm(const float* x, const float* weight, float* out,
                          void* out_bf16, std::size_t batch_size,
                          std::size_t dim, float eps, hipStream_t stream) {
  // Verification and draft rows can retain the hidden vector across the
  // reduction. Large prefill batches keep the generic occupancy profile.
  if (dim == 5120 && batch_size <= 64) {
    BatchedRMSNormKernel<5120><<<batch_size, 256, 0, stream>>>(
        x, weight, out, static_cast<hip_bfloat16*>(out_bf16), batch_size, dim,
        eps, nullptr);
    return;
  }
  BatchedRMSNormKernel<><<<batch_size, 256, 0, stream>>>(
      x, weight, out, static_cast<hip_bfloat16*>(out_bf16), batch_size, dim,
      eps, nullptr);
}

void LaunchBatchedRMSNormRotated(const float* x, const float* weight,
                                 float* out, void* out_bf16,
                                 std::size_t batch_size, std::size_t dim,
                                 float eps, const std::uint32_t* signs,
                                 hipStream_t stream) {
  // Same kernel selection as LaunchBatchedRMSNorm, so the normed values are
  // those the unrotated launch would store.
  const std::size_t lds = dim * sizeof(float);
  if (dim == 5120 && batch_size <= 64) {
    BatchedRMSNormKernel<5120, true><<<batch_size, 256, lds, stream>>>(
        x, weight, out, static_cast<hip_bfloat16*>(out_bf16), batch_size, dim,
        eps, signs);
    return;
  }
  BatchedRMSNormKernel<0, true><<<batch_size, 256, lds, stream>>>(
      x, weight, out, static_cast<hip_bfloat16*>(out_bf16), batch_size, dim,
      eps, signs);
}

__global__ void BatchedPerHeadRMSNormKernel(const float* __restrict__ x,
                                            const float* __restrict__ weight,
                                            float* __restrict__ out,
                                            std::size_t batch_size,
                                            std::uint32_t num_heads,
                                            std::uint32_t head_dim, float eps) {
  const std::size_t b = blockIdx.y;
  const std::uint32_t h = blockIdx.x;
  if (b >= batch_size || h >= num_heads)
    return;

  const std::size_t tid = threadIdx.x;
  const std::size_t head_total = static_cast<std::size_t>(num_heads) * head_dim;
  const float* x_head =
      x + (b * head_total) + (static_cast<std::size_t>(h) * head_dim);
  float* out_head =
      out + (b * head_total) + (static_cast<std::size_t>(h) * head_dim);

  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float val = x_head[i];
    sum_sq += val * val;
  }

  __shared__ float s_sum[256];
  s_sum[tid] = sum_sq;
  __syncthreads();

  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    __syncthreads();
  }

  const float mean_sq = s_sum[0] / static_cast<float>(head_dim);
  const float inv_rms = rsqrtf(mean_sq + eps);

  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float w = (weight != nullptr) ? weight[i] : 1.0F;
    out_head[i] = (x_head[i] * inv_rms) * w;
  }
}

void LaunchBatchedPerHeadRMSNorm(const float* x, const float* weight,
                                 float* out, std::size_t batch_size,
                                 std::uint32_t num_heads,
                                 std::uint32_t head_dim, float eps,
                                 hipStream_t stream) {
  const dim3 grid(num_heads, static_cast<unsigned int>(batch_size));
  BatchedPerHeadRMSNormKernel<<<grid, 256, 0, stream>>>(
      x, weight, out, batch_size, num_heads, head_dim, eps);
}
}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
