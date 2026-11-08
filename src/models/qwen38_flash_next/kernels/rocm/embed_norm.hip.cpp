// Token embedding lookup and RMSNorm rows.
// HIP kernels follow the layouts and operator formulas in reference.cpp.

#include "src/models/qwen38_flash_next/kernels/rocm/kernels_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

namespace {

__global__ void EmbedKernel(const void* table, WeightType type,
                            const std::int32_t* tokens, float* res,
                            std::uint32_t hidden, std::uint32_t streams) {
  const std::uint32_t t = blockIdx.x;
  const auto* row = static_cast<const std::uint8_t*>(table) +
                    RowBytes(type, hidden) * tokens[t];
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = RowElement(row, type, i);
    for (std::uint32_t s = 0; s < streams; ++s) {
      res[(static_cast<std::size_t>(t) * streams + s) * hidden + i] = v;
    }
  }
}

__global__ void RmsNormKernel(const float* x, const float* gamma, float* out,
                              std::uint32_t group_dim, std::uint32_t groups,
                              float eps) {
  __shared__ float shared[32];
  const std::size_t row = blockIdx.x;
  const float* src = x + row * group_dim;
  float* dst = out + row * group_dim;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < group_dim; i += blockDim.x) {
    ss += src[i] * src[i];
  }
  ss = BlockSum(ss, shared);
  const float scale = rsqrtf(ss / static_cast<float>(group_dim) + eps);
  // A grouped norm shares one gamma row across the groups of a token.
  const float* g =
      gamma == nullptr ? nullptr : gamma + (row % groups) * group_dim;
  for (std::uint32_t i = threadIdx.x; i < group_dim; i += blockDim.x) {
    dst[i] = src[i] * scale * (g != nullptr ? g[i] : 1.0f);
  }
}

/// RmsNormKernel for rows of exactly kPer * kThreads elements: every load,
/// gamma included, is issued before the sum and the scaling pass reuses the
/// registers. Each thread keeps its elements and its fma chain in the same
/// order, so the sums and the outputs are bit for bit those of the loop.
template<unsigned kPer>
__global__ void RmsNormCachedKernel(const float* x, const float* gamma,
                                    float* out, std::uint32_t group_dim,
                                    std::uint32_t groups, float eps) {
  __shared__ float shared[32];
  const std::size_t row = blockIdx.x;
  const float* src = x + row * group_dim;
  float* dst = out + row * group_dim;
  const float* g =
      gamma == nullptr ? nullptr : gamma + (row % groups) * group_dim;
  float v[kPer];
  float gv[kPer];
#pragma unroll
  for (unsigned j = 0; j < kPer; ++j) {
    v[j] = src[threadIdx.x + j * kThreads];
  }
#pragma unroll
  for (unsigned j = 0; j < kPer; ++j) {
    gv[j] = g != nullptr ? g[threadIdx.x + j * kThreads] : 1.0f;
  }
  float ss = 0.0f;
#pragma unroll
  for (unsigned j = 0; j < kPer; ++j) {
    ss = Materialize(__builtin_fmaf(v[j], v[j], ss));
  }
  ss = BlockSum(ss, shared);
  const float scale = rsqrtf(ss / static_cast<float>(group_dim) + eps);
#pragma unroll
  for (unsigned j = 0; j < kPer; ++j) {
    dst[threadIdx.x + j * kThreads] = Materialize(v[j] * scale) * gv[j];
  }
}

}  // namespace

void EmbedTokens(const void* table, WeightType type, const std::int32_t* tokens,
                 float* res, std::uint32_t n_tokens, std::uint32_t hidden,
                 std::uint32_t streams, hipStream_t stream) {
  hipLaunchKernelGGL(EmbedKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     table, type, tokens, res, hidden, streams);
}

void RmsNormRows(const float* x, const float* gamma, float* out,
                 std::uint32_t n_rows, std::uint32_t dim, std::uint32_t groups,
                 float eps, hipStream_t stream) {
  // Every (row, group) pair is one block; the kernel recovers the group
  // from the block index to pick its gamma slice.
  const std::uint32_t group_dim = dim / groups;
  // Hidden (2560) and HC (10240) rows: every load in flight before the sum.
  if (group_dim == 10 * kThreads) {
    hipLaunchKernelGGL(RmsNormCachedKernel<10>, dim3(n_rows * groups),
                       dim3(kThreads), 0, stream, x, gamma, out, group_dim,
                       groups, eps);
    return;
  }
  if (group_dim == 40 * kThreads) {
    hipLaunchKernelGGL(RmsNormCachedKernel<40>, dim3(n_rows * groups),
                       dim3(kThreads), 0, stream, x, gamma, out, group_dim,
                       groups, eps);
    return;
  }
  hipLaunchKernelGGL(RmsNormKernel, dim3(n_rows * groups), dim3(kThreads), 0,
                     stream, x, gamma, out, group_dim, groups, eps);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
