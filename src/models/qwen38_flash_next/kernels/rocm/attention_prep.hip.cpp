// Attention inputs and K/V writes: Q/gate unpack, PrepareAttention, RoPE,
// K/V stores, KVarN rotation and quantization, and the fused query and
// gated-context row passes.
// HIP kernels follow the layouts and operator formulas in reference.cpp.

#include "src/models/qwen38_flash_next/kernels/rocm/kernels_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

namespace {

__global__ void UnpackQGateKernel(const float* qg, std::uint32_t qg_stride,
                                  float* q, float* gate, float* k, float* v,
                                  std::uint32_t heads, std::uint32_t d,
                                  std::uint32_t kv_width) {
  const std::uint32_t t = blockIdx.x;
  const std::size_t width = static_cast<std::size_t>(heads) * d;
  const float* row = qg + static_cast<std::size_t>(t) * qg_stride;
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x) {
    const std::uint32_t h = i / d;
    const std::uint32_t j = i % d;
    q[t * width + i] = row[h * 2 * d + j];
    gate[t * width + i] = row[h * 2 * d + d + j];
  }
  // A stacked [q|gate ; k ; v] projection carries k and v after the heads.
  if (k != nullptr) {
    for (std::size_t i = threadIdx.x; i < kv_width; i += blockDim.x) {
      k[t * kv_width + i] = row[2 * width + i];
      v[t * kv_width + i] = row[2 * width + kv_width + i];
    }
  }
}

// Keep each head's original eight-wave reduction. Heads in a block share
// rotary angles; normalization statistics remain independent.
template<std::uint32_t kHeadsPerBlock, bool kYarn>
__global__ void PrepareAttentionKernel(
    const float* __restrict__ packed, std::uint32_t stride,
    const float* __restrict__ q_gamma, const float* __restrict__ k_gamma,
    float* __restrict__ q, float* __restrict__ gate,
    __half* __restrict__ k_cache, __half* __restrict__ v_cache,
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t d,
    std::uint32_t rotary, const std::uint32_t* start_pos, float theta,
    float eps, const qwen::vision::DeviceRope* rope,
    [[maybe_unused]] YarnRope yarn, const std::uint32_t* kv_blocks,
    const float* __restrict__ k_src, const float* __restrict__ v_src) {
  __shared__ float partial[kHeadsPerBlock][8];
  __shared__ float norm[kHeadsPerBlock][256];
  const std::uint32_t t = blockIdx.x, first = blockIdx.y * kHeadsPerBlock,
                      tid = threadIdx.x;
  const std::uint32_t lane = tid % 32, wave = tid / 32;
  const std::uint32_t live = min(kHeadsPerBlock, heads + kv_heads - first);
  const std::size_t width = std::size_t(heads) * d,
                    kv_width = std::size_t(kv_heads) * d;
  const float* row = packed + std::size_t(t) * stride;
  const std::size_t kv_row = KvWriteRow(kv_blocks, *start_pos + t);
  float values[kHeadsPerBlock];
#pragma unroll
  for (std::uint32_t j = 0; j < kHeadsPerBlock; ++j) {
    const std::uint32_t h = first + j;
    const bool query = h < heads;
    const std::uint32_t head = query ? h : h - heads;
    // Split projections (k_src/v_src: [t][kv_heads * d] rows) carry k and
    // v outside the packed [q|gate] row.
    const float* src = query ? row + head * 2 * d
                       : k_src != nullptr
                           ? k_src + std::size_t(t) * kv_width + head * d
                           : row + 2 * width + head * d;
    values[j] = j < live && tid < d ? src[tid] : 0.0f;
    float ss = values[j] * values[j];
    // Match the separate RMS kernel's rounded square before its reduction.
    // Otherwise fast-math can contract it with the first shuffle addition.
    asm volatile("" : "+v"(ss));
    ss = WaveSum(ss);
    if (lane == 0)
      partial[j][wave] = ss;
  }
  __syncthreads();
#pragma unroll
  for (std::uint32_t j = 0; j < kHeadsPerBlock; ++j) {
    if (j >= live)
      continue;
    const std::uint32_t h = first + j;
    const bool query = h < heads;
    const std::uint32_t head = query ? h : h - heads;
    float total = 0.0f;
    for (int w = 0; w < static_cast<int>(blockDim.x / warpSize); ++w)
      total += partial[j][w];
    const float scale = rsqrtf(total / static_cast<float>(d) + eps);
    const float* gamma = query ? q_gamma : k_gamma;
    if (tid < d) {
      norm[j][tid] = values[j] * scale * (gamma != nullptr ? gamma[tid] : 1.0f);
      if (query)
        gate[std::size_t(t) * width + head * d + tid] =
            row[head * 2 * d + d + tid];
      else
        v_cache[kv_row * kv_width + head * d + tid] = __float2half(
            v_src != nullptr ? v_src[std::size_t(t) * kv_width + head * d + tid]
                             : row[2 * width + kv_width + head * d + tid]);
    }
  }
  __syncthreads();
  const std::uint32_t half = rotary / 2;
  float sine = 0.0f, cosine = 0.0f;
  if (tid < half) {
    float freq = powf(
        theta, -2.0f * static_cast<float>(tid) / static_cast<float>(rotary));
    if constexpr (kYarn)
      freq = YarnFrequency(freq, tid, yarn);
    sincosf(qwen::vision::RopePosition(rope, *start_pos + t, tid) * freq, &sine,
            &cosine);
    if constexpr (kYarn)
      YarnScale(sine, cosine, yarn);
  }
#pragma unroll
  for (std::uint32_t j = 0; j < kHeadsPerBlock; ++j) {
    if (j >= live)
      continue;
    const std::uint32_t h = first + j;
    const bool query = h < heads;
    const std::uint32_t head = query ? h : h - heads;
    if (tid < half) {
      const float a = norm[j][tid], b = norm[j][tid + half];
      const float lo = __fmaf_rn(a, cosine, -__fmul_rn(b, sine)),
                  hi = __fmaf_rn(a, sine, __fmul_rn(b, cosine));
      if (query) {
        q[std::size_t(t) * width + head * d + tid] = lo;
        q[std::size_t(t) * width + head * d + tid + half] = hi;
      } else {
        k_cache[kv_row * kv_width + head * d + tid] = __float2half(lo);
        k_cache[kv_row * kv_width + head * d + tid + half] = __float2half(hi);
      }
    }
    if (tid >= rotary && tid < d) {
      if (query)
        q[std::size_t(t) * width + head * d + tid] = norm[j][tid];
      else
        k_cache[kv_row * kv_width + head * d + tid] =
            __float2half(norm[j][tid]);
    }
  }
}

template<bool kYarn>
__global__ void RopeKernel(float* x, std::uint32_t heads, std::uint32_t d,
                           std::uint32_t rotary_dim,
                           const std::uint32_t* start_pos, float theta,
                           const qwen::vision::DeviceRope* rope,
                           [[maybe_unused]] YarnRope yarn) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t half = rotary_dim / 2;
  for (std::uint32_t idx = threadIdx.x; idx < heads * half; idx += blockDim.x) {
    const std::uint32_t h = idx / half;
    const std::uint32_t i = idx % half;
    float* v = x + (static_cast<std::size_t>(t) * heads + h) * d;
    float freq = powf(
        theta, -2.0f * static_cast<float>(i) / static_cast<float>(rotary_dim));
    if constexpr (kYarn)
      freq = YarnFrequency(freq, i, yarn);
    float s = 0.0f;
    float c = 0.0f;
    sincosf(qwen::vision::RopePosition(rope, *start_pos + t, i) * freq, &s, &c);
    if constexpr (kYarn)
      YarnScale(s, c, yarn);
    const float a = v[i];
    const float b = v[i + half];
    if constexpr (kYarn) {
      // Explicit rounding, as in PrepareAttention and the GEMM epilogue:
      // fast-math may otherwise contract a * s + b * c either way round.
      v[i] = __fmaf_rn(a, c, -__fmul_rn(b, s));
      v[i + half] = __fmaf_rn(a, s, __fmul_rn(b, c));
    } else {
      v[i] = a * c - b * s;
      v[i + half] = a * s + b * c;
    }
  }
}

__global__ void StoreKvKernel(const float* src, __half* cache,
                              std::uint32_t row_dim,
                              const std::uint32_t* start_pos,
                              const std::uint32_t* kv_blocks) {
  const std::uint32_t t = blockIdx.x;
  const std::size_t row = KvWriteRow(kv_blocks, *start_pos + t);
  for (std::uint32_t i = threadIdx.x; i < row_dim; i += blockDim.x) {
    cache[row * row_dim + i] =
        __float2half(src[static_cast<std::size_t>(t) * row_dim + i]);
  }
}

/// KVarN store rotation, in place: one block of 64 x kv_heads threads per
/// position, wave 2h + 0 rotating head h's K row and 2h + 1 its V row, lane l
/// elements 8l..8l+7. The rows already hold the FP16 values (the writers'
/// rounding), so this is the KVarN write kernel's FP16 round, rotation and
/// FP16 store, bit for bit.
/// Row of position `pos` in a KVarN FP16 sink/tail ring, or in a staging
/// buffer whose row 0 holds `staging_first`.
__device__ __forceinline__ std::size_t KvarnFp16RowOf(
    std::uint32_t pos, std::uint32_t staging_first) {
  return staging_first == kKvarnRingRows ? hip::KvarnFp16Row(pos)
                                         : pos - staging_first;
}

__launch_bounds__(256) __global__
    void RotateKvRowsKernel(__half* __restrict__ k_rows,
                            __half* __restrict__ v_rows, std::uint32_t kv_heads,
                            const std::uint32_t* __restrict__ start_pos,
                            std::uint32_t staging_first) {
  const std::uint32_t wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const std::uint32_t head = wave / 2;
  const bool key = (wave & 1) == 0;
  const std::size_t row =
      KvarnFp16RowOf(*start_pos + blockIdx.x, staging_first);
  __half* p = (key ? k_rows : v_rows) +
              (row * kv_heads + head) * hip::kKvarnHeadDim + 8 * lane;
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const auto* in = reinterpret_cast<const __half*>(&raw);
  float y[8];
#pragma unroll
  for (std::uint32_t j = 0; j < 8; ++j)
    y[j] = __half2float(in[j]);
  hip::kvarn_detail::Fwht256(y, lane);
  _Float16 h[8];
#pragma unroll
  for (std::uint32_t j = 0; j < 8; ++j)
    h[j] = static_cast<_Float16>(y[j] * 0.0625F);
  *reinterpret_cast<uint4*>(p) = *reinterpret_cast<const uint4*>(h);
}

/// One wave per FP32 row of 256: the normalized Hadamard in place.
__launch_bounds__(256) __global__
    void RotateHeadRowsKernel(float* __restrict__ x, std::uint32_t rows) {
  const std::uint32_t row = blockIdx.x * 8 + threadIdx.x / 32;
  const std::uint32_t lane = threadIdx.x % 32;
  if (row >= rows)
    return;
  float* p = x + std::size_t{row} * hip::kKvarnHeadDim + 8 * lane;
  float y[8];
  const float4 lo = reinterpret_cast<const float4*>(p)[0];
  const float4 hi = reinterpret_cast<const float4*>(p)[1];
  y[0] = lo.x, y[1] = lo.y, y[2] = lo.z, y[3] = lo.w;
  y[4] = hi.x, y[5] = hi.y, y[6] = hi.z, y[7] = hi.w;
  hip::kvarn_detail::Fwht256(y, lane);
  reinterpret_cast<float4*>(p)[0] = make_float4(y[0] * 0.0625F, y[1] * 0.0625F,
                                                y[2] * 0.0625F, y[3] * 0.0625F);
  reinterpret_cast<float4*>(p)[1] = make_float4(y[4] * 0.0625F, y[5] * 0.0625F,
                                                y[6] * 0.0625F, y[7] * 0.0625F);
}

/// One k4v4 token-major record tile per block: (KV head, group, layer, K or
/// V), from the rotated FP16 tail ring or a staging buffer
/// (hip::kvarn_detail::KvarnQuantizeTile).
template<int Bits>
__launch_bounds__(512) __global__ void KvarnQuantizeGroupsKernel(
    const __half* __restrict__ k_rows, const __half* __restrict__ v_rows,
    std::size_t fp16_layer_elements, std::uint8_t* __restrict__ records,
    std::size_t records_layer_pitch, const std::uint32_t* kv_blocks,
    std::uint32_t kv_heads, std::uint32_t first_group,
    std::uint32_t staging_first) {
  const std::uint32_t kv = blockIdx.x;
  const std::uint32_t group = first_group + blockIdx.y;
  const std::uint32_t layer = blockIdx.z >> 1;
  const bool key = (blockIdx.z & 1) == 0;
  const std::size_t row_width = std::size_t{kv_heads} * hip::kKvarnHeadDim;
  const __half* src =
      (key ? k_rows : v_rows) + layer * fp16_layer_elements +
      KvarnFp16RowOf(group * hip::kKvarnGroupTokens, staging_first) *
          row_width +
      kv * hip::kKvarnHeadDim;
  std::uint8_t* record =
      records + layer * records_layer_pitch +
      KvarnRecordOffset<Bits>(kv_blocks, kv_heads, group, kv);
  hip::kvarn_detail::KvarnQuantizeTile<Bits, kKvarnOrder, Bits>(src, row_width,
                                                                record, key);
}

/// KvarnQuantizePassBoundary: the group ending inside the pass, if any.
template<int Bits>
__launch_bounds__(512) __global__ void KvarnQuantizeBoundaryKernel(
    const __half* __restrict__ k_rows, const __half* __restrict__ v_rows,
    std::uint8_t* __restrict__ records, const std::uint32_t* kv_blocks,
    std::uint32_t kv_heads, const std::uint32_t* __restrict__ start_pos,
    std::uint32_t n_tokens) {
  const std::uint32_t start = *start_pos;
  const std::uint32_t last_end = max(
      hip::kKvarnSinkTokens,
      (start + n_tokens - 1) >> hip::kKvarnGroupShift << hip::kKvarnGroupShift);
  if (!(last_end > start && last_end > hip::kKvarnSinkTokens))
    return;
  const std::uint32_t group = (last_end >> hip::kKvarnGroupShift) - 1;
  const std::uint32_t kv = blockIdx.x;
  const bool key = blockIdx.z == 0;
  const std::size_t row_width = std::size_t{kv_heads} * hip::kKvarnHeadDim;
  const __half* src =
      (key ? k_rows : v_rows) +
      hip::KvarnFp16Row(group * hip::kKvarnGroupTokens) * row_width +
      kv * hip::kKvarnHeadDim;
  hip::kvarn_detail::KvarnQuantizeTile<Bits, kKvarnOrder, Bits>(
      src, row_width,
      records + KvarnRecordOffset<Bits>(kv_blocks, kv_heads, group, kv), key);
}

}  // namespace

void UnpackQGate(const float* qg, std::uint32_t qg_stride, float* q,
                 float* gate, float* k, float* v, std::uint32_t n_tokens,
                 std::uint32_t heads, std::uint32_t d, std::uint32_t kv_width,
                 hipStream_t stream) {
  hipLaunchKernelGGL(UnpackQGateKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, qg, qg_stride, q, gate, k, v, heads, d, kv_width);
}

bool PrepareAttention(const float* packed, std::uint32_t stride,
                      const float* q_gamma, const float* k_gamma, float* q,
                      float* gate, __half* k_cache, __half* v_cache,
                      std::uint32_t n_tokens, std::uint32_t heads,
                      std::uint32_t kv_heads, std::uint32_t d,
                      std::uint32_t rotary_dim, const std::uint32_t* start_pos,
                      float theta, float eps, hipStream_t stream,
                      const qwen::vision::DeviceRope* rope, bool prefill,
                      const YarnRope& yarn, const std::uint32_t* kv_blocks,
                      const float* k_src, const float* v_src) {
  const bool split = k_src != nullptr || v_src != nullptr;
  if (d == 0 || d > 256 || rotary_dim == 0 || rotary_dim > d ||
      rotary_dim % 2 != 0 || kv_heads == 0 ||
      (split && (k_src == nullptr || v_src == nullptr)) ||
      stride <
          static_cast<std::size_t>(2) * (heads + (split ? 0 : kv_heads)) * d) {
    return false;
  }
  if (n_tokens == 0)
    return true;
  if (yarn.enabled != 0) {
    if (!prefill && n_tokens < 32) {
      hipLaunchKernelGGL(
          (PrepareAttentionKernel<1, true>), dim3(n_tokens, heads + kv_heads),
          dim3(kThreads), 0, stream, packed, stride, q_gamma, k_gamma, q, gate,
          k_cache, v_cache, heads, kv_heads, d, rotary_dim, start_pos, theta,
          eps, rope, yarn, kv_blocks, k_src, v_src);
    } else {
      hipLaunchKernelGGL((PrepareAttentionKernel<4, true>),
                         dim3(n_tokens, (heads + kv_heads + 3) / 4),
                         dim3(kThreads), 0, stream, packed, stride, q_gamma,
                         k_gamma, q, gate, k_cache, v_cache, heads, kv_heads, d,
                         rotary_dim, start_pos, theta, eps, rope, yarn,
                         kv_blocks, k_src, v_src);
    }
  } else {
    if (!prefill && n_tokens < 32) {
      hipLaunchKernelGGL(
          (PrepareAttentionKernel<1, false>), dim3(n_tokens, heads + kv_heads),
          dim3(kThreads), 0, stream, packed, stride, q_gamma, k_gamma, q, gate,
          k_cache, v_cache, heads, kv_heads, d, rotary_dim, start_pos, theta,
          eps, rope, yarn, kv_blocks, k_src, v_src);
    } else {
      hipLaunchKernelGGL((PrepareAttentionKernel<4, false>),
                         dim3(n_tokens, (heads + kv_heads + 3) / 4),
                         dim3(kThreads), 0, stream, packed, stride, q_gamma,
                         k_gamma, q, gate, k_cache, v_cache, heads, kv_heads, d,
                         rotary_dim, start_pos, theta, eps, rope, yarn,
                         kv_blocks, k_src, v_src);
    }
  }
  return true;
}

void PrepareAttentionKv(float* k, const float* v, const float* k_gamma,
                        __half* k_cache, __half* v_cache,
                        std::uint32_t n_tokens, std::uint32_t kv_heads,
                        std::uint32_t d, std::uint32_t rotary_dim,
                        const std::uint32_t* start_pos, float theta, float eps,
                        hipStream_t stream,
                        const qwen::vision::DeviceRope* rope,
                        const YarnRope& yarn, const std::uint32_t* kv_blocks) {
  RmsNormRows(k, k_gamma, k, n_tokens * kv_heads, d, 1, eps, stream);
  Rope(k, n_tokens, kv_heads, d, rotary_dim, start_pos, theta, stream, rope,
       yarn);
  StoreKv(k, k_cache, n_tokens, kv_heads * d, start_pos, stream, kv_blocks);
  StoreKv(v, v_cache, n_tokens, kv_heads * d, start_pos, stream, kv_blocks);
}

void Rope(float* x, std::uint32_t n_tokens, std::uint32_t heads,
          std::uint32_t d, std::uint32_t rotary_dim,
          const std::uint32_t* start_pos, float theta, hipStream_t stream,
          const qwen::vision::DeviceRope* rope, const YarnRope& yarn) {
  if (yarn.enabled != 0)
    hipLaunchKernelGGL(RopeKernel<true>, dim3(n_tokens), dim3(kThreads), 0,
                       stream, x, heads, d, rotary_dim, start_pos, theta, rope,
                       yarn);
  else
    hipLaunchKernelGGL(RopeKernel<false>, dim3(n_tokens), dim3(kThreads), 0,
                       stream, x, heads, d, rotary_dim, start_pos, theta, rope,
                       yarn);
}

void StoreKv(const float* src, __half* cache, std::uint32_t n_tokens,
             std::uint32_t row_dim, const std::uint32_t* start_pos,
             hipStream_t stream, const std::uint32_t* kv_blocks) {
  hipLaunchKernelGGL(StoreKvKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     src, cache, row_dim, start_pos, kv_blocks);
}

void RotateKvRows(__half* k_rows, __half* v_rows, std::uint32_t n_tokens,
                  std::uint32_t kv_heads, const std::uint32_t* start_pos,
                  hipStream_t stream, std::uint32_t staging_first) {
  if (n_tokens == 0 || kv_heads == 0 || kv_heads > 4)
    return;
  hipLaunchKernelGGL(RotateKvRowsKernel, dim3(n_tokens), dim3(64 * kv_heads), 0,
                     stream, k_rows, v_rows, kv_heads, start_pos,
                     staging_first);
}

/// AttentionQueryRows: one (token, head) row of 256 per block.
template<bool kYarn, bool kRotate>
__launch_bounds__(256) __global__ void AttentionQueryRowsKernel(
    const float* __restrict__ qg, std::uint32_t qg_stride,
    const float* __restrict__ gamma, float* __restrict__ q,
    float* __restrict__ gate, std::uint32_t heads, std::uint32_t rotary_dim,
    const std::uint32_t* start_pos, float theta, float eps,
    const qwen::vision::DeviceRope* rope, [[maybe_unused]] YarnRope yarn) {
  constexpr std::uint32_t d = hip::kKvarnHeadDim;
  __shared__ float shared[32];
  __shared__ float s_row[d];
  const std::uint32_t t = blockIdx.x / heads;
  const std::uint32_t h = blockIdx.x % heads;
  const std::uint32_t i = threadIdx.x;
  const float* src =
      qg + (static_cast<std::size_t>(t) * qg_stride) + (h * 2 * d);
  const std::size_t row = (static_cast<std::size_t>(t) * heads) + h;
  if (gate != nullptr)
    gate[(row * d) + i] = src[d + i];
  // RmsNormKernel (one element per thread).
  const float x = src[i];
  // The square stays its own rounding: fast-math would otherwise fold the
  // first lane exchange into an FMA with the partner's square.
  const float ss = BlockSum(Materialize(x * x), shared);
  const float scale = rsqrtf(ss / static_cast<float>(d) + eps);
  // RmsNormKernel's (x * scale) * gamma.
  s_row[i] = Materialize(Materialize(x * scale) * gamma[i]);
  __syncthreads();
  // RopeKernel's pair (i, i + half).
  const std::uint32_t half = rotary_dim / 2;
  if (i < half) {
    float freq = powf(
        theta, -2.0f * static_cast<float>(i) / static_cast<float>(rotary_dim));
    if constexpr (kYarn)
      freq = YarnFrequency(freq, i, yarn);
    float s = 0.0f;
    float c = 0.0f;
    sincosf(qwen::vision::RopePosition(rope, *start_pos + t, i) * freq, &s, &c);
    if constexpr (kYarn)
      YarnScale(s, c, yarn);
    const float a = s_row[i];
    const float b = s_row[i + half];
    // RopeKernel's compiled form in both variants: fma(a, c, -(b s)) and
    // fma(a, s, b c).
    s_row[i] = __fmaf_rn(a, c, -__fmul_rn(b, s));
    s_row[i + half] = __fmaf_rn(a, s, __fmul_rn(b, c));
  }
  __syncthreads();
  float* dst = q + (row * d);
  if constexpr (kRotate) {
    // RotateHeadRowsKernel: one wave, eight consecutive values per lane.
    if (i >= 32)
      return;
    float y[8];
#pragma unroll
    for (int k = 0; k < 8; ++k)
      y[k] = s_row[(8 * i) + k];
    hip::kvarn_detail::Fwht256(y, i);
    reinterpret_cast<float4*>(dst + (8 * i))[0] = make_float4(
        y[0] * 0.0625F, y[1] * 0.0625F, y[2] * 0.0625F, y[3] * 0.0625F);
    reinterpret_cast<float4*>(dst + (8 * i))[1] = make_float4(
        y[4] * 0.0625F, y[5] * 0.0625F, y[6] * 0.0625F, y[7] * 0.0625F);
  } else {
    dst[i] = s_row[i];
  }
}

bool AttentionQueryRows(const float* qg, std::uint32_t qg_stride,
                        const float* gamma, float* q, float* gate,
                        std::uint32_t n_tokens, std::uint32_t heads,
                        std::uint32_t d, std::uint32_t rotary_dim,
                        const std::uint32_t* start_pos, float theta, float eps,
                        bool rotate, hipStream_t stream,
                        const qwen::vision::DeviceRope* rope,
                        const YarnRope& yarn) {
  if (d != hip::kKvarnHeadDim || rotary_dim > d || rotary_dim % 2 != 0 ||
      gamma == nullptr || n_tokens == 0)
    return false;
  const auto launch = [&](auto kernel) {
    hipLaunchKernelGGL(kernel, dim3(n_tokens * heads), dim3(256), 0, stream, qg,
                       qg_stride, gamma, q, gate, heads, rotary_dim, start_pos,
                       theta, eps, rope, yarn);
  };
  if (yarn.enabled != 0)
    rotate ? launch(AttentionQueryRowsKernel<true, true>)
           : launch(AttentionQueryRowsKernel<true, false>);
  else
    rotate ? launch(AttentionQueryRowsKernel<false, true>)
           : launch(AttentionQueryRowsKernel<false, false>);
  return true;
}

/// GateRotatedContext: one wave per context row of 256.
/// Non-empty `slabs` also receive the stored row rotated for the EXL3 output
/// projection (lane l of segment s takes elements s * 128 + 4l + j, held by
/// lane s * 16 + l / 2 at 4 * (l % 2) + j).
__launch_bounds__(256) __global__
    void GateRotatedContextKernel(float* __restrict__ ctx,
                                  const float* __restrict__ qg,
                                  std::uint32_t qg_stride, std::uint32_t heads,
                                  std::uint32_t rows, Exl3InputSlabs slabs) {
  constexpr std::uint32_t d = hip::kKvarnHeadDim;
  const std::uint32_t row = blockIdx.x * 8 + threadIdx.x / 32;
  const std::uint32_t lane = threadIdx.x % 32;
  if (row >= rows)
    return;
  float* p = ctx + std::size_t{row} * d + 8 * lane;
  const float* g = qg + (static_cast<std::size_t>(row / heads) * qg_stride) +
                   ((row % heads) * 2 * d) + d + 8 * lane;
  float y[8];
  const float4 lo = reinterpret_cast<const float4*>(p)[0];
  const float4 hi = reinterpret_cast<const float4*>(p)[1];
  y[0] = lo.x, y[1] = lo.y, y[2] = lo.z, y[3] = lo.w;
  y[4] = hi.x, y[5] = hi.y, y[6] = hi.z, y[7] = hi.w;
  hip::kvarn_detail::Fwht256(y, lane);
  float out[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    // RotateHeadRowsKernel's stored value, then SigmoidMulKernel's product.
    const float rotated = Materialize(y[k] * 0.0625F);
    out[k] = rotated * Materialize(SigmoidF(g[k]));
  }
  reinterpret_cast<float4*>(p)[0] = make_float4(out[0], out[1], out[2], out[3]);
  reinterpret_cast<float4*>(p)[1] = make_float4(out[4], out[5], out[6], out[7]);
  if (slabs.count == 0)
    return;
  static_assert(d == 2 * hip::exl3::kSegment);
#pragma unroll
  for (std::uint32_t segment = 0; segment < 2; ++segment) {
    const int source = static_cast<int>(segment * 16 + lane / 2);
    float v[4];
#pragma unroll
    for (std::uint32_t j = 0; j < 4; ++j) {
      const float low = __shfl(out[j], source, 32);
      const float high = __shfl(out[4 + j], source, 32);
      v[j] = lane % 2 != 0 ? high : low;
    }
    RotateIntoSlabs(
        v, lane, (row % heads) * d + segment * hip::exl3::kSegment + 4 * lane,
        row / heads, slabs);
  }
}

bool GateRotatedContext(float* ctx, const float* qg, std::uint32_t qg_stride,
                        std::uint32_t n_tokens, std::uint32_t heads,
                        hipStream_t stream, const Exl3InputSlabs& slabs) {
  const std::uint32_t rows = n_tokens * heads;
  if (rows == 0)
    return false;
  const bool rotate = slabs.count == 1 &&
                      slabs.k == heads * hip::kKvarnHeadDim &&
                      n_tokens <= slabs.pass_rows;
  hipLaunchKernelGGL(GateRotatedContextKernel, dim3((rows + 7) / 8), dim3(256),
                     0, stream, ctx, qg, qg_stride, heads, rows,
                     rotate ? slabs : Exl3InputSlabs{});
  return rotate;
}

void RotateHeadRows(float* x, std::uint32_t rows, hipStream_t stream) {
  if (rows == 0)
    return;
  hipLaunchKernelGGL(RotateHeadRowsKernel, dim3((rows + 7) / 8), dim3(256), 0,
                     stream, x, rows);
}

void KvarnQuantizePassBoundary(const __half* k_fp16, const __half* v_fp16,
                               std::uint8_t* records,
                               const std::uint32_t* kv_blocks,
                               std::uint32_t kv_heads,
                               const std::uint32_t* start_pos,
                               std::uint32_t n_tokens, hipStream_t stream,
                               int bits) {
  if (n_tokens == 0)
    return;
  if (bits == 6) {
    hipLaunchKernelGGL(KvarnQuantizeBoundaryKernel<6>, dim3(kv_heads, 1, 2),
                       dim3(512), 0, stream, k_fp16, v_fp16, records, kv_blocks,
                       kv_heads, start_pos, n_tokens);
  } else if (bits == 4) {
    hipLaunchKernelGGL(KvarnQuantizeBoundaryKernel<4>, dim3(kv_heads, 1, 2),
                       dim3(512), 0, stream, k_fp16, v_fp16, records, kv_blocks,
                       kv_heads, start_pos, n_tokens);
  } else {
    throw std::invalid_argument("KVarN bits must be 4 or 6");
  }
}

void KvarnQuantizeGroups(const __half* k_rows, const __half* v_rows,
                         std::size_t fp16_layer_elements, std::uint8_t* records,
                         std::size_t records_layer_pitch,
                         const std::uint32_t* kv_blocks, std::uint32_t kv_heads,
                         std::uint32_t layers, std::uint32_t first_group,
                         std::uint32_t groups, hipStream_t stream,
                         std::uint32_t staging_first, int bits) {
  if (groups == 0 || layers == 0)
    return;
  if (first_group == 0)
    throw std::invalid_argument("KVarN group 0 is the FP16 sink");
  if (bits == 6) {
    hipLaunchKernelGGL(
        KvarnQuantizeGroupsKernel<6>, dim3(kv_heads, groups, 2 * layers),
        dim3(512), 0, stream, k_rows, v_rows, fp16_layer_elements, records,
        records_layer_pitch, kv_blocks, kv_heads, first_group, staging_first);
  } else if (bits == 4) {
    hipLaunchKernelGGL(
        KvarnQuantizeGroupsKernel<4>, dim3(kv_heads, groups, 2 * layers),
        dim3(512), 0, stream, k_rows, v_rows, fp16_layer_elements, records,
        records_layer_pitch, kv_blocks, kv_heads, first_group, staging_first);
  } else {
    throw std::invalid_argument("KVarN bits must be 4 or 6");
  }
}

}  // namespace gufo::models::qwen38_flash_next::rocm
