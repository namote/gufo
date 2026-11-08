// Frozen copy of the four-lane HC combine of
// src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp,
// HcCombineVec4Kernel, at commit bec85727 (plain residual stores), with its
// device helpers, verbatim apart from the kernel's name and its dropped
// kInject specialization (a __half-route option the FP32 route never
// instantiates). Production now streams the wide prefill residual write-back
// with non-temporal stores, which must not change a byte; the HC prefill
// bench (tools/qwen38_flash_next/bench/flash_next_hc_prefill.hip) compares the production
// combine's bytes against this copy and times both. Include from a HIP TU
// compiled with the production library's flags (-O3 -ffast-math
// -fno-finite-math-only).
#ifndef GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_HC_COMBINE_REFERENCE_HPP_
#define GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_HC_COMBINE_REFERENCE_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "src/models/qwen38_flash_next/kernels/rocm/exl3_prefill_combine.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::tests::hc_combine_reference {

using models::qwen38_flash_next::rocm::DeferredResidual;
using models::qwen38_flash_next::rocm::Exl3PrefillMoeRows;
using models::qwen38_flash_next::rocm::Exl3PrefillMoeSegment;

constexpr unsigned kThreads = 256;

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float Materialize(float v) {
  asm volatile("" : "+v"(v));
  return v;
}

/// Lane i receives lane (i ^ offset)'s value, as __shfl_xor(v, offset) does
/// across a wave32. On gfx1151 it is a DPP row_xmask (offsets below 16) or
/// v_permlanex16 (16): a register move instead of an LDS crossbar round trip
/// (ds_bpermute). Only the transport differs, so reductions stay bitwise;
/// every lane must be active, as the shuffle's callers already require.
__device__ __forceinline__ int XorLane(int v, int offset) {
#if defined(__gfx1151__)
  switch (offset) {
    case 1:
      return __builtin_amdgcn_update_dpp(0, v, 0x161, 0xF, 0xF, false);
    case 2:
      return __builtin_amdgcn_update_dpp(0, v, 0x162, 0xF, 0xF, false);
    case 4:
      return __builtin_amdgcn_update_dpp(0, v, 0x164, 0xF, 0xF, false);
    case 8:
      return __builtin_amdgcn_update_dpp(0, v, 0x168, 0xF, 0xF, false);
    case 16:
      return __builtin_amdgcn_permlanex16(v, v, 0x76543210, 0xfedcba98, false,
                                          false);
  }
#endif
  return __shfl_xor(v, offset);
}

__device__ __forceinline__ unsigned XorLane(unsigned v, int offset) {
  return static_cast<unsigned>(XorLane(static_cast<int>(v), offset));
}

__device__ __forceinline__ float XorLane(float v, int offset) {
  return __builtin_bit_cast(float, XorLane(__builtin_bit_cast(int, v), offset));
}

/// Sum over one wave; every lane receives the total. The wave-per-row
/// kernels split rows across lanes (gfx1151 runs wave32).
__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += XorLane(v, offset);
  }
  return v;
}

__device__ __forceinline__ float4 CombineAdd(float4 r, float4 b, float w) {
  return float4{__fmaf_rn(b.x, w, r.x), __fmaf_rn(b.y, w, r.y),
                __fmaf_rn(b.z, w, r.z), __fmaf_rn(b.w, w, r.w)};
}

__device__ __forceinline__ float4 HcNorm4(float4 v, float sc, float4 gm) {
  // The result is pinned too: a consumer's own products must not fold
  // into this one.
  return float4{Materialize(v.x * Materialize(sc * gm.x)),
                Materialize(v.y * Materialize(sc * gm.y)),
                Materialize(v.z * Materialize(sc * gm.z)),
                Materialize(v.w * Materialize(sc * gm.w))};
}

constexpr std::size_t kQ8ActTileTokens = 16;
constexpr std::size_t kQ8ActTileBytes = 576;
constexpr std::size_t kQ8ActScaleOffset = 512;
__device__ __forceinline__ std::int8_t* Q8ActTile(void* base,
                                                  std::size_t num_blocks,
                                                  std::size_t tt,
                                                  std::size_t kb) {
  return static_cast<std::int8_t*>(base) +
         (((tt * num_blocks) + kb) * kQ8ActTileBytes);
}

template<typename XnT, bool kDefer = false, bool kMoe = false>
__global__ void OldHcCombineVec4Kernel(
    float* res, const float* block_out, const float* inject,
    std::uint32_t inject_parts, const float* gamma, XnT* xn, void* xn_q8,
    std::uint32_t hidden, float eps, float* defer_w,
    const float* inject_w = nullptr, float* inject_out = nullptr,
    __half* xn_half = nullptr, float* xn_scale = nullptr,
    Exl3PrefillMoeRows moe = {}, DeferredResidual prior = {}) {
  constexpr std::uint32_t kStreams = 4;
  constexpr std::uint32_t kMaxChunks = 10;  // 4 x 2560 / 4 / kThreads
  constexpr std::uint32_t kMaxHidden = kMaxChunks * kThreads;  // 2560
  __shared__ float shared[kStreams][kThreads / 32];
  __shared__ float s_moe[kMoe ? kMaxHidden : 1];
  const std::uint32_t t = blockIdx.x;
  const std::size_t hc_dim = static_cast<std::size_t>(kStreams) * hidden;
  float w[kStreams];
#pragma unroll
  for (std::uint32_t s = 0; s < kStreams; ++s) {
    float logit = 0.0f;
    for (std::uint32_t p = 0; p < inject_parts; ++p) {
      logit +=
          inject[(static_cast<std::size_t>(t) * kStreams + s) * inject_parts +
                 p];
    }
    w[s] = 2.0f * SigmoidF(logit / static_cast<float>(kStreams));
  }
  // A deferred update leaves the residual as it was and keeps the stream
  // weights; the next combine reapplies it (HcCombineMoeF16Kernel).
  if constexpr (kDefer) {
    if (threadIdx.x < kStreams) {
      defer_w[static_cast<std::size_t>(t) * kStreams + threadIdx.x] =
          w[threadIdx.x];
    }
  }
  float* dst = res + static_cast<std::size_t>(t) * hc_dim;
  const float* src = block_out + static_cast<std::size_t>(t) * hidden;
  if constexpr (kMoe) {
    const std::uint32_t lane = threadIdx.x & 31u;
    for (std::uint32_t segment = threadIdx.x >> 5u; segment * 128 < hidden;
         segment += kThreads / 32) {
      float o[4];
      Exl3PrefillMoeSegment(moe, t, segment, lane, o);
#pragma unroll
      for (std::uint32_t j = 0; j < 4; ++j)
        s_moe[(segment * 128) + (lane * 4) + j] = o[j];
    }
    __syncthreads();
    src = s_moe;
  }
  const std::uint32_t chunks = static_cast<std::uint32_t>(hc_dim / 4);
  float4 v[kMaxChunks];
  float ss[kStreams] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
  for (std::uint32_t c = 0; c < kMaxChunks; ++c) {
    const std::uint32_t e = (c * kThreads + threadIdx.x) * 4;
    v[c] = float4{0.0F, 0.0F, 0.0F, 0.0F};
    if (e < hc_dim) {
      const std::uint32_t s = e / hidden;
      const std::uint32_t i = e - (s * hidden);
      const float ws = w[s];
      float4 r = *reinterpret_cast<const float4*>(dst + e);
      const float4 b = *reinterpret_cast<const float4*>(src + i);
      if constexpr (std::is_same_v<XnT, float>) {
        // The previous combine's deferred update first, then this one, each
        // as that combine would have stored it.
        if (prior.block_out != nullptr)
          r = CombineAdd(
              r,
              *reinterpret_cast<const float4*>(
                  prior.block_out + (static_cast<std::size_t>(t) * hidden) + i),
              prior.w[(static_cast<std::size_t>(t) * kStreams) + s]);
        v[c] = CombineAdd(r, b, ws);
      } else {
        v[c] = float4{r.x + b.x * ws, r.y + b.y * ws, r.z + b.z * ws,
                      r.w + b.w * ws};
      }
      if constexpr (!kDefer) {
        *reinterpret_cast<float4*>(dst + e) = v[c];
      }
      const float sq =
          v[c].x * v[c].x + v[c].y * v[c].y + v[c].z * v[c].z + v[c].w * v[c].w;
#pragma unroll
      for (std::uint32_t k = 0; k < kStreams; ++k) {
        ss[k] += (k == s) ? sq : 0.0F;
      }
    }
  }
  if (gamma == nullptr) {
    return;
  }
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t wave = threadIdx.x >> 5u;
#pragma unroll
  for (std::uint32_t k = 0; k < kStreams; ++k) {
    ss[k] = WaveSum(ss[k]);
    if (lane == 0) {
      shared[k][wave] = ss[k];
    }
  }
  __syncthreads();
  float scale[kStreams];
#pragma unroll
  for (std::uint32_t k = 0; k < kStreams; ++k) {
    float total = 0.0F;
    for (std::uint32_t q = 0; q < kThreads / 32; ++q) {
      total += shared[k][q];
    }
    scale[k] = rsqrtf(total / static_cast<float>(hidden) + eps);
  }
  if (xn_scale != nullptr) {
#pragma unroll
    for (std::uint32_t k = 0; k < kStreams; ++k) {
      if (threadIdx.x == k)
        xn_scale[(static_cast<std::size_t>(t) * kStreams) + k] = scale[k];
    }
  }
  const std::size_t num_blocks = hc_dim / 32;
#pragma unroll
  for (std::uint32_t c = 0; c < kMaxChunks; ++c) {
    const std::uint32_t e = (c * kThreads + threadIdx.x) * 4;
    // Every lane of the wave takes part in the block reductions below, so
    // lanes past the row carry zeros instead of leaving.
    float4 n{0.0F, 0.0F, 0.0F, 0.0F};
    if (e < hc_dim) {
      const std::uint32_t s = e / hidden;
      const float sc = scale[s];
      const float4 gm = *reinterpret_cast<const float4*>(gamma + e);
      // The F32 route's consumers may recompute this norm (HcNorm4); the F16
      // route keeps its original expression.
      if constexpr (std::is_same_v<XnT, float>) {
        n = HcNorm4(v[c], sc, gm);
      } else {
        n = float4{v[c].x * sc * gm.x, v[c].y * sc * gm.y, v[c].z * sc * gm.z,
                   v[c].w * sc * gm.w};
      }
      XnT* out = xn + (static_cast<std::size_t>(t) * hc_dim) + e;
      if constexpr (std::is_same_v<XnT, float>) {
        if (xn_scale == nullptr)
          *reinterpret_cast<float4*>(out) = n;
        // The F16 copy a projection would otherwise narrow from `out`
        // (NarrowHalf4Kernel's conversion).
        // Streaming stores: the projection that reads these rows next
        // otherwise finds them dirty in the caches.
        if (xn_half != nullptr) {
          __half* h = xn_half + (static_cast<std::size_t>(t) * hc_dim) + e;
          using Half4Bits = std::uint16_t __attribute__((ext_vector_type(4)));
          const Half4Bits bits{
              __half_as_ushort(__half(n.x)), __half_as_ushort(__half(n.y)),
              __half_as_ushort(__half(n.z)), __half_as_ushort(__half(n.w))};
          __builtin_nontemporal_store(bits, reinterpret_cast<Half4Bits*>(h));
        }
      } else {
        const __half2 lo = __floats2half2_rn(n.x, n.y);
        const __half2 hi = __floats2half2_rn(n.z, n.w);
        *reinterpret_cast<__half2*>(out) = lo;
        *reinterpret_cast<__half2*>(out + 2) = hi;
      }
    }
    if (xn_q8 == nullptr) {
      continue;
    }
    // Eight lanes hold one 32-wide block: absmax over them, then each lane
    // stores its four codes as one word.
    float max_abs =
        fmaxf(fmaxf(fabsf(n.x), fabsf(n.y)), fmaxf(fabsf(n.z), fabsf(n.w)));
    for (int off = 4; off > 0; off >>= 1) {
      max_abs = fmaxf(max_abs, XorLane(max_abs, off));
    }
    if (e < hc_dim) {
      const float d = max_abs / 127.0F;
      const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
      const auto q0 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
          static_cast<std::int8_t>(roundf(n.x * id))));
      const auto q1 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
          static_cast<std::int8_t>(roundf(n.y * id))));
      const auto q2 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
          static_cast<std::int8_t>(roundf(n.z * id))));
      const auto q3 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
          static_cast<std::int8_t>(roundf(n.w * id))));
      const std::size_t kb = e / 32;
      const std::uint32_t pos = e % 32;  // 0, 4, ..., 28
      std::int8_t* tile =
          Q8ActTile(xn_q8, num_blocks, t / kQ8ActTileTokens, kb);
      const std::size_t tl = t % kQ8ActTileTokens;
      *reinterpret_cast<std::uint32_t*>(tile + ((pos >> 4u) * 256) + (tl * 16) +
                                        (pos & 15u)) =
          q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
      if (pos == 0) {
        *reinterpret_cast<float*>(tile + kQ8ActScaleOffset +
                                  (tl * sizeof(float))) = d;
      }
    }
  }
}

}  // namespace gufo::tests::hc_combine_reference

#endif  // GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_HC_COMBINE_REFERENCE_HPP_
