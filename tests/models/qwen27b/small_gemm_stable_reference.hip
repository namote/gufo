// Reference copy of Flash-Next's SmallGemmStable, extracted verbatim from
// gufo-f3-integ 03a6afba src/models/qwen38_flash_next/kernels/rocm/
// kernels.hip.cpp (SigmoidF, SiluF, Materialize, XorLane, Bf16ToF32,
// RowElement, RowBytes, WaveSum, SmallGemmKernel, LaunchSmallGemm,
// SmallGemmForType, SmallGemmStable), compiled with that library's flags
// (-O3 -ffast-math -fno-finite-math-only, tests/models/qwen27b/
// CMakeLists.txt) for plain_prefill_stable_test. Only WeightType is restated.
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>

#include "tests/models/qwen27b/small_gemm_stable_reference.hpp"

namespace gufo::test::small_gemm_reference {
namespace {

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float SiluF(float x) {
  return x * SigmoidF(x);
}

/// Keeps fast-math from folding a stored value's arithmetic into what
/// follows: the quantization must see exactly the float that was stored.
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

__device__ __forceinline__ float Bf16ToF32(std::uint16_t h) {
  return __uint_as_float(static_cast<std::uint32_t>(h) << 16);
}

/// Reads element i of a Q8_0 / F32 / BF16 / F16 row.
__device__ __forceinline__ float RowElement(const void* row, WeightType type,
                                            std::uint32_t i) {
  switch (type) {
    case WeightType::kF32:
      return static_cast<const float*>(row)[i];
    case WeightType::kBF16:
      return Bf16ToF32(static_cast<const std::uint16_t*>(row)[i]);
    case WeightType::kF16:
      return __half2float(static_cast<const __half*>(row)[i]);
    case WeightType::kQ8_0: {
      const auto* blk = static_cast<const std::uint8_t*>(row) + (i / 32) * 34;
      const __half d = *reinterpret_cast<const __half*>(blk);
      const auto q = static_cast<const std::int8_t*>(
          static_cast<const void*>(blk + 2))[i % 32];
      return __half2float(d) * static_cast<float>(q);
    }
  }
  return 0.0f;
}

__device__ __forceinline__ std::size_t RowBytes(WeightType type,
                                                std::uint32_t k) {
  switch (type) {
    case WeightType::kF32:
      return static_cast<std::size_t>(k) * 4;
    case WeightType::kBF16:
    case WeightType::kF16:
      return static_cast<std::size_t>(k) * 2;
    case WeightType::kQ8_0:
      return static_cast<std::size_t>(k / 32) * 34;
  }
  return 0;
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

/// tokens accumulate in registers, then a wave reduction per token.
constexpr unsigned kSmallGemmRows = 4;
template<WeightType type, unsigned tokens, bool grouped = false,
         bool kWhole = false, bool stable = false, bool kHcDown = false>
__global__ void SmallGemmKernel(const void* w, const float* x, float* out,
                                std::uint32_t m, std::uint32_t k) {
  static_assert(!kHcDown || (stable && type == WeightType::kF16));
  if constexpr (grouped) {
    x += std::size_t{blockIdx.y} * tokens * k;
    out += std::size_t{blockIdx.y} * tokens * m;
  }
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t row =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= m) {
    return;
  }
  const auto* wrow =
      static_cast<const std::uint8_t*>(w) + RowBytes(type, k) * row;
  // A lane's steps are loaded kSteps at a time so their reads overlap (one
  // step at a time leaves a decode-sized projection latency-bound); the
  // steps still accumulate one after another, in order.
  constexpr unsigned kSteps = stable || kWhole || tokens <= 2 ? 4 : 2;
  const std::uint32_t stride = warpSize * 4;
  float acc[tokens] = {};
  for (std::uint32_t i_base = lane * 4; i_base < k; i_base += stride * kSteps) {
    float wv[kSteps][4];
    float xv[kSteps][tokens][4];
#pragma unroll
    for (unsigned st = 0; st < kSteps; ++st) {
      const std::uint32_t i0 = i_base + st * stride;
      if constexpr (kWhole) {
        // Rows of whole passes (k a multiple of stride * kSteps): every step
        // is in range, so the four elements arrive as one load: 16 bytes for
        // F32, 8 bytes for F16/BF16 (widened exactly, as RowElement does).
        if constexpr (type == WeightType::kF32) {
          const float4 w4 = reinterpret_cast<const float4*>(
              static_cast<const void*>(wrow))[i0 / 4];
          wv[st][0] = w4.x;
          wv[st][1] = w4.y;
          wv[st][2] = w4.z;
          wv[st][3] = w4.w;
        } else {
          static_assert(type == WeightType::kF16 || type == WeightType::kBF16);
          const ushort4 h4 = reinterpret_cast<const ushort4*>(
              static_cast<const void*>(wrow))[i0 / 4];
          if constexpr (type == WeightType::kBF16) {
            wv[st][0] = Bf16ToF32(h4.x);
            wv[st][1] = Bf16ToF32(h4.y);
            wv[st][2] = Bf16ToF32(h4.z);
            wv[st][3] = Bf16ToF32(h4.w);
          } else {
            wv[st][0] = __half2float(__ushort_as_half(h4.x));
            wv[st][1] = __half2float(__ushort_as_half(h4.y));
            wv[st][2] = __half2float(__ushort_as_half(h4.z));
            wv[st][3] = __half2float(__ushort_as_half(h4.w));
          }
        }
#pragma unroll
        for (unsigned j = 0; j < tokens; ++j) {
          const float4 x4 = reinterpret_cast<const float4*>(
              x + static_cast<std::size_t>(j) * k)[i0 / 4];
          xv[st][j][0] = x4.x;
          xv[st][j][1] = x4.y;
          xv[st][j][2] = x4.z;
          xv[st][j][3] = x4.w;
        }
        continue;
      }
#pragma unroll
      for (unsigned r = 0; r < 4; ++r) {
        wv[st][r] = i0 + r < k ? RowElement(wrow, type, i0 + r) : 0.0f;
      }
#pragma unroll
      for (unsigned j = 0; j < tokens; ++j) {
        const float* xr = x + static_cast<std::size_t>(j) * k + i0;
#pragma unroll
        for (unsigned r = 0; r < 4; ++r) {
          xv[st][j][r] = i0 + r < k ? xr[r] : 0.0f;
        }
      }
    }
#pragma unroll
    for (unsigned st = 0; st < kSteps; ++st) {
      const std::uint32_t i0 = i_base + st * stride;
      if (!kWhole && i0 >= k) {
        break;
      }
#pragma unroll
      for (unsigned j = 0; j < tokens; ++j) {
        if constexpr (stable) {
          // EXL3's unquantized operands keep FP32 inputs and an explicit
          // FMA chain in every phase, including ragged HC mixer widths.
          float a = acc[j];
#pragma unroll
          for (unsigned r = 0; r < 4; ++r) {
            const float w32 = Materialize(wv[st][r]);
            const float x32 = Materialize(xv[st][j][r]);
            a = Materialize(__builtin_fmaf(w32, x32, a));
          }
          acc[j] = a;
          continue;
        }
        if constexpr (kWhole) {
          // The bounds-checked path compiles acc + dot to one FMA chain
          // from acc in element order; spelled out so it cannot reassociate.
          float a = acc[j];
#pragma unroll
          for (unsigned r = 0; r < 4; ++r) {
            a = __builtin_fmaf(wv[st][r], xv[st][j][r], a);
          }
          acc[j] = Materialize(a);
          continue;
        }
        float dot = 0.0f;
#pragma unroll
        for (unsigned r = 0; r < 4; ++r) {
          dot += wv[st][r] * xv[st][j][r];
        }
        acc[j] = Materialize(acc[j] + dot);
      }
    }
  }
#pragma unroll
  for (unsigned j = 0; j < tokens; ++j) {
    const float total = WaveSum(acc[j]);
    if (lane == 0) {
      // Preserve the F32 store boundary of SmallGemmStable + SiluScale. The
      // reduction and every input product above are shared with that path.
      out[static_cast<std::size_t>(j) * m + row] =
          kHcDown ? SiluF(Materialize(total) * 0.25F) : total;
    }
  }
}

template<WeightType type, unsigned tokens, bool stable = false,
         bool kHcDown = false>
void LaunchSmallGemm(const void* w, const float* x, float* out, std::uint32_t m,
                     std::uint32_t k, hipStream_t stream) {
  if constexpr (type != WeightType::kQ8_0) {
    // Rows of whole four-step passes (K 2560: router and GDN alpha/beta in
    // F16, indexer in BF16) take the unchecked vector loads, four steps at a
    // time at any width: the same products in the same order.
    const bool aligned = (reinterpret_cast<std::uintptr_t>(w) |
                          reinterpret_cast<std::uintptr_t>(x)) %
                             16 ==
                         0;
    if (aligned && k % (32 * 4 * 4) == 0) {
      hipLaunchKernelGGL(
          (SmallGemmKernel<type, tokens, false, true, stable, kHcDown>),
          dim3((m + kSmallGemmRows - 1) / kSmallGemmRows),
          dim3(kSmallGemmRows * 32), 0, stream, w, x, out, m, k);
      return;
    }
  }
  hipLaunchKernelGGL(
      (SmallGemmKernel<type, tokens, false, false, stable, kHcDown>),
      dim3((m + kSmallGemmRows - 1) / kSmallGemmRows),
      dim3(kSmallGemmRows * 32), 0, stream, w, x, out, m, k);
}

template<WeightType type, bool stable = false, bool kHcDown = false>
void SmallGemmForType(const void* w, const float* x, float* out,
                      std::uint32_t tokens, std::uint32_t m, std::uint32_t k,
                      hipStream_t stream) {
  if (tokens > 8) {
    const auto groups = tokens / 8;
    hipLaunchKernelGGL((SmallGemmKernel<type, 8, true, false, stable, kHcDown>),
                       dim3((m + kSmallGemmRows - 1) / kSmallGemmRows, groups),
                       dim3(kSmallGemmRows * 32), 0, stream, w, x, out, m, k);
    const auto consumed = groups * 8;
    x += std::size_t{consumed} * k;
    out += std::size_t{consumed} * m;
    tokens -= consumed;
    if (tokens == 0)
      return;
  }
  switch (tokens) {
    case 1:
      return LaunchSmallGemm<type, 1, stable, kHcDown>(w, x, out, m, k, stream);
    case 2:
      return LaunchSmallGemm<type, 2, stable, kHcDown>(w, x, out, m, k, stream);
    case 3:
      return LaunchSmallGemm<type, 3, stable, kHcDown>(w, x, out, m, k, stream);
    case 4:
      return LaunchSmallGemm<type, 4, stable, kHcDown>(w, x, out, m, k, stream);
    case 5:
      return LaunchSmallGemm<type, 5, stable, kHcDown>(w, x, out, m, k, stream);
    case 6:
      return LaunchSmallGemm<type, 6, stable, kHcDown>(w, x, out, m, k, stream);
    case 7:
      return LaunchSmallGemm<type, 7, stable, kHcDown>(w, x, out, m, k, stream);
    case 8:
      return LaunchSmallGemm<type, 8, stable, kHcDown>(w, x, out, m, k, stream);
    default:
      throw std::logic_error("small projection requires 1-8 token rows");
  }
}

}  // namespace

void SmallGemmStable(const void* w, WeightType type, const float* x, float* out,
                     std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
                     hipStream_t stream) {
  switch (type) {
    case WeightType::kF32:
      return SmallGemmForType<WeightType::kF32, true>(w, x, out, n_tokens, m, k,
                                                      stream);
    case WeightType::kBF16:
      return SmallGemmForType<WeightType::kBF16, true>(w, x, out, n_tokens, m,
                                                       k, stream);
    case WeightType::kF16:
      return SmallGemmForType<WeightType::kF16, true>(w, x, out, n_tokens, m, k,
                                                      stream);
    default:
      throw std::logic_error("unsupported small projection format");
  }
}

}  // namespace gufo::test::small_gemm_reference
