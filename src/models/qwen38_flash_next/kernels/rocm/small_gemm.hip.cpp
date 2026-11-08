// Small-batch GEMMs over GGUF weights (SmallGemm, SmallGemmStable) and
// the HC down projection's stable deep-step F16 route.
// HIP kernels follow the layouts and operator formulas in reference.cpp.

#include "src/models/qwen38_flash_next/kernels/rocm/kernels_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

namespace {

/// One wave per weight row, kSmallGemmRows rows per block: the row is read
/// once (four consecutive elements per lane per step) while up to eight
/// tokens accumulate in registers, then a wave reduction per token.
constexpr unsigned kSmallGemmRows = 4;
template<WeightType type, unsigned tokens, bool grouped = false,
         bool kWhole = false, bool stable = false, bool kHcDown = false,
         bool kTailVector = false, unsigned kDeepSteps = 0>
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
  // steps still accumulate one after another, in order. kDeepSteps (whole
  // passes only) keeps more of a long row's steps in flight.
  static_assert(kDeepSteps == 0 || kWhole);
  constexpr unsigned kSteps = kDeepSteps != 0                   ? kDeepSteps
                              : stable || kWhole || tokens <= 2 ? 4
                                                                : 2;
  const std::uint32_t stride = warpSize * 4;
  float acc[tokens] = {};
  for (std::uint32_t i_base = lane * 4; i_base < k; i_base += stride * kSteps) {
    float wv[kSteps][4];
    float xv[kSteps][tokens][4];
#pragma unroll
    for (unsigned st = 0; st < kSteps; ++st) {
      const std::uint32_t i0 = i_base + st * stride;
      if constexpr (kWhole || kTailVector) {
        if constexpr (kTailVector) {
          // HC-up has K=320: a lane's four values are wholly valid or
          // absent. Do not read the missing steps of the four-step pass.
          if (i0 >= k) {
#pragma unroll
            for (unsigned r = 0; r < 4; ++r) {
              wv[st][r] = 0.0F;
#pragma unroll
              for (unsigned j = 0; j < tokens; ++j)
                xv[st][j][r] = 0.0F;
            }
            continue;
          }
        }
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

}  // namespace

template<WeightType type, unsigned tokens, bool stable = false,
         bool kHcDown = false, bool grouped = false>
void LaunchSmallGemm(const void* w, const float* x, float* out, std::uint32_t m,
                     std::uint32_t k, hipStream_t stream, unsigned groups = 1) {
  const dim3 grid((m + kSmallGemmRows - 1) / kSmallGemmRows, groups);
  if constexpr (type != WeightType::kQ8_0) {
    // Rows of whole four-step passes (K 2560: router and GDN alpha/beta in
    // F16, indexer in BF16) take the unchecked vector loads, four steps at a
    // time at any width: the same products in the same order.
    const bool aligned = (reinterpret_cast<std::uintptr_t>(w) |
                          reinterpret_cast<std::uintptr_t>(x)) %
                             16 ==
                         0;
    if constexpr (stable && type == WeightType::kF16 && !kHcDown) {
      if (aligned && k == 320) {
        hipLaunchKernelGGL((SmallGemmKernel<type, tokens, grouped, false,
                                            stable, false, true>),
                           grid, dim3(kSmallGemmRows * 32), 0, stream, w, x,
                           out, m, k);
        return;
      }
    }
    if (aligned && k % (32 * 4 * 4) == 0) {
      hipLaunchKernelGGL(
          (SmallGemmKernel<type, tokens, grouped, true, stable, kHcDown>), grid,
          dim3(kSmallGemmRows * 32), 0, stream, w, x, out, m, k);
      return;
    }
  }
  hipLaunchKernelGGL(
      (SmallGemmKernel<type, tokens, grouped, false, stable, kHcDown>), grid,
      dim3(kSmallGemmRows * 32), 0, stream, w, x, out, m, k);
}

template<WeightType type, bool stable = false, bool kHcDown = false>
void SmallGemmForType(const void* w, const float* x, float* out,
                      std::uint32_t tokens, std::uint32_t m, std::uint32_t k,
                      hipStream_t stream) {
  if (tokens > 8) {
    const auto groups = tokens / 8;
    if constexpr (stable) {
      // Each group owns eight complete rows, preserving vector alignment.
      // Stable scalar/vector loads share the same four-step FMA chain.
      LaunchSmallGemm<type, 8, stable, kHcDown, true>(w, x, out, m, k, stream,
                                                      groups);
    } else {
      hipLaunchKernelGGL(
          (SmallGemmKernel<type, 8, true, false, stable, kHcDown>),
          dim3((m + kSmallGemmRows - 1) / kSmallGemmRows, groups),
          dim3(kSmallGemmRows * 32), 0, stream, w, x, out, m, k);
    }
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

/// HC down's stable whole-pass kernel with `steps` four-element steps of a
/// lane in flight per pass instead of four: 80 blocks stream 10240-wide rows
/// and were latency-bound. Every lane still visits its steps in increasing
/// order with the same pinned FMA chain, so each output keeps its bits.
template<unsigned tokens, unsigned steps>
void LaunchHcDownDeep(const void* w, const float* x, float* out,
                      hipStream_t stream) {
  constexpr std::uint32_t m = 320;
  constexpr std::uint32_t k = 10240;
  static_assert(k % (32 * 4 * steps) == 0);
  hipLaunchKernelGGL((SmallGemmKernel<WeightType::kF16, tokens, false, true,
                                      true, false, false, steps>),
                     dim3((m + kSmallGemmRows - 1) / kSmallGemmRows),
                     dim3(kSmallGemmRows * 32), 0, stream, w, x, out, m, k);
}

void HcDownSiluF16(const void* w, const float* x, float* out,
                   std::uint32_t n_tokens, hipStream_t stream) {
  // The fused SiLU compiled differently for some rows despite nominally
  // identical source expressions. Keep the established F32 store boundary.
  // Steps in flight are bounded by the live x registers (4 * tokens each);
  // seven or eight rows already fill the registers at four steps (five
  // spilled), so they keep the generic launch.
  const bool aligned = (reinterpret_cast<std::uintptr_t>(w) |
                        reinterpret_cast<std::uintptr_t>(x)) %
                           16 ==
                       0;
  if (!aligned || n_tokens == 0 || n_tokens > 6) {
    SmallGemmStable(w, WeightType::kF16, x, out, n_tokens, 320, 10240, stream);
  } else {
    switch (n_tokens) {
      case 1:
        LaunchHcDownDeep<1, 10>(w, x, out, stream);
        break;
      case 2:
        LaunchHcDownDeep<2, 10>(w, x, out, stream);
        break;
      case 3:
        LaunchHcDownDeep<3, 8>(w, x, out, stream);
        break;
      case 4:
        LaunchHcDownDeep<4, 8>(w, x, out, stream);
        break;
      case 5:
        LaunchHcDownDeep<5, 5>(w, x, out, stream);
        break;
      default:
        LaunchHcDownDeep<6, 5>(w, x, out, stream);
        break;
    }
  }
  SiluScale(out, 0.25F, std::size_t{n_tokens} * 320, stream);
}

void SmallGemm(const void* w, WeightType type, const float* x, float* out,
               std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
               hipStream_t stream) {
  switch (type) {
    case WeightType::kF32:
      return SmallGemmForType<WeightType::kF32>(w, x, out, n_tokens, m, k,
                                                stream);
    case WeightType::kBF16:
      return SmallGemmForType<WeightType::kBF16>(w, x, out, n_tokens, m, k,
                                                 stream);
    case WeightType::kF16:
      return SmallGemmForType<WeightType::kF16>(w, x, out, n_tokens, m, k,
                                                stream);
    default:
      throw std::logic_error("unsupported small projection format");
  }
}

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

}  // namespace gufo::models::qwen38_flash_next::rocm
