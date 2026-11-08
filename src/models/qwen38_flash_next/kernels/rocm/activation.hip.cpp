// Elementwise activations: SiLU scale, SwiGLU, sigmoid gates and the
// F32 to F16/BF16 narrowing for hipBLAS inputs.
// HIP kernels follow the layouts and operator formulas in reference.cpp.

#include "src/models/qwen38_flash_next/kernels/rocm/kernels_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

namespace {

__global__ void SiluScaleKernel(float* x, float scale, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    x[i] = SiluF(x[i] * scale);
  }
}

/// SiluScaleKernel over rows of `k` that also quantizes its output for the
/// next projection into Q8_1 rows of `k_padded` (k % 32 == 0, so 32-lane
/// groups are whole and stay within a row).
__global__ void SiluScaleQ8Kernel(float* x, float scale, std::size_t count,
                                  std::uint32_t k, std::uint32_t k_padded,
                                  void* x_q8) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }
  const float v = SiluF(x[i] * scale);
  x[i] = v;
  QuantizeQ8_1Lane(v, x_q8, (i / k) * k_padded + i % k);
}

/// out = silu(gate) * up as F16, four elements per thread.
__global__ void SwigluHalfKernel(const float* __restrict__ gate,
                                 const float* __restrict__ up,
                                 __half* __restrict__ out, std::size_t count) {
  const std::size_t i =
      (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) * 4;
  if (i + 4 > count) {
    for (std::size_t j = i; j < count; ++j) {
      out[j] = __float2half(SiluF(gate[j]) * up[j]);
    }
    return;
  }
  const float4 g = *reinterpret_cast<const float4*>(gate + i);
  const float4 u = *reinterpret_cast<const float4*>(up + i);
  const __half2 lo = __floats2half2_rn(SiluF(g.x) * u.x, SiluF(g.y) * u.y);
  const __half2 hi = __floats2half2_rn(SiluF(g.z) * u.z, SiluF(g.w) * u.w);
  *reinterpret_cast<__half2*>(out + i) = lo;
  *reinterpret_cast<__half2*>(out + i + 2) = hi;
}

/// silu(gate) * up over rows of `k` elements, written only as the W8A8
/// tiled Q8 layout: four elements per lane, eight lanes per K block.
__global__ void SwigluQ8Kernel(const float* gate, const float* up, void* out_q8,
                               std::size_t n_rows, std::size_t k) {
  const std::size_t chunk =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t chunks_per_row = k / 4;
  const std::size_t row = chunk / chunks_per_row;
  const std::size_t i = (chunk % chunks_per_row) * 4;
  // Rows past the batch still join the wave reduction below.
  const bool live = row < n_rows;
  float4 v{0.0F, 0.0F, 0.0F, 0.0F};
  if (live) {
    const std::size_t idx = (row * k) + i;
    const float4 g = *reinterpret_cast<const float4*>(gate + idx);
    const float4 u = *reinterpret_cast<const float4*>(up + idx);
    v = float4{SiluF(g.x) * u.x, SiluF(g.y) * u.y, SiluF(g.z) * u.z,
               SiluF(g.w) * u.w};
  }
  float max_abs =
      fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
  for (int off = 4; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, XorLane(max_abs, off));
  }
  if (!live) {
    return;
  }
  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const auto q0 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.x * id))));
  const auto q1 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.y * id))));
  const auto q2 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.z * id))));
  const auto q3 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.w * id))));
  const std::size_t tl = row % kQ8ActTileTokens;
  const std::uint32_t pos = static_cast<std::uint32_t>(i % 32);
  std::int8_t* tile = Q8ActTile(out_q8, k / 32, row / kQ8ActTileTokens, i / 32);
  *reinterpret_cast<std::uint32_t*>(tile + ((pos >> 4u) * 256) + (tl * 16) +
                                    (pos & 15u)) =
      q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
  if (pos == 0) {
    *reinterpret_cast<float*>(tile + kQ8ActScaleOffset + (tl * sizeof(float))) =
        d;
  }
}

__global__ void SwigluKernel(float* gate, const float* up, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    gate[i] = SiluF(gate[i]) * up[i];
  }
}

__global__ void SigmoidMulKernel(float* x, const float* g, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    x[i] *= SigmoidF(g[i]);
  }
}

/// SigmoidMulKernel over rows of `width` (width % 128 == 0), one wave per
/// (row, segment), that also rotates the stored values into `slabs`.
__launch_bounds__(256) __global__
    void SigmoidMulRotateKernel(float* __restrict__ x,
                                const float* __restrict__ g, std::uint32_t rows,
                                std::uint32_t width, Exl3InputSlabs slabs) {
  const std::uint32_t segments = width / hip::exl3::kSegment;
  const std::uint32_t item = blockIdx.x * 8 + threadIdx.x / 32;
  const std::uint32_t lane = threadIdx.x % 32;
  if (item >= rows * segments)
    return;
  const std::uint32_t row = item / segments;
  const std::uint32_t first =
      (item % segments) * hip::exl3::kSegment + 4 * lane;
  float v[4];
#pragma unroll
  for (std::uint32_t j = 0; j < 4; ++j) {
    const std::size_t i = std::size_t{row} * width + first + j;
    v[j] = x[i] * SigmoidF(g[i]);
    x[i] = v[j];
  }
  RotateIntoSlabs(v, lane, first, row, slabs);
}

/// SigmoidMulKernel that also writes x's unpadded Q8_1 rows (count % 32 == 0).
__global__ void SigmoidMulQ8Kernel(float* x, const float* g, std::size_t count,
                                   void* x_q8) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }
  const float v = x[i] * SigmoidF(g[i]);
  x[i] = v;
  QuantizeQ8_1Lane(v, x_q8, i);
}

/// NarrowKernel<__half>, four elements per thread with 16-byte loads (the
/// same conversion per element).
__global__ void NarrowHalf4Kernel(const float4* x, ushort4* out,
                                  std::size_t count4) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count4) {
    const float4 v = x[i];
    out[i] = make_ushort4(
        __half_as_ushort(__half(v.x)), __half_as_ushort(__half(v.y)),
        __half_as_ushort(__half(v.z)), __half_as_ushort(__half(v.w)));
  }
}

template<typename T>
__global__ void NarrowKernel(const float* x, T* out, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    out[i] = T(x[i]);
  }
}

}  // namespace

void SiluScale(float* x, float scale, std::size_t count, hipStream_t stream) {
  hipLaunchKernelGGL(SiluScaleKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, x, scale, count);
}

bool SiluScaleQ8_1(float* x, float scale, std::uint32_t rows, std::uint32_t k,
                   std::uint32_t k_padded, void* x_q8, hipStream_t stream) {
  if (k == 0 || k % 32 != 0 || k_padded < k || k_padded % 32 != 0) {
    return false;
  }
  const std::size_t count = static_cast<std::size_t>(rows) * k;
  hipLaunchKernelGGL(SiluScaleQ8Kernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, x, scale, count, k, k_padded, x_q8);
  return true;
}

void Swiglu(float* gate, const float* up, std::size_t count,
            hipStream_t stream) {
  hipLaunchKernelGGL(SwigluKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, gate, up, count);
}

void SwigluHalf(const float* gate, const float* up, __half* out,
                std::size_t count, hipStream_t stream) {
  hipLaunchKernelGGL(SwigluHalfKernel, dim3(Blocks((count + 3) / 4)),
                     dim3(kThreads), 0, stream, gate, up, out, count);
}

bool SwigluQ8Tiled(const float* gate, const float* up, void* out_q8,
                   std::size_t n_rows, std::size_t k, hipStream_t stream) {
  if (k % 32 != 0) {
    return false;
  }
  const std::size_t chunks = n_rows * (k / 4);
  hipLaunchKernelGGL(SwigluQ8Kernel, dim3(Blocks(chunks)), dim3(kThreads), 0,
                     stream, gate, up, out_q8, n_rows, k);
  return true;
}

void SigmoidMul(float* x, const float* g, std::size_t count,
                hipStream_t stream) {
  hipLaunchKernelGGL(SigmoidMulKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, x, g, count);
}

bool SigmoidMulRotate(float* x, const float* g, std::uint32_t rows,
                      std::uint32_t width, const Exl3InputSlabs& slabs,
                      hipStream_t stream) {
  if (rows == 0 || slabs.count != 1 || slabs.k != width ||
      width % hip::exl3::kSegment != 0 || rows > slabs.pass_rows)
    return false;
  const std::uint32_t items = rows * (width / hip::exl3::kSegment);
  hipLaunchKernelGGL(SigmoidMulRotateKernel, dim3((items + 7) / 8), dim3(256),
                     0, stream, x, g, rows, width, slabs);
  return true;
}

bool SigmoidMulQ8_1(float* x, const float* g, std::size_t count, void* x_q8,
                    hipStream_t stream) {
  if (count == 0 || count % 32 != 0) {
    return false;
  }
  hipLaunchKernelGGL(SigmoidMulQ8Kernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, x, g, count, x_q8);
  return true;
}

void NarrowActivations(const float* x, void* out, bool bf16, std::size_t count,
                       hipStream_t stream) {
  if (bf16) {
    hipLaunchKernelGGL(NarrowKernel<hip_bfloat16>, dim3(Blocks(count)),
                       dim3(kThreads), 0, stream, x,
                       static_cast<hip_bfloat16*>(out), count);
  } else if (count % 4 == 0 && (reinterpret_cast<std::uintptr_t>(x) |
                                reinterpret_cast<std::uintptr_t>(out)) %
                                       16 ==
                                   0) {
    hipLaunchKernelGGL(NarrowHalf4Kernel, dim3(Blocks(count / 4)),
                       dim3(kThreads), 0, stream,
                       reinterpret_cast<const float4*>(x),
                       static_cast<ushort4*>(out), count / 4);
  } else {
    hipLaunchKernelGGL(NarrowKernel<__half>, dim3(Blocks(count)),
                       dim3(kThreads), 0, stream, x, static_cast<__half*>(out),
                       count);
  }
}

}  // namespace gufo::models::qwen38_flash_next::rocm
