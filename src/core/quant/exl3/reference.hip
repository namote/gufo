// X2a reference GPU path for EXL3 projections; see reference.hpp. The trellis
// decode follows the X0 CPU reference (format.cpp), which mirrors upstream
// exllamav3 v1.6.5 (cdc785db, MIT, Copyright (c) 2025 Turboderp):
// exllamav3_ext/quant/exl3_dq.cuh and codebook.cuh. The codebook arithmetic
// is done in FP32 where it is exact and rounded once to FP16, which equals
// the upstream FP16 instructions bit for bit.

#include <hip/hip_fp16.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "src/core/quant/exl3/reference.hpp"

namespace gufo::quant::exl3 {
namespace {

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("EXL3 reference: ") + what + ": " +
                             hipGetErrorString(status));
  }
}

__device__ float HalfBits(std::uint32_t bits) {
  return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}

__device__ float DecodeLop3(std::uint32_t x) {
  x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
  // Sum of two halves is exact in FP32; one rounding to FP16.
  return __half2float(
      __float2half_rn(HalfBits(x & 0xffffu) + HalfBits(x >> 16)));
}

__device__ float DecodeState(std::uint32_t state, int codebook) {
  if (codebook == 0)
    return DecodeLop3(state * 89226354u + 64248484u);
  if (codebook == 1)
    return DecodeLop3(state * 0xCBAC1FEDu);
  const std::uint32_t x = state * 0x83DCD12Du;
  const std::uint32_t sum =
      (x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) + (x >> 24);
  // (1024 + sum) * inv + bias is exact in FP32 here; one rounding to FP16.
  const float value =
      (1024.0f + static_cast<float>(sum)) * HalfBits(0x1eee) + HalfBits(0xc931);
  return __half2float(__float2half_rn(value));
}

__device__ int StateEndBit(int bits, bool half, int p) {
  if (!half)
    return (p + 1) * bits;
  const int pair = 2 * bits + 1;
  return (p / 2) * pair + ((p % 2) == 0 ? bits : pair);
}

// One block per tile, one thread per position.
__global__ void DequantKernel(const std::uint16_t* trellis, int tiles_n,
                              int bits, bool half, int codebook, int tile_words,
                              int tn_begin, int tn_count, int k, float* out) {
  const int tile = blockIdx.x;
  const int tk = tile / tn_count;
  const int tn = tn_begin + tile % tn_count;
  const int p = threadIdx.x;
  const std::uint16_t* t =
      trellis + (static_cast<std::size_t>(tk) * tiles_n + tn) * tile_words;
  const int words32 = tile_words / 2;
  const int total = tile_words * 16;
  const int end = StateEndBit(bits, half, p) + total;
  const int i0 = (end - 16) / 32, i1 = (end - 1) / 32;
  const int shift = (i1 + 1) * 32 - end;
  auto word = [&](int i) {
    const int w = i % words32;
    return static_cast<std::uint32_t>(t[2 * w]) |
           (static_cast<std::uint32_t>(t[2 * w + 1]) << 16);
  };
  const std::uint64_t merged =
      (static_cast<std::uint64_t>(word(i0)) << 32) | word(i1);
  const auto state = static_cast<std::uint32_t>((merged >> shift) & 0xffffu);
  // Tensor-core permutation: position p -> tile element (r, c).
  const int lane = p / 8, j = p % 8;
  const int r0 = (lane % 4) * 2;
  const int rows[4] = {r0, r0 + 1, r0 + 8, r0 + 9};
  const int r = rows[j % 4];
  const int c = (j < 4 ? lane / 4 : lane / 4 + 8);
  const int col = (tn - tn_begin) * 16 + c;  // local output column
  const int row = tk * 16 + r;               // input row
  out[static_cast<std::size_t>(col) * k + row] = DecodeState(state, codebook);
}

// One block of 128 threads per (row, 128-column block).
__global__ void RotateKernel(const float* x, std::size_t x_stride,
                             std::size_t width, const std::uint16_t* pre,
                             const std::uint16_t* post, float* out,
                             std::size_t out_stride) {
  __shared__ float v[128];
  const std::size_t row = blockIdx.y;
  const std::size_t col = blockIdx.x * 128 + threadIdx.x;
  float value = x[row * x_stride + col];
  if (pre != nullptr)
    value *= HalfBits(pre[col]);
  v[threadIdx.x] = value;
  __syncthreads();
  for (int h = 1; h < 128; h *= 2) {
    const int i = threadIdx.x;
    float result = 0.0f;
    if ((i & h) == 0) {
      result = v[i] + v[i + h];
    } else {
      result = v[i - h] - v[i];
    }
    __syncthreads();
    v[i] = result;
    __syncthreads();
  }
  value = v[threadIdx.x] * 0.088388347648318441f;  // 1 / sqrt(128)
  if (post != nullptr)
    value *= HalfBits(post[col]);
  out[row * out_stride + col] = value;
}

// y[r][n_begin + j] = sum_i x[r][i] w[j][i]; 16 x 16 tiles through shared
// memory.
__global__ void GemmKernel(const float* x, std::size_t x_stride, int rows,
                           const float* w, int k, int n_count, float* y,
                           std::size_t y_stride, int n_begin) {
  __shared__ float xs[16][17];
  __shared__ float ws[16][17];
  const int r = blockIdx.y * 16 + threadIdx.y;
  const int j = blockIdx.x * 16 + threadIdx.x;
  float acc = 0.0f;
  for (int i0 = 0; i0 < k; i0 += 16) {
    const int xr = blockIdx.y * 16 + threadIdx.y;
    xs[threadIdx.y][threadIdx.x] =
        xr < rows
            ? x[static_cast<std::size_t>(xr) * x_stride + i0 + threadIdx.x]
            : 0.0f;
    const int wj = blockIdx.x * 16 + threadIdx.y;
    ws[threadIdx.y][threadIdx.x] =
        wj < n_count ? w[static_cast<std::size_t>(wj) * k + i0 + threadIdx.x]
                     : 0.0f;
    __syncthreads();
    for (int t = 0; t < 16; ++t)
      acc += xs[threadIdx.y][t] * ws[threadIdx.x][t];
    __syncthreads();
  }
  if (r < rows && j < n_count)
    y[static_cast<std::size_t>(r) * y_stride + n_begin + j] = acc;
}

}  // namespace

void LaunchReferenceDequantF32(const DeviceLinear& linear, std::size_t n_begin,
                               std::size_t n_count, float* out,
                               hipStream_t stream) {
  if (n_begin % 16 != 0 || n_count % 16 != 0 || n_begin + n_count > linear.n) {
    throw std::invalid_argument("EXL3 reference dequant: bad column range");
  }
  const int tiles_k = static_cast<int>(linear.k / 16);
  const int tn_count = static_cast<int>(n_count / 16);
  if (tiles_k == 0 || tn_count == 0)
    return;
  const int codebook = linear.codebook == Codebook::k3Inst
                           ? 0
                           : (linear.codebook == Codebook::kMcg ? 1 : 2);
  hipLaunchKernelGGL(DequantKernel, dim3(tiles_k * tn_count), dim3(256), 0,
                     stream, linear.trellis, static_cast<int>(linear.n / 16),
                     linear.rate.bits, linear.rate.half, codebook,
                     linear.rate.TileWords(), static_cast<int>(n_begin / 16),
                     tn_count, static_cast<int>(linear.k), out);
  Check(hipGetLastError(), "dequant launch");
}

void LaunchReferenceRotateRows(const float* x, std::size_t x_stride,
                               std::size_t rows, std::size_t width,
                               const std::uint16_t* pre,
                               const std::uint16_t* post, float* out,
                               std::size_t out_stride, hipStream_t stream) {
  if (width % 128 != 0)
    throw std::invalid_argument("EXL3 rotation width % 128");
  if (rows == 0 || width == 0)
    return;
  hipLaunchKernelGGL(
      RotateKernel,
      dim3(static_cast<unsigned>(width / 128), static_cast<unsigned>(rows)),
      dim3(128), 0, stream, x, x_stride, width, pre, post, out, out_stride);
  Check(hipGetLastError(), "rotate launch");
}

void LaunchReferenceProjection(const DeviceLinear& linear, const float* x,
                               std::size_t x_stride, std::size_t rows, float* y,
                               std::size_t y_stride,
                               const ReferenceScratch& scratch,
                               hipStream_t stream) {
  const std::size_t k = linear.k, n = linear.n;
  if (rows == 0)
    return;
  if (scratch.rotated_elements < rows * k) {
    throw std::runtime_error("EXL3 reference: rotated scratch too small");
  }
  const std::size_t chunk =
      std::min<std::size_t>(n, scratch.weight_elements / k / 128 * 128);
  if (chunk == 0)
    throw std::runtime_error("EXL3 reference: weight scratch too small");
  LaunchReferenceRotateRows(x, x_stride, rows, k, linear.suh, nullptr,
                            scratch.rotated, k, stream);
  for (std::size_t n0 = 0; n0 < n; n0 += chunk) {
    const std::size_t count = std::min(chunk, n - n0);
    LaunchReferenceDequantF32(linear, n0, count, scratch.weight, stream);
    const dim3 grid(static_cast<unsigned>((count + 15) / 16),
                    static_cast<unsigned>((rows + 15) / 16));
    hipLaunchKernelGGL(
        GemmKernel, grid, dim3(16, 16), 0, stream, scratch.rotated, k,
        static_cast<int>(rows), scratch.weight, static_cast<int>(k),
        static_cast<int>(count), y, y_stride, static_cast<int>(n0));
    Check(hipGetLastError(), "gemm launch");
  }
  LaunchReferenceRotateRows(y, y_stride, rows, n, nullptr, linear.svh, y,
                            y_stride, stream);
}

}  // namespace gufo::quant::exl3
