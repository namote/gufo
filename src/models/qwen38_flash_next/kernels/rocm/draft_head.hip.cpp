// Load-time 4-bit copy of the MTP draft head prefix
// (DeviceModel::SetDraftVocab, upstream #1209 on an EXL3 head). Built without
// fast math or contraction (CMakeLists.txt): the quantizer rounds as ggml's
// single-precision reference, like its CPU test (draft_head_q4k.hpp).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "src/core/quant/exl3/reference.hpp"
#include "src/models/qwen38_flash_next/draft_head_q4k.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

#pragma clang fp contract(off)

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string("Q4_K draft head: ") + what + ": " +
                             hipGetErrorString(status));
}

// out[c][r] = in[r][c] for a rows x cols matrix.
__global__ void TransposeKernel(const float* in, float* out, unsigned rows,
                                unsigned cols) {
  __shared__ float tile[32][33];
  const unsigned c = blockIdx.x * 32 + threadIdx.x;
  const unsigned r0 = blockIdx.y * 32;
  for (unsigned i = threadIdx.y; i < 32; i += blockDim.y)
    if (r0 + i < rows && c < cols)
      tile[i][threadIdx.x] = in[std::size_t{r0 + i} * cols + c];
  __syncthreads();
  const unsigned r = r0 + threadIdx.x;
  const unsigned c0 = blockIdx.x * 32;
  for (unsigned i = threadIdx.y; i < 32; i += blockDim.y)
    if (c0 + i < cols && r < rows)
      out[std::size_t{c0 + i} * rows + r] = tile[threadIdx.x][i];
}

// One thread per superblock of `count` rows. Row n's value k is
// columns[k * count + n] (the transposed dense chunk).
__global__ void QuantizeColumnsKernel(const float* columns, unsigned count,
                                      unsigned superblocks_per_row,
                                      q4k::Block* out) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= std::size_t{count} * superblocks_per_row)
    return;
  const unsigned n = static_cast<unsigned>(index / superblocks_per_row);
  const unsigned s = static_cast<unsigned>(index % superblocks_per_row);
  q4k::QuantizeSuperblock(columns + std::size_t{s} * 256 * count + n, count,
                          out + index);
}

}  // namespace

std::size_t DraftHeadQ4KChunkFloats(std::uint32_t cols) {
  return std::size_t{kDraftHeadQ4KChunkRows} * cols;
}

void BuildDraftHeadQ4K(const quant::exl3::DeviceLinear& head,
                       std::uint32_t rows, void* q4k, float* scratch_a,
                       float* scratch_b, hipStream_t stream) {
  const std::uint32_t k = head.k;
  if (k % 256 != 0 || rows % 128 != 0 || rows > head.n)
    throw std::invalid_argument("Q4_K draft head: unsupported shape");
  auto* out = static_cast<q4k::Block*>(q4k);
  for (std::uint32_t n0 = 0; n0 < rows; n0 += kDraftHeadQ4KChunkRows) {
    const std::uint32_t count = std::min(kDraftHeadQ4KChunkRows, rows - n0);
    // A dense row of the projection y = x W (reference.hpp):
    //   W[n][i] = svh[n] * sum_m H[n][m] * suh[i] * (H W'[m])[i]
    // with H the orthonormal, symmetric 128-block Hadamard rotation.
    quant::exl3::LaunchReferenceDequantF32(head, n0, count, scratch_a, stream);
    quant::exl3::LaunchReferenceRotateRows(scratch_a, k, count, k, nullptr,
                                           head.suh, scratch_a, k, stream);
    hipLaunchKernelGGL(TransposeKernel, dim3((k + 31) / 32, (count + 31) / 32),
                       dim3(32, 8), 0, stream, scratch_a, scratch_b, count, k);
    Check(hipGetLastError(), "transpose launch");
    // scratch_b is [k][count]: rotate each input column along the rows.
    quant::exl3::LaunchReferenceRotateRows(scratch_b, count, k, count, nullptr,
                                           head.svh + n0, scratch_b, count,
                                           stream);
    const unsigned per_row = k / 256;
    const std::size_t blocks = std::size_t{count} * per_row;
    hipLaunchKernelGGL(QuantizeColumnsKernel,
                       dim3(static_cast<unsigned>((blocks + 63) / 64)),
                       dim3(64), 0, stream, scratch_b, count, per_row,
                       out + std::size_t{n0} * per_row);
    Check(hipGetLastError(), "quantize launch");
  }
}

}  // namespace gufo::models::qwen38_flash_next::rocm
