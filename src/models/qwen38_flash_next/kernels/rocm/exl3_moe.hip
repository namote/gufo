#include <algorithm>
#include <stdexcept>
#include <type_traits>

#include "src/core/hip/detail/exl3_gemv.hpp"
#include "src/core/quant/exl3/gemv.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/exl3_moe.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {
namespace ex = quant::exl3;
namespace arithmetic = gufo::hip::exl3;
using arithmetic::Half2;

void Check() {
  const auto status = hipGetLastError();
  if (status != hipSuccess)
    throw std::runtime_error(std::string("Flash-Next EXL3: ") +
                             hipGetErrorString(status));
}

__global__ void CopyColumns(const float* x, float* y, unsigned rows,
                            unsigned stride, unsigned begin, unsigned width) {
  const auto i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i < std::size_t{rows} * width)
    y[i] = x[(i / width) * stride + begin + i % width];
}

template<bool Scatter>
__global__ void MoveRows(const float* x, const std::int32_t* map, float* y,
                         unsigned rows, unsigned width) {
  const auto i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i < std::size_t{rows} * width) {
    const auto r = i / width;
    const auto mapped =
        std::size_t{static_cast<unsigned>(map[r])} * width + i % width;
    if constexpr (Scatter)
      y[mapped] = x[i];
    else
      y[i] = x[mapped];
  }
}

__global__ void BuildTiles(const std::int32_t* ids, unsigned assignments,
                           Exl3RouteTile* tiles, unsigned* count,
                           unsigned experts, unsigned tile_rows) {
  const unsigned expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= experts)
    return;
  Exl3RouteTile tile{};
  tile.expert = expert;
  for (unsigned i = 0; i < assignments; ++i) {
    if (ids[i] != static_cast<int>(expert))
      continue;
    tile.slots[tile.count++] = i;
    if (tile.count == tile_rows) {
      tiles[atomicAdd(count, 1U)] = tile;
      tile.count = 0;
    }
  }
  if (tile.count != 0)
    tiles[atomicAdd(count, 1U)] = tile;
}

// FP32 scaled input -> FP32 butterfly -> FP16 RNE, matching X2b. Each
// assignment uses its own expert's scales, including independent gate/up.
__global__ void RotateAssignments(const ex::DeviceLinear* gate,
                                  const ex::DeviceLinear* up,
                                  const std::int32_t* ids, const float* x,
                                  std::uint16_t* output, unsigned used,
                                  unsigned width, unsigned assignments) {
  const unsigned assignment = blockIdx.y;
  const unsigned projection = blockIdx.z;
  const auto& l = (projection == 0 ? gate : up)[ids[assignment]];
  const unsigned lane = threadIdx.x;
  const unsigned first = blockIdx.x * 128 + lane * 4;
  const auto* suh = reinterpret_cast<const _Float16*>(l.suh);
  const auto row = assignment / used;
  float v[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    v[j] = x[std::size_t{row} * width + first + j] *
           static_cast<float>(suh[first + j]);
    asm volatile("" : "+v"(v[j]));
  }
  arithmetic::Butterfly128(v, lane);
  auto* out = reinterpret_cast<_Float16*>(output) +
              (std::size_t{projection} * assignments + assignment) * width;
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    float scaled = v[j] * arithmetic::kHadamardScale;
    asm volatile("" : "+v"(scaled));
    out[first + j] = static_cast<_Float16>(scaled);
  }
}

// The down slab can be entirely FP16 subnormal after SwiGLU. Widen its
// operands before arithmetic and pin the boundary so LLVM cannot fold it
// back into half/mixed instructions under this TU's fast-math flags.
__device__ __forceinline__ float DotPairF32(Half2 w, Half2 x, float acc) {
  float w0 = static_cast<float>(w.x);
  float w1 = static_cast<float>(w.y);
  float x0 = static_cast<float>(x.x);
  float x1 = static_cast<float>(x.y);
  asm volatile("" : "+v"(w0), "+v"(w1), "+v"(x0), "+v"(x1));
  acc = __builtin_fmaf(w0, x0, acc);
  return __builtin_fmaf(w1, x1, acc);
}

// One wave owns one output tile and up to eight actual rows of one expert.
// No floating-point atomics or masked reads from absent assignment rows.
template<int Bits, bool FullPrecision, unsigned MaxRows = 8,
         bool Scaled = false, bool ExactRows = false>
__device__ __forceinline__ void RoutedGemvBody(
    const ex::DeviceLinear* gate, const ex::DeviceLinear* up,
    const Exl3RouteTile* tiles, const unsigned* count,
    const std::uint16_t* input, float* gate_y, float* up_y, unsigned k,
    unsigned n, unsigned assignments, const std::int32_t* exponents) {
  if (blockIdx.y >= *count)
    return;
  const auto tile_rows = tiles[blockIdx.y];
  const unsigned projection = blockIdx.z;
  const auto& l = (projection == 0 ? gate : up)[tile_rows.expert];
  const unsigned lane = threadIdx.x;
  const unsigned r0 = (lane % 4) * 2;
  const unsigned nt = blockIdx.x;
  float a[MaxRows]{}, b[MaxRows]{};
  const auto lw = arithmetic::MakeLaneWindow<Bits>(lane);
  const auto* trellis = reinterpret_cast<const std::uint32_t*>(l.trellis);
  const auto* x = reinterpret_cast<const _Float16*>(input) +
                  std::size_t{projection} * assignments * k;
  // Tiles are fetched kAhead at a time so a wave keeps several loads in
  // flight (one tile per iteration left decode latency-bound, ~150 GB/s);
  // they are decoded and accumulated in K order exactly as before.
  constexpr unsigned kAhead = 4;
  const unsigned k_tiles = k / 16;
  for (unsigned kt0 = 0; kt0 < k_tiles; kt0 += kAhead) {
    std::uint32_t raw[kAhead][3];
#pragma unroll
    for (unsigned u = 0; u < kAhead; ++u) {
      const unsigned kt = min(kt0 + u, k_tiles - 1);
      const auto* tile = trellis + (std::size_t{kt} * (n / 16) + nt) *
                                       arithmetic::TileDwords(Bits);
      raw[u][0] = tile[lw.o0];
      raw[u][1] = tile[lw.o1];
      raw[u][2] = tile[lw.o2];
    }
#pragma unroll
    for (unsigned u = 0; u < kAhead; ++u) {
      const unsigned kt = kt0 + u;
      if (kt >= k_tiles)
        break;
      Half2 w[4];
      {
        std::uint32_t state[8];
        arithmetic::LaneStates<Bits>(raw[u][0], raw[u][1], raw[u][2], lw.shift,
                                     state);
        std::uint32_t p[8];
#pragma unroll
        for (int j = 0; j < 8; ++j)
          p[j] = (state[j] & 0xffffU) * arithmetic::kMul1Multiplier;
#pragma unroll
        for (int q = 0; q < 4; ++q)
          w[q] = arithmetic::DecodePairMul1(p[2 * q], p[2 * q + 1]);
      }
      // Fixed row indices keep the accumulators and route slots in VGPRs.
      // A dynamically indexed local array otherwise promotes to 40 KiB LDS
      // per one-wave block on the pinned compiler, starving verification.
#pragma unroll
      for (unsigned r = 0; r < MaxRows; ++r) {
        if (!ExactRows && r >= tile_rows.count)
          continue;
        const auto* row =
            x + std::size_t{tile_rows.slots[r]} * k + kt * 16 + r0;
        const Half2 v01{row[0], row[1]};
        const Half2 v89{row[8], row[9]};
        if (FullPrecision || (Scaled && exponents[tile_rows.slots[r]] < 0)) {
          a[r] = DotPairF32(w[0], v01, a[r]);
          a[r] = DotPairF32(w[1], v89, a[r]);
          b[r] = DotPairF32(w[2], v01, b[r]);
          b[r] = DotPairF32(w[3], v89, b[r]);
        } else {
          a[r] = __builtin_amdgcn_fdot2(w[0], v01, a[r], false);
          a[r] = __builtin_amdgcn_fdot2(w[1], v89, a[r], false);
          b[r] = __builtin_amdgcn_fdot2(w[2], v01, b[r], false);
          b[r] = __builtin_amdgcn_fdot2(w[3], v89, b[r], false);
        }
      }
    }
  }
#pragma unroll
  for (unsigned r = 0; r < MaxRows; ++r) {
    if (!ExactRows && r >= tile_rows.count)
      continue;
    a[r] += __shfl_xor(a[r], 1, 32);
    b[r] += __shfl_xor(b[r], 1, 32);
    a[r] += __shfl_xor(a[r], 2, 32);
    b[r] += __shfl_xor(b[r], 2, 32);
    if (lane % 4 == 0) {
      auto* y = (projection == 0 ? gate_y : up_y) +
                std::size_t{tile_rows.slots[r]} * n + nt * 16;
      float scale = 1.0F;
      if constexpr (Scaled) {
        const int exponent = exponents[tile_rows.slots[r]];
        if (exponent > 0)
          scale = __builtin_bit_cast(float, unsigned(127 - exponent) << 23);
      }
      y[lane / 4] = a[r] * scale;
      y[lane / 4 + 8] = b[r] * scale;
    }
  }
}

// Select the live-row loop once per expert tile, avoiding absent-row checks
// in every K step. Rate-specific decoding stays in the body; only the row
// loop is specialized, independently of the bit rate.
// Keep generic dispatch for other shapes and unscaled full-precision down.
template<int Bits, bool FullPrecision, unsigned MaxRows = 8,
         bool Scaled = false>
__global__ void RoutedGemv(const ex::DeviceLinear* gate,
                           const ex::DeviceLinear* up,
                           const Exl3RouteTile* tiles, const unsigned* count,
                           const std::uint16_t* input, float* gate_y,
                           float* up_y, unsigned k, unsigned n,
                           unsigned assignments,
                           const std::int32_t* exponents) {
  if (blockIdx.y >= *count)
    return;
  if constexpr (!FullPrecision && (MaxRows == 3 || MaxRows == 4)) {
    switch (tiles[blockIdx.y].count) {
      case 1:
        RoutedGemvBody<Bits, FullPrecision, 1, Scaled, true>(
            gate, up, tiles, count, input, gate_y, up_y, k, n, assignments,
            exponents);
        return;
      case 2:
        RoutedGemvBody<Bits, FullPrecision, 2, Scaled, true>(
            gate, up, tiles, count, input, gate_y, up_y, k, n, assignments,
            exponents);
        return;
      case 3:
        RoutedGemvBody<Bits, FullPrecision, 3, Scaled, true>(
            gate, up, tiles, count, input, gate_y, up_y, k, n, assignments,
            exponents);
        return;
      case 4:
        if constexpr (MaxRows == 4) {
          RoutedGemvBody<Bits, FullPrecision, 4, Scaled, true>(
              gate, up, tiles, count, input, gate_y, up_y, k, n, assignments,
              exponents);
          return;
        }
    }
  }
  RoutedGemvBody<Bits, FullPrecision, MaxRows, Scaled>(
      gate, up, tiles, count, input, gate_y, up_y, k, n, assignments,
      exponents);
}

__global__ void RestoreDownAssignments(const ex::DeviceLinear* down,
                                       const std::int32_t* ids, float* output,
                                       unsigned width) {
  const unsigned row = blockIdx.y;
  const unsigned lane = threadIdx.x;
  const unsigned first = blockIdx.x * 128 + lane * 4;
  const auto* svh = reinterpret_cast<const _Float16*>(down[ids[row]].svh);
  const auto offset = std::size_t{row} * width + first;
  float v[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j)
    v[j] = output[offset + j];
  arithmetic::Butterfly128(v, lane);
#pragma unroll
  for (unsigned j = 0; j < 4; ++j)
    output[offset + j] = (v[j] * arithmetic::kHadamardScale) *
                         static_cast<float>(svh[first + j]);
}

// Scale only the already-rounded FP16 values. min/max are absolute FP16
// bit patterns, so this classification is independent of half-denormal mode.
__device__ int DownExponent(unsigned minimum, unsigned maximum) {
  if (maximum >= 0x7c00U)
    return -1;
  if (minimum >= 0x400U)  // Includes the all-zero row (minimum == 0x7fff).
    return 0;
  const int shift = __clz(minimum) - 21;  // 1..10, to make minimum normal.
  return (maximum >> 10) + shift <= 30 ? shift : -1;
}

__device__ std::uint16_t ScaleHalf(std::uint16_t bits, int exponent) {
  const unsigned value = bits & 0x7fffU;
  if (value == 0 || exponent <= 0)
    return bits;
  if (value >= 0x400U)
    return bits + (exponent << 10);
  const int normalize = __clz(value) - 21;
  const unsigned significand = value << normalize;
  return (bits & 0x8000U) | ((1 + exponent - normalize) << 10) |
         (significand & 0x3ffU);
}

__global__ void ScaleDownRows(std::uint16_t* input, std::int32_t* exponents,
                              unsigned width) {
  const unsigned row = blockIdx.x;
  unsigned minimum = 0x7fffU, maximum = 0;
  for (unsigned i = threadIdx.x; i < width; i += blockDim.x) {
    const unsigned value = input[std::size_t{row} * width + i] & 0x7fffU;
    if (value != 0)
      minimum = min(minimum, value);
    maximum = max(maximum, value);
  }
  for (unsigned offset = 16; offset != 0; offset >>= 1) {
    minimum = min(minimum, __shfl_xor(minimum, offset, 32));
    maximum = max(maximum, __shfl_xor(maximum, offset, 32));
  }
  __shared__ unsigned minima[8], maxima[8];
  if (threadIdx.x % 32 == 0) {
    minima[threadIdx.x / 32] = minimum;
    maxima[threadIdx.x / 32] = maximum;
  }
  __syncthreads();
  minimum = 0x7fffU;
  maximum = 0;
  for (unsigned i = 0; i < 8; ++i) {
    minimum = min(minimum, minima[i]);
    maximum = max(maximum, maxima[i]);
  }
  const int exponent = DownExponent(minimum, maximum);
  if (threadIdx.x == 0)
    exponents[row] = exponent;
  if (exponent > 0)
    for (unsigned i = threadIdx.x; i < width; i += blockDim.x) {
      auto& value = input[std::size_t{row} * width + i];
      value = ScaleHalf(value, exponent);
    }
}

// Keep the same FP32 activation store boundary as RestoreAssignments, then
// immediately consume that rounded value for the down-input rotation.
__global__ void RestoreActivateRotate(const ex::DeviceLinear* gate,
                                      const ex::DeviceLinear* up,
                                      const ex::DeviceLinear* down,
                                      const std::int32_t* ids, float* gate_y,
                                      const float* up_y, std::uint16_t* output,
                                      unsigned width) {
  const unsigned row = blockIdx.y, lane = threadIdx.x;
  const unsigned first = blockIdx.x * 128 + lane * 4;
  const auto& g = gate[ids[row]];
  const auto& u = up[ids[row]];
  const auto& d = down[ids[row]];
  const auto* gs = reinterpret_cast<const _Float16*>(g.svh);
  const auto* us = reinterpret_cast<const _Float16*>(u.svh);
  const auto* ds = reinterpret_cast<const _Float16*>(d.suh);
  const auto offset = std::size_t{row} * width + first;
  float gv[4], uv[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    gv[j] = gate_y[offset + j];
    uv[j] = up_y[offset + j];
  }
  arithmetic::Butterfly128(gv, lane);
  arithmetic::Butterfly128(uv, lane);
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const float value = (gv[j] * arithmetic::kHadamardScale) *
                        static_cast<float>(gs[first + j]);
    const float other = (uv[j] * arithmetic::kHadamardScale) *
                        static_cast<float>(us[first + j]);
    float activation = (value / (1.0F + expf(-value))) * other;
    asm volatile("" : "+v"(activation));
    gate_y[offset + j] = activation;
    gv[j] = activation * static_cast<float>(ds[first + j]);
    asm volatile("" : "+v"(gv[j]));
  }
  arithmetic::Butterfly128(gv, lane);
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    float scaled = gv[j] * arithmetic::kHadamardScale;
    asm volatile("" : "+v"(scaled));
    reinterpret_cast<_Float16*>(output)[offset + j] =
        static_cast<_Float16>(scaled);
  }
}

// RestoreActivateRotate and ScaleDownRows in one pass: one block per
// assignment (a wave per 128-wide segment) forms the FP16 down input in
// registers with RestoreActivateRotate's expressions, reduces its magnitude
// range over the row exactly as ScaleDownRows does on the stored halves, and
// stores the scaled halves once. The FP32 activation has no reader here.
__global__ __launch_bounds__(1024) void RestoreActivateScale(
    const ex::DeviceLinear* gate, const ex::DeviceLinear* up,
    const ex::DeviceLinear* down, const std::int32_t* ids, const float* gate_y,
    const float* up_y, std::uint16_t* output, std::int32_t* exponents,
    unsigned width) {
  const unsigned row = blockIdx.x, lane = threadIdx.x % 32;
  const unsigned wave = threadIdx.x / 32, waves = blockDim.x / 32;
  const unsigned first = wave * 128 + lane * 4;
  const auto& g = gate[ids[row]];
  const auto& u = up[ids[row]];
  const auto& d = down[ids[row]];
  const auto* gs = reinterpret_cast<const _Float16*>(g.svh);
  const auto* us = reinterpret_cast<const _Float16*>(u.svh);
  const auto* ds = reinterpret_cast<const _Float16*>(d.suh);
  const auto offset = std::size_t{row} * width + first;
  float gv[4], uv[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    gv[j] = gate_y[offset + j];
    uv[j] = up_y[offset + j];
  }
  arithmetic::Butterfly128(gv, lane);
  arithmetic::Butterfly128(uv, lane);
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const float value = (gv[j] * arithmetic::kHadamardScale) *
                        static_cast<float>(gs[first + j]);
    const float other = (uv[j] * arithmetic::kHadamardScale) *
                        static_cast<float>(us[first + j]);
    float activation = (value / (1.0F + expf(-value))) * other;
    asm volatile("" : "+v"(activation));
    gv[j] = activation * static_cast<float>(ds[first + j]);
    asm volatile("" : "+v"(gv[j]));
  }
  arithmetic::Butterfly128(gv, lane);
  std::uint16_t bits[4];
  unsigned minimum = 0x7fffU, maximum = 0;
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    float scaled = gv[j] * arithmetic::kHadamardScale;
    asm volatile("" : "+v"(scaled));
    bits[j] = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(scaled));
    const unsigned value = bits[j] & 0x7fffU;
    if (value != 0)
      minimum = min(minimum, value);
    maximum = max(maximum, value);
  }
  for (unsigned o = 16; o != 0; o >>= 1) {
    minimum = min(minimum, __shfl_xor(minimum, o, 32));
    maximum = max(maximum, __shfl_xor(maximum, o, 32));
  }
  __shared__ unsigned minima[32], maxima[32];
  if (lane == 0) {
    minima[wave] = minimum;
    maxima[wave] = maximum;
  }
  __syncthreads();
  minimum = 0x7fffU;
  maximum = 0;
  for (unsigned i = 0; i < waves; ++i) {
    minimum = min(minimum, minima[i]);
    maximum = max(maximum, maxima[i]);
  }
  const int exponent = DownExponent(minimum, maximum);
  if (threadIdx.x == 0)
    exponents[row] = exponent;
#pragma unroll
  for (unsigned j = 0; j < 4; ++j)
    output[offset + j] = exponent > 0 ? ScaleHalf(bits[j], exponent) : bits[j];
}

// Bucket construction, all on the device and atomic-free, so it is
// deterministic and graph-replay safe. One 256-thread block per expert scans
// the IDs (L2-resident) to count, then again to emit its slots in assignment
// order; one block turns the counts into offsets and descriptors and zeroes
// the inactive tail. (A thread per expert over all IDs, the former shape,
// left the GPU nearly idle: 0.55 + 0.87 ms per layer at pp2048.)
__global__ __launch_bounds__(256) void CountBuckets(const std::int32_t* ids,
                                                    unsigned assignments,
                                                    unsigned* offsets) {
  const unsigned expert = blockIdx.x;
  unsigned count = 0;
  for (unsigned slot = threadIdx.x; slot < assignments; slot += 256)
    count += ids[slot] == static_cast<int>(expert);
  for (unsigned offset = 16; offset != 0; offset >>= 1)
    count += __shfl_xor(count, offset, 32);
  __shared__ unsigned partial[8];
  if (threadIdx.x % 32 == 0)
    partial[threadIdx.x / 32] = count;
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned total = 0;
    for (unsigned w = 0; w < 8; ++w)
      total += partial[w];
    offsets[expert + 1] = total;
  }
}

// A hot expert's rows (more than 64, with 64-row tiles) are split evenly
// into buckets of at most 128 rows for the 128-token X3 tile: fewer
// re-decodes of its weights. The bucket a row lands in never changes its
// value (every tile keeps each output's K order).
__device__ __forceinline__ unsigned BucketRows(unsigned rows,
                                               unsigned tile_rows) {
  return tile_rows == 64 && rows > 64 ? 128 : tile_rows;
}

__global__ __launch_bounds__(1024) void DescribeBuckets(
    unsigned* offsets, unsigned experts, unsigned tile_rows,
    Exl3ExpertBucket* buckets, unsigned* bucket_count, unsigned capacity) {
  // Exclusive scans of rows and buckets per expert, 1024 experts at a time.
  __shared__ unsigned rows_scan[1024], tiles_scan[1024];
  __shared__ unsigned carry_rows, carry_tiles;
  const unsigned tid = threadIdx.x;
  if (tid == 0) {
    carry_rows = 0;
    carry_tiles = 0;
  }
  __syncthreads();
  for (unsigned base = 0; base < experts; base += 1024) {
    const unsigned expert = base + tid;
    const unsigned rows = expert < experts ? offsets[expert + 1] : 0;
    const unsigned size = BucketRows(rows, tile_rows);
    const unsigned tiles = (rows + size - 1) / size;
    rows_scan[tid] = rows;
    tiles_scan[tid] = tiles;
    __syncthreads();
    // Hillis-Steele inclusive scan (exact integer sums).
    for (unsigned step = 1; step < 1024; step <<= 1) {
      const unsigned r = tid >= step ? rows_scan[tid - step] : 0;
      const unsigned t = tid >= step ? tiles_scan[tid - step] : 0;
      __syncthreads();
      rows_scan[tid] += r;
      tiles_scan[tid] += t;
      __syncthreads();
    }
    const unsigned first = carry_rows + rows_scan[tid] - rows;
    unsigned bucket = carry_tiles + tiles_scan[tid] - tiles;
    if (expert < experts) {
      offsets[expert + 1] = first + rows;
      if (size == tile_rows)
        for (unsigned row = 0; row < rows; row += tile_rows)
          buckets[bucket++] = {expert, min(tile_rows, rows - row), first + row};
      else
        for (unsigned t = 0, row = 0; t < tiles; ++t) {
          const unsigned count = (rows - row) / (tiles - t);
          buckets[bucket++] = {expert, count, first + row};
          row += count;
        }
    }
    __syncthreads();
    if (tid == 1023) {
      carry_rows += rows_scan[1023];
      carry_tiles += tiles_scan[1023];
    }
    __syncthreads();
  }
  if (tid == 0) {
    offsets[0] = 0;
    *bucket_count = carry_tiles;
  }
  for (unsigned i = carry_tiles + tid; i < capacity; i += 1024)
    buckets[i] = {};
}

__global__ __launch_bounds__(256) void FillBucketSlots(const std::int32_t* ids,
                                                       unsigned assignments,
                                                       const unsigned* offsets,
                                                       std::int32_t* slots,
                                                       std::int32_t* inverse) {
  // Stable compaction of this expert's slots, 256 IDs per step: a wave
  // ballot ranks the lanes, the eight wave totals rank the waves.
  const unsigned expert = blockIdx.x, tid = threadIdx.x;
  const unsigned lane = tid % 32, wave = tid / 32;
  __shared__ unsigned wave_counts[8];
  unsigned next = offsets[expert];
  for (unsigned base = 0; base < assignments; base += 256) {
    const unsigned slot = base + tid;
    const bool hit =
        slot < assignments && ids[slot] == static_cast<int>(expert);
    const auto mask = static_cast<unsigned>(__ballot(hit));
    const unsigned before = __popc(mask & ((1U << lane) - 1U));
    if (lane == 0)
      wave_counts[wave] = __popc(mask);
    __syncthreads();
    unsigned wave_base = 0, total = 0;
    for (unsigned w = 0; w < 8; ++w) {
      wave_base += w < wave ? wave_counts[w] : 0;
      total += wave_counts[w];
    }
    if (hit) {
      const unsigned packed = next + wave_base + before;
      slots[packed] = static_cast<std::int32_t>(slot);
      if (inverse != nullptr)
        inverse[slot] = static_cast<std::int32_t>(packed);
    }
    next += total;
    __syncthreads();
  }
}

__device__ __forceinline__ unsigned BucketExpert(const unsigned* offsets,
                                                 unsigned experts,
                                                 unsigned row) {
  unsigned lo = 0, hi = experts;
  while (lo + 1 < hi) {
    const unsigned mid = (lo + hi) / 2;
    if (offsets[mid] <= row)
      lo = mid;
    else
      hi = mid;
  }
  return lo;
}

__global__ void RotateBuckets(const ex::DeviceLinear* bank, const float* input,
                              std::uint16_t* output, const unsigned* offsets,
                              const std::int32_t* slots, unsigned experts,
                              unsigned width, unsigned used, bool down) {
  const unsigned packed = blockIdx.y;
  if (packed >= offsets[experts])
    return;
  const unsigned lo = BucketExpert(offsets, experts, packed);
  const auto& l = bank[lo];
  const unsigned row = down ? slots[packed] : slots[packed] / used;
  const unsigned lane = threadIdx.x, first = blockIdx.x * 128 + lane * 4;
  const auto* suh = reinterpret_cast<const _Float16*>(l.suh);
  float v[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    v[j] = input[std::size_t{row} * width + first + j] *
           static_cast<float>(suh[first + j]);
    asm volatile("" : "+v"(v[j]));
  }
  arithmetic::Butterfly128(v, lane);
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    float scaled = v[j] * arithmetic::kHadamardScale;
    asm volatile("" : "+v"(scaled));
    reinterpret_cast<_Float16*>(
        output)[std::size_t{packed} * width + first + j] =
        static_cast<_Float16>(scaled);
  }
}

// Gate and up rotations of the same gathered rows in one pass: each input
// element is read once and rotated with both banks' suh, the arithmetic of
// two RotateBuckets launches.
__global__ void RotateGateUpBuckets(const ex::DeviceLinear* gate,
                                    const ex::DeviceLinear* up,
                                    const float* input, std::uint16_t* gate_out,
                                    std::uint16_t* up_out,
                                    const unsigned* offsets,
                                    const std::int32_t* slots, unsigned experts,
                                    unsigned width, unsigned used) {
  const unsigned packed = blockIdx.y;
  if (packed >= offsets[experts])
    return;
  const unsigned lo = BucketExpert(offsets, experts, packed);
  const unsigned row = slots[packed] / used;
  const unsigned lane = threadIdx.x, first = blockIdx.x * 128 + lane * 4;
  const auto* gs = reinterpret_cast<const _Float16*>(gate[lo].suh);
  const auto* us = reinterpret_cast<const _Float16*>(up[lo].suh);
  float g[4], u[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const float x = input[std::size_t{row} * width + first + j];
    g[j] = x * static_cast<float>(gs[first + j]);
    asm volatile("" : "+v"(g[j]));
    u[j] = x * static_cast<float>(us[first + j]);
    asm volatile("" : "+v"(u[j]));
  }
  arithmetic::Butterfly128(g, lane);
  arithmetic::Butterfly128(u, lane);
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    float a = g[j] * arithmetic::kHadamardScale;
    asm volatile("" : "+v"(a));
    float b = u[j] * arithmetic::kHadamardScale;
    asm volatile("" : "+v"(b));
    const auto at = std::size_t{packed} * width + first + j;
    reinterpret_cast<_Float16*>(gate_out)[at] = static_cast<_Float16>(a);
    reinterpret_cast<_Float16*>(up_out)[at] = static_cast<_Float16>(b);
  }
}

// RotateGateUpBuckets per token: each 128-wide segment of a token's row is
// read once and rotated for each of its `used` assignments (with that
// expert's suh) into the assignment's packed row, found through the inverse
// slot map. The same arithmetic per element; 1/used of the gathered reads.
__global__ __launch_bounds__(256) void RotateGateUpTokens(
    const ex::DeviceLinear* gate, const ex::DeviceLinear* up,
    const float* __restrict__ input, std::uint16_t* __restrict__ gate_out,
    std::uint16_t* __restrict__ up_out, const std::int32_t* __restrict__ ids,
    const std::int32_t* __restrict__ inverse, unsigned width, unsigned used) {
  const unsigned row = blockIdx.y, lane = threadIdx.x % 32;
  const unsigned segment = blockIdx.x * 8 + threadIdx.x / 32;
  if (segment * 128 >= width)
    return;
  const unsigned first = segment * 128 + lane * 4;
  float x[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j)
    x[j] = input[std::size_t{row} * width + first + j];
  // Lane s holds assignment s's packed row and suh pointers (used <= 32).
  const unsigned mine = row * used + min(lane, used - 1);
  const int my_packed = inverse[mine];
  const auto my_expert = static_cast<unsigned>(ids[mine]);
  const auto my_gs = reinterpret_cast<std::uintptr_t>(gate[my_expert].suh);
  const auto my_us = reinterpret_cast<std::uintptr_t>(up[my_expert].suh);
#pragma unroll 2
  for (unsigned s = 0; s < used; ++s) {
    const auto packed = static_cast<unsigned>(__shfl(my_packed, s, 32));
    const auto* gs = reinterpret_cast<const _Float16*>(__shfl(my_gs, s, 32));
    const auto* us = reinterpret_cast<const _Float16*>(__shfl(my_us, s, 32));
    float g[4], u[4];
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      g[j] = x[j] * static_cast<float>(gs[first + j]);
      asm volatile("" : "+v"(g[j]));
      u[j] = x[j] * static_cast<float>(us[first + j]);
      asm volatile("" : "+v"(u[j]));
    }
    arithmetic::Butterfly128(g, lane);
    arithmetic::Butterfly128(u, lane);
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      float a = g[j] * arithmetic::kHadamardScale;
      asm volatile("" : "+v"(a));
      float b = u[j] * arithmetic::kHadamardScale;
      asm volatile("" : "+v"(b));
      const auto at = std::size_t{packed} * width + first + j;
      reinterpret_cast<_Float16*>(gate_out)[at] = static_cast<_Float16>(a);
      reinterpret_cast<_Float16*>(up_out)[at] = static_cast<_Float16>(b);
    }
  }
}

// RestoreActivateRotateBuckets<_Float16> and ScaleDownRows in one pass: one
// block per packed row (a wave per 128-wide segment) forms the FP16 down
// input in registers, reduces its magnitude range over the row exactly as
// ScaleDownRows does on the stored halves, and stores the scaled halves once.
__global__ __launch_bounds__(1024) void RestoreActivateScaleBuckets(
    const ex::DeviceLinear* gate, const ex::DeviceLinear* up,
    const ex::DeviceLinear* down, const unsigned* offsets, unsigned experts,
    const _Float16* gate_y, const _Float16* up_y, std::uint16_t* output,
    std::int32_t* exponents, unsigned width) {
  const unsigned row = blockIdx.x, lane = threadIdx.x % 32;
  const unsigned wave = threadIdx.x / 32, waves = blockDim.x / 32;
  if (row >= offsets[experts])
    return;
  const unsigned lo = BucketExpert(offsets, experts, row);
  const unsigned first = wave * 128 + lane * 4;
  const auto* gs = reinterpret_cast<const _Float16*>(gate[lo].svh);
  const auto* us = reinterpret_cast<const _Float16*>(up[lo].svh);
  const auto* ds = reinterpret_cast<const _Float16*>(down[lo].suh);
  const auto offset = std::size_t{row} * width + first;
  float gv[4], uv[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    gv[j] = static_cast<float>(gate_y[offset + j]);
    uv[j] = static_cast<float>(up_y[offset + j]);
  }
  arithmetic::Butterfly128(gv, lane);
  arithmetic::Butterfly128(uv, lane);
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const float value = (gv[j] * arithmetic::kHadamardScale) *
                        static_cast<float>(gs[first + j]);
    const float other = (uv[j] * arithmetic::kHadamardScale) *
                        static_cast<float>(us[first + j]);
    float activation = (value / (1.0F + expf(-value))) * other;
    asm volatile("" : "+v"(activation));
    gv[j] = activation * static_cast<float>(ds[first + j]);
    asm volatile("" : "+v"(gv[j]));
  }
  arithmetic::Butterfly128(gv, lane);
  std::uint16_t bits[4];
  unsigned minimum = 0x7fffU, maximum = 0;
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    float scaled = gv[j] * arithmetic::kHadamardScale;
    asm volatile("" : "+v"(scaled));
    bits[j] = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(scaled));
    const unsigned value = bits[j] & 0x7fffU;
    if (value != 0)
      minimum = min(minimum, value);
    maximum = max(maximum, value);
  }
  for (unsigned o = 16; o != 0; o >>= 1) {
    minimum = min(minimum, __shfl_xor(minimum, o, 32));
    maximum = max(maximum, __shfl_xor(maximum, o, 32));
  }
  __shared__ unsigned minima[32], maxima[32];
  if (lane == 0) {
    minima[wave] = minimum;
    maxima[wave] = maximum;
  }
  __syncthreads();
  minimum = 0x7fffU;
  maximum = 0;
  for (unsigned i = 0; i < waves; ++i) {
    minimum = min(minimum, minima[i]);
    maximum = max(maximum, maxima[i]);
  }
  const int exponent = DownExponent(minimum, maximum);
  if (threadIdx.x == 0)
    exponents[row] = exponent;
#pragma unroll
  for (unsigned j = 0; j < 4; ++j)
    output[offset + j] = exponent > 0 ? ScaleHalf(bits[j], exponent) : bits[j];
}

// InT: float (Exl3PrefillExperts) or _Float16 (the FP16 raw path); the
// activation itself is kept only in the float form.
template<typename InT>
__global__ void RestoreActivateRotateBuckets(
    const ex::DeviceLinear* gate, const ex::DeviceLinear* up,
    const ex::DeviceLinear* down, const unsigned* offsets, unsigned experts,
    InT* gate_y, const InT* up_y, std::uint16_t* output, unsigned width,
    bool keep_activation) {
  const unsigned row = blockIdx.y, lane = threadIdx.x;
  if (row >= offsets[experts])
    return;
  const unsigned lo = BucketExpert(offsets, experts, row);
  const unsigned first = blockIdx.x * 128 + lane * 4;
  const auto& g = gate[lo];
  const auto& u = up[lo];
  const auto& d = down[lo];
  const auto* gs = reinterpret_cast<const _Float16*>(g.svh);
  const auto* us = reinterpret_cast<const _Float16*>(u.svh);
  const auto* ds = reinterpret_cast<const _Float16*>(d.suh);
  const auto offset = std::size_t{row} * width + first;
  float gv[4], uv[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    gv[j] = static_cast<float>(gate_y[offset + j]);
    uv[j] = static_cast<float>(up_y[offset + j]);
  }
  arithmetic::Butterfly128(gv, lane);
  arithmetic::Butterfly128(uv, lane);
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const float value = (gv[j] * arithmetic::kHadamardScale) *
                        static_cast<float>(gs[first + j]);
    const float other = (uv[j] * arithmetic::kHadamardScale) *
                        static_cast<float>(us[first + j]);
    float activation = (value / (1.0F + expf(-value))) * other;
    asm volatile("" : "+v"(activation));
    // Only the qualification trace reads the FP32 activation back.
    if constexpr (std::is_same_v<InT, float>) {
      if (keep_activation)
        gate_y[offset + j] = activation;
    }
    gv[j] = activation * static_cast<float>(ds[first + j]);
    asm volatile("" : "+v"(gv[j]));
  }
  arithmetic::Butterfly128(gv, lane);
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    float scaled = gv[j] * arithmetic::kHadamardScale;
    asm volatile("" : "+v"(scaled));
    reinterpret_cast<_Float16*>(output)[offset + j] =
        static_cast<_Float16>(scaled);
  }
}

// WMMA returns raw inner dots. Rows with no safe normal FP16 scaling are
// overwritten by the same widened FMA/reduction order as routed decode.
// A bounded block grid checks each row before assigning its output tiles to
// the block's eight waves. The all-safe case launches <= 256 blocks, rather
// than one block for every (assignment, output tile) pair.
template<int Bits, typename OutT = float>
__global__ void BucketDownFallback(const ex::DeviceLinear* bank,
                                   const unsigned* offsets, unsigned experts,
                                   const std::int32_t* exponents,
                                   const std::uint16_t* input, OutT* output,
                                   unsigned k, unsigned n, unsigned assignments,
                                   unsigned* counts) {
  if (counts != nullptr && threadIdx.x == 0)
    atomicAdd(counts, 1U);
  const unsigned lane = threadIdx.x % 32, r0 = (lane % 4) * 2;
  const auto lw = arithmetic::MakeLaneWindow<Bits>(lane);
  for (unsigned row = blockIdx.x; row < assignments; row += gridDim.x) {
    if (row >= offsets[experts] || exponents[row] >= 0)
      continue;
    if (counts != nullptr && threadIdx.x == 0)
      atomicAdd(counts + 1, 1U);
    const unsigned expert = BucketExpert(offsets, experts, row);
    const auto& l = bank[expert];
    const auto* x =
        reinterpret_cast<const _Float16*>(input) + std::size_t{row} * k;
    const auto* trellis = reinterpret_cast<const std::uint32_t*>(l.trellis);
    for (unsigned nt = threadIdx.x / 32; nt < n / 16; nt += blockDim.x / 32) {
      float a = 0, b = 0;
      for (unsigned kt = 0; kt < k / 16; ++kt) {
        Half2 w[4];
        arithmetic::DecodeLaneMul1<Bits>(
            trellis + (std::size_t{kt} * (n / 16) + nt) *
                          arithmetic::TileDwords(Bits),
            lw, w);
        const auto* group = x + kt * 16 + r0;
        const Half2 v01{group[0], group[1]}, v89{group[8], group[9]};
        a = DotPairF32(w[0], v01, a);
        a = DotPairF32(w[1], v89, a);
        b = DotPairF32(w[2], v01, b);
        b = DotPairF32(w[3], v89, b);
      }
      a += __shfl_xor(a, 1, 32);
      b += __shfl_xor(b, 1, 32);
      a += __shfl_xor(a, 2, 32);
      b += __shfl_xor(b, 2, 32);
      if (lane % 4 == 0) {
        output[std::size_t{row} * n + nt * 16 + lane / 4] =
            static_cast<OutT>(a);
        output[std::size_t{row} * n + nt * 16 + lane / 4 + 8] =
            static_cast<OutT>(b);
      }
    }
  }
}

// destination == nullptr restores in place (packed order); otherwise the
// restored row goes to destination row slots[row] (the bucket scatter fused).
__global__ void RestoreDownBuckets(const ex::DeviceLinear* bank,
                                   const unsigned* offsets, unsigned experts,
                                   const std::int32_t* exponents, float* output,
                                   unsigned width, float* destination,
                                   const std::int32_t* slots) {
  const unsigned row = blockIdx.y, lane = threadIdx.x;
  if (row >= offsets[experts])
    return;
  const unsigned lo = BucketExpert(offsets, experts, row);
  const unsigned first = blockIdx.x * 128 + lane * 4;
  const auto offset = std::size_t{row} * width + first;
  const auto* svh = reinterpret_cast<const _Float16*>(bank[lo].svh);
  const int exponent = exponents[row];
  const float scale =
      exponent > 0 ? __builtin_bit_cast(float, unsigned(127 - exponent) << 23)
                   : 1.0F;
  float v[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    v[j] = output[offset + j] * scale;
    asm volatile("" : "+v"(v[j]));
  }
  arithmetic::Butterfly128(v, lane);
  float* target =
      destination == nullptr
          ? output + offset
          : destination +
                std::size_t{static_cast<unsigned>(slots[row])} * width + first;
#pragma unroll
  for (unsigned j = 0; j < 4; ++j)
    target[j] = (v[j] * arithmetic::kHadamardScale) *
                static_cast<float>(svh[first + j]);
}

__global__ void HeadInput(ex::DeviceLinear l, const float* x,
                          std::uint16_t* output) {
  const unsigned lane = threadIdx.x;
  const unsigned first = blockIdx.x * 128 + lane * 4;
  const auto* suh = reinterpret_cast<const _Float16*>(l.suh);
  auto* out = reinterpret_cast<_Float16*>(output);
  float v[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    v[j] = x[first + j] * static_cast<float>(suh[first + j]);
    asm volatile("" : "+v"(v[j]));
  }
  arithmetic::Butterfly128(v, lane);
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    float scaled = v[j] * arithmetic::kHadamardScale;
    asm volatile("" : "+v"(scaled));
    out[arithmetic::LaneMajorOffset(first + j, 0, 1)] =
        static_cast<_Float16>(scaled);
  }
}

__global__ void SelectSegments(const std::int32_t* ids, unsigned extras,
                               unsigned prefix, unsigned n, unsigned* list) {
  if (threadIdx.x != 0)
    return;
  unsigned count = 0;
  for (unsigned i = 0; i < extras; ++i) {
    const int id = ids[i];
    if (id < static_cast<int>(prefix) || id >= static_cast<int>(n))
      continue;
    const unsigned segment = static_cast<unsigned>(id) / 128;
    bool seen = false;
    for (unsigned j = 0; j < count; ++j)
      seen |= list[1 + j] == segment;
    if (!seen)
      list[1 + count++] = segment;
  }
  list[0] = count;
}

// Prefix and selected segments use the original N stride. Segment grid is
// bounded by extras, and its inactive blocks read neither weights nor input.
template<int Bits>
__global__ void HeadGemv(ex::DeviceLinear l, const std::uint16_t* input,
                         float* output, unsigned prefix_segments,
                         const unsigned* selected) {
  const unsigned index = blockIdx.y;
  if (index >= prefix_segments && index - prefix_segments >= selected[0])
    return;
  const unsigned segment =
      index < prefix_segments ? index : selected[1 + index - prefix_segments];
  const unsigned lane = threadIdx.x;
  const unsigned r0 = (lane % 4) * 2;
  const unsigned nt = segment * 8 + blockIdx.x;
  const auto* trellis = reinterpret_cast<const std::uint32_t*>(l.trellis);
  const auto* x = reinterpret_cast<const _Float16*>(input);
  const auto lw = arithmetic::MakeLaneWindow<Bits>(lane);
  float a = 0, b = 0;
  for (unsigned kt = 0; kt < l.k / 16; ++kt) {
    Half2 w[4];
    arithmetic::DecodeLaneMul1<Bits>(
        trellis +
            (std::size_t{kt} * (l.n / 16) + nt) * arithmetic::TileDwords(Bits),
        lw, w);
    const auto offset = arithmetic::LaneMajorOffset(kt * 16 + r0, 0, 1);
    const auto* group = x + offset;
    const Half2 v01{group[0], group[1]}, v89{group[2], group[3]};
    a = __builtin_amdgcn_fdot2(w[0], v01, a, false);
    a = __builtin_amdgcn_fdot2(w[1], v89, a, false);
    b = __builtin_amdgcn_fdot2(w[2], v01, b, false);
    b = __builtin_amdgcn_fdot2(w[3], v89, b, false);
  }
  a += __shfl_xor(a, 1, 32);
  b += __shfl_xor(b, 1, 32);
  a += __shfl_xor(a, 2, 32);
  b += __shfl_xor(b, 2, 32);
  if (lane % 4 == 0) {
    const auto offset = std::size_t{index} * 128 + blockIdx.x * 16 + lane / 4;
    output[offset] = a;
    output[offset + 8] = b;
  }
}

__global__ void RestoreHead(ex::DeviceLinear l, float* values, float* logits,
                            unsigned prefix, const unsigned* selected) {
  const unsigned index = blockIdx.x;
  const unsigned prefix_segments = prefix / 128;
  if (index >= prefix_segments && index - prefix_segments >= selected[0])
    return;
  const unsigned segment =
      index < prefix_segments ? index : selected[1 + index - prefix_segments];
  const unsigned lane = threadIdx.x;
  const auto* svh = reinterpret_cast<const _Float16*>(l.svh);
  float v[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j)
    v[j] = values[std::size_t{index} * 128 + lane * 4 + j];
  arithmetic::Butterfly128(v, lane);
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const unsigned col = lane * 4 + j;
    const float y = (v[j] * arithmetic::kHadamardScale) *
                    static_cast<float>(svh[segment * 128 + col]);
    values[std::size_t{index} * 128 + col] = y;
    if (index < prefix_segments)
      logits[index * 128 + col] = y;
  }
}

__global__ void GatherHead(const float* values, float* logits, unsigned prefix,
                           const std::int32_t* ids, unsigned extras, unsigned n,
                           const unsigned* selected) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= extras)
    return;
  const int id = ids[i];
  float value = -INFINITY;
  if (id >= 0 && static_cast<unsigned>(id) < n) {
    if (static_cast<unsigned>(id) < prefix)
      value = logits[id];
    else
      for (unsigned j = 0; j < selected[0]; ++j)
        if (selected[1 + j] == static_cast<unsigned>(id) / 128)
          value = values[(std::size_t{prefix / 128} + j) * 128 + id % 128];
  }
  logits[prefix + i] = value;
}

}  // namespace

void Exl3Projection(const DeviceTensor& w, const float* x, float* out,
                    std::uint32_t rows, const Exl3Scratch& scratch,
                    hipStream_t stream, bool fast) {
  if (w.exl3 == nullptr) {
    WeightType type;
    switch (w.type) {
      case core::GgmlType::kF32:
        type = WeightType::kF32;
        break;
      case core::GgmlType::kF16:
        type = WeightType::kF16;
        break;
      case core::GgmlType::kBF16:
        type = WeightType::kBF16;
        break;
      default:
        throw std::invalid_argument("unsupported EXL3 plain operand");
    }
    SmallGemmStable(w.data, type, x, out, rows, w.rows, w.cols, stream);
    Check();
    return;
  }
  const auto& l = *w.exl3;
  const bool slice = w.output_begin != 0 || w.rows != l.n;
  if (slice && (scratch.slice == nullptr ||
                scratch.slice_elements < std::size_t{rows} * l.n))
    throw std::runtime_error("EXL3 output view exceeds slice scratch");
  auto* target = slice ? scratch.slice : out;
  const auto& s = scratch.projection;
  const gufo::hip::Exl3FastWorkspace ws{
      s.fast_input,         s.fast_input_elements, s.fast_counters,
      s.fast_counter_count, s.fast_down_input,     s.fast_down_input_elements};
  const gufo::hip::Exl3FastTarget t{&l, target, l.n};
  if (fast && gufo::hip::Exl3FastGroupSupported({&t, 1}, ws)) {
    for (std::uint32_t first = 0; first < rows;
         first += gufo::hip::kExl3FastMaxRows) {
      const auto count =
          std::min<std::uint32_t>(gufo::hip::kExl3FastMaxRows, rows - first);
      const gufo::hip::Exl3FastTarget pass{
          &l, target + std::size_t{first} * l.n, l.n};
      gufo::hip::LaunchExl3FastProjections(
          {&pass, 1}, x + std::size_t{first} * l.k, l.k, count, ws, stream);
    }
  } else {
    ex::LaunchReferenceProjection(l, x, l.k, rows, target, l.n, s, stream);
  }
  if (slice)
    hipLaunchKernelGGL(
        CopyColumns, dim3((std::size_t{rows} * w.rows + 255) / 256), dim3(256),
        0, stream, target, out, rows, l.n, w.output_begin, w.rows);
  Check();
}

void Exl3Gather(const float* x, const std::int32_t* rows, float* out,
                std::uint32_t count, std::uint32_t width, hipStream_t stream) {
  hipLaunchKernelGGL((MoveRows<false>),
                     dim3((std::size_t{count} * width + 255) / 256), dim3(256),
                     0, stream, x, rows, out, count, width);
  Check();
}
void Exl3Scatter(const float* x, const std::int32_t* slots, float* out,
                 std::uint32_t count, std::uint32_t width, hipStream_t stream) {
  hipLaunchKernelGGL((MoveRows<true>),
                     dim3((std::size_t{count} * width + 255) / 256), dim3(256),
                     0, stream, x, slots, out, count, width);
  Check();
}

namespace {
template<class F>
void WithRate(ex::BitRate rate, F&& f) {
  if (rate.half) {
    if (rate.bits != 3)
      throw std::invalid_argument("Flash-Next experts need K3..K8 or K3.5");
    f.template operator()<arithmetic::RateCode(3, true)>();
    return;
  }
  switch (rate.bits) {
    case 3:
      f.template operator()<3>();
      break;
    case 4:
      f.template operator()<4>();
      break;
    case 5:
      f.template operator()<5>();
      break;
    case 6:
      f.template operator()<6>();
      break;
    case 7:
      f.template operator()<7>();
      break;
    case 8:
      f.template operator()<8>();
      break;
    default:
      throw std::invalid_argument("Flash-Next experts need K3..K8 or K3.5");
  }
}
ex::BitRate BankRate(const DeviceTensor& bank) {
  if (bank.exl3 == nullptr || bank.experts == 0)
    throw std::invalid_argument("missing Flash-Next expert host descriptors");
  return bank.exl3[0].rate;
}

// The tile builders use the same cap (Exl3RouteTileRows).
unsigned TileRows(unsigned rows) {
  return Exl3RouteTileRows(rows);
}

template<bool FullPrecision, bool Scaled = false>
void LaunchRouted(const ex::DeviceLinear* gate, const ex::DeviceLinear* up,
                  const std::uint16_t* input, float* gate_y, float* up_y,
                  unsigned k, unsigned n, unsigned assignments,
                  unsigned projections, unsigned tile_rows,
                  const Exl3Scratch& scratch, hipStream_t stream,
                  ex::BitRate rate) {
  WithRate(rate, [&]<int Bits>() {
    const auto launch = [&]<unsigned R>() {
      hipLaunchKernelGGL((RoutedGemv<Bits, FullPrecision, R, Scaled>),
                         dim3(n / 16, assignments, projections), dim3(32), 0,
                         stream, gate, up, scratch.tiles, scratch.tile_count,
                         input, gate_y, up_y, k, n, assignments,
                         scratch.down_exponents);
    };
    switch (tile_rows) {
      case 1:
        launch.template operator()<1>();
        break;
      case 2:
        launch.template operator()<2>();
        break;
      case 3:
        launch.template operator()<3>();
        break;
      case 4:
        launch.template operator()<4>();
        break;
      default:
        launch.template operator()<8>();
        break;
    }
  });
}
}  // namespace

namespace {
// prescaled: RestoreActivateScale already scaled the input and wrote the
// exponents (the ScaleDownRows pass is folded into it).
void RoutedDown(const DeviceTensor& down, std::uint16_t* input, float* out,
                std::uint32_t assignments, std::uint32_t max_tile_rows,
                const Exl3Scratch& scratch, hipStream_t stream,
                bool prescaled) {
  if (assignments == 0)
    return;
  if (max_tile_rows == 0 || max_tile_rows > 8)
    throw std::invalid_argument("EXL3 down tile rows must be 1..8");
  if (scratch.down_exponents != nullptr) {
    if (!prescaled)
      hipLaunchKernelGGL(ScaleDownRows, dim3(assignments), dim3(256), 0, stream,
                         input, scratch.down_exponents, down.cols);
    LaunchRouted<false, true>(down.device_exl3, down.device_exl3, input, out,
                              out, down.cols, down.rows, assignments, 1,
                              TileRows(max_tile_rows), scratch, stream,
                              BankRate(down));
  } else {
    LaunchRouted<true>(down.device_exl3, down.device_exl3, input, out, out,
                       down.cols, down.rows, assignments, 1,
                       TileRows(max_tile_rows), scratch, stream,
                       BankRate(down));
  }
  Check();
}
}  // namespace

void Exl3RoutedDown(const DeviceTensor& down, std::uint16_t* input, float* out,
                    std::uint32_t assignments, std::uint32_t max_tile_rows,
                    const Exl3Scratch& scratch, hipStream_t stream) {
  RoutedDown(down, input, out, assignments, max_tile_rows, scratch, stream,
             false);
}

void Exl3BuildExpertBuckets(const std::int32_t* ids, std::uint32_t assignments,
                            std::uint32_t experts, std::uint32_t tile_rows,
                            std::uint32_t* offsets, std::int32_t* slots,
                            Exl3ExpertBucket* buckets,
                            std::uint32_t* bucket_count,
                            std::uint32_t max_buckets, hipStream_t stream,
                            std::int32_t* inverse) {
  if (experts == 0 || (tile_rows != 16 && tile_rows != 32 && tile_rows != 64))
    throw std::invalid_argument("EXL3 buckets need experts and 16/32/64 rows");
  const auto bound = std::min<std::uint64_t>(
      assignments, std::uint64_t{experts} + assignments / tile_rows);
  if (max_buckets < bound)
    throw std::invalid_argument("EXL3 bucket descriptor capacity too small");
  hipLaunchKernelGGL(CountBuckets, dim3(experts), dim3(256), 0, stream, ids,
                     assignments, offsets);
  hipLaunchKernelGGL(DescribeBuckets, dim3(1), dim3(1024), 0, stream, offsets,
                     experts, tile_rows, buckets, bucket_count, max_buckets);
  hipLaunchKernelGGL(FillBucketSlots, dim3(experts), dim3(256), 0, stream, ids,
                     assignments, offsets, slots, inverse);
  Check();
}

void Exl3RotateExpertBuckets(const ex::DeviceLinear* bank, const float* input,
                             std::uint16_t* rotated,
                             const std::uint32_t* offsets,
                             const std::int32_t* slots, std::uint32_t experts,
                             std::uint32_t assignments, std::uint32_t width,
                             std::uint32_t used, bool down,
                             hipStream_t stream) {
  if (assignments == 0)
    return;
  if (experts == 0 || used == 0 || width == 0 || width % 128 != 0)
    throw std::invalid_argument("EXL3 bucket rotation has invalid dimensions");
  hipLaunchKernelGGL(RotateBuckets, dim3(width / 128, assignments), dim3(32), 0,
                     stream, bank, input, rotated, offsets, slots, experts,
                     width, used, down);
  Check();
}

void Exl3ScatterExpertBuckets(const float* packed, const std::int32_t* slots,
                              float* out, std::uint32_t assignments,
                              std::uint32_t width, hipStream_t stream) {
  if (assignments != 0)
    Exl3Scatter(packed, slots, out, assignments, width, stream);
}

namespace {
void PrefillSnapshot(void* destination, const void* source, std::size_t bytes,
                     hipStream_t stream) {
  if (destination != nullptr &&
      hipMemcpyAsync(destination, source, bytes, hipMemcpyDeviceToDevice,
                     stream) != hipSuccess)
    throw std::runtime_error("EXL3 prefill trace copy failed");
}
}  // namespace

void Exl3PrefillBucketInner(const ex::DeviceLinear* bank,
                            const Exl3ExpertBucket* buckets,
                            const std::uint32_t* bucket_count,
                            std::uint32_t max_buckets, std::uint32_t tile_rows,
                            const std::uint16_t* input, float* inner,
                            std::uint32_t k, std::uint32_t n, ex::BitRate rate,
                            hipStream_t stream) {
  if (max_buckets > 65535 || k == 0 || n == 0 || k % 128 || n % 128 ||
      bank == nullptr || buckets == nullptr || bucket_count == nullptr ||
      input == nullptr || inner == nullptr ||
      (tile_rows != 16 && tile_rows != 32 && tile_rows != 64))
    throw std::invalid_argument("invalid Flash-Next bucket shape");
  // The shared X3 bucket WMMA takes every integer rate (K3..K8 here).
  WithRate(rate, [&]<int Bits>() {
    gufo::hip::LaunchExl3PrefillBucketInner(bank, buckets, bucket_count,
                                            max_buckets, tile_rows, input,
                                            inner, k, n, rate, stream);
  });
  Check();
}

void Exl3PrefillExperts(const DeviceLayer& layer, const float* x,
                        const std::int32_t* ids, float* out, std::uint32_t rows,
                        std::uint32_t used, std::uint32_t tile_rows,
                        const Exl3PrefillScratch& scratch, hipStream_t stream,
                        const Exl3PrefillTrace* trace) {
  Exl3PrefillExperts(layer, x, ids, out, rows, used, tile_rows, scratch,
                     Exl3PrefillBucketInner, stream, trace);
}

void Exl3PrefillDown(const DeviceTensor& down, std::uint32_t assignments,
                     std::uint32_t tile_rows, const Exl3PrefillScratch& scratch,
                     Exl3BucketGemm gemm, hipStream_t stream,
                     const Exl3PrefillTrace* trace, float* scatter_out,
                     bool raw_only) {
  if (assignments == 0)
    return;
  if (assignments > scratch.max_assignments || gemm == nullptr ||
      scratch.input == nullptr || scratch.out == nullptr ||
      scratch.down_exponents == nullptr || scratch.offsets == nullptr ||
      scratch.buckets == nullptr || scratch.bucket_count == nullptr ||
      (scatter_out != nullptr && scratch.slots == nullptr) ||
      (tile_rows != 16 && tile_rows != 32 && tile_rows != 64))
    throw std::invalid_argument(
        "EXL3 prefill down needs bucket scratch and adapter");
  if (trace != nullptr)
    PrefillSnapshot(
        trace->down_input, scratch.input,
        std::size_t{assignments} * down.cols * sizeof(std::uint16_t), stream);
  if (!raw_only)
    hipLaunchKernelGGL(ScaleDownRows, dim3(assignments), dim3(256), 0, stream,
                       scratch.input, scratch.down_exponents, down.cols);
  if (raw_only)
    // FP16 raw dots, unscaled by the row's exponent first (q6-gated).
    gufo::hip::LaunchExl3PrefillBucketInnerHalf(
        down.device_exl3, scratch.buckets, scratch.bucket_count,
        scratch.max_buckets, tile_rows, scratch.input,
        reinterpret_cast<std::uint16_t*>(scratch.out), down.cols, down.rows,
        BankRate(down), scratch.down_exponents, stream);
  else
    gemm(down.device_exl3, scratch.buckets, scratch.bucket_count,
         scratch.max_buckets, tile_rows, scratch.input, scratch.out, down.cols,
         down.rows, BankRate(down), stream);
  auto* counts = trace == nullptr ? nullptr : trace->fallback_counts;
  if (counts != nullptr &&
      hipMemsetAsync(counts, 0, 2 * sizeof(unsigned), stream) != hipSuccess)
    throw std::runtime_error("EXL3 fallback trace reset failed");
  WithRate(BankRate(down), [&]<int Bits>() {
    if (raw_only)
      hipLaunchKernelGGL((BucketDownFallback<Bits, _Float16>),
                         dim3(std::min(assignments, 256U)), dim3(256), 0,
                         stream, down.device_exl3, scratch.offsets,
                         down.experts, scratch.down_exponents, scratch.input,
                         reinterpret_cast<_Float16*>(scratch.out), down.cols,
                         down.rows, assignments, counts);
    else
      hipLaunchKernelGGL(
          (BucketDownFallback<Bits>), dim3(std::min(assignments, 256U)),
          dim3(256), 0, stream, down.device_exl3, scratch.offsets, down.experts,
          scratch.down_exponents, scratch.input, scratch.out, down.cols,
          down.rows, assignments, counts);
  });
  if (raw_only) {
    Check();
    return;
  }
  hipLaunchKernelGGL(RestoreDownBuckets, dim3(down.rows / 128, assignments),
                     dim3(32), 0, stream, down.device_exl3, scratch.offsets,
                     down.experts, scratch.down_exponents, scratch.out,
                     down.rows, scatter_out, scratch.slots);
  Check();
}

namespace {
void PrefillExperts(const DeviceLayer& layer, const float* x,
                    const std::int32_t* ids, float* out, std::uint32_t rows,
                    std::uint32_t used, std::uint32_t tile_rows,
                    const Exl3PrefillScratch& scratch, Exl3BucketGemm gemm,
                    hipStream_t stream, const Exl3PrefillTrace* trace,
                    bool raw_only);
}  // namespace

void Exl3PrefillExperts(const DeviceLayer& layer, const float* x,
                        const std::int32_t* ids, float* out, std::uint32_t rows,
                        std::uint32_t used, std::uint32_t tile_rows,
                        const Exl3PrefillScratch& scratch, Exl3BucketGemm gemm,
                        hipStream_t stream, const Exl3PrefillTrace* trace) {
  PrefillExperts(layer, x, ids, out, rows, used, tile_rows, scratch, gemm,
                 stream, trace, false);
}

void Exl3PrefillExpertsRaw(const DeviceLayer& layer, const float* x,
                           const std::int32_t* ids, std::uint32_t rows,
                           std::uint32_t used, std::uint32_t tile_rows,
                           const Exl3PrefillScratch& scratch,
                           hipStream_t stream) {
  if (scratch.inverse == nullptr)
    throw std::invalid_argument("EXL3 raw prefill needs the inverse slot map");
  PrefillExperts(layer, x, ids, nullptr, rows, used, tile_rows, scratch,
                 Exl3PrefillBucketInner, stream, nullptr, true);
}

namespace {
void PrefillExperts(const DeviceLayer& layer, const float* x,
                    const std::int32_t* ids, float* out, std::uint32_t rows,
                    std::uint32_t used, std::uint32_t tile_rows,
                    const Exl3PrefillScratch& scratch, Exl3BucketGemm gemm,
                    hipStream_t stream, const Exl3PrefillTrace* trace,
                    bool raw_only) {
  const std::uint64_t assignments = std::uint64_t{rows} * used;
  if (assignments == 0)
    return;
  if (assignments > scratch.max_assignments || gemm == nullptr ||
      scratch.offsets == nullptr || scratch.slots == nullptr ||
      scratch.buckets == nullptr || scratch.bucket_count == nullptr ||
      scratch.input == nullptr || scratch.gate == nullptr ||
      scratch.up == nullptr || scratch.out == nullptr ||
      scratch.down_exponents == nullptr)
    throw std::invalid_argument(
        "EXL3 prefill needs bucket scratch and X3 adapter");
  const auto& g = layer.ffn_gate_exps;
  const auto& u = layer.ffn_up_exps;
  const auto& d = layer.ffn_down_exps;
  Exl3BuildExpertBuckets(ids, assignments, g.experts, tile_rows,
                         scratch.offsets, scratch.slots, scratch.buckets,
                         scratch.bucket_count, scratch.max_buckets, stream,
                         scratch.inverse);
  const auto projection = [&](const DeviceTensor& bank, float* target) {
    gemm(bank.device_exl3, scratch.buckets, scratch.bucket_count,
         scratch.max_buckets, tile_rows, scratch.input, target, bank.cols,
         bank.rows, BankRate(bank), stream);
  };
  // One gathered read of x feeds both rotations. The up rows go to
  // scratch.out, free until the down projection writes it (it holds
  // assignments x hidden floats, twice the FP16 slab).
  if (g.cols != u.cols || u.cols > d.rows * 2)
    throw std::invalid_argument("EXL3 prefill gate/up widths differ");
  auto* up_input = reinterpret_cast<std::uint16_t*>(scratch.out);
  if (raw_only && used > 32)
    throw std::invalid_argument("EXL3 raw prefill takes at most 32 experts");
  if (raw_only)
    hipLaunchKernelGGL(
        RotateGateUpTokens, dim3((g.cols / 128 + 7) / 8, assignments / used),
        dim3(256), 0, stream, g.device_exl3, u.device_exl3, x, scratch.input,
        up_input, ids, scratch.inverse, g.cols, used);
  else
    hipLaunchKernelGGL(RotateGateUpBuckets, dim3(g.cols / 128, assignments),
                       dim3(32), 0, stream, g.device_exl3, u.device_exl3, x,
                       scratch.input, up_input, scratch.offsets, scratch.slots,
                       g.experts, g.cols, used);
  if (trace != nullptr)
    PrefillSnapshot(trace->gate_up_input, scratch.input,
                    std::size_t{assignments} * g.cols * sizeof(std::uint16_t),
                    stream);
  // Raw path: the gate/up dots in FP16 (half the bytes; q6-gated).
  const auto half_projection = [&](const DeviceTensor& bank,
                                   const std::uint16_t* input, float* target) {
    gufo::hip::LaunchExl3PrefillBucketInnerHalf(
        bank.device_exl3, scratch.buckets, scratch.bucket_count,
        scratch.max_buckets, tile_rows, input,
        reinterpret_cast<std::uint16_t*>(target), bank.cols, bank.rows,
        BankRate(bank), nullptr, stream);
  };
  if (raw_only)
    half_projection(g, scratch.input, scratch.gate);
  else
    projection(g, scratch.gate);
  if (trace != nullptr)
    PrefillSnapshot(trace->gate_inner, scratch.gate,
                    std::size_t{assignments} * g.rows * sizeof(float), stream);
  if (trace != nullptr && trace->gate_up_input != nullptr)
    PrefillSnapshot(trace->gate_up_input + assignments * g.cols, up_input,
                    std::size_t{assignments} * u.cols * sizeof(std::uint16_t),
                    stream);
  if (raw_only)
    half_projection(u, up_input, scratch.up);
  else
    gemm(u.device_exl3, scratch.buckets, scratch.bucket_count,
         scratch.max_buckets, tile_rows, up_input, scratch.up, u.cols, u.rows,
         BankRate(u), stream);
  if (trace != nullptr)
    PrefillSnapshot(trace->up_inner, scratch.up,
                    std::size_t{assignments} * u.rows * sizeof(float), stream);
  if (raw_only) {
    if (d.cols % 128 != 0 || d.cols > 4096)
      throw std::invalid_argument("EXL3 prefill down width unsupported");
    // Activation, down-input rotation and the down rows' power-of-two
    // scaling in one pass (Exl3PrefillDown skips ScaleDownRows for it).
    hipLaunchKernelGGL(RestoreActivateScaleBuckets, dim3(assignments),
                       dim3(32 * (d.cols / 128)), 0, stream, g.device_exl3,
                       u.device_exl3, d.device_exl3, scratch.offsets, d.experts,
                       reinterpret_cast<const _Float16*>(scratch.gate),
                       reinterpret_cast<const _Float16*>(scratch.up),
                       scratch.input, scratch.down_exponents, d.cols);
  } else
    hipLaunchKernelGGL(
        RestoreActivateRotateBuckets<float>, dim3(d.cols / 128, assignments),
        dim3(32), 0, stream, g.device_exl3, u.device_exl3, d.device_exl3,
        scratch.offsets, d.experts, scratch.gate, scratch.up, scratch.input,
        d.cols, trace != nullptr && trace->activation != nullptr);
  if (trace != nullptr)
    PrefillSnapshot(trace->activation, scratch.gate,
                    std::size_t{assignments} * g.rows * sizeof(float), stream);
  // The restore writes each row to its slot: no separate scatter pass. Raw:
  // scratch.out keeps the packed raw dots for Exl3PrefillCombine.
  Exl3PrefillDown(d, assignments, tile_rows, scratch, gemm, stream, trace,
                  raw_only ? nullptr : out, raw_only);
  Check();
}
}  // namespace

namespace {
// RestoreDownBuckets' arithmetic for each of a token's slots, then
// MoeEpilogueKernel's weighted sum, straight from the packed raw rows
// (Exl3PrefillMoeSegment, shared with the fused HC combine): neither the
// restored rows nor their slot-ordered copy reach memory. One wave per
// 128-wide segment; the fallback when the HC combine does not fuse it.
__global__ __launch_bounds__(256) void PrefillCombineKernel(
    Exl3PrefillMoeRows rows, float* out) {
  const unsigned t = blockIdx.y, lane = threadIdx.x % 32;
  const unsigned segment = blockIdx.x * 8 + threadIdx.x / 32;
  if (segment * 128 >= rows.width)
    return;
  float o[4];
  Exl3PrefillMoeSegment(rows, t, segment, lane, o);
  const auto first = std::size_t{t} * rows.width + segment * 128 + lane * 4;
#pragma unroll
  for (unsigned j = 0; j < 4; ++j)
    out[first + j] = o[j];
}
}  // namespace

void Exl3PrefillCombine(const DeviceLayer& layer,
                        const Exl3PrefillScratch& scratch,
                        const std::int32_t* ids, std::uint32_t rows,
                        std::uint32_t used, const float* weights,
                        const float* shared, const float* gate,
                        std::uint32_t gate_stride, float* out,
                        hipStream_t stream) {
  const auto& d = layer.ffn_down_exps;
  if (rows == 0)
    return;
  if (ids == nullptr || used == 0 || used > 16 || d.rows % 128 != 0 ||
      scratch.inverse == nullptr || scratch.out == nullptr)
    throw std::invalid_argument("EXL3 prefill combine: unsupported call");
  const Exl3PrefillMoeRows moe{
      d.device_exl3, ids,  scratch.inverse, scratch.out, weights,
      shared,        gate, gate_stride,     used,        d.rows};
  hipLaunchKernelGGL(PrefillCombineKernel, dim3((d.rows / 128 + 7) / 8, rows),
                     dim3(256), 0, stream, moe, out);
  Check();
}

bool Exl3SharedMlp(const DeviceLayer& l, const float* x, float* gate, float* up,
                   float* out, std::uint32_t rows, const Exl3Scratch& scratch,
                   hipStream_t stream) {
  if (rows == 0)
    return false;
  const auto& p = scratch.projection;
  const gufo::hip::Exl3FastWorkspace ws{
      p.fast_input,         p.fast_input_elements, p.fast_counters,
      p.fast_counter_count, p.fast_down_input,     p.fast_down_input_elements};
  const gufo::hip::Exl3FastMlp mlp{
      l.shexp_gate.exl3, l.shexp_up.exl3, l.shexp_down.exl3, gate, up, out,
      l.shexp_down.rows};
  if (!gufo::hip::Exl3FastMlpSupported(mlp, ws))
    return false;
  for (std::uint32_t first = 0; first < rows;
       first += gufo::hip::kExl3FastMaxRows) {
    auto pass = mlp;
    pass.out += std::size_t{first} * mlp.out_stride;
    const auto count =
        std::min<std::uint32_t>(gufo::hip::kExl3FastMaxRows, rows - first);
    gufo::hip::LaunchExl3FastMlp(pass,
                                 x + std::size_t{first} * l.shexp_gate.cols,
                                 l.shexp_gate.cols, count, ws, stream);
  }
  return true;
}

void Exl3RoutedMlp(const DeviceLayer& l, const float* x,
                   const std::int32_t* ids, float* gate, float* up, float* out,
                   std::uint32_t rows, std::uint32_t used, const Exl3Scratch& s,
                   hipStream_t stream, const Exl3RoutedTrace* trace,
                   bool tiles_ready, bool raw_down) {
  const unsigned assignments = rows * used;
  if (assignments == 0)
    return;
  const unsigned tile_rows = TileRows(rows);
  const auto& g = l.ffn_gate_exps;
  const auto& u = l.ffn_up_exps;
  const auto& d = l.ffn_down_exps;
  const auto snapshot = [&](void* destination, const void* source,
                            std::size_t bytes) {
    if (destination != nullptr &&
        hipMemcpyAsync(destination, source, bytes, hipMemcpyDeviceToDevice,
                       stream) != hipSuccess)
      throw std::runtime_error("EXL3 routed trace copy failed");
  };
  if (!tiles_ready) {
    if (hipMemsetAsync(s.tile_count, 0, sizeof(unsigned), stream) != hipSuccess)
      throw std::runtime_error("EXL3 route counter reset failed");
    hipLaunchKernelGGL(BuildTiles, dim3((g.experts + 63) / 64), dim3(64), 0,
                       stream, ids, assignments, s.tiles, s.tile_count,
                       g.experts, tile_rows);
  }
  hipLaunchKernelGGL(RotateAssignments, dim3(g.cols / 128, assignments, 2),
                     dim3(32), 0, stream, g.device_exl3, u.device_exl3, ids, x,
                     s.routed_input, used, g.cols, assignments);
  if (trace != nullptr)
    snapshot(trace->gate_up_input, s.routed_input,
             std::size_t{2} * assignments * g.cols * sizeof(std::uint16_t));
  if (BankRate(g) == BankRate(u)) {
    LaunchRouted<false>(g.device_exl3, u.device_exl3, s.routed_input, gate, up,
                        g.cols, g.rows, assignments, 2, tile_rows, s, stream,
                        BankRate(g));
  } else {
    LaunchRouted<false>(g.device_exl3, g.device_exl3, s.routed_input, gate,
                        gate, g.cols, g.rows, assignments, 1, tile_rows, s,
                        stream, BankRate(g));
    LaunchRouted<false>(u.device_exl3, u.device_exl3,
                        s.routed_input + std::size_t{assignments} * g.cols, up,
                        up, u.cols, u.rows, assignments, 1, tile_rows, s,
                        stream, BankRate(u));
  }
  if (trace != nullptr) {
    snapshot(trace->gate_inner, gate,
             std::size_t{assignments} * g.rows * sizeof(float));
    snapshot(trace->up_inner, up,
             std::size_t{assignments} * g.rows * sizeof(float));
  }
  // Without a trace the down rows' power-of-two scaling joins the restore
  // (RestoreActivateScale); RoutedDown then skips ScaleDownRows.
  const bool prescaled = trace == nullptr && s.down_exponents != nullptr &&
                         g.rows == d.cols && d.cols % 128 == 0 &&
                         d.cols <= 4096;
  if (prescaled)
    hipLaunchKernelGGL(RestoreActivateScale, dim3(assignments),
                       dim3(32 * (d.cols / 128)), 0, stream, g.device_exl3,
                       u.device_exl3, d.device_exl3, ids, gate, up,
                       s.routed_input, s.down_exponents, d.cols);
  else
    hipLaunchKernelGGL(RestoreActivateRotate, dim3(g.rows / 128, assignments),
                       dim3(32), 0, stream, g.device_exl3, u.device_exl3,
                       d.device_exl3, ids, gate, up, s.routed_input, g.rows);
  if (trace != nullptr)
    snapshot(trace->activation, gate,
             std::size_t{assignments} * g.rows * sizeof(float));
  if (trace != nullptr)
    snapshot(trace->down_input, s.routed_input,
             std::size_t{assignments} * d.cols * sizeof(std::uint16_t));
  if (trace != nullptr && trace->down_packed_inner != nullptr) {
    LaunchRouted<false>(d.device_exl3, d.device_exl3, s.routed_input,
                        trace->down_packed_inner, trace->down_packed_inner,
                        d.cols, d.rows, assignments, 1, tile_rows, s, stream,
                        BankRate(d));
  }
  if (trace != nullptr && trace->down_widened_inner != nullptr)
    LaunchRouted<true>(d.device_exl3, d.device_exl3, s.routed_input,
                       trace->down_widened_inner, trace->down_widened_inner,
                       d.cols, d.rows, assignments, 1, tile_rows, s, stream,
                       BankRate(d));
  RoutedDown(d, s.routed_input, out, assignments, tile_rows, s, stream,
             prescaled);
  if (trace != nullptr)
    snapshot(trace->down_inner, out,
             std::size_t{assignments} * d.rows * sizeof(float));
  if (!raw_down)
    hipLaunchKernelGGL(RestoreDownAssignments, dim3(d.rows / 128, assignments),
                       dim3(32), 0, stream, d.device_exl3, ids, out, d.rows);
  Check();
}

namespace {
// RestoreDownAssignments for each of a token's slots, then MoeEpilogue's
// weighted sum (Exl3MoeSegment over FP32 rows in assignment order): the
// restored rows never reach memory. One wave per 128-wide segment.
__global__ __launch_bounds__(256) void RoutedCombineKernel(
    Exl3PrefillMoeRows rows, float* out) {
  const unsigned t = blockIdx.y, lane = threadIdx.x % 32;
  const unsigned segment = blockIdx.x * 8 + threadIdx.x / 32;
  if (segment * 128 >= rows.width)
    return;
  float o[4];
  Exl3MoeSegment<float, false>(rows, t, segment, lane, o);
  const auto first = std::size_t{t} * rows.width + segment * 128 + lane * 4;
#pragma unroll
  for (unsigned j = 0; j < 4; ++j)
    out[first + j] = o[j];
}
}  // namespace

void Exl3RoutedCombine(const DeviceLayer& layer, const std::int32_t* ids,
                       std::uint32_t rows, std::uint32_t used, const float* raw,
                       const float* weights, const float* shared,
                       const float* gate, std::uint32_t gate_stride, float* out,
                       hipStream_t stream) {
  const auto& d = layer.ffn_down_exps;
  if (rows == 0)
    return;
  if (ids == nullptr || raw == nullptr || used == 0 || used > 16 ||
      d.rows % 128 != 0)
    throw std::invalid_argument("EXL3 routed combine: unsupported call");
  const Exl3PrefillMoeRows moe{d.device_exl3, ids,  nullptr,     raw,  weights,
                               shared,        gate, gate_stride, used, d.rows};
  hipLaunchKernelGGL(RoutedCombineKernel, dim3((d.rows / 128 + 7) / 8, rows),
                     dim3(256), 0, stream, moe, out);
  Check();
}

void Exl3DraftHead(const DeviceTensor& head, const float* x, float* logits,
                   std::uint32_t prefix, const std::int32_t* ids,
                   std::uint32_t extras, const Exl3Scratch& s,
                   hipStream_t stream) {
  const auto& l = *head.exl3;
  if (prefix % 128 != 0 || prefix > l.n || (l.rate.half && l.rate.bits != 3) ||
      l.rate.bits < 3 || l.rate.bits > 8 || l.codebook != ex::Codebook::kMul1)
    throw std::runtime_error(
        "EXL3 draft head needs a complete K3..K8 or K3.5 prefix");
  if (prefix == 0 && extras == 0)
    return;
  hipLaunchKernelGGL(SelectSegments, dim3(1), dim3(1), 0, stream, ids, extras,
                     prefix, l.n, s.head_segments);
  hipLaunchKernelGGL(HeadInput, dim3(l.k / 128), dim3(32), 0, stream, l, x,
                     s.projection.fast_input);
  WithRate(l.rate, [&]<int Bits>() {
    hipLaunchKernelGGL((HeadGemv<Bits>), dim3(8, prefix / 128 + extras),
                       dim3(32), 0, stream, l, s.projection.fast_input,
                       s.head_values, prefix / 128, s.head_segments);
  });
  hipLaunchKernelGGL(RestoreHead, dim3(prefix / 128 + extras), dim3(32), 0,
                     stream, l, s.head_values, logits, prefix, s.head_segments);
  if (extras != 0)
    hipLaunchKernelGGL(GatherHead, dim3((extras + 255) / 256), dim3(256), 0,
                       stream, s.head_values, logits, prefix, ids, extras, l.n,
                       s.head_segments);
  Check();
}

}  // namespace gufo::models::qwen38_flash_next::rocm
