#include <algorithm>
#include <stdexcept>

#include "src/core/hip/detail/exl3_gemv.hpp"
#include "src/models/qwen/hip/kernels/exl3_gemv.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/exl3_moe.hpp"

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
template<bool FullPrecision, unsigned MaxRows = 8, bool Scaled = false>
__global__ void RoutedGemv(const ex::DeviceLinear* gate,
                           const ex::DeviceLinear* up,
                           const Exl3RouteTile* tiles, const unsigned* count,
                           const std::uint16_t* input, float* gate_y,
                           float* up_y, unsigned k, unsigned n,
                           unsigned assignments,
                           const std::int32_t* exponents) {
  if (blockIdx.y >= *count)
    return;
  const auto tile_rows = tiles[blockIdx.y];
  const unsigned projection = blockIdx.z;
  const auto& l = (projection == 0 ? gate : up)[tile_rows.expert];
  const unsigned lane = threadIdx.x;
  const unsigned r0 = (lane % 4) * 2;
  const unsigned nt = blockIdx.x;
  float a[MaxRows]{}, b[MaxRows]{};
  const auto lw = arithmetic::MakeLaneWindow<5>(lane);
  const auto* trellis = reinterpret_cast<const std::uint32_t*>(l.trellis);
  const auto* x = reinterpret_cast<const _Float16*>(input) +
                  std::size_t{projection} * assignments * k;
  for (unsigned kt = 0; kt < k / 16; ++kt) {
    Half2 w[4];
    arithmetic::DecodeLaneMul1<5>(
        trellis + (std::size_t{kt} * (n / 16) + nt) * 40, lw, w);
    for (unsigned r = 0; r < MaxRows; ++r) {
      if (r >= tile_rows.count)
        break;
      const auto* row = x + std::size_t{tile_rows.slots[r]} * k + kt * 16 + r0;
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
  for (unsigned r = 0; r < MaxRows; ++r) {
    if (r >= tile_rows.count)
      break;
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

// One thread per expert counts without atomics. The prefix pass has no
// allocation or host synchronization and zeroes inactive descriptors, which
// lets a bounded WMMA launch consume it safely during graph replay.
__global__ void CountBuckets(const std::int32_t* ids, unsigned assignments,
                             unsigned experts, unsigned* offsets) {
  const unsigned expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= experts)
    return;
  unsigned count = 0;
  for (unsigned slot = 0; slot < assignments; ++slot)
    count += ids[slot] == static_cast<int>(expert);
  offsets[expert + 1] = count;
}

__global__ void DescribeBuckets(unsigned* offsets, unsigned experts,
                                unsigned tile_rows, Exl3ExpertBucket* buckets,
                                unsigned* bucket_count, unsigned capacity) {
  if (threadIdx.x != 0)
    return;
  unsigned first = 0, count = 0;
  offsets[0] = 0;
  for (unsigned expert = 0; expert < experts; ++expert) {
    const unsigned rows = offsets[expert + 1];
    offsets[expert + 1] = first + rows;
    for (unsigned row = 0; row < rows; row += tile_rows)
      buckets[count++] = {expert, min(tile_rows, rows - row), first + row};
    first += rows;
  }
  *bucket_count = count;
  for (unsigned i = count; i < capacity; ++i)
    buckets[i] = {};
}

__global__ void FillBucketSlots(const std::int32_t* ids, unsigned assignments,
                                unsigned experts, const unsigned* offsets,
                                std::int32_t* slots) {
  const unsigned expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= experts)
    return;
  unsigned next = offsets[expert];
  for (unsigned slot = 0; slot < assignments; ++slot)
    if (ids[slot] == static_cast<int>(expert))
      slots[next++] = slot;
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

__global__ void RestoreActivateRotateBuckets(
    const ex::DeviceLinear* gate, const ex::DeviceLinear* up,
    const ex::DeviceLinear* down, const unsigned* offsets, unsigned experts,
    float* gate_y, const float* up_y, std::uint16_t* output, unsigned width) {
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

// WMMA returns raw inner dots. Rows with no safe normal FP16 scaling are
// overwritten by the same widened FMA/reduction order as routed decode.
__global__ void BucketDownFallback(const ex::DeviceLinear* bank,
                                   const unsigned* offsets, unsigned experts,
                                   const std::int32_t* exponents,
                                   const std::uint16_t* input, float* output,
                                   unsigned k, unsigned n) {
  const unsigned row = blockIdx.y;
  if (row >= offsets[experts] || exponents[row] >= 0)
    return;
  const unsigned lo = BucketExpert(offsets, experts, row);
  const auto& l = bank[lo];
  const unsigned lane = threadIdx.x, r0 = (lane % 4) * 2;
  const unsigned nt = blockIdx.x;
  const auto* x =
      reinterpret_cast<const _Float16*>(input) + std::size_t{row} * k;
  const auto* trellis = reinterpret_cast<const std::uint32_t*>(l.trellis);
  const auto lw = arithmetic::MakeLaneWindow<5>(lane);
  float a = 0, b = 0;
  for (unsigned kt = 0; kt < k / 16; ++kt) {
    Half2 w[4];
    arithmetic::DecodeLaneMul1<5>(
        trellis + (std::size_t{kt} * (n / 16) + nt) * 40, lw, w);
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
    output[std::size_t{row} * n + nt * 16 + lane / 4] = a;
    output[std::size_t{row} * n + nt * 16 + lane / 4 + 8] = b;
  }
}

__global__ void RestoreDownBuckets(const ex::DeviceLinear* bank,
                                   const unsigned* offsets, unsigned experts,
                                   const std::int32_t* exponents, float* output,
                                   unsigned width) {
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
#pragma unroll
  for (unsigned j = 0; j < 4; ++j)
    output[offset + j] = (v[j] * arithmetic::kHadamardScale) *
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
  const auto lw = arithmetic::MakeLaneWindow<6>(lane);
  float a = 0, b = 0;
  for (unsigned kt = 0; kt < l.k / 16; ++kt) {
    Half2 w[4];
    arithmetic::DecodeLaneMul1<6>(
        trellis + (std::size_t{kt} * (l.n / 16) + nt) * 48, lw, w);
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
  if (fast && rows <= gufo::hip::kExl3FastMaxRows &&
      gufo::hip::Exl3FastGroupSupported({&t, 1}, ws))
    gufo::hip::LaunchExl3FastProjections({&t, 1}, x, l.k, rows, ws, stream);
  else
    ex::LaunchReferenceProjection(l, x, l.k, rows, target, l.n, s, stream);
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
// The tile builder uses the same cap. In particular R=3 does not carry
// eight independent accumulator pairs through the complete K loop.
unsigned TileRows(unsigned rows) {
  return rows <= 3 ? rows : rows <= 4 ? 4 : 8;
}

template<bool FullPrecision, bool Scaled = false>
void LaunchRouted(const ex::DeviceLinear* gate, const ex::DeviceLinear* up,
                  const std::uint16_t* input, float* gate_y, float* up_y,
                  unsigned k, unsigned n, unsigned assignments,
                  unsigned projections, unsigned tile_rows,
                  const Exl3Scratch& scratch, hipStream_t stream) {
  const auto launch = [&]<unsigned R>() {
    hipLaunchKernelGGL((RoutedGemv<FullPrecision, R, Scaled>),
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
}
}  // namespace

void Exl3RoutedDown(const DeviceTensor& down, std::uint16_t* input, float* out,
                    std::uint32_t assignments, std::uint32_t max_tile_rows,
                    const Exl3Scratch& scratch, hipStream_t stream) {
  if (assignments == 0)
    return;
  if (max_tile_rows == 0 || max_tile_rows > 8)
    throw std::invalid_argument("EXL3 down tile rows must be 1..8");
  if (scratch.down_exponents != nullptr) {
    hipLaunchKernelGGL(ScaleDownRows, dim3(assignments), dim3(256), 0, stream,
                       input, scratch.down_exponents, down.cols);
    LaunchRouted<false, true>(down.device_exl3, down.device_exl3, input, out,
                              out, down.cols, down.rows, assignments, 1,
                              TileRows(max_tile_rows), scratch, stream);
  } else {
    LaunchRouted<true>(down.device_exl3, down.device_exl3, input, out, out,
                       down.cols, down.rows, assignments, 1,
                       TileRows(max_tile_rows), scratch, stream);
  }
  Check();
}

void Exl3BuildExpertBuckets(const std::int32_t* ids, std::uint32_t assignments,
                            std::uint32_t experts, std::uint32_t tile_rows,
                            std::uint32_t* offsets, std::int32_t* slots,
                            Exl3ExpertBucket* buckets,
                            std::uint32_t* bucket_count,
                            std::uint32_t max_buckets, hipStream_t stream) {
  if (experts == 0 || (tile_rows != 16 && tile_rows != 32 && tile_rows != 64))
    throw std::invalid_argument("EXL3 buckets need experts and 16/32/64 rows");
  const auto bound = std::min<std::uint64_t>(
      assignments, std::uint64_t{experts} + assignments / tile_rows);
  if (max_buckets < bound)
    throw std::invalid_argument("EXL3 bucket descriptor capacity too small");
  hipLaunchKernelGGL(CountBuckets, dim3((experts + 63) / 64), dim3(64), 0,
                     stream, ids, assignments, experts, offsets);
  hipLaunchKernelGGL(DescribeBuckets, dim3(1), dim3(1), 0, stream, offsets,
                     experts, tile_rows, buckets, bucket_count, max_buckets);
  hipLaunchKernelGGL(FillBucketSlots, dim3((experts + 63) / 64), dim3(64), 0,
                     stream, ids, assignments, experts, offsets, slots);
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

void Exl3PrefillExperts(const DeviceLayer& layer, const float* x,
                        const std::int32_t* ids, float* out, std::uint32_t rows,
                        std::uint32_t used, std::uint32_t tile_rows,
                        const Exl3PrefillScratch& scratch, Exl3BucketGemm gemm,
                        hipStream_t stream) {
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
                         scratch.bucket_count, scratch.max_buckets, stream);
  const auto projection = [&](const DeviceTensor& bank, float* target) {
    gemm(bank.device_exl3, scratch.buckets, scratch.bucket_count,
         scratch.max_buckets, tile_rows, scratch.input, target, bank.cols,
         bank.rows, stream);
  };
  Exl3RotateExpertBuckets(g.device_exl3, x, scratch.input, scratch.offsets,
                          scratch.slots, g.experts, assignments, g.cols, used,
                          false, stream);
  projection(g, scratch.gate);
  Exl3RotateExpertBuckets(u.device_exl3, x, scratch.input, scratch.offsets,
                          scratch.slots, u.experts, assignments, u.cols, used,
                          false, stream);
  projection(u, scratch.up);
  hipLaunchKernelGGL(
      RestoreActivateRotateBuckets, dim3(d.cols / 128, assignments), dim3(32),
      0, stream, g.device_exl3, u.device_exl3, d.device_exl3, scratch.offsets,
      d.experts, scratch.gate, scratch.up, scratch.input, d.cols);
  hipLaunchKernelGGL(ScaleDownRows, dim3(assignments), dim3(256), 0, stream,
                     scratch.input, scratch.down_exponents, d.cols);
  projection(d, scratch.out);
  hipLaunchKernelGGL(BucketDownFallback, dim3(d.rows / 16, assignments),
                     dim3(32), 0, stream, d.device_exl3, scratch.offsets,
                     d.experts, scratch.down_exponents, scratch.input,
                     scratch.out, d.cols, d.rows);
  hipLaunchKernelGGL(RestoreDownBuckets, dim3(d.rows / 128, assignments),
                     dim3(32), 0, stream, d.device_exl3, scratch.offsets,
                     d.experts, scratch.down_exponents, scratch.out, d.rows);
  Exl3ScatterExpertBuckets(scratch.out, scratch.slots, out, assignments, d.rows,
                           stream);
  Check();
}

void Exl3RoutedMlp(const DeviceLayer& l, const float* x,
                   const std::int32_t* ids, float* gate, float* up, float* out,
                   std::uint32_t rows, std::uint32_t used, const Exl3Scratch& s,
                   hipStream_t stream, const Exl3RoutedTrace* trace) {
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
  if (hipMemsetAsync(s.tile_count, 0, sizeof(unsigned), stream) != hipSuccess)
    throw std::runtime_error("EXL3 route counter reset failed");
  hipLaunchKernelGGL(BuildTiles, dim3((g.experts + 63) / 64), dim3(64), 0,
                     stream, ids, assignments, s.tiles, s.tile_count, g.experts,
                     tile_rows);
  hipLaunchKernelGGL(RotateAssignments, dim3(g.cols / 128, assignments, 2),
                     dim3(32), 0, stream, g.device_exl3, u.device_exl3, ids, x,
                     s.routed_input, used, g.cols, assignments);
  if (trace != nullptr)
    snapshot(trace->gate_up_input, s.routed_input,
             std::size_t{2} * assignments * g.cols * sizeof(std::uint16_t));
  LaunchRouted<false>(g.device_exl3, u.device_exl3, s.routed_input, gate, up,
                      g.cols, g.rows, assignments, 2, tile_rows, s, stream);
  if (trace != nullptr) {
    snapshot(trace->gate_inner, gate,
             std::size_t{assignments} * g.rows * sizeof(float));
    snapshot(trace->up_inner, up,
             std::size_t{assignments} * g.rows * sizeof(float));
  }
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
                        d.cols, d.rows, assignments, 1, tile_rows, s, stream);
  }
  if (trace != nullptr && trace->down_widened_inner != nullptr)
    LaunchRouted<true>(d.device_exl3, d.device_exl3, s.routed_input,
                       trace->down_widened_inner, trace->down_widened_inner,
                       d.cols, d.rows, assignments, 1, tile_rows, s, stream);
  Exl3RoutedDown(d, s.routed_input, out, assignments, tile_rows, s, stream);
  if (trace != nullptr)
    snapshot(trace->down_inner, out,
             std::size_t{assignments} * d.rows * sizeof(float));
  hipLaunchKernelGGL(RestoreDownAssignments, dim3(d.rows / 128, assignments),
                     dim3(32), 0, stream, d.device_exl3, ids, out, d.rows);
  Check();
}

void Exl3DraftHead(const DeviceTensor& head, const float* x, float* logits,
                   std::uint32_t prefix, const std::int32_t* ids,
                   std::uint32_t extras, const Exl3Scratch& s,
                   hipStream_t stream) {
  const auto& l = *head.exl3;
  if (prefix % 128 != 0 || l.rate != ex::BitRate{6, false})
    throw std::runtime_error("EXL3 draft head needs a complete K6 prefix");
  hipLaunchKernelGGL(SelectSegments, dim3(1), dim3(1), 0, stream, ids, extras,
                     prefix, l.n, s.head_segments);
  hipLaunchKernelGGL(HeadInput, dim3(l.k / 128), dim3(32), 0, stream, l, x,
                     s.projection.fast_input);
  hipLaunchKernelGGL(HeadGemv, dim3(8, prefix / 128 + extras), dim3(32), 0,
                     stream, l, s.projection.fast_input, s.head_values,
                     prefix / 128, s.head_segments);
  hipLaunchKernelGGL(RestoreHead, dim3(prefix / 128 + extras), dim3(32), 0,
                     stream, l, s.head_values, logits, prefix, s.head_segments);
  hipLaunchKernelGGL(GatherHead, dim3((extras + 255) / 256), dim3(256), 0,
                     stream, s.head_values, logits, prefix, ids, extras, l.n,
                     s.head_segments);
  Check();
}

}  // namespace gufo::models::qwen38_flash_next::rocm
