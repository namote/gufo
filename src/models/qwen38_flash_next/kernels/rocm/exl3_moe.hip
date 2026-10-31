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
                           unsigned experts) {
  const unsigned expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= experts)
    return;
  Exl3RouteTile tile{};
  tile.expert = expert;
  for (unsigned i = 0; i < assignments; ++i) {
    if (ids[i] != static_cast<int>(expert))
      continue;
    tile.slots[tile.count++] = i;
    if (tile.count == 8) {
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
                                  unsigned width, unsigned assignments,
                                  bool down) {
  const unsigned assignment = blockIdx.y;
  const unsigned projection = blockIdx.z;
  const auto& l = (projection == 0 ? gate : up)[ids[assignment]];
  const unsigned lane = threadIdx.x;
  const unsigned first = blockIdx.x * 128 + lane * 4;
  const auto* suh = reinterpret_cast<const _Float16*>(l.suh);
  const auto row = down ? assignment : assignment / used;
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
template<bool FullPrecision>
__global__ void RoutedGemv(const ex::DeviceLinear* gate,
                           const ex::DeviceLinear* up,
                           const Exl3RouteTile* tiles, const unsigned* count,
                           const std::uint16_t* input, float* gate_y,
                           float* up_y, unsigned k, unsigned n,
                           unsigned assignments) {
  if (blockIdx.y >= *count)
    return;
  const auto tile_rows = tiles[blockIdx.y];
  const unsigned projection = blockIdx.z;
  const auto& l = (projection == 0 ? gate : up)[tile_rows.expert];
  const unsigned lane = threadIdx.x;
  const unsigned r0 = (lane % 4) * 2;
  const unsigned nt = blockIdx.x;
  float a[8]{}, b[8]{};
  const auto lw = arithmetic::MakeLaneWindow<5>(lane);
  const auto* trellis = reinterpret_cast<const std::uint32_t*>(l.trellis);
  const auto* x = reinterpret_cast<const _Float16*>(input) +
                  std::size_t{projection} * assignments * k;
  for (unsigned kt = 0; kt < k / 16; ++kt) {
    Half2 w[4];
    arithmetic::DecodeLaneMul1<5>(
        trellis + (std::size_t{kt} * (n / 16) + nt) * 40, lw, w);
    for (unsigned r = 0; r < tile_rows.count; ++r) {
      const auto* row = x + std::size_t{tile_rows.slots[r]} * k + kt * 16 + r0;
      const Half2 v01{row[0], row[1]};
      const Half2 v89{row[8], row[9]};
      if constexpr (FullPrecision) {
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
  for (unsigned r = 0; r < tile_rows.count; ++r) {
    a[r] += __shfl_xor(a[r], 1, 32);
    b[r] += __shfl_xor(b[r], 1, 32);
    a[r] += __shfl_xor(a[r], 2, 32);
    b[r] += __shfl_xor(b[r], 2, 32);
    if (lane % 4 == 0) {
      auto* y = (projection == 0 ? gate_y : up_y) +
                std::size_t{tile_rows.slots[r]} * n + nt * 16;
      y[lane / 4] = a[r];
      y[lane / 4 + 8] = b[r];
    }
  }
}

__global__ void RestoreAssignments(const ex::DeviceLinear* gate,
                                   const ex::DeviceLinear* up,
                                   const std::int32_t* ids, float* gate_y,
                                   float* up_y, unsigned width, bool swiglu) {
  const unsigned row = blockIdx.y;
  const unsigned lane = threadIdx.x;
  const unsigned first = blockIdx.x * 128 + lane * 4;
  const auto* gs = reinterpret_cast<const _Float16*>(gate[ids[row]].svh);
  const auto* us =
      swiglu ? reinterpret_cast<const _Float16*>(up[ids[row]].svh) : gs;
  const auto offset = std::size_t{row} * width + first;
  float g[4], u[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    g[j] = gate_y[offset + j];
    u[j] = swiglu ? up_y[offset + j] : 0.0F;
  }
  arithmetic::Butterfly128(g, lane);
  if (swiglu)
    arithmetic::Butterfly128(u, lane);
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const float value =
        (g[j] * arithmetic::kHadamardScale) * static_cast<float>(gs[first + j]);
    const float other =
        (u[j] * arithmetic::kHadamardScale) * static_cast<float>(us[first + j]);
    gate_y[offset + j] =
        swiglu ? (value / (1.0F + expf(-value))) * other : value;
  }
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

void Exl3RoutedMlp(const DeviceLayer& l, const float* x,
                   const std::int32_t* ids, float* gate, float* up, float* out,
                   std::uint32_t rows, std::uint32_t used, const Exl3Scratch& s,
                   hipStream_t stream, const Exl3RoutedTrace* trace) {
  const unsigned assignments = rows * used;
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
                     stream, ids, assignments, s.tiles, s.tile_count,
                     g.experts);
  hipLaunchKernelGGL(RotateAssignments, dim3(g.cols / 128, assignments, 2),
                     dim3(32), 0, stream, g.device_exl3, u.device_exl3, ids, x,
                     s.routed_input, used, g.cols, assignments, false);
  if (trace != nullptr)
    snapshot(trace->gate_up_input, s.routed_input,
             std::size_t{2} * assignments * g.cols * sizeof(std::uint16_t));
  hipLaunchKernelGGL((RoutedGemv<false>), dim3(g.rows / 16, assignments, 2),
                     dim3(32), 0, stream, g.device_exl3, u.device_exl3, s.tiles,
                     s.tile_count, s.routed_input, gate, up, g.cols, g.rows,
                     assignments);
  if (trace != nullptr) {
    snapshot(trace->gate_inner, gate,
             std::size_t{assignments} * g.rows * sizeof(float));
    snapshot(trace->up_inner, up,
             std::size_t{assignments} * g.rows * sizeof(float));
  }
  hipLaunchKernelGGL(RestoreAssignments, dim3(g.rows / 128, assignments),
                     dim3(32), 0, stream, g.device_exl3, u.device_exl3, ids,
                     gate, up, g.rows, true);
  if (trace != nullptr)
    snapshot(trace->activation, gate,
             std::size_t{assignments} * g.rows * sizeof(float));
  hipLaunchKernelGGL(RotateAssignments, dim3(d.cols / 128, assignments),
                     dim3(32), 0, stream, d.device_exl3, d.device_exl3, ids,
                     gate, s.routed_input, used, d.cols, assignments, true);
  if (trace != nullptr)
    snapshot(trace->down_input, s.routed_input,
             std::size_t{assignments} * d.cols * sizeof(std::uint16_t));
  if (trace != nullptr && trace->down_packed_inner != nullptr) {
    hipLaunchKernelGGL((RoutedGemv<false>), dim3(d.rows / 16, assignments),
                       dim3(32), 0, stream, d.device_exl3, d.device_exl3,
                       s.tiles, s.tile_count, s.routed_input,
                       trace->down_packed_inner, trace->down_packed_inner,
                       d.cols, d.rows, assignments);
  }
  hipLaunchKernelGGL((RoutedGemv<true>), dim3(d.rows / 16, assignments),
                     dim3(32), 0, stream, d.device_exl3, d.device_exl3, s.tiles,
                     s.tile_count, s.routed_input, out, out, d.cols, d.rows,
                     assignments);
  if (trace != nullptr)
    snapshot(trace->down_inner, out,
             std::size_t{assignments} * d.rows * sizeof(float));
  hipLaunchKernelGGL(RestoreAssignments, dim3(d.rows / 128, assignments),
                     dim3(32), 0, stream, d.device_exl3, d.device_exl3, ids,
                     out, out, d.rows, false);
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
