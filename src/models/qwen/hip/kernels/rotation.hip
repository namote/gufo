#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/hip/detail/hadamard.hpp"
#include "src/models/qwen/hip/ops/hadamard.hpp"

namespace gufo::hip {
namespace {

constexpr unsigned kRotateWaves = 8;

// One wave32 per 128-element segment, lane l holding elements 4l..4l+3.
__launch_bounds__(kRotateWaves * 32) __global__
    void RotateRowsFp32Kernel(float* __restrict__ x,
                              const std::uint32_t* __restrict__ signs,
                              std::size_t segments, std::size_t row_segments) {
  const std::size_t segment =
      static_cast<std::size_t>(blockIdx.x) * kRotateWaves + threadIdx.x / 32;
  if (segment >= segments) {
    return;
  }
  const unsigned lane = threadIdx.x % 32;
  float4* data = reinterpret_cast<float4*>(x + segment * hadamard::kSegment);
  const float4 in = data[lane];
  float v[4] = {in.x, in.y, in.z, in.w};
  hadamard::RotateSegmentLanes<4>(
      v, signs + (segment % row_segments) * hadamard::kSignWordsPerSegment,
      lane);
  data[lane] = make_float4(v[0], v[1], v[2], v[3]);
}

}  // namespace

void LaunchRotateRowsFp32(float* x, const std::uint32_t* signs,
                          std::size_t rows, std::size_t k, hipStream_t stream) {
  const std::size_t row_segments = k / hadamard::kSegment;
  const std::size_t segments = rows * row_segments;
  if (segments == 0) {
    return;
  }
  const auto grid =
      static_cast<unsigned>((segments + kRotateWaves - 1) / kRotateWaves);
  RotateRowsFp32Kernel<<<grid, kRotateWaves * 32, 0, stream>>>(
      x, signs, segments, row_segments);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
