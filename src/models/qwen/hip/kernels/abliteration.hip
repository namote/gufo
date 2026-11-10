#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <stdexcept>

#include "src/models/qwen/hip/ops/abliteration.hpp"

namespace gufo::hip {
namespace {

constexpr unsigned kAbliterationThreads = 256;

struct RowWeights {
  std::array<QwenAbliterationSpan, kQwenAbliterationMaxSpans> spans;
  unsigned count;
};

// One block per row: an FP32 dot product with the direction, then the
// projection removed in place. No intermediate tensor.
__launch_bounds__(kAbliterationThreads) __global__
    void AbliterateRowsKernel(float* __restrict__ rows,
                              const float* __restrict__ direction,
                              RowWeights weights, std::uint32_t hidden) {
  float weight = 0;
  for (unsigned i = 0; i < weights.count; ++i)
    if (blockIdx.x < weights.spans[i].end_row) {
      weight = weights.spans[i].weight;
      break;
    }
  if (weight == 0)
    return;
  __shared__ float partial[kAbliterationThreads];
  float* row = rows + std::size_t{blockIdx.x} * hidden;
  float sum = 0;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x)
    sum += row[i] * direction[i];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned step = kAbliterationThreads / 2; step; step >>= 1) {
    if (threadIdx.x < step)
      partial[threadIdx.x] += partial[threadIdx.x + step];
    __syncthreads();
  }
  const float amount = weight * partial[0];
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x)
    row[i] -= amount * direction[i];
}

}  // namespace

void LaunchQwenAbliterateRows(float* rows, const float* direction,
                              std::span<const QwenAbliterationSpan> spans,
                              std::uint32_t row_count, std::uint32_t hidden,
                              hipStream_t stream) {
  if (spans.size() > kQwenAbliterationMaxSpans)
    throw std::invalid_argument("abliteration batch exceeds eight sessions");
  if (!row_count ||
      std::ranges::none_of(spans, [](const auto& s) { return s.weight != 0; }))
    return;
  RowWeights weights{};
  weights.count = static_cast<unsigned>(spans.size());
  std::ranges::copy(spans, weights.spans.begin());
  AbliterateRowsKernel<<<row_count, kAbliterationThreads, 0, stream>>>(
      rows, direction, weights, hidden);
}

void LaunchQwenAbliterateRows(float* rows, const float* direction,
                              float weight, std::uint32_t row_count,
                              std::uint32_t hidden, hipStream_t stream) {
  const QwenAbliterationSpan span{row_count, weight};
  LaunchQwenAbliterateRows(rows, direction, std::span(&span, 1), row_count,
                           hidden, stream);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
