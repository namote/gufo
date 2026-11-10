#include <algorithm>
#include <array>
#include <stdexcept>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {
struct RowWeights {
  std::array<AbliterationSpan, 8> spans;
  unsigned count;
};
// One block per row, FP32 dot product and projection. No intermediate tensor.
__global__ void Project(float* output, const float* direction,
                        RowWeights weights, std::uint32_t hidden) {
  float weight = 0;
  for (unsigned i = 0; i < weights.count; ++i)
    if (blockIdx.x < weights.spans[i].end_row) {
      weight = weights.spans[i].weight;
      break;
    }
  if (weight == 0)
    return;
  __shared__ float partial[256];
  float* row = output + std::size_t{blockIdx.x} * hidden;
  float sum = 0;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x)
    sum += row[i] * direction[i];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned step = 128; step; step >>= 1) {
    if (threadIdx.x < step)
      partial[threadIdx.x] += partial[threadIdx.x + step];
    __syncthreads();
  }
  const float amount = weight * partial[0];
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x)
    row[i] -= amount * direction[i];
}
}  // namespace

void AbliterateBatchRows(float* output, const float* direction,
                         std::span<const AbliterationSpan> spans,
                         std::uint32_t rows, std::uint32_t hidden,
                         hipStream_t stream) {
  if (spans.size() > 8)
    throw std::invalid_argument("abliteration batch exceeds eight sessions");
  if (!rows ||
      std::ranges::none_of(spans, [](const auto& s) { return s.weight != 0; }))
    return;
  RowWeights weights{};
  weights.count = spans.size();
  std::copy(spans.begin(), spans.end(), weights.spans.begin());
  hipLaunchKernelGGL(Project, dim3(rows), dim3(256), 0, stream, output,
                     direction, weights, hidden);
}
void AbliterateRows(float* output, const float* direction, float weight,
                    std::uint32_t rows, std::uint32_t hidden,
                    hipStream_t stream) {
  const AbliterationSpan span{rows, weight};
  AbliterateBatchRows(output, direction, std::span(&span, 1), rows, hidden,
                      stream);
}
}  // namespace gufo::models::qwen38_flash_next::rocm
