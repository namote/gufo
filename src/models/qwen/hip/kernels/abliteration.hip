#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <stdexcept>

#include "src/models/qwen/abliteration.hpp"
#include "src/models/qwen/hip/ops/abliteration.hpp"

namespace gufo::hip {
namespace {

constexpr unsigned kAbliterationThreads = 256;

struct RowSettings {
  std::array<QwenAbliterationSpan, kQwenAbliterationMaxSpans> spans;
  unsigned count;
};

// One block per row: an FP32 dot product with the session's direction, then
// the projection removed in place. No intermediate tensor. The direction row
// and strength are read from the session's device settings at run time.
__launch_bounds__(kAbliterationThreads) __global__
    void AbliterateRowsKernel(float* __restrict__ rows,
                              const float* __restrict__ directions,
                              RowSettings sessions, std::uint32_t weight_index,
                              std::uint32_t hidden) {
  const float* settings = nullptr;
  for (unsigned i = 0; i < sessions.count; ++i)
    if (blockIdx.x < sessions.spans[i].end_row) {
      settings = sessions.spans[i].settings;
      break;
    }
  if (settings == nullptr)
    return;
  const float weight = settings[weight_index];
  if (weight == 0)
    return;
  const float* direction =
      directions + std::size_t{__float_as_uint(settings[0])} * hidden;
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

void Launch(float* rows, const float* directions,
            std::span<const QwenAbliterationSpan> spans, std::uint32_t layer,
            models::qwen::AbliterationPoint point, std::uint32_t layers,
            std::uint32_t row_count, std::uint32_t hidden, hipStream_t stream) {
  if (spans.size() > kQwenAbliterationMaxSpans)
    throw std::invalid_argument("abliteration batch exceeds eight sessions");
  if (layer >= layers)
    throw std::invalid_argument("abliteration layer out of range");
  if (!row_count || std::ranges::none_of(spans, [](const auto& s) {
        return s.settings != nullptr;
      }))
    return;
  RowSettings sessions{};
  sessions.count = static_cast<unsigned>(spans.size());
  std::ranges::copy(spans, sessions.spans.begin());
  AbliterateRowsKernel<<<row_count, kAbliterationThreads, 0, stream>>>(
      rows, directions, sessions,
      models::qwen::AbliterationWeightIndex(layers, layer, point), hidden);
}

models::qwen::AbliterationPoint Writer(bool mlp) {
  return mlp ? models::qwen::AbliterationPoint::kMlp
             : models::qwen::AbliterationPoint::kAttention;
}

}  // namespace

void LaunchQwenAbliterateRows(float* rows, const float* directions,
                              std::span<const QwenAbliterationSpan> spans,
                              std::uint32_t layer, bool mlp,
                              std::uint32_t layers, std::uint32_t row_count,
                              std::uint32_t hidden, hipStream_t stream) {
  Launch(rows, directions, spans, layer, Writer(mlp), layers, row_count,
         hidden, stream);
}

void LaunchQwenAbliterateRows(float* rows, const float* directions,
                              const float* settings, std::uint32_t layer,
                              bool mlp, std::uint32_t layers,
                              std::uint32_t row_count, std::uint32_t hidden,
                              hipStream_t stream) {
  const QwenAbliterationSpan span{row_count, settings};
  Launch(rows, directions, std::span(&span, 1), layer, Writer(mlp), layers,
         row_count, hidden, stream);
}

void LaunchQwenAbliterateResidualRows(
    float* rows, const float* directions,
    std::span<const QwenAbliterationSpan> spans, std::uint32_t layer,
    std::uint32_t layers, std::uint32_t row_count, std::uint32_t hidden,
    hipStream_t stream) {
  Launch(rows, directions, spans, layer,
         models::qwen::AbliterationPoint::kResidual, layers, row_count, hidden,
         stream);
}

void LaunchQwenAbliterateResidualRows(float* rows, const float* directions,
                                      const float* settings,
                                      std::uint32_t layer, std::uint32_t layers,
                                      std::uint32_t row_count,
                                      std::uint32_t hidden,
                                      hipStream_t stream) {
  const QwenAbliterationSpan span{row_count, settings};
  LaunchQwenAbliterateResidualRows(rows, directions, std::span(&span, 1),
                                   layer, layers, row_count, hidden, stream);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
