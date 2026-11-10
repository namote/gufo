#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <stdexcept>

#include "src/models/qwen/abliteration.hpp"
#include "src/models/qwen/hip/kernels/abliteration_row.hpp"
#include "src/models/qwen/hip/ops/abliteration.hpp"

namespace gufo::hip {
namespace {

constexpr unsigned kAbliterationThreads =
    models::qwen::hip_detail::kAbliterationRowThreads;

struct RowSettings {
  std::array<QwenAbliterationSpan, kQwenAbliterationMaxSpans> spans;
  unsigned count;
};

// One block per row: the session's direction(s) projected out in place
// (hip_detail::AbliterateRow). No intermediate tensor. The direction rows,
// count, norm preservation and strength are read from the session's device
// settings at run time.
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
  models::qwen::hip_detail::AbliterateRow(
      rows + std::size_t{blockIdx.x} * hidden, directions, settings,
      weight_index, hidden);
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
