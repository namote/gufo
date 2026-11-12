#include "src/models/qwen/abliteration.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

#include "src/models/qwen/hip/kernels/abliteration_row.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {
struct RowSettings {
  std::array<AbliterationSpan, 8> spans;
  unsigned count;
};
// One block per row (qwen::hip_detail::AbliterateRow): the session's
// direction(s) projected out in place, no intermediate tensor. Each row reads
// its session's device settings at run time (direction rows, count, norm
// preservation and this site's strength), so captured graphs follow in-place
// updates. Spans count tokens; each token owns `streams` consecutive rows (1
// for writer outputs, the HC streams for the residual), each projected and
// norm-preserved on its own.
__global__ void Project(float* output, const float* directions,
                        RowSettings rows, std::uint32_t weight_index,
                        std::uint32_t hidden, std::uint32_t streams) {
  const unsigned token = blockIdx.x / streams;
  const float* settings = nullptr;
  for (unsigned i = 0; i < rows.count; ++i)
    if (token < rows.spans[i].end_row) {
      settings = rows.spans[i].settings;
      break;
    }
  if (settings == nullptr)
    return;
  qwen::hip_detail::AbliterateRow(output + std::size_t{blockIdx.x} * hidden,
                                  directions, settings, weight_index, hidden);
}

void Launch(float* output, const float* directions,
            std::span<const AbliterationSpan> spans, std::uint32_t layer,
            qwen::AbliterationPoint point, std::uint32_t layers,
            std::uint32_t tokens, std::uint32_t hidden, std::uint32_t streams,
            hipStream_t stream) {
  if (spans.size() > 8)
    throw std::invalid_argument("abliteration batch exceeds eight sessions");
  if (layer >= layers)
    throw std::invalid_argument("abliteration layer out of range");
  if (streams == 0)
    throw std::invalid_argument("abliteration needs at least one stream");
  if (!tokens || std::ranges::none_of(spans, [](const auto& s) {
        return s.settings != nullptr;
      }))
    return;
  RowSettings settings{};
  settings.count = spans.size();
  std::copy(spans.begin(), spans.end(), settings.spans.begin());
  hipLaunchKernelGGL(Project, dim3(tokens * streams),
                     dim3(qwen::hip_detail::kAbliterationRowThreads), 0, stream,
                     output, directions, settings,
                     qwen::AbliterationWeightIndex(layers, layer, point),
                     hidden, streams);
}
}  // namespace

void AbliterateBatchRows(float* output, const float* directions,
                         std::span<const AbliterationSpan> spans,
                         std::uint32_t layer, bool mlp, std::uint32_t layers,
                         std::uint32_t rows, std::uint32_t hidden,
                         hipStream_t stream) {
  Launch(
      output, directions, spans, layer,
      mlp ? qwen::AbliterationPoint::kMlp : qwen::AbliterationPoint::kAttention,
      layers, rows, hidden, 1, stream);
}
void AbliterateRows(float* output, const float* directions,
                    const float* settings, std::uint32_t layer, bool mlp,
                    std::uint32_t layers, std::uint32_t rows,
                    std::uint32_t hidden, hipStream_t stream) {
  const AbliterationSpan span{rows, settings};
  AbliterateBatchRows(output, directions, std::span(&span, 1), layer, mlp,
                      layers, rows, hidden, stream);
}
namespace {
struct DetectorList {
  std::array<std::uint32_t, qwen::kAbliterationMaxProfiles> detector;
  std::array<std::uint32_t, qwen::kAbliterationMaxProfiles> column;
};
// One block per (row, detector): the HC-stream mean's dot with the detector
// direction, as the sum over streams of each stream's dot divided by the
// stream count.
__global__ void DetectorScores(const float* __restrict__ res,
                               const float* __restrict__ directions,
                               DetectorList list, float* __restrict__ scores,
                               std::uint32_t stride, std::uint32_t hidden,
                               std::uint32_t streams) {
  constexpr unsigned kThreads = qwen::hip_detail::kAbliterationRowThreads;
  __shared__ float partial[kThreads];
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t which = blockIdx.y;
  const float* direction =
      directions + std::size_t{list.detector[which]} * hidden;
  const float* x = res + std::size_t{row} * streams * hidden;
  float sum = 0;
  for (std::uint32_t s = 0; s < streams; ++s)
    for (std::uint32_t i = threadIdx.x; i < hidden; i += kThreads)
      sum += x[std::size_t{s} * hidden + i] * direction[i];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned step = kThreads / 2; step; step >>= 1) {
    if (threadIdx.x < step)
      partial[threadIdx.x] += partial[threadIdx.x + step];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    scores[std::size_t{row} * stride + list.column[which]] =
        partial[0] / static_cast<float>(streams);
}
}  // namespace

void AbliterationDetectorScores(const float* res, const float* directions,
                                std::span<const std::uint32_t> detectors,
                                std::span<const std::uint32_t> columns,
                                float* scores, std::uint32_t stride,
                                std::uint32_t rows, std::uint32_t hidden,
                                std::uint32_t streams, hipStream_t stream) {
  if (detectors.size() != columns.size() ||
      detectors.size() > qwen::kAbliterationMaxProfiles || streams == 0)
    throw std::invalid_argument("invalid abliteration detector list");
  if (!rows || detectors.empty())
    return;
  DetectorList list{};
  std::ranges::copy(detectors, list.detector.begin());
  std::ranges::copy(columns, list.column.begin());
  hipLaunchKernelGGL(DetectorScores,
                     dim3(rows, static_cast<unsigned>(detectors.size())),
                     dim3(qwen::hip_detail::kAbliterationRowThreads), 0, stream,
                     res, directions, list, scores, stride, hidden, streams);
}

void AbliterateResidualBatchRows(float* res, const float* directions,
                                 std::span<const AbliterationSpan> spans,
                                 std::uint32_t layer, std::uint32_t layers,
                                 std::uint32_t tokens, std::uint32_t hidden,
                                 std::uint32_t streams, hipStream_t stream) {
  Launch(res, directions, spans, layer, qwen::AbliterationPoint::kResidual,
         layers, tokens, hidden, streams, stream);
}
}  // namespace gufo::models::qwen38_flash_next::rocm
