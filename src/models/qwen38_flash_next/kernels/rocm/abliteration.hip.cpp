#include "src/models/qwen/abliteration.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {
struct RowSettings {
  std::array<AbliterationSpan, 8> spans;
  unsigned count;
};
// One block per row, FP32 dot product and projection. No intermediate tensor.
// Each row reads its session's device settings at run time (direction row and
// this site's strength), so captured graphs follow in-place updates. Spans
// count tokens; each token owns `streams` consecutive rows (1 for writer
// outputs, the HC streams for the residual).
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
  const float weight = settings[weight_index];
  if (weight == 0)
    return;
  const float* direction =
      directions + std::size_t{__float_as_uint(settings[0])} * hidden;
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
  hipLaunchKernelGGL(Project, dim3(tokens * streams), dim3(256), 0, stream,
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
void AbliterateResidualBatchRows(float* res, const float* directions,
                                 std::span<const AbliterationSpan> spans,
                                 std::uint32_t layer, std::uint32_t layers,
                                 std::uint32_t tokens, std::uint32_t hidden,
                                 std::uint32_t streams, hipStream_t stream) {
  Launch(res, directions, spans, layer, qwen::AbliterationPoint::kResidual,
         layers, tokens, hidden, streams, stream);
}
}  // namespace gufo::models::qwen38_flash_next::rocm
