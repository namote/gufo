#ifndef GUFO_MODELS_QWEN_HIP_OPS_ABLITERATION_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_ABLITERATION_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Rows [previous span's end_row, end_row) use `weight`; rows past the last
/// span are unchanged.
struct QwenAbliterationSpan {
  std::uint32_t end_row;
  float weight;
};
/// One masked launch covers the sessions of a batched pass.
inline constexpr std::size_t kQwenAbliterationMaxSpans = 8;

/// In place, per row: y -= weight * direction * dot(direction, y), with an
/// FP32 reduction and a unit `direction` of `hidden` floats. Launches nothing
/// when every weight is zero. Throws std::invalid_argument past
/// kQwenAbliterationMaxSpans spans.
void LaunchQwenAbliterateRows(float* rows, const float* direction,
                              std::span<const QwenAbliterationSpan> spans,
                              std::uint32_t row_count, std::uint32_t hidden,
                              hipStream_t stream);
/// The same with one weight for every row.
void LaunchQwenAbliterateRows(float* rows, const float* direction, float weight,
                              std::uint32_t row_count, std::uint32_t hidden,
                              hipStream_t stream);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_ABLITERATION_HPP_
