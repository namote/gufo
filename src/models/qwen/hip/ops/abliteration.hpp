#ifndef GUFO_MODELS_QWEN_HIP_OPS_ABLITERATION_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_ABLITERATION_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Rows [previous span's end_row, end_row) use one session's device settings
/// (models::qwen::AbliterationSettings::DeviceWords); null leaves them
/// unchanged, as do rows past the last span.
struct QwenAbliterationSpan {
  std::uint32_t end_row;
  const float* settings;
};
/// One masked launch covers the sessions of a batched pass.
inline constexpr std::size_t kQwenAbliterationMaxSpans = 8;

/// In place, per row of an ON span: y -= w * d * dot(d, y), with an FP32
/// reduction. d is the session's direction row of `directions`
/// ([rows][hidden], unit) and w its strength for `layer`'s attention/GDN
/// (`mlp` false) or MLP writer; both are read on the device at run time, so a
/// captured graph follows in-place settings updates. Launches nothing when
/// every span is OFF; an ON span launches even at strength 0. Throws
/// std::invalid_argument past kQwenAbliterationMaxSpans spans.
void LaunchQwenAbliterateRows(float* rows, const float* directions,
                              std::span<const QwenAbliterationSpan> spans,
                              std::uint32_t layer, bool mlp,
                              std::uint32_t layers, std::uint32_t row_count,
                              std::uint32_t hidden, hipStream_t stream);
/// The same with one session's settings for every row.
void LaunchQwenAbliterateRows(float* rows, const float* directions,
                              const float* settings, std::uint32_t layer,
                              bool mlp, std::uint32_t layers,
                              std::uint32_t row_count, std::uint32_t hidden,
                              hipStream_t stream);
/// The same projection of the post-layer residual rows with each session's
/// `residual` strength for `layer`: r -= w * d * dot(d, r), so w = 1 leaves
/// no component along d. Callers launch it only for sessions whose settings
/// are AbliterationSettings::ResidualActive (null spans otherwise).
void LaunchQwenAbliterateResidualRows(
    float* rows, const float* directions,
    std::span<const QwenAbliterationSpan> spans, std::uint32_t layer,
    std::uint32_t layers, std::uint32_t row_count, std::uint32_t hidden,
    hipStream_t stream);
void LaunchQwenAbliterateResidualRows(float* rows, const float* directions,
                                      const float* settings,
                                      std::uint32_t layer, std::uint32_t layers,
                                      std::uint32_t row_count,
                                      std::uint32_t hidden, hipStream_t stream);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_ABLITERATION_HPP_
