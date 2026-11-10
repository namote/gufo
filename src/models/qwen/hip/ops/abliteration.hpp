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

/// In place, per row of an ON span: y -= w * sum_j d_j * dot(d_j, y), with
/// FP32 reductions of dots taken from the unprojected row. d_j are the
/// session's direction_count orthonormal rows of `directions`
/// ([rows][hidden]) and w its strength for `layer`'s attention/GDN (`mlp`
/// false) or MLP writer; with preserve_norm the row is then rescaled to its
/// former L2 norm. All of it is read on the device at run time
/// (models::qwen::AbliterationSettings::DeviceWords), so a captured graph
/// follows in-place settings updates. One direction without norm
/// preservation is the single-direction projection, bit for bit. Launches
/// nothing when every span is OFF; an ON span launches even at strength 0.
/// Throws std::invalid_argument past kQwenAbliterationMaxSpans spans.
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
/// `residual` strength for `layer`: r -= w * sum_j d_j * dot(d_j, r), so
/// w = 1 leaves no component in the subspace (before any norm rescale). Callers
/// launch it only for sessions whose settings are
/// AbliterationSettings::ResidualActive (null spans otherwise).
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
