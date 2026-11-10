#ifndef GUFO_CORE_QUANT_EXL3_COVERAGE_HPP_
#define GUFO_CORE_QUANT_EXL3_COVERAGE_HPP_

// Which EXL3 projections the fast kernels take, and the startup line naming
// the ones they do not. The decode GEMV (gemv.hpp Exl3FastSupported), the
// WMMA prefill GEMM (Exl3GemmShapeSupported) and the Flash-Next expert
// kernels all decide through FastKernelsSupport; anything else runs the
// reference path (reference.hpp), correct but slow, so a model load reports
// it once instead of slowing down silently.

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

#include "src/core/quant/exl3/device.hpp"

namespace gufo::quant::exl3 {

/// Whether the fast kernels take a k x n projection of `codebook` at `rate`:
/// every format the checkpoint loader accepts (Supports: K1..K8 of every
/// codebook, K1.5..K3.5 of mul1) with k and n positive multiples of 128.
[[nodiscard]] bool FastKernelsSupport(Codebook codebook, BitRate rate,
                                      std::size_t k, std::size_t n) noexcept;

/// The format of `rate` as the logs spell it ("K4", "K3.5").
[[nodiscard]] std::string RateName(BitRate rate);

/// The codebook as the logs spell it ("mul1", "mcg", "3inst").
[[nodiscard]] std::string_view CodebookName(Codebook codebook) noexcept;

/// The startup line of one model load:
///   event=exl3_reference_path model=<model> tensors=<N> of=<total>
///   formats=<codebook>/<rate>/<k>x<n>:<count>,...
/// listing every projection FastKernelsSupport rejects, grouped by format and
/// shape in first-seen order (formats=none when N is 0).
[[nodiscard]] std::string DescribeReferencePath(
    std::string_view model, std::span<const DeviceLinear* const> linears);

/// Logs DescribeReferencePath once: a warning when any projection runs the
/// reference path, otherwise an informational line.
void LogReferencePath(std::string_view model,
                      std::span<const DeviceLinear* const> linears);

}  // namespace gufo::quant::exl3

#endif  // GUFO_CORE_QUANT_EXL3_COVERAGE_HPP_
