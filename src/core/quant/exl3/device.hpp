#ifndef GUFO_CORE_QUANT_EXL3_DEVICE_HPP_
#define GUFO_CORE_QUANT_EXL3_DEVICE_HPP_

// One EXL3 projection as the GPU kernels see it (plan/exl3/XA-RESULTS.md
// "X2a 설계 메모" section 2, agreed with X2b):
//
//   y = svh * H_n((H_k(suh * x)) W')     (docs/EXL3-FORMAT.md "Rotations")
//
// with W' the FP16 values the trellis decodes to, H the 128-point
// natural-order Sylvester matrix / sqrt(128), FP32 out and no re-rounding of
// the weights. The rotated activation H_k(suh * x) is FP32 in the X2a
// reference path and rounded to FP16 in the X2b kernels (as upstream does);
// the reference therefore differs from the fast path by FP16 activation
// rounding (relative ~1e-3), which kernel tests allow for
// (plan/exl3/XB-DESIGN.md section 8).

#include <cstdint>

#include "src/core/quant/exl3/format.hpp"

namespace gufo::quant::exl3 {

/// Trellis memory layout. Only the checkpoint layout exists; the value is
/// reserved so a kernel-friendly re-layout can be added without changing the
/// interface (plan/exl3/XB-DESIGN.md section 8).
enum class TrellisLayout : std::uint8_t {
  kCheckpoint,  ///< [k / 16][n / 16][TileWords] uint16, as stored
};

/// Kernel view of one projection computing y = x W, W [k][n]. The pointers
/// are host addresses after loading and device addresses (256-byte aligned)
/// once the model is on the GPU.
struct DeviceLinear {
  /// [k / 16][n / 16][rate.TileWords()] uint16 in checkpoint order.
  const std::uint16_t* trellis = nullptr;
  const std::uint16_t* suh = nullptr;  ///< FP16 [k]
  const std::uint16_t* svh = nullptr;  ///< FP16 [n]
  std::uint32_t k = 0;                 ///< in_features, a multiple of 128
  std::uint32_t n = 0;                 ///< out_features, a multiple of 128
  BitRate rate;
  Codebook codebook = Codebook::k3Inst;
  TrellisLayout layout = TrellisLayout::kCheckpoint;

  [[nodiscard]] std::size_t TrellisBytes() const noexcept {
    return static_cast<std::size_t>(k / 16) * (n / 16) *
           static_cast<std::size_t>(rate.TileWords()) * 2;
  }
};

}  // namespace gufo::quant::exl3

#endif  // GUFO_CORE_QUANT_EXL3_DEVICE_HPP_
