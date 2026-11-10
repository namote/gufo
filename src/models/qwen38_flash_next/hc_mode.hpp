#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_HC_MODE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_HC_MODE_HPP_

#include <cstdint>
#include <optional>
#include <string_view>

namespace gufo::models::qwen38_flash_next {

/// How the hyper-connection mixers' down/up projections compute
/// (`--hc-mode`). The modes give different numbers: cached states of one
/// are never restored into another.
enum class HcMode : std::uint8_t {
  /// The stored F16 weights, unconverted (default).
  kFp16,
  /// Q8_0 copies of the F16 weights, rounded at load without calibration.
  kQ8,
  /// The calibrated Q8_0 sidecar beside the pack (`hc_q8/`, `hc_q8.json`);
  /// loading fails without a valid one.
  kQ8Calibrated,
};

inline constexpr HcMode kDefaultHcMode = HcMode::kFp16;

[[nodiscard]] constexpr std::string_view HcModeName(HcMode mode) noexcept {
  switch (mode) {
    case HcMode::kQ8:
      return "q8";
    case HcMode::kQ8Calibrated:
      return "q8-calibrated";
    case HcMode::kFp16:
      break;
  }
  return "fp16";
}

/// The one canonical spelling of each mode.
[[nodiscard]] constexpr std::optional<HcMode> ParseHcMode(
    std::string_view value) noexcept {
  for (const auto mode : {HcMode::kFp16, HcMode::kQ8, HcMode::kQ8Calibrated})
    if (value == HcModeName(mode))
      return mode;
  return std::nullopt;
}

/// Device placement of one mixer's down/up weights under a mode.
struct HcMixerPlan {
  bool valid{true};          ///< False: the sidecar presence contradicts mode.
  bool calibrated{false};    ///< Upload the sidecar's down/up Q8_0 pair.
  bool convert_down{false};  ///< Add a Q8_0 copy of the F16 down.
  bool convert_up{false};    ///< Replace the F16 up by its Q8_0 copy.
};

/// `calibrated` whether the mixer carries the sidecar pair; `down_f16` /
/// `up_f16` whether each stored operand is a plain F16 matrix with 32-wide
/// rows; `exl3` a native EXL3 pack. Only q8-calibrated uploads the sidecar,
/// and it must be present there; only q8 on a native EXL3 pack converts (no
/// GGUF reader takes a Q8_0 copy); fp16 uploads the stored weights.
[[nodiscard]] constexpr HcMixerPlan PlanHcMixer(HcMode mode, bool calibrated,
                                                bool down_f16, bool up_f16,
                                                bool exl3) noexcept {
  if (calibrated != (mode == HcMode::kQ8Calibrated))
    return {.valid = false};
  if (calibrated)
    return {.calibrated = true};
  if (mode != HcMode::kQ8 || !exl3 || !down_f16)
    return {};
  return {.convert_down = true, .convert_up = up_f16};
}

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_HC_MODE_HPP_
