#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_RECURRENT_STATE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_RECURRENT_STATE_HPP_

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace gufo::models::qwen38_flash_next::rocm {

/// Storage of the GDN recurrent state ([v_heads][d][d] per layer). The
/// recurrence computes in FP32 either way; FP16 is loaded once per pass and
/// stored once (rounded to nearest, saturated to +-65504), exllamav3
/// 0244652. Rollback records stay FP32.
enum class GdnStateFormat : std::uint8_t { kFp32, kFp16 };

[[nodiscard]] constexpr std::size_t GdnStateElementBytes(
    GdnStateFormat format) noexcept {
  return format == GdnStateFormat::kFp16 ? 2 : 4;
}

/// Canonical name: `--recurrent-state` values.
[[nodiscard]] constexpr std::string_view GdnStateFormatName(
    GdnStateFormat format) noexcept {
  return format == GdnStateFormat::kFp16 ? "fp16" : "fp32";
}

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_RECURRENT_STATE_HPP_
