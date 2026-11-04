#ifndef GUFO_CORE_QUANT_EXL3_LAYOUT_HPP_
#define GUFO_CORE_QUANT_EXL3_LAYOUT_HPP_

// Model-independent pieces shared by the EXL3 loaders of the Qwen families:
// the GGUF metadata keys that name an EXL3 source, and the gated-delta-net
// value-head reorder from Hugging Face order to the GGUF (llama.cpp) order.

#include <cstddef>
#include <string_view>

namespace gufo::quant::exl3 {

/// GGUF metadata key naming the EXL3 checkpoint directory of a metadata
/// image.
inline constexpr std::string_view kCheckpointKey = "gufo.exl3.checkpoint";

/// GGUF metadata key set (true) on metadata built in memory from an EXL3
/// directory instead of read from a file.
inline constexpr std::string_view kNativeKey = "gufo.exl3.native";

/// Index of element `index` of a linear-attention value dimension after the
/// grouped -> tiled head reorder of llama.cpp's convert_hf_to_gguf.py
/// (_LinearAttentionVReorderBase). Hugging Face stores the value heads
/// grouped by key head (k head a, v head b of the group, element h); GGUF
/// stores them tiled (b, a, h). `head_dim` is the elements per head (1 for
/// per-head scalars). num_v_heads must be a multiple of num_k_heads.
[[nodiscard]] constexpr std::size_t TiledVHeadIndex(
    std::size_t index, std::size_t head_dim, std::size_t num_k_heads,
    std::size_t num_v_heads) noexcept {
  const std::size_t per_k = num_v_heads / num_k_heads;
  const std::size_t head = index / head_dim;
  return ((head % per_k) * num_k_heads + head / per_k) * head_dim +
         index % head_dim;
}

}  // namespace gufo::quant::exl3

#endif  // GUFO_CORE_QUANT_EXL3_LAYOUT_HPP_
