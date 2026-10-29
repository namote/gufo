#ifndef GUFO_CORE_QUANT_ROTATION_HPP_
#define GUFO_CORE_QUANT_ROTATION_HPP_

// Input-side Hadamard rotation of GGUF weights in the upstream llama.cpp
// `prism.hadamard.*` format (docs/ROTATED-WEIGHTS.md).
//
// A folded weight W' = W·D·H is listed in `prism.hadamard.weight_names`. D is
// the ±1 sign vector of the weight's input width K (one vector per width,
// shared by every folded weight with that K) and H is the block-diagonal
// normalized 128-point Sylvester Hadamard matrix. The runtime rotates the
// input once, x' = H·D·x, so W'·x' = W·x. The file has no extra tensors.
//
// ReadRotation validates in two stages: first every refusal of the upstream
// loader (llama.cpp src/llama-model.cpp load_hparams_hadamard and
// load_tensors_hadamard, line numbers in the messages), then the subset gufo
// executes, against a group table supplied by the model.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::quant::rotation {

inline constexpr std::string_view kUpstream =
    "llama.cpp 8ab29fae src/llama-model.cpp";
inline constexpr std::string_view kPrefix = "prism.hadamard.";
inline constexpr std::string_view kVersionKey = "prism.hadamard.version";
inline constexpr std::string_view kTiedOutputKey = "prism.hadamard.tied_output";
inline constexpr std::string_view kBlockSizeKey = "prism.hadamard.block_size";
inline constexpr std::string_view kTransformKey = "prism.hadamard.transform";
inline constexpr std::string_view kAxisKey = "prism.hadamard.axis";
inline constexpr std::string_view kSignModeKey = "prism.hadamard.sign_mode";
inline constexpr std::string_view kSignWidthsKey = "prism.hadamard.sign_widths";
inline constexpr std::string_view kSignValuesKey = "prism.hadamard.sign_values";
inline constexpr std::string_view kWeightNamesKey =
    "prism.hadamard.weight_names";
inline constexpr std::string_view kInverseWeightNamesKey =
    "prism.hadamard.inverse_weight_names";
inline constexpr std::string_view kGdnVGroupedKey =
    "prism.hadamard.gdn_v_grouped";
inline constexpr std::string_view kTransform =
    "normalized-sylvester-walsh-hadamard";
inline constexpr std::string_view kAxis = "input-last-dimension";
/// The only block size gufo executes.
inline constexpr std::uint32_t kBlockSize = 128;

/// One group in one layer: the matrices that read the same activation.
struct GroupInstance {
  std::string group;                 // e.g. "attn_in"
  std::string key;                   // e.g. "attn_in:3", or "head_in"
  std::vector<std::string> tensors;  // members present in the file
};

/// Model-supplied description of what gufo can execute rotated.
struct GroupTable {
  /// Every instance the model can execute. A folded weight outside every
  /// instance is refused because the runtime has no rotation path for it.
  std::vector<GroupInstance> instances;
  /// Groups listed in the table whose rotation the runtime cannot apply yet.
  std::vector<std::string> unsupported_groups;
};

/// Packed sign vector of one input width. Bit j of word w is set when
/// element 32·w + j has sign −1.
struct WidthSigns {
  std::size_t k{0};
  std::vector<std::uint32_t> bits;
};

struct RotationInfo {
  /// Rotated group names, in the order they first appear in the table.
  std::vector<std::string> rotated_groups;
  /// One entry per declared width, in file order.
  std::vector<WidthSigns> widths;
  std::vector<std::string> weight_names;

  [[nodiscard]] bool Any() const noexcept { return !widths.empty(); }
  [[nodiscard]] bool IsGroupRotated(std::string_view group) const noexcept;
  [[nodiscard]] const WidthSigns* ForWidth(std::size_t k) const noexcept;
  /// Comma-separated rotated group names, for refusal messages.
  [[nodiscard]] std::string GroupList() const;
};

[[nodiscard]] constexpr std::size_t PackedSignWords(std::size_t k) noexcept {
  return (k + 31) / 32;
}

/// Reads and validates the rotation metadata. Returns an empty RotationInfo
/// for an unrotated file and std::nullopt (with a message naming the key,
/// tensor, group or upstream line) for a file gufo must refuse, including
/// the retired gufo v1 format.
[[nodiscard]] std::optional<RotationInfo> ReadRotation(
    const core::GgufReader& reader, const GroupTable& table,
    std::string* error_msg = nullptr);

/// row ← H·D·row per 128-element segment, with the arithmetic contract of
/// src/core/hip/detail/hadamard.hpp (exact sign flip, butterflies of length
/// 1..64 in order, one final scale). For a weight row this computes
/// W' = W·D·H (H is symmetric); models use it to fold small matrices at load.
void RotateRow(std::span<float> row, std::span<const std::uint32_t> bits);

}  // namespace gufo::quant::rotation

#endif  // GUFO_CORE_QUANT_ROTATION_HPP_
