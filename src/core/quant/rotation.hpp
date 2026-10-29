#ifndef GUFO_CORE_QUANT_ROTATION_HPP_
#define GUFO_CORE_QUANT_ROTATION_HPP_

// Input-side Hadamard rotation markers (docs/ROTATED-WEIGHTS.md, v1).
//
// A rotated GGUF stores W' = W·D·H for the matrices of an input group, where
// D is a ±1 diagonal and H is the block-diagonal normalized 128-point
// Sylvester Hadamard matrix. The runtime rotates the shared input once,
// x' = H·D·x, so W'·x' = W·x. This module reads the global markers and the
// `<name>.rot` sign tensors and applies the model-independent refusal rules
// to a group table supplied by the model. Models own the table, the binding
// of signs to execution paths and the kernels.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::quant::rotation {

inline constexpr std::uint32_t kVersion = 1;
inline constexpr std::uint32_t kBlockSize = 128;
inline constexpr std::string_view kKind = "hadamard";
inline constexpr std::string_view kSide = "input";
inline constexpr std::string_view kSignSuffix = ".rot";
inline constexpr std::string_view kVersionKey = "gufo.rotation.version";
inline constexpr std::string_view kKindKey = "gufo.rotation.kind";
inline constexpr std::string_view kBlockSizeKey = "gufo.rotation.block_size";
inline constexpr std::string_view kSideKey = "gufo.rotation.side";
inline constexpr std::uint16_t kF16PlusOne = 0x3C00;
inline constexpr std::uint16_t kF16MinusOne = 0xBC00;

/// One group in one layer: the matrices that read the same activation.
struct GroupInstance {
  std::string group;                 // e.g. "attn_in"
  std::string key;                   // e.g. "attn_in:3", or "head_in"
  std::vector<std::string> tensors;  // members present in the file
};

/// Model-supplied description of where rotation may appear.
struct GroupTable {
  /// Every instance the model can execute. A `.rot` on any other tensor is
  /// refused because the runtime has no rotation path for its role.
  std::vector<GroupInstance> instances;
  /// Groups listed in the table whose rotation the runtime cannot apply yet.
  std::vector<std::string> unsupported_groups;
  /// Tensors that also serve as an embedding lookup (a tied output head).
  /// Rotating them would change the lookup, so they are refused by name.
  std::vector<std::string> tied_tensors;
};

/// Packed sign vector of one rotated group instance. Bit j of word w is set
/// when element 32·w + j has sign −1.
struct InstanceSigns {
  std::string group;
  std::string key;
  std::size_t k{0};
  std::vector<std::uint32_t> bits;
};

struct RotationInfo {
  /// Rotated group names, in the order they first appear in the table.
  std::vector<std::string> rotated_groups;
  std::vector<InstanceSigns> instances;

  [[nodiscard]] bool Any() const noexcept { return !instances.empty(); }
  [[nodiscard]] bool IsGroupRotated(std::string_view group) const noexcept;
  [[nodiscard]] const InstanceSigns* Find(std::string_view key) const noexcept;
  /// Comma-separated rotated group names, for refusal messages.
  [[nodiscard]] std::string GroupList() const;
};

[[nodiscard]] constexpr std::size_t PackedSignWords(std::size_t k) noexcept {
  return (k + 31) / 32;
}

/// Reads the rotation markers and sign tensors and checks them against
/// `table`. Returns an empty RotationInfo for an unrotated file and
/// std::nullopt (with a message naming the offending key, tensor or group)
/// when the file breaks a v1 rule.
[[nodiscard]] std::optional<RotationInfo> ReadRotation(
    const core::GgufReader& reader, const GroupTable& table,
    std::string* error_msg = nullptr);

}  // namespace gufo::quant::rotation

#endif  // GUFO_CORE_QUANT_ROTATION_HPP_
