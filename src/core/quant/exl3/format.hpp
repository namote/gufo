#ifndef GUFO_CORE_QUANT_EXL3_FORMAT_HPP_
#define GUFO_CORE_QUANT_EXL3_FORMAT_HPP_

// CPU reference of the EXL3 weight format (docs/EXL3-FORMAT.md): trellis
// state extraction, the three procedural codebooks, the 16 x 16 tile layout,
// the 128-point Hadamard rotations and a reference GEMV. Bit-exact with
// upstream exllamav3 v1.6.5 (cdc785db) for the inner dequant; see the format
// document for the rounding of the rotated weight.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace gufo::quant::exl3 {

/// Procedural codebook of a tensor, selected by the marker tensor next to its
/// trellis (none: 3INST).
enum class Codebook : std::uint8_t { k3Inst, kMcg, kMul1 };

/// Bits per weight of a tensor: an integer 1..8, or bits + 0.5 for the
/// half-integer rates 1.5, 2.5 and 3.5 (mul1 only).
struct BitRate {
  int bits = 0;
  bool half = false;

  /// uint16 words per 256-weight tile (16 * K).
  [[nodiscard]] constexpr int TileWords() const noexcept {
    return 16 * bits + (half ? 8 : 0);
  }
  /// Bits per tile (256 * K).
  [[nodiscard]] constexpr int TileBits() const noexcept {
    return TileWords() * 16;
  }
  friend constexpr bool operator==(BitRate, BitRate) = default;
};

/// The bit rate of a trellis whose last dimension is `tile_words`, or nullopt
/// when it is not a supported rate (upstream bits_k.cuh).
[[nodiscard]] std::optional<BitRate> BitRateFromTileWords(
    std::size_t tile_words) noexcept;

/// Whether `codebook` supports `rate` (half-integer rates need mul1).
[[nodiscard]] bool Supports(Codebook codebook, BitRate rate) noexcept;

/// Values per 16 x 16 tile.
inline constexpr int kTileSize = 256;
/// Hadamard block size of the input and output rotations.
inline constexpr int kHadamardSize = 128;

/// Exclusive end bit of position `position` (0..255) in the tile bit string.
[[nodiscard]] int StateEndBit(BitRate rate, int position) noexcept;

/// The 16-bit trellis state of `position`: the cyclic 16-bit window ending at
/// StateEndBit, most significant bit first. `tile` holds rate.TileWords()
/// uint16 words.
[[nodiscard]] std::uint16_t TileState(const std::uint16_t* tile, BitRate rate,
                                      int position) noexcept;

/// FP16 bit pattern of `state` under `codebook`.
[[nodiscard]] std::uint16_t DecodeState(std::uint16_t state,
                                        Codebook codebook) noexcept;

/// Row-major tile element (r * 16 + c) that position p decodes into (the
/// tensor-core permutation of upstream quantize.py tensor_core_perm).
[[nodiscard]] const std::array<std::uint8_t, kTileSize>&
TilePositionToElement() noexcept;

/// Decodes one tile into 256 FP16 values in row-major 16 x 16 order.
void DequantTile(const std::uint16_t* tile, BitRate rate, Codebook codebook,
                 std::span<std::uint16_t, kTileSize> out) noexcept;

/// Decodes a whole trellis [k / 16][n / 16][TileWords] into the inner matrix
/// W' [k][n] (row-major FP16). `k` and `n` must be multiples of 16; `out` has
/// k * n elements.
void DequantInner(const std::uint16_t* trellis, std::size_t k, std::size_t n,
                  BitRate rate, Codebook codebook, std::uint16_t* out);

/// Decodes columns [n_begin, n_begin + n_count) of every row: out is
/// [k][n_count]. n_begin and n_count must be multiples of 16.
void DequantInnerColumns(const std::uint16_t* trellis, std::size_t k,
                         std::size_t n, BitRate rate, Codebook codebook,
                         std::size_t n_begin, std::size_t n_count,
                         std::uint16_t* out);

/// In-place orthonormal Hadamard rotation of every 128-element block of
/// `values` (natural-order Sylvester, scaled by 1 / sqrt(128)).
void HadamardBlocks(std::span<double> values) noexcept;
void HadamardBlocks(std::span<float> values) noexcept;

/// y = ((x * suh) H_k) W' H_n * svh in double precision, the exact form of
/// the EXL3 forward pass (docs/EXL3-FORMAT.md "Rotations"). `inner` is W'
/// [k][n] FP16, `suh` [k] and `svh` [n] FP16, `x` [k], `y` [n].
void GemvReference(std::span<const double> x, const std::uint16_t* inner,
                   const std::uint16_t* suh, const std::uint16_t* svh,
                   std::size_t k, std::size_t n, std::span<double> y);

/// FP16 helpers (round to nearest even, subnormals kept).
[[nodiscard]] double HalfToDouble(std::uint16_t h) noexcept;
[[nodiscard]] std::uint16_t HalfFromDouble(double value) noexcept;

}  // namespace gufo::quant::exl3

#endif  // GUFO_CORE_QUANT_EXL3_FORMAT_HPP_
