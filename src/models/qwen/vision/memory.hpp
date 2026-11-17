#ifndef GUFO_MODELS_QWEN_VISION_MEMORY_HPP_
#define GUFO_MODELS_QWEN_VISION_MEMORY_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace gufo::models::qwen::vision {

/// Row-local stages (patch embedding, layer norms, projections, FFN, merger)
/// run over at most this many patches at a time. Images up to 4096 patches
/// (1024x1024) run in one chunk, the unchunked graph. Multiple of 4 so a chunk
/// holds whole 2x2 merge groups.
inline constexpr std::uint32_t kEncodeChunkRows = 4096;
/// Bound for one attention tile's FP32 scores plus probabilities. Tiles stay
/// 128 rows (64 at 4096 patches) up to 10,922 patches.
inline constexpr std::size_t kAttentionTileBytes = std::size_t{128} << 20;
inline constexpr std::uint32_t kMinAttentionTile = 16;

/// How Encoder::Encode splits one image; also the basis of its memory census.
/// Attention is global over all patches, so Q/K/V, the residual and the
/// attention output stay whole-image; everything else is chunked.
struct EncodePlan {
  std::uint32_t count;       ///< 16x16 patches.
  std::uint32_t chunk_rows;  ///< Patches per row-local chunk.
  std::uint32_t tile;        ///< Query rows per attention tile.
};

inline EncodePlan PlanImageEncoding(std::uint32_t width, std::uint32_t height) {
  const std::uint64_t pixels = std::uint64_t{width} * height;
  if (!width || !height || width % 32 || height % 32 || pixels > 16777216)
    throw std::invalid_argument("invalid vision memory geometry");
  const auto count = static_cast<std::uint32_t>(pixels / 256);
  // Smaller attention tiles reduce scratch traffic for the 4096-patch shape.
  std::uint32_t tile = count == 4096 ? 64 : 128;
  const std::size_t limit =
      kAttentionTileBytes / (std::size_t{16} * count * (4 + 2));
  if (limit < tile)
    tile = std::max<std::uint32_t>(
        kMinAttentionTile, static_cast<std::uint32_t>(
                               limit / kMinAttentionTile * kMinAttentionTile));
  return {count, std::min(count, kEncodeChunkRows), std::min(count, tile)};
}

struct ImageMemory {
  std::size_t embedding_bytes;
  std::size_t scratch_bytes;
};

/// Allocation sizes in Encoder::Encode, excluding resident weights, CPU input
/// and implementation-owned BLAS workspace. Admission retains separate device
/// and host headroom for the runtime. RGB upload's pinned copy is included.
inline ImageMemory EstimateImageMemory(std::uint32_t width,
                                       std::uint32_t height,
                                       std::uint32_t output_width,
                                       bool native_exl3 = false) {
  if (output_width > 65536 || output_width == 0)
    throw std::invalid_argument("invalid vision memory geometry");
  const auto plan = PlanImageEncoding(width, height);
  const std::size_t pixels = std::size_t{width} * height;
  const std::size_t count = plan.count;
  const std::size_t rows = plan.chunk_rows;
  const std::size_t tile = plan.tile;
  const std::size_t ffn = native_exl3 ? 4352 : 4304;
  const std::size_t embedding = count / 4 * output_width * sizeof(float);
  // Whole image: RGB, pinned upload (at most 16 MiB), FP32 residual, Q/K/V,
  // attention output, one attention tile's scores/probabilities/output.
  const std::size_t whole =
      pixels * 3 + std::min(pixels * 3, std::size_t{16} << 20) +
      count * 1152 * 4 + 3 * count * 1152 * 2 + count * 1152 * 2 +
      16 * tile * count * (4 + 2) + tile * 1152 * 4;
  // One chunk: FP32 projection output and 16-bit normalized/activated input.
  const std::size_t chunk = rows * std::max<std::size_t>(ffn, 3 * 1152) * 4 +
                            rows * std::max<std::size_t>(ffn, 1536) * 2;
  // Native projections retain packed resident weights; per chunk they stage
  // FP32/FP16 rotated inputs and an FP32 output. Non-WMMA EXL3 rates reuse one
  // decoded-inner slab per image. Charge the largest supported matrix, even
  // for K6 packs whose fast kernels do not need that slab.
  const std::size_t native_scratch =
      native_exl3 ? rows * (4608 * 4 + 4608 * 2 + 5120 * 4) +
                        std::size_t{4608} * std::max(4608U, output_width) * 6
                  : 0;
  return {embedding, whole + chunk + native_scratch};
}

}  // namespace gufo::models::qwen::vision
#endif
