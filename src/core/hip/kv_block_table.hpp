#ifndef GUFO_CORE_HIP_KV_BLOCK_TABLE_HPP_
#define GUFO_CORE_HIP_KV_BLOCK_TABLE_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::hip {

/// Tokens per KV block. Every sequence tile of the attention kernels (16, 64,
/// split-K stages of 4, online pairs of 2) divides it.
inline constexpr std::uint32_t kKvBlockTokens = 64;
inline constexpr std::uint32_t kKvBlockShift = 6;
static_assert(kKvBlockTokens == 1U << kKvBlockShift);

/// Where a sequence's token rows live in a layer-major KV block pool. Each
/// plane is [layer][pool block][kKvBlockTokens][row]; `blocks` maps a logical
/// block of the sequence to its pool block. A null `blocks` names contiguous
/// storage of `max_context` rows per layer, for caches outside a pool.
struct KvBlockTable {
  const std::uint32_t* blocks{nullptr};
  std::uint32_t layer_tokens{0};
};

/// First row of `layer` (in token rows, before multiplying by the row width).
__host__ __device__ __forceinline__ std::size_t KvLayerRow(
    const KvBlockTable& table, std::uint32_t layer, std::uint32_t max_context) {
  return static_cast<std::size_t>(layer) *
         (table.blocks == nullptr ? max_context : table.layer_tokens);
}

/// Row of `pos` relative to its layer's first row.
__host__ __device__ __forceinline__ std::size_t KvRowInLayer(
    const KvBlockTable& table, std::size_t pos) {
  if (table.blocks == nullptr)
    return pos;
  return (static_cast<std::size_t>(table.blocks[pos >> kKvBlockShift])
          << kKvBlockShift) +
         (pos & (kKvBlockTokens - 1));
}

/// Row of `pos` in `layer`.
__host__ __device__ __forceinline__ std::size_t KvTokenRow(
    const KvBlockTable& table, std::uint32_t layer, std::uint32_t max_context,
    std::size_t pos) {
  return KvLayerRow(table, layer, max_context) + KvRowInLayer(table, pos);
}

/// First position after `pos`'s block: positions [pos, KvBlockEnd(pos)) have
/// consecutive rows.
__host__ __device__ __forceinline__ std::size_t KvBlockEnd(std::size_t pos) {
  return (pos | (kKvBlockTokens - 1)) + 1;
}

}  // namespace gufo::hip

#endif  // GUFO_CORE_HIP_KV_BLOCK_TABLE_HPP_
