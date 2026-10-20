#ifndef GUFO_CORE_HIP_KV_BLOCK_LAYOUT_HPP_
#define GUFO_CORE_HIP_KV_BLOCK_LAYOUT_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gufo::hip {

/// Tokens per KV block. A block id names the same token span in every plane
/// of a pool, so a sequence keeps one block table for all of them. 256 keeps a
/// whole power-of-two row count per block for one row per token, per 4 tokens
/// and per 128 tokens, and equals the cache's 256-row chunk.
inline constexpr std::uint32_t kKvBlockTokens = 256;
inline constexpr std::uint32_t kKvBlockShift = 8;
static_assert(kKvBlockTokens == 1U << kKvBlockShift);

/// One kind of append-only rows: `layers` layers of rows `row_bytes` wide,
/// `rows_per_block` of them per block (kKvBlockTokens for one row per token,
/// fewer when a row covers several tokens). Rows of a block never move.
struct KvBlockPlane {
  std::uint32_t layers{0};
  std::size_t row_bytes{0};
  std::uint32_t rows_per_block{kKvBlockTokens};
  bool operator==(const KvBlockPlane&) const = default;
};

/// Byte layout of a pool of `blocks` blocks over `planes`. Planes follow one
/// another; each is [layer][block][rows_per_block][row_bytes], so one layer
/// of one plane is a single span (layer_pitch bytes) and a block's rows in it
/// are contiguous. Validated on construction.
class KvBlockGeometry {
public:
  KvBlockGeometry() = default;
  KvBlockGeometry(std::vector<KvBlockPlane> planes, std::uint32_t blocks)
      : planes_(std::move(planes)), blocks_(blocks) {
    if (planes_.empty() || blocks_ == 0)
      throw std::invalid_argument("KV block pool needs planes and blocks");
    offsets_.reserve(planes_.size());
    for (const auto& plane : planes_) {
      const auto rows = plane.rows_per_block;
      if (plane.layers == 0 || plane.row_bytes == 0 || rows == 0 ||
          (rows & (rows - 1)) != 0 || rows > kKvBlockTokens)
        throw std::invalid_argument(
            "KV block plane needs layers, row bytes and a power-of-two row "
            "count per block of at most " +
            std::to_string(kKvBlockTokens));
      offsets_.push_back(bytes_);
      bytes_ = Add(bytes_, Multiply(plane.layers, LayerPitch(plane)));
    }
  }

  [[nodiscard]] std::span<const KvBlockPlane> planes() const noexcept {
    return planes_;
  }
  [[nodiscard]] std::uint32_t blocks() const noexcept { return blocks_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] std::size_t plane_offset(std::size_t plane) const {
    return offsets_.at(plane);
  }
  /// Rows of one layer of `plane`: every block's.
  [[nodiscard]] std::size_t layer_rows(std::size_t plane) const {
    return static_cast<std::size_t>(blocks_) * planes_.at(plane).rows_per_block;
  }
  [[nodiscard]] std::size_t layer_pitch(std::size_t plane) const {
    return LayerPitch(planes_.at(plane));
  }
  /// Bytes of one block in one layer of `plane`.
  [[nodiscard]] std::size_t block_bytes(std::size_t plane) const {
    const auto& p = planes_.at(plane);
    return static_cast<std::size_t>(p.rows_per_block) * p.row_bytes;
  }
  /// Bytes one block takes across all planes and layers.
  [[nodiscard]] std::size_t bytes_per_block() const {
    std::size_t total = 0;
    for (std::size_t plane = 0; plane < planes_.size(); ++plane)
      total = Add(total, Multiply(planes_[plane].layers, block_bytes(plane)));
    return total;
  }
  /// Byte offset of row `row` (within the block) of `block` in `layer`.
  [[nodiscard]] std::size_t RowOffset(std::size_t plane, std::uint32_t layer,
                                      std::uint32_t block,
                                      std::uint32_t row = 0) const {
    const auto& p = planes_.at(plane);
    if (layer >= p.layers || block >= blocks_ || row >= p.rows_per_block)
      throw std::out_of_range("KV block row outside the pool");
    return offsets_[plane] + layer * layer_pitch(plane) +
           (static_cast<std::size_t>(block) * p.rows_per_block + row) *
               p.row_bytes;
  }

  /// Blocks covering `tokens` tokens.
  [[nodiscard]] static std::uint32_t BlocksFor(std::size_t tokens) {
    const std::size_t blocks =
        tokens / kKvBlockTokens + (tokens % kKvBlockTokens != 0 ? 1 : 0);
    if (blocks > std::numeric_limits<std::uint32_t>::max())
      throw std::length_error("KV block count overflows");
    return static_cast<std::uint32_t>(blocks);
  }

private:
  [[nodiscard]] std::size_t LayerPitch(const KvBlockPlane& plane) const {
    return Multiply(Multiply(blocks_, plane.rows_per_block), plane.row_bytes);
  }
  static std::size_t Multiply(std::size_t a, std::size_t b) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
      throw std::overflow_error("KV block pool size overflows");
    return a * b;
  }
  static std::size_t Add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
      throw std::overflow_error("KV block pool size overflows");
    return a + b;
  }

  std::vector<KvBlockPlane> planes_;
  std::vector<std::size_t> offsets_;
  std::uint32_t blocks_{0};
  std::size_t bytes_{0};
};

/// Which blocks of a pool are taken, with a reference count per block (a
/// block shared by several sequences or checkpoints is freed by its last
/// owner). Thread-safe. Allocation takes the lowest free indices, so a fresh
/// pool hands one sequence consecutive blocks.
class KvBlockAllocator {
public:
  explicit KvBlockAllocator(std::uint32_t blocks)
      : references_(blocks, 0), free_(blocks) {}

  [[nodiscard]] std::uint32_t blocks() const noexcept {
    return static_cast<std::uint32_t>(references_.size());
  }
  [[nodiscard]] std::uint32_t free_blocks() const {
    const std::lock_guard lock(mutex_);
    return free_;
  }
  [[nodiscard]] std::uint32_t references(std::uint32_t block) const {
    const std::lock_guard lock(mutex_);
    return references_.at(block);
  }
  /// `count` free blocks, lowest index first, each with one reference.
  /// Throws std::length_error, taking nothing, when fewer are free.
  [[nodiscard]] std::vector<std::uint32_t> Allocate(std::uint32_t count) {
    const std::lock_guard lock(mutex_);
    if (count > free_)
      throw std::length_error("KV block pool has " + std::to_string(free_) +
                              " free blocks, " + std::to_string(count) +
                              " requested");
    std::vector<std::uint32_t> taken;
    taken.reserve(count);
    for (std::uint32_t block = 0; taken.size() < count; ++block)
      if (references_[block] == 0) {
        references_[block] = 1;
        taken.push_back(block);
      }
    free_ -= count;
    return taken;
  }
  /// One more reference to each block; every block must be taken.
  void Retain(std::span<const std::uint32_t> blocks) {
    const std::lock_guard lock(mutex_);
    for (const auto block : blocks)
      if (block >= references_.size() || references_[block] == 0)
        throw std::logic_error("retaining a free KV block");
    for (const auto block : blocks)
      ++references_[block];
  }
  /// One reference fewer for each block; a block with none left is free.
  /// Unknown or free blocks are ignored (release cannot fail).
  void Release(std::span<const std::uint32_t> blocks) noexcept {
    const std::lock_guard lock(mutex_);
    for (const auto block : blocks)
      if (block < references_.size() && references_[block] != 0 &&
          --references_[block] == 0)
        ++free_;
  }

private:
  mutable std::mutex mutex_;
  std::vector<std::uint32_t> references_;
  std::uint32_t free_{0};
};

}  // namespace gufo::hip

#endif  // GUFO_CORE_HIP_KV_BLOCK_LAYOUT_HPP_
