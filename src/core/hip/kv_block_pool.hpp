#ifndef GUFO_CORE_HIP_KV_BLOCK_POOL_HPP_
#define GUFO_CORE_HIP_KV_BLOCK_POOL_HPP_

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <vector>

#include "src/core/hip/bounce_stream.hpp"
#include "src/core/hip/kv_block_table.hpp"

namespace gufo::hip {

class KvBlockReservation;

/// One device allocation holding the K and V planes of append-only attention
/// rows, carved into kKvBlockTokens-row blocks. Each plane is
/// [layer][block][kKvBlockTokens][row_bytes]; V follows K. Sequences own
/// blocks through a block table. The pool knows nothing about the model: a
/// caller describes its rows by layer count and bytes per token row.
class KvBlockPool {
public:
  KvBlockPool(std::uint32_t layers, std::size_t row_bytes,
              std::uint32_t blocks);
  ~KvBlockPool();
  KvBlockPool(const KvBlockPool&) = delete;
  KvBlockPool& operator=(const KvBlockPool&) = delete;

  /// Blocks needed for `tokens` rows.
  [[nodiscard]] static std::uint32_t BlocksFor(std::size_t tokens);
  [[nodiscard]] static std::size_t DeviceBytes(std::uint32_t layers,
                                               std::size_t row_bytes,
                                               std::uint32_t blocks);

  /// Takes `count` free blocks, lowest index first, each with one reference.
  /// Blocks held back by reservations are not free. Throws std::length_error
  /// when fewer are free; nothing is taken then.
  [[nodiscard]] std::vector<std::uint32_t> Allocate(std::uint32_t count);
  /// As Allocate, drawing first on `reservation`'s remaining blocks.
  [[nodiscard]] std::vector<std::uint32_t> Allocate(
      std::uint32_t count, KvBlockReservation& reservation);
  /// Adds one reference to each block (a sharer of immutable rows).
  void Retain(std::span<const std::uint32_t> blocks);
  /// Drops one reference from each block; a block with none left is free.
  void Release(std::span<const std::uint32_t> blocks) noexcept;
  [[nodiscard]] std::uint32_t references(std::uint32_t block) const;

  /// Holds back `count` free blocks for a later Allocate; empty (and nothing
  /// held) when fewer are free.
  [[nodiscard]] KvBlockReservation TryReserve(std::uint32_t count);

  [[nodiscard]] void* data() const noexcept { return data_; }
  [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }
  [[nodiscard]] std::size_t row_bytes() const noexcept { return row_bytes_; }
  [[nodiscard]] std::uint32_t blocks() const noexcept { return blocks_; }
  [[nodiscard]] std::uint32_t layer_tokens() const noexcept {
    return blocks_ * kKvBlockTokens;
  }
  [[nodiscard]] std::size_t plane_bytes() const noexcept;
  /// Blocks neither referenced nor reserved.
  [[nodiscard]] std::uint32_t free_blocks() const;
  [[nodiscard]] std::uint32_t reserved_blocks() const;

  /// The rows [0, tokens) of one plane of a sequence, in serialized order
  /// (layer-major, tokens ascending). Consecutive pool blocks merge, so a
  /// sequence on one run of blocks is a single region of `layers` rows.
  /// Rows [first_row, first_row + rows) of one block of one plane, all
  /// layers: `layers` rows of `rows` token rows each.
  [[nodiscard]] MutableDeviceRegion BlockRegion(std::uint32_t plane,
                                                std::uint32_t block,
                                                std::uint32_t first_row,
                                                std::uint32_t rows) const;

  [[nodiscard]] std::vector<MutableDeviceRegion> PlaneRegions(
      std::uint32_t plane, std::span<const std::uint32_t> table,
      std::size_t tokens) const;

private:
  friend class KvBlockReservation;
  [[nodiscard]] std::vector<std::uint32_t> TakeLocked(std::uint32_t count);
  void Unreserve(std::uint32_t count) noexcept;

  std::uint32_t layers_;
  std::size_t row_bytes_;
  std::uint32_t blocks_;
  void* data_{nullptr};
  mutable std::mutex mutex_;
  std::vector<std::uint32_t> references_;
  std::uint32_t free_{0};
  std::uint32_t reserved_{0};
};

/// Blocks a pool holds back for one owner, returned on destruction.
class KvBlockReservation {
public:
  KvBlockReservation() = default;
  ~KvBlockReservation() { Reset(); }
  KvBlockReservation(KvBlockReservation&& other) noexcept;
  KvBlockReservation& operator=(KvBlockReservation&& other) noexcept;
  KvBlockReservation(const KvBlockReservation&) = delete;
  KvBlockReservation& operator=(const KvBlockReservation&) = delete;

  [[nodiscard]] explicit operator bool() const noexcept {
    return pool_ != nullptr;
  }
  [[nodiscard]] std::uint32_t blocks() const noexcept { return blocks_; }
  /// Returns the blocks not yet allocated to the pool.
  void Reset() noexcept;

private:
  friend class KvBlockPool;
  KvBlockReservation(KvBlockPool* pool, std::uint32_t blocks) noexcept
      : pool_(pool), blocks_(blocks) {}
  KvBlockPool* pool_{nullptr};
  std::uint32_t blocks_{0};
};

}  // namespace gufo::hip

#endif  // GUFO_CORE_HIP_KV_BLOCK_POOL_HPP_
