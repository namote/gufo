#ifndef GUFO_CORE_HIP_KV_BLOCK_POOL_HPP_
#define GUFO_CORE_HIP_KV_BLOCK_POOL_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "src/core/hip/kv_block_layout.hpp"
#include "src/core/hip/kv_block_table.hpp"

namespace gufo::hip {

/// One device allocation holding append-only rows of several planes, carved
/// into kKvBlockTokens-token blocks (KvBlockGeometry). Sequences own blocks
/// through a block table. The pool knows nothing about the model: a caller
/// describes its rows as planes.
class KvBlockPool {
public:
  KvBlockPool(std::vector<KvBlockPlane> planes, std::uint32_t blocks);
  ~KvBlockPool();
  KvBlockPool(const KvBlockPool&) = delete;
  KvBlockPool& operator=(const KvBlockPool&) = delete;

  [[nodiscard]] static std::uint32_t BlocksFor(std::size_t tokens) {
    return KvBlockGeometry::BlocksFor(tokens);
  }
  [[nodiscard]] const KvBlockGeometry& geometry() const noexcept {
    return geometry_;
  }
  [[nodiscard]] std::uint32_t blocks() const noexcept {
    return geometry_.blocks();
  }
  [[nodiscard]] void* data() const noexcept { return data_; }
  [[nodiscard]] std::byte* plane_data(std::size_t plane) const {
    return static_cast<std::byte*>(data_) + geometry_.plane_offset(plane);
  }

  [[nodiscard]] std::vector<std::uint32_t> Allocate(
      std::uint32_t count, KvBlockReservation* reservation = nullptr) {
    return allocator_.Allocate(count, reservation);
  }
  [[nodiscard]] KvBlockReservation TryReserve(std::uint32_t count) {
    return allocator_.TryReserve(count);
  }
  void Retain(std::span<const std::uint32_t> blocks) {
    allocator_.Retain(blocks);
  }
  void Release(std::span<const std::uint32_t> blocks) noexcept {
    allocator_.Release(blocks);
  }
  void Release(std::span<const std::uint32_t> blocks,
               KvBlockReservation& reservation) noexcept {
    allocator_.Release(blocks, reservation);
  }
  [[nodiscard]] std::uint32_t references(std::uint32_t block) const {
    return allocator_.references(block);
  }
  /// Blocks neither taken nor reserved.
  [[nodiscard]] std::uint32_t free_blocks() const {
    return allocator_.free_blocks();
  }
  [[nodiscard]] std::uint32_t reserved_blocks() const {
    return allocator_.reserved_blocks();
  }

private:
  KvBlockGeometry geometry_;
  KvBlockAllocator allocator_;
  void* data_{nullptr};
};

}  // namespace gufo::hip

#endif  // GUFO_CORE_HIP_KV_BLOCK_POOL_HPP_
