#include "src/core/hip/kv_block_pool.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace gufo::hip {
namespace {
std::size_t Multiply(std::size_t a, std::size_t b) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::overflow_error("KV block pool size overflows");
  return a * b;
}
}  // namespace

std::uint32_t KvBlockPool::BlocksFor(std::size_t tokens) {
  const std::size_t blocks = (tokens + kKvBlockTokens - 1) / kKvBlockTokens;
  if (blocks > std::numeric_limits<std::uint32_t>::max())
    throw std::length_error("KV block count overflows");
  return static_cast<std::uint32_t>(blocks);
}

std::size_t KvBlockPool::DeviceBytes(std::uint32_t layers,
                                     std::size_t row_bytes,
                                     std::uint32_t blocks) {
  return Multiply(
      Multiply(Multiply(Multiply(layers, blocks), kKvBlockTokens), row_bytes),
      2);
}

KvBlockPool::KvBlockPool(std::uint32_t layers, std::size_t row_bytes,
                         std::uint32_t blocks)
    : layers_(layers),
      row_bytes_(row_bytes),
      blocks_(blocks),
      references_(blocks, 0),
      free_(blocks) {
  if (layers == 0 || row_bytes == 0 || blocks == 0)
    throw std::invalid_argument("KV block pool must not be empty");
  const std::size_t bytes = DeviceBytes(layers, row_bytes, blocks);
  const hipError_t status = hipMalloc(&data_, bytes);
  if (status != hipSuccess) {
    data_ = nullptr;
    throw std::runtime_error("failed to allocate a KV block pool of " +
                             std::to_string(bytes) +
                             " bytes: " + hipGetErrorString(status));
  }
}

KvBlockPool::~KvBlockPool() {
  if (data_ != nullptr)
    (void)hipFree(data_);
}

std::size_t KvBlockPool::plane_bytes() const noexcept {
  return static_cast<std::size_t>(layers_) * layer_tokens() * row_bytes_;
}

std::uint32_t KvBlockPool::free_blocks() const {
  const std::lock_guard lock(mutex_);
  return free_ - reserved_;
}

std::uint32_t KvBlockPool::reserved_blocks() const {
  const std::lock_guard lock(mutex_);
  return reserved_;
}

std::vector<std::uint32_t> KvBlockPool::TakeLocked(std::uint32_t count) {
  std::vector<std::uint32_t> taken;
  taken.reserve(count);
  for (std::uint32_t block = 0; taken.size() < count; ++block) {
    if (references_[block] == 0) {
      references_[block] = 1;
      taken.push_back(block);
    }
  }
  free_ -= count;
  return taken;
}

std::vector<std::uint32_t> KvBlockPool::Allocate(std::uint32_t count) {
  const std::lock_guard lock(mutex_);
  if (count > free_ - reserved_)
    throw std::length_error("KV block pool has " +
                            std::to_string(free_ - reserved_) +
                            " free blocks, " + std::to_string(count) +
                            " requested");
  return TakeLocked(count);
}

std::vector<std::uint32_t> KvBlockPool::Allocate(
    std::uint32_t count, KvBlockReservation& reservation) {
  if (reservation.pool_ != nullptr && reservation.pool_ != this)
    throw std::invalid_argument("KV block reservation names another pool");
  const std::lock_guard lock(mutex_);
  const std::uint32_t held = reservation.pool_ == this ? reservation.blocks_ : 0;
  const std::uint32_t from_reserve = std::min(count, held);
  if (count - from_reserve > free_ - reserved_)
    throw std::length_error(
        "KV block pool has " + std::to_string(free_ - reserved_) +
        " free blocks and " + std::to_string(held) + " reserved, " +
        std::to_string(count) + " requested");
  reservation.blocks_ -= from_reserve;
  reserved_ -= from_reserve;
  return TakeLocked(count);
}

void KvBlockPool::Retain(std::span<const std::uint32_t> blocks) {
  const std::lock_guard lock(mutex_);
  for (const auto block : blocks)
    if (block >= blocks_ || references_[block] == 0)
      throw std::logic_error("KV block retained without a reference");
  for (const auto block : blocks)
    ++references_[block];
}

void KvBlockPool::Release(std::span<const std::uint32_t> blocks) noexcept {
  const std::lock_guard lock(mutex_);
  for (const auto block : blocks) {
    if (block < blocks_ && references_[block] != 0 &&
        --references_[block] == 0)
      ++free_;
  }
}

std::uint32_t KvBlockPool::references(std::uint32_t block) const {
  const std::lock_guard lock(mutex_);
  if (block >= blocks_)
    throw std::out_of_range("KV block outside the pool");
  return references_[block];
}

KvBlockReservation KvBlockPool::TryReserve(std::uint32_t count) {
  const std::lock_guard lock(mutex_);
  if (count > free_ - reserved_)
    return {};
  reserved_ += count;
  return {this, count};
}

void KvBlockPool::Unreserve(std::uint32_t count) noexcept {
  const std::lock_guard lock(mutex_);
  reserved_ -= count;
}

KvBlockReservation::KvBlockReservation(KvBlockReservation&& other) noexcept
    : pool_(std::exchange(other.pool_, nullptr)),
      blocks_(std::exchange(other.blocks_, 0)) {}

KvBlockReservation& KvBlockReservation::operator=(
    KvBlockReservation&& other) noexcept {
  if (this != &other) {
    Reset();
    pool_ = std::exchange(other.pool_, nullptr);
    blocks_ = std::exchange(other.blocks_, 0);
  }
  return *this;
}

void KvBlockReservation::Reset() noexcept {
  if (pool_ != nullptr && blocks_ != 0)
    pool_->Unreserve(blocks_);
  pool_ = nullptr;
  blocks_ = 0;
}

MutableDeviceRegion KvBlockPool::BlockRegion(std::uint32_t plane,
                                             std::uint32_t block,
                                             std::uint32_t first_row,
                                             std::uint32_t rows) const {
  if (plane > 1 || block >= blocks_ || first_row > kKvBlockTokens ||
      rows > kKvBlockTokens - first_row)
    throw std::out_of_range("KV block region is outside the pool");
  auto* base = static_cast<std::uint8_t*>(data_) + plane * plane_bytes();
  const std::size_t pitch =
      static_cast<std::size_t>(layer_tokens()) * row_bytes_;
  return {base + (static_cast<std::size_t>(block) * kKvBlockTokens +
                  first_row) *
                     row_bytes_,
          pitch, rows * row_bytes_, layers_};
}

std::vector<MutableDeviceRegion> KvBlockPool::PlaneRegions(
    std::uint32_t plane, std::span<const std::uint32_t> table,
    std::size_t tokens) const {
  if (plane > 1)
    throw std::invalid_argument("KV block pool has two planes");
  if (BlocksFor(tokens) > table.size())
    throw std::length_error("KV block table is shorter than the rows");
  struct Run {
    std::size_t first_row, rows;
  };
  std::vector<Run> runs;
  for (std::size_t logical = 0; logical * kKvBlockTokens < tokens;
       ++logical) {
    if (table[logical] >= blocks_)
      throw std::out_of_range("KV block table names a block outside the pool");
    const std::size_t first_row =
        static_cast<std::size_t>(table[logical]) * kKvBlockTokens;
    const std::size_t rows = std::min<std::size_t>(
        kKvBlockTokens, tokens - logical * kKvBlockTokens);
    if (!runs.empty() &&
        runs.back().first_row + runs.back().rows == first_row)
      runs.back().rows += rows;
    else
      runs.push_back({first_row, rows});
  }
  auto* base = static_cast<std::uint8_t*>(data_) + plane * plane_bytes();
  const std::size_t pitch = static_cast<std::size_t>(layer_tokens()) * row_bytes_;
  std::vector<MutableDeviceRegion> regions;
  if (runs.size() == 1) {
    regions.push_back({base + runs[0].first_row * row_bytes_, pitch,
                       runs[0].rows * row_bytes_, layers_});
    return regions;
  }
  regions.reserve(runs.size() * layers_);
  for (std::uint32_t layer = 0; layer < layers_; ++layer)
    for (const auto& run : runs)
      regions.push_back({base + layer * pitch + run.first_row * row_bytes_,
                         run.rows * row_bytes_, run.rows * row_bytes_, 1});
  return regions;
}

}  // namespace gufo::hip
