#ifndef GUFO_SERVER_BLOCK_DISK_STORE_HPP_
#define GUFO_SERVER_BLOCK_DISK_STORE_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>
#include <stdexcept>
#include <string_view>

namespace gufo::server {

/// Content address of one block of model rows: a hash chain over the tokens
/// of every block before it and the block itself, from a root that names the
/// model and its row layout. Equal keys mean equal rows.
using BlockKey = std::array<std::uint8_t, 32>;

/// The chain root: no rows. `compatibility_identity` names everything the
/// rows depend on besides the tokens (weights, row dtype and layout, block
/// size, position policy).
[[nodiscard]] BlockKey ChainRootKey(std::string_view compatibility_identity);

/// The key of the block after `parent` holding `tokens`. `identity` is the
/// input identity of the prefix ending with this block (image digests), so a
/// block whose rows depend on an image differs from one with the same
/// placeholder tokens.
[[nodiscard]] BlockKey ChainBlockKey(const BlockKey& parent,
                                     std::span<const std::uint32_t> tokens,
                                     std::span<const std::uint8_t> identity);

/// A block the store does not hold, or held corrupt (then removed).
class BlockMissing : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

struct BlockDiskStats {
  std::size_t stored_blocks{0};
  std::size_t stored_bytes{0};
  std::size_t written_blocks{0};
  std::size_t written_bytes{0};
  /// Put found the block stored already: nothing written.
  std::size_t skipped_blocks{0};
  std::size_t read_blocks{0};
  std::size_t read_bytes{0};
  std::size_t evicted_blocks{0};
  std::size_t corrupt_blocks{0};
};

/// Durable store of fixed-size blocks of model rows, one file per block,
/// addressed by chain keys. Each file holds its key, parent key and tokens,
/// checked on every read, and a SHA-256 of its rows. Past the byte capacity
/// the least recently used leaf block goes first: a block another stored
/// block extends stays while that one does. The index is rebuilt from the
/// file headers at construction. The store knows nothing of models.
class BlockDiskStore {
public:
  struct Options {
    std::filesystem::path directory;
    std::size_t capacity_bytes{0};
    /// Bytes of rows in every block.
    std::size_t block_bytes{0};
  };

  using Sink = std::function<void(std::span<const std::uint8_t>)>;
  using Source = std::function<void(std::span<std::uint8_t>)>;
  /// Fixed file header; the tokens (u32 each) and the rows follow.
  static constexpr std::size_t kHeaderBytes = 120;

  explicit BlockDiskStore(Options options);
  ~BlockDiskStore();

  BlockDiskStore(const BlockDiskStore&) = delete;
  BlockDiskStore& operator=(const BlockDiskStore&) = delete;

  [[nodiscard]] std::size_t block_bytes() const noexcept;
  [[nodiscard]] bool Contains(const BlockKey& key) const;

  /// Writes block `key` unless it is stored or being written: `produce` must
  /// pass exactly block_bytes() to its sink, in order. True when written.
  /// Throws on I/O or production failure; nothing is stored then.
  bool Put(const BlockKey& key, const BlockKey& parent,
           std::span<const std::uint32_t> tokens,
           const std::function<void(const Sink&)>& produce);

  /// Reads block `key`: `consume` pulls exactly block_bytes() through the
  /// source, in order. Its parent and tokens must match. Throws BlockMissing
  /// when the block is absent, its header differs, or its rows fail the
  /// checksum after `consume` returns; a corrupt file is removed. The block
  /// is not evicted while it is read.
  void Get(const BlockKey& key, const BlockKey& parent,
           std::span<const std::uint32_t> tokens,
           const std::function<void(const Source&)>& consume);

  /// Get into memory: the block's rows, checked before they are returned.
  /// Safe to call from several threads at once.
  [[nodiscard]] std::vector<std::uint8_t> Read(
      const BlockKey& key, const BlockKey& parent,
      std::span<const std::uint32_t> tokens);

  /// Marks blocks used now, so a checkpoint that references them keeps them
  /// over older ones.
  void Touch(std::span<const BlockKey> keys);

  /// Makes the blocks written so far durable, rows and names (one file
  /// system sync). A record that references blocks calls this before it is
  /// written.
  void Flush();

  [[nodiscard]] BlockDiskStats Stats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Device blocks of model rows by the chain key of the rows they hold, so a
/// restore or a finished prefill whose block is resident already shares it
/// instead of reading the file or keeping a copy. Each entry holds one
/// reference to its block; an entry whose block nobody else references stays
/// until ReleaseUnshared needs the room. Like the store, it knows nothing of
/// models: the owner of the blocks supplies their reference counting.
class ResidentBlockIndex {
public:
  struct References {
    std::function<void(std::uint32_t)> retain;
    std::function<void(std::uint32_t)> release;
    std::function<std::size_t(std::uint32_t)> count;
  };
  struct Stats {
    std::size_t blocks{0};
    /// Indexed blocks nobody but the index references.
    std::size_t unshared_blocks{0};
    /// Block reads a resident block replaced (CountRestored).
    std::size_t restored{0};
    std::size_t inserted{0};
    /// Duplicate blocks replaced by an indexed one.
    std::size_t deduplicated{0};
    std::size_t released{0};
  };

  explicit ResidentBlockIndex(References references);
  /// Drops every entry's reference.
  ~ResidentBlockIndex();
  ResidentBlockIndex(const ResidentBlockIndex&) = delete;
  ResidentBlockIndex& operator=(const ResidentBlockIndex&) = delete;

  /// The block holding `key`, with one new reference for the caller, who
  /// releases it; nullopt when no block is indexed under `key`.
  [[nodiscard]] std::optional<std::uint32_t> Acquire(const BlockKey& key);
  /// Indexes `block` under `key`, taking one reference. False, with nothing
  /// changed, when `key` is indexed already.
  bool Insert(const BlockKey& key, std::uint32_t block);
  /// Releases the least recently used entries that only the index
  /// references until `blocks` are freed or none is left; returns how many.
  std::size_t ReleaseUnshared(std::size_t blocks);
  void CountDeduplicated() noexcept;
  void CountRestored(std::size_t blocks) noexcept;
  [[nodiscard]] Stats stats() const;

private:
  struct Entry {
    std::uint32_t block{0};
    std::uint64_t last_used{0};
  };
  References references_;
  mutable std::mutex mutex_;
  std::map<BlockKey, Entry> entries_;
  std::uint64_t clock_{0};
  Stats stats_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_BLOCK_DISK_STORE_HPP_
