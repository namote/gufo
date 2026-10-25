#include "src/cli/serve/block_disk_store.hpp"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "src/core/crypto/sha256.hpp"

namespace gufo::server {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{'G', 'U', 'F', 'O',
                                             'B', 'L', 'K', '1'};
constexpr std::uint32_t kVersion = 1;
// magic 8, version 4, token count 4, row bytes 8, key 32, parent 32,
// rows SHA-256 32; then tokens (u32 LE) and rows.
constexpr std::size_t kHeaderBytes = BlockDiskStore::kHeaderBytes;
constexpr std::size_t kKeyOffset = 24;
constexpr std::size_t kParentOffset = 56;
constexpr std::size_t kChecksumOffset = 88;
constexpr std::string_view kSuffix = ".blk";
constexpr std::string_view kTemporaryMarker = ".tmp.";

template <typename T>
void PutLe(std::span<std::uint8_t> bytes, std::size_t offset, T value) {
  for (std::size_t i = 0; i < sizeof(T); ++i)
    bytes[offset + i] = static_cast<std::uint8_t>(value >> (8 * i));
}

template <typename T>
T GetLe(std::span<const std::uint8_t> bytes, std::size_t offset) {
  T value = 0;
  for (std::size_t i = 0; i < sizeof(T); ++i)
    value |= static_cast<T>(bytes[offset + i]) << (8 * i);
  return value;
}

std::string Hex(const BlockKey& key) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string text(64, '0');
  for (std::size_t i = 0; i < key.size(); ++i) {
    text[2 * i] = kDigits[key[i] >> 4];
    text[2 * i + 1] = kDigits[key[i] & 15];
  }
  return text;
}

bool ParseHex(std::string_view text, BlockKey* key) {
  if (text.size() != 64)
    return false;
  const auto nibble = [](char c) -> int {
    if (c >= '0' && c <= '9')
      return c - '0';
    if (c >= 'a' && c <= 'f')
      return c - 'a' + 10;
    return -1;
  };
  for (std::size_t i = 0; i < key->size(); ++i) {
    const int high = nibble(text[2 * i]), low = nibble(text[2 * i + 1]);
    if (high < 0 || low < 0)
      return false;
    (*key)[i] = static_cast<std::uint8_t>(high << 4 | low);
  }
  return true;
}

std::vector<std::uint8_t> TokenBytes(std::span<const std::uint32_t> tokens) {
  std::vector<std::uint8_t> bytes(tokens.size() * 4);
  for (std::size_t i = 0; i < tokens.size(); ++i)
    PutLe<std::uint32_t>(bytes, 4 * i, tokens[i]);
  return bytes;
}

bool WriteAll(int fd, std::span<const std::uint8_t> bytes) {
  while (!bytes.empty()) {
    const ssize_t n = ::write(fd, bytes.data(), bytes.size());
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      return false;
    bytes = bytes.subspan(static_cast<std::size_t>(n));
  }
  return true;
}

bool ReadAll(int fd, std::span<std::uint8_t> bytes) {
  while (!bytes.empty()) {
    const ssize_t n = ::read(fd, bytes.data(), bytes.size());
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      return false;
    bytes = bytes.subspan(static_cast<std::size_t>(n));
  }
  return true;
}

class Descriptor {
public:
  explicit Descriptor(int fd = -1) noexcept : fd_(fd) {}
  ~Descriptor() {
    if (fd_ >= 0)
      (void)::close(fd_);
  }
  Descriptor(const Descriptor&) = delete;
  Descriptor& operator=(const Descriptor&) = delete;
  [[nodiscard]] int get() const noexcept { return fd_; }
  [[nodiscard]] explicit operator bool() const noexcept { return fd_ >= 0; }
  [[nodiscard]] bool Close() noexcept {
    const int fd = std::exchange(fd_, -1);
    return fd < 0 || ::close(fd) == 0;
  }

private:
  int fd_;
};

}  // namespace

BlockKey ChainRootKey(std::string_view compatibility_identity) {
  crypto::Sha256Hasher hasher;
  static constexpr std::string_view kDomain = "gufo-block-chain-v1";
  hasher.Update({reinterpret_cast<const std::uint8_t*>(kDomain.data()),
                 kDomain.size()});
  hasher.Update(
      {reinterpret_cast<const std::uint8_t*>(compatibility_identity.data()),
       compatibility_identity.size()});
  return hasher.Finish();
}

BlockKey ChainBlockKey(const BlockKey& parent,
                       std::span<const std::uint32_t> tokens,
                       std::span<const std::uint8_t> identity) {
  crypto::Sha256Hasher hasher;
  hasher.Update(parent);
  std::array<std::uint8_t, 8> counts{};
  PutLe<std::uint32_t>(counts, 0, static_cast<std::uint32_t>(tokens.size()));
  PutLe<std::uint32_t>(counts, 4, static_cast<std::uint32_t>(identity.size()));
  hasher.Update(counts);
  hasher.Update(TokenBytes(tokens));
  hasher.Update(identity);
  return hasher.Finish();
}

struct BlockDiskStore::Impl {
  struct Entry {
    BlockKey parent{};
    std::size_t bytes{0};
    std::uint64_t last_used{0};
    std::uint32_t children{0};
    std::uint32_t readers{0};
    bool writing{false};
  };

  Options options;
  int directory_fd{-1};
  mutable std::mutex mutex;
  std::map<BlockKey, Entry> entries;
  std::uint64_t clock{0};
  std::atomic<std::uint64_t> temporary_counter{0};
  BlockDiskStats stats;
  // Written since the last Flush: their pages leave the page cache once
  // synced (multi-GB conversations must not accumulate there).
  std::vector<std::string> unsynced;

  [[nodiscard]] std::size_t FileBytes(std::size_t tokens) const {
    return kHeaderBytes + 4 * tokens + options.block_bytes;
  }

  void Unlink(const std::string& name) const noexcept {
    (void)::unlinkat(directory_fd, name.c_str(), 0);
  }

  void AdjustParent(const Entry& entry, int delta) {
    const auto parent = entries.find(entry.parent);
    if (parent != entries.end() && !parent->second.writing)
      parent->second.children = static_cast<std::uint32_t>(
          static_cast<int>(parent->second.children) + delta);
  }

  // Under the lock: least recently used leaves first, never one being read
  // or written, nor `keep`.
  void EvictOverCapacity(const BlockKey& keep) {
    while (stats.stored_bytes > options.capacity_bytes) {
      auto victim = entries.end();
      for (auto it = entries.begin(); it != entries.end(); ++it) {
        const auto& entry = it->second;
        if (entry.writing || entry.readers != 0 || entry.children != 0 ||
            it->first == keep)
          continue;
        if (victim == entries.end() ||
            entry.last_used < victim->second.last_used)
          victim = it;
      }
      if (victim == entries.end())
        return;
      Unlink(Hex(victim->first) + std::string(kSuffix));
      AdjustParent(victim->second, -1);
      stats.stored_bytes -= victim->second.bytes;
      --stats.stored_blocks;
      ++stats.evicted_blocks;
      entries.erase(victim);
    }
  }

  void Scan() {
    DIR* directory = ::fdopendir(::dup(directory_fd));
    if (directory == nullptr)
      throw std::runtime_error("failed to list the block store directory");
    struct Found {
      BlockKey key;
      Entry entry;
      std::int64_t mtime;
    };
    std::vector<Found> found;
    while (const dirent* item = ::readdir(directory)) {
      const std::string name = item->d_name;
      if (name == "." || name == "..")
        continue;
      if (name.find(kTemporaryMarker) != std::string::npos) {
        Unlink(name);  // an interrupted write
        continue;
      }
      BlockKey key{};
      if (!name.ends_with(kSuffix) ||
          !ParseHex(std::string_view(name).substr(0, name.size() -
                                                         kSuffix.size()),
                    &key))
        continue;  // not ours
      Descriptor file(::openat(directory_fd, name.c_str(),
                               O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
      std::array<std::uint8_t, kHeaderBytes> header{};
      struct stat status {};
      const bool readable = file && ::fstat(file.get(), &status) == 0 &&
                            ReadAll(file.get(), header);
      const bool valid =
          readable && std::equal(kMagic.begin(), kMagic.end(), header.begin()) &&
          GetLe<std::uint32_t>(header, 8) == kVersion &&
          GetLe<std::uint64_t>(header, 16) == options.block_bytes &&
          std::equal(key.begin(), key.end(), header.begin() + kKeyOffset) &&
          static_cast<std::size_t>(status.st_size) ==
              FileBytes(GetLe<std::uint32_t>(header, 12));
      if (!valid) {
        Unlink(name);
        ++stats.corrupt_blocks;
        continue;
      }
      Found item_found{.key = key,
                       .entry = {.bytes = static_cast<std::size_t>(
                                     status.st_size)},
                       .mtime = static_cast<std::int64_t>(
                                    status.st_mtim.tv_sec) *
                                    1000000000 +
                                status.st_mtim.tv_nsec};
      std::copy_n(header.begin() + kParentOffset, 32,
                  item_found.entry.parent.begin());
      found.push_back(item_found);
    }
    ::closedir(directory);
    std::ranges::sort(found, {}, &Found::mtime);
    for (auto& item : found) {
      item.entry.last_used = ++clock;
      stats.stored_bytes += item.entry.bytes;
      ++stats.stored_blocks;
      entries.emplace(item.key, item.entry);
    }
    for (const auto& [key, entry] : entries)
      AdjustParent(entry, +1);
  }
};

BlockDiskStore::BlockDiskStore(Options options)
    : impl_(std::make_unique<Impl>()) {
  impl_->options = std::move(options);
  if (impl_->options.block_bytes == 0)
    throw std::invalid_argument("block store needs a nonzero block size");
  if (impl_->options.capacity_bytes < impl_->FileBytes(0))
    throw std::invalid_argument(
        "block store capacity " +
        std::to_string(impl_->options.capacity_bytes) +
        " is below one block file of " +
        std::to_string(impl_->FileBytes(0)) + " bytes");
  std::error_code error;
  std::filesystem::create_directories(impl_->options.directory, error);
  impl_->directory_fd =
      ::open(impl_->options.directory.c_str(),
             O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (impl_->directory_fd < 0)
    throw std::runtime_error("failed to open the block store directory " +
                             impl_->options.directory.string());
  impl_->Scan();
  const std::lock_guard lock(impl_->mutex);
  impl_->EvictOverCapacity({});
}

BlockDiskStore::~BlockDiskStore() {
  if (impl_ != nullptr && impl_->directory_fd >= 0) {
    (void)::fsync(impl_->directory_fd);
    (void)::close(impl_->directory_fd);
  }
}

std::size_t BlockDiskStore::block_bytes() const noexcept {
  return impl_->options.block_bytes;
}

bool BlockDiskStore::Contains(const BlockKey& key) const {
  const std::lock_guard lock(impl_->mutex);
  const auto it = impl_->entries.find(key);
  return it != impl_->entries.end() && !it->second.writing;
}

bool BlockDiskStore::Put(const BlockKey& key, const BlockKey& parent,
                         std::span<const std::uint32_t> tokens,
                         const std::function<void(const Sink&)>& produce) {
  {
    const std::lock_guard lock(impl_->mutex);
    if (impl_->entries.contains(key)) {
      ++impl_->stats.skipped_blocks;
      return false;
    }
    impl_->entries.emplace(key, Impl::Entry{.parent = parent, .writing = true});
  }
  const std::string final_name = Hex(key) + std::string(kSuffix);
  const std::string temporary_name =
      Hex(key) + std::string(kTemporaryMarker) +
      std::to_string(impl_->temporary_counter.fetch_add(1));
  const auto abandon = [&] {
    impl_->Unlink(temporary_name);
    const std::lock_guard lock(impl_->mutex);
    impl_->entries.erase(key);
  };
  try {
    Descriptor file(::openat(impl_->directory_fd, temporary_name.c_str(),
                             O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC |
                                 O_NOFOLLOW,
                             0644));
    if (!file)
      throw std::runtime_error("failed to create block file");
    std::array<std::uint8_t, kHeaderBytes> header{};
    std::copy(kMagic.begin(), kMagic.end(), header.begin());
    PutLe<std::uint32_t>(header, 8, kVersion);
    PutLe<std::uint32_t>(header, 12, static_cast<std::uint32_t>(tokens.size()));
    PutLe<std::uint64_t>(header, 16, impl_->options.block_bytes);
    std::copy(key.begin(), key.end(), header.begin() + kKeyOffset);
    std::copy(parent.begin(), parent.end(), header.begin() + kParentOffset);
    if (!WriteAll(file.get(), header) ||
        !WriteAll(file.get(), TokenBytes(tokens)))
      throw std::runtime_error("block header write failed");
    crypto::Sha256Hasher hasher;
    std::size_t written = 0;
    produce([&](std::span<const std::uint8_t> bytes) {
      if (bytes.size() > impl_->options.block_bytes - written)
        throw std::runtime_error("block rows overflow the block size");
      if (!WriteAll(file.get(), bytes))
        throw std::runtime_error("block rows write failed");
      hasher.Update(bytes);
      written += bytes.size();
    });
    if (written != impl_->options.block_bytes)
      throw std::runtime_error("block rows are shorter than the block size");
    const auto digest = hasher.Finish();
    // No fsync per block: Flush syncs the file system once before a record
    // that references these blocks; a torn block fails its checksum.
    if (::pwrite(file.get(), digest.data(), digest.size(), kChecksumOffset) !=
        static_cast<ssize_t>(digest.size()))
      throw std::runtime_error("block checksum write failed");
    if (!file.Close() ||
        ::renameat(impl_->directory_fd, temporary_name.c_str(),
                   impl_->directory_fd, final_name.c_str()) != 0)
      throw std::runtime_error("block file commit failed");
  } catch (...) {
    abandon();
    throw;
  }
  const std::lock_guard lock(impl_->mutex);
  auto& entry = impl_->entries.at(key);
  entry.writing = false;
  entry.bytes = impl_->FileBytes(tokens.size());
  entry.last_used = ++impl_->clock;
  // Children written before this block were not counted against it.
  for (const auto& [other_key, other] : impl_->entries)
    if (!other.writing && other.parent == key)
      ++entry.children;
  impl_->AdjustParent(entry, +1);
  impl_->stats.stored_bytes += entry.bytes;
  ++impl_->stats.stored_blocks;
  ++impl_->stats.written_blocks;
  impl_->stats.written_bytes += entry.bytes;
  impl_->unsynced.push_back(final_name);
  impl_->EvictOverCapacity(key);
  return true;
}

void BlockDiskStore::Get(const BlockKey& key, const BlockKey& parent,
                         std::span<const std::uint32_t> tokens,
                         const std::function<void(const Source&)>& consume) {
  const std::string name = Hex(key) + std::string(kSuffix);
  {
    const std::lock_guard lock(impl_->mutex);
    const auto it = impl_->entries.find(key);
    if (it == impl_->entries.end() || it->second.writing)
      throw BlockMissing("block " + Hex(key) + " is not stored");
    ++it->second.readers;
  }
  bool corrupt = false;
  const auto finish = [&] {
    const std::lock_guard lock(impl_->mutex);
    const auto it = impl_->entries.find(key);
    if (it == impl_->entries.end())
      return;
    --it->second.readers;
    if (corrupt) {
      impl_->Unlink(name);
      impl_->AdjustParent(it->second, -1);
      impl_->stats.stored_bytes -= it->second.bytes;
      --impl_->stats.stored_blocks;
      ++impl_->stats.corrupt_blocks;
      impl_->entries.erase(it);
    } else {
      it->second.last_used = ++impl_->clock;
      ++impl_->stats.read_blocks;
      impl_->stats.read_bytes += it->second.bytes;
    }
  };
  try {
    Descriptor file(
        ::openat(impl_->directory_fd, name.c_str(),
                 O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    std::array<std::uint8_t, kHeaderBytes> header{};
    std::vector<std::uint8_t> stored_tokens(tokens.size() * 4);
    if (!file || !ReadAll(file.get(), header) ||
        GetLe<std::uint32_t>(header, 12) != tokens.size() ||
        !ReadAll(file.get(), stored_tokens))
      corrupt = true;
    else if (!std::equal(parent.begin(), parent.end(),
                         header.begin() + kParentOffset) ||
             stored_tokens != TokenBytes(tokens))
      corrupt = true;
    if (corrupt)
      throw BlockMissing("block " + Hex(key) + " header does not match");
    crypto::Sha256Hasher hasher;
    std::size_t read = 0;
    consume([&](std::span<std::uint8_t> bytes) {
      if (bytes.size() > impl_->options.block_bytes - read ||
          !ReadAll(file.get(), bytes)) {
        corrupt = true;
        throw BlockMissing("block " + Hex(key) + " rows are short");
      }
      hasher.Update(bytes);
      read += bytes.size();
    });
    const auto digest = hasher.Finish();
    if (read != impl_->options.block_bytes ||
        !std::equal(digest.begin(), digest.end(),
                    header.begin() + kChecksumOffset)) {
      corrupt = true;
      throw BlockMissing("block " + Hex(key) + " fails its checksum");
    }
    (void)::posix_fadvise(file.get(), 0, 0, POSIX_FADV_DONTNEED);
  } catch (...) {
    finish();
    throw;
  }
  finish();
}

std::vector<std::uint8_t> BlockDiskStore::Read(
    const BlockKey& key, const BlockKey& parent,
    std::span<const std::uint32_t> tokens) {
  std::vector<std::uint8_t> rows(impl_->options.block_bytes);
  Get(key, parent, tokens, [&](const Source& source) { source(rows); });
  return rows;
}

void BlockDiskStore::Touch(std::span<const BlockKey> keys) {
  const std::lock_guard lock(impl_->mutex);
  for (const auto& key : keys) {
    const auto it = impl_->entries.find(key);
    if (it == impl_->entries.end() || it->second.writing)
      continue;
    it->second.last_used = ++impl_->clock;
    // The order survives a restart through the file times.
    (void)::utimensat(impl_->directory_fd,
                      (Hex(key) + std::string(kSuffix)).c_str(), nullptr, 0);
  }
}

void BlockDiskStore::Flush() {
  // One commit for every block written since the last one, data and names.
  if (::syncfs(impl_->directory_fd) != 0)
    throw std::runtime_error("block store sync failed");
  std::vector<std::string> synced;
  {
    const std::lock_guard lock(impl_->mutex);
    synced.swap(impl_->unsynced);
  }
  // Clean now: drop their pages (a file evicted meanwhile is simply gone).
  for (const auto& name : synced) {
    Descriptor file(::openat(impl_->directory_fd, name.c_str(),
                             O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (file)
      (void)::posix_fadvise(file.get(), 0, 0, POSIX_FADV_DONTNEED);
  }
}

BlockDiskStats BlockDiskStore::Stats() const {
  const std::lock_guard lock(impl_->mutex);
  return impl_->stats;
}

ResidentBlockIndex::ResidentBlockIndex(References references)
    : references_(std::move(references)) {
  if (!references_.retain || !references_.release || !references_.count)
    throw std::invalid_argument("a resident block index needs reference counting");
}

ResidentBlockIndex::~ResidentBlockIndex() {
  for (const auto& [key, entry] : entries_)
    references_.release(entry.block);
}

std::optional<std::uint32_t> ResidentBlockIndex::Acquire(const BlockKey& key) {
  const std::lock_guard lock(mutex_);
  const auto found = entries_.find(key);
  if (found == entries_.end())
    return std::nullopt;
  // Under the lock, so ReleaseUnshared cannot free it first.
  references_.retain(found->second.block);
  found->second.last_used = ++clock_;
  return found->second.block;
}

bool ResidentBlockIndex::Insert(const BlockKey& key, std::uint32_t block) {
  const std::lock_guard lock(mutex_);
  if (entries_.contains(key))
    return false;
  references_.retain(block);
  entries_.emplace(key, Entry{.block = block, .last_used = ++clock_});
  ++stats_.inserted;
  return true;
}

std::size_t ResidentBlockIndex::ReleaseUnshared(std::size_t blocks) {
  if (blocks == 0)
    return 0;
  const std::lock_guard lock(mutex_);
  std::vector<std::pair<std::uint64_t, BlockKey>> unshared;
  for (const auto& [key, entry] : entries_)
    if (references_.count(entry.block) == 1)
      unshared.emplace_back(entry.last_used, key);
  std::ranges::sort(unshared);
  std::size_t released = 0;
  for (const auto& [last_used, key] : unshared) {
    if (released == blocks)
      break;
    const auto found = entries_.find(key);
    references_.release(found->second.block);
    entries_.erase(found);
    ++released;
  }
  stats_.released += released;
  return released;
}

void ResidentBlockIndex::CountDeduplicated() noexcept {
  const std::lock_guard lock(mutex_);
  ++stats_.deduplicated;
}

void ResidentBlockIndex::CountRestored(std::size_t blocks) noexcept {
  const std::lock_guard lock(mutex_);
  stats_.restored += blocks;
}

ResidentBlockIndex::Stats ResidentBlockIndex::stats() const {
  const std::lock_guard lock(mutex_);
  auto stats = stats_;
  stats.blocks = entries_.size();
  for (const auto& [key, entry] : entries_)
    if (references_.count(entry.block) == 1)
      ++stats.unshared_blocks;
  return stats;
}

}  // namespace gufo::server
