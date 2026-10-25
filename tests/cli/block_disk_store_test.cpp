#include "src/cli/serve/block_disk_store.hpp"

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace {

using gufo::server::BlockDiskStore;
using gufo::server::BlockKey;
using gufo::server::BlockMissing;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    path_ = std::filesystem::temp_directory_path() /
            ("gufo-block-store-" + std::to_string(::getpid()));
    std::filesystem::remove_all(path_);
  }
  ~TemporaryDirectory() { std::filesystem::remove_all(path_); }
  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
};

constexpr std::size_t kBlockBytes = 4096;
constexpr std::size_t kFileBytes = 120 + 4 * 4 + kBlockBytes;

std::vector<std::uint8_t> Rows(std::uint8_t seed) {
  std::vector<std::uint8_t> rows(kBlockBytes);
  for (std::size_t i = 0; i < rows.size(); ++i)
    rows[i] = static_cast<std::uint8_t>(seed + i * 7);
  return rows;
}

bool PutRows(BlockDiskStore& store, const BlockKey& key, const BlockKey& parent,
             const std::vector<std::uint32_t>& tokens,
             const std::vector<std::uint8_t>& rows) {
  return store.Put(key, parent, tokens, [&](const BlockDiskStore::Sink& sink) {
    // Two pieces, as a bounce stream would deliver them.
    sink(std::span(rows).first(1000));
    sink(std::span(rows).subspan(1000));
  });
}

std::vector<std::uint8_t> GetRows(BlockDiskStore& store, const BlockKey& key,
                                  const BlockKey& parent,
                                  const std::vector<std::uint32_t>& tokens) {
  std::vector<std::uint8_t> rows(kBlockBytes);
  store.Get(key, parent, tokens, [&](const BlockDiskStore::Source& read) {
    read(std::span(rows).first(3000));
    read(std::span(rows).subspan(3000));
  });
  return rows;
}

bool Missing(BlockDiskStore& store, const BlockKey& key, const BlockKey& parent,
             const std::vector<std::uint32_t>& tokens) {
  try {
    (void)GetRows(store, key, parent, tokens);
  } catch (const BlockMissing&) {
    return true;
  }
  return false;
}

void TestChainKeys() {
  const auto root = gufo::server::ChainRootKey("model-A");
  Expect(root != gufo::server::ChainRootKey("model-B"),
         "roots name their model");
  const std::vector<std::uint32_t> tokens{1, 2, 3, 4};
  const auto key = gufo::server::ChainBlockKey(root, tokens, {});
  Expect(key == gufo::server::ChainBlockKey(root, tokens, {}),
         "keys are deterministic");
  const std::vector<std::uint8_t> image{9};
  Expect(key != gufo::server::ChainBlockKey(root, tokens, image),
         "an image in the prefix changes the key");
  const auto next = gufo::server::ChainBlockKey(key, tokens, {});
  Expect(next != key, "a key depends on every block before it");
}

void TestRoundTripRestartAndCorruption() {
  TemporaryDirectory directory;
  const auto root = gufo::server::ChainRootKey("model");
  const std::vector<std::uint32_t> tokens_a{1, 2, 3, 4}, tokens_b{5, 6, 7, 8};
  const auto a = gufo::server::ChainBlockKey(root, tokens_a, {});
  const auto b = gufo::server::ChainBlockKey(a, tokens_b, {});
  {
    BlockDiskStore store({.directory = directory.path(),
                          .capacity_bytes = 16 * kFileBytes,
                          .block_bytes = kBlockBytes});
    Expect(PutRows(store, a, root, tokens_a, Rows(1)) &&
               PutRows(store, b, a, tokens_b, Rows(2)),
           "new blocks are written");
    Expect(!PutRows(store, a, root, tokens_a, Rows(1)),
           "a stored block is skipped");
    Expect(GetRows(store, b, a, tokens_b) == Rows(2), "rows round trip");
    Expect(Missing(store, b, a, tokens_a),
           "a block read with other tokens is refused");
    const auto stats = store.Stats();
    Expect(stats.written_blocks == 2 && stats.skipped_blocks == 1 &&
               stats.stored_blocks == 1 && stats.corrupt_blocks == 1,
           "a mismatched header removes the block and counts it");
    Expect(!store.Contains(b) && store.Contains(a),
           "the mismatched block is gone");
    Expect(PutRows(store, b, a, tokens_b, Rows(2)), "rewrite the block");
  }
  {
    BlockDiskStore store({.directory = directory.path(),
                          .capacity_bytes = 16 * kFileBytes,
                          .block_bytes = kBlockBytes});
    Expect(store.Contains(a) && store.Contains(b) &&
               store.Stats().stored_blocks == 2,
           "a restart rebuilds the index from the files");
    Expect(GetRows(store, a, root, tokens_a) == Rows(1),
           "rows survive a restart");
    // Restores read blocks ahead on several threads at once.
    auto read_a = std::async(std::launch::async,
                             [&] { return store.Read(a, root, tokens_a); });
    auto read_b = std::async(std::launch::async,
                             [&] { return store.Read(b, a, tokens_b); });
    Expect(read_a.get() == Rows(1) && read_b.get() == Rows(2),
           "concurrent reads return checked rows");
  }
  // Flip one row byte: the checksum fails and the block goes.
  for (const auto& file : std::filesystem::directory_iterator(directory.path())) {
    std::fstream stream(file.path(), std::ios::in | std::ios::out |
                                         std::ios::binary);
    stream.seekp(static_cast<std::streamoff>(kFileBytes - 1));
    stream.put('\x5a');
  }
  {
    BlockDiskStore store({.directory = directory.path(),
                          .capacity_bytes = 16 * kFileBytes,
                          .block_bytes = kBlockBytes});
    bool refused = false;
    try {
      (void)store.Read(b, a, tokens_b);
    } catch (const BlockMissing&) {
      refused = true;
    }
    Expect(refused && !store.Contains(b),
           "a read into memory checks the rows before returning them");
    Expect(Missing(store, a, root, tokens_a) && !store.Contains(a) &&
               store.Stats().corrupt_blocks == 2,
           "a corrupt block fails its checksum and is removed");
  }
  {
    // A torn write leaves a temporary file; the scan removes it.
    std::ofstream(directory.path() / "abc.tmp.7") << "partial";
    BlockDiskStore store({.directory = directory.path(),
                          .capacity_bytes = 16 * kFileBytes,
                          .block_bytes = kBlockBytes});
    Expect(!std::filesystem::exists(directory.path() / "abc.tmp.7"),
           "interrupted writes are cleaned up");
  }
}

void TestCapacityEvictsOldLeavesFirst() {
  TemporaryDirectory directory;
  BlockDiskStore store({.directory = directory.path(),
                        .capacity_bytes = 3 * kFileBytes,
                        .block_bytes = kBlockBytes});
  const auto root = gufo::server::ChainRootKey("model");
  const std::vector<std::uint32_t> t1{1, 1, 1, 1}, t2{2, 2, 2, 2},
      t3{3, 3, 3, 3}, t4{4, 4, 4, 4};
  // Chain a <- b, and a separate leaf c; then d over capacity.
  const auto a = gufo::server::ChainBlockKey(root, t1, {});
  const auto b = gufo::server::ChainBlockKey(a, t2, {});
  const auto c = gufo::server::ChainBlockKey(root, t3, {});
  const auto d = gufo::server::ChainBlockKey(root, t4, {});
  (void)PutRows(store, a, root, t1, Rows(1));
  (void)PutRows(store, b, a, t2, Rows(2));
  (void)PutRows(store, c, root, t3, Rows(3));
  store.Touch(std::span(&b, 1));
  (void)PutRows(store, d, root, t4, Rows(4));
  // a is the oldest but b extends it; c is the oldest leaf.
  Expect(store.Contains(a) && store.Contains(b) && !store.Contains(c) &&
             store.Contains(d) && store.Stats().evicted_blocks == 1,
         "the least recently used leaf goes first");
}

// The index holds one reference per block; lookups hand the caller one more;
// only blocks nobody else holds are released, least recently used first.
void TestResidentBlockIndex() {
  std::map<std::uint32_t, std::size_t> references;
  {
    gufo::server::ResidentBlockIndex index({
        .retain = [&](std::uint32_t block) { ++references[block]; },
        .release = [&](std::uint32_t block) { --references[block]; },
        .count = [&](std::uint32_t block) { return references[block]; },
    });
    const auto root = gufo::server::ChainRootKey("model");
    const std::vector<std::uint32_t> t1{1}, t2{2}, t3{3};
    const auto a = gufo::server::ChainBlockKey(root, t1, {});
    const auto b = gufo::server::ChainBlockKey(a, t2, {});
    const auto c = gufo::server::ChainBlockKey(root, t3, {});
    references = {{10, 1}, {11, 1}, {12, 1}};  // each owned by a session
    Expect(index.Insert(a, 10) && index.Insert(b, 11) && index.Insert(c, 12),
           "new keys are indexed");
    Expect(!index.Insert(a, 13) && references[13] == 0,
           "an indexed key keeps its block");
    Expect(references[10] == 2, "the index holds a reference");
    Expect(!index.Acquire(gufo::server::ChainBlockKey(root, t2, {})),
           "an unknown key misses");
    const auto held = index.Acquire(a);
    Expect(held == 10u && references[10] == 3,
           "a hit hands the caller a reference");
    --references[10];
    // Sessions finish: a and c are index-only; b is still shared.
    --references[10];
    --references[12];
    (void)index.Acquire(c);  // c is now the most recently used
    --references[12];
    auto stats = index.stats();
    index.CountRestored(2);
    Expect(stats.blocks == 3 && stats.unshared_blocks == 2 &&
               index.stats().restored == 2,
           "stats count index-only blocks and restored blocks");
    Expect(index.ReleaseUnshared(1) == 1 && references[10] == 0 &&
               references[12] == 1,
           "the least recently used index-only block goes first");
    Expect(index.ReleaseUnshared(5) == 1 && references[12] == 0 &&
               references[11] == 2,
           "a block another owner holds stays");
    stats = index.stats();
    Expect(stats.blocks == 1 && stats.released == 2, "released blocks leave");
  }
  Expect(references[11] == 1, "the index drops its references when destroyed");
}

}  // namespace

int main() {
  TestResidentBlockIndex();
  TestChainKeys();
  TestRoundTripRestartAndCorruption();
  TestCapacityEvictsOldLeavesFirst();
  std::cout << "All block disk store tests passed\n";
  return 0;
}
