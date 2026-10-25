// KV block pool geometry and allocation on the host: several planes with
// different row widths, rows per token block and layer counts, as models
// other than Qwen 27B need (Gemma global K/V of different widths, pooled
// keys at one row per 4 tokens, compressed rows at one per 128 tokens).
#include "src/core/hip/kv_block_layout.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using gufo::hip::kKvBlockTokens;
using gufo::hip::KvBlockAllocator;
using gufo::hip::KvBlockGeometry;
using gufo::hip::KvBlockPlane;
using gufo::hip::KvBlockReservation;

void Expect(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

template<typename F>
void ExpectThrows(F&& f, const std::string& message) {
  try {
    f();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error("no exception: " + message);
}

void TestBlocksFor() {
  Expect(KvBlockGeometry::BlocksFor(0) == 0, "no tokens, no blocks");
  Expect(KvBlockGeometry::BlocksFor(1) == 1, "one token, one block");
  Expect(KvBlockGeometry::BlocksFor(kKvBlockTokens) == 1, "a whole block");
  Expect(KvBlockGeometry::BlocksFor(kKvBlockTokens + 1) == 2,
         "a partial block");
  Expect(KvBlockGeometry::BlocksFor(262144) == 1024, "256K tokens");
  Expect(KvBlockGeometry::BlocksFor(16384) == 64, "16K tokens");
}

// Qwen 27B: K and V, 16 layers of 2 KiB rows. One 256K session.
void TestTwoPlanes() {
  const KvBlockPlane kv{.layers = 16, .row_bytes = 2048};
  const KvBlockGeometry g({kv, kv}, KvBlockGeometry::BlocksFor(262144));
  Expect(g.bytes() == 16ULL << 30, "256K Qwen 27B KV is 16 GiB");
  Expect(g.bytes_per_block() == 16ULL << 20, "a block is 16 MiB");
  Expect(g.plane_offset(1) == 8ULL << 30, "V follows K");
  Expect(g.layer_rows(0) == 262144, "one layer holds every block's rows");
  // Past 2^32 bytes: the last row of the last block of V's last layer.
  Expect(g.RowOffset(1, 15, 1023, 255) == g.bytes() - 2048,
         "last row offset is 64-bit");
  Expect(g.RowOffset(0, 1, 0) == g.layer_pitch(0), "layer pitch");
}

// Gemma 31B global layers: derived keys narrower than values.
void TestDifferentWidths() {
  const KvBlockGeometry g(
      {{.layers = 10, .row_bytes = 1024}, {.layers = 10, .row_bytes = 4096}},
      512);
  Expect(g.bytes_per_block() == 256ULL * (1024 + 4096) * 10,
         "Gemma 31B block bytes");
  Expect(g.plane_offset(1) == 10ULL * 512 * 256 * 1024, "V after narrow K");
  Expect(g.RowOffset(1, 0, 1, 0) - g.RowOffset(1, 0, 0, 0) == 256 * 4096,
         "V block stride uses V's width");
}

// Flash-Next: K, V and pooled keys at one row per 4 tokens.
void TestPooledRows() {
  const KvBlockGeometry g(
      {{.layers = 12, .row_bytes = 1024},
       {.layers = 12, .row_bytes = 1024},
       {.layers = 12, .row_bytes = 256, .rows_per_block = kKvBlockTokens / 4}},
      1024);
  Expect(g.layer_rows(2) == 1024 * 64, "pooled rows per layer");
  Expect(g.block_bytes(2) == 64 * 256, "pooled block bytes");
  Expect(g.bytes_per_block() == 12ULL * (2 * 256 * 1024 + 64 * 256),
         "Flash-Next block bytes");
}

// DeepSeek-like compressed planes: 1/4 and 1/128 rows, different layer
// counts, four planes.
void TestCompressedRows() {
  const KvBlockGeometry g(
      {{.layers = 21, .row_bytes = 2048, .rows_per_block = 64},
       {.layers = 21, .row_bytes = 1024, .rows_per_block = 64},
       {.layers = 21, .row_bytes = 512, .rows_per_block = 64},
       {.layers = 20, .row_bytes = 2048, .rows_per_block = 2}},
      1024);
  Expect(g.layer_rows(3) == 2048, "two compressed rows per block");
  Expect(g.bytes_per_block() ==
             21ULL * 64 * (2048 + 1024 + 512) + 20ULL * 2 * 2048,
         "compressed block bytes");
  Expect(g.RowOffset(3, 19, 1023, 1) == g.bytes() - 2048,
         "last compressed row");
  ExpectThrows([&] { (void)g.RowOffset(3, 20, 0); }, "layer past the plane");
  ExpectThrows([&] { (void)g.RowOffset(3, 0, 0, 2); }, "row past the block");
  ExpectThrows([&] { (void)g.RowOffset(0, 0, 1024); }, "block past the pool");
}

void TestInvalid() {
  ExpectThrows([] { KvBlockGeometry({}, 4); }, "no planes");
  ExpectThrows([] { KvBlockGeometry({{.layers = 1, .row_bytes = 2}}, 0); },
               "no blocks");
  ExpectThrows(
      [] {
        KvBlockGeometry({{.layers = 1, .row_bytes = 2, .rows_per_block = 3}},
                        4);
      },
      "rows per block not a power of two");
  ExpectThrows(
      [] {
        KvBlockGeometry({{.layers = 1, .row_bytes = 2, .rows_per_block = 512}},
                        4);
      },
      "more rows than tokens per block");
  ExpectThrows(
      [] {
        KvBlockGeometry({{.layers = 1U << 31, .row_bytes = 1ULL << 40}},
                        1U << 31);
      },
      "size overflow");
}

void TestAllocator() {
  KvBlockAllocator a(8);
  const auto first = a.Allocate(3);
  Expect(first == std::vector<std::uint32_t>{0, 1, 2},
         "a fresh pool gives consecutive blocks");
  const auto second = a.Allocate(2);
  Expect(second == std::vector<std::uint32_t>{3, 4}, "next free blocks");
  ExpectThrows([&] { (void)a.Allocate(4); }, "more than free");
  Expect(a.free_blocks() == 3, "a refused allocation takes nothing");
  // A sharer keeps a block alive after its first owner lets go.
  a.Retain(std::vector<std::uint32_t>{1});
  Expect(a.references(1) == 2, "retained block");
  a.Release(first);
  Expect(a.references(1) == 1 && a.references(0) == 0, "released first");
  Expect(a.free_blocks() == 5, "two blocks freed, one still shared");
  // Freed blocks are reused lowest first, so a sequence may be scattered.
  Expect(a.Allocate(2) == std::vector<std::uint32_t>{0, 2},
         "lowest free blocks around a shared one");
  ExpectThrows([&] { a.Retain(std::vector<std::uint32_t>{7}); },
               "retaining a free block");
  a.Release(std::vector<std::uint32_t>{7, 99});
  Expect(a.free_blocks() == 3, "releasing free or unknown blocks is ignored");
}

// Admission holds blocks back for a request; others cannot take them, and
// what the request does not use goes back.
void TestReservation() {
  KvBlockAllocator a(8);
  auto held = a.TryReserve(5);
  Expect(bool(held) && held.blocks() == 5, "reserved five");
  Expect(a.free_blocks() == 3 && a.reserved_blocks() == 5,
         "reserved blocks are not free");
  Expect(!a.TryReserve(4), "cannot reserve past the unreserved free blocks");
  ExpectThrows([&] { (void)a.Allocate(4); },
               "an unreserved allocation cannot take reserved blocks");
  const auto mine = a.Allocate(2, &held);
  Expect(mine == std::vector<std::uint32_t>{0, 1}, "allocated from reserve");
  Expect(held.blocks() == 3 && a.reserved_blocks() == 3 && a.free_blocks() == 3,
         "allocation draws the reservation down");
  // Past the reservation, a holder takes unreserved free blocks.
  const auto more = a.Allocate(5, &held);
  Expect(more.size() == 5 && held.blocks() == 0 && a.free_blocks() == 1,
         "reservation first, then free blocks");
  ExpectThrows([&] { (void)a.Allocate(2, &held); },
               "nothing reserved and one free");
  Expect(a.free_blocks() == 1, "a refused allocation takes nothing");
  a.Release(mine);
  auto second = a.TryReserve(3);
  Expect(bool(second) && a.free_blocks() == 0, "reserved the rest");
  {
    auto moved = std::move(second);
    Expect(!second && moved.blocks() == 3, "moved reservation");
  }
  Expect(a.free_blocks() == 3 && a.reserved_blocks() == 0,
         "a dropped reservation returns its blocks");
  KvBlockAllocator other(4);
  auto foreign = other.TryReserve(1);
  ExpectThrows([&] { (void)a.Allocate(1, &foreign); },
               "a reservation of another allocator");
  KvBlockReservation empty;
  Expect(a.Allocate(1, &empty).size() == 1, "an empty reservation is none");
}

}  // namespace

int main() {
  try {
    TestBlocksFor();
    TestTwoPlanes();
    TestDifferentWidths();
    TestPooledRows();
    TestCompressedRows();
    TestInvalid();
    TestAllocator();
    TestReservation();
  } catch (const std::exception& error) {
    std::cerr << "kv_block_layout_test: " << error.what() << '\n';
    return 1;
  }
  std::cout << "KV block layout tests passed.\n";
  return 0;
}
