#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/bounce_stream.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/kv_block_pool.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"
#include "tests/testing/test_common.hpp"

namespace {

using gufo::hip::KvBlockPool;
using gufo::hip::KvBlockTable;
using gufo::hip::kKvBlockTokens;
using gufo::test::DeviceBuffer;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

// The production 27B attention shape: 24 query heads over 4 KV heads of 256.
constexpr unsigned kHeads = 24, kKvHeads = 4, kDim = 256;
constexpr unsigned kWidth = kHeads * kDim, kKvWidth = kKvHeads * kDim;
constexpr unsigned kLayers = 2, kLayer = 1;
constexpr unsigned kContext = 4096;
constexpr unsigned kLogicalBlocks = kContext / kKvBlockTokens;
// More pool blocks than the sequence uses, placed out of order, so a kernel
// that ignores the table reads or writes the wrong rows.
constexpr unsigned kPoolBlocks = kLogicalBlocks + 5;
static_assert(kPoolBlocks % 7 != 0);

std::uint32_t PhysicalBlock(unsigned logical) {
  return (logical * 7U + 3U) % kPoolBlocks;
}

std::size_t ContiguousIndex(unsigned layer, unsigned position) {
  return (static_cast<std::size_t>(layer) * kContext + position) * kKvWidth;
}

std::size_t PagedIndex(unsigned layer, unsigned position) {
  return (static_cast<std::size_t>(layer) * kPoolBlocks * kKvBlockTokens +
          static_cast<std::size_t>(PhysicalBlock(position / kKvBlockTokens)) *
              kKvBlockTokens +
          position % kKvBlockTokens) *
         kKvWidth;
}

// One FP16 K or V plane twice: contiguous rows of kContext per layer, and
// the same rows scattered through a pool by the block table.
struct Planes {
  DeviceBuffer<__half> contiguous;
  DeviceBuffer<__half> paged;
};

Planes MakePlanes(float phase) {
  std::vector<__half> contiguous(std::size_t{kLayers} * kContext * kKvWidth);
  std::vector<__half> paged(std::size_t{kLayers} * kPoolBlocks *
                                kKvBlockTokens * kKvWidth,
                            __float2half(-7.0F));
  for (unsigned layer = 0; layer < kLayers; ++layer)
    for (unsigned position = 0; position < kContext; ++position)
      for (unsigned column = 0; column < kKvWidth; ++column) {
        const float value =
            0.4F * std::sin(static_cast<float>((layer * kContext + position) *
                                                   kKvWidth +
                                               column) *
                                0.011F +
                            phase);
        contiguous[ContiguousIndex(layer, position) + column] =
            __float2half(value);
        paged[PagedIndex(layer, position) + column] = __float2half(value);
      }
  return {DeviceBuffer<__half>(contiguous), DeviceBuffer<__half>(paged)};
}

// Every row of the paged plane, read back in contiguous order. Pool blocks
// outside the table keep their sentinel.
void ExpectSameRows(const Planes& planes, std::string_view what) {
  const auto contiguous = planes.contiguous.CopyToHost();
  const auto paged = planes.paged.CopyToHost();
  for (unsigned layer = 0; layer < kLayers; ++layer)
    for (unsigned position = 0; position < kContext; ++position)
      Expect(std::memcmp(&contiguous[ContiguousIndex(layer, position)],
                         &paged[PagedIndex(layer, position)],
                         kKvWidth * sizeof(__half)) == 0,
             std::string(what) + ": paged KV rows differ");
}

std::vector<float> Wave(std::size_t size, float frequency, float scale) {
  std::vector<float> values(size);
  for (std::size_t i = 0; i < size; ++i)
    values[i] = scale * std::sin(static_cast<float>(i) * frequency);
  return values;
}

struct Fixture {
  Planes keys = MakePlanes(0.0F);
  Planes values = MakePlanes(1.3F);
  DeviceBuffer<std::uint32_t> table;
  KvBlockTable paged_table;

  Fixture() {
    std::vector<std::uint32_t> blocks(kLogicalBlocks);
    for (unsigned logical = 0; logical < kLogicalBlocks; ++logical)
      blocks[logical] = PhysicalBlock(logical);
    table = DeviceBuffer<std::uint32_t>(blocks);
    paged_table = {table.data(), kPoolBlocks * kKvBlockTokens};
  }
};

// Runs `launch` once on contiguous storage and once through the table, then
// requires the same output bits and the same KV rows.
void ExpectSame(Fixture& fixture, std::size_t output_size,
                const std::function<void(__half* keys, __half* values,
                                         KvBlockTable table, float* out)>&
                    launch,
                std::string_view what) {
  DeviceBuffer<float> contiguous_out(output_size), paged_out(output_size);
  HIP_CHECK(hipMemset(contiguous_out.data(), 0, output_size * sizeof(float)));
  HIP_CHECK(hipMemset(paged_out.data(), 0, output_size * sizeof(float)));
  launch(fixture.keys.contiguous.data(), fixture.values.contiguous.data(),
         KvBlockTable{}, contiguous_out.data());
  launch(fixture.keys.paged.data(), fixture.values.paged.data(),
         fixture.paged_table, paged_out.data());
  HIP_CHECK(hipDeviceSynchronize());
  const auto expected = contiguous_out.CopyToHost();
  const auto actual = paged_out.CopyToHost();
  Expect(std::memcmp(expected.data(), actual.data(),
                     output_size * sizeof(float)) == 0,
         std::string(what) + ": paged attention changed output bits");
  ExpectSameRows(fixture.keys, what);
  ExpectSameRows(fixture.values, what);
}

void TestDecodeAndVerification(Fixture& fixture) {
  const DeviceBuffer<float> q(Wave(8 * kWidth, 0.073F, 0.3F));
  const DeviceBuffer<float> gate(Wave(8 * kWidth, 0.019F, 1.0F));
  const DeviceBuffer<float> k(Wave(8 * kKvWidth, 0.031F, 0.5F));
  const DeviceBuffer<float> v(Wave(8 * kKvWidth, 0.047F, 0.5F));
  const auto row_scratch =
      gufo::hip::detail::DecodeAttentionScratchElements(kHeads, kDim);
  DeviceBuffer<float> scratch(8 * row_scratch);
  // Online (< 128), split-K below and above the shared-KV threshold (512),
  // partitions that straddle block boundaries, and the last rows.
  for (unsigned pos : {5U, 63U, 64U, 127U, 128U, 600U, 2049U, kContext - 1}) {
    for (bool write : {false, true}) {
      ExpectSame(
          fixture, kWidth,
          [&](__half* keys, __half* values, KvBlockTable table, float* out) {
            gufo::hip::LaunchAttention(
                q.data(), k.data(), v.data(), gate.data(), nullptr, nullptr,
                keys, values, table, out, kLayer, pos, kContext, kHeads,
                kKvHeads, kDim, nullptr, scratch.data(), !write);
          },
          "decode at " + std::to_string(pos));
    }
  }
  // Device-position decode (the graph route) and the fused decode write.
  DeviceBuffer<std::uint32_t> d_pos(1);
  for (unsigned pos : {0U, 64U, 126U}) {
    HIP_CHECK(hipMemcpy(d_pos.data(), &pos, sizeof(pos), hipMemcpyHostToDevice));
    ExpectSame(
        fixture, kWidth,
        [&](__half* keys, __half* values, KvBlockTable table, float* out) {
          gufo::hip::LaunchAttention(q.data(), k.data(), v.data(), gate.data(),
                                     nullptr, nullptr, keys, values, table,
                                     out, kLayer, d_pos.data(), kContext,
                                     kHeads, kKvHeads, kDim, nullptr, false);
        },
        "device-position decode at " + std::to_string(pos));
  }
  for (unsigned pos : {3U, 64U, 1000U}) {
    HIP_CHECK(hipMemcpy(d_pos.data(), &pos, sizeof(pos), hipMemcpyHostToDevice));
    DeviceBuffer<float> q_out(kWidth), k_out(kKvWidth);
    ExpectSame(
        fixture, 1,
        [&](__half* keys, __half* values, KvBlockTable table, float*) {
          gufo::hip::LaunchFusedQKNormRoPEKvWrite(
              q.data(), k.data(), v.data(), nullptr, nullptr, q_out.data(),
              k_out.data(), nullptr, nullptr, keys, values, table, kLayer,
              d_pos.data(), kContext, kHeads, kKvHeads, kDim, 64, 1.0e7F);
        },
        "fused decode write at " + std::to_string(pos));
  }
  // Verification rows: wave64 from 512 with eight rows, the 16-lane shared
  // kernel for odd shapes and the per-row fallback with one row of scratch.
  for (unsigned start : {100U, 504U, 600U, 1500U}) {
    for (unsigned rows : {2U, 3U, 8U}) {
      for (auto capacity : {row_scratch, scratch.size()}) {
        ExpectSame(
            fixture, rows * kWidth,
            [&](__half* keys, __half* values, KvBlockTable table, float* out) {
              gufo::hip::LaunchCausalDecodeAttention(
                  q.data(), gate.data(), nullptr, nullptr, keys, values, table,
                  out, kLayer, start, rows, kContext, kHeads, kKvHeads, kDim,
                  nullptr, {scratch.data(), capacity});
            },
            "verification rows " + std::to_string(rows) + " at " +
                std::to_string(start));
      }
    }
  }
}

void TestPrefill(Fixture& fixture) {
  constexpr unsigned max_batch = 1100;
  const DeviceBuffer<float> q(Wave(max_batch * kWidth, 0.0071F, 0.3F));
  const DeviceBuffer<float> gate(Wave(max_batch * kWidth, 0.0013F, 1.0F));
  const DeviceBuffer<float> k(Wave(max_batch * kKvWidth, 0.0037F, 0.5F));
  const DeviceBuffer<float> v(Wave(max_batch * kKvWidth, 0.0059F, 0.5F));
  // Room for every KV head of the packed head-major copy at 2048 keys.
  DeviceBuffer<float> packed_k(kKvHeads * 2048 * kDim / 2 + 64);
  DeviceBuffer<float> packed_v(kKvHeads * 2048 * kDim / 2 + 64);
  struct Case {
    unsigned start, batch;
  };
  for (const Case c :
       {Case{0, 40}, Case{70, 40}, Case{130, 200}, Case{900, max_batch}}) {
    for (bool write : {false, true}) {
      const std::string where =
          std::to_string(c.batch) + " rows at " + std::to_string(c.start);
      if (c.start + c.batch <= 512) {
        ExpectSame(
            fixture, c.batch * kWidth,
            [&](__half* keys, __half* values, KvBlockTable table, float* out) {
              gufo::hip::LaunchBatchedAttention(
                  q.data(), k.data(), v.data(), gate.data(), nullptr, nullptr,
                  keys, values, table, out, kLayer, c.start, c.batch, kContext,
                  kHeads, kKvHeads, kDim, nullptr, !write);
            },
            "batched prefill " + where);
      }
      ExpectSame(
          fixture, c.batch * kWidth,
          [&](__half* keys, __half* values, KvBlockTable table, float* out) {
            Expect(gufo::hip::LaunchBatchedAttentionTile(
                       q.data(), k.data(), v.data(), gate.data(), nullptr,
                       nullptr, keys, values, table, out, kLayer, c.start,
                       c.batch, kContext, kHeads, kKvHeads, kDim, nullptr,
                       nullptr, 0, !write),
                   "tile attention declined the production shape");
          },
          "tile prefill " + where);
      ExpectSame(
          fixture, c.batch * kWidth,
          [&](__half* keys, __half* values, KvBlockTable table, float* out) {
            Expect(gufo::hip::LaunchQwenWmmaAttention(
                       q.data(), k.data(), v.data(), gate.data(), nullptr,
                       nullptr, keys, values, table, out, kLayer, c.start,
                       c.batch, kContext, kHeads, kKvHeads, kDim, nullptr,
                       nullptr, 0, !write,
                       {packed_k.data(), packed_k.size()},
                       {packed_v.data(), packed_v.size()}),
                   "WMMA attention declined the production shape");
          },
          "WMMA prefill " + where);
      ExpectSame(
          fixture, 1,
          [&](__half* keys, __half* values, KvBlockTable table, float*) {
            DeviceBuffer<float> q_out(c.batch * kWidth),
                k_out(c.batch * kKvWidth);
            gufo::hip::LaunchBatchedFusedQKNormRoPEKvWrite(
                q.data(), k.data(), v.data(), nullptr, nullptr, q_out.data(),
                k_out.data(), nullptr, nullptr, keys, values, table, kLayer,
                c.start, c.batch, kContext, kHeads, kKvHeads, kDim, 64,
                1.0e7F);
          },
          "batched fused write " + where);
    }
  }
}

gufo::core::ModelConfig ProductionQwen27BConfig() {
  gufo::core::ModelConfig config;
  config.model_name = "qwen3.8-27b";
  config.num_layers = 64;
  config.hidden_size = 5120;
  config.intermediate_size = 17408;
  config.num_attention_heads = 24;
  config.num_key_value_heads = 4;
  config.head_dim = 256;
  config.vocab_size = 248320;
  config.context_length = 262144;
  config.full_attention_interval = 4;
  config.mtp_num_layers = 1;
  config.ssm_conv_kernel = 4;
  config.ssm_state_size = 128;
  config.ssm_group_count = 16;
  config.ssm_time_step_rank = 48;
  config.ssm_inner_size = 6144;
  config.rotary_dim = 64;
  config.rope_theta = 10000000.0F;
  return config;
}

std::vector<std::uint8_t> StreamLive(gufo::hip::QwenGpuArena& arena,
                                     std::uint32_t valid_context) {
  gufo::hip::BounceStream bounce(12289);
  std::vector<std::uint8_t> bytes;
  for (const auto& segment : arena.LiveCompactSegments(valid_context, {})) {
    if (!segment.host.empty())
      bytes.insert(bytes.end(), segment.host.begin(), segment.host.end());
    else
      bounce.Out(segment.device, [&](std::span<const std::uint8_t> piece) {
        bytes.insert(bytes.end(), piece.begin(), piece.end());
      });
  }
  return bytes;
}

// An arena on scattered pool blocks keeps the compact snapshot format: its
// snapshots, streams and restores match an arena on its own pool byte for
// byte, and its blocks return to the pool when it is destroyed.
void TestArenaOnSharedPool() {
  const auto config = ProductionQwen27BConfig();
  const auto policy = gufo::hip::QwenExecutionPolicy::Production();
  // Not a whole number of blocks: the last block is partly used.
  constexpr std::uint32_t context = 1000, valid_context = 777;
  const std::uint32_t blocks = KvBlockPool::BlocksFor(context);
  const std::size_t row_bytes = kKvWidth * sizeof(std::uint16_t);
  auto pool = std::make_shared<KvBlockPool>(16, row_bytes, 2 * blocks + 1);
  // Leave every other block free, so the arena's blocks are not one run.
  const auto all = pool->Allocate(pool->blocks());
  std::vector<std::uint32_t> held;
  for (std::size_t i = 0; i < all.size(); ++i)
    if (i % 2 == 0)
      held.push_back(all[i]);
  std::vector<std::uint32_t> released;
  for (std::size_t i = 1; i < all.size(); i += 2)
    released.push_back(all[i]);
  pool->Release(released);

  gufo::hip::QwenGpuArena own(config, context, policy);
  std::vector<std::uint8_t> payload;
  {
    gufo::hip::QwenGpuArena shared(config, context, policy, pool);
    const auto table = shared.GetKvBlocks();
    Expect(table.size() == blocks && table[0] == 1 && table[1] == 3,
           "arena must take the scattered free blocks");
    Expect(pool->free_blocks() == 0, "arena must take exactly its blocks");

    // Fill the live rows of `own`, then move them to `shared` through a
    // snapshot. Both must then serialize and stream the same bytes.
    const std::size_t plane = own.GetAttentionKvPlaneElements();
    std::vector<std::uint16_t> kv(2 * plane);
    for (std::size_t i = 0; i < kv.size(); ++i)
      kv[i] = static_cast<std::uint16_t>(0x1000U + (i * 2654435761U) % 0x3FFFU);
    HIP_CHECK(hipMemcpy(own.d_attention_kv_f16, kv.data(),
                        kv.size() * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());
    const auto own_snapshot = own.SaveSnapshot(valid_context);
    payload.resize(own_snapshot->CompactPayloadBytes());
    (void)own_snapshot->SerializeCompact(payload);
    shared.RestoreSnapshot(*own_snapshot);
    const auto shared_snapshot = shared.SaveSnapshot(valid_context);
    std::vector<std::uint8_t> shared_payload(
        shared_snapshot->CompactPayloadBytes());
    (void)shared_snapshot->SerializeCompact(shared_payload);
    Expect(shared_payload == payload,
           "scattered blocks changed the compact snapshot");
    Expect(StreamLive(shared, valid_context) == payload &&
               StreamLive(own, valid_context) == payload,
           "scattered blocks changed the live compact stream");

    // Restores from bytes land in the scattered rows.
    gufo::hip::QwenGpuArena fresh(config, context, policy);
    shared.Reset();
    shared.RestoreCompactSnapshot(payload, valid_context);
    Expect(StreamLive(shared, valid_context) == payload,
           "compact restore into scattered blocks changed the state");
    std::size_t offset = 0;
    gufo::hip::BounceStream bounce(1 << 20);
    (void)shared.RestoreCompactSnapshotStream(
        bounce,
        [&](std::span<std::uint8_t> piece) {
          std::memcpy(piece.data(), payload.data() + offset, piece.size());
          offset += piece.size();
        },
        payload.size(), valid_context);
    Expect(offset == payload.size() &&
               StreamLive(shared, valid_context) == payload,
           "streamed restore into scattered blocks changed the state");
  }
  Expect(pool->free_blocks() == blocks,
         "a destroyed arena must return its blocks");
  pool->Release(held);
  Expect(pool->free_blocks() == pool->blocks(), "pool must be whole again");

  bool refused = false;
  try {
    auto small = std::make_shared<KvBlockPool>(16, row_bytes, blocks - 1);
    gufo::hip::QwenGpuArena too_big(config, context, policy, small);
  } catch (const std::length_error&) {
    refused = true;
  }
  Expect(refused, "an arena must refuse a pool without enough free blocks");
}


// Reference counts and reservations: a shared block stays taken until its
// last holder releases it, and reserved blocks are only for their owner.
void TestPoolReferencesAndReservations() {
  KvBlockPool pool(16, kKvWidth * sizeof(std::uint16_t), 10);
  const auto taken = pool.Allocate(3);
  pool.Retain(std::span(taken).first(2));
  Expect(pool.references(taken[0]) == 2 && pool.references(taken[2]) == 1 &&
             pool.free_blocks() == 7,
         "retained blocks must count their holders");
  pool.Release(std::span(taken).first(2));
  Expect(pool.references(taken[0]) == 1 && pool.free_blocks() == 7,
         "a block with a holder left must stay taken");
  {
    auto reservation = pool.TryReserve(5);
    Expect(reservation && reservation.blocks() == 5 &&
               pool.free_blocks() == 2 && !pool.TryReserve(3),
           "reserved blocks must not be free to others");
    const auto own = pool.Allocate(4, reservation);
    Expect(own.size() == 4 && reservation.blocks() == 1 &&
               pool.free_blocks() == 2,
           "allocation must draw on its reservation first");
    const auto more = pool.Allocate(3, reservation);
    Expect(reservation.blocks() == 0 && pool.free_blocks() == 0,
           "past its reservation, allocation takes free blocks");
    bool refused = false;
    try {
      (void)pool.Allocate(1);
    } catch (const std::length_error&) {
      refused = true;
    }
    Expect(refused, "an empty pool must refuse allocation");
    pool.Release(own);
    pool.Release(more);
    auto left = pool.TryReserve(4);
    Expect(left && pool.free_blocks() == 3, "reservation of released blocks");
  }
  Expect(pool.free_blocks() == 7 && pool.reserved_blocks() == 0,
         "a dropped reservation must return its blocks");
  pool.Release(taken);
  Expect(pool.free_blocks() == 10, "pool must be whole again");
}

// On-demand arenas map blocks as rows are written. A snapshot shares the
// arena's blocks instead of copying them, a restore adopts them, and the
// first write into a shared block copies it first, so neither holder sees
// the other's later rows. Compact bytes match a whole-context arena.
void TestOnDemandArena() {
  using Growth = gufo::hip::QwenGpuArena::KvGrowth;
  const auto config = ProductionQwen27BConfig();
  const auto policy = gufo::hip::QwenExecutionPolicy::Production();
  constexpr std::uint32_t context = 1000, valid_context = 777;
  const std::uint32_t blocks = KvBlockPool::BlocksFor(context);
  const std::uint32_t valid_blocks = KvBlockPool::BlocksFor(valid_context);
  const std::size_t row_bytes = kKvWidth * sizeof(std::uint16_t);
  auto pool = std::make_shared<KvBlockPool>(16, row_bytes, 3 * blocks + 2);
  {
    // Deterministic rows everywhere, so any mapped block has known bytes.
    const std::size_t pool_bytes = 2 * pool->plane_bytes();
    std::vector<std::uint16_t> fill(pool_bytes / sizeof(std::uint16_t));
    for (std::size_t i = 0; i < fill.size(); ++i)
      fill[i] =
          static_cast<std::uint16_t>(0x1000U + (i * 2654435761U) % 0x3FFFU);
    HIP_CHECK(hipMemcpy(pool->data(), fill.data(), pool_bytes,
                        hipMemcpyHostToDevice));
  }
  gufo::hip::QwenGpuArena own(config, context, policy);
  std::vector<std::uint8_t> payload;
  {
    gufo::hip::QwenGpuArena a(config, context, policy, pool, Growth::kOnDemand);
    Expect(a.GetKvBlocks().empty() && pool->free_blocks() == pool->blocks(),
           "an on-demand arena must start without blocks");
    a.EnsureKvRows(0, valid_context);
    Expect(a.GetKvBlocks().size() == valid_blocks &&
               pool->free_blocks() == pool->blocks() - valid_blocks,
           "an on-demand arena must map the blocks its rows need");
    HIP_CHECK(hipDeviceSynchronize());
    payload = StreamLive(a, valid_context);

    std::unique_ptr<gufo::hip::QwenGpuSnapshot> snapshot =
        a.SaveSnapshot(valid_context);
    const auto a_blocks = std::vector<std::uint32_t>(a.GetKvBlocks().begin(),
                                                     a.GetKvBlocks().end());
    Expect(snapshot->SharedKvBytes() ==
                   valid_blocks * KvBlockPool::DeviceBytes(16, row_bytes, 1) &&
               pool->references(a_blocks.front()) == 2 &&
               pool->free_blocks() == pool->blocks() - valid_blocks,
           "a snapshot must share the arena's blocks without new ones");
    std::vector<std::uint8_t> snapshot_payload(snapshot->CompactPayloadBytes());
    (void)snapshot->SerializeCompact(snapshot_payload);
    Expect(snapshot_payload == payload,
           "a shared snapshot must serialize the live bytes");

    gufo::hip::QwenGpuArena b(config, context, policy, pool, Growth::kOnDemand);
    b.EnsureKvRows(0, 1);
    b.RestoreSnapshot(*snapshot);
    Expect(std::ranges::equal(b.GetKvBlocks(), a_blocks) &&
               pool->references(a_blocks.front()) == 3 &&
               pool->free_blocks() == pool->blocks() - valid_blocks,
           "a restore must adopt the shared blocks and drop its own");
    Expect(StreamLive(b, valid_context) == payload,
           "an adopted restore changed the state");

    // The next row lands in the shared partial block: b copies it first.
    b.EnsureKvRows(valid_context, valid_context + 1);
    const std::uint32_t tail = valid_blocks - 1;
    Expect(b.GetKvBlocks()[tail] != a_blocks[tail] &&
               pool->references(a_blocks[tail]) == 2 &&
               pool->free_blocks() == pool->blocks() - valid_blocks - 1,
           "a write into a shared block must copy it");
    HIP_CHECK(hipDeviceSynchronize());
    Expect(StreamLive(b, valid_context) == payload,
           "copy-on-write changed the copied rows");
    // Scribble over b's copy; the snapshot and a must not change.
    const std::size_t width = kKvBlockTokens * row_bytes;
    const std::size_t pitch = std::size_t{pool->layer_tokens()} * row_bytes;
    for (std::uint32_t plane = 0; plane < 2; ++plane)
      HIP_CHECK(hipMemset2D(static_cast<std::uint8_t*>(pool->data()) +
                                plane * pool->plane_bytes() +
                                b.GetKvBlocks()[tail] * width,
                            pitch, 0, width, 16));
    HIP_CHECK(hipDeviceSynchronize());
    Expect(StreamLive(b, valid_context) != payload &&
               StreamLive(a, valid_context) == payload,
           "a copied block must be private to its writer");
    std::vector<std::uint8_t> again(snapshot->CompactPayloadBytes());
    (void)snapshot->SerializeCompact(again);
    Expect(again == payload, "a writer's copy changed the shared snapshot");

    // A whole-context arena restored from these bytes snapshots the same.
    own.RestoreCompactSnapshot(payload, valid_context);
    const auto own_snapshot = own.SaveSnapshot(valid_context);
    std::vector<std::uint8_t> own_payload(own_snapshot->CompactPayloadBytes());
    (void)own_snapshot->SerializeCompact(own_payload);
    Expect(own_payload == payload && own_snapshot->SharedKvBytes() == 0,
           "paged and whole-context snapshots must serialize the same");

    // A compact restore replaces shared blocks with fresh ones, no copy.
    a.RestoreCompactSnapshot(payload, valid_context);
    Expect(StreamLive(a, valid_context) == payload &&
               pool->references(a_blocks.front()) == 2,
           "a compact restore must leave the shared blocks to their sharers");
    b.TruncateKvRows(0);
    Expect(b.GetKvBlocks().empty(), "truncation must drop every block");
    snapshot.reset();
    Expect(pool->free_blocks() == pool->blocks() - valid_blocks,
           "the last holder must free a shared block");

    // Growth draws on a reservation first and refuses a full pool whole.
    gufo::hip::QwenGpuArena c(config, context, policy, pool, Growth::kOnDemand);
    c.SetKvReservation(pool->TryReserve(2));
    c.EnsureKvRows(0, 2 * kKvBlockTokens);
    Expect(c.GetKvBlocks().size() == 2 && c.GetKvReservedBlocks() == 0 &&
               pool->reserved_blocks() == 0,
           "growth must use the reservation");
    const auto rest = pool->Allocate(pool->free_blocks());
    bool refused = false;
    try {
      c.EnsureKvRows(2 * kKvBlockTokens, 3 * kKvBlockTokens);
    } catch (const std::length_error&) {
      refused = true;
    }
    Expect(refused && c.GetKvBlocks().size() == 2,
           "a full pool must refuse growth and leave the table unchanged");
    pool->Release(rest);
  }
  Expect(pool->free_blocks() == pool->blocks(),
         "destroyed on-demand arenas must return every block");

  bool refused = false;
  try {
    gufo::hip::QwenGpuArena no_pool(config, context, policy, nullptr,
                                    Growth::kOnDemand);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  Expect(refused, "on-demand growth without a shared pool must be refused");
}


std::vector<std::uint8_t> StreamPaged(gufo::hip::QwenGpuArena& arena,
                                      std::uint32_t valid_context) {
  gufo::hip::BounceStream bounce(12289);
  std::vector<std::uint8_t> bytes;
  for (const auto& segment :
       arena.LiveCompactSegments(valid_context, {}, true)) {
    if (!segment.host.empty())
      bytes.insert(bytes.end(), segment.host.begin(), segment.host.end());
    else
      bounce.Out(segment.device, [&](std::span<const std::uint8_t> piece) {
        bytes.insert(bytes.end(), piece.begin(), piece.end());
      });
  }
  return bytes;
}

std::vector<std::uint8_t> StreamRegions(
    const std::array<gufo::hip::DeviceRegion, 2>& regions) {
  gufo::hip::BounceStream bounce(12289);
  std::vector<std::uint8_t> bytes;
  for (const auto& region : regions)
    bounce.Out(region, [&](std::span<const std::uint8_t> piece) {
      bytes.insert(bytes.end(), piece.begin(), piece.end());
    });
  return bytes;
}

// The paged compact form leaves the full blocks out and keeps the partial
// last block inline; restoring it plus each block, block by block, gives
// back the whole state. A block-sharing snapshot gives the same bytes.
void TestPagedCompactRoundTrip() {
  using Growth = gufo::hip::QwenGpuArena::KvGrowth;
  const auto config = ProductionQwen27BConfig();
  const auto policy = gufo::hip::QwenExecutionPolicy::Production();
  constexpr std::uint32_t context = 1000, valid_context = 777;
  const std::uint32_t full = valid_context / kKvBlockTokens;
  const std::size_t row_bytes = kKvWidth * sizeof(std::uint16_t);
  auto pool = std::make_shared<KvBlockPool>(16, row_bytes,
                                            3 * KvBlockPool::BlocksFor(context));
  {
    std::vector<std::uint16_t> fill(pool->plane_bytes());
    for (std::size_t i = 0; i < fill.size(); ++i)
      fill[i] =
          static_cast<std::uint16_t>(0x2000U + (i * 2246822519U) % 0x1FFFU);
    HIP_CHECK(hipMemcpy(pool->data(), fill.data(), 2 * pool->plane_bytes(),
                        hipMemcpyHostToDevice));
  }
  gufo::hip::QwenGpuArena a(config, context, policy, pool, Growth::kOnDemand);
  a.EnsureKvRows(0, valid_context);
  HIP_CHECK(hipDeviceSynchronize());
  const auto whole = StreamLive(a, valid_context);
  const auto paged = StreamPaged(a, valid_context);
  const std::size_t block_bytes = KvBlockPool::DeviceBytes(16, row_bytes, 1);
  Expect(paged.size() + full * block_bytes == whole.size() &&
             paged.size() ==
                 gufo::hip::QwenGpuArena::CompactSnapshotPayloadBytes(
                     config, context, policy, valid_context, true),
         "the paged form leaves exactly the full blocks out");
  std::vector<std::vector<std::uint8_t>> blocks;
  for (std::uint32_t block = 0; block < full; ++block)
    blocks.push_back(StreamRegions(a.KvBlockRegions(block)));

  const auto snapshot = a.SaveSnapshot(valid_context);
  std::vector<std::uint8_t> snapshot_paged(snapshot->CompactPayloadBytes(true));
  (void)snapshot->SerializeCompact(snapshot_paged, true);
  Expect(snapshot_paged == paged &&
             StreamRegions(snapshot->KvBlockRegions(full - 1)) == blocks.back(),
         "a shared snapshot gives the live paged bytes and blocks");

  gufo::hip::QwenGpuArena b(config, context, policy, pool, Growth::kOnDemand);
  std::size_t offset = 0;
  gufo::hip::BounceStream bounce(1 << 20);
  b.Reset();
  (void)b.RestoreCompactSnapshotStream(
      bounce,
      [&](std::span<std::uint8_t> piece) {
        std::memcpy(piece.data(), paged.data() + offset, piece.size());
        offset += piece.size();
      },
      paged.size(), valid_context);
  Expect(offset == paged.size() && b.GetKvBlocks().size() == full + 1,
         "a paged restore maps every block of its rows");
  for (std::uint32_t block = 0; block < full; ++block) {
    std::size_t read = 0;
    b.RestoreKvBlock(block, bounce, [&](std::span<std::uint8_t> piece) {
      std::memcpy(piece.data(), blocks[block].data() + read, piece.size());
      read += piece.size();
    });
    Expect(read == block_bytes, "a block restore reads one whole block");
  }
  HIP_CHECK(hipDeviceSynchronize());
  Expect(StreamLive(b, valid_context) == whole,
         "paged form plus blocks restore the whole state");

  bool refused = false;
  try {
    gufo::hip::QwenGpuArena whole_context(config, context, policy);
    offset = 0;
    (void)whole_context.RestoreCompactSnapshotStream(
        bounce,
        [&](std::span<std::uint8_t> piece) {
          std::memcpy(piece.data(), paged.data() + offset, piece.size());
          offset += piece.size();
        },
        paged.size(), valid_context);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  Expect(refused, "a whole-context arena must refuse a paged snapshot");
}

}  // namespace
#endif

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Qwen KV block table ops test");
  if (device_status != 0) {
    return device_status;
  }
  Fixture fixture;
  TestDecodeAndVerification(fixture);
  TestPrefill(fixture);
  TestArenaOnSharedPool();
  TestPoolReferencesAndReservations();
  TestOnDemandArena();
  TestPagedCompactRoundTrip();
  std::cout << "Qwen KV block table tests passed.\n";
#endif
  return 0;
}
