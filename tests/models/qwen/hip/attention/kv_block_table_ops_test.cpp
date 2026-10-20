#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/kv_block_pool.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"
#include "tests/testing/test_common.hpp"

namespace {

using gufo::hip::kKvBlockTokens;
using gufo::hip::KvBlockTable;
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
constexpr unsigned kPoolBlocks = kLogicalBlocks + 6;
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
  std::vector<__half> paged(
      std::size_t{kLayers} * kPoolBlocks * kKvBlockTokens * kKvWidth,
      __float2half(-7.0F));
  for (unsigned layer = 0; layer < kLayers; ++layer)
    for (unsigned position = 0; position < kContext; ++position)
      for (unsigned column = 0; column < kKvWidth; ++column) {
        const float value =
            0.4F *
            std::sin(static_cast<float>(
                         (layer * kContext + position) * kKvWidth + column) *
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
void ExpectSame(
    Fixture& fixture, std::size_t output_size,
    const std::function<void(__half* keys, __half* values, KvBlockTable table,
                             float* out)>& launch,
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
    HIP_CHECK(
        hipMemcpy(d_pos.data(), &pos, sizeof(pos), hipMemcpyHostToDevice));
    ExpectSame(
        fixture, kWidth,
        [&](__half* keys, __half* values, KvBlockTable table, float* out) {
          gufo::hip::LaunchAttention(q.data(), k.data(), v.data(), gate.data(),
                                     nullptr, nullptr, keys, values, table, out,
                                     kLayer, d_pos.data(), kContext, kHeads,
                                     kKvHeads, kDim, nullptr, false);
        },
        "device-position decode at " + std::to_string(pos));
  }
  for (unsigned pos : {3U, 64U, 1000U}) {
    HIP_CHECK(
        hipMemcpy(d_pos.data(), &pos, sizeof(pos), hipMemcpyHostToDevice));
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
                       nullptr, 0, !write, {packed_k.data(), packed_k.size()},
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
                c.start, c.batch, kContext, kHeads, kKvHeads, kDim, 64, 1.0e7F);
          },
          "batched fused write " + where);
    }
  }
}

// Positions past 200K at block boundaries: row arithmetic through the table
// stays 64-bit. One layer of the production row width over 262144 positions,
// placed in a pool of scattered blocks.
void TestLargePositions() {
  constexpr unsigned context = 262144;
  constexpr unsigned logical_blocks = context / kKvBlockTokens;
  constexpr unsigned pool_blocks = logical_blocks + 3;
  const auto physical = [](unsigned logical) {
    return (logical * 7U + 3U) % pool_blocks;
  };
  const auto paged_index = [&](unsigned position) {
    return (static_cast<std::size_t>(physical(position / kKvBlockTokens)) *
                kKvBlockTokens +
            position % kKvBlockTokens) *
           kKvWidth;
  };
  // Cheap, finite, position-dependent values (sin over 2^28 elements is slow).
  const auto value = [](std::size_t i, unsigned salt) {
    const auto h = static_cast<std::uint32_t>(i * 2654435761U + salt);
    return __float2half((static_cast<float>(h >> 22) / 1024.0F - 0.5F) * 0.8F);
  };
  std::vector<DeviceBuffer<__half>> contiguous, paged;
  for (unsigned plane = 0; plane < 2; ++plane) {
    std::vector<__half> c(std::size_t{context} * kKvWidth);
    std::vector<__half> p(std::size_t{pool_blocks} * kKvBlockTokens * kKvWidth,
                          __float2half(-7.0F));
    for (unsigned position = 0; position < context; ++position)
      for (unsigned column = 0; column < kKvWidth; ++column) {
        const auto x =
            value(std::size_t{position} * kKvWidth + column, plane * 977U);
        c[std::size_t{position} * kKvWidth + column] = x;
        p[paged_index(position) + column] = x;
      }
    contiguous.emplace_back(c);
    paged.emplace_back(p);
  }
  std::vector<std::uint32_t> blocks(logical_blocks);
  for (unsigned logical = 0; logical < logical_blocks; ++logical)
    blocks[logical] = physical(logical);
  const DeviceBuffer<std::uint32_t> table_buffer(blocks);
  const KvBlockTable table{table_buffer.data(), pool_blocks * kKvBlockTokens};

  const DeviceBuffer<float> q(Wave(8 * kWidth, 0.073F, 0.3F));
  const DeviceBuffer<float> gate(Wave(8 * kWidth, 0.019F, 1.0F));
  const DeviceBuffer<float> k(Wave(8 * kKvWidth, 0.031F, 0.5F));
  const DeviceBuffer<float> v(Wave(8 * kKvWidth, 0.047F, 0.5F));
  const auto row_scratch =
      gufo::hip::detail::DecodeAttentionScratchElements(kHeads, kDim);
  DeviceBuffer<float> scratch(8 * row_scratch);
  const auto same = [&](std::size_t output_size, auto&& launch,
                        std::span<const unsigned> written,
                        const std::string& what) {
    DeviceBuffer<float> a(output_size), b(output_size);
    launch(contiguous[0].data(), contiguous[1].data(), KvBlockTable{},
           a.data());
    launch(paged[0].data(), paged[1].data(), table, b.data());
    HIP_CHECK(hipDeviceSynchronize());
    const auto expected = a.CopyToHost();
    const auto actual = b.CopyToHost();
    Expect(std::memcmp(expected.data(), actual.data(),
                       output_size * sizeof(float)) == 0,
           what + ": paged attention changed output bits");
    for (unsigned plane = 0; plane < 2; ++plane)
      for (const unsigned position : written) {
        std::vector<__half> x(kKvWidth), y(kKvWidth);
        HIP_CHECK(hipMemcpy(
            x.data(),
            contiguous[plane].data() + std::size_t{position} * kKvWidth,
            kKvWidth * sizeof(__half), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(y.data(),
                            paged[plane].data() + paged_index(position),
                            kKvWidth * sizeof(__half), hipMemcpyDeviceToHost));
        Expect(std::memcmp(x.data(), y.data(), kKvWidth * sizeof(__half)) == 0,
               what + ": paged write landed in another row");
      }
  };
  for (unsigned pos : {200000U, 204799U, 204800U, context - 1}) {
    for (bool write : {false, true}) {
      const unsigned written[] = {pos};
      same(
          kWidth,
          [&](__half* keys, __half* values, KvBlockTable t, float* out) {
            gufo::hip::LaunchAttention(
                q.data(), k.data(), v.data(), gate.data(), nullptr, nullptr,
                keys, values, t, out, 0, pos, context, kHeads, kKvHeads, kDim,
                nullptr, scratch.data(), !write);
          },
          write ? std::span<const unsigned>(written)
                : std::span<const unsigned>{},
          "large decode at " + std::to_string(pos));
    }
  }
  // Eight verification rows across the 230144 block boundary.
  same(
      8 * kWidth,
      [&](__half* keys, __half* values, KvBlockTable t, float* out) {
        gufo::hip::LaunchCausalDecodeAttention(
            q.data(), gate.data(), nullptr, nullptr, keys, values, t, out, 0,
            230140, 8, context, kHeads, kKvHeads, kDim, nullptr,
            {scratch.data(), scratch.size()});
      },
      {}, "large verification rows at 230140");
  // Batched attention past the LDS score budget takes the online route.
  for (bool write : {false, true}) {
    const unsigned written[] = {230140, 230143, 230144, 230147};
    same(
        8 * kWidth,
        [&](__half* keys, __half* values, KvBlockTable t, float* out) {
          gufo::hip::LaunchBatchedAttention(
              q.data(), k.data(), v.data(), gate.data(), nullptr, nullptr, keys,
              values, t, out, 0, 230140, 8, context, kHeads, kKvHeads, kDim,
              nullptr, !write);
        },
        write ? std::span<const unsigned>(written)
              : std::span<const unsigned>{},
        "large batched rows at 230140");
  }
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
  TestLargePositions();
  std::cout << "Qwen KV block table tests passed.\n";
#endif
  return 0;
}
