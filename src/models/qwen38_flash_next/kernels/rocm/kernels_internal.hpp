#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KERNELS_INTERNAL_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KERNELS_INTERNAL_HPP_

// Private to the kernels/rocm/*.hip.cpp translation units: the device helpers,
// layouts and launch constants more than one of them uses. They stay in the
// anonymous namespace, as when the kernels shared one file, so kernel names and
// code are unchanged. The slab rotation helpers are the exception: callers in
// different translation units would let interprocedural constant propagation
// specialize an internal copy for the one caller it sees, so they are inline
// functions of the model namespace, opaque to it as they were with all callers
// in one file.
// HIP kernels follow the layouts and operator formulas in reference.cpp.

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include "src/core/hip/detail/exl3_gemv.hpp"
#include "src/core/hip/detail/kvarn_device.hpp"
#include "src/core/hip/kv_block_table.hpp"
#include "src/core/hip/kvarn_layout.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/exl3_moe.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/exl3_prefill_combine.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

/// Consumer c's FP16 slab and suh of `slabs` (c < slabs.count; wave-uniform
/// selection, no dynamic indexing of the argument block).
struct SlabTarget {
  const _Float16* suh;
  _Float16* out;
};
__device__ __forceinline__ SlabTarget SlabOf(const Exl3InputSlabs& slabs,
                                             unsigned c) {
  const std::uint16_t* suh = slabs.suh[0];
  if (c == 1)
    suh = slabs.suh[1];
  else if (c == 2)
    suh = slabs.suh[2];
  return {reinterpret_cast<const _Float16*>(suh),
          reinterpret_cast<_Float16*>(slabs.out) +
              std::size_t{c} * hip::exl3::kMaxRows * slabs.k};
}

/// Rotates one 128-element segment (v[j] = element first + j, first =
/// segment * 128 + 4 * lane) of slab row `row` into every consumer's slab.
__device__ __forceinline__ void RotateIntoSlabs(const float (&v)[4],
                                                unsigned lane, unsigned first,
                                                unsigned row,
                                                const Exl3InputSlabs& slabs) {
  for (unsigned c = 0; c < slabs.count; ++c) {
    const SlabTarget target = SlabOf(slabs, c);
    float w[4] = {v[0], v[1], v[2], v[3]};
    hip::exl3::RotateInputSegment(w, lane, first, target.suh, target.out, row,
                                  slabs.pass_rows);
  }
}

namespace {

constexpr unsigned kThreads = 256;

/// Row of token position `pos` in one layer of a K/V plane: through the
/// session's block table, or `pos` itself when `kv_blocks` is null.
__device__ __forceinline__ std::size_t KvRow(const std::uint32_t* kv_blocks,
                                             std::uint32_t pos) {
  return hip::KvRowInLayer(hip::KvBlockTable{kv_blocks, 0}, pos);
}

/// Row a K/V writer stores position `pos` at: KvRow, or the KVarN FP16 sink
/// and tail ring row for kKvarnFp16RowTag (kernels.hpp).
__device__ __forceinline__ std::size_t KvWriteRow(
    const std::uint32_t* kv_blocks, std::uint32_t pos) {
  if (reinterpret_cast<std::uintptr_t>(kv_blocks) == kKvarnFp16RowTag)
    return hip::KvarnFp16Row(pos);
  return KvRow(kv_blocks, pos);
}

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0f / (1.0f + __expf(-x));
}
__device__ __forceinline__ float SoftplusF(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}
__device__ __forceinline__ float SiluF(float x) {
  return x * SigmoidF(x);
}
/// One block_q8_1 (mmq's decode activation format): d and the row sum, then
/// the 32 codes.
struct Q8_1Block {
  __half2 ds;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_1Block) == 36);

/// Keeps fast-math from folding a stored value's arithmetic into what
/// follows: the quantization must see exactly the float that was stored.
__device__ __forceinline__ float Materialize(float v) {
  asm volatile("" : "+v"(v));
  return v;
}

/// Lane i receives lane (i ^ offset)'s value, as __shfl_xor(v, offset) does
/// across a wave32. On gfx1151 it is a DPP row_xmask (offsets below 16) or
/// v_permlanex16 (16): a register move instead of an LDS crossbar round trip
/// (ds_bpermute). Only the transport differs, so reductions stay bitwise;
/// every lane must be active, as the shuffle's callers already require.
__device__ __forceinline__ int XorLane(int v, int offset) {
#if defined(__gfx1151__)
  switch (offset) {
    case 1:
      return __builtin_amdgcn_update_dpp(0, v, 0x161, 0xF, 0xF, false);
    case 2:
      return __builtin_amdgcn_update_dpp(0, v, 0x162, 0xF, 0xF, false);
    case 4:
      return __builtin_amdgcn_update_dpp(0, v, 0x164, 0xF, 0xF, false);
    case 8:
      return __builtin_amdgcn_update_dpp(0, v, 0x168, 0xF, 0xF, false);
    case 16:
      return __builtin_amdgcn_permlanex16(v, v, 0x76543210, 0xfedcba98, false,
                                          false);
  }
#endif
  return __shfl_xor(v, offset);
}

__device__ __forceinline__ unsigned XorLane(unsigned v, int offset) {
  return static_cast<unsigned>(XorLane(static_cast<int>(v), offset));
}

__device__ __forceinline__ float XorLane(float v, int offset) {
  return __builtin_bit_cast(float, XorLane(__builtin_bit_cast(int, v), offset));
}

/// mmq's quantize_q8_1 for element `i` of contiguous rows (row width a
/// multiple of 32), byte for byte: lane i % 32 of an aligned 32-lane group
/// holds element i, and all 32 lanes call this together. A producer that
/// stores `v` and calls this saves the separate quantization launch while
/// handing the projection identical codes.
__device__ __forceinline__ void QuantizeQ8_1Lane(float v, void* y,
                                                 std::size_t i) {
  const float xi = Materialize(v);
  float amax = fabsf(xi);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, XorLane(amax, offset));
  }
  const float d = amax / 127.0f;
  const std::int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
  // The block sum is d*sum(q), as in quantize_q8_1. The steps are spelled
  // out: after the divergent `q`, a `#pragma unroll` loop stays rolled and
  // dispatches XorLane's switch at run time, doubling HcCombineKernel.
  int sumq = q;
  sumq += XorLane(sumq, 16);
  sumq += XorLane(sumq, 8);
  sumq += XorLane(sumq, 4);
  sumq += XorLane(sumq, 2);
  sumq += XorLane(sumq, 1);
  Q8_1Block* blocks = static_cast<Q8_1Block*>(y);
  blocks[i / 32].qs[i % 32] = q;
  if (i % 32 == 0) {
    blocks[i / 32].ds = make_half2(d, d * sumq);
  }
}

__device__ __forceinline__ float Bf16ToF32(std::uint16_t h) {
  return __uint_as_float(static_cast<std::uint32_t>(h) << 16);
}

/// Reads element i of a Q8_0 / F32 / BF16 / F16 row.
__device__ __forceinline__ float RowElement(const void* row, WeightType type,
                                            std::uint32_t i) {
  switch (type) {
    case WeightType::kF32:
      return static_cast<const float*>(row)[i];
    case WeightType::kBF16:
      return Bf16ToF32(static_cast<const std::uint16_t*>(row)[i]);
    case WeightType::kF16:
      return __half2float(static_cast<const __half*>(row)[i]);
    case WeightType::kQ8_0: {
      const auto* blk = static_cast<const std::uint8_t*>(row) + (i / 32) * 34;
      const __half d = *reinterpret_cast<const __half*>(blk);
      const auto q = static_cast<const std::int8_t*>(
          static_cast<const void*>(blk + 2))[i % 32];
      return __half2float(d) * static_cast<float>(q);
    }
  }
  return 0.0f;
}

__device__ __forceinline__ std::size_t RowBytes(WeightType type,
                                                std::uint32_t k) {
  switch (type) {
    case WeightType::kF32:
      return static_cast<std::size_t>(k) * 4;
    case WeightType::kBF16:
    case WeightType::kF16:
      return static_cast<std::size_t>(k) * 2;
    case WeightType::kQ8_0:
      return static_cast<std::size_t>(k / 32) * 34;
  }
  return 0;
}

/// Sum over one wave; every lane receives the total. The wave-per-row
/// kernels split rows across lanes (gfx1151 runs wave32).
__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += XorLane(v, offset);
  }
  return v;
}

/// Block-wide sum over kThreads threads; every thread receives the total.
__device__ float BlockSum(float v, float* shared) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += XorLane(v, offset);
  }
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  __syncthreads();
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  float total = 0.0f;
  for (int w = 0; w < static_cast<int>(blockDim.x / warpSize); ++w) {
    total += shared[w];
  }
  return total;
}

__device__ float BlockMax(float v, float* shared) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, XorLane(v, offset));
  }
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  __syncthreads();
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  float total = -INFINITY;
  for (int w = 0; w < static_cast<int>(blockDim.x / warpSize); ++w) {
    total = fmaxf(total, shared[w]);
  }
  return total;
}

/// Four adjacent hidden lanes per thread. This keeps the stream arithmetic in
/// registers while cutting the number of blocks and inject partials by almost
/// four for the model's 2,560-wide, four-stream rows.
__device__ __forceinline__ float4 Load4(const float* p) {
  return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ float4 Load4(const __half* p) {
  const auto packed = *reinterpret_cast<const __half2*>(p);
  const auto packed_hi = *reinterpret_cast<const __half2*>(p + 2);
  const float2 lo = __half22float2(packed);
  const float2 hi = __half22float2(packed_hi);
  return float4{lo.x, lo.y, hi.x, hi.y};
}

/// The W8A8 GEMM's activation tiles: 16 tokens x one 32-wide K block,
/// codes in fragment order (two 256-byte halves of 16 tokens x 16 codes)
/// followed by the sixteen per-token scales.
constexpr std::size_t kQ8ActTileTokens = 16;
constexpr std::size_t kQ8ActTileBytes = 576;
constexpr std::size_t kQ8ActScaleOffset = 512;
__device__ __forceinline__ std::int8_t* Q8ActTile(void* base,
                                                  std::size_t num_blocks,
                                                  std::size_t tt,
                                                  std::size_t kb) {
  return static_cast<std::int8_t*>(base) +
         (((tt * num_blocks) + kb) * kQ8ActTileBytes);
}

/// SsmConvKernel with one thread per (channel, kTokensPerThread tokens) for
/// the 4-tap kernel: the taps stay in one float4 and the window slides in
/// registers, so a token costs one load and one store instead of eight
/// loads. Lanes run along channels, so every access is a contiguous row.
constexpr std::uint32_t kSsmConvTaps = 4;

// YarnRope travels to the kernels by value as a plain 20-byte argument.
// Every rope kernel is instantiated with and without YaRN (`kYarn`) and the
// launcher picks one on the host: a runtime branch in the default kernel let
// fast-math contract its rotation differently, so the default instantiation
// keeps the original code and only the YaRN one carries the extra work.
static_assert(std::is_trivially_copyable_v<YarnRope> && sizeof(YarnRope) == 20);

// YaRN frequency. The product is kept opaque so every rope site rounds the
// angle identically: fast-math may otherwise reassociate position * freq *
// multiplier differently per kernel and break cross-site byte identity.
__device__ __forceinline__ float YarnFrequency(float freq, std::uint32_t pair,
                                               const YarnRope& yarn) {
  float scaled = freq * YarnPairMultiplier(yarn, pair);
  asm volatile("" : "+v"(scaled));
  return scaled;
}

// YaRN attention factor on cos and sin, opaque for the same reason: the
// sites rotate with different expressions (explicit FMA or fast-math
// contraction), which must all see the same rounded scaled values.
__device__ __forceinline__ void YarnScale(float& sine, float& cosine,
                                          const YarnRope& yarn) {
  sine *= yarn.mscale;
  cosine *= yarn.mscale;
  asm volatile("" : "+v"(sine), "+v"(cosine));
}

constexpr auto kKvarnOrder = hip::KvarnValueOrder::kTokenMajor;
template<int Bits>
constexpr auto kKvarnRecord = hip::KvarnRecord(Bits, kKvarnOrder, Bits);

/// Record of global group `group` and KV head `kv` in one layer's plane.
template<int Bits>
__device__ __forceinline__ std::size_t KvarnRecordOffset(
    const std::uint32_t* kv_blocks, std::uint32_t kv_heads, std::uint32_t group,
    std::uint32_t kv) {
  const std::size_t plane_row =
      kv_blocks == nullptr
          ? group
          : std::size_t{kv_blocks[group / hip::kKvarnGroupsPerBlock]} *
                    hip::kKvarnGroupsPerBlock +
                group % hip::kKvarnGroupsPerBlock;
  return (plane_row * kv_heads + kv) * kKvarnRecord<Bits>.bytes;
}

/// Eight FP16 values, dims d0..d0+7 of KV head `kv`, of position j's K or V
/// row as a query at position `query` reads it (KvarnReadView). Records
/// dequantize as (code * scale + zero) * column in FP32, rounded once to
/// FP16 as the fake-quantized cache stores them.
template<int Bits>
__device__ __forceinline__ uint4 KvarnLoad8(const KvarnReadView& v, bool key,
                                            std::uint32_t query,
                                            std::uint32_t j, std::uint32_t kv,
                                            std::uint32_t d0) {
  const std::size_t row_width = std::size_t{v.kv_heads} * hip::kKvarnHeadDim;
  const auto fp16 = [&](const __half* base, std::size_t row) {
    return *reinterpret_cast<const uint4*>(base + row * row_width +
                                           kv * hip::kKvarnHeadDim + d0);
  };
  if (v.k_staging != nullptr && j >= v.staging_first)
    return fp16(key ? v.k_staging : v.v_staging, j - v.staging_first);
  if (j < hip::kKvarnSinkTokens)
    return fp16(key ? v.k_fp16 : v.v_fp16, j);
  const std::uint32_t end =
      v.quant_end != 0
          ? v.quant_end
          : hip::kKvarnGroupTokens * (1 + hip::KvarnReadGroups(query));
  if (j >= end)
    return fp16(key ? v.k_fp16 : v.v_fp16, hip::KvarnFp16Row(j));
  const std::uint32_t group = j >> hip::kKvarnGroupShift;
  const std::uint32_t t = j & (hip::kKvarnGroupTokens - 1);
  const std::uint8_t* record =
      v.records + KvarnRecordOffset<Bits>(v.blocks, v.kv_heads, group, kv);
  const std::uint32_t word = reinterpret_cast<const std::uint32_t*>(
      record +
      (key ? kKvarnRecord<Bits>.k_codes
           : kKvarnRecord<Bits>.v_codes))[t * (hip::kKvarnHeadDim * Bits / 32) +
                                          d0 / 8];
  unsigned high = 0;
  if constexpr (Bits == 6)
    high = reinterpret_cast<const std::uint16_t*>(
        record +
        (key ? kKvarnRecord<Bits>.k_codes : kKvarnRecord<Bits>.v_codes) +
        t * (hip::kKvarnHeadDim * Bits / 8) + hip::kKvarnHeadDim / 2)[d0 / 8];
  const auto* scale = reinterpret_cast<const __half*>(
      record + (key ? kKvarnRecord<Bits>.k_scale : kKvarnRecord<Bits>.v_scale));
  const auto* zero = reinterpret_cast<const __half*>(
      record + (key ? kKvarnRecord<Bits>.k_zero : kKvarnRecord<Bits>.v_zero));
  const auto* column = reinterpret_cast<const __half*>(
      record +
      (key ? kKvarnRecord<Bits>.k_column : kKvarnRecord<Bits>.v_column));
  _Float16 h[8];
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    // Channels c..c+7 of a word: c + 2i in nibble i, c + 2i + 1 in i + 4.
    const std::uint32_t nibble = (i % 2 == 0) ? i / 2 : 4 + i / 2;
    const auto code = static_cast<float>(((word >> (4 * nibble)) & 15U) |
                                         (((high >> (2 * i)) & 3U) << 4));
    // K: scale and zero per channel, column per token; V the other way.
    const float s = __half2float(key ? scale[d0 + i] : scale[t]);
    const float z = __half2float(key ? zero[d0 + i] : zero[t]);
    const float c = __half2float(key ? column[t] : column[d0 + i]);
    h[i] = static_cast<_Float16>((code * s + z) * c);
  }
  return *reinterpret_cast<const uint4*>(h);
}

// F16 WMMA fragments (wave32): sixteen halves per lane, eight F32
// accumulators per lane.
using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// BF16 fragments carry the same sixteen 16-bit lanes.
using v16bf = __attribute__((ext_vector_type(16))) __bf16;

__device__ __forceinline__ v8f WmmaBf16(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
      __builtin_bit_cast(v16bf, a), __builtin_bit_cast(v16bf, b), c);
}

inline unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

// A barrier for kernels whose waves exchange data through LDS only.
// __syncthreads also waits for every outstanding global load and drops the
// L0 cache, which serializes register prefetches behind the barrier. The
// global inputs are read-only here, so completing LDS traffic is enough.
__device__ __forceinline__ __attribute__((convergent)) void SyncLds() {
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  __builtin_amdgcn_s_barrier();
  asm volatile("" ::: "memory");
}

// W8A8 int8 WMMA GEMM over the GGUF Q8_0 weights, ported from the Qwen 27B
// route (src/models/qwen/hip/kernels/prefill_quant_gemm.hip,
// opt-c163-blocked-w8a8 with the opt-c179 addressing). Weights stay in their
// row-major 34-byte block_q8_0 layout; activations are quantized per 32-wide
// block into WMMA B-fragment order:
//
//   tile(tt, kb) = [b0: 16 tokens x 16 bytes]   offset   0
//                  [b1: 16 tokens x 16 bytes]   offset 256
//                  [16 fp32 token scales    ]   offset 512
//
// with tt = token / 16 and kb the 32-element K block, 576 bytes per tile.
struct Q8_0Block {
  __half d;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34, "block_q8_0 must be 34 bytes");

// Routed F16 WMMA expert GEMM. The int8 kernel above pays a float epilogue
// and an activation-sum correction every K block because the per-32 scales
// of both operands sit outside the integer dot product; here the weights are
// dequantized to F16 right after the LDS read (each wave decodes only its own
// sixteen rows) and the activations are F16 rows, so the matrix core
// accumulates the whole K extent in F32 with no per-block work. The codes
// stay packed in LDS (4 bits for Q4_K, 4 + 1 for Q5_1), which keeps a
// two-K-block stage at 11-12 KB and five blocks resident per WGP.
//
// A code becomes a half through a byte permute into the mantissa of 1024.0
// (0x6400 | q = 1024 + q exactly for q < 32, the half's unit being 1 there),
// a packed subtract of 1024 (exact), then one packed FMA:
//
//     w = q * scale + bias
//
// with (scale, bias) = (d * sc, -dmin * mn) for Q4_K and (d, m) for Q5_1,
// staged per (row, K block) as a half2.
//
// grid (m / BM, tiles): `tiles[y]` packs the expert in the low 16 bits and
// the token macro tile index in the high 16, so no block is launched for an
// empty tile; the row blocks of one tile are consecutive in dispatch order so
// they share the tile's gathered activations through L2. Block (x, y)
// computes rows x*BM.. of the expert against its
// compact rows [pad_bounds[e] + j*BN, +BN) and scatters them to
// out[rows_out[c]][row] (F32, or F16 with the SwiGLU applied when `out_half`
// is given: the up projection then writes the down projection's input).
constexpr std::uint32_t kHalfMagic = 0x64646464U;  // 1024.0 high bytes

/// Four halves from four code bytes: 1024 + q as F16, minus `magic` (1024,
/// or 1152 for a signed byte carried as q + 128), then the affine.
__device__ __forceinline__ void CodesToHalves(std::uint32_t codes,
                                              __half2 magic, __half2 scale2,
                                              __half2 bias2, __half2& lo,
                                              __half2& hi) {
  const std::uint32_t p0 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x01050004U);
  const std::uint32_t p1 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x03070206U);
  lo = __hfma2(__hadd2(__builtin_bit_cast(__half2, p0), magic), scale2, bias2);
  hi = __hfma2(__hadd2(__builtin_bit_cast(__half2, p1), magic), scale2, bias2);
}

}  // namespace

/// HcMixF16Gemm's inject logits only: HcMixEpilogueF16Kernel without the mix
/// outputs (hc.hip.cpp), for the GEMM in dense_gemm.hip.cpp.
void HcMixInjectF16(const __half* xn, const float* inject_w, float* inject,
                    std::uint32_t n_tokens, std::uint32_t hidden,
                    hipStream_t stream);

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KERNELS_INTERNAL_HPP_
