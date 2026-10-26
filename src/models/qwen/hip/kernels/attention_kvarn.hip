#if defined(ENGINE_ENABLE_HIP)
// KVarN attention KV kernels for Qwen3.8-27B (kvarn_layout.hpp): the
// rotating row write, the group-flush quantizer and the fused-dequant split-K
// decode/verification attention. Promoted from the K1b prototype
// (tools/bench/kvarn_decode_bench.hip), with the storage of the block pool
// and the per-sequence FP16 sink and tail ring.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops/kvarn.hpp"

namespace gufo::hip {
namespace {

constexpr unsigned kHeads = kKvarnQueryHeads, kKvHeads = kKvarnKvHeads;
constexpr unsigned kGqa = kHeads / kKvHeads, kDim = kKvarnHeadDim;
constexpr unsigned kGroup = kKvarnGroupTokens, kRowWidth = kKvHeads * kDim;
constexpr unsigned kPartialStride = kDim + 2;
constexpr std::size_t kFp16LayerElements =
    static_cast<std::size_t>(kKvarnFp16Tokens) * kRowWidth;

// Rows of a position: the sink and tail ring (kRingRows), or a contiguous
// staging buffer whose row 0 holds position `staging_first`.
constexpr unsigned kRingRows = ~0U;
__host__ __device__ __forceinline__ std::size_t RowOf(unsigned position,
                                                      unsigned staging_first) {
  return staging_first == kRingRows ? KvarnFp16Row(position)
                                    : position - staging_first;
}

template<int VBits>
struct Record {
  static_assert(VBits == 2 || VBits == 4);
  static constexpr KvarnRecordLayout kLayout = KvarnRecord(VBits);
  static constexpr std::size_t kKCodes = kLayout.k_codes;
  static constexpr std::size_t kVCodes = kLayout.v_codes;
  static constexpr std::size_t kVRowBytes = kLayout.v_row_bytes;
  static constexpr std::size_t kKScale = kLayout.k_scale;
  static constexpr std::size_t kKZero = kLayout.k_zero;
  static constexpr std::size_t kKColumn = kLayout.k_column;
  static constexpr std::size_t kVScale = kLayout.v_scale;
  static constexpr std::size_t kVZero = kLayout.v_zero;
  static constexpr std::size_t kVColumn = kLayout.v_column;
  static constexpr std::size_t kBytes = kLayout.bytes;
};

// Byte offset, in one layer of the record plane, of global group `group`'s
// record for KV head `kv`: plane row (pool block * 2 + group in block).
template<int VBits>
__device__ __forceinline__ std::size_t RecordOffset(const std::uint32_t* blocks,
                                                    unsigned group,
                                                    unsigned kv) {
  const std::size_t row =
      static_cast<std::size_t>(blocks[group >> 1]) * kKvarnGroupsPerBlock +
      (group & 1);
  return (row * kKvHeads + kv) * Record<VBits>::kBytes;
}

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using u32x8 = __attribute__((__vector_size__(8 * sizeof(unsigned)))) unsigned;
typedef _Float16 h2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ v16h LoadFrag(const _Float16* p) {
  const uint4 lo = *reinterpret_cast<const uint4*>(p);
  const uint4 hi = *reinterpret_cast<const uint4*>(p + 8);
  const u32x8 w = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
  return __builtin_bit_cast(v16h, w);
}

// Two codes already masked into the low bits of each 16-bit half -> packed
// halves code * scale + zero. 0x6400 | c is the half 1024 + c, exactly.
__device__ __forceinline__ unsigned Dequant2(unsigned codes, unsigned scale,
                                             unsigned zero) {
  const h2 bias = {static_cast<_Float16>(1024.0F),
                   static_cast<_Float16>(1024.0F)};
  const h2 value = __builtin_bit_cast(h2, codes | 0x64006400U) - bias;
  return __builtin_bit_cast(
      unsigned, __builtin_elementwise_fma(value, __builtin_bit_cast(h2, scale),
                                          __builtin_bit_cast(h2, zero)));
}

// Two masked codes -> packed halves equal to the codes.
__device__ __forceinline__ unsigned Codes2(unsigned codes) {
  const h2 bias = {static_cast<_Float16>(1024.0F),
                   static_cast<_Float16>(1024.0F)};
  return __builtin_bit_cast(unsigned,
                            __builtin_bit_cast(h2, codes | 0x64006400U) - bias);
}

// In-place unnormalized Walsh-Hadamard transform of a 256-vector held by a
// wave32, lane l owning elements 8l..8l+7. Stages run in the CPU reference's
// order with the same single add or subtract per butterfly, so the result
// is bit-identical to kvarn::Hadamard before its 1/16 scale.
__device__ __forceinline__ void Fwht256(float (&y)[8], unsigned lane) {
#pragma unroll
  for (unsigned length = 1; length < 8; length <<= 1)
#pragma unroll
    for (unsigned i = 0; i < 8; i += length << 1)
#pragma unroll
      for (unsigned j = i; j < i + length; ++j) {
        const float a = y[j], b = y[j + length];
        y[j] = a + b;
        y[j + length] = a - b;
      }
#pragma unroll
  for (unsigned mask = 1; mask < 32; mask <<= 1)
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
      const float other = __shfl_xor(y[j], mask);
      y[j] = (lane & mask) != 0 ? other - y[j] : y[j] + other;
    }
}

// ---------------------------------------------------------------------------
// Rotating row write. One 256-thread block per row; wave w handles KV head
// w / 2, K for even w and V for odd w, lane l elements 8l..8l+7.
// ---------------------------------------------------------------------------
__launch_bounds__(256) __global__
    void KvarnWriteRowsKernel(const float* __restrict__ k,
                              const float* __restrict__ v,
                              __half* __restrict__ k_rows,
                              __half* __restrict__ v_rows,
                              const std::uint32_t* __restrict__ d_pos,
                              unsigned start_pos, unsigned staging_first) {
  const unsigned row = blockIdx.x;
  const unsigned wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const unsigned kv = wave / 2;
  const bool key = (wave & 1) == 0;
  const unsigned position = (d_pos != nullptr ? *d_pos : start_pos) + row;
  const float* src = (key ? k : v) +
                     (static_cast<std::size_t>(row) * kKvHeads + kv) * kDim +
                     8 * lane;
  const float4 lo = reinterpret_cast<const float4*>(src)[0];
  const float4 hi = reinterpret_cast<const float4*>(src)[1];
  const float in[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
  float y[8];
#pragma unroll
  for (unsigned j = 0; j < 8; ++j)
    y[j] = __half2float(__float2half_rn(in[j]));
  Fwht256(y, lane);
  _Float16 h[8];
#pragma unroll
  for (unsigned j = 0; j < 8; ++j)
    h[j] = static_cast<_Float16>(y[j] * 0.0625F);
  __half* dst = (key ? k_rows : v_rows) +
                RowOf(position, staging_first) * kRowWidth + kv * kDim +
                8 * lane;
  *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h);
}

// ---------------------------------------------------------------------------
// Group-flush quantizer. One 512-thread block per (KV head, group, layer,
// K or V) tile of rotated FP16 rows from the tail ring. Wave w holds tokens
// 8w..8w+7 and lane l channels 8l..8l+7, so per-token statistics reduce inside
// a wave and per-channel ones across the 16 waves through LDS. Standard
// deviations accumulate in double like the reference; log and exp go through
// double and round once, as the K0 reference
// (tests/models/qwen27b/kvarn_fake_quant.hpp) does.
// ---------------------------------------------------------------------------
template<int VBits>
__launch_bounds__(512) __global__ void KvarnQuantizeKernel(
    const __half* __restrict__ k_rows, const __half* __restrict__ v_rows,
    std::uint8_t* __restrict__ records, std::size_t records_layer_pitch,
    const std::uint32_t* __restrict__ blocks, unsigned first_group,
    std::size_t source_layer_elements, unsigned staging_first) {
  using R = Record<VBits>;
  constexpr unsigned kWaves = 16;
  __shared__ double channel_part[kWaves][kDim];
  __shared__ double channel_result[kDim];
  __shared__ float block_part[kWaves][4];
  __shared__ unsigned value_codes[kDim * kGroup * VBits / 32];
  __shared__ float shared_log[kDim], shared_scale[kDim];

  const unsigned kv = blockIdx.x;
  const unsigned group = first_group + blockIdx.y;
  const unsigned layer = blockIdx.z >> 1;
  const bool key = (blockIdx.z & 1) == 0;
  const unsigned tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
  const __half* src = (key ? k_rows : v_rows) + layer * source_layer_elements +
                      RowOf(group * kGroup, staging_first) * kRowWidth +
                      kv * kDim;
  std::uint8_t* record = records + layer * records_layer_pitch +
                         RecordOffset<VBits>(blocks, group, kv);

  // [token i][channel pair]: the rotated FP16 source, packed.
  h2 x[8][4];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) {
    const uint4 raw = *reinterpret_cast<const uint4*>(
        src + static_cast<std::size_t>(8 * wave + i) * kRowWidth + 8 * lane);
    x[i][0] = __builtin_bit_cast(h2, raw.x);
    x[i][1] = __builtin_bit_cast(h2, raw.y);
    x[i][2] = __builtin_bit_cast(h2, raw.z);
    x[i][3] = __builtin_bit_cast(h2, raw.w);
  }
  const auto source = [&](unsigned i, unsigned j) {
    return static_cast<float>(x[i][j / 2][j % 2]);
  };
  if (!key)
    for (unsigned i = tid; i < sizeof(value_codes) / 4; i += 512)
      value_codes[i] = 0;

  float cur[8][8];
  float token_scale[8], channel_scale[8];
  float token_log[8], channel_log[8];
  float best_token[8], best_channel[8];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) {
    token_scale[i] = channel_scale[i] = 1.0F;
    token_log[i] = channel_log[i] = 0.0F;
    best_token[i] = best_channel[i] = 1.0F;
#pragma unroll
    for (unsigned j = 0; j < 8; ++j)
      cur[i][j] = source(i, j);
  }

  const auto wave_sum = [](double v) {
#pragma unroll
    for (unsigned mask = 1; mask < 32; mask <<= 1)
      v += __shfl_xor(v, mask);
    return v;
  };
  // Unbiased std of every token (over 256 channels), wave-uniform.
  const auto token_std = [&](float (&out)[8]) {
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
      double s = 0;
#pragma unroll
      for (unsigned j = 0; j < 8; ++j)
        s += cur[i][j];
      const double mean = wave_sum(s) / kDim;
      double d2 = 0;
#pragma unroll
      for (unsigned j = 0; j < 8; ++j) {
        // Rounded product then sum, as the reference (no contraction).
        const double d = cur[i][j] - mean;
        d2 = __dadd_rn(d2, __dmul_rn(d, d));
      }
      out[i] = static_cast<float>(sqrt(wave_sum(d2) / (kDim - 1)));
    }
  };
  // Unbiased std of this lane's 8 channels (over 128 tokens).
  const auto channel_std = [&](float (&out)[8]) {
    double mean[8];
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
      double s = 0;
#pragma unroll
      for (unsigned i = 0; i < 8; ++i)
        s += cur[i][j];
      channel_part[wave][8 * lane + j] = s;
    }
    __syncthreads();
    if (tid < kDim) {
      double s = 0;
      for (unsigned w = 0; w < kWaves; ++w)
        s += channel_part[w][tid];
      channel_result[tid] = s / kGroup;
    }
    __syncthreads();
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
      mean[j] = channel_result[8 * lane + j];
      double d2 = 0;
#pragma unroll
      for (unsigned i = 0; i < 8; ++i) {
        const double d = cur[i][j] - mean[j];
        d2 = __dadd_rn(d2, __dmul_rn(d, d));
      }
      channel_part[wave][8 * lane + j] = d2;
    }
    __syncthreads();
    if (tid < kDim) {
      double s = 0;
      for (unsigned w = 0; w < kWaves; ++w)
        s += channel_part[w][tid];
      channel_result[tid] = sqrt(s / (kGroup - 1));
    }
    __syncthreads();
#pragma unroll
    for (unsigned j = 0; j < 8; ++j)
      out[j] = static_cast<float>(channel_result[8 * lane + j]);
  };
  // column / max(column min) + row / max(row min); K columns are tokens.
  const auto imbalance = [&](const float (&tstd)[8], const float (&cstd)[8]) {
    float tmax = 0, tmin = INFINITY, cmax = 0, cmin = INFINITY;
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
      tmax = fmaxf(tmax, tstd[i]);
      tmin = fminf(tmin, tstd[i]);
      cmax = fmaxf(cmax, cstd[i]);
      cmin = fminf(cmin, cstd[i]);
    }
#pragma unroll
    for (unsigned mask = 1; mask < 32; mask <<= 1) {
      cmax = fmaxf(cmax, __shfl_xor(cmax, mask));
      cmin = fminf(cmin, __shfl_xor(cmin, mask));
    }
    if (lane == 0) {
      block_part[wave][0] = tmax;
      block_part[wave][1] = tmin;
      block_part[wave][2] = cmax;
      block_part[wave][3] = cmin;
    }
    __syncthreads();
    for (unsigned w = 0; w < kWaves; ++w) {
      tmax = fmaxf(tmax, block_part[w][0]);
      tmin = fminf(tmin, block_part[w][1]);
      cmax = fmaxf(cmax, block_part[w][2]);
      cmin = fminf(cmin, block_part[w][3]);
    }
    __syncthreads();
    const float token_term = tmax / fmaxf(tmin, 1e-8F);
    const float channel_term = cmax / fmaxf(cmin, 1e-8F);
    return key ? token_term + channel_term : channel_term + token_term;
  };
  // source / column scale / row scale, in the reference's division order.
  const auto apply = [&](const float (&ts)[8], const float (&cs)[8]) {
#pragma unroll
    for (unsigned i = 0; i < 8; ++i)
#pragma unroll
      for (unsigned j = 0; j < 8; ++j)
        cur[i][j] =
            key ? source(i, j) / ts[i] / cs[j] : source(i, j) / cs[j] / ts[i];
  };
  const auto update = [](float& log_scale, float s) {
    const float clamped = fminf(fmaxf(s, 1e-3F), 1e3F);
    log_scale = fminf(
        fmaxf(log_scale + static_cast<float>(log(static_cast<double>(clamped))),
              -0.3F),
        10.0F);
    return static_cast<float>(exp(static_cast<double>(log_scale)));
  };
  // Every lane holds the same eight token scales and every wave the same
  // eight channel scales per lane, so each update runs once and is shared:
  // lane i of a wave updates its token i, wave j (< 8) channel 8l + j. The
  // values are those every lane computed before, bit for bit; the double
  // log and exp were most of the quantizer's time.
  const auto update_tokens = [&](const float (&stds)[8]) {
    float s = stds[0], l = token_log[0];
#pragma unroll
    for (unsigned i = 1; i < 8; ++i)
      if ((lane & 7) == i) {
        s = stds[i];
        l = token_log[i];
      }
    const float scale = update(l, s);
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
      token_log[i] = __shfl(l, i);
      token_scale[i] = __shfl(scale, i);
    }
  };
  const auto update_channels = [&](const float (&stds)[8]) {
    if (wave < 8) {
      float s = stds[0], l = channel_log[0];
#pragma unroll
      for (unsigned j = 1; j < 8; ++j)
        if (wave == j) {
          s = stds[j];
          l = channel_log[j];
        }
      const float scale = update(l, s);
      shared_log[8 * lane + wave] = l;
      shared_scale[8 * lane + wave] = scale;
    }
    __syncthreads();
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
      channel_log[j] = shared_log[8 * lane + j];
      channel_scale[j] = shared_scale[8 * lane + j];
    }
    __syncthreads();
  };

  float tstd[8], cstd[8];
  token_std(tstd);
  channel_std(cstd);
  float best = imbalance(tstd, cstd);
  for (int iteration = 0; iteration < 8; ++iteration) {
    // Column step (K: tokens, V: channels) on the current tile.
    if (key)
      update_tokens(tstd);
    else
      update_channels(cstd);
    apply(token_scale, channel_scale);
    if (key) {
      channel_std(cstd);
      update_channels(cstd);
    } else {
      token_std(tstd);
      update_tokens(tstd);
    }
    apply(token_scale, channel_scale);
    token_std(tstd);
    channel_std(cstd);
    const float value = imbalance(tstd, cstd);
    if (value <= best) {
      best = value;
#pragma unroll
      for (unsigned i = 0; i < 8; ++i) {
        best_token[i] = token_scale[i];
        best_channel[i] = channel_scale[i];
      }
    }
  }
  apply(best_token, best_channel);

  // Per-row asymmetric RTN; K rows are channels, V rows are tokens.
  constexpr float kKeyMax = 15.0F;
  constexpr float kValueMax = static_cast<float>((1 << VBits) - 1);
  const auto halves = [&](std::size_t offset) {
    return reinterpret_cast<__half*>(record + offset);
  };
  // The reference rounds the FP32 product, then to FP16. Left alone, the
  // compiler fuses both into one v_fma_mix rounding, which moved about one
  // stored scale in 40 tiles by an FP16 ulp.
  const auto product_half = [](float a, float b) {
    float p = a * b;
    asm volatile("" : "+v"(p));
    return __float2half(p);
  };
  if (key) {
    auto* part = reinterpret_cast<float2*>(&channel_part[0][0]);
    auto* result = reinterpret_cast<float2*>(&channel_result[0]);
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
      float lo = cur[0][j], hi = cur[0][j];
#pragma unroll
      for (unsigned i = 1; i < 8; ++i) {
        lo = fminf(lo, cur[i][j]);
        hi = fmaxf(hi, cur[i][j]);
      }
      part[wave * kDim + 8 * lane + j] = make_float2(lo, hi);
    }
    __syncthreads();
    if (tid < kDim) {
      float2 r = part[tid];
      for (unsigned w = 1; w < kWaves; ++w) {
        r.x = fminf(r.x, part[w * kDim + tid].x);
        r.y = fmaxf(r.y, part[w * kDim + tid].y);
      }
      result[tid] = r;
    }
    __syncthreads();
    float low[8], step[8];
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
      const float2 r = result[8 * lane + j];
      low[j] = r.x;
      step[j] = fmaxf((r.y - r.x) / kKeyMax, 1e-10F);
    }
    auto* codes = reinterpret_cast<unsigned*>(record + R::kKCodes);
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
      unsigned word = 0;
#pragma unroll
      for (unsigned j = 0; j < 8; ++j) {
        const float q =
            fminf(fmaxf(rintf((cur[i][j] - low[j]) / step[j]), 0.0F), kKeyMax);
        const unsigned nibble = (j % 2 == 0) ? j / 2 : 4 + j / 2;
        word |= static_cast<unsigned>(q) << (4 * nibble);
      }
      codes[(8 * wave + i) * 32 + lane] = word;
    }
    if (wave == 0)
#pragma unroll
      for (unsigned j = 0; j < 8; ++j) {
        halves(R::kKScale)[8 * lane + j] =
            product_half(best_channel[j], step[j]);
        halves(R::kKZero)[8 * lane + j] = product_half(best_channel[j], low[j]);
      }
    if (lane == 0)
#pragma unroll
      for (unsigned i = 0; i < 8; ++i)
        halves(R::kKColumn)[8 * wave + i] = __float2half(best_token[i]);
  } else {
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
      float lo = cur[i][0], hi = cur[i][0];
#pragma unroll
      for (unsigned j = 1; j < 8; ++j) {
        lo = fminf(lo, cur[i][j]);
        hi = fmaxf(hi, cur[i][j]);
      }
#pragma unroll
      for (unsigned mask = 1; mask < 32; mask <<= 1) {
        lo = fminf(lo, __shfl_xor(lo, mask));
        hi = fmaxf(hi, __shfl_xor(hi, mask));
      }
      const float step = fmaxf((hi - lo) / kValueMax, 1e-10F);
      const unsigned token = 8 * wave + i;
#pragma unroll
      for (unsigned j = 0; j < 8; ++j) {
        const unsigned q = static_cast<unsigned>(
            fminf(fmaxf(rintf((cur[i][j] - lo) / step), 0.0F), kValueMax));
        const unsigned channel = 8 * lane + j;
        if constexpr (VBits == 2) {
          const unsigned u = token % 16;
          const unsigned crumb = (u % 2 == 0) ? u / 2 : 8 + u / 2;
          atomicOr(&value_codes[channel * 8 + token / 16], q << (2 * crumb));
        } else {
          const unsigned u = token % 8;
          const unsigned nibble = (u % 2 == 0) ? u / 2 : 4 + u / 2;
          atomicOr(&value_codes[channel * 16 + token / 8], q << (4 * nibble));
        }
      }
      if (lane == 0) {
        halves(R::kVScale)[token] = product_half(best_token[i], step);
        halves(R::kVZero)[token] = product_half(best_token[i], lo);
      }
    }
    if (wave == 0)
#pragma unroll
      for (unsigned j = 0; j < 8; ++j)
        halves(R::kVColumn)[8 * lane + j] = __float2half(best_channel[j]);
    __syncthreads();
    auto* codes = reinterpret_cast<uint4*>(record + R::kVCodes);
    const auto* staged = reinterpret_cast<const uint4*>(value_codes);
    for (unsigned i = tid; i < sizeof(value_codes) / 16; i += 512)
      codes[i] = staged[i];
  }
}

// ---------------------------------------------------------------------------
// Fused-dequant split-K decode attention partials (the K1b-opt2 `common`
// kernel, KVARN.md "K1b-opt2 결과"). One 256-thread block per KV head and
// split serves all 6 * Rows query rows of that KV head (row m = query row *
// 6 + head in group), padded to 16-row WMMA tiles. Splits [0, splits) take
// balanced runs of quantized groups, then two blocks the FP16 sink
// (positions [0, 64) and [64, 128)), then one block per 64 FP16 tail
// positions from the end of the quantized groups. Per chunk: wave w
// computes the scores of tokens 16w..16w+15 with WMMA (B from dequantized
// K), the block runs the online softmax in base 2 (P as FP16 in LDS), and
// wave w accumulates PV for channels 16w.. and 16(w+8)..
//
// Arithmetic rules that make a verification row add exactly what a decode
// step at its position adds:
//   * FP16 sink and tail PV pairs hold one token and a zero: a WMMA k pair
//     with one live A element and one zero lets the zero's B partner change
//     the result (wmmaprobe, K1b-opt2), and a causal cut splits pairs.
//   * The block structure depends on positions only: the quantized groups
//     a row reads (KvarnReadGroups), their split count
//     (SelectKvarnDecodeSplits), fixed sink blocks and 64-token tail blocks;
//     extra fully masked blocks of a wider launch add exact zeros.
//   * Softmax and combine updates are explicit fmaf, contraction is off.
// V codes become WMMA halves bias + code with one shift and one v_and_or per
// pair (BiasedCodes); the bias is subtracted in FP32 inside the P * zero
// sum. Padding rows of a tile are skipped. AR's last block of each KV head
// combines the splits; verification combines in KvarnCombineKernel.
// ---------------------------------------------------------------------------
constexpr float kLog2e = 1.4426950408889634F;
constexpr unsigned kTailBlockTokens = 64;
constexpr unsigned kSinkBlocks = kKvarnSinkTokens / kTailBlockTokens;

// Bias of pair j's code half (see BiasedCodes).
template<int Bits>
__host__ __device__ constexpr float CodeBias(unsigned pair) {
  if constexpr (Bits == 2)
    return (pair == 2 || pair == 4 || pair == 6) ? 4.0F : 16.0F;
  else
    return 16.0F;
}

// One 32-bit code word -> its pairs as packed halves bias + code. A code c
// masked at mantissa bit s, ORed with the exponent of 2^(10 - s), is the
// half 2^(10 - s) + c exactly; codes are moved to s = 6 (bias 16) or, for
// 2-bit codes, s = 8 (bias 4). Pair j is the codes of nibble (crumb) j of
// both 16-bit halves; the shifts never move a bit of one half into the
// masked field of the other.
template<int Bits>
__device__ __forceinline__ void BiasedCodes(unsigned w, unsigned* out) {
  if constexpr (Bits == 2) {
    constexpr unsigned k6 = 0x00C000C0U, k8 = 0x03000300U;
    constexpr unsigned e6 = 0x4C004C00U, e8 = 0x44004400U;  // 16 + c, 4 + c
    const unsigned l6 = w << 6, l4 = w << 4, r4 = w >> 4, r8 = w >> 8;
    out[0] = (l6 & k6) | e6;  // crumb 0: bits 0-1 -> 6-7
    out[1] = (l4 & k6) | e6;  // crumb 1: 2-3 -> 6-7
    out[2] = (l4 & k8) | e8;  // crumb 2: 4-5 -> 8-9
    out[3] = (w & k6) | e6;   // crumb 3 in place
    out[4] = (w & k8) | e8;   // crumb 4 in place
    out[5] = (r4 & k6) | e6;  // crumb 5: 10-11 -> 6-7
    out[6] = (r4 & k8) | e8;  // crumb 6: 12-13 -> 8-9
    out[7] = (r8 & k6) | e6;  // crumb 7: 14-15 -> 6-7
  } else {
    constexpr unsigned k = 0x03C003C0U, e = 0x4C004C00U;  // 16 + c
    out[0] = ((w << 6) & k) | e;
    out[1] = ((w << 2) & k) | e;
    out[2] = ((w >> 2) & k) | e;
    out[3] = ((w >> 6) & k) | e;
  }
}

struct DecodeArgs {
  const float* q;               // [rows][24][256], original frame
  const float* gate;            // [rows][24][256] or null
  float* out;                   // [rows][24][256]
  const std::uint8_t* records;  // this layer's record plane
  const std::uint32_t* blocks;  // logical block -> pool block
  const __half* k_rows;         // this layer's FP16 plane, rotated
  const __half* v_rows;
  float* scratch;              // [kv][split][row m][max, sum, 256 outputs]
  unsigned* counters;          // AR: one zeroed arrival counter per KV head
  const std::uint32_t* d_pos;  // first query position, or null
  unsigned start_pos;          // first query position without d_pos
  unsigned groups;             // quantized groups after the sink
  unsigned splits;             // quantized splits
  unsigned tail_blocks;        // 64-token FP16 tail blocks
};

__host__ __device__ constexpr unsigned SplitCount(const DecodeArgs& a) {
  return a.splits + kSinkBlocks + a.tail_blocks;
}

// Barrier for LDS traffic only: global loads in flight stay in flight.
// __syncthreads() would also drain them (s_waitcnt vmcnt(0)), which stops a
// block from overlapping the next group's loads with this group's math.
__device__ __forceinline__ void LdsBarrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

// XOR exchange inside a 16-lane DPP row (row_xmask), on the ALU path.
template<int kMask>
__device__ __forceinline__ float XorDpp(float x) {
  return __builtin_bit_cast(
      float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x),
                                         0x160 | kMask, 0xF, 0xF, false));
}
__device__ __forceinline__ float RowMax16(float x) {
  x = fmaxf(x, XorDpp<8>(x));
  x = fmaxf(x, XorDpp<4>(x));
  x = fmaxf(x, XorDpp<2>(x));
  return fmaxf(x, XorDpp<1>(x));
}
__device__ __forceinline__ float RowSum16(float x) {
  x += XorDpp<8>(x);
  x += XorDpp<4>(x);
  x += XorDpp<2>(x);
  return x + XorDpp<1>(x);
}

// Orders a k step's WMMAs before the next step's: the intrinsics are pure,
// and the IR otherwise regroups them per accumulator, keeping every k step's
// B fragment live at once (128 VGPRs for 16 steps).
template<unsigned N>
__device__ __forceinline__ void Pin(v8f (&acc)[N]) {
  if constexpr (N == 3)
    asm volatile("" : "+v"(acc[0]), "+v"(acc[1]), "+v"(acc[2]));
  else if constexpr (N == 2)
    asm volatile("" : "+v"(acc[0]), "+v"(acc[1]));
  else
    asm volatile("" : "+v"(acc[0]));
}

// Split combine of row m (of M) of one KV head by one wave: lane l owns
// output channels 8l..8l+7; the maxima are in base 2. Normalizes, rotates
// back and gates like the FP16 decode path.
__device__ __forceinline__ void CombineRow(const float* scratch,
                                           const float* gate, float* out,
                                           unsigned kv, unsigned m, unsigned M,
                                           unsigned nsplit, unsigned lane) {
#pragma clang fp contract(off)
  const float* base =
      scratch +
      (static_cast<std::size_t>(kv) * nsplit * M + m) * kPartialStride;
  const std::size_t stride = static_cast<std::size_t>(M) * kPartialStride;
  float mx = -INFINITY;
  for (unsigned s = lane; s < nsplit; s += 32)
    mx = fmaxf(mx, base[s * stride]);
#pragma unroll
  for (unsigned mask = 1; mask < 32; mask <<= 1)
    mx = fmaxf(mx, __shfl_xor(mx, mask));
  float sum = 0.0F, y[8] = {};
  for (unsigned s0 = 0; s0 < nsplit; s0 += 32) {
    float e = 0.0F;
    if (s0 + lane < nsplit) {
      const float* p = base + (s0 + lane) * stride;
      e = p[0] == -INFINITY ? 0.0F : __builtin_amdgcn_exp2f(p[0] - mx);
      sum = fmaf(p[1], e, sum);
    }
    const unsigned count = min(32U, nsplit - s0);
    // Eight splits' loads in flight per step.
    for (unsigned j0 = 0; j0 < count; j0 += 8) {
      float2 x[8][4];
      float w[8];
#pragma unroll
      for (unsigned j = 0; j < 8; ++j) {
        const unsigned jj = min(j0 + j, count - 1);
        w[j] = j0 + j < count ? __shfl(e, jj) : 0.0F;
        const auto* v = reinterpret_cast<const float2*>(
            base + (s0 + jj) * stride + 2 + 8 * lane);
#pragma unroll
        for (unsigned k = 0; k < 4; ++k)
          x[j][k] = v[k];
      }
#pragma unroll
      for (unsigned j = 0; j < 8; ++j)
#pragma unroll
        for (unsigned k = 0; k < 4; ++k) {
          y[2 * k] = fmaf(w[j], x[j][k].x, y[2 * k]);
          y[2 * k + 1] = fmaf(w[j], x[j][k].y, y[2 * k + 1]);
        }
    }
  }
#pragma unroll
  for (unsigned mask = 1; mask < 32; mask <<= 1)
    sum += __shfl_xor(sum, mask);
#pragma unroll
  for (unsigned k = 0; k < 8; ++k)
    y[k] /= sum;
  Fwht256(y, lane);
  const std::size_t index =
      (static_cast<std::size_t>(m / kGqa) * kHeads + kv * kGqa + m % kGqa) *
          kDim +
      8 * lane;
  float r[8];
#pragma unroll
  for (unsigned k = 0; k < 8; ++k) {
    r[k] = y[k] * 0.0625F;
    if (gate != nullptr)
      r[k] *= 1.0F / (1.0F + expf(-gate[index + k]));
  }
  reinterpret_cast<float4*>(out + index)[0] = float4{r[0], r[1], r[2], r[3]};
  reinterpret_cast<float4*>(out + index)[1] = float4{r[4], r[5], r[6], r[7]};
}

// Verification: one wave per (KV head, row), 8 per block.
__launch_bounds__(256) __global__
    void KvarnCombineKernel(const float* __restrict__ scratch,
                            const float* __restrict__ gate,
                            float* __restrict__ out, unsigned nsplit,
                            unsigned rows) {
  const unsigned M = kGqa * rows;
  const unsigned pair = blockIdx.x * 8 + threadIdx.x / 32;
  if (pair < kKvHeads * M)
    CombineRow(scratch, gate, out, pair / M, pair % M, M, nsplit,
               threadIdx.x % 32);
}

template<int VBits, unsigned Rows>
__launch_bounds__(256) __global__ void KvarnDecodePartialsKernel(DecodeArgs a) {
  // No implicit contraction: which product of a sum the compiler fuses may
  // differ between row-count instances.
#pragma clang fp contract(off)
  using R = Record<VBits>;
  constexpr unsigned M = kGqa * Rows, MT = (M + 15) / 16, MR = MT * 16;
  constexpr unsigned kQStride = kDim + 8, kPStride = kGroup + 8;
  constexpr unsigned kVWords = kGroup * VBits / 32;  // words per channel
  // AR prefetches the next group's K into registers; verification, with
  // three times the accumulators, only its K scale pairs. V codes load at
  // the start of their own group.
  constexpr bool kPrefetch = Rows == 1;
  constexpr float kScoreScale = 0.0625F * kLog2e;
  __shared__ __align__(16) _Float16 q16[MR * kQStride];
  __shared__ __align__(16) _Float16 p16[MR * kPStride];
  __shared__ __align__(16) float wave_max[MR][8];
  __shared__ __align__(16) float wave_psum[MR][8];  // sum of P per wave
  // Sum of P * V zero minus FP16 P * code bias, per wave.
  __shared__ __align__(16) float wave_zsum[MR][8];
  __shared__ __align__(16) unsigned key_scale[kDim / 2], key_zero[kDim / 2];
  // Online-softmax state per row: the max double-buffered by chunk parity,
  // the sum (wave 0 only) and the chunk's rescale factor.
  __shared__ float row_max[2][MR], row_sum[MR], row_alpha[MR];

  const unsigned kv = blockIdx.x, split = blockIdx.z;
  const unsigned tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
  const unsigned sub = lane % 16, half = lane / 16;
  const unsigned length = (a.d_pos != nullptr ? *a.d_pos : a.start_pos) + Rows;
  const unsigned token = 16 * wave + sub;  // this lane's score column

  // q -> H q / 16 as FP16 rows; padding rows are zero.
  for (unsigned m = wave; m < MR; m += 8) {
    float y[8] = {};
    if (m < M) {
      const auto* src = reinterpret_cast<const float4*>(
          a.q +
          (static_cast<std::size_t>(m / kGqa) * kHeads + kv * kGqa + m % kGqa) *
              kDim +
          8 * lane);
      const float4 lo = src[0], hi = src[1];
      y[0] = lo.x, y[1] = lo.y, y[2] = lo.z, y[3] = lo.w;
      y[4] = hi.x, y[5] = hi.y, y[6] = hi.z, y[7] = hi.w;
      Fwht256(y, lane);
    }
    _Float16 h[8];
#pragma unroll
    for (unsigned j = 0; j < 8; ++j)
      h[j] = static_cast<_Float16>(y[j] * 0.0625F);
    *reinterpret_cast<uint4*>(&q16[m * kQStride + 8 * lane]) =
        *reinterpret_cast<const uint4*>(h);
  }
  for (unsigned i = tid; i < MR * kPStride / 2; i += 256)
    reinterpret_cast<unsigned*>(p16)[i] = 0;
  if (tid < MR) {
    row_max[0][tid] = -INFINITY;
    row_sum[tid] = 0.0F;
  }
  v8f out[MT][2];
#pragma unroll
  for (unsigned mt = 0; mt < MT; ++mt) {
    out[mt][0] = v8f{};
    out[mt][1] = v8f{};
  }

  // Padding rows of a tile are skipped (M is even, so 2i < M - 16 mt covers
  // both half-waves); their P stays zero and their outputs unused.
  const auto Live = [](unsigned mt, unsigned i) { return mt * 16 + 2 * i < M; };
  // Scores in acc (scaled, masked) -> P * V scale of this lane's token in
  // LDS, per-wave sums of P and P * V zero, the row state for this chunk.
  // `between` runs after the first barrier, when no wave reads scores
  // inputs any more.
  const auto softmax = [&](v8f(&acc)[MT], unsigned parity, float value_scale,
                           float value_zero, float value_bias, auto&& between) {
#pragma unroll
    for (unsigned mt = 0; mt < MT; ++mt)
#pragma unroll
      for (unsigned i = 0; i < 8; ++i) {
        if (!Live(mt, i))
          continue;
        const float v = RowMax16(acc[mt][i]);
        if (sub == 0)
          wave_max[mt * 16 + 2 * i + half][wave] = v;
      }
    LdsBarrier();
    between();
#pragma unroll
    for (unsigned mt = 0; mt < MT; ++mt)
#pragma unroll
      for (unsigned i = 0; i < 8; ++i) {
        if (!Live(mt, i))
          continue;
        const unsigned row = mt * 16 + 2 * i + half;
        const float4 m0 = *reinterpret_cast<const float4*>(&wave_max[row][0]);
        const float4 m1 = *reinterpret_cast<const float4*>(&wave_max[row][4]);
        const float chunk = fmaxf(fmaxf(fmaxf(m0.x, m0.y), fmaxf(m0.z, m0.w)),
                                  fmaxf(fmaxf(m1.x, m1.y), fmaxf(m1.z, m1.w)));
        const float previous = row_max[parity][row];
        const float next = fmaxf(previous, chunk);
        const bool empty = next == -INFINITY;
        if (wave == 0 && sub == 0) {
          row_max[parity ^ 1][row] = next;
          row_alpha[row] =
              empty ? 1.0F : __builtin_amdgcn_exp2f(previous - next);
        }
        const float s = acc[mt][i];
        const float p =
            (empty || s == -INFINITY) ? 0.0F : __builtin_amdgcn_exp2f(s - next);
        const _Float16 ph = static_cast<_Float16>(p * value_scale);
        p16[row * kPStride + token] = ph;
        const float psum = RowSum16(p);
        const float z =
            fmaf(-static_cast<float>(ph), value_bias, p * value_zero);
        const float zsum = RowSum16(z);
        if (sub == 0) {
          wave_psum[row][wave] = psum;
          wave_zsum[row][wave] = zsum;
        }
      }
    LdsBarrier();
    if (wave == 0 && sub == 0)
#pragma unroll
      for (unsigned mt = 0; mt < MT; ++mt)
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
          if (!Live(mt, i))
            continue;
          const unsigned row = mt * 16 + 2 * i + half;
          const float4 s0 =
              *reinterpret_cast<const float4*>(&wave_psum[row][0]);
          const float4 s1 =
              *reinterpret_cast<const float4*>(&wave_psum[row][4]);
          const float s =
              ((s0.x + s0.y) + (s0.z + s0.w)) + ((s1.x + s1.y) + (s1.z + s1.w));
          row_sum[row] = fmaf(row_sum[row], row_alpha[row], s);
        }
  };
  // Loops over k steps, fully unrolled for register-held codes, rolled for
  // FP16 rows read from memory.
  const auto steps = [](auto unroll, unsigned count, auto&& step) {
    if constexpr (decltype(unroll)::value) {
#pragma unroll
      for (unsigned s = 0; s < 16; ++s)
        if (s < count)
          step(s);
    } else {
#pragma unroll 1
      for (unsigned s = 0; s < count; ++s)
        step(s);
    }
  };
  const auto scores = [&](auto unroll, auto build_k, v8f(&acc)[MT]) {
#pragma unroll
    for (unsigned mt = 0; mt < MT; ++mt)
      acc[mt] = v8f{};
    steps(unroll, kDim / 16, [&](unsigned ks) {
      const v16h b = build_k(ks);
#pragma unroll
      for (unsigned mt = 0; mt < MT; ++mt)
        acc[mt] = Wmma(LoadFrag(&q16[(mt * 16 + sub) * kQStride + 16 * ks]), b,
                       acc[mt]);
      Pin(acc);
    });
  };
  // spread: the k pairs of the product hold one token and a zero (8 steps
  // of 8 tokens for a 64-token FP16 block), so a causal boundary never
  // splits a pair.
  const auto values = [&](auto unroll, auto build_v, const float (&column)[2],
                          auto spread) {
    constexpr bool kSpread = decltype(spread)::value;
    // AR: the zero sums once per chunk, live rows only (verification would
    // hold 24 more registers across both products).
    constexpr bool kZerosOnce = Rows == 1;
    float zeros[kZerosOnce ? MT : 1][8];
    if constexpr (kZerosOnce)
#pragma unroll
      for (unsigned mt = 0; mt < MT; ++mt)
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
          zeros[mt][i] = 0.0F;
          if (!Live(mt, i))
            continue;
          const unsigned row = mt * 16 + 2 * i + half;
          const float4 z0 =
              *reinterpret_cast<const float4*>(&wave_zsum[row][0]);
          const float4 z1 =
              *reinterpret_cast<const float4*>(&wave_zsum[row][4]);
          zeros[mt][i] =
              ((z0.x + z0.y) + (z0.z + z0.w)) + ((z1.x + z1.y) + (z1.z + z1.w));
        }
#pragma unroll
    for (unsigned t = 0; t < 2; ++t) {
      v8f g[MT];
#pragma unroll
      for (unsigned mt = 0; mt < MT; ++mt)
        g[mt] = v8f{};
      steps(unroll, kSpread ? kTailBlockTokens / 8 : kGroup / 16,
            [&](unsigned kt) {
              const v16h b = build_v(t, kt);
#pragma unroll
              for (unsigned mt = 0; mt < MT; ++mt) {
                v16h fa;
                if constexpr (kSpread) {
                  const uint4 x = *reinterpret_cast<const uint4*>(
                      &p16[(mt * 16 + sub) * kPStride + 8 * kt]);
                  const u32x8 w = {x.x & 0xffffU, x.x >> 16,     x.y & 0xffffU,
                                   x.y >> 16,     x.z & 0xffffU, x.z >> 16,
                                   x.w & 0xffffU, x.w >> 16};
                  fa = __builtin_bit_cast(v16h, w);
                } else {
                  fa = LoadFrag(&p16[(mt * 16 + sub) * kPStride + 16 * kt]);
                }
                g[mt] = Wmma(fa, b, g[mt]);
              }
              Pin(g);
            });
#pragma unroll
      for (unsigned mt = 0; mt < MT; ++mt)
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
          if (!Live(mt, i))
            continue;
          const unsigned row = mt * 16 + 2 * i + half;
          float zero;
          if constexpr (kZerosOnce) {
            zero = zeros[mt][i];
          } else {
            const float4 z0 =
                *reinterpret_cast<const float4*>(&wave_zsum[row][0]);
            const float4 z1 =
                *reinterpret_cast<const float4*>(&wave_zsum[row][4]);
            zero = ((z0.x + z0.y) + (z0.z + z0.w)) +
                   ((z1.x + z1.y) + (z1.z + z1.w));
          }
          out[mt][t][i] = fmaf(row_alpha[row], out[mt][t][i],
                               column[t] * (g[mt][i] + zero));
        }
    }
  };

  unsigned final_parity = 0;
  if (split < a.splits) {
    const unsigned begin = static_cast<unsigned>(
        static_cast<std::uint64_t>(split) * a.groups / a.splits);
    const unsigned end = static_cast<unsigned>(
        static_cast<std::uint64_t>(split + 1) * a.groups / a.splits);
    // One group's register-held inputs.
    struct KeyPart {
      uint4 codes[8];
      float column, value_scale, value_zero;
    };
    struct ValuePart {
      uint4 codes[2][kVWords / 4];
      float column[2];
    };
    const auto record_of = [&](unsigned group) {
      return a.records + RecordOffset<VBits>(a.blocks, group + 1, kv);
    };
    // Threads < 64: the K scale (< 32) or zero pairs, for the LDS copy.
    const auto load_scales = [&](unsigned group) {
      uint4 v{};
      if (tid < 64)
        v = reinterpret_cast<const uint4*>(
            record_of(group) + (tid < 32 ? R::kKScale : R::kKZero))[tid % 32];
      return v;
    };
    const auto load_keys = [&](unsigned group, KeyPart& k) {
      const std::uint8_t* record = record_of(group);
      const auto* krow =
          reinterpret_cast<const uint4*>(record + R::kKCodes + token * 128);
#pragma unroll
      for (unsigned i = 0; i < 8; ++i)
        k.codes[i] = krow[i];
      const auto* scales = reinterpret_cast<const __half*>(record);
      k.column = __half2float(scales[R::kKColumn / 2 + token]) * kScoreScale;
      k.value_scale = __half2float(scales[R::kVScale / 2 + token]);
      k.value_zero = __half2float(scales[R::kVZero / 2 + token]);
    };
    const auto load_values = [&](unsigned group, ValuePart& v) {
      const std::uint8_t* record = record_of(group);
#pragma unroll
      for (unsigned t = 0; t < 2; ++t) {
        const unsigned channel = 16 * (wave + 8 * t) + sub;
        const auto* vrow = reinterpret_cast<const uint4*>(
            record + R::kVCodes + channel * R::kVRowBytes);
#pragma unroll
        for (unsigned i = 0; i < kVWords / 4; ++i)
          v.codes[t][i] = vrow[i];
        v.column[t] = __half2float(
            reinterpret_cast<const __half*>(record + R::kVColumn)[channel]);
      }
    };
    const auto stage_scales = [&](const uint4& v) {
      if (tid < 64)
        reinterpret_cast<uint4*>(tid < 32 ? key_scale : key_zero)[tid % 32] = v;
    };
    // The code bias of this lane's token in the V fragments.
    const float value_bias =
        VBits == 2 ? CodeBias<2>(sub / 2) : CodeBias<4>((sub % 8) / 2);
    KeyPart keys{};
    ValuePart vals{};
    if (begin < end) {
      stage_scales(load_scales(begin));
      if constexpr (kPrefetch)
        load_keys(begin, keys);
    }
    LdsBarrier();
    for (unsigned group = begin; group < end; ++group) {
      const bool more = group + 1 < end;
      const uint4 next_scales = more ? load_scales(group + 1) : uint4{};
      KeyPart next_keys{};
      if constexpr (kPrefetch) {
        if (more)
          load_keys(group + 1, next_keys);
      } else {
        load_keys(group, keys);
      }
      load_values(group, vals);

      v8f acc[MT];
      scores(
          std::true_type{},
          [&](unsigned ks) {
            const uint4 v = keys.codes[ks / 2];
            const unsigned w0 = (ks & 1) != 0 ? v.z : v.x;
            const unsigned w1 = (ks & 1) != 0 ? v.w : v.y;
            const auto* s4 = reinterpret_cast<const uint4*>(key_scale);
            const auto* z4 = reinterpret_cast<const uint4*>(key_zero);
            const uint4 s0 = s4[2 * ks], s1 = s4[2 * ks + 1];
            const uint4 z0 = z4[2 * ks], z1 = z4[2 * ks + 1];
            const unsigned sv[8] = {s0.x, s0.y, s0.z, s0.w,
                                    s1.x, s1.y, s1.z, s1.w};
            const unsigned zv[8] = {z0.x, z0.y, z0.z, z0.w,
                                    z1.x, z1.y, z1.z, z1.w};
            u32x8 b;
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
              b[j] = Dequant2((w0 >> (4 * j)) & 0x000F000FU, sv[j], zv[j]);
              b[4 + j] =
                  Dequant2((w1 >> (4 * j)) & 0x000F000FU, sv[4 + j], zv[4 + j]);
            }
            return __builtin_bit_cast(v16h, b);
          },
          acc);
#pragma unroll
      for (unsigned mt = 0; mt < MT; ++mt)
        acc[mt] *= keys.column;
      // The next group's K scales replace this group's once every wave is
      // past its scores; the second softmax barrier publishes them.
      softmax(acc, (group - begin) & 1, keys.value_scale, keys.value_zero,
              value_bias, [&] {
                if (more)
                  stage_scales(next_scales);
              });
      values(
          std::true_type{},
          [&](unsigned t, unsigned kt) {
            u32x8 b;
            if constexpr (VBits == 2) {
              const uint4 v = vals.codes[t][kt / 4];
              const unsigned w = kt % 4 == 0   ? v.x
                                 : kt % 4 == 1 ? v.y
                                 : kt % 4 == 2 ? v.z
                                               : v.w;
              unsigned o[8];
              BiasedCodes<2>(w, o);
#pragma unroll
              for (unsigned j = 0; j < 8; ++j)
                b[j] = o[j];
            } else {
              const uint4 v = vals.codes[t][kt / 2];
              unsigned lo[4], hi[4];
              BiasedCodes<4>((kt & 1) != 0 ? v.z : v.x, lo);
              BiasedCodes<4>((kt & 1) != 0 ? v.w : v.y, hi);
#pragma unroll
              for (unsigned j = 0; j < 4; ++j) {
                b[j] = lo[j];
                b[4 + j] = hi[j];
              }
            }
            return __builtin_bit_cast(v16h, b);
          },
          vals.column, std::false_type{});
      if constexpr (kPrefetch)
        keys = next_keys;
    }
    final_parity = (end - begin) & 1;
  } else {
    // An FP16 block: 64 positions of the sink, or of the tail after the
    // quantized groups, one chunk with spread PV pairs.
    LdsBarrier();
    const unsigned block = split - a.splits;
    const unsigned first = block < kSinkBlocks
                               ? block * kTailBlockTokens
                               : kGroup * (1 + a.groups) +
                                     (block - kSinkBlocks) * kTailBlockTokens;
    const unsigned limit = block < kSinkBlocks ? kKvarnSinkTokens : length;
    const unsigned count =
        first < limit ? min(kTailBlockTokens, limit - first) : 0;
    // The block's rows are consecutive: a sink half or one half of a group
    // in the ring.
    const std::size_t block_offset =
        static_cast<std::size_t>(KvarnFp16Row(first)) * kRowWidth + kv * kDim;
    const __half* vbase = a.v_rows + block_offset;
    // Lanes past the block's 64 tokens (waves 4-7) score nothing.
    const __half* krow =
        a.k_rows + block_offset +
        static_cast<std::size_t>(token < count ? token : 0) * kRowWidth;
    const float ones[2] = {1.0F, 1.0F};
    if (count != 0) {
      v8f acc[MT];
      scores(
          std::false_type{},
          [&](unsigned ks) {
            const auto* p = reinterpret_cast<const uint4*>(krow + 16 * ks);
            const uint4 lo = p[0], hi = p[1];
            const u32x8 w = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
            return __builtin_bit_cast(v16h, w);
          },
          acc);
      const unsigned position = first + token;
#pragma unroll
      for (unsigned mt = 0; mt < MT; ++mt)
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
          const unsigned row = mt * 16 + 2 * i + half;
          const unsigned last = length - Rows + (row < M ? row / kGqa : 0);
          acc[mt][i] = (token < count && position <= last)
                           ? acc[mt][i] * kScoreScale
                           : -INFINITY;
        }
      softmax(acc, 0, 1.0F, 0.0F, 0.0F, [] {});
      values(
          std::false_type{},
          [&](unsigned t, unsigned kt) {
            const unsigned channel = 16 * (wave + 8 * t) + sub;
            v16h b;
#pragma unroll
            for (unsigned k = 0; k < 8; ++k) {
              const unsigned tk = 8 * kt + k;
              b[2 * k] =
                  tk < count
                      ? static_cast<_Float16>(__half2float(
                            vbase[static_cast<std::size_t>(tk) * kRowWidth +
                                  channel]))
                      : static_cast<_Float16>(0.0F);
              b[2 * k + 1] = static_cast<_Float16>(0.0F);
            }
            return b;
          },
          ones, std::true_type{});
      final_parity = 1;
    }
  }

  const unsigned nsplit = SplitCount(a);
  float* partial = a.scratch + (static_cast<std::size_t>(kv) * nsplit + split) *
                                   M * kPartialStride;
#pragma unroll
  for (unsigned mt = 0; mt < MT; ++mt)
#pragma unroll
    for (unsigned t = 0; t < 2; ++t)
#pragma unroll
      for (unsigned i = 0; i < 8; ++i) {
        const unsigned row = mt * 16 + 2 * i + half;
        if (row < M)
          partial[row * kPartialStride + 2 + 16 * (wave + 8 * t) + sub] =
              out[mt][t][i];
      }
  if (wave == 0 && sub == 0)
#pragma unroll
    for (unsigned mt = 0; mt < MT; ++mt)
#pragma unroll
      for (unsigned i = 0; i < 8; ++i) {
        const unsigned row = mt * 16 + 2 * i + half;
        if (row < M) {
          partial[row * kPartialStride] = row_max[final_parity][row];
          partial[row * kPartialStride + 1] = row_sum[row];
        }
      }
  if constexpr (Rows == 1) {
    // The last block of this KV head to finish combines its splits.
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      const unsigned done = atomicAdd(&a.counters[kv], 1U);
      last = done == nsplit - 1;
      if (last)
        a.counters[kv] = 0;
    }
    __syncthreads();
    if (!last)
      return;
    __threadfence();
    for (unsigned m = wave; m < M; m += 8)
      CombineRow(a.scratch, a.gate, a.out, kv, m, M, nsplit, lane);
  }
}

// ---------------------------------------------------------------------------
// Prefill pack: positions [0, length) of KV heads [first_head, ...) in the
// original frame, head-major, as the prefill WMMA attention reads them. One
// 256-thread block per (head, 16 positions); wave w takes positions 2w and
// 2w+1, lane l channels 8l..8l+7. Rotated sources (sink, quantized groups,
// ring rows) are rotated back in FP32 and rounded to FP16 once; the chunk's
// own rows [start, length) are its FP32 K/V rounded to FP16, as the FP16
// cache stores them.
// ---------------------------------------------------------------------------
struct PackArgs {
  const std::uint8_t* records;  // this layer's record plane
  const std::uint32_t* blocks;
  const __half* k_rows;  // this layer's FP16 plane, rotated
  const __half* v_rows;
  const float* k;  // the chunk's rows [row][kv][256]
  const float* v;
  __half* packed_k;
  __half* packed_v;
  unsigned start, length, groups, first_head;
};

template<int VBits>
__launch_bounds__(256) __global__ void KvarnPackHeadsKernel(PackArgs a) {
  using R = Record<VBits>;
  __shared__ __half tile[16][kDim + 8];
  const unsigned head = blockIdx.x, kv = a.first_head + head;
  const unsigned base = blockIdx.y * 16;
  const unsigned wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const unsigned quantized_end = kGroup * (1 + a.groups);
  for (unsigned which = 0; which < 4; ++which) {
    const unsigned key_index = 2 * wave + which / 2;
    const bool key = (which & 1) == 0;
    const unsigned t = base + key_index;
    float y[8] = {};
    bool rotated = true;
    if (t >= a.length) {
      rotated = false;
    } else if (t >= a.start) {
      const float* src =
          (key ? a.k : a.v) +
          (static_cast<std::size_t>(t - a.start) * kKvHeads + kv) * kDim +
          8 * lane;
#pragma unroll
      for (unsigned j = 0; j < 8; ++j)
        y[j] = __half2float(__float2half_rn(src[j]));
      rotated = false;
    } else if (t >= kGroup && t < quantized_end) {
      const unsigned group = t / kGroup, u = t % kGroup;
      const std::uint8_t* record =
          a.records + RecordOffset<VBits>(a.blocks, group, kv);
      const auto* halves = reinterpret_cast<const __half*>(record);
      if (key) {
        const unsigned word = reinterpret_cast<const unsigned*>(
            record + R::kKCodes)[u * 32 + lane];
        const float column = __half2float(halves[R::kKColumn / 2 + u]);
#pragma unroll
        for (unsigned j = 0; j < 8; ++j) {
          const unsigned nibble = (j % 2 == 0) ? j / 2 : 4 + j / 2;
          const float code = static_cast<float>((word >> (4 * nibble)) & 15);
          const unsigned c = 8 * lane + j;
          y[j] = (code * __half2float(halves[R::kKScale / 2 + c]) +
                  __half2float(halves[R::kKZero / 2 + c])) *
                 column;
        }
      } else {
        const float scale = __half2float(halves[R::kVScale / 2 + u]);
        const float zero = __half2float(halves[R::kVZero / 2 + u]);
        const auto* words =
            reinterpret_cast<const unsigned*>(record + R::kVCodes);
#pragma unroll
        for (unsigned j = 0; j < 8; ++j) {
          const unsigned c = 8 * lane + j;
          unsigned code;
          if constexpr (VBits == 2) {
            const unsigned w = u % 16;
            const unsigned crumb = (w % 2 == 0) ? w / 2 : 8 + w / 2;
            code = (words[c * 8 + u / 16] >> (2 * crumb)) & 3;
          } else {
            const unsigned w = u % 8;
            const unsigned nibble = (w % 2 == 0) ? w / 2 : 4 + w / 2;
            code = (words[c * 16 + u / 8] >> (4 * nibble)) & 15;
          }
          y[j] = (static_cast<float>(code) * scale + zero) *
                 __half2float(halves[R::kVColumn / 2 + c]);
        }
      }
    } else {
      const __half* src =
          (key ? a.k_rows : a.v_rows) +
          static_cast<std::size_t>(KvarnFp16Row(t)) * kRowWidth + kv * kDim +
          8 * lane;
#pragma unroll
      for (unsigned j = 0; j < 8; ++j)
        y[j] = __half2float(src[j]);
    }
    if (rotated) {
      Fwht256(y, lane);
#pragma unroll
      for (unsigned j = 0; j < 8; ++j)
        y[j] *= 0.0625F;
    }
    _Float16 h[8];
#pragma unroll
    for (unsigned j = 0; j < 8; ++j)
      h[j] = static_cast<_Float16>(y[j]);
    if (key) {
      if (t < a.length)
        *reinterpret_cast<uint4*>(
            a.packed_k +
            (static_cast<std::size_t>(head) * a.length + t) * kDim + 8 * lane) =
            *reinterpret_cast<const uint4*>(h);
    } else {
      *reinterpret_cast<uint4*>(&tile[key_index][8 * lane]) =
          *reinterpret_cast<const uint4*>(h);
    }
  }
  __syncthreads();
  // V^T per 16-position tile: [dim][16 keys], as PackAttentionHeads.
  const std::size_t padded = (a.length + 15U) & ~std::size_t{15};
  for (unsigned i = threadIdx.x; i < 16 * kDim / 8; i += 256) {
    const unsigned dim = i / 2, key = (i % 2) * 8;
    __half values[8];
#pragma unroll
    for (unsigned j = 0; j < 8; ++j)
      values[j] = tile[key + j][dim];
    *reinterpret_cast<uint4*>(a.packed_v +
                              static_cast<std::size_t>(head) * padded * kDim +
                              static_cast<std::size_t>(base) * kDim + dim * 16 +
                              key) = *reinterpret_cast<const uint4*>(values);
  }
}

template<int VBits, unsigned Rows>
void LaunchPartials(const DecodeArgs& args, hipStream_t stream) {
  const dim3 grid(kKvHeads, 1, SplitCount(args));
  KvarnDecodePartialsKernel<VBits, Rows><<<grid, 256, 0, stream>>>(args);
}

template<int VBits>
void LaunchPartialsRows(const DecodeArgs& args, unsigned rows,
                        hipStream_t stream) {
  switch (rows) {
    case 1:
      return LaunchPartials<VBits, 1>(args, stream);
    case 2:
      return LaunchPartials<VBits, 2>(args, stream);
    case 3:
      return LaunchPartials<VBits, 3>(args, stream);
    case 4:
      return LaunchPartials<VBits, 4>(args, stream);
    case 5:
      return LaunchPartials<VBits, 5>(args, stream);
    case 6:
      return LaunchPartials<VBits, 6>(args, stream);
    case 7:
      return LaunchPartials<VBits, 7>(args, stream);
    case 8:
      return LaunchPartials<VBits, 8>(args, stream);
    default:
      throw std::invalid_argument("KVarN attention takes 1 to 8 query rows");
  }
}

}  // namespace

void LaunchKvarnWriteRows(const float* k, const float* v, const KvarnKvView& kv,
                          std::uint32_t layer, const std::uint32_t* d_pos,
                          std::uint32_t start_pos, std::uint32_t rows,
                          hipStream_t stream) {
  if (rows == 0)
    return;
  auto* const k_rows =
      static_cast<__half*>(kv.k_rows) + layer * kFp16LayerElements;
  auto* const v_rows =
      static_cast<__half*>(kv.v_rows) + layer * kFp16LayerElements;
  KvarnWriteRowsKernel<<<rows, 256, 0, stream>>>(k, v, k_rows, v_rows, d_pos,
                                                 start_pos, kRingRows);
}

void LaunchKvarnFlush(const KvarnKvView& kv, std::uint32_t layers,
                      std::uint32_t first_group, std::uint32_t groups,
                      hipStream_t stream, std::uint32_t first_layer) {
  if (groups == 0 || layers == 0)
    return;
  if (first_group == 0)
    throw std::invalid_argument("KVarN group 0 is the FP16 sink");
  const dim3 grid(kKvHeads, groups, 2 * layers);
  const auto* k_rows =
      static_cast<const __half*>(kv.k_rows) + first_layer * kFp16LayerElements;
  const auto* v_rows =
      static_cast<const __half*>(kv.v_rows) + first_layer * kFp16LayerElements;
  std::uint8_t* records = kv.records + first_layer * kv.records_layer_pitch;
  if (kv.value_bits == 2)
    KvarnQuantizeKernel<2><<<grid, 512, 0, stream>>>(
        k_rows, v_rows, records, kv.records_layer_pitch, kv.blocks, first_group,
        kFp16LayerElements, kRingRows);
  else if (kv.value_bits == 4)
    KvarnQuantizeKernel<4><<<grid, 512, 0, stream>>>(
        k_rows, v_rows, records, kv.records_layer_pitch, kv.blocks, first_group,
        kFp16LayerElements, kRingRows);
  else
    throw std::invalid_argument("KVarN value codes are 2 or 4 bits");
}

void LaunchKvarnDecodeAttention(const float* q, const float* gate,
                                const KvarnKvView& kv, float* out_context,
                                std::uint32_t layer, const std::uint32_t* d_pos,
                                std::uint32_t start_pos, std::uint32_t rows,
                                std::uint32_t groups, std::uint32_t splits,
                                std::span<float> scratch, hipStream_t stream) {
  if (rows == 0 || rows > kKvarnMaxRows)
    throw std::invalid_argument("KVarN attention takes 1 to 8 query rows");
  if (splits == 0)
    splits = SelectKvarnDecodeSplits(groups, kv.value_bits);
  if (splits > groups || (groups != 0 && splits == 0))
    throw std::invalid_argument("KVarN splits must cover the quantized groups");
  const std::uint32_t first = kGroup * (1 + groups);
  if (groups != 0 && start_pos < first)
    throw std::invalid_argument(
        "KVarN queries must not precede the end of the quantized groups");
  const std::uint32_t end = start_pos + rows;
  const std::uint32_t tail = end > first ? end - first : 0;
  const std::uint32_t tail_blocks =
      (tail + kTailBlockTokens - 1) / kTailBlockTokens;
  if (tail_blocks > kKvarnMaxTailBlocks)
    throw std::invalid_argument("KVarN FP16 tail exceeds its blocks");
  if (scratch.size() < KvarnScratchElements(rows, splits))
    throw std::invalid_argument("KVarN attention scratch is too small");
  if (rows == 1 && kv.counters == nullptr)
    throw std::invalid_argument("KVarN decode needs its arrival counters");
  detail::EmitAttentionDispatch(kv.value_bits == 2
                                    ? "decode_split_k_kvarn_k4v2"
                                    : "decode_split_k_kvarn_k4v4",
                                "");
  const DecodeArgs args{
      .q = q,
      .gate = gate,
      .out = out_context,
      .records = kv.records + layer * kv.records_layer_pitch,
      .blocks = kv.blocks,
      .k_rows =
          static_cast<const __half*>(kv.k_rows) + layer * kFp16LayerElements,
      .v_rows =
          static_cast<const __half*>(kv.v_rows) + layer * kFp16LayerElements,
      .scratch = scratch.data(),
      .counters = kv.counters,
      .d_pos = d_pos,
      .start_pos = start_pos,
      .groups = groups,
      .splits = splits,
      .tail_blocks = tail_blocks,
  };
  if (kv.value_bits == 2)
    LaunchPartialsRows<2>(args, rows, stream);
  else if (kv.value_bits == 4)
    LaunchPartialsRows<4>(args, rows, stream);
  else
    throw std::invalid_argument("KVarN value codes are 2 or 4 bits");
  // AR combines inside the partials launch; verification (up to 48 rows per
  // KV head) in a wave-per-row launch.
  if (rows > 1)
    KvarnCombineKernel<<<(kKvHeads * kGqa * rows + 7) / 8, 256, 0, stream>>>(
        scratch.data(), gate, out_context, SplitCount(args), rows);
}

void LaunchKvarnPassAttention(const float* q, const float* k, const float* v,
                              const float* gate, const KvarnKvView& kv,
                              float* out_context, std::uint32_t layer,
                              std::uint32_t start, std::uint32_t rows,
                              const KvarnPassPlan& plan,
                              std::span<float> scratch, hipStream_t stream) {
  LaunchKvarnWriteRows(k, v, kv, layer, nullptr, start, rows, stream);
  const std::uint32_t split =
      plan.boundary != 0 ? std::min(plan.boundary - start, rows) : rows;
  if (plan.boundary != 0)
    LaunchKvarnFlush(kv, 1, (plan.boundary >> kKvarnGroupShift) - 1, 1, stream,
                     layer);
  constexpr std::size_t kWidth = std::size_t{kHeads} * kDim;
  if (split != 0)
    LaunchKvarnDecodeAttention(q, gate, kv, out_context, layer, nullptr, start,
                               split, plan.read_groups, 0, scratch, stream);
  if (split != rows)
    LaunchKvarnDecodeAttention(
        q + split * kWidth, gate == nullptr ? nullptr : gate + split * kWidth,
        kv, out_context + split * kWidth, layer, nullptr, start + split,
        rows - split, plan.read_groups + 1, 0, scratch, stream);
}

void LaunchKvarnPrefillAttention(
    const float* q, const float* k, const float* v, const float* gate,
    const KvarnKvView& kv, float* out_context, std::uint32_t layer,
    std::uint32_t start, std::uint32_t rows, const KvarnPrefillPlan& plan,
    std::span<float> packed_k_workspace, std::span<float> packed_v_workspace,
    const KvarnStaging& staging, std::size_t first_query, hipStream_t stream) {
  if (rows == 0)
    return;
  const std::uint32_t length = start + rows;
  const std::size_t key_head_bytes =
      static_cast<std::size_t>(length) * kDim * sizeof(__half);
  const std::size_t value_head_bytes =
      ((static_cast<std::size_t>(length) + 15U) & ~std::size_t{15}) * kDim *
      sizeof(__half);
  const auto heads_per_group = static_cast<std::uint32_t>(std::min(
      {std::size_t{kKvHeads}, packed_k_workspace.size_bytes() / key_head_bytes,
       packed_v_workspace.size_bytes() / value_head_bytes}));
  if (heads_per_group == 0)
    throw std::length_error("KVarN prefill workspace holds no KV head");
  if (staging.rows.size() < 2 * KvarnStagingRows(rows) * kRowWidth)
    throw std::length_error("KVarN prefill staging is too small");
  detail::EmitAttentionDispatch(kv.value_bits == 2 ? "prefill_wmma_kvarn_k4v2"
                                                   : "prefill_wmma_kvarn_k4v4",
                                "");
  const auto* layer_k =
      static_cast<const __half*>(kv.k_rows) + layer * kFp16LayerElements;
  const auto* layer_v =
      static_cast<const __half*>(kv.v_rows) + layer * kFp16LayerElements;
  const std::uint8_t* layer_records =
      kv.records + layer * kv.records_layer_pitch;
  for (std::uint32_t first = 0; first < kKvHeads; first += heads_per_group) {
    const auto heads = std::min(heads_per_group, kKvHeads - first);
    const PackArgs args{
        .records = layer_records,
        .blocks = kv.blocks,
        .k_rows = layer_k,
        .v_rows = layer_v,
        .k = k,
        .v = v,
        .packed_k = reinterpret_cast<__half*>(packed_k_workspace.data()),
        .packed_v = reinterpret_cast<__half*>(packed_v_workspace.data()),
        .start = start,
        .length = length,
        .groups = plan.read_groups,
        .first_head = first,
    };
    const dim3 grid(heads, (length + 15U) / 16U);
    if (kv.value_bits == 2)
      KvarnPackHeadsKernel<2><<<grid, 256, 0, stream>>>(args);
    else if (kv.value_bits == 4)
      KvarnPackHeadsKernel<4><<<grid, 256, 0, stream>>>(args);
    else
      throw std::invalid_argument("KVarN value codes are 2 or 4 bits");
    LaunchQwenWmmaPackedHeadsAttention(q, gate, args.packed_k, args.packed_v,
                                       out_context, start, rows, first, heads,
                                       first_query, stream);
  }

  // Store the chunk. Rows in the sink go straight there.
  const std::uint32_t end = length;
  const std::uint32_t sink_end = std::min(end, kKvarnSinkTokens);
  if (start < sink_end)
    LaunchKvarnWriteRows(k, v, kv, layer, nullptr, start, sink_end - start,
                         stream);
  if (end <= kKvarnSinkTokens)
    return;
  // Staging holds positions [target, end): the ring's rows before the chunk
  // (one partial group, consecutive in the ring), then the chunk's own.
  const std::uint32_t target = plan.target;
  auto* staging_k = reinterpret_cast<__half*>(staging.rows.data());
  auto* staging_v = staging_k + KvarnStagingRows(rows) * kRowWidth;
  auto* ring_k = static_cast<__half*>(kv.k_rows) + layer * kFp16LayerElements;
  auto* ring_v = static_cast<__half*>(kv.v_rows) + layer * kFp16LayerElements;
  const std::size_t row_bytes = kRowWidth * sizeof(__half);
  if (start > target) {
    const std::size_t ring_row = KvarnFp16Row(target);
    HIP_CHECK(hipMemcpyAsync(staging_k, ring_k + ring_row * kRowWidth,
                             (start - target) * row_bytes,
                             hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(staging_v, ring_v + ring_row * kRowWidth,
                             (start - target) * row_bytes,
                             hipMemcpyDeviceToDevice, stream));
  }
  const std::uint32_t first_row = std::max(start, kKvarnSinkTokens);
  KvarnWriteRowsKernel<<<end - first_row, 256, 0, stream>>>(
      k + static_cast<std::size_t>(first_row - start) * kRowWidth,
      v + static_cast<std::size_t>(first_row - start) * kRowWidth, staging_k,
      staging_v, nullptr, first_row, target);
  const std::uint32_t complete_end = end >> kKvarnGroupShift
                                                << kKvarnGroupShift;
  if (complete_end > target) {
    const dim3 grid(kKvHeads, (complete_end - target) >> kKvarnGroupShift, 2);
    std::uint8_t* records = kv.records + layer * kv.records_layer_pitch;
    const unsigned first_group = target >> kKvarnGroupShift;
    hipStream_t quantize = stream;
    if (staging.quantize_stream != nullptr) {
      HIP_CHECK(hipEventRecord(staging.ready, stream));
      HIP_CHECK(hipStreamWaitEvent(staging.quantize_stream, staging.ready, 0));
      quantize = staging.quantize_stream;
    }
    if (kv.value_bits == 2)
      KvarnQuantizeKernel<2><<<grid, 512, 0, quantize>>>(
          staging_k, staging_v, records, kv.records_layer_pitch, kv.blocks,
          first_group, 0, target);
    else
      KvarnQuantizeKernel<4><<<grid, 512, 0, quantize>>>(
          staging_k, staging_v, records, kv.records_layer_pitch, kv.blocks,
          first_group, 0, target);
    if (staging.quantize_stream != nullptr)
      HIP_CHECK(hipEventRecord(staging.done, staging.quantize_stream));
  }
  // The incomplete group stays FP16 in the ring.
  const std::uint32_t leftover = std::max(target, complete_end);
  if (end > leftover) {
    const std::size_t ring_row = KvarnFp16Row(leftover);
    HIP_CHECK(hipMemcpyAsync(ring_k + ring_row * kRowWidth,
                             staging_k + (leftover - target) * kRowWidth,
                             (end - leftover) * row_bytes,
                             hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(ring_v + ring_row * kRowWidth,
                             staging_v + (leftover - target) * kRowWidth,
                             (end - leftover) * row_bytes,
                             hipMemcpyDeviceToDevice, stream));
  }
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
