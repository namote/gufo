// KVarN K1b prototype for Qwen 27B decode attention (plan/gufo-upstream-port/
// KVARN.md, K1-BRIEF.md): the group-flush quantizer, a fused-dequant split-K
// decode attention for one query (AR) and for eight verification queries
// (DFlash2), and the production FP16 split-K path as the baseline in the same
// binary. Head shape: 24 query heads, 4 KV heads (GQA 6), head_dim 256.
//
//   tools/bench/build.sh kvarn_decode_bench   # links the wave64 verify TU
//   /tmp/kvarn_decode_bench [check] [time] [quant] [probe]
//
// Kernels (kernel=): k1b, the K1b prototype; opt, the K1b-opt candidate
// (codes ORed into FP16 halves bias + code, K scale folded into q for AR,
// base-2 softmax, padding rows skipped, splits combined by the last block
// in AR and by a wave-per-row launch in verification); opt_latev, opt with
// the AR V codes loaded at the start of their own group (fewer registers,
// AR k4v4 back to 8 waves/SIMD).
//
// check: the GPU quantizer is compared code by code and scale by scale with
// an emulation of its own reduction order (bit-exact) and with the K0 rules;
// attention of every kernel is compared with CPU double attention over the
// GPU's codes (kernel arithmetic, at most 1e-3 relative) and over the K0
// fake-quantized FP16 KV in the original frame. time: AR and eight-row
// verification against the production FP16 path at 4K-256K, all kernels,
// formats and splits in rotating order, buffers cycled past the 32 MiB
// MALL; quantized splits 18/38 fill whole rounds of resident blocks (4 KV
// heads x (splits + 2) blocks). quant: one layer, all 16 layers and a
// 64-group batch of tiles. probe: 4K and 32K at split 18 with the launch
// split into partials, reduce, quantized blocks and FP16 sink/tail blocks.
//
// Frame. Every stored token is rotated by the normalized Sylvester Hadamard
// H (1/16, no random signs): the FP16 sink (first 128 tokens) and FP16 tail
// rows as well as the quantized groups. Attention rotates q by H and the
// output by H again (H^-1 = H). The quantization rules (rotation rounded to
// FP16, eight log-domain balancing iterations keeping the lowest imbalance,
// asymmetric per-row RTN, FP16 scales) are those of
// tests/models/qwen27b/kvarn_fake_quant.hpp; `check` compares codes and
// scales with it tile by tile.
//
// Proposed storage, per layer and 256-token pool block: two 128-token groups,
// each with one record per KV head, [group in block][kv head][record]. A
// record (k4v2 26,880 B, k4v4 35,072 B; FP16 rows are 131,072 B) is
//   K codes  [token 128][128 B]: 256 channels x 4 bit, token-major,
//            a 32-bit word holds 8 channels c..c+7 with channel c+2i in
//            nibble i and c+2i+1 in nibble i+4,
//   V codes  [channel 256][128 x b bit]: channel-major, a word holds 16
//            tokens (2 bit: token 2i in crumb i, 2i+1 in crumb i+8) or 8
//            tokens (4 bit: the K nibble order),
//   K scale [256], K zero [256] (per channel), K column scale [128] (per
//            token), V scale [128], V zero [128] (per token), V column scale
//            [256] (per channel), all FP16.
// The interleaved orders let a lane widen a word into packed halves with one
// shift, one mask and one OR per pair. K is token-major so a WMMA B fragment
// lane (one token) reads its 128 B contiguously; V is channel-major for the
// same reason in the PV product (one lane per channel). The value is
// (code * scale + zero) * column scale. The sink stays FP16 for the whole
// request and the unflushed tail stays FP16 until its group is complete and
// confirmed; both keep the existing [token][kv head][256] FP16 row layout
// (rotated) in a per-sequence plane, so the pool's group-0 record of a
// sequence's first block is unused.
#define ENGINE_ENABLE_HIP 1
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string_view>
#include <thread>
#include <vector>

#include "src/models/qwen/hip/kernels/attention_decode.hip"
#include "tests/models/qwen27b/kvarn_fake_quant.hpp"

#define KV_OK(expr)                                                     \
  do {                                                                  \
    const hipError_t err_ = (expr);                                     \
    if (err_ != hipSuccess) {                                           \
      std::fprintf(stderr, "%s:%d hip error: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(err_));                            \
      std::exit(2);                                                     \
    }                                                                   \
  } while (0)

namespace {

namespace kvarn = gufo::testing::kvarn;

constexpr unsigned kHeads = 24, kKvHeads = 4, kGqa = 6, kDim = 256;
constexpr unsigned kGroup = 128, kRowWidth = kKvHeads * kDim;
constexpr unsigned kLayers = 16;
constexpr unsigned kPartialStride = kDim + 2;

template<int VBits>
struct Record {
  static_assert(VBits == 2 || VBits == 4);
  static constexpr std::size_t kKCodes = 0;
  static constexpr std::size_t kVCodes = kKCodes + kGroup * kDim / 2;
  static constexpr std::size_t kVRowBytes = kGroup * VBits / 8;
  static constexpr std::size_t kKScale = kVCodes + kDim * kVRowBytes;
  static constexpr std::size_t kKZero = kKScale + kDim * 2;
  static constexpr std::size_t kKColumn = kKZero + kDim * 2;
  static constexpr std::size_t kVScale = kKColumn + kGroup * 2;
  static constexpr std::size_t kVZero = kVScale + kGroup * 2;
  static constexpr std::size_t kVColumn = kVZero + kGroup * 2;
  static constexpr std::size_t kBytes = kVColumn + kDim * 2;
};
static_assert(Record<2>::kBytes == 26880 && Record<4>::kBytes == 35072);

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
// Group-flush quantizer. One 512-thread block per (KV head, group, layer,
// K or V) tile. Wave w holds tokens 8w..8w+7 and lane l channels 8l..8l+7, so
// per-token statistics reduce inside a wave and per-channel ones across the
// 16 waves through LDS. Standard deviations accumulate in double like the
// reference; log and exp go through double so the float results are
// correctly rounded like glibc's logf/expf in nearly all cases.
// ---------------------------------------------------------------------------
template<int VBits>
__launch_bounds__(512) __global__
    void KvarnQuantizeKernel(const __half* __restrict__ k_rows,
                             const __half* __restrict__ v_rows,
                             std::uint8_t* __restrict__ records,
                             std::size_t rows_layer_stride,
                             std::size_t records_layer_stride,
                             unsigned first_group) {
  using R = Record<VBits>;
  constexpr unsigned kWaves = 16;
  __shared__ double channel_part[kWaves][kDim];
  __shared__ double channel_result[kDim];
  __shared__ float block_part[kWaves][4];
  __shared__ unsigned value_codes[kDim * kGroup * VBits / 32];

  const unsigned kv = blockIdx.x;
  const unsigned group = first_group + blockIdx.y;
  const unsigned layer = blockIdx.z >> 1;
  const bool key = (blockIdx.z & 1) == 0;
  const unsigned tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
  const __half* src = (key ? k_rows : v_rows) + layer * rows_layer_stride +
                      static_cast<std::size_t>(group) * kGroup * kRowWidth +
                      kv * kDim;
  std::uint8_t* record =
      records + layer * records_layer_stride +
      (static_cast<std::size_t>(group - 1) * kKvHeads + kv) * R::kBytes;

  // [token i][channel pair]: the rotated FP16 source, packed.
  h2 x[8][4];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) {
    const uint4 raw = *reinterpret_cast<const uint4*>(
        src + static_cast<std::size_t>(8 * wave + i) * kRowWidth + 8 * lane);
    const auto* halves = reinterpret_cast<const __half*>(&raw);
    float y[8];
#pragma unroll
    for (unsigned j = 0; j < 8; ++j)
      y[j] = __half2float(halves[j]);
    Fwht256(y, lane);
#pragma unroll
    for (unsigned j = 0; j < 4; ++j)
      x[i][j] = h2{static_cast<_Float16>(y[2 * j] * 0.0625F),
                   static_cast<_Float16>(y[2 * j + 1] * 0.0625F)};
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

  float tstd[8], cstd[8];
  token_std(tstd);
  channel_std(cstd);
  float best = imbalance(tstd, cstd);
  for (int iteration = 0; iteration < 8; ++iteration) {
    // Column step (K: tokens, V: channels) on the current tile.
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
      if (key)
        token_scale[i] = update(token_log[i], tstd[i]);
      else
        channel_scale[i] = update(channel_log[i], cstd[i]);
    }
    apply(token_scale, channel_scale);
    if (key) {
      channel_std(cstd);
#pragma unroll
      for (unsigned j = 0; j < 8; ++j)
        channel_scale[j] = update(channel_log[j], cstd[j]);
    } else {
      token_std(tstd);
#pragma unroll
      for (unsigned i = 0; i < 8; ++i)
        token_scale[i] = update(token_log[i], tstd[i]);
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
// Fused-dequant split-K decode attention partials. One 256-thread block per
// KV head and split; the block serves all 6 * Rows query rows of that KV
// head (row m = query row * 6 + head in group), padded to 16-row WMMA tiles.
// Splits [0, splits) take balanced runs of quantized groups; split `splits`
// takes the FP16 sink and split `splits + 1` the FP16 tail. Per 128-token
// chunk: wave w computes the scores of tokens 16w..16w+15 with WMMA (B from
// dequantized K), the block runs the online softmax (P as FP16 in LDS), and
// wave w accumulates PV for channels 16w.. and 16(w+8).. (B from dequantized
// V; the per-channel column scale is applied to the chunk's product).
//
// K1b-opt (Opt = true, same storage): codes become WMMA halves with one
// shift and one v_and_or per pair at most, and no subtraction. A code c
// masked at mantissa bit s, ORed with the exponent of 2^(10 - s), is the
// half 2^(10 - s) + c exactly. Every code is moved to s = 6 (bias 16) or,
// for 2-bit codes, s = 8 (bias 4): a first version kept codes in place at
// s = 0..8 (bias up to 1024) and lost 1e-3 of relative accuracy to the FP32
// cancellation of the bias. The bias is a constant per token (V) or
// channel (K), subtracted once per row in FP32: for V inside the existing
// P * zero sum, for K (AR only) in a per-group row correction. AR keys fold
// the per-channel K scale into the query instead of the codes (A = q *
// scale as FP16, rebuilt per group in LDS), and the zero term sum_c q_c *
// zero_c joins the same correction. The softmax runs in base 2 (log2 e
// folded into the score scale, v_exp_f32 directly), the padding rows of AR
// tiles are skipped, and the last block of each KV head combines the splits
// (no reduce launch).
// ---------------------------------------------------------------------------
constexpr float kLog2e = 1.4426950408889634F;

// Bias of pair j's code half (see BiasedCodes).
template<int Bits>
__host__ __device__ constexpr float CodeBias(unsigned pair) {
  if constexpr (Bits == 2)
    return (pair == 2 || pair == 4 || pair == 6) ? 4.0F : 16.0F;
  else
    return 16.0F;
}

// One 32-bit code word -> its pairs as packed halves bias + code. Pair j is
// the codes of nibble (crumb) j of both 16-bit halves; the shifts never move
// a bit of one half into the masked field of the other.
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
  const std::uint8_t* records;  // [group - 1][kv head][record]
  const __half* sink_k;         // [128][4][256], rotated
  const __half* sink_v;
  const __half* tail_k;  // [tail_tokens][4][256], rotated
  const __half* tail_v;
  float* scratch;   // [kv][split][row m][max, sum, 256 outputs]
  unsigned groups;  // quantized groups after the sink
  unsigned tail_tokens;
  unsigned splits;   // quantized splits; the sink and tail add two more
  unsigned only{0};  // probe: 1 quantized blocks only, 2 sink/tail only
  // Opt: the output [rows][24][256] and one zeroed arrival counter per KV
  // head (left zeroed by the combining block).
  float* out{nullptr};
  unsigned* counters{nullptr};
};

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
// B fragment live at once (128 VGPRs for 16 steps). This removed all spills
// of the eight-row kernels and took AR from 230 to 189 VGPRs (8 waves/SIMD).
template<unsigned N>
__device__ __forceinline__ void Pin(v8f (&acc)[N]) {
  if constexpr (N == 3)
    asm volatile("" : "+v"(acc[0]), "+v"(acc[1]), "+v"(acc[2]));
  else if constexpr (N == 2)
    asm volatile("" : "+v"(acc[0]), "+v"(acc[1]));
  else
    asm volatile("" : "+v"(acc[0]));
}

// Opt split combine of row m (of M) of one KV head by one wave: lane l
// owns output channels 8l..8l+7; the maxima are in base 2.
__device__ __forceinline__ void CombineRow(const float* scratch, float* out,
                                           unsigned kv, unsigned m, unsigned M,
                                           unsigned nsplit, unsigned lane) {
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
      sum += p[1] * e;
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
          y[2 * k] += w[j] * x[j][k].x;
          y[2 * k + 1] += w[j] * x[j][k].y;
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
  float* dst =
      out +
      (static_cast<std::size_t>(m / kGqa) * kHeads + kv * kGqa + m % kGqa) *
          kDim +
      8 * lane;
  reinterpret_cast<float4*>(dst)[0] =
      float4{y[0] * 0.0625F, y[1] * 0.0625F, y[2] * 0.0625F, y[3] * 0.0625F};
  reinterpret_cast<float4*>(dst)[1] =
      float4{y[4] * 0.0625F, y[5] * 0.0625F, y[6] * 0.0625F, y[7] * 0.0625F};
}

// Opt verification: one wave per (KV head, row), 8 per block.
__launch_bounds__(256) __global__
    void KvarnCombineKernel(const float* __restrict__ scratch,
                            float* __restrict__ out, unsigned nsplit,
                            unsigned rows) {
  const unsigned M = kGqa * rows;
  const unsigned pair = blockIdx.x * 8 + threadIdx.x / 32;
  if (pair < kKvHeads * M)
    CombineRow(scratch, out, pair / M, pair % M, M, nsplit, threadIdx.x % 32);
}

// Gen 0: K1b, 1: Opt, 2: Opt with AR V codes loaded at the start of their
// own group instead of one group ahead (fewer live registers).
template<int VBits, unsigned Rows, int Gen>
__launch_bounds__(256) __global__ void KvarnDecodePartialsKernel(DecodeArgs a) {
  constexpr bool Opt = Gen >= 1;
  using R = Record<VBits>;
  constexpr unsigned M = kGqa * Rows, MT = (M + 15) / 16, MR = MT * 16;
  constexpr unsigned kQStride = kDim + 8, kPStride = kGroup + 8;
  constexpr unsigned kVWords = kGroup * VBits / 32;  // words per channel
  // AR prefetches the whole next group into registers; verification, with
  // three times the accumulators, only its K scale pairs.
  constexpr bool kPrefetch = Rows == 1;
  constexpr bool kPrefetchV = kPrefetch && Gen < 2;
  // AR folds the K scale into q (verification would need 48 correction
  // rows and a second q copy past the LDS budget).
  constexpr bool kFoldK = Opt && Rows == 1;
  constexpr float kScoreScale = Opt ? 0.0625F * kLog2e : 0.0625F;
  const auto exp_ = [](float x) {
    if constexpr (Opt)
      return __builtin_amdgcn_exp2f(x);
    else
      return expf(x);
  };
  __shared__ __align__(16) _Float16 q16[MR * kQStride];
  __shared__ __align__(16) _Float16 p16[MR * kPStride];
  __shared__ __align__(16) float wave_max[MR][8];
  __shared__ __align__(16) float wave_psum[MR][8];  // sum of P per wave
  // Sum of P * V zero (Opt: minus FP16 P * code bias).
  __shared__ __align__(16) float wave_zsum[MR][8];
  __shared__ __align__(16) unsigned key_scale[kFoldK ? 1 : kDim / 2],
      key_zero[kFoldK ? 1 : kDim / 2];
  // kFoldK: per-wave parts of sum_c q_c zero_c - q'_c bias_c per row.
  __shared__ __align__(16) float corr_part[kFoldK ? M : 1][8];
  // Online-softmax state per row: the max double-buffered by chunk parity,
  // the sum (wave 0 only) and the chunk's rescale factor.
  __shared__ float row_max[2][MR], row_sum[MR], row_alpha[MR];

  const unsigned kv = blockIdx.x, split = blockIdx.z;
  if ((a.only == 1 && split >= a.splits) || (a.only == 2 && split < a.splits))
    return;
  const unsigned tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
  const unsigned sub = lane % 16, half = lane / 16;
  const unsigned length = kGroup * (1 + a.groups) + a.tail_tokens;
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
    if (kFoldK && split < a.splits) {
      // Quantized AR blocks fold from float q: rows < M as [M][256] floats
      // in the q16 storage, read back per channel below.
      if (m < M) {
        auto* qf = reinterpret_cast<float4*>(q16) + (m * kDim + 8 * lane) / 4;
        qf[0] = float4{y[0] * 0.0625F, y[1] * 0.0625F, y[2] * 0.0625F,
                       y[3] * 0.0625F};
        qf[1] = float4{y[4] * 0.0625F, y[5] * 0.0625F, y[6] * 0.0625F,
                       y[7] * 0.0625F};
      }
      continue;
    }
    _Float16 h[8];
#pragma unroll
    for (unsigned j = 0; j < 8; ++j)
      h[j] = static_cast<_Float16>(y[j] * 0.0625F);
    *reinterpret_cast<uint4*>(&q16[m * kQStride + 8 * lane]) =
        *reinterpret_cast<const uint4*>(h);
  }
  static_assert(!kFoldK || M * kDim * 4 <= sizeof(q16));
  if constexpr (Opt)
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

  // Scores in acc (scaled, masked) -> P * V scale of this lane's token in
  // LDS, per-wave sums of P and P * V zero, the row state for this chunk.
  // `between` runs after the first barrier, when no wave reads scores
  // inputs any more.
  // Opt skips the padding rows of a tile (M is even, so 2i < M - 16 mt
  // covers both half-waves); their P stays zero and their outputs unused.
  const auto Live = [](unsigned mt, unsigned i) {
    return !Opt || mt * 16 + 2 * i < M;
  };
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
          row_alpha[row] = empty ? 1.0F : exp_(previous - next);
        }
        const float s = acc[mt][i];
        const float p = (empty || s == -INFINITY) ? 0.0F : exp_(s - next);
        const _Float16 ph = static_cast<_Float16>(p * value_scale);
        p16[row * kPStride + token] = ph;
        const float psum = RowSum16(p);
        float z = p * value_zero;
        if constexpr (Opt)
          z = fmaf(-static_cast<float>(ph), value_bias, z);
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
          row_sum[row] = row_sum[row] * row_alpha[row] + s;
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
  const auto values = [&](auto unroll, auto build_v, const float (&column)[2]) {
    // AR Opt: the zero sums once per chunk, live rows only (verification
    // would hold 24 more registers across both products).
    constexpr bool kZerosOnce = Opt && Rows == 1;
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
      steps(unroll, kGroup / 16, [&](unsigned kt) {
        const v16h b = build_v(t, kt);
#pragma unroll
        for (unsigned mt = 0; mt < MT; ++mt)
          g[mt] = Wmma(LoadFrag(&p16[(mt * 16 + sub) * kPStride + 16 * kt]), b,
                       g[mt]);
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
          out[mt][t][i] =
              row_alpha[row] * out[mt][t][i] + column[t] * (g[mt][i] + zero);
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
      return a.records +
             (static_cast<std::size_t>(group) * kKvHeads + kv) * R::kBytes;
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
      if constexpr (!kFoldK)
        if (tid < 64)
          reinterpret_cast<uint4*>(tid < 32 ? key_scale : key_zero)[tid % 32] =
              v;
    };
    // kFoldK: thread tid owns channel tid: its rotated q column, and per
    // group its K scale (low half) and zero (high half).
    float qcol[kFoldK ? M : 1];
    const auto load_kscale = [&](unsigned group) {
      const auto* h = reinterpret_cast<const unsigned short*>(record_of(group));
      return static_cast<unsigned>(h[R::kKScale / 2 + tid]) |
             static_cast<unsigned>(h[R::kKZero / 2 + tid]) << 16;
    };
    // q' = q * scale into q16 rows < M and the per-wave correction parts.
    const auto fold = [&](unsigned sz) {
      const float s = __half2float(__ushort_as_half(sz & 0xffff));
      const float z = __half2float(__ushort_as_half(sz >> 16));
      const float bias = CodeBias<4>((tid % 8) / 2);
#pragma unroll
      for (unsigned m = 0; m < M; ++m) {
        const float q = qcol[m];
        const float exact = q * s;
        const _Float16 qp = static_cast<_Float16>(exact);
        q16[m * kQStride + tid] = qp;
        // The FP16 rounding of q' times the mid code 7.5 moves into the
        // correction (exact in FP32), leaving it times (code - 7.5) only:
        // without this, the per-group rounding offset cost 1.5x the K1b
        // kernel error.
        const float qpf = static_cast<float>(qp);
        float t = fmaf(q, z, fmaf(exact - qpf, 7.5F, -qpf * bias));
        t = RowSum16(t);
        t += __shfl_xor(t, 16);
        if (lane == 0)
          corr_part[m][wave] = t;
      }
    };
    if constexpr (kFoldK) {
      LdsBarrier();
#pragma unroll
      for (unsigned m = 0; m < M; ++m)
        qcol[m] = reinterpret_cast<const float*>(q16)[m * kDim + tid];
      LdsBarrier();
      // Zero padding rows of the folded A tile (rows < M come from fold).
      for (unsigned i = tid; i < (MR - M) * kQStride; i += 256)
        q16[M * kQStride + i] = static_cast<_Float16>(0.0F);
      if (begin < end)
        fold(load_kscale(begin));
    }
    // Opt: the code bias of this lane's token in the V fragments.
    const float value_bias =
        !Opt ? 0.0F
             : (VBits == 2 ? CodeBias<2>(sub / 2) : CodeBias<4>((sub % 8) / 2));
    KeyPart keys{};
    ValuePart vals{};
    if (begin < end) {
      stage_scales(load_scales(begin));
      if constexpr (kPrefetch)
        load_keys(begin, keys);
      if constexpr (kPrefetchV)
        load_values(begin, vals);
    }
    LdsBarrier();
    for (unsigned group = begin; group < end; ++group) {
      const bool more = group + 1 < end;
      const uint4 next_scales =
          (!kFoldK && more) ? load_scales(group + 1) : uint4{};
      const unsigned next_sz = (kFoldK && more) ? load_kscale(group + 1) : 0U;
      KeyPart next_keys{};
      ValuePart next_vals{};
      if constexpr (kPrefetch) {
        if (more)
          load_keys(group + 1, next_keys);
      } else {
        load_keys(group, keys);
      }
      if constexpr (kPrefetchV) {
        if (more)
          load_values(group + 1, next_vals);
      } else {
        load_values(group, vals);
      }

      v8f acc[MT];
      scores(
          std::true_type{},
          [&](unsigned ks) {
            const uint4 v = keys.codes[ks / 2];
            const unsigned w0 = (ks & 1) != 0 ? v.z : v.x;
            const unsigned w1 = (ks & 1) != 0 ? v.w : v.y;
            if constexpr (kFoldK) {
              u32x8 b;
              unsigned lo[4], hi[4];
              BiasedCodes<4>(w0, lo);
              BiasedCodes<4>(w1, hi);
#pragma unroll
              for (unsigned j = 0; j < 4; ++j) {
                b[j] = lo[j];
                b[4 + j] = hi[j];
              }
              return __builtin_bit_cast(v16h, b);
            }
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
      if constexpr (kFoldK) {
#pragma unroll
        for (unsigned mt = 0; mt < MT; ++mt)
#pragma unroll
          for (unsigned i = 0; i < 8; ++i)
            if (mt * 16 + 2 * i < M) {  // M even: row < M for both halves
              const unsigned row = mt * 16 + 2 * i + half;
              const float4 c0 =
                  *reinterpret_cast<const float4*>(&corr_part[row][0]);
              const float4 c1 =
                  *reinterpret_cast<const float4*>(&corr_part[row][4]);
              acc[mt][i] += ((c0.x + c0.y) + (c0.z + c0.w)) +
                            ((c1.x + c1.y) + (c1.z + c1.w));
            }
      }
#pragma unroll
      for (unsigned mt = 0; mt < MT; ++mt)
        acc[mt] *= keys.column;
      // The next group's K scales (kFoldK: q') replace this group's once
      // every wave is past its scores; the second softmax barrier publishes
      // them.
      softmax(acc, (group - begin) & 1, keys.value_scale, keys.value_zero,
              value_bias, [&] {
                if (more) {
                  if constexpr (kFoldK)
                    fold(next_sz);
                  else
                    stage_scales(next_scales);
                }
              });
      values(
          std::true_type{},
          [&](unsigned t, unsigned kt) {
            u32x8 b;
            if constexpr (Opt && VBits == 2) {
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
            } else if constexpr (Opt) {
              const uint4 v = vals.codes[t][kt / 2];
              unsigned lo[4], hi[4];
              BiasedCodes<4>((kt & 1) != 0 ? v.z : v.x, lo);
              BiasedCodes<4>((kt & 1) != 0 ? v.w : v.y, hi);
#pragma unroll
              for (unsigned j = 0; j < 4; ++j) {
                b[j] = lo[j];
                b[4 + j] = hi[j];
              }
            } else if constexpr (VBits == 2) {
              const uint4 v = vals.codes[t][kt / 4];
              const unsigned w = kt % 4 == 0   ? v.x
                                 : kt % 4 == 1 ? v.y
                                 : kt % 4 == 2 ? v.z
                                               : v.w;
#pragma unroll
              for (unsigned j = 0; j < 8; ++j)
                b[j] = Codes2((w >> (2 * j)) & 0x00030003U);
            } else {
              const uint4 v = vals.codes[t][kt / 2];
              const unsigned w0 = (kt & 1) != 0 ? v.z : v.x;
              const unsigned w1 = (kt & 1) != 0 ? v.w : v.y;
#pragma unroll
              for (unsigned j = 0; j < 4; ++j) {
                b[j] = Codes2((w0 >> (4 * j)) & 0x000F000FU);
                b[4 + j] = Codes2((w1 >> (4 * j)) & 0x000F000FU);
              }
            }
            return __builtin_bit_cast(v16h, b);
          },
          vals.column);
      if constexpr (kPrefetch)
        keys = next_keys;
      if constexpr (kPrefetchV)
        vals = next_vals;
    }
    final_parity = (end - begin) & 1;
  } else {
    // FP16 rows: the sink or the tail, in chunks of up to 128 tokens.
    LdsBarrier();
    const bool sink = split == a.splits;
    const unsigned total = sink ? kGroup : a.tail_tokens;
    const unsigned first = sink ? 0 : kGroup * (1 + a.groups);
    const __half* rows_k = sink ? a.sink_k : a.tail_k;
    const __half* rows_v = sink ? a.sink_v : a.tail_v;
    const float ones[2] = {1.0F, 1.0F};
    for (unsigned chunk = 0; chunk < total; chunk += kGroup) {
      const unsigned count = min(kGroup, total - chunk);
      const std::size_t chunk_offset =
          static_cast<std::size_t>(chunk) * kRowWidth + kv * kDim;
      const __half* vbase = rows_v + chunk_offset;
      const __half* krow =
          rows_k + chunk_offset +
          static_cast<std::size_t>(token < count ? token : 0) * kRowWidth;
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
      const unsigned position = first + chunk + token;
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
      softmax(acc, (chunk / kGroup) & 1, 1.0F, 0.0F, 0.0F, [] {});
      values(
          std::false_type{},
          [&](unsigned t, unsigned kt) {
            const unsigned channel = 16 * (wave + 8 * t) + sub;
            v16h b;
#pragma unroll
            for (unsigned k = 0; k < 16; ++k) {
              const unsigned tk = 16 * kt + k;
              b[k] = tk < count
                         ? static_cast<_Float16>(__half2float(
                               vbase[static_cast<std::size_t>(tk) * kRowWidth +
                                     channel]))
                         : static_cast<_Float16>(0.0F);
            }
            return b;
          },
          ones);
    }
    final_parity = ((total + kGroup - 1) / kGroup) & 1;
  }

  const unsigned nsplit = a.splits + 2;
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
  if constexpr (Opt && Rows == 1) {
    // The last block of this KV head to finish combines its splits.
    if (a.only != 0)
      return;
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
      CombineRow(a.scratch, a.out, kv, m, M, nsplit, lane);
  }
}

// Combines the splits of one (query row, head), normalizes, rotates back.
// Exp2: the partial maxima are in base 2 (Opt partials).
template<bool Exp2>
__launch_bounds__(256) __global__
    void KvarnReduceKernel(const float* __restrict__ scratch,
                           float* __restrict__ out, unsigned nsplit,
                           unsigned rows) {
  __shared__ float buffer[kDim];
  const unsigned head = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
  const unsigned kv = head / kGqa, m = r * kGqa + head % kGqa;
  const unsigned M = kGqa * rows;
  const auto at = [&](unsigned s) {
    return scratch + ((static_cast<std::size_t>(kv) * nsplit + s) * M + m) *
                         kPartialStride;
  };
  float global_max = -INFINITY;
  for (unsigned s = 0; s < nsplit; ++s)
    global_max = fmaxf(global_max, at(s)[0]);
  float sum = 0, value = 0;
  for (unsigned s = 0; s < nsplit; ++s) {
    const float* p = at(s);
    const float e = p[0] == -INFINITY ? 0.0F
                    : Exp2 ? __builtin_amdgcn_exp2f(p[0] - global_max)
                           : expf(p[0] - global_max);
    sum += p[1] * e;
    value += p[2 + d] * e;
  }
  buffer[d] = value / sum;
  for (unsigned length = 1; length < kDim; length <<= 1) {
    __syncthreads();
    const float lo = buffer[d & ~length], hi = buffer[d | length];
    __syncthreads();
    buffer[d] = (d & length) != 0 ? lo - hi : lo + hi;
  }
  __syncthreads();
  out[(static_cast<std::size_t>(r) * kHeads + head) * kDim + d] =
      buffer[d] * 0.0625F;
}

// parts: 1 partials, 2 reduce, 3 both.
unsigned* ArrivalCounters() {
  static unsigned* counters = [] {
    unsigned* p = nullptr;
    KV_OK(hipMalloc(&p, kKvHeads * sizeof(unsigned)));
    KV_OK(hipMemset(p, 0, kKvHeads * sizeof(unsigned)));
    return p;
  }();
  return counters;
}

// Opt AR combines the splits inside the partials launch (parts 2 empty).
template<int VBits, int Gen>
void LaunchKvarn(DecodeArgs a, unsigned rows, float* out, unsigned parts = 3) {
  constexpr bool Opt = Gen >= 1;
  const dim3 grid(kKvHeads, 1, a.splits + 2);
  if constexpr (Opt) {
    a.out = out;
    a.counters = ArrivalCounters();
    if ((parts & 1) != 0) {
      if (rows == 1)
        KvarnDecodePartialsKernel<VBits, 1, Gen><<<grid, 256>>>(a);
      else if (rows == 8)
        KvarnDecodePartialsKernel<VBits, 8, Gen><<<grid, 256>>>(a);
      else
        std::abort();
    }
    // AR combines inside the partials launch; verification (48 rows per
    // KV head, too many for one block) in a wave-per-row launch.
    if (rows > 1 && (parts & 2) != 0)
      KvarnCombineKernel<<<(kKvHeads * kGqa * rows + 7) / 8, 256>>>(
          a.scratch, out, a.splits + 2, rows);
    return;
  }
  if ((parts & 1) != 0) {
    if (rows == 1)
      KvarnDecodePartialsKernel<VBits, 1, Gen><<<grid, 256>>>(a);
    else if (rows == 8)
      KvarnDecodePartialsKernel<VBits, 8, Gen><<<grid, 256>>>(a);
    else
      std::abort();
  }
  if ((parts & 2) != 0)
    KvarnReduceKernel<Opt>
        <<<dim3(kHeads, rows), 256>>>(a.scratch, out, a.splits + 2, rows);
}

// Kernel generations compared in one binary.
enum class Kernel { kK1b, kOpt, kOptLateV };
constexpr Kernel kKernels[] = {Kernel::kK1b, Kernel::kOpt, Kernel::kOptLateV};
const char* KernelName(Kernel k) {
  return k == Kernel::kK1b ? "k1b" : k == Kernel::kOpt ? "opt" : "opt_latev";
}

void LaunchKvarnAny(int bits, Kernel kernel, const DecodeArgs& a, unsigned rows,
                    float* out, unsigned parts = 3) {
  const auto go = [&](auto bits_c) {
    constexpr int B = decltype(bits_c)::value;
    if (kernel == Kernel::kK1b)
      LaunchKvarn<B, 0>(a, rows, out, parts);
    else if (kernel == Kernel::kOpt)
      LaunchKvarn<B, 1>(a, rows, out, parts);
    else
      LaunchKvarn<B, 2>(a, rows, out, parts);
  };
  if (bits == 2)
    go(std::integral_constant<int, 2>{});
  else
    go(std::integral_constant<int, 4>{});
}

template<int VBits>
void LaunchQuantize(const __half* k, const __half* v, std::uint8_t* records,
                    std::size_t rows_layer_stride,
                    std::size_t records_layer_stride, unsigned first_group,
                    unsigned groups, unsigned layers) {
  KvarnQuantizeKernel<VBits><<<dim3(kKvHeads, groups, 2 * layers), 512>>>(
      k, v, records, rows_layer_stride, records_layer_stride, first_group);
}

void LaunchFp16(const float* q, const __half* k, const __half* v, float* out,
                float* scratch, std::size_t scratch_size, unsigned length,
                unsigned rows) {
  gufo::hip::LaunchCausalDecodeAttention(
      q, nullptr, nullptr, nullptr, const_cast<__half*>(k),
      const_cast<__half*>(v), gufo::hip::KvBlockTable{}, out, 0, length - rows,
      rows, length, kHeads, kKvHeads, kDim, nullptr, {scratch, scratch_size});
}

std::size_t Fp16ScratchSize(unsigned rows) {
  return rows * gufo::hip::detail::DecodeAttentionScratchElements(kHeads, kDim);
}
std::size_t KvarnScratchSize(unsigned rows, unsigned splits) {
  return static_cast<std::size_t>(kKvHeads) * (splits + 2) * kGqa * rows *
         kPartialStride;
}

// ---------------------------------------------------------------------------
// Host helpers.
// ---------------------------------------------------------------------------
template<typename T>
struct Device {
  T* data{nullptr};
  std::size_t size{0};
  explicit Device(std::size_t n) : size(n) {
    KV_OK(hipMalloc(&data, n * sizeof(T)));
  }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  ~Device() { (void)hipFree(data); }
  void Upload(const std::vector<T>& v) {
    KV_OK(
        hipMemcpy(data, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
  }
  std::vector<T> Download() const {
    std::vector<T> v(size);
    KV_OK(hipMemcpy(v.data(), data, size * sizeof(T), hipMemcpyDeviceToHost));
    return v;
  }
};

void ParallelFor(std::size_t n, const std::function<void(std::size_t)>& f) {
  const unsigned threads = std::max(1U, std::thread::hardware_concurrency());
  std::atomic<std::size_t> next{0};
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < threads; ++t)
    pool.emplace_back([&] {
      for (std::size_t i; (i = next++) < n;)
        f(i);
    });
  for (auto& thread : pool)
    thread.join();
}

void HadamardDouble(double* x) {
  for (unsigned length = 1; length < kDim; length <<= 1)
    for (unsigned i = 0; i < kDim; i += length << 1)
      for (unsigned j = i; j < i + length; ++j) {
        const double a = x[j], b = x[j + length];
        x[j] = a + b;
        x[j + length] = a - b;
      }
  for (unsigned i = 0; i < kDim; ++i)
    x[i] /= 16.0;
}

// Causal attention in double. key/value(t, kv) return a 256-float row.
std::vector<double> Attention(
    const std::vector<float>& q, unsigned rows, unsigned length,
    const std::function<const float*(unsigned, unsigned)>& key,
    const std::function<const float*(unsigned, unsigned)>& value, bool rotate) {
  std::vector<double> out(static_cast<std::size_t>(rows) * kHeads * kDim);
  ParallelFor(rows * kHeads, [&](std::size_t index) {
    const unsigned r = index / kHeads, head = index % kHeads, kv = head / kGqa;
    double qv[kDim];
    for (unsigned d = 0; d < kDim; ++d)
      qv[d] = q[index * kDim + d];
    if (rotate)
      HadamardDouble(qv);
    const unsigned last = length - rows + r;
    std::vector<double> s(last + 1);
    double mx = -INFINITY;
    for (unsigned t = 0; t <= last; ++t) {
      const float* k = key(t, kv);
      double dot = 0;
      for (unsigned d = 0; d < kDim; ++d)
        dot += qv[d] * k[d];
      s[t] = dot / 16.0;
      mx = std::max(mx, s[t]);
    }
    double sum = 0, o[kDim] = {};
    for (unsigned t = 0; t <= last; ++t) {
      const double p = std::exp(s[t] - mx);
      sum += p;
      const float* v = value(t, kv);
      for (unsigned d = 0; d < kDim; ++d)
        o[d] += p * v[d];
    }
    for (unsigned d = 0; d < kDim; ++d)
      o[d] /= sum;
    if (rotate)
      HadamardDouble(o);
    std::copy(o, o + kDim, out.begin() + index * kDim);
  });
  return out;
}

struct Diff {
  double rel{0}, max_abs{0};
};
template<typename A>
Diff Compare(const A& got, const std::vector<double>& ref) {
  double num = 0, den = 0, mx = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - ref[i];
    if (!std::isfinite(d))
      return {INFINITY, INFINITY};
    num += d * d;
    den += ref[i] * ref[i];
    mx = std::max(mx, std::abs(d));
  }
  return {std::sqrt(num / den), mx};
}

// CPU tile quantization by the K0 rules, keeping codes and stored scales.
struct RefTile {
  std::vector<std::uint8_t> codes;  // logical [rows][cols]
  std::vector<std::uint16_t> scale, zero, column;
};
// The GPU quantizer's balancing in its own reduction order: double
// partial sums per lane, the lane butterfly, the per-wave channel partials
// in wave order, and log/exp rounded from double. With glibc logf/expf in
// place of the double-rounded ones this reproduces kvarn::Balance bit for
// bit (checked on 800 tiles), so the GPU-vs-reference differences that
// remain are the last-bit rounding of libm's logf/expf.
void EmulateBalance(std::vector<float>& tile, bool key,
                    std::vector<float>& column_scale,
                    std::vector<float>& row_scale) {
  const unsigned cols = key ? kGroup : kDim;
  const auto source = [&](unsigned t, unsigned c) {
    return key ? tile[c * cols + t] : tile[t * cols + c];
  };
  std::vector<float> ts(kGroup, 1), cs(kDim, 1), tl(kGroup, 0), cl(kDim, 0);
  std::vector<float> bt(kGroup, 1), bc(kDim, 1), cur(kGroup * kDim);
  const auto apply = [&](const std::vector<float>& t_,
                         const std::vector<float>& c_) {
    for (unsigned t = 0; t < kGroup; ++t)
      for (unsigned c = 0; c < kDim; ++c)
        cur[t * kDim + c] =
            key ? source(t, c) / t_[t] / c_[c] : source(t, c) / c_[c] / t_[t];
  };
  const auto butterfly = [](double* v) {
    for (unsigned m = 1; m < 32; m <<= 1) {
      double n[32];
      for (unsigned l = 0; l < 32; ++l)
        n[l] = v[l] + v[l ^ m];
      std::copy(n, n + 32, v);
    }
    return v[0];
  };
  const auto token_std = [&](std::vector<float>& out) {
    for (unsigned t = 0; t < kGroup; ++t) {
      double p[32];
      for (unsigned l = 0; l < 32; ++l) {
        double s = 0;
        for (unsigned j = 0; j < 8; ++j)
          s += cur[t * kDim + 8 * l + j];
        p[l] = s;
      }
      const double mean = butterfly(p) / kDim;
      for (unsigned l = 0; l < 32; ++l) {
        double s = 0;
        for (unsigned j = 0; j < 8; ++j) {
          const double d = cur[t * kDim + 8 * l + j] - mean;
          s += d * d;
        }
        p[l] = s;
      }
      out[t] = static_cast<float>(std::sqrt(butterfly(p) / (kDim - 1)));
    }
  };
  const auto channel_std = [&](std::vector<float>& out) {
    for (unsigned c = 0; c < kDim; ++c) {
      double s = 0;
      for (unsigned w = 0; w < 16; ++w) {
        double q = 0;
        for (unsigned i = 0; i < 8; ++i)
          q += cur[(8 * w + i) * kDim + c];
        s += q;
      }
      const double mean = s / kGroup;
      s = 0;
      for (unsigned w = 0; w < 16; ++w) {
        double q = 0;
        for (unsigned i = 0; i < 8; ++i) {
          const double d = cur[(8 * w + i) * kDim + c] - mean;
          q += d * d;
        }
        s += q;
      }
      out[c] = static_cast<float>(std::sqrt(s / (kGroup - 1)));
    }
  };
  const auto imbalance = [&](const std::vector<float>& a,
                             const std::vector<float>& b) {
    float tmax = 0, tmin = INFINITY, cmax = 0, cmin = INFINITY;
    for (const float v : a) {
      tmax = std::max(tmax, v);
      tmin = std::min(tmin, v);
    }
    for (const float v : b) {
      cmax = std::max(cmax, v);
      cmin = std::min(cmin, v);
    }
    const float tt = tmax / std::max(tmin, 1e-8F);
    const float ct = cmax / std::max(cmin, 1e-8F);
    return key ? tt + ct : ct + tt;
  };
  const auto update = [](float& l, float s) {
    const float c = std::min(std::max(s, 1e-3F), 1e3F);
    l = std::min(
        std::max(l + static_cast<float>(std::log(static_cast<double>(c))),
                 -0.3F),
        10.0F);
    return static_cast<float>(std::exp(static_cast<double>(l)));
  };
  apply(ts, cs);
  std::vector<float> tsd(kGroup), csd(kDim);
  token_std(tsd);
  channel_std(csd);
  float best = imbalance(tsd, csd);
  for (int it = 0; it < 8; ++it) {
    if (key)
      for (unsigned t = 0; t < kGroup; ++t)
        ts[t] = update(tl[t], tsd[t]);
    else
      for (unsigned c = 0; c < kDim; ++c)
        cs[c] = update(cl[c], csd[c]);
    apply(ts, cs);
    if (key) {
      channel_std(csd);
      for (unsigned c = 0; c < kDim; ++c)
        cs[c] = update(cl[c], csd[c]);
    } else {
      token_std(tsd);
      for (unsigned t = 0; t < kGroup; ++t)
        ts[t] = update(tl[t], tsd[t]);
    }
    apply(ts, cs);
    token_std(tsd);
    channel_std(csd);
    const float v = imbalance(tsd, csd);
    if (v <= best) {
      best = v;
      bt = ts;
      bc = cs;
    }
  }
  column_scale = key ? bt : bc;
  row_scale = key ? bc : bt;
  for (unsigned t = 0; t < kGroup; ++t)
    for (unsigned c = 0; c < kDim; ++c)
      (key ? tile[c * cols + t] : tile[t * cols + c]) =
          key ? source(t, c) / bt[t] / bc[c] : source(t, c) / bc[c] / bt[t];
}

RefTile ReferenceQuantize(std::vector<float> tile, std::size_t rows,
                          std::size_t cols, int bits, bool emulate) {
  std::vector<float> column_scale, row_scale;
  if (emulate)
    EmulateBalance(tile, rows == kDim, column_scale, row_scale);
  else
    kvarn::Balance(tile, rows, cols, 8, column_scale, row_scale);
  RefTile out;
  out.codes.resize(rows * cols);
  const float qmax = static_cast<float>((1 << bits) - 1);
  for (std::size_t r = 0; r < rows; ++r) {
    const float* row = tile.data() + r * cols;
    const auto [lo, hi] = std::minmax_element(row, row + cols);
    const float low = *lo;
    const float scale = std::max((*hi - low) / qmax, 1e-10F);
    out.scale.push_back(kvarn::FloatToHalf(row_scale[r] * scale));
    out.zero.push_back(kvarn::FloatToHalf(row_scale[r] * low));
    for (std::size_t c = 0; c < cols; ++c)
      out.codes[r * cols + c] = static_cast<std::uint8_t>(
          std::clamp(std::nearbyint((row[c] - low) / scale), 0.0F, qmax));
  }
  for (std::size_t c = 0; c < cols; ++c)
    out.column.push_back(kvarn::FloatToHalf(column_scale[c]));
  return out;
}

// GPU record -> the same logical form.
template<int VBits>
RefTile Unpack(const std::uint8_t* record, bool key) {
  using R = Record<VBits>;
  RefTile out;
  const auto halves = [&](std::size_t offset, std::size_t n) {
    std::vector<std::uint16_t> v(n);
    std::memcpy(v.data(), record + offset, 2 * n);
    return v;
  };
  if (key) {
    out.codes.resize(kDim * kGroup);
    const auto* words = reinterpret_cast<const std::uint32_t*>(record);
    for (unsigned t = 0; t < kGroup; ++t)
      for (unsigned c = 0; c < kDim; ++c) {
        const unsigned j = c % 8;
        const unsigned nibble = (j % 2 == 0) ? j / 2 : 4 + j / 2;
        out.codes[c * kGroup + t] =
            (words[t * 32 + c / 8] >> (4 * nibble)) & 15;
      }
    out.scale = halves(R::kKScale, kDim);
    out.zero = halves(R::kKZero, kDim);
    out.column = halves(R::kKColumn, kGroup);
  } else {
    out.codes.resize(kGroup * kDim);
    const auto* words =
        reinterpret_cast<const std::uint32_t*>(record + R::kVCodes);
    for (unsigned c = 0; c < kDim; ++c)
      for (unsigned t = 0; t < kGroup; ++t) {
        unsigned code;
        if (VBits == 2) {
          const unsigned u = t % 16;
          const unsigned crumb = (u % 2 == 0) ? u / 2 : 8 + u / 2;
          code = (words[c * 8 + t / 16] >> (2 * crumb)) & 3;
        } else {
          const unsigned u = t % 8;
          const unsigned nibble = (u % 2 == 0) ? u / 2 : 4 + u / 2;
          code = (words[c * 16 + t / 8] >> (4 * nibble)) & 15;
        }
        out.codes[t * kDim + c] = code;
      }
    out.scale = halves(R::kVScale, kGroup);
    out.zero = halves(R::kVZero, kGroup);
    out.column = halves(R::kVColumn, kDim);
  }
  return out;
}

// Dequantized row-major logical tile in the rotated frame, float, from a
// RefTile: (code * scale + zero) * column.
std::vector<float> Dequant(const RefTile& t, std::size_t rows,
                           std::size_t cols) {
  std::vector<float> out(rows * cols);
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t c = 0; c < cols; ++c)
      out[r * cols + c] = (static_cast<float>(t.codes[r * cols + c]) *
                               kvarn::HalfToFloat(t.scale[r]) +
                           kvarn::HalfToFloat(t.zero[r])) *
                          kvarn::HalfToFloat(t.column[c]);
  return out;
}

// ---------------------------------------------------------------------------
// check: quantizer against the K0 reference, attention against CPU attention.
// ---------------------------------------------------------------------------
struct CheckTotals {
  bool ok{true};
};

template<int VBits>
void CheckCase(unsigned groups, unsigned tail, unsigned splits,
               CheckTotals& totals) {
  const unsigned length = kGroup * (1 + groups) + tail;
  const std::size_t elems = static_cast<std::size_t>(length) * kRowWidth;
  std::mt19937 rng(1234 + groups * 7 + tail + VBits);
  std::normal_distribution<float> normal;
  // K with per-channel offsets and a few outlier channels, V plain.
  std::vector<float> channel_gain(kRowWidth), channel_mean(kRowWidth);
  for (unsigned i = 0; i < kRowWidth; ++i) {
    const bool outlier = (i * 2654435761U >> 7) % 37 == 0;
    channel_gain[i] = outlier ? 6.0F : 0.5F + 0.5F * std::abs(normal(rng));
    channel_mean[i] = outlier ? 3.0F * normal(rng) : 0.3F * normal(rng);
  }
  std::vector<std::uint16_t> k(elems), v(elems);
  for (std::size_t i = 0; i < elems; ++i) {
    k[i] = kvarn::FloatToHalf(normal(rng) * channel_gain[i % kRowWidth] +
                              channel_mean[i % kRowWidth]);
    v[i] = kvarn::FloatToHalf(normal(rng));
  }

  // K0 fake quantization (original frame) of groups [1, groups + 1).
  const auto config =
      *kvarn::ParseConfig(VBits == 2 ? "kvarn_k4v2_g128" : "kvarn_k4v4_g128");
  std::vector<std::uint16_t> kf = k, vf = v;
  kvarn::FakeQuantize(kf, kKvHeads, kDim, true, config, 1, groups + 1);
  kvarn::FakeQuantize(vf, kKvHeads, kDim, false, config, 1, groups + 1);

  // Rotated FP16 rows, as a rotating KV write would store them.
  const auto rotate = [&](const std::vector<std::uint16_t>& src) {
    std::vector<std::uint16_t> out(src.size());
    ParallelFor(src.size() / kDim, [&](std::size_t row) {
      std::vector<float> x(kDim);
      for (unsigned d = 0; d < kDim; ++d)
        x[d] = kvarn::HalfToFloat(src[row * kDim + d]);
      kvarn::Hadamard(x);
      for (unsigned d = 0; d < kDim; ++d)
        out[row * kDim + d] = kvarn::FloatToHalf(x[d]);
    });
    return out;
  };
  const auto krot = rotate(k), vrot = rotate(v);

  // GPU quantization.
  using R = Record<VBits>;
  Device<std::uint16_t> dk(elems), dv(elems);
  dk.Upload(k);
  dv.Upload(v);
  Device<std::uint8_t> drec(std::max<std::size_t>(1, groups) * kKvHeads *
                            R::kBytes);
  if (groups > 0)
    LaunchQuantize<VBits>(reinterpret_cast<const __half*>(dk.data),
                          reinterpret_cast<const __half*>(dv.data), drec.data,
                          0, 0, 1, groups, 1);
  KV_OK(hipDeviceSynchronize());
  const auto records = drec.Download();

  // Tile comparison against the reference rules and the K0 fake quant.
  std::atomic<std::size_t> tiles_exact{0}, code_diff{0}, scale_diff{0},
      max_ulp{0}, fake_mismatch{0}, emulation_mismatch{0};
  // Rotated-frame dequantized rows from the GPU records, float.
  std::vector<float> kq(elems), vq(elems);
  ParallelFor(
      static_cast<std::size_t>(groups) * kKvHeads * 2, [&](std::size_t index) {
        const bool key = index % 2 == 0;
        const unsigned kv = (index / 2) % kKvHeads;
        const unsigned g = index / 2 / kKvHeads;  // record index, group g + 1
        const std::size_t token0 = static_cast<std::size_t>(g + 1) * kGroup;
        const auto& rot = key ? krot : vrot;
        const std::size_t rows = key ? kDim : kGroup,
                          cols = key ? kGroup : kDim;
        std::vector<float> tile(kDim * kGroup);
        for (unsigned t = 0; t < kGroup; ++t)
          for (unsigned d = 0; d < kDim; ++d) {
            const float x = kvarn::HalfToFloat(
                rot[(token0 + t) * kRowWidth + kv * kDim + d]);
            tile[key ? d * kGroup + t : t * kDim + d] = x;
          }
        const RefTile ref =
            ReferenceQuantize(tile, rows, cols, key ? 4 : VBits, false);
        const RefTile emu =
            ReferenceQuantize(tile, rows, cols, key ? 4 : VBits, true);
        const RefTile got = Unpack<VBits>(
            records.data() +
                (static_cast<std::size_t>(g) * kKvHeads + kv) * R::kBytes,
            key);
        std::size_t codes = 0, scales = 0, ulp = 0;
        for (std::size_t i = 0; i < ref.codes.size(); ++i)
          codes += ref.codes[i] != got.codes[i];
        const auto cmp = [&](const std::vector<std::uint16_t>& a,
                             const std::vector<std::uint16_t>& b) {
          for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i] != b[i]) {
              ++scales;
              ulp = std::max<std::size_t>(
                  ulp,
                  static_cast<std::size_t>(std::abs(int(a[i]) - int(b[i]))));
            }
        };
        cmp(ref.scale, got.scale);
        cmp(ref.zero, got.zero);
        cmp(ref.column, got.column);
        if (codes == 0 && scales == 0)
          ++tiles_exact;
        // GPU against the emulation of its own order: exact expected.
        std::size_t emu_codes = 0;
        for (std::size_t i = 0; i < emu.codes.size(); ++i)
          emu_codes += emu.codes[i] != got.codes[i];
        if (emu_codes != 0 || emu.scale != got.scale || emu.zero != got.zero ||
            emu.column != got.column) {
          ++emulation_mismatch;
          static std::atomic<bool> dumped{false};
          if (!dumped.exchange(true)) {
            const auto count = [](const std::vector<std::uint16_t>& a,
                                  const std::vector<std::uint16_t>& b) {
              std::size_t n = 0;
              for (std::size_t i = 0; i < a.size(); ++i)
                n += a[i] != b[i];
              return n;
            };
            std::printf(
                "emulation-diff %s group=%u kv=%u codes=%zu scale=%zu/%zu "
                "zero=%zu/%zu column=%zu/%zu\n",
                key ? "K" : "V", g + 1, kv, emu_codes,
                count(emu.scale, got.scale), emu.scale.size(),
                count(emu.zero, got.zero), emu.zero.size(),
                count(emu.column, got.column), emu.column.size());
            const auto show = [](const char* name,
                                 const std::vector<std::uint16_t>& a,
                                 const std::vector<std::uint16_t>& b) {
              unsigned shown = 0;
              for (std::size_t i = 0; i < a.size() && shown < 4; ++i)
                if (a[i] != b[i]) {
                  ++shown;
                  std::printf("  %s[%zu] emu=%04x (%.8g) gpu=%04x (%.8g)\n",
                              name, i, a[i], kvarn::HalfToFloat(a[i]), b[i],
                              kvarn::HalfToFloat(b[i]));
                }
            };
            show("scale", emu.scale, got.scale);
            show("zero", emu.zero, got.zero);
            show("column", emu.column, got.column);
          }
        }
        code_diff += codes;
        scale_diff += scales;
        for (std::size_t seen = max_ulp; ulp > seen;)
          if (max_ulp.compare_exchange_weak(seen, ulp))
            break;
        // The reference's codes reproduce the K0 fake quant bit for bit.
        const auto deq_ref = Dequant(ref, rows, cols);
        const auto& fake = key ? kf : vf;
        std::vector<float> y(kDim);
        for (unsigned t = 0; t < kGroup; ++t) {
          for (unsigned d = 0; d < kDim; ++d)
            y[d] =
                kvarn::RoundHalf(deq_ref[key ? d * kGroup + t : t * kDim + d]);
          kvarn::Hadamard(y);
          for (unsigned d = 0; d < kDim; ++d)
            if (kvarn::FloatToHalf(y[d]) !=
                fake[(token0 + t) * kRowWidth + kv * kDim + d])
              ++fake_mismatch;
        }
        const auto deq_got = Dequant(got, rows, cols);
        auto& dst = key ? kq : vq;
        for (unsigned t = 0; t < kGroup; ++t)
          for (unsigned d = 0; d < kDim; ++d)
            dst[(token0 + t) * kRowWidth + kv * kDim + d] =
                deq_got[key ? d * kGroup + t : t * kDim + d];
      });

  // Attention for 1 and 8 query rows.
  for (unsigned rows : {1U, 8U}) {
    std::vector<float> q(static_cast<std::size_t>(rows) * kHeads * kDim);
    for (auto& x : q)
      x = normal(rng);
    const std::size_t sink_elems = static_cast<std::size_t>(kGroup) * kRowWidth;
    const std::size_t tail0 = static_cast<std::size_t>(kGroup) * (1 + groups);
    std::vector<std::uint16_t> sink(krot.begin(), krot.begin() + sink_elems);
    Device<float> dq(q.size()), dout(q.size()), dout16(q.size());
    dq.Upload(q);
    Device<std::uint16_t> dks(sink_elems), dvs(sink_elems),
        dkt(std::max<std::size_t>(1, tail) * kRowWidth),
        dvt(std::max<std::size_t>(1, tail) * kRowWidth);
    dks.Upload(sink);
    dvs.Upload(
        std::vector<std::uint16_t>(vrot.begin(), vrot.begin() + sink_elems));
    if (tail > 0) {
      dkt.Upload(std::vector<std::uint16_t>(krot.begin() + tail0 * kRowWidth,
                                            krot.end()));
      dvt.Upload(std::vector<std::uint16_t>(vrot.begin() + tail0 * kRowWidth,
                                            vrot.end()));
    }
    Device<float> scratch(KvarnScratchSize(rows, splits));
    DecodeArgs args{dq.data,
                    drec.data,
                    reinterpret_cast<const __half*>(dks.data),
                    reinterpret_cast<const __half*>(dvs.data),
                    reinterpret_cast<const __half*>(dkt.data),
                    reinterpret_cast<const __half*>(dvt.data),
                    scratch.data,
                    groups,
                    tail,
                    splits};
    Device<float> s16(Fp16ScratchSize(rows));
    LaunchFp16(dq.data, reinterpret_cast<const __half*>(dk.data),
               reinterpret_cast<const __half*>(dv.data), dout16.data, s16.data,
               s16.size, length, rows);
    KV_OK(hipDeviceSynchronize());
    const auto got16 = dout16.Download();

    // Float rows for the references.
    std::vector<float> kfl(elems), vfl(elems), kff(elems), vff(elems),
        krf(elems), vrf(elems);
    for (std::size_t i = 0; i < elems; ++i) {
      kfl[i] = kvarn::HalfToFloat(k[i]);
      vfl[i] = kvarn::HalfToFloat(v[i]);
      kff[i] = kvarn::HalfToFloat(kf[i]);
      vff[i] = kvarn::HalfToFloat(vf[i]);
      const std::size_t token = i / kRowWidth;
      const bool fp16 = token < kGroup || token >= tail0;
      krf[i] = fp16 ? kvarn::HalfToFloat(krot[i]) : kq[i];
      vrf[i] = fp16 ? kvarn::HalfToFloat(vrot[i]) : vq[i];
    }
    const auto row = [](const std::vector<float>& x) {
      return [&x](unsigned t, unsigned kv) {
        return x.data() + static_cast<std::size_t>(t) * kRowWidth + kv * kDim;
      };
    };
    const auto ref_fp16 = Attention(q, rows, length, row(kfl), row(vfl), false);
    const auto ref_fake = Attention(q, rows, length, row(kff), row(vff), false);
    const auto ref_codes = Attention(q, rows, length, row(krf), row(vrf), true);
    std::vector<double> fake_as(ref_fake);
    const Diff quant_effect = Compare(fake_as, ref_fp16);
    const Diff base = Compare(got16, ref_fp16);
    const std::size_t tiles = static_cast<std::size_t>(groups) * kKvHeads * 2;
    for (const Kernel kernel : kKernels) {
      // Stale scratch must not leak into the result.
      KV_OK(hipMemset(scratch.data, 0xff, scratch.size * sizeof(float)));
      KV_OK(hipMemset(dout.data, 0xff, dout.size * sizeof(float)));
      LaunchKvarnAny(VBits, kernel, args, rows, dout.data);
      KV_OK(hipDeviceSynchronize());
      const auto got = dout.Download();
      const Diff vs_fake = Compare(got, ref_fake);
      const Diff vs_codes = Compare(got, ref_codes);
      // The kernel's own arithmetic (vs its codes) within 1e-3, well inside
      // the quantization effect; the K0 reference adds the scale-ulp cases.
      // Quantizer: bit-exact against the emulation of its order; against
      // the K0 reference at most 2 FP16 ulp of a scale.
      const bool ok = vs_codes.rel <= 1e-3 && vs_fake.rel < 2e-2 &&
                      base.rel < 1e-4 && fake_mismatch == 0 &&
                      emulation_mismatch == 0 && max_ulp <= 2 &&
                      code_diff * 10000 <= tiles * kGroup * kDim;
      totals.ok = totals.ok && ok;
      std::printf(
          "check kernel=%s k4v%d groups=%u tail=%u length=%u rows=%u "
          "splits=%u tiles=%zu gpu_vs_emulation_tile_mismatch=%zu "
          "tiles_exact_vs_k0=%zu code_diff=%zu scale_diff=%zu "
          "scale_max_ulp=%zu ref_vs_k0_fake_mismatch=%zu "
          "kvarn_vs_k0ref_rel=%.3e kvarn_vs_k0ref_maxabs=%.3e "
          "kvarn_vs_gpucodes_rel=%.3e kvarn_vs_gpucodes_maxabs=%.3e "
          "quant_effect_rel=%.3e fp16kernel_vs_fp16ref_rel=%.3e %s\n",
          KernelName(kernel), VBits, groups, tail, length, rows, splits, tiles,
          emulation_mismatch.load(), tiles_exact.load(), code_diff.load(),
          scale_diff.load(), max_ulp.load(), fake_mismatch.load(), vs_fake.rel,
          vs_fake.max_abs, vs_codes.rel, vs_codes.max_abs, quant_effect.rel,
          base.rel, ok ? "PASS" : "FAIL");
      std::fflush(stdout);
    }
  }
}

// The device math the quantizer uses against the host's: float log/exp
// rounded from double (the GPU rule), against glibc's double functions and
// glibc's logf/expf (the K0 reference rule), and double sqrt.
__global__ void LibmProbeKernel(const float* lx, const float* ex,
                                const double* sx, float* lo, float* eo,
                                double* so, unsigned n) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    lo[i] = static_cast<float>(log(static_cast<double>(lx[i])));
    eo[i] = static_cast<float>(exp(static_cast<double>(ex[i])));
    so[i] = sqrt(sx[i]);
  }
}

void ProbeLibm() {
  constexpr unsigned n = 1U << 20;
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> lu(std::log(1e-3F), std::log(1e3F)),
      eu(-0.3F, 10.0F);
  std::uniform_real_distribution<double> su(1e-6, 1e6);
  std::vector<float> lx(n), ex(n);
  std::vector<double> sx(n);
  for (unsigned i = 0; i < n; ++i) {
    lx[i] = std::exp(lu(rng));
    ex[i] = eu(rng);
    sx[i] = su(rng);
  }
  Device<float> dlx(n), dex(n), dlo(n), deo(n);
  Device<double> dsx(n), dso(n);
  dlx.Upload(lx);
  dex.Upload(ex);
  dsx.Upload(sx);
  LibmProbeKernel<<<n / 256, 256>>>(dlx.data, dex.data, dsx.data, dlo.data,
                                    deo.data, dso.data, n);
  KV_OK(hipDeviceSynchronize());
  const auto lo = dlo.Download(), eo = deo.Download();
  const auto so = dso.Download();
  std::size_t log_d = 0, log_f = 0, exp_d = 0, exp_f = 0, sqrt_d = 0;
  for (unsigned i = 0; i < n; ++i) {
    log_d += lo[i] != static_cast<float>(std::log(static_cast<double>(lx[i])));
    log_f += lo[i] != std::log(lx[i]);
    exp_d += eo[i] != static_cast<float>(std::exp(static_cast<double>(ex[i])));
    exp_f += eo[i] != std::exp(ex[i]);
    sqrt_d += so[i] != std::sqrt(sx[i]);
  }
  std::printf(
      "libm samples=%u gpu_log_vs_host_double_rounded=%zu "
      "gpu_log_vs_glibc_logf=%zu gpu_exp_vs_host_double_rounded=%zu "
      "gpu_exp_vs_glibc_expf=%zu gpu_sqrt_vs_host=%zu\n",
      n, log_d, log_f, exp_d, exp_f, sqrt_d);
}

// ---------------------------------------------------------------------------
// time: KVarN vs FP16 at the given depths, alternating formats.
// ---------------------------------------------------------------------------
__global__ void FillHalfKernel(__half* p, std::size_t n, unsigned seed) {
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
       i < n; i += std::size_t{gridDim.x} * blockDim.x) {
    unsigned h = static_cast<unsigned>(i) * 2654435761U ^ seed;
    h ^= h >> 15;
    h *= 2246822519U;
    h ^= h >> 13;
    p[i] = __float2half((static_cast<float>(h & 0xffff) / 32768.0F - 1.0F));
  }
}

template<int VBits>
__global__ void FillRecordsKernel(std::uint8_t* p, std::size_t records,
                                  unsigned seed) {
  using R = Record<VBits>;
  const std::size_t n = records * R::kBytes / 2;
  auto* h = reinterpret_cast<std::uint16_t*>(p);
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
       i < n; i += std::size_t{gridDim.x} * blockDim.x) {
    unsigned x = static_cast<unsigned>(i) * 2654435761U ^ seed;
    x ^= x >> 15;
    x *= 2246822519U;
    x ^= x >> 13;
    const std::size_t offset = (2 * i) % R::kBytes;
    float value;
    if (offset < R::kKScale) {
      h[i] = static_cast<std::uint16_t>(x);
      continue;
    } else if (offset < R::kKColumn) {
      value = (offset < R::kKZero ? 0.02F : -0.15F) *
              (1.0F + static_cast<float>(x & 255) / 512.0F);
    } else if (offset < R::kVScale) {
      value = 0.8F + static_cast<float>(x & 255) / 640.0F;
    } else if (offset < R::kVColumn) {
      value = (offset < R::kVZero ? 0.3F : -0.45F) *
              (1.0F + static_cast<float>(x & 255) / 512.0F);
    } else {
      value = 0.8F + static_cast<float>(x & 255) / 640.0F;
    }
    h[i] = __half_as_ushort(__float2half(value));
  }
}

struct Timer {
  hipEvent_t begin{}, end{};
  Timer() {
    KV_OK(hipEventCreate(&begin));
    KV_OK(hipEventCreate(&end));
  }
  ~Timer() {
    (void)hipEventDestroy(begin);
    (void)hipEventDestroy(end);
  }
  // Microseconds per call of launch(i), i cycling through `copies` buffers.
  template<typename F>
  double Run(F&& launch, unsigned copies, unsigned iterations) {
    for (unsigned i = 0; i < std::min(copies, 4U); ++i)
      launch(i);
    KV_OK(hipEventRecord(begin));
    for (unsigned i = 0; i < iterations; ++i)
      launch(i % copies);
    KV_OK(hipEventRecord(end));
    KV_OK(hipEventSynchronize(end));
    float ms;
    KV_OK(hipEventElapsedTime(&ms, begin, end));
    return 1000.0 * ms / iterations;
  }
};

unsigned CopiesFor(std::size_t bytes) {
  // Cycle at least 512 MiB so the 32 MiB MALL does not serve repeat reads.
  constexpr std::size_t kTarget = std::size_t{512} << 20;
  return static_cast<unsigned>(
      std::clamp<std::size_t>((kTarget + bytes - 1) / bytes, 1, 64));
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// probe: split 18 only, the launch split into its parts (partials alone,
// reduce alone, quantized blocks alone, sink/tail blocks alone).
void TimeDepth(unsigned depth, unsigned rows, Timer& timer, bool probe) {
  constexpr unsigned kTail = 128;
  constexpr unsigned kReps = 7;
  const unsigned groups = depth / kGroup - 2;  // sink, groups, 128-token tail
  const std::size_t fp16_layer = static_cast<std::size_t>(depth) * kRowWidth;
  const std::size_t fp16_bytes = 2 * fp16_layer * sizeof(__half);
  const unsigned fp16_copies = CopiesFor(fp16_bytes);
  Device<__half> k16(fp16_layer * fp16_copies), v16(fp16_layer * fp16_copies);
  FillHalfKernel<<<1024, 256>>>(k16.data, k16.size, 11 + depth);
  FillHalfKernel<<<1024, 256>>>(v16.data, v16.size, 13 + depth);

  const std::size_t sink_elems = static_cast<std::size_t>(kGroup) * kRowWidth;
  const std::size_t tail_elems = static_cast<std::size_t>(kTail) * kRowWidth;
  const std::size_t fp16_side = 2 * (sink_elems + tail_elems) * sizeof(__half);
  struct Format {
    int bits;
    std::size_t record;
  };
  const Format formats[2] = {{2, Record<2>::kBytes}, {4, Record<4>::kBytes}};
  unsigned qcopies[2];
  std::size_t qbytes[2];
  std::vector<std::unique_ptr<Device<std::uint8_t>>> recs;
  std::vector<std::unique_ptr<Device<__half>>> sides;
  for (unsigned f = 0; f < 2; ++f) {
    const std::size_t layer =
        static_cast<std::size_t>(groups) * kKvHeads * formats[f].record;
    qbytes[f] = layer + fp16_side;
    qcopies[f] = CopiesFor(qbytes[f]);
    recs.push_back(std::make_unique<Device<std::uint8_t>>(layer * qcopies[f]));
    if (formats[f].bits == 2)
      FillRecordsKernel<2><<<1024, 256>>>(recs.back()->data,
                                          groups * kKvHeads * qcopies[f], 17);
    else
      FillRecordsKernel<4><<<1024, 256>>>(recs.back()->data,
                                          groups * kKvHeads * qcopies[f], 19);
    sides.push_back(std::make_unique<Device<__half>>(
        2 * (sink_elems + tail_elems) * qcopies[f]));
    FillHalfKernel<<<1024, 256>>>(sides.back()->data, sides.back()->size, 23);
  }
  std::vector<float> q(static_cast<std::size_t>(rows) * kHeads * kDim);
  for (std::size_t i = 0; i < q.size(); ++i)
    q[i] = std::sin(0.37 * static_cast<double>(i));
  Device<float> dq(q.size()), dout(q.size());
  dq.Upload(q);
  Device<float> s16(Fp16ScratchSize(rows));
  // 4 KV heads x (splits + 2) blocks: 18 and 38 fill one and two rounds of
  // 80 resident AR k4v2 blocks (20 WGPs x 4).
  constexpr unsigned kSplitOptions = 5;
  const unsigned split_options[kSplitOptions] = {18, 32, 38, 64, 128};
  Device<float> sq(KvarnScratchSize(rows, 128));
  KV_OK(hipDeviceSynchronize());

  const auto fp16_launch = [&](unsigned c) {
    LaunchFp16(dq.data, k16.data + c * fp16_layer, v16.data + c * fp16_layer,
               dout.data, s16.data, s16.size, depth, rows);
  };
  const auto kvarn_launch = [&](Kernel kernel, unsigned f, unsigned splits,
                                unsigned parts, unsigned only, unsigned c) {
    const std::size_t layer =
        static_cast<std::size_t>(groups) * kKvHeads * formats[f].record;
    // [sink K][tail K][sink V][tail V] per copy.
    const __half* side = sides[f]->data + c * 2 * (sink_elems + tail_elems);
    DecodeArgs args{dq.data,
                    recs[f]->data + c * layer,
                    side,
                    side + sink_elems + tail_elems,
                    side + sink_elems,
                    side + 2 * sink_elems + tail_elems,
                    sq.data,
                    groups,
                    kTail,
                    splits,
                    only};
    LaunchKvarnAny(formats[f].bits, kernel, args, rows, dout.data, parts);
  };
  // fp16, then every kernel x format x split option, in rotating order.
  constexpr unsigned kKernelCount = std::size(kKernels);
  struct Config {
    Kernel kernel;
    unsigned f, splits, parts{3}, only{0};
    const char* part{"full"};
  };
  std::vector<Config> configs;
  for (const Kernel kernel : kKernels)
    for (unsigned f = 0; f < 2; ++f) {
      if (!probe) {
        for (unsigned s : split_options)
          configs.push_back({kernel, f, s});
        continue;
      }
      configs.push_back({kernel, f, 18, 3, 0, "full"});
      configs.push_back({kernel, f, 18, 1, 0, "partials"});
      configs.push_back({kernel, f, 18, 2, 0, "reduce"});
      configs.push_back({kernel, f, 18, 1, 1, "quantized_blocks"});
      configs.push_back({kernel, f, 18, 1, 2, "fp16_blocks"});
    }
  static_assert(kKernelCount == 3);
  std::vector<std::vector<double>> times(1 + configs.size());
  for (unsigned rep = 0; rep < kReps; ++rep)
    for (unsigned j = 0; j < times.size(); ++j) {
      const unsigned which = (j + rep) % times.size();
      if (which == 0) {
        times[0].push_back(
            timer.Run(fp16_launch, fp16_copies, std::max(fp16_copies, 20U)));
      } else {
        const Config& c = configs[which - 1];
        times[which].push_back(timer.Run(
            [&](unsigned copy) {
              kvarn_launch(c.kernel, c.f, c.splits, c.parts, c.only, copy);
            },
            qcopies[c.f], std::max(qcopies[c.f], 20U)));
      }
    }
  const double fp16_us = Median(times[0]);
  std::printf(
      "%s depth=%u rows=%u kernel=fp16 fmt=fp16 splits=32 us=%.1f "
      "us_min=%.1f MiB=%.2f GBps=%.1f copies=%u\n",
      probe ? "probe" : "time", depth, rows, fp16_us,
      *std::min_element(times[0].begin(), times[0].end()),
      fp16_bytes / 1048576.0, fp16_bytes / fp16_us / 1e3, fp16_copies);
  for (unsigned which = 1; which < times.size(); ++which) {
    const Config& c = configs[which - 1];
    const double us = Median(times[which]);
    std::printf(
        "%s depth=%u rows=%u kernel=%s fmt=k4v%d splits=%u part=%s us=%.1f "
        "us_min=%.1f MiB=%.2f GBps=%.1f copies=%u speedup=%.2f\n",
        probe ? "probe" : "time", depth, rows, KernelName(c.kernel),
        formats[c.f].bits, c.splits, c.part, us,
        *std::min_element(times[which].begin(), times[which].end()),
        qbytes[c.f] / 1048576.0, qbytes[c.f] / us / 1e3, qcopies[c.f],
        fp16_us / us);
  }
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// quant: group-flush time for one layer, all 16 layers, and a 64-group batch.
// ---------------------------------------------------------------------------
template<int VBits>
void TimeQuantize(Timer& timer) {
  using R = Record<VBits>;
  constexpr unsigned kBatch = 64;
  const std::size_t layer_rows =
      static_cast<std::size_t>(kGroup) * (1 + kBatch) * kRowWidth;
  const std::size_t layer_records =
      static_cast<std::size_t>(kBatch) * kKvHeads * R::kBytes;
  Device<__half> k(layer_rows * kLayers), v(layer_rows * kLayers);
  FillHalfKernel<<<1024, 256>>>(k.data, k.size, 29);
  FillHalfKernel<<<1024, 256>>>(v.data, v.size, 31);
  Device<std::uint8_t> rec(layer_records * kLayers);
  KV_OK(hipDeviceSynchronize());
  struct Case {
    const char* name;
    unsigned groups, layers;
  };
  for (const Case c : {Case{"one_layer", 1, 1}, Case{"all_16_layers", 1, 16},
                       Case{"prefill_64_groups_x16", kBatch, 16}}) {
    std::vector<double> t;
    for (unsigned rep = 0; rep < 5; ++rep)
      t.push_back(timer.Run(
          [&](unsigned) {
            LaunchQuantize<VBits>(k.data, v.data, rec.data, layer_rows,
                                  layer_records, 1, c.groups, c.layers);
          },
          1, 20));
    const double us = Median(t);
    const double tiles = 2.0 * kKvHeads * c.groups * c.layers;
    std::printf(
        "quant k4v%d case=%s tiles=%.0f us=%.1f us_per_tile=%.2f "
        "us_per_group_all_layers=%.1f\n",
        VBits, c.name, tiles, us, us / tiles,
        us / tiles * 2 * kKvHeads * kLayers);
  }
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  bool check = argc == 1, time = argc == 1, quant = argc == 1, probe = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "check")
      check = true;
    else if (arg == "time")
      time = true;
    else if (arg == "quant")
      quant = true;
    else if (arg == "probe")
      probe = true;
    else {
      std::fprintf(stderr, "usage: %s [check] [time] [quant] [probe]\n",
                   argv[0]);
      return 2;
    }
  }
  hipDeviceProp_t prop{};
  KV_OK(hipGetDeviceProperties(&prop, 0));
  std::printf("device %s arch %s CUs %d\n", prop.name, prop.gcnArchName,
              prop.multiProcessorCount);
  CheckTotals totals;
  if (check) {
    struct Case {
      unsigned groups, tail, splits;
    };
    for (const Case c : {Case{0, 50, 32}, Case{1, 44, 32}, Case{30, 100, 32},
                         Case{254, 133, 32}, Case{254, 133, 7}}) {
      CheckCase<2>(c.groups, c.tail, c.splits, totals);
      CheckCase<4>(c.groups, c.tail, c.splits, totals);
    }
    ProbeLibm();
    std::printf("check %s\n", totals.ok ? "PASS" : "FAIL");
  }
  Timer timer;
  if (quant) {
    TimeQuantize<2>(timer);
    TimeQuantize<4>(timer);
  }
  if (time)
    for (unsigned depth : {4096U, 16384U, 32768U, 65536U, 131072U, 262144U})
      for (unsigned rows : {1U, 8U})
        TimeDepth(depth, rows, timer, false);
  if (probe)
    for (unsigned depth : {4096U, 32768U})
      for (unsigned rows : {1U, 8U})
        TimeDepth(depth, rows, timer, true);
  return totals.ok ? 0 : 1;
}
