#ifndef GUFO_CORE_HIP_DETAIL_KVARN_DEVICE_HPP_
#define GUFO_CORE_HIP_DETAIL_KVARN_DEVICE_HPP_

// KVarN device arithmetic shared by the models that use it
// (src/core/hip/kvarn_layout.hpp): the in-wave 256-point Walsh-Hadamard
// transform and the group-flush quantizer tile, bit-identical to the CPU
// reference tests/models/qwen27b/kvarn_fake_quant.hpp. Include only from HIP
// translation units; each model wraps the tile in its own kernel with its
// row and record addressing. Moved from
// src/models/qwen/hip/kernels/attention_kvarn.hip unchanged in arithmetic.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/hip/kvarn_layout.hpp"

namespace gufo::hip::kvarn_detail {

typedef _Float16 h2 __attribute__((ext_vector_type(2)));
constexpr unsigned kDim = kKvarnHeadDim;
constexpr unsigned kGroup = kKvarnGroupTokens;

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
// Float-float sums for the quantizer's standard deviations, in place of the
// reference's double: each sum is a float h and the float sum l of every
// rounding error of h (Sum2, Ogita-Rump-Oishi), as accurate as a sum in twice
// float precision. Contraction is off so the error-free steps stay exact.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void TwoSum(float a, float b, float& s, float& e) {
  s = a + b;
  const float b_part = s - a;
  e = (a - (s - b_part)) + (b - b_part);
}

__device__ __forceinline__ void SumAdd(float& h, float& l, float x) {
  float e;
  TwoSum(h, x, h, e);
  l += e;
}

// Adds (x - mean)^2 with mean = mh + ml: x - mean as d + e (x - mh exactly,
// less ml), then (d + e)^2 as the exact d^2 plus (2d + e)e. e is not small
// beside d when x is within an ulp of the mean.
__device__ __forceinline__ void SumSquareAdd(float& h, float& l, float x,
                                             float mh, float ml) {
#pragma clang fp contract(off)
  float d, e;
  TwoSum(x, -mh, d, e);
  e -= ml;
  const float p = d * d;
  float pe = fmaf(d, d, -p);
  pe = fmaf(d + d + e, e, pe);
  float s, err;
  TwoSum(h, p, s, err);
  h = s;
  l += err + pe;
}

// Merges another sum; the same in both lanes of a butterfly (s and the
// exact error e do not depend on the order).
__device__ __forceinline__ void SumMerge(float& h, float& l, float h2,
                                         float l2) {
  float s, e;
  TwoSum(h, h2, s, e);
  h = s;
  l = (l + l2) + e;
}

// Mean of `count` (a power of two) values from their sum: exact scaling.
__device__ __forceinline__ void SumMean(float h, float l, float inverse_count,
                                        float& mh, float& ml) {
  TwoSum(h, l, mh, ml);
  mh *= inverse_count;
  ml *= inverse_count;
}

// sqrt((h + l) / divisor) rounded once to float: the float-float quotient
// (exact remainder), then one Newton correction of the rounded square root,
// whose remainder q - y*y is exact.
__device__ __forceinline__ float SumStd(float h, float l, float divisor) {
#pragma clang fp contract(off)
  float s, t;
  TwoSum(h, l, s, t);
  if (s == 0.0F)
    return 0.0F;
  const float q = s / divisor;
  const float ql = (fmaf(-q, divisor, s) + t) / divisor;
  const float y = sqrtf(q);
  return y + (fmaf(-y, y, q) + ql) / (y + y);
}

// v from lane (lane ^ mask) of a wave32, mask 1 to 16, without the LDS
// crossbar of __shfl_xor: DPP row_xmask inside each row of 16 lanes, the
// permlanex16 row swap for 16.
__device__ __forceinline__ float XorLane(float v, unsigned mask) {
  const int bits = __float_as_int(v);
  switch (mask) {
    case 1:
      return __int_as_float(
          __builtin_amdgcn_update_dpp(0, bits, 0x161, 0xf, 0xf, false));
    case 2:
      return __int_as_float(
          __builtin_amdgcn_update_dpp(0, bits, 0x162, 0xf, 0xf, false));
    case 4:
      return __int_as_float(
          __builtin_amdgcn_update_dpp(0, bits, 0x164, 0xf, 0xf, false));
    case 8:
      return __int_as_float(
          __builtin_amdgcn_update_dpp(0, bits, 0x168, 0xf, 0xf, false));
    default:
      return __int_as_float(__builtin_amdgcn_permlanex16(
          bits, bits, 0x76543210, 0xfedcba98, false, false));
  }
}

// a / b correctly rounded from rb = 1 / b correctly rounded: one FMA
// correction of a * rb (Markstein).
__device__ __forceinline__ float DivideBy(float a, float b, float rb) {
  const float q = a * rb;
  return fmaf(fmaf(-q, b, a), rb, q);
}

// Correctly rounded float log and exp of the balancing update without FP64
// in the common case: the value in float-float (relative error below
// 2^-40), rounded to float when the rounding cannot change within 2^-34 of
// it; otherwise (about one value in a thousand) through double as the
// reference does.
struct Float2 {
  float h, l;
};

__device__ __forceinline__ Float2 Normalize(float h, float l) {
  const float s = h + l;
  return {s, l - (s - h)};
}

__device__ __forceinline__ Float2 Add(Float2 a, Float2 b) {
  float s, e;
  TwoSum(a.h, b.h, s, e);
  return Normalize(s, e + (a.l + b.l));
}

__device__ __forceinline__ Float2 Multiply(Float2 a, Float2 b) {
#pragma clang fp contract(off)
  const float p = a.h * b.h;
  float e = fmaf(a.h, b.h, -p);
  e = fmaf(a.h, b.l, e);
  e = fmaf(a.l, b.h, e);
  return Normalize(p, e);
}

// RN(v) when it is the same for every value within 2^-34 |v| of v.
__device__ __forceinline__ bool RoundDecided(Float2 v, float& out) {
  const float margin = fabsf(v.h) * 0x1p-34F;
  const float below = v.h + (v.l - margin);
  const float above = v.h + (v.l + margin);
  out = below;
  return below == above;
}

// ln2 = kLn2Hi + kLn2Lo + kLn2Lo2; kLn2Hi has 16 bits, so k * kLn2Hi is
// exact for |k| < 256.
constexpr float kLn2Hi = 0x1.62e4p-1F, kLn2Lo = 0x1.7f7d1cp-20F,
                kLn2Lo2 = 0x1.ef357ap-45F;

// log x for x in [1e-3, 1e3]: x = 2^k m, m in [sqrt(1/2), sqrt(2)),
// log m = 2 atanh(f) = 2f (1 + f^2 / 3 + f^4 / 5 + ...), f = (m - 1) / (m + 1),
// |f| <= 0.172, through f^16.
__device__ __forceinline__ float RoundedLog(float x) {
#pragma clang fp contract(off)
  const unsigned bits = __float_as_uint(x);
  int k = static_cast<int>(bits >> 23) - 127;
  float m = __uint_as_float((bits & 0x7fffffU) | 0x3f800000U);
  if (m > 1.41421354F) {
    m *= 0.5F;
    ++k;
  }
  const float num = m - 1.0F;  // exact (Sterbenz)
  float dh, dl;
  TwoSum(m, 1.0F, dh, dl);
  const float fh = num / dh;
  const float fl = fmaf(-fh, dl, fmaf(-fh, dh, num)) / dh;
  const Float2 f{fh, fl};
  const Float2 f2 = Multiply(f, f);
  // P = 1/3 + f2 (1/5 + f2 (1/7 + ...)): float below 1/9, float-float above.
  float tail = 0x1.e1e1e2p-5F;                                // 1/17
  tail = fmaf(tail, f2.h, 0x1.111112p-4F);                    // 1/15
  tail = fmaf(tail, f2.h, 0x1.3b13b2p-4F);                    // 1/13
  tail = fmaf(tail, f2.h, 0x1.745d18p-4F);                    // 1/11
  Float2 poly = Normalize(0x1.c71c72p-4F, -0x1.c71c72p-31F);  // 1/9
  poly = Add(poly, Multiply(f2, Float2{tail, 0.0F}));
  poly = Add(Float2{0x1.24924ap-3F, -0x1.b6db6ep-28F}, Multiply(f2, poly));
  poly = Add(Float2{0x1.99999ap-3F, -0x1.99999ap-29F}, Multiply(f2, poly));
  poly = Add(Float2{0x1.555556p-2F, -0x1.555556p-27F}, Multiply(f2, poly));
  const Float2 two_f{2.0F * f.h, 2.0F * f.l};
  const Float2 log_m = Add(two_f, Multiply(two_f, Multiply(f2, poly)));
  const float kf = static_cast<float>(k);
  const Float2 log_x =
      Add(Normalize(kf * kLn2Hi, fmaf(kf, kLn2Lo, kf * kLn2Lo2)), log_m);
  float out;
  if (RoundDecided(log_x, out))
    return out;
  return static_cast<float>(log(static_cast<double>(x)));
}

// exp x for x in [-0.3, 10]: x = k ln2 + r, |r| <= 0.35, exp r through r^13.
__device__ __forceinline__ float RoundedExp(float x) {
#pragma clang fp contract(off)
  const float kf = rintf(x * 0x1.715476p+0F);
  const float rh = fmaf(-kf, kLn2Hi, x);  // exact
  const Float2 r = Normalize(rh, -fmaf(kf, kLn2Lo, kf * kLn2Lo2));
  float tail = 0x1.612462p-33F;             // 1/13!
  tail = fmaf(tail, r.h, 0x1.1eed8ep-29F);  // 1/12!
  tail = fmaf(tail, r.h, 0x1.ae6456p-26F);
  tail = fmaf(tail, r.h, 0x1.27e4fcp-22F);
  tail = fmaf(tail, r.h, 0x1.71de3ap-19F);
  tail = fmaf(tail, r.h, 0x1.a01a02p-16F);                     // 1/8!
  Float2 poly = Normalize(0x1.a01a02p-13F, -0x1.7f97fap-39F);  // 1/7!
  poly = Add(poly, Multiply(r, Float2{tail, 0.0F}));
  poly = Add(Float2{0x1.6c16c2p-10F, -0x1.27d27ep-35F}, Multiply(r, poly));
  poly = Add(Float2{0x1.111112p-7F, -0x1.dddddep-32F}, Multiply(r, poly));
  poly = Add(Float2{0x1.555556p-5F, -0x1.555556p-30F}, Multiply(r, poly));
  poly = Add(Float2{0x1.555556p-3F, -0x1.555556p-28F}, Multiply(r, poly));
  poly = Add(Float2{0.5F, 0.0F}, Multiply(r, poly));
  poly = Add(Float2{1.0F, 0.0F}, Multiply(r, poly));
  poly = Add(Float2{1.0F, 0.0F}, Multiply(r, poly));
  const int k = static_cast<int>(kf);
  const Float2 value{ldexpf(poly.h, k), ldexpf(poly.l, k)};
  float out;
  if (RoundDecided(value, out))
    return out;
  return static_cast<float>(exp(static_cast<double>(x)));
}

// ---------------------------------------------------------------------------
// Group-flush quantizer tile, run by one 512-thread block: one (KV head,
// group, K or V) tile of rotated FP16 rows, token t at src + t * row_stride.
// Wave w holds tokens
// 8w..8w+7 and lane l channels 8l..8l+7, so per-token statistics reduce inside
// a wave and per-channel ones across the 16 waves through LDS. Standard
// deviations sum in float-float where the reference sums in double, and
// round to the same float; log and exp round once, correctly, as the K0
// reference (tests/models/qwen27b/kvarn_fake_quant.hpp) does through double.
// ---------------------------------------------------------------------------
template<int VBits, KvarnValueOrder Order, int KBits = 4>
__device__ __forceinline__ void KvarnQuantizeTile(
    const __half* __restrict__ src, std::size_t row_stride,
    std::uint8_t* __restrict__ record, bool key) {
#pragma clang fp contract(off)
  static_assert(VBits == 2 || VBits == 4 || VBits == 6);
  static_assert(KBits == 4 || KBits == 6);
  static_assert(
      (Order == KvarnValueOrder::kChannelMajor && KBits == 4 && VBits != 6) ||
          (Order == KvarnValueOrder::kTokenMajor && VBits == KBits),
      "6-bit records require token-major K6V6");
  constexpr KvarnRecordLayout R = KvarnRecord(VBits, Order, KBits);
  constexpr unsigned kWaves = 16;
  __shared__ float2 channel_part[kWaves][kDim];
  __shared__ float2 channel_result[kDim];
  __shared__ float block_part[kWaves][4];
  __shared__ unsigned value_codes[kDim * kGroup * VBits / 32];
  __shared__ float shared_scale[kDim];

  const unsigned tid = threadIdx.x, wave = tid / 32, lane = tid % 32;

  // Token 8w + i, channels 8l..8l+7 of the rotated FP16 source. Read again
  // for every rescaling (from L2) rather than held, which leaves registers
  // for the float-float sums; the opaque base keeps the loads in place.
  const auto load_source = [&](unsigned i) {
    const __half* rows = src;
    asm volatile("" : "+s"(rows));
    return *reinterpret_cast<const uint4*>(
        rows + static_cast<std::size_t>(8 * wave + i) * row_stride + 8 * lane);
  };
  const auto source_of = [](const uint4& raw, unsigned j) {
    const unsigned word = j < 2 ? raw.x : j < 4 ? raw.y : j < 6 ? raw.z : raw.w;
    return static_cast<float>(__builtin_bit_cast(h2, word)[j % 2]);
  };
  if constexpr (Order == KvarnValueOrder::kChannelMajor)
    if (!key)
      for (unsigned i = tid; i < sizeof(value_codes) / 4; i += 512)
        value_codes[i] = 0;

  float cur[8][8];
  float token_scale[8], channel_scale[8];
  // The balancing state this lane updates (see update_tokens and
  // update_channels): token 8w + (l & 7), and channel 8l + w in waves 0-7.
  float own_token_log = 0.0F, own_token_scale = 1.0F, best_own_token = 1.0F;
  float own_channel_log = 0.0F, own_channel_scale = 1.0F;
  float best_own_channel = 1.0F;
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) {
    token_scale[i] = channel_scale[i] = 1.0F;
    const uint4 raw = load_source(i);
#pragma unroll
    for (unsigned j = 0; j < 8; ++j)
      cur[i][j] = source_of(raw, j);
  }

  const auto wave_sum = [](float& h, float& l) {
#pragma unroll
    for (unsigned mask = 1; mask < 32; mask <<= 1)
      SumMerge(h, l, XorLane(h, mask), XorLane(l, mask));
  };
  // Unbiased std of every token (over 256 channels), wave-uniform.
  const auto token_std = [&](float (&out)[8]) {
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
      float h = 0, l = 0;
#pragma unroll
      for (unsigned j = 0; j < 8; ++j)
        SumAdd(h, l, cur[i][j]);
      wave_sum(h, l);
      float mh, ml;
      SumMean(h, l, 1.0F / kDim, mh, ml);
      h = l = 0;
#pragma unroll
      for (unsigned j = 0; j < 8; ++j)
        SumSquareAdd(h, l, cur[i][j], mh, ml);
      wave_sum(h, l);
      out[i] = SumStd(h, l, kDim - 1);
    }
  };
  // Unbiased std of this lane's 8 channels (over 128 tokens).
  const auto channel_std = [&](float (&out)[8]) {
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
      float h = 0, l = 0;
#pragma unroll
      for (unsigned i = 0; i < 8; ++i)
        SumAdd(h, l, cur[i][j]);
      channel_part[wave][8 * lane + j] = make_float2(h, l);
    }
    __syncthreads();
    if (tid < kDim) {
      float h = channel_part[0][tid].x, l = channel_part[0][tid].y;
      for (unsigned w = 1; w < kWaves; ++w)
        SumMerge(h, l, channel_part[w][tid].x, channel_part[w][tid].y);
      float mh, ml;
      SumMean(h, l, 1.0F / kGroup, mh, ml);
      channel_result[tid] = make_float2(mh, ml);
    }
    __syncthreads();
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
      const float2 mean = channel_result[8 * lane + j];
      float h = 0, l = 0;
#pragma unroll
      for (unsigned i = 0; i < 8; ++i)
        SumSquareAdd(h, l, cur[i][j], mean.x, mean.y);
      channel_part[wave][8 * lane + j] = make_float2(h, l);
    }
    __syncthreads();
    if (tid < kDim) {
      float h = channel_part[0][tid].x, l = channel_part[0][tid].y;
      for (unsigned w = 1; w < kWaves; ++w)
        SumMerge(h, l, channel_part[w][tid].x, channel_part[w][tid].y);
      channel_result[tid].x = SumStd(h, l, kGroup - 1);
    }
    __syncthreads();
#pragma unroll
    for (unsigned j = 0; j < 8; ++j)
      out[j] = channel_result[8 * lane + j].x;
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
      cmax = fmaxf(cmax, XorLane(cmax, mask));
      cmin = fminf(cmin, XorLane(cmin, mask));
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
  // source / column scale / row scale, in the reference's division order,
  // each division correctly rounded.
  const auto apply = [&](const float (&ts)[8], const float (&cs)[8]) {
    float rt[8], rc[8];
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
      rt[i] = 1.0F / ts[i];
      rc[i] = 1.0F / cs[i];
    }
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
      const uint4 raw = load_source(i);
#pragma unroll
      for (unsigned j = 0; j < 8; ++j) {
        const float x = source_of(raw, j);
        cur[i][j] = key ? DivideBy(DivideBy(x, ts[i], rt[i]), cs[j], rc[j])
                        : DivideBy(DivideBy(x, cs[j], rc[j]), ts[i], rt[i]);
        asm volatile("" : "+v"(cur[i][j]));
      }
    }
  };
  const auto update = [](float& log_scale, float s) {
    const float clamped = fminf(fmaxf(s, 1e-3F), 1e3F);
    log_scale = fminf(fmaxf(log_scale + RoundedLog(clamped), -0.3F), 10.0F);
    return RoundedExp(log_scale);
  };
  // Every lane holds the same eight token scales and every wave the same
  // eight channel scales per lane, so each update runs once and is shared:
  // lane i of a wave updates its token i, wave j (< 8) channel 8l + j. The
  // values are those every lane computed before, bit for bit; the double
  // log and exp were most of the quantizer's time.
  // Keep the unrolled register selections: Clang24 otherwise reconstructs
  // an indexed array load, creating an 80-byte private frame for these stds.
  // The barriers preserve values and emit no instructions.
  const auto update_tokens = [&](const float (&stds)[8]) {
    float s = stds[0];
#pragma unroll
    for (unsigned i = 1; i < 8; ++i) {
      if ((lane & 7) == i)
        s = stds[i];
      asm volatile("" : "+v"(s));
    }
    own_token_scale = update(own_token_log, s);
#pragma unroll
    for (unsigned i = 0; i < 8; ++i)
      token_scale[i] = __int_as_float(
          __builtin_amdgcn_readlane(__float_as_int(own_token_scale), i));
  };
  const auto update_channels = [&](const float (&stds)[8]) {
    if (wave < 8) {
      float s = stds[0];
#pragma unroll
      for (unsigned j = 1; j < 8; ++j) {
        if (wave == j)
          s = stds[j];
        asm volatile("" : "+v"(s));
      }
      own_channel_scale = update(own_channel_log, s);
      shared_scale[8 * lane + wave] = own_channel_scale;
    }
    __syncthreads();
#pragma unroll
    for (unsigned j = 0; j < 8; ++j)
      channel_scale[j] = shared_scale[8 * lane + j];
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
      best_own_token = own_token_scale;
      best_own_channel = own_channel_scale;
    }
  }
  float best_token[8], best_channel[8];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i)
    best_token[i] = __int_as_float(
        __builtin_amdgcn_readlane(__float_as_int(best_own_token), i));
  if (wave < 8)
    shared_scale[8 * lane + wave] = best_own_channel;
  __syncthreads();
#pragma unroll
  for (unsigned j = 0; j < 8; ++j)
    best_channel[j] = shared_scale[8 * lane + j];
  apply(best_token, best_channel);

  // Per-row asymmetric RTN; K rows are channels, V rows are tokens.
  constexpr float kKeyMax = static_cast<float>((1 << KBits) - 1);
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
    float2* part = &channel_part[0][0];
    float2* result = channel_result;
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
    auto* codes = reinterpret_cast<unsigned*>(record + R.k_codes);
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
      unsigned word = 0, high = 0;
#pragma unroll
      for (unsigned j = 0; j < 8; ++j) {
        const float q =
            fminf(fmaxf(rintf((cur[i][j] - low[j]) / step[j]), 0.0F), kKeyMax);
        const unsigned nibble = (j % 2 == 0) ? j / 2 : 4 + j / 2;
        word |= (static_cast<unsigned>(q) & 15U) << (4 * nibble);
        if constexpr (KBits == 6)
          high |= (static_cast<unsigned>(q) >> 4) << (2 * j);
      }
      codes[(8 * wave + i) * (kDim * KBits / 32) + lane] = word;
      if constexpr (KBits == 6)
        reinterpret_cast<std::uint16_t*>(record + R.k_codes +
                                         (8 * wave + i) * (kDim * KBits / 8) +
                                         kDim / 2)[lane] = high;
    }
    if (wave == 0)
#pragma unroll
      for (unsigned j = 0; j < 8; ++j) {
        halves(R.k_scale)[8 * lane + j] =
            product_half(best_channel[j], step[j]);
        halves(R.k_zero)[8 * lane + j] = product_half(best_channel[j], low[j]);
      }
    if (lane == 0)
#pragma unroll
      for (unsigned i = 0; i < 8; ++i)
        halves(R.k_column)[8 * wave + i] = __float2half(best_token[i]);
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
        lo = fminf(lo, XorLane(lo, mask));
        hi = fmaxf(hi, XorLane(hi, mask));
      }
      const float step = fmaxf((hi - lo) / kValueMax, 1e-10F);
      const unsigned token = 8 * wave + i;
      [[maybe_unused]] unsigned token_word = 0, token_high = 0;
#pragma unroll
      for (unsigned j = 0; j < 8; ++j) {
        const unsigned q = static_cast<unsigned>(
            fminf(fmaxf(rintf((cur[i][j] - lo) / step), 0.0F), kValueMax));
        const unsigned channel = 8 * lane + j;
        if constexpr (Order == KvarnValueOrder::kTokenMajor) {
          // The K packing: channels 8l..8l+7 of this token in one word.
          const unsigned nibble = (j % 2 == 0) ? j / 2 : 4 + j / 2;
          token_word |= (q & 15U) << (4 * nibble);
          if constexpr (VBits == 6)
            token_high |= (q >> 4) << (2 * j);
          (void)channel;
        } else if constexpr (VBits == 2) {
          const unsigned u = token % 16;
          const unsigned crumb = (u % 2 == 0) ? u / 2 : 8 + u / 2;
          atomicOr(&value_codes[channel * 8 + token / 16], q << (2 * crumb));
        } else {
          const unsigned u = token % 8;
          const unsigned nibble = (u % 2 == 0) ? u / 2 : 4 + u / 2;
          atomicOr(&value_codes[channel * 16 + token / 8], q << (4 * nibble));
        }
      }
      if constexpr (Order == KvarnValueOrder::kTokenMajor) {
        reinterpret_cast<unsigned*>(
            record + R.v_codes)[token * (kDim * VBits / 32) + lane] =
            token_word;
        if constexpr (VBits == 6)
          reinterpret_cast<std::uint16_t*>(record + R.v_codes +
                                           token * (kDim * VBits / 8) +
                                           kDim / 2)[lane] = token_high;
      }
      if (lane == 0) {
        halves(R.v_scale)[token] = product_half(best_token[i], step);
        halves(R.v_zero)[token] = product_half(best_token[i], lo);
      }
    }
    if (wave == 0)
#pragma unroll
      for (unsigned j = 0; j < 8; ++j)
        halves(R.v_column)[8 * lane + j] = __float2half(best_channel[j]);
    if constexpr (Order == KvarnValueOrder::kChannelMajor) {
      __syncthreads();
      auto* codes = reinterpret_cast<uint4*>(record + R.v_codes);
      const auto* staged = reinterpret_cast<const uint4*>(value_codes);
      for (unsigned i = tid; i < sizeof(value_codes) / 16; i += 512)
        codes[i] = staged[i];
    }
  }
}

}  // namespace gufo::hip::kvarn_detail

#endif  // GUFO_CORE_HIP_DETAIL_KVARN_DEVICE_HPP_
