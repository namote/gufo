#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_DRAFT_HEAD_Q4K_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_DRAFT_HEAD_Q4K_HPP_

// ggml's quantize_row_q4_K_ref for one 256-value superblock, shared by the
// load-time draft-head requantization (kernels/rocm/draft_head.hip.cpp) and
// its CPU test. Every step rounds in IEEE single precision; the device file
// is built without fast math or contraction so both sides agree.

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__HIPCC__)
#define QFN_Q4K_HD __host__ __device__
#else
#define QFN_Q4K_HD
#endif

namespace gufo::models::qwen38_flash_next::q4k {

struct Block {
  std::uint16_t d;     ///< FP16 bits
  std::uint16_t dmin;  ///< FP16 bits
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(Block) == 144, "block_q4_K must be 144 bytes");

QFN_Q4K_HD inline std::uint32_t FloatBits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
QFN_Q4K_HD inline float BitsFloat(std::uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
/// IEEE binary16 bits of `value`, round to nearest even (subnormals kept).
QFN_Q4K_HD inline std::uint16_t HalfBits(float value) {
  const std::uint32_t f = FloatBits(value);
  const std::uint32_t sign = (f >> 16) & 0x8000U;
  const std::uint32_t abs = f & 0x7FFFFFFFU;
  if (abs >= 0x7F800000U)  // Inf or NaN
    return static_cast<std::uint16_t>(sign | 0x7C00U |
                                      (abs > 0x7F800000U ? 0x200U : 0U));
  if (abs >= 0x477FF000U)  // rounds past 65504
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  if (abs < 0x38800000U) {  // below the smallest normal half: subnormal
    if (abs < 0x33000000U)  // below half the smallest subnormal
      return static_cast<std::uint16_t>(sign);
    const std::uint32_t mantissa = (abs & 0x7FFFFFU) | 0x800000U;
    const int shift = 126 - static_cast<int>(abs >> 23);  // 14..24
    const std::uint32_t kept = mantissa >> shift;
    const std::uint32_t rest = mantissa & ((1U << shift) - 1);
    const std::uint32_t halfway = 1U << (shift - 1);
    const std::uint32_t rounded =
        kept + (rest > halfway || (rest == halfway && (kept & 1U)) ? 1U : 0U);
    return static_cast<std::uint16_t>(sign | rounded);
  }
  const std::uint32_t biased = abs - 0x38000000U;  // rebias 127 -> 15
  const std::uint32_t kept = biased >> 13;
  const std::uint32_t rest = biased & 0x1FFFU;
  const std::uint32_t rounded =
      kept + (rest > 0x1000U || (rest == 0x1000U && (kept & 1U)) ? 1U : 0U);
  return static_cast<std::uint16_t>(sign | rounded);
}
QFN_Q4K_HD inline float HalfFloat(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16;
  const std::uint32_t exponent = (h >> 10) & 0x1FU;
  const std::uint32_t mantissa = h & 0x3FFU;
  if (exponent == 0) {
    // Exact: mantissa * 2^-24.
    const float magnitude =
        static_cast<float>(mantissa) * 5.9604644775390625e-8F;
    return sign ? -magnitude : magnitude;
  }
  if (exponent == 31)
    return BitsFloat(sign | 0x7F800000U | (mantissa << 13));
  return BitsFloat(sign | ((exponent + 112) << 23) | (mantissa << 13));
}

QFN_Q4K_HD inline int NearestInt(float value) {
  // ggml's nearest_int: round half to even through the 1.5 * 2^23 trick.
  const float shifted = value + 12582912.0F;
  return static_cast<int>(FloatBits(shifted) & 0x007FFFFFU) - 0x00400000;
}

QFN_Q4K_HD inline int Code(float iscale, float x, float min) {
  const int l = NearestInt(iscale * (x - min));
  return l < 0 ? 0 : (l > 15 ? 15 : l);
}

/// make_qkx2_quants(32, 15, x, w, L, &the_min, Laux, -1, 0.1, 20, false):
/// returns the scale and stores the negated minimum.
QFN_Q4K_HD inline float Qkx2Scale(const float* x, const float* w,
                                  float* the_min) {
  float min = x[0];
  float max = x[0];
  float sum_w = w[0];
  float sum_x = sum_w * x[0];
  for (int i = 1; i < 32; ++i) {
    if (x[i] < min)
      min = x[i];
    if (x[i] > max)
      max = x[i];
    sum_w += w[i];
    sum_x += w[i] * x[i];
  }
  if (min > 0)
    min = 0;
  if (max == min) {
    *the_min = -min;
    return 0.0F;
  }
  float iscale = 15.0F / (max - min);
  float scale = 1 / iscale;
  float best_error = 0;
  for (int i = 0; i < 32; ++i) {
    const int l = Code(iscale, x[i], min);
    const float diff = scale * static_cast<float>(l) + min - x[i];
    best_error += w[i] * (diff * diff);
  }
  for (int step = 0; step <= 20; ++step) {
    iscale = (-1.0F + 0.1F * static_cast<float>(step) + 15.0F) / (max - min);
    float sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < 32; ++i) {
      const float l = static_cast<float>(Code(iscale, x[i], min));
      sum_l += w[i] * l;
      sum_l2 += w[i] * l * l;
      sum_xl += w[i] * l * x[i];
    }
    const float det = sum_w * sum_l2 - sum_l * sum_l;
    if (det <= 0)
      continue;
    float this_scale = (sum_w * sum_xl - sum_x * sum_l) / det;
    float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det;
    if (this_min > 0) {
      this_min = 0;
      this_scale = sum_xl / sum_l2;
    }
    float error = 0;
    for (int i = 0; i < 32; ++i) {
      const float l = static_cast<float>(Code(iscale, x[i], min));
      const float diff = this_scale * l + this_min - x[i];
      error += w[i] * (diff * diff);
    }
    if (error < best_error) {
      best_error = error;
      scale = this_scale;
      min = this_min;
    }
  }
  *the_min = -min;
  return scale;
}

/// quantize_row_q4_K_ref for the 256 values x[0], x[stride], ...
QFN_Q4K_HD inline void QuantizeSuperblock(const float* x, std::size_t stride,
                                          Block* out) {
  float values[256];
  for (int i = 0; i < 256; ++i)
    values[i] = x[static_cast<std::size_t>(i) * stride];
  float scales[8], mins[8];
  float max_scale = 0, max_min = 0;
  for (int j = 0; j < 8; ++j) {
    const float* v = values + 32 * j;
    float sum_x2 = 0;
    for (int i = 0; i < 32; ++i)
      sum_x2 += v[i] * v[i];
    const float av_x = __builtin_sqrtf(sum_x2 / 32);
    float w[32];
    for (int i = 0; i < 32; ++i)
      w[i] = av_x + __builtin_fabsf(v[i]);
    scales[j] = Qkx2Scale(v, w, &mins[j]);
    if (scales[j] > max_scale)
      max_scale = scales[j];
    if (mins[j] > max_min)
      max_min = mins[j];
  }
  const float inv_scale = max_scale > 0 ? 63.0F / max_scale : 0.0F;
  const float inv_min = max_min > 0 ? 63.0F / max_min : 0.0F;
  for (int k = 0; k < 12; ++k)
    out->scales[k] = 0;
  for (int j = 0; j < 8; ++j) {
    int ls = NearestInt(inv_scale * scales[j]);
    int lm = NearestInt(inv_min * mins[j]);
    ls = ls > 63 ? 63 : ls;
    lm = lm > 63 ? 63 : lm;
    if (j < 4) {
      out->scales[j] = static_cast<std::uint8_t>(ls);
      out->scales[j + 4] = static_cast<std::uint8_t>(lm);
    } else {
      out->scales[j + 4] =
          static_cast<std::uint8_t>((ls & 0xF) | ((lm & 0xF) << 4));
      out->scales[j - 4] |= static_cast<std::uint8_t>((ls >> 4) << 6);
      out->scales[j] |= static_cast<std::uint8_t>((lm >> 4) << 6);
    }
  }
  out->d = HalfBits(max_scale / 63.0F);
  out->dmin = HalfBits(max_min / 63.0F);
  std::uint8_t codes[256];
  for (int j = 0; j < 8; ++j) {
    // get_scale_min_k4(j, scales)
    const std::uint8_t* q = out->scales;
    const int sc =
        j < 4 ? q[j] & 63 : (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    const int m = j < 4 ? q[j + 4] & 63 : (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    const float d = HalfFloat(out->d) * static_cast<float>(sc);
    const float dm = HalfFloat(out->dmin) * static_cast<float>(m);
    for (int i = 0; i < 32; ++i) {
      // A zero-scale sub-block dequantizes to its minimum whatever its codes;
      // ggml leaves its search codes there, this writes zeros.
      int l = 0;
      if (d != 0) {
        l = NearestInt((values[32 * j + i] + dm) / d);
        l = l < 0 ? 0 : (l > 15 ? 15 : l);
      }
      codes[32 * j + i] = static_cast<std::uint8_t>(l);
    }
  }
  // qs[32 * pair + l] = L[64 * pair + l] | L[64 * pair + 32 + l] << 4
  for (int byte = 0; byte < 128; ++byte) {
    const int pair = byte / 32;
    const int l = byte % 32;
    out->qs[byte] = static_cast<std::uint8_t>(codes[64 * pair + l] |
                                              (codes[64 * pair + 32 + l] << 4));
  }
}

}  // namespace gufo::models::qwen38_flash_next::q4k

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_DRAFT_HEAD_Q4K_HPP_
