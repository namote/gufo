// Do v_dot2_f32_f16 and v_wmma_f32_16x16x16_f16 keep FP16 subnormal inputs
// on gfx1151? Each probe multiplies FP16 bit patterns and compares the FP32
// result with the widened-FP32 FMA result and with the result of the same
// inputs flushed to zero.
//   1. single product: x * 1 (and 1 * x), accumulator 0, x in
//      {0x0001, 0x0200, 0x03ff (subnormals), 0x0400 (smallest normal)}
//   2. cancellation: x * 1 + (-x) * 1 + y * 1, y normal
//   3. long accumulation: sum over 4096 products of a subnormal x and 1
// Usage: fp16_denorm_probe
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

using h2 = _Float16 __attribute__((ext_vector_type(2)));
using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ _Float16 H(std::uint16_t bits) {
  return __builtin_bit_cast(_Float16, bits);
}

// out[0..3]: dot2 of (x, 0)·(1, 0); (1, 0)·(x, 0); (x, -x)·(1, 1) + y;
// 4096 accumulated (x, x)·(1, 1).
__global__ void Dot2Probe(std::uint16_t xb, std::uint16_t yb, float* out) {
  const _Float16 x = H(xb), one = H(0x3c00), zero = H(0);
  out[0] = __builtin_amdgcn_fdot2(h2{x, zero}, h2{one, zero}, 0.0F, false);
  out[1] = __builtin_amdgcn_fdot2(h2{one, zero}, h2{x, zero}, 0.0F, false);
  const float y = static_cast<float>(H(yb));
  out[2] = __builtin_amdgcn_fdot2(h2{x, static_cast<_Float16>(-x)},
                                  h2{one, one}, y, false);
  float acc = 0.0F;
  for (int i = 0; i < 2048; ++i)
    acc = __builtin_amdgcn_fdot2(h2{x, x}, h2{one, one}, acc, false);
  out[3] = acc;
}

// WMMA: A row 0 = (x, 0, ..., 0), B column 0 = (1, 0, ..., 0) -> C[0][0];
// and A row 0 all x, B column 0 all 1, accumulated over 256 instructions.
__global__ void WmmaProbe(std::uint16_t xb, float* out) {
  const unsigned lane = threadIdx.x;
  const _Float16 x = H(xb), one = H(0x3c00);
  v16h a = {};
  v16h b = {};
  // RDNA3 wave32 WMMA: lane l (and l + 16) holds row/column l % 16 of the
  // 16 x 16 fragment, element j = k index.
  if (lane % 16 == 0) {
    a[0] = x;
    b[0] = one;
  }
  v8f c = {};
  c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
  // Lane 0 holds C rows 0, 2, ..., 14 of column 0 (element 0 = row 0).
  if (lane == 0) out[0] = c[0];
  v16h a2 = {};
  v16h b2 = {};
  if (lane % 16 == 0) {
    for (int j = 0; j < 16; ++j) {
      a2[j] = x;
      b2[j] = one;
    }
  }
  v8f acc = {};
  for (int i = 0; i < 256; ++i)
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a2, b2, acc);
  if (lane == 0) out[1] = acc[0];
}

float HalfToFloat(std::uint16_t bits) {
  const int exp = (bits >> 10) & 0x1f, man = bits & 0x3ff;
  const float sign = (bits & 0x8000) ? -1.0F : 1.0F;
  if (exp == 0) return sign * std::ldexp(static_cast<float>(man), -24);
  return sign * std::ldexp(1.0F + man / 1024.0F, exp - 15);
}

}  // namespace

int main() {
  float* d = nullptr;
  if (hipMalloc(&d, 16 * sizeof(float)) != hipSuccess) return 1;
  const std::uint16_t xs[] = {0x0001, 0x0200, 0x03ff, 0x0400};
  const std::uint16_t y = 0x1400;  // 2^-10
  int flushed = 0;
  std::printf("x\tprobe\tgot\twidened_fp32\tflushed_input\tverdict\n");
  for (const std::uint16_t xb : xs) {
    const float x = HalfToFloat(xb), yf = HalfToFloat(y);
    const bool sub = xb < 0x0400;
    hipLaunchKernelGGL(Dot2Probe, dim3(1), dim3(1), 0, 0, xb, y, d);
    float dot[4];
    hipMemcpy(dot, d, sizeof(dot), hipMemcpyDeviceToHost);
    hipLaunchKernelGGL(WmmaProbe, dim3(1), dim3(32), 0, 0, xb, d + 4);
    float wm[2];
    hipMemcpy(wm, d + 4, sizeof(wm), hipMemcpyDeviceToHost);
    // Widened FP32 references (sequential FP32 FMAs) and flushed inputs.
    float long_dot = 0.0F;
    for (int i = 0; i < 2048; ++i) long_dot = std::fmaf(x, 1.0F, std::fmaf(x, 1.0F, long_dot));
    float long_wmma = 0.0F;
    for (int i = 0; i < 256; ++i) long_wmma += 16.0F * x;
    const float fx = sub ? 0.0F : x;
    float long_dot_f = 0.0F;
    for (int i = 0; i < 2048; ++i) long_dot_f += 2.0F * fx;
    struct Row {
      const char* probe;
      float got, wide, flush;
    } rows[] = {{"dot2 x*1", dot[0], x, fx},
                {"dot2 1*x", dot[1], x, fx},
                {"dot2 x-x+y", dot[2], yf, yf},
                {"dot2 4096 x", dot[3], long_dot, long_dot_f},
                {"wmma x*1", wm[0], x, fx},
                {"wmma 4096 x", wm[1], long_wmma, 256.0F * 16.0F * fx}};
    for (const Row& r : rows) {
      const bool keeps = r.got == r.wide;
      const bool flushes = !keeps && r.got == r.flush;
      if (sub && !keeps) ++flushed;
      std::printf("0x%04x\t%s\t%.9g\t%.9g\t%.9g\t%s\n", xb, r.probe, r.got,
                  r.wide, r.flush,
                  keeps ? "exact" : flushes ? "FLUSHED" : "DIFFERS");
    }
  }
  std::printf("subnormal probes not matching widened FP32: %d\n", flushed);
  (void)hipFree(d);
  return 0;
}
