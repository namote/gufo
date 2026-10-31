// v_dot2_f32_f16 (VOP3) against v_dot2acc_f32_f16 (VOP2 accumulate form) on
// gfx1151, both forced with inline asm, on the same operands: FP16 pairs
// (a0, a1) x (b0, b1) plus an FP32 accumulator c. Each result is compared
// with the exact value a0 b0 + a1 b1 + c rounded once to FP32 (computed in
// double: FP16 products and their sum with c are exact or nearly so in
// double) and with the sequential FP32 FMA fma(a1, b1, fma(a0, b0, c)).
// Case classes: subnormal activations x non-power-of-two weights, products
// that cancel, nonzero dependent accumulators, and a random sweep including
// subnormal ranges.
// Usage: fp16_dot2_forms_probe
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <algorithm>
#include <vector>

namespace {

struct Case {
  std::uint32_t a, b;  // packed FP16 pairs (lo = element 0)
  float c;
};

__global__ void Probe(const Case* cases, int n, float* vop3, float* vop2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const Case k = cases[i];
  float d3;
  asm volatile("v_dot2_f32_f16 %0, %1, %2, %3"
               : "=v"(d3)
               : "v"(k.a), "v"(k.b), "v"(k.c));
  float d2 = k.c;
  asm volatile("v_dot2acc_f32_f16 %0, %1, %2" : "+v"(d2) : "v"(k.a), "v"(k.b));
  vop3[i] = d3;
  vop2[i] = d2;
}

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

// v_wmma_f32_16x16x16_f16 on the same pair: A row 0 = (a0, a1, 0, ...), B
// column 0 = (b0, b1, 0, ...), C[0][0] = c. One wave per case.
__global__ void WmmaProbe(const Case* cases, int n, float* out) {
  const int i = blockIdx.x;
  if (i >= n) return;
  const Case k = cases[i];
  const unsigned lane = threadIdx.x;
  v16h a = {};
  v16h b = {};
  if (lane % 16 == 0) {
    a[0] = __builtin_bit_cast(_Float16, static_cast<std::uint16_t>(k.a & 0xffff));
    a[1] = __builtin_bit_cast(_Float16, static_cast<std::uint16_t>(k.a >> 16));
    b[0] = __builtin_bit_cast(_Float16, static_cast<std::uint16_t>(k.b & 0xffff));
    b[1] = __builtin_bit_cast(_Float16, static_cast<std::uint16_t>(k.b >> 16));
  }
  v8f c = {};
  if (lane == 0) c[0] = k.c;
  c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
  if (lane == 0) out[i] = c[0];
}

double Half(std::uint16_t bits) {
  const int exp = (bits >> 10) & 0x1f, man = bits & 0x3ff;
  const double sign = (bits & 0x8000) ? -1.0 : 1.0;
  if (exp == 0) return sign * std::ldexp(static_cast<double>(man), -24);
  if (exp == 31) return NAN;
  return sign * std::ldexp(1.0 + man / 1024.0, exp - 15);
}

std::uint32_t Pack(std::uint16_t lo, std::uint16_t hi) {
  return static_cast<std::uint32_t>(lo) | (static_cast<std::uint32_t>(hi) << 16);
}

}  // namespace

int main() {
  std::vector<Case> cases;
  std::vector<const char*> klass;
  const std::uint16_t subs[] = {0x0001, 0x0123, 0x0200, 0x03ff, 0x8155};
  const std::uint16_t weights[] = {0x3555, 0x3a8f, 0xb8e4, 0x2e66, 0x4247};
  const float accs[] = {0.0F, 1.0e-7F, -3.14159e-6F, 0.75F};
  for (auto s : subs)
    for (auto w : weights)
      for (float c : accs) {
        cases.push_back({Pack(s, 0x0001), Pack(w, w ^ 0x0400), c});
        klass.push_back("subnormal x weight");
      }
  // Cancellation: a0 b0 = -a1 b1 (+ small accumulator).
  for (auto w : weights)
    for (float c : accs) {
      cases.push_back({Pack(0x3c01, 0x3c01), Pack(w, w ^ 0x8000), c});
      klass.push_back("cancellation");
      cases.push_back({Pack(0x0155, 0x0155), Pack(w, w ^ 0x8000), c});
      klass.push_back("cancellation subnormal");
    }
  // Random sweep: one random half, one subnormal-range half, random weights
  // and accumulators of several scales.
  std::mt19937 rng(20261031);
  for (int i = 0; i < 200000; ++i) {
    const std::uint16_t a0 = static_cast<std::uint16_t>(rng() & 0x7bff) ^
                             static_cast<std::uint16_t>((rng() & 1) << 15);
    const std::uint16_t a1 = static_cast<std::uint16_t>(rng() & 0x07ff);  // < 2^-13
    const std::uint16_t b0 = static_cast<std::uint16_t>(0x3000 + (rng() % 0x1000));
    const std::uint16_t b1 = static_cast<std::uint16_t>(0x3000 + (rng() % 0x1000));
    const float c = std::ldexp(static_cast<float>(rng() % 2001) - 1000.0F,
                               -static_cast<int>(rng() % 40));
    cases.push_back({Pack(a0, a1), Pack(b0, b1), c});
    klass.push_back((a1 & 0x7c00) == 0 || (a0 & 0x7c00) == 0
                        ? "random subnormal"
                        : "random normal");
  }
  const int n = static_cast<int>(cases.size());
  Case* dc = nullptr;
  float *d3 = nullptr, *d2 = nullptr;
  if (hipMalloc(&dc, n * sizeof(Case)) != hipSuccess) return 1;
  (void)hipMalloc(&d3, n * 4);
  (void)hipMalloc(&d2, n * 4);
  (void)hipMemcpy(dc, cases.data(), n * sizeof(Case), hipMemcpyHostToDevice);
  hipLaunchKernelGGL(Probe, dim3((n + 255) / 256), dim3(256), 0, 0, dc, n, d3,
                     d2);
  std::vector<float> v3(n), v2(n);
  (void)hipMemcpy(v3.data(), d3, n * 4, hipMemcpyDeviceToHost);
  (void)hipMemcpy(v2.data(), d2, n * 4, hipMemcpyDeviceToHost);
  struct Count {
    int n = 0, vop3_exact = 0, vop2_exact = 0, vop3_fma = 0, vop2_fma = 0,
        forms_equal = 0;
    double vop2_worst = 0.0, vop3_worst = 0.0;
  };
  const char* names[] = {"subnormal x weight", "cancellation",
                         "cancellation subnormal", "random subnormal",
                         "random normal"};
  Count counts[5];
  int shown = 0;
  for (int i = 0; i < n; ++i) {
    const double a0 = Half(cases[i].a & 0xffff), a1 = Half(cases[i].a >> 16);
    const double b0 = Half(cases[i].b & 0xffff), b1 = Half(cases[i].b >> 16);
    const float exact = static_cast<float>(a0 * b0 + a1 * b1 + cases[i].c);
    const float fma = std::fmaf(static_cast<float>(a1), static_cast<float>(b1),
                                std::fmaf(static_cast<float>(a0),
                                          static_cast<float>(b0), cases[i].c));
    int k = 0;
    while (std::string(names[k]) != klass[i]) ++k;
    Count& c = counts[k];
    ++c.n;
    c.vop3_exact += v3[i] == exact;
    c.vop2_exact += v2[i] == exact;
    c.vop3_fma += v3[i] == fma;
    c.vop2_fma += v2[i] == fma;
    c.forms_equal += v3[i] == v2[i];
    const double scale = std::fabs(a0 * b0) + std::fabs(a1 * b1) +
                         std::fabs(cases[i].c) + 1e-30;
    c.vop2_worst = std::max(c.vop2_worst, std::fabs(v2[i] - exact) / scale);
    c.vop3_worst = std::max(c.vop3_worst, std::fabs(v3[i] - exact) / scale);
    if (v2[i] != v3[i] && shown < 12) {
      ++shown;
      std::printf("differ %s: a=%08x b=%08x c=%.9g exact=%.9g vop3=%.9g "
                  "vop2acc=%.9g\n",
                  klass[i], cases[i].a, cases[i].b, cases[i].c, exact, v3[i],
                  v2[i]);
    }
  }
  // WMMA on the deterministic classes (the first 140 cases).
  const int nw = 140;
  float* dw = nullptr;
  (void)hipMalloc(&dw, nw * 4);
  hipLaunchKernelGGL(WmmaProbe, dim3(nw), dim3(32), 0, 0, dc, nw, dw);
  std::vector<float> vw(nw);
  (void)hipMemcpy(vw.data(), dw, nw * 4, hipMemcpyDeviceToHost);
  int w_exact = 0, w_equal_dot = 0;
  double w_worst = 0.0;
  for (int i = 0; i < nw; ++i) {
    const double a0 = Half(cases[i].a & 0xffff), a1 = Half(cases[i].a >> 16);
    const double b0 = Half(cases[i].b & 0xffff), b1 = Half(cases[i].b >> 16);
    const float exact = static_cast<float>(a0 * b0 + a1 * b1 + cases[i].c);
    w_exact += vw[i] == exact;
    w_equal_dot += vw[i] == v3[i];
    const double scale = std::fabs(a0 * b0) + std::fabs(a1 * b1) +
                         std::fabs(cases[i].c) + 1e-30;
    w_worst = std::max(w_worst, std::fabs(vw[i] - exact) / scale);
  }
  std::printf("wmma on the %d deterministic cases: ==exact %d, ==dot2 %d, "
              "worst rel %.3g\n", nw, w_exact, w_equal_dot, w_worst);
  std::printf("class\tcases\tvop3==exact\tvop2acc==exact\tvop3==fma\t"
              "vop2acc==fma\tforms_equal\tvop3_worst_rel\tvop2acc_worst_rel\n");
  for (int k = 0; k < 5; ++k)
    std::printf("%s\t%d\t%d\t%d\t%d\t%d\t%d\t%.3g\t%.3g\n", names[k],
                counts[k].n, counts[k].vop3_exact, counts[k].vop2_exact,
                counts[k].vop3_fma, counts[k].vop2_fma, counts[k].forms_equal,
                counts[k].vop3_worst, counts[k].vop2_worst);
  return 0;
}
