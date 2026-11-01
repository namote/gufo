// Flash-Next EXL3 prefill MoE glue (W3): per-token gate/up rotation against
// the per-assignment RotateGateUpBuckets, and RestoreActivateScaleBuckets
// against RestoreActivateRotateBuckets<_Float16> + ScaleDownRows, at 2048
// tokens x 10 experts (512): bitwise equality and time.
#include <cstdio>
#include <cstring>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/exl3_moe.hip"

namespace fnr = gufo::models::qwen38_flash_next::rocm;
namespace ex = gufo::quant::exl3;

namespace {
void Ck(hipError_t e) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(e));
    std::exit(1);
  }
}
template<class T>
T* Up(const std::vector<T>& v) {
  T* p;
  Ck(hipMalloc(&p, v.size() * sizeof(T)));
  Ck(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
  return p;
}
template<class T>
std::vector<T> Down(const T* p, std::size_t n) {
  std::vector<T> v(n);
  Ck(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
  return v;
}
float TimeIt(const auto& fn, int reps) {
  hipEvent_t a, b;
  Ck(hipEventCreate(&a));
  Ck(hipEventCreate(&b));
  fn();
  Ck(hipEventRecord(a));
  for (int i = 0; i < reps; ++i)
    fn();
  Ck(hipEventRecord(b));
  Ck(hipEventSynchronize(b));
  float ms = 0;
  Ck(hipEventElapsedTime(&ms, a, b));
  return ms / float(reps);
}
std::uint16_t Half(float f) {
  return __half_as_ushort(__float2half(f));
}
}  // namespace

int main() {
  constexpr unsigned rows = 2048, used = 10, experts = 512, hidden = 2560,
                     ff = 640, A = rows * used;
  std::vector<std::int32_t> ids(A), slots(A), inverse(A);
  std::uint32_t lcg = 5;
  for (unsigned t = 0; t < rows; ++t)
    for (unsigned s = 0; s < used; ++s) {
      lcg = lcg * 1664525U + 1013904223U;
      ids[t * used + s] = int((lcg >> 8) % experts);
    }
  std::vector<std::uint32_t> offsets(experts + 1, 0);
  for (int e : ids)
    ++offsets[e + 1];
  for (unsigned e = 0; e < experts; ++e)
    offsets[e + 1] += offsets[e];
  std::vector<std::uint32_t> next(offsets.begin(), offsets.end() - 1);
  for (unsigned a = 0; a < A; ++a) {
    const unsigned p = next[ids[a]]++;
    slots[p] = int(a);
    inverse[a] = int(p);
  }
  // Per-expert scales: suh[hidden] (gate/up), svh[ff] (gate/up), suh[ff]
  // (down).
  std::vector<std::uint16_t> scales(experts * (2 * hidden + 3 * ff));
  for (std::size_t i = 0; i < scales.size(); ++i)
    scales[i] = Half(((i * 2654435761U) % 2 ? 1.0F : -1.0F) *
                     (0.75F + float(i % 89) / 300.0F));
  auto* d_scales = Up(scales);
  std::vector<ex::DeviceLinear> g(experts), u(experts), d(experts);
  for (unsigned e = 0; e < experts; ++e) {
    const std::size_t base = std::size_t(e) * (2 * hidden + 3 * ff);
    g[e].suh = d_scales + base;
    u[e].suh = d_scales + base + hidden;
    g[e].svh = d_scales + base + 2 * hidden;
    u[e].svh = d_scales + base + 2 * hidden + ff;
    d[e].suh = d_scales + base + 2 * hidden + 2 * ff;
  }
  auto *dg = Up(g), *du = Up(u), *dd = Up(d);
  std::vector<float> x(std::size_t(rows) * hidden);
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = float(int(i * 2246822519U % 2001) - 1000) / 400.0F;
  auto* d_x = Up(x);
  auto* d_ids = Up(ids);
  auto* d_slots = Up(slots);
  auto* d_inv = Up(inverse);
  auto* d_off = Up(offsets);
  std::uint16_t *ga, *ua, *gb, *ub;
  for (auto** p : {&ga, &ua, &gb, &ub})
    Ck(hipMalloc(p, std::size_t(A) * hidden * 2));
  const auto rot_old = [&] {
    hipLaunchKernelGGL(fnr::RotateGateUpBuckets, dim3(hidden / 128, A),
                       dim3(32), 0, nullptr, dg, du, d_x, ga, ua, d_off,
                       d_slots, experts, hidden, used);
  };
  const auto rot_new = [&] {
    hipLaunchKernelGGL(
        fnr::RotateGateUpTokens, dim3((hidden / 128 + 7) / 8, rows), dim3(256),
        0, nullptr, dg, du, d_x, gb, ub, d_ids, d_inv, hidden, used);
  };
  rot_old();
  rot_new();
  Ck(hipDeviceSynchronize());
  const bool rot_same =
      Down(ga, std::size_t(A) * hidden) == Down(gb, std::size_t(A) * hidden) &&
      Down(ua, std::size_t(A) * hidden) == Down(ub, std::size_t(A) * hidden);
  const float t_rot_old = TimeIt(rot_old, 10), t_rot_new = TimeIt(rot_new, 10);
  std::printf(
      "gate/up rotation: per assignment %.3f ms | per token %.3f ms %s\n",
      t_rot_old, t_rot_new, rot_same ? "bit-identical" : "DIFFERS");
  // Activation: gate/up FP16 inner dots with a spread of magnitudes (incl.
  // tiny rows, which take a positive exponent).
  std::vector<std::uint16_t> gi(std::size_t(A) * ff), ui(gi.size());
  for (std::size_t i = 0; i < gi.size(); ++i) {
    const float mag = (i / ff) % 7 == 0 ? 1e-4F : 1.0F;
    gi[i] = Half(mag * float(int(i * 2654435761U % 2001) - 1000) / 300.0F);
    ui[i] = Half(mag * float(int(i * 40503U % 2001) - 1000) / 300.0F);
  }
  auto* d_gi = Up(gi);
  auto* d_ui = Up(ui);
  std::uint16_t *oa, *ob;
  std::int32_t *ea, *eb;
  Ck(hipMalloc(&oa, std::size_t(A) * ff * 2));
  Ck(hipMalloc(&ob, std::size_t(A) * ff * 2));
  Ck(hipMalloc(&ea, A * 4));
  Ck(hipMalloc(&eb, A * 4));
  const auto act_old = [&] {
    hipLaunchKernelGGL(fnr::RestoreActivateRotateBuckets<_Float16>,
                       dim3(ff / 128, A), dim3(32), 0, nullptr, dg, du, dd,
                       d_off, experts, reinterpret_cast<_Float16*>(d_gi),
                       reinterpret_cast<const _Float16*>(d_ui), oa, ff, false);
    hipLaunchKernelGGL(fnr::ScaleDownRows, dim3(A), dim3(256), 0, nullptr, oa,
                       ea, ff);
  };
  const auto act_new = [&] {
    hipLaunchKernelGGL(fnr::RestoreActivateScaleBuckets, dim3(A),
                       dim3(32 * (ff / 128)), 0, nullptr, dg, du, dd, d_off,
                       experts, reinterpret_cast<const _Float16*>(d_gi),
                       reinterpret_cast<const _Float16*>(d_ui), ob, eb, ff);
  };
  act_old();
  act_new();
  Ck(hipDeviceSynchronize());
  const bool act_same =
      Down(oa, std::size_t(A) * ff) == Down(ob, std::size_t(A) * ff) &&
      Down(ea, A) == Down(eb, A);
  const float t_act_old = TimeIt(act_old, 10), t_act_new = TimeIt(act_new, 10);
  std::printf("activate + scale: two passes %.3f ms | fused %.3f ms %s\n",
              t_act_old, t_act_new, act_same ? "bit-identical" : "DIFFERS");
  return rot_same && act_same ? 0 : 1;
}
