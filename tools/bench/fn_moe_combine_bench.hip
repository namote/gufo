// Flash-Next EXL3 prefill MoE tail (W3): RestoreDownBuckets (scatter) +
// MoeEpilogue against the fused PrefillCombineKernel, at 2048 tokens x 10
// experts x 2560: bitwise equality and time.
#include "src/models/qwen38_flash_next/kernels/rocm/exl3_moe.hip"

#include <cstdio>
#include <cstring>
#include <vector>

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
float TimeIt(const auto& fn, int reps) {
  hipEvent_t a, b;
  Ck(hipEventCreate(&a));
  Ck(hipEventCreate(&b));
  for (int i = 0; i < 2; ++i)
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
}  // namespace

int main() {
  constexpr unsigned rows = 2048, used = 10, experts = 512, width = 2560;
  constexpr unsigned A = rows * used;
  std::vector<std::int32_t> ids(A), slots(A), inverse(A), exps(A);
  std::uint32_t lcg = 9;
  for (auto& v : ids) {
    lcg = lcg * 1664525U + 1013904223U;
    v = int((lcg >> 8) % experts);
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
    exps[p] = int(a % 5) - 1;  // -1 (fallback rows), 0, 1..3
  }
  std::vector<std::uint16_t> svh(width);
  for (unsigned i = 0; i < width; ++i)
    svh[i] = __half_as_ushort(__float2half(0.75F + float(i % 97) / 400.0F));
  auto* d_svh = Up(svh);
  std::vector<ex::DeviceLinear> bank(experts);
  for (auto& l : bank) {
    l.svh = d_svh;
    l.n = width;
  }
  std::vector<float> raw(std::size_t(A) * width), w(A), sh(std::size_t(rows) * width),
      gate(rows * 3);
  for (std::size_t i = 0; i < raw.size(); ++i)
    raw[i] = float(int(i * 2654435761U % 2001) - 1000) / 37.0F;
  for (unsigned i = 0; i < A; ++i)
    w[i] = 0.05F + 0.01F * float(i % 13);
  for (std::size_t i = 0; i < sh.size(); ++i)
    sh[i] = float(int(i % 31) - 15) * 0.0125F;
  for (unsigned i = 0; i < gate.size(); ++i)
    gate[i] = float(int(i % 7) - 3) * 0.5F;
  auto* d_bank = Up(bank);
  auto* d_off = Up(offsets);
  auto* d_slots = Up(slots);
  auto* d_inv = Up(inverse);
  auto* d_exp = Up(exps);
  auto* d_raw = Up(raw);
  auto* d_work = Up(raw);
  auto* d_w = Up(w);
  auto* d_sh = Up(sh);
  auto* d_g = Up(gate);
  float *d_scatter, *d_ref, *d_new;
  Ck(hipMalloc(&d_scatter, raw.size() * 4));
  Ck(hipMalloc(&d_ref, sh.size() * 4));
  Ck(hipMalloc(&d_new, sh.size() * 4));
  const auto old_path = [&] {
    Ck(hipMemcpyAsync(d_work, d_raw, raw.size() * 4, hipMemcpyDeviceToDevice));
    hipLaunchKernelGGL(fnr::RestoreDownBuckets, dim3(width / 128, A), dim3(32),
                       0, nullptr, d_bank, d_off, experts, d_exp, d_work, width,
                       d_scatter, d_slots);
    fnr::MoeEpilogue(d_scatter, d_w, d_sh, d_g + 1, 3, d_ref, rows, used,
                     width, nullptr);
  };
  const auto new_path = [&] {
    hipLaunchKernelGGL(fnr::PrefillCombineKernel, dim3((width / 128 + 7) / 8, rows),
                       dim3(256), 0, nullptr, d_bank, d_off, experts, d_inv,
                       d_exp, d_raw, d_w, d_sh, d_g + 1, 3u, d_new, used, width);
  };
  old_path();
  new_path();
  Ck(hipDeviceSynchronize());
  std::vector<float> a(sh.size()), b(sh.size());
  Ck(hipMemcpy(a.data(), d_ref, a.size() * 4, hipMemcpyDeviceToHost));
  Ck(hipMemcpy(b.data(), d_new, b.size() * 4, hipMemcpyDeviceToHost));
  const bool same = std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
  const float copy = TimeIt(
      [&] {
        Ck(hipMemcpyAsync(d_work, d_raw, raw.size() * 4,
                          hipMemcpyDeviceToDevice));
      },
      10);
  const float t_old = TimeIt(old_path, 10) - copy, t_new = TimeIt(new_path, 10);
  std::printf("MoE tail %u tokens: restore+scatter+epilogue %.3f ms | fused "
              "combine %.3f ms %s\n",
              rows, t_old, t_new, same ? "bit-identical" : "DIFFERS");
  return 0;
}
