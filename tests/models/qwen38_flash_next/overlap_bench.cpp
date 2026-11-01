// Kernel-level overlap on gfx1151: a compute-bound prefill GEMM on one stream
// against bandwidth-bound HC passes on another, at the Flash-Next prefill
// shapes (2,048 tokens, four 2,560-wide streams).
//
//   GEMM side: the plain FP16 X3 GEMM (LaunchPlainF16PrefillGemm) at
//   2560 -> 12288 (the shape class of the dense projections) and the
//   W8A8 HC down projection (10240 -> 320).
//   Bandwidth side: the four-lane HC combine writing the F16 norm, then the
//   recompute mix epilogue (HcMixEpilogueVec4FromRes).
//
// Each side runs `reps` times alone, then both sides are issued together on
// two non-blocking streams. overlap = (alone_a + alone_b - together) /
// min(alone_a, alone_b): 1 is perfect hiding of the shorter side, 0 none.
//
//   qwen38_flash_next_overlap_bench [tokens] [reps]
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

void CheckHip(hipError_t e, const char* what) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

template<typename T>
T* Device(std::size_t n, float scale, std::uint32_t seed) {
  std::vector<T> host(n);
  std::mt19937 rng(seed);
  std::normal_distribution<float> d;
  for (auto& v : host) {
    if constexpr (std::is_same_v<T, __half>)
      v = __float2half(d(rng) * scale);
    else
      v = static_cast<T>(d(rng) * scale);
  }
  T* p = nullptr;
  CheckHip(hipMalloc(&p, n * sizeof(T) + 4096), "malloc");
  CheckHip(hipMemcpy(p, host.data(), n * sizeof(T), hipMemcpyHostToDevice),
           "upload");
  return p;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::uint32_t tokens = argc > 1 ? std::stoul(argv[1]) : 2048;
    const int reps = argc > 2 ? std::stoi(argv[2]) : 10;
    constexpr std::uint32_t kHidden = 2560, kStreams = 4, kRank = 320;
    constexpr std::uint32_t kHc = kHidden * kStreams;
    constexpr std::uint32_t kDenseOut = 12288;
    // GEMM side operands.
    auto* w_dense = Device<__half>(std::size_t{kDenseOut} * kHidden, 0.02F, 1);
    auto* x_dense = Device<__half>(std::size_t{tokens} * kHidden, 1.0F, 2);
    auto* y_dense = Device<float>(std::size_t{tokens} * kDenseOut, 0.0F, 3);
    auto* w_down = Device<__half>(std::size_t{kRank} * kHc, 0.02F, 4);
    void* w_down_q8 = nullptr;
    CheckHip(hipMalloc(&w_down_q8, std::size_t{kRank} * kHc / 32 * 34 + 4096),
             "q8");
    q::QuantizeHalfQ8_0(w_down, w_down_q8, std::size_t{kRank} * kHc, nullptr);
    void* xn_q8t = nullptr;
    CheckHip(hipMalloc(&xn_q8t, q::Q8TiledBytes(tokens, kHc)), "xn q8");
    auto* lo = Device<float>(std::size_t{tokens} * kRank, 0.0F, 5);
    // Bandwidth side operands.
    auto* res = Device<float>(std::size_t{tokens} * kHc, 1.0F, 6);
    auto* block_out = Device<float>(std::size_t{tokens} * kHidden, 1.0F, 7);
    const auto parts = q::HcInjectPartsVec4(kHidden);
    auto* inject =
        Device<float>(std::size_t{tokens} * kStreams * parts, 0.1F, 8);
    auto* gamma = Device<float>(kHc, 0.5F, 9);
    auto* xn = Device<float>(std::size_t{tokens} * kHc, 0.0F, 10);
    auto* xn_half = Device<__half>(std::size_t{tokens} * kHc, 0.0F, 11);
    auto* scale = Device<float>(std::size_t{tokens} * kStreams, 0.0F, 12);
    auto* gate = Device<float>(std::size_t{tokens} * kHc, 1.0F, 13);
    auto* inject_w = Device<float>(std::size_t{kStreams} * kHc, 0.1F, 14);
    auto* mixed = Device<float>(std::size_t{tokens} * kHidden, 0.0F, 15);
    auto* inject_out =
        Device<float>(std::size_t{tokens} * kStreams * parts, 0.0F, 16);
    // The combine overwrites res; refreshed copies would only add traffic,
    // and the timing does not depend on the values.

    hipStream_t sa, sb;
    CheckHip(hipStreamCreateWithFlags(&sa, hipStreamNonBlocking), "stream");
    CheckHip(hipStreamCreateWithFlags(&sb, hipStreamNonBlocking), "stream");
    const auto dense = [&](hipStream_t s) {
      gufo::hip::LaunchPlainF16PrefillGemm(w_dense, x_dense, y_dense, tokens,
                                           kDenseOut, kHidden, s);
    };
    const auto down = [&](hipStream_t s) {
      (void)q::W8A8Gemm(w_down_q8, xn_q8t, lo, tokens, kRank, kHc, s);
    };
    const auto hc = [&](hipStream_t s) {
      (void)q::HcCombine(res, block_out, inject, parts, gamma, xn, tokens,
                         kHidden, kStreams, 1e-6F, s, false, true, xn_half,
                         scale);
      q::HcMixEpilogueVec4FromRes(res, gamma, scale, gate, inject_w, mixed,
                                  inject_out, tokens, kHidden, s);
    };
    hipEvent_t e0, e1;
    CheckHip(hipEventCreate(&e0), "event");
    CheckHip(hipEventCreate(&e1), "event");
    const auto time = [&](auto&& body) {
      body();  // warm-up
      CheckHip(hipDeviceSynchronize(), "sync");
      CheckHip(hipEventRecord(e0, sa), "record");
      CheckHip(hipStreamWaitEvent(sb, e0, 0), "wait");
      body();
      hipEvent_t eb;
      CheckHip(hipEventCreate(&eb), "event");
      CheckHip(hipEventRecord(eb, sb), "record");
      CheckHip(hipStreamWaitEvent(sa, eb, 0), "wait");
      CheckHip(hipEventRecord(e1, sa), "record");
      CheckHip(hipEventSynchronize(e1), "sync");
      (void)hipEventDestroy(eb);
      float ms = 0;
      CheckHip(hipEventElapsedTime(&ms, e0, e1), "elapsed");
      return static_cast<double>(ms);
    };
    std::cout << std::fixed << std::setprecision(3);
    for (const auto& [name, gemm] :
         {std::pair<const char*, std::function<void(hipStream_t)>>{
              "dense X3 F16 2560->12288", dense},
          {"W8A8 HC down 10240->320", down}}) {
      const double a = time([&] {
        for (int r = 0; r < reps; ++r)
          gemm(sa);
      });
      const double b = time([&] {
        for (int r = 0; r < reps; ++r)
          hc(sb);
      });
      const double ab = time([&] {
        for (int r = 0; r < reps; ++r) {
          gemm(sa);
          hc(sb);
        }
      });
      const double overlap = (a + b - ab) / std::min(a, b);
      std::cout << name << " x" << reps << ": gemm " << a << " ms, hc " << b
                << " ms, together " << ab << " ms, serial " << a + b
                << " ms, overlap " << overlap << '\n';
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
