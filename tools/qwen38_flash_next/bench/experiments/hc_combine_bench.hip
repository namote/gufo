// Flash-Next HC combine at prefill width (W3): the per-stream
// HcCombineKernel<float> (the EXL3 route) against HcCombineVec4Kernel<float>.
#include "src/models/qwen38_flash_next/kernels/rocm/hc.hip.cpp"

#include <cstdio>

namespace fnr = gufo::models::qwen38_flash_next::rocm;

namespace {
void Check(hipError_t e) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(e));
    std::exit(1);
  }
}
__global__ void FillF(float* p, std::size_t n, std::uint32_t seed) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       i < n; i += std::size_t(gridDim.x) * blockDim.x) {
    std::uint32_t x = static_cast<std::uint32_t>(i) * 0x9E3779B9U + seed;
    x ^= x >> 15;
    x *= 0x2C1B3C6DU;
    p[i] = float(x % 65536) / 32768.0F - 1.0F;
  }
}
float TimeIt(const auto& fn, int reps) {
  hipEvent_t a, b;
  Check(hipEventCreate(&a));
  Check(hipEventCreate(&b));
  for (int i = 0; i < 2; ++i)
    fn();
  Check(hipEventRecord(a));
  for (int i = 0; i < reps; ++i)
    fn();
  Check(hipEventRecord(b));
  Check(hipEventSynchronize(b));
  float ms = 0;
  Check(hipEventElapsedTime(&ms, a, b));
  return ms * 1000.0F / static_cast<float>(reps);
}
}  // namespace

int main(int argc, char** argv) {
  const unsigned tokens = argc > 1 ? unsigned(std::atoi(argv[1])) : 2048;
  constexpr unsigned hidden = 2560, streams = 4, parts = 10;
  float *res, *out, *inject, *gamma, *xn;
  Check(hipMalloc(&res, std::size_t(tokens) * streams * hidden * 4));
  Check(hipMalloc(&out, std::size_t(tokens) * hidden * 4));
  Check(hipMalloc(&inject, std::size_t(tokens) * streams * parts * 4));
  Check(hipMalloc(&gamma, std::size_t(streams) * hidden * 4));
  Check(hipMalloc(&xn, std::size_t(tokens) * streams * hidden * 4));
  for (auto [p, n] : {std::pair{res, std::size_t(tokens) * streams * hidden},
                      std::pair{out, std::size_t(tokens) * hidden},
                      std::pair{inject, std::size_t(tokens) * streams * parts},
                      std::pair{gamma, std::size_t(streams) * hidden}})
    hipLaunchKernelGGL(FillF, dim3(1024), dim3(256), 0, 0, p, n, 1U);
  const float scalar = TimeIt(
      [&] {
        hipLaunchKernelGGL(fnr::HcCombineKernel<float>, dim3(tokens, streams),
                           dim3(fnr::kThreads), 0, nullptr, res, out, inject,
                           parts, gamma, xn, nullptr, hidden, streams, 1e-6F,
                           nullptr);
      },
      20);
  const float vec4 = TimeIt(
      [&] {
        hipLaunchKernelGGL(fnr::HcCombineVec4Kernel<float>, dim3(tokens),
                           dim3(fnr::kThreads), 0, nullptr, res, out, inject,
                           parts, gamma, xn, nullptr, hidden, 1e-6F, nullptr);
      },
      20);
  const double bytes = double(tokens) * hidden * 4 * (streams * 3 + 1);
  std::printf("HC combine %u tokens: per-stream %.1f us (%.0f GB/s) | vec4 "
              "%.1f us (%.0f GB/s)\n",
              tokens, scalar, bytes / scalar / 1e3, vec4, bytes / vec4 / 1e3);
  return 0;
}
