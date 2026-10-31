// Flash-Next plain prefill FP16 WMMA tile variants (W3): kernels.hip.cpp
// compiled in, DenseF16GEMMKernel<..., kPlainPrefill> at the HC shapes with
// other tiles. Each output keeps its two K16 chains, so the variants must be
// bit-identical; the bench checks that.
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp"

#include <cstdio>
#include <cstring>
#include <vector>

namespace fnr = gufo::models::qwen38_flash_next::rocm;

namespace {
void Check(hipError_t e) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(e));
    std::exit(1);
  }
}
__global__ void FillHalfB(std::uint16_t* p, std::size_t n, std::uint32_t seed) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       i < n; i += std::size_t(gridDim.x) * blockDim.x) {
    std::uint32_t x = static_cast<std::uint32_t>(i) * 0x9E3779B9U + seed;
    x ^= x >> 15;
    x *= 0x2C1B3C6DU;
    p[i] = __half_as_ushort(__float2half(float(x % 65536) / 32768.0F - 1.0F));
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
  return ms / static_cast<float>(reps);
}
template<int BM, int BN, int BK, int WM, int WN>
void Plain(const void* w, const __half* x, float* y, std::size_t batch,
           std::size_t m, std::size_t k) {
  hipLaunchKernelGGL(
      (fnr::DenseF16GEMMKernel<BM, BN, BK, WM, WN, 1, false, false, false,
                               true, false, false, false,
                               fnr::WeightType::kF16, true>),
      dim3((batch + BN - 1) / BN, (m + BM - 1) / BM), dim3(256), 0, nullptr,
      w, x, y, batch, m, k);
}
}  // namespace

int main(int argc, char** argv) {
  const unsigned rows = argc > 1 ? unsigned(std::atoi(argv[1])) : 2048;
  for (auto [k, n] : {std::pair{10240U, 320U}, std::pair{320U, 10240U},
                      std::pair{2560U, 544U}}) {
    std::uint16_t *w, *x;
    float *y, *y0;
    Check(hipMalloc(&w, std::size_t(n) * k * 2));
    Check(hipMalloc(&x, std::size_t(rows) * k * 2));
    Check(hipMalloc(&y, std::size_t(rows) * n * 4));
    Check(hipMalloc(&y0, std::size_t(rows) * n * 4));
    hipLaunchKernelGGL(FillHalfB, dim3(1024), dim3(256), 0, 0, w,
                       std::size_t(n) * k, 1U);
    hipLaunchKernelGGL(FillHalfB, dim3(1024), dim3(256), 0, 0, x,
                       std::size_t(rows) * k, 2U);
    const auto* xh = reinterpret_cast<const __half*>(x);
    Plain<128, 128, 2, 2, 4>(w, xh, y0, rows, n, k);
    Check(hipDeviceSynchronize());
    std::vector<float> ref(std::size_t(rows) * n), got(ref.size());
    Check(hipMemcpy(ref.data(), y0, ref.size() * 4, hipMemcpyDeviceToHost));
    const double flop = 2.0 * rows * k * n;
    const auto run = [&](const char* tile, auto launch) {
      Check(hipMemset(y, 0, ref.size() * 4));
      launch();
      Check(hipDeviceSynchronize());
      Check(hipMemcpy(got.data(), y, got.size() * 4, hipMemcpyDeviceToHost));
      const bool same = std::memcmp(got.data(), ref.data(), got.size() * 4) == 0;
      const float ms = TimeIt(launch, 10);
      std::printf("  plain %5u->%-5u %-16s %.3f ms %.2f TFLOPS %s\n", k, n,
                  tile, ms, flop / ms / 1e9, same ? "bit-identical" : "DIFFERS");
    };
    run("128x128 bk2 2x4", [&] { Plain<128, 128, 2, 2, 4>(w, xh, y, rows, n, k); });
    run("64x64 bk2 2x4", [&] { Plain<64, 64, 2, 2, 4>(w, xh, y, rows, n, k); });
    run("64x128 bk2 2x4", [&] { Plain<64, 128, 2, 2, 4>(w, xh, y, rows, n, k); });
    run("128x64 bk2 4x2", [&] { Plain<128, 64, 2, 4, 2>(w, xh, y, rows, n, k); });
    run("128x128 bk4 2x4", [&] { Plain<128, 128, 4, 2, 4>(w, xh, y, rows, n, k); });
    run("64x64 bk4 2x4", [&] { Plain<64, 64, 4, 2, 4>(w, xh, y, rows, n, k); });
    if (n == 320) {
      run("320x64 bk2 2x4", [&] { Plain<320, 64, 2, 2, 4>(w, xh, y, rows, n, k); });
      run("320x128 bk2 2x4", [&] { Plain<320, 128, 2, 2, 4>(w, xh, y, rows, n, k); });
    }
    for (void* p : {static_cast<void*>(w), static_cast<void*>(x),
                    static_cast<void*>(y), static_cast<void*>(y0)})
      Check(hipFree(p));
  }
  return 0;
}
