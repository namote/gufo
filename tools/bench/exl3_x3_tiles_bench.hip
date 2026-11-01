// X3 trellis WMMA tile variants (W3): the production HalfPrefillGemmKernel
// compiled in (single TU) and launched with other tile shapes, for dense
// prefill (Batching 0) and expert buckets (Batching 2). Every variant keeps
// each output's K order, so values are identical; the bench checks that.
//   hipcc -O3 -std=c++20 --offload-arch=gfx1151 -DENGINE_ENABLE_HIP=1 -I. \
//     tools/bench/exl3_x3_tiles_bench.hip -o exl3_x3_tiles_bench
#include "src/models/qwen/hip/kernels/prefill_fp16.hip"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <vector>

namespace {

using gufo::core::GgmlType;
using gufo::hip::Exl3PrefillBucket;
using gufo::hip::HalfEpilogue;
using gufo::hip::HalfPrefillGemmKernel;
using gufo::quant::exl3::BitRate;
using gufo::quant::exl3::Codebook;
using gufo::quant::exl3::DeviceLinear;

void Check(hipError_t e) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(e));
    std::exit(1);
  }
}
template<class T>
T* Alloc(std::size_t n) {
  T* p = nullptr;
  Check(hipMalloc(&p, n * sizeof(T)));
  return p;
}
__global__ void Fill(std::uint32_t* p, std::size_t n, std::uint32_t seed) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       i < n; i += std::size_t(gridDim.x) * blockDim.x) {
    std::uint32_t x = static_cast<std::uint32_t>(i) * 0x9E3779B9U + seed;
    x ^= x >> 16;
    x *= 0x85EBCA6BU;
    x ^= x >> 13;
    p[i] = x;
  }
}
__global__ void FillHalf(std::uint16_t* p, std::size_t n, std::uint32_t seed,
                         bool scale) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       i < n; i += std::size_t(gridDim.x) * blockDim.x) {
    std::uint32_t x = static_cast<std::uint32_t>(i) * 0x9E3779B9U + seed;
    x ^= x >> 15;
    x *= 0x2C1B3C6DU;
    const float f = scale ? 0.75F + float(x % 512) / 1024.0F
                          : (float(x % 65536) / 32768.0F - 1.0F);
    p[i] = __half_as_ushort(__float2half(f));
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

template<int BM, int BN, int WM, int WN, int Bits>
void Dense(const DeviceLinear& l, const std::uint16_t* x, float* y,
           unsigned rows) {
  HalfPrefillGemmKernel<BM, BN, WM, WN, GgmlType::kQ8_0, HalfEpilogue::kStore,
                        false, GgmlType::kQ8_0, Bits>
      <<<dim3((rows + BN - 1) / BN, l.n / BM), WM * WN * 32>>>(
          l.trellis, reinterpret_cast<const short*>(x), y, int(rows), int(l.n),
          int(l.k), reinterpret_cast<const float*>(l.svh));
}

template<int BM, int BN, int WM, int WN>
void Buckets(const DeviceLinear* bank, const Exl3PrefillBucket* b,
             const std::uint32_t* count, unsigned max_buckets, unsigned k,
             unsigned n, const std::uint16_t* x, float* y) {
  HalfPrefillGemmKernel<BM, BN, WM, WN, GgmlType::kQ8_0, HalfEpilogue::kStore,
                        false, GgmlType::kQ8_0, 5, 2>
      <<<dim3(1, n / BM, max_buckets), WM * WN * 32>>>(
          bank, reinterpret_cast<const short*>(x), y, 0, int(n), int(k),
          reinterpret_cast<const float*>(count), b);
}

template<int BM, int BN, int WM, int WN>
void PlainF16(const std::uint16_t* w, const std::uint16_t* x, float* y,
              unsigned rows, unsigned m, unsigned k) {
  HalfPrefillGemmKernel<BM, BN, WM, WN, GgmlType::kF16, HalfEpilogue::kStore>
      <<<dim3((rows + BN - 1) / BN, (m + BM - 1) / BM), WM * WN * 32>>>(
          w, reinterpret_cast<const short*>(x), y, int(rows), int(m), int(k),
          nullptr);
}

}  // namespace

int main(int argc, char** argv) {
  const unsigned rows = argc > 1 ? unsigned(std::atoi(argv[1])) : 2048;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 10;
  if (argc > 3 && std::strcmp(argv[3], "plain") == 0) {
    for (auto [k, m] : {std::pair{10240U, 320U}, std::pair{320U, 10240U},
                        std::pair{2560U, 544U}}) {
      std::uint16_t *w, *x;
      float *y, *y0;
      Check(hipMalloc(&w, std::size_t(m) * k * 2));
      Check(hipMalloc(&x, std::size_t(rows) * k * 2));
      Check(hipMalloc(&y, std::size_t(rows) * m * 4));
      Check(hipMalloc(&y0, std::size_t(rows) * m * 4));
      hipLaunchKernelGGL(FillHalf, dim3(1024), dim3(256), 0, 0, w,
                         std::size_t(m) * k, 1U, false);
      hipLaunchKernelGGL(FillHalf, dim3(1024), dim3(256), 0, 0, x,
                         std::size_t(rows) * k, 2U, false);
      PlainF16<256, 256, 8, 4>(w, x, y0, rows, m, k);
      Check(hipDeviceSynchronize());
      std::vector<float> ref(std::size_t(rows) * m), got(ref.size());
      Check(hipMemcpy(ref.data(), y0, ref.size() * 4, hipMemcpyDeviceToHost));
      const double flop = 2.0 * rows * k * m;
      const auto run = [&](const char* tile, auto launch) {
        Check(hipMemset(y, 0, ref.size() * 4));
        launch();
        Check(hipDeviceSynchronize());
        Check(hipMemcpy(got.data(), y, got.size() * 4, hipMemcpyDeviceToHost));
        const bool same =
            std::memcmp(got.data(), ref.data(), got.size() * 4) == 0;
        const float ms = TimeIt(launch, reps);
        std::printf("  plain %5u->%-5u %-12s %.3f ms %.2f TFLOPS %s\n", k, m,
                    tile, ms, flop / ms / 1e9,
                    same ? "bit-identical" : "DIFFERS");
      };
      run("256x256 8x4", [&] { PlainF16<256, 256, 8, 4>(w, x, y, rows, m, k); });
      run("128x256 4x8", [&] { PlainF16<128, 256, 4, 8>(w, x, y, rows, m, k); });
      run("256x128 8x2", [&] { PlainF16<256, 128, 8, 2>(w, x, y, rows, m, k); });
      run("128x128 4x2", [&] { PlainF16<128, 128, 4, 2>(w, x, y, rows, m, k); });
      run("64x256 2x8", [&] { PlainF16<64, 256, 2, 8>(w, x, y, rows, m, k); });
      run("64x128 2x4", [&] { PlainF16<64, 128, 2, 4>(w, x, y, rows, m, k); });
      run("128x64 4x2", [&] { PlainF16<128, 64, 4, 2>(w, x, y, rows, m, k); });
      for (void* p : {static_cast<void*>(w), static_cast<void*>(x),
                      static_cast<void*>(y), static_cast<void*>(y0)})
        Check(hipFree(p));
    }
    return 0;
  }
  struct S {
    const char* name;
    unsigned k, n;
  } shapes[] = {{"gdn qkv", 2560, 10240}, {"gdn out", 6144, 2560},
                {"attn q", 2560, 12288}};
  for (const S& s : shapes) {
    if (argc > 3)
      break;
    const std::size_t words = std::size_t(s.k) * s.n * 7 / 32;
    auto* t = Alloc<std::uint32_t>(words);
    auto* svh = Alloc<std::uint16_t>(s.n);
    auto* x = Alloc<std::uint16_t>(std::size_t(rows) * s.k);
    auto* y = Alloc<float>(std::size_t(rows) * s.n);
    auto* y0 = Alloc<float>(std::size_t(rows) * s.n);
    hipLaunchKernelGGL(Fill, dim3(1024), dim3(256), 0, 0, t, words, 1U);
    hipLaunchKernelGGL(FillHalf, dim3(64), dim3(256), 0, 0, svh,
                       std::size_t(s.n), 2U, true);
    hipLaunchKernelGGL(FillHalf, dim3(1024), dim3(256), 0, 0, x,
                       std::size_t(rows) * s.k, 3U, false);
    DeviceLinear l{};
    l.trellis = reinterpret_cast<const std::uint16_t*>(t);
    l.svh = svh;
    l.suh = svh;
    l.k = s.k;
    l.n = s.n;
    l.rate = BitRate{7, false};
    l.codebook = Codebook::kMul1;
    Dense<256, 256, 8, 4, 7>(l, x, y0, rows);
    Check(hipDeviceSynchronize());
    std::vector<float> ref(std::size_t(rows) * s.n), got(ref.size());
    Check(hipMemcpy(ref.data(), y0, ref.size() * 4, hipMemcpyDeviceToHost));
    const double flop = 2.0 * rows * s.k * s.n;
    const auto run = [&](const char* tile, auto launch) {
      Check(hipMemset(y, 0, ref.size() * 4));
      launch();
      Check(hipDeviceSynchronize());
      Check(hipMemcpy(got.data(), y, got.size() * 4, hipMemcpyDeviceToHost));
      const bool same = std::memcmp(got.data(), ref.data(), got.size() * 4) == 0;
      const float ms = TimeIt(launch, reps);
      std::printf("  %-8s %-14s %.3f ms %.2f TFLOPS %s\n", s.name, tile, ms,
                  flop / ms / 1e9, same ? "bit-identical" : "DIFFERS");
    };
    run("256x256 8x4", [&] { Dense<256, 256, 8, 4, 7>(l, x, y, rows); });
    run("128x256 4x8", [&] { Dense<128, 256, 4, 8, 7>(l, x, y, rows); });
    run("256x128 8x2", [&] { Dense<256, 128, 8, 2, 7>(l, x, y, rows); });
    run("128x128 4x2", [&] { Dense<128, 128, 4, 2, 7>(l, x, y, rows); });
    run("128x128 4x4", [&] { Dense<128, 128, 4, 4, 7>(l, x, y, rows); });
    run("256x256 4x8", [&] { Dense<256, 256, 4, 8, 7>(l, x, y, rows); });
    for (void* p : {static_cast<void*>(t), static_cast<void*>(svh),
                    static_cast<void*>(x), static_cast<void*>(y),
                    static_cast<void*>(y0)})
      Check(hipFree(p));
  }
  // Routed buckets: 512 K5 experts, rows * 10 uniform assignments, 64-row
  // buckets.
  for (auto [k, n, skew] :
       {std::tuple{2560U, 640U, false}, std::tuple{640U, 2560U, false},
        std::tuple{2560U, 640U, true}, std::tuple{640U, 2560U, true}}) {
    constexpr unsigned kExperts = 512;
    const unsigned assignments = rows * 10;
    const std::size_t words = std::size_t(k) * n * 5 / 32;
    auto* t = Alloc<std::uint32_t>(words * kExperts);
    hipLaunchKernelGGL(Fill, dim3(4096), dim3(256), 0, 0, t, words * kExperts,
                       5U);
    std::vector<DeviceLinear> bank(kExperts);
    for (unsigned e = 0; e < kExperts; ++e) {
      bank[e].trellis = reinterpret_cast<const std::uint16_t*>(t + words * e);
      bank[e].k = k;
      bank[e].n = n;
      bank[e].rate = BitRate{5, false};
      bank[e].codebook = Codebook::kMul1;
    }
    std::vector<unsigned> per(kExperts, 0);
    std::uint32_t lcg = 12345;
    for (unsigned a = 0; a < assignments; ++a) {
      lcg = lcg * 1664525U + 1013904223U;
      // Skewed: half of the assignments go to the first 64 experts.
      const unsigned e = (lcg >> 8) % kExperts;
      ++per[skew && (lcg & 1U) != 0 ? e % 64 : e];
    }
    std::vector<Exl3PrefillBucket> buckets;
    unsigned first = 0;
    for (unsigned e = 0; e < kExperts; ++e) {
      for (unsigned r = 0; r < per[e]; r += 64)
        buckets.push_back({e, std::min(64U, per[e] - r), first + r});
      first += per[e];
    }
    const unsigned count = unsigned(buckets.size());
    const unsigned max_buckets = kExperts + assignments / 64;
    buckets.resize(max_buckets, Exl3PrefillBucket{0, 0, 0});
    auto* d_bank = Alloc<DeviceLinear>(kExperts);
    auto* d_b = Alloc<Exl3PrefillBucket>(max_buckets);
    auto* d_count = Alloc<std::uint32_t>(1);
    Check(hipMemcpy(d_bank, bank.data(), kExperts * sizeof(DeviceLinear),
                    hipMemcpyHostToDevice));
    Check(hipMemcpy(d_b, buckets.data(), max_buckets * sizeof(Exl3PrefillBucket),
                    hipMemcpyHostToDevice));
    Check(hipMemcpy(d_count, &count, 4, hipMemcpyHostToDevice));
    auto* x = Alloc<std::uint16_t>(std::size_t(assignments) * k);
    auto* y = Alloc<float>(std::size_t(assignments) * n);
    auto* y0 = Alloc<float>(std::size_t(assignments) * n);
    hipLaunchKernelGGL(FillHalf, dim3(1024), dim3(256), 0, 0, x,
                       std::size_t(assignments) * k, 6U, false);
    if (argc > 3 && std::strcmp(argv[3], "pmc") == 0) {
      // Counter collection: the production tile, one dispatch per shape.
      Buckets<128, 64, 2, 2>(d_bank, d_b, d_count, max_buckets, k, n, x, y);
      Check(hipDeviceSynchronize());
      std::printf("pmc routed %u->%u %s\n", k, n, skew ? "skew" : "unif");
      for (void* p :
           {static_cast<void*>(t), static_cast<void*>(d_bank),
            static_cast<void*>(d_b), static_cast<void*>(d_count),
            static_cast<void*>(x), static_cast<void*>(y),
            static_cast<void*>(y0)})
        Check(hipFree(p));
      continue;
    }
    Buckets<128, 64, 4, 2>(d_bank, d_b, d_count, max_buckets, k, n, x, y0);
    Check(hipDeviceSynchronize());
    std::vector<float> ref(std::size_t(assignments) * n), got(ref.size());
    Check(hipMemcpy(ref.data(), y0, ref.size() * 4, hipMemcpyDeviceToHost));
    const double unique = double(words) * 4 * kExperts;
    const auto run = [&](const char* tile, auto launch) {
      Check(hipMemset(y, 0, ref.size() * 4));
      launch();
      Check(hipDeviceSynchronize());
      Check(hipMemcpy(got.data(), y, got.size() * 4, hipMemcpyDeviceToHost));
      const bool same = std::memcmp(got.data(), ref.data(), got.size() * 4) == 0;
      const float ms = TimeIt(launch, reps);
      std::printf("  routed %4u->%-4u %s %-12s %.3f ms %6.1f GB/s %s\n", k,
                  n, skew ? "skew" : "unif", tile, ms, unique / ms / 1e6,
                  same ? "bit-identical" : "DIFFERS");
    };
    run("128x64 4x2", [&] {
      Buckets<128, 64, 4, 2>(d_bank, d_b, d_count, max_buckets, k, n, x, y);
    });
    run("64x64 2x2", [&] {
      Buckets<64, 64, 2, 2>(d_bank, d_b, d_count, max_buckets, k, n, x, y);
    });
    run("64x64 4x1", [&] {
      Buckets<64, 64, 4, 1>(d_bank, d_b, d_count, max_buckets, k, n, x, y);
    });
    run("128x64 8x1", [&] {
      Buckets<128, 64, 8, 1>(d_bank, d_b, d_count, max_buckets, k, n, x, y);
    });
    run("128x64 2x2", [&] {
      Buckets<128, 64, 2, 2>(d_bank, d_b, d_count, max_buckets, k, n, x, y);
    });
    run("64x64 2x1", [&] {
      Buckets<64, 64, 2, 1>(d_bank, d_b, d_count, max_buckets, k, n, x, y);
    });
    run("64x64 1x2", [&] {
      Buckets<64, 64, 1, 2>(d_bank, d_b, d_count, max_buckets, k, n, x, y);
    });
    run("128x64 2x1", [&] {
      Buckets<128, 64, 2, 1>(d_bank, d_b, d_count, max_buckets, k, n, x, y);
    });
    for (void* p :
         {static_cast<void*>(t), static_cast<void*>(d_bank),
          static_cast<void*>(d_b), static_cast<void*>(d_count),
          static_cast<void*>(x), static_cast<void*>(y), static_cast<void*>(y0)})
      Check(hipFree(p));
  }
  return 0;
}
