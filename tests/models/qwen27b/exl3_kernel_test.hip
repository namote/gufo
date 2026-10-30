// X2b EXL3 fast decode/verify projection
// (src/models/qwen/hip/kernels/exl3_gemv.hpp) against FP64 references:
//   - the exact projection (src/core/quant/exl3/format.hpp GemvReference):
//     relative L2 below 1e-3, the FP16 rounding of the rotated input;
//   - the same projection fed the FP16-rounded rotated input: below 1e-5;
// plus every row count 1..64 (passes of 8, padded row tiles), input and
// output strides, the self-resetting segment counters and graph replay.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "src/core/quant/exl3/format.hpp"
#include "src/models/qwen/hip/kernels/exl3_gemv.hpp"

namespace {

using namespace gufo::quant::exl3;

int failures = 0;

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << "\n";
    ++failures;
  }
}

void Check(hipError_t e) {
  if (e != hipSuccess) {
    std::cerr << "hip error: " << hipGetErrorString(e) << "\n";
    std::abort();
  }
}

template<typename T>
T* Upload(const std::vector<T>& host) {
  T* device = nullptr;
  Check(hipMalloc(&device, host.size() * sizeof(T)));
  Check(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                  hipMemcpyHostToDevice));
  return device;
}

template<typename T>
std::vector<T> Download(const T* device, std::size_t count) {
  std::vector<T> host(count);
  Check(
      hipMemcpy(host.data(), device, count * sizeof(T), hipMemcpyDeviceToHost));
  return host;
}

std::vector<std::uint16_t> RandomScales(std::mt19937& rng, std::size_t n,
                                        std::uint16_t base) {
  std::vector<std::uint16_t> v(n);
  for (auto& x : v) {
    const std::uint32_t r = rng();
    x = static_cast<std::uint16_t>(((r >> 10) & 1) << 15 | base | (r & 0x3ff));
  }
  return v;
}

struct Problem {
  std::size_t k, n;
  int bits;
};

double RelL2(const std::vector<double>& want, const std::vector<float>& got,
             std::size_t rows, std::size_t n, std::size_t stride) {
  double num = 0.0;
  double den = 0.0;
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t j = 0; j < n; ++j) {
      const double d = got[r * stride + j] - want[r * n + j];
      num += d * d;
      den += want[r * n + j] * want[r * n + j];
    }
  }
  return std::sqrt(num / den);
}

void TestProblem(const Problem& p, std::mt19937& rng) {
  constexpr std::size_t kMaxRowsTested = 64;
  const BitRate rate{p.bits, false};
  std::vector<std::uint16_t> trellis((p.k / 16) * (p.n / 16) *
                                     rate.TileWords());
  for (auto& w : trellis)
    w = static_cast<std::uint16_t>(rng());
  const auto suh = RandomScales(rng, p.k, 0x3800);
  const auto svh = RandomScales(rng, p.n, 0x2000);
  const std::size_t x_stride = p.k + 128;
  const std::size_t y_stride = p.n + 256;
  std::normal_distribution<float> normal;
  std::vector<float> x(kMaxRowsTested * x_stride);
  for (auto& v : x)
    v = normal(rng);

  // FP64 references for every row.
  std::vector<std::uint16_t> inner(p.k * p.n);
  DequantInner(trellis.data(), p.k, p.n, rate, Codebook::kMul1, inner.data());
  std::vector<double> exact(kMaxRowsTested * p.n);
  std::vector<double> fp16_input(kMaxRowsTested * p.n);
  for (std::size_t r = 0; r < kMaxRowsTested; ++r) {
    std::vector<double> xr(p.k);
    for (std::size_t i = 0; i < p.k; ++i)
      xr[i] = x[r * x_stride + i];
    std::vector<double> yr(p.n);
    GemvReference(xr, inner.data(), suh.data(), svh.data(), p.k, p.n, yr);
    std::copy(yr.begin(), yr.end(), exact.begin() + r * p.n);
    // The fast path rounds H(suh * x) to FP16 before the products.
    std::vector<double> rot(p.k);
    for (std::size_t i = 0; i < p.k; ++i)
      rot[i] = xr[i] * HalfToDouble(suh[i]);
    HadamardBlocks(std::span<double>(rot));
    for (auto& v : rot)
      v = HalfToDouble(HalfFromDouble(v));
    std::vector<double> t(p.n, 0.0);
    for (std::size_t i = 0; i < p.k; ++i)
      for (std::size_t j = 0; j < p.n; ++j)
        t[j] += rot[i] * HalfToDouble(inner[i * p.n + j]);
    HadamardBlocks(std::span<double>(t));
    for (std::size_t j = 0; j < p.n; ++j)
      fp16_input[r * p.n + j] = t[j] * HalfToDouble(svh[j]);
  }

  auto* d_trellis = Upload(trellis);
  auto* d_suh = Upload(suh);
  auto* d_svh = Upload(svh);
  auto* d_x = Upload(x);
  float* d_y = nullptr;
  Check(hipMalloc(&d_y, kMaxRowsTested * y_stride * sizeof(float)));
  gufo::hip::Exl3FastWorkspace ws;
  ws.input_elements = 8 * p.k;
  Check(hipMalloc(&ws.input, ws.input_elements * sizeof(std::uint16_t)));
  ws.counter_count = p.n / 128;
  Check(hipMalloc(&ws.counters, ws.counter_count * sizeof(std::uint32_t)));
  Check(hipMemset(ws.counters, 0, ws.counter_count * sizeof(std::uint32_t)));
  const DeviceLinear lin{d_trellis,
                         d_suh,
                         d_svh,
                         static_cast<std::uint32_t>(p.k),
                         static_cast<std::uint32_t>(p.n),
                         rate,
                         Codebook::kMul1};
  Expect(gufo::hip::Exl3FastSupported(lin), "fast path supports the problem");

  const std::string name = "k=" + std::to_string(p.k) +
                           " n=" + std::to_string(p.n) +
                           " K=" + std::to_string(p.bits);
  double worst_exact = 0.0;
  double worst_fp16 = 0.0;
  for (std::size_t rows : {1, 2, 3, 4, 5, 8, 9, 17, 64}) {
    Check(hipMemset(d_y, 0xff, kMaxRowsTested * y_stride * sizeof(float)));
    gufo::hip::LaunchExl3FastProjection(lin, d_x, x_stride, rows, d_y, y_stride,
                                        ws, nullptr);
    Check(hipDeviceSynchronize());
    const auto y = Download(d_y, kMaxRowsTested * y_stride);
    worst_exact = std::max(worst_exact, RelL2(exact, y, rows, p.n, y_stride));
    worst_fp16 =
        std::max(worst_fp16, RelL2(fp16_input, y, rows, p.n, y_stride));
    // Nothing outside the rows x n window is written.
    bool untouched = true;
    for (std::size_t r = 0; r < kMaxRowsTested && untouched; ++r) {
      for (std::size_t j = 0; j < y_stride && untouched; ++j) {
        if (r < rows && j < p.n)
          continue;
        std::uint32_t bits = 0;
        std::memcpy(&bits, &y[r * y_stride + j], 4);
        untouched = bits == 0xffffffffU;
      }
    }
    Expect(untouched, name + " rows=" + std::to_string(rows) +
                          ": writes outside the output window");
    const auto counters = Download(ws.counters, ws.counter_count);
    Expect(std::all_of(counters.begin(), counters.end(),
                       [](std::uint32_t c) { return c == 0; }),
           name + ": counters not back to zero");
  }
  Expect(worst_exact < 1e-3,
         name + " relative L2 vs exact " + std::to_string(worst_exact));
  Expect(worst_fp16 < 1e-5,
         name + " relative L2 vs FP16 input " + std::to_string(worst_fp16));
  std::cout << name << ": relative L2 vs exact " << worst_exact
            << ", vs FP16-rounded input " << worst_fp16 << "\n";

  // Row r of an 8-row pass equals a 1-row call on that row (bits), and a
  // captured graph replays the eager result.
  {
    gufo::hip::LaunchExl3FastProjection(lin, d_x, x_stride, 8, d_y, y_stride,
                                        ws, nullptr);
    Check(hipDeviceSynchronize());
    const auto eager = Download(d_y, 8 * y_stride);
    bool rows_same = true;
    for (std::size_t r = 0; r < 8; ++r) {
      gufo::hip::LaunchExl3FastProjection(lin, d_x + r * x_stride, x_stride, 1,
                                          d_y, y_stride, ws, nullptr);
      Check(hipDeviceSynchronize());
      const auto one = Download(d_y, p.n);
      rows_same = rows_same && std::memcmp(one.data(), &eager[r * y_stride],
                                           p.n * sizeof(float)) == 0;
    }
    Expect(rows_same, name + ": 8-row pass differs from 1-row calls");

    hipStream_t stream = nullptr;
    Check(hipStreamCreate(&stream));
    hipGraph_t graph = nullptr;
    Check(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    gufo::hip::LaunchExl3FastProjection(lin, d_x, x_stride, 8, d_y, y_stride,
                                        ws, stream);
    Check(hipStreamEndCapture(stream, &graph));
    hipGraphExec_t exec = nullptr;
    Check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    bool replay_same = true;
    for (int i = 0; i < 2; ++i) {
      Check(hipMemset(d_y, 0, 8 * y_stride * sizeof(float)));
      Check(hipGraphLaunch(exec, stream));
      Check(hipStreamSynchronize(stream));
      const auto replay = Download(d_y, 8 * y_stride);
      for (std::size_t r = 0; r < 8; ++r)
        replay_same = replay_same &&
                      std::memcmp(&replay[r * y_stride], &eager[r * y_stride],
                                  p.n * sizeof(float)) == 0;
    }
    Expect(replay_same, name + ": graph replay differs from eager launch");
    Check(hipGraphExecDestroy(exec));
    Check(hipGraphDestroy(graph));
    Check(hipStreamDestroy(stream));
  }

  for (void* ptr : {static_cast<void*>(d_trellis), static_cast<void*>(d_suh),
                    static_cast<void*>(d_svh), static_cast<void*>(d_x),
                    static_cast<void*>(d_y), static_cast<void*>(ws.input),
                    static_cast<void*>(ws.counters)})
    Check(hipFree(ptr));
}

}  // namespace

int main() {
  std::mt19937 rng(20261030);
  // Narrow and wide outputs cover the 16-, 8- and 4-wave launch shapes.
  const Problem problems[] = {{1024, 2048, 5}, {6144, 1024, 5}, {512, 16384, 6},
                              {2048, 1024, 4}, {384, 256, 3},   {256, 512, 2},
                              {256, 384, 1}};
  for (const Problem& p : problems)
    TestProblem(p, rng);
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "qwen27b_exl3_kernel_test: all checks passed\n";
  return 0;
}
