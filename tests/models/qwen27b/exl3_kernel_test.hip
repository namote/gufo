// X2b EXL3 fast decode/verify projection
// (src/models/qwen/hip/kernels/exl3_gemv.hpp) against FP64 references:
//   - the exact projection (src/core/quant/exl3/format.hpp GemvReference):
//     relative L2 below 1e-3, the FP16 rounding of the rotated input;
//   - the same projection fed the kernel's FP16 rotated input (FP32 rotation
//     arithmetic replicated on the CPU): below 1e-5, FP32 accumulation;
// plus every row count 1..64 (passes of 8, padded row tiles), input and
// output strides, the self-resetting segment counters, graph replay,
// shared-input groups of two and three matrices in one launch, and the fused
// SwiGLU MLP against its unfused launches (bits).

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
#include "src/models/qwen/hip/ops/swiglu.hpp"

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

// The rotated FP16 input exactly as the kernel makes it
// (src/core/hip/detail/exl3_gemv.hpp "Input rotation arithmetic"): FP32
// product, FP32 butterflies in reference order, one FP32 scale, FP16. An FP64
// rotation rounded to FP16 lands on the other side of an FP16 rounding
// boundary for about 1 element in 2000, which alone moves the output by 2e-6
// to 2e-5 (relative L2).
std::vector<double> KernelRotatedInput(const float* x, const std::uint16_t* suh,
                                       std::size_t k) {
  std::vector<double> rot(k);
  for (std::size_t s = 0; s < k; s += 128) {
    float seg[128];
    for (std::size_t i = 0; i < 128; ++i)
      seg[i] = x[s + i] * static_cast<float>(HalfToDouble(suh[s + i]));
    for (std::size_t len = 1; len < 128; len <<= 1)
      for (std::size_t i = 0; i < 128; i += len << 1)
        for (std::size_t j = i; j < i + len; ++j) {
          const float a = seg[j];
          const float b = seg[j + len];
          seg[j] = a + b;
          seg[j + len] = a - b;
        }
    for (std::size_t i = 0; i < 128; ++i)
      rot[s + i] = HalfToDouble(HalfFromDouble(seg[i] * 0.0883883461356163F));
  }
  return rot;
}

// svh * H(rot W') in FP64 for one row.
std::vector<double> ProjectRotated(const std::vector<double>& rot,
                                   const std::vector<std::uint16_t>& inner,
                                   const std::vector<std::uint16_t>& svh,
                                   std::size_t k, std::size_t n) {
  std::vector<double> t(n, 0.0);
  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = 0; j < n; ++j)
      t[j] += rot[i] * HalfToDouble(inner[i * n + j]);
  HadamardBlocks(std::span<double>(t));
  for (std::size_t j = 0; j < n; ++j)
    t[j] *= HalfToDouble(svh[j]);
  return t;
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
    const std::vector<double> rot =
        KernelRotatedInput(&x[r * x_stride], suh.data(), p.k);
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

// Shared-input groups (q + k + v, gate + up shapes): one launch per pass,
// every slot against its FP64 reference fed the kernel's rotated input, and
// nothing written outside any slot's window.
void TestGroup(std::size_t k, std::vector<std::size_t> ns, int bits,
               std::mt19937& rng) {
  constexpr std::size_t kRows = 17;
  const BitRate rate{bits, false};
  const std::size_t x_stride = k + 64;
  std::normal_distribution<float> normal;
  std::vector<float> x(kRows * x_stride);
  for (auto& v : x)
    v = normal(rng);
  auto* d_x = Upload(x);
  struct Slot {
    std::vector<std::uint16_t> trellis, suh, svh, inner;
    std::uint16_t *d_trellis, *d_suh, *d_svh;
    float* d_y;
    std::size_t y_stride;
    DeviceLinear lin;
  };
  std::vector<Slot> slots(ns.size());
  std::size_t counters = 0;
  for (std::size_t s = 0; s < ns.size(); ++s) {
    Slot& sl = slots[s];
    const std::size_t n = ns[s];
    sl.trellis.resize((k / 16) * (n / 16) * rate.TileWords());
    for (auto& w : sl.trellis)
      w = static_cast<std::uint16_t>(rng());
    sl.suh = RandomScales(rng, k, 0x3800);
    sl.svh = RandomScales(rng, n, 0x2000);
    sl.inner.resize(k * n);
    DequantInner(sl.trellis.data(), k, n, rate, Codebook::kMul1,
                 sl.inner.data());
    sl.d_trellis = Upload(sl.trellis);
    sl.d_suh = Upload(sl.suh);
    sl.d_svh = Upload(sl.svh);
    sl.y_stride = n + 128;
    Check(hipMalloc(&sl.d_y, kRows * sl.y_stride * sizeof(float)));
    sl.lin = DeviceLinear{sl.d_trellis,
                          sl.d_suh,
                          sl.d_svh,
                          static_cast<std::uint32_t>(k),
                          static_cast<std::uint32_t>(n),
                          rate,
                          Codebook::kMul1};
    counters += n / 128;
  }
  gufo::hip::Exl3FastWorkspace ws;
  ws.input_elements = ns.size() * 8 * k;
  Check(hipMalloc(&ws.input, ws.input_elements * sizeof(std::uint16_t)));
  ws.counter_count = counters;
  Check(hipMalloc(&ws.counters, ws.counter_count * sizeof(std::uint32_t)));
  Check(hipMemset(ws.counters, 0, ws.counter_count * sizeof(std::uint32_t)));
  std::vector<gufo::hip::Exl3FastTarget> targets;
  for (Slot& sl : slots)
    targets.push_back({&sl.lin, sl.d_y, sl.y_stride});
  Expect(gufo::hip::Exl3FastGroupSupported(targets, ws),
         "group supported by the fast path");

  std::string name = "group k=" + std::to_string(k) + " n=";
  for (std::size_t n : ns)
    name += std::to_string(n) + "/";
  name += " K=" + std::to_string(bits);
  double worst = 0.0;
  for (std::size_t rows : {1, 3, 8, 17}) {
    for (Slot& sl : slots)
      Check(hipMemset(sl.d_y, 0xff, kRows * sl.y_stride * sizeof(float)));
    gufo::hip::LaunchExl3FastProjections(targets, d_x, x_stride, rows, ws,
                                         nullptr);
    Check(hipDeviceSynchronize());
    for (std::size_t s = 0; s < slots.size(); ++s) {
      const Slot& sl = slots[s];
      const std::size_t n = ns[s];
      const auto y = Download(sl.d_y, kRows * sl.y_stride);
      std::vector<double> want(rows * n);
      for (std::size_t r = 0; r < rows; ++r) {
        const auto row = ProjectRotated(
            KernelRotatedInput(&x[r * x_stride], sl.suh.data(), k), sl.inner,
            sl.svh, k, n);
        std::copy(row.begin(), row.end(), want.begin() + r * n);
      }
      worst = std::max(worst, RelL2(want, y, rows, n, sl.y_stride));
      bool untouched = true;
      for (std::size_t r = 0; r < kRows && untouched; ++r)
        for (std::size_t j = 0; j < sl.y_stride && untouched; ++j) {
          if (r < rows && j < n)
            continue;
          std::uint32_t b = 0;
          std::memcpy(&b, &y[r * sl.y_stride + j], 4);
          untouched = b == 0xffffffffU;
        }
      Expect(untouched, name + ": writes outside a slot window");
    }
    const auto c = Download(ws.counters, ws.counter_count);
    Expect(
        std::all_of(c.begin(), c.end(), [](std::uint32_t v) { return v == 0; }),
        name + ": counters not back to zero");
  }
  Expect(worst < 1e-5, name + " relative L2 " + std::to_string(worst));
  std::cout << name << ": relative L2 vs FP16-rounded input " << worst << "\n";
  for (Slot& sl : slots)
    for (void* ptr :
         {static_cast<void*>(sl.d_trellis), static_cast<void*>(sl.d_suh),
          static_cast<void*>(sl.d_svh), static_cast<void*>(sl.d_y)})
      Check(hipFree(ptr));
  for (void* ptr : {static_cast<void*>(d_x), static_cast<void*>(ws.input),
                    static_cast<void*>(ws.counters)})
    Check(hipFree(ptr));
}

// The fused MLP (gate/up epilogue writes the down input) against the same
// three projections with the SwiGLU kernel in between: bit-identical outputs.
void TestMlp(std::size_t hidden, std::size_t inter, int bits,
             std::mt19937& rng) {
  constexpr std::size_t kRows = 17;
  const BitRate rate{bits, false};
  auto make = [&](std::size_t k, std::size_t n, std::vector<void*>& owned) {
    std::vector<std::uint16_t> trellis((k / 16) * (n / 16) * rate.TileWords());
    for (auto& w : trellis)
      w = static_cast<std::uint16_t>(rng());
    auto* t = Upload(trellis);
    auto* su = Upload(RandomScales(rng, k, 0x3800));
    auto* sv = Upload(RandomScales(rng, n, 0x2000));
    owned.insert(owned.end(), {t, su, sv});
    return DeviceLinear{t,
                        su,
                        sv,
                        static_cast<std::uint32_t>(k),
                        static_cast<std::uint32_t>(n),
                        rate,
                        Codebook::kMul1};
  };
  std::vector<void*> owned;
  const DeviceLinear gate = make(hidden, inter, owned);
  const DeviceLinear up = make(hidden, inter, owned);
  const DeviceLinear down = make(inter, hidden, owned);
  std::normal_distribution<float> normal;
  std::vector<float> x(kRows * hidden);
  for (auto& v : x)
    v = normal(rng);
  auto* d_x = Upload(x);
  float *d_gate = nullptr, *d_up = nullptr, *d_act = nullptr;
  float *d_fused = nullptr, *d_plain = nullptr;
  for (float** p : {&d_gate, &d_up, &d_act})
    Check(hipMalloc(p, kRows * inter * sizeof(float)));
  for (float** p : {&d_fused, &d_plain})
    Check(hipMalloc(p, kRows * hidden * sizeof(float)));
  gufo::hip::Exl3FastWorkspace ws;
  ws.input_elements = 2 * 8 * std::max(hidden, inter);
  Check(hipMalloc(&ws.input, ws.input_elements * sizeof(std::uint16_t)));
  ws.counter_count = std::max(hidden, inter) / 128;
  Check(hipMalloc(&ws.counters, ws.counter_count * sizeof(std::uint32_t)));
  Check(hipMemset(ws.counters, 0, ws.counter_count * sizeof(std::uint32_t)));
  ws.down_input_elements = 8 * inter;
  Check(hipMalloc(&ws.down_input,
                  ws.down_input_elements * sizeof(std::uint16_t)));
  const gufo::hip::Exl3FastMlp mlp{&gate, &up,     &down, d_gate,
                                   d_up,  d_fused, hidden};
  Expect(gufo::hip::Exl3FastMlpSupported(mlp, ws), "fused MLP supported");
  const std::string name = "mlp hidden=" + std::to_string(hidden) +
                           " inter=" + std::to_string(inter) +
                           " K=" + std::to_string(bits);
  bool same = true;
  for (std::size_t rows : {1, 3, 8, 17}) {
    gufo::hip::LaunchExl3FastMlp(mlp, d_x, hidden, rows, ws, nullptr);
    const gufo::hip::Exl3FastTarget pair[] = {{&gate, d_gate, inter},
                                              {&up, d_up, inter}};
    gufo::hip::LaunchExl3FastProjections(pair, d_x, hidden, rows, ws, nullptr);
    gufo::hip::LaunchBatchedSwiGLUActivation(d_gate, d_up, d_act, nullptr,
                                             rows * inter, nullptr);
    gufo::hip::LaunchExl3FastProjection(down, d_act, inter, rows, d_plain,
                                        hidden, ws, nullptr);
    Check(hipDeviceSynchronize());
    const auto fused = Download(d_fused, rows * hidden);
    const auto plain = Download(d_plain, rows * hidden);
    const bool rows_same =
        std::memcmp(fused.data(), plain.data(), fused.size() * 4) == 0;
    Expect(rows_same, name + " rows=" + std::to_string(rows) +
                          ": fused MLP differs from the unfused launches");
    same = same && rows_same;
  }
  std::cout << name << ": fused "
            << (same ? "bit-identical to" : "DIFFERS from") << " unfused\n";
  for (void* p : owned)
    Check(hipFree(p));
  for (void* p : {static_cast<void*>(d_x), static_cast<void*>(d_gate),
                  static_cast<void*>(d_up), static_cast<void*>(d_act),
                  static_cast<void*>(d_fused), static_cast<void*>(d_plain),
                  static_cast<void*>(ws.input), static_cast<void*>(ws.counters),
                  static_cast<void*>(ws.down_input)})
    Check(hipFree(p));
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
  // q + k + v (four tiles per wave) and gate + up shapes at small k.
  TestGroup(512, {12288, 1024, 1024}, 5, rng);
  TestGroup(512, {4096, 4096}, 4, rng);
  TestGroup(1024, {2048, 1024}, 6, rng);
  TestMlp(512, 1024, 5, rng);
  TestMlp(1024, 2048, 4, rng);
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "qwen27b_exl3_kernel_test: all checks passed\n";
  return 0;
}
