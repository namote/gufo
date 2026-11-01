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
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "src/core/hip/detail/exl3_gemv.hpp"
#include "src/core/quant/exl3/format.hpp"
#include "src/models/qwen/hip/exl3_dispatch.hpp"
#include "src/models/qwen/hip/kernels/exl3_gemv.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
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
  // The unfused comparison launches gate and up as one group: counters for
  // both outputs.
  ws.counter_count = 2 * std::max(hidden, inter) / 128;
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

// The fused decode RMSNorm (normed row plus each consumer's rotated slab)
// followed by the pre-rotated group and MLP launches, against LaunchRMSNorm
// and the same launches rotating the normed row themselves: bits.
void TestNormFusion(std::mt19937& rng) {
  constexpr std::size_t kHidden = 5120;
  constexpr std::size_t kInter = 1024;
  const BitRate rate{5, false};
  std::vector<void*> owned;
  auto make = [&](std::size_t k, std::size_t n) {
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
  const DeviceLinear q = make(kHidden, 2048);
  const DeviceLinear k = make(kHidden, 1024);
  const DeviceLinear v = make(kHidden, 1024);
  const DeviceLinear gate = make(kHidden, kInter);
  const DeviceLinear up = make(kHidden, kInter);
  const DeviceLinear down = make(kInter, kHidden);
  std::normal_distribution<float> normal;
  std::vector<float> hidden(kHidden), norm_w(kHidden);
  for (auto& x : hidden)
    x = 3.0F * normal(rng);
  for (auto& x : norm_w)
    x = 1.0F + 0.1F * normal(rng);
  auto* d_hidden = Upload(hidden);
  auto* d_norm_w = Upload(norm_w);
  std::vector<float*> buffers;
  auto alloc = [&](std::size_t n) {
    float* p = nullptr;
    Check(hipMalloc(&p, n * sizeof(float)));
    buffers.push_back(p);
    return p;
  };
  float* normed_a = alloc(kHidden);
  float* normed_b = alloc(kHidden);
  float* ya[3] = {alloc(2048), alloc(1024), alloc(1024)};
  float* yb[3] = {alloc(2048), alloc(1024), alloc(1024)};
  float *g = alloc(kInter), *u = alloc(kInter), *act = alloc(kInter);
  float *out_a = alloc(kHidden), *out_b = alloc(kHidden);
  gufo::hip::Exl3FastWorkspace ws;
  ws.input_elements = 3 * 8 * kHidden;
  Check(hipMalloc(&ws.input, ws.input_elements * sizeof(std::uint16_t)));
  ws.counter_count = kHidden / 128;
  Check(hipMalloc(&ws.counters, ws.counter_count * sizeof(std::uint32_t)));
  Check(hipMemset(ws.counters, 0, ws.counter_count * sizeof(std::uint32_t)));
  ws.down_input_elements = 8 * kInter;
  Check(hipMalloc(&ws.down_input,
                  ws.down_input_elements * sizeof(std::uint16_t)));

  // q + k + v.
  const DeviceLinear* qkv[] = {&q, &k, &v};
  const gufo::hip::Exl3FastTarget ta[] = {
      {&q, ya[0], 2048}, {&k, ya[1], 1024}, {&v, ya[2], 1024}};
  const gufo::hip::Exl3FastTarget tb[] = {
      {&q, yb[0], 2048}, {&k, yb[1], 1024}, {&v, yb[2], 1024}};
  const bool fused = gufo::hip::LaunchExl3FastRmsNorm(
      d_hidden, d_norm_w, normed_a, kHidden, 1e-6F, qkv, ws, nullptr);
  Expect(fused, "fused norm applies to the q + k + v group");
  gufo::hip::LaunchExl3FastProjections(ta, normed_a, kHidden, 1, ws, nullptr,
                                       true);
  gufo::hip::LaunchRMSNorm(d_hidden, d_norm_w, normed_b, kHidden, 1e-6F,
                           nullptr);
  gufo::hip::LaunchExl3FastProjections(tb, normed_b, kHidden, 1, ws, nullptr);
  Check(hipDeviceSynchronize());
  bool same = Download(normed_a, kHidden) == Download(normed_b, kHidden);
  Expect(same, "fused norm row differs from LaunchRMSNorm");
  for (int i = 0; i < 3; ++i) {
    const std::size_t n = i == 0 ? 2048 : 1024;
    const bool slot_same = Download(ya[i], n) == Download(yb[i], n);
    Expect(slot_same, "pre-rotated group slot differs");
    same = same && slot_same;
  }

  // gate + up + down.
  const DeviceLinear* gu[] = {&gate, &up};
  Expect(gufo::hip::LaunchExl3FastRmsNorm(d_hidden, d_norm_w, normed_a, kHidden,
                                          1e-6F, gu, ws, nullptr),
         "fused norm applies to the MLP");
  const gufo::hip::Exl3FastMlp mlp_a{&gate, &up, &down, g, u, out_a, kHidden};
  gufo::hip::LaunchExl3FastMlp(mlp_a, normed_a, kHidden, 1, ws, nullptr, true);
  const gufo::hip::Exl3FastMlp mlp_b{&gate, &up, &down, g, u, out_b, kHidden};
  gufo::hip::LaunchExl3FastMlp(mlp_b, normed_b, kHidden, 1, ws, nullptr);
  Check(hipDeviceSynchronize());
  const bool mlp_same = Download(out_a, kHidden) == Download(out_b, kHidden);
  Expect(mlp_same, "pre-rotated MLP differs");
  same = same && mlp_same;
  std::cout << "norm fusion: " << (same ? "bit-identical to" : "DIFFERS from")
            << " the unfused launches\n";
  (void)act;
  for (void* p : owned)
    Check(hipFree(p));
  for (float* p : buffers)
    Check(hipFree(p));
  for (void* p : {static_cast<void*>(d_hidden), static_cast<void*>(d_norm_w),
                  static_cast<void*>(ws.input), static_cast<void*>(ws.counters),
                  static_cast<void*>(ws.down_input)})
    Check(hipFree(p));
}

// Every 16-bit state through the packed hash and the pair decode (lo = s, hi =
// a permutation of s) against the CPU codebook (DecodeState).
__global__ void DecodeAllStates(std::uint32_t* out) {
  const unsigned s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= 65536)
    return;
  const unsigned s2 = (s * 40503U + 12345U) & 0xffffU;
  // Garbage in the upper halves, as the window shift leaves it.
  const auto v = gufo::hip::exl3::DecodePairMul1(
      gufo::hip::exl3::HashMul1Packed(s | (s2 << 16)),
      gufo::hip::exl3::HashMul1Packed(s2 | (s << 16)));
  out[s] = __builtin_bit_cast(std::uint32_t, v);
}

void TestPackedHash() {
  std::uint32_t* d = nullptr;
  Check(hipMalloc(&d, 65536 * 4));
  hipLaunchKernelGGL(DecodeAllStates, dim3(256), dim3(256), 0, 0, d);
  const auto got = Download(d, 65536);
  std::size_t diff = 0;
  for (std::uint32_t s = 0; s < 65536; ++s) {
    const std::uint32_t s2 = (s * 40503U + 12345U) & 0xffffU;
    diff += (got[s] & 0xffffU) != DecodeState(s, Codebook::kMul1) ||
            (got[s] >> 16) != DecodeState(s2, Codebook::kMul1);
  }
  Expect(diff == 0,
         "packed hash decode differs in " + std::to_string(diff) + " states");
  std::cout << "packed hash: " << diff << " / 65536 states differ\n";
  Check(hipFree(d));
}

// Relative L2 error against FP64 of the production GGUF FP16 prefill GEMM
// (LaunchBatchedQuantGEMMFp16, Q8_0 weights, 256 output rows) on the FP16
// rounding of rows x k FP32 activations: the reference error of the WMMA FP16
// GEMM frame both formats share.
double GgufWmmaError(std::size_t k, const std::vector<std::uint16_t>& xh,
                     std::size_t rows) {
  constexpr std::size_t kRows = 256;
  std::mt19937 rng(static_cast<unsigned>(k * 7 + rows));
  std::vector<std::uint8_t> blocks(kRows * (k / 32) * 34);
  std::vector<double> w(kRows * k);
  for (std::size_t r = 0; r < kRows; ++r)
    for (std::size_t b = 0; b < k / 32; ++b) {
      std::uint8_t* block = &blocks[(r * (k / 32) + b) * 34];
      const std::uint16_t d = HalfFromDouble(
          0.002 + 0.001 * static_cast<double>(rng() % 1000) / 1000.0);
      std::memcpy(block, &d, 2);
      for (int j = 0; j < 32; ++j) {
        const auto q = static_cast<std::int8_t>(rng());
        block[2 + j] = static_cast<std::uint8_t>(q);
        // The kernel's FP16 weight: FP32 product, rounded to FP16 once.
        const float v =
            static_cast<float>(HalfToDouble(d)) * static_cast<float>(q);
        w[r * k + b * 32 + j] = HalfToDouble(HalfFromDouble(v));
      }
    }
  auto* d_w = Upload(blocks);
  auto* d_x = Upload(xh);
  float* d_y = nullptr;
  Check(hipMalloc(&d_y, rows * kRows * 4));
  gufo::hip::LaunchBatchedQuantGEMMFp16(gufo::core::GgmlType::kQ8_0, d_w, d_x,
                                        d_y, rows, kRows, k, nullptr);
  Check(hipDeviceSynchronize());
  const auto y = Download(d_y, rows * kRows);
  double num = 0.0, den = 0.0;
  for (std::size_t t = 0; t < rows; ++t)
    for (std::size_t r = 0; r < kRows; ++r) {
      double want = 0.0;
      for (std::size_t i = 0; i < k; ++i)
        want += HalfToDouble(xh[t * k + i]) * w[r * k + i];
      if (!std::isfinite(y[t * kRows + r]))
        num = std::numeric_limits<double>::infinity();
      num += (y[t * kRows + r] - want) * (y[t * kRows + r] - want);
      den += want * want;
    }
  for (void* p : {static_cast<void*>(d_w), static_cast<void*>(d_x),
                  static_cast<void*>(d_y)})
    Check(hipFree(p));
  return std::sqrt(num / den);
}

// X3 prefill (and X3b narrow) trellis WMMA GEMM: the row rotation against the
// FP32 contract (FP16 bytes), then store and residual GEMMs against FP64 fed
// that rotated input (relative L2 within (k / 16) * 2^-23, finite).
void TestPrefill(std::size_t k, std::size_t n, int bits, std::size_t rows,
                 std::mt19937& rng) {
  const BitRate rate{bits, false};
  std::vector<std::uint16_t> trellis((k / 16) * (n / 16) * rate.TileWords());
  for (auto& w : trellis)
    w = static_cast<std::uint16_t>(rng());
  const auto suh = RandomScales(rng, k, 0x3800);
  const auto svh = RandomScales(rng, n, 0x2000);
  std::normal_distribution<float> normal;
  std::vector<float> x(rows * k), y0(rows * n);
  for (auto& v : x)
    v = normal(rng);
  for (auto& v : y0)
    v = normal(rng);
  std::vector<std::uint16_t> inner(k * n);
  DequantInner(trellis.data(), k, n, rate, Codebook::kMul1, inner.data());
  std::vector<double> want(rows * n);
  std::size_t rot_diff = 0;
  auto* d_x = Upload(x);
  auto* d_suh = Upload(suh);
  std::uint16_t* d_rot = nullptr;
  Check(hipMalloc(&d_rot, rows * k * 2));
  gufo::hip::LaunchExl3RotateRows(d_x, k, rows, k, d_suh, d_rot, nullptr);
  Check(hipDeviceSynchronize());
  const auto rot_got = Download(d_rot, rows * k);
  for (std::size_t r = 0; r < rows; ++r) {
    const auto rot = KernelRotatedInput(&x[r * k], suh.data(), k);
    for (std::size_t i = 0; i < k; ++i)
      rot_diff += rot_got[r * k + i] != HalfFromDouble(rot[i]);
    const auto yr = ProjectRotated(rot, inner, svh, k, n);
    std::copy(yr.begin(), yr.end(), want.begin() + r * n);
  }
  auto* d_t = Upload(trellis);
  auto* d_svh = Upload(svh);
  const DeviceLinear lin{d_t,
                         d_suh,
                         d_svh,
                         static_cast<std::uint32_t>(k),
                         static_cast<std::uint32_t>(n),
                         rate,
                         Codebook::kMul1};
  const std::string name =
      "prefill k=" + std::to_string(k) + " n=" + std::to_string(n) +
      " K=" + std::to_string(bits) + " rows=" + std::to_string(rows);
  Expect(gufo::hip::Exl3PrefillSupported(lin), name + ": supported");
  auto* d_y = Upload(y0);
  gufo::hip::LaunchExl3PrefillGemm(lin, d_rot, d_y, rows, false, nullptr);
  Check(hipDeviceSynchronize());
  const auto y = Download(d_y, rows * n);
  // Residual form: y0 + the same projection.
  Check(hipMemcpy(d_y, y0.data(), y0.size() * 4, hipMemcpyHostToDevice));
  gufo::hip::LaunchExl3PrefillGemm(lin, d_rot, d_y, rows, true, nullptr);
  Check(hipDeviceSynchronize());
  const auto yres = Download(d_y, rows * n);
  double num = 0.0, den = 0.0, rnum = 0.0;
  bool finite = true;
  for (std::size_t i = 0; i < want.size(); ++i) {
    finite = finite && std::isfinite(y[i]) && std::isfinite(yres[i]);
    num += (y[i] - want[i]) * (y[i] - want[i]);
    den += want[i] * want[i];
    const double r = yres[i] - (static_cast<double>(y0[i]) + want[i]);
    rnum += r * r;
  }
  const double rel = std::sqrt(num / den), rel_res = std::sqrt(rnum / den);
  Expect(rot_diff == 0, name +
                            ": rotated rows differ from the FP32 contract in " +
                            std::to_string(rot_diff) + " values");
  Expect(finite, name + ": non-finite output");
  // The WMMA FP16 GEMM does not accumulate like FP32 FMAs: its error grows
  // linearly with k (x3u2: about 3.1e-9 * k, above even a K16-step
  // round-toward-zero emulation). Fixed bar: one FP32 ulp of relative error
  // per K16 step, (k / 16) * 2^-23, the bound of an accumulator that loses up
  // to one ulp per step. The production GGUF FP16 GEMM's error on the same k
  // and the same FP16 input rows (Q8_0 weights, its own tile) is printed as a
  // diagnostic only.
  const double bar = static_cast<double>(k) / 16.0 * std::ldexp(1.0, -23);
  const double gguf = GgufWmmaError(k, rot_got, rows);
  Expect(std::isfinite(gguf), name + ": GGUF diagnostic not finite");
  Expect(rel < bar, name + " relative L2 " + std::to_string(rel) + " (bar " +
                        std::to_string(bar) + ")");
  Expect(rel_res < bar,
         name + " residual relative L2 " + std::to_string(rel_res));
  std::cout << name << ": relative L2 " << rel << ", residual " << rel_res
            << " (bar " << bar << ", GGUF WMMA " << gguf << ", ratio "
            << rel / gguf << "), rotation diffs " << rot_diff << "\n";
  for (void* p : {static_cast<void*>(d_x), static_cast<void*>(d_suh),
                  static_cast<void*>(d_rot), static_cast<void*>(d_t),
                  static_cast<void*>(d_svh), static_cast<void*>(d_y)})
    Check(hipFree(p));
}

// Dispatch level (src/models/qwen/hip/exl3_dispatch.hpp) at 16 and 17 rows:
// decode/verify groups, MLPs and single projections stay on the GEMV at any
// row count (bit-identical to the direct GEMV launches); a prefill projection
// switches to the WMMA GEMM at kExl3WmmaMinRows (bit-identical to the direct
// rotation + WMMA launches).
void TestDispatch(std::mt19937& rng) {
  constexpr std::size_t kHidden = 1024, kInter = 2048, kRows = 17;
  const BitRate rate{5, false};
  std::vector<void*> owned;
  auto make = [&](std::size_t k, std::size_t n) {
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
  const DeviceLinear q = make(kHidden, 1024), kk = make(kHidden, 256),
                     v = make(kHidden, 256), gate = make(kHidden, kInter),
                     up = make(kHidden, kInter), down = make(kInter, kHidden);
  const auto ref = [](const DeviceLinear& lin) {
    gufo::models::QwenTensorRef r;
    r.data = lin.trellis;
    r.num_elements = static_cast<std::size_t>(lin.k) * lin.n;
    r.exl3 = &lin;
    return r;
  };
  const auto rq = ref(q), rk = ref(kk), rv = ref(v), rg = ref(gate),
             ru = ref(up), rd = ref(down);
  std::normal_distribution<float> normal;
  std::vector<float> x(kRows * kHidden);
  for (auto& e : x)
    e = normal(rng);
  auto* d_x = Upload(x);
  gufo::quant::exl3::ReferenceScratch scratch;
  scratch.rotated_elements = kRows * kInter;
  scratch.weight_elements = 128 * kInter;
  Check(hipMalloc(&scratch.rotated, scratch.rotated_elements * 4));
  Check(hipMalloc(&scratch.weight, scratch.weight_elements * 4));
  // One 8-row slab per matrix of the widest group (q + k + v).
  scratch.fast_input_elements = 3 * 8 * kInter;
  Check(hipMalloc(&scratch.fast_input, scratch.fast_input_elements * 2));
  scratch.fast_counter_count = 2 * kInter / 128;
  Check(hipMalloc(&scratch.fast_counters, scratch.fast_counter_count * 4));
  Check(hipMemset(scratch.fast_counters, 0, scratch.fast_counter_count * 4));
  scratch.fast_down_input_elements = 8 * kInter;
  Check(hipMalloc(&scratch.fast_down_input,
                  scratch.fast_down_input_elements * 2));
  scratch.fast_prefill_input_elements = kRows * kInter;
  Check(hipMalloc(&scratch.fast_prefill_input,
                  scratch.fast_prefill_input_elements * 2));
  const gufo::hip::Exl3FastWorkspace ws{
      scratch.fast_input,      scratch.fast_input_elements,
      scratch.fast_counters,   scratch.fast_counter_count,
      scratch.fast_down_input, scratch.fast_down_input_elements};
  std::vector<float*> bufs;
  auto alloc = [&](std::size_t n) {
    float* p = nullptr;
    Check(hipMalloc(&p, n * 4));
    bufs.push_back(p);
    return p;
  };
  float* a[3] = {alloc(kRows * 1024), alloc(kRows * 256), alloc(kRows * 256)};
  float* b[3] = {alloc(kRows * 1024), alloc(kRows * 256), alloc(kRows * 256)};
  float *g = alloc(kRows * kInter), *u = alloc(kRows * kInter),
        *act = alloc(kRows * kInter), *out_a = alloc(kRows * kHidden),
        *out_b = alloc(kRows * kHidden);
  for (std::size_t rows : {16, 17}) {
    const std::string tag = "dispatch rows=" + std::to_string(rows);
    // Decode/verify group vs the direct GEMV group.
    const gufo::hip::Exl3GroupOutput group[] = {
        {&rq, a[0]}, {&rk, a[1]}, {&rv, a[2]}};
    gufo::hip::LaunchExl3Projections(group, d_x, rows, &scratch, nullptr);
    const gufo::hip::Exl3FastTarget direct[] = {
        {&q, b[0], 1024}, {&kk, b[1], 256}, {&v, b[2], 256}};
    gufo::hip::LaunchExl3FastProjections(direct, d_x, kHidden, rows, ws,
                                         nullptr);
    Check(hipDeviceSynchronize());
    bool same = true;
    for (int i = 0; i < 3; ++i) {
      const std::size_t n = i == 0 ? 1024 : 256;
      same = same && Download(a[i], rows * n) == Download(b[i], rows * n);
    }
    Expect(same, tag + ": decode/verify group left the GEMV");
    // Decode/verify MLP vs the direct fused GEMV MLP.
    gufo::hip::LaunchExl3SwiGLUMlp(rg, ru, rd, d_x, rows, g, u, act, out_a,
                                   &scratch, nullptr);
    const gufo::hip::Exl3FastMlp mlp{&gate, &up, &down, g, u, out_b, kHidden};
    gufo::hip::LaunchExl3FastMlp(mlp, d_x, kHidden, rows, ws, nullptr);
    Check(hipDeviceSynchronize());
    Expect(Download(out_a, rows * kHidden) == Download(out_b, rows * kHidden),
           tag + ": decode/verify MLP left the fused GEMV");
    // Single decode/verify projection vs the direct GEMV.
    gufo::hip::LaunchExl3Projection(rq, d_x, rows, a[0], &scratch, nullptr);
    gufo::hip::LaunchExl3FastProjection(q, d_x, kHidden, rows, b[0], 1024, ws,
                                        nullptr);
    Check(hipDeviceSynchronize());
    Expect(Download(a[0], rows * 1024) == Download(b[0], rows * 1024),
           tag + ": decode/verify projection left the GEMV");
    // Prefill projection: GEMV below kExl3WmmaMinRows, WMMA from it.
    gufo::hip::LaunchExl3Projection(rq, d_x, rows, a[0], &scratch, nullptr, 0,
                                    0, gufo::hip::Exl3Pass::kPrefill);
    if (rows >= gufo::hip::kExl3WmmaMinRows) {
      std::uint16_t* rot = nullptr;
      Check(hipMalloc(&rot, rows * kHidden * 2));
      gufo::hip::LaunchExl3RotateRows(d_x, kHidden, rows, kHidden, q.suh, rot,
                                      nullptr);
      gufo::hip::LaunchExl3PrefillGemm(q, rot, b[0], rows, false, nullptr);
      Check(hipDeviceSynchronize());
      Check(hipFree(rot));
    } else {
      gufo::hip::LaunchExl3FastProjection(q, d_x, kHidden, rows, b[0], 1024, ws,
                                          nullptr);
      Check(hipDeviceSynchronize());
    }
    Expect(Download(a[0], rows * 1024) == Download(b[0], rows * 1024),
           tag + ": prefill projection took the wrong path");
    std::cout << tag << ": decode/verify on the GEMV, prefill on the "
              << (rows >= gufo::hip::kExl3WmmaMinRows ? "WMMA" : "GEMV")
              << " path\n";
  }
  for (void* p : owned)
    Check(hipFree(p));
  for (float* p : bufs)
    Check(hipFree(p));
  for (void* p : {static_cast<void*>(d_x), static_cast<void*>(scratch.rotated),
                  static_cast<void*>(scratch.weight),
                  static_cast<void*>(scratch.fast_input),
                  static_cast<void*>(scratch.fast_counters),
                  static_cast<void*>(scratch.fast_down_input),
                  static_cast<void*>(scratch.fast_prefill_input)})
    Check(hipFree(p));
}

// Every K's lane decode (DecodeLaneMul1, the GEMV producer; K7 through the
// shared 65-bit lane window) against the CPU dequant, FP16 bits.
template<int K>
__global__ void DecodeTilesKernel(const std::uint16_t* trellis, unsigned,
                                  unsigned n, std::uint16_t* out) {
  const unsigned lane = threadIdx.x;
  const unsigned tile = blockIdx.x, n_tiles = n / 16;
  const unsigned kt = tile / n_tiles, nt = tile % n_tiles;
  const auto lw = gufo::hip::exl3::MakeLaneWindow<K>(lane);
  gufo::hip::exl3::Half2 w[4];
  gufo::hip::exl3::DecodeLaneMul1<K>(
      reinterpret_cast<const std::uint32_t*>(trellis) +
          std::size_t(tile) * 8 * K,
      lw, w);
  const unsigned r0 = (lane % 4) * 2, c0 = lane / 4;
  for (int q = 0; q < 4; ++q) {
    const unsigned col = nt * 16 + c0 + 8 * (q / 2);
    const unsigned row = kt * 16 + r0 + 8 * (q % 2);
    // Split the pair as a dword: hipcc 7.2 stores the low half for both
    // elements of __builtin_bit_cast(std::uint16_t, w[q].y) (true16).
    const auto pair = __builtin_bit_cast(std::uint32_t, w[q]);
    out[std::size_t(row) * n + col] = static_cast<std::uint16_t>(pair);
    out[std::size_t(row + 1) * n + col] =
        static_cast<std::uint16_t>(pair >> 16);
  }
}

template<int K>
std::size_t DecodeTilesDiff(std::mt19937& rng) {
  constexpr unsigned k = 256, n = 512;
  const BitRate rate{K, false};
  std::vector<std::uint16_t> trellis((k / 16) * (n / 16) * rate.TileWords());
  for (auto& w : trellis)
    w = static_cast<std::uint16_t>(rng());
  std::vector<std::uint16_t> want(k * n);
  DequantInner(trellis.data(), k, n, rate, Codebook::kMul1, want.data());
  auto* d_t = Upload(trellis);
  std::uint16_t* d_out = nullptr;
  Check(hipMalloc(&d_out, k * n * 2));
  hipLaunchKernelGGL(DecodeTilesKernel<K>, dim3((k / 16) * (n / 16)), dim3(32),
                     0, 0, d_t, k, n, d_out);
  Check(hipGetLastError());
  Check(hipDeviceSynchronize());
  const auto got = Download(d_out, k * n);
  std::size_t diff = 0;
  for (std::size_t i = 0; i < got.size(); ++i)
    diff += got[i] != want[i];
  Check(hipFree(d_t));
  Check(hipFree(d_out));
  return diff;
}

void TestDecodeTiles(std::mt19937& rng) {
  const std::size_t diffs[] = {DecodeTilesDiff<1>(rng), DecodeTilesDiff<2>(rng),
                               DecodeTilesDiff<3>(rng), DecodeTilesDiff<4>(rng),
                               DecodeTilesDiff<5>(rng), DecodeTilesDiff<6>(rng),
                               DecodeTilesDiff<7>(rng)};
  for (int b = 0; b < 7; ++b)
    Expect(diffs[b] == 0, "lane decode K" + std::to_string(b + 1) + ": " +
                              std::to_string(diffs[b]) + " FP16 values differ");
  std::cout << "lane decode K1..K7 vs CPU: ";
  for (int b = 0; b < 7; ++b)
    std::cout << diffs[b] << (b < 6 ? "/" : " values differ\n");
}

// Grouped rotation (gathered rows) and grouped GEMM over expert-like buckets
// with ragged rows: every problem against its FP64 reference at the WMMA
// bound, finite, and rotated rows against the FP32 contract. With residual
// the outputs start from a known prior; the zero-row problem's output must
// stay untouched (sentinel). max_rows picks the tile (<= 64: narrow).
void TestGrouped(std::mt19937& rng, const std::size_t (&rows)[4],
                 std::size_t max_rows, bool residual) {
  constexpr std::size_t k = 2560, n = 640, source_rows = 400;
  const BitRate rate{5, false};
  constexpr std::size_t count = 4;
  constexpr std::uint32_t kSentinel = 0x7fc0beefU;
  std::normal_distribution<float> normal;
  std::vector<float> x(source_rows * k);
  for (auto& v : x)
    v = normal(rng);
  auto* d_x = Upload(x);
  std::vector<void*> owned;
  std::vector<gufo::hip::Exl3RotateProblem> rot(count);
  std::vector<gufo::hip::Exl3GemmProblem> gemm(count);
  struct Host {
    std::vector<std::uint16_t> trellis, suh, svh, inner;
    std::vector<std::uint32_t> index;
    std::vector<float> prior;
  };
  std::vector<Host> host(count);
  for (std::size_t p = 0; p < count; ++p) {
    Host& h = host[p];
    h.trellis.resize((k / 16) * (n / 16) * rate.TileWords());
    for (auto& w : h.trellis)
      w = static_cast<std::uint16_t>(rng());
    h.suh = RandomScales(rng, k, 0x3800);
    h.svh = RandomScales(rng, n, 0x2000);
    h.inner.resize(k * n);
    DequantInner(h.trellis.data(), k, n, rate, Codebook::kMul1, h.inner.data());
    for (std::size_t r = 0; r < rows[p]; ++r)
      h.index.push_back(static_cast<std::uint32_t>(rng() % source_rows));
    if (h.index.empty())
      h.index.push_back(0);
    auto* d_t = Upload(h.trellis);
    auto* d_suh = Upload(h.suh);
    auto* d_svh = Upload(h.svh);
    auto* d_index = Upload(h.index);
    std::uint16_t* d_slab = nullptr;
    float* d_y = nullptr;
    Check(hipMalloc(&d_slab, std::max<std::size_t>(rows[p], 1) * k * 2));
    Check(hipMalloc(&d_y, std::max<std::size_t>(rows[p], 1) * n * 4));
    {
      // Prior: a sentinel for the zero-row problem, values for residual.
      std::vector<float> prior(std::max<std::size_t>(rows[p], 1) * n);
      for (std::size_t i = 0; i < prior.size(); ++i)
        prior[i] = rows[p] == 0 ? std::bit_cast<float>(kSentinel)
                                : (static_cast<float>(i % 7) - 3.0F) * 0x1p-8F;
      h.prior = prior;
      Check(hipMemcpy(d_y, prior.data(), prior.size() * 4,
                      hipMemcpyHostToDevice));
    }
    owned.insert(owned.end(), {d_t, d_suh, d_svh, d_index, d_slab, d_y});
    rot[p] = {
        d_x, k, d_index, d_suh, d_slab, static_cast<std::uint32_t>(rows[p]), 0};
    gemm[p] = {d_t, d_svh, d_slab, d_y, static_cast<std::uint32_t>(rows[p]), 0};
  }
  auto* d_rot = Upload(rot);
  auto* d_gemm = Upload(gemm);
  gufo::hip::LaunchExl3RotateRowsGrouped(d_rot, count, max_rows, k, nullptr);
  gufo::hip::LaunchExl3PrefillGemmGrouped(d_gemm, count, max_rows, k, n, rate,
                                          residual, nullptr);
  Check(hipDeviceSynchronize());
  const double bar = static_cast<double>(k) / 16.0 * std::ldexp(1.0, -23);
  for (std::size_t p = 0; p < count; ++p) {
    const Host& h = host[p];
    if (rows[p] == 0) {
      const auto y = Download(gemm[p].y, n);
      bool untouched = true;
      for (float v : y)
        untouched = untouched && std::bit_cast<std::uint32_t>(v) == kSentinel;
      Expect(untouched, "grouped zero-row problem: output written");
      continue;
    }
    const auto slab =
        Download(static_cast<const std::uint16_t*>(rot[p].out), rows[p] * k);
    const auto y = Download(gemm[p].y, rows[p] * n);
    std::size_t rot_diff = 0;
    double num = 0.0, den = 0.0;
    bool finite = true;
    for (std::size_t r = 0; r < rows[p]; ++r) {
      const auto rr = KernelRotatedInput(&x[h.index[r] * k], h.suh.data(), k);
      for (std::size_t i = 0; i < k; ++i)
        rot_diff += slab[r * k + i] != HalfFromDouble(rr[i]);
      const auto want = ProjectRotated(rr, h.inner, h.svh, k, n);
      for (std::size_t j = 0; j < n; ++j) {
        finite = finite && std::isfinite(y[r * n + j]);
        // Residual: compare the added part (the prior is exact in FP32
        // only up to the final rounding, which the bar covers).
        const double got =
            static_cast<double>(y[r * n + j]) -
            (residual ? static_cast<double>(h.prior[r * n + j]) : 0.0);
        num += (got - want[j]) * (got - want[j]);
        den += want[j] * want[j];
      }
    }
    const double rel = std::sqrt(num / den);
    const std::string name =
        std::string("grouped ") + (max_rows <= 64 ? "narrow" : "wide") +
        (residual ? " residual" : "") + " problem " + std::to_string(p) +
        " rows=" + std::to_string(rows[p]);
    Expect(rot_diff == 0, name + ": gathered rotation differs");
    Expect(finite, name + ": non-finite output");
    Expect(rel < bar, name + " relative L2 " + std::to_string(rel));
    std::cout << name << ": relative L2 " << rel << " (bar " << bar
              << "), rotation diffs " << rot_diff << "\n";
  }
  for (void* p : owned)
    Check(hipFree(p));
  for (void* p : {static_cast<void*>(d_x), static_cast<void*>(d_rot),
                  static_cast<void*>(d_gemm)})
    Check(hipFree(p));
}

// Bucketed raw inner products (Flash-Next routed prefill): an expert bank
// with ragged per-expert rows split into 16/32/64-row tiles. Every
// partition gives bit-identical values; each against the FP64 inner product
// at the WMMA bound; rows outside every bucket and descriptors past the
// device count stay untouched; a bank entry of the wrong rate writes NaN.
void TestBuckets(std::size_t k, std::size_t n, int bits, std::mt19937& rng) {
  constexpr std::size_t kExperts = 4;
  const BitRate rate{bits, false};
  const std::size_t expert_rows[kExperts] = {1, 37, 0, 70};
  std::size_t total = 1;  // row 0 belongs to no bucket (sentinel)
  for (std::size_t e : expert_rows)
    total += e;
  std::vector<std::vector<std::uint16_t>> inner(kExperts);
  std::vector<DeviceLinear> bank(kExperts + 1);
  std::vector<void*> owned;
  for (std::size_t e = 0; e <= kExperts; ++e) {
    // Entry kExperts: another rate, a contract violation.
    const BitRate r = e < kExperts ? rate : BitRate{bits == 6 ? 5 : 6, false};
    std::vector<std::uint16_t> trellis((k / 16) * (n / 16) * r.TileWords());
    for (auto& w : trellis)
      w = static_cast<std::uint16_t>(rng());
    if (e < kExperts) {
      inner[e].resize(k * n);
      DequantInner(trellis.data(), k, n, rate, Codebook::kMul1,
                   inner[e].data());
    }
    auto* d_t = Upload(trellis);
    owned.push_back(d_t);
    bank[e].trellis = d_t;
    bank[e].k = static_cast<std::uint32_t>(k);
    bank[e].n = static_cast<std::uint32_t>(n);
    bank[e].rate = r;
    bank[e].codebook = Codebook::kMul1;
  }
  std::normal_distribution<double> normal;
  std::vector<std::uint16_t> input(total * k);
  for (auto& v : input)
    v = HalfFromDouble(normal(rng));
  auto* d_bank = Upload(bank);
  auto* d_input = Upload(input);
  constexpr std::uint32_t kSentinel = 0x7fc0beefU;
  const std::vector<float> sentinel(total * n, std::bit_cast<float>(kSentinel));
  float* d_inner = nullptr;
  Check(hipMalloc(&d_inner, total * n * 4));
  const double bar = static_cast<double>(k) / 16.0 * std::ldexp(1.0, -23);
  std::vector<float> first_result;
  // 128: 64-row tiles plus the hot-expert class (the 70-row expert in one
  // 128-token bucket).
  for (std::uint32_t partition : {16U, 32U, 64U, 7U, 128U}) {
    const std::uint32_t tile_rows = std::min(partition, 64U);
    std::vector<gufo::hip::Exl3PrefillBucket> buckets;
    std::uint32_t row = 1;
    for (std::uint32_t e = 0; e < kExperts; ++e) {
      const std::uint32_t size =
          partition == 128 && expert_rows[e] <= 64 ? 64 : partition;
      for (std::uint32_t r = 0; r < expert_rows[e]; r += size)
        buckets.push_back(
            {e, std::min<std::uint32_t>(size, expert_rows[e] - r), row + r});
      row += static_cast<std::uint32_t>(expert_rows[e]);
    }
    const auto active = static_cast<std::uint32_t>(buckets.size());
    // Past the device count: would overwrite row 0 if read.
    buckets.push_back({0, 1, 0});
    buckets.push_back({1, 1, 0});
    auto* d_buckets = Upload(buckets);
    auto* d_count = Upload(std::vector<std::uint32_t>{active});
    Check(hipMemcpy(d_inner, sentinel.data(), sentinel.size() * 4,
                    hipMemcpyHostToDevice));
    gufo::hip::LaunchExl3PrefillBucketInner(
        d_bank, d_buckets, d_count, static_cast<std::uint32_t>(buckets.size()),
        tile_rows, d_input, d_inner, static_cast<std::uint32_t>(k),
        static_cast<std::uint32_t>(n), rate, nullptr);
    Check(hipDeviceSynchronize());
    const auto got = Download(d_inner, total * n);
    const std::string name =
        "bucket K" + std::to_string(bits) + " k=" + std::to_string(k) +
        " n=" + std::to_string(n) + " partition=" + std::to_string(partition);
    bool untouched = true;
    for (std::size_t j = 0; j < n; ++j)
      untouched =
          untouched && std::bit_cast<std::uint32_t>(got[j]) == kSentinel;
    Expect(untouched, name + ": row outside every bucket written");
    double num = 0.0, den = 0.0;
    bool finite = true;
    row = 1;
    for (std::size_t e = 0; e < kExperts; ++e) {
      for (std::size_t r = 0; r < expert_rows[e]; ++r, ++row) {
        std::vector<double> want(n, 0.0);
        for (std::size_t i = 0; i < k; ++i) {
          const double xi = HalfToDouble(input[row * k + i]);
          for (std::size_t j = 0; j < n; ++j)
            want[j] += xi * HalfToDouble(inner[e][i * n + j]);
        }
        for (std::size_t j = 0; j < n; ++j) {
          const double g = got[row * n + j];
          finite = finite && std::isfinite(g);
          num += (g - want[j]) * (g - want[j]);
          den += want[j] * want[j];
        }
      }
    }
    const double rel = std::sqrt(num / den);
    Expect(finite, name + ": non-finite output");
    Expect(rel < bar, name + " relative L2 " + std::to_string(rel));
    std::size_t partition_diffs = 0;
    if (first_result.empty())
      first_result = got;
    else
      for (std::size_t i = 0; i < got.size(); ++i)
        partition_diffs += std::bit_cast<std::uint32_t>(got[i]) !=
                           std::bit_cast<std::uint32_t>(first_result[i]);
    Expect(partition_diffs == 0,
           name + ": " + std::to_string(partition_diffs) +
               " values differ from the 16-row partition");
    std::cout << name << ": relative L2 " << rel << " (bar " << bar
              << "), bits differing from 16-row partition " << partition_diffs
              << "\n";
    Check(hipFree(d_buckets));
    Check(hipFree(d_count));
  }
  {
    // A K6 bank entry under the K5 entry point poisons its rows only.
    std::vector<gufo::hip::Exl3PrefillBucket> buckets{
        {static_cast<std::uint32_t>(kExperts), 3, 1}};
    auto* d_buckets = Upload(buckets);
    auto* d_count = Upload(std::vector<std::uint32_t>{1});
    Check(hipMemcpy(d_inner, sentinel.data(), sentinel.size() * 4,
                    hipMemcpyHostToDevice));
    gufo::hip::LaunchExl3PrefillBucketInner(
        d_bank, d_buckets, d_count, 1, 16, d_input, d_inner,
        static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(n), rate,
        nullptr);
    Check(hipDeviceSynchronize());
    const auto got = Download(d_inner, total * n);
    bool poisoned = true, rest = true;
    for (std::size_t i = 0; i < got.size(); ++i) {
      const bool in = i >= n && i < 4 * n;
      if (in)
        poisoned = poisoned && std::isnan(got[i]) &&
                   std::bit_cast<std::uint32_t>(got[i]) != kSentinel;
      else
        rest = rest && std::bit_cast<std::uint32_t>(got[i]) == kSentinel;
    }
    Expect(poisoned && rest, "bucket rate mismatch: rows not poisoned alone");
    Check(hipFree(d_buckets));
    Check(hipFree(d_count));
  }
  for (void* p : owned)
    Check(hipFree(p));
  Check(hipFree(d_bank));
  Check(hipFree(d_input));
  Check(hipFree(d_inner));
}

// A subset of a grouped launch (Flash-Next MTP cache-only catch-up: k and v
// without q) with shape_basis_n = the full group's output width reproduces
// the full launch bit for bit; without it the geometry, and so the
// reduction, may change (reported, not required).
void TestShapeBasis(std::size_t q_n, std::mt19937& rng) {
  constexpr std::size_t k = 2560, kv_n = 512;
  const BitRate rate{7, false};
  const std::size_t ns[3] = {q_n, kv_n, kv_n};
  std::normal_distribution<float> normal;
  std::vector<void*> owned;
  DeviceLinear lin[3];
  float* full_y[3];
  float* sub_y[3];
  for (int s = 0; s < 3; ++s) {
    std::vector<std::uint16_t> trellis((k / 16) * (ns[s] / 16) *
                                       rate.TileWords());
    for (auto& w : trellis)
      w = static_cast<std::uint16_t>(rng());
    auto* d_t = Upload(trellis);
    auto* d_suh = Upload(RandomScales(rng, k, 0x3800));
    auto* d_svh = Upload(RandomScales(rng, ns[s], 0x2000));
    // Up to 9 rows (two passes); the workspace slabs stay 8 rows.
    Check(hipMalloc(&full_y[s], 9 * ns[s] * 4));
    Check(hipMalloc(&sub_y[s], 9 * ns[s] * 4));
    owned.insert(owned.end(), {d_t, d_suh, d_svh, full_y[s], sub_y[s]});
    lin[s] = DeviceLinear{d_t,
                          d_suh,
                          d_svh,
                          static_cast<std::uint32_t>(k),
                          static_cast<std::uint32_t>(ns[s]),
                          rate,
                          Codebook::kMul1};
  }
  std::vector<float> x(9 * k);
  for (auto& v : x)
    v = normal(rng);
  auto* d_x = Upload(x);
  gufo::hip::Exl3FastWorkspace ws;
  ws.input_elements = 3 * 8 * k;
  Check(hipMalloc(&ws.input, ws.input_elements * sizeof(std::uint16_t)));
  ws.counter_count = (q_n + 2 * kv_n) / 128;
  Check(hipMalloc(&ws.counters, ws.counter_count * sizeof(std::uint32_t)));
  Check(hipMemset(ws.counters, 0, ws.counter_count * sizeof(std::uint32_t)));
  const std::size_t basis = q_n + 2 * kv_n;
  for (std::size_t rows : {1, 3, 8, 9}) {
    const gufo::hip::Exl3FastTarget full[3] = {{&lin[0], full_y[0], q_n},
                                               {&lin[1], full_y[1], kv_n},
                                               {&lin[2], full_y[2], kv_n}};
    const gufo::hip::Exl3FastTarget sub[2] = {{&lin[1], sub_y[1], kv_n},
                                              {&lin[2], sub_y[2], kv_n}};
    gufo::hip::LaunchExl3FastProjections(full, d_x, k, rows, ws, nullptr);
    gufo::hip::LaunchExl3FastProjections(sub, d_x, k, rows, ws, nullptr, false,
                                         basis);
    Check(hipDeviceSynchronize());
    std::size_t diffs = 0;
    for (int s = 1; s < 3; ++s) {
      const auto a = Download(full_y[s], rows * kv_n);
      const auto b = Download(sub_y[s], rows * kv_n);
      for (std::size_t i = 0; i < a.size(); ++i)
        diffs += std::bit_cast<std::uint32_t>(a[i]) !=
                 std::bit_cast<std::uint32_t>(b[i]);
    }
    // Control: the subset with its own geometry.
    gufo::hip::LaunchExl3FastProjections(sub, d_x, k, rows, ws, nullptr);
    Check(hipDeviceSynchronize());
    std::size_t own = 0;
    for (int s = 1; s < 3; ++s) {
      const auto a = Download(full_y[s], rows * kv_n);
      const auto b = Download(sub_y[s], rows * kv_n);
      for (std::size_t i = 0; i < a.size(); ++i)
        own += std::bit_cast<std::uint32_t>(a[i]) !=
               std::bit_cast<std::uint32_t>(b[i]);
    }
    const std::string name = "shape basis q=" + std::to_string(q_n) +
                             " k/v=" + std::to_string(kv_n) +
                             " rows=" + std::to_string(rows);
    Expect(diffs == 0, name + ": " + std::to_string(diffs) +
                           " k/v outputs differ from the full group");
    std::cout << name << ": k/v bits differing from the full group " << diffs
              << " (own geometry: " << own << ")\n";
  }
  for (void* p : owned)
    Check(hipFree(p));
  Check(hipFree(d_x));
  Check(hipFree(ws.input));
  Check(hipFree(ws.counters));
}

}  // namespace

int main() {
  // Flush every line, so a later abort keeps the earlier results.
  std::cout << std::unitbuf;
  std::mt19937 rng(20261030);
  // Narrow and wide outputs cover the 16-, 8- and 4-wave launch shapes.
  const Problem problems[] = {{1024, 2048, 5},
                              {6144, 1024, 5},
                              {512, 16384, 6},
                              {2048, 1024, 4},
                              {384, 256, 3},
                              {256, 512, 2},
                              {256, 384, 1},
                              // Flash-Next: K7 trunk, K5 experts (640 wide).
                              {2560, 512, 7},
                              {2560, 640, 5},
                              {640, 2560, 5},
                              {6144, 2560, 7}};
  for (const Problem& p : problems)
    TestProblem(p, rng);
  // q + k + v (four tiles per wave) and gate + up shapes at small k.
  TestGroup(512, {12288, 1024, 1024}, 5, rng);
  TestGroup(512, {4096, 4096}, 4, rng);
  TestGroup(1024, {2048, 1024}, 6, rng);
  TestMlp(512, 1024, 5, rng);
  TestMlp(1024, 2048, 4, rng);
  TestNormFusion(rng);
  TestPackedHash();
  // X3b narrow tile (17..64 rows) and the X3 prefill tile, k up to 17408.
  TestPrefill(1024, 512, 5, 17, rng);
  TestPrefill(1024, 512, 5, 64, rng);
  TestPrefill(1024, 1024, 5, 300, rng);
  TestPrefill(512, 256, 4, 65, rng);
  TestPrefill(17408, 256, 5, 33, rng);
  TestPrefill(5120, 256, 6, 520, rng);
  // Flash-Next shapes: K7 trunk (prefill and narrow), 640-wide experts
  // (the 128 x 256 tile), 640 -> 2560 down.
  TestPrefill(2560, 512, 7, 300, rng);
  TestPrefill(2560, 512, 7, 33, rng);
  TestPrefill(2560, 640, 5, 300, rng);
  TestPrefill(640, 2560, 5, 65, rng);
  TestDecodeTiles(rng);
  TestGrouped(rng, {0, 7, 33, 300}, 300, false);
  TestGrouped(rng, {5, 0, 64, 1}, 64, true);
  TestBuckets(2560, 640, 5, rng);
  TestBuckets(640, 2560, 5, rng);
  // Mixed-rate packs (4.05 / Kyojin): K4 and K7 expert banks.
  TestBuckets(2560, 640, 4, rng);
  TestBuckets(640, 2560, 7, rng);
  // Flash-Next attention q (production geometry 4 x 4 for q + k + v) and a
  // smaller q (2 x 4); k + v alone would take 2 x 16.
  TestShapeBasis(12288, rng);
  TestShapeBasis(4096, rng);
  TestDispatch(rng);
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "qwen27b_exl3_kernel_test: all checks passed\n";
  return 0;
}
