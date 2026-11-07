// Flash-Next EXL3 dense decode GEMV (B5 J1): the pre-B5 kernel (one K tile
// per step, every load waited at its use; the verbatim copy in
// tests/models/qwen38_flash_next/exl3_gemv_reference.hpp) against the
// production kernel, which keeps the next K tile group in flight. Real decode
// shapes and rates of the c03 pack: GDN qkv+z K5, ssm_out / attention out K6,
// q+k+v K6, shared MLP K6 (gate/up SwiGLU then down) and a head-shaped K8
// launch. Rows R = 1, 2, 4, 8 (AR and MTP verify widths). Weights rotate over
// enough copies (>= 256 MB) to miss the 32 MB last-level cache, each variant
// captured in one graph of kCalls calls; µs/call is the best of kReps
// replays and GB/s counts trellis bytes. Output bytes of both kernels must be
// equal on the first copy.
//
// Build (CMake, links the production library):
//   cmake --build --preset gpu-test --target qwen38_flash_next_dense_gemv_bench
// Run:
//   build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_dense_gemv_bench
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "src/core/quant/exl3/gemv.hpp"
#include "tests/models/qwen38_flash_next/exl3_gemv_reference.hpp"

namespace ex = gufo::quant::exl3;
namespace fast = gufo::hip;
namespace ref = gufo::tests::exl3_gemv_reference;

namespace {

constexpr int kCalls = 48;
constexpr int kReps = 10;
constexpr std::size_t kColdBytes = std::size_t{256} << 20;

void Ck(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}

std::uint32_t Next(std::uint32_t* s) {
  *s ^= *s << 13;
  *s ^= *s >> 17;
  *s ^= *s << 5;
  return *s;
}

template<class T>
T* Upload(const std::vector<T>& h) {
  T* d = nullptr;
  Ck(hipMalloc(&d, h.size() * sizeof(T)), "malloc");
  Ck(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice),
     "upload");
  return d;
}

std::uint16_t Half(float v) {
  _Float16 h = static_cast<_Float16>(v);
  std::uint16_t bits;
  std::memcpy(&bits, &h, 2);
  return bits;
}

// One linear: random trellis words (every state valid), suh/svh near +-1.
ex::DeviceLinear Linear(unsigned k, unsigned n, int bits, std::uint32_t seed) {
  const ex::BitRate rate{bits, false};
  std::vector<std::uint16_t> trellis(std::size_t{k / 16} * (n / 16) *
                                     rate.TileWords());
  for (auto& w : trellis)
    w = static_cast<std::uint16_t>(Next(&seed));
  std::vector<std::uint16_t> suh(k), svh(n);
  for (auto& v : suh)
    v = Half((Next(&seed) & 1 ? -1.0F : 1.0F) * 0.05F);
  for (auto& v : svh)
    v = Half((Next(&seed) & 1 ? -1.0F : 1.0F) * 0.05F);
  return {Upload(trellis),    Upload(suh), Upload(svh), k, n, rate,
          ex::Codebook::kMul1};
}

template<class F>
double TimeGraph(hipStream_t stream, int copies, F&& body) {
  hipGraph_t graph;
  hipGraphExec_t exec;
  Ck(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "capture");
  for (int c = 0; c < kCalls; ++c)
    body(c % copies, stream);
  Ck(hipStreamEndCapture(stream, &graph), "end capture");
  Ck(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
  hipEvent_t a, b;
  Ck(hipEventCreate(&a), "event");
  Ck(hipEventCreate(&b), "event");
  Ck(hipGraphLaunch(exec, stream), "warm");
  Ck(hipStreamSynchronize(stream), "warm sync");
  float best = 1e30F;
  for (int r = 0; r < kReps; ++r) {
    Ck(hipEventRecord(a, stream), "record");
    Ck(hipGraphLaunch(exec, stream), "launch");
    Ck(hipEventRecord(b, stream), "record");
    Ck(hipEventSynchronize(b), "sync");
    float ms = 0;
    Ck(hipEventElapsedTime(&ms, a, b), "elapsed");
    best = ms < best ? ms : best;
  }
  Ck(hipEventDestroy(a), "destroy");
  Ck(hipEventDestroy(b), "destroy");
  Ck(hipGraphExecDestroy(exec), "destroy");
  Ck(hipGraphDestroy(graph), "destroy");
  return 1e3 * best / kCalls;
}

struct Shape {
  const char* name;
  unsigned k;
  std::vector<unsigned> n;
  int bits;
  bool mlp;  // n = {ff}: gate/up k x ff (SwiGLU), down ff x k
};

std::vector<float> Download(const float* d, std::size_t count) {
  std::vector<float> h(count);
  Ck(hipMemcpy(h.data(), d, count * 4, hipMemcpyDeviceToHost), "download");
  return h;
}

}  // namespace

int main() {
  hipStream_t stream;
  Ck(hipStreamCreate(&stream), "stream");
  const std::vector<Shape> shapes{
      {"GDN qkv+z K5 2560->10240+6144", 2560, {10240, 6144}, 5, false},
      {"ssm_out/out K6 6144->2560", 6144, {2560}, 6, false},
      {"attn q+k+v K6 2560->12288+512+512", 2560, {12288, 512, 512}, 6, false},
      {"shared MLP K6 2560->640->2560", 2560, {640}, 6, true},
      {"head-shaped K8 2560->16384", 2560, {16384}, 8, false},
  };
  constexpr unsigned kMaxK = 6144;
  std::vector<std::uint16_t> slab_h(std::size_t{3} * 8 * kMaxK);
  std::uint32_t seed = 0xB51U;
  for (auto& v : slab_h)
    v = Half(float(int(Next(&seed) % 2001) - 1000) * 5e-4F);
  auto* slab = Upload(slab_h);
  std::uint32_t *counters, *ref_counters;
  std::uint16_t *down_slab, *ref_down_slab;
  Ck(hipMalloc(&counters, 4096), "malloc");
  Ck(hipMalloc(&ref_counters, 4096), "malloc");
  Ck(hipMemset(counters, 0, 4096), "memset");
  Ck(hipMemset(ref_counters, 0, 4096), "memset");
  Ck(hipMalloc(&down_slab, 8 * kMaxK * 2), "malloc");
  Ck(hipMalloc(&ref_down_slab, 8 * kMaxK * 2), "malloc");
  const fast::Exl3FastWorkspace pre{slab, slab_h.size(), counters,
                                    1024, down_slab,     8 * kMaxK};
  std::printf("%-36s %4s %10s %10s %8s %8s %7s\n", "shape", "R", "old us",
              "new us", "old GB/s", "new GB/s", "bits");
  for (const Shape& s : shapes) {
    std::vector<std::vector<ex::DeviceLinear>> copies;
    std::size_t bytes = 0;
    while (copies.empty() || bytes * copies.size() < kColdBytes) {
      std::vector<ex::DeviceLinear> group;
      bytes = 0;
      const auto c = static_cast<std::uint32_t>(copies.size());
      if (s.mlp) {
        group.push_back(Linear(s.k, s.n[0], s.bits, 11 + 3 * c));
        group.push_back(Linear(s.k, s.n[0], s.bits, 12 + 3 * c));
        group.push_back(Linear(s.n[0], s.k, s.bits, 13 + 3 * c));
      } else {
        for (std::size_t i = 0; i < s.n.size(); ++i)
          group.push_back(Linear(s.k, s.n[i], s.bits, 17 + 5 * c + i));
      }
      for (const auto& l : group)
        bytes += l.TrellisBytes();
      copies.push_back(std::move(group));
    }
    const int count = static_cast<int>(copies.size());
    unsigned out_n = 0;
    for (unsigned n : s.n)
      out_n += n;
    if (s.mlp)
      out_n = s.k;
    float *y, *y_ref, *gate, *up, *gate_ref, *up_ref;
    Ck(hipMalloc(&y, std::size_t{8} * out_n * 4), "malloc");
    Ck(hipMalloc(&y_ref, std::size_t{8} * out_n * 4), "malloc");
    Ck(hipMalloc(&gate, std::size_t{8} * s.n[0] * 4), "malloc");
    Ck(hipMalloc(&up, std::size_t{8} * s.n[0] * 4), "malloc");
    Ck(hipMalloc(&gate_ref, std::size_t{8} * s.n[0] * 4), "malloc");
    Ck(hipMalloc(&up_ref, std::size_t{8} * s.n[0] * 4), "malloc");
    for (unsigned rows : {1U, 2U, 4U, 8U}) {
      const auto run = [&](bool old, int c, hipStream_t st) {
        const auto& g = copies[c];
        float* out = old ? y_ref : y;
        if (s.mlp) {
          const fast::Exl3FastMlp m{
              &g[0], &g[1], &g[2], old ? gate_ref : gate, old ? up_ref : up,
              out,   s.k};
          if (old)
            ref::Mlp(m, slab, rows, ref_down_slab, ref_counters, st);
          else
            fast::LaunchExl3FastMlp(m, nullptr, s.k, rows, pre, st, true);
          return;
        }
        std::vector<fast::Exl3FastTarget> targets;
        std::size_t offset = 0;
        for (std::size_t i = 0; i < g.size(); ++i) {
          targets.push_back({&g[i], out + offset * 8, s.n[i]});
          offset += s.n[i];
        }
        if (old)
          ref::Projections(targets, slab, rows, ref_counters, st);
        else
          fast::LaunchExl3FastProjections(targets, nullptr, s.k, rows, pre, st,
                                          true);
      };
      // Unwritten padded rows stay zero in both outputs.
      Ck(hipMemset(y, 0, std::size_t{8} * out_n * 4), "memset");
      Ck(hipMemset(y_ref, 0, std::size_t{8} * out_n * 4), "memset");
      run(true, 0, stream);
      run(false, 0, stream);
      Ck(hipStreamSynchronize(stream), "check sync");
      const auto a = Download(y, std::size_t{8} * out_n);
      const auto b = Download(y_ref, std::size_t{8} * out_n);
      const bool same = std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
      const double t_old = TimeGraph(
          stream, count, [&](int c, hipStream_t st) { run(true, c, st); });
      const double t_new = TimeGraph(
          stream, count, [&](int c, hipStream_t st) { run(false, c, st); });
      std::printf("%-36s %4u %10.2f %10.2f %8.1f %8.1f %7s\n", s.name, rows,
                  t_old, t_new, bytes / t_old * 1e-3, bytes / t_new * 1e-3,
                  same ? "equal" : "DIFFER");
      if (!same)
        return 1;
    }
    for (void* p : {static_cast<void*>(y), static_cast<void*>(y_ref),
                    static_cast<void*>(gate), static_cast<void*>(up),
                    static_cast<void*>(gate_ref), static_cast<void*>(up_ref)})
      Ck(hipFree(p), "free");
    for (auto& g : copies)
      for (auto& l : g) {
        Ck(hipFree(const_cast<std::uint16_t*>(l.trellis)), "free");
        Ck(hipFree(const_cast<std::uint16_t*>(l.suh)), "free");
        Ck(hipFree(const_cast<std::uint16_t*>(l.svh)), "free");
      }
  }
  return 0;
}
