// GDN prefill microbenchmark at the Flash-Next shape (16 key heads, 48 value
// heads, head dim 128, conv kernel 4): GatedDeltaNet on the EXL3 prefill
// route (row split, stable conv, F32 output) for a 2,048-token chunk, plus a
// checkpointed call and a chunk-split replay of the same tokens.
//
// Prints the mean time per call and FNV-1a hashes of the outputs, the final
// recurrent and conv states and the checkpoint state, so two builds can be
// compared bit for bit: the hashes must not change when only the kernels'
// scheduling does. "chunk invariance" checks that 2048 tokens as 1 x 2048,
// 2 x 1024 and 1000 + 1048 leave the same final state and outputs.
//
//   qwen38_flash_next_gdn_prefill_bench [tokens] [iterations]
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

constexpr std::uint32_t kKHeads = 16, kVHeads = 48, kDim = 128, kKernel = 4;
constexpr std::uint32_t kChannels = (2 * kKHeads + kVHeads) * kDim;
constexpr std::uint32_t kZ = kVHeads * kDim;
constexpr float kEps = 1e-6F;

void CheckHip(hipError_t e, const char* what) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

template<typename T>
struct Buffer {
  explicit Buffer(std::size_t n) : count(n) {
    CheckHip(hipMalloc(&data, n * sizeof(T) + 256), "hipMalloc");
    CheckHip(hipMemset(data, 0, n * sizeof(T) + 256), "hipMemset");
  }
  ~Buffer() { (void)hipFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void Upload(const std::vector<T>& v) {
    CheckHip(
        hipMemcpy(data, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice),
        "upload");
  }
  std::vector<T> Download() const {
    std::vector<T> v(count);
    CheckHip(
        hipMemcpy(v.data(), data, count * sizeof(T), hipMemcpyDeviceToHost),
        "download");
    return v;
  }
  T* data{nullptr};
  std::size_t count;
};

std::uint64_t Fnv(const void* p, std::size_t bytes) {
  const auto* b = static_cast<const std::uint8_t*>(p);
  std::uint64_t h = 1469598103934665603ULL;
  for (std::size_t i = 0; i < bytes; ++i) {
    h ^= b[i];
    h *= 1099511628211ULL;
  }
  return h;
}

std::vector<float> Normal(std::size_t n, std::uint32_t seed, float scale) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d;
  std::vector<float> v(n);
  for (auto& x : v)
    x = d(rng) * scale;
  return v;
}

struct Inputs {
  explicit Inputs(std::uint32_t tokens)
      : qkv(std::size_t{tokens} * kChannels),
        z(std::size_t{tokens} * kZ),
        ab(std::size_t{tokens} * 2 * kVHeads),
        conv_w(std::size_t{kChannels} * kKernel),
        a(kVHeads),
        dt(kVHeads),
        norm_w(kDim) {
    qkv.Upload(Normal(qkv.count, 1, 1.0F));
    z.Upload(Normal(z.count, 2, 1.0F));
    ab.Upload(Normal(ab.count, 3, 1.0F));
    conv_w.Upload(Normal(conv_w.count, 4, 0.5F));
    std::vector<float> av(kVHeads), dtv(kVHeads), nw(kDim);
    for (std::uint32_t h = 0; h < kVHeads; ++h) {
      av[h] = -0.5F - 0.03F * h;
      dtv[h] = 0.1F * (h % 7) - 0.3F;
    }
    for (std::uint32_t i = 0; i < kDim; ++i)
      nw[i] = 1.0F + 0.01F * (i % 13);
    a.Upload(av);
    dt.Upload(dtv);
    norm_w.Upload(nw);
  }
  Buffer<float> qkv, z, ab, conv_w, a, dt, norm_w;
};

struct State {
  explicit State(std::uint32_t tokens)
      : conv(std::size_t{kKernel - 1} * kChannels),
        state(std::size_t{kVHeads} * kDim * kDim),
        scratch(std::size_t{tokens + kKernel} * kChannels),
        qn(std::size_t{tokens} * kKHeads * kDim * 3),
        kn(std::size_t{tokens} * kVHeads * 2 * 2),
        raw(std::size_t{tokens} * kVHeads * kDim),
        out(std::size_t{tokens} * kZ) {
    state.Upload(Normal(state.count, 5, 0.05F));
  }
  Buffer<float> conv, state, scratch, qn, kn, raw, out;
};

/// One GatedDeltaNet prefill call over tokens [first, first + n) of `in`.
void Run(const Inputs& in, State& s, std::uint32_t first, std::uint32_t n,
         q::GdnCheckpoint checkpoint = {}) {
  if (!q::GatedDeltaNet(
          in.qkv.data + std::size_t{first} * kChannels, kChannels,
          in.z.data + std::size_t{first} * kZ, kZ,
          in.ab.data + std::size_t{first} * 2 * kVHeads, in.conv_w.data,
          in.a.data, in.dt.data, in.norm_w.data, s.conv.data, s.scratch.data,
          s.qn.data, s.kn.data, s.raw.data, s.state.data,
          s.out.data + std::size_t{first} * kZ, nullptr, {}, {}, n, kKHeads,
          kVHeads, kDim, kKernel, true, false, kEps, nullptr, nullptr,
          checkpoint, 0, nullptr, nullptr, true))
    throw std::runtime_error("GatedDeltaNet launch failed");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::uint32_t tokens = argc > 1 ? std::stoul(argv[1]) : 2048;
    const int iterations = argc > 2 ? std::stoi(argv[2]) : 20;
    // The chunk-invariance splits below need nonempty chunks.
    if (tokens < 64 || iterations < 1)
      throw std::runtime_error("tokens must be at least 64, iterations >= 1");
    Inputs in(tokens);
    State s(tokens);
    const auto initial_state = s.state.Download();

    // Timing: every call restarts from the same state.
    hipEvent_t e0, e1;
    CheckHip(hipEventCreate(&e0), "event");
    CheckHip(hipEventCreate(&e1), "event");
    Run(in, s, 0, tokens);  // warm-up
    float total_ms = 0;
    for (int it = 0; it < iterations; ++it) {
      s.state.Upload(initial_state);
      CheckHip(hipMemset(s.conv.data, 0, s.conv.count * sizeof(float)),
               "conv reset");
      CheckHip(hipEventRecord(e0, nullptr), "record");
      Run(in, s, 0, tokens);
      CheckHip(hipEventRecord(e1, nullptr), "record");
      CheckHip(hipEventSynchronize(e1), "sync");
      float ms = 0;
      CheckHip(hipEventElapsedTime(&ms, e0, e1), "elapsed");
      total_ms += ms;
    }
    std::cout << "GatedDeltaNet prefill tokens=" << tokens
              << " mean_ms=" << std::fixed << std::setprecision(4)
              << total_ms / iterations << '\n';

    // Fingerprints of one full call from the initial state.
    const auto full = [&](std::vector<float>* out, std::vector<float>* st,
                          std::vector<float>* conv,
                          const std::vector<std::uint32_t>& chunks) {
      s.state.Upload(initial_state);
      CheckHip(hipMemset(s.conv.data, 0, s.conv.count * sizeof(float)),
               "conv reset");
      std::uint32_t first = 0;
      for (const auto n : chunks) {
        Run(in, s, first, n);
        first += n;
      }
      CheckHip(hipDeviceSynchronize(), "sync");
      *out = s.out.Download();
      *st = s.state.Download();
      *conv = s.conv.Download();
    };
    std::vector<float> out, st, conv;
    full(&out, &st, &conv, {tokens});
    std::cout << std::hex << "hash out=" << Fnv(out.data(), out.size() * 4)
              << " state=" << Fnv(st.data(), st.size() * 4)
              << " conv=" << Fnv(conv.data(), conv.size() * 4) << std::dec
              << '\n';
    // Checkpoint at the middle token.
    {
      Buffer<float> cp_state(s.state.count), cp_hist(s.conv.count);
      s.state.Upload(initial_state);
      CheckHip(hipMemset(s.conv.data, 0, s.conv.count * sizeof(float)),
               "conv reset");
      q::GdnCheckpoint cp{};
      cp.state = cp_state.data;
      cp.history = cp_hist.data;
      cp.tokens = tokens / 2 + 3;
      Run(in, s, 0, tokens, cp);
      CheckHip(hipDeviceSynchronize(), "sync");
      const auto c = cp_state.Download();
      const auto hc = cp_hist.Download();
      std::cout << std::hex << "hash checkpoint=" << Fnv(c.data(), c.size() * 4)
                << " checkpoint_history=" << Fnv(hc.data(), hc.size() * 4)
                << std::dec << '\n';
    }
    // Chunk invariance: the same tokens in other chunkings.
    bool invariant = true;
    for (const auto& chunks :
         {std::vector<std::uint32_t>{tokens / 2, tokens - tokens / 2},
          std::vector<std::uint32_t>{tokens / 2 - 24,
                                     tokens - tokens / 2 + 24}}) {
      std::vector<float> o2, s2, c2;
      full(&o2, &s2, &c2, chunks);
      const auto same = [](const std::vector<float>& x,
                           const std::vector<float>& y) {
        return x.size() == y.size() &&
               std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0;
      };
      invariant = invariant && same(o2, out) && same(s2, st) && same(c2, conv);
    }
    std::cout << "chunk invariance " << (invariant ? "PASS" : "FAIL") << '\n';
    return invariant ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
