// Flash-Next EXL3 prefill HC combine (B6 P5): the four-lane combine as the
// EXL3 prefill route runs it (residual update written back, F32 norm left
// out, per-stream scales and the tiled Q8 norm for the next mixer), against
// the frozen pre-B6 kernel (tests/models/qwen38_flash_next/
// hc_combine_reference.hpp), then the W8A8 HC down projection (10240 -> 320)
// that reads the Q8 norm. Three specializations: the plain combine, the
// fused-MoE combine with a deferred prior update (the post-MoE combine of
// every layer), and the deferring combine (no residual store). c03 shapes,
// 256 rows (a short chunk, plain stores) and 512 / 2,048 / 4,096 rows
// (streaming residual stores). Every output byte (residual, Q8 norm, scales,
// deferred weights) must equal the frozen kernel's; the bench exits nonzero
// otherwise. µs/call is the best of kReps averages over kCalls calls, old
// and new alternating.
//
// Build (CMake, links the production library):
//   cmake --build --preset gpu-test --target qwen38_flash_next_hc_prefill_bench
// Run:
//   build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_hc_prefill_bench
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/core/quant/exl3/device.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/exl3_prefill_combine.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "tests/models/qwen38_flash_next/hc_combine_reference.hpp"

namespace ex = gufo::quant::exl3;
namespace fn = gufo::models::qwen38_flash_next::rocm;
namespace ref = gufo::tests::hc_combine_reference;

namespace {

constexpr unsigned kHidden = 2560, kStreams = 4, kRank = 320;
constexpr unsigned kHc = kHidden * kStreams;
constexpr unsigned kExperts = 512, kUsed = 10;
constexpr float kEps = 1.0e-6F;
constexpr int kCalls = 8;
constexpr int kReps = 5;

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

template<class T>
std::vector<T> Download(const T* d, std::size_t count) {
  std::vector<T> h(count);
  Ck(hipMemcpy(h.data(), d, count * sizeof(T), hipMemcpyDeviceToHost),
     "download");
  return h;
}

std::vector<float> Values(std::size_t n, std::uint32_t seed, float scale) {
  std::vector<float> v(n);
  for (auto& x : v)
    x = scale * (static_cast<float>(Next(&seed) % 2001) / 1000.0F - 1.0F);
  return v;
}

std::uint16_t Half(float v) {
  _Float16 h = static_cast<_Float16>(v);
  std::uint16_t bits;
  std::memcpy(&bits, &h, 2);
  return bits;
}

template<class F>
double TimeUs(hipStream_t stream, F&& body) {
  hipEvent_t a, b;
  Ck(hipEventCreate(&a), "event");
  Ck(hipEventCreate(&b), "event");
  body();
  Ck(hipStreamSynchronize(stream), "warm");
  float best = 1e30F;
  for (int r = 0; r < kReps; ++r) {
    Ck(hipEventRecord(a, stream), "record");
    for (int c = 0; c < kCalls; ++c)
      body();
    Ck(hipEventRecord(b, stream), "record");
    Ck(hipEventSynchronize(b), "sync");
    float ms = 0;
    Ck(hipEventElapsedTime(&ms, a, b), "elapsed");
    best = std::min(best, ms);
  }
  Ck(hipEventDestroy(a), "destroy");
  Ck(hipEventDestroy(b), "destroy");
  return 1e3 * best / kCalls;
}

// One combine's outputs, old and new, from the same inputs.
struct Outputs {
  float* res = nullptr;
  std::uint8_t* q8 = nullptr;
  float* scale = nullptr;
  float* defer_w = nullptr;
};

}  // namespace

int main() {
  hipStream_t stream;
  Ck(hipStreamCreate(&stream), "stream");
  // Q8_0 HC down weights: 320 rows of 10240 (34-byte blocks).
  std::vector<std::uint8_t> w(std::size_t{kRank} * (kHc / 32) * 34);
  {
    std::uint32_t s = 0xC0DE0001U;
    for (std::size_t b = 0; b < w.size(); b += 34) {
      const std::uint16_t d = Half(0.002F);
      std::memcpy(&w[b], &d, 2);
      for (int i = 2; i < 34; ++i)
        w[b + i] = static_cast<std::uint8_t>(Next(&s));
    }
  }
  const auto* d_w = Upload(w);
  const auto* d_gamma = Upload(Values(kHc, 0x6A33A001U, 1.0F));
  // Routed down bank for the fused MoE output: only svh is read.
  std::vector<ex::DeviceLinear> bank(kExperts);
  for (unsigned e = 0; e < kExperts; ++e) {
    std::vector<std::uint16_t> svh(kHidden);
    std::uint32_t s = 0x5A5A0000U + e;
    for (auto& v : svh)
      v = Half((Next(&s) & 1 ? -1.0F : 1.0F) * 0.05F);
    bank[e].svh = Upload(svh);
    bank[e].k = 640;
    bank[e].n = kHidden;
  }
  const auto* d_bank = Upload(bank);
  bool ok = true;
  for (const unsigned rows : {256U, 512U, 2048U, 4096U}) {
    const std::size_t res_count = std::size_t{rows} * kHc;
    const auto res0 = Values(res_count, 0x5E5E0000U + rows, 4.0F);
    const auto* d_block =
        Upload(Values(std::size_t{rows} * kHidden, 0xB10C0000U + rows, 1.0F));
    const auto* d_inject =
        Upload(Values(std::size_t{rows} * kStreams, 0x1A1A0000U + rows, 2.0F));
    // The deferred prior update (attention output and its stream weights).
    const auto* d_prior_out =
        Upload(Values(std::size_t{rows} * kHidden, 0x9A1A0000U + rows, 1.0F));
    const auto* d_prior_w =
        Upload(Values(std::size_t{rows} * kStreams, 0x9B1B0000U + rows, 1.0F));
    // Routed rows: distinct experts per token, identity packing.
    std::uint32_t seed = 0x1D500000U + rows;
    std::vector<std::int32_t> ids(std::size_t{rows} * kUsed), inv(ids.size());
    for (unsigned t = 0; t < rows; ++t)
      for (unsigned s = 0; s < kUsed; ++s) {
        ids[t * kUsed + s] = static_cast<std::int32_t>(
            (Next(&seed) % (kExperts / kUsed)) * kUsed + s);
        inv[t * kUsed + s] = static_cast<std::int32_t>(t * kUsed + s);
      }
    std::vector<std::uint16_t> raw(ids.size() * kHidden);
    for (auto& v : raw)
      v = Half(static_cast<float>(Next(&seed) % 2001) / 100.0F - 10.0F);
    const auto* d_ids = Upload(ids);
    const auto* d_inv = Upload(inv);
    const auto* d_raw = Upload(raw);
    const auto* d_weights =
        Upload(Values(ids.size(), 0x3E3E0000U + rows, 0.2F));
    const auto* d_shared =
        Upload(Values(std::size_t{rows} * kHidden, 0x5B5B0000U + rows, 1.0F));
    const auto* d_router = Upload(
        Values(std::size_t{rows} * (kExperts + 1), 0x7C7C0000U + rows, 2.0F));
    const fn::Exl3PrefillMoeRows moe{d_bank,
                                     d_ids,
                                     d_inv,
                                     d_raw,
                                     d_weights,
                                     d_shared,
                                     d_router + kExperts,
                                     kExperts + 1,
                                     kUsed,
                                     kHidden};
    const fn::DeferredResidual prior{d_prior_out, d_prior_w};
    const std::size_t q8_bytes = fn::Q8TiledBytes(rows, kHc);
    const auto alloc = [&] {
      Outputs o;
      o.res = Upload(res0);
      o.q8 = Upload(std::vector<std::uint8_t>(q8_bytes, 0xA5));
      o.scale = Upload(std::vector<float>(std::size_t{rows} * kStreams, -7.0F));
      o.defer_w =
          Upload(std::vector<float>(std::size_t{rows} * kStreams, -7.0F));
      return o;
    };
    Outputs old_out = alloc(), new_out = alloc();
    auto* d_xn = Upload(std::vector<float>(kStreams));  // F32 norm left out
    auto* d_lo = Upload(std::vector<float>(std::size_t{rows} * kRank));
    enum class Kind { kPlain, kMoePrior, kDeferred };
    const auto old_combine = [&](Kind kind, Outputs& o) {
      const dim3 grid(rows), block(ref::kThreads);
      if (kind == Kind::kPlain)
        hipLaunchKernelGGL((ref::OldHcCombineVec4Kernel<float>), grid, block, 0,
                           stream, o.res, d_block, d_inject, 1U, d_gamma, d_xn,
                           o.q8, kHidden, kEps, nullptr, nullptr, nullptr,
                           nullptr, o.scale, fn::Exl3PrefillMoeRows{},
                           fn::DeferredResidual{});
      else if (kind == Kind::kMoePrior)
        hipLaunchKernelGGL((ref::OldHcCombineVec4Kernel<float, false, true>),
                           grid, block, 0, stream, o.res, nullptr, d_inject, 1U,
                           d_gamma, d_xn, o.q8, kHidden, kEps, nullptr, nullptr,
                           nullptr, nullptr, o.scale, moe, prior);
      else
        hipLaunchKernelGGL((ref::OldHcCombineVec4Kernel<float, true>), grid,
                           block, 0, stream, o.res, d_block, d_inject, 1U,
                           d_gamma, d_xn, o.q8, kHidden, kEps, o.defer_w,
                           nullptr, nullptr, nullptr, o.scale,
                           fn::Exl3PrefillMoeRows{}, prior);
      Ck(hipGetLastError(), "old combine");
    };
    const auto new_combine = [&](Kind kind, Outputs& o) {
      fn::HcCombine(o.res, kind == Kind::kMoePrior ? nullptr : d_block,
                    d_inject, 1, d_gamma, d_xn, rows, kHidden, kStreams, kEps,
                    stream, false, true, nullptr, o.scale,
                    kind == Kind::kMoePrior ? &moe : nullptr, o.q8,
                    kind == Kind::kDeferred ? o.defer_w : nullptr,
                    kind == Kind::kPlain ? fn::DeferredResidual{} : prior);
      Ck(hipGetLastError(), "combine");
    };
    const auto down = [&](const Outputs& o) {
      if (!fn::W8A8Gemm(d_w, o.q8, d_lo, rows, kRank, kHc, stream))
        std::exit(1);
    };
    const auto reset = [&](Outputs& o) {
      Ck(hipMemcpy(o.res, res0.data(), res0.size() * sizeof(float),
                   hipMemcpyHostToDevice),
         "reset");
    };
    // Sentinels in every other output before each equality check, so a
    // store one kernel misses cannot match the previous case's bytes.
    const std::vector<float> sentinel(std::size_t{rows} * kStreams, -7.0F);
    const auto poison = [&](Outputs& o) {
      Ck(hipMemset(o.q8, 0xA5, q8_bytes), "poison q8");
      for (float* p : {o.scale, o.defer_w})
        Ck(hipMemcpy(p, sentinel.data(), sentinel.size() * sizeof(float),
                     hipMemcpyHostToDevice),
           "poison");
    };
    for (const Kind kind : {Kind::kPlain, Kind::kMoePrior, Kind::kDeferred}) {
      const char* name = kind == Kind::kPlain      ? "plain"
                         : kind == Kind::kMoePrior ? "MoE + prior"
                                                   : "deferred";
      reset(old_out);
      reset(new_out);
      poison(old_out);
      poison(new_out);
      old_combine(kind, old_out);
      new_combine(kind, new_out);
      Ck(hipStreamSynchronize(stream), "equality");
      const std::size_t sc = std::size_t{rows} * kStreams;
      // Byte comparisons (not float ==: signed zeros, NaNs).
      const auto same_bytes = [](const void* a, const void* b,
                                 std::size_t bytes) {
        return Download(static_cast<const std::uint8_t*>(a), bytes) ==
               Download(static_cast<const std::uint8_t*>(b), bytes);
      };
      bool same =
          same_bytes(old_out.res, new_out.res, res_count * sizeof(float)) &&
          same_bytes(old_out.q8, new_out.q8, q8_bytes) &&
          same_bytes(old_out.scale, new_out.scale, sc * sizeof(float));
      if (kind == Kind::kDeferred)
        same = same &&
               same_bytes(old_out.defer_w, new_out.defer_w, sc * sizeof(float));
      // The residual must actually have been updated (or, deferred, kept).
      const bool updated = Download(new_out.res, res_count) != res0;
      same = same && updated == (kind != Kind::kDeferred);
      ok = ok && same;
      // Timed calls keep updating res: restore it before each section.
      reset(old_out);
      reset(new_out);
      double old_us = TimeUs(stream, [&] { old_combine(kind, old_out); });
      double new_us = TimeUs(stream, [&] { new_combine(kind, new_out); });
      new_us =
          std::min(new_us, TimeUs(stream, [&] { new_combine(kind, new_out); }));
      old_us =
          std::min(old_us, TimeUs(stream, [&] { old_combine(kind, old_out); }));
      reset(old_out);
      reset(new_out);
      double old_chain = TimeUs(stream, [&] {
        old_combine(kind, old_out);
        down(old_out);
      });
      double new_chain = TimeUs(stream, [&] {
        new_combine(kind, new_out);
        down(new_out);
      });
      new_chain = std::min(new_chain, TimeUs(stream, [&] {
                             new_combine(kind, new_out);
                             down(new_out);
                           }));
      old_chain = std::min(old_chain, TimeUs(stream, [&] {
                             old_combine(kind, old_out);
                             down(old_out);
                           }));
      std::printf(
          "rows %4u %-12s bytes %-6s combine old %8.1f new %8.1f (%+.1f%%)  "
          "+ HC down old %8.1f new %8.1f (%+.1f%%) µs\n",
          rows, name, same ? "equal" : "DIFFER", old_us, new_us,
          100.0 * (new_us - old_us) / old_us, old_chain, new_chain,
          100.0 * (new_chain - old_chain) / old_chain);
    }
    std::vector<const void*> owned{d_block,  d_inject, d_prior_out, d_prior_w,
                                   d_ids,    d_inv,    d_raw,       d_weights,
                                   d_shared, d_router, d_xn,        d_lo};
    for (const Outputs& o : {old_out, new_out})
      for (const void* p :
           {static_cast<const void*>(o.res), static_cast<const void*>(o.q8),
            static_cast<const void*>(o.scale),
            static_cast<const void*>(o.defer_w)})
        owned.push_back(p);
    for (const void* p : owned)
      Ck(hipFree(const_cast<void*>(p)), "free");
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
