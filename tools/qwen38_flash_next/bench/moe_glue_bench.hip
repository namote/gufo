// Flash-Next EXL3 prefill MoE glue: the gate/up rotation (RotateGateUpTokens)
// and down-input preparation (RestoreActivateScaleBuckets) with plain stores,
// as of commit bec85727, verbatim below, against the production kernels, which
// store their FP16 slabs with streaming (non-temporal) stores. Each is timed
// alone and followed by the routed bucket GEMMs that read its output (gate
// and up, or down), since the store policy mainly moves cost between the
// producer and its consumer. c03 shapes: 512 experts, top-10, hidden 2560,
// expert FF 640, K4 gate/up/down banks, 2,048 and 4,096 token chunks with a
// skewed routing. µs/call is the best of kReps averages over kCalls calls.
// Output bytes (and the down exponents) of old and new must be equal.
// Results: docs/GFX1151-KNOWHOW.md, "Flash-Next B6".
//
// Build (CMake, links the production library):
//   cmake --build --preset gpu-test --target qwen38_flash_next_moe_glue_bench
// Run:
//   build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_moe_glue_bench
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/core/hip/detail/exl3_gemv.hpp"
#include "src/core/quant/exl3/device.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/exl3_moe.hpp"

namespace ex = gufo::quant::exl3;
namespace arithmetic = gufo::hip::exl3;
namespace fn = gufo::models::qwen38_flash_next::rocm;

namespace {

constexpr unsigned kExperts = 512, kUsed = 10, kHidden = 2560, kFf = 640;
constexpr unsigned kTileRows = 64;
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
T* Alloc(std::size_t count) {
  T* d = nullptr;
  Ck(hipMalloc(&d, count * sizeof(T)), "malloc");
  Ck(hipMemset(d, 0, count * sizeof(T)), "memset");
  return d;
}

template<class T>
std::vector<T> Download(const T* d, std::size_t count) {
  std::vector<T> h(count);
  Ck(hipMemcpy(h.data(), d, count * sizeof(T), hipMemcpyDeviceToHost),
     "download");
  return h;
}

std::uint16_t Half(float v) {
  _Float16 h = static_cast<_Float16>(v);
  std::uint16_t bits;
  std::memcpy(&bits, &h, 2);
  return bits;
}

// One expert linear: random trellis words, suh/svh near +-1.
ex::DeviceLinear Linear(unsigned k, unsigned n, std::uint32_t seed) {
  const ex::BitRate rate{4, false};
  std::vector<std::uint16_t> trellis(std::size_t{k / 16} * (n / 16) *
                                     rate.TileWords());
  for (auto& w : trellis)
    w = static_cast<std::uint16_t>(Next(&seed));
  std::vector<std::uint16_t> suh(k), svh(n);
  for (auto& v : suh)
    v = Half((Next(&seed) & 1 ? -1.0F : 1.0F) *
             (0.75F + 0.5F * static_cast<float>(Next(&seed) % 1024) / 1024));
  for (auto& v : svh)
    v = Half((Next(&seed) & 1 ? -1.0F : 1.0F) * 0.05F);
  return {Upload(trellis),    Upload(suh), Upload(svh), k, n, rate,
          ex::Codebook::kMul1};
}

// ---- plain-store kernels (bec85727), verbatim apart from the names ----

__device__ __forceinline__ unsigned BucketExpert(const unsigned* offsets,
                                                 unsigned experts,
                                                 unsigned row) {
  unsigned lo = 0, hi = experts;
  while (lo + 1 < hi) {
    const unsigned mid = (lo + hi) / 2;
    if (offsets[mid] <= row)
      lo = mid;
    else
      hi = mid;
  }
  return lo;
}

__device__ int DownExponent(unsigned minimum, unsigned maximum) {
  if (maximum >= 0x7c00U)
    return -1;
  if (minimum >= 0x400U)  // Includes the all-zero row (minimum == 0x7fff).
    return 0;
  const int shift = __clz(minimum) - 21;  // 1..10, to make minimum normal.
  return (maximum >> 10) + shift <= 30 ? shift : -1;
}

__device__ std::uint16_t ScaleHalf(std::uint16_t bits, int exponent) {
  const unsigned value = bits & 0x7fffU;
  if (value == 0 || exponent <= 0)
    return bits;
  if (value >= 0x400U)
    return bits + (exponent << 10);
  const int normalize = __clz(value) - 21;
  const unsigned significand = value << normalize;
  return (bits & 0x8000U) | ((1 + exponent - normalize) << 10) |
         (significand & 0x3ffU);
}

__global__ __launch_bounds__(256) void OldRotateGateUpTokens(
    const ex::DeviceLinear* gate, const ex::DeviceLinear* up,
    const float* __restrict__ input, std::uint16_t* __restrict__ gate_out,
    std::uint16_t* __restrict__ up_out, const std::int32_t* __restrict__ ids,
    const std::int32_t* __restrict__ inverse, unsigned width, unsigned used) {
  const unsigned row = blockIdx.y, lane = threadIdx.x % 32;
  const unsigned segment = blockIdx.x * 8 + threadIdx.x / 32;
  if (segment * 128 >= width)
    return;
  const unsigned first = segment * 128 + lane * 4;
  float x[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j)
    x[j] = input[std::size_t{row} * width + first + j];
  // Lane s holds assignment s's packed row and suh pointers (used <= 32).
  const unsigned mine = row * used + min(lane, used - 1);
  const int my_packed = inverse[mine];
  const auto my_expert = static_cast<unsigned>(ids[mine]);
  const auto my_gs = reinterpret_cast<std::uintptr_t>(gate[my_expert].suh);
  const auto my_us = reinterpret_cast<std::uintptr_t>(up[my_expert].suh);
#pragma unroll 2
  for (unsigned s = 0; s < used; ++s) {
    const auto packed = static_cast<unsigned>(__shfl(my_packed, s, 32));
    const auto* gs = reinterpret_cast<const _Float16*>(__shfl(my_gs, s, 32));
    const auto* us = reinterpret_cast<const _Float16*>(__shfl(my_us, s, 32));
    float g[4], u[4];
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      g[j] = x[j] * static_cast<float>(gs[first + j]);
      asm volatile("" : "+v"(g[j]));
      u[j] = x[j] * static_cast<float>(us[first + j]);
      asm volatile("" : "+v"(u[j]));
    }
    arithmetic::Butterfly128(g, lane);
    arithmetic::Butterfly128(u, lane);
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      float a = g[j] * arithmetic::kHadamardScale;
      asm volatile("" : "+v"(a));
      float b = u[j] * arithmetic::kHadamardScale;
      asm volatile("" : "+v"(b));
      const auto at = std::size_t{packed} * width + first + j;
      reinterpret_cast<_Float16*>(gate_out)[at] = static_cast<_Float16>(a);
      reinterpret_cast<_Float16*>(up_out)[at] = static_cast<_Float16>(b);
    }
  }
}

__global__ __launch_bounds__(1024) void OldRestoreActivateScaleBuckets(
    const ex::DeviceLinear* gate, const ex::DeviceLinear* up,
    const ex::DeviceLinear* down, const unsigned* offsets, unsigned experts,
    const _Float16* gate_y, const _Float16* up_y, std::uint16_t* output,
    std::int32_t* exponents, unsigned width) {
  const unsigned row = blockIdx.x, lane = threadIdx.x % 32;
  const unsigned wave = threadIdx.x / 32, waves = blockDim.x / 32;
  if (row >= offsets[experts])
    return;
  const unsigned lo = BucketExpert(offsets, experts, row);
  const unsigned first = wave * 128 + lane * 4;
  const auto* gs = reinterpret_cast<const _Float16*>(gate[lo].svh);
  const auto* us = reinterpret_cast<const _Float16*>(up[lo].svh);
  const auto* ds = reinterpret_cast<const _Float16*>(down[lo].suh);
  const auto offset = std::size_t{row} * width + first;
  float gv[4], uv[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    gv[j] = static_cast<float>(gate_y[offset + j]);
    uv[j] = static_cast<float>(up_y[offset + j]);
  }
  arithmetic::Butterfly128(gv, lane);
  arithmetic::Butterfly128(uv, lane);
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const float value = (gv[j] * arithmetic::kHadamardScale) *
                        static_cast<float>(gs[first + j]);
    const float other = (uv[j] * arithmetic::kHadamardScale) *
                        static_cast<float>(us[first + j]);
    float activation = (value / (1.0F + expf(-value))) * other;
    asm volatile("" : "+v"(activation));
    gv[j] = activation * static_cast<float>(ds[first + j]);
    asm volatile("" : "+v"(gv[j]));
  }
  arithmetic::Butterfly128(gv, lane);
  std::uint16_t bits[4];
  unsigned minimum = 0x7fffU, maximum = 0;
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    float scaled = gv[j] * arithmetic::kHadamardScale;
    asm volatile("" : "+v"(scaled));
    bits[j] = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(scaled));
    const unsigned value = bits[j] & 0x7fffU;
    if (value != 0)
      minimum = min(minimum, value);
    maximum = max(maximum, value);
  }
  for (unsigned o = 16; o != 0; o >>= 1) {
    minimum = min(minimum, __shfl_xor(minimum, o, 32));
    maximum = max(maximum, __shfl_xor(maximum, o, 32));
  }
  __shared__ unsigned minima[32], maxima[32];
  if (lane == 0) {
    minima[wave] = minimum;
    maxima[wave] = maximum;
  }
  __syncthreads();
  minimum = 0x7fffU;
  maximum = 0;
  for (unsigned i = 0; i < waves; ++i) {
    minimum = min(minimum, minima[i]);
    maximum = max(maximum, maxima[i]);
  }
  const int exponent = DownExponent(minimum, maximum);
  if (threadIdx.x == 0)
    exponents[row] = exponent;
#pragma unroll
  for (unsigned j = 0; j < 4; ++j)
    output[offset + j] = exponent > 0 ? ScaleHalf(bits[j], exponent) : bits[j];
}

// ---- timing ----

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

// Old and new alternate (old, new, new, old) against clock drift; the best
// of each is kept.
template<class Old, class New>
void Compare(hipStream_t stream, const char* what, Old&& old_body,
             New&& new_body) {
  double old_us = TimeUs(stream, old_body);
  double new_us = TimeUs(stream, new_body);
  new_us = std::min(new_us, TimeUs(stream, new_body));
  old_us = std::min(old_us, TimeUs(stream, old_body));
  std::printf("  %-40s old %9.1f  new %9.1f  (%+.1f%%)\n", what, old_us, new_us,
              100.0 * (new_us - old_us) / old_us);
}

}  // namespace

int main() {
  hipStream_t stream;
  Ck(hipStreamCreate(&stream), "stream");
  std::vector<ex::DeviceLinear> gate(kExperts), up(kExperts), down(kExperts);
  for (unsigned e = 0; e < kExperts; ++e) {
    gate[e] = Linear(kHidden, kFf, 0x1000U + e);
    up[e] = Linear(kHidden, kFf, 0x2000U + e);
    down[e] = Linear(kFf, kHidden, 0x3000U + e);
  }
  const auto* d_gate = Upload(gate);
  const auto* d_up = Upload(up);
  const auto* d_down = Upload(down);
  const ex::BitRate rate{4, false};
  bool ok = true;
  for (const unsigned rows : {2048U, 4096U}) {
    const unsigned assignments = rows * kUsed;
    // Skewed top-10: distinct experts per token, low ids favoured.
    std::uint32_t seed = 0x5EED0000U + rows;
    std::vector<std::int32_t> ids(assignments);
    for (unsigned t = 0; t < rows; ++t) {
      for (unsigned s = 0; s < kUsed; ++s) {
        for (;;) {
          const float u = static_cast<float>(Next(&seed) % 65536) / 65536.0F;
          const auto e = static_cast<std::int32_t>(u * u * kExperts);
          bool fresh = true;
          for (unsigned p = 0; p < s; ++p)
            fresh = fresh && ids[t * kUsed + p] != e;
          if (fresh) {
            ids[t * kUsed + s] = e;
            break;
          }
        }
      }
    }
    std::vector<float> x(std::size_t{rows} * kHidden);
    for (auto& v : x)
      v = (static_cast<float>(Next(&seed) % 4096) - 2048.0F) / 1024.0F;
    const auto* d_ids = Upload(ids);
    const auto* d_x = Upload(x);
    const unsigned max_buckets =
        std::min(assignments, kExperts + assignments / kTileRows);
    auto* offsets = Alloc<std::uint32_t>(kExperts + 1);
    auto* slots = Alloc<std::int32_t>(assignments);
    auto* inverse = Alloc<std::int32_t>(assignments);
    auto* buckets = Alloc<fn::Exl3ExpertBucket>(max_buckets);
    auto* bucket_count = Alloc<std::uint32_t>(1);
    fn::Exl3BuildExpertBuckets(d_ids, assignments, kExperts, kTileRows, offsets,
                               slots, buckets, bucket_count, max_buckets,
                               stream, inverse);
    const std::size_t in_count = std::size_t{assignments} * kHidden;
    const std::size_t ff_count = std::size_t{assignments} * kFf;
    auto* old_gin = Alloc<std::uint16_t>(in_count);
    auto* old_uin = Alloc<std::uint16_t>(in_count);
    auto* new_gin = Alloc<std::uint16_t>(in_count);
    auto* new_uin = Alloc<std::uint16_t>(in_count);
    auto* gate_y = Alloc<std::uint16_t>(ff_count);
    auto* up_y = Alloc<std::uint16_t>(ff_count);
    auto* old_down_in = Alloc<std::uint16_t>(ff_count);
    auto* new_down_in = Alloc<std::uint16_t>(ff_count);
    auto* old_exp = Alloc<std::int32_t>(assignments);
    auto* new_exp = Alloc<std::int32_t>(assignments);
    auto* down_out = Alloc<std::uint16_t>(in_count);

    const auto old_rotate = [&] {
      hipLaunchKernelGGL(OldRotateGateUpTokens,
                         dim3((kHidden / 128 + 7) / 8, rows), dim3(256), 0,
                         stream, d_gate, d_up, d_x, old_gin, old_uin, d_ids,
                         inverse, kHidden, kUsed);
      Ck(hipGetLastError(), "old rotation");
    };
    const auto new_rotate = [&] {
      fn::Exl3RotateGateUpTokens(d_gate, d_up, d_x, new_gin, new_uin, d_ids,
                                 inverse, rows, kHidden, kUsed, stream);
    };
    const auto gate_up = [&](const std::uint16_t* gin,
                             const std::uint16_t* uin) {
      gufo::hip::LaunchExl3PrefillBucketInnerHalf(
          d_gate, buckets, bucket_count, max_buckets, kTileRows, gin, gate_y,
          kHidden, kFf, rate, ex::Codebook::kMul1, nullptr, stream);
      gufo::hip::LaunchExl3PrefillBucketInnerHalf(
          d_up, buckets, bucket_count, max_buckets, kTileRows, uin, up_y,
          kHidden, kFf, rate, ex::Codebook::kMul1, nullptr, stream);
    };
    const auto old_restore = [&] {
      hipLaunchKernelGGL(
          OldRestoreActivateScaleBuckets, dim3(assignments),
          dim3(32 * (kFf / 128)), 0, stream, d_gate, d_up, d_down, offsets,
          kExperts, reinterpret_cast<const _Float16*>(gate_y),
          reinterpret_cast<const _Float16*>(up_y), old_down_in, old_exp, kFf);
      Ck(hipGetLastError(), "old restore");
    };
    const auto new_restore = [&] {
      fn::Exl3RestoreActivateScale(d_gate, d_up, d_down, offsets, kExperts,
                                   gate_y, up_y, new_down_in, new_exp,
                                   assignments, kFf, stream);
    };
    const auto down_gemm = [&](const std::uint16_t* in, const std::int32_t* e) {
      gufo::hip::LaunchExl3PrefillBucketInnerHalf(
          d_down, buckets, bucket_count, max_buckets, kTileRows, in, down_out,
          kFf, kHidden, rate, ex::Codebook::kMul1, e, stream);
    };

    // Equality first: rotations, then (from the same GEMM outputs) the
    // down inputs and exponents.
    old_rotate();
    new_rotate();
    gate_up(new_gin, new_uin);
    old_restore();
    new_restore();
    Ck(hipStreamSynchronize(stream), "equality");
    const bool same =
        Download(old_gin, in_count) == Download(new_gin, in_count) &&
        Download(old_uin, in_count) == Download(new_uin, in_count) &&
        Download(old_down_in, ff_count) == Download(new_down_in, ff_count) &&
        Download(old_exp, assignments) == Download(new_exp, assignments);
    ok = ok && same;
    const auto exponents = Download(new_exp, assignments);
    const auto scaled = std::count_if(exponents.begin(), exponents.end(),
                                      [](std::int32_t e) { return e > 0; });
    std::printf(
        "rows %u (%u assignments, %u bucket slots, %ld scaled down rows): "
        "bytes %s\n",
        rows, assignments, max_buckets, static_cast<long>(scaled),
        same ? "equal" : "DIFFER");
    Compare(stream, "gate/up rotation", old_rotate, new_rotate);
    Compare(
        stream, "rotation + gate/up GEMMs",
        [&] {
          old_rotate();
          gate_up(old_gin, old_uin);
        },
        [&] {
          new_rotate();
          gate_up(new_gin, new_uin);
        });
    Compare(stream, "restore/activate/scale", old_restore, new_restore);
    Compare(
        stream, "restore + down GEMM",
        [&] {
          old_restore();
          down_gemm(old_down_in, old_exp);
        },
        [&] {
          new_restore();
          down_gemm(new_down_in, new_exp);
        });
    const std::vector<const void*> owned{
        old_gin,     old_uin,     new_gin, new_uin,      gate_y,   up_y,
        old_down_in, new_down_in, old_exp, new_exp,      down_out, offsets,
        slots,       inverse,     buckets, bucket_count, d_ids,    d_x};
    for (const void* p : owned)
      Ck(hipFree(const_cast<void*>(p)), "free");
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
