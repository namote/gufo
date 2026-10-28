// R0b (plan/rotated-quant/PLAN.md): effective weight bandwidth per GGUF
// quantization type on gfx1151, at the Qwen 27B matrix shapes and batches.
//
// Production types (Q8_0, Q6_K, Q5_K, Q4_K, IQ4_XS, IQ4_NL, Q3_K, IQ3_S) run
// the Qwen production launchers, linked from src/models/qwen/hip/kernels with
// their CMake source options, and follow the routing of the production model:
//   AR (1 row)      decode_step.cpp: LaunchGEMV (kHipQuantDirect route) per
//                   matrix; also the fused decode launches that read Q/K/V,
//                   SSM qkv/gate/alpha/beta and FFN gate/up together.
//   verify (8 rows) batched_decode.cpp LaunchProjection:
//                   LaunchBatchedQuantGEMMFp32 (DFlash2: 7 drafts + anchor).
//   prefill 512     prefill_chunk.cpp: Q8_1 activation +
//                   LaunchBatchedQuantGEMMPreQuantized.
//   prefill 2048    UseQwen27bFp16Prefill (batch >= 1024, low-bit FFN):
//                   LaunchBatchedQuantGEMMFp16; an all-Q8_0 model keeps Q8_1.
// The model is assumed to use one type for every matrix of the estimate.
// Two-bit types (Q2_K, IQ2_XXS, IQ2_XS, IQ2_S) run upstream ggml mmvq through
// rq_type_bench_mmvq.hpp, for AR and verify only (no prefill kernel).
//
// Weights are random bytes in each type's real block layout with finite FP16
// scales, placed like production weights (anonymous huge-page mapping with
// read-only host registration) and rotated over copies well beyond the 32 MiB
// MALL. Times are per launch (hipEvent pairs), after warm-up, in rounds whose
// type order alternates; GB/s counts weight bytes only.
//
// Modes: check | time [rounds] [reps] | fwht
//   check  production and two-bit kernels against CPU references on the first
//          and last output rows; FWHT against a CPU Hadamard.
//   time   every type x shape x batch, then the FWHT table and token estimates.
//   fwht   only the FWHT table.
// Build: see plan/rotated-quant/R0B-RESULTS.md.
#define ENGINE_ENABLE_HIP 1
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <sys/mman.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#include "src/models/qwen/hip/ops/ssm.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"
#include "tools/bench/rq_type_bench_mmvq.hpp"

namespace {

using gufo::core::GgmlType;

void Check(hipError_t error, const char* what) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " +
                             hipGetErrorString(error));
}
#define HIP_CHECK(expr) Check((expr), #expr)

// ---------------------------------------------------------------- types

struct TypeInfo {
  const char* name;
  GgmlType type;  // production type, kF32 for two-bit
  int mmvq;       // ggml_type for two-bit, 0 for production
  int qk;         // values per block
  int block_bytes;
  int scale;      // FP16 d offset
  int scale_min;  // FP16 dmin offset or -1
  [[nodiscard]] bool production() const { return mmvq == 0; }
  [[nodiscard]] double bpw() const { return 8.0 * block_bytes / qk; }
  [[nodiscard]] std::size_t Bytes(std::size_t m, std::size_t k) const {
    return m * (k / qk) * block_bytes;
  }
};

constexpr TypeInfo kTypes[] = {
    {"Q8_0", GgmlType::kQ8_0, 0, 32, 34, 0, -1},
    {"Q6_K", GgmlType::kQ6_K, 0, 256, 210, 208, -1},
    {"Q5_K", GgmlType::kQ5_K, 0, 256, 176, 0, 2},
    {"Q4_K", GgmlType::kQ4_K, 0, 256, 144, 0, 2},
    {"IQ4_XS", GgmlType::kIQ4_XS, 0, 256, 136, 0, -1},
    {"IQ4_NL", GgmlType::kIQ4_NL, 0, 32, 18, 0, -1},
    {"Q3_K", GgmlType::kQ3_K, 0, 256, 110, 108, -1},
    {"IQ3_S", GgmlType::kIQ3_S, 0, 256, 110, 0, -1},
    {"Q2_K", GgmlType::kF32, 10, 256, 84, 80, 82},
    {"IQ2_XXS", GgmlType::kF32, 16, 256, 66, 0, -1},
    {"IQ2_XS", GgmlType::kF32, 17, 256, 74, 0, -1},
    {"IQ2_S", GgmlType::kF32, 22, 256, 82, 0, -1},
};
constexpr std::size_t kTypeCount = std::size(kTypes);

// ---------------------------------------------------------------- shapes

struct Shape {
  const char* name;
  std::size_t k;
  std::size_t m;
  int per_token;  // matrices of this shape read per token
};

// Qwen 27B: 48 DeltaNet and 16 full-attention layers, hidden 5120.
constexpr Shape kShapes[] = {
    {"attn_qkv", 5120, 10240, 48},        {"attn_gate", 5120, 6144, 48},
    {"attn_q", 5120, 12288, 16},          {"attn_k_v", 5120, 1024, 32},
    {"ssm_out_attn_out", 6144, 5120, 64}, {"ffn_gate_up", 5120, 17408, 128},
    {"ffn_down", 17408, 5120, 64},        {"output", 5120, 248320, 1},
};
constexpr std::size_t kOutputShape = 7;
constexpr std::size_t kTimeStepRank = 48;  // ssm_alpha / ssm_beta rows
constexpr std::size_t kBatches[] = {1, 8, 512, 2048};

// ---------------------------------------------------------------- data

__device__ unsigned Hash(unsigned value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

__global__ void InitWeights(unsigned char* data, std::size_t blocks,
                            int block_bytes, int scale, int scale_min,
                            unsigned seed) {
  const std::size_t b =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (b >= blocks)
    return;
  unsigned char* block = data + b * block_bytes;
  const unsigned base = Hash(static_cast<unsigned>(b) ^ seed);
  for (int i = 0; i < block_bytes; ++i)
    block[i] = static_cast<unsigned char>(Hash(base + i));
  *reinterpret_cast<__half*>(block + scale) =
      __float2half(0.002F + (base % 1000) * 0.000002F);
  if (scale_min >= 0)
    *reinterpret_cast<__half*>(block + scale_min) =
        __float2half(0.001F + (Hash(base) % 1000) * 0.000001F);
}

// Production placement: anonymous huge-page mapping, read-only host
// registration (src/models/qwen/hip/model_loader.cpp).
class MappedWeights {
public:
  MappedWeights(const TypeInfo& type, std::size_t m, std::size_t k,
                std::size_t copies, unsigned seed)
      : stride_((type.Bytes(m, k) + 4095) / 4096 * 4096),
        copies_(copies),
        bytes_(stride_ * copies) {
    host_ = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (host_ == MAP_FAILED)
      throw std::runtime_error("weight mmap failed");
    (void)madvise(host_, bytes_, MADV_HUGEPAGE);
    const std::size_t matrix = type.Bytes(m, k);
    const std::size_t blocks = matrix / type.block_bytes;
    void* staging = nullptr;
    HIP_CHECK(hipMalloc(&staging, matrix));
    InitWeights<<<(blocks + 255) / 256, 256>>>(
        static_cast<unsigned char*>(staging), blocks, type.block_bytes,
        type.scale, type.scale_min, seed);
    HIP_CHECK(hipGetLastError());
    for (std::size_t c = 0; c < copies_; ++c)
      HIP_CHECK(hipMemcpy(static_cast<unsigned char*>(host_) + c * stride_,
                          staging, matrix, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(staging));
    if (mprotect(host_, bytes_, PROT_READ) != 0)
      throw std::runtime_error("weight mprotect failed");
    HIP_CHECK(hipHostRegister(host_, bytes_,
                              hipHostRegisterMapped | hipHostRegisterReadOnly));
    HIP_CHECK(hipHostGetDevicePointer(&device_, host_, 0));
  }
  ~MappedWeights() {
    (void)hipHostUnregister(host_);
    (void)munmap(host_, bytes_);
  }
  MappedWeights(const MappedWeights&) = delete;
  MappedWeights& operator=(const MappedWeights&) = delete;
  [[nodiscard]] const void* Copy(std::size_t i) const {
    return static_cast<const unsigned char*>(device_) + (i % copies_) * stride_;
  }
  [[nodiscard]] const void* Host(std::size_t i) const {
    return static_cast<const unsigned char*>(host_) + (i % copies_) * stride_;
  }
  [[nodiscard]] std::size_t copies() const { return copies_; }

private:
  std::size_t stride_;
  std::size_t copies_;
  std::size_t bytes_;
  void* host_{};
  void* device_{};
};

template<typename T>
class Device {
public:
  explicit Device(std::size_t count) : count_(count) {
    HIP_CHECK(hipMalloc(&data_, std::max<std::size_t>(count, 1) * sizeof(T)));
  }
  ~Device() { (void)hipFree(data_); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  T* Get() const { return data_; }

private:
  T* data_{};
  std::size_t count_;
};

std::vector<float> RandomValues(std::size_t count, unsigned seed) {
  std::vector<float> values(count);
  std::uint32_t state = seed * 2654435761U + 1;
  for (auto& v : values) {
    state = state * 1664525U + 1013904223U;
    v = (static_cast<int>((state >> 8) % 20001) - 10000) * 0.00027F;
  }
  return values;
}

// Activations in every form the kernels read, for one (batch, K).
struct Activations {
  Activations(std::size_t max_batch, std::size_t max_k)
      : fp32(max_batch * max_k),
        bf16(max_batch * max_k),
        fp16(max_batch * max_k),
        q8_tiled(gufo::hip::QuantizedActivationBytes(max_batch, max_k)),
        q8_1(rq_mmvq::Q8_1Bytes(8, max_k)) {}
  // Fills every form from host values x[batch, k]; returns the rounded hosts.
  void Load(const std::vector<float>& x, std::size_t batch, std::size_t k,
            std::vector<float>* as_bf16, std::vector<float>* as_fp16) {
    std::vector<hip_bfloat16> b(x.size());
    std::vector<__half> h(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
      b[i] = hip_bfloat16(x[i]);
      h[i] = __float2half(x[i]);
    }
    HIP_CHECK(
        hipMemcpy(fp32.Get(), x.data(), x.size() * 4, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(bf16.Get(), b.data(), x.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(fp16.Get(), h.data(), x.size() * 2, hipMemcpyHostToDevice));
    gufo::hip::LaunchQuantizeActivationQ8_1(bf16.Get(), q8_tiled.Get(), batch,
                                            k, nullptr);
    if (batch <= 8)
      rq_mmvq::QuantizeQ8_1(fp32.Get(), q8_1.Get(), batch, k, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    if (as_bf16 != nullptr) {
      as_bf16->resize(x.size());
      for (std::size_t i = 0; i < x.size(); ++i)
        (*as_bf16)[i] = static_cast<float>(b[i]);
    }
    if (as_fp16 != nullptr) {
      as_fp16->resize(x.size());
      for (std::size_t i = 0; i < x.size(); ++i)
        (*as_fp16)[i] = __half2float(h[i]);
    }
  }
  Device<float> fp32;
  Device<hip_bfloat16> bf16;
  Device<__half> fp16;
  Device<unsigned char> q8_tiled;
  Device<unsigned char> q8_1;
};

// ---------------------------------------------------------------- kernels

enum class Family { kAr, kVerify, kPrefillQ8, kPrefillFp16, kMmvq };

const char* FamilyName(Family f) {
  switch (f) {
    case Family::kAr:
      return "LaunchGEMV";
    case Family::kVerify:
      return "BatchedQuantGEMMFp32";
    case Family::kPrefillQ8:
      return "Q8_1+PreQuantized";
    case Family::kPrefillFp16:
      return "BatchedQuantGEMMFp16";
    case Family::kMmvq:
      return "ggml-mmvq";
  }
  return "?";
}

// Production routing for a model whose every matrix has `type`.
Family Route(const TypeInfo& type, std::size_t batch) {
  if (!type.production())
    return Family::kMmvq;
  if (batch == 1)
    return Family::kAr;
  if (batch <= 8)
    return Family::kVerify;
  if (batch >= 1024 && type.type != GgmlType::kQ8_0)
    return Family::kPrefillFp16;
  return Family::kPrefillQ8;
}

void Launch(Family family, const TypeInfo& type, const void* w,
            Activations& act, float* y, std::size_t batch, std::size_t m,
            std::size_t k) {
  using namespace gufo::hip;
  switch (family) {
    case Family::kAr:
      LaunchGEMV(w, type.type, act.fp32.Get(), y, m, k, nullptr);
      return;
    case Family::kVerify:
      LaunchBatchedQuantGEMMFp32(type.type, w, act.fp32.Get(), y, batch, m, k,
                                 nullptr);
      return;
    case Family::kPrefillQ8:
      LaunchBatchedQuantGEMMPreQuantized(type.type, w, act.q8_tiled.Get(), y,
                                         batch, m, k, nullptr);
      return;
    case Family::kPrefillFp16:
      LaunchBatchedQuantGEMMFp16(type.type, w, act.fp16.Get(), y, batch, m, k,
                                 nullptr);
      return;
    case Family::kMmvq:
      rq_mmvq::Mmvq(static_cast<rq_mmvq::Type>(type.mmvq), w, act.q8_1.Get(), y,
                    batch, m, k, nullptr);
      return;
  }
}

// Independent FWHT: x'[r, c*128 + i] = (H D x)[i] / sqrt(128) per 128-value
// chunk, H the natural-order Sylvester Hadamard and D the sign vector. One
// wave32 per chunk, four values per lane, in place on FP32 rows.
__global__ __launch_bounds__(256) void Fwht128Kernel(
    float* __restrict__ x, const float* __restrict__ signs, std::size_t k,
    std::size_t chunks) {
  const std::size_t chunk = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (chunk >= chunks)
    return;
  const unsigned lane = threadIdx.x & 31;
  const std::size_t per_row = k / 128;
  const std::size_t column = (chunk % per_row) * 128 + lane * 4;
  float4* p = reinterpret_cast<float4*>(x + (chunk / per_row) * k + column);
  const float4 s = *reinterpret_cast<const float4*>(signs + column);
  float4 v = *p;
  float v0 = v.x * s.x, v1 = v.y * s.y, v2 = v.z * s.z, v3 = v.w * s.w;
  const float a0 = v0 + v1, a1 = v0 - v1, a2 = v2 + v3, a3 = v2 - v3;
  v0 = a0 + a2;
  v1 = a1 + a3;
  v2 = a0 - a2;
  v3 = a1 - a3;
#pragma unroll
  for (unsigned mask = 1; mask < 32; mask <<= 1) {
    const bool upper = (lane & mask) != 0;
    const float o0 = __shfl_xor(v0, mask), o1 = __shfl_xor(v1, mask);
    const float o2 = __shfl_xor(v2, mask), o3 = __shfl_xor(v3, mask);
    v0 = upper ? o0 - v0 : v0 + o0;
    v1 = upper ? o1 - v1 : v1 + o1;
    v2 = upper ? o2 - v2 : v2 + o2;
    v3 = upper ? o3 - v3 : v3 + o3;
  }
  constexpr float kScale = 0.08838834764831845F;  // 1 / sqrt(128)
  *p = make_float4(v0 * kScale, v1 * kScale, v2 * kScale, v3 * kScale);
}

void LaunchFwht(float* x, const float* signs, std::size_t batch,
                std::size_t k) {
  const std::size_t chunks = batch * k / 128;
  Fwht128Kernel<<<(chunks + 7) / 8, 256>>>(x, signs, k, chunks);
}

// ---------------------------------------------------------------- timing

class EventPool {
public:
  explicit EventPool(std::size_t n) : events_(n) {
    for (auto& e : events_)
      HIP_CHECK(hipEventCreate(&e));
  }
  ~EventPool() {
    for (auto& e : events_)
      (void)hipEventDestroy(e);
  }
  hipEvent_t operator[](std::size_t i) const { return events_[i]; }

private:
  std::vector<hipEvent_t> events_;
};

constexpr int kWarmup = 3;

// Appends `reps` per-launch times in microseconds.
void TimeLaunches(const std::function<void(std::size_t)>& launch, int reps,
                  std::vector<double>& out, std::size_t& copy) {
  static EventPool pool(2 * 64);
  for (int i = 0; i < kWarmup; ++i)
    launch(copy++);
  for (int i = 0; i < reps; ++i) {
    HIP_CHECK(hipEventRecord(pool[2 * i]));
    launch(copy++);
    HIP_CHECK(hipEventRecord(pool[2 * i + 1]));
  }
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipEventSynchronize(pool[2 * reps - 1]));
  for (int i = 0; i < reps; ++i) {
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, pool[2 * i], pool[2 * i + 1]));
    out.push_back(ms * 1000.0);
  }
}

double Percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  const double rank = p * (v.size() - 1);
  const auto lo = static_cast<std::size_t>(rank);
  const auto hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (rank - lo) * (v[hi] - v[lo]);
}

struct Item {
  std::string type;
  std::string shape;
  std::size_t batch;
  std::string kernel;
  double bytes;
  double flops;
  std::function<void(std::size_t)> launch;
  std::function<void()> prepare;
  std::vector<double> us;
};

std::size_t CopiesFor(std::size_t bytes) {
  constexpr std::size_t kFootprint = std::size_t{96} << 20;  // 3x MALL
  return std::max<std::size_t>(1, (kFootprint + bytes - 1) / bytes);
}

// Times every item of every type in `rounds` rounds; type order alternates.
void RunRounds(std::vector<std::vector<Item>>& per_type, int rounds, int reps) {
  std::size_t copy = 0;
  const std::size_t n = per_type.size();
  for (int r = 0; r < rounds; ++r) {
    for (std::size_t j = 0; j < n; ++j) {
      // Forward on even rounds, reversed on odd ones, rotated by the round.
      const std::size_t t = (r % 2 == 0) ? (j + r) % n : (n - 1 - j + r) % n;
      for (auto& item : per_type[t]) {
        if (item.prepare)
          item.prepare();
        HIP_CHECK(hipDeviceSynchronize());
        TimeLaunches(item.launch, reps, item.us, copy);
      }
    }
    std::fprintf(stderr, "round %d/%d done\n", r + 1, rounds);
  }
}

void PrintItem(const Item& item) {
  const double median = Percentile(item.us, 0.5);
  const double p95 = Percentile(item.us, 0.95);
  std::printf(
      "RESULT type=%s shape=%s batch=%zu kernel=%s bytes=%.0f n=%zu "
      "median_us=%.2f p95_us=%.2f gbps=%.1f tops=%.2f\n",
      item.type.c_str(), item.shape.c_str(), item.batch, item.kernel.c_str(),
      item.bytes, item.us.size(), median, p95, item.bytes / median / 1e3,
      item.flops / median / 1e6);
}

// ---------------------------------------------------------------- time mode

struct Measured {
  std::map<std::string, double> median_us;  // key type|shape|batch|kernel
  [[nodiscard]] double Get(const std::string& type, const std::string& shape,
                           std::size_t batch, const std::string& kernel) const {
    const auto it = median_us.find(type + "|" + shape + "|" +
                                   std::to_string(batch) + "|" + kernel);
    return it == median_us.end() ? -1.0 : it->second;
  }
};

void TimeTypes(bool output_phase, int rounds, int reps, Measured& measured) {
  std::vector<std::size_t> shapes;
  for (std::size_t s = 0; s < std::size(kShapes); ++s)
    if ((s == kOutputShape) == output_phase)
      shapes.push_back(s);

  // Activations for every (batch, K), loaded once.
  std::map<std::pair<std::size_t, std::size_t>, std::unique_ptr<Activations>>
      inputs;
  for (const auto s : shapes)
    for (const auto b : kBatches) {
      auto& slot = inputs[{b, kShapes[s].k}];
      if (slot)
        continue;
      slot = std::make_unique<Activations>(b, kShapes[s].k);
      slot->Load(RandomValues(b * kShapes[s].k,
                              static_cast<unsigned>(b * 131 + kShapes[s].k)),
                 b, kShapes[s].k, nullptr, nullptr);
    }
  std::size_t max_out = 0;
  for (const auto s : shapes)
    for (const auto b : kBatches)
      if (!(s == kOutputShape && b > 8))
        max_out = std::max(max_out, b * kShapes[s].m);
  Device<float> y(max_out);
  Device<float> y2(max_out);
  Device<float> y3(max_out);
  Device<float> y4(max_out);

  std::vector<std::unique_ptr<MappedWeights>> storage;
  std::vector<std::vector<Item>> per_type(kTypeCount);
  for (std::size_t t = 0; t < kTypeCount; ++t) {
    const TypeInfo& type = kTypes[t];
    std::map<std::size_t, const MappedWeights*> weights;
    for (const auto s : shapes) {
      const Shape& shape = kShapes[s];
      const std::size_t bytes = type.Bytes(shape.m, shape.k);
      storage.push_back(std::make_unique<MappedWeights>(
          type, shape.m, shape.k, CopiesFor(bytes),
          static_cast<unsigned>(t * 977 + s * 31)));
      weights[s] = storage.back().get();
      for (const auto batch : kBatches) {
        if (s == kOutputShape && batch > 8)
          continue;  // prefill computes logits for the last token only
        if (!type.production() && batch > 8)
          continue;  // no two-bit prefill kernel
        const Family family = Route(type, batch);
        const MappedWeights* w = weights[s];
        Activations* act = inputs.at({batch, shape.k}).get();
        Item item{type.name,
                  shape.name,
                  batch,
                  FamilyName(family),
                  static_cast<double>(bytes),
                  2.0 * batch * shape.m * shape.k,
                  [=, &y](std::size_t c) {
                    Launch(family, type, w->Copy(c), *act, y.Get(), batch,
                           shape.m, shape.k);
                  },
                  nullptr,
                  {}};
        per_type[t].push_back(std::move(item));
      }
    }
    if (output_phase || !type.production())
      continue;
    // Fused AR launches of decode_step.cpp, same weights as above.
    using namespace gufo::hip;
    const GgmlType g = type.type;
    storage.push_back(std::make_unique<MappedWeights>(
        type, 2 * kTimeStepRank, 5120, 1, static_cast<unsigned>(t * 7 + 3)));
    const MappedWeights* ab = storage.back().get();
    const MappedWeights* q = weights[2];
    const MappedWeights* kv = weights[3];
    const MappedWeights* qkv = weights[0];
    const MappedWeights* gate = weights[1];
    const MappedWeights* ffn = weights[5];
    const double ab_bytes =
        static_cast<double>(type.Bytes(kTimeStepRank, 5120));
    Activations* act = inputs.at({1, 5120}).get();
    per_type[t].push_back(Item{type.name,
                               "fused_qkv(q+k+v)",
                               1,
                               "LaunchFusedQKVProjections",
                               static_cast<double>(type.Bytes(12288, 5120) +
                                                   2 * type.Bytes(1024, 5120)),
                               2.0 * (12288 + 2048) * 5120,
                               [=, &y, &y2, &y3](std::size_t c) {
                                 LaunchFusedQKVProjections(
                                     q->Copy(c), g, kv->Copy(c), g,
                                     kv->Copy(c + 1), g, act->fp32.Get(),
                                     y.Get(), y2.Get(), y3.Get(), 12288, 1024,
                                     5120, nullptr);
                               },
                               nullptr,
                               {}});
    per_type[t].push_back(Item{
        type.name,
        "fused_ssm_in(qkv+gate+a+b)",
        1,
        "LaunchFusedSSMInputProjections",
        static_cast<double>(type.Bytes(10240, 5120) + type.Bytes(6144, 5120)) +
            2 * ab_bytes,
        2.0 * (10240 + 6144 + 2 * kTimeStepRank) * 5120,
        [=, &y, &y2, &y3, &y4](std::size_t c) {
          const auto* a = static_cast<const unsigned char*>(ab->Copy(0));
          LaunchFusedSSMInputProjections(qkv->Copy(c), g, gate->Copy(c), g, a,
                                         g, a + type.Bytes(kTimeStepRank, 5120),
                                         g, act->fp32.Get(), y.Get(), y2.Get(),
                                         y3.Get(), y4.Get(), 5120, 10240, 6144,
                                         kTimeStepRank, nullptr);
        },
        nullptr,
        {}});
    per_type[t].push_back(Item{
        type.name,
        "fused_ffn(gate+up)",
        1,
        "LaunchFusedSwiGLUGEMV",
        2.0 * type.Bytes(17408, 5120),
        2.0 * 2 * 17408 * 5120,
        [=, &y](std::size_t c) {
          LaunchFusedSwiGLUGEMV(ffn->Copy(c), g, ffn->Copy(c + 1), g,
                                act->fp32.Get(), y.Get(), 17408, 5120, nullptr);
        },
        nullptr,
        {}});
  }

  RunRounds(per_type, rounds, reps);
  for (const auto& items : per_type)
    for (const auto& item : items) {
      PrintItem(item);
      measured.median_us[item.type + "|" + item.shape + "|" +
                         std::to_string(item.batch) + "|" + item.kernel] =
          Percentile(item.us, 0.5);
    }
}

void TimeFwht(int rounds, int reps) {
  constexpr std::size_t kKs[] = {5120, 6144, 17408};
  Device<float> x(2048 * 17408);
  Device<float> signs(17408);
  const auto host = RandomValues(2048 * 17408, 7);
  HIP_CHECK(
      hipMemcpy(x.Get(), host.data(), host.size() * 4, hipMemcpyHostToDevice));
  std::vector<float> s(17408);
  for (std::size_t i = 0; i < s.size(); ++i)
    s[i] = (host[i] > 0) ? 1.0F : -1.0F;
  HIP_CHECK(
      hipMemcpy(signs.Get(), s.data(), s.size() * 4, hipMemcpyHostToDevice));
  std::vector<Item> items;
  for (const auto k : kKs)
    for (const auto batch : kBatches) {
      items.push_back(Item{"FWHT",
                           "K=" + std::to_string(k),
                           batch,
                           "Fwht128Kernel",
                           2.0 * 4 * batch * k,
                           // 7 butterfly stages + sign + scale per value
                           9.0 * batch * k,
                           [&x, &signs, batch, k](std::size_t) {
                             LaunchFwht(x.Get(), signs.Get(), batch, k);
                           },
                           nullptr,
                           {}});
    }
  std::size_t copy = 0;
  for (int r = 0; r < rounds; ++r)
    for (std::size_t j = 0; j < items.size(); ++j) {
      auto& item = items[r % 2 == 0 ? j : items.size() - 1 - j];
      HIP_CHECK(hipDeviceSynchronize());
      TimeLaunches(item.launch, reps, item.us, copy);
    }
  for (const auto& item : items) {
    const double median = Percentile(item.us, 0.5);
    std::printf(
        "FWHT K=%s batch=%zu median_us=%.2f p95_us=%.2f io_gbps=%.1f "
        "n=%zu\n",
        item.shape.c_str() + 2, item.batch, median, Percentile(item.us, 0.95),
        item.bytes / median / 1e3, item.us.size());
  }
}

void PrintEstimates(const Measured& m) {
  // Per-matrix sums: every matrix of the 27B dense token path.
  std::printf("\n# token estimates (ms per step; per token = step / batch)\n");
  for (const auto& type : kTypes) {
    for (const auto batch : kBatches) {
      if (!type.production() && batch > 8)
        continue;
      const char* kernel = FamilyName(Route(type, batch));
      double us = 0;
      double bytes = 0;
      bool complete = true;
      for (std::size_t s = 0; s < std::size(kShapes); ++s) {
        if (s == kOutputShape && batch > 8)
          continue;
        const double t = m.Get(type.name, kShapes[s].name, batch, kernel);
        if (t < 0)
          complete = false;
        us += kShapes[s].per_token * t;
        bytes += kShapes[s].per_token *
                 static_cast<double>(type.Bytes(kShapes[s].m, kShapes[s].k));
      }
      std::printf(
          "TOKEN type=%s bpw=%.4f batch=%zu path=per-matrix complete=%d "
          "weight_gb=%.3f step_ms=%.3f per_token_ms=%.4f gbps=%.1f\n",
          type.name, type.bpw(), batch, complete ? 1 : 0, bytes / 1e9, us / 1e3,
          us / 1e3 / batch, bytes / us / 1e3);
      if (batch == 1 && type.production()) {
        // decode_step.cpp: fused SSM inputs, fused Q/K/V, fused gate/up.
        const double fused =
            48 * m.Get(type.name, "fused_ssm_in(qkv+gate+a+b)", 1,
                       "LaunchFusedSSMInputProjections") +
            16 * m.Get(type.name, "fused_qkv(q+k+v)", 1,
                       "LaunchFusedQKVProjections") +
            64 * m.Get(type.name, "fused_ffn(gate+up)", 1,
                       "LaunchFusedSwiGLUGEMV") +
            64 * m.Get(type.name, "ssm_out_attn_out", 1, kernel) +
            64 * m.Get(type.name, "ffn_down", 1, kernel) +
            m.Get(type.name, "output", 1, kernel);
        const double fused_bytes =
            bytes + 96.0 * type.Bytes(kTimeStepRank, 5120);
        std::printf(
            "TOKEN type=%s bpw=%.4f batch=1 path=decode-fused complete=1 "
            "weight_gb=%.3f step_ms=%.3f per_token_ms=%.4f gbps=%.1f\n",
            type.name, type.bpw(), fused_bytes / 1e9, fused / 1e3, fused / 1e3,
            fused_bytes / fused / 1e3);
      }
    }
  }
}

// ---------------------------------------------------------------- check mode

void DequantizeHost(const TypeInfo& type, const void* row, float* out,
                    std::size_t k) {
  using namespace gufo::quant;
  switch (type.type) {
    case GgmlType::kQ8_0:
      DequantizeQ8_0(row, out, k);
      return;
    case GgmlType::kQ6_K:
      DequantizeQ6_K(row, out, k);
      return;
    case GgmlType::kQ5_K:
      DequantizeQ5_K(row, out, k);
      return;
    case GgmlType::kQ4_K:
      DequantizeQ4_K(row, out, k);
      return;
    case GgmlType::kIQ4_XS:
      DequantizeIQ4_XS(row, out, k);
      return;
    case GgmlType::kIQ4_NL:
      DequantizeIQ4_NL(row, out, k);
      return;
    case GgmlType::kQ3_K:
      DequantizeQ3_K(row, out, k);
      return;
    case GgmlType::kIQ3_S:
      DequantizeIQ3_S(row, out, k);
      return;
    default:
      throw std::logic_error("no host dequantizer");
  }
}

// Worst |gpu - ref| / sum|w x| over the first and last 32 output rows of
// activation rows 0, 1 and batch - 1. Returns false on a failure.
bool CheckOne(const TypeInfo& type, const Shape& shape, std::size_t batch,
              double tolerance) {
  const Family family = Route(type, batch);
  const std::size_t m = shape.m;
  const std::size_t k = shape.k;
  MappedWeights w(type, m, k, 1, 12345U);
  Activations act(batch, k);
  const auto x = RandomValues(batch * k, 99);
  std::vector<float> xb;
  std::vector<float> xh;
  act.Load(x, batch, k, &xb, &xh);
  const std::vector<float>& xr = family == Family::kPrefillQ8     ? xb
                                 : family == Family::kPrefillFp16 ? xh
                                                                  : x;
  Device<float> y(batch * m);
  HIP_CHECK(hipMemset(y.Get(), 0xff, batch * m * 4));
  Launch(family, type, w.Copy(0), act, y.Get(), batch, m, k);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> out(batch * m);
  HIP_CHECK(
      hipMemcpy(out.data(), y.Get(), out.size() * 4, hipMemcpyDeviceToHost));

  std::vector<std::size_t> rows;
  for (std::size_t r = 0; r < 32; ++r) {
    rows.push_back(r);
    rows.push_back(m - 1 - r);
  }
  std::vector<float> dq(rows.size() * k);
  if (type.production()) {
    const std::size_t row_bytes = type.Bytes(1, k);
    for (std::size_t i = 0; i < rows.size(); ++i)
      DequantizeHost(
          type,
          static_cast<const unsigned char*>(w.Host(0)) + rows[i] * row_bytes,
          dq.data() + i * k, k);
  } else {
    Device<float> full(m * k);
    rq_mmvq::Dequantize(static_cast<rq_mmvq::Type>(type.mmvq), w.Copy(0),
                        full.Get(), m, k, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    for (std::size_t i = 0; i < rows.size(); ++i)
      HIP_CHECK(hipMemcpy(dq.data() + i * k, full.Get() + rows[i] * k, k * 4,
                          hipMemcpyDeviceToHost));
  }
  double worst = 0;
  bool finite = true;
  const std::size_t checks[] = {0, std::min<std::size_t>(1, batch - 1),
                                batch - 1};
  for (const auto r : checks)
    for (std::size_t i = 0; i < rows.size(); ++i) {
      double dot = 0;
      double mag = 0;
      for (std::size_t j = 0; j < k; ++j) {
        const double p = static_cast<double>(dq[i * k + j]) * xr[r * k + j];
        dot += p;
        mag += std::fabs(p);
      }
      const float got = out[r * m + rows[i]];
      finite = finite && std::isfinite(got) && std::isfinite(dot);
      worst = std::max(worst, std::fabs(got - dot) / std::max(mag, 1e-30));
    }
  const bool ok = finite && worst <= tolerance;
  std::printf(
      "CHECK %s type=%s shape=%s batch=%zu kernel=%s rel_err=%.3e "
      "tolerance=%.0e\n",
      ok ? "ok  " : "FAIL", type.name, shape.name, batch, FamilyName(family),
      worst, tolerance);
  return ok;
}

bool CheckFwht() {
  bool ok = true;
  for (const std::size_t k : {5120, 6144, 17408}) {
    const std::size_t batch = 3;
    auto x = RandomValues(batch * k, static_cast<unsigned>(k));
    const auto sign_source = RandomValues(k, 5);
    std::vector<float> s(k);
    for (std::size_t i = 0; i < k; ++i)
      s[i] = sign_source[i] > 0 ? 1.0F : -1.0F;
    Device<float> dx(batch * k);
    Device<float> ds(k);
    HIP_CHECK(
        hipMemcpy(dx.Get(), x.data(), x.size() * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ds.Get(), s.data(), k * 4, hipMemcpyHostToDevice));
    LaunchFwht(dx.Get(), ds.Get(), batch, k);
    HIP_CHECK(hipGetLastError());
    std::vector<float> got(x.size());
    HIP_CHECK(
        hipMemcpy(got.data(), dx.Get(), got.size() * 4, hipMemcpyDeviceToHost));
    double worst = 0;
    double amax = 0;
    for (std::size_t r = 0; r < batch; ++r)
      for (std::size_t c = 0; c < k; c += 128)
        for (std::size_t i = 0; i < 128; ++i) {
          double ref = 0;
          for (std::size_t j = 0; j < 128; ++j) {
            const double h =
                (__builtin_popcount(static_cast<unsigned>(i & j)) & 1) ? -1.0
                                                                       : 1.0;
            ref += h * s[c + j] * x[r * k + c + j];
          }
          ref /= std::sqrt(128.0);
          worst = std::max(worst, std::fabs(ref - got[r * k + c + i]));
          amax = std::max(amax, std::fabs(ref));
        }
    // Orthogonality: applying H D twice with D = 1 must restore the input.
    std::vector<float> ones(k, 1.0F);
    HIP_CHECK(
        hipMemcpy(dx.Get(), x.data(), x.size() * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ds.Get(), ones.data(), k * 4, hipMemcpyHostToDevice));
    LaunchFwht(dx.Get(), ds.Get(), batch, k);
    LaunchFwht(dx.Get(), ds.Get(), batch, k);
    HIP_CHECK(
        hipMemcpy(got.data(), dx.Get(), got.size() * 4, hipMemcpyDeviceToHost));
    double round_trip = 0;
    for (std::size_t i = 0; i < x.size(); ++i)
      round_trip = std::max(round_trip, std::fabs(double(got[i]) - x[i]));
    const bool pass = worst <= 1e-5 * amax && round_trip <= 1e-5 * amax;
    std::printf(
        "CHECK %s FWHT K=%zu batch=%zu max_abs_err=%.3e max_abs_ref=%.3e "
        "round_trip=%.3e\n",
        pass ? "ok  " : "FAIL", k, batch, worst, amax, round_trip);
    ok = ok && pass;
  }
  return ok;
}

int RunCheck() {
  bool ok = CheckFwht();
  const Shape& gate = kShapes[1];
  const Shape& down = kShapes[6];
  for (const auto& type : kTypes) {
    const double fp32_tol = 1e-4;
    const double q8_tol = 1e-2;
    const double fp16_tol = 3e-3;
    for (const auto batch : kBatches) {
      if (!type.production() && batch > 8)
        continue;
      const Family f = Route(type, batch);
      const double tol = f == Family::kAr || f == Family::kVerify ? fp32_tol
                         : f == Family::kPrefillFp16              ? fp16_tol
                                                                  : q8_tol;
      ok = CheckOne(type, gate, batch, tol) && ok;
      if (batch <= 8)
        ok = CheckOne(type, down, batch, tol) && ok;
    }
  }
  std::printf("CHECK %s\n", ok ? "PASS" : "FAILED");
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "time";
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("# device %s %s, CUs %d\n", prop.name, prop.gcnArchName,
                prop.multiProcessorCount);
    if (mode == "check")
      return RunCheck();
    const int rounds = argc > 2 ? std::atoi(argv[2]) : 4;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 8;
    if (rounds < 1 || reps < 1 || reps > 64)
      throw std::invalid_argument("rounds >= 1, 1 <= reps <= 64");
    if (mode == "fwht") {
      TimeFwht(rounds, reps);
      return 0;
    }
    if (mode != "time")
      throw std::invalid_argument("mode: check | time [rounds] [reps] | fwht");
    std::printf("# rounds %d, reps %d per round, warm-up %d per item\n", rounds,
                reps, kWarmup);
    Measured measured;
    TimeTypes(false, rounds, reps, measured);
    TimeTypes(true, rounds, reps, measured);
    TimeFwht(rounds, reps);
    PrintEstimates(measured);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
