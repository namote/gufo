#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "src/core/hip/detail/exl3_gemv.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/exl3_moe.hpp"

namespace gpu = gufo::models::qwen38_flash_next::rocm;
namespace ex = gufo::quant::exl3;
namespace ar = gufo::hip::exl3;
namespace {
void Require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}
struct Buffers {
  std::vector<void*> allocations;
  ~Buffers() {
    for (auto* p : allocations)
      (void)hipFree(p);
  }
  template<class T>
  T* Make(std::size_t count) {
    T* p = nullptr;
    Check(hipMalloc(&p, count * sizeof(T)));
    allocations.push_back(p);
    Check(hipMemset(p, 0, count * sizeof(T)));
    return p;
  }
  template<class T>
  T* Copy(const std::vector<T>& data) {
    auto* p = Make<T>(data.size());
    Check(hipMemcpy(p, data.data(), data.size() * sizeof(T),
                    hipMemcpyHostToDevice));
    return p;
  }
};
template<class T>
std::vector<T> Read(const T* pointer, std::size_t count) {
  std::vector<T> data(count);
  Check(hipMemcpy(data.data(), pointer, count * sizeof(T),
                  hipMemcpyDeviceToHost));
  return data;
}
void Near(std::span<const float> actual, std::span<const float> expected,
          const char* message) {
  Require(actual.size() == expected.size(), "comparison size mismatch");
  double error = 0, energy = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    Require(std::isfinite(actual[i]) && std::isfinite(expected[i]),
            "nonfinite stage");
    const double delta = double(actual[i]) - expected[i];
    error += delta * delta;
    energy += double(expected[i]) * expected[i];
  }
  const double relative =
      energy == 0 ? std::sqrt(error) : std::sqrt(error / energy);
  if (relative > 1e-5)
    std::cerr << message << " relative L2=" << relative << '\n';
  Require(relative <= 1e-5, message);
}

struct Bank {
  unsigned k, n, experts;
  std::vector<std::uint16_t> trellis, suh, svh, inner;
  gpu::DeviceTensor weight;
  Bank(Buffers& buffers, unsigned inputs, unsigned outputs, unsigned count,
       unsigned seed, bool decode = true)
      : k(inputs), n(outputs), experts(count) {
    std::mt19937 rng(seed);
    const auto words = std::size_t{k / 16} * (n / 16) * 80;
    trellis.resize(words * experts);
    suh.resize(k * experts);
    svh.resize(n * experts);
    if (decode)
      inner.resize(std::size_t{k} * n * experts);
    for (auto& v : trellis)
      v = rng();
    for (auto& v : suh)
      v = ex::HalfFromDouble((rng() & 1 ? 1 : -1) *
                             (0.03 + (rng() % 100) * 0.001));
    for (auto& v : svh)
      v = ex::HalfFromDouble((rng() & 1 ? 1 : -1) *
                             (0.03 + (rng() % 100) * 0.001));
    auto* dt = buffers.Copy(trellis);
    auto* ds = buffers.Copy(suh);
    auto* dv = buffers.Copy(svh);
    std::vector<ex::DeviceLinear> descriptors;
    for (unsigned e = 0; e < experts; ++e) {
      descriptors.push_back({dt + e * words,
                             ds + e * k,
                             dv + e * n,
                             k,
                             n,
                             {5, false},
                             ex::Codebook::kMul1});
      if (decode)
        ex::DequantInner(trellis.data() + e * words, k, n, {5, false},
                         ex::Codebook::kMul1,
                         inner.data() + std::size_t{e} * k * n);
    }
    weight.cols = k;
    weight.rows = n;
    weight.experts = experts;
    weight.device_exl3 = buffers.Copy(descriptors);
  }
  std::vector<std::uint16_t> Rotate(unsigned expert, const float* input) const {
    std::vector<std::uint16_t> result(k);
    for (unsigned base = 0; base < k; base += 128) {
      float v[128];
      for (unsigned i = 0; i < 128; ++i) {
        volatile float product =
            input[base + i] *
            float(ex::HalfToDouble(suh[expert * k + base + i]));
        v[i] = product;
      }
      for (unsigned length = 1; length < 128; length *= 2)
        for (unsigned first = 0; first < 128; first += 2 * length)
          for (unsigned i = first; i < first + length; ++i) {
            const float a = v[i], b = v[i + length];
            v[i] = a + b;
            v[i + length] = a - b;
          }
      for (unsigned i = 0; i < 128; ++i) {
        volatile float scaled = v[i] * 0.0883883461356163F;
        result[base + i] = ex::HalfFromDouble(scaled);
      }
    }
    return result;
  }
  std::vector<float> Dot(unsigned expert, const std::uint16_t* input) const {
    std::vector<float> result(n);
    for (unsigned j = 0; j < n; ++j) {
      double sum = 0;
      for (unsigned i = 0; i < k; ++i)
        sum += ex::HalfToDouble(input[i]) *
               ex::HalfToDouble(inner[(std::size_t{expert} * k + i) * n + j]);
      result[j] = sum;
    }
    return result;
  }
};

// Frozen pre-change producer: restore/SwiGLU writes FP32, and only a
// subsequent launch rotates it. Compare exact bytes, not an ideal FP64
// activation that can cross a different FP16 down-input bin.
__global__ void UnfusedActivation(const ex::DeviceLinear* gate,
                                  const ex::DeviceLinear* up,
                                  const std::int32_t* ids, const float* gi,
                                  const float* ui, float* activation,
                                  unsigned width) {
  const unsigned row = blockIdx.y, lane = threadIdx.x;
  const unsigned first = blockIdx.x * 128 + lane * 4;
  const auto offset = std::size_t{row} * width + first;
  const auto* gs = reinterpret_cast<const _Float16*>(gate[ids[row]].svh);
  const auto* us = reinterpret_cast<const _Float16*>(up[ids[row]].svh);
  float g[4], u[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    g[j] = gi[offset + j];
    u[j] = ui[offset + j];
  }
  ar::Butterfly128(g, lane);
  ar::Butterfly128(u, lane);
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const float value = (g[j] * ar::kHadamardScale) * float(gs[first + j]);
    const float other = (u[j] * ar::kHadamardScale) * float(us[first + j]);
    activation[offset + j] = (value / (1.0F + expf(-value))) * other;
  }
}

void FusedProducer(unsigned rows, bool shared) {
  constexpr unsigned k = 256, ff = 640, used = 3, experts = 12;
  const unsigned assignments = rows * used;
  Buffers buffers;
  Bank gate(buffers, k, ff, experts, 1), up(buffers, k, ff, experts, 2),
      down(buffers, ff, k, experts, 3);
  gpu::DeviceLayer layer;
  layer.ffn_gate_exps = gate.weight;
  layer.ffn_up_exps = up.weight;
  layer.ffn_down_exps = down.weight;
  std::vector<float> input(rows * k);
  std::mt19937 rng(4);
  for (auto& v : input)
    v = (int(rng() % 2000) - 1000) * 0.00003125F;
  // Include a genuinely zero token alongside small unsaturated activations.
  std::fill_n(input.begin(), k, 0.0F);
  std::vector<std::int32_t> ids(assignments);
  for (unsigned i = 0; i < assignments; ++i)
    ids[i] = shared ? i % used : i % experts;
  auto* dx = buffers.Copy(input);
  auto* di = buffers.Copy(ids);
  auto* g = buffers.Make<float>(assignments * ff);
  auto* u = buffers.Make<float>(assignments * ff);
  auto* y = buffers.Make<float>(assignments * k);
  gpu::Exl3Scratch s;
  s.tiles = buffers.Make<gpu::Exl3RouteTile>(assignments);
  s.tile_count = buffers.Make<std::uint32_t>(1);
  s.down_exponents = buffers.Make<std::int32_t>(assignments);
  s.routed_input =
      buffers.Make<std::uint16_t>(assignments * std::max(2 * k, ff));
  gpu::Exl3RoutedTrace trace;
  trace.gate_up_input = buffers.Make<std::uint16_t>(2 * assignments * k);
  trace.gate_inner = buffers.Make<float>(assignments * ff);
  trace.up_inner = buffers.Make<float>(assignments * ff);
  trace.activation = buffers.Make<float>(assignments * ff);
  trace.down_input = buffers.Make<std::uint16_t>(assignments * ff);
  trace.down_inner = buffers.Make<float>(assignments * k);
  trace.down_widened_inner = buffers.Make<float>(assignments * k);
  gpu::Exl3RoutedMlp(layer, dx, di, g, u, y, rows, used, s, nullptr, &trace);
  const auto actual = Read(y, assignments * k);
  const auto frozen = Read(trace.down_input, assignments * ff);
  const auto down_inner = Read(trace.down_inner, assignments * k);
  Near(down_inner, Read(trace.down_widened_inner, assignments * k),
       "scaled down vs widened control");
  auto* unfused = buffers.Make<float>(assignments * ff);
  hipLaunchKernelGGL(UnfusedActivation, dim3(ff / 128, assignments), dim3(32),
                     0, nullptr, gate.weight.device_exl3, up.weight.device_exl3,
                     di, trace.gate_inner, trace.up_inner, unfused, ff);
  Check(hipGetLastError());
  const auto old_activation = Read(unfused, assignments * ff);
  const auto activation = Read(trace.activation, assignments * ff);
  Require(std::memcmp(old_activation.data(), activation.data(),
                      activation.size() * sizeof(float)) == 0,
          "fused activation changed FP32 bytes");
  const auto gu = Read(trace.gate_up_input, 2 * assignments * k);
  const auto gi = Read(trace.gate_inner, assignments * ff);
  const auto ui = Read(trace.up_inner, assignments * ff);
  for (unsigned slot = 0; slot < assignments; ++slot) {
    const unsigned expert = ids[slot];
    const auto gr = gate.Rotate(expert, input.data() + (slot / used) * k);
    const auto ur = up.Rotate(expert, input.data() + (slot / used) * k);
    Require(std::equal(gr.begin(), gr.end(), gu.begin() + slot * k),
            "gate producer bits");
    Require(
        std::equal(ur.begin(), ur.end(), gu.begin() + (assignments + slot) * k),
        "up producer bits");
    Near(std::span(gi).subspan(slot * ff, ff), gate.Dot(expert, gr.data()),
         "gate FP64 stage");
    Near(std::span(ui).subspan(slot * ff, ff), up.Dot(expert, ur.data()),
         "up FP64 stage");
    const auto rotated = down.Rotate(expert, old_activation.data() + slot * ff);
    Require(
        std::equal(rotated.begin(), rotated.end(), frozen.begin() + slot * ff),
        "fused down producer changed FP16 bits");
    const auto reference = down.Dot(expert, frozen.data() + slot * ff);
    Near(std::span(down_inner).subspan(slot * k, k), reference,
         "down FP64 frozen-input stage");
    std::vector<double> restored(reference.begin(), reference.end());
    ex::HadamardBlocks(std::span<double>(restored));
    std::vector<float> expected(k);
    for (unsigned j = 0; j < k; ++j)
      expected[j] = restored[j] * ex::HalfToDouble(down.svh[expert * k + j]);
    Near(std::span(actual).subspan(slot * k, k), expected,
         "routed output FP64 frozen-input contract");
  }
  // Capture the complete producer + scale + dot path. Input is regenerated
  // on each replay; the scaled scratch must never be scaled twice.
  if (rows == 3) {
    hipStream_t stream;
    Check(hipStreamCreate(&stream));
    Check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    gpu::Exl3RoutedMlp(layer, dx, di, g, u, y, rows, used, s, stream);
    hipGraph_t graph;
    Check(hipStreamEndCapture(stream, &graph));
    hipGraphExec_t executable;
    Check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (unsigned replay = 0; replay < 2; ++replay) {
      Check(hipGraphLaunch(executable, stream));
      Check(hipStreamSynchronize(stream));
      Require(actual == Read(y, assignments * k),
              "scaled routed graph replay changed output");
    }
    Check(hipGraphExecDestroy(executable));
    Check(hipGraphDestroy(graph));
    Check(hipStreamDestroy(stream));
  }
}

void DownFixtures() {
  constexpr unsigned rows = 8, k = 640, n = 128;
  Buffers buffers;
  Bank down(buffers, k, n, 1, 5);
  std::vector<std::uint16_t> input(rows * k);
  for (unsigned j = 0; j < k; ++j) {
    input[j] = j & 1 ? 0x8000 : 0;            // signed zero
    input[k + j] = (j & 1 ? 0x8000 : 0) | 1;  // smallest subnormal
    input[2 * k + j] = (j & 1 ? 0x8000 : 0) | (1 + j % 1023);
    input[3 * k + j] = (j & 1 ? 0x8000 : 0) | 0x0400;  // smallest normal
    input[4 * k + j] = j == 0 ? 1 : 0x7bff;            // no overflow headroom
    input[5 * k + j] = j == 0 ? 1 : 0x53ff;            // exact headroom edge
    input[6 * k + j] = j == 0 ? 1 : 0x5400;            // one exponent too high
    input[7 * k + j] = j % 3 == 0 ? 0x8000 : 0x3c00;
  }
  input[3 * k - 1] = 0x83ff;  // largest negative subnormal
  gpu::Exl3RouteTile tile{};
  tile.count = rows;
  for (unsigned r = 0; r < rows; ++r)
    tile.slots[r] = r;
  gpu::Exl3Scratch s;
  s.tiles = buffers.Copy(std::vector{tile});
  s.tile_count = buffers.Copy(std::vector<std::uint32_t>{1});
  auto* original = buffers.Copy(input);
  auto* scaled = buffers.Copy(input);
  auto* control = buffers.Make<float>(rows * n);
  auto* output = buffers.Make<float>(rows * n);
  gpu::Exl3RoutedDown(down.weight, original, control, rows, 8, s, nullptr);
  s.down_exponents = buffers.Make<std::int32_t>(rows);
  gpu::Exl3RoutedDown(down.weight, scaled, output, rows, 8, s, nullptr);
  const auto exponents = Read(s.down_exponents, rows);
  Require(exponents == std::vector<std::int32_t>({0, 10, 10, 0, -1, 10, -1, 0}),
          "down exponent classification");
  const auto packed = Read(scaled, rows * k);
  const auto actual = Read(output, rows * n);
  const auto widened = Read(control, rows * n);
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned j = 0; j < k; ++j) {
      const auto index = r * k + j;
      if (exponents[r] <= 0)
        Require(packed[index] == input[index],
                "fallback or zero row was mutated");
      else {
        Require((packed[index] & 0x7fff) == 0 ||
                    ((packed[index] & 0x7fff) >= 0x400 &&
                     (packed[index] & 0x7fff) < 0x7c00),
                "scaled row retains a subnormal or overflows");
        const auto restored = ex::HalfFromDouble(
            std::ldexp(ex::HalfToDouble(packed[index]), -exponents[r]));
        Require(restored == input[index],
                "power-of-two scaling changed producer bits");
      }
    }
    Near(std::span(actual).subspan(r * n, n), down.Dot(0, input.data() + r * k),
         "down fixture FP64");
    Near(std::span(actual).subspan(r * n, n),
         std::span(widened).subspan(r * n, n), "down fixture widened control");
    if (exponents[r] < 0)
      Require(std::equal(actual.begin() + r * n, actual.begin() + (r + 1) * n,
                         widened.begin() + r * n),
              "widened fallback changed bits");
  }
}

// Deliberately simple raw-inner adapter to qualify our dispatch/producer/
// fallback/restore independently of X3. The oracle below is independent FP64;
// this adapter is not a substitute for W3's actual WMMA unit qualification.
__global__ void ControlBucketDots(const std::uint16_t* weights,
                                  const gpu::Exl3ExpertBucket* buckets,
                                  const std::uint32_t* count,
                                  const std::uint16_t* input, float* output,
                                  unsigned k, unsigned n) {
  if (blockIdx.x >= *count)
    return;
  const auto bucket = buckets[blockIdx.x];
  for (unsigned i = threadIdx.x; i < bucket.count * n; i += blockDim.x) {
    const unsigned row = bucket.first + i / n, column = i % n;
    float sum = 0;
    for (unsigned j = 0; j < k; ++j) {
      float w = static_cast<float>(reinterpret_cast<const _Float16*>(
          weights)[(std::size_t{bucket.expert} * k + j) * n + column]);
      float x = static_cast<float>(
          reinterpret_cast<const _Float16*>(input)[std::size_t{row} * k + j]);
      asm volatile("" : "+v"(w), "+v"(x));
      sum = __builtin_fmaf(w, x, sum);
    }
    output[std::size_t{row} * n + column] = sum;
  }
}
const ex::DeviceLinear* control_banks[3];
const std::uint16_t* control_weights[3];
unsigned control_calls = 0;
void ControlGemm(const ex::DeviceLinear* bank,
                 const gpu::Exl3ExpertBucket* buckets,
                 const std::uint32_t* count, std::uint32_t capacity,
                 std::uint32_t tile_rows, const std::uint16_t* input,
                 float* output, std::uint32_t k, std::uint32_t n,
                 hipStream_t stream) {
  Require(tile_rows == 16, "adapter tile row contract");
  unsigned index = 0;
  while (index < 3 && control_banks[index] != bank)
    ++index;
  Require(index < 3, "adapter expert bank contract");
  ++control_calls;
  hipLaunchKernelGGL(ControlBucketDots, dim3(capacity), dim3(128), 0, stream,
                     control_weights[index], buckets, count, input, output, k,
                     n);
  Check(hipGetLastError());
}

void PrefillDispatch() {
  constexpr unsigned rows = 17, used = 2, experts = 4, k = 128, ff = 640;
  constexpr unsigned assignments = rows * used, tile_rows = 16;
  Buffers buffers;
  Bank gate(buffers, k, ff, experts, 9), up(buffers, k, ff, experts, 10),
      down(buffers, ff, k, experts, 11);
  gpu::DeviceLayer layer;
  layer.ffn_gate_exps = gate.weight;
  layer.ffn_up_exps = up.weight;
  layer.ffn_down_exps = down.weight;
  control_banks[0] = gate.weight.device_exl3;
  control_banks[1] = up.weight.device_exl3;
  control_banks[2] = down.weight.device_exl3;
  control_weights[0] = buffers.Copy(gate.inner);
  control_weights[1] = buffers.Copy(up.inner);
  control_weights[2] = buffers.Copy(down.inner);
  control_calls = 0;
  std::vector<float> input(rows * k);
  std::vector<std::int32_t> ids(assignments);
  for (unsigned i = 0; i < input.size(); ++i)
    input[i] = (int(i % 97) - 48) * 0.00003125F;
  for (unsigned i = 0; i < assignments; ++i)
    ids[i] = i % experts;
  gpu::Exl3PrefillScratch s;
  s.max_assignments = assignments;
  s.max_buckets = experts + assignments / tile_rows;
  s.offsets = buffers.Make<std::uint32_t>(experts + 1);
  s.slots = buffers.Make<std::int32_t>(assignments);
  s.buckets = buffers.Make<gpu::Exl3ExpertBucket>(s.max_buckets);
  s.bucket_count = buffers.Make<std::uint32_t>(1);
  s.input = buffers.Make<std::uint16_t>(assignments * ff);
  s.gate = buffers.Make<float>(assignments * ff);
  s.up = buffers.Make<float>(assignments * ff);
  s.out = buffers.Make<float>(assignments * k);
  s.down_exponents = buffers.Make<std::int32_t>(assignments);
  auto* dx = buffers.Copy(input);
  auto* di = buffers.Copy(ids);
  auto* y = buffers.Make<float>(assignments * k);
  gpu::Exl3PrefillExperts(layer, dx, di, y, rows, used, tile_rows, s,
                          ControlGemm, nullptr);
  Require(control_calls == 3,
          "prefill must dispatch three bounded bank launches");
  const auto map = Read(s.slots, assignments);
  const auto activation = Read(s.gate, assignments * ff);
  const auto scaled = Read(s.input, assignments * ff);
  const auto exponents = Read(s.down_exponents, assignments);
  const auto actual = Read(y, assignments * k);
  for (unsigned packed = 0; packed < assignments; ++packed) {
    const unsigned slot = map[packed], expert = ids[slot];
    const auto rotated = down.Rotate(expert, activation.data() + packed * ff);
    for (unsigned j = 0; j < ff; ++j) {
      const auto bits = scaled[packed * ff + j];
      const auto restored =
          exponents[packed] > 0
              ? ex::HalfFromDouble(
                    std::ldexp(ex::HalfToDouble(bits), -exponents[packed]))
              : bits;
      Require(restored == rotated[j], "prefill down producer bits");
    }
    const auto dot = down.Dot(expert, rotated.data());
    std::vector<double> restored(dot.begin(), dot.end());
    ex::HadamardBlocks(std::span<double>(restored));
    std::vector<float> expected(k);
    for (unsigned j = 0; j < k; ++j)
      expected[j] = restored[j] * ex::HalfToDouble(down.svh[expert * k + j]);
    Near(std::span(actual).subspan(slot * k, k), expected,
         "bucket prefill frozen-input FP64 output");
  }
  hipStream_t stream;
  Check(hipStreamCreate(&stream));
  Check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
  gpu::Exl3PrefillExperts(layer, dx, di, y, rows, used, tile_rows, s,
                          ControlGemm, stream);
  hipGraph_t graph;
  Check(hipStreamEndCapture(stream, &graph));
  hipGraphExec_t executable;
  Check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  for (unsigned replay = 0; replay < 2; ++replay) {
    Check(hipGraphLaunch(executable, stream));
    Check(hipStreamSynchronize(stream));
    const auto repeated = Read(y, assignments * k);
    Require(std::memcmp(actual.data(), repeated.data(),
                        actual.size() * sizeof(float)) == 0,
            "bucket prefill graph replay changed output");
  }
  Check(hipGraphExecDestroy(executable));
  Check(hipGraphDestroy(graph));
  Check(hipStreamDestroy(stream));
}

void Buckets(unsigned rows, bool shared, unsigned tile_rows) {
  constexpr unsigned used = 3, experts = 12, width = 640;
  const unsigned assignments = rows * used;
  const unsigned capacity =
      std::min(assignments, experts + assignments / tile_rows);
  Buffers buffers;
  Bank bank(buffers, width, 128, experts, 6, false);
  std::vector<std::int32_t> ids(assignments);
  for (unsigned i = 0; i < assignments; ++i)
    ids[i] = shared ? i % used : (i * 5 + 3) % experts;
  auto* di = buffers.Copy(ids);
  auto* offsets = buffers.Make<std::uint32_t>(experts + 1);
  auto* slots = buffers.Make<std::int32_t>(assignments);
  auto* descriptors = buffers.Make<gpu::Exl3ExpertBucket>(capacity + 1);
  auto* count = buffers.Make<std::uint32_t>(1);
  const gpu::Exl3ExpertBucket canary{0xdeadbeefU, 0xdeadbeefU, 0xdeadbeefU};
  Check(hipMemcpy(descriptors + capacity, &canary, sizeof(canary),
                  hipMemcpyHostToDevice));
  const auto build = [&] {
    gpu::Exl3BuildExpertBuckets(di, assignments, experts, tile_rows, offsets,
                                slots, descriptors, count, capacity, nullptr);
  };
  build();
  const auto bounds = Read(offsets, experts + 1);
  const auto map = Read(slots, assignments);
  const auto buckets = Read(descriptors, capacity + 1);
  const unsigned active = Read(count, 1)[0];
  Require(bounds.front() == 0 && bounds.back() == assignments,
          "bucket prefix bounds");
  Require(buckets.back().expert == canary.expert &&
              buckets.back().count == canary.count &&
              buckets.back().first == canary.first,
          "descriptor capacity canary");
  unsigned next = 0;
  for (unsigned e = 0; e < experts; ++e) {
    unsigned cursor = bounds[e];
    for (unsigned slot = 0; slot < assignments; ++slot)
      if (ids[slot] == int(e)) {
        Require(map[cursor] == int(slot), "bucket assignment order");
        ++cursor;
      }
    Require(cursor == bounds[e + 1], "bucket expert count");
    for (unsigned row = bounds[e]; row < bounds[e + 1]; row += tile_rows) {
      Require(
          next < active && buckets[next].expert == e &&
              buckets[next].first == row &&
              buckets[next].count == std::min(tile_rows, bounds[e + 1] - row),
          "bucket ragged tile");
      ++next;
    }
  }
  Require(next == active, "bucket descriptor count");
  for (unsigned i = active; i < capacity; ++i)
    Require(buckets[i].count == 0, "inactive bucket not cleared");
  std::vector<float> input(rows * width), down_input(assignments * width);
  for (unsigned i = 0; i < input.size(); ++i)
    input[i] = (int(i % 97) - 48) * 0.00003125F;
  for (unsigned i = 0; i < down_input.size(); ++i)
    down_input[i] = (int(i % 61) - 30) * 0.00000003125F;
  auto* dx = buffers.Copy(input);
  auto* dd = buffers.Copy(down_input);
  auto* rotated = buffers.Make<std::uint16_t>(assignments * width);
  for (bool down : {false, true}) {
    gpu::Exl3RotateExpertBuckets(bank.weight.device_exl3, down ? dd : dx,
                                 rotated, offsets, slots, experts, assignments,
                                 width, used, down, nullptr);
    const auto actual = Read(rotated, assignments * width);
    for (unsigned packed = 0; packed < assignments; ++packed) {
      const unsigned slot = map[packed];
      const auto expected =
          bank.Rotate(ids[slot], down ? down_input.data() + slot * width
                                      : input.data() + (slot / used) * width);
      Require(std::equal(expected.begin(), expected.end(),
                         actual.begin() + packed * width),
              "bucket gather/rotation producer bits");
    }
  }
  auto* scattered = buffers.Make<float>(assignments * width);
  gpu::Exl3ScatterExpertBuckets(dd, slots, scattered, assignments, width,
                                nullptr);
  const auto output = Read(scattered, assignments * width);
  for (unsigned packed = 0; packed < assignments; ++packed)
    Require(std::equal(down_input.begin() + packed * width,
                       down_input.begin() + (packed + 1) * width,
                       output.begin() + map[packed] * width),
            "bucket scatter slot identity");
  // Replay with fewer assignments: stale active descriptors must be cleared.
  gpu::Exl3BuildExpertBuckets(di, 0, experts, tile_rows, offsets, slots,
                              descriptors, count, capacity, nullptr);
  Require(Read(count, 1)[0] == 0, "zero-row bucket count");
  for (const auto& bucket : Read(descriptors, capacity))
    Require(bucket.count == 0, "zero-row stale descriptor");
}
}  // namespace

int main() {
  try {
    DownFixtures();
    PrefillDispatch();
    for (unsigned rows : {1U, 2U, 3U, 4U, 7U, 8U, 9U})
      for (bool shared : {false, true})
        FusedProducer(rows, shared);
    for (unsigned tile_rows : {16U, 32U, 64U})
      for (unsigned rows : {1U, 3U, 17U, 33U, 65U})
        for (bool shared : {false, true})
          Buckets(rows, shared, tile_rows);
    // No zero-dimensional GPU launches, and no scratch accesses, for no work.
    gpu::Exl3RoutedDown({}, nullptr, nullptr, 0, 0, {}, nullptr);
    gpu::Exl3RoutedMlp({}, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 10,
                       {}, nullptr);
    gpu::Exl3PrefillExperts({}, nullptr, nullptr, nullptr, 0, 10, 16, {},
                            nullptr, nullptr);
    std::cout << "Flash-Next EXL3 expert kernel contracts passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
