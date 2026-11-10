#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/hip/detail/exl3_gemv.hpp"
#include "src/core/quant/exl3/coverage.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/exl3_moe.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "tests/models/qwen38_flash_next/exl3_routed_reference.hpp"

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

// The codebook of every Bank a test builds (--format-matrix sets it per
// format; mul1 otherwise).
ex::Codebook bank_codebook = ex::Codebook::kMul1;

struct Bank {
  unsigned k, n, experts;
  ex::BitRate rate;
  ex::Codebook codebook = bank_codebook;
  std::vector<ex::DeviceLinear> descriptors;
  std::vector<std::uint16_t> trellis, suh, svh, inner;
  gpu::DeviceTensor weight;
  Bank(Buffers& buffers, unsigned inputs, unsigned outputs, unsigned count,
       unsigned seed, bool decode = true, int bits = 5, bool half = false)
      : k(inputs), n(outputs), experts(count), rate{bits, half} {
    std::mt19937 rng(seed);
    const auto words = std::size_t{k / 16} * (n / 16) * rate.TileWords();
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
    for (unsigned e = 0; e < experts; ++e) {
      descriptors.push_back(
          {dt + e * words, ds + e * k, dv + e * n, k, n, rate, codebook});
      if (decode)
        ex::DequantInner(trellis.data() + e * words, k, n, rate, codebook,
                         inner.data() + std::size_t{e} * k * n);
    }
    weight.cols = k;
    weight.rows = n;
    weight.experts = experts;
    weight.exl3 = descriptors.data();
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
  std::vector<float> Restore(unsigned expert,
                             std::span<const float> raw) const {
    std::vector<double> values(raw.begin(), raw.end());
    ex::HadamardBlocks(std::span<double>(values));
    std::vector<float> result(n);
    for (unsigned j = 0; j < n; ++j)
      result[j] = values[j] * ex::HalfToDouble(svh[expert * n + j]);
    return result;
  }
};

// Mode2 is unchanged by the Mode3 staging optimization. Check its raw FP32
// dots against FP64, then require one FP16 rounding of those same dots in
// Mode3. Exact-size input and inactive descriptors exercise the skipped tiles.
void HalfBucketStaging(int bits, bool half = false, unsigned k = 256) {
  constexpr unsigned n = 128;
  Buffers weights;
  Bank bank(weights, k, n, 1, 73, true, bits, half);
  for (unsigned rows :
       {1U, 15U, 16U, 17U, 31U, 32U, 33U, 63U, 64U, 65U, 127U, 128U}) {
    Buffers buffers;
    // Row zero is outside the active bucket; no allocation padding follows
    // its final input row. The extra descriptor is outside the device count.
    std::vector<std::uint16_t> input((rows + 1) * k);
    for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = ex::HalfFromDouble((int(i % 61) - 30) / 4096.0);
    auto* x = buffers.Copy(input);
    auto* buckets = buffers.Copy(
        std::vector<gpu::Exl3ExpertBucket>{{0, rows, 1}, {0xffffffffU, 1, 0}});
    auto* count = buffers.Copy(std::vector<std::uint32_t>{1});
    auto* control = buffers.Make<float>((rows + 1) * n);
    const unsigned tile_rows = rows <= 16 ? 16 : rows <= 32 ? 32 : 64;
    gufo::hip::LaunchExl3PrefillBucketInner(
        bank.weight.device_exl3, buckets, count, 2, tile_rows, x, control, k, n,
        bank.rate, bank.codebook, nullptr);
    const auto dots = Read(control, (rows + 1) * n);
    for (unsigned r = 1; r <= rows; ++r)
      Near(std::span(dots).subspan(r * n, n), bank.Dot(0, input.data() + r * k),
           "bucket FP32 vs independent FP64");
    std::vector<std::int32_t> exponents(rows + 1);
    for (unsigned r = 0; r <= rows; ++r)
      exponents[r] = std::array{-1, 0, 1, 5}[r % 4];
    auto* e = buffers.Copy(exponents);
    constexpr std::uint16_t sentinel = 0x7e55;
    std::vector<std::uint16_t> poison((rows + 1) * n + 2, sentinel);
    auto* output = buffers.Copy(poison);
    for (bool scaled : {false, true}) {
      for (unsigned repeat = 0; repeat < 2; ++repeat) {
        Check(hipMemcpy(output, poison.data(), poison.size() * sizeof(sentinel),
                        hipMemcpyHostToDevice));
        gufo::hip::LaunchExl3PrefillBucketInnerHalf(
            bank.weight.device_exl3, buckets, count, 2, tile_rows, x,
            output + 1, k, n, bank.rate, bank.codebook, scaled ? e : nullptr,
            nullptr);
        const auto actual = Read(output, poison.size());
        for (unsigned i = 0; i <= n; ++i)
          Require(actual[i] == sentinel, "half bucket prefix guard");
        Require(actual.back() == sentinel, "half bucket suffix guard");
        for (unsigned r = 1; r <= rows; ++r)
          for (unsigned j = 0; j < n; ++j) {
            const auto idx = std::size_t{r} * n + j;
            const auto expected = ex::HalfFromDouble(std::ldexp(
                double(dots[idx]), scaled ? -std::max(exponents[r], 0) : 0));
            Require((actual[idx + 1] & 0x7c00) != 0x7c00 &&
                        actual[idx + 1] == expected,
                    "half bucket differs from rounded FP32 control");
          }
      }
    }
  }
}

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

// `k` 2560 (the model's hidden width) stages the gate/up rows in several
// LDS chunks for R >= 2; 384 is three whole tile-group pairs.
void FusedProducer(unsigned rows, bool shared, int bits = 5, int up_bits = 0,
                   int down_bits = 0, bool half = false, unsigned k = 256) {
  constexpr unsigned ff = 640, used = 3, experts = 12;
  const unsigned assignments = rows * used;
  Buffers buffers;
  Bank gate(buffers, k, ff, experts, 1, true, bits, half),
      up(buffers, k, ff, experts, 2, true, up_bits == 0 ? bits : up_bits, half),
      down(buffers, ff, k, experts, 3, true, down_bits == 0 ? bits : down_bits,
           half);
  gpu::DeviceLayer layer;
  layer.ffn_gate_exps = gate.weight;
  layer.ffn_up_exps = up.weight;
  layer.ffn_down_exps = down.weight;
  std::vector<float> input(rows * k);
  std::mt19937 rng(4);
  for (auto& v : input)
    v = (int(rng() % 2000) - 1000) * 0.00003125F;
  // Include a genuinely zero token alongside small unsaturated activations.
  if (rows > 1)
    std::fill_n(input.end() - k, k, 0.0F);
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
  // Decode glue: the untraced call (restore folded into the down-input
  // scaling) stores the traced call's bytes; raw down dots plus the fused
  // restore/combine equal the restored rows plus MoeEpilogue, byte for byte.
  {
    const auto same = [](const auto& a, const auto& b) {
      return a.size() == b.size() &&
             std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0;
    };
    auto* untraced = buffers.Make<float>(assignments * k);
    gpu::Exl3RoutedMlp(layer, dx, di, g, u, untraced, rows, used, s, nullptr);
    Require(same(actual, Read(untraced, assignments * k)),
            "fused restore/scale changed routed rows");
    std::vector<float> weights(assignments), shared_rows(rows * k),
        gates(rows * 2);
    for (auto& v : weights)
      v = (int(rng() % 1000) + 1) * 0.001F;
    for (auto& v : shared_rows)
      v = (int(rng() % 2000) - 1000) * 0.001F;
    for (auto& v : gates)
      v = (int(rng() % 2000) - 1000) * 0.004F;
    auto* dw = buffers.Copy(weights);
    auto* dsh = buffers.Copy(shared_rows);
    auto* dg = buffers.Copy(gates);
    auto* reference = buffers.Make<float>(rows * k);
    gpu::MoeEpilogue(y, dw, dsh, dg + 1, 2, reference, rows, used, k, nullptr);
    auto* raw = buffers.Make<float>(assignments * k);
    gpu::Exl3RoutedMlp(layer, dx, di, g, u, raw, rows, used, s, nullptr,
                       nullptr, false, true);
    auto* combined = buffers.Make<float>(rows * k);
    gpu::Exl3RoutedCombine(layer, di, rows, used, raw, dw, dsh, dg + 1, 2,
                           combined, nullptr);
    Require(same(Read(reference, rows * k), Read(combined, rows * k)),
            "fused routed combine differs from restore + MoeEpilogue");
    // Router top-k with the routed tiles: the same IDs/weights as RouterTopK
    // and, through tiles_ready, the same routed rows as BuildTiles' tiles.
    // Two launches check that the arrival counter returns to zero.
    std::vector<float> logits(rows * (experts + 1));
    for (auto& v : logits)
      v = (int(rng() % 4000) - 2000) * 0.002F;
    auto* dl = buffers.Copy(logits);
    auto* ref_ids = buffers.Make<std::int32_t>(assignments);
    auto* ref_w = buffers.Make<float>(assignments);
    gpu::RouterTopK(dl, experts + 1, ref_ids, ref_w, rows, experts, used,
                    nullptr);
    auto* arrivals = buffers.Make<std::uint32_t>(1);
    auto* tile_ids = buffers.Make<std::int32_t>(assignments);
    auto* tile_w = buffers.Make<float>(assignments);
    auto* built = buffers.Make<float>(assignments * k);
    auto* fused_rows = buffers.Make<float>(assignments * k);
    gpu::Exl3RoutedMlp(layer, dx, ref_ids, g, u, built, rows, used, s, nullptr,
                       nullptr, false, true);
    for (unsigned launch = 0; launch < 2; ++launch) {
      gpu::RouterTopKTiles(dl, experts + 1, tile_ids, tile_w, rows, experts,
                           used, s.tiles, s.tile_count, arrivals,
                           gpu::Exl3RouteTileRows(rows), nullptr);
      Require(Read(arrivals, 1)[0] == 0, "router tile arrivals not reset");
      Require(same(Read(ref_ids, assignments), Read(tile_ids, assignments)) &&
                  same(Read(ref_w, assignments), Read(tile_w, assignments)),
              "router tiles changed routing");
      gpu::Exl3RoutedMlp(layer, dx, tile_ids, g, u, fused_rows, rows, used, s,
                         nullptr, nullptr, true, true);
      Require(
          same(Read(built, assignments * k), Read(fused_rows, assignments * k)),
          "router-built tiles changed routed rows");
    }
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

// k = 272 and 352 end the routed GEMV K loop partway through a pair's first
// or second group; n = 80 leaves waves past the last output tile.
void DownFixtures(int bits, bool half = false, unsigned k = 640,
                  unsigned n = 128) {
  constexpr unsigned rows = 8;
  Buffers buffers;
  Bank down(buffers, k, n, 1, 5, true, bits, half);
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
  // Exercise every live count in the specialized R3/R4/R8 bodies, including
  // negative-exponent FP32 fallback rows, against the eight-row tiles above
  // (themselves a live-row body since R5-R8 tiles specialize too); unused
  // slots are poisoned and output has guards.
  for (unsigned cap : {3U, 4U, 8U}) {
    for (unsigned live = 1; live <= cap; ++live) {
      std::vector<gpu::Exl3RouteTile> tiles;
      for (unsigned first = 0; first < rows; first += live) {
        gpu::Exl3RouteTile part{};
        std::fill(std::begin(part.slots), std::end(part.slots), -1);
        part.count = std::min(live, rows - first);
        for (unsigned r = 0; r < part.count; ++r)
          part.slots[r] = first + r;
        tiles.push_back(part);
      }
      s.tiles = buffers.Copy(tiles);
      s.tile_count = buffers.Copy(
          std::vector<std::uint32_t>{static_cast<std::uint32_t>(tiles.size())});
      Check(hipMemcpy(scaled, input.data(), input.size() * sizeof(input[0]),
                      hipMemcpyHostToDevice));
      auto* guarded = buffers.Copy(std::vector<float>(rows * n + 2, 12345.0F));
      gpu::Exl3RoutedDown(down.weight, scaled, guarded + 1, rows, cap, s,
                          nullptr);
      const auto result = Read(guarded, rows * n + 2);
      Require(result.front() == 12345.0F && result.back() == 12345.0F,
              "live-row down output guard");
      Require(std::memcmp(result.data() + 1, actual.data(),
                          actual.size() * sizeof(float)) == 0,
              "live-row down differs from eight-row tiles");
    }
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
__global__ void PoisonUnsafeDots(const std::int32_t* exponents, unsigned rows,
                                 unsigned n, float* output) {
  const unsigned row = blockIdx.x;
  if (row < rows && exponents[row] < 0)
    for (unsigned column = threadIdx.x; column < n; column += blockDim.x)
      output[std::size_t{row} * n + column] = __builtin_nanf("");
}
const std::int32_t* poison_exponents = nullptr;
unsigned poison_rows = 0;
const ex::DeviceLinear* control_banks[3];
const std::uint16_t* control_weights[3];
unsigned control_calls = 0;
void ControlGemm(const ex::DeviceLinear* bank,
                 const gpu::Exl3ExpertBucket* buckets,
                 const std::uint32_t* count, std::uint32_t capacity,
                 std::uint32_t tile_rows, const std::uint16_t* input,
                 float* output, std::uint32_t k, std::uint32_t n,
                 ex::BitRate rate, ex::Codebook codebook, hipStream_t stream) {
  // The control reads the CPU-decoded weights: any format.
  Require(ex::Supports(codebook, rate), "control bank format");
  Require(tile_rows == 16 || tile_rows == 32 || tile_rows == 64,
          "adapter tile row contract");
  unsigned index = 0;
  while (index < 3 && control_banks[index] != bank)
    ++index;
  Require(index < 3, "adapter expert bank contract");
  hipLaunchKernelGGL(ControlBucketDots, dim3(capacity), dim3(128), 0, stream,
                     control_weights[index], buckets, count, input, output, k,
                     n);
  Check(hipGetLastError());
}

gpu::Exl3BucketGemm candidate_gemm = nullptr;
void ObservedGemm(const ex::DeviceLinear* bank,
                  const gpu::Exl3ExpertBucket* buckets,
                  const std::uint32_t* count, std::uint32_t capacity,
                  std::uint32_t tile_rows, const std::uint16_t* input,
                  float* output, std::uint32_t k, std::uint32_t n,
                  ex::BitRate rate, ex::Codebook codebook, hipStream_t stream) {
  ++control_calls;
  candidate_gemm(bank, buckets, count, capacity, tile_rows, input, output, k, n,
                 rate, codebook, stream);
  if (poison_exponents != nullptr && bank == control_banks[2])
    hipLaunchKernelGGL(PoisonUnsafeDots, dim3(poison_rows), dim3(256), 0,
                       stream, poison_exponents, poison_rows, n, output);
  Check(hipGetLastError());
}

void PrefillDispatch(gpu::Exl3BucketGemm gemm, int bits, int up_bits = 0,
                     int down_bits = 0) {
  candidate_gemm = gemm;
  constexpr unsigned rows = 17, used = 2, experts = 4, k = 128, ff = 640;
  constexpr unsigned assignments = rows * used, tile_rows = 16;
  Buffers buffers;
  Bank gate(buffers, k, ff, experts, 9, true, bits),
      up(buffers, k, ff, experts, 10, true, up_bits == 0 ? bits : up_bits),
      down(buffers, ff, k, experts, 11, true,
           down_bits == 0 ? bits : down_bits);
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
  gpu::Exl3PrefillTrace trace;
  trace.gate_up_input = buffers.Make<std::uint16_t>(2 * assignments * k);
  trace.gate_inner = buffers.Make<float>(assignments * ff);
  trace.up_inner = buffers.Make<float>(assignments * ff);
  trace.activation = buffers.Make<float>(assignments * ff);
  trace.down_input = buffers.Make<std::uint16_t>(assignments * ff);
  trace.fallback_counts = buffers.Make<std::uint32_t>(2);
  auto* dx = buffers.Copy(input);
  auto* di = buffers.Copy(ids);
  auto* y = buffers.Make<float>(assignments * k);
  gpu::Exl3PrefillExperts(layer, dx, di, y, rows, used, tile_rows, s,
                          ObservedGemm, nullptr, &trace);
  Require(control_calls == 3,
          "prefill must dispatch three bounded bank launches");
  const auto map = Read(s.slots, assignments);
  const auto gate_input = Read(trace.gate_up_input, 2 * assignments * k);
  const auto gate_inner = Read(trace.gate_inner, assignments * ff);
  const auto up_inner = Read(trace.up_inner, assignments * ff);
  const auto activation = Read(trace.activation, assignments * ff);
  const auto scaled = Read(s.input, assignments * ff);
  const auto exponents = Read(s.down_exponents, assignments);
  const auto actual = Read(y, assignments * k);
  auto* control_gate = buffers.Make<float>(assignments * ff);
  auto* control_up = buffers.Make<float>(assignments * ff);
  auto* control_down = buffers.Make<float>(assignments * k);
  ControlGemm(gate.weight.device_exl3, s.buckets, s.bucket_count, s.max_buckets,
              tile_rows, trace.gate_up_input, control_gate, k, ff,
              {bits, false}, gate.codebook, nullptr);
  ControlGemm(up.weight.device_exl3, s.buckets, s.bucket_count, s.max_buckets,
              tile_rows, trace.gate_up_input + assignments * k, control_up, k,
              ff, up.descriptors[0].rate, up.codebook, nullptr);
  ControlGemm(down.weight.device_exl3, s.buckets, s.bucket_count, s.max_buckets,
              tile_rows, s.input, control_down, ff, k, down.descriptors[0].rate,
              down.codebook, nullptr);
  const auto control_gate_inner = Read(control_gate, assignments * ff);
  const auto control_up_inner = Read(control_up, assignments * ff);
  const auto control_down_inner = Read(control_down, assignments * k);
  for (unsigned packed = 0; packed < assignments; ++packed) {
    const unsigned slot = map[packed], expert = ids[slot];
    const float* x = input.data() + (slot / used) * k;
    const auto gr = gate.Rotate(expert, x), ur = up.Rotate(expert, x);
    Require(std::equal(gr.begin(), gr.end(), gate_input.begin() + packed * k),
            "bucket gate producer bits");
    Require(std::equal(ur.begin(), ur.end(),
                       gate_input.begin() + (assignments + packed) * k),
            "bucket up producer bits");
    const auto gd = gate.Dot(expert, gr.data()), ud = up.Dot(expert, ur.data());
    Near(std::span(gate_inner).subspan(packed * ff, ff), gd,
         "bucket gate FP64 inner stage");
    Near(std::span(up_inner).subspan(packed * ff, ff), ud,
         "bucket up FP64 inner stage");
    Near(std::span(gate_inner).subspan(packed * ff, ff),
         std::span(control_gate_inner).subspan(packed * ff, ff),
         "bucket gate vs FMA control on frozen input");
    Near(std::span(up_inner).subspan(packed * ff, ff),
         std::span(control_up_inner).subspan(packed * ff, ff),
         "bucket up vs FMA control on frozen input");
    auto expected_activation = gate.Restore(expert, gd);
    const auto restored_up = up.Restore(expert, ud);
    for (unsigned j = 0; j < ff; ++j) {
      const float g = expected_activation[j];
      expected_activation[j] = (g / (1.0F + std::exp(-g))) * restored_up[j];
    }
    Require(std::any_of(expected_activation.begin(), expected_activation.end(),
                        [](float value) { return value != 0; }),
            "bucket fixture must exercise nonzero activation");
    Near(std::span(activation).subspan(packed * ff, ff), expected_activation,
         "bucket restored gate/up + activation FP64 stage");
    // Only after independent validation may this FP16 boundary be frozen.
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
    std::vector<float> unscaled_control(k);
    for (unsigned j = 0; j < k; ++j) {
      const float raw = control_down_inner[packed * k + j];
      unscaled_control[j] =
          exponents[packed] > 0 ? std::ldexp(raw, -exponents[packed]) : raw;
    }
    Near(std::span(actual).subspan(slot * k, k),
         down.Restore(expert, unscaled_control),
         "bucket down vs widened FMA control on frozen input");
  }
  if (gemm == gpu::Exl3PrefillBucketInner) {
    gpu::Exl3PrefillExperts(layer, dx, di, y, rows, used, tile_rows, s,
                            nullptr);
    const auto production = Read(y, assignments * k);
    Require(std::memcmp(actual.data(), production.data(),
                        actual.size() * sizeof(float)) == 0,
            "default production adapter differs from explicit X3");
    // Raw dots + fused combine == restored rows + MoeEpilogue, bit for bit.
    std::vector<float> weights(assignments), shared(rows * k),
        gate_logits(rows * 3);
    for (unsigned i = 0; i < assignments; ++i)
      weights[i] = 0.05F + 0.01F * float(i % 13);
    for (unsigned i = 0; i < shared.size(); ++i)
      shared[i] = (int(i % 31) - 15) * 0.0125F;
    for (unsigned i = 0; i < gate_logits.size(); ++i)
      gate_logits[i] = (int(i % 7) - 3) * 0.5F;
    auto* dw = buffers.Copy(weights);
    auto* dsh = buffers.Copy(shared);
    auto* dg = buffers.Copy(gate_logits);
    auto* combined = buffers.Make<float>(rows * k);
    auto* reference = buffers.Make<float>(rows * k);
    gpu::MoeEpilogue(y, dw, dsh, dg + 1, 3, reference, rows, used, k, nullptr);
    auto raw_scratch = s;
    raw_scratch.inverse = buffers.Make<std::int32_t>(assignments);
    gpu::Exl3PrefillExpertsRaw(layer, dx, di, rows, used, tile_rows,
                               raw_scratch, nullptr);
    gpu::Exl3PrefillCombine(layer, raw_scratch, di, rows, used, dw, dsh, dg + 1,
                            3, combined, nullptr);
    Check(hipDeviceSynchronize());
    const auto want = Read(reference, rows * k);
    const auto got = Read(combined, rows * k);
    // The raw path keeps its gate/up and down dots in FP16 (q6-gated):
    // within FP16 rounding of the FP32 path, finite, and deterministic.
    double num = 0.0, den = 0.0;
    bool finite = true;
    for (std::size_t i = 0; i < want.size(); ++i) {
      finite = finite && std::isfinite(got[i]);
      num += (double(got[i]) - want[i]) * (double(got[i]) - want[i]);
      den += double(want[i]) * want[i];
    }
    Require(finite, "fused FP16 prefill combine must be finite");
    Require(std::sqrt(num / den) < 2e-3,
            "fused FP16 prefill combine vs FP32 restore + MoeEpilogue");
    std::cout << "fused FP16 prefill combine rel L2 " << std::sqrt(num / den)
              << "\n";
    gpu::Exl3PrefillExpertsRaw(layer, dx, di, rows, used, tile_rows,
                               raw_scratch, nullptr);
    gpu::Exl3PrefillCombine(layer, raw_scratch, di, rows, used, dw, dsh, dg + 1,
                            3, combined, nullptr);
    Check(hipDeviceSynchronize());
    const auto again = Read(combined, rows * k);
    Require(
        std::memcmp(again.data(), got.data(), got.size() * sizeof(float)) == 0,
        "fused FP16 prefill combine must be deterministic");
  }
  hipStream_t stream;
  Check(hipStreamCreate(&stream));
  Check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
  if (gemm == gpu::Exl3PrefillBucketInner)
    gpu::Exl3PrefillExperts(layer, dx, di, y, rows, used, tile_rows, s, stream);
  else
    gpu::Exl3PrefillExperts(layer, dx, di, y, rows, used, tile_rows, s,
                            ObservedGemm, stream, nullptr);
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

// Independent dense Sylvester oracle on frozen FP16 raw dots. Integer sums
// and power-of-two weights isolate the restore products; irregular scales
// also exercise their FP32 rounding order without a second GPU reference.
void PrefillCombineFrozenRows(bool irregular_scales = false) {
  constexpr unsigned rows = 5, used = 3, experts = 6, width = 256;
  constexpr unsigned assignments = rows * used;
  Buffers buffers;
  std::vector<std::int32_t> ids(assignments), inverse(assignments);
  constexpr std::int32_t selected[] = {5, 0, 3};
  for (unsigned a = 0; a < assignments; ++a)
    ids[a] = selected[(a + a / used) % used];
  unsigned packed = 0;
  for (unsigned e = 0; e < experts; ++e)
    for (unsigned a = 0; a < assignments; ++a)
      if (ids[a] == static_cast<std::int32_t>(e))
        inverse[a] = packed++;
  std::vector<std::uint16_t> raw(assignments * width);
  const auto dot = [](unsigned a, unsigned j) {
    return int((a * 13 + j * 7) % 17) - 8;
  };
  for (unsigned a = 0; a < assignments; ++a)
    for (unsigned j = 0; j < width; ++j)
      raw[inverse[a] * width + j] = ex::HalfFromDouble(dot(a, j) / 64.0);
  const auto scale = [irregular_scales](unsigned e, unsigned j) {
    const float mantissa =
        irregular_scales ? 1.0F + float((e * 13 + j) % 7) / 32.0F : 1.0F;
    return std::ldexp(((e + j) % 3) == 0 ? -mantissa : mantissa,
                      int((e + j / 128) % 3) - 1);
  };
  std::vector<ex::DeviceLinear> bank(experts);
  for (unsigned e = 0; e < experts; ++e) {
    std::vector<std::uint16_t> svh(width);
    for (unsigned j = 0; j < width; ++j)
      svh[j] = ex::HalfFromDouble(scale(e, j));
    bank[e].svh = buffers.Copy(svh);
  }
  std::vector<float> weights(assignments), shared(rows * width), gates(rows);
  for (unsigned a = 0; a < assignments; ++a)
    weights[a] = std::ldexp(1.0F, -int(1 + a % 4));
  for (unsigned i = 0; i < shared.size(); ++i)
    shared[i] = (int(i % 11) - 5) / 32.0F;
  std::vector<float> expected(rows * width);
  for (unsigned t = 0; t < rows; ++t)
    for (unsigned j = 0; j < width; ++j) {
      volatile float acc = 0.0F;
      for (unsigned slot = 0; slot < used; ++slot) {
        const unsigned a = t * used + slot;
        int sum = 0;
        for (unsigned col = 0; col < 128; ++col) {
          const int sign = (std::popcount((j % 128) & col) % 2) ? -1 : 1;
          sum += sign * dot(a, (j / 128) * 128 + col);
        }
        // Preserve the two FP32 products independently of compiler
        // reassociation; non-power-of-two scales exercise this boundary.
        volatile float scaled = (sum / 64.0F) * ar::kHadamardScale;
        const float restored = scaled * scale(ids[a], j);
        acc = acc + weights[a] * restored;
      }
      expected[t * width + j] = acc + 0.5F * shared[t * width + j];
    }
  gpu::DeviceLayer layer;
  layer.ffn_down_exps.device_exl3 = buffers.Copy(bank);
  layer.ffn_down_exps.rows = width;
  layer.ffn_down_exps.experts = experts;
  gpu::Exl3PrefillScratch scratch;
  scratch.inverse = buffers.Copy(inverse);
  scratch.out = reinterpret_cast<float*>(buffers.Copy(raw));
  auto* di = buffers.Copy(ids);
  auto* dw = buffers.Copy(weights);
  auto* ds = buffers.Copy(shared);
  auto* dg = buffers.Copy(gates);
  auto* output = buffers.Make<float>(rows * width);
  for (unsigned first : {0U, rows - 1}) {
    auto tail = scratch;
    tail.inverse += first * used;
    gpu::Exl3PrefillCombine(layer, tail, di + first * used, rows - first, used,
                            dw + first * used, ds + first * width, dg + first,
                            1, output, nullptr);
    Check(hipDeviceSynchronize());
    const auto actual = Read(output, (rows - first) * width);
    Require(std::memcmp(actual.data(), expected.data() + first * width,
                        actual.size() * sizeof(float)) == 0,
            "frozen raw combine differs from dense Sylvester oracle");
  }
}

void PrefillUnsafeRows(gpu::Exl3BucketGemm gemm, int bits) {
  candidate_gemm = gemm;
  // More rows than the worker grid, and more N tiles than eight waves.
  constexpr unsigned rows = 513, k = 128, n = 640, experts = 2, tile_rows = 32;
  Buffers buffers;
  Bank down(buffers, k, n, experts, 17, true, bits);
  control_banks[0] = control_banks[1] = nullptr;
  control_weights[0] = control_weights[1] = nullptr;
  control_banks[2] = down.weight.device_exl3;
  control_weights[2] = buffers.Copy(down.inner);
  gpu::Exl3PrefillScratch s;
  s.max_assignments = rows;
  s.max_buckets = experts + rows / tile_rows;
  s.offsets = buffers.Make<std::uint32_t>(experts + 1);
  s.slots = buffers.Make<std::int32_t>(rows);
  s.buckets = buffers.Make<gpu::Exl3ExpertBucket>(s.max_buckets);
  s.bucket_count = buffers.Make<std::uint32_t>(1);
  s.input = buffers.Make<std::uint16_t>(rows * k);
  s.out = buffers.Make<float>(rows * n);
  s.down_exponents = buffers.Make<std::int32_t>(rows);
  std::vector<std::int32_t> ids(rows);
  for (unsigned r = 0; r < rows; ++r)
    ids[r] = r % experts;
  gpu::Exl3BuildExpertBuckets(buffers.Copy(ids), rows, experts, tile_rows,
                              s.offsets, s.slots, s.buckets, s.bucket_count,
                              s.max_buckets, nullptr);
  const auto slots = Read(s.slots, rows);
  std::vector<std::uint16_t> input(rows * k);
  const std::vector<unsigned> unsafe{0, 255, 256, 512};
  for (unsigned r : unsafe) {
    std::fill_n(input.begin() + r * k, k, 0x7bff);
    input[r * k] = 1;
  }
  std::fill_n(input.begin() + k, k, 1);  // Safe, nonzero subnormal row.
  auto* original = buffers.Copy(input);
  gpu::Exl3PrefillTrace trace;
  trace.fallback_counts = buffers.Make<std::uint32_t>(2);
  poison_exponents = s.down_exponents;
  poison_rows = rows;
  const auto enqueue = [&](hipStream_t stream) {
    Check(hipMemcpyAsync(s.input, original,
                         input.size() * sizeof(std::uint16_t),
                         hipMemcpyDeviceToDevice, stream));
    gpu::Exl3PrefillDown(down.weight, rows, tile_rows, s, ObservedGemm, stream,
                         &trace);
  };
  enqueue(nullptr);
  const auto actual = Read(s.out, rows * n);
  const auto exponents = Read(s.down_exponents, rows);
  const auto counts = Read(trace.fallback_counts, 2);
  Require(counts[0] == 256 && counts[1] == unsafe.size(),
          "bounded fallback block/unsafe-row counts");
  for (unsigned r = 0; r < rows; ++r)
    Require((exponents[r] < 0) ==
                (std::find(unsafe.begin(), unsafe.end(), r) != unsafe.end()),
            "prefill unsafe-to-scale classification");
  Require(exponents[1] == 10, "prefill safe subnormal row was not scaled");

  // Widened GPU control uses the original, unscaled FP16 row and exactly
  // the decode FMA/reduction tree. Never allow the packed adapter's poisoned
  // unsafe results to act as the expected output.
  gpu::Exl3Scratch widened_scratch;
  std::vector<gpu::Exl3RouteTile> tiles;
  for (unsigned r : unsafe) {
    gpu::Exl3RouteTile tile{};
    tile.expert = ids[slots[r]];
    tile.count = 1;
    tile.slots[0] = r;
    tiles.push_back(tile);
  }
  widened_scratch.tiles = buffers.Copy(tiles);
  widened_scratch.tile_count =
      buffers.Copy(std::vector<std::uint32_t>{unsigned(tiles.size())});
  auto* widened = buffers.Make<float>(rows * n);
  gpu::Exl3RoutedDown(down.weight, original, widened, rows, 1, widened_scratch,
                      nullptr);
  const auto widened_inner = Read(widened, rows * n);
  for (unsigned r : unsafe) {
    const unsigned expert = ids[slots[r]];
    Near(std::span(widened_inner).subspan(r * n, n),
         down.Dot(expert, input.data() + r * k),
         "prefill widened control FP64 inner");
    Near(std::span(actual).subspan(r * n, n),
         down.Restore(expert, std::span(widened_inner).subspan(r * n, n)),
         "prefill unsafe fallback vs widened control");
  }
  Near(std::span(actual).subspan(n, n),
       down.Restore(ids[slots[1]], down.Dot(ids[slots[1]], input.data() + k)),
       "prefill scaled subnormal row FP64 output");
  // Put each same rounded row alone in a bucket, including an expert other
  // than zero, and require exact bytes against the larger worker grid.
  gpu::Exl3PrefillScratch single;
  single.max_assignments = single.max_buckets = 1;
  single.offsets = buffers.Make<std::uint32_t>(experts + 1);
  single.buckets = buffers.Make<gpu::Exl3ExpertBucket>(1);
  single.bucket_count = buffers.Copy(std::vector<std::uint32_t>{1});
  single.input = buffers.Make<std::uint16_t>(k);
  single.out = buffers.Make<float>(n);
  single.down_exponents = buffers.Make<std::int32_t>(1);
  for (unsigned row : {0U, 1U, 255U, 256U, 512U}) {
    const unsigned expert = ids[slots[row]];
    const std::vector<std::uint32_t> offsets{0, expert == 0 ? 1U : 0U, 1};
    const gpu::Exl3ExpertBucket descriptor{expert, 1, 0};
    Check(hipMemcpy(single.offsets, offsets.data(),
                    offsets.size() * sizeof(std::uint32_t),
                    hipMemcpyHostToDevice));
    Check(hipMemcpy(single.buckets, &descriptor, sizeof(descriptor),
                    hipMemcpyHostToDevice));
    Check(hipMemcpy(single.input, original + row * k, k * sizeof(std::uint16_t),
                    hipMemcpyDeviceToDevice));
    poison_exponents = single.down_exponents;
    poison_rows = 1;
    gpu::Exl3PrefillDown(down.weight, 1, tile_rows, single, ObservedGemm,
                         nullptr);
    const auto alone = Read(single.out, n);
    Require(std::memcmp(alone.data(), actual.data() + row * n,
                        n * sizeof(float)) == 0,
            "prefill down scalar/worker-grid partition bits");
  }
  poison_exponents = s.down_exponents;
  poison_rows = rows;
  hipStream_t stream;
  Check(hipStreamCreate(&stream));
  Check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
  enqueue(stream);
  hipGraph_t graph;
  Check(hipStreamEndCapture(stream, &graph));
  hipGraphExec_t executable;
  Check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  for (unsigned replay = 0; replay < 2; ++replay) {
    Check(hipGraphLaunch(executable, stream));
    Check(hipStreamSynchronize(stream));
    const auto repeated = Read(s.out, rows * n);
    Require(std::memcmp(actual.data(), repeated.data(),
                        actual.size() * sizeof(float)) == 0,
            "prefill fallback graph replay changed bytes");
    Require(Read(trace.fallback_counts, 2) == counts,
            "fallback replay retained stale counts");
  }
  // The same graph must stop falling back when device row eligibility changes.
  std::fill(input.begin(), input.end(), 0);
  Check(hipMemcpy(original, input.data(), input.size() * sizeof(std::uint16_t),
                  hipMemcpyHostToDevice));
  Check(hipGraphLaunch(executable, stream));
  Check(hipStreamSynchronize(stream));
  const auto no_work = Read(trace.fallback_counts, 2);
  Require(no_work[0] == 256 && no_work[1] == 0,
          "all-safe replay fallback launch bound");
  for (float value : Read(s.out, rows * n))
    Require(value == 0, "all-safe zero replay retained stale output");
  Check(hipGraphExecDestroy(executable));
  Check(hipGraphDestroy(graph));
  Check(hipStreamDestroy(stream));
  poison_exponents = nullptr;
  poison_rows = 0;
}

struct ExpertRun {
  std::vector<std::uint16_t> gate_up, down;
  std::vector<float> gate, up, activation, inner, output;
};

template<class T>
void SameBits(std::span<const T> a, std::span<const T> b, const char* message) {
  Require(a.size() == b.size() &&
              std::memcmp(a.data(), b.data(), a.size_bytes()) == 0,
          message);
}
void SameRun(const ExpertRun& a, const ExpertRun& b, unsigned k, unsigned ff,
             unsigned used, bool first_token_only = false) {
  const auto floats = [&](const std::vector<float>& x,
                          const std::vector<float>& y, unsigned width,
                          const char* message) {
    if (x.empty() && y.empty())
      return;
    const auto count = first_token_only ? used * width : x.size();
    SameBits<float>(std::span(x).first(count), std::span(y).first(count),
                    message);
  };
  const auto down_count = first_token_only ? used * ff : a.down.size();
  SameBits<std::uint16_t>(std::span(a.down).first(down_count),
                          std::span(b.down).first(down_count),
                          "down producer partition bits");
  const auto plane = a.gate_up.size() / 2;
  const auto gate_count = first_token_only ? used * k : plane;
  for (unsigned projection = 0; projection < 2; ++projection)
    SameBits<std::uint16_t>(
        std::span(a.gate_up).subspan(projection * plane, gate_count),
        std::span(b.gate_up).subspan(projection * plane, gate_count),
        "gate/up input partition bits");
  floats(a.gate, b.gate, ff, "gate inner partition bits");
  floats(a.up, b.up, ff, "up inner partition bits");
  floats(a.activation, b.activation, ff, "activation partition bits");
  floats(a.inner, b.inner, k, "down inner partition bits");
  floats(a.output, b.output, k, "output partition bits");
}

// Reassemble every result by original token and expert slot. This observes
// actual production launches: width, tile position, route overlap and caller
// partition boundaries vary, while the compared row's input/expert stays fixed.
ExpertRun RunExperts(const gpu::DeviceLayer& layer,
                     const std::vector<float>& input,
                     const std::vector<std::int32_t>& ids,
                     const std::vector<unsigned>& order, unsigned cap,
                     bool scale, bool prefill, unsigned tile_rows,
                     gpu::Exl3BucketGemm gemm, bool reorder_slots) {
  const unsigned k = layer.ffn_gate_exps.cols, ff = layer.ffn_gate_exps.rows;
  const unsigned rows = order.size(), used = ids.size() / rows;
  const unsigned assignments = rows * used, capacity = cap * used;
  Buffers buffers;
  std::vector<float> reordered_input(input.size());
  std::vector<std::int32_t> reordered_ids(ids.size());
  for (unsigned r = 0; r < rows; ++r) {
    std::copy_n(input.data() + order[r] * k, k, reordered_input.data() + r * k);
    for (unsigned j = 0; j < used; ++j) {
      const unsigned original_slot = reorder_slots ? (j + order[r]) % used : j;
      reordered_ids[r * used + j] = ids[order[r] * used + original_slot];
    }
  }
  auto* dx = buffers.Copy(reordered_input);
  auto* di = buffers.Copy(reordered_ids);
  auto* g = buffers.Make<float>(capacity * ff);
  auto* u = buffers.Make<float>(capacity * ff);
  auto* y = buffers.Make<float>(capacity * k);
  gpu::Exl3Scratch routed;
  routed.tiles = buffers.Make<gpu::Exl3RouteTile>(capacity);
  routed.tile_count = buffers.Make<std::uint32_t>(1);
  routed.routed_input =
      buffers.Make<std::uint16_t>(capacity * std::max(2 * k, ff));
  if (scale)
    routed.down_exponents = buffers.Make<std::int32_t>(capacity);
  gpu::Exl3PrefillScratch bucket;
  if (prefill) {
    bucket.max_assignments = capacity;
    bucket.max_buckets =
        std::min(capacity, layer.ffn_gate_exps.experts + capacity / tile_rows);
    bucket.offsets =
        buffers.Make<std::uint32_t>(layer.ffn_gate_exps.experts + 1);
    bucket.slots = buffers.Make<std::int32_t>(capacity);
    bucket.buckets = buffers.Make<gpu::Exl3ExpertBucket>(bucket.max_buckets);
    bucket.bucket_count = buffers.Make<std::uint32_t>(1);
    bucket.input = routed.routed_input;
    bucket.gate = g;
    bucket.up = u;
    bucket.out = buffers.Make<float>(capacity * k);
    bucket.down_exponents = routed.down_exponents;
  }
  gpu::Exl3RoutedTrace trace;
  trace.gate_up_input = buffers.Make<std::uint16_t>(2 * capacity * k);
  trace.gate_inner = buffers.Make<float>(capacity * ff);
  trace.up_inner = buffers.Make<float>(capacity * ff);
  trace.activation = buffers.Make<float>(capacity * ff);
  trace.down_input = buffers.Make<std::uint16_t>(capacity * ff);
  trace.down_inner = buffers.Make<float>(capacity * k);
  const gpu::Exl3PrefillTrace bucket_trace{
      trace.gate_up_input, trace.gate_inner, trace.up_inner,
      trace.activation,    trace.down_input, nullptr};
  ExpertRun result;
  result.gate_up.resize(2 * assignments * k);
  result.down.resize(assignments * ff);
  result.gate.resize(assignments * ff);
  result.up.resize(assignments * ff);
  result.activation.resize(assignments * ff);
  if (!prefill)
    result.inner.resize(assignments * k);
  result.output.resize(assignments * k);
  for (unsigned first = 0; first < rows; first += cap) {
    const unsigned count = std::min(cap, rows - first), slots = count * used;
    if (prefill)
      gpu::Exl3PrefillExperts(layer, dx + first * k, di + first * used, y,
                              count, used, tile_rows, bucket, gemm, nullptr,
                              &bucket_trace);
    else
      gpu::Exl3RoutedMlp(layer, dx + first * k, di + first * used, g, u, y,
                         count, used, routed, nullptr, &trace);
    const auto gu = Read(trace.gate_up_input, 2 * slots * k);
    const auto gi = Read(trace.gate_inner, slots * ff);
    const auto ui = Read(trace.up_inner, slots * ff);
    const auto activation = Read(trace.activation, slots * ff);
    const auto down = Read(trace.down_input, slots * ff);
    const auto inner =
        prefill ? std::vector<float>{} : Read(trace.down_inner, slots * k);
    const auto output = Read(y, slots * k);
    std::vector<std::int32_t> map(slots);
    if (prefill)
      map = Read(bucket.slots, slots);
    else
      for (unsigned j = 0; j < slots; ++j)
        map[j] = j;
    for (unsigned packed = 0; packed < slots; ++packed) {
      const unsigned slot = map[packed], local_row = slot / used;
      const unsigned original_row = order[first + local_row];
      const unsigned original_slot =
          reorder_slots ? (slot % used + original_row) % used : slot % used;
      const unsigned destination = original_row * used + original_slot;
      for (unsigned projection = 0; projection < 2; ++projection)
        std::copy_n(gu.data() + (projection * slots + packed) * k, k,
                    result.gate_up.data() +
                        (projection * assignments + destination) * k);
      std::copy_n(gi.data() + packed * ff, ff,
                  result.gate.data() + destination * ff);
      std::copy_n(ui.data() + packed * ff, ff,
                  result.up.data() + destination * ff);
      std::copy_n(activation.data() + packed * ff, ff,
                  result.activation.data() + destination * ff);
      std::copy_n(down.data() + packed * ff, ff,
                  result.down.data() + destination * ff);
      if (!prefill)
        std::copy_n(inner.data() + slot * k, k,
                    result.inner.data() + destination * k);
      std::copy_n(output.data() + slot * k, k,
                  result.output.data() + destination * k);
    }
  }
  Require(std::any_of(
              result.output.begin(), result.output.end(),
              [](float value) { return value != 0 && std::isfinite(value); }),
          "invariance fixture must have nonzero finite output");
  return result;
}

void ExpertInvariance(bool prefill, gpu::Exl3BucketGemm gemm, int bits,
                      bool half = false) {
  if (prefill) {
    candidate_gemm = gemm;
    gemm = ObservedGemm;
  }
  constexpr unsigned k = 128, ff = 640, used = 3, experts = 12;
  const unsigned rows = prefill ? 65 : 17;
  Buffers buffers;
  Bank gate(buffers, k, ff, experts, 21, true, bits, half),
      up(buffers, k, ff, experts, 22, true, bits, half),
      down(buffers, ff, k, experts, 23, true, bits, half);
  gpu::DeviceLayer layer;
  layer.ffn_gate_exps = gate.weight;
  layer.ffn_up_exps = up.weight;
  layer.ffn_down_exps = down.weight;
  if (prefill) {
    control_banks[0] = gate.weight.device_exl3;
    control_banks[1] = up.weight.device_exl3;
    control_banks[2] = down.weight.device_exl3;
    control_weights[0] = buffers.Copy(gate.inner);
    control_weights[1] = buffers.Copy(up.inner);
    control_weights[2] = buffers.Copy(down.inner);
  }
  std::vector<float> input(rows * k);
  std::vector<std::int32_t> ids(rows * used);
  std::vector<unsigned> order(rows);
  std::mt19937 rng(24);
  for (auto& value : input)
    value = (int(rng() % 2000) - 1000) * 0.00003125F;
  for (unsigned r = 0; r < rows; ++r) {
    order[r] = r;
    for (unsigned j = 0; j < used; ++j)
      ids[r * used + j] = prefill ? j : (r * used + j) % experts;
  }
  for (bool scale : {false, true}) {
    if (prefill && !scale)
      continue;
    const auto serial = RunExperts(layer, input, ids, order, 1, scale, prefill,
                                   16, gemm, false);
    const std::vector<unsigned> widths =
        prefill ? std::vector<unsigned>{1, 3, 17, 32, 33, 64, 65}
                : std::vector<unsigned>{1, 2, 3, 4, 5, 6, 7, 8, 9, 17};
    const std::vector<unsigned> tiles =
        prefill ? std::vector<unsigned>{16, 32, 64} : std::vector<unsigned>{16};
    for (unsigned tile : tiles)
      for (unsigned width : widths) {
        std::cout << "CASE " << (prefill ? "prefill" : "routed")
                  << " invariance scale=" << scale << " width=" << width
                  << " tile=" << tile << '\n';
        SameRun(serial,
                RunExperts(layer, input, ids, order, width, scale, prefill,
                           tile, gemm, false),
                k, ff, used);
        std::reverse(order.begin(), order.end());
        SameRun(serial,
                RunExperts(layer, input, ids, order, width, scale, prefill,
                           tile, gemm, true),
                k, ff, used);
        std::reverse(order.begin(), order.end());
      }
    // Every row on the same experts: tile counts equal the width, so each
    // live-row body (1..8) is compared on all tokens.
    if (!prefill) {
      std::vector<std::int32_t> same(ids.size());
      for (unsigned r = 0; r < rows; ++r)
        for (unsigned j = 0; j < used; ++j)
          same[r * used + j] = static_cast<std::int32_t>(j);
      const auto alone = RunExperts(layer, input, same, order, 1, scale,
                                    prefill, 16, gemm, false);
      for (unsigned width = 2; width <= 9; ++width) {
        std::cout << "CASE routed shared-expert invariance scale=" << scale
                  << " width=" << width << '\n';
        SameRun(alone,
                RunExperts(layer, input, same, order, width, scale, prefill, 16,
                           gemm, false),
                k, ff, used);
      }
    }
    // Keep token zero's input and IDs fixed while changing its position and
    // the other tokens' route overlap, including single-expert row tiles.
    for (bool shared : {false, true}) {
      auto mixed = ids;
      for (unsigned r = 1; r < rows; ++r)
        for (unsigned j = 0; j < used; ++j)
          mixed[r * used + j] = shared ? j : 3 + (r * used + j) % (experts - 3);
      for (unsigned position : {0U, rows / 2, rows - 1}) {
        std::rotate(order.begin(), order.begin() + 1,
                    order.begin() + position + 1);
        SameRun(serial,
                RunExperts(layer, input, mixed, order, rows, scale, prefill, 32,
                           gemm, true),
                k, ff, used, true);
        for (unsigned r = 0; r < rows; ++r)
          order[r] = r;
      }
    }
  }
}

void SelectedHeadRate(int bits, bool half = false, unsigned k = 128) {
  constexpr unsigned n = 640;
  Buffers buffers;
  Bank bank(buffers, k, n, 1, 31, true, bits, half);
  std::vector<float> input(k);
  for (unsigned i = 0; i < k; ++i)
    input[i] = (int(i % 73) - 36) * 0.00003125F;
  // Duplicated IDs/segments, both prefix boundaries, and invalid upper IDs.
  const std::vector<std::int32_t> ids{0,  127, 128,   129, 257, 639,
                                      -1, 640, 99999, 257, 128};
  const auto rotated = bank.Rotate(0, input.data());
  const auto expected = bank.Restore(0, bank.Dot(0, rotated.data()));
  for (unsigned prefix : {0U, 128U, 256U, n}) {
    gpu::Exl3Scratch scratch;
    scratch.projection.fast_input = buffers.Make<std::uint16_t>(k);
    scratch.head_segments = buffers.Make<std::uint32_t>(ids.size() + 1);
    const unsigned values = (prefix / 128 + ids.size()) * 128;
    scratch.head_values = buffers.Make<float>(values + 16);
    constexpr float sentinel = 12345.0F;
    std::vector<float> initial(values + 16, sentinel);
    Check(hipMemcpy(scratch.head_values, initial.data(),
                    initial.size() * sizeof(float), hipMemcpyHostToDevice));
    auto* storage =
        buffers.Copy(std::vector<float>(prefix + ids.size() + 32, sentinel));
    auto* y = storage + 16;
    auto* dx = buffers.Copy(input);
    auto* di = buffers.Copy(ids);
    std::vector<float> first;
    for (int repeat = 0; repeat < 2; ++repeat) {
      gpu::Exl3DraftHead(bank.weight, dx, y, prefix, di, ids.size(), scratch,
                         nullptr);
      const auto actual = Read(y, prefix + ids.size());
      Near(std::span(actual).first(prefix), std::span(expected).first(prefix),
           "selected-head prefix FP64");
      std::vector<float> extras, reference;
      for (unsigned i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || ids[i] >= int(n)) {
          Require(actual[prefix + i] == -INFINITY, "invalid selected-head ID");
        } else {
          extras.push_back(actual[prefix + i]);
          reference.push_back(expected[ids[i]]);
        }
      }
      Near(extras, reference, "selected-head extras FP64");
      if (repeat == 0)
        first = actual;
      else
        Require(std::memcmp(first.data(), actual.data(),
                            first.size() * sizeof(float)) == 0,
                "selected-head repeat");
      for (float value : Read(storage, 16))
        Require(value == sentinel, "head prefix guard");
      for (float value : Read(y + prefix + ids.size(), 16))
        Require(value == sentinel, "head suffix guard");
      for (float value : Read(scratch.head_values + values, 16))
        Require(value == sentinel, "head scratch guard");
    }
  }
  // Prefix-only requests and completely empty requests must not launch a
  // zero-sized gather grid or touch absent ID/output storage.
  for (unsigned prefix : {0U, 128U, n}) {
    gpu::Exl3Scratch scratch;
    scratch.projection.fast_input = buffers.Make<std::uint16_t>(k);
    scratch.head_segments = buffers.Make<std::uint32_t>(1);
    scratch.head_values = buffers.Make<float>(n);
    auto* y = buffers.Make<float>(n);
    gpu::Exl3DraftHead(bank.weight, buffers.Copy(input), y, prefix, nullptr, 0,
                       scratch, nullptr);
    if (prefix != 0)
      Near(Read(y, prefix), std::span(expected).first(prefix),
           "prefix-only head FP64");
    else
      Require(Read(y, n) == std::vector<float>(n), "empty head wrote output");
  }
  // Formats no kernel takes for this codebook (K2.5 is a mul1 format since
  // every EXL3 codebook and rate run on the fast kernels).
  for (ex::BitRate invalid :
       {ex::BitRate{2, true}, ex::BitRate{4, true}, ex::BitRate{9, false}}) {
    if (ex::Supports(bank.codebook, invalid))
      continue;
    auto descriptor = bank.descriptors[0];
    descriptor.rate = invalid;
    auto weight = bank.weight;
    weight.exl3 = &descriptor;
    bool rejected = false;
    try {
      gpu::Exl3DraftHead(weight, nullptr, nullptr, 128, nullptr, 0, {},
                         nullptr);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    Require(rejected, "unsupported head format accepted");
  }
  std::cout << "selected head K=" << bits << (half ? ".5" : "") << " passed\n";
}

template<class F>
float DenseTime(F&& launch) {
  hipEvent_t start, end;
  Check(hipEventCreate(&start));
  Check(hipEventCreate(&end));
  for (int i = 0; i < 2; ++i)
    launch();
  std::vector<float> samples;
  for (int r = 0; r < 7; ++r) {
    Check(hipEventRecord(start));
    for (int i = 0; i < 5; ++i)
      launch();
    Check(hipEventRecord(end));
    Check(hipEventSynchronize(end));
    float ms = 0;
    Check(hipEventElapsedTime(&ms, start, end));
    samples.push_back(ms / 5);
  }
  Check(hipEventDestroy(start));
  Check(hipEventDestroy(end));
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

void DenseTraversalRates(const char* evidence) {
  constexpr unsigned k = 6144, n = 2560, rows = 2048;
  std::ofstream file(evidence, std::ios::binary);
  Require(bool(file), "open dense evidence");
  for (int bits = 1; bits <= 8; ++bits) {
    Buffers buffers;
    Bank bank(buffers, k, n, 1, 941, false, bits);
    std::vector<std::uint16_t> input(std::size_t(rows) * k);
    for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = ex::HalfFromDouble((int(i % 119) - 59) * 0.00003125);
    auto* dx = buffers.Copy(input);
    constexpr float sentinel = 23456.0F;
    auto* storage =
        buffers.Copy(std::vector<float>(std::size_t(rows) * n + 32, sentinel));
    auto* y = storage + 16;
    auto* direct = buffers.Make<float>(std::size_t(rows) * n);
    gufo::hip::LaunchExl3PrefillGemm(bank.descriptors[0], dx, direct, rows,
                                     false, nullptr);
    const auto reference = Read(direct, std::size_t(rows) * n);
    const auto& l = bank.descriptors[0];
    gufo::hip::Exl3GemmProblem problem{l.trellis, l.svh, dx, y, rows, 0};
    auto* dp = buffers.Copy(std::vector{problem});
    const auto launch = [&] {
      gufo::hip::LaunchExl3PrefillGemmGrouped(
          dp, 1, rows, k, n, bank.rate, ex::Codebook::kMul1, false, nullptr);
    };
    for (int repeat = 0; repeat < 2; ++repeat) {
      launch();
      const auto actual = Read(y, std::size_t(rows) * n);
      Require(std::memcmp(actual.data(), reference.data(),
                          actual.size() * sizeof(float)) == 0,
              "dense traversal differs from direct K16 control");
      for (float v : Read(storage, 16))
        Require(v == sentinel, "dense prefix guard");
      for (float v : Read(y + std::size_t(rows) * n, 16))
        Require(v == sentinel, "dense suffix guard");
      if (repeat == 0)
        file.write(reinterpret_cast<const char*>(actual.data()),
                   actual.size() * sizeof(float));
    }
    std::cout << "DENSE K=" << bits
              << " exact=1 median_ms=" << DenseTime(launch) << '\n';
  }
}

void Buckets(unsigned rows, bool shared, unsigned tile_rows, int bits) {
  constexpr unsigned used = 3, experts = 12, width = 640;
  const unsigned assignments = rows * used;
  const unsigned capacity =
      std::min(assignments, experts + assignments / tile_rows);
  Buffers buffers;
  Bank bank(buffers, width, 128, experts, 6, false, bits);
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
    // With 64-row tiles, a hot expert (more than 64 rows) is split evenly
    // into buckets of at most 128 rows.
    const unsigned rows = bounds[e + 1] - bounds[e];
    const bool hot = tile_rows == 64 && rows > 64;
    const unsigned tiles =
        hot ? (rows + 127) / 128 : (rows + tile_rows - 1) / tile_rows;
    for (unsigned t = 0, row = bounds[e]; t < tiles; ++t) {
      const unsigned want = hot ? (bounds[e + 1] - row) / (tiles - t)
                                : std::min(tile_rows, bounds[e + 1] - row);
      Require(next < active && buckets[next].expert == e &&
                  buckets[next].first == row && buckets[next].count == want,
              "bucket ragged tile");
      row += want;
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

// Both adapters must pass the independent FP64 stage/output bars, poisoned
// fallback fixtures and exact partition/reordering contract.
void QualifyFlashNextExl3PrefillAdapter(gpu::Exl3BucketGemm gemm, int bits) {
  Require(gemm != nullptr,
          "prefill adapter qualification needs the actual callback");
  PrefillDispatch(gemm, bits);
  PrefillUnsafeRows(gemm, bits);
  ExpertInvariance(true, gemm, bits);
}

using TileKey = std::vector<std::uint32_t>;

/// BuildTiles' grouping of `ids` as sorted keys {expert, count, slots...}:
/// per expert, consecutive runs of tile_rows assignments in assignment
/// order. `split` reports an expert spread over several tiles.
std::vector<TileKey> ExpectedTiles(std::span<const std::int32_t> ids,
                                   unsigned experts, unsigned tile_rows,
                                   bool* split = nullptr) {
  std::vector<TileKey> expected;
  for (unsigned e = 0; e < experts; ++e) {
    TileKey tile{e, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (unsigned i = 0; i < ids.size(); ++i) {
      if (ids[i] != static_cast<int>(e))
        continue;
      tile[2 + tile[1]++] = i;
      if (tile[1] == tile_rows) {
        expected.push_back(tile);
        tile = TileKey{e, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if (split != nullptr)
          *split = true;
      }
    }
    if (tile[1] != 0)
      expected.push_back(tile);
  }
  std::sort(expected.begin(), expected.end());
  return expected;
}

/// The device tiles as sorted ExpectedTiles keys. RouterTopKTiles zeroes
/// slots past a tile's count; BuildTiles leaves them stale, so `live_only`
/// compares only the live slots.
std::vector<TileKey> ActualTiles(const gpu::Exl3RouteTile* tiles,
                                 const std::uint32_t* count,
                                 bool live_only = false) {
  std::vector<TileKey> actual;
  for (const auto& t : Read(tiles, Read(count, 1)[0])) {
    TileKey k{t.expert, t.count};
    k.insert(k.end(), std::begin(t.slots), std::end(t.slots));
    for (unsigned r = t.count; live_only && r < 8; ++r)
      k[2 + r] = 0;
    actual.push_back(k);
  }
  std::sort(actual.begin(), actual.end());
  return actual;
}

/// RouterTopKTiles at the production expert count with skewed routing: its
/// IDs equal RouterTopK's and its tiles equal BuildTiles' grouping (per
/// expert, consecutive runs of tile_rows assignments in assignment order)
/// as a set, including experts split over several tiles and every register
/// chunk of the 256-assignment bound.
void RouterTilesSet(unsigned rows, unsigned used) {
  constexpr unsigned experts = 512;
  const unsigned assignments = rows * used;
  const unsigned tile_rows = gpu::Exl3RouteTileRows(rows);
  Buffers buffers;
  std::vector<float> logits(rows * (experts + 1));
  std::mt19937 rng(rows * 131 + used);
  for (unsigned t = 0; t < rows; ++t)
    for (unsigned e = 0; e <= experts; ++e) {
      float v = (int(rng() % 4000) - 2000) * 0.001F;
      // A few hot experts take most tokens; the rest stay random.
      if (e < used + 2 && rng() % 4 != 0)
        v += 8.0F;
      logits[t * (experts + 1) + e] = v;
    }
  auto* dl = buffers.Copy(logits);
  auto* ref_ids = buffers.Make<std::int32_t>(assignments);
  auto* ref_w = buffers.Make<float>(assignments);
  gpu::RouterTopK(dl, experts + 1, ref_ids, ref_w, rows, experts, used,
                  nullptr);
  auto* ids = buffers.Make<std::int32_t>(assignments);
  auto* weights = buffers.Make<float>(assignments);
  auto* tiles = buffers.Make<gpu::Exl3RouteTile>(assignments);
  auto* count = buffers.Make<std::uint32_t>(1);
  auto* arrivals = buffers.Make<std::uint32_t>(1);
  gpu::RouterTopKTiles(dl, experts + 1, ids, weights, rows, experts, used,
                       tiles, count, arrivals, tile_rows, nullptr);
  const auto host_ids = Read(ref_ids, assignments);
  Require(host_ids == Read(ids, assignments), "router tiles changed IDs");
  const auto w0 = Read(ref_w, assignments), w1 = Read(weights, assignments);
  Require(std::memcmp(w0.data(), w1.data(), w0.size() * sizeof(float)) == 0,
          "router tiles changed weights");
  Require(Read(arrivals, 1)[0] == 0, "router tile arrivals not reset");
  bool split = false;
  const auto expected = ExpectedTiles(host_ids, experts, tile_rows, &split);
  const auto actual = ActualTiles(tiles, count);
  Require(actual.size() == expected.size(),
          "router tile count differs from BuildTiles");
  Require(actual == expected, "router tiles differ from BuildTiles grouping");
  Require(split || assignments < tile_rows * 2,
          "router tile fixture never split an expert");
}

// The routed decode GEMV's output bytes against the frozen one-tile kernel
// (exl3_routed_reference.hpp) on the model's shapes (hidden 2560, expert
// width 640, ten experts per token): gate/up, the scaled down projection
// and its widened FP32 control, every tile-row cap R = 1..8, distinct and
// shared experts (1..8-row tiles), K4 and K3.5.
/// A non-empty `routed` replaces the synthetic IDs (rows * used entries).
void RoutedReference(unsigned rows, bool shared, int bits, bool half,
                     std::span<const std::int32_t> routed = {}) {
  namespace ref = gufo::tests::exl3_routed_reference;
  constexpr unsigned k = 2560, ff = 640, used = 10, experts = 24;
  const unsigned assignments = rows * used;
  Buffers buffers;
  Bank gate(buffers, k, ff, experts, 11, false, bits, half),
      up(buffers, k, ff, experts, 12, false, bits, half),
      down(buffers, ff, k, experts, 13, false, bits, half);
  gpu::DeviceLayer layer;
  layer.ffn_gate_exps = gate.weight;
  layer.ffn_up_exps = up.weight;
  layer.ffn_down_exps = down.weight;
  std::vector<float> input(rows * k);
  std::mt19937 rng(17 + rows);
  for (auto& v : input)
    v = (int(rng() % 2000) - 1000) * 0.0005F;
  if (rows > 1)
    std::fill_n(input.end() - k, k, 0.0F);
  std::vector<std::int32_t> ids(assignments);
  for (unsigned i = 0; i < assignments; ++i)
    ids[i] = shared ? i % used : (i * 7 + i / used) % experts;
  if (!routed.empty()) {
    Require(routed.size() == assignments, "routed reference ID count");
    ids.assign(routed.begin(), routed.end());
  }
  auto* dx = buffers.Copy(input);
  auto* di = buffers.Copy(ids);
  auto* g = buffers.Make<float>(assignments * ff);
  auto* u = buffers.Make<float>(assignments * ff);
  auto* y = buffers.Make<float>(assignments * k);
  gpu::Exl3Scratch s;
  s.tiles = buffers.Make<gpu::Exl3RouteTile>(assignments);
  s.tile_count = buffers.Make<std::uint32_t>(1);
  s.down_exponents = buffers.Make<std::int32_t>(assignments);
  s.routed_input = buffers.Make<std::uint16_t>(assignments * 2 * k);
  gpu::Exl3RoutedTrace trace;
  trace.gate_up_input = buffers.Make<std::uint16_t>(2 * assignments * k);
  trace.gate_inner = buffers.Make<float>(assignments * ff);
  trace.up_inner = buffers.Make<float>(assignments * ff);
  trace.down_input = buffers.Make<std::uint16_t>(assignments * ff);
  trace.down_inner = buffers.Make<float>(assignments * k);
  trace.down_widened_inner = buffers.Make<float>(assignments * k);
  gpu::Exl3RoutedMlp(layer, dx, di, g, u, y, rows, used, s, nullptr, &trace);
  Check(hipDeviceSynchronize());
  const unsigned tile_rows = gpu::Exl3RouteTileRows(rows);
  const ex::BitRate rate{bits, half};
  auto* g_ref = buffers.Make<float>(assignments * ff);
  auto* u_ref = buffers.Make<float>(assignments * ff);
  ref::LaunchRate<false, false>(rate, tile_rows, gate.weight.device_exl3,
                                up.weight.device_exl3, s.tiles, s.tile_count,
                                trace.gate_up_input, g_ref, u_ref, k, ff,
                                assignments, 2, nullptr, nullptr);
  // The routed input now holds the scaled down rows and s.down_exponents
  // their exponents (the trace path runs ScaleDownRows).
  auto* d_ref = buffers.Make<float>(assignments * k);
  ref::LaunchRate<false, true>(rate, tile_rows, down.weight.device_exl3,
                               down.weight.device_exl3, s.tiles, s.tile_count,
                               s.routed_input, d_ref, d_ref, ff, k, assignments,
                               1, s.down_exponents, nullptr);
  auto* w_ref = buffers.Make<float>(assignments * k);
  ref::LaunchRate<true, false>(rate, tile_rows, down.weight.device_exl3,
                               down.weight.device_exl3, s.tiles, s.tile_count,
                               trace.down_input, w_ref, w_ref, ff, k,
                               assignments, 1, nullptr, nullptr);
  Check(hipGetLastError());
  Check(hipDeviceSynchronize());
  const auto same = [&](const float* a, const float* b, std::size_t count,
                        const char* what) {
    const auto x = Read(a, count), z = Read(b, count);
    const std::string name =
        std::string("routed GEMV changed bytes vs the frozen kernel: ") + what +
        " rows=" + std::to_string(rows) + (shared ? " shared" : " distinct") +
        " K" + std::to_string(bits) + (half ? ".5" : "");
    Require(std::memcmp(x.data(), z.data(), count * sizeof(float)) == 0,
            name.c_str());
  };
  same(trace.gate_inner, g_ref, assignments * ff, "gate");
  same(trace.up_inner, u_ref, assignments * ff, "up");
  same(trace.down_inner, d_ref, assignments * k, "scaled down");
  same(trace.down_widened_inner, w_ref, assignments * k, "widened down");
  // The production widened down (no exponents) on the same input.
  auto* widened_in = buffers.Make<std::uint16_t>(assignments * ff);
  Check(hipMemcpy(widened_in, trace.down_input,
                  std::size_t{assignments} * ff * sizeof(std::uint16_t),
                  hipMemcpyDeviceToDevice));
  auto* w_out = buffers.Make<float>(assignments * k);
  gpu::Exl3Scratch plain = s;
  plain.down_exponents = nullptr;
  // The tile-row cap, not the row count (equal up to eight rows).
  gpu::Exl3RoutedDown(down.weight, widened_in, w_out, assignments, tile_rows,
                      plain, nullptr);
  Check(hipDeviceSynchronize());
  same(w_out, w_ref, assignments * k, "production widened down");
}

/// Routed rows of `logits` through the fused router tiles against the
/// unfused route, on the model's shapes (hidden 2560, expert width 640, ten
/// of 24 experts): RouterTopKTiles, launched twice on one zero-initialized
/// arrival counter, must keep RouterTopK's IDs and weights, build
/// BuildTiles' grouping, return the counter to zero and, through
/// tiles_ready, give Exl3RoutedMlp's BuildTiles rows bit for bit.
/// Returns the router IDs.
std::vector<std::int32_t> FusedRouterRows(const std::vector<float>& logits,
                                          unsigned rows, const char* what) {
  constexpr unsigned k = 2560, ff = 640, used = 10, experts = 24;
  const unsigned assignments = rows * used;
  const unsigned tile_rows = gpu::Exl3RouteTileRows(rows);
  const std::string name =
      std::string(what) + " rows=" + std::to_string(rows) + ": ";
  const auto require = [&](bool value, const char* message) {
    Require(value, (name + message).c_str());
  };
  Buffers buffers;
  Bank gate(buffers, k, ff, experts, 21, false, 4, false),
      up(buffers, k, ff, experts, 22, false, 4, false),
      down(buffers, ff, k, experts, 23, false, 4, false);
  gpu::DeviceLayer layer;
  layer.ffn_gate_exps = gate.weight;
  layer.ffn_up_exps = up.weight;
  layer.ffn_down_exps = down.weight;
  std::vector<float> input(rows * k);
  std::mt19937 rng(41 + rows);
  for (auto& v : input)
    v = (int(rng() % 2000) - 1000) * 0.0005F;
  auto* dx = buffers.Copy(input);
  auto* dl = buffers.Copy(logits);
  auto* ref_ids = buffers.Make<std::int32_t>(assignments);
  auto* ref_w = buffers.Make<float>(assignments);
  gpu::RouterTopK(dl, experts + 1, ref_ids, ref_w, rows, experts, used,
                  nullptr);
  gpu::Exl3Scratch s;
  s.tiles = buffers.Make<gpu::Exl3RouteTile>(assignments);
  s.tile_count = buffers.Make<std::uint32_t>(1);
  s.down_exponents = buffers.Make<std::int32_t>(assignments);
  s.routed_input =
      buffers.Make<std::uint16_t>(assignments * std::max(2 * k, ff));
  auto* g = buffers.Make<float>(assignments * ff);
  auto* u = buffers.Make<float>(assignments * ff);
  auto* built = buffers.Make<float>(assignments * k);
  gpu::Exl3RoutedMlp(layer, dx, ref_ids, g, u, built, rows, used, s, nullptr);
  const auto host_ids = Read(ref_ids, assignments);
  for (const auto id : host_ids)
    require(id >= 0 && id < static_cast<int>(experts),
            "router ID outside the experts");
  require(ActualTiles(s.tiles, s.tile_count, true) ==
              ExpectedTiles(host_ids, experts, tile_rows),
          "BuildTiles grouping");
  auto* arrivals = buffers.Make<std::uint32_t>(1);
  auto* ids = buffers.Make<std::int32_t>(assignments);
  auto* weights = buffers.Make<float>(assignments);
  auto* fused = buffers.Make<float>(assignments * k);
  const auto a = Read(built, assignments * k);
  for (unsigned launch = 0; launch < 2; ++launch) {
    // Stale tiles from BuildTiles (or the last launch) must be replaced,
    // not merged.
    gpu::RouterTopKTiles(dl, experts + 1, ids, weights, rows, experts, used,
                         s.tiles, s.tile_count, arrivals, tile_rows, nullptr);
    require(Read(arrivals, 1)[0] == 0, "arrivals did not return to zero");
    require(Read(ids, assignments) == host_ids, "fused router changed IDs");
    const auto w0 = Read(ref_w, assignments), w1 = Read(weights, assignments);
    require(std::memcmp(w0.data(), w1.data(), w0.size() * sizeof(float)) == 0,
            "fused router changed weights");
    require(ActualTiles(s.tiles, s.tile_count) ==
                ExpectedTiles(host_ids, experts, tile_rows),
            "fused router tiles differ from BuildTiles grouping");
    Check(hipMemset(fused, 0xFF, std::size_t{assignments} * k * sizeof(float)));
    gpu::Exl3RoutedMlp(layer, dx, ids, g, u, fused, rows, used, s, nullptr,
                       nullptr, true);
    Check(hipDeviceSynchronize());
    const auto b = Read(fused, assignments * k);
    require(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
            "tiles_ready routed rows differ from BuildTiles rows");
  }
  return host_ids;
}

/// Concentrated routing of a wide copy verification's 9..25-row segments:
/// identical router rows (every row picks the same ten experts, so each
/// fills consecutive eight-row tiles) and rows on a few overlapping hot
/// experts. Fused router tiles match BuildTiles, and the routed GEMV on
/// those IDs matches the frozen kernel.
void ConcentratedRouting(unsigned rows, bool identical) {
  std::cout << "CASE concentrated routing rows=" << rows
            << (identical ? " identical\n" : " hot\n");
  constexpr unsigned experts = 24, used = 10;
  std::vector<float> logits(rows * (experts + 1));
  std::mt19937 rng(rows * 7 + (identical ? 1 : 2));
  for (unsigned t = 0; t < rows; ++t)
    for (unsigned e = 0; e <= experts; ++e) {
      if (identical && t != 0) {
        logits[t * (experts + 1) + e] = logits[e];
        continue;
      }
      float v = (int(rng() % 2000) - 1000) * 0.001F;
      // Overlapping rows share a hot set of 13 experts, each row ranking
      // them differently.
      if (!identical && e < used + 3)
        v += 4.0F + static_cast<float>((e * 5 + t * 3) % 7) * 0.5F;
      logits[t * (experts + 1) + e] = v;
    }
  const auto ids = FusedRouterRows(
      logits, rows, identical ? "identical routing" : "hot routing");
  if (identical) {
    const auto tiles =
        ExpectedTiles(ids, experts, gpu::Exl3RouteTileRows(rows));
    Require(tiles.size() == used * ((rows + 7) / 8),
            "identical routing fixture is not concentrated");
  }
  RoutedReference(rows, false, 4, false, ids);
}

/// Router rows without a finite softmax (all NaN, a +Inf or a NaN logit,
/// all -Inf) take the lowest experts at weight 0 instead of ID -1; finite
/// rows keep their IDs and weights, and the fused tiles and routed rows
/// still match the unfused route.
void RouterNonFinite() {
  constexpr unsigned experts = 24, used = 10, rows = 6;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<float> logits(rows * (experts + 1));
  std::mt19937 rng(5);
  for (auto& v : logits)
    v = (int(rng() % 2000) - 1000) * 0.003F;
  const auto row = [&](unsigned t) {
    return logits.begin() + t * (experts + 1);
  };
  std::fill_n(row(1), experts + 1, nan);
  row(2)[7] = inf;
  row(3)[11] = nan;
  std::fill_n(row(5), experts + 1, -inf);
  // Finite rows 0 and 4 alone.
  std::vector<float> finite(row(0), row(1));
  finite.insert(finite.end(), row(4), row(5));
  Buffers buffers;
  auto* dl = buffers.Copy(logits);
  auto* df = buffers.Copy(finite);
  auto* ids = buffers.Make<std::int32_t>(rows * used);
  auto* weights = buffers.Make<float>(rows * used);
  auto* fids = buffers.Make<std::int32_t>(2 * used);
  auto* fweights = buffers.Make<float>(2 * used);
  gpu::RouterTopK(dl, experts + 1, ids, weights, rows, experts, used, nullptr);
  gpu::RouterTopK(df, experts + 1, fids, fweights, 2, experts, used, nullptr);
  const auto i = Read(ids, rows * used), fi = Read(fids, 2 * used);
  const auto w = Read(weights, rows * used), fw = Read(fweights, 2 * used);
  for (unsigned t = 0; t < rows; ++t) {
    const bool bad = t == 1 || t == 2 || t == 3 || t == 5;
    const auto f = t == 0 ? 0U : 1U;
    for (unsigned slot = 0; slot < used; ++slot) {
      const auto id = i[t * used + slot];
      const auto weight = w[t * used + slot];
      Require(id >= 0 && id < static_cast<int>(experts),
              "non-finite router row emitted an invalid expert ID");
      if (bad) {
        Require(id == static_cast<int>(slot) && weight == 0.0F &&
                    !std::signbit(weight),
                "non-finite router row must take the lowest experts at 0");
      } else {
        Require(id == fi[f * used + slot] &&
                    std::bit_cast<std::uint32_t>(weight) ==
                        std::bit_cast<std::uint32_t>(fw[f * used + slot]),
                "finite router row changed beside non-finite rows");
      }
    }
  }
  const auto fused = FusedRouterRows(logits, rows, "non-finite routing");
  Require(fused == i, "fused non-finite routing differs from RouterTopK");
}

/// Waits at most one second for `stream`. A stalled launch ends the process
/// at once: a hung kernel would block every later HIP call, teardown too.
void Watchdog(hipStream_t stream, const std::string& what) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(1);
  for (;;) {
    const auto status = hipStreamQuery(stream);
    if (status == hipSuccess)
      return;
    if (status != hipErrorNotReady)
      Check(status);
    if (std::chrono::steady_clock::now() > deadline) {
      std::cerr << "watchdog: " << what << " did not finish within 1 s\n";
      std::_Exit(3);
    }
    std::this_thread::yield();
  }
}

/// A routed decode layer on the model's shapes (hidden 2560, expert width
/// 640, ten of `experts` experts per row) with router-built IDs on `hot`
/// overlapping experts, as copy verification produces them.
struct RoutedPass {
  static constexpr unsigned k = 2560, ff = 640, used = 10;
  unsigned rows, experts, assignments;
  Buffers buffers;
  Bank gate, up, down;
  gpu::DeviceLayer layer;
  float* x;
  std::int32_t* ids;
  float *g, *u, *out;
  gpu::Exl3Scratch s;
  RoutedPass(unsigned rows, unsigned experts, unsigned hot, unsigned seed)
      : rows(rows),
        experts(experts),
        assignments(rows * used),
        gate(buffers, k, ff, experts, seed, false, 4, false),
        up(buffers, k, ff, experts, seed + 1, false, 4, false),
        down(buffers, ff, k, experts, seed + 2, false, 4, false) {
    layer.ffn_gate_exps = gate.weight;
    layer.ffn_up_exps = up.weight;
    layer.ffn_down_exps = down.weight;
    std::mt19937 rng(seed);
    std::vector<float> input(rows * k);
    for (auto& v : input)
      v = (int(rng() % 2000) - 1000) * 0.0005F;
    std::vector<float> logits(rows * (experts + 1));
    for (unsigned t = 0; t < rows; ++t)
      for (unsigned e = 0; e <= experts; ++e) {
        float v = (int(rng() % 2000) - 1000) * 0.001F;
        if (e < hot)
          v += 4.0F + static_cast<float>((e * 5 + t * 3) % 7) * 0.5F;
        logits[t * (experts + 1) + e] = v;
      }
    x = buffers.Copy(input);
    ids = buffers.Make<std::int32_t>(assignments);
    gpu::RouterTopK(buffers.Copy(logits), experts + 1, ids,
                    buffers.Make<float>(assignments), rows, experts, used,
                    nullptr);
    g = buffers.Make<float>(assignments * ff);
    u = buffers.Make<float>(assignments * ff);
    out = buffers.Make<float>(assignments * k);
    s.tiles = buffers.Make<gpu::Exl3RouteTile>(assignments);
    s.tile_count = buffers.Make<std::uint32_t>(1);
    s.down_exponents = buffers.Make<std::int32_t>(assignments);
    s.routed_input =
        buffers.Make<std::uint16_t>(assignments * std::max(2 * k, ff));
  }
  /// Exl3RoutedMlp's routed rows with `grid_rows` blocks in y (0: default).
  std::vector<float> Run(std::uint32_t grid_rows, bool scaled, bool raw_down,
                         bool tiles_ready = false, hipStream_t stream = nullptr,
                         const std::string& what = "routed pass") {
    auto scratch = s;
    scratch.routed_grid_rows = grid_rows;
    if (!scaled)
      scratch.down_exponents = nullptr;
    Check(hipMemsetAsync(out, 0xFF, std::size_t{assignments} * k * 4, stream));
    gpu::Exl3RoutedMlp(layer, x, ids, g, u, out, rows, used, scratch, stream,
                       nullptr, tiles_ready, raw_down);
    Watchdog(stream, what);
    Check(hipStreamSynchronize(stream));
    return Read(out, assignments * k);
  }
};

bool SameFloats(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

/// The persistent routed grid against one block per live route tile (the
/// old launch's blocks past the tile count exited at once; its full height
/// is the wide grid under suspicion, so it is not relaunched): one block,
/// odd strides and the default cap give the same bytes, for the scaled and
/// widened down routes, restored and raw.
void RoutedGridIdentity(unsigned rows) {
  RoutedPass pass(rows, 24, 20, 61 + rows);
  const auto prefix = "routed grid rows=" + std::to_string(rows);
  (void)pass.Run(0, true, false, false, nullptr, prefix + " tiles");
  const auto live = Read(pass.s.tile_count, 1)[0];
  for (const bool scaled : {true, false})
    for (const bool raw : {false, true}) {
      const auto route = std::string(scaled ? " scaled" : " widened") +
                         (raw ? " raw" : " restored");
      const auto old_grid = pass.Run(live, scaled, raw, false, nullptr,
                                     prefix + route + " per-tile grid");
      for (const std::uint32_t grid : {0U, 1U, 7U, 33U}) {
        const auto name = prefix + route + " grid=" + std::to_string(grid);
        Require(SameFloats(old_grid,
                           pass.Run(grid, scaled, raw, false, nullptr, name)),
                (name + ": persistent grid changed bytes").c_str());
      }
    }
}

/// Test A: a 64-row copy verification layer's route tiles (~100-200 tiles,
/// as in production), replayed through tiles_ready many times, alternately
/// on 80 strided rows and on the default grid; every launch finishes within
/// the watchdog and repeats the bytes of one block per tile.
void RoutedWideReplay() {
  // 64 hot experts: about ten assignments, two tiles, each.
  RoutedPass pass(64, 80, 64, 71);
  constexpr std::uint32_t kStrided = 80;
  (void)pass.Run(0, true, false);
  const auto n = Read(pass.s.tile_count, 1)[0];
  const auto tiles = Read(pass.s.tiles, n);
  std::cout << "CASE routed wide replay tiles=" << n << '\n';
  Require(n > kStrided && n <= pass.assignments,
          "wide replay fixture must exceed the strided grid");
  const auto reference = pass.Run(n, true, false, true);
  hipStream_t stream;
  Check(hipStreamCreate(&stream));
  for (unsigned launch = 0; launch < 200; ++launch) {
    Check(hipMemcpyAsync(pass.s.tiles, tiles.data(),
                         tiles.size() * sizeof(tiles[0]), hipMemcpyHostToDevice,
                         stream));
    Check(hipMemcpyAsync(pass.s.tile_count, &n, sizeof(n),
                         hipMemcpyHostToDevice, stream));
    const auto name = "routed wide replay launch " + std::to_string(launch);
    Require(SameFloats(reference, pass.Run(launch % 2 == 0 ? kStrided : 0, true,
                                           false, true, stream, name.c_str())),
            (name + " changed bytes").c_str());
  }
  Check(hipStreamDestroy(stream));
}

/// Test B: adversarial route tiles for the routed down GEMV (production
/// expert width 640 -> 2560): an empty tile, a count of 9 (past any tile
/// row cap; read as 8), every assignment on one expert, a NaN input row,
/// and tile counts of 0 and of 640 one-row tiles. Every launch finishes in
/// the watchdog and the persistent grid repeats the bytes of one block per
/// tile up to 160 tiles (the old launch's live blocks), strided beyond.
void RoutedAdversarialTiles() {
  std::cout << "CASE routed adversarial tiles\n";
  constexpr unsigned k = 640, n = 2560, experts = 8, assignments = 640;
  Buffers buffers;
  Bank down(buffers, k, n, experts, 81, false, 4, false);
  std::vector<std::uint16_t> input(assignments * k);
  std::mt19937 rng(82);
  for (auto& v : input)
    v = ex::HalfFromDouble((int(rng() % 2000) - 1000) * 0.0005);
  std::fill_n(input.begin() + 5 * k, k, std::uint16_t{0x7e00});  // NaN row
  hipStream_t stream;
  Check(hipStreamCreate(&stream));
  const auto run = [&](const std::vector<gpu::Exl3RouteTile>& tiles,
                       std::uint32_t count, std::uint32_t grid, bool scaled,
                       const std::string& what) {
    gpu::Exl3Scratch s;
    s.tiles = buffers.Copy(tiles);
    s.tile_count = buffers.Copy(std::vector<std::uint32_t>{count});
    s.down_exponents =
        scaled ? buffers.Make<std::int32_t>(assignments) : nullptr;
    s.routed_grid_rows = grid;
    auto* in = buffers.Copy(input);  // the scaled route rescales in place
    auto* out = buffers.Make<float>(std::size_t{assignments} * n);
    Check(hipMemsetAsync(out, 0xFF, std::size_t{assignments} * n * 4, stream));
    gpu::Exl3RoutedDown(down.weight, in, out, assignments, 8, s, stream);
    Watchdog(stream, what);
    Check(hipStreamSynchronize(stream));
    return Read(out, std::size_t{assignments} * n);
  };
  const auto tile = [](unsigned expert, unsigned count, unsigned first) {
    gpu::Exl3RouteTile t{};
    t.expert = expert;
    t.count = count;
    for (unsigned r = 0; r < 8; ++r)
      t.slots[r] = first + r;
    return t;
  };
  std::vector<std::pair<std::string, std::vector<gpu::Exl3RouteTile>>> sets;
  // Empty and over-full tiles beside ordinary ones (and the NaN row 5).
  sets.push_back({"mixed",
                  {tile(0, 0, 0), tile(1, 9, 0), tile(2, 3, 8), tile(3, 8, 16),
                   tile(4, 1, 40)}});
  // Every assignment on one expert, eight per tile.
  std::vector<gpu::Exl3RouteTile> one;
  for (unsigned first = 0; first < assignments; first += 8)
    one.push_back(tile(6, 8, first));
  sets.push_back({"one expert", one});
  // 640 one-row tiles, the old grid's full height.
  std::vector<gpu::Exl3RouteTile> single;
  for (unsigned i = 0; i < assignments; ++i)
    single.push_back(tile(i % experts, 1, i));
  sets.push_back({"640 one-row tiles", single});
  for (const auto& [name, tiles] : sets)
    for (const bool scaled : {false, true}) {
      const auto count = static_cast<std::uint32_t>(tiles.size());
      const auto label = name + (scaled ? " scaled" : " widened");
      // One block per tile up to 160 tiles, strided beyond: short of the
      // 640-row grid under suspicion.
      const auto old_grid = run(tiles, count, std::min(count, 160U), scaled,
                                label + " per-tile grid");
      for (const std::uint32_t grid : {0U, 1U, 7U})
        Require(
            SameFloats(old_grid, run(tiles, count, grid, scaled,
                                     label + " grid " + std::to_string(grid))),
            (label + ": persistent grid changed bytes").c_str());
      // No tiles: nothing is written.
      const auto none = run(tiles, 0, 0, scaled, label + " count 0");
      for (const float v : none)
        Require(std::bit_cast<std::uint32_t>(v) == 0xFFFFFFFFU,
                "a zero tile count wrote routed rows");
    }
  // The empty tile writes nothing and the count-9 tile is its first eight
  // rows (the generic body, against the eight-row body when scaled): the
  // mixed set equals the set with those tiles corrected.
  auto fixed = sets[0].second;
  fixed[1].count = 8;
  fixed.erase(fixed.begin());
  for (const bool scaled : {false, true}) {
    const auto mixed = run(sets[0].second, 5, 0, scaled, "mixed check");
    const auto corrected = run(fixed, 4, 0, scaled, "corrected check");
    Require(SameFloats(mixed, corrected),
            "empty or over-full route tiles changed other rows");
  }
  Check(hipStreamDestroy(stream));
}

// The scaled down projection on DownFixtures' edge rows (signed zeros,
// subnormals, overflow headroom: exponents 0, 10 and -1, i.e. packed,
// scaled and widened-fallback rows) at the model's shape, against the
// frozen kernel, for every tile-row cap and live count.
void RoutedDownReference(int bits, bool half) {
  namespace ref = gufo::tests::exl3_routed_reference;
  constexpr unsigned rows = 8, k = 640, n = 2560;
  Buffers buffers;
  Bank down(buffers, k, n, 1, 5, false, bits, half);
  std::vector<std::uint16_t> input(rows * k);
  for (unsigned j = 0; j < k; ++j) {
    input[j] = j & 1 ? 0x8000 : 0;
    input[k + j] = (j & 1 ? 0x8000 : 0) | 1;
    input[2 * k + j] = (j & 1 ? 0x8000 : 0) | (1 + j % 1023);
    input[3 * k + j] = (j & 1 ? 0x8000 : 0) | 0x0400;
    input[4 * k + j] = j == 0 ? 1 : 0x7bff;
    input[5 * k + j] = j == 0 ? 1 : 0x53ff;
    input[6 * k + j] = j == 0 ? 1 : 0x5400;
    input[7 * k + j] = j % 3 == 0 ? 0x8000 : 0x3c00;
  }
  for (unsigned cap : {1U, 2U, 3U, 4U, 8U}) {
    for (unsigned live = 1; live <= cap; ++live) {
      std::vector<gpu::Exl3RouteTile> tiles;
      for (unsigned first = 0; first < rows; first += live) {
        gpu::Exl3RouteTile part{};
        std::fill(std::begin(part.slots), std::end(part.slots), -1);
        part.count = std::min(live, rows - first);
        for (unsigned r = 0; r < part.count; ++r)
          part.slots[r] = first + r;
        tiles.push_back(part);
      }
      gpu::Exl3Scratch s;
      s.tiles = buffers.Copy(tiles);
      s.tile_count = buffers.Copy(
          std::vector<std::uint32_t>{static_cast<std::uint32_t>(tiles.size())});
      s.down_exponents = buffers.Make<std::int32_t>(rows);
      auto* scaled = buffers.Copy(input);
      auto* out = buffers.Make<float>(rows * n);
      gpu::Exl3RoutedDown(down.weight, scaled, out, rows, cap, s, nullptr);
      auto* expected = buffers.Make<float>(rows * n);
      ref::LaunchRate<false, true>(
          down.rate, gpu::Exl3RouteTileRows(cap), down.weight.device_exl3,
          down.weight.device_exl3, s.tiles, s.tile_count, scaled, expected,
          expected, k, n, rows, 1, s.down_exponents, nullptr);
      Check(hipGetLastError());
      const auto exponents = Read(s.down_exponents, rows);
      Require(
          exponents == std::vector<std::int32_t>({0, 10, 10, 0, -1, 10, -1, 0}),
          "down reference exponent classification");
      const auto a = Read(out, rows * n), b = Read(expected, rows * n);
      Require(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
              "scaled down edge rows changed bytes vs the frozen kernel");
    }
  }
}

// Every codebook x rate (synthetic banks) through the Flash-Next expert
// kernels against FP64: the bucket WMMA (FP32 and FP16 outputs), the routed
// down decode (scaled and widened), the MTP draft head, the routed gate/up/
// down decode at 1, 2 and 8 rows (one and many experts), the prefill expert
// dispatch and the bucket partitions.
void FormatMatrix() {
  for (const auto cb :
       {ex::Codebook::kMcg, ex::Codebook::k3Inst, ex::Codebook::kMul1}) {
    bank_codebook = cb;
    for (int bits = 1; bits <= 8; ++bits) {
      for (const bool half : {false, true}) {
        if (!ex::Supports(cb, {bits, half}))
          continue;
        std::cout << "CASE format " << ex::CodebookName(cb) << ' '
                  << ex::RateName({bits, half}) << '\n';
        HalfBucketStaging(bits, half);
        DownFixtures(bits, half);
        SelectedHeadRate(bits, half);
        for (unsigned rows : {1U, 2U, 8U})
          for (bool shared : {false, true})
            FusedProducer(rows, shared, bits, 0, 0, half);
        ExpertInvariance(false, nullptr, bits, half);
        if (!half) {
          PrefillDispatch(gpu::Exl3PrefillBucketInner, bits);
          for (unsigned tile_rows : {16U, 64U})
            for (bool shared : {false, true})
              Buckets(17, shared, tile_rows, bits);
        }
      }
    }
  }
  bank_codebook = ex::Codebook::kMul1;
}

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--format-matrix") {
      FormatMatrix();
      std::cout << "Flash-Next EXL3 format matrix passed\n";
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--head-generalization") {
      for (int bits = 3; bits <= 8; ++bits)
        SelectedHeadRate(bits);
      SelectedHeadRate(3, true);
      SelectedHeadRate(3, true, 256);
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--dense-generalization") {
      DenseTraversalRates(argv[2]);
      return 0;
    }
    const bool rates_only =
        argc == 2 && std::string(argv[1]) == "--rate-generalization";
    Require(argc == 1 || rates_only, "unknown EXL3 kernel test option");
    for (int bits : {1, 2, 3, 4, 5, 6, 7, 8})
      HalfBucketStaging(bits);
    HalfBucketStaging(3, true);
    HalfBucketStaging(4, false, 2560);
    for (int bits : {3, 4, 5, 6, 7, 8}) {
      std::cout << "CASE expert K=" << bits << '\n';
      bool rejected = false;
      try {
        gpu::Exl3PrefillBucketInner(nullptr, nullptr, nullptr, 1, 16, nullptr,
                                    nullptr, 128, 128, {bits, false},
                                    ex::Codebook::kMul1, nullptr);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      Require(rejected, "bucket adapter must reject null pointers for every K");
      DownFixtures(bits);
      if (!rates_only) {
        SelectedHeadRate(bits);
        QualifyFlashNextExl3PrefillAdapter(ControlGemm, bits);
        QualifyFlashNextExl3PrefillAdapter(gpu::Exl3PrefillBucketInner, bits);
      }
      ExpertInvariance(false, nullptr, bits);
      for (unsigned rows : {1U, 2U, 3U, 4U, 7U, 8U, 9U})
        for (bool shared : {false, true})
          FusedProducer(rows, shared, bits);
      if (!rates_only)
        for (unsigned tile_rows : {16U, 32U, 64U})
          for (unsigned rows : {1U, 3U, 17U, 33U, 65U})
            for (bool shared : {false, true})
              Buckets(rows, shared, tile_rows, bits);
    }
    // Fractional-rate decoding also uses the live-row routed path.
    DownFixtures(3, true);
    DownFixtures(4, false, 272, 80);
    DownFixtures(3, true, 352, 80);
    ExpertInvariance(false, nullptr, 3, true);
    for (unsigned rows : {3U, 4U})
      for (bool shared : {false, true})
        FusedProducer(rows, shared, 3, 0, 0, true);
    // Multi-chunk and partial-group activation staging, an integer rate and
    // K3.5, every tile row width.
    for (unsigned rows : {1U, 2U, 3U, 4U, 8U}) {
      for (bool shared : {false, true})
        FusedProducer(rows, shared, 4, 0, 0, false, 2560);
      FusedProducer(rows, true, 3, 0, 0, true, 2560);
      FusedProducer(rows, false, 5, 0, 0, false, 384);
    }
    for (unsigned rows = 1; rows <= 8; ++rows)
      for (bool shared : {false, true}) {
        RoutedReference(rows, shared, 4, false);
        RoutedReference(rows, shared, 3, true);
      }
    RoutedDownReference(4, false);
    RoutedDownReference(3, true);
    RoutedDownReference(5, false);
    if (rates_only) {
      std::cout << "EXL3 rate generalization contracts passed\n";
      return 0;
    }
    for (const auto [rows, used] :
         {std::pair{1U, 10U}, std::pair{3U, 10U}, std::pair{8U, 10U},
          std::pair{25U, 10U}, std::pair{64U, 4U}})
      RouterTilesSet(rows, used);
    RouterNonFinite();
    for (unsigned rows : {3U, 8U, 9U, 16U, 25U, 64U})
      RoutedGridIdentity(rows);
    RoutedWideReplay();
    RoutedAdversarialTiles();
    for (unsigned rows = 9; rows <= 25; ++rows)
      for (bool identical : {true, false})
        ConcentratedRouting(rows, identical);
    PrefillCombineFrozenRows();
    PrefillCombineFrozenRows(true);
    FusedProducer(3, true, 4, 3, 7);  // Different rates in distinct banks.
    PrefillDispatch(ControlGemm, 4, 3, 7);
    PrefillDispatch(gpu::Exl3PrefillBucketInner, 4, 3, 7);
    // No zero-dimensional GPU launches, and no scratch accesses, for no work.
    gpu::Exl3RoutedDown({}, nullptr, nullptr, 0, 0, {}, nullptr);
    gpu::Exl3PrefillDown({}, 0, 0, {}, nullptr, nullptr);
    gpu::Exl3RoutedMlp({}, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 10,
                       {}, nullptr);
    gpu::Exl3PrefillExperts({}, nullptr, nullptr, nullptr, 0, 10, 16, {},
                            nullptr);
    gpu::Exl3PrefillExperts({}, nullptr, nullptr, nullptr, 0, 10, 16, {},
                            nullptr, nullptr);
    std::cout << "Flash-Next EXL3 expert kernel contracts passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
