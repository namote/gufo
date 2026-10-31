#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "src/core/hip/detail/exl3_gemv.hpp"
#include "src/models/qwen/hip/kernels/exl3_gemv.hpp"
#include "src/models/qwen38_flash_next/exl3_weights.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/exl3_moe.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

// No alternate executor implementation: this access supplies a small resident
// model, seeds the known trunk hidden rows and reads the surviving state of
// real public MtpForward/MtpForwardBatch calls. Production owns no test
// buffers.
struct ExecutorTestAccess {
  static void Hip(hipError_t status) {
    if (status != hipSuccess)
      throw std::runtime_error(hipGetErrorString(status));
  }
  static std::unique_ptr<DeviceModel> Model(const Config& config,
                                            const DeviceLayer& layer,
                                            DeviceTensor embedding,
                                            DeviceTensor output,
                                            DeviceMixer head) {
    std::unique_ptr<DeviceModel> model(new DeviceModel);
    model->config_ = config;
    model->token_embd_ = embedding;
    model->output_ = output;
    model->hc_head_ = head;
    model->layers_.push_back(layer);
    model->mtp_ = layer;
    model->has_mtp_ = true;
    model->max_half_cols_ = config.AttentionQDim();
    model->max_q8_cols_ = config.AttentionQDim();
    return model;
  }
  static void Hidden(Session& session, std::span<const float> hidden) {
    Hip(hipMemcpy(session.mtp_.target_hidden, hidden.data(),
                  hidden.size_bytes(), hipMemcpyHostToDevice));
  }
  static const Exl3PrefillScratch& PrefillScratch(const Executor& exec) {
    return exec.exl3_prefill_;
  }
  static const Exl3Scratch& RoutedScratch(const Executor& exec) {
    return exec.exl3_;
  }
  static float* RoutedOutput(const Executor& exec) { return exec.s_.down_e; }
  struct State {
    std::vector<std::uint16_t> k, v, pooled;
    std::vector<float> raw;
    unsigned position, blocks;
    bool residual_valid;
  };
  template<class T>
  static std::vector<T> Read(const void* device, std::size_t count) {
    std::vector<T> values(count);
    if (count != 0)
      Hip(hipMemcpy(values.data(), device, count * sizeof(T),
                    hipMemcpyDeviceToHost));
    return values;
  }
  static std::vector<float> PrefillResiduals(const Executor& exec,
                                             unsigned rows) {
    return Read<float>(exec.s_.res, std::size_t{rows} * exec.config().HcDim());
  }
  static State ReadState(const Session& session, const Config& config) {
    const auto& m = session.mtp_;
    return {Read<std::uint16_t>(
                m.k_cache, std::size_t{m.position} * config.AttentionKvDim()),
            Read<std::uint16_t>(
                m.v_cache, std::size_t{m.position} * config.AttentionKvDim()),
            Read<std::uint16_t>(
                m.block_k, std::size_t{m.blocks} * config.indexer_head_dim),
            Read<float>(m.index_k, std::size_t{session.index_capacity_} *
                                       config.indexer_head_dim),
            m.position,
            m.blocks,
            m.residual_valid};
  }
  static std::size_t Graphs(const Session& session) {
    // MtpForward's cache-only graph-key bit; ignore unrelated full-body graphs.
    return std::ranges::count_if(session.graphs_, [](const auto& entry) {
      return (entry.first & (std::uint64_t{1} << 45)) != 0;
    });
  }
  static void PoisonSelection(const Executor& exec) {
    Hip(hipMemsetAsync(exec.s_.mask, 0xFF,
                       exec.mask_words_ * sizeof(std::uint32_t),
                       exec.stream()));
  }
  static bool HasExpectedSparseSelection(const Executor& exec,
                                         const Session& session) {
    const auto& c = exec.config();
    const auto complete = session.mtp_.position / c.compress_ratio;
    const auto budget = c.indexer_top_k / c.compress_ratio;
    if (budget == 0 || complete <= budget)
      return false;  // Require actual ranking, beyond the select-all prefix.
    const auto mask = Read<std::uint32_t>(exec.s_.mask, exec.mask_words_);
    unsigned selected = 0;
    for (std::size_t w = 0; w < mask.size(); ++w) {
      selected += std::popcount(mask[w]);
      for (unsigned bit = 0; bit < 32; ++bit)
        if ((mask[w] & (1U << bit)) != 0 && w * 32 + bit >= complete)
          return false;
    }
    return selected == budget;
  }
  static std::vector<float> FullLogits(const Executor& exec,
                                       const Session& session) {
    std::string error;
    if (!exec.MtpHead(exec.model_->mtp().nextn_head, session.mtp_.h, false,
                      false, nullptr, &error))
      throw std::runtime_error(error);
    Hip(hipStreamSynchronize(exec.stream()));
    return Read<float>(exec.s_.logits, exec.config().vocab_size);
  }
  static bool YarnWitness(const Executor& exec, const Session& session) {
    const auto& c = exec.config();
    const auto& m = session.mtp_;
    if (m.blocks == 0 || m.position > session.index_capacity_)
      throw std::runtime_error("YaRN witness requires live raw prefix");
    std::uint32_t* zero = nullptr;
    __half* unscaled = nullptr;
    Hip(hipMalloc(&zero, sizeof(*zero)));
    Hip(hipMemset(zero, 0, sizeof(*zero)));
    const auto count = std::size_t{m.blocks} * c.indexer_head_dim;
    Hip(hipMalloc(&unscaled, count * sizeof(__half)));
    PoolIndexerBlocks(m.index_k, exec.model_->mtp().indexer_k_norm.f32(),
                      unscaled, zero, zero, m.position, m.blocks,
                      c.compress_ratio, c.indexer_head_dim, c.rotary_dim,
                      c.rope_theta, c.rms_eps, session.index_capacity_,
                      exec.stream());
    Hip(hipStreamSynchronize(exec.stream()));
    const auto control = Read<std::uint16_t>(unscaled, count);
    const auto actual = Read<std::uint16_t>(m.block_k, count);
    Hip(hipFree(unscaled));
    Hip(hipFree(zero));
    return actual != control;
  }
};
}  // namespace gufo::models::qwen38_flash_next::rocm

namespace q = gufo::models::qwen38_flash_next;
namespace gpu = q::rocm;
namespace ex = gufo::quant::exl3;
namespace ar = gufo::hip::exl3;
namespace {
std::string current_case = "initialization";

void BeginCase(const std::string& name) {
  current_case = name;
  std::cout << "CASE " << current_case << '\n';
}

std::string Shape(const std::string& operation, unsigned k, unsigned n,
                  int bits, unsigned rows) {
  return operation + " k=" + std::to_string(k) + " n=" + std::to_string(n) +
         " K=" + std::to_string(bits) + " rows=" + std::to_string(rows);
}

void Require(bool value, const std::string& message) {
  std::cout << (value ? "PASS " : "FAIL ") << current_case << " : " << message
            << '\n';
  if (!value)
    throw std::runtime_error(current_case + " : " + message);
}

void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(current_case + " : " + hipGetErrorString(status));
}

struct Buffers {
  std::vector<void*> allocations;
  ~Buffers() {
    for (void* p : allocations)
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
  T* Copy(const std::vector<T>& values) {
    T* p = Make<T>(values.size());
    Check(hipMemcpy(p, values.data(), values.size() * sizeof(T),
                    hipMemcpyHostToDevice));
    return p;
  }
};

struct Bank {
  unsigned k, n, experts;
  int bits;
  std::vector<std::uint16_t> trellis, suh, svh;
  std::vector<ex::DeviceLinear> descriptors;
  gpu::DeviceTensor weight;

  Bank(Buffers& buffers, unsigned inputs, unsigned outputs, int rate,
       unsigned count = 1, unsigned seed = 12345)
      : k(inputs), n(outputs), experts(count), bits(rate) {
    const auto words = std::size_t{k / 16} * (n / 16) * 16 * bits;
    trellis.resize(words * experts);
    suh.resize(std::size_t{k} * experts);
    svh.resize(std::size_t{n} * experts);
    std::mt19937 rng(seed);
    for (auto& x : trellis)
      x = static_cast<std::uint16_t>(rng());
    for (auto& x : suh)
      x = ex::HalfFromDouble((rng() & 1 ? -1 : 1) *
                             (0.03 + (rng() % 100) * 0.001));
    for (auto& x : svh)
      x = ex::HalfFromDouble((rng() & 1 ? -1 : 1) *
                             (0.03 + (rng() % 100) * 0.001));
    auto* dt = buffers.Copy(trellis);
    auto* ds = buffers.Copy(suh);
    auto* dv = buffers.Copy(svh);
    for (unsigned e = 0; e < experts; ++e)
      descriptors.push_back({dt + e * words,
                             ds + e * k,
                             dv + e * n,
                             k,
                             n,
                             {bits, false},
                             ex::Codebook::kMul1});
    weight.data = dt;
    weight.cols = k;
    weight.rows = n;
    weight.experts = experts;
    weight.exl3 = descriptors.data();
    weight.device_exl3 = buffers.Copy(descriptors);
  }

  // Match the specified FP32 producer before the FP16 boundary. The core
  // HadamardBlocks(float) helper multiplies by a double scale, which can
  // select the neighbouring FP16 bin and is not this arithmetic contract.
  std::vector<float> RotatedInput(unsigned expert, const float* x,
                                  bool rounded) const {
    std::vector<float> rotated(k);
    for (unsigned base = 0; base < k; base += 128) {
      float segment[128];
      for (unsigned i = 0; i < 128; ++i) {
        volatile float product =
            x[base + i] *
            static_cast<float>(ex::HalfToDouble(suh[expert * k + base + i]));
        segment[i] = product;
      }
      for (unsigned length = 1; length < 128; length *= 2)
        for (unsigned first = 0; first < 128; first += 2 * length)
          for (unsigned i = first; i < first + length; ++i) {
            const float a = segment[i], b = segment[i + length];
            segment[i] = a + b;
            segment[i + length] = a - b;
          }
      for (unsigned i = 0; i < 128; ++i) {
        volatile float scaled = segment[i] * 0.0883883461356163F;
        const float value = scaled;
        rotated[base + i] = rounded ? static_cast<float>(ex::HalfToDouble(
                                          ex::HalfFromDouble(value)))
                                    : value;
      }
    }
    return rotated;
  }

  // Diagnostic control reproducing the original test's double-scaled input.
  std::vector<float> LegacyRotatedInput(unsigned expert, const float* x,
                                        bool rounded) const {
    std::vector<float> rotated(k);
    for (unsigned i = 0; i < k; ++i)
      rotated[i] =
          x[i] * static_cast<float>(ex::HalfToDouble(suh[expert * k + i]));
    ex::HadamardBlocks(rotated);
    if (rounded)
      for (auto& value : rotated)
        value = static_cast<float>(ex::HalfToDouble(ex::HalfFromDouble(value)));
    return rotated;
  }

  // The inner dot product and output Hadamard are independent FP64 formulas.
  std::vector<float> ProjectRotated(unsigned expert,
                                    std::span<const float> rotated,
                                    std::vector<float>* raw = nullptr) const {
    const auto words = std::size_t{k / 16} * (n / 16) * 16 * bits;
    std::vector<std::uint16_t> inner(std::size_t{k} * n);
    ex::DequantInner(trellis.data() + expert * words, k, n, {bits, false},
                     ex::Codebook::kMul1, inner.data());
    std::vector<double> y(n);
    for (unsigned j = 0; j < n; ++j)
      for (unsigned i = 0; i < k; ++i)
        y[j] += rotated[i] * ex::HalfToDouble(inner[std::size_t{i} * n + j]);
    if (raw != nullptr)
      raw->assign(y.begin(), y.end());
    ex::HadamardBlocks(y);
    std::vector<float> result(n);
    for (unsigned j = 0; j < n; ++j)
      result[j] =
          static_cast<float>(y[j] * ex::HalfToDouble(svh[expert * n + j]));
    return result;
  }

  std::vector<float> Reference(unsigned expert, const float* x, bool rounded,
                               std::vector<float>* raw = nullptr) const {
    return ProjectRotated(expert, RotatedInput(expert, x, rounded), raw);
  }
};

gpu::Exl3Scratch Scratch(Buffers& buffers, unsigned rows, unsigned k,
                         unsigned n, unsigned used = 10) {
  gpu::Exl3Scratch s;
  auto& p = s.projection;
  p.rotated_elements = std::size_t{rows} * k;
  p.rotated = buffers.Make<float>(p.rotated_elements);
  p.weight_elements = std::size_t{128} * k;
  p.weight = buffers.Make<float>(p.weight_elements);
  p.fast_input_elements = std::size_t{24} * k;
  p.fast_input = buffers.Make<std::uint16_t>(p.fast_input_elements);
  p.fast_down_input_elements = std::size_t{8} * k;
  p.fast_down_input = buffers.Make<std::uint16_t>(p.fast_down_input_elements);
  p.fast_counter_count = 3 * n / 128;
  p.fast_counters = buffers.Make<std::uint32_t>(p.fast_counter_count);
  s.slice_elements = std::size_t{rows} * n;
  s.slice = buffers.Make<float>(s.slice_elements);
  s.tiles = buffers.Make<gpu::Exl3RouteTile>(rows * used);
  s.tile_count = buffers.Make<std::uint32_t>(1);
  s.down_exponents = buffers.Make<std::int32_t>(rows * used);
  s.routed_input =
      buffers.Make<std::uint16_t>(std::size_t{2} * rows * used * k);
  s.head_segments = buffers.Make<std::uint32_t>(33);
  s.head_values = buffers.Make<float>(n + 32 * 128);
  return s;
}

std::vector<float> Input(unsigned count) {
  std::mt19937 rng(56234);
  std::vector<float> x(count);
  for (auto& value : x)
    value = (static_cast<int>(rng() % 2000) - 1000) * 0.00003125F;
  return x;
}
double RelativeL2(std::span<const float> a, std::span<const float> b) {
  if (a.size() != b.size())
    throw std::runtime_error(current_case + " : comparison size mismatch");
  double error = 0, energy = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
      throw std::runtime_error(current_case + " : nonfinite output at " +
                               std::to_string(i));
    error += std::pow(static_cast<double>(a[i]) - b[i], 2);
    energy += static_cast<double>(b[i]) * b[i];
  }
  return std::sqrt(error / std::max(energy, 1e-30));
}

void Near(std::span<const float> a, std::span<const float> b, double tolerance,
          const std::string& check) {
  std::cout << "CHECK " << current_case << " : " << check
            << " bar=" << tolerance << '\n';
  const auto relative = RelativeL2(a, b);
  std::cout << (relative <= tolerance ? "PASS " : "FAIL ") << current_case
            << " : " << check << " relative_L2=" << std::scientific
            << std::setprecision(9) << relative << " bar=" << tolerance << '\n';
  if (relative > tolerance)
    throw std::runtime_error(current_case + " : " + check + " relative L2 " +
                             std::to_string(relative));
}

void InputRoundingControl() {
  BeginCase("input rounding control k=128 n=128 K=5 rows=1 suh=1");
  const float impulse = std::bit_cast<float>(std::uint32_t{0x36931406});
  std::vector<float> legacy(128);
  legacy[0] = impulse;
  ex::HadamardBlocks(legacy);
  volatile float matched = impulse * 0.0883883461356163F;
  const auto old_bits = ex::HalfFromDouble(legacy[0]);
  const auto matched_bits = ex::HalfFromDouble(matched);
  std::cout << "CONTROL " << current_case << " : legacy_FP16=" << old_bits
            << " matched_FP16=" << matched_bits << '\n';
  Require(old_bits == 7 && matched_bits == 6,
          "double vs FP32 scale crosses an FP16 rounding boundary");
}

template<int K>
__global__ void Decode(const std::uint16_t* trellis, std::uint16_t* out) {
  const unsigned lane = threadIdx.x;
  ar::Half2 w[4];
  ar::DecodeLaneMul1<K>(
      reinterpret_cast<const std::uint32_t*>(trellis) + blockIdx.x * 8 * K,
      ar::MakeLaneWindow<K>(lane), w);
  const unsigned r0 = lane % 4 * 2;
  const unsigned c0 = lane / 4;
  for (unsigned j = 0; j < 8; ++j) {
    const unsigned row = r0 + (j % 2) + (j % 4 / 2) * 8;
    const unsigned col = c0 + (j / 4) * 8;
    const auto pair = __builtin_bit_cast(std::uint32_t, w[j / 2]);
    out[blockIdx.x * 256 + row * 16 + col] =
        static_cast<std::uint16_t>(pair >> (j % 2 * 16));
  }
}
void DecodeK7() {
  BeginCase("decode k=16 n=16 K=7 tiles=256");
  Buffers buffers;
  std::mt19937 rng(998877);
  std::vector<std::uint16_t> trellis(112 * 256), expected(256 * 256),
      actual(expected.size());
  for (auto& word : trellis)
    word = static_cast<std::uint16_t>(rng());
  for (unsigned i = 0; i < 256; ++i)
    ex::DequantTile(
        trellis.data() + i * 112, {7, false}, ex::Codebook::kMul1,
        std::span<std::uint16_t, 256>(expected.data() + i * 256, 256));
  auto* t = buffers.Copy(trellis);
  auto* y = buffers.Make<std::uint16_t>(actual.size());
  hipLaunchKernelGGL((Decode<7>), dim3(256), dim3(32), 0, nullptr, t, y);
  Check(hipMemcpy(actual.data(), y, actual.size() * 2, hipMemcpyDeviceToHost));
  Require(actual == expected, "K7 lane decoder differs in FP16 bits");
}

void Projection(unsigned k, unsigned n, int bits,
                std::initializer_list<unsigned> sizes = {1U, 2U, 3U, 7U, 8U, 9U,
                                                         32U, 33U, 64U, 65U}) {
  Buffers buffers;
  Bank weight(buffers, k, n, bits);
  const auto x = Input(65 * k);
  auto* dx = buffers.Copy(x);
  auto* y = buffers.Make<float>(65 * n + 16);
  auto scratch = Scratch(buffers, 65, std::max(k, n), n);
  for (unsigned rows : sizes) {
    BeginCase(Shape("projection fast", k, n, bits, rows));
    Check(hipMemset(y, 0x7F, (65 * n + 16) * sizeof(float)));
    gpu::Exl3Projection(weight.weight, dx, y, rows, scratch, nullptr);
    std::vector<float> actual(rows * n), expected, legacy;
    std::size_t legacy_changes = 0;
    Check(hipMemcpy(actual.data(), y, actual.size() * sizeof(float),
                    hipMemcpyDeviceToHost));
    for (unsigned r = 0; r < rows; ++r) {
      const auto input = weight.RotatedInput(0, x.data() + r * k, true);
      const auto old_input =
          weight.LegacyRotatedInput(0, x.data() + r * k, true);
      for (unsigned i = 0; i < k; ++i)
        legacy_changes += input[i] != old_input[i];
      const auto reference = weight.ProjectRotated(0, input);
      const auto old_reference =
          input == old_input ? reference : weight.ProjectRotated(0, old_input);
      expected.insert(expected.end(), reference.begin(), reference.end());
      legacy.insert(legacy.end(), old_reference.begin(), old_reference.end());
    }
    std::cout << "CONTROL " << current_case
              << " : legacy_reference_L2=" << RelativeL2(actual, legacy)
              << " corrected_reference_L2=" << RelativeL2(actual, expected)
              << " changed_input_values=" << legacy_changes << '\n';
    Near(actual, expected, 1e-5, "projection vs FP64 with matched input");
    std::uint32_t canary;
    Check(hipMemcpy(&canary, y + rows * n, 4, hipMemcpyDeviceToHost));
    Require(canary == 0x7F7F7F7FU, "projection wrote beyond row allocation");
    std::vector<std::uint32_t> counters(scratch.projection.fast_counter_count);
    Check(hipMemcpy(counters.data(), scratch.projection.fast_counters,
                    counters.size() * 4, hipMemcpyDeviceToHost));
    Require(std::all_of(counters.begin(), counters.end(),
                        [](auto v) { return v == 0; }),
            "projection counters not reset");
  }
  if (bits == 5 && n == 640) {
    auto view = weight.weight;
    for (unsigned begin : {0U, 512U}) {
      BeginCase(
          Shape("indexer slice begin=" + std::to_string(begin), k, n, bits, 3));
      view.output_begin = begin;
      view.rows = begin == 0 ? 512 : 128;
      gpu::Exl3Projection(view, dx, y, 3, scratch, nullptr);
      std::vector<float> actual(3 * view.rows), expected;
      Check(hipMemcpy(actual.data(), y, actual.size() * 4,
                      hipMemcpyDeviceToHost));
      for (unsigned r = 0; r < 3; ++r) {
        const auto reference = weight.Reference(0, x.data() + r * k, true);
        expected.insert(expected.end(), reference.begin() + begin,
                        reference.begin() + begin + view.rows);
      }
      Near(actual, expected, 1e-5, "restored output slice vs FP64");
    }
  }
}

void Routed(bool shared, unsigned rows, unsigned k = 128, unsigned ff = 128,
            unsigned experts = 512) {
  BeginCase(
      Shape(shared ? "routed shared" : "routed disjoint", k, ff, 5, rows) +
      " down_n=" + std::to_string(k) + " experts=" + std::to_string(experts) +
      " topk=10");
  Buffers buffers;
  constexpr unsigned used = 10;
  Bank gate(buffers, k, ff, 5, experts, 1), up(buffers, k, ff, 5, experts, 2),
      down(buffers, ff, k, 5, experts, 3);
  gpu::DeviceLayer layer;
  layer.ffn_gate_exps = gate.weight;
  layer.ffn_up_exps = up.weight;
  layer.ffn_down_exps = down.weight;
  const auto x = Input(rows * k);
  std::vector<std::int32_t> ids(rows * used);
  for (unsigned r = 0; r < rows; ++r)
    for (unsigned j = 0; j < used; ++j)
      ids[r * used + j] = shared ? j : (r * used + j) % experts;
  auto* dx = buffers.Copy(x);
  auto* di = buffers.Copy(ids);
  auto* g = buffers.Make<float>(rows * used * ff);
  auto* u = buffers.Make<float>(rows * used * ff);
  auto* y = buffers.Make<float>(rows * used * k);
  auto scratch = Scratch(buffers, rows, k, k, used);
  const auto assignments = rows * used;
  gpu::Exl3RoutedTrace trace;
  trace.gate_up_input = buffers.Make<std::uint16_t>(2 * assignments * k);
  trace.gate_inner = buffers.Make<float>(assignments * ff);
  trace.up_inner = buffers.Make<float>(assignments * ff);
  trace.activation = buffers.Make<float>(assignments * ff);
  trace.down_input = buffers.Make<std::uint16_t>(assignments * ff);
  trace.down_inner = buffers.Make<float>(assignments * k);
  trace.down_packed_inner = buffers.Make<float>(assignments * k);
  gpu::Exl3RoutedMlp(layer, dx, di, g, u, y, rows, used, scratch, nullptr,
                     &trace);
  const auto download = []<class T>(const T* pointer, std::size_t count) {
    std::vector<T> result(count);
    Check(hipMemcpy(result.data(), pointer, count * sizeof(T),
                    hipMemcpyDeviceToHost));
    return result;
  };
  const auto actual = download(y, std::size_t{assignments} * k);
  const auto gate_up_input =
      download(trace.gate_up_input, std::size_t{2} * assignments * k);
  const auto gate_inner =
      download(trace.gate_inner, std::size_t{assignments} * ff);
  const auto up_inner = download(trace.up_inner, std::size_t{assignments} * ff);
  const auto activation =
      download(trace.activation, std::size_t{assignments} * ff);
  const auto down_input =
      download(trace.down_input, std::size_t{assignments} * ff);
  const auto down_inner =
      download(trace.down_inner, std::size_t{assignments} * k);
  const auto down_packed_inner =
      download(trace.down_packed_inner, std::size_t{assignments} * k);
  std::vector<float> expected_gate, expected_up, expected_activation,
      expected_down;
  std::vector<float> restored_down;
  std::vector<float> ideal, legacy, matched;
  std::size_t producer_mismatches = 0, down_mismatches = 0,
              ideal_down_changes = 0;
  const auto compare_half = [&](std::uint16_t expected, std::uint16_t observed,
                                unsigned slot, unsigned column,
                                const char* stage, std::size_t& mismatches) {
    if (expected != observed && mismatches++ == 0)
      std::cout << "CONTROL " << current_case << " : first " << stage
                << " mismatch slot=" << slot << " column=" << column
                << " expected_FP16=" << expected
                << " observed_FP16=" << observed << '\n';
  };
  for (unsigned slot = 0; slot < assignments; ++slot) {
    const auto expert = ids[slot];
    const float* input = x.data() + (slot / used) * k;
    const auto gate_rotated = gate.RotatedInput(expert, input, true);
    const auto up_rotated = up.RotatedInput(expert, input, true);
    for (unsigned i = 0; i < k; ++i) {
      compare_half(ex::HalfFromDouble(gate_rotated[i]),
                   gate_up_input[std::size_t{slot} * k + i], slot, i,
                   "gate producer", producer_mismatches);
      compare_half(ex::HalfFromDouble(up_rotated[i]),
                   gate_up_input[(std::size_t{assignments} + slot) * k + i],
                   slot, i, "up producer", producer_mismatches);
    }
    std::vector<float> raw;
    auto gv = gate.ProjectRotated(expert, gate_rotated, &raw);
    expected_gate.insert(expected_gate.end(), raw.begin(), raw.end());
    const auto uv = up.ProjectRotated(expert, up_rotated, &raw);
    expected_up.insert(expected_up.end(), raw.begin(), raw.end());
    for (unsigned j = 0; j < ff; ++j)
      gv[j] = (gv[j] / (1.0F + std::exp(-gv[j]))) * uv[j];
    expected_activation.insert(expected_activation.end(), gv.begin(), gv.end());
    const auto old_gate_input = gate.LegacyRotatedInput(expert, input, true);
    const auto old_up_input = up.LegacyRotatedInput(expert, input, true);
    std::vector<float> old_activation;
    if (old_gate_input == gate_rotated && old_up_input == up_rotated) {
      old_activation = gv;
    } else {
      old_activation = gate.ProjectRotated(expert, old_gate_input);
      const auto old_up = up.ProjectRotated(expert, old_up_input);
      for (unsigned j = 0; j < ff; ++j)
        old_activation[j] =
            (old_activation[j] / (1.0F + std::exp(-old_activation[j]))) *
            old_up[j];
    }
    const auto old_down_input =
        down.LegacyRotatedInput(expert, old_activation.data(), true);
    const auto ideal_rotated = down.RotatedInput(expert, gv.data(), true);
    const auto actual_rotated = down.RotatedInput(
        expert, activation.data() + std::size_t{slot} * ff, true);
    std::vector<float> frozen(ff);
    for (unsigned j = 0; j < ff; ++j) {
      const auto observed = down_input[std::size_t{slot} * ff + j];
      compare_half(ex::HalfFromDouble(actual_rotated[j]), observed, slot, j,
                   "down producer", down_mismatches);
      ideal_down_changes += ex::HalfFromDouble(ideal_rotated[j]) != observed;
      frozen[j] = static_cast<float>(ex::HalfToDouble(observed));
    }
    const auto ideal_result = down.ProjectRotated(expert, ideal_rotated);
    const auto legacy_result =
        old_down_input == ideal_rotated
            ? ideal_result
            : down.ProjectRotated(expert, old_down_input);
    legacy.insert(legacy.end(), legacy_result.begin(), legacy_result.end());
    const auto matched_result = down.ProjectRotated(expert, frozen, &raw);
    expected_down.insert(expected_down.end(), raw.begin(), raw.end());
    const auto actual_inner =
        std::span(down_inner).subspan(std::size_t{slot} * k, k);
    const auto packed_inner =
        std::span(down_packed_inner).subspan(std::size_t{slot} * k, k);
    std::vector<double> restored(actual_inner.begin(), actual_inner.end());
    ex::HadamardBlocks(std::span<double>(restored));
    for (unsigned j = 0; j < k; ++j)
      restored_down.push_back(static_cast<float>(
          restored[j] * ex::HalfToDouble(down.svh[expert * k + j])));
    if (k == 128 && ff == 128 && experts == 512 && rows == 1) {
      unsigned subnormals = 0;
      for (unsigned j = 0; j < ff; ++j) {
        const auto bits = down_input[std::size_t{slot} * ff + j] & 0x7fffU;
        subnormals += bits != 0 && bits < 0x400U;
      }
      std::cout << "CONTROL " << current_case << " : down slot=" << slot
                << " expert=" << expert << " subnormal_inputs=" << subnormals
                << " packed_inner_L2=" << RelativeL2(packed_inner, raw)
                << " scaled_inner_L2=" << RelativeL2(actual_inner, raw)
                << " restored_L2="
                << RelativeL2(
                       std::span(actual).subspan(std::size_t{slot} * k, k),
                       matched_result)
                << '\n';
    }
    ideal.insert(ideal.end(), ideal_result.begin(), ideal_result.end());
    matched.insert(matched.end(), matched_result.begin(), matched_result.end());
  }
  // Preserve the old oracle as a measured control, not a relaxed bar. A
  // separately accumulated FP64 gate/up oracle can cross a different down
  // FP16 bin. Freeze that boundary and require 1e-5 at each projection,
  // plus exact producer bits. No tolerance grows with K.
  const double frozen_input_control = RelativeL2(matched, legacy);
  const double e2e_bound = frozen_input_control * 1.05 + 1e-5;
  std::cout << "CONTROL " << current_case
            << " : legacy_E2E_L2=" << RelativeL2(actual, legacy)
            << " matched_control_bound=" << e2e_bound
            << " corrected_ideal_E2E_L2=" << RelativeL2(actual, ideal)
            << " frozen_input_control_L2=" << frozen_input_control
            << " actual_vs_frozen_FP64_L2=" << RelativeL2(actual, matched)
            << " down_FP16_changes=" << ideal_down_changes
            << " packed_down_inner_L2="
            << RelativeL2(down_packed_inner, expected_down)
            << " scaled_down_inner_L2=" << RelativeL2(down_inner, expected_down)
            << '\n';
  Require(producer_mismatches == 0,
          "gate/up FP32 rotation + FP16 RNE mismatches=" +
              std::to_string(producer_mismatches));
  Near(gate_inner, expected_gate, 1e-5, "gate inner dot vs FP64 matched input");
  Near(up_inner, expected_up, 1e-5, "up inner dot vs FP64 matched input");
  Near(activation, expected_activation, 1e-5,
       "restored svh/H and SwiGLU vs FP64 projections");
  Require(down_mismatches == 0, "down FP32 rotation + FP16 RNE mismatches=" +
                                    std::to_string(down_mismatches));
  Near(down_inner, expected_down, 1e-5,
       "down inner scaled vs FP64 matched input");
  Near(actual, restored_down, 1e-5,
       "down restoration vs FP64 with frozen GPU inner output");
  Near(actual, matched, 1e-5, "down output vs FP64 with frozen GPU FP16 input");
  // The FP16 intermediate contract can move a down-input bin relative to
  // independently accumulated FP64 gate/up results. An absolute 2e-4 E2E
  // bar therefore does not apply. Bound the measured displacement by its
  // matched frozen-input control, while retaining the strict stage checks.
  Near(actual, legacy, e2e_bound, "E2E vs frozen-input matched-control bound");
  if (rows == 3) {
    hipStream_t stream;
    Check(hipStreamCreate(&stream));
    Check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    gpu::Exl3RoutedMlp(layer, dx, di, g, u, y, rows, used, scratch, stream);
    hipGraph_t graph;
    Check(hipStreamEndCapture(stream, &graph));
    hipGraphExec_t executable;
    Check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    Check(hipGraphLaunch(executable, stream));
    Check(hipStreamSynchronize(stream));
    std::vector<float> replay(actual.size());
    Check(
        hipMemcpy(replay.data(), y, replay.size() * 4, hipMemcpyDeviceToHost));
    Require(replay == actual, "routed graph replay changed slot results");
    Check(hipGraphExecDestroy(executable));
    Check(hipGraphDestroy(graph));
    Check(hipStreamDestroy(stream));
  }
}

void CompressedPrefillChunks() {
  constexpr unsigned k = 256, n = 640, rows = 136;
  BeginCase(Shape("compressed prefill chunks", k, n, 7, rows));
  Buffers buffers;
  Bank weight(buffers, k, n, 7);
  const auto x = Input(rows * k);
  const auto* dx = buffers.Copy(x);
  auto* y = buffers.Make<float>(rows * n);
  auto scratch = Scratch(buffers, rows, std::max(k, n), n);
  bool fast = false;
  const auto project = [&](unsigned first, unsigned count) {
    gpu::Exl3Projection(weight.weight, dx + first * k, y + first * n, count,
                        scratch, nullptr, fast);
  };
  const auto download = [&] {
    std::vector<float> result(rows * n);
    Check(hipMemcpy(result.data(), y, result.size() * sizeof(float),
                    hipMemcpyDeviceToHost));
    return result;
  };
  for (bool mode : {false, true}) {
    fast = mode;
    project(0, rows);
    const auto bulk = download();
    for (unsigned boundary : {94U, 103U, 104U, 127U, 128U, 135U}) {
      project(0, boundary);
      project(boundary, rows - boundary);
      const auto split = download();
      Require(std::memcmp(bulk.data(), split.data(), bulk.size() * 4) == 0,
              std::string(fast ? "fast catch-up" : "reference prefill") +
                  " differs at boundary " + std::to_string(boundary));
    }
  }
}

void SharedMlpInvariance() {
  constexpr unsigned k = 2560, ff = 640, rows = 65;
  BeginCase(Shape("shared MLP scalar/cohort", k, ff, 7, rows));
  Buffers buffers;
  Bank gate(buffers, k, ff, 7, 1, 4), up(buffers, k, ff, 7, 1, 5),
      down(buffers, ff, k, 7, 1, 6);
  gpu::DeviceLayer layer;
  layer.shexp_gate = gate.weight;
  layer.shexp_up = up.weight;
  layer.shexp_down = down.weight;
  auto x = Input(rows * k);
  for (auto& value : x)
    value *= 32.0F;
  const auto* dx = buffers.Copy(x);
  auto* g = buffers.Make<float>(rows * ff);
  auto* u = buffers.Make<float>(rows * ff);
  auto* y = buffers.Make<float>(rows * k);
  auto scratch = Scratch(buffers, rows, k, k);
  const auto download = [&] {
    std::vector<float> result(rows * k);
    Check(hipMemcpy(result.data(), y, result.size() * sizeof(float),
                    hipMemcpyDeviceToHost));
    return result;
  };
  for (unsigned r = 0; r < rows; ++r)
    Require(gpu::Exl3SharedMlp(layer, dx + r * k, g, u, y + r * k, 1, scratch,
                               nullptr),
            "serial shared MLP was not dispatched");
  const auto serial = download();
  // Reproduce the former MoeBatch path as a diagnostic: its native __expf
  // activation differs from the serial fused MLP's expf/division contract.
  gpu::Exl3Projection(gate.weight, dx, g, rows, scratch, nullptr);
  gpu::Exl3Projection(up.weight, dx, u, rows, scratch, nullptr);
  gpu::Swiglu(g, u, std::size_t{rows} * ff, nullptr);
  gpu::Exl3Projection(down.weight, g, y, rows, scratch, nullptr);
  std::cout << "CONTROL " << current_case << " : former_unfused_vs_serial_L2="
            << RelativeL2(download(), serial) << '\n';
  for (unsigned count : {2U, 4U, 6U, 8U, 9U, 18U, 29U, rows}) {
    Require(gpu::Exl3SharedMlp(layer, dx, g, u, y, count, scratch, nullptr),
            "cohort shared MLP was not dispatched");
    const auto actual = download();
    Require(
        std::memcmp(actual.data(), serial.data(),
                    std::size_t{count} * k * sizeof(float)) == 0,
        "shared MLP scalar/cohort bits differ rows=" + std::to_string(count));
  }
}

void SelectedHead() {
  BeginCase(Shape("selected head prefix=128 extras=8", 256, 640, 6, 1));
  Buffers buffers;
  Bank head(buffers, 256, 640, 6);
  const auto x = Input(256);
  const std::vector<std::int32_t> ids{128, 255, 256, -1, 639, 638, 128, 127};
  auto* dx = buffers.Copy(x);
  auto* di = buffers.Copy(ids);
  auto* logits = buffers.Make<float>(128 + ids.size());
  auto s = Scratch(buffers, 1, 640, 640);
  gpu::Exl3DraftHead(head.weight, dx, logits, 128, di, ids.size(), s, nullptr);
  std::vector<float> result(128 + ids.size());
  Check(hipMemcpy(result.data(), logits, result.size() * 4,
                  hipMemcpyDeviceToHost));
  const auto reference = head.Reference(0, x.data(), true);
  Near(std::span(result).first(128), std::span(reference).first(128), 1e-5,
       "selected prefix vs FP64 matched input");
  for (unsigned i = 0; i < ids.size(); ++i) {
    if (ids[i] < 0)
      Require(result[128 + i] == -INFINITY,
              "invalid draft slot=" + std::to_string(i) + " was not masked");
    else
      Require(std::abs(result[128 + i] - reference[ids[i]]) < 1e-7F,
              "selected head slot=" + std::to_string(i) +
                  " id=" + std::to_string(ids[i]) + " absolute error < 1e-7");
  }
  unsigned count;
  Check(hipMemcpy(&count, s.head_segments, 4, hipMemcpyDeviceToHost));
  Require(count == 3, "selected head did not deduplicate Hadamard segments");
}

// Small deterministic resident EXL3 model: real 256-wide attention and
// 128-wide indexer arithmetic, without a downloaded checkpoint or test skip.
struct MtpExecutorFixture {
  Buffers buffers;
  std::vector<std::unique_ptr<Bank>> banks;
  q::Config config;
  std::unique_ptr<gpu::DeviceModel> model;
  std::unique_ptr<gpu::Executor> exec;
  unsigned seed{9001};

  gpu::DeviceTensor Plain(unsigned rows, unsigned cols, bool f32 = false,
                          bool norm = false) {
    std::mt19937 rng(seed++);
    std::vector<float> values(std::size_t{rows} * cols);
    for (auto& value : values)
      value = (norm ? 1.0F : 0.0F) + (static_cast<int>(rng() % 2000) - 1000) *
                                         (norm ? 0.0001F : 0.00003F);
    gpu::DeviceTensor t;
    t.rows = rows;
    t.cols = cols;
    t.type = f32 ? gufo::core::GgmlType::kF32 : gufo::core::GgmlType::kF16;
    if (f32) {
      t.data = buffers.Copy(values);
    } else {
      std::vector<std::uint16_t> halves(values.size());
      for (std::size_t i = 0; i < values.size(); ++i)
        halves[i] = ex::HalfFromDouble(values[i]);
      t.data = buffers.Copy(halves);
    }
    return t;
  }
  gpu::DeviceTensor Compressed(unsigned k, unsigned n, int bits = 7,
                               unsigned experts = 1) {
    banks.push_back(
        std::make_unique<Bank>(buffers, k, n, bits, experts, seed++));
    return banks.back()->weight;
  }
  gpu::DeviceMixer Mixer(bool inject = true) {
    return {Plain(config.HcDim(), 1, true, true),
            Plain(config.hc_low_rank, config.HcDim()),
            Plain(config.HcDim(), config.hc_low_rank),
            inject ? Plain(config.hc_count, config.HcDim(), true)
                   : gpu::DeviceTensor{}};
  }
  explicit MtpExecutorFixture(unsigned max_batch = 16,
                              unsigned expert_ff = 128) {
    config.num_layers = 1;
    config.num_layers_all = 2;
    config.hidden_size = 128;
    config.vocab_size = 256;
    config.context_length = 64;
    config.hc_count = 4;
    config.hc_low_rank = 32;
    config.full_attention_interval = 1;
    config.num_heads = 4;
    config.num_kv_heads = 1;
    config.head_dim = 256;
    config.rotary_dim = 64;
    config.rope_theta = 10000;
    config.rope_scaling = {.factor = 2.0F, .original_context = 64};
    config.indexer_heads = 4;
    config.indexer_head_dim = 128;
    config.indexer_top_k = 8;
    config.compress_ratio = 4;
    config.ssm_conv_kernel = 4;
    config.ssm_head_dim = 128;
    config.ssm_num_k_heads = config.ssm_num_v_heads = 1;
    config.ssm_inner_size = 128;
    config.num_experts = 4;
    config.num_experts_used = 2;
    config.expert_ff = config.shared_expert_ff = expert_ff;
    config.nextn_layers = 1;
    gpu::DeviceLayer l;
    l.hc_attn = Mixer();
    l.hc_ffn = Mixer();
    l.nextn_head = Mixer(false);
    l.nextn_enorm = Plain(config.hidden_size, 1, true, true);
    l.nextn_hnorm = Plain(config.HcDim(), 1, true, true);
    l.nextn_fc_embedding = Compressed(config.hidden_size, config.hidden_size);
    l.nextn_fc_hidden = Compressed(config.hidden_size, config.hidden_size);
    l.attn_q = Compressed(config.hidden_size, 2 * config.AttentionQDim());
    l.attn_k = Compressed(config.hidden_size, config.AttentionKvDim());
    l.attn_v = Compressed(config.hidden_size, config.AttentionKvDim());
    l.attn_out = Compressed(config.AttentionQDim(), config.hidden_size);
    l.attn_q_norm = Plain(config.head_dim, 1, true, true);
    l.attn_k_norm = Plain(config.head_dim, 1, true, true);
    l.indexer_k = Compressed(config.hidden_size, config.indexer_head_dim);
    l.indexer_q = Compressed(config.hidden_size,
                             config.indexer_heads * config.indexer_head_dim);
    l.indexer_k_norm = Plain(config.indexer_head_dim, 1, true, true);
    l.indexer_q_norm = Plain(config.indexer_head_dim, 1, true, true);
    l.router = Plain(config.num_experts + 1, config.hidden_size);
    l.shexp_gate = Compressed(config.hidden_size, config.shared_expert_ff, 6);
    l.shexp_up = Compressed(config.hidden_size, config.shared_expert_ff, 6);
    l.shexp_down = Compressed(config.shared_expert_ff, config.hidden_size, 6);
    l.ffn_gate_exps =
        Compressed(config.hidden_size, config.expert_ff, 5, config.num_experts);
    l.ffn_up_exps =
        Compressed(config.hidden_size, config.expert_ff, 5, config.num_experts);
    l.ffn_down_exps =
        Compressed(config.expert_ff, config.hidden_size, 5, config.num_experts);
    const auto embedding = Plain(config.vocab_size, config.hidden_size);
    const auto output = Compressed(config.hidden_size, config.vocab_size, 6);
    model = gpu::ExecutorTestAccess::Model(config, l, embedding, output,
                                           Mixer(false));
    std::string error;
    exec = gpu::Executor::Create(*model, nullptr,
                                 {.max_batch = max_batch,
                                  .max_logit_rows = 8,
                                  .max_speculative = 8,
                                  .max_context = 128},
                                 &error);
    Require(exec != nullptr, error);
    Require(q::MakeYarnRope(config.rope_scaling, config.rotary_dim,
                            config.rope_theta)
                    .enabled != 0,
            "executor fixture must exercise enabled YaRN");
  }
};

template<class T>
void SameBits(const std::vector<T>& a, const std::vector<T>& b,
              const std::string& name) {
  Require(a.size() == b.size() &&
              (a.empty() ||
               std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0),
          name + " differs bitwise");
}

void MtpExecutorCatchUp(bool batched) {
  BeginCase(batched ? "executor EXL3 ragged batch cache-only YaRN"
                    : "executor EXL3 scalar cache-only YaRN");
  MtpExecutorFixture f;
  auto& exec = *f.exec;
  using Access = gpu::ExecutorTestAccess;
  const unsigned sessions = batched ? 3 : 1;
  std::vector<std::unique_ptr<gpu::Session>> reference(sessions),
      cached(sessions);
  const std::array<unsigned, 3> prefix_rows{6, 7, 5}, hidden_rows{2, 0, 4};
  const std::array<std::vector<std::int32_t>, 3> tokens{
      {{11, 12, 13}, {21, 22}, {31}}};
  std::string error;
  const auto state_equal = [&](unsigned i, const std::string& stage) {
    const auto a = Access::ReadState(*reference[i], f.config);
    const auto b = Access::ReadState(*cached[i], f.config);
    Require(a.position == b.position && a.blocks == b.blocks,
            stage + " position/block shadows differ");
    SameBits(a.k, b.k, stage + " K cache");
    SameBits(a.v, b.v, stage + " V cache");
    SameBits(a.raw, b.raw,
             stage + " raw indexer ring including untouched rows");
    SameBits(a.pooled, b.pooled, stage + " pooled indexer keys");
  };
  std::vector<gpu::Executor::MtpBatchItem> items;
  unsigned expected_rows = 0;
  for (unsigned i = 0; i < sessions; ++i) {
    reference[i] =
        exec.CreateSession(gufo::core::SessionMode::kSpeculative, 128, &error);
    cached[i] =
        exec.CreateSession(gufo::core::SessionMode::kSpeculative, 128, &error);
    Require(reference[i] && cached[i], error);
    auto hidden = Input(8 * f.config.HcDim());
    for (std::size_t j = 0; j < hidden.size(); ++j)
      hidden[j] += 0.001F * static_cast<float>((j + 7 * i) % 17);
    for (auto* session : {reference[i].get(), cached[i].get()}) {
      Access::Hidden(*session, hidden);
      std::vector<std::int32_t> prefix(prefix_rows[i]);
      for (unsigned r = 0; r < prefix.size(); ++r)
        prefix[r] = 40 + i * 10 + r;
      Require(exec.MtpForward(*session, prefix, 0, {}, &error), error);
    }
    Require(
        exec.MtpForward(*reference[i], tokens[i], hidden_rows[i], {}, &error),
        error);
    items.push_back({cached[i].get(), tokens[i],
                     static_cast<std::int32_t>(hidden_rows[i]), true});
    expected_rows += tokens[i].size();
  }
  Require(exec.BeginMtpCycleDiagnostics(&error), error);
  if (batched)
    Require(exec.MtpForwardBatch(items, &error), error);
  else
    Require(exec.MtpForward(*cached[0], tokens[0], hidden_rows[0],
                            {.kv_only = true}, &error),
            error);
  gpu::Executor::MtpCycleDiagnostics diagnostic;
  Require(exec.EndMtpCycleDiagnostics(0, 0, &diagnostic, &error), error);
  Require(diagnostic.cache_only_rows == expected_rows,
          "EXL3 cache-only dispatch did not execute every requested row");
  Require(diagnostic.expert_tile_reads == 0,
          "cache-only catch-up unexpectedly executed experts");
  for (unsigned i = 0; i < sessions; ++i) {
    state_equal(i, "ragged catch-up session=" + std::to_string(i));
    Require(!Access::ReadState(*cached[i], f.config).residual_valid,
            "cache-only catch-up exposed a carried residual");
  }
  // The first two ragged sessions cross the sparse threshold and finish
  // pooling blocks. Prove the fixture distinguishes the formerly buggy
  // unscaled pooling kernel, rather than merely comparing two equal zeros.
  Require(Access::ReadState(*cached[0], f.config).blocks >= 2,
          "fixture did not pool blocks across sparse threshold");
  Require(Access::YarnWitness(exec, *reference[0]),
          "fixture cannot distinguish scaled and unscaled pooled keys");
  // Eager -> capture -> replay of the same scalar cache-only shape, at new
  // positions with changed tokens. The ragged batch above remains a real
  // MtpForwardBatch call; none of these checks substitutes projection helpers.
  std::size_t captured_graphs = 0;
  for (unsigned pass = 0; pass < 3; ++pass) {
    const std::array<std::int32_t, 3> more{
        static_cast<std::int32_t>(70 + pass),
        static_cast<std::int32_t>(80 + pass),
        static_cast<std::int32_t>(90 + pass)};
    Require(exec.MtpForward(*reference[0], more, 2, {}, &error), error);
    Require(exec.MtpForward(*cached[0], more, 2, {.kv_only = true}, &error),
            error);
    state_equal(0, "graph catch-up pass=" + std::to_string(pass));
    if (pass == 1) {
      captured_graphs = Access::Graphs(*cached[0]);
      Require(captured_graphs != 0, "cache-only graph was not captured");
    }
    if (pass == 2)
      Require(Access::Graphs(*cached[0]) == captured_graphs,
              "cache-only replay created another graph instead of reusing it");
  }
  for (unsigned i = 0; i < sessions; ++i) {
    const std::int32_t next = 101 + i;
    for (auto* session : {reference[i].get(), cached[i].get()}) {
      if (i == 0)
        Access::PoisonSelection(exec);
      Require(exec.MtpForward(*session, std::span(&next, 1), 1, {}, &error),
              error);
      if (i == 0)
        Require(Access::HasExpectedSparseSelection(exec, *session),
                "sparse selector must replace poison with exactly the nonempty "
                "causal budget");
    }
    state_equal(i, "known-hidden continuation");
    const auto a = Access::FullLogits(exec, *reference[i]);
    const auto b = Access::FullLogits(exec, *cached[i]);
    Require(std::ranges::all_of(a, [](float v) { return std::isfinite(v); }),
            "continuation logits are nonfinite");
    SameBits(a, b, "subsequent full-vocabulary logits");
    for (auto* session : {reference[i].get(), cached[i].get()})
      Require(exec.MtpForward(*session, std::span(&next, 1), -1, {}, &error),
              error);
    state_equal(i, "recursive continuation");
    SameBits(Access::FullLogits(exec, *reference[i]),
             Access::FullLogits(exec, *cached[i]),
             "recursive full-vocabulary logits");
  }
}

void PrefillScratchEstimate() {
  BeginCase("host EXL3 prefill scratch estimate");
  q::Config c;
  c.num_experts = 512;
  c.num_experts_used = 10;
  c.hidden_size = 2560;
  c.expert_ff = 640;
  std::string error;
  const auto chunk =
      gpu::Executor::EstimateExl3PrefillScratchBytes(c, 4096, &error);
  const auto through =
      gpu::Executor::EstimateExl3PrefillScratchBytes(c, 4224, &error);
  Require(chunk == 839204360, "4096-row disjoint scratch byte count");
  Require(through == 865429240, "4224-row tail-capacity scratch byte count");
  std::cout << "MEMORY EXL3 prefill P4096=" << chunk << " bytes (800.328 MiB)"
            << " P4224=" << through << " bytes (825.338 MiB)\n";
  Require(gpu::Executor::EstimateExl3PrefillScratchBytes(c, 0, &error) == 0 &&
              !error.empty(),
          "zero-row scratch must be rejected");
  Require(
      gpu::Executor::EstimateExl3PrefillScratchBytes(c, 1000000, &error) == 0,
      "device bucket grid limit must be enforced");
  Require(gpu::Executor::EstimateExl3PrefillScratchBytes(c, 0xFFFFFFFFU,
                                                         &error) == 0,
          "assignment counter overflow must be rejected");
  c.num_experts_used = c.num_experts + 1;
  Require(gpu::Executor::EstimateExl3PrefillScratchBytes(c, 1, &error) == 0,
          "invalid used-expert count must be rejected");
}

void ExecutorPrefillExperts() {
  BeginCase("executor X3 prefill dispatch and chunk bits, F=640");
  constexpr unsigned rows = 65;
  MtpExecutorFixture f(rows, 640);
  auto& exec = *f.exec;
  const auto& c = f.config;
  using Access = gpu::ExecutorTestAccess;
  const auto& pp = Access::PrefillScratch(exec);
  const auto assignments = rows * c.num_experts_used;
  Require(pp.max_assignments == assignments &&
              pp.max_buckets ==
                  std::min(assignments, c.num_experts + assignments / 64),
          "executor bucket capacities differ from production 64-row plan");
  Require(exec.exl3_prefill_scratch_bytes() ==
              gpu::Executor::EstimateExl3PrefillScratchBytes(c, rows),
          "allocated prefill bytes differ from host estimate");
  // Check all persistent slabs, including the caller's assignment output.
  // An alias can appear to work for a single expert, then fail under scatter.
  const std::vector<std::pair<const void*, std::size_t>> slabs{
      {pp.offsets, (c.num_experts + 1) * sizeof(std::uint32_t)},
      {pp.slots, assignments * sizeof(std::int32_t)},
      {pp.buckets, pp.max_buckets * sizeof(gpu::Exl3ExpertBucket)},
      {pp.bucket_count, sizeof(std::uint32_t)},
      {pp.input, assignments * std::max(c.hidden_size, c.expert_ff) *
                     sizeof(std::uint16_t)},
      {pp.gate, assignments * c.expert_ff * sizeof(float)},
      {pp.up, assignments * c.expert_ff * sizeof(float)},
      {pp.out, assignments * c.hidden_size * sizeof(float)},
      {pp.down_exponents, assignments * sizeof(std::int32_t)},
      {Access::RoutedOutput(exec),
       assignments * c.hidden_size * sizeof(float)}};
  for (std::size_t i = 0; i < slabs.size(); ++i) {
    Require(slabs[i].first != nullptr, "executor omitted a prefill slab");
    const auto a = reinterpret_cast<std::uintptr_t>(slabs[i].first);
    for (std::size_t j = 0; j < i; ++j) {
      const auto b = reinterpret_cast<std::uintptr_t>(slabs[j].first);
      Require(a + slabs[i].second <= b || b + slabs[j].second <= a,
              "executor prefill scratch slabs overlap");
    }
  }
  std::vector<std::int32_t> tokens(rows);
  for (unsigned r = 0; r < rows; ++r)
    tokens[r] = 1 + (r * 17) % (c.vocab_size - 1);
  std::string error;
  const auto prefill = [&](unsigned chunk) {
    auto session = exec.CreateSession(gufo::core::SessionMode::kAutoregressive,
                                      128, &error);
    Require(session != nullptr, error);
    std::vector<float> logits(c.vocab_size);
    std::vector<float> hidden(std::size_t{rows} * c.HcDim());
    Require(exec.BeginMtpCycleDiagnostics(&error), error);
    for (unsigned first = 0; first < rows; first += chunk) {
      const auto count = std::min(chunk, rows - first);
      Require(
          exec.Forward(*session, std::span(tokens).subspan(first, count), 0,
                       nullptr, gpu::Executor::ForwardMode::kPrefill, &error),
          error);
      // n_logits=0 disables final-query attention pruning; fewer than 128
      // rows disables the FFN skip. Compare every fully computed residual,
      // then replay the one-row head without changing those residuals.
      const auto residual = Access::PrefillResiduals(exec, count);
      std::copy(residual.begin(), residual.end(),
                hidden.begin() + std::size_t{first} * c.HcDim());
      Require(exec.ReplayOutputHead(count, logits, &error), error);
    }
    gpu::Executor::MtpCycleDiagnostics d;
    Require(exec.EndMtpCycleDiagnostics(0, 0, &d, &error), error);
    Require(d.prefill_expert_rows == rows * c.num_layers &&
                d.prefill_expert_buckets != 0 && d.scaled_down_rows == 0,
            "kPrefill did not dispatch bucketed experts for every row, "
            "including tails");
    Require(std::ranges::all_of(
                logits, [](float value) { return std::isfinite(value); }),
            "prefill logits are not finite");
    return std::pair{logits, hidden};
  };
  const auto bulk = prefill(rows);
  for (unsigned chunk : {1U, 3U, 8U, 9U, 32U, 33U, 64U}) {
    const auto split = prefill(chunk);
    SameBits(bulk.first, split.first,
             "executor prefill chunks=" + std::to_string(chunk));
    SameBits(bulk.second, split.second,
             "every prefill residual row chunks=" + std::to_string(chunk));
  }

  // Exercise the allocated scaled-down path through a real decode forward,
  // including the executor's 64+1 partition and expert_ff > 2 * hidden.
  const auto& routed = Access::RoutedScratch(exec);
  Require(routed.down_exponents != nullptr,
          "executor omitted decode down exponents");
  Check(hipMemset(routed.down_exponents, 0x7F,
                  64 * c.num_experts_used * sizeof(std::int32_t)));
  auto decode =
      exec.CreateSession(gufo::core::SessionMode::kAutoregressive, 128, &error);
  Require(decode != nullptr, error);
  std::vector<float> logits(c.vocab_size);
  Require(exec.BeginMtpCycleDiagnostics(&error), error);
  Require(exec.Forward(*decode, tokens, 1, logits.data(),
                       gpu::Executor::ForwardMode::kDecode, &error),
          error);
  gpu::Executor::MtpCycleDiagnostics d;
  Require(exec.EndMtpCycleDiagnostics(0, 0, &d, &error), error);
  Require(d.scaled_down_rows == assignments && d.prefill_expert_rows == 0,
          "decode did not dispatch scaled routed down across 64+1 rows");
  const auto exponents = Access::Read<std::int32_t>(routed.down_exponents,
                                                    64 * c.num_experts_used);
  Require(
      std::ranges::all_of(exponents, [](auto e) { return e >= -1 && e <= 10; }),
      "scaled-down dispatch did not overwrite poisoned exponents");

  // Isolate the actual production X3 operator from the other transformer
  // stages, using the very same persistent executor allocations and tile.
  auto input = Input(rows * c.hidden_size);
  auto* x = f.buffers.Copy(input);
  auto* y = f.buffers.Make<float>(std::size_t{assignments} * c.hidden_size + 8);
  for (bool shared : {false, true}) {
    std::vector<std::int32_t> ids(assignments);
    for (unsigned r = 0; r < rows; ++r)
      for (unsigned j = 0; j < c.num_experts_used; ++j)
        ids[r * c.num_experts_used + j] = shared ? j : (r + j) % c.num_experts;
    auto* id = f.buffers.Copy(ids);
    const auto project = [&](unsigned chunk) {
      // NaN payload poison catches omitted writes; the separate finite
      // bit-pattern guard detects writes past the assignment allocation.
      Check(hipMemset(
          y, 0xFF, std::size_t{assignments} * c.hidden_size * sizeof(float)));
      Check(hipMemset(y + std::size_t{assignments} * c.hidden_size, 0x7F,
                      8 * sizeof(float)));
      for (unsigned first = 0; first < rows; first += chunk)
        gpu::Exl3PrefillExperts(
            f.model->layers()[0], x + first * c.hidden_size,
            id + first * c.num_experts_used,
            y + std::size_t{first} * c.num_experts_used * c.hidden_size,
            std::min(chunk, rows - first), c.num_experts_used, 64, pp,
            exec.stream());
      Check(hipStreamSynchronize(exec.stream()));
      const auto output =
          Access::Read<float>(y, std::size_t{assignments} * c.hidden_size);
      Require(std::ranges::all_of(
                  output, [](float value) { return std::isfinite(value); }),
              "X3 assignment output is not finite");
      Require(
          std::ranges::any_of(output, [](float value) { return value != 0; }),
          "X3 assignment fixture must produce nonzero output");
      Require(Access::Read<std::uint32_t>(
                  y + std::size_t{assignments} * c.hidden_size, 1)[0] ==
                  0x7F7F7F7FU,
              "X3 assignment output overwrote its guard");
      return output;
    };
    const auto reference = project(rows);
    if (shared)
      Require(
          Access::Read<std::uint32_t>(pp.bucket_count, 1)[0] ==
              2 * c.num_experts_used,
          "shared history must create 64+1 buckets for each selected expert");
    for (unsigned chunk : {1U, 3U, 8U, 9U, 32U, 33U, 64U})
      SameBits(reference, project(chunk),
               "X3 expert operator chunks=" + std::to_string(chunk));
  }
}

void MtpKvGroup() {
  namespace fast = gufo::hip;
  constexpr unsigned k = 256, q_width = 1024, kv_width = 512;
  Buffers buffers;
  Bank q(buffers, k, q_width, 7, 1, 17);
  Bank key(buffers, k, kv_width, 7, 1, 71);
  Bank value(buffers, k, kv_width, 7, 1, 137);
  auto scratch = Scratch(buffers, 64, k, q_width);
  const auto& p = scratch.projection;
  const fast::Exl3FastWorkspace ws{
      p.fast_input,         p.fast_input_elements, p.fast_counters,
      p.fast_counter_count, p.fast_down_input,     p.fast_down_input_elements};
  for (const unsigned rows : {1U, 2U, 3U, 7U, 8U, 9U, 32U, 33U, 64U}) {
    BeginCase("MTP K7 QKV vs KV group rows=" + std::to_string(rows));
    const auto input = Input(rows * k);
    auto* x = buffers.Copy(input);  // ends at the last row's allocation
    auto* query = buffers.Make<float>(rows * q_width);
    auto* k_full = buffers.Make<float>(rows * kv_width);
    auto* v_full = buffers.Make<float>(rows * kv_width);
    auto* k_only = buffers.Make<float>(rows * kv_width + 8);
    auto* v_only = buffers.Make<float>(rows * kv_width + 8);
    const std::array<fast::Exl3FastTarget, 3> full{
        {{q.weight.exl3, query, q_width},
         {key.weight.exl3, k_full, kv_width},
         {value.weight.exl3, v_full, kv_width}}};
    const std::array<fast::Exl3FastTarget, 2> kv{
        {{key.weight.exl3, k_only, kv_width},
         {value.weight.exl3, v_only, kv_width}}};
    Require(fast::Exl3FastGroupSupported(full, ws) &&
                fast::Exl3FastGroupSupported(kv, ws),
            "K7 full and K/V groups supported");
    Check(hipMemset(k_only, 0x7F, (rows * kv_width + 8) * sizeof(float)));
    Check(hipMemset(v_only, 0x7F, (rows * kv_width + 8) * sizeof(float)));
    fast::LaunchExl3FastProjections(full, x, k, rows, ws, nullptr);
    fast::LaunchExl3FastProjections(kv, x, k, rows, ws, nullptr);
    for (const auto& [bank, a, b] : {std::tuple{&key, k_full, k_only},
                                     std::tuple{&value, v_full, v_only}}) {
      std::vector<float> expected(rows * kv_width), actual(expected.size());
      Check(hipMemcpy(expected.data(), a, expected.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
      Check(hipMemcpy(actual.data(), b, actual.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
      Require(std::memcmp(expected.data(), actual.data(),
                          actual.size() * sizeof(float)) == 0,
              "removing Q preserves every K/V projection bit");
      for (const unsigned row : {0U, rows - 1})
        Near(std::span(actual).subspan(row * kv_width, kv_width),
             bank->Reference(0, input.data() + row * k, true), 1e-5,
             "K/V group vs independent FP64 projection");
      std::uint32_t guard;
      Check(hipMemcpy(&guard, b + rows * kv_width, sizeof(guard),
                      hipMemcpyDeviceToHost));
      Require(guard == 0x7F7F7F7FU, "K/V group preserves output guard");
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << std::unitbuf;
  try {
    if (argc == 2 && std::string(argv[1]) == "--prefill-scratch-only") {
      PrefillScratchEstimate();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--prefill-experts-only") {
      ExecutorPrefillExperts();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--mtp-kv-executor") {
      MtpExecutorCatchUp(false);
      MtpExecutorCatchUp(true);
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--mtp-kv-group") {
      MtpKvGroup();
      return 0;
    }
    if (argc == 2) {
      Require(std::string_view(argv[1]) == "--shared-invariance-only",
              "unknown operator test argument");
      SharedMlpInvariance();
      CompressedPrefillChunks();
      return 0;
    }
    Require(argc == 1, "unexpected operator test arguments");
    SharedMlpInvariance();
    CompressedPrefillChunks();
    InputRoundingControl();
    DecodeK7();
    for (int bits : {5, 6, 7})
      Projection(256, 640, bits);
    Projection(640, 2560, 7);
    Projection(2560, 6144, 7, {1U, 3U});
    Projection(2560, 10240, 7, {1U});
    Projection(6144, 2560, 7, {1U});
    Routed(true, 3, 2560, 640, 16);
    SelectedHead();
    for (unsigned rows : {1U, 2U, 3U, 7U, 8U, 9U, 32U, 33U, 64U}) {
      Routed(true, rows);
      Routed(false, rows);
    }
    std::cout << "Flash-Next EXL3 GPU contracts passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
