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
#include <type_traits>
#include <vector>

#include "src/core/hip/detail/exl3_gemv.hpp"
#include "src/models/qwen/hip/kernels/exl3_gemv.hpp"
#include "src/models/qwen/hip/ops/plain_prefill_stable.hpp"
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
  static std::unique_ptr<DeviceModel> Model(
      const Config& config, const DeviceLayer& layer, DeviceTensor embedding,
      DeviceTensor output, DeviceMixer head,
      const DeviceLayer* linear = nullptr) {
    std::unique_ptr<DeviceModel> model(new DeviceModel);
    model->config_ = config;
    model->token_embd_ = embedding;
    model->output_ = output;
    model->hc_head_ = head;
    if (linear != nullptr)
      model->layers_.push_back(*linear);
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
  static Exl3PrefillScratch PrefillScratch(const Executor& exec) {
    return exec.Exl3PrefillView();
  }
  static bool PrefillBorrowsStage(const Executor& exec, unsigned offset) {
    const auto base = exec.s_;
    exec.UseScratch(exec.RowScratch(base, offset));
    const auto view = exec.Exl3PrefillView();
    exec.UseScratch(base);
    const auto elements = std::size_t{offset} * exec.config().num_experts_used *
                          exec.config().expert_ff;
    return exec.exl3_prefill_.gate == nullptr &&
           exec.exl3_prefill_.up == nullptr &&
           view.gate == base.gate_e + elements &&
           view.up == base.up_e + elements &&
           view.out == exec.exl3_prefill_.out;
  }
  static const Exl3Scratch& RoutedScratch(const Executor& exec) {
    return exec.exl3_;
  }
  static float* RoutedOutput(const Executor& exec) { return exec.s_.down_e; }
  static void HalfWeightCols(DeviceModel& model, unsigned cols) {
    model.max_half_cols_ = cols;
  }
  static bool DensePrefill(const Executor& exec,
                           std::initializer_list<const DeviceTensor*> weights,
                           std::initializer_list<float*> outputs,
                           const float* x, unsigned rows, std::string* error) {
    return exec.Exl3DensePrefill(weights, outputs, x, rows, error);
  }
  static bool PlainPrefill(const Executor& exec, const DeviceTensor& weight,
                           const float* x, float* out, unsigned rows,
                           std::string* error) {
    return exec.PlainDensePrefill(weight, x, out, rows, error);
  }
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
  static bool PrefillLinearOnly(Executor& exec, Session& session,
                                std::span<const std::int32_t> tokens,
                                std::string* error) {
    // CreateSession saw the valid two-layer geometry and allocated its KV
    // plane. Stop this test's public prefill after layer0 to observe GDN
    // before attention; all production dispatch/phase handling stays active.
    auto& model = const_cast<DeviceModel&>(*exec.model_);
    struct Restore {
      std::uint32_t& layers;
      std::uint32_t value;
      ~Restore() { layers = value; }
    } restore{model.config_.num_layers, model.config_.num_layers};
    model.config_.num_layers = 1;
    return exec.Forward(session, tokens, 0, nullptr,
                        Executor::ForwardMode::kPrefill, error);
  }
  static auto GdnState(const Session& session, const Config& c) {
    const auto& state = session.linear_[0];
    return std::pair{
        Read<float>(state.conv_state,
                    std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels()),
        Read<float>(state.state, std::size_t{c.ssm_num_v_heads} *
                                     c.ssm_head_dim * c.ssm_head_dim)};
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
  explicit MtpExecutorFixture(unsigned max_batch = 16, unsigned expert_ff = 128,
                              int shared_bits = 6, bool linear_prefix = false,
                              unsigned half_weight_cols = 0) {
    config.num_layers = linear_prefix ? 2 : 1;
    config.num_layers_all = config.num_layers + 1;
    config.hidden_size = 128;
    config.vocab_size = 256;
    config.context_length = 64;
    config.hc_count = 4;
    config.hc_low_rank = 32;
    config.full_attention_interval = linear_prefix ? 2 : 1;
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
    l.shexp_gate =
        Compressed(config.hidden_size, config.shared_expert_ff, shared_bits);
    l.shexp_up =
        Compressed(config.hidden_size, config.shared_expert_ff, shared_bits);
    l.shexp_down =
        Compressed(config.shared_expert_ff, config.hidden_size, shared_bits);
    l.ffn_gate_exps =
        Compressed(config.hidden_size, config.expert_ff, 5, config.num_experts);
    l.ffn_up_exps =
        Compressed(config.hidden_size, config.expert_ff, 5, config.num_experts);
    l.ffn_down_exps =
        Compressed(config.expert_ff, config.hidden_size, 5, config.num_experts);
    const auto embedding = Plain(config.vocab_size, config.hidden_size);
    const auto output = Compressed(config.hidden_size, config.vocab_size, 6);
    auto linear = l;
    if (linear_prefix) {
      linear.linear = true;
      linear.ssm_qkv = Compressed(config.hidden_size, config.SsmConvChannels());
      linear.ssm_gate = Compressed(config.hidden_size, config.SsmValueDim());
      linear.ssm_out = Compressed(config.SsmValueDim(), config.hidden_size);
      linear.ssm_alpha_beta =
          Plain(2 * config.ssm_num_v_heads, config.hidden_size, true);
      linear.ssm_conv1d =
          Plain(config.SsmConvChannels(), config.ssm_conv_kernel, true);
      linear.ssm_a = Plain(config.ssm_num_v_heads, 1, true);
      const std::vector<float> decay(config.ssm_num_v_heads, -0.5F);
      Check(hipMemcpy(linear.ssm_a.data, decay.data(),
                      decay.size() * sizeof(float), hipMemcpyHostToDevice));
      linear.ssm_dt = Plain(config.ssm_num_v_heads, 1, true);
      linear.ssm_norm = Plain(config.ssm_head_dim, 1, true, true);
    }
    model = gpu::ExecutorTestAccess::Model(config, l, embedding, output,
                                           Mixer(false),
                                           linear_prefix ? &linear : nullptr);
    if (half_weight_cols != 0)
      gpu::ExecutorTestAccess::HalfWeightCols(*model, half_weight_cols);
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
bool CompareBits(const std::vector<T>& a, const std::vector<T>& b,
                 const std::string& name, std::size_t columns = 0) {
  if (a.size() != b.size()) {
    std::cerr << "FAIL " << name << " sizes=" << a.size() << '/' << b.size()
              << '\n';
    return false;
  }
  using Bits = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint16_t>;
  static_assert(sizeof(T) == sizeof(Bits));
  constexpr std::uint64_t sign = std::uint64_t{1} << (sizeof(T) * 8 - 1);
  const auto ordered = [&](Bits v) {
    return (v & sign) ? ((2 * sign - 1) ^ v) : (std::uint64_t{v} | sign);
  };
  const auto value = [](T v) {
    if constexpr (std::is_same_v<T, float>)
      return double(v);
    else if constexpr (sizeof(T) == 2)
      return ex::HalfToDouble(std::bit_cast<std::uint16_t>(v));
    else
      return double(v);
  };
  for (std::size_t i = 0; i < a.size(); ++i) {
    const auto av = std::bit_cast<Bits>(a[i]), bv = std::bit_cast<Bits>(b[i]);
    if (av == bv)
      continue;
    const auto ao = ordered(av), bo = ordered(bv);
    std::cerr << "FAIL " << name << " first index=" << i;
    if (columns != 0)
      std::cerr << " row=" << i / columns << " col=" << i % columns;
    std::cerr << " expected=" << std::setprecision(10) << value(a[i])
              << " actual=" << value(b[i]) << " bits=0x" << std::hex << av
              << "/0x" << bv << std::dec
              << " ulps=" << (ao > bo ? ao - bo : bo - ao) << '\n';
    return false;
  }
  std::cout << "PASS " << name << " matches bitwise\n";
  return true;
}

template<class T>
void SameBits(const std::vector<T>& a, const std::vector<T>& b,
              const std::string& name, std::size_t columns = 0) {
  if (!CompareBits(a, b, name, columns))
    throw std::runtime_error(current_case + " : " + name + " differs bitwise");
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
    bool equal = a.position == b.position && a.blocks == b.blocks;
    std::cout << (equal ? "PASS " : "FAIL ") << stage
              << " position/block shadows\n";
    equal =
        CompareBits(a.k, b.k, stage + " K cache", f.config.AttentionKvDim()) &&
        equal;
    equal =
        CompareBits(a.v, b.v, stage + " V cache", f.config.AttentionKvDim()) &&
        equal;
    equal = CompareBits(a.raw, b.raw,
                        stage + " raw indexer ring including untouched rows",
                        f.config.indexer_head_dim) &&
            equal;
    equal = CompareBits(a.pooled, b.pooled, stage + " pooled indexer keys",
                        f.config.indexer_head_dim) &&
            equal;
    return equal;
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
  bool equal = true;
  for (unsigned i = 0; i < sessions; ++i) {
    equal =
        state_equal(i, "ragged catch-up session=" + std::to_string(i)) && equal;
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
    equal =
        state_equal(0, "graph catch-up pass=" + std::to_string(pass)) && equal;
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
    equal = state_equal(i, "known-hidden continuation") && equal;
    const auto a = Access::FullLogits(exec, *reference[i]);
    const auto b = Access::FullLogits(exec, *cached[i]);
    Require(std::ranges::all_of(a, [](float v) { return std::isfinite(v); }),
            "continuation logits are nonfinite");
    equal = CompareBits(a, b, "subsequent full-vocabulary logits",
                        f.config.vocab_size) &&
            equal;
    for (auto* session : {reference[i].get(), cached[i].get()})
      Require(exec.MtpForward(*session, std::span(&next, 1), -1, {}, &error),
              error);
    equal = state_equal(i, "recursive continuation") && equal;
    equal =
        CompareBits(Access::FullLogits(exec, *reference[i]),
                    Access::FullLogits(exec, *cached[i]),
                    "recursive full-vocabulary logits", f.config.vocab_size) &&
        equal;
  }
  Require(equal, "all MTP states and continuation logits match bitwise");
}

void DensePrefillProjection() {
  BeginCase("executor grouped K7 dense X3, slices and chunk bits");
  constexpr unsigned rows = 65, k = 256;
  MtpExecutorFixture f(rows, 640);
  auto& exec = *f.exec;
  using Access = gpu::ExecutorTestAccess;
  Bank q(f.buffers, k, 128, 7, 1, 17);
  Bank key(f.buffers, k, 640, 7, 1, 23);
  Bank value(f.buffers, k, 640, 7, 1, 71);
  const std::array<Bank*, 3> banks{&q, &key, &value};
  const auto input = Input(rows * k);
  auto* x = f.buffers.Copy(input);
  std::array<float*, 3> output{}, second{};
  for (unsigned i = 0; i < banks.size(); ++i) {
    output[i] = f.buffers.Make<float>(rows * banks[i]->n + 8);
    second[i] = f.buffers.Make<float>(rows * banks[i]->n + 8);
  }
  std::string error;
  const auto queue = [&](unsigned first, unsigned count, const auto& dst) {
    Require(Access::DensePrefill(exec, {&q.weight, &key.weight, &value.weight},
                                 {dst[0] + first * q.n, dst[1] + first * key.n,
                                  dst[2] + first * value.n},
                                 x + first * k, count, &error),
            error);
  };
  const auto download = [&](const auto& dst) {
    std::array<std::vector<float>, 3> result;
    Check(hipStreamSynchronize(exec.stream()));
    for (unsigned i = 0; i < banks.size(); ++i) {
      result[i] = Access::Read<float>(dst[i], rows * banks[i]->n);
      Require(std::ranges::all_of(result[i],
                                  [](float v) { return std::isfinite(v); }),
              "dense X3 left unwritten or nonfinite outputs");
      Require(Access::Read<std::uint32_t>(dst[i] + rows * banks[i]->n, 1)[0] ==
                  0x7F7F7F7FU,
              "dense X3 overwrote its output guard");
    }
    return result;
  };
  const auto poison = [&](const auto& dst) {
    for (unsigned i = 0; i < banks.size(); ++i) {
      Check(hipMemset(dst[i], 0xFF, rows * banks[i]->n * sizeof(float)));
      Check(hipMemset(dst[i] + rows * banks[i]->n, 0x7F, 8 * sizeof(float)));
    }
  };
  const auto project = [&](unsigned chunk) {
    poison(output);
    Require(exec.BeginMtpCycleDiagnostics(&error), error);
    for (unsigned first = 0; first < rows; first += chunk)
      queue(first, std::min(chunk, rows - first), output);
    gpu::Executor::MtpCycleDiagnostics d;
    Require(exec.EndMtpCycleDiagnostics(0, 0, &d, &error), error);
    const auto groups = (rows + chunk - 1) / chunk;
    Require(d.prefill_dense_rows == 3 * rows &&
                d.prefill_dense_rotation_groups == groups &&
                d.prefill_dense_gemm_groups == 2 * groups,
            "shared-input rotations or equal-N GEMMs were not grouped");
    return download(output);
  };
  const auto bulk = project(rows);
  for (unsigned chunk : {1U, 3U, 8U, 9U, 32U, 33U, 64U}) {
    const auto split = project(chunk);
    for (unsigned i = 0; i < banks.size(); ++i)
      SameBits(bulk[i], split[i], "dense X3 chunk=" + std::to_string(chunk));
  }
  // Independent FP64 formula after exactly the specified FP16 input boundary;
  // use W3's existing X3 accumulation bar, not the looser decode tolerance.
  const double bound = std::ldexp(double(k / 16), -23);
  for (unsigned i = 0; i < banks.size(); ++i)
    for (unsigned row : {0U, rows - 1})
      Near(std::span(bulk[i]).subspan(row * banks[i]->n, banks[i]->n),
           banks[i]->Reference(0, input.data() + row * k, true), bound,
           "dense X3 vs FP64 with rounded input");
  // Two descriptor uploads are queued before either GEMM completes. Host
  // mapped tables reused without stream ordering would leave output poisoned.
  poison(output);
  poison(second);
  queue(0, rows, output);
  queue(0, rows, second);
  const auto a = download(output), b = download(second);
  for (unsigned i = 0; i < banks.size(); ++i) {
    SameBits(bulk[i], a[i], "first queued descriptor group");
    SameBits(bulk[i], b[i], "second queued descriptor group");
  }
  // Production indexer Q/K are views of a 640-column parent. Require exact
  // extraction from full X3 output at the allocation boundary, for all tails.
  auto slice = key.weight;
  slice.output_begin = 512;
  slice.rows = 128;
  auto* sliced = f.buffers.Make<float>(rows * slice.rows + 8);
  for (unsigned chunk : {1U, 9U, 33U, 64U, 65U}) {
    Check(hipMemset(sliced, 0xFF, rows * slice.rows * sizeof(float)));
    Check(hipMemset(sliced + rows * slice.rows, 0x7F, 8 * sizeof(float)));
    for (unsigned first = 0; first < rows; first += chunk)
      Require(Access::DensePrefill(exec, {&slice},
                                   {sliced + first * slice.rows}, x + first * k,
                                   std::min(chunk, rows - first), &error),
              error);
    Check(hipStreamSynchronize(exec.stream()));
    const auto actual = Access::Read<float>(sliced, rows * slice.rows);
    for (unsigned row = 0; row < rows; ++row)
      Require(std::memcmp(actual.data() + row * slice.rows,
                          bulk[1].data() + row * key.n + slice.output_begin,
                          slice.rows * sizeof(float)) == 0,
              "dense X3 output slice differs from full parent bytes");
    Require(Access::Read<std::uint32_t>(sliced + rows * slice.rows, 1)[0] ==
                0x7F7F7F7FU,
            "dense X3 slice overwrote its guard");
  }
}

void PlainOperandPrecision() {
  BeginCase("plain prefill rounds to FP16; decode retains FP32 cancellation");
  constexpr unsigned rows = 65, k = 128, n = 128;
  MtpExecutorFixture fixture(rows, 128, 6, false, k);
  auto& exec = *fixture.exec;
  std::string error;
  Buffers buffers;
  std::vector<float> input(rows * k), expected(rows * n);
  for (unsigned row = 0; row < rows; ++row) {
    const float difference = std::ldexp(float(1 + row % 7), -17);
    input[row * k] = 1.0F + difference;
    input[row * k + 1] = 1.0F;
    std::fill_n(expected.data() + row * n, n, difference);
  }
  auto* x = buffers.Copy(input);
  auto* y = buffers.Make<float>(rows * n);
  gpu::DeviceTensor plain;
  plain.rows = n;
  plain.cols = k;
  for (auto type : {gufo::core::GgmlType::kF16, gufo::core::GgmlType::kBF16,
                    gufo::core::GgmlType::kF32}) {
    plain.type = type;
    std::vector<float> weights(n * k);
    for (unsigned j = 0; j < n; ++j) {
      weights[j * k] = 1;
      weights[j * k + 1] = -1;
    }
    if (type == gufo::core::GgmlType::kF32) {
      plain.data = buffers.Copy(weights);
    } else {
      std::vector<std::uint16_t> bits(weights.size());
      for (std::size_t j = 0; j < bits.size(); ++j)
        bits[j] = type == gufo::core::GgmlType::kF16
                      ? ex::HalfFromDouble(weights[j])
                      : static_cast<std::uint16_t>(
                            std::bit_cast<std::uint32_t>(weights[j]) >> 16);
      plain.data = buffers.Copy(bits);
    }
    for (unsigned chunk : {1U, 8U, 9U, 32U, 33U, 64U, 65U}) {
      Check(hipMemset(y, 0xFF, rows * n * sizeof(float)));
      for (unsigned first = 0; first < rows; first += chunk)
        gpu::Exl3Projection(plain, x + first * k, y + first * n,
                            std::min(chunk, rows - first), {}, nullptr, false);
      SameBits(expected, gpu::ExecutorTestAccess::Read<float>(y, rows * n),
               "plain projection vs exact FP64 cancellation result");
      // Approved prefill contract returns to the pre-66b2d89c FP16 input
      // boundary. These two inputs become identical halves, so their exact
      // dot product is zero; decode above must retain the nonzero FP32 dot.
      Check(hipMemsetAsync(y, 0xFF, rows * n * sizeof(float), exec.stream()));
      for (unsigned first = 0; first < rows; first += chunk)
        Require(gpu::ExecutorTestAccess::PlainPrefill(
                    exec, plain, x + first * k, y + first * n,
                    std::min(chunk, rows - first), &error),
                error);
      Check(hipStreamSynchronize(exec.stream()));
      SameBits(std::vector<float>(rows * n),
               gpu::ExecutorTestAccess::Read<float>(y, rows * n),
               "prefill vs exact FP16 cancellation oracle");
    }
    auto rounded = input;
    for (auto& v : rounded)
      v = float(ex::HalfToDouble(ex::HalfFromDouble(v)));
    auto* hx = buffers.Copy(rounded);
    gpu::Exl3Projection(plain, hx, y, rows, {}, nullptr, false);
    const auto narrowed = gpu::ExecutorTestAccess::Read<float>(y, rows * n);
    Require(std::ranges::all_of(narrowed, [](float v) { return v == 0; }),
            "witness must expose FP16 input's loss of the nonzero dot product");
  }
  // F32 weights are omitted from the uploader's half-weight width bound.
  // A valid wide F32 HC input still needs narrowing scratch from the model
  // geometry, rather than rejecting it against that narrower metadata.
  const unsigned wide_k = fixture.config.HcDim();
  gpu::DeviceTensor wide;
  wide.type = gufo::core::GgmlType::kF32;
  wide.rows = n;
  wide.cols = wide_k;
  wide.data = buffers.Copy(std::vector<float>(n * wide_k, 0.25F));
  auto* wide_x = buffers.Copy(std::vector<float>(rows * wide_k, 0.25F));
  Require(wide_k > fixture.model->max_half_cols(), "F32 HC capacity witness");
  Require(gpu::ExecutorTestAccess::PlainPrefill(exec, wide, wide_x, y, rows,
                                                &error),
          error);
  Check(hipStreamSynchronize(exec.stream()));
  SameBits(std::vector<float>(rows * n, wide_k * 0.0625F),
           gpu::ExecutorTestAccess::Read<float>(y, rows * n),
           "wide F32 HC prefill scratch and exact FP16 oracle", n);
}

void PlainPrefillHalf() {
  BeginCase("plain FP16 WMMA fixed reduction: chunk bytes and rounded FP64");
  using gufo::core::GgmlType;
  using Access = gpu::ExecutorTestAccess;
  constexpr unsigned rows = 129;
  std::mt19937 rng(0xF316A11U);
  std::uniform_real_distribution<float> random(-0.125F, 0.125F);
  for (const auto [k, n] : {std::pair{128U, 33U}, std::pair{320U, 128U},
                            std::pair{10240U, 320U}, std::pair{320U, 10240U}})
    for (auto type : {GgmlType::kF16, GgmlType::kBF16, GgmlType::kF32}) {
      if (type != GgmlType::kF16 && (k == 10240 || n == 10240))
        continue;
      BeginCase("plain FP16 WMMA k=" + std::to_string(k) +
                " n=" + std::to_string(n) +
                " type=" + std::to_string(static_cast<unsigned>(type)));
      Buffers buffers;
      std::vector<float> input(rows * k), weights(n * k);
      for (auto& v : input)
        v = random(rng);
      for (auto& v : weights)
        v = random(rng);
      const void* w = nullptr;
      const auto wt = type == GgmlType::kF16    ? gpu::WeightType::kF16
                      : type == GgmlType::kBF16 ? gpu::WeightType::kBF16
                                                : gpu::WeightType::kF32;
      if (type == GgmlType::kF32) {
        w = buffers.Copy(weights);
      } else {
        std::vector<std::uint16_t> bits(weights.size());
        for (std::size_t i = 0; i < bits.size(); ++i) {
          bits[i] = type == GgmlType::kF16
                        ? ex::HalfFromDouble(weights[i])
                        : std::bit_cast<std::uint32_t>(weights[i]) >> 16;
          weights[i] = type == GgmlType::kF16
                           ? float(ex::HalfToDouble(bits[i]))
                           : std::bit_cast<float>(std::uint32_t{bits[i]} << 16);
        }
        w = buffers.Copy(bits);
      }
      // WMMA consumes these rounded operands. The F64 oracle includes both
      // FP16 boundaries; its 1e-5 bar measures accumulation, not input loss.
      for (auto& v : weights)
        v = float(ex::HalfToDouble(ex::HalfFromDouble(v)));
      auto rounded = input;
      for (auto& v : rounded)
        v = float(ex::HalfToDouble(ex::HalfFromDouble(v)));
      auto* x = buffers.Copy(input);
      auto* half = buffers.Make<std::uint16_t>(input.size());
      gpu::NarrowActivations(x, half, false, input.size(), nullptr);
      auto* y = buffers.Make<float>(rows * n + 8);
      const auto project = [&](unsigned chunk) {
        Check(hipMemset(y, 0xA5, (rows * n + 8) * sizeof(float)));
        for (unsigned first = 0; first < rows; first += chunk)
          Require(
              gpu::PlainPrefillGemm(
                  w, wt, reinterpret_cast<const __half*>(half + first * k),
                  y + first * n, std::min(chunk, rows - first), n, k, nullptr),
              "plain FP16 WMMA shape supported");
        const auto guard = Access::Read<std::uint32_t>(y + rows * n, 8);
        Require(
            std::ranges::all_of(guard, [](auto v) { return v == 0xA5A5A5A5U; }),
            "plain FP16 WMMA preserves ragged output guard");
        return Access::Read<float>(y, rows * n);
      };
      const auto bulk = project(rows);
      for (unsigned chunk : {1U, 8U, 9U, 32U, 33U, 64U, 65U, 128U})
        SameBits(bulk, project(chunk),
                 "plain FP16 WMMA chunk=" + std::to_string(chunk), n);
      for (unsigned r : {0U, rows - 1}) {
        std::vector<float> oracle(n);
        for (unsigned j = 0; j < n; ++j) {
          double dot = 0;
          for (unsigned i = 0; i < k; ++i)
            dot += double(weights[j * k + i]) * rounded[r * k + i];
          oracle[j] = float(dot);
        }
        Near(std::span(bulk).subspan(r * n, n), oracle, 1e-5,
             "plain WMMA vs FP64 of rounded FP16 operands");
      }
    }
}

void PlainPrefillStable() {
  BeginCase("tiled plain prefill vs current SmallGemmStable bytes and FP64");
  using gufo::core::GgmlType;
  using Access = gpu::ExecutorTestAccess;
  constexpr unsigned rows = 65, n = 33;
  std::mt19937 rng(0xF3316U);
  std::uniform_real_distribution<float> random(-0.125F, 0.125F);
  // k=512 selects the reference's vector-load specialization for <=8 rows;
  // ragged k exercises the masked-step break and partial four-element step.
  for (unsigned k : {1U, 127U, 128U, 129U, 132U, 512U})
    for (auto type : {GgmlType::kF16, GgmlType::kBF16, GgmlType::kF32}) {
      Buffers buffers;
      std::vector<float> input(rows * k), weights(n * k);
      for (auto& v : input)
        v = random(rng);
      for (auto& v : weights)
        v = random(rng);
      const void* w = nullptr;
      gpu::WeightType wt = gpu::WeightType::kF32;
      if (type == GgmlType::kF32) {
        w = buffers.Copy(weights);
      } else {
        wt = type == GgmlType::kF16 ? gpu::WeightType::kF16
                                    : gpu::WeightType::kBF16;
        std::vector<std::uint16_t> bits(weights.size());
        for (std::size_t i = 0; i < bits.size(); ++i) {
          bits[i] = type == GgmlType::kF16
                        ? ex::HalfFromDouble(weights[i])
                        : std::bit_cast<std::uint32_t>(weights[i]) >> 16;
          weights[i] = type == GgmlType::kF16
                           ? float(ex::HalfToDouble(bits[i]))
                           : std::bit_cast<float>(std::uint32_t{bits[i]} << 16);
        }
        w = buffers.Copy(bits);
      }
      auto* x = buffers.Copy(input);
      auto* baseline = buffers.Make<float>(rows * n);
      auto* y = buffers.Make<float>(rows * n + 8);
      for (unsigned r = 0; r < rows; ++r)
        gpu::SmallGemmStable(w, wt, x + r * k, baseline + r * n, 1, n, k,
                             nullptr);
      const auto scalar = Access::Read<float>(baseline, rows * n);
      std::vector<float> oracle(rows * n);
      for (unsigned r = 0; r < rows; ++r)
        for (unsigned j = 0; j < n; ++j) {
          double dot = 0;
          for (unsigned i = 0; i < k; ++i)
            dot += double(weights[j * k + i]) * input[r * k + i];
          oracle[r * n + j] = float(dot);
        }
      Near(scalar, oracle, 1e-5, "plain stable vs independent FP64");
      for (unsigned chunk : {1U, 8U, 9U, 15U, 16U, 17U, 32U, 33U, 64U, 65U}) {
        Check(hipMemset(y, 0xA5, (rows * n + 8) * sizeof(float)));
        for (unsigned first = 0; first < rows; first += chunk) {
          const auto count = std::min(chunk, rows - first);
          gpu::SmallGemmStable(w, wt, x + first * k, baseline + first * n,
                               count, n, k, nullptr);
          gufo::hip::LaunchPlainPrefillStable(
              type, w, x + first * k, y + first * n, count, n, k, nullptr);
        }
        SameBits(scalar, Access::Read<float>(baseline, rows * n),
                 "current stable scalar vs grouped/tail bytes");
        SameBits(scalar, Access::Read<float>(y, rows * n),
                 "tiled plain scalar/batch/chunk bytes");
        const auto guard = Access::Read<std::uint32_t>(y + rows * n, 8);
        Require(
            std::ranges::all_of(guard, [](auto v) { return v == 0xA5A5A5A5U; }),
            "tiled plain ragged output overwrote guard");
      }
    }
  // Padding an entirely invalid step with FMAs can turn an underflowed -0
  // partial into +0. Compare the live reference, including its signed bytes.
  Buffers buffers;
  constexpr unsigned k = 132;
  auto* w = buffers.Copy(std::vector<float>(n * k, -0x1p-100F));
  auto* x = buffers.Copy(std::vector<float>(rows * k, 0x1p-100F));
  auto* baseline = buffers.Make<float>(rows * n);
  auto* y = buffers.Make<float>(rows * n);
  gpu::SmallGemmStable(w, gpu::WeightType::kF32, x, baseline, rows, n, k,
                       nullptr);
  gufo::hip::LaunchPlainPrefillStable(GgmlType::kF32, w, x, y, rows, n, k,
                                      nullptr);
  SameBits(Access::Read<float>(baseline, rows * n),
           Access::Read<float>(y, rows * n), "underflowed signed-zero chains");

  // Exercise the two translation units' different optimization flags with
  // explicit zero signs and exactly representable subnormal results. Width8
  // uses vector loads at aligned k; width9 uses the grouped masked path.
  constexpr unsigned special_rows = 9, special_n = 8;
  constexpr std::array<float, 4> zero_operands{0.0F, -0.0F, 1.0F, -1.0F};
  for (unsigned special_k : {133U, 512U})
    for (auto type : {GgmlType::kF16, GgmlType::kBF16, GgmlType::kF32})
      for (unsigned mode = 0; mode < 3; ++mode) {
        Buffers special;
        std::vector<float> weights(special_n * special_k);
        std::vector<float> input(special_rows * special_k);
        std::vector<float> expected(special_rows * special_n);
        for (unsigned j = 0; j < special_n; ++j) {
          if (mode == 0) {
            std::fill_n(weights.data() + j * special_k, special_k,
                        zero_operands[j % zero_operands.size()]);
          } else {
            // Normal * normal = 2^-130, then subnormal * 1 = 2^-130.
            // Only the final K element contributes, so the result stays
            // subnormal even after the wave reduction, with an exact oracle.
            weights[(j + 1) * special_k - 1] =
                (j % 2 ? -1.0F : 1.0F) * (mode == 1 ? 0x1p-12F : 1.0F);
          }
        }
        for (unsigned r = 0; r < special_rows; ++r) {
          const float value = mode == 0
                                  ? zero_operands[r % zero_operands.size()]
                                  : (r % 2 ? -1.0F : 1.0F) *
                                        (mode == 1 ? 0x1p-118F : 0x1p-130F);
          std::fill_n(input.data() + r * special_k, special_k, value);
          if (mode != 0)
            for (unsigned j = 0; j < special_n; ++j)
              expected[r * special_n + j] =
                  (r % 2 == j % 2 ? 1.0F : -1.0F) * 0x1p-130F;
        }
        const void* sw = nullptr;
        const auto wt = type == GgmlType::kF16    ? gpu::WeightType::kF16
                        : type == GgmlType::kBF16 ? gpu::WeightType::kBF16
                                                  : gpu::WeightType::kF32;
        if (type == GgmlType::kF32) {
          sw = special.Copy(weights);
        } else {
          std::vector<std::uint16_t> bits(weights.size());
          for (std::size_t i = 0; i < bits.size(); ++i)
            bits[i] = type == GgmlType::kF16
                          ? ex::HalfFromDouble(weights[i])
                          : std::bit_cast<std::uint32_t>(weights[i]) >> 16;
          sw = special.Copy(bits);
        }
        auto* sx = special.Copy(input);
        auto* reference = special.Make<float>(special_rows * special_n);
        auto* tiled = special.Make<float>(special_rows * special_n);
        std::vector<float> width8;
        for (unsigned width : {8U, 9U}) {
          Check(hipMemset(reference, 0xA5,
                          special_rows * special_n * sizeof(float)));
          Check(
              hipMemset(tiled, 0xA5, special_rows * special_n * sizeof(float)));
          gpu::SmallGemmStable(sw, wt, sx, reference, width, special_n,
                               special_k, nullptr);
          gufo::hip::LaunchPlainPrefillStable(type, sw, sx, tiled, width,
                                              special_n, special_k, nullptr);
          const auto actual = Access::Read<float>(tiled, width * special_n);
          const auto label = "plain special mode=" + std::to_string(mode) +
                             " k=" + std::to_string(special_k) +
                             " width=" + std::to_string(width);
          SameBits(Access::Read<float>(reference, width * special_n), actual,
                   label);
          if (width == 8)
            width8 = actual;
          else
            SameBits(width8,
                     std::vector<float>(actual.begin(),
                                        actual.begin() + 8 * special_n),
                     label + " width8/9 prefix bytes");
          if (mode != 0)
            SameBits(std::vector<float>(expected.begin(),
                                        expected.begin() + width * special_n),
                     actual, label + " exact subnormal oracle");
        }
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
  Require(chunk == 629489160, "4096-row additional scratch byte count");
  Require(through == 649160440, "4224-row tail-capacity scratch byte count");
  std::cout << "MEMORY EXL3 prefill P4096=" << chunk << " bytes (600.328 MiB)"
            << " P4224=" << through << " bytes (619.088 MiB)\n";
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
  MtpExecutorFixture f(rows, 640, 7);
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
  Require(Access::PrefillBorrowsStage(exec, 0) &&
              Access::PrefillBorrowsStage(exec, 1) &&
              Access::PrefillBorrowsStage(exec, rows - 1),
          "prefill gate/up must borrow the current RowScratch stage view");
  // Check all live slabs, including borrowed gate/up and assignment output.
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
    std::uint64_t dense_rows = 0, rotations = 0, gemms = 0;
    Require(exec.BeginMtpCycleDiagnostics(&error), error);
    for (unsigned first = 0; first < rows; first += chunk) {
      const auto count = std::min(chunk, rows - first);
      const bool sparse = first + count > c.indexer_top_k;
      // Q/K/V rotate together, K/V share one GEMM; shared gate/up share
      // both. Indexer Q runs for every row of a sparse call. This exact
      // count catches reverting either caller grouping to separate Dense.
      dense_rows += (std::uint64_t{8} + sparse) * count;
      rotations += 5 + sparse;
      gemms += 6 + sparse;
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
    Require(d.prefill_dense_rows == dense_rows &&
                d.prefill_dense_rotation_groups == rotations &&
                d.prefill_dense_gemm_groups == gemms &&
                d.prefill_plain_rows != 0,
            "real prefill must group Q/K/V and gate/up, dispatch dense X3, and "
            "dispatch plain operands through FP16 WMMA");
    Require(d.prefill_plain_half_rows == d.prefill_plain_rows &&
                d.prefill_plain_half_rows != 0,
            "every EXL3 plain prefill row must dispatch FP16 WMMA");
    Require(std::ranges::all_of(
                logits, [](float value) { return std::isfinite(value); }),
            "prefill logits are not finite");
    return std::pair{logits, hidden};
  };
  const auto bulk = prefill(rows);
  bool equal = true;
  for (unsigned chunk : {1U, 3U, 8U, 9U, 32U, 33U, 64U}) {
    const auto split = prefill(chunk);
    equal = CompareBits(bulk.first, split.first,
                        "executor prefill chunks=" + std::to_string(chunk),
                        c.vocab_size) &&
            equal;
    equal = CompareBits(
                bulk.second, split.second,
                "every prefill residual row chunks=" + std::to_string(chunk),
                c.HcDim()) &&
            equal;
  }
  Require(equal, "all prefill residuals and logits match bitwise");

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
  Require(d.prefill_dense_rows == 0 && d.prefill_plain_rows == 0 &&
              d.prefill_plain_half_rows == 0,
          "decode incorrectly entered prefill dense dispatch");
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

bool ExecutorGdnPrefillCase(bool linear_only) {
  BeginCase(linear_only
                ? "isolated GDN output and state before attention"
                : "executor linear GDN dense X3 wiring and 1/64/65 chunk bits");
  constexpr unsigned rows = 65;
  // A genuine linear layer followed by attention keeps the resident model
  // geometry valid and observes the GDN output through its downstream consumer.
  MtpExecutorFixture f(rows, 640, 7, true);
  auto& exec = *f.exec;
  const auto& c = f.config;
  using Access = gpu::ExecutorTestAccess;
  Require(c.IsLinearLayer(0) && f.model->layers()[0].linear &&
              (linear_only || !f.model->layers()[1].linear),
          "GDN fixture must execute its real linear layer");
  std::vector<std::int32_t> tokens(rows);
  for (unsigned r = 0; r < rows; ++r)
    tokens[r] = 1 + (r * 17) % (c.vocab_size - 1);
  std::string error;
  struct Result {
    std::vector<float> hidden, logits, conv, recurrent;
  };
  const auto run = [&](unsigned chunk) {
    auto session = exec.CreateSession(gufo::core::SessionMode::kAutoregressive,
                                      128, &error);
    Require(session != nullptr, error);
    Result result{std::vector<float>(std::size_t{rows} * c.HcDim()),
                  std::vector<float>(c.vocab_size),
                  {},
                  {}};
    std::uint64_t dense_rows = 0, rotations = 0, gemms = 0;
    Require(exec.BeginMtpCycleDiagnostics(&error), error);
    for (unsigned first = 0; first < rows; first += chunk) {
      const auto count = std::min(chunk, rows - first);
      const bool sparse = first + count > c.indexer_top_k;
      // GDN: QKV/Z grouped rotation (two unequal-N GEMMs), output,
      // shared gate/up and shared down = 6 projection rows, 4 rotations,
      // 5 GEMMs. Add the existing attention layer's exact totals.
      dense_rows += (linear_only ? 6 : std::uint64_t{14} + sparse) * count;
      rotations += linear_only ? 4 : 9 + sparse;
      gemms += linear_only ? 5 : 11 + sparse;
      Require(linear_only
                  ? Access::PrefillLinearOnly(
                        exec, *session, std::span(tokens).subspan(first, count),
                        &error)
                  : exec.Forward(
                        *session, std::span(tokens).subspan(first, count), 0,
                        nullptr, gpu::Executor::ForwardMode::kPrefill, &error),
              error);
      // Disable final-query pruning (n_logits=0); all65 rows fit below the
      // final FFN skip. Preserve every row before scratch is reused.
      const auto hidden = Access::PrefillResiduals(exec, count);
      std::copy(hidden.begin(), hidden.end(),
                result.hidden.begin() + std::size_t{first} * c.HcDim());
      Require(exec.ReplayOutputHead(count, result.logits, &error), error);
    }
    gpu::Executor::MtpCycleDiagnostics d;
    Require(exec.EndMtpCycleDiagnostics(0, 0, &d, &error), error);
    Require(d.prefill_dense_rows == dense_rows &&
                d.prefill_dense_rotation_groups == rotations &&
                d.prefill_dense_gemm_groups == gemms,
            "GDN QKV/Z grouping or output X3 dispatch was reverted");
    Require(d.prefill_expert_rows == rows * (linear_only ? 1 : c.num_layers) &&
                d.prefill_plain_rows != 0 && d.scaled_down_rows == 0,
            "GDN fixture must retain the prefill phase through both layers");
    Require(d.prefill_plain_half_rows == d.prefill_plain_rows &&
                d.prefill_plain_half_rows != 0,
            "GDN plain prefill rows must all dispatch FP16 WMMA");
    auto state = Access::GdnState(*session, c);
    result.conv = std::move(state.first);
    result.recurrent = std::move(state.second);
    for (const auto* values :
         {&result.hidden, &result.logits, &result.conv, &result.recurrent})
      Require(std::ranges::all_of(*values,
                                  [](float v) { return std::isfinite(v); }),
              "GDN prefill fixture produced nonfinite values");
    Require(
        std::ranges::any_of(result.recurrent, [](float v) { return v != 0; }),
        "GDN recurrent state must actually be updated");
    return result;
  };
  const auto bulk = run(rows);
  bool equal = true;
  for (unsigned chunk : {1U, 64U, 65U}) {
    const auto split = run(chunk);
    const auto suffix = " chunks=" + std::to_string(chunk);
    equal = CompareBits(bulk.hidden, split.hidden,
                        "GDN every residual row" + suffix, c.HcDim()) &&
            equal;
    equal = CompareBits(bulk.logits, split.logits,
                        "GDN downstream full logits" + suffix, c.vocab_size) &&
            equal;
    equal =
        CompareBits(bulk.conv, split.conv, "GDN convolution history" + suffix,
                    c.SsmConvChannels()) &&
        equal;
    equal = CompareBits(bulk.recurrent, split.recurrent,
                        "GDN recurrent state" + suffix, c.ssm_head_dim) &&
            equal;
  }
  return equal;
}

void ExecutorGdnPrefill() {
  bool equal = ExecutorGdnPrefillCase(true);
  equal = ExecutorGdnPrefillCase(false) && equal;
  Require(equal,
          "all isolated/downstream GDN outputs and states match bitwise");
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
  bool equal = true;
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
    fast::LaunchExl3FastProjections(kv, x, k, rows, ws, nullptr, false,
                                    std::size_t{q_width} + 2 * kv_width);
    for (const auto& [bank, a, b] : {std::tuple{&key, k_full, k_only},
                                     std::tuple{&value, v_full, v_only}}) {
      std::vector<float> expected(rows * kv_width), actual(expected.size());
      Check(hipMemcpy(expected.data(), a, expected.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
      Check(hipMemcpy(actual.data(), b, actual.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
      equal = CompareBits(expected, actual,
                          std::string(bank == &key ? "K" : "V") +
                              " removing Q, rows=" + std::to_string(rows),
                          kv_width) &&
              equal;
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
  Require(equal, "all K/V group projections preserve full-QKV bits");
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << std::unitbuf;
  try {
    if (argc == 2 && std::string(argv[1]) == "--gdn-prefill-only") {
      ExecutorGdnPrefill();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--dense-prefill-only") {
      DensePrefillProjection();
      PlainOperandPrecision();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--plain-prefill-only") {
      PlainPrefillHalf();
      PlainPrefillStable();
      PlainOperandPrecision();
      return 0;
    }
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
