#ifndef GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_EXL3_OPS_FIXTURE_HPP_
#define GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_EXL3_OPS_FIXTURE_HPP_

// Shared fixture of the Flash-Next EXL3 operator test binary
// (qwen38_flash_next_exl3_ops_test): executor access, buffers, inputs and
// bit comparisons used by its decode, prefill and MTP cases.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

#include "src/core/hip/detail/exl3_gemv.hpp"
#include "src/core/quant/exl3/gemv.hpp"
#include "src/models/qwen/hip/ops/plain_prefill_stable.hpp"
#include "src/models/qwen38_flash_next/exl3_weights.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/exl3_moe.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/runtime/executor.hpp"

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
  static bool PrefillWithoutSharedWorkspace(
      Executor& exec, Session& session, std::span<const std::int32_t> tokens,
      std::string* error) {
    // Exercise the production serial fallback with identical rows/kernels,
    // without adding a runtime optimization switch.
    struct Restore {
      Executor& exec;
      Executor::Exl3DenseWorkspace workspace;
      ~Restore() { exec.exl3_shared_workspace_ = workspace; }
    } restore{exec, exec.exl3_shared_workspace_};
    exec.exl3_shared_workspace_ = {};
    return exec.Forward(session, tokens, tokens.size(), nullptr,
                        Executor::ForwardMode::kPrefill, error);
  }
  // The production serial order of a decode/verify forward: without the
  // side branch's scratch (no runtime switch), ForkSide is never taken.
  static bool ForwardWithoutSideBranch(Executor& exec, Session& session,
                                       std::span<const std::int32_t> tokens,
                                       std::uint32_t n_logits, float* logits,
                                       Executor::ForwardMode mode,
                                       std::string* error) {
    struct Restore {
      Executor& exec;
      Exl3Scratch side;
      ~Restore() { exec.exl3_side_ = side; }
    } restore{exec, exec.exl3_side_};
    exec.exl3_side_ = {};
    return exec.Forward(session, tokens, n_logits, logits, mode, error);
  }
  static bool SideBranchAvailable(const Executor& exec) {
    return exec.SideBranchAvailable();
  }
  static std::uint64_t SideBranches(const Executor& exec) {
    return exec.side_branches_;
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
  /// Executor-owned decode and wide verification storage: a pass that
  /// allocated any of it lazily would change this tuple.
  static auto DecodeAllocations(const Executor& exec) {
    return std::tuple(static_cast<const void*>(exec.batch_logits_),
                      static_cast<const void*>(exec.batch_controls_),
                      static_cast<const void*>(exec.batch_candidates_host_),
                      static_cast<const void*>(exec.batch_gdn_host_),
                      static_cast<const void*>(exec.wide_controls_),
                      exec.wide_allocations_.size(), exec.wide_segments_.size(),
                      exec.wide_state_.size(),
                      static_cast<const void*>(exec.verification_penalties_),
                      exec.verification_penalty_capacity_);
  }
  /// Device bytes ReserveDecodeScratch moves out of DeferredScratchBytes:
  /// the batch logits and the wide segments past the first and their state.
  static std::size_t ReservedDecodeBytes(const Executor& exec) {
    const auto segments =
        (exec.max_copy_rows() + Executor::kWideSegmentRows - 1) /
        Executor::kWideSegmentRows;
    return std::size_t{8} * std::min(8U, exec.options_.max_logit_rows) *
               exec.config().vocab_size * sizeof(float) +
           (segments > 1 ? ((segments - 1) * exec.WideSegmentFloats() +
                            exec.WideStateFloats()) *
                               sizeof(float)
                         : 0);
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

/// The running case, named in every failure message (BeginCase).
inline std::string current_case = "initialization";

namespace {

inline void BeginCase(const std::string& name) {
  current_case = name;
  std::cout << "CASE " << current_case << '\n';
}

inline std::string Shape(const std::string& operation, unsigned k, unsigned n,
                         int bits, unsigned rows) {
  return operation + " k=" + std::to_string(k) + " n=" + std::to_string(n) +
         " K=" + std::to_string(bits) + " rows=" + std::to_string(rows);
}

inline void Require(bool value, const std::string& message) {
  std::cout << (value ? "PASS " : "FAIL ") << current_case << " : " << message
            << '\n';
  if (!value)
    throw std::runtime_error(current_case + " : " + message);
}

inline void Check(hipError_t status) {
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
  bool half;
  std::vector<std::uint16_t> trellis, suh, svh;
  std::vector<ex::DeviceLinear> descriptors;
  gpu::DeviceTensor weight;

  Bank(Buffers& buffers, unsigned inputs, unsigned outputs, int rate,
       unsigned count = 1, unsigned seed = 12345, bool half_rate = false)
      : k(inputs), n(outputs), experts(count), bits(rate), half(half_rate) {
    const auto words =
        std::size_t{k / 16} * (n / 16) * ex::BitRate{bits, half}.TileWords();
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
                             {bits, half},
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
    const auto words =
        std::size_t{k / 16} * (n / 16) * ex::BitRate{bits, half}.TileWords();
    std::vector<std::uint16_t> inner(std::size_t{k} * n);
    ex::DequantInner(trellis.data() + expert * words, k, n, {bits, half},
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

inline gpu::Exl3Scratch Scratch(Buffers& buffers, unsigned rows, unsigned k,
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

inline std::vector<float> Input(unsigned count) {
  std::mt19937 rng(56234);
  std::vector<float> x(count);
  for (auto& value : x)
    value = (static_cast<int>(rng() % 2000) - 1000) * 0.00003125F;
  return x;
}
inline double RelativeL2(std::span<const float> a, std::span<const float> b) {
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

inline void Near(std::span<const float> a, std::span<const float> b,
                 double tolerance, const std::string& check) {
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
                              unsigned half_weight_cols = 0,
                              unsigned max_context = 128,
                              unsigned max_logit_rows = 8) {
    config.num_layers = linear_prefix ? 2 : 1;
    config.num_layers_all = config.num_layers + 1;
    config.hidden_size = 128;
    config.vocab_size = 256;
    config.context_length = max_context / 2;
    config.hc_count = 4;
    config.hc_low_rank = 32;
    config.full_attention_interval = linear_prefix ? 2 : 1;
    config.num_heads = 4;
    config.num_kv_heads = 1;
    config.head_dim = 256;
    config.rotary_dim = 64;
    config.rope_theta = 10000;
    config.rope_scaling = {.factor = 2.0F,
                           .original_context = config.context_length};
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
                                  .max_logit_rows = max_logit_rows,
                                  .max_speculative = 8,
                                  .max_context = max_context},
                                 &error);
    Require(exec != nullptr, error);
    Require(q::MakeYarnRope(config.rope_scaling, config.rotary_dim,
                            config.rope_theta)
                    .enabled != 0,
            "executor fixture must exercise enabled YaRN");
  }
};

template<class T>
inline bool CompareBits(const std::vector<T>& a, const std::vector<T>& b,
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
inline void SameBits(const std::vector<T>& a, const std::vector<T>& b,
                     const std::string& name, std::size_t columns = 0) {
  if (!CompareBits(a, b, name, columns))
    throw std::runtime_error(current_case + " : " + name + " differs bitwise");
}

}  // namespace

// Cases defined in exl3_prefill_ops.cpp and exl3_mtp_ops.cpp, run from
// exl3_ops_test.cpp main().
void MtpExecutorCatchUp(bool batched);
void WideVerification();
void WideDecodeAllocations();
void NarrowIndexerPrefill();
void DensePrefillProjection();
void PlainOperandPrecision();
void PlainPrefillHalf();
void PlainPrefillStable();
void PrefillScratchEstimate();
void ExecutorPrefillExperts();
void ExecutorPrefillFinalTail();
void ExecutorSharedOverlap();
void ExecutorGdnPrefill();
void MtpKvGroup();

#endif  // GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_EXL3_OPS_FIXTURE_HPP_
