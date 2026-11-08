// Flash-Next EXL3 operator contracts on the GPU: decode projections, routed
// experts, shared MLP invariance, producer rotation and decode side branches.
// main() selects the cases; the prefill and MTP cases are in
// exl3_prefill_ops.cpp and exl3_mtp_ops.cpp.

#include "tests/models/qwen38_flash_next/exl3_ops_fixture.hpp"

namespace {

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

// gate_up_half: K3.5 gate/up banks (Vemloir Flash-Next), K5 down.
void Routed(bool shared, unsigned rows, unsigned k = 128, unsigned ff = 128,
            unsigned experts = 512, bool gate_up_half = false) {
  const int gate_up_bits = gate_up_half ? 3 : 5;
  BeginCase(Shape(std::string(shared ? "routed shared" : "routed disjoint") +
                      (gate_up_half ? " gate_up=K3.5" : ""),
                  k, ff, 5, rows) +
            " down_n=" + std::to_string(k) +
            " experts=" + std::to_string(experts) + " topk=10");
  Buffers buffers;
  constexpr unsigned used = 10;
  Bank gate(buffers, k, ff, gate_up_bits, experts, 1, gate_up_half),
      up(buffers, k, ff, gate_up_bits, experts, 2, gate_up_half),
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

// Decode/verify with the GDN alpha/beta projection and the attention
// indexer on side branches must keep every logit and state byte of the
// serial order (a GDN layer then a sparse attention layer, past the
// indexer's top-k so selection runs, AR decode steps and verify passes).
void DecodeSideBranches() {
  BeginCase("EXL3 decode side branches: alpha/beta and indexer");
  MtpExecutorFixture f(32, 128, 6, true);
  auto& exec = *f.exec;
  const auto& c = f.config;
  using Access = gpu::ExecutorTestAccess;
  Require(Access::SideBranchAvailable(exec), "decode side branch allocated");
  std::vector<std::int32_t> prompt(20);
  for (unsigned i = 0; i < prompt.size(); ++i)
    prompt[i] = 1 + (i * 29) % (c.vocab_size - 1);
  std::string error;
  struct Result {
    std::vector<float> logits, conv, recurrent;
  };
  const auto run = [&](bool side, gufo::core::SessionMode session_mode) {
    auto session = exec.CreateSession(session_mode, 128, &error);
    Require(session != nullptr, error);
    Require(exec.Forward(*session, prompt, 1, nullptr,
                         gpu::Executor::ForwardMode::kPrefill, &error),
            error);
    const bool verify = session_mode == gufo::core::SessionMode::kSpeculative;
    const unsigned width = verify ? 4 : 1;
    const auto mode = verify ? gpu::Executor::ForwardMode::kVerify
                             : gpu::Executor::ForwardMode::kDecode;
    Result result;
    const auto forks = Access::SideBranches(exec);
    for (unsigned step = 0; step < 6; ++step) {
      std::vector<std::int32_t> tokens(width);
      for (unsigned i = 0; i < width; ++i)
        tokens[i] = 1 + (step * 37 + i * 11) % (c.vocab_size - 1);
      std::vector<float> logits(std::size_t{width} * c.vocab_size);
      Require(
          side ? exec.Forward(*session, tokens, width, logits.data(), mode,
                              &error)
               : Access::ForwardWithoutSideBranch(exec, *session, tokens, width,
                                                  logits.data(), mode, &error),
          error);
      result.logits.insert(result.logits.end(), logits.begin(), logits.end());
    }
    // Steps fork the GDN and attention branches (graph replays do not
    // count), or never.
    const auto taken = Access::SideBranches(exec) - forks;
    Require(side ? taken >= 2 : taken == 0,
            "decode side branches were not taken as expected");
    auto state = Access::GdnState(*session, c);
    result.conv = std::move(state.first);
    result.recurrent = std::move(state.second);
    Require(std::ranges::all_of(result.logits,
                                [](float v) { return std::isfinite(v); }),
            "nonfinite side-branch logits");
    return result;
  };
  bool equal = true;
  for (const auto mode : {gufo::core::SessionMode::kAutoregressive,
                          gufo::core::SessionMode::kSpeculative}) {
    const auto serial = run(false, mode);
    const auto branched = run(true, mode);
    equal = CompareBits(serial.logits, branched.logits, "side branch logits",
                        c.vocab_size) &&
            equal;
    equal = CompareBits(serial.conv, branched.conv, "side branch conv state",
                        c.SsmConvChannels()) &&
            equal;
    equal = CompareBits(serial.recurrent, branched.recurrent,
                        "side branch recurrent state", c.ssm_head_dim) &&
            equal;
  }
  Require(equal, "decode side branches changed executor results");
}

// Producer epilogues that also write the rotated FP16 input of the next
// EXL3 decode projections. The rotation runs the standalone kernels'
// arithmetic on the same stored FP32 values, so producer outputs and every
// projection must keep their bits (rows 1..8, every pass width, signed zero
// and subnormal inputs, a slab prefilled with NaN so padded rows are unused).
void ProducerRotation() {
  namespace fast = gufo::hip;
  constexpr unsigned hidden = 2560, streams = 4, width = 6144;
  constexpr unsigned k_heads = 16, v_heads = 48, head = 128, taps = 4;
  constexpr unsigned channels = (2 * k_heads + v_heads) * head;
  constexpr unsigned heads = 24, head_dim = 256;
  Buffers buffers;
  Bank q(buffers, hidden, 512, 6, 1, 31), key(buffers, hidden, 256, 6, 1, 32),
      value(buffers, hidden, 256, 6, 1, 33);
  Bank gate(buffers, hidden, 640, 4, 1, 34), up(buffers, hidden, 640, 4, 1, 35),
      down(buffers, 640, hidden, 5, 1, 36);
  Bank out(buffers, width, hidden, 3, 1, 37);
  auto scratch = Scratch(buffers, 8, width, width);
  scratch.rotated_input_elements = std::size_t{24} * width;
  scratch.rotated_input =
      buffers.Make<std::uint16_t>(scratch.rotated_input_elements);
  const auto& p = scratch.projection;
  const fast::Exl3FastWorkspace ws{
      p.fast_input,         p.fast_input_elements, p.fast_counters,
      p.fast_counter_count, p.fast_down_input,     p.fast_down_input_elements};
  const fast::Exl3FastWorkspace pre{
      scratch.rotated_input, scratch.rotated_input_elements,
      p.fast_counters,       p.fast_counter_count,
      p.fast_down_input,     p.fast_down_input_elements};
  const auto special = [](std::vector<float> x) {
    for (std::size_t i = 0; i + 64 < x.size(); i += 997) {
      x[i] = -0.0F;
      x[i + 1] = 0.0F;
      x[i + 2] = 1e-40F;
      x[i + 3] = -3e-39F;
      x[i + 4] = std::numeric_limits<float>::denorm_min();
    }
    return x;
  };
  const auto download = [](const auto* d, std::size_t count) {
    std::vector<std::remove_cv_t<std::remove_pointer_t<decltype(d)>>> h(count);
    Check(hipMemcpy(h.data(), d, count * sizeof(h[0]), hipMemcpyDeviceToHost));
    return h;
  };
  const auto slabs = [&](std::initializer_list<const Bank*> banks,
                         unsigned rows) {
    gpu::Exl3InputSlabs s{};
    for (const Bank* b : banks)
      s.suh[s.count++] = b->weight.exl3->suh;
    s.out = scratch.rotated_input;
    s.pass_rows = fast::Exl3FastPassRows(rows);
    s.k = banks.begin()[0]->k;
    // Padded rows and stale slabs must never be read.
    Check(hipMemset(scratch.rotated_input, 0xFF,
                    scratch.rotated_input_elements * 2));
    return s;
  };
  // Pre-rotated launches get NaN rows in place of x: they must not read it.
  auto* nan_x = buffers.Make<float>(std::size_t{8} * width);
  Check(hipMemset(nan_x, 0xFF, std::size_t{8} * width * sizeof(float)));
  const auto project = [&](const float* x, unsigned rows, bool rotated) {
    auto* y = buffers.Make<float>(std::size_t{rows} * hidden);
    gpu::Exl3Projection(out.weight, rotated ? nan_x : x, y, rows, scratch,
                        nullptr, true,
                        rotated ? scratch.rotated_input : nullptr);
    return download(y, std::size_t{rows} * hidden);
  };
  for (unsigned rows = 1; rows <= 8; ++rows) {
    const std::string tag = " rows=" + std::to_string(rows);
    BeginCase("producer rotation HC mix" + tag);
    {
      const auto* xn = buffers.Copy(special(Input(rows * streams * hidden)));
      auto g = Input(rows * streams * hidden);
      std::reverse(g.begin(), g.end());
      const auto* dg = buffers.Copy(g);
      const std::size_t q8_bytes = std::size_t{rows} * hidden / 32 * 36;
      float* mixed[3];
      std::uint8_t* q8[3];
      for (int i = 0; i < 3; ++i) {
        mixed[i] = buffers.Make<float>(rows * hidden);
        q8[i] = buffers.Make<std::uint8_t>(q8_bytes);
      }
      const auto group = [&](const float* x, bool rotated) {
        std::array<float*, 3> y{buffers.Make<float>(rows * 512),
                                buffers.Make<float>(rows * 256),
                                buffers.Make<float>(rows * 256)};
        const std::array<fast::Exl3FastTarget, 3> targets{
            {{q.weight.exl3, y[0], 512},
             {key.weight.exl3, y[1], 256},
             {value.weight.exl3, y[2], 256}}};
        fast::LaunchExl3FastProjections(targets, rotated ? nan_x : x, hidden,
                                        rows, rotated ? pre : ws, nullptr,
                                        rotated);
        std::vector<float> all = download(y[0], rows * 512);
        for (int i = 1; i < 3; ++i) {
          const auto part = download(y[i], rows * 256);
          all.insert(all.end(), part.begin(), part.end());
        }
        return all;
      };
      const auto mlp = [&](const float* x, bool rotated) {
        auto* y = buffers.Make<float>(rows * hidden);
        const fast::Exl3FastMlp m{gate.weight.exl3,
                                  up.weight.exl3,
                                  down.weight.exl3,
                                  buffers.Make<float>(8 * 640),
                                  buffers.Make<float>(8 * 640),
                                  y,
                                  hidden};
        fast::LaunchExl3FastMlp(m, rotated ? nan_x : x, hidden, rows,
                                rotated ? pre : ws, nullptr, rotated);
        return download(y, rows * hidden);
      };
      // The production decode epilogue fuses the inject partials too.
      const auto* inject_w = buffers.Copy(Input(streams * streams * hidden));
      const std::size_t parts = gpu::HcInjectParts(hidden);
      float* inject[3];
      for (auto& partials : inject)
        partials = buffers.Make<float>(rows * streams * parts);
      gpu::HcMixEpilogue(xn, dg, inject_w, mixed[0], inject[0], rows, hidden,
                         streams, nullptr, q8[0]);
      const auto group_ref = group(mixed[0], false);
      const auto mlp_ref = mlp(mixed[0], false);
      Require(gpu::HcMixEpilogue(xn, dg, inject_w, mixed[1], inject[1], rows,
                                 hidden, streams, nullptr, q8[1],
                                 slabs({&q, &key, &value}, rows)),
              "HC mix epilogue rotates for q/k/v");
      SameBits(group_ref, group(mixed[1], true), "q/k/v from HC mix slabs");
      Require(gpu::HcMixEpilogue(xn, dg, inject_w, mixed[2], inject[2], rows,
                                 hidden, streams, nullptr, q8[2],
                                 slabs({&gate, &up}, rows)),
              "HC mix epilogue rotates for the shared MLP");
      SameBits(mlp_ref, mlp(mixed[2], true), "shared MLP from HC mix slabs");
      for (int i = 1; i < 3; ++i) {
        SameBits(download(mixed[0], rows * hidden),
                 download(mixed[i], rows * hidden), "HC mixed rows");
        SameBits(download(inject[0], rows * streams * parts),
                 download(inject[i], rows * streams * parts),
                 "HC inject partials");
        Require(download(q8[0], q8_bytes) == download(q8[i], q8_bytes),
                "HC mixed Q8_1 rows differ");
      }
    }
    BeginCase("producer rotation GDN epilogue" + tag);
    {
      auto qkv = Input(rows * channels);
      for (auto& v : qkv)
        v *= 16.0F;
      const auto* dqkv = buffers.Copy(special(qkv));
      const auto* dz = buffers.Copy(special(Input(rows * width)));
      const auto* dab = buffers.Copy(Input(rows * 2 * v_heads));
      const auto* conv_w = buffers.Copy(Input(channels * taps));
      const auto* a = buffers.Copy(Input(v_heads));
      const auto* dt = buffers.Copy(Input(v_heads));
      auto norm = Input(head);
      for (auto& v : norm)
        v += 1.0F;
      const auto* norm_w = buffers.Copy(norm);
      auto* conv_state = buffers.Make<float>((taps - 1) * channels);
      auto* conv_scratch = buffers.Make<float>((rows + taps) * channels);
      auto* qn = buffers.Make<float>(rows * k_heads * head);
      auto* kn = buffers.Make<float>(rows * k_heads * head);
      auto* raw = buffers.Make<float>(rows * width);
      auto* state = buffers.Make<float>(std::size_t{v_heads} * head * head);
      float* gdn[2] = {buffers.Make<float>(rows * width),
                       buffers.Make<float>(rows * width)};
      std::vector<float> states[2];
      for (int run = 0; run < 2; ++run) {
        Check(hipMemset(conv_state, 0, (taps - 1) * channels * 4));
        Check(hipMemset(state, 0, std::size_t{v_heads} * head * head * 4));
        auto s = run == 1 ? slabs({&out}, rows) : gpu::Exl3InputSlabs{};
        Require(gpu::GatedDeltaNet(dqkv, channels, dz, width, dab, conv_w, a,
                                   dt, norm_w, conv_state, conv_scratch, qn, kn,
                                   raw, state, gdn[run], nullptr, {}, {}, rows,
                                   k_heads, v_heads, head, taps, false, false,
                                   1e-6F, nullptr, nullptr, {}, 0, nullptr,
                                   nullptr, true, run == 1 ? &s : nullptr),
                "GDN launch");
        Require(run == 0 || s.count == 1, "GDN epilogue rotates for ssm_out");
        states[run] = download(state, std::size_t{v_heads} * head * head);
      }
      SameBits(states[0], states[1], "GDN state");
      SameBits(download(gdn[0], rows * width), download(gdn[1], rows * width),
               "GDN output rows");
      // The reference projection rotates through the fast workspace and
      // leaves run 1's slab in place.
      const auto ref = project(gdn[0], rows, false);
      SameBits(ref, project(gdn[1], rows, true), "ssm_out from GDN slabs");
    }
    BeginCase("producer rotation attention gate" + tag);
    {
      const auto ctx = special(Input(rows * width));
      const auto* g = buffers.Copy(Input(rows * width));
      float* x[2] = {buffers.Copy(ctx), buffers.Copy(ctx)};
      gpu::SigmoidMul(x[0], g, std::size_t{rows} * width, nullptr);
      const auto ref = project(x[0], rows, false);
      Require(gpu::SigmoidMulRotate(x[1], g, rows, width, slabs({&out}, rows),
                                    nullptr),
              "gate rotates for attn_out");
      SameBits(ref, project(x[1], rows, true), "attn_out from gate slabs");
      SameBits(download(x[0], rows * width), download(x[1], rows * width),
               "gated context rows");
    }
    BeginCase("producer rotation KVarN context gate" + tag);
    {
      const auto ctx = special(Input(rows * width));
      const auto* qg = buffers.Copy(Input(rows * 2 * width));
      float* x[2] = {buffers.Copy(ctx), buffers.Copy(ctx)};
      Require(
          !gpu::GateRotatedContext(x[0], qg, 2 * width, rows, heads, nullptr),
          "plain KVarN gate writes no slab");
      const auto ref = project(x[0], rows, false);
      Require(gpu::GateRotatedContext(x[1], qg, 2 * width, rows, heads, nullptr,
                                      slabs({&out}, rows)),
              "KVarN gate rotates for attn_out");
      SameBits(ref, project(x[1], rows, true), "attn_out from KVarN slabs");
      SameBits(download(x[0], rows * width), download(x[1], rows * width),
               "KVarN gated context rows");
      static_assert(heads * head_dim == width);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << std::unitbuf;
  try {
    if (argc == 2 && std::string(argv[1]) == "--prefill-overlap-only") {
      ExecutorSharedOverlap();
      ExecutorGdnPrefill();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--gdn-prefill-only") {
      ExecutorGdnPrefill();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--dense-prefill-only") {
      NarrowIndexerPrefill();
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
      ExecutorPrefillFinalTail();
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
    if (argc == 2 && std::string(argv[1]) == "--decode-side-branches") {
      DecodeSideBranches();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--producer-rotation") {
      ProducerRotation();
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
    for (int bits : {5, 6, 7, 8})
      Projection(256, 640, bits);
    Projection(640, 2560, 7);
    Projection(2560, 6144, 7, {1U, 3U});
    Projection(2560, 10240, 7, {1U});
    Projection(6144, 2560, 7, {1U});
    Routed(true, 3, 2560, 640, 16);
    ProducerRotation();
    DecodeSideBranches();
    SelectedHead();
    for (unsigned rows : {1U, 2U, 3U, 7U, 8U, 9U, 32U, 33U, 64U}) {
      Routed(true, rows);
      Routed(false, rows);
    }
    for (unsigned rows : {1U, 3U, 8U, 9U})
      Routed(false, rows, 256, 256, 64, true);
    std::cout << "Flash-Next EXL3 GPU contracts passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
