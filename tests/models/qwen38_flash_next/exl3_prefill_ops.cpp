// Flash-Next EXL3 prefill cases of qwen38_flash_next_exl3_ops_test: dense,
// plain and indexer projections, expert buckets and tail, GDN prefill, shared
// overlap and the scratch estimate.

#include "tests/models/qwen38_flash_next/exl3_ops_fixture.hpp"

void NarrowIndexerPrefill() {
  BeginCase("singleton K5 indexer X3 stage and row boundaries");
  constexpr unsigned rows = 2049, k = 2560, n = 640;
  Buffers buffers;
  Bank bank(buffers, k, n, 5, 1, 23);
  const auto input = Input(rows * k);
  std::vector<std::uint16_t> half(input.size());
  std::transform(input.begin(), input.end(), half.begin(),
                 [](float x) { return ex::HalfFromDouble(x); });
  auto* x = buffers.Copy(half);
  const auto& weight = bank.descriptors[0];
  using Access = gpu::ExecutorTestAccess;
  auto* reference = buffers.Make<float>(rows * n);
  std::vector<gufo::hip::Exl3GemmProblem> chunks;
  for (unsigned first = 0; first < rows; first += 64)
    chunks.push_back({weight.trellis, weight.svh, x + first * k,
                      reference + first * n, std::min(64U, rows - first), 0});
  auto* descriptors = buffers.Copy(chunks);
  for (unsigned i = 0; i < chunks.size(); ++i)
    gufo::hip::LaunchExl3PrefillGemmGrouped(descriptors + i, 1, chunks[i].rows,
                                            k, n, weight.rate, weight.codebook,
                                            false, nullptr);
  const auto expected = Access::Read<float>(reference, rows * n);
  Require(
      std::ranges::all_of(expected, [](float v) { return std::isfinite(v); }) &&
          std::ranges::any_of(expected, [](float v) { return v != 0; }),
      "indexer control must be finite and nonzero");
  for (unsigned count : {64U, 65U, 2048U, 2049U}) {
    auto* output = buffers.Make<float>(count * n + 8);
    Check(hipMemset(output, 0xFF, count * n * sizeof(float)));
    Check(hipMemset(output + count * n, 0x7F, 8 * sizeof(float)));
    const std::vector<gufo::hip::Exl3GemmProblem> problem{
        {weight.trellis, weight.svh, x, output, count, 0}};
    gufo::hip::LaunchExl3PrefillGemmGrouped(buffers.Copy(problem), 1, count, k,
                                            n, weight.rate, weight.codebook,
                                            false, nullptr);
    const auto actual = Access::Read<float>(output, count * n);
    Require(std::memcmp(expected.data(), actual.data(),
                        actual.size() * sizeof(float)) == 0,
            "singleton indexer differs across stage/row dispatch boundary");
    for (auto word : Access::Read<std::uint32_t>(output + count * n, 8))
      Require(word == 0x7F7F7F7FU, "singleton indexer overwrote its guard");
  }
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
  // A prefix uses the wider tile; an unaligned suffix must retain the full
  // parent's 128-channel rotations before extraction.
  for (auto [begin, width] : {std::pair{0U, 512U}, std::pair{513U, 127U}}) {
    auto view = key.weight;
    view.output_begin = begin;
    view.rows = width;
    auto* dst = f.buffers.Make<float>(rows * width + 8);
    for (unsigned chunk : {64U, 65U}) {
      Check(hipMemset(dst, 0xFF, rows * width * sizeof(float)));
      Check(hipMemset(dst + rows * width, 0x7F, 8 * sizeof(float)));
      for (unsigned first = 0; first < rows; first += chunk)
        Require(Access::DensePrefill(exec, {&view}, {dst + first * width},
                                     x + first * k,
                                     std::min(chunk, rows - first), &error),
                error);
      Check(hipStreamSynchronize(exec.stream()));
      const auto actual = Access::Read<float>(dst, rows * width);
      for (unsigned row = 0; row < rows; ++row)
        Require(std::memcmp(actual.data() + row * width,
                            bulk[1].data() + row * key.n + begin,
                            width * sizeof(float)) == 0,
                "aligned prefix or unaligned slice differs from parent");
      for (auto word : Access::Read<std::uint32_t>(dst + rows * width, 8))
        Require(word == 0x7F7F7F7FU, "output slice overwrote its guard");
    }
  }
  auto query = key.weight;
  query.rows = 512;
  auto* queries = f.buffers.Make<float>(rows * query.rows + 8);
  for (unsigned chunk : {1U, 9U, 33U, 64U, 65U}) {
    Check(hipMemset(queries, 0xFF, rows * query.rows * sizeof(float)));
    Check(hipMemset(queries + rows * query.rows, 0x7F, 8 * sizeof(float)));
    Check(hipMemset(sliced, 0xFF, rows * slice.rows * sizeof(float)));
    Check(hipMemset(sliced + rows * slice.rows, 0x7F, 8 * sizeof(float)));
    Require(exec.BeginMtpCycleDiagnostics(&error), error);
    for (unsigned first = 0; first < rows; first += chunk)
      Require(Access::DensePrefill(
                  exec, {&query, &slice},
                  {queries + first * query.rows, sliced + first * slice.rows},
                  x + first * k, std::min(chunk, rows - first), &error),
              error);
    gpu::Executor::MtpCycleDiagnostics d;
    Require(exec.EndMtpCycleDiagnostics(0, 0, &d, &error), error);
    const auto groups = (rows + chunk - 1) / chunk;
    Require(d.prefill_dense_rows == rows &&
                d.prefill_dense_rotation_groups == groups &&
                d.prefill_dense_gemm_groups == groups,
            "shared indexer slices repeated their parent projection");
    Check(hipStreamSynchronize(exec.stream()));
    for (const auto& [view, dst] :
         {std::pair{&query, queries}, std::pair{&slice, sliced}}) {
      const auto actual = Access::Read<float>(dst, rows * view->rows);
      for (unsigned row = 0; row < rows; ++row)
        Require(std::memcmp(actual.data() + row * view->rows,
                            bulk[1].data() + row * key.n + view->output_begin,
                            view->rows * sizeof(float)) == 0,
                "shared indexer slice differs from full parent bytes");
      Require(Access::Read<std::uint32_t>(dst + rows * view->rows, 1)[0] ==
                  0x7F7F7F7FU,
              "shared indexer slice overwrote its guard");
    }
  }
  auto other_slice = slice;
  other_slice.exl3 = value.weight.exl3;
  Require(!Access::DensePrefill(exec, {&query, &other_slice}, {queries, sliced},
                                x, rows, &error),
          "slices of different parents must not share a projection");
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
    std::vector<float> cancel_bits;
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
      // gfx1151 WMMA (f32_16x16x16_f16) does not cancel exactly: 1*1 +
      // (-1)*1 returns -2^-24 (probe: a=(1,-1), b=(1,1) -> -0x1p-24; scaled
      // operands give the same -2^-24 relative residue). Bound the residue
      // by 2^-23 of the absolute product sum (2 here) and keep it chunk
      // invariant; the nonzero FP32 dot (2^-17..7*2^-17) stays excluded.
      Check(hipMemsetAsync(y, 0xFF, rows * n * sizeof(float), exec.stream()));
      for (unsigned first = 0; first < rows; first += chunk)
        Require(gpu::ExecutorTestAccess::PlainPrefill(
                    exec, plain, x + first * k, y + first * n,
                    std::min(chunk, rows - first), &error),
                error);
      Check(hipStreamSynchronize(exec.stream()));
      const auto cancelled = gpu::ExecutorTestAccess::Read<float>(y, rows * n);
      Require(std::ranges::all_of(
                  cancelled,
                  [](float v) { return std::fabs(v) <= std::ldexp(2.0, -23); }),
              "prefill vs FP16 cancellation oracle within the WMMA residue");
      if (chunk == 1)
        cancel_bits = cancelled;
      else
        SameBits(cancel_bits, cancelled,
                 "prefill FP16 cancellation chunk=" + std::to_string(chunk));
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
        // WMMA accumulation error grows with k
        // (plan/exl3/archive/X3-RESULTS.md: ~3.1e-9 k relative L2; the
        // non-cancelling residue above), so the bar is the X3 contract's (k/16)
        // 2^-23, at least 1e-5.
        Near(std::span(bulk).subspan(r * n, n), oracle,
             std::max(1e-5, double(k) / 16.0 * std::ldexp(1.0, -23)),
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
  for (unsigned special_k : {133U, 320U, 512U})
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
  // The inverse slot map adds 4 bytes per assignment (fused combine).
  Require(chunk == 629489160 + 4096 * 10 * 4,
          "4096-row additional scratch byte count");
  Require(through == 649160440 + 4224 * 10 * 4,
          "4224-row tail-capacity scratch byte count");
  std::cout << "MEMORY EXL3 prefill P4096=" << chunk << " bytes (600.48 MiB)"
            << " P4224=" << through << " bytes (619.25 MiB)\n";
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
              c.num_experts_used,
          "shared history must create one 65-row hot-expert bucket for each "
          "selected expert");
    for (unsigned chunk : {1U, 3U, 8U, 9U, 32U, 33U, 64U})
      SameBits(reference, project(chunk),
               "X3 expert operator chunks=" + std::to_string(chunk));
  }
}

void ExecutorPrefillFinalTail() {
  BeginCase("EXL3 final-layer prefill tail routing IDs");
  constexpr unsigned rows = 225;
  MtpExecutorFixture f(rows, 640, 7, false, 0, 256);
  auto& exec = *f.exec;
  std::vector<std::int32_t> tokens(rows);
  for (unsigned r = 0; r < rows; ++r)
    tokens[r] = 1 + (r * 17) % (f.config.vocab_size - 1);
  std::string error;
  const auto prefill = [&](unsigned chunk) {
    auto session = exec.CreateSession(gufo::core::SessionMode::kAutoregressive,
                                      256, &error);
    Require(session != nullptr, error);
    std::vector<float> logits(f.config.vocab_size);
    Require(exec.BeginMtpCycleDiagnostics(&error), error);
    for (unsigned first = 0; first < rows; first += chunk) {
      const auto count = std::min(chunk, rows - first);
      Require(exec.Forward(*session, std::span(tokens).subspan(first, count), 1,
                           logits.data(), gpu::Executor::ForwardMode::kPrefill,
                           &error),
              error);
    }
    gpu::Executor::MtpCycleDiagnostics d;
    Require(exec.EndMtpCycleDiagnostics(0, 0, &d, &error), error);
    Require(d.prefill_expert_rows == (chunk == rows ? 97U : rows),
            "prefill tail fixture did not exercise the FFN skip");
    return logits;
  };
  // The 225-row call skips 128 final-layer FFN rows and shifts RowScratch
  // to its 97-row tail. The 64-row chunks compute all FFN rows, with a fresh
  // router/inverse map per chunk. Their final logits must still be exact.
  SameBits(prefill(rows), prefill(64), "final-layer prefill RowScratch tail");
}

namespace {

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

}  // namespace

void ExecutorSharedOverlap() {
  BeginCase("EXL3 shared overlap: two 512-row chunks, serial tail and state");
  constexpr unsigned rows = 1025;
  MtpExecutorFixture f(512, 640, 7, true, 0, 2048, 512);
  auto& exec = *f.exec;
  const auto& c = f.config;
  using Access = gpu::ExecutorTestAccess;
  std::vector<std::int32_t> tokens(rows);
  for (unsigned r = 0; r < rows; ++r)
    tokens[r] = 1 + (r * 17) % (c.vocab_size - 1);
  struct Result {
    std::vector<float> hidden, logits, conv, recurrent;
  };
  std::string error;
  const auto run = [&](bool overlap) {
    auto session = exec.CreateSession(gufo::core::SessionMode::kAutoregressive,
                                      2048, &error);
    Require(session != nullptr, error);
    Result result{std::vector<float>(std::size_t{rows} * c.HcDim()),
                  std::vector<float>(c.vocab_size),
                  {},
                  {}};
    Require(exec.BeginMtpCycleDiagnostics(&error), error);
    for (unsigned first : {0U, 512U, 1024U}) {
      const auto count = std::min(512U, rows - first);
      const auto part = std::span(tokens).subspan(first, count);
      Require(overlap
                  ? exec.Forward(*session, part, count, nullptr,
                                 gpu::Executor::ForwardMode::kPrefill, &error)
                  : Access::PrefillWithoutSharedWorkspace(exec, *session, part,
                                                          &error),
              error);
      const auto hidden = Access::PrefillResiduals(exec, count);
      std::copy(hidden.begin(), hidden.end(),
                result.hidden.begin() + std::size_t{first} * c.HcDim());
      Require(exec.ReplayOutputHead(count, result.logits, &error), error);
    }
    gpu::Executor::MtpCycleDiagnostics d;
    Require(exec.EndMtpCycleDiagnostics(0, 0, &d, &error), error);
    // Request all head rows to retain both full FFNs. Two chunks exercise
    // shared -> attention -> shared ordering and repeated event generations.
    Require(
        c.num_layers == 2 && d.prefill_shared_overlaps == (overlap ? 4U : 0U),
        "shared overlap eligibility was not exercised at 512 rows");
    auto state = Access::GdnState(*session, c);
    result.conv = std::move(state.first);
    result.recurrent = std::move(state.second);
    for (const auto* values :
         {&result.hidden, &result.logits, &result.conv, &result.recurrent})
      Require(std::ranges::all_of(*values,
                                  [](float v) { return std::isfinite(v); }),
              "nonfinite overlap result");
    return result;
  };
  const auto serial = run(false);
  const auto concurrent = run(true);
  bool equal = CompareBits(serial.hidden, concurrent.hidden,
                           "shared overlap every residual", c.HcDim());
  equal = CompareBits(serial.logits, concurrent.logits,
                      "shared overlap tail logits", c.vocab_size) &&
          equal;
  equal =
      CompareBits(serial.conv, concurrent.conv,
                  "shared overlap convolution history", c.SsmConvChannels()) &&
      equal;
  equal = CompareBits(serial.recurrent, concurrent.recurrent,
                      "shared overlap recurrent state", c.ssm_head_dim) &&
          equal;
  Require(equal, "shared overlap changed executor results");
}

void ExecutorGdnPrefill() {
  bool equal = ExecutorGdnPrefillCase(true);
  equal = ExecutorGdnPrefillCase(false) && equal;
  Require(equal,
          "all isolated/downstream GDN outputs and states match bitwise");
}
