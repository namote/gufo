// Dense projections: Q8 activation inputs, GGUF/F16/EXL3 dense GEMM
// routes, EXL3 rotated-input slabs and grouped projections.

#include "src/models/qwen38_flash_next/kernels/rocm/executor_internal.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

/// Column-tile width for the routed expert GEMMs: the tile at or above twice
/// the mean bucket, so most experts fit one tile with little padding.
int RoutedTileCols(std::uint32_t n_tokens, std::uint32_t n_used,
                   std::uint32_t n_experts) {
  const std::uint32_t mean =
      std::max<std::uint32_t>(1, n_tokens * n_used / std::max(n_experts, 1u));
  for (int cols = 16; cols < 80; cols += 16) {
    if (static_cast<std::uint32_t>(cols) >= 2 * mean) {
      return cols;
    }
  }
  return 80;
}

const void* Executor::ReadyQ8(const float* x, std::uint32_t n_tokens,
                              std::uint32_t k) const {
  for (const ReadyRows* r : {&xn_q8_ready_, &mixed_q8_ready_}) {
    if (r->src != nullptr && r->src == x && r->n == n_tokens && r->k == k) {
      return r->data;
    }
  }
  return nullptr;
}

void Executor::DropReadyQ8() const {
  xn_q8_ready_ = {};
  mixed_q8_ready_ = {};
  exl3_rotated_ = {};  // also producer-rotated rows
}

bool Executor::Quantize(const float* x, std::uint32_t n_tokens, std::uint32_t k,
                        Q8Input* q, std::string* error_msg,
                        bool q8_weight) const {
  q->x = x;
  q->data = nullptr;
  q->n = n_tokens;
  q->k = k;
  if ((model_->output().exl3 != nullptr && !q8_weight) ||
      MatrixRows(n_tokens)) {
    return true;  // EXL3 consumes the original FP32 rows; wide Q8 stages per
                  // call
  }
  if (const void* ready = ReadyQ8(x, n_tokens, k)) {
    q->data = ready;
    return true;
  }
  // Two slots alternate, so an input stays valid across one other
  // quantization; captured graphs replay the same alternation.
  void* slot = s_.x_q8[q8_slot_];
  q8_slot_ ^= 1u;
  if (qfn_mmq_quantize_q8_1(x, slot, static_cast<int>(n_tokens),
                            static_cast<int>(k), stream_) != 0) {
    AssignError(error_msg, "activation quantization failed");
    return false;
  }
  q->data = slot;
  return true;
}

bool Executor::Dense(const DeviceTensor& w, const Q8Input& q, float* out,
                     std::string* error_msg) const {
  if (w.type == GgmlType::kQ8_0 && q.data != nullptr) {
    if (w.cols != q.k) {
      AssignError(error_msg, "quantized input width mismatch");
      return false;
    }
    if (qfn_mmq_q8_0_dense_vec_preq(
            w.data, nullptr, q.data, out, static_cast<int>(w.rows),
            static_cast<int>(q.n), static_cast<int>(w.cols), stream_) != 0) {
      AssignError(error_msg, "Q8_0 GEMV failed");
      return false;
    }
    return true;
  }
  return Dense(w, q.x, out, q.n, error_msg);
}

bool Executor::GatedDense(const DeviceTensor& up, const DeviceTensor& gate,
                          const float* x, float* out, std::uint32_t n_tokens,
                          const DeviceTensor* down,
                          std::string* error_msg) const {
  if (prefill_phase && gate.exl3 != nullptr && up.exl3 != nullptr &&
      gate.cols == up.cols) {
    if (!Exl3DensePrefill({&gate, &up}, {out, s_.shexp_gate}, x, n_tokens,
                          error_msg))
      return false;
    Swiglu(out, s_.shexp_gate, std::size_t{n_tokens} * gate.rows, stream_);
    return true;
  }
  // Decode and verification use the same fused projections and activation.
  // A different SwiGLU rounding can change later activation quantization.
  if (!MatrixRows(n_tokens) && up.type == GgmlType::kQ8_0 &&
      gate.type == GgmlType::kQ8_0 && up.rows == gate.rows &&
      up.cols == gate.cols) {
    Q8Input xq;
    if (!Quantize(x, n_tokens, up.cols, &xq, error_msg)) {
      return false;
    }
    if (qfn_mmq_q8_0_dense_vec_preq(up.data, gate.data, xq.data, out,
                                    static_cast<int>(up.rows),
                                    static_cast<int>(n_tokens),
                                    static_cast<int>(up.cols), stream_) != 0) {
      AssignError(error_msg, "gated Q8_0 GEMV failed");
      return false;
    }
    return true;
  }
  // Swiglu is in place over its first operand.
  if (!Dense(gate, x, out, n_tokens, error_msg) ||
      !Dense(up, x, s_.shexp_gate, n_tokens, error_msg)) {
    return false;
  }
  // A wide batch writes the down projection's staged input directly (the
  // F32 rows are read by nothing else): F16 rows for the F16 route, else
  // the tiled Q8 layout; the cache lets Dense skip its activation pass.
  if (down != nullptr && down->cols == up.rows && MatrixRows(n_tokens) &&
      n_tokens <= options_.max_batch) {
    if (DenseF16Route(*down, n_tokens)) {
      // A private F16 buffer: s_.x_half keeps the narrowed token rows the
      // router left there, which the routed experts read next.
      SwigluHalf(out, s_.shexp_gate, s_.shexp_half,
                 static_cast<std::size_t>(n_tokens) * up.rows, stream_);
      shexp_half_ready_ = true;
      return true;
    }
    if (down->type == GgmlType::kQ8_0 &&
        SwigluQ8Tiled(out, s_.shexp_gate, s_.x_q8t, n_tokens, up.rows,
                      stream_)) {
      q8t_src_ = out;
      q8t_rows_ = n_tokens;
      q8t_cols_ = up.rows;
      return true;
    }
  }
  Swiglu(out, s_.shexp_gate, static_cast<std::size_t>(n_tokens) * up.rows,
         stream_);
  return true;
}

void Executor::PrepareHalfInput(const float* x, std::uint32_t rows,
                                std::uint32_t cols) const {
  if (half_src_ == x && half_rows_ == rows && half_cols_ == cols &&
      !half_bf16_) {
    return;
  }
  NarrowActivations(x, s_.x_half, false, static_cast<std::size_t>(rows) * cols,
                    stream_);
  half_src_ = x;
  half_rows_ = rows;
  half_cols_ = cols;
  half_bf16_ = false;
}

bool Executor::DenseF16Route(const DeviceTensor& w,
                             std::uint32_t n_tokens) const {
  return w.type == GgmlType::kQ8_0 && MatrixRows(n_tokens) &&
         w.rows >= kDenseF16MinRows && w.cols <= kDenseF16MaxCols &&
         w.cols <= model_->max_half_cols();
}

bool Executor::Exl3DensePrefill(
    std::initializer_list<const DeviceTensor*> weights,
    std::initializer_list<float*> outputs, const float* x, std::uint32_t rows,
    std::string* error, const Exl3DenseWorkspace* workspace,
    hipStream_t launch_stream) const {
  const auto execution = launch_stream != nullptr ? launch_stream : stream_;
  auto* rotations = exl3_dense_rotations_;
  auto* gemms = exl3_dense_gemms_;
  try {
    if (weights.size() == 0 || weights.size() != outputs.size() ||
        weights.size() > 3)
      throw std::invalid_argument(
          "EXL3 dense prefill requires 1..3 matching outputs");
    struct Target {
      const DeviceTensor* weight;
      float* output;
    };
    std::array<Target, 3> targets{};
    const auto output_count = weights.size();
    const auto k = (*weights.begin())->cols;
    auto output = outputs.begin();
    bool sliced = false;
    for (std::size_t i = 0; const auto* w : weights) {
      if (w->exl3 == nullptr || w->experts != 1 || w->cols != k ||
          w->exl3->k != k || !gufo::hip::Exl3PrefillSupported(*w->exl3) ||
          std::uint64_t{w->output_begin} + w->rows > w->exl3->n)
        throw std::invalid_argument(
            "unsupported EXL3 dense prefill projection");
      sliced |= w->output_begin != 0 || w->rows != w->exl3->n;
      targets[i++] = {w, *output++};
    }
    if (sliced)
      for (const auto* w : weights)
        if (w->exl3 != targets[0].weight->exl3)
          throw std::invalid_argument(
              "EXL3 dense slices require a shared projection");
    // Output rotation is independent in each 128-channel Hadamard block.
    // An aligned singleton can project only its requested blocks, retaining
    // the parent's input rotation and packed trellis K-step stride.
    const bool direct_slice = sliced && output_count == 1 &&
                              targets[0].weight->output_begin % 128 == 0 &&
                              targets[0].weight->rows % 128 == 0;
    // Multiple views still share one full parent projection and are copied
    // before scratch is reused; an unaligned view also needs the full blocks.
    const auto count = sliced ? std::size_t{1} : output_count;
    if (rows == 0)
      return true;
    // q/k/v have different N; K/V and gate/up can share a GEMM launch.
    // All three still rotate together, each with its own checkpoint suh.
    std::sort(targets.begin(), targets.begin() + count, [](auto a, auto b) {
      return std::pair{a.weight->exl3->n, a.weight->exl3->rate.bits} <
             std::pair{b.weight->exl3->n, b.weight->exl3->rate.bits};
    });
    // Main-stream dense work reuses the larger routed/decode FP16 slab.
    // Concurrent shared experts supply their own operands and descriptors.
    // Tiny batches still retain the decode slab (24 * widest K).
    auto* rotated = exl3_prefill_.input;
    std::size_t capacity = std::size_t{exl3_prefill_.max_assignments} *
                           std::max(config().hidden_size, config().expert_ff);
    if (exl3_.projection.fast_input_elements > capacity) {
      rotated = exl3_.projection.fast_input;
      capacity = exl3_.projection.fast_input_elements;
    }
    if (workspace != nullptr) {
      rotated = workspace->input;
      capacity = workspace->elements;
      rotations = workspace->rotations;
      gemms = workspace->gemms;
    }
    const auto piece = static_cast<std::uint32_t>(
        std::min<std::size_t>(options_.max_batch, capacity / (count * k)));
    if (rotated == nullptr || piece == 0 || rotations == nullptr ||
        gemms == nullptr)
      throw std::runtime_error("EXL3 dense prefill exceeds rotation scratch");
    for (std::uint32_t first = 0; first < rows; first += piece) {
      const auto n = std::min(piece, rows - first);
      std::array<Exl3DensePrefillProblem, 3> problems{};
      for (std::size_t i = 0; i < count; ++i) {
        const auto& t = targets[i];
        const auto& l = *t.weight->exl3;
        if (sliced && !direct_slice &&
            exl3_.slice_elements < std::size_t{n} * l.n)
          throw std::runtime_error("EXL3 dense prefill slice exceeds scratch");
        const auto begin = direct_slice ? t.weight->output_begin : 0;
        problems[i] = {l.trellis + std::size_t{begin} * l.rate.bits,
                       l.suh,
                       l.svh + begin,
                       x + std::size_t{first} * k,
                       k,
                       rotated + i * std::size_t{n} * k,
                       sliced && !direct_slice
                           ? exl3_.slice
                           : t.output + std::size_t{first} * t.weight->rows,
                       n,
                       direct_slice ? l.n : 0};
      }
      PrepareExl3DensePrefill(std::span(problems.data(), count), rotations,
                              gemms, execution);
      gufo::hip::LaunchExl3RotateRowsGrouped(rotations, count, n, k, execution);
      for (std::size_t begin = 0; begin < count;) {
        const auto& l = *targets[begin].weight->exl3;
        auto end = begin + 1;
        while (end < count && targets[end].weight->exl3->n == l.n &&
               targets[end].weight->exl3->rate == l.rate)
          ++end;
        // Always X3, including a singleton/tail; a GEMV row threshold would
        // change the reduction tree when a prompt/checkpoint chunk moves.
        gufo::hip::LaunchExl3PrefillGemmGrouped(
            gemms + begin, end - begin, n, k,
            direct_slice ? targets[begin].weight->rows : l.n, l.rate, false,
            execution, direct_slice);
        if (diagnostics_active_)
          ++cycle_diagnostics_.prefill_dense_gemm_groups;
        begin = end;
      }
      if (sliced && !direct_slice) {
        for (std::size_t i = 0; i < output_count; ++i) {
          const auto& t = targets[i];
          if (!Check(hipMemcpy2DAsync(
                         t.output + std::size_t{first} * t.weight->rows,
                         t.weight->rows * sizeof(float),
                         exl3_.slice + t.weight->output_begin,
                         t.weight->exl3->n * sizeof(float),
                         t.weight->rows * sizeof(float), n,
                         hipMemcpyDeviceToDevice, execution),
                     "EXL3 dense prefill slice", error))
            return false;
        }
      }
      if (diagnostics_active_) {
        cycle_diagnostics_.prefill_dense_rows += std::uint64_t{n} * count;
        ++cycle_diagnostics_.prefill_dense_rotation_groups;
      }
    }
    return true;
  } catch (const std::exception& e) {
    AssignError(error, e.what());
    return false;
  }
}

Exl3InputSlabs Executor::Exl3Slabs(
    std::span<const DeviceTensor* const> consumers, std::uint32_t rows) const {
  if (prefill_phase || exl3_.rotated_input == nullptr || rows == 0 ||
      rows > gufo::hip::kExl3FastRotatedRows || consumers.empty() ||
      consumers.size() > gufo::hip::kExl3FastMaxGroup)
    return {};
  const auto* first = consumers.front()->exl3;
  Exl3InputSlabs slabs{};
  for (const auto* w : consumers) {
    const auto* l = w->exl3;
    if (l == nullptr || first == nullptr || w->experts != 1 ||
        w->output_begin != 0 || w->rows != l->n || w->cols != l->k ||
        !gufo::hip::Exl3FastSupported(*l) || l->k != first->k ||
        !(l->rate == first->rate))
      return {};
    slabs.suh[slabs.count++] = l->suh;
  }
  if (std::size_t{slabs.count} * gufo::hip::kExl3FastRotatedRows * first->k >
      exl3_.rotated_input_elements)
    return {};
  slabs.out = exl3_.rotated_input;
  slabs.pass_rows = gufo::hip::Exl3FastPassRows(rows);
  slabs.k = first->k;
  return slabs;
}

void Executor::NoteExl3Rotated(
    const Exl3InputSlabs& slabs, const float* x, std::uint32_t rows,
    std::span<const DeviceTensor* const> consumers) const {
  exl3_rotated_ = {};
  if (slabs.count == 0 || slabs.count != consumers.size())
    return;
  exl3_rotated_.x = x;
  exl3_rotated_.rows = rows;
  exl3_rotated_.count = slabs.count;
  for (std::size_t c = 0; c < consumers.size(); ++c)
    exl3_rotated_.consumers[c] = consumers[c]->exl3;
}

const std::uint16_t* Executor::TakeExl3Rotated(
    const float* x, std::uint32_t rows,
    std::span<const DeviceTensor* const> consumers) const {
  const auto& r = exl3_rotated_;
  if (r.x == nullptr || r.x != x || r.rows != rows ||
      r.count != consumers.size())
    return nullptr;
  for (std::size_t c = 0; c < consumers.size(); ++c)
    if (r.consumers[c] != consumers[c]->exl3)
      return nullptr;
  exl3_rotated_ = {};
  return exl3_.rotated_input;
}

bool Executor::Exl3Group(std::initializer_list<const DeviceTensor*> weights,
                         std::initializer_list<float*> outputs, const float* x,
                         std::uint32_t rows, std::string* error,
                         std::size_t shape_basis_n) const {
  if (prefill_phase)
    return Exl3DensePrefill(weights, outputs, x, rows, error);
  try {
    if (weights.size() != outputs.size() || weights.size() > 3)
      throw std::logic_error("invalid EXL3 projection group");
    std::array<gufo::hip::Exl3FastTarget, 3> targets{};
    auto output = outputs.begin();
    std::size_t count = 0;
    for (const auto* weight : weights) {
      if (weight->exl3 == nullptr || weight->output_begin != 0 ||
          weight->rows != weight->exl3->n)
        throw std::logic_error("EXL3 group requires complete projections");
      targets[count++] = {weight->exl3, *output++, weight->rows};
    }
    const auto& p = exl3_.projection;
    const gufo::hip::Exl3FastWorkspace ws{
        p.fast_input,      p.fast_input_elements,
        p.fast_counters,   p.fast_counter_count,
        p.fast_down_input, p.fast_down_input_elements};
    const std::span<const gufo::hip::Exl3FastTarget> group(targets.data(),
                                                           count);
    if (!prefill_phase && gufo::hip::Exl3FastGroupSupported(group, ws)) {
      // A producer epilogue may already have rotated x for this group.
      if (const auto* rotated = TakeExl3Rotated(
              x, rows, std::span(weights.begin(), weights.size()))) {
        const gufo::hip::Exl3FastWorkspace pre{
            const_cast<std::uint16_t*>(rotated),
            exl3_.rotated_input_elements,
            p.fast_counters,
            p.fast_counter_count,
            p.fast_down_input,
            p.fast_down_input_elements};
        gufo::hip::LaunchExl3FastProjections(group, x, (*weights.begin())->cols,
                                             rows, pre, stream_, true,
                                             shape_basis_n);
        return true;
      }
      for (std::uint32_t first = 0; first < rows;
           first += gufo::hip::kExl3FastMaxRows) {
        auto pass = targets;
        for (std::size_t i = 0; i < count; ++i)
          pass[i].y += std::size_t{first} * pass[i].y_stride;
        const auto n =
            std::min<std::uint32_t>(gufo::hip::kExl3FastMaxRows, rows - first);
        gufo::hip::LaunchExl3FastProjections(
            std::span(pass.data(), count),
            x + std::size_t{first} * (*weights.begin())->cols,
            (*weights.begin())->cols, n, ws, stream_, false, shape_basis_n);
      }
      return true;
    }
    output = outputs.begin();
    for (const auto* weight : weights)
      if (!Dense(*weight, x, *output++, rows, error))
        return false;
    return true;
  } catch (const std::exception& e) {
    AssignError(error, e.what());
    return false;
  }
}

bool Executor::PlainDensePrefill(const DeviceTensor& w, const float* x,
                                 float* out, std::uint32_t rows,
                                 std::string* error) const {
  if ((w.type != GgmlType::kF16 && w.type != GgmlType::kBF16 &&
       w.type != GgmlType::kF32) ||
      w.cols > half_input_cols_ || w.cols % 32 != 0) {
    AssignError(error, "unsupported plain half prefill geometry/type");
    return false;
  }
  const auto piece = std::max<std::uint32_t>(1, options_.max_batch);
  for (std::uint32_t first = 0; first < rows; first += piece) {
    const auto count = std::min(piece, rows - first);
    PrepareHalfInput(x + std::size_t{first} * w.cols, count, w.cols);
    if (!PlainPrefillGemm(w.data, SmallType(w.type),
                          static_cast<const __half*>(s_.x_half),
                          out + std::size_t{first} * w.rows, count, w.rows,
                          w.cols, stream_)) {
      AssignError(error, "plain half prefill GEMM failed");
      return false;
    }
  }
  if (diagnostics_active_)
    cycle_diagnostics_.prefill_plain_half_rows += rows;
  return true;
}

bool Executor::Dense(const DeviceTensor& w, const float* x, float* out,
                     std::uint32_t n_tokens, std::string* error_msg) const {
  if (prefill_phase && w.exl3 != nullptr)
    return Exl3DensePrefill({&w}, {out}, x, n_tokens, error_msg);
  if (diagnostics_active_ && prefill_phase && model_->output().exl3 != nullptr)
    cycle_diagnostics_.prefill_plain_rows += n_tokens;
  // Prefill plain operands use the accepted FP16 activation boundary
  // at every width. Decode/verify retain the established FP32-input path.
  // An EXL3 model's plain Q8_0 weights (its mixer copies) take the Q8_0
  // routes below.
  if (w.exl3 != nullptr ||
      (model_->output().exl3 != nullptr && w.type != GgmlType::kQ8_0)) {
    try {
      if (prefill_phase && w.exl3 == nullptr &&
          (w.type == GgmlType::kF16 || w.type == GgmlType::kBF16 ||
           w.type == GgmlType::kF32))
        return PlainDensePrefill(w, x, out, n_tokens, error_msg);
      const auto piece = std::max<std::uint32_t>(1, options_.max_batch);
      const DeviceTensor* const consumer[] = {&w};
      // A side branch (ForkSide) never takes the main stream's hand-off.
      const auto* rotated = w.exl3 != nullptr && !prefill_phase &&
                                    n_tokens <= piece && dense_exl3_ == nullptr
                                ? TakeExl3Rotated(x, n_tokens, consumer)
                                : nullptr;
      for (std::uint32_t first = 0; first < n_tokens; first += piece) {
        const auto count = std::min(piece, n_tokens - first);
        Exl3Projection(w, x + std::size_t{first} * w.cols,
                       out + std::size_t{first} * w.rows, count,
                       dense_exl3_ != nullptr ? *dense_exl3_ : exl3_, stream_,
                       !prefill_phase, rotated);
      }
      return true;
    } catch (const std::exception& error) {
      AssignError(error_msg, error.what());
      return false;
    }
  }
  if (w.type == GgmlType::kQ8_0) {
    if (!MatrixRows(n_tokens)) {
      Q8Input q;
      return Quantize(x, n_tokens, w.cols, &q, error_msg, true) &&
             Dense(w, q, out, error_msg);
    }
    // Wide batches: the F16 route for the shapes it wins (F16 rows in
    // s_.x_half, often left there by a producer), else activations
    // quantized per 32-wide block into the tiled layout and the int8 WMMA
    // GEMM. Both staging buffers hold max_batch rows; a wider call (the
    // draft block folds its streams into rows) runs in pieces.
    const bool f16 = DenseF16Route(w, n_tokens);
    const std::uint32_t piece = static_cast<std::uint32_t>(options_.max_batch);
    for (std::uint32_t r0 = 0; r0 < n_tokens; r0 += piece) {
      const std::uint32_t rows = std::min(piece, n_tokens - r0);
      // The staging buffer may already hold this input (a producer wrote
      // it, or the previous projection read the same rows).
      const float* src = x + static_cast<std::size_t>(r0) * w.cols;
      if (f16) {
        PrepareHalfInput(src, rows, w.cols);
        if (!DenseF16Gemm(w.data, static_cast<const __half*>(s_.x_half),
                          out + static_cast<std::size_t>(r0) * w.rows, rows,
                          w.rows, w.cols, stream_)) {
          AssignError(error_msg, "dense F16 GEMM failed");
          return false;
        }
        continue;
      }
      if (!(q8t_src_ == src && q8t_rows_ == rows && q8t_cols_ == w.cols)) {
        QuantizeQ8Tiled(src, s_.x_q8t, rows, w.cols, stream_);
        q8t_src_ = src;
        q8t_rows_ = rows;
        q8t_cols_ = w.cols;
      }
      if (!W8A8Gemm(w.data, s_.x_q8t,
                    out + static_cast<std::size_t>(r0) * w.rows, rows, w.rows,
                    w.cols, stream_)) {
        AssignError(error_msg, "W8A8 GEMM failed");
        return false;
      }
    }
    return true;
  }
  if (!MatrixRows(n_tokens)) {
    SmallGemm(w.data, SmallType(w.type), x, out, n_tokens, w.rows, w.cols,
              stream_);
    return true;
  }
  // Wide batches of the unquantized projections (router, alpha/beta,
  // indexer): out[t][m] = sum_k w[m][k] x[t][k].
  const int m = static_cast<int>(w.rows);
  const int k = static_cast<int>(w.cols);
  const int n = static_cast<int>(n_tokens);
  if (w.type == GgmlType::kF32) {
    const float alpha = 1.0F;
    const float beta = 0.0F;
    if (hipblasSgemm(blas_, HIPBLAS_OP_T, HIPBLAS_OP_N, m, n, k, &alpha,
                     static_cast<const float*>(w.data), k, x, k, &beta, out,
                     m) != HIPBLAS_STATUS_SUCCESS) {
      AssignError(error_msg, "hipBLAS GEMM failed");
      return false;
    }
    return true;
  }
  const bool bf16 = w.type == GgmlType::kBF16;
  const hipDataType type = bf16 ? HIP_R_16BF : HIP_R_16F;
  if (!(half_src_ == x && half_rows_ == n_tokens && half_cols_ == w.cols &&
        half_bf16_ == bf16)) {
    NarrowActivations(x, s_.x_half, bf16, static_cast<std::size_t>(n) * k,
                      stream_);
    half_src_ = x;
    half_rows_ = n_tokens;
    half_cols_ = w.cols;
    half_bf16_ = bf16;
  }
  return blaslt_->Gemm(w.data, s_.x_half, out, type, m, n, k, error_msg);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
