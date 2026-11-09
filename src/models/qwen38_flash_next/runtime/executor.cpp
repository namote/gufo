// Executor and Session lifetime: model upload, session creation and
// sizing, K/V row management, rollback depth, KVarN planning, the side
// stream and MTP cycle diagnostics, vision input and cancellation.

#include "src/models/qwen38_flash_next/runtime/executor_snapshot.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

// Keep prompt arithmetic independent of chunk width (see
// executor_internal.hpp).
constinit thread_local bool prefill_phase = false;

Session::~Session() {
  if (kv_pool_) {
    (void)hipStreamSynchronize(owner_->stream());
    ReleaseMappedKvBlocks();
    kv_reservation_.Reset();
  }
  try {
    PreserveSnapshots(0, 0);
  } catch (...) {
    // The backing device may be lost. Leave retained checkpoints readable as
    // failures instead of allowing a later reader to touch freed allocations.
    std::lock_guard lock(snapshots_->mutex);
    snapshots_->failure = std::current_exception();
  }
  TrimRollback(0);
  for (auto& [key, exec] : graphs_) {
    (void)hipGraphExecDestroy(exec);
  }
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

void Session::RestoreVisionLayout(const qwen::vision::RopeLayout& layout,
                                  hipStream_t stream) {
  layout.Validate(max_context_);
  const auto* previous = vision_input_.rope();
  vision_input_.RestoreLayout(layout, stream);
  for (auto& attention : attention_)
    attention.rope = vision_input_.rope();
  if (previous != vision_input_.rope()) {
    for (const auto& [key, graph] : graphs_)
      (void)hipGraphExecDestroy(graph);
    graphs_.clear();
    warmed_.clear();
  }
}

void Session::PrepareVision(hipStream_t stream) {
  vision_input_.PrepareRope(stream);
  for (auto& attention : attention_)
    attention.rope = vision_input_.rope();
}

void Session::ConfigureVision(
    std::shared_ptr<const qwen::vision::Prompt> prompt,
    std::shared_ptr<qwen::vision::Encoder> encoder, hipStream_t stream) {
  if (prompt)
    prompt->rope.Validate(max_context_);
  const auto* previous = vision_input_.rope();
  vision_input_.Configure(std::move(prompt), std::move(encoder), stream);
  for (auto& attention : attention_)
    attention.rope = vision_input_.rope();
  if (previous != vision_input_.rope()) {
    for (const auto& [key, graph] : graphs_)
      (void)hipGraphExecDestroy(graph);
    graphs_.clear();
    warmed_.clear();
  }
}

void Session::SetCancellationCheck(std::function<bool()> check) {
  // Installing a check starts a new record. Clearing it keeps whether the
  // request observed cancellation until Reset or the next installed check.
  last_request_cancelled_ = !check && (last_request_cancelled_ || cancelled_);
  cancelled_ = false;
  is_cancelled_ = std::move(check);
  vision_input_.SetCancellationCheck(is_cancelled_);
}

bool Session::CheckCancellation(std::string* error) const {
  try {
    cancelled_ = cancelled_ || (is_cancelled_ && is_cancelled_());
  } catch (...) {
    cancelled_ = true;
  }
  if (cancelled_) {
    AssignError(error, "generation cancelled");
    return false;
  }
  return true;
}

std::size_t Session::AllocatedBytes() const noexcept {
  return allocated_bytes_ + rollback_bytes_ + vision_input_.Bytes();
}

void Session::TrimRollback(std::uint32_t depth) noexcept {
  if (depth >= rollback_depth_)
    return;
  // Keep graphs whose rows still exist. Deeper verifier graphs capture
  // discarded pointers; ordinary decode and MTP graphs do not.
  for (auto it = graphs_.begin(); it != graphs_.end();) {
    if ((it->first & (std::uint64_t{1} << 32)) != 0 &&
        (it->first & (std::uint64_t{1} << 40)) == 0 &&
        (it->first & 0xFFFFU) > depth + 1) {
      (void)hipGraphExecDestroy(it->second);
      it = graphs_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto row = depth; row < rollback_depth_; ++row) {
    (void)hipFree(rollback_allocations_[row]);
    for (auto& layer : linear_) {
      layer.conv_snapshots.rows[row] = nullptr;
      layer.state_snapshots.rows[row] = nullptr;
    }
    ple_snapshots_.rows[row] = nullptr;
  }
  rollback_allocations_.resize(depth);
  rollback_depth_ = depth;
  rollback_bytes_ =
      owner_->SessionBytes(core::SessionMode::kAutoregressive, max_context_,
                           depth) -
      owner_->SessionBytes(core::SessionMode::kAutoregressive, max_context_, 0);
  ngram_snapshots_.resize(depth);
}

void Session::EnsureKvRows(std::uint32_t rows) {
  if (!kv_pool_)
    return;
  if (kv_unmapped_blocks_)
    throw std::logic_error("Flash-Next KV restore has unmapped blocks");
  if (rows > max_context_)
    throw std::length_error("Flash-Next KV rows exceed context");
  const auto need =
      rows ? hip::KvBlockGeometry::BlocksFor(std::size_t{rows} + kKvPooledRatio)
           : 0;
  if (need <= kv_blocks_.size())
    return;
  auto fresh = kv_pool_->Allocate(need - kv_blocks_.size(), &kv_reservation_);
  if (fresh.size() != need - kv_blocks_.size())
    throw std::length_error("Flash-Next shared KV pool exhausted");
  const auto first = kv_blocks_.size();
  kv_blocks_.insert(kv_blocks_.end(), fresh.begin(), fresh.end());
  std::copy(fresh.begin(), fresh.end(), kv_table_host_ + first);
  const auto result = hipMemcpyAsync(kv_table_ + first, kv_table_host_ + first,
                                     fresh.size() * sizeof(std::uint32_t),
                                     hipMemcpyHostToDevice, owner_->stream());
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}

void Session::TrimKvRows(std::uint32_t rows) {
  if (!kv_pool_)
    return;
  const auto keep =
      rows ? hip::KvBlockGeometry::BlocksFor(std::size_t{rows} + kKvPooledRatio)
           : 0;
  if (keep >= kv_blocks_.size())
    return;
  const auto result = hipStreamSynchronize(owner_->stream());
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
  ReleaseMappedKvBlocks(keep);
  kv_blocks_.resize(keep);
}

void Session::ReleaseMappedKvBlocks(std::size_t first) noexcept {
  const auto begin = kv_blocks_.begin() + static_cast<std::ptrdiff_t>(first);
  const auto end = std::remove(begin, kv_blocks_.end(), kUnmappedKvBlock);
  kv_unmapped_blocks_ -= std::distance(end, kv_blocks_.end());
  kv_pool_->Release(std::span(begin, end));
}

void Session::PrepareKvRestoreRows(std::uint32_t rows) {
  if (!kv_pool_)
    return;
  if (rows > max_context_)
    throw std::length_error("Flash-Next KV restore exceeds context");
  const auto need =
      rows ? hip::KvBlockGeometry::BlocksFor(std::size_t{rows} + kKvPooledRatio)
           : 0;
  kv_blocks_.reserve(need);
  TrimKvRows(rows);
  const auto previous = kv_blocks_.size();
  kv_blocks_.resize(need, kUnmappedKvBlock);
  kv_unmapped_blocks_ += need - previous;
}

void Session::EnsureKvRestoreBlock(std::uint32_t logical) {
  auto& physical = kv_blocks_.at(logical);
  if (physical != kUnmappedKvBlock)
    return;
  physical = kv_pool_->Allocate(1, &kv_reservation_).front();
  kv_table_host_[logical] = physical;
  --kv_unmapped_blocks_;
}

void Session::MapKvBlock(std::uint32_t logical, std::uint32_t physical) {
  auto& entry = kv_blocks_.at(logical);
  if (entry == physical)
    return;
  kv_pool_->Retain(std::span(&physical, 1));
  const auto replaced = std::exchange(entry, physical);
  kv_table_host_[logical] = physical;
  if (replaced == kUnmappedKvBlock)
    --kv_unmapped_blocks_;
  else
    kv_pool_->Release(std::span(&replaced, 1), kv_reservation_);
}

void Session::UploadKvTable() {
  if (!kv_pool_ || kv_blocks_.empty())
    return;
  if (kv_unmapped_blocks_)
    throw std::logic_error("Flash-Next KV restore has unmapped blocks");
  const auto result = hipMemcpyAsync(kv_table_, kv_table_host_,
                                     kv_blocks_.size() * sizeof(std::uint32_t),
                                     hipMemcpyHostToDevice, owner_->stream());
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}

void Session::UnshareKvRows(std::uint32_t first, std::uint32_t end) {
  if (!kv_pool_ || first >= end)
    return;
  const auto check = [](hipError_t result) {
    if (result != hipSuccess)
      throw std::runtime_error(hipGetErrorString(result));
  };
  const auto& geometry = kv_pool_->geometry();
  const auto last = std::min<std::size_t>(hip::KvBlockGeometry::BlocksFor(end),
                                          kv_blocks_.size());
  for (std::size_t logical = first >> hip::kKvBlockShift; logical < last;
       ++logical) {
    const auto shared = kv_blocks_[logical];
    if (shared == kUnmappedKvBlock)
      throw std::logic_error("cannot write unmapped Flash-Next KV rows");
    if (kv_pool_->references(shared) < 2)
      continue;
    // Drain graph/table users before changing the pinned table. Copy every
    // plane: target, predictor and pooled indexer share one physical ID.
    check(hipStreamSynchronize(owner_->stream()));
    const auto copy = kv_pool_->Allocate(1, &kv_reservation_).front();
    try {
      for (std::size_t plane = 0; plane < geometry.planes().size(); ++plane) {
        const auto pitch = geometry.layer_pitch(plane);
        const auto width = geometry.block_bytes(plane);
        auto* base = kv_pool_->plane_data(plane);
        check(hipMemcpy2DAsync(base + copy * width, pitch,
                               base + shared * width, pitch, width,
                               geometry.planes()[plane].layers,
                               hipMemcpyDeviceToDevice, owner_->stream()));
      }
      check(hipStreamSynchronize(owner_->stream()));
    } catch (...) {
      if (hipStreamSynchronize(owner_->stream()) == hipSuccess)
        kv_pool_->Release(std::span(&copy, 1), kv_reservation_);
      throw;
    }
    kv_blocks_[logical] = copy;
    kv_table_host_[logical] = copy;
    try {
      check(hipMemcpyAsync(kv_table_ + logical, kv_table_host_ + logical,
                           sizeof(std::uint32_t), hipMemcpyHostToDevice,
                           owner_->stream()));
      check(hipStreamSynchronize(owner_->stream()));
    } catch (...) {
      kv_pool_->Release(std::span(&shared, 1));
      throw;
    }
    kv_pool_->Release(std::span(&shared, 1));
  }
}

std::uint32_t Session::ReserveKvRows(std::uint32_t rows, std::uint32_t margin) {
  if (!kv_pool_)
    return 0;
  if (rows > max_context_)
    throw std::length_error("Flash-Next KV reservation exceeds context");
  const auto target = std::min(hip::KvBlockGeometry::BlocksFor(rows) + margin,
                               hip::KvBlockGeometry::BlocksFor(
                                   std::size_t{max_context_} + kKvPooledRatio));
  if (kv_unmapped_blocks_)
    throw std::logic_error("cannot reserve an incomplete Flash-Next restore");
  auto first = mtp_enabled_ ? std::min(position_, mtp_.position) : position_;
  if (kvarn_)
    first = std::min(first, kvarn_tail_.flushed_end);
  if (rows > owner_->config().indexer_top_k) {
    first = std::min(first, blocks_ * kKvPooledRatio);
    if (mtp_enabled_)
      first = std::min(first, mtp_.blocks * kKvPooledRatio);
  }
  std::uint32_t copies = 0;
  for (std::size_t i = first >> hip::kKvBlockShift;
       i < std::min<std::size_t>(target, kv_blocks_.size()); ++i)
    copies += kv_pool_->references(kv_blocks_[i]) > 1;
  const auto need =
      (target > kv_blocks_.size() ? target - kv_blocks_.size() : 0) + copies;
  kv_reservation_.Reset();
  if (need) {
    kv_reservation_ = kv_pool_->TryReserve(need);
    if (!kv_reservation_)
      throw std::length_error(
          "Flash-Next shared KV pool reservation exhausted");
  }
  return need;
}
std::uint32_t Session::ReleaseKvReservation() noexcept {
  const auto unused = kv_reservation_.blocks();
  kv_reservation_.Reset();
  return unused;
}

void Session::Reset(bool reuse_mtp) {
  if (kv_pool_) {
    const auto result = hipStreamSynchronize(owner_->stream());
    if (result != hipSuccess)
      throw std::runtime_error(hipGetErrorString(result));
    ReleaseMappedKvBlocks();
    kv_blocks_.clear();
    kv_reservation_.Reset();
  }

  // Reset leaves append-only rows intact. Preserve them at the first write
  // that would overwrite them, allowing cancellation to resume by rewind.
  // Scratch is overwritten by the next verification. Retaining its stable
  // addresses also retains the graphs; freeing it at every cache reassignment
  // makes later MTP requests allocate and record those shapes again. Its full
  // bound is already included in speculative SessionBytes and AllocatedBytes.
  if (!reuse_mtp || !mtp_enabled_)
    TrimRollback(0);
  last_request_cancelled_ = false;
  snapshot_lineage_.reset();
  position_ = 0;
  spec_tokens_ = 0;
  kvarn_tail_ = {};
  kvarn_pass_ = {};
  ngram_.Reset();
  mtp_.position = 0;
  mtp_.residual_valid = false;
  mtp_.blocks = 0;
  const Config& c = owner_->config();
  const auto zero = [&](void* pointer, std::size_t bytes) {
    const auto result = hipMemsetAsync(pointer, 0, bytes, owner_->stream());
    // Adapter invalidation may report success only after every zero fill was
    // submitted and completed. Keep legacy cleanup's error policy unchanged.
    if (result != hipSuccess && continuation_hooks_)
      throw std::runtime_error(hipGetErrorString(result));
  };
  for (auto& l : linear_) {
    if (l.state != nullptr) {
      zero(l.conv_state, static_cast<std::size_t>(c.ssm_conv_kernel - 1) *
                             c.SsmConvChannels() * sizeof(float));
      zero(l.state, static_cast<std::size_t>(c.ssm_num_v_heads) *
                        c.ssm_head_dim * c.ssm_head_dim * sizeof(float));
    }
  }
  blocks_ = 0;
  if (ple_history_ != nullptr) {
    zero(ple_history_, static_cast<std::size_t>(c.PleConvHistory()) *
                           c.HcDim() * sizeof(float));
  }
}

Executor::~Executor() {
  // Readers write pinned staging memory; drain them before freeing it,
  // including when a forward failed before reaching PLE.
  if (ple_pending_ || prefetch_.pending) {
    (void)ngram_->WaitRead();
  }
  (void)hipFree(batch_logits_);
  (void)hipFree(wide_controls_);
  for (float* p : wide_allocations_)
    (void)hipFree(p);
  (void)hipFree(verification_penalties_);
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
  host_allocations_.Reset();
  if (blas_ != nullptr) {
    (void)hipblasDestroy(blas_);
  }
  if (counts_ready_ != nullptr) {
    (void)hipEventDestroy(counts_ready_);
  }
  for (hipEvent_t event : {side_start_, side_done_}) {
    if (event != nullptr)
      (void)hipEventDestroy(event);
  }
  if (side_stream_ != nullptr)
    (void)hipStreamDestroy(side_stream_);
  if (stream_ != nullptr) {
    qfn_mmq_release_stream(stream_);
    (void)hipStreamDestroy(stream_);
  }
}

std::unique_ptr<Executor> Executor::Create(const DeviceModel& model,
                                           NgramTable* ngram, Options options,
                                           std::string* error_msg) {
  if (options.kvarn_bits != 4 && options.kvarn_bits != 6) {
    if (error_msg)
      *error_msg = "KVarN bits must be 4 or 6";
    return nullptr;
  }

  std::unique_ptr<Executor> e(new Executor());
  e->model_ = &model;
  e->ngram_ = ngram;
  e->options_ = options;
  e->options_.max_batch = std::max<std::uint32_t>(1, options.max_batch);
  e->options_.max_logit_rows = std::clamp<std::uint32_t>(
      options.max_logit_rows, 1, e->options_.max_batch);
  e->options_.max_speculative = std::clamp<std::uint32_t>(
      options.max_speculative, 1, e->options_.max_logit_rows);
  if (qfn_mmq_init(0) != 0) {
    AssignError(error_msg, "quantized GEMM tier initialization failed");
    return nullptr;
  }
  if (!Check(PrepareW8A8GemmWave64(), "wide Q8 GEMM preparation", error_msg))
    return nullptr;
  if (!Check(PrepareHcDecodeKernels(), "HC decode kernel preparation",
             error_msg))
    return nullptr;
  // The first greedy verification otherwise resolves the shared argmax
  // module on the decode path. Preparation launches no trial kernels.
  if (e->has_mtp() && !Check(gufo::hip::PrepareBatchedGPUArgmax(),
                             "MTP argmax preparation", error_msg))
    return nullptr;
  if (!Check(hipStreamCreate(&e->stream_), "hipStreamCreate", error_msg)) {
    return nullptr;
  }
  if (!Check(hipEventCreateWithFlags(&e->counts_ready_, hipEventDisableTiming),
             "expert counts event", error_msg) ||
      !Check(hipStreamCreate(&e->side_stream_), "side stream", error_msg) ||
      !Check(hipEventCreateWithFlags(&e->side_start_, hipEventDisableTiming),
             "side stream fork event", error_msg) ||
      !Check(hipEventCreateWithFlags(&e->side_done_, hipEventDisableTiming),
             "side stream join event", error_msg)) {
    return nullptr;
  }
  if (hipblasCreate(&e->blas_) != HIPBLAS_STATUS_SUCCESS ||
      hipblasSetStream(e->blas_, e->stream_) != HIPBLAS_STATUS_SUCCESS) {
    AssignError(error_msg, "hipblasCreate failed");
    return nullptr;
  }
  const Config& c = model.config();
  if (options.max_context > c.MaxContextLength()) {
    AssignError(error_msg, "executor context exceeds the model context");
    return nullptr;
  }
  e->context_capacity_ = std::max(c.context_length, options.max_context);
  e->yarn_ = MakeYarnRope(c.rope_scaling, c.rotary_dim, c.rope_theta);
  const std::size_t T = e->options_.max_batch;
  e->blaslt_ = BlasLt::Create(e->stream_, error_msg);
  if (e->blaslt_ == nullptr) {
    return nullptr;
  }
  if (MatrixRows(T)) {
    // The first indexer heuristic query also loads the library's code object.
    // Do that before readiness instead of leaving a ~100 ms GPU idle gap in
    // the first prefill. Exact geometry still determines every plan; no trial
    // GEMM runs and no model activations or session state are touched.
    for (const auto& layer : model.layers()) {
      for (const auto* tensor : {&layer.indexer_k, &layer.indexer_q}) {
        if (!tensor->empty() &&
            (tensor->type == GgmlType::kBF16 ||
             tensor->type == GgmlType::kF16) &&
            !e->blaslt_->Prepare(
                tensor->type == GgmlType::kBF16 ? HIP_R_16BF : HIP_R_16F,
                tensor->rows, T, tensor->cols, error_msg)) {
          return nullptr;
        }
      }
    }
  }
  const std::size_t hc_dim = c.HcDim();
  const std::size_t hidden = c.hidden_size;
  const std::size_t slots = T * c.num_experts_used;
  auto& a = e->allocations_;
  Scratch& s = e->s_;
  auto f32 = [&](std::size_t n) { return Alloc<float>(a, n, error_msg); };
  if (model.output().exl3 != nullptr) {
    auto& ex = e->exl3_;
    auto& projection = ex.projection;
    const auto width = std::max(
        {c.hidden_size, c.SsmValueDim(), c.AttentionQDim(), c.expert_ff});
    projection.rotated_elements = std::max(T, std::size_t{64}) * width;
    projection.rotated = f32(projection.rotated_elements);
    projection.weight_elements = std::size_t{128} * width;
    projection.weight = f32(projection.weight_elements);
    projection.fast_input_elements = std::size_t{3 * 8} * width;
    projection.fast_input =
        Alloc<std::uint16_t>(a, projection.fast_input_elements, error_msg);
    projection.fast_counter_count =
        std::max(c.vocab_size / 128, 3 * c.AttentionQDim() / 128);
    projection.fast_counters =
        Alloc<std::uint32_t>(a, projection.fast_counter_count, error_msg);
    projection.fast_down_input_elements =
        std::size_t{8} * std::max(c.expert_ff, c.shared_expert_ff);
    projection.fast_down_input =
        Alloc<std::uint16_t>(a, projection.fast_down_input_elements, error_msg);
    ex.rotated_input_elements = std::size_t{3 * 8} * width;
    ex.rotated_input =
        Alloc<std::uint16_t>(a, ex.rotated_input_elements, error_msg);
    ex.slice_elements = std::max(T, std::size_t{64}) * (c.indexer_heads + 1) *
                        c.indexer_head_dim;
    ex.slice = f32(ex.slice_elements);
    // The decode side branch's projections (ForkSide): kVecBatch rows of
    // one hidden-width consumer.
    auto& side = e->exl3_side_;
    auto& side_projection = side.projection;
    side_projection.fast_input_elements = std::size_t{8} * c.hidden_size;
    side_projection.fast_input =
        Alloc<std::uint16_t>(a, side_projection.fast_input_elements, error_msg);
    side_projection.fast_counter_count =
        std::max(3 * c.AttentionQDim(),
                 (c.indexer_heads + 1) * c.indexer_head_dim) /
        128;
    side_projection.fast_counters =
        Alloc<std::uint32_t>(a, side_projection.fast_counter_count, error_msg);
    side.slice_elements =
        std::size_t{kVecBatch} * (c.indexer_heads + 1) * c.indexer_head_dim;
    side.slice = f32(side.slice_elements);
    const auto assignments = std::size_t{64} * c.num_experts_used;
    ex.routed_input = Alloc<std::uint16_t>(
        a,
        assignments *
            std::max(std::size_t{2} * c.hidden_size, std::size_t{c.expert_ff}),
        error_msg);
    ex.tiles = Alloc<Exl3RouteTile>(a, assignments, error_msg);
    ex.tile_count = Alloc<std::uint32_t>(a, 1, error_msg);
    ex.route_arrivals = Alloc<std::uint32_t>(a, 1, error_msg);
    ex.down_exponents = Alloc<std::int32_t>(a, assignments, error_msg);
    e->exl3_dense_rotations_ =
        Alloc<gufo::hip::Exl3RotateProblem>(a, 3, error_msg);
    e->exl3_dense_gemms_ = Alloc<gufo::hip::Exl3GemmProblem>(a, 3, error_msg);
    if (e->options_.max_batch >= 512) {
      // Shared EXL3 projections may run beside routed prefill. Their rotated
      // operands and descriptors must not borrow either routed or x_half space.
      auto& shared = e->exl3_shared_workspace_;
      shared.elements = std::size_t{e->options_.max_batch} *
                        std::max(std::size_t{2} * c.hidden_size,
                                 std::size_t{c.shared_expert_ff});
      shared.input = Alloc<std::uint16_t>(a, shared.elements, error_msg);
      shared.rotations = Alloc<gufo::hip::Exl3RotateProblem>(a, 3, error_msg);
      shared.gemms = Alloc<gufo::hip::Exl3GemmProblem>(a, 3, error_msg);
    }
    ex.head_segments =
        Alloc<std::uint32_t>(a, kMtpDraftExtraRows + 1, error_msg);
    ex.head_values = f32(c.vocab_size + std::size_t{kMtpDraftExtraRows} * 128);
    Exl3PrefillLayout layout;
    if (!PlanExl3Prefill(c, e->options_.max_batch, &layout, error_msg))
      return nullptr;
    auto& pp = e->exl3_prefill_;
    pp.max_assignments = layout.assignments;
    pp.max_buckets = layout.buckets;
    const std::size_t A = layout.assignments;
    pp.offsets =
        Alloc<std::uint32_t>(a, std::size_t{c.num_experts} + 1, error_msg);
    pp.slots = Alloc<std::int32_t>(a, A, error_msg);
    pp.buckets = Alloc<Exl3ExpertBucket>(a, layout.buckets, error_msg);
    pp.bucket_count = Alloc<std::uint32_t>(a, 1, error_msg);
    pp.input = Alloc<std::uint16_t>(a, A * std::max(c.hidden_size, c.expert_ff),
                                    error_msg);
    // Gate/up borrow the current stage view at dispatch. Packed out must
    // remain separate from stage down_e, which receives the scattered rows.
    pp.out = f32(A * c.hidden_size);
    pp.down_exponents = Alloc<std::int32_t>(a, A, error_msg);
    pp.inverse = Alloc<std::int32_t>(a, A, error_msg);
    e->exl3_prefill_scratch_bytes_ = layout.bytes;
  }
  s.tokens = Alloc<std::int32_t>(a, T, error_msg);
  // F32 plain weights do not contribute to max_half_cols during upload.
  // EXL3 prefill nevertheless narrows their inputs, including F32 HC/PLE.
  e->half_input_cols_ =
      model.output().exl3 != nullptr
          ? std::max(
                {model.max_half_cols(), hc_dim, hidden,
                 std::size_t{c.PleEmbeddingDim()}, std::size_t{c.SsmValueDim()},
                 std::size_t{c.AttentionQDim()}, std::size_t{c.hc_low_rank},
                 std::size_t{c.expert_ff}, std::size_t{c.shared_expert_ff}})
          : model.max_half_cols();
  s.x_half = Alloc<std::uint16_t>(a, T * e->half_input_cols_, error_msg);
  if (model.output().exl3 != nullptr) {
    e->hc_down_q8_ = !model.layers().empty();
    for (const auto& layer : model.layers())
      e->hc_down_q8_ = e->hc_down_q8_ && !layer.hc_attn.down_q8.empty() &&
                       !layer.hc_ffn.down_q8.empty();
  }
  for (void*& slot : s.x_q8) {
    slot = Alloc<std::uint8_t>(
        a,
        qfn_mmq_q8_1_bytes(static_cast<int>(kVecBatch),
                           static_cast<int>(model.max_q8_cols())),
        error_msg);
  }
  const auto q8_1 = [&](std::size_t k) {
    return Alloc<std::uint8_t>(
        a, qfn_mmq_q8_1_bytes(static_cast<int>(kVecBatch), static_cast<int>(k)),
        error_msg);
  };
  s.xn_q8_1 = q8_1(hc_dim);
  s.lo_q8_1 = q8_1(c.hc_low_rank);
  s.mixed_q8_1 = q8_1(hidden);
  s.gdn_q8_1 = q8_1(c.SsmValueDim());
  s.ctx_q8_1 = q8_1(c.AttentionQDim());
  s.x_q8t =
      Alloc<std::uint8_t>(a, Q8TiledBytes(T, model.max_q8_cols()), error_msg);
  s.res = f32(T * hc_dim);
  s.xn = f32(T * hc_dim);
  s.xn_half = Alloc<__half>(a, T * hc_dim, error_msg);
  s.xn_q8t = Alloc<std::uint8_t>(a, Q8TiledBytes(T, hc_dim), error_msg);
  s.lo = f32(T * c.hc_low_rank);
  s.mixed = f32(T * hidden);
  s.inject = f32(T * c.hc_count * HcInjectParts(hidden));
  s.block_out = f32(T * hidden);
  s.hc_defer_w = f32(T * c.hc_count);
  s.xn_scale = f32(T * c.hc_count);
  e->mask_words_ = c.SparseMaskWords(e->context_capacity_);
  const std::size_t linear_projection = std::max<std::size_t>(
      c.SsmConvChannels() + c.SsmValueDim(), c.ple_layer >= 0 ? hc_dim : 0);
  const std::size_t linear_floats =
      T * (linear_projection + 2 * c.ssm_num_v_heads + 2 * c.SsmKeyDim() +
           2 * c.SsmValueDim()) +
      (T + c.ssm_conv_kernel) * c.SsmConvChannels();
  const std::size_t attention_floats =
      T * (5 * c.AttentionQDim() + 4 * c.AttentionKvDim() +
           c.indexer_heads * c.indexer_head_dim + c.indexer_head_dim +
           e->mask_words_);
  const std::size_t expert_floats = slots * (2 * c.expert_ff + hidden);
  // Linear/full attention and routed experts run sequentially on stream_.
  // Their intermediates die before the next stage, including batched rows,
  // MTP catch-up and the deferred MoE epilogue consumed by Combine. Keep
  // stable addresses for graph replay without reserving all three stages.
  float* const stage =
      f32(std::max({linear_floats, attention_floats, expert_floats,
                    T * (linear_projection + hc_dim)}));
  if (stage == nullptr) {
    return nullptr;
  }
  // Mixer gates die before attention/experts start. PLE also uses this
  // buffer as its key, so keep it beyond PLE's gated output at stage.
  s.hc_gate = stage + T * linear_projection;
  float* cursor = stage;
  const auto take = [&](std::size_t count) {
    float* result = cursor;
    cursor += count;
    return result;
  };
  // Stacked and separate SSM projections are mutually exclusive. MTP's
  // projected embedding is consumed before attention and reuses this space.
  s.qkvz = take(T * linear_projection);
  s.qkv = s.qkvz;
  s.z = s.qkvz != nullptr ? s.qkvz + T * c.SsmConvChannels() : nullptr;
  s.alpha_beta = take(T * 2 * c.ssm_num_v_heads);
  s.conv_scratch = take((T + c.ssm_conv_kernel) * c.SsmConvChannels());
  s.qn = take(T * c.SsmKeyDim());
  s.kn = take(T * c.SsmKeyDim());
  s.gdn_raw = take(T * c.SsmValueDim());
  s.gdn_out = take(T * c.SsmValueDim());
  cursor = stage;
  s.qg = take(T * (2 * c.AttentionQDim() + 2 * c.AttentionKvDim()));
  s.q = take(T * c.AttentionQDim());
  s.attn_gate = take(T * c.AttentionQDim());
  s.k = take(T * c.AttentionKvDim());
  s.v = take(T * c.AttentionKvDim());
  s.iq = take(T * c.indexer_heads * c.indexer_head_dim);
  s.ik = take(T * c.indexer_head_dim);
  s.mask = reinterpret_cast<std::uint32_t*>(take(T * e->mask_words_));
  s.ctx = take(T * c.AttentionQDim());
  // Keep score/selection traffic near the device cache size. Captured
  // verification still needs all its rows at the maximum context.
  const std::size_t score_stride = std::size_t{e->mask_words_} * 32;
  e->select_score_floats_ =
      std::max(kVecBatch * score_stride,
               std::min(std::size_t{5 * 1024 * 1024}, 512 * score_stride));
  s.scores = f32(e->select_score_floats_);
  s.attn_partials = f32(static_cast<std::size_t>(kVecBatch) * c.num_heads *
                        kAttnSplits * (c.head_dim + 2));
  s.attn_lists = Alloc<std::uint32_t>(
      a, std::size_t{kVecBatch} * (c.indexer_top_k / c.compress_ratio),
      error_msg);
  s.attn_counts = Alloc<std::uint32_t>(a, kVecBatch, error_msg);
  if (c.ple_layer >= 0) {
    // PLE finishes before this layer's mixer/SSM. Its gate consumes key
    // and query before normalization/convolution reuse those two buffers.
    // The gated input stays separate until PleInject consumes both outputs.
    s.ple_key = s.hc_gate;
    s.ple_value = s.block_out;
    s.ple_query = s.xn;
    s.ple_gated = s.qkvz;
    s.ple_norm = s.hc_gate;
    s.ple_conv = s.xn;
    s.ple_history_scratch =
        f32(static_cast<std::size_t>(c.PleConvHistory()) * hc_dim);
    // The quantizer reads each n-gram row once. On this UMA target it can
    // consume the coherent reader buffer directly, without another device
    // allocation or upload. Its address stays fixed for graph replay; the
    // next-chunk prefetch uses separate storage until this forward finishes.
    void* pinned = nullptr;
    if (!Check(e->host_allocations_.Allocate(
                   &pinned, T * c.PleEmbeddingDim() * sizeof(float),
                   hipHostMallocMapped | hipHostMallocCoherent),
               "pinned n-gram buffer", error_msg)) {
      return nullptr;
    }
    e->host_emb_ = static_cast<float*>(pinned);
    if (!Check(hipHostGetDevicePointer(reinterpret_cast<void**>(&s.ple_emb),
                                       pinned, 0),
               "mapped n-gram buffer", error_msg)) {
      return nullptr;
    }
    e->host_rows_.resize(T * c.ple_heads);
    if (ngram)
      ngram->ReserveGather(T * c.ple_heads);
  }
  if (options.kvarn) {
    if (c.head_dim != hip::kKvarnHeadDim) {
      AssignError(error_msg, "KVarN needs 256-wide attention heads");
      return nullptr;
    }
    const std::size_t staging =
        hip::KvarnStagingRows(T) * std::size_t{c.AttentionKvDim()};
    e->kvarn_staging_k_ = Alloc<__half>(a, staging, error_msg);
    e->kvarn_staging_v_ = Alloc<__half>(a, staging, error_msg);
    if (e->kvarn_staging_k_ == nullptr || e->kvarn_staging_v_ == nullptr)
      return nullptr;
  }
  s.router = f32(T * (c.num_experts + 1));
  s.ids = Alloc<std::int32_t>(a, slots, error_msg);
  s.expert_counts = Alloc<std::uint32_t>(a, c.num_experts, error_msg);
  {
    const std::size_t compact = RoutedCompactRows(slots, c.num_experts);
    s.routed_bounds = Alloc<std::int32_t>(a, c.num_experts + 1, error_msg);
    s.routed_cursors = Alloc<std::int32_t>(a, c.num_experts, error_msg);
    s.rows_token = Alloc<std::int32_t>(a, compact, error_msg);
    s.rows_slot = Alloc<std::int32_t>(a, compact, error_msg);
    s.routed_tiles =
        Alloc<std::int32_t>(a, 3 * RoutedTileCapacity(slots, c), error_msg);
  }
  s.weights = f32(slots);
  cursor = stage;
  s.gate_e = take(slots * c.expert_ff);
  s.up_e = take(slots * c.expert_ff);
  s.down_e = take(slots * hidden);
  s.shexp_gate = f32(T * c.shared_expert_ff);
  s.shexp_up = f32(T * c.shared_expert_ff);
  s.shexp_out = f32(T * hidden);
  s.shexp_half = Alloc<__half>(a, T * c.shared_expert_ff, error_msg);
  s.logits =
      f32(static_cast<std::size_t>(e->options_.max_logit_rows) * c.vocab_size);
  {
    void* control = nullptr;
    void* tokens = nullptr;
    void* logits = nullptr;
    if (!Check(
            e->host_allocations_.Allocate(&control, sizeof(Session::Control)),
            "pinned control buffer", error_msg) ||
        !Check(e->host_allocations_.Allocate(&tokens, T * sizeof(std::int32_t)),
               "pinned token buffer", error_msg) ||
        !Check(
            e->host_allocations_.Allocate(
                &logits, static_cast<std::size_t>(e->options_.max_logit_rows) *
                             c.vocab_size * sizeof(float)),
            "pinned logits buffer", error_msg)) {
      return nullptr;
    }
    void* counts = nullptr;
    if (!Check(e->host_allocations_.Allocate(
                   &counts, c.num_experts * sizeof(std::uint32_t)),
               "pinned expert counts", error_msg)) {
      return nullptr;
    }
    e->counts_host_ = static_cast<std::uint32_t*>(counts);
    void* tiles = nullptr;
    if (!Check(e->host_allocations_.Allocate(
                   &tiles,
                   3 * RoutedTileCapacity(slots, c) * sizeof(std::int32_t)),
               "pinned routed tile map", error_msg)) {
      return nullptr;
    }
    e->tiles_host_ = static_cast<std::int32_t*>(tiles);
    e->control_host_ = static_cast<Session::Control*>(control);
    e->tokens_host_ = static_cast<std::int32_t*>(tokens);
    e->logits_host_ = static_cast<float*>(logits);
  }
  // The wide mixer route (F16 norm for the epilogue, tiled Q8 norm for the
  // W8A8 down projection) needs the four-stream geometry, 32-wide blocks and
  // a Q8_0 down projection.
  e->wide_mixer_ = c.hc_count == 4 && c.hidden_size % 32 == 0 &&
                   !model.layers().empty() &&
                   model.layers()[0].hc_ffn.down.type == GgmlType::kQ8_0;
  if (model.has_mtp()) {
    // The trunk's kept rows are session-owned. Its transient residual and
    // mixer buffers are free while MTP constructs its input. The split
    // projections consume h/embd before the first mixer rewrites xn/mixed.
    s.mtp_h = s.xn;
    s.mtp_embd = s.mixed;
    s.mtp_eproj = s.qkvz;
    s.mtp_res = s.res;
    s.mtp_argmax = Alloc<ArgmaxCandidate>(a, kArgmaxParts, error_msg);
    s.mtp_token = Alloc<std::int32_t>(a, 1, error_msg);
    const auto candidate_ids = std::max<std::size_t>(
        MtpCandidateWorkspaceSize(c.vocab_size),
        e->options_.max_speculative * kArgmaxParts *
            sizeof(PenaltyArgmaxCandidate) / sizeof(std::uint32_t));
    s.mtp_ids = Alloc<std::uint32_t>(a, candidate_ids, error_msg);
    s.mtp_scratch_ids = Alloc<std::uint32_t>(a, candidate_ids, error_msg);
    // Final selection consumes intermediate IDs before writing its scores.
    // Pack those scores behind the 64 returned IDs for one host transfer.
    s.mtp_scores = reinterpret_cast<float*>(s.mtp_ids + kMtpCandidates);
    if (!Check(e->host_allocations_.Allocate(&e->mtp_token_host_,
                                             sizeof(std::int32_t)),
               "pinned draft token", error_msg) ||
        !Check(e->host_allocations_.Allocate(&e->mtp_candidates_host_,
                                             sizeof(MtpCandidateLogits)),
               "pinned draft candidates", error_msg)) {
      return nullptr;
    }
    std::construct_at(e->mtp_candidates_host_);
    e->mtp_candidates_host_->size =
        std::min<std::size_t>(c.vocab_size, kMtpCandidates);
    const auto rows = std::min<std::size_t>(7, e->options_.max_speculative - 1);
    if (rows != 0) {
      e->verification_ids_ = Alloc<std::uint32_t>(
          a, rows * MtpCandidateWorkspaceSize(c.vocab_size), error_msg);
      e->verification_scratch_ids_ = Alloc<std::uint32_t>(
          a, rows * MtpCandidateWorkspaceSize(c.vocab_size), error_msg);
      const auto candidates = rows * (kMtpCandidates + kCompactPenaltyLimit);
      e->verification_candidates_ =
          Alloc<sampling::LogitCandidate>(a, candidates, error_msg);
      if (!Check(e->host_allocations_.Allocate(
                     &e->verification_candidates_host_,
                     candidates * sizeof(sampling::LogitCandidate)),
                 "pinned verification candidates", error_msg) ||
          !Check(hip::AllocateDevice(&e->verification_penalties_,
                                     rows * kCompactPenaltyLimit *
                                         sizeof(sampling::TokenPenalty)),
                 "verification penalties", error_msg))
        return nullptr;
      e->verification_penalty_capacity_ = rows * kCompactPenaltyLimit;
      e->verification_penalty_host_.reserve(e->verification_penalty_capacity_);
    }
  }
  for (void* p : a) {
    if (p == nullptr) {
      return nullptr;
    }
  }
  return e;
}

hip::KvBlockGeometry Executor::KvGeometry(core::SessionMode mode,
                                          std::uint32_t context) const {
  return SessionKvPlanes(config(), mode == core::SessionMode::kSpeculative,
                         options_.kvarn, context, options_.kvarn_bits)
      .geometry;
}

std::unique_ptr<Session> Executor::CreateSession(
    core::SessionMode mode, std::uint32_t max_context, std::string* error_msg,
    std::shared_ptr<hip::KvBlockPool> pool) const {
  std::unique_ptr<Session> s(new Session());
  s->owner_ = this;
  s->mtp_enabled_ = mode == core::SessionMode::kSpeculative;
  if (s->mtp_enabled_ && !has_mtp()) {
    AssignError(error_msg, "MTP session requires a loaded predictor");
    return nullptr;
  }
  const Config& c = config();
  if (max_context == 0 || max_context > context_capacity_) {
    AssignError(error_msg, "session context exceeds the model context");
    return nullptr;
  }
  s->max_context_ = max_context;
  // Before sparse attention starts, pooling can lag by indexer_top_k
  // rows. Afterwards only an incomplete block precedes the current batch.
  // Completed block keys remain in block_k; their raw rows are dead.
  s->index_capacity_ = IndexerCapacity(c, options_.max_batch, max_context);
  s->linear_.resize(c.num_layers);
  s->attention_.resize(c.num_layers);
  auto& a = s->allocations_;
  if (c.compress_ratio != kKvPooledRatio) {
    AssignError(error_msg, "pooled indexer keys need a ratio of 4");
    return nullptr;
  }
  const auto layout = SessionKvPlanes(c, s->mtp_enabled_, options_.kvarn,
                                      max_context, options_.kvarn_bits);
  const auto& logical = layout.geometry;
  if (pool &&
      !std::ranges::equal(pool->geometry().planes(), logical.planes())) {
    AssignError(error_msg, "Flash-Next KV pool has incompatible planes");
    return nullptr;
  }
  s->kv_pool_ = std::move(pool);
  const auto& kv = s->kv_pool_ ? s->kv_pool_->geometry() : logical;
  auto* kv_rows = s->kv_pool_ ? static_cast<std::byte*>(s->kv_pool_->data())
                              : Alloc<std::byte>(a, kv.bytes(), error_msg,
                                                 &s->allocated_bytes_);
  auto* kv_table = Alloc<std::uint32_t>(a, logical.blocks(), error_msg,
                                        &s->allocated_bytes_);
  if (kv_rows == nullptr || kv_table == nullptr)
    return nullptr;
  if (s->kv_pool_) {
    s->kv_blocks_.reserve(logical.blocks());
    if (!Check(
            s->kv_table_host_owner_.Allocate(
                &s->kv_table_host_, logical.blocks() * sizeof(std::uint32_t)),
            "KV host block table", error_msg))
      return nullptr;
  } else {
    std::vector<std::uint32_t> identity(logical.blocks());
    for (std::uint32_t b = 0; b < logical.blocks(); ++b)
      identity[b] = b;
    if (!Check(hipMemcpy(kv_table, identity.data(),
                         identity.size() * sizeof(std::uint32_t),
                         hipMemcpyHostToDevice),
               "KV block table", error_msg))
      return nullptr;
  }
  const auto plane_layer = [&](std::size_t plane, std::uint32_t slot) {
    return kv_rows + kv.plane_offset(plane) + slot * kv.layer_pitch(plane);
  };
  s->kv_table_ = kv_table;
  const std::uint32_t attention_layers =
      c.num_layers / c.full_attention_interval;
  const std::size_t fp16_layer =
      std::size_t{hip::kKvarnFp16Tokens} * c.AttentionKvDim();
  if (options_.kvarn) {
    s->kvarn_ = true;
    s->kvarn_bits_ = options_.kvarn_bits;
    s->kvarn_fp16_ =
        Alloc<__half>(a, KvarnFp16Elements(c), error_msg, &s->allocated_bytes_);
    if (s->kvarn_fp16_ == nullptr)
      return nullptr;
    s->kvarn_records_ =
        reinterpret_cast<std::uint8_t*>(plane_layer(layout.records, 0));
    s->kvarn_records_pitch_ = kv.layer_pitch(layout.records);
  }
  std::uint32_t kv_slot = 0;
  const std::size_t conv_elems =
      static_cast<std::size_t>(c.ssm_conv_kernel - 1) * c.SsmConvChannels();
  const std::size_t state_elems = static_cast<std::size_t>(c.ssm_num_v_heads) *
                                  c.ssm_head_dim * c.ssm_head_dim;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    if (c.IsLinearLayer(il)) {
      auto& l = s->linear_[il];
      l.conv_state =
          Alloc<float>(a, conv_elems, error_msg, &s->allocated_bytes_);
      l.state = Alloc<float>(a, state_elems, error_msg, &s->allocated_bytes_);
    } else {
      auto& at = s->attention_[il];
      if (layout.fp16_trunk) {
        at.k_cache =
            reinterpret_cast<__half*>(plane_layer(layout.keys, kv_slot));
        at.v_cache =
            reinterpret_cast<__half*>(plane_layer(layout.values, kv_slot));
      } else {
        at.kvarn_k = s->kvarn_fp16_ + kv_slot * fp16_layer;
        at.kvarn_v = s->kvarn_fp16_ + (attention_layers + kv_slot) * fp16_layer;
        at.kvarn_records =
            s->kvarn_records_ + std::size_t{kv_slot} * s->kvarn_records_pitch_;
        at.kvarn_pass = &s->kvarn_pass_;
      }
      at.block_k =
          reinterpret_cast<__half*>(plane_layer(layout.pooled, kv_slot));
      at.kv_blocks = kv_table;
      ++kv_slot;
      at.index_k = Alloc<float>(
          a, static_cast<std::size_t>(s->index_capacity_) * c.indexer_head_dim,
          error_msg, &s->allocated_bytes_);
    }
  }
  if (c.ple_layer >= 0) {
    const std::size_t hist =
        static_cast<std::size_t>(c.PleConvHistory()) * c.HcDim();
    s->ple_history_ = Alloc<float>(a, hist, error_msg, &s->allocated_bytes_);
  }
  s->control_ = Alloc<Session::Control>(a, 1, error_msg, &s->allocated_bytes_);
  if (s->mtp_enabled_) {
    // Wide copy verification keeps every row for the draft catch-up.
    s->mtp_.target_hidden =
        Alloc<float>(a,
                     static_cast<std::size_t>(
                         std::max(options_.max_speculative, max_copy_rows())) *
                         c.HcDim(),
                     error_msg, &s->allocated_bytes_);
    // The draft block's FP16 K/V follow the trunk's FP16 layers, or are the
    // only FP16 layer of a KVarN session.
    const std::uint32_t draft_slot = layout.fp16_trunk ? kv_slot : 0;
    s->mtp_.k_cache =
        reinterpret_cast<__half*>(plane_layer(layout.keys, draft_slot));
    s->mtp_.v_cache =
        reinterpret_cast<__half*>(plane_layer(layout.values, draft_slot));
    s->mtp_.block_k =
        reinterpret_cast<__half*>(plane_layer(layout.pooled, kv_slot));
    s->mtp_.kv_blocks = kv_table;
    s->mtp_.index_k =
        Alloc<float>(a, std::size_t{s->index_capacity_} * c.indexer_head_dim,
                     error_msg, &s->allocated_bytes_);
    s->mtp_.h = Alloc<float>(a, c.HcDim(), error_msg, &s->allocated_bytes_);
    if (!model_->draft_output().empty()) {
      s->mtp_.draft_rows = Alloc<std::uint8_t>(
          a,
          std::size_t{kMtpDraftExtraRows} * model_->output_row_bytes() + 4096,
          error_msg, &s->allocated_bytes_);
      s->mtp_.draft_ids = Alloc<std::int32_t>(a, kMtpDraftExtraRows, error_msg,
                                              &s->allocated_bytes_);
    }
  }
  for (void* p : a) {
    if (p == nullptr) {
      return nullptr;
    }
  }
  // The zero fills above run on the null stream; nothing may read the new
  // buffers until they have landed.
  if (!Check(hipDeviceSynchronize(), "session init", error_msg)) {
    return nullptr;
  }
  // The table starts with the special tokens; every other slot is empty.
  if (s->mtp_.draft_rows != nullptr &&
      (!ClearDraftRows(*s, 0, error_msg) ||
       !AppendDraftRows(*s, 0, model_->draft_specials(), error_msg) ||
       !Check(hipStreamSynchronize(stream_), "draft table init", error_msg))) {
    return nullptr;
  }
  return s;
}

void Executor::FlushKvarnGroups(const Session& session, std::uint32_t first,
                                std::uint32_t groups) const {
  const Config& c = config();
  const std::uint32_t attention = c.num_layers / c.full_attention_interval;
  const std::size_t layer =
      std::size_t{hip::kKvarnFp16Tokens} * c.AttentionKvDim();
  KvarnQuantizeGroups(
      session.kvarn_fp16_, session.kvarn_fp16_ + std::size_t{attention} * layer,
      layer, session.kvarn_records_, session.kvarn_records_pitch_,
      session.kv_table_, c.num_kv_heads, attention, first, groups, stream_,
      kKvarnRingRows, options_.kvarn_bits);
}

bool Executor::PlanKvarn(Session& session, std::uint32_t start, std::uint32_t n,
                         std::string* error_msg) const {
  KvarnPass pass{.start = start, .rows = n};
  if (n <= hip::kKvarnMaxPassRows) {
    const auto plan = hip::PlanKvarnPass(session.kvarn_tail_, start, n);
    if (!plan.error.empty()) {
      AssignError(error_msg, std::string(plan.error));
      return false;
    }
    FlushKvarnGroups(session, plan.first_group, plan.groups);
    pass.flushed_end = plan.flushed_end;
  } else {
    const auto plan = hip::PlanKvarnPrefill(session.kvarn_tail_, start, n);
    if (!plan.error.empty()) {
      AssignError(error_msg, std::string(plan.error));
      return false;
    }
    FlushKvarnGroups(session, plan.first_group, plan.groups);
    pass.prefill = true;
    // A chunk inside the sink stages from position 0 and stores its sink
    // rows afterwards; otherwise the ring rows of its first group lead.
    pass.staging_first = start < hip::kKvarnSinkTokens ? 0 : plan.target;
    pass.quant_end = plan.target;
    pass.flushed_end = plan.flushed_end;
    if (start + n - pass.staging_first >
        hip::KvarnStagingRows(options_.max_batch)) {
      AssignError(error_msg, "KVarN prompt chunk exceeds its staging rows");
      return false;
    }
  }
  session.kvarn_pass_ = pass;
  return true;
}

bool Executor::FinishKvarn(Session& session, std::string* error_msg) const {
  const auto& pass = session.kvarn_pass_;
  const std::uint32_t end = pass.start + pass.rows;
  session.kvarn_tail_.flushed_end = pass.flushed_end;
  session.kvarn_tail_.written_end =
      std::max(session.kvarn_tail_.written_end, end);
  // A pass ending exactly at a group end quantizes that group now, so the
  // records of every committed state reach its group floor (snapshots).
  if (!pass.prefill && end % hip::kKvarnGroupTokens == 0 &&
      end > hip::kKvarnSinkTokens && pass.flushed_end < end) {
    FlushKvarnGroups(session, (end >> hip::kKvarnGroupShift) - 1, 1);
    session.kvarn_tail_.flushed_end = end;
    // The forward already synchronized; a snapshot capture right after it
    // copies these records on its own stream, so they must be complete.
    return Check(hipStreamSynchronize(stream_), "KVarN flush", error_msg);
  }
  return Check(hipGetLastError(), "KVarN flush", error_msg);
}

bool Executor::EnsureRollback(Session& session, std::uint32_t depth,
                              std::string* error_msg) const {
  if (depth <= session.rollback_depth_)
    return true;
  if (depth >= options_.max_speculative ||
      depth > std::size(session.ple_snapshots_.rows) ||
      session.spec_tokens_ != 0) {
    AssignError(error_msg, "invalid rollback allocation depth");
    return false;
  }
  const auto& c = config();
  const std::size_t conv =
      std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const std::size_t linear =
      c.num_layers - c.num_layers / c.full_attention_interval;
  const std::size_t ple =
      c.ple_layer >= 0 ? std::size_t{c.PleConvHistory()} * c.HcDim() : 0;
  session.rollback_allocations_.reserve(depth);
  session.ngram_snapshots_.resize(depth);
  for (auto row = session.rollback_depth_; row < depth; ++row) {
    const auto state = GdnRollbackRowFloats(row, c.ssm_num_k_heads,
                                            c.ssm_num_v_heads, c.ssm_head_dim);
    const auto bytes = (linear * (conv + state) + ple) * sizeof(float);
    float* allocation = nullptr;
    if (!Check(hip::AllocateDevice(&allocation, bytes), "rollback allocation",
               error_msg))
      return false;
    session.rollback_allocations_.push_back(allocation);
    session.rollback_bytes_ += bytes;
    auto* next = allocation;
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      if (!c.IsLinearLayer(il))
        continue;
      session.linear_[il].conv_snapshots.rows[row] = next;
      next += conv;
      session.linear_[il].state_snapshots.rows[row] = next;
      next += state;
    }
    if (ple != 0) {
      session.ple_snapshots_.rows[row] = next;
    }
    session.rollback_depth_ = row + 1;
  }
  return true;
}

std::size_t Executor::SessionBytes(
    core::SessionMode mode, std::uint32_t max_context,
    std::uint32_t rollback_depth) const noexcept {
  const auto& c = config();
  const std::size_t attention = c.num_layers / c.full_attention_interval;
  const std::size_t linear = c.num_layers - attention;
  const std::size_t conv =
      std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const std::size_t state =
      std::size_t{c.ssm_num_v_heads} * c.ssm_head_dim * c.ssm_head_dim;
  const std::size_t ple =
      c.ple_layer >= 0 ? std::size_t{c.PleConvHistory()} * c.HcDim() : 0;
  // K/V/pooled rows and the block table (CreateSession); the raw indexer
  // rings stay per layer.
  const auto kv_layout =
      SessionKvPlanes(c, mode == core::SessionMode::kSpeculative,
                      options_.kvarn, max_context, options_.kvarn_bits);
  const std::size_t kv_blocks =
      kv_layout.geometry.bytes() +
      std::size_t{kv_layout.geometry.blocks()} * sizeof(std::uint32_t) +
      (options_.kvarn ? KvarnFp16Elements(c) * sizeof(__half) : 0);
  const std::size_t index =
      std::size_t{IndexerCapacity(c, options_.max_batch, max_context)} *
      c.indexer_head_dim * sizeof(float);
  const auto rollback_state =
      rollback_depth == 0
          ? 0
          : GdnRollbackRowFloats(0, c.ssm_num_k_heads, c.ssm_num_v_heads,
                                 c.ssm_head_dim) +
                (rollback_depth - 1) *
                    GdnRollbackRowFloats(1, c.ssm_num_k_heads,
                                         c.ssm_num_v_heads, c.ssm_head_dim);
  // MTP keeps every verified trunk hidden row (target_hidden: the verifier
  // width, or a wide copy's) plus the draft block's own row (h).
  const std::size_t hidden_rows =
      std::size_t{std::max(options_.max_speculative, max_copy_rows())} + 1;
  return ((linear * conv + ple) * (rollback_depth + 1) +
          linear * (state + rollback_state)) *
             sizeof(float) +
         kv_blocks + attention * index + sizeof(Session::Control) +
         (mode == core::SessionMode::kSpeculative
              ? index + hidden_rows * c.HcDim() * sizeof(float) +
                    (model_->draft_output().empty()
                         ? 0
                         : std::size_t{kMtpDraftExtraRows} *
                                   (model_->output_row_bytes() +
                                    sizeof(std::int32_t)) +
                               4096)
              : 0);
}

std::size_t Executor::DeferredScratchBytes() const {
  const auto& c = config();
  std::size_t bytes = batch_logits_ == nullptr
                          ? std::size_t{8} *
                                std::min(8U, options_.max_logit_rows) *
                                c.vocab_size * sizeof(float)
                          : 0;
  const auto linear = c.num_layers - c.num_layers / c.full_attention_interval;
  const auto conv = std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const auto state =
      std::size_t{c.ssm_num_v_heads} * c.ssm_head_dim * c.ssm_head_dim;
  const auto ple =
      c.ple_layer >= 0 ? std::size_t{c.PleConvHistory()} * c.HcDim() : 0;
  const auto mutable_bytes = (linear * (conv + state) + ple) * sizeof(float);
  if (!checkpoint_storage_) {
    bytes +=
        mutable_bytes + (has_mtp() ? std::size_t{options_.max_speculative} *
                                         c.HcDim() * sizeof(float)
                                   : 0);
  }
  // Rollback rows and the starting states of a wide copy verification's
  // later segments (ForwardWide), allocated on first use.
  const std::size_t wide_segments =
      (max_copy_rows() + kWideSegmentRows - 1) / kWideSegmentRows;
  if (wide_segments > 1 + wide_segments_.size())
    bytes += (wide_segments - 1 - wide_segments_.size()) * WideSegmentFloats() *
             sizeof(float);
  if (wide_segments > 1 && wide_state_.empty())
    bytes += WideStateFloats() * sizeof(float);
  // Committed spare blocks for captures, whether or not they exist yet.
  return bytes + SnapshotAllocator::SpareBytes(mutable_bytes);
}

std::size_t Executor::WideStateFloats() const {
  const auto& c = config();
  const std::size_t linear =
      c.num_layers - c.num_layers / c.full_attention_interval;
  return linear * c.ssm_num_v_heads * c.ssm_head_dim * c.ssm_head_dim;
}

std::size_t Executor::WideSegmentFloats() const {
  const auto& c = config();
  const std::size_t conv =
      std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const std::size_t ple =
      c.ple_layer >= 0 ? std::size_t{c.PleConvHistory()} * c.HcDim() : 0;
  constexpr std::uint32_t kRows = kWideSegmentRows - 1;
  std::size_t floats = ple * (kRows + 1);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    if (!c.IsLinearLayer(il))
      continue;
    floats += conv * (kRows + 1);
    for (std::uint32_t row = 0; row < kRows; ++row)
      floats += GdnRollbackRowFloats(row, c.ssm_num_k_heads, c.ssm_num_v_heads,
                                     c.ssm_head_dim);
  }
  return floats;
}

bool Executor::EnsureWideSegments(std::uint32_t segments,
                                  std::string* error_msg) const {
  if (segments == 0)
    return true;
  const auto& c = config();
  if (wide_controls_ == nullptr &&
      !Check(hip::AllocateDevice(&wide_controls_,
                                 std::size_t{kWideRows / kWideSegmentRows} *
                                     sizeof(Session::Control)),
             "wide verification controls", error_msg))
    return false;
  if (wide_state_.empty()) {
    float* allocation = nullptr;
    if (!Check(
            hip::AllocateDevice(&allocation, WideStateFloats() * sizeof(float)),
            "wide verification state", error_msg))
      return false;
    wide_allocations_.push_back(allocation);
    wide_state_.assign(c.num_layers, nullptr);
    const std::size_t state =
        std::size_t{c.ssm_num_v_heads} * c.ssm_head_dim * c.ssm_head_dim;
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      if (!c.IsLinearLayer(il))
        continue;
      wide_state_[il] = allocation;
      allocation += state;
    }
  }
  const std::size_t conv =
      std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const std::size_t ple =
      c.ple_layer >= 0 ? std::size_t{c.PleConvHistory()} * c.HcDim() : 0;
  constexpr std::uint32_t kRows = kWideSegmentRows - 1;
  static_assert(kRows == std::size(RollbackRows{}.rows));
  while (wide_segments_.size() < segments) {
    float* allocation = nullptr;
    if (!Check(hip::AllocateDevice(&allocation,
                                   WideSegmentFloats() * sizeof(float)),
               "wide rollback allocation", error_msg))
      return false;
    wide_allocations_.push_back(allocation);
    WideSegment seg;
    seg.conv.resize(c.num_layers);
    seg.state.resize(c.num_layers);
    seg.conv_start.resize(c.num_layers, nullptr);
    auto* next = allocation;
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      if (!c.IsLinearLayer(il))
        continue;
      seg.conv_start[il] = next;
      next += conv;
      for (std::uint32_t row = 0; row < kRows; ++row) {
        seg.conv[il].rows[row] = next;
        next += conv;
        seg.state[il].rows[row] = next;
        next += GdnRollbackRowFloats(row, c.ssm_num_k_heads, c.ssm_num_v_heads,
                                     c.ssm_head_dim);
      }
    }
    if (ple != 0) {
      seg.ple_start = next;
      next += ple;
      for (std::uint32_t row = 0; row < kRows; ++row) {
        seg.ple.rows[row] = next;
        next += ple;
      }
    }
    wide_segments_.push_back(std::move(seg));
  }
  return true;
}

bool Executor::SideCapturing() const {
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  return hipStreamIsCapturing(side_stream_, &status) == hipSuccess &&
         status != hipStreamCaptureStatusNone;
}

bool Executor::SideBranchAvailable() const noexcept {
  return !prefill_phase && side_stream_ != nullptr &&
         exl3_side_.projection.fast_counters != nullptr &&
         exl3_side_.slice != nullptr;
}

bool Executor::ForkSide(const std::function<bool()>& enqueue,
                        std::string* error_msg) const {
  if (!SideBranchAvailable() ||
      !Check(hipEventRecord(side_start_, stream_), "side branch fork",
             error_msg) ||
      !Check(hipStreamWaitEvent(side_stream_, side_start_, 0),
             "side branch wait", error_msg))
    return false;
  const hipStream_t main = stream_;
  // However the branch exits, later work returns to the main stream and
  // the side launches queued so far are joined (JoinSide).
  struct Restore {
    const Executor* executor;
    hipStream_t main;
    bool joined = false;
    bool Join() {
      executor->stream_ = main;
      executor->dense_exl3_ = nullptr;
      if (joined)
        return true;
      joined = true;
      if (hipEventRecord(executor->side_done_, executor->side_stream_) ==
          hipSuccess) {
        executor->side_pending_ = true;
        return true;
      }
      // Nothing to join: wait for the whole side stream instead (not while
      // a graph captures it).
      if (!executor->SideCapturing())
        (void)hipStreamSynchronize(executor->side_stream_);
      return false;
    }
    ~Restore() { (void)Join(); }
  } restore{this, main};
  stream_ = side_stream_;
  dense_exl3_ = &exl3_side_;
  ++side_branches_;
  const bool ok = enqueue();
  if (!restore.Join()) {
    AssignError(error_msg, "side branch join record failed");
    return false;
  }
  return ok;
}

bool Executor::JoinSide(std::string* error_msg) const {
  if (!side_pending_)
    return true;
  const auto status = hipStreamWaitEvent(stream_, side_done_, 0);
  if (status != hipSuccess && !SideCapturing())
    (void)hipStreamSynchronize(side_stream_);
  side_pending_ = false;
  return Check(status, "side stream join wait", error_msg);
}

Exl3PrefillScratch Executor::Exl3PrefillView() const {
  auto view = exl3_prefill_;
  // Attention intermediates are dead. Resolve these pointers after any
  // RowScratch narrowing, rather than retaining the arena's base addresses.
  view.gate = s_.gate_e;
  view.up = s_.up_e;
  return view;
}

std::size_t Executor::EstimateExl3PrefillScratchBytes(const Config& config,
                                                      std::uint32_t rows,
                                                      std::string* error_msg) {
  Exl3PrefillLayout layout;
  return PlanExl3Prefill(config, rows, &layout, error_msg) ? layout.bytes : 0;
}

std::uint32_t Executor::PrefillChunkRows(
    std::uint32_t requested, std::uint32_t until_checkpoint) const noexcept {
  auto limit = options_.max_batch;
  if (model_->output().exl3 != nullptr && options_.exl3_prefill_chunk_rows != 0)
    limit = std::min(limit, options_.exl3_prefill_chunk_rows);
  return std::min({requested, until_checkpoint, limit});
}

bool Executor::BeginMtpCycleDiagnostics(std::string* error_msg) const {
  if (diagnostics_active_ || model_->output().exl3 == nullptr) {
    AssignError(error_msg, "MTP cycle diagnostics require idle EXL3 executor");
    return false;
  }
  if (!Check(hipStreamSynchronize(stream_), "diagnostic cycle start",
             error_msg))
    return false;
  cycle_diagnostics_ = {};
  diagnostic_depth_ = 0;
  diagnostic_phase_ = nullptr;
  diagnostic_started_ = std::chrono::steady_clock::now();
  diagnostics_active_ = true;
  return true;
}

bool Executor::EndMtpCycleDiagnostics(std::uint64_t committed_tokens,
                                      std::uint64_t drafts,
                                      MtpCycleDiagnostics* output,
                                      std::string* error_msg) const {
  if (!diagnostics_active_ || diagnostic_depth_ != 0 || output == nullptr) {
    AssignError(error_msg, "invalid MTP diagnostic cycle end");
    return false;
  }
  // Always release the diagnostic scope, including a failed GPU cycle.
  diagnostics_active_ = false;
  if (!Check(hipStreamSynchronize(stream_), "diagnostic cycle end", error_msg))
    return false;
  cycle_diagnostics_.committed_tokens = committed_tokens;
  cycle_diagnostics_.drafts = drafts;
  cycle_diagnostics_.wall_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - diagnostic_started_)
          .count();
  *output = cycle_diagnostics_;
  return true;
}

Executor::DiagnosticPhase::DiagnosticPhase(const Executor& e, DiagnosticStage s)
    : executor(e), stage(s), active(e.diagnostics_active_) {
  if (active) {
    ++executor.diagnostic_depth_;
    parent = executor.diagnostic_phase_;
    executor.diagnostic_phase_ = this;
    started = std::chrono::steady_clock::now();
  }
}

Executor::DiagnosticPhase::~DiagnosticPhase() {
  if (!active)
    return;
  --executor.diagnostic_depth_;
  executor.diagnostic_phase_ = parent;
  const auto elapsed = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - started)
                           .count();
  if (parent != nullptr)
    parent->children_ms += elapsed;
  const auto ms = elapsed - children_ms;
  auto& d = executor.cycle_diagnostics_;
  switch (stage) {
    case DiagnosticStage::kTarget:
      d.target_ms += ms;
      break;
    case DiagnosticStage::kDraft:
      d.draft_ms += ms;
      break;
    case DiagnosticStage::kHead:
      d.head_ms += ms;
      break;
    case DiagnosticStage::kCatchUp:
      d.catch_up_ms += ms;
      break;
  }
}

bool Executor::DiagnosticCount(const std::uint32_t* device,
                               std::uint64_t* total,
                               std::string* error_msg) const {
  std::uint32_t count = 0;
  if (!Check(hipMemcpyAsync(&count, device, sizeof(count),
                            hipMemcpyDeviceToHost, stream_),
             "diagnostic count", error_msg) ||
      !Check(hipStreamSynchronize(stream_), "diagnostic count ready",
             error_msg))
    return false;
  *total += count;
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
