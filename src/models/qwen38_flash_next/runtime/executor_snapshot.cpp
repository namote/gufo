// Session snapshots: the pinned snapshot allocator and registry,
// prefill checkpoints, and the saved-session format (save and restore).

#include "src/models/qwen38_flash_next/runtime/executor_snapshot.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

namespace {

void PopulateSnapshotPrefix(std::uint8_t* data, std::uint64_t bytes,
                            std::uint64_t worker_limit = 4) {
  platform::PopulateWritablePages(data, static_cast<std::size_t>(bytes),
                                  static_cast<std::size_t>(worker_limit));
}

}  // namespace

std::shared_ptr<SnapshotAllocator> Executor::GetSnapshotAllocator() const {
  std::call_once(snapshot_allocator_once_, [&] {
    // Admission reserved the lazy scratch; snapshots must not consume it.
    snapshot_allocator_ = std::make_shared<SnapshotAllocator>(
        DeferredScratchBytes() + kSnapshotDeviceSlackBytes);
  });
  return snapshot_allocator_;
}

void SnapshotState::Preserve(std::uint32_t position, std::uint32_t mtp_position,
                             std::uint32_t blocks, std::uint32_t mtp_blocks,
                             bool detach) const {
  std::unique_ptr<gufo::hip::SnapshotTransfer> transfer;
  for (auto& region : regions_) {
    auto rows = region.draft ? mtp_position : position;
    if (region.pooled)
      rows = region.draft ? mtp_blocks : blocks;
    const auto keep = std::uint64_t{rows} * region.row_bytes;
    if (keep < region.pending_bytes) {
      if (!transfer)
        transfer = registry_->allocator->Transfer();
      transfer->Copy(region.host + keep,
                     static_cast<const std::uint8_t*>(region.device) + keep,
                     region.pending_bytes - keep);
      region.pending_bytes = keep;
    }
    if (detach)
      region.live_bytes = std::min(region.live_bytes, keep);
  }
}

void SnapshotState::Materialize() const {
  std::lock_guard lock(registry_->mutex);
  if (registry_->failure)
    std::rethrow_exception(registry_->failure);
  MaterializeLocked();
}

void SnapshotState::MaterializeLocked() const {
  const auto transfer = registry_->allocator->Transfer();
  for (const auto& region : regions_) {
    PopulateSnapshotPrefix(region.host, region.bytes);
    for (const auto& saved : region.saved)
      transfer->Enqueue(region.host + saved.offset, saved.chunk->device,
                        saved.bytes);
  }
  transfer->Finish();
  Preserve(0, 0, 0, 0, false);
  if (mutable_storage_) {
    PopulateSnapshotPrefix(payload_ + mutable_offset_, mutable_bytes_);
    transfer->Copy(payload_ + mutable_offset_, mutable_device_, mutable_bytes_);
    mutable_storage_.reset();
  }
  if (extra_storage_) {
    for (const auto& extra : extras_)
      transfer->Enqueue(payload_ + extra.offset, extra.device, extra.bytes);
    transfer->Finish();
    extras_.clear();
    extra_storage_.reset();
  }
  for (auto& region : regions_)
    region.saved.clear();
}

std::uint64_t SnapshotState::DeviceBytes() const {
  std::lock_guard lock(registry_->mutex);
  std::uint64_t bytes = (mutable_storage_ ? mutable_bytes_ : 0) +
                        (extra_storage_ ? extra_bytes_ : 0);
  std::vector<const SnapshotBlock*> blocks;
  for (const auto& region : regions_)
    for (const auto& saved : region.saved) {
      const auto* block = saved.chunk->block.get();
      if (std::find(blocks.begin(), blocks.end(), block) == blocks.end()) {
        blocks.push_back(block);
        bytes += block->capacity;
      }
    }
  return bytes;
}

void SnapshotRegistry::Preserve(std::uint32_t position,
                                std::uint32_t mtp_position,
                                std::uint32_t blocks,
                                std::uint32_t mtp_blocks) {
  struct Group {
    void* device;
    std::uint64_t keep;
    std::uint64_t end;
    std::vector<std::pair<SnapshotState::Region*, std::size_t>> regions;
  };
  std::vector<std::shared_ptr<SnapshotState>> snapshots;
  std::vector<Group> groups;
  std::vector<std::shared_ptr<SnapshotBlock>> available;
  // Only overwritten rows need backing storage. Appending borrows K/V in
  // place and does not allocate another context's worth of device pages.
  for (auto it = reserve.begin(); it != reserve.end();) {
    if (auto block = it->lock()) {
      if (block->chunks == 0)
        block->used = 0;
      if (block->used < block->capacity &&
          block->capacity - block->used <= kSnapshotReuseBytes)
        available.push_back(std::move(block));
      ++it;
    } else {
      it = reserve.erase(it);
    }
  }
  for (auto it = pending.begin(); it != pending.end();) {
    if (auto snapshot = it->lock()) {
      for (auto& region : snapshot->regions_) {
        auto rows = region.draft ? mtp_position : position;
        if (region.pooled)
          rows = region.draft ? mtp_blocks : blocks;
        const auto keep = std::uint64_t{rows} * region.row_bytes;
        if (keep < region.pending_bytes) {
          auto group = std::find_if(
              groups.begin(), groups.end(),
              [&](const auto& entry) { return entry.device == region.device; });
          if (group == groups.end()) {
            groups.push_back({region.device,
                              keep,
                              region.pending_bytes,
                              {{&region, snapshots.size()}}});
          } else {
            group->end = std::max(group->end, region.pending_bytes);
            group->regions.emplace_back(&region, snapshots.size());
          }
        }
      }
      snapshots.push_back(std::move(snapshot));
      ++it;
    } else {
      it = pending.erase(it);
    }
  }
  struct Copy {
    void* device;
    std::uint64_t begin;
    std::uint64_t end;
    std::vector<SnapshotState::Region*> regions;
    std::vector<std::size_t> owners;
  };
  std::vector<Copy> copies;
  for (const auto& group : groups) {
    std::vector<std::uint64_t> ends{group.keep};
    for (const auto& [region, owner] : group.regions) {
      (void)owner;
      ends.push_back(region->pending_bytes);
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    for (std::size_t index = 1; index < ends.size(); ++index) {
      Copy copy{group.device, ends[index - 1], ends[index], {}, {}};
      for (const auto& [region, owner] : group.regions)
        if (region->pending_bytes >= copy.end) {
          copy.regions.push_back(region);
          copy.owners.push_back(owner);
        }
      copies.push_back(std::move(copy));
    }
  }
  // Pack rows with the same consumers together across layers. Each consumer
  // set gets one allocation, so protecting a deep prefix avoids dozens of
  // small allocations. A short checkpoint retains only its own rows and at
  // most bounded slack from a reused block belonging to other consumers.
  std::stable_sort(copies.begin(), copies.end(),
                   [](const auto& left, const auto& right) {
                     return left.owners < right.owners;
                   });
  std::uint64_t capacity = 0;
  for (const auto& block : available)
    capacity += block->capacity - block->used;
  // Allocate before enqueuing any copy or changing a checkpoint. Allocation
  // failure then leaves the source and every retained checkpoint intact.
  std::vector<std::uint64_t> commits;
  std::uint64_t needed = 0;
  for (std::size_t first = 0; first < copies.size();) {
    auto last = first;
    std::uint64_t bytes = 0;
    while (last < copies.size() &&
           copies[last].owners == copies[first].owners) {
      bytes += copies[last].end - copies[last].begin;
      ++last;
    }
    const auto reused = std::min(capacity, bytes);
    capacity -= reused;
    bytes -= reused;
    if (bytes != 0) {
      commits.push_back(bytes);
      needed += bytes;
    }
    first = last;
  }
  bool device = needed == 0 || allocator->Fits(needed);
  if (device) {
    try {
      for (const auto bytes : commits) {
        auto block = std::make_shared<SnapshotBlock>(allocator, bytes);
        reserve.push_back(block);
        available.push_back(std::move(block));
      }
    } catch (const SnapshotDeviceExhausted&) {
      device = false;
    }
  }
  if (!device) {
    // No device room for the overwritten rows: move each affected checkpoint
    // into its host payload, which the snapshot budget already charges. This
    // also releases the device copies it held from earlier rewinds.
    std::vector<bool> affected(snapshots.size());
    for (const auto& copy : copies)
      for (const auto owner : copy.owners)
        affected[owner] = true;
    for (std::size_t index = 0; index < snapshots.size(); ++index)
      if (affected[index])
        snapshots[index]->MaterializeLocked();
    copies.clear();
    available.clear();
  }
  std::size_t block_index = 0;
  std::unique_ptr<gufo::hip::SnapshotTransfer> transfer;
  for (auto& copy : copies) {
    for (auto begin = copy.begin; begin < copy.end;) {
      if (block_index < available.size() &&
          available[block_index]->used == available[block_index]->capacity)
        ++block_index;
      if (block_index >= available.size())
        throw std::logic_error(
            "snapshot protection exceeded allocated storage");
      auto& used = available[block_index]->used;
      const auto bytes =
          std::min(available[block_index]->capacity - used, copy.end - begin);
      auto chunk = std::make_shared<SnapshotChunk>();
      chunk->block = available[block_index];
      ++chunk->block->chunks;
      chunk->device =
          static_cast<std::uint8_t*>(chunk->block->storage.get()) + used;
      if (!transfer)
        transfer = allocator->Transfer();
      transfer->Enqueue(chunk->device,
                        static_cast<const std::uint8_t*>(copy.device) + begin,
                        bytes, hipMemcpyDeviceToDevice);
      for (auto* region : copy.regions)
        region->saved.push_back({begin, bytes, chunk});
      used += bytes;
      begin += bytes;
    }
  }
  if (transfer)
    transfer->Finish();
  for (const auto& snapshot : snapshots) {
    for (auto& region : snapshot->regions_) {
      auto rows = region.draft ? mtp_position : position;
      if (region.pooled)
        rows = region.draft ? mtp_blocks : blocks;
      const auto keep = std::uint64_t{rows} * region.row_bytes;
      region.pending_bytes = std::min(region.pending_bytes, keep);
      region.live_bytes = std::min(region.live_bytes, keep);
    }
  }
}

void Session::PreserveSnapshots(std::uint32_t position,
                                std::uint32_t mtp_position) const {
  if (!snapshots_ || (position >= snapshot_position_ &&
                      mtp_position >= snapshot_mtp_position_))
    return;
  std::lock_guard lock(snapshots_->mutex);
  if (snapshots_->failure)
    std::rethrow_exception(snapshots_->failure);
  const auto ratio = owner_->config().compress_ratio;
  snapshots_->Preserve(position, mtp_position,
                       ratio ? std::min(blocks_, position / ratio) : 0,
                       ratio ? std::min(mtp_.blocks, mtp_position / ratio) : 0);
  snapshot_position_ = std::min(snapshot_position_, position);
  snapshot_mtp_position_ = std::min(snapshot_mtp_position_, mtp_position);
}

PrefillCheckpoint::~PrefillCheckpoint() {
  if (!state)
    return;
  source->snapshot_position_ =
      std::max(source->snapshot_position_, state->snapshot_position_);
  source->snapshot_mtp_position_ =
      std::max(source->snapshot_mtp_position_, state->snapshot_mtp_position_);
  // The view owns no cache allocations. Its borrowed rows belong to source.
  state->snapshots_.reset();
}

std::uint32_t Executor::PrefillCheckpointTail() const noexcept {
  return kVecBatch + std::max<std::uint32_t>(1, config().compress_ratio) - 1;
}

std::unique_ptr<PrefillCheckpoint> Executor::MakePrefillCheckpoint(
    Session& source, std::span<const std::int32_t> tokens, std::uint32_t prefix,
    std::string* error_msg) const {
  if (source.kvarn_ || source.kv_pool_) {
    AssignError(error_msg,
                "KVarN or shared KV sessions take no in-pass checkpoints");
    return nullptr;
  }
  if (source.owner_ != this || prefix == 0 || prefix > tokens.size() ||
      tokens.size() - prefix > PrefillCheckpointTail() ||
      !source.VisionLayout().images.empty()) {
    AssignError(error_msg, "prefill checkpoint must be in the text tail");
    return nullptr;
  }
  const auto& c = config();
  auto checkpoint = std::make_unique<PrefillCheckpoint>();
  checkpoint->source = &source;
  checkpoint->tokens = prefix;
  checkpoint->hidden_rows =
      source.mtp_enabled_ ? std::min(prefix, options_.max_speculative) : 0;
  checkpoint->logits.resize(c.vocab_size);
  checkpoint->state.reset(new Session);
  auto& view = *checkpoint->state;
  view.owner_ = this;
  view.max_context_ = source.max_context_;
  view.index_capacity_ = source.index_capacity_;
  view.position_ = source.position_ + prefix;
  view.mtp_enabled_ = source.mtp_enabled_;
  view.abliteration_ = source.abliteration_;
  view.abliteration_device_ = source.abliteration_device_;
  view.abliteration_sites_ = source.abliteration_sites_;
  view.attention_ = source.attention_;
  view.linear_.resize(source.linear_.size());
  view.blocks_ = c.compress_ratio > 0 && view.position_ > c.indexer_top_k
                     ? view.position_ / c.compress_ratio
                     : source.blocks_;
  view.mtp_ = source.mtp_;
  if (source.mtp_enabled_) {
    view.mtp_.position = view.position_ - 1;
    view.mtp_.blocks = view.mtp_.position > c.indexer_top_k
                           ? view.mtp_.position / c.compress_ratio
                           : source.mtp_.blocks;
    if (prefix > 1)
      view.mtp_.residual_valid = false;
  }
  view.ngram_ = source.ngram_;
  std::vector<std::uint32_t> rows(std::size_t{prefix} * c.ple_heads);
  if (c.ple_layer >= 0)
    HashNgramRows(c, view.ngram_, tokens.first(prefix), rows);
  if (!source.snapshots_)
    source.snapshots_ =
        std::make_shared<SnapshotRegistry>(GetSnapshotAllocator());
  if (!source.snapshot_lineage_)
    source.snapshot_lineage_ = std::make_shared<SnapshotLineage>();
  view.snapshots_ = source.snapshots_;
  view.snapshot_lineage_ = source.snapshot_lineage_;
  const auto conv = std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const auto recurrent =
      std::size_t{c.ssm_num_v_heads} * c.ssm_head_dim * c.ssm_head_dim;
  std::size_t floats =
      has_mtp() ? std::size_t{options_.max_speculative} * c.HcDim() : 0;
  for (const auto& linear : source.linear_)
    if (linear.state)
      floats += conv + recurrent;
  if (c.ple_layer >= 0)
    floats += std::size_t{c.PleConvHistory()} * c.HcDim();
  if (!checkpoint_storage_) {
    // Forward scratch is shared by the executor. The checkpoint is frozen
    // into private snapshot storage before another session can reuse it.
    checkpoint_storage_ =
        source.snapshots_->allocator->Allocate(floats * sizeof(float));
  }
  auto* storage = static_cast<float*>(checkpoint_storage_.get());
  for (std::size_t il = 0; il < source.linear_.size(); ++il) {
    if (!source.linear_[il].state)
      continue;
    view.linear_[il].conv_state = storage;
    storage += conv;
    view.linear_[il].state = storage;
    storage += recurrent;
  }
  if (c.ple_layer >= 0) {
    view.ple_history_ = storage;
    storage += std::size_t{c.PleConvHistory()} * c.HcDim();
  }
  if (source.mtp_enabled_)
    view.mtp_.target_hidden = storage;
  return checkpoint;
}

/// KVarN sessions' payloads (records, FP16 sink and tail rows in place of
/// the FP16 K/V of the trunk layers). FP16 payloads keep kSnapshotMagic.
constexpr std::array<char, 8> kSnapshotKvarnMagic{'Q', 'F', 'N', 'S',
                                                  'N', 'A', 'P', 'K'};
constexpr std::array<char, 8> kSnapshotKvarn6Magic{'Q', 'F', 'N', 'S',
                                                   'N', 'A', 'K', '6'};
constexpr std::array<char, 8> kSnapshotMagic{'Q', 'F', 'N', 'S',
                                             'N', 'A', 'P', '6'};

/// Fixed header ahead of the section bytes. It carries every geometry
/// value the section sizes derive from, so a payload of another artifact
/// or executor configuration is rejected before anything is copied.
struct SnapshotHeader {
  std::array<char, 8> magic;
  std::uint32_t num_layers;
  std::uint32_t full_attention_interval;
  std::uint32_t conv_elems;
  std::uint32_t state_elems;
  std::uint32_t kv_row;
  std::uint32_t indexer_head_dim;
  std::uint32_t compress_ratio;
  std::uint32_t hc_dim;
  std::uint32_t ple_elems;
  std::uint32_t has_mtp;
  std::uint32_t position;
  std::uint32_t blocks;
  std::uint32_t mtp_position;
  std::uint32_t mtp_blocks;
  std::uint32_t mtp_residual_valid;
  std::uint32_t hidden_rows;
  std::uint32_t image_count;
  // Rotated K depends on the rope scaling at every position.
  float rope_factor;
  std::uint32_t rope_original_context;
  float rope_beta_fast;
  float rope_beta_slow;
  std::array<std::int32_t, Config::kMaxPleNgram - 1> ngram_prev;
  std::uint64_t payload_bytes;
};
static_assert(std::is_trivially_copyable_v<SnapshotHeader>);
// The four rope fields must not push payload_bytes off its 8-byte alignment.
static_assert(offsetof(SnapshotHeader, payload_bytes) == 104);

namespace {

SnapshotHeader MakeSnapshotHeader(const Config& c, bool has_mtp,
                                  const Session& session) {
  SnapshotHeader h{};
  h.magic = session.kvarn() ? (session.kvarn_bits() == 6 ? kSnapshotKvarn6Magic
                                                         : kSnapshotKvarnMagic)
                            : kSnapshotMagic;
  h.num_layers = c.num_layers;
  h.full_attention_interval = c.full_attention_interval;
  h.conv_elems = (c.ssm_conv_kernel - 1) * c.SsmConvChannels();
  h.state_elems = c.ssm_num_v_heads * c.ssm_head_dim * c.ssm_head_dim;
  h.kv_row = c.AttentionKvDim();
  h.indexer_head_dim = c.indexer_head_dim;
  h.compress_ratio = c.compress_ratio;
  h.hc_dim = c.HcDim();
  h.ple_elems = c.ple_layer >= 0 ? c.PleConvHistory() * c.HcDim() : 0;
  h.has_mtp = has_mtp ? 1 : 0;
  h.position = session.position();
  h.image_count = std::ranges::count_if(
      session.VisionLayout().images,
      [&](const auto& image) { return image.offset < h.position; });
  // Disabled scaling has one disk-cache identity ("absolute-v1" from
  // RopePositionPolicy) regardless of whatever original_context/beta the
  // config happens to carry, so the header must normalize to the same
  // defaults rather than echo them raw (else two "off" configs that differ
  // only in those unused fields would collide on identity but disagree on
  // SameGeometry, and neither could restore the other's snapshot).
  const RopeScaling rope = NormalizedRopeScaling(c.rope_scaling);
  h.rope_factor = rope.factor;
  h.rope_original_context = rope.original_context;
  h.rope_beta_fast = rope.beta_fast;
  h.rope_beta_slow = rope.beta_slow;
  return h;
}

/// Whether `h` describes this executor's geometry (positions aside).
bool SameGeometry(const SnapshotHeader& h, const SnapshotHeader& mine) {
  return h.magic == mine.magic && h.num_layers == mine.num_layers &&
         h.full_attention_interval == mine.full_attention_interval &&
         h.conv_elems == mine.conv_elems && h.state_elems == mine.state_elems &&
         h.kv_row == mine.kv_row &&
         h.indexer_head_dim == mine.indexer_head_dim &&
         h.compress_ratio == mine.compress_ratio && h.hc_dim == mine.hc_dim &&
         h.ple_elems == mine.ple_elems && h.has_mtp == mine.has_mtp &&
         h.rope_factor == mine.rope_factor &&
         h.rope_original_context == mine.rope_original_context &&
         h.rope_beta_fast == mine.rope_beta_fast &&
         h.rope_beta_slow == mine.rope_beta_slow;
}

}  // namespace

template<typename Visit>
std::uint64_t Executor::WalkSnapshot(const SnapshotHeader& h,
                                     const Session* session, Visit&& visit) {
  std::uint64_t offset = sizeof(SnapshotHeader);
  const auto region = [&](void* device, std::uint64_t bytes, const char* what,
                          std::uint64_t row_bytes = 0, bool draft = false,
                          bool pooled = false) {
    if (bytes != 0 &&
        !visit(device, offset, bytes, what, row_bytes, draft, pooled)) {
      return false;
    }
    offset += bytes;
    return true;
  };
  const auto linear = [&](std::uint32_t il) -> const Session::LinearState* {
    return session != nullptr ? &session->linear_[il] : nullptr;
  };
  const auto attention =
      [&](std::uint32_t il) -> const Session::AttentionState* {
    return session != nullptr ? &session->attention_[il] : nullptr;
  };
  for (std::uint32_t il = 0; il < h.num_layers; ++il) {
    if (((il + 1) % h.full_attention_interval) != 0) {
      const auto* l = linear(il);
      if (!region(l != nullptr ? l->conv_state : nullptr,
                  std::uint64_t{h.conv_elems} * sizeof(float), "conv state") ||
          !region(l != nullptr ? l->state : nullptr,
                  std::uint64_t{h.state_elems} * sizeof(float),
                  "recurrent state")) {
        return 0;
      }
    }
  }
  if (!region(session != nullptr ? session->ple_history_ : nullptr,
              std::uint64_t{h.ple_elems} * sizeof(float), "PLE history")) {
    return 0;
  }
  const std::uint64_t kv_bytes =
      std::uint64_t{h.position} * h.kv_row * sizeof(__half);
  const std::uint64_t index_row_bytes =
      std::uint64_t{h.indexer_head_dim} * sizeof(float);
  const std::uint32_t index_begin = h.blocks * h.compress_ratio;
  const std::uint32_t index_rows = h.position - index_begin;
  const std::uint64_t block_bytes =
      std::uint64_t{h.blocks} * h.indexer_head_dim * sizeof(__half);
  for (std::uint32_t il = 0; il < h.num_layers; ++il) {
    if (((il + 1) % h.full_attention_interval) == 0) {
      const auto* at = attention(il);
      if (h.magic == kSnapshotKvarnMagic || h.magic == kSnapshotKvarn6Magic) {
        const int bits = h.magic == kSnapshotKvarn6Magic ? 6 : 4;
        // The sink, the records of every group before the position's group
        // floor (FinishKvarn keeps them quantized) and the FP16 tail rows
        // after it, all copied when captured: a rewind can requantize a
        // group and the ring is rewritten. Records lie in group order in
        // the session's private span (identity block table).
        const std::uint32_t base = std::max(
            hip::kKvarnSinkTokens,
            h.position >> hip::kKvarnGroupShift << hip::kKvarnGroupShift);
        const std::uint64_t row = std::uint64_t{h.kv_row} * sizeof(__half);
        const std::uint64_t sink =
            std::min(h.position, hip::kKvarnSinkTokens) * row;
        const std::uint64_t record =
            std::uint64_t{h.kv_row / hip::kKvarnHeadDim} *
            hip::KvarnRecord(bits, hip::KvarnValueOrder::kTokenMajor, bits)
                .bytes;
        const std::uint64_t records =
            std::uint64_t{(base >> hip::kKvarnGroupShift) - 1} * record;
        const std::uint64_t tail =
            h.position > base ? std::uint64_t{h.position - base} * row : 0;
        const std::size_t tail_row = hip::KvarnFp16Row(base) * h.kv_row;
        if (!region(at != nullptr ? at->kvarn_k : nullptr, sink,
                    "KVarN K sink") ||
            !region(at != nullptr ? at->kvarn_v : nullptr, sink,
                    "KVarN V sink") ||
            !region(at != nullptr ? at->kvarn_records + record : nullptr,
                    records, "KVarN records") ||
            !region(at != nullptr ? at->kvarn_k + tail_row : nullptr, tail,
                    "KVarN K tail") ||
            !region(at != nullptr ? at->kvarn_v + tail_row : nullptr, tail,
                    "KVarN V tail"))
          return 0;
      } else if (!region(at != nullptr ? at->k_cache : nullptr, kv_bytes,
                         "K cache", std::uint64_t{h.kv_row} * sizeof(__half)) ||
                 !region(at != nullptr ? at->v_cache : nullptr, kv_bytes,
                         "V cache", std::uint64_t{h.kv_row} * sizeof(__half))) {
        return 0;
      }
      // Serialize only unpooled rows, in chronological order. The physical
      // ring may wrap, and a destination session may have another capacity.
      if (session == nullptr) {
        if (!region(nullptr, index_rows * index_row_bytes, "indexer keys"))
          return 0;
      } else {
        const auto first = index_begin & (session->index_capacity_ - 1);
        const auto tail =
            std::min(index_rows, session->index_capacity_ - first);
        if (!region(at->index_k + std::size_t{first} * h.indexer_head_dim,
                    tail * index_row_bytes, "indexer tail") ||
            !region(at->index_k, (index_rows - tail) * index_row_bytes,
                    "indexer wrap"))
          return 0;
      }
      if (!region(at != nullptr ? at->block_k : nullptr, block_bytes,
                  "pooled block keys",
                  std::uint64_t{h.indexer_head_dim} * sizeof(__half), false,
                  true))
        return 0;
    }
  }
  if (h.has_mtp != 0) {
    const auto* mtp = session != nullptr ? &session->mtp_ : nullptr;
    const std::uint32_t begin = h.mtp_blocks * h.compress_ratio;
    const std::uint32_t rows = h.mtp_position - begin;
    if (session == nullptr) {
      if (!region(nullptr, rows * index_row_bytes, "draft indexer keys"))
        return 0;
    } else {
      const auto first = begin & (session->index_capacity_ - 1);
      const auto tail = std::min(rows, session->index_capacity_ - first);
      if (!region(mtp->index_k + std::size_t{first} * h.indexer_head_dim,
                  tail * index_row_bytes, "draft indexer tail") ||
          !region(mtp->index_k, (rows - tail) * index_row_bytes,
                  "draft indexer wrap"))
        return 0;
    }
    if (!region(
            mtp ? mtp->block_k : nullptr,
            std::uint64_t{h.mtp_blocks} * h.indexer_head_dim * sizeof(__half),
            "draft pooled keys",
            std::uint64_t{h.indexer_head_dim} * sizeof(__half), true, true))
      return 0;
    const std::uint64_t mtp_kv_bytes =
        std::uint64_t{h.mtp_position} * h.kv_row * sizeof(__half);
    if (!region(mtp != nullptr ? mtp->k_cache : nullptr, mtp_kv_bytes,
                "draft K cache", std::uint64_t{h.kv_row} * sizeof(__half),
                true) ||
        !region(mtp != nullptr ? mtp->v_cache : nullptr, mtp_kv_bytes,
                "draft V cache", std::uint64_t{h.kv_row} * sizeof(__half),
                true) ||
        !region(
            mtp != nullptr ? mtp->h : nullptr,
            h.mtp_residual_valid ? std::uint64_t{h.hc_dim} * sizeof(float) : 0,
            "draft residual") ||
        !region(mtp != nullptr ? mtp->target_hidden : nullptr,
                std::uint64_t{h.hidden_rows} * h.hc_dim * sizeof(float),
                "kept trunk rows")) {
      return 0;
    }
  }
  return offset +
         std::uint64_t{h.image_count} * sizeof(qwen::vision::ImageGrid);
}

std::uint64_t Executor::SnapshotBytes(const Session& session,
                                      std::uint32_t hidden_rows) const {
  SnapshotHeader h =
      MakeSnapshotHeader(config(), session.mtp_enabled_, session);
  h.blocks = session.blocks_;
  h.mtp_blocks = session.mtp_.blocks;
  h.mtp_position = session.mtp_.position;
  h.mtp_residual_valid = session.mtp_.residual_valid;
  h.hidden_rows = hidden_rows;
  return WalkSnapshot(h, nullptr,
                      [](void*, std::uint64_t, std::uint64_t, const char*,
                         std::uint64_t, bool, bool) { return true; });
}

bool Executor::SaveSnapshot(const Session& session, std::uint32_t hidden_rows,
                            std::span<std::uint8_t> payload,
                            std::string* error_msg,
                            std::shared_ptr<SnapshotState>* deferred,
                            std::shared_ptr<void> storage) const {
  if (session.kv_pool_) {
    AssignError(error_msg, "shared KV sessions use component snapshots");
    return false;
  }
  if (session.owner_ != this) {
    AssignError(error_msg, "session belongs to another executor");
    return false;
  }
  if (session.spec_tokens_ != 0) {
    AssignError(error_msg,
                "snapshot with a pending speculative batch; roll back first");
    return false;
  }
  if (hidden_rows > session.position_ ||
      hidden_rows > options_.max_speculative) {
    AssignError(error_msg, "kept trunk rows exceed the position or batch");
    return false;
  }
  SnapshotHeader h =
      MakeSnapshotHeader(config(), session.mtp_enabled_, session);
  h.blocks = session.blocks_;
  h.mtp_blocks = session.mtp_.blocks;
  h.mtp_position = session.mtp_.position;
  h.mtp_residual_valid = session.mtp_.residual_valid;
  h.hidden_rows = hidden_rows;
  h.ngram_prev = session.ngram_.prev;
  h.payload_bytes = SnapshotBytes(session, hidden_rows);
  if (payload.size() != h.payload_bytes) {
    AssignError(error_msg, "snapshot buffer size does not match the payload");
    return false;
  }
  std::shared_ptr<SnapshotState> pending;
  const auto transfer = GetSnapshotAllocator()->Transfer();
  if (deferred) {
    if (!storage) {
      AssignError(error_msg, "borrowed snapshot needs owned payload storage");
      return false;
    }
    if (!session.snapshots_)
      session.snapshots_ =
          std::make_shared<SnapshotRegistry>(GetSnapshotAllocator());
    pending = std::make_shared<SnapshotState>();
    pending->registry_ = session.snapshots_;
    if (!session.snapshot_lineage_)
      session.snapshot_lineage_ = std::make_shared<SnapshotLineage>();
    pending->lineage_ = session.snapshot_lineage_;
    pending->storage_ = std::move(storage);
    pending->payload_ = payload.data();
    std::uint64_t mutable_prefix = sizeof(SnapshotHeader);
    (void)WalkSnapshot(h, nullptr,
                       [&](void*, std::uint64_t offset, std::uint64_t bytes,
                           const char*, std::uint64_t row_bytes, bool, bool) {
                         if (row_bytes != 0)
                           return false;
                         mutable_prefix = offset + bytes;
                         return true;
                       });
    std::uint64_t extra_bytes = 0;
    (void)WalkSnapshot(h, nullptr,
                       [&](void*, std::uint64_t offset, std::uint64_t bytes,
                           const char*, std::uint64_t row_bytes, bool, bool) {
                         if (row_bytes == 0 && offset >= mutable_prefix)
                           extra_bytes += bytes;
                         return true;
                       });
    pending->mutable_offset_ = sizeof(SnapshotHeader);
    pending->mutable_bytes_ = mutable_prefix - sizeof(SnapshotHeader);
    try {
      pending->mutable_storage_ =
          session.snapshots_->allocator->AllocateSpare(pending->mutable_bytes_);
      pending->mutable_device_ =
          static_cast<std::uint8_t*>(pending->mutable_storage_.get());
      if (extra_bytes != 0) {
        pending->extra_storage_ =
            session.snapshots_->allocator->Allocate(extra_bytes);
        pending->extra_bytes_ = extra_bytes;
      }
    } catch (const SnapshotDeviceExhausted&) {
      // No device room to freeze this capture: copy it to its host payload.
      pending.reset();
    }
  }
  if (!pending)
    PopulateSnapshotPrefix(payload.data(), payload.size());
  std::memcpy(payload.data(), &h, sizeof(h));
  const auto grid_bytes =
      std::size_t{h.image_count} * sizeof(qwen::vision::ImageGrid);
  if (grid_bytes != 0) {
    std::memcpy(payload.data() + payload.size() - grid_bytes,
                session.VisionLayout().images.data(), grid_bytes);
  }
  std::uint64_t extra_used = 0;
  const bool saved =
      WalkSnapshot(
          h, &session,
          [&](void* device, std::uint64_t offset, std::uint64_t bytes,
              const char*, std::uint64_t row_bytes, bool draft, bool pooled) {
            if (!session.CheckCancellation(error_msg))
              return false;
            if (pending && row_bytes != 0) {
              pending->regions_.push_back({device,
                                           payload.data() + offset,
                                           bytes,
                                           bytes,
                                           bytes,
                                           row_bytes,
                                           draft,
                                           pooled,
                                           {}});
            } else {
              auto* destination = payload.data() + offset;
              auto kind = hipMemcpyDeviceToHost;
              if (pending && offset >= pending->mutable_offset_ &&
                  offset + bytes <=
                      pending->mutable_offset_ + pending->mutable_bytes_) {
                destination = pending->mutable_device_ + offset -
                              pending->mutable_offset_;
                kind = hipMemcpyDeviceToDevice;
              } else if (pending && pending->extra_storage_) {
                if (extra_used + bytes > pending->extra_bytes_)
                  throw std::logic_error("snapshot extra regions overflow");
                destination =
                    static_cast<std::uint8_t*>(pending->extra_storage_.get()) +
                    extra_used;
                kind = hipMemcpyDeviceToDevice;
                pending->extras_.push_back({offset, bytes, destination});
                extra_used += bytes;
              }
              if (pending)
                transfer->Enqueue(destination, device, bytes, kind);
              else
                transfer->Copy(destination, device, bytes);
            }
            return true;
          }) != 0;
  transfer->Finish();
  if (saved && pending) {
    std::lock_guard lock(session.snapshots_->mutex);
    std::erase_if(session.snapshots_->pending,
                  [](const auto& entry) { return entry.expired(); });
    session.snapshots_->pending.push_back(pending);
    session.snapshot_position_ =
        std::max(session.snapshot_position_, session.position_);
    session.snapshot_mtp_position_ =
        std::max(session.snapshot_mtp_position_, session.mtp_.position);
    *deferred = std::move(pending);
  }
  return saved;
}

std::uint64_t Executor::PrefillCheckpointBytes(const Session& session,
                                               std::uint32_t position) const {
  if (session.kvarn_ || session.kv_pool_)
    return 0;
  const auto& c = config();
  auto h = MakeSnapshotHeader(c, session.mtp_enabled_, session);
  h.position = position;
  h.blocks = c.compress_ratio > 0 && position > c.indexer_top_k
                 ? position / c.compress_ratio
                 : session.blocks_;
  if (session.mtp_enabled_) {
    h.mtp_position = position - 1;
    h.mtp_blocks = h.mtp_position > c.indexer_top_k
                       ? h.mtp_position / c.compress_ratio
                       : session.mtp_.blocks;
    h.hidden_rows =
        std::min(position - session.position_, options_.max_speculative);
  }
  return WalkSnapshot(h, nullptr,
                      [](void*, std::uint64_t, std::uint64_t, const char*,
                         std::uint64_t, bool, bool) { return true; });
}

bool Executor::RestoreSnapshot(Session& session,
                               std::span<const std::uint8_t> payload,
                               SnapshotInfo* info, std::string* error_msg,
                               std::uint32_t next_drafts,
                               const SnapshotState* deferred) const {
  if (session.kv_pool_) {
    AssignError(error_msg, "shared KV sessions use component snapshots");
    return false;
  }
  if (session.owner_ != this) {
    AssignError(error_msg, "session belongs to another executor");
    return false;
  }
  if (payload.size() < sizeof(SnapshotHeader)) {
    AssignError(error_msg, "snapshot payload is truncated");
    return false;
  }
  SnapshotHeader h{};
  std::memcpy(&h, payload.data(), sizeof(h));
  const SnapshotHeader mine =
      MakeSnapshotHeader(config(), session.mtp_enabled_, session);
  if (!SameGeometry(h, mine)) {
    AssignError(error_msg, "snapshot was taken with another model geometry");
    return false;
  }
  const std::uint32_t max_blocks =
      h.compress_ratio == 0 ? 0 : h.position / h.compress_ratio;
  if (h.image_count > h.position || h.position == 0 ||
      h.position > session.max_context_ || h.blocks > max_blocks ||
      h.mtp_position > h.position ||
      h.mtp_blocks > h.mtp_position / h.compress_ratio ||
      h.mtp_position - h.mtp_blocks * h.compress_ratio >
          session.index_capacity_ ||
      h.hidden_rows > h.position || h.hidden_rows > options_.max_speculative ||
      h.mtp_residual_valid > 1 ||
      (h.mtp_residual_valid != 0 && h.mtp_position == 0) ||
      (h.has_mtp == 0 && (h.mtp_position != 0 || h.hidden_rows != 0 ||
                          h.mtp_residual_valid != 0))) {
    AssignError(error_msg, "snapshot positions do not fit this session");
    return false;
  }
  if (h.position - h.blocks * h.compress_ratio > session.index_capacity_) {
    AssignError(error_msg, "unpooled indexer rows exceed the ring capacity");
    return false;
  }
  if (h.payload_bytes != payload.size() ||
      WalkSnapshot(h, nullptr,
                   [](void*, std::uint64_t, std::uint64_t, const char*,
                      std::uint64_t, bool, bool) { return true; }) !=
          payload.size()) {
    AssignError(error_msg, "snapshot payload size does not match its header");
    return false;
  }
  qwen::vision::RopeLayout layout;
  layout.images.resize(h.image_count);
  const auto grid_bytes =
      layout.images.size() * sizeof(qwen::vision::ImageGrid);
  if (grid_bytes != 0) {
    std::memcpy(layout.images.data(),
                payload.data() + payload.size() - grid_bytes, grid_bytes);
  }
  try {
    layout.Validate(session.max_context_);
  } catch (const std::exception& e) {
    AssignError(error_msg, e.what());
    return false;
  }
  if (!layout.images.empty() &&
      layout.Prefix(h.position) != session.VisionLayout().Prefix(h.position)) {
    AssignError(error_msg,
                "image snapshot layout does not match its prompt attachment");
    return false;
  }
  if (layout.images.empty())
    session.ConfigureVision(nullptr, nullptr, stream_);
  // The session's queued work targets buffers the copies overwrite.
  if (!Check(hipStreamSynchronize(stream_), "restore drain", error_msg)) {
    return false;
  }
  const bool borrowed = deferred && deferred->registry_ == session.snapshots_;
  std::unique_lock<std::mutex> snapshot_lock;
  std::unique_lock<std::mutex> source_lock;
  if (session.snapshots_)
    snapshot_lock =
        std::unique_lock(session.snapshots_->mutex, std::defer_lock);
  if (deferred && !borrowed)
    source_lock = std::unique_lock(deferred->registry_->mutex, std::defer_lock);
  if (snapshot_lock.mutex() && source_lock.mutex())
    std::lock(snapshot_lock, source_lock);
  else if (snapshot_lock.mutex())
    snapshot_lock.lock();
  else if (source_lock.mutex())
    source_lock.lock();
  if (session.snapshots_ && session.snapshots_->failure)
    std::rethrow_exception(session.snapshots_->failure);
  if (deferred && deferred->registry_->failure)
    std::rethrow_exception(deferred->registry_->failure);
  // A same-session restore retains the still-live prefix. Only detached rows
  // are uploaded. Save other checkpoints' overwritten suffixes first.
  std::uint32_t trunk_keep = 0;
  std::uint32_t draft_keep = 0;
  std::uint32_t trunk_blocks = 0;
  std::uint32_t draft_blocks = 0;
  if (borrowed) {
    trunk_keep = h.position;
    draft_keep = h.mtp_position;
    trunk_blocks = h.blocks;
    draft_blocks = h.mtp_blocks;
    for (const auto& region : deferred->regions_) {
      const auto rows =
          static_cast<std::uint32_t>(region.live_bytes / region.row_bytes);
      if (region.pooled) {
        if (region.draft)
          draft_blocks = std::min(draft_blocks, rows);
        else
          trunk_blocks = std::min(trunk_blocks, rows);
      } else {
        if (region.draft)
          draft_keep = std::min(draft_keep, rows);
        else
          trunk_keep = std::min(trunk_keep, rows);
      }
    }
  } else if (deferred) {
    const auto common = SnapshotLineage::Common(
        session.snapshot_lineage_, deferred->lineage_,
        {std::min(session.position_, h.position),
         std::min(session.mtp_.position, h.mtp_position),
         std::min(session.blocks_, h.blocks),
         std::min(session.mtp_.blocks, h.mtp_blocks)});
    trunk_keep = common[0];
    draft_keep = common[1];
    trunk_blocks = common[2];
    draft_blocks = common[3];
  }
  trunk_blocks = h.compress_ratio
                     ? std::min(trunk_blocks, trunk_keep / h.compress_ratio)
                     : 0;
  draft_blocks = h.compress_ratio
                     ? std::min(draft_blocks, draft_keep / h.compress_ratio)
                     : 0;
  auto restored_lineage =
      deferred ? std::make_shared<SnapshotLineage>(
                     deferred->lineage_,
                     SnapshotLineage::Rows{h.position, h.mtp_position, h.blocks,
                                           h.mtp_blocks})
               : nullptr;
  if (session.snapshots_) {
    session.snapshots_->Preserve(trunk_keep, draft_keep, trunk_blocks,
                                 draft_blocks);
  }
  // If restoration fails after a partial write, no provenance may survive.
  session.snapshot_lineage_.reset();
  // Every live state region is overwritten below. Retain only scratch that
  // the restored operation can use; this never allocates new rollback rows.
  // Explicit reset and failed restoration still release all scratch.
  const auto remaining = session.max_context_ - h.position;
  session.TrimRollback(std::min({next_drafts, options_.max_speculative - 1,
                                 remaining ? remaining - 1 : 0}));
  // A borrowed restore may reuse its source slot. Keep recorded graphs, but
  // let unrecorded shapes run eagerly once in the restored request before
  // recording them, just as they would in a fresh execution slot.
  std::erase_if(session.warmed_,
                [&](const auto key) { return !session.graphs_.contains(key); });
  std::unique_ptr<gufo::hip::SnapshotTransfer> fork_transfer;
  const auto upload = [&](void* device, const std::uint8_t* source,
                          std::uint64_t bytes, const char* what,
                          hipMemcpyKind kind = hipMemcpyHostToDevice) {
    if (!session.CheckCancellation(error_msg))
      return false;
    if (deferred) {
      if (!fork_transfer)
        fork_transfer = GetSnapshotAllocator()->Transfer();
      fork_transfer->Enqueue(device, source, bytes, kind);
      return true;
    }
    return Check(hipMemcpy(device, source, bytes, hipMemcpyHostToDevice), what,
                 error_msg);
  };

  if (WalkSnapshot(
          h, &session,
          [&](void* device, std::uint64_t offset, std::uint64_t bytes,
              const char* what, std::uint64_t row_bytes, bool draft,
              bool pooled) {
            const auto original_offset = offset;
            const auto original_bytes = bytes;
            if (deferred && row_bytes != 0) {
              auto rows = draft ? draft_keep : trunk_keep;
              if (pooled)
                rows = draft ? draft_blocks : trunk_blocks;
              const auto skip =
                  std::min(bytes, std::uint64_t{rows} * row_bytes);
              device = static_cast<std::uint8_t*>(device) + skip;
              offset += skip;
              bytes -= skip;
            }
            if (deferred && !borrowed && row_bytes != 0) {
              // Fork directly from the source's still-live prefix.
              // Its detached tail is already present in the payload.
              // Hold both registries until the copy completes so a
              // reset or destruction cannot invalidate either side.
              for (const auto& region : deferred->regions_) {
                if (region.host != payload.data() + original_offset)
                  continue;
                const auto skip = offset - original_offset;
                const auto copied =
                    region.live_bytes > skip
                        ? std::min(bytes, region.live_bytes - skip)
                        : 0;
                if (copied != 0) {
                  if (!session.CheckCancellation(error_msg))
                    return false;
                  if (!fork_transfer)
                    fork_transfer = GetSnapshotAllocator()->Transfer();
                  fork_transfer->Enqueue(
                      device,
                      static_cast<const std::uint8_t*>(region.device) + skip,
                      copied, hipMemcpyDeviceToDevice);
                }
                device = static_cast<std::uint8_t*>(device) + copied;
                offset += copied;
                bytes -= copied;
                break;
              }
            }
            if (bytes == 0)
              return true;
            if (deferred && row_bytes != 0) {
              for (const auto& region : deferred->regions_) {
                if (region.host != payload.data() + original_offset ||
                    region.saved.empty())
                  continue;
                const auto skip = offset - original_offset;
                for (const auto& saved : region.saved) {
                  const auto begin = std::max(skip, saved.offset);
                  const auto end =
                      std::min(original_bytes, saved.offset + saved.bytes);
                  if (begin < end) {
                    if (!session.CheckCancellation(error_msg))
                      return false;
                    if (!fork_transfer)
                      fork_transfer = GetSnapshotAllocator()->Transfer();
                    fork_transfer->Enqueue(
                        static_cast<std::uint8_t*>(device) + begin - skip,
                        saved.chunk->device + begin - saved.offset, end - begin,
                        hipMemcpyDeviceToDevice);
                  }
                }
                return true;
              }
            }
            const auto* source = payload.data() + offset;
            auto kind = hipMemcpyHostToDevice;
            if (deferred && deferred->mutable_storage_ &&
                offset >= deferred->mutable_offset_ &&
                offset + bytes <=
                    deferred->mutable_offset_ + deferred->mutable_bytes_) {
              source = deferred->mutable_device_ + offset -
                       deferred->mutable_offset_;
              kind = hipMemcpyDeviceToDevice;
            } else if (deferred && deferred->extra_storage_ && row_bytes == 0) {
              // The destination may split a ring region differently from the
              // source, so copy every overlapping frozen piece.
              auto* target = static_cast<std::uint8_t*>(device);
              for (const auto& extra : deferred->extras_) {
                const auto begin = std::max(offset, extra.offset);
                const auto end =
                    std::min(offset + bytes, extra.offset + extra.bytes);
                if (begin < end &&
                    !upload(target + (begin - offset),
                            extra.device + (begin - extra.offset), end - begin,
                            what, hipMemcpyDeviceToDevice))
                  return false;
              }
              return true;
            }
            return upload(device, source, bytes, what, kind);
          }) == 0) {
    if (fork_transfer)
      fork_transfer->Finish();
    if (snapshot_lock.owns_lock())
      snapshot_lock.unlock();
    session.Reset();
    return false;
  }
  if (fork_transfer)
    fork_transfer->Finish();
  session.RestoreVisionLayout(layout, stream_);
  session.position_ = h.position;
  session.blocks_ = h.blocks;
  if (session.kvarn_) {
    // The payload's records end at the group floor and its ring rows start
    // there, so a rewind before it must be refused: written_end is raised
    // past the ring's reach of the group before (hip::PlanKvarnPass).
    const std::uint32_t base =
        std::max(hip::kKvarnSinkTokens,
                 h.position >> hip::kKvarnGroupShift << hip::kKvarnGroupShift);
    session.kvarn_tail_ = {
        .flushed_end = base,
        .written_end =
            base == hip::kKvarnSinkTokens
                ? h.position
                : std::max(h.position, base + hip::kKvarnTailRingTokens -
                                           hip::kKvarnGroupTokens + 1)};
    session.kvarn_pass_ = {};
  }
  session.mtp_.position = h.mtp_position;
  session.mtp_.blocks = h.mtp_blocks;
  session.mtp_.residual_valid = h.mtp_residual_valid != 0;
  session.snapshot_lineage_ = std::move(restored_lineage);
  if (borrowed)
    for (auto& region : deferred->regions_)
      region.live_bytes = region.bytes;
  session.snapshot_position_ = borrowed ? h.position : trunk_keep;
  session.snapshot_mtp_position_ = borrowed ? h.mtp_position : draft_keep;
  session.spec_base_ = h.position;
  session.spec_tokens_ = 0;
  session.ngram_.prev = h.ngram_prev;
  if (info != nullptr) {
    info->position = h.position;
    info->hidden_rows = h.hidden_rows;
  }
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
