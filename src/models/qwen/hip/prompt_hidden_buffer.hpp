#ifndef GUFO_MODELS_QWEN_HIP_PROMPT_HIDDEN_BUFFER_HPP_
#define GUFO_MODELS_QWEN_HIP_PROMPT_HIDDEN_BUFFER_HPP_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "src/cache/ledger.hpp"
#include "src/core/hip/host_allocations.hpp"

namespace gufo::hip {

/// Host destination of the DFlash2 prompt capture. Up to a bound (one prompt
/// batch of every captured layer) it is pinned, so the per-layer device-to-host
/// copies avoid HIP's staged pageable path; a larger request (a full prompt
/// from a non-serve caller) uses ordinary pageable memory instead, so the
/// pinned size stays bounded. Grow-only like the std::vector it replaces:
/// Clear() keeps the pinned capacity. Resize() does not zero new elements;
/// every capture fills the rows it adds. With a ledger, the pinned bytes are
/// charged as RAM before they are allocated. After a refused charge or a
/// failed pinned allocation, every later capture uses pageable memory, so
/// requests do not retry the allocation or pin a capacity sized to themselves.
///
/// Buffers may share one pinned allocation (PromptHiddenPinned): the rows are
/// only needed from a capture until the draft has injected them, and a serve
/// process captures for one slot at a time. A buffer takes the pages over when
/// it reserves or grows a capture, after the previous holder's stream drained.
/// Rows of a buffer whose pages were taken over are gone: View() is then empty,
/// which callers already reject as an incomplete capture.
///
/// PromptHiddenPinned: pinned pages shared by the PromptHiddenBuffers of one
/// process's slots.
class PromptHiddenPinned {
public:
  PromptHiddenPinned() = default;
  PromptHiddenPinned(const PromptHiddenPinned&) = delete;
  PromptHiddenPinned& operator=(const PromptHiddenPinned&) = delete;
  /// Pinned bytes charged to the ledger.
  [[nodiscard]] std::size_t ChargedBytes() const noexcept {
    return charged_bytes_.load(std::memory_order_relaxed);
  }

private:
  friend class PromptHiddenBuffer;
  std::mutex mutex_;
  cache::ResourceLedger* ledger_{};
  // Declared before its allocation, which is freed first.
  cache::ResourceCharge charge_;
  std::unique_ptr<HostAllocations> owner_;
  float* data_{};
  std::size_t capacity_{};
  bool failed_{};
  std::atomic<std::uint64_t> holder_{};
  hipStream_t holder_stream_{};
  std::atomic<std::size_t> charged_bytes_{};
};

class PromptHiddenBuffer {
public:
  /// Owns its pinned pages.
  PromptHiddenBuffer();
  /// Shares `pinned` with other buffers; `stream` is the one this buffer's
  /// captures copy on, drained before another buffer takes the pages over.
  PromptHiddenBuffer(std::shared_ptr<PromptHiddenPinned> pinned,
                     hipStream_t stream);
  ~PromptHiddenBuffer();
  PromptHiddenBuffer(const PromptHiddenBuffer&) = delete;
  PromptHiddenBuffer& operator=(const PromptHiddenBuffer&) = delete;
  PromptHiddenBuffer(PromptHiddenBuffer&&) = delete;
  PromptHiddenBuffer& operator=(PromptHiddenBuffer&&) = delete;
  /// Before any capture: shares `pinned` from now on, as the constructor does.
  void Share(std::shared_ptr<PromptHiddenPinned> pinned, hipStream_t stream);
  /// Sets the pinned bound for the next prompt and makes room for `elements`.
  /// A ledger, once given, charges this and every later pinned allocation.
  void Reserve(std::size_t elements, std::size_t pinned_limit,
               cache::ResourceLedger* ledger = nullptr);
  void Resize(std::size_t elements);
  void Clear() noexcept;
  [[nodiscard]] float* Data() noexcept;
  [[nodiscard]] std::size_t Size() const noexcept { return size_; }
  /// Pinned bytes charged to the ledger (of the shared pages, when shared).
  [[nodiscard]] std::size_t ChargedBytes() const noexcept {
    return pinned_->ChargedBytes();
  }
  [[nodiscard]] std::span<const float> View() const noexcept;

private:
  [[nodiscard]] bool Holds() const noexcept;
  // Takes the shared pages over; throws if this buffer's rows were lost.
  void Take(std::unique_lock<std::mutex>& lock);
  void GrowPinned(std::size_t elements);
  void StopPinning(std::size_t elements);
  void MoveToPageable(std::size_t elements);

  std::shared_ptr<PromptHiddenPinned> pinned_;
  hipStream_t stream_{};
  std::uint64_t id_;
  std::size_t pinned_limit_{};
  std::vector<float> pageable_;
  bool use_pageable_{};
  std::size_t size_{};
};

}  // namespace gufo::hip
#endif
