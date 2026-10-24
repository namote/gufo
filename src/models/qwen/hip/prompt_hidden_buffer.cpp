#include "src/models/qwen/hip/prompt_hidden_buffer.hpp"

#include <algorithm>
#include <cstring>
#include <new>
#include <stdexcept>

namespace gufo::hip {
namespace {
std::atomic<std::uint64_t> next_buffer_id{1};
}  // namespace

PromptHiddenBuffer::PromptHiddenBuffer()
    : PromptHiddenBuffer(std::make_shared<PromptHiddenPinned>(), nullptr) {}

PromptHiddenBuffer::PromptHiddenBuffer(
    std::shared_ptr<PromptHiddenPinned> pinned, hipStream_t stream)
    : pinned_(std::move(pinned)),
      stream_(stream),
      id_(next_buffer_id.fetch_add(1, std::memory_order_relaxed)) {
  if (!pinned_)
    throw std::invalid_argument("prompt hidden buffer needs pinned pages");
  use_pageable_ = pinned_->failed_;
}

PromptHiddenBuffer::~PromptHiddenBuffer() {
  const std::lock_guard lock(pinned_->mutex_);
  if (Holds()) {
    pinned_->holder_.store(0, std::memory_order_release);
    pinned_->holder_stream_ = nullptr;
  }
}

void PromptHiddenBuffer::Share(std::shared_ptr<PromptHiddenPinned> pinned,
                               hipStream_t stream) {
  if (!pinned)
    throw std::invalid_argument("prompt hidden buffer needs pinned pages");
  if (size_ > 0 || pinned_->capacity_ > 0)
    throw std::logic_error("prompt hidden buffer already captured");
  pinned_ = std::move(pinned);
  stream_ = stream;
  use_pageable_ = pinned_->failed_;
}

bool PromptHiddenBuffer::Holds() const noexcept {
  return pinned_->holder_.load(std::memory_order_acquire) == id_;
}

void PromptHiddenBuffer::Take(std::unique_lock<std::mutex>& lock) {
  (void)lock;
  if (Holds())
    return;
  if (size_ > 0 && !use_pageable_)
    throw std::logic_error(
        "prompt hidden rows were taken over by another capture");
  // The previous holder's copies may still target the pages.
  if (pinned_->holder_.load(std::memory_order_relaxed) != 0 &&
      pinned_->holder_stream_ != nullptr &&
      pinned_->holder_stream_ != stream_ &&
      hipStreamSynchronize(pinned_->holder_stream_) != hipSuccess) {
    (void)hipGetLastError();
    throw std::runtime_error("cannot drain the previous prompt hidden capture");
  }
  pinned_->holder_.store(id_, std::memory_order_release);
  pinned_->holder_stream_ = stream_;
}

void PromptHiddenBuffer::Reserve(std::size_t elements, std::size_t pinned_limit,
                                 cache::ResourceLedger* ledger) {
  {
    const std::lock_guard lock(pinned_->mutex_);
    if (ledger)
      pinned_->ledger_ = ledger;
    use_pageable_ = use_pageable_ || pinned_->failed_;
  }
  pinned_limit_ = pinned_limit;
  if (use_pageable_ || elements > pinned_limit_) {
    MoveToPageable(elements);
    return;
  }
  GrowPinned(elements);
}

void PromptHiddenBuffer::Resize(std::size_t elements) {
  if (!use_pageable_) {
    if (elements <= pinned_limit_) {
      std::unique_lock lock(pinned_->mutex_);
      const auto capacity = pinned_->capacity_;
      lock.unlock();
      GrowPinned(elements > capacity
                     ? std::min(std::max(elements, 2 * capacity), pinned_limit_)
                     : elements);
    } else {
      MoveToPageable(elements);
    }
  }
  if (use_pageable_) {
    pageable_.resize(elements);
  }
  size_ = elements;
}

void PromptHiddenBuffer::Clear() noexcept {
  size_ = 0;
  pageable_.clear();
  use_pageable_ = pinned_->failed_;
}

float* PromptHiddenBuffer::Data() noexcept {
  if (use_pageable_)
    return pageable_.data();
  return Holds() ? pinned_->data_ : nullptr;
}

std::span<const float> PromptHiddenBuffer::View() const noexcept {
  if (use_pageable_)
    return {pageable_.data(), size_};
  if (!Holds())
    return {};
  return {pinned_->data_, size_};
}

void PromptHiddenBuffer::GrowPinned(std::size_t elements) {
  std::unique_lock lock(pinned_->mutex_);
  if (pinned_->failed_) {
    lock.unlock();
    MoveToPageable(elements);
    return;
  }
  Take(lock);
  if (elements <= pinned_->capacity_)
    return;
  // Locked pages are host RAM the cache cannot reclaim. Admit them before
  // allocating, as streaming.cpp admits staging; a failed allocation drops
  // the admission.
  cache::ResourceReservation admission;
  if (pinned_->ledger_) {
    try {
      admission = pinned_->ledger_->Reserve(cache::ResourceCategory::kMetadata,
                                            elements * sizeof(float));
    } catch (const std::bad_alloc&) {
      lock.unlock();
      StopPinning(elements);
      return;
    }
  }
  auto owner = std::make_unique<HostAllocations>();
  float* data = nullptr;
  if (owner->Allocate(&data, elements * sizeof(float)) != hipSuccess) {
    // Pinned memory is an optimization; keep capturing into pageable memory.
    (void)hipGetLastError();
    lock.unlock();
    StopPinning(elements);
    return;
  }
  // Callers have synchronized the stream: no copy is still in flight to the
  // old pages, which this reads and then frees.
  if (size_ > 0) {
    std::memcpy(data, pinned_->data_, size_ * sizeof(float));
  }
  auto charge = admission ? admission.Convert() : cache::ResourceCharge{};
  // Free the previous pages before releasing their charge.
  pinned_->owner_ = std::move(owner);
  pinned_->charge_ = std::move(charge);
  pinned_->data_ = data;
  pinned_->capacity_ = elements;
  pinned_->charged_bytes_.store(pinned_->charge_ ? elements * sizeof(float) : 0,
                                std::memory_order_relaxed);
}

void PromptHiddenBuffer::StopPinning(std::size_t elements) {
  MoveToPageable(elements);
  const std::lock_guard lock(pinned_->mutex_);
  if (pinned_->failed_)
    return;
  // No capture pins again: free the pages, then release their charge. Every
  // buffer sharing them now captures into pageable memory; only the holder
  // can have rows in them, and its rows were copied above.
  pinned_->failed_ = true;
  pinned_->owner_.reset();
  pinned_->charge_ = {};
  pinned_->data_ = nullptr;
  pinned_->capacity_ = 0;
  pinned_->charged_bytes_.store(0, std::memory_order_relaxed);
}

void PromptHiddenBuffer::MoveToPageable(std::size_t elements) {
  if (!use_pageable_) {
    const std::lock_guard lock(pinned_->mutex_);
    if (size_ > 0) {
      if (!Holds() || pinned_->data_ == nullptr)
        throw std::logic_error(
            "prompt hidden rows were taken over by another capture");
      pageable_.assign(pinned_->data_, pinned_->data_ + size_);
    }
    use_pageable_ = true;
  }
  pageable_.reserve(elements);
}

}  // namespace gufo::hip
