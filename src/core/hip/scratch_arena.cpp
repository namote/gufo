#include "src/core/hip/scratch_arena.hpp"

#include <algorithm>
#include <utility>

namespace gufo::hip {
namespace {

void SetError(std::string* error, std::string message) {
  if (error)
    *error = std::move(message);
}

[[nodiscard]] std::size_t PageCount(std::size_t bytes) {
  return (bytes + ScratchArena::kPage - 1) / ScratchArena::kPage;
}

}  // namespace

std::unique_ptr<ScratchArena> ScratchArena::Reserve(
    std::unique_ptr<Pages> pages, std::size_t capacity, std::string* error) {
  if (!pages) {
    SetError(error, "no scratch page backend");
    return nullptr;
  }
  if (capacity == 0) {
    SetError(error, "empty scratch reservation");
    return nullptr;
  }
  std::unique_ptr<ScratchArena> arena(new ScratchArena());
  arena->capacity_ = PageCount(capacity) * kPage;
  void* base = pages->Reserve(arena->capacity_, error);
  if (base == nullptr) {
    if (error && error->empty())
      *error = "scratch address reservation failed";
    return nullptr;
  }
  arena->base_ = static_cast<std::uint8_t*>(base);
  arena->pages_ = std::move(pages);
  return arena;
}

ScratchArena::~ScratchArena() {
  if (base_ == nullptr)
    return;
  ReleaseTransient();
  pages_->Free(base_, capacity_);
}

void* ScratchArena::Take(std::size_t bytes, std::string* error,
                         std::size_t alignment) {
  if (alignment == 0 || alignment > kPage ||
      (alignment & (alignment - 1)) != 0) {
    SetError(error, "invalid scratch alignment");
    return nullptr;
  }
  const std::size_t offset = (used_bytes_ + alignment - 1) & ~(alignment - 1);
  bytes = std::max<std::size_t>(bytes, 1);
  if (transient_mapped_ || offset > capacity_ || bytes > capacity_ - offset) {
    SetError(error, "scratch arena exhausted");
    return nullptr;
  }
  used_bytes_ = offset + bytes;
  used_pages_ = PageCount(used_bytes_);
  return base_ + offset;
}

bool ScratchArena::MapTransient(std::string* error) {
  if (transient_mapped_)
    return true;
  constexpr std::size_t run_pages = kMaxRunBytes / kPage;
  for (std::size_t first = 0; first < used_pages_; first += run_pages) {
    Run run{first, std::min(run_pages, used_pages_ - first), 0};
    if (!pages_->Map(base_ + run.first * kPage, run.pages * kPage, &run.handle,
                     error)) {
      ReleaseTransient();
      return false;
    }
    runs_.push_back(run);
  }
  transient_mapped_ = true;
  return true;
}

void ScratchArena::ReleaseTransient() noexcept {
  // Newest first, the reverse of mapping.
  for (auto run = runs_.rbegin(); run != runs_.rend(); ++run)
    pages_->Unmap(base_ + run->first * kPage, run->pages * kPage, run->handle);
  runs_.clear();
  transient_mapped_ = false;
}

}  // namespace gufo::hip
