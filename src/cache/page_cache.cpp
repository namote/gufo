#include "src/cache/page_cache.hpp"

#include <atomic>
#include <system_error>

#include "src/core/platform/fs.hpp"

namespace gufo::cache {
namespace {
std::atomic<std::uint64_t> hint_failures{};
}  // namespace

std::uint64_t PageCacheHintFailures() noexcept {
  return hint_failures.load(std::memory_order_relaxed);
}

void DropCleanPages(platform::PlatformFile file, std::uint64_t offset,
                    std::uint64_t length) noexcept {
  if (!platform::FileValid(file))
    return;
  if (!platform::Advise(file, platform::AdviseHint::kDontNeed, offset, length))
    hint_failures.fetch_add(1, std::memory_order_relaxed);
}

WriteBehind::WriteBehind(platform::PlatformFile file,
                         std::uint64_t window) noexcept
    : file_(file),
      window_(window),
      enabled_(window != 0 && platform::FileValid(file)) {}

void WriteBehind::Fail(const std::error_code& ec) noexcept {
  enabled_ = false;
  if (ec == std::errc::not_supported)
    return;  // No range write-back on this platform.
  if (ec == std::errc::invalid_argument || ec == std::errc::invalid_seek ||
      ec == std::errc::function_not_supported ||
      ec == std::errc::not_enough_memory)
    hint_failures.fetch_add(1, std::memory_order_relaxed);
  else
    error_ = ec;  // A device error: the later sync would not report it.
}

void WriteBehind::Wrote(std::uint64_t bytes) noexcept {
  written_ += bytes;
  std::error_code ec;
  while (enabled_ && written_ - queued_ >= window_) {
    if (!platform::WriteBack(file_, platform::WriteBackMode::kStart, queued_,
                             window_, ec))
      return Fail(ec);
    queued_ += window_;
    // Keep the two newest queued windows in flight; wait for the older ones
    // and drop them, now that they are clean.
    if (queued_ < 2 * window_)
      continue;
    const auto clean = queued_ - 2 * window_;
    if (clean > dropped_) {
      if (!platform::WriteBack(file_, platform::WriteBackMode::kWait, dropped_,
                               clean - dropped_, ec))
        return Fail(ec);
      DropCleanPages(file_, dropped_, clean - dropped_);
      dropped_ = clean;
    }
  }
}

void WriteBehind::Synced() noexcept {
  DropCleanPages(file_, 0, 0);
  enabled_ = false;
}

}  // namespace gufo::cache
