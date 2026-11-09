#ifndef GUFO_CACHE_PAGE_CACHE_HPP_
#define GUFO_CACHE_PAGE_CACHE_HPP_

#include <cstdint>
#include <system_error>

#include "src/core/platform/mapped_file.hpp"

// Keeps continuation-cache file I/O out of the host page cache. The disk
// stores write and read up to tens of GiB of checkpoint payloads; left cached,
// those pages fragment host memory over long uptimes and slow decode. Every
// call here is a hint: none fails or alters a write or read.
namespace gufo::cache {

/// Write-behind and drop-behind granularity.
inline constexpr std::uint64_t kPageCacheWindowBytes = std::uint64_t{8} << 20;

/// Rounds a drop start down to a page boundary. The kernel rounds a drop's
/// start up, so a reader dropping behind itself starts at the page holding
/// its previous end, whose bytes it already consumed.
[[nodiscard]] constexpr std::uint64_t PageCacheDropStart(
    std::uint64_t offset) noexcept {
  return offset & ~std::uint64_t{4095};
}

/// Page-cache hints the OS refused since process start. Hints never fail the
/// operation they accompany; this counter is their only error report.
[[nodiscard]] std::uint64_t PageCacheHintFailures() noexcept;

/// Drops the cached pages of `[offset, offset + length)`; a zero `length`
/// reaches the end of the file. Call it only on clean ranges: pages read and
/// consumed, or written back by WriteBack(kWait) or a file sync. A drop of
/// dirty pages frees nothing.
void DropCleanPages(platform::PlatformFile file, std::uint64_t offset,
                    std::uint64_t length) noexcept;

/// Bounds the page-cache footprint of a file written sequentially from offset
/// zero to about three windows. Each full window is queued for write-back
/// when it completes; windows more than two behind are then waited for and
/// dropped, so only written-back pages are dropped and two windows stay in
/// flight. Synced() drops the rest once the caller's durability sync has
/// cleaned the whole file. Run it on a writer thread: the wait may block for
/// one window of device write-back.
///
/// A failed write-back wait consumes the file's write-back error, so the
/// caller's later sync can no longer see it. WriteBackError() reports it; the
/// caller must treat it like a failed sync. Hint refusals (unsupported or
/// invalid ranges) only disable write-behind and count as hint failures.
class WriteBehind {
public:
  explicit WriteBehind(platform::PlatformFile file,
                       std::uint64_t window = kPageCacheWindowBytes) noexcept;
  WriteBehind(const WriteBehind&) = delete;
  WriteBehind& operator=(const WriteBehind&) = delete;

  /// Records `bytes` more written at the end of the sequential range.
  void Wrote(std::uint64_t bytes) noexcept;
  /// The caller's file sync succeeded: drops every cached page of the file.
  void Synced() noexcept;
  /// A device write-back error the waits consumed; empty when none.
  [[nodiscard]] std::error_code WriteBackError() const noexcept {
    return error_;
  }

private:
  void Fail(const std::error_code& ec) noexcept;

  platform::PlatformFile file_;
  std::uint64_t window_;
  std::uint64_t written_{};
  std::uint64_t queued_{};   // [0, queued_) is queued for write-back.
  std::uint64_t dropped_{};  // [0, dropped_) is written back and dropped.
  bool enabled_;
  std::error_code error_;
};

}  // namespace gufo::cache

#endif  // GUFO_CACHE_PAGE_CACHE_HPP_
