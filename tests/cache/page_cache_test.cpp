// Page-cache hints of the continuation disk cache. Wraps the syscalls under
// the platform leaf to check that writers drop only written-back ranges, keep
// a bounded footprint, readers drop what they consumed on every exit, and a
// refused hint never fails a write or read.
#include "src/cache/page_cache.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cache/disk.hpp"
#include "src/core/platform/fs.hpp"

using namespace gufo;
using namespace gufo::cache;
namespace fs = std::filesystem;

namespace {
constexpr std::string_view kRootPrefix = "/tmp/gufo-page-cache-test-";

struct Range {
  std::uint64_t begin, end;
};
// What the wrapped syscalls saw for one descriptor until it closed.
struct FileTrace {
  std::string path;
  std::uint64_t written{}, clean_end{}, dropped_end{};
  std::uint64_t max_outstanding{};  // Most bytes written but not yet dropped.
  bool wrote{}, synced{};
  std::size_t drops_before_sync{};
  std::vector<Range> drops;
};
std::mutex trace_mutex;
std::map<int, FileTrace> open_traces;
std::vector<FileTrace> closed_traces;
bool fail_fadvise{};
int sync_range_errno{};  // Nonzero: sync_file_range fails with it.

std::string PathOf(int fd) {
  char path[4096];
  const auto n = readlink(("/proc/self/fd/" + std::to_string(fd)).c_str(), path,
                          sizeof(path));
  return n > 0 ? std::string(path, static_cast<std::size_t>(n)) : "";
}
// Returns the trace of `fd` when it is a test file; trace_mutex is held.
FileTrace* Trace(int fd) {
  if (auto it = open_traces.find(fd); it != open_traces.end()) {
    it->second.path = PathOf(fd);
    return &it->second;
  }
  auto path = PathOf(fd);
  if (path.rfind(kRootPrefix, 0) != 0)
    return nullptr;
  auto& trace = open_traces[fd];
  trace.path = std::move(path);
  return &trace;
}
}  // namespace

extern "C" ssize_t __real_write(int, const void*, size_t);
extern "C" ssize_t __wrap_write(int fd, const void* data, size_t size) {
  const auto n = __real_write(fd, data, size);
  if (n > 0) {
    const std::lock_guard lock(trace_mutex);
    if (auto* trace = Trace(fd)) {
      trace->wrote = true;
      trace->written += static_cast<std::uint64_t>(n);
      trace->max_outstanding =
          std::max(trace->max_outstanding, trace->written - trace->dropped_end);
    }
  }
  return n;
}
extern "C" int __real_fsync(int);
extern "C" int __wrap_fsync(int fd) {
  const int result = __real_fsync(fd);
  if (result == 0) {
    const std::lock_guard lock(trace_mutex);
    if (auto* trace = Trace(fd)) {
      trace->clean_end = trace->written;
      trace->synced = true;
    }
  }
  return result;
}
extern "C" int __real_sync_file_range(int, off64_t, off64_t, unsigned);
extern "C" int __wrap_sync_file_range(int fd, off64_t offset, off64_t bytes,
                                      unsigned flags) {
  // A device error is reported by the waits, which consume it; a refusal by
  // every call.
  if (sync_range_errno &&
      (sync_range_errno != EIO || (flags & SYNC_FILE_RANGE_WAIT_AFTER))) {
    errno = sync_range_errno;
    return -1;
  }
  const int result = __real_sync_file_range(fd, offset, bytes, flags);
  constexpr unsigned kWait = SYNC_FILE_RANGE_WAIT_BEFORE |
                             SYNC_FILE_RANGE_WRITE | SYNC_FILE_RANGE_WAIT_AFTER;
  if (result == 0 && (flags & kWait) == kWait) {
    const std::lock_guard lock(trace_mutex);
    if (auto* trace = Trace(fd)) {
      const auto begin = static_cast<std::uint64_t>(offset);
      const auto end =
          bytes ? begin + static_cast<std::uint64_t>(bytes) : trace->written;
      if (begin <= trace->clean_end)
        trace->clean_end = std::max(trace->clean_end, end);
    }
  }
  return result;
}
extern "C" int __real_posix_fadvise(int, off_t, off_t, int);
extern "C" int __wrap_posix_fadvise(int fd, off_t offset, off_t bytes,
                                    int advice) {
  if (advice == POSIX_FADV_DONTNEED) {
    const std::lock_guard lock(trace_mutex);
    if (auto* trace = Trace(fd)) {
      const auto begin = static_cast<std::uint64_t>(offset);
      std::uint64_t end = bytes ? begin + static_cast<std::uint64_t>(bytes)
                                : std::max(trace->written, begin);
      if (!bytes && !trace->wrote) {
        struct stat st{};
        if (fstat(fd, &st) == 0)
          end = std::max<std::uint64_t>(begin, st.st_size);
      }
      // Writers drop only written-back pages.
      if (trace->wrote && end > trace->clean_end) {
        std::cerr << "dropped dirty range [" << begin << ", " << end << ") of "
                  << trace->path << " clean to " << trace->clean_end << "\n";
        std::abort();
      }
      trace->drops.push_back({begin, end});
      if (begin <= trace->dropped_end)
        trace->dropped_end = std::max(trace->dropped_end, end);
      if (!trace->synced)
        ++trace->drops_before_sync;
    }
  }
  if (fail_fadvise)
    return EINVAL;
  return __real_posix_fadvise(fd, offset, bytes, advice);
}
extern "C" int __real_close(int);
extern "C" int __wrap_close(int fd) {
  {
    const std::lock_guard lock(trace_mutex);
    if (auto it = open_traces.find(fd); it != open_traces.end()) {
      closed_traces.push_back(std::move(it->second));
      open_traces.erase(it);
    }
  }
  return __real_close(fd);
}

namespace {
ResourceLedger ledger{{1ULL << 32, 1ULL << 32, 0, 0}};
struct Directory {
  fs::path path;
  Directory() {
    std::string name = std::string(kRootPrefix) + "XXXXXX";
    path = mkdtemp(name.data());
  }
  ~Directory() { fs::remove_all(path); }
};
// Closed traces of files that saw a drop; directory handles are left out.
std::vector<FileTrace> TakeTraces() {
  const std::lock_guard lock(trace_mutex);
  auto traces = std::exchange(closed_traces, {});
  std::erase_if(traces, [](const auto& t) { return t.drops.empty(); });
  return traces;
}
std::vector<std::uint8_t> Pattern(std::size_t bytes, unsigned seed) {
  std::vector<std::uint8_t> data(bytes);
  for (std::size_t i = 0; i < bytes; ++i)
    data[i] = static_cast<std::uint8_t>((i * 2654435761u + seed) >> 13);
  return data;
}
// True when the drops cover [0, bytes).
bool Covers(std::vector<Range> drops, std::uint64_t bytes) {
  std::sort(drops.begin(), drops.end(),
            [](auto a, auto b) { return a.begin < b.begin; });
  std::uint64_t covered{};
  for (const auto& d : drops)
    if (d.begin <= covered)
      covered = std::max(covered, d.end);
  return covered >= bytes;
}
std::vector<std::uint8_t> ReadBack(const fs::path& path) {
  std::vector<std::uint8_t> bytes(fs::file_size(path));
  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  assert(fd >= 0);
  std::size_t done{};
  while (done < bytes.size()) {
    const auto n = read(fd, bytes.data() + done, bytes.size() - done);
    assert(n > 0);
    done += static_cast<std::size_t>(n);
  }
  close(fd);
  return bytes;
}

// Writes `data` through WriteBehind in uneven pieces, then syncs. Returns the
// write-back error WriteBehind consumed.
std::error_code WriteThrough(const fs::path& root, const std::string& name,
                             std::span<const std::uint8_t> data,
                             std::uint64_t window) {
  std::error_code ec;
  platform::Directory dir;
  const bool opened = dir.Open(root, ec);
  assert(opened);
  auto file = platform::OpenAt(dir, name, platform::FileMode::kCreateNew, ec);
  assert(platform::FileValid(file));
  std::error_code error;
  {
    WriteBehind behind(file, window);
    for (auto rest = data; !rest.empty();) {
      const auto piece = rest.first(std::min<std::size_t>(rest.size(), 10007));
      std::size_t done{};
      while (done < piece.size()) {
        const auto n =
            platform::Write(file, piece.data() + done, piece.size() - done, ec);
        assert(n > 0);
        done += n;
      }
      behind.Wrote(piece.size());
      rest = rest.subspan(piece.size());
    }
    const bool synced = platform::SyncFile(file, ec);
    assert(synced);
    behind.Synced();
    error = behind.WriteBackError();
  }
  platform::CloseFile(file);
  return error;
}

void WriteBehindBoundsFootprint() {
  Directory directory;
  constexpr std::uint64_t kWindow = 64 << 10;
  const auto data = Pattern((1 << 20) + 12345, 1);
  assert(!WriteThrough(directory.path, "image", data, kWindow));
  const auto traces = TakeTraces();
  assert(traces.size() == 1);
  const auto& trace = traces[0];
  assert(trace.written == data.size() && trace.synced);
  // Written-back windows left the cache while the file was still growing.
  assert(trace.drops_before_sync >= 10);
  assert(trace.max_outstanding <= 3 * kWindow + 10007);
  assert(Covers(trace.drops, data.size()));
  assert(ReadBack(directory.path / "image") == data);
}

void RefusedHintsNeverFail() {
  Directory directory;
  const auto before = PageCacheHintFailures();
  fail_fadvise = true;
  sync_range_errno = EINVAL;
  const auto data = Pattern(300 << 10, 2);
  assert(!WriteThrough(directory.path, "image", data, 64 << 10));
  fail_fadvise = false;
  sync_range_errno = 0;
  (void)TakeTraces();
  // One refused write-back disables write-behind; the final drop also fails.
  assert(PageCacheHintFailures() == before + 2);
  assert(ReadBack(directory.path / "image") == data);
}

// A device error from a write-back wait consumes the error the file sync
// would report, so WriteBehind must surface it instead of counting a hint.
void WriteBackErrorIsReported() {
  Directory directory;
  const auto before = PageCacheHintFailures();
  sync_range_errno = EIO;
  const auto error =
      WriteThrough(directory.path, "image", Pattern(300 << 10, 6), 64 << 10);
  sync_range_errno = 0;
  (void)TakeTraces();
  assert(error == std::errc::io_error && PageCacheHintFailures() == before);
}

DiskFileId Id(unsigned n) {
  DiskFileId id{};
  id[0] = n & 255;
  id[1] = (n >> 8) & 255;
  return id;
}

void DiskStoreDropsWrittenAndRestoredPayloads() {
  Directory directory;
  // Five write-behind windows and a bit, so write-behind and drop-behind run.
  const auto state = Pattern(5 * kPageCacheWindowBytes + 4099, 3);
  const auto chunk = Pattern(8, 4);
  const auto tail = Pattern(4, 5);
  DiskManifest manifest;
  manifest.checkpoint = {7};
  manifest.lineage = {3};
  manifest.compatibility.fill(0x42);
  manifest.input = {1, 2, 3};
  manifest.tokens = {17, 18, 19};
  manifest.components.push_back({{{1}, 4, ComponentKind::kAppendRows, 4, 2, 0},
                                 {{1}, 3},
                                 {{Id(1), chunk.size(), 0}},
                                 DiskPayload{Id(2), tail.size(), 0},
                                 {}});
  manifest.components.push_back(
      {{{2}, 5, ComponentKind::kPrivateState, 0, 0, state.size()},
       {{2}, 3},
       {},
       {},
       DiskPayload{Id(3), state.size(), 0}});
  const auto source = [](const std::vector<std::uint8_t>& bytes) {
    return [&bytes](std::uint64_t offset, std::span<std::uint8_t> out) {
      std::copy_n(bytes.begin() + offset, out.size(), out.begin());
    };
  };
  const std::array<DiskWriteSource, 3> sources{{
      {false, Id(1), chunk.size(), source(chunk)},
      {true, Id(2), tail.size(), source(tail)},
      {true, Id(3), state.size(), source(state)},
  }};
  std::vector<std::uint8_t> staging(1 << 20);
  DiskStore disk(ledger, directory.path, 1ULL << 30);
  (void)disk.PublishStream(Id(7), manifest, sources, staging);
  {
    const auto traces = TakeTraces();
    const auto written =
        std::find_if(traces.begin(), traces.end(),
                     [&](const auto& t) { return t.written == state.size(); });
    assert(written != traces.end() && written->synced);
    assert(written->drops_before_sync >= 2);
    assert(written->max_outstanding <=
           3 * kPageCacheWindowBytes + staging.size());
    assert(Covers(written->drops, state.size()));
  }
  auto snapshot = disk.Open({7});
  assert(snapshot);
  const auto payload = *snapshot->Manifest().components[1].private_state;
  std::vector<std::uint8_t> restored;
  (void)disk.ReadPayload(
      *snapshot, true, payload, staging, [&](std::uint64_t offset, auto bytes) {
        assert(offset == restored.size());
        restored.insert(restored.end(), bytes.begin(), bytes.end());
      });
  assert(restored == state);
  {
    const auto traces = TakeTraces();
    assert(traces.size() == 1 && !traces[0].wrote);
    // Dropped behind the read, not only at the end.
    assert(traces[0].drops.size() >= 3);
    assert(traces[0].drops.front().begin == 0 &&
           traces[0].drops.front().end <= state.size() / 2);
    assert(Covers(traces[0].drops, state.size()));
  }
  // A consumer failure still drops what the read brought in.
  bool failed{};
  try {
    (void)disk.ReadPayload(*snapshot, true, payload, staging,
                           [](std::uint64_t, auto) {
                             throw std::runtime_error("consumer failed");
                           });
  } catch (const std::runtime_error&) {
    failed = true;
  }
  assert(failed);
  {
    const auto traces = TakeTraces();
    assert(traces.size() == 1 && Covers(traces[0].drops, state.size()));
  }
  // A dropped prefetch drops the whole file.
  disk.DropPayloadCache(*snapshot, true, payload);
  {
    const auto traces = TakeTraces();
    assert(traces.size() == 1 && Covers(traces[0].drops, state.size()));
  }
  // Refused hints leave restores intact.
  const auto before = PageCacheHintFailures();
  fail_fadvise = true;
  restored.clear();
  (void)disk.ReadPayload(
      *snapshot, true, payload, staging, [&](std::uint64_t, auto bytes) {
        restored.insert(restored.end(), bytes.begin(), bytes.end());
      });
  fail_fadvise = false;
  (void)TakeTraces();
  assert(restored == state && PageCacheHintFailures() > before);
  // A write-back error fails the publication, as a failed fsync would.
  manifest.checkpoint = {8};
  manifest.components[1].private_state->file = Id(13);
  const std::array<DiskWriteSource, 1> fresh{
      {{true, Id(13), state.size(), source(state)}}};
  sync_range_errno = EIO;
  bool refused{};
  try {
    (void)disk.PublishStream(Id(8), manifest, fresh, staging);
  } catch (const std::exception&) {
    refused = true;
  }
  sync_range_errno = 0;
  (void)TakeTraces();
  assert(refused && !disk.Open({8}));
}
}  // namespace

int main() {
  WriteBehindBoundsFootprint();
  RefusedHintsNeverFail();
  WriteBackErrorIsReported();
  DiskStoreDropsWrittenAndRestoredPayloads();
  std::cout << "page cache hints ok\n";
  return 0;
}
