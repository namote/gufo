#ifndef GUFO_SERVER_SNAPSHOT_POOL_HPP_
#define GUFO_SERVER_SNAPSHOT_POOL_HPP_

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

#include "src/cache/streaming.hpp"
#include "src/cli/serve/logging.hpp"

namespace gufo::server {
using SnapshotClock = std::chrono::steady_clock;
// Pinned host arenas of reassignment snapshots, at most `limit` bytes
// allocated at once (reassignment_snapshot_bytes), pooled or in use. An
// arena the writer is done with goes back to the pool (no free on the
// writer's path), a later snapshot reuses it, and a background thread frees
// arenas left unused for `idle`. Nothing is allocated before the first
// snapshot.
class SnapshotPool : public std::enable_shared_from_this<SnapshotPool> {
public:
  struct Arena {
    std::size_t bytes{};
    cache::StagingAllocation allocation;
    bool in_use{};
    SnapshotClock::time_point freed{};
  };
  // One snapshot's hold on an arena; returns it to the pool when destroyed.
  class Use {
  public:
    explicit Use(std::shared_ptr<SnapshotPool> pool) : pool_(std::move(pool)) {}
    Use(const Use&) = delete;
    ~Use() {
      if (arena_)
        pool_->Return(*arena_);
    }
    [[nodiscard]] std::byte* Data() const noexcept {
      return reinterpret_cast<std::byte*>(arena_->allocation.bytes.data());
    }

  private:
    friend class SnapshotPool;
    std::shared_ptr<SnapshotPool> pool_;
    Arena* arena_{};
  };
  struct Taken {
    // In the order of the requested sizes; empty with `refused`.
    std::vector<std::shared_ptr<Use>> arenas;
    std::size_t reused{}, allocated{};
    std::string refused, error;
  };
  SnapshotPool(std::size_t limit, std::chrono::milliseconds idle,
               cache::StagingAllocator allocate)
      : limit_(limit), idle_(idle), allocate_(std::move(allocate)) {}
  SnapshotPool(const SnapshotPool&) = delete;
  // Bytes allocated (pooled or in use) or reserved for allocation.
  [[nodiscard]] std::size_t Charged() const {
    const std::lock_guard lock(mutex_);
    return charged_;
  }
  // Arenas of `sizes`: pooled ones of the same size first, then new ones,
  // making room by freeing other pooled ones. `room(bytes)` may refuse the
  // new allocation ("host_memory"); "cap" when even an empty pool would not
  // fit them.
  Taken Take(std::span<const std::size_t> sizes,
             const std::function<bool(std::size_t)>& room) {
    Taken taken;
    std::size_t need = 0, reserved = 0, fresh_count = 0;
    try {
      taken.arenas.reserve(sizes.size());
      for (std::size_t i = 0; i < sizes.size(); ++i)
        taken.arenas.push_back(std::make_shared<Use>(shared_from_this()));
      const std::lock_guard lock(mutex_);
      if (!trimmer_.joinable())
        trimmer_ = std::jthread([this](std::stop_token stop) { Trim(stop); });
      arenas_.reserve(arenas_.size() + sizes.size());
      for (std::size_t i = 0; i < sizes.size(); ++i) {
        const auto pooled =
            std::ranges::find_if(arenas_, [&](const auto& arena) {
              return !arena->in_use && arena->bytes == sizes[i];
            });
        if (pooled == arenas_.end()) {
          need += sizes[i];
          ++fresh_count;
          continue;
        }
        (*pooled)->in_use = true;
        taken.arenas[i]->arena_ = pooled->get();
        taken.reused += sizes[i];
      }
    } catch (const std::exception& error) {
      // Before any arena was taken.
      taken.refused = "allocation_failed";
      taken.error = error.what();
    }
    // Before giving up pooled arenas for new ones the host cannot take.
    if (taken.refused.empty() && need > limit_)
      taken.refused = "cap";
    if (taken.refused.empty() && need && room && !room(need))
      taken.refused = "host_memory";
    if (taken.refused.empty()) {
      const std::lock_guard lock(mutex_);
      // Under this lock: the trim thread may have taken the list meanwhile.
      evicted_.reserve(evicted_.size() + arenas_.size());
      // Make room with pooled arenas of other sizes. The trim thread frees
      // them, not this (scheduler) thread; until it has, the pinned bytes
      // briefly exceed the cap by theirs.
      for (auto it = arenas_.begin();
           need > limit_ - std::min(limit_, charged_) && it != arenas_.end();) {
        if ((*it)->in_use) {
          ++it;
          continue;
        }
        charged_ -= (*it)->bytes;
        evicted_.push_back(std::move((*it)->allocation));
        it = arenas_.erase(it);
        ++generation_;
      }
      if (need > limit_ - std::min(limit_, charged_))
        taken.refused = "cap";
      else
        charged_ += reserved = need;
    }
    changed_.notify_all();
    if (!taken.refused.empty()) {
      Uncharge(reserved);
      taken.arenas.clear();  // Returns the pooled arenas it took.
      return taken;
    }
    std::vector<std::unique_ptr<Arena>> fresh;
    try {
      fresh.reserve(fresh_count);
      for (std::size_t i = 0; i < sizes.size(); ++i) {
        if (taken.arenas[i]->arena_)
          continue;
        auto arena = std::make_unique<Arena>();
        arena->bytes = sizes[i];
        arena->allocation = allocate_(sizes[i]);
        if (arena->allocation.bytes.size() < sizes[i])
          throw std::bad_alloc();
        fresh.push_back(std::move(arena));
      }
    } catch (const std::exception& error) {
      taken.refused = "allocation_failed";
      taken.error = error.what();
    }
    std::size_t allocated = 0;
    {
      const std::lock_guard lock(mutex_);
      // Room reserved above; Take reserved the vector capacity.
      auto next = fresh.begin();
      for (std::size_t i = 0; i < sizes.size() && next != fresh.end(); ++i) {
        if (taken.arenas[i]->arena_)
          continue;
        auto& arena = *next++;
        allocated += arena->bytes;
        // After a failed allocation, what was allocated stays pooled.
        arena->in_use = taken.refused.empty();
        arena->freed = SnapshotClock::now();
        if (arena->in_use)
          taken.arenas[i]->arena_ = arena.get();
        arenas_.push_back(std::move(arena));
      }
      ++generation_;
    }
    changed_.notify_all();
    if (!taken.refused.empty()) {
      Uncharge(reserved - allocated);
      taken.arenas.clear();
      return taken;
    }
    taken.allocated = allocated;
    return taken;
  }

private:
  void Uncharge(std::size_t bytes) noexcept {
    const std::lock_guard lock(mutex_);
    charged_ -= bytes;
  }
  void Return(Arena& arena) noexcept {
    {
      const std::lock_guard lock(mutex_);
      arena.in_use = false;
      arena.freed = SnapshotClock::now();
      ++generation_;
    }
    changed_.notify_all();
  }
  void Trim(std::stop_token stop) {
    std::unique_lock lock(mutex_);
    while (!stop.stop_requested()) {
      if (!evicted_.empty()) {
        // Room a snapshot made: free it now.
        auto evicted = std::move(evicted_);
        evicted_.clear();
        lock.unlock();
        evicted.clear();
        lock.lock();
        continue;
      }
      std::optional<SnapshotClock::time_point> oldest;
      for (const auto& arena : arenas_)
        if (!arena->in_use && (!oldest || arena->freed < *oldest))
          oldest = arena->freed;
      const auto seen = generation_;
      const auto changed = [&] { return generation_ != seen; };
      if (!oldest) {
        changed_.wait(lock, stop, changed);
        continue;
      }
      if (SnapshotClock::now() < *oldest + idle_) {
        changed_.wait_until(lock, stop, *oldest + idle_, changed);
        continue;
      }
      std::vector<cache::StagingAllocation> released;
      std::size_t bytes = 0;
      try {
        const auto now = SnapshotClock::now();
        for (auto it = arenas_.begin(); it != arenas_.end();) {
          if ((*it)->in_use || now - (*it)->freed < idle_) {
            ++it;
            continue;
          }
          released.push_back(std::move((*it)->allocation));
          bytes += (*it)->bytes;
          charged_ -= (*it)->bytes;
          it = arenas_.erase(it);
        }
      } catch (...) {
        // No room to collect them: try again after the next change.
        changed_.wait(lock, stop, changed);
        continue;
      }
      const auto charged = charged_;
      lock.unlock();
      const auto count = released.size();
      released.clear();
      try {
        Logger::Info("cache", "event=slot_snapshot_released arenas=" +
                                  std::to_string(count) +
                                  " bytes=" + std::to_string(bytes) +
                                  " charged_bytes=" + std::to_string(charged));
      } catch (...) {
      }
      lock.lock();
    }
  }
  const std::size_t limit_;
  const std::chrono::milliseconds idle_;
  const cache::StagingAllocator allocate_;
  mutable std::mutex mutex_;
  std::condition_variable_any changed_;
  std::vector<std::unique_ptr<Arena>> arenas_;
  // Pooled arenas a snapshot gave up for room, for the trim thread to free.
  std::vector<cache::StagingAllocation> evicted_;
  // Bytes of arenas_ plus room reserved for arenas being allocated.
  std::size_t charged_{};
  std::uint64_t generation_{};
  // Last: stops and joins before the arenas are freed.
  std::jthread trimmer_;
};
}  // namespace gufo::server
#endif
