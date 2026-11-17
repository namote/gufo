#ifndef GUFO_SERVER_REQUEST_MEMORY_BUDGET_HPP_
#define GUFO_SERVER_REQUEST_MEMORY_BUDGET_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace gufo::server {

struct RequestMemoryClaim {
  std::size_t persistent_bytes{0};
  std::size_t scratch_bytes{0};
};

/// Memory a new claim may take now, with the inputs it came from so a
/// refusal can be explained. Inputs are absent when the caller only knows
/// the result.
struct MemoryHeadroom {
  std::size_t bytes{0};
  std::optional<std::uint64_t> host_available_bytes;
  std::uint64_t host_reserve_bytes{0};
  std::optional<std::uint64_t> device_free_bytes;
  std::uint64_t device_reserve_bytes{0};
  std::uint64_t deferred_bytes{0};
};

/// min(host available - host reserve, device free - device reserve), less the
/// deferred runtime scratch from each; zero when either is exhausted.
[[nodiscard]] inline MemoryHeadroom ComputeMemoryHeadroom(
    std::uint64_t host_available, std::uint64_t host_reserve,
    std::uint64_t device_free, std::uint64_t device_reserve,
    std::uint64_t deferred) {
  const auto left = [deferred](std::uint64_t free, std::uint64_t reserve) {
    return free > reserve && free - reserve > deferred
               ? free - reserve - deferred
               : std::uint64_t{0};
  };
  const auto bytes = std::min(
      {left(host_available, host_reserve), left(device_free, device_reserve),
       std::uint64_t{std::numeric_limits<std::size_t>::max()}});
  return {.bytes = static_cast<std::size_t>(bytes),
          .host_available_bytes = host_available,
          .host_reserve_bytes = host_reserve,
          .device_free_bytes = device_free,
          .device_reserve_bytes = device_reserve,
          .deferred_bytes = deferred};
}

/// Whether a claim fits now, may fit once memory others hold is released,
/// or cannot fit.
enum class MemoryFit : std::uint8_t { kFits, kWait, kRefused };

/// One admission check: the verdict and what it was based on.
struct MemoryCheck {
  MemoryFit fit{MemoryFit::kFits};
  RequestMemoryClaim claim;
  std::size_t reserved_bytes{0};  ///< Held by other claims.
  MemoryHeadroom headroom;
};

/// Short, client-facing; the numbers go to the server log (detail()).
inline constexpr const char* kRequestMemoryRefused =
    "image needs more memory than the server can free; try a smaller image";

class RequestMemoryRefused : public std::length_error {
public:
  explicit RequestMemoryRefused(std::string detail, bool logged = false)
      : std::length_error(kRequestMemoryRefused),
        detail_(std::move(detail)),
        logged_(logged) {}
  /// DescribeMemoryCheck of the refusal, for the server log.
  [[nodiscard]] const std::string& detail() const noexcept { return detail_; }
  /// Whether the thrower already logged detail().
  [[nodiscard]] bool logged() const noexcept { return logged_; }

private:
  std::string detail_;
  bool logged_;
};

/// key=value explanation of a check: the claim, what other claims hold, the
/// headroom and its inputs.
[[nodiscard]] inline std::string DescribeMemoryCheck(const MemoryCheck& check) {
  const auto& headroom = check.headroom;
  std::string out =
      "claim_persistent_bytes=" + std::to_string(check.claim.persistent_bytes);
  out += " claim_scratch_bytes=" + std::to_string(check.claim.scratch_bytes);
  out += " reserved_by_others_bytes=" + std::to_string(check.reserved_bytes);
  out += " headroom_bytes=" + std::to_string(headroom.bytes);
  if (headroom.host_available_bytes) {
    out += " host_available_bytes=" +
           std::to_string(*headroom.host_available_bytes);
    out += " host_reserve_bytes=" + std::to_string(headroom.host_reserve_bytes);
  }
  if (headroom.device_free_bytes) {
    out += " device_free_bytes=" + std::to_string(*headroom.device_free_bytes);
    out += " device_reserve_bytes=" +
           std::to_string(headroom.device_reserve_bytes);
  }
  if (headroom.host_available_bytes || headroom.device_free_bytes)
    out += " deferred_scratch_bytes=" + std::to_string(headroom.deferred_bytes);
  return out;
}

/// Claims are additional to CURRENT physical usage, not a startup free-memory
/// figure. Materialized buffers are conservatively charged twice until the
/// request ends; idle slots remain accounted by physical usage afterwards.
/// Serialized encoders share the largest scratch promise; input state sums.
/// `available` returns a MemoryHeadroom or a plain byte count.
class RequestMemoryBudget {
public:
  [[nodiscard]] std::size_t ReservedBytes() const {
    const std::lock_guard lock(mutex_);
    return Total({});
  }
  /// kWait while the claim does not fit but releasable memory is held:
  /// another claim here, or `releasable_elsewhere` (e.g. host memory pinned
  /// by pending cache writes). kRefused otherwise.
  template<typename Available>
  [[nodiscard]] MemoryCheck Check(RequestMemoryClaim claim,
                                  Available&& available,
                                  bool releasable_elsewhere = false) const {
    MemoryCheck check;
    check.claim = claim;
    if (claim.persistent_bytes == 0 && claim.scratch_bytes == 0)
      return check;
    const std::lock_guard lock(mutex_);
    check.headroom = Query(available);
    check.reserved_bytes = Total({});
    if (Total(claim) <= check.headroom.bytes)
      return check;
    check.fit = check.reserved_bytes != 0 || releasable_elsewhere
                    ? MemoryFit::kWait
                    : MemoryFit::kRefused;
    return check;
  }
  /// Throws RequestMemoryRefused when the claim does not fit now.
  template<typename Available>
  [[nodiscard]] std::shared_ptr<void> Acquire(RequestMemoryClaim claim,
                                              Available&& available) {
    if (claim.persistent_bytes == 0 && claim.scratch_bytes == 0)
      return {};
    auto owner = std::make_shared<RequestMemoryClaim>(claim);
    const std::lock_guard lock(mutex_);
    const auto headroom = Query(available);
    if (Total(claim) > headroom.bytes) {
      throw RequestMemoryRefused(
          "stage=acquire " + DescribeMemoryCheck({.fit = MemoryFit::kRefused,
                                                  .claim = claim,
                                                  .reserved_bytes = Total({}),
                                                  .headroom = headroom}));
    }
    std::erase_if(claims_, [](const auto& item) { return item.expired(); });
    claims_.push_back(owner);
    return owner;
  }

private:
  template<typename Available>
  static MemoryHeadroom Query(Available& available) {
    auto value = available();
    if constexpr (std::is_same_v<decltype(value), MemoryHeadroom>)
      return value;
    else {
      MemoryHeadroom headroom;
      headroom.bytes = static_cast<std::size_t>(value);
      return headroom;
    }
  }
  static std::size_t Add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
      throw std::length_error("request memory claim overflows");
    return a + b;
  }
  std::size_t Total(RequestMemoryClaim extra) const {
    for (const auto& weak : claims_) {
      if (const auto claim = weak.lock()) {
        extra.persistent_bytes =
            Add(extra.persistent_bytes, claim->persistent_bytes);
        extra.scratch_bytes =
            std::max(extra.scratch_bytes, claim->scratch_bytes);
      }
    }
    return Add(extra.persistent_bytes, extra.scratch_bytes);
  }
  mutable std::mutex mutex_;
  std::vector<std::weak_ptr<RequestMemoryClaim>> claims_;
};

}  // namespace gufo::server
#endif
