#ifndef GUFO_SERVER_REQUEST_MEMORY_BUDGET_HPP_
#define GUFO_SERVER_REQUEST_MEMORY_BUDGET_HPP_

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace gufo::server {

struct RequestMemoryClaim {
  std::size_t persistent_bytes{0};
  std::size_t scratch_bytes{0};
};

/// Claims are additional to CURRENT physical usage, not a startup free-memory
/// figure. Materialized buffers are conservatively charged twice until the
/// request ends; idle slots remain accounted by physical usage afterwards.
/// Serialized encoders share the largest scratch promise; input state sums.
class RequestMemoryBudget {
public:
  [[nodiscard]] std::size_t ReservedBytes() const {
    const std::lock_guard lock(mutex_);
    return Total({});
  }
  template<typename Available>
  [[nodiscard]] bool Ready(RequestMemoryClaim claim,
                           Available&& available) const {
    if (claim.persistent_bytes == 0 && claim.scratch_bytes == 0)
      return true;
    const std::lock_guard lock(mutex_);
    if (Total(claim) <= available())
      return true;
    if (Total({}) == 0)
      throw std::length_error(
          "image request exceeds available memory headroom");
    return false;
  }
  template<typename Available>
  [[nodiscard]] std::shared_ptr<void> Acquire(RequestMemoryClaim claim,
                                              Available&& available) {
    if (claim.persistent_bytes == 0 && claim.scratch_bytes == 0)
      return {};
    auto owner = std::make_shared<RequestMemoryClaim>(claim);
    const std::lock_guard lock(mutex_);
    if (Total(claim) > available())
      throw std::length_error(
          "image request exceeds available memory headroom");
    std::erase_if(claims_, [](const auto& item) { return item.expired(); });
    claims_.push_back(owner);
    return owner;
  }

private:
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
