#include "src/core/hip/kv_block_pool.hpp"

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace gufo::hip {

KvBlockPool::KvBlockPool(std::vector<KvBlockPlane> planes, std::uint32_t blocks)
    : geometry_(std::move(planes), blocks), allocator_(blocks) {
  const hipError_t status = hipMalloc(&data_, geometry_.bytes());
  if (status != hipSuccess) {
    data_ = nullptr;
    throw std::runtime_error("failed to allocate a KV block pool of " +
                             std::to_string(geometry_.bytes()) +
                             " bytes: " + hipGetErrorString(status));
  }
}

KvBlockPool::~KvBlockPool() {
  if (data_ != nullptr)
    (void)hipFree(data_);
}

}  // namespace gufo::hip
