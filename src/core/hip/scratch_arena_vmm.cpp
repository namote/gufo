// HIP virtual-memory backend of ScratchArena (upstream #1251's demand-mapped
// pages, only what the arena needs).

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <string>

#include "src/core/hip/scratch_arena.hpp"

namespace gufo::hip {
namespace {

void SetError(std::string* error, const char* what, hipError_t status) {
  if (error)
    *error = std::string(what) + ": " + hipGetErrorString(status);
}

class VmmPages final : public ScratchArena::Pages {
public:
  explicit VmmPages(int device) {
    prop_.type = hipMemAllocationTypePinned;
    prop_.location.type = hipMemLocationTypeDevice;
    prop_.location.id = device;
    access_.location = prop_.location;
    access_.flags = hipMemAccessFlagsProtReadWrite;
  }
  [[nodiscard]] const hipMemAllocationProp& prop() const noexcept {
    return prop_;
  }

  void* Reserve(std::size_t bytes, std::string* error) override {
    void* base = nullptr;
    if (const auto status =
            hipMemAddressReserve(&base, bytes, ScratchArena::kPage, nullptr, 0);
        status != hipSuccess) {
      (void)hipGetLastError();
      SetError(error, "scratch address reservation", status);
      return nullptr;
    }
    return base;
  }

  void Free(void* base, std::size_t bytes) noexcept override {
    (void)hipMemAddressFree(base, bytes);
    (void)hipGetLastError();
  }

  bool Map(void* at, std::size_t bytes, std::uint64_t* handle,
           std::string* error) override {
    hipMemGenericAllocationHandle_t pages{};
    if (const auto status = hipMemCreate(&pages, bytes, &prop_, 0);
        status != hipSuccess) {
      (void)hipGetLastError();
      SetError(error, "scratch page allocation", status);
      return false;
    }
    if (const auto status = hipMemMap(at, bytes, 0, pages, 0);
        status != hipSuccess) {
      (void)hipGetLastError();
      (void)hipMemRelease(pages);
      SetError(error, "scratch page map", status);
      return false;
    }
    if (const auto status = hipMemSetAccess(at, bytes, &access_, 1);
        status != hipSuccess) {
      (void)hipGetLastError();
      (void)hipMemUnmap(at, bytes);
      (void)hipMemRelease(pages);
      SetError(error, "scratch page access", status);
      return false;
    }
    *handle =
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pages));
    return true;
  }

  void Unmap(void* at, std::size_t bytes,
             std::uint64_t handle) noexcept override {
    (void)hipMemUnmap(at, bytes);
    (void)hipMemRelease(reinterpret_cast<hipMemGenericAllocationHandle_t>(
        static_cast<std::uintptr_t>(handle)));
    (void)hipGetLastError();
  }

private:
  hipMemAllocationProp prop_{};
  hipMemAccessDesc access_{};
};

}  // namespace

std::unique_ptr<ScratchArena::Pages> VmmScratchPages(int device,
                                                     std::string* reason) {
  const auto refuse = [&](std::string why) {
    if (reason)
      *reason = std::move(why);
    return nullptr;
  };
  int supported = 0;
  if (const auto status = hipDeviceGetAttribute(
          &supported, hipDeviceAttributeVirtualMemoryManagementSupported,
          device);
      status != hipSuccess) {
    (void)hipGetLastError();
    return refuse(std::string("vmm_query_failed:") + hipGetErrorName(status));
  }
  if (!supported)
    return refuse("vmm_unsupported");
  auto pages = std::make_unique<VmmPages>(device);
  std::size_t granularity = 0;
  if (const auto status = hipMemGetAllocationGranularity(
          &granularity, &pages->prop(), hipMemAllocationGranularityMinimum);
      status != hipSuccess) {
    (void)hipGetLastError();
    return refuse(std::string("vmm_granularity_failed:") +
                  hipGetErrorName(status));
  }
  if (granularity == 0 || ScratchArena::kPage % granularity != 0)
    return refuse("vmm_granularity_" + std::to_string(granularity));
  return pages;
}

}  // namespace gufo::hip
