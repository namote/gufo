#ifndef GUFO_CORE_HIP_SCRATCH_ARENA_HPP_
#define GUFO_CORE_HIP_SCRATCH_ARENA_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gufo::hip {

/// Pass scratch at fixed addresses that takes memory only while a pass needs
/// it (upstream #1265, #952). Buffers are carved from reserved address space
/// once, so kernel arguments never change; MapTransient() backs every carved
/// page and ReleaseTransient() returns the pages to the driver, so the memory
/// is free again for the RAM cache and the rest of the host between such
/// passes. The arena promises nothing to any budget: a caller maps only when
/// the host has the memory to spare.
class ScratchArena {
public:
  /// Address space and the pages behind it. VmmScratchPages() is the HIP
  /// virtual-memory backend; tests supply their own.
  class Pages {
  public:
    virtual ~Pages() = default;
    /// Reserves `bytes` (a multiple of kPage) of address space aligned to
    /// kPage. Null on failure.
    [[nodiscard]] virtual void* Reserve(std::size_t bytes,
                                        std::string* error) = 0;
    virtual void Free(void* base, std::size_t bytes) noexcept = 0;
    /// Backs [at, at + bytes) with new pages; `handle` identifies them for
    /// Unmap. On failure nothing stays mapped.
    [[nodiscard]] virtual bool Map(void* at, std::size_t bytes,
                                   std::uint64_t* handle,
                                   std::string* error) = 0;
    /// Unmaps and frees the pages of one Map call.
    virtual void Unmap(void* at, std::size_t bytes,
                       std::uint64_t handle) noexcept = 0;
  };

  /// Reserves `capacity` bytes (rounded up to kPage) of address space. Null,
  /// with `error` set, when `pages` is null or the reservation fails:
  /// callers then never run passes on this scratch.
  [[nodiscard]] static std::unique_ptr<ScratchArena> Reserve(
      std::unique_ptr<Pages> pages, std::size_t capacity, std::string* error);
  ~ScratchArena();
  ScratchArena(const ScratchArena&) = delete;
  ScratchArena& operator=(const ScratchArena&) = delete;

  /// Carves `bytes` at the next multiple of `alignment` (a power of two of
  /// at most kPage) after the previous carve. Null when the reservation is
  /// exhausted or transient pages are mapped.
  [[nodiscard]] void* Take(std::size_t bytes, std::string* error,
                           std::size_t alignment = kPage);

  /// Bytes MapTransient() maps: every page a carve touches.
  [[nodiscard]] std::size_t transient_bytes() const noexcept {
    return used_pages_ * kPage;
  }
  /// Bytes carved, alignment padding included.
  [[nodiscard]] std::size_t carved_bytes() const noexcept {
    return used_bytes_;
  }
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  /// The first carved byte.
  [[nodiscard]] void* base() const noexcept { return base_; }
  [[nodiscard]] bool transient_mapped() const noexcept {
    return transient_mapped_;
  }
  /// Backs every carved page, in runs of at most kMaxRunBytes. On failure
  /// nothing stays mapped. True when already mapped.
  [[nodiscard]] bool MapTransient(std::string* error);
  /// Unmaps and frees the transient pages. The backend may wait for device
  /// work in flight: call it only where no pass uses them. No-op when
  /// nothing is mapped.
  void ReleaseTransient() noexcept;

  static constexpr std::size_t kPage = std::size_t{2} << 20;
  /// Largest single page allocation MapTransient() asks the backend for.
  static constexpr std::size_t kMaxRunBytes = std::size_t{256} << 20;

private:
  ScratchArena() = default;
  struct Run {
    std::size_t first{0};
    std::size_t pages{0};
    std::uint64_t handle{0};
  };

  std::unique_ptr<Pages> pages_;
  std::uint8_t* base_{nullptr};
  std::size_t capacity_{0};
  std::size_t used_bytes_{0};
  std::size_t used_pages_{0};
  std::vector<Run> runs_;
  bool transient_mapped_{false};
};

/// HIP virtual-memory pages on `device` (hipMemAddressReserve, hipMemCreate,
/// hipMemMap): null, with `reason` set, where the device has no virtual
/// memory management or its allocation granularity does not divide
/// ScratchArena::kPage. Defined in scratch_arena_vmm.cpp.
[[nodiscard]] std::unique_ptr<ScratchArena::Pages> VmmScratchPages(
    int device, std::string* reason);

}  // namespace gufo::hip

#endif  // GUFO_CORE_HIP_SCRATCH_ARENA_HPP_
