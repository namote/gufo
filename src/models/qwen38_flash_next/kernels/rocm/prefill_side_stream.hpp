#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_PREFILL_SIDE_STREAM_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_PREFILL_SIDE_STREAM_HPP_

#include <cstdint>
#include <vector>

namespace gufo::models::qwen38_flash_next::rocm {

/// Where the EXL3 prefill shared MLP runs beside the routed experts (the
/// decode/verify side branches keep their own stream either way).
enum class PrefillSideStream {
  kShared,       ///< The decode branch stream (normal priority, all CUs).
  kLowPriority,  ///< Its own stream at the lowest HIP stream priority.
  kCuMask,       ///< Its own stream restricted to kPrefillSideWgps WGPs.
};

/// B6 P7 contention policy, chosen by measurement: a scheduling choice only,
/// every launch and value is the same. kCuMask takes the WGP count below
/// (CUs / 2 on gfx1151's 20 WGPs: 2, 4, 6 for 4, 8, 12 CUs). HIP has no
/// stream create with both a CU mask and a priority
/// (hipExtStreamCreateWithCUMask fixes normal priority), so there is no
/// combined policy.
inline constexpr PrefillSideStream kPrefillSideStream =
    PrefillSideStream::kShared;
inline constexpr std::uint32_t kPrefillSideWgps = 4;

/// hipExtStreamCreateWithCUMask words selecting `wgps` of `total` WGPs. CLR
/// in WGP mode (the gfx11 default, GPU_ENABLE_WGP_MODE) reports WGPs as
/// compute units and expands each mask bit to the WGP's CU pair
/// (rocclr/device/rocm/rocdevice.cpp), so bit j is WGP j. The driver deals
/// mask bits round-robin over shader engines, then shader arrays, in CU
/// pairs on gfx10+ (KFD's mqd_symmetrically_map_cu_mask), so the lowest
/// `wgps` bits are already balanced across engines and arrays. Empty when
/// no mask applies (wgps == 0 or wgps >= total).
inline std::vector<std::uint32_t> PrefillSideCuMask(std::uint32_t wgps,
                                                    std::uint32_t total) {
  if (wgps == 0 || total == 0 || wgps >= total)
    return {};
  std::vector<std::uint32_t> words((total + 31) / 32, 0U);
  for (std::uint32_t wgp = 0; wgp < wgps; ++wgp)
    words[wgp / 32] |= 1U << (wgp % 32);
  return words;
}

/// The policy a prefill side stream ends up with: `create(policy)` makes the
/// dedicated stream and reports success; a failed (or unsupported) creation
/// falls back to the shared branch stream, the current behaviour.
template<typename Create>
PrefillSideStream ResolvePrefillSideStream(PrefillSideStream policy,
                                           Create&& create) {
  if (policy == PrefillSideStream::kShared)
    return policy;
  return create(policy) ? policy : PrefillSideStream::kShared;
}

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_PREFILL_SIDE_STREAM_HPP_
