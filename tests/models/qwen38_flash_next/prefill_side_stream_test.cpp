// B6 P7 prefill side-stream policy (CPU): the CU-mask words balance whole
// WGPs over gfx1151's engines and arrays, and a dedicated stream that cannot be
// created falls back to the shared branch stream.
#include "src/models/qwen38_flash_next/kernels/rocm/prefill_side_stream.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace r = gufo::models::qwen38_flash_next::rocm;

namespace {

void Expect(bool condition, const char* what) {
  if (!condition)
    throw std::runtime_error(what);
}

}  // namespace

int main() try {
  constexpr std::uint32_t kWgps = 20;  // gfx1151 in WGP mode
  for (const std::uint32_t wgps : {1U, 2U, 4U, 6U, 10U, 19U}) {
    const auto mask = r::PrefillSideCuMask(wgps, kWgps);
    Expect(mask.size() == 1, "one mask word for 20 WGPs");
    Expect(static_cast<std::uint32_t>(std::popcount(mask[0])) == wgps,
           "one bit per selected WGP");
    Expect((mask[0] >> kWgps) == 0, "no bit past the last WGP");
    // Under the driver's round-robin dealing (bit j on engine j % 2, array
    // (j / 2) % 2) the selection is balanced to within one WGP.
    std::uint32_t per_array[4] = {};
    for (std::uint32_t j = 0; j < kWgps; ++j)
      if ((mask[0] >> j) & 1U)
        ++per_array[j % 4];
    const auto [low, high] = std::minmax_element(per_array, per_array + 4);
    Expect(*high - *low <= 1, "selected WGPs balanced over engines/arrays");
  }
  Expect(r::PrefillSideCuMask(0, kWgps).empty(), "no WGPs: no mask");
  Expect(r::PrefillSideCuMask(kWgps, kWgps).empty(), "all WGPs: no mask");
  Expect(r::PrefillSideCuMask(4, 0).empty(), "unknown device: no mask");
  Expect(r::PrefillSideCuMask(40, 64).size() == 2, "wide devices");

  int calls = 0;
  const auto fail = [&](r::PrefillSideStream) {
    ++calls;
    return false;
  };
  const auto succeed = [&](r::PrefillSideStream) {
    ++calls;
    return true;
  };
  Expect(r::ResolvePrefillSideStream(r::PrefillSideStream::kShared, fail) ==
                 r::PrefillSideStream::kShared &&
             calls == 0,
         "the shared policy creates nothing");
  Expect(r::ResolvePrefillSideStream(r::PrefillSideStream::kCuMask, fail) ==
             r::PrefillSideStream::kShared,
         "a failed creation falls back to the shared stream");
  Expect(r::ResolvePrefillSideStream(r::PrefillSideStream::kLowPriority,
                                     succeed) ==
             r::PrefillSideStream::kLowPriority,
         "a created stream keeps its policy");
  Expect(calls == 2, "one creation attempt per dedicated policy");
  std::cout << "prefill side stream policy passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
