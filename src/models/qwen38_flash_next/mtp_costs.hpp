#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_COSTS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_COSTS_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen38_flash_next {

// gfx1151, serving Flash-Next EXL3 c03 pack (target and MTP), K4V4 trunk
// cache, YaRN factor 4, 4224-row prefill capacity, 2026-11-20 (cells are 7-28%
// cheaper than the 2026-09-20 GGUF Q4/Q8 MTP table, whose depth slope was too
// steep: a 131072-token C1/C4 audit costs within 4% of 32768; the linear
// extrapolation below still overprices C4 width 6-8 there by about 6%).
// Milliseconds per cohort, including
// catch-up, recursive proposals and target verification. Rows are widths 1..8
// (width 1 is headless MTP catch-up plus ordinary decode). Reproduce with
// qwen38_flash_next_gpu_probe --cost-audit 0 --kv-cache-type kvarn-k4v4
// --yarn-factor 4 --context 1048576 --batch 4224. Allocation and C1 graph
// capture are warmed outside the measurement; each row is the median of three
// complete cycles. Use --depth N to audit one depth. See benchmarks.
inline constexpr float kMtpCycleMilliseconds[3][5][8] = {
    {
        // Context 0
        {35.2F, 43.6F, 51.6F, 60.7F, 66.8F, 75.7F, 83.4F, 90.2F},         // C1
        {42.1F, 55.3F, 65.2F, 79.3F, 98.6F, 109.7F, 122.8F, 133.4F},      // C2
        {52.5F, 71.9F, 97.5F, 117.3F, 143.8F, 161.5F, 189.2F, 206.3F},    // C4
        {60.5F, 97.2F, 127.5F, 154.1F, 180.3F, 212.7F, 248.6F, 272.8F},   // C6
        {70.3F, 113.1F, 147.9F, 183.6F, 217.6F, 255.3F, 298.5F, 338.7F},  // C8
    },
    {
        // Context 4096
        {36.1F, 45.1F, 53.5F, 62.2F, 68.3F, 77.7F, 87.1F, 96.3F},         // C1
        {44.5F, 59.9F, 70.3F, 84.0F, 104.6F, 117.8F, 134.2F, 145.6F},     // C2
        {56.5F, 78.3F, 106.5F, 128.5F, 156.0F, 176.0F, 204.9F, 222.3F},   // C4
        {66.5F, 108.1F, 140.7F, 171.1F, 201.5F, 233.7F, 275.5F, 299.0F},  // C6
        {78.0F, 126.4F, 165.4F, 204.5F, 241.7F, 281.4F, 332.8F, 372.6F},  // C8
    },
    {
        // Context 32768
        {36.2F, 45.1F, 53.3F, 61.9F, 68.7F, 78.7F, 87.9F, 96.2F},         // C1
        {44.7F, 59.3F, 69.3F, 84.3F, 104.7F, 117.7F, 132.9F, 143.9F},     // C2
        {57.1F, 78.9F, 106.3F, 128.8F, 157.4F, 179.4F, 207.4F, 226.8F},   // C4
        {67.0F, 108.6F, 140.8F, 172.3F, 200.7F, 234.0F, 275.9F, 303.1F},  // C6
        {79.2F, 127.1F, 165.9F, 205.2F, 245.1F, 285.9F, 333.3F, 377.8F},  // C8
    },
};

// C1 sampled-prefix head, same pack and settings, 2026-11-20 (--sampled-head
// --cost-audit 1; within 7% of the 2026-10-23 GGUF table). Fixed token
// histories at depths 0/4096/32768, median of three warmed complete cycles on
// ROCm 10.1. Full-head and concurrent costs
// above remain separate: pricing the smaller head as a full projection
// unnecessarily shortens profitable draft chains. See the model's ROCM.md.
// Full target-row transfers also retain the original curve in the engine.
inline constexpr float kMtpSampledSubsetMilliseconds[3][8] = {
    {35.3F, 42.0F, 48.6F, 56.3F, 60.7F, 68.0F, 74.9F, 80.1F},
    {36.0F, 43.5F, 50.3F, 57.4F, 62.3F, 70.5F, 78.4F, 84.4F},
    {36.2F, 43.6F, 50.2F, 57.1F, 62.7F, 71.0F, 78.9F, 85.1F},
};

// Seeded sampled policy uses fixed configured capacity for reproducibility;
// greedy batch policy uses these curves to bootstrap each physical occupancy,
// and a greedy or unseeded sampled request uses the row for the requests
// decoding with it (MtpPolicyConcurrency).
// Intermediate capacities use the next measured cohort. Context costs
// interpolate, then extrapolate the measured QSA slope to the native limit.
[[nodiscard]] inline std::array<float, 8> MtpCycleCosts(
    std::uint32_t context, std::uint32_t concurrency,
    bool sampled_subset = false) noexcept {
  const auto cohort = concurrency <= 1   ? 0
                      : concurrency <= 2 ? 1
                      : concurrency <= 4 ? 2
                      : concurrency <= 6 ? 3
                                         : 4;
  const auto interval = context <= 4096 ? 0 : 1;
  const float fraction =
      interval == 0
          ? static_cast<float>(context) / 4096.0F
          : static_cast<float>(std::min(context, 262144U) - 4096) / 28672.0F;
  const auto* low_row = sampled_subset && cohort == 0
                            ? kMtpSampledSubsetMilliseconds[interval]
                            : kMtpCycleMilliseconds[interval][cohort];
  const auto* high_row = sampled_subset && cohort == 0
                             ? kMtpSampledSubsetMilliseconds[interval + 1]
                             : kMtpCycleMilliseconds[interval + 1][cohort];
  std::array<float, 8> costs{};
  for (std::size_t i = 0; i < costs.size(); ++i) {
    const auto low = low_row[i];
    const auto high = high_row[i];
    costs[i] = low + fraction * std::max(high - low, 0.0F);
    if (fraction > 1.0F && i != 0) {
      // Independently extrapolated noisy slopes can cross at long contexts.
      // Preserve at least the last measured marginal cost of another draft.
      const auto marginal = high - high_row[i - 1];
      costs[i] = std::max(costs[i], costs[i - 1] + marginal);
    }
  }
  return costs;
}

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_COSTS_HPP_
