#ifndef GUFO_MODELS_QWEN_HIP_OPS_EXL3_PREFILL_BUCKET_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_EXL3_PREFILL_BUCKET_HPP_

// Device descriptor of one expert bucket tile for the EXL3 bucketed WMMA
// GEMM (LaunchExl3PrefillBucketInner, ops/prefill_fp16.hpp). HIP-free so
// model code can alias it (plan/exl3/F3-PLAN.md "F3-API").

#include <cstdint>

namespace gufo::hip {

/// One tile of an expert bucket: `count` packed rows starting at row
/// `first` of the input/output slabs, multiplied by expert `expert` of the
/// bank. count == 0 marks an inactive descriptor.
struct Exl3PrefillBucket {
  std::uint32_t expert;
  std::uint32_t count;
  std::uint32_t first;
};
static_assert(sizeof(Exl3PrefillBucket) == 3 * sizeof(std::uint32_t));

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_OPS_EXL3_PREFILL_BUCKET_HPP_
