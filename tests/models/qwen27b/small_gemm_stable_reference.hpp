#ifndef GUFO_TESTS_MODELS_QWEN27B_SMALL_GEMM_STABLE_REFERENCE_HPP_
#define GUFO_TESTS_MODELS_QWEN27B_SMALL_GEMM_STABLE_REFERENCE_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::test::small_gemm_reference {

// Flash-Next's kernels.hpp WeightType (the values the copy switches on).
enum class WeightType : std::uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ5_1 = 7,
  kQ8_0 = 8,
  kQ4_K = 12,
  kQ5_K = 13,
  kBF16 = 30,
};

void SmallGemmStable(const void* w, WeightType type, const float* x, float* out,
                     std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
                     hipStream_t stream);

}  // namespace gufo::test::small_gemm_reference

#endif  // GUFO_TESTS_MODELS_QWEN27B_SMALL_GEMM_STABLE_REFERENCE_HPP_
