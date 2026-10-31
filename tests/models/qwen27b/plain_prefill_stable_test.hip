// LaunchPlainPrefillStable (src/models/qwen/hip/ops/plain_prefill_stable.hpp)
// against a verbatim copy of Flash-Next's SmallGemmStable
// (small_gemm_stable_reference.hip, built with its library's flags): byte
// equality for every row count and tile position, plus an FP64 oracle, and
// the underflowed -0 chain case.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/ops/plain_prefill_stable.hpp"
#include "tests/models/qwen27b/small_gemm_stable_reference.hpp"

namespace {

using gufo::core::GgmlType;

int failures = 0;

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << "\n";
    ++failures;
  }
}

void Check(hipError_t e) {
  if (e != hipSuccess) {
    std::cerr << "hip error: " << hipGetErrorString(e) << "\n";
    std::abort();
  }
}

// The reference is the verbatim copy in small_gemm_stable_reference.hip.
namespace ref = gufo::test::small_gemm_reference;

ref::WeightType RefType(GgmlType type) {
  switch (type) {
    case GgmlType::kF32:
      return ref::WeightType::kF32;
    case GgmlType::kBF16:
      return ref::WeightType::kBF16;
    default:
      return ref::WeightType::kF16;
  }
}

// --- Host helpers ---------------------------------------------------------

double WeightValue(const std::vector<std::uint16_t>& w16,
                   const std::vector<float>& w32, GgmlType type,
                   std::size_t i) {
  switch (type) {
    case GgmlType::kF32:
      return w32[i];
    case GgmlType::kBF16:
      return std::bit_cast<float>(static_cast<std::uint32_t>(w16[i]) << 16);
    default:
      return static_cast<double>(std::bit_cast<_Float16>(w16[i]));
  }
}

void TestShape(GgmlType type, std::uint32_t n, std::uint32_t k,
               std::mt19937& rng) {
  constexpr std::uint32_t kMaxRows = 2048;
  std::normal_distribution<float> normal;
  std::vector<std::uint16_t> w16;
  std::vector<float> w32;
  if (type == GgmlType::kF32) {
    w32.resize(std::size_t(n) * k);
    for (auto& v : w32)
      v = normal(rng) * 0.05F;
  } else {
    w16.resize(std::size_t(n) * k);
    for (auto& v : w16) {
      const float f = normal(rng) * 0.05F;
      v = type == GgmlType::kBF16
              ? static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(f) >>
                                           16)
              : std::bit_cast<std::uint16_t>(static_cast<_Float16>(f));
    }
  }
  std::vector<float> x(std::size_t(kMaxRows) * k);
  for (auto& v : x)
    v = normal(rng);
  void* d_w = nullptr;
  float *d_x = nullptr, *d_ref = nullptr, *d_got = nullptr;
  const std::size_t w_bytes =
      type == GgmlType::kF32 ? w32.size() * 4 : w16.size() * 2;
  Check(hipMalloc(&d_w, w_bytes));
  Check(hipMemcpy(d_w,
                  type == GgmlType::kF32 ? static_cast<const void*>(w32.data())
                                         : static_cast<const void*>(w16.data()),
                  w_bytes, hipMemcpyHostToDevice));
  Check(hipMalloc(&d_x, x.size() * 4));
  Check(hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice));
  Check(hipMalloc(&d_ref, std::size_t(kMaxRows) * n * 4));
  Check(hipMalloc(&d_got, std::size_t(kMaxRows) * n * 4));
  const char* tname = type == GgmlType::kF32    ? "F32"
                      : type == GgmlType::kBF16 ? "BF16"
                                                : "F16";
  for (std::uint32_t rows : {1U, 8U, 9U, 64U, 65U, 2048U}) {
    // Offset by one row: the tile position of every row shifts.
    for (std::uint32_t offset : {0U, 1U}) {
      if (rows + offset > kMaxRows)
        continue;
      const float* xs = d_x + std::size_t(offset) * k;
      Check(hipMemset(d_ref, 0xff, std::size_t(rows) * n * 4));
      Check(hipMemset(d_got, 0xff, std::size_t(kMaxRows) * n * 4));
      ref::SmallGemmStable(d_w, RefType(type), xs, d_ref, rows, n, k, nullptr);
      gufo::hip::LaunchPlainPrefillStable(type, d_w, xs, d_got, rows, n, k,
                                          nullptr);
      Check(hipDeviceSynchronize());
      std::vector<float> ref(std::size_t(rows) * n),
          got(std::size_t(kMaxRows) * n);
      Check(
          hipMemcpy(ref.data(), d_ref, ref.size() * 4, hipMemcpyDeviceToHost));
      Check(
          hipMemcpy(got.data(), d_got, got.size() * 4, hipMemcpyDeviceToHost));
      std::size_t diffs = 0, past = 0;
      bool finite = true;
      for (std::size_t i = 0; i < ref.size(); ++i) {
        diffs += std::bit_cast<std::uint32_t>(ref[i]) !=
                 std::bit_cast<std::uint32_t>(got[i]);
        finite = finite && std::isfinite(ref[i]) && std::isfinite(got[i]);
      }
      for (std::size_t i = ref.size(); i < got.size(); ++i)
        past += std::bit_cast<std::uint32_t>(got[i]) != 0xffffffffU;
      // FP64 oracle on up to 9 rows (first, middle, last).
      double worst = 0.0;
      for (std::uint32_t q = 0; q < std::min(rows, 9U); ++q) {
        const std::uint32_t r =
            rows <= 9 ? q : (q < 3 ? q : (q < 6 ? rows / 2 + q : rows - 9 + q));
        for (std::uint32_t j = 0; j < n; ++j) {
          double dot = 0.0, mag = 0.0;
          for (std::uint32_t i = 0; i < k; ++i) {
            const double p =
                WeightValue(w16, w32, type, std::size_t(j) * k + i) *
                x[std::size_t(offset + r) * k + i];
            dot += p;
            mag += std::fabs(p);
          }
          worst = std::max(worst,
                           std::fabs(got[std::size_t(r) * n + j] - dot) / mag);
        }
      }
      const std::string name = std::string(tname) + " n=" + std::to_string(n) +
                               " k=" + std::to_string(k) +
                               " rows=" + std::to_string(rows) +
                               " offset=" + std::to_string(offset);
      Expect(diffs == 0, name + ": " + std::to_string(diffs) +
                             " outputs differ from SmallGemmStable");
      Expect(past == 0, name + ": wrote past the rows");
      Expect(finite, name + ": non-finite output");
      // FP32 chains of k / 32 terms plus a 32-way tree: |err| <= (k/32 + 5)
      // * 2^-24 * sum |w x| (first order), with margin.
      const double bound = (k / 32.0 + 5.0) * std::ldexp(1.0, -23);
      Expect(worst < bound, name + ": FP64 error " + std::to_string(worst));
      std::cout << name << ": byte diffs " << diffs << ", FP64 worst " << worst
                << " (bound " << bound << ")\n";
    }
  }
  Check(hipFree(d_w));
  Check(hipFree(d_x));
  Check(hipFree(d_ref));
  Check(hipFree(d_got));
}

// Every product underflows to -0: the reference's chains stay -0 (lanes
// whose steps start past k run no FMA), so the output is -0, not +0.
void TestNegativeZero() {
  constexpr std::uint32_t n = 3, k = 132, rows = 5;
  const std::vector<float> w(n * k, -0x1p-100F);
  const std::vector<float> x(rows * k, 0x1p-100F);
  float *d_w = nullptr, *d_x = nullptr, *d_ref = nullptr, *d_got = nullptr;
  Check(hipMalloc(&d_w, w.size() * 4));
  Check(hipMalloc(&d_x, x.size() * 4));
  Check(hipMalloc(&d_ref, rows * n * 4));
  Check(hipMalloc(&d_got, rows * n * 4));
  Check(hipMemcpy(d_w, w.data(), w.size() * 4, hipMemcpyHostToDevice));
  Check(hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice));
  ref::SmallGemmStable(d_w, ref::WeightType::kF32, d_x, d_ref, rows, n, k,
                       nullptr);
  gufo::hip::LaunchPlainPrefillStable(GgmlType::kF32, d_w, d_x, d_got, rows, n,
                                      k, nullptr);
  Check(hipDeviceSynchronize());
  std::vector<float> want(rows * n), got(rows * n);
  Check(hipMemcpy(want.data(), d_ref, want.size() * 4, hipMemcpyDeviceToHost));
  Check(hipMemcpy(got.data(), d_got, got.size() * 4, hipMemcpyDeviceToHost));
  std::size_t diffs = 0, negative_zero = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    diffs += std::bit_cast<std::uint32_t>(want[i]) !=
             std::bit_cast<std::uint32_t>(got[i]);
    negative_zero += std::bit_cast<std::uint32_t>(got[i]) == 0x80000000U;
  }
  Expect(diffs == 0, "-0 chains: " + std::to_string(diffs) + " outputs differ");
  std::cout << "-0 chains: byte diffs " << diffs << ", -0 outputs "
            << negative_zero << " of " << got.size() << "\n";
  for (float* p : {d_w, d_x, d_ref, d_got})
    Check(hipFree(p));
}

}  // namespace

int main() {
  std::cout << std::unitbuf;
  std::mt19937 rng(20261031);
  TestShape(GgmlType::kF16, 320, 10240, rng);  // HC down
  TestShape(GgmlType::kBF16, 67, 2597, rng);   // ragged n and k
  TestShape(GgmlType::kF32, 40, 300, rng);     // F32, k < one 512 pass
  TestShape(GgmlType::kF16, 512, 2560, rng);   // router-like
  TestNegativeZero();
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "all plain stable checks passed\n";
  return 0;
}
