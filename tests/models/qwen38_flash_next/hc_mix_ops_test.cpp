#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

constexpr std::uint32_t kTokens = 19;
constexpr std::uint32_t kHidden = 2560;
constexpr std::uint32_t kStreams = 4;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

template<typename T>
class HipBuffer {
public:
  explicit HipBuffer(std::size_t count) : count_(count) {
    void* allocation = nullptr;
    CheckHip(hipMalloc(&allocation, bytes()), "hipMalloc");
    data_ = static_cast<T*>(allocation);
  }
  ~HipBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }

  HipBuffer(const HipBuffer&) = delete;
  HipBuffer& operator=(const HipBuffer&) = delete;
  HipBuffer(HipBuffer&&) = delete;
  HipBuffer& operator=(HipBuffer&&) = delete;

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return count_ * sizeof(T);
  }

private:
  T* data_{nullptr};
  std::size_t count_{0};
};

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

std::vector<float> MakeValues(std::size_t count, std::uint32_t seed,
                              float scale) {
  std::vector<float> values(count);
  for (float& value : values) {
    value = scale *
            static_cast<float>(static_cast<int>(NextRandom(&seed) & 0xFFFFU) -
                               32768) /
            32768.0F;
  }
  return values;
}

void Upload(HipBuffer<float>* destination, const std::vector<float>& source) {
  CheckHip(hipMemcpy(destination->get(), source.data(), destination->bytes(),
                     hipMemcpyHostToDevice),
           "upload");
}

std::vector<float> Download(HipBuffer<float>* source, std::size_t count) {
  std::vector<float> values(count);
  CheckHip(hipMemcpy(values.data(), source->get(), source->bytes(),
                     hipMemcpyDeviceToHost),
           "download");
  return values;
}

// Fusing the projection must preserve every downstream representation,
// including the inject partials' reduction order and quantization ties.
void CheckFusedProjection(std::uint32_t tokens) {
  constexpr std::size_t kRank = 320;
  constexpr std::size_t kDim = kStreams * kHidden;
  const std::size_t mixed_count = static_cast<std::size_t>(tokens) * kHidden;
  const std::size_t inject_count = static_cast<std::size_t>(tokens) * kStreams *
                                   q::HcInjectPartsVec4(kHidden);
  const std::size_t q8_bytes = q::Q8TiledBytes(tokens, kHidden);
  const auto low =
      MakeValues(static_cast<std::size_t>(tokens) * kRank, 0x12345678U, 1.0F);
  const auto norm =
      MakeValues(static_cast<std::size_t>(tokens) * kDim, 0x87654321U, 0.5F);
  std::vector<__half> low_half(low.size()), norm_half(norm.size());
  std::transform(low.begin(), low.end(), low_half.begin(),
                 [](float v) { return __float2half_rn(v); });
  std::transform(norm.begin(), norm.end(), norm_half.begin(),
                 [](float v) { return __float2half_rn(v); });
  std::vector<std::uint8_t> weights(kDim * (kRank / 32) * 34);
  std::uint32_t seed = 0xABCD0123U;
  for (std::size_t block = 0; block < weights.size() / 34; ++block) {
    const __half scale = __float2half_rn(
        static_cast<float>(1 + NextRandom(&seed) % 13) / 2048.0F);
    std::memcpy(weights.data() + block * 34, &scale, sizeof(scale));
    for (std::size_t j = 0; j < 32; ++j) {
      weights[block * 34 + 2 + j] = static_cast<std::uint8_t>(
          static_cast<int>(NextRandom(&seed) % 255) - 127);
    }
  }
  HipBuffer<std::uint8_t> d_up(weights.size());
  HipBuffer<__half> d_low(low.size()), d_norm(norm.size());
  HipBuffer<float> d_inject_w(kStreams * kDim);
  HipBuffer<float> d_gate(norm.size()), d_ref(mixed_count), d_out(mixed_count);
  HipBuffer<__half> d_ref_half(mixed_count), d_out_half(mixed_count);
  HipBuffer<std::uint8_t> d_ref_q8(q8_bytes), d_out_q8(q8_bytes);
  HipBuffer<float> d_ref_inject(inject_count), d_out_inject(inject_count);
  CheckHip(hipMemcpy(d_up.get(), weights.data(), d_up.bytes(),
                     hipMemcpyHostToDevice),
           "up weights");
  CheckHip(hipMemcpy(d_low.get(), low_half.data(), d_low.bytes(),
                     hipMemcpyHostToDevice),
           "low rank input");
  CheckHip(hipMemcpy(d_norm.get(), norm_half.data(), d_norm.bytes(),
                     hipMemcpyHostToDevice),
           "normalized streams");
  Upload(&d_inject_w, MakeValues(kStreams * kDim, 0xDEADBEEFU, 0.1F));
  CheckHip(hipMemset(d_ref_q8.get(), 0, q8_bytes), "clear reference Q8");
  CheckHip(hipMemset(d_out_q8.get(), 0, q8_bytes), "clear fused Q8");
  if (!q::DenseF16Gemm(d_up.get(), d_low.get(), d_gate.get(), tokens, kDim,
                       kRank, nullptr)) {
    throw std::runtime_error("HC reference projection rejected");
  }
  q::HcMixEpilogueVec4F16(d_norm.get(), d_gate.get(), d_inject_w.get(),
                          d_ref.get(), d_ref_half.get(), d_ref_q8.get(),
                          d_ref_inject.get(), tokens, kHidden, nullptr);
  const auto exact = [](auto& reference, auto& actual, const char* name) {
    std::vector<std::uint8_t> a(reference.bytes()), b(actual.bytes());
    CheckHip(
        hipMemcpy(a.data(), reference.get(), a.size(), hipMemcpyDeviceToHost),
        "reference download");
    CheckHip(hipMemcpy(b.data(), actual.get(), b.size(), hipMemcpyDeviceToHost),
             "fused download");
    if (a != b) {
      throw std::runtime_error(std::string("HC fusion changed ") + name);
    }
  };
  // Small case covers all optional outputs; the large ragged case also
  // replays the full path to catch cross-block races and tail writes.
  const unsigned first = tokens == 96 ? 0 : 7;
  const unsigned end = tokens == 96 ? 8 : 9;
  for (unsigned iteration = first; iteration < end; ++iteration) {
    const unsigned outputs = std::min(iteration, 7U);
    if (!q::HcMixF16Gemm(d_up.get(), d_low.get(), d_norm.get(),
                         outputs & 4 ? d_inject_w.get() : nullptr, d_out.get(),
                         outputs & 1 ? d_out_half.get() : nullptr,
                         outputs & 2 ? d_out_q8.get() : nullptr,
                         outputs & 4 ? d_out_inject.get() : nullptr, tokens,
                         kHidden, kRank, nullptr)) {
      throw std::runtime_error("HC fused projection rejected");
    }
    exact(d_ref, d_out, "F32 mixed row");
    if (outputs & 1)
      exact(d_ref_half, d_out_half, "F16 mixed row");
    if (outputs & 2)
      exact(d_ref_q8, d_out_q8, "Q8 mixed row");
    if (outputs & 4)
      exact(d_ref_inject, d_out_inject, "inject partials");
  }
  if (!q::HcMixF16Gemm(d_up.get(), d_low.get(), d_norm.get(), d_inject_w.get(),
                       nullptr, d_out_half.get(), nullptr, d_out_inject.get(),
                       tokens, kHidden, kRank, nullptr))
    throw std::runtime_error("HC F16-only projection rejected");
  exact(d_ref_half, d_out_half, "F16-only mixed row");
  exact(d_ref_inject, d_out_inject, "F16-only inject partials");
  std::cerr << "HC projection fusion exact at " << tokens << " tokens\n";
}

}  // namespace

/// Random Q8_0 blocks (F16 scale, 32 codes) for `blocks` blocks; every
/// 17th block has a zero scale.
std::vector<std::uint8_t> MakeQ8_0(std::size_t blocks, std::uint32_t seed) {
  std::vector<std::uint8_t> bytes(blocks * 34);
  for (std::size_t b = 0; b < blocks; ++b) {
    const float d =
        b % 17 == 3 ? 0.0F
                    : 1.0e-3F * static_cast<float>(1 + NextRandom(&seed) % 64);
    const __half h = __float2half(d);
    std::memcpy(bytes.data() + b * 34, &h, 2);
    for (int i = 0; i < 32; ++i)
      bytes[b * 34 + 2 + i] = static_cast<std::uint8_t>(NextRandom(&seed));
  }
  return bytes;
}

template<typename T>
std::vector<T> DownloadRaw(const T* source, std::size_t count) {
  std::vector<T> values(count);
  CheckHip(hipMemcpy(values.data(), source, count * sizeof(T),
                     hipMemcpyDeviceToHost),
           "download");
  return values;
}

// The fused decode HC projections (down + SiLU/Q8_1, up + mix epilogue)
// against qfn_mmq's GEMVs, SiluScaleQ8_1 and HcMixEpilogue, byte for byte,
// for every decode width, twice (arrival counters wrap back to zero), with
// and without inject partials and for zero, two and three rotation slabs.
void CheckFusedDecodeHc(bool with_inject, std::uint32_t slab_count) {
  constexpr std::uint32_t kRank = 320;
  constexpr std::uint32_t kHcDim = kHidden * kStreams;
  constexpr std::uint32_t kRows = q::kHcDecodeMaxRows;
  const std::uint32_t parts = q::HcInjectParts(kHidden);
  const auto down = MakeQ8_0(std::size_t{kRank} * kHcDim / 32, 0x51EDU);
  const auto up = MakeQ8_0(std::size_t{kHcDim} * kRank / 32, 0x7A11U);
  auto xn = MakeValues(std::size_t{kRows} * kHcDim, 0xC0DEU, 0.5F);
  // A zero block, a signed zero and subnormals in the norm rows.
  for (std::uint32_t i = 0; i < 32; ++i)
    xn[64 + i] = 0.0F;
  xn[200] = -0.0F;
  xn[201] = 1.0e-40F;
  xn[202] = -2.0e-39F;
  const auto inject_w =
      MakeValues(std::size_t{kStreams} * kHcDim, 0xFEEDU, 0.1F);
  std::vector<std::uint16_t> suh(kHidden);
  std::uint32_t seed = 0x5EEDU;
  for (auto& h : suh) {
    const __half v = __float2half(NextRandom(&seed) & 1U ? 1.0F : -1.0F);
    std::memcpy(&h, &v, 2);
  }
  const std::size_t xq_bytes = qfn_mmq_q8_1_bytes(kRows, kHcDim);
  const std::size_t lo_bytes = qfn_mmq_q8_1_bytes(kRows, kRank);
  const std::size_t mixed_bytes = qfn_mmq_q8_1_bytes(kRows, kHidden);
  const std::uint32_t lo_cols =
      static_cast<std::uint32_t>(qfn_mmq_q8_1_bytes(1, kRank) / 36 * 32);
  const std::size_t slab = std::size_t{3} * 8 * kHidden;  // three consumers
  void *d_down, *d_up, *d_xq, *d_suh;
  CheckHip(hipMalloc(&d_down, down.size()), "hipMalloc");
  CheckHip(hipMalloc(&d_up, up.size()), "hipMalloc");
  CheckHip(hipMalloc(&d_xq, xq_bytes), "hipMalloc");
  CheckHip(hipMalloc(&d_suh, suh.size() * 2), "hipMalloc");
  CheckHip(hipMemcpy(d_down, down.data(), down.size(), hipMemcpyHostToDevice),
           "upload");
  CheckHip(hipMemcpy(d_up, up.data(), up.size(), hipMemcpyHostToDevice),
           "upload");
  CheckHip(hipMemcpy(d_suh, suh.data(), suh.size() * 2, hipMemcpyHostToDevice),
           "upload");
  HipBuffer<float> d_xn(std::size_t{kRows} * kHcDim);
  HipBuffer<float> d_inject_w(std::size_t{kStreams} * kHcDim);
  Upload(&d_xn, xn);
  Upload(&d_inject_w, inject_w);
  struct Outputs {
    HipBuffer<float> lo{std::size_t{kRows} * kRank};
    HipBuffer<float> gate{std::size_t{kRows} * kHcDim};
    HipBuffer<float> mixed{std::size_t{kRows} * kHidden};
    HipBuffer<float> inject{std::size_t{kRows} * kStreams * 16};
    HipBuffer<std::uint8_t> lo_q8;
    HipBuffer<std::uint8_t> mixed_q8;
    HipBuffer<std::uint16_t> slabs;
    Outputs(std::size_t lo_b, std::size_t mixed_b, std::size_t slab_n)
        : lo_q8(lo_b), mixed_q8(mixed_b), slabs(slab_n) {}
  };
  Outputs ref(lo_bytes, mixed_bytes, slab), fused(lo_bytes, mixed_bytes, slab);
  for (Outputs* o : {&ref, &fused}) {
    CheckHip(hipMemset(o->lo_q8.get(), 0, lo_bytes), "memset");
    CheckHip(hipMemset(o->mixed_q8.get(), 0, mixed_bytes), "memset");
    CheckHip(hipMemset(o->slabs.get(), 0, slab * 2), "memset");
  }
  std::uint32_t* d_arrivals;
  CheckHip(hipMalloc(&d_arrivals, q::kHcArrivals * 4), "hipMalloc");
  CheckHip(hipMemset(d_arrivals, 0, q::kHcArrivals * 4), "memset");
  const float scale = 1.0F / kStreams;
  const float* inject_w_ptr = with_inject ? d_inject_w.get() : nullptr;
  bool ok = true;
  std::vector<float> row0_lo, row0_gate;
  for (int pass = 0; pass < 2; ++pass) {
    for (std::uint32_t rows = 1; rows <= kRows; ++rows) {
      if (qfn_mmq_quantize_q8_1(d_xn.get(), d_xq, static_cast<int>(rows),
                                static_cast<int>(kHcDim), nullptr) != 0)
        throw std::runtime_error("quantize_q8_1 failed");
      const auto slabs_of = [&](Outputs& o) {
        q::Exl3InputSlabs s{};
        s.suh[0] = static_cast<const std::uint16_t*>(d_suh);
        s.suh[1] = static_cast<const std::uint16_t*>(d_suh);
        s.suh[2] = static_cast<const std::uint16_t*>(d_suh);
        s.out = o.slabs.get();
        s.count = slab_count;
        s.pass_rows = rows;
        s.k = kHidden;
        return s;
      };
      // Reference: the four separate launches.
      if (qfn_mmq_q8_0_dense_vec_preq(
              d_down, nullptr, d_xq, ref.lo.get(), static_cast<int>(kRank),
              static_cast<int>(rows), static_cast<int>(kHcDim), nullptr) != 0 ||
          !q::SiluScaleQ8_1(ref.lo.get(), scale, rows, kRank, lo_cols,
                            ref.lo_q8.get(), nullptr) ||
          qfn_mmq_q8_0_dense_vec_preq(d_up, nullptr, ref.lo_q8.get(),
                                      ref.gate.get(), static_cast<int>(kHcDim),
                                      static_cast<int>(rows),
                                      static_cast<int>(kRank), nullptr) != 0)
        throw std::runtime_error("reference HC projections failed");
      const bool ref_rot = q::HcMixEpilogue(
          d_xn.get(), ref.gate.get(), inject_w_ptr, ref.mixed.get(),
          ref.inject.get(), rows, kHidden, kStreams, nullptr,
          ref.mixed_q8.get(), slabs_of(ref));
      q::HcDownSiluQ8_1(d_down, d_xq, fused.lo.get(), scale, rows, lo_cols,
                        fused.lo_q8.get(), d_arrivals, nullptr);
      const bool fused_rot = q::HcUpMix(
          d_up, fused.lo_q8.get(), lo_cols / 32, kRank, fused.gate.get(),
          d_xn.get(), inject_w_ptr, fused.mixed.get(), fused.inject.get(), rows,
          kHidden, nullptr, fused.mixed_q8.get(), slabs_of(fused),
          d_arrivals + 1);
      CheckHip(hipDeviceSynchronize(), "fused decode HC");
      const auto same = [](const auto& a, const auto& b) {
        return a.size() == b.size() &&
               std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0;
      };
      const bool lo_ok = same(DownloadRaw(ref.lo.get(), rows * kRank),
                              DownloadRaw(fused.lo.get(), rows * kRank)) &&
                         same(DownloadRaw(ref.lo_q8.get(), lo_bytes),
                              DownloadRaw(fused.lo_q8.get(), lo_bytes));
      const bool gate_ok = same(DownloadRaw(ref.gate.get(), rows * kHcDim),
                                DownloadRaw(fused.gate.get(), rows * kHcDim));
      const bool mix_ok =
          ref_rot == (slab_count != 0) && fused_rot == ref_rot &&
          same(DownloadRaw(ref.mixed.get(), rows * kHidden),
               DownloadRaw(fused.mixed.get(), rows * kHidden)) &&
          same(DownloadRaw(ref.mixed_q8.get(), mixed_bytes),
               DownloadRaw(fused.mixed_q8.get(), mixed_bytes)) &&
          (!with_inject ||
           same(DownloadRaw(ref.inject.get(), rows * kStreams * parts),
                DownloadRaw(fused.inject.get(), rows * kStreams * parts))) &&
          same(DownloadRaw(ref.slabs.get(), slab),
               DownloadRaw(fused.slabs.get(), slab));
      const auto counters = DownloadRaw(d_arrivals, q::kHcArrivals);
      const bool counters_ok =
          std::all_of(counters.begin(), counters.end(),
                      [](std::uint32_t v) { return v == 0; });
      // Split invariance: row 0 must not depend on the launch width.
      const auto lo_all = DownloadRaw(fused.lo.get(), rows * kRank);
      const auto gate_all = DownloadRaw(fused.gate.get(), rows * kHcDim);
      if (rows == 1) {
        row0_lo.assign(lo_all.begin(), lo_all.begin() + kRank);
        row0_gate.assign(gate_all.begin(), gate_all.begin() + kHcDim);
      }
      const bool split_ok =
          std::equal(row0_lo.begin(), row0_lo.end(), lo_all.begin(),
                     [](float a, float b) {
                       return std::memcmp(&a, &b, sizeof a) == 0;
                     }) &&
          std::equal(row0_gate.begin(), row0_gate.end(), gate_all.begin(),
                     [](const auto& a, const auto& b) {
                       return std::memcmp(&a, &b, sizeof a) == 0;
                     });
      if (!lo_ok || !gate_ok || !mix_ok || !counters_ok || !split_ok) {
        std::cerr << "fused decode HC inject " << with_inject << " slabs "
                  << slab_count << " rows " << rows << " pass " << pass
                  << ": lo " << lo_ok << " gate " << gate_ok << " mix "
                  << mix_ok << " counters " << counters_ok << " split "
                  << split_ok << '\n';
        ok = false;
      }
    }
  }
  (void)hipFree(d_arrivals);
  (void)hipFree(d_down);
  (void)hipFree(d_up);
  (void)hipFree(d_xq);
  (void)hipFree(d_suh);
  if (!ok)
    throw std::runtime_error("fused decode HC projections differ");
  std::cout << "fused decode HC projections (inject " << with_inject
            << ", slabs " << slab_count << "): rows 1-8 byte-identical\n";
}

int main() {
  try {
    CheckFusedDecodeHc(true, 2);
    CheckFusedDecodeHc(false, 0);
    CheckFusedDecodeHc(true, 3);
    CheckFusedProjection(96);
    CheckFusedProjection(2049);
    constexpr std::size_t kHcDim = static_cast<std::size_t>(kHidden) * kStreams;
    constexpr std::size_t kRows = static_cast<std::size_t>(kTokens) * kHcDim;
    constexpr std::size_t kMixed = static_cast<std::size_t>(kTokens) * kHidden;
    const std::uint32_t scalar_parts = q::HcInjectParts(kHidden);
    const std::uint32_t vec_parts = q::HcInjectPartsVec4(kHidden);
    const std::size_t scalar_inject =
        static_cast<std::size_t>(kTokens) * kStreams * scalar_parts;
    const std::size_t vec_inject =
        static_cast<std::size_t>(kTokens) * kStreams * vec_parts;

    const auto xn = MakeValues(kRows, 0x1234ABCDU, 0.5F);
    const auto gate = MakeValues(kRows, 0xBADC0FFEU, 2.0F);
    const auto inject_w = MakeValues(kStreams * kHcDim, 0xDEADBEEFU, 0.1F);
    HipBuffer<float> d_xn(kRows);
    HipBuffer<float> d_gate(kRows);
    HipBuffer<float> d_weight(kStreams * kHcDim);
    HipBuffer<float> d_mixed_scalar(kMixed);
    HipBuffer<float> d_mixed_vec(kMixed);
    HipBuffer<float> d_inject_scalar(scalar_inject);
    HipBuffer<float> d_inject_vec(vec_inject);
    Upload(&d_xn, xn);
    Upload(&d_gate, gate);
    Upload(&d_weight, inject_w);

    q::HcMixEpilogue(d_xn.get(), d_gate.get(), d_weight.get(),
                     d_mixed_scalar.get(), d_inject_scalar.get(), kTokens,
                     kHidden, kStreams, nullptr);
    q::HcMixEpilogueVec4(d_xn.get(), d_gate.get(), d_weight.get(),
                         d_mixed_vec.get(), d_inject_vec.get(), kTokens,
                         kHidden, kStreams, nullptr);
    CheckHip(hipDeviceSynchronize(), "HC mix synchronization");

    const auto mixed_scalar = Download(&d_mixed_scalar, kMixed);
    const auto mixed_vec = Download(&d_mixed_vec, kMixed);
    const auto inject_scalar = Download(&d_inject_scalar, scalar_inject);
    const auto inject_vec = Download(&d_inject_vec, vec_inject);
    double worst_mixed = 0.0;
    for (std::size_t i = 0; i < kMixed; ++i) {
      worst_mixed = std::max(
          worst_mixed,
          std::abs(static_cast<double>(mixed_scalar[i] - mixed_vec[i])));
    }
    double worst_inject = 0.0;
    for (std::uint32_t t = 0; t < kTokens; ++t) {
      for (std::uint32_t s = 0; s < kStreams; ++s) {
        double scalar = 0.0;
        double vectorized = 0.0;
        const std::size_t scalar_base =
            (static_cast<std::size_t>(t) * kStreams + s) * scalar_parts;
        const std::size_t vec_base =
            (static_cast<std::size_t>(t) * kStreams + s) * vec_parts;
        for (std::uint32_t p = 0; p < scalar_parts; ++p) {
          scalar += inject_scalar[scalar_base + p];
        }
        for (std::uint32_t p = 0; p < vec_parts; ++p) {
          vectorized += inject_vec[vec_base + p];
        }
        const double scale = std::max(1.0, std::abs(scalar));
        worst_inject =
            std::max(worst_inject, std::abs(scalar - vectorized) / scale);
      }
    }
    std::cerr << "HC mix vec4 worst mixed absolute error " << worst_mixed
              << ", inject relative error " << worst_inject << std::endl;
    if (worst_mixed > 1e-6 || worst_inject > 5e-5) {
      return 1;
    }

    // F16 mixer input route: the combine writes the next mixer's norm as
    // F16 and the epilogue reads it back. The residual update must stay
    // bit-identical; the norm and the mixed output carry F16 rounding.
    constexpr float kEps = 1e-6F;
    const auto res = MakeValues(kRows, 0x0F0F0F0FU, 1.0F);
    const auto block_out = MakeValues(kMixed, 0xF0F0F0F0U, 1.0F);
    const auto gamma = MakeValues(kHcDim, 0x600DCAFEU, 0.5F);
    HipBuffer<float> d_res_ref(kRows);
    HipBuffer<float> d_res_f16(kRows);
    HipBuffer<float> d_block_out(kMixed);
    HipBuffer<float> d_gamma(kHcDim);
    HipBuffer<float> d_xn_ref(kRows);
    HipBuffer<__half> d_xn_f16(kRows);
    HipBuffer<float> d_inject_ref(vec_inject);
    Upload(&d_res_ref, res);
    Upload(&d_res_f16, res);
    Upload(&d_block_out, block_out);
    Upload(&d_gamma, gamma);
    q::HcCombine(d_res_ref.get(), d_block_out.get(), d_inject_vec.get(),
                 vec_parts, d_gamma.get(), d_xn_ref.get(), kTokens, kHidden,
                 kStreams, kEps, nullptr);
    const std::size_t q8_bytes = q::Q8TiledBytes(kTokens, kHcDim);
    HipBuffer<std::uint8_t> d_xn_q8(q8_bytes);
    q::HcCombineF16(d_res_f16.get(), d_block_out.get(), d_inject_vec.get(),
                    vec_parts, d_gamma.get(), d_xn_f16.get(), d_xn_q8.get(),
                    kTokens, kHidden, kStreams, kEps, nullptr);
    q::HcMixEpilogueVec4(d_xn_ref.get(), d_gate.get(), d_weight.get(),
                         d_mixed_scalar.get(), d_inject_ref.get(), kTokens,
                         kHidden, kStreams, nullptr);
    q::HcMixEpilogueVec4F16(d_xn_f16.get(), d_gate.get(), d_weight.get(),
                            d_mixed_vec.get(), nullptr, nullptr,
                            d_inject_vec.get(), kTokens, kHidden, nullptr);
    CheckHip(hipDeviceSynchronize(), "HC F16 synchronization");
    const auto res_ref = Download(&d_res_ref, kRows);
    const auto res_f16 = Download(&d_res_f16, kRows);
    const auto xn_ref = Download(&d_xn_ref, kRows);
    // CPU reference of the combine (residual update and grouped norm) over
    // the same inject partials.
    {
      double worst_cpu_res = 0.0;
      double worst_cpu_xn = 0.0;
      for (std::uint32_t t = 0; t < kTokens; ++t) {
        for (std::uint32_t s = 0; s < kStreams; ++s) {
          double logit = 0.0;
          for (std::uint32_t p = 0; p < vec_parts; ++p) {
            logit += inject_vec[(static_cast<std::size_t>(t) * kStreams + s) *
                                    vec_parts +
                                p];
          }
          const double w =
              2.0 / (1.0 + std::exp(-logit / static_cast<double>(kStreams)));
          const std::size_t base =
              (static_cast<std::size_t>(t) * kStreams + s) * kHidden;
          std::vector<double> v(kHidden);
          double ss = 0.0;
          for (std::uint32_t i = 0; i < kHidden; ++i) {
            v[i] = res[base + i] +
                   block_out[static_cast<std::size_t>(t) * kHidden + i] * w;
            ss += v[i] * v[i];
          }
          const double scale = 1.0 / std::sqrt(ss / kHidden + kEps);
          for (std::uint32_t i = 0; i < kHidden; ++i) {
            worst_cpu_res =
                std::max(worst_cpu_res, std::abs(v[i] - res_ref[base + i]));
            const double n = v[i] * scale * gamma[s * kHidden + i];
            worst_cpu_xn =
                std::max(worst_cpu_xn, std::abs(n - xn_ref[base + i]) /
                                           std::max(1e-2, std::abs(n)));
          }
        }
      }
      std::cerr << "HC combine vs CPU: residual " << worst_cpu_res << ", norm "
                << worst_cpu_xn << std::endl;
      if (worst_cpu_res > 1e-4 || worst_cpu_xn > 1e-3) {
        return 1;
      }
    }
    std::vector<__half> xn_f16(kRows);
    CheckHip(hipMemcpy(xn_f16.data(), d_xn_f16.get(), kRows * sizeof(__half),
                       hipMemcpyDeviceToHost),
             "download");
    const auto mixed_ref = Download(&d_mixed_scalar, kMixed);
    const auto mixed_f16 = Download(&d_mixed_vec, kMixed);
    const auto inject_ref = Download(&d_inject_ref, vec_inject);
    const auto inject_f16 = Download(&d_inject_vec, vec_inject);
    double worst_res = 0.0;
    double worst_xn = 0.0;
    for (std::size_t i = 0; i < kRows; ++i) {
      worst_res = std::max(
          worst_res, std::abs(static_cast<double>(res_ref[i] - res_f16[i])));
      const double scale =
          std::max(1e-2, std::abs(static_cast<double>(xn_ref[i])));
      worst_xn = std::max(
          worst_xn,
          std::abs(static_cast<double>(xn_ref[i] - __half2float(xn_f16[i]))) /
              scale);
    }
    // The mix sums four streams of either sign, so it is scored absolutely
    // on values of order one.
    double worst_mixed_f16 = 0.0;
    for (std::size_t i = 0; i < kMixed; ++i) {
      worst_mixed_f16 =
          std::max(worst_mixed_f16,
                   std::abs(static_cast<double>(mixed_ref[i] - mixed_f16[i])));
    }
    double worst_inject_f16 = 0.0;
    for (std::uint32_t t = 0; t < kTokens; ++t) {
      for (std::uint32_t s = 0; s < kStreams; ++s) {
        double reference = 0.0;
        double candidate = 0.0;
        const std::size_t base =
            (static_cast<std::size_t>(t) * kStreams + s) * vec_parts;
        for (std::uint32_t p = 0; p < vec_parts; ++p) {
          reference += inject_ref[base + p];
          candidate += inject_f16[base + p];
        }
        const double scale = std::max(1.0, std::abs(reference));
        worst_inject_f16 =
            std::max(worst_inject_f16, std::abs(reference - candidate) / scale);
      }
    }
    // The fused tiled-Q8 norm: dequantize each 32-block back and compare
    // with the F32 norm (8-bit rounding of the block's absmax).
    std::vector<std::uint8_t> xn_q8(q8_bytes);
    CheckHip(
        hipMemcpy(xn_q8.data(), d_xn_q8.get(), q8_bytes, hipMemcpyDeviceToHost),
        "download");
    double worst_q8 = 0.0;
    for (std::uint32_t t = 0; t < kTokens; ++t) {
      for (std::size_t kb = 0; kb < kHcDim / 32; ++kb) {
        const std::uint8_t* tile =
            xn_q8.data() + (((t / 16) * (kHcDim / 32)) + kb) * 576;
        float d = 0.0F;
        std::memcpy(&d, tile + 512 + (t % 16) * 4, 4);
        for (std::size_t j = 0; j < 32; ++j) {
          const auto code = static_cast<std::int8_t>(
              tile[((j / 16) * 256) + ((t % 16) * 16) + (j % 16)]);
          const double ref =
              xn_ref[static_cast<std::size_t>(t) * kHcDim + kb * 32 + j];
          const double err = std::abs(ref - d * static_cast<double>(code));
          // One rounding step of the block's absmax / 127 grid.
          worst_q8 = std::max(
              worst_q8, err / std::max(1e-3, static_cast<double>(std::abs(d))));
        }
      }
    }
    std::cout << "HC F16 mixer input: residual absolute error " << worst_res
              << ", tiled Q8 norm worst error " << worst_q8
              << " quantization steps\n";
    std::cout << "HC F16 mixer input: residual absolute error " << worst_res
              << ", norm relative error " << worst_xn
              << ", mixed absolute error " << worst_mixed_f16
              << ", inject relative error " << worst_inject_f16 << '\n';
    if (worst_res != 0.0 || worst_xn > 2e-3 || worst_mixed_f16 > 1e-3 ||
        worst_inject_f16 > 4e-3 || worst_q8 > 0.51) {
      return 1;
    }

    // Side outputs are independent: Q8 must also work without an F16
    // output or inject weights, including a one-token batch.
    for (const auto count : {1U, kTokens}) {
      const std::size_t mixed_count = static_cast<std::size_t>(count) * kHidden;
      const auto bytes = q::Q8TiledBytes(count, kHidden);
      HipBuffer<__half> d_half(mixed_count);
      HipBuffer<std::uint8_t> d_q8(bytes);
      for (unsigned outputs = 0; outputs < 8; ++outputs) {
        const bool half_output = (outputs & 1U) != 0;
        const bool q8_output = (outputs & 2U) != 0;
        const bool inject_output = (outputs & 4U) != 0;
        CheckHip(hipMemset(d_q8.get(), 0xA5, bytes), "poison Q8 output");
        q::HcMixEpilogueVec4F16(d_xn_f16.get(), d_gate.get(),
                                inject_output ? d_weight.get() : nullptr,
                                d_mixed_vec.get(),
                                half_output ? d_half.get() : nullptr,
                                q8_output ? d_q8.get() : nullptr,
                                inject_output ? d_inject_vec.get() : nullptr,
                                count, kHidden, nullptr);
        const auto actual = Download(&d_mixed_vec, kMixed);
        if (std::memcmp(actual.data(), mixed_f16.data(),
                        mixed_count * sizeof(float)) != 0) {
          throw std::runtime_error("optional outputs changed the mixed row");
        }
        if (inject_output) {
          const auto actual_inject = Download(&d_inject_vec, vec_inject);
          if (std::memcmp(actual_inject.data(), inject_f16.data(),
                          static_cast<std::size_t>(count) * kStreams *
                              vec_parts * sizeof(float)) != 0) {
            throw std::runtime_error("mixer replay changed inject partials");
          }
        }
        if (half_output) {
          std::vector<__half> half(mixed_count);
          CheckHip(hipMemcpy(half.data(), d_half.get(), d_half.bytes(),
                             hipMemcpyDeviceToHost),
                   "download mixed F16");
          for (std::size_t i = 0; i < mixed_count; ++i) {
            if (__half2float(half[i]) !=
                __half2float(__float2half(mixed_f16[i]))) {
              throw std::runtime_error("mixed F16 rounding differs");
            }
          }
        }
        if (q8_output) {
          std::vector<std::uint8_t> packed(bytes);
          CheckHip(hipMemcpy(packed.data(), d_q8.get(), bytes,
                             hipMemcpyDeviceToHost),
                   "download mixed Q8");
          for (std::uint32_t t = 0; t < count; ++t) {
            for (std::uint32_t kb = 0; kb < kHidden / 32; ++kb) {
              const auto* tile =
                  packed.data() + ((t / 16) * (kHidden / 32) + kb) * 576;
              float scale;
              std::memcpy(&scale, tile + 512 + (t % 16) * 4, 4);
              if (!std::isfinite(scale) || scale < 0.0F) {
                throw std::runtime_error("invalid mixed Q8 scale");
              }
              for (std::uint32_t j = 0; j < 32; ++j) {
                const auto code = static_cast<std::int8_t>(
                    tile[(j / 16) * 256 + (t % 16) * 16 + j % 16]);
                const float expected =
                    mixed_f16[static_cast<std::size_t>(t) * kHidden + kb * 32 +
                              j];
                if (std::abs(expected - scale * code) >
                    0.501F * scale + 1e-7F) {
                  throw std::runtime_error("mixed Q8 rounding differs");
                }
              }
            }
          }
        }
      }
    }
    for (const std::uint32_t chunk : {1U, 8U, 9U, 16U}) {
      Upload(&d_res_f16, res);
      Upload(&d_inject_vec, inject_vec);
      for (std::uint32_t off = 0; off < kTokens; off += chunk) {
        q::HcCombineF16(
            d_res_f16.get() + std::size_t(off) * kHcDim,
            d_block_out.get() + std::size_t(off) * kHidden,
            d_inject_vec.get() + std::size_t(off) * kStreams * vec_parts,
            vec_parts, d_gamma.get(),
            d_xn_f16.get() + std::size_t(off) * kHcDim, nullptr,
            std::min(chunk, kTokens - off), kHidden, kStreams, kEps, nullptr);
      }
      CheckHip(hipDeviceSynchronize(), "short HC prefill");
      const auto actual_res = Download(&d_res_f16, kRows);
      std::vector<__half> actual_norm(kRows);
      CheckHip(hipMemcpy(actual_norm.data(), d_xn_f16.get(),
                         kRows * sizeof(__half), hipMemcpyDeviceToHost),
               "short HC norm");
      if (actual_res != res_f16 ||
          std::memcmp(actual_norm.data(), xn_f16.data(),
                      kRows * sizeof(__half)) != 0)
        throw std::runtime_error("HC prefill depends on chunk width");
    }
    // EXL3 prefill: the combine writes the F16 norm and its stream scales
    // instead of the F32 norm, and the mixer recomputes that norm from the
    // residual. Everything must match the F32 route byte for byte.
    {
      const auto bytes_equal = [](const void* x, const void* y, std::size_t n) {
        return std::memcmp(x, y, n) == 0;
      };
      Upload(&d_inject_vec, inject_vec);
      HipBuffer<float> d_res_a(kRows);
      HipBuffer<float> d_res_b(kRows);
      HipBuffer<float> d_xn_a(kRows);
      HipBuffer<float> d_xn_b(kRows);
      HipBuffer<__half> d_half_a(kRows);
      HipBuffer<__half> d_half_b(kRows);
      HipBuffer<float> d_scale(static_cast<std::size_t>(kTokens) * kStreams);
      HipBuffer<float> d_mixed_a(kMixed);
      HipBuffer<float> d_mixed_b(kMixed);
      HipBuffer<float> d_inj_a(vec_inject);
      HipBuffer<float> d_inj_b(vec_inject);
      Upload(&d_res_a, res);
      Upload(&d_res_b, res);
      q::HcCombine(d_res_a.get(), d_block_out.get(), d_inject_vec.get(),
                   vec_parts, d_gamma.get(), d_xn_a.get(), kTokens, kHidden,
                   kStreams, kEps, nullptr, false, true);
      q::NarrowActivations(d_xn_a.get(), d_half_a.get(), false, kRows, nullptr);
      if (!q::HcCombine(d_res_b.get(), d_block_out.get(), d_inject_vec.get(),
                        vec_parts, d_gamma.get(), d_xn_b.get(), kTokens,
                        kHidden, kStreams, kEps, nullptr, false, true,
                        d_half_b.get(), d_scale.get()))
        throw std::runtime_error("HC combine did not write the F16 norm");
      q::HcMixEpilogueVec4(d_xn_a.get(), d_gate.get(), d_weight.get(),
                           d_mixed_a.get(), d_inj_a.get(), kTokens, kHidden,
                           kStreams, nullptr);
      q::HcMixEpilogueVec4FromRes(d_res_b.get(), d_gamma.get(), d_scale.get(),
                                  d_gate.get(), d_weight.get(), d_mixed_b.get(),
                                  d_inj_b.get(), kTokens, kHidden, nullptr);
      q::HcNormFromRes(d_res_b.get(), d_gamma.get(), d_scale.get(),
                       d_xn_b.get(), kTokens, kHidden, nullptr);
      CheckHip(hipDeviceSynchronize(), "HC deferred norm");
      const auto res_a = Download(&d_res_a, kRows);
      const auto res_b = Download(&d_res_b, kRows);
      const auto xn_a = Download(&d_xn_a, kRows);
      const auto xn_b = Download(&d_xn_b, kRows);
      const auto mixed_a = Download(&d_mixed_a, kMixed);
      const auto mixed_b = Download(&d_mixed_b, kMixed);
      const auto inj_a = Download(&d_inj_a, vec_inject);
      const auto inj_b = Download(&d_inj_b, vec_inject);
      std::vector<__half> half_a(kRows);
      std::vector<__half> half_b(kRows);
      CheckHip(hipMemcpy(half_a.data(), d_half_a.get(), kRows * sizeof(__half),
                         hipMemcpyDeviceToHost),
               "F32-route half norm");
      CheckHip(hipMemcpy(half_b.data(), d_half_b.get(), kRows * sizeof(__half),
                         hipMemcpyDeviceToHost),
               "deferred half norm");
      const bool res_ok = bytes_equal(res_a.data(), res_b.data(), kRows * 4);
      const bool half_ok =
          bytes_equal(half_a.data(), half_b.data(), kRows * sizeof(__half));
      const bool xn_ok = bytes_equal(xn_a.data(), xn_b.data(), kRows * 4);
      const bool mix_ok =
          bytes_equal(mixed_a.data(), mixed_b.data(), kMixed * 4) &&
          bytes_equal(inj_a.data(), inj_b.data(), vec_inject * 4);
      std::cout << "HC deferred norm: residual " << res_ok << ", F16 norm "
                << half_ok << ", recomputed F32 norm " << xn_ok
                << ", mix epilogue " << mix_ok << '\n';
      if (!res_ok || !half_ok || !xn_ok || !mix_ok)
        throw std::runtime_error("HC deferred norm changed the F32 route");
      // Both epilogues' F16 copy of `mixed` equals narrowing the stored
      // FP32 rows, and writing it changes neither FP32 output.
      HipBuffer<__half> d_mixed_half(kMixed), d_narrowed(kMixed);
      HipBuffer<float> d_mixed_c(kMixed), d_inj_c(vec_inject);
      q::NarrowActivations(d_mixed_a.get(), d_narrowed.get(), false, kMixed,
                           nullptr);
      std::vector<__half> narrowed(kMixed), mixed_half(kMixed);
      CheckHip(hipDeviceSynchronize(), "narrowed mix");
      CheckHip(hipMemcpy(narrowed.data(), d_narrowed.get(),
                         kMixed * sizeof(__half), hipMemcpyDeviceToHost),
               "narrowed mix download");
      for (int route = 0; route < 2; ++route) {
        if (route == 0) {
          if (!q::HcMixEpilogueVec4(d_xn_a.get(), d_gate.get(), d_weight.get(),
                                    d_mixed_c.get(), d_inj_c.get(), kTokens,
                                    kHidden, kStreams, nullptr,
                                    d_mixed_half.get()))
            throw std::runtime_error("vec4 epilogue skipped its F16 mix");
        } else {
          q::HcMixEpilogueVec4FromRes(
              d_res_b.get(), d_gamma.get(), d_scale.get(), d_gate.get(),
              d_weight.get(), d_mixed_c.get(), d_inj_c.get(), kTokens, kHidden,
              nullptr, d_mixed_half.get());
        }
        CheckHip(hipMemcpy(mixed_half.data(), d_mixed_half.get(),
                           kMixed * sizeof(__half), hipMemcpyDeviceToHost),
                 "F16 mix download");
        if (!bytes_equal(mixed_half.data(), narrowed.data(),
                         kMixed * sizeof(__half)) ||
            !bytes_equal(Download(&d_mixed_c, kMixed).data(), mixed_a.data(),
                         kMixed * 4) ||
            !bytes_equal(Download(&d_inj_c, vec_inject).data(), inj_a.data(),
                         vec_inject * 4))
          throw std::runtime_error("HC epilogue F16 mix differs");
      }
      // E2: the attention combine leaves its update out of res; the FFN
      // mixer's recomputed norm, HcNormFromRes and the next combine add it
      // in the combine's FMA. Every output and the final residual keep the
      // writing route's bits.
      HipBuffer<float> d_res_c(kRows), d_res_d(kRows),
          d_defer_w(static_cast<std::size_t>(kTokens) * kStreams);
      HipBuffer<float> d_xn_c(kRows), d_xn_d(kRows), d_next_out(kMixed);
      HipBuffer<__half> d_half_c(kRows), d_half_d(kRows);
      HipBuffer<float> d_scale_c(static_cast<std::size_t>(kTokens) * kStreams);
      HipBuffer<float> d_mixed_d(kMixed), d_inj_d(vec_inject);
      Upload(&d_res_c, res);
      Upload(&d_res_d, res);
      Upload(&d_next_out, MakeValues(kMixed, 0x5EED1234U, 2.0F));
      q::HcCombine(d_res_c.get(), d_block_out.get(), d_inject_vec.get(),
                   vec_parts, d_gamma.get(), d_xn_c.get(), kTokens, kHidden,
                   kStreams, kEps, nullptr, false, true, d_half_c.get(),
                   d_scale_c.get());
      if (!q::HcCombine(d_res_d.get(), d_block_out.get(), d_inject_vec.get(),
                        vec_parts, d_gamma.get(), d_xn_d.get(), kTokens,
                        kHidden, kStreams, kEps, nullptr, false, true,
                        d_half_d.get(), d_scale.get(), nullptr, nullptr,
                        d_defer_w.get()))
        throw std::runtime_error("deferring HC combine skipped its norm");
      const q::DeferredResidual prior{d_block_out.get(), d_defer_w.get()};
      if (!bytes_equal(Download(&d_res_d, kRows).data(), res.data(), kRows * 4))
        throw std::runtime_error("deferring HC combine wrote the residual");
      q::HcMixEpilogueVec4FromRes(d_res_c.get(), d_gamma.get(), d_scale_c.get(),
                                  d_gate.get(), d_weight.get(), d_mixed_c.get(),
                                  d_inj_c.get(), kTokens, kHidden, nullptr);
      q::HcMixEpilogueVec4FromRes(d_res_d.get(), d_gamma.get(), d_scale.get(),
                                  d_gate.get(), d_weight.get(), d_mixed_d.get(),
                                  d_inj_d.get(), kTokens, kHidden, nullptr,
                                  nullptr, nullptr, prior);
      q::HcNormFromRes(d_res_c.get(), d_gamma.get(), d_scale_c.get(),
                       d_xn_c.get(), kTokens, kHidden, nullptr);
      q::HcNormFromRes(d_res_d.get(), d_gamma.get(), d_scale.get(),
                       d_xn_d.get(), kTokens, kHidden, nullptr, prior);
      CheckHip(hipDeviceSynchronize(), "HC deferred residual readers");
      std::vector<__half> half_c(kRows), half_d(kRows);
      CheckHip(hipMemcpy(half_c.data(), d_half_c.get(), kRows * sizeof(__half),
                         hipMemcpyDeviceToHost),
               "half norm c");
      CheckHip(hipMemcpy(half_d.data(), d_half_d.get(), kRows * sizeof(__half),
                         hipMemcpyDeviceToHost),
               "half norm d");
      const bool readers_ok =
          bytes_equal(half_c.data(), half_d.data(), kRows * sizeof(__half)) &&
          bytes_equal(Download(&d_scale_c, kTokens * kStreams).data(),
                      Download(&d_scale, kTokens * kStreams).data(),
                      kTokens * kStreams * 4) &&
          bytes_equal(Download(&d_mixed_c, kMixed).data(),
                      Download(&d_mixed_d, kMixed).data(), kMixed * 4) &&
          bytes_equal(Download(&d_inj_c, vec_inject).data(),
                      Download(&d_inj_d, vec_inject).data(), vec_inject * 4) &&
          bytes_equal(Download(&d_xn_c, kRows).data(),
                      Download(&d_xn_d, kRows).data(), kRows * 4);
      // The next combine (no norm), then the applied update alone.
      HipBuffer<float> d_res_e(kRows);
      Upload(&d_res_e, res);
      q::HcCombine(d_res_c.get(), d_next_out.get(), d_inject_vec.get(),
                   vec_parts, nullptr, d_xn_c.get(), kTokens, kHidden, kStreams,
                   kEps, nullptr, false, true);
      q::HcCombine(d_res_d.get(), d_next_out.get(), d_inject_vec.get(),
                   vec_parts, nullptr, d_xn_d.get(), kTokens, kHidden, kStreams,
                   kEps, nullptr, false, true, nullptr, nullptr, nullptr,
                   nullptr, nullptr, prior);
      q::HcApplyDeferred(d_res_e.get(), d_block_out.get(), d_defer_w.get(),
                         kTokens, kHidden, nullptr);
      HipBuffer<float> d_res_f(kRows);
      Upload(&d_res_f, res);
      q::HcCombine(d_res_f.get(), d_block_out.get(), d_inject_vec.get(),
                   vec_parts, nullptr, d_xn_c.get(), kTokens, kHidden, kStreams,
                   kEps, nullptr, false, true);
      CheckHip(hipDeviceSynchronize(), "HC deferred residual combine");
      const bool chain_ok =
          bytes_equal(Download(&d_res_c, kRows).data(),
                      Download(&d_res_d, kRows).data(), kRows * 4) &&
          bytes_equal(Download(&d_res_e, kRows).data(),
                      Download(&d_res_f, kRows).data(), kRows * 4);
      std::cout << "HC deferred residual: readers " << readers_ok
                << ", next combine and applied update " << chain_ok << '\n';
      if (!readers_ok || !chain_ok)
        throw std::runtime_error("HC deferred residual changed the F32 route");
    }
    std::cout << "HC mixed outputs and short prefill chunks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
