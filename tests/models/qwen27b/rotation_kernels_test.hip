// GPU checks of the 128-point input rotation (rotated GGUF, plan R2 ①).
//
// Every helper layout and every fused producer is compared bit for bit with
// "unrotated producer, then the CPU reference rotation
// (tests/core/quant/rotation_reference.hpp), then the unrotated encoder".
// Where the unrotated FP32 value is not observable from a production
// launcher, a test kernel repeats the producer's arithmetic. The FP16 norm
// keeps its own reduction tree and is held to one FP16 ulp instead.

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/hip/detail/hadamard.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/hadamard.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#include "tests/core/quant/rotation_reference.hpp"

namespace {

using gufo::test::rotation::RotateRow;
namespace hadamard = gufo::hip::hadamard;

int failures = 0;

void Check(hipError_t status, std::string_view operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template<typename T>
class Device {
public:
  explicit Device(std::size_t count) : count_(count) {
    Check(hipMalloc(&data_, std::max<std::size_t>(count, 1) * sizeof(T)),
          "hipMalloc");
    Check(hipMemset(data_, 0, std::max<std::size_t>(count, 1) * sizeof(T)),
          "hipMemset");
  }
  explicit Device(const std::vector<T>& host) : Device(host.size()) {
    Check(hipMemcpy(data_, host.data(), host.size() * sizeof(T),
                    hipMemcpyHostToDevice),
          "upload");
  }
  ~Device() { (void)hipFree(data_); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  [[nodiscard]] T* get() const { return data_; }
  [[nodiscard]] std::vector<T> Read() const {
    Check(hipDeviceSynchronize(), "synchronize");
    std::vector<T> host(count_);
    Check(hipMemcpy(host.data(), data_, count_ * sizeof(T),
                    hipMemcpyDeviceToHost),
          "download");
    return host;
  }

private:
  T* data_{nullptr};
  std::size_t count_{0};
};

std::vector<float> Random(std::size_t n, std::uint32_t seed, float scale) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0F, scale);
  std::vector<float> values(n);
  for (auto& value : values) {
    value = normal(rng);
  }
  // Activation outliers, the case the rotation exists for.
  for (std::size_t i = 7; i < n; i += 997) {
    values[i] *= 40.0F;
  }
  return values;
}

std::vector<std::uint32_t> RandomSigns(std::size_t k, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<std::uint32_t> bits(k / 32);
  for (auto& word : bits) {
    word = rng();
  }
  return bits;
}

std::vector<float> RotateRowsCpu(std::vector<float> rows, std::size_t k,
                                 const std::vector<std::uint32_t>& bits) {
  for (std::size_t r = 0; r * k < rows.size(); ++r) {
    RotateRow(std::span<float>(rows.data() + r * k, k), bits);
  }
  return rows;
}

bool BitEqual(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

std::size_t FirstDifference(const std::vector<float>& a,
                            const std::vector<float>& b) {
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    if (std::bit_cast<std::uint32_t>(a[i]) !=
        std::bit_cast<std::uint32_t>(b[i])) {
      return i;
    }
  }
  return a.size();
}

void ExpectBitEqual(const std::vector<float>& got,
                    const std::vector<float>& want, const std::string& label) {
  if (!BitEqual(got, want)) {
    const auto i = FirstDifference(got, want);
    Expect(false, label + ": first difference at " + std::to_string(i) + " (" +
                      std::to_string(got[i]) + " vs " +
                      std::to_string(want[i]) + ")");
  }
}

// --- Device helper layouts ------------------------------------------------

template<unsigned E>
__global__ void LanesKernel(float* data, const std::uint32_t* signs) {
  constexpr unsigned kLanes = hadamard::kSegment / E;
  const unsigned lane = threadIdx.x % kLanes;
  float* segment = data + (threadIdx.x / kLanes) * hadamard::kSegment;
  float v[E];
  for (unsigned j = 0; j < E; ++j) {
    v[j] = segment[lane * E + j];
  }
  hadamard::RotateSegmentLanes<E>(
      v, signs + (threadIdx.x / kLanes) * hadamard::kSignWordsPerSegment, lane);
  for (unsigned j = 0; j < E; ++j) {
    segment[lane * E + j] = v[j];
  }
}

template<unsigned E>
__global__ void StridedKernel(float* data, const std::uint32_t* signs) {
  constexpr unsigned kLanes = hadamard::kSegment / E;
  const unsigned lane = threadIdx.x % kLanes;
  float* segment = data + (threadIdx.x / kLanes) * hadamard::kSegment;
  float v[E];
  for (unsigned j = 0; j < E; ++j) {
    v[j] = segment[lane + kLanes * j];
  }
  hadamard::RotateSegmentStrided<E>(
      v, signs + (threadIdx.x / kLanes) * hadamard::kSignWordsPerSegment, lane);
  for (unsigned j = 0; j < E; ++j) {
    segment[lane + kLanes * j] = v[j];
  }
}

__global__ void LdsRowKernel(float* data, const std::uint32_t* signs,
                             std::size_t dim) {
  extern __shared__ float row[];
  for (std::size_t i = threadIdx.x; i < dim; i += blockDim.x) {
    row[i] = data[i];
  }
  __syncthreads();
  hadamard::RotateRowSegments<32>(row, dim, signs, threadIdx.x, blockDim.x);
  __syncthreads();
  for (std::size_t i = threadIdx.x; i < dim; i += blockDim.x) {
    data[i] = row[i];
  }
}

void TestHelpers() {
  // Two segments (256 values) per layout; wave32 holds 32 lanes.
  constexpr std::size_t k = 256;
  const auto input = Random(k, 1, 1.0F);
  const auto bits = RandomSigns(k, 2);
  const auto want = RotateRowsCpu(input, k, bits);
  Device<std::uint32_t> signs(bits);
  {
    Device<float> data(input);
    LanesKernel<4><<<1, 64>>>(data.get(), signs.get());
    ExpectBitEqual(data.Read(), want, "RotateSegmentLanes<4>");
  }
  {
    Device<float> data(input);
    LanesKernel<8><<<1, 32>>>(data.get(), signs.get());
    ExpectBitEqual(data.Read(), want, "RotateSegmentLanes<8>");
  }
  {
    Device<float> data(input);
    StridedKernel<4><<<1, 64>>>(data.get(), signs.get());
    ExpectBitEqual(data.Read(), want, "RotateSegmentStrided<4>");
  }
  {
    Device<float> data(input);
    StridedKernel<8><<<1, 32>>>(data.get(), signs.get());
    ExpectBitEqual(data.Read(), want, "RotateSegmentStrided<8>");
  }
  // The LDS row helper serves norm rows (5120, 6144); a 17408-float row
  // would exceed the 64 KiB of LDS a workgroup can allocate.
  for (const std::size_t dim : {5120U, 6144U}) {
    const auto row = Random(dim, 3, 1.0F);
    const auto row_bits = RandomSigns(dim, 4);
    Device<float> data(row);
    Device<std::uint32_t> row_signs(row_bits);
    LdsRowKernel<<<1, 256, dim * sizeof(float)>>>(data.get(), row_signs.get(),
                                                  dim);
    Check(hipGetLastError(), "LdsRowKernel launch");
    ExpectBitEqual(data.Read(), RotateRowsCpu(row, dim, row_bits),
                   "RotateRowSegments dim " + std::to_string(dim));
  }
}

// --- Standalone FP32 rotation (decode / verification) ---------------------

void TestStandalone() {
  for (const std::size_t k : {5120U, 6144U, 17408U}) {
    for (const std::size_t rows : {1U, 3U, 8U}) {
      const auto input = Random(rows * k, 10 + k + rows, 1.0F);
      const auto bits = RandomSigns(k, 20 + k);
      Device<float> data(input);
      Device<std::uint32_t> signs(bits);
      gufo::hip::LaunchRotateRowsFp32(data.get(), signs.get(), rows, k);
      ExpectBitEqual(data.Read(), RotateRowsCpu(input, k, bits),
                     "LaunchRotateRowsFp32 k " + std::to_string(k) + " rows " +
                         std::to_string(rows));
    }
  }
}

// --- Norm producers --------------------------------------------------------

void TestDecodeNorm() {
  for (const std::size_t dim : {5120U, 6144U}) {
    const auto x = Random(dim, 30, 2.0F);
    const auto w = Random(dim, 31, 1.0F);
    const auto bits = RandomSigns(dim, 32);
    Device<float> dx(x), dw(w), plain(dim), rotated(dim);
    Device<std::uint32_t> signs(bits);
    gufo::hip::LaunchRMSNorm(dx.get(), dw.get(), plain.get(), dim, 1e-6F);
    gufo::hip::LaunchRMSNormRotated(dx.get(), dw.get(), rotated.get(), dim,
                                    1e-6F, signs.get());
    ExpectBitEqual(rotated.Read(), RotateRowsCpu(plain.Read(), dim, bits),
                   "LaunchRMSNormRotated dim " + std::to_string(dim));
  }
}

std::uint16_t Bf16Bits(float value) {
  return std::bit_cast<std::uint16_t>(hip_bfloat16(value));
}

void TestBatchedNorm() {
  constexpr std::size_t dim = 5120;
  // 8 rows takes the cached verification kernel, 70 the generic one.
  for (const std::size_t batch : {1U, 8U, 70U}) {
    const auto x = Random(batch * dim, 40 + batch, 2.0F);
    const auto w = Random(dim, 41, 1.0F);
    const auto bits = RandomSigns(dim, 42);
    Device<float> dx(x), dw(w), plain(batch * dim), rotated(batch * dim);
    Device<std::uint16_t> bf16(batch * dim);
    Device<std::uint32_t> signs(bits);
    gufo::hip::LaunchBatchedRMSNorm(dx.get(), dw.get(), plain.get(), nullptr,
                                    batch, dim, 1e-6F);
    gufo::hip::LaunchBatchedRMSNormRotated(dx.get(), dw.get(), rotated.get(),
                                           bf16.get(), batch, dim, 1e-6F,
                                           signs.get());
    const auto want = RotateRowsCpu(plain.Read(), dim, bits);
    const auto got = rotated.Read();
    ExpectBitEqual(
        got, want,
        "LaunchBatchedRMSNormRotated batch " + std::to_string(batch));
    const auto got_bf16 = bf16.Read();
    bool same = true;
    for (std::size_t i = 0; i < want.size(); ++i) {
      same = same && got_bf16[i] == Bf16Bits(want[i]);
    }
    Expect(same, "rotated BF16 copy, batch " + std::to_string(batch));
  }
}

std::vector<std::uint8_t> QuantizeFp32(const std::vector<float>& rows,
                                       std::size_t batch, std::size_t k) {
  Device<float> data(rows);
  Device<std::uint8_t> q8(gufo::hip::QuantizedActivationBytes(batch, k));
  gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(data.get(), q8.get(), batch,
                                                  k);
  return q8.Read();
}

void TestQ8Norm() {
  constexpr std::size_t dim = 5120;
  for (const bool residual : {false, true}) {
    for (const std::size_t batch : {1U, 17U, 70U}) {
      const auto x = Random(batch * dim, 50 + batch, 2.0F);
      const auto r = Random(batch * dim, 51 + batch, 1.0F);
      const auto w = Random(dim, 52, 1.0F);
      const auto bits = RandomSigns(dim, 53);
      Device<float> dx(x), dr(r), dw(w), sum(batch * dim), plain(batch * dim),
          added(batch * dim);
      Device<std::uint32_t> signs(bits);
      Device<std::uint8_t> q8(gufo::hip::QuantizedActivationBytes(batch, dim));
      // The fused kernel's normed values equal ResidualAdd + the generic
      // BatchedRMSNorm (opt-c173-norm-quant); 70 rows select that kernel,
      // smaller batches are normed one row group at a time below.
      const float* norm_input = dx.get();
      if (residual) {
        gufo::hip::LaunchBatchedResidualAdd(dx.get(), dr.get(), added.get(),
                                            batch, dim);
        norm_input = added.get();
      }
      for (std::size_t row = 0; row < batch; row += 70) {
        const std::size_t rows = std::min<std::size_t>(70, batch - row);
        Device<float> padded(70 * dim);
        Check(hipMemcpy(padded.get(), norm_input + row * dim,
                        rows * dim * sizeof(float), hipMemcpyDeviceToDevice),
              "copy rows");
        Device<float> normed(70 * dim);
        gufo::hip::LaunchBatchedRMSNorm(padded.get(), dw.get(), normed.get(),
                                        nullptr, 70, dim, 1e-6F);
        Check(hipMemcpy(plain.get() + row * dim, normed.get(),
                        rows * dim * sizeof(float), hipMemcpyDeviceToDevice),
              "copy normed");
      }
      gufo::hip::LaunchBatchedFusedRMSNormRotateQuantizeQ8_1(
          dx.get(), residual ? dr.get() : nullptr, dw.get(),
          residual ? sum.get() : nullptr, q8.get(), batch, dim, 1e-6F,
          signs.get());
      const auto want =
          QuantizeFp32(RotateRowsCpu(plain.Read(), dim, bits), batch, dim);
      Expect(q8.Read() == want, "rotated RMSNorm Q8_1, batch " +
                                    std::to_string(batch) +
                                    (residual ? " with residual" : ""));
      if (residual) {
        ExpectBitEqual(sum.Read(), added.Read(), "residual sum output");
      }
    }
  }
}

void TestFp16Norm() {
  for (const std::size_t dim : {5120U, 6144U}) {
    constexpr std::size_t batch = 9;
    const auto x = Random(batch * dim, 60, 2.0F);
    const auto w = Random(dim, 61, 1.0F);
    const auto bits = RandomSigns(dim, 62);
    Device<float> dx(x), dw(w), fp32(batch * dim);
    Device<std::uint16_t> half(batch * dim);
    Device<std::uint32_t> signs(bits);
    gufo::hip::LaunchBatchedRMSNormRotated(dx.get(), dw.get(), fp32.get(),
                                           nullptr, batch, dim, 1e-6F,
                                           signs.get());
    gufo::hip::LaunchBatchedRMSNormRotatedFp16(dx.get(), dw.get(), half.get(),
                                               batch, dim, 1e-6F, signs.get(),
                                               nullptr);
    const auto want = fp32.Read();
    const auto got = half.Read();
    // Different reduction trees can move inv_rms by an ulp: allow one FP16
    // ulp against the FP32 rotated row rounded to FP16.
    const auto ordered = [](std::uint16_t bits) {
      const int magnitude = bits & 0x7FFF;
      return (bits & 0x8000U) != 0 ? -magnitude : magnitude;
    };
    std::size_t bad = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
      const auto expected =
          std::bit_cast<std::uint16_t>(__half(__float2half_rn(want[i])));
      bad += std::abs(ordered(got[i]) - ordered(expected)) > 1 ? 1 : 0;
    }
    Expect(bad == 0, "rotated FP16 RMSNorm within one ulp, dim " +
                         std::to_string(dim) + " (" + std::to_string(bad) +
                         " outside)");
  }
}

// --- SwiGLU producers ------------------------------------------------------

// The unrotated FP32 values of BatchedFusedSwiGLUQuantizeQ8_1Kernel.
__global__ void SwiGluQ8Reference(const float* gate, const float* up,
                                  float* out, std::size_t n) {
  const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float g = gate[i];
    const float u = up[i];
    const float silu_g = g / (1.0F + expf(-g));
    out[i] = silu_g * u;
  }
}

// The unrotated FP32 values of LaunchBatchedSwiGLURotateFp16.
__global__ void SwiGluFp16Reference(const float* gate, const float* up,
                                    float* out, std::size_t n) {
  const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float g = gate[i];
    const float sig = 1.0F / (1.0F + expf(-g));
    out[i] = __fmul_rn(g * sig, up[i]);
  }
}

__global__ void ToHalfKernel(const float* in, __half* out, std::size_t n) {
  const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = __float2half_rn(in[i]);
  }
}

void TestSwiGlu() {
  constexpr std::size_t k = 17408;
  for (const std::size_t batch : {1U, 5U, 33U}) {
    const std::size_t n = batch * k;
    const auto gate = Random(n, 70 + batch, 1.5F);
    const auto up = Random(n, 71 + batch, 1.0F);
    const auto bits = RandomSigns(k, 72);
    Device<float> dg(gate), du(up), plain(n);
    Device<std::uint32_t> signs(bits);
    const unsigned grid = static_cast<unsigned>((n + 255) / 256);

    SwiGluQ8Reference<<<grid, 256>>>(dg.get(), du.get(), plain.get(), n);
    const auto want_q8 =
        QuantizeFp32(RotateRowsCpu(plain.Read(), k, bits), batch, k);
    Device<std::uint8_t> q8(gufo::hip::QuantizedActivationBytes(batch, k));
    gufo::hip::LaunchBatchedFusedSwiGLURotateQuantizeQ8_1(
        dg.get(), du.get(), q8.get(), batch, k, signs.get());
    Expect(q8.Read() == want_q8,
           "rotated SwiGLU Q8_1, batch " + std::to_string(batch));

    SwiGluFp16Reference<<<grid, 256>>>(dg.get(), du.get(), plain.get(), n);
    const auto plain_values = plain.Read();
    const auto rotated = RotateRowsCpu(plain_values, k, bits);
    Device<std::uint16_t> half(n);
    gufo::hip::LaunchBatchedSwiGLURotateFp16(dg.get(), du.get(), half.get(),
                                             batch, k, signs.get());
    const auto got = half.Read();
    // The same rotated FP32 rows converted by the device instruction, to
    // tell a conversion difference from a value difference.
    Device<float> rotated_device(rotated);
    Device<std::uint16_t> device_half(n);
    ToHalfKernel<<<grid, 256>>>(rotated_device.get(),
                                reinterpret_cast<__half*>(device_half.get()),
                                n);
    const auto want_device = device_half.Read();
    // Fully on the device: the reference kernel's pre-rotation FP32 (device
    // expf), the production standalone rotation, the device conversion.
    Device<float> device_rotated(plain_values);
    gufo::hip::LaunchRotateRowsFp32(device_rotated.get(), signs.get(), batch,
                                    k);
    const auto device_rotated_values = device_rotated.Read();
    Device<std::uint16_t> device_chain(n);
    ToHalfKernel<<<grid, 256>>>(device_rotated.get(),
                                reinterpret_cast<__half*>(device_chain.get()),
                                n);
    const auto want_chain = device_chain.Read();
    std::size_t chain_differing = 0;
    std::size_t host_device_differing = 0;
    for (std::size_t i = 0; i < n; ++i) {
      chain_differing += got[i] != want_chain[i] ? 1 : 0;
      host_device_differing += want_device[i] != std::bit_cast<std::uint16_t>(
                                                     __half(__float2half_rn(
                                                         rotated[i])))
                                   ? 1
                                   : 0;
    }
    std::cerr << "  SwiGLU FP16 batch " << batch << ": kernel vs device chain "
              << chain_differing << " differ; host vs device conversion of "
              << "the CPU-rotated row " << host_device_differing
              << " differ; CPU vs device rotation "
              << (BitEqual(device_rotated_values, rotated) ? "equal" : "differ")
              << "\n";
    std::size_t printed = 0;
    std::size_t differing = 0;
    std::size_t first = n;
    std::size_t last = 0;
    std::size_t rows_hit = 0;
    std::size_t last_row = n;
    int max_ulp = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const auto want = std::bit_cast<std::uint16_t>(
          __half(__float2half_rn(rotated[i])));
      if (got[i] != want && printed < 4) {
        ++printed;
        std::cerr << "  row " << i / k << " column " << i % k << ": fp32 "
                  << std::hexfloat << rotated[i] << std::defaultfloat
                  << " host half 0x" << std::hex << want << " device half 0x"
                  << want_device[i] << " kernel half 0x" << got[i] << std::dec
                  << " (pre-rotation " << plain_values[i] << ")\n";
      }
      if (got[i] != want) {
        const auto ordered = [](std::uint16_t bits) {
          const int magnitude = bits & 0x7FFF;
          return (bits & 0x8000U) != 0 ? -magnitude : magnitude;
        };
        max_ulp = std::max(max_ulp, std::abs(ordered(got[i]) - ordered(want)));
        first = std::min(first, i);
        last = i;
        rows_hit += (i / k) != last_row ? 1 : 0;
        last_row = i / k;
        ++differing;
      }
    }
    Expect(differing == 0,
           "rotated SwiGLU FP16, batch " + std::to_string(batch) + ": " +
               std::to_string(differing) + " of " + std::to_string(n) +
               " differ in " + std::to_string(rows_hit) + " row(s), first " +
               "row " + std::to_string(first / k) + " column " +
               std::to_string(first % k) + ", last row " +
               std::to_string(last / k) + " column " +
               std::to_string(last % k) + ", max " + std::to_string(max_ulp) +
               " FP16 ulp");
  }
}

// --- SSM epilogue ----------------------------------------------------------

// The unrotated FP32 values of BatchedFusedSSMPostNormGateQuantizeQ8_1Kernel
// (one 128-thread block per head and token, same reduction and gate).
__global__ void SsmEpilogueReference(const float* raw_out,
                                     const float* ssm_norm, const float* gate,
                                     float* out, std::size_t inner_size,
                                     std::uint32_t val_dim) {
  const std::size_t h = blockIdx.x;
  const std::size_t t = blockIdx.y;
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const float* in_h = raw_out + (t * inner_size) + (h * val_dim);
  const float in_val = (tid < val_dim) ? in_h[tid] : 0.0F;
  float o_sq = in_val * in_val;
  for (int offset = 16; offset > 0; offset >>= 1) {
    o_sq += __shfl_xor(o_sq, offset);
  }
  __shared__ float s_warp_rms[4];
  if (lane_id == 0) {
    s_warp_rms[warp_id] = o_sq;
  }
  __syncthreads();
  __shared__ float s_inv_rms;
  if (tid == 0) {
    const float total_o_sq =
        s_warp_rms[0] + s_warp_rms[1] + s_warp_rms[2] + s_warp_rms[3];
    const float mean_sq = total_o_sq / static_cast<float>(val_dim);
    s_inv_rms = rsqrtf(mean_sq + 1e-6F);
  }
  __syncthreads();
  const float normed = in_val * s_inv_rms;
  float val = normed * ssm_norm[tid];
  const float g = (gate + (t * inner_size))[(h * val_dim) + tid];
  const float sig = 1.0F / (1.0F + expf(-g));
  val *= (g * sig);
  out[(t * inner_size) + (h * val_dim) + tid] = val;
}

void TestSsmEpilogue() {
  constexpr std::uint32_t heads = 48;
  constexpr std::uint32_t val_dim = 128;
  constexpr std::size_t inner = heads * val_dim;
  for (const std::size_t batch : {1U, 19U}) {
    const auto raw = Random(batch * inner, 80 + batch, 1.0F);
    const auto norm = Random(val_dim, 81, 1.0F);
    const auto gate = Random(batch * inner, 82 + batch, 1.0F);
    const auto bits = RandomSigns(inner, 83);
    Device<float> draw(raw), dnorm(norm), dgate(gate), plain(batch * inner);
    Device<std::uint32_t> signs(bits);
    SsmEpilogueReference<<<dim3(heads, batch), 128>>>(
        draw.get(), dnorm.get(), dgate.get(), plain.get(), inner, val_dim);
    const auto want =
        QuantizeFp32(RotateRowsCpu(plain.Read(), inner, bits), batch, inner);
    Device<std::uint8_t> q8(gufo::hip::QuantizedActivationBytes(batch, inner));
    gufo::hip::LaunchBatchedFusedSSMPostNormGateQuantizeQ8_1(
        draw.get(), dnorm.get(), dgate.get(), q8.get(), batch, inner, heads,
        val_dim, nullptr, signs.get());
    Expect(q8.Read() == want,
           "rotated SSM epilogue Q8_1, batch " + std::to_string(batch));
  }
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cout << "SKIP: no HIP device\n";
    return 77;
  }
  try {
    TestHelpers();
    TestStandalone();
    TestDecodeNorm();
    TestBatchedNorm();
    TestQ8Norm();
    TestFp16Norm();
    TestSwiGlu();
    TestSsmEpilogue();
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " rotation kernel check(s) failed\n";
    return 1;
  }
  std::cout << "qwen27b_rotation_kernels_test: all checks passed\n";
  return 0;
}
