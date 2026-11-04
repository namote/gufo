#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/host_allocations.hpp"
#include "src/core/hip/managed_alloc.hpp"
#include "src/core/quant/exl3/reference.hpp"
#include "src/models/qwen/hip/kernels/exl3_gemv.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#include "src/models/qwen/vision/encoder.hpp"
#include "src/models/qwen/vision/exl3_weights.hpp"

namespace gufo::models::qwen::vision {
namespace {
using Bf16 = hip_bfloat16;
using Half = __half;
constexpr unsigned kHidden = 1152;
constexpr unsigned kHeads = 16;
constexpr unsigned kHeadDim = 72;
constexpr unsigned kFfn = 4304;
constexpr unsigned kMergeHidden = kHidden * 4;
constexpr unsigned kPatchDim = 3 * 2 * 16 * 16;
constexpr unsigned kThreads = 256;

void Check(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}
void CheckBlas(hipblasStatus_t status) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("vision hipBLAS failure: " +
                             std::to_string(status));
  }
}

struct Allocation {
  void* data{nullptr};
  Allocation() = default;
  explicit Allocation(std::size_t bytes) {
    if (bytes)
      Check(hipMalloc(&data, bytes));
  }
  static Allocation Weight(std::size_t bytes) {
    // Keep the copied weights in the device heap when its reserve permits.
    // Movable managed pages can evict text inference queues even when no
    // image request is using this sidecar.
    Allocation allocation;
    Check(gufo::hip::AllocateDevice(&allocation.data, bytes));
    return allocation;
  }
  ~Allocation() {
    if (data != nullptr)
      (void)hipFree(data);
  }
  Allocation(const Allocation&) = delete;
  Allocation& operator=(const Allocation&) = delete;
  Allocation(Allocation&& other) noexcept
      : data(std::exchange(other.data, nullptr)) {}
  Allocation& operator=(Allocation&& other) noexcept {
    if (this != &other) {
      if (data != nullptr)
        (void)hipFree(data);
      data = std::exchange(other.data, nullptr);
    }
    return *this;
  }
  template<class T>
  T* As() const {
    return static_cast<T*>(data);
  }
};

// Pageable async copies leave the source ranges registered in ROCr's SVM
// cache. Moving those file/heap pages can then evict unrelated text queues
// long after upload. Copy through one bounded, explicitly owned host buffer.
class PinnedUpload {
public:
  explicit PinnedUpload(hipStream_t stream, std::size_t bytes = kBytes)
      : capacity_(std::min(bytes, kBytes)), stream_(stream) {
    Check(allocations_.Allocate(&data_, capacity_));
  }
  PinnedUpload(const PinnedUpload&) = delete;
  PinnedUpload& operator=(const PinnedUpload&) = delete;

  void Copy(void* destination, const void* source, std::size_t bytes) {
    auto* out = static_cast<std::byte*>(destination);
    const auto* in = static_cast<const std::byte*>(source);
    for (std::size_t offset = 0; offset < bytes;) {
      const auto count = std::min(capacity_, bytes - offset);
      std::memcpy(data_, in + offset, count);
      Check(hipMemcpyAsync(out + offset, data_, count, hipMemcpyHostToDevice,
                           stream_));
      // The next CPU copy must not overwrite a buffer still read by the GPU.
      Check(hipStreamSynchronize(stream_));
      offset += count;
    }
  }

private:
  static constexpr std::size_t kBytes = 16ULL << 20;
  gufo::hip::HostAllocations allocations_;
  void* data_{nullptr};
  std::size_t capacity_;
  hipStream_t stream_;
};

template<class T = Bf16>
__device__ __forceinline__ float Round(float x) {
  return static_cast<float>(T(x));
}

// Packed EXL3 linears produce FP16 before their separately applied bias.
// Legacy GGUF projections retain the existing single BF16 rounding boundary.
template<class T>
__device__ __forceinline__ float ProjectionBias(float x, float bias) {
  if constexpr (std::is_same_v<T, Half>)
    return Round<T>(Round<T>(x) + bias);
  return Round<T>(x + bias);
}

__device__ __forceinline__ float PositionRound(float x, unsigned dtype) {
  if (dtype == 1)
    return Round<Half>(x);
  if (dtype == 2)
    return Round<Bf16>(x);
  return x;
}

template<class T>
__device__ __forceinline__ float AttentionScore(float x) {
  if constexpr (std::is_same_v<T, Half>)
    return x * 0.1178511301977579F;
  return Round<T>(Round<T>(x) * 0.1178511301977579F);
}

template<class T = Bf16>
__global__ void Patchify(const std::uint8_t* pixels, T* patches, unsigned width,
                         unsigned grid_width, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * kPatchDim)
    return;
  const unsigned patch = i / kPatchDim;
  const unsigned element = i % kPatchDim;
  const unsigned merge = patch / 4;
  const unsigned y = (merge / (grid_width / 2)) * 2 + (patch % 4) / 2;
  const unsigned x = (merge % (grid_width / 2)) * 2 + patch % 2;
  const unsigned channel = element / (2 * 16 * 16);
  const unsigned spatial = element % (16 * 16);
  const unsigned pixel =
      ((y * 16 + spatial / 16) * width + x * 16 + spatial % 16) * 3;
  // Match rescale_and_normalize: multiply by 1/255 then subtract/divide.
  patches[i] =
      T((static_cast<float>(pixels[pixel + channel]) * (1.0F / 255.0F) - 0.5F) /
        0.5F);
}

template<class T = Bf16>
__global__ void PatchPosition(float* hidden, const float* bias,
                              const float* positions, unsigned grid_height,
                              unsigned grid_width, unsigned count,
                              unsigned position_dtype = 0) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * kHidden)
    return;
  const unsigned patch = i / kHidden;
  const unsigned dim = i % kHidden;
  const unsigned merge = patch / 4;
  const unsigned y = (merge / (grid_width / 2)) * 2 + (patch % 4) / 2;
  const unsigned x = (merge % (grid_width / 2)) * 2 + patch % 2;
  const float fy = static_cast<float>(y) * (47.0F / (grid_height - 1));
  const float fx = static_cast<float>(x) * (47.0F / (grid_width - 1));
  const unsigned y0 = min(static_cast<unsigned>(fy), 47U);
  const unsigned x0 = min(static_cast<unsigned>(fx), 47U);
  const unsigned y1 = min(y0 + 1, 47U);
  const unsigned x1 = min(x0 + 1, 47U);
  const float wy = fy - y0;
  const float wx = fx - x0;
  // The reference sums four interpolation products in this order.
  float position =
      positions[(y0 * 48 + x0) * kHidden + dim] * ((1 - wy) * (1 - wx));
  position += positions[(y0 * 48 + x1) * kHidden + dim] * ((1 - wy) * wx);
  position += positions[(y1 * 48 + x0) * kHidden + dim] * (wy * (1 - wx));
  position += positions[(y1 * 48 + x1) * kHidden + dim] * (wy * wx);
  if constexpr (std::is_same_v<T, Half>) {
    // Upstream interpolation multiplies and adds in the embedding's stored
    // dtype, then adds that position to the FP32 patch residual.
    const float w00 = PositionRound((1 - wy) * (1 - wx), position_dtype);
    const float w01 = PositionRound((1 - wy) * wx, position_dtype);
    const float w10 = PositionRound(wy * (1 - wx), position_dtype);
    const float w11 = PositionRound(wy * wx, position_dtype);
    position = PositionRound(positions[(y0 * 48 + x0) * kHidden + dim] * w00,
                             position_dtype);
    position = PositionRound(
        position +
            PositionRound(positions[(y0 * 48 + x1) * kHidden + dim] * w01,
                          position_dtype),
        position_dtype);
    position = PositionRound(
        position +
            PositionRound(positions[(y1 * 48 + x0) * kHidden + dim] * w10,
                          position_dtype),
        position_dtype);
    position = PositionRound(
        position +
            PositionRound(positions[(y1 * 48 + x1) * kHidden + dim] * w11,
                          position_dtype),
        position_dtype);
    hidden[i] = (hidden[i] + bias[dim]) + position;
  } else {
    hidden[i] = Round<T>(Round<T>(Round<T>(hidden[i]) + bias[dim]) +
                         Round<T>(position));
  }
}

template<class T = Bf16>
__global__ void LayerNorm(const float* input, T* output, const float* weight,
                          const float* bias, unsigned width) {
  const unsigned row = blockIdx.x;
  __shared__ float sum[kThreads];
  float value = 0;
  for (unsigned j = threadIdx.x; j < width; j += kThreads)
    value += input[row * width + j];
  sum[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      sum[threadIdx.x] += sum[threadIdx.x + stride];
    __syncthreads();
  }
  const float mean = sum[0] / width;
  __syncthreads();
  value = 0;
  for (unsigned j = threadIdx.x; j < width; j += kThreads) {
    const float centered = input[row * width + j] - mean;
    value += centered * centered;
  }
  sum[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      sum[threadIdx.x] += sum[threadIdx.x + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(sum[0] / width + 1e-6F);
  for (unsigned j = threadIdx.x; j < width; j += kThreads) {
    output[row * width + j] =
        T((input[row * width + j] - mean) * scale * weight[j] + bias[j]);
  }
}

template<class T = Bf16>
__global__ void BiasResidual(float* hidden, const float* projection,
                             const float* bias, unsigned size, unsigned width) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    const float value = ProjectionBias<T>(projection[i], bias[i % width]);
    if constexpr (std::is_same_v<T, Half>)
      hidden[i] += value;
    else
      hidden[i] = Round<T>(hidden[i] + value);
  }
}

template<class T = Bf16>
__global__ void Activate(const float* input, const float* bias, T* output,
                         unsigned size, unsigned width, bool approximate) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;
  const float x = ProjectionBias<T>(input[i], bias[i % width]);
  const float gelu = approximate
                         ? 0.5F * x *
                               (1.0F + tanhf(0.7978845608028654F *
                                             (x + 0.044715F * x * x * x)))
                         : 0.5F * x * (1.0F + erff(x * 0.7071067811865475F));
  output[i] = T(gelu);
}

template<class T = Bf16>
__global__ void QkvRope(const float* input, const float* bias, T* q, T* k, T* v,
                        unsigned count, unsigned grid_width) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * kHidden)
    return;
  const unsigned token = i / kHidden;
  const unsigned head = (i % kHidden) / kHeadDim;
  const unsigned d = i % kHeadDim;
  const unsigned pair = d % (kHeadDim / 2);
  const unsigned merge = token / 4;
  const unsigned y = (merge / (grid_width / 2)) * 2 + (token % 4) / 2;
  const unsigned x = (merge % (grid_width / 2)) * 2 + token % 2;
  const unsigned position = pair < kHeadDim / 4 ? y : x;
  const float angle =
      position / powf(10000.0F, static_cast<float>(pair % (kHeadDim / 4)) /
                                    (kHeadDim / 4));
  const float cosine = cosf(angle);
  const float sine = sinf(angle);
  const unsigned first = head * kHeadDim + pair;
  const unsigned base = token * 3 * kHidden;
  const float q0 = ProjectionBias<T>(input[base + first], bias[first]);
  const float q1 = ProjectionBias<T>(input[base + first + kHeadDim / 2],
                                     bias[first + kHeadDim / 2]);
  const float k0 =
      ProjectionBias<T>(input[base + kHidden + first], bias[kHidden + first]);
  const float k1 =
      ProjectionBias<T>(input[base + kHidden + first + kHeadDim / 2],
                        bias[kHidden + first + kHeadDim / 2]);
  const unsigned target = (head * count + token) * kHeadDim + d;
  q[target] =
      T(d < kHeadDim / 2 ? q0 * cosine - q1 * sine : q1 * cosine + q0 * sine);
  k[target] =
      T(d < kHeadDim / 2 ? k0 * cosine - k1 * sine : k1 * cosine + k0 * sine);
  v[target] =
      T(ProjectionBias<T>(input[base + 2 * kHidden + head * kHeadDim + d],
                          bias[2 * kHidden + head * kHeadDim + d]));
}

// Keep bounded rows in registers through normalization. Larger rows retain
// the streaming path; caching them would spill and reread private memory.
template<unsigned kValues, class T = Bf16>
__global__ void Softmax(const float* scores, T* probabilities,
                        unsigned length) {
  const unsigned row = blockIdx.x;
  __shared__ float reduce[kThreads];
  float values[kValues != 0 ? kValues : 1];
  float maximum = -INFINITY;
  if constexpr (kValues != 0) {
#pragma unroll
    for (unsigned j = 0; j < kValues; ++j) {
      const unsigned i = threadIdx.x + j * kThreads;
      values[j] =
          i < length ? AttentionScore<T>(scores[row * length + i]) : -INFINITY;
      maximum = fmaxf(maximum, values[j]);
    }
  } else {
    for (unsigned i = threadIdx.x; i < length; i += kThreads) {
      maximum = fmaxf(maximum, AttentionScore<T>(scores[row * length + i]));
    }
  }
  reduce[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride)
      reduce[threadIdx.x] =
          fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
    __syncthreads();
  }
  maximum = reduce[0];
  __syncthreads();
  float sum = 0;
  if constexpr (kValues != 0) {
#pragma unroll
    for (unsigned j = 0; j < kValues; ++j) {
      if (threadIdx.x + j * kThreads < length) {
        values[j] = expf(values[j] - maximum);
        sum += values[j];
        // Limit overlapping exponential temporaries without changing their
        // order. Wider shapes otherwise spill registers on gfx1151.
        if constexpr (kValues > 16)
          __builtin_amdgcn_sched_barrier(0);
      }
    }
  } else {
    for (unsigned i = threadIdx.x; i < length; i += kThreads)
      sum += expf(AttentionScore<T>(scores[row * length + i]) - maximum);
  }
  reduce[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride)
      reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    __syncthreads();
  }
  const float inverse = 1.0F / reduce[0];
  if constexpr (kValues != 0) {
#pragma unroll
    for (unsigned j = 0; j < kValues; ++j) {
      const unsigned i = threadIdx.x + j * kThreads;
      if (i < length)
        probabilities[row * length + i] = T(values[j] * inverse);
    }
  } else {
    for (unsigned i = threadIdx.x; i < length; i += kThreads) {
      probabilities[row * length + i] =
          T(expf(AttentionScore<T>(scores[row * length + i]) - maximum) *
            inverse);
    }
  }
}

template<class T = Bf16>
__global__ void AttentionRows(const float* input, T* output, unsigned begin,
                              unsigned rows) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * kHidden)
    return;
  const unsigned token = i / kHidden;
  const unsigned head = (i % kHidden) / kHeadDim;
  const unsigned d = i % kHeadDim;
  output[(begin + token) * kHidden + head * kHeadDim + d] =
      T(input[(head * rows + token) * kHeadDim + d]);
}

template<class T = Bf16>
__global__ void Finish(float* output, const float* bias, unsigned size,
                       unsigned width) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    if constexpr (std::is_same_v<T, Half>)
      output[i] += bias[i % width];
    else
      output[i] = Round<T>(output[i] + bias[i % width]);
  }
}

// Preserve EXL3 FFN padding; only raw-to-quantized boundaries add zeros.
__global__ void NativeInput(const Half* input, float* output, unsigned rows,
                            unsigned input_width, unsigned width) {
  const auto i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i < std::size_t{rows} * width) {
    const auto col = i % width;
    output[i] = col < input_width
                    ? static_cast<float>(input[(i / width) * input_width + col])
                    : 0.0F;
  }
}
__global__ void NativeOutput(const float* input, float* output, unsigned rows,
                             unsigned input_width, unsigned width,
                             unsigned output_stride) {
  const auto i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i < std::size_t{rows} * width)
    output[(i / width) * output_stride + i % width] =
        input[(i / width) * input_width + i % width];
}

std::uint16_t NativeHalf(float value) {
  const auto bits = quant::exl3::HalfFromDouble(value);
  if ((bits & 0x7c00U) == 0x7c00U)
    throw std::invalid_argument("EXL3 vision weight exceeds finite FP16 range");
  return bits;
}

std::uint16_t ToBf16(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  return static_cast<std::uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
}  // namespace

struct Encoder::Impl {
  std::unique_ptr<core::GgufReader> reader;
  std::unique_ptr<Exl3Weights> exl3;
  std::unordered_map<std::string, quant::exl3::DeviceLinear> linears;
  std::size_t max_inner_elements{0};
  unsigned position_dtype{0};
  std::uint32_t output_width;
  std::string identity;
  std::once_flag identity_once;
  std::atomic<std::size_t> resident_bytes{0};
  std::unordered_map<std::string, Allocation> weights;
  hipStream_t stream{nullptr};
  hipblasHandle_t blas{nullptr};
  std::unique_ptr<gufo::hip::HipblasLtGemm> lt;
  bool uploaded{false};

  Impl(const std::filesystem::path& path, std::uint32_t width)
      : output_width(width) {
    if (std::filesystem::is_directory(path)) {
      exl3 = std::make_unique<Exl3Weights>(path, width);
      const auto dtype = exl3->tensors().at("v.position_embd.weight")->dtype;
      position_dtype = dtype == quant::exl3::DType::kBF16  ? 2U
                       : dtype == quant::exl3::DType::kF16 ? 1U
                                                           : 0U;
      return;
    }
    std::string error;
    reader = core::GgufReader::OpenFile(path, &error);
    if (!reader)
      throw std::invalid_argument(error);
    const auto expect = [&](std::string_view key, std::uint64_t value) {
      if (reader->GetMetadataUint64(key) != value) {
        throw std::invalid_argument("incompatible Qwen vision metadata: " +
                                    std::string(key));
      }
    };
    expect("clip.vision.projection_dim", width);
    expect("clip.vision.patch_size", 16);
    expect("clip.vision.embedding_length", kHidden);
    expect("clip.vision.feed_forward_length", kFfn);
    expect("clip.vision.block_count", 27);
    expect("clip.vision.attention.head_count", kHeads);
    expect("clip.vision.spatial_merge_size", 2);
    expect("clip.vision.image_size", 768);
    if ((width != 5120 && width != 2560) ||
        reader->GetMetadataString("general.architecture") != "clip" ||
        reader->GetMetadataString("general.type") != "mmproj" ||
        reader->GetMetadataBool("clip.has_vision_encoder") != true ||
        reader->GetMetadataBool("clip.use_gelu") != true ||
        reader->GetMetadataFloat32(
            "clip.vision.attention.layer_norm_epsilon") != 1e-6F) {
      throw std::invalid_argument(
          "vision sidecar does not match the Qwen3.8 model");
    }
    for (const auto key : {"clip.vision.image_mean", "clip.vision.image_std"}) {
      const auto* value = reader->FindMetadata(key);
      const auto* array =
          value ? std::get_if<std::vector<double>>(&value->value) : nullptr;
      if (!array || *array != std::vector<double>{0.5, 0.5, 0.5}) {
        throw std::invalid_argument("unsupported Qwen image normalization");
      }
    }
    const auto* deepstack_value =
        reader->FindMetadata("clip.vision.is_deepstack_layers");
    const auto* deepstack =
        deepstack_value
            ? std::get_if<std::vector<std::uint64_t>>(&deepstack_value->value)
            : nullptr;
    if (!deepstack || deepstack->size() != 27 ||
        std::any_of(deepstack->begin(), deepstack->end(),
                    [](auto enabled) { return enabled != 0; })) {
      throw std::invalid_argument(
          "Qwen3.8 vision requires a final-layer-only merger");
    }
    if (reader->GetMetadataString("clip.projector_type") != "qwen3vl_merger") {
      throw std::invalid_argument("unsupported Qwen vision projector");
    }
    const auto require = [&](const std::string& name,
                             std::initializer_list<std::uint64_t> shape,
                             core::GgmlType type) {
      const auto* tensor = reader->FindTensor(name);
      if (tensor == nullptr ||
          tensor->dimensions != std::vector<std::uint64_t>(shape) ||
          tensor->type != type) {
        throw std::invalid_argument("invalid BF16 Qwen vision tensor: " + name);
      }
      const auto bytes =
          tensor->ElementCount() * (type == core::GgmlType::kF32 ? 4 : 2);
      const auto address = reinterpret_cast<std::uintptr_t>(tensor->data);
      const bool mapped = std::any_of(
          reader->GetMappedRegions().begin(), reader->GetMappedRegions().end(),
          [&](const auto& region) {
            const auto begin = reinterpret_cast<std::uintptr_t>(region.data);
            return address >= begin && address - begin <= region.size &&
                   bytes <= region.size - (address - begin);
          });
      if (!mapped)
        throw std::invalid_argument("truncated Qwen vision tensor: " + name);
    };
    for (unsigned layer = 0; layer < 27; ++layer) {
      const auto prefix = "v.blk." + std::to_string(layer) + ".";
      for (const auto* norm : {"ln1", "ln2"}) {
        require(prefix + norm + ".weight", {kHidden}, core::GgmlType::kF32);
        require(prefix + norm + ".bias", {kHidden}, core::GgmlType::kF32);
      }
      for (const auto& [name, in, out] :
           std::array{std::tuple{"attn_qkv", kHidden, 3 * kHidden},
                      std::tuple{"attn_out", kHidden, kHidden},
                      std::tuple{"ffn_up", kHidden, kFfn},
                      std::tuple{"ffn_down", kFfn, kHidden}}) {
        require(prefix + name + ".weight", {in, out}, core::GgmlType::kBF16);
        require(prefix + name + ".bias", {out}, core::GgmlType::kF32);
      }
    }
    require("v.patch_embd.weight", {16, 16, 3, kHidden}, core::GgmlType::kF32);
    require("v.patch_embd.weight.1", {16, 16, 3, kHidden},
            core::GgmlType::kF32);
    require("v.patch_embd.bias", {kHidden}, core::GgmlType::kF32);
    require("v.position_embd.weight", {kHidden, 2304}, core::GgmlType::kF32);
    require("v.post_ln.weight", {kHidden}, core::GgmlType::kF32);
    require("v.post_ln.bias", {kHidden}, core::GgmlType::kF32);
    require("mm.0.weight", {kMergeHidden, kMergeHidden}, core::GgmlType::kBF16);
    require("mm.0.bias", {kMergeHidden}, core::GgmlType::kF32);
    require("mm.2.weight", {kMergeHidden, width}, core::GgmlType::kBF16);
    require("mm.2.bias", {width}, core::GgmlType::kF32);
    if (reader->GetTensorCount() != 334) {
      throw std::invalid_argument(
          "unexpected tensors in Qwen vision projector");
    }
  }

  ~Impl() {
    if (stream != nullptr)
      (void)hipStreamSynchronize(stream);
    lt.reset();
    weights.clear();
    if (blas != nullptr)
      (void)hipblasDestroy(blas);
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }

  void Upload() {
    if (uploaded)
      return;
    if (stream == nullptr)
      Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    Check(hipStreamSynchronize(stream));
    weights.clear();
    resident_bytes.store(0);
    std::size_t bytes_uploaded = 0;
    if (blas == nullptr)
      CheckBlas(hipblasCreate(&blas));
    CheckBlas(hipblasSetStream(blas, stream));
    lt = std::make_unique<gufo::hip::HipblasLtGemm>(
        gufo::hip::HipblasLtGemmOptions{.ignore_environment = true});
    PinnedUpload upload(stream);
    if (exl3) {
      UploadExl3(upload);
      uploaded = true;
      return;
    }
    for (const auto& tensor : reader->GetTensors()) {
      if (tensor.name == "v.patch_embd.weight" ||
          tensor.name == "v.patch_embd.weight.1")
        continue;
      const std::size_t bytes =
          tensor.ElementCount() * (tensor.type == core::GgmlType::kF32 ? 4 : 2);
      Allocation buffer = Allocation::Weight(bytes);
      bytes_uploaded += bytes;
      upload.Copy(buffer.data, tensor.data, bytes);
      weights.emplace(tensor.name, std::move(buffer));
    }
    const auto* first = static_cast<const float*>(
        reader->FindTensor("v.patch_embd.weight")->data);
    const auto* second = static_cast<const float*>(
        reader->FindTensor("v.patch_embd.weight.1")->data);
    std::vector<std::uint16_t> patch(kHidden * kPatchDim);
    for (unsigned h = 0; h < kHidden; ++h) {
      for (unsigned c = 0; c < 3; ++c) {
        for (unsigned t = 0; t < 2; ++t) {
          for (unsigned p = 0; p < 256; ++p) {
            patch[h * kPatchDim + (c * 2 + t) * 256 + p] =
                ToBf16((t == 0 ? first : second)[(h * 3 + c) * 256 + p]);
          }
        }
      }
    }
    Allocation buffer =
        Allocation::Weight(patch.size() * sizeof(std::uint16_t));
    upload.Copy(buffer.data, patch.data(),
                patch.size() * sizeof(std::uint16_t));
    weights.emplace("patch", std::move(buffer));
    Check(hipStreamSynchronize(stream));
    resident_bytes.store(bytes_uploaded + patch.size() * sizeof(std::uint16_t));
    uploaded = true;
  }

  void UploadExl3(PinnedUpload& upload) {
    linears.clear();
    max_inner_elements = 0;
    std::size_t bytes_uploaded = 0;
    const auto put = [&](const std::string& name, const void* data,
                         std::size_t bytes) -> void* {
      auto buffer = Allocation::Weight(bytes);
      upload.Copy(buffer.data, data, bytes);
      auto* pointer = buffer.data;
      weights.emplace(name, std::move(buffer));
      bytes_uploaded += bytes;
      return pointer;
    };
    for (const auto& [name, tensor] : exl3->tensors()) {
      auto values = Exl3Weights::FloatValues(*tensor);
      if (name != "v.position_embd.weight")
        for (auto& value : values)
          value =
              static_cast<float>(quant::exl3::HalfToDouble(NativeHalf(value)));
      put(name, values.data(), values.size() * sizeof(float));
    }
    for (const auto& [name, matrix] : exl3->matrices()) {
      if (matrix.raw) {
        const auto values = Exl3Weights::FloatValues(*matrix.raw);
        std::vector<std::uint16_t> half(values.size());
        std::transform(values.begin(), values.end(), half.begin(),
                       [](float x) { return NativeHalf(x); });
        put(name, half.data(), half.size() * sizeof(std::uint16_t));
        continue;
      }
      const auto& source = *matrix.quantized;
      quant::exl3::DeviceLinear device{
          .k = static_cast<std::uint32_t>(source.in_features),
          .n = static_cast<std::uint32_t>(source.out_features),
          .rate = source.rate,
          .codebook = source.codebook};
      device.trellis = static_cast<const std::uint16_t*>(
          put(name + ".trellis", source.trellis, device.TrellisBytes()));
      device.suh = static_cast<const std::uint16_t*>(
          put(name + ".suh", source.suh, device.k * sizeof(std::uint16_t)));
      device.svh = static_cast<const std::uint16_t*>(
          put(name + ".svh", source.svh, device.n * sizeof(std::uint16_t)));
      if (!gufo::hip::Exl3PrefillSupported(device))
        max_inner_elements =
            std::max(max_inner_elements, std::size_t{device.k} * device.n);
      linears.emplace(name, device);
    }
    // The graph's QKV RoPE consumes interleaved rows, with one concatenated
    // bias.
    for (unsigned layer = 0; layer < 27; ++layer) {
      const auto prefix = "v.blk." + std::to_string(layer) + ".attn_";
      if (!exl3->matrices().contains(prefix + "q.weight"))
        continue;
      std::vector<float> bias;
      for (const auto* part : {"q", "k", "v"}) {
        auto values = Exl3Weights::FloatValues(
            *exl3->tensors().at(prefix + part + ".bias"));
        for (auto& value : values)
          value =
              static_cast<float>(quant::exl3::HalfToDouble(NativeHalf(value)));
        bias.insert(bias.end(), values.begin(), values.end());
      }
      put(prefix + "qkv.bias", bias.data(), bias.size() * sizeof(float));
    }
    Check(hipStreamSynchronize(stream));
    resident_bytes.store(bytes_uploaded);
  }

  struct ProjectionScratch {
    Allocation input, rotated, output, inner_float, inner_half;
    ProjectionScratch(std::size_t rows, std::size_t inner)
        : input(rows * kMergeHidden * sizeof(float)),
          rotated(rows * kMergeHidden * sizeof(std::uint16_t)),
          output(rows * 5120 * sizeof(float)),
          inner_float(inner ? inner * sizeof(float) : 0),
          inner_half(inner ? inner * sizeof(std::uint16_t) : 0) {}
  };

  const float* FloatWeight(const std::string& name) const {
    return weights.at(name).As<float>();
  }
  unsigned FfnWidth(unsigned layer) const {
    if (!exl3)
      return kFfn;
    return static_cast<unsigned>(
        exl3->matrices()
            .at("v.blk." + std::to_string(layer) + ".ffn_up.weight")
            .out_features);
  }
  void Gemm(const std::string& weight, const void* input, float* output,
            unsigned rows, unsigned out, unsigned in,
            ProjectionScratch* scratch = nullptr, unsigned output_stride = 0) {
    if (!output_stride)
      output_stride = out;
    if (exl3 && weight.ends_with("attn_qkv.weight") &&
        !exl3->matrices().contains(weight)) {
      const auto prefix = weight.substr(0, weight.size() - 10);
      for (unsigned part = 0; part < 3; ++part) {
        const auto name = prefix + std::array{"q", "k", "v"}[part] + ".weight";
        Gemm(name, input, output + part * kHidden, rows, kHidden, in, scratch,
             3 * kHidden);
      }
      return;
    }
    const auto it = linears.find(weight);
    if (it != linears.end()) {
      const auto& linear = it->second;
      NativeInput<<<(rows * linear.k + kThreads - 1) / kThreads, kThreads, 0,
                    stream>>>(static_cast<const __half*>(input),
                              scratch->input.As<float>(), rows, in, linear.k);
      gufo::hip::LaunchExl3RotateRows(
          scratch->input.As<float>(), linear.k, rows, linear.k, linear.suh,
          scratch->rotated.As<std::uint16_t>(), stream);
      float* target = scratch->output.As<float>();
      if (gufo::hip::Exl3PrefillSupported(linear)) {
        gufo::hip::LaunchExl3PrefillGemm(linear, scratch->rotated.data, target,
                                         rows, false, stream);
      } else {
        // Generic EXL3 rates/codebooks use the same packed weights and FP16
        // rotated input. Decode into one bounded per-image slab, never a
        // persistent dense copy of the encoder. No retry after a failure.
        quant::exl3::LaunchReferenceDequantF32(
            linear, 0, linear.n, scratch->inner_float.As<float>(), stream);
        gufo::hip::LaunchFloatToFp16(scratch->inner_float.As<float>(),
                                     scratch->inner_half.data,
                                     std::size_t{linear.k} * linear.n, stream);
        constexpr float alpha = 1, beta = 0;
        CheckBlas(hipblasGemmEx(
            blas, HIPBLAS_OP_T, HIPBLAS_OP_N, linear.n, rows, linear.k, &alpha,
            scratch->inner_half.data, HIP_R_16F, linear.k,
            scratch->rotated.data, HIP_R_16F, linear.k, &beta, target,
            HIP_R_32F, linear.n, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
        quant::exl3::LaunchReferenceRotateRows(target, linear.n, rows, linear.n,
                                               nullptr, linear.svh, target,
                                               linear.n, stream);
      }
      NativeOutput<<<(rows * out + kThreads - 1) / kThreads, kThreads, 0,
                     stream>>>(target, output, rows, linear.n, out,
                               output_stride);
      return;
    }
    const auto* w = weights.at(weight).data;
    if (!exl3 && lt->RunBf16(w, input, output, rows, out, in, stream))
      return;
    constexpr float alpha = 1, beta = 0;
    const auto type = exl3 ? HIP_R_16F : HIP_R_16BF;
    unsigned weight_in = in;
    if (exl3) {
      weight_in =
          static_cast<unsigned>(exl3->matrices().at(weight).in_features);
      if (weight_in > in)
        throw std::logic_error("raw vision projection requires wider input");
    }
    CheckBlas(hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, out, rows,
                            weight_in, &alpha, w, type, weight_in, input, type,
                            in, &beta, output, HIP_R_32F, output_stride,
                            HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
  }
};

Encoder::Encoder(const std::filesystem::path& path, std::uint32_t width)
    : impl_(std::make_unique<Impl>(path, width)) {}
Encoder::~Encoder() = default;
std::shared_ptr<Encoder> Encoder::Open(const std::filesystem::path& target,
                                       const std::filesystem::path& sidecar,
                                       std::uint32_t width) {
  if (!sidecar.empty())
    return std::make_shared<Encoder>(sidecar, width);
  const bool native = std::filesystem::is_directory(target);
  if (native && quant::exl3::Checkpoint::IsExl3(target) &&
      Exl3Weights::HasVision(target))
    return std::make_shared<Encoder>(target, width);
  auto directory = native ? target : target.parent_path();
  for (unsigned level = 0; level < (native ? 1U : 2U); ++level) {
    const auto path = directory / "mmproj-BF16.gguf";
    if (std::filesystem::is_regular_file(path))
      return std::make_shared<Encoder>(path, width);
    directory = directory.parent_path();
  }
  return {};
}
const std::string& Encoder::identity() const {
  if (impl_->exl3)
    return impl_->exl3->identity();
  std::call_once(impl_->identity_once, [&] {
    // Hash the complete projector once. OpenSSL uses the CPU's SHA
    // instructions, avoiding seconds of scalar hashing on the first image.
    const std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> hash(
        EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!hash || EVP_DigestInit_ex(hash.get(), EVP_sha256(), nullptr) != 1)
      throw std::runtime_error("cannot initialize vision fingerprint");
    for (const auto& region : impl_->reader->GetMappedRegions()) {
      if (EVP_DigestUpdate(hash.get(), region.data, region.size) != 1)
        throw std::runtime_error("cannot hash vision projector");
    }
    std::array<unsigned char, 32> digest{};
    unsigned length = 0;
    if (EVP_DigestFinal_ex(hash.get(), digest.data(), &length) != 1 ||
        length != digest.size())
      throw std::runtime_error("cannot finish vision fingerprint");
    constexpr std::string_view hex = "0123456789abcdef";
    impl_->identity.reserve(digest.size() * 2);
    for (const auto byte : digest) {
      impl_->identity.push_back(hex[byte >> 4]);
      impl_->identity.push_back(hex[byte & 15]);
    }
  });
  return impl_->identity;
}
std::size_t Encoder::ResidentBytes() const noexcept {
  return impl_->resident_bytes.load();
}
bool Encoder::IsExl3() const noexcept {
  return impl_->exl3 != nullptr;
}
void Encoder::Upload() {
  impl_->Upload();
}
Encoder::Embedding::~Embedding() {
  if (data_ != nullptr)
    (void)hipFree(data_);
}

std::shared_ptr<const Encoder::Embedding> Encoder::Encode(
    const core::Image& image, const Observer& observer,
    const CancellationCheck& is_cancelled) {
  const auto check_cancelled = [&] {
    if (is_cancelled && is_cancelled())
      throw std::runtime_error("vision encoding cancelled");
  };
  check_cancelled();
  if (image.width == 0 || image.height == 0 || image.width % 32 != 0 ||
      image.height % 32 != 0 ||
      std::uint64_t{image.width} * image.height > 16777216 ||
      image.pixels.size() != std::size_t{image.width} * image.height * 3) {
    throw std::invalid_argument("vision encoder requires a resized RGB32 grid");
  }
  impl_->Upload();
  const auto encode = [&]<class T>() -> std::shared_ptr<const Embedding> {
    constexpr auto data_type = std::is_same_v<T, Half> ? HIP_R_16F : HIP_R_16BF;
    const auto stream = impl_->stream;
    const unsigned count = (image.width / 16) * (image.height / 16);
    const unsigned merged = count / 4;
    std::unique_ptr<Impl::ProjectionScratch> projection;
    if (impl_->exl3)
      projection = std::make_unique<Impl::ProjectionScratch>(
          count, impl_->max_inner_elements);
    const auto gemm = [&](const std::string& name, const T* input,
                          float* output, unsigned rows, unsigned out,
                          unsigned in) {
      impl_->Gemm(name, input, output, rows, out, in, projection.get());
    };
    const unsigned max_ffn = impl_->exl3 ? 4352U : kFfn;
    Allocation pixels(image.pixels.size());
    Allocation hidden(std::size_t{count} * kHidden * sizeof(float));
    Allocation work(std::size_t{count} * std::max(max_ffn, 3 * kHidden) *
                    sizeof(float));
    Allocation normalized(std::size_t{count} * std::max(max_ffn, kPatchDim) *
                          sizeof(T));
    Allocation qkv(std::size_t{3} * count * kHidden * sizeof(T));
    Allocation attention(std::size_t{count} * kHidden * sizeof(T));
    // Smaller attention tiles reduce scratch traffic for the 4096-patch
    // shape while preserving its BLAS reductions. Ragged tails require
    // their existing GEMM shapes to preserve rounding.
    const unsigned tile = std::min(count, count == 4096 ? 64U : 128U);
    Allocation scores(std::size_t{kHeads} * tile * count * sizeof(float));
    Allocation probabilities(std::size_t{kHeads} * tile * count * sizeof(T));
    Allocation output_tile(std::size_t{tile} * kHidden * sizeof(float));
    {
      // The input image can outlive encoding as part of a cached prompt.
      // Do not leave its pageable vector in ROCr's implicit SVM copy cache.
      PinnedUpload upload(stream, image.pixels.size());
      upload.Copy(pixels.data, image.pixels.data(), image.pixels.size());
    }
    Patchify<<<(count * kPatchDim + kThreads - 1) / kThreads, kThreads, 0,
               stream>>>(pixels.As<std::uint8_t>(), normalized.As<T>(),
                         image.width, image.width / 16, count);
    gemm("patch", normalized.As<T>(), hidden.As<float>(), count, kHidden,
         kPatchDim);
    PatchPosition<T>
        <<<(count * kHidden + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            hidden.As<float>(), impl_->FloatWeight("v.patch_embd.bias"),
            impl_->FloatWeight("v.position_embd.weight"), image.height / 16,
            image.width / 16, count, impl_->position_dtype);
    const auto observe = [&](const std::string& stage, const float* data,
                             std::size_t size) {
      if (!observer)
        return;
      std::vector<float> copy(size);
      Check(hipMemcpyAsync(copy.data(), data, size * sizeof(float),
                           hipMemcpyDeviceToHost, stream));
      Check(hipStreamSynchronize(stream));
      observer(stage, copy);
    };
    observe("patch", hidden.As<float>(), std::size_t{count} * kHidden);
    for (unsigned layer = 0; layer < 27; ++layer) {
      check_cancelled();
      const auto prefix = "v.blk." + std::to_string(layer) + ".";
      const auto ffn = impl_->FfnWidth(layer);
      LayerNorm<<<count, kThreads, 0, stream>>>(
          hidden.As<float>(), normalized.As<T>(),
          impl_->FloatWeight(prefix + "ln1.weight"),
          impl_->FloatWeight(prefix + "ln1.bias"), kHidden);
      gemm(prefix + "attn_qkv.weight", normalized.As<T>(), work.As<float>(),
           count, 3 * kHidden, kHidden);
      auto* q = qkv.As<T>();
      auto* k = q + count * kHidden;
      auto* v = k + count * kHidden;
      QkvRope<<<(count * kHidden + kThreads - 1) / kThreads, kThreads, 0,
                stream>>>(work.As<float>(),
                          impl_->FloatWeight(prefix + "attn_qkv.bias"), q, k, v,
                          count, image.width / 16);
      for (unsigned begin = 0; begin < count; begin += tile) {
        check_cancelled();
        const unsigned rows = std::min(tile, count - begin);
        constexpr float zero = 0, one = 1;
        CheckBlas(hipblasGemmStridedBatchedEx(
            impl_->blas, HIPBLAS_OP_T, HIPBLAS_OP_N, count, rows, kHeadDim,
            &one, k, data_type, kHeadDim, count * kHeadDim,
            q + begin * kHeadDim, data_type, kHeadDim, count * kHeadDim, &zero,
            scores.data, HIP_R_32F, count, rows * count, kHeads,
            HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
        if (count <= 1024)
          Softmax<4, T><<<kHeads * rows, kThreads, 0, stream>>>(
              scores.As<float>(), probabilities.As<T>(), count);
        else if (count <= 2048)
          Softmax<8, T><<<kHeads * rows, kThreads, 0, stream>>>(
              scores.As<float>(), probabilities.As<T>(), count);
        else if (count <= 4096)
          Softmax<16, T><<<kHeads * rows, kThreads, 0, stream>>>(
              scores.As<float>(), probabilities.As<T>(), count);
        else if (count <= 8192)
          Softmax<32, T><<<kHeads * rows, kThreads, 0, stream>>>(
              scores.As<float>(), probabilities.As<T>(), count);
        else
          Softmax<0, T><<<kHeads * rows, kThreads, 0, stream>>>(
              scores.As<float>(), probabilities.As<T>(), count);
        CheckBlas(hipblasGemmStridedBatchedEx(
            impl_->blas, HIPBLAS_OP_N, HIPBLAS_OP_N, kHeadDim, rows, count,
            &one, v, data_type, kHeadDim, count * kHeadDim, probabilities.data,
            data_type, count, rows * count, &zero, output_tile.data, HIP_R_32F,
            kHeadDim, rows * kHeadDim, kHeads, HIPBLAS_COMPUTE_32F,
            HIPBLAS_GEMM_DEFAULT));
        AttentionRows<<<(rows * kHidden + kThreads - 1) / kThreads, kThreads, 0,
                        stream>>>(output_tile.As<float>(), attention.As<T>(),
                                  begin, rows);
      }
      gemm(prefix + "attn_out.weight", attention.As<T>(), work.As<float>(),
           count, kHidden, kHidden);
      BiasResidual<T>
          <<<(count * kHidden + kThreads - 1) / kThreads, kThreads, 0,
             stream>>>(hidden.As<float>(), work.As<float>(),
                       impl_->FloatWeight(prefix + "attn_out.bias"),
                       count * kHidden, kHidden);
      LayerNorm<<<count, kThreads, 0, stream>>>(
          hidden.As<float>(), normalized.As<T>(),
          impl_->FloatWeight(prefix + "ln2.weight"),
          impl_->FloatWeight(prefix + "ln2.bias"), kHidden);
      gemm(prefix + "ffn_up.weight", normalized.As<T>(), work.As<float>(),
           count, ffn, kHidden);
      Activate<<<(count * ffn + kThreads - 1) / kThreads, kThreads, 0,
                 stream>>>(work.As<float>(),
                           impl_->FloatWeight(prefix + "ffn_up.bias"),
                           normalized.As<T>(), count * ffn, ffn, true);
      gemm(prefix + "ffn_down.weight", normalized.As<T>(), work.As<float>(),
           count, kHidden, ffn);
      BiasResidual<T>
          <<<(count * kHidden + kThreads - 1) / kThreads, kThreads, 0,
             stream>>>(hidden.As<float>(), work.As<float>(),
                       impl_->FloatWeight(prefix + "ffn_down.bias"),
                       count * kHidden, kHidden);
      observe("layer" + std::to_string(layer), hidden.As<float>(),
              std::size_t{count} * kHidden);
    }
    LayerNorm<<<count, kThreads, 0, stream>>>(
        hidden.As<float>(), normalized.As<T>(),
        impl_->FloatWeight("v.post_ln.weight"),
        impl_->FloatWeight("v.post_ln.bias"), kHidden);
    gemm("mm.0.weight", normalized.As<T>(), work.As<float>(), merged,
         kMergeHidden, kMergeHidden);
    Activate<<<(merged * kMergeHidden + kThreads - 1) / kThreads, kThreads, 0,
               stream>>>(work.As<float>(), impl_->FloatWeight("mm.0.bias"),
                         normalized.As<T>(), merged * kMergeHidden,
                         kMergeHidden, impl_->exl3 != nullptr);
    auto embedding = std::shared_ptr<Embedding>(new Embedding);
    embedding->rows_ = merged;
    embedding->width_ = impl_->output_width;
    Check(hipMalloc(&embedding->data_,
                    std::size_t{merged} * embedding->width_ * sizeof(float)));
    gemm("mm.2.weight", normalized.As<T>(), embedding->data_, merged,
         embedding->width_, kMergeHidden);
    Finish<T>
        <<<(merged * embedding->width_ + kThreads - 1) / kThreads, kThreads, 0,
           stream>>>(embedding->data_, impl_->FloatWeight("mm.2.bias"),
                     merged * embedding->width_, embedding->width_);
    Check(hipGetLastError());
    Check(hipStreamSynchronize(stream));
    check_cancelled();
    observe("embedding", embedding->data_,
            std::size_t{merged} * embedding->width_);
    return embedding;
  };
  if (impl_->exl3)
    return encode.template operator()<Half>();
  return encode.template operator()<Bf16>();
}
}  // namespace gufo::models::qwen::vision
