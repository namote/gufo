#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/hip/weight_upload.hpp"
#include "src/core/platform/mapped_file.hpp"
#include "src/core/platform/pages.hpp"
#include "src/core/platform/prefault.hpp"
#include "src/core/quant/exl3/coverage.hpp"
#include "src/core/quant/exl3/layout.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/exl3_weights.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/native_pack.hpp"

namespace gufo::hip {
namespace {

void ReleaseWeightRegions(std::vector<QwenGpuWeightRegion>& regions) noexcept {
  for (auto& region : regions) {
    if (region.owns_device_allocation && region.device_data != nullptr) {
      (void)hipFree(region.device_data);
    } else if (region.host_copy != nullptr) {
      (void)hipHostUnregister(region.host_copy);
      platform::ReleaseCommittedPages(region.host_copy, region.size);
    }
    region = {};
  }
}

[[nodiscard]] hipError_t MapRegisteredRegion(
    const core::GgufMappedRegion& source, QwenGpuWeightRegion& destination,
    bool& source_was_read) {
  source_was_read = false;
  void* host_copy =
      platform::CloneMappedWeightsToAnonymousPages(source.data, source.size);
  if (host_copy == nullptr)
    return hipErrorOutOfMemory;
  source_was_read = true;
  const auto unmap = [&](void* pointer) {
    platform::ReleaseCommittedPages(pointer, source.size);
  };
  std::unique_ptr<void, decltype(unmap)> owned_copy(host_copy, unmap);
  const auto register_error = hipHostRegister(
      host_copy, source.size, hipHostRegisterMapped | hipHostRegisterReadOnly);
  if (register_error != hipSuccess)
    return register_error;

  void* device_data = nullptr;
  const auto pointer_error =
      hipHostGetDevicePointer(&device_data, host_copy, 0);
  if (pointer_error != hipSuccess) {
    (void)hipHostUnregister(host_copy);
    return pointer_error;
  }

  destination = {.host_data = source.data,
                 .device_data = device_data,
                 .host_copy = owned_copy.release(),
                 .size = source.size};
  return hipSuccess;
}

/// Device-resident placement: one hipMalloc copy in (on this APU target)
/// carve-out memory, filled from the cached mapping or by FillDeviceRegions.
[[nodiscard]] hipError_t AllocateDeviceRegion(
    const core::GgufMappedRegion& source, QwenGpuWeightRegion& destination) {
  void* device_data = nullptr;
  const auto status = hipMalloc(&device_data, source.size);
  if (status != hipSuccess)
    return status;
  destination = {.host_data = source.data,
                 .device_data = device_data,
                 .host_copy = nullptr,
                 .size = source.size,
                 .owns_device_allocation = true};
  return hipSuccess;
}

/// Streams one region from the read-only mapping through a pinned staging
/// buffer. Fallback for sources WeightUpload cannot read, such as in-memory
/// images without a file.
[[nodiscard]] hipError_t CopyMappedRegion(const core::GgufMappedRegion& source,
                                          void* device_data) {
  platform::PrefaultMappedRange(source.data, source.size);

  constexpr std::size_t kChunkBytes = 16ULL << 20;
  void* staging = nullptr;
  hipError_t status = hipHostMalloc(&staging, kChunkBytes);
  if (status != hipSuccess)
    return status;
  const auto free_staging = [&](void* pointer) { (void)hipHostFree(pointer); };
  std::unique_ptr<void, decltype(free_staging)> owned_staging(staging,
                                                              free_staging);

  const auto chunks =
      source.size / kChunkBytes + (source.size % kChunkBytes != 0);
  for (std::size_t index = 0; index < chunks; ++index) {
    const auto offset = index * kChunkBytes;
    const auto bytes = std::min(kChunkBytes, source.size - offset);
    std::memcpy(staging, static_cast<const std::uint8_t*>(source.data) + offset,
                bytes);
    status = hipMemcpy(static_cast<std::uint8_t*>(device_data) + offset,
                       staging, bytes, hipMemcpyHostToDevice);
    if (status != hipSuccess)
      return status;
  }
  return hipSuccess;
}

/// Fills device-resident regions from their shard files without touching the
/// mappings. Copying through a mapping faults the whole shard into host memory
/// first; on Windows those pages also stay in the process working set, so a
/// 16 GiB target needs as much free RAM beside the GPU carve-out. WeightUpload
/// keeps host residency at its staging buffers.
[[nodiscard]] bool FillDeviceRegions(
    std::span<const core::GgufMappedRegion> sources,
    std::span<QwenGpuWeightRegion> destinations,
    std::span<const std::size_t> indices, std::string* error_msg) {
  std::string upload_error;
  if (auto uploader = WeightUpload::Create(sources, &upload_error)) {
    for (const auto index : indices) {
      if (!uploader->Copy(static_cast<std::uint32_t>(index), 0,
                          sources[index].size, destinations[index].device_data,
                          error_msg))
        break;
    }
    // Finish waits for every queued copy, also after a failed Copy, so the
    // destinations stay valid until the uploader is gone.
    return uploader->Finish(error_msg);
  }
  for (const auto index : indices) {
    const auto status =
        CopyMappedRegion(sources[index], destinations[index].device_data);
    if (status != hipSuccess) {
      if (error_msg != nullptr) {
        *error_msg = "Failed to upload GGUF shard " + std::to_string(index) +
                     ": " + hipGetErrorString(status);
      }
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool CreateWeightRegions(
    const core::GgufReader& reader,
    std::vector<QwenGpuWeightRegion>& weight_regions, std::string* error_msg) {
  const auto source_regions = reader.GetMappedRegions();
  if (source_regions.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF reader has no mapped weight regions";
    }
    return false;
  }

  weight_regions.resize(source_regions.size());
  std::vector<std::size_t> device_regions;
  for (std::size_t i = 0; i < source_regions.size(); ++i) {
    const auto& source = source_regions[i];
    auto& destination = weight_regions[i];
    hipError_t map_error;
    try {
      if (source.file_descriptor == platform::kInvalidFile) {
        // A memory image (a native EXL3 pack's GGUF image) lives on the heap:
        // it is neither page aligned nor a mapping that may be dropped, so
        // it is copied to device memory instead of cloned and registered.
        map_error = AllocateDeviceRegion(source, destination);
        if (map_error == hipSuccess)
          map_error = CopyMappedRegion(source, destination.device_data);
        if (map_error != hipSuccess) {
          ReleaseWeightRegions(weight_regions);
          if (error_msg != nullptr)
            *error_msg = std::string("Failed to upload a GGUF memory image: ") +
                         hipGetErrorString(map_error);
          return false;
        }
        continue;
      }
      bool source_was_read = false;
      map_error = MapRegisteredRegion(source, destination, source_was_read);
      if (map_error != hipSuccess) {
        // Mapped registration is the Linux fast path; the device-resident
        // upload preserves single-copy residency wherever it is refused.
        destination = {};
        map_error = AllocateDeviceRegion(source, destination);
        if (map_error == hipSuccess) {
          if (source_was_read) {
            // A successful clone has already read the mapping. Reuse those
            // cached bytes if registration fails instead of rereading the
            // shard with direct I/O. When cloning is unavailable (Windows),
            // keep the bounded file pipeline without touching the mapping.
            map_error = CopyMappedRegion(source, destination.device_data);
          } else {
            device_regions.push_back(i);
          }
        }
      }
    } catch (const std::exception& e) {
      ReleaseWeightRegions(weight_regions);
      if (error_msg)
        *error_msg = e.what();
      return false;
    }
    if (map_error != hipSuccess || destination.device_data == nullptr) {
      ReleaseWeightRegions(weight_regions);
      if (error_msg != nullptr) {
        *error_msg = "Failed to make GGUF shard " + std::to_string(i) +
                     " GPU-visible: " + hipGetErrorString(map_error);
      }
      return false;
    }
  }
  if (!device_regions.empty() &&
      !FillDeviceRegions(source_regions, weight_regions, device_regions,
                         error_msg)) {
    ReleaseWeightRegions(weight_regions);
    return false;
  }
  return true;
}

[[nodiscard]] bool RemapTensor(
    models::QwenTensorRef& tensor,
    std::span<const QwenGpuWeightRegion> regions) noexcept {
  if (tensor.empty()) {
    return true;
  }
  const auto tensor_address = reinterpret_cast<std::uintptr_t>(tensor.data);
  for (const auto& region : regions) {
    const auto region_address =
        reinterpret_cast<std::uintptr_t>(region.host_data);
    if (tensor_address >= region_address) {
      const auto offset = tensor_address - region_address;
      const std::size_t encoded_bytes = tensor.EncodedSizeBytes();
      if (offset < region.size && encoded_bytes != 0 &&
          encoded_bytes <= region.size - offset) {
        tensor.data =
            static_cast<const std::uint8_t*>(region.device_data) + offset;
        tensor.available_bytes = region.size - offset;
        return true;
      }
    }
  }
  return false;
}

// True when `tensor` is one of the alpha/beta copies folded at load.
bool IsFolded(const models::QwenModelWeights& weights,
              const models::QwenTensorRef& tensor) noexcept {
  if (weights.folded_weights == nullptr || tensor.data == nullptr) {
    return false;
  }
  const auto* begin = weights.folded_weights->data();
  const auto* pointer = static_cast<const std::uint8_t*>(tensor.data);
  return pointer >= begin && pointer < begin + weights.folded_weights->size();
}

// Copies the packed rotation signs and the alpha/beta matrices folded at
// load to device memory once and rebinds every pointer to the copy. Graph
// capture then sees fixed device pointers and no allocation happens during
// decode.
bool UploadRotationSigns(models::QwenModelWeights& weights, void** device,
                         std::string* error_msg) {
  *device = nullptr;
  if (!weights.IsRotated() || weights.rotation_signs == nullptr) {
    return true;
  }
  const auto& host = *weights.rotation_signs;
  const std::size_t sign_bytes = host.size() * sizeof(std::uint32_t);
  // Folded matrices start on a 256-byte boundary like the GEMV kernels'
  // other weights in a well-aligned file.
  const std::size_t folded_offset = (sign_bytes + 255) / 256 * 256;
  const std::size_t folded_bytes =
      weights.folded_weights != nullptr ? weights.folded_weights->size() : 0;
  const std::size_t bytes = folded_offset + folded_bytes;
  void* buffer = nullptr;
  hipError_t status = hipMalloc(&buffer, bytes);
  if (status == hipSuccess) {
    status = hipMemcpy(buffer, host.data(), sign_bytes, hipMemcpyHostToDevice);
  }
  if (status == hipSuccess && folded_bytes != 0) {
    status = hipMemcpy(static_cast<std::uint8_t*>(buffer) + folded_offset,
                       weights.folded_weights->data(), folded_bytes,
                       hipMemcpyHostToDevice);
  }
  if (status != hipSuccess) {
    if (buffer != nullptr) {
      (void)hipFree(buffer);
    }
    if (error_msg != nullptr) {
      *error_msg = std::string("Failed to upload rotation signs: ") +
                   hipGetErrorString(status);
    }
    return false;
  }
  const auto* base = host.data();
  auto* device_base = static_cast<const std::uint32_t*>(buffer);
  const auto rebind = [&](const std::uint32_t*& pointer) {
    if (pointer != nullptr) {
      pointer = device_base + (pointer - base);
    }
  };
  for (auto& layer : weights.layers) {
    rebind(layer.rotation.attn_in);
    rebind(layer.rotation.ssm_out_in);
    rebind(layer.rotation.attn_out_in);
    rebind(layer.rotation.ffn_in);
    rebind(layer.rotation.ffn_down_in);
  }
  rebind(weights.head_in_rotation);
  if (folded_bytes != 0) {
    const auto* folded_base = weights.folded_weights->data();
    auto* device_folded = static_cast<std::uint8_t*>(buffer) + folded_offset;
    for (auto& layer : weights.layers) {
      for (auto* tensor : {&layer.ssm_alpha, &layer.ssm_beta}) {
        if (IsFolded(weights, *tensor)) {
          const auto offset =
              static_cast<const std::uint8_t*>(tensor->data) - folded_base;
          tensor->data = device_folded + offset;
          tensor->available_bytes = folded_bytes - offset;
        }
      }
    }
  }
  *device = buffer;
  return true;
}

std::vector<models::QwenTensorRef*> Exl3Refs(models::QwenModelWeights& w) {
  std::vector<models::QwenTensorRef*> refs;
  const auto add = [&](models::QwenTensorRef& t) {
    if (t.exl3 != nullptr)
      refs.push_back(&t);
  };
  add(w.output);
  for (auto& l : w.layers)
    for (auto* t : {&l.attn_q, &l.attn_k, &l.attn_v, &l.attn_output,
                    &l.attn_qkv, &l.attn_gate, &l.ssm_out, &l.ssm_alpha,
                    &l.ssm_beta, &l.ffn_gate, &l.ffn_up, &l.ffn_down})
      add(*t);
  return refs;
}

// Uploads every EXL3 projection into one allocation (every trellis and scale
// vector 256-byte aligned), rebinds the references to the device descriptors
// and drops the host copies. Returns the device bytes, or 0 on failure.
std::size_t UploadExl3(models::QwenModelWeights& weights,
                       std::string* error_msg) {
  constexpr std::size_t kAlign = 256;
  const auto align = [](std::size_t v) {
    return (v + kAlign - 1) / kAlign * kAlign;
  };
  const auto refs = Exl3Refs(weights);
  auto storage = std::make_shared<models::QwenExl3Storage>();
  for (const auto* ref : refs) {
    const auto& h = *ref->exl3;
    storage->device_bytes +=
        align(h.TrellisBytes()) + align(h.k * 2U) + align(h.n * 2U);
  }
  void* buffer = nullptr;
  if (const auto status = hipMalloc(&buffer, storage->device_bytes);
      status != hipSuccess) {
    if (error_msg != nullptr)
      *error_msg =
          std::string("EXL3 projection upload: ") + hipGetErrorString(status);
    return 0;
  }
  storage->device_buffer.reset(buffer, [](void* p) { (void)hipFree(p); });
  storage->device.reserve(refs.size());  // views must not move
  auto* base = static_cast<std::uint8_t*>(buffer);
  std::size_t offset = 0;
  const auto copy = [&](const void* src, std::size_t bytes) {
    auto* dst = base + offset;
    offset += align(bytes);
    if (hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice) != hipSuccess)
      throw std::runtime_error("EXL3 projection upload copy failed");
    return reinterpret_cast<const std::uint16_t*>(dst);
  };
  try {
    for (auto* ref : refs) {
      auto view = *ref->exl3;
      view.trellis = copy(view.trellis, view.TrellisBytes());
      view.suh = copy(view.suh, view.k * 2U);
      view.svh = copy(view.svh, view.n * 2U);
      storage->device.push_back(view);
      ref->exl3 = &storage->device.back();
      ref->data = view.trellis;
      ref->available_bytes = view.TrellisBytes();
    }
  } catch (const std::exception& e) {
    if (error_msg != nullptr)
      *error_msg = e.what();
    return 0;
  }
  weights.exl3_storage = storage;  // releases the host copies
  std::vector<const quant::exl3::DeviceLinear*> views;
  views.reserve(storage->device.size());
  for (const auto& view : storage->device)
    views.push_back(&view);
  quant::exl3::LogReferencePath("qwen", views);
  return storage->device_bytes;
}

}  // namespace

QwenGpuModel::QwenGpuModel(
    std::shared_ptr<const core::GgufReader> reader,
    models::QwenModelWeights weights,
    std::shared_ptr<const tokenization::QwenTokenizer> tokenizer,
    std::vector<QwenGpuWeightRegion> weight_regions,
    std::shared_ptr<models::qwen::vision::Encoder> vision, void* rotation_signs,
    std::size_t exl3_bytes,
    std::shared_ptr<const models::qwen::AbliterationProfile> abliteration,
    float* abliteration_directions)
    : reader_(std::move(reader)),
      weights_(std::move(weights)),
      tokenizer_(std::move(tokenizer)),
      weight_regions_(std::move(weight_regions)),
      vision_(std::move(vision)),
      rotation_signs_(rotation_signs),
      exl3_bytes_(exl3_bytes),
      abliteration_(std::move(abliteration)),
      abliteration_directions_(abliteration_directions) {}

QwenGpuModel::~QwenGpuModel() {
  if (rotation_signs_ != nullptr) {
    (void)hipFree(rotation_signs_);
  }
  if (abliteration_directions_ != nullptr)
    (void)hipFree(abliteration_directions_);
  ReleaseWeightRegions(weight_regions_);
}

std::size_t QwenGpuModel::GetResidentBytes() const noexcept {
  std::size_t total = (vision_ ? vision_->ResidentBytes() : 0) + exl3_bytes_;
  for (const auto& region : weight_regions_) {
    if (region.size > std::numeric_limits<std::size_t>::max() - total) {
      return std::numeric_limits<std::size_t>::max();
    }
    total += region.size;
  }
  return total;
}

std::shared_ptr<const QwenGpuModel> QwenGpuModel::CreateFromGguf(
    std::shared_ptr<const core::GgufReader> reader, std::string* error_msg,
    std::shared_ptr<models::qwen::vision::Encoder> vision,
    const std::string& abliteration_profile) {
  if (reader == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF reader must not be null";
    }
    return nullptr;
  }
  // The artifact as opened: a native pack's reader is replaced below.
  const auto artifact = reader;

  if (!tokenization::QwenChatTemplate::ValidateGgufTemplate(*reader,
                                                            error_msg)) {
    return nullptr;
  }

  std::optional<models::QwenModelWeights> weights_opt;
  if (reader->GetMetadataBool(quant::exl3::kNativeKey).value_or(false)) {
    // A native EXL3 pack directory: the projections stay EXL3, the rest
    // becomes an in-memory GGUF image carrying the native metadata.
    auto loaded = models::qwen_exl3::LoadFromDirectory(*reader, error_msg);
    if (!loaded)
      return nullptr;
    reader = std::move(loaded->reader);
    weights_opt = std::move(loaded->weights);
  } else if (reader->FindMetadata(quant::exl3::kCheckpointKey)) {
    if (error_msg != nullptr)
      *error_msg =
          "EXL3 metadata sidecar files are not supported; pass the EXL3 pack "
          "directory";
    return nullptr;
  } else {
    weights_opt = models::QwenModelWeights::LoadFromGguf(*reader, error_msg);
  }
  if (!weights_opt.has_value()) {
    return nullptr;
  }
  std::shared_ptr<const models::qwen::AbliterationProfile> abliteration;
  try {
    // Only the explicit option loads a profile; none is discovered.
    if (!abliteration_profile.empty())
      abliteration = models::qwen::LoadAbliterationOption(
          abliteration_profile, weights_opt->config.hidden_size,
          weights_opt->config.num_layers,
          models::qwen::ArtifactIdentity(*artifact),
          models::qwen::kAbliterationWriterBeforeResidual);
    // Detection (dynamic profiles) runs only in the Flash-Next engine.
    if (abliteration && abliteration->trigger)
      throw std::invalid_argument(
          "abliteration: Qwen3.8-27B applies no profile 'trigger'; "
          "load a profile without one");
  } catch (const std::exception& e) {
    if (error_msg != nullptr)
      *error_msg = e.what();
    return nullptr;
  }

  auto tokenizer =
      tokenization::QwenTokenizer::CreateFromGguf(*reader, error_msg);
  if (!tokenizer)
    return nullptr;

  std::vector<QwenGpuWeightRegion> weight_regions;
  if (!CreateWeightRegions(*reader, weight_regions, error_msg)) {
    return nullptr;
  }

  const auto remap = [&](models::QwenTensorRef& tensor) {
    // Folded alpha/beta live in host memory until UploadRotationSigns; EXL3
    // projections are uploaded by UploadExl3.
    return tensor.exl3 != nullptr || IsFolded(*weights_opt, tensor) ||
           RemapTensor(tensor, weight_regions);
  };
  bool remapped = remap(weights_opt->token_embd) &&
                  remap(weights_opt->output_norm) && remap(weights_opt->output);
  for (auto& layer : weights_opt->layers) {
    remapped = remapped && remap(layer.attn_norm) && remap(layer.attn_q) &&
               remap(layer.attn_k) && remap(layer.attn_v) &&
               remap(layer.attn_output) && remap(layer.attn_q_norm) &&
               remap(layer.attn_k_norm) && remap(layer.attn_qkv) &&
               remap(layer.attn_gate) && remap(layer.ssm_out) &&
               remap(layer.ssm_conv1d) && remap(layer.ssm_alpha) &&
               remap(layer.ssm_beta) && remap(layer.ssm_a) &&
               remap(layer.ssm_dt) && remap(layer.ssm_norm) &&
               remap(layer.ffn_norm) && remap(layer.ffn_gate) &&
               remap(layer.ffn_up) && remap(layer.ffn_down);
  }
  if (!remapped) {
    ReleaseWeightRegions(weight_regions);
    if (error_msg != nullptr) {
      *error_msg = "A Qwen tensor does not belong to any mapped GGUF shard";
    }
    return nullptr;
  }

  std::size_t exl3_bytes = 0;
  if (weights_opt->exl3_storage != nullptr) {
    exl3_bytes = UploadExl3(*weights_opt, error_msg);
    if (exl3_bytes == 0) {
      ReleaseWeightRegions(weight_regions);
      return nullptr;
    }
  }

  void* rotation_signs = nullptr;
  if (!UploadRotationSigns(*weights_opt, &rotation_signs, error_msg)) {
    ReleaseWeightRegions(weight_regions);
    return nullptr;
  }

  float* abliteration_directions = nullptr;
  if (abliteration != nullptr) {
    // Every direction row at once (64 x 5120 floats for a v2 27B profile),
    // so a request switches direction without an upload.
    const auto bytes = abliteration->directions.size() * sizeof(float);
    void* direction = nullptr;
    hipError_t status = hipMalloc(&direction, bytes);
    if (status == hipSuccess)
      status = hipMemcpy(direction, abliteration->directions.data(), bytes,
                         hipMemcpyHostToDevice);
    if (status != hipSuccess) {
      if (direction != nullptr)
        (void)hipFree(direction);
      if (rotation_signs != nullptr)
        (void)hipFree(rotation_signs);
      ReleaseWeightRegions(weight_regions);
      if (error_msg != nullptr)
        *error_msg = std::string("abliteration directions upload failed: ") +
                     hipGetErrorString(status);
      return nullptr;
    }
    abliteration_directions = static_cast<float*>(direction);
  }

  std::shared_ptr<const tokenization::QwenTokenizer> shared_tokenizer(
      std::move(tokenizer));
  return std::make_shared<const QwenGpuModel>(
      std::move(reader), std::move(*weights_opt), std::move(shared_tokenizer),
      std::move(weight_regions), std::move(vision), rotation_signs, exl3_bytes,
      std::move(abliteration), abliteration_directions);
}

std::unique_ptr<QwenGpuExecutor> QwenGpuExecutor::Create(
    std::shared_ptr<const QwenGpuModel> model, std::string* error_msg,
    std::uint32_t max_context, QwenExecutionPolicy policy) {
  if (model == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "Qwen GPU model must not be null";
    }
    return nullptr;
  }
  if (const auto unsupported = QwenExecutionPolicyUnsupportedReason(policy);
      !unsupported.empty()) {
    if (error_msg != nullptr) {
      *error_msg = unsupported;
    }
    return nullptr;
  }

  const auto& config = model->GetConfig();
  const std::uint32_t model_context =
      config.context_length > 0 ? config.context_length : max_context;
  if (max_context == 0 || max_context > model_context) {
    if (error_msg != nullptr) {
      *error_msg = "Requested GPU context exceeds the model context length";
    }
    return nullptr;
  }

  return std::make_unique<QwenGpuExecutor>(std::move(model), max_context,
                                           policy);
}

std::unique_ptr<QwenGpuExecutor> QwenGpuExecutor::CreateFromGguf(
    std::shared_ptr<const core::GgufReader> reader, std::string* error_msg,
    std::uint32_t max_context, QwenExecutionPolicy policy) {
  // Refuse before mapping weights; Create repeats the same check.
  if (const auto unsupported = QwenExecutionPolicyUnsupportedReason(policy);
      !unsupported.empty()) {
    if (error_msg != nullptr) {
      *error_msg = unsupported;
    }
    return nullptr;
  }
  auto model = QwenGpuModel::CreateFromGguf(std::move(reader), error_msg);
  if (model == nullptr) {
    return nullptr;
  }
  return Create(std::move(model), error_msg, max_context, policy);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
