#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_EXL3_WEIGHTS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_EXL3_WEIGHTS_HPP_

#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/quant/exl3/checkpoint.hpp"
#include "src/core/quant/exl3/device.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next {

/// Source descriptors borrow the checkpoint. Permutations move whole 128-wide
/// Hadamard blocks; no compressed payload is duplicated on the host.
struct Exl3HostLinear {
  quant::exl3::DeviceLinear view;
  std::uint32_t shard{0};
  std::uint64_t trellis_offset{0}, suh_offset{0}, svh_offset{0};
  std::vector<std::uint32_t> input_blocks, output_blocks;
};

struct Exl3Storage {
  std::shared_ptr<quant::exl3::Checkpoint> checkpoint;
  std::deque<std::vector<Exl3HostLinear>> linears;
  std::deque<std::vector<std::uint8_t>> converted;
  std::deque<std::string> names;
  /// Only missing in-shard MTP mixer tensors; bytes live in converted.
  std::map<std::string, quant::exl3::Tensor, std::less<>> supplementary;
};

/// Bind actual per-tensor K3..K8 or K3.5 rates. Every expert in this bank
/// must agree; different banks/projections may use different rates.
[[nodiscard]] TensorRef BindExl3LinearBank(Exl3Storage& storage,
                                           const std::string& module,
                                           std::uint32_t k, std::uint32_t n,
                                           std::uint32_t experts = 1);
/// Prefer in-shard MTP mixer operands. A patch supplies missing tensors;
/// overlapping operands must have identical dtype, shape and payload bytes.
void PrepareExl3MtpMixer(Exl3Storage& storage, const Config& config,
                         const std::filesystem::path& patch = {});

/// Binds the native EXL3 pack directory `path` whose metadata
/// OpenArtifactMetadata built. Header/metadata validation only; projection
/// payloads remain demand mapped.
[[nodiscard]] std::optional<ModelWeights> LoadExl3Model(
    const std::filesystem::path& path, const core::GgufReader& metadata,
    std::string* error);
/// The MTP block of the same native pack as `trunk`.
[[nodiscard]] std::optional<MtpWeights> LoadExl3Mtp(
    const std::filesystem::path& path, const core::GgufReader& metadata,
    const ModelWeights& trunk, std::string* error);

/// Grouped Hugging Face V head -> tiled runtime head.
[[nodiscard]] std::uint32_t Exl3VHeadDestination(std::uint32_t index,
                                                 std::uint32_t head_dim,
                                                 const Config& config);
/// Independently usable CPU projection, with lossless block permutations.
/// decode_storage includes the decode path's FP32 input rotation and FP16
/// storage boundary; the default retains the mathematical FP64 formula.
void Exl3MatVec(const Exl3HostLinear& linear, std::span<const float> x,
                std::span<float> y, bool decode_storage = false);

}  // namespace gufo::models::qwen38_flash_next
#endif
