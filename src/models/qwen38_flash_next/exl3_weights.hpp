#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_EXL3_WEIGHTS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_EXL3_WEIGHTS_HPP_

#include <deque>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/quant/exl3/checkpoint.hpp"
#include "src/core/quant/exl3/device.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next {

inline constexpr std::string_view kExl3Checkpoint = "gufo.exl3.checkpoint";
inline constexpr std::string_view kExl3PleSource = "gufo.exl3.ple_source";
inline constexpr std::string_view kExl3MtpMixer = "gufo.exl3.mtp_mixer";

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
  std::shared_ptr<core::GgufReader> ple_reader;
  std::deque<std::vector<Exl3HostLinear>> linears;
  std::deque<std::vector<std::uint8_t>> converted;
  std::deque<std::string> names;
};

/// Identity of retained GGUF shard filenames, sizes and aligned headers.
[[nodiscard]] std::string Exl3PleHeadersDigest(const core::GgufReader& reader);

/// Header/metadata validation only; projection payloads remain demand mapped.
[[nodiscard]] std::optional<ModelWeights> LoadExl3Model(
    const std::filesystem::path& path, const core::GgufReader& sidecar,
    std::string* error);
[[nodiscard]] std::optional<MtpWeights> LoadExl3Mtp(
    const std::filesystem::path& path, const core::GgufReader& sidecar,
    const ModelWeights& trunk, std::string* error);

/// Grouped Hugging Face V head -> tiled runtime head.
[[nodiscard]] std::uint32_t Exl3VHeadDestination(std::uint32_t index,
                                                 std::uint32_t head_dim,
                                                 const Config& config);
/// Independently usable CPU projection, with lossless block permutations.
void Exl3MatVec(const Exl3HostLinear& linear, std::span<const float> x,
                std::span<float> y);

}  // namespace gufo::models::qwen38_flash_next
#endif
