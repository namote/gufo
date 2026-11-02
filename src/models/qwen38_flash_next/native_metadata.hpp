#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_NATIVE_METADATA_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_NATIVE_METADATA_HPP_
#include "src/core/gguf_reader.hpp"
namespace gufo::models::qwen38_flash_next {
// File inputs keep their existing reader. Directory inputs must be a native
// Flash-Next EXL3 pack; failures never select a different weight source.
std::unique_ptr<core::GgufReader> OpenArtifactMetadata(
    const std::filesystem::path& path, std::string* error = nullptr);
void ValidateNativeSources(const core::GgufReader& metadata);
std::string NativeArtifactIdentity(const core::GgufReader& metadata);
}  // namespace gufo::models::qwen38_flash_next
#endif
