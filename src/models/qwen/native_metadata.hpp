#ifndef GUFO_MODELS_QWEN_NATIVE_METADATA_HPP_
#define GUFO_MODELS_QWEN_NATIVE_METADATA_HPP_

// GGUF-equivalent metadata (architecture qwen35) of a native Qwen3.5/3.8
// dense EXL3 pack directory, with the keys llama.cpp's convert_hf_to_gguf.py
// writes for the same model. The text model only: no MTP block.

#include <filesystem>
#include <memory>
#include <string>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen {

/// Metadata of the native EXL3 pack `directory` (text_config.model_type
/// "qwen3_5_text"), or nullptr with `error` set.
[[nodiscard]] std::unique_ptr<core::GgufReader> OpenNativeMetadata(
    const std::filesystem::path& directory, std::string* error = nullptr);

}  // namespace gufo::models::qwen

#endif  // GUFO_MODELS_QWEN_NATIVE_METADATA_HPP_
