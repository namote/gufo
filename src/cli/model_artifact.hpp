#ifndef GUFO_CLI_MODEL_ARTIFACT_HPP_
#define GUFO_CLI_MODEL_ARTIFACT_HPP_

#include <filesystem>
#include <memory>
#include <string>

#include "src/core/gguf_reader.hpp"

namespace gufo::cli {

/// Metadata of a text model artifact: a GGUF file (or its first shard), or a
/// native EXL3 pack directory, dispatched on config.json
/// text_config.model_type (Qwen3.5/3.8 dense or Qwen3.8 Flash-Next). EXL3
/// GGUF metadata sidecars are rejected. Returns nullptr with `error` set.
[[nodiscard]] std::unique_ptr<core::GgufReader> OpenModelArtifact(
    const std::filesystem::path& path, std::string* error = nullptr);

}  // namespace gufo::cli

#endif  // GUFO_CLI_MODEL_ARTIFACT_HPP_
