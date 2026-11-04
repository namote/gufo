#ifndef GUFO_MODELS_QWEN_NATIVE_PACK_HPP_
#define GUFO_MODELS_QWEN_NATIVE_PACK_HPP_

// GGUF-equivalent metadata of a native Qwen EXL3 pack directory, shared by
// the Qwen families that load such packs. The pack supplies config.json,
// tokenizer.json, tokenizer_config.json, chat_template.jinja,
// model.safetensors.index.json and its indexed shards. A family adds its
// architecture keys; Finish() adds the tokenizer, chat template, EXL3 source
// and source-identity keys. No GGUF bytes, tensors or files are created.

#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"

namespace gufo::models::qwen {

/// Member `key` of `value`; throws when it is missing.
[[nodiscard]] const json::Value& NativeMember(const json::Value& value,
                                              const char* key);
/// Boolean member `key` of `value`; throws when missing or not a boolean.
[[nodiscard]] bool NativeBoolean(const json::Value& value, const char* key);
/// A non-negative integer no larger than UINT32_MAX; throws otherwise.
[[nodiscard]] std::uint64_t NativeUInt(const json::Value& value);

class NativePackMetadata {
public:
  /// Validates `directory` as an EXL3 pack whose text_config.model_type is
  /// `model_type`, stamps every source file (the common files, the indexed
  /// shards, `required_files` and those `optional_files` present) and parses
  /// the configuration and tokenizer files. Throws std::runtime_error naming
  /// the problem.
  NativePackMetadata(const std::filesystem::path& directory,
                     std::string_view model_type,
                     std::span<const std::string_view> required_files = {},
                     std::span<const std::string_view> optional_files = {});

  [[nodiscard]] const std::filesystem::path& directory() const noexcept {
    return directory_;
  }
  [[nodiscard]] const json::Value& config() const noexcept { return config_; }
  [[nodiscard]] const json::Value& text_config() const;
  [[nodiscard]] const json::Value& tokenizer() const noexcept {
    return tokenizer_;
  }
  [[nodiscard]] const json::Value& tokenizer_config() const noexcept {
    return tokenizer_config_;
  }
  /// text_config member `key` as NativeUInt.
  [[nodiscard]] std::uint64_t Number(const char* key) const;

  void Text(std::string key, std::string value);
  void U32(std::string key, std::uint64_t value);
  void F32(std::string key, double value);
  void Bool(std::string key, bool value);
  void Array(std::string key, std::vector<std::uint64_t> values);

  /// Adds the tokenizer (tokens, merges, token types, BPE model and qwen35
  /// pre-tokenizer, EOS and padding IDs from tokenizer_config.json) and the
  /// chat template. With `bos_token_id`, also the BOS ID and
  /// add_bos_token = false. Then rechecks every source stamp and returns the
  /// metadata reader.
  [[nodiscard]] std::unique_ptr<core::GgufReader> Finish(
      std::optional<std::uint64_t> bos_token_id = std::nullopt);

private:
  std::string_view Own(std::string value);

  std::filesystem::path directory_;
  json::Value config_;
  json::Value tokenizer_;
  json::Value tokenizer_config_;
  std::vector<std::string> file_names_;
  std::vector<std::string> stamps_;
  std::deque<std::string> strings_;
  std::vector<std::pair<std::string, core::GgufMetadataValue>> metadata_;
};

/// Throws when a native metadata reader's source files changed since it was
/// built; returns for other readers.
void ValidateNativeSources(const core::GgufReader& metadata);

/// Content identity of a native metadata reader's source files.
[[nodiscard]] std::string NativeArtifactIdentity(
    const core::GgufReader& metadata);

}  // namespace gufo::models::qwen

#endif  // GUFO_MODELS_QWEN_NATIVE_PACK_HPP_
