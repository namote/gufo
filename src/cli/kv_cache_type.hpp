#ifndef GUFO_CLI_KV_CACHE_TYPE_HPP_
#define GUFO_CLI_KV_CACHE_TYPE_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "src/cli/arg_parser.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"

namespace gufo::cli {

/// Attention KV storage selected by `--kv-cache-type`.
enum class KvCacheType : std::uint8_t {
  kFp16,
  kKvarnK4V2,
  kKvarnK4V4,
};

/// Parses the one canonical spelling of each type.
[[nodiscard]] std::optional<KvCacheType> ParseKvCacheType(
    std::string_view value) noexcept;

[[nodiscard]] std::string_view KvCacheTypeName(KvCacheType type) noexcept;

/// Registers `--kv-cache-type`; an absent option leaves `*target` empty so
/// the model default applies.
void RegisterKvCacheTypeOption(ArgParser& parser,
                               std::optional<KvCacheType>* target,
                               std::string_view group);

/// The type used when `--kv-cache-type` is absent, by GGUF architecture.
/// This is the only per-model default.
[[nodiscard]] KvCacheType DefaultKvCacheType(
    std::string_view architecture) noexcept;

/// Resolves the effective type for a GGUF architecture. Models without
/// KVarN accept only fp16 and refuse KVarN by name; there is no fallback.
[[nodiscard]] std::optional<KvCacheType> ResolveKvCacheType(
    std::string_view architecture, std::optional<KvCacheType> requested,
    std::string* error);

/// Qwen3.8-27B executor storage for a resolved type.
[[nodiscard]] hip::QwenKvCacheStorage QwenKvCacheStorageFor(
    KvCacheType type) noexcept;

/// Production Qwen3.8-27B policy with the resolved KV storage; fp16 is
/// exactly QwenExecutionPolicy::Production().
[[nodiscard]] hip::QwenExecutionPolicy QwenExecutionPolicyFor(
    KvCacheType type) noexcept;

}  // namespace gufo::cli

#endif  // GUFO_CLI_KV_CACHE_TYPE_HPP_
