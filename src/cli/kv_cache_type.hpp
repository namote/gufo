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

/// The type used when `--kv-cache-type` is absent, by GGUF architecture;
/// empty when the model's KV is not one of these types (DeepSeek's FP8).
/// This is the only per-model default.
[[nodiscard]] std::optional<KvCacheType> DefaultKvCacheType(
    std::string_view architecture) noexcept;

/// Resolves the effective type for a GGUF architecture into `*resolved`
/// (empty: the model's own KV storage). A model accepts an explicit value
/// only when it stores attention KV that way; otherwise this returns false
/// and names the model in `*error`. There is no fallback.
[[nodiscard]] bool ResolveKvCacheType(std::string_view architecture,
                                      std::optional<KvCacheType> requested,
                                      std::optional<KvCacheType>* resolved,
                                      std::string* error);

/// Qwen3.8-27B executor storage for a resolved type.
[[nodiscard]] hip::QwenKvCacheStorage QwenKvCacheStorageFor(
    KvCacheType type) noexcept;

/// Production Qwen3.8-27B policy with the resolved KV storage; fp16 and an
/// empty type are exactly QwenExecutionPolicy::Production().
[[nodiscard]] hip::QwenExecutionPolicy QwenExecutionPolicyFor(
    std::optional<KvCacheType> type) noexcept;

}  // namespace gufo::cli

#endif  // GUFO_CLI_KV_CACHE_TYPE_HPP_
