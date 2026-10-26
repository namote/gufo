#include "src/cli/kv_cache_type.hpp"

namespace gufo::cli {
namespace {

// Text model families by GGUF architecture, for user-facing refusals.
std::string TextModelName(std::string_view architecture) {
  if (architecture == "qwen35")
    return "Qwen3.8-27B";
  if (architecture == "qwen4exp")
    return "Qwen3.8-Flash-Next";
  if (architecture == "deepseek4")
    return "DeepSeek V4 Flash";
  if (architecture == "gemma4")
    return "Gemma 4";
  return "architecture '" + std::string(architecture) + "'";
}

// Only Qwen3.8-27B has a KVarN representation.
bool SupportsKvarn(std::string_view architecture) noexcept {
  return architecture == "qwen35";
}

}  // namespace

std::optional<KvCacheType> ParseKvCacheType(std::string_view value) noexcept {
  for (const auto type :
       {KvCacheType::kFp16, KvCacheType::kKvarnK4V2, KvCacheType::kKvarnK4V4}) {
    if (value == KvCacheTypeName(type))
      return type;
  }
  return std::nullopt;
}

std::string_view KvCacheTypeName(KvCacheType type) noexcept {
  return hip::QwenKvCacheStorageName(QwenKvCacheStorageFor(type));
}

void RegisterKvCacheTypeOption(ArgParser& parser,
                               std::optional<KvCacheType>* target,
                               std::string_view group) {
  parser.AddCustomOption(
      "", "--kv-cache-type", "TYPE",
      "Attention KV storage: fp16, kvarn-k4v2 or kvarn-k4v4 (default: per "
      "model, currently fp16)",
      group,
      [target](std::string_view, std::string_view value, std::string* error) {
        const auto type = ParseKvCacheType(value);
        if (!type) {
          if (error != nullptr)
            *error = "--kv-cache-type must be fp16, kvarn-k4v2 or kvarn-k4v4";
          return false;
        }
        *target = type;
        return true;
      });
}

KvCacheType DefaultKvCacheType(std::string_view architecture) noexcept {
  // Qwen3.8-27B moves to kvarn-k4v2 once its quality and speed gate passes.
  (void)architecture;
  return KvCacheType::kFp16;
}

std::optional<KvCacheType> ResolveKvCacheType(
    std::string_view architecture, std::optional<KvCacheType> requested,
    std::string* error) {
  const KvCacheType type = requested.value_or(DefaultKvCacheType(architecture));
  if (type != KvCacheType::kFp16 && !SupportsKvarn(architecture)) {
    if (error != nullptr)
      *error = "--kv-cache-type " + std::string(KvCacheTypeName(type)) +
               " is not supported by " + TextModelName(architecture) +
               "; it accepts only fp16";
    return std::nullopt;
  }
  return type;
}

hip::QwenKvCacheStorage QwenKvCacheStorageFor(KvCacheType type) noexcept {
  switch (type) {
    case KvCacheType::kFp16:
      return hip::QwenKvCacheStorage::kFp16;
    case KvCacheType::kKvarnK4V2:
      return hip::QwenKvCacheStorage::kKvarnK4V2;
    case KvCacheType::kKvarnK4V4:
      return hip::QwenKvCacheStorage::kKvarnK4V4;
  }
  return hip::QwenKvCacheStorage::kFp16;
}

hip::QwenExecutionPolicy QwenExecutionPolicyFor(KvCacheType type) noexcept {
  auto policy = hip::QwenExecutionPolicy::Production();
  policy.kv_cache_storage = QwenKvCacheStorageFor(type);
  return policy;
}

}  // namespace gufo::cli
