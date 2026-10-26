#include "src/cli/kv_cache_type.hpp"

namespace gufo::cli {
namespace {

// Attention KV each text runtime stores, by GGUF architecture. A model takes
// an explicit --kv-cache-type value only when it stores KV that way.
struct TextModelKv {
  std::string_view architecture;
  std::string_view name;
  std::string_view kv_storage;  // Empty: no text runtime in this build.
  bool fp16;
  bool kvarn;
};

// Qwen3.8-27B: FP16 rows by default (QwenExecutionPolicy::kv_cache_storage).
// Flash-Next: __half k_cache/v_cache (qwen38_flash_next executor.cpp).
// DeepSeek V4 Flash: E4M3 FP8 KV (ds4_rocm_fp8_kv.hip.hpp).
constexpr TextModelKv kTextModels[] = {
    {"qwen35", "Qwen3.8-27B", "FP16", true, true},
    {"qwen4exp", "Qwen3.8-Flash-Next", "FP16", true, false},
    {"deepseek4", "DeepSeek V4 Flash", "FP8", false, false},
    {"gemma4", "Gemma 4", "", false, false},
};

TextModelKv FindTextModel(std::string_view architecture) {
  for (const auto& model : kTextModels) {
    if (model.architecture == architecture)
      return model;
  }
  return {architecture, {}, {}, false, false};
}

std::string TextModelName(const TextModelKv& model) {
  return model.name.empty()
             ? "architecture '" + std::string(model.architecture) + "'"
             : std::string(model.name);
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
      "Attention KV storage: fp16, kvarn-k4v2 or kvarn-k4v4, where the model "
      "stores it (default: per model; fp16 on Qwen3.8-27B and Flash-Next)",
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

std::optional<KvCacheType> DefaultKvCacheType(
    std::string_view architecture) noexcept {
  // Qwen3.8-27B moves to kvarn-k4v2 once its quality and speed gate passes.
  if (FindTextModel(architecture).fp16)
    return KvCacheType::kFp16;
  return std::nullopt;
}

bool ResolveKvCacheType(std::string_view architecture,
                        std::optional<KvCacheType> requested,
                        std::optional<KvCacheType>* resolved,
                        std::string* error) {
  const auto model = FindTextModel(architecture);
  if (!requested) {
    *resolved = DefaultKvCacheType(architecture);
    return true;
  }
  const bool accepted =
      *requested == KvCacheType::kFp16 ? model.fp16 : model.kvarn;
  if (accepted) {
    *resolved = requested;
    return true;
  }
  if (error != nullptr) {
    if (!model.fp16) {
      *error = TextModelName(model) + " does not take --kv-cache-type";
      if (!model.kv_storage.empty())
        *error += " (its KV cache is " + std::string(model.kv_storage) + ")";
    } else {
      *error = "--kv-cache-type " + std::string(KvCacheTypeName(*requested)) +
               " is not supported by " + TextModelName(model) + "; its KV " +
               "cache is " + std::string(model.kv_storage) +
               ", so it accepts only fp16";
    }
  }
  return false;
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

hip::QwenExecutionPolicy QwenExecutionPolicyFor(
    std::optional<KvCacheType> type) noexcept {
  auto policy = hip::QwenExecutionPolicy::Production();
  if (type)
    policy.kv_cache_storage = QwenKvCacheStorageFor(*type);
  return policy;
}

}  // namespace gufo::cli
