#include "src/cli/kv_cache_type.hpp"

#include <stdexcept>

namespace gufo::cli {
namespace {

// Attention KV each text runtime stores, by GGUF architecture. A model takes
// an explicit --kv-cache-type value only when it stores KV that way.
struct TextModelKv {
  std::string_view architecture;
  std::string_view name;
  std::string_view kv_storage;  // Empty: no text runtime in this build.
  bool fp16;
  bool k4v2;
  bool k4v4;
  bool k6v6;
  bool kvarn_default;  // kvarn-k4v2 when --kv-cache-type is absent
};

// Qwen3.8-27B: KVarN k4v2 by default (DefaultKvCacheType); fp16 stays the
// QwenExecutionPolicy::Production() storage that explicit fp16 selects.
// Flash-Next: __half k_cache/v_cache by default, KVarN k4v4 on request
// (qwen38_flash_next ModelOptions::kvarn,
// plan/exl3/archive/F1-KVARN-DESIGN.md). DeepSeek V4 Flash: E4M3 FP8 KV
// (ds4_rocm_fp8_kv.hip.hpp).
constexpr TextModelKv kTextModels[] = {
    {"qwen35", "Qwen3.8-27B", "FP16", true, true, true, false, true},
    {"qwen4exp", "Qwen3.8-Flash-Next", "FP16", true, false, true, true, false},
    {"deepseek4", "DeepSeek V4 Flash", "FP8", false, false, false, false,
     false},
    {"gemma4", "Gemma 4", "", false, false, false, false, false},
};

TextModelKv FindTextModel(std::string_view architecture) {
  for (const auto& model : kTextModels) {
    if (model.architecture == architecture)
      return model;
  }
  return {architecture, {}, {}, false, false, false, false, false};
}

std::string TextModelName(const TextModelKv& model) {
  return model.name.empty()
             ? "architecture '" + std::string(model.architecture) + "'"
             : std::string(model.name);
}

}  // namespace

std::optional<KvCacheType> ParseKvCacheType(std::string_view value) noexcept {
  for (const auto type : {KvCacheType::kFp16, KvCacheType::kKvarnK4V2,
                          KvCacheType::kKvarnK4V4, KvCacheType::kKvarnK6V6}) {
    if (value == KvCacheTypeName(type))
      return type;
  }
  return std::nullopt;
}

std::string_view KvCacheTypeName(KvCacheType type) noexcept {
  if (type == KvCacheType::kKvarnK6V6)
    return "kvarn-k6v6";
  return hip::QwenKvCacheStorageName(QwenKvCacheStorageFor(type));
}

void RegisterKvCacheTypeOption(ArgParser& parser,
                               std::optional<KvCacheType>* target,
                               std::string_view group) {
  parser.AddCustomOption(
      "", "--kv-cache-type", "TYPE",
      "Attention KV storage: fp16, kvarn-k4v2, kvarn-k4v4 or kvarn-k6v6, where "
      "the model "
      "stores it (default: per model; kvarn-k4v2 on Qwen3.8-27B, fp16 on "
      "Flash-Next, which also takes kvarn-k4v4 and kvarn-k6v6)",
      group,
      [target](std::string_view, std::string_view value, std::string* error) {
        const auto type = ParseKvCacheType(value);
        if (!type) {
          if (error != nullptr)
            *error =
                "--kv-cache-type must be fp16, kvarn-k4v2, kvarn-k4v4 or "
                "kvarn-k6v6";
          return false;
        }
        *target = type;
        return true;
      });
}

std::optional<KvCacheType> DefaultKvCacheType(
    std::string_view architecture) noexcept {
  // Qwen3.8-27B stores KVarN k4v2 (its K1 quality/speed and K2 serving
  // gates passed); the other FP16-KV models keep fp16.
  const auto model = FindTextModel(architecture);
  if (model.kvarn_default)
    return KvCacheType::kKvarnK4V2;
  if (model.fp16)
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
  const bool accepted = *requested == KvCacheType::kFp16        ? model.fp16
                        : *requested == KvCacheType::kKvarnK4V2 ? model.k4v2
                        : *requested == KvCacheType::kKvarnK4V4 ? model.k4v4
                                                                : model.k6v6;
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
      std::string accepts = "fp16";
      if (model.k4v2)
        accepts +=
            model.k4v4 ? ", kvarn-k4v2 and kvarn-k4v4" : " and kvarn-k4v2";
      else if (model.k4v4)
        accepts += " and kvarn-k4v4";
      if (model.k6v6)
        accepts += " and kvarn-k6v6";
      *error = "--kv-cache-type " + std::string(KvCacheTypeName(*requested)) +
               " is not supported by " + TextModelName(model) + "; its KV " +
               "cache is " + std::string(model.kv_storage) +
               ", so it accepts " +
               (model.k4v2 || model.k4v4 ? accepts : "only fp16");
    }
  }
  return false;
}

hip::QwenKvCacheStorage QwenKvCacheStorageFor(KvCacheType type) {
  switch (type) {
    case KvCacheType::kFp16:
      return hip::QwenKvCacheStorage::kFp16;
    case KvCacheType::kKvarnK4V2:
      return hip::QwenKvCacheStorage::kKvarnK4V2;
    case KvCacheType::kKvarnK4V4:
      return hip::QwenKvCacheStorage::kKvarnK4V4;
    case KvCacheType::kKvarnK6V6:
      throw std::invalid_argument("K6V6 requires the Flash-Next backend");
  }
  return hip::QwenKvCacheStorage::kFp16;
}

std::optional<hip::QwenRecurrentStateStorage> ParseRecurrentState(
    std::string_view value) noexcept {
  using Storage = hip::QwenRecurrentStateStorage;
  for (const auto storage : {Storage::kFp32, Storage::kFp16, Storage::kBf16}) {
    if (value == hip::QwenRecurrentStateStorageName(storage))
      return storage;
  }
  return std::nullopt;
}

void RegisterRecurrentStateOption(
    ArgParser& parser, std::optional<hip::QwenRecurrentStateStorage>* target,
    std::string_view group) {
  parser.AddCustomOption(
      "", "--recurrent-state", "TYPE",
      "Qwen3.8-27B DeltaNet recurrent state storage: fp32, fp16 (FP32 "
      "arithmetic, rounded once per pass on store) or the rejected bf16 route "
      "(default: fp32)",
      group,
      [target](std::string_view, std::string_view value, std::string* error) {
        const auto storage = ParseRecurrentState(value);
        if (!storage) {
          if (error != nullptr)
            *error = "--recurrent-state must be fp32, fp16 or bf16";
          return false;
        }
        *target = storage;
        return true;
      });
}

bool ResolveRecurrentState(
    std::string_view architecture,
    std::optional<hip::QwenRecurrentStateStorage> requested,
    hip::QwenRecurrentStateStorage* resolved, std::string* error) {
  using Storage = hip::QwenRecurrentStateStorage;
  *resolved = Storage::kFp32;
  if (!requested)
    return true;
  const auto model = FindTextModel(architecture);
  const bool accepted =
      architecture == "qwen35" ||
      (architecture == "qwen4exp" && *requested == Storage::kFp32);
  if (accepted) {
    *resolved = *requested;
    return true;
  }
  if (error != nullptr) {
    *error =
        architecture == "qwen4exp"
            ? "--recurrent-state " +
                  std::string(hip::QwenRecurrentStateStorageName(*requested)) +
                  " is not supported by Qwen3.8-Flash-Next; it accepts "
                  "only fp32"
            : TextModelName(model) +
                  " does not take --recurrent-state (it has no "
                  "recurrent state)";
  }
  return false;
}

hip::QwenExecutionPolicy QwenExecutionPolicyFor(
    std::optional<KvCacheType> type, hip::QwenRecurrentStateStorage recurrent) {
  auto policy = hip::QwenExecutionPolicy::Production();
  if (type)
    policy.kv_cache_storage = QwenKvCacheStorageFor(*type);
  policy.recurrent_state_storage = recurrent;
  return policy;
}

}  // namespace gufo::cli
