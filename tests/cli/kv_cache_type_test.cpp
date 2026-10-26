// `--kv-cache-type` parsing, per-model defaults and refusals.
#include "src/cli/kv_cache_type.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

namespace {

using gufo::cli::KvCacheType;

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

void TestParse() {
  for (const auto type :
       {KvCacheType::kFp16, KvCacheType::kKvarnK4V2, KvCacheType::kKvarnK4V4}) {
    Expect(
        gufo::cli::ParseKvCacheType(gufo::cli::KvCacheTypeName(type)) == type,
        "canonical names round-trip");
  }
  Expect(
      gufo::cli::KvCacheTypeName(KvCacheType::kFp16) == "fp16" &&
          gufo::cli::KvCacheTypeName(KvCacheType::kKvarnK4V2) == "kvarn-k4v2" &&
          gufo::cli::KvCacheTypeName(KvCacheType::kKvarnK4V4) == "kvarn-k4v4",
      "canonical spellings");
  // One spelling per behavior: no case variants, aliases or reference fp32.
  for (const std::string_view alias :
       {"FP16", "f16", "half", "fp32", "kvarn", "k4v2", "kvarn_k4v2",
        "kvarn-k4v2 ", "q8_0", ""}) {
    Expect(!gufo::cli::ParseKvCacheType(alias).has_value(),
           "aliases are rejected");
  }
}

void TestOption() {
  std::optional<KvCacheType> type;
  gufo::cli::ArgParser parser("test", "test");
  gufo::cli::RegisterKvCacheTypeOption(parser, &type, "Model");
  const std::array<const char*, 2> args = {"--kv-cache-type", "kvarn-k4v4"};
  std::string error;
  Expect(parser.Parse(args, &error) && type == KvCacheType::kKvarnK4V4,
         "option stores the parsed type");

  std::optional<KvCacheType> absent;
  gufo::cli::ArgParser empty("test", "test");
  gufo::cli::RegisterKvCacheTypeOption(empty, &absent, "Model");
  Expect(empty.Parse(std::span<const char* const>{}, &error) && !absent,
         "an absent option leaves the model default");

  std::optional<KvCacheType> bad;
  gufo::cli::ArgParser invalid("test", "test");
  gufo::cli::RegisterKvCacheTypeOption(invalid, &bad, "Model");
  const std::array<const char*, 2> bad_args = {"--kv-cache-type", "q4"};
  error.clear();
  Expect(!invalid.Parse(bad_args, &error) && !bad &&
             error.find("fp16, kvarn-k4v2 or kvarn-k4v4") != std::string::npos,
         "invalid values name the accepted set");
}

std::optional<KvCacheType> Resolve(std::string_view architecture,
                                   std::optional<KvCacheType> requested,
                                   bool* accepted, std::string* error) {
  std::optional<KvCacheType> resolved{KvCacheType::kKvarnK4V4};
  *accepted =
      gufo::cli::ResolveKvCacheType(architecture, requested, &resolved, error);
  return resolved;
}

void TestDefaults() {
  // No flag works for every model: FP16 models resolve fp16, others their
  // own storage.
  for (const std::string_view architecture : {"qwen35", "qwen4exp"}) {
    Expect(gufo::cli::DefaultKvCacheType(architecture) == KvCacheType::kFp16,
           "FP16-KV models default to fp16 until a KVarN gate passes");
    bool accepted = false;
    std::string error;
    Expect(Resolve(architecture, std::nullopt, &accepted, &error) ==
                   KvCacheType::kFp16 &&
               accepted && error.empty(),
           "absent type resolves to the default");
    Expect(Resolve(architecture, KvCacheType::kFp16, &accepted, &error) ==
                   KvCacheType::kFp16 &&
               accepted,
           "explicit fp16 is accepted where KV is FP16");
  }
  for (const std::string_view architecture : {"deepseek4", "gemma4", "llama"}) {
    Expect(!gufo::cli::DefaultKvCacheType(architecture),
           "models without FP16 KV have no type default");
    bool accepted = false;
    std::string error;
    Expect(!Resolve(architecture, std::nullopt, &accepted, &error) &&
               accepted && error.empty(),
           "absent type keeps the model's own KV");
  }
}

void TestRefusals() {
  struct Case {
    std::string_view architecture;
    std::string_view message;
  };
  // Models whose KV is not FP16 refuse every explicit value.
  for (const auto& [architecture, message] :
       {Case{"deepseek4",
             "DeepSeek V4 Flash does not take --kv-cache-type (its KV cache "
             "is FP8)"},
        Case{"gemma4", "Gemma 4 does not take --kv-cache-type"},
        Case{"llama", "architecture 'llama' does not take --kv-cache-type"}}) {
    for (const auto type : {KvCacheType::kFp16, KvCacheType::kKvarnK4V2,
                            KvCacheType::kKvarnK4V4}) {
      bool accepted = true;
      std::string error;
      (void)Resolve(architecture, type, &accepted, &error);
      Expect(!accepted && error == message,
             "the refusal names the model and its KV storage");
    }
  }
  for (const auto type : {KvCacheType::kKvarnK4V2, KvCacheType::kKvarnK4V4}) {
    bool accepted = true;
    std::string error;
    (void)Resolve("qwen4exp", type, &accepted, &error);
    Expect(!accepted &&
               error == "--kv-cache-type " +
                            std::string(gufo::cli::KvCacheTypeName(type)) +
                            " is not supported by Qwen3.8-Flash-Next; its KV "
                            "cache is FP16, so it accepts only fp16",
           "Flash-Next refuses KVarN by name");
  }
  for (const auto type : {KvCacheType::kKvarnK4V2, KvCacheType::kKvarnK4V4}) {
    bool accepted = false;
    std::string error;
    Expect(Resolve("qwen35", type, &accepted, &error) == type && accepted,
           "Qwen3.8-27B accepts KVarN at startup");
    // The executor refuses it until the kernels exist.
    const auto policy = gufo::cli::QwenExecutionPolicyFor(type);
    const auto reason = gufo::hip::QwenExecutionPolicyUnsupportedReason(policy);
    Expect(reason.find("not implemented yet") != std::string_view::npos &&
               reason.find(gufo::cli::KvCacheTypeName(type)) !=
                   std::string_view::npos,
           "the executor refuses KVarN by name");
  }
}

void TestPolicy() {
  const auto production = gufo::hip::QwenExecutionPolicy::Production();
  const auto fp16 = gufo::cli::QwenExecutionPolicyFor(KvCacheType::kFp16);
  Expect(
      fp16.Fingerprint() == production.Fingerprint() &&
          fp16.kv_cache_storage == production.kv_cache_storage &&
          fp16.recurrent_state_storage == production.recurrent_state_storage &&
          fp16.fuse_qk_norm_rope_kv == production.fuse_qk_norm_rope_kv,
      "fp16 is exactly the production policy");
  Expect(gufo::cli::QwenExecutionPolicyFor(std::nullopt).Fingerprint() ==
             production.Fingerprint(),
         "no type is exactly the production policy");
  const auto k4v2 = gufo::cli::QwenExecutionPolicyFor(KvCacheType::kKvarnK4V2);
  const auto k4v4 = gufo::cli::QwenExecutionPolicyFor(KvCacheType::kKvarnK4V4);
  Expect(k4v2.kv_cache_storage == gufo::hip::QwenKvCacheStorage::kKvarnK4V2 &&
             k4v4.kv_cache_storage == gufo::hip::QwenKvCacheStorage::kKvarnK4V4,
         "KVarN types map to their storage");
  Expect(k4v2.Fingerprint() != fp16.Fingerprint() &&
             k4v4.Fingerprint() != fp16.Fingerprint() &&
             k4v2.Fingerprint() != k4v4.Fingerprint(),
         "each KV type has its own fingerprint");
}

}  // namespace

int main() {
  TestParse();
  TestOption();
  TestDefaults();
  TestRefusals();
  TestPolicy();
  std::cout << "KV cache type checks passed.\n";
}
