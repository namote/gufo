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

void TestDefaults() {
  for (const std::string_view architecture :
       {"qwen35", "qwen4exp", "deepseek4", "unknown"}) {
    Expect(gufo::cli::DefaultKvCacheType(architecture) == KvCacheType::kFp16,
           "every model defaults to fp16 until a KVarN gate passes");
    std::string error;
    const auto resolved =
        gufo::cli::ResolveKvCacheType(architecture, std::nullopt, &error);
    Expect(resolved == KvCacheType::kFp16 && error.empty(),
           "absent type resolves to the default");
    Expect(gufo::cli::ResolveKvCacheType(architecture, KvCacheType::kFp16,
                                         &error) == KvCacheType::kFp16,
           "explicit fp16 is accepted by every model");
  }
}

void TestRefusals() {
  struct Case {
    std::string_view architecture;
    std::string_view name;
  };
  for (const auto& [architecture, name] :
       {Case{"qwen4exp", "Qwen3.8-Flash-Next"},
        Case{"deepseek4", "DeepSeek V4 Flash"}, Case{"gemma4", "Gemma 4"},
        Case{"llama", "architecture 'llama'"}}) {
    for (const auto type : {KvCacheType::kKvarnK4V2, KvCacheType::kKvarnK4V4}) {
      std::string error;
      Expect(!gufo::cli::ResolveKvCacheType(architecture, type, &error),
             "models without KVarN refuse it");
      Expect(
          error.find(name) != std::string::npos &&
              error.find(gufo::cli::KvCacheTypeName(type)) != std::string::npos,
          "the refusal names the model and the type");
    }
  }
  for (const auto type : {KvCacheType::kKvarnK4V2, KvCacheType::kKvarnK4V4}) {
    std::string error;
    Expect(gufo::cli::ResolveKvCacheType("qwen35", type, &error) == type,
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
