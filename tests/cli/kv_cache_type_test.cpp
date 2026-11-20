// `--kv-cache-type` parsing, per-model defaults and refusals.
#include "src/cli/kv_cache_type.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

using gufo::cli::KvCacheType;

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

void TestParse() {
  for (const auto type : {KvCacheType::kFp16, KvCacheType::kKvarnK4V2,
                          KvCacheType::kKvarnK4V4, KvCacheType::kKvarnK6V6}) {
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
             error.find("fp16, kvarn-k4v2, kvarn-k4v4 or kvarn-k6v6") !=
                 std::string::npos,
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
  // No flag works for every model: Qwen3.8-27B resolves kvarn-k4v2,
  // Flash-Next fp16, others their own storage.
  for (const auto [architecture, expected] :
       {std::pair{std::string_view{"qwen35"}, KvCacheType::kKvarnK4V2},
        std::pair{std::string_view{"qwen4exp"}, KvCacheType::kFp16}}) {
    Expect(gufo::cli::DefaultKvCacheType(architecture) == expected,
           "Qwen3.8-27B defaults to kvarn-k4v2, Flash-Next to fp16");
    bool accepted = false;
    std::string error;
    Expect(Resolve(architecture, std::nullopt, &accepted, &error) == expected &&
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
  {
    // Flash-Next stores KVarN k4v4 only (plan/exl3/archive/F1-KVARN-DESIGN.md).
    bool accepted = true;
    std::string error;
    (void)Resolve("qwen4exp", KvCacheType::kKvarnK4V2, &accepted, &error);
    Expect(!accepted && error ==
                            "--kv-cache-type kvarn-k4v2 is not supported by "
                            "Qwen3.8-Flash-Next; its KV cache is FP16, so it "
                            "accepts fp16 and kvarn-k4v4 and kvarn-k6v6",
           "Flash-Next refuses kvarn-k4v2 by name");
    accepted = false;
    error.clear();
    Expect(Resolve("qwen4exp", KvCacheType::kKvarnK4V4, &accepted, &error) ==
                   KvCacheType::kKvarnK4V4 &&
               accepted,
           "Flash-Next accepts kvarn-k4v4");
  }
  for (const auto type : {KvCacheType::kKvarnK4V2, KvCacheType::kKvarnK4V4}) {
    bool accepted = false;
    std::string error;
    Expect(Resolve("qwen35", type, &accepted, &error) == type && accepted,
           "Qwen3.8-27B accepts KVarN at startup");
    // The executor and the server's prefix cache both hold it.
    const auto policy = gufo::cli::QwenExecutionPolicyFor(type);
    Expect(gufo::hip::QwenExecutionPolicyUnsupportedReason(policy).empty() &&
               gufo::hip::QwenContinuationUnsupportedReason(policy).empty(),
           "KVarN runs and continues");
  }
}

void TestPolicy() {
  bool accepted = false;
  std::string error;
  Expect(Resolve("qwen4exp", KvCacheType::kKvarnK6V6, &accepted, &error) ==
                 KvCacheType::kKvarnK6V6 &&
             accepted,
         "Flash-Next K6V6 accepted");
  (void)Resolve("qwen35", KvCacheType::kKvarnK6V6, &accepted, &error);
  Expect(!accepted, "Other Qwen backend rejects K6V6");
  bool threw = false;
  try {
    (void)gufo::cli::QwenExecutionPolicyFor(KvCacheType::kKvarnK6V6);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Expect(threw, "K6V6 cannot silently become FP16 in another backend");
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

// `--recurrent-state`: one spelling per storage, fp32 when absent, and only
// the models with recurrent state take a value.
void TestRecurrentState() {
  using Storage = gufo::hip::QwenRecurrentStateStorage;
  for (const auto storage : {Storage::kFp32, Storage::kFp16, Storage::kBf16})
    Expect(gufo::cli::ParseRecurrentState(
               gufo::hip::QwenRecurrentStateStorageName(storage)) == storage,
           "recurrent state names round-trip");
  for (const std::string_view alias : {"FP16", "half", "f16", "float", ""})
    Expect(!gufo::cli::ParseRecurrentState(alias),
           "recurrent state aliases are rejected");

  std::optional<Storage> option;
  gufo::cli::ArgParser parser("test", "test");
  gufo::cli::RegisterRecurrentStateOption(parser, &option, "Model");
  const std::array<const char*, 2> args = {"--recurrent-state", "fp16"};
  std::string error;
  Expect(parser.Parse(args, &error) && option == Storage::kFp16,
         "--recurrent-state stores the parsed storage");
  std::optional<Storage> bad;
  gufo::cli::ArgParser invalid("test", "test");
  gufo::cli::RegisterRecurrentStateOption(invalid, &bad, "Model");
  const std::array<const char*, 2> bad_args = {"--recurrent-state", "fp8"};
  Expect(!invalid.Parse(bad_args, &error) && !bad &&
             error.find("fp32, fp16 or bf16") != std::string::npos,
         "invalid recurrent state names the accepted set");

  const auto resolve = [](std::string_view architecture,
                          std::optional<Storage> requested, Storage* resolved,
                          std::string* message) {
    *resolved = Storage::kBf16;
    message->clear();
    return gufo::cli::ResolveRecurrentState(architecture, requested, resolved,
                                            message);
  };
  Storage resolved{};
  for (const std::string_view architecture :
       {"qwen35", "qwen4exp", "deepseek4", "gemma4"})
    Expect(resolve(architecture, std::nullopt, &resolved, &error) &&
               resolved == Storage::kFp32 && error.empty(),
           "an absent option stores fp32");
  for (const auto storage : {Storage::kFp32, Storage::kFp16, Storage::kBf16})
    Expect(resolve("qwen35", storage, &resolved, &error) && resolved == storage,
           "Qwen3.8-27B takes every recurrent storage");
  Expect(resolve("qwen4exp", Storage::kFp32, &resolved, &error) &&
             resolved == Storage::kFp32,
         "Flash-Next takes fp32");
  Expect(!resolve("qwen4exp", Storage::kBf16, &resolved, &error) &&
             error.find("Qwen3.8-Flash-Next") != std::string::npos,
         "Flash-Next refuses bf16 by name");
  Expect(!resolve("deepseek4", Storage::kFp32, &resolved, &error) &&
             error ==
                 "DeepSeek V4 Flash does not take --recurrent-state (it has "
                 "no recurrent state)",
         "models without recurrent state refuse every value");

  const auto policy =
      gufo::cli::QwenExecutionPolicyFor(KvCacheType::kFp16, Storage::kFp16);
  Expect(policy.UsesFp16RecurrentState() && policy.UsesFp16AttentionKv(),
         "the policy carries the recurrent storage");
  Expect(gufo::cli::QwenExecutionPolicyFor(KvCacheType::kFp16).Fingerprint() ==
             gufo::hip::QwenExecutionPolicy::Production().Fingerprint(),
         "fp32 state stays the production policy");
}

int main() {
  TestParse();
  TestOption();
  TestDefaults();
  TestRefusals();
  TestPolicy();
  TestRecurrentState();
  std::cout << "KV cache type checks passed.\n";
}
