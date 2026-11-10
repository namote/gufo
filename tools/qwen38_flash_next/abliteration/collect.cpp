// Standalone native activation/evaluation collector. No HTTP service and no
// weight edits; completed prompt receipts are reused after interruption.
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/device_code_identity.hpp"
#include "src/core/json.hpp"
#include "src/core/platform/os_info.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace fs = std::filesystem;
namespace json = gufo::json;
namespace q = gufo::models::qwen38_flash_next;

namespace {
void Require(bool ok, const std::string& error) {
  if (!ok)
    throw std::runtime_error(error);
}
std::string Read(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  Require(bool(input), "cannot open " + path.string());
  return {std::istreambuf_iterator<char>(input), {}};
}
void Save(const fs::path& path, const json::Value& value) {
  const fs::path temporary = path.string() + ".tmp";
  {
    std::ofstream out(temporary);
    out << value.dump() << '\n';
    Require(bool(out), "write failed");
  }
  fs::rename(temporary, path);
}
void Floats(const fs::path& path, std::span<const float> values) {
  const fs::path temporary = path.string() + ".tmp";
  {
    std::ofstream out(temporary, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()),
              values.size_bytes());
    Require(bool(out), "activation write failed");
  }
  fs::rename(temporary, path);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Require(argc >= 5,
            "usage: collector MODEL DATA.jsonl OUTPUT collect|evaluate "
            "[--profile FILE] [--max-new-tokens N]");
    const fs::path output = argv[3];
    const bool collect = std::string_view(argv[4]) == "collect";
    Require(collect || std::string_view(argv[4]) == "evaluate", "invalid mode");
    std::string profile;
    unsigned max_new_tokens = 1024;
    for (int i = 5; i < argc; i += 2) {
      Require(i + 1 < argc, "missing option value");
      const std::string_view flag = argv[i];
      if (flag == "--profile")
        profile = argv[i + 1];
      else if (flag == "--max-new-tokens") {
        std::size_t consumed = 0;
        const auto value = std::stoul(argv[i + 1], &consumed);
        Require(consumed == std::string_view(argv[i + 1]).size() && value > 0 &&
                    value <= 16384,
                "max-new-tokens must be in [1, 16384]");
        max_new_tokens = static_cast<unsigned>(value);
      } else
        throw std::runtime_error("unknown collector option");
    }
    Require(!collect || profile.empty(),
            "collection must use the unedited model");
    const bool enabled = !profile.empty();
    q::ModelOptions options;
    options.max_context = 2048 + (collect ? 0 : max_new_tokens);
    options.abliteration_profile = profile;
    std::string error;
    auto model = q::Model::Load(argv[1], options, &error);
    Require(bool(model), error);
    auto session = model->CreateSession(
        gufo::core::SessionMode::kAutoregressive, options.max_context, &error);
    Require(bool(session), error);
    // The profile defaults; a v2 profile's per-request settings are a
    // server feature.
    session->ConfigureAbliteration(
        enabled ? q::ResolveAbliterationSettings(*model->Abliteration())
                : nullptr);
    const auto& config = model->config();
    std::ifstream input(argv[2]);
    Require(bool(input), "cannot open dataset");
    auto records = json::Value::array();
    std::string split;
    for (std::string line; std::getline(input, line);) {
      if (line.empty())
        continue;
      auto record = json::parse(line);
      if (split.empty())
        split = record.member_str("split");
      Require(record.member_str("split") == split, "mixed dataset splits");
      const auto id = record.member_str("id");
      Require(
          !id.empty() && id.find_first_not_of("0123456789abcdef-plainthk") ==
                             std::string::npos,
          "invalid record ID");
      Require(record.member_str("group") == "refusal" ||
                  record.member_str("group") == "ordinary",
              "invalid record class");
      Require(record.find("thinking") && record.find("thinking")->is_bool(),
              "missing thinking mode");
      records.push_back(std::move(record));
    }
    auto manifest = json::Value::object();
    manifest["schema"] = "gufo-abliteration-capture-v1";
    manifest["model_identity"] = model->ArtifactIdentity();
    manifest["device_code"] = gufo::core::ExecutableDeviceCodeSha256();
    const auto executable = gufo::platform::RunningExecutablePath();
    Require(executable.has_value(), "cannot identify collector executable");
    manifest["executable_sha256"] = gufo::crypto::Sha256FileHex(*executable);
    manifest["context"] = static_cast<unsigned long long>(options.max_context);
    manifest["max_new_tokens"] =
        static_cast<unsigned long long>(collect ? 0 : max_new_tokens);
    manifest["template"] = "gufo-qwen-native-medium-v1";
    manifest["readout"] = collect
                              ? "last-prompt-token-post-layer-hc-mean-fp32-v1"
                              : "first-next-token-logits-fp32-v1";
    manifest["data_sha256"] = gufo::crypto::Sha256FileHex(argv[2]);
    manifest["hidden"] = static_cast<unsigned long long>(config.hidden_size);
    manifest["layers"] = static_cast<unsigned long long>(config.num_layers);
    manifest["split"] = split;
    manifest["mode"] = argv[4];
    manifest["profile"] = enabled ? model->Abliteration()->identity : "off";
    manifest["records"] = records;
    fs::create_directories(output);
    if (fs::exists(output / "manifest.json"))
      Require(
          json::parse(Read(output / "manifest.json")).dump() == manifest.dump(),
          "existing collection identity differs; use a new output directory");
    else
      Save(output / "manifest.json", manifest);
    std::vector<float> activations(std::size_t{config.num_layers} *
                                   config.hidden_size);
    std::vector<bool> captured(config.num_layers);
    if (collect)
      session->SetActivationObserver([&](std::uint32_t layer,
                                         std::span<const float> row) {
        Require(row.size() == config.HcDim(), "collector HC layout mismatch");
        for (std::size_t h = 0; h < config.hidden_size; ++h) {
          double sum = 0;
          for (std::size_t c = 0; c < config.hc_count; ++c)
            sum += row[c * config.hidden_size + h];
          activations[std::size_t{layer} * config.hidden_size + h] =
              float(sum / config.hc_count);
        }
        captured[layer] = true;
      });
    std::size_t done = 0;
    for (const auto& record : records.items()) {
      const auto id = record.member_str("id");
      const auto receipt = output / (id + ".json");
      const auto data = output / (id + ".f32");
      if (fs::exists(receipt)) {
        const auto saved = json::parse(Read(receipt));
        Require(fs::exists(data) && saved.member_str("activation_sha256") ==
                                        gufo::crypto::Sha256FileHex(data),
                "saved prompt data integrity failure");
        ++done;
        continue;
      }
      session->Reset();
      std::fill(captured.begin(), captured.end(), false);
      gufo::tokenization::ChatMessage message;
      message.role = gufo::tokenization::ChatRole::kUser;
      message.content = record.member_str("text");
      gufo::tokenization::ChatTemplateOptions chat;
      chat.enable_thinking = record.find("thinking")->as_bool();
      chat.reasoning_effort = gufo::tokenization::QwenReasoningEffort::kMedium;
      auto tokens = gufo::tokenization::QwenChatTemplate::RenderAndTokenize(
          model->tokenizer(), std::span(&message, 1), chat, &error);
      Require(
          bool(tokens) && !tokens->empty() && tokens->size() <= 2048,
          "prompt invalid/too long; no silent truncation: " + id + " " + error);
      const std::vector<std::int32_t> prompt(tokens->begin(), tokens->end());
      Require(session->Sync(prompt, &error), error);
      auto result = json::Value::object();
      result["id"] = id;
      result["prompt_tokens"] = tokens->size();
      if (collect) {
        Require(std::ranges::all_of(captured, [](bool x) { return x; }),
                "missing target layer capture");
        Floats(data, activations);
      } else {
        Floats(data, session->Logits());
        std::vector<std::int32_t> generated;
        for (unsigned i = 0; i < max_new_tokens; ++i) {
          const auto logits = session->Logits();
          const auto token = static_cast<std::int32_t>(
              std::max_element(logits.begin(), logits.end()) - logits.begin());
          if (model->IsStopToken(token))
            break;
          generated.push_back(token);
          Require(session->Evaluate(token, &error), error);
        }
        const auto text = model->Decode(generated);
        result["text"] = text;
        const bool reasoning_complete =
            !chat.enable_thinking || text.find("</think>") != std::string::npos;
        result["reasoning_complete"] = reasoning_complete;
        result["generation_complete"] =
            reasoning_complete && generated.size() < max_new_tokens;
        result["generated_tokens"] = generated.size();
        result["truncated"] = generated.size() == max_new_tokens;
      }
      result["activation_sha256"] = gufo::crypto::Sha256FileHex(data);
      Save(receipt, result);
      std::cout << "completed " << ++done << "/" << records.size() << " " << id
                << std::endl;
    }
    std::cout << "complete: " << output << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
