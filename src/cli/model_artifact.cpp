#include "src/cli/model_artifact.hpp"

#include <fstream>
#include <sstream>
#include <system_error>

#include "src/core/json.hpp"
#include "src/core/quant/exl3/layout.hpp"
#include "src/models/qwen/native_metadata.hpp"
#include "src/models/qwen38_flash_next/native_metadata.hpp"

namespace gufo::cli {

std::unique_ptr<core::GgufReader> OpenModelArtifact(
    const std::filesystem::path& path, std::string* error) {
  std::error_code ec;
  if (!std::filesystem::is_directory(path, ec)) {
    auto reader = core::GgufReader::OpenFile(path, error);
    if (reader && reader->FindMetadata(quant::exl3::kCheckpointKey)) {
      if (error)
        *error =
            "EXL3 metadata sidecar files are not supported; pass the EXL3 "
            "pack directory";
      return nullptr;
    }
    return reader;
  }
  std::string model_type;
  try {
    std::ifstream in(path / "config.json", std::ios::binary);
    if (!in)
      throw std::runtime_error("cannot read " +
                               (path / "config.json").string());
    std::ostringstream text;
    text << in.rdbuf();
    const auto config = json::parse(text.str());
    if (const auto* text_config = config.find("text_config"))
      model_type = text_config->member_str("model_type");
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return nullptr;
  }
  if (model_type == "qwen3_5_text")
    return models::qwen::OpenNativeMetadata(path, error);
  if (model_type == "qwen4_exp_text")
    return models::qwen38_flash_next::OpenArtifactMetadata(path, error);
  if (error)
    *error = "unsupported EXL3 pack directory (text_config.model_type \"" +
             model_type + "\")";
  return nullptr;
}

}  // namespace gufo::cli
