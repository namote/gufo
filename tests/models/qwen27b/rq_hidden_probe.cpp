// RQ R2 diagnosis probe: per-layer residual hidden states of a Qwen27B GGUF
// at the last rows of a corpus prefix, to compare a rotated file with its
// unrotated source layer by layer (plan/rotated-quant/R2-RESULTS.md).
//
//   qwen27b_rq_hidden_probe <model.gguf> <windows.bin> <window offset>
//       <position> <rows> <out prefix>
//
// Prefills tokens [offset, offset + position - rows] without capture, then
// the last `rows` tokens up to and including `position` with every layer's
// residual output captured. Writes <out>.hidden.f32 ([rows][layers][hidden]
// floats), <out>.logits.f32 (the last row's logits) and <out>.txt (shape).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace {

void Require(bool ok, const std::string& error) {
  if (!ok) {
    throw std::runtime_error(error);
  }
}

void WriteFloats(const std::string& path, std::span<const float> values) {
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size_bytes()));
  Require(file.good(), "cannot write " + path);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Require(argc == 7,
            "usage: qwen27b_rq_hidden_probe <model.gguf> <windows.bin> "
            "<window offset> <position> <rows> <out prefix>");
    const std::string model_path = argv[1];
    const std::size_t offset = std::stoull(argv[3]);
    const std::size_t position = std::stoull(argv[4]);
    const std::size_t rows = std::stoull(argv[5]);
    const std::string out = argv[6];
    Require(rows >= 1 && rows <= position + 1, "invalid row count");

    std::ifstream tokens_file(argv[2], std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(tokens_file)),
                                  {});
    Require((offset + position + 1) * 4 <= bytes.size(),
            "position past the token file");
    std::vector<gufo::tokenization::TokenId> tokens(position + 1);
    for (std::size_t i = 0; i <= position; ++i) {
      std::uint32_t value = 0;
      std::memcpy(&value, bytes.data() + (offset + i) * 4, 4);
      tokens[i] = value;
    }

    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
    Require(reader != nullptr, error);
    auto executor = gufo::hip::QwenGpuExecutor::CreateFromGguf(
        std::shared_ptr<const gufo::core::GgufReader>(std::move(reader)),
        &error, static_cast<std::uint32_t>(position + 8));
    Require(executor != nullptr, error);
    const auto& config = executor->GetConfig();

    const std::size_t prefix = position + 1 - rows;
    executor->SetPromptHiddenCapture(false);
    if (prefix != 0) {
      (void)executor->ForwardPromptBatch(std::span(tokens).first(prefix), 0,
                                         false);
    }
    std::vector<std::uint32_t> layers(config.num_layers);
    std::iota(layers.begin(), layers.end(), 0U);
    executor->SetPromptHiddenCapture(true, layers);
    (void)executor->ForwardPromptBatch(std::span(tokens).subspan(prefix),
                                       static_cast<std::uint32_t>(prefix));
    const auto hidden = executor->GetPromptHiddenStates();
    Require(hidden.size() == rows * layers.size() * config.hidden_size,
            "unexpected hidden capture size");
    WriteFloats(out + ".hidden.f32", hidden);
    WriteFloats(out + ".logits.f32", executor->CopyLastLogits());
    std::ofstream meta(out + ".txt");
    meta << "model " << model_path << "\nrows " << rows << "\nlayers "
         << layers.size() << "\nhidden " << config.hidden_size << "\nvocab "
         << config.vocab_size << "\nfirst_position " << prefix
         << "\nlast_position " << position << "\n";
    const auto& weights = executor->GetModel().GetWeights();
    std::cout << "RESULT rq_hidden_probe rows=" << rows
              << " layers=" << layers.size() << " position=" << position
              << " rotated_groups=\"" << weights.rotated_groups
              << "\" folded_bytes=" << weights.folded_bytes
              << " fold_seconds=" << weights.fold_seconds << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
