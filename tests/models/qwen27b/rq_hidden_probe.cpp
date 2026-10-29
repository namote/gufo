// RQ R2 diagnosis probe: per-layer residual hidden states of a Qwen27B GGUF
// at the last rows of a corpus prefix, to compare a rotated file with its
// unrotated source layer by layer (plan/rotated-quant/R2-RESULTS.md).
//
//   qwen27b_rq_hidden_probe <model.gguf> <windows.bin> <window offset>
//       <position> <rows> <out prefix>
//
// Prefills tokens [offset, offset + position - rows] without capture, then
// the last `rows` tokens up to and including `position` with every layer's
// residual output captured. The arena taps five layers per pass, so the
// probe reruns the prompt from a reset state once per five layers and
// requires byte-equal logits across passes. Writes <out>.hidden.f32
// ([rows][layers][hidden] floats), <out>.logits.f32 (the last row's logits) and
// <out>.txt (shape).

#include <algorithm>
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

    // The arena taps at most a few layers per pass, so every chunk of
    // layers reruns the whole prefix from a reset state. Prefill is
    // deterministic: the logits of every pass must be byte-equal.
    constexpr std::size_t kTapsPerPass = 5;
    const std::size_t prefix = position + 1 - rows;
    const std::size_t layer_count = config.num_layers;
    const std::size_t hidden_size = config.hidden_size;
    std::vector<float> hidden(rows * layer_count * hidden_size);
    std::vector<float> logits;
    std::size_t passes = 0;
    for (std::size_t first = 0; first < layer_count; first += kTapsPerPass) {
      const std::size_t count = std::min(kTapsPerPass, layer_count - first);
      std::vector<std::uint32_t> layers(count);
      std::iota(layers.begin(), layers.end(),
                static_cast<std::uint32_t>(first));
      executor->Reset();
      executor->SetPromptHiddenCapture(false);
      if (prefix != 0) {
        (void)executor->ForwardPromptBatch(std::span(tokens).first(prefix), 0,
                                           false);
      }
      executor->SetPromptHiddenCapture(true, layers);
      (void)executor->ForwardPromptBatch(std::span(tokens).subspan(prefix),
                                         static_cast<std::uint32_t>(prefix));
      const auto taps = executor->GetPromptHiddenStates();
      Require(taps.size() == rows * count * hidden_size,
              "unexpected hidden capture size");
      for (std::size_t row = 0; row < rows; ++row) {
        std::memcpy(hidden.data() + ((row * layer_count) + first) * hidden_size,
                    taps.data() + (row * count * hidden_size),
                    count * hidden_size * sizeof(float));
      }
      const auto pass_logits = executor->CopyLastLogits();
      if (passes == 0) {
        logits.assign(pass_logits.begin(), pass_logits.end());
      } else {
        Require(std::equal(logits.begin(), logits.end(), pass_logits.begin(),
                           pass_logits.end()),
                "logits differ between capture passes");
      }
      ++passes;
    }
    WriteFloats(out + ".hidden.f32", hidden);
    WriteFloats(out + ".logits.f32", logits);
    std::ofstream meta(out + ".txt");
    meta << "model " << model_path << "\nrows " << rows << "\nlayers "
         << layer_count << "\nhidden " << config.hidden_size << "\nvocab "
         << config.vocab_size << "\nfirst_position " << prefix << "\npasses "
         << passes << "\nlast_position " << position << "\n";
    const auto& weights = executor->GetModel().GetWeights();
    std::cout << "RESULT rq_hidden_probe rows=" << rows
              << " layers=" << layer_count << " passes=" << passes
              << " position=" << position << " rotated_groups=\""
              << weights.rotated_groups
              << "\" folded_bytes=" << weights.folded_bytes
              << " fold_seconds=" << weights.fold_seconds << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
