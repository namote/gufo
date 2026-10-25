// Streamed Qwen snapshot serialization equals the whole-buffer form.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/bounce_stream.hpp"
#include "src/models/qwen/generator.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace {
void Expect(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::vector<std::uint8_t> Streamed(const gufo::hip::QwenGpuSnapshot& snapshot,
                                   std::size_t buffer_bytes) {
  gufo::hip::BounceStream bounce(buffer_bytes);
  std::vector<std::uint8_t> bytes;
  std::size_t largest = 0;
  snapshot.StreamCompact(bounce, [&](std::span<const std::uint8_t> piece) {
    largest = std::max(largest, piece.size());
    bytes.insert(bytes.end(), piece.begin(), piece.end());
  });
  // Host pieces (header, image grids) are small; device pieces never exceed
  // the bounce buffer.
  Expect(largest <= std::max<std::size_t>(buffer_bytes, 80),
         "a streamed piece exceeds the bounce buffer");
  return bytes;
}
}  // namespace

int main(int argc, const char* const* argv) {
  try {
    const char* model_path =
        argc > 1 ? argv[1] : std::getenv("GUFO_QWEN27B_MODEL");
    if (model_path == nullptr) {
      std::cout << "SKIP: pass a Qwen 27B GGUF path\n";
      return 77;
    }
    std::string error;
    auto reader_owner = gufo::core::GgufReader::OpenFile(model_path, &error);
    Expect(reader_owner != nullptr, error);
    const std::shared_ptr<const gufo::core::GgufReader> reader(
        std::move(reader_owner));
    auto model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
    Expect(model != nullptr, error);
    constexpr std::uint32_t context = 4096;
    auto executor = gufo::hip::QwenGpuExecutor::Create(model, &error, context);
    Expect(executor != nullptr, error);

    std::string text;
    for (int i = 0; i < 120; ++i)
      text += "Line " + std::to_string(i) + ": the quick brown fox. ";
    const auto prompt = model->GetTokenizer().Encode(text);
    gufo::models::GenerationOptions options;
    options.max_new_tokens = 2;
    options.sampling.temperature = 0.0F;
    const auto generated = executor->Generate(prompt, options);
    const auto valid =
        static_cast<std::uint32_t>(prompt.size() + generated.size());
    const auto snapshot = executor->SaveSnapshot(valid);

    std::vector<std::uint8_t> whole(snapshot->CompactPayloadBytes());
    Expect(snapshot->SerializeCompact(whole) == whole.size(),
           "whole-buffer serialization size");
    // Default buffer, a buffer smaller than one KV row (column pieces) and
    // an odd size that splits rows unevenly.
    for (const std::size_t buffer :
         {gufo::hip::BounceStream::kDefaultBufferBytes, std::size_t{12289},
          std::size_t{1} << 20}) {
      Expect(Streamed(*snapshot, buffer) == whole,
             "streamed bytes differ with a " + std::to_string(buffer) +
                 "-byte bounce buffer");
    }
    // Streaming the live session in place gives the snapshot's bytes.
    for (const std::size_t buffer :
         {gufo::hip::BounceStream::kDefaultBufferBytes, std::size_t{12289}}) {
      gufo::hip::BounceStream bounce(buffer);
      std::vector<std::uint8_t> live;
      executor->StreamLiveCompact(
          valid, bounce, [&](std::span<const std::uint8_t> piece) {
            live.insert(live.end(), piece.begin(), piece.end());
          });
      Expect(live == whole, "live stream differs from the snapshot bytes");
    }

    // Streamed restore: the same payload into a slot that held another
    // history, read in odd-sized pieces. Saving it again must give the same
    // bytes, and continuing must match the whole-buffer restore.
    const auto other = model->GetTokenizer().Encode(
        "An unrelated and longer history that leaves stale rows behind. " +
        text + text);
    const auto continue_from =
        [&](gufo::hip::QwenGpuExecutor& executor) {
          auto suffix = prompt;
          suffix.insert(suffix.end(), generated.begin(), generated.end());
          const auto tail = model->GetTokenizer().Encode(" Line 120:");
          suffix.insert(suffix.end(), tail.begin(), tail.end());
          gufo::models::GenerationOptions next;
          next.max_new_tokens = 4;
          next.sampling.temperature = 0.0F;
          return executor.GenerateFromPrefix(suffix, valid, next);
        };
    auto whole_slot = gufo::hip::QwenGpuExecutor::Create(model, &error, context);
    Expect(whole_slot != nullptr, error);
    (void)whole_slot->Generate(std::span(other).first(
                                   std::min<std::size_t>(other.size(), 3000)),
                               options);
    whole_slot->RestoreCompactSnapshot(whole, valid);
    // Continuing advances recurrent state past the boundary, so each slot
    // continues once and is compared with this reference.
    const auto reference = continue_from(*whole_slot);
    for (const std::size_t buffer :
         {gufo::hip::BounceStream::kDefaultBufferBytes, std::size_t{12289}}) {
      auto streamed_slot =
          gufo::hip::QwenGpuExecutor::Create(model, &error, context);
      Expect(streamed_slot != nullptr, error);
      (void)streamed_slot->Generate(
          std::span(other).first(std::min<std::size_t>(other.size(), 3000)),
          options);
      gufo::hip::BounceStream bounce(buffer);
      std::size_t offset = 0;
      streamed_slot->RestoreCompactSnapshotStream(
          bounce,
          [&](std::span<std::uint8_t> bytes) {
            Expect(offset + bytes.size() <= whole.size(),
                   "streamed restore read past the payload");
            std::copy_n(whole.begin() + offset, bytes.size(), bytes.begin());
            offset += bytes.size();
          },
          whole.size(), valid);
      Expect(offset == whole.size(), "streamed restore left payload unread");
      const auto saved = streamed_slot->SaveSnapshot(valid);
      std::vector<std::uint8_t> again(saved->CompactPayloadBytes());
      Expect(saved->SerializeCompact(again) == again.size() && again == whole,
             "streamed restore does not reproduce the snapshot bytes");
      Expect(continue_from(*streamed_slot) == reference,
             "streamed restore continues differently from a whole restore");
    }
    auto cold = gufo::hip::QwenGpuExecutor::Create(model, &error, context);
    Expect(cold != nullptr, error);
    (void)cold->Generate(prompt, options);
    Expect(continue_from(*cold) == reference,
           "restored continuation differs from the uninterrupted run");
    std::cout << "Qwen snapshot stream test passed: " << valid << " tokens, "
              << whole.size() << " bytes\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
