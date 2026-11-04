#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_EXL3_NGRAM_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_EXL3_NGRAM_HPP_

#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "src/core/platform/mapped_file.hpp"
#include "src/models/qwen38_flash_next/config.hpp"

namespace gufo::models::qwen38_flash_next {

// Native embedding rings are not EXL3 matrix tiles: no Hadamard rotation.
// Keep compressed rows on disk, including non-contiguous shard tensors and
// tables split over several safetensors files.
struct Exl3Ngram {
  struct Segment {
    std::uint64_t first, rows, offset;
    std::uint32_t file;  ///< index into `files`
  };
  std::vector<platform::ReadOnlyMappedFile> files;
  std::string prefix;
  unsigned bits{0};
  std::uint32_t dim{0};
  std::uint64_t rows{0};
  std::vector<Segment> segments;
  std::vector<float> bias;
  std::vector<std::uint64_t> offsets, vocab, multipliers;

  /// The table files of a native pack directory: ngram_embedding.safetensors,
  /// or every part ngram_embedding-NNNNN-of-MMMMM.safetensors (1..M, all
  /// present). Throws when neither form is complete or both are present.
  static std::vector<std::filesystem::path> PackFiles(
      const std::filesystem::path& directory);
  /// Opens one table from `files` (one or more parts with identical codec
  /// metadata). The hash metadata lives in exactly one part; row shards may
  /// lie in any part but must be unique and cover every row in order.
  static Exl3Ngram Open(std::span<const std::filesystem::path> files);
  void Validate(const Config& config) const;
};

}  // namespace gufo::models::qwen38_flash_next
#endif
