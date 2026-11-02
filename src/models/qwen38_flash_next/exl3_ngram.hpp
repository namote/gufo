#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_EXL3_NGRAM_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_EXL3_NGRAM_HPP_

#include <filesystem>
#include <vector>

#include "src/core/platform/mapped_file.hpp"
#include "src/models/qwen38_flash_next/config.hpp"

namespace gufo::models::qwen38_flash_next {

// Native embedding rings are not EXL3 matrix tiles: no Hadamard rotation.
// Keep compressed rows on disk, including non-contiguous shard tensors.
struct Exl3Ngram {
  struct Segment {
    std::uint64_t first, rows, offset;
  };
  platform::ReadOnlyMappedFile file;
  std::string prefix;
  unsigned bits{0};
  std::uint32_t dim{0};
  std::uint64_t rows{0};
  std::vector<Segment> segments;
  std::vector<float> bias;
  std::vector<std::uint64_t> offsets, vocab, multipliers;

  static Exl3Ngram Open(const std::filesystem::path& path);
  void Validate(const Config& config) const;
};

}  // namespace gufo::models::qwen38_flash_next
#endif
