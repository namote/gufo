#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/qwen38_flash_next/exl3_weights.hpp"

namespace q = gufo::models::qwen38_flash_next;
namespace {
using Bytes = std::vector<std::uint8_t>;

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void Integer(Bytes& bytes, std::uint64_t value, unsigned width) {
  for (unsigned i = 0; i < width; ++i)
    bytes.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}

void String(Bytes& bytes, std::string_view value) {
  Integer(bytes, value.size(), 8);
  bytes.insert(bytes.end(), value.begin(), value.end());
}

struct Shard {
  Bytes bytes;
  std::size_t data_offset;
};

Shard MakeShard(unsigned index) {
  Bytes bytes{'G', 'G', 'U', 'F'};
  Integer(bytes, 3, 4);
  Integer(bytes, index == 1 ? 1 : 0, 8);
  Integer(bytes, 5, 8);
  String(bytes, "general.architecture");
  Integer(bytes, 8, 4);
  String(bytes, "qwen4exp");
  const unsigned alignment = index == 0 ? 32 : 128;
  const auto scalar = [&](std::string_view key, std::uint64_t value,
                          unsigned width = 4) {
    String(bytes, key);
    Integer(bytes, width == 8 ? 10 : 4, 4);
    Integer(bytes, value, width);
  };
  scalar("general.alignment", alignment);
  scalar("split.no", index);
  scalar("split.count", 2);
  scalar("split.tensors.count", 1, 8);
  if (index == 1) {
    String(bytes, "ple.weight");
    Integer(bytes, 1, 4);
    Integer(bytes, 4, 8);
    Integer(bytes, 0, 4);
    Integer(bytes, 128, 8);
  }
  // Nonzero padding is part of the header; the unused prefix before the
  // first tensor is data. A minimum-tensor-address heuristic hashes both.
  while (bytes.size() % alignment != 0)
    bytes.push_back(0xa5);
  const auto data_offset = bytes.size();
  if (index == 1)
    bytes.resize(data_offset + 128 + 16, 0x5a);
  return {std::move(bytes), data_offset};
}

void Write(const std::filesystem::path& path, const Bytes& bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  Require(static_cast<bool>(out), "could not write fixture shard");
}

void TensorlessShardDigest() {
  const auto directory =
      std::filesystem::temp_directory_path() /
      ("qfn-exl3-ple-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(directory);
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{directory};
  const auto first_path = directory / "ple-00001-of-00002.gguf";
  const auto second_path = directory / "ple-00002-of-00002.gguf";
  auto first = MakeShard(0);
  auto second = MakeShard(1);
  Write(first_path, first.bytes);
  Write(second_path, second.bytes);

  // Independently computed with Python hashlib over each basename + NUL +
  // decimal file size + NUL + raw aligned header, in shard-number order.
  constexpr std::string_view expected =
      "5ce1e58d6d5ea10e901d167f4b9bcc5ff114294c2e26ebe15de22fbbfda85fc9";
  const auto digest = [&](const std::filesystem::path& path) {
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
    Require(reader != nullptr, "fixture open failed: " + error);
    const auto regions = reader->GetMappedRegions();
    Require(regions.size() == 2 && reader->GetTensorCount() == 1,
            "split fixture did not retain both shards");
    Require(regions[0].data_offset == first.data_offset &&
                regions[1].data_offset == second.data_offset,
            "mapped regions lost the parsed per-shard data offset");
    const auto* tensor = reader->FindTensor("ple.weight");
    const auto* expected_data =
        static_cast<const std::uint8_t*>(regions[1].data) + second.data_offset +
        128;
    Require(tensor != nullptr && tensor->data == expected_data,
            "GGUF tensor address changed");
    return q::Exl3PleHeadersDigest(*reader);
  };
  Require(digest(first_path) == expected, "PLE digest differs from Python");
  Require(digest(second_path) == expected,
          "opening another shard changed digest ordering");
  for (const auto* shard : {&first, &second}) {
    const auto reader = gufo::core::GgufReader::OpenMemory(shard->bytes.data(),
                                                           shard->bytes.size());
    Require(reader != nullptr, "in-memory fixture open failed");
    Require(reader->GetMappedRegions()[0].data_offset == shard->data_offset,
            "in-memory region lost the parsed data offset");
  }
  for (std::size_t i = second.data_offset; i < second.bytes.size(); ++i)
    second.bytes[i] ^= 0xff;
  Write(second_path, second.bytes);
  Require(digest(first_path) == expected,
          "tensor payload or pre-tensor gap leaked into the header digest");
  first.bytes.back() ^= 1;
  Write(first_path, first.bytes);
  Require(digest(first_path) != expected,
          "tensorless shard's alignment padding was omitted from digest");
}
}  // namespace

int main() {
  try {
    TensorlessShardDigest();
    std::cout << "Flash-Next EXL3 PLE header identity passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
