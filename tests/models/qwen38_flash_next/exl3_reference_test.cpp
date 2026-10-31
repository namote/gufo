#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

#include "src/models/qwen38_flash_next/cpu_ops.hpp"
#include "src/models/qwen38_flash_next/exl3_weights.hpp"

namespace q = gufo::models::qwen38_flash_next;
namespace ex = gufo::quant::exl3;
namespace {
void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

void Projection(int bits, unsigned k, unsigned n, bool permute) {
  std::mt19937 rng(20261031 + bits + k);
  std::vector<std::uint16_t> trellis(std::size_t{k / 16} * (n / 16) * 16 *
                                     bits);
  std::vector<std::uint16_t> suh(k), svh(n), inner(std::size_t{k} * n);
  for (auto& value : trellis)
    value = static_cast<std::uint16_t>(rng());
  for (auto& value : suh)
    value = ex::HalfFromDouble((rng() & 1 ? -1 : 1) *
                               (0.01 + (rng() % 100) * 0.0001));
  for (auto& value : svh)
    value = ex::HalfFromDouble((rng() & 1 ? -1 : 1) *
                               (0.1 + (rng() % 100) * 0.001));
  q::Exl3HostLinear h;
  h.view = {trellis.data(), suh.data(),         svh.data(), k, n,
            {bits, false},  ex::Codebook::kMul1};
  if (permute) {
    h.input_blocks.resize(k / 128);
    h.output_blocks.resize(n / 128);
    for (unsigned i = 0; i < k / 128; ++i)
      h.input_blocks[i] = k / 128 - 1 - i;
    for (unsigned i = 0; i < n / 128; ++i)
      h.output_blocks[i] = n / 128 - 1 - i;
  }
  ex::DequantInner(trellis.data(), k, n, {bits, false}, ex::Codebook::kMul1,
                   inner.data());
  std::vector<float> x(k), actual(n);
  for (auto& value : x)
    value = (static_cast<int>(rng() % 1000) - 500) * 0.00001F;
  q::Exl3MatVec(h, x, actual);
  // Independent dense Sylvester multiplication, with no Hadamard helper or
  // source permutation recipe in the expected path.
  std::vector<double> rotated(k), sums(n);
  const double scale = 1 / std::sqrt(128.0);
  for (unsigned i = 0; i < k; ++i) {
    double sum = 0;
    for (unsigned j = i / 128 * 128; j < i / 128 * 128 + 128; ++j) {
      const auto source = permute ? (k / 128 - 1 - j / 128) * 128 + j % 128 : j;
      sum += x[source] * ex::HalfToDouble(suh[j]) *
             (std::popcount((i % 128) & (j % 128)) % 2 ? -1 : 1);
    }
    rotated[i] = sum * scale;
  }
  for (unsigned j = 0; j < n; ++j)
    for (unsigned i = 0; i < k; ++i)
      sums[j] += rotated[i] * ex::HalfToDouble(inner[std::size_t{i} * n + j]);
  for (unsigned j = 0; j < n; ++j) {
    double sum = 0;
    for (unsigned i = j / 128 * 128; i < j / 128 * 128 + 128; ++i)
      sum += sums[i] * (std::popcount((i % 128) & (j % 128)) % 2 ? -1 : 1);
    const auto dest = permute ? (n / 128 - 1 - j / 128) * 128 + j % 128 : j;
    const auto expected =
        static_cast<float>(sum * scale * ex::HalfToDouble(svh[j]));
    Require(std::abs(actual[dest] - expected) <=
                1e-10F + std::abs(expected) * 1e-6F,
            "CPU projection differs from independent Sylvester formula");
  }
  q::TensorRef view;
  view.data = trellis.data();
  view.cols = k;
  view.rows = n;
  view.exl3 = {&h, 1};
  Require(view.RowBytes() == 0 && view.Expert(0) == nullptr,
          "EXL3 exposed a GGUF packed row");
  view.output_begin = 128;
  view.rows = n - 128;
  std::vector<float> sliced(n - 128);
  q::cpu::MatVec(view, 0, x, sliced);
  Require(std::equal(sliced.begin(), sliced.end(), actual.begin() + 128),
          "indexer slice happened before the output Hadamard");
}

void Permutations() {
  q::Config c;
  c.ssm_num_k_heads = 16;
  c.ssm_num_v_heads = 48;
  c.ssm_head_dim = 128;
  std::vector<bool> seen(6144);
  for (unsigned head = 0; head < 48; ++head) {
    for (unsigned d = 0; d < 128; ++d) {
      const auto actual = q::Exl3VHeadDestination(head * 128 + d, 128, c);
      const auto expected = ((head % 3) * 16 + head / 3) * 128 + d;
      Require(actual == expected && !seen[actual],
              "GDN V permutation is not bijective");
      seen[actual] = true;
    }
  }
}

void WriteTensor(const std::filesystem::path& path, std::string header,
                 std::size_t payload) {
  std::ofstream out(path, std::ios::binary);
  const std::uint64_t size = header.size();
  out.write(reinterpret_cast<const char*>(&size), 8);
  out.write(header.data(), header.size());
  const std::vector<char> bytes(payload);
  out.write(bytes.data(), bytes.size());
}

void CheckpointSources() {
  const auto directory =
      std::filesystem::temp_directory_path() /
      ("qfn-exl3-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(directory);
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{directory};
  std::ofstream(directory / "config.json")
      << R"({"quantization_config":{"quant_method":"exl3","bits":5,"version":"test"}})";
  std::ofstream(directory / "model.safetensors.index.json")
      << R"({"weight_map":{"source":"model.safetensors"}})";
  WriteTensor(
      directory / "model.safetensors",
      R"({"source":{"dtype":"F16","shape":[128],"data_offsets":[0,256]}})",
      256);
  const std::vector<std::filesystem::path> patch{directory /
                                                 "patch.safetensors"};
  WriteTensor(
      patch[0],
      R"({"mixer":{"dtype":"F16","shape":[128],"data_offsets":[0,256]}})", 256);
  {
    auto source = ex::Checkpoint::Open(directory, patch);
    Require(source.Find("mixer") != nullptr &&
                source.GetMappedRegions().size() == 2,
            "supplementary mixer was not mapped");
    const auto* tensor = source.Find("source");
    const auto regions = source.GetMappedRegions();
    Require(static_cast<const std::uint8_t*>(regions[tensor->shard].data) +
                    tensor->file_offset ==
                tensor->data,
            "source upload region differs from bound tensor");
  }
  const auto reject = [&](const char* message) {
    bool rejected = false;
    try {
      (void)ex::Checkpoint::Open(directory, patch);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    Require(rejected, message);
  };
  WriteTensor(
      patch[0],
      R"({"source":{"dtype":"F16","shape":[128],"data_offsets":[0,256]}})",
      256);
  reject("supplementary file silently overlaid a trunk tensor");
  WriteTensor(
      patch[0],
      R"({"mixer":{"dtype":"F16","shape":[128],"data_offsets":[0,256]}})", 128);
  reject("partial supplementary payload was accepted");
  std::filesystem::remove(directory / "model.safetensors");
  reject("missing indexed shard was accepted");
}
}  // namespace

int main() {
  try {
    Permutations();
    CheckpointSources();
    for (int bits : {5, 6, 7}) {
      Projection(bits, 256, 640, false);
      Projection(bits, 640, 256, true);
    }
    Projection(7, 6144, 128, true);
    Projection(7, 10240, 128, true);
    std::cout << "Flash-Next EXL3 CPU contracts passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
