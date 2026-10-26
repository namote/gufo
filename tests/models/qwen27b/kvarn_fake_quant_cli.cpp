// Applies the KVarN fake quantization to a raw FP16 [token][head][dim] file,
// for the independent NumPy check in tools/qwen27b/kvarn.py.
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "tests/models/qwen27b/kvarn_fake_quant.hpp"

int main(int argc, char** argv) {
  namespace kvarn = gufo::testing::kvarn;
  try {
    if (argc != 9) {
      std::cerr << "Usage: qwen27b_kvarn_fake_quant CONFIG key|value HEADS DIM "
                   "FIRST_GROUP END_GROUP IN.f16 OUT.f16\n";
      return 2;
    }
    const auto config = kvarn::ParseConfig(argv[1]);
    const std::string kind = argv[2];
    if (!config || (kind != "key" && kind != "value"))
      throw std::invalid_argument("invalid configuration or kind");
    std::ifstream in(argv[7], std::ios::binary);
    std::vector<char> bytes{std::istreambuf_iterator<char>(in), {}};
    if (!in.is_open() || bytes.empty() || bytes.size() % 2 != 0)
      throw std::runtime_error("cannot read FP16 rows");
    std::vector<std::uint16_t> rows(bytes.size() / 2);
    std::memcpy(rows.data(), bytes.data(), bytes.size());
    const auto error = kvarn::FakeQuantize(
        rows, std::stoul(argv[3]), std::stoul(argv[4]), kind == "key", *config,
        std::stoul(argv[5]), std::stoul(argv[6]));
    std::ofstream out(argv[8], std::ios::binary);
    out.write(reinterpret_cast<const char*>(rows.data()),
              static_cast<std::streamsize>(rows.size() * 2));
    if (!out)
      throw std::runtime_error("cannot write FP16 rows");
    std::cout << "squared=" << error.squared << " reference=" << error.reference
              << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
