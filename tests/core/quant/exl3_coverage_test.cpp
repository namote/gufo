// EXL3 kernel coverage (src/core/quant/exl3/coverage.hpp) and the trellis
// code tables of the fast kernels (src/core/hip/detail/exl3_gemv.hpp, host
// part): every format the checkpoint loader accepts has exactly one
// compile-time code, the mul1 codes are the plain rate codes the kernels had
// before the other codebooks, the dispatcher reaches each format's code and
// rejects everything else, and the startup line names the projections that
// fall back to the reference path.

#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/hip/detail/exl3_gemv.hpp"
#include "src/core/quant/exl3/coverage.hpp"

namespace {

namespace ex = gufo::quant::exl3;
namespace code = gufo::hip::exl3;

int failures = 0;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

constexpr ex::Codebook kCodebooks[] = {ex::Codebook::k3Inst, ex::Codebook::kMcg,
                                       ex::Codebook::kMul1};

std::vector<ex::BitRate> AllRates() {
  std::vector<ex::BitRate> rates;
  for (int bits = 0; bits <= 9; ++bits)
    for (bool half : {false, true})
      rates.push_back({bits, half});
  return rates;
}

std::string Name(ex::Codebook cb, ex::BitRate rate) {
  return std::string(ex::CodebookName(cb)) + "/" + ex::RateName(rate);
}

void TestCodeTable() {
  // The mul1 codes are the rate codes (unchanged kernel instantiations).
  for (int bits = 1; bits <= 8; ++bits)
    Expect(code::TrellisCode(ex::Codebook::kMul1, {bits, false}) == bits,
           "mul1 integer code is the rate");
  for (int bits = 1; bits <= 3; ++bits)
    Expect(code::TrellisCode(ex::Codebook::kMul1, {bits, true}) ==
               code::kHalfRate + bits,
           "mul1 half code is the half rate code");
  std::set<int> table(std::begin(code::kTrellisCodes),
                      std::end(code::kTrellisCodes));
  Expect(table.size() == code::kTrellisCodeCount, "codes are distinct");
  Expect(code::kTrellisCodeCount == 27, "27 formats: 8 + 8 + 8 + 3 half");
  std::set<int> supported;
  for (const auto cb : kCodebooks) {
    for (const auto rate : AllRates()) {
      const bool loader = ex::Supports(cb, rate);
      Expect(code::TrellisSupported(cb, rate) == loader,
             "kernel formats equal the loader's: " + Name(cb, rate));
      Expect(ex::FastKernelsSupport(cb, rate, 128, 256) == loader,
             "fast kernels take every loader format: " + Name(cb, rate));
      if (!loader)
        continue;
      const int c = code::TrellisCode(cb, rate);
      supported.insert(c);
      Expect(table.contains(c), "code listed: " + Name(cb, rate));
      Expect(code::CodeSupported(c), "code supported: " + Name(cb, rate));
      Expect(code::CodeCodebookEnum(c) == cb, "codebook round trip");
      Expect(code::RateBits(c) == rate.bits && code::RateHalf(c) == rate.half,
             "rate round trip: " + Name(cb, rate));
      Expect(code::TileDwords(c) * 2 == static_cast<unsigned>(rate.TileWords()),
             "tile dwords: " + Name(cb, rate));
      for (int p = 0; p < 256; ++p)
        Expect(code::StateEnd(c, p) == ex::StateEndBit(rate, p),
               "state end: " + Name(cb, rate));
    }
  }
  Expect(supported == table, "table lists exactly the loader formats");
}

void TestDispatch() {
  for (const auto cb : kCodebooks) {
    for (const auto rate : AllRates()) {
      int reached = -1;
      bool threw = false;
      try {
        code::DispatchTrellisCode(cb, rate,
                                  [&]<int Code>() { reached = Code; });
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      if (ex::Supports(cb, rate)) {
        Expect(!threw && reached == code::TrellisCode(cb, rate),
               "dispatch reaches the format's code: " + Name(cb, rate));
      } else {
        Expect(threw && reached == -1,
               "dispatch rejects an unsupported format: " + Name(cb, rate));
      }
    }
  }
}

void TestShapes() {
  const ex::BitRate k4{4, false};
  Expect(ex::FastKernelsSupport(ex::Codebook::kMcg, k4, 128, 128), "128 x 128");
  Expect(!ex::FastKernelsSupport(ex::Codebook::kMcg, k4, 0, 128), "k = 0");
  Expect(!ex::FastKernelsSupport(ex::Codebook::kMcg, k4, 128, 0), "n = 0");
  Expect(!ex::FastKernelsSupport(ex::Codebook::kMcg, k4, 144, 128), "k % 128");
  Expect(!ex::FastKernelsSupport(ex::Codebook::kMcg, k4, 128, 80), "n % 128");
  Expect(!ex::FastKernelsSupport(ex::Codebook::kMcg, {3, true}, 128, 128),
         "mcg K3.5 is not a format");
}

void TestDescribe() {
  ex::DeviceLinear fast{};
  fast.k = 256;
  fast.n = 512;
  fast.rate = {3, true};
  fast.codebook = ex::Codebook::kMul1;
  ex::DeviceLinear odd = fast;
  odd.n = 80;
  odd.rate = {4, false};
  odd.codebook = ex::Codebook::k3Inst;
  ex::DeviceLinear odd2 = odd;
  odd2.k = 144;
  odd2.n = 128;
  odd2.codebook = ex::Codebook::kMcg;
  const std::vector<const ex::DeviceLinear*> none{&fast, &fast};
  Expect(
      ex::DescribeReferencePath("qwen", none) ==
          "event=exl3_reference_path model=qwen tensors=0 of=2 "
          "formats=none",
      "no reference projections: " + ex::DescribeReferencePath("qwen", none));
  const std::vector<const ex::DeviceLinear*> some{&odd, &fast, &odd2, &odd};
  const auto line = ex::DescribeReferencePath("qwen38_flash_next", some);
  Expect(line ==
             "event=exl3_reference_path model=qwen38_flash_next tensors=3 "
             "of=4 formats=3inst/K4/256x80:2,mcg/K4/144x128:1",
         "grouped reference projections: " + line);
  Expect(ex::RateName({3, true}) == "K3.5" && ex::RateName({8, false}) == "K8",
         "rate names");
}

}  // namespace

int main() {
  TestCodeTable();
  TestDispatch();
  TestShapes();
  TestDescribe();
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "exl3_coverage_test: OK\n";
  return 0;
}
