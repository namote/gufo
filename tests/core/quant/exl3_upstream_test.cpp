// EXL3 CPU reference against the upstream ExLlamaV3 runtime
// (tools/qwen27b/rq_quality/exl3/dump_ref.py, image exl3-eval:1.6.5-rocm100).
//
//   exl3_upstream_test SYNTH_JSON [PACK_JSON]
//
// SYNTH_JSON (committed fixture): synthetic tiles for every K, codebook and
// half-integer K. The trellis, suh, svh and x are regenerated here from the
// same splitmix64 streams. Checks: the inner FP16 matrix W' is bit-identical
// (sha256 of the row-major bits), the FP64 forward ((x suh) H) W' H svh
// matches upstream's FP64 evaluation, and upstream's FP16 runtime output lies
// within FP16 rounding of it.
// PACK_JSON (optional, needs the checkpoints at the recorded paths): the
// sha256 of the full inner matrix of real checkpoint tensors.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"
#include "src/core/quant/exl3/checkpoint.hpp"
#include "src/core/quant/exl3/format.hpp"

namespace {

using namespace gufo::quant::exl3;

int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

gufo::json::Value ReadJson(const char* path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream text;
  text << in.rdbuf();
  return gufo::json::parse(text.str());
}

struct SplitMix64 {
  std::uint64_t state;
  std::uint64_t Next() {
    state += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
};

std::vector<std::uint16_t> StreamU16(std::uint64_t seed, std::size_t count) {
  SplitMix64 rng{seed};
  std::vector<std::uint16_t> out(count);
  for (auto& v : out)
    v = static_cast<std::uint16_t>(rng.Next() & 0xffff);
  return out;
}

std::vector<std::uint16_t> ScaleBits(std::uint64_t seed, std::size_t count,
                                     std::uint16_t base) {
  auto r = StreamU16(seed, count);
  for (auto& v : r) {
    v = static_cast<std::uint16_t>(((v >> 10) & 1) << 15 | base | (v & 0x3ff));
  }
  return r;
}

std::string Sha256Bits(const std::vector<std::uint16_t>& bits) {
  return gufo::crypto::Sha256Hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(bits.data()), bits.size() * 2));
}

Codebook ParseCodebook(const std::string& name) {
  if (name == "mcg")
    return Codebook::kMcg;
  if (name == "mul1")
    return Codebook::kMul1;
  return Codebook::k3Inst;
}

BitRate RateOf(double K) {
  const int bits = static_cast<int>(K);
  return BitRate{bits, K - bits > 0.25};
}

void CheckSynth(const gufo::json::Value& report) {
  const auto* cases = report.find("cases");
  Expect(cases != nullptr && cases->is_array() && cases->size() == 27,
         "synthetic fixture has 27 cases");
  if (cases == nullptr)
    return;
  for (const auto& c : cases->items()) {
    const auto cb = ParseCodebook(c.member_str("codebook"));
    const auto rate = RateOf(c.member_double("K"));
    const std::size_t k = c.member_size("k"), n = c.member_size("n");
    const auto seed = static_cast<std::uint64_t>(c.member_double("seed"));
    const std::string name =
        c.member_str("codebook") + " K=" + std::to_string(c.member_double("K"));
    const auto trellis =
        StreamU16(seed, (k / 16) * (n / 16) * rate.TileWords());
    const auto suh = ScaleBits(seed + 0x100, k, 0x3800);
    const auto svh = ScaleBits(seed + 0x200, n, 0x2000);
    const auto x16 = ScaleBits(seed + 0x300, k, 0x3c00);
    std::vector<std::uint16_t> inner(k * n);
    DequantInner(trellis.data(), k, n, rate, cb, inner.data());
    Expect(Sha256Bits(inner) == c.member_str("inner_sha256"),
           name + ": inner W' bits differ from upstream reconstruct");
    const auto* head = c.find("inner_head");
    if (head != nullptr) {
      for (std::size_t i = 0; i < head->size(); ++i) {
        if (inner[i] != head->items()[i].as_size()) {
          std::cerr << "  " << name << ": first difference at " << i
                    << ": ours 0x" << std::hex << inner[i] << " upstream 0x"
                    << head->items()[i].as_size() << std::dec << "\n";
          break;
        }
      }
    }
    std::vector<double> x(k), y(n);
    for (std::size_t i = 0; i < k; ++i)
      x[i] = HalfToDouble(x16[i]);
    GemvReference(x, inner.data(), suh.data(), svh.data(), k, n, y);
    const auto& y64 = c.find("y_fp64")->items();
    const auto& yrt = c.find("y_runtime_f16")->items();
    double scale = 0.0;
    for (const auto& v : y64)
      scale = std::max(scale, std::fabs(v.as_double()));
    double err64 = 0.0, errrt = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      err64 = std::max(err64, std::fabs(y[j] - y64[j].as_double()));
      errrt = std::max(errrt, std::fabs(y[j] - yrt[j].as_double()));
    }
    Expect(err64 <= 1e-9 * scale, name +
                                      ": FP64 forward differs from upstream (" +
                                      std::to_string(err64 / scale) + ")");
    Expect(errrt <= 2e-2 * scale, name +
                                      ": upstream FP16 runtime is not within "
                                      "FP16 rounding (" +
                                      std::to_string(errrt / scale) + ")");
    std::cout << name << ": inner bits ok="
              << (Sha256Bits(inner) == c.member_str("inner_sha256"))
              << " fp64 rel " << err64 / scale << " runtime rel "
              << errrt / scale << "\n";
  }
}

void CheckPacks(const gufo::json::Value& report) {
  const auto* tensors = report.find("tensors");
  if (tensors == nullptr)
    return;
  for (const auto& t : tensors->items()) {
    const std::string module = t.member_str("module");
    const auto checkpoint = Checkpoint::Open(t.member_str("pack"));
    const auto linear = checkpoint.GetLinear(module);
    Expect(linear.in_features == t.member_size("k") &&
               linear.out_features == t.member_size("n"),
           module + ": shape");
    std::vector<std::uint16_t> inner(linear.in_features * linear.out_features);
    DequantInner(linear.trellis, linear.in_features, linear.out_features,
                 linear.rate, linear.codebook, inner.data());
    const bool ok = Sha256Bits(inner) == t.member_str("inner_sha256");
    Expect(ok, module + ": inner W' bits differ from upstream");
    std::cout << t.member_str("pack") << " " << module
              << ": K=" << t.member_double("K") << " " << linear.in_features
              << "x" << linear.out_features << " bits ok=" << ok << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: exl3_upstream_test SYNTH_JSON [PACK_JSON]\n";
    return 2;
  }
  CheckSynth(ReadJson(argv[1]));
  if (argc >= 3) {
    CheckPacks(ReadJson(argv[2]));
  }
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "exl3_upstream_test: OK\n";
  return 0;
}
