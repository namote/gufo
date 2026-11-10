#include "src/core/quant/exl3/coverage.hpp"

#include <cstdint>
#include <vector>

#include "src/cli/serve/logging.hpp"

namespace gufo::quant::exl3 {

bool FastKernelsSupport(Codebook codebook, BitRate rate, std::size_t k,
                        std::size_t n) noexcept {
  return Supports(codebook, rate) && k > 0 && n > 0 && k % kHadamardSize == 0 &&
         n % kHadamardSize == 0;
}

std::string RateName(BitRate rate) {
  return "K" + std::to_string(rate.bits) + (rate.half ? ".5" : "");
}

std::string_view CodebookName(Codebook codebook) noexcept {
  switch (codebook) {
    case Codebook::kMul1:
      return "mul1";
    case Codebook::kMcg:
      return "mcg";
    case Codebook::k3Inst:
      return "3inst";
  }
  return "unknown";
}

std::string DescribeReferencePath(
    std::string_view model, std::span<const DeviceLinear* const> linears) {
  struct Group {
    Codebook codebook;
    BitRate rate;
    std::uint32_t k, n;
    std::size_t count;
  };
  std::vector<Group> groups;
  std::size_t slow = 0;
  for (const DeviceLinear* l : linears) {
    if (FastKernelsSupport(l->codebook, l->rate, l->k, l->n))
      continue;
    ++slow;
    bool found = false;
    for (Group& g : groups) {
      if (g.codebook == l->codebook && g.rate == l->rate && g.k == l->k &&
          g.n == l->n) {
        ++g.count;
        found = true;
        break;
      }
    }
    if (!found)
      groups.push_back({l->codebook, l->rate, l->k, l->n, 1});
  }
  std::string line = "event=exl3_reference_path model=";
  line += model;
  line += " tensors=" + std::to_string(slow);
  line += " of=" + std::to_string(linears.size());
  line += " formats=";
  if (groups.empty())
    line += "none";
  for (std::size_t i = 0; i < groups.size(); ++i) {
    const Group& g = groups[i];
    if (i != 0)
      line += ',';
    line += CodebookName(g.codebook);
    line += '/' + RateName(g.rate) + '/' + std::to_string(g.k) + 'x' +
            std::to_string(g.n) + ':' + std::to_string(g.count);
  }
  return line;
}

void LogReferencePath(std::string_view model,
                      std::span<const DeviceLinear* const> linears) {
  const std::string line = DescribeReferencePath(model, linears);
  bool slow = false;
  for (const DeviceLinear* l : linears)
    slow |= !FastKernelsSupport(l->codebook, l->rate, l->k, l->n);
  if (slow)
    server::Logger::Warn("exl3", line);
  else
    server::Logger::Info("exl3", line);
}

}  // namespace gufo::quant::exl3
