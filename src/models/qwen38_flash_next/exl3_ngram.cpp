#include "src/models/qwen38_flash_next/exl3_ngram.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

#include "src/core/json.hpp"
#include "src/core/quant/exl3/format.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {
[[noreturn]] void Fail(const std::string& s) {
  throw std::runtime_error("EXL3 n-gram: " + s);
}
std::uint64_t Integer(const json::Value& v) {
  const double x = v.as_double(-1);
  if (!v.is_number() || !std::isfinite(x) || x < 0 || x > 9e15 ||
      std::floor(x) != x)
    Fail("invalid tensor dimension/offset");
  return static_cast<std::uint64_t>(x);
}
unsigned Decimal(const json::Value& meta, const char* key) {
  const auto s = meta.member_str(key);
  unsigned n = 0;
  const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), n);
  if (ec != std::errc{} || end != s.data() + s.size())
    Fail(std::string("invalid ") + key);
  return n;
}
}  // namespace

Exl3Ngram Exl3Ngram::Open(const std::filesystem::path& path) {
  Exl3Ngram out;
  std::string error;
  out.file = platform::ReadOnlyMappedFile::Open(path, &error);
  if (!out.file.valid())
    Fail(error);
  if (out.file.size() < 8)
    Fail("truncated header length");
  std::uint64_t n;
  std::memcpy(&n, out.file.data(), 8);
  if (n == 0 || n > (64U << 20) || n > out.file.size() - 8)
    Fail("invalid header length");
  const auto header = json::parse(
      std::string_view(reinterpret_cast<const char*>(out.file.data() + 8), n));
  const auto* meta = header.find("__metadata__");
  if (!meta || meta->member_str("format") != "exl3_ngram_trellis" ||
      meta->member_str("version") != "1" ||
      meta->member_str("codebook") != "mul1")
    Fail("unsupported codec");
  out.bits = Decimal(*meta, "K");
  out.dim = Decimal(*meta, "row_dim");
  out.rows = Decimal(*meta, "rows");
  if (out.bits < 1 || out.bits > 8 || out.dim != 160 || out.rows == 0)
    Fail("unsupported ring geometry");
  const auto row_bytes = 2 * (1 + out.dim * out.bits / 16);
  struct Entry {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t begin, end;
  };
  std::map<std::string, Entry> entries;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
  std::string prefix;
  for (const auto& [name, v] : header.members()) {
    if (name == "__metadata__")
      continue;
    const auto* shape = v.find("shape");
    const auto* range = v.find("data_offsets");
    if (!shape || !shape->is_array() || !range || !range->is_array() ||
        range->size() != 2)
      Fail("invalid tensor header");
    Entry e{v.member_str("dtype"),
            {},
            Integer(range->items()[0]),
            Integer(range->items()[1])};
    std::uint64_t count = 1;
    for (const auto& d : shape->items()) {
      const auto dim = Integer(d);
      if (dim == 0 || count > UINT64_MAX / dim)
        Fail("invalid tensor size");
      count *= dim;
      e.shape.push_back(dim);
    }
    const unsigned size =
        e.dtype == "I64" ? 8 : (e.dtype == "I16" || e.dtype == "F16" ? 2 : 0);
    if (!size || count > UINT64_MAX / size || e.begin > e.end ||
        e.end > out.file.size() - 8 - n || e.end - e.begin != count * size)
      Fail("invalid dtype/payload for " + name);
    ranges.emplace_back(e.begin, e.end);
    entries.emplace(name, std::move(e));
    if (name.ends_with(".head_bias")) {
      if (!prefix.empty())
        Fail("multiple embedding tables");
      prefix = name.substr(0, name.size() - 10);
    }
  }
  std::sort(ranges.begin(), ranges.end());
  for (std::size_t i = 1; i < ranges.size(); ++i)
    if (ranges[i].first < ranges[i - 1].second)
      Fail("overlapping tensors");
  if (prefix.empty())
    Fail("missing head bias");
  const auto& bias = entries.at(prefix + ".head_bias");
  if (bias.dtype != "F16" || bias.shape.size() != 2 ||
      bias.shape[1] != out.dim || bias.shape[0] > Config::kMaxPleHeads)
    Fail("invalid head bias");
  for (std::uint64_t i = bias.begin; i < bias.end; i += 2) {
    std::uint16_t h;
    std::memcpy(&h, out.file.data() + 8 + n + i, 2);
    const float value = quant::exl3::HalfToDouble(h);
    if (!std::isfinite(value))
      Fail("non-finite head bias");
    out.bias.push_back(value);
  }
  const auto integers = [&](const char* suffix,
                            std::vector<std::uint64_t>& dst) {
    const auto& e = entries.at(prefix + suffix);
    if (e.dtype != "I64" || e.shape.size() != 1 ||
        e.shape[0] > Config::kMaxPleHeads)
      Fail("invalid hash metadata");
    for (std::uint64_t i = e.begin; i < e.end; i += 8) {
      std::int64_t v;
      std::memcpy(&v, out.file.data() + 8 + n + i, 8);
      if (v < 0)
        Fail("negative hash metadata");
      dst.push_back(v);
    }
  };
  integers(".head_offsets", out.offsets);
  integers(".head_vocab_sizes", out.vocab);
  integers(".layer_multipliers", out.multipliers);
  if (out.offsets.size() != bias.shape[0] ||
      out.vocab.size() != out.offsets.size() || out.offsets.empty() ||
      out.offsets[0] != 0)
    Fail("invalid hash head count");
  for (std::size_t h = 0; h < out.offsets.size(); ++h) {
    if (out.vocab[h] == 0 || out.offsets[h] > out.rows ||
        out.vocab[h] > out.rows - out.offsets[h] ||
        (h && out.offsets[h] != out.offsets[h - 1] + out.vocab[h - 1]))
      Fail("invalid hash head range");
  }
  std::uint64_t first = 0;
  const bool single = entries.contains(prefix + ".trellis");
  for (std::size_t i = 0;; ++i) {
    const auto key =
        prefix +
        (single ? ".trellis" : ".shard_" + std::to_string(i) + ".trellis");
    const auto it = entries.find(key);
    if (it == entries.end())
      break;
    const auto& e = it->second;
    if (e.dtype != "I16" || e.shape.size() != 2 ||
        e.shape[1] * 2 != row_bytes || e.shape[0] > out.rows - first)
      Fail("invalid row tensor");
    out.segments.push_back({first, e.shape[0], 8 + n + e.begin});
    first += e.shape[0];
    if (single)
      break;
  }
  if (first != out.rows || entries.size() != out.segments.size() + 4)
    Fail("missing, extra or incomplete shard tensors");
  out.prefix = prefix;
  return out;
}

void Exl3Ngram::Validate(const Config& c) const {
  if (c.ple_layer < 0 ||
      prefix != "model.language_model.layers." + std::to_string(c.ple_layer) +
                    ".ple.ple_embedding.ngram_embedding" ||
      multipliers.size() > c.ple_multipliers.size() ||
      offsets.size() > c.ple_head_offsets.size() ||
      c.ple_rows != offsets.back() + vocab.back() || c.ple_head_dim != dim ||
      c.ple_heads != offsets.size() || c.ple_ngram_size != multipliers.size() ||
      c.ple_rows > rows ||
      !std::equal(offsets.begin(), offsets.end(), c.ple_head_offsets.begin()) ||
      !std::equal(vocab.begin(), vocab.end(), c.ple_head_vocab.begin()) ||
      !std::equal(multipliers.begin(), multipliers.end(),
                  c.ple_multipliers.begin()))
    Fail("model and table hash geometry differ");
}
}  // namespace gufo::models::qwen38_flash_next
