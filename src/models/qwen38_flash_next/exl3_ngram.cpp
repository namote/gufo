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

std::vector<std::filesystem::path> Exl3Ngram::PackFiles(
    const std::filesystem::path& directory) {
  const auto single = directory / "ngram_embedding.safetensors";
  std::error_code ec;
  // Parts: ngram_embedding-NNNNN-of-MMMMM.safetensors.
  std::map<unsigned, std::filesystem::path> parts;
  unsigned total = 0;
  for (const auto& entry : std::filesystem::directory_iterator(directory, ec)) {
    const auto name = entry.path().filename().string();
    constexpr std::string_view kHead = "ngram_embedding-";
    constexpr std::string_view kTail = ".safetensors";
    if (!name.starts_with(kHead) || !name.ends_with(kTail))
      continue;
    const auto middle = std::string_view(name).substr(
        kHead.size(), name.size() - kHead.size() - kTail.size());
    unsigned index = 0, count = 0;
    const auto of = middle.find("-of-");
    if (of != 5 || middle.size() != 14 ||
        std::from_chars(middle.data(), middle.data() + 5, index).ptr !=
            middle.data() + 5 ||
        std::from_chars(middle.data() + 9, middle.data() + 14, count).ptr !=
            middle.data() + 14 ||
        index == 0 || index > count || (total && count != total))
      Fail("invalid table part name " + name);
    total = count;
    if (!parts.emplace(index, entry.path()).second)
      Fail("duplicate table part " + name);
  }
  if (ec)
    Fail("cannot list " + directory.string() + ": " + ec.message());
  const bool has_single = std::filesystem::is_regular_file(single);
  if (has_single && !parts.empty())
    Fail("both ngram_embedding.safetensors and table parts are present");
  if (has_single)
    return {single};
  if (parts.empty())
    Fail("missing ngram_embedding.safetensors");
  if (parts.size() != total)
    Fail("missing table parts: " + std::to_string(parts.size()) + " of " +
         std::to_string(total));
  std::vector<std::filesystem::path> out;
  for (auto& [index, path] : parts)
    out.push_back(std::move(path));
  return out;
}

Exl3Ngram Exl3Ngram::Open(std::span<const std::filesystem::path> paths) {
  if (paths.empty() || paths.size() > 4096)
    Fail("invalid table part count");
  Exl3Ngram out;
  struct Entry {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t begin, end;  // absolute file offsets
    std::uint32_t file;
  };
  std::map<std::string, Entry> entries;
  std::string prefix;
  std::uint64_t row_bytes = 0;
  std::string codec;  // the first part's codec metadata, for equality
  for (std::uint32_t f = 0; f < paths.size(); ++f) {
    std::string error;
    auto file = platform::ReadOnlyMappedFile::Open(paths[f], &error);
    if (!file.valid())
      Fail(error);
    if (file.size() < 8)
      Fail("truncated header length");
    std::uint64_t n;
    std::memcpy(&n, file.data(), 8);
    if (n == 0 || n > (64U << 20) || n > file.size() - 8)
      Fail("invalid header length");
    const auto header = json::parse(
        std::string_view(reinterpret_cast<const char*>(file.data() + 8), n));
    const auto* meta = header.find("__metadata__");
    if (!meta || meta->member_str("format") != "exl3_ngram_trellis" ||
        meta->member_str("version") != "1" ||
        meta->member_str("codebook") != "mul1")
      Fail("unsupported codec");
    const auto this_codec = meta->member_str("K") + "/" +
                            meta->member_str("row_dim") + "/" +
                            meta->member_str("rows");
    if (f == 0) {
      codec = this_codec;
      out.bits = Decimal(*meta, "K");
      out.dim = Decimal(*meta, "row_dim");
      out.rows = Decimal(*meta, "rows");
      if (out.bits < 1 || out.bits > 8 || out.dim != 160 || out.rows == 0)
        Fail("unsupported ring geometry");
      row_bytes = 2 * (1 + out.dim * out.bits / 16);
    } else if (this_codec != codec) {
      Fail("table parts disagree on K/row_dim/rows");
    }
    const std::uint64_t data = 8 + n;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
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
              Integer(range->items()[1]),
              f};
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
          e.end > file.size() - data || e.end - e.begin != count * size)
        Fail("invalid dtype/payload for " + name);
      ranges.emplace_back(e.begin, e.end);
      e.begin += data;
      e.end += data;
      if (name.ends_with(".head_bias")) {
        if (!prefix.empty())
          Fail("multiple embedding tables");
        prefix = name.substr(0, name.size() - 10);
      }
      if (!entries.emplace(name, std::move(e)).second)
        Fail("tensor " + name + " appears in several table parts");
    }
    std::sort(ranges.begin(), ranges.end());
    for (std::size_t i = 1; i < ranges.size(); ++i)
      if (ranges[i].first < ranges[i - 1].second)
        Fail("overlapping tensors");
    out.files.push_back(std::move(file));
  }
  if (prefix.empty())
    Fail("missing head bias");
  const auto at = [&](const std::string& suffix) -> const Entry& {
    const auto it = entries.find(prefix + suffix);
    if (it == entries.end())
      Fail("missing " + suffix.substr(1));
    return it->second;
  };
  const auto bytes = [&](const Entry& e) {
    return out.files[e.file].data() + e.begin;
  };
  const auto& bias = at(".head_bias");
  if (bias.dtype != "F16" || bias.shape.size() != 2 ||
      bias.shape[1] != out.dim || bias.shape[0] > Config::kMaxPleHeads)
    Fail("invalid head bias");
  for (std::uint64_t i = 0; i < bias.end - bias.begin; i += 2) {
    std::uint16_t h;
    std::memcpy(&h, bytes(bias) + i, 2);
    const float value = quant::exl3::HalfToDouble(h);
    if (!std::isfinite(value))
      Fail("non-finite head bias");
    out.bias.push_back(value);
  }
  const auto integers = [&](const char* suffix,
                            std::vector<std::uint64_t>& dst) {
    const auto& e = at(suffix);
    if (e.dtype != "I64" || e.shape.size() != 1 ||
        e.shape[0] > Config::kMaxPleHeads)
      Fail("invalid hash metadata");
    for (std::uint64_t i = 0; i < e.end - e.begin; i += 8) {
      std::int64_t v;
      std::memcpy(&v, bytes(e) + i, 8);
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
    out.segments.push_back({first, e.shape[0], e.begin, e.file});
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
