#include "src/core/quant/exl3/checkpoint.hpp"

#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "src/core/json.hpp"
#include "src/core/platform/mapped_file.hpp"

namespace gufo::quant::exl3 {

struct Checkpoint::Shard {
  platform::ReadOnlyMappedFile file;
  std::filesystem::path path;
};

namespace {

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error("EXL3 checkpoint: " + message);
}

json::Value ReadJson(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    Fail("cannot read " + path.string());
  }
  std::ostringstream text;
  text << in.rdbuf();
  try {
    return json::parse(text.str());
  } catch (const std::exception& error) {
    Fail(path.string() + ": " + error.what());
  }
}

DType ParseDType(const std::string& name) {
  if (name == "F16")
    return DType::kF16;
  if (name == "BF16")
    return DType::kBF16;
  if (name == "F32")
    return DType::kF32;
  if (name == "I16")
    return DType::kI16;
  if (name == "I32")
    return DType::kI32;
  if (name == "U8")
    return DType::kU8;
  return DType::kOther;
}

std::size_t ElementBytes(DType dtype) {
  switch (dtype) {
    case DType::kF16:
    case DType::kBF16:
    case DType::kI16:
      return 2;
    case DType::kF32:
    case DType::kI32:
      return 4;
    case DType::kU8:
      return 1;
    case DType::kOther:
      return 0;
  }
  return 0;
}

std::int64_t Integer(const json::Value& value, const std::string& what) {
  if (!value.is_number()) {
    Fail(what + " is not a number");
  }
  const double d = value.as_double();
  if (d < 0 || d > 9.0e15 ||
      d != static_cast<double>(static_cast<std::int64_t>(d))) {
    Fail(what + " is not a non-negative integer");
  }
  return static_cast<std::int64_t>(d);
}

QuantizationConfig ParseConfig(const json::Value& config,
                               const std::filesystem::path& path) {
  const json::Value* q = config.find("quantization_config");
  if (q == nullptr) {
    if (const json::Value* text = config.find("text_config")) {
      q = text->find("quantization_config");
    }
  }
  if (q == nullptr || !q->is_object()) {
    Fail(path.string() + " has no quantization_config");
  }
  QuantizationConfig out;
  out.quant_method = q->member_str("quant_method");
  if (out.quant_method != "exl3") {
    Fail(path.string() + ": quant_method is \"" + out.quant_method +
         "\", not \"exl3\"");
  }
  out.version = q->member_str("version");
  out.bits = q->member_double("bits");
  out.head_bits = static_cast<int>(q->member_double("head_bits"));
  out.mtp_bits = static_cast<int>(q->member_double("mtp_bits"));
  out.codebook = q->member_str("codebook");
  out.out_scales = q->member_str("out_scales");
  return out;
}

const std::uint16_t* U16(const Tensor& tensor) {
  return reinterpret_cast<const std::uint16_t*>(tensor.data);
}

}  // namespace

std::int64_t Tensor::Elements() const noexcept {
  std::int64_t count = 1;
  for (const auto dim : shape) {
    if (dim == 0 || count > std::numeric_limits<std::int64_t>::max() / dim)
      return 0;
    count *= dim;
  }
  return count;
}

Checkpoint::Checkpoint() = default;
Checkpoint::Checkpoint(Checkpoint&&) noexcept = default;
Checkpoint& Checkpoint::operator=(Checkpoint&&) noexcept = default;
Checkpoint::~Checkpoint() = default;

bool Checkpoint::IsExl3(const std::filesystem::path& directory) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(directory / "config.json", ec)) {
    return false;
  }
  try {
    const auto config = ReadJson(directory / "config.json");
    ParseConfig(config, directory / "config.json");
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

Checkpoint Checkpoint::Open(
    const std::filesystem::path& directory,
    std::span<const std::filesystem::path> supplementary) {
  Checkpoint checkpoint;
  checkpoint.config_ = ParseConfig(ReadJson(directory / "config.json"),
                                   directory / "config.json");

  std::set<std::filesystem::path> shard_names;
  std::map<std::string, std::string> indexed;
  const auto index_path = directory / "model.safetensors.index.json";
  std::error_code ec;
  if (std::filesystem::is_regular_file(index_path, ec)) {
    const auto index = ReadJson(index_path);
    const json::Value* map = index.find("weight_map");
    if (map == nullptr || !map->is_object()) {
      Fail(index_path.string() + " has no weight_map");
    }
    for (const auto& [name, file] : map->members()) {
      if (!file.is_string() || file.str().find('/') != std::string::npos ||
          file.str().find("..") != std::string::npos) {
        Fail("invalid shard name for " + name);
      }
      shard_names.insert(directory / file.str());
      indexed.emplace(name, file.str());
    }
  } else {
    shard_names.insert(directory / "model.safetensors");
  }

  for (const auto& path : supplementary) {
    if (!shard_names.insert(path).second)
      Fail("duplicate supplementary shard " + path.string());
  }
  for (const auto& path : shard_names) {
    auto shard = std::make_unique<Shard>();
    shard->path = path;
    std::string error;
    shard->file = platform::ReadOnlyMappedFile::Open(path, &error);
    if (!shard->file.valid() || shard->file.size() < 8) {
      Fail("cannot map " + path.string() + (error.empty() ? "" : ": " + error));
    }
    const std::uint8_t* base = shard->file.data();
    const std::size_t size = shard->file.size();
    std::uint64_t header_size = 0;
    std::memcpy(&header_size, base, 8);
    if (header_size == 0 || header_size > (64u << 20) ||
        header_size > size - 8) {
      Fail(path.string() + ": invalid safetensors header size");
    }
    const auto header = [&] {
      try {
        return json::parse(std::string_view(
            reinterpret_cast<const char*>(base) + 8, header_size));
      } catch (const std::exception& e) {
        Fail(path.string() + ": " + e.what());
      }
    }();
    if (!header.is_object()) {
      Fail(path.string() + ": safetensors header is not an object");
    }
    const std::size_t payload = size - 8 - header_size;
    const std::uint8_t* payload_base = base + 8 + header_size;
    for (const auto& [name, entry] : header.members()) {
      if (name == "__metadata__") {
        continue;
      }
      Tensor tensor;
      tensor.dtype_name = entry.member_str("dtype");
      tensor.dtype = ParseDType(tensor.dtype_name);
      const json::Value* shape = entry.find("shape");
      const json::Value* offsets = entry.find("data_offsets");
      if (shape == nullptr || !shape->is_array() || offsets == nullptr ||
          !offsets->is_array() || offsets->size() != 2) {
        Fail(path.string() + ": invalid entry " + name);
      }
      for (const auto& dim : shape->items()) {
        tensor.shape.push_back(Integer(dim, name + " shape"));
      }
      const auto begin = Integer(offsets->items()[0], name + " offset");
      const auto end = Integer(offsets->items()[1], name + " offset");
      if (begin > end || static_cast<std::uint64_t>(end) > payload) {
        Fail(path.string() + ": tensor " + name + " lies outside the payload");
      }
      tensor.data = payload_base + begin;
      tensor.bytes = static_cast<std::size_t>(end - begin);
      tensor.shard = static_cast<std::uint32_t>(checkpoint.shards_.size());
      tensor.file_offset = 8 + header_size + begin;
      const auto expected = indexed.find(name);
      if (expected != indexed.end() && expected->second != path.filename())
        Fail("tensor " + name + " is in the wrong indexed shard");
      if (tensor.Elements() == 0)
        Fail(path.string() + ": invalid/overflowed shape " + name);
      const std::size_t element = ElementBytes(tensor.dtype);
      if (element != 0 && static_cast<std::uint64_t>(tensor.Elements()) >
                              std::numeric_limits<std::size_t>::max() / element)
        Fail(path.string() + ": overflowed tensor bytes " + name);
      if (element != 0 &&
          tensor.bytes !=
              static_cast<std::size_t>(tensor.Elements()) * element) {
        Fail(path.string() + ": tensor " + name + " size does not match shape");
      }
      if (!checkpoint.tensors_.emplace(name, std::move(tensor)).second) {
        Fail("duplicate tensor " + name);
      }
    }
    checkpoint.shards_.push_back(std::move(shard));
  }
  for (const auto& [name, file] : indexed) {
    if (checkpoint.Find(name) == nullptr)
      Fail("indexed tensor " + name + " is missing from " + file);
  }
  return checkpoint;
}

std::vector<core::GgufMappedRegion> Checkpoint::GetMappedRegions() const {
  std::vector<core::GgufMappedRegion> regions;
  regions.reserve(shards_.size());
  for (const auto& shard : shards_)
    regions.push_back({shard->file.data(), shard->file.size(),
                       shard->file.PlatformHandle(), shard->path});
  return regions;
}

const Tensor* Checkpoint::Find(std::string_view name) const noexcept {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

bool Checkpoint::IsQuantized(std::string_view module) const noexcept {
  return Find(std::string(module) + ".trellis") != nullptr;
}

Linear Checkpoint::GetLinear(std::string_view module) const {
  const std::string m(module);
  const Tensor* trellis = Find(m + ".trellis");
  if (trellis == nullptr) {
    Fail(m + ": no trellis tensor");
  }
  if (trellis->dtype != DType::kI16 || trellis->shape.size() != 3) {
    Fail(m + ".trellis: expected I16 [k/16, n/16, 16K], found " +
         trellis->dtype_name);
  }
  const auto rate =
      BitRateFromTileWords(static_cast<std::size_t>(trellis->shape[2]));
  if (!rate) {
    Fail(m + ".trellis: unsupported tile width " +
         std::to_string(trellis->shape[2]));
  }
  Linear linear;
  linear.name = m;
  linear.rate = *rate;
  linear.in_features = static_cast<std::size_t>(trellis->shape[0]) * 16;
  linear.out_features = static_cast<std::size_t>(trellis->shape[1]) * 16;
  linear.trellis = U16(*trellis);

  const bool mcg = Find(m + ".mcg") != nullptr;
  const bool mul1 = Find(m + ".mul1") != nullptr;
  if (mcg && mul1) {
    Fail(m + ": both mcg and mul1 markers are present");
  }
  for (const auto* marker : {Find(m + ".mcg"), Find(m + ".mul1")}) {
    if (marker != nullptr && (marker->dtype != DType::kI32 ||
                              !marker->shape.empty() || marker->bytes != 4))
      Fail(m + ": codebook marker must be an I32 scalar");
  }
  linear.codebook =
      mul1 ? Codebook::kMul1 : (mcg ? Codebook::kMcg : Codebook::k3Inst);
  if (!Supports(linear.codebook, linear.rate)) {
    Fail(m + ": half-integer bit rates require the mul1 codebook");
  }
  if (linear.in_features % kHadamardSize != 0 ||
      linear.out_features % kHadamardSize != 0) {
    Fail(m + ": in and out features must be multiples of 128");
  }

  auto vector = [&](const char* suffix,
                    std::size_t length) -> const std::uint16_t* {
    const Tensor* t = Find(m + suffix);
    if (t == nullptr) {
      return nullptr;
    }
    if (t->dtype != DType::kF16 || t->shape.size() != 1 ||
        static_cast<std::size_t>(t->shape[0]) != length) {
      Fail(m + suffix + ": expected F16 [" + std::to_string(length) + "]");
    }
    return U16(*t);
  };
  linear.suh = vector(".suh", linear.in_features);
  linear.svh = vector(".svh", linear.out_features);
  linear.bias = vector(".bias", linear.out_features);
  if (linear.suh == nullptr || linear.svh == nullptr) {
    if (Find(m + ".su") != nullptr || Find(m + ".sv") != nullptr) {
      Fail(m +
           ": packed su/sv sign bitfields are not supported (suh/svh only)");
    }
    Fail(m + ": suh or svh is missing");
  }
  return linear;
}

}  // namespace gufo::quant::exl3
