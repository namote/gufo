// Native row codec and metadata ownership checks need no GPU or model.
#include <algorithm>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "src/core/gguf_identity.hpp"
#include "src/core/json.hpp"
#include "src/core/quant/exl3/format.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/native_metadata.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "tests/support/temporary_directory.hpp"

namespace q = gufo::models::qwen38_flash_next;
namespace ex = gufo::quant::exl3;
namespace core = gufo::core;
namespace {
void Check(bool value, const std::string& message) {
  if (!value)
    throw std::runtime_error(message);
}
unsigned Symbol(unsigned row, unsigned col, unsigned bits) {
  return (row * 73 + col * 31 + col * col * 17) & ((1U << bits) - 1);
}
float Reference(unsigned row, unsigned col, unsigned bits) {
  // Independent bit-at-a-time cyclic register; no production unpacker/codebook.
  unsigned state = 0;
  for (unsigned bit = 0; bit < 16; ++bit) {
    const unsigned symbol = Symbol(row, (col + 160 - bit / bits) % 160, bits);
    state |= ((symbol >> (bit % bits)) & 1) << bit;
  }
  const std::uint32_t product = state * 0x83DCD12DU;
  unsigned sum = 1024;
  for (unsigned byte = 0; byte < 4; ++byte)
    sum += (product >> (8 * byte)) & 255;
  const float code = ex::HalfToDouble(ex::HalfFromDouble(
      sum * ex::HalfToDouble(0x1eee) + ex::HalfToDouble(0xc931)));
  const float value =
      code * (row % 2 ? -2.0F : 0.5F) + (row < 3 ? 0.25F : -0.5F);
  return ex::HalfToDouble(ex::HalfFromDouble(value));
}
q::Config Geometry() {
  q::Config c;
  c.ple_layer = 1;
  c.ple_head_dim = 160;
  c.ple_heads = 2;
  c.ple_ngram_size = 3;
  c.ple_rows = 6;
  c.ple_head_offsets = {0, 3};
  c.ple_head_vocab = {3, 3};
  c.ple_multipliers = {11, 13, 17};
  return c;
}
// A native table with 7 rows. With `second`, shard_1 is written to that
// separate part file (a split table); otherwise everything goes to `path`.
void Fixture(const std::filesystem::path& path, unsigned bits, bool shards,
             const std::filesystem::path& second = {},
             bool duplicate_shard = false) {
  const std::string prefix =
      "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";
  const std::string meta =
      "{\"__metadata__\":{\"format\":\"exl3_ngram_trellis\",\"version\":\"1\","
      "\"codebook\":\"mul1\",\"row_dim\":\"160\",\"rows\":\"7\",\"K\":\"" +
      std::to_string(bits) + "\"}";
  struct Part {
    std::string header;
    std::vector<char> payload;
  };
  Part parts[2] = {{meta, {}}, {meta, {}}};
  const auto add = [&](int part, const std::string& name, const char* type,
                       std::string shape, const auto& data) {
    auto& [header, payload] = parts[part];
    const auto begin = payload.size();
    const auto* bytes = reinterpret_cast<const char*>(data.data());
    payload.insert(payload.end(), bytes, bytes + data.size() * sizeof(data[0]));
    header += ",\"" + prefix + name + "\":{\"dtype\":\"" + type +
              "\",\"shape\":" + shape + ",\"data_offsets\":[" +
              std::to_string(begin) + "," + std::to_string(payload.size()) +
              "]}";
  };
  std::vector<std::uint16_t> bias(320, 0x3400);
  std::fill(bias.begin() + 160, bias.end(), 0xb800);
  add(0, ".head_bias", "F16", "[2,160]", bias);
  add(0, ".head_offsets", "I64", "[2]", std::vector<std::int64_t>{0, 3});
  add(0, ".head_vocab_sizes", "I64", "[2]", std::vector<std::int64_t>{3, 3});
  add(0, ".layer_multipliers", "I64", "[3]",
      std::vector<std::int64_t>{11, 13, 17});
  const unsigned width = 1 + 10 * bits;
  const auto rows = [&](unsigned begin, unsigned end) {
    std::vector<std::uint16_t> data((end - begin) * width);
    for (unsigned row = begin; row < end; ++row) {
      auto* dst = data.data() + (row - begin) * width;
      dst[0] = row % 2 ? 0xc000 : 0x3800;
      for (unsigned col = 0; col < 160; ++col)
        for (unsigned bit = 0; bit < bits; ++bit) {
          const unsigned index = col * bits + bit;
          dst[1 + index / 16] |= ((Symbol(row, col, bits) >> bit) & 1)
                                 << (index % 16);
        }
    }
    return data;
  };
  if (shards || !second.empty()) {
    // Shard and hash-head boundaries deliberately differ.
    add(0, ".shard_0.trellis", "I16", "[2," + std::to_string(width) + "]",
        rows(0, 2));
    add(second.empty() ? 0 : 1, ".shard_1.trellis", "I16",
        "[5," + std::to_string(width) + "]", rows(2, 7));
    if (duplicate_shard)
      add(1, ".shard_0.trellis", "I16", "[2," + std::to_string(width) + "]",
          rows(0, 2));
  } else
    add(0, ".trellis", "I16", "[7," + std::to_string(width) + "]", rows(0, 7));
  const auto write = [](const std::filesystem::path& file, Part& part) {
    part.header += "}";
    std::ofstream out(file, std::ios::binary);
    const std::uint64_t size = part.header.size();
    out.write(reinterpret_cast<const char*>(&size), 8);
    out << part.header;
    out.write(part.payload.data(), part.payload.size());
    Check(out.good(), "write native fixture");
  };
  write(path, parts[0]);
  if (!second.empty())
    write(second, parts[1]);
}
// Split tables: part discovery and a gather across both parts.
void SplitRows() {
  TemporaryDirectory temp("native-ngram-split");
  const auto dir = temp.path();
  const auto first = dir / "ngram_embedding-00001-of-00002.safetensors";
  const auto second = dir / "ngram_embedding-00002-of-00002.safetensors";
  for (unsigned bits : {3U, 8U}) {
    Fixture(first, bits, true, second);
    const auto files = q::Exl3Ngram::PackFiles(dir);
    Check(files.size() == 2 && files[0] == first && files[1] == second,
          "split table parts discovered in order");
    const auto layout = q::Exl3Ngram::Open(files);
    Check(layout.segments.size() == 2 && layout.segments[0].file == 0 &&
              layout.segments[1].file == 1,
          "split shards keep their part");
    std::string error;
    auto table = q::NgramTable::OpenExl3(files, Geometry(), &error);
    Check(table != nullptr, error);
    const std::vector<std::uint32_t> ids{0, 1, 2, 3, 5, 1, 4};
    std::vector<float> out(ids.size() * 160);
    Check(table->Read(ids, out), "split gather");
    for (std::size_t i = 0; i < ids.size(); ++i)
      for (unsigned col = 0; col < 160; ++col)
        Check(out[i * 160 + col] == Reference(ids[i], col, bits),
              "split row exact oracle");
  }
  const auto rejects = [&](const char* why, const char* reason) {
    std::string message;
    try {
      (void)q::Exl3Ngram::Open(q::Exl3Ngram::PackFiles(dir));
    } catch (const std::exception& e) {
      message = e.what();
    }
    Check(message.find(reason) != std::string::npos,
          std::string(why) + ": " + message);
  };
  std::filesystem::remove(second);
  rejects("missing table part rejected", "missing table parts");
  Fixture(first, 3, true, second);
  Fixture(dir / "ngram_embedding.safetensors", 3, false);
  rejects("single table beside parts rejected", "both");
  std::filesystem::remove(dir / "ngram_embedding.safetensors");
  // The same row shard in both parts.
  Fixture(first, 3, true, second, true);
  rejects("duplicated shards rejected", "several table parts");
}
void Rows() {
  TemporaryDirectory temp("native-ngram");
  const auto path = temp.path() / "ngram.safetensors";
  for (unsigned bits = 1; bits <= 8; ++bits) {
    for (bool shards : {false, true}) {
      Fixture(path, bits, shards);
      std::string error;
      auto c = Geometry();
      auto table = q::NgramTable::OpenExl3(
          std::span<const std::filesystem::path>(&path, 1), c, &error);
      Check(table != nullptr, error);
      std::vector<std::uint32_t> ids{0, 2, 3, 5, 1, 5, 3};
      for (unsigned pass = 0; pass < 3; ++pass) {
        if (pass == 2)
          ids.resize(2049, 2);
        std::vector<float> out(ids.size() * 160 + 16, -999);
        Check(table->Read(ids, std::span(out).first(ids.size() * 160)),
              "native gather");
        for (std::size_t i = 0; i < ids.size(); ++i)
          for (unsigned col = 0; col < 160; ++col)
            Check(out[i * 160 + col] == Reference(ids[i], col, bits),
                  "native row exact oracle");
        Check(std::all_of(out.end() - 16, out.end(),
                          [](float x) { return x == -999; }),
              "native output guards");
      }
      std::array<std::uint32_t, 1> padded{6};
      std::array<float, 160> output{};
      Check(!table->Read(padded, output), "padded rows are inaccessible");
      c.ple_multipliers[0]++;
      Check(!q::NgramTable::OpenExl3(
                std::span<const std::filesystem::path>(&path, 1), c, &error),
            "hash mismatch rejected");
      table.reset();
      std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
      Check(!q::NgramTable::OpenExl3(
                std::span<const std::filesystem::path>(&path, 1), Geometry(),
                &error),
            "truncated native tensor rejected");
    }
  }
}
void Ownership() {
  using V = core::GgufMetadataValue;
  using T = core::GgufValueType;
  std::unique_ptr<core::GgufReader> reader;
  {
    std::string key = "x", text = "short", long_text(200, 'z');
    reader = core::GgufReader::FromMetadata(
        {{key, V{T::kString, std::string_view(text)}},
         {"array",
          V{T::kArray, std::vector<std::string_view>{text, long_text}}}});
  }
  core::GgufReader moved(std::move(*reader));
  reader.reset();
  Check(moved.GetMetadataString("x") == "short",
        "owned short text survives move");
  auto other = core::GgufReader::FromMetadata({});
  *other = std::move(moved);
  auto values = other->GetMetadataStringArray("array");
  Check(values.size() == 2 && values[0] == "short" &&
            values[1] == std::string(200, 'z'),
        "owned array survives move assignment");
  Check(other->GetMappedRegions().empty(),
        "native metadata has no fake GGUF image");
  bool rejected = false;
  try {
    (void)core::GgufReader::FromMetadata(
        {{"x", V{T::kBool, true}}, {"x", V{T::kBool, false}}});
  } catch (const std::exception&) {
    rejected = true;
  }
  Check(rejected, "duplicate metadata rejected");
}
void Identity() {
  TemporaryDirectory temp("native-identity");
  const auto dir = temp.path();
  std::ofstream(dir / "a") << "metadata";
  std::ofstream(dir / "b") << "weights";
  const auto first = core::FileSetIdentityHex(dir, {"a", "b"});
  Check(first == core::FileSetIdentityHex(dir, {"a", "b"}),
        "cached native identity");
  std::filesystem::create_directory(dir / "copy");
  for (const auto name : {"a", "b"})
    std::filesystem::copy_file(dir / name, dir / "copy" / name);
  Check(first == core::FileSetIdentityHex(dir / "copy", {"a", "b"}),
        "relocated artifact identity");
  std::ofstream(dir / "b") << "changed";
  Check(first != core::FileSetIdentityHex(dir, {"a", "b"}),
        "same-size weight change invalidates identity");
  std::ofstream(dir / "hc_q8.json") << "bound-calibrated-HC-artifact";
  const auto calibrated =
      core::FileSetIdentityHex(dir, {"a", "b", "hc_q8.json"});
  Check(calibrated != core::FileSetIdentityHex(dir, {"a", "b"}),
        "calibrated HC separates continuation identity");
  std::ofstream(dir / "hc_q8.json") << "different-calibrated-HC-set";
  Check(calibrated != core::FileSetIdentityHex(dir, {"a", "b", "hc_q8.json"}),
        "changed HC manifest invalidates continuation identity");
  for (const auto& names :
       {std::vector<std::string>{}, {"b", "a"}, {"a", "a"}, {"../a"}}) {
    bool rejected = false;
    try {
      (void)core::FileSetIdentityHex(dir, names);
    } catch (const std::exception&) {
      rejected = true;
    }
    Check(rejected, "invalid native file list rejected");
  }
}
// A tensorless GGUF holding only string keys (an old metadata sidecar).
void WriteStringGguf(
    const std::filesystem::path& path,
    std::initializer_list<std::pair<std::string_view, std::string_view>> kvs) {
  std::ofstream out(path, std::ios::binary);
  const auto u32 = [&](std::uint32_t v) {
    out.write(reinterpret_cast<const char*>(&v), 4);
  };
  const auto u64 = [&](std::uint64_t v) {
    out.write(reinterpret_cast<const char*>(&v), 8);
  };
  const auto str = [&](std::string_view v) {
    u64(v.size());
    out.write(v.data(), static_cast<std::streamsize>(v.size()));
  };
  out.write("GGUF", 4);
  u32(3);
  u64(0);
  u64(kvs.size());
  for (const auto& [key, value] : kvs) {
    str(key);
    u32(8);  // string
    str(value);
  }
  // The (empty) data section starts at the next 32-byte boundary.
  const auto size = static_cast<std::size_t>(out.tellp());
  out.write(std::string((32 - size % 32) % 32, '\0').data(),
            static_cast<std::streamsize>((32 - size % 32) % 32));
}
void SidecarRejected() {
  TemporaryDirectory temp("native-sidecar");
  const auto file = temp.path() / "gufo.gguf";
  WriteStringGguf(file, {{"general.architecture", "qwen4exp"},
                         {"gufo.exl3.checkpoint", "pack"}});
  std::string error;
  const bool opened = q::OpenArtifactMetadata(file, &error) != nullptr;
  Check(!opened && error.find("sidecar") != std::string::npos,
        "EXL3 metadata sidecar rejected: " + error);
}
void MetadataRejections() {
  TemporaryDirectory temp("native-metadata-contract");
  const auto dir = temp.path();
  std::ofstream(dir / "config.json")
      << R"({"quantization_config":{"quant_method":"exl3"},"text_config":{"model_type":"qwen4_exp_text"}})";
  std::ofstream(dir / "model.safetensors.index.json")
      << R"({"weight_map":{"x":"weights.safetensors"}})";
  for (const auto name : {"weights.safetensors", "ngram_embedding.safetensors",
                          "chat_template.jinja"})
    std::ofstream(dir / name) << "x";
  std::ofstream(dir / "tokenizer_config.json") << "{}";
  const std::string tokenizer =
      R"FIXTURE({"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":248044,"content":"<|endoftext|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":{"type":"NFC"},"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{},"merges":[]}})FIXTURE";
  const auto reject = [&](std::string text, const std::string& reason) {
    std::ofstream(dir / "tokenizer.json") << text;
    std::string error;
    Check(!q::OpenArtifactMetadata(dir, &error) &&
              error.find(reason) != std::string::npos,
          "native rejection: " + error);
  };
  reject(
      tokenizer,
      "truncated header");  // Accepted tokenizer, deliberately invalid weights.
  const auto altered = [&](std::string from, std::string to) {
    auto text = tokenizer;
    const auto at = text.find(from);
    Check(at != std::string::npos, "contract mutation exists");
    text.replace(at, from.size(), to);
    return text;
  };
  reject(altered("NFC", "NFKC"), "tokenizer contract");
  reject(altered("Isolated", "Removed"), "tokenizer contract");
  reject(altered("\"add_prefix_space\":false", "\"add_prefix_space\":true"),
         "tokenizer contract");
  reject(altered("\"ignore_merges\":false", "\"ignore_merges\":true"),
         "tokenizer contract");
  reject(altered("\"lstrip\":false", "\"lstrip\":true"), "tokenizer contract");
  std::ofstream(dir / "model.safetensors.index.json")
      << R"({"weight_map":{"x":"../escape"}})";
  reject(tokenizer, "invalid native shard name");
}
void Pack(const std::filesystem::path& path) {
  std::string error;
  auto reader = q::OpenArtifactMetadata(path, &error);
  Check(reader != nullptr, error);
  Check(reader->GetMappedRegions().empty(), "pack has no GGUF mappings");
  auto config = q::Config::FromGguf(*reader, true, &error);
  Check(config.has_value(), error);
  auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  Check(tokenizer != nullptr, error);
  Check(tokenizer->GetEosTokenId() == 248046 && config->ple_eos_token == 248044,
        "generation and PLE EOS remain distinct");
  Check(gufo::tokenization::QwenChatTemplate::ValidateGgufTemplate(*reader,
                                                                   &error),
        error);
  const std::string sample = "Hello, 세계! e\xcc\x81 123\n";
  Check(tokenizer->Decode(tokenizer->Encode(sample)) == "Hello, 세계! é 123\n",
        "native NFC multilingual roundtrip");
  const auto types = std::get<std::vector<std::uint64_t>>(
      reader->FindMetadata("tokenizer.ggml.token_type")->value);
  const auto tokens = reader->GetMetadataStringArray("tokenizer.ggml.tokens");
  for (std::size_t i = 0; i < types.size(); ++i)
    if (types[i] == 3 || types[i] == 4)
      Check(tokenizer->Encode(tokens[i]) ==
                std::vector<gufo::tokenization::TokenId>{
                    static_cast<std::uint32_t>(i)},
            "native added token is atomic");
  const auto files = q::Exl3Ngram::PackFiles(path);
  auto layout = q::Exl3Ngram::Open(files);
  auto table = q::NgramTable::OpenExl3(files, *config, &error);
  Check(table != nullptr, error);
  std::vector<std::uint32_t> ids{
      0, 1, static_cast<std::uint32_t>(config->ple_rows - 1)};
  for (const auto offset : layout.offsets)
    if (offset) {
      ids.push_back(offset - 1);
      ids.push_back(offset);
    }
  for (const auto& segment : layout.segments)
    if (segment.first && segment.first < config->ple_rows) {
      ids.push_back(segment.first - 1);
      ids.push_back(segment.first);
    }
  std::vector<float> values(ids.size() * 160);
  Check(table->Read(ids, values), "real native boundary rows");
  for (std::size_t r = 0; r < ids.size(); ++r) {
    const auto row = ids[r];
    const auto segment = std::prev(std::upper_bound(
        layout.segments.begin(), layout.segments.end(), row,
        [](auto id, const auto& part) { return id < part.first; }));
    std::ifstream raw_file(files[segment->file], std::ios::binary);
    std::vector<std::uint16_t> packed(1 + 10 * layout.bits);
    raw_file.seekg(segment->offset +
                   (row - segment->first) * packed.size() * 2);
    raw_file.read(reinterpret_cast<char*>(packed.data()), packed.size() * 2);
    Check(raw_file.good(), "read independent real packed row");
    const auto head =
        std::upper_bound(layout.offsets.begin(), layout.offsets.end(), row) -
        layout.offsets.begin() - 1;
    for (unsigned col = 0; col < 160; ++col) {
      unsigned state = 0;
      for (unsigned bit = 0; bit < 16; ++bit) {
        const unsigned index =
            ((col + 160 - bit / layout.bits) % 160) * layout.bits +
            bit % layout.bits;
        state |= ((packed[1 + index / 16] >> (index % 16)) & 1) << bit;
      }
      const std::uint32_t product = state * 0x83DCD12DU;
      unsigned sum = 1024;
      for (unsigned b = 0; b < 4; ++b)
        sum += (product >> (8 * b)) & 255;
      const float code = ex::HalfToDouble(ex::HalfFromDouble(
          sum * ex::HalfToDouble(0x1eee) + ex::HalfToDouble(0xc931)));
      const float unrounded =
          code * static_cast<float>(ex::HalfToDouble(packed[0])) +
          layout.bias[head * 160 + col];
      const float expected = ex::HalfToDouble(ex::HalfFromDouble(unrounded));
      Check(values[r * 160 + col] == expected,
            "real native row matches bit oracle and FP16 boundary");
    }
  }
  std::cout << "PACK PASS " << path << "\n";
}
}  // namespace
int main(int argc, char** argv) {
  try {
    MetadataRejections();
    SidecarRejected();
    Ownership();
    std::cerr << "ownership PASS\n";
    Identity();
    std::cerr << "identity PASS\n";
    Rows();
    SplitRows();
    std::cerr << "split PASS\n";
    std::cerr << "rows PASS\n";
    for (int i = 1; i < argc; ++i)
      Pack(argv[i]);
    std::cout << "PASS native codec K1..K8, sharding, cache, bounds, metadata, "
                 "identity\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
