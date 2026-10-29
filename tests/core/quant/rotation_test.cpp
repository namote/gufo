// prism.hadamard rotation metadata checks (src/core/quant/rotation.hpp):
// upstream llama.cpp refusals with their line numbers, gufo's subset rules,
// the retired v1 format, and the CPU rotation reference
// (tests/core/quant/rotation_reference.hpp) against the library RotateRow.

#include "src/core/quant/rotation.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "tests/core/quant/rotation_reference.hpp"
#include "tests/core/quant/rotation_test_gguf.hpp"

namespace {

using gufo::quant::rotation::GroupTable;
using gufo::quant::rotation::ReadRotation;
using gufo::test::rotation::GgufImage;

int failures = 0;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

std::vector<std::int32_t> Signs(std::size_t k, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<std::int32_t> signs(k);
  for (auto& sign : signs) {
    sign = (rng() & 1U) != 0 ? -1 : 1;
  }
  return signs;
}

// Two layers with groups "in" (ffn_gate, ffn_up: K = 256) and "down"
// (ffn_down: K = 384), plus token_embd and output (K = 256, group "head").
// Every prism key is set to a valid untied, explicit, version 1 value and
// can be overridden or removed per test.
struct Fixture {
  bool with_output = true;
  std::vector<std::string> weights = {
      "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
      "blk.1.ffn_gate.weight", "blk.1.ffn_up.weight", "blk.1.ffn_down.weight"};
  std::map<std::string, std::function<void(GgufImage&)>> keys;
  std::map<std::int32_t, std::vector<std::int32_t>> signs = {
      {256, Signs(256, 1)}, {384, Signs(384, 2)}};
  std::string architecture = "qwen35";

  Fixture() {
    Set("prism.hadamard.version",
        [](GgufImage& g) { g.AddUint32("prism.hadamard.version", 1); });
    Set("prism.hadamard.block_size",
        [](GgufImage& g) { g.AddUint32("prism.hadamard.block_size", 128); });
    Set("prism.hadamard.transform", [](GgufImage& g) {
      g.AddString("prism.hadamard.transform",
                  "normalized-sylvester-walsh-hadamard");
    });
    Set("prism.hadamard.axis", [](GgufImage& g) {
      g.AddString("prism.hadamard.axis", "input-last-dimension");
    });
    Set("prism.hadamard.sign_mode", [](GgufImage& g) {
      g.AddString("prism.hadamard.sign_mode", "explicit");
    });
    Set("prism.hadamard.sign_widths", [this](GgufImage& g) {
      std::vector<std::int32_t> widths;
      for (const auto& [width, unused] : signs) {
        widths.push_back(width);
      }
      g.AddInt32Array("prism.hadamard.sign_widths", widths);
    });
    Set("prism.hadamard.sign_values", [this](GgufImage& g) {
      std::vector<std::int32_t> values;
      for (const auto& [unused, vector] : signs) {
        values.insert(values.end(), vector.begin(), vector.end());
      }
      g.AddInt32Array("prism.hadamard.sign_values", values);
    });
    Set("prism.hadamard.weight_names", [this](GgufImage& g) {
      g.AddStringArray("prism.hadamard.weight_names", weights);
    });
  }

  void Set(const std::string& key, std::function<void(GgufImage&)> add) {
    keys[key] = std::move(add);
  }

  [[nodiscard]] std::vector<std::uint8_t> Build() const {
    GgufImage image;
    image.AddString("general.architecture", architecture);
    for (int layer = 0; layer < 2; ++layer) {
      const std::string p = "blk." + std::to_string(layer) + ".";
      image.AddTensorF32(p + "ffn_gate.weight", {256, 4});
      image.AddTensorF32(p + "ffn_up.weight", {256, 4});
      image.AddTensorF32(p + "ffn_down.weight", {384, 4});
    }
    image.AddTensorF32("token_embd.weight", {256, 8});
    if (with_output) {
      image.AddTensorF32("output.weight", {256, 8});
    }
    for (const auto& [key, add] : keys) {
      add(image);
    }
    return image.Build();
  }
};

GroupTable Table() {
  GroupTable table;
  for (int layer = 0; layer < 2; ++layer) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    const std::string l = std::to_string(layer);
    table.instances.push_back(
        {"in", "in:" + l, {p + "ffn_gate.weight", p + "ffn_up.weight"}});
    table.instances.push_back({"down", "down:" + l, {p + "ffn_down.weight"}});
  }
  table.instances.push_back({"head", "head", {"output.weight"}});
  return table;
}

struct Outcome {
  bool ok{false};
  std::string error;
  gufo::quant::rotation::RotationInfo info;
};

Outcome Read(const std::vector<std::uint8_t>& bytes,
             const GroupTable& table = Table()) {
  std::string error;
  auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
  if (reader == nullptr) {
    return {.ok = false, .error = "reader: " + error, .info = {}};
  }
  auto info = ReadRotation(*reader, table, &error);
  if (!info.has_value()) {
    return {.ok = false, .error = error, .info = {}};
  }
  return {.ok = true, .error = {}, .info = std::move(*info)};
}

void ExpectRefusal(const Fixture& fixture, std::string_view fragment,
                   std::string_view label, const GroupTable& table = Table()) {
  const auto outcome = Read(fixture.Build(), table);
  if (outcome.ok) {
    Expect(false, std::string(label) + ": accepted");
    return;
  }
  Expect(outcome.error.find(fragment) != std::string::npos,
         std::string(label) + ": message '" + outcome.error + "' lacks '" +
             std::string(fragment) + "'");
  Expect(outcome.error.rfind("Rotated GGUF refused: ", 0) == 0,
         std::string(label) + ": message prefix");
}

// Upstream refusal: the message must carry the upstream line.
void ExpectUpstream(const Fixture& fixture, int line, std::string_view label) {
  ExpectRefusal(
      fixture,
      "(llama.cpp 8ab29fae src/llama-model.cpp:" + std::to_string(line) + ")",
      label);
}

void TestUnrotated() {
  Fixture fixture;
  fixture.keys.clear();
  const auto outcome = Read(fixture.Build());
  Expect(outcome.ok && !outcome.info.Any(), "unrotated file has no rotation");

  // Converter provenance keys are not rotation metadata.
  fixture.Set("gufo.rotation.source_converter", [](GgufImage& g) {
    g.AddString("gufo.rotation.source_converter", "rotquant 2");
  });
  const auto provenance = Read(fixture.Build());
  Expect(provenance.ok && !provenance.info.Any(),
         "source_* keys are not rotation markers");
}

void TestV1Refused() {
  {
    Fixture fixture;
    fixture.keys.clear();
    fixture.Set("gufo.rotation.version",
                [](GgufImage& g) { g.AddUint32("gufo.rotation.version", 1); });
    ExpectRefusal(fixture, "v1 rotation format", "v1 marker key");
  }
  {
    Fixture fixture;
    fixture.Set("v1 tensor", [](GgufImage& g) {
      g.AddTensorF16("blk.0.ffn_gate.weight.rot",
                     std::vector<float>(256, 1.0F));
    });
    ExpectRefusal(fixture, "reconvert the model with tools/rotquant",
                  "v1 .rot tensor");
  }
}

void TestAccepted() {
  Fixture fixture;
  fixture.weights.push_back("output.weight");
  const auto outcome = Read(fixture.Build());
  Expect(outcome.ok, "prism file accepted: " + outcome.error);
  if (!outcome.ok) {
    return;
  }
  const auto& info = outcome.info;
  Expect(info.Any(), "rotation present");
  Expect(info.rotated_groups == std::vector<std::string>{"in", "down", "head"},
         "rotated groups in table order: " + info.GroupList());
  Expect(info.widths.size() == 2, "one sign vector per width");
  for (const auto& [width, values] : fixture.signs) {
    const auto* signs = info.ForWidth(static_cast<std::size_t>(width));
    Expect(signs != nullptr &&
               signs->bits.size() == static_cast<std::size_t>(width) / 32,
           "width present and packed");
    if (signs == nullptr) {
      continue;
    }
    bool same = true;
    for (std::size_t i = 0; i < values.size(); ++i) {
      same = same &&
             gufo::test::rotation::SignBit(signs->bits, i) == (values[i] < 0);
    }
    Expect(same, "packed bits match sign_values");
  }

  // One complete group is enough; its width alone is declared.
  Fixture partial;
  partial.weights = {"blk.0.ffn_down.weight", "blk.1.ffn_down.weight"};
  partial.signs.erase(256);
  const auto only_down = Read(partial.Build());
  Expect(only_down.ok &&
             only_down.info.rotated_groups == std::vector<std::string>{"down"},
         "one complete group rotated: " + only_down.error);
}

void TestUpstreamRules() {
  {
    Fixture f;
    f.keys.erase("prism.hadamard.version");
    f.Set("prism.hadamard.tied_output",
          [](GgufImage& g) { g.AddBool("prism.hadamard.tied_output", true); });
    ExpectUpstream(f, 2224, "tied_output without version");
  }
  {
    Fixture f;
    f.Set("prism.hadamard.version",
          [](GgufImage& g) { g.AddUint32("prism.hadamard.version", 3); });
    ExpectUpstream(f, 2078, "version 3");
  }
  {
    Fixture f;
    f.Set("prism.hadamard.version",
          [](GgufImage& g) { g.AddUint32("prism.hadamard.version", 2); });
    ExpectUpstream(f, 2082, "version 2 untied");
  }
  {
    Fixture f;
    f.Set("prism.hadamard.version",
          [](GgufImage& g) { g.AddUint32("prism.hadamard.version", 2); });
    f.Set("prism.hadamard.tied_output",
          [](GgufImage& g) { g.AddBool("prism.hadamard.tied_output", true); });
    ExpectUpstream(f, 2085, "tied with output.weight");
  }
  {
    Fixture f;
    f.keys.erase("prism.hadamard.block_size");
    ExpectRefusal(f, "key not found in model: prism.hadamard.block_size",
                  "missing block size");
  }
  {
    Fixture f;
    f.Set("prism.hadamard.version",
          [](GgufImage& g) { g.AddString("prism.hadamard.version", "1"); });
    ExpectRefusal(f, "prism.hadamard.version has the wrong GGUF type",
                  "string version");
  }
  {
    Fixture f;
    f.Set("prism.hadamard.block_size",
          [](GgufImage& g) { g.AddUint32("prism.hadamard.block_size", 96); });
    ExpectUpstream(f, 2101, "block 96");
  }
  {
    Fixture f;
    f.Set("prism.hadamard.transform",
          [](GgufImage& g) { g.AddString("prism.hadamard.transform", "dct"); });
    ExpectUpstream(f, 2104, "transform");
  }
  {
    Fixture f;
    f.Set("prism.hadamard.axis",
          [](GgufImage& g) { g.AddString("prism.hadamard.axis", "output"); });
    ExpectUpstream(f, 2107, "axis");
  }
  {
    Fixture f;
    f.Set("prism.hadamard.sign_mode", [](GgufImage& g) {
      g.AddString("prism.hadamard.sign_mode", "random");
    });
    ExpectUpstream(f, 2110, "sign mode");
  }
  {
    Fixture f;
    f.weights.clear();
    ExpectUpstream(f, 2113, "empty weight names");
  }
  {
    Fixture f;
    f.signs.clear();
    ExpectUpstream(f, 2123, "empty sign widths");
  }
  {
    Fixture f;
    f.Set("prism.hadamard.sign_widths", [](GgufImage& g) {
      g.AddInt32Array("prism.hadamard.sign_widths", {256, 200});
    });
    ExpectUpstream(f, 2128, "width not a block multiple");
  }
  {
    Fixture f;
    f.Set("prism.hadamard.sign_widths", [](GgufImage& g) {
      g.AddInt32Array("prism.hadamard.sign_widths", {256, 256});
    });
    f.Set("prism.hadamard.sign_values", [](GgufImage& g) {
      g.AddInt32Array("prism.hadamard.sign_values",
                      std::vector<std::int32_t>(512, 1));
    });
    ExpectUpstream(f, 2131, "duplicate width");
  }
  {
    Fixture f;
    f.signs[256][9] = 2;
    ExpectUpstream(f, 2137, "sign value 2");
  }
  {
    Fixture f;
    f.Set("prism.hadamard.sign_values", [](GgufImage& g) {
      g.AddInt32Array("prism.hadamard.sign_values",
                      std::vector<std::int32_t>(256 + 384 + 1, 1));
    });
    ExpectUpstream(f, 2143, "sign values too long");
  }
  {
    Fixture f;
    f.architecture = "gemma3";
    ExpectUpstream(f, 2159, "architecture");
  }
  {
    Fixture f;
    f.weights.push_back("blk.0.ssm_alpha.weight");
    ExpectUpstream(f, 2199, "ssm_alpha is not foldable upstream");
  }
  {
    Fixture f;
    f.weights.push_back("blk.0.ffn_gate.weight");
    ExpectUpstream(f, 2203, "duplicate weight name");
  }
  {
    Fixture f;
    f.Set("prism.hadamard.inverse_weight_names", [](GgufImage& g) {
      g.AddStringArray("prism.hadamard.inverse_weight_names",
                       {"output.weight"});
    });
    ExpectUpstream(f, 2213, "inverse name other than token_embd");
  }
  {
    Fixture f;
    f.Set("prism.hadamard.inverse_weight_names", [](GgufImage& g) {
      g.AddStringArray("prism.hadamard.inverse_weight_names",
                       {"token_embd.weight", "token_embd.weight"});
    });
    ExpectUpstream(f, 2217, "duplicate inverse name");
  }
  {
    Fixture f;
    f.with_output = false;
    f.Set("prism.hadamard.version",
          [](GgufImage& g) { g.AddUint32("prism.hadamard.version", 2); });
    f.Set("prism.hadamard.tied_output",
          [](GgufImage& g) { g.AddBool("prism.hadamard.tied_output", true); });
    ExpectUpstream(f, 2228, "tied without latent embedding");
  }
  {
    Fixture f;
    f.with_output = false;
    f.Set("prism.hadamard.inverse_weight_names", [](GgufImage& g) {
      g.AddStringArray("prism.hadamard.inverse_weight_names",
                       {"token_embd.weight"});
    });
    ExpectUpstream(f, 2232, "untied inverse table without output");
  }
  {
    Fixture f;
    f.weights.push_back("blk.7.ffn_up.weight");
    ExpectUpstream(f, 2271, "weight not in the file");
  }
  {
    Fixture f;
    f.Set("prism.hadamard.block_size",
          [](GgufImage& g) { g.AddUint32("prism.hadamard.block_size", 512); });
    f.Set("prism.hadamard.sign_widths", [](GgufImage& g) {
      g.AddInt32Array("prism.hadamard.sign_widths", {512});
    });
    f.Set("prism.hadamard.sign_values", [](GgufImage& g) {
      g.AddInt32Array("prism.hadamard.sign_values",
                      std::vector<std::int32_t>(512, 1));
    });
    ExpectUpstream(f, 2274, "block does not divide K");
  }
  {
    Fixture f;
    f.signs.erase(384);
    ExpectUpstream(f, 2348, "no sign vector for a width");
  }
}

void TestGufoSubset() {
  {
    Fixture f;
    f.with_output = false;
    f.Set("prism.hadamard.version",
          [](GgufImage& g) { g.AddUint32("prism.hadamard.version", 2); });
    f.Set("prism.hadamard.tied_output",
          [](GgufImage& g) { g.AddBool("prism.hadamard.tied_output", true); });
    f.Set("prism.hadamard.inverse_weight_names", [](GgufImage& g) {
      g.AddStringArray("prism.hadamard.inverse_weight_names",
                       {"token_embd.weight"});
    });
    ExpectRefusal(f, "only untied prism.hadamard version 1", "tied file");
  }
  {
    Fixture f;
    f.Set("prism.hadamard.block_size",
          [](GgufImage& g) { g.AddUint32("prism.hadamard.block_size", 64); });
    ExpectRefusal(f, "unsupported prism.hadamard.block_size = 64", "block 64");
  }
  {
    Fixture f;
    f.Set("prism.hadamard.sign_mode", [](GgufImage& g) {
      g.AddString("prism.hadamard.sign_mode", "identity");
    });
    ExpectRefusal(f, "unsupported prism.hadamard.sign_mode = 'identity'",
                  "identity mode");
  }
  {
    Fixture f;
    f.Set("prism.hadamard.inverse_weight_names", [](GgufImage& g) {
      g.AddStringArray("prism.hadamard.inverse_weight_names",
                       {"token_embd.weight"});
    });
    ExpectRefusal(f, "prism.hadamard.inverse_weight_names is not supported",
                  "inverse table");
  }
  {
    Fixture f;
    f.Set("prism.hadamard.gdn_v_grouped", [](GgufImage& g) {
      g.AddBool("prism.hadamard.gdn_v_grouped", true);
    });
    ExpectRefusal(f, "prism.hadamard.gdn_v_grouped is not supported",
                  "GDN permutation");
  }
  {
    Fixture f;
    f.weights.push_back("output.weight");
    GroupTable table = Table();
    table.instances.pop_back();
    ExpectRefusal(f, "output.weight is rotated but its role has no rotation",
                  "weight outside the table", table);
  }
  {
    Fixture f;
    GroupTable table = Table();
    table.unsupported_groups.push_back("down");
    ExpectRefusal(f, "group down has no rotation path in this runtime",
                  "unsupported group", table);
  }
  {
    Fixture f;
    std::erase(f.weights, std::string("blk.1.ffn_up.weight"));
    ExpectRefusal(f, "group instance in:1 mixes rotated and unrotated",
                  "mixed instance");
  }
  {
    Fixture f;
    std::erase(f.weights, std::string("blk.1.ffn_down.weight"));
    ExpectRefusal(f, "group down is rotated in some layers only (1 of 2",
                  "partial layers");
  }
  {
    Fixture f;
    f.weights = {"blk.0.ffn_down.weight", "blk.1.ffn_down.weight"};
    ExpectRefusal(f, "lists a width no folded weight uses: 256",
                  "unused width");
  }
  {
    Fixture f;
    f.keys.erase("prism.hadamard.version");
    ExpectRefusal(f, "without prism.hadamard.version", "stray keys");
  }
}

void TestRotateRow() {
  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  for (const std::size_t k : {128U, 640U, 5120U, 17408U}) {
    std::vector<float> row(k);
    for (auto& value : row) {
      value = normal(rng);
    }
    row[3] = 300.0F;
    std::vector<std::uint32_t> bits(k / 32);
    for (auto& word : bits) {
      word = rng();
    }
    auto library = row;
    gufo::quant::rotation::RotateRow(library, bits);
    auto reference = row;
    gufo::test::rotation::RotateRow(reference, bits);
    Expect(
        std::memcmp(library.data(), reference.data(), k * sizeof(float)) == 0,
        "library RotateRow equals the test reference, K = " +
            std::to_string(k));
    const auto exact = gufo::test::rotation::RotateRowFp64(row, bits);
    double max_error = 0.0;
    double max_value = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
      max_error = std::max(max_error, std::abs(reference[i] - exact[i]));
      max_value = std::max(max_value, std::abs(exact[i]));
    }
    Expect(max_error <= 1e-5 * std::max(1.0, max_value),
           "FP32 reference matches the FP64 product, K = " + std::to_string(k));
  }
  Expect(gufo::hip::hadamard::kHadamard128Scale ==
             static_cast<float>(1.0 / std::sqrt(128.0)),
         "scale constant");
}

}  // namespace

int main() {
  TestUnrotated();
  TestV1Refused();
  TestAccepted();
  TestUpstreamRules();
  TestGufoSubset();
  TestRotateRow();
  if (failures != 0) {
    std::cerr << failures << " rotation check(s) failed\n";
    return 1;
  }
  std::cout << "rotation_test: all checks passed\n";
  return 0;
}
