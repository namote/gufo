// Rotated-GGUF marker checks (src/core/quant/rotation.hpp) and the CPU
// rotation reference (tests/core/quant/rotation_reference.hpp).

#include "src/core/quant/rotation.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "tests/core/quant/rotation_reference.hpp"
#include "tests/core/quant/rotation_test_gguf.hpp"

namespace {

using gufo::core::GgmlType;
using gufo::quant::rotation::GroupInstance;
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

std::vector<float> Signs(std::size_t k, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<float> signs(k);
  for (auto& sign : signs) {
    sign = (rng() & 1U) != 0 ? -1.0F : 1.0F;
  }
  return signs;
}

// Two layers, each with groups "in" (a, b: K = 256) and "down" (c: K = 384),
// plus an embedding "emb" outside every group.
struct Fixture {
  GgufImage image;
  std::map<std::string, std::vector<float>> signs;
  bool markers = true;

  Fixture() {
    for (int layer = 0; layer < 2; ++layer) {
      const std::string p = "blk." + std::to_string(layer) + ".";
      image.AddTensorF32(p + "a.weight", {256, 4});
      image.AddTensorF32(p + "b.weight", {256, 2});
      image.AddTensorF32(p + "c.weight", {384, 4});
    }
    image.AddTensorF32("emb.weight", {256, 8});
  }

  void RotateAll() {
    for (int layer = 0; layer < 2; ++layer) {
      const std::string p = "blk." + std::to_string(layer) + ".";
      const auto in = Signs(256, 10 + layer);
      signs[p + "a.weight"] = in;
      signs[p + "b.weight"] = in;
      signs[p + "c.weight"] = Signs(384, 20 + layer);
    }
  }

  void AddMarkers(GgufImage& out) const {
    out.AddUint32("gufo.rotation.version", 1);
    out.AddString("gufo.rotation.kind", "hadamard");
    out.AddUint32("gufo.rotation.block_size", 128);
    out.AddString("gufo.rotation.side", "input");
  }

  [[nodiscard]] std::vector<std::uint8_t> Build() const {
    GgufImage out = image;
    if (markers) {
      AddMarkers(out);
    }
    for (const auto& [name, values] : signs) {
      out.AddTensorF16(name + ".rot", values);
    }
    return out.Build();
  }
};

GroupTable Table() {
  GroupTable table;
  for (int layer = 0; layer < 2; ++layer) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    const std::string l = std::to_string(layer);
    table.instances.push_back(
        {"in", "in:" + l, {p + "a.weight", p + "b.weight"}});
    table.instances.push_back({"down", "down:" + l, {p + "c.weight"}});
  }
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

void ExpectRefusal(const std::vector<std::uint8_t>& bytes,
                   std::string_view fragment, std::string_view label,
                   const GroupTable& table = Table()) {
  const auto outcome = Read(bytes, table);
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

void TestUnrotated() {
  Fixture fixture;
  fixture.markers = false;
  const auto outcome = Read(fixture.Build());
  Expect(outcome.ok && !outcome.info.Any(), "unrotated file has no rotation");

  // Provenance keys alone are not markers; the converter writes them for
  // unrotated outputs as well.
  GgufImage image = fixture.image;
  image.AddString("gufo.rotation.source_converter", "rotquant 1");
  image.AddString("gufo.rotation.source_recipe_sha256", "00");
  const auto provenance = Read(image.Build());
  Expect(provenance.ok && !provenance.info.Any(),
         "source_* keys are not rotation markers");
}

void TestAccepted() {
  Fixture fixture;
  fixture.RotateAll();
  const auto outcome = Read(fixture.Build());
  Expect(outcome.ok, "rotated file accepted: " + outcome.error);
  if (!outcome.ok) {
    return;
  }
  const auto& info = outcome.info;
  Expect(info.Any(), "rotation present");
  Expect(info.rotated_groups == std::vector<std::string>{"in", "down"},
         "rotated groups in table order");
  Expect(info.GroupList() == "in, down", "group list text");
  Expect(info.instances.size() == 4, "four rotated instances");
  for (int layer = 0; layer < 2; ++layer) {
    const auto l = std::to_string(layer);
    const auto* in = info.Find("in:" + l);
    const auto* down = info.Find("down:" + l);
    Expect(in != nullptr && down != nullptr, "instances found");
    if (in == nullptr || down == nullptr) {
      continue;
    }
    Expect(in->k == 256 && in->bits.size() == 8, "in: K and words");
    Expect(down->k == 384 && down->bits.size() == 12, "down: K and words");
    const auto& want = fixture.signs.at("blk." + l + ".c.weight");
    bool same = true;
    for (std::size_t i = 0; i < want.size(); ++i) {
      same = same &&
             gufo::test::rotation::SignBit(down->bits, i) == (want[i] < 0.0F);
    }
    Expect(same, "packed signs match the .rot values");
  }

  // Rotating only some groups is fine as long as each group is complete.
  Fixture partial;
  partial.RotateAll();
  partial.signs.erase("blk.0.c.weight");
  partial.signs.erase("blk.1.c.weight");
  const auto only_in = Read(partial.Build());
  Expect(only_in.ok &&
             only_in.info.rotated_groups == std::vector<std::string>{"in"},
         "one complete group rotated");
}

void TestMarkerRefusals() {
  {
    Fixture fixture;
    fixture.RotateAll();
    fixture.markers = false;
    ExpectRefusal(fixture.Build(), "no gufo.rotation metadata",
                  ".rot without markers");
  }
  {
    Fixture fixture;
    GgufImage image = fixture.image;
    fixture.AddMarkers(image);
    ExpectRefusal(image.Build(), "no .rot tensors", "markers without .rot");
  }
  const auto with_markers = [](std::string_view skip, std::string_view key,
                               auto add) {
    Fixture fixture;
    fixture.RotateAll();
    fixture.markers = false;
    GgufImage image = fixture.image;
    const std::vector<std::pair<std::string_view, int>> keys = {
        {"gufo.rotation.version", 0},
        {"gufo.rotation.kind", 1},
        {"gufo.rotation.block_size", 2},
        {"gufo.rotation.side", 3}};
    for (const auto& [name, index] : keys) {
      if (name == skip) {
        continue;
      }
      if (name == key) {
        add(image);
        continue;
      }
      if (index == 0) {
        image.AddUint32(name, 1);
      } else if (index == 1) {
        image.AddString(name, "hadamard");
      } else if (index == 2) {
        image.AddUint32(name, 128);
      } else {
        image.AddString(name, "input");
      }
    }
    for (const auto& [name, values] : fixture.signs) {
      image.AddTensorF16(name + ".rot", values);
    }
    return image.Build();
  };
  const auto none = [](GgufImage&) {};
  for (const auto key : {"gufo.rotation.version", "gufo.rotation.kind",
                         "gufo.rotation.block_size", "gufo.rotation.side"}) {
    ExpectRefusal(with_markers(key, "", none), "missing " + std::string(key),
                  key);
  }
  ExpectRefusal(with_markers("", "gufo.rotation.version",
                             [](GgufImage& image) {
                               image.AddUint32("gufo.rotation.version", 2);
                             }),
                "unsupported gufo.rotation.version = 2", "version 2");
  ExpectRefusal(with_markers("", "gufo.rotation.version",
                             [](GgufImage& image) {
                               image.AddString("gufo.rotation.version", "1");
                             }),
                "unsupported gufo.rotation.version", "string version");
  ExpectRefusal(with_markers("", "gufo.rotation.kind",
                             [](GgufImage& image) {
                               image.AddString("gufo.rotation.kind", "dct");
                             }),
                "unsupported gufo.rotation.kind = \"dct\"", "kind");
  ExpectRefusal(with_markers("", "gufo.rotation.block_size",
                             [](GgufImage& image) {
                               image.AddUint32("gufo.rotation.block_size", 64);
                             }),
                "unsupported gufo.rotation.block_size = 64", "block size");
  ExpectRefusal(with_markers("", "gufo.rotation.side",
                             [](GgufImage& image) {
                               image.AddString("gufo.rotation.side", "output");
                             }),
                "unsupported gufo.rotation.side = \"output\"", "side");
}

void TestTensorRefusals() {
  {
    Fixture fixture;
    fixture.RotateAll();
    fixture.signs["emb.weight"] = Signs(256, 3);
    ExpectRefusal(fixture.Build(),
                  "emb.weight is rotated but its role has no rotation path",
                  "tensor outside every group");
  }
  {
    Fixture fixture;
    fixture.RotateAll();
    fixture.signs["missing.weight"] = Signs(256, 3);
    ExpectRefusal(fixture.Build(),
                  "missing.weight.rot has no matching tensor missing.weight",
                  "orphan .rot");
  }
  {
    Fixture fixture;
    fixture.RotateAll();
    fixture.signs["emb.weight"] = Signs(256, 3);
    auto table = Table();
    table.tied_tensors.push_back("emb.weight");
    ExpectRefusal(fixture.Build(), "tied to the token embedding", "tied head",
                  table);
  }
  {
    Fixture fixture;
    fixture.RotateAll();
    auto table = Table();
    table.unsupported_groups.push_back("down");
    ExpectRefusal(fixture.Build(),
                  "group down has no rotation path in this runtime",
                  "unsupported group", table);
  }
  {
    Fixture fixture;
    fixture.RotateAll();
    fixture.signs.erase("blk.1.b.weight");
    ExpectRefusal(fixture.Build(),
                  "group instance in:1 mixes rotated and unrotated tensors",
                  "mixed instance");
  }
  {
    Fixture fixture;
    fixture.RotateAll();
    fixture.signs["blk.0.b.weight"][7] *= -1.0F;
    ExpectRefusal(fixture.Build(),
                  "group instance in:0: blk.0.b.weight uses a different sign "
                  "vector",
                  "different signs");
  }
  {
    Fixture fixture;
    fixture.RotateAll();
    fixture.signs.erase("blk.1.c.weight");
    ExpectRefusal(fixture.Build(),
                  "group down is rotated in some layers only (1 of 2; down:1",
                  "partial layers");
  }
  {
    Fixture fixture;
    fixture.RotateAll();
    fixture.signs["blk.0.c.weight"][5] = 0.5F;
    ExpectRefusal(fixture.Build(),
                  "blk.0.c.weight.rot holds values other than +1 and -1",
                  "non-sign value");
  }
  {
    Fixture fixture;
    fixture.RotateAll();
    fixture.signs["blk.0.c.weight"].resize(256);
    ExpectRefusal(fixture.Build(),
                  "blk.0.c.weight.rot has 256 values, but blk.0.c.weight has "
                  "K = 384",
                  "wrong length");
  }
  {
    Fixture fixture;
    fixture.RotateAll();
    fixture.markers = false;
    GgufImage image = fixture.image;
    fixture.AddMarkers(image);
    for (const auto& [name, values] : fixture.signs) {
      if (name == "blk.0.c.weight") {
        image.AddTensorF32Values(name + ".rot", {384}, values);
      } else {
        image.AddTensorF16(name + ".rot", values);
      }
    }
    ExpectRefusal(image.Build(),
                  "blk.0.c.weight.rot must be a one-dimensional F16 tensor",
                  "F32 signs");
  }
  {
    // K = 96 is not a whole rotation block.
    GgufImage image;
    image.AddTensorF32("blk.0.a.weight", {96, 2});
    image.AddUint32("gufo.rotation.version", 1);
    image.AddString("gufo.rotation.kind", "hadamard");
    image.AddUint32("gufo.rotation.block_size", 128);
    image.AddString("gufo.rotation.side", "input");
    image.AddTensorF16("blk.0.a.weight.rot", Signs(96, 4));
    GroupTable table;
    table.instances.push_back({"in", "in:0", {"blk.0.a.weight"}});
    ExpectRefusal(image.Build(), "not a multiple of 128", "K = 96", table);
  }
}

void TestReference() {
  using gufo::test::rotation::RotateRow;
  using gufo::test::rotation::RotateRowFp64;
  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  for (const std::size_t k : {128U, 640U, 5120U, 6144U, 17408U}) {
    std::vector<float> row(k);
    for (auto& value : row) {
      value = normal(rng);
    }
    row[3] = 300.0F;  // an activation outlier
    std::vector<std::uint32_t> bits(k / 32);
    for (auto& word : bits) {
      word = rng();
    }
    auto rotated = row;
    RotateRow(rotated, bits);
    const auto exact = RotateRowFp64(row, bits);
    double max_error = 0.0;
    double max_value = 0.0;
    double norm_in = 0.0;
    double norm_out = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
      max_error = std::max(max_error, std::abs(rotated[i] - exact[i]));
      max_value = std::max(max_value, std::abs(exact[i]));
      norm_in += static_cast<double>(row[i]) * row[i];
      norm_out += exact[i] * exact[i];
    }
    Expect(max_error <= 1e-5 * std::max(1.0, max_value),
           "FP32 reference matches the FP64 product, K = " + std::to_string(k) +
               " (error " + std::to_string(max_error) + ")");
    Expect(std::abs(norm_in - norm_out) <= 1e-9 * norm_in,
           "rotation preserves the norm");
    // H·D is orthogonal: rotating the FP64 result back with H, then D,
    // recovers the input. With D = I, applying the FP32 rotation twice is
    // the identity up to rounding.
    std::vector<std::uint32_t> plus(k / 32, 0U);
    auto twice = row;
    RotateRow(twice, plus);
    RotateRow(twice, plus);
    double round_trip = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
      round_trip = std::max(round_trip,
                            static_cast<double>(std::abs(twice[i] - row[i])));
    }
    Expect(round_trip <= 1e-4, "H·H = I in FP32");
  }
  // The scale constant is the correctly rounded float of 1 / sqrt(128).
  Expect(gufo::hip::hadamard::kHadamard128Scale ==
             static_cast<float>(1.0 / std::sqrt(128.0)),
         "scale constant");
}

}  // namespace

int main() {
  TestUnrotated();
  TestAccepted();
  TestMarkerRefusals();
  TestTensorRefusals();
  TestReference();
  if (failures != 0) {
    std::cerr << failures << " rotation check(s) failed\n";
    return 1;
  }
  std::cout << "rotation_test: all checks passed\n";
  return 0;
}
