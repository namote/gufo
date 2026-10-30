// X1 correctness path tool (plan/exl3/XA-BRIEF.md): Qwen3.8 27B tensors in
// the llama.cpp qwen35 layout from an EXL3 checkpoint
// (src/models/qwen/exl3_weights.hpp).
//
//   exl3_to_gguf compare EXL3_DIR GGUF_SHARD... [--only SUBSTRING] [--threads
//   N]
//     Builds every tensor the BF16 GGUF holds and compares: unquantized
//     tensors must match exactly; quantized projections report the relative
//     error and correlation against the BF16 original (the EXL3 quantization
//     error, about 0.04-0.14 at 5-3 bits; a layout error shows as ~1 or more).
//   exl3_to_gguf write EXL3_DIR OUT.gguf GGUF_SHARD... [--threads N]
//     Writes a single-file BF16 GGUF: the metadata of the given (template)
//     GGUF without split.* keys, and every template tensor built from the
//     EXL3 checkpoint.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/exl3/checkpoint.hpp"
#include "src/models/qwen/exl3_weights.hpp"

namespace {

using gufo::core::GgmlType;
using gufo::core::GgufReader;
namespace qe = gufo::models::qwen_exl3;

struct RawKv {
  std::string key;
  std::vector<std::uint8_t> bytes;  // key + type + value, as stored
};

// Minimal GGUF v3 metadata scanner (key/value byte ranges of the first file).
std::vector<RawKv> ReadRawMetadata(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot open " + path);
  auto read = [&](void* p, std::size_t n) {
    if (!in.read(static_cast<char*>(p), static_cast<std::streamsize>(n)))
      throw std::runtime_error("truncated GGUF " + path);
  };
  char magic[4];
  read(magic, 4);
  if (std::memcmp(magic, "GGUF", 4) != 0)
    throw std::runtime_error("not GGUF");
  std::uint32_t version;
  std::uint64_t tensors, kvs;
  read(&version, 4);
  read(&tensors, 8);
  read(&kvs, 8);
  std::vector<RawKv> out;
  std::function<void(std::uint32_t)> skip = [&](std::uint32_t type) {
    static const int sizes[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (type == 8) {
      std::uint64_t n;
      read(&n, 8);
      in.seekg(static_cast<std::streamoff>(n), std::ios::cur);
    } else if (type == 9) {
      std::uint32_t et;
      std::uint64_t n;
      read(&et, 4);
      read(&n, 8);
      if (et != 8 && et != 9) {
        in.seekg(static_cast<std::streamoff>(n * sizes[et]), std::ios::cur);
      } else {
        for (std::uint64_t i = 0; i < n; ++i)
          skip(et);
      }
    } else {
      in.seekg(sizes[type], std::ios::cur);
    }
  };
  for (std::uint64_t i = 0; i < kvs; ++i) {
    const auto start = in.tellg();
    std::uint64_t len;
    read(&len, 8);
    std::string key(len, '\0');
    read(key.data(), len);
    std::uint32_t type;
    read(&type, 4);
    skip(type);
    const auto end = in.tellg();
    RawKv kv;
    kv.key = key;
    kv.bytes.resize(static_cast<std::size_t>(end - start));
    in.seekg(start);
    read(kv.bytes.data(), kv.bytes.size());
    out.push_back(std::move(kv));
  }
  return out;
}

struct Template {
  std::vector<std::unique_ptr<GgufReader>> shards;
  std::vector<const gufo::core::GgufTensorInfo*> tensors;
};

Template OpenTemplate(const std::vector<std::string>& paths) {
  Template t;
  for (const auto& p : paths) {
    std::string error;
    auto reader = GgufReader::OpenFile(p, &error);
    if (!reader)
      throw std::runtime_error("cannot open " + p + ": " + error);
    for (const auto& info : reader->GetTensors())
      t.tensors.push_back(&info);
    t.shards.push_back(std::move(reader));
  }
  return t;
}

float ValueOf(GgmlType type, const void* data, std::size_t i) {
  if (type == GgmlType::kF32) {
    float f;
    std::memcpy(&f, static_cast<const std::uint8_t*>(data) + 4 * i, 4);
    return f;
  }
  std::uint16_t b;
  std::memcpy(&b, static_cast<const std::uint8_t*>(data) + 2 * i, 2);
  return qe::Bf16ToFloat(b);
}

int Compare(const std::string& exl3_dir, const std::vector<std::string>& shards,
            const std::string& only, int threads) {
  const auto checkpoint = gufo::quant::exl3::Checkpoint::Open(exl3_dir);
  const auto tmpl = OpenTemplate(shards);
  std::string error;
  const auto geometry = qe::GeometryFromGguf(*tmpl.shards[0], &error);
  if (!geometry)
    throw std::runtime_error(error);
  const qe::Source source(checkpoint, *geometry);
  int failures = 0;
  std::size_t done = 0;
  for (const auto* info : tmpl.tensors) {
    const std::string name(info->name);
    if (!only.empty() && name.find(only) == std::string::npos)
      continue;
    error.clear();
    const auto plan = source.Plan(name, &error);
    if (!plan) {
      std::cout << "MISSING " << error << "\n";
      ++failures;
      continue;
    }
    if (plan->type != info->type) {
      std::cout << "TYPE " << name << "\n";
      ++failures;
      continue;
    }
    std::size_t elements = 1;
    for (std::size_t d = 0; d < plan->ne.size(); ++d) {
      if (d >= info->dimensions.size() || plan->ne[d] != info->dimensions[d]) {
        std::cout << "SHAPE " << name << "\n";
        ++failures;
        elements = 0;
        break;
      }
      elements *= plan->ne[d];
    }
    if (elements == 0)
      continue;
    std::vector<std::uint8_t> ours(plan->bytes);
    const auto t0 = std::chrono::steady_clock::now();
    source.Produce(*plan, ours, threads);
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    const bool exact = std::memcmp(ours.data(), info->data, plan->bytes) == 0;
    double se = 0, so = 0, sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0;
    std::size_t mismatched = 0;
    for (std::size_t i = 0; i < elements; ++i) {
      const double x = ValueOf(plan->type, ours.data(), i);
      const double y = ValueOf(info->type, info->data, i);
      se += (x - y) * (x - y);
      so += y * y;
      sx += x;
      sy += y;
      sxy += x * y;
      sxx += x * x;
      syy += y * y;
      mismatched += x != y;
    }
    const double n = static_cast<double>(elements);
    const double corr = (sxy - sx * sy / n) /
                        std::sqrt((sxx - sx * sx / n) * (syy - sy * sy / n));
    const double rel = std::sqrt(se / std::max(so, 1e-300));
    // Quantized projections are BF16 matrices other than the embedding and
    // the unquantized alpha / beta projections, which must match exactly.
    const bool quantized = plan->type == GgmlType::kBF16 &&
                           plan->ne.size() == 2 &&
                           name != "token_embd.weight" &&
                           name.find("ssm_alpha") == std::string::npos &&
                           name.find("ssm_beta") == std::string::npos;
    // NEAR: unquantized tensors the checkpoint stores as F16 where the
    // original is BF16 (linear_attn in_proj_a / in_proj_b): FP16 rounding of
    // tiny values only.
    const char* verdict = exact                      ? "EXACT"
                          : (quantized && rel < 0.3) ? "QUANT"
                          : rel < 1e-6               ? "NEAR"
                                                     : "DIFF";
    if (std::string(verdict) == "DIFF")
      ++failures;
    std::cout << verdict << " " << name << " rel " << rel << " corr " << corr
              << " mismatched " << mismatched << "/" << elements << " (" << secs
              << " s)\n";
    ++done;
  }
  std::cout << "compared " << done << " tensors, " << failures
            << " failure(s)\n";
  return failures == 0 ? 0 : 1;
}

void WriteU32(std::ofstream& o, std::uint32_t v) {
  o.write(reinterpret_cast<char*>(&v), 4);
}
void WriteU64(std::ofstream& o, std::uint64_t v) {
  o.write(reinterpret_cast<char*>(&v), 8);
}

int Write(const std::string& exl3_dir, const std::string& out_path,
          const std::vector<std::string>& shards, int threads) {
  const auto checkpoint = gufo::quant::exl3::Checkpoint::Open(exl3_dir);
  const auto tmpl = OpenTemplate(shards);
  std::string error;
  const auto geometry = qe::GeometryFromGguf(*tmpl.shards[0], &error);
  if (!geometry)
    throw std::runtime_error(error);
  const qe::Source source(checkpoint, *geometry);
  std::vector<RawKv> kvs;
  for (auto& kv : ReadRawMetadata(shards[0])) {
    if (!kv.key.starts_with("split."))
      kvs.push_back(std::move(kv));
  }
  std::vector<qe::TensorPlan> plans;
  for (const auto* info : tmpl.tensors) {
    error.clear();
    auto plan = source.Plan(info->name, &error);
    if (!plan)
      throw std::runtime_error(error);
    if (plan->type != info->type)
      throw std::runtime_error("type " + plan->gguf_name);
    plans.push_back(std::move(*plan));
  }
  constexpr std::uint64_t kAlign = 32;
  auto align = [](std::uint64_t x) {
    return (x + kAlign - 1) / kAlign * kAlign;
  };
  std::ofstream out(out_path, std::ios::binary);
  if (!out)
    throw std::runtime_error("cannot write " + out_path);
  out.write("GGUF", 4);
  WriteU32(out, 3);
  WriteU64(out, plans.size());
  WriteU64(out, kvs.size());
  for (const auto& kv : kvs) {
    out.write(reinterpret_cast<const char*>(kv.bytes.data()),
              static_cast<std::streamsize>(kv.bytes.size()));
  }
  std::uint64_t offset = 0;
  for (const auto& plan : plans) {
    WriteU64(out, plan.gguf_name.size());
    out.write(plan.gguf_name.data(),
              static_cast<std::streamsize>(plan.gguf_name.size()));
    WriteU32(out, static_cast<std::uint32_t>(plan.ne.size()));
    for (const auto d : plan.ne)
      WriteU64(out, d);
    WriteU32(out, static_cast<std::uint32_t>(plan.type));
    WriteU64(out, offset);
    offset = align(offset + plan.bytes);
  }
  const auto pad = [&](std::uint64_t pos) {
    const std::uint64_t target = align(pos);
    const std::vector<char> zeros(target - pos, 0);
    out.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
  };
  pad(static_cast<std::uint64_t>(out.tellp()));
  std::vector<std::uint8_t> buffer;
  std::size_t index = 0;
  for (const auto& plan : plans) {
    buffer.resize(plan.bytes);
    const auto t0 = std::chrono::steady_clock::now();
    source.Produce(plan, buffer, threads);
    out.write(reinterpret_cast<const char*>(buffer.data()),
              static_cast<std::streamsize>(buffer.size()));
    const auto data_pos = align(plan.bytes) - plan.bytes;
    const std::vector<char> zeros(data_pos, 0);
    out.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
    ++index;
    std::cout << "[" << index << "/" << plans.size() << "] " << plan.gguf_name
              << " "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t0)
                     .count()
              << " s\n"
              << std::flush;
  }
  out.close();
  if (!out)
    throw std::runtime_error("write failed: " + out_path);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::vector<std::string> args(argv + 1, argv + argc);
    int threads =
        static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    std::string only;
    std::vector<std::string> positional;
    for (std::size_t i = 0; i < args.size(); ++i) {
      if (args[i] == "--threads" && i + 1 < args.size()) {
        threads = std::stoi(args[++i]);
      } else if (args[i] == "--only" && i + 1 < args.size()) {
        only = args[++i];
      } else {
        positional.push_back(args[i]);
      }
    }
    if (positional.size() >= 3 && positional[0] == "compare") {
      return Compare(positional[1], {positional.begin() + 2, positional.end()},
                     only, threads);
    }
    if (positional.size() >= 4 && positional[0] == "write") {
      return Write(positional[1], positional[2],
                   {positional.begin() + 3, positional.end()}, threads);
    }
    std::cerr << "usage: exl3_to_gguf compare EXL3_DIR GGUF_SHARD... [--only "
                 "S] [--threads N]\n"
                 "       exl3_to_gguf write EXL3_DIR OUT.gguf GGUF_SHARD... "
                 "[--threads N]\n";
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "exl3_to_gguf: " << error.what() << "\n";
    return 1;
  }
}
