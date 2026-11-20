#include "src/cli/serve/persistence_identity.hpp"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/device_code_identity.hpp"
#include "src/core/gguf_identity.hpp"
#include "src/core/platform/mapped_file.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace {

using gufo::core::FindElfSection;
using gufo::server::QwenFlashNextCompatibilityIdentity;

std::string Identity(std::string_view device_code,
                     std::string_view hc_mode = "fp16",
                     std::string_view recurrent_state = "fp32") {
  const auto bytes = QwenFlashNextCompatibilityIdentity(
      std::string(64, 'a'), std::string(64, 'b'), true, 8192, 7, 2,
      "absolute-v1", true, recurrent_state == "fp16" ? 20 : 19, device_code,
      hc_mode, recurrent_state);
  return {bytes.begin(), bytes.end()};
}

// FP16 recurrent state payloads never restore into an FP32 server and back.
void RecurrentStateChangesTheIdentity() {
  const std::string code(64, '1');
  const auto fp32 = Identity(code, "fp16", "fp32");
  const auto fp16 = Identity(code, "fp16", "fp16");
  // The FP32 identity is byte for byte the one written before
  // --recurrent-state existed (c3b09033), so its retained states still match.
  const std::string scheme(gufo::core::kNativeIdentityScheme);
  const std::string expected_fp32 =
      "schema=gufo-text-continuation-v2\n"
      "model_kind=qwen38-flash-next\n"
      "artifact_id=" +
      scheme + ':' + std::string(64, 'a') +
      "\n"
      "tokenizer=embedded-in-artifact\n"
      "chat_template=qwen38-reasoning-compiled-v3\n"
      "chat_template_reference_sha256=" +
      std::string(
          gufo::tokenization::QwenChatTemplate::OfficialTemplateSha256()) +
      "\n"
      "state_abi=qwen38-flash-next-rocm-session-v1\n"
      "payload_layout=qfn-rocm-session-snapshot-v19\n"
      "device_code=sha256:" +
      code +
      "\n"
      "context_tokens=8192\n"
      "position_policy=absolute-v1\n"
      "hc_mode=fp16\n"
      "adapters=none\n"
      "draft_backend=qfn-mtp-v1\n"
      "draft_artifact_id=" +
      scheme + ':' + std::string(64, 'b') +
      "\n"
      "draft_max_tokens=7\n"
      "draft_cost_concurrency=2\n";
  if (fp32 != expected_fp32 ||
      fp16.find("hc_mode=fp16\nrecurrent_state=fp16\nadapters=none\n") ==
          std::string::npos ||
      fp16.find("payload_layout=qfn-rocm-session-snapshot-v20\n") ==
          std::string::npos) {
    std::cerr << "recurrent state is not part of the identity\n";
    std::exit(1);
  }
}

// Each --hc-mode computes different states: none restores into another.
void HcModeChangesTheIdentity() {
  const std::string code(64, '1');
  const auto fp16 = Identity(code, "fp16");
  const auto q8 = Identity(code, "q8");
  const auto calibrated = Identity(code, "q8-calibrated");
  assert(fp16 != q8 && fp16 != calibrated && q8 != calibrated);
  assert(fp16.find("hc_mode=fp16\n") != std::string::npos);
  assert(q8.find("hc_mode=q8\n") != std::string::npos);
  assert(calibrated.find("hc_mode=q8-calibrated\n") != std::string::npos);
  assert(Identity(code, "q8") == q8);
}

// A kernel change alters the device code hash and with it the identity, so
// disk states captured by older kernels no longer match.
void DeviceCodeHashChangesTheIdentity() {
  const std::string first(64, '1'), second(64, '2');
  assert(Identity(first) == Identity(first));
  assert(Identity(first) != Identity(second));
  assert(Identity(first).find("device_code=sha256:" + first + "\n") !=
         std::string::npos);
  assert(Identity("").find("device_code=unknown\n") != std::string::npos);
  assert(Identity("") != Identity(first));
}

template<typename T>
void Put(std::vector<std::uint8_t>& image, std::size_t offset, T value) {
  std::memcpy(image.data() + offset, &value, sizeof(value));
}

// A minimal ELF64 image: null, .shstrtab and .hip_fatbin section headers.
std::vector<std::uint8_t> ElfImage(std::string_view payload) {
  const std::string names = std::string("\0.shstrtab\0.hip_fatbin\0", 23);
  const std::size_t names_offset = 64;
  const std::size_t payload_offset = names_offset + names.size();
  const std::size_t headers = payload_offset + payload.size();
  std::vector<std::uint8_t> image(headers + 3 * 64);
  const std::uint8_t ident[] = {0x7f, 'E', 'L', 'F', 2, 1, 1};
  std::memcpy(image.data(), ident, sizeof(ident));
  Put<std::uint64_t>(image, 0x28, headers);
  Put<std::uint16_t>(image, 0x3a, 64);
  Put<std::uint16_t>(image, 0x3c, 3);
  Put<std::uint16_t>(image, 0x3e, 1);
  std::memcpy(image.data() + names_offset, names.data(), names.size());
  std::memcpy(image.data() + payload_offset, payload.data(), payload.size());
  const auto section = [&](std::size_t index, std::uint32_t name,
                           std::uint64_t offset, std::uint64_t size) {
    const auto base = headers + index * 64;
    Put<std::uint32_t>(image, base, name);
    Put<std::uint32_t>(image, base + 4, 1);
    Put<std::uint64_t>(image, base + 0x18, offset);
    Put<std::uint64_t>(image, base + 0x20, size);
  };
  section(1, 1, names_offset, names.size());
  section(2, 11, payload_offset, payload.size());
  return image;
}

void FindsElfSections() {
  const auto image = ElfImage("device code");
  const auto found = FindElfSection(image, gufo::core::kHipFatbinSection);
  assert(found && std::string_view(reinterpret_cast<const char*>(found->data()),
                                   found->size()) == "device code");
  assert(!FindElfSection(image, ".hip_fat"));
  assert(!FindElfSection(image, ".hip_fatbinx"));
  // Truncated section headers and images that are not ELF64 find nothing.
  const std::vector<std::uint8_t> truncated(image.begin(), image.end() - 64);
  assert(!FindElfSection(truncated, gufo::core::kHipFatbinSection));
  auto pe = image;
  pe[0] = 'M';
  pe[1] = 'Z';
  assert(!FindElfSection(pe, gufo::core::kHipFatbinSection));
  assert(!FindElfSection({}, gufo::core::kHipFatbinSection));
  // A section past the image end, or a name table index out of range.
  auto beyond = image;
  const std::uint64_t headers = beyond.size() - 3 * 64;
  Put<std::uint64_t>(beyond, headers + 2 * 64 + 0x20, beyond.size());
  assert(!FindElfSection(beyond, gufo::core::kHipFatbinSection));
  auto index = image;
  Put<std::uint16_t>(index, 0x3e, 3);
  assert(!FindElfSection(index, gufo::core::kHipFatbinSection));
  // SHT_NOBITS occupies no file bytes.
  auto nobits = image;
  Put<std::uint32_t>(nobits, headers + 2 * 64 + 4, 8);
  const auto empty = FindElfSection(nobits, gufo::core::kHipFatbinSection);
  assert(empty && empty->empty());
}

void ExecutableHashIsStable() {
  const auto& hash = gufo::core::ExecutableDeviceCodeSha256();
  assert(hash.empty() || gufo::server::IsSha256Hex(hash));
  assert(&hash == &gufo::core::ExecutableDeviceCodeSha256());
}

}  // namespace

int main(int argc, char** argv) {
  // Optional: print the device code hash of another executable.
  if (argc == 2) {
    const auto file = gufo::platform::ReadOnlyMappedFile::Open(argv[1]);
    const auto section = FindElfSection({file.data(), file.size()},
                                        gufo::core::kHipFatbinSection);
    std::cout << (section ? gufo::crypto::Sha256Hex(*section) : "none") << '\n';
    return section ? 0 : 1;
  }
  DeviceCodeHashChangesTheIdentity();
  HcModeChangesTheIdentity();
  RecurrentStateChangesTheIdentity();
  FindsElfSections();
  ExecutableHashIsStable();
  std::cout << "persistence identity tests passed\n";
  return 0;
}
