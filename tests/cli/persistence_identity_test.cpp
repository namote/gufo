#include "src/cli/serve/persistence_identity.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/device_code_identity.hpp"
#include "src/core/platform/mapped_file.hpp"

namespace {

using gufo::core::FindElfSection;
using gufo::server::QwenFlashNextCompatibilityIdentity;

std::string Identity(std::string_view device_code) {
  const auto bytes = QwenFlashNextCompatibilityIdentity(
      std::string(64, 'a'), std::string(64, 'b'), true, 8192, 7, 2,
      "absolute-v1", true, 19, device_code);
  return {bytes.begin(), bytes.end()};
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
  FindsElfSections();
  ExecutableHashIsStable();
  std::cout << "persistence identity tests passed\n";
  return 0;
}
