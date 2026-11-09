#include "src/core/device_code_identity.hpp"

#include <cstddef>
#include <cstring>

#include "src/core/crypto/sha256.hpp"
#include "src/core/platform/mapped_file.hpp"
#include "src/core/platform/os_info.hpp"

namespace gufo::core {
namespace {

template<typename T>
bool Read(std::span<const std::uint8_t> image, std::uint64_t offset,
          T* value) noexcept {
  if (offset > image.size() || image.size() - offset < sizeof(T))
    return false;
  std::memcpy(value, image.data() + offset, sizeof(T));
  return true;
}

}  // namespace

std::optional<std::span<const std::uint8_t>> FindElfSection(
    std::span<const std::uint8_t> image, std::string_view name) noexcept {
  // e_ident: magic, ELFCLASS64, ELFDATA2LSB. Fields are read as host
  // integers, which are little-endian on every supported target.
  static constexpr std::uint8_t kIdent[] = {0x7f, 'E', 'L', 'F', 2, 1};
  if (image.size() < 64 || std::memcmp(image.data(), kIdent, sizeof(kIdent)))
    return std::nullopt;
  std::uint64_t section_offset{};
  std::uint16_t entry_size{}, count{}, names_index{};
  if (!Read(image, 0x28, &section_offset) || !Read(image, 0x3a, &entry_size) ||
      !Read(image, 0x3c, &count) || !Read(image, 0x3e, &names_index) ||
      entry_size < 64 || names_index >= count ||
      section_offset > image.size() ||
      std::uint64_t{count} * entry_size > image.size() - section_offset)
    return std::nullopt;
  const auto header = [&](std::uint16_t index, std::uint32_t* name_offset,
                          std::uint32_t* type, std::uint64_t* offset,
                          std::uint64_t* size) {
    const auto base = section_offset + std::uint64_t{index} * entry_size;
    return Read(image, base, name_offset) && Read(image, base + 4, type) &&
           Read(image, base + 0x18, offset) && Read(image, base + 0x20, size) &&
           *offset <= image.size() && *size <= image.size() - *offset;
  };
  std::uint32_t names_name{}, names_type{};
  std::uint64_t names_offset{}, names_size{};
  if (!header(names_index, &names_name, &names_type, &names_offset,
              &names_size))
    return std::nullopt;
  const auto names = image.subspan(names_offset, names_size);
  for (std::uint16_t i = 0; i < count; ++i) {
    std::uint32_t name_offset{}, type{};
    std::uint64_t offset{}, size{};
    if (!header(i, &name_offset, &type, &offset, &size))
      return std::nullopt;
    if (name_offset >= names.size() ||
        names.size() - name_offset <= name.size() ||
        std::memcmp(names.data() + name_offset, name.data(), name.size()) ||
        names[name_offset + name.size()] != 0)
      continue;
    // SHT_NOBITS occupies no file bytes.
    if (type == 8)
      return std::span<const std::uint8_t>{};
    return image.subspan(offset, size);
  }
  return std::nullopt;
}

const std::string& ExecutableDeviceCodeSha256() {
  static const std::string hash = [] {
    const auto path = platform::RunningExecutablePath();
    if (!path)
      return std::string{};
    const auto file = platform::ReadOnlyMappedFile::Open(*path);
    if (!file.valid())
      return std::string{};
    const auto section =
        FindElfSection({file.data(), file.size()}, kHipFatbinSection);
    if (!section || section->empty())
      return std::string{};
    return crypto::Sha256Hex(*section);
  }();
  return hash;
}

}  // namespace gufo::core
