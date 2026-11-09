#ifndef GUFO_CORE_DEVICE_CODE_IDENTITY_HPP_
#define GUFO_CORE_DEVICE_CODE_IDENTITY_HPP_

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace gufo::core {

/// The bytes of section `name` in a little-endian ELF64 image. Returns
/// nullopt when `image` is not such an image (a Windows PE file, say), its
/// headers are truncated, or it has no section of that name.
[[nodiscard]] std::optional<std::span<const std::uint8_t>> FindElfSection(
    std::span<const std::uint8_t> image, std::string_view name) noexcept;

/// The ELF section holding every HIP device code object linked into an
/// executable, one clang offload bundle per translation unit.
inline constexpr std::string_view kHipFatbinSection = ".hip_fatbin";

/// SHA-256 (64 lowercase hex characters) of the running executable's
/// embedded device code objects, as stored (compressed bundles). Any kernel
/// change alters it, and so does a toolchain or flag change. Host-only C++
/// sources do not. A host-only edit inside a HIP translation unit (or a
/// header it includes) can, because RelWithDebInfo device code carries DWARF
/// line tables: it errs towards invalidating. Empty when the executable
/// embeds none (a CPU-only build) or cannot be read. Read once per process.
[[nodiscard]] const std::string& ExecutableDeviceCodeSha256();

}  // namespace gufo::core

#endif  // GUFO_CORE_DEVICE_CODE_IDENTITY_HPP_
