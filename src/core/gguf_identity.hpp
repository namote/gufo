#ifndef GUFO_CORE_GGUF_IDENTITY_HPP_
#define GUFO_CORE_GGUF_IDENTITY_HPP_

#include <string>
#include <string_view>

#include "src/core/gguf_reader.hpp"

namespace gufo::core {

/// Scheme tag for full-content GGUF identities.
inline constexpr std::string_view kNativeIdentityScheme =
    "native-model-files-sha256-v1";
inline constexpr std::string_view kGgufIdentityScheme = "gguf-sha256-v1";

/// Hashes every byte of every shard, including metadata and tensor padding.
/// File digests are cached privately by inode, size, mtime and ctime; memory
/// images are always hashed. Copies have identical identities. The initial
/// scan reads the full artifact; later launches reuse its cached full digest.
/// Throws if the mapped artifact changes while its identity is computed.
[[nodiscard]] std::string GgufIdentityHex(const GgufReader& reader);

/// Full-content identity of a native model file set. Reuses the same per-file
/// digest cache. Names must be relative, unique and deterministically ordered.
[[nodiscard]] std::string FileSetIdentityHex(
    const std::filesystem::path& directory,
    const std::vector<std::string>& names);

}  // namespace gufo::core

#endif  // GUFO_CORE_GGUF_IDENTITY_HPP_
