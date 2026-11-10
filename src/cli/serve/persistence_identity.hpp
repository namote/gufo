#ifndef GUFO_SERVER_PERSISTENCE_IDENTITY_HPP_
#define GUFO_SERVER_PERSISTENCE_IDENTITY_HPP_

#include <cstdint>
#include <string_view>
#include <vector>

namespace gufo::server {

/// Whether `value` is 64 lowercase hex characters.
[[nodiscard]] bool IsSha256Hex(std::string_view value) noexcept;

/// The canonical disk-cache compatibility identity of a Flash-Next server.
/// `device_code_sha256` is core::ExecutableDeviceCodeSha256(): a kernel
/// change invalidates retained disk states without a payload version bump,
/// while host-only C++ changes keep them (see that function for HIP sources).
/// Empty records the device code as unknown (no embedded code objects).
/// `hc_mode` is the loaded HyperConnection mode (`--hc-mode`); each mode
/// computes different states, so none restores into another.
[[nodiscard]] std::vector<std::uint8_t> QwenFlashNextCompatibilityIdentity(
    std::string_view artifact_fingerprint, std::string_view mtp_fingerprint,
    bool has_mtp, std::uint32_t max_context, std::uint32_t max_draft_tokens,
    std::uint32_t decode_concurrency, std::string_view position_policy,
    bool native, std::uint32_t payload_version,
    std::string_view device_code_sha256, std::string_view hc_mode);

}  // namespace gufo::server

#endif  // GUFO_SERVER_PERSISTENCE_IDENTITY_HPP_
