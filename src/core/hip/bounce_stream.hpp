#ifndef GUFO_CORE_HIP_BOUNCE_STREAM_HPP_
#define GUFO_CORE_HIP_BOUNCE_STREAM_HPP_

#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>

namespace gufo::hip {

/// `rows` rows of `width` bytes in device memory, `pitch` bytes apart.
/// Serialized form packs the rows back to back.
struct DeviceRegion {
  const void* data{};
  std::size_t pitch{}, width{}, rows{};
  [[nodiscard]] std::size_t PackedBytes() const noexcept {
    return width * rows;
  }
};
struct MutableDeviceRegion {
  void* data{};
  std::size_t pitch{}, width{}, rows{};
  [[nodiscard]] std::size_t PackedBytes() const noexcept {
    return width * rows;
  }
};

/// Moves device regions to and from a byte stream through a small ring of
/// pinned host buffers, so serializing a multi-GB session needs no full-size
/// host or device copy. DMA copies run up to kSlots - 1 pieces ahead of the
/// caller, so a copy queued behind other GPU work rarely stalls the stream. Not thread-safe; callers own one instance per concurrent stream.
/// The caller keeps every source region unchanged until Out returns.
class BounceStream {
public:
  using Sink = std::function<void(std::span<const std::uint8_t>)>;
  /// Fills the whole span with the next bytes of the stream.
  using Source = std::function<void(std::span<std::uint8_t>)>;

  explicit BounceStream(std::size_t buffer_bytes = kDefaultBufferBytes);
  ~BounceStream();
  BounceStream(const BounceStream&) = delete;
  BounceStream& operator=(const BounceStream&) = delete;

  void Out(const DeviceRegion& region, const Sink& sink);
  void In(const MutableDeviceRegion& region, const Source& source);
  [[nodiscard]] std::size_t buffer_bytes() const noexcept { return bytes_; }

  static constexpr std::size_t kDefaultBufferBytes = 32ULL << 20;
  static constexpr std::size_t kSlots = 4;

private:
  struct Piece {
    std::size_t row{}, column{}, rows{}, width{};
    [[nodiscard]] std::size_t Bytes() const noexcept { return rows * width; }
  };
  void Release() noexcept;
  [[nodiscard]] Piece Next(std::size_t width, std::size_t rows,
                           std::size_t* row, std::size_t* column) const;

  std::size_t bytes_;
  hipStream_t stream_{};
  std::array<std::uint8_t*, kSlots> buffers_{};
  std::array<hipEvent_t, kSlots> done_{};
};

}  // namespace gufo::hip

#endif  // GUFO_CORE_HIP_BOUNCE_STREAM_HPP_
