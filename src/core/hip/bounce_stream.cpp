#include "src/core/hip/bounce_stream.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace gufo::hip {
namespace {
void Check(hipError_t status, const char* what) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " +
                             hipGetErrorString(status));
}
}  // namespace

BounceStream::BounceStream(std::size_t buffer_bytes) : bytes_(buffer_bytes) {
  if (bytes_ == 0)
    throw std::invalid_argument("bounce buffer must not be empty");
  try {
    Check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
          "bounce stream creation failed");
    for (std::size_t i = 0; i < buffers_.size(); ++i) {
      void* data = nullptr;
      Check(hipHostMalloc(&data, bytes_, hipHostMallocDefault),
            "bounce buffer allocation failed");
      buffers_[i] = static_cast<std::uint8_t*>(data);
      Check(hipEventCreateWithFlags(&done_[i], hipEventDisableTiming),
            "bounce event creation failed");
    }
  } catch (...) {
    Release();
    throw;
  }
}

BounceStream::~BounceStream() { Release(); }

void BounceStream::Release() noexcept {
  if (stream_ != nullptr)
    (void)hipStreamSynchronize(stream_);
  for (std::size_t i = 0; i < buffers_.size(); ++i) {
    if (done_[i] != nullptr)
      (void)hipEventDestroy(done_[i]);
    if (buffers_[i] != nullptr)
      (void)hipHostFree(buffers_[i]);
    done_[i] = nullptr;
    buffers_[i] = nullptr;
  }
  if (stream_ != nullptr)
    (void)hipStreamDestroy(stream_);
  stream_ = nullptr;
}

// Whole rows while they fit in one buffer; otherwise one row in column
// pieces. Either way the pieces follow the packed byte order.
BounceStream::Piece BounceStream::Next(std::size_t width, std::size_t rows,
                                       std::size_t* row,
                                       std::size_t* column) const {
  Piece piece{.row = *row, .column = *column};
  if (width <= bytes_) {
    piece.width = width;
    piece.rows = std::min(rows - *row, bytes_ / width);
    *row += piece.rows;
    return piece;
  }
  piece.rows = 1;
  piece.width = std::min(width - *column, bytes_);
  *column += piece.width;
  if (*column == width) {
    *column = 0;
    ++*row;
  }
  return piece;
}

void BounceStream::Out(const DeviceRegion& region, const Sink& sink) {
  if (region.width == 0 || region.rows == 0)
    return;
  if (region.data == nullptr || region.pitch < region.width)
    throw std::invalid_argument("bounce source region is invalid");
  const auto* base = static_cast<const std::uint8_t*>(region.data);
  std::array<Piece, kSlots> queued{};
  std::size_t row = 0, column = 0, issued = 0, drained = 0;
  const auto drain = [&] {
    const auto slot = drained % kSlots;
    Check(hipEventSynchronize(done_[slot]), "bounce copy failed");
    sink({buffers_[slot], queued[slot].Bytes()});
    ++drained;
  };
  while (row < region.rows) {
    // A slot is refilled only after the sink consumed its previous piece.
    if (issued - drained == kSlots)
      drain();
    const auto slot = issued % kSlots;
    const auto piece = Next(region.width, region.rows, &row, &column);
    Check(hipMemcpy2DAsync(buffers_[slot], piece.width,
                           base + piece.row * region.pitch + piece.column,
                           region.pitch, piece.width, piece.rows,
                           hipMemcpyDeviceToHost, stream_),
          "bounce device-to-host copy failed");
    Check(hipEventRecord(done_[slot], stream_), "bounce event record failed");
    queued[slot] = piece;
    ++issued;
  }
  while (drained < issued)
    drain();
}

void BounceStream::In(const MutableDeviceRegion& region,
                      const Source& source) {
  if (region.width == 0 || region.rows == 0)
    return;
  if (region.data == nullptr || region.pitch < region.width)
    throw std::invalid_argument("bounce destination region is invalid");
  auto* base = static_cast<std::uint8_t*>(region.data);
  std::size_t row = 0, column = 0, index = 0;
  while (row < region.rows) {
    const auto slot = index % kSlots;
    const auto piece = Next(region.width, region.rows, &row, &column);
    // Refill a slot only after its previous upload finished reading it.
    if (index >= kSlots)
      Check(hipEventSynchronize(done_[slot]), "bounce upload failed");
    source({buffers_[slot], piece.Bytes()});
    Check(hipMemcpy2DAsync(base + piece.row * region.pitch + piece.column,
                           region.pitch, buffers_[slot], piece.width,
                           piece.width, piece.rows, hipMemcpyHostToDevice,
                           stream_),
          "bounce host-to-device copy failed");
    Check(hipEventRecord(done_[slot], stream_), "bounce event record failed");
    ++index;
  }
  Check(hipStreamSynchronize(stream_), "bounce upload failed");
}

}  // namespace gufo::hip
