#include <array>
#include <iostream>
#include <stdexcept>

#include "src/cli/serve/text_model_runner.hpp"

namespace {
void Require(bool ok) {
  if (!ok)
    throw std::runtime_error(
        "row-block admission disagrees with physical capacity");
}
}  // namespace
int main() {
  using Pool = gufo::server::TextModelRunner::RowBlockPool;
  const Pool guarded{.total_blocks = 2050,
                     .free_blocks = 0,
                     .block_rows = 256,
                     .row_guard_rows = 4,
                     .checkpoint_blocks_retained = false};
  const std::array<std::pair<std::size_t, std::size_t>, 9> frontiers{
      {{0, 0},
       {1, 1},
       {252, 1},
       {253, 2},
       {255, 2},
       {256, 2},
       {513, 3},
       {262143, 1025},
       {262144, 1025}}};
  for (const auto [rows, expected] : frontiers)
    Require(guarded.MappedBlocks(rows) == expected);
  Require(guarded.AdmissionBlocks(513, 513, 2) == 3);
  Require(guarded.AdmissionBlocks(1, 262144, 2) == 3);
  Require(guarded.AdmissionBlocks(262144, 262144, 2) == 1025);
  // Two full-capacity resident slots leave no free blocks, but either can
  // finish its final token with the guard block it already maps.
  Require(guarded.AdmissionBlocks(262144, 262144, 2) -
              guarded.MappedBlocks(262143) ==
          0);
  const Pool plain{.total_blocks = 2048, .free_blocks = 0, .block_rows = 256};
  Require(plain.MappedBlocks(0) == 0 && plain.MappedBlocks(253) == 1 &&
          plain.MappedBlocks(256) == 1 && plain.MappedBlocks(257) == 2);
  Require(plain.AdmissionBlocks(262144, 262144, 2) == 1024);
  Require(plain.checkpoint_blocks_retained &&
          !guarded.checkpoint_blocks_retained);
  Require(Pool{}.AdmissionBlocks(42, 256, 2) == 0);
  std::cout << "row-block admission boundaries: PASS\n";
}
