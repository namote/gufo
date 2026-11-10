#ifndef GUFO_CORE_HAN_SCRIPT_HPP_
#define GUFO_CORE_HAN_SCRIPT_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "src/core/sampling.hpp"

namespace gufo::sampling {

/// CJK ideographs: Unified (U+4E00-9FFF), Extension A (U+3400-4DBF),
/// Compatibility (U+F900-FAFF) and Extensions B-H (U+20000-323AF). Hangul,
/// kana, CJK punctuation (U+3000-303F), radicals and fullwidth forms are not
/// Han.
[[nodiscard]] constexpr bool IsHanCodePoint(std::uint32_t code) noexcept {
  return (code >= 0x3400 && code <= 0x4DBF) ||
         (code >= 0x4E00 && code <= 0x9FFF) ||
         (code >= 0xF900 && code <= 0xFAFF) ||
         (code >= 0x20000 && code <= 0x323AF);
}

/// Whether a token's decoded bytes can write a Han character: they contain a
/// complete Han code point, or end in an incomplete UTF-8 sequence that can
/// only begin one (the token commits the text to Han). Byte-level and byte
/// fallback tokens that end in such a prefix count: a lone lead byte E4-E9
/// (E4 can also begin the 64 U+4DC0-4DFF hexagrams; Han dominates it),
/// E3 9x-Bx (Extension A), EF A4-AB (Compatibility) and F0 A0-B2 (Extensions
/// B-H). Continuation bytes without their lead never count; their lead token
/// already made the choice, and they also end Hangul or kana characters.
[[nodiscard]] constexpr bool IsHanToken(std::string_view bytes) noexcept {
  std::size_t i = 0;
  while (i < bytes.size()) {
    const auto lead = static_cast<unsigned char>(bytes[i]);
    const unsigned length = lead >= 0xC2 && lead <= 0xDF   ? 2
                            : lead >= 0xE0 && lead <= 0xEF ? 3
                            : lead >= 0xF0 && lead <= 0xF4 ? 4
                                                           : 1;
    if (length == 1) {  // ASCII, a stray continuation or an invalid byte
      ++i;
      continue;
    }
    std::uint32_t code = lead & (0xFFU >> (length + 1));
    std::size_t have = 1;
    while (have < length && i + have < bytes.size()) {
      const auto next = static_cast<unsigned char>(bytes[i + have]);
      if ((next & 0xC0U) != 0x80U)
        break;
      code = (code << 6U) | (next & 0x3FU);
      ++have;
    }
    if (have == length) {
      if (IsHanCodePoint(code))
        return true;
      i += length;
      continue;
    }
    if (i + have < bytes.size()) {  // malformed: resume after the lead byte
      ++i;
      continue;
    }
    // An incomplete final sequence: does every completion start a Han code
    // point? `code` holds the lead and `have - 1` continuation payloads.
    const auto second = have > 1 ? static_cast<unsigned char>(bytes[i + 1]) : 0;
    if (length == 3) {
      if (have == 1)
        return lead >= 0xE4 && lead <= 0xE9;
      // U+3400-3FFF, U+4000-9FFF except the U+4DC0-4DFF hexagrams (E4 B7),
      // or U+F900-FAFF.
      return (lead == 0xE3 && second >= 0x90) ||
             (lead == 0xE4 && second != 0xB7) ||
             (lead >= 0xE5 && lead <= 0xE9) ||
             (lead == 0xEF && second >= 0xA4 && second <= 0xAB);
    }
    if (length == 4 && have >= 2)
      return lead == 0xF0 && second >= 0xA0 && second <= 0xB2;
    return false;
  }
  return false;
}

/// Builds the Han set of a vocabulary from each token's decoded bytes.
template<typename DecodeToken>
[[nodiscard]] std::shared_ptr<const TokenSet> BuildHanTokens(
    std::size_t vocabulary_size, DecodeToken&& decode) {
  auto set = std::make_shared<TokenSet>();
  set->words.assign((vocabulary_size + 31) / 32, 0);
  for (std::size_t token = 0; token < vocabulary_size; ++token) {
    const std::string text = decode(static_cast<TokenId>(token));
    if (IsHanToken(text)) {
      set->words[token / 32] |= std::uint32_t{1} << (token % 32);
      ++set->count;
    }
  }
  return set;
}

}  // namespace gufo::sampling

#endif  // GUFO_CORE_HAN_SCRIPT_HPP_
