# EXL3 weight format

This document describes the EXL3 checkpoint format (QTIP-style trellis codes with input and output
Hadamard rotations) as gufo reads it. It is written against upstream
[turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) v1.6.5, commit
`cdc785db89e263e65e5849cb53e1ada156411414` (MIT, Copyright 2025 Turboderp). References of the form
`path:line` point into that commit. The packs gufo has measured (`turboderp/Qwen3.8-27B-exl3`,
branches `3.00bpw`–`5.00bpw`) were written by exllamav3 1.4.2 and load unchanged in 1.6.5.

gufo's CPU reference implementation is `src/core/quant/exl3/`. Its tests compare it bit for bit
with the upstream runtime (`tests/core/quant/exl3_*`).

## Checkpoint layout

An EXL3 checkpoint is a Hugging Face model directory:

- `config.json` with a `quantization_config` object (top level, or inside the text config for
  multimodal models such as Qwen3.8 27B, which also carries `text_config`).
- `model.safetensors.index.json` and one or more `model-*.safetensors` shards.
- `quantization_config.json`: the same top-level keys plus `tensor_storage`, a map from module name
  to the stored tensors (name, shape, byte count, torch dtype) for every module, quantized or not.
  It is informational; the tensors themselves are authoritative.
- Tokenizer files as in the source model.

`quantization_config` keys (`exllamav3/conversion/compile.py:197-216`):

| Key | Meaning |
|---|---|
| `quant_method` | `"exl3"` |
| `version` | exllamav3 version that wrote the checkpoint (`"1.4.2"` for the turboderp 27B packs) |
| `bits` | target average bits per weight of the quantized body (for example `4.0`) |
| `head_bits` | bits of the output head (`lm_head`), `6` in the 27B packs |
| `mtp_bits` | bits of the MTP block, `4` in the 27B packs |
| `calibration` | `{"rows", "cols"}` of the calibration set (250 × 2048 in the 27B packs) |
| `out_scales` | `"always"`, `"never"` or `"auto"`: whether output-channel scales were folded into `svh` |
| `codebook` | `"mul1"`, `"mcg"` or absent (3INST); a default for the writer only, see "Codebooks" |

The per-tensor bit width is not stored as a key: it follows from the trellis shape (below). The
27B packs mix bit widths only at `3.50bpw` (about half of the body at K = 3 and half at K = 4); the
other packs use one K for the body.

## Tensors of one quantized linear layer

A quantized linear layer with `in_features = k` and `out_features = n` (computing `y = x · W` with
`W` of shape `[k, n]`) stores, under the module name `<m>` (`exllamav3/modules/quant/exl3.py:20-96`):

| Tensor | dtype | Shape | Meaning |
|---|---|---|---|
| `<m>.trellis` | I16 | `[k / 16, n / 16, 16 · K]` | trellis code of each 16 × 16 tile, `16 · K` uint16 words per tile |
| `<m>.suh` | F16 | `[k]` | input-side signs times scales |
| `<m>.svh` | F16 | `[n]` | output-side signs times scales (and output-channel scales) |
| `<m>.mul1` or `<m>.mcg` | I32 | `[]` | codebook marker; only its presence matters (`quantize.py:32-41`) |
| `<m>.bias` | F16 | `[n]` | optional bias |

- Older checkpoints may store `su`/`sv` (I16 bitfields, one sign bit per element, bit `i` of word
  `j` is element `16·j + i`, set bit means −1) instead of `suh`/`svh` (`exl3.py:147-164`). The 27B
  packs store `suh`/`svh`.
- **Bit width:** `K = trellis.shape[-1] / 16` (`exl3.py:66-69`). An integer K is 1–8. A
  half-integer K (1.5, 2.5, 3.5) is allowed only with the mul1 codebook
  (`exllamav3_ext/quant/bits_k.cuh:20-27`); its tiles hold `16 · K` words too (for example 56 at
  K = 3.5).
- `k` and `n` must be multiples of 128 (the Hadamard block size); upstream checks `n % 128` when
  reconstructing (`reconstruct.cu:168`).
- Codebook selection: a `mul1` marker selects mul1, a `mcg` marker selects mcg, no marker selects
  3INST (`exl3.py:77-81`). All 27B pack tensors carry `mul1`.

Tensors that are not quantized keep their source dtype (BF16 norms and embeddings, F16 small
projections). In the Qwen3.8 27B packs:

| Part | Storage |
|---|---|
| `model.language_model.embed_tokens.weight` | BF16, unquantized |
| norms, `A_log`, `dt_bias`, `conv1d.weight` | BF16 |
| `linear_attn.in_proj_a`, `in_proj_b` | F16 `[48, 5120]`, unquantized |
| other `layers.*` projections | EXL3 at the pack's body K |
| `lm_head` | EXL3, K = `head_bits` (6), `n = 248320` |
| `mtp.*` projections | EXL3, K = `mtp_bits` (4); `mtp.fc` is `[10240, 5120]` |
| `model.visual.*` | BF16, unquantized |

## Trellis code of one tile

A tile is 256 weights. Its `16 · K` uint16 words are read as `4 · K` little-endian uint32 words
(word `i` = uint16 `2i` | uint16 `2i + 1` << 16). The tile is a cyclic bit string of `256 · K`
bits, numbered from the most significant bit of uint32 word 0 (`exllamav3_ext/quant/exl3_dq.cuh:15-31`).

**Positions.** The 256 positions `p = 0 … 255` are decoded in order. Position `p` takes `b(p)` new
bits:

- integer K: `b(p) = K`;
- half-integer K = KA + 0.5: `b(p) = KA` for even `p` and `KA + 1` for odd `p` (mask `0xAAAA`,
  `quantize.py:103-112`, `exl3_dq.cuh:253-283`).

Let `E(p) = b(0) + … + b(p)` be the end bit of position `p`. The **state** of position `p` is the
16-bit window of the cyclic bit string that ends at bit `E(p)` (exclusive): bits
`E(p) − 16 … E(p) − 1`, taken modulo `256 · K`, most significant bit first. Upstream computes the
same window with a funnel shift over the two uint32 words that hold it (`exl3_dq.cuh:18-29`). Because
the bit string is cyclic, the first positions' windows wrap around to the end of the tile.

**Codebook value.** Each 16-bit state `x` decodes to an FP16 value with one of three procedural
codebooks (`exllamav3_ext/quant/codebook.cuh`). All arithmetic is FP16 with round to nearest even
(`__hadd` and `__hfma`, fused, single rounding):

Below, `bits(v)` reinterprets a 16-bit pattern as FP16.

- **3INST** (no marker, `codebook.cuh:69-76`):
  `x = x · 89226354 + 64248484` (uint32, wrapping), then
  `x = (x & 0x8fff8fff) ^ 0x3b603b60`, and the value is `bits(x & 0xffff) + bits(x >> 16)`.
- **mcg** (`codebook.cuh:77-85`): `x = x · 0xCBAC1FED` (uint32, wrapping), then the same mask, xor
  and FP16 sum of the two halves.
- **mul1** (`codebook.cuh:86-99`): `x = x · 0x83DCD12D` (uint32, wrapping); `s` = the sum of the four
  bytes of `x` plus `0x6400`; `bits(s)` is exactly `1024 + bytesum` (1024 … 2044); the value is
  `fma(bits(s), bits(0x1eee), bits(0xc931))`, rounded once (about `(bytesum − 510) / 147.7`).

The 16-bit state is zero-extended to 32 bits before the multiply.

**Tile layout.** Position `p` holds the tile element at row `r` and column `c` given by the
tensor-core permutation (`quantize.py:44-66`): with `t = p / 8` and `j = p % 8`,
`r0 = (t % 4) · 2`, `c0 = t / 4`, rows `(r0, r0 + 1, r0 + 8, r0 + 9)` for `j = 0 … 3` and again for
`j = 4 … 7`, column `c0` for `j < 4` and `c0 + 8` for `j ≥ 4`. The element is `W[16 · tk + r][16 · tn + c]`
for the tile at `trellis[tk][tn]` (`reconstruct.cu:12-85`).

## Rotations

The trellis decodes to the inner matrix `W'` (`[k, n]`, FP16). The effective weight is
(`exl3.py:272-282`, Hadamard construction `exllamav3/util/hadamard.py:34-43,108-132`):

```text
W = diag(suh) · H_k · W' · H_n · diag(svh)
```

- `H_k` and `H_n` are block-diagonal with 128 × 128 blocks `H = S_128 / sqrt(128)`, where `S_128`
  is the Sylvester Hadamard matrix in natural order: `S[i][j] = (−1)^popcount(i & j)`. (128 is not
  among the stored Paley matrices, so `get_hadamard(128)` recurses Sylvester from 1.)
- In the forward pass this is `y = ((x ⊙ suh) · H_k) · W' · H_n ⊙ svh`: the input is scaled by
  `suh`, rotated per 128-element block, multiplied by `W'`, and the output is rotated per
  128-element block and scaled by `svh`. The runtime does exactly this in FP16
  (`had_r_128(x, xh, suh, None, 1.0)` before and `had_r_128(y, y, None, svh, 1.0)` after the inner
  GEMM, `exl3.py:166-224`; `hadamard.cu:144` uses the scale `1 / sqrt(128) = 0.088388347648`).
- `get_weight_tensor` builds `W` in FP32 steps with an FP16 rounding after each step: `H_k · W'` →
  FP16 → `· suh` (row scale) → FP16 → `· H_n` → FP16 → `· svh` → FP16. Its result therefore differs
  from an exact product in the last FP16 bit for some elements; the FP32 summation order of the
  GPU matmul also affects those roundings. gufo compares the inner `W'` bit for bit and `W` or `y`
  within a relative tolerance.

## What gufo checks

| Check | Reference | Criterion |
|---|---|---|
| inner dequant `W'` for every K (1–8), codebook (3INST, mcg, mul1) and half-integer K (1.5, 2.5, 3.5, mul1) | upstream `ext.reconstruct` on random trellis words | FP16 bits identical |
| inner dequant of real 27B tensors (K = 3, 4, 5 body, K = 6 head, K = 4 MTP) | upstream `ext.reconstruct` | FP16 bits identical |
| `y = x · W` | upstream `get_weight_tensor` and the FP16 runtime path | FP32 accumulation, relative error near 1e-6 against an FP64 evaluation of the same formula; upstream values within their FP16 rounding |

Any 16-bit state decodes, so random trellis words are valid test inputs; no quantizer run is
needed for the decode tests.
