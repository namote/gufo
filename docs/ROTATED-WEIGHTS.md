# Rotated weights (`prism.hadamard` GGUF metadata)

A rotated GGUF stores some matrices with a fixed orthogonal transform folded
into their input side, so the same quantization type loses less precision on
outlier-heavy inputs. Gufo uses the upstream llama.cpp format
(`prism.hadamard.*`, llama.cpp #29600, merged as `a9d27ac6`) and detects
rotation from the file; there is no command-line option. Files are produced
by `tools/rotquant/convert.py`.

> **Warning.** llama.cpp before `a9d27ac6` and other runtimes that do not
> read `prism.hadamard.*` load a rotated file as an ordinary model and
> produce garbage without an error: the file has exactly the tensors of the
> unrotated model. Use llama.cpp `a9d27ac6` or later, and keep the rotation
> visible in file names and release notes.

## Transform

For a folded matrix `W` with input dimension `K` (GGUF `ne0`, a multiple of
128):

```text
stored:  W' = W·D·H
runtime: x' = H·D·x        so  W'·x' = W·x   (H·H = I, D·D = I)
```

- `D` is the ±1 sign vector of the input width `K`. A file stores one vector
  per width, shared by every folded matrix with that `K`.
- `H` is block diagonal: `K / 128` normalized 128-point Sylvester Hadamard
  blocks (`H[i][j] = (-1)^popcount(i & j) / sqrt(128)`).
- Matrix outputs are unchanged, so residual streams, KV caches, recurrent
  state and speculative-decoding features are those of the unrotated model.

## Metadata

| Key | Type | gufo accepts |
|---|---|---|
| `prism.hadamard.version` | u32 | `1` |
| `prism.hadamard.block_size` | u32 | `128` |
| `prism.hadamard.transform` | string | `"normalized-sylvester-walsh-hadamard"` |
| `prism.hadamard.axis` | string | `"input-last-dimension"` |
| `prism.hadamard.sign_mode` | string | `"explicit"` |
| `prism.hadamard.sign_widths` | i32 array | one entry per folded input width |
| `prism.hadamard.sign_values` | i32 array | the ±1 vectors, concatenated in width order |
| `prism.hadamard.weight_names` | string array | the folded matrices |
| `prism.hadamard.tied_output`, `inverse_weight_names`, `gdn_v_grouped` | | refused |

`gufo.rotation.source_*` keys record converter provenance and are not
rotation metadata.

## Loader rules

`src/core/quant/rotation.hpp` refuses a file, naming the key, tensor, group
or rule, in two stages:

1. **Upstream rules**, in the order of llama.cpp `load_hparams_hadamard` and
   `load_tensors_hadamard`; each message carries the upstream source line
   (`llama.cpp 8ab29fae src/llama-model.cpp:<line>`). Examples: unknown
   version, invalid block size, transform, axis or sign mode, empty, repeated
   or non-±1 sign data, an architecture outside the verified list, a weight
   that is not on a Hadamard-aware matmul path, a listed weight missing from
   the file, a width without a sign vector.
2. **The subset gufo executes:** untied version 1, block 128, explicit signs,
   no inverse lookup tables, no GDN permutation, every folded weight in an
   input group of the model, each group folded in all of its layers or in
   none, and no unused sign width.

The retired gufo v1 format (`gufo.rotation.*` markers and `<name>.rot`
tensors) is refused with a request to reconvert. Execution paths without
rotation support refuse a rotated file instead of falling back.

## Qwen3.8 27B

Input groups (hidden 5120):

| Group | Activation | Folded matrices | Width |
|---|---|---|---:|
| `attn_in` | `attn_norm` output | `attn_qkv`, `attn_gate` / `attn_q`, `attn_k`, `attn_v` | 5120 |
| `ffn_in` | `post_attention_norm` output | `ffn_gate`, `ffn_up` | 5120 |
| `head_in` | `output_norm` output | `output` | 5120 |
| `ssm_out_in` | SSM per-head norm and gate | `ssm_out` | 6144 |
| `attn_out_in` | gated attention output | `attn_output` | 6144 |
| `ffn_down_in` | SwiGLU output | `ffn_down` | 17408 |

- Only main-stack layers belong to groups. Embedded MTP (`nextn`) layers stay
  unrotated, and folding one of their weights is refused.
- `ssm_alpha` and `ssm_beta` read the `attn_in` activation but are not
  foldable upstream, so the file keeps them unrotated. When `attn_in` is
  rotated, gufo folds them at load (`W·D·H` with the 5120-width signs) and
  re-encodes them in their own type (F32, BF16 or Q8_0), so every DeltaNet
  input projection reads the one rotated activation. Q8_0 is therefore
  quantized twice and BF16 rounded twice; the numerics differ slightly from
  llama.cpp, which keeps the original activation for these two matrices.
- The HIP backend rotates every group on decode, prefill, multi-session
  decode and DFlash2 verification, and the DFlash2 draft head rotates its
  input before the shared target `output` head. (Upstream DFlash borrows the
  target head without the transform, so its drafts degrade with a rotated
  head.)
- The `--cpu` backend refuses rotated files. MTP drafting refuses a rotated
  `head_in`.
