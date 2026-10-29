# Rotated weights (GGUF marker rules v1)

A rotated GGUF stores some matrices with a fixed orthogonal transform folded
into their input side, so the same quantization type loses less precision on
outlier-heavy inputs. Gufo detects rotation from the file; there is no
command-line option. Files are produced by `tools/rotquant/convert.py`.

## Transform

For a matrix `W` with input dimension `K` (GGUF `ne0`, a multiple of 128):

```text
stored:  W' = W·D·H
runtime: x' = H·D·x        so  W'·x' = W·x   (H·H = I, D·D = I)
```

- `D` is a ±1 diagonal sign vector stored in the file.
- `H` is block diagonal: `K / 128` normalized 128-point Sylvester Hadamard
  blocks (`H[i][j] = (-1)^popcount(i & j) / sqrt(128)`).
- Matrices that read the same activation form an input group and share one
  `D`, so the activation is rotated once. A group instance is one group in one
  layer; different layers use different signs.
- Matrix outputs are unchanged, so residual streams, KV caches, recurrent
  state and speculative-decoding features are those of the unrotated model.

## Markers

A rotated file has all four global keys and at least one sign tensor:

| Key | Type | Value |
|---|---|---|
| `gufo.rotation.version` | u32 | `1` |
| `gufo.rotation.kind` | string | `"hadamard"` |
| `gufo.rotation.block_size` | u32 | `128` |
| `gufo.rotation.side` | string | `"input"` |

- Every rotated tensor `<name>` has a one-dimensional F16 tensor
  `<name>.rot` with `K` values, each exactly `+1` (`0x3C00`) or `-1`
  (`0xBC00`).
- `gufo.rotation.source_*` keys record converter provenance. They are not
  markers and may appear in unrotated files.
- `general.architecture` and all hyperparameter keys are unchanged. Other
  runtimes see extra tensors and should refuse the file.

## Loader rules

`src/core/quant/rotation.hpp` checks the markers against a group table that
the model supplies. The load fails, naming the key, tensor or group, when:

- a `version`, `kind`, `block_size` or `side` value is missing or different
  from the table above;
- `.rot` tensors exist without the markers, or the markers without `.rot`
  tensors;
- a `.rot` tensor has no base tensor, is not F16 and one-dimensional, has a
  length other than the base `K`, holds a value other than ±1, or `K` is not a
  multiple of 128;
- the base tensor belongs to no input group of the model (no rotation path
  exists for its role), or to a group the runtime cannot rotate yet;
- the base tensor is an embedding that also serves as a tied output head;
- one group instance mixes rotated and unrotated matrices, or its matrices
  use different sign vectors;
- a group is rotated in some layers but not all.

Execution paths without rotation support refuse a rotated file instead of
falling back.

## Qwen3.8 27B

Input groups (hidden 5120):

| Group | Activation | Matrices |
|---|---|---|
| `attn_in` | `attn_norm` output | `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta` / `attn_q`, `attn_k`, `attn_v` |
| `ssm_out_in` | SSM per-head norm and gate | `ssm_out` |
| `attn_out_in` | gated attention output | `attn_output` |
| `ffn_in` | `post_attention_norm` output | `ffn_gate`, `ffn_up` |
| `ffn_down_in` | SwiGLU output | `ffn_down` |
| `head_in` | `output_norm` output | `output` |

- Only main-stack layers belong to groups. Embedded MTP (`nextn`) layers stay
  unrotated, and a `.rot` on them is refused.
- The HIP backend rotates every group on decode, prefill, multi-session
  decode and DFlash2 verification, and the DFlash2 draft head rotates its
  input before the shared target `output` head.
- The `--cpu` backend refuses rotated files. MTP drafting refuses a rotated
  `head_in`.
