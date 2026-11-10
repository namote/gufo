# Optional runtime abliteration

Implementation, independent code review and the focused CPU checks have
passed. No qualified profile exists yet; GPU/model validation and behavioral
evaluation are required before deployment. This is a reversible inference
intervention, not a weight conversion. It mirrors the
[Flash-Next design](../qwen3.8-flash-next/ABLITERATION.md); this page records
what differs for the dense Qwen3.8-27B (`qwen35`, GGUF or native EXL3 pack).

## Ownership and execution

- `src/models/qwen/abliteration.{hpp,cpp}` owns the versioned, immutable,
  model-bound profile, its cache identity and the continuation record. Flash-Next
  binds the same code to its own placement. It has no HIP dependency.
- `QwenGpuModel` loads one profile once and keeps its direction on the GPU.
  Each session (`QwenGpuExecutor`) owns its ON/OFF state; there is no
  server-global mutable toggle.
- Placement `writer-before-residual`: the hook projects the attention or
  GatedDeltaNet output projection and the MLP down projection, each
  immediately before its residual add (27B has no hyper-connections). It does
  not project the accumulated residual, attention probabilities or logits.
  DFlash2 drafts stay unmodified; target verification uses the selected
  intervention.

| Path | Attention/GDN writer | MLP writer |
| --- | --- | --- |
| Decode (graph-captured) | `hip/decode_step.cpp`, `attention_scratch.output` | `hip/decode_step.cpp`, `ffn_scratch.out` |
| Prompt chunk | `hip/prefill_chunk.cpp`, `d_attn_out` | `hip/prefill_chunk.cpp`, `d_ffn_out` |
| Batched decode, 2-8 sessions | `hip/batched_decode.cpp` `ForwardTokenBatch` | same |
| Speculative verification, 1-8 sessions | `hip/batched_decode.cpp` `ForwardVerificationBatch` | same |

  Committing verified rows replays recorded recurrent inputs or reruns
  `ForwardToken`, so both stay within the selected mode.
- The FP16 prompt path fuses both residual adds into its GEMMs. ON prompt
  chunks take the FP32 path instead, which can lose that speedup for Q8/K-quant
  GGUFs at 1024+ rows (OFF collection of such prompts still reads the FP16
  path's residuals). Actual end-to-end overhead remains unmeasured.
- A zero layer strength launches no projection. With no profile or every
  session OFF, nothing launches. One masked kernel covers up to eight sessions
  at each writer, in place, without a second activation tensor.
- A mode change resets the session (the server first invalidates the slot,
  including its draft) and drops its captured decode graph. An ON decode graph
  key is extended by a mode bit; OFF keeps the original key. The profile
  identity salts the text and every image-prefix cache identity, so ON/OFF or
  different profiles never share a state. Continuation metadata records the
  mode in four bytes every earlier capture zero-filled, and ON salts the image
  identity with the full profile identity; a mismatch is a cache miss. OFF
  records exactly the earlier bytes, so existing caches remain valid.

The GPU operation is `y -= strength * direction * dot(direction, y)` with an
FP32 reduction and unit direction.

## Profile and API

A profile is a small JSON file with schema `gufo-abliteration-v1`, placement
`writer-before-residual`, `model_identity` (the artifact content identity, as
for persistent caches), `hidden` (5120), `layers` (64), `direction`, and
per-layer `attention` (attention and GDN layers alike) and `mlp` strengths.
Flash-Next profiles (`writer-before-hc`) and profiles of another artifact are
rejected, as are nonfinite values, nonunit vectors, incompatible dimensions and
files over 1 MiB.

```sh
gufo serve llm --model models/qwen3.8-27b-exl3 --abliteration-profile PROFILE.json
```

The option is the only way to load one; nothing beside the model is
discovered, and loading never enables it. Requests select it exactly as for
Flash-Next (Chat Completions, Responses, Messages):

```json
{"abliteration": {"enabled": true, "profile": "default"}}
```

## Prepare once, retain receipts

The native collector is the explicit CMake target
`qwen27b_abliteration_collect` (`tools/qwen27b/abliteration/collect.cpp`). It
loads the target alone (no draft, no vision encoder, no HTTP server), uses the
native template with medium reasoning, and rejects prompts over 2048 tokens.
`fit` and data import use the shared
`tools/qwen38_flash_next/abliteration/prepare.py`; it reads hidden size, layer
count, model identity and placement from the capture.

```sh
COLLECT=build/obj/tests/models/qwen27b/qwen27b_abliteration_collect
PREP=tools/qwen38_flash_next/abliteration/prepare.py
MODEL=/models/q27   # the EXL3 5.00bpw pack directory
python3 $PREP import-data --archive KYOJIN.tar.gz --output DATA
$COLLECT $MODEL DATA/collection.jsonl CAPTURE collect
python3 $PREP fit --collection CAPTURE --output CANDIDATES
$COLLECT $MODEL DATA/tuning.jsonl OFF evaluate --max-new-tokens 1024
$COLLECT $MODEL DATA/tuning.jsonl ON-L38-S1 evaluate --max-new-tokens 1024 \
  --profile CANDIDATES/layer38-strength1.json
```

`collect` records, per prompt, the residual of the final prompt token after
each of the 64 layers (FP32, 64 x 5120). `evaluate` stores first-next-token
logits, greedy text and reasoning/truncation flags. Receipts, resume checks and
the selection rules (tuning only, then untouched evaluation inputs; an
incomplete answer is inconclusive) are those of the Flash-Next workflow.

## Focused checks

CPU: `qwen27b_abliteration_test` (profile binding, placement, OFF metadata
bytes, mode and profile mismatch), `openai_chat_test` and
`python3 tools/qwen38_flash_next/abliteration/test_prepare.py`.
GPU: `qwen27b_abliteration_ops_test` covers an independent FP64 formula,
unaligned widths, mixed ON/OFF rows and the writer-then-residual order. Run it
with the final combined GPU checks: OFF preservation, ON/OFF C2, graph replay,
batched decode, DFlash2 verification, text/image continuation, cache
reload/fork and per-request timings.
