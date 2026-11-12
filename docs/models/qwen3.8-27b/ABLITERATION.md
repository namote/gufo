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
- `QwenGpuModel` loads one profile once and keeps all its directions on the
  GPU (64 x 5120 floats, 1.25 MiB, for a rank-1 v2 profile; 20 MiB at rank 8
  with both direction kinds). Each session
  (`QwenGpuExecutor`) owns its effective settings (OFF, or a direction layer
  and per-layer strengths) and, from its first ON request, a device copy of
  them (1 + 3 x 64 = 193 words); there is no server-global mutable toggle.
- Placement `writer-before-residual` (the name predates residual strengths)
  has two kinds of hook. Writer strengths (`attention`, `mlp`) project the
  attention or GatedDeltaNet output projection and the MLP down projection,
  each immediately before its residual add (27B has no hyper-connections).
  Residual strengths (`residual`) project the residual itself right after the
  layer's MLP residual add, before the next layer, the DFlash2 feature tap,
  the collector's observer or the head reads it. Neither projects attention
  probabilities or logits. DFlash2 drafts stay unmodified; target
  verification uses the selected intervention.

| Path | Attention/GDN writer | MLP writer | Post-layer residual |
| --- | --- | --- | --- |
| Decode (graph-captured) | `hip/decode_step.cpp`, `attention_scratch.output` | `hip/decode_step.cpp`, `ffn_scratch.out` | `hip/decode_step.cpp`, `decode_scratch.hidden` |
| Prompt chunk | `hip/prefill_chunk.cpp`, `d_attn_out` | `hip/prefill_chunk.cpp`, `d_ffn_out` | `hip/prefill_chunk.cpp`, `d_hidden` |
| Batched decode, 2-8 sessions | `hip/batched_decode.cpp` `ForwardTokenBatch` | same | same, `scratch.decode.hidden` |
| Speculative verification, 1-8 sessions | `hip/batched_decode.cpp` `ForwardVerificationBatch` | same | same, `scratch.decode.hidden` |

  Committing verified rows replays recorded recurrent inputs or reruns
  `ForwardToken`, so both stay within the selected mode.
- The FP16 prompt path fuses both residual adds into its GEMMs. ON prompt
  chunks take the FP32 path instead (so the residual projection always reads
  the FP32 residual after the separate MLP add, and the next layer's fused
  norm-quantize reads the projected rows), which can lose that speedup for Q8/K-quant
  GGUFs at 1024+ rows (OFF collection of such prompts still reads the FP16
  path's residuals). Actual end-to-end overhead remains unmeasured.
- With no profile or every session OFF, nothing launches. An ON session
  launches at every writer, also at strength zero (the kernel returns at
  once). The residual projection launches only for a session with a nonzero
  residual strength, then at every layer (zero-strength layers return at
  once); writer-only ON settings therefore launch exactly what they did
  before residual strengths existed. One masked kernel covers up to eight sessions at each writer, in
  place, without a second activation tensor; each row reads its session's
  direction row and strength from that session's device settings.
- Any settings change resets the session (the server first invalidates the
  slot, including its draft). An ON graph key is extended by a mode value (1
  writer-only, 3 with residual projections) and OFF keeps the original key; direction and strength changes are written in
  place to the session's device settings (an upload on its stream, then a
  synchronization), which the decode graph reads when it runs, so no key
  carries their values. Unlike Flash-Next, the 27B session reset itself drops
  the captured decode graph (as for every reset), so the next decode records
  it once again. The settings identity (SHA-256, `gufo-abliteration-settings-v3`,
  over the profile identity, the direction layer, the exact float bytes of
  every attention, MLP and residual strength, -0 folded into +0, the direction
  kind, the direction count and norm preservation)
  salts the text and every image-prefix cache identity, so ON/OFF or different
  settings never share a state. Continuation metadata records the mode in four
  bytes every earlier capture zero-filled, and ON salts the image identity
  with the settings identity; a mismatch is a cache miss. OFF records exactly
  the earlier bytes and allocates, uploads or launches nothing new, so
  existing caches remain valid.

The GPU operation is `y -= strength * sum_j d_j * dot(d_j, y)` over the
session's k orthonormal directions (k = 1 by default) with FP32 reductions,
optionally followed by rescaling the row to its former norm, on a writer
output or on the residual. Direction kinds (`mean_diff`, `projected`),
subspaces (`directions`) and norm preservation (`preserve_norm`) are those of
[Flash-Next](../qwen3.8-flash-next/ABLITERATION.md#direction-kinds-subspaces-and-norm-preservation),
shared code included; one direction without norm preservation is
bit-identical to the single-direction kernel. The kind, count and norm flag
live in word 0 of the session's device settings, so changing them re-records
nothing.

### Writer or residual strengths

The collector measures each direction on the residual after a layer, but
writer strengths act on what the layers add to it. They cannot remove the
component the residual already carries (from the token embedding and earlier
layers), so writer strength 1 under-removes, and pushing a late-layer
direction (59 of 64) to strength 2 on every writer collapsed the model
(`専門専門`, 21 of 24 ordinary answers empty). Residual strengths are the
canonical directional ablation: `r -= s * d * dot(d, r)` on the post-layer
residual, the collector's readout point, so `residual` 1 leaves exactly no
component along `d` after that layer whatever its source. Writer and
residual strengths combine (writers first, then the residual projection);
use `"attention":0,"mlp":0,"residual":1` for residual-only ablation.

## Profile and API

A profile is a JSON file with schema `gufo-abliteration-v2`, placement
`writer-before-residual`, `model_identity` (the artifact content identity, as
for persistent caches), `hidden` (5120), `layers` (64), `directions` (64 unit
vectors of 5120 floats), and the request defaults `direction_layer` and
per-layer `attention` (attention and GDN layers alike), `mlp` and optional
`residual` strengths in [0, 4]. A profile without `residual` (every profile
written before it existed) reads it as all zero and behaves as before; v1
profiles cannot carry it. `prepare.py` writes it as compact JSON of float32 round-trip values,
about 4.5 MB at rank 1; the optional `rank`, `subspace_directions`,
`projected_directions`, `direction_kind`, `direction_count`, `preserve_norm`
and `subspace_method` members (base64 float32 rows) bring a rank-4 profile
with both kinds to about 17 MB and a rank-8 one to about 31 MB; the loader
accepts files up to 48 MiB. The schema and its v1
compatibility (one fixed `direction`) are those of
[Flash-Next](../qwen3.8-flash-next/ABLITERATION.md#profile-and-api).
Flash-Next profiles (`writer-before-hc`) and profiles of another artifact are
rejected, as are nonfinite values, nonunit vectors, out-of-range layers or
strengths and incompatible dimensions.

```sh
gufo serve llm --model models/qwen3.8-27b-exl3 --abliteration-profiles PROFILE_DIR
```

PROFILE_DIR holds exactly one profile without a `trigger` (detection runs
only on Flash-Next). The option is the only way to load one; nothing beside
the model is discovered, and loading never enables it. Requests select it and their own
settings exactly as for Flash-Next (Chat Completions, Responses, Messages):

```json
{"abliteration": {"enabled": true}}
```

```json
{"abliteration": {"enabled": true, "direction_layer": 40, "attention": 1.5,
                  "mlp": 0.5}}
```

```json
{"abliteration": {"enabled": true, "direction_layer": 40, "attention": 0,
                  "mlp": 0, "residual": 1}}
```

```json
{"abliteration": {"enabled": true, "direction_layer": 40, "residual": 1,
                  "direction_kind": "projected", "directions": 4,
                  "preserve_norm": true}}
```

`tools/qwen38_flash_next/abliteration/search.py` searches settings over HTTP
for either model (by default the residual curve, `--mode residual`, plus the
direction kind, count and norm preservation axes the profile offers); see
[Flash-Next](../qwen3.8-flash-next/ABLITERATION.md#automatic-strength-search).

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
python3 $PREP fit --collection CAPTURE --output CANDIDATES --rank 4 --method pca
$COLLECT $MODEL DATA/tuning.jsonl OFF evaluate --max-new-tokens 1024
$COLLECT $MODEL DATA/tuning.jsonl ON-L38-S1 evaluate --max-new-tokens 1024 \
  --profile CANDIDATES/layer38-strength1.json
python3 $PREP make-profile --fit CANDIDATES --direction-layer 38 \
  --residual 1 --output NEW.json   # residual-only; writers 0 from a fit
python3 $PREP make-profile --fit CANDIDATES --direction-layer 38 \
  --attention max_w=1.5,peak=38,min_w=0.5,width=12 --mlp 0.5 --output NEW2.json
```

`fit` also writes `layerL-residual1.json` per screened layer: residual 1 on
every layer, writers 0. Its candidates default to the projected kind, one
direction and no norm preservation; `make-profile --direction-kind
--directions --preserve-norm` changes those defaults. With 616 refusal
prompts `fit` holds about 0.8 GB of refusal activations.

The collector applies a profile's defaults; per-request settings are a
server feature.

`collect` records, per prompt, the residual of the final prompt token after
each of the 64 layers (FP32, 64 x 5120): after the layer's MLP residual add,
the same point the residual projection acts on. `evaluate` stores first-next-token
logits, greedy text and reasoning/truncation flags. Receipts, resume checks and
the selection rules (tuning only, then untouched evaluation inputs; an
incomplete answer is inconclusive) are those of the Flash-Next workflow.

## Focused checks

CPU: `qwen27b_abliteration_test` (profile binding, placement, v1 and v2
loading, optional residual strengths, settings resolution and identities
recomputed from their documented layout, device-word layout, OFF metadata
bytes, mode and settings mismatch; the direction members, their
validation, kind/count/norm overrides and identities, and unchanged rank-1
behavior), `openai_chat_test`, `http_server_test`,
`python3 tools/qwen38_flash_next/abliteration/test_prepare.py` and
`test_search.py`.
GPU: `qwen27b_abliteration_ops_test` covers an independent FP64 formula,
unaligned widths, mixed rows with their own directions and strengths,
in-place settings updates under a captured graph and the writer-then-residual
order; and for the residual projection the FP64 formula with mixed sessions,
zero-strength rows left bit-identical, graph replay with in-place updates, and
the writer, add, residual-projection order removing a component the incoming
residual carried; the subspace projection with k in {1, 2, 5, 8} against
FP64 for writer and residual rows with mixed sessions and norm preservation,
one direction bit-identical to the single-direction kernel, and graph replay
with in-place count and norm updates. Run it
with the final combined GPU checks: OFF preservation, ON/OFF C2, graph replay,
batched decode, DFlash2 verification, text/image continuation, cache
reload/fork and per-request timings.
