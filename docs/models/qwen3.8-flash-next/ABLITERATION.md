# Optional runtime abliteration

Implementation, independent code review, Nix 10.1 build and focused CPU checks
have passed. No qualified profile is bundled;
GPU/model validation and behavioral evaluation are required before deployment.
This is a reversible inference intervention, not a weight conversion.

## Ownership and execution

- `abliteration.{hpp,cpp}` owns the versioned, immutable, model-bound profile and
  cache identity. It has no HIP dependency.
- The model loads one direction once; the executor keeps its GPU copy. Each
  session owns its ON/OFF state. There is no server-global mutable toggle.
- The hook acts on the attention/GDN output projection and the **aggregated**
  routed-plus-shared MoE output, immediately before HC/residual injection.
  It does not project the accumulated residual, attention probabilities, logits,
  or every expert separately. MTP drafts remain unchanged; target verification
  uses the selected intervention.
- A zero layer strength launches no projection. With all sessions OFF, no
  projection kernel launches. With mixed sessions, one masked kernel covers
  up to eight sessions at each selected boundary. It updates the existing
  output in place, without a second activation tensor or per-token upload.
- ON materializes deferred MoE outputs where necessary before projection; this
  can lose an existing fusion. Actual end-to-end overhead remains unmeasured.
  Small operator cost is not evidence of negligible complete-request overhead.
- Mode changes invalidate session execution graphs and resident token history.
  Profile identity salts text and every image-prefix cache identity. Matching
  mode caches can be reused; ON/OFF or different profiles cannot share states.
  Engine snapshots also check the mode and identity.

The GPU operation is `y -= strength * direction * dot(direction, y)` with an
FP32 reduction and unit direction. The file carries separate per-layer attention
and MLP strengths. Versioning the placement and dimensions allows future
architectures to implement their own adapter rather than scatter model-specific
conditions through the server. This implementation supports Flash-Next only.

## Profile and API

A profile is a small JSON file with schema `gufo-abliteration-v1`, placement
`writer-before-hc`, `model_identity`, `hidden`, `layers`, `direction`, `attention`
and `mlp`. It must match the loaded artifact; nonfinite values, nonunit vectors,
incompatible dimensions and oversized files are rejected. Candidate provenance
can be stored as additional fields. Structural loading does not certify quality.

Load a chosen file with `gufo serve llm ... --abliteration-profile /path/profile.json`.
Alternatively the model loader discovers `abliteration/default.json` under the
model directory (beside the artifact for GGUF). Loading never enables it by
itself. A cache directory can be used with the explicit path option. Do not copy
an unqualified candidate into the default location.

Supported chat-style requests (Chat Completions, Responses, Messages):

```json
{"abliteration": {"enabled": true, "profile": "default"}}
```

Omit the object or send `{"abliteration":{"enabled":false}}` for OFF.
Unknown fields, profile names/paths and unsupported backends are rejected;
clients cannot load arbitrary local files through requests. Raw completions do
not accept this control. The option is independent of thinking and Han-token
sampling controls. Existing clients need to send this additional JSON field;
this change does not add a DSH UI control.

## Prepare once, retain receipts

Tools live in `tools/qwen38_flash_next/abliteration/`, outside serving code.
The native collector is an explicit CMake target:
`qwen38_flash_next_abliteration_collect`. It runs locally without an HTTP server.
The Python fitter needs NumPy; it does not run GPU work.

1. `prepare.py import-data --archive KYOJIN.tar.gz --output NEW_DATA_DIR` copies
   the exact prompt members and source notices. It deduplicates texts, reserves
   heldout texts before splitting collection/tuning, and includes thinking ON
   and OFF for each text. Dataset licenses are independent of Gufo's code;
   retain `SOURCES.md` (including the Alpaca noncommercial restriction).
2. `collector MODEL DATA/collection.jsonl CAPTURE collect` records the final
   prompt token after each trunk layer, averaged over HC streams in FP32.
   It uses the model's native template, medium reasoning, no draft and no
   intervention. Prompts over 2048 tokens are rejected, never silently clipped.
3. `prepare.py fit --collection CAPTURE --output NEW_CANDIDATES` computes
   refusal-minus-ordinary means and unit directions. A bounded screen varies
   source layer and strength; files remain `candidate-not-qualified`.
4. `collector MODEL DATA/tuning.jsonl OFF evaluate --max-new-tokens 1024` stores
   an OFF reference. Candidate runs use a **different** output directory and
   `--profile CANDIDATE.json`. Reuse each successful collection/reference;
   resume verifies model, executable, device code, dataset, profile and settings.
5. Select using tuning only. Evaluate the selected candidate and OFF on untouched
   evaluation inputs; inspect complete answers for refusal behavior and ordinary
   capability. The collector saves first-next-token logits, generated text and
   truncation/reasoning flags. `generation_complete` describes EOS and reasoning
   termination only; inspect whether a meaningful final answer is present. First-token divergence is not whole-model KLD;
   an incomplete answer is inconclusive, not a success. Longer evaluation budgets
   (up to 16384) need separate receipt directories.
6. Run one combined integration/correctness/performance pass: OFF preservation,
   ON/OFF C2, graph replay, text/image continuation, cache reload/fork, cancellation,
   greedy and sampled MTP, and per-request timings. Only then install the selected
   profile. Do not infer MTP output parity or quality from acceptance alone.

The collector never starts the production service, overwrites model weights,
installs profiles or interprets dataset prompts as host instructions. Concurrent
collectors must use separate output directories and coordinate GPU ownership.

## Focused checks

CPU: `qwen38_flash_next.abliteration`, `openai_chat_test`, and
`python3 tools/qwen38_flash_next/abliteration/test_prepare.py`.
GPU: `qwen38_flash_next.abliteration_ops` covers an independent FP64 formula,
unaligned dimensions, mixed ON/OFF rows and the production deferred-attention /
MoE / projection / HC order. Build it during iteration; run it with the final
combined GPU checks after the serving GPU is available.
