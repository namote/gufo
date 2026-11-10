# Optional runtime abliteration

Implementation, independent code review, Nix 10.1 build and focused CPU checks
have passed. No qualified profile is bundled;
GPU/model validation and behavioral evaluation are required before deployment.
This is a reversible inference intervention, not a weight conversion.

## Ownership and execution

- `abliteration.{hpp,cpp}` binds the shared Qwen profile
  (`src/models/qwen/abliteration`: versioned, immutable, model-bound, with its
  cache identity) to this placement. It has no HIP dependency.
- The model uploads every profile direction once at load (48 x 2560 floats
  for a v2 profile); the executor keeps the GPU copy. Each session owns its
  effective settings (OFF, or a direction layer and per-layer attention, MLP
  and residual strengths) and, from its first ON request, a small device copy
  of them (1 + 3 x layers words). There is no server-global mutable toggle.
- Writer strengths (`attention`, `mlp`) act on the attention/GDN output
  projection and the **aggregated** routed-plus-shared MoE output,
  immediately before HC/residual injection. They cannot reach per-layer
  embedding (PLE) contributions (injected into the residual directly), the
  embedding, or anything already in the residual.
- Residual strengths (`residual`) act on the residual after each layer's FFN
  combine: each of the four HC streams of each token loses
  `s * d * dot(d, stream)`. The collector reads the directions there (the FP32
  mean of the four streams); projecting every stream removes the mean's
  component too, so `residual` 1 leaves exactly none after that layer,
  whatever its source (embedding, PLE, earlier layers). The placement name
  `writer-before-hc` predates residual strengths.
- Neither projects attention probabilities, logits or every expert
  separately. MTP drafts remain unchanged (they read the projected trunk
  rows); target verification uses the selected intervention.

| Path | Writers | Post-layer residual |
| --- | --- | --- |
| Single session (prompt, decode graphs, verification) | `runtime/executor_hc.cpp` `Combine` | `runtime/executor_forward.cpp` `ForwardBody`, after the FFN `Combine` |
| Mixed batch, up to eight sessions | `runtime/executor_batch.cpp` | `runtime/executor_batch.cpp`, after the FFN `CombineBatch` |

- The FFN combine normally also writes the next mixer's grouped norm (F32,
  F16, tiled Q8, inject partials) and may leave its residual update deferred.
  A session with a nonzero residual strength takes the unfused layer end
  instead: the combine gets no next norm (as before a PLE layer), any pending
  residual update is applied, all four streams are projected in place, and
  the next mixer normalizes the projected residual itself. ON already
  disables the deferring MoE combines. In a mixed batch the combine writes
  every session's norm, then only the rows of sessions with residual
  strengths are projected and renormalized; other sessions keep exactly the
  norm they had.
- With all sessions OFF, no projection kernel launches. An ON session launches
  the projection at every writer, also where its strength is zero (the kernel
  then returns at once), so its launches do not depend on strength values.
  Residual projections (and the unfused layer end) run only for a session
  with a nonzero residual strength, then at every layer; writer-only ON
  settings launch and fuse exactly as before residual strengths existed.
  With mixed sessions, one masked kernel covers up to eight sessions at each
  selected boundary; each row reads its session's direction row and strength
  from that session's device settings. It updates the existing output in
  place, without a second activation tensor or per-token upload.
- ON materializes deferred MoE outputs where necessary before projection; this
  can lose an existing fusion, on every layer of an ON session. Actual
  end-to-end overhead remains unmeasured. Small operator cost is not evidence
  of negligible complete-request overhead.
- ON/OFF changes invalidate session execution graphs, as do changes between
  all-zero and nonzero residual strengths. Other settings changes
  are written in place to the session's device copy (an ordered upload on the
  executor stream, then a stream synchronization) and keep recorded graphs,
  which read the settings when they run. Any settings change drops resident
  token history. The settings identity (`gufo-abliteration-settings-v2`),
  SHA-256 over the profile identity, the direction layer and the exact float
  bytes of every attention, MLP and residual strength (-0 folded into +0), salts text and every image-prefix cache identity: matching
  settings can reuse caches; ON/OFF or different settings cannot share
  states. Two independent guards also check each restored state.
  Component-cache continuation metadata records the mode and, ON, the image
  identity salted with the settings identity; a mismatch is a cache miss.
  Engine snapshots record the mode in their header and the salted identity
  likewise. OFF records mode 0 in bytes every earlier capture zero-filled, so
  existing OFF caches remain valid byte for byte, and OFF requests allocate,
  upload and launch nothing new.

The GPU operation is `y -= strength * direction * dot(direction, y)` with an
FP32 reduction and unit direction. The file carries separate per-layer attention,
MLP and residual strengths. Writer strength 1 under-removes, because the
direction measured on the residual also enters through the embedding and PLE,
which is why writer strength 2 was needed here; residual strength 1 is the
canonical directional ablation and removes it exactly. Versioning the placement and dimensions allows future
architectures to implement their own adapter rather than scatter model-specific
conditions through the server. Flash-Next and
[Qwen3.8-27B](../qwen3.8-27b/ABLITERATION.md) (`writer-before-residual`)
implement it.

## Profile and API

A profile is a JSON file with schema `gufo-abliteration-v2`, placement
`writer-before-hc`, `model_identity`, `hidden`, `layers`, `directions` (one unit
vector of `hidden` floats per layer, `[layers][hidden]`), and the request
defaults `direction_layer`, `attention[layers]`, `mlp[layers]` and the
optional `residual[layers]` (strengths in [0, 4]; absent means all zero, so
every earlier v2 profile behaves as before; v1 cannot carry it). A v2 Flash-Next profile is about 1.6 MB as written by
`prepare.py`. Schema `gufo-abliteration-v1` (one `direction`, whose source
layer is unknown) still loads: requests may change its strengths but not its
direction. Profiles must match the loaded artifact; nonfinite values, nonunit
vectors, out-of-range layers or strengths, incompatible dimensions and files
over 16 MiB are rejected. Candidate provenance can be stored as additional
fields. Structural loading does not certify quality.

Load a chosen file with `gufo serve llm ... --abliteration-profile /path/profile.json`.
This option is the only way to load one: nothing beside the model (such as an
`abliteration/` directory) is discovered, so a file in a shared models
directory never makes the control available to clients or affects other
models. Loading never enables it by itself.

Supported chat-style requests (Chat Completions, Responses, Messages):

```json
{"abliteration": {"enabled": true, "profile": "default"}}
```

uses the profile defaults. Each request can also choose its own settings,
without reloading anything:

```json
{"abliteration": {"enabled": true, "direction_layer": 36,
                  "attention": 2.0, "mlp": 0.5}}
```

or residual-stream ablation:

```json
{"abliteration": {"enabled": true, "direction_layer": 36,
                  "attention": 0, "mlp": 0, "residual": 1}}
```

`direction_layer` is an integer in [0, layers) and needs a v2 profile.
`attention`, `mlp` and `residual` are each one number for every layer or an
array of exactly `layers` numbers, each in [0, 4]. Omitted members keep the profile
defaults. Sessions in one batch may run different settings. Omit the object or
send `{"abliteration":{"enabled":false}}` for OFF; settings with
`"enabled": false` are rejected. Unknown fields, profile names/paths, invalid
settings and unsupported backends are rejected with `invalid_abliteration`;
clients cannot load arbitrary local files through requests. `GET /props`
reports the loaded profile under `abliteration`. Raw completions do
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
   prompt token after each trunk layer (after its FFN combine, where residual
   strengths act), averaged over HC streams in FP32.
   It uses the model's native template, medium reasoning, no draft and no
   intervention. Prompts over 2048 tokens are rejected, never silently clipped.
3. `prepare.py fit --collection CAPTURE --output NEW_CANDIDATES` computes
   refusal-minus-ordinary means and unit directions. A bounded screen varies
   the default direction layer and strength; every candidate is a v2 profile
   with all directions, and files remain `candidate-not-qualified`.
   `fit` also writes `layerL-residual1.json` per screened layer (residual 1
   on every layer, writers 0).
   `prepare.py make-profile --fit NEW_CANDIDATES --direction-layer 36
   --residual 1 --output NEW.json` (or `--profile V2.json` to re-default an
   existing profile) writes one v2 profile. `--attention`, `--mlp` (default 0)
   and `--residual` (default omitted, read as 0) take one number, `layers`
   comma-separated numbers, or a curve
   `max_w=2,peak=36,min_w=0.5,width=10`: weight(i) = max_w + (min_w - max_w)
   * |i - peak| / width when |i - peak| <= width, else 0 (Heretic's form).
4. `collector MODEL DATA/tuning.jsonl OFF evaluate --max-new-tokens 1024` stores
   an OFF reference. Candidate runs use a **different** output directory and
   `--profile CANDIDATE.json`. Reuse each successful collection/reference;
   resume verifies model, executable, device code, dataset, profile and settings.
   Alternatively, serve a v2 candidate and let `search.py` choose settings
   over HTTP (see below); it also uses tuning only.
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

## Automatic strength search

`search.py` (shared with Qwen3.8-27B; standard library only) drives a running
server with a v2 profile through per-request settings. Each trial picks a
direction layer and Heretic curves (max_w, peak, min_w, width) for the
strengths `--mode` selects, sending the others as 0: `residual` (default; the
residual curve), `writer` (attention and MLP, the earlier search, whose logs
still resume) or `both`. It then counts refusals on harmful tuning prompts (greedy, short
`--max-tokens`, a phrase heuristic on the final answer) and the mean
first-token KL(OFF || ON) on ordinary tuning prompts from the server's 20
`top_logprobs`, which are raw model log-probabilities after the in-forward
projection. The KL is approximated on the union of both top-20 sets, each
side's missing tokens sharing its leftover mass, both renormalized over the
union (see the script's docstring). An empty final answer counts as a
refusal, so a broken model never scores well. Trial 0 is the profile defaults,
then `--initial` random samples, then Gaussian perturbations of the current
Pareto front. Thinking is explicit (`--thinking off|on|both`, sent as
`chat_template_kwargs.enable_thinking`); thinking prompts need a larger
budget, and answers still empty at the budget are also reported as
unfinished. `--base-profile` must be the file the server loaded (its SHA-256
is the profile identity in `/props`). Every
trial is appended to `--log` (rerunning resumes; the mode is part of the log's
fingerprint); the end prints the Pareto
front and writes the trial with the fewest refusals within `--kl-budget` as a
new v2 profile:

```sh
python3 tools/qwen38_flash_next/abliteration/search.py \
  --base-url http://127.0.0.1:8080 --model gufo \
  --data NEW_DATA_DIR/tuning.jsonl --refusal-prompts 24 --ordinary-prompts 24 \
  --thinking off --max-tokens 64 --trials 80 --initial 20 \
  --mode residual --log SEARCH/trials.jsonl --kl-budget 0.1 \
  --base-profile SERVED_V2.json --output SEARCH/chosen.json
```

The chosen profile is still a tuning result: evaluate it on held-out inputs
as in step 5.

## Focused checks

CPU: `qwen38_flash_next.abliteration`, `qwen27b_abliteration_test` (shared v2
profile, settings and identity contracts), `openai_chat_test`,
`http_server_test`,
`python3 tools/qwen38_flash_next/abliteration/test_prepare.py` (NumPy) and
`python3 tools/qwen38_flash_next/abliteration/test_search.py`.
GPU: `qwen38_flash_next.abliteration_ops` covers an independent FP64 formula,
unaligned dimensions, mixed rows with their own directions and strengths,
in-place settings updates under a captured graph and the production
deferred-attention / MoE / projection / HC order; and for the residual
projection, all four HC streams against an FP64 reference with mixed
sessions, the stream mean's component removed at strength 1, zero-strength
and OFF tokens left bit-identical, and graph replay with in-place updates. Build it during iteration; run it with the final
combined GPU checks after the serving GPU is available.
