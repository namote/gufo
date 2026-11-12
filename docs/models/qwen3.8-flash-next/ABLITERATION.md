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
  for a rank-1 v2 profile; kinds x 48 x rank x 2560 with the subspace
  members, 7.5 MiB at rank 8 with both kinds); the executor keeps the GPU
  copy. Each session owns its effective settings (OFF, or a direction layer,
  kind, direction count, norm preservation and per-layer attention, MLP and
  residual strengths) and, from its first ON request, a small device copy of
  them (1 + 3 x layers words; word 0 packs the first direction row, the
  count and the norm flag, and is the bare row index for one direction
  without norm preservation). There is no server-global mutable toggle.
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
  F16, tiled Q8, inject partials), and it settles any update the attention
  combine deferred.
  A session with a nonzero residual strength takes the unfused layer end
  instead: the combine gets no next norm (as before a PLE layer), any pending
  residual update is applied (none should remain), all four streams are
  projected in place, and the next mixer normalizes the projected residual
  itself. ON already
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
  token history. The settings identity (`gufo-abliteration-settings-v3`),
  SHA-256 over the profile identity, the direction layer, the exact float
  bytes of every attention, MLP and residual strength (-0 folded into +0),
  the direction kind, the direction count and norm preservation, salts text
  and every image-prefix cache identity (v3 replaced v2 when these options
  arrived, so ON states cached by earlier builds are misses; OFF is
  unchanged): matching
  settings can reuse caches; ON/OFF or different settings cannot share
  states. Two independent guards also check each restored state.
  Component-cache continuation metadata records the mode and, ON, the image
  identity salted with the settings identity; a mismatch is a cache miss.
  Engine snapshots record the mode in their header and the salted identity
  likewise. OFF records mode 0 in bytes every earlier capture zero-filled, so
  existing OFF caches remain valid byte for byte, and OFF requests allocate,
  upload and launch nothing new.

The GPU operation is `y -= strength * sum_j d_j * dot(d_j, y)` over the
session's k orthonormal directions (k = 1 by default), every dot product
taken from the unprojected row with FP32 sums and a fixed reduction order,
then, with norm preservation, `y *= |y_before| / |y_after|` per row (per HC
stream for the residual). One direction without norm preservation runs the
same arithmetic in the same order as the single-direction kernel, so it is
bit-identical to it. The file carries separate per-layer attention,
MLP and residual strengths. Writer strength 1 under-removes, because the
direction measured on the residual also enters through the embedding and PLE,
which is why writer strength 2 was needed here; residual strength 1 is the
canonical directional ablation and removes it exactly. Versioning the placement and dimensions allows future
architectures to implement their own adapter rather than scatter model-specific
conditions through the server. Flash-Next and
[Qwen3.8-27B](../qwen3.8-27b/ABLITERATION.md) (`writer-before-residual`)
implement it.

## Direction kinds, subspaces and norm preservation

Three options refine the direction(s) a site removes. All are optional
profile members and per-request settings; without them a profile behaves
exactly as before (mean difference, one direction, no norm preservation).

- `direction_kind`: `mean_diff` is r = mean(refusal) - mean(ordinary) at the
  layer, normalized. `projected` removes from r its component along the
  normalized ordinary mean h (Gram-Schmidt): p = r - (r . h) h, normalized
  (projected abliteration, arXiv 2512.13655; Heretic's default; Kyojin's
  "minus harmless-mean component" preset). The ordinary mean is what every
  prompt shares; removing p instead of r leaves that shared part alone, which
  those sources report keeps ordinary behavior closer to the original at the
  same refusal reduction. `prepare.py fit` therefore makes `projected` the
  default of new profiles whenever every layer has one (a refusal direction
  lying along the ordinary mean has none; `fit` then stores only
  `mean_diff`).
- `directions` (rank k): refusal is not always one direction (Wollschlaeger et
  al., "The Geometry of Refusal in Large Language Models: Concept Cones and
  Representational Independence", ICML 2025; multi-directional refusal found
  with a self-organizing map, arXiv 2603.22061). `fit --rank K` stores, per
  layer and kind, K orthonormal rows whose first is the kind's single
  direction; a request removes the first k (1 <= k <= K). The remaining rows
  come from the per-prompt deviations x_i - mean(ordinary) of the refusal
  prompts: `--method pca` takes their top right singular vectors,
  `--method som` the prototypes of a small batch self-organizing map
  (g x g units, g = ceil(sqrt(2K)), seeded) most aligned with r. Gram-Schmidt
  (FP64, twice) orthonormalizes them after the first row, also against h for
  `projected`; dependent data yields fewer rows (the rank `fit` reports).
  K is at most 8: the kernel keeps one FP32 dot sum per direction in
  registers and reduces all of them in one shared block, a rank-8 profile
  with both kinds is already 2 x 8 times the rank-1 directions, and the
  refusal subspaces in the work above are low-dimensional, while removing
  more directions approaches removing a generic subspace.
- `preserve_norm`: after the projection each row (writer output row, or each
  HC stream of the residual) is rescaled to its L2 norm before the
  projection. This is the activation-space analogue of grimjim's
  norm-preserving biprojected abliteration, which edits the weights so that
  their norms are kept; Gufo edits no weights and only keeps each projected
  activation's norm.

## Profile and API

A profile is a JSON file with schema `gufo-abliteration-v2`, placement
`writer-before-hc`, `model_identity`, `hidden`, `layers`, `directions` (one unit
vector of `hidden` floats per layer, `[layers][hidden]`), and the request
defaults `direction_layer`, `attention[layers]`, `mlp[layers]` and the
optional `residual[layers]` (strengths in [0, 4]; absent means all zero, so
every earlier v2 profile behaves as before; v1 cannot carry it). A rank-1 v2 Flash-Next profile is about 1.6 MB as written by
`prepare.py`. The optional direction members (v2 only; none may appear in v1):

| Member | Meaning |
| --- | --- |
| `rank` | K, an integer in [1, 8]; absent is 1 |
| `subspace_directions` | with K > 1 only: `layers` strings, each base64 of (K - 1) x `hidden` little-endian float32, rows 1..K-1 of `mean_diff` (row 0 is `directions[l]`) |
| `projected_directions` | optional: `layers` strings, each base64 of K x `hidden` float32, the `projected` rows |
| `direction_kind` | default kind, `mean_diff` (absent) or `projected` (needs `projected_directions`) |
| `direction_count` | default k in [1, K]; absent is 1 |
| `preserve_norm` | default boolean; absent is false |
| `subspace_method` | `pca` or `som`, informational |

Each layer's block of each kind must be orthonormal (unit rows, pairwise
dot products within 1e-4). Base64 keeps a rank-8 27B profile with both kinds
near 31 MB, where JSON numbers would need about 60 MB and far more memory to
parse. Schema `gufo-abliteration-v1` (one `direction`, whose source
layer is unknown) still loads: requests may change its strengths but not its
direction. Profiles must match the loaded artifact; nonfinite values, nonunit
vectors or blocks, out-of-range layers, strengths, ranks or counts, unknown
kinds, incompatible dimensions and files over 48 MiB are rejected. Candidate provenance can be stored as additional
fields. Structural loading does not certify quality.

Load chosen files with `gufo serve llm ... --abliteration-profiles DIR`, where
DIR holds them (every `*.json` file in it loads; the file name without `.json`
is the profile name).
This option is the only way to load one: nothing beside the model (such as an
`abliteration/` directory) is discovered, so a file in a shared models
directory never makes the control available to clients or affects other
models. Loading never enables it by itself.

Supported chat-style requests (Chat Completions, Responses, Messages):

```json
{"abliteration": {"enabled": true, "profile": "pol-avg"}}
```

uses the defaults of the profile named `pol-avg` (file `pol-avg.json`);
`profile` may be omitted when one profile is loaded. Each request can also choose its own settings,
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

or a projected rank-4 subspace with norm preservation:

```json
{"abliteration": {"enabled": true, "direction_layer": 36, "residual": 1,
                  "direction_kind": "projected", "directions": 4,
                  "preserve_norm": true}}
```

`direction_layer` is an integer in [0, layers) and needs a v2 profile.
`attention`, `mlp` and `residual` are each one number for every layer or an
array of exactly `layers` numbers, each in [0, 4]. `direction_kind` is
`mean_diff` or, when the profile stores it, `projected`; `directions` is an
integer in [1, rank]; `preserve_norm` is a boolean. Omitted members keep the profile
defaults. Sessions in one batch may run different settings. Send
`{"abliteration":{"enabled":false}}` for OFF; settings with
`"enabled": false` are rejected. Omitting the object is OFF too unless a
loaded profile has a `trigger`, which then decides per request
([dynamic detectors](#dynamic-detectors-trigger)). Unknown fields, unknown
profile names or paths, invalid
settings and unsupported backends are rejected with `invalid_abliteration`;
clients cannot load arbitrary local files through requests. `GET /props`
lists the loaded profiles under `abliteration.profiles` (with the only one
also at the top level of `abliteration`), each with its `trigger` (layer and
threshold, or null), and
`direction_kinds`, `direction_kind`, `max_directions` (the rank),
`directions`, `preserve_norm` and `subspace_method`. Raw completions do
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
3. `prepare.py fit --collection CAPTURE --output NEW_CANDIDATES [--rank 4]
   [--method pca|som] [--seed 0]` computes refusal-minus-ordinary means, unit
   directions, the projected kind and the rank-K subspace of both (rank 4
   and PCA by default; `SUMMARY.json` records the rank reached). It keeps the
   refusal prompts' activations in memory (prompts x layers x hidden FP32).
   A bounded screen varies the default direction layer and strength; every
   candidate is a v2 profile with all directions, both kinds and the
   subspace, defaulting to `projected`, one direction and no norm
   preservation, and files remain `candidate-not-qualified`.
   `fit` also writes `layerL-residual1.json` per screened layer (residual 1
   on every layer, writers 0).
   `prepare.py make-profile --fit NEW_CANDIDATES --direction-layer 36
   --residual 1 --output NEW.json` (or `--profile V2.json` to re-default an
   existing profile) writes one v2 profile; `--direction-kind`,
   `--directions` and `--preserve-norm on|off` set the option defaults (each
   omitted one keeps the base value). `--attention`, `--mlp` and
   `--residual` (each omitted one keeps the `--profile` value; from a fit, 0,
   with no `residual` member) take one number, `layers` comma-separated
   numbers, or a curve
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
refusal, so a broken model never scores well. On a server whose `/props`
reports them, every mode also searches the projection options: the
direction kind (`--direction-kinds`, default every stored kind), the
direction count k in 1..`--max-directions` (default the stored rank) and
norm preservation (`--preserve-norm search|on|off`, default search); trials
send all three explicitly. With one kind, one direction and
`--preserve-norm off` the log fingerprint is the earlier one, so earlier
logs resume. Trial 0 is the profile defaults,
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

## Dynamic detectors (`trigger`)

A profile may carry a detector so the server turns it ON only for requests
that need it (see `docs/SERVER.md`, `--abliteration-profiles`).
`fit_trigger.py` fits one from collector output (CPU, NumPy) and writes a
copy of the profile with a `trigger` member; the input profile is unchanged:

```sh
python3 tools/qwen38_flash_next/abliteration/fit_trigger.py \
  --collection COLLECTION --target-group refusal --layer 38 \
  --topics TOPICS.jsonl --negatives HARMLESS_COLLECTION:ordinary \
  --profile SERVED_V2.json --output PROFILES_DIR/name.json
```

- Direction: mean(target group) − mean(ordinary) at `--layer`, normalized.
  Extra negatives never enter it.
- Validation is grouped by topic (`--topics` JSONL `{"id", "topic"}`, a
  record's `topic`, else the ID without `-plain`/`-think`), so paraphrases of
  one question never sit on both sides of a fold.
- The threshold keeps the out-of-fold false-positive rate at or below
  `--max-false-positive` (default 1%) on each negative set separately. The
  report also gives AUC, recall (by thinking mode) and a nested estimate
  whose threshold never saw the fold it is scored on.

The score is read where the collector reads: the post-layer residual of the
detector layer, averaged over the HC streams, at the last prompt row (the
server also scores every decoded row). Thresholds therefore transfer only
between captures with the same readout
(`last-prompt-token-post-layer-hc-mean-fp32-v1`) and model.

## Focused checks

CPU: `qwen38_flash_next.abliteration`, `qwen27b_abliteration_test` (shared v2
profile, settings and identity contracts), `openai_chat_test`,
`http_server_test`,
`python3 tools/qwen38_flash_next/abliteration/test_prepare.py` (NumPy;
projected, PCA and SOM math on planted data, orthonormality, SOM
determinism, profile members),
`python3 tools/qwen38_flash_next/abliteration/test_search.py` and
`python3 tools/qwen38_flash_next/abliteration/test_fit_trigger.py` (topic
folds, per-set threshold, profile copy).
GPU: `qwen38_flash_next.abliteration_ops` covers an independent FP64 formula,
unaligned dimensions, mixed rows with their own directions and strengths,
in-place settings updates under a captured graph and the production
deferred-attention / MoE / projection / HC order; and for the residual
projection, all four HC streams against an FP64 reference with mixed
sessions, the stream mean's component removed at strength 1, zero-strength
and OFF tokens left bit-identical, and graph replay with in-place updates;
and the subspace projection with k in {1, 2, 5, 8} for every HC stream
against FP64 with mixed sessions and norm preservation, plus one direction
bit-identical to the single-direction kernel for streams and writer rows. Build it during iteration; run it with the final
combined GPU checks after the serving GPU is available.
