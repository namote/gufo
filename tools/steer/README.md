# Steering tool (generate, fit, train, test)

A Python tool and HTTP job service, separate from gufo, for steering
experiments on Qwen3.8-Flash-Next. It only calls a gufo server, using the
endpoints in [docs/SERVER.md](../../docs/SERVER.md):

- runtime profile updates (`--allow-profile-updates`);
- activation collection jobs (`--collect-dir`);
- Chat Completions.

Design: [plan/research/STEERING-API-DESIGN.md](../../plan/research/STEERING-API-DESIGN.md) (C, C+ and Jobs).

| Path | Role |
| --- | --- |
| `__main__.py`, `cli.py` | Command line (`python3 -I tools/steer COMMAND`); every command runs as a job through the same code as the service |
| `service.py` | The HTTP job API (standard library `http.server`) |
| `jobs.py` | Keyed jobs persisted under the work folder, the FIFO worker and cancellation |
| `data.py` | Datasets and their builders (traces, CAA pairs, import) |
| `fitting.py` | Fit job: gufo collection job, then per-layer analysis and default profiles (NumPy) |
| `generation.py` | Test job (a math, coding or prompt set under configurations, with resume and import) and the pairs generation job |
| `objective.py` | Objective job: traces to SPSA rows through `recheck_objective.py` |
| `training.py`, `spsa_child.py` | Train job: `spsa.py` as a child process, with progress per step, the fast-recipe defaults and the trained profile |
| `pipeline.py` | Pipeline job: ordered steps as child jobs, with references between them |
| `client.py` | gufo HTTP client (base URL and optional API key) |
| `ops.py` | Operation table, job context, the profile and load operations, profile upload sources |
| `examples/r10.json` | The R10 pipeline |
| `test_steer.py`, `test_steer_cplus.py` | Unit tests against a fake gufo server |

Never point the tool at the production server for experiments. Use an
experiment server started with `--collect-dir` and `--allow-profile-updates`.
`--gufo-url` has no default. When the server has `--api-key`, the tool reads
the key from the environment variable `GUFO_API_KEY`. The key is never written
to job files or passed on a command line.

## Running

Dataset building, test, train, load and the job API need only the standard
library. Fit needs NumPy, which the host lacks. There are two ways to run it:

- **A local venv with NumPy** (`python3 -m venv ~/.venv/steer && ~/.venv/steer/bin/pip install numpy`).
  This runs everything, including the service, in one process:
  `~/.venv/steer/bin/python3 -I tools/steer ...`.
- **The pinned container** (`/opt/venv/bin/python3 -s`). It has no network,
  so it cannot reach gufo. Collect on the host with `fit --collect-only`, then
  fit the reported collection paths in the container (pass `--prefix`, or
  the profiles are named after the collection folder):

  ```sh
  python3 -I tools/docker/pinned_rocm101.py run --build-dir build/lbias --source . \
      --mount "$WORK":/work --mount "$COLLECT":/collect -- sh -c \
      'cd /build/source && /opt/venv/bin/python3 -s -I tools/steer --work /work fit \
         --prefix PREFIX --collection /collect/NAME --control-collection /collect/NAME-CONTROL'
  ```

Global options: `--work DIR` (default `~/.local/state/gufo-steer`),
`--gufo-url URL`, `--collections-dir DIR` (the local path of the server's
`--collect-dir`, where fit reads the collected activations),
`--poll-seconds`, `--json` (print results as JSON lines).

Tests (fake gufo server; fit tests skip without NumPy):

```sh
python3 -I -m unittest discover -s tools/steer -p 'test_*.py'
python3 -I tools/docker/pinned_rocm101.py run --build-dir build/lbias --source . -- sh -c \
    'cd /build/source && /opt/venv/bin/python3 -s -m unittest discover -s tools/steer -p "test_*.py"'
```

## Datasets

A dataset is `WORK/datasets/NAME/`:

- `records.jsonl`: records in a collector mode's input format (collect.cpp,
  `collect`, `answers` or `segments`);
- `dataset.json`: `kind`, `groups` (counts), `source`, `created_at`,
  `records`, `data_sha256`, `control` (the control dataset's name or null) and
  `stats`.

| Command | Builds |
| --- | --- |
| `dataset traces --name N --source NAME=PROBLEMS,TRACES ...` | Segments records from reasoning traces. The reasoning is cut to `--max-chars` (36000), split at blank lines, and each thought end is labelled by `--opener` (default: a re-check opener such as "Wait", "Hmm", "Let me verify") matching the next thought. A trace needs `--min-each` (3) of both labels. Also writes `N-shuf`, with the labels permuted within each trace (`--seed`), as N's control. |
| `dataset pairs --name N --instruction TEXT --mode answers\|segments --source ...` | CAA pairs: the same continuation under instruction + prompt (exemplar) against the bare prompt (own). `answers` takes each row's model answer (`text`, else `content`; a math row's `answer` is the reference, used only with `--answer-field answer`; rows not ending in `stop` are skipped unless `--keep-unfinished`). `segments` takes every thought end of the row's `reasoning`. |
| `dataset import --name N --path F [--control F2]` | An existing collector JSONL (mode inferred, or `--mode`) |
| `datasets` | Lists them |
| `run OPERATION PARAMS` | Any operation with its params (a JSON file or object), e.g. `run objective objective.json` |
| `pipeline FILE [--resume KEY]` | A pipeline (`{"steps": [...]}`) |
| `test ... --resume KEY\|CONFIG=FILE` | A test job reusing finished rows (repeatable) |

In a source, PROBLEMS is JSONL with `id` and `problem` (or `prompt`/`text`),
and TRACES/ROWS is JSONL keyed by `id` (think_eval results, generate.py
results or a test job's `generations/CONFIG.jsonl`). `--suffix` is appended to
every problem text. `--exclude-ids FILE` leaves problems out, for example the
held-out evaluation set.

## Jobs

| Operation | Params | Results (`type`) | `output` |
| --- | --- | --- | --- |
| `dataset` | `builder` (traces, pairs, import), `name`, builder options as above | `dataset` (the dataset.json of each written dataset) | dataset names |
| `fit` | `dataset`, or `collection` (+ `control_collection`) for an existing local collection; `control` (true), `collect_only`, `collection_name`, `collections_dir`, `top_n` (4), `profile_layers` (default the top 2), `alpha` (1.0), `additive_kind` (mean_diff), `prefix` | `collect`, `collection`, `layer`, `cosine`, `transfer`, `top`, `control_cosine`, `profile` | collection, pairs, top, per-layer rows, control, profile names and folder |
| `test` | `set` {`kind` math\|coding\|prompts, `path`, `split`, `ids`, `limit`, `max_tokens`, `suffix`, `seed_base`, `splits`, `image`}, `configs`, `concurrency` (2), `model`, `request_timeout`, `resume` | `config`, `resume` (with resume), then `row` and running `totals` per finished problem | `totals` per configuration, `generations` folder, `configs`, `set_kind` |
| `objective` | `sets` [{`name`, `kind` math\|coding, `problems`, `traces` \| `generations`+`config` \| `job`+`config`, `effort`, `splits`}], `keep_first_confirmations` (false), `max_boundaries` (8), `opener_tokens` (3), `guard_tokens` (1024), `redundant_after` (1), `seed` (0), `examples` (3), `max_prefix_tokens` (12288), `max_prefix_chars` (0), `max_guard_tokens` (12288), `max_guard_chars` (0), `guard_cap` (skip\|cut) | `objective_set` per set, `objective_rows` | `pairs`, `keep`, `guard` (paths, null when empty), `counts`, `labels`, `chains`, `longest_row_tokens`, `context_tokens`, `caps`, `effort` |
| `train` | `objective` (job key) or `pairs`/`ordinary`/`guard` paths; `options` {spsa option: value} and/or `args` (spsa.py options; the tool sets `--base-url`, `--log`, `--output`, and `--profile-output` with `basis`); `fast_defaults` (true), `server_check` (warn\|off), `base_profile` (path or {`job`, `profile`}) + `profile_name`, `basis` (a direction basis: also learn the direction, see "R11"), `resume` (train job key), `python` | `warning`, `train`, `step` (one per spsa.py log line), `best`, `profile` (with `profile_name`) | `log`, `best`, `best_path`, `recipe`, `profile`, `profile_path` |
| `profile` | `train` (job key) or `settings`, `base_profile` (default the train job's), `name`, `upload` (false) | `profile`, `upload` | `name`, `path`, `uploaded`, `reply` |
| `load` | `name`, `kind` (abliteration), `file` \| `job`+`profile` (+ `base` for a train job) \| `profile` {...}, `unload` [names to delete first] | `unload`, `upload` | `kind`, `name`, `reply` |
| `pairs` | `name`, `prompts` (JSONL) or `set`, `instruction`, `abliteration`, `reasoning_effort`, `sampling`, `mode` (answers\|segments), `concurrency`, `model`, `request_timeout`, `resume`, `max_chars`, `require_stop` (true), `contrast` (true), `exclude_ids`, `replace` | as `test` (configurations `with` and `without`), then `dataset` | `datasets`, `totals`, `generations` |
| `pipeline` | `steps` [{`name`, `operation`, `params`, `await_server`, `gufo_url`}], `resume` (pipeline key) | `server`, `pipeline_step` (start and end of each step) | `steps` [{`name`, `operation`, `job`, `status`, `reused`}] |

Fit:

1. It starts a gufo collection job for the dataset, and for its control.
   The gufo job name is `DATASET-<first 12 hex of data_sha256>`, sent with
   `"resume": true`, so a rerun reuses the finished records. A server built
   or configured differently refuses that name (`collection_differs`); set
   `collection_name` then. While gufo is down (up to 15 minutes, as during a
   restart) the fit keeps polling, and when gufo forgot the job it resumes it
   by name. The fit job polls
   until the collection is done and reports a `collect` result per poll that
   brings news. Cancelling the fit job also cancels the gufo job.
2. It then reads `collections_dir/NAME` and computes the paired differences
   D_i = exemplar - own per pair, as `fit_steering.py` does.
3. Per layer (`layer`, emitted one by one), it computes
   `build/b1/think/analyze.py`'s held-out numbers:
   - leave-one-out separation (`separated`/`pairs` and `acc`);
   - `snr`, the mean over the standard deviation of the held-out projections;
   - `norm` = |mean D|.
4. `cosine` is the full layer × layer cosine table of the mean differences.
5. `transfer` gives, for each top layer L, the held-out separation of every
   layer's differences along L's leave-one-out direction
   (`build/b1/think/xlayer.py`).
6. `top` ranks layers by SNR, then accuracy.
7. The control gets the same results with `dataset: "control"`.
   `control_cosine` is the per-layer cosine between the real and control mean
   differences.
8. It writes the default profiles to the job's `profiles/` folder. Each is
   built with `fit_steering.fit` or `prepare.make_profile`, which validate it
   as the runtime loader does. `profile` results carry their paths:
   - `PREFIX-add`: additive (`mode: add`), fit_steering's increments at
     `alpha`. Its `residual` is in the result, so a negated or scaled copy can
     go in a request's `residual`.
   - `PREFIX-pL`: the pol-avg recipe from layer L. It uses L's projected mean
     difference, `residual` 1.0 at every layer, and attention and mlp 0
     (`mksup.py`).
   - `PREFIX-sL`: the same direction with `residual` 1.0 at layer L only.

A test configuration is `{name, abliteration, logit_bias_profiles,
logit_bias_overrides, reasoning_effort, sampling}`:

- `abliteration` defaults to `{"enabled": false}`, so dynamic detection is
  off.
- `sampling` is merged into the request (`max_tokens`, `temperature`,
  `seed`, `chat_template_kwargs`, ...).

Math sets follow `think_eval.py`:

- the problem plus "Put the final answer in \boxed{}.";
- the server's sampling preset, with seed 1000 + the problem's index in the
  file after the `split` filter;
- 16384 tokens;
- the last `\boxed{}` normalized as think_eval does.

Coding sets follow `coding_eval/generate.py`: greedy, thinking off, 600
tokens. Answers are graded by its Docker sandbox; model code never runs
unsandboxed. Each `row` has `config`, `id`, `correct`, `tokens`,
`reasoning_tokens`, `hesitations` (think_eval's regex over the reasoning),
`truncated`, `finish_reason`, `seconds` and the graded fields (math: `got`,
`answer`; coding: `passed`, `tests`). Configurations run in order, with
`concurrency` requests at once (set it to the server's sessions). Full
answers go to the job's `generations/CONFIG.jsonl`. A cancel takes effect at
once and drops the replies of requests still in flight (gufo finishes them).
A malformed reply or a grader failure becomes a row with `error`.

## Job API

`python3 -I tools/steer [--gufo-url URL] [--collections-dir DIR] serve [--host 127.0.0.1] [--port 18790]`.
It binds to loopback by default. The API reads and writes local files named in
job parameters, so keep it to yourself.

| Method | Path | Reply |
| --- | --- | --- |
| `POST` | `/jobs` `{"operation", "params"}` | `{"job": KEY, "status": "queued"}` at once |
| `GET` | `/jobs` | `{"object": "list", "data": [...]}`: every job, newest first, without results |
| `GET` | `/jobs/{key}?since=N` | the job: `status` (queued, running, done, failed, cancelled), `progress` {`done`, `total`, `step`, `stage`}, `created_at`, `started_at`, `updated_at`, `error`, `cancel_requested`, `output`, `last_seq`, and `results` with `seq` > N (default 0) |
| `DELETE` | `/jobs/{key}` | cancels a queued or running job; forgets a finished one and removes its folder |
| `GET` | `/jobs?parent=KEY` | the jobs a pipeline started |
| `GET` | `/jobs/{key}/generations` | `list` of {`config`, `rows`, `bytes`}: a test or pairs job's generations files |
| `GET` | `/jobs/{key}/generations/{config}` | a `page` of that file's full rows (`row` fields plus `content`, `reasoning`, `usage`, `settings`, `request_sha256`, `resumed`, `resumed_from`) |
| `GET` | `/jobs/{key}/rows`, `/jobs/{key}/rows/{pairs\|keep\|guard}` | an objective job's row files (`list` of {`name`, `rows`}), or a `page` of one (spsa.py rows) |
| `GET` | `/jobs/{key}/profiles/{name}` | a profile file a job wrote |
| `GET` | `/datasets`, `/datasets/{name}` | dataset.json of every dataset, or of one |
| `GET` | `/datasets/{name}/records` | a `page` of its records |
| `GET` | `/profiles` | `list` of every profile file in the work folder: {`name`, `job`, `operation`, `path`, `bytes`, `mtime`, `mode`, `layers`, `direction_layer`, `direction_kind`} |
| `GET` | `/gufo/props` | gufo's `GET /props`, passed through (sessions, context, loaded profiles) |
| `POST` | `/profiles/{abliteration\|logit-bias}/{name}` `{"file": PATH}` \| `{"job": KEY, "profile": NAME}` \| `{"profile": {...}}` | gufo's reply to the upload |
| `DELETE` | `/profiles/{abliteration\|logit-bias}/{name}` | gufo's reply |
| `GET` | `/health` | status and the gufo URL |

Errors are `{"error": {"code", "message"}}`.

A `page` is `{"object": "page", "offset", "limit", "total", "data"}`:
`?offset=` (0), `?limit=` (20, at most 200), `?fields=a,b` (keep only these
keys of each row) and `?id=` (only rows with that id). A `list` is
`{"object": "list", "data"}`.

### Result types

Every job result has `seq` and `type`; these are the fields per type.

| `type` | Operation | Fields |
| --- | --- | --- |
| `dataset` | dataset, pairs | dataset.json: `name`, `kind`, `groups`, `source`, `created_at`, `records`, `data_sha256`, `control`, `stats` |
| `collect` | fit | `dataset` (real\|control), `gufo_job`, `collection`, `status`, `done`, `total`, `numerics_deviations` |
| `collected` | fit (`collect_only`) | `collection`, `control_collection` |
| `collection` | fit | `dataset`, `path`, `pairs`, `skipped`; real also `layers`, `hidden`, `readout` |
| `layer` | fit | `dataset`, `layer`, `separated`, `pairs`, `acc`, `snr` (null when undefined), `norm` |
| `cosine` | fit | `dataset`, `layers`, `matrix` (layers x layers) |
| `transfer` | fit | `dataset`, `sources` (top layers), `separated` {source: [count per layer]}, `pairs` |
| `top` | fit | `dataset`, `layers` (ranked) |
| `control_cosine` | fit | `cosines` (per layer, null when undefined) |
| `profile` | fit, train, profile | `name`, `path`, `mode` (suppress\|add), `direction_layer`, `direction_kind`; fit's add profile also `residual`; train and profile also `attention`, `mlp`, `residual` (per layer) |
| `config` | test, pairs | `config`, `fields` (request fields), `instruction`, `problems`, `set_kind` |
| `resume` | test, pairs | `config`, `reused`, `mismatched`, `unfinished`, `sources` |
| `row` | test, pairs | `config`, `id`, `split`, `tokens`, `reasoning_tokens`, `hesitations`, `truncated`, `finish_reason`, `seconds`, `think_chars`; math `got`, `answer`, `correct`; coding `correct`, `passed`, `tests`, `extract`; optional `abliteration` (gufo's report), `resumed`, `error` (then only `id`, `seconds`, `config`) |
| `totals` | test, pairs | `config`, `n`, `errors`, `correct`, `accuracy` (both null for an ungraded set), `tokens_mean`, `tokens_median`, `truncated`, `hesitations_mean`, `seconds_mean` |
| `objective_set` | objective | `set`, `kind`, `effort`, `counts` (math: `traces`, `rechecks`, `useful`, `harmful`, `redundant`, `neutral`, `with_candidate`, `no_alternative`, `capped`, `past_prefix_cap`, `pairs`, `keep`, `guard`, `guard_skipped`, `guard_cut`, `non_integer_gold`, `candidate_rate`; coding: `traces`, `passed`, `truncated`, `other_split`, `guard`, `guard_skipped`, `guard_cut`), `neutral_reasons` {reason: count}, `examples` {label: [{`trace`, `thought`, `before`, `after`, `reason`, `gold`, `text`}]}, `cost` {`requests`, `chains`, `prefill_without_reuse`, `prefill_with_reuse`, `max_prefix`, `max_row_tokens`, `scored_tokens`} (estimated tokens per theta) |
| `objective_rows` | objective | the job output (see the table above) |
| `warning` | train | `code` (no_props, few_sessions, long_context), `message` |
| `train` | train | `args` (the spsa.py options sent), `steps`, `layers`, `block_size`, `recipe` {`sessions`, `n_ctx`, `context_needed`, `applied` {`concurrency`, `minibatch_chains`}} |
| `step` | train | `step` (-1 is the initial eval), `theta` (block parameters, kinds attention, mlp, residual), `strengths` {`attention`, `mlp`, `residual`: per layer}, `j_plus`, `j_minus`, `eval_j` (null between evals), `plus`, `minus`, `eval` (spsa.py's terms: `j`, `margin`, `drop`, `drop_max`, `margin_min`, `guard_excess`, `guard_drop_max`, ...); with `basis` also `alpha` (basis coefficients) and `angle` (degrees to d0) |
| `best` | train | `step`, `eval`, `settings` (the request settings: per-layer `attention`, `mlp`, `residual`, direction options, `profile` (absent with `basis`)); with `basis` also `alpha`, `angle`, `basis`, `basis_sha256` |
| `upload`, `unload` | profile, load | `kind`, `name`, `reply` (gufo's) |
| `server` | pipeline | `step`, `matched`, `sessions`, `n_ctx` (an `await_server` that matched) |
| `pipeline_step` | pipeline | `index`, `name`, `operation`, `job` (the child key), `status` (running, done, failed, cancelled), `params` (resolved), and at the end `output`, `error`; `reused` (an earlier run's step), `resumed_from` (the earlier child passed as `resume`) |

Fit, test, train, pairs and pipeline jobs use the gufo server for long, so
they run one at a time in arrival order. Dataset, objective, profile and
load jobs run at once. A pipeline runs its steps inside its own run.

Each job is stored in `WORK/jobs/KEY/`:

- `job.json`: the state;
- `results.jsonl`: the results;
- the job's own files.

A restarted service keeps that history. Jobs that were queued or running
fail as "interrupted by a restart", with their results kept.

The CLI writes jobs into the same folder. A running service lists them only
after its next restart, and it would mark a CLI job still running then as
interrupted. CLI fit, test and train jobs do not wait in the service's queue,
so do not run them beside a busy service against the same gufo server.
Stopping the service (Ctrl-C or SIGTERM) cancels its jobs: a fit cancels its
gufo collection, a train stops its spsa.py child.

For a page, `layer` results are the per-layer chart points (`dataset` real or
control), `cosine` is the heat map, `row`/`totals` are the live tables, and
`step` (`j_plus`, `j_minus`, `eval_j`, `strengths`) is the training chart.
Poll with `since` set to the last `seq` you have seen; a pipeline's
`pipeline_step` results name the child job to poll.

## Resume and imported traces

A test (or pairs) job given `resume` reuses finished rows instead of
generating them again, per (configuration, problem id):

- an earlier job's key (or `{"job": KEY}`, or `{"generations": DIR}`): rows
  are reused only when their `request_sha256` equals this job's request;
- `{"config": NAME, "path": FILE}` (FILE may be a glob): imported results for
  configuration NAME. generate.py rows (`build/b1/r5-coding/results/`)
  must match the request through their `settings`; think_eval.py rows
  (`build/b1/think/results/`, `build/b1/r10/results/`) carry no request and
  are trusted (`"trust": false` refuses them). The math request is
  think_eval.py's (the problem plus the boxed suffix, seed 1000 + the index
  in the file, `cache_persist` false, abliteration off), so a configuration
  `{"name": "xhigh", "reasoning_effort": "xhigh"}` with `max_tokens` 32768
  is the one `THINK_EXTRA='{"reasoning_effort":"xhigh"}' THINK_MAX=32768`
  ran.

Math rows are graded again from their content; coding rows keep their
sandbox score. Reused rows are copied into the new job's generations (with
`resumed`), so its generations and totals cover the whole set. A finished
configuration's generations file is rewritten in the set's order, so files
built from it are deterministic.

Checked on the real files: importing `aime24-xhigh.jsonl` and `xhigh32.jsonl`
reuses 30/30 and 100/100 rows (25 and 98 correct, as recorded for R9), the
11 direction-split `think-xhigh-*.jsonl` coding rows match their
`settings`, and an uncapped objective over the imported rows writes
`pairs.jsonl`, `keep.jsonl` and `guard.jsonl` byte-identical to
`build/b1/r9/xhigh/`; the coding guard rows equal `build/b1/r10/code_guard.py`'s.

Coding with thinking: the configuration
`{"reasoning_effort": "xhigh", "sampling": {"chat_template_kwargs": {"enable_thinking": true}, "temperature": 1.0, "top_p": 0.95, "seed": 7, "max_tokens": 32768}}`
(`generation.THINK_XHIGH`) is the R5 `think-xhigh-*` request.

## Objective

The objective job turns traces into spsa.py rows with
`tools/qwen38_flash_next/abliteration/recheck_objective.py` (imported): math
sets give re-check pairs (harmful and redundant re-checks against the forward
thought after them), keep rows (useful re-checks, and first confirmations
with `keep_first_confirmations`) and guard rows (the final answer after the
reasoning of right, finished runs); coding sets give guard rows from answers
that pass every hidden test (one per problem, direction split by default).
AIME 2025 (level 2025 or an `aime2025` id) and the coding eval split are
refused.

Scoring cost grows with the reasoning prefix, and the longest row sets the
training server's context. The default caps are sized for a 16384-token
context:

- `max_prefix_tokens` 12288: pairs and keep rows whose boundary lies further
  into the trace are dropped (`past_prefix_cap`), before the per-trace
  `max_boundaries` sample;
- `max_guard_tokens` 12288 with `guard_cap` skip: guard rows with a longer
  trace are left out (`guard_skipped`); `cut` keeps them, cut at the last
  thought boundary inside the cap (`guard_cut`).

12288 tokens of reasoning, plus a prompt of a few hundred tokens and at most
1024 scored tokens, stay under 16384 with the 10% margin for the token
estimate (tokens are estimated from each trace's characters per generated
token). `context_tokens` in the output is that size for the actual rows.
On the R9 traces (AIME 2024 and MATH 100 at xhigh) the caps keep 31 of 58
pairs, 47 of 51 keep rows and 120 of 123 guard rows, and the longest row is
12,884 tokens (`context_tokens` 14336). An 8192-token context
(`max_prefix_tokens` 6144) would keep only 16 pairs; set 0 to turn a cap off.

## Training server

spsa.py scores theta+ and theta- in one request burst, and with
`--minibatch-chains` it runs every chain of the minibatch at once (a chain's
rows one after another, so each reuses the previous one's prompt
checkpoints). A packing server turns that burst into wide prefill passes
only with many resident sessions. The political SPSA runs measured this
(docs/GFX1151-KNOWHOW.md, "Device scoring log-softmax" and "Main SPSA run":
48 sessions, 8k context, native RoPE, FP16 KV, `--max-connections 160
--max-pending 160`, 8.4 s per step), against about 5 minutes per step for R9
on a 2-session production-shaped server.

The train job therefore defaults to the fast recipe (`fast_defaults`): from
the server's `/props` it sets `--concurrency` to 2 x sessions and, when the
rows carry chains and no minibatch option is given, `--minibatch-chains` to
max(12, sessions). It warns when the server has fewer than 8 sessions or a
context above 65536 or 4x the objective's longest row, and fails before
scoring when the context is shorter than that row. `train` results report the
`recipe` used; set the options explicitly to override it.

Start the training server with the context from the objective
(`context_tokens`, rounded up; 16384 for the default caps), native RoPE (no
`--enable-yarn`), FP16 KV (packing refuses KVarN K4V4), many sessions and
large connection limits:

```sh
gufo serve --sessions 24 --max-connections 160 llm --model <model> \
    --context 16384 --kv-cache-type fp16 --max-pending 160 \
    --allow-profile-updates --logit-bias-profiles <dir>
```

(the production container's other options otherwise; the runtime directory
must hold the base profile, `recheck-avg` for R10). 48 sessions at 8192 is
the measured configuration; at 16384 the KV cache per session doubles, so
the recipe halves the sessions to keep the same KV budget (not measured).
Check device memory after the load before a long run. Scores on this server
differ slightly from the 1M-YaRN K4V4 production numerics (the knowhow
records init J 0.4581 on FP16 against 0.4658 on K4V4 for the political
profile), so compare only runs on the same server, and judge the trained
profile by generation on the production-shaped server.

## R10 pipeline

`examples/r10.json` replaces `build/b1/r10/r10run.sh`. Its paths are
relative to the repository root, so run the CLI, or start the service the
lab page uses, from there:

```sh
python3 -I tools/steer --gufo-url $URL pipeline tools/steer/examples/r10.json
python3 -I tools/steer --gufo-url $URL pipeline tools/steer/examples/r10.json --resume PIPELINEKEY
```

| Step | What | Server |
| --- | --- | --- |
| `traces-aime24`, `traces-math`, `traces-aime2223` | xhigh math traces (32768 tokens, 4 at a time), importing `build/b1/think/results/aime24-xhigh.jsonl`, `xhigh32.jsonl` and `build/b1/r10/results/aime2223-xhigh.jsonl` | generation (`await_server` context >= 40960) |
| `traces-code` | R5 coding direction split at xhigh with thinking (`THINK_XHIGH`), importing `think-xhigh-*.jsonl` and `xhigh10-*.jsonl` (r10run.sh's names) | generation |
| `objective` | the four sets, `keep_first_confirmations`, 8 boundaries, 3 opener tokens, 1024 guard tokens, the default caps | (no gufo) |
| `train` | SPSA on `recheck-avg` (L45 projected, one direction, norm off), residual only, blocks of 4, init 0.5, max 2, lam 2, nu 4, guard margin 0.02, 100 steps, a 0.2, c 0.15, eval every 20, seed 1; concurrency 2 x sessions and minibatch chains max(12, sessions) from the recipe (r10run.sh used 12 chains at concurrency 2: at 24 sessions this is 24 chains per step, a less noisy gradient at twice the cost per step; add `"minibatch_chains": 12` to the options to match R10) | training (`await_server`: >= 16 sessions, context between the objective's `context_tokens` and 32768) |
| `profile` | `recheck-avg` with the best strengths, uploaded as `recheck-r10` | generation again |
| `eval-aime25` | AIME 2025 at xhigh and low, with `recheck-r10` | generation |
| `eval-code` | the coding eval split at low thinking, OFF and ON | generation |

The tool never starts or stops servers. Each `await_server` step waits
(polling `/props` every 10 s) until the operator has the matching server up
on the same URL: the generation server is the production configuration with
`--sessions 4 --allow-profile-updates --collect-dir ... --max-connections 64
--max-pending 64` (as `r10run.sh`'s `start_server 4`), the training server is
the recipe above. A step may also name its own `gufo_url`. The evaluation
sets never feed data: the pipeline refuses any step that reads an AIME 2025
or coding-eval test step's output, and the objective refuses those problems.
A rerun with `--resume` reuses finished steps up to the first one that runs
again (every later step runs again from scratch, since its inputs may have
changed). The first step that runs again, when it is an interrupted test or
pairs step, keeps its finished rows (the earlier child is
added to its `resume` list); an interrupted train step continues its spsa.py
log when its params are unchanged, with the earlier job's recipe values.

## R11: direction and strengths

R10 learns per-layer residual strengths with the direction fixed to
`recheck-avgx` (the L40-L47 projected rows of the xhigh-trace fit, averaged
and sign-aligned to L45). R11 also moves the direction, inside a small
subspace, starting from R10's best strengths.

1. **Basis (offline, NumPy; pinned container).**
   `tools/qwen38_flash_next/abliteration/direction_basis.py` reads the
   collection the direction was fitted from and pools each pair's projected
   difference at L40-L47. Each layer's rows are scaled to a unit mean and
   sign-aligned to L45, as `build/b1/r10/newavg.py` aligns them. It removes
   the component along d0 (the base profile's L45 row) and writes d0 and the
   top-k principal axes of the rest, with their explained variance, as a
   `gufo-direction-basis-v1` file:

   ```sh
   python3 -I tools/docker/pinned_rocm101.py run --build-dir build/lbias --source . -- sh -c \
       'cd /build/source && /opt/venv/bin/python3 -s -I tools/qwen38_flash_next/abliteration/direction_basis.py \
          --collection build/b1/r10/collect/rcx-e1142a4f3142 \
          --base-profile build/b1/r10/rt10/abliteration/recheck-avgx.json \
          --layers 40-47 --ref-layer 45 --direction-layer 45 -k 6 --output build/b1/r11/basis-avgx-k6.json'
   ```

2. **Training.** `spsa.py --basis` (spsa.py's docstring, "Direction")
   learns theta = [strength blocks, alpha]. The direction is
   d(alpha) = normalize(d0 + sum alpha_k u_k), with |alpha| capped at
   tan(`--max-angle`), so the direction stays within that angle of d0.
   alpha has its own perturbation and step sizes and its own random stream.
   Before each burst, spsa.py uploads the profile of theta+ and theta- under
   two of the `--dir-names` (the third is for evals). Each upload replaces
   the previous one, and all are deleted at the end. The run refuses to
   start when the loaded profiles plus these names would pass gufo's cap of
   8 (two names then fit one more). The check runs once at start, so do
   not upload other profiles to the training server during the run.
   alpha = 0 keeps the base rows bit for bit, so it scores exactly as the
   fixed direction (the upload's bytes, and so its cache identity, differ
   from the base file's).

   | Train option | spsa.py | Default |
   | --- | --- | --- |
   | `basis` (param) | `--basis` | (none: fixed direction) |
   | `base_profile` (param) | `--base-profile` (set by the job) | required with `basis` |
   | `max_angle` | `--max-angle` | 20 (degrees) |
   | `c_dir`, `a_dir` | `--c-dir`, `--a-dir` | 0.04, 0.05 (strengths: `c` 0.15, `a` 0.15) |
   | `dir_names` | `--dir-names` | `r11-p,r11-m,r11-e` |
   | `init_alpha` | `--init-alpha` | 0 (or the alpha of a previous `--output` with the same basis) |
   |  | `--profile-output` (set by the job: `best-profile.json`) | |

   Leave `profile` out: a direction run scores only its own uploads. Step
   results add `alpha` and `angle` (degrees to d0), and `best` adds
   `alpha`, `angle`, `basis` and `basis_sha256`. The trained profile
   (`profile_name`, the `profile` operation, `load` from the train job) holds
   the tuned L45 row: `projected_directions[45]` and `directions[45]`, as
   `newavg.py` writes them. The lab page's Train view adds the angle and the
   alpha curves.

3. **Pipeline.** `examples/r11.json` runs train, profile (`recheck-r11`,
   uploaded), then AIME 2025 and the coding eval as `r10.json` does. It uses
   R10's objective job, so run it with R10's work folder, after R10's
   evaluation:

   ```sh
   python3 -I tools/steer --work build/b1/r10/steer10 --gufo-url $URL pipeline tools/steer/examples/r11.json
   ```

   The train step starts from R10's best (`init`:
   `build/b1/r10/steer10/jobs/631b98c155de13d8f1ec90ee5ab4b61c/best.json`).
   It uses the same objective rows, guard, chain ordering and minibatch (12
   chains), 60 steps, strength `a` 0.1, eval every 10. Each theta costs one
   upload of the 2.3 MB profile (about 1 s): two per step and one per eval,
   on top of R10's scoring.

## Lab page

The service also serves a single-page lab (`web/`: plain HTML, CSS and JS,
no build step, nothing fetched from another host):

```sh
python3 -I tools/steer --work DIR --gufo-url URL --collections-dir COLLECT serve
# then open http://127.0.0.1:18790/
```

`GET /` and `GET /web/FILE` return the files in `web/` (hidden names, `..`
and unknown file types are refused; the page's Content-Security-Policy
allows only this service). Every step goes through the job API above; the
page polls `GET /jobs/{key}?since=N` every 1.5 s while a job runs.

| Tab | Starts | Shows |
| --- | --- | --- |
| Datasets | `dataset` (traces, pairs, import), `pairs` (model-generated) | `/datasets`; records paged (`/datasets/{name}/records`); for a traces dataset, each thought boundary labelled "before a re-check" or "moving on" next to its shuffled control's label |
| Run (fit) | `fit` | collection progress; per-layer held-out accuracy and SNR (real against control), \|mean D\|, the cross-layer cosine heat map, the real/control cosine, transfer, top layers; default profiles with upload (`POST /profiles/...` `{job, profile}`) |
| Train | `train` | J+ and J- per step with eval J, eval margin and drop, per-layer strengths (any step, next to the best eval), warnings and recipe; `profile` (build, or build and upload) and the trained profile into a test configuration |
| Test and tune | `test` | configurations with per-layer strength sliders (all equal, a layer range, block averages, or a train job's strengths), logit bias, effort and sampling; per-configuration totals against the baseline (the first configuration), per-problem rows with a diff of two configurations; "try next setting" clones a configuration into a new job compared with the old one |
| Pipeline | `pipeline` (a JSON file such as `examples/r10.json`, edited in place) | each step's status and child job (`pipeline_step`), linked to its own view |
| History | re-run with the same params | `/jobs` with cancel/delete, any job's results; objective counts; profile files with upload |

The header shows the service and gufo URL, the loaded profiles
(`/gufo/props`), and a warning when the gufo URL uses the production port
18740 or gufo refused an upload (no `--allow-profile-updates`). Promoting a
profile into the runtime profile directory is a production change: the page
only shows the command to run.

## Load

```sh
python3 -I tools/steer --gufo-url URL load --name recheck-p45 --job FITKEY
python3 -I tools/steer --gufo-url URL load --kind logit-bias --name wait --file docs/examples/logit-bias/wait.json
python3 -I tools/steer --gufo-url URL unload --name recheck-p45
```

## Reproducing `build/b1/think/` end to end

`T=build/b1/think`. Start an experiment server with production settings,
`--collect-dir $COLLECT --allow-profile-updates`, and
`--logit-bias-profiles` (for the wait profile). `URL` is its address.

1. **Traces** (`think_eval.py run`). Medium-effort traces on the problems not
   used for evaluation:

   ```sh
   steer() { python3 -I tools/steer --gufo-url $URL --collections-dir $COLLECT "$@"; }
   steer test --set-kind math --set $T/aime24.jsonl --max-tokens 32768 --concurrency 4 \
       --config 'medium={"reasoning_effort":"medium"}'
   steer test --set-kind math --set $T/math.jsonl --config 'base={}'
   ```

   Each job's `generations/CONFIG.jsonl` has the `reasoning` field that the
   builders read.
2. **Datasets** (`build_think_dirs.py`):

   ```sh
   S=$(printf '\n\nPut the final answer in \\boxed{}.')
   steer dataset traces --name recheck --suffix "$S" \
       --source aime24=$T/aime24.jsonl,$AIME_JOB/generations/medium.jsonl \
       --source math=$T/math.jsonl,$MATH_JOB/generations/base.jsonl
   steer dataset pairs --name depth --mode segments --suffix "$S" \
       --instruction 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' \
       --source aime24=... --source math=...
   ```

   This writes `recheck`, `recheck-shuf` and `depth`. Given the original
   traces (`$T/results/aime24-medium.jsonl`, `$T/results/base.jsonl`), the
   records equal `$T/dir/{recheck,recheck-shuf,depth}.jsonl` exactly
   (checked: 82, 82 and 260 records).
3. **Collect and fit** (`step3.sh`, `analyze.py`, `xlayer.py`, `mksup.py`).
   `steer fit --dataset recheck --profile-layers 32,45` (venv), or
   `--collect-only` on the host followed by `fit --collection` in the
   container. On the existing collections
   (`--prefix recheck --collection $T/dir/col-recheck --control-collection $T/dir/col-recheck-shuf`):
   - The per-layer rows equal `$T/dir/analyze.txt`, for example L45:
     41/41, SNR 3.80, |d| 5.091; and on the control, 26/41, SNR 0.29.
   - The top layers are 45, 46, 47, 42.
   - `recheck-add` equals `$T/profiles/recheck.json`.
   - `recheck-p32`/`-p45` equal `$T/rt/abliteration/recheck-p{32,45}.json`,
     except for mksup's ignored extra member `strength`.
4. **Load**: `steer load --name recheck-p45 --job FITKEY` (and the others).
   No restart is needed.
5. **Strength grid** (`step4.sh`, `grid4.json`): one test job on the 15
   longest AIME 2024 problems at low effort:

   ```sh
   steer test --set-kind math --set $T/aime24.jsonl --ids-file $T/aime24-dev-ids.json \
       --max-tokens 32768 --concurrency 4 \
       --config 'base={"reasoning_effort":"low"}' \
       --config 'p45={"reasoning_effort":"low","abliteration":{"enabled":true,"profile":"recheck-p45"}}' \
       --config 'wait2={"reasoning_effort":"low","logit_bias_profiles":["wait"],"logit_bias_overrides":{"wait":-2}}' \
       ...
   ```

   The `rc05`/`rc1` configurations send `recheck-add` with `residual` set to
   -0.5 and -1 times the profile's residual.
6. **Train** (optional): `steer train -- --pairs P --ordinary O --init 0 --additive
   --min-strength -64 --max-strength 64 --profile recheck-add --steps 50`.
   This is spsa.py on teacher-forced scores; the test job is the
   generation-grid alternative.

## Not done

- Events pushed over server-sent events; polling is the contract.
- Fit does not choose `alpha` or strengths; that is the test grid's and SPSA's job.
- Prompt-set generation for a named behavior (design C+ 3) and the
  three-way useful/harmful/neutral labeller as a dataset builder.
- Token counts in the objective are estimates (characters per generated
  token); gufo's `/tokenize` could make the context size exact.
