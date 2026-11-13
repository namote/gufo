# Steering tool (fit, train, test)

A Python tool and HTTP job service, separate from gufo, for steering
experiments on Qwen3.8-Flash-Next. It only calls a gufo server, using the
endpoints in [docs/SERVER.md](../../docs/SERVER.md):

- runtime profile updates (`--allow-profile-updates`);
- activation collection jobs (`--collect-dir`);
- Chat Completions.

Design: [plan/research/STEERING-API-DESIGN.md](../../plan/research/STEERING-API-DESIGN.md) (C and Jobs).

| Path | Role |
| --- | --- |
| `__main__.py`, `cli.py` | Command line (`python3 -I tools/steer COMMAND`); every command runs as a job through the same code as the service |
| `service.py` | The HTTP job API (standard library `http.server`) |
| `jobs.py` | Keyed jobs persisted under the work folder, the FIFO worker and cancellation |
| `data.py` | Datasets and their builders (traces, CAA pairs, import) |
| `fitting.py` | Fit job: gufo collection job, then per-layer analysis and default profiles (NumPy) |
| `generation.py` | Test job: a math or coding generation set under configurations |
| `training.py`, `spsa_child.py` | Train job: `spsa.py` as a child process, with progress per step |
| `client.py` | gufo HTTP client (base URL and optional API key) |
| `ops.py` | Operation table, job context and profile upload sources |
| `test_steer.py` | Unit tests against a fake gufo server |

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
| `test` | `set` {`kind` math\|coding, `path`, `split`, `ids`, `limit`, `max_tokens`, `suffix`, `seed_base`, `splits`, `image`}, `configs`, `concurrency` (2), `model`, `request_timeout` | `config`, then `row` and running `totals` per finished problem | totals per configuration, generations folder |
| `train` | `args` (spsa.py options; the tool sets `--base-url`, `--log`, `--output`), `python` | `train`, `step` (one per spsa.py log line), `best` | log path, best settings |

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
| `GET` | `/datasets`, `/datasets/{name}` | dataset.json of every dataset, or of one |
| `POST` | `/profiles/{abliteration\|logit-bias}/{name}` `{"file": PATH}` \| `{"job": KEY, "profile": NAME}` \| `{"profile": {...}}` | gufo's reply to the upload |
| `DELETE` | `/profiles/{abliteration\|logit-bias}/{name}` | gufo's reply |
| `GET` | `/health` | status and the gufo URL |

Errors are `{"error": {"code", "message"}}`.

Fit, test and train use the gufo server, so they run one at a time in
arrival order. Dataset jobs run at once.

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
control), `cosine` is the heat map, and `row`/`totals` are the live tables.
Poll with `since` set to the last `seq` you have seen.

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

- The lab page (design D).
- Events pushed over server-sent events; polling is the contract.
- Resuming an interrupted test or train job: start a new job.
- Fit does not choose `alpha` or strengths; that is the test grid's and SPSA's job.
