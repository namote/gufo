# Brief C: steering tool (fit, train, test) with a polled job API

Read `plan/research/STEERING-API-DESIGN.md` (C, Jobs, D), `docs/SERVER.md` (runtime profile updates, activation collection jobs, abliteration, logit-bias profiles), and the existing scripts it wraps:
- `tools/qwen38_flash_next/abliteration/{fit_steering.py,prepare.py,spsa.py,fit_trigger.py}`,
- `build/b1/think/{think_eval.py,build_think_dirs.py,analyze.py,xlayer.py,mksup.py,step4.sh}` (read only; this is the recipe the tool must reproduce),
- `tools/qwen38_flash_next/coding_eval/` (generation plus a sandboxed grader).

Also read the repo's CLAUDE.md/AGENTS.md.

## Build `tools/steer/`

A Python package and HTTP service, separate from gufo, that only calls a gufo server (base URL and optional API key from options).

1. **Datasets:** a dataset is a folder of JSONL in the collector's record formats (collect/answers/segments) plus `dataset.json` (kind, groups, source, created_at). Provide:
   - **Build from traces:** split reasoning at blank lines and label the thought ends by a regex for the next thought's opener, as `build_think_dirs.py` does, with a character cap per trace and a shuffled-label control dataset made alongside.
   - **Build from instruction pairs:** CAA style, as R5b: the same answer under instruction plus prompt against the bare prompt.
   - **Import** an existing JSONL.
2. **Fit job:** dataset → gufo collect job (`POST /v1/abliteration/jobs`, polled) → fit. The output:
   - a per-layer table (held-out separation, SNR, |d|);
   - a cross-layer cosine table;
   - the same for the control dataset;
   - the top-N layers;
   - default profiles: suppress from one layer, the pol-avg recipe (projected, residual 1.0 everywhere, attention/mlp 0), and additive. They are built through `prepare.make_profile` / `fit_steering` so they validate as the loader does.

   NumPy is required: run where NumPy exists (the host has none; document how to run it in the pinned container's `/opt/venv/bin/python3 -s`, or a local venv).
3. **Test job:** a generation set (math with boxed answers as `think_eval.py`, or a coding set via coding_eval) under a list of configurations. Each configuration has:
   - `abliteration` (profile, overrides);
   - `logit_bias_profiles`/`logit_bias_overrides`;
   - `reasoning_effort` and sampling.

   It streams per-problem rows (correct, tokens, hesitations, truncation, seconds) and running totals per configuration. Concurrency is an option (the server's session count).
4. **Train job:** wrap `spsa.py` (direction fixed, per-layer strengths) with progress per step.
5. **Load:** upload a profile to gufo with `POST /v1/abliteration/profiles/{name}` (or logit-bias), and delete it.
6. **Job API (the tool's own HTTP server, standard library only):**
   - `POST /jobs` returns a key;
   - `GET /jobs/{key}?since=n` returns status, progress, results after n, and errors;
   - `GET /jobs`, `DELETE /jobs/{key}`.

   Jobs and results persist on disk under a work directory, so a restart keeps history. Bind to 127.0.0.1 by default.
7. **CLI** for each operation (the same code path as the jobs).
8. **Tests:** unittest with a fake gufo HTTP server (collect jobs, profile upload, chat completions), the dataset builders (labels, caps, shuffled control), fit on synthetic activations, job lifecycle and `since` paging. Tests needing NumPy run in the pinned container.
9. **README** in `tools/steer/` with the workflow that reproduces `build/b1/think/` end to end.

The page (D) is not part of this brief, but design the job results so a page can draw per-layer charts and live tables from them.

## Do not

- Use the GPU or call a live gufo server (production on 18740 and experiment servers are in use); tests use fakes only.
- Start/stop `llm` or any `gufo-*` container, push, or use `pgrep -f`/`pkill -f`.
- Modify `~/.local/share/gufo/`, `build/b1/` (read only), `tools/exl3_quant/hc/`, `plan/exl3/HC-Q8-CALIBRATION.md`.

## Commit and review

Single-line Conventional Commit messages ending with:
```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016Kh8svoeuCWVkepKUHx2XF
```
Stage only your files. Before finishing, have a second reviewer agent review the diff, fix what it finds, and have it re-review.

## Report

Layout, CLI and job API, what each job returns, how it reproduces the think experiment, tests and results, review findings, anything left undone.
