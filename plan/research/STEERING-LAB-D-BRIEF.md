# Brief D: the steering lab page

Build the lab page from `plan/research/STEERING-API-DESIGN.md` section D: a web page served by the steer tool's HTTP service (`tools/steer/service.py`, port 18790, bound to 127.0.0.1). It runs every experiment step through the tool's job API. Another agent is extending the backend at the same time (`plan/research/STEERING-API-CPLUS-BRIEF.md`): new operations `objective`, `profile`, `pairs` generation and `pipeline`, resume for `test`, and read-only listing endpoints. Code against the job API and the README's result-type table (`tools/steer/README.md`). Re-read it before finishing, since the other agent updates it. Where an endpoint is not there yet, degrade gracefully (show the raw JSON results).

Read: the design doc (D, Jobs, C+), `tools/steer/README.md`, `service.py`, `jobs.py`, `ops.py`, and the existing result types in `fitting.py`, `generation.py`, `training.py`, `data.py`.

## Pages (one single-page app, plain HTML/CSS/JS, no build step, no external network)

The page is opened from the user's machine, possibly offline. Vendor nothing from a CDN; draw charts with inline SVG or canvas.

1. **Datasets:**
   - list and create datasets: the traces builder, the pairs builder or the model-generated pairs job, and import;
   - browse records paged;
   - for trace datasets, show the comparison groups (e.g. "before a re-check" against "moving on") and the shuffled control.
2. **Run (fit):**
   - pick a dataset and parameters, then start;
   - live charts, polling `GET /jobs/{key}?since=N` every 1–2 s:
     - per-layer held-out separation and SNR (bars or line) as `layer` results arrive;
     - the cross-layer cosine heat map;
     - the control next to the real labels;
     - the top layers and the default profiles produced, with an upload button.
3. **Train:**
   - start a train job (base profile, learn kinds, block size, steps, a, c, minibatch, guard settings);
   - live J per step (J+ and J− and eval J);
   - per-layer strengths as a bar chart updated each step;
   - best so far; buttons to build and upload the trained profile.
4. **Test and tune:**
   - choose a set (math or coding), configurations (profile plus per-layer strength sliders for attention/mlp/residual, with presets: all-equal, block, from a train job; logit-bias profiles and overrides; `reasoning_effort`; sampling);
   - run against a baseline configuration;
   - a live table per configuration (correct, mean/median tokens, truncated, hesitations, seconds), filled from `row` and `totals` results, and per-problem rows with a diff view between configurations;
   - "try next setting": clone a configuration with adjusted sliders and run it.
5. **Pipeline:** load a pipeline JSON (such as `tools/steer/examples/r10.json`), edit params, run; show each step's status and drill into its job view.
6. **History:**
   - every job with its operation, params, status and times, from `GET /jobs`;
   - open any job's view, re-run with the same params, delete.
   - Promoting a profile into a runtime directory is a production change: show the command to run, never perform it.

Header: the gufo URL and health (`/health`), the loaded profiles (gufo `/props` through the tool when available), and a warning banner if the gufo URL looks like production (port 18740 with no `--allow-profile-updates`, i.e. profile upload refused).

## Service change

Add one route in `service.py`: `GET /` and `GET /web/<file>` serve `tools/steer/web/` files with correct content types, path-traversal safe. Touch nothing else in `service.py`. If you need a backend endpoint that does not exist, write it down in your report instead of adding it.

## Tests

- unittest for the static route (types, traversal refused);
- a small JS-free check that the page's HTML references only local files;
- run `python3 -I -m unittest discover -s tools/steer -p 'test_*.py'`.

Use the `artifact-design`/`dataviz` principles for charts if available to you: thin marks, a legend for two or more series, no dual axes, dark and light themes.

## Do not

- Use the GPU, call a live gufo server, start/stop `llm` or any `gufo-*` container, push, or use `pgrep -f`/`pkill -f`.
- Modify files outside `tools/steer/web/**`, the one route in `tools/steer/service.py`, its test, and a "Lab page" section in `tools/steer/README.md`.

## Commit and review

Single-line Conventional Commit messages ending with:
```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016Kh8svoeuCWVkepKUHx2XF
```
Stage only your files. Before finishing, have a second reviewer agent review the diff, fix, and re-review.

## Report

Screens and what each calls, how to open the page (`python3 -I tools/steer --work DIR --gufo-url URL serve`, then the browser URL), endpoints you assumed or found missing, tests, review findings.
