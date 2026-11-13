# Steering API: load, collect, train (design)

Status: design, November 13, 2026. Item A is being implemented (`STEERING-API-A-BRIEF.md`); B and C follow after A is reviewed.

## Why

Every steering experiment today needs a production stop:
- the collector (`tools/qwen38_flash_next/abliteration/collect.cpp`) loads a second copy of the model (64 GB);
- new profiles are only read from `--abliteration-profiles DIR` at startup.

Restarts caused most of the day's failures: start races between containers, and a host-bound cache budget that starved admission (`cf79d767`). Generation-based checks (accuracy plus thinking length) also showed that teacher-forced scoring alone cannot tune every behavior.

## What exists

- Several profiles per request (up to 8), each with its own direction and per-layer strengths; overlapping directions are orthonormalized per site (docs/SERVER.md, "Several profiles").
- Per-request per-layer strengths (`attention`, `mlp`, `residual`), and per-request `directions` for suppression profiles.
- Additive (`mode: "add"`) profiles, triggers with prompt and decode thresholds, logit-bias profiles (`--logit-bias-profiles`).
- Teacher-forced scoring (`score_assistant`) and external SPSA (`spsa.py`).

## A. Load profiles at runtime (server)

- `POST /v1/abliteration/profiles` with a profile JSON (same schema and validation as the startup loader); the name comes from the body or the request path. `DELETE` removes one; `GET /props` already lists them.
- Disabled unless the server is started with an explicit option, and refused without the server's API key when one is set. It changes answers for every client.
- New and in-flight requests: a request snapshots the profile set at admission; replacing or deleting a profile never changes a running request. Cache identities already include the profile settings, so a replaced profile cannot reuse the old state.
- Same for logit-bias profiles (`/v1/logit-bias/profiles`), sharing the code path.

## B. Collect activations in the server

- `POST /v1/abliteration/collect`: records in the collector's JSONL formats (`collect`, `answers`, `segments`), returns per-record per-layer means (FP32, HC mean, the same readout as collect.cpp) or writes them under a server-side directory given by the option.
- Runs as low-priority work through the scheduler: it yields to chat requests and uses the loaded model (no second copy).
- Outputs must equal collect.cpp's for the same records, which is the acceptance test.

## C. Fit and train (outside the server)

A separate Python tool (`tools/steer/`), which only calls the server:
- **Fit:** dataset folder (groups and row selection) → collect via B → direction per layer, held-out separation and SNR per layer, the top-N layers, a cross-layer cosine table, a shuffled-label control, and a default profile (suppress from one layer, the pol-avg recipe, or additive).
- **Train:** direction fixed; per-layer strengths by SPSA on teacher-forced scores (existing), or a generation grid on accuracy and length (as in `build/b1/think/step4.sh`).
- **Load:** upload the result via A, without a restart.

## D. Steering lab page (TODO after A–C)

Requested by the user on November 13. A web page over C (and through it A and B), served by the external tool, not by gufo:
1. **Datasets:** create a dataset (prompts or traces, from files or generated), browse and edit records.
2. **Comparison groups:** define the two groups and the rows to read (last prompt row, answer tokens, thought boundaries such as "before a re-check" versus "moving on"), with a shuffled-label control.
3. **Run:** collect and fit with one button. Per-layer charts appear as results arrive, pushed live:
   - held-out separation and SNR per layer,
   - the cross-layer cosine heat map,
   - the control next to the real labels,
   - the recommended layers.
4. **Test and tune:** pick a recipe (suppression from one layer, additive, logit bias) and strengths with sliders, upload via A, and run a generation set (accuracy, length, hesitations, truncations) against a baseline. The result table and per-problem rows stream in live, so the next setting can be tried at once.
5. **History:** every run with its settings, and a button to save a profile or to promote it into the runtime's profile directory (a production change: confirm with the user).

## Order

A, then B (the largest gain: no production stop for collection), then C (wrapping the existing scripts), then D (the page).
