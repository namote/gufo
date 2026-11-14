# Brief C+: self-generated data, objective, fast training and pipelines in `tools/steer`

The user wants every step of a steering experiment (R10: re-check suppression training with more data) to run through the tool's job API and its lab page (D, built in parallel by another agent), not hand-run scripts. Build the backend here; the page only calls the job API.

Read:
- `plan/research/STEERING-API-DESIGN.md` (C, Jobs, C+, D);
- `tools/steer/README.md` and its code;
- `docs/SERVER.md` (score_assistant with `reasoning_content`/`score_reasoning`, collection jobs, runtime profile updates);
- `tools/qwen38_flash_next/abliteration/{recheck_objective.py,spsa.py}`;
- `tools/qwen38_flash_next/coding_eval/`;
- the hand-run R10 pipeline this must replace: `build/b1/r10/{r10run.sh,code_guard.py,aime2223.jsonl,code-todo.txt}` and `build/b1/r9/r9run.sh` (read only);
- `docs/GFX1151-KNOWHOW.md`, the sections "Held-out AIME 2025 and fixed-direction training (R9)" and the political SPSA speed notes ("Device scoring log-softmax", "Main SPSA run": 8.4 s/step at 48 sessions, 8k native-RoPE context).

## Build

1. **Trace generation reuses `test`.** A test job already generates math and coding answers with reasoning and grades them. Add only what R10 needs:
   - `resume`: a new job given an earlier test job's key, or a generations folder, skips finished (config, id) rows;
   - math per-problem seeds as `think_eval.py`, so existing `build/b1/think/results/*.jsonl` and `build/b1/r10/results/*.jsonl` rows can be imported as finished generations (an import option, or a `traces` dataset from those files);
   - coding with thinking on and `reasoning_effort` (the R5 `think-xhigh-*` settings: temperature 1.0, top_p 0.95, seed 7, 32768 tokens).
2. **`objective` job:** test generations (or imported trace files) to SPSA rows, wrapping `recheck_objective.py` (import it, don't copy it). Requirements:
   - math sets give pairs/keep/guard, and coding sets give guard rows from answers that pass every hidden test (as `build/b1/r10/code_guard.py`);
   - new options to bound scoring cost:
     - a cap on the boundary prefix length (tokens or characters of reasoning before the boundary; boundaries past it are dropped);
     - a cap on guard reasoning length (skip, or cut at a thought boundary, guard rows whose trace is longer).
   - Results: label counts per set and label, with examples. Output: the row files and their counts.
3. **Train job:**
   - `--max-prefix` style options flow from the objective, and the tool's docs give the fast training-server recipe the political SPSA used: short context, native RoPE (no YaRN), many sessions, `--max-connections`/`--max-pending` large, concurrency near 2 × sessions. Size the context from the objective's longest row. Note in the docs that scores then differ slightly from the 1M-YaRN production numerics, as recorded for the political profile.
   - Results: the per-step J+/J−, theta per layer and evals, so the page can chart them.
   - The user says they already optimized SPSA earlier. Keep `spsa.py`'s interface; if you find uncommitted or newer SPSA work anywhere in the repo, use it rather than duplicating.
4. **`profile` from a train job:** build the trained profile (the base profile with the best strengths) and optionally upload it (`POST /profiles/...` with `{"job": KEY}` should accept train jobs too).
5. **Instruction-pair generation (C+ item 2):** a `pairs` generation job: the model answers each prompt with and without an instruction, and the result becomes a CAA dataset via the existing pairs builder.
6. **`pipeline` job:** an ordered list of steps (`test`, `objective`, `train`, `profile`/`load`, `test`), each step's params able to reference an earlier step's output (e.g. `{"from": 0, "key": "generations"}`). Progress shows the current step, and cancel cancels the running child. This is what the page's "run" button and the R10 rerun will use. Write `tools/steer/examples/r10.json`, the R10 pipeline:
   - data: AIME 2024 + MATH 100 + AIME 2022–2023 xhigh traces (import what exists, generate the rest), plus R5 coding direction-split xhigh traces;
   - objective with the caps;
   - SPSA, 100 steps, residual blocks of 4, init 0.5, direction fixed from `recheck-avg`;
   - upload;
   - AIME 2025 at xhigh and low;
   - the coding eval split at low, OFF and ON.

   AIME 2025 and the coding eval split must never feed data or objective.
7. **Server-side resources the page needs (read only):** list a job's generations and rows paged, a dataset's records paged, profile files in the work folder, and gufo `/props` passthrough. Add result `type`s rather than ad-hoc shapes, and document every type's fields in the README (the page agent codes against that table).

## Ownership

The page agent owns `tools/steer/web/**` and adds one static-file route in `service.py`. You own everything else in `tools/steer/` and may touch `recheck_objective.py`. Coordinate through the README's result-type table; do not edit `tools/steer/web/`.

## Do not

- Use the GPU, call a live gufo server, start/stop `llm` or any `gufo-*` container, push, or use `pgrep -f`/`pkill -f`. Tests use the fake gufo server.
- Modify `build/b1/` (read only), `~/.local/share/gufo/`, `tools/exl3_quant/hc/`, `plan/exl3/HC-Q8-CALIBRATION.md`.

## Commit and review

Single-line Conventional Commit messages ending with:
```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016Kh8svoeuCWVkepKUHx2XF
```
Stage only your files. Run `python3 -I -m unittest discover -s tools/steer -p 'test_*.py'` and the recheck_objective tests. Before finishing, have a second reviewer agent review the diff, fix, and re-review.

## Report

New operations and params, result types, the R10 pipeline file and its estimated cost per stage (with the caps), the training-server recipe, tests, review findings, anything left open.
