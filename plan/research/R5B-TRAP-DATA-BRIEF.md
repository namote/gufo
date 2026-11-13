# R5b brief: trap-heavy coding problems

Background: `plan/research/ABLITERATION-RESULTS-20261112.md` §9 and `docs/GFX1151-KNOWHOW.md` (pass 15 stages 3–5). The first R5 set (`build/b1/r5-coding/`, 105 problems) was too easy. The model passes 81/105. Of the 24 failures, 14 were answers cut off at 600 tokens, and only 10 were real mistakes. A steering direction for "check the pitfalls" needs many problems where the model actually falls into a trap.

Real mistakes the model made (use these as the pattern to scale up):

| Kind | Example |
|---|---|
| Month ends and leap years | adding a month to Jan 31; `date + int` instead of `timedelta` |
| Date range limits | `OverflowError` near `date.max` |
| Non-ASCII digits | accepting "５" because `str.isdigit()` is True |
| Large or deep input | recursion limit on deep nesting; quadratic time on large lists |
| Touching ranges | merging [1,2] and [3,4] when only overlaps should merge |
| Missing type checks | not raising the `TypeError` the task asks for |
| Input format assumptions | unpacking a fixed number of fields from a line that has fewer |

## Build (`build/b1/r5-trap/`)

1. **60 or more new problems**, in the same format as `build/b1/r5-coding/problems.jsonl`. Read its README and `src/` first, and reuse `src/common.py` and the build/check flow.
   - Every problem contains at least one realistic trap from the table above or a similar one: float rounding and `round()` banker's rounding, `str.split()` versus `split(' ')`, case folding (`casefold` versus `lower`, "ß"), Unicode normalization, surrogate pairs and emoji length, integer division with negatives (`//`, `%`), mutable default arguments, aliasing (`[[0]*n]*m`), dict mutation during iteration, sort stability and key ties, time zones and DST, `datetime` versus `date` comparisons, off-by-one in ranges and slicing, empty versus None, `bool` being an `int`, and so on.
   - The trap must be stated or implied clearly by the task text: a careful programmer reading the task would handle it. No hidden gotchas the task does not specify.
   - The hidden tests must include the trap cases (`edge` labels) and ordinary cases.
   - Mix Korean and English instructions about half and half. Mix difficulty.
   - Write the problems yourself; do not copy known benchmark problems.
2. **Splits:** direction 30, strength 15, eval 15 or more (held out). Assign deterministically, as `build.py` does.
3. **Canonical solutions** for every problem, passing all hidden tests (CPU, Docker sandbox, as in `check_references.py`).
4. **Reference answers are not needed.** The next step pairs the model's own answers with and without a pitfall instruction.
5. **Difficulty check without the model:** for each problem, write a short "naive" solution that a hurried programmer would write and that falls into the trap. Confirm that it fails at least one hidden test. Record these in `naive.jsonl` and report how many fail.
6. **README.md** in the set directory: files, counts by split, language and trap kind, check results, and how to run `generate.py` and `evaluate.py` on this set (they take `--set`).

## Do not

- Use the GPU or call the model server (production runs on 18740).
- Modify `~/.local/share/gufo/`, `tools/exl3_quant/hc/`, `plan/exl3/HC-Q8-CALIBRATION.md` or `build/b1/r5-coding/`.
- Push. Use `pgrep -f`/`pkill -f`.

## Commit

The data stays uncommitted in `build/b1/r5-trap/`. If you change a tool under `tools/qwen38_flash_next/coding_eval/`, keep its unit tests passing (`cd tools/qwen38_flash_next/coding_eval && python3 -m unittest test_coding_eval`) and commit with a single-line Conventional Commit message ending with:

```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016Kh8svoeuCWVkepKUHx2XF
```

## Report

Give the file list, counts (split, language, trap kind), the solution check result, the naive-solution failure count, and anything you could not do.
