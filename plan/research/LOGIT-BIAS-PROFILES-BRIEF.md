# Brief: logit-bias profiles (generalize the Han scale)

## Why

Thinking-depth steering needs a trivial baseline: suppress re-check openers ("Wait", "Hmm", "Alternatively", ...) in the reasoning phase only. Today the server rejects the OpenAI `logit_bias` field (`src/cli/serve/sampling_request.hpp`), and the only token-level control is the Han scale (`--han-scale`, `--han-scale-scope`, `src/core/sampling.*`, `src/core/han_script.hpp`), which applies one bias to one fixed token set. Generalize that mechanism into named, file-loaded token-set biases that can be combined (several at once), like abliteration profiles.

## What to build

1. **Profile files.** `--logit-bias-profiles DIR` loads every `*.json` in DIR at startup. One file is one named set:
   ```json
   {"format": "gufo-logit-bias-v1", "name": "wait", "scope": "reasoning",
    "bias": -4.0, "tokens": [1234], "strings": ["Wait", "Hmm", "Alternatively"]}
   ```
   - `bias`: additive logit offset (negative suppresses). Finite, bounded (choose and document a bound, e.g. [-100, 100]).
   - `scope`: `all`, `answer` (after the reasoning phase, as Han `answer`), or `reasoning` (only inside the reasoning phase: new).
   - `tokens`: vocabulary ids. `strings`: expanded at load time against the model vocabulary. Choose a precise, documented rule (for example: every token whose decoded text, after stripping leading whitespace, equals the string exactly; list case variants explicitly in the file). Log how many tokens each profile resolved to; reject a profile that resolves to none.
   - Validate strictly (unknown keys, bad scope, ids out of range) and fail startup with a clear message, as the abliteration profile loader does.
2. **Requests (multi).** Select any number of loaded profiles per request, up to a documented cap (8): e.g. `"logit_bias_profiles": ["wait", "hmm"]`, with optional per-profile bias overrides (`"logit_bias_overrides": {"wait": -6}`). Unknown names are a 400 error. Biases of overlapping sets add. Also accept the standard OpenAI `logit_bias` map (token id string to bias) as an ad hoc set with scope `all`; keep rejecting malformed values. Pick request field names consistent with existing ones and document them; one canonical name per behavior, no aliases.
3. **Han stays as is.** `--han-scale` / `han_scale` / `han_scale_scope` keep their exact behavior and defaults (production uses `--han-scale 0.1 --han-scale-scope answer`). It is fine, even preferred, to implement Han as one more internal set on the same path, as long as outputs with no logit-bias profiles active are unchanged.
4. **Every sampling path that applies Han must apply these sets identically**: the CPU sampler (`src/core/sampling.cpp`), the Qwen HIP executor (`src/models/qwen/hip/executor.cpp`), dflash (`src/models/qwen/hip/dflash/executor.cpp`), and Flash-Next MTP draft and verification (`src/models/qwen38_flash_next/runtime/executor_mtp.cpp`, `kernels/rocm/mtp.hip.cpp`). Draft and target must see the same biased distribution, or speculative sampling stops being exact. A dense per-phase bias array (vocab floats, built once per active combination and cached) may be simpler than several bitsets in the kernels; your call, justify it. Phase tracking for `reasoning`/`answer` reuses what Han `answer` already uses.
5. **Greedy and the fast path:** any active set disables shortcuts that skip the bias, as Han does (`sampling.cpp` around the han checks).
6. **Tests (CPU only):** profile parsing and validation, string expansion on a small fake vocabulary, request parsing (multi, overrides, unknown names, OpenAI map), CPU sampler effect per scope including Han unchanged. Keep existing tests passing. GPU tests may be built but **not run**.
7. **Docs:** the serve flag in the CLI/serving docs, and an example `wait.json` under `tools/` or `docs/`.

## Build

Use the pinned container and a build dir of your own, building only affected targets:
`python3 -I tools/docker/pinned_rocm101.py run --build-dir build/lbias --source . -- sh -c 'cmake ... && cmake --build /build/obj --target <targets>'` (see how `build/int-20261111` was configured; reuse its cache settings). clang-format may be missing; then wrap to the project style (80 columns) by hand and say so.

## Do not

- Touch the GPU, start or stop any server or container (`llm`, `gufo-exp`, production on 18740 are in use), or run model/GPU tests.
- Modify `~/.local/share/gufo/`, `tools/exl3_quant/hc/`, `plan/exl3/HC-Q8-CALIBRATION.md`, `build/b1/`.
- Push. Use `pgrep -f`/`pkill -f`.

## Commit

Commit on the current branch with single-line Conventional Commit messages ending with:
```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016Kh8svoeuCWVkepKUHx2XF
```
Stage only your own files (other work is in progress in the tree).

## Report

Design choices (data structure in kernels, string rule, field names), files changed, test results, build targets built, and anything left undone.
