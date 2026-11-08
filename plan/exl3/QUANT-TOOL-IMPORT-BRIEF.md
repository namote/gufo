# Brief: bring the EXL3 conversion tooling into gufo (2026-11-08)

User decision: keep Kyojin as a **pinned external source** (not vendored) and
move **our own** conversion scripts, recipes and build steps into gufo, so the
c03 Flash-Next pack can be reproduced from this repository.

Sources to read:
- Converter runtime: `/home/m2on/projects/_ai.tools/test/experiments/gufo_upstream_port_20261023/kyojin_runtime/`
  (`SOURCE.json` pin: Yamz-Labs/kyojin `v1.8.6`
  `fbde8a3670c80727a202da3bce581da2de7c88da`, archive sha256
  `dae498bd…7194`; `build.sh`, `WHEELS.json`, `INPUTS.json`, `MODEL.json`;
  the `source/` tree is the unchanged upstream archive — confirm by checking
  the archive sha / diffing against a fresh fetch if network allows; if it is
  NOT unchanged, list the local patches and carry them as patch files).
- Our c03 job code: `/home/m2on/projects/_ai.tools/test/experiments/flash_next_quant_20261104/`
  (`model_job.py`, `ngram_job.py`, `run_stage.py`, recipes such as
  `b1-recipe.json`, calibration input builders `c0x-input-ids.*`, the many
  `capture_*`/probe files). Status/receipts:
  `/home/m2on/projects/_ai.tools/tmp/test-results/flash-next-quant-20261104/`.
  The final c03 pack metadata: `/home/m2on/projects/models/llm/flash-next-exl3-gufo-c03/`
  (`quantization_config.json`, provenance files).
- Existing gufo wrapper: `tools/exl3_quant/` (run_convert.py, provenance.py,
  check_pack.py, check_fn_pack.py, c02_to_caldata.py, …) and
  `plan/exl3/archive/F3-QUANT-BIT-ALLOCATION.md`, `F3-MTP-CALIBRATION.md`,
  `F3-CONVERSION-WATCH.md` (or their current paths) for what actually produced
  c03 and which settings (EXL3_HIP_F32OUT_VIA_F16=0, EXL3_QUANT_THREADS=2,
  grouping off, ngram opt/32768/2 streams, MTP calibration mix, vision BF16).

## Tasks

1. **Trace the production path.** From the receipts and job logs, determine
   the exact files, commands, environment and inputs that produced the
   shipped c03 pack (main body, lm head/terminal head recovery, ngram K8, MTP
   calibrated mix, vision). Separate them from probes/diagnostics/abandoned
   candidates. Write the result as `tools/exl3_quant/README.md` (pipeline,
   stages, inputs with hashes, outputs, time/memory per stage, known caveats).
2. **Pin Kyojin.** Add a pinned fetch of the upstream archive (URL, rev,
   sha256), preferably as a flake input or a Nix derivation for the converter
   runtime (Python env + the `exllamav3_ext` native build for gfx1151 on the
   pinned ROCm 10.1), following the existing flake structure; if a Nix build is
   too large for this task, provide `tools/exl3_quant/fetch_kyojin.sh` +
   `build_kyojin.sh` that reproduce `kyojin_runtime/build.sh` with the pin and
   verify the sha. Keep upstream's MIT LICENSE notice in whatever is fetched;
   add a `THIRD_PARTY.md`/notice entry in gufo naming Kyojin/ExLlamaV3 (MIT,
   Turboderp/Yamz-Labs) as an external build-time dependency.
3. **Move our scripts.** Copy into `tools/exl3_quant/` (subfolders as
   appropriate, e.g. `flash_next/`) only what the production path uses:
   job drivers, recipes (`b1-recipe.json` → a named c03 recipe), calibration
   builders, pack assembly/check, ngram conversion, MTP calibration, head
   recovery. Rewrite hard-coded `ai.tools` paths into arguments/config with the
   old paths as documented defaults only where needed; keep standard-library
   or the converter env as the only dependencies. Do not copy model weights,
   calibration token files larger than a few MB (reference them by path +
   sha256 instead), build trees or receipts.
4. **Tests.** Add CPU-only tests for the moved pure-Python parts (recipe
   generation/validation, provenance, calibration packing) following the
   existing `tools/exl3_quant/test_*.py` style, wired into whatever runs the
   repo's Python checks (`tools/ci/`).
5. **Do not** run conversions, GPU jobs, or modify ai.tools. Leave the
   ai.tools originals in place (they are evidence); mention in the README
   where the historical receipts live.

Rules: main checkout `/home/m2on/projects/gufo`, branch `ai-tools/port-u4f`,
no worktrees, no push; second-reviewer subagent review of the whole change,
fix, re-review; Conventional single-line commits with the session attribution
lines; repo Python checks and format checks must pass.

## Amendment (user, 2026-11-08): copy whole folders; ai.tools deprecated

- **Copy our folders whole, not a cherry-picked subset.** Copy the complete
  code of `ai.tools/test/experiments/flash_next_quant_20261104/` into
  `tools/exl3_quant/flash_next/` (all scripts, HIP probes, contracts, recipes,
  JSON/MD notes), excluding only: `__pycache__`, built binaries
  (`final-quality-runtime/rq_logits`, `*.so`, ELF files), and the
  `final-current-nix101/` runtime/source snapshot (that is a gufo build copy).
  Include `b1-recipe.json` and the `c02/c03-input-ids.safetensors` (≈8 MB each)
  so the pack stays reproducible after ai.tools goes away. Also copy the
  Kyojin-related scripts from `ai.tools/test/experiments/gufo_upstream_port_20261023/`
  (`kyojin_*.py`, `gufo_kyojin_*.py`, and the `kyojin_runtime/` metadata:
  `SOURCE.json`, `WHEELS.json`, `INPUTS.json`, `MODEL.json`, `build.sh`) into
  `tools/exl3_quant/kyojin/`. README task 1 still marks which files form the
  production c03 path and which are probes/diagnostics kept for history.
- **The local Kyojin folder will be deleted by the user.** Therefore the pin
  must work without it: fetch the v1.8.6 archive from GitHub now, verify its
  sha256 equals `SOURCE.json`'s, diff it against the local `source/` (minus
  build artifacts) to prove "unchanged" (or extract patches), and save the
  archive outside git at `~/.local/share/gufo/third_party/kyojin-v1.8.6.tar.gz`
  as an offline backup referenced by the fetch script.
- **Rewrite hard-coded ai.tools paths** in the copied code to repo-relative
  paths or arguments where the copied code is on the production path; for the
  rest, leave them but note in the README that they reference the deprecated
  ai.tools layout.
- **ai.tools deprecation notice** (the user asked for this; it is the one
  allowed ai.tools edit): add a short DEPRECATED banner at the top of
  `ai.tools/AGENTS.md` and its README (create `README.md` if absent) saying
  the gufo tooling moved to `~/projects/gufo/tools/exl3_quant/` and new work
  happens in gufo. Also list what still depends on ai.tools and must not be
  removed yet: the live server launcher `~/.local/share/gufo/gufo-server.py`
  imports `test.experiments.gufo_upstream_port_20261023.window` and
  `test.experiments.vllm_operator_cost_20261019.{journal,fence}`, uses image
  `ai-tools/gufo-eval:native-exl3-vision-rocm101`, and serves the model at
  `ai.tools/storage/models/llm/flash-next-exl3-gufo-c03`. Commit that in the
  ai.tools repo as a single docs commit with the session attribution lines
  (do not touch its other 221 uncommitted changes; stage only your files).

## Amendment 2 (2026-11-08): ai.tools retired

ai.tools was renamed to `/home/m2on/projects/_ai.tools` (read-only source for
this copy; never import or execute its code) and its models moved to
`/home/m2on/projects/models/`. The deprecation notice and server-ops migration
were done separately; skip those parts of the first amendment. Rewrite copied
default paths to `/home/m2on/projects/models/...` where they point at models.

## Amendment 3 (2026-11-08): the pinned build/run environment too

`build/b1/build.sh`, `build/b1/driver.py` and `build/b1/gpu/gpu.sh` (local,
git-ignored helpers used for every B1–B7 build and GPU pass) mount
`_ai.tools/test/experiments/flash_next_quant_20261104` at `/task` and import
`final_current_build.environment()` and `run_stage.py` docker flags from it.
Bring that logic into gufo as tracked tooling (e.g.
`tools/docker/pinned_rocm101.py` + `build.sh`/`gpu-run.sh` wrappers: image
digest `sha256:e7a761a4…`, `gufo-nix` volume, GPU device/group flags, runtime
env/LD_LIBRARY_PATH), document it in the tools README, and repoint
`build/b1/*` (local) at it, with model/corpus/reference paths under
`/home/m2on/projects/models/` or arguments. Verify by running
`build/b1/build.sh gufo` (CPU build only, must be a no-op/incremental success).
