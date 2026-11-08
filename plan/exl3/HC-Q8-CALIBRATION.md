# HC Q8_0 calibration and refinement

2026-11-08. User approved BF16-source calibration files, then a comparison of
code-only and code-plus-scale refinement. GPU collection followed the completed
B10 optimization benchmark handoff. Only the owned server was temporarily stopped
and restored with exactly the same command/settings.

## Current state

- Implementation: [`tools/exl3_quant/hc/`](../../tools/exl3_quant/hc/README.md).
  BF16-source generator, five-way comparison and bounded B10 capture.
  First-attention HC pilot artifacts now exist; full200-tensor pack remains pending.
- Original BF16 snapshot:
  `/run/media/m2on/T5 EVO/AI/model/originals-flash-next-hf-20261030/Qwen-Qwen3.8-Flash-Next`.
- Target: `/home/m2on/projects/models/llm/flash-next-exl3-gufo-c03`.
- Header inventory: 100 HC sites / 200 down/up matrices; main 97 sites,
  MTP 3. Q8_0 payload about 664 MiB per complete variant. Norm/inject, body
  EXL3, MTP EXL3, ngram and BF16 vision are unchanged by this task.
- Calibration IDs: `tools/exl3_quant/flash_next/c03-input-ids.{json,safetensors}`,
  2309 rows / 1999496 tokens. The earlier 25% reduction remains rejected.
- Separate diagnostic holdout was copied byte-exact to
  `/home/m2on/projects/models/eval/flash-next/hc-holdout/` (DATA/TEXTS and input IDs).
  Its 8 documents include synthetic Korean and local C++; it is not a public
  benchmark. No semantic decontamination or general quality claim.
- Original code, preparation receipts and CPU logs are preserved with hashes
  in `/home/m2on/projects/_backup/gufo-hc-q8-20261108/`. They are evidence, not
  runtime dependencies. The retired project is never imported or executed.
- Previous frozen capture prototype was built against B5 `5912ed64`, first
  attention HC only, with no GPU validation. Do not use it as the B10 collector.
  Its patch/build recipe and identities are retained in the backup; its object
  cache is not a required input and is not migrated into the active tool.
- Current service is B10 `9eb3cbdb`, installed supervisor `d01fd32a`; refer to
  [HANDOFF](HANDOFF.md) and inspect current status again before a GPU window.
- Historical B10 collector001 is CPU-built at `build/exl3-hc/capture-b10-001`.
  ELF SHA `53572ac57b10bbcd7db49eff5996479aeec77e55fa12a568d9cf9b5db659a6f0`;
  SOURCE/BUILD/CACHE-PROVENANCE receipts retain exact identities. Three
  instrumented translation units rebuilt; remaining B10 objects reused.
  This001 build was not used for GPU capture. Collector002 below supersedes it.
- Fixed pilot inputs: `/home/m2on/projects/models/eval/flash-next/hc-pilot-001`,
  8 calibration + 8 separate holdout windows, 5942 tokens, max513 per document.
  INPUTS.json retains full original row hashes, selected byte hashes and the
  sampled64/stride32 lexical-overlap screen. This is not a 25% replacement.

## Next execution steps

1. Migration CPU checks and second-reviewer review completed. Preserve arithmetic
   and reusable evidence; use Gufo's shared pinned environment and host guard.
2. Bounded activation export port and separate frozen B10 CPU build completed.
   Down: after unfolded HC norm; up: after actual scaled SiLU. Do not mix
   source/ELF identities or assume old B5 statistics describe current execution.
   Initial layer0 pilot is only a screen, not a full calibration substitute.
3. Confirm benchmark completion and fresh GPU ownership; stop only the owned
   server for the bounded window, then restore its exact recorded configuration.
4. Capture once, verify finite/counts/IDs and the capture-on/off tiny control.
   Establish native F16 RTN packing parity separately from the BF16 reference.
5. Generate GPTQ and reuse its H/control in code-only and code+scale comparison.
   Report raw and regularized H error, separate-document output error, time
   and memory; do not claim whole-model improvement from projection results.
6. Extend only qualified collection paths to main/mixer/MTP; no fabricated or
   same-shape-substituted statistics. MTP needs actual draft inputs. Recheck
   connected down→SiLU→up after changing down weights.
7. Final loader integration is separate. Batch decode KLD, native packing,
   connected HC, prefill and MTP tokens/s/acceptance checks. Reuse only matching
   saved baselines. No automatic pack overwrite or service deployment.

## Comparison contract

Original BF16 → exact FP32 expansion; CPU feedback/refinement uses FP64.
Same statistics and saved controls for all five branches:
legacy F16 RTN / BF16 RTN / fixed-scale GPTQ / GPTQ+codes / GPTQ+codes+scales.
The last two use the same six-sweep budget. Objective: `H + damping I`;
the scale candidate snaps the fixed-code optimum to actual F16 storage.
Holdout is evaluation-only. Damping/sweep count are declared candidates, not
established optimal choices. Improved calibration loss does not guarantee
held-out loss, connected-HC quality, or MTP speed; never auto-select a winner.

CPU validation before migration: 13 numerical checks and 432 old infrastructure
checks passed (receipts in the backup). After migration: 13 numerical checks
passed in `build/exl3-hc/cpu-migration-002/cpu.log`; the hosted EXL3 suite passed
22 checks, including namespace CLI help without optional packages and the
CPU runner's new-directory/owned-container-timeout cleanup contract. No GPU
was used. The shared format check passed 801 C++ files using the retained
clang-tools21.1.8 (initial attempt lacked clang-format on PATH; both logs retained).
These checks do not establish real HC quality or throughput.

## B10 capture completed, November 8, 2026

- Collector002: `build/exl3-hc/capture-b10-002`, source `9eb3cbdb`, ELF
  `55d1eaeee55db02f0ecac7124fa5abfc639581442fe0e5a42b23a47d231fd70d`.
  Adds a streaming SHA over full-vocabulary logits alongside the top8 record SHA.
- `build/exl3-hc/capture-run-001`: six bounded cases completed. Both prefill32
  and decode8 capture-off/on controls match full-vocabulary and record bytes.
  Main prefill:5926 rows (4092 calibration/1834 held out). Decode:1897 rows
  (1024 calibration/873 held out), excluding each window's32-row prefix from H.
- Independent read-only review checked every input payload's finite values,
  size, consecutive row positions, phase and down/up coverage. Kernel error
  delta empty, owned containers cleaned, service ready and full before/after
  configuration identical. See `INDEPENDENT-REVIEW.json` and `STATE.json`.
- Target down/up raw payload hashes are frozen before collection and compared
  again at statistics creation; model-index identity alone is insufficient.
- CPU generation: `build/exl3-hc/pilot-weights-001`. Decode H only; saved prefill
  is for cross-phase held-out evaluation, not a second independently authored
  dataset. All branches reuse H and completed packed controls.
- Down H has1024 observations for10240 channels: intentionally underdetermined
  pilot, stabilized by the declared damping. These data do not replace the
  full2309-row/1999496-token calibration or qualify a full-model release.
- Ideal connected gate evaluation uses `up(SiLU(down(x)/4))` in FP64. Review
  caught an initially misplaced division after SiLU before any real connected
  evaluation was run; the formula and independent ±1 oracle were corrected.
  This screen excludes activation Q8, native rounding and final HC mixing.

## Pilot result

Generation completed, and existing H/control/packed files were reused for all
comparisons. Detailed report: `build/exl3-hc/pilot-weights-001/REPORT.md`, individual
document results in `SCORE.json`. Relative to legacy RTN, decode held-out SSE:
GPTQ down0.528719/up0.182783; code+scale refinement down0.541117/up0.121988.
Ideal connected gate SSE ratios: GPTQ decode0.191044/prefill0.198557;
code+scale decode0.131878/prefill0.137968. No automatic selection/deployment.
Down refinement slightly worsens held-out error despite lower training error.
The full HC pack still requires other-site collection, full calibration,
native arithmetic checks and loader integration; this is a useful first-site
screen, not a whole-model quality/latency result.
