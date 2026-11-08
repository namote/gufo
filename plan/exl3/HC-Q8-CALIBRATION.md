# HC Q8_0 calibration and refinement

2026-11-08. User approved BF16-source calibration files, then a comparison of
code-only and code-plus-scale refinement. GPU collection waits until the
optimization benchmark handoff; no automatic service changes during migration.

## Current state

- Implementation: [`tools/exl3_quant/hc/`](../../tools/exl3_quant/hc/README.md).
  CPU-only generator and five-way comparison. No model weights generated yet.
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

## Next execution steps

1. Finish migration CPU checks and second-reviewer review. Preserve arithmetic
   and reusable evidence; use Gufo's shared pinned environment and host guard.
2. Port bounded activation export to a separate frozen current-source build.
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
20 checks, including namespace CLI help without optional packages and the
CPU runner's new-directory/owned-container-timeout cleanup contract. No GPU
was used. These checks do not establish real HC quality or throughput.
