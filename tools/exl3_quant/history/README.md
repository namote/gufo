# EXL3 conversion history

These files are kept as evidence only. Nothing on the production c03 path
imports them, and they are not maintained. The production pipeline is in
[`../flash_next/`](../flash_next/) and [`../kyojin/`](../kyojin/), and the
[EXL3 tooling README](../README.md) describes it.

The files are byte-for-byte copies from the retired ai.tools tree
(`/home/m2on/projects/_ai.tools`, read-only), with their relative layout kept:

- `flash_next/` came from
  `test/experiments/flash_next_quant_20261104/`. The copy leaves out
  `__pycache__`, the built `final-quality-runtime/rq_logits` and
  `native/capture-gram.so`, and the `final-current-nix101/` gufo build
  snapshot.
- `kyojin/` came from `test/experiments/gufo_upstream_port_20261023/`.

They still import `test.experiments.*` modules and use ai.tools paths
(`/home/m2on/projects/ai.tools/...`, `/workspace/...`). Neither exists any
more, so do not expect them to run.

## What they were

| Files | Purpose |
| --- | --- |
| `flash_next/capture_*_probe.py` and `capture_*_job.py` (except the pinned ones listed below), `capture_{chunk,inputs,lt_chunk,lt_ref_cpu,owner_audit,pack,selected,subset,time_guard}.py`, `capture_forward.hip`, `capture-forward-pair-contract.json`, `integrate_capture_*.py`, `select_capture_quarter.py`, `prepare_quarter_holdout.py` | Calibration-capture speed and exactness probes that led to the `capture_fast`, `capture_lt` and `capture_static_*` code used in production |
| `flash_next/convert_main_{io,large,lt,owner}_candidate.py`, `model_job_{desktop,disk,final,io,large,lt,owner}_candidate.py`, `runner_stagebudget_candidate.py` | Superseded drafts of `convert_main.py` and `model_job.py`. `model_job_final_candidate.py` is the pre-freeze copy of `model_job.py` |
| `flash_next/mtp_calibration_cpu.py`, `mtp_gram_saved_cpu.py`, `mtp_full_decoder_audit.py` | MTP-calibration CPU checks and audits that no production receipt pins |
| `flash_next/check_*.py`, `quant_stall_probe*.py`, `hourly_watch.py`, `audit_long_sources.py` | Checkpoint, disk and stall diagnostics, the conversion watch, and the research audit of long calibration sources |
| `flash_next/final_*.py` | Post-conversion quality, smoke and throughput validation of the pack with a gufo build snapshot (results in `FINAL-QUALIFICATION.json`) |
| `flash_next/test_*.py` | Tests of the files above |
| `kyojin/kyojin_prepare.py`, `kyojin_update_build.py` | The earlier and v1.8.6 converter builds; [`../kyojin/build_kyojin.sh`](../kyojin/build_kyojin.sh) replaces them |
| `kyojin/kyojin_local*.py`, `gufo_kyojin_*.py`, `kyojin_update_{check,http}.py` | Kyojin local-serving comparisons and update checks against the Kyojin pack (plans `F3-KYOJIN-LOCAL` and `F3-KYOJIN-SAME-PACK`) |

Some probes stay in `../flash_next/` because the production receipts hash
them: `model-job-c03/FROZEN.json`, the full MTP runs' contracts and CPU
evidence, or a pinned test that reads them. The production drivers re-hash
these files before every stage. They are listed in
[`../flash_next/PINNED-FILES.json`](../flash_next/PINNED-FILES.json) and
include the capture-lean and capture-static screens, `state_io_*`, and the
MTP qualifiers and probes.

The receipts for all of this are in
`/home/m2on/projects/_ai.tools/tmp/test-results/flash-next-quant-20261104/`.
