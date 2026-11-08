# EXL3 conversion tooling

This directory produces and checks native EXL3 packs for gufo. Its main job is
reproducing the shipped Qwen3.8 Flash-Next **c03** pack at
`/home/m2on/projects/models/llm/flash-next-exl3-gufo-c03`. The
converter is Kyojin, an MIT-licensed ExLlamaV3 fork. It is a pinned external
build-time dependency, not vendored code (see
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)).

| Path | Contents |
| --- | --- |
| [`flash_next/`](flash_next/) | The c03 production path: job drivers, container stages, recipe, calibration inputs and builders, head recovery, ngram, MTP calibration, final assembly, and their converter-environment tests |
| [`kyojin/`](kyojin/) | The Kyojin pin (`pin.py`, `fetch_kyojin.sh`, `build_kyojin.sh`), the recorded runtime metadata (`runtime/`), and one serving-only patch |
| [`history/`](history/README.md) | Probes, diagnostics, abandoned candidates and validation scripts that no production receipt pins, kept as evidence |
| `run_convert.py`, `provenance.py`, `check_pack.py`, `check_fn_pack.py`, `c02_to_caldata.py`, `compare_qtensors.py`, `verify_hf_download.py` | Older gufo wrappers from the XQ/F3 work. None of them is on the c03 path |
| `test_*.py` | Hosted CPU tests (`exl3_quant_tools_test`), standard library only |

The `flash_next/` and `kyojin/` files came from the retired ai.tools tree
(`/home/m2on/projects/_ai.tools`, read-only evidence). The first commit that
added them, `chore(exl3): copy the c03 conversion ...`, holds the exact bytes
that `model-job-c03/FROZEN.json` hashed. A later commit moved their host paths
into [`flash_next/paths.py`](flash_next/paths.py) and their GPU owner and quiet-host
checks into [`flash_next/host_guard.py`](flash_next/host_guard.py). Apart from the
recorded calibration builder, the edited drivers no longer match FROZEN.json.
Use `git show <first commit>:<path>` for the frozen bytes.

[`flash_next/PINNED-FILES.json`](flash_next/PINNED-FILES.json) lists the 82
task files that the production receipts pin by sha256:
`model-job-c03/FROZEN.json`, the full MTP input and decoder contracts, the MTP
CPU evidence, and `ngram-full/FROZEN.json`. The drivers re-hash them before
running, so all of them stay in `flash_next/`. That includes probes the
receipts pin, such as the capture-lean and capture-static screens and the MTP
qualifiers, and the files those probes import. The hosted test checks that
every pinned file is present and unchanged, except `native/capture-gram.so`
and the drivers edited later (`model_job.py`, `mtp_calibration_job.py`,
`ngram_job.py`, `prefix_replay_job.py`, `run_stage.py` and
`test_disk_boundary.py`). Six of those pinned probe drivers still import the
ai.tools `window` module, and they are kept byte-identical.

## Pinned converter

| Item | Value |
| --- | --- |
| Source | [Yamz-Labs/kyojin](https://github.com/Yamz-Labs/kyojin) v1.8.6, commit `fbde8a3670c80727a202da3bce581da2de7c88da` |
| Archive | `https://github.com/Yamz-Labs/kyojin/archive/fbde8a3670c80727a202da3bce581da2de7c88da.tar.gz`, sha256 `dae498bd6bb2767dd06e46a94fa1dafa9b33be9362d0c8bbce0ae1c2b99f7194` (the `refs/tags/v1.8.6` archive has a different top directory and sha256) |
| Offline backup | `/home/m2on/projects/_backup/third_party/kyojin-v1.8.6.tar.gz` (same sha256), tried first by `fetch_kyojin.sh` |
| Verification | On 2026-11-08 the archive was fetched from GitHub and matched the sha256. Its 1,798 files and 15 symlinks equal every `source/` entry of `runtime/INPUTS.json`, the hashes recorded before the c03 converter build. The local ai.tools tree had only hipify outputs, `build/` and the built `exllamav3_ext*.so` beyond them, and one later edit, `tools/qwen/serve.py` and `test_serve.py` (`--max-context`, serving only). That edit is `patches/qwen-serve-max-context.patch` and is applied only with `--serve-patch` |
| Python site | ROCm 10.1.0a20260822 nightly wheels (torch 2.14.0+rocm10.1.0a20260822, triton 3.8.0, rocm-sdk 10.1.0a20260822, ...), each named with its sha256 in `runtime/WHEELS.json`. Index: `https://rocm.nightlies.amd.com/v2/gfx1151/`. All 17 were verified on 2026-11-08 and copied, 2.3 GB with a `SHA256SUMS`, to `/home/m2on/projects/_backup/third_party/kyojin-wheels-rocm10.1.0a20260822/` (from `_ai.tools/tmp/cache/llm/kyojin-wheels/`). The recorded site also held marisa_trie 1.3.1, which is in neither list |
| Build | `runtime/build.sh`: `rocm-sdk init`, `tools/strix_halo/env.sh build`, `python setup.py build_ext --inplace` with `PYTORCH_ROCM_ARCH=gfx1151 MAX_JOBS=4 EXL3_HIP_DEFINES=EXL3_HIP_STG_PAD`, CPU only, in image `ai-tools/gufo-eval:f3-native-b4ebe27c-rocm101` (`sha256:79e3e0b7bc23ed230dc90a11e342aeae109086d145bb0b302363ac2069b43126`) |
| Metadata sha256 | SOURCE.json `2623d114…`, WHEELS.json `a8d8e00c…`, INPUTS.json `142ce969…`, MODEL.json `5a5a8bc5…`, build.sh `85ff5262…` |

```sh
tools/exl3_quant/kyojin/fetch_kyojin.sh build/kyojin-runtime          # archive -> source/, verified
tools/exl3_quant/kyojin/build_kyojin.sh build/kyojin-runtime [wheels] # site/ + exllamav3_ext
```

`build_kyojin.sh` reproduces `runtime/build.sh` with the pin. It has not been
run, because no build or GPU job was allowed when it was written. The
wheel-installation step (`pip install --no-deps --no-index --target site`) is
reconstructed: ai.tools recorded the installed wheels, not the command that
installed them. The converter image and the gufo build image exist only on the
development host.

## Production c03 pipeline

All GPU stages ran in the converter image with `--network none --read-only
--user 1000:1000 --memory 100g --memory-swap 100g`, `/dev/kfd` and `/dev/dri`,
and entrypoint `/opt/venv/bin/python3`. The Kyojin runtime was bind-mounted
read-only at `/workspace/test/experiments/gufo_upstream_port_20261023/kyojin_runtime`
with `PYTHONPATH=source:site`. The BF16 source was mounted at `/source`, this
directory at `/task`, and the receipt tree at `/output`. The base environment
is in `run_stage.py`: `EXL3_ABLIT_RUNTIME=off EXL3_CKPT_KEEP_OLD=0 EXL3_RWH=0
PYTORCH_ROCM_ARCH=gfx1151 MAX_JOBS=4`, with `LD_PRELOAD` set to the site's
`libhsa-runtime64.so.1`. Host supervisors take the GPU owner lock, require a
quiet GPU and an idle NPU, and stop only their own labelled container.

| # | Stage | Files | Settings and inputs | Output | Time and memory |
| --- | --- | --- | --- | --- | --- |
| 1 | Source | `model_inventory.py` (`run_stage.py inventory`) | Qwen/Qwen3.8-Flash-Next `de4b8e4d43b917e7706784d8bb445c9af86a3540`, 131 shards, 360 GB; sha256 list in the pack's `source-provenance/` (`EXPECTED-SHA256.txt` `4346111f…`, `VERIFY-SHA256.log` `bd454ec2…`, all OK) | `source-inventory.json` (size and mtime per shard), rechecked before every module | — |
| 2 | Calibration | `build_long_calibration.py` (CPU, needs `tokenizers`/`transformers`/`safetensors` and gufo `tools/rotquant/calibration.py` on `PYTHONPATH`), `check_long_calibration.py`, `ragged_calibration.py`; GPU screens `long_row_job.py` + `long_row_pilot.py` and `bank_pilot.py` | seed 20261105; about 55 % reused c02 rows (`c02-input-ids.safetensors` `b1d6e428…`, 3906×512) plus SWE-smith, Open-SWE, Hermes-ko and Korean-wiki rows at the revisions and parquet hashes listed in `c03-input-ids.json` | `c03-input-ids.safetensors` `36c8eddb18c51c3b9788d73195aa72527d9b9d48f818b79a6f0ab251acfebf3d`, `c03-input-ids.json` `d4fc88ae…`: 2,309 rows, 1,999,496 tokens, up to 8,192 | long-row pilot 169 s, 15.8 GiB GPU peak |
| 3 | Recipe | `b1-recipe.json` (`83bf62ad…`) | 74,040 tensors: K4 73,728, K5 72, K6 204, K8 36; target 4.036 bpw; head 8 bits | — | — |
| 4 | ngram K8 | `ngram_job.py` → `run_stage.py ngram-full --seconds 3600` → `ngram_full.py`; the gate and settings came from `ngram_pilot.py`, `ngram_speed.py` and `ngram_streams.py` | `EXL3_QT_KERNEL=opt`, 32,768-row chunks, 2 streams on device 0, resumable. Ran in the c02 phase and was reused | `ngram_embedding.safetensors`, 51,840,275,568 B, readback rfn 0.00433 | 3,614 s + 3,500 s windows; host MemAvailable ≥ 62.9 GiB |
| 5 | Main body | `model_job.py` (systemd user unit `gufo-quant-model-c03.service`), one container per module boundary running `convert_main.py`; helpers `owned_cleanup.py`, `conversion_admission.py`, `disk_boundary.py`, `main_only_disk_boundary.py`, `finish_checkpoint.py`, `row_checkpoint.py`, `disk_state*.py`, `capture_{fast,progress,static_runtime,static_moe,repeat_counts,lean_gram,lt,owner}.py`, `capture_gram.hip` (native `capture-gram.so`, built by `build_capture_gram.py`), and `*-contract.json` | `--recipe b1-recipe.json --head_bits 8 --mtp_bits 4 --vision_bits 16 --ngram_bits 8 --cal_data c03 --cal_rows 2309 --cal_cols 8192 --checkpoint_interval 0 --shard_size 4096 --devices 0 --max_module N [--resume]` with `EXL3_SKIP_SIDE_MODELS=1 EXL3_QUANT_THREADS=2 EXL3_ENABLE_GROUPING=0 EXL3_HIP_F32OUT_VIA_F16=0` | converter work directory, `MAIN-CONVERSION.json` | 62 attempts, 2026-11-05 00:47 to 11-06 23:59 KST (47.2 h wall, about 33.1 h in containers); 87.97 GB peak converter RSS |
| 6 | LM head recovery | `head_recovery_job.py` → `head_recovery.py`, then the CPU-only `head_recovery_commit.py` | The K8 head module failed after quantizing ("Single row exceeds asynchronous write budget"). Recovery reused its H and K8 pack and recomputed only the first five terminal-logit rows | head committed to the checkpoint, mean rfn 0.00386 | 554.5 s |
| 7 | MTP calibrated mix | `mtp_calibration_boundary.py`, `mtp_calibration_job.py` → `mtp_calibration_candidate.py --stage input`, then `--stage decoder` (the production code, despite its name); helpers `mtp_{borrowed_rows,calibration_cache,calibration_contract,calibration_gufo,capture_identity,stage_disk_budget}.py` | `GUFO_MTP_LEASE_VERIFIED=1`. K8: fc_hidden, fc_embedding, k/v/index_qk; K6: q/o and the shared expert; K4: 1,536 experts. Semantics `gufo_mtp_f32_full_hc_shifted_v1`, 1,997,187 pairs | `DECODER-COMPLETE.json` | 0.41 h + 0.38 h; MemAvailable ≥ 74.2 GiB |
| 8 | Vision | (part of stage 9) | 333 original BF16 tensors (897,862,112 B), copied exactly by the user's decision | — | — |
| 9 | Final assembly | `prepare_final_assembly_contract.py`, `mtp_final_compile_job.py` → `mtp_final_compile_candidate.py` | combines the main, MTP and ngram work with the MTP mixed-bits recipe; `EXL3_SKIP_SIDE_MODELS=1` | the pack: `quantization_config.json` `af6f783f…` (exl3 1.8.6, 4.04 bpw, head 8, mixed MTP, BF16 vision), `config.json` `d22fdbc8…`, `GUFO-COMPILE.json` `f63aa35c…` | 159.3 s; MemAvailable ≥ 104 GiB |

The shipped directory under `/home/m2on/projects/models/llm/` matches the
assembly output byte for byte: its 17 shards are hard links and its metadata
is identical.

The checkpoint repairs that changed conversion state also belong to the
production path: `prefix_replay*.py`, `recover_checkpoint_004.py` and
`recover_disk_checkpoint.py`. The main run took 18 such repairs, each with a
`MIGRATION.json` of before and after hashes. Early modules therefore ran older
versions of `convert_main.py` and `model_job.py`. The final set is the 470
paths in `model-job-c03/FROZEN.json`. `convert_main_final_candidate.py` is
identical to `convert_main.py`, and `test_mtp_final_compile_candidate.py`
compares the two.

### Running it again

There is no single rerun command. `model_job.py` asserts gate receipts
(capture, prefix recovery and FP32 bank-pilot results) under its output root
before every module. A rerun must either copy those receipts from the
historical tree or repeat the gates. The drivers read their host paths from
`GUFO_QUANT_OUT`, `GUFO_KYOJIN_RUNTIME`, `GUFO_FN_SOURCE` and `GUFO_OPS` (see
`flash_next/paths.py`). The defaults are `build/flash-next-quant-c03`, `build/kyojin-runtime`,
the BF16 snapshot on the external drive, and `~/.local/share/gufo/ops`, the GPU
owner lock and checks of the gufo server. Several drivers still queue
follow-up messages with `/home/m2on/.local/bin/codex`. Run them as
`python3 flash_next/<driver>.py` so their directory is on `sys.path`.

The `flash_next/test_*.py` files are contract tests from the converter
environment. Most need torch, safetensors or exllamav3, and several write
`CPU.json` receipts under `GUFO_QUANT_OUT`, so they stay outside hosted CI.
The hosted `test_flash_next_inputs.py` checks the retained recipe and
calibration inputs against the hashes above, together with the path
configuration.

## Historical receipts

All receipts stay in the retired tree. The paths written in them start with
`/home/m2on/projects/ai.tools/`, which is now `/home/m2on/projects/_ai.tools/`:

| Stage | Under `_ai.tools/tmp/test-results/flash-next-quant-20261104/` |
| --- | --- |
| Source | `source-inventory.json`, `inventory/` |
| Calibration | `calibration.json`, `C03-PREPARED.json`, `READINESS-REVIEW-20261105.json`, `long-row-pilot/`, `bank-pilot-fp32/`; builder outputs in `../long-calibration-20261104/c03` |
| Recipe | `recipe-summary.json` (says vision 8; `convert_main.py` passes `--vision_bits 16`) |
| ngram | `ngram-pilot/`, `ngram-speed/`, `ngram-streams/`, `ngram-full/` |
| Main body | `model-job-c03/` (module-001…062, repair-001…018, `FROZEN.json`), `MAIN-*.json`, `capture-*/`, `prefix-recovery/`, `disk-boundary/`, `work/` |
| Head | `head-recovery/` |
| MTP | `MTP-*.json`, `mtp-calibration-*/`, `mtp-protected-idx51/` |
| Vision | `VISION-BF16-USER-DECISION.json` |
| Assembly and validation | `mtp-final-assembly*/`, `final-validation/`, `FINAL-QUALIFICATION.json` (NLL 2.1627 against reference 2.1728, top-1 0.908) |

The plans behind these settings are archived:
[F3-QUANT-BIT-ALLOCATION](../../plan/exl3/archive/F3-QUANT-BIT-ALLOCATION.md),
[F3-LONG-CALIBRATION](../../plan/exl3/archive/F3-LONG-CALIBRATION.md),
[F3-MTP-CALIBRATION](../../plan/exl3/archive/F3-MTP-CALIBRATION.md) and
[F3-CONVERSION-WATCH](../../plan/exl3/archive/F3-CONVERSION-WATCH.md).

## Known gaps

- The command and mounts that ran `build_long_calibration.py` were not
  recorded. It reads `/research`, `/old` (the c02 MANIFEST and chunks),
  `/sources` and `/source`.
- No script for `b1-recipe.json` or `c02-input-ids.safetensors` was found.
  Both are kept as data.
- The tool that wrote the source `VERIFY-SHA256.log` is not recorded.
- How the pack was moved into `models/llm` is not recorded. The shards are
  hard links of the assembly output.
- `convert_main.py` loads `/task/native/capture-gram.so`, and FROZEN.json pins
  it, but it is a build product and is not tracked (`*.so` is ignored).
  `build_capture_gram.py` compiles `capture_gram.hip` in the converter image
  to `/output/capture-gram/gram.so`. Copy that file to
  `flash_next/native/capture-gram.so` before a run, because `/task` is mounted
  read-only.
- The follow-up messages in `prefix_replay_job.py`, `mtp_calibration_boundary.py`
  and the capture jobs still name the old ai.tools receipt paths and the
  pre-archive plan path `plan/exl3/F3-CONVERSION-WATCH.md`. They are message
  text only; the files they name are now under `_ai.tools/` and
  `plan/exl3/archive/`.
