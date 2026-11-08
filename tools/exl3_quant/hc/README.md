# HC Q8_0 calibration

Offline conversion tools for Flash-Next HC down/up weights. They supplement
the EXL3 pack with calibrated **Q8_0**, not another EXL3 tensor format. The
source is the original BF16 snapshot; the existing F16 pack supplies a separate
legacy rounding control. No command loads a model on the GPU or changes a pack.

| File | Role |
| --- | --- |
| `inventory.py`, `weights.py` | Header inventory and direct BF16-bit expansion to FP32; no intermediate F16 |
| `quant.py` | Full-H GPTQ-style sequential error feedback with fixed stored F16 group scales |
| `refine.py` | Reuses the saved GPTQ/control files and H; compares code-only and code-plus-scale coordinate refinement |
| `test_cpu.py`, `test_refine.py`, `check.py` | Optional NumPy/safetensors CPU checks, independent brute-force numerical oracles |
| `ORIGIN.json` | Original source hashes and the preserved migration evidence |

Status: CPU reference tools. **No real HC calibration run or runtime loader
qualification yet.** The original B5 capture prototype is historical; current
B10 input capture needs a separate port and review. See the active
[HC plan](../../../plan/exl3/HC-Q8-CALIBRATION.md).

## Commands

From the gufo repository root, with NumPy and safetensors installed:

```sh
python3 -m tools.exl3_quant.hc --help
python3 -m tools.exl3_quant.hc inventory --model /path/to/BF16 --dtype BF16 --output build/exl3-hc/inventory.json
python3 -m tools.exl3_quant.hc quantize --source-model /path/to/BF16 --target-model /path/to/pack \
  --tensor exact.hc.input_mix_weight_down.weight --stats /path/to/stats.npz \
  --receipt /path/to/receipt.json --output build/exl3-hc/down-gptq
python3 -m tools.exl3_quant.hc refine --source-model /path/to/BF16 --target-model /path/to/pack \
  --baseline build/exl3-hc/down-gptq --stats /path/to/stats.npz \
  --output build/exl3-hc/down-refined --sweeps 6
```

The target on this host is `~/projects/models/llm/flash-next-exl3-gufo-c03`.
Both model paths are explicit; there is no retired-project path or automatic
substitute. Source models, completed controls and output directories are not
overwritten. New outputs belong under `build/exl3-hc/`.

Run the optional numerical tests in the existing shared pinned image:

```sh
python3 -m tools.exl3_quant.hc.check --output build/exl3-hc/cpu-001
```

The command uses `tools/docker/pinned_rocm101.py`, mounts source read-only,
exposes no GPU devices, disables network, and writes `cpu.log` in a new run
directory. It rejects an existing output directory and removes only its own
uniquely named container if the 120-second CPU check times out. It does not
install packages. The top-level hosted EXL3 suite stays
standard-library-only; `hc/` intentionally has no `__init__.py`, so unittest
discovery does not recurse into these optional numerical tests. Direct local
tests: `TMPDIR=/tmp python3 -B -m unittest tools.exl3_quant.hc.test_cpu tools.exl3_quant.hc.test_refine`.

## Numerical and file contract

- Statistics: `H = X.T @ X / calibration_rows`; full input-channel correlations,
  not diagonal/group-only H. Down inputs are after HC norm; up inputs are after
  scaled SiLU. Norm is not folded into weights. Keep phase, upstream down-weight
  state and runtime/source identities. A changed down path can change up inputs.
- `stats.npz` contains `H` and `X_holdout`. The receipt schema is
  `gufo-hc-input-gram-v2`: tensor, BF16 source and F16 target identities
  (`dtype`, `model_index_sha256`, raw `weight_sha256`, `shape`), `stats_sha256`,
  `input_space=unfolded_projection_input`, `H_normalization=sum_xxt_div_rows`,
  `synthetic=false`, calibration/holdout row counts and disjoint document IDs,
  `capture_source_sha256`, `capture_runtime_sha256`, `upstream_weight_state_sha256`.
  Content overlap must also be audited by the collector; the generator cannot
  establish provenance or semantic independence from IDs alone.
- GPTQ fixes actual F16 stored scales from BF16 RTN and uses full-H feedback
  with 1% mean-diagonal damping by default. No act-order permutation. Invalid
  data, unsupported scales or factorization failure stop explicitly.
- Refinement starts both branches from that same saved GPTQ candidate. Up to
  six sweeps; codes move ±1 only when the regularized objective decreases.
  The scale branch also chooses the fixed-code least-squares optimum on the
  positive F16 grid. Both use `H + existing damping I`; raw-H and held-out
  errors are reported separately. Holdout never drives the updates.
- Payload: U8 `[out,in/32,34]`, little-endian F16 scale + 32 int8 codes,
  codes [-127,127]. A zero group may retain zero scale. Code-only refinement
  retains scale bytes; joint refinement may change them. No new inference
  format or kernel is implied.
- Five-way comparison: legacy F16 RTN, BF16 RTN, GPTQ, GPTQ+codes,
  GPTQ+codes+scales. Reuse existing packed controls and H; no baseline rerun.
  Results include hashes, weighted SSE, held-out SSE/p99/max and refinement
  time. Outputs remain `NOT_RUNTIME_QUALIFIED`; there is no automatic selection.

The full-H CPU implementation is a correctness reference, not a demonstrated
fast full-pack converter. A 10240-column H alone needs 800 MiB in FP64, with
additional factorization/work buffers. Screen a bounded real site first.
Projection error is not connected-HC/model quality, and this weight correction
does not remove activation-Q8 error. Final deployment requires native packing,
connected HC, decode quality and MTP acceptance/whole-model throughput checks.
