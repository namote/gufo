# Flash-Next full-vocabulary quality gate

Implementation handoff, 2026-10-31; **not a numerical qualification result**.
The collector uses the production Flash-Next trunk and full 248,320-row head
for GGUF or EXL3 sidecars, without MTP. FP16 KV is the default; `--kvarn`
selects the existing k4v4 API for a separate experiment. Every scored forward
replays its HC mixer and head at the same row width and requires bitwise equality.
The replay adds diagnostic cost: these runs are not speed benchmarks.

The record schema and scorer are the existing 27B R0 implementation. Normalizers,
NLL and argmax use the **entire vocabulary**; `KL_256` coarsens the reference's
256 largest probabilities plus a rest bucket, so it is a lower bound on full KL.
Report its coverage alongside paired delta NLL, top-1, flips, KL tails and NLL
loss tails, including categories and token/block/window bootstrap intervals.
There is no invented absolute pass threshold. Compare EXL3 against UD-Q4_K_XL
on identical positions and inspect tails before claiming better quality.

## Reference runtime and memory

The pinned local `config.json` says `model_type=qwen4_exp`, text type
`qwen4_exp_text`, architecture `Qwen4ExpForConditionalGeneration`, and records
`transformers_version=5.8.0.dev0`. That last field is export provenance, not a
sufficient runtime requirement. Native **Transformers 5.17.0** provides
[`Qwen4ExpForCausalLM`](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)
and handles the real checkpoint's full-attention/QSA config and PLE shard layout
([config source](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py)).
This tool constructs its text model directly, excluding vision and MTP.
`trust_remote_code` is unnecessary and does not fix an unsupported installation;
the local snapshot has neither an `auto_map` entry nor custom modeling Python.

**The installed packages in the two images have not been inspected:** this worker
was forbidden to use Docker. The checked-in EXL3 image recipe at
`../ai.tools/test/experiments/gufo_upstream_port_20261023/exl3/Dockerfile` does not
explicitly install or pin Transformers/Accelerate. The gufo recipe inherits an
external base and also does not pin them. Neither recipe proves installed native
architecture support. Run `--check-environment` with the selected image's Python;
it prints versions and actually imports both required HF classes, failing if
support or dependencies are missing. Use a separate oracle environment with
`transformers==5.17.0`, CPU-capable Torch (the EXL3 image has Torch 2.13), Accelerate
and NumPy; retain its package versions. Do not upgrade a measured gufo toolchain.
For an existing CPU Torch environment, the missing Python packages can be added
with `python3 -m pip install 'transformers==5.17.0' accelerate numpy`.

`bf16_reference.py` uses Accelerate meta initialization, not `from_pretrained`'s
whole-checkpoint loader. A decoder layer's tensors are mmap views only while its
HF forward runs; unmapping bounds process RSS and does not drop the kernel's file
cache. Unselected expert pages need not be read. A custom embedding lookup gathers
exact BF16 rows from the 128 disk shards, preserving HF's hashing, EOS resets,
PLE/GDN states and QSA. No giant concatenated embedding is created. Persistent
PLE integer buffers come from the checkpoint. Normal floating-point weights
remain BF16, with HF eager CPU arithmetic and native cache dtypes recorded in
metadata. **HF attention KV is normally BF16, whereas gufo KV is FP16**; this is
an engine-arithmetic difference to report, not a matched KV quantization study.
HF BF16 head outputs are upcast to FP32 for record construction; that upcast
retains HF's BF16 rounding and does not turn the reference into an FP32 model.

Header-only inspection of all 131 shards verified: trunk 251,487,307,800 bytes,
PLE 102,400,491,520 bytes, vision/MTP/other 6,112,163,808 bytes; largest tensor
3,355,443,200 bytes. Index SHA256:
`99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de`.
The Flash-Next `tokenizer.json` SHA256 also matches R0 (`0997f410…b9f3`);
the oracle checks this identity before collection.
One layer's weights plus 4K eager attention/GDN activations are expected to stay
well below 100 GB RSS (estimate, not measured). The JSON receipt records peak RSS;
retain `/usr/bin/time -v` too. This does not verify full weight payload integrity;
the downloaded snapshot has separate `EXPECTED-SHA256.txt`/`VERIFY-SHA256.log`.

**Runtime estimate, unmeasured:** prefill reads about 2 TB of trunk weights across
eight windows, around 1–3 hours of sequential I/O alone at 0.2–0.5 GB/s. Decode
can touch roughly 10–15 GB per step across 4,096 scored steps, 40–60 TB, or
roughly 1–4 days of ideal I/O. mmap readahead, random reads, CPU BF16 GEMMs,
Python expert loops and GDN fallback can extend this to several days. Start with
the tiny numerical test and one-window/8-position smoke, then estimate from its
per-window receipt before scheduling the complete CPU run. No timing was measured.

## Exact run commands

Run from the repository root. Paths containing spaces stay quoted. Existing a01
windows are preferred; rebuild only if unavailable, using its **original** source
revision and tokenizer, then require the manifest's hashes. Do not rebuild from
current HEAD or tokenize the text differently.

```sh
HF='/run/media/m2on/T5 EVO/AI/model/originals-flash-next-hf-20261030/Qwen-Qwen3.8-Flash-Next'
OUT="$PWD/build/fn-quality"
A01="$HOME/projects/ai.tools/storage/evaluations/rq-r0-20261029/corpus-a01/windows.json"
mkdir -p "$OUT"

# Header preflight needs only Python's standard library.
python3 tools/qwen38_flash_next/rq_quality/bf16_reference.py "$HF" --inspect-checkpoint
# Run in the provisioned oracle environment, with no GPU access required.
python3 tools/qwen38_flash_next/rq_quality/bf16_reference.py --check-environment
python3 -m pip freeze > "$OUT/oracle-pip-freeze.txt"
python3 -I tests/models/qwen38_flash_next/rq_quality_test.py
python3 -I tests/models/qwen27b/rq_quality_test.py

# Only if the original a01 files are unavailable:
# python3 tools/qwen27b/rq_quality/corpus.py build --repo . \
#   --revision f95d1302080c5f893dfd1293312b1245fd362c60 \
#   --tokenizer "$HF/tokenizer.json" --out "$OUT/corpus-a01"
# A01="$OUT/corpus-a01/windows.json"
python3 - "$A01" <<'PY'
import hashlib, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
manifest = json.loads(pathlib.Path('tools/qwen27b/rq_quality/corpus_manifest.json').read_text())
for name in ('windows.json', 'windows.bin'):
    assert hashlib.sha256((path.parent / name).read_bytes()).hexdigest() == manifest['outputs'][name], name
print('a01 token and index hashes match R0')
PY

# Optional smoke; keep its outputs separate from complete references.
python3 tools/qwen38_flash_next/rq_quality/bf16_reference.py "$HF" \
  --windows "$A01" --path prefill --windows-limit 1 --positions-limit 8 \
  --out "$OUT/smoke-bf16" --threads 16

# Complete BF16 reference: 8 * 4095 = 32760 prefill rows;
# decode positions 2048..2559 inclusive in every window = 4096 rows.
for path in prefill decode; do
  /usr/bin/time -v python3 tools/qwen38_flash_next/rq_quality/bf16_reference.py "$HF" \
    --windows "$A01" --path "$path" --prefix 2048 --top-k 256 \
    --threads 16 --row-chunk 16 --out "$OUT/bf16-$path" \
    --toolchain 'HF transformers 5.17.0; see oracle-pip-freeze.txt' \
    > "$OUT/bf16-$path.log" 2> "$OUT/bf16-$path.time"
done

# Build on the qualified HIP machine, not the editing host.
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target qwen38_flash_next_rq_logits
BIN="$PWD/build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_rq_logits"
# If the configured preset uses another binary directory, use its same target output.
# Set these to the first UD-Q4_K_XL shard and the generated EXL3 sidecar:
UD='/absolute/path/to/UD-Q4_K_XL-first-shard.gguf'
EXL3='/absolute/path/to/exl3/gufo.gguf'
for path in prefill decode; do
  for variant in ud exl3; do
    model="$UD"
    if [ "$variant" = exl3 ]; then model="$EXL3"; fi
    "$BIN" "$model" --windows "$A01" --path "$path" --batch 512 \
      --prefix 2048 --top-k 256 --reference "$OUT/bf16-$path" \
      --out "$OUT/$variant-$path" --label "fn-$variant-fp16kv" \
      --toolchain 'qualified gpu-test build; retain compiler and binary identity'
  done
  python3 tools/qwen27b/rq_quality/metrics.py score \
    --reference "$OUT/bf16-$path" --bootstrap 2000 --block-tokens 256 \
    --out "$OUT/score-$path.json" "$OUT/ud-$path" "$OUT/exl3-$path"
  # First minus second: EXL3 improves if delta-NLL/KL are negative, top-1 positive.
  python3 tools/qwen27b/rq_quality/metrics.py compare \
    --reference "$OUT/bf16-$path" --bootstrap 2000 --block-tokens 256 \
    --out "$OUT/compare-$path.json" "$OUT/exl3-$path" "$OUT/ud-$path"
  python3 tools/qwen27b/rq_quality/metrics.py flips \
    --reference "$OUT/bf16-$path" --out "$OUT/flips-$path.json" \
    "ud=$OUT/ud-$path" "exl3=$OUT/exl3-$path"
done
```

The candidate `--reference` is mandatory for scoring against BF16: collecting
without it writes self top-k logits, which cannot be retroactively paired to
another reference. Every run produces a packed `.bin` and a hashed `.json`
receipt. An interrupted collection has no success receipt. A smoke with fewer
rows is not a full-window gate. Inspect the expected record counts, all head
checks, finite metrics, coverage, category losses and worst positions before
accepting a run. Token CIs alone overstate independence; retain block/window CIs.

## Validation status and commit split

Passed on the editing host: all real checkpoint headers/ranges; the three
original a01 file hashes and eight 4096-token windows; three synthetic
standard-library loader tests; Python syntax checks; `git diff --check`.
The tiny resident-vs-streaming HF test explicitly **skipped** because Torch,
Transformers, Accelerate and NumPy are absent. No model forward, C++ build,
GPU/Docker operation or quality/speed benchmark ran. The shared format command
cannot run here because Nix/clang-format are absent. Run those checks on the
qualified machine before committing. Leave changes uncommitted for integration.

Suggested commits:

1. `test(qwen38-flash-next): add full-vocabulary quality collectors`
   — collector/CMake, diagnostic head replay API, BF16 oracle and loader tests.
2. `test(quality): report target-token loss tails in shared RQ metrics`
   — shared scorer and its two synthetic tail tests.
3. `docs(exl3): document the Flash-Next BF16 quality protocol`
   — this handoff and exact commands.
