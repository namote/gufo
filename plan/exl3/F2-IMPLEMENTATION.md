# F2 implementation handoff (2026-10-31)

Stages A–C and the bounded Stage D reference route are implemented in the
`ai-tools/fn-exl3` worktree. The supervisor subsequently built the implementation
cleanly (zero errors) in `ai-tools/gufo-eval:64849ca6ec7b`, at
`build/fn-gpu-test`; see the follow-up below. This editing environment has
Python but no Nix, CMake, C++ compiler, hipcc or clang-format. No GPU, Docker,
push or commit was attempted by the implementation worker. Existing model
benchmark/quality claims are unchanged. The requested Nix configure and format
commands here both returned `nix: command not found`.

## Representation and entry points

- `exl3_weights.*` binds the actual K5 expert/indexer, K6 head/FC and K7
  trunk/shared inventory. BF16 embedding and F16 HC/router weights stay in
  their source formats. Norm, convolution, A/dt and GDN V-head transformations
  are model-local. The combined indexer is restored before taking q/k views.
- Core checkpoint descriptors expose retained upload handles/offsets and
  supplementary safetensors files. Missing/partial shards, missing indexed
  tensors, duplicate supplemental names and malformed codebook markers fail.
- Upload allocates one aligned slab per projection/expert bank, including its
  device descriptors, and streams source ranges through `WeightUpload`.
  GDN block permutations happen in the upload destination. It keeps no second
  complete host copy. Resident accounting includes padding and descriptors.
- The loader recognizes `gufo.exl3.checkpoint`; CLI/backend names are unchanged.
  The sidecar supplies relative IQ4_NL PLE and MTP mixer patch references.
  The IQ4_NL table continues through `NgramTable`; NG6 is not decoded/uploaded.
- Header/size fingerprints bind the EXL3 shards, mixer patch and retained GGUF
  source. Config/tokenizer JSON are separately hashed. These are **header
  identities, not full payload integrity hashes**. Vocabulary/merges, source
  special IDs/template and PLE hash constants are checked before upload.
  The tokenizer's 243 padded vocabulary rows are retained from the GGUF.

Generate a tensorless metadata view after the download completes (the same
file can serve the existing target and MTP path options):

```sh
python3 tools/exl3_quant/fn_metadata.py EXL3_DIR IQ4_NL_TRUNK.gguf gufo.gguf \
  --mtp-mixer EXL3_DIR/mtp_hyper_connection_mixer_patch.safetensors
```

This tool imports the existing NumPy-based GGUF metadata helpers. Generation
reads safetensors headers and GGUF headers, never hashes full weight files.
It refuses missing/short payloads. No sidecar was generated in the pack root.

## Execution

- Shared X2b adds a separate K7 extraction branch; K1–K6 arithmetic is unchanged
  in source. Dense q/k/v and GDN qkv/z use the existing grouped launch; shared
  expert decode uses the existing fused EXL3 MLP.
- Routed decode/verify builds a bounded expert tile list on device, retaining
  assignment/slot IDs. Each expert's actual 1–8 rows reuse its decoded weights;
  gate/up rotations are independent. Down outputs remain separate until the
  existing ordered MoE epilogue. There is no decode count download or floating
  atomic into the residual. C2/C4/C8 concatenated verification uses this path.
- Prefill compacts assignments with the existing scheduler, downloads counts,
  and runs one expert bucket at a time through shared dequant + FP32 GEMM.
  Scratch holds 128 output columns of decoded weights, not an expanded bank.
  This correctness route is intentionally slow; W3's X3 path is still pending.
- MTP uses EXL3 FC/body/head weights with existing full-HC normalization,
  recurrence/rollback, sampling and eligibility control. The draft head scores
  the prefix plus deduplicated complete 128-column segments for dynamic IDs,
  with original N stride. Mixed subset/full cohorts retain full-head gathering.
  EXL3 cache-only catch-up currently computes the full predictor body because
  its K/V are separate projections; this is a remaining performance limitation.

## Checks and next validation

Passed here: three standard-library Python header/identity tests, Python
syntax checks, `git diff --check`, and an independent Python cyclic-bit-reader
comparison of the K7 extraction formula for 131,072 states. These do not count
as a C++ compile or GPU/numerical/model pass.

After formatting and compiling, run once at the end as the user requested:

```sh
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --parallel 4 --target \
  qwen38_flash_next_exl3_reference_test qwen38_flash_next_exl3_ops_test \
  qwen27b_exl3_kernel_test qwen27b_exl3_reference_test \
  qwen38_flash_next_gpu_probe gufo
nix develop -c ctest --preset gpu-full \
  -R '^(qwen38_flash_next\.exl3_(metadata|reference|ops)|qwen27b_exl3_(kernel|reference)_test)$' \
  --output-on-failure
```

The GPU operator test covers K7 decoded FP16 bits, K5/K6/K7 projections,
rows 1/2/3/7/8/9/32/33/64/65, slot-preserving shared/disjoint top-10 routes,
selected head segments, output canaries, counters and routed graph replay.
The supervisor compiled and ran it; follow-up evidence is recorded below.
Projection and routed per-stage L2 bars remain 1e-5. Routed end-to-end L2 uses
`frozen_input_control_L2 * 1.05 + 1e-5` under the FP16 intermediate contract.

Then use the existing Flash-Next AR/MTP numerical/functional harnesses with
both target/MTP paths pointing to this metadata view. Include image, real
proposals, every rejection prefix, seeded sampling, chunk/continuation,
long-context/cache and C2/C4/C8 shared/disjoint histories. Compare against the
same IQ4_NL PLE reference; upstream per-stream MTP normalization remains a
known oracle mismatch. Validate actual pack loading/upload/permutation bytes,
resident/staging memory and selected/full head agreement before quality or
speed claims. Retain pp2048/tg128 and the category/depth MTP targets from the
design; X3 prefill performance remains separate.

## Proposed commits (not staged)

1. `feat(quant): support K7 EXL3 decode and supplementary checkpoint sources`
   - `src/core/quant/exl3/checkpoint.{cpp,hpp}`
   - `src/core/hip/detail/exl3_gemv.hpp`
   - `src/models/qwen/hip/kernels/exl3_gemv.{hip,hpp}`
2. `feat(qwen38-flash-next): load compressed EXL3 trunk and MTP weights`
   - Model root: `exl3_weights.{cpp,hpp}`, `weights.{cpp,hpp}`, `cpu_ops.cpp`,
     `reference.cpp`, `engine.cpp`, CPU-source hunk of `CMakeLists.txt`
   - `kernels/rocm/device_model.{cpp,hpp}`
   - `tools/exl3_quant/fn_metadata.py`
3. `feat(qwen38-flash-next): execute grouped EXL3 experts and selected draft heads`
   - `kernels/rocm/exl3_moe.{hip,hpp}`, `executor.{cpp,hpp}`, `batch.cpp`
   - HIP-source hunk of model `CMakeLists.txt`
4. `test(qwen38-flash-next): cover EXL3 sources projections and routing`
   - `tests/models/qwen38_flash_next/{CMakeLists.txt,exl3_metadata_test.py,exl3_reference_test.cpp,exl3_ops_test.cpp,gpu_probe.cpp,reference_probe.cpp}`
   - `plan/exl3/F2-IMPLEMENTATION.md`

Model-relative paths above are under `src/models/qwen38_flash_next/`.

## Unit follow-up (2026-10-31)

Supervisor-provided `build/fn-unit.log`: 27B EXL3 reference/kernel and Flash-Next
metadata/CPU-reference tests passed. Flash-Next `exl3_ops` failed after 21.76 s,
printing only `EXL3 relative L2 0.000331`. The missing format-test executable
was not a numerical failure. The old log does **not** identify which `Near`
invocation failed: projection/selected-head checks used 1e-5, while the routed
whole-MLP check used 2e-4. Do not identify a particular shape from elapsed time.

The test input oracle had a verified rounding mismatch: `HadamardBlocks(float)`
multiplies by a double scale, whereas X2b multiplies by an FP32 scale before
FP16 RNE. A scalar impulse with FP32 bits `0x36931406` demonstrates it: after
128-point rotation, the old oracle rounds to FP16 bits 7 and the specified
FP32 producer rounds to bits 6. The test now has an explicit FP32 producer
and a deterministic rounding control; the old oracle remains a measured
control rather than disappearing.

Every numerical check now prints case/shape/K/row count, metric and limit.
Optional between-launch device copies capture the routed gate/up inputs and
inner sums, SwiGLU activation and down input. Production kernels and their
arithmetic are unchanged. These allow exact producer/RNE-bit checks, 1e-5
independent FP64 gate/up/activation checks, and a 1e-5 down check using the
same captured FP16 input. The old 2e-4 whole-MLP gate is **retained** pending
GPU control evidence; no K-dependent tolerance or relaxed limit was added.
The control reports whether accumulated gate/up error changed a down FP16 bin
and whether that displacement accounts for the original end-to-end error.

Only source/Python checks have run on this follow-up. Rebuild and rerun just
`qwen38_flash_next_exl3_ops_test` with verbose CTest output. A named failure
and its control are still needed to classify the original 0.000331 conclusively.

### Matched-control rerun

Supervisor-provided `build/fn-unit2.log` identifies the former failure as
`routed shared k=2560 n=640 K=5 rows=3 down_n=2560 experts=16 topk=10`.
Its legacy E2E L2 is 3.311551950e-4; the frozen-input control is 3.310581865e-4,
while actual versus the frozen-input FP64 result is 7.470798014e-6.
Both producer/RNE mismatch counts are zero; gate/up inner L2 is about 2e-6,
and restored SwiGLU L2 is 3.755994792e-6. Two down-input FP16 bins differ
from the independently accumulated FP64-intermediate oracle. This isolates
the 3.3e-4 displacement to the specified FP16 intermediate boundary, rather
than a kernel defect or an unsupported K-dependent accumulation allowance.

Per the supervisor's decision, the final E2E bound is now
`E2E_L2 <= frozen_input_control_L2 * 1.05 + 1e-5` (3.57611095825e-4 for that
observed control). The legacy number remains printed diagnostically; the
absolute 2e-4 gate does not apply under the FP16 intermediate contract.
All 1e-5 stage checks and exact producer-bit checks remain. No kernel changes
were made for this bound update. Rebuild/rerun `exl3_ops` to finish the cases
after the formerly failing gate, including graph replay and selected heads.

### Small routed-down follow-up (GPU confirmation pending)

`build/fn-unit3.log` passes the larger case but reports down L2
4.564744017e-5 for `k=128 n=128 K=5 rows=1 experts=512 topk=10`.
The frozen-input control is zero; gate/up and activation checks pass around
1e-7, and down rotation/RNE is exact. This is a separate discrepancy after
the down-input boundary, not evidence for relaxing that boundary's gate.
There is no routing-weight combination inside Exl3RoutedMlp: its output is
still per assignment. Gate/up and down consume the same expert tile list.

A local independent Python fixture replay found the nonzero down halves
are subnormal; ordinary FP32 lane accumulation/restoration versus FP64 was
about 7.3e-8. This suggests a device arithmetic/lowering issue, but does not
prove whether packed dot or output restoration caused the observed error.
The candidate changes only routed-down accumulation to explicit widened
FP32 FMAs with pinned conversion boundaries. Gate/up, 27B arithmetic,
rotation, scales and grouping are unchanged. All tolerances are unchanged.
This candidate is uncompiled and unqualified; its performance is unmeasured.

Trace qualification now runs the original packed down dot on the same
frozen slab and tile list into a separate buffer. Both raw inner products
are compared diagnostically against FP64; the candidate raw result and
restoration from the captured inner result each have strict 1e-5 checks.
The small case prints per-slot expert IDs, subnormal counts, packed/raw
FP32 L2 and final restored L2 before any gate can throw. This distinguishes
packed-dot error from restoration error on the next GPU run. Do not report
the suspected root cause or candidate fix as confirmed before that run.

Rebuild `qwen38_flash_next_exl3_ops_test` and rerun
`ctest --test-dir build/fn-gpu-test -R '^qwen38_flash_next[.]exl3_ops$' -V`.
The host has no Nix/compiler; the formatting command failed because `nix`
is absent. `git diff --check` passes. No GPU or Docker was executed locally.

## Exact staging commands

Run from this worktree root, committing each group before staging the next.
The model CMake file has two task-owned hunks: commit 2 takes only the first
(`exl3_weights.cpp`); commit 3 takes the remaining HIP-source hunk.

### 1. feat(quant): support K7 EXL3 decode and supplementary checkpoint sources

```sh
git add -- \
  src/core/quant/exl3/checkpoint.cpp \
  src/core/quant/exl3/checkpoint.hpp \
  src/core/hip/detail/exl3_gemv.hpp \
  src/models/qwen/hip/kernels/exl3_gemv.hip \
  src/models/qwen/hip/kernels/exl3_gemv.hpp
```

### 2. feat(qwen38-flash-next): load compressed EXL3 trunk and MTP weights

```sh
git add -- \
  src/models/qwen38_flash_next/exl3_weights.cpp \
  src/models/qwen38_flash_next/exl3_weights.hpp \
  src/models/qwen38_flash_next/weights.cpp \
  src/models/qwen38_flash_next/weights.hpp \
  src/models/qwen38_flash_next/cpu_ops.cpp \
  src/models/qwen38_flash_next/reference.cpp \
  src/models/qwen38_flash_next/engine.cpp \
  src/models/qwen38_flash_next/kernels/rocm/device_model.cpp \
  src/models/qwen38_flash_next/kernels/rocm/device_model.hpp \
  tools/exl3_quant/fn_metadata.py
# Answer y for the exl3_weights.cpp hunk, n for the exl3_moe.hip hunk.
git add -p -- src/models/qwen38_flash_next/CMakeLists.txt
```

### 3. feat(qwen38-flash-next): execute grouped EXL3 experts and selected draft heads

```sh
git add -- \
  src/models/qwen38_flash_next/kernels/rocm/exl3_moe.hip \
  src/models/qwen38_flash_next/kernels/rocm/exl3_moe.hpp \
  src/models/qwen38_flash_next/kernels/rocm/executor.cpp \
  src/models/qwen38_flash_next/kernels/rocm/executor.hpp \
  src/models/qwen38_flash_next/kernels/rocm/batch.cpp \
  src/models/qwen38_flash_next/CMakeLists.txt
```

### 4. test(qwen38-flash-next): cover EXL3 sources projections and routing

```sh
git add -- \
  tests/models/qwen38_flash_next/CMakeLists.txt \
  tests/models/qwen38_flash_next/exl3_metadata_test.py \
  tests/models/qwen38_flash_next/exl3_reference_test.cpp \
  tests/models/qwen38_flash_next/exl3_ops_test.cpp \
  tests/models/qwen38_flash_next/gpu_probe.cpp \
  tests/models/qwen38_flash_next/reference_probe.cpp \
  plan/exl3/F2-IMPLEMENTATION.md
```

## Session invariance follow-up (GPU qualification pending)

The integration logs report `CheckBatchedSessions` at C6/cycle 0 and
`CheckPrefillChunks` at length 136/boundary 94/step 0. Both use
`RequireExact` (`session_test.cpp:29`), a byte comparison, with no tolerance.
The logs contain no intermediate tensors; they cannot establish which
operator first differed. The following numerical dispatch defects are
verified from source and addressed together:

- `MoeBatch` used the generic shared expert above eight total rows, with
  separate Flash-Next `Swiglu` (`__expf`). Serial `Moe` used the shared Qwen
  fused EXL3 MLP (`expf`, followed by its specified FP16 down-input slab).
  The built shared kernel confirms the exp/division correction sequence.
  EXL3 cohorts now enter `Moe`, using the common `Exl3SharedMlp` helper.
- The pack retains F16 HC down/up/router and F32 GDN operands. `Dense`
  switched them from `SmallGemm` to BLAS during prefill, including activation
  narrowing for F16/BF16, while `DenseBatch` used the small kernel. BLAS
  solution/reduction selection depends on shape. EXL3 plain operands now use
  pinned widened FP32 FMAs and the same wave reduction at every phase and
  width. Existing GGUF calls keep the original kernel specialization.
- Non-prefill compressed projections/groups/shared MLPs and routed experts
  previously switched to the different reference arithmetic above 64 rows.
  MTP catch-up can exceed that width even though each decode cohort is
  bounded. Such calls now use bounded fast passes with unchanged arithmetic.
  Trunk prefill keeps its reference route. The two reported trunk prefill
  calls both requested reference arithmetic; the 64-row cutoff alone does
  not explain their differing target logits.

27B is not evidence that arbitrary EXL3 prefill/fast histories are exact:
`exl3_dispatch.hpp` also switches at 64 rows. Its kernel test compares eight
fast rows against scalar calls exactly (`exl3_kernel_test.hip:236`), whereas
`target_test.cpp:95` limits scheduled chunk equality to Q8 weights. There is
no `Exl3Pass` symbol or `F3-PLAN.md` in this worktree revision.

Build `qwen38_flash_next_exl3_invariance_test` and
`qwen38_flash_next_exl3_ops_test`. Focused CTests:
`qwen38_flash_next.exl3_invariance` (plain HC/router/GDN, C6 and 136-row
chunk boundaries, plus independent FP64 checks) and
`qwen38_flash_next.exl3_shared_invariance` (shared MLP scalar/cohort bits,
former unfused numerical control, and compressed 136-row splits in each
phase, including the former 64-row fast cutoff).
Then rerun the full `qwen38_flash_next.exl3_ops`, existing
`qwen38_flash_next.projection_ops` for the unchanged GGUF specialization,
and the same model session `--batch-only` / `--prefill-only` checks.
No GPU/compiler runs or timing claims were made locally; Nix is absent.
Actual first-divergence attribution and the full-session fix remain subject
to that rerun. All equality checks and numerical limits remain unchanged.

F3's grouped compressed prefill GEMM and decode union can still replace the
respective compressed implementations. Keep the fused shared activation and
its down-input boundary across serial/cohort calls, and preserve the plain
operand FP32 contract. This does not force trunk prefill through GEMV; plain
HC prefill now uses a fixed reduction instead of BLAS, whose performance
impact has not been measured. F3 needs the new equality tests when changing
those arithmetic boundaries.
