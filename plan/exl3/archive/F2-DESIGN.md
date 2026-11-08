# F2 design: Flash-Next from the turboderp EXL3 pack

Design only, 2026-10-31, fn-exl3. Code inspected at `fe707ff3` on
`ai-tools/port-u4f`; no build or GPU execution. **Verified** below means source,
JSON or safetensors-header inspection, not a runtime result. Estimates are labeled.
Follow `PLAN.md:93-99`: use the downloaded pack, preserve gufo quality, retain
the disk n-gram table, and reach the Kyojin MTP targets.

## 1. Loading and representation

**Decision:** bind compressed EXL3 projections directly; retain existing
Flash-Next recurrence, attention, routing and serving behavior. Do not materialize
the whole model as FP16/BF16. The existing binder only accepts GGUF matrices
(`src/models/qwen38_flash_next/weights.cpp:134-216`); EXL3 needs a model-local
binding layer, rather than pretending its bytes are a GGML quant type.

Pack root inspected: `/run/media/m2on/T5 EVO/AI/model/flash-next-exl3-5.05-h6-ng6/`
(turboderp revision `7cef615f`, recorded in `F2-PLAN.md:59`). Download is incomplete:
completed shards 1, 5, 9, 10, 11 were available; shard 2 was `.part`, and the
n-gram payload was absent. Header examples below are verified in those completed
shards; the remaining trunk inventory comes from `quantization_config.json`.
Do not declare pack integrity or loadability until every indexed shard arrives.

Let `L = model.language_model.layers.i`. A projection suffix denotes its
`.trellis/.suh/.svh/.mul1` quartet. All inspected markers use mul1; rates must
come from each tensor's tile width, not global `bits` or `mtp_bits`
(`src/core/quant/exl3/checkpoint.cpp:252-305`).

| Pack module | Flash-Next role / disposition |
| --- | --- |
| `L.mlp.experts.e.{gate,up,down}_proj` | Three descriptor arrays of 512 experts; K5 resident. Gate/up `k=2560,n=640`, down `k=640,n=2560`. Header trellises `[160,40,80]` / `[40,160,80]`; each projection has its own F16 scales. |
| `L.mlp.shared_expert.{gate,up,down}_proj` | Existing shared SwiGLU roles; K7 resident, same matrix dimensions. |
| `L.mlp.gate.weight`, `L.mlp.shared_expert_gate.weight` | F16 `[512,2560]` and `[1,2560]`; form the existing 513-output router without requantization. |
| `L.linear_attn.{in_proj_qkv,in_proj_z,out_proj}` | `ssm_qkv`, `ssm_gate`, `ssm_out`; K7 resident. Widths 10240, 6144, and 6144→2560. |
| `L.linear_attn.in_proj_{a,b}.weight` | F16 alpha/beta, stack as today; not EXL3. |
| `L.self_attn.{q,k,v,o}_proj` | K7 attention projections; q includes its interleaved gate, outputs 12288/512/512, o input 6144. |
| `L.self_attn.indexer.index_qk_proj` | K5 2560→640; compute the combined projection then expose q[512] and k[128] views. Do not slice checkpoint bytes as contiguous output rows. |
| `L.{attn,mlp}_hyper_connection.*`, `model.language_model.hyper_connection_mixer.*` | `hc_attn`, `hc_ffn`, `hc_head`: F16 down/up/inject projections, F32 folded norm. Preserve four streams. |
| `L.ple.{key_proj,value_proj}.weight`, norms, convolution | Existing PLE roles; retain F16 projections, convert small operands to the runtime's F32 convention. PLE is at layer index 1. |
| `model.language_model.embed_tokens.weight`, `lm_head` | Keep BF16 embedding (1.184 GiB) and compressed K6 head. No Q8 embedding conversion. |
| `mtp.layers.0.*`, `mtp.fc_{embedding,hidden}`, `mtp.pre_fc_norm_*` | Bind the draft block directly, sharing trunk embedding/head. Actual headers: experts/indexer K5, **FC K6, attention/shared expert K7**, not uniformly K5. FC trellis `[160,160,96]`. |
| `mtp_hyper_connection_mixer_patch.safetensors` | Explicit supplementary source of `mtp.hyper_connection_mixer.{hc_norm,input_mix_weight_down,input_mix_weight_up}.weight` (13,127,680 bytes). These three keys are absent from the model index. Require this patch for MTP; never silently borrow the trunk mixer. |
| `model.visual.*` | Keep the existing separately supplied vision artifact in the first release; pack vision conversion is separate scope. Include an image regression when qualifying the integrated model. |

Destination roles/shapes are defined at
`src/models/qwen38_flash_next/weights.hpp:75-150`; upload stacking is at
`src/models/qwen38_flash_next/kernels/rocm/device_model.cpp:237-299`.
MTP header details refine the shorthand allocation in `F2-PLAN.md:65-75`.

**Layout and small conversions.** Apply the model's grouped→tiled GDN V-head
permutation consistently to qkv V outputs, z, out-projection inputs, conv channels,
alpha/beta, A and dt. The consumer explicitly uses `h % num_k_heads`
(`src/models/qwen38_flash_next/reference.cpp:277-281`). Reorder whole 128-channel
segments and matching trellis tiles/scales losslessly, following the mechanism in
`src/models/qwen/exl3_weights.cpp:567-605`, with Flash-Next dimensions. Do not copy
27B recipes wholesale. Convert norm gamma to F32 with the appropriate +1
(HC, attention/indexer, PLE, nextn norms); GDN output norm has no +1. Convert
`A_log` to F32 `-exp(A_log)` and conv/dt to their existing layout/type. Confirm
each recipe against independent dequantized operands before model execution;
the 27B transform categories are at `src/models/qwen/exl3_weights.cpp:32-37`.

**Sidecar.** Reuse the tensorless GGUF approach (`XA-RESULTS.md:110-129`): copy
the same Flash-Next GGUF metadata/tokenizer/template, omit `split.*`, set
`gufo.exl3.checkpoint` relative to the sidecar. Add explicit relative references
for the retained GGUF PLE source and MTP mixer patch, plus immutable source
identities. Generate trunk/MTP metadata views for the existing model/MTP path
options; introduce no new backend alias. Check text geometry, HC/PLE hash
constants, vocabulary/merges/special IDs/template and artifact identities before
upload. The current engine binds GGUF before creating tokenizer, PLE and MTP
(`src/models/qwen38_flash_next/engine.cpp:92-150`), so detect the sidecar before
ordinary `ModelWeights::Bind`; a tensorless file cannot pass that binder.

Use `core/quant/exl3::Checkpoint` for shard parsing and validation. Add only
model-independent source-region/supplementary-file support there if needed;
the existing index walk does not discover the mixer patch
(`src/core/quant/exl3/checkpoint.cpp:149-175`). Put name recipes and overlay policy
in `qwen38_flash_next/exl3_weights.*`. Add an explicit EXL3 storage alternative
to host/device weight views, backed by stable owned `DeviceLinear` arrays;
guard GGUF `RowBytes/Expert` operations (`weights.hpp:38-63`). Upload bounded,
aligned slabs through the weight allocator, including scales and descriptors in
resident accounting; avoid tens of thousands of individual allocations or a
second complete host weight copy. Keep mapped files alive until uploads finish.

### Disk n-gram decision

**Keep the current IQ4_NL GGUF table in F2**, as already decided in `PLAN.md:97`.
Reference it explicitly from the sidecar; do not copy the full GGUF or upload
the table. Current hashing, EOS/history, prefetch, 128 MiB encoded-row cache and
platform random-read path remain (`ngram.cpp:18-25`, `29-62`, `79-117`, `150-207`).
The table is excluded from GPU weights (`kernels/rocm/device_model.hpp:64-65`).

NG6 would be a **different codec**, not another `DeviceLinear`: the manifest says
`[320001536,61]` I16, 122 bytes/160-value row, 36.359 GiB payload versus IQ4_NL's
90 bytes/row and approximately 26.82 GiB. It is a 160-wide tail-biting ring with
a row scale and per-head bias, without the matrix Hadamards
(`build/w9-src/exllamav3/exllamav3/modules/quant/exl3_lib/ngram_codec.py:5-19`,
pinned upstream `cdc785db`). Switching gains no GPU capacity; it adds about
9.54 GiB disk and CPU trellis work. At the current power-of-two cache sizing,
both formats retain 1,048,576 slots, occupying 90 versus 122 MiB; NG6 does not
reduce the slot count at this cache budget.
Both usually read a 4 KiB direct-I/O window, sometimes 8 KiB across a boundary;
payload size alone does not predict latency (`ngram.cpp:160-177`). F0 already
observed material cold-prefill cost (`F1-RESULTS.md:85-101`).

This deployment is **EXL3 trunk + IQ4_NL PLE**, not numerically identical to the
complete NG6 pack. For engine parity, use an ExLlamaV3 reference adapter with the
same decoded IQ4_NL PLE rows, and independently check block outputs with frozen
PLE inputs from the unmodified pack. Report full NG6-reference quality separately;
do not charge a table difference to the EXL3 matrix engine. If IQ4_NL limits
quality, propose a separate NG6 disk-row reader using the same cache/I/O framework.

## 2. Kernel and MTP design

**Reuse arithmetic, extend dispatch.** `DeviceLinear` already represents
`svh * H((H(suh*x)) W')` (`src/core/quant/exl3/device.hpp:4-15`). Reuse the
exact mul1 decoder, FP32 butterflies, FP16 rotated-input rounding and FP32
accumulation in `src/core/hip/detail/exl3_gemv.hpp:8-33`. Current fast support
ends at K6 (`src/models/qwen/hip/kernels/exl3_gemv.hip:567-570`); its 64-bit lane
window explicitly requires K≤6 (`src/core/hip/detail/exl3_gemv.hpp:76-89`).
**K7 is a prerequisite**, not a dispatch-enum change: implement wider/two-window
state extraction and cyclic tails, independently bit-check all states/positions.
Keep K5/K6 behavior unchanged. The public path accepts up to 64 rows in passes
of eight and at most three matrices (`exl3_gemv.hpp:38-72` under the Qwen kernel
directory); it cannot directly express 10 independently routed experts.

- **Decode:** new Flash-Next routed entry points consume device expert IDs and
  per-expert gate/up/down descriptors. Rotate the token separately with each
  selected matrix's `suh`; never share one rotation across all experts or even
  gate/up without proving identical scales. One grouped gate/up grid over the
  ten slots, restored-output SwiGLU then down-input rotation, one grouped down
  grid, followed by the existing ordered weighted sum/shared gate/HC combine.
  Give every slot separate scratch and segment counters; no floating atomics
  into the hidden row. Avoid 30 host projection calls per layer.
- **Batched decode/verify:** compact assignments by expert on device, retaining
  token, request and top-k slot IDs. Reuse each expert's weights across its
  actual rows in 1–8-row tiles; mask ragged tails. Test fully shared and fully
  disjoint routing. Use a bounded device tile list to avoid a CPU count download
  on every decode layer. Existing prefill downloads counts and constructs a tile
  map (`executor.cpp:1839-1881`); reuse its semantic row/slot mapping, not its
  synchronization by default.
- **Shared expert/router/dense:** use the same fused EXL3 SwiGLU primitive for
  K7 shared weights, keeping its sigmoid scalar gate. Keep router F16 and
  `RouterTopK`/normalization/tie rules (`executor.cpp:2725-2735`). Group K7 q/k/v
  and GDN qkv/z descriptors without concatenating their rotations. Do not reuse
  the 5120-wide RMSNorm fusion unchanged: its shape guard is explicit
  (`src/models/qwen/hip/kernels/exl3_gemv.hpp:99-108`); preserve HC's 10240-wide
  norm and 2560-wide projection inputs first.
- **Prefill:** request W3's X3 trellis→LDS→FP16 WMMA B producer and small-row
  kernel as a shared primitive, adding expert descriptor, gather/scatter and
  bucket-length inputs. At pp512/2048/8192, uniform routing averages only
  10/40/160 rows/expert: choose by **bucket rows**, not total prompt rows.
  Dense BM=256 tiles waste most expert lanes. Start with 16/32/64-row buckets,
  fused gate/up and masked 640-wide tails; tune larger hot buckets separately.
  The current WMMA routing/compaction contract is at `executor.cpp:2852-2904`;
  X3's dense proposal is `X3-DESIGN.md:36-49,78-86`. Do not expand the 70.75 GiB
  expert bank to FP16. A bounded one-expert dequant/GEMM path is the correctness
  fallback, not the final prefill implementation.

**MTP:** route both draft and target verification through these same primitives,
including odd widths and C2/C4/C8 ragged cohorts. Preserve gufo's full-HC hidden
normalization, per-stream FC, recursive carry, rejection correction, final-row
catch-up and F1 rollback/KV interfaces (`executor.cpp:4495-4529`). The independent
norm audit is `tests/models/qwen38_flash_next/mtp_audit.cpp:111-126`. Upstream
currently normalizes each stream separately
(`build/w9-src/exllamav3/exllamav3/modules/arch_specific/qwen4_exp_mtp.py:149-152`)
and labels its semantics empirical (`:13-23`): record this reference mismatch,
do not alter gufo to make that oracle agree. Compare EXL3 projections independently
and use gufo's scalar contract for the predictor body until reconciled.

**Draft-head blocker:** existing prefix/head-row copies assume GGUF row-addressable
storage (`device_model.cpp:340-352`, `executor.cpp:1402-1428`). For K6 head, retain
the original full-n stride and evaluate complete 128-column Hadamard segments
covering requested prefix/special/dynamic IDs, then gather selected logits.
Never shorten `n` to view a packed prefix or extract a single effective output
before its segment rotation. Deduplicate segments; qualify subset/full candidates,
and retain the existing eligibility/fallback policy. Full-head evaluation is an
initial correctness fallback, not acceptable evidence for the final MTP speed gate.

## 3. Memory and speed budget

**Calculated, not measured.** The coarse F1 model gives `14.06*5 + 7.6 = 77.9 GiB`
versus measured UD MTP 79.31 GiB / AR 76.73 GiB
(`F1-RESULTS.md:65-74,103-115`). A more specific inventory is:

| Component | GiB | Evidence / accounting |
| --- | ---: | --- |
| Trunk routed experts, trellis + scales/markers | 70.752 | Sum of manifest entries; agrees with `F2-PLAN.md:65`. |
| Remaining trunk, embedding/head included, PLE table excluded | 5.403 | Manifest sum; includes 1.193 HC, 1.184 embedding, 0.444 head. |
| MTP indexed tensors | 1.556 | Sum of all 6,200 `mtp.*` headers in completed shards 10/11. |
| Missing indexed MTP mixer, supplied by patch | 0.0122 | Three patch header entries. |
| Small unlisted conv/A/dt operands, F32 conversions, alignment/descriptors | Allow <0.1 | Manifest omits some buffers; count actual allocations, not just manifest totals. |

Plan for **about 77.8–78.0 GiB resident with MTP**, about 76.2–76.3 GiB AR,
before executor/session buffers and the separate vision encoder. Retain the
128 MiB host row cache; account for host mappings/page-cache and staging peaks
on unified memory. The observed HIP budget is 112 GiB, not 128 GiB
(`F1-RESULTS.md:60`). With roughly 2.8 GiB runtime overhead, only about 31 GiB
remains for sessions/workspaces. Do not promise eight 262K sessions using F1's
old k4v2 projection: the chosen format is k4v4 (`PLAN.md:91`); use F1's final
measured admission accounting after integration.

For one target AR token, no assumed cross-layer/cache reuse, decimal bytes:

```
routed = 48 * 10 * 3 * (2560*640*5/8 + 2*(2560+640))
       = 1.484 GB (including per-projection scales; markers negligible)
dense  ≈ 4.53 GB (all always-used projections, HC, router and full head)
total  ≈ 6.02 GB/token + activation/recurrent/KV traffic
200 GB/s / 6.02 GB ≈ 33.2 target tokens/s, weight-only ceiling
```

The dense budget is a manifest sum excluding embedding and disk table, with
small buffer traffic additional. **Only about 25% is routed experts**; HC alone
is about 1.28 GB/token. At this bandwidth, 2–7 ms of other work implies roughly
27–31 AR tok/s [estimate], not a guaranteed uplift over today's ~28.3
(`docs/models/qwen3.8-flash-next/BENCHMARKS.md:17-24`). Measure recurrence,
Hadamard launches, QSA selection and disk stalls separately before optimizing.

For M verify rows, let U be the actual distinct experts per layer. A useful
traffic estimate is `Bverify ≈ 4.53 + 1.484*(U/10) GB`; U is not always ten.
Independent uniform routes give `U=512*(1-(1-10/512)^M)`: at M=4, U≈38.8 and
weight-only verify≈51.5 ms; at M=8, U≈74.7 and ≈78.1 ms. Correlated rows reuse
more. The indexed MTP body reads about 0.119 GB/draft with top-10 experts,
plus up to 0.477 GB for a full head (before patch/activation/state traffic).
Use `accepted output tokens / (draft + verify + sampling/state time)` for speed;
arithmetic cannot predict acceptance. Under perfect acceptance, three full-head
drafts plus a four-row verify have a weight-only ceiling near 66 tok/s; a smaller
head or stronger routing reuse can raise it. This is a scenario, not a forecast.

Kyojin's **MTP** chat/prose/code targets are 52.8/47.5/73.2, 8K/32K 52.2/51.1
(`plan/archive/rotated-quant/W9-RESULTS.md:105-118`). They are not AR rates or a matched
runtime comparison. EXL3 bytes alone do not establish those targets; expert
reuse, efficient K7 trunk, draft-head selection and actual acceptance dominate.
Record per-category results without averaging a missed target away.

## 4. Delivery sequence and ownership

| Stage | Files / owner boundary | Pass criterion |
| --- | --- | --- |
| A: inventory and loader | F2: new `src/models/qwen38_flash_next/exl3_weights.*`; `weights.*`, `engine.*`, `kernels/rocm/device_model.*`; focused model tests/CMake. Core checkpoint changes only after coordination. | Complete pack identity; every required name, shape, dtype, rate, scale and mixer patch accounted for; missing/partial sources rejected; tokenizer/template equivalence; bounded upload memory and exact resident accounting. CPU dequant bits match pinned ExLlamaV3 for actual K5/K6/K7 tensors and permuted roles. |
| B: projection and routed correctness | F2 model dispatch and `kernels/rocm/exl3_moe.*`; W3 owns shared X2 decoder/GEMV and K7 extension. | Independent FP64/ExLlama projection checks, including full decoded FP16 weight equality; widths 640/2560/6144/10240 and head; rows 1/2/3/7/8/9/32/33/64/65, routing overlaps, boundary allocations. Freeze numerical tolerances before tuning; diagnose first divergent operand/layer. |
| C: model and MTP | F2 executor integration/tests; W3 shares grouped/selected-segment kernels. | Fixed-history engine KL against the same-IQ4 reference: provisional median ≤1e-4, mean ≤1e-3 (27B precedent `XA-RESULTS.md:78`), additionally establish Flash-Next's unquantized engine floor. Preserve independent HC/MTP contract; scalar/batched and chunk/continuation parity, real drafts, every rejection prefix, full/subset head, greedy and seeded sampled replay. Unresolved reference mismatch blocks a parity claim. |
| D: prefill and integration | W3 owns X3 primitive/layout/dispatch; F2 owns expert gather/scatter and bucket scheduler. F1 owns paged KV/KVarN. | No whole-model expansion; pp512/2048/8192 and d32K prefill no regression against matched UD-Q4_K_XL; operator, model and long-context checks pass. Rebase onto F1's accepted k4v4 result; validate page/rollback/cache behavior without changing its format. |
| E: adoption | Numerical tools under `tests/models/qwen38_flash_next/`; functional/quality/speed harnesses under `tests/functional/`; retained results in F2 results + model QUALITY/BENCHMARKS/EXPERIMENTS. | Against BF16 on matched token histories, EXL3+IQ4 PLE is at least UD-Q4_K_XL quality: full-vocabulary KL, ΔNLL, top-1, per-position tails/flips and task checks; retain paired block confidence intervals and inspect worsened tails. Include Korean, tool calls, long-context/needle and actual MTP acceptance. Do not substitute KL_256 or missing-model skips for quality. |

For E, freeze candidate and main production binaries, toolchain, exact model
identities, prompts, sampling and cache history. Run candidate correctness on the
Flash-Next AR/MTP profiles and affected `long-context`/`cache` suites; run matched
main controls for timing, not duplicate full correctness. Retain the standard
pp2048/tg128 speed benchmark and card prompt categories at d0/8K/32K, affected
C2/C4/C8 shared/disjoint histories, and cold/warm PLE I/O separately. Use ABBA
and **5%/10 ms per request and phase**; keep every result, investigate flagged
histories with alternating controls, fix confirmed regressions, label noise
inconclusive. Different quantizations can change free-running text/draft counts:
also compare fixed-history execution costs and report completed-task latency,
rather than treating unequal work as a kernel regression or gain. For reported
coding loops, replay `tests/functional/pi_agent.py` and inspect real tool results.
Profile in separate runs, reproducing production allocation and cache traffic
per `.agents/skills/optimize-kernel/SKILL.md`.

**Conflict policy:** F2 does not edit W3's `src/core/hip/detail/exl3_gemv.hpp`,
`src/models/qwen/hip/kernels/exl3_gemv.*`, `exl3_dispatch.hpp` or X3 files without
an ownership handoff. Agree on model-independent descriptors, K7 extraction,
ragged rows and selected output segments first; do not include a 27B tensor-ref
adapter inside Flash-Next. F1's continuation layouts, page pool, KV writers,
snapshots and KVarN arithmetic stay F1-owned. Both F1/F2 touch the large executor:
integrate sequentially at projection/routing call sites and rerun affected
continuation/MTP checks. W9 owns draft-policy/cost changes; supply measured EXL3
costs, keeping policy experiments separate from format qualification.

## Open decisions for the user

1. If the retained IQ4_NL table prevents the quality gate, authorize NG6 disk-row
   support as an explicit scope extension? Default: keep the existing decision;
   report the measured deficit and do not silently switch tables.
2. For final Kyojin parity, accept the published card as a target only, or require
   a fresh matched client-timed run on the same prompts/hardware/settings?
   Recommendation: retain the card targets and require the matched run before
   claiming engine superiority; current server/client timing evidence differs.
