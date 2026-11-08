# EXL3 / Flash-Next handoff (updated 2026-11-08; B5 runtime deployed)

## Open: B9 GPU pass, 2026-11-08

B9 ([brief](archive/B9-STRUCTURE-BRIEF.md)) split the kernels and executor by
stage, moved the host runtime to `src/models/qwen38_flash_next/runtime/`, the
model benches and tools to `tools/qwen38_flash_next/`, and the serving
supervisor into `tools/serving/flash_next/` (installed with `install.py`).
All 7,059 device symbols are identical to the pre-B9 build; host code changed
only by relocation. Still to run: the GPU operator tests and one combined
correctness and timing pass of the B9 binary against B5, then a server restart
onto the installed supervisor copy.

## Active default: c03 Gufo B5 runtime, 2026-11-08

The service runs the B5 binary (sha256 `31cee0b6…`, 126,048,792 B, source
`5912ed64` (src `eb789c77`), build `build/b1` RelWithDebInfo gfx1151, pinned Nix ROCm 10.1)
from `~/.local/share/gufo/runtime-b5-20261108/` (`RUNTIME.json`: image,
Nix volume, `LD_LIBRARY_PATH`, evidence `build/b1/gpu-b5/RESULT.json`).
Model, C2/native262144 FP16 pool, MTP and sampling settings are unchanged
from the c03 section below. B6 and B7 were measured but not deployed; the
B8 cleanup changes no production code. Measured results and lessons:
[docs/GFX1151-KNOWHOW.md](../../docs/GFX1151-KNOWHOW.md); batch index:
[B-SERIES-SUMMARY.md](B-SERIES-SUMMARY.md). Previous runtime (c03 initial,
`6edf0d44`) and the pre-B5 server script
(`~/.local/share/gufo/gufo-server.py.bak-20261108-pre-b5`) are kept for
rollback.

## Previous default: c03 Gufo, 2026-11-07

User authorized temperature0.9 and starting our model. `llm` now delegates to
`/usr/local/bin/gufo`; persistent controller uses the qualified current Nix10.1
ELF6edf0d44 mounted from the frozen test runtime (not the older image ELF).
Model: `ai.tools/storage/models/llm/flash-next-exl3-gufo-c03`, immutable payload
hardlinks with copied metadata; original final assembly and evidence preserved.
C2/native262144 per conversation, FP16 shared512K pool14,408,908,800bytes,
MTP max7/medium, embedded BF16 vision, top_k20/top_p0.95/min_p0, temperature0.9.
RAMcache2GiB/disk50GiB,0.0.0.0:18740, no serving/request/stall expiration;
4GiB host floor and desktop diagnostics log-only remain.
Run cae6f0a41f5e ready; default-parameter arithmetic and image smoke passed,
actual FP16 and drafts confirmed. Available RAM after checks about27GiB.
Not a full256K stress test. Kyojin stays stopped. `llm start/stop/status` control
this service; old conversion/hourly monitors remain disabled.
Evidence/rollback: `/home/m2on/.local/share/gufo/c03-deployment-20261107/`.


## Native EXL3 vision (2026-11-04)

Gufo direct embedded EXL3 vision is implemented and tested on Turboderp K6
and Vemloir K8. The permanent Gufo controller selects native-exl3-vision-rocm101
and no longer mounts/requires the BF16 GGUF projector. Original weights retained.
Kyojin remains the default; restored run `873264740ec0` is healthy.
[Results, image identity, numerical limits and artifacts](archive/F3-VISION-NATIVE.md).

## Current default and Flash-Next KV references (2026-11-04)

Latest user switch: Gufo stopped; Kyojin is ready again, run5872fb29bf5c,
confirmed health source=kyojin. Active/default: native256K,C2,shared512K FP16,
vision,medium,MTPmix7 at0.0.0.0:18740. Gufo FP16 profile below is preserved
for its next explicit start; its memory measurements are historical. Stopped
Gufo continuation cache (~15GiB) was cleared to meet Kyojin disk admission;
models and user conversation files were retained.

Previous Gufo measurement run used C2, native256K per
conversation, FP16 shared512K KV, vision, medium/MTP7,0.0.0.0:18740. Kyojin
is stopped; `llm`/`kyojin` default aliases remain Kyojin. The permanent Gufo
controller selects `ai-tools/gufo-eval:fn-kv-references-a03-rocm101` and now
uses FP16 with14,408,908,800B (13.419GiB),2,050 blocks including guards.
This costs7.296GiB more than the previous K6V6 pool. YaRN remains disabled.
Run d46df13dd50a passed a bounded memory check: two concurrent2,431/9,011-token
prompts with96-token output caps, then a224px image with correct red/blue
answer. Host MemAvailable minimum10.687GiB; after checks10.725GiB. This is
not a full256K stress/quality/performance comparison. Other scratch/graph/image
memory can grow; the4GiB host shutdown floor remains. Controller cgroup stats
omit most UMA GPU allocations and must not substitute for host MemAvailable.
Receipt: `/home/m2on/.local/share/gufo/fp16-c2-256k-20261104/RESULT.json`.
Stopped Kyojin's regenerable continuation cache (6.658GiB) was cleared to
retain existing Gufo disk reserve admission; models and user sessions retained.

Flash-Next now physically shares immutable GPU prefix blocks across branches,
not only pool capacity. Private recurrent and incomplete frontier state remains
per conversation. Layout version4 excludes old component disk checkpoints.
Core88 GPU checks, byte-exact32760 saved-K6 numerics, standard speed observations,
CPU/review, C4 pressure, branch/image replay, Pi and disk restart all pass.
A duplicate-prefix reclaim bug found during validation is fixed; failed runs
are retained. No full256K or matched speedup claim. Keep baseline and candidate
binaries/Nix10.1 intermediate objects in build/fn-kv-sharing/.
[Implementation, qualification and limits](archive/F3-KV-REFERENCES.md).


## Archived local profile: K6V6 / native256K (2026-11-03)

User requested K6V6 and disabled extended context/YaRN. Global `gufo-start.sh`
uses permanent `/home/m2on/.local/share/gufo/gufo-server.py`: C2/262144,
K6V6 two-context pool6574694400B, vision BF16 sidecar, medium/MTP, no serving
expiry/request/stall timeout,64MiB HTTP bodies,4GiB host reserve. Bind remains
0.0.0.0:18740. Pi/DSH context settings now262144; DSH web restarted.
Image `ai-tools/gufo-eval:k6v6-256k-rocm101`, ID
`sha256:b9338d46667d8c54f58e2876053a4b853f520c226d3db2224f5658aa5b012610`.

K4/K6 operators, K6 shared/native restoration, numerical32760-position collection
and candidate PP/AR/MTP pass. Desktop activity makes timing advisory; short
MTP observation35.66 vs saved40.92 is not performance-neutral. No full256K or
matched K4/K6 task-quality claim. Independent review approved. Preserve prior
images/baselines and `build/k6v6/` incremental objects. HTTP/client verification
receipts and complete current status are in `/home/m2on/.local/state/gufo/k6v6-256k-20261103/`.

Final HTTP:C2,9637-token repeat fully cached, image colors and medium arithmetic pass.
DSH/Pi both replyOK. Actual restart restores9630tokens from K6 disk cache and
prefills7, same answer. Port preflight now tolerates TIME_WAIT while rejecting
active listeners. Final selected server is ready; see RESULT.json.

Old K4/YaRN continuation disk cache was cleared while stopped to fund the build;
conversation histories and weights remain. New cache grows within50GiB quota.
The older pitest/worktree launch instructions below are historical, superseded
by permanent global launchers and retained-worktrees build storage.

## Local Pi configuration update: C4 / 512K

User selected C4 with524288 tokens per session. Same shared-KV image below,
static YaRN factor2 enabled,4098 blocks/9920176128B pool (two full contexts),
RAM continuation retention2GiB and disk50GiB. Pi client context also524288.
Ready36.14s; short C4 first/repeat eight requests and Pi1.4.0 three tasks/nine
tools pass. Host16GiB floor remains enforced. Initial Pi harness run used a
relative config path and sent no request; corrected absolute-path run passes.
No full512K quality/occupancy/latency test, and earlier native256K numerical
and speed qualification does not establish YaRN equivalence. Receipts/config:
ai.tools test/experiments/gufo_upstream_port_20261023/c4_512k_artifacts.
`/home/m2on/projects/pitest/README.txt` describes current execution.

## Qualified implementation: Flash-Next shared K4V4 KV and Pi (256K evidence)

Implemented shared two-context pool (2050 blocks,4.62GiB), C4 with262144
tokens per session,4GiB RAM continuation cache and50GiB disk quota. Current
image `ai-tools/gufo-eval:f3-shared-kv-4757e04f-rocm101`; Pi1.4.0 workspace
`/home/m2on/projects/pitest`, binds0.0.0.0:18740. Shared fixtures, numerical
bytes,32K state, C4 HTTP, actual restart/disk restore and3 Pi tasks pass.
PP/AR/MTP differ by at most0.264% from saved baseline; no baseline rerun.
Full two-times-256K inference remains untested. Disk cache stores continuations,
not active-page swap. The supervisor expires each window after one hour;
`./server status` / `./server start` / `./pi`. Preserve all incremental objects.
[Implementation, limits, failed attempts and complete results](archive/F3-SHARED-KV.md).

## Latest: prefill live-subtile port rejected at compiler screen

Three bounded X3 prototypes (four-way K loop, four-way full body, two-way
full-subtile/generic) all increase VGPR substantially and introduce BN128
private spills. No GPU/model/baseline rerun. Keep the qualified image and
all existing prefill/TG/MTP improvements below. Preserve incremental objects
in build/f3-prefill-live-tiles and ai.tools prefill_live_tiles_artifacts.
Do not repeat these clones without a distinct code-generation mechanism.
[Resources, limits and reproduction](archive/F3-PREFILL-LIVE-TILES.md).

## Retained kernels: exact live-row verification

K5 R3/R4 gate/up and scaled down choose the actual row loop once per tile.
Native5.05 MTP d0=40.85tok/s versus saved40.09 (+1.90%); d32K=79.48 versus78.53
(lookup-heavy, within saved spread). PP2048/8192=1426.88/1460.75, AR=25.20/24.49;
prefill and AR preserved. Quality collector bytes, generated output hashes and
all cycle/draft/acceptance counts unchanged. Operators, sampling, C2/4/6/8,
32K state, cache and health pass in one combined final window.

Local image `ai-tools/gufo-eval:f3-rows-91fce75a-rocm101`; qualified ELF91fce75a.
Preserve `build/f3-tgmtp3/` and all prior Nix intermediates. Baselineb4ebe27c and
saved screens remain; no baseline rerun. Cold component gains21–32% gate/up,
10–14% down are not full-model gains. No Kyojin parity claim.
Dense TN2 load grouping already exists; global-address-space/aligned-K probes
were neutral. Do not repeat them. Saved b4ebe27c profile is pre-change evidence,
still useful for unchanged dense K7; it is not a post-change latency profile.
[Mechanism, final results and reusable evidence](archive/F3-TGMTP-ROWS.md).

## Previous: same-pack TG/MTP attribution

Corrected neutral-greedy comparison using the same internal Kyojin pack:
Gufo AR30.49/30.51/30.23, MTP42.55/46.94/62.66tok/s (prose/chat/code).
Saved Kyojin MTP42.52/49.44/63.22; no rerun. Gufo paired text matches3/3;
Kyojin's earlier AR/MTP token/text gate failed3/3. Same files still differ in
HC effective quantization, draft head, KV codec and runtime, and cross-engine
outputs differ. These are single-pass observations, not statistical/pure-engine
parity. Native5.05 selection remains unchanged; no new kernel speedup claimed.

Current native5.05 real-prose64 profile afterwarm16: AR dense35.44%,experts20.39%;
MTP experts27.43%,dense22.21%,draftsubsethead2.77% of decode span.
Prior pre-LDSfix profile is historical. Reuse this new trace; next distinct
operator candidates target denseK7 and routedK5R3/4, with small checks batched.
Possible graph first-use/capture is not separated from expected eager work;
do not call all gaps CPU waste. No full quality/PP sweep or rebuild repeated. Both completed windows
healthPASS; first invalid penalty-default comparison was stopped and preserved.
[Results, exact limits and next batch](archive/F3-KYOJIN-SAME-PACK.md).

## Local Kyojin measured; equality failed

Kyojin and native K4 n-gram now reside on internal NVMe at
`ai.tools/storage/models/llm/flash-next-exl3-kyojin/` (88.82 GiB).
Five unused internal Flash-Next GGUF payloads were removed with user approval;
external originals and provenance remain. About 166 GiB free. Reusable Kyojin
source/extension/SDK/JIT are retained; do not rebuild or recopy without cause.

All 12 requests completed; GPU health PASS. PP2048/8192=1550.88/1386.46tok/s
(real text, not the Gufo fixed-pattern fixture). AR prose/chat/code≈31.3tok/s;
MTP=42.52/49.44/63.22tok/s. **AR/MTP token and text equality failed 3/3**;
no production acceptance or matched engine/quality parity is established.
Gufo native selection below remains unchanged. The frozen Kyojin source uses
its pinned ROCm10.1 nightly wheels, distinct from the Gufo Nix10.1 release.
[Exact scope, provenance and results](archive/F3-KYOJIN-LOCAL.md).

## CPU PLE diagnostic

Native production selection below is unchanged. A bounded host-timing run finds
PLE host work0.46% of PP2048/8192 and1.74%/1.69% of AR/MTP128 on the current
fixture. Outputs and MTP counts match the saved native run; healthPASS.
The old first-Embed GPU trace omitted CPU setup. This diagnostic addresses host
waits only, not worker/shared-bandwidth or GPU mapped-memory costs; cold/prose
behavior remains unresolved. No baseline/full quality rerun.
[Evidence and exact limitations](archive/F3-PLE-HOST-PROFILE.md).


## Current native EXL3 selection, November 2, 2026

**Use the pack directory directly; no GGUF metadata/PLE dependency.**
Image `ai-tools/gufo-eval:f3-native-b4ebe27c-rocm101`,
wrapper `/opt/gufo-native/bin/gufo`, ELF SHA
`b4ebe27cb5cbd3173143ddf129b9188db6f13178682921fb3c8dccbaadbc9455`.
`--model /models/fnx/ --speculative mtp --mtp-model /models/fnx/`.
The 5.05 pack uses its internal-NVMe native K6 n-gram file; Kyojin's sharded K4
reader is CPU-qualified; the pack is now on internal NVMe (local Kyojin results above).

One combined window: quality, sampling/EOS, C2/4/6/8,32K,cache/policy replay,
PP/TG and hardware health all passed. No GGUF mappings, no old baseline rerun.
PP2048/8192=1423.76/1461.67tok/s; AR d0/d32K=25.20/24.43;
MTP=40.09/78.53. Short MTP is14.15% below the saved rate: new n-gram weights
change output, acceptance77.88%→64.91% and cycles40→54. Preserve this caveat;
there is no claim of universal speed parity. PP/AR/same-output d32K MTP stay
within1%. Saved-corpus NLL improves by.007451 with negative paired windowCI.

[Native implementation, exact scope and results](archive/F3-NATIVE-PACK.md) ·
[Machine-readable receipt](archive/F3-NATIVE-PACK.json). Previous selected images and
all baseline/model artifacts remain available. Keep native input for the next
MTP investigation; do not silently mix in old IQ4_NL PLE.


## Kyojin target comparison, November 2, 2026

The qualified TG/MTP image completed the frozen K17/D8 serving suites in one
window (40.1s ready,95.6s K,175.9s D);25 measured requests and GPU health pass.
No rebuild, baseline rerun or new numerical qualification. All17 historical K
completion hashes agree; that is output continuity, not an independent oracle.

C1, context49152, EXL3 5.05, pinned Nix ROCm10.1, KVarN k4v4, MTP+lookup,
thinking off, temperature0, up to256tokens, prompt cache off. Client rate uses
(completion_tokens−1)/(first-to-last useful SSE event span), matching pinned
Kyojin bench.py; server decode rate is retained separately. Medians below are
across distinct prompts, one pass each, not repeated-run confidence estimates.

| Category | Prompts | Gufo client tok/s | Gufo server tok/s | Published Kyojin target | Client gap |
|---|---:|---:|---:|---:|---:|
| Chat | 3 | 41.67 | 41.83 | 52.8 | −21.08% |
| Prose | 3 | 34.65 | 34.79 | 47.5 | −27.05% |
| Code | 3 | 57.25 | 57.48 | 73.2 | −21.79% |
| Multilingual | 4 | 38.94 | 39.09 | — | — |
| Copy/edit | 4 | 64.63 | 65.00 | — | — |

All three published category targets remain unmet. This is a cross-machine,
cross-pack reference: Kyojin had not been run locally at that stage.
See the later local run above; it does not turn this published table into a
control. Frozen source2132bd2 has
three card prompts/category, while today's public description says four.
The saved published table is not a matched performance control.

D labels are approximate: d8k actually contains7274–7509tokens; d32k contains
27158–39256tokens. Client mixed-task medians are40.55 and35.61tok/s, respectively.
Story subgroups are35.06/31.91; documentation subgroups47.56/43.26 (two each).
Published Kyojin8K/32K essay rates52.2/51.1 are different workloads and exact
depths. Do not call this a matched depth comparison. Server prefill medians
1293.51/1280.75 also belong to these documentation inputs and the serving path;
they do not replace the frozen fixed-pattern prefill benchmark.

This confirms that79.31tok/s on the earlier repetitive lookup fixture is not
a general-generation rate. Preserve current kernel improvements. Any next
optimization should first attribute the remaining cost on a real prose prompt
using this optimized ELF: dense/HC, draft+verify cost per committed token, and
PLE/host waits. The prior native profile predates verifier LDS removal and
cannot establish the new bottleneck; no new profile was run in this comparison.

Receipt `fn-tgmtp-kd-nix101`; per-request client/server rates, output hashes,
acceptance/lookup counts, runtime maps, frozen inputs and published-page SHA
are retained in the comparison report. Existing numerical qualification and
its NLL caveat remain unchanged. No Kyojin quality parity is claimed.

[Comparison](archive/F3-TGMTP-KYOJIN-COMPARISON.json).

## TG/MTP routed and HC optimization, November 2, 2026

Pinned Nix ROCm10.1, EXL3 5.05, C1 TG128, KVarN k4v4, temperature0,
seed314159, r2, depth0/32768 and common context capacity32897. Frozen
batch6 was measured once per mode and saved as the current control; historical
T1/T2 had different source/capacity34817 and are not reused as matched controls.
Both repetitions retain identical output hashes and draft/acceptance/lookup
counts before and after; candidate AR and MTP outputs also agree.

| Mode | Depth | Frozen batch6 tok/s | Candidate tok/s | Change |
|---|---:|---:|---:|---:|
| AR | 0 | 24.95 ±0.06 | 25.24 ±0.04 | +1.16% |
| AR | 32768 | 24.30 ±0.02 | 24.51 ±0.03 | +0.86% |
| MTP + lookup | 0 | 17.05 ±0.58 | 46.70 ±3.90 | +173.90% |
| MTP + lookup | 32768 | 22.23 ±0.01 | 79.31 ±0.94 | +256.77% |

Shallow MTP has material spread (46.70±3.90, only2repetitions): the large
improvement is clear, but neither precise steady-state speed nor the variance
cause is established. Deep MTP is the complete flow on the fixed repetitive
history:111accepted includes101lookup tokens and10neural drafts,17cycles.
It is not pure neural-MTP throughput or evidence that depth itself speeds decoding.
C2/4/6/8 correctness passed; their serving throughput was not measured here.

Image `ai-tools/gufo-eval:f3-tgmtp-9fc9a496-rocm101`; as-built `0ff1cb20-tgmtp-9fc9a496`.
Use `/opt/gufo-tgmtp/bin/gufo` (the image entrypoint); the inherited legacy
`/opt/gufo/src/build/release/gufo` is an older binary. Installed ELF SHA256
`f4dca2bca9efe4da63e8e1309f9b15dd31792c5460d5e27cff35ae19740578d5` matches the qualified mounted candidate.

Retain routed static row indexes and aligned HC320 vector loads. Both preserve
existing arithmetic; no policy/precision/cacheformat changes. All model/unit
checks above and final GPU health pass. Formatting758files and second review
pass. Exact raw results, runtime maps, build/link commands, frozen binaries,
operator modules and Nix intermediates are preserved under
`build/f3-tgmtp{1,2}` and `build/f3-tgmtp-profile`; do not rerun their controls.
The qualification receipt is `fn-tgmtp2-qualification-nix101` under ai.tools
`tmp/test-results/gufo-upstream-port-20261023/`.

[Qualification](archive/F3-TGMTP-QUALIFICATION.json), [screens](archive/F3-TGMTP-SCREENS.json),
[current TG/MTP profile](archive/F3-TGMTP-PROFILE.json), [full flow review](archive/F3-TGMTP-FLOW.json).
The initial profile belongs to frozenbatch6, not the optimized candidate.
Prefill is frozen at1435.59tok/s/pp2048; its2000target remains unmet. The user
has moved priority toTG/MTP. Do not restart prefill experiments automatically.
Future TG/MTP work must begin from these saved results, use small checks,
preserve current controls and batch long verification. Any new grouped>8 HC
or cost-model calibration remains a distinct future scope, not a measured win.
Implementation `3bdabdad` is fast-forwarded locally into all three worktrees;
harness plans are saved at ai.tools `a594c870`. The committed code/test patch
exactly matches the qualified image patch hash.
Root disk floor150GiB; disposable inactive browser caches and unused image
packaging cache were cleared, with both cleanup receipts retained. Main worktree `.dockerignore` is
pre-existing and untouched. No remote push or deployment.


## HC/MoE combined qualification, November 2, 2026

HC operand-load grouping and K5 Mode3 dead-subtile/K-tail staging are now
qualified together on pinned Nix ROCm10.1. Dense K7 K-tail removal is not
selected. Current image: `ai-tools/gufo-eval:f3-prefill6-f15434b2-rocm101`,
as-built `1dbee4d6-prefill6-f15434b2`.

| Case | Saved shared5 tok/s | Qualified tok/s | Change |
|---|---:|---:|---:|
| pp512 | 1068.58 ±3.09 | 1078.15 ±.40 | +.90% |
| pp2048 | 1429.13 ±1.13 | 1435.59 ±.07 | +.45% |
| pp8192 | 1460.97 ±.73 | 1466.18 ±1.12 | +.36% |
| pp2048 at d32768 | 1301.99 ±1.41 | 1311.98 ±1.03 | +.77% |

One fresh32760-record/64-head-replay FP16-KV collector is byte-exact to the
saved Nix baseline; affected full-FFN1025/shared-state and expert/final-tail
checks pass. Performance uses k4v4/temp0/seed314159/r2. The initial pp2048
1342.22±103.86 was inconclusive. A shortened-list replay inadvertently changed
context8193→4096 and is diagnostic only; the final d0 replay restored the
original512,2048,8192 list and8193 capacity. All attempts are retained; no
baseline, correctness or d32K rerun. Target2000 is still unmet.
[Qualification, runtime/code identities and all attempts](archive/F3-PREFILL-BATCH6-QUALIFICATION.json).

Retained this batch: HC grouped raw/SVH/shared reads preserve the original
restore rounding; K5 Mode3 skips wholly dead16-row B preparation and the
invariant K-tail check. Both are defaults, with no switches/workspace.
[HC screens](archive/F3-HC-LOAD-SCREENS.json), [MoE screens](archive/F3-MOE-STAGING-SCREENS.json).
The saved [MoE patch](archive/F3-MOE-STAGING-CANDIDATE.patch) is now applied; do not
apply it twice. Completed TG/MTP code is retained; this batch does not requalify those modes.

Dense K7 K-tail screening is exact but not selected: QKV+5.98% and mixed
smaller results against historical controls. Other inspected address/LDS
scheduling already exists. [ISA and all results](archive/F3-DENSE-KTAIL-SCREEN.json).
No repeat tile/cache/buffering/gate-load sweep is justified without a new
mechanism. A larger algorithm/schedule remains research, not a ready win.

The saved pre-batch6 shared5 profile remains the bottleneck map: MoE32.31%,
HC-family22.74% including routed combine, dense20.62%,99.676% kernel-active
(not ALU utilization). [Raw profile](archive/F3-CURRENT-PREFILL-PROFILE.json). It is
not a new profile of batch6. No current-batch full profile was repeated.
Earlier shared-MoE overlap/KVarN register selection stays retained; Q/KV overlap
and the previous rejected schedules remain rejected. [Stream qualification](archive/F3-STREAM-QUALIFICATION.json).

The batch6 build keeps all original Nix CMake objects/archives and links three
production replacements.74 GPU codeobjects are identical to shared5; exactly
three changed, with unchanged kernel-name sets/multiplicities. K5 cold/hot
103/148VGPR and HC96VGPR have no private/spills. Preserve intermediates,
baselines and rejected prototypes. Root free-space floor150GiB; unused Python
package and Docker packaging caches were reclaimed. The main folder's existing
untracked `.dockerignore` must remain untouched. No push was performed.

Current continuation: the concrete bounded batch is fully qualified. Reuse its
results; do not start another long test for each small idea. Future work needs
a distinct mechanism with small operator evidence first. Target2000 remains
unmet; no claim of exhausting all possible algorithms is made.

## Previous qualified matrix batch

Retained production changes: small-row K5 narrow tile, aligned singleton parent
output slices, and token-first traversal for singleton K7 6144→2560/R2048.
[Final qualification](archive/F3-PREFILL-GRID.md) records all performance and correctness
scopes. Batch512 collector32760 records/64 replays stayed exact; the final2048
path separately passed saved-output/guard checks through the actual release
object. Prior hc7 NLL drift+.006600 remains. All completed TG/MTP code is kept.

Image: `ai-tools/gufo-eval:f3-prefill3-67a50024-rocm101`, ID `31fd777e…`.
As-built source: `0b5d58a3-prefill3-67a50024`; patch SHA67a50024….
Integration commits are source-identity attestations, not newly built image
revisions. The source patch and installed hashes are in
[F3-PREFILL-GRID-QUALIFICATION.json](archive/F3-PREFILL-GRID-QUALIFICATION.json).

Qualified code commit: `b13b712a`. It was fast-forwarded locally into
`ai-tools/port-u4f` (the main gufo folder) and `ai-tools/integ`.
[Integration proof](archive/F3-INTEGRATION-BATCH3.json) confirms identical compiled
inputs and the tested source patch across all three worktrees. The main
folder's pre-existing untracked `.dockerignore` was preserved. This final
documentation-only integration record needs no rebuild or repeated model test.
No remote push was performed.

Keep `build/f3-nix-incremental/{release,units,batch2,batch3,before-batch2}` and
all frozen controls/receipts/images. New operator artifacts are in
`build/f3-prefill-{store,epilogue,wave64,predecode,grid,grid-rest,cold}` and
`build/f3-prod-gridcheck`. The old stage/partition controls remain dependencies.
Only unused unshared Docker packaging cache was deleted; Nix objects were not.
No new full-model profiling or baseline inference ran. gufo-00 has ended;
this chat coordinates machine use. Long verification remains batched.

## Earlier qualified state (superseded)

### Matrix follow-up screen

Latest work: [matrix follow-up screen](archive/F3-NEXT-MATRIX-SCREEN.md) completed two
short healthy operator windows. K32/workgroup and masked-wide variants were
mostly slower; tighter MoE launch bounds/descriptor partition had no aggregate
benefit. One narrow K5 indexer candidate is preserved for a later larger batch.
No production code or image changed, no full-model baseline was rerun, and the
qualified pp2048 result remains 1400.15 tok/s. All new artifacts stay under
f3-integ `build/f3-prefill-stage` and `build/f3-prefill-partition`. Do not retry
the rejected mechanisms unchanged. Production incremental objects and TG/MTP
work remain intact. This follow-up does not meet the >=2000 tok/s goal.

Qualified production status: the F3 batch is committed and locally integrated.
Code checkpoint: `2c159a12` on `ai-tools/f3-integ`.
Port integration merge: `9fcaffea`;
integ merge: `ffc0738a`. See
[F3-INTEGRATION.json](archive/F3-INTEGRATION.json) for the compiled-input identity proof.
All tracked production/test/toolchain inputs match the qualified checkpoint;
only documentation/guidance differ. No new build, GPU window or baseline
inference was needed for integration. No remote push was performed.

Retained: direct routing-ID combine, attention mask-pruning eligibility, and
grouped dense X3 scheduling. `fn-prefill-batch-nix101` is COMPLETE/health_PASS:
32,760 collected records and 64 head checks match saved Nix q6 exactly.
Prefill versus saved pre-batch Nix T1: pp512 +2.14%, pp2048 +2.04%, pp8192
+2.18%, d32K pp2048 +1.86% (one candidate r2 window). These are batch-level
observations, with prior hc7-relative quality drift still recorded.

The tested image remains `ai-tools/gufo-eval:f3-prefill-batch-f712b2d0-rocm101`
(image `93197735...`, as-built source patch `f712b2d0...`). Integration commits
attest source identity; they are not newly built image revisions. Preserve
`~/projects/gufo-f3-integ/build/f3-nix-incremental/{release,units}`, the operator
objects and all baseline/candidate artifacts for incremental work.

Performance-neutral simplifications/correctness remain eligible for retention.
Wider and persistent matrix prototypes added complexity without a qualified
benefit; their source/binaries are preserved. Previously completed TG/MTP work
is kept. Further prefill work remains the priority; the >=2,000 tok/s goal is
not met. TG/MTP expansion, deferred serving/27B qualification and deployment
are separate remaining work, not implied complete by this integration.
gufo-00 has ended; this chat coordinates machine use. Long validations stay
batched, and stored matching baselines are reused.

## Historical resume record (superseded by the latest status above)

Resume update (2026-11-01): T1 probe completed; shared indexer Q/K projection
reuse is implemented and independently reviewed in the uncommitted f3-integ
worktree. FP16-KV/KVarN collector bytes match hc7; focused Nix ROCm 10.1 unit
passes. Candidate images are `ai-tools/gufo-eval:f3-indexer-b03a1f1e-rocm101`
and `-tests-rocm101`. Full evidence and runtime caveats are in
[archive/F3-RESUME-RESULTS.md](archive/F3-RESUME-RESULTS.md).
The user requires Nix ROCm 10.1, one combined final model/timing pass, and
reuse of stored matching baselines. No merge or final model qualification yet.
The first q3 batch item's Thai rare-vocabulary fixture failure is resolved:
raw repetition produced identical AR/MTP English reasoning, so the fixture
now explicitly continues a non-thinking assistant's requested repetitions.
All original coverage/exactness gates remain strict. Focused lifecycle and
sampled-head checks PASS on EXL3 and GGUF (`fn-draft-fixture-nix101`, health_PASS).
The q3 resume scope is now complete: reuse the original passed batch prefix
plus the repaired fixture tail; sampling, prefill and execution32k passed in
`fn-final-q3-remaining-nix101`. The EXL3 MTP oracle needed storage-boundary
and full-head repairs; independently reviewed, original tolerances unchanged.
Its final matched-input audit passed in `fn-final-q3-mtpinput-nix101`
(maximum carry relative RMS 4.71e-5, limit 5e-5). That window also completed
the three-token CPU/GPU floor diagnostic (604.27 s, top-1 3/3, finite KL up
to 0.00175); this is not a full-quality gate. health_PASS, actual loaded
libraries all Nix ROCm 10.1. Do not repeat this expensive CPU probe routinely.
Final test overlay: `ai-tools/gufo-eval:f3-indexer-mtpinput-rocm101`.
Evidence index: `fn-final-q3-mtpinput-nix101/q3-summary.json`; details and
hashes are in the F3 resume results. q6, Nix FN/27B timing, performance goals
and formal integration remain. Next q6 must use the Nix collector and reuse
saved BF16/UD results with their recorded provenance (the saved UD quality
metadata says 7.2.4 despite an old plan note calling it Nix; do not conflate
that with the separate Nix timing baselines).
q6 resume update: `fn-final-q6-nix101` completed32760 rows/64 head checks,
health_PASS, one candidate collection with stored BF16/UD/hc7 data reused.
KL2560.026481 and top1 92.0543% remain better than historical UD; versus hc7,
NLL increases0.006600 (window95%CI [0.001729,0.013902]), about0.66% perplexity.
This warning remains recorded; q6 is measured with documented drift.
The user directs prioritizing performance over eliminating small quality
changes. No precision patch was applied. Stop further quantizer diagnosis
and proceed to current-candidate Nix timing; see Q6-SUMMARY.json and the F3
resume results for tails, provenance and the isolated compiler diagnostic.
Latest user direction: **prefill is the highest priority**. Defer TG/MTP
optimization and T4. Completed pure-Nix T1 AR prefill baselines are pp512
1039.61, pp2048 1372.10, pp8192 1411.52 tok/s; d32K pp2048 1257.46.
Reuse `fn-final-t1-x-ar-nix101`; do not repeat it. T2 is also saved.
T3 worktree wrapper completed K17, then D was intentionally stopped;
`fn-final-t3-worktree-nix101/USER-PRIORITY-STOP.json`, health_PASS. Do not
interpret that cancellation as a failing candidate. Historical UD timing
images also inherited /opt/rocm/lib, so they are not proven matched controls.
An unqualified prefill candidate now replaces repeated MoE bucket searches
with retained original routing IDs; focused exactness tests and one combined
candidate validation/prefill timing pass are next. See F3 resume results.
The original hold-point inventory below remains historical context.

This lists what has **not** been done, what is **still unexplained**, and the
order to resume. Read [docs/GFX1151-KNOWHOW.md](../../docs/GFX1151-KNOWHOW.md)
first. It records what was already measured, so do not re-run that. It is a
living document: update it with every new result. History is in
[F3-PLAN.md](archive/F3-PLAN.md) under "Status at hold", in
[F1-RESULTS.md](archive/F1-RESULTS.md) and in [F3-SESSION-FINDINGS.md](archive/F3-SESSION-FINDINGS.md).

## Where things are

| What | Where |
|---|---|
| Newest code (all F3 work, Flash-Next EXL3 prefill) | `ai-tools/f3-integ` @ `966f84b8`, worktree `~/projects/gufo-f3-integ`. Not merged into `ai-tools/port-u4f` or `ai-tools/integ`. |
| Last final images | `ai-tools/gufo-eval:final-68d4e3c8-rocm101` and `-tests-rocm101`. They predate all F3 speed work. |
| Plans and LOCKS | `~/projects/ai.tools/test/experiments/gufo_upstream_port_20261023/` (PLAN-FINAL-FN2*.json, PLAN-FN-*.json, PLAN-FINAL-27B.json) |
| Receipts | `~/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/` (final-fn2/, fn-713c, fn-434f, fn-hc2…hc9, fn-att1/2, gdn/) |
| UD-Q4 timing baselines | Measured once on 68d4e3c8 with nix: T1 probe AR, T2-q MTP, T3-q card. Retain as historical evidence; internal GGUF payloads now removed, external originals preserved. |
| Packs | 5.05: internal `ai.tools/storage/models/llm/flash-next-exl3-5.05-h6-ng6/`. 4.05: T5 EVO USB. Kyojin (≈4.0 bpw): internal `ai.tools/storage/models/llm/flash-next-exl3-kyojin/`; earlier USB load observations are historical. |
| Dropped branch, kept as record | `ai-tools/fn-hc-int8` (HC int8 decode: +3.5 % tg, failed the gate) |

## Current numbers (docker 7.2.4 RelWithDebInfo probes, 5.05 + KVarN)

| | Start (68d4) | Now (hc7/att2) | UD-Q4 (nix) | Target |
|---|---:|---:|---:|---:|
| pp2048 | 454 | ≈1,255–1,260 | 1,762 | ≥2,000 |
| pp8192 | 466 | ≈1,290–1,324 | 1,796 | ≥2,000 |
| tg128 | 24.7 | ≈24.7 | 29.2 | at least Kyojin |
| Quality KL vs BF16 | 0.0269 | 0.0264 | 0.0392 | better than UD (met) |

The docker build runs about 10 % below nix: UD measured 1,567 under the
profiler vs 1,762 on nix. So EXL3 on nix is probably about 1,390–1,450 at
pp2048. That is an estimate and has not been measured.

## Not done yet

1. **No end-to-end measurement of `966f84b8`.** These have never been measured
   together: the 8x1 bucket tile (df1f7409), the hot-expert 128-row buckets
   (56202401), the PrefillCombine wrapper (8c9fb296) and the attention fusion
   (2e054760).
2. **Branches not merged.** `f3-integ` is not merged into port-u4f/integ, so no
   final nix images exist for it.
3. **Combined final pass never run** (one pass, with integration and timing
   together, per AGENTS.md "Iteration workflow"):
   - **q3**, the EXL3 session parity run: `--batch-only`, `--prefill-only`,
     execution-32k, MTP audit and engine floor. The parity bugs were fixed and
     pass the unit tests, but q3 itself was never rerun at model level.
   - **q6** (full-vocabulary prefill vs HF BF16, 32,760 rows) on the final
     build. Last pass: hc7 8737db4e, KL 0.02643.
   - **Flash-Next timing on nix:** EXL3 arms only, against the stored UD
     baselines (T1-x AR, T2-x MTP, T3-x card, T4 KVarN cost). There is still
     no measured MTP/card comparison against Kyojin's card.
   - **27B final timing:** the PLAN-FINAL-27B final-t-* windows have never run.
     Run them on the final image after a 27B fingerprint check, because W3 also
     changed shared EXL3 kernels.
4. **Decision pending: the default Flash-Next pack.** 5.05 is recommended
   (KL 0.0264 vs UD 0.0392). 4.05 is about UD quality (KL 0.0424) and +7 % tg.
   Kyojin's pack is worse (KL 0.0598).
5. **Decode not worked on since D2.** tg is about 15 % below UD. Routed
   experts run at 141–158 GB/s and HC plain F16 at 126–165 GB/s, against
   205–221 GB/s for the trunk GEMV.
6. **KVarN serving has no prefix cache.** The cache-reuse asserts are
   expected-fail (q5/q5b); adding the cache is an F3 follow-up.
7. **Know-how doc commit:** docs/GFX1151-KNOWHOW.md, F3-SESSION-FINDINGS.md,
   the SKILL.md pointer and the AGENTS.md line are waiting for the cross-review
   (Codex thread `knowhow-review`) and a commit.

## Still unexplained (no root cause found)

| Symptom | Numbers | What is known / why it stopped |
|---|---|---|
| Routed bucket GEMM issue stalls | 3.2–3.4 ms/call vs ≈2.05 ms floor (≈1.2 ms stalls) | Ping-pong, stagger, sync removal and hoisting all failed. This ROCm build has no SQ stall counters. |
| Dense X3 efficiency | 31–36 TFLOPS, about 61 % of peak | Not fragment reads (a 0.5-read tile gives no gain). Suspects: LDS staging and syncs in 32-wave blocks, plus serialized decode. Not broken down. |
| kWideHalf 128×256 tile (n % 256 ≠ 0) | 20–29 vs 31–36 TFLOPS | Used by Flash-Next shared-expert gate/up and the combined indexer (n=640, rows > 64); no 27B projection uses it. The cause is not measured. The estimated impact is small (≈5–10 ms/pass); confirm call counts first. |
| HC down in situ | 1.78 ms/call vs 0.77 ms warm bench | The dirty-cache effect is confirmed. The gap left after non-temporal producer stores is not explained. |
| HC int8 decode gain | +3.5 % tg vs ≈7–10 % expected | Not investigated; the experiment was dropped at the user's 10 % bar. |
| Fused MoE combine (HcCombine kMoe) | 2.3 ms/call | Suspected slot-gather latency (2.5 segments per wave); not measured. |
| First-request kernel gaps | 27/11/11 ms | Suspected lazy code-object loading; affects TTFT only. Not measured. |
| MTP audit tolerance (q3, first run) | rms ≈2e-4, over the oracle tolerance | It failed together with the parity bugs and was never rerun after the fixes. |

## Open leads not tried (from the code sweep; see the doc's open-leads section)

- Sparse prefill computes the combined indexer projection twice; reuse it. This
  is a cheap, sure prefill win.
- Combine repeats expert-ID searches.
- A tested scheduling hint (iglp) is excluded from grouped X3.
- The 27B crossover threshold is still provisional.
- Check the remaining GGUF dense-core assumptions; revisit the EXL3-specific
  MTP costs; inventory the standalone stable fallbacks still reachable.
- A masked 256×256 tile in place of kWideHalf (see above).
- A persistent/stream-K bucket schedule and larger prefill chunks. Both pay
  only once routed is weight-bound, which it is not yet.
- Code cleanups flagged: stale dispatch comments, the ISA help path, bench vs
  production no-SLP flags, and profiler classification of EXL3 kernels.

## Historical resume order (superseded by current continuation above)

1. T1 probe on `966f84b8` (EXL3 only, docker binary; label it as a probe).
2. Fix the double indexer projection; gate it with the collector byte check.
3. Get the default pack decided, then merge `f3-integ` into port-u4f/integ and
   build the nix final images (X-A's recipe, label = merge commit).
4. Run the combined final pass once (q3 incl. the MTP audit, q6, FN timing,
   27B timing). Diagnose failures, fix, and rerun only the affected checks.
5. Then decode (routed and HC GB/s), and the unexplained items above if
   prefill is still short of the target.

## Rules that still apply

- GPU windows one at a time under `ABBA-OWNER.lock`. gufo-00 has ended; this chat coordinates builds,
  hashing and large writes. No builds during timing windows.
- Root free space must be ≥150 GiB.
- Commit your own files only, as one-line Conventional Commits, and do not push.
- Bit changes need the q6 gate; chunk invariance is mandatory. The collector
  `.bin` byte check is authoritative for "bit-exact".
- Timing: measure each baseline once and reuse it; one run per arm; repeat
  only results inside 5 %·10 ms.
- Iterate as implement → second-reviewer review → fix. No per-item model
  windows; one combined pass at the end.

## Local repository cleanup, 2026-11-07

- Current development branch remains `ai-tools/port-u4f`; `main` is the only
  other local branch. Eleven inactive worktrees and22 historical local branch
  names were removed after verifying their tips (including detached HEADs)
  in a complete Git bundle. This does not mean all experimental branches were
  merged. No remote branch or tag was changed.
- Recovery bundle and before/after receipts:
  `/home/m2on/projects/ai.tools/tmp/gufo-cleanup-20261107/`.
  `BEFORE.json` maps old names to commits; `branches-before.bundle` retains
  their history. Recover an old branch with
  `git fetch <bundle-path> refs/heads/<old-name>:refs/heads/<restored-name>`.
  `CLEANUP.json` maps relocated build directories and `ASSET-MOVES.json`
  locates the unrelated cyclist HTML/images/browser inspection artifacts.
- Only retired compilation objects were deleted (6564 files,11.67GiB logical).
  Current release/CPU/GPU and f3 incremental builds, saved controls, runtimes,
  model files and measurement receipts remain. Source snapshots and final
  binaries are retained; historical build directories may need recompilation.
- Validated source/test changes are committed as `f24b10ea`; rejected wide
  MoE and fused gate/up experiments remain isolated and are not deployed.
  The live service still uses its qualified original `6edf0d44` ELF.
- The old root `issue.md` is preserved as
  [the November4 compatibility investigation](archive/VEMLOIR-COMPATIBILITY-20261104.md).
  Conversion-plan headings now distinguish completed c03 service from the
  historical conversion instructions. Cleanup did not restart the server.
