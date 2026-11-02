# Same-pack TG/MTP attribution, November 2, 2026

With the same Kyojin pack and neutral greedy sampling, Gufo's observed MTP
rates are close to the saved local Kyojin run. **Gufo AR/MTP text agrees on
all three prompts**; the saved Kyojin run fails its own token/text equality
check on all three. No inference kernel, precision or selected model changed.
This work establishes a useful comparison and current optimization priorities,
not a new speedup or full quality qualification.

| Case | Gufo AR | Kyojin AR | Gufo MTP | Kyojin MTP | Gufo MTP difference |
|---|---:|---:|---:|---:|---:|
| Prose | 30.49 | 31.30 | 42.55 | 42.52 | +0.08% |
| Chat | 30.51 | 31.33 | 46.94 | 49.44 | −5.06% |
| Code | 30.23 | 31.28 | 62.66 | 63.22 | −0.89% |

Client tok/s = `(completion_tokens − 1) / (last − first useful SSE time)`.
C1, context16384,256 output tokens, prompt counts25/33/33, temperature0,
seed314159, thinking off, no prompt-cache hits. One measured pass per case;
these are descriptive differences, not repeated-run confidence estimates.
Gufo AR is2.58–3.36% lower. Do not spend another long control run resolving
these small differences without a concrete implementation change.

The saved Kyojin result and all old Gufo baselines were reused. Gufo requires
separate server selections for AR/MTP, so two sequential loads (26.05/29.02s)
were used, with a warm24-token budget before each mode. The warm prompt stopped
after2tokens; it does not fully warm every later verification shape.
Gufo binary remains native `b4ebe27c`, pinned Nix ROCm10.1. GPU health PASS.
Raw output token IDs are unavailable on this Gufo HTTP endpoint; the checked
property is **text equality**, not independently observed token-ID equality.

## What remains different

Same on-disk weights do not mean identical effective arithmetic:

- Kyojin folds normalization and uses group-int8 HC down/up plus the sidecar
  (`hyperconnections.py:590–623`); Gufo decode keeps F16 HC operands
  (`src/models/qwen38_flash_next/kernels/rocm/device_model.cpp:362`).
- Kyojin's `draft2s` uses a compressed preview and top16 exact candidates;
  Gufo uses its subset vocabulary and confidence policy. Approximate drafting
  does not itself justify different committed text with a correct verifier.
- Kyojin quant K4V4 and Gufo KVarN k4v4 are different KV codecs. Kyojin's pinned
  ROCm10.1 nightly wheel runtime/clang23 differs from Gufo Nix10.1/clang24.
- All three cross-engine outputs differ, including AR. Acceptance/cycle costs
  therefore are not matched identical-output or pure-engine measurements.
  Prompt counts and request messages match; Kyojin prompt token IDs were not
  captured, so count matching is not a token-ID framing proof.

Gufo proposed249/252/248 drafts and accepted135/153/194, with120/103/62cycles
(2.13/2.49/4.13 output tokens per cycle). Kyojin proposed243/221/230 and accepted
138/160/194; it does not expose a comparable cycle count in this receipt.
Do not infer a per-cycle Kyojin cost from acceptance alone.

The previous historical Gufo5.05 weights with old IQ4 PLE versus Kyojin≈4bpw
comparison mixed weight precision, PLE and engine settings. Its larger gap is not evidence that Gufo's engine
itself is20% slower. This same-pack comparison substantially narrows that gap,
without isolating the share attributable to each remaining difference.
Native5.05 remains selected; no quality-driven model switch was requested.

## Corrected setup attempt

The first partial Gufo attempt omitted penalties, inheriting default
presence_penalty1.5. Kyojin's `serve.py:1027` uses ArgmaxSampler at temperature0
with no penalties. Gufo applies penalties before its greedy branch
(`src/core/sampling.cpp:394`), so this was a real comparison error.
The attempt was stopped during MTP loading; its three AR results and explicit
STOP-REASON remain recorded but are excluded from the comparison above.
Only the corrected Gufo comparison was rerun with repeat_penalty1,
presence0,frequency0,topk1,topp1,minp0. Stopped and completed windows both
passed GPU health. The stop's exit137 is intentional, not a GPU crash.

## Current native5.05 bottleneck and next batch

The previous TG/MTP profile predates removal of verifier LDS promotion and is
not a current bottleneck control. A short current trace used native5.05,
real prose64 afterwarm16, neutral greedy sampling, C1/context16384/KVarN4/4.
The existing qualified Nix tracer was reused without rebuilding.
This is diagnostic work, not another unprofiled speed baseline.

| Kernel family | AR share of decode span | MTP share |
|---|---:|---:|
| Dense EXL3 | 35.44% | 22.21% |
| Routed experts | 20.39% | 27.43% |
| Stable small projections, including HC-down | 14.82% | 9.88% |
| HC-up K320 | 7.20% | 5.12% |
| Draft subset head | — | 2.77% |

AR kernel span2579.785ms, kernel-active90.13%; MTP2064.302ms,78.24%.
MTP generated64tokens in28cycles with63drafts/35accepted; paired text agrees.
All requests and GPU health passed. No full quality/cold/deep/concurrent claim.
Selected HIP APIs include19,141direct launches and4allocations in MTP. Possible
first-use/capture costs are not separated from expected eager work in this trace. The non-kernel span is **not** a measured
CPU bottleneck or removable-overhead estimate. Do not publish profiled rates
as production throughput. Graph replay dispatch names are retained; absent
launch dimensions are not inferred.

GPU range selection uses actual Embed sequences, not an assumed shared clock
with Python. HTTP prefill25 splits into18+7. MTP prefix sequence18,17,1,7,6
includes proactive n−1-row draft KV and catch-up between chunks; the following
1-row Embed begins timed decode. AR starts after18,7. This source-backed
boundary gives decode spans within1ms of the independent server timer.

The next candidate batch, preserving the existing arithmetic and prefill:

1. Dense K7 decode: inspect emitted load/reuse scheduling for the actual
   `Exl3GemvKernel<7,1,4,2>` / `<7,1,8,2>` groups. Together these account for
   613.94ms of the2579.79ms AR trace. Reuse the prior failed prefetch/tile ledger;
   only screen a distinct mechanism with an actual register/load reason.
2. Routed K5 verification at3/4rows: gate/up are the top MTP kernels, with
   125.29/164.55ms respectively. Screen ragged-route weight reuse or load
   scheduling with both shared/disjoint experts and preserved accumulation.
3. Separate graph first-use from steady replay using the saved trace before
   proposing scheduler changes. Source already overlaps target graph recording
   with its eager execution; do not repeat that optimization.

The draft subset head is2.77% of this MTP span; replacing it first would chase a
small component while changing candidate selection. Old HC group-int8 failed
its quality gate; no silent reintroduction. Fixed Q4/Q8 cost curves exist, but
C1 greedy switches to confidence stopping after initial observation
(`engine.cpp:1077`), so blindly retuning those curves is not an established fix.
Prepare distinct small operator candidates together, then one combined model
qualification after useful evidence. No unqualified production patch was added.

Kyojin mismatch source audit found first differences at total positions82/109/182,
before a256-token KV page transition or sparse QSA. R1/multirow arithmetic and
rejection-state restoration remain hypotheses. A saved-prefix same-state
sequential-versus-verify check, followed by one forced rejection, is the focused
future diagnostic; no cause is claimed from this timing experiment.

[Same-pack result and provenance](F3-KYOJIN-SAME-PACK.json) ·
[Current profile and kernel totals](F3-NATIVE-PROSE-PROFILE.json).
Infrastructure checks:432passed. Independent code/receipt review passed.
Raw ai.tools receipts: `gufo-kyojin-pack-nix101` (stopped setup),
`gufo-kyojin-pack-neutral-nix101`, `fn-native-prose-profile-nix101` under
`tmp/test-results/gufo-upstream-port-20261023/`.
