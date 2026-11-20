# Kyojin and exllamav3 survey for Gufo (2026-11-20)

Sources (read only, no code run):

- Kyojin `656dab5..b2f0d5f`: 57 commits, 2026-11-08 to 11-17. Clone: `scratchpad/ref/kyojin`.
- exllamav3 `master` (v1.6.7, 11-13) plus `origin/dev` (13 more commits, up to `f001dea` on 11-19), from
  2026-09-20 on: 483 commits, 127 of them touch `exllamav3_ext`. Clone: `scratchpad/ref/exllamav3`.
- Gufo branch `ai-tools/port-u4f`. Checked against `plan/upstream-sync/PLAN.md`, `docs/GFX1151-KNOWHOW.md`,
  `docs/models/qwen3.8-flash-next/EXPERIMENTS.md`, `benchmarks/qwen3.8-27b/README.md` (in git history) and `src/`.

Headline: neither project has a new EXL3 GEMV/GEMM kernel change that Gufo can port.

- **exllamav3.** Nothing under `exllamav3_ext/rocm/` has changed since v1.6.5 (`cdc785d`, 10-25). That is the
  version Gufo's `src/core/hip/detail/exl3_gemv.hpp` cites as its reference. The later GEMV and GEMM work is
  CUDA only: `ff88a8d`, `b85c615` and `3bbc777`.
- **Kyojin.** Most of its gfx1151 work in this range was ported *from* Gufo: the routed F16 WMMA MoE, the
  MoeEpilogueVec4 epilogue, the routed MMQ, and the 4-row union sparse WMMA attention.

What is left is in the serving and speculation layer and in GDN state storage.

## Ranked candidates (benefit / effort)

### 1. Bring typical acceptance back, with Kyojin's loop fixes

- **Source:** Kyojin `9f1b368` (#177, 11-08) and `2484e31` (#183, 11-09).
  - `9f1b368` keeps copied (lookup) drafts exact by default. It also refuses a relaxed acceptance when the
    token would complete a 12-gram already seen at least twice in the last 2048 generated tokens. After such
    a loop signal it switches to exact acceptance for an adaptive window of 512 tokens, doubled on retrigger
    and capped at 16384. It adds an optional entropy gate `HMAX` and an optional rewind of the thinking text
    to the first copy of the loop.
  - `2484e31` adds `EXL3_SPEC_TYPICAL_PHASE=answer`: relaxed acceptance only after `</think>` closes, exact
    acceptance inside the thinking block.
- **Category:** speed, made safe for quality.
- **Reported gain:**
  - Gufo measured its own port (`da021c03`) in U5: 18 of 18 sampled requests were faster, by +2.5% to
    +27.6%. Chat median about +21%, code about +17% (PLAN.md line 189).
  - Kyojin's original change, which the U4-4 plan row cites, reported about +7%.
  - The loop fixes themselves cost nothing on greedy requests.
- **In Gufo:** No longer. KNOWHOW "Flash-Next quality versus Kyojin, November 10" says typical acceptance
  caused the thinking loops (3 of 4 seeds looped at xhigh), and it was removed (D10). No loop guard exists
  in `src/`.
  - These Kyojin commits are new evidence for exactly that failure mode. Copied drafts and in-thinking
    drafts are the sources that fed the loop.
  - Gufo's removed port applied the relaxed rule to lookup copies too; Kyojin now keeps those exact.
- **Where:**
  - Revert the D10 removal, which brings back `da021c03`: `mtp_sampling.hpp`, the sampled verify in
    `runtime/executor_mtp.cpp`, `src/cli/serve/sampling_request.hpp` and the shared sampling option.
  - Add a source gate: MTP drafts only.
  - Add a phase gate. Use the shared Qwen control-token constants for `<think>` and `</think>`.
  - Add a per-request 12-gram repetition tracker and a cooldown window.
- **Effort:** low to medium. The core code is in git history. The guards are about 150 lines of host C++
  with CPU unit tests.
- **Risk:** quality only. Treat it as opt-in until it passes the 11-10 lodging-review repro: xhigh, seeds
  1–4, plus the text-only xhigh case. The output distribution is not preserved, which is already known
  and accepted for an opt-in.

### 2. Streaming loop detection: stop on loop, or close an open think block

- **Source:**
  - exllamav3 `805caee` (10-29): `exllamav3/generator/loop_detect.py`, used by `Job(stop_on_loop=...)`,
    with `a093206` tests. It hashes rolling k-grams (k=6) and tracks candidate periods with ±5% drift.
    A loop needs 8 repetitions over 8192 tokens, or 4 exact repetitions over 1024 tokens. It exempts text
    that is copying prompt items (Bloom filter) and handles low-diversity text. Cost is O(1) per token,
    about 3.5 MiB.
  - Kyojin `9f1b368` `tools/loopguard.py`: inside thinking, a non-fenced line of at least 40 characters
    seen 3 times is a loop. The guard then closes the think block, and it forces a close when EOS arrives
    inside an open think block.
- **Category:** quality and bug mitigation. It also saves wasted GPU time on runaway replies.
- **Reported gain:** none in tokens/s.
  - KNOWHOW records that thinking at xhigh text-only finished in 0 of 4 runs for Gufo and 1 of 4 for
    Kyojin within 6000 tokens.
  - The 27B greedy loops come from the model itself ("구자철이 아니라…").
- **In Gufo:** no. `grep -ri 'loop guard|loopguard|loop detect' src` finds nothing.
- **Where:** the token stream in `src/cli/serve/text_generation_scheduler.cpp`, or the per-request
  generation loop.
  - New finish reason `loop_detected`. Optionally close the think block on a loop that starts inside
    thinking.
  - It is model independent, so Flash-Next and 27B share it.
- **Effort:** medium. Port the detector (about 500 lines of Python to about 300 lines of C++) plus CPU
  tests. No GPU work.
- **Risk:** false positives on legitimately repetitive code or JSON. Keep it opt-in per request or server
  option at first, with upstream's conservative defaults. Replay `tests/functional/pi_agent.py` coding
  loops.

### 3. Frequency-ranked dynamic draft vocabulary for the MTP head

- **Source:** exllamav3 `47d42fe` (10-24), `exllamav3/modules/sparse_head.py` and `sparse_head.cu`.
  - It keeps an F16 copy of the lm-head columns for the most frequent N tokens, ranked by running counts
    with decay. Prompt tokens are weighted 0.2.
  - It rebuilds every 128 tokens. A token replaces one already in the subset only if its count is 25%
    higher.
  - The subset is used only while the measured attributable miss rate stays below the break-even
    `2 * saving / round_time`. Every 128 rounds it probes the full head.
  - Counts persist across runs. The argmax returns the winner's embedding on the device, so the MTP chain
    never goes through the host.
  - It is opt-in (`--sparse_draft_head N`), and upstream published no gain number.
- **Category:** speed. It changes only what the draft proposes; the target always verifies over the full
  vocabulary.
- **Reported gain:** none upstream.
  - Gufo's own numbers bound it. The static 98,304-row subset makes C1 w8 cost 80.1 ms against 90.2 ms for
    the full head (KNOWHOW, 11-20 cost audit). That is about 1.4 ms per drafted position for about 150K
    rows dropped.
  - Going from 98K to about 16–32K rows would save very roughly 3–5 ms per w8 cycle, about 4–6% of greedy
    and sampled C1 MTP. This is an estimate, not a measurement.
  - Larger if the subset is extended to the C2+ batch paths, which today use the full-head table.
- **In Gufo:** partly. `draft_vocab.hpp` has a static prefix of 98,304 rows plus a per-request table of
  1,024 rows with fallback. There is no frequency ranking, no adaptive size and no miss-rate switch.
- **Where:**
  - `draft_vocab.hpp` and the draft head GEMV in `kernels/rocm/mtp.hip.cpp`.
  - Re-run the cost table in `mtp_costs.hpp` with `qwen38_flash_next_gpu_probe --cost-audit`.
  - The subset needs a gathered row buffer, not a prefix: about 2.7 KB per row, so 32K rows is about
    87 MB.
- **Effort:** medium. Row gather and rebuild, a device-side remap from subset index to token id, and
  counters. Gufo's per-request table already does the remap for 1,024 rows.
- **Risk:** low for correctness, because the target verifies. The real risk is lost acceptance when
  coverage drops on code, CJK or tools, which the miss-rate switch is meant to catch. Seeded sampled
  replay must keep the subset deterministic per request: keep the counts per session, or freeze them per
  request.

### 4. GDN recurrent state stored in FP16 (FP32 compute, one rounding per pass)

- **Source:**
  - exllamav3 `0244652` (10-13): FP16 by default for GDN/KDA state. Compute stays FP32 and the state is
    rounded once per pass on store. `--recurrent_state_fp32` opts out.
  - Kyojin `fce7501` (#215, 11-15): BF16 storage, CUDA only, measured.
- **Category:** speed and memory.
- **Reported gain:**
  - Kyojin, RTX 3090, Qwen3.8-27B EXL3, 8 × 16K decode: 232.0 to 249.9 tok/s (+7.7%). Text after 32K:
    PPL −0.21%, KL 0.0004.
  - Gufo estimate for Flash-Next at C1: GDN decode costs 1.16 ms per token (B2 profile, about 3.6% of AR).
    Halving the state traffic gives about +1.5–2% AR at C1, and roughly 2× that at C4, because state
    traffic grows with sessions while the weight stream is shared.
  - State drops from about 108 to 54 MiB per session: 36 GDN layers × 48 heads × 128² values. Component
    cache and checkpoint payloads shrink the same way, so restores are faster and more checkpoints fit in
    RAM.
- **In Gufo:** no. Flash-Next `recurrent.hip.cpp` uses `float* state` throughout.
  - The 27B path has a BF16 option, `GUFO_QWEN_RECURRENT_STATE=bf16` (`686a0928`, #138, Aug 28). It was
    rejected: full-logit RMSE 0.1126 widened to 0.1396, and a 200-token greedy reply diverged after 824
    bytes. The rejection note says to revisit "only with a different update/storage formulation".
  - FP16 storage is that different formulation, with 3 more mantissa bits than BF16, i.e. an 8 times finer
    step. The current user rule "speed over bit-exactness, quality pass decides" also removes the greedy
    divergence test as a blocker.
- **Where:**
  - Flash-Next: `kernels/rocm/recurrent.hip.cpp`, covering the decode, verify and replay kernels,
    `RestoreGdnStateKernel` and `GdnRowSplitKernel` prefill.
  - The snapshot and continuation layouts get a new layout version, in `continuation_layout.cpp` and the
    component cache.
  - 27B: add an FP16 `QwenRecurrentStateStorage` value next to `kBf16` in `src/models/qwen/hip/`, reusing
    the #138 plumbing.
- **Effort:** medium to high on Flash-Next because of the many state consumers and the cache format bump;
  low to medium on 27B, where the templates exist.
- **Risk:**
  - Quality: needs a KL check and a long-context greedy and sampled check at 128K+.
  - FP16 overflow: the decay gates keep the state bounded, and upstream made it the default for all
    GDN/KDA models.
  - Breaks the old cache ABI: old snapshots must be rejected cleanly.
  - Run 27B first; it is the cheaper place to qualify the arithmetic.

### 5. Request metrics: first-token time, cancellations, evictions, finish reasons (small)

- **Source:** Kyojin `2495025` (#231) and `d5fe8f6` (#227).
- **Category:** operations.
- **In Gufo:** partly. `src/cli/serve/generation_metrics.hpp` logs the finish reason per request. Check
  whether `/metrics` exports TTFT histograms, cancellations and cache evictions.
- **Effort:** low. **Risk:** none.

## Checked and already in Gufo (do not port)

| Source | Item | Gufo evidence |
|---|---|---|
| exl3 `494edd2` | FP32 router logits for 512-expert top-k (`qwen4_exp`) | `executor.cpp:1258` `s.router = f32(...)`; `RouterTopK(const float*)` |
| exl3 `63efe98` | FP32 QSA indexer scores through top-k | `attention_select.hip.cpp` `SelectScoreKernel(..., float* scores)` |
| exl3 `e6871da` | Accurate sin/cos range reduction for RoPE under fast-math | Flash-Next builds with `-ffast-math`, but the gfx1151 ISA of `attention_prep.hip.cpp.o` (build/b1) has no `v_sin_f32` or `v_cos_f32`: OCML does the full reduction |
| exl3 `7b3e689` | top-p includes the token that crosses the threshold | `src/core/sampling.cpp:147` |
| exl3 `85cae3e` | `draft_max_rows` row ceiling per batch | `mtp_policy.hpp` `MtpBatchController` plus the measured `mtp_costs.hpp` table |
| exl3 `d33d186` | Multi-row GDN b/a projection in one pass | narrow projections go through one GEMM (`kernels.hpp:447`) |
| exl3 `3a8e2fb`, `c1136a0` | GDN rewind by replay, not a snapshot | B2 E6 (`9cc91779`) |
| exl3 `47d42fe` (on-device chain) | MTP chain without a host round trip | graph-captured MTP decode |
| Kyojin `69f2360`, `5aedcfc`; exl3 `2ece09d` | 0.4 s decode-priority prefill chunks | Gufo decode-credit scheduler; smaller active-decode chunks rejected 11-18 (KNOWHOW line 633) |
| Kyojin `59daecd` | `GPU_MAX_HW_QUEUES` workaround for multi-process NaN | `docs/SERVER.md:389` already manages `GPU_MAX_HW_QUEUES` |
| Kyojin `1f34177`, `be2f283`, `b55e3e5`, `21c1e1f`, `7316d6b` | routed F16 WMMA MoE, epilogue, routed MMQ, union sparse WMMA attention | ported from Gufo (see Kyojin `THIRD_PARTY_NOTICES.md`) |
| Kyojin `9a466cb` | `__vcmpne4` fallback sign bug (IQ2_XXS) | Gufo's DS4 `__vcmpne4` (`ds4_rocm_device.hip.hpp:9`) ORs all 8 bits of each byte, so it does not have the bug |
| exl3 `2b5223f` | Off-by-one in rep-pen decay range | exllamav3-specific sustain/decay ranges; Gufo uses `repeat_last_n` |

## Rejected before, or not applicable

- **Rejected before:** Kyojin `28fe094` (blocked-GQA pairing in the HF chunk GDN kernel) and the gfx1151
  GDN parts of `eee77f7`. The U4-2 rejection of the F16 chunked GDN still holds: F16 operand error and a
  result that depends on where the pass splits.
- **CUDA only:**
  - exl3 `ff88a8d` and `b85c615`: ptxas register bound, two blocks per SM.
  - exl3 `3bbc777`: 32/48/64-row GEMM tiles. Gufo U4-3 already has a wide 9–64-row path.
  - exl3 `f4db698` and `0315f22`; Kyojin `fce7501` as code, though its evidence is used in item 4.
  - Kyojin `6f78955`, `88c615a` and `06294c8`.
- **Multi-GPU, TP, CPU-MoE:** exl3 pipelined prefill (`85c27f0`, `346d4c0`), layer split (Kyojin `5bd9b34`,
  `8955569`), every CPU-MoE and TP commit.
- **Other models:** KDA, DSv4.1, Gemma4 assistant (`e7af342`, `2bd9970`), LFM3, Nemotron, Ling, DFlash
  variants. Gemma is not on this branch.
- **Python host-overhead fixes with no C++ equivalent:**
  - exl3 `288d043`, `16c695e`, `9f88805`, `51d1e79`.
  - Kyojin `43337b9`: lag-1 pipeline, default off.
  - Kyojin `d9af888`: union batch decode, default off.
- **Generator-specific checkpoint bugs:** exl3 `f001dea`, `b781845`, `b8273e2`, `89c35ce`. Gufo's own
  checkpoint and alignment contract (U2-2) differs. They are worth one test idea: draft and target
  checkpoint alignment after a cancel and rewind inside an atomic prefill span.
- **Possible audit, low priority:** exl3 `4bd4946` rejects expert-range filtering past a 128-slot index
  buffer. Check that Gufo's routed tile and compact buffers have the same kind of bound. U1-1 already
  bounded the selection and remap buffers.
- **Conversion:** exl3 `56ffdae` (scaled Hessians default), `ca45ec6` (affinity Hessians) and `1d90bef`
  (`-hq 2`). These are relevant only to a new requant. Per memory, prefer prebuilt packs.

## Notes for the integration pass

- Items 1 and 2 are host-only. Build and run the CPU tests per item. Correctness goes into the single
  combined pass: the lodging-review loop repro, pi_agent replay, sampled MTP timing.
- Item 3 needs one `--cost-audit` run in the combined pass. Item 4 needs KL, a long-context check and a
  decode timing at C1 and C4.
- Record every result, kept or rejected, in `docs/GFX1151-KNOWHOW.md`.
