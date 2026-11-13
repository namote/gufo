# What to adopt for Gufo (Flash-Next) steering

Assessment from the papers in this folder (see README.md) against our engine and the R5/R5b results (`docs/GFX1151-KNOWHOW.md`, pass 15 and R5b). Written 2026-11-13.

## What the engine already has

| Paper idea | Gufo today |
|---|---|
| CAST prompt-level condition gate | Dynamic abliteration: a detector at L38, a threshold, ON re-prefill, `trigger.with` companions |
| Manifold directional ablation at all layers with one direction | Suppression profiles with `direction_layer` (one direction for every layer) and strengths below 1 (partial removal) |
| Additive steering (ASC, SEAL, Thinking Speed) | `mode: "add"` profiles, signed per-layer strengths, up to 8 at once |
| Several behaviors at once | Multi-profile union; additive terms commute |
| Strength search | SPSA (teacher-forced margin), plus generation grids |

Missing:
- **Position gating** (SEAL: steer only `\n\n` thought-start tokens). The device already mixes ON and OFF rows in one batch, so a per-row gate on the row's input token may be a small change.
- **Per-token strength schedules** (Thinking Speed).
- **A thinking-trace collector.** The `answers` mode frames thinking off with an empty think block. We need thinking-on framing and per-segment rows, such as the `\n\n` row or the row before "Wait".

## Candidates, in order

### 1. Overthinking reduction by Manifold-style ablation (no engine change)

- **Why first:** thinking-mode answers are decode-bound at about 54 tok/s, so every thinking token cut is wall time saved. The papers report 30–70% fewer tokens with accuracy kept on math.
- **Data:** our own thinking-on runs on math and code (r5-coding, r5-trap, and a math set). Split traces at `\n\n`. Redundant states are the rows that start a thought beginning "Wait"/"Alternatively"/"Let me double-check" in long traces; concise states are execution-thought rows.
- **Direction:** a difference of means at one late layer (Manifold: the last-but-one; try L44–L46), projected onto the top-10 PCA subspace of the pooled states (CPU), and stored as a suppression profile with `direction_layer`.
- **Strength:** partial removal, alpha 0.1–0.7, swept on held-out problems for accuracy and thinking tokens.
- **Needs:** the thinking-trace collector mode, a CPU fit script, and a baseline of how long the model thinks now.

### 2. SEAL position gating

- Engine: apply an additive profile only on rows whose input token is the `\n\n` delimiter (prefill and decode, MTP verification rows included).
- Direction: `mean(execution) − mean(reflection ∪ transition)` at the `\n\n` rows; alpha about 1 at one mid-late layer (about 0.7 of depth, so L33–L36).
- Expected to have fewer side effects than every-token steering. SEAL also beats a matched truncation budget.

### 3. Coding detector plus trap direction (CAST, no engine change)

- Detector: coding versus ordinary prompts at an early-to-mid layer, fitted like the political detector.
- The R5b trap ×1 profile turns on via that trigger.
- Low value until a harder set shows a margin (R5b: +1–2 problems, within noise).

### 4. Process changes for every new direction (cheap)

- **Ungated test on inputs the model already gets right** (semantic specificity, Cross-Rule Transfer). Our ordinary check is the start; add tasks in the same output space, such as code the model already passes.
- **Trivial baselines:** a keyword logit penalty, truncation or a think budget, and the instruction itself (R5b showed the instruction as the ceiling).
- **A reverse-direction sanity check**, and a shuffled-label control for detector and contrast directions.
- **Remove shared components:** project the shared "an instruction is present" direction (R5b: cosine 0.3–0.46 in L0–L27) out of each instruction direction, or contrast two instructions on the same text.
- **Strength:** sweep to just below the cliff. ASC's next-token KL budget (1e-3) can seed the sweep through the scoring path.

## Not adopted

- **ThinkEdit** (edits attention output weights): our weights are EXL3-quantized and shared across profiles, so runtime steering fits better.
- **Thinking Speed difficulty schedule:** only after position gating exists. The gain reported is small (+1.3% accuracy, −8.6% tokens).

## Gaps none of the papers cover (ours to measure)

MoE models, per-layer strengths fitted jointly, and several active profiles. Record results in KNOWHOW.
