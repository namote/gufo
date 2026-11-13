# Activation steering references

Summaries of papers on steering reasoning behavior, reasoning length and
conditional steering. Each file has citation, problem, method (extraction,
application, conditioning), models, numbers, limitations and reproduction
notes. Numbers are copied from the papers; anything not verified is marked.
Read on 2026-11-13 from arXiv HTML/PDF pages.

## Files

| File | Paper |
|---|---|
| `venhoff-2025-reasoning-steering.md` | Venhoff et al. 2025, Understanding Reasoning in Thinking LMs via Steering Vectors (arXiv 2506.18167, ICLR 2025 workshop) |
| `asc-azizi-2025-cot-compression.md` | Azizi et al. 2025, ASC: Activation Steering for CoT Compression (arXiv 2507.04742) |
| `seal-chen-2025-reasoning-calibration.md` | Chen et al. 2025, SEAL: Steerable Reasoning Calibration (arXiv 2504.07986) |
| `manifold-steering-huang-2025-overthinking.md` | Huang et al. 2025, Mitigating Overthinking via Manifold Steering (NeurIPS 2025, arXiv 2505.22411) |
| `thinking-speed-lin-2025.md` | Lin et al. 2025, Controlling Thinking Speed in Reasoning Models (NeurIPS 2025, arXiv 2507.03704) - extra |
| `thinkedit-sun-2025.md` | Sun et al. 2025, ThinkEdit: mitigating overly short thinking (EMNLP 2025, arXiv 2503.22048) - extra |
| `semantic-specificity-steering-reasoning.md` | Yuan & Vlachos 2026, Evaluating the Semantic Specificity of Representation Steering in *Language Models* (arXiv 2608.29431). Requested "in Reasoning" title and EMNLP 2026 Findings venue not found/unverified |
| `cast-conditional-steering.md` | Lee et al. 2025, Programming Refusal with Conditional Activation Steering (CAST, ICLR 2025, arXiv 2409.05907) |

## Core summaries

**Venhoff et al. (reasoning behaviors).** GPT-4o labels segments of R1-distill
reasoning (backtracking, uncertainty, example testing, adding knowledge).
Difference of means: behavior segments vs all segments, over the token just
*before* the segment plus up to 10 segment tokens; rescaled to the mean
activation norm. A single middle layer chosen by attribution patching (KL
effect on next token), skipping embedding-like early layers. Adding or
subtracting the vector raises or lowers the behavior; no coefficients or
accuracy effects reported.

**ASC (CoT compression).** 50 pairs of (model's verbose CoT, GPT-4o concise
CoT); last-token residual at one layer (~0.75 depth); v = mean(concise -
verbose); added at every decoding step. Strength from a closed-form KL bound
(epsilon 1e-3) that lands near the empirical accuracy breakpoint. Up to -67%
tokens (Llama-8B GSM8K) with accuracy unchanged; MATH and GSM8K vectors have
cosine 0.92 and transfer to each other. Math only.

**SEAL (reasoning calibration).** Split thoughts at `\n\n`, label by keywords
(execution / reflection / transition). Vector = mean(execution) -
mean(reflection ∪ transition) at the `\n\n` token that starts each thought;
added (alpha = 1) only at `\n\n` tokens during decoding at one mid-late layer.
+11 points / -30% tokens on 1.5B MATH500; transfers MATH -> GSM8K and code.
Suppressing execution instead collapses accuracy; logit penalties on keywords
only make the model rephrase.

**Manifold Steering (overthinking).** Direction = final-token difference of
means between runaway "wait"-loop states and brief-answer prompts; projected
onto the top-10 PCA subspace of the activations; then *ablated* (h -= alpha
(r.h) r) at all layers. Plain directional ablation plateaus and then
reverses/collapses as alpha grows; projection removes that. Up to -71% tokens
(1.5B GSM8K) and -42% (7B MATH500) with accuracy kept; direction from the
last-but-one layer, alpha 0.3-0.7. Intermediate-layer directions cut more
tokens but break accuracy.

**Controlling Thinking Speed (extra).** Length-matched, both-correct
fast/slow response pairs; PCA first component of paired differences per
layer; added to every generated token over a block of late layers. An
early-vs-final layer JS-divergence difficulty signal switches the strength per
token between slow (-4) and fast (+4): +1.3% accuracy, -8.6% tokens.

**ThinkEdit (extra).** Opposite failure: overly short CoT. Long-vs-short CoT
direction averaged over all CoT tokens; projected out of W_o of the 4% of
heads that write it. Only the short cases get longer; +3.34% overall,
+6.39% on short cases; mean length nearly unchanged.

**Semantic specificity (Yuan & Vlachos).** A late-layer "success minus
failure" vector that "repairs" contradiction detection is just a label
bias: with an oracle gate it shows 100% recovery and zero side effects, but
ungated on competent rules accuracy falls 99.6% -> 40.4%. It equals a direct
logit bias, works only when injected at the label token, and even a
label-shuffled vector works. Proposes Cross-Rule Transfer: test vectors
ungated on competent tasks in the same label space.

**CAST (conditional steering).** Behavior vector from same-instruction
refusal vs compliance continuations (suffix tokens, PCA); condition vector
from condition vs non-condition prompts (all prompt tokens, PCA). On the
prompt pass, cos(h, tanh(proj_c h)) vs a grid-searched threshold at an early
layer decides once whether the behavior vector is added for the whole
generation. Harmless refusal drops from 46-99% (unconditional) to 0-7%,
harmful refusal stays 70-91%. Conditions compose with OR and complement.

## Key takeaways across papers

### 1. Extraction contrast: hold everything constant except the behavior

- The vectors that worked contrast **the same model on matched inputs**:
  SEAL (same traces, same delimiter token, different thought type), CAST
  (same instruction, refusal vs compliance continuation), Thinking Speed
  (both-correct, length-matched fast/slow pairs), Manifold Steering (the
  model's own runaway vs brief states).
- When the two sides differ in more than the target behavior, the vector
  carries the extra difference. ASC deliberately mixes in style (GPT-4o
  concise, "minimal English") because short style *is* its goal. The
  semantic-specificity paper shows the extreme: success-vs-failure
  contrasts on a task where failure always means the other label produce a
  pure **label/output-token direction**, not a reasoning direction; even a
  label-shuffled split gives the same vector.
- This matches our observation: "reference answer vs the model's own answer"
  differs mainly in who wrote it (style), so it captured style. "Same
  answer read with vs without the instruction" holds the text fixed and
  varies only the condition, which is the CAST/SEAL-style matched contrast;
  it transferred the instruction's effect.
- Check the vector against simple confounds before trusting it: cosine with
  unembedding rows of likely output tokens (label words, "Wait", refusal
  words), cosine with token embeddings in early layers (Venhoff excludes
  embedding-like layers), and a shuffled-label control.
- Shared components: if two behavior directions share a generic part (for
  us: an "an instruction is present" component in layers 0-30), the
  matched-contrast approach (difference between two instructions on the same
  text, or projecting the shared mean direction out of each) isolates the
  specific part. None of these papers tests this explicitly; it follows from
  the confound results above (inference, not a paper result).
- Mean-difference and PCA-of-paired-differences gave similar behavior in
  the semantic-specificity audit (40.4% vs 44.2%): the estimator matters
  less than the contrast.

### 2. Token positions for extraction and application

- Extract where the decision is made: the token *before* a behavior starts
  (Venhoff), the `\n\n` that opens each thought (SEAL), the last token of a
  matched prefix (Thinking Speed, ASC, Manifold), prompt-average for a
  condition (CAST), response-suffix average for a behavior (CAST).
- Applying at the same kind of position is the most surgical choice: SEAL
  steers only `\n\n` delimiter tokens and keeps or raises accuracy. Every-
  token application (ASC, Thinking Speed, Manifold) works but needs smaller
  strengths and shows the strength cliffs.
- Warning sign: if steering only the final answer/label token reproduces the
  whole effect and steering the reasoning tokens does nothing, the vector is
  an output bias (semantic-specificity token ablation).

### 3. Layer choice

- Middle-to-late layers dominate for reasoning-style behaviors: ASC ~0.75 of
  depth, SEAL 0.7-0.86 (layer 20 of 28, 55 of 64), Thinking Speed the last
  ~third excluding the final layer, Venhoff mid layers by attribution
  patching.
- The last layers are closest to the unembedding: vectors there act like
  logit biases (semantic-specificity paper). Manifold Steering does extract
  at the last-but-one layer but applies a *projection* at all layers, not a
  late addition.
- Early layers are weak and unspecific (semantic-specificity: +15% with 3,465
  side effects) or embedding-like (Venhoff). Condition detection, by
  contrast, works in early-to-middle layers (CAST searches only the first
  half).
- Choose the layer by a measured effect (attribution patching / KL effect,
  or a sweep that reports both target metric and accuracy): Manifold's layer
  table shows neighboring layers can cut 3x more tokens while losing 15-25
  accuracy points.

### 4. Strength calibration

- All papers see a cliff: the effect grows, plateaus, then accuracy drops or
  output degenerates (ASC breakpoint, Manifold plateau and collapse,
  Thinking Speed repetition at extreme alpha, semantic-specificity cliff
  between alpha 1.0 and 1.5).
- Normalize the vector (unit norm or mean activation norm) so alpha is
  comparable across layers, then sweep and pick just before the cliff on a
  held-out set measuring both the target behavior and unrelated quality.
- ASC's KL budget (epsilon = 1e-3 on the next-token distribution) is a
  cheap principled default; Manifold's alpha is a fraction of the projected
  component (0.3-0.7), which self-scales with the activation.
- Projection (remove the component) is safer than addition for suppression,
  because it cannot push past zero. Projecting the direction onto the
  activation PCA subspace first (Manifold) removes off-manifold noise that
  otherwise causes the plateau/reversal; useful for our project-out
  profiles.

### 5. Conditioning

- Unconditional steering produces large side effects on non-target inputs
  (CAST: harmless refusal 46-99%; semantic specificity: competent accuracy
  99.6 -> 40.4%).
- CAST-style prompt-level gating (our detector) is the proven fix; the
  condition vector is extracted from prompt tokens and thresholded at an
  early layer by an F1 grid search. Condition and behavior are separate
  vectors at separate layers.
- Finer conditions exist: position gating (SEAL delimiters) and token-level
  strength schedules from a difficulty signal (Thinking Speed).
- A gate hides problems: a vector that is only a label bias still looks
  perfect under a good gate. Always also evaluate the vector **ungated** on
  inputs where the model is already correct, and with the gate on
  near-miss negatives.

### 6. Avoiding and measuring side effects

- Measure accuracy on the target task *and* on competent tasks in the same
  output space (Cross-Rule Transfer), not only on unrelated benchmarks.
- Compare against trivial baselines: logit bias on the relevant tokens
  (semantic specificity), keyword logit penalty (SEAL - surface penalties
  make the model rephrase), truncation / budget forcing (SEAL, Thinking
  Speed). If the vector is no better, it is not doing anything deeper.
- Inspect output distributions (label histograms, format compliance): the
  DeepSeek "improvement" in the semantic-specificity paper was a formatting
  effect.
- Suppressing the wrong component is costly (SEAL: weakening execution drops
  7B MATH500 89 -> 65); steer the narrowest behavior that explains the
  problem.
- Reverse direction tests are cheap sanity checks (Venhoff, Manifold: the
  negative direction should produce the opposite behavior).

### Gaps

No paper here tests MoE models, models above ~32B dense, or per-layer
strength vectors (`a_L` per layer) fitted jointly; all use one layer, a
block of layers with one alpha, or all layers with one alpha. None reports
how shared "instruction present" components interact across multiple
active profiles.
