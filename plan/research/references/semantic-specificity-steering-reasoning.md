# Evaluating the Semantic Specificity of Representation Steering

## Identification status (read first)

- The requested title, "Evaluating the Semantic Specificity of
  Representation Steering in **Reasoning**" (EMNLP 2026 Findings), was **not
  found** under that exact title.
- The closest match, almost certainly the same work, is:
  Zhangdie Yuan, Andreas Vlachos (University of Cambridge). "Evaluating the
  Semantic Specificity of Representation Steering in **Language Models**."
  arXiv:2608.29431v1, cs.CL, 29 Aug 2026.
  - arXiv: https://arxiv.org/abs/2608.29431 (PDF: https://arxiv.org/pdf/2608.29431)
  - Dataset: https://huggingface.co/datasets/MoyYuan/Asymmetricity-2.0
- **Venue unverified.** The arXiv page has no comments field naming a venue;
  the PDF contains an appendix answering "reviewer inquiries" and mentions a
  double-blind period, so it was under review somewhere. EMNLP 2026 Findings
  acceptance could not be confirmed by search or the ACL Anthology.
- Read: full PDF text (converted locally).

## Problem

Steering vectors are reported to "repair" reasoning failures. The paper
asks whether a vector that fixes a target failure actually changes the
reasoning, or just pushes the output toward one label. Standard locality
checks (unrelated tasks such as trivia) cannot tell, because those tasks do
not produce the target label.

## Setup

- Task: Asymmetricity-2.0, a procedurally generated binary NLI benchmark from
  Wikidata relations (72M+ entries). Rule families:
  - competent: `swap_symmetry` ("Alice is a friend of Bob" -> "Bob is a friend
    of Alice", entailment), `transitivity_chain`;
  - failing ("contradiction blindness"): `symmetric_nonreflexive` ("Alice is
    married to Bob" -> "Alice is married to Alice"), `symmetry_not_transitive`.
- Anchor model Llama-3.2-1B-Instruct: predicts entailment almost always;
  contradiction recall 0.0000 (0/4,957 at the 16,384 scale), F1-macro 0.4102.
  Across 18 models at the 65,536 scale: 7 have recall 0, 5 near zero, 2 above
  0.5.
- Prompting: structured JSON CoT ("reasoning" and "label" fields), greedy,
  max 512 tokens.

## Method (the steering being audited)

- Vector: difference of means at a late layer between successful and failing
  contexts: S_success = 100 contradiction items answered correctly with
  few-shot help; S_fail = 100 contradiction items answered wrongly. Both sets
  are contradiction inputs (text balanced), but success ends in the
  "contradiction" token and failure in the "entailment" token.
- Layers: Llama 13/16, Gemma-4-E4B-It 21/26, "Qwen-3.5-9B-Instruct" 32/40,
  DeepSeek-R1-Distill-Qwen3-8B 26/32 (model names as written in the paper;
  the Qwen checkpoint name is inconsistent inside the paper).
- Application: h_t <- h_t + alpha v/||v||, alpha = 1.5, injected at the output
  of the target layer for all generated tokens after the prompt.
- "Oracle gating": inject only on the failing rule families. A diagnostic
  upper bound, not deployable.

## Cross-Rule Transfer (CRT) audit

1. Find rule families in the same domain and label space where the unsteered
   model is competent (>= 95%).
2. Apply the vector unconditionally on those families.
3. Measure accuracy loss and count of injected target labels; McNemar test.
4. If accuracy stays >= 95%, call it a reasoning repair; if it collapses and
   errors are the target label, call it a label-bias override.

## Key numbers

- Oracle-gated: 100% recovery on the failing families for all four models,
  zero side effects (because of the gate).
- CRT (Llama, N = 500 entailment items): 99.6% -> 40.4% (298 false
  contradictions; McNemar chi^2 290.1-299.8, p < 1e-12 over 5 resamples;
  40.4 +- 0.8%). Temperature sampling: 38.6 +- 1.2%.
- PCA first-component vector instead of difference of means: 44.2%.
- Naturalistic MNLI subset (competent >= 95%), vector from MNLI failures: 35.4%.
- Strength sweep (Llama, ungated): source accuracy 0.696 at alpha 0, 1.000 at
  0.25-1.0, 0.683 at 1.5-2.0; no alpha restores contradictions without
  destroying entailment on competent rules.
- Direct logit bias (beta = +-3.0 added to the label logits, matched to the
  logit shift of the vector via the unembedding): identical accuracy to the
  steering vector on CLUTRR / SpartQA / ProntoQA under greedy decoding;
  Cohen's kappa >= 0.94 under sampling.
- Token-position ablation: steering only the final label token reproduces
  100% of the recovery; steering only the preceding CoT tokens gives 0%. The
  vector does not change the reasoning text, only the final commit.
- Shuffle control (random success/failure labels) still "works", because the
  label-outcome confound makes any split collinear with the label direction.
  Random Gaussian and orthogonal vectors of equal norm do nothing.
- Early-layer steering (Llama layer 2): only +15.4% with 3,465 side effects.
- Cross-model graft (Llama vector mapped to Qwen through a least-squares
  linear map): 96.6% accuracy, 17 side effects; random orthogonal map 0.062.
- DeepSeek-R1-Distill-Qwen3-8B: apparent 94-100% downstream accuracy is a
  format-compliance effect (baseline outputs were mostly unparsed).

## Recommendations from the paper (checklist)

1. Evaluate the vector unconditionally on natively competent tasks in the
   same label space.
2. Report gated and ungated side effects.
3. Plot output-label histograms before/after.
4. Compare against a direct logit-bias baseline.
5. For open weights, report cosine of the vector with unembedding label
   directions per layer.
Mitigations named (not tested in depth): balanced contrastive subtraction
(class-balanced pools), projecting the vector orthogonal to the unembedding
label directions, counterfactual label-swapped data.

## Limitations

English only; stylized NLI and downstream tasks with explicit label tokens;
small models (0.5B-9B); several table/name inconsistencies in v1; the CRT
logic needs a competent family that shares the label space.

## Reproduction notes

Build success/failure pools on the *same* inputs and look at the token where
the label is emitted; inject at a late layer, alpha around 1-1.5 on a
normalized vector; then run CRT on competent families and the DLB baseline.
