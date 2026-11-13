# SEAL: Steerable Reasoning Calibration of Large Language Models for Free

## Citation

Runjin Chen, Zhenyu Zhang, Junyuan Hong, Souvik Kundu, Zhangyang Wang
(UT Austin; Intel). "SEAL: Steerable Reasoning Calibration of Large Language
Models for Free." arXiv:2504.07986 (v3, 25 Oct 2025). Venue: a secondary
source says COLM 2025; **unverified** from the paper page.

- arXiv: https://arxiv.org/abs/2504.07986 (HTML: https://arxiv.org/html/2504.07986v3)
- Code: https://github.com/VITA-Group/SEAL

## Problem

Long CoT contains redundant reflection ("Wait, let me check") and transition
("Alternatively, ...") thoughts. Incorrect answers contain more of every
thought type, mostly reflection and transition. On R1-Distill-Qwen-1.5B
(MATH500), correct responses grow from 1,534 to 3,323 tokens with problem
difficulty. Too much reflection/transition costs tokens and can derail the
answer.

## Method

**Thought segmentation and labeling.**
- Split the output into thoughts at `\n\n`.
- Keyword rules (case-insensitive):
  - Transition: starts with "Alternatively"; or contains "another way /
    approach / method / solution / strategy / technique", "think differently".
  - Reflection: starts with "Wait"; or contains "verify", "make sure",
    "hold on", "think again", "Let me check", "seems right".
  - Execution: everything else.
- GPT-4o labeling instead of keywords: 88.6 vs 89.4 (keywords) on MATH500 7B.

**Extraction.**
- 1,000 MATH training problems; the target model generates full traces.
- Representation: hidden state at block i at the `\n\n` token that *starts*
  each thought (a fixed, structurally identical position for all thoughts).
- S = mean(Execution) - mean(Reflection ∪ Transition).
- t-SNE: execution separates from the others, better in deeper layers.

**Application.**
- During decoding, at the end of each thought, the `\n\n` token
  representation is updated: H <- H + alpha * S. Only the delimiter tokens are
  steered, not every token.
- alpha = 1.0 by default; layer 20 for R1-Distill-Qwen-1.5B and 7B, layer 55
  for QwQ-32B-Preview.
- No extra decoding latency (vector precomputed).

## Models and benchmarks

R1-Distill-Qwen-1.5B, R1-Distill-Qwen-7B, QwQ-32B-Preview. MATH500 (plus
hard subset, levels 4-5), GSM8K (1,319), LiveCodeBench (400 problems).
Greedy decoding. Baseline: logits penalty on reflection/transition words.

## Key numbers (accuracy / tokens, baseline -> SEAL)

| Model | MATH500 | MATH500 hard | GSM8K | LiveCodeBench |
|---|---|---|---|---|
| 1.5B | 67.0 -> 78.0; 4526 -> 3154 | 54.2 -> 68.3; 5737 -> 4086 | 74.1 -> 82.0; 2015 -> 999 | 18.5 -> 28.5; 8205 -> 6923 |
| 7B | 85.8 -> 89.4; 3389 -> 2661 | 79.8 -> 84.0; 4176 -> 3365 | 88.0 -> 88.4; 1142 -> 811 | 44.5 -> 51.7; 6856 -> 5974 |
| QwQ-32B | 90.4 -> 91.0; 2113 -> 1716 | 86.6 -> 85.5; 2793 -> 2323 | 95.4 -> 95.7; 698 -> 525 | 62.5 -> 63.5; 6016 -> 5309 |

- Token reductions range from about 12% to 50%; accuracy mostly up (largest
  gains on small models), one drop (QwQ hard set -1.1).
- The MATH-derived vector transfers to GSM8K and to code (LiveCodeBench);
  in-domain vectors were not better. NaturalPlan calendar (7B): 14.9 -> 18.6%,
  length 6341 -> 3531.
- Wall time per sample (50 MATH500 samples, GH200): 112.7 -> 70.0 s (1.5B),
  94.6 -> 63.5 s (7B).
- Logit penalty on keywords is weaker and makes the model *rephrase*
  reflections with other words; steering the representation removes the
  behavior rather than the surface word.
- Truncation control: a 3,500-token think budget gives 85.0% at 2,540 tokens
  vs SEAL 89.4% at 2,661 tokens (7B MATH500) - the gain is not only from
  shortening.

**Ablations (7B: MATH500 / GSM8K / LiveCodeBench).**
- Suppress reflection only: 88.6 / 88.4 / 50.7. Transition only: 87.8 /
  88.5 / 49.0. Both (SEAL): 89.4 / 88.9 / 51.7.
- Weakening *execution* instead: 65.0 / 79.6 / 27.5 (large damage). Steering
  the wrong component is costly.
- Layer: mid-to-late layers best (figure). Strength: removing too much
  reflection does not always help; alpha = 1.0 works consistently.

## Limitations

- Keyword rules are hand-made and English-specific.
- alpha and layer were picked by ablation on the same benchmarks reported.
- Only R1-style models; QwQ gains small.
- Minor inconsistencies in the paper (GSM8K 7B 88.4 vs 88.9 in two tables).

## Reproduction notes

1. Generate ~1,000 traces on MATH train with the target model.
2. Split by `\n\n`; label each thought by the keyword rules.
3. At a mid-to-late layer (about 0.7 of depth), take the hidden state at the
   `\n\n` token that begins each thought; S = mean(exec) - mean(refl ∪ trans).
4. At inference, add alpha S (alpha = 1) only to `\n\n` delimiter positions at
   that layer.
