# ThinkEdit: Mitigating Overly Short Thinking (extra related paper)

## Citation

Chung-En Sun, Ge Yan, Tsui-Wei Weng (UC San Diego). "ThinkEdit:
Interpretable Weight Editing to Mitigate Overly Short Thinking in Reasoning
Models." EMNLP 2025 (ACL Anthology preview id 2025.emnlp-main.861; venue from
search listing, the arXiv page itself does not state it). arXiv:2503.22048
(v4, 29 Sep 2025).

- arXiv: https://arxiv.org/abs/2503.22048 (HTML: https://arxiv.org/html/2503.22048v4)
- Code: https://github.com/Trustworthy-ML-Lab/ThinkEdit

Why included: the opposite failure (too-short reasoning) and the step from a
steering direction to a permanent, *input-dependent* weight projection, which
is the weight-space analogue of "project a direction out".

## Problem

R1 distills sometimes write very short CoT, and those answers are much less
accurate (about 20% lower on MATH Level 5 per the introduction).

## Method

**Length direction.**
- 2,000 GSM8K training problems; D_long = CoT >1,000 tokens, D_short = CoT
  <100 tokens.
- Hidden states averaged over *all CoT token positions*, then per group, per
  layer, for both the post-attention and post-MLP residual.
- v_l = mean_long - mean_short. Middle layers carry most of the effect.

**Steering test.** Add alpha v_l at every layer with alpha in [-0.08, 0.08].
Negative alpha shortens reasoning and lowers accuracy consistently; positive
alpha helps some models (Llama-8B about +10% on GSM8K) and hurts others
(1.5B about -10%).

**Head selection and edit.**
- Per-head output contribution C_h = A_h W_o^h, averaged over CoT tokens of
  short-reasoning problems; score = <C_h, -v_hat>.
- Top 4% of heads (clustered in middle layers); edit
  W_o^h <- W_o^h (I - v_hat v_hat^T) (about 0.2% of parameters). Because it
  is a projection inside weights, the effect depends on the input: heads only
  lose the component when they actually write it.

## Models and benchmarks

R1-Distill-Qwen-1.5B / -Llama-8B / -Qwen-14B (+32B in appendix); GSM8K,
MMLU Elementary Math, MATH Level 1, MATH Level 5, MATH-500; mean of 10 runs,
max 16,384 tokens.

## Key numbers (original -> ThinkEdit)

| Model | GSM8K | MMLU Elem. Math | MATH-L5 | MATH-500 |
|---|---|---|---|---|
| Qwen-14B | 90.80 -> 93.78 | 95.08 -> 96.56 | 90.25 -> 91.03 | 91.48 -> 91.92 |
| Llama-8B | 82.26 -> 89.44 | 96.01 -> 96.19 | 85.49 -> 86.49 | 87.26 -> 88.06 |
| Qwen-1.5B | 79.15 -> 84.56 | 68.52 -> 90.66 | 75.48 -> 75.05 | 82.22 -> 82.24 |

- Abstract: +6.39% on short-reasoning cases, +3.34% overall.
- Shortest 5%: e.g. Llama-8B MATH-L5 78.64 -> 95.76.
- Average length barely changes (about -0.27% overall; per model +2.94%,
  +3.53%, -5.73%): only the overly short cases get longer.
- Beats appending "Wait" to short reasoning.

## Limitations

Helps only when reasoning is overly short; little benefit for already
verbose models; math only.

## Reproduction notes

Bucket CoTs by length (<100 vs >1,000 tokens), mean over CoT tokens per
layer, score heads by projection of their mean output on -v_hat, project v_hat
out of W_o of the top 4% heads.
