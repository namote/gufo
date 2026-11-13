# Controlling Thinking Speed in Reasoning Models (extra related paper)

## Citation

Zhengkai Lin, Zhihang Fu, Ze Chen, Chao Chen, Liang Xie, Wenxiao Wang,
Deng Cai, Zheng Wang, Jieping Ye (Zhejiang University, Alibaba Cloud,
Zhejiang University of Technology). "Controlling Thinking Speed in Reasoning
Models." arXiv:2507.03704 (v2, 30 Oct 2025). NeurIPS 2025 (listed as
Spotlight on neurips.cc/virtual/2025/poster/119132).

- arXiv: https://arxiv.org/abs/2507.03704 (HTML: https://arxiv.org/html/2507.03704v2)
- Code: https://github.com/D2I-ai/thinking-speed-control

Why included: it is the clearest 2025 example of *token-level conditional*
steering for reasoning length (strength switched per token by a difficulty
signal), which bridges SEAL/ASC and CAST.

## Problem

Reasoning models always think slowly. The paper asks how to steer thinking
speed (fast vs slow) and *when* to change it inside one response.

## Method

**Observation.** R1-Distill-Qwen-7B MATH500: the shortest responses (mean
262.5 tokens) tend to start with "To"/"First"; the longest (mean 24,389.2)
with "Okay"/"Alright". Forcing a "To" opening cuts tokens ~5.4x (7B) and
~3.3x (32B).

**Extraction (PCA of paired differences).**
- 7.5k MATH training questions; keep fast/slow response pairs where both are
  correct. Split thoughts at double newline.
- Fast stimulus: first 2 steps of the fast response. Slow stimulus: the slow
  response truncated at the step closest in length to the fast one
  (length-matched pairs).
- Hidden state at the final token of each stimulus (ablation: end of the
  initial segment works best).
- Differences in both directions (fast-slow and slow-fast), PCA over the
  combined set, first PC per layer; sign set to point slow -> fast.
  ~6k pairs: 4k for vectors, 2k for validation.

**Application.**
- h_l <- h_l + alpha v_l on the last layers excluding the final one, at every
  generated token:
  - R1-Distill-Qwen-7B: layers 19-27; Qwen3-8B: 26-35;
    R1-Distill-Qwen-32B: 44-63; QwQ-32B: 49-63.
- alpha > 0 faster, alpha < 0 slower; swept over {16, 12, 8, 4, 0, -2, -4, -6}.

**Adaptive (conditional) control.**
- Difficulty signal per token: Jensen-Shannon divergence between early-layer
  logit-lens distributions and the final-layer distribution, averaged over an
  early-layer bucket (7B: layers 1-14; Qwen3-8B: 1-18; 32B models: 22-42).
  High values flag "Wait", "Alternatively", calculations.
- Sliding window k = 8: a token is difficult if its score > mu_W + lambda
  sigma_W (lambda 2.0 for 7B/8B, 1.5 for 32B). A difficult token resets alpha
  to -4 (slow); otherwise alpha rises by 2 per token up to +4 (fast).

## Models and benchmarks

R1-Distill-Qwen-7B/32B, QwQ-32B, Qwen3-8B; AIME24, AIME25, MATH-500, GPQA
Diamond, LiveCodeBench; AMC for validation. vLLM, max 32,768 tokens, T 0.6,
top-p 0.95, 8 samples per query.

## Key numbers

- Constant steering, 7B MATH500: baseline 92.9% / 3403.9 tokens; alpha = -4:
  94.1% / 4193.6; alpha = 16: 81.1% / 1410.5.
- Qwen3-8B: fast steering beats budget forcing by +8.2 to +11.4 average
  Pass@1 at the same speed-up; slow steering gives +0.5 to +1.5.
- Adaptive control: +1.26% accuracy and -8.56% tokens on average (e.g. 7B
  AIME24 52.5/12451 -> 53.8/10851; Qwen3-8B AIME25 62.9/17797 -> 65.4/17411).
- Random switching instead of difficulty-driven switching used more tokens
  (7B AIME24 14505 vs 10851).
- Extreme alpha causes repetitive generation.

## Limitations

Heuristic controller; shared strengths across models although models respond
unevenly; mechanism of the "To" opening not understood.

## Reproduction notes

Length-matched, both-correct fast/slow pairs; last-token states; PCA first
PC per layer; add on a block of late layers to every generated token; for
adaptive mode compute early-vs-final JS divergence per token and switch alpha
between -4 and +4 as above.
