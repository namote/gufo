# ASC: Activation Steering for Chain-of-Thought Compression

## Citation

Seyedarmin Azizi, Erfan Baghaei Potraghloo, Massoud Pedram (University of
Southern California). "Activation Steering for Chain-of-Thought Compression."
arXiv:2507.04742 (v2, 8 Jul 2025). Listed on the NeurIPS 2025 virtual site
(https://neurips.cc/virtual/2025/126680); the exact track (main conference or
workshop) is **unverified**.

- arXiv: https://arxiv.org/abs/2507.04742 (HTML: https://arxiv.org/html/2507.04742)
- Code: https://github.com/ArminAzizi98/ASC

## Problem

Reasoning models write chains of thought far longer than needed. The paper
asks whether CoT can be compressed at inference time, without retraining, by
moving the hidden state from the "verbose" region toward the "concise" region
of the residual stream.

## Method

**Extraction (difference of means over paired texts).**
- 50 calibration questions sampled from MATH500 and GSM8K.
- Verbose CoT: the target model's own CoT under standard prompting.
- Concise CoT: written by GPT-4o, instructed to use concise, math-centric
  reasoning with minimal English.
- Each "question + CoT" string is run through the target model; the
  residual-stream activation at the final token (described as the EOS token)
  at layer l is recorded.
- v = mean over pairs of [h(concise) - h(verbose)].
- Note: the contrast is "other writer's concise text vs the model's own
  verbose text". The vector therefore also contains style differences
  (fewer English words, more math notation); the paper treats that as the
  desired effect.

**Application.**
- h <- h + gamma * v at a single layer, at every decoding step until EOS.
  Whether prompt tokens are also steered is **not stated**.

**Strength calibration (KL bound).**
- Bound: KL(unsteered || steered) <= (1/4) gamma^2 a^2 + (1/4) L a gamma^3 +
  (1/16) L^2 gamma^4, where a = ||W v|| (Jacobian-vector product norm of the
  next-token logits w.r.t. the hidden state) and L is a curvature constant.
- With budget epsilon = 1e-3: solve x^3 + x^2 - 4 epsilon L^2 / a^4 = 0,
  gamma_raw = (a/L) x, gamma_max = max(0, (1 - L gamma_raw / (4a)) gamma_raw).
- a = median JVP norm over the 50 calibration hidden states; L = 95th
  percentile of Hessian-vector-product norms.
- The paper reports that the KL-derived gamma lines up with the empirical
  "accuracy breakpoint" of a gamma sweep (figure only).

| Model | gamma | Layer |
|---|---|---|
| DeepSeek-R1-Distill-Qwen-7B | 0.275 | 21 (of 28) |
| DeepSeek-R1-Distill-Llama-8B | 0.46 | 21 (of 32) |
| QwQ-32B | 0.50 | 57 (of 64) |

The a, L and gamma_max values behind these gammas are not reported.

## Models and benchmarks

R1-Distill-Qwen-7B, R1-Distill-Llama-8B, QwQ-32B; MATH500 and GSM8K.
Sampling: temperature 0.7, top-p 0.9, repetition penalty 1.1. Baselines:
vanilla CoT, TCC, DEER, Chain-of-Draft (CoD), SEAL.

## Key numbers (accuracy % / mean tokens)

| Model | Method | MATH500 | GSM8K |
|---|---|---|---|
| Qwen-7B | CoT | 88.8 / 3984 | 88.6 / 1080 |
| Qwen-7B | SEAL | 89.4 / 2661 | 88.4 / 811 |
| Qwen-7B | CoD | 88.2 / 1852 | 87.9 / 550 |
| Qwen-7B | ASC | 89.0 / 1543 | 88.6 / 536 |
| Llama-8B | CoT | 89.2 / 3554 | 89.1 / 2610 |
| Llama-8B | ASC | 89.2 / 2353 | 89.3 / 850 |
| QwQ-32B | CoT | 93.8 / 4508 | 96.5 / 1530 |
| QwQ-32B | ASC | 94.2 / 2222 | 96.4 / 830 |

- Headline: up to 67.43% shorter CoT (Llama-8B GSM8K, 2610 -> 850) with
  accuracy unchanged or +0.2; QwQ-32B MATH500 -51%.
- Wall-clock: "average" (abstract) / "up to" (body) 2.73x on MATH500 with the
  8B model.
- Cross-task: the MATH500 and GSM8K vectors have cosine 0.92; the MATH500
  vector on GSM8K gives 88.4 / 611 tokens; the GSM8K vector on MATH500 gives
  88.8 / 1631 tokens (Qwen-7B). The concise direction is largely task-generic.
- Strength sweep (Qwen-7B MATH500, figure only): compression grows with gamma;
  accuracy is flat until a breakpoint, then falls.

## Limitations

- Only math benchmarks and three models; no evaluation on non-math or
  non-reasoning tasks (side effects outside math unmeasured).
- Concise targets come from a different model (GPT-4o); 50 pairs only; no
  ablation over the number of pairs.
- KL bound relies on a smoothness assumption and estimated constants; no
  direct check that the bound holds.
- Layer chosen empirically in the middle-to-late range; no layer sweep table.

## Reproduction notes

1. 50 questions; target model writes verbose CoT; a strong model writes a
   concise CoT for the same question.
2. Last-token residual activation at one layer (about 0.75 of depth) for
   each string; v = mean(concise - verbose).
3. Estimate a (median JVP norm) and L (95th pct HVP norm) on the calibration
   states; compute gamma_max for epsilon = 1e-3, or sweep gamma and pick the
   value just before accuracy drops.
4. Add gamma v at that layer at every decoding step.
