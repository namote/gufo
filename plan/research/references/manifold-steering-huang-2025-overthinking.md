# Manifold Steering: Mitigating Overthinking in Large Reasoning Models

## Citation

Yao Huang, Huanran Chen, Shouwei Ruan, Yichi Zhang, Xingxing Wei, Yinpeng
Dong (Beihang University, Tsinghua University, Shanghai Qi Zhi Institute).
"Mitigating Overthinking in Large Reasoning Models via Manifold Steering."
NeurIPS 2025. arXiv:2505.22411 (v2, 16 Nov 2025).

- arXiv: https://arxiv.org/abs/2505.22411 (HTML: https://arxiv.org/html/2505.22411v2)
- NeurIPS: https://proceedings.neurips.cc/paper_files/paper/2025/hash/946827aadb9f7dbdb27f4a0c77dd1859-Abstract-Conference.html
- Code: https://github.com/Aries-iai/Manifold_Steering

## Problem

Overthinking (repeated verification, "wait" loops) appears linearly encoded,
but plain directional steering against it plateaus and then *reverses* as the
strength grows; at high strength the output collapses (e.g. a string of
zeros). The paper attributes this to noise components of the direction that
lie off the low-dimensional subspace where activations actually live.

## Method

**Direction extraction (difference of means).**
- 20k questions from OpenMathInstruct-2; 5 samples per question per model
  (temperature 0.6, top-p 0.95, max 16k tokens).
- Redundant set: questions where all 5 responses exceed 16k tokens without
  finishing and contain >20 hesitation keywords ("wait", "alternatively",
  ...). The response is truncated at the first "wait".
- Concise set: questions where all 5 responses are <1k tokens with no
  hesitation keywords; the prompt alone is used.
- 500 samples kept per set; 100 per set used for the direction, outliers
  removed with IsolationForest.
- r = unit-normalized difference of means of the *final-token* residual
  activations at one layer l. Early layers overlap; middle-to-late layers
  separate the two sets.
- Note the contrast design: the positive side is "the state right before the
  model says 'wait'" in a runaway trace; the negative side is the state at
  the end of a prompt that the model answers briefly. Both are the model's
  own activations.

**Manifold projection.**
- PCA on activations of the full set (500 redundant + 500 concise) at layer
  l. Top k = 10 components explain >70% of variance.
- U_eff = top-k PCs; r_ot = normalize(U_eff U_eff^T r).
- Theory (approximate, Gaussian assumptions): the off-manifold part of r
  adds noise that is amplified layer by layer; projecting removes it.

**Application (projection/ablation style, not additive).**
- h' = h - alpha * r_ot r_ot^T h, applied to every activation across **all
  layers** ("this direction is applied to all layers"); the per-model layer
  number is only where the direction is computed. Prompt vs generated token
  scope is not stated (the text says "every activation").
- alpha = 1 is full removal of the component; alpha < 1 is partial.

| Model | Direction layer | alpha |
|---|---|---|
| R1-Distill-Qwen-1.5B | 27 (of 28) | 0.7 |
| R1-Distill-Qwen-7B | 27 (of 28) | 0.3 |
| R1-Distill-Llama-8B | 31 (of 32) | 0.5 |
| R1-Distill-Qwen-14B | 47 (of 48) | 0.3 |

- Direction-layer sweep (MATH500): directions from intermediate layers often
  cut many more tokens but damage accuracy badly (e.g. 7B layer 25: 72.2% /
  1070 tokens vs vanilla 88.2% / 3824; 1.5B layer 15: 57.6% / 1469). The
  last-but-one layer kept accuracy (7B L27: 88.4% / 2239).
- alpha sweep on R1-7B MATH500: effect visible from alpha = 0.1; 77.1% token
  reduction at alpha = 0.5 with accuracy loss; alpha = 0.3 chosen.

## Models and benchmarks

R1-Distill-Qwen-1.5B/7B/14B and R1-Distill-Llama-8B; GSM8K, MATH500,
AMC2023, AIME2024; cross-domain LiveCodeBench and GPQA-Diamond. Baselines:
Dynasor (probing early exit) and SEAL.

## Key numbers (Pass@1 % / tokens; vanilla -> Manifold Steering)

| Model | GSM8K | MATH500 | AMC2023 | AIME2024 |
|---|---|---|---|---|
| 1.5B | 76.7/2035 -> 77.2/593 (-71%) | 76.4/4762 -> 78.6/3458 (-27%) | 70.0/7089 -> 72.5/6236 | 26.7/11352 -> 30.0/10134 |
| 7B | 87.5/1143 -> 87.6/440 (-62%) | 88.2/3824 -> 88.4/2239 (-42%) | 87.5/5871 -> 87.5/4440 | 50.0/10784 -> 53.3/8457 |
| 8B | 82.7/1217 -> 82.8/542 (-55%) | 87.8/4009 -> 88.0/2873 (-29%) | 85.0/5723 -> 85.0/4400 | 33.3/11278 -> 50.0/9457 |
| 14B | 93.2/742 -> 93.6/438 (-41%) | 92.8/3496 -> 92.8/2074 (-41%) | 90.0/5484 -> 90.0/4061 | 66.7/9986 -> 63.3/8132 |

- SEAL as reimplemented by these authors: e.g. 7B MATH500 87.8 / 2651.
- Code and GPQA: 12-27% fewer tokens with accuracy maintained (figure only).
- Reverse steering (adding the direction) produces verbose, over-cautious
  output - the direction is bidirectional.
- Side effect check: the same manifold-projection recipe applied to the
  refusal direction on Qwen2.5-7B-Instruct gives 100% jailbreak success on
  AdvBench (vs 74% for plain directional ablation at alpha = 2.0) - i.e.
  projection makes ablation *stronger and cleaner*, for good or bad.
- Latency (R1-7B MATH500, s): original 1.74, Dynasor 39.89, SEAL 1.37,
  Manifold Steering 1.05.

## Limitations

- Fixed alpha per model; authors suggest adaptive, task-dependent strength.
- Only math-trained R1 distills; no multimodal or specialized domains.
- Some accuracy drops (14B AIME -3.4).
- Theory relies on Gaussian approximations; k = 10 chosen by observation,
  no threshold rule.

## Reproduction notes

1. Build redundant (runaway, many "wait") and concise (short, no hesitation)
   question sets from the model's own samples.
2. Final-token residual activations at a late layer; difference of means on
   ~100 per side after outlier removal; normalize.
3. PCA on ~1,000 activations from both sets at that layer; keep top 10 PCs;
   project the direction into that span; renormalize.
4. At inference subtract alpha (r_ot . h) r_ot from the residual at every
   layer; sweep alpha in [0.1, 0.7] on a validation set.
