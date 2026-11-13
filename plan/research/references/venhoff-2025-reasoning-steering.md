# Understanding Reasoning in Thinking Language Models via Steering Vectors

## Citation

Constantin Venhoff, Iván Arcuschin, Philip Torr, Arthur Conmy, Neel Nanda.
"Understanding Reasoning in Thinking Language Models via Steering Vectors."
arXiv:2506.18167 (v1 22 Jun 2025, v4 22 Oct 2025). arXiv comment: accepted to
the Workshop on Reasoning and Planning for LLMs at ICLR 2025.

- arXiv: https://arxiv.org/abs/2506.18167 (HTML: https://arxiv.org/html/2506.18167v4)
- Code and data: https://github.com/cvenhoff/steering-thinking-llms

Read: arXiv HTML v3 and v4. The workshop version is smaller (300 tasks, two
models, according to search listings); the arXiv version is summarized here.

## Problem

Thinking models (DeepSeek-R1 distills) produce long chains with characteristic
behaviors: backtracking, uncertainty statements, testing examples. The paper
asks whether these behaviors are linearly represented and whether a steering
vector can turn each one up or down in an interpretable way.

## Method

**Behavior taxonomy and annotation.**
- 500 tasks in 10 categories (mathematical logic, lateral thinking,
  scientific reasoning, ...), generated with Claude 3.5 Sonnet.
- Responses: greedy decoding, max 1000 tokens.
- GPT-4o splits each reasoning chain into labeled segments
  (`["label"] ... ["end-section"]`). Six categories: initializing, deduction,
  adding-knowledge, example-testing, uncertainty-estimation, backtracking.
  Initializing and deduction are dropped for steering (not specific to
  thinking models).

**Vector extraction (difference of means, per layer, residual stream).**
- Positive set D+: activations from annotated segments of the target behavior.
- Negative set D-: the whole dataset (all segments), i.e. "behavior vs
  average", not "behavior vs a matched opposite".
- Token positions: the token *just before* the annotated segment plus up to
  10 tokens of the segment itself, averaged. Including the preceding token is
  deliberate: the decision to start the behavior is made there.
- Each vector is rescaled to the norm of the mean overall activation.

**Layer selection.**
- Attribution patching: for each layer's candidate vector, estimate the
  effect on the next-token distribution (absolute mean KL divergence) at the
  relevant positions. Effects peak in the middle layers.
- Early layers whose vectors are highly similar to token embeddings are
  excluded (strong in Llama-8B; Qwen-14B more uniform).
- Chosen layers (Table 2):

| Behavior | Llama-8B | Qwen-1.5B | Qwen-14B |
|---|---|---|---|
| Uncertainty estimation | 12 | 18 | 29 |
| Example testing | 12 | 15 | 29 |
| Backtracking | 12 | 17 | 29 |
| Adding knowledge | 12 | 18 | 24 |

**Application.** The vector is added (positive steering) or subtracted
(negative steering) at the residual stream of the chosen layer at inference
time. **Unverified/not reported:** the steering coefficient, and whether all
token positions (prompt and generated) receive the vector. The README of the
repository also does not state these; they would have to be read from
`steering/evaluate_steering.py`.

## Models and benchmarks

- Steering: DeepSeek-R1-Distill-Qwen-1.5B, -Qwen-14B, -Llama-8B (the text is
  inconsistent in places and mentions only two models in some sections).
- Behavior comparison: five R1 distills (Qwen-1.5B/14B/32B, Llama-8B/70B)
  against five non-thinking baselines on 100 tasks.
- Steering evaluated on 50 unseen tasks from the same generated task set. No
  standard accuracy benchmark.

## Key numbers

- Thinking models average 27.6 sentences per response versus 14.4 for
  baselines, with more backtracking, uncertainty estimation and example
  testing and less knowledge addition.
- Steering results (change in the fraction of sentences per behavior) are
  shown only in figures; the text states direction only: positive steering
  raises backtracking, uncertainty and example testing; negative steering
  lowers them. **No numeric effect sizes or accuracy effects are reported in
  the text.**
- Cosine similarity among behavior vectors is mostly low to moderate;
  uncertainty and backtracking are moderately correlated (heatmap only).

## Limitations

- Annotation by GPT-4o without human validation; known false positives and
  negatives.
- Only distilled models; RL-trained thinking models (e.g. QwQ) untested.
- Taxonomy not claimed complete; no task-accuracy evaluation of the steered
  models; no side-effect measurement beyond behavior fractions.
- Negative set is the whole dataset, so the vector also contains whatever
  differs between "segments of this behavior" and "average text" (e.g.
  sentence-start tokens such as "Wait").

## Reproduction notes

1. Generate reasoning chains (greedy, 1000 tokens) on a diverse task set.
2. Label segments with an LLM annotator using a fixed tag format.
3. For each behavior and layer, mean residual activations over
   [token before segment .. first 10 tokens of segment]; subtract the mean
   over all segments; rescale to the mean activation norm.
4. Pick the layer by attribution patching (KL effect on next token), skipping
   embedding-like early layers.
5. Add or subtract at that single layer; measure behavior fractions with the
   same annotator on held-out tasks.
6. Coefficient and token scope must be taken from the repository code
   (not in the paper).
