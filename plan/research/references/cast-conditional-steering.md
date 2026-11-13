# CAST: Programming Refusal with Conditional Activation Steering

## Citation

Bruce W. Lee (University of Pennsylvania, IBM Research), Inkit Padhi,
Karthikeyan Natesan Ramamurthy, Erik Miehling, Pierre Dognin, Manish
Nagireddy, Amit Dhurandhar (IBM Research). "Programming Refusal with
Conditional Activation Steering." ICLR 2025 (spotlight per secondary
sources). arXiv:2409.05907 (v3, 17 Feb 2025).

- arXiv: https://arxiv.org/abs/2409.05907 (HTML: https://arxiv.org/html/2409.05907v3)
- ICLR proceedings: https://proceedings.iclr.cc/paper_files/paper/2025/hash/e2dd53601de57c773343a7cdf09fae1c-Abstract-Conference.html
- Code / library: https://github.com/IBM/activation-steering

## Problem

Adding a refusal vector makes a model refuse everything, including harmless
prompts. CAST makes steering *conditional*: a condition vector checks the
prompt's hidden state, and the behavior vector is applied only when the
condition fires.

## Method

**Behavior vector (refusal).**
- 100 random Alpaca instructions x 100 refusal prefixes and 100 compliance
  prefixes = 10,000 contrast pairs (same instruction, different response
  start).
- Hidden states averaged over the response (suffix) tokens.
- Per layer: mean-center the positive and negative activations and take the
  first principal component (PCA), not a raw difference of means.

**Condition vector.**
- Positive: prompts that have the condition (Section 4: 90 synthetic
  harmful prompts for each of 45 Sorry-Bench categories, 4,050 total);
  negative: Alpaca instructions.
- Hidden states averaged over *all prompt tokens*; same mean-centering and
  PCA per layer.

**Condition check.**
- proj_c(h) = (c c^T / c^T c) h; similarity = cos(h, tanh(proj_c(h))).
- Fires if similarity > theta or < theta (direction is a search parameter).
- Layer(s), theta and direction chosen by grid search maximizing F1 on
  labeled positive/negative prompts; a prompt is positive if any layer in the
  chosen combination crosses theta. Search limited to the first half of the
  layers. Example: Hermes 2 Pro, layer 7, fires when sim < 0.048; Qwen 1.5
  1.8B harmful condition at layer 8, > 0.031.
- The check runs once, on the prompt forward pass. If it fires, the behavior
  vector is added at every later forward pass (the switch stays on for the
  whole generation).

**Behavior application.** h <- h + f(h) alpha v on a block of layers. Visible
examples (Table C.1): Qwen 1.5 Chat 1.8B layers 10-20, alpha 4; Qwen 1.5 32B
layers 36-61, alpha 14; Llama 3.1 Instruct add refusal at 17-24, alpha 1.7,
and remove refusal at 14-28, alpha -1. (Some table rows were garbled in the
HTML; per-model values beyond these are unverified.)

**Composition.**
- OR: several conditions, any firing triggers refusal ("hate speech or adult
  content").
- NOT / duality: flip the comparison direction to act on the exact
  complement ("if not health, refuse" restricts the model to health topics).
- Mixed rules: suppress refusal on one condition, induce on another
  (conflicts can reduce effectiveness).
- theta widens or narrows the triggered region.
- An explicit AND operator is not described in the parts read.

## Models and data

Qwen 1.5 Chat 1.8B and 32B, Llama 2 Chat 13B, Llama 3.1 Instruct 8B,
NeuralDaredevil 8B, Hermes 2 Pro 8B, OLMo SFT 7B, Zephyr Beta 7B, Danube 3
Chat 4B. Data: Sorry-Bench, Alpaca, harm categories from Liu et al. (2023);
test sets 500 unseen Alpaca + 450 unseen Sorry-Bench. Refusal judged by
protectai/distilroberta-base-rejection-v1 plus ~28 refusal keywords.

## Key numbers (refusal %, harmful / harmless)

| Model | Base | + refusal vector | + condition (CAST) |
|---|---|---|---|
| Qwen 1.5 1.8B | 45.78 / 0.00 | 100.00 / 96.40 | 90.67 / 2.20 |
| Danube 3 Chat | 46.22 / 0.60 | 77.11 / 46.00 | 69.78 / 1.80 |
| OLMo SFT | 53.11 / 5.20 | 93.33 / 89.60 | 86.22 / 6.00 |
| Zephyr Beta | 35.78 / 0.20 | 99.33 / 94.80 | 88.22 / 6.80 |
| Hermes 2 Pro | 19.33 / 1.00 | 98.00 / 91.80 | 83.33 / 2.40 |
| Qwen 1.5 32B | 80.67 / 3.00 | 84.44 / 80.60 | 86.67 / 3.20 |
| NeuralDaredevil | 25.78 / 2.40 | 99.11 / 98.60 | 83.33 / 3.00 |

- Unconditional steering raises harmless refusal to 46-99%; CAST keeps it
  near base (0-7%) at the cost of some harmful-refusal rate versus the
  unconditional vector.
- Category conditions work better for semantically distinct categories
  (hate speech better than legal opinion). Conditional steering beats a
  prompting baseline in most categories. Condition quality saturates with
  more data; extraction time grows linearly.

## Limitations

Strong hyperparameter dependence (useful theta ranges differ per model, e.g.
0.0-0.1 vs 0.4-0.6); imperfect behavior vectors still give inconsistent
refusals when the condition fires; conflicting rules; overlapping categories
are hard; one decision per prompt (no change during generation).

## Implementation details

Library `activation-steering` (IBM), names verified in `docs/quickstart.md`:
- `SteeringDataset(tokenizer, examples, suffixes, disable_suffixes)`.
- `SteeringVector.train(model, tokenizer, steering_dataset, method,
  accumulate_last_x_tokens)`; `method` in `pca_pairwise` (README: default,
  closest to the paper), `pca_center`, `pca_diff`; `accumulate_last_x_tokens`
  `"suffix-only"` (behavior vectors) or `"all"` (condition vectors).
- `MalleableModel.steer(behavior_vector, behavior_layer_ids,
  behavior_vector_strength, condition_vector, condition_layer_ids,
  condition_vector_threshold, condition_comparator_threshold_is)`.
- `MalleableModel.multisteer(..., rules)` for composed conditions.
- `MalleableModel.find_best_condition_point(positive_strings,
  negative_strings, condition_vector, layer_range, max_layers_to_combine,
  threshold_range, threshold_step, ...)` returns layer, threshold, direction.
- Layer notation like "10-15_interval2" = [10, 12, 14].
Environment in the paper: A100 80GB, PyTorch 2.3.0, transformers 4.43.3.
Note: the paper text describes mean-centering + PCA; the library's
recommended `pca_pairwise` may differ in detail (unverified).

## Related conditional steering (from the papers read here)

- **SEAL** applies its vector only at `\n\n` thought-boundary tokens - a
  position condition rather than a prompt condition.
- **Controlling Thinking Speed** (thinking-speed-lin-2025.md) switches
  strength *per token* using an early-vs-final-layer divergence signal.
- **Semantic specificity audit** (semantic-specificity-steering-reasoning.md)
  shows that a perfect gate can hide that a vector is only a label bias:
  zero side effects under oracle gating, collapse when ungated on competent
  inputs. Gating is not evidence that the vector is specific.
- Other 2025-2026 conditional-steering follow-ups were not located by search
  in this pass; none are summarized.
