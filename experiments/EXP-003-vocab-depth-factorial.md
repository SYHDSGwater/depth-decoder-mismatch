# EXP-003 — Controlled output-vocabulary inflation × recurrent depth

## Motivation
This is the direct causal vocabulary-size test, adapted from Murugan (2026), *Does the LM Head Create a Harmful Gradient Bottleneck? A Causal Test*.

Changing a tokenizer changes tokenization, sequence length, token frequency and data exposure. EXP-003 therefore does **not** change the tokenizer. It keeps the text, targets, input embeddings and backbone fixed, and changes only the number of output softmax classes by appending classes that are never correct targets.

The new ingredient relative to Murugan is recurrent depth. The key question is not merely whether larger `V` hurts, but whether its causal penalty grows with loop depth `T`.

## Question
Does increasing output dimension `V_out` causally impose a larger modeling/optimization penalty at deeper recurrent computation?

Formally, define the vocabulary-inflation penalty

`P(T, V) = L(T, V) - L(T, V_base)`.

The depth–vocab interaction is

`I_VT = [L(T_hi,V_hi)-L(T_hi,V_base)] - [L(T_lo,V_hi)-L(T_lo,V_base)]`.

The simple small-vocab/DDM mechanism predicts `I_VT > 0`.

## Primary design: small controlled Ouro-style LoopLM from scratch

### Fixed tokenizer and data
- Train one BPE tokenizer with `V_base = 16,384` on a frozen tokenizer-training corpus.
- Use the **same tokenizer in every arm**.
- Use the same FineWeb-Edu raw documents, token IDs, targets, sequence length, batch order and number of supervised tokens in every arm.
- Input embedding vocabulary remains exactly 16,384 in every arm.
- Output embedding / LM head is untied from the input embedding.

### Output-vocabulary intervention
Primary output dimensions:
- `V_out = 16,384` — control;
- `V_out = 49,152` — 3× output space, chosen to approximate the 16K ↔ 49K contrast motivating the Claude discussion.

Secondary stress condition:
- `V_out = 131,072`.

For `V_out > V_base`, append `V_out - V_base` output rows. These extra classes:
- appear in the softmax denominator;
- are trainable;
- are **never valid targets**;
- never enter the input sequence or input embedding table.

Initialize active-token rows identically across matched arms. Initialize dummy rows from the same row-norm / variance distribution as active rows; record the exact initialization policy and pair it by seed.

### Recurrent-depth intervention
Primary depths:
- `T = 1`;
- `T = 4`.

Secondary curve:
- `T = 2`.

Use one shared recurrent block / Ouro-style shared-weight architecture. Unique backbone parameters and hidden width are identical across T. Depth changes repeated computation, not parameter count.

A practical pilot configuration is a downsized Ouro-style model (e.g. hidden width around 512 with a small shared stack). The exact small-model width/layer count must be frozen before confirmatory runs; the scientific invariant is that it is identical across every `(T, V_out)` arm.

## Pairing and tuning
- Paired initialization seeds across all arms.
- Same backbone, input embedding and active output-row initialization within each seed.
- Same document order, batches, optimizer family, token budget and scheduler family.
- Use one tuning seed to screen learning rates for each arm, then freeze hyperparameters before confirmatory seeds.
- Confirm with >=5 paired seeds if feasible; minimum acceptable confirmatory set is 3 paired seeds.
- Also report a shared-LR sensitivity analysis so arm-specific tuning cannot manufacture the interaction.

## Loss decomposition: do not confuse dummy competition with model degradation
Raw CE is not sufficient because extra classes mechanically enlarge the softmax denominator.

For one example let

`m_dummy = sum_{j in dummy} p_j`.

For the true active token `y`, define conditional active-vocab probability

`p_active(y) = p(y) / (1 - m_dummy)`.

Then

`CE_raw = -log p(y)`

`CE_active = -log p_active(y)`

and exactly

`CE_raw - CE_active = -log(1 - m_dummy)`.

Report all three quantities:
1. `CE_raw` — actual model loss with the enlarged softmax;
2. `CE_active` — loss after renormalizing probability over the original 16K active classes;
3. `CE_competition = CE_raw - CE_active` — probability mass wasted on dummy classes.

This decomposition is central to interpretation.

### Strong evidence for harmful vocabulary inflation
A positive depth × vocabulary interaction in **`CE_active`**, not merely `CE_raw`:

`I_active = [CE_active(T_hi,V_hi)-CE_active(T_hi,V_base)] - [CE_active(T_lo,V_hi)-CE_active(T_lo,V_base)] > 0`.

This means the larger output space has impaired learning/modeling among the real tokens, after removing the trivial denominator penalty.

### Weak evidence / competition-only result
If `I_raw > 0` but `I_active ≈ 0`, and the difference is explained by `CE_competition`, then deeper models are only slower/worse at suppressing never-target classes. That is an output-competition cost, not strong evidence that the linear decoder cannot represent the active-token distribution.

## Primary outcomes
For every `(T, V_out)` arm report:
- validation `CE_raw`;
- validation `CE_active`;
- `CE_competition`;
- mean / quantiles of `m_dummy`;
- training loss curves;
- active-row and dummy-row norm statistics;
- validation BPB may be reported, but because tokenization is fixed it is not required for cross-arm comparability.

Primary causal statistic:
- `I_active` for `T=1` vs `T=4`, `V_out=16,384` vs `49,152`.

Secondary statistics:
- `I_raw`;
- corresponding interaction for `CE_competition`;
- T=1,2,4 trend;
- 131K output-space stress test.

Use paired-seed differences and two-sided 95% intervals. Do not use individual training tokens as independent statistical units.

## Projection diagnostics — secondary only
Optionally track:
- retained logit-gradient norm through the LM head;
- angle between original and projected logit gradients;
- LM-head singular spectrum / condition number;
- gradient norms for active vs dummy output rows.

These are mechanism diagnostics, **not** primary evidence. Murugan (2026) found that strong projection geometry did not reliably predict learning progress.

## Interpretation table

| Result | Interpretation |
|---|---|
| `I_active > 0`, robust across seeds | Strong support that pure output-space inflation becomes more harmful with recurrent depth; supports the direct small-vocab mechanism. |
| `I_raw > 0`, `I_active ≈ 0` | Larger V creates depth-dependent softmax competition / dummy suppression cost, but not clear active-token modeling damage. |
| `I_raw ≈ 0`, `I_active ≈ 0` | Weakens the simple claim that output class count or `V/D` alone explains a small-vocab advantage. |
| interaction < 0 | Opposite of the proposed DDM-vocab mechanism; deeper recurrence may suppress/absorb the extra-output burden better. |

## Important limitation
Never-target classes isolate output dimensionality, but they do not reproduce a natural larger tokenizer. They do not increase the rank/complexity of the *target* distribution over genuinely used symbols. Therefore:
- a null EXP-003 strongly weakens a **pure output-dimension / competition / gradient-geometry** explanation;
- it does not prove that natural vocabulary size is irrelevant;
- natural 16K-vs-large-tokenizer effects are tested separately in EXP-004.

## Optional Ouro-1.4B bridge experiment
Before full from-scratch confirmation, a cheap diagnostic can reuse EXP-001 hidden states:
- freeze Ouro-1.4B backbone;
- exploit `tie_word_embeddings=false`;
- append never-target rows to a refit LM head;
- compare output-inflation penalty over native `T=1..4` on exactly paired hidden states.

This is a decoder-only diagnostic, not a substitute for end-to-end training, but it can estimate effect size and choose useful `V_out` levels before the controlled pretraining sweep.

## Invalidity / overclaim guards
- Do not change tokenizer or tokenized sequences across `V_out` arms.
- Do not resize the input embedding table when adding output-only classes.
- Do not allow dummy IDs to appear as targets.
- Do not interpret raw-CE degradation alone as an expressivity bottleneck without the active-CE decomposition.
- Do not claim EXP-003 proves a natural 16K tokenizer is better than a 49K tokenizer.
- Do not infer Claude's design motivation from a positive result.

## Result
Pending.
