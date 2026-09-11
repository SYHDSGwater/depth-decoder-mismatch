# EXP-003C — Tiny recurrent × output-vocabulary causal control

## Role
**Final low-cost causal test for this project.**

EXP-003A (Ouro continued pretraining) is scientifically direct but costs tens of H100-hours. EXP-003B (from-scratch 16K-vocab LoopLM) is also more expensive than justified by the current evidence. EXP-003C asks the minimal remaining causal question at Murugan-scale compute:

> Murugan (2026) found that adding never-target output classes does not impair learning in a small ordinary Transformer. Does weight-shared recurrent depth create a `depth × output-space` interaction that is absent without recurrence?

If EXP-003C is null, the output-class-count / softmax-bottleneck branch of this project should stop. EXP-003A/B remain future-work designs rather than experiments required for the current project.

## Prior evidence being extended
Murugan's byte-level WikiText-2 control uses a four-layer Transformer with hidden width 32. Keeping the 256-byte input/target vocabulary fixed while increasing output classes from 256 to 1,024 and 4,096 did not worsen tuned validation CE across five seeds. The reported means were approximately:

- `V_out=256`: 2.2591;
- `V_out=1024`: 2.2532;
- `V_out=4096`: 2.2642.

Thus pure output-class inflation was null in the non-recurrent setting. EXP-003C changes only one conceptual axis: the same Transformer body is applied recurrently with shared weights.

## Primary question
Does pure output-vocabulary inflation become more harmful when the same Transformer computation is recurrently repeated?

Define

`P_active(T) = CE_active(T, 4096) - CE_active(T, 256)`

and the primary interaction

`I_active = P_active(T=4) - P_active(T=1)`.

The remaining recurrent-output-space hypothesis predicts

`I_active > 0`.

The null is

`I_active ~= 0`.

## Architecture
Start from the compact Murugan-style byte-level Transformer recipe and make the minimum recurrent modification.

### Fixed dimensions
- dataset/tokenization: WikiText-2, byte-level;
- active input/target vocabulary: `V_active = 256`;
- hidden width: `D = 32`;
- Transformer body: 4 layers;
- causal self-attention and FFN dimensions/norm/position recipe: inherit the public Murugan `response_vocab` configuration wherever possible;
- untied input embedding and output head;
- no adaptive exit gate;
- no intermediate-loop supervision.

### Weight-shared recurrence
Let `B_theta` be the complete 4-layer Transformer body. Add token/position embeddings once, then apply the same body recurrently:

`h^(0) = embed(x)`

`h^(t) = B_theta(h^(t-1)),  t=1..T`

`logits = W_out norm(h^(T))`.

All parameters inside `B_theta` are shared across recurrent passes. Parameter count is therefore identical for T=1 and T=4 except for the intended output-head-size intervention.

If the inherited implementation uses absolute positional embeddings, add them only at `h^(0)`; do not re-add them on every recurrent pass. If it uses position information inside attention (e.g. RoPE), preserve that mechanism identically on every pass.

## Primary 2 × 2 design

| Arm | Recurrent depth | Output classes | Never-target classes |
|---|---:|---:|---:|
| A | T=1 | 256 | 0 |
| B | T=1 | 4096 | 3840 |
| C | T=4 | 256 | 0 |
| D | T=4 | 4096 | 3840 |

Optional `V_out=1024` is **secondary only** and should not be added until the primary 2×2 result is frozen.

## Pairing and initialization
For each seed:
- A/B/C/D use the same initial input embeddings and the same initial recurrent-body parameters;
- active output rows `W[0:256]` are exactly identical across all arms;
- the 4096-class head appends 3840 independently initialized dummy rows using the same row-initialization distribution as active rows;
- dummy IDs never occur in inputs or labels;
- batch order and data examples are paired across all four arms.

### Required step-zero checks
For each T pair before training:
- hidden states are identical between 256- and 4096-output arms;
- active logits are identical;
- active-renormalized CE is identical within numerical tolerance;
- labels are always `<256`;
- dummy rows receive no positive-target labels.

Raw CE need not be identical because dummy classes enter the softmax denominator.

## Training objective
Use exactly one loss per token from the final recurrent state:

`L_train = CE_raw(softmax(logits over V_out), target)`.

Do **not** supervise intermediate recurrent states. Otherwise T=4 would receive four LM losses while T=1 receives one, confounding recurrence with supervision count.

Train the entire model, including:
- input embeddings;
- recurrent Transformer body;
- active LM-head rows;
- dummy LM-head rows in inflated arms.

## Training budget
### Primary budget
Match the compact Murugan regime as closely as practical:
- `600 optimizer steps` per run;
- same sequence length, batch size, optimizer family, scheduler and data construction as the public `response_vocab` recipe unless explicitly overridden below;
- five paired report seeds: `[1,2,3,4,5]` (or the exact five public Murugan seeds if different; freeze before results);
- no early stopping; all arms consume exactly the same number of supervised tokens for a given seed.

This yields 20 primary fits (`2 depths × 2 output sizes × 5 seeds`). At this model size, compute should be well below the Ouro-scale experiments and is intended to be sub-H100-hour rather than tens of H100-hours.

### Learning-rate selection
Do not tune the 4096-output arms independently in the primary comparison.

For each depth T separately:
1. use a dedicated tuning seed not in the five report seeds;
2. select LR using only the `V_out=256` control;
3. freeze that LR;
4. use the same LR for `V_out=256` and `4096` report arms at that T.

Use the public Murugan LR as the center of a small 3-point grid if the recurrent T=4 model needs retuning. The grid must be frozen before report-seed results.

### Tuned-arm robustness
Only if the primary experiment produces a positive `I_active`, run a secondary per-arm LR check to rule out a trivial optimizer-mismatch explanation. Do not pay this cost after a clear null.

## Evaluation and loss decomposition
For every validation example in a 4096-output arm, let

`m_dummy = sum_{j=256}^{4095} p_j`.

Define

`p_active(y) = p_raw(y) / (1 - m_dummy)`

`CE_raw = -log p_raw(y)`

`CE_active = -log p_active(y)`

`CE_competition = CE_raw - CE_active = -log(1-m_dummy)`.

For 256-output arms, `CE_raw = CE_active` and `CE_competition = 0`.

### Evaluation checkpoints
Evaluate the fixed validation set at:

`step = [0, 20, 50, 100, 200, 400, 600]`.

The final primary endpoint is step 600. Earlier checkpoints are trajectory diagnostics only and must not be used to select the endpoint.

## Primary statistics
At step 600 compute, for each paired seed:

`P_active(1) = CE_active(T1,4096) - CE_active(T1,256)`

`P_active(4) = CE_active(T4,4096) - CE_active(T4,256)`

`I_active = P_active(4) - P_active(1)`.

Also report:

`P_raw(T)` and `I_raw = P_raw(4)-P_raw(1)`

plus mean `m_dummy` / `CE_competition`.

### Uncertainty
Primary uncertainty is across the five paired training seeds:
- mean paired `I_active`;
- two-sided 95% Student-t confidence interval;
- all five individual seed interactions;
- paired effect size `d_z` as secondary reporting.

Do not use individual tokens as the independent unit for the training-seed causal claim.

## Pre-registered interpretation

### Outcome A — null interaction
`I_active ~= 0` and no consistent positive seed-level effect.

Interpretation: Murugan's never-target-vocabulary null survives weight-shared recurrence at this controlled scale. Together with EXP-001/001b, this strongly weakens the project's proposed `small vocab -> relieve recurrent softmax/decoder bottleneck` mechanism. **Stop this topic.** EXP-003A/B become future work only.

### Outcome B — raw-only interaction
`I_raw > 0` but `I_active ~= 0`.

Interpretation: recurrence changes dummy-class suppression / softmax competition but does not measurably harm learning among real symbols. This is not sufficient support for the small-vocab optimization hypothesis. **Stop this topic unless the systems cost itself becomes the research question.**

### Outcome C — positive active interaction
`I_active > 0`, consistent across seeds, with a 95% paired-seed interval excluding zero.

Interpretation: pure output-space inflation becomes more harmful under recurrent computation in this tiny controlled setting. This is the only result that justifies revisiting an expensive Ouro-scale EXP-003A in future work.

### Outcome D — negative interaction
`I_active < 0`.

Interpretation: deeper recurrence absorbs output-space inflation at least as well as shallow computation, opposite to the proposed mechanism. **Stop this topic.**

## Sanity controls
Required:
- T=1, V=256 should approximately reproduce the public compact-model baseline after accounting for implementation/environment differences;
- parameter count excluding output-head rows must be identical across all arms;
- T=1/T=4 parameter tensors have identical shapes and initialization within each seed;
- 4096-arm active rows exactly match the 256-arm head at step 0;
- active CE equality at step 0 must pass before training;
- same train/validation text and batch ordering across paired arms.

Optional smoke-only control:
- mask dummy logits to `-inf` in a 4096-shaped head. It should reproduce the 256-class objective up to numerical precision.

## Why this experiment is worth doing
It adds exactly one missing factor to an existing causal null: recurrent depth. Unlike EXP-003A, it does not try to establish billion-scale effect size. Its value is hypothesis triage:

- null -> terminate the output-space branch cheaply;
- positive -> demonstrate that recurrence changes the causal effect and motivate future scale-up.

## What this experiment cannot establish
- that natural tokenizer vocabulary size behaves like never-target classes;
- that Ouro-1.4B has the same interaction;
- that Claude's ~16K vocabulary was chosen for this reason;
- that any effect at tiny scale survives long pretraining.

## Result
Pending.
