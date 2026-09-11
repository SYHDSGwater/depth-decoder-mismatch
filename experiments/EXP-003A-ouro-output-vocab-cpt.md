# EXP-003A — Ouro continued-pretraining output-vocabulary intervention

## Research link
- Parent question: does pure output-space inflation causally become more harmful as recurrent depth increases?
- Model: `ByteDance/Ouro-1.4B`
- Status: proposed
- Role: primary low-cost causal screen before any from-scratch replication.

## Motivation
EXP-001 and EXP-001b do **not** support the forward-expressivity version of depth–decoder mismatch in Ouro: nonlinear decoder gain collapses as native recurrent depth increases. The remaining small-vocabulary hypothesis should therefore be tested separately as an **optimization / output-space** question rather than as a claim that deeper hidden states become unreadable by a linear LM head.

EXP-003A directly manipulates only the number of output softmax classes during continued pretraining of the real Ouro checkpoint. Tokenizer, input token IDs, targets, input embeddings, data order and backbone architecture remain fixed. Additional output classes are trainable but are never valid targets.

The key question is not whether a larger softmax mechanically increases raw loss, but whether it causes collateral degradation of learning among the original 49,152 active tokens, and whether that degradation is stronger at larger recurrent depth.

## Core causal design

### Fixed model and tokenizer
Use the same pinned `ByteDance/Ouro-1.4B` checkpoint/tokenizer revision as EXP-001.

Native active vocabulary:

`V_active = 49,152`.

Ouro has untied input embeddings and LM-head weights, so output vocabulary can be expanded without resizing or changing the input embedding table.

### Primary 2 × 2 factorial

| arm | recurrent depth | output classes | description |
|---|---:|---:|---|
| A | T=1 | 49,152 | shallow control |
| B | T=1 | 98,304 | shallow inflated output space |
| C | T=4 | 49,152 | deep control |
| D | T=4 | 98,304 | deep inflated output space |

`T=1` and `T=4` are the primary native depths. `T=2` is optional secondary interpolation only after the primary 2×2 result is frozen.

The tokenizer and target IDs remain in `[0, 49151]` in **every** arm.

### What changes in inflated arms
For `V_out = 98,304`, append 49,152 dummy LM-head rows. Dummy classes:
- are included in the raw softmax denominator;
- are trainable;
- never appear in the input sequence;
- never appear as labels/targets;
- have optimizer state like ordinary output rows.

The original 49,152 active LM-head rows are copied bit-for-bit from the Ouro checkpoint and are identical across paired control/inflated arms at step 0.

## Dummy-row initialization

### Primary initialization — paired clones
For every active row `W_j`, initialize one dummy row as an exact clone under a fixed one-to-one mapping:

`W_dummy[j] = W_active[j]`.

Therefore at step 0 every active logit has one identical dummy logit. For the inflated model:

`CE_raw(step0) = CE_active(step0) + log(2)`

up to numerical error, while

`CE_active(step0, 98K) = CE_active(step0, 49K)`.

This gives an exact, deterministic causal starting point and removes random dummy-row norm/logit-distribution confounds.

### Secondary initialization robustness
Only after the primary result is frozen, optionally repeat one seed with dummy rows sampled from the active-row empirical norm/variance distribution. Do not replace the paired-clone primary intervention based on which initialization gives the preferred result.

## Training objective and recurrent-depth control

### Force fixed recurrent depth
Disable adaptive exit / routing for this experiment and force exactly `T` recurrent passes in each arm.

### Use one final-step loss
To isolate recurrent depth from the number of supervised exits, train every arm with exactly one next-token cross-entropy loss computed from the final hidden state at the forced depth:

`L_train = CE_raw(z^(T), y)`.

Do not add intermediate-loop LM losses in the primary experiment. Otherwise T=4 would receive more output-head supervision than T=1 and depth would no longer be the only recurrent-computation intervention.

Adaptive-exit/gating parameters that are unused under fixed-T training should be excluded from the optimizer.

### Trainable parameters
In all four arms, continued pretraining updates:
- recurrent backbone parameters;
- input embeddings;
- normalization parameters;
- original 49,152 active LM-head rows;
- dummy LM-head rows in inflated arms.

Do **not** freeze the backbone or active output rows in the primary experiment. If they were frozen, `CE_active` could not reveal collateral optimization damage caused by the enlarged output space.

## Data
Use a fresh FineWeb-Edu sample that does not reuse the exact EXP-001 1M documents/windows.

Requirements:
- same raw documents across all four arms;
- same tokenizer and token IDs;
- same packed sequences;
- same batch order per paired seed;
- `seq_len = 1024`;
- identical supervised-token count in all arms;
- document-level train/validation/test separation;
- record document IDs, source shard IDs and hashes.

Prefer source shards not used by EXP-001. Freeze the corpus manifest before any training result is inspected.

## Staged budget

### Smoke
Purpose: implementation and decomposition checks only.

- 1 paired seed;
- 1–2M training tokens;
- all four primary arms;
- verify step-0 parity and exact loss decomposition;
- no hypothesis conclusion.

### Primary screen
- training budget: **50M supervised tokens per arm**;
- report seeds: **11, 29, 47**;
- fixed sequence length 1024;
- fixed global batch of **32 sequences = 32,768 tokens/update** unless hardware constraints require a pre-registered microbatch/accumulation decomposition;
- approximately 1,526 optimizer updates for 50M tokens;
- no early stopping; every arm consumes the same token budget.

If resource constraints require a smaller first launch, 20M tokens/arm is allowed as an explicitly labeled pilot, but the 50M recipe must remain frozen before inspecting the 20M test result.

## Optimizer and LR policy
Because this is continued pretraining from a mature checkpoint, use a conservative LR range.

### LR selection
For each recurrent depth separately, choose LR using the **49K control arm only** on validation data:

`lr in {1e-6, 3e-6, 1e-5}`.

Use tuning seed 83. Freeze the selected LR for that depth and apply the **same LR** to both 49K and 98K arms for all report seeds.

This prevents the inflated arm from receiving special retuning that could hide the optimization penalty being measured.

### Other optimizer invariants
- AdamW;
- same betas/eps across arms;
- same weight decay across arms;
- same scheduler shape across arms;
- same warmup fraction across arms;
- same gradient clipping policy across arms;
- same precision / loss scaling policy across arms;
- same batch order and RNG seed for paired 49K/98K arms at a given T.

A reasonable default is short linear warmup followed by cosine decay, but the exact scheduler must be frozen in config before the primary screen.

## Loss decomposition
For an inflated arm, let

`m_dummy = sum_{j in dummy} p_j`.

For a real target `y < 49152`, define

`p_active(y) = p_raw(y) / (1 - m_dummy)`.

Then report:

`CE_raw = -log p_raw(y)`

`CE_active = -log p_active(y)`

`CE_competition = CE_raw - CE_active = -log(1 - m_dummy)`.

For 49K control arms:

`CE_raw = CE_active`, `CE_competition = 0`.

### Step-zero invariants
Before the first optimizer update, verify on the same validation batch:

1. active logits of paired 49K and 98K arms are bitwise/effectively identical;
2. `CE_active_98K == CE_active_49K` within numerical tolerance;
3. paired-clone initialization yields `CE_raw_98K - CE_active_98K == log(2)` within numerical tolerance;
4. `m_dummy == 0.5` within numerical tolerance for paired clones;
5. model hidden states before the LM head are identical within each T pair.

Failure of any step-zero invariant invalidates the run.

## Evaluation schedule
Use a fixed validation set and evaluate at cumulative supervised-token counts:

`S = {0, 1M, 2M, 5M, 10M, 20M, 35M, 50M}`.

At every S report:
- validation `CE_raw`;
- validation `CE_active`;
- validation `CE_competition`;
- mean / median / p90 / p99 `m_dummy`;
- active-row and dummy-row weight norms;
- gradient norms for backbone, active rows and dummy rows;
- tokens/sec and peak memory as systems diagnostics.

The validation trajectory is descriptive/mechanistic. The primary test set is evaluated **only at the fixed 50M endpoint** for the primary screen.

## Primary causal statistics
Define the active-vocabulary inflation penalty at training-token count `s`:

`P_active(T,s) = CE_active(T,98K,s) - CE_active(T,49K,s)`.

Primary endpoint at 50M:

`I_active(50M) = P_active(T=4,50M) - P_active(T=1,50M)`.

The small-vocab / depth-dependent optimization hypothesis predicts:

`I_active(50M) > 0`.

Also report:

`P_raw(T,s) = CE_raw(T,98K,s) - CE_raw(T,49K,s)`

`P_comp(T,s) = CE_competition(T,98K,s)`

and

`I_raw(50M) = P_raw(4,50M) - P_raw(1,50M)`.

### Persistence analysis
Plot `I_active(s)` over the frozen validation checkpoints. This distinguishes:
- transient adaptation cost: positive early `I_active`, then decay toward zero;
- persistent optimization penalty: positive `I_active` maintained through 50M;
- no active-token damage: `I_active ≈ 0` despite raw competition cost.

Do not choose the final checkpoint based on this trajectory; 50M is fixed in advance.

## Statistical analysis
Use paired report seeds and paired evaluation examples.

For the final test endpoint:
- compute per-document mean CE differences;
- bootstrap documents, not tokens;
- preserve pairing across all four arms inside each bootstrap replicate;
- report two-sided 95% intervals for `P_active(1)`, `P_active(4)` and `I_active`;
- also report all per-seed interactions.

Do not treat individual tokens as IID observations.

## Sanity control — masked 98K head
Optional but strongly recommended for the smoke stage.

Construct the same 98K output matrix, but mask dummy logits to `-inf` before softmax so they contribute neither probability mass nor gradient.

This control has the same output tensor shape / extra parameters but should reproduce the 49K active objective. Run at least one seed for T=1 and T=4 during smoke.

If masked-98K diverges materially from 49K beyond numerical / systems effects, the implementation must be audited before the primary screen.

## Interpretation table

| Result | Interpretation |
|---|---|
| `I_active > 0`, persistent to 50M | Strong evidence that pure output-space inflation creates a depth-dependent optimization penalty in pretrained Ouro. |
| `I_active > 0` early, `≈0` by 50M | Enlarged output space slows adaptation but does not produce persistent active-token damage at this budget. |
| `I_raw > 0`, `I_active ≈ 0` | Depth-dependent cost is mainly dummy-class competition/suppression, not collateral active-token modeling harm. |
| `I_active ≈0`, `I_raw≈0` after adaptation | Strongly weakens a pure output-class-count explanation for the small-vocab hypothesis in Ouro. |
| `I_active < 0` | Opposite interaction: deeper recurrence is at least as able to absorb/suppress output inflation as shallow recurrence. |

## Relation to EXP-001 / EXP-001b
EXP-001/001b test whether deeper recurrent hidden states increasingly require a richer decoder family. Current evidence says no: decoder nonlinearity gain collapses with depth.

EXP-003A asks a different question:

> even if deep Ouro representations remain linearly readable, does a larger softmax output space make learning/optimization harder at larger recurrent depth?

A positive EXP-003A therefore supports an optimization/competition mechanism, **not** the rejected forward-expressivity story.

## Relation to EXP-003B and EXP-004
EXP-003A begins from a pretrained 49K-vocab checkpoint. Therefore it tests a continued-pretraining intervention, not from-scratch architectural optimality.

- If EXP-003A is clearly positive, run EXP-003B: controlled small LoopLM from scratch with output-only vocabulary inflation.
- If EXP-003A is null, EXP-003B becomes optional / lower priority.
- EXP-004 remains the natural-tokenizer experiment and is conceptually separate because changing tokenizer also changes sequence length, token frequency and compositional structure.

## Invalidity / overclaim guards
- Do not resize or change the input embedding vocabulary.
- Do not change tokenizer or token IDs across output-vocab arms.
- Dummy IDs must never appear as inputs or labels.
- 49K and 98K paired arms must start from identical backbone and active output rows.
- Train the inflated arm with raw full-softmax CE; training on active-renormalized CE would remove the intervention.
- Do not freeze the backbone/active rows in the primary experiment.
- Keep one final-step loss per token in every depth arm; no extra intermediate-loop supervision in T=4 primary runs.
- LR may differ by T only if selected from the 49K control and then shared across V_out within that T.
- Do not interpret raw CE alone as harmful bottleneck evidence.
- Do not claim a positive result proves that a naturally smaller tokenizer is better.
- Do not claim a positive result explains Anthropic/Claude design choices.
- EXP-003A is a pretrained-checkpoint intervention; causal conclusions are limited to that regime unless EXP-003B replicates from scratch.

## Target config sketch

```yaml
experiment: EXP-003A
model:
  name: ByteDance/Ouro-1.4B
  revision: <same pinned revision as EXP-001>
  fixed_depths: [1, 4]
  adaptive_exit: false
  final_step_loss_only: true
vocab:
  active_size: 49152
  output_sizes: [49152, 98304]
  dummy_init: paired_clone
  input_vocab_unchanged: true
data:
  dataset: HuggingFaceFW/fineweb-edu
  seq_len: 1024
  train_tokens: 50000000
  fresh_from_exp001: true
  paired_batch_order: true
training:
  report_seeds: [11, 29, 47]
  tuning_seed: 83
  lr_grid: [1.0e-6, 3.0e-6, 1.0e-5]
  global_sequences_per_step: 32
  optimizer: AdamW
  early_stopping: false
eval:
  validation_token_checkpoints: [0, 1000000, 2000000, 5000000, 10000000, 20000000, 35000000, 50000000]
  test_once_at_tokens: 50000000
  bootstrap_cluster: document_id
  bootstrap_replicates: 2000
```

## Result
Pending.
