# EXP-003B — From-scratch output-vocabulary inflation × recurrent depth replication

## Role
This is the from-scratch replication of EXP-003A. Run only if EXP-003A shows a stable nonzero depth × output-vocabulary interaction worth testing outside the pretrained-checkpoint regime.

EXP-003A uses the real `ByteDance/Ouro-1.4B` checkpoint and continued pretraining. EXP-003B asks whether the same interaction appears when the looped model is trained from random initialization, removing the checkpoint-adaptation confound.

## Question
Holding tokenizer, raw data, target IDs, input vocabulary and non-output architecture fixed, does larger output-only vocabulary causally impose a larger modeling/optimization penalty at deeper recurrent computation from scratch?

## Primary design
Train a controlled small Ouro-style LoopLM with one fixed BPE tokenizer:

- `V_active = 16,384`;
- input vocabulary fixed at 16,384 in all arms;
- output vocabulary `V_out = 16,384` control vs `49,152` inflated;
- optional stress condition `V_out = 131,072`;
- recurrent depth T=1 vs T=4 primary, T=2 secondary;
- same raw corpus, token IDs, target IDs, batch order, optimizer family and token budget across arms;
- extra output rows are trainable, appear in the raw softmax denominator and are never valid targets.

Use an untied LM head so output vocabulary can change without resizing the input embedding table.

## Output-row initialization
Use the same intervention logic as EXP-003A where possible:
- active rows paired identically across V_out arms within each seed;
- dummy rows initialized by a pre-registered policy;
- prefer paired-clone initialization if dimensions permit, with a random-distribution robustness check only after the primary result is frozen.

## Recurrent-depth intervention
Use one shared recurrent block / Ouro-style shared-weight architecture. Unique backbone parameters and hidden width are identical across depths. Force fixed T and use one final-step LM loss in the primary experiment so T=4 does not receive extra supervised exits relative to T=1.

## Loss decomposition
For inflated arms define:

`m_dummy = sum_{j in dummy} p_j`

`p_active(y) = p_raw(y) / (1 - m_dummy)`

`CE_raw = -log p_raw(y)`

`CE_active = -log p_active(y)`

`CE_competition = CE_raw - CE_active = -log(1 - m_dummy)`.

Primary evidence is the active-token interaction:

`I_active = [CE_active(T_hi,V_hi)-CE_active(T_hi,V_base)] - [CE_active(T_lo,V_hi)-CE_active(T_lo,V_base)]`.

A raw-only interaction is weaker evidence because it may be explained by dummy-class probability mass.

## Pairing and tuning
- paired random seeds across all arms;
- same backbone/input-embedding/active-output initialization within each seed;
- same document order and batches;
- same token budget;
- tune LR on control arms first, then share the selected LR across V_out within each T;
- >=3 paired report seeds; >=5 preferred if compute permits;
- report a shared-LR sensitivity analysis.

## Primary outcomes
Report for each arm:
- validation/test `CE_raw`;
- `CE_active`;
- `CE_competition`;
- dummy probability mass distribution;
- active/dummy row norms;
- training curves;
- per-seed and paired document-bootstrap interactions.

Primary causal statistic:

`I_active(T=1 vs T=4, V_out=16K vs 49K)`.

## Interpretation
| Result | Interpretation |
|---|---|
| `I_active > 0`, robust across seeds | Strong from-scratch support that pure output-space inflation becomes more harmful with recurrent depth. |
| `I_raw > 0`, `I_active ≈ 0` | Depth-dependent softmax competition/suppression cost without clear active-token modeling damage. |
| both interactions ≈0 | Weakens pure output-class-count / V-D explanations. |
| interaction <0 | Opposite of the proposed mechanism. |

## Relation to EXP-003A
EXP-003A is the primary causal screen because it is much cheaper and directly uses the real Ouro architecture/checkpoint already studied in EXP-001/001b.

EXP-003B is necessary only if we need to establish that any positive EXP-003A effect is not merely a continued-pretraining adaptation phenomenon around a checkpoint originally trained with `V=49,152`.

## Relation to EXP-004
EXP-004 changes the tokenizer itself. That simultaneously changes sequence length, token frequency, compositional structure and output vocabulary. Therefore EXP-004 tests a broader natural-tokenizer mechanism and must not be conflated with EXP-003A/B's output-only intervention.

## Invalidity / overclaim guards
- same tokenizer/token IDs across V_out arms;
- input embedding vocabulary fixed;
- dummy IDs never appear as inputs or targets;
- one final-step loss per token for primary T comparison;
- active rows paired at initialization;
- raw CE alone is not sufficient for the strong claim;
- do not claim a positive result proves a naturally smaller tokenizer is better;
- do not infer Claude design motivation.

## Result
Pending.
