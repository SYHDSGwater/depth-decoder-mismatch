# Decision Log

Append-only.

## 2026-09-11 — Use decoder-family regret, not verbalizability
Reason: LOTUS directly demonstrates that looped latent states can remain readable by the base LM head. The weaker marginal-bottleneck hypothesis remains testable via held-out loss regret.

## 2026-09-11 — Make the earlier decoder-compensation idea an explicit competing hypothesis
The earlier discussion predicted a rich decoder may help shallow loop states more, implying `dR/dT < 0`. This is not reconciled away; it is H2 and gives the study a clean opposite-sign outcome.

## 2026-09-11 — Refit the linear head
Primary Head Regret uses a refit linear head versus a nested richer head. The untouched checkpoint LM head is diagnostic only, avoiding a joint-training-history confound.

## 2026-09-11 — Ouro first, Nanbeige second, matched vocab study third
Within-checkpoint depth must show a stable signal before paying for a causal vocab × depth pretraining sweep.

## 2026-09-11 — Split vocabulary causality into output-space and tokenizer interventions
Before changing the tokenizer, directly manipulate only the LM-head output dimension by adding trainable never-target classes while keeping tokenizer, token IDs, targets, input embeddings, data order and backbone fixed. This follows the causal-control logic of Murugan (2026) while adding recurrent depth as an interacting factor. The natural ~16K-vs-~64K tokenizer experiment is moved to EXP-004.

## 2026-09-11 — Make active-vocabulary CE the strong EXP-003 endpoint
Raw CE under dummy-class inflation contains a mechanical denominator / competition penalty. EXP-003 therefore decomposes raw CE into active-vocabulary conditional CE and dummy competition cost. A positive depth×V interaction in active-vocabulary CE is the primary evidence for harmful output-space inflation; a raw-only interaction is interpreted as weaker competition/suppression cost.
