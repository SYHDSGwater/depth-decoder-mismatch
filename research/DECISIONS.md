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
