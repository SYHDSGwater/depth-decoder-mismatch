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
Before changing the tokenizer, directly manipulate only the LM-head output dimension by adding trainable never-target classes while keeping tokenizer, token IDs, targets, input embeddings, data order and backbone fixed. This follows the causal-control logic of Murugan (2026) while adding recurrent depth as an interacting factor. The natural tokenizer experiment is separate.

## 2026-09-11 — Make active-vocabulary CE the strong output-space endpoint
Raw CE under dummy-class inflation contains a mechanical denominator / competition penalty. Decompose it into active-vocabulary conditional CE and dummy competition cost. A positive depth×V interaction in active-vocabulary CE is the strong evidence; a raw-only interaction is weaker competition/suppression evidence.

## 2026-09-11 — Add EXP-001b frozen-native-head audit before accepting the deep null
EXP-001 pilot and 1M place essentially all rich-head gain at T=1, while T=2..4 select step 1 for both linear and rich full-head refits and even slightly underperform the untouched native head. This creates an optimization-drift confound. EXP-001b therefore freezes `W_native`, compares matched linear and nonlinear residual adapters, includes step 0 in model selection and uses dense early validation. The primary expressivity endpoint is `G_nonlin = CE_linear_residual - CE_nonlinear_residual`.

## 2026-09-11 — Reframe Phase B around real Ouro continued pretraining
EXP-001b still shows decoder nonlinearity gain collapsing strongly with native depth: T1 substantial, T2 small, T3/T4 near zero. Therefore do not motivate small vocab as a remedy for increasing forward decoder expressivity pressure in Ouro.

The next causal question is instead whether a larger **output space** creates a depth-dependent optimization burden even when deeper states remain linearly readable. Use the real `ByteDance/Ouro-1.4B` checkpoint first rather than immediately training a small imitation model from scratch.

`EXP-003A` is now the primary Phase-B screen:
- force T=1 vs T=4;
- keep tokenizer/input vocabulary at 49,152;
- compare V_out=49,152 vs 98,304 by appending 49,152 trainable never-target rows;
- continue training the full model on paired data;
- use one final-step raw-softmax LM loss per token in every arm;
- primary endpoint is persistent `I_active` at a fixed 50M-token endpoint.

The previous small Ouro-style from-scratch experiment becomes `EXP-003B` and is gated on a meaningful EXP-003A result. This separates a cheap, architecture-faithful pretrained intervention from the more expensive question of whether the interaction is intrinsic from random initialization.

## 2026-09-11 — Use paired-clone dummy rows in EXP-003A primary intervention
For V_out=98,304, initialize each dummy output row as a one-to-one clone of one active row. This creates exact step-zero structure: `m_dummy=0.5`, `CE_competition=log 2`, and unchanged active-renormalized CE. It avoids random dummy-logit scale as a confound. Random-distribution dummy initialization is secondary robustness only.

## 2026-09-11 — Match supervision count across recurrent depths in EXP-003A
Force fixed recurrent depth and use only the final-step LM loss in both T=1 and T=4 primary arms. Do not apply intermediate-loop LM losses in T=4. Otherwise deeper arms would receive more output-head supervision and the depth intervention would be entangled with objective multiplicity.
