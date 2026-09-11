# Claim ↔ Evidence Matrix

Do not promote a claim until its required evidence exists.

| ID | Candidate claim | Required evidence | Current status |
|---|---|---|---|
| C1 | Linear-vs-rich decoder regret changes systematically with native recurrent depth in Ouro. | Valid paired EXP-001 with robust slope CI across probe seeds/capacity. | observed in pilot/1M, non-confirmatory |
| C2 | Forward DDM: deeper recurrent states become increasingly under-exploited by a linear decoder. | Positive depth slope in decoder-family regret, robust to optimization controls. | **not supported by current Ouro evidence** |
| C3 | Recurrent depth reduces marginal nonlinear decoder gain / improves decoder alignment. | EXP-001 negative slope plus EXP-001b declining `G_nonlin(T)`, ideally independent/cross-architecture replication. | provisional support; T2 small positive gain, T3/T4 near zero |
| C4 | Decoder-gain collapse replicates across looped architectures. | Nanbeige or another looped model with the same native-depth trend. | untested |
| C5 | Pure output-space inflation creates a larger active-token optimization penalty at deeper recurrent depth in pretrained Ouro. | EXP-003A persistent positive `I_active` at fixed 50M endpoint with paired seeds/document bootstrap; raw-only interaction is insufficient. | untested |
| C6 | The EXP-003A interaction is not just checkpoint adaptation and replicates from scratch. | EXP-003B controlled small LoopLM from-scratch positive interaction. | untested |
| C7 | Natural larger tokenizer vocab causally amplifies a depth-related bottleneck. | EXP-004 matched tokenizer-vocab × depth training with BPB-normalized positive interaction. | untested |
| C8 | Claude's small vocab was chosen because of any mechanism in this project. | Proprietary causal/design evidence from Anthropic. | **not claimable** |

## EXP-001b guardrail
A nonlinear residual improving over the frozen native head is not sufficient decoder-expressivity evidence. The relevant quantity is:

`G_nonlin(T) = CE_linear_residual(T) - CE_nonlinear_residual(T)`.

Current Ouro evidence shows `G_nonlin` collapses with depth rather than increasing.

## EXP-003A guardrail
Strong evidence is the active-vocabulary interaction:

`I_active = [CE_active(T4,98K)-CE_active(T4,49K)] - [CE_active(T1,98K)-CE_active(T1,49K)]`.

- `I_raw > 0` with `I_active ≈ 0` supports only output competition / dummy suppression cost.
- Persistent `I_active > 0` supports collateral active-token optimization damage that grows with recurrent depth in the pretrained Ouro CPT regime.
- Early positive `I_active` that decays to zero is adaptation slowdown, not persistent harm.

## Overclaim guardrails
- Ouro vs Nanbeige cannot establish natural-vocab causality.
- EXP-001b is post-hoc and reuses an inspected test set.
- EXP-003A starts from a checkpoint originally trained with V=49,152; a positive result is limited to continued-pretraining intervention unless EXP-003B replicates from scratch.
- Never-target output classes isolate output dimensionality/competition but do not reproduce a natural tokenizer.
- A positive raw-CE interaction alone is insufficient for C5.
- LOTUS verbalizability does not establish or refute C2 by itself.
- Gradient projection norm does not establish C2 or C5.
- OOD loop depths do not count toward native-depth decoder claims.
