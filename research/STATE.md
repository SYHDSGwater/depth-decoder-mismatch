# Research State

> Keep <= ~100 lines. This is the first file an agent reads.

## Current research question
Does increasing recurrent/looped depth make a fixed-width linear-softmax decoder an increasingly strong *marginal* bottleneck, measurable as rising held-out decoder-family regret on the same frozen hidden states?

## Primary metric
`R_head(T) = L_linear_refit*(T) - L_rich*(T)` on paired held-out examples.

Confirmatory statistic for Phase A: slope / paired depth contrast of `R_head(T)` within one checkpoint's native loop range.

## Competing hypotheses
- **H1 DDM:** `dR/dT > 0`.
- **H2 Depth-as-linearization / decoder compensation:** `dR/dT < 0`.
- **H0:** `dR/dT ≈ 0`.

## Current baseline
- Primary model: `ByteDance/Ouro-1.4B`
- Native loop depths: `T=1..4` extracted from one max-depth forward where possible.
- Hidden size: 2048
- Vocab size: 49,152
- Probe baseline: refit bias-free linear decoder initialized from native LM head.
- Rich probe: residual nonlinear decoder `Wh + U GELU(Ah)`, zero-initialized residual output.
- Reproduction status: not started.

## Current experiment
- `EXP-001`: Ouro native-depth Head Regret sweep.
- Purpose: decide the sign of `dR/dT` before doing any vocabulary-causal study.

## Supported facts
- A standard linear LM head constrains cross-context logits to rank <= hidden width.
- Ouro exposes recurrent computation with a native 4-step depth range and per-step hidden states.
- Nanbeige4.2-3B uses 22 shared layers for 2 loops, with V=166,144 and D=3072.
- LOTUS falsifies the strong claim that deeper latent states must become unreadable by the base LM head; this project therefore targets decoder-family *regret*, not verbalizability.
- Large null-space gradient norm alone is not sufficient evidence of harmful optimization; the 2026 causal test is a required counterpoint.

## Open uncertainties
- Does probe regret monotonically increase, decrease, or remain flat over Ouro T=1..4?
- How sensitive is the sign to rich-head width, probe data size and early stopping?
- Is any effect reproduced on Nanbeige T=1->2?
- Does a matched tokenizer-vocab × depth experiment show a positive interaction in BPB?

## Protocol invariants for EXP-001
- one checkpoint and tokenizer revision across all depths;
- same raw examples and sampled token positions at every depth;
- same train/val/test split for all heads and depths;
- same optimizer, LR search policy, step budget and early stopping across head families;
- no task-specific finetuning of the backbone;
- only native trained loop depths count as confirmatory.

## Next action
Run a small Ouro extraction smoke test (T=1..4), verify hidden/label alignment and measure native-head CE at each depth before training probes.
