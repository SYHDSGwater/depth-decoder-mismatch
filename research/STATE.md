# Research State

> Keep <= ~100 lines. This is the first file an agent reads.

## Current research question
Does increasing recurrent/looped depth make a fixed-width linear-softmax decoder an increasingly strong *marginal* bottleneck, and does pure output-space inflation causally amplify that problem?

## Primary metrics
Phase A decoder-family diagnostic:
`R_head(T) = L_linear_refit*(T) - L_rich*(T)` on paired held-out examples.

Phase B pure-vocab causal test:
`I_active = [CE_active(T_hi,V_hi)-CE_active(T_hi,V_base)] - [CE_active(T_lo,V_hi)-CE_active(T_lo,V_base)]`.

## Competing hypotheses
- **H1 DDM:** `dR_head/dT > 0`; additionally, pure output-vocab inflation should hurt more at larger T (`I_active > 0`).
- **H2 Depth-as-linearization / decoder compensation:** `dR_head/dT < 0`; deeper recurrence may reduce or absorb decoder burden.
- **H0:** no material decoder-depth or depth×output-vocab interaction.

## Current baselines
### EXP-001
- Primary model: `ByteDance/Ouro-1.4B`
- Native loop depths: `T=1..4`
- Hidden size: 2048
- Vocab size: 49,152
- Probe baseline: refit bias-free linear decoder initialized from native LM head.
- Rich probe: residual nonlinear decoder `Wh + U GELU(Ah)`, zero-initialized residual output.

### EXP-003
- Controlled small Ouro-style LoopLM trained from scratch.
- One fixed BPE tokenizer with `V_base=16,384`.
- Input vocabulary remains 16,384 in every arm.
- Output-only vocabulary levels: 16,384 (control), 49,152 (primary inflated), 131,072 (stress).
- Recurrent depth: T=1 and T=4 primary; T=2 secondary.
- Extra output classes are trainable, enter the softmax denominator, and are never targets.

## Current experiment order
1. `EXP-001`: Ouro native-depth Head Regret sweep.
2. `EXP-002`: Nanbeige native-loop replication.
3. `EXP-003`: direct output-vocabulary inflation × recurrent-depth causal test.
4. `EXP-004`: natural 16K-vs-larger tokenizer × depth study, only after the isolated output-space test.

## Supported facts / guardrails
- A standard linear LM head constrains cross-context logits to rank <= hidden width.
- LOTUS falsifies the strong claim that deeper latent states must become unreadable by the base LM head; target decoder-family regret instead.
- Large null-space gradient norm alone is not evidence of harmful optimization; Murugan (2026) is the required causal counterpoint.
- Changing a natural tokenizer introduces sequence-length, token-frequency and compositional confounds.
- Never-target output classes isolate output dimensionality/competition while leaving tokenizer and targets fixed.
- Raw CE under output inflation includes a mechanical dummy-class denominator penalty; EXP-003 therefore decomposes it into `CE_active` and `CE_competition`.

## Open uncertainties
- Does `R_head(T)` increase, decrease, or remain flat over Ouro T=1..4?
- Does the sign replicate on Nanbeige T=1→2?
- Does output-only vocabulary inflation produce `I_active > 0`, i.e. active-token modeling damage that grows with recurrent depth?
- Or is any raw-loss penalty entirely explained by dummy-class probability mass (`CE_competition`)?
- If EXP-003 is positive, does the mechanism survive a natural tokenizer change in EXP-004?

## Protocol invariants
### EXP-001
- one checkpoint/tokenizer revision across depths;
- same examples and target positions at every T;
- same train/val/test split and probe optimization policy;
- only native trained loop depths are confirmatory.

### EXP-003
- same tokenizer, token IDs, targets, documents and batch order across `V_out` arms;
- same input embedding vocabulary across arms;
- output-only extra classes never appear as targets;
- paired initialization seeds and matched backbone/active output rows;
- raw CE alone is not sufficient for the strong bottleneck claim; `CE_active` interaction is primary.

## Next action
EXP-001 smoke, pilot and 1M have completed; retain their optimization and non-independent-sampling limitations. In parallel, implement an EXP-003 decoder-only bridge on frozen Ouro hidden states to validate the output-inflation decomposition and estimate useful V_out levels before committing to from-scratch training.

## EXP-001 completed runs (2026-09-11)
- Smoke: 10240 targets, 24 fits, 50-step budget; slope +0.00009894, document interval [-0.00344752,+0.00366996]; plumbing passed, no hypothesis conclusion.
- Pilot: 102400 targets, 24 fits; slope -0.02479354, interval [-0.02685600,-0.02270657]; observed direction favors H2 under the fixed recipe.
- 1M: 800000/100000/100000 targets, 31250 documents, 24 fits; slope -0.02339529, interval [-0.02406723,-0.02274578]; observed direction favors H2.
- Pilot and 1M: T1 selected step 100, T2..4 step 1; optimized family optima are not established. Rich held-out CE improves with depth.
- 1M includes all pilot/smoke records unchanged; all stages are non-confirmatory and independent confirmation remains pending.
- Exact-text deduplication preceded pilot and 1M extraction. Paired data, disjoint splits, native depths and nested extra capacity checked.
- Versioned configurations: configs/experiments/exp001-{smoke,pilot,1m}.yaml; full results and audit: research/RESULTS.md and research/runs/.
- 1M remote output: /root/autodl-fs/depth-decoder-mismatch/outputs/EXP-001-1M-20260911/ on pro-78730289ac36; weights remain outside Git.
