# Research State

> Keep <= ~100 lines. This is the first file an agent reads.

## Current research question
Does increasing recurrent/looped depth make a fixed-width linear-softmax decoder an increasingly strong *marginal* bottleneck, and does pure output-space inflation causally create a depth-dependent optimization penalty even when deeper states remain linearly readable?

## Primary metrics
Phase A decoder-family diagnostic:
`R_head(T) = L_linear_refit*(T) - L_rich*(T)` on paired held-out examples.

EXP-001b frozen-head mechanistic audit:
`G_nonlin(T) = CE_linear_residual(T) - CE_nonlinear_residual(T)` with the native LM head frozen in both arms.

Phase B output-space causal test:
`I_active = [CE_active(T_hi,V_hi)-CE_active(T_hi,V_base)] - [CE_active(T_lo,V_hi)-CE_active(T_lo,V_base)]`.

## Current evidence
- EXP-001 pilot/1M show negative depth slope of rich-vs-linear regret, concentrated in T1→T2.
- EXP-001b freezes the native LM head and still finds `G_nonlin` collapsing with depth: T1=0.054595, T2=0.002830, T3≈0, T4=0.
- Therefore current Ouro evidence does **not** support the forward-expressivity version of DDM; deeper native states do not increasingly require a richer decoder.
- The remaining small-vocab hypothesis is now framed as an output-space optimization/competition question, tested separately in EXP-003A.

## Competing hypotheses
- **H1-forward DDM:** deeper states become increasingly under-exploited by a linear decoder. Current Ouro evidence does not support this.
- **H2 depth-as-linearization / decoder alignment:** recurrent computation reduces marginal nonlinear decoder gain. Current Ouro evidence supports this direction, pending independent/cross-architecture replication.
- **H3 output-space optimization interaction:** larger `V_out` causes greater collateral active-token learning damage at larger T even if forward decoder expressivity is not limiting; predicts `I_active > 0` in EXP-003A.
- **H0-output:** no material depth × output-vocabulary interaction after removing mechanical dummy competition.

## Current baselines
### EXP-001 / EXP-001b
- model: `ByteDance/Ouro-1.4B`;
- native T=1..4, D=2048, V=49,152;
- EXP-001b primary expressivity endpoint: matched linear-vs-nonlinear residual gain with `W_native` frozen.

### EXP-003A — primary Phase-B screen
- same pinned Ouro-1.4B checkpoint/tokenizer as EXP-001;
- tokenizer/input vocabulary fixed at 49,152;
- output vocabulary: 49,152 control vs 98,304 output-only inflation;
- dummy rows are trainable and never inputs/targets;
- primary recurrent depths: T=1 vs T=4, forced fixed depth;
- one final-step raw-softmax LM loss per token in every arm;
- all ordinary model parameters plus active/dummy output rows update during continued pretraining;
- primary screen budget: 50M supervised tokens/arm;
- primary endpoint: `I_active(50M)`; validation trajectory `I_active(s)` distinguishes transient vs persistent cost.

### EXP-003B — from-scratch replication
Run only if EXP-003A shows a stable nonzero interaction worth testing outside pretrained-checkpoint adaptation.

### EXP-004 — natural tokenizer study
Changes tokenizer itself; separate because sequence length, token frequency and compositional structure also change.

## Current experiment order
1. EXP-001 — Ouro Head Regret sweep: completed, non-confirmatory.
2. EXP-001b — frozen-native-head audit: completed, post-hoc mechanistic diagnostic.
3. EXP-003A — Ouro continued-pretraining output-vocab intervention: **next primary experiment**.
4. EXP-002 — Nanbeige cross-architecture decoder-gain replication.
5. EXP-003B — small from-scratch output-vocab replication if EXP-003A positive.
6. EXP-004 — natural tokenizer-vocab experiment if isolated output-space evidence warrants it.

## EXP-003A protocol invariants
- same tokenizer/token IDs/targets/input embeddings across V_out arms;
- same starting backbone and active LM-head rows within paired arms;
- 98K arm adds exactly 49,152 never-target output rows;
- primary dummy initialization uses one-to-one clones of active rows, giving step-0 `m_dummy=0.5`, `CE_competition=log 2`, and identical `CE_active` across paired V arms;
- fixed recurrent T; adaptive exit disabled;
- one final-step LM loss in both T=1 and T=4 arms;
- raw full-softmax CE is used for training; active-renormalized CE is evaluation only;
- backbone and active rows are trainable; freezing them would make `CE_active` unable to reveal collateral learning damage;
- LR selected from 49K control only within each T, then shared with 98K arm;
- same training documents, token budget and paired batch order across all arms;
- fresh data relative to EXP-001 preferred;
- test set evaluated only at the fixed 50M endpoint;
- document-level paired bootstrap; tokens are not IID units.

## Strong evidence standard for EXP-003A
- `I_raw > 0` alone = competition/suppression cost only;
- persistent `I_active > 0` through the 50M endpoint = strong evidence for a depth-dependent output-space optimization penalty in pretrained Ouro;
- early positive `I_active` that decays to zero = transient adaptation slowdown;
- null `I_active` substantially weakens a pure output-class-count explanation for small vocab in Ouro.

## Next action
Implement and smoke-test EXP-003A, including the masked-98K sanity control and exact step-zero loss decomposition, before spending compute on the 50M × 4-arm × 3-seed screen.
