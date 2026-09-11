# Problem Definition

## One-sentence research question
As looped Transformers increase effective compute depth while holding hidden width and linear LM-head family fixed, does decoder-family regret increase with depth, and is that depth effect amplified by larger vocabulary size?

## Motivation
Looped/recurrent depth decouples compute depth from parameterized width. This creates a clean setting in which backbone computation can increase without increasing the rank capacity of a standard `V × D` linear LM head. The geometry is real, but whether it becomes an empirically meaningful marginal bottleneck is unknown.

The question also connects two otherwise separate observations: Claude's reported shift from roughly 49K to roughly 16K vocabulary and renewed interest in recurrent-depth architectures. The project does **not** assume that Claude is looped or that its tokenizer change was caused by decoder geometry.

## Scope
### In scope
- post-hoc frozen-state decoder-family probes;
- native recurrent-depth sweeps on open looped checkpoints;
- paired held-out loss / Head Regret;
- matched small-model vocab × depth study if Phase A shows a signal;
- null and opposite-sign results.

### Out of scope
- claiming Claude's proprietary architecture from indirect evidence;
- treating verbalizability as equivalent to decoder optimality;
- using cross-model Ouro-vs-Nanbeige differences as a causal vocab test;
- treating backward gradient projection norm as proof of optimization harm.

## Intended contribution
Primary type: mechanism / empirical finding / negative result.

Smallest worthwhile claim: determine whether the sign of the loop-depth × decoder-expressivity interaction is positive, negative, or negligible in native looped LMs under a controlled frozen-state protocol.

## Kill criteria
Stop the DDM direction as a primary mechanism if:
1. `R_head(T)` is flat or decreases across native Ouro depth under multiple reasonable rich-head capacities and probe budgets; and
2. the same conclusion replicates on Nanbeige or another independent looped checkpoint.

A negative result should be written up as evidence for depth-as-linearization / decoder compensation rather than tuned away.
