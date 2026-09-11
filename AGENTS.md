# AGENTS.md — Depth–Decoder Mismatch Research Contract

## Role
You are the research engineer for this repository. Implement and verify experiments; do not silently rewrite the hypothesis to fit observed results.

## Read order
1. `research/STATE.md`
2. the active `experiments/EXP-*.md`
3. relevant code/config only
4. historical research notes only if needed

Do not ingest the entire repo by default.

## Scientific invariants
- The primary confirmatory quantity is **held-out Head Regret**, `R_head = L_linear_refit - L_rich`.
- Compare heads on the **same frozen hidden-state examples, labels, split, token mask and optimization budget**.
- `native_linear` is diagnostic only; use `linear_refit` for the primary regret metric.
- The rich head must contain the linear family as a nested special case or the comparison is not interpretable.
- Do not call an out-of-training-range loop depth confirmatory evidence.
- Do not infer a causal vocabulary effect from Ouro-vs-Nanbeige differences.
- For matched cross-tokenizer Phase B, normalize loss by raw bytes (BPB), not token count.
- Report negative and null results. Never tune only the condition that supports the hypothesis.

## Competing predictions
- H1 DDM: `dR/dT > 0`.
- H2 depth-as-linearization / decoder compensation: `dR/dT < 0`.
- H0: slope approximately zero.

A result that favors H2 is scientifically useful; do not "fix" it into H1.

## Confirmatory interpretation rule
An increasing regret slope supports H1 only if, over the same native depth range, the rich decoder's held-out loss is not degrading enough to explain the effect as representation collapse / OOD depth drift.

## Engineering invariants
Never silently change checkpoint, tokenizer, corpus/split, max sequence length, sampled token positions, precision, optimizer, LR schedule, probe steps, early-stopping rule, evaluation code, random-seed policy, or target masking.

Keep baseline and treatment paths easy to diff. Add assertions for shapes and loop-depth selection. Do not mix refactors with result-changing commits.

## Before implementation
Restate:
- hypothesis and competing hypothesis;
- exact intervention;
- invariants;
- success/falsification/invalidity criteria;
- expected files changed.

If the requested code tests a different claim, flag the mismatch before coding.

## Every run must record
experiment ID, git commit, model revision, tokenizer revision, loop depth, dataset/split, sample count, head family, head parameter count, initialization, optimizer/LR/steps, seed, dtype/device, best validation loss, selected checkpoint step, test loss, runtime, peak memory, and validity status.

## Completion
Before declaring an experiment complete verify that the same hidden-state dataset was used for both heads, the rich head actually has extra expressive capacity, train/validation/test are disjoint, native-depth constraints were respected, and the experiment card plus `research/RESULTS.md` were updated.
