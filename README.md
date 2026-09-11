# Depth–Decoder Mismatch

A falsifiable study of whether **recurrent / looped depth increases representational capacity faster than a fixed-width linear-softmax LM head can exploit it**.

The motivating observation is simple. A looped Transformer can increase effective compute depth `T` while keeping hidden width `D` and the LM head shape `V × D` fixed. If the backbone learns increasingly fine context distinctions as `T` grows, a fixed linear decoder may become a larger *marginal* bottleneck. The project tests that claim directly rather than inferring it from verbalizability or gradient geometry.

## Primary statistic: Head Regret

For frozen hidden states at loop depth `T`, fit a best-effort linear decoder and a strictly richer decoder on the same probe data:

```text
R_head(T) = L_linear*(T) - L_rich*(T)
```

Three competing predictions are pre-registered:

- **H1 — Depth–Decoder Mismatch:** `d R_head / dT > 0`.
- **H2 — Depth-as-Linearization / Decoder Compensation:** `d R_head / dT < 0`.
- **H0 — No systematic interaction:** `d R_head / dT ≈ 0`.

The sign of the slope matters more than whether deeper loops improve task accuracy.

## Phase A
Use native recurrent-depth sweeps on Ouro-1.4B and then Nanbeige4.2-3B-Base. Cross-model differences are not treated as a causal vocabulary test.

## Phase B
Train matched looped models with multiple vocabulary sizes and loop depths. Compare decoder regret in bits per byte and test the vocab × depth interaction.

## Decoder families
Both probes start from the native LM-head weights and use the same frozen states: a refit linear head and a residual richer head `Wh + U GELU(Ah)` with zero-initialized residual output.

## Repository map
`research/` contains the scientific state and competing hypotheses; `experiments/` contains preregistered cards; `src/ddm/` contains heads, metrics and model adapters; `scripts/` contains extraction, probe training and analysis entrypoints.

Start with `research/STATE.md`, then `experiments/EXP-001-ouro-head-regret.md`.
