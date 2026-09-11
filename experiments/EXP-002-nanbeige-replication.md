# EXP-002 — Nanbeige native-loop replication

## Question
Does the sign of the T=1→2 Head Regret contrast replicate on Nanbeige4.2-3B-Base?

## Important limitation
Nanbeige is trained with `num_loops=2`. Use only T=1 and T=2 for confirmatory analysis. T>2 is exploratory OOD depth.

## Natural contrast, not causal vocab test
Nanbeige has D=3072 and V=166,144 (`V/D≈54.1`) versus Ouro D=2048/V=49,152 (`V/D=24`). A larger effect would be *consistent* with the vocab interaction but cannot establish it because almost everything else differs.

## Predictions
- H1: positive `R(2)-R(1)`.
- H2: negative contrast.
- H0: near zero.

## Controls
Same paired hidden-state / probe-training protocol as EXP-001.

## Result
Pending; do not start until EXP-001 pipeline is validated.
