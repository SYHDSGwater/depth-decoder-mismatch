# Research State

> Keep <= ~100 lines. This is the first file an agent reads.

## Current research question
Does increasing recurrent/looped depth make a fixed-width linear-softmax decoder an increasingly strong *marginal* bottleneck, or does recurrent computation instead align hidden states to the pretrained decoder? Separately, does pure output-space inflation causally create a depth-dependent optimization penalty?

## Primary metrics
Phase A decoder-family diagnostic:
`R_head(T) = L_linear_refit*(T) - L_rich*(T)` on paired held-out examples.

EXP-001b frozen-head mechanistic audit:
`G_nonlin(T) = CE_linear_residual(T) - CE_nonlinear_residual(T)` with the native LM head frozen in both arms.

Final low-cost output-space causal test (EXP-003C):
`I_active = [CE_active(T4,4096)-CE_active(T4,256)] - [CE_active(T1,4096)-CE_active(T1,256)]`.

## Current evidence
- EXP-001 pilot and 1M both show a strongly negative depth slope of rich-vs-linear regret, concentrated in T1→T2.
- In both pilot and 1M, **linear refit itself also stops helping at T>=2**: T2/T3/T4 all select the earliest evaluated checkpoint (step 1), and the refit head is slightly worse than the untouched native head.
- This is important because linear-softmax CE is convex in W for fixed hidden states. The deep failure is therefore not naturally explained by a non-convex head-optimization trap; it is more consistent with the pretrained native W already being near a good optimum for deep states, while unrestricted 100M-parameter refitting adds estimation noise / overfits finite FineWeb-Edu samples.
- EXP-001b strengthens this interpretation: with W_native frozen, a constrained low-rank linear residual gains +0.303925 nat at T1, +0.011143 at T2, and exactly ~0 at T3/T4; nonlinear incremental gain similarly collapses T1=0.054595, T2=0.002830, T3≈0, T4=0.
- Therefore current Ouro evidence does **not** support forward DDM. It instead suggests recurrent computation progressively moves representations into a geometry already well matched to the pretrained LM head.

## Competing hypotheses
- **H1-forward DDM:** deeper states become increasingly under-exploited by a linear decoder. Current Ouro evidence strongly argues against this direction.
- **H2 decoder alignment / depth-as-linearization:** recurrent computation reduces both post-hoc linear adaptation gain and nonlinear decoder gain. Current Ouro evidence supports this within-model mechanism; independent replication is absent.
- **H3 recurrent output-space interaction:** larger V_out causes greater active-token learning damage at larger T even if deep states are linearly readable; predicts `I_active > 0` in EXP-003C.
- **H0-output:** no material recurrent-depth × output-vocabulary interaction after removing dummy competition.

## Current baselines
### EXP-001 / EXP-001b
- model: `ByteDance/Ouro-1.4B`;
- native T=1..4, D=2048, V=49,152;
- EXP-001 full linear/rich probes start from native W;
- EXP-001b freezes W_native and compares matched linear vs nonlinear residual adapters.

### EXP-003C — final low-cost causal screen
- WikiText-2 byte-level, active/input/target vocabulary fixed at 256;
- compact 4-layer, D=32 Transformer body;
- same 4-layer body recurrently weight-shared for T=1 vs T=4;
- output vocabulary: 256 control vs 4096 with 3840 trainable never-target classes;
- one final-loop LM loss only;
- 600 optimizer steps/run;
- 5 paired report seeds;
- primary endpoint: paired-seed `I_active` at step 600.

### EXP-003A / EXP-003B
Scientifically stronger but currently too expensive for this project. Keep as future-work designs only. EXP-003A uses Ouro continued pretraining; EXP-003B is a larger from-scratch LoopLM replication.

## Current experiment order
1. EXP-001 — Ouro Head Regret sweep: completed, non-confirmatory.
2. EXP-001b — frozen-native-head residual audit: completed, post-hoc mechanistic diagnostic.
3. EXP-003C — tiny recurrent × output-vocab causal control: **completed: no robust positive interaction**.
4. If EXP-003C is null/raw-only/negative: stop the topic and leave EXP-003A/B/004 as future work.
5. If EXP-003C shows robust positive `I_active`: expensive Ouro-scale validation may be reconsidered in future work.

## EXP-003C protocol invariants
- same byte-level inputs/targets and data order across all arms;
- same initial input embeddings, recurrent-body weights and 256 active head rows within each paired seed;
- T=1/T=4 differ only in the number of applications of the same shared Transformer body;
- V_out=4096 appends trainable never-target rows that enter the raw softmax denominator;
- dummy IDs never appear as inputs or labels;
- one final-loop raw-softmax loss in both depth arms; no intermediate-loop supervision;
- LR chosen from V_out=256 control separately per T, then shared with V_out=4096 at that T;
- final endpoint fixed at step 600; earlier validation checkpoints are trajectory diagnostics only;
- strong evidence requires positive `I_active`, not raw CE alone.

## Stop rule
- `I_active ~= 0`: Murugan-style output-vocab null survives recurrence at controlled scale; stop the output-space branch.
- `I_raw > 0` but `I_active ~= 0`: competition/suppression only; stop unless systems cost becomes a separate topic.
- `I_active < 0`: opposite interaction; stop.
- robust `I_active > 0` across paired seeds: only outcome that justifies future expensive Ouro-scale testing.

## Next action
EXP-003C completed: I_active=-0.024424, paired-seed 95% interval [-0.085251,+0.036402], four negative seeds and one positive. Stop output-space experiments under the project rule. This is not a proof of zero or a significant negative effect. EXP-003A/B/004 remain future work; no additional training.

## Preserved EXP-003A pause
- User paused before smoke; only engineering preflight ran. Do not resume training.
- Local implementation and launch record remain preserved; EXP-003A/B are future work.

## EXP-003C result
- I_active=-0.024424, paired-seed t95 [-0.085251,+0.036402]; 1/5 positive seeds.
- Zero-crossing uncertainty does not establish equivalence or a negative causal effect. No positive-trigger secondary run.
- Six tuning and twenty primary fits audited; results and full record in research/RESULTS.md and research/runs/EXP-003C-20260911.json.
