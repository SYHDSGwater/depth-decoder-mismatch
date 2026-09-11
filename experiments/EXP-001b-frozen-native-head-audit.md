# EXP-001b — Frozen-native-head residual audit

## Research link
- Parent experiment: EXP-001
- Status: primary-width audit completed and reviewed (2026-09-11), post-hoc diagnostic
- Purpose: distinguish genuine depth-as-linearization from a probe-optimization artifact in EXP-001.

## Motivation
EXP-001 pilot and 1M runs produced the same qualitative pattern:

- T=1: rich decoder improves held-out CE by about 0.08 nats/token over the refit linear head;
- T=2..4: rich-vs-linear regret is approximately zero and slightly negative;
- all T=1 fits select step 100;
- all T=2..4 fits select step 1, after which validation CE rapidly worsens;
- at T=2..4 even the refit linear head is slightly worse than the untouched native LM head.

Therefore the current result is consistent with H2 (depth-as-linearization / decoder compensation), but it does **not** yet establish that a richer decoder family has no useful capacity at T>=2. The original probe jointly updates the ~100M-parameter native projection W and the residual branch. For deep states, W may already be near a good optimum; moving it can dominate any small residual gain before the nonlinear branch learns.

EXP-001b removes this confound by freezing the native LM head exactly and training only zero-initialized residual adapters.

## Data
Reuse the exact cached 1M paired hidden-state dataset from `EXP-001-1M-20260911`:

- model: `ByteDance/Ouro-1.4B`, pinned revision from EXP-001;
- depths: native T=1,2,3,4 only;
- train / validation / test targets: 800k / 100k / 100k;
- same 31,250 documents and document-level split;
- same labels and token positions across every depth and head family;
- no new hidden-state extraction and no new corpus sampling.

This supplement is **not confirmatory** because the 1M test set has already been inspected in EXP-001. Its role is mechanistic / optimization diagnosis.

## Head families
Let `W_native` denote the untouched Ouro LM-head matrix. It is frozen in every trainable arm.

### A0 — frozen native baseline

`z_native(h) = W_native h`

No trainable parameters. This is the reference point and exactly reproduces native-head CE from EXP-001 up to projection arithmetic.

### A1 — frozen-base linear residual

`z_lin(h) = W_native h + U (A h + b)`

- `W_native`: frozen;
- `A`: trainable projection `D -> r`;
- `b`: trainable bottleneck bias;
- `U`: trainable projection `r -> V`, initialized to zero.

This arm tests whether a low-rank **linear/domain-adaptation correction** helps without moving the pretrained decoder.

### A2 — frozen-base nonlinear residual

`z_nl(h) = W_native h + U GELU(A h + b)`

- same `W_native`, `A`, `b`, `U` shapes as A1;
- identical parameter count to A1;
- `U` initialized to zero;
- paired `A,b` initialization with A1 for the same depth/seed whenever possible.

A1 and A2 begin exactly at the same native function. Their only functional difference is the GELU nonlinearity.

### Primary width
`r = 512`, matching the original EXP-001 rich residual width.

### Secondary width robustness
`r in {128, 1024}` may be run after the primary audit. These are robustness checks and must be labeled secondary.

## Why the linear-residual control is necessary
A gain of A2 over A0 does not by itself prove a nonlinear decoder bottleneck. FineWeb-Edu differs from the unknown Ouro pretraining mixture, so a residual branch can simply adapt token frequencies / domain geometry.

The primary quantity is therefore the **incremental nonlinear gain**:

`G_nonlin(T) = CE_A1(T) - CE_A2(T)`.

Also report:

`G_linear(T) = CE_A0(T) - CE_A1(T)`

`G_rich(T) = CE_A0(T) - CE_A2(T)`.

Interpretation:
- `G_linear > 0`, `G_nonlin ~= 0`: adaptation is largely linear / low-rank, not evidence for decoder nonlinearity pressure;
- `G_nonlin > 0`: nonlinear decoder capacity extracts additional held-out information beyond matched linear adaptation.

## Optimization protocol

### Critical change from EXP-001
`W_native` is never updated.

### Dense early validation
Because `U=0`, step 0 is an exact native-head checkpoint and must participate in model selection.

Evaluate validation CE at:

`steps = [0, 1, 2, 5, 10, 20, 50, 100, 200, 400, 800, 1200, 1600, 2000]`.

Do not use the old `step 1, 100, 200, ...` schedule for this audit.

Run a fixed 2000 optimization steps for tuning runs; do not early-stop before the dense schedule is observed. The selected checkpoint is the minimum validation CE among the pre-registered steps.

### Optimizer
- AdamW;
- weight decay = 0;
- batch size = 256;
- FP32 residual-head training;
- constant LR within a run;
- same batch-index stream for A1/A2 when seed/depth are paired.

### Learning-rate search
Use the same validation-only grid for both residual families and every depth:

`lr in {3e-5, 1e-4, 3e-4}`.

Use tuning seed `11` to select the best LR **separately for each (depth, family)** from validation CE only. This is allowed because EXP-001b is explicitly an optimization audit whose target is the attainable held-out family optimum, not a fixed-training-recipe comparison.

After LR selection, freeze the selected LR before report-seed runs.

### Report seeds
Primary report seeds:

`[29, 47, 83]`.

Seeds 29 and 47 allow direct continuity with EXP-001; seed 83 is new. The test set is evaluated only once per selected report-seed checkpoint after validation selection.

## Initialization details
- copy `W_native` exactly from the cached EXP-001 payload and set `requires_grad=False`;
- initialize `A,b` once per `(depth, seed, width)` and reuse the same initialization for A1 and A2 where shape-compatible;
- initialize `U=0` in A1 and A2, so both exactly equal A0 at step 0;
- record a numerical equality check at step 0 for every depth/family/seed;
- record that `grad(A)=0` at the first backward when `U=0`; from step 2 onward A must acquire nonzero gradient / update. This is expected, not a failure.

## Primary endpoints
For each T=1..4, report seed-averaged held-out:

- `CE_native(T)`;
- `CE_linear_residual(T)`;
- `CE_nonlinear_residual(T)`;
- `G_linear(T)`;
- `G_rich(T)`;
- `G_nonlin(T)`.

The principal mechanistic statistic is:

`G_nonlin(T) = CE_linear_residual(T) - CE_nonlinear_residual(T)`.

Also report:

`Delta_G_nonlin = G_nonlin(4) - G_nonlin(1)`

and the slope of `G_nonlin(T)` over T=1..4.

Use document-level clustered bootstrap over the 3,125 test documents, paired across A1/A2 and depths. Report per-seed values as well as the seed-mean estimate.

## Pre-registered interpretations

### Outcome A — original deep null survives
Pattern:
- `G_nonlin(1) > 0`;
- `G_nonlin(2..4) ~= 0`;
- result robust across report seeds and width checks.

Interpretation: strong support for **depth-as-linearization / decoder compensation** within Ouro. Additional recurrent computation removes most of the nonlinear decoder advantage visible at T=1. This substantially weakens the forward-expressivity version of DDM for this model.

### Outcome B — nonlinear gain reappears at deep T after freezing W
Pattern:
- A2 beats A1 at T>=2 by a stable held-out margin;
- old full-refit probes failed to reveal it because W drift degraded the pretrained solution.

Interpretation: original EXP-001 deep null was an optimization artifact. H1 is reopened; repeat the main decoder-family comparison with an optimization protocol that preserves the pretrained base function.

### Outcome C — A1 and A2 both improve, but similarly
Pattern:
- `G_linear > 0`;
- `G_rich > 0`;
- `G_nonlin ~= 0`.

Interpretation: post-hoc improvement is mainly low-rank / domain adaptation, not evidence that nonlinear decoder expressivity is limiting.

### Outcome D — neither residual arm improves over A0
Interpretation: the native LM head is locally hard to improve on this held-out distribution under the residual parameterization. The original T>=2 null is not explained by W drift, but absence of gain is weaker evidence about the globally optimal decoder family.

## Secondary analyses
Repeat the EXP-001 difficulty and loop-benefit quartile slicing using the same frozen test positions. The key question is whether the large T=1 nonlinear gain on high-loop-benefit tokens survives the matched linear-residual control.

Do not use secondary slices for LR, width, checkpoint or architecture selection.

## Required diagnostics
For every run retain:
- validation curve at all pre-registered steps including step 0;
- selected step and selected LR;
- norms of `A`, `U` and their updates;
- residual-logit RMS relative to native-logit RMS;
- exact step-0 native-logit parity;
- gradient norms of A and U at steps 1,2,5,20;
- trainable parameter count;
- test per-token losses for paired document bootstrap.

## Invalidity / overclaim guards
- Any update to `W_native` invalidates the run.
- A1/A2 must have matched residual width and parameterization except for GELU.
- Step 0 must be a model-selection candidate.
- LR selection uses validation only; do not choose LR from test CE.
- Do not call this an independent replication or confirmatory experiment: it reuses a previously inspected test set.
- A null `G_nonlin` for this residual family does not prove that every possible nonlinear decoder family is useless.
- Do not infer anything about natural tokenizer vocabulary size from EXP-001b alone.

## Target implementation interface
A dedicated audit trainer should support an interface equivalent to:

```bash
python scripts/train_residual_audit.py \
  --hidden-root <EXP-001-1M hidden root> \
  --depth 1 \
  --family linear_residual \
  --width 512 \
  --lr 1e-4 \
  --seed 11 \
  --freeze-native-head \
  --eval-steps 0,1,2,5,10,20,50,100,200,400,800,1200,1600,2000
```

The implementation must not mutate or regenerate the cached EXP-001 hidden dataset.

## Result
Primary-width run `EXP-001b-20260911` completed on `pro-78730289ac36`; reviewed outcome below. Dedicated [configuration](../configs/experiments/exp001b.yaml) and [trainer](../scripts/train_residual_audit.py) implement the unchanged protocol above. All 24 seed-11 tuning fits finish before selected learning rates are sealed; then 24 report fits use seeds 29/47/83. Every fit runs 2000 steps, including all 14 validation checkpoints. Checkpoint ties retain the earliest step; LR ties retain grid order. Bootstrap uses 2000 document resamples with seed 20260911, conditional on the selected LRs and report seeds. Widths 128/1024 are not part of this primary launch.

The trainer verifies the parent hidden SHA-256 `81237a75fa432ececd49bc3c8d7d3e2706892222c9ff669f2b03b69de644c695` before loading; original cache is read-only. Residual-only checkpoints omit the immutable native matrix and reference that parent hash. Parameter/update norms cover complete adapter tensors; gradient diagnostics use the sampled training batches. Residual/native logit RMS uses the fixed first 256 training positions. None uses test examples.

Preflight: 11 unit tests and two full 2000-step synthetic GPU integration fits passed (tuning excludes test; report restores selected parameters before test). Frozen execution source and launch metadata are in [run record](../research/runs/EXP-001b-20260911.json). Server outputs: `/root/autodl-fs/depth-decoder-mismatch/outputs/EXP-001b-20260911/`.

## EXP-001b completed audit — 2026-09-11 (post-hoc)

All 24 tuning fits and 24 report fits completed the fixed 2000-step budget at width 512. Seed 11 selected LR separately for each depth/family using validation only; report seeds were 29/47/83. All 14 scheduled validation points included step zero. Native W remained a non-trainable buffer; each residual arm had 26,214,912 trainable parameters and paired initialization. Same exact 1M cached hidden dataset, document split, positions and labels as EXP-001.

Held-out CE and incremental nonlinear gain, averaged across report seeds (nats/token):

| T | Native CE | Linear residual CE | Nonlinear residual CE | G_linear | G_rich | G_nonlin | G_nonlin document-bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 3.639455 | 3.335530 | 3.280936 | +0.303925 | +0.358520 | +0.054595 | [+0.052344, +0.056788] |
| 2 | 2.460401 | 2.449258 | 2.446428 | +0.011143 | +0.013973 | +0.002830 | [+0.002340, +0.003355] |
| 3 | 2.247896 | 2.247896 | 2.247898 | +0.000000 | -0.000002 | -0.000002 | [-0.000008, +0.000004] |
| 4 | 2.224508 | 2.224508 | 2.224508 | +0.000000 | +0.000000 | +0.000000 | [+0.000000, +0.000000] |

G_nonlin slope = **-0.01666167**, document-bootstrap 95% percentile interval **[-0.01732787, -0.01599306]**. T4-minus-T1 contrast = **-0.05459478**, interval **[-0.05678787, -0.05234422]**. All three seed slopes are negative. These are 2000 paired document resamples over 3125 test documents, conditional on selected LRs and report seeds; intervals do not cover LR-search or model/corpus uncertainty.

**Interpretation:** T1 has substantial linear adaptation gain (0.303925) plus nonlinear incremental gain (0.054595). T2 has a smaller but consistent positive nonlinear increment (0.002830; all three seeds positive). Thus the original blanket T>=2 null does not survive this protocol. T3 remains near zero and T4 selects the untouched native function in every report fit. The declining-depth pattern remains compatible with H2, but the specific all-deep-null criterion for C3 is not met. This is a mixed outcome: T2 shows the Outcome-B pattern, while T3/T4 show no useful selected residual gain. It does not support an increasing-depth H1 slope.

Freezing W, LR search, residual-only parameterization and the selection schedule all changed relative to EXP-001. Their individual causal contributions are not isolated; this audit cannot attribute the T2 change solely to W drift. No independent confirmation or globally optimal family comparison is claimed. Width robustness at 128/1024 remains unrun.

| Depth | Selected LR linear / nonlinear | Report selected steps linear (29/47/83) | Report selected steps nonlinear (29/47/83) |
|---|---|---|---|
| 1 | 1e-4 / 3e-4 | 2000 / 2000 / 2000 | 2000 / 2000 / 2000 |
| 2 | 3e-5 / 1e-4 | 1200 / 1200 / 1200 | 800 / 800 / 400 |
| 3 | 3e-5 / 3e-5 | 0 / 0 / 0 | 0 / 0 / 5 |
| 4 | 3e-5 / 3e-5 | 0 / 0 / 0 | 0 / 0 / 0 |

T1 selects the budget boundary in all report runs, so convergence is not established. Eleven report fits selected step zero. T4's exact zero gain and degenerate bootstrap interval follow from selecting identical native functions; they are not proof that every nonlinear decoder family is useless. At T3 the seed-83 nonlinear head selected step 5 but slightly worsened held-out CE; the negative result is retained.

Secondary slices: the highest loop-benefit quartile has G_nonlin 0.121332 at T1 and 0.009953 at T2. The hardest native-T1 difficulty quartile has negative incremental gain (-0.010086 at T1, -0.003267 at T2). These slices were not used for selection.

**Audit:** independently recomputed per-token means and all reported document-bootstrap intervals. Verified 24 unique tuning and 24 unique report combinations, all 2000-step budgets and dense validation histories, validation-minimum checkpoint selection, validation-only LR selection and sealed-LR hash, matching 26,214,912 trainable parameters, step-zero parity, first/second-step gradient startup, recorded immutable-W execution assertions, all 48 residual checkpoint files, parent hidden hash, disjoint unique document/text hashes and next-token labels. Native test-loss arrays exactly match EXP-001 1M. No new training or model test inference was run during review.

Summed tuning/report runtime: 956.14 s (15.94 min), excluding input hash verification/loading and final analysis. Peak allocated GPU memory 1.49 GiB on RTX 5090 D, not reserved VRAM. Eleven unit tests and synthetic GPU integration passed before execution.

Detailed provenance, config, all validation curves, gradient/norm/RMS diagnostics, selected LRs and per-run metrics: [EXP-001b run record](../research/runs/EXP-001b-20260911.json). Source snapshot SHA-256 `f87539a12850f8db6aa69b3f017f93058c5143a894360146156128ca1e25bd58`, based on `43455b5ff0ce67cf0a9fa636e13bcc8b1e6231e1` plus archived uncommitted code. Per-token arrays and residual checkpoints remain at `/root/autodl-fs/depth-decoder-mismatch/outputs/EXP-001b-20260911/`, outside Git.
