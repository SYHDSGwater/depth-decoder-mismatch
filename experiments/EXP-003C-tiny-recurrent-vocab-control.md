# EXP-003C — Tiny recurrent × output-vocabulary causal control

## Role
**Final low-cost causal test for this project.**

EXP-003A (Ouro continued pretraining) is scientifically direct but costs tens of H100-hours. EXP-003B (from-scratch 16K-vocab LoopLM) is also more expensive than justified by the current evidence. EXP-003C asks the minimal remaining causal question at Murugan-scale compute:

> Murugan (2026) found that adding never-target output classes does not impair learning in a small ordinary Transformer. Does weight-shared recurrent depth create a `depth × output-space` interaction that is absent without recurrence?

If EXP-003C is null, the output-class-count / softmax-bottleneck branch of this project should stop. EXP-003A/B remain future-work designs rather than experiments required for the current project.

## Prior evidence being extended
Murugan's byte-level WikiText-2 control uses a four-layer Transformer with hidden width 32. Keeping the 256-byte input/target vocabulary fixed while increasing output classes from 256 to 1,024 and 4,096 did not worsen tuned validation CE across five seeds. The reported means were approximately:

- `V_out=256`: 2.2591;
- `V_out=1024`: 2.2532;
- `V_out=4096`: 2.2642.

Thus pure output-class inflation was null in the non-recurrent setting. EXP-003C changes only one conceptual axis: the same Transformer body is applied recurrently with shared weights.

## Primary question
Does pure output-vocabulary inflation become more harmful when the same Transformer computation is recurrently repeated?

Define

`P_active(T) = CE_active(T, 4096) - CE_active(T, 256)`

and the primary interaction

`I_active = P_active(T=4) - P_active(T=1)`.

The remaining recurrent-output-space hypothesis predicts

`I_active > 0`.

The null is

`I_active ~= 0`.

## Architecture
Start from the compact Murugan-style byte-level Transformer recipe and make the minimum recurrent modification.

### Fixed dimensions
- dataset/tokenization: WikiText-2, byte-level;
- active input/target vocabulary: `V_active = 256`;
- hidden width: `D = 32`;
- Transformer body: 4 layers;
- causal self-attention and FFN dimensions/norm/position recipe: inherit the public Murugan `response_vocab` configuration wherever possible;
- untied input embedding and output head;
- no adaptive exit gate;
- no intermediate-loop supervision.

### Weight-shared recurrence
Let `B_theta` be the complete 4-layer Transformer body. Add token/position embeddings once, then apply the same body recurrently:

`h^(0) = embed(x)`

`h^(t) = B_theta(h^(t-1)),  t=1..T`

`logits = W_out norm(h^(T))`.

All parameters inside `B_theta` are shared across recurrent passes. Parameter count is therefore identical for T=1 and T=4 except for the intended output-head-size intervention.

If the inherited implementation uses absolute positional embeddings, add them only at `h^(0)`; do not re-add them on every recurrent pass. If it uses position information inside attention (e.g. RoPE), preserve that mechanism identically on every pass.

## Primary 2 × 2 design

| Arm | Recurrent depth | Output classes | Never-target classes |
|---|---:|---:|---:|
| A | T=1 | 256 | 0 |
| B | T=1 | 4096 | 3840 |
| C | T=4 | 256 | 0 |
| D | T=4 | 4096 | 3840 |

Optional `V_out=1024` is **secondary only** and should not be added until the primary 2×2 result is frozen.

## Pairing and initialization
For each seed:
- A/B/C/D use the same initial input embeddings and the same initial recurrent-body parameters;
- active output rows `W[0:256]` are exactly identical across all arms;
- the 4096-class head appends 3840 independently initialized dummy rows using the same row-initialization distribution as active rows;
- dummy IDs never occur in inputs or labels;
- batch order and data examples are paired across all four arms.

### Required step-zero checks
For each T pair before training:
- hidden states are identical between 256- and 4096-output arms;
- active logits are identical;
- active-renormalized CE is identical within numerical tolerance;
- labels are always `<256`;
- dummy rows receive no positive-target labels.

Raw CE need not be identical because dummy classes enter the softmax denominator.

## Training objective
Use exactly one loss per token from the final recurrent state:

`L_train = CE_raw(softmax(logits over V_out), target)`.

Do **not** supervise intermediate recurrent states. Otherwise T=4 would receive four LM losses while T=1 receives one, confounding recurrence with supervision count.

Train the entire model, including:
- input embeddings;
- recurrent Transformer body;
- active LM-head rows;
- dummy LM-head rows in inflated arms.

## Training budget
### Primary budget
Match the compact Murugan regime as closely as practical:
- `600 optimizer steps` per run;
- same sequence length, batch size, optimizer family, scheduler and data construction as the public `response_vocab` recipe unless explicitly overridden below;
- five paired report seeds: `[1,2,3,4,5]` (or the exact five public Murugan seeds if different; freeze before results);
- no early stopping; all arms consume exactly the same number of supervised tokens for a given seed.

This yields 20 primary fits (`2 depths × 2 output sizes × 5 seeds`). At this model size, compute should be well below the Ouro-scale experiments and is intended to be sub-H100-hour rather than tens of H100-hours.

### Learning-rate selection
Do not tune the 4096-output arms independently in the primary comparison.

For each depth T separately:
1. use a dedicated tuning seed not in the five report seeds;
2. select LR using only the `V_out=256` control;
3. freeze that LR;
4. use the same LR for `V_out=256` and `4096` report arms at that T.

Use the public Murugan LR as the center of a small 3-point grid if the recurrent T=4 model needs retuning. The grid must be frozen before report-seed results.

### Tuned-arm robustness
Only if the primary experiment produces a positive `I_active`, run a secondary per-arm LR check to rule out a trivial optimizer-mismatch explanation. Do not pay this cost after a clear null.

## Evaluation and loss decomposition
For every validation example in a 4096-output arm, let

`m_dummy = sum_{j=256}^{4095} p_j`.

Define

`p_active(y) = p_raw(y) / (1 - m_dummy)`

`CE_raw = -log p_raw(y)`

`CE_active = -log p_active(y)`

`CE_competition = CE_raw - CE_active = -log(1-m_dummy)`.

For 256-output arms, `CE_raw = CE_active` and `CE_competition = 0`.

### Evaluation checkpoints
Evaluate the fixed validation set at:

`step = [0, 20, 50, 100, 200, 400, 600]`.

The final primary endpoint is step 600. Earlier checkpoints are trajectory diagnostics only and must not be used to select the endpoint.

## Primary statistics
At step 600 compute, for each paired seed:

`P_active(1) = CE_active(T1,4096) - CE_active(T1,256)`

`P_active(4) = CE_active(T4,4096) - CE_active(T4,256)`

`I_active = P_active(4) - P_active(1)`.

Also report:

`P_raw(T)` and `I_raw = P_raw(4)-P_raw(1)`

plus mean `m_dummy` / `CE_competition`.

### Uncertainty
Primary uncertainty is across the five paired training seeds:
- mean paired `I_active`;
- two-sided 95% Student-t confidence interval;
- all five individual seed interactions;
- paired effect size `d_z` as secondary reporting.

Do not use individual tokens as the independent unit for the training-seed causal claim.

## Pre-registered interpretation

### Outcome A — null interaction
`I_active ~= 0` and no consistent positive seed-level effect.

Interpretation: Murugan's never-target-vocabulary null survives weight-shared recurrence at this controlled scale. Together with EXP-001/001b, this strongly weakens the project's proposed `small vocab -> relieve recurrent softmax/decoder bottleneck` mechanism. **Stop this topic.** EXP-003A/B become future work only.

### Outcome B — raw-only interaction
`I_raw > 0` but `I_active ~= 0`.

Interpretation: recurrence changes dummy-class suppression / softmax competition but does not measurably harm learning among real symbols. This is not sufficient support for the small-vocab optimization hypothesis. **Stop this topic unless the systems cost itself becomes the research question.**

### Outcome C — positive active interaction
`I_active > 0`, consistent across seeds, with a 95% paired-seed interval excluding zero.

Interpretation: pure output-space inflation becomes more harmful under recurrent computation in this tiny controlled setting. This is the only result that justifies revisiting an expensive Ouro-scale EXP-003A in future work.

### Outcome D — negative interaction
`I_active < 0`.

Interpretation: deeper recurrence absorbs output-space inflation at least as well as shallow computation, opposite to the proposed mechanism. **Stop this topic.**

## Sanity controls
Required:
- T=1, V=256 should approximately reproduce the public compact-model baseline after accounting for implementation/environment differences;
- parameter count excluding output-head rows must be identical across all arms;
- T=1/T=4 parameter tensors have identical shapes and initialization within each seed;
- 4096-arm active rows exactly match the 256-arm head at step 0;
- active CE equality at step 0 must pass before training;
- same train/validation text and batch ordering across paired arms.

Optional smoke-only control:
- mask dummy logits to `-inf` in a 4096-shaped head. It should reproduce the 256-class objective up to numerical precision.

## Why this experiment is worth doing
It adds exactly one missing factor to an existing causal null: recurrent depth. Unlike EXP-003A, it does not try to establish billion-scale effect size. Its value is hypothesis triage:

- null -> terminate the output-space branch cheaply;
- positive -> demonstrate that recurrence changes the causal effect and motivate future scale-up.

## What this experiment cannot establish
- that natural tokenizer vocabulary size behaves like never-target classes;
- that Ouro-1.4B has the same interaction;
- that Claude's ~16K vocabulary was chosen for this reason;
- that any effect at tiny scale survives long pretraining.

## Result
Completed and audited; see detailed outcome below.

## Frozen implementation — 2026-09-11

Primary run EXP-003C-20260911 inherits public reference commit `fa8b2c5ba73e0350c9a34fbfcd95a582c0f798df` (MIT license retained). Public model/feedback modules are vendored unchanged. Dedicated TinyRecurrent wrapper repeats the same four layers, adds absolute positions once, and normalizes once at the final exit. D=32, four heads, FFN=128, pre-LayerNorm, GELU, zero dropout, untied bias-free head. T1 has exactly 69312 parameters; V4096 has 192192 at both depths. Paired arms construct the same 256-class model first, then append independent normal(0,.02) rows, avoiding RNG perturbations of active/backbone initialization.

The public output-vocab result selected LR .02 (the YAML default .002 was not the tuned result). Freeze control-only grid [.01,.02,.04], tuning seed 83, report seeds 1..5. Each tuning/report fit uses 600 steps, batch32, seq64, AdamW betas(.9,.95), eps1e-8, weight decay .1, clipping1.0, 60-step warmup/cosine exactly matching public scheduler. FP32 CUDA with TF32 off replaces public MPS. Validation endpoint remains step600; best-validation step is diagnostic only. No test split is accessed because the card defines a validation endpoint.

WikiText-2 raw revision b08601e uses nonempty parquet text rows joined with newline, verified against both published byte-stream SHA-256 values. Train batches sample contiguous 65-byte spans using independent per-seed CPU generators, identically across all arms. As required by this card, fix eight validation batches of 32 spans (16384 supervised bytes) at all evaluations, using seed10001. This differs from the public runner's advancing validation generator and is disclosed for baseline comparability. Original train/validation split and byte-stream construction are retained; no new document deduplication or tokenizer changes.

Preflight passed two focused tests: exact parameter/logit equality with public T1 model, paired tensors and logits across vocab arms, correct shared-layer call count, loss decomposition and zero masked-dummy gradients. Baseline CE near the reported 2.2591 is a plausibility check, not a target to tune toward; environment and fixed-validation-sample differences must be reported.

Primary positive mean interaction triggers the card's secondary per-arm LR robustness: same seed83/grid/600-step endpoint on inflated arms, followed by paired report seeds with separately selected inflated LRs. Reuse identical already-completed primary fits when LR unchanged; never replace primary results. No expensive Ouro run is authorized by a positive tiny-model result.

Config: configs/experiments/exp003c.yaml. Code: scripts/run_tiny_recurrent.py and src/ddm/tiny_recurrent.py. Outputs are on local server disk `/root/autodl-tmp/EXP-003C-20260911/` to preserve the user's shared-storage limit. Original EXP-003A remains paused/future work.

## EXP-003C completed — 2026-09-11

Six control-only tuning fits and twenty primary fits completed. T1 selected LR .01 and T4 .04, shared across V_out arms at each depth. All runs used the fixed 600-step endpoint and 1,228,800 supervised byte targets. Primary uncertainty is across paired training seeds, not tokens. Fixed validation sample: 16384 bytes in 256 windows; no test split was accessed.

| T | V_out | Mean active CE | Mean raw CE | Mean dummy mass |
|---|---:|---:|---:|---:|
| 1 | 256 | 2.230330 | 2.230330 | 0.00000000 |
| 1 | 4096 | 2.255715 | 2.255920 | 0.00019254 |
| 4 | 256 | 2.422211 | 2.422211 | 0.00000000 |
| 4 | 4096 | 2.423171 | 2.423267 | 0.00009518 |

Mean inflation penalty: T1 +0.025384; T4 +0.000960 nats/byte. Primary interaction **I_active=-0.02442425**, paired-seed Student-t 95% interval **[-0.08525068,+0.03640217]**, df=4, d_z=-0.49858. Seed interactions (1..5): [-0.07057318,-0.02658845,-0.05312874,-0.02878161,+0.05695073]. Raw interaction -0.02453370, interval [-0.08538666,+0.03631925]. Dummy competition at the endpoint is tiny.

**Decision:** no robust positive recurrent-depth × output-vocabulary interaction. The point estimate is negative, but the interval spans zero and nontrivial positive values; this is neither a significant negative effect nor an equivalence-to-zero result. Four of five seed effects are negative. Under the project's resource/stop rule, stop this output-space branch; do not run EXP-003A/B/004. Secondary per-arm tuning was not triggered because the primary mean was not positive.

The T1/256 baseline CE 2.23033 is close in scale to the public result 2.2591; it is not an exact replication because this protocol freezes validation windows and uses CUDA instead of MPS. T4 is worse than T1 in both output arms under the 600-step budget; this may reflect recurrent optimization difficulty and must not be interpreted as a general result about mature recurrent LMs. T4's selected LR is the top of the preregistered grid (.04), so no global optimizer-optimality claim is made. Five seeds and a small fixed validation sample limit precision. Tiny-model results do not establish Ouro-scale or natural-tokenizer causal effects.

Audit recomputed saved validation arrays, all four arm means, paired batch hashes, initial active-CE parity, fixed step-600 selection, supervised token budgets and the paired-seed interval. All 26 checkpoint files exist. Public byte-stream hashes match; T1 tensor/logit parity with pinned public code and the recurrent/masked/decomposition tests passed before training. Summed tuning/primary runtime 522.85 seconds on RTX 5090. No reruns or hyperparameter changes were made after outcomes.

Detailed configuration, provenance, all curves, per-run metrics and audit: [run record](../research/runs/EXP-003C-20260911.json). Artifacts remain at `/root/autodl-tmp/EXP-003C-20260911/`; shared storage was not increased.

## EXP-003C completed — 2026-09-11

All six control-only tuning fits and twenty paired primary fits completed. Each fit used 600 steps and 1,228,800 supervised bytes. Selected LR: T1=.01, T4=.04; shared across output sizes. Five report seeds: 1..5. The endpoint is fixed-validation CE at step600, not a held-out test evaluation.

| Depth | Output classes | Mean active CE | Mean raw CE | Mean dummy mass |
|---|---:|---:|---:|---:|
| 1 | 256 | 2.230330 | 2.230330 | 0 |
| 1 | 4096 | 2.255715 | 2.255920 | 0.000193 |
| 4 | 256 | 2.422211 | 2.422211 | 0 |
| 4 | 4096 | 2.423171 | 2.423267 | 0.000095 |

Output-inflation penalty P_active is +0.025384 at T1 and +0.000960 at T4. Primary I_active = **-0.024424**, paired-seed Student-t 95% CI **[-0.085251,+0.036402]**, df=4; d_z=-0.498578. Individual interactions: [-0.070573,-0.026588,-0.053129,-0.028782,+0.056951]. I_raw=-0.024534, CI [-0.085387,+0.036319].

No robust positive active interaction was found. The negative point estimate and wide zero-crossing interval are not proof of equivalence or a statistically established negative effect. By the experiment's triage rule, stop the output-space branch; do not launch EXP-003A/B/004. The secondary per-arm LR check was not triggered because the primary mean was negative.

T1 control mean 2.2303 is close to the public compact baseline 2.2591, with disclosed fixed-validation-sampling/CUDA differences. T4 is worse in absolute CE under the same 600-step budget; recurrence does not automatically improve this tiny model. This limits claims about mature recurrent models. LR differs by depth as preregistered, so the interaction is under this control-tuned training policy. Five seeds and a single fixed 16384-byte validation sample give limited precision; no test-set or natural-tokenizer generalization claim is made.

Audit verified all 26 fixed budgets and endpoint checkpoints, exact paired initialization assertions, same batch hashes within seed, same validation sampling hash, control-only LR selection, per-position array means, loss decomposition and independently recomputed paired-seed interval. Parameter count is 69312 for V256 and 192192 for V4096, independent of T. T1 parameter/logit parity with the pinned public implementation and recurrence/masked-control tests passed before launch. Total fit runtime 522.85 seconds (8.71 min), excluding data setup and analysis.

This is a minimal four-layer shared-body Transformer causal control, not a reproduction of Ouro pretraining. Positions are added once, all four body layers repeat T times, a final LayerNorm is applied once, and only final-loop raw CE is optimized. No adaptive exit or intermediate-loop objective is present.

Detailed config, provenance, all histories and audit: [run record](../research/runs/EXP-003C-20260911.json). Source SHA-256: `7f7b43a5988a619c021929bb26beb7e6a463efc15ec321fcc5f089ab2c736a04`; base git `d331751af192be7ef8783ab1d25a10cbb8ee4a90`; upstream MIT reference `fa8b2c5ba73e0350c9a34fbfcd95a582c0f798df`. Server artifacts: `/root/autodl-tmp/EXP-003C-20260911/`. No shared-storage allocation or Ouro training was performed.
