# Results

No confirmatory runs yet.

## Primary summary

| Experiment | Model | Intervention | Primary statistic | Interpretation | Status |
|---|---|---|---:|---|---|
| EXP-001 | Ouro-1.4B | native recurrent depth T=1..4 | `dR_head/dT`, `R(4)-R(1)` | non-confirmatory; pilot/1M direction favors H2 | smoke/pilot/1M completed |
| EXP-002 | Nanbeige4.2-3B-Base | native T=1→2 replication | paired Head-Regret contrast | architecture replication | not started |
| EXP-003 | controlled small Ouro-style LoopLM | output-only `V_out` inflation × recurrent depth | `I_active` | pure output-space causal test | not started |
| EXP-004 | matched small models | natural tokenizer vocab × recurrent depth | BPB-normalized interaction | natural tokenizer replication | gated on EXP-003 / Phase B |

Do not add exploratory OOD-depth results to the confirmatory columns.

## EXP-001 smoke — 2026-09-11 (non-confirmatory)

**Outcome: extraction, pairing and probe plumbing passed.** This is a 50-step smoke run, not a converged estimate of decoder-family regret or a completed confirmatory experiment.

- Run: `EXP-001-smoke-20260911-r3`, AutoDL `pro-78730289ac36`, RTX 5090.
- Base commit: `6b7eeb3069b9c1300d24c8d91fda90b61dbb3113`; execution-source SHA-256: `1d2f4d0abd2bfb0bee00a5309d3c4d0d81a413bc042fd45009274671ad04b6fb`. The source snapshot includes the uncommitted smoke implementation and is retained in local run artifacts.
- Model/tokenizer: `ByteDance/Ouro-1.4B`, revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`; all 13 files hash-verified.
- Corpus: FineWeb-Edu `sample-10BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, first shard, first full window per eligible document. Document split precedes windowing. 320 documents yield 10,240 paired targets: 8192/1024/1024 train/validation/test. Sampling seed 20260911.
- Extraction and hidden storage: BF16, native T=1..4, 1024 tokens/window, 32 predictor positions in [128,1022]. Native normalization audited for every window; raw native logits match explicit native-loop model outputs exactly at all four depths on the first window.
- Probes: FP32 AdamW, constant LR 1e-4, weight decay 0, batch 256, 50 steps, validation every 10 steps plus step 1, patience 8 evaluations; seeds 11/29/47. Both heads use identical data and batch-index seed policy. Linear has 100,663,296 parameters; residual rich width 512 has 126,878,208. Native W initialization; rich output projection starts at zero.
- Native diagnostic CE also uses cached BF16 hidden states and native W promoted to FP32, matching probe projection arithmetic. The separate native-logit parity check uses the model's BF16 projection; the reported CE is not a claim of bitwise equality to BF16 full-model CE.
- All 24 training runs passed. Best validation checkpoint restored before each trained head's single held-out evaluation. Head states were discarded after evaluation; per-token losses, histories and selected steps are retained.

Held-out CE and regret, averaged across the three probe seeds (nats/token):

| T | Native CE | Linear refit CE | Rich CE | Head Regret |
|---|---:|---:|---:|---:|
| 1 | 3.654745 | 3.609716 | 3.610036 | -0.000320 |
| 2 | 2.510912 | 2.510592 | 2.510583 | +0.000010 |
| 3 | 2.302784 | 2.302379 | 2.302369 | +0.000011 |
| 4 | 2.271101 | 2.270769 | 2.270759 | +0.000010 |

Smoke slope = **+0.00009894**, Delta R = **+0.00032951**. A 500-replicate test-document bootstrap conditional on the three seeds gives slope percentile interval **[-0.00344752, +0.00366996]**. Difficulty and loop-benefit quartile results are retained in the detailed record; no resampling or tuning used those slices.

Interpretation: the native decoder's held-out CE improves across the native depths in this sample. Head-regret differences are tiny, include a negative rich gain at T=1, and the smoke interval includes zero. This establishes no support for H1 or H2 and does not establish H0. In 18/24 runs (all T=2..4 runs), validation selected step 1; the short run cannot establish sufficiently optimized head-family optima. Longer or changed optimization must be planned symmetrically without using these test effects to tune settings.

Validity checks: one shared hidden-file SHA-256 for every head and depth; labels and positions verified against stored input IDs; disjoint document/content IDs across splits; native depths only; exact rich/linear equality at initialization; explicit nonlinear-capacity unit test; residual branch acquired nonzero weights. Six tests passed in the isolated server environment. Extraction took 32.8 s, summed probe training/evaluation time 28.3 s; peak allocated GPU memory was 2.82/2.85 GiB for extraction/probes (excludes dataset preparation, setup and allocator reserve).

Engineering issue: the initial `datasets`/PyArrow reader crashed during process finalization after writing valid samples. The final synchronous PyArrow reader exited normally and reproduced byte-identical windows. Failed preparation logs are retained remotely; no sample, split, model, or optimization change was made to obtain a favorable effect.

Detailed versioned record: [research/runs/EXP-001-smoke-20260911.json](runs/EXP-001-smoke-20260911.json). Local artifacts: `outputs/EXP-001-smoke-20260911-r3/`; remote artifacts and paired hidden tensor: `/root/autodl-tmp/depth-decoder-mismatch-smoke/outputs/EXP-001-smoke-20260911-r3/`. Model/hidden data and output artifacts are excluded from Git.

## EXP-001 pilot review — 2026-09-11 (non-confirmatory)

**Observed direction favors H2 over H1 under the fixed training recipe.** The decrease is concentrated between T=1 and T=2. It does not establish population-optimal decoder regret or an independent replication.

Run `EXP-001-pilot-20260911-r2`: Ouro-1.4B, 102,400 targets in 3200 unique documents; train/validation/test = 81920/10240/10240 targets. Same pinned model/tokenizer/corpus revisions as smoke. All 24 fits completed with the frozen symmetric 5000-step cap, LR 1e-4, batch 256, constant AdamW, validation every 100 plus step 1, patience 8, rich width 512, and seeds 11/29/47. BF16 extraction/storage and FP32 projection arithmetic; native CE is diagnostic only. Base commit `6b7eeb3069b9c1300d24c8d91fda90b61dbb3113`, execution-source hash `297b1426fe10bbcf2ae5a090e1c5d193148252b8f19f17220a44b35e1c738a3c`.

| T | Native CE | Linear refit CE | Rich CE | Head Regret |
|---|---:|---:|---:|---:|
| 1 | 3.724237 | 3.647180 | 3.564628 | +0.082553 |
| 2 | 2.504394 | 2.504639 | 2.504700 | -0.000061 |
| 3 | 2.286927 | 2.287356 | 2.287442 | -0.000086 |
| 4 | 2.263911 | 2.264317 | 2.264401 | -0.000084 |

CE units are nats/token; trained-head results average three seeds. Regret is recomputed from per-token differences (minor displayed rounding differences are expected).

- Slope: **-0.02479354**, document-bootstrap 95% percentile interval **[-0.02685600, -0.02270657]**.
- Delta R (T4-T1): **-0.08263671**, interval **[-0.08951288, -0.07567963]**.
- Seed slopes (11/29/47): -0.02470583 / -0.02378702 / -0.02588777.
- Intervals use 2000 test-document resamples of 320 documents, conditional on the three seeds. They do not account for model, corpus or hyperparameter uncertainty.

Rich held-out CE decreases from 3.564628 to 2.264401; there is no rich-loss depth deterioration in this sample. Rich capacity benefits T1 substantially and contributes essentially no gain at T2..4 under this recipe. The tiny negative deep regrets are retained, not clipped. Secondary loop-benefit Q4 shows larger shallow rich gain (T1 regret 0.164844) than Q1 (0.029949); this is a larger negative depth contrast, not H1's positive interaction. Difficulty quartiles are non-monotonic, with the largest T1 gain in Q3. Slices remain secondary and were not used for tuning.

**Optimization limitation:** all six T1 fits selected step 100 and early-stopped at 900; all 18 deeper fits selected step 1 and early-stopped at 800. None reached the 5000-step cap. Later validation losses worsen materially. No step-zero checkpoint competed in model selection, and steps 2–99 were not evaluated. Native diagnostic CE is slightly better than deeper refit CE. Thus numerical completion and a negative bootstrap interval do not certify optimized decoder-family optima; genuine linearization and limitations of this finite-data fitting/selection scheme are not fully separated. Further diagnostics must be symmetric and validation-only, with the pilot test result frozen.

Audit: independently recomputed per-token means and both bootstrap intervals; checked unique document/content hashes, all positions/next-token labels, exact sample counts, shared hidden hash across all heads/depths, validation-minimum selected steps, and 24 unique depth/seed/family combinations. Executed code enforces native depths, exact nested initialization, extra rich capacity and nonzero residual learning, and restores validation-selected weights before a single test evaluation. All 320 smoke documents/10,240 target records are unchanged within the actual pilot, so this is staged scaling rather than independent replication. The deduplication amendment skipped one duplicate text before any model outcomes; the failed first preparation remains separate.

Extraction runtime: 305.85 s; summed probe runtimes: 214.89 s; maximum allocated GPU memory: 2.82 GiB extraction / 2.85 GiB probing (not reserved VRAM or total wall time). Checkpoints and the shared hidden tensor remain on `pro-78730289ac36` under `/root/autodl-tmp/depth-decoder-mismatch-pilot/outputs/EXP-001-pilot-20260911-r2/`. Local lightweight artifacts and executed-source copies are in `outputs/EXP-001-pilot-20260911-r2/`.

Detailed config, provenance, all per-run training histories and the audit: [pilot run record](runs/EXP-001-pilot-20260911.json). No new training or test inference was run during review. Confirmatory status remains pending.


## EXP-001 1M completed — 2026-09-11 (non-confirmatory)

Run `EXP-001-1M-20260911` completed all 24 fits on 1,000,000 paired targets from 31,250 documents (800000/100000/100000 targets). The model/tokenizer/corpus revisions and symmetric optimizer recipe are unchanged from pilot; see [1M config](../configs/experiments/exp001-1m.yaml). Frozen execution source SHA-256: `6188bb3e5febf64d1cc575e681ae8fa97d0edc0c3b902cac7d623c760dedae5c`, based on commit `6b7eeb3069b9c1300d24c8d91fda90b61dbb3113` with archived uncommitted implementation.

| T | Native CE | Linear refit CE | Rich CE | Head Regret |
|---|---:|---:|---:|---:|
| 1 | 3.639455 | 3.573632 | 3.495709 | +0.077923 |
| 2 | 2.460401 | 2.460558 | 2.460604 | -0.000046 |
| 3 | 2.247896 | 2.248119 | 2.248177 | -0.000058 |
| 4 | 2.224508 | 2.224716 | 2.224773 | -0.000057 |

CE units are nats/token; trained-head results average seeds 11/29/47. Slope **-0.02339529**, document-bootstrap 95% percentile interval **[-0.02406723, -0.02274578]**; T4-minus-T1 regret **-0.07798033**, interval **[-0.08022013, -0.07581604]**. All three seed slopes are negative. The 2000 bootstrap replicates resample 3125 test documents, conditional on the three seeds; intervals are runner-produced, not independently recomputed during this sync.

The observed direction favors H2 under this recipe, concentrated at T1-to-T2; rich CE improves with depth. Negative deep regrets are retained. All T1 fits selected step 100 and stopped at 900; T2..4 selected step 1 and stopped at 800. Step zero is not a selection candidate and steps 2–99 are not evaluated. The same optimization limitation as pilot persists; these results do not establish optimized decoder-family regret.

Sync audit checked 24 unique depth/seed/family fits, identical recorded hidden hashes, 100000 per-token test losses per fit and their means, validation-minimum selection and all 24 saved checkpoint files. Independently checked 31250 unique document IDs and text hashes, split quotas, all next-token labels/position bounds, and the windows file hash. Execution assertions and extraction records verify native T=1..4, exact native-logit parity, shared paired hidden states, nested initialization and extra rich capacity; no new model inference or training was run during sync.

All 3200 pilot documents and all 320 smoke documents are present with unchanged split, tokens, positions, labels and content hashes. This is sequential scaling after pilot inspection, not independent confirmation (`confirmatory: false`). Secondary difficulty and loop-benefit slices are retained in the detailed record. No test-driven retuning was performed.

Extraction took 2996.16 s; summed probe runtimes 485.15 s. Peak allocated GPU memory: 2.82 GiB extraction / 2.85 GiB probes on RTX 5090 D. These are not total pipeline wall time or reserved VRAM.

Detailed provenance, configuration, training histories, per-fit metrics and audit: [1M run record](runs/EXP-001-1M-20260911.json). Large weights, hidden states and per-token loss arrays remain on the server at `/root/autodl-fs/depth-decoder-mismatch/outputs/EXP-001-1M-20260911/`, outside Git.

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

Detailed provenance, config, all validation curves, gradient/norm/RMS diagnostics, selected LRs and per-run metrics: [EXP-001b run record](runs/EXP-001b-20260911.json). Source snapshot SHA-256 `f87539a12850f8db6aa69b3f017f93058c5143a894360146156128ca1e25bd58`, based on `43455b5ff0ce67cf0a9fa636e13bcc8b1e6231e1` plus archived uncommitted code. Per-token arrays and residual checkpoints remain at `/root/autodl-fs/depth-decoder-mismatch/outputs/EXP-001b-20260911/`, outside Git.

## EXP-003A staged CPT pipeline — launched 2026-09-11

No scientific result yet. Implementation and hardware preflight passed; the driver gates control-only LR selection and the 50M primary screen on fresh-data manifest validation and the six-arm 1M smoke (including masked-98K controls). Three unit tests and real-model paired checks passed after correcting clipping-norm numerical sensitivity. See [experiment card](../experiments/EXP-003A-ouro-output-vocab-cpt.md) and [launch record](runs/EXP-003A-20260911.json).

User pause (2026-09-11): automatic training driver stopped before smoke. No smoke, LR-tuning or primary training run completed or remains active. Only engineering preflight has run; data preparation may continue. Await explicit resume before any training.

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

Detailed config, provenance, all histories and audit: [run record](runs/EXP-003C-20260911.json). Source SHA-256: `7f7b43a5988a619c021929bb26beb7e6a463efc15ec321fcc5f089ab2c736a04`; base git `d331751af192be7ef8783ab1d25a10cbb8ee4a90`; upstream MIT reference `fa8b2c5ba73e0350c9a34fbfcd95a582c0f798df`. Server artifacts: `/root/autodl-tmp/EXP-003C-20260911/`. No shared-storage allocation or Ouro training was performed.
