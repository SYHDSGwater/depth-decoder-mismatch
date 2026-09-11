# EXP-001 — Ouro native-depth Head Regret

## Research link
- Hypotheses: H1 vs H2 vs H0
- Claims affected: C1, C2, C3
- Status: smoke, pilot and 1M completed (2026-09-11); confirmatory pending

## Question
Across Ouro-1.4B's native recurrent steps `T=1..4`, does held-out regret between a refit linear head and a nested richer decoder increase, decrease, or remain flat?

Primary quantity:

`R_head(T) = CE_linear_refit(T) - CE_rich(T)`

Primary contrasts:
- slope of `R_head(T)` over `T=1..4`;
- `Delta R = R_head(4) - R_head(1)`.

## Intervention
Only the recurrent depth/index of the frozen hidden state changes. The same checkpoint is run once at its native max depth and per-step final hidden states are paired by the exact same token positions.

## Primary corpus and sampling recipe

### Corpus
Use `HuggingFaceFW/fineweb-edu`, preferably the `sample-10BT` subset, as the primary corpus.

Rationale: EXP-001 is a representation/decoder experiment, not a downstream benchmark. The primary test should therefore use broad next-token prediction data close to a generic pretraining distribution rather than GSM8K/AIME-style prompted evaluation.

### Context construction
- tokenize documents with the Ouro tokenizer;
- split documents before windowing;
- use fixed non-overlapping or deterministically generated windows of `seq_len=1024`;
- do not sample target positions before position 128;
- sample `32` target positions per window using a fixed seed;
- every selected position defines one example `x_1:t -> y_{t+1}`.

For every selected target position, persist the exact paired record:

```text
document_id
window_id
position
target_token
h_T1
h_T2
h_T3
h_T4
```

Do not construct separate token samples for different depths.

### Split
Split by document, not by token or window:
- train: 80%
- validation: 10%
- test: 10%

No document may contribute positions to more than one split.

### Target sample counts
Use staged scaling:

| stage | paired target positions | purpose |
| --- | ---: | --- |
| smoke | 10k–20k | verify extraction, labels, pairing and probe plumbing |
| pilot | 100k–200k | estimate effect magnitude and storage/runtime |
| confirmatory | 1,000,000 | final EXP-001 analysis |

Confirmatory target allocation:
- train: 800k
- validation: 100k
- test: 100k

If storage/runtime becomes limiting, reducing the confirmatory total to 500k is acceptable only if decided before inspecting the final test effect.

## Controls / invariants
- checkpoint/tokenizer revision fixed;
- backbone frozen;
- identical documents, windows, target positions, labels and split at every T;
- same native final normalization convention used at every loop step;
- same probe initialization policy, optimizer, LR search, max steps, early stopping and seed set;
- native `T=1..4` only for confirmatory analysis;
- no task prompts or generation decoding involved;
- no test-set-driven probe or sampling changes.

## Probe families
1. `native_linear`: untouched native LM head, diagnostic only.
2. `linear_refit`: copy native `W`, then refit `W` on the probe train split.
3. `rich_residual`: same `W` initialization plus `U GELU(Ah)` residual; `U` is zero-initialized so the rich model begins exactly at the linear function.

Primary regret uses 2 vs 3. Both probes must use the same frozen hidden-state dataset and the same train/validation/test split. Hyperparameter search must be symmetric. Final test CE is evaluated once after validation-based model selection / early stopping.

## Pre-registered predictions

### H1 — Depth–Decoder Mismatch
`R(1) < R(2) < R(3) < R(4)` approximately, with positive `R_head` slope and positive `Delta R`.

Interpretation: extra recurrent computation increases backbone representational capacity faster than the fixed-width linear decoder family can exploit it.

### H2 — Depth-as-Linearization / Decoder Compensation
`R_head(T)` decreases with depth; the richer decoder helps shallow states more than deep states.

Interpretation: recurrent depth absorbs nonlinear function complexity into the backbone and makes final representations more linearly decodable.

### H0 — No meaningful depth–decoder interaction
The slope and `Delta R` are small relative to probe-seed and document-bootstrap uncertainty.

## Primary statistical analysis

Report for each `T`:
- held-out CE of `native_linear`;
- held-out CE of `linear_refit`;
- held-out CE of `rich_residual`;
- `R_head(T)`.

Primary test statistics:

```text
slope_R = slope of R_head against T in {1,2,3,4}
Delta_R = R_head(4) - R_head(1)
```

Uncertainty must use document-level clustered resampling, not IID token bootstrap. Window-level cluster bootstrap is acceptable only if document IDs are unavailable, and must be reported as a weaker fallback.

Use >=3 probe seeds (default: `11, 29, 47`) and report both per-seed values and seed-aggregated estimates.

## Pre-registered secondary slices

These analyses use the same held-out test positions; they do not alter primary sampling.

### A. Difficulty quartiles
For each test token, compute native-head T=1 loss:

`difficulty_i = -log p_native(y_i | h_i^(1))`

Partition test examples into quartiles `Q1...Q4` from easiest to hardest and report `R_head(T | Qk)`.

Mechanistic expectation under H1: the positive depth-regret slope may be stronger on hard tokens if additional recurrent refinement is most useful there.

### B. Loop-benefit quartiles
Using the untouched native LM head, define:

`loop_benefit_i = loss_native(T=1, i) - loss_native(T=4, i)`

Positive values mean the same token prediction benefits from extra recurrent computation. Partition the test set into quartiles by `loop_benefit_i` and report the Head-Regret slope within each quartile.

A stronger H1 prediction is:

`d R_head / dT` should be larger among tokens that actually benefit more from recurrent depth.

This is secondary only; do not sample or exclude examples based on loop benefit.

## Replication corpus
After the primary FineWeb-Edu result is frozen, repeat the same protocol on a reasoning-heavy natural-text corpus (math/science derivations or solution text; e.g. an OpenWebMath-like corpus).

Do not use GSM8K/AIME prompted answer accuracy as the primary EXP-001 dataset. If reasoning benchmarks are later added, treat them as downstream external validation rather than the main Head-Regret estimate.

Keep web and math results separate instead of mixing domains into one aggregate corpus.

## Robustness before claim update
- >=3 probe seeds;
- rich rank/width sweep, e.g. `256, 512, 1024`, to check sign stability;
- train-set-size sweep to expose sample-efficiency / overfitting artifacts;
- document-level clustered bootstrap;
- report native-vs-refit linear gap separately;
- verify that rich-head gains are not driven by a few documents or token types;
- confirm that rich-head held-out CE is not worsened by optimization failure despite containing the linear family at initialization.

## Invalidity
EXP-001 is invalid or must be rerun if any of the following occurs:
- hidden states and labels are not exactly paired across T;
- T values are produced from independently sampled token positions;
- train/validation/test split leaks documents across partitions;
- target positions or corpus composition are changed after inspecting final test effects;
- probe hyperparameters are tuned asymmetrically after seeing test results;
- test data participates in early stopping or model selection;
- rich head does not contain the linear function at initialization;
- uncertainty is reported using naive IID-token confidence intervals only.

## Target extraction interface
The current scripts may evolve, but the intended confirmatory data interface is equivalent to:

```bash
python scripts/extract_hidden.py \
  --config configs/ouro.yaml \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset-config sample-10BT \
  --seq-len 1024 \
  --positions-per-window 32 \
  --min-position 128 \
  --target-count 1000000 \
  --split-policy document-80-10-10 \
  --seed 20260911
```

Run the smoke and pilot stages first with smaller `--target-count`; do not alter the confirmatory recipe based solely on a favorable pilot test-set effect.

## Probe training interface

```bash
for T in 1 2 3 4; do
  for seed in 11 29 47; do
    python scripts/train_probe.py --config configs/experiments/exp001-linear.yaml --depth $T --seed $seed
    python scripts/train_probe.py --config configs/experiments/exp001-rich.yaml --depth $T --seed $seed
  done
done

python scripts/analyze_regret.py \
  --experiment EXP-001 \
  --probe-root outputs/probes/EXP-001 \
  --split test \
  --cluster-by document_id \
  --secondary-slices difficulty,loop_benefit
```

## Result
### Pilot plan frozen before execution (2026-09-11)

Use [exp001-pilot.yaml](../configs/experiments/exp001-pilot.yaml) with the shared staged runner `scripts/smoke_exp001.py`. Target 102,400 paired positions from 3,200 documents (2560/320/320 documents; 81920/10240/10240 targets). Keep the smoke model/tokenizer/dataset revisions, first-shard/first-full-window policy, document hash split and position seed unchanged. This is nested staged scaling: previously observed smoke positions stay in their original splits, not an independent replication.

Restore the original probe budget for both families: AdamW, LR 1e-4 constant, weight decay 0, batch 256, at most 5000 steps, validation every 100 steps plus step 1, patience 8; seeds 11/29/47; rich width 512. No LR/width search or condition-specific adjustment is included. Keep BF16 extraction/storage, FP32 probing and TF32 disabled. Save every validation-selected checkpoint, then evaluate its test split once. Report negative and null results without rerunning toward a desired sign.

Primary pilot quantities remain held-out linear-refit minus rich CE, native-depth slope and T4-minus-T1 contrast. Report per-seed values and 2000-replicate document bootstrap intervals conditional on the three seeds; difficulty and loop-benefit slices use the same fixed test positions. In addition to data/shape/nesting checks, report validation-selected steps and whether each fit exhausts its budget. Numerical completion is not proof of optimizer convergence. A positive slope is not H1 evidence if rich held-out CE degrades with depth. Pilot is exploratory; confirmatory status remains pending.

Expected disk allocation is about 1.9 GB for paired hidden states plus 10.9 GB for 24 FP32 selected head checkpoints. Check available space with a 2 GiB reserve before sampling. Model hashes, runtime source hash, all configuration fields and per-run metrics are persisted. No backbone fitting or benchmark prompts are introduced.

**Pre-outcome data-quality amendment:** The first pilot preparation stopped on exact duplicate text under different document IDs crossing partitions. No pilot hidden extraction, training or test evaluation had occurred. Enable exact-text SHA-256 deduplication before splitting, preserving the first corpus occurrence and its original document ID. Fill the same quotas in the same corpus order. All model, position, split-hash and optimization settings remain fixed. Consequently the final pilot is no longer guaranteed to be a strict superset of every smoke position; overlap and content/split consistency must be reported rather than assumed. Retain the failed preparation log, initial config/source snapshot, and the deduplicated rerun as separate attempts.

### Completed smoke
Smoke completed on 2026-09-11 using the dedicated [smoke config](../configs/experiments/exp001-smoke.yaml) and [runner](../scripts/smoke_exp001.py). The original extraction/training entrypoints and confirmatory defaults were not changed.

Executed smoke settings: 10,240 paired positions from 320 FineWeb-Edu sample-10BT documents, 1024-token windows, 32 positions/window at predictor indices >=128, fixed document split and seed 20260911. For bounded plumbing validation, use the first full window per eligible document in the pinned first shard and exactly 8192/1024/1024 targets. Native T=1..4 and BF16 extraction/storage are fixed. Both head families use FP32, LR 1e-4, batch 256, 50 steps, evaluation every 10 steps plus step 1, patience 8, and all three seeds. These smoke-specific budget settings do not replace the 5000-step probe configs.

Example invocation on the prepared server, with the isolated environment activated and `PYTHONPATH=src`:

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/smoke_exp001.py prepare --model-path /root/autodl-tmp/models/Ouro-1.4B --output outputs/EXP-001-smoke-new
python scripts/smoke_exp001.py extract --model-path /root/autodl-tmp/models/Ouro-1.4B --output outputs/EXP-001-smoke-new
python scripts/smoke_exp001.py probes --model-path /root/autodl-tmp/models/Ouro-1.4B --output outputs/EXP-001-smoke-new
```

Outcome: all 24 probes completed; exact native-logit parity, shared frozen data, disjoint splits and nested/extra rich capacity verified. Native held-out CE decreased from 3.654745 at T=1 to 2.271101 at T=4. Smoke regret slope was +0.00009894 with document-bootstrap interval [-0.00344752,+0.00366996]. Eighteen runs selected step 1, so these short-budget, near-zero regret estimates do not decide H1/H2/H0. Confirmatory result remains pending.

See [RESULTS](../research/RESULTS.md) and the [complete run record](../research/runs/EXP-001-smoke-20260911.json), including model/dataset revisions, per-run metadata, negative/null findings, and the preparation-reader repair.

### Completed pilot review (2026-09-11)

Run `EXP-001-pilot-20260911-r2` completed all 24 fits on 102,400 targets. Head regret at T=1..4 was [0.082553, -0.000061, -0.000086, -0.000084] nats/token. Slope = -0.024794 (2000-replicate document-bootstrap 95% percentile interval [-0.026856,-0.022707]); T4-minus-T1 contrast = -0.082637 (interval [-0.089513,-0.075680]). All three seed slopes are negative. This is a pilot trend favoring H2 over H1, concentrated in T1-to-T2, not a confirmatory or converged-family conclusion. Rich held-out CE decreases with depth.

T1 fits selected step 100 and stopped at 900; all T2..4 fits selected step 1 and stopped at 800, with later validation deterioration. No fit exhausted the 5000-step cap. Step-zero initialization was not a selection candidate and the interval between steps 1 and 100 was not evaluated; native diagnostic CE slightly outperforms selected deeper refits. These limits prevent claiming optimized deep decoder-family regret. No retuning or test reevaluation was performed during review.

Independent review verified 3200 unique document IDs and text hashes, target alignment and the 81920/10240/10240 split; recomputed means and the document bootstrap; checked identical hidden-file hashes across all fits. Despite the pre-outcome deduplication amendment, all 320 smoke windows are present unchanged in this actual pilot, so it is not independent replication. See [RESULTS](../research/RESULTS.md) and [pilot run record](../research/runs/EXP-001-pilot-20260911.json).

### 1M scale-up frozen before execution (2026-09-11)

User requested continuation to 1,000,000 paired target positions after the pilot review. Config: `configs/experiments/exp001-1m.yaml`, stage `scale_1m`. Exact document quotas are 25,000/3,125/3,125 (800,000/100,000/100,000 targets). Preserve the pinned Ouro-1.4B/tokenizer/corpus revisions, first sample-10BT shard, first full window per eligible exact-text-deduplicated document, split before windowing, seq_len 1024, 32 positions/window, minimum predictor position 128 and seed 20260911.

Only sample count and storage location change from the deduplicated pilot. All 24 probes retain native T=1..4, seeds 11/29/47, rich width 512, FP32 AdamW LR 1e-4 constant, weight decay 0, batch 256, max 5000 steps, validation every 100 plus step 1, patience 8. BF16 extraction/storage and TF32 disabled remain fixed; save selected checkpoints and evaluate trained heads on test only after validation selection. No step-zero checkpoint, added learning-rate search, or asymmetrical optimizer change is introduced in response to pilot outcomes. Record optimization limitations even if they persist.

Primary quantity remains held-out `linear_refit - rich` CE, its native-depth slope and T4-T1 contrast; use 2000 document bootstrap replicates conditional on the three seeds, and retain both pre-registered secondary slices. H1/H2/H0 and the rich-loss degradation qualification remain unchanged. Pairing, label, normalization, native-depth, data leakage or nested-capacity failures invalidate the run. All negative/null findings must be reported.

This sequential scale-up uses the same corpus order after pilot test inspection; quantify overlap. It is not an independent untouched confirmation and stays `confirmatory: false`, despite reaching the originally planned 1M sample count. No optimizer-convergence claim follows solely from reaching this size.

Storage: `/root/autodl-fs/depth-decoder-mismatch/outputs/EXP-001-1M-20260911/` on the existing shared filesystem (about 198 GB available at preflight). Expected hidden tensor ~16.6 GB and selected heads ~10.9 GB, plus sampled-window records, metadata and reserve. The server currently reports RTX 5090 D and a 62 GiB container memory limit; preserve device information per run. Source code and launch/config snapshots are archived before execution; the pilot directory is retained.

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

Detailed provenance, configuration, training histories, per-fit metrics and audit: [1M run record](../research/runs/EXP-001-1M-20260911.json). Large weights, hidden states and per-token loss arrays remain on the server at `/root/autodl-fs/depth-decoder-mismatch/outputs/EXP-001-1M-20260911/`, outside Git.
