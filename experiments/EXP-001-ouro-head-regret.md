# EXP-001 — Ouro native-depth Head Regret

## Research link
- Hypotheses: H1 vs H2 vs H0
- Claims affected: C1, C2, C3
- Status: proposed

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
Pending.
