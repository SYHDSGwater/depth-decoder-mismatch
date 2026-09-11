# EXP-001 — Ouro native-depth Head Regret

## Research link
- Hypotheses: H1 vs H2 vs H0
- Claims affected: C1, C2, C3
- Status: proposed

## Question
Across Ouro-1.4B's native recurrent steps `T=1..4`, does held-out regret between a refit linear head and a nested richer decoder increase, decrease, or remain flat?

## Intervention
Only the recurrent depth/index of the frozen hidden state changes. The same checkpoint is run once at its native max depth and per-step final hidden states are paired by the same token positions.

## Controls / invariants
- checkpoint/tokenizer revision fixed;
- backbone frozen;
- identical documents, token positions, labels and split at every T;
- same native final normalization convention used at every loop step;
- same probe initialization policy, optimizer, LR search, max steps, early stopping and seed set;
- native T=1..4 only for confirmatory analysis;
- no task prompts or generation decoding involved.

## Probe families
1. `native_linear`: untouched native LM head (diagnostic).
2. `linear_refit`: copy native W, then refit W on probe train split.
3. `rich_residual`: same W initialization plus `U GELU(Ah)` residual; U zero-init so it begins exactly at the linear function.

Primary regret uses 2 vs 3.

## Pre-registered predictions
### H1 DDM
`R(1) < R(2) < R(3) < R(4)` approximately; positive slope / high-vs-low paired contrast. Rich-head held-out loss should be stable or improve over the range.

### H2 depth-as-linearization
Regret decreases; rich decoder helps T=1 more than T=4.

### H0
Regret differences are small relative to bootstrap/seed uncertainty.

## Invalidity
Invalid if hidden states and labels are not exactly paired across T, if probe hyperparameters are tuned asymmetrically after seeing test results, if T values are produced by separate token samples, or if rich head fails to contain the linear function at initialization.

## Robustness before claim update
- >=3 probe seeds;
- rich rank/width sweep (e.g. 256, 512, 1024) to check sign stability;
- train-set-size sweep to expose overfitting;
- document-level or sequence-level bootstrap, not naive independent-token CI;
- report native/refit gap separately.

## Commands (target interface)
```bash
python scripts/extract_hidden.py --config configs/ouro.yaml --split train --text-file data/train.txt
python scripts/extract_hidden.py --config configs/ouro.yaml --split validation --text-file data/validation.txt
python scripts/extract_hidden.py --config configs/ouro.yaml --split test --text-file data/test.txt

for T in 1 2 3 4; do
  for seed in 11 29 47; do
    python scripts/train_probe.py --config configs/experiments/exp001-linear.yaml --depth $T --seed $seed
    python scripts/train_probe.py --config configs/experiments/exp001-rich.yaml --depth $T --seed $seed
  done
done

python scripts/analyze_regret.py --experiment EXP-001 --probe-root outputs/probes/EXP-001 --split test
```

## Result
Pending.
