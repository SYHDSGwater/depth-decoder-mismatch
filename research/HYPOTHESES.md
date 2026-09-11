# Hypotheses

## H1 — Depth–Decoder Mismatch (DDM)
**Mechanism.** Recurrent depth increases the complexity/fineness of context representations while the decoder remains a fixed `D -> V` linear-softmax family. The additional backbone distinctions become increasingly under-exploited by a linear head relative to a richer nested decoder family.

**Predictions.**
1. `R_head(T)` increases over native loop depth.
2. The increase is not explained by rich-head loss deteriorating at deeper T.
3. In a matched vocab × depth study, `dR/dT` is larger at larger V (positive interaction).
4. Rich-head gains persist on held-out data and across probe seeds/capacity sweeps.

**Weakening evidence.** Flat/decreasing regret across native depth, or an apparent increase driven only by OOD-depth representation degradation.

## H2 — Depth-as-Linearization / Decoder Compensation
**Mechanism.** More recurrent computation moves nonlinear function complexity into the backbone and produces features that are easier for a simple LM head to decode. A rich decoder mainly compensates for insufficient shallow computation.

**Predictions.**
1. `R_head(T)` decreases with T.
2. Shallow states gain more from the rich decoder than deep states.
3. The linear decoder captures a growing fraction of the rich decoder's attainable improvement as depth increases.

This is the explicit competing hypothesis suggested in the earlier Ouro discussion; it predicts the opposite sign from H1.

## H0 — No material decoder-depth interaction
**Mechanism.** Recurrence changes representation quality, but linear and richer decoders benefit similarly.

**Prediction.** `R_head(T)` is approximately constant within confidence intervals and robust to probe capacity/budget.

## H3 — Apparent mismatch is probe or OOD artifact
Potential artifacts:
- depth not seen during checkpoint training;
- rich head overfits due to parameter count;
- unequal optimization budget / early stopping;
- hidden-state dataset differs by depth;
- tokenization/sequence-length confounds in cross-vocab comparisons.

H3 is handled by native-depth confirmatory ranges, nested head initialization, matched splits/training, capacity sweeps, paired examples, and BPB in Phase B.
