# Claim ↔ Evidence Matrix

Do not promote a claim until its required evidence exists.

| ID | Candidate claim | Required evidence | Current status |
|---|---|---|---|
| C1 | Linear-vs-rich decoder regret changes systematically with native recurrent depth in Ouro. | Valid paired EXP-001 with robust slope CI across probe seeds/capacity. | observed in pilot/1M, non-confirmatory |
| C2 | The sign is positive (supports forward DDM). | C1 with `dR/dT > 0`, plus rich-head loss not degrading enough to explain slope. | not supported by current EXP-001 |
| C3 | The sign is negative (supports depth-as-linearization/decoder compensation). | EXP-001 negative slope plus EXP-001b showing `G_nonlin(1)>0` and `G_nonlin(2..4)≈0` under frozen-native-head matched residual controls, ideally followed by independent replication. | provisional; EXP-001b pending |
| C4 | Effect replicates across looped architectures. | Valid Nanbeige EXP-002 with same sign in native T=1→2 range. | untested |
| C5 | Pure output-space inflation causally becomes more harmful as recurrent depth increases. | EXP-003 positive paired `depth × V_out` interaction, preferably in `CE_active` after removing dummy probability-mass competition. | untested |
| C6 | Natural larger tokenizer vocab causally amplifies depth–decoder mismatch. | EXP-004 matched tokenizer-vocab × depth training with BPB-normalized positive interaction. | untested |
| C7 | Claude's small vocab was chosen because of DDM. | Proprietary causal/design evidence from Anthropic, not supplied by this project. | **not claimable** |

## EXP-001b-specific guardrail
A nonlinear residual improving over the frozen native head is not sufficient decoder-expressivity evidence. The primary evidence is the incremental gain over a matched linear residual adapter:

`G_nonlin(T) = CE_linear_residual(T) - CE_nonlinear_residual(T)`.

Only a stable positive `G_nonlin` demonstrates value from nonlinear decoder capacity beyond low-rank/domain adaptation.

## Overclaim guardrails
- Ouro vs Nanbeige cannot establish C5 or C6.
- EXP-001b reuses a previously inspected 1M test set and cannot by itself convert C3 into a confirmatory claim.
- EXP-003 never-target classes test output dimensionality/competition, not the full effect of a natural tokenizer; they cannot establish C6.
- A positive raw-CE interaction in EXP-003 is insufficient for a strong bottleneck claim if it disappears after conditioning on the active vocabulary.
- LOTUS verbalizability does not establish or refute C1 by itself.
- Gradient projection norm does not establish C2 or C5.
- OOD loop depths do not count toward confirmatory C1/C2/C3.
