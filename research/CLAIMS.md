# Claim ↔ Evidence Matrix

Do not promote a claim until its required evidence exists.

| ID | Candidate claim | Required evidence | Current status |
|---|---|---|---|
| C1 | Linear-vs-rich decoder regret changes systematically with native recurrent depth in Ouro. | Valid paired EXP-001 with robust slope CI across probe seeds/capacity. | untested |
| C2 | The sign is positive (supports DDM). | C1 with `dR/dT > 0`, plus rich-head loss not degrading enough to explain slope. | untested |
| C3 | The sign is negative (supports depth-as-linearization/decoder compensation). | C1 with `dR/dT < 0`, robust to probe capacity. | untested |
| C4 | Effect replicates across looped architectures. | Valid Nanbeige EXP-002 with same sign in native T=1→2 range. | untested |
| C5 | Larger vocab causally amplifies depth–decoder mismatch. | Matched Phase-B training with tokenizer/vocab × depth factorial and BPB-normalized positive interaction. | untested |
| C6 | Claude's small vocab was chosen because of DDM. | Proprietary causal/design evidence from Anthropic, not supplied by this project. | **not claimable** |

## Overclaim guardrails
- Ouro vs Nanbeige cannot establish C5.
- LOTUS verbalizability does not establish or refute C1 by itself.
- Gradient projection norm does not establish C2.
- OOD loop depths do not count toward confirmatory C1/C2/C3.
