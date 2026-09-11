# Results

No confirmatory runs yet.

## Primary summary

| Experiment | Model | Intervention | Primary statistic | Interpretation | Status |
|---|---|---|---:|---|---|
| EXP-001 | Ouro-1.4B | native recurrent depth T=1..4 | `dR_head/dT`, `R(4)-R(1)` | decoder-family depth interaction | not started |
| EXP-002 | Nanbeige4.2-3B-Base | native T=1→2 replication | paired Head-Regret contrast | architecture replication | not started |
| EXP-003 | controlled small Ouro-style LoopLM | output-only `V_out` inflation × recurrent depth | `I_active` | pure output-space causal test | not started |
| EXP-004 | matched small models | natural tokenizer vocab × recurrent depth | BPB-normalized interaction | natural tokenizer replication | gated on EXP-003 / Phase B |

Do not add exploratory OOD-depth results to the confirmatory columns.
