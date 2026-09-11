# EXP-003 — Matched vocabulary × recurrent-depth factorial

## Gate
Run only if Phase A yields a stable nonzero decoder-depth interaction worth explaining.

## Question
Holding raw data and backbone design fixed, does vocabulary size change the slope of Head Regret over recurrent depth?

## Design
At minimum:
- vocab: ~16K vs ~64K BPE trained on the same tokenizer-training corpus;
- recurrence: shallow vs deep (prefer multiple T values rather than only 2×2);
- same raw pretraining text, optimization recipe and non-embedding architecture;
- evaluate NLL in **bits per raw byte**.

## Primary interaction
`I = [R(T_hi,V_hi)-R(T_lo,V_hi)] - [R(T_hi,V_lo)-R(T_lo,V_lo)]`.

H1 predicts `I>0`; H2 does not require that sign.

## Required reporting
Report total params and separately non-embedding params because vocab size changes embedding/head parameter count. Report token count *and* raw-byte count; do not equalize training only by token count because tokenizer compression changes the amount of underlying text seen.

## Causal controls
- same corpus bytes and document order;
- matched raw-data exposure;
- tied/untied embedding policy fixed;
- same probe protocol;
- optional artificial never-target output-class expansion as a geometry-only diagnostic, explicitly separated from natural tokenizer vocab change.
