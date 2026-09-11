# EXP-004 — Natural tokenizer vocabulary × recurrent-depth factorial

## Gate
Run only after EXP-003 establishes whether pure output-class inflation shows a depth interaction worth explaining. This experiment changes the tokenizer itself and is therefore a less isolated but more realistic test.

## Question
Holding raw data and backbone design fixed, does natural tokenizer vocabulary size change the slope of Head Regret or downstream modeling quality over recurrent depth?

## Design
At minimum:
- tokenizer vocab: ~16K vs ~64K BPE trained on the same tokenizer-training corpus;
- recurrence: shallow vs deep, preferably multiple T values rather than only 2×2;
- same raw pretraining text, document order, optimization recipe and non-embedding architecture;
- evaluate language modeling in bits per raw byte (BPB), not raw token CE.

## Primary interaction
`I = [R(T_hi,V_hi)-R(T_lo,V_hi)] - [R(T_hi,V_lo)-R(T_lo,V_lo)]`.

H1 predicts `I > 0`: larger natural vocab amplifies the depth–decoder mismatch.

## Required reporting
- total parameters and non-embedding parameters separately;
- token count and raw-byte count;
- compression ratio / tokens per raw byte;
- BPB, not only token-level perplexity;
- training FLOPs and decode-cost implications of the different tokenizers.

## Causal controls
- same corpus bytes and document order;
- matched raw-data exposure;
- tied/untied embedding policy fixed;
- same backbone width and recurrent block;
- same probe protocol when measuring Head Regret.

## Interpretation guardrail
Unlike EXP-003, this experiment changes tokenization, sequence length, token frequencies and compositional structure together with vocabulary size. A positive result supports a natural-tokenizer interaction, but does not by itself identify output-class count as the mechanism.

## Result
Pending.
