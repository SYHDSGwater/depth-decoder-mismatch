# Literature / Evidence Map

## Direct architectural basis

### Ouro — *Scaling Latent Reasoning via Looped Language Models* (2025)
- Open looped LM family with shared-weight recurrent computation and adaptive depth.
- `Ouro-1.4B`: hidden size 2048, vocab 49,152, 4 recurrent steps; public model code exposes per-step hidden states.
- Relevance: cleanest Phase-A within-checkpoint depth sweep.
- Paper: https://arxiv.org/abs/2510.25741
- Model: https://huggingface.co/ByteDance/Ouro-1.4B

### Nanbeige4.2-3B (2026)
- 22 unique decoder layers, `num_loops=2`, hidden size 3072, vocab 166,144.
- Relevance: independent looped checkpoint and substantially higher `V/D` natural contrast, but not a causal vocab comparison.
- Model: https://huggingface.co/Nanbeige/Nanbeige4.2-3B-Base

### LOTUS — *Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers* (2026)
- Important counterexample to the strong claim that deeper latent reasoning must leave the base LM-head-readable manifold.
- Its training explicitly encourages latent-to-explicit alignment, so it does not directly settle marginal decoder regret.
- Paper: https://arxiv.org/abs/2606.31779

## Decoder geometry

### *Breaking the Softmax Bottleneck* (2018)
- Standard linear-softmax language models restrict the family of cross-context log-probability matrices.
- Motivates richer output families such as Mixture of Softmaxes.
- Paper: https://arxiv.org/abs/1711.03953

### *Which Transformer architecture fits my data? A vocabulary bottleneck in self-attention* (2021)
- Links embedding/vocabulary rank to depth-vs-width efficiency; smaller vocabularies can favor deeper architectures in the paper's regime.
- Adjacent rather than direct evidence for DDM.
- Paper: https://proceedings.mlr.press/v139/wies21a.html

## Backward bottleneck debate

### Godey & Artzi — *Lost in Backpropagation: The LM Head is a Gradient Bottleneck* (2026)
- Reports 95–99% logit-gradient norm can lie outside the subspace returned through the LM head and argues for optimization harm.
- Paper: https://arxiv.org/abs/2603.10145

### Murugan — *Does the LM Head Create a Harmful Gradient Bottleneck? A Causal Test* (2026)
- Confirms geometric compression but shows that large projected-away norm does not by itself establish harmful optimization; forward capacity effects are stronger in its controlled experiments, and adding never-target output classes did not impair learning.
- Required counterevidence: DDM should be tested in forward decoder regret, not inferred from gradient norm.
- Paper: https://arxiv.org/abs/2608.16671

## Source research note
- JiangHongwei, *Claude 的 16K 赌注：Tokenizer 是否正在成为 Reasoning Architecture 的核心部分？* (2026-09-04)
- https://syhdsgwater.github.io/zh/notes/claude-16k-tokenizer-reasoning-architecture/
- The note defines Head Regret and the depth × vocabulary interaction this repo operationalizes.
