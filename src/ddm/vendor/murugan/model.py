from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .feedback import DecoupledLinear, ProjectedHeadGradientLinearFunction


class ResidualMLPBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.ff(self.norm(hidden))


class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        tie_weights: bool,
        architecture: str = "transformer",
        head_type: str = "linear",
        head_rank: int = 0,
        logit_scale_init: float = 10.0,
        normalized_weight_norm_init: float = 0.0,
        readout_rank: int = 0,
        readout_depths: Optional[Sequence[int]] = None,
        readout_scale: float = 1.0,
        detach_readout_states: bool = False,
        readout_gradient_scale: float = 1.0,
        input_vocab_size: Optional[int] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_vocab_size = vocab_size if input_vocab_size is None else input_vocab_size
        self.d_model = d_model
        self.architecture = architecture
        self.head_type = head_type
        self.head_rank = head_rank
        self.readout_depths = tuple(readout_depths or ())
        self.readout_scale = readout_scale
        self.detach_readout_states = detach_readout_states
        self.readout_gradient_scale = 0.0 if detach_readout_states else readout_gradient_scale
        self.token_embedding = nn.Embedding(self.input_vocab_size, d_model)
        self.position_embedding = (
            nn.Embedding(seq_len, d_model) if architecture == "transformer" else None
        )
        if architecture == "transformer":
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=n_heads,
                        dim_feedforward=d_ff,
                        dropout=dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.final_norm = nn.LayerNorm(d_model)
        elif architecture == "embedding_mlp":
            self.layers = nn.ModuleList(
                [ResidualMLPBlock(d_model, d_ff, dropout) for _ in range(n_layers)]
            )
            self.final_norm = nn.LayerNorm(d_model)
        elif architecture == "embedding_linear":
            self.layers = nn.ModuleList()
            self.final_norm = nn.Identity()
        else:
            raise ValueError(f"Unknown architecture {architecture!r}")
        self.head_projection = (
            nn.Linear(d_model, head_rank, bias=False) if 0 < head_rank < d_model else None
        )
        head_input_dimension = head_rank if self.head_projection is not None else d_model
        self.lm_head = nn.Linear(head_input_dimension, vocab_size, bias=False)
        if tie_weights:
            if self.input_vocab_size != vocab_size:
                raise ValueError("Tied weights require equal input and output vocabularies")
            if self.head_projection is not None:
                raise ValueError("A factorized head cannot share the input embedding matrix")
            self.lm_head.weight = self.token_embedding.weight
        self.logit_scale = (
            nn.Parameter(torch.tensor(logit_scale_init).log())
            if head_type in {"output_norm", "hidden_norm", "cosine"}
            else None
        )
        self.output_log_magnitude = (
            nn.Parameter(torch.zeros(vocab_size)) if head_type == "weight_norm" else None
        )
        self.decoupled_head = DecoupledLinear(self.lm_head.weight)
        self.readout_paths = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, readout_rank, bias=False),
                    nn.Linear(readout_rank, vocab_size, bias=False),
                )
                for _ in self.readout_depths
            ]
        )
        self.apply(self._init_weights)
        if (
            head_type in {"output_norm", "cosine"}
            and normalized_weight_norm_init > 0
            and not tie_weights
        ):
            with torch.no_grad():
                self.lm_head.weight.copy_(
                    F.normalize(self.lm_head.weight, dim=-1) * normalized_weight_norm_init
                )
        if self.output_log_magnitude is not None:
            with torch.no_grad():
                self.output_log_magnitude.copy_(
                    self.lm_head.weight.norm(dim=-1).clamp_min(1e-12).log()
                )

    def effective_output_weight(self) -> torch.Tensor:
        """Materialize the `V x D` linear map used by the forward head."""
        if self.head_projection is None:
            if self.output_log_magnitude is not None:
                return F.normalize(self.lm_head.weight, dim=-1) * self.output_log_magnitude.exp()[
                    :, None
                ]
            return self.lm_head.weight
        return self.lm_head.weight.matmul(self.head_projection.weight)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        feedback: Optional[torch.Tensor] = None,
        match_scale: bool = True,
        project_head_gradient: bool = False,
        match_head_gradient_scale: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hidden = self.token_embedding(tokens)
        mask = None
        if self.position_embedding is not None:
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            hidden = hidden + self.position_embedding(positions)[None, :, :]
            mask = nn.Transformer.generate_square_subsequent_mask(
                tokens.shape[1], device=tokens.device
            )
        states = []
        for layer in self.layers:
            if self.architecture == "transformer":
                hidden = layer(hidden, src_mask=mask, is_causal=True)
            else:
                hidden = layer(hidden)
            states.append(hidden)
        hidden = self.final_norm(hidden)
        if feedback is None:
            if self.head_type == "linear" and project_head_gradient:
                if self.head_projection is not None:
                    raise ValueError(
                        "Projected head gradients for factorized heads are not implemented"
                    )
                logits = ProjectedHeadGradientLinearFunction.apply(
                    hidden, self.lm_head.weight, match_head_gradient_scale
                )
            elif self.head_type == "linear":
                projected = (
                    hidden if self.head_projection is None else self.head_projection(hidden)
                )
                logits = self.lm_head(projected)
            elif self.head_type == "weight_norm":
                assert self.output_log_magnitude is not None
                logits = F.linear(hidden, self.effective_output_weight())
            elif self.head_type == "output_norm":
                assert self.logit_scale is not None
                logits = self.logit_scale.exp() * F.linear(
                    hidden,
                    F.normalize(self.lm_head.weight, dim=-1),
                )
            elif self.head_type == "hidden_norm":
                assert self.logit_scale is not None
                logits = self.logit_scale.exp() * self.lm_head(F.normalize(hidden, dim=-1))
            elif self.head_type == "cosine":
                assert self.logit_scale is not None
                logits = self.logit_scale.exp() * F.linear(
                    F.normalize(hidden, dim=-1),
                    F.normalize(self.lm_head.weight, dim=-1),
                )
            else:
                raise ValueError(f"Unknown head type {self.head_type!r}")
        else:
            if self.head_type != "linear":
                raise ValueError("Decoupled feedback is implemented only for a linear head")
            if self.head_projection is not None:
                raise ValueError("Decoupled feedback for factorized heads is not implemented")
            logits = self.decoupled_head(hidden, feedback, match_scale)
        for depth, path in zip(self.readout_depths, self.readout_paths):
            state = states[depth]
            # Forward-identical gradient gate: its value is exactly `state`, while its
            # derivative with respect to `state` is `readout_gradient_scale`.
            detached = state.detach()
            state = detached + self.readout_gradient_scale * (state - detached)
            logits = logits + self.readout_scale * path(self.final_norm(state))
        return logits, states

    def feedback_vocabulary_matrix(self) -> torch.Tensor:
        """Vocabulary columns that provide exact gradient paths to hidden states."""
        matrices = [self.effective_output_weight()]
        if self.readout_gradient_scale != 0.0:
            matrices.extend(path[1].weight for path in self.readout_paths)
        return torch.cat(matrices, dim=1)

    def readout_complementarity_loss(self) -> torch.Tensor:
        if not self.readout_paths:
            return self.lm_head.weight.new_zeros(())
        references = [F.normalize(self.effective_output_weight().detach(), dim=0)]
        penalties = []
        for path in self.readout_paths:
            columns = F.normalize(path[1].weight, dim=0)
            penalties.extend(
                reference.t().matmul(columns).square().mean() for reference in references
            )
            references.append(columns.detach())
        return torch.stack(penalties).mean()
