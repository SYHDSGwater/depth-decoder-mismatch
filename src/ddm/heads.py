from __future__ import annotations

import torch
from torch import nn


class LinearRefitHead(nn.Module):
    """Bias-free linear LM head used as the primary restricted family."""

    def __init__(self, hidden_size: int, vocab_size: int, init_weight: torch.Tensor | None = None):
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)
        if init_weight is not None:
            if tuple(init_weight.shape) != tuple(self.proj.weight.shape):
                raise ValueError(f"weight shape {tuple(init_weight.shape)} != {tuple(self.proj.weight.shape)}")
            with torch.no_grad():
                self.proj.weight.copy_(init_weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


class ResidualRichHead(nn.Module):
    """Nested richer decoder: Wh + U GELU(Ah).

    U is zero-initialized, so the module starts exactly at the supplied linear
    function while adding nonlinear output features during probe training.
    """

    def __init__(self, hidden_size: int, vocab_size: int, residual_width: int, init_weight: torch.Tensor | None = None):
        super().__init__()
        if residual_width <= 0:
            raise ValueError("residual_width must be positive")
        self.base = nn.Linear(hidden_size, vocab_size, bias=False)
        self.in_proj = nn.Linear(hidden_size, residual_width, bias=True)
        self.out_proj = nn.Linear(residual_width, vocab_size, bias=False)
        self.act = nn.GELU()
        nn.init.zeros_(self.out_proj.weight)
        if init_weight is not None:
            if tuple(init_weight.shape) != tuple(self.base.weight.shape):
                raise ValueError(f"weight shape {tuple(init_weight.shape)} != {tuple(self.base.weight.shape)}")
            with torch.no_grad():
                self.base.weight.copy_(init_weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.base(hidden) + self.out_proj(self.act(self.in_proj(hidden)))


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
