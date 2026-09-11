"""Matched residual adapters with an immutable native projection (EXP-001b)."""
import torch
from torch import nn
from torch.nn import functional as F


EVAL_STEPS = (0, 1, 2, 5, 10, 20, 50, 100, 200, 400, 800, 1200, 1600, 2000)
FAMILIES = ('linear_residual', 'nonlinear_residual')


class FrozenResidualHead(nn.Module):
    def __init__(self, native_weight, width, family):
        super().__init__()
        if family not in FAMILIES or width <= 0:
            raise ValueError('Invalid residual family or width')
        self.register_buffer('native_weight', native_weight.detach().clone())
        self.a = nn.Linear(native_weight.shape[1], width, bias=True)
        self.u = nn.Linear(width, native_weight.shape[0], bias=False)
        nn.init.zeros_(self.u.weight)
        self.family = family

    def residual(self, x):
        z = self.a(x)
        return self.u(F.gelu(z) if self.family == 'nonlinear_residual' else z)

    def forward(self, x):
        return F.linear(x, self.native_weight) + self.residual(x)


def select_validation_candidate(candidates):
    """Stable tie-breaking by supplied grid order; never consume test metrics."""
    return min(candidates, key=lambda row: row['best_validation_loss'])
