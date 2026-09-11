from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
from torch.autograd import Function


def orthonormal_columns(matrix: torch.Tensor) -> torch.Tensor:
    # QR/SVD support on Apple MPS is still incomplete in current PyTorch.
    work = matrix.cpu() if matrix.device.type == "mps" else matrix
    return torch.linalg.qr(work, mode="reduced").Q.to(matrix.device)


def singular_values(matrix: torch.Tensor) -> torch.Tensor:
    work = matrix.cpu() if matrix.device.type == "mps" else matrix
    return torch.linalg.svdvals(work).to(matrix.device)


@torch.no_grad()
def truncated_svd_feedback(weight: torch.Tensor, rank: int) -> torch.Tensor:
    """Best rank-`rank` approximation used only for the hidden-state gradient.

    Computing the right singular subspace from the `D x D` Gram matrix avoids a
    full SVD of the usually much taller `V x D` head.
    """
    maximum_rank = min(weight.shape)
    if rank <= 0 or rank > maximum_rank:
        raise ValueError(f"rank must be in [1, {maximum_rank}], got {rank}")
    matrix = weight.detach().float()
    if rank == maximum_rank:
        return matrix.to(dtype=weight.dtype).clone()
    work = matrix.cpu() if matrix.device.type == "mps" else matrix
    gram = work.t().matmul(work)
    _, eigenvectors = torch.linalg.eigh(gram)
    top = eigenvectors[:, -rank:]
    approximation = work.matmul(top).matmul(top.t())
    return approximation.to(device=weight.device, dtype=weight.dtype)


class DecoupledLinearFunction(Function):
    """Linear forward/head gradient, with a separately specified input gradient."""

    @staticmethod
    def forward(
        ctx, hidden: torch.Tensor, weight: torch.Tensor, feedback: torch.Tensor, match_scale: bool
    ):
        ctx.save_for_backward(hidden, weight, feedback)
        ctx.match_scale = match_scale
        return hidden.matmul(weight.t())

    @staticmethod
    def backward(ctx, grad_logits: torch.Tensor):
        hidden, weight, feedback = ctx.saved_tensors
        standard = grad_logits.matmul(weight)
        alternative = grad_logits.matmul(feedback)
        if ctx.match_scale:
            standard_rms = standard.float().square().mean().sqrt()
            alternative_rms = alternative.float().square().mean().sqrt().clamp_min(1e-12)
            alternative = alternative * (standard_rms / alternative_rms).to(alternative.dtype)
        flat_g = grad_logits.reshape(-1, grad_logits.shape[-1])
        flat_h = hidden.reshape(-1, hidden.shape[-1])
        grad_weight = flat_g.t().matmul(flat_h)
        return alternative, grad_weight, None, None


class ProjectedHeadGradientLinearFunction(Function):
    """Exact forward/hidden gradient, with the head update restricted to `col(W)`."""

    @staticmethod
    def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor, match_scale: bool):
        ctx.save_for_backward(hidden, weight)
        ctx.match_scale = match_scale
        return hidden.matmul(weight.t())

    @staticmethod
    def backward(ctx, grad_logits: torch.Tensor):
        hidden, weight = ctx.saved_tensors
        grad_hidden = grad_logits.matmul(weight)
        flat_g = grad_logits.reshape(-1, grad_logits.shape[-1])
        flat_h = hidden.reshape(-1, hidden.shape[-1])
        work = weight.detach().float()
        if work.device.type == "mps":
            basis = torch.linalg.qr(work.cpu(), mode="reduced").Q.to(work.device)
        else:
            basis = torch.linalg.qr(work, mode="reduced").Q
        projected_g = flat_g.float().matmul(basis).matmul(basis.t()).to(flat_g.dtype)
        full_grad_weight = flat_g.t().matmul(flat_h)
        grad_weight = projected_g.t().matmul(flat_h)
        if ctx.match_scale:
            full_rms = full_grad_weight.float().square().mean().sqrt()
            projected_rms = grad_weight.float().square().mean().sqrt().clamp_min(1e-12)
            grad_weight = grad_weight * (full_rms / projected_rms).to(grad_weight.dtype)
        return grad_hidden, grad_weight, None


class DecoupledLinear(nn.Module):
    def __init__(self, weight: nn.Parameter):
        super().__init__()
        self.weight = weight

    def forward(
        self, hidden: torch.Tensor, feedback: torch.Tensor, match_scale: bool = True
    ) -> torch.Tensor:
        return DecoupledLinearFunction.apply(hidden, self.weight, feedback, match_scale)


@dataclass
class TrackerUpdate:
    drift: float
    captured_energy: float


class OjaSubspace:
    """Block Oja/GHA tracker for a vocabulary-space gradient subspace."""

    def __init__(
        self,
        dimension: int,
        rank: int,
        lr: float,
        orthogonalize_every: int,
        seed: int,
        device: torch.device,
    ):
        if rank > dimension:
            raise ValueError("Feedback rank cannot exceed vocabulary size")
        generator = torch.Generator(device=device).manual_seed(seed)
        initial = torch.randn(dimension, rank, generator=generator, device=device)
        self.basis = orthonormal_columns(initial)
        self.lr = lr
        self.orthogonalize_every = orthogonalize_every
        self.steps = 0

    @torch.no_grad()
    def update(self, samples: torch.Tensor) -> TrackerUpdate:
        samples = samples.detach().reshape(-1, samples.shape[-1]).float()
        # Mean-reduced cross entropy gives the entire batch an arbitrary global
        # scale. Remove it so tracker_lr controls subspace adaptation consistently.
        mean_row_energy = samples.square().sum(dim=1).mean().clamp_min(1e-12)
        samples = samples / mean_row_energy.sqrt()
        old = orthonormal_columns(self.basis)
        scores = samples.matmul(self.basis)
        covariance_action = samples.t().matmul(scores) / max(samples.shape[0], 1)
        gram = scores.t().matmul(scores) / max(samples.shape[0], 1)
        self.basis.add_(self.lr * (covariance_action - self.basis.matmul(gram)))
        self.steps += 1
        if self.steps % self.orthogonalize_every == 0:
            self.basis = orthonormal_columns(self.basis)
        overlap = singular_values(old.t().matmul(orthonormal_columns(self.basis)))
        drift = (1.0 - overlap.square().mean()).clamp_min(0).item()
        total = samples.square().sum().clamp_min(1e-12)
        captured = samples.matmul(orthonormal_columns(self.basis)).square().sum() / total
        return TrackerUpdate(drift=drift, captured_energy=captured.item())


class FrequentDirections:
    """Deterministic streaming sketch; intended for small-V diagnostic experiments."""

    def __init__(self, dimension: int, rank: int, device: torch.device):
        self.rank = rank
        self.sketch_size = min(2 * rank, dimension)
        self.sketch = torch.zeros(self.sketch_size, dimension, device=device)
        self.basis = orthonormal_columns(torch.randn(dimension, rank, device=device))

    @torch.no_grad()
    def update(self, samples: torch.Tensor) -> TrackerUpdate:
        old = self.basis.clone()
        rows = torch.cat(
            [self.sketch, samples.detach().reshape(-1, samples.shape[-1]).float()], dim=0
        )
        work = rows.cpu() if rows.device.type == "mps" else rows
        _, singular, vh = torch.linalg.svd(work, full_matrices=False)
        singular = singular.to(rows.device)
        vh = vh.to(rows.device)
        keep = min(self.sketch_size, singular.numel())
        delta = singular[keep - 1].square()
        shrunk = (singular[:keep].square() - delta).clamp_min(0).sqrt()
        self.sketch.zero_()
        self.sketch[:keep] = shrunk[:, None] * vh[:keep]
        self.basis = vh[: self.rank].t().contiguous()
        overlap = singular_values(old.t().matmul(self.basis))
        drift = (1.0 - overlap.square().mean()).clamp_min(0).item()
        flat = samples.reshape(-1, samples.shape[-1]).float()
        captured = flat.matmul(self.basis).square().sum() / flat.square().sum().clamp_min(1e-12)
        return TrackerUpdate(drift=drift, captured_energy=captured.item())


class FeedbackBank:
    """Build one or more V x D feedback operators from a joint vocabulary basis."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        rank: int,
        channels: int,
        adaptive: bool,
        tracker: str,
        tracker_lr: float,
        orthogonalize_every: int,
        seed: int,
        device: torch.device,
    ):
        self.rank = min(rank, d_model)
        self.channels = channels
        total_rank = self.rank * channels
        if total_rank > vocab_size:
            raise ValueError("channels * rank must not exceed vocabulary size")
        generator = torch.Generator(device=device).manual_seed(seed)
        joint = orthonormal_columns(
            torch.randn(vocab_size, total_rank, generator=generator, device=device)
        )
        self.adaptive = adaptive
        self.tracker = None
        if adaptive:
            if tracker == "oja":
                self.tracker = OjaSubspace(
                    vocab_size, total_rank, tracker_lr, orthogonalize_every, seed, device
                )
            elif tracker == "frequent_directions":
                self.tracker = FrequentDirections(vocab_size, total_rank, device)
            else:
                raise ValueError(f"Unknown tracker {tracker!r}")
            self.tracker.basis = joint
        self.fixed_basis = joint
        maps = []
        for _ in range(channels):
            maps.append(
                orthonormal_columns(
                    torch.randn(d_model, self.rank, generator=generator, device=device)
                )
            )
        self.hidden_maps = maps

    @property
    def basis(self) -> torch.Tensor:
        return self.tracker.basis if self.tracker is not None else self.fixed_basis

    def matrices(self) -> List[torch.Tensor]:
        matrices = []
        for index in range(self.channels):
            start = index * self.rank
            vocab_basis = self.basis[:, start : start + self.rank]
            matrices.append(vocab_basis.matmul(self.hidden_maps[index].t()))
        return matrices

    def update(self, samples: torch.Tensor) -> Optional[TrackerUpdate]:
        return self.tracker.update(samples) if self.tracker is not None else None


def scale_like(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    ref_rms = reference.float().square().mean().sqrt()
    candidate_rms = candidate.float().square().mean().sqrt().clamp_min(1e-12)
    return candidate * (ref_rms / candidate_rms).to(candidate.dtype)


def zero_value_surrogate(
    hidden: torch.Tensor, gradient: torch.Tensor, scale: float
) -> torch.Tensor:
    value = scale * (hidden * gradient.detach()).sum()
    return value - value.detach()
