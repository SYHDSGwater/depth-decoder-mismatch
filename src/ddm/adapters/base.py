from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import torch

@dataclass
class DepthHiddenStates:
    by_depth: Mapping[int, torch.Tensor]
    vocab_size: int
    hidden_size: int
    native_lm_head_weight: torch.Tensor

class LoopedModelAdapter:
    def extract_depth_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> DepthHiddenStates:
        raise NotImplementedError
