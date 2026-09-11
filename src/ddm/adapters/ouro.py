from __future__ import annotations
import torch
from .base import DepthHiddenStates, LoopedModelAdapter

class OuroAdapter(LoopedModelAdapter):
    def __init__(self, model, depths=(1,2,3,4)):
        self.model=model; self.depths=tuple(int(d) for d in depths)
        if not self.depths or min(self.depths)<1: raise ValueError("depths must be positive")
    @torch.no_grad()
    def extract_depth_hidden(self,input_ids,attention_mask=None):
        max_depth=max(self.depths); old_steps=int(self.model.model.total_ut_steps); old_cfg=int(getattr(self.model.config,"total_ut_steps",old_steps))
        try:
            self.model.model.total_ut_steps=max_depth; self.model.config.total_ut_steps=max_depth
            _,hidden_list,_=self.model.model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False)
            if len(hidden_list)<max_depth: raise RuntimeError(f"expected >= {max_depth} UT states, got {len(hidden_list)}")
            by_depth={d:hidden_list[d-1] for d in self.depths}
        finally:
            self.model.model.total_ut_steps=old_steps; self.model.config.total_ut_steps=old_cfg
        w=self.model.get_output_embeddings().weight.detach()
        return DepthHiddenStates(by_depth,int(self.model.config.vocab_size),int(self.model.config.hidden_size),w)
