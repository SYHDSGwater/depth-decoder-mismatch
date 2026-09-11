from __future__ import annotations
import torch
from .base import DepthHiddenStates, LoopedModelAdapter

class NanbeigeAdapter(LoopedModelAdapter):
    def __init__(self,model,depths=(1,2)):
        self.model=model; self.depths=tuple(int(d) for d in depths)
    @torch.no_grad()
    def extract_depth_hidden(self,input_ids,attention_mask=None):
        old_loops=int(getattr(self.model.config,"num_loops",1)); by_depth={}
        try:
            for d in self.depths:
                self.model.config.num_loops=d; self.model.model.config.num_loops=d
                out=self.model.model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False,return_dict=True)
                by_depth[d]=out.last_hidden_state
        finally:
            self.model.config.num_loops=old_loops; self.model.model.config.num_loops=old_loops
        w=self.model.get_output_embeddings().weight.detach()
        return DepthHiddenStates(by_depth,int(self.model.config.vocab_size),int(self.model.config.hidden_size),w)
