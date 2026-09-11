"""EXP-003C minimal recurrent modification of the pinned public model."""
import torch
from torch import nn
from torch.nn import functional as F
from ddm.vendor.murugan.model import TinyTransformerLM


class TinyRecurrent(TinyTransformerLM):
    def __init__(self, depth=1, output_size=256):
        assert depth in (1, 4) and output_size in (256, 4096)
        # Always construct the exact original 256-head model first. Extra rows
        # must not perturb RNG consumption for backbone or active initialization.
        super().__init__(256, 64, 32, 4, 4, 128, 0., False, input_vocab_size=256)
        self.depth = depth
        self.dummy = None
        if output_size > 256:
            self.dummy = nn.Parameter(torch.empty(output_size-256, 32))
            nn.init.normal_(self.dummy, std=.02)

    def hidden(self, x):
        h = self.token_embedding(x) + self.position_embedding(torch.arange(x.shape[1],device=x.device))[None]
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1],device=x.device)
        for _ in range(self.depth):
            for layer in self.layers:
                h = layer(h, src_mask=mask, is_causal=True)
        return self.final_norm(h)

    def forward(self, x, masked=False):
        h = self.hidden(x)
        active = self.lm_head(h)
        if self.dummy is None:
            return active
        extra = F.linear(h, self.dummy)
        if masked:
            extra = extra*0 + float('-inf')
        return torch.cat([active, extra],dim=-1)


def decomposition(z, y):
    active = F.cross_entropy(z[...,:256].reshape(-1,256),y.flatten(),reduction='none')
    raw = F.cross_entropy(z.reshape(-1,z.shape[-1]),y.flatten(),reduction='none')
    competition = torch.logsumexp(z,dim=-1)-torch.logsumexp(z[...,:256],dim=-1)
    mass = -torch.expm1(-competition)
    return raw, active, competition.flatten(), mass.flatten()
