import torch
from ddm.heads import LinearRefitHead, ResidualRichHead

def test_rich_head_is_nested_at_initialization():
    torch.manual_seed(0); d,v,r=8,13,5; w=torch.randn(v,d); x=torch.randn(7,d)
    linear=LinearRefitHead(d,v,w); rich=ResidualRichHead(d,v,r,w)
    torch.testing.assert_close(linear(x),rich(x))
