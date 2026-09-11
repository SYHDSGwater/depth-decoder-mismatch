import torch
from ddm.heads import LinearRefitHead, ResidualRichHead

def test_rich_head_is_nested_at_initialization():
    torch.manual_seed(0); d,v,r=8,13,5; w=torch.randn(v,d); x=torch.randn(7,d)
    linear=LinearRefitHead(d,v,w); rich=ResidualRichHead(d,v,r,w)
    torch.testing.assert_close(linear(x),rich(x))


def test_rich_family_can_represent_a_nonlinear_function():
    rich = ResidualRichHead(1, 1, 1, torch.zeros(1, 1))
    with torch.no_grad():
        rich.in_proj.weight.fill_(1)
        rich.in_proj.bias.zero_()
        rich.out_proj.weight.fill_(1)
    values = rich(torch.tensor([[-1.0], [0.0], [1.0]]))
    assert abs(float(values[0] + values[2] - 2 * values[1])) > 0.1
