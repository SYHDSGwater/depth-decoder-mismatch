import torch
from ddm.residual_audit import FrozenResidualHead, FAMILIES, select_validation_candidate


def test_matched_initialization_and_frozen_native_after_training():
    torch.manual_seed(2)
    w = torch.randn(13, 8)
    x = torch.randn(16, 8)
    y = torch.arange(16) % 13
    heads = []
    for family in FAMILIES:
        torch.manual_seed(11)
        head = FrozenResidualHead(w, 5, family)
        assert sum(p.numel() for p in head.parameters()) == 110
        torch.testing.assert_close(head(x), x @ w.T, rtol=0, atol=0)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0)
        for step in (1, 2):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(head(x), y).backward()
            assert head.native_weight.grad is None
            assert (head.a.weight.grad.norm().item() == 0) == (step == 1)
            opt.step()
        assert torch.equal(head.native_weight, w)
        heads.append(head)
    assert not heads[0].native_weight.requires_grad


def test_initial_a_bias_are_paired_and_gelu_is_only_functional_change():
    w = torch.zeros(1, 1)
    heads = []
    for family in FAMILIES:
        torch.manual_seed(4)
        head = FrozenResidualHead(w, 1, family)
        heads.append(head)
    assert torch.equal(heads[0].a.weight, heads[1].a.weight)
    assert torch.equal(heads[0].a.bias, heads[1].a.bias)
    with torch.no_grad():
        for head in heads:
            head.a.weight.fill_(1); head.a.bias.zero_(); head.u.weight.fill_(1)
        x = torch.tensor([[-1.], [0.], [1.]])
        linear, nonlinear = [h(x).flatten() for h in heads]
        assert abs((linear[0]+linear[2]-2*linear[1]).item()) < 1e-7
        assert abs((nonlinear[0]+nonlinear[2]-2*nonlinear[1]).item()) > .1


def test_validation_selection_ignores_test_and_retains_step_zero():
    zero = dict(selected_checkpoint_step=0, best_validation_loss=1., test_loss=100.)
    later = dict(selected_checkpoint_step=2000, best_validation_loss=2., test_loss=0.)
    assert select_validation_candidate([zero, later]) is zero
