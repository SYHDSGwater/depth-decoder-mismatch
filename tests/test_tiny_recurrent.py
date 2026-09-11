import torch
from ddm.tiny_recurrent import TinyRecurrent, TinyTransformerLM, decomposition


def test_public_baseline_exact_parity():
    torch.manual_seed(1)
    ref=TinyTransformerLM(256,64,32,4,4,128,0.,False,input_vocab_size=256).eval()
    torch.manual_seed(1)
    model=TinyRecurrent().eval()
    assert sum(p.numel() for p in model.parameters())==69312
    for k,v in ref.state_dict().items():
        assert torch.equal(v,model.state_dict()[k])
    x=torch.arange(64).reshape(1,64)
    torch.testing.assert_close(ref(x)[0],model(x),rtol=0,atol=0)


def test_pairing_recurrence_and_masked_objective():
    x=torch.arange(64).reshape(1,64);y=(x+1)%256
    for depth in [1,4]:
        torch.manual_seed(2);a=TinyRecurrent(depth,256).eval()
        torch.manual_seed(2);b=TinyRecurrent(depth,4096).eval()
        for k,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[k])
        assert sum(p.numel() for p in b.parameters())==192192
        calls=[];hook=b.layers[0].register_forward_hook(lambda *args:calls.append(1))
        z=b(x);hook.remove();assert len(calls)==depth
        torch.testing.assert_close(a(x),z[...,:256],rtol=0,atol=0)
        raw,active,comp,mass=decomposition(z,y)
        torch.testing.assert_close(raw,active+comp,rtol=1e-5,atol=1e-6)
        masked=b(x,masked=True)
        loss=torch.nn.functional.cross_entropy(masked.flatten(0,1),y.flatten());loss.backward()
        assert torch.count_nonzero(b.dummy.grad)==0
        torch.testing.assert_close(decomposition(masked,y)[0],decomposition(a(x),y)[0])
