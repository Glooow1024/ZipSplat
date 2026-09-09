"""Analytic transform checks plus a real SDPA backward/checkpoint check."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from torch.utils.checkpoint import checkpoint
from tools.scene_token.position_1d import ScenePosition1D
from splatfactory.models.modules.transformer_block import SelfAttentionBlock


def main():
    torch.manual_seed(256)
    adapter=object.__new__(ScenePosition1D)
    adapter.mode='distinct';adapter.scale=1.;adapter.base=10000.
    adapter.start=1;adapter.count=4;adapter.tokens_per_view=8;adapter.cache={}
    # Identical vectors across slots must get distinct Q/K; all other tokens exact.
    x=torch.randn(1,2,1,8).expand(1,2,16,8).clone().requires_grad_()
    y=adapter.transform(x)
    mask=(torch.arange(16)%8>=1)&(torch.arange(16)%8<5)
    assert torch.equal(x[...,~mask,:],y[...,~mask,:])
    torch.testing.assert_close(x.norm(dim=-1),y.norm(dim=-1),rtol=1e-6,atol=1e-6)
    assert not torch.equal(y[:,:,1],y[:,:,2])
    assert torch.equal(y[:,:,:8],y[:,:,8:])
    y.sum().backward();assert torch.isfinite(x.grad).all()
    # Rotation applied to the same slot Q and K preserves its self dot product.
    a=torch.randn(1,2,16,8);b=torch.randn_like(a)
    torch.testing.assert_close((a*b).sum(-1),(adapter.transform(a)*adapter.transform(b)).sum(-1),rtol=1e-5,atol=2e-6)
    block=SelfAttentionBlock(16,2).eval()
    handles=[block.attn.q_norm.register_forward_hook(adapter.hook),block.attn.k_norm.register_forward_hook(adapter.hook)]
    z=torch.randn(1,16,16,requires_grad=True)
    out=block(z);out.square().sum().backward();expected=z.grad.clone();z.grad=None
    ck=checkpoint(block,z,use_reentrant=False);ck.square().sum().backward()
    torch.testing.assert_close(out,ck,rtol=0,atol=0)
    torch.testing.assert_close(expected,z.grad,rtol=0,atol=0)
    for h in handles:h.remove()
    print(json.dumps(dict(passed=True,checks=['non_scene_exact','norm_preserved','distinct_slots','shared_across_views','finite_gradient','self_dot_preserved','checkpoint_output_gradient_parity'])))


if __name__=='__main__':main()
