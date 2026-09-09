"""Experimental 1D scene-slot RoPE, after QK norm, at all DA3 blocks.

Hooks are active in forward and checkpoint recomputation. Nothing is added to V
or the residual stream. A persistent specification buffer makes these checkpoints
fail strict loading in a model that has not installed this experimental adapter.
This is an ablation adapter, not the public inference API.
"""
import math
import torch


class ScenePosition1D:
    MODES = {'none': 0, 'shared': 1, 'distinct': 2}

    def __init__(self, model, mode, scale=1.0, base=10000.0):
        if mode not in self.MODES or not math.isfinite(scale) or scale < 0 or base <= 1:
            raise ValueError('Invalid 1D position specification')
        self.bb = model.backbone.backbone
        assert self.bb.num_scene_tokens > 0
        assert not hasattr(self.bb, 'scene_position_experiment_spec')
        self.mode, self.scale, self.base = mode, scale, base
        self.start, self.count = self.bb.scene_start_idx, self.bb.num_scene_tokens
        self.tokens_per_view = self.bb.patch_start_idx + 324
        self.bb.register_buffer('scene_position_experiment_spec',
            torch.tensor([1., self.MODES[mode], scale, base],dtype=torch.float64,
                         device=self.bb.scene_tokens.device),persistent=True)
        self.handles = []
        self.cache = {}
        if mode != 'none' and scale != 0:
            for block in self.bb.blocks:
                self.handles.append(block.attn.q_norm.register_forward_hook(self.hook))
                self.handles.append(block.attn.k_norm.register_forward_hook(self.hook))

    def transform(self, x):
        # B,H,N,head_dim; each local sequence has one view and a global
        # sequence concatenates complete views. Experiments are fixed at 252px.
        n, dim = x.shape[-2:]
        assert n % self.tokens_per_view == 0 and dim % 2 == 0
        key = (n,dim,x.device)
        if key not in self.cache:
            offset = torch.arange(n,device=x.device) % self.tokens_per_view
            idx = torch.where((offset>=self.start)&(offset<self.start+self.count))[0]
            pos = (offset[idx]-self.start).float()
            if self.mode == 'shared': pos = torch.full_like(pos,(self.count-1)/2)
            freq = self.base ** (-torch.arange(0,dim,2,device=x.device).float()/dim)
            angle = self.scale * pos[:,None] * freq[None,:]
            self.cache[key] = idx,angle.cos(),angle.sin()
        idx,cos,sin = self.cache[key]
        slots = x[...,idx,:].float()
        a,b = slots.chunk(2,dim=-1)
        rotated = torch.cat([a*cos-b*sin, a*sin+b*cos],dim=-1).to(x.dtype)
        out = x.clone()
        out[...,idx,:] = rotated
        return out

    def hook(self, _module, _args, out):
        return self.transform(out)

    def remove(self):
        for handle in self.handles: handle.remove()
        self.handles.clear()
        delattr(self.bb,'scene_position_experiment_spec')

    def specification(self):
        return dict(version=1,mode=self.mode,scale=self.scale,base=self.base,
                    injection='after QK norm, before existing 2D RoPE; every DA3 block',
                    coordinate='slot index 0..255, repeated across views; no 2D anchors/time encoding',
                    checkpoint_requires='Install ScenePosition1D with this specification before strict loading',
                    image_size=252)
