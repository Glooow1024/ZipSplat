"""Read-only layer tracing on fixed pilot batches; no training or model edits.

Original module forwards are retained. Hooks return None, method wrappers delegate
to the original. Compare complete Gaussian tensors/render with hooks on/off.
Ranks use FP64 centered Gram eigenvalues, with autocast disabled.
"""
import argparse
import gc
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from splatfactory.datasets import get_dataset
from splatfactory.models.networks.zipsplat import ZipSplat
from splatfactory.utils.mappings import batch_to_device
from tools.scene_token.train_initial_test import rng_state, set_rng


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def describe(batch):
    return dict(scene=batch['name'][0], **{
        k: batch[k]['index'].flatten().tolist() for k in ('context', 'target')})


@torch.no_grad()
def metrics(x):
    """x = N,D on CPU; all slots included, no rank subsampling."""
    with torch.autocast('cuda', enabled=False):
        x = x.to(device='cuda', dtype=torch.float64)
        assert torch.isfinite(x).all()
        z = x - x.mean(0, keepdim=True)
        n = x.shape[0]
        unit = F.normalize(x, dim=-1)
        cos = (unit.sum(0).square().sum() - unit.square().sum()) / (n * (n-1))
        eig = torch.linalg.eigvalsh(z @ z.T).clamp_min(0)
        p = eig / eig.sum().clamp_min(1e-30)
        return dict(n=n, dim=x.shape[-1], cosine=float(cos),
                    relative_variation=float(z.norm()/x.norm().clamp_min(1e-30)),
                    centered_rms=float(z.square().mean().sqrt()),
                    rms=float(x.square().mean().sqrt()),
                    effective_rank=float(torch.exp(-(p*p.clamp_min(1e-30).log()).sum())),
                    top1_energy=float(p[-1]), top5_energy=float(p[-5:].sum()))


def grouped_metrics(x, ordered):
    x = x.flatten(0, -3) if x.ndim > 3 else x
    if ordered:
        views = [metrics(v) for v in x]
        result = dict(per_view=views, within_view_mean={
            k: float(np.mean([v[k] for v in views])) for k in views[0]})
        if len(views) > 1:
            u = F.normalize(x.double(), dim=-1)
            result['same_slot_cross_view_cosine'] = float(torch.stack([
                (u[i]*u[j]).sum(-1).mean() for i in range(len(views))
                for j in range(i+1, len(views))]).mean())
        # Full pooled rank for V2; V6 uses exact per-view ranks to avoid a
        # large, redundant 1536x1536 eigensolve at each substep.
        if len(views) <= 2:
            result['pooled'] = metrics(x.flatten(0, 1))
        else:
            y = x.double().flatten(0, 1)
            z = y-y.mean(0)
            result['pooled_relative_variation'] = float(z.norm()/y.norm())
        return result
    return dict(pooled=metrics(x.reshape(-1, x.shape[-1])) )


class Trace:
    def __init__(self, model, views):
        self.model, self.views = model, views
        self.bb = model.backbone.backbone
        self.scene = bool(model.conf.scene_tokens_enabled)
        self.start = self.bb.scene_start_idx if self.scene else self.bb.patch_start_idx
        self.count = self.bb.num_scene_tokens if self.scene else 324
        self.n = self.bb.patch_start_idx + 324
        self.raw, self.attention, self.handles, self.restore = {}, {}, [], []

    def add(self, name, x, ordered=True):
        if isinstance(x, tuple): x = x[0]
        if ordered: x = x.reshape(self.views, -1, x.shape[-1])
        self.raw[name] = (x.detach().float().cpu(), ordered)

    def tokens(self, x, patch=False):
        x = x.reshape(self.views, self.n, -1)
        return x[:, self.bb.patch_start_idx:] if patch else x[:, self.start:self.start+self.count]

    def wrap(self, obj, name, after):
        original = getattr(obj, name)
        def wrapped(*args, **kwargs):
            out = original(*args, **kwargs)
            after(out)
            return out
        setattr(obj, name, wrapped)
        self.restore.append((obj, name, original))

    def output(self, module, name, ordered=True):
        def hook(_m, _a, out): self.add(name, out, ordered)
        self.handles.append(module.register_forward_hook(hook))

    def input(self, module, name, ordered=True):
        def hook(_m, args): self.add(name, args[0], ordered)
        self.handles.append(module.register_forward_pre_hook(hook))

    def probability_stats(self, name, q, k, scale, mask=None, key_groups=None):
        # First view's complete S queries, all heads and all available keys.
        # q/k already have the actual module's QK norm and RoPE applied.
        with torch.autocast('cuda', enabled=False):
            scores = (q.float()*scale) @ k.float().transpose(-2, -1)
            if mask is not None:
                scores = scores.masked_fill(~mask, -float('inf')) if mask.dtype == torch.bool else scores+mask
            p = scores.softmax(-1)
            u = F.normalize(p, dim=-1)
            n = p.shape[-2]
            cosine = ((u.sum(-2).square().sum(-1)-u.square().sum((-2,-1))) / (n*(n-1))).mean()
            mean = p.mean(-2, keepdim=True)
            record = dict(query_distribution_cosine=float(cosine),
                normalized_entropy=float((-(p*p.clamp_min(1e-30).log()).sum(-1)/np.log(p.shape[-1])).mean()),
                relative_query_variation=float((p-mean).norm()/p.norm()),
                key_count=p.shape[-1], query_count=n)
            if key_groups:
                record['key_mass'] = {label:float(p[..., idx].sum(-1).mean()) for label,idx in key_groups.items()}
            self.attention[name] = record

    def da3_attention(self, i):
        def hook(module, args, kwargs):
            x = args[0]
            b, n, d = x.shape
            q, k, _v = module.qkv(x).reshape(b,n,3,module.num_heads,module.head_dim).permute(2,0,3,1,4).unbind(0)
            q, k = module.q_norm(q), module.k_norm(k)
            if module.rope is not None:
                q, k = module.rope(q,kwargs.get('pos')), module.rope(k,kwargs.get('pos'))
            q = q[:1,:,self.start:self.start+self.count]
            k = k[:1]
            mask = kwargs.get('attn_mask')
            if mask is not None: mask = mask[:1,None,self.start:self.start+self.count]
            idx = torch.arange(n,device=x.device) % self.n
            groups = dict(special=idx < self.bb.scene_start_idx,
                          patch=idx >= self.bb.patch_start_idx)
            if self.scene: groups['scene'] = (idx >= self.start)&(idx < self.start+self.count)
            self.probability_stats(f'da3.{i:02d}',q,k,module.scale,mask,groups)
        return hook

    def cross_attention(self, name):
        def hook(module, args):
            queries, context = args[:2]
            b, n, _ = queries.shape
            q = module.q_proj(queries).reshape(b,n,module.num_heads,module.head_dim).permute(0,2,1,3)
            k,_v = module.kv_proj(context).reshape(b,context.shape[1],2,module.num_heads,module.head_dim).permute(2,0,3,1,4).unbind(0)
            q,k = module.q_norm(q),module.k_norm(k)
            assert module.rope is None and module.attn_mode == 'softmax'
            self.probability_stats(name,q[:1,:,:min(self.count,n)],k[:1],module.scale)
        return hook

    def install(self):
        ordered = self.scene
        if self.scene:
            self.add('initial_parameter', self.bb.scene_tokens[None].expand(self.views,-1,-1))
        self.wrap(self.bb,'prepare_tokens_with_masks',lambda out:self.add('embedding',self.tokens(out)))
        for i, block in enumerate(self.bb.blocks):
            prefix = f'da3.{i:02d}'
            def before(_m,args,p=prefix): self.add(p+'.input',self.tokens(args[0]))
            def after_attn(_m,args,p=prefix): self.add(p+'.after_attention',self.tokens(args[0]))
            def after(_m,args,out,p=prefix):
                self.add(p+'.output',self.tokens(out))
                if self.scene: self.add(p+'.patch_output',self.tokens(out,True))
            self.handles.extend([block.register_forward_pre_hook(before),
                block.norm2.register_forward_pre_hook(after_attn),block.register_forward_hook(after),
                block.attn.register_forward_pre_hook(self.da3_attention(i),with_kwargs=True)])
        def features(out):
            for i,x in enumerate(out):
                self.add(f'prepare.{i}.raw_local',x[...,:self.model.embed_dim])
                self.add(f'prepare.{i}.raw_global_norm',x[...,self.model.embed_dim:])
        self.wrap(self.model,'_backbone_features',features)
        def prepared(out):
            for i,x in enumerate(out): self.add(f'prepare.{i}.output',x)
        self.wrap(self.model,'_prepare',prepared)
        for i in range(self.model.num_layers):
            self.output(self.model.pre_norm_local[i],f'prepare.{i}.local_norm')
            self.output(self.model.pre_norm_global[i],f'prepare.{i}.global_norm')
            self.output(self.model.downscale[i],f'prepare.{i}.downscale')
            self.input(self.model.cross_attention[i],f'fusion.{i}.queries',ordered)
            self.output(self.model.cross_attention[i],f'fusion.{i}.ca_delta',ordered)
            self.input(self.model.self_attention[i],f'fusion.{i}.after_ca',ordered)
            self.input(self.model.self_attention[i].norm2,f'fusion.{i}.after_sa_attention',ordered)
            self.output(self.model.self_attention[i],f'fusion.{i}.output',ordered)
            self.handles.append(self.model.cross_attention[i].attn.register_forward_pre_hook(self.cross_attention(f'fusion.{i}.ca')))
        if self.scene: self.output(self.model.scene_color_query,'color.query',ordered)
        self.input(self.model.color_cross_attention,'color.query_input',ordered)
        self.output(self.model.color_embed,'color.patch',True)
        self.input(self.model.color_cross_attention.norm2,'color.after_attention',ordered)
        self.output(self.model.color_cross_attention,'color.output',ordered)
        self.handles.append(self.model.color_cross_attention.attn.register_forward_pre_hook(self.cross_attention('color.ca')))
        def head_input(_m,args): self.add('head.input',args[0]['tokens'],ordered)
        self.handles.append(self.model.gaussian_head.register_forward_pre_hook(head_input))

    def remove(self):
        for handle in self.handles: handle.remove()
        for obj,name,original in self.restore: setattr(obj,name,original)

    def summarize(self):
        results = {}
        for name,(x,ordered) in self.raw.items():
            results[name] = grouped_metrics(x,ordered)
        updates = {}
        for i in range(40):
            prefix=f'da3.{i:02d}'
            for a,b,label in [('input','after_attention','attention'),('after_attention','output','mlp')]:
                x=self.raw[prefix+'.'+a][0].double()
                y=self.raw[prefix+'.'+b][0].double()
                z=x-x.mean(-2,keepdim=True); w=y-y.mean(-2,keepdim=True)
                updates[prefix+'.'+label] = dict(delta_over_input=float((y-x).norm()/x.norm()),
                    centered_output_over_input=float(w.norm()/z.norm().clamp_min(1e-30)),
                    centered_delta_over_input=float((w-z).norm()/z.norm().clamp_min(1e-30)))
        return dict(stages=results, attention=self.attention, updates=updates,
                    alt_start=self.bb.alt_start,rope_start=self.bb.rope_start,
                    scene_start=self.bb.scene_start_idx,patch_start=self.bb.patch_start_idx)


def snapshot(pred):
    tensors={k:getattr(pred['gaussians'],k) for k in ('means','scales','quats','opacities','sh_coeffs')}
    tensors.update(scene_tokens=pred['scene_tokens'],target_rgb=pred['target_rgb'])
    return {k:v.detach().cpu() for k,v in tensors.items()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(256); np.random.seed(256); random.seed(256)
    conf=json.loads((args.run/'config.json').read_text())
    # Replay original data-loading RNG consumption before initializing the model.
    dc=OmegaConf.create(conf['data'])
    train=list(get_dataset(dc.name)(dc,split='train').get_loader(num_workers=0))
    vc=OmegaConf.merge(dc,dc.val_overrides)
    val=list(get_dataset(vc.name)(vc,split='val').get_loader(num_workers=0))
    expected=json.loads((args.run/'batches.json').read_text())
    actual=dict(train=[describe(b) for b in train],validation=[describe(b) for b in val])
    assert actual == expected, 'Initial RNG replay did not reproduce original batches'
    init_rng=rng_state()
    dump(args.output/'protocol.json',dict(run=str(args.run),config=conf,batches=actual,
        selected=['train[0]','validation[0]'],dtype='BF16 forward; FP64 centered rank',
        baseline='original released ZipSplat; K=V*256, geometry KV retains all patches',
        rank='exp(entropy(normalized squared singular values after token mean removal))',
        attention='first view all slot queries, per-head probabilities; FP32 QK score recomputation after actual BF16 projections/QKnorm/RoPE; diagnostics only',
        init='RNG replay of all original batch reads, verify initial loss against stored evaluation; no saved step0 checkpoint',
        scope='Inference only. No model parameter, optimizer, or dataset mutation.'))
    files=['splatfactory/models/encoders/dav3_dino.py','splatfactory/models/networks/zipsplat.py',
           'splatfactory/models/modules/transformer_block.py','splatfactory/models/modules/attention.py',
           'zipsplat/scene_tokens.py','tools/scene_token/diagnose_layers.py']
    dump(args.output/'provenance.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        status=subprocess.check_output(['git','status','--short'],text=True),torch=torch.__version__,
        files={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}))
    for label in ('initial_replayed','step100','step200','baseline'):
        set_rng(init_rng)
        mc=dict(conf['model'])
        mc.update(weights='zipsplat' if label in ('initial_replayed','baseline') else None,
                  scene_token_init_from_base=label=='initial_replayed')
        if label=='baseline': mc.update(scene_tokens_enabled=False,freeze_backbone_except_scene=False,query_sample_ratio=[256/324]*2)
        model=ZipSplat(mc).cuda().eval()
        if label.startswith('step'):
            cp=torch.load(args.run/f'checkpoint_{label[4:]}.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(cp['model'],strict=True); del cp
        torch.set_float32_matmul_precision('high')
        for sample,batch in [('train0',train[0]),('val0',val[0])]:
            print(f'START {label} {sample}',flush=True)
            data=batch_to_device(batch,'cuda')
            state=rng_state()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                reference=model(data)
                losses, quality=model.loss_metrics(reference,data)
            snap=snapshot(reference)
            loss=float(losses['total'].mean()); psnr=float(quality['psnr'].mean())
            del reference,losses,quality
            set_rng(state)
            trace=Trace(model,len(describe(batch)['context']))
            trace.install()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): pred=model(data)
            trace.remove()
            check=snapshot(pred)
            parity={k:float((snap[k]-check[k]).abs().max()) for k in snap}
            assert all(v==0 for v in parity.values()),parity
            result=trace.summarize()
            result.update(label=label,sample=describe(batch),loss=loss,psnr=psnr,hook_parity_max_abs=parity)
            if label=='initial_replayed':
                old=json.loads((args.run/'evaluations.json').read_text())[0]
                expected_loss=old['train' if sample=='train0' else 'validation']['scenes'][0]['total']
                result['initial_replay_loss_check']=dict(recorded=expected_loss,actual=loss,abs_error=abs(loss-expected_loss))
            dump(args.output/f'{label}_{sample}.json',result)
            print(f'DONE {label} {sample} loss={loss:.8f} psnr={psnr:.6f} parity={max(parity.values())}',flush=True)
            del trace,pred,snap,check,data,result
            gc.collect(); torch.cuda.empty_cache()
        del model
        gc.collect(); torch.cuda.empty_cache()
    dump(args.output/'completion.json',dict(completed=True,runs=8,optimizer_updates=0))


if __name__=='__main__': main()
