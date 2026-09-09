"""Original ZipSplat budget sweep. No model weights or architecture are changed.

Run from the repository using its own Python environment. Gaussians are fixed;
the only optimized variables are seven shared camera-alignment parameters.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
import lpips
from skimage.metrics import structural_similarity
from zipsplat import Camera, Pose, ZipSplat
from zipsplat.utils import load_image, to_square
from audit_kmeans import sha


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, time.perf_counter() - t


def rotation(vector):
    # matrix_exp is differentiable at zero, avoiding singular Rodrigues branches.
    x, y, z = vector.unbind()
    zero = x * 0
    skew = torch.stack([zero,-z,y,z,zero,-x,-y,x,zero]).reshape(3,3)
    return torch.matrix_exp(skew)


def transform(base, rot, trans, log_scale):
    R = rotation(rot)
    return Pose.from_Rt(R[None] @ base[:, :3, :3],
        log_scale.exp() * (R[None] @ base[:, :3, 3, None]).squeeze(-1) + trans)


def params_pose(base, params):
    return transform(base, *[torch.tensor(params[k], device='cuda', dtype=torch.float32)
                            for k in ['rotation_vector','translation','log_scale']])


def render(g, cameras, poses):
    return g.render(cameras, poses, backgrounds=torch.ones(len(cameras),3,device='cuda'))[0]


def align(g, cameras, base, rgb, steps):
    """Shared similarity fit on context RGB only; transfer unchanged to targets."""
    with torch.no_grad():
        scores = []
        def score(s):
            p = transform(base, torch.zeros(3,device='cuda'), torch.zeros(3,device='cuda'),
                          torch.tensor(math.log(s),device='cuda'))
            return float((render(g,cameras,p)-rgb).square().mean())
        for scale in np.geomspace(.03,30,17):
            scores.append((score(float(scale)),float(scale)))
        best_scale = min(scores)[1]
        for scale in np.geomspace(best_scale/1.6,best_scale*1.6,11):
            scores.append((score(float(scale)),float(scale)))
        initial_loss, initial_scale = min(scores)
    rot = torch.nn.Parameter(torch.zeros(3,device='cuda'))
    trans = torch.nn.Parameter(torch.zeros(3,device='cuda'))
    log_scale = torch.nn.Parameter(torch.tensor(math.log(initial_scale),device='cuda'))
    opt = torch.optim.Adam([{'params':[rot,trans],'lr':.01},{'params':[log_scale],'lr':.02}])
    best_loss = float('inf')
    best = None
    history = []
    # Evaluate the initial state and every update; save parameters BEFORE opt.step.
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = (render(g,cameras,transform(base,rot,trans,log_scale))-rgb).square().mean()
        value = float(loss.detach())
        if not math.isfinite(value):
            raise RuntimeError('Nonfinite camera alignment')
        history.append(value)
        if value < best_loss:
            best_loss = value
            best = {k:v.detach().cpu().tolist() for k,v in
                    [('rotation_vector',rot),('translation',trans),('log_scale',log_scale)]}
        if step < steps:
            loss.backward()
            opt.step()
    best.update(context_mse=best_loss, initial_mse=initial_loss,
                initial_scale=initial_scale, history=history, steps=steps,
                fit_images='context_only', transform='c2w Rnew=Rglobal@R; tnew=scale*Rglobal@t+tglobal')
    return best


@torch.no_grad()
def evaluate(g, cameras, base, params, refs, names, metric, out):
    preds, seconds = timed(lambda:render(g,cameras,params_pose(base,params)))
    assert torch.isfinite(preds).all()
    preds = preds.clamp(0,1)
    perceptual = metric(preds*2-1, refs*2-1).flatten().cpu().tolist()
    pred_np = preds.permute(0,2,3,1).cpu().numpy()
    ref_np = refs.permute(0,2,3,1).cpu().numpy()
    rows=[]
    out.mkdir(parents=True, exist_ok=True)
    for i,(p,r,name) in enumerate(zip(pred_np,ref_np,names)):
        mse=float(np.mean((p-r)**2))
        rows.append({'frame':name,'psnr':-10*math.log10(max(mse,1e-10)),
                     'ssim':float(structural_similarity(r,p,data_range=1,channel_axis=2)),
                     'lpips':perceptual[i]})
        Image.fromarray((p*255).round().astype(np.uint8)).save(out/name)
    return {'mean':{k:float(np.mean([r[k] for r in rows])) for k in ['psnr','ssim','lpips']},
            'per_view':rows,'render_seconds':seconds}


def prepare(scene, v, output):
    ctx=scene['contexts'][str(v)]['names']; names=ctx+scene['targets']
    imgs=[]; Ks=[]; mats=[]
    cpu_preprocessing_seconds=0.
    image_load_seconds=0.
    for name in names:
        f=scene['frames'][name]
        assert sha(f['path']) == f['sha256'], f"Source changed: {f['path']}"
        started=time.perf_counter()
        raw=load_image(Path(f['path']))
        image_load_seconds+=time.perf_counter()-started
        started=time.perf_counter()
        imgs.append(to_square(raw))
        cpu_preprocessing_seconds+=time.perf_counter()-started
        Ks.append(f['K']); mats.append(f['c2w_cv'])
    rgb=torch.stack(imgs).cuda()
    cameras=Camera.from_K(torch.tensor(Ks,device='cuda',dtype=torch.float32),252,252)
    base=torch.tensor(np.linalg.inv(np.asarray(mats[0]))[None] @ np.asarray(mats),device='cuda',dtype=torch.float32)
    for name,im in zip(names,imgs):
        p=output/'reference'/name; p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            Image.fromarray((im.permute(1,2,0).numpy()*255).round().astype(np.uint8)).save(p)
    return rgb, cameras, base, names, {'image_load_all_views':image_load_seconds,
        'cpu_crop_resize_all_views':cpu_preprocessing_seconds}


@torch.no_grad()
def decode(model, prepared, layers, ratio):
    timings={}
    idx,timings['kmeans']=timed(lambda:model._cluster(layers[0],ratio))
    scene,timings['geometry']=timed(lambda:model._fuse(layers,idx))
    color,timings['color']=timed(lambda:model._color(prepared,idx))
    gs,timings['head']=timed(lambda:model.gaussian_head(torch.cat([scene,color],-1)))
    g=gs[0]
    assert g.num_gaussians == idx.shape[1]*32 and torch.isfinite(g.data_).all()
    return g,idx,timings


@torch.no_grad()
def profile(model, prepared, ratio, warmup, repeats):
    for _ in range(warmup):
        g=model(prepared,compression=ratio)
        del g
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before=torch.cuda.memory_allocated()
    values=[]
    for _ in range(repeats):
        g,t=timed(lambda:model(prepared,compression=ratio))
        values.append(t)
        del g
    return {'full_forward_seconds':values,'median_seconds':statistics.median(values),
            'min_seconds':min(values),'max_seconds':max(values),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'allocated_before_bytes':allocated_before,
            'incremental_peak_bytes':torch.cuda.max_memory_allocated()-allocated_before,
            'memory_note':'process retains quality features and metric; incremental peak measures extra forward allocations',
            'warmup':warmup,'repeats':repeats,'includes_preprocessing':False}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--scenes',default='all')
    ap.add_argument('--views',default='4,8,12,16')
    ap.add_argument('--ratios',default='1,.5,.25,.125,.0625,.03125')
    ap.add_argument('--worker',type=int,default=0)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--steps',type=int,default=120)
    ap.add_argument('--warmup',type=int,default=3)
    ap.add_argument('--repeats',type=int,default=10)
    ap.add_argument('--equivalence',action='store_true')
    ap.add_argument('--fixed-k',default='')
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260907); np.random.seed(20260907)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((args.root/'manifest.json').read_text())
    weights=Path('/root/.cache/torch/hub/zipsplat/zipsplat-da3g-252p.tar')
    fingerprint={'manifest_sha256':sha(args.root/'manifest.json'),'weights_sha256':sha(weights),
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'zipsplat').glob('*.py'))},
        'script_sha256':sha(__file__),'steps':args.steps,'dtype':'float32_no_tf32','size':252,
        'metric':'PSNR/SSIM-skimage-default/LPIPS-Alex-v0.1; predictions clamped to [0,1]',
        'protocol':'shared-context-only-Sim3; A=r1-frozen; B=per-budget-fit; targets never optimized',
        'warmup':args.warmup,'repeats':args.repeats}
    signature=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
    wrapper=ZipSplat(weights=str(weights)).cuda().eval()
    model=wrapper.model
    for p in model.parameters(): p.requires_grad_(False)
    metric=lpips.LPIPS(net='alex',version='0.1').cuda().eval()
    for p in metric.parameters(): p.requires_grad_(False)
    write_json(args.root/f'environment_worker{args.worker}.json',dict(fingerprint,
        signature=signature,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        command=sys.argv))
    selected=args.scenes.split(',')
    jobs=[(s,v) for s in manifest['scenes'] for v in map(int,args.views.split(','))
          if args.scenes=='all' or s['scene'][:2] in selected]
    failures=[]
    for scene,v in jobs[args.worker::args.workers]:
        prefix=f"{scene['scene'][:2]}/{v}views"
        out=args.root/scene['scene']/f'{v}views'; out.mkdir(parents=True,exist_ok=True)
        try:
            (rgb,cameras,base,names,input_times),pre_seconds=timed(lambda:prepare(scene,v,out))
            prepared=rgb[:v][None]
            with torch.no_grad():
                features,backbone_seconds=timed(lambda:model._backbone_features(prepared,None,None))
                layers,prepare_seconds=timed(lambda:model._prepare(features))
                del features
            variants=[('r'+format(r,'.8g'),r) for r in sorted(set([1.]+list(map(float,args.ratios.split(',')))),reverse=True)]
            for k in filter(None,args.fixed_k.split(',')):
                k=int(k); assert k<=v*324
                ratio=(k+.25)/(v*324) if k<v*324 else 1.
                variants.append((f'k{k}',ratio))
            baseline_params=None
            for label,ratio in variants:
                dest=out/label; resultpath=dest/'result.json'
                if resultpath.exists():
                    old=json.loads(resultpath.read_text())
                    assert old['signature']==signature, f'Configuration changed at {resultpath}'
                    if label=='r1': baseline_params=old['alignment_B']
                    print('SKIP',prefix,label,flush=True)
                    continue
                print('START',prefix,label,flush=True)
                g,idx,stages=decode(model,prepared,layers,ratio)
                equivalence=None
                if args.equivalence and label in ['r1','r0.125']:
                    with torch.no_grad():
                        direct=model(prepared,compression=ratio)[0]
                        max_error=float((direct.data_-g.data_).abs().max())
                        torch.testing.assert_close(direct.data_,g.data_,rtol=1e-5,atol=1e-6)
                        repeat=model(prepared,compression=ratio)[0]
                        torch.testing.assert_close(direct.data_,repeat.data_,rtol=1e-5,atol=1e-6)
                        equivalence={'cached_vs_direct_max_abs':max_error,
                                     'repeat_max_abs':float((direct.data_-repeat.data_).abs().max())}
                        del direct,repeat
                fit,fit_seconds=timed(lambda:align(g,cameras[:v],base[:v],rgb[:v],args.steps))
                if label=='r1': baseline_params=fit
                assert baseline_params is not None
                quality={}
                for protocol,params in [('A',baseline_params),('B',fit)]:
                    quality[protocol]={}
                    for split,sl in [('context',slice(0,v)),('target',slice(v,None))]:
                        quality[protocol][split]=evaluate(g,cameras[sl],base[sl],params,rgb[sl],names[sl],metric,dest/protocol/split)
                # The repeated full forward includes backbone again; cached stages are separate.
                performance=profile(model,prepared,ratio,args.warmup,args.repeats)
                # Stage measurements are separately warmed; cold quality-pass stages remain available.
                sampled=[]
                for i in range(args.warmup+args.repeats):
                    tg,ti,tt=decode(model,prepared,layers,ratio)
                    if i>=args.warmup: sampled.append(tt)
                    del tg,ti
                performance['cached_stage_medians']={k:statistics.median(t[k] for t in sampled) for k in sampled[0]}
                performance['cached_stage_samples']=sampled
                result={'signature':signature,'scene':scene['scene'],'views':v,'label':label,'ratio':ratio,
                    'K':idx.shape[1],'unique_indices':int(idx.unique().numel()),'gaussians':g.num_gaussians,
                    'quality':quality,'alignment_B':fit,'alignment_A':baseline_params,
                    'timing':dict(stages,input_audit_preparation_and_reference_io=pre_seconds,**input_times,backbone=backbone_seconds,
                                  feature_prepare=prepare_seconds,alignment=fit_seconds),
                    'performance':performance,'equivalence':equivalence}
                write_json(resultpath,result)
                print('DONE',prefix,label,'K',result['K'],'B_target',quality['B']['target']['mean'],flush=True)
                del g,idx
            del layers,rgb,cameras,base,prepared
            gc.collect(); torch.cuda.empty_cache()
        except Exception:
            error={'job':prefix,'traceback':traceback.format_exc()}
            failures.append(error); write_json(out/f'failure_worker{args.worker}.json',error)
            print(error['traceback'],flush=True)
            gc.collect(); torch.cuda.empty_cache()
    write_json(args.root/f'completion_worker{args.worker}.json',{'jobs':len(jobs[args.worker::args.workers]),'failures':failures,'command':sys.argv})
    if failures: raise SystemExit(1)


if __name__=='__main__':
    main()
