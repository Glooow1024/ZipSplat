"""Bounded P3 test using the real Trainer optimizer/step and fixed pilot batches.

100 single-batch + 100 eight-scene updates; held-out val; disk resume equivalence.
No DDP, benchmark pose refinement, or claim of generalization. Output must be new.
"""
import argparse
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from splatfactory.datasets import get_dataset
from splatfactory.models.networks.zipsplat import ZipSplat
from splatfactory.trainer import Trainer
from splatfactory.utils.mappings import batch_to_device


def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def set_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def param_hash(model, frozen):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.requires_grad != frozen:
            h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build(model_conf, train_conf, initial):
    conf = dict(model_conf)
    conf['weights'] = 'zipsplat' if initial else None
    conf['scene_token_init_from_base'] = initial
    model = ZipSplat(conf).cuda().train()
    trainer = Trainer.init(OmegaConf.create(train_conf), model, device=torch.device('cuda'))
    trainer._view_schedule_frac = 0.
    return model, trainer


def floats(metrics):
    return {k: float(v.detach().float().mean()) for k,v in metrics.items()}


def evaluate(model, batches, output, label):
    state = rng_state()
    model.eval()
    records = []
    with torch.no_grad():
        for i, batch in enumerate(batches):
            data = batch_to_device(batch, 'cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(data)
                losses, metrics = model.loss_metrics(pred, data)
            row = dict(scene=batch['name'][0], **floats(losses), **floats(metrics))
            assert all(np.isfinite(v) for k,v in row.items() if k!='scene')
            records.append(row)
            if label in ['before', 'after']:
                for j in range(data['target']['image'].shape[1]):
                    for tag, tensor in [('gt',data['target']['image']), ('render', pred['target_rgb'])]:
                        image = (tensor[0,j].detach().float().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
                        Image.fromarray(image).save(output/f'{label}_{i}_{j}_{tag}.png')
    model.train()
    set_rng(state)
    return {'scenes': records, 'mean': {k: float(np.mean([r[k] for r in records])) for k in records[0] if k!='scene'}}


def update(trainer, batch, step):
    trainer.model.train()
    trainer.tot_it = step-1
    grad_info = {}
    grad_scale = trainer.scaler.get_scale()
    tracked = ['backbone.backbone.scene_tokens', 'scene_color_query.weight',
               'downscale.0.weight', 'gaussian_head.gaussian_head.1.weight']
    handles = []
    def hook(name):
        def capture(grad):
            assert torch.isfinite(grad).all(), name
            grad_info[name] = float(grad.float().norm()) / grad_scale
        return capture
    for name in tracked:
        handles.append(dict(trainer.model.named_parameters())[name].register_hook(hook(name)))
    torch.cuda.synchronize(); start = time.perf_counter()
    trainer.step_timer.reset()
    pred, metrics = trainer.train_step(batch, log_grad_norm=True)
    torch.cuda.synchronize(); elapsed = time.perf_counter()-start
    for handle in handles: handle.remove()
    assert all(np.isfinite(v) and v>0 for v in grad_info.values()) and len(grad_info)==4
    slot = trainer.model.backbone.backbone.scene_tokens
    assert int(trainer.optimizer.state[slot]['step']) == step, 'Optimizer skipped an update'
    values = floats(metrics)
    assert all(np.isfinite(v) for v in values.values())
    trainer.tot_it = step
    trainer.tot_n_samples = step
    del pred
    return dict(step=step, seconds=elapsed, scene=batch['name'][0], metrics=values,
                gradients=grad_info, gradient_scale=grad_scale, lr=[g['lr'] for g in trainer.optimizer.param_groups])


def save_checkpoint(path, model, trainer, conf):
    trainer.optimizer.zero_grad(set_to_none=True)
    state = dict(model=model.state_dict(), optimizer=trainer.optimizer.state_dict(),
                 lr_scheduler=trainer.lr_scheduler.state_dict(), epoch=0,
                 scaler=trainer.scaler.state_dict(),
                 tot_it=trainer.tot_it, tot_n_samples=trainer.tot_n_samples,
                 conf=conf, rng=rng_state(), schema='scene-token-initial-test-v1')
    tmp = path.with_suffix('.tmp')
    torch.save(state,tmp); tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(256); np.random.seed(256); random.seed(256)
    data_conf = OmegaConf.load('splatfactory/configs/data/dl3dv_scene_token_pilot.yaml')
    data_conf.image_num_range=[2,2]; data_conf.train_batch_size=2
    dataset = get_dataset(data_conf.name)(data_conf,split='train')
    batches = list(dataset.get_loader(num_workers=0))
    assert len(batches)==8 and len({b['name'][0] for b in batches})==8
    val_conf = OmegaConf.merge(data_conf,data_conf.val_overrides)
    val = list(get_dataset(val_conf.name)(val_conf,split='val').get_loader(num_workers=0))
    assert len(val)==2
    for b in val:
        assert not set(b['context']['index'].flatten().tolist()) & set(b['target']['index'].flatten().tolist())
    def describe(b):
        return dict(scene=b['name'][0],context=b['context']['index'].flatten().tolist(),target=b['target']['index'].flatten().tolist())
    dump(args.output/'batches.json',{'train':[describe(b) for b in batches], 'validation':[describe(b) for b in val]})
    model_conf = dict(scene_tokens_enabled=True,num_scene_tokens=256,
                      backbone=dict(weights=None,use_checkpoint=True),freeze_backbone_except_scene=True,
                      query_sample_ratio=[1.,1.],query_ratio_schedule=0.,query_scale_with_views=0.,
                      train_prior_probability=0.,eval_use_priors=False,use_checkpoint=True,
                      return_attention=False,skip_head_render=False,compile=False)
    train_conf = dict(num_steps=200,epochs=None,mixed_precision='bfloat16',matmul_precision='high',
                      optimizer='AdamW',optimizer_options=dict(weight_decay=.05,betas=[.9,.95]),
                      lr=3e-5,lr_scaling={'scene_tokens':10.,'scene_color_query':10.},clip_grad=1.,
                      lr_schedule=dict(type='LinearLR',on_epoch=False,options=dict(start_factor=.1,total_iters=10)),
                      print_arch=None,run_benchmarks=[],debug_nan=True,writer=None)
    conf = dict(model=model_conf,train=train_conf,data=OmegaConf.to_container(data_conf,resolve=True),
                scope='100 fixed-single-batch then 100 fixed-eight-scene updates; cached CPU batches; no DDP',
                validation_protocol='GT relative cameras + context-depth scale, no pose refinement; 6 context/8 disjoint targets')
    dump(args.output/'config.json',conf)
    model,trainer=build(model_conf,train_conf,True)
    frozen_before = param_hash(model,True)
    trainable_before=param_hash(model,False)
    groups=[]
    named={id(p):n for n,p in model.named_parameters()}
    for g in trainer.optimizer.param_groups:
        names=[named[id(p)] for p in g['params']]
        groups.append(dict(initial_lr=g['initial_lr'],numel=sum(p.numel() for p in g['params']),names=names))
    dump(args.output/'optimizer_groups.json',groups)
    assert sum(g['numel'] for g in groups if g['initial_lr']>1e-4)==589952
    torch.cuda.reset_peak_memory_stats()
    evaluations=[]
    def assess(step):
        tag='before' if step==0 else 'after' if step==200 else f'step{step}'
        tr=evaluate(model,batches,args.output,tag+'_train')
        va=evaluate(model,val,args.output,tag)
        row=dict(step=step,train=tr,validation=va)
        evaluations.append(row); dump(args.output/'evaluations.json',evaluations)
        print(json.dumps({'evaluation_step':step,'train_loss':tr['mean']['total'],
                          'fixed_loss':tr['scenes'][0]['total'],'val_loss':va['mean']['total'],
                          'val_psnr':va['mean']['psnr']}),flush=True)
    assess(0)
    records=[]; resume=None; resume_peak=0.
    for step in range(1,201):
        batch=batches[0] if step<=100 else batches[(step-101)%8]
        row=update(trainer,batch,step)
        records.append(row)
        with (args.output/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if step%10==0:
            print(json.dumps({'step':step,'loss':row['metrics']['loss/total'],'seconds':row['seconds']}),flush=True)
        if step in [50,100,150,200]:assess(step)
        if step==100:
            save_checkpoint(args.output/'checkpoint_100.pt',model,trainer,conf)
            # Compare one uninterrupted update with a newly constructed model and
            # optimizer loaded from disk, using identical batch and RNG state.
            golden=update(trainer,batches[0],101)
            expected_scaler=trainer.scaler.state_dict()
            expected={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
            del model,trainer
            gc.collect();torch.cuda.empty_cache()
            model,trainer=build(model_conf,train_conf,False)
            checkpoint=torch.load(args.output/'checkpoint_100.pt',map_location='cpu',weights_only=False)
            trainer.load_checkpoint(checkpoint,strict=True,load_state=True)
            set_rng(checkpoint['rng']);del checkpoint
            restored=update(trainer,batches[0],101)
            max_abs=0.
            for n,p in model.named_parameters():
                if p.requires_grad:
                    diff=float((p.detach().cpu()-expected[n]).abs().max());max_abs=max(max_abs,diff)
            loss_diff=abs(restored['metrics']['loss/total']-golden['metrics']['loss/total'])
            resume=dict(step=101,loss_abs_difference=loss_diff,trainable_parameter_max_abs_difference=max_abs,
                        scaler_state_equal=trainer.scaler.state_dict()==expected_scaler,
                        passed=loss_diff<1e-5 and max_abs<1e-5 and trainer.scaler.state_dict()==expected_scaler)
            dump(args.output/'resume_check.json',resume);assert resume['passed'],resume
            del expected
            # Return to saved step 100 so the main loop has exactly 200 updates.
            checkpoint=torch.load(args.output/'checkpoint_100.pt',map_location='cpu',weights_only=False)
            trainer.load_checkpoint(checkpoint,strict=True,load_state=True);set_rng(checkpoint['rng']);del checkpoint
            resume_peak=torch.cuda.max_memory_allocated()/2**30
            torch.cuda.reset_peak_memory_stats()
    frozen_after=param_hash(model,True)
    assert frozen_before==frozen_after, 'Frozen backbone or loss weights changed'
    assert trainable_before!=param_hash(model,False)
    save_checkpoint(args.output/'checkpoint_200.pt',model,trainer,conf)
    result=dict(completed_steps=200,diagnostic_extra_updates=2,checkpoint_resume=resume,
                frozen_parameters_unchanged=True,optimizer_skipped_updates=0,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_including_resume_gib=resume_peak,
                median_update_seconds=float(np.median([r['seconds'] for r in records[10:]])),
                mean_update_seconds=float(np.mean([r['seconds'] for r in records[10:]])),
                initial=evaluations[0],after_single_batch=evaluations[2],final=evaluations[-1],
                checkpoint_bytes={p.name:p.stat().st_size for p in args.output.glob('checkpoint_*.pt')},
                note='Training diagnostics only; no final test, camera alignment or DDP; timings exclude data loading/eval/checkpoint.')
    dump(args.output/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['initial','after_single_batch','final']},indent=2),flush=True)


if __name__=='__main__':main()
