"""Bounded 8-scene DDP pilot; every global update contains one batch per scene."""
import argparse,gc,hashlib,json,os,random,subprocess,sys,time
from datetime import timedelta
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from splatfactory.models.networks.zipsplat import ZipSplat
from splatfactory.trainer import Trainer
from splatfactory.utils.mappings import batch_to_device
from tools.scene_token.multiscene_data import PilotData
from tools.scene_token.position_1d import ScenePosition1D
from tools.scene_token.audit_repeatability import controlled_runtime
from tools.scene_token.train_initial_test import evaluate,dump,param_hash,rng_state,set_rng,floats
from tools.scene_token.continue_scale_fit import equal_state

TRACKED=['backbone.backbone.scene_tokens','scene_color_query.weight','downscale.0.weight','gaussian_head.gaussian_head.1.weight']


def hash_consensus(model):
    h=param_hash(model,False); hashes=[None]*dist.get_world_size();dist.all_gather_object(hashes,h)
    assert len(set(hashes))==1,hashes
    return h


def gradient_check(model,trainer,batch):
    state=rng_state();local={};handles=[];params=dict(model.named_parameters())
    for name in TRACKED:
        handles.append(params[name].register_hook(lambda g,n=name:local.__setitem__(n,g.detach().clone())))
    data=batch_to_device(batch,trainer.device)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        pred=trainer.model(data);losses,_=model.loss_metrics(pred,data);loss=losses['total'].mean()
    trainer.scaler.scale(loss).backward()
    for h in handles:h.remove()
    errors={}
    for name in TRACKED:
        expected=local[name];dist.all_reduce(expected);expected/=dist.get_world_size()
        actual=params[name].grad
        assert actual is not None and torch.isfinite(actual).all() and actual.norm()>0
        error=float((actual-expected).norm()/expected.norm().clamp_min(1e-20))
        assert error<1e-5,(name,error);errors[name]=error
    trainer.optimizer.zero_grad(set_to_none=True);set_rng(state)
    return errors


def save(path,model,trainer,conf):
    trainer.optimizer.zero_grad(set_to_none=True)
    states=[None]*dist.get_world_size();dist.all_gather_object(states,rng_state())
    if dist.get_rank()==0:
        state=dict(model=model.state_dict(),optimizer=trainer.optimizer.state_dict(),
            lr_scheduler=trainer.lr_scheduler.state_dict(),scaler=trainer.scaler.state_dict(),
            epoch=0,tot_it=trainer.tot_it,tot_n_samples=trainer.tot_n_samples,
            conf=conf,rng=states[0],rng_by_rank=states,schema='scene-token-multiscene-v1')
        tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)
    dist.barrier()


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=5000);p.add_argument('--smoke',action='store_true');p.add_argument('--require-smoke',type=Path)
    a=p.parse_args();rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==8
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(4)
    dist.init_process_group('nccl',timeout=timedelta(minutes=60),device_id=torch.device('cuda',rank))
    start=time.time()
    if rank==0:a.output.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    if not a.smoke:
        assert a.require_smoke is not None
        assert json.loads((a.require_smoke/'completion.json').read_text())['passed']
    plan=json.loads((a.protocol/'plan.json').read_text());old=json.loads((a.protocol/'base_config.json').read_text())
    data=PilotData(old['data'],plan);assert data.tar_hashes==plan['tar_hashes']
    keys=sorted(k for k,e in plan['scenes'].items() if e['split']=='train');key=keys[rank]
    audit=json.loads((a.protocol/'data_audit.json').read_text());probes={k:audit['scenes'][k]['samples'][0] for k in keys}
    probe=data.fixed(key,probes[key]['context'],probes[key]['target'])
    torch.manual_seed(256);np.random.seed(256);random.seed(256)
    mc=dict(old['model']);mc.update(weights='zipsplat',scene_token_init_from_base=True)
    model=ZipSplat(mc).cuda().train();adapter=ScenePosition1D(model,'distinct')
    tc=dict(old['train']);tc.update(num_steps=a.steps,ddp_find_unused_parameters=True,compile=None)
    trainer=Trainer.init(OmegaConf.create(tc),model,device=torch.device('cuda',rank));trainer._view_schedule_frac=0.
    controlled_runtime(model)
    frozen=param_hash(model,True);initial=hash_consensus(model)
    np.random.seed(51000+rank);random.seed(51000+rank);torch.manual_seed(51000+rank)
    gradient_errors=gradient_check(model,trainer,probe)
    dump(a.output/f'gradient_check_rank{rank}.json',dict(passed=True,relative_errors=gradient_errors,optimizer_updates=0))
    conf=dict(model=old['model'],train=tc,data=old['data'],experiment=adapter.specification(),
        multiscene_experiment=dict(world_size=world,global_scene_batch=8,local_scene_batch=1,views=2,targets=4,
            steps=a.steps,scene_exposures=world*a.steps,initial_std=.02,seed=256,rank_scenes=keys,
            plan=str(a.protocol/'plan.json'),fresh_released_initialization=True,
            learning_rate='Same absolute group LRs as previous B1 test; no linear batch scaling',
            limitation='8 scenes per averaged optimizer update; exposure comparable but optimizer trajectory differs from B1 sequential updates; cached decoded CPU views, not production DataLoader throughput'))
    if rank==0:
        dump(a.output/'config.json',conf);dump(a.output/'plan.json',plan);dump(a.output/'train_probes.json',probes)
        source_files=['tools/scene_token/train_multiscene.py','tools/scene_token/multiscene_data.py',
            'tools/scene_token/position_1d.py','tools/scene_token/audit_repeatability.py','tools/scene_token/train_initial_test.py',
            'splatfactory/trainer.py','splatfactory/models/networks/zipsplat.py','splatfactory/models/encoders/dav3_dino.py',
            'splatfactory/models/encoders/dav3_encoder.py','splatfactory/models/decoders/gaussian_head.py']
        for f in source_files:
            dst=a.output/'source'/f;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(Path(f).read_bytes())
        dump(a.output/'provenance.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            status=subprocess.check_output(['git','status','--short'],text=True),torch=torch.__version__,gpu=torch.cuda.get_device_name(),
            initial_trainable_sha256=initial,frozen_sha256=frozen,files={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in source_files}))
    evaluations=[]

    def assess(step):
        dist.barrier()
        if rank==0:
            folder=a.output/f'step{step:04d}';folder.mkdir()
            sets=dict(train_probe=[data.fixed(k,probes[k]['context'],probes[k]['target']) for k in keys],
                trained_scene_v2=data.evaluation('train',2),new_scene_v2=data.evaluation('validation',2),
                new_scene_v6=data.evaluation('validation',6))
            row={'step':step}
            for name,batches in sets.items():
                out=folder/name;out.mkdir();row[name]=evaluate(model,batches,out,'after')
            evaluations.append(row);dump(a.output/'evaluations.json',evaluations)
            print(json.dumps(dict(evaluation=step,metrics={name:{m:row[name]['mean'][m] for m in ['psnr','lpips']} for name in sets})),flush=True)
        dist.barrier()

    assess(0);times=[];local_rows=[];torch.cuda.reset_peak_memory_stats()
    for step in range(1,a.steps+1):
        before=time.perf_counter()
        if step==1:batch=probe;selection=dict(probes[key],rejected=0)
        else:batch,selection=data.training(key)
        trainer.model.train();trainer.tot_it=step-1;trainer.step_timer.reset()
        grads={};handles=[];scale=trainer.scaler.get_scale();params=dict(model.named_parameters())
        for name in TRACKED:
            def capture(g,n=name):
                assert torch.isfinite(g).all(),n
                grads[n]=float(g.float().norm())/scale
            handles.append(params[name].register_hook(capture))
        pred,metrics=trainer.train_step(batch,log_grad_norm=True)
        for h in handles:h.remove()
        assert len(grads)==4 and all(np.isfinite(v) and v>0 for v in grads.values())
        slot=model.backbone.backbone.scene_tokens
        assert int(trainer.optimizer.state[slot]['step'])==step,'Skipped optimizer update'
        vals=floats(metrics);assert all(np.isfinite(v) for v in vals.values())
        trainer.tot_it=step;trainer.tot_n_samples=step*world
        torch.cuda.synchronize();seconds=time.perf_counter()-before;times.append(seconds)
        record=dict(step=step,scene=key,selection=selection,metrics=vals,gradients=grads,seconds=seconds,
            lr=[g['lr'] for g in trainer.optimizer.param_groups],gradient_scale=trainer.scaler.get_scale())
        with (a.output/f'steps_rank{rank}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        reduced=torch.tensor([vals['loss/total'],vals['psnr'],vals['lpips'],seconds],device=trainer.device)
        dist.all_reduce(reduced);reduced/=world
        if rank==0:
            row=dict(step=step,global_scene_exposures=step*world,loss=float(reduced[0]),psnr=float(reduced[1]),lpips=float(reduced[2]),mean_rank_seconds=float(reduced[3]))
            with (a.output/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if step%100==0:print(json.dumps(row),flush=True)
        del pred,metrics,batch
        if step in [500,1000,2500,a.steps]:
            hash_consensus(model);assess(step)
    assert frozen==param_hash(model,True);final_hash=hash_consensus(model)
    checkpoint=a.output/f'checkpoint_{a.steps}.pt';save(checkpoint,model,trainer,conf)
    if a.smoke:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        with torch.no_grad():model.backbone.backbone.scene_tokens.add_(.01)
        trainer.load_checkpoint(saved,strict=True,load_state=True)
        assert param_hash(model,False)==final_hash and param_hash(model,True)==frozen
        assert equal_state(trainer.optimizer.state_dict(),saved['optimizer'])
        assert equal_state(trainer.lr_scheduler.state_dict(),saved['lr_scheduler'])
        assert equal_state(trainer.scaler.state_dict(),saved['scaler'])
        assert trainer.tot_it==a.steps and trainer.tot_n_samples==a.steps*world
        del saved
    summary=dict(rank=rank,scene=key,steps=a.steps,exposures=a.steps,median_step_seconds=float(np.median(times[5:])),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,frozen_unchanged=True,
        final_trainable_sha256=final_hash,wall_seconds=time.time()-start)
    dump(a.output/f'completion_rank{rank}.json',summary)
    all_summaries=[None]*world;dist.all_gather_object(all_summaries,summary)
    if rank==0:
        with checkpoint.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        dump(a.output/'completion.json',dict(passed=True,smoke=a.smoke,optimizer_updates=a.steps,scene_exposures=a.steps*world,
            ddp_gradient_mean_verified=True,rank_parameter_hashes_identical=True,full_state_restore_verified=a.smoke,
            checkpoint_sha256=sha,checkpoint_bytes=checkpoint.stat().st_size,ranks=all_summaries))
        print('COMPLETE '+str(a.output),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
