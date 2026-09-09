"""Resumable staged scene-token training with atomic snapshots and live reports."""
import argparse, gc, hashlib, json, math, os, random, shutil, signal, sys, time, traceback
from datetime import timedelta
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from PIL import Image
from splatfactory.models.networks.zipsplat import ZipSplat
from splatfactory.trainer import Trainer
from splatfactory.utils.mappings import batch_to_device
from tools.scene_token.training_run_data import TrainingData
from tools.scene_token.training_monitor import atomic,publish
from tools.scene_token.position_1d import ScenePosition1D
from tools.scene_token.audit_repeatability import controlled_runtime
from tools.scene_token.train_initial_test import rng_state,set_rng,floats

def optimizer_names(model,optimizer):
    names={id(p):n for n,p in model.named_parameters()}
    return [[names[id(p)] for p in g['params']] for g in optimizer.param_groups]

def migrate_optimizer(model,optimizer,saved):
    """Preserve moments of existing parameters while adding newly unfrozen blocks."""
    old={name:saved['optimizer']['state'][pid] for names,group in zip(saved['optimizer_names'],saved['optimizer']['param_groups'])
         for name,pid in zip(names,group['params']) if pid in saved['optimizer']['state']}
    current=optimizer.state_dict();matched=0
    for names,group in zip(optimizer_names(model,optimizer),current['param_groups']):
        for name,pid in zip(names,group['params']):
            if name in old:current['state'][pid]=old[name];matched+=1
    assert matched==len(old),'Lost optimizer history on stage transition'
    optimizer.load_state_dict(current)
    return matched

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--phase',choices=['warmup','joint'],required=True)
    loading=ap.add_mutually_exclusive_group();loading.add_argument('--resume',type=Path);loading.add_argument('--initialize-from',type=Path)
    a=ap.parse_args();root=a.run
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);torch.set_num_threads(4)
    dist.init_process_group('nccl',timeout=timedelta(minutes=20),device_id=torch.device('cuda',local))
    conf=json.loads((root/'config.json').read_text());phase=conf['stages'][a.phase];end=phase['end_step']
    assert world==conf['world_size']==8 and conf['global_scene_batch']==world
    if rank==0:publish(root,dict(state='loading',phase=a.phase,step=0,total_steps=conf['total_steps'],updated=time.time(),message='加载模型与训练数据'))
    saved=torch.load(a.resume or a.initialize_from,map_location='cpu',weights_only=False) if (a.resume or a.initialize_from) else None
    torch.manual_seed(conf['seed']);np.random.seed(conf['seed']);random.seed(conf['seed'])
    mc=dict(conf['model'],weights=None if saved else 'zipsplat',scene_token_init_from_base=not bool(saved))
    model=ZipSplat(mc).cuda().train();adapter=ScenePosition1D(model,'distinct')
    assert len(model.backbone.backbone.blocks)==40
    assert 'backbone.backbone.blocks.39.attn.qkv.weight' in dict(model.named_parameters())
    if a.phase=='joint':
        for block in model.backbone.backbone.blocks[phase['unfreeze_from']:]:block.requires_grad_(True)
    if saved:model.load_state_dict(saved['model'],strict=True)
    elif rank==0:
        slots=model.backbone.backbone.scene_tokens.detach().float().cpu()
        assert abs(float(slots.std())-conf['initial_std'])<.001
        atomic(root/'initial_state.json',dict(slot_std=float(slots.std()),slot_mean=float(slots.mean()),slot_sha256=hashlib.sha256(slots.numpy().tobytes()).hexdigest(),seed=conf['seed']))
    original_loss_metrics=model.loss_metrics
    def guarded_loss_metrics(pred,batch):
        losses,metrics=original_loss_metrics(pred,batch)
        total=losses['total'].detach()
        if not torch.isfinite(total).all() or total.abs().max()>conf.get('max_total_loss',100.):
            raise RuntimeError(f"Abnormal loss before backward: scene={batch.get('name')}, total={total.cpu().tolist()}")
        return losses,metrics
    model.loss_metrics=guarded_loss_metrics
    tc=dict(conf['train']);tc['num_steps']=end
    # Stage-specific warmup followed by cosine decay, independent of old pilots.
    duration=end-phase['start_step'];warm=phase['lr_warmup']
    tc['lr_schedule']=dict(type='SequentialLR',on_epoch=False,options=dict(milestones=[warm],schedulers=[
        dict(type='LinearLR',options=dict(start_factor=.1,total_iters=warm)),
        dict(type='CosineAnnealingLR',options=dict(T_max=max(1,duration-warm),eta_min=0.))]))
    trainer=Trainer.init(OmegaConf.create(tc),model,device=torch.device('cuda',local));trainer._view_schedule_frac=0.
    if 'lr_floor_factor' in phase:
        from tools.scene_token.continuation_schedule import stage_factor
        for group in trainer.optimizer.param_groups:group['lr']=group['initial_lr']
        trainer.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(trainer.optimizer,lambda elapsed:stage_factor(elapsed,duration,warm,phase['lr_floor_factor']))
    controlled_runtime(model)
    start=phase['start_step']
    if saved:
        expected=conf['continuation']['parent_config_sha256'] if a.initialize_from else hashlib.sha256((root/'config.json').read_bytes()).hexdigest()
        assert saved['run_config_sha256']==expected,'Run configuration changed'
        start=saved['step'];assert phase['start_step']<=start<=end
        if a.initialize_from:
            assert a.phase=='joint' and saved['phase']=='joint' and start==phase['start_step']==conf['continuation']['parent_step']
            assert a.initialize_from.resolve()==Path(conf['continuation']['checkpoint']).resolve()
            matched=migrate_optimizer(model,trainer.optimizer,saved)
            if rank==0:atomic(root/'stage_transition.json',dict(step=start,optimizer_states_preserved=matched,new_block_range=[phase['unfreeze_from'],39],scheduler='new stage warmup and cosine with relative floor',parent=str(a.initialize_from)))
        elif saved['phase']==a.phase:
            trainer.optimizer.load_state_dict(saved['optimizer']);trainer.lr_scheduler.load_state_dict(saved['lr_scheduler'])
        else:
            assert a.phase=='joint' and saved['phase']=='warmup' and start==phase['start_step']
            matched=migrate_optimizer(model,trainer.optimizer,saved)
            if rank==0:atomic(root/'stage_transition.json',dict(step=start,optimizer_states_preserved=matched,new_block_range=[phase['unfreeze_from'],39],scheduler='new stage warmup and cosine'))
        trainer.scaler.load_state_dict(saved['scaler']);set_rng(saved['rng_by_rank'][rank])
        del saved;gc.collect()
    else:
        torch.manual_seed(51000+rank);np.random.seed(51000+rank);random.seed(51000+rank)
    trainer.tot_it=start;trainer.tot_n_samples=start*world
    data=TrainingData(conf['dataset'],conf['data']);params=dict(model.named_parameters());names=optimizer_names(model,trainer.optimizer)
    if rank==0:
        atomic(root/f'effective_{a.phase}.json',dict(model=OmegaConf.to_container(model.conf,resolve=True),head=OmegaConf.to_container(model.gaussian_head.conf,resolve=True),
            train=tc,actual_stage_schedule=dict(phase),position=adapter.specification(),train_scenes=len(data.train),val_scenes=len(data.val),
            trainable_numel=sum(p.numel() for p in model.parameters() if p.requires_grad),optimizer_groups=[dict(lr=g['initial_lr'],names=n) for g,n in zip(trainer.optimizer.param_groups,names)]))
    stop=[False]
    def request_stop(*_):stop[0]=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    def status(state,step,**extra):
        if rank==0:publish(root,dict(state=state,phase=a.phase,step=step,total_steps=conf['total_steps'],updated=time.time(),free_gib=shutil.disk_usage(root).free/2**30,**extra))
    def assessment(step):
        status('evaluating',step,message='固定训练参考及独立验证 V2/V6 评估中')
        state=rng_state();model.eval();rows=[]
        probes=conf.get('train_probe_scenes',data.train[:8]);assert set(probes)<=set(data.train)
        groups=[('train_probe',probes,2),('val_v2',data.val,2),('val_v6',data.val,6)]
        if conf.get('eval_train_v6'):groups.append(('train_probe_v6',probes,6))
        for group,keys,views in groups:
            for key in keys[rank::world]:
                window=data.window(key);batch=data.fixed(key,window[f'context{views}'],window['target']);device_batch=batch_to_device(batch,trainer.device)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=model(device_batch);losses,metrics=model.loss_metrics(pred,device_batch)
                values=floats(losses)|floats(metrics)
                assert all(math.isfinite(x) for x in values.values()),'Nonfinite evaluation'
                folder=root/'renders'/f'step_{step:06d}'/group/key;folder.mkdir(parents=True,exist_ok=True);images=[]
                for j in range(len(window['target'])):
                    item={}
                    for tag,tensor in [('gt',device_batch['target']['image']),('render',pred['target_rgb'])]:
                        dest=folder/f'{j}_{tag}.png';array=(tensor[0,j].detach().float().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
                        Image.fromarray(array).save(dest);item[tag]=dest.relative_to(root).as_posix()
                    images.append(item)
                rows.append(dict(group=group,scene=key,context=window[f'context{views}'],targets=window['target'],images=images,**values))
                del pred,losses,metrics,device_batch,batch
        gathered=[None]*world;dist.all_gather_object(gathered,rows)
        if rank==0:
            allrows=[r for part in gathered for r in part];sets={}
            for group,_,_ in groups:
                rs=sorted([r for r in allrows if r['group']==group],key=lambda r:r['scene'])
                sets[group]=dict(scenes=rs,mean={k:float(np.mean([r[k] for r in rs])) for k in ['total','psnr','lpips','activated_pct']})
            path=root/'evaluations.json';history=json.loads(path.read_text()) if path.exists() else []
            history=[e for e in history if e['step']!=step]+[dict(step=step,phase=a.phase,updated=time.time(),sets=sets)];history.sort(key=lambda e:e['step']);atomic(path,history)
            atomic(root/'evaluation_plan.json',dict(train_probe_role='training reference, not held-out',validation_role='scene-disjoint',scenes={r['group']+'/'+r['scene']:dict(context=r['context'],target=r['targets']) for r in allrows}))
            print(json.dumps(dict(evaluation=step,sets={k:v['mean'] for k,v in sets.items()})),flush=True)
            if a.initialize_from and step==phase['start_step']:
                from tools.scene_token.continuation_schedule import check_parent_evaluation
                parent=json.loads((Path(conf['continuation']['run'])/'evaluations.json').read_text())
                checks=check_parent_evaluation(next(e for e in parent if e['step']==step),sets)
                atomic(root/'continuity_check.json',checks)
        set_rng(state);model.train();dist.barrier();status('running',step)
    def checkpoint(step,reason):
        status('saving',step,message='原子保存模型、优化器、调度器、scaler 和八卡 RNG')
        states=[None]*world;dist.all_gather_object(states,rng_state())
        if rank==0:
            folder=root/'checkpoints';folder.mkdir(exist_ok=True);dest=folder/f'step_{step:06d}.pt';tmp=dest.with_suffix('.tmp')
            if shutil.disk_usage(root).free<40*2**30:raise RuntimeError('Less than 40 GiB free before checkpoint')
            obj=dict(schema='scene-token-main-v1',phase=a.phase,step=step,model=model.state_dict(),optimizer=trainer.optimizer.state_dict(),
                optimizer_names=optimizer_names(model,trainer.optimizer),lr_scheduler=trainer.lr_scheduler.state_dict(),scaler=trainer.scaler.state_dict(),
                rng_by_rank=states,conf=conf,run_config_sha256=hashlib.sha256((root/'config.json').read_bytes()).hexdigest(),tot_n_samples=step*world)
            torch.save(obj,tmp);tmp.replace(dest)
            if step==conf.get('first_check_step',50):
                reloaded=torch.load(dest,map_location='cpu',weights_only=False)
                for name in ['backbone.backbone.scene_tokens','scene_color_query.weight','gaussian_head.gaussian_head.1.weight']:
                    assert torch.equal(reloaded['model'][name],params[name].detach().cpu())
                assert reloaded['step']==step and len(reloaded['rng_by_rank'])==world
                atomic(root/'first_checkpoint_check.json',dict(passed=True,step=step,strict_schema=True,selected_parameters_exact=True,optimizer_entries=len(reloaded['optimizer']['state']),bytes=dest.stat().st_size))
                del reloaded
            path=root/'checkpoints.json';idx=json.loads(path.read_text()) if path.exists() else dict(records=[])
            idx['records']=[r for r in idx['records'] if r['step']!=step]+[dict(step=step,path=str(dest),bytes=dest.stat().st_size,reason=reason,phase=a.phase)]
            idx['records'].sort(key=lambda r:r['step']);idx['latest']=str(dest)
            history=json.loads((root/'evaluations.json').read_text())
            available={r['step'] for r in idx['records']};ev=[e for e in history if e['step'] in available]
            best_set=conf.get('best_validation_set','val_v2')
            best_psnr=max(ev,key=lambda e:e['sets'][best_set]['mean']['psnr'])['step'] if ev else step
            best_lpips=min(ev,key=lambda e:e['sets'][best_set]['mean']['lpips'])['step'] if ev else step
            keep={r['step'] for r in idx['records'][-3:]}|{best_psnr,best_lpips}|set(conf['milestones'])
            removed=[]
            for r in idx['records']:
                if r['step'] not in keep:
                    candidate=Path(r['path']).resolve();assert candidate.parent==folder.resolve() and candidate.name==f"step_{r['step']:06d}.pt"
                    candidate.unlink(missing_ok=True);removed.append(r)
            idx['records']=[r for r in idx['records'] if r['step'] in keep];idx['best_psnr_step']=best_psnr;idx['best_lpips_step']=best_lpips;idx['best_validation_set']=best_set
            atomic(path,idx)
            if removed:
                with (root/'checkpoint_retention.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),removed=removed))+'\n')
        dist.barrier();status('running',step)
    if start==0 or a.initialize_from:assessment(start)
    if rank==0:atomic(root/f'process_{a.phase}.json',dict(pid=os.getpid(),world_size=world,start=start,end=end))
    last=start
    for step in range(start+1,end+1):
        began=time.perf_counter();views=phase['views'] if 'views' in phase else min(6,2+(step-phase['start_step'])//phase['view_interval'])
        batch,selection=data.sample(step,rank,world,views)
        trainer.model.train();trainer.tot_it=step-1;trainer.step_timer.reset()
        slot=model.backbone.backbone.scene_tokens;before=int(trainer.optimizer.state.get(slot,{}).get('step',0))
        grads={};hooks=[]
        tracked=['backbone.backbone.scene_tokens','scene_color_query.weight','gaussian_head.gaussian_head.1.weight']
        if a.phase=='joint':tracked+=['backbone.backbone.blocks.39.attn.qkv.weight']
        if conf.get('continuation'):tracked+=['backbone.backbone.blocks.18.attn.qkv.weight','backbone.backbone.blocks.29.attn.qkv.weight']
        if step==start+1 or step%100==0:
            for name in tracked:
                assert name in params,name
                def capture(g,n=name):
                    assert torch.isfinite(g).all(),f'Nonfinite gradient: {n}'
                    grads[n]=float(g.float().norm())/trainer.scaler.get_scale()
                hooks.append(params[name].register_hook(capture))
        pred,metrics=trainer.train_step(batch,log_grad_norm=True)
        for h in hooks:h.remove()
        assert int(trainer.optimizer.state[slot]['step'])==before+1,'Optimizer update was skipped'
        if hooks:assert len(grads)==len(tracked) and all(math.isfinite(v) and v>0 for v in grads.values())
        values=floats(metrics);assert all(math.isfinite(v) for v in values.values()),'Nonfinite metrics'
        trainer.tot_it=step;trainer.tot_n_samples=step*world;torch.cuda.synchronize();seconds=time.perf_counter()-began
        record=dict(step=step,phase=a.phase,views=views,selection=selection,metrics=values,gradients=grads,seconds=seconds,
            lr=[g['lr'] for g in trainer.optimizer.param_groups],memory_gib=torch.cuda.max_memory_allocated()/2**30)
        with (root/f'rank{rank}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        reduced=torch.tensor([values['loss/total'],values['psnr'],values['lpips'],seconds,values['loss/mse_loss'],values['loss/depth_loss'],values['loss/location_loss'],values['activated_pct']],device=trainer.device)
        dist.all_reduce(reduced);reduced/=world
        if rank==0:
            row=dict(step=step,phase=a.phase,views=views,loss=float(reduced[0]),psnr=float(reduced[1]),lpips=float(reduced[2]),seconds=float(reduced[3]),rgb_l1=float(reduced[4]),depth=float(reduced[5]),location=float(reduced[6]),activated_pct=float(reduced[7]),lr=record['lr'],updated=time.time())
            with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if step%10==0 or step==start+1:
                status('running',step,views=views,seconds=row['seconds']);print(json.dumps(row),flush=True)
        del pred,metrics,batch;last=step
        requested=torch.tensor(int(stop[0] or (root/'STOP').exists()),device=trainer.device);dist.all_reduce(requested,op=dist.ReduceOp.MAX)
        if step==conf.get('first_check_step',50) or step%conf['eval_every']==0 or step==end:assessment(step)
        if step==conf.get('first_check_step',50) or step%conf['checkpoint_every']==0 or step==end or requested.item():checkpoint(step,'pause' if requested.item() else 'periodic')
        if requested.item():status('paused',step,message='已保存后停止；需手动恢复');break
    else:status('phase_complete' if a.phase=='warmup' else 'complete',last,message='当前阶段已完成并保存')
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':
    try:main()
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
        if '--run' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--run')+1]);rank=os.environ.get('RANK','0')
            atomic(root/f'error_rank{rank}.json',dict(updated=time.time(),error=error))
        raise
