"""Audit every sampled frame and restore the final distributed checkpoint on one GPU."""
import argparse,hashlib,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from tools.scene_token.multiscene_data import PilotData
from tools.scene_token.train_initial_test import build,evaluate,param_hash,dump
from tools.scene_token.continue_scale_fit import equal_state
from tools.scene_token.position_1d import ScenePosition1D
from tools.scene_token.audit_repeatability import controlled_runtime


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args();root=a.input
    torch.set_num_threads(4)
    completion=json.loads((root/'completion.json').read_text());assert completion['passed'] and completion['optimizer_updates']==5000
    plan=json.loads((root/'plan.json').read_text());conf=json.loads((root/'config.json').read_text());source=json.loads((root/'provenance.json').read_text())
    for name,expected in source['files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
    keys=conf['multiscene_experiment']['rank_scenes'];coverage={};lr_reference=None
    for rank,key in enumerate(keys):
        records=[json.loads(x) for x in (root/f'steps_rank{rank}.jsonl').read_text().splitlines()]
        assert [r['step'] for r in records]==list(range(1,5001))
        forbidden=set(plan['scenes'][key]['reserved']);seen=set();contexts=set();targets=set();context_counts=Counter();target_counts=Counter()
        for r in records:
            assert r['scene']==key
            c=r['selection']['context'];t=r['selection']['target']
            assert len(c)==2 and len(set(c))==2 and len(t)==4
            assert 8<=abs(c[1]-c[0])<=22
            assert all(min(c)<=i<=max(c) for i in t)
            assert not forbidden.intersection(c+t)
            assert all(np.isfinite(v) and v>0 for v in r['gradients'].values())
            assert all(np.isfinite(v) for v in r['metrics'].values())
            seen.update(c+t);contexts.add(tuple(c));targets.add(tuple(t));context_counts.update(c);target_counts.update(t)
        lrs=[r['lr'] for r in records]
        if lr_reference is None:lr_reference=lrs
        assert lrs==lr_reference
        coverage[key]=dict(updates=len(records),distinct_context_pairs=len(contexts),distinct_target_sets=len(targets),
            distinct_frames=len(seen),allowed_frames=plan['scenes'][key]['num_views']-len(forbidden),
            sampled_frames=sorted(seen),reserved_frames=sorted(forbidden),total_rejections=sum(r['selection']['rejected'] for r in records))
        allowed=sorted(set(range(plan['scenes'][key]['num_views']))-forbidden)
        coverage[key]['frame_exposures']={str(i):dict(context=context_counts[i],target=target_counts[i]) for i in allowed}
        coverage[key]['target_exposures_per_allowed_frame']=dict(min=min(target_counts[i] for i in allowed),
            median=float(np.median([target_counts[i] for i in allowed])),max=max(target_counts[i] for i in allowed))
    assert len(completion['ranks'])==8 and len({r['final_trainable_sha256'] for r in completion['ranks']})==1
    gradient_checks=list(root.glob('gradient_check_rank*.json'));assert len(gradient_checks)==8
    for f in gradient_checks:assert json.loads(f.read_text())['passed']
    isolation=json.loads((root.parent/'depth_isolation.json').read_text())
    assert isolation['passed'] and isolation['heldout_isolated_from_nonheldout_teacher_inputs']
    for r in isolation['reports']:
        reserved=set(plan['scenes'][r['key']]['target'])
        assert all(not reserved.intersection(ids) for ids in r['allowed_frame_teacher_inputs'].values())
    checkpoint=root/'checkpoint_5000.pt'
    with checkpoint.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    assert sha==completion['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['tot_it']==5000 and saved['tot_n_samples']==40000 and len(saved['rng_by_rank'])==8
    model,trainer=build(saved['conf']['model'],saved['conf']['train'],False)
    spec=saved['conf']['experiment'];adapter=ScenePosition1D(model,spec['mode'],scale=spec['scale'],base=spec['base'])
    trainer.load_checkpoint(saved,strict=True,load_state=True)
    assert equal_state(trainer.optimizer.state_dict(),saved['optimizer'])
    assert equal_state(trainer.lr_scheduler.state_dict(),saved['lr_scheduler'])
    assert equal_state(trainer.scaler.state_dict(),saved['scaler'])
    assert trainer.tot_it==5000 and trainer.tot_n_samples==40000
    assert all(int(s['step'])==5000 for s in trainer.optimizer.state.values())
    assert param_hash(model,True)==source['frozen_sha256']
    assert param_hash(model,False)==completion['ranks'][0]['final_trainable_sha256']
    del saved;controlled_runtime(model)
    data=PilotData(conf['data'],plan);assert data.tar_hashes==plan['tar_hashes']
    probes=json.loads((root/'train_probes.json').read_text())
    sets=dict(train_probe=[data.fixed(k,probes[k]['context'],probes[k]['target']) for k in keys],
        trained_scene_v2=data.evaluation('train',2),new_scene_v2=data.evaluation('validation',2),new_scene_v6=data.evaluation('validation',6))
    expected=json.loads((root/'evaluations.json').read_text())[-1];errors={};scene_errors={}
    for name,batches in sets.items():
        actual=evaluate(model,batches,root,'check_only')
        errors[name]={m:abs(actual['mean'][m]-expected[name]['mean'][m]) for m in ['total','psnr','lpips']}
        assert all(v<(1e-2 if m=='psnr' else 1e-3) for m,v in errors[name].items()),errors[name]
        assert len(actual['scenes'])==len(expected[name]['scenes'])
        scene_errors[name]={}
        for got,want in zip(actual['scenes'],expected[name]['scenes']):
            assert got['scene']==want['scene']
            delta={m:abs(got[m]-want[m]) for m in ['total','psnr','lpips']}
            assert all(v<(1e-2 if m=='psnr' else 1e-3) for m,v in delta.items()),(name,got['scene'],delta)
            scene_errors[name][got['scene']]=delta
    images=0
    for step in [0,500,1000,2500,5000]:
        for name,batches in sets.items():
            for i,b in enumerate(batches):
                for j in range(b['target']['image'].shape[1]):
                    for tag in ['gt','render']:
                        assert (root/f'step{step:04d}'/name/f'after_{i}_{j}_{tag}.png').exists();images+=1
    dump(root/'validation.json',dict(passed=True,global_optimizer_updates=5000,scene_exposures=40000,
        all_sampled_frames_audited=True,teacher_exclusion_verified=True,rank_parameters_identical=True,
        frozen_backbone_unchanged=True,full_optimizer_scheduler_scaler_restored=True,checkpoint_sha256=sha,
        restored_metrics_errors=errors,restored_per_scene_errors=scene_errors,coverage=coverage,png_count=images,
        limitation='Single-process restore validates full state and inference; not an additional distributed resumed-update equivalence experiment'))
    print(json.dumps(dict(passed=True,errors=errors,coverage={k:{m:r[m] for m in ['distinct_context_pairs','distinct_frames','allowed_frames']} for k,r in coverage.items()},png_count=images)),flush=True)


if __name__=='__main__':main()
