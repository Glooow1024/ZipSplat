"""Validate six continuations and strict inference restoration at step5000."""
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from omegaconf import OmegaConf
from splatfactory.datasets import get_dataset
from splatfactory.models.networks.zipsplat import ZipSplat
from splatfactory.utils.mappings import batch_to_device
from tools.scene_token.position_1d import ScenePosition1D
from tools.scene_token.audit_repeatability import controlled_runtime
from tools.scene_token.diagnose_layers import snapshot


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args(); root=args.input
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    protocol=json.loads((root/'seed256/scale1/protocol.json').read_text())
    conf=json.loads((root/'seed256/scale1/config.json').read_text())
    entry=protocol['batches']['train'][0]; indices=root/'reload_indices.json'
    indices.write_text(json.dumps({entry['scene'].removeprefix('dl3dv-'):
        {k:[int(i) for i in entry[k]] for k in ['context','target']}}))
    dc=OmegaConf.create(conf['data']); dc.test_shard_dir=dc.train_shard_dir
    dc.random_reference_view=False; dc.view_sampler={'name':'eval_sampler','indices_file':str(indices)}
    batch=next(iter(get_dataset(dc.name)(dc,split='val').get_loader(num_workers=0)))
    for k in ['context','target']: assert batch[k]['index'].flatten().tolist()==entry[k]
    data=batch_to_device(batch,'cuda'); results={}; frozen=[]; lrs=None
    for seed in [256,257,258]:
        for arm in ['scale1','scale4']:
            path=root/f'seed{seed}'/arm; key=f'{seed}/{arm}'
            complete=json.loads((path/'completion.json').read_text())
            resume=json.loads((path/'resume_check.json').read_text())
            assert complete['completed'] and complete['optimizer_updates']==4000 and complete['final_step']==5000
            assert resume['passed'] and resume['full_model_optimizer_scheduler_scaler_equal']
            assert complete['frozen_parameters_unchanged']; frozen.append(complete['frozen_parameters_sha256'])
            p=json.loads((path/'protocol.json').read_text()); assert p['batches']==protocol['batches']
            records=[json.loads(x) for x in (path/'steps.jsonl').read_text().splitlines()]
            assert [r['step'] for r in records]==list(range(1001,5001))
            assert all(all(np.isfinite(v) and v>0 for v in r['gradients'].values()) for r in records)
            if lrs is None: lrs=records[0]['lr']
            assert all(r['lr']==lrs for r in records)
            for name in ['train_trace.json','val_trace.json']:
                trace=json.loads((path/'step5000'/name).read_text())
                assert all(v==0 for v in trace['hook_parity_max_abs'].values())
            checkpoint=path/'checkpoint_5000.pt'
            with checkpoint.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
            assert sha==complete['checkpoint_sha256']
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert saved['tot_it']==saved['tot_n_samples']==5000
            assert all(int(s['step'])==5000 for s in saved['optimizer']['state'].values())
            torch.use_deterministic_algorithms(False)
            mc=dict(saved['conf']['model']); mc.update(weights=None,scene_token_init_from_base=False)
            model=ZipSplat(mc).cuda().eval()
            spec=saved['conf']['experiment']; adapter=ScenePosition1D(model,spec['mode'],scale=spec['scale'],base=spec['base'])
            model.load_state_dict(saved['model'],strict=True); del saved
            controlled_runtime(model)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(data); losses,metrics=model.loss_metrics(pred,data)
                first=snapshot(pred); second=snapshot(model(data))
            parity={k:float((first[k]-second[k]).abs().max()) for k in first}
            assert all(v==0 for v in parity.values())
            expected=json.loads((path/'evaluations.json').read_text())[-1]['train']['mean']
            errors=dict(loss=abs(float(losses['total'].mean())-expected['total']),psnr=abs(float(metrics['psnr'].mean())-expected['psnr']))
            assert errors['loss']<1e-3 and errors['psnr']<1e-2,(key,errors)
            results[key]=dict(passed=True,new_updates=4000,final_step=5000,checkpoint_sha256=sha,
                full_state_resume=resume,strict_final_restore=True,restore_errors=errors,repeated_forward_max_abs=parity)
            print(json.dumps({key:results[key]}),flush=True)
            adapter.remove(); del model,adapter,pred,losses,metrics,first,second
            gc.collect(); torch.cuda.empty_cache()
    assert len(set(frozen))==1
    (root/'validation.json').write_text(json.dumps(dict(passed=True,new_optimizer_updates=24000,
        total_updates_including_source=30000,trace_groups=12,frozen_hash_equal=True,identical_lrs=True,
        arms=results,limitation='Full initial state equality and final fresh inference; no bitwise-identical independent continuation claim'),indent=2)+'\n')


if __name__=='__main__': main()
