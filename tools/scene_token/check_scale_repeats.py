"""Validate all paired runs, source records, and six fresh checkpoint loads."""
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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
    parser=argparse.ArgumentParser(); parser.add_argument('--input', type=Path, required=True)
    args=parser.parse_args(); root=args.input
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    protocol=json.loads((root/'seed256/protocol.json').read_text())
    entry=protocol['batches']['train'][0]; indices=root/'reload_indices.json'
    indices.write_text(json.dumps({entry['scene'].removeprefix('dl3dv-'):
        {k:[int(i) for i in entry[k]] for k in ['context','target']}}))
    dc=OmegaConf.create(protocol['conf']['data']); dc.test_shard_dir=dc.train_shard_dir
    dc.random_reference_view=False; dc.view_sampler={'name':'eval_sampler','indices_file':str(indices)}
    batch=next(iter(get_dataset(dc.name)(dc,split='val').get_loader(num_workers=0)))
    for k in ['context','target']: assert batch[k]['index'].flatten().tolist()==entry[k]
    data=batch_to_device(batch,'cuda'); result={}; first_lrs=None; init_hashes=[]; frozen_hashes=[]
    for seed in [256,257,258]:
        run=root/f'seed{seed}'; completion=json.loads((run/'completion.json').read_text())
        assert completion['completed'] and completion['optimizer_updates']==2000
        p=json.loads((run/'protocol.json').read_text()); assert p['batches']==protocol['batches']
        summary=json.loads((run/'summary.json').read_text())
        pair_hashes=[]
        for name in ['scale1','scale4']:
            arm=run/name; key=f'{seed}/{name}'
            pair_hashes.append(summary[name]['initial_parameters_before_scaling_sha256'])
            frozen_hashes.append(summary[name]['frozen_parameters_sha256'])
            assert summary[name]['frozen_parameters_unchanged']
            records=[json.loads(line) for line in (arm/'steps.jsonl').read_text().splitlines()]
            assert [r['step'] for r in records]==list(range(1,1001))
            assert all(all(np.isfinite(v) and v>0 for v in r['gradients'].values()) for r in records)
            lrs=[r['lr'] for r in records]
            if first_lrs is None: first_lrs=lrs
            assert lrs==first_lrs
            for relative in ['step0000/train_trace.json','step1000/train_trace.json','step1000/val_trace.json']:
                trace=json.loads((arm/relative).read_text())
                assert all(v==0 for v in trace['hook_parity_max_abs'].values())
            path=arm/'checkpoint_1000.pt'
            with path.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
            saved=torch.load(path,map_location='cpu',weights_only=False)
            assert saved['tot_it']==1000 and saved['conf']['repetition_experiment']['seed']==seed
            conf=dict(saved['conf']['model']); conf.update(weights=None,scene_token_init_from_base=False)
            torch.use_deterministic_algorithms(False)
            model=ZipSplat(conf).cuda().eval(); marker='backbone.backbone.scene_position_experiment_spec'
            try: model.load_state_dict(saved['model'],strict=True)
            except RuntimeError as error: assert marker in str(error)
            else: raise AssertionError('Unconfigured positional checkpoint was accepted')
            spec=saved['conf']['experiment']; adapter=ScenePosition1D(model,spec['mode'],scale=spec['scale'],base=spec['base'])
            assert torch.equal(saved['model'][marker],model.state_dict()[marker].cpu())
            model.load_state_dict(saved['model'],strict=True); del saved
            controlled_runtime(model)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(data); losses,metrics=model.loss_metrics(pred,data); first=snapshot(pred)
                second=snapshot(model(data))
            parity={k:float((first[k]-second[k]).abs().max()) for k in first}
            assert all(v==0 for v in parity.values())
            expected=json.loads((arm/'evaluations.json').read_text())[-1]['train']['mean']
            errors=dict(loss=abs(float(losses['total'].mean())-expected['total']),
                        psnr=abs(float(metrics['psnr'].mean())-expected['psnr']))
            assert errors['loss']<1e-3 and errors['psnr']<1e-2,(key,errors)
            result[key]=dict(checkpoint_sha256=sha,strict_restore=True,restore_errors=errors,
                repeated_forward_max_abs=parity,steps=1000,gradients_finite_nonzero=True)
            print(json.dumps({key:result[key]}),flush=True)
            adapter.remove(); del model,adapter,pred,losses,metrics,first,second
            gc.collect(); torch.cuda.empty_cache()
        assert len(set(pair_hashes))==1; init_hashes.append(pair_hashes[0])
    assert len(set(init_hashes))==3 and len(set(frozen_hashes))==1
    (root/'validation.json').write_text(json.dumps(dict(passed=True,optimizer_updates=6000,arms=result,
        paired_initialization=True,distinct_seed_initializations=True,frozen_backbone_equal_across_all=True,
        identical_lr_schedules=True,trace_groups=18,
        limitation='Fresh inference restoration and repeated forward; no cross-process identical-training claim'),indent=2)+'\n')


if __name__=='__main__': main()
