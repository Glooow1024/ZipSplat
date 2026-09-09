"""Paired 200-step fixed-batch ablation: no/shared/distinct 1D scene QK RoPE."""
import argparse
import gc
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from omegaconf import OmegaConf
from splatfactory.datasets import get_dataset
from splatfactory.utils.mappings import batch_to_device
from tools.scene_token.train_initial_test import build,update,evaluate,save_checkpoint,rng_state,set_rng,param_hash,dump
from tools.scene_token.diagnose_layers import Trace,describe,snapshot
from tools.scene_token.position_1d import ScenePosition1D


def trace_model(model,batch,path):
    model.eval();data=batch_to_device(batch,'cuda');state=rng_state()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): pred=model(data)
    ref=snapshot(pred);del pred
    set_rng(state)
    trace=Trace(model,data['context']['image'].shape[1]);trace.install()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): pred=model(data)
    trace.remove();out=snapshot(pred)
    parity={k:float((out[k]-ref[k]).abs().max()) for k in ref}
    assert all(v==0 for v in parity.values()),parity
    result=trace.summarize()
    result.update(hook_parity_max_abs=parity,sample=describe(batch))
    dump(path,result)
    del trace,pred,result,data,out,ref
    set_rng(state);model.train()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(256);np.random.seed(256);random.seed(256)
    old=json.loads((args.run/'config.json').read_text())
    dc=OmegaConf.create(old['data'])
    train=list(get_dataset(dc.name)(dc,split='train').get_loader(num_workers=0))
    vc=OmegaConf.merge(dc,dc.val_overrides)
    val=list(get_dataset(vc.name)(vc,split='val').get_loader(num_workers=0))
    batches=dict(train=[describe(b) for b in train],validation=[describe(b) for b in val])
    assert batches==json.loads((args.run/'batches.json').read_text())
    initial_rng=rng_state()
    protocol=dict(seed=256,modes=['none','shared','distinct'],steps=200,
        train_scope='all 200 steps use train[0], V2/B1/4 targets; fresh released initialization in each arm',
        validation='two fixed val scenes V6/8 targets, no camera alignment/TTO, no target features in reconstruction',
        position='1D slot RoPE after QK norm at all 40 blocks; scale1,base10000; patch/camera/V/residual unchanged',
        shared_control='all scene slots use position127.5; distinct uses0..255; shared across views',
        conf=old,batches=batches,note='Not directly comparable to previous 100 fixed +100 eight-batch run. No production-model changes.')
    dump(args.output/'protocol.json',protocol)
    files=['tools/scene_token/position_1d.py','tools/scene_token/validate_position_1d.py',
           'tools/scene_token/train_initial_test.py','tools/scene_token/diagnose_layers.py',
           'splatfactory/models/modules/attention.py','splatfactory/models/modules/transformer_block.py',
           'splatfactory/models/encoders/dav3_dino.py','splatfactory/models/networks/zipsplat.py']
    dump(args.output/'provenance.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        status=subprocess.check_output(['git','status','--short'],text=True),torch=torch.__version__,
        files={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}))
    summary={};initial_hashes=[]
    for mode in protocol['modes']:
        output=args.output/mode;output.mkdir()
        set_rng(initial_rng)
        model,trainer=build(old['model'],old['train'],True)
        start_hash=param_hash(model,False);frozen_hash=param_hash(model,True)
        initial_hashes.append(start_hash)
        assert len(set(initial_hashes))==1,'Arms did not start from identical parameters'
        # Check mode=none is exactly the current model, despite persistent marker.
        if mode=='none':
            data=batch_to_device(train[0],'cuda');model.eval()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): before=snapshot(model(data))
        adapter=ScenePosition1D(model,mode)
        if mode=='none':
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): after=snapshot(model(data))
            errors={k:float((before[k]-after[k]).abs().max()) for k in before}
            assert all(v==0 for v in errors.values())
            dump(output/'disabled_parity.json',errors);del before,after,data
        spec=adapter.specification()
        conf={**old,'experiment':spec,'scope':protocol['train_scope']}
        dump(output/'config.json',conf)
        evals=[];records=[]
        def assess(step):
            folder=output/f'step{step:03d}';folder.mkdir()
            (folder/'train').mkdir();(folder/'validation').mkdir()
            tr=evaluate(model,[train[0]],folder/'train','after')
            va=evaluate(model,val,folder/'validation','after')
            record=dict(step=step,train=tr,validation=va)
            evals.append(record);dump(output/'evaluations.json',evals)
            print(json.dumps(dict(mode=mode,step=step,train_loss=tr['mean']['total'],train_psnr=tr['mean']['psnr'],val_psnr=va['mean']['psnr'])),flush=True)
            if step in (0,200):
                trace_model(model,train[0],folder/'train_trace.json')
                trace_model(model,val[0],folder/'val_trace.json')
        assess(0);torch.cuda.reset_peak_memory_stats()
        for step in range(1,201):
            row=update(trainer,train[0],step);records.append(row)
            with (output/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if step%25==0:print(json.dumps(dict(mode=mode,step=step,loss=row['metrics']['loss/total'])),flush=True)
            if step in (100,200):assess(step)
        assert frozen_hash==param_hash(model,True)
        save_checkpoint(output/'checkpoint_200.pt',model,trainer,conf)
        # Verify marker and on-disk model values; installed adapter required.
        saved=torch.load(output/'checkpoint_200.pt',map_location='cpu',weights_only=False)
        marker='backbone.backbone.scene_position_experiment_spec'
        assert saved['conf']['experiment']==spec and marker in saved['model']
        expected_spec=model.state_dict()[marker].cpu()
        assert torch.equal(saved['model'][marker],expected_spec)
        model.load_state_dict(saved['model'],strict=True)
        del saved
        summary[mode]=dict(completed_steps=200,frozen_parameters_unchanged=True,initial_trainable_sha256=start_hash,
            initial=evals[0],final=evals[-1],position=spec,checkpoint_bytes=(output/'checkpoint_200.pt').stat().st_size,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            median_update_seconds=float(np.median([r['seconds'] for r in records[10:]])))
        dump(args.output/'summary.json',summary)
        print('COMPLETE '+mode,flush=True)
        adapter.remove();del adapter,trainer,model;gc.collect();torch.cuda.empty_cache()
    dump(args.output/'completion.json',dict(completed=True,arms=3,total_optimizer_updates=600,initial_parameters_identical=True))


if __name__=='__main__':main()
