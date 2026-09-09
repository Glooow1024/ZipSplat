"""Freeze reserved frames and validate dynamic samples before any training."""
import argparse,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from PIL import Image,ImageDraw
from tools.scene_token.multiscene_data import PilotData,create_plan
from tools.scene_token.train_initial_test import dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);np.random.seed(256);random.seed(256);torch.manual_seed(256)
    old=json.loads(a.source_config.read_text());data=PilotData(old['data']);plan=create_plan(data);data.plan=plan
    dump(a.output/'plan.json',plan);dump(a.output/'base_config.json',old)
    audit={};canvas=Image.new('RGB',(3*252,10*280),'white');draw=ImageDraw.Draw(canvas)
    for row,key in enumerate(sorted(data.raw)):
        entry=plan['scenes'][key];batch=data.fixed(key,entry['context2'],entry['target'])
        assert not set(entry['context6']).intersection(entry['target']) and set(entry['context2']).issubset(entry['context6'])
        assert batch['context']['image'].shape==(1,2,3,252,252)
        assert batch['target']['image'].shape==(1,8,3,252,252)
        views=[batch['context']['image'][0,0],batch['context']['image'][0,1],batch['target']['image'][0,0]]
        for col,x in enumerate(views):
            im=Image.fromarray((x.permute(1,2,0).numpy().clip(0,1)*255).round().astype('uint8'))
            canvas.paste(im,(col*252,row*280+28))
            draw.text((col*252+4,row*280+5),f'{data.split[key]} {key[:8]} '+['context0','context1','target0'][col],fill='black')
        if data.split[key]=='train':
            records=[]
            for _ in range(64):
                b,record=data.training(key)
                assert not set(entry['reserved']).intersection(record['context']+record['target'])
                assert torch.isfinite(b['context']['image']).all() and torch.isfinite(b['target']['image']).all()
                records.append(record)
            audit[key]=dict(samples=records,unique_contexts=len({tuple(r['context']) for r in records}),
                unique_target_sets=len({tuple(r['target']) for r in records}),rejections=sum(r['rejected'] for r in records))
    canvas.save(a.output/'sampling_preview.png')
    dump(a.output/'data_audit.json',dict(passed=True,training_samples=512,train_scenes=8,validation_scenes=2,
        reserved_training_frames=64,scenes=audit,limitation='512 deterministic sampled batches checked, not exhaustive future sampling; runtime asserts enforce exclusions'))
    print(json.dumps({k:{m:v[m] for m in ['unique_contexts','unique_target_sets','rejections']} for k,v in audit.items()}),flush=True)


if __name__=='__main__':main()
