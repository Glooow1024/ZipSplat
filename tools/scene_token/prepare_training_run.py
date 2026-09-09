"""Prepare an expanded, scene-disjoint training set with eight bounded workers."""
import argparse, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot',type=Path,default=Path('/data/datasets/3dvision/ZipSplat-scene-token/pilot_20260907'))
    p.add_argument('--train-scenes',type=int,default=256);p.add_argument('--val-scenes',type=int,default=32)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--rank',type=int,default=-1)
    a=p.parse_args()
    if a.rank>=0:
        import torch
        from tools.scene_token.prepare_training_pilot import prepare_scene,digest
        from depth_anything_3.api import DepthAnything3
        torch.set_num_threads(4);torch.manual_seed(256+a.rank)
        m=json.loads((a.output/'manifest.json').read_text())
        jobs=m['selected'][a.rank::a.workers];teacher=None;errors=[];done=0
        args=SimpleNamespace(source=Path(m['source']),output=a.output,max_views=16)
        for record in jobs:
            report=a.output/'provenance'/(record['key']+'.json')
            try:
                if report.exists():
                    info=json.loads(report.read_text());assert digest(a.output/record['split']/record['shard'])==info['tar_sha256']
                else:
                    if teacher is None:teacher=DepthAnything3.from_pretrained(m['teacher']).cuda().eval()
                    prepare_scene(args,record,teacher)
                done+=1
            except Exception:
                err=dict(scene=record['scene'],traceback=traceback.format_exc());errors.append(err)
                write(a.output/'failures'/(record['key']+'.json'),err)
                print(json.dumps(err),flush=True);torch.cuda.empty_cache()
                if len(errors)>max(2,len(jobs)//10):raise RuntimeError('Too many conversion failures')
            write(a.output/f'worker{a.rank}.json',dict(rank=a.rank,done=done,total=len(jobs),failed=len(errors),updated=time.time(),scene=record['key']))
        return
    old=json.loads((a.pilot/'manifest.json').read_text());splits=json.loads((a.pilot/'splits.json').read_text())
    selected=[]
    for split,count in [('train',a.train_scenes),('validation',a.val_scenes)]:
        ordered=sorted(splits[split],key=lambda s:hashlib.sha256(('scene-token-pilot-256:'+s).encode()).hexdigest())
        for i,s in enumerate(ordered[:count]):
            frames=len(json.loads((Path(old['source'])/s/'transforms.json').read_text())['frames'])
            selected.append(dict(scene=s,key=Path(s).name,frames=frames,split=split,shard=f'shard-{i:06d}.tar'))
    assert len(set(r['key'] for r in selected))==len(selected)
    assert not {r['key'] for r in selected}.intersection(splits['official_test'])
    m=dict(old,version='scene-token-training-252-v1',selected=selected,requested_train=a.train_scenes,requested_validation=a.val_scenes,
        note='Frozen original scene split; validation scenes never enter training teacher groups; training probes are not held-out frames.')
    if (a.output/'manifest.json').exists():assert json.loads((a.output/'manifest.json').read_text())==m
    else:a.output.mkdir(parents=True,exist_ok=False);write(a.output/'manifest.json',m)
    write(a.output/'splits.json',splits)
    for folder in ['train','validation','provenance','failures','logs']:(a.output/folder).mkdir(exist_ok=True)
    jobs=[]
    for rank in range(a.workers):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        log=(a.output/'logs'/f'worker{rank}.log').open('a')
        cmd=[sys.executable,__file__,'--output',str(a.output),'--workers',str(a.workers),'--rank',str(rank)]
        jobs.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env));log.close()
    while any(j.poll() is None for j in jobs):
        reports=list((a.output/'provenance').glob('*.json'))
        write(a.output/'progress.json',dict(state='preparing',done=len(reports),total=len(selected),failed=len(list((a.output/'failures').glob('*.json'))),updated=time.time(),pids=[j.pid for j in jobs]))
        time.sleep(10)
    codes=[j.returncode for j in jobs]
    reports=[json.loads(p.read_text()) for p in sorted((a.output/'provenance').glob('*.json'))]
    counts={split:sum(r['split']==split for r in reports) for split in ['train','validation']}
    passed=all(c==0 for c in codes) and counts['train']>=int(a.train_scenes*.95) and counts['validation']>=int(a.val_scenes*.95)
    for split in counts:write(a.output/split/'index.json',{r['key']:r['shard'] for r in reports if r['split']==split})
    summary=dict(state='ready' if passed else 'failed',done=len(reports),total=len(selected),counts=counts,updated=time.time(),exit_codes=codes,
        frames=sum(r['frames'] for r in reports),tar_bytes=sum(r['tar_bytes'] for r in reports),failed=len(selected)-len(reports),passed=passed)
    write(a.output/'progress.json',summary);write(a.output/'summary.json',summary)
    if not passed:raise RuntimeError(summary)

if __name__=='__main__':main()
