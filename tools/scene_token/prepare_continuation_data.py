"""Expand the frozen split, reuse verified shards, audit geometry before training."""
import argparse, hashlib, io, json, os, shutil, subprocess, sys, tarfile, time, traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.scene_token.training_monitor import atomic

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def pose_audit(path):
    tf=json.loads(path.read_text());p=np.asarray([f['transform_matrix'] for f in tf['frames']],dtype=np.float64)
    assert p.shape[1:]==(4,4) and len(p)>=28 and np.isfinite(p).all(),'invalid pose shape/values'
    assert np.max(np.abs(p[:,3]-[0,0,0,1]))<1e-4,'invalid homogeneous row'
    rot=p[:,:3,:3];assert np.max(np.abs(rot.transpose(0,2,1)@rot-np.eye(3)))<.02,'invalid rotation'
    assert np.max(np.abs(np.linalg.det(rot)-1))<.02,'invalid rotation determinant'
    c=p[:,:3,3];r=np.linalg.norm(c-np.median(c,axis=0),axis=1);steps=np.linalg.norm(np.diff(c,axis=0),axis=1)
    med=float(np.median(r));jump=float(steps.max()/max(np.median(steps),1e-12))
    assert r.max()>1e-8,'degenerate camera trajectory'
    assert np.abs(c).max()<1e8 and r.max()/max(med,1e-12)<1e6 and jump<1e6,'catastrophic camera trajectory'
    for frame in tf['frames']:
        assert (path.parent/'images_4'/Path(frame['file_path']).name).is_file(),'missing RGB'
    return dict(frames=len(p),camera_radius_max=float(r.max()),camera_radius_median=med,jump_to_median=jump)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reuse',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--train-scenes',type=int,default=1024);a=ap.parse_args();root=a.output
    root.mkdir(parents=True,exist_ok=False)
    try:
        old=json.loads((a.reuse/'manifest.json').read_text());splits=json.loads((a.reuse/'splits.json').read_text())
        source=Path(old['source']);selected=[];rejected=[];audits=[]
        for folder in ['train','validation','provenance','failures','logs']:(root/folder).mkdir()
        for split,count in [('train',a.train_scenes),('validation',32)]:
            ordered=sorted(splits[split],key=lambda s:hashlib.sha256(('scene-token-pilot-256:'+s).encode()).hexdigest())
            accepted=0
            for s in ordered:
                if accepted==count:break
                try:info=pose_audit(source/s/'transforms.json')
                except Exception as e:
                    rejected.append(dict(scene=s,split=split,error=str(e)));continue
                selected.append(dict(scene=s,key=Path(s).name,frames=info['frames'],split=split,shard=f'shard-{accepted:06d}.tar'))
                audits.append(dict(scene=s,split=split,**info));accepted+=1
                if len(selected)%20==0:atomic(root/'progress.json',dict(state='auditing_source',done=len(selected),total=a.train_scenes+32,failed=len(rejected),updated=time.time()))
            assert accepted==count,'Insufficient acceptable scenes'
        keys={r['key'] for r in selected};assert len(keys)==len(selected) and not keys&set(splits['official_test'])
        previous_val=set(json.loads((a.reuse/'validation'/'index.json').read_text()))
        assert {r['key'] for r in selected if r['split']=='validation'}==previous_val,'Validation set changed'
        atomic(root/'source_audit.json',dict(accepted=audits,rejected=rejected,policy='Conservative finite/rotation/trajectory audit, not full semantic geometry certification'))
        manifest=dict(old,version='scene-token-expanded-252-v1',selected=selected,requested_train=a.train_scenes,requested_validation=32,reused_from=str(a.reuse))
        atomic(root/'manifest.json',manifest);atomic(root/'splits.json',splits)
        reused=0
        for record in selected:
            report=a.reuse/'provenance'/(record['key']+'.json')
            if not report.exists():continue
            info=json.loads(report.read_text());original=a.reuse/info['split']/info['shard']
            assert info['split']==record['split'] and digest(original)==info['tar_sha256']
            assert digest(source/record['scene']/'transforms.json')==info['source_transforms_sha256']
            dest=root/record['split']/record['shard']
            try:os.link(original,dest)
            except OSError:shutil.copy2(original,dest)
            atomic(root/'provenance'/report.name,dict(info,**record));reused+=1
        children=[]
        for rank in range(8):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
            with (root/'logs'/f'worker{rank}.log').open('a') as log:
                children.append(subprocess.Popen([sys.executable,'-u','tools/scene_token/prepare_training_run.py','--output',str(root),'--workers','8','--rank',str(rank)],stdout=log,stderr=subprocess.STDOUT,env=env))
        while any(c.poll() is None for c in children):
            atomic(root/'progress.json',dict(state='preparing',done=len(list((root/'provenance').glob('*.json'))),total=len(selected),reused=reused,failed=len(list((root/'failures').glob('*.json'))),updated=time.time(),pids=[c.pid for c in children]))
            time.sleep(10)
        assert all(c.returncode==0 for c in children),'Conversion worker failed; inspect logs'
        good=[];depth_rejected=[]
        for report in sorted((root/'provenance').glob('*.json')):
            r=json.loads(report.read_text());path=root/r['split']/r['shard']
            assert digest(path)==r['tar_sha256'],'Shard checksum changed'
            with tarfile.open(path) as tar:
                bounds=np.asarray(json.load(tar.extractfile(r['key']+'.depth_ranges.json')),dtype=np.float64)
                poses=np.load(io.BytesIO(tar.extractfile(r['key']+'.poses.npy').read()))
            center=poses[:,-3:];radius=np.linalg.norm(center-np.median(center,axis=0),axis=1)
            depth_ok=np.isfinite(bounds).all() and (bounds>0).all() and (bounds[:,1]>=bounds[:,0]).all() and bounds.max()<1e8 and bounds.max()/max(float(radius.max()),1e-8)<1e6
            if not depth_ok:depth_rejected.append(dict(scene=r['key'],split=r['split'],maximum=float(bounds.max())));continue
            good.append(r)
        assert not [r for r in depth_rejected if r['split']=='validation'],'Validation audit failed'
        counts={s:sum(r['split']==s for r in good) for s in ['train','validation']}
        assert counts['train']>=int(a.train_scenes*.97) and counts['validation']==32,counts
        for split in counts:atomic(root/split/'index.json',{r['key']:r['shard'] for r in good if r['split']==split})
        atomic(root/'depth_audit.json',dict(flagged=depth_rejected,passed=len(good),policy='finite/positive/bounds ordering, max depth<1e8 and depth/trajectory-radius<1e6; discarded shards retained outside index'))
        summary=dict(passed=True,state='ready',counts=counts,done=len(good),total=len(selected),reused=reused,failed=len(selected)-len(good),frames=sum(r['frames'] for r in good),tar_bytes=sum(r['tar_bytes'] for r in good),updated=time.time())
        atomic(root/'progress.json',summary);atomic(root/'summary.json',summary)
    except Exception:
        error=traceback.format_exc();atomic(root/'summary.json',dict(passed=False,state='failed',error=error,updated=time.time()));raise

if __name__=='__main__':main()
