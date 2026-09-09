"""Freeze a concrete 50k-step training configuration and source provenance."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.scene_token.training_monitor import atomic,publish

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--base-config',type=Path,required=True);p.add_argument('--depth-audit',type=Path);a=p.parse_args()
    a.run.mkdir(parents=True,exist_ok=False);old=json.loads(a.base_config.read_text())
    model=dict(old['model']);model.update(gaussian_head=dict(max_scale=.1,min_scale=1e-6,opacity_penalty_threshold=.01,location_penalty_range=10.,
        use_l1_loss=True,use_chamfer=True,detach_activated_gaussians=True,loss_on_context=False,mse_weight=1.,lpips_weight=.05,chamfer_weight=.1,depth_weight=.01))
    train=dict(old['train']);train.update(num_steps=50000,lr=3e-5,lr_scaling={'scene_tokens':10.,'scene_color_query':10.,'backbone':.1},compile=None,ddp_find_unused_parameters=True)
    data=dict(old['data']);data.update(dataset_dir=str(a.dataset),train_shard_dir=str(a.dataset/'train'),test_shard_dir=str(a.dataset/'validation'),train_batch_size=1,image_num_range=[2,6],random_reference_view=False)
    audit=json.loads(a.depth_audit.read_text()) if a.depth_audit else None
    data['excluded_scenes']=[r['scene'] for r in audit['flagged']] if audit else []
    if audit:atomic(a.run/'depth_audit.json',audit)
    conf=dict(version='scene-token-main-v1',seed=256,dataset=str(a.dataset),model=model,train=train,data=data,total_steps=50000,checkpoint_every=1000,eval_every=1000,
        milestones=[2500,10000,25000,50000],initial_std=.02,world_size=8,global_scene_batch=8,targets=4,max_total_loss=100.,
        stages=dict(warmup=dict(start_step=0,end_step=2500,lr_warmup=250,views=2),joint=dict(start_step=2500,end_step=50000,lr_warmup=500,unfreeze_from=30,view_interval=2000)),
        protocol='Fresh released full ZipSplat initialization + S256 std0.02 distinct1D; original train split expanded to256+32 before explicit catastrophic-depth exclusions; no target pose refinement; train probes are not held-out.',
        limitation='First medium-sized DL3DV-only training, not full9820scene or RE10K mixture/paper reproduction. V2->6 curriculum caps total GS at49152. Old pilots used max_scale1; this new run explicitly uses official0.1.',
        cache='Bounded per-rank CPU tar8/frame512 caches; synchronous official preprocessing, not claimed production DataLoader throughput.')
    atomic(a.run/'config.json',conf)
    files=[]
    for folder in ['splatfactory','zipsplat','tools/scene_token']:
        files += [p for p in Path(folder).rglob('*.py') if '__pycache__' not in str(p)]
    hashes={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for f in files:
        dest=a.run/'source'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
    weights=Path('/root/.cache/torch/hub/zipsplat/zipsplat-da3g-252p.tar');h=hashlib.sha256()
    with weights.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    atomic(a.run/'provenance.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),git_status=subprocess.check_output(['git','status','--short'],text=True),source_sha256=hashes,base_config=str(a.base_config),released_weights=dict(path=str(weights),bytes=weights.stat().st_size,sha256=h.hexdigest())))
    publish(a.run,dict(state='preparing_data',phase='data',step=0,total_steps=50000,updated=__import__('time').time(),message='正在扩充数据，完成后自动开始两阶段训练'))
if __name__=='__main__':main()
