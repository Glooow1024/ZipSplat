"""Freeze a new 100k-update stage from the completed 50k checkpoint."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.scene_token.training_monitor import atomic,publish
from tools.scene_token.prepare_continuation_data import digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--dataset',type=Path,required=True);a=ap.parse_args()
    parent=a.parent.resolve();root=a.run.resolve();old=json.loads((parent/'config.json').read_text());status=json.loads((parent/'status.json').read_text())
    assert status['state']=='complete' and status['step']==50000
    checkpoint=parent/'checkpoints/step_050000.pt';assert checkpoint.is_file()
    root.mkdir(parents=True,exist_ok=False)
    conf=json.loads(json.dumps(old));conf.update(version='scene-token-continuation-v1',dataset=str(a.dataset),total_steps=150000,
        first_check_step=50050,checkpoint_every=2000,eval_every=2000,milestones=[50050,75000,100000,125000,150000],
        best_validation_set='val_v6',eval_train_v6=True,
        stages={'joint':dict(start_step=50000,end_step=150000,lr_warmup=1000,unfreeze_from=18,views=6,lr_floor_factor=.1)})
    conf['train']['num_steps']=150000
    # Specific substrings precede the broad backbone filter (first-match policy).
    new_blocks='+'.join(f'backbone.backbone.blocks.{i}.' for i in range(18,30))
    conf['train']['lr_scaling']={'scene_tokens':10/3,'scene_color_query':10/3,new_blocks:1/30,'backbone':.1}
    conf['data'].update(dataset_dir=str(a.dataset),train_shard_dir=str(a.dataset/'train'),test_shard_dir=str(a.dataset/'validation'),excluded_scenes=[])
    conf['continuation']=dict(run=str(parent),checkpoint=str(checkpoint),parent_step=50000,parent_config_sha256=digest(parent/'config.json'),checkpoint_sha256=digest(checkpoint),
        policy='Strict model restore; migrate all AdamW moments by parameter name; initialize moments only for newly unfrozen blocks; restore scaler/RNG; restart stage scheduler')
    evaluation=next(e for e in json.loads((parent/'evaluations.json').read_text()) if e['step']==50000)
    conf['train_probe_scenes']=[r['scene'] for r in evaluation['sets']['train_probe']['scenes']]
    conf['protocol']='Continue main_v2 50k weights/moments; 1024 requested train + unchanged32 val; S256, original std0.02, distinct1D,252px,V6,4targets/global8,full patch+scene global attention; limited unfreeze18-39; no target camera refinement.'
    conf['limitation']='Combined next-stage training, not a causal single-variable ablation; conservative geometry audit does not guarantee perfect labels; no RE10K or full9820 training.'
    atomic(root/'config.json',conf)
    files=[p for folder in ['splatfactory','zipsplat','tools/scene_token'] for p in Path(folder).rglob('*.py') if '__pycache__' not in str(p)]
    hashes={p.as_posix():digest(p) for p in files}
    for p in files:
        dest=root/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    atomic(root/'provenance.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),git_status=subprocess.check_output(['git','status','--short'],text=True),source_sha256=hashes,parent=str(parent),checkpoint_sha256=conf['continuation']['checkpoint_sha256']))
    publish(root,dict(state='preparing_data',phase='data',step=50000,total_steps=150000,updated=time.time(),message='扩充数据至1024场景；审计通过后从50k继续，先自动检查恢复与首50步保存'))
    page=(root/'index.html').read_text()
    page=page.replace('<option value="val_v2">验证 V2</option><option value="val_v6">验证 V6</option>','<option value="val_v6">验证 V6</option><option value="val_v2">验证 V2</option>')
    page=page.replace('<option value="train_probe">训练参考 V2</option>','<option value="train_probe">训练参考 V2</option><option value="train_probe_v6">训练参考 V6</option>')
    page=page.replace('首 50 步及每 1000 步','本阶段首 50 步（累计 50050）及每 2000 步').replace('每 1000 步评估','每 2000 步评估').replace('验证最佳 checkpoint','V6 验证最佳 checkpoint')
    page=page.replace('<div class="grid">','<div class="card">从累计 50k 续训至 150k；固定 V6，解冻18–39层；1000步LR热身，余弦降至各组峰值的10%。数据、解冻和调度均改变，本阶段不用于单变量因果比较。<br><a href="http://127.0.0.1:18767/">查看上一轮 0–50k 完整结果</a></div><div class="grid">',1)
    (root/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(run=str(root),checkpoint=str(checkpoint),source_files=len(files))))

if __name__=='__main__':main()
