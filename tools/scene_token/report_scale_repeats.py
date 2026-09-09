"""Paired per-seed results; show every run, never select the best seed."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args(); root=args.input
    seeds=[256,257,258]; names=['scale1','scale4']; colors=['#1971c2','#e67700']
    quality=[]; final=[]; layer_rows=[]
    fig,axes=plt.subplots(3,3,figsize=(14,11))
    for row,seed in enumerate(seeds):
        for name,color in zip(names,colors):
            arm=root/f'seed{seed}'/name
            records=[json.loads(x) for x in (arm/'steps.jsonl').read_text().splitlines()]
            axes[row,0].plot([r['step'] for r in records],[r['metrics']['loss/total'] for r in records],color=color,label=name)
            evaluations=json.loads((arm/'evaluations.json').read_text())
            for ev in evaluations:
                for split in ['train','validation']:
                    quality.append(dict(seed=seed,arm=name,step=ev['step'],split=split,
                        **{k:ev[split]['mean'][k] for k in ['total','psnr','lpips','activated_pct']}))
            for col,split in [(1,'train'),(2,'validation')]:
                axes[row,col].plot([e['step'] for e in evaluations],[e[split]['mean']['psnr'] for e in evaluations],color=color,marker='o',label=name)
            last=evaluations[-1]; assert last['step']==1000
            final.append(dict(seed=seed,arm=name,train_psnr=last['train']['mean']['psnr'],train_lpips=last['train']['mean']['lpips'],
                val_psnr=last['validation']['mean']['psnr'],val_lpips=last['validation']['mean']['lpips']))
            trace=json.loads((arm/'step1000/train_trace.json').read_text())
            for stage in ['da3.00.output','da3.39.output','fusion.2.output','color.output']:
                layer_rows.append(dict(seed=seed,arm=name,stage=stage,**trace['stages'][stage]['within_view_mean']))
        for col,title in enumerate(['Training loss','Fixed training PSNR','Held-out PSNR']):
            axes[row,col].set_title(f'Seed {seed}: {title}'); axes[row,col].grid(alpha=.2); axes[row,col].set_xlabel('Step')
    axes[0,0].legend(); fig.suptitle('Paired Gaussian seeds: std0.02 vs0.08, 1000 fixed-bookcase updates; original DA3 frozen')
    fig.tight_layout(); fig.savefig(root/'training_curves.png',dpi=150); plt.close(fig)
    write_csv(root/'quality.csv',quality); write_csv(root/'final_per_seed.csv',final); write_csv(root/'layer_metrics.csv',layer_rows)
    result={'per_seed':final,'aggregate':{},'paired_differences':[]}
    for name in names:
        rows=[r for r in final if r['arm']==name]; result['aggregate'][name]={}
        for metric in ['train_psnr','train_lpips','val_psnr','val_lpips']:
            values=[r[metric] for r in rows]
            result['aggregate'][name][metric]=dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)),min=min(values),max=max(values))
    for seed in seeds:
        a=next(r for r in final if r['seed']==seed and r['arm']=='scale1')
        b=next(r for r in final if r['seed']==seed and r['arm']=='scale4')
        result['paired_differences'].append(dict(seed=seed,**{k:b[k]-a[k] for k in ['train_psnr','train_lpips','val_psnr','val_lpips']}))
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    for split,scene,target,label in [('train',0,0,'bookcase'),('validation',0,0,'validation0'),('validation',1,0,'validation1')]:
        fig,axes=plt.subplots(3,3,figsize=(11,11))
        for row,seed in enumerate(seeds):
            for col,(name,tag,title) in enumerate([('scale1','gt','GT'),('scale1','render','std0.02'),('scale4','render','std0.08')]):
                path=root/f'seed{seed}'/name/'step1000'/split/f'after_{scene}_{target}_{tag}.png'
                axes[row,col].imshow(Image.open(path)); axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
                if row==0: axes[row,col].set_title(title)
                if col==0: axes[row,col].set_ylabel(f'Seed {seed}')
        fig.suptitle(f'{label}: all seeds, same cameras and input; after1000 updates')
        fig.tight_layout(); fig.savefig(root/f'{label}_comparison.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(14,7))
    for row,name in enumerate(names):
        for col,step in enumerate([0,200,500,1000]):
            path=root/'seed256'/name/f'step{step:04d}'/'train/after_0_0_render.png'
            axes[row,col].imshow(Image.open(path)); axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            if row==0: axes[row,col].set_title(f'Step {step}')
            if col==0: axes[row,col].set_ylabel(name)
    fig.suptitle('Seed256 progression (preselected seed, not best-of-three)')
    fig.tight_layout(); fig.savefig(root/'bookcase_progression.png',dpi=150); plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
