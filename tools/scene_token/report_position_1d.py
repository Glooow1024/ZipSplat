"""Report all predefined position arms, including failed quality outcomes."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args()
    modes=['none','shared','distinct'];colors=['#868e96','#e67700','#1971c2']
    summary=json.loads((a.input/'summary.json').read_text())
    rows=[]
    for mode in modes:
        for ev in json.loads((a.input/mode/'evaluations.json').read_text()):
            for split in ['train','validation']:
                rows.append(dict(mode=mode,step=ev['step'],split=split,
                    **{k:ev[split]['mean'][k] for k in ['total','psnr','lpips','activated_pct']}))
    with (a.input/'quality.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(1,4,figsize=(17,4))
    for mode,color in zip(modes,colors):
        steps=[json.loads(line) for line in (a.input/mode/'steps.jsonl').read_text().splitlines()]
        axes[0].plot([s['step'] for s in steps],[s['metrics']['loss/total'] for s in steps],label=mode,color=color)
        for ax,split,metric in zip(axes[1:],['train','validation','validation'],['psnr','psnr','lpips']):
            chosen=[r for r in rows if r['mode']==mode and r['split']==split]
            ax.plot([r['step'] for r in chosen],[r[metric] for r in chosen],label=mode,color=color,marker='o')
    for ax,title in zip(axes,['Fixed-batch training loss','Fixed training PSNR (dB)','Held-out mean PSNR (dB)','Held-out LPIPS (lower is better)']):
        ax.set_title(title);ax.set_xlabel('Optimizer step');ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('1D slot Q/K RoPE: identical initialization, 200 fixed-batch steps per arm')
    fig.tight_layout();fig.savefig(a.input/'training_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    trace_rows=[]
    for row,sample in enumerate(['train','val']):
        for mode,color in zip(modes,colors):
            for step,style in [(0,':'),(200,'-')]:
                trace=json.loads((a.input/mode/f'step{step:03d}'/f'{sample}_trace.json').read_text())
                for col,metric in enumerate(['cosine','effective_rank','relative_variation']):
                    ys=[trace['stages'][f'da3.{i:02d}.output']['within_view_mean'][metric] for i in range(40)]
                    axes[row,col].plot(range(1,41),ys,color=color,linestyle=style,label=f'{mode} step{step}')
                    axes[row,col].set_title(f'{sample}: {metric}');axes[row,col].set_xlabel('DA3 block (1-based)');axes[row,col].grid(alpha=.2)
                    if metric=='effective_rank':axes[row,col].set_yscale('log')
                for stage in ['initial_parameter','da3.00.after_attention','da3.00.output','da3.12.output','da3.19.output','da3.29.output','da3.39.output','fusion.2.output','color.output']:
                    stat=trace['stages'][stage]['within_view_mean']
                    trace_rows.append(dict(mode=mode,sample=sample,step=step,stage=stage,**stat))
    axes[0,0].legend(fontsize=7);fig.suptitle('Within-view slot diversity; dotted=initial, solid=step200; no 2D anchors')
    fig.tight_layout();fig.savefig(a.input/'layer_comparison.png',dpi=160);plt.close(fig)
    with (a.input/'layer_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(trace_rows[0]));w.writeheader();w.writerows(trace_rows)
    fig,axes=plt.subplots(4,4,figsize=(12,12))
    selections=[('train',0,0),('train',0,2),('validation',0,0),('validation',1,0)]
    for row,(split,scene,target) in enumerate(selections):
        for col,(mode,tag,title) in enumerate([('none','gt','GT'),('none','render','No position'),('shared','render','Shared position'),('distinct','render','Distinct 1D position')]):
            path=a.input/mode/'step200'/split/f'after_{scene}_{target}_{tag}.png'
            axes[row,col].imshow(Image.open(path));axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(title)
            if col==0:axes[row,col].set_ylabel(f'{split} {scene}, target {target}')
    fig.suptitle('Same inputs, target cameras and budget; after 200 fixed-batch steps')
    fig.tight_layout();fig.savefig(a.input/'reconstruction_comparison.png',dpi=150);plt.close(fig)
    result={}
    for mode in modes:
        final=summary[mode]['final']
        result[mode]=dict(train_psnr=final['train']['mean']['psnr'],val_psnr=final['validation']['mean']['psnr'],
            train_lpips=final['train']['mean']['lpips'],val_lpips=final['validation']['mean']['lpips'],
            train_loss=final['train']['mean']['total'],val_loss=final['validation']['mean']['total'],
            median_update_seconds=summary[mode]['median_update_seconds'])
    result['differences']=dict(distinct_minus_none_val=result['distinct']['val_psnr']-result['none']['val_psnr'],
        distinct_minus_none_train=result['distinct']['train_psnr']-result['none']['train_psnr'],
        distinct_minus_none_val_lpips=result['distinct']['val_lpips']-result['none']['val_lpips'],
        distinct_minus_shared_val=result['distinct']['val_psnr']-result['shared']['val_psnr'])
    (a.input/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
