"""Summarize the fixed, inference-only layer diagnosis without selecting runs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


LABELS=('initial_replayed','step100','step200','baseline')
COLORS=('#868e96','#e67700','#c2255c','#1971c2')


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args()
    data={(label,sample):json.loads((a.input/f'{label}_{sample}.json').read_text())
          for label in LABELS for sample in ('train0','val0')}
    rows=[]
    for (label,sample),run in data.items():
        for stage,value in run['stages'].items():
            for scope in ('within_view_mean','pooled'):
                if scope in value: rows.append(dict(model=label,sample=sample,stage=stage,scope=scope,**value[scope]))
    with (a.input/'stage_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for row,sample in enumerate(('train0','val0')):
        for label,color in zip(LABELS,COLORS):
            run=data[label,sample]
            seq=[run['stages'][f'da3.{i:02d}.output']['within_view_mean'] for i in range(40)]
            for col,(metric,title) in enumerate((('cosine','Within-view mean pairwise cosine'),
                                                ('effective_rank','Centered effective rank'),
                                                ('relative_variation','Centered / total feature norm'))):
                axes[row,col].plot(range(1,41),[s[metric] for s in seq],label=label,color=color)
                axes[row,col].set_title(title)
        # Patch tokens within the modified step200 model, as an internal control.
        run=data['step200',sample]
        for col,metric in enumerate(('cosine','effective_rank','relative_variation')):
            axes[row,col].plot(range(1,41),[run['stages'][f'da3.{i:02d}.patch_output']['within_view_mean'][metric] for i in range(40)],
                               color='#2b8a3e',linestyle='--',label='step200 retained patches')
            axes[row,col].axvline(run['alt_start']+1,color='gray',linestyle=':',alpha=.7)
            axes[row,col].grid(alpha=.2);axes[row,col].set_xlabel('DA3 block (1-based)')
            axes[row,col].set_ylabel('Train V2' if sample=='train0' else 'Held-out val V6')
            if metric=='effective_rank': axes[row,col].set_yscale('log')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('All 40 blocks: similarity, independent variation, and effective rank\nDotted line: first global-attention block; no optimizer updates during diagnosis')
    fig.tight_layout();fig.savefig(a.input/'da3_layers.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for row,sample in enumerate(('train0','val0')):
        run=data['step200',sample]
        for col,(metric,title) in enumerate((('cosine','Within-view cosine'),('relative_variation','Relative variation'))):
            for suffix,color in [('input','#868e96'),('after_attention','#e67700'),('output','#c2255c')]:
                axes[row,col].plot(range(1,41),[run['stages'][f'da3.{i:02d}.{suffix}']['within_view_mean'][metric] for i in range(40)],color=color,label=suffix)
            axes[row,col].set_title(f'{sample}: {title}');axes[row,col].grid(alpha=.2)
            axes[row,col].set_xlabel('DA3 block (1-based)')
    axes[0,0].legend();fig.suptitle('Step200: separate attention and MLP residuals')
    fig.tight_layout();fig.savefig(a.input/'attention_mlp_steps.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    stages=['prepare.0.output']+[f'fusion.{i}.{s}' for i in range(3) for s in ('after_ca','after_sa_attention','output')]+['color.query_input','color.after_attention','color.output']
    names=['Prepare39']+[f'{i+1}:{s}' for i in range(3) for s in ('CA','SA','MLP')]+['ColorQ','ColorAttn','ColorOut']
    for label,color in zip(LABELS,COLORS):
        run=data[label,'train0']
        for ax,metric,title in zip(axes,('cosine','effective_rank','relative_variation'),('Pooled cosine','Pooled centered effective rank','Pooled relative variation')):
            ax.plot(range(len(stages)),[run['stages'][s]['pooled'][metric] for s in stages],label=label,color=color,marker='.')
            ax.set_xticks(range(len(stages)),names,rotation=70,fontsize=8);ax.set_title(title);ax.grid(alpha=.2)
            if metric=='effective_rank':ax.set_yscale('log')
    axes[0].legend(fontsize=8);fig.suptitle('Train V2 downstream: all 512 slots pooled (baseline prepare has 648 patches)')
    fig.tight_layout();fig.savefig(a.input/'downstream_stages.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for row,sample in enumerate(('train0','val0')):
        for label,color in zip(LABELS,COLORS):
            seq=[data[label,sample]['attention'][f'da3.{i:02d}'] for i in range(40)]
            for col,metric in enumerate(('query_distribution_cosine','normalized_entropy')):
                axes[row,col].plot(range(1,41),[v[metric] for v in seq],color=color,label=label)
                axes[row,col].set_title(f'{sample}: {metric}');axes[row,col].grid(alpha=.2)
                axes[row,col].set_xlabel('DA3 block (1-based)')
    axes[0,0].legend(fontsize=8);fig.suptitle('First view: all queries, per-head attention probabilities; diagnostic recomputation')
    fig.tight_layout();fig.savefig(a.input/'attention_distributions.png',dpi=160);plt.close(fig)
    summary={}
    for (label,sample),run in data.items():
        transitions=[]
        for i in range(40):
            for before,after,part in [('input','after_attention','attention'),('after_attention','output','mlp')]:
                u=run['stages'][f'da3.{i:02d}.{before}']['within_view_mean']
                v=run['stages'][f'da3.{i:02d}.{after}']['within_view_mean']
                transitions.append(dict(block_zero_based=i,block_one_based=i+1,part=part,
                    cosine_increase=v['cosine']-u['cosine'],relative_variation_ratio=v['relative_variation']/max(u['relative_variation'],1e-30),
                    centered_rms_ratio=v['centered_rms']/max(u['centered_rms'],1e-30),
                    rank_before=u['effective_rank'],rank_after=v['effective_rank']))
        summary[f'{label}_{sample}']=dict(loss=run['loss'],psnr=run['psnr'],
            largest_cosine_increases=sorted(transitions,key=lambda r:r['cosine_increase'],reverse=True)[:8],
            initial_replay_loss_check=run.get('initial_replay_loss_check'),parity=run['hook_parity_max_abs'])
    (a.input/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
