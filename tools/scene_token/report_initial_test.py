"""Plot the bounded initial-training diagnostic, preserving validation limitations."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    result=json.loads((a.run/'summary.json').read_text())
    rows=[json.loads(line) for line in (a.run/'steps.jsonl').read_text().splitlines()]
    evaluations=json.loads((a.run/'evaluations.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].plot([r['step'] for r in rows],[r['metrics']['loss/total'] for r in rows],lw=1)
    axes[0].axvline(100,color='gray',linestyle='--');axes[0].set_title('Optimization loss (sample changes after 100)')
    for key,label in [('train','Fixed 8 training batches'),('validation','2 held-out scenes')]:
        axes[1].plot([r['step'] for r in evaluations],[r[key]['mean']['total'] for r in evaluations],'o-',label=label)
        axes[2].plot([r['step'] for r in evaluations],[r[key]['mean']['psnr'] for r in evaluations],'o-',label=label)
    axes[1].set_title('Fixed-data total loss');axes[1].legend(fontsize=8)
    axes[2].set_title('PSNR at fixed GT cameras');axes[2].set_ylabel('dB')
    for ax in axes:ax.set_xlabel('Optimizer step');ax.grid(alpha=.25)
    fig.suptitle('S=256 readiness test: V=2 training / V=6 validation; no pose refinement')
    fig.tight_layout();fig.savefig(a.run/'training_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(9,12))
    for row,(scene,target) in enumerate([(0,0),(0,4),(1,0),(1,4)]):
        for col,(phase,tag,title) in enumerate([('before','gt','Ground truth'),('before','render','Before training'),('after','render','After 200 steps')]):
            axes[row,col].imshow(Image.open(a.run/f'{phase}_{scene}_{target}_{tag}.png'))
            axes[row,col].axis('off')
            axes[row,col].set_title(f'{title}\nVal {scene+1}, target {target+1}',fontsize=10)
    fig.suptitle('Held-out validation; identical cameras, no alignment or target pose fitting')
    fig.tight_layout();fig.savefig(a.run/'validation_before_after.png',dpi=125);plt.close(fig)
    initial=result['initial'];single=result['after_single_batch'];final=result['final']
    def delta(before,after):return dict(before=before,after=after,relative_reduction=(before-after)/before)
    stats=dict(single_batch_loss=delta(initial['train']['scenes'][0]['total'],single['train']['scenes'][0]['total']),
               eight_batch_loss=delta(initial['train']['mean']['total'],final['train']['mean']['total']),
               validation_loss=delta(initial['validation']['mean']['total'],final['validation']['mean']['total']),
               validation_psnr_before=initial['validation']['mean']['psnr'],validation_psnr_after=final['validation']['mean']['psnr'],
               update_seconds_percentiles=dict(zip(['p10','p50','p90'],np.percentile([r['seconds'] for r in rows[10:]],[10,50,90]).tolist())),
               min_scene_token_gradient=min(r['gradients']['backbone.backbone.scene_tokens'] for r in rows),
               parameter_groups=json.loads((a.run/'optimizer_groups.json').read_text()))
    (a.run/'analysis.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps({k:v for k,v in stats.items() if k!='parameter_groups'},indent=2))


if __name__=='__main__':main()
