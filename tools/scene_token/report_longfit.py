"""Aggregate continuation results and provide an offline image comparison page."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args(); root = args.input
    rows = []; curves = []; final = []; layers = []; gallery_metrics = {}
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for i, seed in enumerate([256,257,258]):
        for arm, color in [('scale1','#1971c2'),('scale4','#e67700')]:
            path = root/f'seed{seed}'/arm
            protocol = json.loads((path/'protocol.json').read_text())
            source = Path(protocol['source'])
            previous = json.loads((source/'evaluations.json').read_text())
            current = json.loads((path/'evaluations.json').read_text())
            assert [e['step'] for e in current] == [1000,2000,3000,5000]
            evals = previous[:-1] + current
            for ev in current:
                for split in ['train','validation']:
                    for scene, metrics in enumerate(ev[split]['scenes']):
                        gallery_metrics[f'{seed}/{arm}/{ev["step"]}/{split}/{scene}'] = {
                            m:metrics[m] for m in ['psnr','lpips']}
            trace = json.loads((path/'step5000/train_trace.json').read_text())
            for stage in ['da3.00.output','da3.39.output','fusion.2.output','color.output']:
                layers.append(dict(seed=seed,arm=arm,stage=stage,**trace['stages'][stage]['within_view_mean']))
            records = [json.loads(line) for p in [source/'steps.jsonl',path/'steps.jsonl'] for line in p.read_text().splitlines()]
            assert [r['step'] for r in records] == list(range(1,5001))
            axes[i,0].plot([r['step'] for r in records], [r['metrics']['loss/total'] for r in records],color=color,label=arm,alpha=.8)
            for split, col in [('train',1),('validation',2)]:
                axes[i,col].plot([e['step'] for e in evals], [e[split]['mean']['psnr'] for e in evals], color=color,marker='o',label=arm)
                for ev in evals:
                    rows.append(dict(seed=seed,arm=arm,step=ev['step'],split=split,
                        **{m:ev[split]['mean'][m] for m in ['total','psnr','lpips','activated_pct']}))
            final.append(dict(seed=seed,arm=arm,train_psnr=current[-1]['train']['mean']['psnr'],
                train_lpips=current[-1]['train']['mean']['lpips'],val_psnr=current[-1]['validation']['mean']['psnr'],
                val_lpips=current[-1]['validation']['mean']['lpips']))
        for col,title in enumerate(['Training loss','Fixed bookcase PSNR','Held-out PSNR']):
            axes[i,col].set_title(f'Seed {seed}: {title}'); axes[i,col].grid(alpha=.2)
            axes[i,col].axvline(1000,color='gray',linestyle='--',alpha=.6); axes[i,col].set_xlabel('Total updates')
    axes[0,0].legend(); fig.suptitle('Full-state continuation 1000 to5000; original DA3 frozen; same bookcase/data/LR')
    fig.tight_layout(); fig.savefig(root/'training_curves.png',dpi=150); plt.close(fig)
    write_csv(root/'quality.csv',rows); write_csv(root/'final_per_seed.csv',final); write_csv(root/'layer_metrics.csv',layers)
    for step in [200,1000,2000,3000,5000]:
        for arm in ['scale1','scale4']:
            for split in ['train','validation']:
                selected = [r for r in rows if r['step']==step and r['arm']==arm and r['split']==split]
                curves.append(dict(step=step,arm=arm,split=split,
                    **{f'{m}_{stat}':float(fn([r[m] for r in selected])) for m in ['psnr','lpips']
                       for stat,fn in [('mean',np.mean),('std',lambda x:np.std(x,ddof=1))]}))
    write_csv(root/'aggregate.csv',curves)
    result = dict(final=final,aggregate=curves,within_arm_improvements=[])
    for seed in [256,257,258]:
        for arm in ['scale1','scale4']:
            for split in ['train','validation']:
                before = next(r for r in rows if r['seed']==seed and r['arm']==arm and r['split']==split and r['step']==1000)
                after = next(r for r in rows if r['seed']==seed and r['arm']==arm and r['split']==split and r['step']==5000)
                result['within_arm_improvements'].append(dict(seed=seed,arm=arm,split=split,
                    psnr=after['psnr']-before['psnr'],lpips=after['lpips']-before['lpips']))
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    for seed in [256,257,258]:
        fig, axes = plt.subplots(2,5,figsize=(16,7))
        for i,arm in enumerate(['scale1','scale4']):
            for j,step in enumerate([1000,1000,2000,3000,5000]):
                tag = 'gt' if j==0 else 'render'
                axes[i,j].imshow(Image.open(root/f'seed{seed}'/arm/f'step{step:04d}'/'train'/f'after_0_0_{tag}.png'))
                axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
                if i==0: axes[i,j].set_title('GT' if j==0 else f'Step {step}')
                if j==0: axes[i,j].set_ylabel('std0.02' if arm=='scale1' else 'std0.08')
        fig.suptitle(f'Bookcase, seed {seed}; same target, both scales'); fig.tight_layout()
        fig.savefig(root/f'bookcase_seed{seed}_progression.png',dpi=150); plt.close(fig)
    for split,scene,label in [('train',0,'bookcase'),('validation',0,'validation0'),('validation',1,'validation1')]:
        fig, axes = plt.subplots(3,3,figsize=(11,11))
        for i,seed in enumerate([256,257,258]):
            for j,(arm,tag,title) in enumerate([('scale1','gt','GT'),('scale1','render','std0.02'),('scale4','render','std0.08')]):
                axes[i,j].imshow(Image.open(root/f'seed{seed}'/arm/'step5000'/split/f'after_{scene}_0_{tag}.png'))
                axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
                if i==0: axes[i,j].set_title(title)
                if j==0: axes[i,j].set_ylabel(f'Seed {seed}')
        fig.suptitle(f'{label}: step5000, all seeds'); fig.tight_layout(); fig.savefig(root/f'{label}_comparison.png',dpi=150); plt.close(fig)
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Scene token 长拟合对照</title>
<style>body{font:16px system-ui;margin:24px;background:#f4f5f7;color:#222}select{font:inherit;margin:8px;padding:5px}table{border-collapse:collapse;width:100%}td,th{padding:6px;text-align:center}img{width:100%;min-width:170px;image-rendering:auto}td{width:19%}small{color:#555}#wrap{overflow:auto}</style>
<h1>1000 → 5000 步重建对照</h1><p>固定书架训练；已有 DA3 冻结；每视角 256 个 scene token。验证为另外两个场景，未训练。</p>
<label>种子<select id="seed"><option>256</option><option>257</option><option>258</option></select></label>
<label>场景<select id="scene"><option value="train/0">训练书架</option><option value="validation/0">验证场景 1</option><option value="validation/1">验证场景 2</option></select></label>
<label>目标图<select id="target"></select></label><p><small>两行分别为初始化标准差 0.02 / 0.08。可点击图片查看原图；切换目标图检查所有监督/验证视角。图片下方指标为该场景全部目标图的平均值，并非单图指标。</small></p><div id="wrap"><table id="grid"></table></div>
<script>const quality=__GALLERY_METRICS__;const seed=document.getElementById('seed'),scene=document.getElementById('scene'),target=document.getElementById('target');
function refreshTargets(){target.innerHTML=Array.from({length:scene.value.startsWith('train')?4:8},(_,i)=>`<option>${i}</option>`).join('');render()}
function render(){let [split,n]=scene.value.split('/');let s=seed.value,t=target.value;let h='<tr><th>标准差</th><th>GT</th>'+[1000,2000,3000,5000].map(x=>`<th>${x} 步</th>`).join('')+'</tr>';
for(let [arm,std] of [['scale1','0.02'],['scale4','0.08']]){h+=`<tr><th>${std}</th>`;for(let step of [0,1000,2000,3000,5000]){let p=`seed${s}/${arm}/step${step||1000}/${split}/after_${n}_${t}_${step?'render':'gt'}.png`;let m=quality[`${s}/${arm}/${step}/${split}/${n}`];h+=`<td><a href="${p}" target="_blank"><img src="${p}"></a>`+(m?`<small>PSNR ${m.psnr.toFixed(2)} / LPIPS ${m.lpips.toFixed(4)}</small>`:'')+'</td>'}h+='</tr>'}document.getElementById('grid').innerHTML=h}
seed.onchange=render;scene.onchange=refreshTargets;target.onchange=render;refreshTargets();</script></html>'''
    (root/'comparison.html').write_text(html.replace('__GALLERY_METRICS__',json.dumps(gallery_metrics)),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
