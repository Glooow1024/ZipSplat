"""Summaries and offline gallery for the isolated-label 8-scene pilot."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SETS=['train_probe','trained_scene_v2','new_scene_v2','new_scene_v6']


def csv_write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args();root=a.input
    es=json.loads((root/'evaluations.json').read_text());assert [e['step'] for e in es]==[0,500,1000,2500,5000]
    steps=[json.loads(x) for x in (root/'steps.jsonl').read_text().splitlines()]
    rows=[];scene_rows=[];metrics={}
    for e in es:
        for name in SETS:
            rows.append(dict(step=e['step'],set=name,**{m:e[name]['mean'][m] for m in ['total','psnr','lpips','activated_pct']}))
            for i,r in enumerate(e[name]['scenes']):
                scene_rows.append(dict(step=e['step'],set=name,scene=r['scene'],**{m:r[m] for m in ['total','psnr','lpips']}))
                metrics[f'{e["step"]}/{name}/{i}']={m:r[m] for m in ['psnr','lpips']}
    csv_write(root/'quality.csv',rows);csv_write(root/'per_scene.csv',scene_rows)
    blocks=[]
    for end in [1000,2000,3000,4000,5000]:
        block=[r for r in steps if end-500<r['step']<=end];assert len(block)==500
        blocks.append(dict(first_step=end-499,last_step=end,**{m:float(np.mean([r[m] for r in block])) for m in ['loss','psnr','lpips']}))
    result=dict(final={name:es[-1][name] for name in SETS},progress=rows,dynamic_training_500step_means=blocks)
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    window=100
    for col,metric in enumerate(['loss','psnr','lpips']):
        y=np.array([r[metric] for r in steps]);axes[col].plot([r['step'] for r in steps],y,alpha=.2,color='#555')
        axes[col].plot(np.arange(window,len(y)+1),np.convolve(y,np.ones(window)/window,'valid'),color='#1971c2')
        axes[col].set_title(f'Dynamic training: {metric} (100-step mean)');axes[col].set_xlabel('Global updates');axes[col].grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'training_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for name in SETS:
        for col,metric in enumerate(['psnr','lpips']):
            axes[col].plot([e['step'] for e in es],[e[name]['mean'][metric] for e in es],marker='o',label=name)
            axes[col].set_title(metric);axes[col].set_xlabel('Global updates');axes[col].grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Fixed evaluation sets; equal scene weights; no target optimization')
    fig.tight_layout();fig.savefig(root/'evaluation_curves.png',dpi=150);plt.close(fig)
    for name in SETS:
        n=len(es[-1][name]['scenes']);fig,axes=plt.subplots(n,4,figsize=(12,3*n),squeeze=False)
        for i in range(n):
            for j,(step,tag) in enumerate([(0,'gt'),(0,'render'),(1000,'render'),(5000,'render')]):
                axes[i,j].imshow(Image.open(root/f'step{step:04d}'/name/f'after_{i}_0_{tag}.png'))
                axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
                if i==0:axes[i,j].set_title('GT' if tag=='gt' else f'Step {step}')
                if j==0:axes[i,j].set_ylabel(es[-1][name]['scenes'][i]['scene'].removeprefix('dl3dv-')[:8])
        fig.suptitle(name);fig.tight_layout();fig.savefig(root/f'{name}_comparison.png',dpi=120);plt.close(fig)
    conf=json.loads((root/'config.json').read_text());plan=json.loads((root/'plan.json').read_text());probes=json.loads((root/'train_probes.json').read_text())
    options={name:[r['scene'] for r in es[-1][name]['scenes']] for name in SETS}
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>8场景训练验证</title>
<style>body{font:16px system-ui;margin:24px;background:#f5f6f8;color:#222}select{font:inherit;padding:6px;margin:8px}table{width:100%;border-collapse:collapse}td,th{text-align:center;padding:6px}img{width:100%;min-width:180px}small{color:#555}#wrap{overflow:auto}</style>
<h1>8 场景训练：固定视角对照</h1><p>std0.02，S256，已有DA3冻结；8卡同步训练，每次更新8场景。5000次更新对应40000场景曝光。预留目标RGB也已从训练深度的生成输入中排除。</p>
<p><a href="../../bookcase_novel_20260908_v2/comparison.html">查看单书架模型：监督帧与新视角</a></p>
<label>评估集合<select id="set"><option value="train_probe">训练采样参考（8场景）</option><option value="trained_scene_v2">已训练场景：未监督视角 V2</option><option value="new_scene_v2">未训练场景 V2</option><option value="new_scene_v6">未训练场景 V6</option></select></label>
<label>场景<select id="scene"></select></label><label>目标图<select id="target"></select></label>
<p id="frames"></p><p><small>图下指标为所选场景全部目标图的平均值，PSNR越高越好、LPIPS越低越好。V2/V6使用同目标、同跨度、嵌套输入；高斯数分别16384/49152，属于视角数迁移检查。不是论文正式测试集。</small></p>
<div id="wrap"><table id="grid"></table></div><p><a href="evaluation_curves.png">评估曲线</a> · <a href="training_curves.png">训练曲线</a></p>
<script>const options=__OPTIONS__,quality=__METRICS__,plan=__PLAN__,probes=__PROBES__;
const set=document.getElementById('set'),scene=document.getElementById('scene'),target=document.getElementById('target');
function draw(){const name=set.value,i=scene.value,t=target.value;const key=options[name][i].replace('dl3dv-','');const e=name==='train_probe'?probes[key]:{context:plan.scenes[key][name==='new_scene_v6'?'context6':'context2'],target:plan.scenes[key].target};
document.getElementById('frames').textContent=`输入帧：${e.context.join(', ')}；当前目标帧：${e.target[t]}`;
let h='<tr>'+['GT','0步','500步','1000步','2500步','5000步'].map(x=>`<th>${x}</th>`).join('')+'</tr><tr>';
for(const step of [-1,0,500,1000,2500,5000]){const tag=step===-1?'gt':'render',s=String(Math.max(0,step)).padStart(4,'0');const p=`step${s}/${name}/after_${i}_${t}_${tag}.png`,m=quality[`${Math.max(0,step)}/${name}/${i}`];h+=`<td><a href="${p}" target="_blank"><img src="${p}"></a>`+(step>=0?`<br><small>PSNR ${m.psnr.toFixed(2)} / LPIPS ${m.lpips.toFixed(4)}</small>`:'')+'</td>'}h+='</tr>';document.getElementById('grid').innerHTML=h}
function targets(){const n=set.value==='train_probe'?4:8;target.innerHTML=Array.from({length:n},(_,i)=>`<option value="${i}">${i}</option>`).join('');draw()}
function scenes(){scene.innerHTML=options[set.value].map((x,i)=>`<option value="${i}">${i+1}: ${x.replace('dl3dv-','').slice(0,12)}</option>`).join('');targets()}
set.onchange=scenes;scene.onchange=targets;target.onchange=draw;scenes();</script></html>'''
    for token,obj in [('__OPTIONS__',options),('__METRICS__',metrics),('__PLAN__',plan),('__PROBES__',probes)]:html=html.replace(token,json.dumps(obj))
    (root/'comparison.html').write_text(html,encoding='utf-8')
    print(json.dumps({name:{m:es[-1][name]['mean'][m] for m in ['psnr','lpips']} for name in SETS}),flush=True)


if __name__=='__main__':main()
