"""Display supervised, interpolation and extrapolation frames without training."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args();root=a.input
    protocol=json.loads((root/'protocol.json').read_text());results=json.loads((root/'results.json').read_text())
    fig,axes=plt.subplots(3,6,figsize=(18,9))
    sets=['supervised','unseen_interpolation','unseen_extrapolation']
    for row,seed in enumerate([256,257,258]):
        for k,name in enumerate(sets):
            for j,tag in enumerate(['gt','render']):
                ax=axes[row,k*2+j];ax.imshow(Image.open(root/f'seed{seed}'/name/f'after_0_0_{tag}.png'))
                ax.set_xticks([]);ax.set_yticks([])
                if row==0:ax.set_title(f'{name.replace("unseen_","")}\nframe{protocol["sets"][name][0]} {tag}')
                if k==0 and j==0:ax.set_ylabel(f'Seed {seed}')
    fig.suptitle('std0.02, step5000, context186/194; three distinct target sets with their own GT')
    fig.tight_layout();fig.savefig(root/'novel_view_comparison.png',dpi=150);plt.close(fig)
    payload={'protocol':protocol,'results':results}
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>书架新视角检查</title>
<style>body{font:16px system-ui;margin:24px;background:#f5f6f8}select{font:inherit;padding:6px;margin:8px}.pair{display:flex;gap:16px}.pair>div{width:45%;max-width:600px}img{width:100%}</style>
<h1>书架：未参与训练的视角</h1><p>固定输入帧186/194，std0.02、5000步。插值/外推按帧序划分，不保证远处内容仍被输入观察到。所选新视角RGB也未参与训练帧的伪深度生成。</p>
<select id="seed"><option>256</option><option>257</option><option>258</option></select>
<select id="group"><option value="supervised">监督帧</option><option value="unseen_interpolation">未监督插值帧</option><option value="unseen_extrapolation">未监督范围外帧</option></select>
<select id="frame"></select><p id="metrics"></p><div class="pair"><div>GT<img id="gt"></div><div>重建<img id="render"></div></div>
<script>const data=__DATA__;const seed=document.getElementById('seed'),group=document.getElementById('group'),frame=document.getElementById('frame');
function draw(){for(const tag of ['gt','render'])document.getElementById(tag).src=`seed${seed.value}/${group.value}/after_0_${frame.value}_${tag}.png`;const m=data.results[seed.value][group.value].mean;document.getElementById('metrics').textContent=`该帧组平均：PSNR ${m.psnr.toFixed(2)}，LPIPS ${m.lpips.toFixed(4)}（非单图指标）`}
function populate(){frame.innerHTML=data.protocol.sets[group.value].map((x,i)=>`<option value="${i}">帧 ${x}</option>`).join('');draw()}
seed.onchange=draw;group.onchange=populate;frame.onchange=draw;populate();</script></html>'''
    (root/'comparison.html').write_text(html.replace('__DATA__',json.dumps(payload)),encoding='utf-8')


if __name__=='__main__':main()
