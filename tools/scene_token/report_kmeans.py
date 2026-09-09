"""Aggregate a sweep without conflating fixed-camera and independently fit results."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def dump_csv(path, rows):
    if not rows: return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def mean(values): return statistics.mean(values)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    results=[json.loads(p.read_text()) for p in sorted(args.root.glob('*/*/*/result.json'))]
    assert len({(r['scene'],r['views'],r['label']) for r in results})==len(results)
    assert len({r['signature'] for r in results})<=1
    rows=[]; per_view=[]
    for r in results:
        for protocol in ['A','B']:
            for split in ['context','target']:
                common={'scene':r['scene'],'views':r['views'],'label':r['label'],'ratio':r['ratio'],
                    'K':r['K'],'gaussians':r['gaussians'],'unique_indices':r['unique_indices'],
                    'protocol':protocol,'split':split}
                rows.append(dict(common,**r['quality'][protocol][split]['mean'],
                    forward_ms=1000*r['performance']['median_seconds'],
                    alignment_seconds=r['timing']['alignment'],
                    render_ms=1000*r['quality'][protocol][split]['render_seconds'],
                    **{k+'_ms':1000*v for k,v in r['performance']['cached_stage_medians'].items()}))
                per_view.extend(dict(common,**v) for v in r['quality'][protocol][split]['per_view'])
    dump_csv(args.output/'per_experiment.csv',rows); dump_csv(args.output/'per_view.csv',per_view)
    aggregates=[]
    for protocol in ['A','B']:
        for split in ['context','target']:
            for label in sorted({r['label'] for r in results if r['label'].startswith('r')},key=lambda s:-float(s[1:])):
                subset=[r for r in rows if r['protocol']==protocol and r['split']==split and r['label']==label]
                if not subset: continue
                # Equal V counts per scene are required for an overall mean to be comparable.
                aggregates.append({'protocol':protocol,'split':split,'label':label,'ratio':subset[0]['ratio'],
                    'experiments':len(subset),'scenes':len({r['scene'] for r in subset}),
                    **{k:mean(r[k] for r in subset) for k in ['psnr','ssim','lpips','K','gaussians','forward_ms']}})
    dump_csv(args.output/'aggregate.csv',aggregates)
    by_views=[]
    for v in [4,8,12,16]:
        for label in sorted({r['label'] for r in results if r['label'].startswith('r')},key=lambda s:-float(s[1:])):
            subset=[r for r in rows if r['views']==v and r['protocol']=='B' and r['split']=='target' and r['label']==label]
            if subset:
                by_views.append({'views':v,'label':label,'ratio':subset[0]['ratio'],'scenes':len(subset),
                    **{k:mean(r[k] for r in subset) for k in ['psnr','ssim','lpips','K','gaussians','forward_ms']}})
    dump_csv(args.output/'by_views.csv',by_views)
    groups={(r['scene'],r['views']):r for r in results if r['label']=='r1'}
    thresholds=[]
    for key,baseline in groups.items():
        variants=[r for r in results if (r['scene'],r['views'])==key]
        for protocol in ['A','B']:
            for split in ['context','target']:
                ref=baseline['quality'][protocol][split]['mean']
                for loss in [.5,1.,2.]:
                    allowed=[r for r in variants if ref['psnr']-r['quality'][protocol][split]['mean']['psnr']<=loss]
                    best=min(allowed,key=lambda r:r['K'])
                    q=best['quality'][protocol][split]['mean']
                    thresholds.append({'scene':key[0],'views':key[1],'protocol':protocol,'split':split,
                        'max_psnr_loss_db':loss,'minimum_tested_K':best['K'],'equivalent_S':best['K']/key[1],
                        'label':best['label'],'psnr_loss':ref['psnr']-q['psnr'],
                        'ssim_loss':ref['ssim']-q['ssim'],'lpips_increase':q['lpips']-ref['lpips']})
    dump_csv(args.output/'quality_budgets.csv',thresholds)
    fixed=[]
    for K in [324,648]:
        for v in [4,8,12,16]:
            candidates=[r for r in results if r['views']==v and r['K']==K]
            by_scene={r['scene']:r for r in candidates}
            if by_scene:
                fixed.append({'K':K,'views':v,'scenes':len(by_scene),
                    **{k:mean(r['quality']['B']['target']['mean'][k] for r in by_scene.values()) for k in ['psnr','ssim','lpips']}})
    dump_csv(args.output/'fixed_k.csv',fixed)
    if aggregates:
        fig,axs=plt.subplots(1,3,figsize=(13,3.7))
        for protocol in ['A','B']:
            for split in ['context','target']:
                rr=[r for r in aggregates if r['protocol']==protocol and r['split']==split]
                for ax,metric in zip(axs,['psnr','ssim','lpips']):
                    ax.plot([r['ratio'] for r in rr],[r[metric] for r in rr],marker='o',label=protocol+' '+split)
                    ax.set_xscale('log',base=2); ax.set_xlabel('Query retention ratio');ax.set_ylabel(metric.upper());ax.grid(alpha=.25)
        axs[0].legend(fontsize=8); fig.tight_layout();fig.savefig(args.output/'quality_curves.png',dpi=180);plt.close(fig)
        fig,axs=plt.subplots(3,3,figsize=(12,10))
        for ax,scene in zip(axs.flat,sorted({r['scene'] for r in results})):
            for v in [4,8,12,16]:
                rr=sorted([r for r in results if r['scene']==scene and r['views']==v and r['label'].startswith('r')],key=lambda r:r['ratio'])
                if rr: ax.plot([r['K'] for r in rr],[r['quality']['B']['target']['mean']['psnr'] for r in rr],marker='.',label=str(v)+' views')
            ax.set_title('Scene '+scene[:2]);ax.set_xscale('log',base=2);ax.set_xlabel('K');ax.set_ylabel('Target PSNR');ax.grid(alpha=.25);ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(args.output/'per_scene.png',dpi=160);plt.close(fig)
    mainlabels=['r1','r0.5','r0.25','r0.125','r0.0625','r0.03125']
    manifest=json.loads((args.root/'manifest.json').read_text())
    for views in [8,16]:
        scenes=[s for s in manifest['scenes'] if s['scene'][:2] in ['04','05','09']]
        board=Image.new('RGB',(5*252,len(scenes)*280),'white');draw=ImageDraw.Draw(board)
        for i,s in enumerate(scenes):
            name=s['targets'][0];base=args.root/s['scene']/f'{views}views'
            choices=[('GT',base/'reference'/name)]+[(label,base/label/'B/target'/name) for label in ['r1','r0.5','r0.125','r0.03125']]
            for j,(label,path) in enumerate(choices):
                draw.text((j*252+4,i*280+5),s['scene'][:2]+' '+label,fill='black')
                if path.exists():
                    with Image.open(path) as im: board.paste(im.convert('RGB'),(j*252,i*280+25))
        board.save(args.output/f'pilot_comparison_{views}views.jpg')
    expected={(s['scene'],v,label) for s in manifest['scenes'] for v in manifest['views'] for label in mainlabels}
    found={(r['scene'],r['views'],r['label']) for r in results}
    missing=sorted(expected-found)
    if not missing:
        coarse=[r for r in aggregates if r['protocol']=='B' and r['split']=='target' and r['label'] in mainlabels]
        coarse.sort(key=lambda r:-r['ratio']);ref=coarse[0]['psnr'];fine=set();crossings=[]
        for threshold in [.5,1.,2.]:
            crossing=None
            for a,b in zip(coarse,coarse[1:]):
                if ref-a['psnr']<=threshold<ref-b['psnr']:
                    midpoint=(a['ratio']+b['ratio'])/2;fine.add(midpoint)
                    crossing={'threshold_db':threshold,'interval':[a['ratio'],b['ratio']],'midpoint':midpoint};break
            crossings.append(crossing or {'threshold_db':threshold,'interval':None})
        (args.output/'fine_plan.json').write_text(json.dumps({'rule':'first coarse aggregate B-target crossing of 0.5/1/2dB; arithmetic midpoint; all9scenes/all4viewcounts',
            'ratios':sorted(fine,reverse=True),'crossings':crossings},indent=2))
    (args.output/'coverage.json').write_text(json.dumps({'coarse_expected':216,'coarse_complete':len(expected&found),
        'all_results':len(results),'missing':missing,'fixed_k':fixed},indent=2))
    lines=['# ZipSplat k-means 预实验报告','',f'结果目录：`{args.root}`。',
        f'粗扫完成 {len(expected&found)}/216，全部结果 {len(results)} 组。未完整覆盖时各行不能作为正式配对比较。','',
        '原始权重、FP32（关闭TF32）、252×252、LPIPS-Alex v0.1；高斯固定，每query生成32个高斯。',
        'A：r=1的context共享Sim(3)跨预算复用；B：每个预算独立在context拟合相同120步共享Sim(3)。',
        'target RGB从未进入重建或相机拟合。使用针孔近似，没有畸变矫正；共享配准不包含逐target位姿优化。',
        '推理时间是3次warmup后10次完整模型前向中位数（输入已预处理/在GPU），不含磁盘、裁剪、指标或配准。',
        '缓存阶段计时独立热身；显存记录包含进程常驻质量特征，另有本次前向增量峰值，不能冒充独立部署峰值。','',
        '## 主表：B held-out target','',
        '| 保留率 | 组数 | PSNR | SSIM | LPIPS | 平均GS | 模型前向ms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in aggregates:
        if r['protocol']=='B' and r['split']=='target':
            lines.append(f"| {r['ratio']:.5g} | {r['experiments']} | {r['psnr']:.3f} | {r['ssim']:.4f} | {r['lpips']:.4f} | {r['gaussians']:.0f} | {r['forward_ms']:.2f} |")
    lines+=['','![质量曲线](quality_curves.png)','','![逐场景曲线](per_scene.png)','',
        '## 解读边界','',
        '- 本实验压缩query/高斯预算，几何和颜色KV仍然完整，不能当作场景token编码容量的极限。',
        '- 最小K仅指实测候选里满足PSNR分档者，SSIM/LPIPS和局部缺陷同时查看，不是理论最优。',
        '- k-means初始化为均匀linspace；GPU浮点归约仍可能非确定，冒烟单独验证重复一致性。',
        '- 固定K对照见fixed_k.csv；A与B以及context与target均分别保存，不能混合均值。',
        '- 该9场景用于开发选参，最终泛化仍需未参与选参的场景。','',
        '明细：per_experiment.csv、per_view.csv、quality_budgets.csv；覆盖：coverage.json。']
    if not missing:
        main=[r for r in aggregates if r['protocol']=='B' and r['split']=='target' and r['label'] in mainlabels]
        main.sort(key=lambda r:-r['ratio']);baseline=main[0]
        lines+=['','## 粗扫量化结论','',
            '| 保留率 | PSNR损失dB | LPIPS增量 | 高斯缩减倍数 | 模型前向加速倍数 |','|---|---:|---:|---:|---:|']
        for r in main:
            lines.append(f"| {r['ratio']:.5g} | {baseline['psnr']-r['psnr']:.3f} | {r['lpips']-baseline['lpips']:.4f} | {baseline['gaussians']/r['gaussians']:.2f} | {baseline['forward_ms']/r['forward_ms']:.3f} |")
        lines+=['','这些加速数仅比较模型前向，不含CPU预处理；不能将32倍高斯缩减解释为32倍编码加速。',
                '非单调的局部结果保留原值；预算分档按实测点选择，不做单调平滑。']
        lines+=['','## k-means 索引重复检查','',
            '| 保留率 | 平均唯一索引占K比例 | 最低唯一索引比例 |','|---|---:|---:|']
        for label in mainlabels:
            rr=[r for r in results if r['label']==label]
            fractions=[r['unique_indices']/r['K'] for r in rr]
            lines.append(f"| {label[1:]} | {mean(fractions):.4f} | {min(fractions):.4f} |")
        lines+=['','原算法返回各中心最近的原token，不强制这些索引互异。此处仅审计、不修改选择算法；名义K不能自动当作K个不同源特征。']
    prof=args.root/'standalone_profile.json'
    if prof.exists():
        profile=json.loads(prof.read_text());pr=profile['rows'];prows=[]
        for v in [4,8,12,16]:
            for ratio in [1,.5,.125,.03125]:
                rr=[r for r in pr if r['views']==v and r['ratio']==ratio]
                if not rr:continue
                prows.append({'views':v,'ratio':ratio,'scenes':len(rr),
                    'cpu_rgb_to_gaussians_ms':mean(r['median_seconds'] for r in rr)*1000,
                    'peak_GiB':mean(r['peak_bytes'] for r in rr)/2**30,
                    **{k+'_ms':mean(r['stage_medians'][k] for r in rr)*1000 for k in rr[0]['stage_medians']}})
        dump_csv(args.output/'standalone_profile.csv',prows)
        lines+=['','## 独立完整管线性能','',f"已完成 {len(pr)}/32 个代表配置，涵盖04/05两个场景。CPU RGB已加载，计入裁剪缩放和H2D，排除磁盘、配准和指标。",'',
            '| 视角 | 保留率 | 场景数 | CPU RGB到高斯ms | 预处理ms | backbone ms | 峰值GiB |','|---|---|---:|---:|---:|---:|---:|']
        for r in prows:
            lines.append(f"| {r['views']} | {r['ratio']} | {r['scenes']} | {r['cpu_rgb_to_gaussians_ms']:.1f} | {r['preprocessing_cpu_and_h2d_ms']:.1f} | {r['_backbone_features_ms']:.1f} | {r['peak_GiB']:.2f} |")
        lines+=['','阶段耗时由另一次带同步钩子的运行测量，因此各阶段和不要求等于无钩子的端到端值。CPU预处理同时受共享主机负载影响。']
    (args.output/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'coarse_complete':len(expected&found),'all_results':len(results),'main':[r for r in aggregates if r['protocol']=='B' and r['split']=='target']},indent=2))


if __name__=='__main__':main()
