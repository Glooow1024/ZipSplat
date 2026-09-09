"""Report every predefined initial-scale arm, including quality failures."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args(); root = args.input
    protocol = json.loads((root/'protocol.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    names = list(protocol['factors']); colors = ['#868e96', '#e67700', '#1971c2', '#2f9e44']
    quality, layers, analysis = [], [], {}
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for name, color in zip(names, colors):
        rows = [json.loads(line) for line in (root/name/'steps.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows] == list(range(1, 201))
        axes[0].plot([r['step'] for r in rows], [r['metrics']['loss/total'] for r in rows], color=color, label=name)
        evaluations = json.loads((root/name/'evaluations.json').read_text())
        for ev in evaluations:
            for split in ['train', 'validation']:
                quality.append(dict(arm=name, step=ev['step'], split=split,
                    **{k:ev[split]['mean'][k] for k in ['total', 'psnr', 'lpips', 'activated_pct']}))
        for ax, split, metric in zip(axes[1:], ['train','validation','validation'], ['psnr','psnr','lpips']):
            ax.plot([ev['step'] for ev in evaluations], [ev[split]['mean'][metric] for ev in evaluations],
                    marker='o', color=color, label=name)
        final = summary[name]['final']
        analysis[name] = dict(initialization=summary[name]['initialization'],
            train_psnr=final['train']['mean']['psnr'], val_psnr=final['validation']['mean']['psnr'],
            train_lpips=final['train']['mean']['lpips'], val_lpips=final['validation']['mean']['lpips'])
    for ax, title in zip(axes, ['Fixed-batch loss', 'Train PSNR', 'Held-out PSNR', 'Held-out LPIPS (lower better)']):
        ax.set_title(title); ax.set_xlabel('Optimizer step'); ax.grid(alpha=.2)
    axes[0].legend(); fig.suptitle('Only initial scene amplitude changes; same Gaussian directions and distinct 1D RoPE')
    fig.tight_layout(); fig.savefig(root/'training_curves.png', dpi=160); plt.close(fig)
    write_csv(root/'quality.csv', quality)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, sample in enumerate(['train','val']):
        for name, color in zip(names, colors):
            for step, style in [(0, ':'), (200, '-')]:
                trace = json.loads((root/name/f'step{step:03d}'/f'{sample}_trace.json').read_text())
                for col, metric in enumerate(['cosine','effective_rank','relative_variation']):
                    ys = [trace['stages'][f'da3.{i:02d}.output']['within_view_mean'][metric] for i in range(40)]
                    axes[row,col].plot(range(1,41), ys, color=color, linestyle=style, label=f'{name} step{step}')
                    axes[row,col].set_title(f'{sample}: {metric}'); axes[row,col].grid(alpha=.2)
                    axes[row,col].set_xlabel('DA3 block (1-based)')
                    if metric == 'effective_rank': axes[row,col].set_yscale('log')
                for stage in ['initial_parameter','da3.00.after_attention','da3.00.output','da3.12.output',
                              'da3.19.output','da3.29.output','da3.39.output','fusion.2.output','color.output']:
                    layers.append(dict(arm=name, sample=sample, step=step, stage=stage,
                        **trace['stages'][stage]['within_view_mean']))
                if sample == 'train':
                    analysis[name][f'step{step}'] = dict(
                        attention_update_ratio=trace['updates']['da3.00.attention']['delta_over_input'],
                        mlp_update_ratio=trace['updates']['da3.00.mlp']['delta_over_input'],
                        first_attention=trace['attention']['da3.00'],
                        first_block=json.loads((root/name/f'step{step:03d}'/'first_block.json').read_text()))
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Slot diversity: initial (dotted), step200 (solid); effective rank is not scene information')
    fig.tight_layout(); fig.savefig(root/'layer_comparison.png', dpi=160); plt.close(fig)
    write_csv(root/'layer_metrics.csv', layers)

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    for row, (split, scene, target) in enumerate([('train',0,0),('train',0,2),('validation',0,0),('validation',1,0)]):
        for col, (name, tag) in enumerate([(names[0],'gt')]+[(n,'render') for n in names]):
            axes[row,col].imshow(Image.open(root/name/'step200'/split/f'after_{scene}_{target}_{tag}.png'))
            axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            if row == 0: axes[row,col].set_title('GT' if col == 0 else f'{name} (std {protocol["nominal_initial_std"][name]:.2f})')
            if col == 0: axes[row,col].set_ylabel(f'{split} {scene}, target {target}')
    fig.suptitle('Same inputs, target cameras and budget; 200 fixed-batch steps per arm')
    fig.tight_layout(); fig.savefig(root/'reconstruction_comparison.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    xs = [protocol['nominal_initial_std'][n] for n in names]
    axes[0].plot(xs, [analysis[n]['step0']['attention_update_ratio'] for n in names], marker='o')
    axes[0].set_title('First attention update / input')
    for stage, label in [('da3.00.after_attention','After attention'), ('da3.00.output','After full block')]:
        ys = [next(r['cosine'] for r in layers if r['arm']==n and r['sample']=='train' and r['step']==0 and r['stage']==stage) for n in names]
        axes[1].plot(xs, ys, marker='o', label=label)
    axes[1].set_title('First-layer cosine similarity'); axes[1].legend()
    for group in ['patch', 'scene']:
        axes[2].plot(xs, [analysis[n]['step0']['first_attention']['key_mass'][group] for n in names], marker='o', label=group)
    axes[2].set_title('First scene attention: key mass'); axes[2].legend()
    for kind in ['q', 'k']:
        for group, style in [('scene','-'), ('patch',':')]:
            ys = [analysis[n]['step0']['first_block']['groups'][group][f'{kind}_head_norm_rms'] for n in names]
            axes[3].plot(xs, ys, marker='o', linestyle=style, label=f'{group} {kind.upper()}')
    axes[3].set_title('Actual Q/K head norm'); axes[3].legend()
    for ax in axes:
        ax.set_xscale('log'); ax.set_xticks(xs, [f'{x:.2f}' for x in xs]); ax.minorticks_off()
        ax.set_xlabel('Initial Gaussian std'); ax.grid(alpha=.2)
    fig.suptitle('Before training: residual scale changes strongly; first attention routing changes little')
    fig.tight_layout(); fig.savefig(root/'first_block_scales.png', dpi=160); plt.close(fig)
    (root/'analysis.json').write_text(json.dumps(analysis, indent=2, allow_nan=False)+'\n')
    print(json.dumps({n:{k:v for k,v in analysis[n].items() if k in ['train_psnr','val_psnr','train_lpips','val_lpips']} for n in names}, indent=2))


if __name__ == '__main__': main()
