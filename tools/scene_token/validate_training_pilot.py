"""Read every pilot frame, exercise real loaders, and optionally full loss backward."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from omegaconf import OmegaConf
from splatfactory.datasets import get_dataset
from splatfactory.datasets.utils.io import iter_scenes_from_tar, decode_image, decode_depth
from splatfactory.utils.mappings import batch_to_device


def check_batch(batch, views, targets, heldout=False):
    c, t = batch['context'], batch['target']
    assert list(c['image'].shape) == [1, views, 3, 252, 252]
    assert list(t['image'].shape) == [1, targets, 3, 252, 252]
    for view in [c, t]:
        for key in ['image', 'depth', 'near', 'far']:
            assert torch.isfinite(view[key]).all(), key
        assert torch.isfinite(view['camera'].data_).all()
        assert torch.isfinite(view['pose'].data_).all()
        assert view['depth_mask'].float().mean() > .99
        assert (view['depth'][view['depth_mask']] > 0).all()
    torch.testing.assert_close(c['pose'].R[0, 0], torch.eye(3), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(c['pose'].t[0, 0], torch.zeros(3), atol=1e-5, rtol=1e-5)
    ci, ti = set(c['index'].flatten().tolist()), set(t['index'].flatten().tolist())
    if heldout:
        assert not ci & ti
    return {'scene': batch['name'][0], 'context': sorted(ci), 'target': sorted(ti),
            'overlap': len(ci & ti), 'depth_valid_fraction': float(t['depth_mask'].float().mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=Path('splatfactory/configs/data/dl3dv_scene_token_pilot.yaml'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backward', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(256)
    np.random.seed(256)
    random.seed(256)
    conf = OmegaConf.load(args.config)
    root = Path(conf.dataset_dir)
    manifest = json.loads((root/'manifest.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    assert summary['complete']
    splits = json.loads((root/'splits.json').read_text())
    assert not set(splits['train']) & set(splits['validation'])
    assert not {Path(s).name for s in splits['train']+splits['validation']} & set(splits['official_test'])
    report = {'tar_checks': [], 'loader_checks': [], 'model_backward': None}
    previews = []
    for r in manifest['selected']:
        scenes = list(iter_scenes_from_tar(str(root/r['split']/r['shard'])))
        assert len(scenes) == 1
        s = scenes[0]
        assert s['key'] == r['key'] and s['meta']['has_depth']
        n = s['meta']['num_views']
        assert n == r['frames'] and set(s['images']) == set(s['depths']) == set(range(n))
        assert len(s['depth_ranges']) == n
        assert s['cameras'].shape == (n, 6) and s['poses'].shape == (n, 12)
        assert np.isfinite(s['cameras']).all() and np.isfinite(s['poses']).all()
        assert (s['cameras'][:, :2] == 252).all()
        valid = []
        for i in range(n):
            image = decode_image(s['images'][i])
            depth = decode_depth(s['depths'][i], *s['depth_ranges'][i])
            assert image.shape == (252, 252, 3) and depth.shape == (252, 252)
            assert np.isfinite(depth).all()
            valid.append(float((depth > 0).mean()))
            assert valid[-1] > .99
            if i == n//2:
                previews.append((r['split']+': '+r['key'][:8], image, depth))
        steps = np.linalg.norm(np.diff(s['poses'][:, 9:12], axis=0), axis=1)
        report['tar_checks'].append({'scene': r['key'], 'frames': n, 'min_valid_depth': min(valid),
                                     'full_trajectory_max_median_step_ratio': float(steps.max()/np.median(steps)),
                                     'largest_step_index': int(steps.argmax())})
    # Batch size exactly one, all scenes, all four representative context counts.
    for views in [2, 8, 16, 24]:
        cfg = OmegaConf.merge(conf, {'image_num_range': [views, views], 'train_batch_size': views})
        ds = get_dataset(cfg.name)(cfg, split='train')
        rows = [check_batch(b, views, 4) for b in ds.get_loader(num_workers=0)]
        expected = {'dl3dv-'+r['key'] for r in manifest['selected'] if r['split']=='train'}
        assert {r['scene'] for r in rows} == expected, ('Skipped train scenes', views, rows)
        report['loader_checks'].append({'split': 'train', 'views': views, 'workers': 0, 'batches': rows})
    # Multiprocessing must work with packed Camera/Pose objects as well.
    cfg = OmegaConf.merge(conf, {'image_num_range': [2, 2], 'train_batch_size': 2})
    ds = get_dataset(cfg.name)(cfg, split='train')
    rows = [check_batch(b, 2, 4) for b in ds.get_loader(num_workers=2)]
    assert len(rows) == 8 and len({r['scene'] for r in rows}) == 8
    report['loader_checks'].append({'split': 'train', 'views': 2, 'workers': 2, 'batches': rows})
    # Identical override merge to the training entry point.
    val_cfg = OmegaConf.merge(conf, conf.val_overrides)
    val_ds = get_dataset(val_cfg.name)(val_cfg, split='val')
    val_rows = [check_batch(b, 6, 8, heldout=True) for b in val_ds.get_loader(num_workers=0)]
    assert len(val_rows) == 2
    repeated = [check_batch(b, 6, 8, heldout=True) for b in val_ds.get_loader(num_workers=0)]
    assert val_rows == repeated
    report['loader_checks'].append({'split': 'val', 'views': 6, 'targets': 8, 'batches': val_rows, 'repeat_identical': True})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.backward:
        from splatfactory.models.networks.zipsplat import ZipSplat
        model = ZipSplat(dict(scene_tokens_enabled=True, num_scene_tokens=256,
                             scene_token_init_from_base=True, weights='zipsplat',
                             backbone=dict(weights=None, use_checkpoint=True),
                             freeze_backbone_except_scene=True,
                             query_sample_ratio=[1., 1.], query_ratio_schedule=0.,
                             query_scale_with_views=0., train_prior_probability=0.,
                             eval_use_priors=False, use_checkpoint=True,
                             return_attention=False, skip_head_render=False, compile=False)).cuda().train()
        batch = batch_to_device(next(iter(ds.get_loader(num_workers=0))), 'cuda')
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(batch)
        # Same float32 loss boundary as the trainer; full 252px target renders.
        losses, _ = model.loss(pred, batch)
        assert pred['target_rgb'].shape == batch['target']['image'].shape
        assert all(torch.isfinite(v).all() for v in losses.values())
        losses['total'].mean().backward()
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), name
        for name in ['backbone.backbone.scene_tokens', 'scene_color_query.weight',
                     'downscale.0.weight', 'gaussian_head.gaussian_head.1.weight']:
            grad = dict(model.named_parameters())[name].grad
            assert grad is not None and grad.abs().sum() > 0, name
            gradients[name] = float(grad.norm())
        torch.cuda.synchronize()
        report['model_backward'] = {'scene': batch['name'][0], 'context_views': 2,
                                    'target_views': 4, 'render_resolution': 252,
                                    'scene_tokens_per_view': 256, 'gaussians': pred['gaussians'].num_gaussians,
                                    'losses': {k: v.detach().cpu().tolist() for k,v in losses.items()},
                                    'gradients': gradients, 'seconds': time.perf_counter()-start,
                                    'peak_cuda_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                                    'optimizer_updates': 0, 'quality_evaluation': False}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(previews), 2, figsize=(6, 3*len(previews)))
    for row, (label, image, depth) in enumerate(previews):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(label)
        axes[row, 1].imshow(np.where(depth>0, np.log(np.maximum(depth, 1e-6)), np.nan), cmap='magma')
        axes[row, 1].set_title('Pseudo depth (log scale)')
        axes[row, 0].axis('off'); axes[row, 1].axis('off')
    fig.tight_layout()
    fig.savefig(args.output.parent/'rgb_depth_preview.png', dpi=110)
    plt.close(fig)
    report['passed'] = True
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    summary['validation_passed'] = True
    summary['full_model_backward_passed'] = report['model_backward'] is not None
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
