"""Read-only dataset/environment audit; writes JSON report only, never trains.

Metadata/file-presence checks cover the valid DL3DV manifest. Image decoding
and RE10K shard loading are explicitly sampled, not exhaustive corruption tests.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
ROOT = Path('/data/datasets/3dvision/DL3DV-10K')


def check_scene(rel):
    result = {'scene': rel, 'errors': []}
    try:
        path = ROOT / rel
        tf = json.loads((path / 'transforms.json').read_text())
        frames = tf['frames']
        entries = list(os.scandir(path / 'images_4'))
        sizes = {e.name: e.stat().st_size for e in entries if e.is_file()}
        names = [Path(f['file_path']).name for f in frames]
        result.update(frames=len(frames), images=len(sizes), bytes=sum(sizes.values()),
                      distortion={k: tf.get(k, 0) for k in ('k1', 'k2', 'k3', 'k4', 'p1', 'p2')})
        missing = [n for n in names if sizes.get(n, 0) == 0]
        if missing:
            result['errors'].append({'missing_or_empty_frames': missing[:10], 'count': len(missing)})
        if len(set(names)) != len(names):
            result['errors'].append('duplicate_frame_names')
        mat = np.asarray([f['transform_matrix'] for f in frames], dtype=np.float64)
        if mat.shape != (len(frames), 4, 4) or not np.isfinite(mat).all():
            result['errors'].append('invalid_pose_matrix')
        else:
            rot = mat[:, :3, :3]
            result['max_rotation_orthogonality_error'] = float(np.abs(rot @ rot.transpose(0, 2, 1) - np.eye(3)).max())
            if np.max(np.abs(mat[:, 3] - [0, 0, 0, 1])) > 1e-4:
                result['errors'].append('invalid_homogeneous_row')
        intrinsics = np.array([tf[k] for k in ('w', 'h', 'fl_x', 'fl_y', 'cx', 'cy')])
        if not np.isfinite(intrinsics).all() or (intrinsics[:4] <= 0).any():
            result['errors'].append('invalid_intrinsics')
        if len(frames) < 28:
            result['errors'].append('less_than_24_context_plus_4_target')
        result['sample_image'] = str(path / 'images_4' / names[0])
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    return result


def audit_torch_split(path):
    import torch
    files = sorted(path.glob('*.torch'))
    result = {'path': str(path), 'files': len(files), 'bytes': sum(p.stat().st_size for p in files)}
    index_file = path / 'index.json'
    if index_file.exists():
        index = json.loads(index_file.read_text())
        result['index_scenes'] = len(index)
        result['index_missing_files'] = sorted({str(v) for v in index.values() if not (path / str(v)).exists()})
        result['scene_keys'] = sorted(index)
    result['samples'] = []
    for f in ([files[0], files[-1]] if len(files) > 1 else files):
        record = {'file': str(f)}
        try:
            scenes = torch.load(f, map_location='cpu', weights_only=True)
            record['scenes'] = len(scenes)
            scene = scenes[0]
            record['keys'] = sorted(scene)
            record['first_scene_key'] = scene.get('key')
            record['camera_shape'] = list(scene['cameras'].shape)
            record['cameras_finite'] = bool(torch.isfinite(scene['cameras']).all())
            record['images'] = len(scene['images'])
            with Image.open(io.BytesIO(scene['images'][0].numpy().tobytes())) as img:
                img.load()
                record['first_image_size'] = list(img.size)
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        result['samples'].append(record)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'scope': 'metadata and file presence exhaustive over valid manifest; sampled RGB/shards; no optimizer/GPU training'}
    manifest = (ROOT / 'MANIFEST_COMPLETE.txt').read_text().splitlines()
    with ThreadPoolExecutor(max_workers=8) as pool:
        scenes = list(pool.map(check_scene, manifest))
    keys = {Path(s).name for s in manifest}
    official_test = set(json.loads((REPO / 'splatfactory/datasets/scripts/dl3dv/split.json').read_text())['test'])
    eval_keys = {p.name for p in Path('/data/datasets/3dvision/DL3DV-Evaluation/images').iterdir() if p.is_dir()}
    report['dl3dv'] = {
        'valid_manifest_count': len(manifest), 'unique_scene_hashes': len(keys),
        'failed_manifest_count': len((ROOT / 'MANIFEST_FAILED.tsv').read_text().splitlines()) - 1,
        'audited_frames': sum(s.get('frames', 0) for s in scenes),
        'image_bytes': sum(s.get('bytes', 0) for s in scenes),
        'scenes_with_errors': [s for s in scenes if s['errors']],
        'official_test_count': len(official_test), 'official_test_in_valid': sorted(keys & official_test),
        'official_test_missing': sorted(official_test - keys), 'evaluation_count': len(eval_keys),
        'evaluation_in_valid': sorted(keys & eval_keys),
        'evaluation_in_official_test': sorted(eval_keys & official_test),
        'eligible_excluding_both': len(keys - official_test - eval_keys),
        'nonzero_distortion_scenes': sum(any(abs(float(v)) > 1e-12 for v in s.get('distortion', {}).values()) for s in scenes),
        'scene_records': scenes,
    }
    samples = []
    for bucket in sorted({Path(s).parts[0] for s in manifest}):
        candidates = [s for s in scenes if s['scene'].startswith(bucket + '/') and not s['errors']]
        for s in candidates[:2]:
            record = {'scene': s['scene']}
            try:
                with Image.open(s['sample_image']) as img:
                    img.load()
                    record.update(size=list(img.size), mode=img.mode)
            except Exception as exc:
                record['error'] = str(exc)
            samples.append(record)
    report['dl3dv']['decoded_samples'] = samples
    report['re10k'] = {split: audit_torch_split(Path('/data/datasets/RE10K/re10k_torch') / split) for split in ('train', 'test')}
    report['re10k']['train_test_overlap'] = sorted(set(report['re10k']['train'].get('scene_keys', [])) & set(report['re10k']['test'].get('scene_keys', [])))
    report['imports'] = {}
    for module in ('torch', 'torchvision', 'h5py', 'pandas', 'hydra', 'albumentations', 'chamferdist', 'fused_ssim', 'gsplat', 'lpips', 'plotly', 'sklearn', 'tensordict', 'wandb', 'depth_anything_3', 'splatfactory.trainer'):
        try:
            mod = importlib.import_module(module)
            report['imports'][module] = {'ok': True, 'file': getattr(mod, '__file__', None), 'version': getattr(mod, '__version__', None)}
        except Exception as exc:
            report['imports'][module] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
    try:
        dist = metadata.distribution('gsplat')
        report['gsplat_direct_url'] = dist.read_text('direct_url.json')
        gsroot = Path(dist.locate_file('gsplat'))
        report['gsplat_activated_source_files'] = [str(p.relative_to(gsroot)) for p in gsroot.rglob('*') if p.suffix in ('.py', '.cu', '.h', '.cpp') and 'activated' in p.read_text(errors='replace')]
    except Exception as exc:
        report['gsplat_audit_error'] = str(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    compact = {**report, 'dl3dv': {k:v for k,v in report['dl3dv'].items() if k != 'scene_records'}, 're10k': {}}
    for split in ('train', 'test'):
        compact['re10k'][split] = {k:v for k,v in report['re10k'][split].items() if k != 'scene_keys'}
    compact['re10k']['overlap_count'] = len(report['re10k']['train_test_overlap'])
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
