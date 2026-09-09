"""Bounded DL3DV pilot: deterministic scene split, official tar IO, DA3 pseudo depth.

This writes a NEW output directory. A matching manifest permits resuming completed
scenes; source data is never modified. Not a full paper reproduction.
"""
import argparse
import hashlib
import json
import math
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from splatfactory.datasets.scripts.dl3dv.convert import _opencv_pose_12, _scale_intrinsics
from splatfactory.datasets.scripts.utils import update_camera_intrinsics
from splatfactory.datasets.utils.io import encode_depth, encode_image, write_scene_to_tar
from splatfactory.geometry import Camera, Pose
from splatfactory.utils.image import ImagePreprocessor, crop_to_principal_point, resize_to_cover

VERSION = 'dl3dv-pilot-252-da3g-v1'


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def prepare_scene(args, record, teacher):
    start = time.perf_counter()
    source = args.source / record['scene']
    tf_path = source / 'transforms.json'
    tf = json.loads(tf_path.read_text())
    frames = tf['frames']
    images, cameras, poses, sources = [], [], [], []
    original_shape = None
    for frame in frames:
        p = source / 'images_4' / Path(frame['file_path']).name
        with Image.open(p) as im:
            img = np.array(im.convert('RGB'))
        h, w = img.shape[:2]
        if original_shape is None:
            original_shape = (h, w)
        assert (h, w) == original_shape and min(h, w) >= 252, (p, img.shape)
        # Same principal-point normalization as the official DL3DV converter.
        fx, fy, cx, cy = _scale_intrinsics(tf, w)
        img, t1 = crop_to_principal_point(img, cx, cy)
        img, t2 = resize_to_cover(img, h, w)
        camera = update_camera_intrinsics(fx, fy, cx, cy, t2 @ t1, w, h)
        pose = _opencv_pose_12(np.asarray(frame['transform_matrix'], dtype=np.float32))
        assert np.isfinite(camera).all() and np.isfinite(pose).all()
        images.append(img)
        cameras.append(camera)
        poses.append(pose)
        sources.append({'file': str(p.relative_to(args.source)), 'sha256': digest(p)})
    cameras = np.asarray(cameras, dtype=np.float32)
    poses = np.asarray(poses, dtype=np.float32)
    K = Camera(torch.from_numpy(cameras)).K.numpy()
    w2c = Pose(torch.from_numpy(poses)).inv().Rt.numpy()
    n = len(images)
    assert n == record['frames'] and n >= 28
    preprocessor = ImagePreprocessor({'resize': 252})
    rgb_bytes, depth_bytes, ranges, final_cameras = [None]*n, [None]*n, [None]*n, [None]*n
    valid_fractions = []
    chunks = [list(range(c, n, math.ceil(n / args.max_views))) for c in range(math.ceil(n / args.max_views))]
    for ci, ids in enumerate(chunks):
        prediction = teacher.inference(
            image=[images[i] for i in ids], intrinsics=K[ids], extrinsics=w2c[ids],
            align_to_input_ext_scale=True, process_res=504,
            process_res_method='upper_bound_resize', infer_gs=False,
        )
        depths = F.interpolate(torch.from_numpy(prediction.depth).float()[:, None],
                               size=original_shape, mode='bilinear', align_corners=False)[:, 0].numpy()
        for j, i in enumerate(ids):
            assert np.isfinite(depths[j]).all() and (depths[j] > 0).all(), (record['scene'], i)
            # Equivalent to official train preprocessing, applied once before storage.
            result = preprocessor(images[i], depth=depths[j], aspect_ratio=1.0)
            rgb = (result['image'].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
            depth = result['depth'].numpy()
            assert rgb.shape == (252, 252, 3) and depth.shape == (252, 252)
            rgb_bytes[i] = encode_image(rgb)
            depth_bytes[i], lo, hi = encode_depth(depth)
            assert 0 < lo <= hi and np.isfinite([lo, hi]).all()
            ranges[i] = [lo, hi]
            valid_fractions.append(float(((depth >= lo) & (depth <= hi)).mean()))
            cam = cameras[i]
            final_cameras[i] = update_camera_intrinsics(*cam[2:6], result['transform'], 252, 252)
        print(json.dumps({'scene': record['scene'], 'chunk': ci+1, 'chunks': len(chunks),
                          'elapsed_s': round(time.perf_counter()-start, 1)}), flush=True)
        del prediction, depths
    scene = {'key': record['key'], 'num_views': n, 'has_depth': True,
             'images': rgb_bytes, 'cameras': np.asarray(final_cameras, dtype=np.float32),
             'poses': poses, 'depths': depth_bytes, 'depth_ranges': ranges}
    dest = args.output / record['split'] / record['shard']
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.tar.tmp')
    with tarfile.open(tmp, 'w') as tar:
        write_scene_to_tar(tar, scene)
    tmp.replace(dest)
    report = dict(record, frames=n, source_shape=list(original_shape),
                  source_transforms_sha256=digest(tf_path), source_frames=sources,
                  distortion={k: tf.get(k, 0) for k in ['k1', 'k2', 'k3', 'k4', 'p1', 'p2']},
                  undistorted=False, teacher_chunks=chunks,
                  rgb_bytes=sum(map(len, rgb_bytes)), depth_bytes=sum(map(len, depth_bytes)),
                  tar_bytes=dest.stat().st_size, tar_sha256=digest(dest),
                  minimum_depth_valid_fraction=min(valid_fractions),
                  seconds=time.perf_counter()-start,
                  peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    save_json(args.output / 'provenance' / (record['key'] + '.json'), report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('/data/datasets/3dvision/DL3DV-10K'))
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-views', type=int, default=16)
    parser.add_argument('--limit', type=int, default=10)
    args = parser.parse_args()
    assert args.max_views >= 2 and 0 < args.limit <= 10
    torch.set_num_threads(4)
    torch.manual_seed(256)
    candidates = args.candidates.read_text().splitlines()
    assert len(candidates) == len(set(candidates))
    official = json.loads((Path(__file__).resolve().parents[2] / 'splatfactory/datasets/scripts/dl3dv/split.json').read_text())['test']
    assert not ({Path(s).name for s in candidates} & set(official))
    ranked = sorted(candidates, key=lambda s: hashlib.sha256(('scene-token-split-256:' + s).encode()).hexdigest())
    nval = math.ceil(len(ranked)*0.05)
    splits = {'validation': sorted(ranked[:nval]), 'train': sorted(ranked[nval:])}
    records = {r['scene']: r for r in json.loads(args.audit.read_text())['dl3dv']['scene_records']}
    selected = []
    for split, count in [('train', 8), ('validation', 2)]:
        ordered = sorted(splits[split], key=lambda s: hashlib.sha256(('scene-token-pilot-256:' + s).encode()).hexdigest())
        for i, s in enumerate(ordered[:count]):
            assert not records[s]['errors']
            selected.append({'scene': s, 'key': Path(s).name, 'split': split,
                             'frames': records[s]['frames'], 'shard': f'shard-{i:06d}.tar'})
    manifest = {'version': VERSION, 'source': str(args.source.resolve()),
                'candidates_sha256': digest(args.candidates), 'teacher': str(args.teacher.resolve()),
                'teacher_weights_sha256': digest(args.teacher / 'model.safetensors'),
                'teacher_max_views': args.max_views, 'teacher_process_res': 504,
                'teacher_chunking': 'round_robin', 'storage_resolution': [252, 252],
                'training_targets': 4, 'training_context_target_overlap': 'official bounded sampler permits overlap',
                'source_undistorted': False, 'split_seed': 'scene-token-split-256',
                'train_count': len(splits['train']), 'validation_count': len(splits['validation']),
                'official_test_ids_reserved': len(official), 'selected': selected}
    manifest_path = args.output / 'manifest.json'
    if args.output.exists():
        assert manifest_path.exists() and json.loads(manifest_path.read_text()) == manifest, 'Refuse mismatching output'
    else:
        args.output.mkdir(parents=True)
        save_json(manifest_path, manifest)
    (args.output / 'provenance').mkdir(exist_ok=True)
    save_json(args.output / 'splits.json', dict(splits, official_test=list(official)))
    pending = []
    for r in selected[:args.limit]:
        report = args.output / 'provenance' / (r['key'] + '.json')
        tar = args.output / r['split'] / r['shard']
        if report.exists():
            assert tar.exists() and digest(tar) == json.loads(report.read_text())['tar_sha256']
        else:
            assert not tar.exists(), f'Untracked shard: {tar}'
            pending.append(r)
    if pending:
        from depth_anything_3.api import DepthAnything3
        teacher = DepthAnything3.from_pretrained(str(args.teacher)).cuda().eval()
        for r in pending:
            prepare_scene(args, r, teacher)
    reports = [json.loads(p.read_text()) for p in sorted((args.output / 'provenance').glob('*.json'))]
    for split in splits:
        folder = args.output / split
        folder.mkdir(exist_ok=True)
        save_json(folder / 'index.json', {r['key']: r['shard'] for r in reports if r['split'] == split})
    # Small custom validation: a fixed 150-frame window, 6 inputs + 8 held-out views.
    # These are NOT the paper benchmark indices or its farthest-point sampler.
    validation = {}
    for r in reports:
        if r['split'] == 'validation':
            ids = np.linspace(0, min(149, r['frames']-1), 14).round().astype(int).tolist()
            context = [ids[i] for i in [0, 2, 5, 8, 11, 13]]
            target = [i for i in ids if i not in context]
            assert len(context) == 6 and len(target) == 8 and not set(context)&set(target)
            validation[r['key']] = {'context': context, 'target': target}
    save_json(args.output / 'validation_ctx6_tgt8.json', validation)
    summary = {'version': VERSION, 'prepared_scenes': len(reports),
               'prepared_frames': sum(r['frames'] for r in reports),
               'train_scenes': sum(r['split']=='train' for r in reports),
               'validation_scenes': sum(r['split']=='validation' for r in reports),
               'tar_bytes': sum(r['tar_bytes'] for r in reports),
               'rgb_bytes': sum(r['rgb_bytes'] for r in reports),
               'depth_bytes': sum(r['depth_bytes'] for r in reports),
               'conversion_seconds': sum(r['seconds'] for r in reports),
               'complete': len(reports)==10, 'validation_passed': False}
    save_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
