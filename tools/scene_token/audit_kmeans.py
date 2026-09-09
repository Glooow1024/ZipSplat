"""Freeze the short60 context/target manifests before any quality measurement."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def colmap_poses(path):
    result = {}
    with open(path, "rb") as f:
        count = struct.unpack("<Q", f.read(8))[0]
        for _ in range(count):
            _, w, x, y, z, tx, ty, tz, _ = struct.unpack("<i7di", f.read(64))
            name = bytearray()
            while True:
                ch = f.read(1)
                if not ch:
                    raise EOFError(path)
                if ch == b"\0":
                    break
                name.extend(ch)
            points = struct.unpack("<Q", f.read(8))[0]
            f.seek(points * 24, 1)
            R = np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                          [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                          [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = R.T, -R.T @ np.array([tx, ty, tz])
            result[Path(name.decode()).name] = pose
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path('/root/multiview_compare/experiments/short60'))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists(), f"Refusing to replace frozen manifest: {args.output}"
    output = {"version": 1, "image_size": 252, "views": [4, 8, 12, 16],
              "ratios": [1, .5, .25, .125, .0625, .03125], "scenes": [], "excluded": [],
              "protocol": "NoPrior; context-only shared Sim(3); fixed-baseline and per-budget alignment; no target RGB pose fitting; pinhole approximation (no undistortion)"}
    thumbs = []
    for scene in sorted(args.input.glob('[0-9][0-9]_*')):
        tfpath = scene / 'transforms.json'
        tf = json.loads(tfpath.read_text())
        selections = {}
        union = set()
        for v in output['views']:
            mp = scene / f'{v}views/querysplat/selection_manifest.json'
            selection = json.loads(mp.read_text())
            names = selection['selected_frames']
            assert len(names) == v and len(set(names)) == v
            source = Path(selection['source_image_directory'])
            for name in names:
                linked = scene / f'{v}views/querysplat/input' / name
                assert linked.resolve() == (source / name).resolve() and linked.is_file()
            selections[str(v)] = {"names": names, "source_manifest": str(mp), "sha256": sha(mp)}
            union.update(names)
        poses = {Path(f['file_path']).name: np.asarray(f['transform_matrix']) @ np.diag([1,-1,-1,1]) for f in tf['frames']}
        pool = sorted(p.name for p in source.glob('*.png'))[:60]
        pose_source = str(tfpath)
        if not set(pool).issubset(poses):
            cp = next((p/'colmap/sparse/0/images.bin' for p in source.parents if (p/'colmap/sparse/0/images.bin').is_file()), None)
            if cp:
                poses, pose_source = colmap_poses(cp), str(cp)
        missing = sorted(union - poses.keys())
        if missing:
            output['excluded'].append({"scene": scene.name, "reason": "missing_context_GT", "frames": missing})
            continue
        candidates = [n for n in pool if n not in union and n in poses]
        assert len(candidates) >= 8, (scene.name, len(candidates))
        targets = [candidates[i] for i in np.linspace(0, len(candidates)-1, 8).round().astype(int)]
        frames = {}
        for name in sorted(union | set(targets)):
            path = source / name
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                width, height = im.size
            pose = poses[name]
            assert np.isfinite(pose).all() and np.allclose(pose[3], [0,0,0,1])
            assert np.allclose(pose[:3,:3].T @ pose[:3,:3], np.eye(3), atol=1e-4)
            meta = next((f for f in tf['frames'] if Path(f['file_path']).name == name), {})
            intr = {k: meta.get(k, tf.get(k)) for k in ['w','h','fl_x','fl_y','cx','cy']}
            assert all(v is not None for v in intr.values())
            sx, sy = width/intr['w'], height/intr['h']
            side = min(width, height)
            K = [[intr['fl_x']*sx*252/side, 0, (intr['cx']*sx-(width-side)//2)*252/side],
                 [0,intr['fl_y']*sy*252/side,(intr['cy']*sy-(height-side)//2)*252/side],[0,0,1]]
            frames[name] = {"path": str(path), "sha256": sha(path), "source_size": [width,height], "K": K, "c2w_cv": pose.tolist()}
        output['scenes'].append({"scene":scene.name,"contexts":selections,"targets":targets,"frames":frames,
            "transforms":str(tfpath),"transforms_sha256":sha(tfpath),"pose_source":pose_source,"pose_sha256":sha(pose_source),
            "distortion":{k:tf.get(k,0) for k in ['k1','k2','k3','k4','p1','p2']}})
        with Image.open(source / selections['8']['names'][0]) as im:
            thumb = im.convert('RGB'); thumb.thumbnail((252,180)); thumbs.append((scene.name[:2],thumb.copy()))
        print(scene.name[:2], 'contexts', len(union), 'targets', targets, 'poses', len(poses), flush=True)
    assert len(output['scenes']) == 9, len(output['scenes'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2))
    from PIL import ImageDraw
    contact = Image.new('RGB',(3*252,3*205),'white'); draw=ImageDraw.Draw(contact)
    for i,(label,thumb) in enumerate(thumbs):
        x,y=(i%3)*252,(i//3)*205; contact.paste(thumb,(x,y+20)); draw.text((x+5,y+3),label,fill='black')
    contact.save(args.output.parent/'scene_contact.jpg')
    print('AUDIT_OK', len(output['scenes']), 'excluded', output['excluded'],flush=True)


if __name__ == '__main__':
    main()
