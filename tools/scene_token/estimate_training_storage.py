"""Sample real RGB encodings in memory; estimate storage, without writing shards."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from splatfactory.datasets.utils.io import decode_image, encode_image
from splatfactory.utils.image import crop_to_principal_point, crop_to_ar, resize_to_cover


def encode252(raw, cx, cy):
    im = decode_image(raw)
    im, _ = crop_to_principal_point(im, cx, cy)
    im, _ = crop_to_ar(im, 1.0)
    im, _ = resize_to_cover(im, 252, 252)
    return len(encode_image(im))


def dl_sample(scene):
    root = Path('/data/datasets/3dvision/DL3DV-10K') / scene
    tf = json.loads((root / 'transforms.json').read_text())
    frames = tf['frames']
    sizes = []
    for i in sorted({0, len(frames)//2, len(frames)-1}):
        raw = (root / 'images_4' / Path(frames[i]['file_path']).name).read_bytes()
        im = decode_image(raw)
        scale = im.shape[1]/tf['w']
        sizes.append(encode252(raw, tf['cx']*scale, tf['cy']*scale))
    return {'scene':scene, 'frame_count':len(frames), 'encoded_rgb_bytes':sizes}


def re_sample(path):
    scenes = torch.load(path, map_location='cpu', weights_only=True)
    records = []
    for s in scenes:
        records.append({'key':s['key'], 'frames':len(s['images'])})
    sizes = []
    for s in scenes[:3]:
        for i in sorted({0, len(s['images'])//2, len(s['images'])-1}):
            raw = s['images'][i].numpy().tobytes()
            im = decode_image(raw)
            cam = s['cameras'][i]
            sizes.append(encode252(raw, float(cam[2])*im.shape[1], float(cam[3])*im.shape[0]))
    return {'file':str(path), 'file_bytes':path.stat().st_size,'scenes':records,'encoded_rgb_bytes':sizes}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    rng=random.Random(256)
    audit=json.loads((args.audit_dir/'audit.json').read_text())['dl3dv']
    follow=json.loads((args.audit_dir/'followup.json').read_text())
    low={x['scene'] for x in follow['non_960x540_first_images'] if 'size' in x and min(x['size'])<252}
    candidates=set((args.audit_dir/'training_candidates_provisional.txt').read_text().splitlines())
    eligible=sorted(candidates-low)
    with ThreadPoolExecutor(max_workers=4) as pool:
        dl=list(pool.map(dl_sample,rng.sample(eligible,88)))
    frame_count=sum(s['frames'] for s in audit['scene_records'] if s['scene'] in set(eligible))
    dl_sizes=[b for r in dl for b in r['encoded_rgb_bytes']]
    re_root=Path('/data/datasets/RE10K/re10k_torch')
    re=[]
    for split in ['train','test']:
        files=sorted((re_root/split).glob('*.torch'))
        for path in rng.sample(files,12): re.append(re_sample(path))
    re_sizes=[b for r in re for b in r['encoded_rgb_bytes']]
    re_scenes=sum(len(r['scenes']) for r in re)
    re_frames=sum(s['frames'] for r in re for s in r['scenes'])
    re_est_frames=re_frames/re_scenes*(66033+7286)
    def estimate(sizes,frames):
        mean=float(np.mean(sizes))
        return {'sample_rgb_frames':len(sizes),'rgb_bytes_mean':mean,'rgb_bytes_p10_p50_p90':np.percentile(sizes,[10,50,90]).tolist(),
                'frames_for_extrapolation':frames,'rgb_GB':mean*frames/1e9,
                'depth_GB_assuming_16_to_64_KiB_per_frame':[frames*16384/1e9,frames*65536/1e9],
                'tar_metadata_budget_GB':frames*2048/1e9}
    result={'seed':256,'note':'RGB measured in memory at252; depth NOT generated,16-64KiB is budgeting scenario; RE frame count extrapolated from sampled scenes',
            'low_resolution_count':len(low),'low_resolution_in_candidates':sorted(low & candidates),
            'excluded_unique_total_in_old_complete':len(audit['scenes_with_errors'])+len(low-{s['scene'] for s in audit['scenes_with_errors']}),
            'training_candidates_after_min252':len(eligible),
            'dl3dv':estimate(dl_sizes,frame_count),'re10k':estimate(re_sizes,re_est_frames),
            're_sample_scenes':re_scenes,'re_sample_frames':re_frames,'dl_samples':dl,'re_samples':re}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['dl_samples','re_samples']},indent=2))


if __name__=='__main__':main()
