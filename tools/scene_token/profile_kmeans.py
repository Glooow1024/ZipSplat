"""Standalone full-pipeline profiling without resident quality features/LPIPS.

Inputs are already loaded RGB tensors. Disk decode is deliberately excluded.
Hooked stage times are separate from uninstrumented end-to-end timings.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from zipsplat import ZipSplat
from zipsplat.utils import load_image
from audit_kmeans import sha


def timed(fn):
    torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize()
    return value,time.perf_counter()-start


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--scenes',default='04,05');args=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20260907)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((args.root/'manifest.json').read_text())
    model=ZipSplat(weights='/root/.cache/torch/hub/zipsplat/zipsplat-da3g-252p.tar').cuda().eval()
    model.requires_grad_(False)
    rows=[]
    for scene in manifest['scenes']:
        if scene['scene'][:2] not in args.scenes.split(','):continue
        for v in [4,8,12,16]:
            raw=torch.stack([load_image(Path(scene['frames'][n]['path'])) for n in scene['contexts'][str(v)]['names']])
            for ratio in [1,.5,.125,.03125]:
                with torch.no_grad():
                    for _ in range(3): g=model(raw,compression=ratio);del g
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    before=torch.cuda.memory_allocated();latencies=[]
                    for _ in range(10):
                        g,t=timed(lambda:model(raw,compression=ratio));latencies.append(t);del g
                    peak=torch.cuda.max_memory_allocated()
                    sample={};orig=[]
                    def hook(obj,name,key):
                        fn=getattr(obj,name);orig.append((obj,name,fn))
                        def wrapped(*a,**kw):
                            value,seconds=timed(lambda:fn(*a,**kw));sample.setdefault(key,[]).append(seconds);return value
                        setattr(obj,name,wrapped)
                    hook(model,'_prepare_inputs','preprocessing_cpu_and_h2d')
                    for name in ['_backbone_features','_prepare','_cluster','_fuse','_color']:
                        hook(model.model,name,name)
                    hook(model.model.gaussian_head,'forward','gaussian_head')
                    for _ in range(10):g=model(raw,compression=ratio);del g
                    for obj,name,fn in orig:setattr(obj,name,fn)
                rows.append({'scene':scene['scene'],'views':v,'ratio':ratio,'K':max(1,int(v*324*ratio)),
                    'latency_seconds':latencies,'median_seconds':statistics.median(latencies),
                    'stage_medians':{k:statistics.median(vals) for k,vals in sample.items()},
                    'stage_samples':sample,'allocated_before':before,'peak_bytes':peak,'incremental_peak':peak-before})
                payload={'script_sha256':sha(__file__),'manifest_sha256':sha(args.root/'manifest.json'),
                    'protocol':'CPU RGB tensors to Gaussians; includes crop-resize+H2D, excludes disk/metric/registration; 3warmup+10repeats; FP32 noTF32; stage synchronization instrumented separately',
                    'rows':rows}
                (args.root/'standalone_profile.json').write_text(json.dumps(payload,indent=2))
                print(scene['scene'][:2],v,ratio,rows[-1]['median_seconds'],rows[-1]['stage_medians'],flush=True)


if __name__=='__main__':main()
