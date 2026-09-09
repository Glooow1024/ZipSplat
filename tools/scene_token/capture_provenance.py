"""Capture final source/environment provenance independently of worker log files."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from audit_kmeans import sha

ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    def git(*a):return subprocess.check_output(['git',*a],cwd=ROOT,text=True)
    assert not git('diff','c674892ae8787c83c8f6980ec1c8bdc308c524ee','--','zipsplat'), 'Baseline model code was changed'
    assert not git('ls-files','--others','--exclude-standard','--','zipsplat'), 'Unexpected model files'
    packages={}
    for name in ['torch','torchvision','lpips','scikit-image','numpy','Pillow','gsplat','einops','matplotlib']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]='unavailable'
    payload={'python':sys.version,'executable':sys.executable,'packages':packages,
        'git_head':git('rev-parse','HEAD').strip(),'branch':git('branch','--show-current').strip(),
        'model_identical_to_baseline':True,'model_baseline':'c674892ae8787c83c8f6980ec1c8bdc308c524ee',
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'zipsplat').glob('*.py'))},
        'scripts_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tools/scene_token').glob('*.py'))},
        'manifest_sha256':sha(args.root/'manifest.json'),
        'weights_sha256':sha('/root/.cache/torch/hub/zipsplat/zipsplat-da3g-252p.tar'),
        'launches':[{'phase':'pilot','scenes':['04','05','09'],'views':[8,16],'gpus':[0,1,2]},
                    {'phase':'coarse','scenes':'all9','views':[4,8,12,16],'gpus':[0,1,2,3]},
                    {'phase':'fine','ratios':[.75,.375,.1875],'gpus':[0,1,2,3]},
                    {'phase':'fixed_k','views':[12],'K':[324,648],'gpus':[5]},
                    {'phase':'standalone_profile','scenes':['04','05'],'gpus':[4]}],
        'gpu_snapshot':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total','--format=csv'],text=True)}
    (args.root/'provenance.json').write_text(json.dumps(payload,indent=2));print(json.dumps(packages,indent=2))


if __name__=='__main__':main()
