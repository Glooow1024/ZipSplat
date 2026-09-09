"""Verify budgets, camera reuse, coverage and independently recomputed PNG PSNR."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--complete',action='store_true');args=ap.parse_args()
    manifest=json.loads((args.root/'manifest.json').read_text())
    results={}
    for p in args.root.glob('*/*/*/result.json'):
        row=json.loads(p.read_text());results[(row['scene'],row['views'],row['label'])]=(p,row)
    signatures={r['signature'] for _,r in results.values()};assert len(signatures)==1
    max_png_psnr_error=0.;equivalence_count=0;verified=0
    for key,(path,row) in results.items():
        scene=next(s for s in manifest['scenes'] if s['scene']==key[0]);v=key[1]
        assert set(scene['targets']).isdisjoint(set().union(*(set(c['names']) for c in scene['contexts'].values())))
        assert row['K']==max(1,min(v*324,int(v*324*row['ratio'])))
        assert row['gaussians']==32*row['K'] and 1<=row['unique_indices']<=row['K']
        assert len(row['performance']['full_forward_seconds'])==10
        baseline=results[(key[0],v,'r1')][1]
        assert row['alignment_A']==baseline['alignment_B']
        if row['equivalence']:
            assert row['equivalence']['cached_vs_direct_max_abs']<=1e-6
            assert row['equivalence']['repeat_max_abs']<=1e-6
            equivalence_count+=1
        for protocol in ['A','B']:
            for split in ['context','target']:
                q=row['quality'][protocol][split]
                expected=scene['contexts'][str(v)]['names'] if split=='context' else scene['targets']
                assert [x['frame'] for x in q['per_view']]==expected
                assert all(math.isfinite(x[k]) for x in q['per_view'] for k in ['psnr','ssim','lpips'])
                for k in ['psnr','ssim','lpips']:
                    assert abs(np.mean([x[k] for x in q['per_view']])-q['mean'][k])<1e-8
                # Independent file-level check on the first frame of each output group.
                name=expected[0]
                pred=np.asarray(Image.open(path.parent/protocol/split/name),dtype=np.float32)/255
                ref=np.asarray(Image.open(path.parent.parent/'reference'/name),dtype=np.float32)/255
                assert pred.shape==ref.shape==(252,252,3)
                recomputed=-10*math.log10(max(float(((pred-ref)**2).mean()),1e-10))
                error=abs(recomputed-q['per_view'][0]['psnr'])
                max_png_psnr_error=max(max_png_psnr_error,error)
                assert error<.1,(key,protocol,split,error)
        verified+=1
    coarse={'r1','r0.5','r0.25','r0.125','r0.0625','r0.03125'}
    ncoarse=sum(k[2] in coarse for k in results)
    if args.complete:
        assert ncoarse==216 and verified==342,(ncoarse,verified)
        for scene in manifest['scenes']:
            for v in [4,8,12,16]:
                for label in ['r0.75','r0.375','r0.1875']:
                    assert (scene['scene'],v,label) in results
                for K in [324,648]:
                    assert any(r['scene']==scene['scene'] and r['views']==v and r['K']==K for _,r in results.values())
        profile=json.loads((args.root/'standalone_profile.json').read_text())
        assert len(profile['rows'])==32
        assert len({(r['scene'],r['views'],r['ratio']) for r in profile['rows']})==32
    payload={'verified_results':verified,'coarse_results':ncoarse,'expected_coarse':216,
        'equivalence_checks':equivalence_count,'max_png_psnr_error_db':max_png_psnr_error,
        'signature':next(iter(signatures)),'all_checks_passed':True,'complete_mode':args.complete}
    (args.root/'verification.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
