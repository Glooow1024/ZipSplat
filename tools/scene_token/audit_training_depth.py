"""Audit per-frame pseudo-depth ranges; flag catastrophic finite teacher outliers."""
import argparse,io,json,tarfile
from pathlib import Path
import numpy as np
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
    for split in ['train','validation']:
        index=json.loads((a.dataset/split/'index.json').read_text())
        for key,file in index.items():
            with tarfile.open(a.dataset/split/file) as tar:
                bounds=np.asarray(json.load(tar.extractfile(key+'.depth_ranges.json')),dtype=np.float64)
                poses=np.load(io.BytesIO(tar.extractfile(key+'.poses.npy').read()))
            assert np.isfinite(bounds).all() and (bounds>0).all()
            assert poses.ndim==2 and poses.shape[1]==12
            center=poses[:,-3:];extent=float(np.linalg.norm(center-center.mean(0),axis=1).max())
            typical=np.sqrt(bounds[:,0]*bounds[:,1])
            rows.append(dict(scene=key,split=split,frames=len(bounds),minimum=float(bounds[:,0].min()),maximum=float(bounds[:,1].max()),
                median_midpoint=float(np.median(typical)),frame_midpoint_ratio=float(typical.max()/typical.min()),camera_extent=extent,
                extreme_frames=np.where(bounds[:,1]>1e8)[0].tolist(),maximum_to_camera_extent=float(bounds[:,1].max()/max(extent,1e-8))))
    report=dict(rows=rows,flagged=[r for r in rows if r['extreme_frames']],policy='Conservative catastrophic-outlier flag: any stored depth upper bound >1e8 COLMAP units. Not a general depth-quality certification.')
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(dict(total=len(rows),flagged=report['flagged'],largest=sorted(rows,key=lambda r:r['maximum'],reverse=True)[:5]),indent=2))
if __name__=='__main__':main()
