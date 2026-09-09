"""Recompute only depth-teacher groups affected by held-out target frames."""
import argparse,hashlib,json,sys,tarfile,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from depth_anything_3.api import DepthAnything3
from splatfactory.datasets.scripts.dl3dv.convert import _opencv_pose_12,_scale_intrinsics
from splatfactory.datasets.scripts.utils import update_camera_intrinsics
from splatfactory.datasets.utils.io import iter_scenes_from_tar,write_scene_to_tar,encode_depth,decode_image
from splatfactory.geometry import Camera,Pose
from splatfactory.utils.image import ImagePreprocessor,crop_to_principal_point,resize_to_cover
from tools.scene_token.train_initial_test import dump


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(256)
    plan=json.loads(a.plan.read_text());manifest=json.loads((a.pilot/'manifest.json').read_text())
    teacher_path=Path(manifest['teacher']);assert sha(teacher_path/'model.safetensors')==manifest['teacher_weights_sha256']
    teacher=DepthAnything3.from_pretrained(str(teacher_path)).cuda().eval();pre=ImagePreprocessor({'resize':252})
    reports=[]
    for record in manifest['selected']:
        start=time.time();key=record['key'];source=Path(manifest['source'])/record['scene']
        old=list(iter_scenes_from_tar(str(a.pilot/record['split']/record['shard'])));assert len(old)==1;old=old[0]
        prov=json.loads((a.pilot/'provenance'/f'{key}.json').read_text())
        assert sha(a.pilot/record['split']/record['shard'])==plan['tar_hashes'][str(a.pilot/record['split']/record['shard'])]
        assert sha(source/'transforms.json')==prov['source_transforms_sha256']
        tf=json.loads((source/'transforms.json').read_text());held=set(plan['scenes'][key]['target']);n=old['meta']['num_views']
        groups=prov['teacher_chunks'];depths=[old['depths'][i] for i in range(n)];ranges=np.array(old['depth_ranges']).tolist()
        cached={};updates=[];teacher_map={};min_valid=1.
        for group in groups:
            allowed=[i for i in group if i not in held]
            for i in allowed:teacher_map[str(i)]=allowed if held.intersection(group) else group
            if not held.intersection(group):continue
            assert len(allowed)>=2 and not held.intersection(allowed)
            images=[];cameras=[];poses=[];shape=None
            for i in allowed:
                frame=tf['frames'][i];path=source/'images_4'/Path(frame['file_path']).name
                assert sha(path)==prov['source_frames'][i]['sha256']
                im=np.array(Image.open(path).convert('RGB'));h,w=im.shape[:2]
                if shape is None:shape=(h,w)
                assert shape==(h,w)
                fx,fy,cx,cy=_scale_intrinsics(tf,w);im,t1=crop_to_principal_point(im,cx,cy);im,t2=resize_to_cover(im,h,w)
                cameras.append(update_camera_intrinsics(fx,fy,cx,cy,t2@t1,w,h));images.append(im)
                poses.append(_opencv_pose_12(np.asarray(frame['transform_matrix'],dtype=np.float32)))
            cameras=np.asarray(cameras,dtype=np.float32);poses=np.asarray(poses,dtype=np.float32)
            K=Camera(torch.from_numpy(cameras)).K.numpy();w2c=Pose(torch.from_numpy(poses)).inv().Rt.numpy()
            pred=teacher.inference(image=images,intrinsics=K,extrinsics=w2c,align_to_input_ext_scale=True,
                process_res=504,process_res_method='upper_bound_resize',infer_gs=False)
            ds=F.interpolate(torch.from_numpy(pred.depth).float()[:,None],size=shape,mode='bilinear',align_corners=False)[:,0].numpy()
            for j,i in enumerate(allowed):
                assert np.isfinite(ds[j]).all() and (ds[j]>0).all()
                result=pre(images[j],depth=ds[j],aspect_ratio=1.)
                rgb=(result['image'].permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8')
                assert np.array_equal(rgb,decode_image(old['images'][i])),(key,i,'RGB changed')
                cam=update_camera_intrinsics(*cameras[j][2:6],result['transform'],252,252)
                assert np.allclose(cam,old['cameras'][i],atol=1e-5,rtol=0),(key,i,'K changed')
                depth=result['depth'].numpy();depths[i],lo,hi=encode_depth(depth);ranges[i]=[lo,hi]
                min_valid=min(min_valid,float(((depth>=lo)&(depth<=hi)).mean()));updates.append(i)
            del pred,ds
        assert set(teacher_map)=={str(i) for i in range(n) if i not in held}
        assert all(not held.intersection(ids) for ids in teacher_map.values())
        scene=dict(key=key,num_views=n,has_depth=True,images=[old['images'][i] for i in range(n)],
            cameras=old['cameras'],poses=old['poses'],depths=depths,depth_ranges=ranges)
        dest=a.output/record['split']/record['shard'];dest.parent.mkdir(parents=True,exist_ok=True)
        with tarfile.open(dest.with_suffix('.tmp'),'w') as tar:write_scene_to_tar(tar,scene)
        dest.with_suffix('.tmp').replace(dest)
        report=dict(key=key,split=record['split'],shard=record['shard'],frames=n,heldout=sorted(held),updated_depth_frames=sorted(updates),
            allowed_frame_teacher_inputs=teacher_map,all_nonheldout_depths_exclude_heldout=True,
            unchanged_rgb_camera_pose=True,minimum_updated_depth_valid_fraction=min_valid,tar_sha256=sha(dest),
            source_tar_sha256=prov['tar_sha256'],seconds=time.time()-start,
            heldout_depth='Original depth retained only for evaluation metrics; never sampled for training context/target')
        (a.output/'provenance').mkdir(exist_ok=True);dump(a.output/'provenance'/f'{key}.json',report);reports.append(report)
        print(json.dumps(dict(scene=key,updated_depth_frames=len(updates),seconds=report['seconds'])),flush=True)
    for split in ['train','validation']:
        dump(a.output/split/'index.json',{r['key']:r['shard'] for r in reports if r['split']==split})
    dump(a.output/'completion.json',dict(passed=True,scenes=len(reports),teacher_weights_sha256=manifest['teacher_weights_sha256'],
        source_plan_sha256=sha(a.plan),total_updated_depth_frames=sum(len(r['updated_depth_frames']) for r in reports),
        heldout_isolated_from_nonheldout_teacher_inputs=True,rgb_camera_pose_unchanged=True,reports=reports))


if __name__=='__main__':main()
