"""Pilot raw-scene cache, fixed evaluation sets and reserved-frame sampling."""
import copy
import functools
import hashlib
import json
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from splatfactory.datasets import get_dataset
from splatfactory.datasets.utils import io, workers


class ExactSampler:
    def __init__(self, context, target):
        self.context = list(context); self.target = list(target)
    def sample(self, **kwargs):
        return dict(context_indices=self.context.copy(), target_indices=self.target.copy(), overlap_scores=-1.)


class PilotData:
    def __init__(self, data_conf, plan=None):
        dc=OmegaConf.create(data_conf)
        dc.image_num_range=[2,2]; dc.train_batch_size=2; dc.num_target_views=4
        dc.random_reference_view=False
        self.dataset=get_dataset(dc.name)(dc,split='train')
        self.raw={}; self.split={}; self.tar_hashes={}
        for split,folder in [('train',dc.train_shard_dir),('validation',dc.test_shard_dir)]:
            for path in sorted(Path(folder).glob('shard-*.tar')):
                self.tar_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                for raw in io.iter_scenes_from_tar(str(path)):
                    self.raw[raw['key']]=raw; self.split[raw['key']]=split
        assert sorted(self.split.values()).count('train')==8 and sorted(self.split.values()).count('validation')==2
        original=self.dataset._load_view
        @functools.lru_cache(maxsize=None)
        def cached(key,index):
            return original(self.raw[key],index,1.0,self.dataset.preprocessor)
        self.dataset._load_view=lambda raw,index,aspect_ratio=1.,preprocessor=None:cached(raw['key'],index)
        self.plan=plan

    def fixed(self,key,context,target):
        raw=self.raw[key]
        assert max(context+target)<raw['meta']['num_views'] and min(context+target)>=0
        self.dataset.step_timer.reset()
        sample=self.dataset._process_scene(raw,len(context),view_sampler=ExactSampler(context,target))
        return workers.collate([sample])

    def training(self,key):
        reserved=set(self.plan['scenes'][key]['reserved'])
        for attempt in range(10000):
            indices=self.dataset.view_sampler.sample(scene_id=key,num_context_views=2,num_target_views=4,
                total_num_views=self.raw[key]['meta']['num_views'])
            context=indices['context_indices']; target=indices['target_indices']
            if reserved.intersection(context+target):continue
            # Original training protocol randomizes reference view.
            context=list(np.random.permutation(context))
            try: batch=self.fixed(key,context,target)
            except ValueError as e:
                if 'pose jump ratio' not in str(e):raise
                continue
            return batch,dict(context=[int(x) for x in context],target=[int(x) for x in target],rejected=attempt)
        raise RuntimeError(f'Unable to sample {key} without held-out frames')

    def evaluation(self,split,views):
        batches=[]
        for key in sorted(self.raw):
            if self.split[key]!=split:continue
            entry=self.plan['scenes'][key]
            batches.append(self.fixed(key,entry[f'context{views}'],entry['target']))
        return batches


def create_plan(data):
    scenes={}
    for key,raw in sorted(data.raw.items()):
        n=raw['meta']['num_views']
        # Prefer the middle of the trajectory, shift deterministically if the
        # existing pose-jump guard rejects that window. No image-quality choice.
        candidates=sorted(range(0,n-18),key=lambda a:(abs(a-(n-19)//2),a))
        for a in candidates:
            target=[a+i for i in [2,4,6,8,10,12,14,16]]
            c2=[a,a+18]; c6=[a+i for i in [0,3,7,11,15,18]]
            try:data.fixed(key,c2,target);data.fixed(key,c6,target)
            except ValueError as e:
                if 'pose jump ratio' not in str(e):raise
                continue
            scenes[key]=dict(split=data.split[key],num_views=n,context2=c2,context6=c6,target=target,
                reserved=target if data.split[key]=='train' else [],window_start=a)
            break
        assert key in scenes
    return dict(scenes=scenes,tar_hashes=data.tar_hashes,
        protocol='8 train +2 validation scenes; 8 reserved targets per train scene excluded from ALL training context and target sampling; nested V2/V6 context over identical18-frame span and identical targets',
        limitation='One fixed local evaluation window per scene, not full trajectory or official benchmark')
