"""Bounded CPU caches, official preprocessing/sampling, fixed validation windows."""
import functools, hashlib, json, random
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from splatfactory.datasets import get_dataset
from splatfactory.datasets.utils import io, workers
from tools.scene_token.multiscene_data import ExactSampler

class TrainingData:
    def __init__(self,root,conf):
        self.root=Path(root);dc=OmegaConf.create(conf)
        self.excluded=set(dc.pop('excluded_scenes',[]))
        dc.dataset_dir=str(root);dc.train_shard_dir=str(self.root/'train');dc.test_shard_dir=str(self.root/'validation')
        dc.image_num_range=[2,6];dc.train_batch_size=1;dc.random_reference_view=False
        self.dataset=get_dataset(dc.name)(dc,split='train')
        self.entries={};self.splits={}
        for split in ['train','validation']:
            for key,shard in json.loads((self.root/split/'index.json').read_text()).items():
                self.entries[key]=self.root/split/shard;self.splits[key]=split
        assert self.excluded<=self.splits.keys()
        self.train=sorted(k for k,v in self.splits.items() if v=='train' and k not in self.excluded)
        self.val=sorted(k for k,v in self.splits.items() if v=='validation' and k not in self.excluded)
        assert self.train and self.val and not set(self.train)&set(self.val)
        original=self.dataset._load_view
        @functools.lru_cache(maxsize=512)
        def view(key,index):return original(self.raw(key),index,1.0,self.dataset.preprocessor)
        self.dataset._load_view=lambda raw,index,aspect_ratio=1.,preprocessor=None:view(raw['key'],index)
        self.plan={}

    @functools.lru_cache(maxsize=8)
    def raw(self,key):
        values=list(io.iter_scenes_from_tar(str(self.entries[key])));assert len(values)==1 and values[0]['key']==key
        return values[0]

    def fixed(self,key,context,target):
        self.dataset.step_timer.reset()
        sample=self.dataset._process_scene(self.raw(key),len(context),view_sampler=ExactSampler(context,target))
        batch=workers.collate([sample])
        for group in ['context','target']:
            depth=batch[group]['depth']
            if not torch.isfinite(depth).all() or depth.abs().max()>1e6:
                raise RuntimeError(f'Invalid normalized depth for {key}, {group}, context={context}, target={target}')
        return batch

    def sample(self,step,rank,world,views):
        # All ranks traverse the same epoch permutation at different positions.
        position=(step-1)*world+rank;epoch,index=divmod(position,len(self.train))
        order=np.random.default_rng(256+epoch).permutation(len(self.train));key=self.train[int(order[index])]
        np.random.seed((256+position*104729)%2**32);random.seed(256+position)
        for attempt in range(128):
            ids=self.dataset.view_sampler.sample(scene_id=key,num_context_views=views,num_target_views=4,total_num_views=self.raw(key)['meta']['num_views'])
            context=list(np.random.permutation(ids['context_indices']));target=ids['target_indices']
            try:batch=self.fixed(key,context,target)
            except ValueError as e:
                if 'pose jump ratio' not in str(e):raise
                continue
            return batch,dict(scene=key,context=list(map(int,context)),target=list(map(int,target)),rejected=attempt)
        raise RuntimeError(f'128 failed pose-filter samples for {key}; not silently dropping scene')

    def window(self,key):
        if key in self.plan:return self.plan[key]
        n=self.raw(key)['meta']['num_views']
        for a in sorted(range(n-18),key=lambda a:(abs(a-(n-19)//2),a)):
            entry=dict(context2=[a,a+18],context6=[a+i for i in [0,3,7,11,15,18]],target=[a+i for i in [2,4,6,8,10,12,14,16]])
            try:
                for v in [2,6]:self.fixed(key,entry[f'context{v}'],entry['target'])
            except ValueError as e:
                if 'pose jump ratio' not in str(e):raise
                continue
            self.plan[key]=entry;return entry
        raise RuntimeError('No valid evaluation window for '+key)
