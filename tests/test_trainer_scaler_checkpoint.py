"""Regression: checkpoint resume preserves AMP scale and its growth tracker."""
import copy
import unittest
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
from splatfactory.trainer import Trainer


class ScalerCheckpointTest(unittest.TestCase):
    def make_trainer(self):
        t=Trainer.__new__(Trainer)
        t.model=torch.nn.Linear(2,1)
        t.optimizer=torch.optim.AdamW(t.model.parameters(),lr=1e-3)
        t.lr_scheduler=torch.optim.lr_scheduler.ConstantLR(t.optimizer,factor=1.)
        t.scaler=torch.amp.GradScaler('cpu',init_scale=16.,growth_interval=2)
        t.rank=0;t.distributed=False;t.epoch=0;t.tot_it=1;t.tot_n_samples=1
        return t

    def step(self,t):
        t.optimizer.zero_grad()
        t.scaler.scale(t.model(torch.ones(1,2)).square().sum()).backward()
        t.scaler.step(t.optimizer);t.scaler.update();t.lr_scheduler.step()

    def test_scaler_saved_and_restored_across_growth_boundary(self):
        a=self.make_trainer();self.step(a)
        with patch('splatfactory.trainer.experiments.save_experiment') as save:
            a.save_checkpoint(None,OmegaConf.create({}))
        custom=save.call_args.kwargs['custom']
        self.assertEqual(custom['scaler'],a.scaler.state_dict())
        checkpoint=copy.deepcopy(dict(model=a.model.state_dict(),optimizer=a.optimizer.state_dict(),
                                      lr_scheduler=a.lr_scheduler.state_dict(),epoch=0,**custom))
        b=self.make_trainer();b.load_checkpoint(checkpoint,strict=True,load_state=True)
        self.assertEqual(a.scaler.state_dict(),b.scaler.state_dict())
        self.step(a);self.step(b)
        self.assertEqual(a.scaler.get_scale(),32.)
        self.assertEqual(a.scaler.state_dict(),b.scaler.state_dict())
        for x,y in zip(a.model.parameters(),b.model.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
