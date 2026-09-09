"""Real CPU DDP: a zero local gradient can produce a valid shared update."""
import os,tempfile,unittest
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tools.scene_token.train_main import validate_gradient_observation

def worker(rank,rendezvous,output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        model=torch.nn.Linear(1,1,bias=False);model.weight.data.fill_(1.)
        ddp=torch.nn.parallel.DistributedDataParallel(model);opt=torch.optim.SGD(model.parameters(),lr=.1)
        local={};applied={}
        handle=model.weight.register_hook(lambda g:local.update(weight=float(g.norm())))
        oh=opt.register_step_pre_hook(lambda *_:applied.update(weight=float(model.weight.grad.norm())))
        ddp(torch.tensor([[float(rank)]])).square().sum().backward();opt.step()
        validate_gradient_observation(local,applied,['weight'])
        assert local['weight']==(0. if rank==0 else 2.) and applied['weight']==1.
        assert abs(model.weight.item()-.9)<1e-6
        Path(output,f'rank{rank}.ok').write_text('passed')
        handle.remove();oh.remove()
    finally:dist.destroy_process_group()

class GradientMonitorTests(unittest.TestCase):
    def test_real_ddp_accepts_local_zero_and_observes_reduced_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            rendezvous=Path(tmp,'rendezvous').as_uri()
            mp.spawn(worker,args=(rendezvous,tmp),nprocs=2,join=True)
            self.assertEqual(len(list(Path(tmp).glob('rank*.ok'))),2)

    def test_missing_nonfinite_and_global_zero_remain_errors(self):
        for local,applied in [({},{'w':1.}),({'w':1.},{}),({'w':float('nan')},{'w':1.}),({'w':0.},{'w':0.}),({'w':1.},{'w':float('inf')})]:
            with self.subTest(local=local,applied=applied),self.assertRaises(AssertionError):validate_gradient_observation(local,applied,['w'])

if __name__=='__main__':unittest.main()
