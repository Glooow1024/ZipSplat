"""Validate official activated-mask loss semantics on a tiny synthetic scene."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from splatfactory.gaussians import Gaussians
from splatfactory.geometry import Camera, Pose
from splatfactory.models.decoders.gaussian_head import GaussianHead


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.manual_seed(256)
    torch.set_num_threads(4)
    device='cuda'
    means=torch.tensor([[[0.,0.,2.],[0.,0.,-2.],[0.3,0.1,2.]]],device=device,requires_grad=True)
    scales=torch.full((1,3,3),0.1,device=device,requires_grad=True)
    quats=torch.tensor([[[1.,0.,0.,0.]]*3],device=device,requires_grad=True)
    opacity=torch.full((1,3),0.5,device=device,requires_grad=True)
    sh=torch.full((1,3,1,3),0.1,device=device,requires_grad=True)
    gs=Gaussians.from_parameters(means,scales,quats,opacity,sh)
    k=torch.tensor([[[[30.,0.,16.],[0.,30.,16.],[0.,0.,1.]]]],device=device)
    pose=Pose.from_4x4mat(torch.eye(4,device=device)[None,None])
    camera=Camera.from_calibration_matrix(k)
    view={'image':torch.rand(1,1,3,32,32,device=device),'depth':torch.full((1,1,32,32),2.,device=device),
          'camera':camera,'pose':pose,'depth_mask':torch.ones(1,1,32,32,device=device,dtype=torch.bool)}
    # Synthetic component diagnostic, not a held-out reconstruction experiment.
    data={'context':view,'target':view}
    head=GaussianHead({'embed_dim':16,'gaussians_per_token':2,'random_background':False}).cuda()
    pred=head._render_results(gs,data)|{'gaussians':gs}
    activated=pred['activated_gaussians']
    assert activated.shape==(1,3)
    assert activated[0,0]==1 and activated[0,1]==0 and activated[0,2]==1,activated
    losses,metrics=head.loss(pred,data)
    assert all(torch.isfinite(t).all() for t in losses.values())
    geom_grad=torch.autograd.grad(losses['location_loss'].sum(),means,retain_graph=True)[0]
    assert geom_grad[0,[0,2]].abs().max()==0,geom_grad
    assert geom_grad[0,1].abs().sum()>0,geom_grad
    losses['total'].mean().backward()
    for name,p in [('means',means),('scales',scales),('opacity',opacity),('sh',sh)]:
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    result={'scope':'32px synthetic GaussianHead RGB/LPIPS/depth/Chamfer forward/backward; no model training or optimizer',
            'gsplat_version':metadata.version('gsplat'),'activated':activated.detach().cpu().tolist(),
            'location_gradient':geom_grad.detach().cpu().tolist(),
            'losses':{k:v.detach().cpu().tolist() for k,v in losses.items()},
            'finite_backward':True,'activated_detach_semantics_passed':True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
