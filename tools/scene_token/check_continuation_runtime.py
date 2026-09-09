"""CPU checks: per-group LR floors, moment migration, exact resumed next update."""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from tools.scene_token.train_main import migrate_optimizer,optimizer_names
from tools.scene_token.continuation_schedule import stage_factor,check_parent_evaluation
from tools.scene_token.training_monitor import atomic
from splatfactory.utils.tools import pack_lr_parameters

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(2)
    model=torch.nn.ModuleDict({'existing':torch.nn.Linear(3,2),'fresh':torch.nn.Linear(3,2)})
    x=torch.tensor([[.2,.7,-.4]]);old=torch.optim.AdamW(model['existing'].parameters(),lr=3e-5)
    for _ in range(3):old.zero_grad();model['existing'](x).square().sum().backward();old.step()
    saved=copy.deepcopy(dict(optimizer=old.state_dict(),optimizer_names=optimizer_names(model,old)))
    opt=torch.optim.AdamW([{'params':model['fresh'].parameters(),'lr':1e-6},{'params':model['existing'].parameters(),'lr':3e-5}])
    schedule=lambda e:stage_factor(e,100000,1000,.1)
    sch=torch.optim.lr_scheduler.LambdaLR(opt,schedule)
    assert migrate_optimizer(model,opt,saved)==2
    assert [g['lr'] for g in opt.param_groups]==[1e-7,3e-6]
    assert all(p not in opt.state for p in model['fresh'].parameters())
    assert stage_factor(0,100000,1000,.1)==.1 and stage_factor(1000,100000,1000,.1)==1.
    assert stage_factor(100000,100000,1000,.1)==.1 and stage_factor(110000,100000,1000,.1)==.1
    assert all(stage_factor(i,100000,1000,.1)>=.1 for i in range(100001))
    for _ in range(5):opt.zero_grad();(model['existing'](x)+model['fresh'](x)).square().sum().backward();opt.step();sch.step()
    other=copy.deepcopy(model);o2=torch.optim.AdamW([{'params':other['fresh'].parameters(),'lr':1e-6},{'params':other['existing'].parameters(),'lr':3e-5}]);s2=torch.optim.lr_scheduler.LambdaLR(o2,schedule)
    o2.load_state_dict(copy.deepcopy(opt.state_dict()));s2.load_state_dict(copy.deepcopy(sch.state_dict()))
    for m,o,s in [(model,opt,sch),(other,o2,s2)]:o.zero_grad();(m['existing'](x)+m['fresh'](x)).square().sum().backward();o.step();s.step()
    assert all(torch.equal(p,q) for p,q in zip(model.parameters(),other.parameters())) and opt.param_groups[0]['lr']==o2.param_groups[0]['lr']
    names=['backbone.backbone.scene_tokens','scene_color_query.weight','backbone.backbone.blocks.18.attn.qkv.weight','backbone.backbone.blocks.29.attn.qkv.weight','backbone.backbone.blocks.30.attn.qkv.weight','head.weight']
    params=[(n,torch.nn.Parameter(torch.ones(1))) for n in names]
    groups=pack_lr_parameters(params,3e-5,{'scene_tokens':10/3,'scene_color_query':10/3,'+'.join(f'backbone.backbone.blocks.{i}.' for i in range(18,30)):1/30,'backbone':.1})
    found={id(p):g['lr'] for g in groups for p in g['params']}
    expected=[1e-4,1e-4,1e-6,1e-6,3e-6,3e-5]
    assert all(abs(found[id(p)]-v)<1e-12 for (_,p),v in zip(params,expected))
    row=dict(scene='a',context=[1,2],targets=[3],psnr=20.,lpips=.3,total=.1);sets={k:dict(scenes=[row]) for k in ['train_probe','val_v2','val_v6']}
    assert check_parent_evaluation({'sets':sets},sets)['passed']
    bad=copy.deepcopy(sets);bad['val_v6']['scenes'][0]['targets']=[4]
    try:check_parent_evaluation({'sets':sets},bad)
    except AssertionError:pass
    else:raise AssertionError('Changed evaluation target was not detected')
    atomic(a.output,dict(passed=True,device='CPU',moments_migrated=True,new_layers_start_without_moments=True,next_update_after_restore_exact=True,relative_floor_all_groups=True,per_name_learning_rates=dict(zip(names,expected)),changed_evaluation_rejected=True))
    print(a.output.read_text())

if __name__=='__main__':main()
