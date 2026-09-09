"""CPU-only checks for optimizer migration/resume and real data preprocessing."""
import argparse,json,tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from tools.scene_token.train_main import migrate_optimizer,optimizer_names
from tools.scene_token.training_run_data import TrainingData
from tools.scene_token.training_monitor import atomic,publish

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4)
    model=torch.nn.ModuleDict({'old':torch.nn.Linear(3,2),'new':torch.nn.Linear(3,2)})
    model['new'].requires_grad_(False);old=torch.optim.AdamW(model['old'].parameters(),lr=.003)
    x=torch.tensor([[.1,.2,.3]])
    model['old'](x).square().mean().backward();old.step();old.zero_grad()
    saved=dict(optimizer=old.state_dict(),optimizer_names=optimizer_names(model,old))
    model['new'].requires_grad_(True)
    current=torch.optim.AdamW([dict(params=model['new'].parameters(),lr=.0001),dict(params=model['old'].parameters(),lr=.002)])
    assert migrate_optimizer(model,current,saved)==2
    for parameter in model['old'].parameters():
        for key,value in old.state[parameter].items():assert torch.equal(value,current.state[parameter][key])
    assert all(p not in current.state for p in model['new'].parameters())
    assert [g['lr'] for g in current.param_groups]==[.0001,.002]
    # Full optimizer serialization preserves state and the next update on CPU.
    clone=torch.nn.ModuleDict({'old':torch.nn.Linear(3,2),'new':torch.nn.Linear(3,2)});clone.load_state_dict(model.state_dict())
    other=torch.optim.AdamW([dict(params=clone['new'].parameters(),lr=.0001),dict(params=clone['old'].parameters(),lr=.002)])
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'state.pt';torch.save(current.state_dict(),path);other.load_state_dict(torch.load(path,weights_only=False))
        for m,opt in [(model,current),(clone,other)]:
            (m['old'](x)+m['new'](x)).square().mean().backward();opt.step();opt.zero_grad()
        assert all(torch.equal(p,q) for p,q in zip(model.parameters(),clone.parameters()))
        root=Path(tmp)/'report';atomic(root/'metrics.jsonl',{})
        (root/'metrics.jsonl').write_text(json.dumps(dict(step=1,loss=.4))+'\n'+json.dumps(dict(step=2,loss=.2))+'\n')
        publish(root,dict(state='running',step=2));report=json.loads((root/'dashboard.json').read_text())
        assert abs(report['train'][0]['loss']-.3)<1e-9 and report['status']['step']==2
    conf=json.loads(a.config.read_text());data=TrainingData(conf['data']['dataset_dir'],conf['data']);samples=[]
    for views in [2,3,4,5,6]:
        batch,selection=data.sample(views,0,8,views)
        assert batch['context']['image'].shape[:2]==(1,views) and batch['target']['image'].shape[:2]==(1,4)
        assert len(set(selection['context']))==views
        samples.append(dict(views=views,**selection))
    for key in data.val:
        w=data.window(key);assert not set(w['target'])&(set(w['context2'])|set(w['context6']))
    atomic(a.output,dict(passed=True,device='CPU only; no model training',optimizer_moments_preserved=True,new_group_has_no_old_state=True,next_cpu_update_exact_after_serialization=True,data_view_counts=[2,3,4,5,6],samples=samples,validation_windows=len(data.val),dashboard_json=True))
    print(a.output.read_text())
if __name__=='__main__':main()
