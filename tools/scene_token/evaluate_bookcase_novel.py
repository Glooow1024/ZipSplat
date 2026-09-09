"""Read-only near/extrapolated unsupervised views from three std.02 checkpoints."""
import argparse,gc,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from splatfactory.models.networks.zipsplat import ZipSplat
from tools.scene_token.multiscene_data import PilotData
from tools.scene_token.position_1d import ScenePosition1D
from tools.scene_token.audit_repeatability import controlled_runtime
from tools.scene_token.train_initial_test import evaluate,dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    conf=json.loads((a.source/'seed256/scale1/config.json').read_text())
    data=PilotData(conf['data']);key=next(k for k in data.raw if k.startswith('7e6564c7'))
    context=[186,194];supervised=[187,189,190,193]
    sets=dict(supervised=supervised,unseen_interpolation=[188,191,192],unseen_extrapolation=[178,180,182,184,196,198,200,202])
    assert not set(context+supervised).intersection(sets['unseen_interpolation']+sets['unseen_extrapolation'])
    batches={name:[data.fixed(key,context,target)] for name,target in sets.items()}
    dump(a.output/'protocol.json',dict(scene=key,context=context,sets=sets,optimizer_updates=0,
        exclusion='Held out from this fixed-bookcase fine-tuning; no claim about released pretraining membership',
        sampling='Frame-index interpolation/extrapolation, not a guarantee of visibility overlap; fixed GT camera/context scale, no target optimization'))
    expected=json.loads((a.source/'validation.json').read_text())
    result={}
    for seed in [256,257,258]:
        path=a.source/f'seed{seed}/scale1/checkpoint_5000.pt'
        with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        assert sha==expected['arms'][f'{seed}/scale1']['checkpoint_sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False)
        mc=dict(saved['conf']['model']);mc.update(weights=None,scene_token_init_from_base=False)
        torch.use_deterministic_algorithms(False);model=ZipSplat(mc).cuda().eval()
        spec=saved['conf']['experiment'];adapter=ScenePosition1D(model,spec['mode'],scale=spec['scale'],base=spec['base'])
        model.load_state_dict(saved['model'],strict=True);del saved;controlled_runtime(model)
        out=a.output/f'seed{seed}';out.mkdir();result[str(seed)]={}
        for name,batch in batches.items():
            folder=out/name;folder.mkdir();result[str(seed)][name]=evaluate(model,batch,folder,'after')
        old=json.loads((a.source/f'seed{seed}/scale1/evaluations.json').read_text())[-1]['train']['mean']
        errors={m:abs(old[m]-result[str(seed)]['supervised']['mean'][m]) for m in ['psnr','total','lpips']}
        assert max(errors.values())<1e-3,errors
        result[str(seed)]['checkpoint_sha256']=sha;result[str(seed)]['supervised_replay_error']=errors
        dump(a.output/'results.json',result)
        print(json.dumps({seed:{n:{m:result[str(seed)][n]['mean'][m] for m in ['psnr','lpips']} for n in sets}}),flush=True)
        adapter.remove();del model,adapter;gc.collect();torch.cuda.empty_cache()
    dump(a.output/'completion.json',dict(passed=True,checkpoints=3,optimizer_updates=0))


if __name__=='__main__':main()
