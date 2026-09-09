"""Fresh-model strict restore and fixed-input inference check for all three arms."""
import argparse,gc,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from omegaconf import OmegaConf
from splatfactory.datasets import get_dataset
from splatfactory.models.networks.zipsplat import ZipSplat
from splatfactory.utils.mappings import batch_to_device
from tools.scene_token.position_1d import ScenePosition1D


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    protocol=json.loads((a.input/'protocol.json').read_text());entry=protocol['batches']['train'][0]
    indices=a.input/'reload_indices.json'
    indices.write_text(json.dumps({entry['scene'].removeprefix('dl3dv-'):{k:[int(i) for i in entry[k]] for k in ['context','target']}}))
    dc=OmegaConf.create(protocol['conf']['data']);dc.test_shard_dir=dc.train_shard_dir;dc.random_reference_view=False
    dc.view_sampler={'name':'eval_sampler','indices_file':str(indices)}
    batch=next(iter(get_dataset(dc.name)(dc,split='val').get_loader(num_workers=0)))
    assert batch['context']['index'].flatten().tolist()==entry['context']
    assert batch['target']['index'].flatten().tolist()==entry['target']
    data=batch_to_device(batch,'cuda');result={}
    for mode in ['none','shared','distinct']:
        checkpoint=torch.load(a.input/mode/'checkpoint_200.pt',map_location='cpu',weights_only=False)
        spec=checkpoint['conf']['experiment'];assert spec['mode']==mode and spec['version']==1
        conf=dict(checkpoint['conf']['model']);conf.update(weights=None,scene_token_init_from_base=False)
        model=ZipSplat(conf).cuda().eval()
        marker='backbone.backbone.scene_position_experiment_spec'
        try:model.load_state_dict(checkpoint['model'],strict=True)
        except RuntimeError as e:assert marker in str(e)
        else:raise AssertionError('An unconfigured model silently accepted the experiment checkpoint')
        adapter=ScenePosition1D(model,mode,scale=spec['scale'],base=spec['base'])
        assert torch.equal(checkpoint['model'][marker],model.state_dict()[marker].cpu())
        model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            pred=model(data);loss,metric=model.loss_metrics(pred,data)
        recorded=json.loads((a.input/mode/'evaluations.json').read_text())[-1]['train']['mean']
        error_loss=abs(float(loss['total'].mean())-recorded['total'])
        error_psnr=abs(float(metric['psnr'].mean())-recorded['psnr'])
        assert error_loss<1e-3 and error_psnr<1e-2,(mode,error_loss,error_psnr)
        result[mode]=dict(unconfigured_strict_load_rejected=True,configured_strict_load_passed=True,
            loss_abs_error=error_loss,psnr_abs_error=error_psnr,
            note='Fresh model inference restore, not an additional optimizer resume test')
        print(json.dumps({mode:result[mode]}),flush=True)
        adapter.remove();del adapter,model,pred,loss,metric;gc.collect();torch.cuda.empty_cache()
    (a.input/'reload_check.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
