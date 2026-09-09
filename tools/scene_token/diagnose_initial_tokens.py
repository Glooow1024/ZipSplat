"""Inspect fixed training reconstruction and fused-slot diversity after pilot training."""
import argparse,gc,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from splatfactory.datasets import get_dataset
from splatfactory.models.networks.zipsplat import ZipSplat
from splatfactory.utils.mappings import batch_to_device


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4)
    conf=json.loads((a.run/'config.json').read_text())
    entry=json.loads((a.run/'batches.json').read_text())['train'][0]
    key=entry['scene'].removeprefix('dl3dv-')
    index={key:{k:[int(i) for i in entry[k]] for k in ['context','target']}}
    ip=a.run/'diagnostic_train_indices.json';ip.write_text(json.dumps(index))
    dc=OmegaConf.create(conf['data'])
    dc.test_shard_dir=dc.train_shard_dir;dc.random_reference_view=False
    dc.view_sampler={'name':'eval_sampler','indices_file':str(ip)}
    batch=next(iter(get_dataset(dc.name)(dc,split='val').get_loader(num_workers=0)))
    assert batch['context']['index'].flatten().tolist()==entry['context']
    data=batch_to_device(batch,'cuda')
    output=a.run/'fixed_training_diagnostic';output.mkdir(exist_ok=False)
    records=[]
    for label in ['step100','step200','baseline']:
        torch.manual_seed(256)
        mc=dict(conf['model'])
        mc.update(weights='zipsplat' if label in ['initial','baseline'] else None,
                  scene_token_init_from_base=label=='initial')
        if label=='baseline':
            mc.update(scene_tokens_enabled=False,freeze_backbone_except_scene=False,query_sample_ratio=[256/324,256/324])
        model=ZipSplat(mc).cuda().eval()
        if label.startswith('step'):
            cp=torch.load(a.run/f'checkpoint_{label[4:]}.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(cp['model'],strict=True);del cp
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            pred=model(data);losses,metrics=model.loss_metrics(pred,data)
        x=pred['scene_tokens'][0].float()
        normalized=torch.nn.functional.normalize(x,dim=-1)
        n=x.shape[0];mean_cos=float(((normalized@normalized.T).sum()-n)/(n*(n-1)))
        eigen=torch.linalg.svdvals(x-x.mean(0,keepdim=True)).square()
        weights=eigen/eigen.sum().clamp_min(1e-20)
        entropy_rank=float(torch.exp(-(weights*weights.clamp_min(1e-20).log()).sum()))
        gs=pred['gaussians'];scales=gs.scales.float()
        record=dict(label=label,token_count=n,mean_pairwise_cosine=mean_cos,
                    centered_effective_rank=entropy_rank,
                    relative_token_variation=float((x-x.mean(0)).norm()/x.norm()),
                    gs_scale_median=float(scales.median()),gs_scale_p90=float(scales.quantile(.9)),
                    total_loss=float(losses['total'].mean()),psnr=float(metrics['psnr'].mean()))
        records.append(record);print(json.dumps(record),flush=True)
        for j in range(4):
            for tag,tensor in [('gt',data['target']['image']),('render',pred['target_rgb'])]:
                im=(tensor[0,j].float().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
                Image.fromarray(im).save(output/f'{label}_{j}_{tag}.png')
        del model,pred,losses,metrics,x,normalized,gs,scales,eigen,weights
        gc.collect();torch.cuda.empty_cache()
    (a.run/'token_diagnostic.json').write_text(json.dumps(dict(scene=entry,records=records,
        note='Fused tokens, centered entropy effective rank. High similarity is a diagnostic, not proof of causation.'),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(12,6))
    for row,j in enumerate([0,2]):
        for col,(label,tag,title) in enumerate([('step100','gt','Training GT'),('step100','render','After 100 fixed-batch steps'),('step200','render','After 200 steps'),('baseline','render','Original at matched GS count')]):
            axes[row,col].imshow(Image.open(output/f'{label}_{j}_{tag}.png'));axes[row,col].axis('off');axes[row,col].set_title(title,fontsize=10)
    fig.suptitle('Same training sample, input images, target images and GT cameras; no pose fitting')
    fig.tight_layout();fig.savefig(a.run/'fixed_training_comparison.png',dpi=130)


if __name__=='__main__':main()
