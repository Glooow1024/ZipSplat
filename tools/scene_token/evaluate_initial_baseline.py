"""Two-scene original-checkpoint reference at matched GS budget and cameras."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from omegaconf import OmegaConf
from splatfactory.datasets import get_dataset
from splatfactory.models.networks.zipsplat import ZipSplat
from splatfactory.utils.mappings import batch_to_device
from tools.scene_token.train_initial_test import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(256)
    conf=json.loads((a.run/'config.json').read_text())
    dc=OmegaConf.create(conf['data']);dc=OmegaConf.merge(dc,dc.val_overrides)
    batches=list(get_dataset(dc.name)(dc,split='val').get_loader(num_workers=0))
    assert len(batches)==2
    mc=conf['model'];mc.update(scene_tokens_enabled=False,freeze_backbone_except_scene=False,
                              scene_token_init_from_base=False,weights='zipsplat',query_sample_ratio=[256/324,256/324])
    model=ZipSplat(mc).cuda().eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        pred=model(batch_to_device(batches[0],'cuda'))
    assert pred['gaussians'].num_gaussians==6*256*32
    del pred
    out=a.run/'baseline_matched_budget';out.mkdir(exist_ok=False)
    result=evaluate(model,batches,out,'after')
    result.update(context_views=6,target_views=8,gaussians=6*256*32,
                  compression_ratio=256/324,checkpoint='released zipsplat-da3g-252p',
                  camera_protocol=conf['validation_protocol'])
    (a.run/'baseline_matched_budget.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['mean'],indent=2))


if __name__=='__main__':main()
