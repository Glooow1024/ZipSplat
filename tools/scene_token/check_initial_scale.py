"""Check scale-run records and fresh-model strict checkpoint inference."""
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from omegaconf import OmegaConf
from splatfactory.datasets import get_dataset
from splatfactory.models.networks.zipsplat import ZipSplat
from splatfactory.utils.mappings import batch_to_device
from tools.scene_token.position_1d import ScenePosition1D


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args(); root = args.input
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    # Match Trainer.setup_torch(): cuDNN's default algorithm can differ in BF16.
    torch.backends.cudnn.benchmark = True
    protocol = json.loads((root/'protocol.json').read_text())
    completion = json.loads((root/'completion.json').read_text())
    assert completion['total_optimizer_updates'] == 800 and completion['completed']
    entry = protocol['batches']['train'][0]
    indices = root/'reload_indices.json'
    indices.write_text(json.dumps({entry['scene'].removeprefix('dl3dv-'):
        {k:[int(i) for i in entry[k]] for k in ['context','target']}}))
    dc = OmegaConf.create(protocol['conf']['data']); dc.test_shard_dir = dc.train_shard_dir
    dc.random_reference_view = False
    dc.view_sampler = {'name':'eval_sampler', 'indices_file':str(indices)}
    batch = next(iter(get_dataset(dc.name)(dc, split='val').get_loader(num_workers=0)))
    for key in ['context','target']: assert batch[key]['index'].flatten().tolist() == entry[key]
    data = batch_to_device(batch, 'cuda'); results = {}; reference_lrs = None
    for name, factor in protocol['factors'].items():
        records = [json.loads(line) for line in (root/name/'steps.jsonl').read_text().splitlines()]
        assert [r['step'] for r in records] == list(range(1,201))
        lrs = [r['lr'] for r in records]
        if reference_lrs is None: reference_lrs = lrs
        assert lrs == reference_lrs
        assert all(all(np.isfinite(v) and v > 0 for v in r['gradients'].values()) for r in records)
        for step in [0,200]:
            for sample in ['train','val']:
                trace = json.loads((root/name/f'step{step:03d}'/f'{sample}_trace.json').read_text())
                assert all(v == 0 for v in trace['hook_parity_max_abs'].values())
                for stats in trace['attention'].values():
                    if 'key_mass' in stats: assert abs(sum(stats['key_mass'].values())-1) < 1e-5
        path = root/name/'checkpoint_200.pt'
        with path.open('rb') as f: sha = hashlib.file_digest(f, 'sha256').hexdigest()
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['tot_it'] == 200 and checkpoint['conf']['scale_experiment']['factor'] == factor
        spec = checkpoint['conf']['experiment']; assert spec['mode'] == 'distinct'
        conf = dict(checkpoint['conf']['model']); conf.update(weights=None, scene_token_init_from_base=False)
        model = ZipSplat(conf).cuda().eval()
        marker = 'backbone.backbone.scene_position_experiment_spec'
        try: model.load_state_dict(checkpoint['model'], strict=True)
        except RuntimeError as error: assert marker in str(error)
        else: raise AssertionError('Unconfigured model accepted positional checkpoint')
        adapter = ScenePosition1D(model, spec['mode'], scale=spec['scale'], base=spec['base'])
        assert torch.equal(checkpoint['model'][marker], model.state_dict()[marker].cpu())
        # Initial scale is already represented by learned parameter values: do not reapply it.
        model.load_state_dict(checkpoint['model'], strict=True); del checkpoint
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(data); losses, metrics = model.loss_metrics(pred, data)
        recorded = json.loads((root/name/'evaluations.json').read_text())[-1]['train']['mean']
        loss_error = abs(float(losses['total'].mean())-recorded['total'])
        psnr_error = abs(float(metrics['psnr'].mean())-recorded['psnr'])
        print(json.dumps(dict(arm=name, loss_error=loss_error, psnr_error=psnr_error,
            actual_loss=float(losses['total'].mean()), actual_psnr=float(metrics['psnr'].mean()),
            expected_loss=recorded['total'], expected_psnr=recorded['psnr'])), flush=True)
        assert loss_error < 1e-3 and psnr_error < 1e-2, (name, loss_error, psnr_error)
        results[name] = dict(checkpoint_sha256=sha, configured_strict_restore=True,
            unconfigured_restore_rejected=True, loss_abs_error=loss_error, psnr_abs_error=psnr_error,
            optimizer_steps=200, gradients_finite_nonzero=True, hook_parity=True)
        print(json.dumps({name:results[name]}), flush=True)
        adapter.remove(); del adapter, model, pred, losses, metrics
        gc.collect(); torch.cuda.empty_cache()
    (root/'validation.json').write_text(json.dumps(dict(passed=True, arms=results,
        lr_schedules_identical=True, cudnn_benchmark_matches_trainer=True,
        note='Fresh inference restoration; no extra optimizer resume updates'), indent=2)+'\n')


if __name__ == '__main__': main()
