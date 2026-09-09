"""Repeated VJPs on identical graphs, before any optimizer updates.

Separate the feature path from rendering/losses; do not assume PyTorch's
deterministic switch covers custom CUDA extensions such as gsplat.
"""
import argparse
import gc
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from omegaconf import OmegaConf
from splatfactory.datasets import get_dataset
from splatfactory.utils.mappings import batch_to_device
from tools.scene_token.train_initial_test import build, rng_state, set_rng, param_hash, dump
from tools.scene_token.position_1d import ScenePosition1D
from tools.scene_token.diagnose_layers import describe


def difference(a, b):
    a, b = a.double(), b.double()
    return dict(equal=torch.equal(a, b), max_abs=float((a-b).abs().max()),
                relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-30)))


def controlled_runtime(model):
    """Enable deterministic PyTorch gradients, exempt no-grad logging only."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    original = model.gaussian_head.metrics

    @torch.no_grad()
    def logging_metrics(*args, **kwargs):
        enabled = torch.are_deterministic_algorithms_enabled()
        warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            # CUDA median-with-indices is used only for detached health logs.
            torch.use_deterministic_algorithms(False)
            return original(*args, **kwargs)
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn)
    model.gaussian_head.metrics = logging_metrics


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4); torch.manual_seed(256); np.random.seed(256); random.seed(256)
    conf = json.loads((args.run/'config.json').read_text()); dc = OmegaConf.create(conf['data'])
    train = list(get_dataset(dc.name)(dc, split='train').get_loader(num_workers=0))
    vc = OmegaConf.merge(dc, dc.val_overrides)
    val = list(get_dataset(vc.name)(vc, split='val').get_loader(num_workers=0))
    assert dict(train=[describe(b) for b in train], validation=[describe(b) for b in val]) == json.loads((args.run/'batches.json').read_text())
    initial = rng_state(); result = {}
    data = batch_to_device(train[0], 'cuda')
    for mode in ['native', 'controlled']:
        torch.use_deterministic_algorithms(False)
        set_rng(initial); model, trainer = build(conf['model'], conf['train'], True)
        adapter = ScenePosition1D(model, 'distinct')
        before_hash = param_hash(model, False)
        torch.backends.cudnn.benchmark = mode == 'native'
        torch.backends.cudnn.deterministic = mode == 'controlled'
        torch.use_deterministic_algorithms(mode == 'controlled')
        if mode == 'controlled': controlled_runtime(model)
        record = dict(trainable_parameter_sha256=before_hash, optimizer_updates=0,
                      cudnn_benchmark=torch.backends.cudnn.benchmark,
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(data); losses, metrics = model.loss_metrics(pred, data)
                feature_loss = pred['scene_tokens'].float().square().mean()
            tracked = dict(scene_parameter=model.backbone.backbone.scene_tokens,
                           scene_features=pred['scene_tokens'],
                           rendered_rgb=pred['target_rgb'])
            record['loss'] = float(losses['total'].detach())
            for label, loss, keys in [('full_loss', losses['total'], list(tracked)),
                                      ('render_only', pred['target_rgb'].float().square().mean(),
                                       ['scene_parameter','scene_features']),
                                      ('feature_only', feature_loss, ['scene_parameter'])]:
                baseline = None; pairs = []
                for repeat in range(3):
                    grads = torch.autograd.grad(loss, [tracked[k] for k in keys], retain_graph=True)
                    values = {k:g.detach().float().cpu() for k,g in zip(keys,grads)}
                    assert all(torch.isfinite(g).all() for g in values.values())
                    if baseline is None: baseline = values
                    else: pairs.append({k:difference(baseline[k], values[k]) for k in keys})
                record[label] = pairs
            record['completed'] = True
            del pred, losses, metrics, feature_loss, tracked, grads, values, baseline
        except RuntimeError as error:
            record.update(completed=False, error=str(error))
        assert before_hash == param_hash(model, False)
        result[mode] = record; dump(args.output/'audit.json', result)
        print(json.dumps({mode:record}), flush=True)
        adapter.remove(); del model, trainer, adapter
        gc.collect(); torch.cuda.empty_cache()
    dump(args.output/'protocol.json', dict(conf=conf, sample=describe(train[0]),
        repetitions=3, graph='Same graph, same upstream loss, repeated autograd.grad; no parameter updates',
        limitation='Controlled mode may not cover custom CUDA kernels; no claim of cross-process reproducibility',
        exemption='Only GaussianHead.metrics no-grad logging runs with deterministic flag temporarily disabled'))


if __name__ == '__main__': main()
