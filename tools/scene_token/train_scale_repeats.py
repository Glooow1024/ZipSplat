"""One paired seed: std .02 vs .08, 1000 fixed-batch updates each.

Run three workers with seeds 256/257/258. Existing DA3 remains frozen.
PyTorch deterministic gradients are enabled, but custom renderer reductions may
still differ: the companion audit measures this explicitly.
"""
import argparse
import gc
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from omegaconf import OmegaConf
from splatfactory.datasets import get_dataset
from tools.scene_token.train_initial_test import (
    build, update, evaluate, save_checkpoint, rng_state, set_rng, param_hash, dump,
)
from tools.scene_token.diagnose_layers import describe
from tools.scene_token.position_1d import ScenePosition1D
from tools.scene_token.validate_position_1d import trace_model
from tools.scene_token.audit_repeatability import controlled_runtime


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=[256,257,258], required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(256); np.random.seed(256); random.seed(256)
    old = json.loads((args.run/'config.json').read_text()); dc = OmegaConf.create(old['data'])
    train = list(get_dataset(dc.name)(dc, split='train').get_loader(num_workers=0))
    vc = OmegaConf.merge(dc, dc.val_overrides)
    val = list(get_dataset(vc.name)(vc, split='val').get_loader(num_workers=0))
    batches = dict(train=[describe(b) for b in train], validation=[describe(b) for b in val])
    assert batches == json.loads((args.run/'batches.json').read_text())
    # Preserve the historical initialization for seed256, vary model init only
    # for other seeds; never let seed changes alter sampled training images.
    if args.seed != 256:
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    initial_rng = rng_state()
    train_conf = dict(old['train']); train_conf['num_steps'] = 1000
    protocol = dict(seed=args.seed, steps=1000, factors={'scale1':1., 'scale4':4.}, batches=batches,
        conf={**old, 'train':train_conf},
        initialization='Paired Gaussian directions per seed; seed256 replays historical data RNG consumption; others reseed after fixed data load',
        scope='All1000 updates on same train[0]; B1/V2/4 targets; fresh released weights; no DDP',
        controlled_runtime='cuDNN benchmark=False, deterministic=True; torch deterministic algorithms=True; CUBLAS_WORKSPACE_CONFIG=:4096:8',
        exemption='Detached GaussianHead.metrics logging only; no loss/gradient exemption',
        limitation='Custom gsplat kernels are outside PyTorch deterministic enforcement; audit records residual backward variation',
        validation='two held-out scenes V6/8 targets; GT cameras/context-depth scale; no pose refinement/TTO')
    dump(args.output/'protocol.json', protocol)
    files = ['tools/scene_token/train_scale_repeats.py','tools/scene_token/audit_repeatability.py',
             'tools/scene_token/train_initial_test.py','tools/scene_token/validate_position_1d.py',
             'tools/scene_token/diagnose_layers.py','tools/scene_token/position_1d.py',
             'splatfactory/models/networks/zipsplat.py','splatfactory/models/encoders/dav3_dino.py',
             'splatfactory/models/decoders/gaussian_head.py','splatfactory/trainer.py']
    for name in files:
        dst = args.output/'source'/name; dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Path(name).read_bytes())
    dump(args.output/'provenance.json', dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        status=subprocess.check_output(['git','status','--short'],text=True), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0), files={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}))
    hashes = []; frozen_hashes = []; summary = {}
    for name, factor in protocol['factors'].items():
        output = args.output/name; output.mkdir()
        torch.use_deterministic_algorithms(False); set_rng(initial_rng)
        model, trainer = build(old['model'], train_conf, True)
        hashes.append(param_hash(model, False)); frozen_hash = param_hash(model, True); frozen_hashes.append(frozen_hash)
        assert len(set(hashes)) == len(set(frozen_hashes)) == 1
        slot = model.backbone.backbone.scene_tokens
        with torch.no_grad():
            base = slot.clone(); slot.mul_(factor)
            assert torch.equal(slot, base*factor)
        del base
        adapter = ScenePosition1D(model, 'distinct'); controlled_runtime(model)
        conf = {**old, 'train':train_conf, 'experiment':adapter.specification(),
            'scale_experiment':dict(factor=factor, nominal_std=.02*factor),
            'repetition_experiment':dict(seed=args.seed, steps=1000, controlled_runtime=True),
            'scope':protocol['scope']}
        dump(output/'config.json', conf)
        evaluations = []; rows = []

        def assess(step):
            folder = output/f'step{step:04d}'; folder.mkdir()
            (folder/'train').mkdir(); (folder/'validation').mkdir()
            tr = evaluate(model, [train[0]], folder/'train', 'after')
            va = evaluate(model, val, folder/'validation', 'after')
            evaluations.append(dict(step=step, train=tr, validation=va))
            dump(output/'evaluations.json', evaluations)
            print(json.dumps(dict(seed=args.seed, arm=name, step=step, train_psnr=tr['mean']['psnr'],
                                 val_psnr=va['mean']['psnr'], val_lpips=va['mean']['lpips'])), flush=True)
            if step in (0, 1000): trace_model(model, train[0], folder/'train_trace.json')
            if step == 1000: trace_model(model, val[0], folder/'val_trace.json')

        assess(0); torch.cuda.reset_peak_memory_stats()
        for step in range(1, 1001):
            before = slot.detach().clone()
            row = update(trainer, train[0], step)
            with torch.no_grad():
                row['scene_parameter_relative_update'] = float((slot-before).norm()/before.norm())
                row['scene_parameter_rms'] = float(slot.square().mean().sqrt())
            del before
            rows.append(row)
            with (output/'steps.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            if step % 100 == 0:
                print(json.dumps(dict(seed=args.seed, arm=name, step=step, loss=row['metrics']['loss/total'])), flush=True)
            if step in (200, 500, 1000): assess(step)
        assert frozen_hash == param_hash(model, True)
        save_checkpoint(output/'checkpoint_1000.pt', model, trainer, conf)
        summary[name] = dict(seed=args.seed, completed_steps=1000, initial=evaluations[0], final=evaluations[-1],
            initial_parameters_before_scaling_sha256=hashes[-1], frozen_parameters_sha256=frozen_hash,
            frozen_parameters_unchanged=True, median_update_seconds=float(np.median([r['seconds'] for r in rows[10:]])),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            checkpoint_bytes=(output/'checkpoint_1000.pt').stat().st_size)
        dump(args.output/'summary.json', summary)
        print('COMPLETE '+name+' seed'+str(args.seed), flush=True)
        adapter.remove(); del model, trainer, adapter, slot
        gc.collect(); torch.cuda.empty_cache()
    dump(args.output/'completion.json', dict(completed=True, seed=args.seed, optimizer_updates=2000,
        paired_parameters_equal_before_scaling=True, original_backbone_frozen=True))


if __name__ == '__main__': main()
