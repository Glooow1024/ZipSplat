"""Paired scene initialization scale ablation; existing DA3 stays frozen.

Only the initial slot parameter is multiplied, once, before optimizer updates.
All arms use identical Gaussian directions, data, 1D RoPE, and learning rates.
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
from splatfactory.utils.mappings import batch_to_device
from tools.scene_token.train_initial_test import (
    build, update, evaluate, save_checkpoint, rng_state, set_rng, param_hash, dump,
)
from tools.scene_token.diagnose_layers import describe
from tools.scene_token.position_1d import ScenePosition1D
from tools.scene_token.validate_position_1d import trace_model

FACTORS = {'scale1': 1., 'scale4': 4., 'scale10': 10., 'scale16': 16.}


def fingerprint(model, batch, path, baseline=None):
    """Actual first-block tensors, plus image-response with fixed slot identities.

    Horizontal flip is an input sensitivity probe, never a quality evaluation.
    Hooks only observe; Q/K include the installed positional adapter.
    """
    state = rng_state()
    model.eval()
    data = batch_to_device(batch, 'cuda')
    bb = model.backbone.backbone
    start, end = bb.scene_start_idx, bb.patch_start_idx
    captured, handles = {}, []

    def output(name):
        def hook(_module, _args, value):
            captured[name] = value.detach().float().cpu()
        return hook

    def before(_module, args):
        captured['attention_state'] = args[0].detach().float().cpu()

    handles.append(bb.blocks[0].norm1.register_forward_hook(output('normalized_input')))
    handles.append(bb.blocks[0].attn.q_norm.register_forward_hook(output('q')))
    handles.append(bb.blocks[0].attn.k_norm.register_forward_hook(output('k')))
    handles.append(bb.blocks[0].attn.qkv.register_forward_hook(output('qkv')))
    handles.append(bb.blocks[0].norm2.register_forward_pre_hook(before))
    handles.append(bb.blocks[-1].register_forward_hook(output('last_state')))
    try:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(data)
        del pred
        original = dict(captured)
        altered = dict(data)
        altered['context'] = dict(data['context'])
        altered['context']['image'] = data['context']['image'].flip(-1)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(altered)
        del pred
        flipped = dict(captured)
    finally:
        for handle in handles:
            handle.remove()
        set_rng(state)
        model.train()

    result = {'groups': {}, 'image_response': {}, 'relative_to_scale1': {}}
    for group, sl in [('scene', slice(start, end)), ('patch', slice(end, None))]:
        z = original['normalized_input'][:, sl].double()
        q = original['q'][:, :, sl].double()
        k = original['k'][:, :, sl].double()
        v = original['qkv'].chunk(3, -1)[2][:, sl].double()
        result['groups'][group] = dict(
            normalized_input_rms=float(z.square().mean().sqrt()),
            q_head_norm_rms=float(q.square().sum(-1).mean().sqrt()),
            k_head_norm_rms=float(k.square().sum(-1).mean().sqrt()),
            v_full_norm_rms=float(v.square().sum(-1).mean().sqrt()),
        )
    for key in ['attention_state', 'last_state']:
        a = original[key].reshape(-1, end+324, bb.embed_dim)[:, start:end].double()
        b = flipped[key].reshape(-1, end+324, bb.embed_dim)[:, start:end].double()
        diff = b-a
        centered = lambda x: x-x.mean(-2, keepdim=True)
        result['image_response'][key] = dict(
            difference_rms=float(diff.square().mean().sqrt()),
            relative_difference=float(diff.norm()/a.norm()),
            centered_difference_rms=float(centered(diff).square().mean().sqrt()),
            centered_difference_over_original=float(centered(diff).norm()/centered(a).norm().clamp_min(1e-30)),
        )
    if baseline is not None:
        for key in ['normalized_input', 'q', 'k', 'qkv']:
            a, b = original[key].double(), baseline[key].double()
            result['relative_to_scale1'][key] = dict(
                relative_difference=float((a-b).norm()/b.norm()),
                max_abs=float((a-b).abs().max()),
            )
    result['note'] = 'Observed actual Q/K after 1D rotation. Flip probe changes RGB only; no training/quality claim.'
    dump(path, result)
    return {k: original[k] for k in ['normalized_input', 'q', 'k', 'qkv']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(256); np.random.seed(256); random.seed(256)
    old = json.loads((args.run/'config.json').read_text())
    dc = OmegaConf.create(old['data'])
    train = list(get_dataset(dc.name)(dc, split='train').get_loader(num_workers=0))
    vc = OmegaConf.merge(dc, dc.val_overrides)
    val = list(get_dataset(vc.name)(vc, split='val').get_loader(num_workers=0))
    batches = dict(train=[describe(b) for b in train], validation=[describe(b) for b in val])
    assert batches == json.loads((args.run/'batches.json').read_text())
    initial_rng = rng_state()
    protocol = dict(seed=256, factors=FACTORS, steps=200, conf=old, batches=batches,
        train_scope='200 updates on train[0], B1/V2/4 targets, per arm; fresh released weights',
        variable='Multiply the freshly initialized Gaussian scene parameter by factor ONCE; no forward gain',
        position='distinct 1D RoPE, base10000/scale1, all40 blocks, identical across arms',
        nominal_initial_std={k: .02*v for k, v in FACTORS.items()},
        fixed='S256, original DA3 frozen, identical data/LR/optimizer/loss/position/downstream initialization',
        validation='two fixed scenes, V6/8 held-out targets; GT relative cameras/depth scale; no alignment/TTO',
        limits='single seed and short fixed-batch pilot; larger parameter norms change relative optimizer step size')
    dump(args.output/'protocol.json', protocol)
    files = ['tools/scene_token/validate_initial_scale.py', 'tools/scene_token/validate_position_1d.py',
             'tools/scene_token/train_initial_test.py', 'tools/scene_token/diagnose_layers.py',
             'tools/scene_token/position_1d.py', 'splatfactory/models/modules/attention.py',
             'splatfactory/models/modules/transformer_block.py', 'splatfactory/models/encoders/dav3_dino.py',
             'splatfactory/models/networks/zipsplat.py']
    source = args.output/'source'; source.mkdir()
    for name in files:
        dst = source/name; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(Path(name).read_bytes())
    dump(args.output/'provenance.json', dict(
        head=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        status=subprocess.check_output(['git','status','--short'], text=True), torch=torch.__version__,
        files={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}))
    summary, hashes, frozen_hashes = {}, [], []
    reference_fingerprint = None
    for name, factor in FACTORS.items():
        output = args.output/name; output.mkdir()
        set_rng(initial_rng)
        model, trainer = build(old['model'], old['train'], True)
        hashes.append(param_hash(model, False)); frozen_hash = param_hash(model, True)
        frozen_hashes.append(frozen_hash)
        assert len(set(hashes)) == len(set(frozen_hashes)) == 1
        assert not trainer.optimizer.state
        slot = model.backbone.backbone.scene_tokens
        base = slot.detach().clone()
        with torch.no_grad(): slot.mul_(factor)
        assert torch.equal(slot, base*factor)
        scale_spec = dict(factor=factor, nominal_std=.02*factor,
            actual_initial_rms=float(slot.square().mean().sqrt()),
            initial_direction_cosine=float(torch.nn.functional.cosine_similarity(slot, base).mean()),
            exact_scaled_parameter_check=True, initial_parameters_before_scaling_sha256=hashes[-1])
        assert scale_spec['initial_direction_cosine'] > .99999
        del base
        adapter = ScenePosition1D(model, 'distinct')
        conf = {**old, 'experiment':adapter.specification(), 'scale_experiment':scale_spec,
                'scope':protocol['train_scope']}
        dump(output/'config.json', conf)
        evaluations, records = [], []

        def assess(step):
            folder = output/f'step{step:03d}'; folder.mkdir()
            (folder/'train').mkdir(); (folder/'validation').mkdir()
            tr = evaluate(model, [train[0]], folder/'train', 'after')
            va = evaluate(model, val, folder/'validation', 'after')
            evaluations.append(dict(step=step, train=tr, validation=va))
            dump(output/'evaluations.json', evaluations)
            print(json.dumps(dict(arm=name, step=step, train_psnr=tr['mean']['psnr'],
                val_psnr=va['mean']['psnr'], val_lpips=va['mean']['lpips'])), flush=True)
            if step in (0, 200):
                trace_model(model, train[0], folder/'train_trace.json')
                trace_model(model, val[0], folder/'val_trace.json')
                return fingerprint(model, train[0], folder/'first_block.json',
                    reference_fingerprint if step == 0 else None)

        first = assess(0)
        if name == 'scale1': reference_fingerprint = first
        del first
        torch.cuda.reset_peak_memory_stats()
        for step in range(1, 201):
            row = update(trainer, train[0], step); records.append(row)
            with (output/'steps.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            if step % 25 == 0:
                print(json.dumps(dict(arm=name, step=step, loss=row['metrics']['loss/total'])), flush=True)
            if step in (100, 200): assess(step)
        assert frozen_hash == param_hash(model, True)
        save_checkpoint(output/'checkpoint_200.pt', model, trainer, conf)
        summary[name] = dict(completed_steps=200, frozen_parameters_unchanged=True,
            initialization=scale_spec, initial=evaluations[0], final=evaluations[-1],
            checkpoint_bytes=(output/'checkpoint_200.pt').stat().st_size,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            median_update_seconds=float(np.median([r['seconds'] for r in records[10:]])))
        dump(args.output/'summary.json', summary)
        print('COMPLETE '+name, flush=True)
        adapter.remove(); del adapter, trainer, model, slot
        gc.collect(); torch.cuda.empty_cache()
    dump(args.output/'completion.json', dict(completed=True, arms=4, total_optimizer_updates=800,
        initial_parameters_equal_before_scaling=True, frozen_parameters_equal_across_arms=True))


if __name__ == '__main__': main()
