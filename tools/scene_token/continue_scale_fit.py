"""Resume one paired-scale checkpoint without restarting optimizer or warmup."""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from omegaconf import OmegaConf
from splatfactory.datasets import get_dataset
from tools.scene_token.train_initial_test import build, update, evaluate, save_checkpoint, set_rng, param_hash, dump
from tools.scene_token.position_1d import ScenePosition1D
from tools.scene_token.audit_repeatability import controlled_runtime
from tools.scene_token.diagnose_layers import describe
from tools.scene_token.validate_position_1d import trace_model


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--end-step', type=int, default=5000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start_time = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(256); np.random.seed(256); random.seed(256)
    protocol = json.loads((args.source.parent/'protocol.json').read_text())
    # Verify training implementation still matches the source experiment.
    provenance = json.loads((args.source.parent/'provenance.json').read_text())
    for name, sha in provenance['files'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, name
    dc = OmegaConf.create(protocol['conf']['data'])
    train = list(get_dataset(dc.name)(dc, split='train').get_loader(num_workers=0))
    vc = OmegaConf.merge(dc, dc.val_overrides)
    val = list(get_dataset(vc.name)(vc, split='val').get_loader(num_workers=0))
    batches = dict(train=[describe(b) for b in train], validation=[describe(b) for b in val])
    assert batches == protocol['batches']
    source_checkpoint = args.source/'checkpoint_1000.pt'
    with source_checkpoint.open('rb') as f:
        source_sha = hashlib.file_digest(f, 'sha256').hexdigest()
    known = json.loads((args.source.parent.parent/'validation.json').read_text())
    key = f"{protocol['seed']}/{args.source.name}"
    assert known['passed'] and known['arms'][key]['checkpoint_sha256'] == source_sha
    saved = torch.load(source_checkpoint, map_location='cpu', weights_only=False)
    assert saved['tot_it'] == 1000 and args.end_step > 1000
    conf = saved['conf']; conf['train'] = dict(conf['train'], num_steps=args.end_step)
    conf['continuation_experiment'] = dict(source=str(source_checkpoint), source_sha256=source_sha,
        start_step=1000, end_step=args.end_step, change='Total training duration only; full optimizer/scheduler/scaler/RNG restore')
    model, trainer = build(conf['model'], conf['train'], False)
    spec = conf['experiment']
    adapter = ScenePosition1D(model, spec['mode'], scale=spec['scale'], base=spec['base'])
    trainer.load_checkpoint(saved, strict=True, load_state=True)
    assert equal_state(model.state_dict(), saved['model'])
    assert equal_state(trainer.optimizer.state_dict(), saved['optimizer'])
    assert equal_state(trainer.lr_scheduler.state_dict(), saved['lr_scheduler'])
    assert equal_state(trainer.scaler.state_dict(), saved['scaler'])
    assert trainer.tot_it == trainer.tot_n_samples == 1000
    set_rng(saved['rng']); del saved
    controlled_runtime(model)
    frozen_hash = param_hash(model, True)
    old_summary = json.loads((args.source.parent/'summary.json').read_text())[args.source.name]
    assert frozen_hash == old_summary['frozen_parameters_sha256']
    initial_lrs = [g['lr'] for g in trainer.optimizer.param_groups]
    dump(args.output/'config.json', conf)
    dump(args.output/'protocol.json', dict(seed=protocol['seed'], arm=args.source.name,
        batches=batches, source=str(args.source), start_step=1000, end_step=args.end_step,
        checkpoints='Final only; prior checkpoint retained; evaluations at1000/2000/3000/5000',
        scope='Same fixed bookcase batch; original DA3 frozen; no new data or loss or learning-rate change'))
    source = args.output/'source'; source.mkdir()
    files = list(provenance['files']) + ['tools/scene_token/continue_scale_fit.py']
    for name in files:
        dest = source/name; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(Path(name).read_bytes())
    dump(args.output/'provenance.json', dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        status=subprocess.check_output(['git','status','--short'],text=True), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(), source_checkpoint_sha256=source_sha,
        source_implementation_unchanged=True, files={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}))
    evaluations = []; rows = []

    def assess(step):
        folder = args.output/f'step{step:04d}'; folder.mkdir()
        (folder/'train').mkdir(); (folder/'validation').mkdir()
        tr = evaluate(model, [train[0]], folder/'train', 'after')
        va = evaluate(model, val, folder/'validation', 'after')
        evaluations.append(dict(step=step, train=tr, validation=va))
        dump(args.output/'evaluations.json', evaluations)
        print(json.dumps(dict(key=key, step=step, train=tr['mean'], validation=va['mean'])), flush=True)
        if step == args.end_step:
            trace_model(model, train[0], folder/'train_trace.json')
            trace_model(model, val[0], folder/'val_trace.json')

    assess(1000)
    expected = json.loads((args.source/'evaluations.json').read_text())[-1]
    errors = {f'{split}/{metric}': abs(evaluations[0][split]['mean'][metric]-expected[split]['mean'][metric])
              for split in ['train','validation'] for metric in ['total','psnr','lpips']}
    assert all(v < (1e-2 if k.endswith('psnr') else 1e-3) for k,v in errors.items()), errors
    dump(args.output/'resume_check.json', dict(passed=True, full_model_optimizer_scheduler_scaler_equal=True,
        restored_rng=True, source_step=1000, metric_errors=errors, initial_lrs=initial_lrs,
        limitation='Checks restoration, not identical continuation across independent runs'))
    slot = model.backbone.backbone.scene_tokens
    torch.cuda.reset_peak_memory_stats()
    for step in range(1001, args.end_step+1):
        before = slot.detach().clone()
        row = update(trainer, train[0], step)
        with torch.no_grad():
            row['scene_parameter_relative_update'] = float((slot-before).norm()/before.norm())
            row['scene_parameter_rms'] = float(slot.square().mean().sqrt())
        del before
        assert row['lr'] == initial_lrs, 'Unexpected learning-rate change'
        rows.append(row)
        with (args.output/'steps.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        if step % 100 == 0:
            print(json.dumps(dict(key=key, step=step, loss=row['metrics']['loss/total'])), flush=True)
        if step in [2000,3000,args.end_step]: assess(step)
    assert frozen_hash == param_hash(model, True)
    path = args.output/f'checkpoint_{args.end_step}.pt'
    save_checkpoint(path, model, trainer, conf)
    with path.open('rb') as f: final_sha = hashlib.file_digest(f,'sha256').hexdigest()
    dump(args.output/'completion.json', dict(completed=True, source_step=1000, final_step=args.end_step,
        optimizer_updates=len(rows), frozen_parameters_unchanged=True, frozen_parameters_sha256=frozen_hash,
        checkpoint_sha256=final_sha, checkpoint_bytes=path.stat().st_size,
        median_update_seconds=float(np.median([r['seconds'] for r in rows[10:]])),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, wall_seconds=time.time()-start_time))
    print('COMPLETE '+key, flush=True)


if __name__ == '__main__':
    main()
