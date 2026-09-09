"""Tiny NCCL/BF16 collective check; no model, dataset, or optimizer."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
dist.init_process_group('nccl', timeout=timedelta(seconds=60))
try:
    rank, world = dist.get_rank(), dist.get_world_size()
    value = torch.tensor([float(rank + 1)], device='cuda', dtype=torch.bfloat16)
    dist.all_reduce(value)
    expected = world * (world + 1) / 2
    assert value.item() == expected, (rank, value.item(), expected)
    records = [None] * world
    dist.all_gather_object(records, {
        'rank': rank, 'device': torch.cuda.get_device_name(),
        'bf16_supported': torch.cuda.is_bf16_supported(), 'sum': value.item(),
    })
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = {'world_size': world, 'backend': 'nccl', 'torch': torch.__version__,
                  'records': records, 'scope': 'one tiny collective; not model DDP/backward or throughput'}
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
finally:
    dist.destroy_process_group()
