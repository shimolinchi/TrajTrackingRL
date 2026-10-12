"""Opt-in allocator counters; no explicit CUDA synchronization or cache clearing."""
import json
import sys
import time
import os
from pathlib import Path

import torch


class MemoryBudgetError(RuntimeError):
    """Stop before an exhausted device corrupts the simulation."""


class MemoryTelemetry:
    def __init__(self, path, device, *, cuda=None, snapshot_update=-1, min_free_bytes=0):
        self.min_free_bytes = min_free_bytes
        self.path = Path(path)
        self.snapshot_update = snapshot_update
        self.device = torch.device(device)
        self.cuda = torch.cuda if cuda is None else cuda

    def __call__(self, stage, update, **details):
        if self.device.type != 'cuda':
            return
        try:
            s = self.cuda.memory_stats(self.device)
            row = dict(unix_seconds=time.time(), stage=stage, update=update,
                       pid=os.getpid(), details=details, device=str(self.device), allocator_backend=self.cuda.get_allocator_backend())
            for key in ('allocated_bytes.all.current', 'reserved_bytes.all.current',
                        'active_bytes.all.current', 'inactive_split_bytes.all.current',
                        'allocated_bytes.all.peak', 'reserved_bytes.all.peak',
                        'num_alloc_retries', 'num_ooms'):
                row[key] = s.get(key)
            free, total = self.cuda.mem_get_info(self.device)
            row.update(device_free_bytes=free, device_total_bytes=total,
                       device_used_minus_this_process_reserved_bytes=total-free-s.get('reserved_bytes.all.current',0))
            # Residual includes other processes/context allocations; NOT pure PhysX.
            with self.path.open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            if stage == "update_end" and update == self.snapshot_update:
                target = self.path.with_suffix(f".u{update}.snapshot.json")
                with target.open("x") as stream:
                    json.dump(self.cuda.memory_snapshot(), stream)
            if free < self.min_free_bytes:
                raise MemoryBudgetError(f"device free memory {free} below guard {self.min_free_bytes}; stage={stage} update={update}")
        except MemoryBudgetError:
            raise
        except Exception as error:
            print(f'[memory telemetry] {stage}: {error!r}', file=sys.stderr, flush=True)
