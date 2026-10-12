"""Global minibatches represented by local indices; no rollout all-gather."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalMinibatch:
    indices: torch.Tensor
    count: int
    global_count: int

    @property
    def weight(self):
        return self.count / self.global_count


class GlobalBatchPlan:
    def __init__(self, local_count, context, sample_keys=None):
        counts = context.gather(int(local_count))
        if any(n < 0 for n in counts) or sum(counts) < 2:
            raise ValueError("PPO needs at least two global valid transitions")
        self.start = sum(counts[:context.rank])
        self.stop = self.start + local_count
        self.size = sum(counts)
        self.device = context.device
        positions = torch.arange(self.start, self.stop, device=self.device)
        if sample_keys is not None:
            if sample_keys.shape != (local_count,) or sample_keys.dtype != torch.long:
                raise ValueError("sample keys must be int64 [local valid count]")
            keys = context.gather_vector(sample_keys, counts)
            if keys.unique().numel() != self.size:
                raise ValueError("global rollout sample IDs must be unique")
            order = keys.argsort()
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(self.size, device=self.device)
            positions = inverse[self.start:self.stop]
        self.lookup = torch.full((self.size,), -1, dtype=torch.long, device=self.device)
        self.lookup[positions] = torch.arange(local_count, device=self.device)

    def batches(self, *, generator=None, count=None, batch_size=None):
        if (count is None) == (batch_size is None):
            raise ValueError("choose minibatch count or global minibatch size")
        order = (torch.randperm(self.size, device=self.device, generator=generator)
                 if generator is not None else torch.arange(self.size, device=self.device))
        chunks = (torch.tensor_split(order, min(count, self.size)) if count is not None
                  else order.split(batch_size))
        for chunk in chunks:
            local = self.lookup[chunk]
            local = local[local >= 0]
            n = local.numel()
            # A zero-weight finite dummy participates in backward on empty ranks.
            if n == 0:
                local = torch.zeros(1, dtype=torch.long, device=self.device)
            yield LocalMinibatch(local, n, chunk.numel())
