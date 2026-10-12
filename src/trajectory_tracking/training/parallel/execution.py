"""Mechanisms shared by PPO recipes; formulas and scheduling stay in the learner."""

from dataclasses import fields, replace

import torch

from .batches import GlobalBatchPlan
from .statistics import merge_running, moments


class PPOExecution:
    def __init__(self, context, *, global_environment_ids=None):
        self.context = context
        self.environment_ids = global_environment_ids
        self.sample_keys = None

    def set_rollout_order(self, valid, *, environment_major):
        if self.environment_ids is None:
            self.sample_keys = None
            return
        steps, local_envs = valid.shape
        ids = torch.tensor(self.environment_ids, device=valid.device, dtype=torch.long)
        if ids.shape != (local_envs,):
            raise ValueError("global environment IDs do not match local rollout slots")
        time = torch.arange(steps, device=valid.device)[:, None]
        if environment_major:
            keys = ids[None, :] * steps + time
            self.sample_keys = keys.T.reshape(-1)[valid.T.reshape(-1)]
        else:
            keys = time * (local_envs * self.context.world_size) + ids[None, :]
            self.sample_keys = keys.reshape(-1)[valid.reshape(-1)]

    def prepare(self, batch):
        plan = GlobalBatchPlan(batch.batch_size, self.context, self.sample_keys)
        self.sample_keys = None
        if batch.batch_size == 0:
            batch = replace(batch, **{
                field.name: getattr(batch, field.name).new_zeros(
                    (1,) + getattr(batch, field.name).shape[1:])
                for field in fields(batch)
            })
        return batch, plan

    def merge(self, model, values, prefix, *, benchmark):
        merge_running(model, values, prefix, self.context, benchmark=benchmark)

    def advantages(self, values, *, unbiased):
        mean, variance, _ = moments(values, self.context, unbiased=unbiased)
        return (values - mean.to(values.dtype)) / (variance.sqrt().to(values.dtype) + 1e-8)

    def weighted(self, value, count):
        tensor = torch.as_tensor(value, dtype=torch.float64, device=self.context.device)
        packed = torch.stack((tensor * count, tensor.new_tensor(count)))
        total = self.context.reduce(packed)
        return float(total[0] / total[1])

    def backward(self, loss, weight, model):
        valid = torch.isfinite(loss).to(torch.int32)
        if not self.context.reduce(valid, "min"):
            raise RuntimeError("non-finite PPO loss on a distributed worker")
        (loss * weight).backward()
        self.context.synchronize_gradients(model.parameters())

    def explained_variance(self, returns, old_values):
        _, variance, _ = moments(returns, self.context)
        _, error_variance, _ = moments(returns - old_values, self.context)
        return float(1 - error_variance / variance) if variance > 1e-12 else float("nan")

    def totals(self, values, count):
        return self.context.reduce(values) / self.context.reduce(
            torch.tensor(float(count), dtype=torch.float64, device=values.device))
