"""Reduce sufficient statistics before converting them to rates or RMS values."""

from dataclasses import fields

import torch


def merge_rollout_metrics(metrics, context):
    names = [field.name for field in fields(metrics)
             if isinstance(getattr(metrics, field.name), torch.Tensor)
             and field.name not in ("flow_error_max", "last_flow_error")]
    sums = context.reduce(torch.stack([getattr(metrics, name) for name in names]))
    for name, value in zip(names, sums, strict=True):
        setattr(metrics, name, value)
    metrics.flow_error_max = context.reduce(metrics.flow_error_max, "max")
    # No completed episode: use the pooled rollout error, not an unweighted
    # average of ranks' last nonempty steps. Explicitly documented fallback.
    metrics.last_flow_error = metrics.flow_error / metrics.valid_samples.clamp_min(1)
    for name in ("action_smoothness", "joint_acceleration"):
        accumulator = getattr(metrics, name)
        shape = None if accumulator.sums is None else tuple(accumulator.sums.shape)
        shapes = context.gather(shape)
        present = [item for item in shapes if item is not None]
        if not present:
            continue
        if any(item != present[0] for item in present):
            raise ValueError(f"incompatible diagnostic shapes: {name}")
        if accumulator.sums is None:
            accumulator.sums = torch.zeros(present[0], dtype=torch.float64, device=context.device)
        accumulator.sums = context.reduce(accumulator.sums)
