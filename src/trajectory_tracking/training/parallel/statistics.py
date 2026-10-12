"""Global batch moments, not averages of rank-local means or lifetime counters."""

import torch


def moments(values, context, *, unbiased=False):
    values = values.double()
    n = context.reduce(values.new_tensor(values.shape[0]))
    if n <= 0 or (unbiased and n <= 1):
        raise ValueError("insufficient global samples for variance")
    mean = context.reduce(values.sum(0)) / n
    # Two-pass variance avoids catastrophic cancellation for large coordinates.
    m2 = context.reduce((values - mean).square().sum(0))
    return mean, m2 / (n - int(unbiased)), n


@torch.no_grad()
def merge_running(model, values, prefix, context, *, benchmark):
    mean, variance, count = (getattr(model, prefix + suffix)
                             for suffix in ("_mean", "_variance", "_count"))
    batch_mean, batch_variance, n = moments(values, context)
    if n > 1:
        batch_variance *= n / (n - 1)
    elif benchmark:
        raise ValueError("benchmark normalization requires two global samples")
    # Preserve each learner's batch-statistic precision and merge order.
    if not benchmark:
        batch_mean = batch_mean.to(values.dtype)
        batch_variance = batch_variance.to(values.dtype)
    delta, total = batch_mean - mean, count + n
    combined = (variance * count + batch_variance * n
                + delta.square() * count * n / total) / total
    mean.add_(delta * n / total)
    variance.copy_(combined.clamp_min(1e-6) if benchmark else combined)
    count.copy_(total)
