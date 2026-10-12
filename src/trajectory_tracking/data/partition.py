"""Verified source/asset views for local environments; no distributed imports.

Partition global environment slots, not uniformly shuffled trajectory rows:
multi-object resets must retain the original per-object sampling probabilities.
"""

from dataclasses import dataclass, replace
from collections import Counter, deque
from hashlib import sha256
import json


@dataclass(frozen=True)
class EnvironmentPartition:
    global_environment_ids: tuple[int, ...]
    object_uids: tuple[str, ...]
    source_indices: tuple[int, ...]

    def manifest(self):
        payload = {"global_environment_ids": list(self.global_environment_ids),
                   "object_uids": list(self.object_uids),
                   "source_indices": list(self.source_indices)}
        payload["sha256"] = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return payload


def source_object_uid(source):
    audit = getattr(source.entry, "replay_audit", {})
    return audit.get("object_uid")


def partition_environments(source, sources, *, total_envs, world_size):
    if total_envs <= 0 or world_size <= 0 or total_envs % world_size:
        raise ValueError("global environments must divide evenly over workers")
    capacity = total_envs // world_size
    if not source.object_variants:
        # A homogeneous scene retains the complete uniform trajectory bank on
        # each worker. Sharding rows equally could bias unequal-sized shards.
        return tuple(EnvironmentPartition(tuple(range(r * capacity, (r + 1) * capacity)),
                                          (), tuple(range(len(sources))))
                     for r in range(world_size))
    variants = [v.object_override.object_uid for v in source.object_variants]
    layout = [variants[source.object_layout[i % len(source.object_layout)]] for i in range(total_envs)]
    identities = [source_object_uid(item) for item in sources]
    if None in identities or set(layout) != set(identities):
        raise ValueError("global layout must cover exactly the admitted training objects")
    # Group existing slots by asset to minimize each worker's bank. Counts are
    # unchanged: global env IDs explicitly record this deterministic permutation.
    order = sorted(range(total_envs), key=lambda index: (layout[index], index))
    result = []
    for rank in range(world_size):
        ids = tuple(order[rank * capacity:(rank + 1) * capacity])
        uids = tuple(layout[index] for index in ids)
        indices = tuple(i for i, uid in enumerate(identities) if uid in set(uids))
        result.append(EnvironmentPartition(ids, uids, indices))
    return tuple(result)


def local_scene(source, partition):
    if not partition.object_uids:
        return source
    variants = tuple(v for v in source.object_variants
                     if v.object_override.object_uid in set(partition.object_uids))
    lookup = {v.object_override.object_uid: i for i, v in enumerate(variants)}
    return replace(variants[0], object_variants=variants,
                   object_layout=tuple(lookup[uid] for uid in partition.object_uids))


def local_data_config(config, sources):
    """Only for already verified explicit sources; never re-resolve this view.

    The global config and admission hash stay authoritative. The runtime view
    gives the environment its actual bank size and is saved separately.
    """
    if not sources:
        raise ValueError("empty local bank")
    return replace(config, data=replace(config.data, trajectory_count=len(sources),
                                       expected_episode_uid=sources[0].entry.episode_uid))


def evaluation_batches(environment_uids, trajectory_uids, owned_indices):
    """Assign each owned row once to a matching physical slot; fill others safely.

    Returns (all physical trajectory indices, scored slot indices). Duplicate
    filler rows never enter the scoring denominator.
    """
    candidates = {}
    for index, uid in enumerate(trajectory_uids):
        candidates.setdefault(uid, []).append(index)
    pending = {uid: deque() for uid in candidates}
    for index in owned_indices:
        pending[trajectory_uids[index]].append(index)
    if set(environment_uids) - candidates.keys():
        raise ValueError("evaluation bank lacks an environment's object")
    if any(pending[uid] and uid not in environment_uids for uid in pending):
        raise ValueError("evaluation row has no compatible physical slot")
    while any(pending.values()):
        rows, scored = [], []
        for slot, uid in enumerate(environment_uids):
            if pending[uid]:
                rows.append(pending[uid].popleft())
                scored.append(slot)
            else:
                rows.append(candidates[uid][0])
        yield tuple(rows), tuple(scored)


def evaluation_owners(partitions, trajectory_uids):
    """Balance test work by resident same-object capacity, not equal rank votes."""
    capacities = [Counter(part.object_uids or (None,) * len(part.global_environment_ids))
                  for part in partitions]
    loads = [Counter() for _ in partitions]
    owners = []
    for uid in trajectory_uids:
        eligible = [rank for rank, capacity in enumerate(capacities) if capacity[uid]]
        if not eligible:
            raise ValueError("held-out object has no matching training scene")
        owner = min(eligible, key=lambda rank: ((loads[rank][uid] + 1) / capacities[rank][uid], rank))
        loads[owner][uid] += 1
        owners.append(owner)
    return tuple(owners)


def resident_evaluation_sources(test_sources, training_sources, resident):
    """Keep train-only assets as unscored fillers without manufacturing test IDs."""
    indices = [i for i, source in enumerate(test_sources) if source_object_uid(source) in resident]
    sources = [test_sources[i] for i in indices]
    missing = resident - {source_object_uid(source) for source in sources}
    for source in training_sources:
        uid = source_object_uid(source)
        if uid in missing:
            sources.append(source)
            indices.append(-1)  # Never an owned test identity.
            missing.remove(uid)
    if missing:
        raise ValueError("resident object has neither test data nor a training filler")
    return tuple(sources), tuple(indices)
