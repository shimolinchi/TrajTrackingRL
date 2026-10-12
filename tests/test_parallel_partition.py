"""Partitioning preserves the physical object mixture and exact held-out IDs."""

from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from trajectory_tracking.data.partition import (
    EnvironmentPartition, evaluation_batches, evaluation_owners, local_scene, partition_environments,
)


@dataclass(frozen=True)
class Scene:
    object_override: object = None
    object_variants: tuple = ()
    object_layout: tuple = ()


def sources(uids):
    return tuple(SimpleNamespace(entry=SimpleNamespace(replay_audit={"object_uid": uid})) for uid in uids)


def test_object_counts_and_all_global_slots_preserved():
    variants = tuple(Scene(SimpleNamespace(object_uid=uid)) for uid in ("a", "b", "c"))
    scene = Scene(object_variants=variants, object_layout=(0, 2, 0, 1, 2))
    data = sources(("c", "a", "b", "a", "c", "b"))
    parts = partition_environments(scene, data, total_envs=24, world_size=8)
    assert sorted(i for p in parts for i in p.global_environment_ids) == list(range(24))
    expected = Counter(("a", "c", "a", "b", "c")[i % 5] for i in range(24))
    assert Counter(uid for p in parts for uid in p.object_uids) == expected
    for part in parts:
        local = local_scene(scene, part)
        assert tuple(local.object_variants[i].object_override.object_uid
                     for i in local.object_layout) == part.object_uids
        for i, item in enumerate(data):
            assert (i in part.source_indices) == (item.entry.replay_audit["object_uid"] in part.object_uids)
        assert len(part.manifest()["sha256"]) == 64


def test_homogeneous_bank_keeps_uniform_sampling():
    parts = partition_environments(Scene(), sources(("a",) * 7), total_envs=32, world_size=8)
    assert all(part.source_indices == tuple(range(7)) for part in parts)
    with pytest.raises(ValueError, match="divide"):
        partition_environments(Scene(), sources(("a",)), total_envs=31, world_size=8)


def test_test_rows_once_with_asset_compatible_unscored_fillers():
    slots = ("a", "a", "b")
    uids = ("a", "b", "a", "b", "b", "a", "a")
    seen = []
    for indices, scored in evaluation_batches(slots, uids, tuple(range(len(uids)))):
        assert tuple(uids[i] for i in indices) == slots
        seen.extend(indices[i] for i in scored)
    assert sorted(seen) == list(range(len(uids)))
    assert list(evaluation_batches(slots, uids, ())) == []
    with pytest.raises(ValueError, match="no compatible"):
        list(evaluation_batches(("a",), uids, (1,)))


def test_boundary_object_evaluation_respects_slot_capacity():
    parts = (EnvironmentPartition(tuple(range(8)), ("a",) * 7 + ("b",), ()),
             EnvironmentPartition(tuple(range(8, 16)), ("a",) + ("b",) * 7, ()))
    owners = evaluation_owners(parts, ("a",) * 8 + ("b",) * 8)
    assert Counter(owners[:8]) == {0: 7, 1: 1}
    assert Counter(owners[8:]) == {0: 1, 1: 7}


def test_seven_workers_each_keep_8192_environment_slots():
    names = tuple(f"object_{i:02d}" for i in range(52))
    variants = tuple(Scene(SimpleNamespace(object_uid=uid)) for uid in names)
    scene = Scene(object_variants=variants, object_layout=tuple(range(52)))
    data = sources(tuple(names[i % 52] for i in range(1129)))
    parts = partition_environments(scene, data, total_envs=57344, world_size=7)
    assert len(parts) == 7
    assert all(len(part.global_environment_ids) == 8192 for part in parts)
    assert sorted(i for part in parts for i in part.global_environment_ids) == list(range(57344))
    assert Counter(uid for part in parts for uid in part.object_uids) == Counter(names[i % 52] for i in range(57344))
    assert set(i for part in parts for i in part.source_indices) == set(range(1129))


@pytest.mark.parametrize('world_size', [7, 8])
def test_complete_heldout_603_with_six_train_only_singletons(world_size):
    from trajectory_tracking.data.partition import resident_evaluation_sources
    names = tuple(f'object_{i:03d}' for i in range(113))
    test_names = tuple(names[i % 107] for i in range(603))
    train_names = tuple(names[i % 107] for i in range(2396)) + names[107:]
    train, test = sources(train_names), sources(test_names)
    scene = Scene(object_variants=tuple(Scene(SimpleNamespace(object_uid=n)) for n in names),
                  object_layout=tuple(names.index(n) for n in test_names + names[107:]))
    partitions = partition_environments(scene, train, total_envs=world_size * 8192, world_size=world_size)
    owners = evaluation_owners(partitions, test_names)
    scored_global, fillers = [], set()
    for rank, partition in enumerate(partitions):
        local_train = tuple(train[i] for i in partition.source_indices)
        local, indices = resident_evaluation_sources(test, local_train, set(partition.object_uids))
        uids = tuple(s.entry.replay_audit['object_uid'] for s in local)
        fillers.update(uid for uid, index in zip(uids, indices) if index < 0)
        owned = tuple(i for i,g in enumerate(indices) if g >= 0 and owners[g] == rank)
        for rows, scored in evaluation_batches(partition.object_uids, uids, owned):
            assert all(uids[r] == uid for r,uid in zip(rows, partition.object_uids))
            scored_global.extend(indices[rows[i]] for i in scored)
    assert sorted(scored_global) == list(range(603))
    assert fillers == set(names[107:])
