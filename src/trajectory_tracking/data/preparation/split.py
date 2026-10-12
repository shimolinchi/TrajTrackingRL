"""Freeze a whole-episode train/test split of replay-admitted numeric windows.

Writes a sibling selection, never modifies the source selection or its caches.
Attempts, raw episodes and exact object pose tracks cannot cross the boundary.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np

VERSION = "whole_episode_grouped_train_test_v1"


def grouped_partition(rows, identities, *, seed, test_fraction):
    if not 0 < test_fraction < 1 or len(rows) < 2:
        raise ValueError("split requires two or more episodes and a valid fraction")
    parents = list(range(len(rows)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for identity in identities[row["episode_uid"]]:
            if identity in seen:
                parents[root(i)] = root(seen[identity])
            seen[identity] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(root(i), []).append(row)
    ordered = sorted(groups.values(), key=lambda g: min(r["episode_uid"] for r in g))
    random.Random(seed).shuffle(ordered)
    target = max(1, round(len(rows) * test_fraction))
    test, train = [], []
    for group in ordered:
        (test if len(test) < target else train).extend(
            sorted(group, key=lambda r: r["episode_uid"])
        )
    if not train or not test:
        raise ValueError("duplicate groups leave no disjoint train/test partition")
    return train, test, len(groups)


def freeze(source: Path, output: Path, *, dataset: Path, seed=42, test_fraction=0.2):
    source, output = source.resolve(), output.resolve()
    if source.parent != output.parent or source == output:
        raise ValueError("split must be a new sibling of the source selection")
    if output.exists():
        raise FileExistsError(output)
    original = source.read_bytes()
    payload = json.loads(original)
    rows = payload["episodes"]
    if len({r["episode_uid"] for r in rows}) != len(rows):
        raise ValueError("duplicate episode UID")
    identities, source_maps, map_hashes = {}, {}, {}
    for row in rows:
        shard_name = row["shard_relative"]
        if shard_name not in source_maps:
            mapping_path = dataset / "success/shards" / shard_name / "source_episode_table.jsonl"
            map_hashes[shard_name] = sha256(mapping_path.read_bytes()).hexdigest()
            mapping = {}
            with mapping_path.open() as stream:
                for line in stream:
                    item = json.loads(line)
                    mapping[item["derived_slot"]] = (
                        item["source_run_uid"], item["source_shard"], item["source_slot"]
                    )
            source_maps[shard_name] = mapping
        raw_identity = source_maps[shard_name][row["source_episode"]]
        cache_path = (source.parent / row["cache_relative"]).resolve()
        if not cache_path.is_relative_to(source.parent):
            raise ValueError("cache escapes source directory")
        if sha256(cache_path.read_bytes()).hexdigest() != row["cache_sha256"]:
            raise ValueError("numeric cache checksum changed")
        with np.load(cache_path, allow_pickle=False) as cache:
            poses = np.ascontiguousarray(cache["object_root_current_pose"])
            flow_hash = sha256(
                str((poses.shape, poses.dtype.str)).encode() + poses.tobytes()
            ).hexdigest()
        identities[row["episode_uid"]] = [
            "attempt:" + row["attempt_uid"],
            "raw:" + ":".join(map(str, raw_identity)),
            "object_pose_track:" + flow_hash,
        ]
    train, test, group_count = grouped_partition(
        rows, identities, seed=seed, test_fraction=test_fraction
    )
    split = {
        "protocol": VERSION, "seed": seed, "requested_test_fraction": test_fraction,
        "source_selection": source.name,
        "source_selection_sha256": sha256(original).hexdigest(),
        "train_count": len(train), "test_count": len(test),
        "independent_group_count": group_count,
        "group_identities": identities,
        "source_episode_table_sha256": map_hashes,
        "ordering": "train_groups_then_test_groups; sorted_groups_python_random_shuffle",
        "scope": "held_out_episodes_same_cube_publication_not_unseen_objects",
        "legacy_policy_warning": "exp93-101 trained on the unsplit pool; never call their results held-out",
    }
    result = {**payload, "episodes": train + test, "split": split,
              "ordering": split["ordering"]}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return {"path": str(output), "sha256": sha256(output.read_bytes()).hexdigest(),
            "train_count": len(train), "test_count": len(test),
            "group_count": group_count,
            "train_first_uid": train[0]["episode_uid"],
            "test_first_uid": test[0]["episode_uid"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()
    print(json.dumps(freeze(args.source, args.output, dataset=args.dataset, seed=args.seed,
                            test_fraction=args.test_fraction), indent=2))


if __name__ == "__main__":
    main()
