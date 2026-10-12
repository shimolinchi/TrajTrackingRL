"""Optional frozen train/test admission; no simulator or learner dependency."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

VERSION = "whole_episode_grouped_train_test_v1"


def heldout_test_config(config, project_root):
    """Accept only an exact training partition and derive its disjoint test bank.

    All training task fields remain untouched. The selection SHA binds both
    partitions into the checkpoint contract; changing membership is not resume.
    """
    path = config.data.selection_path(Path(project_root))
    encoded = path.read_bytes()
    if sha256(encoded).hexdigest() != config.data.selection_sha256:
        raise ValueError("split selection checksum changed")
    payload = json.loads(encoded)
    split = payload.get("split", {})
    if split.get("protocol") != VERSION:
        raise ValueError("held-out evaluation requires a frozen split selection")
    source_name = split["source_selection"]
    if Path(source_name).name != source_name:
        raise ValueError("split source must be a sibling filename")
    original = (path.parent / source_name).read_bytes()
    if sha256(original).hexdigest() != split["source_selection_sha256"]:
        raise ValueError("split source checksum changed")
    source_rows = json.loads(original)["episodes"]
    rows = payload["episodes"]
    train_count, test_count = split["train_count"], split["test_count"]
    if (not isinstance(train_count, int) or not isinstance(test_count, int)
            or min(train_count, test_count) <= 0
            or train_count + test_count != len(rows)
            or len(rows) != config.data.selection_count):
        raise ValueError("split counts disagree")
    if (config.data.selection_index != 0
            or config.data.trajectory_count != train_count
            or config.data.expected_episode_uid != rows[0]["episode_uid"]):
        raise ValueError("training config must select exactly the train partition")
    source_by_uid = {r["episode_uid"]: r for r in source_rows}
    if (len(source_by_uid) != len(rows)
            or len({r["episode_uid"] for r in rows}) != len(rows)
            or any(source_by_uid.get(r["episode_uid"]) != r for r in rows)):
        raise ValueError("split must preserve every original episode exactly once")
    groups = split["group_identities"]
    train_ids, test_ids = [
        {identity for row in partition for identity in groups[row["episode_uid"]]}
        for partition in (rows[:train_count], rows[train_count:])
    ]
    if train_ids & test_ids:
        raise ValueError("train/test leakage: shared attempt, raw episode or object track")
    test = replace(config, data=replace(
        config.data, selection_index=train_count, trajectory_count=test_count,
        expected_episode_uid=rows[train_count]["episode_uid"],
    ))
    return test, {
        "protocol": VERSION, "seed": split["seed"],
        "selection": config.data.selection_relative,
        "selection_sha256": config.data.selection_sha256,
        "source_selection_sha256": split["source_selection_sha256"],
        "train_count": train_count, "test_count": test_count,
        "test_data": test.data.to_manifest(),
        "test_used_for_gradients_normalization_or_early_stop": False,
        "periodic_C_dataset": "train",
    }
