"""Runtime split labels reflect CLI selection without changing data or contracts."""
from dataclasses import replace
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest
from parallel_fixtures import recipe_config
from trajectory_tracking.cli import distributed_train as cli
from trajectory_tracking.data.heldout_split import VERSION, heldout_test_config


@pytest.mark.parametrize("dataset", ["train", "test"])
def test_admission_labels_actual_evaluation_without_changing_selection(monkeypatch, tmp_path, dataset):
    rows = [{"episode_uid": sha256(str(i).encode()).hexdigest()} for i in range(4)]
    original = tmp_path / "original.json"
    original.write_text(json.dumps({"episodes": rows}))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"episodes": rows, "split": {
        "protocol": VERSION, "seed": 42, "source_selection": original.name,
        "source_selection_sha256": sha256(original.read_bytes()).hexdigest(),
        "train_count": 3, "test_count": 1,
        "group_identities": {r["episode_uid"]: [r["episode_uid"]] for r in rows},
    }}))
    base = recipe_config()
    assert base.observation.feature_dim == 427
    assert base.direct_action.mode == "arm_target_increment"
    config = replace(base, data=replace(base.data, selection_relative=selection.name,
        selection_sha256=sha256(selection.read_bytes()).hexdigest(), selection_count=4,
        trajectory_count=3, expected_episode_uid=rows[0]["episode_uid"]))
    before = config.to_manifest()
    frozen = [p.read_bytes() for p in (original, selection)]
    expected_test, generic_split = heldout_test_config(config, tmp_path)
    seen = []
    def sources(c, **kwargs):
        seen.append(c)
        return (c.data.expected_episode_uid,)
    monkeypatch.setattr(cli, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(cli, "resolve_interprior_trajectory_sources", sources)
    monkeypatch.setattr(cli, "resolve_trajectory_scene_source", lambda **kwargs: "scene")
    monkeypatch.setattr(cli, "resolve_trajectory_scene_source_for_data", lambda c, source, **kwargs: source)
    admitted = cli.admit_data(SimpleNamespace(heldout_test_interval=1,
        evaluation_data_split=dataset, scene_asset_root="unused"), config)
    assert admitted[2] == expected_test
    assert seen == [config, expected_test]
    assert admitted[4] == dict(generic_split, periodic_C_dataset=dataset)
    assert generic_split["periodic_C_dataset"] == "train"
    assert heldout_test_config(config, tmp_path)[1] == generic_split
    assert config.to_manifest() == before
    assert [p.read_bytes() for p in (original, selection)] == frozen
