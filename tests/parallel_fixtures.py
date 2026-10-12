"""Synthetic parallel tests retain the recipe without historical run artifacts."""

from hashlib import sha256
import json
from pathlib import Path

import yaml

from trajectory_tracking.configuration import (
    load_trajectory_direct_config, resolve_trajectory_direct_config,
)
def recipe(path):
    return yaml.safe_load(path.read_text())

ROOT = Path(__file__).resolve().parents[1]
LEARNING = ROOT / "src/trajectory_tracking/resources"
CURRENT = "baseline.yaml"


def recipe_config(name=CURRENT):
    """Resolve algorithm parameters only; no data is consumed by these unit tests."""
    return resolve_trajectory_direct_config(recipe(LEARNING / name))


def synthetic_cli_config(directory):
    """Give the real CLI a checksummed test selection, preserving strict loading."""
    path = directory / "synthetic_parallel.yaml"
    if path.exists():
        return path
    raw = recipe(LEARNING / CURRENT)
    selection = directory / "synthetic_selection.json"
    uid = sha256(b"synthetic_parallel_fixture").hexdigest()
    selection.write_text(json.dumps(dict(
        schema_version=1, count=1, frame_start=0, frame_stop=raw["data"]["frame_stop"],
        episodes=[dict(episode_uid=uid)], synthetic_test_only=True,
    )))
    raw["run"]["environment_version"] = "synthetic_parallel_test_v1"
    raw["data"].update(
        selection=selection.relative_to(ROOT).as_posix(),
        selection_sha256=sha256(selection.read_bytes()).hexdigest(),
        selection_count=1, trajectory_count=1, selection_index=0,
        expected_episode_uid=uid,
    )
    path.write_text(yaml.safe_dump(raw))
    load_trajectory_direct_config(path, project_root=ROOT)
    return path
