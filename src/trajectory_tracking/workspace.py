"""Workspace and immutable package-resource locations."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("TRAJTRACKING_WORKSPACE", Path.cwd())).resolve()
PACKAGE = Path(__file__).resolve().parent
CONFIG = PACKAGE / "resources/baseline.yaml"
CONTRACT = PACKAGE / "resources/physics_contract_template.json"

def alignment_source_path(root, relative):
    relative = relative.replace("src/trajectory_residual/", "src/trajectory_tracking/")
    return PACKAGE / relative.removeprefix("src/trajectory_tracking/")
