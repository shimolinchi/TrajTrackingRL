"""Workspace and immutable package-resource locations."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("TRAJTRACKING_WORKSPACE", Path.cwd())).resolve()
PACKAGE = Path(__file__).resolve().parent
CONFIG = PACKAGE / "resources/baseline.yaml"
CONTRACT = PACKAGE / "resources/physics_contract_template.json"
SCENE_ASSETS = PACKAGE / "resources/scene"
OBJECT_ASSETS = PACKAGE / "resources/objects"

# Bundled assets are the default in source checkouts and installed wheels.
os.environ.setdefault("TRAJTRACKING_SCENE_ASSET_ROOT", str(SCENE_ASSETS))
os.environ.setdefault("TRAJTRACKING_OBJECT_ASSET_ROOT", str(OBJECT_ASSETS))

def alignment_source_path(root, relative):
    relative = relative.replace("src/trajectory_residual/", "src/trajectory_tracking/")
    return PACKAGE / relative.removeprefix("src/trajectory_tracking/")
