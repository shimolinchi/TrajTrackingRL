"""Release geometry must work without private checkouts or producer machine paths."""
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest
from trajectory_tracking.workspace import SCENE_ASSETS, OBJECT_ASSETS
from trajectory_tracking.environment.scene_config import resolve_trajectory_scene_source, TrajectorySceneConfigError
from trajectory_tracking.data.preparation.publication import catalog_asset_path


def test_every_bundled_file_matches_its_manifest_and_urdf_meshes_exist():
    for root in (SCENE_ASSETS, OBJECT_ASSETS):
        manifest = json.loads((root / "manifest.json").read_text())
        for relative, expected in manifest["files"].items():
            path = root / relative
            assert sha256(path.read_bytes()).hexdigest() == expected, relative
            if path.suffix == ".urdf":
                for mesh in ET.parse(path).getroot().findall(".//mesh"):
                    target = (path.parent / mesh.attrib["filename"]).resolve()
                    assert target.is_relative_to(root.resolve()) and target.is_file()
    assert json.loads((OBJECT_ASSETS / "manifest.json").read_text())["object_count"] == 113


def test_scene_resolves_without_git_or_an_upstream_checkout(tmp_path):
    source = resolve_trajectory_scene_source(project_root=tmp_path, scene_asset_root=SCENE_ASSETS)
    assert source.robot_urdf.is_relative_to(SCENE_ASSETS)
    assert source.ground_usd.is_relative_to(SCENE_ASSETS)


def test_corrupt_geometry_fails_closed(tmp_path):
    from dataclasses import replace
    from trajectory_tracking.environment.scene_config import SceneAssetSpec, TRAJECTORY_ISAAC_SCENE
    spec = replace(TRAJECTORY_ISAAC_SCENE, robot=SceneAssetSpec(TRAJECTORY_ISAAC_SCENE.robot.relative_path, "0" * 64))
    with pytest.raises(TrajectorySceneConfigError, match="hash mismatch"):
        resolve_trajectory_scene_source(project_root=tmp_path, scene_asset_root=SCENE_ASSETS, spec=spec)


def test_catalog_relocates_old_absolute_uris_and_rejects_changed_identity():
    catalog = json.loads((OBJECT_ASSETS / "catalog.json").read_text())
    for uid, item in catalog.items():
        spec = dict(item, urdf_uri="/nonexistent/producer/object.urdf", mesh_uri="/nonexistent/producer/object.stl")
        for kind in ("urdf", "mesh"):
            assert catalog_asset_path(OBJECT_ASSETS, uid, spec, kind).is_file()
    with pytest.raises(ValueError, match="catalog differs"):
        catalog_asset_path(OBJECT_ASSETS, uid, dict(spec, mesh_sha256="0" * 64), "urdf")


def test_changed_object_mesh_is_rejected_before_simulation(tmp_path):
    from trajectory_tracking.data.tro_object_replay import object_scene
    (tmp_path / "shape.urdf").write_text('<robot name="shape"><link name="object"><visual><geometry><mesh filename="shape.stl"/></geometry></visual></link></robot>')
    (tmp_path / "shape.stl").write_bytes(b"changed geometry")
    spec = dict(urdf_relative="shape.urdf", mesh_sha256=sha256(b"original geometry").hexdigest())
    with pytest.raises(ValueError, match="mesh checksum changed"):
        object_scene(None, spec, tmp_path)
