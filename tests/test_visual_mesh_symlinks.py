"""Explicit shared visual files remain an exact-file browser allowlist."""
from pathlib import Path

import pytest

from trajectory_tracking.visualization.three_scene import BrowserUrdfActor


def make_urdf(root: Path, filename: str) -> Path:
    urdf = root / "object.urdf"
    urdf.write_text(
        '<robot name="object"><link name="object"><visual><geometry>'
        f'<mesh filename="{filename}"/>'
        '</geometry></visual></link></robot>'
    )
    return urdf


def test_shared_visual_symlink_requires_explicit_opt_in(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    shared_mesh = tmp_path / "shared.stl"
    shared_mesh.write_bytes(b"visual-mesh")
    (package / "visual.stl").symlink_to(shared_mesh)
    urdf = make_urdf(package, "visual.stl")
    with pytest.raises(ValueError, match="escapes its asset root"):
        BrowserUrdfActor.from_file("object", urdf, joint_names=())
    actor = BrowserUrdfActor.from_file(
        "object", urdf, joint_names=(), allow_external_mesh_symlinks=True,
    )
    assert len(actor.assets) == 1
    assert actor.assets[0].local_path == shared_mesh.resolve()
    assert actor.assets[0].url_path == "/visualization-assets/object/visual.stl"
    assert str(tmp_path) not in actor.urdf_xml


@pytest.mark.parametrize("path_kind", ["absolute", "parent"])
def test_opt_in_still_rejects_path_escape(tmp_path, path_kind):
    package = tmp_path / "package"
    package.mkdir()
    shared_mesh = tmp_path / "shared.stl"
    shared_mesh.write_bytes(b"visual-mesh")
    filename = str(shared_mesh) if path_kind == "absolute" else "../shared.stl"
    urdf = make_urdf(package, filename)
    with pytest.raises(ValueError, match="escapes its asset root"):
        BrowserUrdfActor.from_file(
            "object", urdf, joint_names=(), allow_external_mesh_symlinks=True,
        )
