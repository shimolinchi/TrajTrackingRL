from pathlib import Path
import pytest
from trajectory_tracking.visualization.playback import advance_clip
from trajectory_tracking.visualization.three_scene import BrowserUrdfActor, BrowserThreeScene
from trajectory_tracking.visualization.theme import apply_scene_theme, TABLE_MATERIAL, CAMERA_DRAG_MODES


def test_playlist_keeps_real_speed_without_a_sixty_second_freeze():
    assert advance_clip(0, 1.0, [1.65, 4.0, 2.83]) == (0, 1.0)
    index, time = advance_clip(0, 2.15, [1.65, 4.0, 2.83])
    assert index == 1 and time == pytest.approx(0)
    index, time = advance_clip(0, 3.15, [1.65, 4.0, 2.83])
    assert index == 1 and time == pytest.approx(1)
    assert advance_clip(0, 60, [1, 2]) == (0, 0)
    assert CAMERA_DRAG_MODES["0"] == "orbit"
    assert CAMERA_DRAG_MODES["2"] == "pan"


def test_common_theme_overrides_authored_table_color_without_touching_asset(tmp_path):
    p = tmp_path / "table.urdf"
    xml = '<robot name="table"><link name="top"><visual><geometry><box size="1 1 .3"/></geometry><material name="wood"><color rgba="1 .9 .5 1"/></material></visual></link></robot>'
    p.write_text(xml)
    original = BrowserUrdfActor.from_file("table", p, joint_names=())
    scene = apply_scene_theme(BrowserThreeScene(urdf_actors=(original,)))
    themed = scene.urdf_actors[0]
    assert themed.link_colors == (("top", "#2E3845"),)
    assert themed.roughness == TABLE_MATERIAL["roughness"]
    assert themed.urdf_xml == original.urdf_xml
    assert p.read_text() == xml
