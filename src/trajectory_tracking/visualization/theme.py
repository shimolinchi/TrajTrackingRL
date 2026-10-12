"""Shared browser appearance, controls and playback defaults for every task.

Only presentation is changed. Asset files, geometry, physical parameters,
policy inputs and recorded states remain owned by their producers.
"""
from dataclasses import replace
from types import MappingProxyType
import xml.etree.ElementTree as ET
from .presentation import BrowserSceneStyle, VisualizationTrail

CAMERA_DRAG_MODES = MappingProxyType({"0": "orbit", "1": "orbit", "2": "pan"})
CAMERA_HELP = "左键拖动：旋转"
SHOW_DASHBOARD = False
REPLAY_TERMINAL_HOLD_SECONDS = 0.5

ROBOT_MATERIAL = dict(fallback_color="#B8B8B8", roughness=0.80, metalness=0.20)
TABLE_MATERIAL = dict(color=(0.18, 0.22, 0.27), roughness=0.88, metalness=0.04)
OBJECT_MATERIAL = dict(color=(0.10, 0.65, 0.30), roughness=0.80, metalness=0.0, opacity=0.72)
REFERENCE_COLOR = "#31D7C5"
REFERENCE_MATERIAL = dict(color=(0.19, 0.84, 0.77), roughness=0.8, metalness=0.0, opacity=0.30)
FLOW_REFERENCE_COLOR = "#1FD1FF"
FLOW_ELAPSED_COLOR = "#0B78B8"
FLOW_ACTUAL_COLOR = "#FFBE5C"

# Exact presentation values from the pinned Interprior_train Three.js viewer.
# These values are browser-only and must not be applied to the Isaac scene.
DEFAULT_SCENE_STYLE = BrowserSceneStyle(
    background_color="#0B0F15",
    floor_tint="#2D3B4E",
    grid_major_color="#35557A",
    grid_minor_color="#203447",
    x_axis_color="#FFBE5C",
    y_axis_color="#31D7C5",
    key_light_color="#FFFFFF",
    fill_light_color="#F6FAFF",
    rim_light_color="#E8F3FF",
    grid_plane_z_m=-0.1,
    grid_size_m=6.0,
    grid_divisions=24,
    color_grade_strength=0.30,
    light_strength=0.22,
)

DEFAULT_TRAILS = (
    VisualizationTrail(
        "link7",
        "Link7",
        color="#FF6B14",
        max_points=360,
        line_width_px=3.2,
        marker_radius_px=5.0,
        tube_radius_m=0.0042,
        marker_radius_m=0.010,
    ),
    VisualizationTrail(
        "object",
        "Object",
        color="#1FD1FF",
        max_points=360,
        line_width_px=2.6,
        marker_radius_px=4.5,
        tube_radius_m=0.0034,
        marker_radius_m=0.0085,
    ),
)


def apply_scene_theme(scene):
    """Apply common material roles to a visual scene, preserving its geometry."""
    def urdf(actor):
        if actor.actor_id == "robot":
            return replace(actor, **ROBOT_MATERIAL)
        material = (TABLE_MATERIAL if actor.actor_id == "table" else
                    OBJECT_MATERIAL if actor.actor_id.startswith("object") else
                    REFERENCE_MATERIAL if actor.actor_id.startswith("target") else None)
        if material is None:
            return actor
        color = "#" + "".join(f"{round(c * 255):02X}" for c in material["color"])
        links = tuple((link.attrib["name"], color)
                      for link in ET.fromstring(actor.urdf_xml).findall("link"))
        return replace(actor, link_colors=links, fallback_color=color,
                       roughness=material["roughness"], metalness=material["metalness"],
                       opacity=material.get("opacity", 1.0))
    def primitive(actor):
        material = (TABLE_MATERIAL if actor.actor_id == "table" else
                    OBJECT_MATERIAL if actor.actor_id == "object" else None)
        return replace(actor, **material) if material else actor
    return replace(scene, urdf_actors=tuple(urdf(a) for a in scene.urdf_actors),
                   primitive_actors=tuple(primitive(a) for a in scene.primitive_actors))
