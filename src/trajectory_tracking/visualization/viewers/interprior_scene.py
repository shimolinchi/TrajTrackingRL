"""Default browser-only presentation for the pinned Interprior scene.

This module is the single source of truth for the default appearance of the
Interprior scene in project browser viewers.  It composes the generic
``trajectory_tracking.visualization`` contracts into an independent Three.js visual twin;
it never edits USD/URDF assets, simulator materials, physics settings, or
camera tensors used by a policy.
"""

from __future__ import annotations

from pathlib import Path

from trajectory_tracking.visualization import (
    BrowserPrimitiveActor,
    BrowserSceneStyle,
    BrowserThreeScene,
    BrowserUrdfActor,
    VisualizationTrail,
)


INTERPRIOR_CANONICAL_JOINT_NAMES = tuple(
    [f"joint{index}" for index in range(1, 8)]
    + [
        f"right_finger{finger}_joint{depth}"
        for finger in range(1, 6)
        for depth in range(1, 5)
    ]
)

# Backward-compatible imports; appearance is owned by the shared tool theme.
from trajectory_tracking.visualization.theme import (
    DEFAULT_SCENE_STYLE as DEFAULT_INTERPRIOR_BROWSER_SCENE_STYLE,
    DEFAULT_TRAILS as DEFAULT_INTERPRIOR_BROWSER_TRAILS,
    ROBOT_MATERIAL, TABLE_MATERIAL, OBJECT_MATERIAL,
)

def build_default_interprior_browser_scene(
    robot_urdf: str | Path,
) -> BrowserThreeScene:
    """Build the default robot/table/object visual twin for browser viewers."""

    robot = BrowserUrdfActor.from_file(
        "robot",
        Path(robot_urdf),
        joint_names=INTERPRIOR_CANONICAL_JOINT_NAMES,
        # GLB materials remain untouched.  This is only the upstream fallback
        # for STL visuals without an authored material (not a link override).
        **ROBOT_MATERIAL,
    )
    return BrowserThreeScene(
        urdf_actors=(robot,),
        primitive_actors=(
            BrowserPrimitiveActor.cylinder(
                "table",
                radius_m=0.7,
                height_m=0.3,
                **TABLE_MATERIAL,
            ),
            BrowserPrimitiveActor.box(
                "object",
                (0.06, 0.06, 0.06),
                **OBJECT_MATERIAL,
            ),
        ),
    )


__all__ = [
    "DEFAULT_INTERPRIOR_BROWSER_SCENE_STYLE",
    "DEFAULT_INTERPRIOR_BROWSER_TRAILS",
    "INTERPRIOR_CANONICAL_JOINT_NAMES",
    "build_default_interprior_browser_scene",
]
