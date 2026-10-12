"""Static and filesystem contracts for the independent task-local scene."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest import mock

from trajectory_tracking.environment.scene_config import (
    ScenePhysicsSpec,
    TRAJECTORY_ISAAC_SCENE,
    resolve_trajectory_scene_source,
)
from trajectory_tracking.environment.scene import DEFAULT_ENV_SPACING_M, build_simulation_cfg


ROOT = Path(__file__).resolve().parents[1]


class TrajectoryTrackingSceneTest(unittest.TestCase):
    def test_default_parallel_environment_spacing_is_one_point_five_metres(self) -> None:
        self.assertEqual(DEFAULT_ENV_SPACING_M, 1.5)

    def test_bundled_scene_assets_resolve(self) -> None:
        if not os.environ.get("TRAJTRACKING_SCENE_ASSET_ROOT"):
            self.skipTest("real scene assets are separately configured; native acceptance validates them")
        source = resolve_trajectory_scene_source(
            project_root=ROOT,
            scene_asset_root=Path(os.environ["TRAJTRACKING_SCENE_ASSET_ROOT"]),
        )
        self.assertEqual(source.spec, TRAJECTORY_ISAAC_SCENE)
        self.assertEqual(source.robot_urdf.name, "xarm7_wuji_right.urdf")
        self.assertEqual(source.table_urdf.name, "table_round_large.urdf")
        self.assertEqual(source.object_urdf.name, "cube_0p06m.urdf")
        self.assertEqual(source.spec.physics.dt, 1.0 / 120.0)
        self.assertEqual(source.spec.version, "trajectory_isaac_scene_v2")
        self.assertTrue(
            source.spec.physics.enable_external_forces_every_iteration
        )

    def test_simulation_cfg_enables_external_forces_on_every_tgs_iteration(self) -> None:
        class CaptureConfig:
            def __init__(self, **kwargs: object) -> None:
                for name, value in kwargs.items():
                    setattr(self, name, value)

        fake_isaaclab = ModuleType("isaaclab")
        fake_isaaclab.__path__ = []  # type: ignore[attr-defined]
        fake_sim = ModuleType("isaaclab.sim")
        fake_sim.SimulationCfg = CaptureConfig  # type: ignore[attr-defined]
        fake_sim.PhysxCfg = CaptureConfig  # type: ignore[attr-defined]
        fake_isaaclab.sim = fake_sim  # type: ignore[attr-defined]

        with mock.patch.dict(
            sys.modules,
            {"isaaclab": fake_isaaclab, "isaaclab.sim": fake_sim},
        ):
            simulation = build_simulation_cfg(
                device="cuda:0", physics=ScenePhysicsSpec()
            )

        self.assertTrue(simulation.physx.enable_external_forces_every_iteration)
        with mock.patch.dict(sys.modules, {"isaaclab": fake_isaaclab, "isaaclab.sim": fake_sim}):
            expanded = build_simulation_cfg(device="cuda:0", physics=ScenePhysicsSpec(),
                                             gpu_collision_stack_size_bytes=256 * 1024**2)
        self.assertEqual(expanded.physx.gpu_collision_stack_size, 256 * 1024**2)
        self.assertEqual(expanded.dt, simulation.dt)
        self.assertEqual(expanded.physx.solver_type, simulation.physx.solver_type)


    def test_isaac_imports_stay_in_scene_or_env_modules(self) -> None:
        package = ROOT / "src/trajectory_tracking"
        violations: list[str] = []
        for path in (p for folder in ("data", "environment", "training") for p in (package / folder).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                if any(
                    name == "isaaclab" or name.startswith("isaaclab.") for name in names
                ):
                    if path.name not in {"scene.py", "core.py"}:
                        violations.append(str(path.relative_to(ROOT)))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
