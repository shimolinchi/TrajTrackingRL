"""CPU-only contracts for the trajectory-residual browser projection."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch

from trajectory_tracking.visualization.viewers.interprior_scene import INTERPRIOR_CANONICAL_JOINT_NAMES
from trajectory_tracking.visualization.viewers.trajectory_tracking import (
    TrajectoryTrackingBrowser,
    TrajectoryTrackingBrowserFrame,
    _build_trajectory_tracking_browser_scene,
    build_trajectory_tracking_browser_frame,
    isolate_trajectory_tracking_render_environment,
)


ROOT = Path(__file__).resolve().parents[1]


def _pose(x: float, y: float, z: float) -> torch.Tensor:
    return torch.tensor((x, y, z, 1.0, 0.0, 0.0, 0.0), dtype=torch.float32)


def _fake_environment(*, episode_step: int = 2) -> SimpleNamespace:
    joint_names = tuple(reversed(INTERPRIOR_CANONICAL_JOINT_NAMES))
    joint_values = torch.tensor(
        [INTERPRIOR_CANONICAL_JOINT_NAMES.index(name) for name in joint_names],
        dtype=torch.float32,
    )
    tracks = torch.arange(6 * 4 * 3, dtype=torch.float32).reshape(6, 4, 3)
    object_poses = torch.stack([_pose(float(index), 0.0, 0.0) for index in range(6)])
    trajectory = SimpleNamespace(
        joint_names=joint_names,
        frame_count=6,
        num_object_points=4,
        robot_root_pose_env_local=_pose(1.0, 2.0, 3.0),
        table_root_pose_env_local=_pose(1.5, 2.25, 3.75),
        object_pose_robot_base=object_poses,
        object_surface_points_robot_base=tracks,
    )
    live_state = SimpleNamespace(
        joint_position=torch.stack((joint_values, joint_values + 100.0)),
        object_pose_robot_base=torch.stack(
            (_pose(0.1, 0.2, 0.3), _pose(1.0, 1.0, 1.0))
        ),
        object_surface_points_robot_base=torch.stack((tracks[episode_step], tracks[0])),
    )
    episode = SimpleNamespace(
        ppo_start_cursors=torch.tensor((1, 1), dtype=torch.int64),
        episode_lengths=torch.tensor((4, 4), dtype=torch.int64),
    )
    return SimpleNamespace(
        num_envs=2,
        trajectory=trajectory,
        live_state=live_state,
        episode=episode,
        episode_steps=torch.tensor((episode_step, 0), dtype=torch.int64),
        reference_cursors=torch.tensor((1 + episode_step, 1), dtype=torch.int64),
        episode_ids=torch.tensor((0, 0), dtype=torch.int64),
    )


def _frame(
    *,
    episode_step: int,
    reference_flow: tuple[tuple[tuple[float, float, float], ...], ...] | None,
) -> TrajectoryTrackingBrowserFrame:
    points = tuple((float(index), float(episode_step), 0.0) for index in range(4))
    return TrajectoryTrackingBrowserFrame(
        actor_states=(),
        reference_object_flow=reference_flow,
        actual_object_surface_points=points,
        reference_start_cursor=1,
        reference_cursor=1 + episode_step,
        episode_step=episode_step,
    )


class _FakeVisualization:
    instances: list[_FakeVisualization] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self.frames: list[dict[str, object]] = []
        self.camera_control_calls = 0
        self.initialized = False
        self.closed = False
        self.url = "http://127.0.0.1:8002/"
        self.instances.append(self)

    def initialize(self) -> None:
        self.initialized = True

    def apply_camera_controls(self) -> bool:
        self.camera_control_calls += 1
        return False

    def publish_presentation_frame(self, **kwargs: object) -> int:
        self.frames.append(kwargs)
        return len(self.frames)

    def close(self) -> None:
        self.closed = True


class _FakePrim:
    def __init__(self) -> None:
        self.visibility = "unset"

    def IsValid(self) -> bool:
        return True


class _FakeVisibilityAttribute:
    def __init__(self, prim: _FakePrim) -> None:
        self._prim = prim

    def Set(self, value: str) -> None:
        self._prim.visibility = value


class _FakeImageable:
    def __init__(self, prim: _FakePrim) -> None:
        self._prim = prim

    def GetPrim(self) -> _FakePrim:
        return self._prim

    def GetVisibilityAttr(self) -> _FakeVisibilityAttribute:
        return _FakeVisibilityAttribute(self._prim)


class TrajectoryTrackingBrowserFrameTest(unittest.TestCase):
    def test_scene_uses_reference_object_and_three_bounded_p4_overlays(self) -> None:
        robot_urdf = (
            ROOT
            / "src/trajectory_tracking/resources/scene/assets/urdf/"
            "xarm7_wuji_right_description/xarm7_wuji_right.urdf"
        )

        scene = _build_trajectory_tracking_browser_scene(
            robot_urdf,
            flow_points=4,
            flow_horizon_steps=768,
        )

        primitives = {actor.actor_id: actor for actor in scene.primitive_actors}
        overlays = {overlay.overlay_id: overlay for overlay in scene.polyline_overlays}
        self.assertFalse(primitives["reference_object"].wireframe)
        self.assertEqual(primitives["reference_object"].label, "当前参考物体")
        self.assertEqual(overlays["reference_object_flow"].color, "#1FD1FF")
        self.assertEqual(overlays["reference_object_flow_elapsed"].color, "#0B78B8")
        self.assertEqual(overlays["actual_object_flow"].color, "#FFBE5C")
        self.assertEqual(
            scene.polyline_limits,
            {
                "reference_object_flow": (4, 768),
                "reference_object_flow_elapsed": (4, 768),
                "actual_object_flow": (4, 768),
            },
        )

    def test_frame_uses_explicit_joint_mapping_and_full_p4_episode_flow(self) -> None:
        env = _fake_environment()

        frame = build_trajectory_tracking_browser_frame(env, env_index=0)

        actors = {actor.actor_id: actor for actor in frame.actor_states}
        self.assertEqual(actors["robot"].joint_positions, tuple(map(float, range(27))))
        self.assertEqual(actors["table"].position_m, (0.5, 0.25, 0.75))
        for actual, expected in zip(
            actors["object"].position_m,
            (0.1, 0.2, 0.3),
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(actors["reference_object"].position_m, (3.0, 0.0, 0.0))
        self.assertEqual(frame.reference_start_cursor, 1)
        self.assertEqual(frame.reference_cursor, 3)
        self.assertEqual(frame.episode_step, 2)
        self.assertEqual(len(frame.reference_object_flow or ()), 4)
        self.assertTrue(
            torch.equal(
                torch.tensor(frame.reference_object_flow),
                env.trajectory.object_surface_points_robot_base[1:].permute(1, 0, 2),
            )
        )
        self.assertTrue(
            torch.equal(
                torch.tensor(frame.actual_object_surface_points),
                env.live_state.object_surface_points_robot_base[0],
            )
        )

    def test_frame_can_skip_the_static_reference_flow_after_first_publish(self) -> None:
        frame = build_trajectory_tracking_browser_frame(
            _fake_environment(),
            env_index=0,
            include_reference_flow=False,
        )

        self.assertIsNone(frame.reference_object_flow)

    def test_camera_visibility_keeps_only_the_selected_environment(self) -> None:
        prims = {
            f"/World/envs/env_{index}": _FakePrim() for index in range(5)
        }
        scene = SimpleNamespace(
            num_envs=5,
            env_prim_paths=tuple(prims),
            stage=SimpleNamespace(GetPrimAtPath=prims.__getitem__),
        )
        fake_usd_geom = SimpleNamespace(
            Imageable=_FakeImageable,
            Tokens=SimpleNamespace(inherited="inherited", invisible="invisible"),
        )

        with patch.dict(sys.modules, {"pxr": SimpleNamespace(UsdGeom=fake_usd_geom)}):
            hidden = isolate_trajectory_tracking_render_environment(
                scene,
                env_index=2,
            )

        self.assertEqual(len(hidden), 4)
        self.assertEqual(prims["/World/envs/env_2"].visibility, "inherited")
        self.assertEqual(
            {
                prim.visibility
                for path, prim in prims.items()
                if path != "/World/envs/env_2"
            },
            {"invisible"},
        )


class TrajectoryTrackingBrowserPublisherTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeVisualization.instances.clear()

    def test_publisher_decimates_and_resets_history_at_environment_boundary(
        self,
    ) -> None:
        reference_flow = tuple(
            tuple((float(point), float(step), 0.0) for step in range(5))
            for point in range(4)
        )
        frames = (
            _frame(episode_step=0, reference_flow=reference_flow),
            _frame(episode_step=2, reference_flow=None),
            _frame(episode_step=0, reference_flow=reference_flow),
        )
        env = _fake_environment(episode_step=0)
        scene = SimpleNamespace(num_envs=2)
        source = SimpleNamespace(robot_urdf="unused-by-this-test")

        with (
            patch(
                "trajectory_tracking.visualization.viewers.trajectory_tracking.Visualization",
                _FakeVisualization,
            ),
            patch(
                "trajectory_tracking.visualization.viewers.trajectory_tracking."
                "_build_trajectory_tracking_browser_scene",
                return_value=object(),
            ),
            patch(
                "trajectory_tracking.visualization.viewers.trajectory_tracking."
                "build_trajectory_tracking_browser_frame",
                side_effect=frames,
            ) as build_frame,
        ):
            browser = TrajectoryTrackingBrowser(
                scene=scene,
                source=source,
                trajectory=env.trajectory,
                num_envs=2,
                host="127.0.0.1",
                port=8002,
                env_index=0,
                publish_every_steps=2,
                title="Trajectory PPO Evaluation | update 100",
            )
            browser.observe(env)
            env.episode_steps[0] = 1
            env.reference_cursors[0] = 2
            browser.observe(env)
            env.episode_steps[0] = 2
            env.reference_cursors[0] = 3
            browser.observe(env)
            # Evaluation keeps already-completed environments static while the
            # rest of the batch finishes; duplicate callbacks must not erase
            # the real trajectory with repeated terminal points.
            browser.observe(env)
            replacement = _fake_environment(episode_step=0)
            browser.observe(replacement)
            browser.close()

        visualization = _FakeVisualization.instances[0]
        self.assertTrue(visualization.initialized)
        self.assertTrue(visualization.closed)
        self.assertFalse(visualization.kwargs["camera_enabled"])
        self.assertEqual(
            visualization.kwargs["title"],
            "Trajectory PPO Evaluation | update 100",
        )
        self.assertEqual(visualization.camera_control_calls, 3)
        self.assertEqual(build_frame.call_count, 3)
        self.assertTrue(build_frame.call_args_list[0].kwargs["include_reference_flow"])
        self.assertFalse(build_frame.call_args_list[1].kwargs["include_reference_flow"])
        self.assertTrue(build_frame.call_args_list[2].kwargs["include_reference_flow"])
        self.assertEqual(len(visualization.frames), 3)

        second_overlays = visualization.frames[1]["polyline_overlays"]
        self.assertNotIn("reference_object_flow", second_overlays)
        self.assertEqual(
            tuple(
                len(line)
                for line in second_overlays["reference_object_flow_elapsed"]
            ),
            (3, 3, 3, 3),
        )
        self.assertEqual(
            tuple(len(line) for line in second_overlays["actual_object_flow"]),
            (2, 2, 2, 2),
        )
        reset_overlays = visualization.frames[2]["polyline_overlays"]
        self.assertIn("reference_object_flow", reset_overlays)
        self.assertEqual(
            tuple(len(line) for line in reset_overlays["actual_object_flow"]),
            (1, 1, 1, 1),
        )

    def test_trajectory_training_core_does_not_import_outer_tools(self) -> None:
        source = (ROOT / "src/trajectory_tracking/training/ppo.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("tools.", source)


if __name__ == "__main__":
    unittest.main()
