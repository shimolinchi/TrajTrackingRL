"""Browser-only visual twin for trajectory-tracking training.

The viewer is an outer read-only projection.  It consumes one already-built
trajectory task environment and never advances the simulator, reference
cursor, policy, reward, or optimizer.
"""

from __future__ import annotations

from trajectory_tracking.visualization.theme import (REFERENCE_COLOR, FLOW_REFERENCE_COLOR, FLOW_ELAPSED_COLOR, FLOW_ACTUAL_COLOR)

from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

from trajectory_tracking.visualization import (
    BrowserActorState,
    BrowserOverlayVisibilityControl,
    BrowserPolylineOverlay,
    BrowserPrimitiveActor,
    BrowserThreeScene,
    BrowserUrdfActor,
    Visualization,
)
from trajectory_tracking.visualization.viewers.interprior_scene import (
    DEFAULT_INTERPRIOR_BROWSER_SCENE_STYLE,
    INTERPRIOR_CANONICAL_JOINT_NAMES,
    build_default_interprior_browser_scene,
)


_REFERENCE_FLOW_OVERLAY_ID = "reference_object_flow"
_REFERENCE_FLOW_ELAPSED_OVERLAY_ID = "reference_object_flow_elapsed"
_ACTUAL_FLOW_OVERLAY_ID = "actual_object_flow"
_PolylineBatch = tuple[tuple[tuple[float, float, float], ...], ...]

# Keep the entire tabletop/front rim and robot in frame, from a slightly lower
# right-side view. Aim near the table centre so sky does not displace the table,
# while retaining enough detail to inspect the hand/object interaction.
# These coordinates are browser-presentation only and remain in robot-base
# coordinates; they never affect an Isaac sensor or policy observation.
_TRAJECTORY_BROWSER_CAMERA_EYE = (1.35, -1.85, 0.90)
_TRAJECTORY_BROWSER_CAMERA_TARGET = (0.00, 0.02, 0.22)

# The historical 769-frame benchmark is still compact enough to show in full.
# Full-source lqr_test trajectories are 2,400 frames (2,280 after crop120), so
# presenting all four paths obscures the hand.  For those long trajectories,
# show recent context plus the same 120-step future horizon used by the object
# target observation instead.
_LONG_TRAJECTORY_MIN_STEPS = 1_200
_LONG_FLOW_PAST_STEPS = 30
_LONG_FLOW_FUTURE_STEPS = 120
_LONG_FLOW_WINDOW_STEPS = (
    _LONG_FLOW_PAST_STEPS + 1 + _LONG_FLOW_FUTURE_STEPS
)


def isolate_trajectory_tracking_render_environment(
    scene: Any,
    *,
    env_index: int,
) -> tuple[str, ...]:
    """Hide every non-selected clone from RTX without changing its physics.

    Parallel Isaac environments intentionally remain active for metric
    collection.  Only their inherited USD visibility is changed, so a camera
    attached to one rollout cannot confuse overlapping robot poses with a
    disconnected articulation.
    """

    num_envs = int(scene.num_envs)
    if not 0 <= env_index < num_envs:
        raise ValueError("env_index is outside the environment batch")
    env_prim_paths = tuple(str(path) for path in scene.env_prim_paths)
    if len(env_prim_paths) != num_envs:
        raise RuntimeError("scene environment prim paths disagree with num_envs")

    from pxr import UsdGeom

    hidden: list[str] = []
    for index, prim_path in enumerate(env_prim_paths):
        prim = scene.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"scene environment prim is missing: {prim_path}")
        imageable = UsdGeom.Imageable(prim)
        if not imageable.GetPrim().IsValid():
            raise RuntimeError(f"scene environment prim is not imageable: {prim_path}")
        imageable.GetVisibilityAttr().Set(
            UsdGeom.Tokens.inherited
            if index == env_index
            else UsdGeom.Tokens.invisible
        )
        if index != env_index:
            hidden.append(prim_path)
    return tuple(hidden)


@dataclass(frozen=True, slots=True)
class TrajectoryTrackingBrowserFrame:
    actor_states: tuple[BrowserActorState, ...]
    reference_object_flow: _PolylineBatch | None
    actual_object_surface_points: tuple[tuple[float, float, float], ...]
    reference_start_cursor: int
    reference_cursor: int
    episode_step: int


def _quat_normalize(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(
        value,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-8)


def _quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quaternion = _quat_normalize(quaternion)
    xyz = quaternion[..., 1:]
    twice_cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return (
        vector
        + quaternion[..., :1] * twice_cross
        + torch.cross(xyz, twice_cross, dim=-1)
    )


def _quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = _quat_normalize(left)
    right = _quat_normalize(right)
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return _quat_normalize(
        torch.stack(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ),
            dim=-1,
        )
    )


def _relative_pose_wxyz(
    parent_pose: torch.Tensor,
    child_pose: torch.Tensor,
) -> torch.Tensor:
    inverse = torch.cat((parent_pose[..., 3:4], -parent_pose[..., 4:]), dim=-1)
    return torch.cat(
        (
            _quat_apply(inverse, child_pose[..., :3] - parent_pose[..., :3]),
            _quat_multiply(inverse, child_pose[..., 3:]),
        ),
        dim=-1,
    )


def _xyzw(values_wxyz: list[float]) -> tuple[float, float, float, float]:
    return (
        float(values_wxyz[1]),
        float(values_wxyz[2]),
        float(values_wxyz[3]),
        float(values_wxyz[0]),
    )


def _browser_joint_indices(joint_names: tuple[str, ...]) -> tuple[int, ...]:
    if (
        len(joint_names) != len(INTERPRIOR_CANONICAL_JOINT_NAMES)
        or set(joint_names) != set(INTERPRIOR_CANONICAL_JOINT_NAMES)
    ):
        raise RuntimeError(
            "trajectory and Interprior browser expose different robot joints"
        )
    return tuple(joint_names.index(name) for name in INTERPRIOR_CANONICAL_JOINT_NAMES)


def _build_trajectory_tracking_browser_scene(
    robot_urdf: Any,
    *,
    flow_points: int,
    flow_horizon_steps: int,
    local_flow_window: bool = False,
    object_flow_enabled: bool = True,
    object_urdf: Any | None = None,
    object_size_m: tuple[float, float, float] = (0.06, 0.06, 0.06),
) -> BrowserThreeScene:
    if not isinstance(object_flow_enabled, bool):
        raise TypeError("object_flow_enabled must be a boolean")
    base = build_default_interprior_browser_scene(robot_urdf)
    flow_overlays = (
        (
            BrowserPolylineOverlay(
                _REFERENCE_FLOW_OVERLAY_ID,
                (
                    "局部参考 object flow（过去 30 / 未来 120 帧）"
                    if local_flow_window
                    else "完整参考 object flow"
                ),
                FLOW_REFERENCE_COLOR,
                max_polylines=flow_points,
                max_points_per_polyline=flow_horizon_steps,
                point_size_px=2.2 if local_flow_window else 3.0,
                opacity=0.55 if local_flow_window else 0.42,
            ),
            BrowserPolylineOverlay(
                _REFERENCE_FLOW_ELAPSED_OVERLAY_ID,
                "局部参考已走过区间" if local_flow_window else "参考已走过区间",
                FLOW_ELAPSED_COLOR,
                max_polylines=flow_points,
                max_points_per_polyline=flow_horizon_steps,
                point_size_px=2.8 if local_flow_window else 4.0,
                opacity=0.88 if local_flow_window else 0.98,
            ),
            BrowserPolylineOverlay(
                _ACTUAL_FLOW_OVERLAY_ID,
                "近期真实 object flow" if local_flow_window else "真实 object flow",
                FLOW_ACTUAL_COLOR,
                max_polylines=flow_points,
                max_points_per_polyline=flow_horizon_steps,
                point_size_px=3.0 if local_flow_window else 4.0,
                opacity=0.88 if local_flow_window else 0.95,
            ),
        )
        if object_flow_enabled
        else ()
    )
    flow_controls = (
        (
            BrowserOverlayVisibilityControl(
                "target_object_flow",
                "目标 Object Flow",
                (
                    _REFERENCE_FLOW_OVERLAY_ID,
                    _REFERENCE_FLOW_ELAPSED_OVERLAY_ID,
                ),
            ),
            BrowserOverlayVisibilityControl(
                "actual_object_flow",
                "真实 Object Flow",
                (_ACTUAL_FLOW_OVERLAY_ID,),
            ),
        )
        if object_flow_enabled
        else ()
    )
    object_actors = (() if object_urdf is None else (
        BrowserUrdfActor.from_file(
            "object", object_urdf, joint_names=(),
            allow_external_mesh_symlinks=True,
        ),
    ))
    reference_actors = ()
    if object_urdf is not None:
        from xml.etree import ElementTree
        links = ElementTree.parse(object_urdf).getroot().findall("link")
        reference_actors = (BrowserUrdfActor.from_file(
            "reference_object", object_urdf, joint_names=(),
            link_colors={link.attrib["name"]: "#3b82f6" for link in links},
            allow_external_mesh_symlinks=True,
        ),)
    return BrowserThreeScene(
        urdf_actors=(*base.urdf_actors, *object_actors, *reference_actors),
        primitive_actors=(
            *(actor for actor in base.primitive_actors if object_urdf is None or actor.actor_id != "object"),
            *((BrowserPrimitiveActor.box(
                "reference_object",
                object_size_m,
                color="#3b82f6",
                roughness=0.75,
                metalness=0.0,
                opacity=1.0,
                wireframe=False,
                label="当前参考物体",
            ),) if object_urdf is None else ()),
        ),
        polyline_overlays=flow_overlays,
        overlay_visibility_controls=flow_controls,
    )


def _elapsed_reference_flow(
    reference_flow: _PolylineBatch,
    elapsed_steps: int,
) -> _PolylineBatch:
    if elapsed_steps < 0:
        raise ValueError("elapsed_steps must be non-negative")
    return tuple(
        polyline[: min(elapsed_steps + 1, len(polyline))]
        for polyline in reference_flow
    )


def _local_reference_flow_window(
    reference_flow: _PolylineBatch,
    elapsed_steps: int,
) -> tuple[_PolylineBatch, _PolylineBatch]:
    """Return bounded reference context and its already-elapsed portion."""

    if elapsed_steps < 0:
        raise ValueError("elapsed_steps must be non-negative")
    if not reference_flow or any(not polyline for polyline in reference_flow):
        raise ValueError("reference flow must contain non-empty polylines")
    lengths = {len(polyline) for polyline in reference_flow}
    if len(lengths) != 1:
        raise ValueError("reference flow polylines must have equal lengths")
    flow_steps = lengths.pop()
    cursor = min(elapsed_steps, flow_steps - 1)
    start = max(0, cursor - _LONG_FLOW_PAST_STEPS)
    stop = min(flow_steps, cursor + _LONG_FLOW_FUTURE_STEPS + 1)
    return (
        tuple(polyline[start:stop] for polyline in reference_flow),
        tuple(polyline[start : cursor + 1] for polyline in reference_flow),
    )


@torch.no_grad()
def build_trajectory_tracking_browser_frame(
    env: Any,
    *,
    env_index: int,
    include_reference_flow: bool = True,
) -> TrajectoryTrackingBrowserFrame:
    """Snapshot one active trajectory environment in the robot-base frame."""

    if isinstance(env_index, bool) or not isinstance(env_index, int):
        raise TypeError("env_index must be an integer")
    if not 0 <= env_index < int(env.num_envs):
        raise ValueError("env_index is outside the environment batch")
    if not isinstance(include_reference_flow, bool):
        raise TypeError("include_reference_flow must be a boolean")
    episode = env.episode
    live_state = env.live_state
    if episode is None or live_state is None:
        raise RuntimeError("trajectory browser requires an active episode")

    trajectory = env.trajectory
    episode_step = int(env.episode_steps[env_index].item())
    episode_length = int(episode.episode_lengths[env_index].item())
    if not 0 <= episode_step <= episode_length:
        raise RuntimeError("trajectory browser episode step is out of range")
    start_cursor = int(episode.ppo_start_cursors[env_index].item())
    reference_cursor = int(env.reference_cursors[env_index].item())
    if not 0 <= reference_cursor < int(trajectory.frame_count):
        raise RuntimeError("trajectory browser reference cursor is out of range")

    browser_indices = torch.tensor(
        _browser_joint_indices(tuple(trajectory.joint_names)),
        dtype=torch.int64,
        device=live_state.joint_position.device,
    )
    joints = live_state.joint_position[env_index].index_select(0, browser_indices)
    if hasattr(episode, "trajectory_values"):
        robot_root_pose = episode.trajectory_values(
            "robot_root_pose_env_local"
        )[env_index]
        table_root_pose = episode.trajectory_values(
            "table_root_pose_env_local"
        )[env_index]
    else:
        # Small viewer fixtures and historical single-trajectory callers.
        robot_root_pose = trajectory.robot_root_pose_env_local
        table_root_pose = trajectory.table_root_pose_env_local
    table_pose = _relative_pose_wxyz(
        robot_root_pose,
        table_root_pose,
    )
    object_pose = live_state.object_pose_robot_base[env_index]
    reference_pose_sequence = (
        episode.frame_sequence("object_pose_robot_base", env_index)
        if hasattr(episode, "frame_sequence")
        else trajectory.object_pose_robot_base
    )
    reference_pose = reference_pose_sequence[reference_cursor]
    actual_points = live_state.object_surface_points_robot_base[env_index]
    if actual_points.shape != (trajectory.num_object_points, 3):
        raise RuntimeError("trajectory browser live object points have invalid shape")

    packed_terms = [
        joints,
        table_pose,
        object_pose,
        reference_pose,
        actual_points.reshape(-1),
    ]
    reference_plan = None
    if include_reference_flow:
        reference_sequence = (
            episode.frame_sequence("object_surface_points_robot_base", env_index)
            if hasattr(episode, "frame_sequence")
            else trajectory.object_surface_points_robot_base
        )
        reference_plan = reference_sequence[start_cursor:]
        packed_terms.append(reference_plan.reshape(-1))
    packed = torch.cat(packed_terms).detach().cpu()
    if not torch.isfinite(packed).all().item():
        raise RuntimeError("trajectory browser frame contains non-finite state")

    joint_count = len(INTERPRIOR_CANONICAL_JOINT_NAMES)
    point_count = int(trajectory.num_object_points)
    cursor = joint_count
    table_values = packed[cursor : cursor + 7].tolist()
    cursor += 7
    object_values = packed[cursor : cursor + 7].tolist()
    cursor += 7
    reference_values = packed[cursor : cursor + 7].tolist()
    cursor += 7
    actual_values = packed[cursor : cursor + point_count * 3].reshape(
        point_count,
        3,
    )
    cursor += point_count * 3
    plan_values = None
    if reference_plan is not None:
        plan_steps = int(reference_plan.shape[0])
        plan_values = packed[
            cursor : cursor + plan_steps * point_count * 3
        ].reshape(plan_steps, point_count, 3)

    return TrajectoryTrackingBrowserFrame(
        actor_states=(
            BrowserActorState(
                actor_id="robot",
                position_m=(0.0, 0.0, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                joint_positions=tuple(float(value) for value in packed[:joint_count]),
            ),
            BrowserActorState(
                actor_id="table",
                position_m=tuple(float(value) for value in table_values[:3]),
                quaternion_xyzw=_xyzw(table_values[3:]),
            ),
            BrowserActorState(
                actor_id="object",
                position_m=tuple(float(value) for value in object_values[:3]),
                quaternion_xyzw=_xyzw(object_values[3:]),
            ),
            BrowserActorState(
                actor_id="reference_object",
                position_m=tuple(float(value) for value in reference_values[:3]),
                quaternion_xyzw=_xyzw(reference_values[3:]),
            ),
        ),
        reference_object_flow=(
            None
            if plan_values is None
            else tuple(
                tuple(tuple(float(value) for value in point) for point in polyline)
                for polyline in plan_values.permute(1, 0, 2).tolist()
            )
        ),
        actual_object_surface_points=tuple(
            tuple(float(value) for value in point)
            for point in actual_values.tolist()
        ),
        reference_start_cursor=start_cursor,
        reference_cursor=reference_cursor,
        episode_step=episode_step,
    )


class TrajectoryTrackingBrowser:
    """Publish one decimated trajectory environment through the browser."""

    def __init__(
        self,
        *,
        scene: Any,
        source: Any,
        trajectory: Any,
        num_envs: int,
        host: str,
        port: int,
        env_index: int,
        publish_every_steps: int,
        object_flow_enabled: bool = True,
        title: str = "Trajectory Tracking PPO Training",
    ) -> None:
        if publish_every_steps <= 0:
            raise ValueError("publish_every_steps must be positive")
        if not 0 <= env_index < num_envs:
            raise ValueError("env_index is outside the environment batch")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("browser title must be a non-empty string")
        if not isinstance(object_flow_enabled, bool):
            raise TypeError("object_flow_enabled must be a boolean")
        _browser_joint_indices(tuple(trajectory.joint_names))
        self._env_index = env_index
        self._publish_every_steps = publish_every_steps
        self._steps_since_publish = 0
        self._local_flow_window = (
            int(trajectory.frame_count) >= _LONG_TRAJECTORY_MIN_STEPS
        )
        flow_horizon_steps = (
            _LONG_FLOW_WINDOW_STEPS
            if self._local_flow_window
            else int(trajectory.frame_count)
        )
        self._actual_flow_history: deque[
            tuple[tuple[float, float, float], ...]
        ] = deque(
            maxlen=(
                _LONG_FLOW_PAST_STEPS + 1
                if self._local_flow_window
                else flow_horizon_steps
            )
        )
        self._last_env: Any | None = None
        self._last_episode_id: int | None = None
        self._last_episode_step: int | None = None
        self._reference_object_flow: _PolylineBatch | None = None
        self._object_flow_enabled = object_flow_enabled
        object_source = source
        if getattr(source, "object_variants", ()):
            object_source = source.object_variants[source.object_layout[env_index % len(source.object_layout)]]
        object_visual = {}
        if getattr(object_source, "object_override", None) is not None:
            object_visual = {"object_urdf": object_source.object_urdf,
                             "object_size_m": object_source.object_override.object_size_m}
        self._visualization = Visualization(
            scene,
            num_envs=num_envs,
            env_index=env_index,
            host=host,
            port=port,
            title=title.strip(),
            scene_style=DEFAULT_INTERPRIOR_BROWSER_SCENE_STYLE,
            camera_eye=_TRAJECTORY_BROWSER_CAMERA_EYE,
            camera_target=_TRAJECTORY_BROWSER_CAMERA_TARGET,
            three_scene=_build_trajectory_tracking_browser_scene(
                source.robot_urdf,
                flow_points=int(trajectory.num_object_points),
                flow_horizon_steps=flow_horizon_steps,
                local_flow_window=self._local_flow_window,
                object_flow_enabled=object_flow_enabled,
                **object_visual,
            ),
            camera_enabled=False,
        )
        self._visualization.initialize()

    @property
    def url(self) -> str:
        return self._visualization.url

    def set_recording_paused(self, paused: bool) -> int:
        """Gate browser recording while continuing live evaluation rendering."""

        return self._visualization.set_recording_paused(paused)

    def observe(self, env: Any) -> None:
        """Publish an active post-reset/post-step state without advancing it."""

        if env.episode is None or env.live_state is None:
            raise RuntimeError("trajectory browser observed an inactive environment")
        episode_id = int(env.episode_ids[self._env_index].item())
        episode_step = int(env.episode_steps[self._env_index].item())
        unchanged_state = (
            env is self._last_env
            and episode_id == self._last_episode_id
            and episode_step == self._last_episode_step
        )
        if unchanged_state:
            return
        reset_history = (
            env is not self._last_env
            or episode_id != self._last_episode_id
            or (
                self._last_episode_step is not None
                and episode_step < self._last_episode_step
            )
        )
        self._last_env = env
        self._last_episode_id = episode_id
        self._last_episode_step = episode_step
        if reset_history:
            self._steps_since_publish = 0
        else:
            self._steps_since_publish += 1
            if self._steps_since_publish < self._publish_every_steps:
                return
            self._steps_since_publish = 0

        self._visualization.apply_camera_controls()
        frame = build_trajectory_tracking_browser_frame(
            env,
            env_index=self._env_index,
            include_reference_flow=(
                self._object_flow_enabled
                and (reset_history or self._reference_object_flow is None)
            ),
        )
        self._publish_frame(frame, reset_history=reset_history)

    def _publish_frame(
        self,
        frame: TrajectoryTrackingBrowserFrame,
        *,
        reset_history: bool,
    ) -> None:
        if not self._object_flow_enabled:
            self._visualization.publish_presentation_frame(
                actor_states=frame.actor_states,
                polyline_overlays={},
            )
            return
        if reset_history:
            self._actual_flow_history.clear()
            self._reference_object_flow = None
        if frame.reference_object_flow is not None:
            self._reference_object_flow = frame.reference_object_flow
        if self._reference_object_flow is None:
            raise RuntimeError("trajectory browser has no cached reference flow")
        if len(self._reference_object_flow) != len(
            frame.actual_object_surface_points
        ):
            raise RuntimeError(
                "trajectory browser reference/live point counts disagree"
            )

        self._actual_flow_history.append(frame.actual_object_surface_points)
        actual_flow = tuple(
            tuple(sample[index] for sample in self._actual_flow_history)
            for index in range(len(frame.actual_object_surface_points))
        )
        elapsed_steps = max(
            0,
            frame.reference_cursor - frame.reference_start_cursor,
        )
        if self._local_flow_window:
            displayed_reference_flow, elapsed_reference_flow = (
                _local_reference_flow_window(
                    self._reference_object_flow,
                    elapsed_steps,
                )
            )
        else:
            displayed_reference_flow = self._reference_object_flow
            elapsed_reference_flow = _elapsed_reference_flow(
                self._reference_object_flow,
                elapsed_steps,
            )
        overlays: dict[str, _PolylineBatch] = {
            _REFERENCE_FLOW_ELAPSED_OVERLAY_ID: elapsed_reference_flow,
            _ACTUAL_FLOW_OVERLAY_ID: actual_flow,
        }
        if self._local_flow_window or frame.reference_object_flow is not None:
            overlays[_REFERENCE_FLOW_OVERLAY_ID] = displayed_reference_flow
        self._visualization.publish_presentation_frame(
            actor_states=frame.actor_states,
            polyline_overlays=overlays,
        )

    def close(self) -> None:
        self._visualization.close()


__all__ = [
    "TrajectoryTrackingBrowser",
    "TrajectoryTrackingBrowserFrame",
    "build_trajectory_tracking_browser_frame",
    "isolate_trajectory_tracking_render_environment",
]
