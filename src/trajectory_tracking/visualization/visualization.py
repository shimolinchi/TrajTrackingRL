"""Public Isaac Camera to browser visualization facade."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from io import BytesIO
from math import atan, tan
from numbers import Real
from threading import Condition, Thread, get_ident
from time import monotonic
from typing import Any, Callable

from .buttons import ButtonEventBuffer, VisualizationButton, normalize_buttons
from .camera_control import CameraControlBuffer, OrbitCameraState
from .dashboard import (
    DashboardEventStream,
    VisualizationCurve,
    normalize_curves,
)
from .presentation import (
    BrowserActorState,
    BrowserPresentationState,
    BrowserSceneStyle,
    VisualizationTrail,
    normalize_trails,
)
from .three_scene import BrowserThreeScene
from .web_server import LatestFrame, VisualizationServer, build_request_handler


def _cuda_device_ordinal(device: object) -> int | None:
    """Return the explicit CUDA ordinal, or ``None`` for non-CUDA devices."""

    value = str(device)
    if value == "cuda":
        return 0
    prefix = "cuda:"
    if not value.startswith(prefix):
        return None
    ordinal = value[len(prefix) :]
    if not ordinal.isdecimal():
        return None
    return int(ordinal)


def _disable_unsupported_camera_fabric_view(camera: Any) -> bool:
    """Use USD camera poses when USDRT cannot target the simulation GPU.

    Isaac Lab 2.3.2's ``XformPrimView`` forwards its tensor device to
    ``UsdStage.SelectPrims``.  The USDRT version bundled with Isaac Sim 5.1
    supports that operation only on ``cuda:0``.  Physics and RTX rendering do
    support other GPUs, so only the camera pose view falls back to USD; the
    simulation-wide Fabric setting remains untouched.

    Newer Isaac Lab implementations may already select a safe view and no
    longer expose ``_use_fabric``.  In that case there is nothing to backport.
    """

    ordinal = _cuda_device_ordinal(camera.device)
    if ordinal is None or ordinal == 0:
        return False

    view = getattr(camera, "_view", None)
    if view is None:
        raise RuntimeError(
            "camera initialized without an Xform view; cannot apply the "
            "Isaac Sim 5.1 nonzero-CUDA compatibility path"
        )
    if not hasattr(view, "_use_fabric"):
        return False
    if not bool(view._use_fabric):
        return False
    if bool(getattr(view, "_fabric_initialized", False)):
        raise RuntimeError(
            "camera Fabric view was used before the nonzero-CUDA "
            "compatibility path could be applied"
        )

    # This is the same narrow fallback used by newer Isaac Lab releases for
    # nonzero CUDA devices.  Do not change /physics/fabricEnabled: articulation
    # transforms must still reach the RTX renderer through scene Fabric.
    view._use_fabric = False
    return True


def _camera_type_with_nonzero_cuda_fallback(camera_type: type) -> type:
    """Wrap an Isaac Camera so its pose view is fixed before first use."""

    class _VisualizationCamera(camera_type):
        def _initialize_impl(self) -> None:
            super()._initialize_impl()
            if _disable_unsupported_camera_fabric_view(self):
                print(
                    "[INFO][Visualization]: USDRT SelectPrims supports only "
                    "cuda:0; camera poses use the USD view while PhysX, RTX, "
                    f"and camera tensors remain on {self.device}.",
                    flush=True,
                )

    _VisualizationCamera.__name__ = f"Visualization{camera_type.__name__}"
    _VisualizationCamera.__qualname__ = _VisualizationCamera.__name__
    return _VisualizationCamera


class _LatestJpegEncoder:
    """Encode off the simulation thread while retaining only the newest frame."""

    def __init__(
        self,
        frames: LatestFrame,
        *,
        image_type: Any,
        quality: int,
    ) -> None:
        self._frames = frames
        self._image_type = image_type
        self._quality = quality
        self._condition = Condition()
        self._pending: Any | None = None
        self._error: Exception | None = None
        self._closed = False
        self._thread = Thread(
            target=self._run,
            name="visualization-jpeg",
            daemon=True,
        )
        self._thread.start()

    def submit(self, frame: Any) -> None:
        with self._condition:
            self._raise_if_failed()
            if self._closed:
                raise RuntimeError("JPEG encoder is closed")
            self._pending = frame
            self._condition.notify()

    def check_health(self) -> None:
        with self._condition:
            self._raise_if_failed()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("background JPEG encoding failed") from self._error

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or self._pending is not None
                )
                if self._closed:
                    return
                frame = self._pending
                self._pending = None

            try:
                output = BytesIO()
                self._image_type.fromarray(frame).save(
                    output,
                    format="JPEG",
                    quality=self._quality,
                )
                jpeg = output.getvalue()
            except Exception as error:
                with self._condition:
                    self._error = error
                    self._closed = True
                    self._pending = None
                    self._condition.notify_all()
                return

            with self._condition:
                if self._closed:
                    return
                if self._pending is not None:
                    continue
                self._frames.publish(jpeg)


class Visualization:
    """Preview one environment through an interactive browser presentation.

    The class only relies on the scene layout interface: ``num_envs`` and
    ``env_origins``. It does not inspect scene entities and has no dependency
    on an environment, policy, or algorithm.

    Camera-backed instances must be constructed before the first ``sim.reset()``.
    Browser-native Three.js instances may set ``camera_enabled=False`` and be
    constructed after an externally owned environment has initialized. Each
    simulation tick should call :meth:`initialize`, then
    :meth:`dispatch_button_callbacks` and :meth:`apply_camera_controls` before
    stepping, followed by :meth:`update` after the scene has been updated.
    """

    _FOCAL_LENGTH = 24.0
    _HORIZONTAL_APERTURE = 20.955

    def __init__(
        self,
        scene: Any,
        *,
        num_envs: int | None = None,
        env_index: int = 0,
        camera_eye: tuple[float, float, float] = (0.6, -1.8, 1.3),
        camera_target: tuple[float, float, float] = (0.0, 0.4, 0.5),
        resolution: tuple[int, int] = (1280, 960),
        host: str = "127.0.0.1",
        port: int = 8000,
        max_fps: float = 20.0,
        jpeg_quality: int = 90,
        title: str = "Isaac Sim",
        page_background: str = "#111111",
        buttons: Iterable[VisualizationButton] = (),
        curves: Iterable[VisualizationCurve] = (),
        show_output: bool = False,
        scene_style: BrowserSceneStyle | None = None,
        trails: Iterable[VisualizationTrail] = (),
        three_scene: BrowserThreeScene | None = None,
        camera_enabled: bool = True,
        frame_sink: Callable[[Any], None] | None = None,
    ) -> None:
        from .theme import apply_scene_theme
        if three_scene is not None:
            three_scene = apply_scene_theme(three_scene)
        frozen_buttons = normalize_buttons(buttons)
        frozen_curves = normalize_curves(curves)
        frozen_trails = normalize_trails(trails)
        if not isinstance(show_output, bool):
            raise TypeError("show_output must be a bool")
        if scene_style is not None and not isinstance(scene_style, BrowserSceneStyle):
            raise TypeError("scene_style must be a BrowserSceneStyle or None")
        if three_scene is not None and not isinstance(three_scene, BrowserThreeScene):
            raise TypeError("three_scene must be a BrowserThreeScene or None")
        if not isinstance(camera_enabled, bool):
            raise TypeError("camera_enabled must be a bool")
        if frame_sink is not None and not callable(frame_sink):
            raise TypeError("frame_sink must be callable or None")
        actual_num_envs = int(scene.num_envs)
        if num_envs is not None and int(num_envs) != actual_num_envs:
            raise ValueError(
                "num_envs does not match scene.num_envs: "
                f"expected={actual_num_envs} actual={num_envs}"
            )
        if not 0 <= env_index < actual_num_envs:
            raise ValueError(
                f"env_index must be in [0, {actual_num_envs}): {env_index}"
            )
        if len(camera_eye) != 3 or len(camera_target) != 3:
            raise ValueError("camera_eye and camera_target must be xyz triples")
        width, height = (int(value) for value in resolution)
        if width <= 0 or height <= 0:
            raise ValueError("resolution must contain positive width and height")
        if not 0 <= port <= 65535:
            raise ValueError("port must be in [0, 65535]")
        if max_fps <= 0.0:
            raise ValueError("max_fps must be positive")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")

        try:
            import numpy as np
            from PIL import Image
        except ImportError as error:
            raise RuntimeError(
                "visualization requires NumPy and Pillow in the Isaac environment"
            ) from error

        self._np = np
        self._scene = scene
        self._env_index = env_index
        self._camera_enabled = camera_enabled
        self._frame_sink = frame_sink
        self._resolution = (width, height)
        horizontal_fov = 2.0 * atan(
            self._HORIZONTAL_APERTURE / (2.0 * self._FOCAL_LENGTH)
        )
        vertical_fov = 2.0 * atan(tan(0.5 * horizontal_fov) * height / width)
        self._camera_view = OrbitCameraState(
            camera_eye,
            camera_target,
            vertical_fov=vertical_fov,
        )
        self._camera_vertical_fov = vertical_fov
        self._camera_aspect = width / height
        self._camera_origin = None
        self._publish_period = 1.0 / float(max_fps)
        self._last_publish_time = float("-inf")
        self._elapsed_since_capture = 0.0
        self._simulation_thread_id = get_ident()
        self._initialized = False
        self._closed = False
        self._button_callbacks = {
            button.button_id: button.callback for button in frozen_buttons
        }
        self._button_events = ButtonEventBuffer(self._button_callbacks)
        self._dashboard_events = DashboardEventStream(
            (curve.metric_id for curve in frozen_curves),
            log_enabled=show_output,
        )
        self._presentation_state = (
            BrowserPresentationState(
                frozen_trails,
                actor_joint_names=(
                    None if three_scene is None else three_scene.actor_joint_names
                ),
                polyline_limits=(
                    None if three_scene is None else three_scene.polyline_limits
                ),
            )
            if scene_style is not None or frozen_trails or three_scene is not None
            else None
        )
        self._three_scene_enabled = three_scene is not None
        self._update_browser_presentation_view()

        self._camera = None
        if camera_enabled:
            import isaaclab.sim as sim_utils
            from isaaclab.sensors.camera import Camera, CameraCfg

            VisualizationCamera = _camera_type_with_nonzero_cuda_fallback(Camera)
            camera_cfg = CameraCfg(
                prim_path="/World/VisualizationCamera",
                update_period=0.0,
                height=height,
                width=width,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=self._FOCAL_LENGTH,
                    focus_distance=400.0,
                    horizontal_aperture=self._HORIZONTAL_APERTURE,
                    clipping_range=(0.1, 100.0),
                ),
            )
            self._camera = VisualizationCamera(camera_cfg)

        self._frames = LatestFrame()
        self._controls = CameraControlBuffer()
        handler = build_request_handler(
            self._frames,
            self._controls,
            title=title,
            background=page_background,
            buttons=frozen_buttons,
            button_events=self._button_events,
            curves=frozen_curves,
            dashboard_events=self._dashboard_events,
            show_output=show_output,
            scene_style=scene_style,
            trails=frozen_trails,
            presentation_state=self._presentation_state,
            three_scene=three_scene,
            resolution=(width, height),
        )
        self._server = VisualizationServer((host, port), handler)
        self._encoder = _LatestJpegEncoder(
            self._frames,
            image_type=Image,
            quality=int(jpeg_quality),
        )
        self._server_thread = Thread(
            target=self._server.serve_forever,
            name="visualization-http",
            daemon=True,
        )
        self._server_thread.start()

    @property
    def url(self) -> str:
        """Browser URL for the visualization page."""

        host, port = self._server.server_address[:2]
        display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        return f"http://{display_host}:{port}/"

    @property
    def published_frames(self) -> int:
        """Number of frames encoded and published to the browser."""

        return self._frames.sequence

    @property
    def three_scene_enabled(self) -> bool:
        """Whether this browser uses the independent Three.js geometry scene."""

        return self._three_scene_enabled

    @property
    def camera_enabled(self) -> bool:
        """Whether this presentation owns an Isaac RGB camera."""

        return self._camera_enabled

    def initialize(self) -> None:
        """Position the initialized camera over the selected environment."""

        self._assert_simulation_thread()
        if self._closed:
            raise RuntimeError("visualization is closed")
        if self._camera is None:
            self._initialized = True
            width, height = self._resolution
            self.publish(self._np.zeros((height, width, 3), dtype=self._np.uint8))
            return
        if not self._camera.is_initialized:
            raise RuntimeError(
                "camera is not initialized; construct Visualization before "
                "sim.reset() and enable cameras in AppLauncher"
            )

        import torch

        self._camera_origin = self._scene.env_origins[self._env_index].to(
            device=self._camera.device,
            dtype=torch.float32,
        )
        self._set_camera_pose()
        self._camera.reset()
        self._initialized = True

    def apply_camera_controls(self) -> bool:
        """Apply queued browser input from the simulation thread.

        Call this before the next simulation/render step so the frame captured
        by :meth:`update` already uses the new pose.
        """

        self._assert_simulation_thread()
        if not self._initialized:
            raise RuntimeError("call initialize() after sim.reset() first")
        if self._closed:
            return False
        command = self._controls.consume()
        changed = self._camera_view.apply(command)
        if changed and self._camera is not None:
            self._set_camera_pose()
        if changed or command.client_sequence > 0:
            # Echo the command sequence in the authoritative view so a
            # browser-local Three.js preview can reconcile without snap-back.
            self._update_browser_presentation_view(
                client_sequence=command.client_sequence,
            )
        return changed

    def _update_browser_presentation_view(
        self,
        *,
        client_sequence: int = 0,
    ) -> None:
        if self._presentation_state is None:
            return
        eye, target = self._camera_view.pose()
        self._presentation_state.update_view(
            eye,
            target,
            vertical_fov=self._camera_vertical_fov,
            aspect=self._camera_aspect,
            client_sequence=client_sequence,
        )

    def dispatch_button_callbacks(self) -> int:
        """Invoke queued custom-button callbacks on the simulation thread.

        Browser and HTTP worker threads only enqueue identifiers. Call this
        method once before each simulation step to perform any environment or
        scene mutation safely. Callbacks run synchronously in click order; an
        exception is reported with the failing button identifier.
        """

        self._assert_simulation_thread()
        if not self._initialized:
            raise RuntimeError("call initialize() after sim.reset() first")
        if self._closed:
            return 0

        dispatched = 0
        while True:
            button_id = self._button_events.consume_one()
            if button_id is None:
                return dispatched
            callback = self._button_callbacks.get(button_id)
            if callback is None:
                raise RuntimeError(
                    "queued unknown visualization button identifier: " f"{button_id!r}"
                )
            try:
                callback()
            except Exception as error:
                raise RuntimeError(
                    f"visualization button callback failed: {button_id!r}"
                ) from error
            dispatched += 1

    def _set_camera_pose(self) -> None:
        if self._camera is None:
            raise RuntimeError("browser-only visualization has no Isaac camera")
        if self._camera_origin is None:
            raise RuntimeError("camera origin is not initialized")

        import torch

        eye_values, target_values = self._camera_view.pose()
        eye = self._camera_origin + torch.tensor(
            eye_values,
            device=self._camera.device,
            dtype=torch.float32,
        )
        target = self._camera_origin + torch.tensor(
            target_values,
            device=self._camera.device,
            dtype=torch.float32,
        )
        self._camera.set_world_poses_from_view(
            eye.unsqueeze(0),
            target.unsqueeze(0),
        )

    def _assert_simulation_thread(self) -> None:
        if get_ident() != self._simulation_thread_id:
            raise RuntimeError(
                "visualization callbacks and Isaac camera APIs must run on "
                "the simulation thread"
            )

    def update(self, dt: float, *, force_publish: bool = False) -> bool:
        """Capture and publish the current camera frame at most ``max_fps``."""

        self._assert_simulation_thread()
        if not self._initialized:
            raise RuntimeError("call initialize() after sim.reset() first")
        if self._closed:
            return False

        self._encoder.check_health()
        if self._camera is None:
            return False
        self._elapsed_since_capture += float(dt)
        now = monotonic()
        if not force_publish and now - self._last_publish_time < self._publish_period:
            return False

        self._camera.update(self._elapsed_since_capture)
        self._elapsed_since_capture = 0.0
        outputs = self._camera.data.output
        frame = outputs.get("rgb")
        if frame is None:
            frame = outputs.get("rgba")
        if frame is None:
            raise RuntimeError("camera did not produce an RGB frame")
        self.publish(frame[0, ..., :3])
        self._last_publish_time = now
        return True

    def publish(self, frame: Any) -> None:
        """Queue one HWC RGB/RGBA frame for latest-only JPEG encoding."""

        if self._closed:
            raise RuntimeError("visualization is closed")
        if hasattr(frame, "detach"):
            frame = frame.detach().to(device="cpu").numpy()
        else:
            frame = self._np.asarray(frame)
        if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            raise ValueError(
                "frame must have shape (height, width, 3|4): " f"actual={frame.shape}"
            )
        frame = frame[..., :3]
        if frame.dtype != self._np.uint8:
            if self._np.issubdtype(frame.dtype, self._np.floating):
                frame = self._np.nan_to_num(
                    frame,
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                )
                if frame.size and float(frame.max()) <= 1.0:
                    frame = frame * 255.0
            frame = self._np.clip(frame, 0, 255).astype(self._np.uint8)
        # The encoder outlives this simulation tick, so it must not retain a
        # view into a camera buffer that the next sensor update may reuse.
        frame = self._np.array(frame, copy=True, order="C")

        if self._frame_sink is not None:
            self._frame_sink(frame)
        self._encoder.submit(frame)

    def publish_metrics(
        self,
        values: Mapping[str, Real],
        *,
        time_s: float | None = None,
    ) -> int:
        """Publish registered scalar metrics without touching Isaac APIs.

        The method is thread-safe. Producers should decimate high-rate physics
        values before publishing; the browser retains only a bounded history.
        """

        if self._closed:
            raise RuntimeError("visualization is closed")
        return self._dashboard_events.publish_metrics(values, time_s=time_s)

    def write_log(
        self,
        message: str,
        *,
        level: str = "info",
        source: str = "",
        time_s: float | None = None,
    ) -> int:
        """Publish one structured line to the optional browser output panel."""

        if self._closed:
            raise RuntimeError("visualization is closed")
        return self._dashboard_events.publish_log(
            message,
            level=level,
            source=source,
            time_s=time_s,
        )

    def publish_trail_point(
        self,
        trail_id: str,
        point_env_m: tuple[Real, Real, Real],
    ) -> int:
        """Append one selected-environment-local XYZ point to a browser trail."""

        if self._closed:
            raise RuntimeError("visualization is closed")
        if self._presentation_state is None:
            raise RuntimeError("browser presentation is not enabled")
        return self._presentation_state.append_trail_point(trail_id, point_env_m)

    def set_trail_points(
        self,
        trail_id: str,
        points_env_m: Iterable[tuple[Real, Real, Real]],
    ) -> int:
        """Replace one browser trail with bounded environment-local XYZ points."""

        if self._closed:
            raise RuntimeError("visualization is closed")
        if self._presentation_state is None:
            raise RuntimeError("browser presentation is not enabled")
        return self._presentation_state.set_trail_points(trail_id, points_env_m)

    def clear_trail(self, trail_id: str) -> int:
        """Clear one browser-only trajectory without changing simulator state."""

        if self._closed:
            raise RuntimeError("visualization is closed")
        if self._presentation_state is None:
            raise RuntimeError("browser presentation is not enabled")
        return self._presentation_state.clear_trail(trail_id)

    def set_polyline_overlay(
        self,
        overlay_id: str,
        polylines_env_m: Iterable[Iterable[tuple[Real, Real, Real]]],
    ) -> int:
        """Replace one bounded browser-native line batch."""

        if self._closed:
            raise RuntimeError("visualization is closed")
        if self._presentation_state is None:
            raise RuntimeError("browser presentation is not enabled")
        return self._presentation_state.set_polyline_overlay(
            overlay_id,
            polylines_env_m,
        )

    def set_recording_paused(self, paused: bool) -> int:
        """Pause or resume browser recordings without stopping rendering."""

        if self._closed:
            raise RuntimeError("visualization is closed")
        if self._presentation_state is None:
            raise RuntimeError("browser presentation is not enabled")
        return self._presentation_state.set_recording_paused(paused)

    def set_three_scene_actor_states(
        self,
        states: Iterable[BrowserActorState],
    ) -> int:
        """Atomically publish all browser-native visual actor poses."""

        if self._closed:
            raise RuntimeError("visualization is closed")
        if self._presentation_state is None or not self._presentation_state.actor_ids:
            raise RuntimeError("browser Three.js scene is not enabled")
        return self._presentation_state.set_actor_states(states)

    def publish_presentation_frame(
        self,
        *,
        actor_states: Iterable[BrowserActorState] | None = None,
        trail_points: Mapping[str, tuple[Real, Real, Real]] | None = None,
        polyline_overlays: Mapping[
            str,
            Iterable[Iterable[tuple[Real, Real, Real]]],
        ]
        | None = None,
    ) -> int:
        """Atomically publish one browser actor/trail update.

        Grouping values that describe the same simulator step prevents the
        browser from receiving partial snapshots and avoids redundant redraws.
        """

        if self._closed:
            raise RuntimeError("visualization is closed")
        if self._presentation_state is None:
            raise RuntimeError("browser presentation is not enabled")
        return self._presentation_state.update_frame(
            actor_states=actor_states,
            trail_points=trail_points,
            polyline_overlays=polyline_overlays,
        )

    def close(self) -> None:
        """Stop the HTTP server and release its client threads."""

        if self._closed:
            return
        self._closed = True
        self._button_events.close()
        self._dashboard_events.close()
        if self._presentation_state is not None:
            self._presentation_state.close()
        self._encoder.close()
        self._frames.close()
        self._server.close_websockets()
        self._server.shutdown()
        self._server.server_close()
        self._server_thread.join(timeout=2.0)

    def __enter__(self) -> Visualization:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = [
    "BrowserActorState",
    "BrowserSceneStyle",
    "BrowserThreeScene",
    "Visualization",
    "VisualizationButton",
    "VisualizationCurve",
    "VisualizationTrail",
]
