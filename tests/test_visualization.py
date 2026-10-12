"""Tests for the simulator-independent browser visualization control plane."""

from __future__ import annotations

import gzip
import json
from math import dist, isfinite, pi, sqrt
import socket
import struct
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread, get_ident
import types
import unittest
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from trajectory_tracking.visualization.buttons import (
    ButtonEventBuffer,
    VisualizationButton,
    normalize_buttons,
)
from trajectory_tracking.visualization.browser import build_browser_page
from trajectory_tracking.visualization.camera_control import (
    CameraCommand,
    CameraControlBuffer,
    OrbitCameraState,
)
from trajectory_tracking.visualization.dashboard import (
    DashboardEventStream,
    VisualizationCurve,
    normalize_curves,
)
from trajectory_tracking.visualization.presentation import (
    BrowserActorState,
    BrowserPresentationState,
    BrowserSceneStyle,
    VisualizationTrail,
    normalize_trails,
)
from trajectory_tracking.visualization.three_scene import (
    BrowserPolylineOverlay,
    BrowserPrimitiveActor,
    BrowserThreeScene,
    BrowserUrdfActor,
)
from trajectory_tracking.visualization.web_server import (
    LatestFrame,
    VisualizationServer,
    build_request_handler,
)
from trajectory_tracking.visualization.visualization import (
    Visualization,
    _LatestJpegEncoder,
    _camera_type_with_nonzero_cuda_fallback,
    _cuda_device_ordinal,
    _disable_unsupported_camera_fabric_view,
)


def _recv_exact(client: socket.socket, length: int) -> bytes:
    result = bytearray()
    while len(result) < length:
        chunk = client.recv(length - len(result))
        if not chunk:
            raise AssertionError("WebSocket closed before the frame completed")
        result.extend(chunk)
    return bytes(result)


def _send_client_websocket_frame(
    client: socket.socket,
    opcode: int,
    payload: bytes,
) -> None:
    if len(payload) > 125:
        raise AssertionError("test helper only supports short frames")
    mask = b"\x12\x34\x56\x78"
    masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    client.sendall(bytes((0x80 | opcode, 0x80 | len(payload))) + mask + masked)


def _recv_server_websocket_frame(client: socket.socket) -> tuple[int, bytes]:
    first, second = _recv_exact(client, 2)
    if second & 0x80:
        raise AssertionError("server WebSocket frames must not be masked")
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", _recv_exact(client, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _recv_exact(client, 8))[0]
    return first & 0x0F, _recv_exact(client, length)


class OrbitCameraStateTest(unittest.TestCase):
    def test_invalid_or_pole_initial_view_is_handled_safely(self) -> None:
        with self.assertRaises(ValueError):
            OrbitCameraState((float("nan"), 0.0, 1.0), (0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            OrbitCameraState(
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                vertical_fov=pi,
            )

        state = OrbitCameraState((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        eye, target = state.pose()
        self.assertTrue(all(isfinite(value) for value in eye + target))
        self.assertNotEqual((eye[0], eye[1]), (target[0], target[1]))

    def test_orbit_zoom_pan_and_reset(self) -> None:
        state = OrbitCameraState(
            eye=(0.6, -1.8, 1.3),
            target=(0.0, 0.4, 0.5),
        )
        initial_eye, initial_target = state.pose()
        initial_distance = dist(initial_eye, initial_target)

        self.assertTrue(state.apply(CameraCommand(orbit_x=0.1, orbit_y=-0.05)))
        orbit_eye, orbit_target = state.pose()
        self.assertNotEqual(orbit_eye, initial_eye)
        self.assertEqual(orbit_target, initial_target)
        self.assertAlmostEqual(dist(orbit_eye, orbit_target), initial_distance)

        self.assertTrue(state.apply(CameraCommand(zoom=100.0)))
        zoom_eye, zoom_target = state.pose()
        self.assertGreater(dist(zoom_eye, zoom_target), initial_distance)

        self.assertTrue(state.apply(CameraCommand(pan_x=0.03, pan_y=-0.015)))
        pan_eye, pan_target = state.pose()
        self.assertNotEqual(pan_target, initial_target)
        self.assertTrue(all(isfinite(value) for value in pan_eye + pan_target))

        self.assertTrue(state.apply(CameraCommand(reset=True)))
        reset_eye, reset_target = state.pose()
        for actual, expected in zip(reset_eye, initial_eye, strict=True):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(reset_target, initial_target, strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_extreme_orbit_and_zoom_remain_finite(self) -> None:
        state = OrbitCameraState((1.0, -1.0, 1.0), (0.0, 0.0, 0.0))
        state.apply(
            CameraCommand(
                orbit_x=10_000.0,
                orbit_y=10_000.0,
                zoom=-10_000.0,
                pan_x=10_000.0,
                pan_y=10_000.0,
            )
        )
        eye, target = state.pose()
        self.assertTrue(all(isfinite(value) for value in eye + target))
        self.assertGreaterEqual(dist(eye, target), 0.1)

    def test_orbit_and_pan_follow_screen_drag_directions(self) -> None:
        horizontal = OrbitCameraState(
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            vertical_fov=pi / 2.0,
        )
        horizontal.apply(CameraCommand(orbit_x=0.25))
        eye, _ = horizontal.pose()
        self.assertAlmostEqual(eye[0], 0.0, places=7)
        self.assertAlmostEqual(eye[1], -1.0, places=7)
        self.assertAlmostEqual(eye[2], 0.0, places=7)

        vertical = OrbitCameraState(
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            vertical_fov=pi / 2.0,
        )
        vertical.apply(CameraCommand(orbit_y=0.125))
        eye, _ = vertical.pose()
        self.assertAlmostEqual(eye[0], sqrt(0.5), places=7)
        self.assertAlmostEqual(eye[1], 0.0, places=7)
        self.assertAlmostEqual(eye[2], sqrt(0.5), places=7)

        horizontal_pan = OrbitCameraState(
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            vertical_fov=pi / 2.0,
        )
        horizontal_pan.apply(CameraCommand(pan_x=0.1))
        _, target = horizontal_pan.pose()
        self.assertAlmostEqual(target[0], 0.0, places=7)
        self.assertAlmostEqual(target[1], -0.2, places=7)
        self.assertAlmostEqual(target[2], 0.0, places=7)

        vertical_pan = OrbitCameraState(
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            vertical_fov=pi / 2.0,
        )
        vertical_pan.apply(CameraCommand(pan_y=0.1))
        _, target = vertical_pan.pose()
        self.assertAlmostEqual(target[0], 0.0, places=7)
        self.assertAlmostEqual(target[1], 0.0, places=7)
        self.assertAlmostEqual(target[2], 0.2, places=7)

    def test_small_orbit_batches_match_one_normalized_drag(self) -> None:
        incremental = OrbitCameraState((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        combined = OrbitCameraState((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for _ in range(100):
            incremental.apply(CameraCommand(orbit_x=0.001, orbit_y=0.0005))
        combined.apply(CameraCommand(orbit_x=0.1, orbit_y=0.05))
        incremental_eye, incremental_target = incremental.pose()
        combined_eye, combined_target = combined.pose()
        for actual, expected in zip(incremental_eye, combined_eye, strict=True):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(incremental_target, combined_target)


class CameraControlBufferTest(unittest.TestCase):
    def test_commands_accumulate_and_are_consumed_once(self) -> None:
        controls = CameraControlBuffer()
        controls.submit({"orbit_x": 0.02, "zoom": 3.0})
        controls.submit({"orbit_x": 0.04, "pan_y": -0.05})

        self.assertEqual(
            controls.consume(),
            CameraCommand(orbit_x=0.06, pan_y=-0.05, zoom=3.0),
        )
        self.assertEqual(controls.consume(), CameraCommand())

    def test_reset_discards_queued_motion(self) -> None:
        controls = CameraControlBuffer()
        controls.submit({"orbit_x": 0.2})
        controls.submit({"reset": True})
        self.assertEqual(controls.consume(), CameraCommand(reset=True))

    def test_motion_after_reset_is_preserved(self) -> None:
        controls = CameraControlBuffer()
        controls.submit({"orbit_x": 0.2})
        controls.submit({"reset": True})
        controls.submit({"orbit_x": 0.05, "zoom": 2.0})
        command = controls.consume()
        self.assertEqual(
            command,
            CameraCommand(orbit_x=0.05, zoom=2.0, reset=True),
        )

        state = OrbitCameraState((1.0, -1.0, 1.0), (0.0, 0.0, 0.0))
        default_pose = state.pose()
        state.apply(CameraCommand(orbit_x=0.1))
        state.apply(command)
        self.assertNotEqual(state.pose(), default_pose)

    def test_client_sequence_is_coalesced_and_consumed_with_motion(self) -> None:
        controls = CameraControlBuffer()
        controls.submit({"orbit_x": 0.02, "client_sequence": 41})
        controls.submit({"pan_y": -0.01, "client_sequence": 42})
        self.assertEqual(
            controls.consume(),
            CameraCommand(
                orbit_x=0.02,
                pan_y=-0.01,
                client_sequence=42,
            ),
        )
        self.assertEqual(controls.consume(), CameraCommand())

    def test_invalid_command_fails(self) -> None:
        controls = CameraControlBuffer()
        for payload in (
            {},
            {"unknown": 1},
            {"zoom": "far"},
            {"zoom": float("nan")},
            {"orbit_x": 0.1, "client_sequence": -1},
            {"orbit_x": 0.1, "client_sequence": 1.5},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                controls.submit(payload)


class VisualizationButtonTest(unittest.TestCase):
    def test_button_definition_and_collection_are_validated(self) -> None:
        callback = lambda: None
        button = VisualizationButton(
            "reset_object",
            "重置物体",
            callback,
            style="primary",
        )
        self.assertEqual(normalize_buttons([button]), (button,))

        for button_id in ("", "1reset", "reset object", "../reset"):
            with self.subTest(button_id=button_id), self.assertRaises(ValueError):
                VisualizationButton(button_id, "重置", callback)
        with self.assertRaises(ValueError):
            VisualizationButton("reset", " 重置 ", callback)
        with self.assertRaises(ValueError):
            VisualizationButton("reset", "重置", callback, style="unknown")
        with self.assertRaises(TypeError):
            VisualizationButton("reset", "重置", object())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            VisualizationButton(
                "reset",
                "重置",
                callback,
                style=object(),  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            normalize_buttons(
                [
                    VisualizationButton("reset", "重置", callback),
                    VisualizationButton("reset", "再重置", callback),
                ]
            )

    def test_button_event_buffer_is_bounded_fifo(self) -> None:
        events = ButtonEventBuffer(("reset", "release"), max_pending=2)
        events.submit("reset")
        events.submit("release")
        self.assertEqual(events.pending_count, 2)
        with self.assertRaises(OverflowError):
            events.submit("reset")
        with self.assertRaises(ValueError):
            events.submit("unknown")

        self.assertEqual(events.consume_one(), "reset")
        self.assertEqual(events.consume_one(), "release")
        self.assertIsNone(events.consume_one())
        events.close()
        with self.assertRaises(RuntimeError):
            events.submit("reset")

    def test_page_omits_empty_panel_and_escapes_button_labels(self) -> None:
        empty_page = build_browser_page(
            title="viewer",
            background="#111111",
        ).decode("utf-8")
        self.assertNotIn('id="environment-actions"', empty_page)

        button_page = build_browser_page(
            title="viewer",
            background="#111111",
            buttons=(
                VisualizationButton(
                    "reset",
                    "<重置>",
                    lambda: None,
                    style="danger",
                ),
            ),
        ).decode("utf-8")
        self.assertIn('data-button-id="reset"', button_page)
        self.assertIn('data-style="danger"', button_page)
        self.assertIn("&lt;重置&gt;", button_page)
        self.assertNotIn("><重置></button>", button_page)


class VisualizationDashboardTest(unittest.TestCase):
    def test_curve_definition_and_collection_are_validated(self) -> None:
        curve = VisualizationCurve(
            "link7_height",
            "机械臂高度",
            color="#66BB6A",
            unit="m",
        )
        self.assertEqual(normalize_curves((curve,)), (curve,))

        for metric_id in ("", "1height", "link7 height", "../height"):
            with self.subTest(metric_id=metric_id), self.assertRaises(ValueError):
                VisualizationCurve(metric_id, "高度")
        with self.assertRaises(ValueError):
            VisualizationCurve("height", " 高度")
        with self.assertRaises(ValueError):
            VisualizationCurve("height", "高度", color="red")
        with self.assertRaisesRegex(ValueError, "unique"):
            normalize_curves(
                (
                    VisualizationCurve("height", "高度"),
                    VisualizationCurve("height", "另一高度"),
                )
            )

    def test_dashboard_event_stream_is_bounded_and_validated(self) -> None:
        stream = DashboardEventStream(
            ("height", "left_right"),
            log_enabled=True,
            max_events=2,
        )
        self.assertEqual(
            stream.publish_metrics(
                {"height": 0.5, "left_right": -0.1},
                time_s=0.0,
            ),
            1,
        )
        self.assertEqual(
            stream.publish_log(
                "机械臂上移",
                source="button",
                time_s=0.1,
            ),
            2,
        )
        self.assertEqual(
            stream.publish_metrics({"height": 0.6}, time_s=0.2),
            3,
        )

        batch = stream.wait_after(0, timeout=0.0)
        self.assertEqual(batch.sequence, 3)
        self.assertEqual(batch.dropped, 1)
        self.assertEqual(
            tuple(event["sequence"] for event in batch.events),
            (2, 3),
        )
        self.assertEqual(batch.events[0]["type"], "log")
        self.assertEqual(batch.events[1]["values"], {"height": 0.6})

        with self.assertRaisesRegex(ValueError, "unknown"):
            stream.publish_metrics({"unknown": 1.0}, time_s=0.3)
        with self.assertRaisesRegex(ValueError, "finite"):
            stream.publish_metrics({"height": float("nan")}, time_s=0.3)
        with self.assertRaisesRegex(ValueError, "monotonically"):
            stream.publish_metrics({"height": 0.4}, time_s=0.1)
        with self.assertRaises(ValueError):
            stream.publish_log("two\nlines", time_s=0.3)
        stream.close()
        self.assertTrue(stream.wait_after(3, timeout=0.0).closed)
        with self.assertRaises(RuntimeError):
            stream.publish_log("closed")

    def test_page_builds_three_resizable_reorderable_dashboard_panels(self) -> None:
        page = build_browser_page(
            title="dashboard",
            background="#111111",
            buttons=(VisualizationButton("up", "上移", lambda: None),),
            curves=(
                VisualizationCurve(
                    "height",
                    "</script><高度>",
                    color="#66BB6A",
                    unit="m",
                ),
            ),
            show_output=True,
            resolution=(960, 720),
        ).decode("utf-8")

        for panel_id in ("buttons", "curves", "output"):
            self.assertIn(f'data-panel-id="{panel_id}"', page)
        self.assertIn('id="dashboard-width-handle"', page)
        self.assertIn('class="panel-resize-handle"', page)
        self.assertIn('draggable="true"', page)
        self.assertIn("localStorage.setItem(dashboardLayoutKey", page)
        self.assertIn("dashboard.insertBefore(draggedPanel", page)
        self.assertIn("new ResizeObserver(scheduleCurveDraw)", page)
        self.assertIn("/dashboard?after=", page)
        self.assertIn("&lt;/script&gt;&lt;高度&gt;", page)
        self.assertNotIn("</script><高度>", page)
        self.assertIn('width="960" height="720"', page)
        for typography_fragment in (
            "--ui-control-font-size:8px",
            "--ui-detail-font-size:7px",
            "font-size:var(--ui-control-font-size)",
            "font-size:var(--ui-detail-font-size)",
            "rootComputedStyle.getPropertyValue('--ui-control-font-size')",
            "rootComputedStyle.getPropertyValue('--ui-detail-font-size')",
            "`600 ${uiFontPixels.control}px sans-serif`",
            "`${uiFontPixels.detail}px monospace`",
        ):
            self.assertIn(typography_fragment, page)
        for stale_font_size in (
            "font-size:10px",
            "font-size:11px",
            "font-size:12px",
            "--ui-control-font-size:9px",
            "--ui-detail-font-size:8px",
            "600 10px sans-serif",
            "9px monospace",
        ):
            self.assertNotIn(stale_font_size, page)

    def test_visualization_facade_publishes_dashboard_events(self) -> None:
        stream = DashboardEventStream(("height",), log_enabled=True)
        visualization = object.__new__(Visualization)
        visualization._closed = False
        visualization._dashboard_events = stream

        self.assertEqual(
            visualization.publish_metrics({"height": 0.7}, time_s=2.0),
            1,
        )
        self.assertEqual(
            visualization.write_log(
                "机械臂上移",
                source="button",
                time_s=2.0,
            ),
            2,
        )
        batch = stream.wait_after(0, timeout=0.0)
        self.assertEqual(
            tuple(event["type"] for event in batch.events),
            ("metrics", "log"),
        )

        visualization._closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            visualization.publish_metrics({"height": 0.8}, time_s=3.0)


class BrowserPresentationTest(unittest.TestCase):
    def test_style_and_trail_contracts_are_strict(self) -> None:
        style = BrowserSceneStyle(
            background_color="#0b0f15",
            grid_divisions=24,
        )
        self.assertEqual(style.background_color, "#0B0F15")
        self.assertEqual(style.as_payload()["grid_divisions"], 24)
        self.assertEqual(style.grid_plane_z_m, -0.1)
        self.assertEqual(style.color_grade_strength, 0.30)
        self.assertEqual(style.light_strength, 0.22)

        trail = VisualizationTrail(
            "link7",
            "Link7",
            color="#ff6b14",
            max_points=3,
        )
        self.assertEqual(normalize_trails((trail,)), (trail,))
        self.assertEqual(trail.color, "#FF6B14")
        for kwargs in (
            {"background_color": "navy"},
            {"grid_size_m": 0.0},
            {"grid_divisions": 1},
            {"light_strength": 1.1},
        ):
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaises((TypeError, ValueError)),
            ):
                BrowserSceneStyle(**kwargs)
        with self.assertRaises(ValueError):
            VisualizationTrail("../link7", "Link7")
        with self.assertRaisesRegex(ValueError, "unique"):
            normalize_trails((trail, trail))

    def test_presentation_state_keeps_latest_view_and_bounded_3d_trails(self) -> None:
        trail = VisualizationTrail("link7", "Link7", max_points=3)
        state = BrowserPresentationState((trail,))
        self.assertEqual(
            state.update_view(
                (0.6, -1.8, 1.3),
                (0.0, 0.4, 0.5),
                vertical_fov=0.7,
                aspect=4 / 3,
            ),
            1,
        )
        for x_position in (0.0, 0.1, 0.2, 0.3):
            state.append_trail_point("link7", (x_position, 0.2, 0.7))
        snapshot = state.wait_after(0, timeout=0.0)
        self.assertEqual(snapshot.sequence, 5)
        self.assertEqual(snapshot.view["client_sequence"], 0)  # type: ignore[index]
        self.assertEqual(snapshot.trail_sequences["link7"], 5)
        self.assertEqual(
            snapshot.trails["link7"],
            ((0.1, 0.2, 0.7), (0.2, 0.2, 0.7), (0.3, 0.2, 0.7)),
        )
        self.assertEqual(snapshot.view["aspect"], 4 / 3)  # type: ignore[index]

        state.set_trail_points("link7", ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
        state.clear_trail("link7")
        self.assertEqual(state.wait_after(5, timeout=0.0).trails["link7"], ())
        with self.assertRaisesRegex(ValueError, "unknown"):
            state.append_trail_point("object", (0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "finite"):
            state.append_trail_point("link7", (float("nan"), 0.0, 0.0))
        state.close()
        self.assertTrue(state.wait_after(7, timeout=0.0).closed)

    def test_presentation_state_validates_atomic_three_scene_actor_updates(
        self,
    ) -> None:
        state = BrowserPresentationState(
            (),
            actor_joint_names={"robot": ("joint1",), "object": ()},
        )
        states = (
            BrowserActorState(
                "robot",
                (0.0, 0.0, 0.53),
                (0.0, 0.0, 0.0, 2.0),
                (0.4,),
            ),
            BrowserActorState(
                "object",
                (0.45, 0.0, 0.59),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )
        self.assertEqual(state.set_actor_states(states), 1)
        snapshot = state.wait_after(0, timeout=0.0)
        self.assertEqual(snapshot.actor_sequence, 1)
        self.assertEqual(snapshot.actors["robot"].joint_positions, (0.4,))
        self.assertTrue(snapshot.actors["robot"].visible)
        self.assertTrue(snapshot.actors["robot"].as_payload()["visible"])
        self.assertEqual(
            snapshot.actors["robot"].quaternion_xyzw,
            (0.0, 0.0, 0.0, 1.0),
        )
        with self.assertRaisesRegex(ValueError, "identifiers"):
            state.set_actor_states(states[:1])
        with self.assertRaisesRegex(ValueError, "requires 1"):
            state.set_actor_states(
                (
                    BrowserActorState(
                        "robot",
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0, 1.0),
                    ),
                    states[1],
                )
            )
        with self.assertRaisesRegex(TypeError, "visible"):
            BrowserActorState(
                "object",
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
                visible=1,  # type: ignore[arg-type]
            )

    def test_presentation_frame_updates_actors_and_trails_atomically(self) -> None:
        state = BrowserPresentationState(
            (VisualizationTrail("link7", "Link7"),),
            actor_joint_names={"robot": ("joint1",)},
        )
        actor = BrowserActorState(
            "robot",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.25,),
        )
        self.assertEqual(
            state.update_frame(
                actor_states=(actor,),
                trail_points={"link7": (0.4, 0.1, 0.7)},
            ),
            1,
        )
        snapshot = state.wait_after(0, timeout=0.0)
        self.assertEqual(snapshot.sequence, 1)
        self.assertEqual(snapshot.actor_sequence, 1)
        self.assertEqual(snapshot.trail_sequences, {"link7": 1})
        self.assertEqual(snapshot.actors["robot"], actor)
        self.assertEqual(snapshot.trails["link7"], ((0.4, 0.1, 0.7),))
        with self.assertRaisesRegex(TypeError, "keys must be strings"):
            state.update_frame(trail_points={1: (0.0, 0.0, 0.0)})  # type: ignore[dict-item]

    def test_polyline_overlays_are_bounded_and_updated_atomically(self) -> None:
        state = BrowserPresentationState(
            (),
            actor_joint_names={"object": ()},
            polyline_limits={"flow": (2, 3)},
        )
        actor = BrowserActorState(
            "object",
            (0.45, 0.0, 0.59),
            (0.0, 0.0, 0.0, 1.0),
        )
        polylines = (
            ((0.4, 0.0, 0.6), (0.5, 0.0, 0.7)),
            ((0.4, 0.1, 0.6),),
        )
        self.assertEqual(
            state.update_frame(
                actor_states=(actor,),
                polyline_overlays={"flow": polylines},
            ),
            1,
        )
        snapshot = state.wait_after(0, timeout=0.0)
        self.assertEqual(snapshot.polyline_overlays["flow"], polylines)
        self.assertEqual(snapshot.polyline_overlay_sequences["flow"], 1)
        self.assertEqual(snapshot.actor_sequence, 1)
        with self.assertRaisesRegex(ValueError, "exceeds 2 lines"):
            state.set_polyline_overlay("flow", (((0.0, 0.0, 0.0),),) * 3)
        with self.assertRaisesRegex(ValueError, "exceeds 3 points"):
            state.set_polyline_overlay(
                "flow",
                (((0.0, 0.0, 0.0),) * 4,),
            )
        with self.assertRaisesRegex(ValueError, "empty"):
            state.set_polyline_overlay("flow", ((),))

    def test_visualization_facade_publishes_atomic_presentation_frame(self) -> None:
        state = BrowserPresentationState(
            (VisualizationTrail("link7", "Link7"),),
            actor_joint_names={"robot": ("joint1",)},
        )
        actor = BrowserActorState(
            "robot",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.25,),
        )
        visualization = object.__new__(Visualization)
        visualization._closed = False
        visualization._presentation_state = state

        self.assertEqual(
            visualization.publish_presentation_frame(
                actor_states=(actor,),
                trail_points={"link7": (0.4, 0.1, 0.7)},
            ),
            1,
        )
        snapshot = state.wait_after(0, timeout=0.0)
        self.assertEqual(snapshot.actor_sequence, snapshot.sequence)
        self.assertEqual(snapshot.trail_sequences["link7"], snapshot.sequence)

    def test_page_contains_browser_only_grade_grid_lights_and_markers(self) -> None:
        page = build_browser_page(
            title="styled",
            background="#111111",
            scene_style=BrowserSceneStyle(),
            trails=(VisualizationTrail("link7", "Link7"),),
        ).decode("utf-8")

        for fragment in (
            '"background_color":"#0B0F15"',
            "function drawPresentationLighting()",
            "function drawColoredGridFloor()",
            "function drawTrajectoryMarkers()",
            "globalCompositeOperation = 'soft-light'",
            "globalCompositeOperation = 'screen'",
            "/presentation?after=",
            '"trail_id":"link7"',
        ):
            self.assertIn(fragment, page)
        self.assertIn("latestFrameImage = probe", page)
        self.assertIn("drawCompositedFrame()", page)

    def test_visualization_facade_publishes_browser_only_trails(self) -> None:
        state = BrowserPresentationState((VisualizationTrail("link7", "Link7"),))
        visualization = object.__new__(Visualization)
        visualization._closed = False
        visualization._presentation_state = state

        self.assertEqual(
            visualization.publish_trail_point("link7", (0.4, 0.1, 0.7)),
            1,
        )
        visualization.set_trail_points(
            "link7",
            ((0.4, 0.1, 0.7), (0.5, 0.1, 0.72)),
        )
        visualization.clear_trail("link7")
        self.assertEqual(state.wait_after(0, timeout=0.0).trails["link7"], ())

        visualization._closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            visualization.publish_trail_point("link7", (0.0, 0.0, 0.0))

    def test_three_scene_rewrites_only_visual_assets_and_builds_native_page(
        self,
    ) -> None:
        project_tmp = Path(__file__).resolve().parents[1] / "tmp" / "unit-tests"
        project_tmp.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=project_tmp) as directory:
            root = Path(directory)
            visual_mesh = root / "visual mesh.STL"
            collision_mesh = root / "collision.stl"
            visual_mesh.write_bytes(b"visual-stl" * 4096)
            collision_mesh.write_bytes(b"collision-stl")
            urdf = root / "robot.urdf"
            urdf.write_text(
                """<robot name="test">
                  <link name="base">
                    <visual><geometry><mesh filename="visual mesh.STL"/></geometry></visual>
                    <collision><geometry><mesh filename="collision.stl"/></geometry></collision>
                  </link>
                  <link name="tip"/>
                  <joint name="joint1" type="revolute">
                    <parent link="base"/><child link="tip"/>
                    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
                  </joint>
                </robot>""",
                encoding="utf-8",
            )
            robot = BrowserUrdfActor.from_file(
                "robot",
                urdf,
                joint_names=("joint1",),
                link_colors={"base": "#12abef"},
            )
            scene = BrowserThreeScene(
                urdf_actors=(robot,),
                primitive_actors=(
                    BrowserPrimitiveActor.box(
                        "object",
                        (0.06, 0.06, 0.06),
                        color="#ff496c",
                        wireframe=True,
                        label="Goal",
                    ),
                ),
                polyline_overlays=(
                    BrowserPolylineOverlay(
                        "flow",
                        "Object flow",
                        "#1fd1ff",
                        max_polylines=16,
                        max_points_per_polyline=8,
                    ),
                ),
            )

            self.assertNotIn("<collision>", robot.urdf_xml)
            self.assertNotIn(str(root), robot.urdf_xml)
            self.assertEqual(len(scene.asset_files), 1)
            self.assertEqual(scene.polyline_limits, {"flow": (16, 8)})
            asset_url = next(iter(scene.asset_files))
            self.assertIn("visual%20mesh.STL", asset_url)
            asset = scene.asset_files[asset_url]
            self.assertIn(f"?v={asset.content_sha256}", robot.urdf_xml)
            page = build_browser_page(
                title="native",
                background="#111111",
                scene_style=BrowserSceneStyle(),
                three_scene=scene,
            ).decode("utf-8")
            for fragment in (
                "three@0.160.1",
                "urdf-loader@0.12.4",
                "new THREE.GridHelper",
                "new THREE.DirectionalLight",
                "new THREE.TubeGeometry",
                "new THREE.LineSegments",
                "THREE.DynamicDrawUsage",
                "function updatePolylineOverlay",
                "polylineOverlayOrder * 2",
                "manager.onProgress",
                "manager.onLoad = finish",
                "geometry.getAttribute('normal') === undefined",
                "robot.visible = handle.hasState && handle.stateVisible",
                "handle.stateVisible = state.visible !== false",
                "handle.root.visible = handle.ready",
                "visualization-three-assets-ready",
                "visualizationThreeRenderer.drawTo",
                "markFrameReady();",
                "    requestCompositedFrame();",
                "preserveDrawingBuffer: false",
                "function previewCameraCommand(command)",
                "function requestCompositedFrame()",
                "snapshot.trail_sequences",
                "snapshot.actor_sequence",
                "if (browserThreeScene === null) {",
                '"link_colors":{"base":"#12ABEF"}',
                '"wireframe":true',
                '"overlay_id":"flow"',
            ):
                self.assertIn(fragment, page)
            self.assertNotIn("    else requestCompositedFrame();", page)
            self.assertNotIn(str(root), page)

            three_draw = page.index("window.visualizationThreeRenderer.drawTo")
            three_ready = page.index("markFrameReady();", three_draw)
            three_return = page.index("return;", three_ready)
            self.assertLess(three_ready, three_return)

            frames = LatestFrame()
            controls = CameraControlBuffer()
            presentation = BrowserPresentationState(
                (),
                actor_joint_names=scene.actor_joint_names,
                polyline_limits=scene.polyline_limits,
            )
            handler = build_request_handler(
                frames,
                controls,
                title="asset-cache",
                background="#111111",
                scene_style=BrowserSceneStyle(),
                presentation_state=presentation,
                three_scene=scene,
            )
            server = VisualizationServer(("127.0.0.1", 0), handler)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            opener = build_opener(ProxyHandler({}))
            url = (
                f"http://127.0.0.1:{server.server_port}"
                f"{asset.versioned_url}"
            )
            try:
                request = Request(url, headers={"Accept-Encoding": "gzip"})
                with opener.open(request, timeout=2.0) as response:
                    compressed = response.read()
                    etag = response.headers["ETag"]
                    self.assertEqual(response.headers["Content-Encoding"], "gzip")
                    self.assertIn("immutable", response.headers["Cache-Control"])
                    self.assertEqual(gzip.decompress(compressed), visual_mesh.read_bytes())
                conditional = Request(
                    url,
                    headers={"Accept-Encoding": "gzip", "If-None-Match": etag},
                )
                with self.assertRaises(HTTPError) as context:
                    opener.open(conditional, timeout=2.0)
                self.assertEqual(context.exception.code, 304)
            finally:
                presentation.close()
                frames.close()
                server.close_websockets()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)


class LatestJpegEncoderTest(unittest.TestCase):
    def test_busy_encoder_discards_superseded_frames(self) -> None:
        first_started = Event()
        release_first = Event()

        class FakeImage:
            def __init__(self, value: bytes) -> None:
                self._value = value

            @classmethod
            def fromarray(cls, value: bytes) -> FakeImage:
                return cls(value)

            def save(self, output: object, **_: object) -> None:
                if self._value == b"first":
                    first_started.set()
                    if not release_first.wait(1.0):
                        raise RuntimeError("test encoder was not released")
                output.write(self._value)  # type: ignore[attr-defined]

        frames = LatestFrame()
        encoder = _LatestJpegEncoder(
            frames,
            image_type=FakeImage,
            quality=90,
        )
        try:
            encoder.submit(b"first")
            self.assertTrue(first_started.wait(1.0))
            encoder.submit(b"second")
            encoder.submit(b"third")
            release_first.set()
            jpeg, sequence, closed = frames.wait_after(0, timeout=1.0)
            self.assertEqual((jpeg, sequence, closed), (b"third", 1, False))
        finally:
            release_first.set()
            encoder.close()
            frames.close()


class NonzeroCudaCameraCompatibilityTest(unittest.TestCase):
    def test_cuda_ordinal_parser_is_explicit_and_conservative(self) -> None:
        self.assertEqual(_cuda_device_ordinal("cuda"), 0)
        self.assertEqual(_cuda_device_ordinal("cuda:0"), 0)
        self.assertEqual(_cuda_device_ordinal("cuda:6"), 6)
        self.assertIsNone(_cuda_device_ordinal("cpu"))
        self.assertIsNone(_cuda_device_ordinal("cuda:invalid"))

    def test_only_nonzero_cuda_camera_view_falls_back_to_usd(self) -> None:
        for device, expected_changed in (
            ("cpu", False),
            ("cuda", False),
            ("cuda:0", False),
            ("cuda:6", True),
        ):
            with self.subTest(device=device):
                view = types.SimpleNamespace(
                    _use_fabric=True,
                    _fabric_initialized=False,
                )
                camera = types.SimpleNamespace(device=device, _view=view)
                self.assertIs(
                    _disable_unsupported_camera_fabric_view(camera),
                    expected_changed,
                )
                self.assertIs(view._use_fabric, not expected_changed)

    def test_wrapper_applies_fallback_immediately_after_camera_init(self) -> None:
        class FakeCamera:
            def __init__(self, device: str) -> None:
                self.device = device
                self._view = None

            def _initialize_impl(self) -> None:
                self._view = types.SimpleNamespace(
                    _use_fabric=True,
                    _fabric_initialized=False,
                )

        camera_type = _camera_type_with_nonzero_cuda_fallback(FakeCamera)
        camera = camera_type("cuda:6")
        camera._initialize_impl()
        self.assertFalse(camera._view._use_fabric)

    def test_late_fallback_fails_instead_of_mutating_live_fabric(self) -> None:
        camera = types.SimpleNamespace(
            device="cuda:6",
            _view=types.SimpleNamespace(
                _use_fabric=True,
                _fabric_initialized=True,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "before"):
            _disable_unsupported_camera_fabric_view(camera)


class BrowserOnlyVisualizationTest(unittest.TestCase):
    def test_initializes_after_sim_start_without_importing_isaac_camera(self) -> None:
        scene = types.SimpleNamespace(num_envs=1)
        visualization = Visualization(
            scene,
            num_envs=1,
            port=0,
            resolution=(32, 24),
            scene_style=BrowserSceneStyle(),
            camera_enabled=False,
        )
        try:
            visualization.initialize()
            self.assertFalse(visualization.camera_enabled)
            self.assertFalse(visualization.apply_camera_controls())
            self.assertFalse(visualization.update(1.0 / 60.0, force_publish=True))
        finally:
            visualization.close()

    def test_frame_sink_receives_owned_normalized_rgb_frame(self) -> None:
        scene = types.SimpleNamespace(num_envs=1)
        frames: list[object] = []
        visualization = Visualization(
            scene,
            num_envs=1,
            port=0,
            resolution=(32, 24),
            camera_enabled=False,
            frame_sink=frames.append,
        )
        try:
            visualization.initialize()
            self.assertEqual(len(frames), 1)
            frame = frames[0]
            self.assertEqual(frame.shape, (24, 32, 3))
            self.assertEqual(str(frame.dtype), "uint8")
            self.assertTrue(frame.flags.c_contiguous)
            frame[0, 0, 0] = 255
            visualization.publish(frame * 0)
            self.assertEqual(len(frames), 2)
            self.assertEqual(int(frames[1][0, 0, 0]), 0)
        finally:
            visualization.close()


class VisualizationHttpControlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = LatestFrame()
        self.controls = CameraControlBuffer()
        self.callback_events: list[tuple[str, int]] = []
        self.buttons = (
            VisualizationButton(
                "reset_object",
                "重置物体",
                lambda: self.callback_events.append(("reset", get_ident())),
                style="primary",
            ),
            VisualizationButton(
                "release_object",
                "释放物体",
                lambda: self.callback_events.append(("release", get_ident())),
                style="danger",
            ),
        )
        self.button_events = ButtonEventBuffer(
            (button.button_id for button in self.buttons),
            max_pending=2,
        )
        self.curves = (
            VisualizationCurve(
                "height",
                "机械臂高度",
                color="#66BB6A",
                unit="m",
            ),
            VisualizationCurve(
                "left_right",
                "机械臂左右位置",
                color="#42A5F5",
                unit="m",
            ),
        )
        self.dashboard_events = DashboardEventStream(
            (curve.metric_id for curve in self.curves),
            log_enabled=True,
            max_events=4,
        )
        self.scene_style = BrowserSceneStyle()
        self.trails = (
            VisualizationTrail("link7", "Link7", max_points=4),
            VisualizationTrail("object", "Object", max_points=4),
        )
        self.presentation_state = BrowserPresentationState(self.trails)
        self.presentation_state.update_view(
            (0.6, -1.8, 1.3),
            (0.0, 0.4, 0.5),
            vertical_fov=0.7,
            aspect=4 / 3,
        )
        handler = build_request_handler(
            self.frames,
            self.controls,
            title="control smoke",
            background="#111111",
            buttons=self.buttons,
            button_events=self.button_events,
            curves=self.curves,
            dashboard_events=self.dashboard_events,
            show_output=True,
            scene_style=self.scene_style,
            trails=self.trails,
            presentation_state=self.presentation_state,
        )
        self.server = VisualizationServer(("127.0.0.1", 0), handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.opener = build_opener(ProxyHandler({}))

    def tearDown(self) -> None:
        self.button_events.close()
        self.dashboard_events.close()
        self.presentation_state.close()
        self.frames.close()
        self.server.close_websockets()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)

    def test_page_exposes_controls_and_post_queues_command(self) -> None:
        with self.opener.open(f"{self.base_url}/", timeout=2.0) as response:
            page = response.read().decode("utf-8")
        self.assertIn("<title>control smoke</title>", page)
        self.assertNotIn('class="hud title"', page)
        for help_line in (
            "<span>左键拖动：旋转</span>",
            "<span>右键拖动：平移</span>",
            "<span>滚轮：缩放</span>",
            "<span>R：重置</span>",
        ):
            self.assertIn(help_line, page)
        self.assertNotIn("中键拖动：旋转", page)
        self.assertIn("font-size:7px", page)
        self.assertIn("flex-direction:column", page)
        self.assertIn("height:20px", page)
        self.assertIn("font-size:8px", page)
        self.assertIn("event.button === 0 && event.altKey", page)
        self.assertIn('"0": "orbit", "1": "orbit", "2": "pan"', page)
        self.assertNotIn("event.button === 0) dragMode = 'orbit'", page)
        self.assertIn("requiredButton = [1, 4, 2][event.button]", page)
        self.assertIn("visualization-primary-pointer", page)
        self.assertIn("window.visualizationInteractions", page)
        self.assertIn("new WebSocket", page)
        self.assertIn("client_sequence", page)
        self.assertIn("previewCameraCommand", page)
        self.assertIn("/frame.jpg?after=", page)
        self.assertIn("fetch('/camera'", page)
        self.assertIn('id="environment-actions"', page)
        self.assertIn('data-button-id="reset_object"', page)
        self.assertIn('data-style="primary"', page)
        self.assertIn('data-button-id="release_object"', page)
        for panel_id in ("buttons", "curves", "output"):
            self.assertIn(f'data-panel-id="{panel_id}"', page)
        self.assertIn("top:8px;right:8px", page)
        self.assertIn("flex-direction:column", page)
        self.assertIn("align-items:stretch", page)
        self.assertIn("environmentActionQueue = environmentActionQueue", page)
        self.assertIn("fetch('/button'", page)
        self.assertIn("environmentActionFromEvent(event)", page)
        self.assertIn("/dashboard?after=", page)
        self.assertIn("localStorage.setItem(dashboardLayoutKey", page)

        for required_recording_fragment in (
            '<canvas id="frame"',
            'id="record-start"',
            'id="record-pause"',
            'id="record-stop"',
            "frame.captureStream()",
            "new MediaRecorder",
            "addEventListener('dataavailable'",
            "addEventListener('pause'",
            "addEventListener('resume'",
            "addEventListener('stop'",
            "mediaRecorder.pause()",
            "mediaRecorder.resume()",
            "mediaRecorder.stop()",
            "anchor.download =",
            "saveRecording(chunks, outputType)",
            "'video/mp4;codecs=avc1.42E01E'",
            "effectiveType.toLowerCase().includes('mp4') ? 'mp4' : 'webm'",
        ):
            self.assertIn(required_recording_fragment, page)

        self.assertLess(
            page.index("'video/mp4;codecs=avc1.42E01E'"),
            page.index("'video/webm;codecs=vp9'"),
        )

        body = json.dumps({"orbit_x": 0.12, "zoom": -5.0}).encode("utf-8")
        request = Request(
            f"{self.base_url}/camera",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.opener.open(request, timeout=2.0) as response:
            self.assertEqual(response.status, 204)
        self.assertEqual(
            self.controls.consume(),
            CameraCommand(orbit_x=0.12, zoom=-5.0),
        )

    def test_button_post_only_runs_callback_when_main_thread_dispatches(self) -> None:
        for button_id in ("reset_object", "release_object"):
            body = json.dumps({"button_id": button_id}).encode("utf-8")
            request = Request(
                f"{self.base_url}/button",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.opener.open(request, timeout=2.0) as response:
                self.assertEqual(response.status, 204)

        self.assertEqual(self.callback_events, [])
        self.assertEqual(self.button_events.pending_count, 2)

        visualization = object.__new__(Visualization)
        visualization._simulation_thread_id = get_ident()
        visualization._initialized = True
        visualization._closed = False
        visualization._button_events = self.button_events
        visualization._button_callbacks = {
            button.button_id: button.callback for button in self.buttons
        }
        self.assertEqual(visualization.dispatch_button_callbacks(), 2)
        self.assertEqual(
            self.callback_events,
            [("reset", get_ident()), ("release", get_ident())],
        )
        self.assertEqual(visualization.dispatch_button_callbacks(), 0)

    def test_button_dispatch_rejects_background_thread(self) -> None:
        self.button_events.submit("reset_object")
        visualization = object.__new__(Visualization)
        visualization._simulation_thread_id = get_ident()
        visualization._initialized = True
        visualization._closed = False
        visualization._button_events = self.button_events
        visualization._button_callbacks = {
            button.button_id: button.callback for button in self.buttons
        }
        errors: list[Exception] = []

        def dispatch_in_background() -> None:
            try:
                visualization.dispatch_button_callbacks()
            except Exception as error:
                errors.append(error)

        thread = Thread(target=dispatch_in_background)
        thread.start()
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIn("simulation thread", str(errors[0]))
        self.assertEqual(self.button_events.pending_count, 1)
        self.assertEqual(self.callback_events, [])
        self.assertEqual(visualization.dispatch_button_callbacks(), 1)

    def test_invalid_button_command_is_rejected(self) -> None:
        for payload in (
            {"button_id": "unknown"},
            {"button_id": "reset_object", "extra": True},
            {},
        ):
            with self.subTest(payload=payload):
                body = json.dumps(payload).encode("utf-8")
                request = Request(
                    f"{self.base_url}/button",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(HTTPError) as context:
                    self.opener.open(request, timeout=2.0)
                self.assertEqual(context.exception.code, 400)

    def test_full_button_queue_is_rejected_without_running_callbacks(self) -> None:
        body = json.dumps({"button_id": "reset_object"}).encode("utf-8")
        request = Request(
            f"{self.base_url}/button",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        for _ in range(2):
            with self.opener.open(request, timeout=2.0) as response:
                self.assertEqual(response.status, 204)
        with self.assertRaises(HTTPError) as context:
            self.opener.open(request, timeout=2.0)
        self.assertEqual(context.exception.code, 429)
        self.assertEqual(self.button_events.pending_count, 2)
        self.assertEqual(self.callback_events, [])

    def test_styled_page_keeps_raw_latest_frame_endpoints_byte_exact(self) -> None:
        self.frames.publish(b"frame-one")
        self.frames.publish(b"frame-two")
        self.frames.publish(b"frame-three")
        with self.opener.open(
            f"{self.base_url}/frame.jpg?after=0",
            timeout=2.0,
        ) as response:
            self.assertEqual(response.read(), b"frame-three")
            self.assertEqual(response.headers["X-Frame-Sequence"], "3")
        with self.opener.open(
            f"{self.base_url}/snapshot.jpg",
            timeout=2.0,
        ) as response:
            self.assertEqual(response.read(), b"frame-three")

        completed = Event()
        result: dict[str, object] = {}

        def receive_next() -> None:
            with self.opener.open(
                f"{self.base_url}/frame.jpg?after=3",
                timeout=2.0,
            ) as response:
                result["body"] = response.read()
                result["sequence"] = response.headers["X-Frame-Sequence"]
            completed.set()

        receiver = Thread(target=receive_next)
        receiver.start()
        self.assertFalse(completed.wait(0.05))
        self.frames.publish(b"frame-four")
        self.assertTrue(completed.wait(1.0))
        receiver.join(timeout=1.0)
        self.assertEqual(result, {"body": b"frame-four", "sequence": "4"})

    def test_dashboard_endpoint_returns_metric_and_log_batches(self) -> None:
        self.dashboard_events.publish_metrics(
            {"height": 0.52, "left_right": -0.08},
            time_s=1.0,
        )
        self.dashboard_events.publish_log(
            "按钮按下：机械臂上移",
            source="button",
            time_s=1.0,
        )
        with self.opener.open(
            f"{self.base_url}/dashboard?after=0",
            timeout=2.0,
        ) as response:
            self.assertEqual(response.headers["X-Dashboard-Sequence"], "2")
            batch = json.loads(response.read())

        self.assertEqual(batch["sequence"], 2)
        self.assertEqual(batch["dropped"], 0)
        self.assertEqual(
            batch["events"][0],
            {
                "type": "metrics",
                "time_s": 1.0,
                "values": {"height": 0.52, "left_right": -0.08},
                "sequence": 1,
            },
        )
        self.assertEqual(batch["events"][1]["message"], "按钮按下：机械臂上移")

        with self.assertRaises(HTTPError) as context:
            self.opener.open(
                f"{self.base_url}/dashboard?after=3",
                timeout=2.0,
            )
        self.assertEqual(context.exception.code, 400)

    def test_presentation_endpoint_returns_full_latest_snapshot(self) -> None:
        self.presentation_state.append_trail_point("link7", (0.42, 0.08, 0.71))
        self.presentation_state.append_trail_point("object", (0.45, 0.0, 0.59))
        with self.opener.open(
            f"{self.base_url}/presentation?after=0",
            timeout=2.0,
        ) as response:
            snapshot = json.loads(response.read())
            self.assertEqual(
                response.headers["X-Presentation-Sequence"],
                str(snapshot["sequence"]),
            )

        self.assertEqual(snapshot["view"]["aspect"], 4 / 3)
        self.assertEqual(snapshot["view"]["client_sequence"], 0)
        self.assertEqual(snapshot["trails"]["link7"], [[0.42, 0.08, 0.71]])
        self.assertEqual(snapshot["trails"]["object"], [[0.45, 0.0, 0.59]])
        self.assertGreater(snapshot["trail_sequences"]["link7"], 0)
        self.assertGreater(snapshot["trail_sequences"]["object"], 0)
        self.assertEqual(snapshot["actor_sequence"], 0)

        with self.assertRaises(HTTPError) as context:
            self.opener.open(
                f"{self.base_url}/presentation?after={snapshot['sequence'] + 1}",
                timeout=2.0,
            )
        self.assertEqual(context.exception.code, 400)

    def test_websocket_control_stream_needs_no_per_command_response(self) -> None:
        client = socket.create_connection(
            ("127.0.0.1", self.server.server_port),
            timeout=2.0,
        )
        client.settimeout(2.0)
        try:
            request = (
                (b"\x09" + b"\x00" * 12) * 2 + b"GET /control HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Upgrade: websocket\r\n"
                b"Connection: Upgrade\r\n"
                b"Sec-WebSocket-Version: 13\r\n"
                b"Sec-WebSocket-Key: MDEyMzQ1Njc4OWFiY2RlZg==\r\n\r\n"
            )
            client.sendall(request)
            response = bytearray()
            while b"\r\n\r\n" not in response:
                response.extend(client.recv(4096))
            self.assertIn(b"101 Switching Protocols", response)

            payload = json.dumps({"orbit_x": 0.01}).encode("utf-8")
            for _ in range(25):
                _send_client_websocket_frame(client, 0x1, payload)
            _send_client_websocket_frame(client, 0x9, b"processed")
            opcode, pong = _recv_server_websocket_frame(client)
            self.assertEqual((opcode, pong), (0xA, b"processed"))
            command = self.controls.consume()
            self.assertAlmostEqual(command.orbit_x, 0.25)
            self.assertEqual(command, CameraCommand(orbit_x=command.orbit_x))
            _send_client_websocket_frame(client, 0x8, b"")
        finally:
            client.close()

    def test_forwarder_prefixed_post_is_accepted(self) -> None:
        body = b'{"pan_x":0.07}'
        request = (
            (b"\x09" + b"\x00" * 12) * 2
            + b"POST /camera HTTP/1.1\r\n"
            + b"Host: localhost\r\n"
            + f"Content-Length: {len(body)}\r\n".encode("ascii")
            + b"Content-Type: application/json\r\n"
            + b"Connection: close\r\n\r\n"
            + body
        )
        with socket.create_connection(
            ("127.0.0.1", self.server.server_port),
            timeout=2.0,
        ) as client:
            client.sendall(request)
            response = bytearray()
            while chunk := client.recv(4096):
                response.extend(chunk)

        self.assertIn(b"HTTP/1.1 204 No Content", response)
        self.assertEqual(
            self.controls.consume(),
            CameraCommand(pan_x=0.07),
        )


if __name__ == "__main__":
    unittest.main()
