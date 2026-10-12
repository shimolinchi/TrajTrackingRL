"""Standard-library HTTP server for browser visualization and camera input."""

from __future__ import annotations

import base64
from collections.abc import Iterable
import gzip
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import struct
import sys
from threading import Condition, Lock
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .buttons import ButtonEventBuffer, VisualizationButton, normalize_buttons
from .browser import build_browser_page
from .camera_control import CameraControlBuffer
from .dashboard import (
    DashboardEventStream,
    VisualizationCurve,
    normalize_curves,
)
from .presentation import (
    BrowserPresentationState,
    BrowserSceneStyle,
    VisualizationTrail,
    normalize_trails,
)
from .three_scene import BrowserThreeScene


_WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_MAX_CONTROL_PAYLOAD = 4096


class WebSocketProtocolError(Exception):
    """Raised when a browser sends an unsupported WebSocket frame."""


class LatestFrame:
    """Thread-safe storage for exactly one encoded frame."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._jpeg: bytes | None = None
        self._sequence = 0
        self._closed = False

    def publish(self, jpeg: bytes) -> None:
        with self._condition:
            if self._closed:
                return
            self._jpeg = jpeg
            self._sequence += 1
            self._condition.notify_all()

    def latest(self) -> bytes | None:
        with self._condition:
            return self._jpeg

    @property
    def sequence(self) -> int:
        with self._condition:
            return self._sequence

    def wait_after(
        self,
        sequence: int,
        *,
        timeout: float = 1.0,
    ) -> tuple[bytes | None, int, bool]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed or self._sequence > sequence,
                timeout=timeout,
            )
            return self._jpeg, self._sequence, self._closed

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class VisualizationServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._websocket_lock = Lock()
        self._websocket_connections: set[socket.socket] = set()
        super().__init__(*args, **kwargs)

    def register_websocket(self, connection: socket.socket) -> None:
        with self._websocket_lock:
            self._websocket_connections.add(connection)

    def unregister_websocket(self, connection: socket.socket) -> None:
        with self._websocket_lock:
            self._websocket_connections.discard(connection)

    def close_websockets(self) -> None:
        with self._websocket_lock:
            connections = tuple(self._websocket_connections)
            self._websocket_connections.clear()
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass

    def handle_error(
        self,
        request: Any,
        client_address: tuple[str, int],
    ) -> None:
        """Silence expected disconnect races when a polling browser closes."""

        error = sys.exc_info()[1]
        if isinstance(error, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def build_request_handler(
    frames: LatestFrame,
    controls: CameraControlBuffer,
    *,
    title: str,
    background: str,
    buttons: Iterable[VisualizationButton] = (),
    button_events: ButtonEventBuffer | None = None,
    curves: Iterable[VisualizationCurve] = (),
    dashboard_events: DashboardEventStream | None = None,
    show_output: bool = False,
    scene_style: BrowserSceneStyle | None = None,
    trails: Iterable[VisualizationTrail] = (),
    presentation_state: BrowserPresentationState | None = None,
    three_scene: BrowserThreeScene | None = None,
    resolution: tuple[int, int] = (1280, 960),
) -> type[BaseHTTPRequestHandler]:
    frozen_buttons = normalize_buttons(buttons)
    frozen_curves = normalize_curves(curves)
    frozen_trails = normalize_trails(trails)
    if not isinstance(show_output, bool):
        raise TypeError("show_output must be a bool")
    if scene_style is not None and not isinstance(scene_style, BrowserSceneStyle):
        raise TypeError("scene_style must be a BrowserSceneStyle or None")
    if three_scene is not None and not isinstance(three_scene, BrowserThreeScene):
        raise TypeError("three_scene must be a BrowserThreeScene or None")
    if (
        not isinstance(resolution, tuple)
        or len(resolution) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in resolution
        )
    ):
        raise ValueError("resolution must be a pair of positive integers")
    expected_button_ids = frozenset(button.button_id for button in frozen_buttons)
    if button_events is None:
        event_buffer = ButtonEventBuffer(expected_button_ids)
    elif button_events.button_ids != expected_button_ids:
        raise ValueError(
            "button event buffer identifiers do not match the rendered buttons"
        )
    else:
        event_buffer = button_events
    expected_metric_ids = frozenset(curve.metric_id for curve in frozen_curves)
    dashboard_enabled = bool(frozen_curves) or show_output
    if dashboard_enabled and dashboard_events is None:
        raise ValueError(
            "dashboard_events is required when curves or output are enabled"
        )
    if dashboard_events is not None and (
        dashboard_events.metric_ids != expected_metric_ids
        or dashboard_events.log_enabled != show_output
    ):
        raise ValueError("dashboard event stream does not match rendered curves/output")
    presentation_enabled = (
        scene_style is not None or bool(frozen_trails) or three_scene is not None
    )
    expected_trail_ids = frozenset(trail.trail_id for trail in frozen_trails)
    expected_actor_ids = (
        frozenset() if three_scene is None else frozenset(three_scene.actor_joint_names)
    )
    expected_polyline_overlay_ids = (
        frozenset()
        if three_scene is None
        else frozenset(three_scene.polyline_limits)
    )
    if presentation_enabled and presentation_state is None:
        raise ValueError(
            "presentation_state is required when scene styling or trails are enabled"
        )
    if presentation_state is not None and (
        presentation_state.trail_ids != expected_trail_ids
        or presentation_state.actor_ids != expected_actor_ids
        or presentation_state.polyline_overlay_ids
        != expected_polyline_overlay_ids
    ):
        raise ValueError(
            "presentation state identifiers do not match the rendered "
            "trails/actors/polyline overlays"
        )
    page = build_browser_page(
        title=title,
        background=background,
        buttons=frozen_buttons,
        curves=frozen_curves,
        show_output=show_output,
        scene_style=scene_style,
        trails=frozen_trails,
        three_scene=three_scene,
        resolution=resolution,
    )
    asset_files = {} if three_scene is None else dict(three_scene.asset_files)
    asset_payloads = {
        url_path: (
            asset,
            raw_body,
            compressed
            if len(compressed) + 128 < len(raw_body)
            else None,
        )
        for url_path, asset in asset_files.items()
        for raw_body in (asset.local_path.read_bytes(),)
        for compressed in (gzip.compress(raw_body, compresslevel=5, mtime=0),)
    }

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def parse_request(self) -> bool:
            """Accept control-byte framing inserted by some port forwarders."""

            offsets = (
                self.raw_requestline.find(method) for method in (b"GET ", b"POST ")
            )
            method_offset = min(
                (value for value in offsets if value >= 0),
                default=-1,
            )
            if method_offset > 0:
                prefix = self.raw_requestline[:method_offset]
                if all(byte in (0x00, 0x09) for byte in prefix):
                    self.raw_requestline = self.raw_requestline[method_offset:]
            return super().parse_request()

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlsplit(self.path)
            path = parsed.path
            if path == "/":
                self._send_bytes(page, "text/html; charset=utf-8")
            elif path in asset_payloads:
                self._send_asset(path)
            elif path == "/snapshot.jpg":
                frame = frames.latest()
                if frame is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                else:
                    self._send_bytes(frame, "image/jpeg")
            elif path == "/frame.jpg":
                self._send_latest_frame(parsed.query)
            elif path == "/dashboard" and dashboard_enabled:
                self._send_dashboard_events(parsed.query)
            elif path == "/presentation" and presentation_enabled:
                self._send_presentation_state(parsed.query)
            elif path == "/stream.mjpg":
                self._stream_frames()
            elif path == "/control":
                self._serve_control_websocket()
            elif path == "/healthz":
                self._send_bytes(b"ok\n", "text/plain; charset=utf-8")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            path = self.path.partition("?")[0]
            if path not in ("/camera", "/button") or (
                path == "/button" and not frozen_buttons
            ):
                self.close_connection = True
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if self.headers.get_content_type() != "application/json":
                self.close_connection = True
                self.send_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
                return

            length_value = self.headers.get("Content-Length")
            if length_value is None:
                self.close_connection = True
                self.send_error(HTTPStatus.LENGTH_REQUIRED)
                return
            try:
                length = int(length_value)
            except ValueError:
                self.close_connection = True
                self.send_error(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
                return
            if not 0 < length <= _MAX_CONTROL_PAYLOAD:
                self.close_connection = True
                self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return

            try:
                payload = json.loads(self.rfile.read(length))
                if path == "/camera":
                    controls.submit(payload)
                else:
                    if not isinstance(payload, dict) or set(payload) != {"button_id"}:
                        raise ValueError(
                            "button command must contain exactly one button_id"
                        )
                    event_buffer.submit(payload["button_id"])
            except (UnicodeDecodeError, ValueError) as error:
                self.send_error(HTTPStatus.BAD_REQUEST, str(error))
                return
            except OverflowError as error:
                self.send_error(HTTPStatus.TOO_MANY_REQUESTS, str(error))
                return
            except RuntimeError as error:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, str(error))
                return

            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self._send_no_cache_headers()
            self.end_headers()

        def _send_no_cache_headers(self) -> None:
            self.send_header(
                "Cache-Control",
                "no-store, no-cache, must-revalidate, max-age=0",
            )
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")

        def _send_bytes(
            self,
            body: bytes,
            content_type: str,
            *,
            headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self._send_no_cache_headers()
            if headers is not None:
                for name, value in headers.items():
                    self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _send_asset(self, path: str) -> None:
            asset, raw_body, compressed_body = asset_payloads[path]
            accepted_encodings: set[str] = set()
            for raw_token in self.headers.get("Accept-Encoding", "").split(","):
                name, _, raw_parameters = raw_token.partition(";")
                quality = 1.0
                for parameter in raw_parameters.split(";"):
                    key, separator, value = parameter.strip().partition("=")
                    if separator and key.lower() == "q":
                        try:
                            quality = float(value)
                        except ValueError:
                            quality = 0.0
                if quality > 0.0:
                    accepted_encodings.add(name.strip().lower())
            use_gzip = compressed_body is not None and (
                "gzip" in accepted_encodings or "*" in accepted_encodings
            )
            body = compressed_body if use_gzip else raw_body
            representation = "gzip" if use_gzip else "identity"
            etag = f'"sha256-{asset.content_sha256}-{representation}"'
            if self.headers.get("If-None-Match") == etag:
                self.send_response(HTTPStatus.NOT_MODIFIED)
                self.send_header("Content-Length", "0")
                self._send_asset_headers(etag)
                self.end_headers()
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", asset.content_type)
            self.send_header("Content-Length", str(len(body)))
            if use_gzip:
                self.send_header("Content-Encoding", "gzip")
            self._send_asset_headers(etag)
            self.end_headers()
            self.wfile.write(body)

        def _send_asset_headers(self, etag: str) -> None:
            self.send_header("ETag", etag)
            self.send_header("Vary", "Accept-Encoding")
            self.send_header(
                "Cache-Control",
                "public, max-age=31536000, immutable",
            )
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")

        def _send_latest_frame(self, query: str) -> None:
            parameters = parse_qs(query, keep_blank_values=True)
            if set(parameters) != {"after"} or len(parameters["after"]) != 1:
                self.send_error(
                    HTTPStatus.BAD_REQUEST,
                    "expected exactly one non-negative 'after' parameter",
                )
                return
            try:
                after = int(parameters["after"][0])
            except ValueError:
                after = -1
            if after < 0:
                self.send_error(
                    HTTPStatus.BAD_REQUEST,
                    "'after' must be a non-negative integer",
                )
                return

            frame, sequence, closed = frames.wait_after(after, timeout=1.0)
            if closed:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
            elif frame is None or sequence <= after:
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Content-Length", "0")
                self.send_header("X-Frame-Sequence", str(sequence))
                self._send_no_cache_headers()
                self.end_headers()
            else:
                self._send_bytes(
                    frame,
                    "image/jpeg",
                    headers={"X-Frame-Sequence": str(sequence)},
                )

        def _send_dashboard_events(self, query: str) -> None:
            parameters = parse_qs(query, keep_blank_values=True)
            if set(parameters) != {"after"} or len(parameters["after"]) != 1:
                self.send_error(
                    HTTPStatus.BAD_REQUEST,
                    "expected exactly one non-negative 'after' parameter",
                )
                return
            try:
                after = int(parameters["after"][0])
            except ValueError:
                after = -1
            if after < 0:
                self.send_error(
                    HTTPStatus.BAD_REQUEST,
                    "'after' must be a non-negative integer",
                )
                return
            if dashboard_events is None:
                raise RuntimeError("dashboard endpoint has no event stream")
            try:
                batch = dashboard_events.wait_after(after, timeout=1.0)
            except ValueError as error:
                self.send_error(HTTPStatus.BAD_REQUEST, str(error))
                return
            if batch.closed and not batch.events:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
            elif not batch.events:
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Content-Length", "0")
                self.send_header("X-Dashboard-Sequence", str(batch.sequence))
                self._send_no_cache_headers()
                self.end_headers()
            else:
                body = json.dumps(
                    {
                        "sequence": batch.sequence,
                        "dropped": batch.dropped,
                        "events": batch.events,
                        "closed": batch.closed,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                self._send_bytes(
                    body,
                    "application/json; charset=utf-8",
                    headers={"X-Dashboard-Sequence": str(batch.sequence)},
                )

        def _send_presentation_state(self, query: str) -> None:
            parameters = parse_qs(query, keep_blank_values=True)
            if set(parameters) != {"after"} or len(parameters["after"]) != 1:
                self.send_error(
                    HTTPStatus.BAD_REQUEST,
                    "expected exactly one non-negative 'after' parameter",
                )
                return
            try:
                after = int(parameters["after"][0])
            except ValueError:
                after = -1
            if after < 0:
                self.send_error(
                    HTTPStatus.BAD_REQUEST,
                    "'after' must be a non-negative integer",
                )
                return
            if presentation_state is None:
                raise RuntimeError("presentation endpoint has no state")
            try:
                snapshot = presentation_state.wait_after(after, timeout=1.0)
            except (TypeError, ValueError) as error:
                self.send_error(HTTPStatus.BAD_REQUEST, str(error))
                return
            if snapshot.closed and snapshot.sequence <= after:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
            elif snapshot.sequence <= after:
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Content-Length", "0")
                self.send_header(
                    "X-Presentation-Sequence",
                    str(snapshot.sequence),
                )
                self._send_no_cache_headers()
                self.end_headers()
            else:
                body = json.dumps(
                    {
                        "sequence": snapshot.sequence,
                        "recording_paused": snapshot.recording_paused,
                        "view": snapshot.view,
                        "trails": snapshot.trails,
                        "trail_sequences": snapshot.trail_sequences,
                        "polyline_overlays": snapshot.polyline_overlays,
                        "polyline_overlay_sequences": (
                            snapshot.polyline_overlay_sequences
                        ),
                        "actors": {
                            actor_id: state.as_payload()
                            for actor_id, state in snapshot.actors.items()
                        },
                        "actor_sequence": snapshot.actor_sequence,
                        "closed": snapshot.closed,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                self._send_bytes(
                    body,
                    "application/json; charset=utf-8",
                    headers={"X-Presentation-Sequence": str(snapshot.sequence)},
                )

        def _serve_control_websocket(self) -> None:
            if self.headers.get("Upgrade", "").lower() != "websocket":
                self.send_error(HTTPStatus.UPGRADE_REQUIRED)
                return
            connection_tokens = {
                token.strip().lower()
                for token in self.headers.get("Connection", "").split(",")
            }
            key = self.headers.get("Sec-WebSocket-Key", "")
            if "upgrade" not in connection_tokens or not _valid_websocket_key(key):
                self.send_error(HTTPStatus.BAD_REQUEST, "invalid WebSocket handshake")
                return
            if self.headers.get("Sec-WebSocket-Version") != "13":
                self.send_response(HTTPStatus.UPGRADE_REQUIRED)
                self.send_header("Sec-WebSocket-Version", "13")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            accept = base64.b64encode(
                hashlib.sha1((key + _WEBSOCKET_GUID).encode("ascii")).digest()
            ).decode("ascii")
            self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept)
            self.end_headers()
            self.wfile.flush()
            self.close_connection = True

            server = self.server
            if not isinstance(server, VisualizationServer):
                raise RuntimeError("unexpected HTTP server type")
            server.register_websocket(self.connection)
            try:
                self._receive_websocket_controls()
            finally:
                server.unregister_websocket(self.connection)

        def _receive_websocket_controls(self) -> None:
            try:
                while True:
                    frame = _read_websocket_frame(self.rfile)
                    if frame is None:
                        return
                    opcode, payload = frame
                    if opcode == 0x8:
                        _write_websocket_frame(self.wfile, 0x8, payload[:125])
                        return
                    if opcode == 0x9:
                        _write_websocket_frame(self.wfile, 0xA, payload)
                        continue
                    if opcode == 0xA:
                        continue
                    if opcode != 0x1:
                        raise WebSocketProtocolError("only text commands are supported")
                    try:
                        controls.submit(json.loads(payload.decode("utf-8")))
                    except (UnicodeDecodeError, ValueError) as error:
                        raise WebSocketProtocolError(str(error)) from error
            except WebSocketProtocolError as error:
                reason = str(error).encode("utf-8")[:123]
                try:
                    _write_websocket_frame(
                        self.wfile,
                        0x8,
                        struct.pack("!H", 1008) + reason,
                    )
                except OSError:
                    pass
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def _stream_frames(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame",
            )
            self._send_no_cache_headers()
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            sequence = 0
            try:
                while True:
                    frame, current, closed = frames.wait_after(sequence)
                    if closed:
                        return
                    if frame is None or current == sequence:
                        continue
                    sequence = current
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                    )
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def _valid_websocket_key(key: str) -> bool:
    try:
        decoded = base64.b64decode(key, validate=True)
    except (ValueError, UnicodeEncodeError):
        return False
    return len(decoded) == 16


def _read_exact(stream: Any, length: int) -> bytes:
    data = stream.read(length)
    if len(data) != length:
        raise WebSocketProtocolError("truncated WebSocket frame")
    return data


def _read_websocket_frame(stream: Any) -> tuple[int, bytes] | None:
    header = stream.read(2)
    if not header:
        return None
    if len(header) != 2:
        raise WebSocketProtocolError("truncated WebSocket header")

    first, second = header
    if first & 0x70:
        raise WebSocketProtocolError("WebSocket extensions are not supported")
    if not first & 0x80:
        raise WebSocketProtocolError("fragmented WebSocket frames are not supported")
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    if not masked:
        raise WebSocketProtocolError("browser frames must be masked")

    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", _read_exact(stream, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _read_exact(stream, 8))[0]
    if length > _MAX_CONTROL_PAYLOAD:
        raise WebSocketProtocolError("camera command is too large")
    if opcode >= 0x8 and length > 125:
        raise WebSocketProtocolError("invalid WebSocket control frame")

    mask = _read_exact(stream, 4)
    payload = bytearray(_read_exact(stream, length))
    for index in range(length):
        payload[index] ^= mask[index % 4]
    return opcode, bytes(payload)


def _write_websocket_frame(stream: Any, opcode: int, payload: bytes) -> None:
    length = len(payload)
    if length <= 125:
        header = bytes((0x80 | opcode, length))
    elif length <= 0xFFFF:
        header = bytes((0x80 | opcode, 126)) + struct.pack("!H", length)
    else:
        header = bytes((0x80 | opcode, 127)) + struct.pack("!Q", length)
    stream.write(header + payload)
    stream.flush()


__all__ = ["LatestFrame", "VisualizationServer", "build_request_handler"]
