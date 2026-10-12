"""Minimal Wuji hand FK used to materialize reference keypoint trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from torch import Tensor


WUJI_ACTUATED_JOINTS_PER_FINGER = 4
WUJI_KEYPOINTS_PER_FINGER = 5
WUJI_FINGER_COUNT = 5
WUJI_KEYPOINT_COUNT = WUJI_KEYPOINTS_PER_FINGER * WUJI_FINGER_COUNT
WUJI_TIP_KEYPOINT_INDICES = (4, 9, 14, 19, 24)
WUJI_LEVEL_1_KEYPOINT_INDICES = (0, 1, 5, 6, 10, 11, 15, 16, 20, 21)
WUJI_LEVEL_2_KEYPOINT_INDICES = (2, 3, 7, 8, 12, 13, 17, 18, 22, 23)


def _vector(
    text: str | None, *, default: tuple[float, float, float]
) -> tuple[float, ...]:
    if text is None:
        return default
    values = tuple(float(item) for item in text.split())
    if len(values) != 3:
        raise ValueError(f"URDF vector must have three entries: {text!r}")
    return values


def _rpy_matrix(rpy: tuple[float, ...]) -> Tensor:
    roll, pitch, yaw = rpy
    rx = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, cos(roll), -sin(roll)), (0.0, sin(roll), cos(roll))),
        dtype=torch.float64,
    )
    ry = torch.tensor(
        (
            (cos(pitch), 0.0, sin(pitch)),
            (0.0, 1.0, 0.0),
            (-sin(pitch), 0.0, cos(pitch)),
        ),
        dtype=torch.float64,
    )
    rz = torch.tensor(
        ((cos(yaw), -sin(yaw), 0.0), (sin(yaw), cos(yaw), 0.0), (0.0, 0.0, 1.0)),
        dtype=torch.float64,
    )
    return rz @ ry @ rx


def _origin_matrix(node: ET.Element) -> Tensor:
    origin = node.find("origin")
    xyz = _vector(
        None if origin is None else origin.get("xyz"), default=(0.0, 0.0, 0.0)
    )
    rpy = _vector(
        None if origin is None else origin.get("rpy"), default=(0.0, 0.0, 0.0)
    )
    result = torch.eye(4, dtype=torch.float64)
    result[:3, :3] = _rpy_matrix(rpy)
    result[:3, 3] = torch.tensor(xyz, dtype=torch.float64)
    return result


@dataclass(frozen=True, slots=True)
class _RevoluteJoint:
    origin: Tensor
    axis: Tensor
    position_index: int


class WujiHandKinematics:
    """Batched FK from recorded link7 pose and canonical 27-joint position."""

    def __init__(
        self,
        *,
        joint_eef_origin: Tensor,
        finger_joints: tuple[tuple[_RevoluteJoint, ...], ...],
        tip_origins: tuple[Tensor, ...],
    ) -> None:
        self.joint_eef_origin = joint_eef_origin
        self.finger_joints = finger_joints
        self.tip_origins = tip_origins

    @classmethod
    def from_urdf(
        cls,
        path: str | Path,
        *,
        canonical_joint_names: tuple[str, ...],
        device: torch.device,
    ) -> "WujiHandKinematics":
        urdf_path = Path(path)
        root = ET.parse(urdf_path).getroot()
        joints = {node.get("name"): node for node in root.findall("joint")}
        if "joint_eef" not in joints:
            raise ValueError(f"Wuji URDF lacks joint_eef: {urdf_path}")
        name_to_index = {
            name: index for index, name in enumerate(canonical_joint_names)
        }
        finger_joints: list[tuple[_RevoluteJoint, ...]] = []
        tips: list[Tensor] = []
        for finger in range(1, WUJI_FINGER_COUNT + 1):
            chain: list[_RevoluteJoint] = []
            for level in range(1, WUJI_ACTUATED_JOINTS_PER_FINGER + 1):
                name = f"right_finger{finger}_joint{level}"
                node = joints.get(name)
                if node is None or name not in name_to_index:
                    raise ValueError(
                        f"Wuji FK joint is missing from URDF/action order: {name}"
                    )
                axis_node = node.find("axis")
                axis = torch.tensor(
                    _vector(
                        None if axis_node is None else axis_node.get("xyz"),
                        default=(1.0, 0.0, 0.0),
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                axis = axis / torch.linalg.vector_norm(axis).clamp_min(1.0e-8)
                chain.append(
                    _RevoluteJoint(
                        origin=_origin_matrix(node).to(
                            dtype=torch.float32, device=device
                        ),
                        axis=axis,
                        position_index=name_to_index[name],
                    )
                )
            tip_name = f"right_finger{finger}_tip_fixed"
            tip = joints.get(tip_name)
            if tip is None:
                raise ValueError(f"Wuji URDF lacks fingertip transform: {tip_name}")
            finger_joints.append(tuple(chain))
            tips.append(_origin_matrix(tip).to(dtype=torch.float32, device=device))
        return cls(
            joint_eef_origin=_origin_matrix(joints["joint_eef"]).to(
                dtype=torch.float32, device=device
            ),
            finger_joints=tuple(finger_joints),
            tip_origins=tuple(tips),
        )

    @staticmethod
    def _pose_matrix(pose_wxyz: Tensor) -> Tensor:
        if pose_wxyz.ndim != 2 or pose_wxyz.shape[1] != 7:
            raise ValueError("link7 pose must have shape [N,7]")
        quaternion = pose_wxyz[:, 3:]
        quaternion = quaternion / torch.linalg.vector_norm(
            quaternion, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        w, x, y, z = quaternion.unbind(-1)
        rotation = torch.stack(
            (
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ),
            dim=-1,
        ).reshape(-1, 3, 3)
        result = (
            torch.eye(4, dtype=pose_wxyz.dtype, device=pose_wxyz.device)
            .expand(pose_wxyz.shape[0], -1, -1)
            .clone()
        )
        result[:, :3, :3] = rotation
        result[:, :3, 3] = pose_wxyz[:, :3]
        return result

    @staticmethod
    def _axis_rotation(axis: Tensor, angle: Tensor) -> Tensor:
        count = angle.shape[0]
        x, y, z = axis
        zeros = torch.zeros_like(angle)
        skew = torch.stack(
            (
                zeros,
                -z * torch.ones_like(angle),
                y * torch.ones_like(angle),
                z * torch.ones_like(angle),
                zeros,
                -x * torch.ones_like(angle),
                -y * torch.ones_like(angle),
                x * torch.ones_like(angle),
                zeros,
            ),
            dim=-1,
        ).reshape(count, 3, 3)
        identity = torch.eye(3, dtype=angle.dtype, device=angle.device).expand(
            count, -1, -1
        )
        outer = axis[:, None] * axis[None, :]
        rotation = (
            torch.cos(angle)[:, None, None] * identity
            + torch.sin(angle)[:, None, None] * skew
            + (1.0 - torch.cos(angle))[:, None, None] * outer
        )
        result = (
            torch.eye(4, dtype=angle.dtype, device=angle.device)
            .expand(count, -1, -1)
            .clone()
        )
        result[:, :3, :3] = rotation
        return result

    def keypoint_positions(
        self, link7_pose_robot_base: Tensor, joint_position: Tensor
    ) -> Tensor:
        if (
            joint_position.ndim != 2
            or joint_position.shape != (link7_pose_robot_base.shape[0], 27)
            or joint_position.dtype != torch.float32
            or link7_pose_robot_base.dtype != torch.float32
            or joint_position.device != link7_pose_robot_base.device
        ):
            raise ValueError("Wuji FK inputs must be matching float32 [N,7]/[N,27]")
        transform = self._pose_matrix(link7_pose_robot_base)
        palm = transform @ self.joint_eef_origin
        points: list[Tensor] = []
        for chain, tip in zip(self.finger_joints, self.tip_origins):
            current = palm
            for joint in chain:
                current = (
                    current
                    @ joint.origin
                    @ self._axis_rotation(
                        joint.axis, joint_position[:, joint.position_index]
                    )
                )
                points.append(current[:, :3, 3])
            points.append((current @ tip)[:, :3, 3])
        result = torch.stack(points, dim=1)
        if result.shape != (joint_position.shape[0], WUJI_KEYPOINT_COUNT, 3):
            raise RuntimeError("Wuji FK emitted an unexpected keypoint layout")
        return result


def finite_difference_linear_velocity(position: Tensor, *, frequency_hz: int) -> Tensor:
    if position.ndim < 2 or position.shape[0] < 2 or frequency_hz <= 0:
        raise ValueError("finite-difference input must contain at least two frames")
    velocity = torch.empty_like(position)
    velocity[0] = (position[1] - position[0]) * frequency_hz
    velocity[-1] = (position[-1] - position[-2]) * frequency_hz
    if position.shape[0] > 2:
        velocity[1:-1] = (position[2:] - position[:-2]) * (0.5 * frequency_hz)
    return velocity


def quaternion_angular_velocity_wxyz(
    quaternion: Tensor, *, frequency_hz: int
) -> Tensor:
    if quaternion.ndim != 2 or quaternion.shape[1] != 4 or quaternion.shape[0] < 2:
        raise ValueError("quaternion trajectory must have shape [F,4], F>=2")
    q = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    left = q[1:]
    right = torch.cat((q[:-1, :1], -q[:-1, 1:]), dim=-1)
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    delta = torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )
    delta = torch.where(delta[:, :1] < 0.0, -delta, delta)
    xyz = delta[:, 1:]
    norm = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(norm, delta[:, :1].clamp_min(1.0e-8))
    omega = xyz / norm.clamp_min(1.0e-8) * angle * frequency_hz
    omega = torch.where(norm > 1.0e-8, omega, torch.zeros_like(omega))
    result = torch.empty(
        quaternion.shape[0], 3, dtype=quaternion.dtype, device=quaternion.device
    )
    result[:-1] = omega
    result[-1] = omega[-1]
    return result


__all__ = [
    "WUJI_FINGER_COUNT",
    "WUJI_KEYPOINT_COUNT",
    "WUJI_LEVEL_1_KEYPOINT_INDICES",
    "WUJI_LEVEL_2_KEYPOINT_INDICES",
    "WUJI_TIP_KEYPOINT_INDICES",
    "WujiHandKinematics",
    "finite_difference_linear_velocity",
    "quaternion_angular_velocity_wxyz",
]
