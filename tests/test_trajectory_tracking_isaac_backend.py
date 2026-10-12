"""CPU contract tests for the duck-typed Isaac trajectory backend."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from trajectory_tracking.environment.core import (
    IsaacTrajectoryBackend,
    action_spec_from_isaac_robot,
)
from trajectory_tracking.data.reference import build_reference_episode
from trajectory_tracking.utils.synthetic import synthetic_reference_trajectory


_FINGERTIP_NAMES = tuple(f"right_finger{index}_link4" for index in range(1, 6))
_BODY_NAMES = (*_FINGERTIP_NAMES, "link7", *(f"right_finger{i}_link{j}" for i in range(1, 6) for j in range(1, 4)))


class _FakeScene:
    def __init__(self, num_envs: int) -> None:
        self.env_origins = torch.zeros(num_envs, 3, dtype=torch.float32)
        self.env_origins[:, 0] = torch.arange(num_envs, dtype=torch.float32) + 10.0
        self.write_count = 0
        self.update_dts: list[float] = []

    def __getitem__(self, key):
        raise KeyError(key)

    def write_data_to_sim(self) -> None:
        self.write_count += 1

    def update(self, dt: float) -> None:
        self.update_dts.append(float(dt))


class _FakeRigidAsset:
    def __init__(self, num_envs: int) -> None:
        root_quaternion = torch.zeros(num_envs, 4, dtype=torch.float32)
        root_quaternion[:, 0] = 1.0
        self.data = SimpleNamespace(
            root_pos_w=torch.zeros(num_envs, 3, dtype=torch.float32),
            root_quat_w=root_quaternion,
            root_lin_vel_w=torch.zeros(num_envs, 3, dtype=torch.float32),
            root_ang_vel_w=torch.zeros(num_envs, 3, dtype=torch.float32),
        )

    def write_root_pose_to_sim(
        self, pose: torch.Tensor, *, env_ids: torch.Tensor
    ) -> None:
        self.data.root_pos_w[env_ids] = pose[:, :3]
        self.data.root_quat_w[env_ids] = pose[:, 3:]

    def write_root_velocity_to_sim(
        self, velocity: torch.Tensor, *, env_ids: torch.Tensor
    ) -> None:
        self.data.root_lin_vel_w[env_ids] = velocity[:, :3]
        self.data.root_ang_vel_w[env_ids] = velocity[:, 3:]


class _FakeRobot(_FakeRigidAsset):
    def __init__(self, num_envs: int, canonical_names: tuple[str, ...]) -> None:
        super().__init__(num_envs)
        lab_names = tuple(reversed(canonical_names))
        lab_width = torch.arange(1, 28, dtype=torch.float32)
        limits = torch.stack((-lab_width, lab_width), dim=-1).unsqueeze(0)
        body_state = torch.zeros(num_envs, len(_BODY_NAMES), 13, dtype=torch.float32)
        body_state[..., 3] = 1.0
        self.data.joint_names = lab_names
        self.data.joint_pos_limits = limits
        self.data.joint_pos = torch.zeros(num_envs, 27, dtype=torch.float32)
        self.data.joint_vel = torch.zeros(num_envs, 27, dtype=torch.float32)
        self.data.applied_torque = torch.zeros(num_envs, 27, dtype=torch.float32)
        self.data.body_names = _BODY_NAMES
        self.data.body_state_w = body_state
        self.target = torch.zeros(num_envs, 27, dtype=torch.float32)

    def write_root_pose_to_sim(
        self, pose: torch.Tensor, *, env_ids: torch.Tensor
    ) -> None:
        super().write_root_pose_to_sim(pose, env_ids=env_ids)
        self.data.body_state_w[env_ids, :, :3] = pose[:, None, :3]
        self.data.body_state_w[env_ids, :, 3:7] = pose[:, None, 3:]

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        *,
        env_ids: torch.Tensor,
    ) -> None:
        self.data.joint_pos[env_ids] = position
        self.data.joint_vel[env_ids] = velocity

    def set_joint_position_target(
        self, target: torch.Tensor, *, env_ids: torch.Tensor
    ) -> None:
        self.target[env_ids] = target


class _FakeSimulation:
    def __init__(self, robot: _FakeRobot) -> None:
        self.robot = robot
        self.forward_count = 0
        self.step_count = 0

    def forward(self) -> None:
        self.forward_count += 1

    def step(self, *, render: bool) -> None:
        if render:
            raise AssertionError("headless backend step must not request rendering")
        self.step_count += 1
        self.robot.data.joint_vel.copy_(
            (self.robot.target - self.robot.data.joint_pos) * 120.0
        )
        self.robot.data.joint_pos.copy_(self.robot.target)


class IsaacTrajectoryBackendContractTest(unittest.TestCase):
    def test_restore_mapping_and_one_control_step_without_isaac_import(self) -> None:
        num_envs = 2
        trajectory = synthetic_reference_trajectory()
        robot = _FakeRobot(num_envs, trajectory.joint_names)
        action_spec = action_spec_from_isaac_robot(
            robot, canonical_joint_names=trajectory.joint_names
        )
        self.assertEqual(action_spec.position_lower_rad[0], -27.0)
        self.assertEqual(action_spec.position_lower_rad[-1], -1.0)

        scene = _FakeScene(num_envs)
        object_asset = _FakeRigidAsset(num_envs)
        table = _FakeRigidAsset(num_envs)
        simulation = _FakeSimulation(robot)
        backend = IsaacTrajectoryBackend(
            simulation_context=simulation,
            scene=scene,
            robot=robot,
            table=table,
            object_asset=object_asset,
            trajectory=trajectory,
            action_spec=action_spec,
            physics_dt=1.0 / 120.0,
        )
        episode = build_reference_episode(
            trajectory,
            start_indices=torch.tensor([121, 121], dtype=torch.int64),
            pre_roll_policy_steps=1,
            flow_anchor_offsets_steps=(0, 30, 60, 90, 120),
        )
        backend.restore(
            episode.reset_state(),
            env_ids=torch.arange(num_envs, dtype=torch.int64),
        )
        self.assertEqual(simulation.forward_count, 1)
        self.assertEqual(scene.update_dts, [0.0])
        self.assertTrue(torch.equal(robot.data.root_pos_w, scene.env_origins))
        self.assertTrue(torch.equal(object_asset.data.root_pos_w, scene.env_origins))
        link7_offset = torch.tensor((0.10, -0.20, 0.30), dtype=torch.float32)
        robot.data.body_state_w[:, _BODY_NAMES.index("link7"), :3] += link7_offset

        canonical_target = torch.linspace(-0.5, 0.5, 27).repeat(num_envs, 1)
        state = backend.step(canonical_target)
        self.assertEqual(simulation.step_count, 2)
        self.assertEqual(scene.write_count, 3)
        self.assertEqual(scene.update_dts, [0.0, 1.0 / 120.0, 1.0 / 120.0])
        self.assertTrue(torch.equal(robot.target, canonical_target.flip(dims=(1,))))
        self.assertTrue(torch.equal(state.joint_position, canonical_target))
        self.assertTrue(
            torch.allclose(
                state.link7_position_robot_base,
                link7_offset.expand(num_envs, -1),
                rtol=0.0,
                atol=1.0e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                state.object_surface_points_robot_base,
                trajectory.object_points_object_frame.expand(num_envs, -1, -1),
            )
        )
        expected_fingertips = torch.tensor(
            (
                (-0.0011048, 0.00023218, 0.0283),
                (-0.00105, 0.0, 0.0274),
                (-0.0010491, 0.0, 0.0267),
                (-0.0010528, 0.0, 0.0267),
                (-0.0010446, 0.00001131, 0.0267),
            ),
            dtype=torch.float32,
        ).expand(num_envs, -1, -1)
        self.assertTrue(
            torch.allclose(
                state.fingertip_positions_robot_base,
                expected_fingertips,
                rtol=0.0,
                atol=1.0e-6,
            )
        )
        state.validate(
            num_envs=num_envs,
            num_object_points=4,
            fingertip_count=5,
            hand_keypoint_count=25,
            device=torch.device("cpu"),
        )

        # Resetting one logical environment must not overwrite the simulator
        # state or the cached actuator target of any other environment.
        robot.data.root_pos_w[0] = torch.tensor((101.0, 102.0, 103.0))
        object_asset.data.root_pos_w[0] = torch.tensor((201.0, 202.0, 203.0))
        untouched_joint = robot.data.joint_pos[0].clone()
        untouched_target = robot.target[0].clone()
        untouched_robot_root = robot.data.root_pos_w[0].clone()
        untouched_object_root = object_asset.data.root_pos_w[0].clone()
        subset_episode = build_reference_episode(
            trajectory,
            start_indices=torch.tensor([122], dtype=torch.int64),
            pre_roll_policy_steps=1,
            flow_anchor_offsets_steps=(0, 30, 60, 90, 120),
        )

        backend.restore(
            subset_episode.reset_state(),
            env_ids=torch.tensor([1], dtype=torch.int64),
        )

        self.assertTrue(torch.equal(robot.data.joint_pos[0], untouched_joint))
        self.assertTrue(torch.equal(robot.target[0], untouched_target))
        self.assertTrue(torch.equal(robot.data.root_pos_w[0], untouched_robot_root))
        self.assertTrue(
            torch.equal(object_asset.data.root_pos_w[0], untouched_object_root)
        )
        self.assertTrue(torch.equal(robot.data.joint_pos[1], torch.zeros(27)))
        self.assertTrue(torch.equal(robot.target[1], torch.zeros(27)))
        self.assertTrue(torch.equal(robot.data.root_pos_w[1], scene.env_origins[1]))
        self.assertTrue(
            torch.equal(object_asset.data.root_pos_w[1], scene.env_origins[1])
        )


if __name__ == "__main__":
    unittest.main()
