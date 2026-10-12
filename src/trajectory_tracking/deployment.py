"""Deterministic inference using the same weights, normalization and action mapping."""
from dataclasses import dataclass
from pathlib import Path

import torch

from .environment.incremental_arm_action import ArmTargetIncrementProjector
from .training.artifacts import load_ppo_policy_checkpoint
from .training.ppo import PPOTrainer, create_ppo_trainer


@dataclass
class TrackingPolicy:
    trainer: PPOTrainer
    projector: ArmTargetIncrementProjector
    update: int

    @classmethod
    def load(cls, checkpoint, *, config, action_spec, trajectory_id, scene_manifest,
             adapter_version, device="cpu"):
        trainer = create_ppo_trainer(observation_dim=config.observation.feature_dim,
            action_dim=27, config=config.ppo, device=torch.device(device), seed=config.seed)
        checkpoint = Path(checkpoint)
        if checkpoint.is_dir():
            checkpoint = checkpoint / "policy.pt"
        state = load_ppo_policy_checkpoint(checkpoint, trainer=trainer, config=config,
            action_spec=action_spec, trajectory_id=trajectory_id,
            scene_manifest=scene_manifest, adapter_version=adapter_version)
        projector = ArmTargetIncrementProjector(action_spec, config.direct_action)
        return cls(trainer, projector, state.update)

    @torch.no_grad()
    def normalized_action(self, observation):
        return self.trainer.deterministic_action(observation)

    @torch.no_grad()
    def joint_targets(self, observation, previous_joint_target):
        return self.projector.apply(self.normalized_action(observation),
            previous_joint_target=previous_joint_target).executed_action
