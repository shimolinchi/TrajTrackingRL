"""Explicit static object slots and independent, same-object trajectory resets."""

import torch


class ObjectTrajectoryAssignment:
    def __init__(self, trajectory_uids, environment_uids, *, device):
        if not trajectory_uids or not environment_uids:
            raise ValueError("object assignment requires non-empty identities")
        names = sorted(set(trajectory_uids))
        if not set(environment_uids) <= set(names):
            raise ValueError("scene object has no matching reference trajectory")
        mapping = {uid: index for index, uid in enumerate(names)}
        self.trajectory_objects = torch.tensor([mapping[uid] for uid in trajectory_uids], device=device)
        self.environment_objects = torch.tensor([mapping[uid] for uid in environment_uids], device=device)
        groups = [[i for i, uid in enumerate(trajectory_uids) if uid == name] for name in names]
        self.counts = torch.tensor([len(group) for group in groups], device=device)
        self.indices = torch.zeros((len(groups), max(map(len, groups))), dtype=torch.long, device=device)
        for index, group in enumerate(groups):
            self.indices[index, :len(group)] = torch.tensor(group, device=device)

    def sample(self, env_ids, *, generator):
        objects = self.environment_objects[env_ids]
        unit = torch.rand(len(env_ids), device=env_ids.device, generator=generator)
        offsets = (unit * self.counts[objects]).long()
        return self.indices[objects, offsets]

    def validate(self, env_ids, trajectories):
        if not torch.equal(self.environment_objects[env_ids], self.trajectory_objects[trajectories]):
            raise ValueError("reference object identity differs from the environment asset")
