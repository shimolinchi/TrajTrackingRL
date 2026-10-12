"""Placement-only storage contract; no dataset, GPU or simulator dependency."""
import pytest
import torch
from trajectory_tracking.data.host_reference import HostReferenceBank, load_reference_collection
from trajectory_tracking.data.variable_length import VariableLengthReferenceBank


def cpu_bank():
    bank = VariableLengthReferenceBank.__new__(VariableLengthReferenceBank)
    bank.trajectory_ids = ('a', 'b')
    bank.joint_names = ('q',)
    bank.control_hz, bank.physics_hz = 60, 120
    for name in bank._FRAME_FIELDS:
        setattr(bank, name, torch.arange(24, dtype=torch.float32).reshape(2, 4, 3))
    bank.object_points_object_frame = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    bank.robot_root_pose_env_local = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    bank.table_root_pose_env_local = bank.robot_root_pose_env_local.clone()
    bank.valid_frame_counts = torch.tensor([3, 4])
    return bank


def test_all_fields_exact_no_duplicate_bank():
    bank = cpu_bank()
    host = HostReferenceBank.from_bank(bank, device='cpu')
    rows, frames = torch.tensor([1, 0]), torch.tensor([[3, 1], [0, 2]])
    for name in bank._FRAME_FIELDS:
        assert torch.equal(bank.gather_frames(name, rows, frames), host.gather_frames(name, rows, frames))
        assert getattr(bank, name).data_ptr() == getattr(host, name).data_ptr()
    for name in ('object_points_object_frame', 'robot_root_pose_env_local', 'table_root_pose_env_local'):
        assert torch.equal(bank.gather_trajectory_values(name, rows), host.gather_trajectory_values(name, rows))
    assert host.terminal_indices(rows).tolist() == [3, 2]


@pytest.mark.parametrize("host", [False, True])
def test_terminal_frame_view_shortens_only_evaluation_bank(host):
    original = cpu_bank()
    bank = HostReferenceBank.from_bank(original, device="cpu") if host else original
    view = bank.with_terminal_frame_counts(
        [2, 3], protocol="heldout_local0_pre_place_descend_v1",
        metadata={"terminal_phase_excluded": "place_descend"},
    )
    assert bank.valid_frame_counts.tolist() == [3, 4]
    assert view.valid_frame_counts.tolist() == [2, 3]
    assert view.terminal_indices(torch.tensor([0, 1])).tolist() == [1, 2]
    assert view.heldout_terminal_protocol == "heldout_local0_pre_place_descend_v1"
    if host:
        assert view._host_valid_frame_counts.tolist() == [2, 3]
        assert bank._host_valid_frame_counts.tolist() == [3, 4]


def test_terminal_frame_view_rejects_longer_or_too_short_rows():
    bank = cpu_bank()
    for counts in ([4, 4], [1, 3], [2]):
        with pytest.raises(ValueError, match="shorten existing rows"):
            bank.with_terminal_frame_counts(counts, protocol="test", metadata={})


@pytest.mark.parametrize('rows,frames', [([0], [3]), ([2], [0]), ([0], [-1])])
def test_reject_padding(rows, frames):
    with pytest.raises(ValueError):
        HostReferenceBank.from_bank(cpu_bank(), device='cpu').gather_frames(
            'joint_position', torch.tensor(rows), torch.tensor(frames))


@pytest.mark.parametrize('device', ['cpu', 'meta'])
def test_browser_moves_only_one_valid_sequence(device):
    host = HostReferenceBank.from_bank(cpu_bank(), device=device)
    for name in host._FRAME_FIELDS:
        sequence = host.frame_sequence(name, 0)
        assert sequence.device.type == device and sequence.shape == (3, 3)
        assert getattr(host, name).device.type == 'cpu'


def test_public_loader_preserves_adapter_config_and_sources(monkeypatch):
    from trajectory_tracking.data import interprior
    config, sources = object(), (object(),)
    def load(actual, **kwargs):
        assert actual is config and kwargs['sources'] is sources
        assert kwargs['device'] == torch.device('cpu')
        return cpu_bank()
    monkeypatch.setattr(interprior, 'load_interprior_trajectory_collection', load)
    host = load_reference_collection(config, project_root='unused', device='meta', sources=sources, storage='cpu')
    assert isinstance(host, HostReferenceBank) and host.device.type == 'meta'
    resident = load_reference_collection(config, project_root='unused', device=torch.device('cpu'), sources=sources)
    assert type(resident) is VariableLengthReferenceBank


def test_temporal_environment_with_host_bank_matches_resident_cpu():
    from dataclasses import replace
    from parallel_fixtures import recipe_config
    from trajectory_tracking.environment.core import TrajectoryDirectEnvCore
    from trajectory_tracking.utils.synthetic import (
        synthetic_reference_trajectory, synthetic_action_spec, LinearRecoveryBackend,
    )
    rows = tuple(replace(synthetic_reference_trajectory(frame_count=n), trajectory_id=f'row-{n}')
                 for n in (24, 30))
    bank = VariableLengthReferenceBank(rows)
    config = recipe_config()
    config = replace(config, data=replace(config.data, trajectory_count=2))
    envs = [TrajectoryDirectEnvCore(config=config, trajectory=reference,
                action_spec=synthetic_action_spec(), backend=LinearRecoveryBackend(num_envs=2, device=torch.device('cpu')),
                num_envs=2) for reference in (bank, HostReferenceBank.from_bank(bank, device='cpu'))]
    for env in envs:
        env.reset(start_indices=torch.tensor([1, 20]), trajectory_indices=torch.tensor([0, 1]))
    for _ in range(12):
        transitions = [env.step(torch.zeros(2, 27), auto_reset=True) for env in envs]
        torch.testing.assert_close(transitions[0].reward, transitions[1].reward, atol=0, rtol=0)
        torch.testing.assert_close(envs[0].current_observation(), envs[1].current_observation(), atol=0, rtol=0)
        assert torch.equal(envs[0].reference_cursors, envs[1].reference_cursors)
