from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
from trajectory_tracking.environment.scene import _object_collider_mask, _check_collider_values


def test_shape_mask_uses_view_path_order_and_reuses_same_asset_count():
    paths = ["/World/envs/env_2/Object", "/World/envs/env_0/Object", "/World/envs/env_1/Object"]
    maker = Mock(side_effect=lambda path: SimpleNamespace(max_shapes=1 if "env_1/" in path else 3))
    asset = SimpleNamespace(root_physx_view=SimpleNamespace(prim_paths=paths, max_shapes=3),
                            _physics_sim_view=SimpleNamespace(create_rigid_body_view=maker))
    mask = _object_collider_mask(SimpleNamespace(object_uids=("a", "b", "a")), asset)
    assert mask.tolist() == [[True, True, True], [True, True, True], [True, False, False]]
    assert maker.call_count == 2


@pytest.mark.parametrize("channels", [None, 3])
def test_padding_ignored_but_real_mismatch_rejected(channels):
    shape = (2, 3) if channels is None else (2, 3, channels)
    wanted = torch.full(shape, 0.6)
    actual = wanted.clone()
    mask = torch.tensor([[True, False, False], [True, True, True]])
    actual[~mask] = float("nan")
    _check_collider_values("object", "test", actual, wanted, mask)
    actual[1, 0] = 0.2
    with pytest.raises(RuntimeError, match="real collider shapes"):
        _check_collider_values("object", "test", actual, wanted, mask)


def test_homogeneous_view_still_checks_all_entries():
    with pytest.raises(RuntimeError, match="real collider shapes"):
        _check_collider_values("robot", "contact", torch.zeros(2, 2), torch.ones(2, 2), None)
