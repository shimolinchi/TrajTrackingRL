"""Complete distributed rows versus explicitly permitted single-scene fillers."""
from types import SimpleNamespace

import pytest

from trajectory_tracking.training.heldout import test_summary as summarize


def rows():
    return [SimpleNamespace(trajectory_index=i, start_index=0, episode_success=i % 2,
        mean_object_flow_error_m=.1, final_object_flow_error_m=.2, object_departed=False,
        object_lifted_2cm=True, mean_link7_position_error_m=.3, mean_hand_position_rmse_rad=.4)
        for i in range(603)]


def test_complete_603_and_single_scene_fillers():
    data = rows()
    assert summarize(data, 603, require_exact_count=True)['episode_count'] == 603
    assert summarize(data + data[:3], 603) == summarize(data, 603)


@pytest.mark.parametrize('mutation', ['tail_duplicate', 'missing', 'duplicate', 'nonzero_start'])
def test_strict_aggregation_rejects_invalid_rows(mutation):
    data = rows()
    if mutation == 'tail_duplicate':
        data.append(data[-1])
    elif mutation == 'missing':
        data.pop()
    elif mutation == 'duplicate':
        data[-2] = data[-1]
    else:
        data[0].start_index = 1
    with pytest.raises(ValueError):
        summarize(data, 603, require_exact_count=True)
