"""The release permits dataset changes but fixes the algorithm."""
from dataclasses import replace
import pytest
from parallel_fixtures import recipe_config
from trajectory_tracking.cli.launch import validate_baseline

def test_data_change_preserves_baseline_and_reward_change_is_rejected():
    config=recipe_config()
    validate_baseline(replace(config, data=replace(config.data, trajectory_count=2)))
    with pytest.raises(ValueError, match="frozen delta-action"):
        validate_baseline(replace(config, reward=replace(config.reward, object_flow_reward_weight=1.15)))
