"""Arguments shared by the multi-process training composition root."""

import os
import argparse
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Iterator
from ..environment.scene import DEFAULT_ENV_SPACING_M

REPOSITORY_ROOT = Path(os.environ.get("TRAJTRACKING_WORKSPACE", Path.cwd())).resolve()


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train trajectory tracking PPO on recorded trajectories."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument(
        "--env-spacing-m",
        type=float,
        default=DEFAULT_ENV_SPACING_M,
    )
    parser.add_argument("--max-updates", type=int, required=True)
    parser.add_argument("--checkpoint-interval", type=int, required=True)
    parser.add_argument("--evaluation-interval", type=int, required=True)
    parser.add_argument("--heldout-test-interval", type=int, default=0,
                        help="opt-in frozen split test at update 0, every N updates and final; 0 disables")
    parser.add_argument("--target-success-rate", type=float, required=True)
    parser.add_argument("--success-patience", type=int, required=True)
    parser.add_argument("--disable-success-early-stop", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--gate-no-offset-mean-error-m", type=float, required=True)
    parser.add_argument("--gate-no-offset-success-rate", type=float, required=True)
    parser.add_argument("--scene-asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--browser-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--browser-host", default="127.0.0.1")
    parser.add_argument("--browser-port", type=int, default=8002)
    parser.add_argument("--browser-env-index", type=int, default=0)
    parser.add_argument("--browser-publish-every-steps", type=int, default=2)
    parser.add_argument(
        "--browser-object-flow-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "show reference/actual object-flow polylines; the reference object "
            "wireframe remains visible when disabled"
        ),
    )
    parser.add_argument(
        "--wandb-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline"),
        default="online",
    )
    parser.add_argument("--wandb-project", default="interprior-bc2resrl")
    parser.add_argument("--wandb-group", default="trajectory-residual-ppo")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument(
        "--nominal-replay-gate-mode",
        choices=("blocking", "diagnostic"),
        default="blocking",
        help="whether a failed pre-training source-action replay stops training",
    )
    parser.add_argument(
        "--scene-build-mode", choices=("legacy", "batched_fabric_notices"),
        default="legacy", help="USD construction scheduling only; physics settings are unchanged",
    )
    return parser.parse_args(argv)


@contextmanager
def _pause_browser_recording_during_evaluation(
    browser: Any | None,
) -> Iterator[None]:
    """Pause MediaRecorder clients while an evaluation uses the live viewer."""

    if browser is None:
        yield
        return
    browser.set_recording_paused(True)
    try:
        yield
    finally:
        browser.set_recording_paused(False)
