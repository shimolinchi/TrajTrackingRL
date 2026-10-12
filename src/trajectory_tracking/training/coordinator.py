"""Synchronous update boundaries; environment creation and presentation stay in CLI."""

from contextlib import nullcontext
from dataclasses import asdict
import math
import gc
import time

import torch

from .parallel.checkpoint import load_checkpoint, save_checkpoint


def json_metrics(value):
    """Undefined metrics are null, never invalid JSON NaN or invented zeros."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_metrics(item) for item in value]
    return value


class SynchronousCoordinator:
    def __init__(self, *, runner, context, artifacts, tracker, contract,
                 checkpoint_kwargs, evaluate_train, evaluate_test=None,
                 pause_recording=nullcontext, periodic_heldout=False, evaluation_scope=None, memory_observer=None, evaluation_cache_cleanup=False):
        self.runner, self.context = runner, context
        self.artifacts, self.tracker = artifacts, tracker
        self.contract, self.checkpoint_kwargs = contract, checkpoint_kwargs
        self.evaluate_train, self.evaluate_test = evaluate_train, evaluate_test
        if periodic_heldout and evaluate_test is None:
            raise ValueError("test-only periodic evaluation requires a test callback")
        self.periodic_heldout = periodic_heldout
        self.pause_recording = pause_recording
        self.evaluation_scope = evaluation_scope or (lambda **metadata: nullcontext())
        self.memory_observer = memory_observer
        self.evaluation_cache_cleanup = evaluation_cache_cleanup
        self.started = time.perf_counter()
        self.evaluation_seconds = 0.0

    def emit(self, payload):
        if not self.context.primary:
            return
        payload = json_metrics(dict(payload, training_elapsed_seconds=time.perf_counter() - self.started))
        self.artifacts.append_metrics(payload)
        if self.tracker is not None:
            self.tracker.log_event(payload)
        print(f"[{payload['event']}] update={payload.get('update', 0)} "
              f"success={payload.get('episode_success_rate')} "
              f"valid={payload.get('valid_transition_count')}", flush=True)

    def evaluate(self, update, *, heldout=False):
        # Inference has no collectives. Different-length ranks join only after
        # finishing evaluation. Restore the training RNG and live state first.
        if self.memory_observer is not None:
            self.memory_observer("evaluation_begin", update)
        if self.evaluation_cache_cleanup and self.context.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            if self.memory_observer is not None:
                self.memory_observer("evaluation_after_cache_release", update)
        self.context.barrier()
        start = time.perf_counter()
        saved = self.runner.env.rollout_state_dict()
        mode = self.runner.trainer.model.training
        with self.pause_recording():
            try:
                with self.evaluation_scope(update=update, split="test" if heldout else "train", policy="learned"):
                    payload = self.evaluate_test() if heldout else self.evaluate_train()
            finally:
                self.runner.env.load_rollout_state_dict(saved)
                self.runner.trainer.model.train(mode)
                callback = self.runner.observation_callback
                if callback is not None:
                    callback(self.runner.env)
            self.context.barrier()
        elapsed = float(self.context.reduce(torch.tensor(time.perf_counter() - start), "max"))
        if self.memory_observer is not None:
            self.memory_observer("evaluation_end_snapshot_still_live", update)
        del saved
        if self.evaluation_cache_cleanup and self.context.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        if self.memory_observer is not None:
            self.memory_observer("evaluation_after_snapshot_release", update)
        self.evaluation_seconds += elapsed
        self.emit(dict(payload, update=update, evaluation_seconds=elapsed,
                       parallel_metrics={"evaluation_fraction": self.evaluation_seconds /
                                         (time.perf_counter() - self.started)}))

    def run(self, *, max_updates, evaluation_interval, test_interval,
            checkpoint_interval, resume=None):
        initial = 0 if resume is None else load_checkpoint(
            resume, context=self.context, contract=self.contract, runner=self.runner)
        if initial >= max_updates:
            raise ValueError("max updates must exceed resume update")
        if self.runner.env.needs_reset:
            self.runner.env.reset()
        if self.evaluate_test is not None and test_interval:
            self.evaluate(initial, heldout=True)
        for update in range(initial + 1, max_updates + 1):
            start = time.perf_counter()
            if self.memory_observer is not None:
                self.memory_observer("update_begin", update)
                self.runner.memory_observer = lambda stage, **details: self.memory_observer(stage, update, **details)
                self.runner.trainer.memory_observer = self.runner.memory_observer
            stats = self.runner.run_update()
            if self.memory_observer is not None:
                self.memory_observer("update_end", update)
            payload = asdict(stats)
            payload.update({f"ppo_{key}": value for key, value in payload.pop("ppo").items()})
            # These diagnostics describe live local slots, unlike globally pooled
            # transition metrics. Aggregate explicitly rather than publish rank0.
            for key in ("start_index_min", "start_index_mean", "start_index_max",
                        "reference_cursor_min", "reference_cursor_mean", "reference_cursor_max",
                        "episode_progress_mean"):
                operation = "min" if key.endswith("_min") else "max" if key.endswith("_max") else "sum"
                value = self.context.reduce(torch.tensor(float(payload[key]), dtype=torch.float64), operation)
                payload[key] = float(value) / (self.context.world_size if operation == "sum" else 1)
            timings = {name: float(self.context.reduce(torch.tensor(value), "max"))
                       for name, value in self.runner.last_timing.items()}
            seconds = float(self.context.reduce(torch.tensor(time.perf_counter() - start), "max"))
            self.emit(dict(payload, event="update", update=update, iteration_seconds=seconds,
                           parallel_metrics={**timings, "world_size": self.context.world_size,
                               "valid_transitions_per_second": stats.valid_transition_count / seconds}))
            if update % evaluation_interval == 0 or update == max_updates:
                self.evaluate(update, heldout=self.periodic_heldout)
            if not self.periodic_heldout and self.evaluate_test is not None and test_interval and (
                    update % test_interval == 0 or update == max_updates):
                self.evaluate(update, heldout=True)
            if update % checkpoint_interval == 0 or update == max_updates:
                path = save_checkpoint(
                    self.artifacts.run_dir / "checkpoints" / f"update_{update:06d}",
                    context=self.context, contract=self.contract, runner=self.runner,
                    update=update, checkpoint_kwargs=self.checkpoint_kwargs)
                self.emit({"event": "checkpoint", "update": update, "path": str(path)})
        self.emit({"event": "complete", "update": max_updates})
        return max_updates
