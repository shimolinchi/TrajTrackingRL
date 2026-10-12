"""torchrun composition root: one local scene per worker, one policy/run globally."""

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
from functools import partial
from hashlib import sha256
import os
import shlex
from pathlib import Path
import subprocess
import sys
from threading import Timer
import time
import traceback
from types import SimpleNamespace

import torch

from .arguments import REPOSITORY_ROOT, _parse_args, _pause_browser_recording_during_evaluation
from .producer_admission import producer_admission
from ..configuration import load_trajectory_direct_config
from ..data.interprior import (
    interprior_trajectory_adapter_version, load_interprior_trajectory_collection,
    resolve_interprior_trajectory_sources, resolve_trajectory_scene_source_for_data,
)
from ..data.partition import (
    evaluation_batches, evaluation_owners, local_data_config, local_scene, partition_environments,
    source_object_uid, resident_evaluation_sources,
)
from ..environment.core import TrajectoryDirectEnvCore
from ..environment.scene_config import resolve_trajectory_scene_source, resolved_trajectory_scene_manifest
from ..training.artifacts import (
    RunArtifacts, TrajectoryWandbMirror, WandbTrackingSpec, _atomic_json,
    repository_lineage, stable_json_hash,
)
from ..training.coordinator import SynchronousCoordinator
from ..training.parallel import ParallelContext, PPOExecution
from ..training.ppo import create_ppo_trainer
from ..training.run_control import NominalReplayGateSpec, TrainingControlSpec
from ..training.trainer import PPOOverfitRunner, evaluate_episode



def _close_simulator_process(app, exit_code):
    """Finish this worker after artifacts, browser and collectives are closed.

    No Replicator output is owned here. Match the single-process runner's
    immediate Isaac shutdown; full stage cleanup can stall in timeline callbacks.
    """
    if sys.exc_info()[0] is not None:
        traceback.print_exc()
    print(f"[EXIT] distributed worker terminating with code {exit_code}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    # Native fast shutdown may terminate with zero before Python finally runs.
    # External resources are already closed by the caller; preserve failures now.
    if exit_code != 0:
        os._exit(exit_code)
    try:
        app.close(wait_for_replicator=False, skip_cleanup=True)
    finally:
        os._exit(exit_code)


def _abort_worker(runtime, tracker):
    """Report the original failure before best-effort, process-local cleanup.

    Never destroy distributed groups here: peers may be in another collective,
    and even barrier-free NCCL destruction can block. Recorder scopes have
    already unwound; only this runtime's browser and tracker are owned here.
    """
    try:
        traceback.print_exc()
        sys.stderr.flush()
        # A stuck optional cleanup must not keep a failed torchrun worker alive.
        deadline = Timer(5.0, os._exit, args=(1,))
        deadline.daemon = True
        deadline.start()
        browser = None if runtime is None else runtime.browser
        for name, close in (("browser", None if browser is None else browser.close),
                            ("tracker", None if tracker is None else
                             partial(tracker.finish, exit_code=1))):
            if close is not None:
                try:
                    close()
                except BaseException:
                    print(f"[cleanup] {name} failed after the original worker error:",
                          file=sys.stderr, flush=True)
                    traceback.print_exc()
        print("[EXIT] distributed worker terminating with code 1", flush=True)
        sys.stderr.flush()
    finally:
        # Do not enter Kit shutdown or Python's implicit process-group finalizers.
        os._exit(1)


def _arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--evaluate-checkpoint", type=Path)
    parser.add_argument("--reference-storage", choices=("execution", "cpu"), default="execution")
    parser.add_argument("--new-wandb-run-on-resume", action="store_true",
                        help="Keep checkpoint lineage but fork tracking history after invalid training")
    parser.add_argument("--gpu-collision-stack-mib", type=int, default=None)
    parser.add_argument("--release-cuda-cache-before-evaluation", action="store_true")
    parser.add_argument("--memory-min-free-mib", type=int, default=0)
    parser.add_argument("--memory-snapshot-update", type=int, default=-1)
    parser.add_argument("--memory-telemetry", action="store_true",
                        help="Log per-rank allocator counters at update/evaluation boundaries")
    parser.add_argument("--source-label-admission", action="store_true",
                        help="Waive nominal replay only for a frozen producer-success admission proof")
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--parallel-timeout-seconds", type=int, default=7200)
    parser.add_argument("--cuda-device-indices",
                        help="Comma-separated CUDA ordinals in rank order; does not remap device visibility")
    parser.add_argument("--evaluation-data-split", choices=("train", "test"), default="train",
                        help="Use the fixed local-zero held-out protocol for every evaluation when set to test")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--record-evaluation-video", action="store_true",
                        help="Record 60s live browser at every evaluation; retain latest three")
    parser.add_argument("--evaluation-html-port", type=int, default=None)
    parser.add_argument("--video-duration-seconds", type=int, default=40)
    extra, remaining = parser.parse_known_args()
    args = _parse_args(remaining)
    for key, value in vars(extra).items():
        setattr(args, key, value)
    if args.source_label_admission and args.synthetic_smoke:
        raise ValueError("source-label-admission cannot be used with synthetic smoke")
    if args.expected_world_size <= 0 or args.num_envs <= 0 or args.num_envs % args.expected_world_size:
        raise ValueError("--num-envs is GLOBAL and must divide evenly over workers")
    if min(args.max_updates, args.checkpoint_interval, args.evaluation_interval) <= 0:
        raise ValueError("update/evaluation/checkpoint intervals must be positive")
    if args.heldout_test_interval < 0:
        raise ValueError("held-out interval must be nonnegative")
    if args.evaluation_data_split == "test" and (args.synthetic_smoke or not args.heldout_test_interval):
        raise ValueError("test-only evaluation requires real frozen held-out data")
    if args.parallel_timeout_seconds <= 0:
        raise ValueError("parallel timeout must be positive")
    if not args.disable_success_early_stop:
        raise ValueError("synchronous v1 requires explicit --disable-success-early-stop")
    if args.synthetic_smoke and (args.browser_enabled or args.record_video or args.record_evaluation_video or args.wandb_enabled
                                 or args.heldout_test_interval):
        raise ValueError("synthetic smoke cannot record physics video, use W&B or real held-out data")
    if args.record_video and args.record_evaluation_video:
        raise ValueError("choose evaluation video or training-periodic video, not both")
    if args.record_video:
        raise ValueError("periodic training video is disabled; use --record-evaluation-video")
    if args.record_evaluation_video and not args.browser_enabled:
        raise ValueError("record-video requires browser-enabled")
    if args.video_duration_seconds <= 0:
        raise ValueError("video duration must be a positive integer")
    if not 0 <= args.browser_env_index < args.num_envs // args.expected_world_size:
        raise ValueError("browser-env-index must fit rank0's local environment count")
    return args


def _run_nominal_gate(args, config, coordinator, evaluation_scope, evaluate, test_callback,
                      source_admission):
    """Only the pretraining nominal gate is waived; learned evaluation is separate."""
    if args.resume_checkpoint is not None or source_admission is not None:
        return
    no_offset = replace(config, perturbation=replace(config.perturbation,
                                                    mode="none", pre_roll_policy_steps=0))
    with coordinator.pause_recording(), evaluation_scope(
            update=0, split=args.evaluation_data_split, policy="nominal"):
        result = (test_callback(nominal=True) if args.evaluation_data_split == "test"
                  else evaluate(no_offset, nominal=True))
    gate = NominalReplayGateSpec(mean_error_threshold_m=args.gate_no_offset_mean_error_m,
                                success_rate_threshold=args.gate_no_offset_success_rate,
                                enforcement=args.nominal_replay_gate_mode)
    passed = gate.passes(mean_flow_error_m=result["mean_flow_error_m"],
                         episode_success_rate=result["episode_success_rate"])
    coordinator.emit(dict(result, update=0, nominal_replay_gate_passed=passed,
                          nominal_replay_gate_enforcement=gate.enforcement))
    if not passed and gate.enforcement == "blocking":
        raise RuntimeError("nominal replay gate failed; distributed training not started")


def _bank_id(sources):
    ids = [source.entry.episode_uid for source in sources]
    return ids[0] if len(ids) == 1 else "trajectory_bank:" + sha256("\n".join(ids).encode("ascii")).hexdigest()


def admit_data(args, config):
    """Verify the global dataset once, before sharing immutable source metadata."""
    root = REPOSITORY_ROOT
    sources = resolve_interprior_trajectory_sources(config, project_root=root)
    source = resolve_trajectory_scene_source(project_root=root, scene_asset_root=args.scene_asset_root)
    source = resolve_trajectory_scene_source_for_data(config, source, project_root=root)
    test_config = test_sources = split = None
    if args.heldout_test_interval:
        from ..data.heldout_split import heldout_test_config
        test_config, split = heldout_test_config(config, root)
        split["periodic_C_dataset"] = args.evaluation_data_split
        test_sources = resolve_interprior_trajectory_sources(test_config, project_root=root)
    return sources, source, test_config, test_sources, split


def prepare_data(args, config, context, *, admission=None):
    """Build local views only after global source/selection/asset hash validation."""
    sources, source, test_config, test_sources, split = (
        admit_data(args, config) if admission is None else admission)
    partitions = partition_environments(source, sources, total_envs=args.num_envs,
                                        world_size=context.world_size)
    partition = partitions[context.rank]
    selected = tuple(sources[i] for i in partition.source_indices)
    local_config = local_data_config(config, selected)
    scene_manifest = resolved_trajectory_scene_manifest(source)
    if config.experiment_plugin is not None:
        scene_manifest["experiment_runtime"] = config.scene.to_manifest()
    test = None
    if args.heldout_test_interval:
        # Assign each test row once. Local banks also contain same-object fillers
        # for all resident physical slots; these are explicitly never scored.
        uids = (tuple(source_object_uid(item) for item in test_sources)
                if source.object_variants else (None,) * len(test_sources))
        owners = evaluation_owners(partitions, uids)
        resident = set(partition.object_uids or (None,))
        local_sources, test_indices = resident_evaluation_sources(test_sources, selected, resident)
        # Homogeneous legacy scenes have no explicit object UID.
        if not source.object_variants:
            test_indices = tuple(range(len(test_sources)))
            local_sources = tuple(test_sources)
        test = SimpleNamespace(
            config=local_data_config(test_config, local_sources), sources=local_sources,
            global_indices=tuple(test_indices),
            owned_indices=tuple(i for i, g in enumerate(test_indices) if g >= 0 and owners[g] == context.rank),
            count=len(test_sources), split=split,
        )
    return SimpleNamespace(
        config=local_config, sources=selected, source=local_scene(source, partition),
        partition=partition, partitions=[p.manifest() for p in partitions],
        scene_manifest=scene_manifest, trajectory_id=_bank_id(sources), test=test,
    )


def create_runtime(args, config, context, data):
    """CLI owns simulator lifetime; training modules only receive an environment."""
    n = args.num_envs // context.world_size
    if args.synthetic_smoke:
        from ..utils.synthetic import LinearRecoveryBackend, synthetic_action_spec, synthetic_reference_trajectory
        trajectory = synthetic_reference_trajectory(frame_count=130, device=context.device)
        return SimpleNamespace(
            trajectory=trajectory, action_spec=synthetic_action_spec(),
            backend=LinearRecoveryBackend(num_envs=n, device=context.device),
            browser=None, app=None,
        )
    from ..environment.scene import (
        launch_simulation_app, create_simulation_context, create_scene, apply_scene_physics,
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if visible and all(value.startswith("GPU-") for value in visible):
        actual = str(torch.cuda.get_device_properties(context.device).uuid)
        expected = visible[context.device.index]
        if actual.removeprefix("GPU-").lower() != expected.removeprefix("GPU-").lower():
            raise RuntimeError("CUDA device UUID differs from the exclusive launch allocation")
    native_log = REPOSITORY_ROOT / "tmp" / f"physics_native_{os.getpid()}_rank_{context.rank}.log"
    os.environ["ISAACSIM_KIT_ARGS"] = os.environ.get("ISAACSIM_KIT_ARGS", "") + (
        f" --/log/file={native_log}"
        f" --/renderer/activeGpu={os.environ.get('TRAJTRACKING_PHYSICAL_GPUS', str(context.device.index)).split(',')[context.rank]}"
        " --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false"
    )
    started = time.perf_counter()
    launcher = launch_simulation_app(headless=args.headless, enable_cameras=False,
                                     device=str(context.device))
    try:
        from ..environment.core import IsaacTrajectoryBackend, action_spec_from_isaac_robot
        from trajectory_tracking.visualization.viewers.trajectory_tracking import TrajectoryTrackingBrowser
        from ..environment.physics_health import PhysicsHealthGuard
        health = PhysicsHealthGuard(REPOSITORY_ROOT / "tmp" / f"physics_failure_{os.getpid()}_rank_{context.rank}.json")
        capacity = None if args.gpu_collision_stack_mib is None else args.gpu_collision_stack_mib * 1024**2
        sim = create_simulation_context(device=str(context.device), physics=data.source.spec.physics,
                                        gpu_collision_stack_size_bytes=capacity)
        scene = create_scene(data.source, num_envs=n, env_spacing=args.env_spacing_m,
                             runtime=config.scene, build_mode=args.scene_build_mode)
        health.watch(native_log)
        health.check()
        sim.reset()
        health.check()
        apply_scene_physics(scene, data.source)
        scene_ready = time.perf_counter()
        startup_memory = None
        if args.memory_telemetry:
            from ..training.memory_telemetry import MemoryTelemetry
            startup_memory = MemoryTelemetry(REPOSITORY_ROOT / "tmp" / f"memory_startup_{os.getpid()}_rank_{context.rank}.jsonl", context.device)
            startup_memory("scene_ready", 0)
        from ..data.host_reference import load_reference_collection
        trajectory = load_reference_collection(
            data.config, project_root=REPOSITORY_ROOT, device=context.device, sources=data.sources,
            storage=args.reference_storage)
        data_ready = time.perf_counter()
        if startup_memory is not None:
            startup_memory("data_bank_ready", 0)
        action_spec = action_spec_from_isaac_robot(scene["robot"], canonical_joint_names=trajectory.joint_names)
        backend = IsaacTrajectoryBackend(
            simulation_context=sim, scene=scene, robot=scene["robot"], table=scene["table"],
            object_asset=scene["object"], trajectory=trajectory, action_spec=action_spec,
            physics_dt=data.source.spec.physics.dt, object_physics_mode=config.scene.object_physics_mode,
            render=not args.headless, health_check=health.check)
        browser = None
        if args.browser_enabled and context.primary:
            browser = TrajectoryTrackingBrowser(
                scene=scene, source=data.source, trajectory=trajectory, num_envs=n,
                host=args.browser_host, port=args.browser_port, env_index=args.browser_env_index,
                publish_every_steps=args.browser_publish_every_steps,
                object_flow_enabled=args.browser_object_flow_enabled)
        return SimpleNamespace(trajectory=trajectory, action_spec=action_spec, backend=backend,
                               browser=browser, app=launcher.app,
                               timing={"scene_initialization_seconds": scene_ready - started,
                                       "materialization_seconds": data_ready - scene_ready})
    except BaseException:
        _close_simulator_process(launcher.app, 1)


def heldout_callback(runtime, data, trainer, context, run_dir, observation_callback=None, *,
                     reference_storage="execution"):
    """Streaming batches reuse resident asset slots, retaining only episode rows."""
    from ..training.evaluation import evaluate_independent_metrics
    from ..training.heldout import test_summary
    test = data.test
    from ..data.host_reference import load_reference_collection
    bank = load_reference_collection(
        test.config, project_root=REPOSITORY_ROOT, device=context.device, sources=test.sources,
        storage=reference_storage)
    from ..data.heldout_terminal import apply_heldout_terminal
    bank, heldout_protocol, terminal_metadata = apply_heldout_terminal(
        test.config, test.sources, bank)
    n = len(data.partition.global_environment_ids)
    env = TrajectoryDirectEnvCore(
        config=test.config, trajectory=bank, action_spec=runtime.action_spec,
        backend=runtime.backend, num_envs=n, episode_completion_mode="reference_endpoint_only")
    uids = (tuple(source_object_uid(item) for item in test.sources)
            if data.partition.object_uids else (None,) * len(test.sources))
    environment_uids = data.partition.object_uids or (uids[0],) * n
    callback = observation_callback or (None if runtime.browser is None else runtime.browser.observe)
    evaluation_number = 0

    def evaluate(nominal=False):
        nonlocal evaluation_number
        rows = []
        for indices, scored in evaluation_batches(environment_uids, uids, test.owned_indices):
            diagnostics = []
            evaluate_independent_metrics(
                env, (lambda observation: env.normalized_nominal_action()) if nominal else trainer.deterministic_action,
                start_indices=torch.zeros(n, dtype=torch.long, device=context.device),
                trajectory_indices=torch.tensor(indices, device=context.device),
                observation_callback=callback, diagnostics_callback=diagnostics.extend)
            rows.extend(replace(diagnostics[slot], trajectory_index=test.global_indices[indices[slot]])
                        for slot in scored)
        all_rows = sorted((row for shard in context.gather(rows) for row in shard),
                          key=lambda row: row.trajectory_index)
        summary = test_summary(all_rows, test.count, require_exact_count=True)
        if context.primary:
            _atomic_json(run_dir / "heldout_test" / f"evaluation_{evaluation_number:06d}.json",
                         {"protocol": heldout_protocol, **terminal_metadata,
                          "policy": "nominal" if nominal else "learned",
                          "episodes": [asdict(row) for row in all_rows], **summary})
        evaluation_number += 1
        return dict(summary, event="evaluation" if nominal else "test_evaluation", dataset_split="test",
                    mode="A_no_offset_nominal_replay" if nominal else "C_learned_policy",
                    mean_flow_error_m=summary["object_flow_error_m"],
                    protocol=heldout_protocol,
                    terminal_phase_excluded=terminal_metadata.get("terminal_phase_excluded"),
                    terminal_phase_id=terminal_metadata.get("terminal_phase_id"),
                    used_for_early_stop=False, selection_sha256=test.config.data.selection_sha256)
    return evaluate


def _configure_rank_runtime_paths(repository_root=REPOSITORY_ROOT):
    """Override inherited shared Kit roots before simulator startup, per rank."""
    rank = int(os.environ.get("RANK", "0"))
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    cache = Path(repository_root) / "tmp" / "parallel_runtime_cache" / f"rank_{rank}"
    tokens = iter(shlex.split(os.environ.get("ISAACSIM_KIT_ARGS", "")))
    flags = []
    for token in tokens:
        if token == "--portable-root":
            value = next(tokens, None)
            if value is None or value.startswith("--"):
                raise ValueError("--portable-root requires a path")
        elif not token.startswith("--portable-root="):
            flags.append(token)
    kit = cache / "kit"
    kit.mkdir(parents=True, exist_ok=True)
    if "--portable" not in flags:
        flags.append("--portable")
    os.environ["ISAACSIM_USER_ROOT"] = str(kit)
    os.environ["ISAACSIM_KIT_ARGS"] = shlex.join(flags + ["--portable-root", str(kit)])
    for variable, name in (("OPTIX_CACHE_PATH", "optix"), ("CUDA_CACHE_PATH", "cuda"),
                           ("XDG_CACHE_HOME", "xdg"), ("XDG_DATA_HOME", "data"),
                           ("__GL_SHADER_DISK_CACHE_PATH", "gl_shader")):
        path = cache / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(Path(repository_root) / "tmp/ms-playwright"))


def main():
    args = _arguments()
    if int(os.environ.get("WORLD_SIZE", "1")) != args.expected_world_size:
        raise ValueError("torchrun world size differs from the requested fixed world size")
    if int(os.environ.get("LOCAL_WORLD_SIZE", str(args.expected_world_size))) != args.expected_world_size:
        raise ValueError("this launcher supports a single shared-filesystem node")
    # Reject missing/invalid producer evidence before creating a CUDA context.
    config = load_trajectory_direct_config(args.config, project_root=REPOSITORY_ROOT)
    if not args.synthetic_smoke:
        from .launch import validate_baseline
        validate_baseline(config)
    source_admission = producer_admission(config, REPOSITORY_ROOT,
        enabled=args.source_label_admission, synthetic=args.synthetic_smoke)
    _configure_rank_runtime_paths()
    indices = (None if args.cuda_device_indices is None else
               tuple(int(i) for i in args.cuda_device_indices.split(",")))
    context = ParallelContext.from_torchrun(cpu=args.synthetic_smoke,
                                           timeout_seconds=args.parallel_timeout_seconds,
                                           cuda_device_indices=indices)
    runtime = tracker = None
    writer_fd = None
    try:
        print(f"[rank {context.rank}] assigned {context.device}; validating configuration on CPU", flush=True)
        preparation_started = time.perf_counter()
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
        config.joint_acceleration.require_ready()
        context.agree(config.compatibility_hash, name="global configuration before admission")
        context.agree(args.source_label_admission, name="producer-success replay waiver")
        context.agree(source_admission, name="verified producer-success admission proof")
        print(f"[rank {context.rank}] startup Gloo ready; rank0 verifying complete dataset hashes", flush=True)
        if config.start_index.trajectory_assignment == "fixed_cyclic":
            raise ValueError("distributed v1 supports independent trajectory sampling, not fixed_cyclic")
        if config.start_index.mode == "object_lift_phase_weighted":
            raise ValueError("phase-weighted C-evaluation recipe needs a separate distributed contract")
        if args.synthetic_smoke:
            config = replace(config, experiment_plugin=None,
                             data=replace(config.data, trajectory_count=1),
                             ppo=replace(config.ppo, hidden_dims=(16, 16),
                                         rollout_steps_per_env=4, update_epochs=2, num_minibatches=4))
            data = SimpleNamespace(config=config, scene_manifest={"backend": "synthetic_not_physics"},
                                   trajectory_id="synthetic-single-trajectory", partitions=[], test=None)
        else:
            # Every shard/asset is checked, but not read and hashed eight times.
            # A failing rank0 never broadcasts an admitted dataset to workers.
            admission = context.broadcast(admit_data(args, config) if context.primary else None)
            data = prepare_data(args, config, context, admission=admission)
        prepared = time.perf_counter()
        print(f"[rank {context.rank}] data admitted; constructing local simulation", flush=True)
        torch.manual_seed(config.seed)
        runtime = create_runtime(args, config, context, data)
        print(f"[rank {context.rank}] runtime ready; initializing training collectives", flush=True)
        context.initialize_training_collectives(timeout_seconds=args.parallel_timeout_seconds)
        device_binding = context.device_manifest()
        if not args.synthetic_smoke and os.environ.get("CUDA_VISIBLE_DEVICES"):
            expected_uuid = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[context.rank]
            if device_binding["uuid"].removeprefix("GPU-").lower() != expected_uuid.removeprefix("GPU-").lower():
                raise ValueError("actual CUDA UUID differs from the exclusive GPU plan")
            device_binding["physical_gpu"] = int(os.environ["TRAJTRACKING_PHYSICAL_GPUS"].split(",")[context.rank])
        print(f"[rank {context.rank}] training collectives ready; device={device_binding}", flush=True)
        startup_timings = context.gather(dict(getattr(runtime, "timing", {}),
                                              device_binding=device_binding,
                                              validation_seconds=prepared - preparation_started))
        n = args.num_envs // context.world_size
        make_env = lambda local_config, **kwargs: TrajectoryDirectEnvCore(
            config=local_config, trajectory=runtime.trajectory, action_spec=runtime.action_spec,
            backend=runtime.backend, num_envs=n, **kwargs)
        trainer = create_ppo_trainer(observation_dim=config.observation.feature_dim, action_dim=27,
                                    config=config.ppo, device=context.device, seed=config.seed)
        global_ids = (tuple(range(context.rank * n, (context.rank + 1) * n)) if args.synthetic_smoke
                      else data.partition.global_environment_ids)
        trainer.execution = PPOExecution(context, global_environment_ids=global_ids)
        context.synchronize_model(trainer.model)
        # Initial weights/shuffle are common; exploration and reset streams differ.
        trainer.sample_generator.manual_seed(config.seed + 1 + 1000003 * context.rank)
        env = make_env(data.config)
        env.start_sampler.generator.manual_seed(config.seed + 101 + 1000003 * context.rank)
        env.perturbation.generator.manual_seed(config.seed + 202 + 1000003 * context.rank)
        callback = None if runtime.browser is None else runtime.browser.observe
        runner = PPOOverfitRunner(env, trainer, observation_callback=callback)
        control = TrainingControlSpec(max_updates=args.max_updates, evaluation_interval=args.evaluation_interval,
                                     target_success_rate=args.target_success_rate, success_patience=args.success_patience,
                                     success_early_stop_enabled=False)
        contract = dict(version="synchronous_ppo_v1", world_size=context.world_size,
                        collective_layout="gloo_startup_then_nccl_tensors_v1",
                        sample_order="global_environment_id_recipe_order_v1",
                        global_num_envs=args.num_envs, config_hash=config.compatibility_hash,
                        action_hash=stable_json_hash(runtime.action_spec.to_manifest()),
                        scene_hash=stable_json_hash(data.scene_manifest), partitions=data.partitions,
                        trajectory_id=data.trajectory_id, seed=config.seed,
                        evaluation_gate=control.gate_manifest(),
                        scene_build_mode=args.scene_build_mode,
                        env_spacing_m=args.env_spacing_m,
                        adapter_version=interprior_trajectory_adapter_version(config),
                        synthetic_smoke=args.synthetic_smoke)
        if args.evaluation_data_split == "test":
            contract["evaluation_protocol"] = dict(dataset_split="test", start="local_zero",
                completion="reference_endpoint_only", periodic_interval=args.evaluation_interval)
        if source_admission is not None:
            contract["source_admission"] = source_admission
        context.agree(contract, name="resolved training contract")
        artifacts = RunArtifacts(args.output_root, args.experiment_name) if context.primary else None
        run_dir = Path(context.gather(str(artifacts.run_dir) if artifacts else None)[0])
        if artifacts is None:
            artifacts = SimpleNamespace(run_dir=run_dir)
        _atomic_json(run_dir / "ranks" / f"rank_{context.rank:03d}_resolved.json", {
            "device": str(context.device), "num_envs": n, "config": data.config.to_manifest(),
            "device_binding": device_binding,
            "partition": None if args.synthetic_smoke else data.partition.manifest(),
        })
        continuation = None
        video_owner = run_dir
        if context.primary:
            from ..training.run_continuation import continuation_identity, acquire_run_writer
            continuation = continuation_identity(args.resume_checkpoint, contract)
            video_owner = run_dir if continuation is None else Path(continuation["video_owner_run_dir"])
            writer_fd = acquire_run_writer(video_owner)
            artifacts.write_resolved_config(config)
            manifest = dict(task="trajectory_tracking", parallel=contract, source=repository_lineage(REPOSITORY_ROOT),
                            software={"torch": torch.__version__, "cuda": torch.version.cuda},
                            startup_timings_by_rank=startup_timings,
                            action_spec=runtime.action_spec.to_manifest(),
                            observation_spec=config.observation.to_manifest(), scene=data.scene_manifest,
                            data=config.data.to_manifest(), split=None if data.test is None else data.test.split,
                            adapter_version=interprior_trajectory_adapter_version(config),
                            base_checkpoint=None, base_policy="not_applicable_zero_base",
                            runtime=dict(global_num_envs=args.num_envs, local_num_envs=n,
                                         reference_storage=args.reference_storage,
                                         allocator_config=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                                         memory_telemetry=args.memory_telemetry,
                                         gpu_collision_stack_mib=args.gpu_collision_stack_mib,
                                         new_wandb_run_on_resume=args.new_wandb_run_on_resume,
                                         memory_min_free_mib=args.memory_min_free_mib,
                                         release_cuda_cache_before_evaluation=args.release_cuda_cache_before_evaluation,
                                         max_updates=args.max_updates, evaluation_interval=args.evaluation_interval,
                                         heldout_test_interval=args.heldout_test_interval,
                                         evaluation_data_split=args.evaluation_data_split,
                                         checkpoint_interval=args.checkpoint_interval,
                                         browser_enabled=args.browser_enabled, video=args.record_video or args.record_evaluation_video,
                                         video_duration_seconds=60 if args.record_evaluation_video else args.video_duration_seconds,
                                         evaluation_video=args.record_evaluation_video,
                                         video_retention_count=3,
                                         object_flow_visible=args.browser_object_flow_enabled,
                                         recording_clock="wall_clock_not_simulation_step_synchronized"))
            if source_admission is not None:
                manifest["source_admission"] = source_admission
                _atomic_json(run_dir / "source_admission.json", source_admission)
            resume_tracking = None
            if continuation is not None:
                manifest["continuation"] = continuation
                if args.wandb_enabled and args.wandb_mode == "online" and not args.new_wandb_run_on_resume:
                    from ..training.run_continuation import resume_tracking_identity
                    resume_tracking = resume_tracking_identity(
                        continuation, project=args.wandb_project, group=args.wandb_group)
            tracker = TrajectoryWandbMirror(run_dir=run_dir, spec=WandbTrackingSpec(
                enabled=args.wandb_enabled, mode=args.wandb_mode, project=args.wandb_project,
                group=args.wandb_group, run_name=args.wandb_run_name or args.experiment_name,
                resume_run_id=None if resume_tracking is None else resume_tracking["run_id"],
                entity=None if resume_tracking is None else resume_tracking["entity"],
                resume_next_history_step=0 if continuation is None or args.new_wandb_run_on_resume else continuation["next_history_step"]))
            manifest["wandb"] = dict(run_id=tracker.run_id, url=tracker.run_url, group=args.wandb_group,
                                     project=args.wandb_project, mode=args.wandb_mode)
            artifacts.write_manifest(manifest)
            tracker.bind_manifest(manifest)

        evaluation_recorder = None
        evaluation_scope = lambda **metadata: nullcontext()
        if args.record_evaluation_video and context.primary:
            from trajectory_tracking.visualization.evaluation_browser_recorder import EvaluationBrowserRecorder
            evaluation_recorder = EvaluationBrowserRecorder(
                url=runtime.browser.url, run_dir=video_owner,
                commit=manifest["source"]["commit"], tmp_root=REPOSITORY_ROOT / "tmp")
            evaluation_scope = partial(evaluation_recorder.scope, segment_run_dir=str(run_dir),
                parent_checkpoint=None if continuation is None else continuation["parent_checkpoint"])

        html_replay = None
        if args.evaluation_html_port is not None and context.primary:
            from trajectory_tracking.visualization.viewers.evaluation_state_replay import EvaluationStateReplay
            if evaluation_recorder is None:
                raise ValueError("HTML capture requires evaluation video scope")
            html_replay = EvaluationStateReplay(source=data.source, run_dir=video_owner,
                commit=manifest["source"]["commit"], port=args.evaluation_html_port,
                event=evaluation_recorder.event, experiment_name=args.experiment_name,
                test_sources=() if data.test is None else data.test.sources,
                test_indices=() if data.test is None else data.test.global_indices)
            prior_scope = evaluation_scope
            @contextmanager
            def evaluation_scope(**metadata):
                with prior_scope(**metadata), html_replay.scope(**metadata):
                    yield

        def evaluation_observe(evaluation_env):
            if html_replay is not None:
                html_replay.observe(evaluation_env)
            if runtime.browser is not None:
                # The first callback is post-reset, on a distinct evaluation env.
                # Publish that actual state before opening a fresh browser page.
                runtime.browser.observe(evaluation_env)
                if evaluation_recorder is not None:
                    evaluation_recorder.frame_ready()

        def evaluate(local_config, nominal=False):
            evaluation_env = make_env(local_config)
            evaluation_env.start_sampler.generator.manual_seed(config.seed + 101 + 1000003 * context.rank)
            evaluation_env.perturbation.generator.manual_seed(config.seed + 202 + 1000003 * context.rank)
            policy = (lambda observation: evaluation_env.normalized_nominal_action()) if nominal else trainer.deterministic_action
            stats = evaluate_episode(evaluation_env, policy, observation_callback=evaluation_observe, metric_context=context)
            return dict(asdict(stats), event="evaluation", dataset_split="train",
                        mode="A_no_offset_nominal_replay" if nominal else "C_learned_policy")

        test_callback = (None if data.test is None else
                         heldout_callback(runtime, data, trainer, context, run_dir, evaluation_observe,
                                          reference_storage=args.reference_storage))
        memory_observer = None
        if args.memory_telemetry:
            from ..training.memory_telemetry import MemoryTelemetry
            memory_observer = MemoryTelemetry(run_dir / f"memory_rank_{context.rank}.jsonl", context.device, snapshot_update=args.memory_snapshot_update,
                min_free_bytes=args.memory_min_free_mib * 1024**2)
            memory_observer("coordinator_ready", 0)
        coordinator = SynchronousCoordinator(memory_observer=memory_observer,
            runner=runner, context=context, artifacts=artifacts, tracker=tracker, contract=contract,
            checkpoint_kwargs=dict(config=config, action_spec=runtime.action_spec,
                trajectory_id=data.trajectory_id, scene_manifest=data.scene_manifest,
                adapter_version=interprior_trajectory_adapter_version(config), num_envs=args.num_envs,
                control=control, consecutive_successful_evaluations=0),
            evaluate_train=partial(evaluate, data.config), evaluate_test=test_callback,
            periodic_heldout=args.evaluation_data_split == "test",
            evaluation_cache_cleanup=args.release_cuda_cache_before_evaluation,
            pause_recording=(nullcontext if args.record_evaluation_video else
                             partial(_pause_browser_recording_during_evaluation, runtime.browser)),
            evaluation_scope=evaluation_scope)
        _run_nominal_gate(args, data.config, coordinator, evaluation_scope, evaluate, test_callback,
                          source_admission)
        if args.evaluate_checkpoint is not None:
            if test_callback is None or args.resume_checkpoint is not None:
                raise ValueError("evaluation needs held-out data and cannot resume training")
            from ..training.artifacts import load_ppo_policy_checkpoint
            state = load_ppo_policy_checkpoint(args.evaluate_checkpoint, trainer=trainer, config=config,
                action_spec=runtime.action_spec, trajectory_id=data.trajectory_id,
                scene_manifest=data.scene_manifest, adapter_version=interprior_trajectory_adapter_version(config))
            with evaluation_scope(update=state.update, split="test", policy="learned"):
                result = test_callback()
            if context.primary:
                _atomic_json(run_dir / "evaluation_summary.json", dict(result, checkpoint_update=state.update))
            final_update = state.update
        else:
            final_update = coordinator.run(max_updates=args.max_updates, evaluation_interval=args.evaluation_interval,
                                          test_interval=args.heldout_test_interval, checkpoint_interval=args.checkpoint_interval,
                                          resume=args.resume_checkpoint)
        if context.primary:
            artifacts.write_summary({"status": "complete", "final_update": final_update, "parallel": contract})
        if tracker is not None:
            tracker.finish(exit_code=0)
        context.close()
        if runtime is not None:
            if runtime.browser is not None:
                runtime.browser.close()
            if runtime.app is not None:
                _close_simulator_process(runtime.app, 0)
        return 0
    except BaseException:
        _abort_worker(runtime, tracker)
    finally:
        if writer_fd is not None:
            os.close(writer_fd)


if __name__ == "__main__":
    raise SystemExit(main())
