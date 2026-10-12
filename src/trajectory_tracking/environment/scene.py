"""Task-local Isaac Lab scene for the independent trajectory experiment.

The module contains no reset distribution, action, reward, termination, PPO, or
trajectory cursor logic.  Isaac imports are lazy so CPU contract tests can
import the package without starting Kit.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Iterator
import xml.etree.ElementTree as ET

from .scene_config import (
    ResolvedTrajectorySceneSource,
    ScenePhysicsSpec,
    TrajectorySceneConfigError,
    TrajectorySceneRuntimeSpec,
)


DEFAULT_ENV_SPACING_M = 1.5
RUNTIME_ASSET_CONVERSION_VERSION = "isaacsim-5.1-gltf-scene-transform-bake-v4"


_FINGERTIP_BODY_NAMES = frozenset(
    name
    for finger in range(1, 6)
    for name in (
        f"right_finger{finger}_link4",
        f"right_finger{finger}_tip_link",
    )
)
_XARM_GLTF_VISUAL_LINKS = frozenset(
    ("link_base", *(f"link{index}" for index in range(1, 8)))
)
_XARM_VISUAL_COLLISION_BOUNDS_ATOL = 1.1e-3


def _project_tmp_root() -> Path:
    root = Path(__file__).resolve().parents[3] / "tmp" / "trajectory_residual"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _runtime_usd_scratch(asset_stem: str, asset_sha256: str) -> str:
    root = _project_tmp_root() / "usd" / RUNTIME_ASSET_CONVERSION_VERSION
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(
        prefix=f"{asset_stem}_{asset_sha256[:16]}_",
        dir=root,
    )


def _usd_safe_identifier(name: str) -> str:
    safe = "".join(char if (char.isalnum() or char == "_") else "_" for char in name)
    if not safe or not (safe[0].isalpha() or safe[0] == "_"):
        safe = f"mesh_{safe}"
    return safe


def _prepare_rigid_urdf_mesh_identifiers(
    asset_path: Path,
    work_dir: Path,
) -> Path:
    """Alias mesh stems that Isaac's URDF-to-USD importer cannot represent."""

    tree = ET.parse(asset_path)
    root = tree.getroot()
    aliases: dict[Path, Path] = {}
    changed = False
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        source = Path(filename)
        source = source if source.is_absolute() else (asset_path.parent / source)
        source = source.resolve()
        if not source.is_file():
            raise TrajectorySceneConfigError(f"rigid URDF mesh is missing: {source}")
        safe_stem = _usd_safe_identifier(source.stem)
        if safe_stem == source.stem:
            mesh.set("filename", str(source))
            continue
        changed = True
        alias = aliases.get(source)
        if alias is None:
            alias_dir = work_dir / "mesh_aliases"
            alias_dir.mkdir(parents=True, exist_ok=True)
            alias = alias_dir / f"{safe_stem}_{len(aliases):04d}{source.suffix}"
            shutil.copy2(source, alias)
            aliases[source] = alias
        mesh.set("filename", str(alias))
    if not changed:
        return asset_path
    output = work_dir / "prepared_urdf" / asset_path.name
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def _geometry_dependencies() -> tuple[Any, Any]:
    try:
        import numpy
        import trimesh
    except ImportError as error:
        raise TrajectorySceneConfigError(
            "xArm visual import requires the simulator asset dependencies; "
            "install this project with the 'simulator' extra in the Python "
            "environment used to launch Isaac Lab"
        ) from error
    return numpy, trimesh


def _scene_geometry_signature(scene: Any) -> list[tuple[Any, ...]]:
    """Describe topology and material assignment without relying on names."""

    signature = []
    for node_name in scene.graph.nodes_geometry:
        _, geometry_name = scene.graph[node_name]
        geometry = scene.geometry[geometry_name]
        visual = getattr(geometry, "visual", None)
        material = getattr(visual, "material", None)
        signature.append(
            (
                len(getattr(geometry, "vertices", ())),
                len(getattr(geometry, "faces", ())),
                type(visual).__name__ if visual is not None else None,
                type(material).__name__ if material is not None else None,
                getattr(material, "name", None),
            )
        )
    return sorted(signature, key=repr)


def _normalize_gltf_scene_transform(source_path: Path, target_path: Path) -> Any:
    """Bake each valid glTF scene transform into geometry exactly once."""

    numpy, trimesh = _geometry_dependencies()
    try:
        source_scene = trimesh.load(source_path, process=False, force="scene")
    except Exception as error:
        raise TrajectorySceneConfigError(
            f"cannot load source glTF visual: {source_path}: {error}"
        ) from error

    source_nodes = sorted(source_scene.graph.nodes_geometry)
    if not source_nodes:
        raise TrajectorySceneConfigError(f"glTF visual has no geometry: {source_path}")
    source_bounds = numpy.asarray(source_scene.bounds, dtype=numpy.float64)
    if source_bounds.shape != (2, 3) or not numpy.isfinite(source_bounds).all():
        raise TrajectorySceneConfigError(
            f"glTF visual has invalid scene bounds: {source_path}"
        )
    source_signature = _scene_geometry_signature(source_scene)

    normalized_scene = trimesh.Scene(base_frame="world")
    for index, node_name in enumerate(source_nodes):
        transform, geometry_name = source_scene.graph[node_name]
        geometry = source_scene.geometry[geometry_name].copy()
        geometry.apply_transform(transform)
        normalized_scene.add_geometry(
            geometry,
            geom_name=f"geometry_{index:02d}",
            node_name=f"geometry_{index:02d}",
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        exported = normalized_scene.export(file_type="glb")
        if not isinstance(exported, bytes):
            raise TypeError(f"expected GLB bytes, got {type(exported).__name__}")
        target_path.write_bytes(exported)
        verified_scene = trimesh.load(target_path, process=False, force="scene")
    except Exception as error:
        raise TrajectorySceneConfigError(
            f"cannot normalize glTF visual: {source_path}: {error}"
        ) from error

    verified_bounds = numpy.asarray(verified_scene.bounds, dtype=numpy.float64)
    if not numpy.allclose(source_bounds, verified_bounds, rtol=0.0, atol=1.0e-6):
        raise TrajectorySceneConfigError(
            "normalized glTF changed scene-space geometry bounds: "
            f"source={source_path} before={source_bounds.tolist()} "
            f"after={verified_bounds.tolist()}"
        )
    if _scene_geometry_signature(verified_scene) != source_signature:
        raise TrajectorySceneConfigError(
            f"normalized glTF changed geometry/material topology: {source_path}"
        )
    for node_name in verified_scene.graph.nodes_geometry:
        transform, _ = verified_scene.graph[node_name]
        if not numpy.allclose(
            transform,
            numpy.eye(4),
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise TrajectorySceneConfigError(
                "normalized glTF retained a non-identity scene transform: "
                f"source={source_path} node={node_name}"
            )
    return source_bounds


def _prepare_robot_urdf_for_isaacsim(
    robot_urdf: Path,
    output_dir: Path,
) -> Path:
    """Create a temporary URDF view with normalized xArm GLB visuals."""

    numpy, trimesh = _geometry_dependencies()
    robot_urdf = robot_urdf.resolve()
    output_dir = output_dir.resolve()
    tree = ET.parse(robot_urdf)
    root = tree.getroot()
    normalized_links: list[str] = []

    for link in root.findall("link"):
        link_name = link.get("name", "")
        gltf_visuals: list[tuple[ET.Element, Path]] = []
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            filename = mesh.get("filename", "") if mesh is not None else ""
            source_path = (robot_urdf.parent / filename).resolve()
            if mesh is not None and source_path.suffix.lower() in {".glb", ".gltf"}:
                gltf_visuals.append((mesh, source_path))
        if not gltf_visuals:
            continue
        if len(gltf_visuals) != 1:
            raise TrajectorySceneConfigError(
                f"expected one glTF visual for xArm link {link_name!r}"
            )

        collision_objs = []
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            filename = mesh.get("filename", "") if mesh is not None else ""
            path = (robot_urdf.parent / filename).resolve()
            if mesh is not None and path.suffix.lower() == ".obj":
                collision_objs.append(path)
        if len(collision_objs) != 1:
            raise TrajectorySceneConfigError(
                f"expected one OBJ collision for xArm link {link_name!r}"
            )

        mesh, source_path = gltf_visuals[0]
        target_path = output_dir / "visuals" / link_name / f"{source_path.stem}.glb"
        source_bounds = _normalize_gltf_scene_transform(source_path, target_path)
        try:
            collision_scene = trimesh.load(
                collision_objs[0],
                process=False,
                force="scene",
            )
        except Exception as error:
            raise TrajectorySceneConfigError(
                f"cannot load xArm collision mesh: {collision_objs[0]}: {error}"
            ) from error
        collision_bounds = numpy.asarray(collision_scene.bounds, dtype=numpy.float64)
        bounds_error = float(numpy.max(numpy.abs(source_bounds - collision_bounds)))
        if bounds_error > _XARM_VISUAL_COLLISION_BOUNDS_ATOL:
            raise TrajectorySceneConfigError(
                "xArm visual/collision frames do not align before conversion: "
                f"link={link_name} max_bounds_error={bounds_error:.9f}"
            )

        mesh.set("filename", str(target_path))
        normalized_links.append(link_name)

    if set(normalized_links) != _XARM_GLTF_VISUAL_LINKS:
        raise TrajectorySceneConfigError(
            "xArm glTF visual set changed: "
            f"expected={sorted(_XARM_GLTF_VISUAL_LINKS)} "
            f"actual={sorted(normalized_links)}"
        )

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        path = Path(filename)
        if not path.is_absolute():
            path = (robot_urdf.parent / path).resolve()
        if not path.is_file():
            raise TrajectorySceneConfigError(f"prepared robot mesh is missing: {path}")
        mesh.set("filename", str(path))

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_urdf = output_dir / robot_urdf.name
    ET.indent(tree, space="  ")
    tree.write(prepared_urdf, encoding="utf-8", xml_declaration=True)
    return prepared_urdf


@contextmanager
def _prepared_robot_urdf_for_isaacsim(robot_urdf: Path) -> Iterator[Path]:
    """Keep normalized inputs alive for the synchronous USD conversion."""

    with tempfile.TemporaryDirectory(
        prefix="trajectory_residual_xarm_import_",
        dir=_project_tmp_root(),
    ) as workdir:
        yield _prepare_robot_urdf_for_isaacsim(robot_urdf, Path(workdir))


def build_simulation_cfg(
    *,
    device: str,
    physics: ScenePhysicsSpec,
    gpu_collision_stack_size_bytes: int | None = None,
) -> Any:
    """Build the exact 120 Hz PhysX configuration after AppLauncher starts."""

    import isaaclab.sim as sim_utils

    capacity = {}
    if gpu_collision_stack_size_bytes is not None:
        if gpu_collision_stack_size_bytes <= 0:
            raise ValueError("collision stack capacity must be positive")
        capacity["gpu_collision_stack_size"] = gpu_collision_stack_size_bytes
    return sim_utils.SimulationCfg(
        dt=physics.dt,
        render_interval=2,
        device=device,
        gravity=physics.gravity,
        physx=sim_utils.PhysxCfg(
            **capacity,
            solver_type=physics.solver_type,
            enable_external_forces_every_iteration=(
                physics.enable_external_forces_every_iteration
            ),
            min_position_iteration_count=physics.min_position_iteration_count,
            max_position_iteration_count=physics.max_position_iteration_count,
            min_velocity_iteration_count=physics.min_velocity_iteration_count,
            max_velocity_iteration_count=physics.max_velocity_iteration_count,
            bounce_threshold_velocity=physics.bounce_threshold_velocity,
            friction_offset_threshold=physics.friction_offset_threshold,
            friction_correlation_distance=physics.friction_correlation_distance,
            gpu_max_rigid_contact_count=physics.gpu_max_rigid_contact_count,
            gpu_max_rigid_patch_count=physics.gpu_max_rigid_patch_count,
        ),
    )


def launch_simulation_app(*, headless: bool, enable_cameras: bool, device: str) -> Any:
    """Start Kit before any other simulator module is imported."""

    from isaaclab.app import AppLauncher

    return AppLauncher(
        headless=headless,
        enable_cameras=enable_cameras,
        device=device,
        multi_gpu=False,
    )


def create_simulation_context(*, device: str, physics: ScenePhysicsSpec,
                              gpu_collision_stack_size_bytes: int | None = None) -> Any:
    """Create the task simulation context after Kit has started."""

    from isaaclab.sim import SimulationContext

    return SimulationContext(build_simulation_cfg(device=device, physics=physics,
        gpu_collision_stack_size_bytes=gpu_collision_stack_size_bytes))


def _runtime_usd(
    asset_path: Path,
    *,
    asset_sha256: str,
    fix_base: bool,
    joint_drive: Any | None,
    self_collision: bool | None = None,
    identity_gltf_visual_links: frozenset[str] | None = None,
    link_density: float | None = None,
    collider_type: str | None = None,
    replace_cylinders_with_capsules: bool = False,
) -> str:
    """Convert a pinned URDF into a validated project-local runtime USD."""

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from pxr import Usd, UsdGeom

    usd_dir = Path(_runtime_usd_scratch(asset_path.stem, asset_sha256))
    conversion_asset_path = _prepare_rigid_urdf_mesh_identifiers(
        asset_path,
        usd_dir,
    )
    arguments: dict[str, Any] = {
        "asset_path": str(conversion_asset_path),
        "usd_dir": str(usd_dir),
        "force_usd_conversion": True,
        "fix_base": fix_base,
        "merge_fixed_joints": True,
        "make_instanceable": False,
        "joint_drive": joint_drive,
        "replace_cylinders_with_capsules": replace_cylinders_with_capsules,
    }
    if self_collision is not None:
        arguments["self_collision"] = self_collision
    if link_density is not None:
        if not math.isfinite(link_density) or link_density <= 0.0:
            raise TrajectorySceneConfigError("URDF link density must be positive")
        arguments["link_density"] = link_density
    if collider_type is not None:
        if collider_type not in ("convex_hull", "convex_decomposition"):
            raise TrajectorySceneConfigError("unsupported URDF collider type")
        arguments["collider_type"] = collider_type
    usd_path = UrdfConverter(UrdfConverterCfg(**arguments)).usd_path

    # The importer may retain instanceable referenced payloads even when
    # make_instanceable is false. Fabric consumes the fully de-instanced stage.
    stage = Usd.Stage.Open(usd_path, Usd.Stage.LoadAll)
    if stage is None:
        raise TrajectorySceneConfigError(f"failed to open converted USD: {usd_path}")
    root = stage.GetDefaultPrim()
    if not (root and root.IsValid()):
        root = next(
            (prim for prim in stage.GetPseudoRoot().GetChildren() if prim.IsValid()),
            None,
        )
    if root is None:
        raise TrajectorySceneConfigError(f"converted USD has no root prim: {usd_path}")
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    stage.GetRootLayer().Save()

    expected_identity_links = identity_gltf_visual_links or frozenset()
    validated_identity_links: list[str] = []
    bounds_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
        False,
        True,
    )
    urdf_root = ET.parse(asset_path).getroot()
    for link in urdf_root.findall("link"):
        link_name = link.get("name", "")
        if link_name not in expected_identity_links:
            continue
        collision_meshes = []
        for collision in link.findall("collision"):
            collision_mesh = collision.find("geometry/mesh")
            collision_filename = (
                collision_mesh.get("filename", "") if collision_mesh is not None else ""
            )
            if Path(collision_filename).suffix.lower() == ".obj":
                collision_meshes.append(Path(collision_filename))
        if len(collision_meshes) != 1:
            raise TrajectorySceneConfigError(
                f"cannot resolve converted OBJ collision for {link_name!r}"
            )

        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            filename = mesh.get("filename", "") if mesh is not None else ""
            if Path(filename).suffix.lower() not in {".glb", ".gltf"}:
                continue

            mesh_stem = Path(filename).stem
            wrapper_path = (
                root.GetPath()
                .AppendChild(link_name)
                .AppendChild("visuals")
                .AppendChild(mesh_stem)
            )
            wrapper = stage.GetPrimAtPath(wrapper_path)
            if not wrapper.IsValid():
                raise TrajectorySceneConfigError(
                    f"cannot resolve converted GLB visual: {wrapper_path}"
                )

            for prim in Usd.PrimRange(wrapper):
                xform = UsdGeom.Xformable(prim)
                if not xform:
                    continue
                for op in xform.GetOrderedXformOps():
                    matrix = op.GetOpTransform(Usd.TimeCode.Default())
                    identity_error = max(
                        abs(
                            float(matrix[row][column]) - (1.0 if row == column else 0.0)
                        )
                        for row in range(4)
                        for column in range(4)
                    )
                    if identity_error > 1.0e-6:
                        raise TrajectorySceneConfigError(
                            "converted normalized GLB has a non-identity "
                            "scene transform: "
                            f"path={prim.GetPath()} op={op.GetOpName()} "
                            f"max_error={identity_error:.9f}"
                        )

            link_prim = stage.GetPrimAtPath(root.GetPath().AppendChild(link_name))
            collision_wrapper_path = (
                link_prim.GetPath()
                .AppendChild("collisions")
                .AppendChild(collision_meshes[0].stem)
            )
            collision_wrapper = stage.GetPrimAtPath(collision_wrapper_path)
            if not collision_wrapper.IsValid():
                raise TrajectorySceneConfigError(
                    f"cannot resolve converted OBJ collision: {collision_wrapper_path}"
                )
            visual_range = bounds_cache.ComputeRelativeBound(
                wrapper,
                link_prim,
            ).ComputeAlignedRange()
            collision_range = bounds_cache.ComputeRelativeBound(
                collision_wrapper,
                link_prim,
            ).ComputeAlignedRange()
            visual_bounds = (
                *(float(value) for value in visual_range.GetMin()),
                *(float(value) for value in visual_range.GetMax()),
            )
            collision_bounds = (
                *(float(value) for value in collision_range.GetMin()),
                *(float(value) for value in collision_range.GetMax()),
            )
            if not all(
                math.isfinite(value) for value in (*visual_bounds, *collision_bounds)
            ):
                raise TrajectorySceneConfigError(
                    f"converted xArm bounds are non-finite: link={link_name}"
                )
            bounds_error = max(
                abs(visual - collision)
                for visual, collision in zip(
                    visual_bounds,
                    collision_bounds,
                    strict=True,
                )
            )
            if bounds_error > _XARM_VISUAL_COLLISION_BOUNDS_ATOL:
                raise TrajectorySceneConfigError(
                    "converted xArm visual/collision frames do not align: "
                    f"link={link_name} max_bounds_error={bounds_error:.9f}"
                )
            validated_identity_links.append(link_name)

    if set(validated_identity_links) != expected_identity_links:
        raise TrajectorySceneConfigError(
            "converted normalized GLB visual set changed: "
            f"expected={sorted(expected_identity_links)} "
            f"actual={sorted(validated_identity_links)}"
        )

    verified_stage = Usd.Stage.Open(usd_path, Usd.Stage.LoadAll)
    verified_root = verified_stage.GetDefaultPrim() if verified_stage else None
    remaining_instances = (
        [
            str(prim.GetPath())
            for prim in Usd.PrimRange(
                verified_root,
                Usd.TraverseInstanceProxies(),
            )
            if prim.IsInstance()
        ]
        if verified_root and verified_root.IsValid()
        else ["<missing-root>"]
    )
    if remaining_instances:
        raise TrajectorySceneConfigError(
            "converted USD still contains instanceable payloads after bake: "
            f"{remaining_instances[:8]}"
        )
    return usd_path


def _apply_wuji_palm_self_collision_filters(usd_path: str) -> None:
    """Exclude only palm/link2 overlaps; all other nonadjacent pairs stay on."""
    from pxr import Usd, UsdPhysics

    raw = Path(usd_path)
    physics = raw.parent / "configuration" / f"{raw.stem}_physics.usd"
    stage = Usd.Stage.Open(str(physics if physics.is_file() else raw), Usd.Stage.LoadAll)
    if stage is None:
        raise TrajectorySceneConfigError("cannot open robot collision layer")
    required = {"link7", *(f"right_finger{i}_link2" for i in range(2, 6))}
    bodies = {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() in required:
            if prim.GetName() in bodies:
                raise TrajectorySceneConfigError("duplicate Wuji collision body")
            bodies[prim.GetName()] = prim
    if required != set(bodies):
        raise TrajectorySceneConfigError(f"missing Wuji collision bodies: {required - set(bodies)}")
    relation = UsdPhysics.FilteredPairsAPI.Apply(bodies["link7"]).CreateFilteredPairsRel()
    for name in sorted(required - {"link7"}):
        relation.AddTarget(bodies[name].GetPath())
    stage.GetRootLayer().Save()


def _robot_cfg(
    source: ResolvedTrajectorySceneSource,
    *,
    runtime: TrajectorySceneRuntimeSpec,
) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    arm_stiffness = {
        **{f"joint{index}": 400.0 for index in range(1, 5)},
        "joint5": 150.0,
        "joint6": 150.0,
        "joint7": 15.0,
    }
    arm_damping = {
        **{f"joint{index}": 10.0 for index in range(1, 5)},
        **{f"joint{index}": 5.0 for index in range(5, 8)},
    }
    hand_stiffness = {
        f"right_finger{finger}_joint{joint}": (
            2.0 if joint <= 2 else 1.0 if joint == 3 else 0.8
        )
        for finger in range(1, 6)
        for joint in range(1, 5)
    }
    hand_damping = {
        name: (0.05 if name.endswith(("joint1", "joint2")) else 0.03)
        for name in hand_stiffness
    }
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=0.0,
            damping=0.0,
        ),
    )
    with _prepared_robot_urdf_for_isaacsim(source.robot_urdf) as robot_urdf:
        robot_usd = _runtime_usd(
            robot_urdf,
            asset_sha256=source.spec.robot.sha256,
            fix_base=True,
            self_collision=source.spec.robot_self_collision_enabled,
            joint_drive=joint_drive,
            identity_gltf_visual_links=_XARM_GLTF_VISUAL_LINKS,
        )
    if source.spec.robot_self_collision_enabled:
        _apply_wuji_palm_self_collision_filters(robot_usd)
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot_usd,
            activate_contact_sensors=runtime.contact_sensors_enabled,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=source.spec.robot_self_collision_enabled,
                solver_position_iteration_count=(
                    source.spec.physics.robot_solver_position_iterations
                ),
                solver_velocity_iteration_count=(
                    source.spec.physics.robot_solver_velocity_iterations
                ),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.53),
            joint_pos={
                "joint1": 0.0,
                "joint2": -0.548838,
                "joint3": -0.000004,
                "joint4": 0.096461,
                "joint5": 0.0,
                "joint6": 0.643490,
                "joint7": 0.000005,
                r"right_finger[1-5]_joint[1-4]$": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[r"joint[1-7]$"],
                stiffness=arm_stiffness,
                damping=arm_damping,
                armature=None,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=[r"right_finger[1-5]_joint[1-4]$"],
                stiffness=hand_stiffness,
                damping=hand_damping,
                armature=0.0,
            ),
        },
    )


def _replace_table_cylinder_collision(usd_path: str) -> None:
    """Replace only the converted table collider, retaining its visual and pose."""
    from pxr import Usd, UsdGeom, UsdPhysics
    from .table_collision import circular_table_mesh

    stage = Usd.Stage.Open(usd_path)
    cylinders = [
        prim for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Cylinder) and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if len(cylinders) != 1:
        raise TrajectorySceneConfigError("flat table requires one cylindrical collider")
    prim = cylinders[0]
    cylinder = UsdGeom.Cylinder(prim)
    if cylinder.GetAxisAttr().Get() != "Z":
        raise TrajectorySceneConfigError("flat table requires a Z-axis cylinder")
    points, faces = circular_table_mesh(
        cylinder.GetRadiusAttr().Get(), cylinder.GetHeightAttr().Get()
    )
    transform = UsdGeom.Xformable(prim).GetLocalTransformation()
    mesh = UsdGeom.Mesh.Define(stage, str(prim.GetPath()) + "_flat_mesh_v1")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateVisibilityAttr("invisible")
    mesh.AddTransformOp().Set(transform)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    # Referenced USD prims cannot be removed with RemovePrim; deactivate explicitly.
    prim.SetActive(False)
    stage.GetRootLayer().Save()


def _rigid_cfg(
    source: ResolvedTrajectorySceneSource,
    *,
    kind: str,
    runtime: TrajectorySceneRuntimeSpec,
) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    if kind == "table":
        path = source.table_urdf
        digest = source.spec.table.sha256
        kinematic = True
        initial_position = (0.0, 0.0, 0.38)
        prim_name = "Table"
    elif kind == "object":
        path = source.object_urdf
        digest = source.object_sha256
        kinematic = runtime.object_physics_mode == "disabled"
        initial_position = (0.45, 0.0, 0.59)
        prim_name = "Object"
    else:  # pragma: no cover - private call sites are closed.
        raise ValueError(f"unknown rigid asset kind: {kind}")
    usd_path = _runtime_usd(
        path,
        asset_sha256=digest,
        fix_base=False,
        joint_drive=None,
        link_density=(
            source.object_override.missing_inertial_density_kg_m3
            if kind == "object" and source.object_override is not None
            else None
        ),
        collider_type=(
            source.object_override.collider_type
            if kind == "object" and source.object_override is not None
            else None
        ),
        replace_cylinders_with_capsules=(
            source.object_override.replace_cylinders_with_capsules
            if kind == "object" and source.object_override is not None
            else False
        ),
    )
    if kind == "table" and runtime.table_collision == "flat_mesh_v1":
        _replace_table_cylinder_collision(usd_path)
    spawn_arguments: dict[str, Any] = {
        "usd_path": usd_path,
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=kinematic,
            disable_gravity=kinematic,
            max_depenetration_velocity=1000.0,
        ),
    }
    if kind == "object" and runtime.object_physics_mode == "disabled":
        spawn_arguments["collision_props"] = sim_utils.CollisionPropertiesCfg(
            collision_enabled=False
        )
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.UsdFileCfg(**spawn_arguments),
        init_state=RigidObjectCfg.InitialStateCfg(pos=initial_position),
    )


def _spawn_object_layout(prim_path, cfg, translation=None, orientation=None, *, layout, **kwargs):
    """Batch-spawn each asset into its deterministic slots using native spawners."""
    import re
    import isaaclab.sim as sim_utils

    root_pattern, name = prim_path.rsplit("/", 1)
    roots = sim_utils.find_matching_prim_paths(root_pattern)
    indexed = {int(path.rsplit("_", 1)[1]): path for path in roots}
    if set(indexed) != set(range(len(layout))):
        raise ValueError("object layout does not match the scene environment namespaces")
    for variant, asset in enumerate(cfg.assets_cfg):
        selected = [indexed[i] for i, value in enumerate(layout) if value == variant]
        if not selected:
            continue
        parent = selected[0].rsplit("/", 1)[0]
        if any(path.rsplit("/", 1)[0] != parent for path in selected):
            raise ValueError("object slots must share an environment namespace parent")
        expression = parent + "/(" + "|".join(re.escape(path.rsplit("/", 1)[1]) for path in selected) + ")/" + name
        asset.func(expression, asset, translation=translation, orientation=orientation, **kwargs)
    return sim_utils.get_current_stage().GetPrimAtPath(f"{indexed[0]}/{name}")


def build_scene_cfg(
    source: ResolvedTrajectorySceneSource,
    *,
    num_envs: int,
    env_spacing: float,
    runtime: TrajectorySceneRuntimeSpec = TrajectorySceneRuntimeSpec(),
) -> Any:
    """Build only assets and deterministic spawn state for the task scene."""

    if num_envs <= 0 or env_spacing <= 0.0:
        raise ValueError("num_envs and env_spacing must be positive")
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg

    cfg = InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        replicate_physics=False,
        clone_in_fabric=False,
    )
    cfg.robot = _robot_cfg(source, runtime=runtime)
    cfg.table = _rigid_cfg(source, kind="table", runtime=runtime)
    cfg.object = _rigid_cfg(source, kind="object", runtime=runtime)
    if source.object_variants:
        from functools import partial

        if not source.object_layout or any(
            index < 0 or index >= len(source.object_variants) for index in source.object_layout
        ):
            raise ValueError("invalid static scene object layout")
        layout = tuple(source.object_layout[i % len(source.object_layout)] for i in range(num_envs))
        cfg.object.spawn = sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[_rigid_cfg(item, kind="object", runtime=runtime).spawn
                        for item in source.object_variants],
            random_choice=False,
            func=partial(_spawn_object_layout, layout=layout),
        )
    if runtime.contact_sensors_enabled:
        from isaaclab.sensors import ContactSensorCfg

        for finger in range(1, 6):
            setattr(
                cfg,
                f"finger{finger}_object_contact",
                ContactSensorCfg(
                    prim_path=(
                        f"{{ENV_REGEX_NS}}/Robot/right_finger{finger}_link4"
                    ),
                    update_period=0.0,
                    history_length=1,
                ),
            )
    cfg.ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(usd_path=str(source.ground_usd)),
    )
    cfg.light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )
    return cfg


@contextmanager
def _batched_fabric_notices(enabled: bool) -> Iterator[None]:
    """Defer Fabric ingestion during USD authoring; restore before physics starts.

    Native Cloner observes the disabled handler and does not re-enable it after
    each asset group. No cloning, collision, asset, or physics setting changes.
    """
    if not enabled:
        yield
        return
    import isaaclab.sim as sim_utils
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import UsdUtils

    stage_id = UsdUtils.StageCache.Get().Insert(sim_utils.get_current_stage()).ToLongInt()
    was_enabled = SimulationManager.is_fabric_usd_notice_handler_enabled(stage_id)
    if was_enabled:
        SimulationManager.enable_fabric_usd_notice_handler(stage_id, False)
    try:
        yield
    finally:
        if was_enabled:
            started = time.monotonic()
            print("[scene build] restoring Fabric notices", flush=True)
            SimulationManager.enable_fabric_usd_notice_handler(stage_id, True)
            print(f"[scene build] Fabric restored in {time.monotonic() - started:.1f}s", flush=True)


def create_scene(
    source: ResolvedTrajectorySceneSource,
    *,
    num_envs: int,
    env_spacing: float = DEFAULT_ENV_SPACING_M,
    runtime: TrajectorySceneRuntimeSpec = TrajectorySceneRuntimeSpec(),
    build_mode: str = "legacy",
) -> Any:
    from isaaclab.scene import InteractiveScene

    if build_mode not in ("legacy", "batched_fabric_notices"):
        raise ValueError(f"unknown scene build mode: {build_mode}")
    started = time.monotonic()
    cfg = build_scene_cfg(
        source, num_envs=num_envs, env_spacing=env_spacing, runtime=runtime,
    )
    if build_mode != "legacy":
        print(f"[scene build] asset configuration ready in {time.monotonic() - started:.1f}s; "
              f"authoring {num_envs} environments", flush=True)
    with _batched_fabric_notices(build_mode == "batched_fabric_notices"):
        scene = InteractiveScene(cfg)
    if build_mode != "legacy":
        print(f"[scene build] completed in {time.monotonic() - started:.1f}s", flush=True)
    if source.object_variants:
        scene.object_uids = tuple(
            source.object_variants[source.object_layout[i % len(source.object_layout)]].object_override.object_uid
            for i in range(num_envs)
        )
    return scene


def _object_collider_mask(scene: Any, asset: Any) -> Any:
    """Select real shapes in the padded PhysX multi-object tensors."""
    import torch

    view = asset.root_physx_view
    counts_by_uid = {}
    counts = []
    for path in view.prim_paths:
        env_index = int(path.split("/env_", 1)[1].split("/", 1)[0])
        uid = scene.object_uids[env_index]
        if uid not in counts_by_uid:
            body = asset._physics_sim_view.create_rigid_body_view(path)
            counts_by_uid[uid] = body.max_shapes
        count = counts_by_uid[uid]
        if not 0 < count <= view.max_shapes:
            raise RuntimeError(f"invalid collider count for {uid}: {count}")
        counts.append(count)
    return torch.arange(view.max_shapes)[None, :] < torch.tensor(counts)[:, None]


def _check_collider_values(name: str, parameter: str, actual: Any,
                           expected: Any, valid_shapes: Any | None) -> None:
    import torch

    if valid_shapes is not None:
        actual, expected = actual[valid_shapes], expected[valid_shapes]
    if not torch.allclose(actual, expected):
        difference = (actual - expected).abs()
        raise RuntimeError(
            f"failed to apply {name} {parameter} on real collider shapes; "
            f"max_abs_error={difference.max().item():.6g}, "
            f"nonfinite={int((~torch.isfinite(actual)).sum())}"
        )


def apply_scene_physics(
    scene: Any,
    source: ResolvedTrajectorySceneSource,
) -> None:
    """Apply the recorded-scene collider material and offset parameters."""

    import torch

    physics = source.spec.physics
    env_ids = torch.arange(scene.num_envs, dtype=torch.int64, device="cpu")
    for name in ("robot", "table", "object"):
        asset = scene[name]
        view = asset.root_physx_view
        valid_shapes = (_object_collider_mask(scene, asset)
                        if name == "object" and hasattr(scene, "object_uids") else None)
        rest_offsets = view.get_rest_offsets().clone()
        rest_offsets.fill_(physics.rest_offset)
        view.set_rest_offsets(rest_offsets, env_ids)
        contact_offsets = view.get_contact_offsets().clone()
        contact_offsets.fill_(physics.contact_offset)
        view.set_contact_offsets(contact_offsets, env_ids)
        materials = view.get_material_properties().clone()
        materials[..., 0].fill_(physics.default_friction)
        materials[..., 1].fill_(physics.default_friction)
        materials[..., 2].fill_(physics.restitution)
        if name == "robot":
            shape_start = 0
            for body_name, body_path in zip(
                view.shared_metatype.link_names,
                view.link_paths[0],
                strict=True,
            ):
                body_view = asset._physics_sim_view.create_rigid_body_view(body_path)
                shape_end = shape_start + body_view.max_shapes
                if body_name in _FINGERTIP_BODY_NAMES:
                    materials[:, shape_start:shape_end, :2].fill_(
                        physics.fingertip_friction
                    )
                shape_start = shape_end
            if shape_start != view.max_shapes:
                raise RuntimeError("robot collider shape count changed")
        view.set_material_properties(materials, env_ids)
        for parameter, expected in (
            ("rest_offsets", rest_offsets),
            ("contact_offsets", contact_offsets),
            ("material_properties", materials),
        ):
            _check_collider_values(
                name, parameter, getattr(view, f"get_{parameter}")(), expected, valid_shapes,
            )
        if valid_shapes is not None:
            print(f"[scene physics] {name}: verified {int(valid_shapes.sum())} real shapes; "
                  f"excluded {int((~valid_shapes).sum())} padding slots", flush=True)



__all__ = [
    "DEFAULT_ENV_SPACING_M",
    "RUNTIME_ASSET_CONVERSION_VERSION",
    "apply_scene_physics",
    "build_scene_cfg",
    "build_simulation_cfg",
    "create_simulation_context",
    "create_scene",
    "launch_simulation_app",
]
