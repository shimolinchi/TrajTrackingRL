# Architecture

```text
scripts/                     short user entry points
src/trajectory_tracking/
  resources/                 one baseline and physics-contract template
  workspace.py               configured workspace and packaged resource locations
  deployment.py              deterministic inference through the shared action projector
  configuration.py           typed configuration and compatibility identity
  data/                      source labels, immutable frames, identity, splits
    preparation/             raw conversion and cache creation
  environment/               scene/backend, actions, observations, rewards, reset
  training/                  one PPO implementation and run artifacts
    parallel/                partitioning, collectives and complete resume
  cli/                       preparation, preflight, launch and runtime composition
  visualization/             HTML, asset serving and browser recording
    viewers/                 robot/trajectory scene presentation
tests/                       numerical and contract checks
```

CLI is the composition boundary. Data, environment and learner do not depend on visualization.
Simulation state and targets are managed by the environment; PPO only consumes transitions.
Single-card and distributed training use the same PPO and action implementation.
A saved rollout includes joint-target integrator state and robot observation history.
No module is duplicated at the package root and inside a subdirectory.

Scene assets live in resources/scene/; all 113 baseline objects live in resources/objects/.
Relative URDF mesh references and original asset bytes are preserved. Geometry is shipped in wheels.
SHA-256 manifests validate the scene files. Collector revision fields are provenance, not Git dependencies.
