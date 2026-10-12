# TrajTrackingRL
This repository owns one baseline: delta arm targets and absolute hand targets.
Source provenance and frozen settings are documented in docs/baseline.md.

- Keep one implementation per responsibility. configuration.py parses settings; data/ owns source admission and splits; environment/ owns simulation and control; training/ owns PPO and checkpoints; cli/ composes them.
- Core data/environment/training modules must not import CLI or visualization code. Simulation belongs only to the scene/backend boundary. The environment never calls the learner.
- Preserve the 60 Hz action equation, reset integrator state, 427D observation order, reward mathematics and held-out protocol during structural changes.
- Add a focused numerical or integration check when those contracts change. Do not copy experiment-specific stacks.
- scripts/ contains short, no-argument Shell entry points with a user configuration block. Machine paths belong in ignored configs/local.env.
- Runtime files, datasets, model weights and caches belong in ignored tmp/, logs/ or data/. Never commit them.
- Evaluation uses genuine simulation state. HTML playback follows simulation timestamps at 1x. Record 60 seconds per evaluation and retain the latest three captures.
- GPU jobs use explicit UUID bindings and exclusive assigned GPUs. Do not stop other jobs or modify shared dependencies.

- Geometry is bundled in resources/scene/ and resources/objects/. Do not add private Git dependencies; preserve source notices, file hashes and URDF mesh closure when updating assets.
