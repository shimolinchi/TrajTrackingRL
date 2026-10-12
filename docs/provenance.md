# Provenance and external inputs

The baseline source is Interprior-rui commit 4434b2247429c392a3a5e19b97256f9b29dd3fb9.
This extraction retains its delta action, temporal observation, rewards, PPO and data-admission mathematics.
Changes concern package boundaries, resource locations, public entry points, UUID binding and installation.
Old experiment configurations and other learning tasks are not carried into the application.

Bundled geometry retains its original bytes and SHA-256 identity:
- Robot/table/cube geometry extracted from Interprior revision 32198186b64bcef98d73b5f372c8adf74b3bbaf0.
  Only the referenced URDF/mesh files are bundled; its MIT notice is retained in resources/scene/LICENSE.Interprior.txt.
- Collector revision 8515739c561e1db4ba6dbd9088091348c76f4281 is historical data provenance only.
  No collector source is imported, cloned or needed at runtime.
- Isaac Sim 5.1.0.0 and Isaac Lab 2.3.2.post1 are separately installed software.
- The 113 baseline object URDFs and meshes, their catalog and checksums are bundled under resources/objects/.
- The four required NVIDIA Grid scene files are bundled under resources/scene/Isaac/ (Isaac 5.1 asset origin).
- Trajectory data, trained policies and SDK extensions are not committed.
- Neither upstream repository remains a Git submodule or runtime dependency.

A minimal Isaac installation may try to obtain omni.kit.pip_archive over HTTP while enabling the URDF importer.
The optional setup_isaac_extensions.sh downloads the official SDK package over HTTPS, verifies its fixed SHA-256,
and extracts it into this workspace's ignored cache. It does not change an existing Isaac installation.
The archive is approximately 43.5 MB. Its own metadata/licensing remains in the extracted package.

Historical configuration/checkpoint identifiers retain `trajectory_residual` for compatibility.
They are format identifiers; the released policy uses direct delta actions and has no residual learner.

The two-GPU Blackwell acceptance used an isolated NCCL 2.28.9 preload with Torch 2.7.0.
Set `TRAJTRACKING_NCCL_LIBRARY` to an installed compatible library when the bundled Torch NCCL
is unsuitable for the GPU generation; the launcher records the actual preloaded runtime version.
