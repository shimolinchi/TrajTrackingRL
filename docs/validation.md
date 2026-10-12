# Validation

The original v0.1.0 extraction was accepted on 2026-10-12. This verifies the extracted baseline's execution and contracts;
it does not establish convergence or new benchmark quality.

| Check | Result |
|---|---|
| Server CPU/contract suite | 244 passed; 29 unittest subtests passed |
| Clean GitHub install, CPU suite | 232 passed; two FFmpeg checks skipped on the runner |
| External scene/viewer asset checks | 10 passed on the server |
| Non-editable wheel | Packaged 427D recipe, all three public commands, HTML/JS resources and shared-mesh loading verified |
| Real state alignment | Frames 0 and 10: joint/object position errors zero; no action replay |
| Real distributed PPO | Two RTX PRO 5000 GPUs, 256 environments each; updates 1 and 2 completed |
| Exact continuation | Complete two-rank checkpoint restored, update 3 completed and saved |
| Fresh policy deployment | Update 3 policy loaded into a new one-GPU environment; full held-out evaluation completed without optimization |
| Browser recording | Deployment recording verified by ffprobe: 60.000 seconds |
| HTML playback | Actual recorded states, solid blue reference object, proper depth and 1x simulation-time playback checked in the browser |

The real acceptance dataset contains four complete producer-success trajectories from two objects:
two training and two held-out records, preserving their original partition membership. No trajectory
was shortened; lengths were 1704, 1676, 1675 and 1704 frames. Algorithm settings remained frozen.
Only data size, GPU/environment count and update budget were reduced for the acceptance run.
The freshly initialized policy had zero success on these two held-out records after three updates;
this is a run-through check, not a trained-policy delivery.

Runtime: Python 3.11.14, Torch 2.7.0+cu128, Isaac Sim 5.1.0.0, Isaac Lab 2.3.2.post1;
NCCL 2.28.9 was preloaded for Blackwell. Existing simulator dependencies were reused without
modifying their installation. CPU tests used a dedicated copy of dependency files on the node's
memory disk to avoid cold-import timeouts on the shared filesystem. Failure cleanup still has
an independently tested eight-second bound.

Artifacts remain outside Git in ignored logs and temporary directories. GCRL code, branches,
workspaces and training processes were not changed. The short GPU acceptance jobs have exited.

## Bundled assets (v0.1.1)

Both private submodules have been removed. Robot visual/collision meshes, table, cube,
the four required Grid files and 113 baseline object URDF/mesh pairs are now package resources.
Asset bytes and physical parameters are preserved; the upstream MIT notice is retained.

| Check | Result |
|---|---|
| Asset integrity/admission checks | 5 passed, including all manifests, URDF mesh closure and rejection of changed geometry |
| Real Isaac source-state restoration | Two objects at frames 0 and 10; joint/object position errors zero, no replay actions |
| Training entry preflight | Passed with the bundled roots and a newly frozen recipe/contract |
| Non-editable wheel | All final source/asset bytes included; 307 asset hashes verified in an extracted installation with no Git checkout |
| Existing HTML review | All 38 asset URLs match their original hashes after relocation |

The asset payload is approximately 76.5 MiB; the wheel is approximately 31 MiB.
Neither the source checkout nor the installed wheel needs access to the former private repositories.
Isaac software, trajectory datasets and checkpoints remain separate inputs.
