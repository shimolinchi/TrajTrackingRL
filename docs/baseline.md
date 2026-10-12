# Frozen baseline

Source: `shimolinchi/Interprior-rui`, commit `4434b2247429c392a3a5e19b97256f9b29dd3fb9`.
The source's validated delta baseline is the only release recipe. Data preparation and visualization
were moved into the installable package; machine-path environment keys use the TRAJTRACKING prefix.
The new repository has its own source/recipe identity; historical resume compatibility is not implied by identical architecture.

| Setting | Baseline |
|---|---|
| Robot | xArm7 + Wuji, arm7 + hand20 |
| Control / physics | 60 / 120 Hz |
| Arm action | target increment, 1.5 rad/s * 1/60 s |
| Hand action | normalized absolute joint target |
| Base/residual/EMA | none |
| Observation | 427D; actual q history -8/-4/-2/-1, current actual/reference q, future reference q +1/+2/+4/+8 |
| Original data | 3005 producer-success complete trajectories, 113 objects |
| Original split | 2402 train / 603 held-out, seed42; six singleton objects train-only |
| Default scale | 8192 environments per GPU; 8 GPUs = 65536 |
| PPO | rollout32, 5 epochs, 52 global minibatches; adaptive KL; initial actor/critic LR 2e-4 |
| Actor / critic MLP | 512,512,512,256,512,128,64; ELU |
| Robot / object tracking | 1 : 0.5 |
| Power weights | hand and wrist 0.635 |
| Reward scale | 0.064 |
| Target smoothness | first/second difference weights 4/1.5, computed on executed targets |
| Evaluation / checkpoint | update0 evaluation, every100 and final; no test-driven early stop |

The authoritative numeric recipe is `src/trajectory_tracking/resources/baseline.yaml`.
Training samples random valid suffix starts. Held-out evaluation starts at local frame zero and covers
the full reference endpoint. Complete success and random-suffix training success have different denominators.
The CoACD pre-placement evaluation protocol is not part of this baseline.
