# Deterministic policy deployment

Evaluation loads model weights and normalization buffers only, using the same configuration,
data, action, scene and observation contracts as training. It does not restore the optimizer or training episodes.
Use scripts/evaluate.sh with the training recipe and its policy.pt.

Policy output is **not an absolute arm angle**. Its first seven entries are normalized arm increments.
Integrate them at 60 Hz, initialize the previous target from actual reset joints, and clamp to joint limits.
The remaining twenty entries are normalized absolute hand targets. Do not apply a second integration or EMA.

Construct the 427D observation in the declared order, including four real-state history samples and reference states.
Reset the history per environment. Preserve joint names/order, limits, coordinate conventions and normalization.
Inference without the reference trajectory or with a different robot is not the same policy contract.

Only load trusted project checkpoints. This release does not provide a real-robot driver or claim hardware safety.
