# Data

Only the TRO schema-v10 producer-success baseline adapter is exposed.
Required raw input includes `dataset_metadata.json`, `object_catalog.parquet`, and
`success_continuous/{commit.json,episodes.parquet,outcomes.parquet,flow_ranges.parquet,shards/}`.
Each shard needs its commit, original producer table, episode table and numeric frames.

A record must be marked success by the original producer and completed, training-eligible and written.
Preparation never relabels failed records. Invalid records either fail preparation or are explicitly excluded
with recorded reasons. Whole-trajectory grouped splitting prevents train/test identity leakage;
singleton objects stay train-only.

`scripts/prepare.sh` creates a fingerprinted immutable cache, selection, recipe and pending contract.
It does not train. Set `certify=true` only when its selected GPU is exclusively available; this performs
a small real source-state restore/read check and signs the current physics identity. It does not replay actions.
A changed source or physics implementation requires new evidence. Signed historical hashes are never rewritten to pass.

Provide the generated recipe and contract to `scripts/train.sh`.
For the original prepared 3005 baseline, use its original frozen selection and numeric cache files,
with the checksum declared by the recipe. Dataset and asset roots are configured through `configs/local.env`.
The baseline recipe's relative selection location is shown directly in its data section.
