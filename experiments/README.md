# Experiments Directory

## Active Entry Points

- `_feature_bank.py`
- `_inference_auto.py`
- `_evaluate.py`
- `run_ablation_main.py`

## Shared Modules

- `dataset_layout.py`
- `paper_constants.py`
- `resize_policy.py`

## Legacy

- `legacy/v29/`
  Historical V29 pipeline snapshots.
- `legacy/sweeps/`
  Older quick-sweep and parallel-sweep runners kept only for traceability.

Rule:
- Root `experiments/` only keeps the current mainline and the paper-facing ablation entry.
- Older version scripts go under `legacy/` unless they are promoted back to the active pipeline.

## Run Modes

- Default:
  `python experiments/run_ablation_main.py`
- Force sequential:
  `FORCE_SEQUENTIAL=1 python experiments/run_ablation_main.py`
- Explicit parallel worker count:
  `PARALLEL_WORKERS=2 python experiments/run_ablation_main.py`
