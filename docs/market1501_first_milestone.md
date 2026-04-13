# Market-1501 First Milestone

The first milestone is not "finish the whole project". It is to produce one credible `Market-1501` baseline result with logs, metrics, and retrieval visualizations.

## Success Criteria

- `scripts/preflight_check.py` passes under `py -3.12`
- `scripts/inspect_dataset.py` prints the expected Market-1501 counts
- the baseline training job finishes and writes checkpoints and metrics
- `scripts/evaluate.py` can re-evaluate the saved checkpoint
- `scripts/visualize_rankings.py` exports retrieval figures
- the result is recorded in `docs/experiment_log_template.md`

## Step 1: Environment

```bash
py -3.12 -m pip install -r requirements-reid.txt
py -3.12 scripts/preflight_check.py --dataset-name market1501 --run-tests
```

The preflight script should fail only if the dataset is still missing.

## Step 2: Dataset

Prepare the folder structure described in `docs/dataset_setup.md`.

Then validate it:

```bash
py -3.12 scripts/inspect_dataset.py --data-root datasets --dataset-name market1501
py -3.12 scripts/preflight_check.py --dataset-name market1501
```

## Step 3: Smoke Run

Use a short run first to catch data or training issues before the full experiment.

```bash
py -3.12 scripts/train.py \
  --data-root datasets \
  --dataset-name market1501 \
  --experiment-name market1501_baseline_smoke \
  --epochs 5 \
  --eval-period 5 \
  --use-local-branch false \
  --use-transformer false \
  --use-fusion-gate false
```

Check that the output directory contains:

- `config.json`
- `dataset_summary.json`
- `train_log.jsonl`
- `best_model.pth`
- `final_metrics.json`

## Step 4: Full Baseline

```bash
py -3.12 scripts/train.py \
  --data-root datasets \
  --dataset-name market1501 \
  --experiment-name market1501_baseline \
  --use-local-branch false \
  --use-transformer false \
  --use-fusion-gate false
```

## Step 5: Evaluation And Visualization

```bash
py -3.12 scripts/evaluate.py \
  --data-root datasets \
  --dataset-name market1501 \
  --checkpoint outputs/market1501/baseline/market1501_baseline/best_model.pth

py -3.12 scripts/visualize_rankings.py \
  --data-root datasets \
  --dataset-name market1501 \
  --checkpoint outputs/market1501/baseline/market1501_baseline/best_model.pth
```

## Step 6: Record The Result

Copy one section from `docs/experiment_log_template.md` and fill in:

- the exact command
- the final metrics
- whether training was stable
- what the successful retrievals and failures look like
- the next single change to test

Do not move to `local_branch` until the baseline result is recorded and you can explain it clearly.
