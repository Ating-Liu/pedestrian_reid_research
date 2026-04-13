# Pedestrian Re-Identification Research Project

This workspace is a standalone pedestrian re-identification project for resume, interview, and experiment use. All active development should now happen here.

## Project Positioning

- Task: pedestrian re-identification on public benchmarks
- Baseline: `ResNet50 + BNNeck + CrossEntropy + Triplet Loss`
- Method line: global-local dual branch, lightweight Transformer on local tokens, adaptive fusion gate
- Expected benchmarks: `Market-1501`, `CUHK03-NP`, `MSMT17`
- Outputs: training checkpoints, evaluation metrics, ranking visualizations, experiment tables

## Workspace Layout

- `reid/`: model, dataset pipeline, training engine, evaluation, utilities
- `scripts/train.py`: train one experiment
- `scripts/evaluate.py`: evaluate a saved checkpoint
- `scripts/visualize_rankings.py`: generate retrieval ranking figures
- `scripts/benchmark.py`: print or run the full ablation matrix
- `scripts/summarize_results.py`: build a Markdown experiment table from saved metrics
- `docs/dataset_setup.md`: expected dataset folder structure
- `docs/market1501_first_milestone.md`: exact first-stage execution checklist
- `docs/project_brief.md`: resume bullets and interview notes

## Environment

Install the new dependencies with:

```bash
py -3.12 -m pip install -r requirements-reid.txt
```

The implementation was verified against:

- Python 3.12
- PyTorch 2.5.1
- torchvision 0.20.1

On Windows, prefer `py -3.12` for every command. The default `python` on this machine may point to a different environment.

## Quick Start

Run a preflight check before the first training job:

```bash
py -3.12 scripts/preflight_check.py --dataset-name market1501 --run-tests
```

Inspect dataset structure:

```bash
py -3.12 scripts/inspect_dataset.py --data-root datasets --dataset-name market1501
```

Train the baseline on Market-1501 first:

```bash
py -3.12 scripts/train.py \
  --data-root datasets \
  --dataset-name market1501 \
  --experiment-name market1501_baseline \
  --use-local-branch false \
  --use-transformer false \
  --use-fusion-gate false
```

Train the full model on Market-1501 after the baseline is stable:

```bash
py -3.12 scripts/train.py \
  --data-root datasets \
  --dataset-name market1501 \
  --experiment-name market1501_full \
  --use-local-branch true \
  --use-transformer true \
  --use-fusion-gate true
```

Run the four ablations on all three datasets:

```bash
py -3.12 scripts/benchmark.py --data-root datasets
py -3.12 scripts/benchmark.py --data-root datasets --execute
```

Evaluate a checkpoint:

```bash
py -3.12 scripts/evaluate.py \
  --data-root datasets \
  --dataset-name market1501 \
  --checkpoint outputs/market1501/full_model/market1501_full/best_model.pth
```

Generate ranking visualizations:

```bash
py -3.12 scripts/visualize_rankings.py \
  --data-root datasets \
  --dataset-name market1501 \
  --checkpoint outputs/market1501/full_model/market1501_full/best_model.pth
```

Summarize metrics into a Markdown table:

```bash
py -3.12 scripts/summarize_results.py --output-root outputs
```

## Ablation Matrix

The recommended experiment matrix is:

1. `baseline`: global branch only
2. `local_branch`: global + local pooled tokens
3. `transformer_branch`: global + local branch + lightweight Transformer
4. `full_model`: transformer branch + adaptive fusion gate

Use the same training protocol on `Market-1501`, `CUHK03-NP`, and `MSMT17`, then compare `Rank-1`, `Rank-5`, `Rank-10`, and `mAP`.
