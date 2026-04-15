# Experiment Log

## 2026-04-11 Market-1501 Baseline

### Metadata

- Dataset: `Market-1501`
- Variant: `baseline`
- Model: `ResNet50 + BNNeck`
- Loss: `CrossEntropy Loss + Batch-Hard Triplet Loss`
- Goal: establish a credible baseline before adding the local branch and Transformer modules.

### Environment

- Python: `3.12`
- PyTorch: `2.5.1+cu121`
- Torchvision: `0.20.1+cu121`
- GPU: `NVIDIA GeForce RTX 4080 Laptop GPU`
- Batch size: `64`
- DataLoader: `num_workers=12`, `prefetch_factor=2`, `persistent_workers=false`
- AMP: enabled
- Seed: `42`

### Command

```powershell
py -3.12 scripts\train.py --data-root datasets --dataset-name market1501 --experiment-name market1501_baseline --device cuda --use-amp true --num-workers 12 --prefetch-factor 2 --persistent-workers false --batch-size 64 --num-instances 4 --use-local-branch false --use-transformer false --use-fusion-gate false
```

### Outputs

- Output directory: `outputs/market1501/baseline/market1501_baseline`
- Best checkpoint: `outputs/market1501/baseline/market1501_baseline/best_model.pth`
- Training log: `outputs/market1501/baseline/market1501_baseline/train_log.jsonl`
- Final metrics: `outputs/market1501/baseline/market1501_baseline/final_metrics.json`
- Independent evaluation: `outputs/market1501/baseline/market1501_baseline/eval_metrics.json`
- Ranking visualization directory: `outputs/market1501/baseline/market1501_baseline/rankings`

### Metrics

- Rank-1: `93.32%`
- Rank-5: `97.77%`
- Rank-10: `98.60%`
- mAP: `82.93%`
- Best epoch by Rank-1: `60`

### Training Notes

- Training completed successfully in about `103 minutes`.
- Loss decreased from about `6.14` at epoch 1 to about `1.09` at epoch 60.
- Training accuracy saturated near `100%` after the learning-rate decay stage, which is expected for a closed-set identity classifier on the training split.
- Evaluation improved sharply after the first LR milestone around epoch 40, with Rank-1 moving from about `89%` to `93%`.

### Decision

- Keep this result as the main `Market-1501` baseline.
- Next experiment: `local_branch`, keeping the same training protocol and only enabling `--use-local-branch true --use-transformer false --use-fusion-gate false`.
- Reason: this isolates whether part-level pooled local tokens improve retrieval before introducing Transformer relation modeling.

## 2026-04-11 Market-1501 Local Branch

### Metadata

- Dataset: `Market-1501`
- Variant: `local_branch`
- Model: `ResNet50 + global branch + local pooled part tokens`
- Loss: `CrossEntropy Loss + Batch-Hard Triplet Loss`
- Goal: isolate whether adding part-level local tokens improves retrieval before introducing Transformer relation modeling.

### Environment

- Python: `3.12`
- PyTorch: `2.5.1+cu121`
- Torchvision: `0.20.1+cu121`
- GPU: `NVIDIA GeForce RTX 4080 Laptop GPU`
- Batch size: `64`
- DataLoader: `num_workers=12`, `prefetch_factor=2`, `persistent_workers=false`
- AMP: enabled
- Seed: `42`

### Command

```powershell
py -3.12 scripts\train.py --data-root datasets --dataset-name market1501 --experiment-name market1501_local_branch --device cuda --use-amp true --num-workers 12 --prefetch-factor 2 --persistent-workers false --batch-size 64 --num-instances 4 --use-local-branch true --use-transformer false --use-fusion-gate false
```

### Outputs

- Output directory: `outputs/market1501/local_branch/market1501_local_branch`
- Best checkpoint: `outputs/market1501/local_branch/market1501_local_branch/best_model.pth`
- Training log: `outputs/market1501/local_branch/market1501_local_branch/train_log.jsonl`
- Final metrics: `outputs/market1501/local_branch/market1501_local_branch/final_metrics.json`
- Independent evaluation: `outputs/market1501/local_branch/market1501_local_branch/eval_metrics.json`
- Ranking visualization directory: `outputs/market1501/local_branch/market1501_local_branch/rankings`

### Metrics

- Rank-1: `90.29%`
- Rank-5: `96.79%`
- Rank-10: `98.10%`
- mAP: `75.93%`
- Best epoch by Rank-1: `55`

### Comparison With Baseline

- Rank-1 changed from `93.32%` to `90.29%`, a drop of about `3.03` percentage points.
- mAP changed from `82.93%` to `75.93%`, a drop of about `7.00` percentage points.
- This means the current local branch implementation weakens the baseline rather than improving it.

### Training Notes

- Training completed successfully in about `129 minutes`.
- The model converged more slowly than the baseline and reached lower retrieval quality.
- The post-milestone improvement after epoch 40 was visible, but the final result still stayed below the baseline.

### Decision

- Do not treat the current local branch as a valid improvement.
- Before running the Transformer variant, inspect the local feature fusion design. The likely issue is that the current fusion projection replaces a strong global embedding with a newly initialized fused embedding, making optimization harder.
- Next recommended experiment: revise the local branch to keep the global descriptor as the dominant path, then add the local descriptor as a residual or auxiliary contribution instead of forcing a full projection replacement.

## 2026-04-11 Market-1501 Local Residual Fusion

### Metadata

- Dataset: `Market-1501`
- Variant: `local_branch`
- Experiment: `market1501_local_residual`
- Model: `ResNet50 + global branch + local pooled part tokens + residual local fusion`
- Loss: `CrossEntropy Loss + Batch-Hard Triplet Loss`
- Goal: test whether preserving the global descriptor as the main path fixes the degradation observed in the projection-based local branch.

### Method Change

- Added `--fusion-mode residual`.
- The final embedding is computed as `global_embedding + local_residual_scale * local_embedding`.
- `local_residual_scale` is learnable and initialized to `0.1`.
- The original projection-based fusion remains available as `--fusion-mode projection` for reproducibility.

### Environment

- Python: `3.12`
- PyTorch: `2.5.1+cu121`
- Torchvision: `0.20.1+cu121`
- GPU: `NVIDIA GeForce RTX 4080 Laptop GPU`
- Batch size: `64`
- DataLoader: `num_workers=12`, `prefetch_factor=2`, `persistent_workers=false`
- AMP: enabled
- Seed: `42`

### Command

```powershell
py -3.12 scripts\train.py --data-root datasets --dataset-name market1501 --experiment-name market1501_local_residual --device cuda --use-amp true --num-workers 12 --prefetch-factor 2 --persistent-workers false --batch-size 64 --num-instances 4 --use-local-branch true --use-transformer false --use-fusion-gate false --fusion-mode residual --local-residual-weight 0.1
```

### Outputs

- Output directory: `outputs/market1501/local_branch/market1501_local_residual`
- Best checkpoint: `outputs/market1501/local_branch/market1501_local_residual/best_model.pth`
- Training log: `outputs/market1501/local_branch/market1501_local_residual/train_log.jsonl`
- Final metrics: `outputs/market1501/local_branch/market1501_local_residual/final_metrics.json`
- Independent evaluation: `outputs/market1501/local_branch/market1501_local_residual/eval_metrics.json`
- Ranking visualization directory: `outputs/market1501/local_branch/market1501_local_residual/rankings`

### Metrics

- Rank-1: `93.41%`
- Rank-5: `97.92%`
- Rank-10: `98.93%`
- mAP: `82.85%`
- Best epoch by Rank-1: `60`

### Comparison

- Compared with baseline, Rank-1 changed from `93.32%` to `93.41%`, a small gain of about `0.09` percentage points.
- Compared with baseline, mAP changed from `82.93%` to `82.85%`, a small drop of about `0.08` percentage points.
- Compared with the projection-based local branch, Rank-1 recovered from `90.29%` to `93.41%`, and mAP recovered from `75.93%` to `82.85%`.

### Decision

- Keep residual fusion as the corrected local-branch implementation.
- Do not claim a strong improvement over baseline yet; the current conclusion is that residual fusion avoids the degradation caused by naive projection fusion.
- Next experiment: add the lightweight Transformer on top of the corrected residual local branch, keeping `--fusion-mode residual`.

## 2026-04-12 Market-1501 Transformer Residual

### Metadata

- Dataset: `Market-1501`
- Variant: `transformer_branch`
- Experiment: `market1501_transformer_residual`
- Model: `ResNet50 + global branch + local pooled part tokens + lightweight Transformer + residual local fusion`
- Loss: `CrossEntropy Loss + Batch-Hard Triplet Loss`
- Goal: test whether modeling relations among local part tokens improves retrieval beyond the corrected residual local branch.

### Method

- Local part tokens are extracted from the late CNN feature map.
- A lightweight Transformer encoder models relationships between the part tokens.
- The transformed local descriptor is fused through the residual path: `global_embedding + local_residual_scale * local_embedding`.
- Fusion gate is disabled, so this experiment isolates Transformer relation modeling.

### Environment

- Python: `3.12`
- PyTorch: `2.5.1+cu121`
- Torchvision: `0.20.1+cu121`
- GPU: `NVIDIA GeForce RTX 4080 Laptop GPU`
- Batch size: `64`
- DataLoader: `num_workers=12`, `prefetch_factor=2`, `persistent_workers=false`
- AMP: enabled
- Seed: `42`

### Command

```powershell
py -3.12 scripts\train.py --data-root datasets --dataset-name market1501 --experiment-name market1501_transformer_residual --device cuda --use-amp true --num-workers 12 --prefetch-factor 2 --persistent-workers false --batch-size 64 --num-instances 4 --use-local-branch true --use-transformer true --use-fusion-gate false --fusion-mode residual --local-residual-weight 0.1
```

### Outputs

- Output directory: `outputs/market1501/transformer_branch/market1501_transformer_residual`
- Best checkpoint: `outputs/market1501/transformer_branch/market1501_transformer_residual/best_model.pth`
- Training log: `outputs/market1501/transformer_branch/market1501_transformer_residual/train_log.jsonl`
- Final metrics: `outputs/market1501/transformer_branch/market1501_transformer_residual/final_metrics.json`
- Independent evaluation: `outputs/market1501/transformer_branch/market1501_transformer_residual/eval_metrics.json`
- Ranking visualization directory: `outputs/market1501/transformer_branch/market1501_transformer_residual/rankings`

### Metrics

- Rank-1: `93.17%`
- Rank-5: `97.95%`
- Rank-10: `98.60%`
- mAP: `82.79%`
- Best epoch by Rank-1: `60`

### Comparison

- Compared with baseline, Rank-1 changed from `93.32%` to `93.17%`, a small drop of about `0.15` percentage points.
- Compared with baseline, mAP changed from `82.93%` to `82.79%`, a small drop of about `0.14` percentage points.
- Compared with local residual fusion, Rank-1 changed from `93.41%` to `93.17%`, and mAP changed from `82.85%` to `82.79%`.

### Decision

- Do not claim that the current Transformer module improves Market-1501.
- The useful conclusion is that residual fusion stabilizes the local branch, while the current Transformer relation modeling does not add measurable benefit on Market-1501.
- Next recommended experiment: test an adaptive residual/gated local contribution, or move to CUHK03-NP/MSMT17 to see whether local relation modeling helps more under harder conditions.

## 2026-04-12 Market-1501 Gated Residual Full Model

### Metadata

- Dataset: `Market-1501`
- Variant: `full_model`
- Experiment: `market1501_gated_residual`
- Model: `ResNet50 + global branch + local pooled part tokens + lightweight Transformer + gated residual local fusion`
- Loss: `CrossEntropy Loss + Batch-Hard Triplet Loss`
- Goal: test whether an adaptive gate can control the contribution of Transformer-enhanced local features better than a fixed residual scale.

### Method

- Local part tokens are extracted from the late CNN feature map.
- A lightweight Transformer encoder models relationships between local part tokens.
- The local descriptor is fused through a gated residual path, keeping the global descriptor as the dominant retrieval feature.
- The residual local contribution is initialized with `--local-residual-weight 0.1`.

### Environment

- Python: `3.12`
- PyTorch: `2.5.1+cu121`
- Torchvision: `0.20.1+cu121`
- GPU: `NVIDIA GeForce RTX 4080 Laptop GPU`
- Batch size: `64`
- DataLoader: `num_workers=12`, `prefetch_factor=2`, `persistent_workers=false`
- AMP: enabled
- Seed: `42`

### Command

```powershell
py -3.12 scripts\train.py --data-root datasets --dataset-name market1501 --experiment-name market1501_gated_residual --device cuda --use-amp true --num-workers 12 --prefetch-factor 2 --persistent-workers false --batch-size 64 --num-instances 4 --use-local-branch true --use-transformer true --use-fusion-gate true --fusion-mode gated_residual --local-residual-weight 0.1
```

### Outputs

- Output directory: `outputs/market1501/full_model/market1501_gated_residual`
- Best checkpoint: `outputs/market1501/full_model/market1501_gated_residual/best_model.pth`
- Training log: `outputs/market1501/full_model/market1501_gated_residual/train_log.jsonl`
- Final metrics: `outputs/market1501/full_model/market1501_gated_residual/final_metrics.json`
- Independent evaluation: `outputs/market1501/full_model/market1501_gated_residual/eval_metrics.json`
- Ranking visualization directory: `outputs/market1501/full_model/market1501_gated_residual/rankings`

### Metrics

- Rank-1: `93.26%`
- Rank-5: `98.01%`
- Rank-10: `98.87%`
- mAP: `82.67%`
- Best epoch by Rank-1: `60`
- Independent evaluation on `best_model.pth`: Rank-1 `93.35%`, mAP `82.67%`

### Comparison

- Compared with baseline, Rank-1 changed from `93.32%` to `93.26%`, a small drop of about `0.06` percentage points.
- Compared with baseline, mAP changed from `82.93%` to `82.67%`, a drop of about `0.26` percentage points.
- Compared with local residual fusion, Rank-1 changed from `93.41%` to `93.26%`, and mAP changed from `82.85%` to `82.67%`.
- Compared with Transformer residual fusion, Rank-1 changed from `93.17%` to `93.26%`, but mAP changed from `82.79%` to `82.67%`.

### Decision

- Do not claim that the full gated Transformer model improves Market-1501.
- The best current Market-1501 result by Rank-1 is still `market1501_local_residual`; the best result by mAP is still the plain baseline.
- The current experimental story is credible but conservative: naive local projection hurts, residual local fusion fixes the degradation, and the current lightweight Transformer/gated fusion does not provide a measurable Market-1501 gain.
- Next recommended step: validate the corrected residual local branch and full model on a harder dataset such as `CUHK03-NP` or `MSMT17`, instead of over-tuning Market-1501.

## 2026-04-12 CUHK03-NP Dataset Preparation

### Metadata

- Dataset: `CUHK03-NP`
- Variant: `detected`
- Goal: prepare the second benchmark for cross-dataset validation.

### Inputs

- Archive: `datasets/archive.zip`
- Split file: `datasets/cuhk03_new_protocol_config_detected.mat`
- Source image folder inside archive: `images_detected`

### Command

```powershell
py -3.12 scripts\prepare_cuhk03_np.py --data-root datasets --variant detected --output-name cuhk03_np --jobs 24 --force
```

### Output Layout

- Output directory: `datasets/cuhk03_np`
- Train directory: `datasets/cuhk03_np/bounding_box_train`
- Query directory: `datasets/cuhk03_np/query`
- Gallery directory: `datasets/cuhk03_np/bounding_box_test`
- Manifest: `datasets/cuhk03_np/manifest.csv`
- Preparation summary: `datasets/cuhk03_np/prepare_summary.json`

### Dataset Check

```json
{
  "train_images": 7365,
  "query_images": 1400,
  "gallery_images": 5332,
  "train_ids": 767
}
```

### Decision

- Use `cuhk03_np` as the project's CUHK03-NP detected benchmark.
- The detected split is harder and more realistic than the labeled split, so it is suitable for showing cross-dataset robustness.

## 2026-04-12 CUHK03-NP Baseline

### Metadata

- Dataset: `CUHK03-NP`
- Variant: `baseline`
- Experiment: `cuhk03_np_baseline`
- Model: `ResNet50 + BNNeck`
- Loss: `CrossEntropy Loss + Batch-Hard Triplet Loss`
- Goal: establish a second-dataset baseline before testing the corrected local residual branch.

### Environment

- Python: `3.12`
- PyTorch: `2.5.1+cu121`
- Torchvision: `0.20.1+cu121`
- GPU: `NVIDIA GeForce RTX 4080 Laptop GPU`
- Batch size: `64`
- DataLoader: `num_workers=12`, `prefetch_factor=2`, `persistent_workers=false`
- AMP: enabled
- Seed: `42`

### Command

```powershell
py -3.12 scripts\train.py --data-root datasets --dataset-name cuhk03_np --experiment-name cuhk03_np_baseline --device cuda --use-amp true --num-workers 12 --prefetch-factor 2 --persistent-workers false --batch-size 64 --num-instances 4 --use-local-branch false --use-transformer false --use-fusion-gate false
```

### Outputs

- Output directory: `outputs/cuhk03_np/baseline/cuhk03_np_baseline`
- Best checkpoint: `outputs/cuhk03_np/baseline/cuhk03_np_baseline/best_model.pth`
- Training log: `outputs/cuhk03_np/baseline/cuhk03_np_baseline/train_log.jsonl`
- Final metrics: `outputs/cuhk03_np/baseline/cuhk03_np_baseline/final_metrics.json`
- Independent evaluation: `outputs/cuhk03_np/baseline/cuhk03_np_baseline/eval_metrics.json`
- Ranking visualization directory: `outputs/cuhk03_np/baseline/cuhk03_np_baseline/rankings`

### Metrics

- Rank-1: `61.43%`
- Rank-5: `80.36%`
- Rank-10: `87.00%`
- mAP: `59.52%`
- Best epoch by Rank-1: `60`
- Independent evaluation on `best_model.pth`: Rank-1 `61.43%`, mAP `59.52%`

### Training Notes

- Training completed successfully in about `72 minutes`.
- Training accuracy reached nearly `100%`, while retrieval performance remained much lower than Market-1501. This confirms CUHK03-NP detected is a harder validation benchmark.
- Evaluation improved after the learning-rate decay stage, with Rank-1 moving from about `50-54%` before epoch 40 to `61.43%` at epoch 60.

### Decision

- Keep this result as the CUHK03-NP detected baseline.
- Next experiment: run `cuhk03_np_local_residual` with the same training protocol and only enable the residual local branch.
- Reason: this tests whether the local-branch correction that stabilized Market-1501 also transfers to a harder benchmark.

## 2026-04-12 CUHK03-NP Local Residual Fusion

### Metadata

- Dataset: `CUHK03-NP`
- Variant: `local_branch`
- Experiment: `cuhk03_np_local_residual`
- Model: `ResNet50 + global branch + local pooled part tokens + learnable residual local fusion`
- Loss: `CrossEntropy Loss + Batch-Hard Triplet Loss`
- Goal: test the same learnable residual local branch used on Market-1501.

### Command

```powershell
py -3.12 scripts\train.py --data-root datasets --dataset-name cuhk03_np --experiment-name cuhk03_np_local_residual --device cuda --use-amp true --num-workers 12 --prefetch-factor 2 --persistent-workers false --batch-size 64 --num-instances 4 --use-local-branch true --use-transformer false --use-fusion-gate false --fusion-mode residual --local-residual-weight 0.1
```

### Outputs

- Output directory: `outputs/cuhk03_np/local_branch/cuhk03_np_local_residual`
- Best checkpoint: `outputs/cuhk03_np/local_branch/cuhk03_np_local_residual/best_model.pth`
- Training log: `outputs/cuhk03_np/local_branch/cuhk03_np_local_residual/train_log.jsonl`
- Final metrics: `outputs/cuhk03_np/local_branch/cuhk03_np_local_residual/final_metrics.json`
- Independent evaluation: `outputs/cuhk03_np/local_branch/cuhk03_np_local_residual/eval_metrics.json`
- Ranking visualization directory: `outputs/cuhk03_np/local_branch/cuhk03_np_local_residual/rankings`

### Metrics

- Rank-1: `59.50%`
- Rank-5: `77.50%`
- Rank-10: `84.93%`
- mAP: `57.45%`
- Best epoch by Rank-1: `55`
- Independent evaluation on `best_model.pth`: Rank-1 `59.50%`, mAP `57.45%`

### Diagnosis

- Compared with baseline, Rank-1 changed from `61.43%` to `59.50%`, a drop of about `1.93` percentage points.
- Compared with baseline, mAP changed from `59.52%` to `57.45%`, a drop of about `2.07` percentage points.
- The learned `local_residual_scale` in this checkpoint is about `0.0029`.
- Market-1501 residual/Transformer/gated residual checkpoints also learned residual scales close to `0`.

### Decision

- Do not treat learnable residual fusion as a meaningful local-branch improvement.
- The important diagnosis is that the model learned to suppress the local branch rather than benefit from it.
- Next experiment: add explicit local auxiliary supervision and keep the residual scale fixed, so the local branch receives a direct training signal and cannot be silently shut off.

## 2026-04-12 CUHK03-NP Local Auxiliary Residual Fusion

### Metadata

- Dataset: `CUHK03-NP`
- Variant: `local_branch`
- Experiment: `cuhk03_np_local_aux_residual`
- Model: `ResNet50 + global branch + local pooled part tokens + fixed residual fusion + local auxiliary BNNeck/classifier`
- Loss: `Global CrossEntropy + Global Triplet + 0.3 * (Local CrossEntropy + Local Triplet)`
- Goal: verify whether direct local supervision makes the local branch useful on a harder benchmark.

### Method Change

- Added `--local-loss-weight 0.3`.
- Added a local auxiliary BNNeck/classifier so local features receive identity supervision.
- Added `--local-residual-learnable false` so the local residual contribution remains fixed at `0.1` instead of being trained toward zero.
- Preserved old checkpoint compatibility by enabling the auxiliary local head only when `local_loss_weight > 0`.

### Command

```powershell
py -3.12 scripts\train.py --data-root datasets --dataset-name cuhk03_np --experiment-name cuhk03_np_local_aux_residual --device cuda --use-amp true --num-workers 12 --prefetch-factor 2 --persistent-workers false --batch-size 64 --num-instances 4 --use-local-branch true --use-transformer false --use-fusion-gate false --fusion-mode residual --local-residual-weight 0.1 --local-residual-learnable false --local-loss-weight 0.3
```

### Outputs

- Output directory: `outputs/cuhk03_np/local_branch/cuhk03_np_local_aux_residual`
- Best checkpoint: `outputs/cuhk03_np/local_branch/cuhk03_np_local_aux_residual/best_model.pth`
- Training log: `outputs/cuhk03_np/local_branch/cuhk03_np_local_aux_residual/train_log.jsonl`
- Final metrics: `outputs/cuhk03_np/local_branch/cuhk03_np_local_aux_residual/final_metrics.json`
- Independent evaluation: `outputs/cuhk03_np/local_branch/cuhk03_np_local_aux_residual/eval_metrics.json`
- Ranking visualization directory: `outputs/cuhk03_np/local_branch/cuhk03_np_local_aux_residual/rankings`

### Metrics

- Rank-1: `63.79%`
- Rank-5: `79.50%`
- Rank-10: `85.93%`
- mAP: `60.29%`
- Best epoch by Rank-1: `50`
- Independent evaluation on `best_model.pth`: Rank-1 `63.71%`, mAP `60.28%`
- Fixed `local_residual_scale`: `0.1`

### Comparison

- Compared with baseline, Rank-1 improved from `61.43%` to `63.79%`, a gain of about `2.36` percentage points.
- Compared with baseline, mAP improved from `59.52%` to `60.29%`, a gain of about `0.77` percentage points.
- Compared with the learnable residual local branch, Rank-1 improved from `59.50%` to `63.79%`, and mAP improved from `57.45%` to `60.29%`.

### Decision

- Use local auxiliary residual fusion as the corrected local-branch method going forward.
- This is now a stronger and more defensible project result: the experiment found a failure mode, diagnosed it through the learned residual scale, then fixed it with direct local supervision and fixed residual contribution.
- Next experiment: run the Transformer branch with the same auxiliary local supervision and fixed residual contribution to isolate whether local relation modeling adds value beyond supervised local tokens.

## 2026-04-12 CUHK03-NP Transformer Auxiliary Residual Fusion

### Metadata

- Dataset: `CUHK03-NP`
- Variant: `transformer_branch`
- Experiment: `cuhk03_np_transformer_aux_residual`
- Model: `ResNet50 + global branch + local pooled part tokens + lightweight Transformer + fixed residual fusion + local auxiliary BNNeck/classifier`
- Loss: `Global CrossEntropy + Global Triplet + 0.3 * (Local CrossEntropy + Local Triplet)`
- Goal: isolate whether Transformer relation modeling among local part tokens adds value beyond directly supervised local tokens.

### Command

```powershell
py -3.12 scripts\train.py --data-root datasets --dataset-name cuhk03_np --experiment-name cuhk03_np_transformer_aux_residual --device cuda --use-amp true --num-workers 12 --prefetch-factor 2 --persistent-workers false --batch-size 64 --num-instances 4 --use-local-branch true --use-transformer true --use-fusion-gate false --fusion-mode residual --local-residual-weight 0.1 --local-residual-learnable false --local-loss-weight 0.3
```

### Outputs

- Output directory: `outputs/cuhk03_np/transformer_branch/cuhk03_np_transformer_aux_residual`
- Best checkpoint: `outputs/cuhk03_np/transformer_branch/cuhk03_np_transformer_aux_residual/best_model.pth`
- Training log: `outputs/cuhk03_np/transformer_branch/cuhk03_np_transformer_aux_residual/train_log.jsonl`
- Final metrics: `outputs/cuhk03_np/transformer_branch/cuhk03_np_transformer_aux_residual/final_metrics.json`
- Independent evaluation: `outputs/cuhk03_np/transformer_branch/cuhk03_np_transformer_aux_residual/eval_metrics.json`
- Ranking visualization directory: `outputs/cuhk03_np/transformer_branch/cuhk03_np_transformer_aux_residual/rankings`

### Metrics

- Rank-1: `62.86%`
- Rank-5: `79.93%`
- Rank-10: `87.29%`
- mAP: `61.19%`
- Best epoch by Rank-1: `60`
- Independent evaluation on `best_model.pth`: Rank-1 `62.93%`, mAP `61.20%`

### Comparison

- Compared with baseline, Rank-1 improved from `61.43%` to `62.86%`, a gain of about `1.43` percentage points.
- Compared with baseline, mAP improved from `59.52%` to `61.19%`, a gain of about `1.67` percentage points.
- Compared with local auxiliary residual fusion, Rank-1 changed from `63.79%` to `62.86%`, a drop of about `0.93` percentage points.
- Compared with local auxiliary residual fusion, mAP changed from `60.29%` to `61.19%`, a gain of about `0.90` percentage points.

### Decision

- The Transformer branch should not be claimed as uniformly better because it lowers Rank-1 relative to local auxiliary residual fusion.
- It can be described as improving retrieval list quality on CUHK03-NP because mAP is higher than both baseline and local auxiliary residual fusion.
- Next experiment: run the full gated residual model with the same auxiliary local supervision to test whether adaptive gating can recover Rank-1 while keeping the mAP gain.

## 2026-04-12 CUHK03-NP Full Auxiliary Gated Residual Model

### Metadata

- Dataset: `CUHK03-NP`
- Variant: `full_model`
- Experiment: `cuhk03_np_full_aux_gated_residual`
- Model: `ResNet50 + global branch + local pooled part tokens + lightweight Transformer + gated residual fusion + local auxiliary BNNeck/classifier`
- Loss: `Global CrossEntropy + Global Triplet + 0.3 * (Local CrossEntropy + Local Triplet)`
- Goal: test whether gated residual fusion improves the full global-local Transformer model.

### Command

```powershell
py -3.12 scripts\train.py --data-root datasets --dataset-name cuhk03_np --experiment-name cuhk03_np_full_aux_gated_residual --device cuda --use-amp true --num-workers 12 --prefetch-factor 2 --persistent-workers false --batch-size 64 --num-instances 4 --use-local-branch true --use-transformer true --use-fusion-gate true --fusion-mode gated_residual --local-residual-weight 0.1 --local-residual-learnable false --local-loss-weight 0.3
```

### Outputs

- Output directory: `outputs/cuhk03_np/full_model/cuhk03_np_full_aux_gated_residual`
- Best checkpoint: `outputs/cuhk03_np/full_model/cuhk03_np_full_aux_gated_residual/best_model.pth`
- Training log: `outputs/cuhk03_np/full_model/cuhk03_np_full_aux_gated_residual/train_log.jsonl`
- Final metrics: `outputs/cuhk03_np/full_model/cuhk03_np_full_aux_gated_residual/final_metrics.json`
- Independent evaluation: `outputs/cuhk03_np/full_model/cuhk03_np_full_aux_gated_residual/eval_metrics.json`
- Ranking visualization directory: `outputs/cuhk03_np/full_model/cuhk03_np_full_aux_gated_residual/rankings`

### Metrics

- Rank-1: `63.29%`
- Rank-5: `80.64%`
- Rank-10: `86.43%`
- mAP: `61.41%`
- Best epoch by Rank-1: `55`
- Independent evaluation on `best_model.pth`: Rank-1 `63.36%`, mAP `61.43%`
- Best mAP observed in the training log: `61.48%` at epoch `60`

### Comparison

- Compared with baseline, Rank-1 improved from `61.43%` to `63.29%`, a gain of about `1.86` percentage points.
- Compared with baseline, mAP improved from `59.52%` to `61.41%`, a gain of about `1.89` percentage points.
- Compared with local auxiliary residual fusion, Rank-1 changed from `63.79%` to `63.29%`, a drop of about `0.50` percentage points.
- Compared with local auxiliary residual fusion, mAP improved from `60.29%` to `61.41%`, a gain of about `1.12` percentage points.
- Compared with Transformer auxiliary residual fusion, Rank-1 improved from `62.86%` to `63.29%`, and mAP improved from `61.19%` to `61.41%`.

### Decision

- On CUHK03-NP, the best Rank-1 result is `cuhk03_np_local_aux_residual`.
- On CUHK03-NP, the best mAP result is `cuhk03_np_full_aux_gated_residual`.
- The full model should be described as improving overall retrieval-list quality rather than producing the best top-1 hit rate.
- The most defensible project conclusion is that auxiliary-supervised local features help on the harder CUHK03-NP benchmark, while Transformer/gated fusion mainly improves mAP.

## 2026-04-12 CUHK03-NP Ablation Summary

| Experiment | Rank-1 | mAP | Main conclusion |
| --- | ---: | ---: | --- |
| `cuhk03_np_baseline` | `61.43%` | `59.52%` | Strong baseline on a harder detected benchmark |
| `cuhk03_np_local_residual` | `59.50%` | `57.45%` | Learnable residual scale suppresses local features and hurts |
| `cuhk03_np_local_aux_residual` | `63.79%` | `60.29%` | Direct local supervision fixes the local branch and gives best Rank-1 |
| `cuhk03_np_transformer_aux_residual` | `62.86%` | `61.19%` | Transformer improves mAP but not Rank-1 over local auxiliary |
| `cuhk03_np_full_aux_gated_residual` | `63.29%` | `61.41%` | Full model gives best mAP and better balance than Transformer-only |

### Project-Level Decision

- Keep the corrected method as `local auxiliary residual fusion`.
- Keep the full model as the named final architecture if the resume bullet emphasizes mAP and retrieval-list quality.
- In interviews, avoid claiming unconditional superiority. The correct statement is: the method improves CUHK03-NP mAP and Rank-1 over baseline, but Rank-1 and mAP peak in different variants.

## 2026-04-13 Market-1501 Local Auxiliary Residual Fusion

### Metadata

- Dataset: `Market-1501`
- Variant: `local_branch`
- Experiment: `market1501_local_aux_residual`
- Model: `ResNet50 + global branch + local pooled part tokens + fixed residual fusion + local auxiliary BNNeck/classifier`
- Loss: `Global CrossEntropy + Global Triplet + 0.3 * (Local CrossEntropy + Local Triplet)`
- Goal: rerun Market-1501 with the corrected local-branch design that worked on CUHK03-NP.

### Command

```powershell
py -3.12 scripts\train.py --data-root datasets --dataset-name market1501 --experiment-name market1501_local_aux_residual --device cuda --use-amp true --num-workers 12 --prefetch-factor 2 --persistent-workers false --batch-size 64 --num-instances 4 --use-local-branch true --use-transformer false --use-fusion-gate false --fusion-mode residual --local-residual-weight 0.1 --local-residual-learnable false --local-loss-weight 0.3
```

### Outputs

- Output directory: `outputs/market1501/local_branch/market1501_local_aux_residual`
- Best checkpoint: `outputs/market1501/local_branch/market1501_local_aux_residual/best_model.pth`
- Training log: `outputs/market1501/local_branch/market1501_local_aux_residual/train_log.jsonl`
- Final metrics: `outputs/market1501/local_branch/market1501_local_aux_residual/final_metrics.json`
- Independent evaluation: `outputs/market1501/local_branch/market1501_local_aux_residual/eval_metrics.json`
- Ranking visualization directory: `outputs/market1501/local_branch/market1501_local_aux_residual/rankings`

### Metrics

- Rank-1: `93.62%`
- Rank-5: `97.92%`
- Rank-10: `98.69%`
- mAP: `83.81%`
- Best epoch by Rank-1: `60`
- Independent evaluation on `best_model.pth`: Rank-1 `93.59%`, mAP `83.81%`

### Comparison

- Compared with baseline, Rank-1 improved from `93.32%` to `93.62%`, a gain of about `0.30` percentage points.
- Compared with baseline, mAP improved from `82.93%` to `83.81%`, a gain of about `0.88` percentage points.
- Compared with the old learnable residual branch, Rank-1 improved from `93.41%` to `93.62%`, and mAP improved from `82.85%` to `83.81%`.

### Decision

- Keep `market1501_local_aux_residual` as the current best Market-1501 local-branch result.
- This validates that the corrected local auxiliary supervision is not only useful on CUHK03-NP; it also improves Market-1501 over the strong baseline.
- Next experiment: run `market1501_transformer_aux_residual` to isolate whether Transformer relation modeling adds value after the local branch is properly supervised.

## 2026-04-13 Market-1501 Transformer Auxiliary Residual Fusion

### Metadata

- Dataset: `Market-1501`
- Variant: `transformer_branch`
- Experiment: `market1501_transformer_aux_residual`
- Model: `ResNet50 + global branch + local pooled part tokens + lightweight Transformer + fixed residual fusion + local auxiliary BNNeck/classifier`
- Loss: `Global CrossEntropy + Global Triplet + 0.3 * (Local CrossEntropy + Local Triplet)`
- Goal: test whether Transformer relation modeling adds value after the local branch receives direct auxiliary supervision.

### Command

```powershell
py -3.12 scripts\train.py --data-root datasets --dataset-name market1501 --experiment-name market1501_transformer_aux_residual --device cuda --use-amp true --num-workers 12 --prefetch-factor 2 --persistent-workers false --batch-size 64 --num-instances 4 --use-local-branch true --use-transformer true --use-fusion-gate false --fusion-mode residual --local-residual-weight 0.1 --local-residual-learnable false --local-loss-weight 0.3
```

### Outputs

- Output directory: `outputs/market1501/transformer_branch/market1501_transformer_aux_residual`
- Best checkpoint: `outputs/market1501/transformer_branch/market1501_transformer_aux_residual/best_model.pth`
- Training log: `outputs/market1501/transformer_branch/market1501_transformer_aux_residual/train_log.jsonl`
- Final metrics: `outputs/market1501/transformer_branch/market1501_transformer_aux_residual/final_metrics.json`
- Independent evaluation: `outputs/market1501/transformer_branch/market1501_transformer_aux_residual/eval_metrics.json`
- Ranking visualization directory: `outputs/market1501/transformer_branch/market1501_transformer_aux_residual/rankings`

### Metrics

- Rank-1: `93.65%`
- Rank-5: `97.77%`
- Rank-10: `98.75%`
- mAP: `83.73%`
- Best epoch by Rank-1: `60`
- Independent evaluation on `best_model.pth`: Rank-1 `93.68%`, mAP `83.73%`

### Comparison

- Compared with baseline, Rank-1 improved from `93.32%` to `93.65%`, a gain of about `0.33` percentage points.
- Compared with baseline, mAP improved from `82.93%` to `83.73%`, a gain of about `0.80` percentage points.
- Compared with local auxiliary residual fusion, Rank-1 changed from `93.62%` to `93.65%`, a small gain of about `0.03` percentage points.
- Compared with local auxiliary residual fusion, mAP changed from `83.81%` to `83.73%`, a small drop of about `0.08` percentage points.

### Decision

- Transformer relation modeling does not provide a clear Market-1501 improvement beyond supervised local tokens.
- The Transformer variant is competitive and slightly improves Rank-1, but local auxiliary residual fusion remains better by mAP.
- Next experiment: run the full auxiliary gated residual model to see whether gating improves the global-local Transformer fusion balance.

## 2026-04-13 Market-1501 Full Auxiliary Gated Residual Fusion

### Metadata

- Dataset: `Market-1501`
- Variant: `full_model`
- Experiment: `market1501_full_aux_gated_residual`
- Model: `ResNet50 + global branch + local pooled part tokens + lightweight Transformer + gated residual fusion + local auxiliary BNNeck/classifier`
- Loss: `Global CrossEntropy + Global Triplet + 0.3 * (Local CrossEntropy + Local Triplet)`
- Goal: test whether gated residual fusion improves the full global-local Transformer model on Market-1501.

### Command

```powershell
py -3.12 scripts\train.py --data-root datasets --dataset-name market1501 --experiment-name market1501_full_aux_gated_residual --device cuda --use-amp true --num-workers 12 --prefetch-factor 2 --persistent-workers false --batch-size 64 --num-instances 4 --use-local-branch true --use-transformer true --use-fusion-gate true --fusion-mode gated_residual --local-residual-weight 0.1 --local-residual-learnable false --local-loss-weight 0.3
```

### Outputs

- Output directory: `outputs/market1501/full_model/market1501_full_aux_gated_residual`
- Best checkpoint: `outputs/market1501/full_model/market1501_full_aux_gated_residual/best_model.pth`
- Training log: `outputs/market1501/full_model/market1501_full_aux_gated_residual/train_log.jsonl`
- Final metrics: `outputs/market1501/full_model/market1501_full_aux_gated_residual/final_metrics.json`
- Independent evaluation: `outputs/market1501/full_model/market1501_full_aux_gated_residual/eval_metrics.json`
- Ranking visualization directory: `outputs/market1501/full_model/market1501_full_aux_gated_residual/rankings`

### Metrics

- Rank-1: `93.26%`
- Rank-5: `97.71%`
- Rank-10: `98.78%`
- mAP: `83.27%`
- Best epoch by Rank-1: `60`
- Independent evaluation on `best_model.pth`: Rank-1 `93.29%`, mAP `83.28%`
- Best mAP observed in the training log: `83.27%` at epoch `60`

### Comparison

- Compared with baseline, Rank-1 changed from `93.32%` to `93.26%`, a drop of about `0.06` percentage points.
- Compared with baseline, mAP improved from `82.93%` to `83.27%`, a gain of about `0.34` percentage points.
- Compared with local auxiliary residual fusion, Rank-1 dropped from `93.62%` to `93.26%`, and mAP dropped from `83.81%` to `83.27%`.
- Compared with Transformer auxiliary residual fusion, Rank-1 dropped from `93.65%` to `93.26%`, and mAP dropped from `83.73%` to `83.27%`.

### Decision

- Gated residual fusion is not the best Market-1501 choice.
- The full model still improves mAP over the baseline, but it weakens the stronger corrected local and Transformer residual variants.
- Use this as an ablation result rather than the headline Market-1501 method.

## 2026-04-13 Market-1501 Ablation Summary

| Experiment | Rank-1 | mAP | Main conclusion |
| --- | ---: | ---: | --- |
| `market1501_baseline` | `93.32%` | `82.93%` | Strong baseline; already near saturation on Market-1501 |
| `market1501_local_branch` | `90.29%` | `75.93%` | Direct projection fusion hurts the strong global feature |
| `market1501_local_residual` | `93.41%` | `82.85%` | Learnable residual scale mostly suppresses the local feature |
| `market1501_transformer_residual` | `93.17%` | `82.79%` | Transformer without auxiliary local supervision is not effective |
| `market1501_gated_residual` | `93.26%` | `82.67%` | Gating without auxiliary local supervision does not fix the issue |
| `market1501_local_aux_residual` | `93.62%` | `83.81%` | Best Market-1501 mAP; local auxiliary supervision makes the local branch useful |
| `market1501_transformer_aux_residual` | `93.65%` | `83.73%` | Best Market-1501 Rank-1; Transformer gives a small top-1 gain |
| `market1501_full_aux_gated_residual` | `93.26%` | `83.27%` | Improves mAP over baseline, but gated fusion is weaker than fixed residual fusion |

### Project-Level Decision

- On Market-1501, report `market1501_local_aux_residual` as the best mAP variant and `market1501_transformer_aux_residual` as the best Rank-1 variant.
- The strongest project narrative is not "Transformer always wins"; it is "local branches are easy to ignore or damage a strong ReID baseline, so auxiliary supervision and controlled residual fusion are necessary."
- This conclusion is consistent with CUHK03-NP: the corrected local auxiliary design improves over baseline, while Transformer/gated fusion mainly changes the Rank-1 and mAP tradeoff.

## 2026-04-14 Branch Marginalization Diagnostics And Report Refresh

### Metadata

- Type: offline checkpoint analysis and documentation update
- Datasets: `Market-1501`, `CUHK03-NP`
- Compared variants: `local_residual` vs `local_aux_residual`
- Goal: strengthen the evidence chain for the claim that local branches can be marginalized without fixed residual fusion and local auxiliary supervision.

### Method

- Added reusable branch diagnostics that measure:
  - final `local_residual_scale`;
  - global/local/fused feature norms;
  - fused-vs-global logit contribution;
  - parameter-group gradient norms from a few training batches;
  - fused/global/local retrieval metrics from trained checkpoints.
- Added per-epoch model-state logging to future training runs, including `local_residual_scale`.
- Added optional `--checkpoint-period` so future runs can save stage checkpoints for global/local discriminability trend analysis.
- Added report generation for ablation status, stability status, experiment queue, and the final project narrative.

### Commands

```powershell
py -3.12 scripts\analyze_branch_diagnostics.py --data-root datasets --dataset-name market1501 --output-dir outputs\analysis\branch_marginalization\market1501 --device cuda --batch-size 64 --num-workers 4 --prefetch-factor 2 --persistent-workers true --pin-memory true --channels-last true --use-amp true --max-train-batches 4 --run learnable_no_aux=outputs\market1501\local_branch\market1501_local_residual\best_model.pth --run fixed_aux=outputs\market1501\local_branch\market1501_local_aux_residual\best_model.pth
```

```powershell
py -3.12 scripts\analyze_branch_diagnostics.py --data-root datasets --dataset-name cuhk03_np --output-dir outputs\analysis\branch_marginalization\cuhk03_np --device cuda --batch-size 64 --num-workers 4 --prefetch-factor 2 --persistent-workers true --pin-memory true --channels-last true --use-amp true --max-train-batches 4 --run learnable_no_aux=outputs\cuhk03_np\local_branch\cuhk03_np_local_residual\best_model.pth --run fixed_aux=outputs\cuhk03_np\local_branch\cuhk03_np_local_aux_residual\best_model.pth
```

```powershell
py -3.12 scripts\build_research_reports.py --output-root outputs --docs-dir docs --stability-seeds 42,123,3407
```

### Outputs

- Market diagnostics: `outputs/analysis/branch_marginalization/market1501`
- CUHK03-NP diagnostics: `outputs/analysis/branch_marginalization/cuhk03_np`
- Branch evidence document: `docs/branch_marginalization_analysis.md`
- Ablation summary: `docs/ablation_summary.md`
- Stability summary: `docs/stability_summary.md`
- Final project narrative: `docs/project_narrative.md`
- Missing experiment queue: `outputs/analysis/experiment_queue.md`
- Refreshed experiment table: `outputs/experiment_table.md`

### Key Findings

- Market-1501 learnable residual without local auxiliary supervision has final `local_residual_scale = 4.29e-08`; CUHK03-NP has final `local_residual_scale = 0.00289`.
- In these no-auxiliary runs, fused and global retrieval are effectively identical, while local-only retrieval is near random.
- With fixed residual and local auxiliary supervision, local-only retrieval becomes meaningful:
  - Market-1501 local-only: Rank-1 `91.83%`, mAP `79.05%`.
  - CUHK03-NP local-only: Rank-1 `61.21%`, mAP `59.26%`.
- Gradient diagnostics show the no-auxiliary local branch receives near-zero gradients, while the corrected local branch receives measurable gradients.

### Decision

- Use `docs/branch_marginalization_analysis.md` as the main evidence-chain document for the local-branch marginalization claim.
- Do not claim multi-seed stability yet; `docs/stability_summary.md` explicitly records that only seed 42 is complete for the key baseline-vs-corrected comparison.
- Prioritize the missing synergy ablations in `outputs/analysis/experiment_queue.md` before adding new model modules or MSMT17 experiments.

## 2026-04-15 Narrative Strengthening Ablations And Multi-Seed Runs

### Metadata

- Type: long-running ablation and stability queue
- Runner: `scripts/run_research_queue.py`
- Goal: complete the remaining reinforcement items from `docs/project_narrative.md`:
  - fixed residual only;
  - local auxiliary with learnable residual;
  - residual weight trend;
  - local part number trend;
  - baseline vs corrected variant multi-seed stability.

### Runtime Notes

- Initial `num_workers=12`, `persistent_workers=true` run hit a Windows DataLoader shared-memory mapping error `1455` during evaluation.
- Queue commands were changed to `num_workers=4`, `prefetch_factor=2`, `persistent_workers=false` for long-run stability on Windows.
- `market1501_fixed_residual_no_aux_w0_10` was resumed from `last_model.pth` after the worker failure.
- `market1501_local_aux_residual_parts4` had one transient early exit with code `3221226505` and then completed successfully on rerun.
- Full queue log: `outputs/analysis/run_logs/queue.log`.

### Result Table

| Dataset | Experiment | Seed | Rank-1 | mAP | Rank-5 | Rank-10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| market1501 | `market1501_fixed_residual_no_aux_w0_10` | `42` | `93.14%` | `83.07%` | `97.68%` | `98.63%` |
| market1501 | `market1501_local_aux_learnable_residual_w0_10` | `42` | `93.56%` | `83.84%` | `97.80%` | `98.69%` |
| market1501 | `market1501_local_aux_residual_w0_05` | `42` | `93.59%` | `84.08%` | `97.68%` | `98.46%` |
| market1501 | `market1501_local_aux_residual_w0_20` | `42` | `93.41%` | `83.52%` | `97.65%` | `98.60%` |
| market1501 | `market1501_local_aux_residual_parts4` | `42` | `93.65%` | `83.67%` | `97.77%` | `98.63%` |
| market1501 | `market1501_local_aux_residual_parts8` | `42` | `94.06%` | `83.96%` | `97.83%` | `98.55%` |
| market1501 | `market1501_baseline_seed123` | `123` | `93.68%` | `82.51%` | `98.01%` | `98.66%` |
| market1501 | `market1501_local_aux_residual_seed123` | `123` | `93.29%` | `83.69%` | `97.89%` | `98.72%` |
| market1501 | `market1501_baseline_seed3407` | `3407` | `93.68%` | `82.76%` | `98.04%` | `98.84%` |
| market1501 | `market1501_local_aux_residual_seed3407` | `3407` | `93.38%` | `83.58%` | `97.95%` | `98.57%` |
| cuhk03_np | `cuhk03_np_baseline_seed123` | `123` | `61.79%` | `59.35%` | `78.64%` | `86.07%` |
| cuhk03_np | `cuhk03_np_local_aux_residual_seed123` | `123` | `63.07%` | `60.81%` | `80.50%` | `87.29%` |
| cuhk03_np | `cuhk03_np_baseline_seed3407` | `3407` | `61.50%` | `58.97%` | `78.71%` | `86.36%` |
| cuhk03_np | `cuhk03_np_local_aux_residual_seed3407` | `3407` | `61.57%` | `60.20%` | `79.29%` | `86.43%` |

### Ablation Conclusions

- Fixed residual alone is not enough to be the main explanation: compared with Market-1501 baseline, it changes Rank-1 by `-0.18` percentage points and mAP by `+0.14` percentage points.
- Local auxiliary supervision is the main source of the corrected-branch gain: local auxiliary with learnable residual reaches Rank-1 `93.56%`, mAP `83.84%`.
- Fixed residual plus local auxiliary reaches Rank-1 `93.62%`, mAP `83.81%`, so the final wording should be that fixed residual stabilizes and constrains the local contribution, while auxiliary supervision provides the strongest training signal.
- Residual weight trend: `0.05` gives the best mAP (`84.08%`), `0.1` is a balanced and interpretable default, and `0.2` drops to mAP `83.52%`, suggesting the local residual should stay small.
- Part number trend: `8` parts gives the best single-seed Market-1501 result (`94.06%` Rank-1, `83.96%` mAP), but this setting has not been multi-seed validated, so it should be reported as sensitivity analysis rather than the main contribution.

### Stability Conclusions

- Market-1501 multi-seed:
  - Baseline: Rank-1 `93.56% ± 0.21 pp`, mAP `82.73% ± 0.21 pp`.
  - Corrected local auxiliary residual: Rank-1 `93.43% ± 0.17 pp`, mAP `83.69% ± 0.11 pp`.
  - Paired mAP gains are positive for all three seeds; Rank-1 is not stable on Market-1501.
- CUHK03-NP multi-seed:
  - Baseline: Rank-1 `61.57% ± 0.19 pp`, mAP `59.28% ± 0.28 pp`.
  - Corrected local auxiliary residual: Rank-1 `62.81% ± 1.13 pp`, mAP `60.44% ± 0.33 pp`.
  - Paired mAP gains are positive for all three seeds; Rank-1 is also positive but has larger variance.

### Decision

- Update final project narrative to emphasize stable mAP gains and avoid claiming universal Rank-1 improvement.
- Keep `local_aux_residual` as the main corrected variant because it is simple and interpretable.
- Treat `num_parts=8` and `local_residual_weight=0.05` as single-seed sensitivity findings, not as a new headline architecture.
