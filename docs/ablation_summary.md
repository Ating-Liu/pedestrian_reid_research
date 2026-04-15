# Ablation Summary

## Completed Evidence

| Dataset | Experiment | Rank-1 | mAP | ΔRank-1 vs baseline | ΔmAP vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| market1501 | market1501_baseline | 93.32% | 82.93% | +0.00 pp | +0.00 pp |
| market1501 | market1501_local_branch | 90.29% | 75.93% | -3.03 pp | -6.99 pp |
| market1501 | market1501_local_residual | 93.41% | 82.85% | +0.09 pp | -0.07 pp |
| market1501 | market1501_local_aux_residual | 93.62% | 83.81% | +0.30 pp | +0.88 pp |
| market1501 | market1501_transformer_aux_residual | 93.65% | 83.73% | +0.33 pp | +0.80 pp |
| market1501 | market1501_full_aux_gated_residual | 93.26% | 83.27% | -0.06 pp | +0.35 pp |
| cuhk03_np | cuhk03_np_baseline | 61.43% | 59.52% | +0.00 pp | +0.00 pp |
| cuhk03_np | cuhk03_np_local_residual | 59.50% | 57.45% | -1.93 pp | -2.07 pp |
| cuhk03_np | cuhk03_np_local_aux_residual | 63.79% | 60.29% | +2.36 pp | +0.78 pp |
| cuhk03_np | cuhk03_np_transformer_aux_residual | 62.86% | 61.19% | +1.43 pp | +1.67 pp |
| cuhk03_np | cuhk03_np_full_aux_gated_residual | 63.29% | 61.41% | +1.86 pp | +1.89 pp |

## Required Synergy Ablations

| Question | Experiment | Status | Rank-1 | mAP |
| --- | ---: | ---: | ---: | ---: |
| fixed residual only | market1501_fixed_residual_no_aux_w0_10 | done | 93.14% | 83.07% |
| local auxiliary only | market1501_local_aux_learnable_residual_w0_10 | done | 93.56% | 83.84% |
| fixed residual + local auxiliary | market1501_local_aux_residual | done | 93.62% | 83.81% |
| residual weight 0.05 | market1501_local_aux_residual_w0_05 | done | 93.59% | 84.08% |
| residual weight 0.2 | market1501_local_aux_residual_w0_20 | done | 93.41% | 83.52% |
| num parts 4 | market1501_local_aux_residual_parts4 | done | 93.65% | 83.67% |
| num parts 8 | market1501_local_aux_residual_parts8 | done | 94.06% | 83.96% |

## Current Conclusion

- 协同消融已经闭合：`fixed residual only`、`local auxiliary + learnable residual`、`fixed residual + local auxiliary` 三个关键点都有结果。
- `fixed residual only`: Rank-1 93.14%, mAP 83.07%；`local auxiliary + learnable residual`: Rank-1 93.56%, mAP 83.84%；`fixed residual + local auxiliary`: Rank-1 93.62%, mAP 83.81%。
- 相对 Market-1501 baseline，fixed-only 为 Rank-1 -0.18 pp, mAP +0.14 pp；aux+learnable 为 Rank-1 +0.24 pp, mAP +0.92 pp；fixed+aux 为 Rank-1 +0.30 pp, mAP +0.88 pp。
- 更准确的解释是：local auxiliary supervision 是主要收益来源；fixed residual 的价值是保证局部路径不会被 scale 压没，并让 Rank-1 更稳一点。不要把 fixed residual alone 夸大成主要提升来源。
- residual weight 趋势已补齐：0.05 为 Rank-1 93.59%, mAP 84.08%，0.1 为 Rank-1 93.62%, mAP 83.81%，0.2 为 Rank-1 93.41%, mAP 83.52%。0.05 的 mAP 最高，0.2 开始回落，说明局部残差不宜过大；0.1 仍是解释最稳妥的默认设置。
- part 数趋势已补齐：4 parts 为 Rank-1 93.65%, mAP 83.67%，6 parts 为 Rank-1 93.62%, mAP 83.81%，8 parts 为 Rank-1 94.06%, mAP 83.96%。8 parts 单 seed 最强，但没有多 seed 验证；报告中可以说 part granularity 有影响，但不应包装成新的复杂模块贡献。

## Command Example

```powershell
py -3.12 scripts\train.py --data-root datasets --dataset-name market1501 --experiment-name market1501_fixed_residual_no_aux_w0_10 --seed 42 --device cuda --use-amp true --channels-last true --cuda-prefetch true --fused-optimizer true --cudnn-benchmark true --allow-tf32 true --num-workers 4 --prefetch-factor 2 --persistent-workers false --pin-memory true --batch-size 64 --num-instances 4 --use-local-branch true --use-transformer false --use-fusion-gate false --fusion-mode residual --local-residual-weight 0.1 --local-residual-learnable false --local-loss-weight 0.0
```
