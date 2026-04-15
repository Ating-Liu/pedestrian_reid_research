# Stability Summary

## Current Seed Coverage

| Dataset/Method | Seeds | Runs | Rank-1 | mAP |
| --- | ---: | ---: | ---: | ---: |
| market1501/baseline | 42, 123, 3407 | 3 | 93.56% ± 0.21 pp | 82.73% ± 0.21 pp |
| market1501/local_aux_residual | 42, 123, 3407 | 3 | 93.43% ± 0.17 pp | 83.69% ± 0.11 pp |
| cuhk03_np/baseline | 42, 123, 3407 | 3 | 61.57% ± 0.19 pp | 59.28% ± 0.28 pp |
| cuhk03_np/local_aux_residual | 42, 123, 3407 | 3 | 62.81% ± 1.13 pp | 60.44% ± 0.33 pp |

## Current Conclusion

- 多 seed 稳定性补跑已完成，当前可以报告 mean ± std，并用 paired seed 差值说明提升是否稳定出现。
- Market-1501 上 corrected variant 的 mAP 每个 seed 都提升，平均 paired 提升 +0.96 pp；Rank-1 不稳定，平均 paired 变化 -0.13 pp。
- CUHK03-NP 上 corrected variant 的 mAP 每个 seed 都提升，平均 paired 提升 +1.16 pp；Rank-1 也为正向但波动较大，平均 paired 提升 +1.24 pp。
- 简历中可以写“多 seed 结果显示 mAP 提升较稳定”，但不要写“Rank-1 和 mAP 全面稳定提升”。

## Paired Seed Differences

### Market-1501

| Seed | Baseline Rank-1 | Corrected Rank-1 | ΔRank-1 | Baseline mAP | Corrected mAP | ΔmAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 93.32% | 93.62% | +0.30 pp | 82.93% | 83.81% | +0.88 pp |
| 123 | 93.68% | 93.29% | -0.39 pp | 82.51% | 83.69% | +1.18 pp |
| 3407 | 93.68% | 93.38% | -0.30 pp | 82.76% | 83.58% | +0.82 pp |

### CUHK03-NP

| Seed | Baseline Rank-1 | Corrected Rank-1 | ΔRank-1 | Baseline mAP | Corrected mAP | ΔmAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 61.43% | 63.79% | +2.36 pp | 59.52% | 60.29% | +0.78 pp |
| 123 | 61.79% | 63.07% | +1.29 pp | 59.35% | 60.81% | +1.47 pp |
| 3407 | 61.50% | 61.57% | +0.07 pp | 58.97% | 60.20% | +1.23 pp |

## Missing Stability Commands

All planned stability runs have final metrics.