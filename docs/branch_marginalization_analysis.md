# 局部分支边缘化证据链分析

## 目标

本分析回答一个具体问题：在强 ReID baseline 上，加入局部分支后，局部分支是否真的参与了检索，还是被训练过程压制成接近无效的结构。

分析输出目录：

- `outputs/analysis/branch_marginalization/market1501`
- `outputs/analysis/branch_marginalization/cuhk03_np`

核心脚本：

```powershell
py -3.12 scripts\analyze_branch_diagnostics.py --data-root datasets --dataset-name market1501 --output-dir outputs\analysis\branch_marginalization\market1501 --device cuda --batch-size 64 --num-workers 4 --prefetch-factor 2 --persistent-workers true --pin-memory true --channels-last true --use-amp true --max-train-batches 4 --run learnable_no_aux=outputs\market1501\local_branch\market1501_local_residual\best_model.pth --run fixed_aux=outputs\market1501\local_branch\market1501_local_aux_residual\best_model.pth
```

```powershell
py -3.12 scripts\analyze_branch_diagnostics.py --data-root datasets --dataset-name cuhk03_np --output-dir outputs\analysis\branch_marginalization\cuhk03_np --device cuda --batch-size 64 --num-workers 4 --prefetch-factor 2 --persistent-workers true --pin-memory true --channels-last true --use-amp true --max-train-batches 4 --run learnable_no_aux=outputs\cuhk03_np\local_branch\cuhk03_np_local_residual\best_model.pth --run fixed_aux=outputs\cuhk03_np\local_branch\cuhk03_np_local_aux_residual\best_model.pth
```

## Residual Scale 证据

| Dataset | Model | Learnable | Final Scale | 解释 |
| --- | ---: | ---: | ---: | --- |
| Market-1501 | `local_residual` | true | `4.29e-08` | 几乎为 0，局部残差路径被压制 |
| Market-1501 | `local_aux_residual` | false | `0.1000` | 固定残差保证局部路径保留 |
| CUHK03-NP | `local_residual` | true | `0.00289` | 仍远低于初始化 0.1 |
| CUHK03-NP | `local_aux_residual` | false | `0.1000` | 固定残差稳定参与融合 |

结论：learnable residual 的“自适应”并没有自动学到有用局部贡献，反而倾向于把局部分支 scale 收缩到很小。这个结果支持“局部分支被边缘化”的核心判断。

限制：这些旧实验当时没有逐 epoch 保存 `local_residual_scale`，所以当前只能报告 checkpoint 端点值。现在训练日志已经新增 `local_residual_scale`、`local_residual_scale_abs` 和 `local_residual_scale_learnable` 字段，后续训练可以直接观察 scale 轨迹。

## Feature Norm 与 Logit Contribution

| Dataset | Model | Local/Global Norm | Fused Delta/Global | Logit Delta Norm | Local Aux Conf |
| --- | ---: | ---: | ---: | ---: | ---: |
| Market-1501 | `local_residual` | `0.0000` | `0.000000` | `0.0069` | - |
| Market-1501 | `local_aux_residual` | `0.6792` | `0.067904` | `1.2901` | `0.8711` |
| CUHK03-NP | `local_residual` | `0.0001` | `0.000000` | `0.0070` | - |
| CUHK03-NP | `local_aux_residual` | `0.6990` | `0.069883` | `1.3099` | `0.8939` |

解释：

- 无局部辅助监督时，local embedding 的 norm 接近 0，fused feature 与 global feature 几乎一致。
- logit delta norm 只有约 `0.007`，说明加入局部分支前后分类 logits 基本不变。
- 固定残差 + 局部辅助监督后，local/global norm ratio 约 `0.68-0.70`，fused feature 中能观测到非零局部扰动，logit delta norm 提升到约 `1.3`。

结论：局部分支不是只在结构图里存在；在 corrected variant 中，它对特征和 logits 都产生了可测贡献。

## Gradient Norm 证据

| Dataset | Model | Global Projection Grad | Local Branch Grad | Local Aux Head Grad |
| --- | ---: | ---: | ---: | ---: |
| Market-1501 | `local_residual` | `0.3810` | `0.0000` | - |
| Market-1501 | `local_aux_residual` | `0.4158` | `0.1516` | `0.0882` |
| CUHK03-NP | `local_residual` | `0.3919` | `0.0004` | - |
| CUHK03-NP | `local_aux_residual` | `0.3782` | `0.1187` | `0.0515` |

解释：

- learnable residual scale 接近 0 后，来自主损失的梯度几乎传不到局部分支。
- local auxiliary head 让局部分支获得直接监督，local branch gradient norm 从接近 0 恢复到可观量级。

结论：局部分支被边缘化不是单纯的推理现象，也反映在训练信号上。

## Global / Local 判别性

| Dataset | Model | Feature | Rank-1 | mAP |
| --- | ---: | ---: | ---: | ---: |
| Market-1501 | `local_residual` | fused | `93.41%` | `82.85%` |
| Market-1501 | `local_residual` | global | `93.41%` | `82.85%` |
| Market-1501 | `local_residual` | local | `0.00%` | `0.11%` |
| Market-1501 | `local_aux_residual` | fused | `93.62%` | `83.81%` |
| Market-1501 | `local_aux_residual` | global | `93.56%` | `83.83%` |
| Market-1501 | `local_aux_residual` | local | `91.83%` | `79.05%` |
| CUHK03-NP | `local_residual` | fused | `59.43%` | `57.44%` |
| CUHK03-NP | `local_residual` | global | `59.43%` | `57.44%` |
| CUHK03-NP | `local_residual` | local | `0.14%` | `0.45%` |
| CUHK03-NP | `local_aux_residual` | fused | `63.79%` | `60.32%` |
| CUHK03-NP | `local_aux_residual` | global | `63.57%` | `60.24%` |
| CUHK03-NP | `local_aux_residual` | local | `61.21%` | `59.26%` |

解释：

- learnable residual 无辅助监督时，fused 和 global 的检索结果完全一致或几乎一致，local-only 检索接近随机。
- corrected variant 中，local-only 特征本身已经具备较强检索能力，说明局部辅助监督确实让局部分支学到了身份判别信息。
- Market-1501 上 `local_aux_residual` 的 global-only mAP 略高于 fused mAP，说明收益不只来自测试时残差相加，也来自局部辅助监督对共享 backbone/global 表征的训练正则化。这个现象不能被夸大成“局部残差一定单独提升所有指标”。

## 训练阶段趋势怎么补

旧实验没有保存中间 epoch checkpoint，因此无法严格回溯“每个阶段 global/local 判别性”。当前代码已经补上两个入口：

- 训练日志每个 epoch 自动记录 residual scale 状态；
- 可选 `--checkpoint-period N` 保存阶段 checkpoint。

建议对后续关键补跑使用：

```powershell
py -3.12 scripts\train.py ... --checkpoint-period 10
```

训练完成后，把多个阶段 checkpoint 传给诊断脚本：

```powershell
py -3.12 scripts\analyze_branch_diagnostics.py --data-root datasets --dataset-name market1501 --output-dir outputs\analysis\branch_marginalization\market1501_stage_trend --run epoch10=outputs\market1501\local_branch\<experiment>\epoch_010_model.pth --run epoch20=outputs\market1501\local_branch\<experiment>\epoch_020_model.pth --run final=outputs\market1501\local_branch\<experiment>\best_model.pth
```

这样可以得到不同训练阶段的 residual scale、feature/logit contribution、gradient norm 和 global/local/fused 检索能力变化。

## 总结

任务 A 的现有证据链已经比较清楚：

- scale 证据：learnable residual 在两个数据集上都明显收缩；
- feature 证据：无辅助监督时 local norm 和 fused delta 接近 0；
- gradient 证据：无辅助监督时局部分支梯度接近 0；
- logit 证据：无辅助监督时加入局部分支几乎不改变 logits；
- retrieval 证据：无辅助监督时 local-only 检索几乎失效，而 corrected variant 的 local-only 检索具备明显判别性。

因此，更稳妥的项目结论应写成：局部分支在强 baseline 上不是天然有效，必须通过固定残差和直接局部监督避免被边缘化。
