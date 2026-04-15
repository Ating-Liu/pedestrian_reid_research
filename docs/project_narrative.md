# 行人重识别项目最终叙事版本

## 项目定位

本项目不是为了盲目堆复杂模型，而是围绕一个强 ReID baseline 做完整研究闭环：复现、发现失败模式、诊断原因、提出修正、做消融和案例分析。最终可讲的重点是实验诊断能力，而不是简单宣称 Transformer 更强。

## 从项目创建到当前版本的路线

1. 建立 `ResNet50 + BNNeck + CE + Triplet` 强 baseline，使用 Rank-1/mAP 作为检索指标。
2. 加入局部分支后，直接投影融合在 Market-1501 上明显退化，说明局部结构并不天然有效。
3. 将融合改成 residual 后，退化被缓解，但 learnable residual scale 在训练后容易接近 0，局部分支仍可能被边缘化。
4. 加入固定小残差和局部辅助 `CE + Triplet` 后，局部分支获得直接身份监督，并能稳定参与 fused retrieval。
5. 再测试 Transformer/gated fusion，结论保持保守：它们在部分指标上有收益，但不是全面优于 corrected local branch。

## 当前核心结论

- 在强 ReID baseline 上，局部分支如果融合和监督设计不当，容易被训练过程边缘化。
- learnable residual 融合可能把局部分支权重压到接近 0，使结构存在但贡献很小。
- 消融显示 local auxiliary CE/Triplet 是主要收益来源；fixed residual 的价值是保证局部路径稳定参与融合，避免 learnable scale 把局部分支压没。
- Transformer 只能作为局部关系建模的补充消融，不能写成“全面提升”。

## 关键结果

| Dataset | Experiment | Rank-1 | mAP | Use in report |
| --- | ---: | ---: | ---: | ---: |
| Market-1501 | market1501_baseline | 93.32% | 82.93% | baseline |
| Market-1501 | market1501_local_aux_residual | 93.62% | 83.81% | mAP +0.88 pp |
| Market-1501 | market1501_transformer_aux_residual | 93.65% | 83.73% | Rank-1 +0.33 pp |
| CUHK03-NP | cuhk03_np_baseline | 61.43% | 59.52% | baseline |
| CUHK03-NP | cuhk03_np_local_aux_residual | 63.79% | 60.29% | Rank-1 +2.36 pp |
| CUHK03-NP | cuhk03_np_full_aux_gated_residual | 63.29% | 61.41% | mAP +1.89 pp |

## 可写进简历的内容

- 基于 PyTorch 搭建行人重识别实验框架，复现 `ResNet50 + BNNeck + CE + Triplet` 强 baseline，并在 Market-1501 与 CUHK03-NP 上完成 Rank-1/mAP 评测。
- 发现局部分支在 learnable residual 融合下可能被训练过程压制，围绕 feature norm、gradient norm、logit contribution、residual scale 与检索案例构建诊断证据链。
- 设计固定残差融合与局部辅助 `CE + Triplet` 监督，使局部特征真正参与检索；在 seed 42 结果中，Market-1501 mAP 从 `82.93%` 提升到 `83.81%`，CUHK03-NP Rank-1 从 `61.43%` 提升到 `63.79%`。
- 补齐协同消融、residual weight、part 数和多 seed 稳定性实验；多 seed 下 Market-1501 与 CUHK03-NP 的 mAP 提升更稳定，Rank-1 提升存在数据集差异。

## 只适合内部分析的内容

- 多 seed 稳定性如果只在部分指标上稳定，简历中要保留 mean ± std 和具体数据，不写成所有指标全面提升。
- residual weight 和 local part 数量趋势只服务于说明设计选择，不适合包装成新的复杂模块贡献。
- Transformer/gated fusion 的结果更适合讲成“指标取舍和边界条件”，不适合作为项目主卖点。

## 面试讲述版本

我先复现了 ReID 强基线，而不是直接堆模块。第一次加入局部分支后，Market-1501 反而下降，这说明局部结构本身不保证有效。我随后把问题拆成融合方式和监督方式两部分：learnable residual 虽然看起来灵活，但训练后会把局部分支 scale 压到接近 0；于是我加入固定小残差约束，并给局部分支增加辅助 CE 和 Triplet 监督。后续消融显示，主要收益来自局部辅助监督，fixed residual 更像是防止局部路径被压没的稳定约束。多 seed 结果进一步说明，mAP 的提升比 Rank-1 更稳定。因此这个项目的核心不是 Transformer 全面更强，而是强 baseline 上局部特征需要直接监督和受控融合，才能可靠参与检索。

## 补强完成情况

- 协同消融已完成：fixed-only 为 Rank-1 93.14%, mAP 83.07%，aux + learnable residual 为 Rank-1 93.56%, mAP 83.84%，fixed + aux 为 Rank-1 93.62%, mAP 83.81%。
- 这部分用于回答“fixed residual 和 local auxiliary 是否协同解决边缘化问题”，具体结论以 `docs/ablation_summary.md` 为准。
- residual weight 趋势已完成：0.05 为 Rank-1 93.59%, mAP 84.08%，0.1 为 Rank-1 93.62%, mAP 83.81%，0.2 为 Rank-1 93.41%, mAP 83.52%。
- local part 数趋势已完成：4 parts 为 Rank-1 93.65%, mAP 83.67%，6 parts 为 Rank-1 93.62%, mAP 83.81%，8 parts 为 Rank-1 94.06%, mAP 83.96%。
- Market-1501 多 seed：baseline Rank-1 93.56% ± 0.21 pp，corrected Rank-1 93.43% ± 0.17 pp；baseline mAP 82.73% ± 0.21 pp，corrected mAP 83.69% ± 0.11 pp。
- CUHK03-NP 多 seed：baseline Rank-1 61.57% ± 0.19 pp，corrected Rank-1 62.81% ± 1.13 pp；baseline mAP 59.28% ± 0.28 pp，corrected mAP 60.44% ± 0.33 pp。
- paired seed 口径：Market-1501 mAP 平均提升 +0.96 pp，Rank-1 平均变化 -0.13 pp；CUHK03-NP mAP 平均提升 +1.16 pp，Rank-1 平均提升 +1.24 pp。
