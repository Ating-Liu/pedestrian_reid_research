# 基于全局-局部特征融合的行人重识别方法研究

## 1. 项目定位

本项目面向行人重识别（Person Re-ID）任务，目标是在公开 benchmark 上完成从强基线复现、模块改进、消融实验到结果分析的完整算法研究流程。项目不做 GUI、摄像头演示或产品化包装，核心价值是证明自己具备深度学习视觉任务中的数据处理、模型训练、指标评测、实验诊断和技术表达能力。

项目与已有 `SW-RTV / MRI 重建 / 非凸优化` 工作形成互补：原论文体现图像重建与优化能力，本项目补齐模式识别、检索式视觉任务和深度学习实验能力。

## 2. Baseline

Baseline 采用 ReID 中常用的强基线配置：

- Backbone: `ResNet50`
- Head: `BNNeck`
- Loss: `CrossEntropy Loss + Triplet Loss`
- Metric: `Rank-1 / mAP`
- Datasets: `Market-1501`, `CUHK03-NP`

该 baseline 的意义是避免从弱模型上堆技巧。后续所有局部分支、Transformer 和融合模块都必须在强基线之上比较，结论才有说服力。

## 3. 方法设计

核心方法是全局-局部特征融合：

- 全局分支从 backbone 后期特征图中提取整图身份表征。
- 局部分支将后期特征图按垂直方向划分为多个局部区域 token，对应头部、上身、下身等人体局部区域的近似语义。
- Transformer 分支用于建模局部 token 之间的关系，缓解单个局部区域受遮挡、错位、姿态变化影响的问题。
- 融合阶段采用受控残差方式，将局部特征以较小权重注入全局特征，避免破坏强全局表征。
- 局部分支加入辅助 `BNNeck + classifier`，并用 `0.3 * (Local CE + Local Triplet)` 进行直接监督，避免局部分支被主分支忽略。

最终不是简单宣称“Transformer 更强”，而是围绕一个更扎实的问题展开：在强 ReID baseline 上，局部特征怎样才能真正有效参与检索？

## 4. 关键实验结果

### Market-1501

| Experiment | Rank-1 | mAP | Conclusion |
| --- | ---: | ---: | --- |
| `market1501_baseline` | `93.32%` | `82.93%` | 强 baseline |
| `market1501_local_branch` | `90.29%` | `75.93%` | 直接局部融合明显破坏全局表征 |
| `market1501_local_residual` | `93.41%` | `82.85%` | Learnable residual 基本压制局部分支 |
| `market1501_transformer_residual` | `93.17%` | `82.79%` | 无局部辅助监督时 Transformer 不稳定 |
| `market1501_gated_residual` | `93.26%` | `82.67%` | 单独 gating 不能解决局部分支无效问题 |
| `market1501_local_aux_residual` | `93.62%` | `83.81%` | 最优 mAP，较 baseline mAP +0.88 |
| `market1501_transformer_aux_residual` | `93.65%` | `83.73%` | 最优 Rank-1，较 baseline Rank-1 +0.33 |
| `market1501_full_aux_gated_residual` | `93.26%` | `83.27%` | mAP 高于 baseline，但弱于固定残差融合 |

### CUHK03-NP

| Experiment | Rank-1 | mAP | Conclusion |
| --- | ---: | ---: | --- |
| `cuhk03_np_baseline` | `61.43%` | `59.52%` | 更难 benchmark 上的强基线 |
| `cuhk03_np_local_residual` | `59.50%` | `57.45%` | Learnable residual 继续失效 |
| `cuhk03_np_local_aux_residual` | `63.79%` | `60.29%` | 最优 Rank-1，较 baseline +2.36 |
| `cuhk03_np_transformer_aux_residual` | `62.86%` | `61.19%` | Transformer 提升 mAP |
| `cuhk03_np_full_aux_gated_residual` | `63.29%` | `61.41%` | 最优 mAP，较 baseline +1.89 |

## 5. 关键发现

最重要的实验发现不是某个模块“绝对有效”，而是发现并修正了局部分支失效问题。

早期 learnable residual 方案中，局部分支的残差尺度会收缩到接近 0，例如 Market-1501 上多个局部残差模型的 `local_residual_scale` 约为 `1e-8` 量级。这说明模型虽然结构上有局部分支，但训练后几乎没有真正使用局部特征。

修正策略是：

- 使用固定小权重残差，保证局部特征稳定参与融合。
- 给局部分支增加辅助分类和度量学习监督，迫使局部 token 本身具备身份判别能力。
- 通过消融实验分别比较 baseline、无监督局部分支、辅助监督局部分支、Transformer 和 gated fusion。

这个过程比单纯堆模块更有含金量，因为它体现了实验诊断能力：发现模块无效、解释原因、提出修正、用多数据集验证。

## 6. 面试可讲技术点

### Triplet Loss 为什么需要

CrossEntropy 主要把样本分类到训练身份 ID 上，优化的是分类边界；Triplet Loss 直接优化特征空间中的相对距离，让同一行人更近、不同人更远。ReID 是检索任务，测试身份通常不在训练集中，因此特征空间的距离结构比单纯分类准确率更关键。

### BNNeck 的作用

BNNeck 将用于分类的特征和用于检索的特征做一定解耦。分类器更适合使用 BN 后特征稳定优化 CE，而检索时通常使用 BN 前或规范化后的 embedding 计算距离。这样可以缓解 CE 和 Triplet 对特征分布要求不完全一致的问题。

### Rank-1 和 mAP 的含义

Rank-1 表示每个 query 的检索列表中第一张 gallery 图片是否匹配正确，更关注 top-1 命中。mAP 考虑所有正确匹配在整个排序列表中的位置，更能反映检索列表整体质量。实验中出现 Rank-1 和 mAP 最优模型不同是正常现象。

### 局部分支解决什么问题

全局特征容易受遮挡、姿态变化、检测框错位和局部相似背景影响。局部分支把人体拆成多个区域，让模型能利用衣服纹理、背包、裤子、鞋子等局部线索。当局部区域被遮挡时，其他局部 token 仍可能提供判别信息。

### Transformer 在本项目中解决什么问题

Transformer 不是为了追求复杂结构，而是用于建模局部区域之间的关系。例如上衣和裤子颜色组合、背包与上身区域的关联、局部 token 之间的上下文一致性。实验结果显示，Transformer 在 CUHK03-NP 上更明显提升 mAP，在 Market-1501 上主要带来轻微 Rank-1 改善。

## 7. 简历表达建议

可写成以下 bullet：

- 基于 PyTorch 构建行人重识别实验框架，复现 `ResNet50 + BNNeck + CE + Triplet` 强基线，并在 `Market-1501` 和 `CUHK03-NP` 上完成 Rank-1/mAP 评测与消融实验。
- 设计全局-局部双分支 ReID 模型，从 backbone 后期特征图提取局部区域 token，并引入轻量 Transformer 建模局部区域关系，分析其对 top-1 命中率和检索列表质量的影响。
- 发现 learnable residual 融合中局部分支权重坍缩至接近 0，提出固定残差融合与局部辅助 `CE + Triplet` 监督，使 Market-1501 mAP 从 `82.93%` 提升至 `83.81%`，CUHK03-NP Rank-1 从 `61.43%` 提升至 `63.79%`。
- 通过 `baseline / local branch / Transformer / full model` 多组消融和检索可视化，验证局部辅助监督对 ReID 局部特征有效性的关键作用，并总结 Rank-1 与 mAP 指标差异。

更稳妥的口径是：“我不是追 SOTA，而是围绕强基线做了完整的实验诊断和模块有效性验证。”不要说“Transformer 全面提升”，因为当前实验不支持这个绝对结论。

## 8. 下一步

当前两数据集已经能支撑一份有质量的保研项目说明。下一步建议优先做两件事：

- 检索可视化成功/失败案例已生成，详见 `docs/retrieval_case_analysis.md`。
- 准备 `MSMT17` 作为更大规模泛化验证。如果时间和显存允许，再跑 baseline 与最佳 corrected variant；如果时间紧，先把 Market-1501 和 CUHK03-NP 的分析写扎实。
