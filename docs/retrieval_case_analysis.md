# 检索成功与失败案例分析

## 1. 可视化说明

案例图由 `scripts/analyze_retrieval_cases.py` 自动生成。每张图包含两行：

- 第一行：baseline 的 query 和 Top-5 gallery 检索结果。
- 第二行：目标模型的 query 和 Top-5 gallery 检索结果。
- 蓝框：query。
- 绿框：与 query 属于同一行人 ID 的正确匹配。
- 红框：错误匹配。

图标题中的 `first correct rank` 表示第一个正确匹配在过滤同 ID 同 camera 后的 gallery 排序中的位置。该值越小，说明正确目标越靠前。

## 2. Market-1501 案例分析

对比对象：

- Baseline: `market1501_baseline`
- Target: `market1501_local_aux_residual`
- Target 选择原因：Market-1501 上 mAP 最优，`83.81%`

统计结果：

| Item | Count |
| --- | ---: |
| Query 数量 | `3368` |
| Baseline Rank-1 correct | `3143` |
| Target Rank-1 correct | `3152` |
| Target 修复 baseline Rank-1 错误 | `86` |
| Target 相比 baseline Rank-1 回退 | `77` |
| 两者都 Rank-1 正确 | `3066` |
| 两者都 Rank-1 错误 | `139` |

代表性成功案例：

| Case | Baseline first correct rank | Target first correct rank | Visualization |
| --- | ---: | ---: | --- |
| `query_0100` | `18` | `1` | `outputs/case_analysis/market1501_local_aux_vs_baseline/recovered_00_query_0100.jpg` |
| `query_0321` | `9` | `1` | `outputs/case_analysis/market1501_local_aux_vs_baseline/recovered_01_query_0321.jpg` |
| `query_0674` | `9` | `1` | `outputs/case_analysis/market1501_local_aux_vs_baseline/recovered_02_query_0674.jpg` |
| `query_2125` | `8` | `1` | `outputs/case_analysis/market1501_local_aux_vs_baseline/recovered_03_query_2125.jpg` |

代表性失败案例：

| Case | Baseline first correct rank | Target first correct rank | Visualization |
| --- | ---: | ---: | --- |
| `query_2545` | `3433` | `8528` | `outputs/case_analysis/market1501_local_aux_vs_baseline/target_failures_00_query_2545.jpg` |
| `query_1147` | `1493` | `5851` | `outputs/case_analysis/market1501_local_aux_vs_baseline/target_failures_01_query_1147.jpg` |
| `query_0436` | `7212` | `2070` | `outputs/case_analysis/market1501_local_aux_vs_baseline/target_failures_02_query_0436.jpg` |
| `query_0380` | `603` | `626` | `outputs/case_analysis/market1501_local_aux_vs_baseline/target_failures_03_query_0380.jpg` |

结论：

- Market-1501 已经接近饱和，Rank-1 提升空间很小，因此新增局部分支的收益主要体现在 mAP。
- `local_aux_residual` 修复了部分 baseline 将外观相似行人排在第一的问题，说明局部辅助监督确实让局部区域特征参与了检索。
- 失败案例说明局部特征也可能放大局部相似干扰，例如衣服颜色、背包、姿态接近但身份不同的样本。

## 3. CUHK03-NP 案例分析

对比对象：

- Baseline: `cuhk03_np_baseline`
- Target: `cuhk03_np_full_aux_gated_residual`
- Target 选择原因：CUHK03-NP 上 mAP 最优，`61.41%`

统计结果：

| Item | Count |
| --- | ---: |
| Query 数量 | `1400` |
| Baseline Rank-1 correct | `862` |
| Target Rank-1 correct | `886` |
| Target 修复 baseline Rank-1 错误 | `127` |
| Target 相比 baseline Rank-1 回退 | `103` |
| 两者都 Rank-1 正确 | `759` |
| 两者都 Rank-1 错误 | `411` |

代表性成功案例：

| Case | Baseline first correct rank | Target first correct rank | Visualization |
| --- | ---: | ---: | --- |
| `query_0235` | `35` | `1` | `outputs/case_analysis/cuhk03_np_full_aux_gated_vs_baseline/recovered_00_query_0235.jpg` |
| `query_0439` | `30` | `1` | `outputs/case_analysis/cuhk03_np_full_aux_gated_vs_baseline/recovered_01_query_0439.jpg` |
| `query_0420` | `28` | `1` | `outputs/case_analysis/cuhk03_np_full_aux_gated_vs_baseline/recovered_02_query_0420.jpg` |
| `query_0399` | `21` | `1` | `outputs/case_analysis/cuhk03_np_full_aux_gated_vs_baseline/recovered_03_query_0399.jpg` |

代表性失败案例：

| Case | Baseline first correct rank | Target first correct rank | Visualization |
| --- | ---: | ---: | --- |
| `query_1391` | `481` | `1141` | `outputs/case_analysis/cuhk03_np_full_aux_gated_vs_baseline/target_failures_00_query_1391.jpg` |
| `query_1383` | `126` | `576` | `outputs/case_analysis/cuhk03_np_full_aux_gated_vs_baseline/target_failures_01_query_1383.jpg` |
| `query_0553` | `741` | `547` | `outputs/case_analysis/cuhk03_np_full_aux_gated_vs_baseline/target_failures_02_query_0553.jpg` |
| `query_1371` | `172` | `541` | `outputs/case_analysis/cuhk03_np_full_aux_gated_vs_baseline/target_failures_03_query_1371.jpg` |

结论：

- CUHK03-NP 难度明显高于 Market-1501，两个模型同时失败的 query 达到 `411` 个。
- 完整模型修复了 `127` 个 baseline 的 Rank-1 错误，说明局部 token、Transformer 关系建模和 gated residual 在更难数据集上更有价值。
- 失败案例集中体现了检测框质量、低分辨率、姿态变化和局部外观相似带来的困难，这些是后续引入更强对齐策略或鲁棒局部建模的合理动机。

## 4. 汇报建议

PPT 中建议放 4 张图：

- Market-1501 成功案例：`market1501_local_aux_vs_baseline/recovered_00_query_0100.jpg`
- Market-1501 失败案例：`market1501_local_aux_vs_baseline/target_failures_00_query_2545.jpg`
- CUHK03-NP 成功案例：`cuhk03_np_full_aux_gated_vs_baseline/recovered_00_query_0235.jpg`
- CUHK03-NP 失败案例：`cuhk03_np_full_aux_gated_vs_baseline/target_failures_00_query_1391.jpg`

讲解口径：

- “我没有只报最终指标，而是进一步比较了每个 query 的排序变化。”
- “目标模型在 Market-1501 上修复 86 个 baseline top-1 错误，在 CUHK03-NP 上修复 127 个 baseline top-1 错误。”
- “同时我保留了回退和失败案例，说明该方法不是无条件提升。失败主要来自检测框偏移、相似衣着、姿态变化和局部区域误导。”
- “这支持了项目的核心结论：局部特征需要辅助监督和受控融合，否则可能被忽略，也可能引入噪声。”
