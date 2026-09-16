# HRF IQA 高分辨率读出压力测试

## 状态

已完成并通过完整性审计：True。本报告不是论文结果表，也不与既往 mBRSET 或 Fundus-AVSeg 数值合并。

## 数据与边界

- 官方 HRF Image Quality Assessment 子集：36 张原图、18 个已公布同眼配对；原始分辨率为 3888x2592, 5184x3456。
- 每对均为 Canon CR-1 的低质/失焦首拍与改善后的复拍。它是采集质量扰动测试，而非同条件 test–retest；没有跨相机配对。
- 没有血管人工标注，因此本轮不评价结构准确性；只测试冻结 readout 对采集质量变化的测量敏感性。
- 36 图、5 个读出、12 个特征，未训练、未调参、未读取临床结局。

## 预先指定的三项测量端点

数值为 bad→good 配对差异。CI 为以 18 个配对 ID 为聚类单位的 10,000 次 bootstrap、三端点 Bonferroni 后的 98.333% 中心区间。LoA 半宽比、MAD 比和 SEM 比均为 `joint/raw VascX`，小于 1 表示 joint 对质量扰动更不敏感。

|端点|raw ICC|joint ICC|ΔICC [CI]|MAD 比 [CI]|SEM 比 [CI]|LoA 半宽比 [CI]|Gate|
|---|---:|---:|---|---|---|---|---|
|branch_density|-0.6503|-0.2516|0.3986 [0.0868, 0.7330]|1.0650 [0.8611, 1.3133]|1.0703 [0.8925, 1.3134]|1.0817 [0.8099, 1.6030]|mixed_tradeoff|
|endpoint_density|-0.7300|-0.3762|0.3537 [0.0772, 0.7186]|0.2609 [0.1954, 0.3280]|0.2816 [0.2160, 0.3406]|0.3965 [0.2968, 0.5482]|concordant_relative_improvement|
|occupancy_entropy|0.7958|0.2844|-0.5113 [-1.1349, -0.1105]|4.2009 [2.3405, 6.7862]|4.4656 [2.4709, 7.1285]|3.1481 [1.7574, 5.2511]|not_improved|

## 读法

- `endpoint_density`：joint 的 ΔICC 为正，MAD、SEM、LoA 半宽比均显著小于 1，属于在该质量扰动下的协同相对改善。
- `branch_density`：ICC 的区间改善，但 MAD、SEM、LoA 没有同时改善，属于 mixed trade-off，不能声称稳定性整体提升。
- `occupancy_entropy`：joint 的 ICC、MAD、SEM 和 LoA 都更差，明确属于 not improved。

因此，这个独立高分辨率压力测试支持“不同 readout 暴露/分离 trade-off”，不支持“RMDB 解决了稳定性问题”或“joint 全面优于 VascX”。

所有五个读出、全部 12 个特征及原始逐图数值见 `per_image_features.csv`、`feature_values_long.csv` 和 `measurement_stress_gate.json`；审计见 `audit.json`。
