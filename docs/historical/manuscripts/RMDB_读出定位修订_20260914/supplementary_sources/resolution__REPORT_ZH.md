# HRF IQA 数字分辨率敏感性实验

## 状态与边界

本实验已完成并通过完整性审计。它使用 HRF IQA 子集的 18 张此前未被本项目分析过的高分辨率好质量原图；每张图保留原图，并以固定的 LANCZOS、JPEG quality=100、无色度子采样生成 75% 与 50% 两档输入。共 54 个输入、5 个冻结 readout、270 次读出和 3240 个特征值。

这是受控的**数字降采样敏感性**实验：它隔离了输入像素减少的影响，但不能模拟相机光学、曝光、视场或真实跨相机差异。因此不能作为跨相机临床可迁移性证据，也不与 HRF IQA 的 bad/good 质量扰动实验或历史结果合并。

## 预定主端点：joint 相对 raw VascX

每项比较均以同一原始图为配对单位（18 对），报告 ICC(1,1)、MAD、SEM、Bland–Altman LoA；区间为 10,000 次配对 bootstrap、三端点 Bonferroni 后 98.333% 中心区间。MAD、SEM 和 LoA 半宽比小于 1 才表示 joint 对该分辨率扰动更不敏感。

|输入比较|端点|ΔICC [CI]|MAD 比|SEM 比|LoA 半宽比|Gate|
|---|---|---|---:|---:|---:|---|
|原图 vs 75%|branch_density|+0.0254 [+0.0006, +0.0744]|0.7485|0.6996|0.6709|mixed_tradeoff|
|原图 vs 75%|endpoint_density|−0.0821 [−0.5213, +0.0101]|0.8462|0.9827|1.0439|not_improved|
|原图 vs 75%|occupancy_entropy|−0.0003 [−0.0088, +0.0032]|2.9829|2.3737|2.2151|not_improved|
|原图 vs 50%|branch_density|+0.0468 [−0.0119, +0.1963]|0.8039|0.6831|0.6307|not_improved|
|原图 vs 50%|endpoint_density|−0.0775 [−0.4856, +0.0077]|0.8389|0.8832|1.1055|not_improved|
|原图 vs 50%|occupancy_entropy|−0.0021 [−0.0183, +0.0003]|3.9167|3.5996|2.4671|not_improved|

## 结论

降采样不会产生单向、通用的 joint 优势。75% 下 branch density 的 ICC 向好，但绝对一致性区间未能同时满足 Gate；50% 下它也没有通过。两档降采样中 occupancy entropy 的 MAD、SEM 和 LoA 都明显恶化。这与质量扰动压力测试一起，支持“readout 暴露/分离 measurement trade-off”，不支持“RMDB 解决分辨率敏感性”或“joint 全面优于 VascX”。

完整逐图数据：`per_image_features.csv`、`feature_values_long.csv`；两份锁定 Gate：`gate_native_vs_75pct.json`、`gate_native_vs_50pct.json`；审计：`audit.json`。
