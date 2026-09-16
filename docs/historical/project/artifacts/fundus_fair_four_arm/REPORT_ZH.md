# Fundus-AVSeg：原始分数追溯与公平四组对照

状态：100 张图像、四组共 400 个评估全部完成。数据集已被反复查看，本次为探索性审计与对照，不是新的独立确认。

## 1. 先纠正比较的基础

旧流程把经过视野裁剪的预测图与仅缩放、未应用同一裁剪的原始标注比较。新流程逐图重放保存在 bounds_latest_batch.csv 中的图像裁剪，100 张 RGB 均与缓存逐像素完全一致；同一变换以最近邻应用于标注。没有用标注拟合配准。

另一处问题来自近期脚本复用 RAV 的颜色解码。Fundus-AVSeg 的绿色是动静脉交叉点，不应忽略；白色是无法判断动静脉类型的血管。现在绿色对动脉和静脉均计入，白色仅从身份评价剔除，仍计入血管检出。该解释与原始 Fundus 评估器一致。

标注依据：[数据集原论文](https://www.nature.com/articles/s41597-025-05381-2)。

|512 像素追溯，使用同一缓存预测和权重|VascX mean AV Dice|Mass UDCT mean AV Dice|
|---|---:|---:|
|旧错误坐标，仅用于复现|0.554470|0.563484|
|仅修正坐标，同一 512 评分尺度|0.813999|0.829771|

修正坐标后的 512 缓存评估：Mass 相对 VascX 的配对差为 +0.015772，95% CI [+0.014877, +0.016662]。这一分辨率敏感性结果与下方 768 四组实验分别呈现，不混作同一设置。

复现论文四位小数：{'vascx': True, 'mass': True}。原始记录已从 GitHub 只读恢复到 ARCHIVED_20260729_original_report.md（若网络恢复成功）。
**不能把 0.55 与修正后的 0.8 左右作算法前后对比。坐标修复、标注口径修复与评分分辨率变化都不是模型增益。旧报告根据该错误评分作出的失败域、增益及通过判定也必须重新审查。**

## 2. 四组公平结果

本表统一为 1024 输入，五成员冻结 VascX，512 分块、0.5 重叠、不启用 TTA；768 评分网格，统一正确坐标与标注解释。CLAHE 固定为 LAB 亮度通道 clip=1.5、8×8 网格、与原图各占 0.5；没有搜索参数。Mass UDCT 使用已冻结原权重、原 11 通道特征与两步修正，并未重新训练。此处 RMDB 分支仅指 Mass UDCT，不等同于完整 BiRAP 患者级流程。

|方法|mean AV Dice ↑|血管 Dice ↑|血管精确率 ↑|血管召回率 ↑|细血管中心线召回率 ↑|AV 比例对数误差 ↓|四类平均 Brier ↓|
|---|---:|---:|---:|---:|---:|---:|---:|
|VascX|0.814554|0.847989|0.929925|0.780616|0.734491|0.071480|0.011555|
|VascX＋固定 CLAHE|0.828289|0.865633|0.905807|0.830323|0.773281|0.075412|0.010886|
|VascX＋冻结 Mass UDCT|0.820036|0.857775|0.899763|0.820813|0.791387|0.071866|0.011499|
|VascX＋固定 CLAHE＋冻结 Mass UDCT|0.826181|0.867396|0.870932|0.865304|0.824696|0.072755|0.011416|

## 3. 配对增量与误检代价

差值均为前者减后者；Brier 和比例误差的负值有利。置信区间为图像配对 bootstrap 5000 次百分位区间，未作多重比较校正，也未计入此前挑选 CLAHE 候选的选择不确定性；不能据此作确认性显著结论。数据集论文说明 100 张图来自 100 名患者；这里不进行临床推断。

|对照|指标|平均差|95% CI|改善图像数|
|---|---|---:|---|---:|
|mass_minus_vascx|mean_av_dice|+0.005482|[+0.004450, +0.006481]|82/100|
|mass_minus_vascx|vessel_dice|+0.009787|[+0.008572, +0.010980]|97/100|
|mass_minus_vascx|vessel_precision|-0.030162|[-0.030894, -0.029431]|0/100|
|mass_minus_vascx|vessel_recall|+0.040197|[+0.039099, +0.041324]|100/100|
|mass_minus_vascx|thin_centerline_recall|+0.056896|[+0.054844, +0.059043]|100/100|
|mass_minus_vascx|av_ratio_log_error|+0.000386|[-0.002701, +0.003451]|53/100|
|mass_minus_vascx|brier|-0.000055|[-0.000131, +0.000025]|58/100|
|clahe_minus_vascx|mean_av_dice|+0.013736|[+0.012124, +0.015360]|94/100|
|clahe_minus_vascx|vessel_dice|+0.017644|[+0.016141, +0.019092]|100/100|
|clahe_minus_vascx|vessel_precision|-0.024117|[-0.025459, -0.022778]|0/100|
|clahe_minus_vascx|vessel_recall|+0.049707|[+0.048175, +0.051181]|100/100|
|clahe_minus_vascx|thin_centerline_recall|+0.038790|[+0.037003, +0.040565]|100/100|
|clahe_minus_vascx|av_ratio_log_error|+0.003932|[+0.000432, +0.007481]|45/100|
|clahe_minus_vascx|brier|-0.000669|[-0.000782, -0.000556]|87/100|
|clahe_mass_minus_clahe|mean_av_dice|-0.002109|[-0.003025, -0.001181]|37/100|
|clahe_mass_minus_clahe|vessel_dice|+0.001763|[+0.000688, +0.002838]|59/100|
|clahe_mass_minus_clahe|vessel_precision|-0.034876|[-0.035568, -0.034181]|0/100|
|clahe_mass_minus_clahe|vessel_recall|+0.034981|[+0.033907, +0.036091]|100/100|
|clahe_mass_minus_clahe|thin_centerline_recall|+0.051415|[+0.049510, +0.053351]|100/100|
|clahe_mass_minus_clahe|av_ratio_log_error|-0.002657|[-0.005834, +0.000578]|62/100|
|clahe_mass_minus_clahe|brier|+0.000531|[+0.000453, +0.000613]|10/100|
|clahe_mass_minus_mass|mean_av_dice|+0.006144|[+0.004664, +0.007632]|78/100|
|clahe_mass_minus_mass|vessel_dice|+0.009621|[+0.008342, +0.010846]|92/100|
|clahe_mass_minus_mass|vessel_precision|-0.028831|[-0.029977, -0.027675]|0/100|
|clahe_mass_minus_mass|vessel_recall|+0.044491|[+0.043128, +0.045822]|100/100|
|clahe_mass_minus_mass|thin_centerline_recall|+0.033309|[+0.031624, +0.034992]|100/100|
|clahe_mass_minus_mass|av_ratio_log_error|+0.000889|[-0.002322, +0.004349]|58/100|
|clahe_mass_minus_mass|brier|-0.000083|[-0.000195, +0.000028]|55/100|
|clahe_mass_minus_vascx|mean_av_dice|+0.011627|[+0.009229, +0.013959]|79/100|
|clahe_mass_minus_vascx|vessel_dice|+0.019407|[+0.016995, +0.021717]|95/100|
|clahe_mass_minus_vascx|vessel_precision|-0.058993|[-0.060629, -0.057373]|0/100|
|clahe_mass_minus_vascx|vessel_recall|+0.084688|[+0.082656, +0.086721]|100/100|
|clahe_mass_minus_vascx|thin_centerline_recall|+0.090205|[+0.087109, +0.093305]|100/100|
|clahe_mass_minus_vascx|av_ratio_log_error|+0.001275|[-0.002691, +0.005408]|53/100|
|clahe_mass_minus_vascx|brier|-0.000138|[-0.000322, +0.000048]|58/100|

## 4. 本轮结论

冻结 Mass UDCT 加在固定 CLAHE 之后使 Dice 下降。本轮不支持组合路线，不应继续在本数据上调参寻找正结果。

预处理的 Dice 改善不能说明恢复了原本不存在的真实血管，也不能只凭更清晰的外观判断有效。必须同时看召回与精确率，并与原始标注比较。

## 5. 预定义分层

|分层|图像数|VascX|CLAHE|Mass|CLAHE＋Mass|
|---|---:|---:|---:|---:|---:|
|image quality: High-quality|83|0.817238|0.830931|0.822443|0.828587|
|image quality: Low-quality|17|0.801449|0.815395|0.808285|0.814434|
|disease type: AMD|20|0.826492|0.840051|0.831724|0.837150|
|disease type: DR|20|0.812504|0.828279|0.818344|0.826631|
|disease type: Glaucoma|20|0.821252|0.835011|0.827579|0.833454|
|disease type: Normal|40|0.806260|0.819053|0.811267|0.816834|

探索性交互差 [(CLAHE＋Mass)−CLAHE]−[Mass−VascX]：ΔDice=-0.007591，95% CI [-0.007922, -0.007253]。它描述组合效应是否可加，不单独证明任何生物学或算法机制。

## 6. 指标解释与限制

- mean AV Dice 为逐图动脉、静脉 Dice 均值再跨图平均，不是像素准确率。
- 细血管定义为标注血管骨架上距离变换半径≤2 个像素的点，在 768 尺度固定；这是敏感性端点，不是临床统一的细血管定义。
- AV 比例误差为硬预测与标注的动脉/静脉像素数比值之对数差绝对值，+1 平滑；不等同于临床血管管径 AVR，也不等同于软概率质量比。
- Brier 对四类取平均并跨有效像素平均，含大量背景；不是某些文献使用的按类别求和 Brier。
- CLAHE 来自前一轮探索后的候选，组合是未重新训练的冻结修正器。不证明通用域泛化，不证明临床收益。
- 未读取 CIMT 固定测试/验证或任何临床结局。本轮不新增模型搜索；后续确认需要未使用样本。
- 原始论文旧分数和近期忽略绿色交叉点的分数分别保留为历史记录，不得混入新结果表。

## 7. 可复现文件

- protocol.json：实验口径与全部图像 ID。
- per_image_metrics.csv、summary.json：逐图和汇总结果。
- legacy_reproduction.csv/json：旧流程复现及仅修正坐标的比较。
- logits/、probabilities/、predictions/：200 份输入 logits、400 份概率和真实推理分割。
- sha256_manifest.json、output_sha256.json、output_integrity.json：来源校验及全部有限输出核查。
- 源码：src/run_fundus_fair_four_arm.py；回归测试：src/test_fundus_label_convention.py。
