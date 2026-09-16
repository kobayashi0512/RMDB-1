# 血管结构约束图像恢复：固定小规模验证

状态：两种恢复网络完成 20 轮 FIVES 开发训练；RAV 30 张图像、两种条件、四种处理，共 240 个推理完成。本报告是可行性试验，不是临床验证或新算法优越性的最终证明。

## 设计与隔离

- 复用已存在的 FIVES 480/120 开发划分。原图是配对参考，不保证完全清晰；未读取 FIVES 官方 test。
- 两网络架构、初始化、训练批次和优化器相同；仅结构约束损失不同。每图每轮一个 192 像素 patch，共每分支 9600 个训练 patch。
- 普通模型使用重建 MAE。结构模型另加血管区域 MAE 与血管加权图像梯度误差；并非严格拓扑约束、不是 clDice，也未使用动静脉训练监督。
- 输出保留原图残差旁路，单像素修正上界 0.25；25% 训练样本无人工退化。限制修正幅度不保证没有假血管，因此另作像素评价。
- 检查点仅按 FIVES 固定验证 patch 的共同标准 MAE+0.2 梯度误差选择。RAV 结果不用于选择或调参。
- RAV 已被既往研究查看，且冻结 VascX 与其存在开发背景关联；不能称为全新独立外部队列。未读取 Fundus-AVSeg、CIMT 或 mBRSET。
- 固定中等退化：模糊σ1.3、缩小到65%再放大、噪声0.012、gamma1.3、不均匀照明0.25、JPEG65。未人工退化条件用于检查增强是否损伤现有信息。

每模型参数：118307。选择轮次：{'reconstruction': 20, 'structure': 20}。是否充分收敛需结合 history.json，20 轮只代表本轮固定预算，不保证达到性能上限。

## 结果

|条件|处理|mean AV Dice|血管精确率|血管召回率|细中心线召回|AV比例误差↓|四类均值Brier↓|参考MAE↓|参考PSNR↑|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|moderate|input|0.71074|0.85776|0.68036|0.43566|0.11551|0.01302|0.08541|19.72001|
|moderate|clahe|0.70497|0.82504|0.71413|0.46522|0.11905|0.01351|0.07521|20.80057|
|moderate|reconstruction|0.69180|0.86528|0.65431|0.41041|0.12321|0.01361|0.07683|21.12960|
|moderate|structure|0.69243|0.86490|0.65644|0.41288|0.11503|0.01358|0.07729|21.06024|
|undegraded|input|0.80825|0.86496|0.81191|0.75062|0.06235|0.00904|0.00000|120.00000|
|undegraded|clahe|0.80223|0.82886|0.84630|0.77661|0.07025|0.00958|0.01157|35.54730|
|undegraded|reconstruction|0.80695|0.86712|0.80838|0.74698|0.06281|0.00907|0.00392|47.85496|
|undegraded|structure|0.80711|0.86703|0.80858|0.74738|0.06376|0.00907|0.00372|48.35442|

## 主对照与约束贡献

|条件|对照|Δmean AV Dice|95% CI|
|---|---|---:|---|
|moderate|structure_minus_clahe|-0.01254|[-0.02525, -0.00180]|
|moderate|structure_minus_reconstruction|+0.00063|[-0.00094, +0.00188]|
|moderate|structure_minus_input|-0.01831|[-0.02979, -0.00863]|
|undegraded|structure_minus_clahe|+0.00489|[+0.00104, +0.00832]|
|undegraded|structure_minus_reconstruction|+0.00016|[-0.00021, +0.00062]|
|undegraded|structure_minus_input|-0.00114|[-0.00208, -0.00037]|

## 预先固定的可行性判断

- dice_gain_at_least_005: 未通过
- dice_lower_CI_positive: 未通过
- precision_loss_within_005: 通过
- av_error_increase_within_005: 通过
- undegraded_dice_loss_within_005: 通过

未通过全部守则。本轮停止该分支，不在 RAV 或 Fundus-AVSeg 上继续调参找正结果。

## 解读边界

人工退化可检验已知信息被损坏后的恢复，但不能证明能恢复真实拍摄时丢失的信息。重建更好与分割更好是不同终点。普通重建网络是同架构消融对照，不是 Cofe-Net 等成熟方法的替代复现；本轮不能声称优于这些方法。

置信区间是图像级配对 bootstrap 3000 次，未作多重校正；RAV 的患者独立性不能由文件名保证。两种退化条件不当作独立患者合并。PSNR 对零误差采用 1e-12 MSE 下限，120dB 是数值上限而非估计值。

CLAHE 配置固定，来源于之前探索；此处禁止把0.55错误坐标结果用作基线。所有 RAV 评分使用原始图像裁剪的同一几何变换，绿色未知类别只忽略身份、保留血管支持，红蓝交叠属于交叉点。

已有相关工作：[Cofe-Net](https://arxiv.org/abs/2005.05594)、[MAGE-Net](https://arxiv.org/abs/2302.11795)。本原型没有首次提出结构保持增强的资格声明。

复现文件：protocol.json、history.json、training_complete.json、evaluation/per_image.csv、evaluation/summary.json、integrity.json。图像与概率保存在 evaluation/inputs、evaluation/predictions、evaluation/probabilities。
