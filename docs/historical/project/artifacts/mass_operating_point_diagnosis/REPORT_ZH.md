# RMDB：排序改进还是工作点变化？一次固定诊断

状态：已完成。30 张 RAV 开发图像确定简单偏置；100 张 Fundus-AVSeg 图像、4 个既有方法完成全阈值排序与工作点分析。没有新训练，没有读取 CIMT 或临床结局。

## 1. 结论

Mass UDCT 的血管决策排序 AP 低于原始 VascX。默认阈值 Dice 的小幅改善不能据此解释为整体排序能力提升。

简单RAV偏置已把Fundus默认Dice从0.814554提高到0.818543；Mass UDCT为0.820036。Mass在该简单对照之外只剩Δ=+0.001493，95% CI [+0.000740, +0.002228]。这不是说Mass与偏置完全等价，而是说明不能忽略工作点因素。

在逐图精确率至少90%的描述性前沿上，Mass相对VascX的召回差为-0.013916，95% CI [-0.014795, -0.012991]。默认阈值下的细血管召回优势必须结合此误检控制结果解释。

当前结果不支持将该冻结修正器作为“显著提高血管识别能力”的核心创新。按约定停止这条增益搜索分支，不继续在Fundus调阈值或修改损失寻找正结果。若继续论文，应将其定位为测量与评价研究，保留该方法作为透明消融，而不是优越的新分割网络。

也不能隐藏局部正结果：Mass在90%和95%精确率点的AV像素比例误差较小，但这是次要、探索性、工作点相关的差异；85%点未体现稳定优势。它不足以替代失败的主终点或证明临床用途。

**本次不挑选新的 Fundus 最优部署阈值，也不把描述性的 oracle 阈值当成可部署算法。所有结果为探索性。**

## 2. 冻结的简单对照

只在已有 RAV 开发划分上，以 mean A/V Dice 选择统一前景 logit 偏置。网格为 −3 至 +3、步长 0.1，冻结 b=0.2。开发集默认 Dice=0.808247；加偏置后=0.808596。

这相当于同时给动脉、静脉、交叉类别加相同偏置，只改变前景与背景的取舍，不改变前景内部类别的大小关系。没有拟合空间模型，也不是新网络。RAV 已用于既往开发，不是独立验证。

## 3. 冻结规则的实际比较（统一768评分网格；不代表已获临床部署验证）

|规则|mean AV Dice|精确率|召回率|细血管召回率|clDice|AV比例对数误差↓|
|---|---:|---:|---:|---:|---:|---:|
|VascX 默认|0.814554|0.929925|0.780616|0.734491|0.892555|0.071480|
|VascX＋RAV冻结偏置|0.818543|0.923674|0.793927|0.750988|0.896323|0.071550|
|Mass UDCT 默认|0.820036|0.899763|0.820813|0.791387|0.902237|0.071866|
|固定 CLAHE＋VascX|0.828289|0.905807|0.830323|0.773281|0.902289|0.075412|
|固定 CLAHE＋Mass UDCT|0.826181|0.870932|0.865304|0.824696|0.908511|0.072755|

**不可用0.55作此处基线。原有0.55→0.8左右包含坐标修复，绝不是算法提高；512与768尺度的增量分别报告。**

## 4. 不依赖单一阈值的排序

|方法|决策排序 AP（主终点）|血管边际概率 AP|平均动静脉边际概率 AP|
|---|---:|---:|---:|
|VascX|0.947775|0.947898|0.908935|
|CLAHE + VascX|0.948180|0.948698|0.909117|
|Mass UDCT|0.943795|0.943663|0.903940|
|CLAHE + Mass|0.945032|0.945325|0.904608|

主终点对每张图像使用 score=log(max(PA,PV,PC))−log(Pbackground)，求精确的非插值 average precision，再跨图平均。统一前景常数偏置只给 score 加常数，不改变排序，因此不能改变其 AP。AP 仍依赖标注口径与患病/像素分布，不等于临床判别性能。

血管边际概率为1−Pbackground；动静脉边际概率分别为PA+PC、PV+PC。这两个辅助 AP 与主终点不可混称。

## 5. 精确率至少90%时的描述性比较

在每张图像全部经验阈值中选取精确率≥90%条件下召回率最高的点。四组都采用同样规则。此处消耗评价标注，只是 PR 前沿诊断；每张图像阈值不同，绝不能拿去宣称部署成绩。相同精确率控制的是假发现比例，不等于相同假阳性像素绝对数。

|方法|实际精确率|召回率|细血管召回率|clDice|AV比例对数误差↓|不可达到图像数|
|---|---:|---:|---:|---:|---:|---:|
|VascX|0.900007|0.829574|0.789402|0.897250|0.076764|0|
|CLAHE + VascX|0.900007|0.833369|0.773584|0.895194|0.080565|0|
|Mass UDCT|0.900007|0.815658|0.781248|0.894688|0.075474|0|
|CLAHE + Mass|0.900006|0.821631|0.769706|0.893471|0.077716|0|

85%与95%精确率的全部结果保存在逐图表与summary.json中，不从三个点里挑一个重新定义主终点。

## 6. 关键配对差及95%区间

5000次图像级配对bootstrap，差值为前者减后者；AV比例误差负值有利。区间未校正多重比较，且未包含先前模型筛选与RAV阈值选择的不确定性。

|比较|终点|均值差|95% CI|
|---|---|---:|---|
|mass_minus_vascx|decision_ap|-0.003980|[-0.004242, -0.003707]|
|mass_minus_vascx|vessel_marginal_ap|-0.004236|[-0.004504, -0.003958]|
|mass_minus_vascx|mean_av_marginal_ap|-0.004996|[-0.005224, -0.004774]|
|mass_minus_vascx|default/mean_av_dice|+0.005482|[+0.004466, +0.006480]|
|mass_minus_vascx|oracle_P0.90/vessel_recall|-0.013916|[-0.014795, -0.012991]|
|mass_minus_vascx|oracle_P0.90/thin_recall|-0.008154|[-0.009310, -0.006908]|
|mass_minus_vascx|oracle_P0.90/cldice|-0.002561|[-0.002973, -0.002114]|
|mass_minus_vascx|oracle_P0.90/av_ratio_log_error|-0.001290|[-0.002616, -0.000030]|
|mass_default_minus_vascx_RAV_frozen_bias|mean_av_dice|+0.001493|[+0.000740, +0.002228]|
|mass_default_minus_vascx_RAV_frozen_bias|vessel_precision|-0.023911|[-0.024478, -0.023374]|
|mass_default_minus_vascx_RAV_frozen_bias|vessel_recall|+0.026886|[+0.026040, +0.027749]|
|mass_default_minus_vascx_RAV_frozen_bias|thin_recall|+0.040398|[+0.038738, +0.042112]|
|mass_default_minus_vascx_RAV_frozen_bias|cldice|+0.005914|[+0.005094, +0.006725]|
|mass_default_minus_vascx_RAV_frozen_bias|av_ratio_log_error|+0.000317|[-0.002199, +0.003005]|
|clahe_mass_minus_clahe|decision_ap|-0.003148|[-0.003443, -0.002834]|
|clahe_mass_minus_clahe|default/mean_av_dice|-0.002109|[-0.003027, -0.001233]|
|clahe_mass_minus_clahe|oracle_P0.90/thin_recall|-0.003878|[-0.005011, -0.002655]|
|clahe_mass_minus_clahe|oracle_P0.90/cldice|-0.001724|[-0.002220, -0.001161]|

## 7. 各指标能说明什么，不能说明什么

- 细血管定义固定为768网格上标注骨架距离变换半径≤2像素；不等同于临床按微米测量的小血管。
- clDice 衡量骨架与血管区域的相互覆盖，能辅助检查连接结构，但不保证正确的拓扑，也不能替代人工核查每条分支。
- AV比例误差是硬分割的动静脉像素数比值对数误差，不是临床管径AVR。交叉点计入动静脉两者，未知身份像素不计入身份评分。
- 统一常数偏置只是一个明确、低复杂度的替代解释；击败它不等于击败所有后处理方法。没击败它也不能自动证明完全等效。
- 简单基线的偏置在RAV选择，目标域可能具有不同工作点；它在Fundus的结果必须结合阈值无关排序与oracle前沿解读。
- 未声称提升临床结局预测、泛化到新医院或证明重大新算法。没有新增模型调参或预处理参数搜索。

## 8. 图注

Diagnostic curves：A为逐图插值精确率包络的宏平均，B为高召回区域放大。图形仅在1001个召回网格点展示，主表AP由全部经验阈值精确计算，不是插值曲线下面积。C、D是在每张图像精确率分别至少85%、90%、95%的oracle点计算细血管召回和clDice后求宏平均。所有四组均展示，不隐去组合失败结果。

## 9. 完整性与复现

独立检查：400份概率、1700个工作点。独立AP最大绝对误差3.33e-16；逐掩模指标最大误差0。所有原图/标注哈希与前一轮一致，主终点bootstrap已复算。

- protocol.json：在本次评价前固定的分析口径。
- frozen_baseline.json、development_bias_grid.csv：RAV选择记录与冻结偏置。
- ranking_per_image.csv、operating_points_per_image.csv、summary.json：全部逐图数据和配对差。
- diagnostic_curves.svg/pdf/png：由真实数值渲染，可编辑矢量图。
- independent_audit.json：独立计数、排序AP、最佳可达召回与哈希验证。
- src/diagnose_mass_operating_point.py、test_mass_operating_point.py、audit_mass_operating_point.py、report_mass_operating_point.py：全流程源码。

复现命令（项目目录内）：
```sh
.venv/bin/python src/test_mass_operating_point.py
.venv/bin/python src/diagnose_mass_operating_point.py prepare
.venv/bin/python src/diagnose_mass_operating_point.py develop
.venv/bin/python src/diagnose_mass_operating_point.py evaluate
.venv/bin/python src/audit_mass_operating_point.py
.venv/bin/python src/report_mass_operating_point.py
```
