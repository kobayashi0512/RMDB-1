# VascX 增量定位（实验前固定）

## 比较对象

所有表中 `raw_vascx` 都是冻结的 VascX A/V 概率输出。`mass`、`joint`、`smooth`、`cleanup` 是其上游概率不变时的读出层/控制臂；它们没有重新训练或替换 VascX。

## 已经能说什么

1. 在历史 mBRSET 测试重复眼中，joint 相对 raw VascX 的三个预先指定读出（branch density、endpoint density、occupancy entropy）均有较高 ICC 和较小 MAD。这是历史、224px 同源重复性证据。
2. 在 100 张带标注的 Fundus-AVSeg 控制中，joint 的 vessel Dice 相对 raw VascX 下降；这排除了“joint 改善/替代 VascX 分割”的表述，并直接显示结构保真与测量读出存在取舍。
3. 在新完成的 HRF 同眼高分辨率质量压力测试中，endpoint density 是一致相对改善，branch density 为 mixed tradeoff，而 occupancy entropy 为 not improved。
4. 在同一 HRF good-capture 的数字降采样实验中，没有出现所有终点一致改善；尤其 occupancy entropy 两个比较均为 not improved。

## 结论边界

唯一可用的核心结论是：**在冻结 VascX 分割上，候选 readout 会分离并暴露不同测量终点的 trade-off。**

不得使用 “Resolves”、 “replaces VascX”、 “improves VascX segmentation”、 “cross-camera validated” 或 “clinically interchangeable”。跨相机、真实高分辨率重复采集以及临床可互换性仍未得到证据支持。
