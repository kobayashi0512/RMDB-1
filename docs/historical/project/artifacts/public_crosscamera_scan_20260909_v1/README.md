# 公开跨相机确认队列检索（2026-09-09）

目的：在**不联系数据持有方**、不申请受限访问的前提下，寻找一个可直接下载、可由同一眼配对、含相机/设备字段的高分辨率眼底队列，用于跨相机测量确认。

## 预先指定的可用条件

1. 图像可公开直接取得；不需要联系人、审核或受限访问审批。
2. 至少可以用 `patient_id + eye + visit/session + camera/device` 建立同一眼的跨相机配对。
3. 原始图像而不是仅特征向量；原生图像具有高分辨率。
4. 该队列没有参与本项目任何模型训练或读出选择。

## 结论

本轮公开检索没有找到同时满足四项条件的可直接接入队列。因此没有伪造“跨相机确认”实验，也没有用数字降采样替代光学/相机变化。

| 候选 | 有什么 | 排除原因 | 结论 |
|---|---|---|---|
| FLUID | 纵向、多设备（OPTOS/EIDON）、含病人/眼别/访视/设备元数据 | Zenodo 记录当前为 **restricted**，公开 API 没有可下载文件；在“不联系人、不申请访问”的约束下不可用 | 拒绝 |
| BRSET v1.0.2 | Canon CR-2 与 Nikon NF505 相机字段、16,266 图像、公开 | 官方说明每位患者只保留一次黄斑中心配对检查；没有同眼跨相机复拍，因此设备比较会被个体/病情混杂 | 拒绝作为跨相机测量确认 |
| HYGD | 2576×1934 或 1960×1934，高分辨率，多图/患者 | 单一 TOPCON DRI OCT Triton，相机不变；并且没有眼别字段 | 拒绝 |
| HRF IQA allQuality | 18 个同眼坏质量→好质量高分辨率配对，Canon CR-1 | 单相机；可用作已完成的质量压力测试，不能用于跨相机结论 | 已使用，边界锁定 |

## 对后续实验的影响

- 已完成的 HRF 质量压力和数字分辨率敏感性结果仍然有效，但分别仅回答“同相机质量扰动”和“数字输入分辨率”问题。
- 在没有合格公开队列时，项目可继续保持“readout 暴露终点特异性 trade-off”的结论；跨相机临床可互换性保持为未解决的外部效度缺口。
- 任何未来新增队列必须先通过 `experiments/measurement_confirmation_v1/cohort_eligibility_audit.py`；不得与已经冻结的 HRF 结果混合或覆盖。

## 来源（检索时状态）

- FLUID Zenodo record/API: <https://zenodo.org/records/19241555>
- BRSET v1.0.2 官方数据说明：<https://www.physionet.org/content/brazilian-ophthalmological/1.0.2/> 
- HYGD 官方数据说明：<https://www.physionet.org/content/hillel-yaffe-glaucoma-dataset/1.0.0/Images/>
- HRF 官方数据说明：<https://www5.cs.fau.de/research/data/fundus-images/>
