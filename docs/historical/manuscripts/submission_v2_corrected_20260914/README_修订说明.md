# RMDB 图文同步校正版 · 2026-09-14

## 正式交付

- `RMDB_BiRAP_Corrected_Manuscript.docx`：完整论文，19 页，含六张同步修订图、六张正文/补充表。
- `RMDB_BiRAP_Corrected_Manuscript.pdf`：上述 Word 的版面预览。
- `figures/06_editable_pptx/RMDB_Figures_1-6_Corrected_Final.pptx`：六页可编辑论文图。第 2 页为完整模型框架。
- `figures/08_previews/final/`：与正式 PPT 对应的六张 PNG。
- `frozen_evidence/`：统一结果数据、五种 readout 的 60 行一致性结果、样本清点及输入哈希。
- `correction_log.json`：35 处段落/图注的前后对照；不含另行重建的表格与公式。
- `validation_report.json`：图文数量、表格一致性、原始结果哈希和 PPT 对象核对。
- `supplementary_sources/`：HRF 质量及分辨率敏感性实验的既有冻结报告。

## 本次实质性校正

1. 区分总划分与重复拍摄分析：测试划分 258 人；实际重复拍摄分析 255 人、475 眼、950 图。开发集重复贡献人数为 253。
2. 以实际代码修订 FOV、1024→512→256 网格、softmax 顺序、残差校正器、独立 occupancy 分支及两分支的并行关系；移除未经本次实现支持的视盘–黄斑轴配准描述。
3. A/V 是未取对数的概率质量比；交叉类同时计入动脉和静脉质量。并非临床血管口径 AVR。
4. 明确 mass-preserving 是 readout 名称，不表示残差更新严格守恒。联合 readout 使用原始身份概率，而非 mass 校正后的概率。
5. 同时报告 ICC、MAD、SEM、LoA；保留所有十二项特征。纠正把 MAD 明确恶化写成“不确定”的状态；identity entropy 为构造上的数值保持。
6. 结果图补入区间和全部五种 readout；未来双置信界 Gate 与本次探索性结果明确分开。
7. 结论限定为 separates/exposes the trade-off，明确对 VascX 的增量是测量评价，不是替代分割模型。
8. HRF 补充结果不冒充独立跨相机确认；AFIO 缺少本地配对链路。修正 mBRSET 原始论文引用，并区分原始 1600 px 与本次 224 px 本地镜像。

## PPT 编辑范围与视觉规范

延续浅蓝、香槟金和暖白底，少量青绿区分 occupancy 分支，暗红仅提示不利变化。按 scientific-figure-workflow 的可编辑交付要求，用原生文本、图形、连接线、坐标、点及置信区间绘图；非“整张图片贴进 PPT”。结果图是可编辑图形对象，而不是绑定 Excel 的原生 Chart。真实眼底照片及概率图为可移动/替换的位图，其像素内容不做矢量化。资产路径记录在 `figures/09_manifests/asset_manifest.json`。

## 检查范围与未解决问题

已检查六页 PPT 导出图及十九页 Word 导出版面；PPT 包结构、重新导入及布局检查通过。验证使用 LibreOffice 和文件级检查，没有声称已在微软 PowerPoint 内运行验证。Word 中六张图来自正式 PPT 导出；图形编辑请使用 PPT 源文件。

未重训模型、未新增临床结果、未完成高分辨率跨相机确认，也没有确立临床误差容限。本次不是对全部参考文献真实性或投稿合规性的独立完整审计。原有论文、实验及历史版本均保留；以下划线开头的目录为构建/预览中间文件，不应作为论文投稿附件。

重建与验证脚本位于上一级：`prepare_correction_v2.py`、`build_corrected_figures.mjs`、`build_corrected_manuscript.py`、`validate_correction_v2.py`。它们依赖本项目原始路径、既有冻结结果及本机运行环境。
