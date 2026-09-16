#!/usr/bin/env python3
"""Build the trial report only from complete, traceable outputs."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/fixed_preprocessing_trial'


def main():
    d=json.loads((OUT/'summary.json').read_text())
    g=json.loads((ROOT/'artifacts/fundus_coordinate_alignment_audit/summary.json').read_text())
    assert d['complete'] and d['arm_evaluations_completed']==400 and g['n']==100
    frame=pd.read_csv(OUT/'per_image_metrics.csv')
    assert len(frame)==400 and not frame[['id','arm']].duplicated().any()
    assert frame.groupby('arm').size().eq(100).all()
    assert frame.select_dtypes('number').notna().all().all()
    names={'default':'默认 VascX 预处理','illumination':'额外照明校正',
           'clahe':'温和局部对比度增强','denoise_sharpen':'轻度去噪与锐化'}
    m=g['means'];delta=g['mass_paired_delta']
    lines=['# 图像预处理试验与坐标审计', '',
        '本次试验覆盖已查看过的 Fundus-AVSeg 100 张图像：固定四个分支，共 400 次评估。原始图像、标注及旧结果均保留。没有使用 CIMT 或临床结局。', '',
        '## 先纠正评估坐标', '',
        '历史评估直接缩放原图标注，却没有将 VascX 预处理保存的裁剪变换应用到标注。此次重放图像预处理时，100 张缓存 RGB 图像均逐像素完全一致。变换来自保存的图像边界，不通过标注或预测拟合。历史平均 A/V Dice 逐图复现误差小于 1e-7。', '',
        '|同一份缓存 VascX 预测|Mean A/V Dice|', '|---|---:|',
        f"|历史评估复现|{m['historical_mean_av_dice']:.6f}|",
        f"|正确坐标，768 网格|{m['aligned768_mean_av_dice']:.6f}|",
        f"|正确坐标，保持历史 logits 插值路径|{m['aligned768_old_interpolation_mean_av_dice']:.6f}|",
        f"|正确坐标，1024 网格|{m['aligned1024_mean_av_dice']:.6f}|", '',
        '上述变化来自评估修正，不能作为算法性能提升。本地近期的 0.5619 重评估及上次据此进行的 oracle 误差归因已发现坐标问题，不能直接作为论文最终结论。稿件中的 0.5545→0.5635 属于另一份历史报告，尚需单独追溯，不能直接用本次数值覆盖。不同评估网格的结果也不能混用。', '',
        f"同样使用正确坐标与 768 网格，缓存预测经过冻结 Mass corrector 后为 {m['aligned768_mass_mean_av_dice']:.6f}；相对基线差值 {delta['mean']:+.6f}，图像级配对 bootstrap 95% CI [{delta['ci95'][0]:+.6f}, {delta['ci95'][1]:+.6f}]。这是既有权重的重评估，没有训练。", '',
        '## 四个固定预处理分支', '',
        '所有分支采用同一组冻结 VascX 权重、1024 输入、512 滑窗、0.5 重叠，统一关闭翻转测试增强。额外处理作用于固定裁剪后的 RGB，再重建 VascX 所需的对比度增强输入。标注使用相同裁剪后在 768 网格评分。', '',
        '照明校正采用平滑亮度场和受限增益；局部对比度采用 LAB 亮度通道 CLAHE；去噪锐化采用温和双边滤波与非锐化掩模。三个分支均与原图各混合 50%，参数见 protocol.json。', '',
        '这里的新鲜默认对照与历史缓存输出的推理设置可能不同，所有额外处理只与本次新鲜对照配对比较。', '',
        '|预处理|Mean A/V Dice|相对本次默认对照 Δ [95% CI]|Vessel Dice|血管精确率|血管召回率|细血管中心线召回率|Brier ↓|',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for arm in ['default','illumination','clahe','denoise_sharpen']:
        armdata=d['overall'][arm];a=armdata['means']
        change='—'
        if arm!='default':
            c=armdata['paired_against_default']['mean_av_dice'];change=f"{c['delta']:+.6f} [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}]"
        lines.append(f"|{names[arm]}|{a['mean_av_dice']:.6f}|{change}|{a['vessel_dice']:.6f}|{a['vessel_precision']:.6f}|{a['vessel_recall']:.6f}|{a['thin_centerline_recall']:.6f}|{a['brier']:.6f}|")
    contrast=d['overall']['clahe']['paired_against_default']
    lines+=['',
        f"本轮 CLAHE 分支的 A/V Dice 在 {contrast['mean_av_dice']['improved']}/100 张图像上提高。血管召回率平均变化 {contrast['vessel_recall']['delta']*100:+.2f} 个百分点，细血管中心线召回率变化 {contrast['thin_centerline_recall']['delta']*100:+.2f} 个百分点；同时血管精确率变化 {contrast['vessel_precision']['delta']*100:+.2f} 个百分点。这表明检出范围扩大并伴随误检代价，不能描述为无代价地恢复真实血管。", '',
        '## 原有图像质量分层', '', '|分层|分支|n|Mean A/V Dice|配对 Δ [95% CI]|', '|---|---|---:|---:|---|']
    for level in ['High-quality','Low-quality']:
        for arm in ['default','illumination','clahe','denoise_sharpen']:
            a=d['strata'][level][arm];change='—'
            if arm!='default':
                c=a['paired_against_default']['mean_av_dice'];change=f"{c['delta']:+.6f} [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}]"
            lines.append(f"|{level}|{names[arm]}|{a['n']}|{a['means']['mean_av_dice']:.6f}|{change}|")
    lines+=['', '## 解释范围', '',
        '本试验回答固定预处理是否让既有模型读出更多有效血管信息。输出更清晰不等于恢复了真实缺失结构；这一结论必须由分割和假阳性指标支撑。完全缺失的细节无法由这类确定性处理凭空恢复。', '',
        '该数据集已经被用于分析失败原因，本轮属于探索性补充试验；不能将最好的分支宣称为独立确认的最优方案。所有分支一并报告，95% CI 是未经多重比较调整的图像级 bootstrap 区间，若同一患者存在多图，区间可能偏窄。细血管定义为 768 网格中距离变换半径不超过 2 像素的标注骨架点，不能直接解释为生理血管直径。', '',
        '代码：src/test_fixed_preprocessing.py、src/audit_fundus_coordinate_alignment.py、src/report_fixed_preprocessing.py。逐图结果、固定协议及 JSON 汇总与本报告同目录；坐标审计在相邻 fundus_coordinate_alignment_audit 目录。', '',
        '## 真实图像示例', '',
        '固定使用文件名排序的第一张 001_G，未按提升幅度挑选。第一张图展示实际输入与相同位置的裁剪，第二张图展示预测和坐标对齐后的原始标注。', '',
        f'![实际图像预处理对照]({OUT}/input_comparison.png)', '',
        f'![预测与原始标注对照]({OUT}/prediction_comparison.png)']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(OUT/'REPORT_ZH.md')


if __name__=='__main__':main()
