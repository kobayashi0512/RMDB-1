"""Validate every stored output and produce a transparent result handoff."""
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from run_fundus_fair_four_arm import OUT, ROOT, ARMS, METRICS, sha, paired

def main():
    s=json.loads((OUT/'summary.json').read_text())
    protocol=json.loads((OUT/'protocol.json').read_text())
    photometric=json.loads((OUT/'photometric_input_audit.json').read_text())
    assert photometric['n']==100 and photometric['clahe_rgb_exact'] and photometric['clahe_contrast_enhanced_exact']
    legacy=json.loads((OUT/'legacy_reproduction.json').read_text())
    legacy_frame=pd.read_csv(OUT/'legacy_reproduction.csv')
    legacy_delta={}
    for mode,g in legacy_frame.groupby('geometry'):
        legacy_delta[mode]={}
        for metric in METRICS:
            p=g.pivot(index='id',columns='arm',values=metric)
            legacy_delta[mode][metric]=paired(p['mass']-p['vascx'],metric)
    (OUT/'legacy_paired_deltas.json').write_text(json.dumps(legacy_delta,indent=2))
    assert s['complete'] and s['images']==100 and s['arm_evaluations']==400
    frame=pd.read_csv(OUT/'per_image_metrics.csv')
    assert not frame.duplicated(['id','arm']).any()
    assert np.isfinite(frame[METRICS].to_numpy()).all()
    interactions={}
    for metric in METRICS:
        p=frame.pivot(index='id',columns='arm',values=metric)
        interactions[metric]=paired((p.clahe_mass-p.clahe)-(p.mass-p.vascx),metric)
    (OUT/'factorial_interaction.json').write_text(json.dumps(interactions,indent=2))
    manifest={};max_probability_sum_error=0.
    for ident in protocol['ids']:
        for arm in ARMS:
            path=OUT/'probabilities'/f'{ident}__{arm}.npy';p=np.load(path)
            assert p.shape==(4,768,768) and np.isfinite(p).all()
            assert p.min()>=0 and p.max()<=1
            err=float(np.max(np.abs(p.sum(0)-1)));assert err<2e-6
            max_probability_sum_error=max(max_probability_sum_error,err)
            manifest[str(path.relative_to(OUT))]=sha(path)
        for arm in ('default','clahe'):
            path=OUT/'logits'/f'{ident}__{arm}.npy';p=np.load(path)
            assert p.shape==(768,768,4) and np.isfinite(p).all()
            manifest[str(path.relative_to(OUT))]=sha(path)
    audit={'images':100,'four_arm_probabilities':400,'fresh_input_logits':200,'all_finite':True,'max_probability_sum_error':max_probability_sum_error,'all_geometry_verified':True,'all_metrics_finite':True,'all_arms_same_ids':True}
    audit['photometric_inputs']=photometric
    release=(OUT/'MODEL_CONTENTS_SHA256.txt').read_text()
    for rel in ('baseline/artifacts/udct_av_mass_calibrated/best.pt','third_party/vascx_weights/av_july24.pt'):
        expected=next(line.split()[0] for line in release.splitlines() if line.endswith('  '+rel))
        assert sha(ROOT/'models'/rel)==expected
    audit['weights_match_GitHub_v1_0_0_release_manifest']=True
    (OUT/'output_integrity.json').write_text(json.dumps(audit,indent=2))
    (OUT/'output_sha256.json').write_text(json.dumps(manifest,indent=2))
    # Read-only archive recovery. Save verbatim and label it historical, not current evidence.
    remote=subprocess.run(['gh','api','repos/kobayashi0512/RMDB/contents/protocols/Fundus_AVSeg_外部验证结果与反思.md','-H','Accept: application/vnd.github.raw+json'],capture_output=True,text=True)
    if remote.returncode==0:(OUT/'ARCHIVED_20260729_original_report.md').write_text(remote.stdout)
    remote_code=subprocess.run(['gh','api','repos/kobayashi0512/RMDB/contents/src/evaluate_fundus_avseg_external.py','-H','Accept: application/vnd.github.raw+json'],capture_output=True,text=True)
    if remote_code.returncode==0:
        (OUT/'ARCHIVED_original_evaluator.py.txt').write_text(remote_code.stdout)
        audit['original_evaluator_matches_GitHub']=remote_code.stdout==(ROOT/'src/evaluate_fundus_avseg_external.py').read_text()
    audit['GitHub_archive_commit']='fd0732249eacb373d0b6168b891fa310966e7a70'
    (OUT/'output_integrity.json').write_text(json.dumps(audit,indent=2))
    labels={'vascx':'VascX','clahe':'VascX＋固定 CLAHE','mass':'VascX＋冻结 Mass UDCT','clahe_mass':'VascX＋固定 CLAHE＋冻结 Mass UDCT'}
    lines=['# Fundus-AVSeg：原始分数追溯与公平四组对照','',
           '状态：100 张图像、四组共 400 个评估全部完成。数据集已被反复查看，本次为探索性审计与对照，不是新的独立确认。','',
           '## 1. 先纠正比较的基础','',
           '旧流程把经过视野裁剪的预测图与仅缩放、未应用同一裁剪的原始标注比较。新流程逐图重放保存在 bounds_latest_batch.csv 中的图像裁剪，100 张 RGB 均与缓存逐像素完全一致；同一变换以最近邻应用于标注。没有用标注拟合配准。','',
           '另一处问题来自近期脚本复用 RAV 的颜色解码。Fundus-AVSeg 的绿色是动静脉交叉点，不应忽略；白色是无法判断动静脉类型的血管。现在绿色对动脉和静脉均计入，白色仅从身份评价剔除，仍计入血管检出。该解释与原始 Fundus 评估器一致。','',
           '标注依据：[数据集原论文](https://www.nature.com/articles/s41597-025-05381-2)。','',
           '|512 像素追溯，使用同一缓存预测和权重|VascX mean AV Dice|Mass UDCT mean AV Dice|',
           '|---|---:|---:|']
    for g,title in [('legacy_unregistered','旧错误坐标，仅用于复现'),('aligned','仅修正坐标，同一 512 评分尺度')]:
        lines.append(f"|{title}|{legacy['means'][g+'/vascx']['mean_av_dice']:.6f}|{legacy['means'][g+'/mass']['mean_av_dice']:.6f}|")
    d512=legacy_delta['aligned']['mean_av_dice']
    lines+=['',f"修正坐标后的 512 缓存评估：Mass 相对 VascX 的配对差为 {d512['delta']:+.6f}，95% CI [{d512['ci95'][0]:+.6f}, {d512['ci95'][1]:+.6f}]。这一分辨率敏感性结果与下方 768 四组实验分别呈现，不混作同一设置。"]
    lines+=['',f"复现论文四位小数：{legacy['reproduces_paper_to_4_decimals']}。原始记录已从 GitHub 只读恢复到 ARCHIVED_20260729_original_report.md（若网络恢复成功）。",
            '**不能把 0.55 与修正后的 0.8 左右作算法前后对比。坐标修复、标注口径修复与评分分辨率变化都不是模型增益。旧报告根据该错误评分作出的失败域、增益及通过判定也必须重新审查。**','',
            '## 2. 四组公平结果','',
            '本表统一为 1024 输入，五成员冻结 VascX，512 分块、0.5 重叠、不启用 TTA；768 评分网格，统一正确坐标与标注解释。CLAHE 固定为 LAB 亮度通道 clip=1.5、8×8 网格、与原图各占 0.5；没有搜索参数。Mass UDCT 使用已冻结原权重、原 11 通道特征与两步修正，并未重新训练。此处 RMDB 分支仅指 Mass UDCT，不等同于完整 BiRAP 患者级流程。','',
            '|方法|mean AV Dice ↑|血管 Dice ↑|血管精确率 ↑|血管召回率 ↑|细血管中心线召回率 ↑|AV 比例对数误差 ↓|四类平均 Brier ↓|',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    columns=['mean_av_dice','vessel_dice','vessel_precision','vessel_recall','thin_centerline_recall','av_ratio_log_error','brier']
    for arm in ARMS:lines.append('|'+labels[arm]+'|'+'|'.join(f"{s['overall'][arm][m]:.6f}" for m in columns)+'|')
    lines+=['','## 3. 配对增量与误检代价','','差值均为前者减后者；Brier 和比例误差的负值有利。置信区间为图像配对 bootstrap 5000 次百分位区间，未作多重比较校正，也未计入此前挑选 CLAHE 候选的选择不确定性；不能据此作确认性显著结论。数据集论文说明 100 张图来自 100 名患者；这里不进行临床推断。','',
            '|对照|指标|平均差|95% CI|改善图像数|','|---|---|---:|---|---:|']
    for contrast,d in s['contrasts'].items():
        for m in columns:
            x=d[m];lines.append(f"|{contrast}|{m}|{x['delta']:+.6f}|[{x['ci95'][0]:+.6f}, {x['ci95'][1]:+.6f}]|{x['improved']}/100|")
    x=s['contrasts']['clahe_mass_minus_clahe']['mean_av_dice']
    if x['ci95'][0]>0:verdict='冻结 Mass UDCT 在固定 CLAHE 之外仍有可检测的 Dice 增量；幅度和误差代价应同时报告，不能仅据统计显著性称为重大提升。'
    elif x['ci95'][1]<0:verdict='冻结 Mass UDCT 加在固定 CLAHE 之后使 Dice 下降。本轮不支持组合路线，不应继续在本数据上调参寻找正结果。'
    else:verdict='冻结 Mass UDCT 在固定 CLAHE 之外的 Dice 增量尚不确定；本轮不足以证明组合优于简单预处理。'
    lines+=['','## 4. 本轮结论','',verdict,'',
            '预处理的 Dice 改善不能说明恢复了原本不存在的真实血管，也不能只凭更清晰的外观判断有效。必须同时看召回与精确率，并与原始标注比较。','',
            '## 5. 预定义分层','',
            '|分层|图像数|VascX|CLAHE|Mass|CLAHE＋Mass|','|---|---:|---:|---:|---:|---:|']
    for col,groups in s['strata'].items():
        for name,g in groups.items():lines.append(f"|{col}: {name}|{g['n']}|"+'|'.join(f"{g['arms'][a]['mean_av_dice']:.6f}" for a in ARMS)+'|')
    ix=interactions['mean_av_dice']
    lines+=['',f"探索性交互差 [(CLAHE＋Mass)−CLAHE]−[Mass−VascX]：ΔDice={ix['delta']:+.6f}，95% CI [{ix['ci95'][0]:+.6f}, {ix['ci95'][1]:+.6f}]。它描述组合效应是否可加，不单独证明任何生物学或算法机制。",
            '','## 6. 指标解释与限制','',
            '- mean AV Dice 为逐图动脉、静脉 Dice 均值再跨图平均，不是像素准确率。',
            '- 细血管定义为标注血管骨架上距离变换半径≤2 个像素的点，在 768 尺度固定；这是敏感性端点，不是临床统一的细血管定义。',
            '- AV 比例误差为硬预测与标注的动脉/静脉像素数比值之对数差绝对值，+1 平滑；不等同于临床血管管径 AVR，也不等同于软概率质量比。',
            '- Brier 对四类取平均并跨有效像素平均，含大量背景；不是某些文献使用的按类别求和 Brier。',
            '- CLAHE 来自前一轮探索后的候选，组合是未重新训练的冻结修正器。不证明通用域泛化，不证明临床收益。',
            '- 未读取 CIMT 固定测试/验证或任何临床结局。本轮不新增模型搜索；后续确认需要未使用样本。',
            '- 原始论文旧分数和近期忽略绿色交叉点的分数分别保留为历史记录，不得混入新结果表。','',
            '## 7. 可复现文件','',
            '- protocol.json：实验口径与全部图像 ID。',
            '- per_image_metrics.csv、summary.json：逐图和汇总结果。',
            '- legacy_reproduction.csv/json：旧流程复现及仅修正坐标的比较。',
            '- logits/、probabilities/、predictions/：200 份输入 logits、400 份概率和真实推理分割。',
            '- sha256_manifest.json、output_sha256.json、output_integrity.json：来源校验及全部有限输出核查。',
            '- 源码：src/run_fundus_fair_four_arm.py；回归测试：src/test_fundus_label_convention.py。','']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines))
    print(json.dumps(audit,indent=2));print(verdict)

if __name__=='__main__':main()
