"""Joint report of fixed-model external evidence; does not replace a main model."""
from pathlib import Path
import pandas as pd
from evaluate_residual_ablation_external import OUT, ARMS

NAMES={'raw':'原始 VascX','legacy_mass':'旧 mass','full':'新完整网络','no_rgb':'去 RGB 输入',
       'no_logit_input':'去静态 logits 输入','no_entropy_input':'去显式熵输入',
       'no_hessian_input':'去 Hessian 输入','one_step':'单步网络','plain_conv':'普通卷积'}
ORDER=['raw','legacy_mass',*ARMS]

def main():
    assert (OUT/'audit.json').exists()
    f=pd.read_csv(OUT/'fundus_group_estimates.csv')
    fc=pd.read_csv(OUT/'fundus_group_contrasts.csv')
    r=pd.read_csv(OUT/'repeat_group_estimates.csv')
    c=pd.read_csv(OUT/'repeat_group_contrasts.csv')
    def sv(arm,metric):return float(f.query('arm==@arm and metric==@metric').iloc[0].estimate)
    def rv(arm,feature,metric,split='test'):return float(r.query('arm==@arm and feature==@feature and metric==@metric and split==@split').iloc[0].estimate)
    def ci(row):return f'{row.delta:+.6f} [{row.ci_low:+.6f}, {row.ci_high:+.6f}]'
    lines=['# 21 个固定权重：外部结构与重复性联合评估','',
           '本轮无训练、无调参、无最佳模型筛选。21 个权重分别评估，不做概率集成；另含原始 VascX 和旧 mass 冻结参照。', '',
           '- Fundus：100 张已分析过的结构参考图像，共 2,300 条图像—模型记录，其中旧 mass 控制复用。',
           '- mBRSET：1,960 张已分析过的重复拍摄图像，共 45,080 条图像—模型记录，其中旧 mass 控制复用。开发子集与历史 test 分开统计，患者不跨子集。',
           '- 新模型实际推理为 43,260 次图像—权重组合。各组表中为 3 个独立模型指标的均值，不是一个集成模型的成绩。',
           '- 95% 区间使用 5,000 次患者/图像与种子配对重采样，未作多重比较调整；全部结果是探索性证据。','',
           '## 1. 结构恢复与误检','',
           '| 方案 | 血管 Dice | Precision | 细中心线召回 |', '|---|---:|---:|---:|']
    for a in ORDER:lines.append(f'| {NAMES[a]} | {sv(a,"vessel_dice"):.6f} | {sv(a,"vessel_precision"):.6f} | {sv(a,"thin_centerline_recall"):.6f} |')
    lines+=['','新完整网络相对旧 mass 的差（95% 探索性区间）：','']
    for metric in ['vessel_dice','vessel_precision','thin_centerline_recall']:
        row=fc.query('arm=="full" and reference=="legacy_mass" and metric==@metric').iloc[0]
        lines.append(f'- {metric}：{ci(row)}。')
    lines+=['','## 2. A/V 概率质量比：历史 test 重复性','',
            '不是临床血管管径比。MAD 指对本眼均值的绝对偏差，SEM 为池化眼内标准差；LoA 使用前两张清单顺序视图，正负方向不代表时间效应。新模型行的 LoA 是三个模型界限的均值，不是单个部署模型的 LoA；逐模型界限见 CSV。','',
            '| 方案 | ICC | MAD | SEM | LoA 下界 | LoA 上界 |', '|---|---:|---:|---:|---:|---:|']
    for a in ORDER:
        vals=[rv(a,'av_ratio',m) for m in ['icc','within_eye_mad','within_eye_sd','loa_low','loa_high']]
        lines.append('| '+NAMES[a]+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines+=['','## 3. 新完整网络相对 raw：全部表型','',
            '正 ICC 差表示数值上提高，正 MAD 差表示绝对重复偏差增加。眼均值 SD 比只描述离散程度，不证明生物差异保留。','',
            '| 表型 | ICC 差 [95% CI] | MAD 差 [95% CI] | 眼均值 SD / raw |', '|---|---|---|---:|']
    features=r.feature.drop_duplicates().tolist()
    for feat in features:
        a=c.query('split=="test" and arm=="full" and reference=="raw" and feature==@feat and metric=="icc"').iloc[0]
        b=c.query('split=="test" and arm=="full" and reference=="raw" and feature==@feat and metric=="within_eye_mad"').iloc[0]
        ratio=rv('full',feat,'eye_mean_sd')/rv('raw',feat,'eye_mean_sd')
        lines.append(f'| {feat} | {ci(a)} | {ci(b)} | {ratio:.3f} |')
    lines+=['','## 4. 开发子集是否同方向','',
            '| 对照 | 子集 | A/V ICC 差 [95% CI] | A/V MAD 差 [95% CI] |', '|---|---|---|---|']
    for ref in ['raw','legacy_mass']:
        for split in ['development','test']:
            a=c.query('split==@split and arm=="full" and reference==@ref and feature=="av_ratio" and metric=="icc"').iloc[0]
            b=c.query('split==@split and arm=="full" and reference==@ref and feature=="av_ratio" and metric=="within_eye_mad"').iloc[0]
            lines.append(f'| 新完整网络 − {NAMES[ref]} | {split} | {ci(a)} | {ci(b)} |')
    lines+=['','## 5. 结论边界','',
            '- Dice、Precision、绝对结构误差、ICC 与 MAD 必须分别解释，不以某一项最好就宣布整体成功。',
            '- 去显式输入不代表去掉相关信息的全部路径；普通卷积只近参数量匹配，计算量不同。',
            '- 新权重来自 RAV 默认损失短程重训练，不是旧 mass 校准训练配方的精确复现。',
            '- Fundus 与 mBRSET 是不同队列，不能据两组结果证明同一患者的准确性与重复性同时提升。',
            '- mBRSET 使用 224 像素公开镜像，上采样不能补回解剖细节；没有新增独立高分辨率跨相机确认。',
            '- 图像重复筛查不等于患者独立性证明，也不构成对 VascX 全部预训练数据的重叠审计。',
            '- 所有新权重与结果保留；不根据这些已看过的外部数据选一个“最佳”模型替换主模型，不把探索性区间当临床准入证明。','',
            '完整单种子结果、组均值、所有对照区间、输入/权重哈希与独立点估计审计均在本目录。']
    (OUT/'联合评估报告.md').write_text('\n'.join(lines)+'\n')
    print(OUT/'联合评估报告.md')

if __name__=='__main__':main()
