#!/usr/bin/env python3
"""Render the completed development screen from immutable JSON result files."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/backbone_candidates_recovery_20260907/final_audit'
NAMES = {
    'baseline': 'ResNet34 U-Net 基线',
    'highres': '紧凑 U-Net',
    'global_local': '紧凑 U-Net + 全局通道门控',
    'dualstream': '紧凑 U-Net + 高频绿色通道',
    'direction': '紧凑 U-Net + Hessian 通道',
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]])

def main():
    data = json.loads((OUT / 'comparison.json').read_text())
    audit = json.loads((OUT / 'training_integrity.json').read_text())
    assert data['status'] == 'complete' and data['n_jobs'] == 15
    rows, aggregates = data['per_run'], data['aggregates']
    byname = {a['candidate']: a for a in aggregates}
    supplemental = [json.loads(p.read_text()) for p in sorted(OUT.glob('cldice_seed_*.json'))]
    assert len(supplemental) == 3 and all(s['status'] == 'complete' for s in supplemental)
    replay_rows = [r for s in supplemental for rs in s['rows'].values() for r in rs]
    assert len(replay_rows) == 2250
    max_delta = max(r['replay_max_metric_delta'] for r in replay_rows)
    for supplement in supplemental:
        assert supplement['script_sha256'] == digest(ROOT / 'src/finalize_backbone_recovery.py')
        assert not supplement['thresholds_changed'] and supplement['post_training_descriptive_only']
        for name, records in supplement['rows'].items():
            assert len(records) == len({r['id'] for r in records}) == 150
            assert all(math.isfinite(r['cldice']) and 0 <= r['cldice'] <= 1 for r in records)
            baseline = supplement['rows']['baseline']
            assert [(r['id'], r['target_vessel_pixels']) for r in records] == [(r['id'], r['target_vessel_pixels']) for r in baseline]
            checkpoint = OUT.parent / name / f"seed_{supplement['seed']}" / 'best.pt'
            assert digest(checkpoint) == supplement['best_checkpoint_sha256'][name]
    def metric(name, field):
        v = byname[name]['metrics'][field]
        return f"{v['mean']:.4f} ± {v['sd']:.4f}"
    def delta(name, field):
        return f"{byname[name]['metrics'][field]['mean_paired_delta']:+.5f}"
    baseline_params = byname['baseline']['occupancy_parameters']
    parts = [
        '# RMDB–BiRAP 主干候选实验：完成报告',
        '## 结论',
        '15 组训练（5 个候选 × 3 个开发划分种子）均完成 12 个 epoch，共 180 个完整 epoch。'
        '本轮没有得到可靠的精度提升：四个紧凑候选的平均 RAV 动静脉 Dice 均略低于基线，'
        '独立占据分支的 FIVES 血管 Dice 也均低于基线。参数量明显减少，但不能据此宣称同等精度、推理加速或达到某一投稿档次。',
        '这是开发筛选的负结果，不是四种结构已被公平、充分训练后证伪。预训练暴露不等、训练预算较短、'
        '冻结路径占比较高，使其不足以单独判定主干结构优劣。当前不自动追加训练，也不对外部标签选模型。',
        '## 1. 可比性与数据范围',
        '同一个种子的五个模型使用完全相同的训练和验证 ID；三个种子也改变开发划分，因此下列标准差是跨开发划分的描述性离散程度，'
        '不是独立外部测试的不确定性，也不能由三个均值推导统计显著性。',
        '- 每个种子：RAV 169 张训练、30 张验证；7 张排除设备图像未参与评估。\n'
        '- FIVES 官方训练集内部划分为 480 张训练、120 张验证；没有使用官方测试标签。\n'
        '- 输入训练输出为 512 像素，评估为 768 像素，训练随机裁剪为 384/512/640/768。\n'
        '- 12 epoch，batch size 4，学习率 5×10⁻⁵；检查点仍按原先固定的复合分数选择。\n'
        '- 本轮没有读取或评估 CIMT 固定测试集、Fundus-AVSeg 标签、mBRSET 结局或 RAV 排除设备。',
        'RAV 的分数来自四分类最终融合读出；FIVES 分数来自学习到的独立占据分支。两者不应混作同一个任务。'
        '本轮 RAV 约 0.79 的分数与旧稿 Fundus-AVSeg 约 0.55 的分数属于不同数据、不同评价条件，绝不能相减后声称模型提高约 0.24。',
        '## 2. 主要结果',
        '数值为三个种子的均值 ± 样本标准差。FIVES 主表统一使用非空标注样本：种子 20260906 为 119 张，另两个种子各 120 张，'
        '五个候选使用完全相同的样本子集。该补充仅描述已有检查点，不改变原模型选择。',
        table(['主干', 'RAV A/V Dice ↑', 'RAV vessel Dice ↑', 'FIVES vessel Dice ↑（共同非空标注）'],
              [[NAMES[n], metric(n, 'rav_mean_av_dice'), metric(n, 'rav_vessel_dice'), metric(n, 'fives_vessel_dice_nonempty')] for n in NAMES]),
        table(['主干', 'RAV clDice ↑', 'FIVES clDice ↑（共同非空标注）', 'RAV log A/V 比例误差 ↓'],
              [[NAMES[n], metric(n, 'rav_cldice'), metric(n, 'fives_cldice_nonempty'), metric(n, 'rav_av_ratio_log_error')] for n in NAMES]),
        '## 3. 相对基线的同种子差值',
        '以下为原始 0–1 指标尺度上的绝对差值；例如 0.001 等于 0.1 个百分点，不是 1%。',
        table(['候选', 'Δ RAV A/V Dice ↑', 'Δ FIVES Dice ↑', 'Δ FIVES clDice ↑', 'Δ A/V 比例误差 ↓'],
              [[NAMES[n], delta(n, 'rav_mean_av_dice'), delta(n, 'fives_vessel_dice_nonempty'),
                delta(n, 'fives_cldice_nonempty'), delta(n, 'rav_av_ratio_log_error')] for n in NAMES if n != 'baseline']),
        '逐个种子的差值、最小值和最大值见 comparison.json；不只展示有利种子。',
        '## 4. 容量、耗时和实际实现',
        table(['主干', '占据分支参数', '相对基线减少', '平均训练耗时（小时）'],
              [[NAMES[n], f"{byname[n]['occupancy_parameters']:,}", f"{(1-byname[n]['occupancy_parameters']/baseline_params)*100:.2f}%",
                metric(n, 'training_hours')] for n in NAMES]),
        '这里是占据分支参数量，不是包括冻结 VascX 在内的全系统参数量。训练耗时受两个并发任务、系统负载及暂停恢复影响，'
        '不是控制条件下的效率基准，更不是部署推理延迟。',
        '- baseline：已有 FIVES 权重初始化的 ResNet34 U-Net。\n'
        '- highres：从零初始化的三尺度紧凑 U-Net；名称不代表本轮实施了 1024 像素全图训练。\n'
        '- global_local：增加全局通道上下文门控；没有实施整图和局部 patch 间的交互网络。\n'
        '- dualstream：RGB 与重复的高通绿色通道在输入处拼接；不是两个独立编码器。\n'
        '- direction：RGB 与 Hessian 导出通道拼接；不是新定义的方向卷积。',
        '## 5. 为什么最终 A/V 分数变化很小',
        '代码中的占据融合为 O_fused = b × O_VascX + (1 − b) × O_learned；其后还有固定质量路由将联合读出与冻结的 mass 读出混合。'
        '四个紧凑候选的 RAV 平均 b 约为 0.94，意味着在这一中间融合公式里学习占据分支的平均系数仅约 0.06。'
        '最终 A/V 分数接近、但直接评估学习占据分支时差距明显，与这种保守融合机制一致。'
        '这是一项代码和已记录权重支持的机制解释，不是去除旁路后的因果消融结论；不能把 94% 理解为正确像素的比例。',
        table(['主干', 'RAV 平均旁路系数 b', 'RAV Brier ↓'],
              [[NAMES[n], metric(n, 'rav_bypass'), metric(n, 'rav_brier')] for n in NAMES]),
        '## 6. 收敛、历史暴露与完整性限制',
        f"四个紧凑候选的 12 组运行中，{sum(r['best_epoch'] >= 11 for r in rows if r['candidate'] != 'baseline')} 组最佳检查点在第 11 或 12 epoch；全局门控模型三个种子均在第 12 epoch 最佳。"
        '部分训练损失仍下降，因此不能称其已充分收敛。也不能据此保证增加 epoch 就会超过基线。',
        '基线使用历史 FIVES 权重，而紧凑候选随机初始化；历史 FIVES 预训练与当前开发验证划分的重叠尚未排除。'
        '共享 mass corrector 曾在 RAV 开发，故 RAV 也不是身份模块的独立测试。'
        '这些因素使本轮不能识别纯粹的架构效应。',
        'RAV 图像与标注坐标恢复对全部候选一致实施，206 张缓存 RGB 的原图变换已逐像素复现；'
        '204 张采用已存边界，2 张仅用原图检测器恢复。坐标纠正不是算法增益，旧的错位结果未混入本轮。',
        '旧计划中的细血管召回率/精确率成功门槛没有被本轮评估器实现，所以该门槛不能宣称通过。'
        'clDice 只补充中心线结构重叠，也不能代替细血管分层结果、临床终点或完整拓扑正确性的证明。',
        '## 7. 审计发现：一张全空标注导致原均值分母不同',
        'FIVES_448_G 的开发验证标注像素全部为 0。种子 20260906 的基线预测也为空，旧 Dice 函数返回 NaN，'
        'pandas 均值跳过该行，因此原基线均值用 119 张图；四个候选在该图有预测，记为 0，原均值用 120 张图。'
        '这不是缺失检查点或训练失败，而是评价空掩膜约定造成的分母不一致。不能在未核查图像语义前将全空标注直接认定为错误标注。',
        '处理方式：保留全部原始 CSV、summary 和 checkpoint；另报所有候选共同非空标注子集的 Dice/clDice。'
        '完整样本 clDice 也保留在 JSON，明确采用双方全空为 1、仅一方空为 0。该发现没有触发重新选检查点或调整阈值。',
        table(['主干', '原始 FIVES Dice（分母不一致，勿作主比较）', '共同非空标注 FIVES Dice'],
              [[NAMES[n], metric(n, 'fives_vessel_dice'), metric(n, 'fives_vessel_dice_nonempty')] for n in NAMES]),
        '## 8. 全部 15 组明细',
        table(['主干代码', '种子', '最佳 epoch', 'RAV A/V Dice', 'FIVES Dice（共同非空）', 'FIVES clDice（共同非空）', '训练小时'],
              [[r['candidate'], r['seed'], r['best_epoch'], f"{r['rav_mean_av_dice']:.6f}",
                f"{r['fives_vessel_dice_nonempty']:.6f}", f"{r['fives_cldice_nonempty']:.6f}",
                f"{r['training_hours']:.3f}"] for r in rows]),
        '## 9. 完整性与复现',
        f"训练代码及几何审计哈希一致；15 组均含完整 12 epoch 历史、模型、优化器、随机状态和分割清单。"
        f"已补算 2,250 个模型—图像组合（不是 2,250 个独立患者）。有限原始指标的复算最大绝对差为 {max_delta:.3g}。"
        '空/空 Dice 作为显式未定义值保留，不冒充有限分数。补算前后最佳检查点哈希一致。',
        '运行命令（项目根目录）：\n\n```sh\n.venv/bin/python src/finalize_backbone_recovery.py\n.venv/bin/python src/report_backbone_recovery.py\n```',
        '补算使用二维二值骨架的重叠精确率与召回率的调和平均，即 hard clDice；与训练中 soft clDice 损失分开。'
        '实现依据：[clDice 作者代码](https://github.com/jocpae/clDice)。补充只在已选检查点和原开发验证 ID 上运行，不参与选模。',
        '- [机器可读比较与逐种子结果](comparison.json)\n'
        '- [训练完整性审计及警告](training_integrity.json)\n'
        '- [种子 20260906 逐图补算](cldice_seed_20260906.json)\n'
        '- [种子 20260907 逐图补算](cldice_seed_20260907.json)\n'
        '- [种子 20260908 逐图补算](cldice_seed_20260908.json)',
        '## 10. 对论文的影响',
        '本轮可作为诚实报告的开发性补充或探索失败记录，不能作为“新主干显著提升”的核心证据。'
        '现有论文已锁定的外部评价不能被这些内部开发数字替代。若未来重做公平主干试验，需要预先统一初始化与数据暴露、'
        '制定不依赖本轮有利结果的收敛预算、修订空标注口径并冻结评价；那应是新的独立协议，而不是把本轮最好的数字移到主文。',
    ]
    report = '\n\n'.join(parts) + '\n'
    (OUT / 'BACKBONE_RESULTS_FINAL_20260908.md').write_text(report)
    manifest = {'status': 'complete_with_disclosed_metric_warning', 'n_jobs': 15, 'completed_epochs': 180,
                'replayed_model_image_pairs': len(replay_rows), 'max_replay_metric_delta': max_delta,
                'checkpoint_hashes_unchanged': True, 'same_nonempty_subsets_verified': True,
                'artifacts_sha256': {p.name: digest(p) for p in sorted(OUT.iterdir()) if p.suffix in ('.json', '.md') and p.name != 'completion_manifest.json'},
                'report_source_sha256': digest(Path(__file__))}
    (OUT / 'completion_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(OUT / 'BACKBONE_RESULTS_FINAL_20260908.md')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
