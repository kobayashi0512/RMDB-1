"""Report every fixed arm/seed; no winner selection or manuscript mutation."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from train_controlled_residual_ablation import ARMS, SEEDS, EPOCHS, Cases, OUT, ROOT, metrics, target_labels
from prepare_residual_ablation import sha
from backbone_runtime import atomic_json

METRICS = ['artery_dice', 'vein_dice', 'mean_av_dice', 'type_accuracy', 'av_ratio_log_error']
NBOOT = 5000

def prepare():
    p = OUT / 'analysis_protocol.json'
    protocol = {'all_endpoints': METRICS, 'bootstrap': NBOOT, 'seed': 20260917,
                'unit': 'Crossed seed-and-image bootstrap: jointly resample seeds and image IDs, paired across arms. Image independence is assumed, not verified patient independence.',
                'estimate': 'Equal-weight seed mean of equal-weight image means; include all three seeds',
                'interval': 'Unadjusted exploratory percentile95%; not confirmatory significance',
                'comparators': 'Each ablation minus full; each trained arm minus raw',
                'interpretation': 'Positive Dice/type accuracy favours candidate; negative log-ratio error favours candidate',
                'split_reporting': 'Validation20 and historical camera7 reported separately, not pooled',
                'missing': 'Fail if run, image or finite endpoint is missing; never silently drop',
                'stopping': 'No arm or epoch selection based on these results',
                'source_sha256': sha(Path(__file__))}
    if p.exists(): assert json.loads(p.read_text()) == protocol
    else: atomic_json(p, protocol)

def raw_frame(cases, ids):
    rows = []
    for key in ids:
        rgb, z, ann = cases.get(key)
        z = cv2.resize(z.astype(np.float32), (512, 512), interpolation=cv2.INTER_LINEAR)
        ann = np.asarray(Image.fromarray(ann).resize((512, 512), Image.Resampling.NEAREST))
        rows.append({'id': key, **metrics(z.argmax(-1), target_labels(ann))})
    return pd.DataFrame(rows)

def report():
    prepare()
    assert (OUT / 'training_complete.json').exists()
    protocol = json.loads((OUT / 'training_protocol.json').read_text())
    for name, h in protocol['source_sha256'].items(): assert sha(ROOT / 'src' / name) == h
    assert sha(OUT / 'split.json') == protocol['split_sha256']
    assert sha(OUT / 'input_manifest.csv') == protocol['manifest_sha256']
    split = json.loads((OUT / 'split.json').read_text())
    runs = []
    for arm in ARMS:
        for seed in SEEDS:
            folder = OUT / 'runs' / arm / str(seed)
            complete = json.loads((folder / 'complete.json').read_text())
            assert complete['checkpoint_sha256'] == sha(folder / 'last.pt')
            assert complete['protocol_sha256'] == sha(OUT / 'training_protocol.json')
            history = json.loads((folder / 'history.json').read_text())
            assert [r['epoch'] for r in history] == [1, 2, 3]
            assert all(r['optimizer_steps'] == 90 for r in history)
            for subset, file in [('validation', 'validation_epoch_3.csv'), ('historical_camera_holdout', 'historical_camera_holdout.csv')]:
                frame = pd.read_csv(folder / file, dtype={'id': str})
                assert set(frame.id) == set(split[subset]) and frame.id.is_unique
                assert np.isfinite(frame[METRICS].to_numpy()).all()
                frame['arm'], frame['seed'], frame['split'] = arm, seed, subset
                runs.append(frame)
    cases = Cases()
    for subset in ['validation', 'historical_camera_holdout']:
        raw = raw_frame(cases, split[subset]); raw['arm'] = 'raw'; raw['split'] = subset
        for seed in SEEDS:
            part = raw.copy(); part['seed'] = seed; runs.append(part)
    data = pd.concat(runs, ignore_index=True)
    data.to_csv(OUT / 'all_final_per_image.csv', index=False)
    summary, contrasts = [], []
    rng = np.random.default_rng(20260917)
    for subset in ['validation', 'historical_camera_holdout']:
        ids = sorted(split[subset]); n = len(ids)
        bi = rng.integers(0, n, (NBOOT, n)); bs = rng.integers(0, len(SEEDS), (NBOOT, len(SEEDS)))
        for metric in METRICS:
            arrays = {}
            for arm in [*ARMS, 'raw']:
                arr = data.query('split==@subset and arm==@arm').pivot(index='seed', columns='id', values=metric).loc[SEEDS, ids].to_numpy()
                assert arr.shape == (3, n)
                arrays[arm] = arr
                boots = arr[bs[:, :, None], bi[:, None, :]].mean((1, 2))
                lo, hi = np.quantile(boots, [.025, .975])
                summary.append({'split': subset, 'arm': arm, 'metric': metric, 'mean': arr.mean(),
                                'seed_mean_sd': arr.mean(1).std(ddof=1), 'ci_low': lo, 'ci_high': hi, 'images': n, 'seeds': 3})
            pairs = [(a, 'full') for a in ARMS if a != 'full'] + [(a, 'raw') for a in ARMS]
            for arm, ref in pairs:
                delta = arrays[arm] - arrays[ref]
                boots = delta[bs[:, :, None], bi[:, None, :]].mean((1, 2))
                lo, hi = np.quantile(boots, [.025, .975])
                contrasts.append({'split': subset, 'arm': arm, 'reference': ref, 'metric': metric,
                                  'delta': delta.mean(), 'ci_low': lo, 'ci_high': hi,
                                  'positive_seed_count': int((delta.mean(1) > 0).sum())})
    summary = pd.DataFrame(summary); contrasts = pd.DataFrame(contrasts)
    summary.to_csv(OUT / 'final_summary.csv', index=False)
    contrasts.to_csv(OUT / 'paired_contrasts.csv', index=False)
    lines = ['# 残差校正受控重训练：阶段结果', '',
             '本轮为新训练协议下的探索性设计检验，不是旧 mass 校准模型的精确复现。7 个方案 × 3 个种子，每次固定 3 个 epoch、270 次优化更新；不选最佳 epoch。', '',
             '## 必须先知道的边界', '',
             '- 已逐张重建 206 张 RAV 缓存图像的裁剪；87 张标注需同步变换。新训练使用对齐标注，旧文件未覆盖。',
             '- 固定 179 张训练、20 张验证、7 张历史模拟相机留出。元数据缺患者 ID，不能证明患者独立；相机留出也不是新独立队列。',
             '- 旧权重缺完整训练配置。新协议显式使用原训练代码的默认损失，A/V mass 和 vessel mass 两项附加损失权重均为 0；不能把本轮称为旧 mass 校准配方的直接消融。',
             '- 去掉特征只影响网络输入；损失函数仍使用完整特征。这是输入贡献检验，不是移除该信息的所有路径。',
             '- 普通卷积对照有 263,332 个参数，完整网络 269,164 个；相差约 2.2%。训练更新数匹配，不声称 FLOPs 匹配。单步网络也具有更少计算量。',
             '- 3 个 epoch 是固定短程实验，不足以证明各方案充分收敛。所有区间均为未调整的探索性区间。', '',
             '## 最终 epoch：平均 A/V Dice（种子间标准差）', '',
             '| 方案 | 验证 20 张 | 历史相机 7 张 |', '|---|---:|---:|']
    for arm in [*ARMS, 'raw']:
        cells = []
        for subset in ['validation', 'historical_camera_holdout']:
            r = summary.query('arm==@arm and split==@subset and metric=="mean_av_dice"').iloc[0]
            cells.append(f'{r["mean"]:.4f} ({r.seed_mean_sd:.4f})')
        lines.append('| ' + arm + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 各消融相对完整网络：平均 A/V Dice 配对差', '',
              '| 方案 | 验证差 [95% CI] | 历史相机差 [95% CI] |', '|---|---:|---:|']
    for arm in ARMS[1:]:
        cells = []
        for subset in ['validation', 'historical_camera_holdout']:
            r = contrasts.query('arm==@arm and reference=="full" and split==@subset and metric=="mean_av_dice"').iloc[0]
            cells.append(f'{r.delta:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]')
        lines.append('| ' + arm + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '全部 5 个指标、全部种子和配对差见 CSV。负差表示该消融的 Dice 低于完整网络，不自动等于特征普遍必要。', '',
              '## 下一步边界', '',
              '这批新权重还没有经过 Fundus 结构和 mBRSET 重复性统一评估。因此不替换论文主模型，不把 RAV 分割提升直接写成测量可靠性提升。后续应固定本轮全部模型，在既有外部数据上作探索性评估，并优先恢复旧训练配置；不依据这 7 张图继续调参。']
    (OUT / '阶段结果说明.md').write_text('\n'.join(lines) + '\n')
    atomic_json(OUT / 'final_audit.json', {'runs_verified': 21, 'updates_each_run': 270, 'source_hashes_verified': True,
                                         'final_per_image_rows_including_repeated_raw': len(data),
                                         'all_five_endpoints_finite': True, 'all_arms_and_seeds_reported': True,
                                         'new_manuscript_claims_applied': False})
    print('\n'.join(lines))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'report']); args = p.parse_args()
    if args.stage == 'prepare': prepare()
    else: report()
